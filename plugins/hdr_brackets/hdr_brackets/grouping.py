"""Group source images into exposure brackets (one bracket = one camera position).

Why this rule: a camera's auto-bracketing fires its frames within a second or two and never repeats an exposure time
inside one sequence, while moving the tripod between positions takes many seconds. So, in capture order (EXIF
DateTimeOriginal, falling back to the file name when any image lacks one), a new bracket starts where
  - the gap to the previous frame exceeds `max_gap_s`, or
  - the exposure time repeats one already in the current bracket (the camera restarted its sequence — this also
    splits brackets shot faster than `max_gap_s` apart).
Nothing is merged silently: a group smaller than `min_size` (a singleton), larger than `max_size`, or a frame
without an exposure time (set aside before the split) is kept as a Bracket with an `issue` and `position_index = -1`;
the pipeline skips and reports it. `apply_overrides` is the manual escape hatch for the odd case the rule gets wrong.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any

from .exif import ExposureInfo

log = logging.getLogger(__name__)

#: Two exposure times closer than this (relative) are "the same" — EXIF rationals round (1/60 vs 0.0167), while the
#: smallest bracket step a camera offers (1/3 EV) is a 26 % change.
SAME_TIME_RTOL = 0.05

ISSUE_SINGLETON = "singleton"
ISSUE_TOO_SMALL = "too_small"
ISSUE_OVERSIZE = "oversize"
ISSUE_NO_EXPOSURE = "no_exposure_time"


@dataclass
class Bracket:
    """One camera position: its frames in capture order. `position_index` is the dataset position (pos_NNN) or -1."""

    id: int
    members: list[ExposureInfo]
    position_index: int
    issue: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.issue is None

    @property
    def names(self) -> list[str]:
        return [m.name for m in self.members]

    def by_exposure(self) -> list[ExposureInfo]:
        """Members sorted by exposure time (darkest first); members without one go last."""
        return sorted(self.members, key=lambda m: (m.exposure_time is None, m.exposure_time or 0.0, m.name))


# ── Grouping ──────────────────────────────────────────────────────────────────────────────────────────────────────────


def same_time(a: float | None, b: float | None, rtol: float = SAME_TIME_RTOL) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= rtol * max(abs(a), abs(b))


def capture_order(infos: list[ExposureInfo]) -> list[ExposureInfo]:
    """By timestamp (then name) when every image has one; else by file name for all (a mixed order is ill-defined)."""
    if infos and all(i.timestamp is not None for i in infos):
        return sorted(infos, key=lambda i: (i.timestamp, i.name))
    if any(i.timestamp is not None for i in infos):
        log.warning("some images have no capture time: grouping by file-name order")
    return sorted(infos, key=lambda i: i.name)


def _classify(members: list[ExposureInfo], min_size: int, max_size: int) -> str | None:
    if any(m.exposure_time is None for m in members):
        return ISSUE_NO_EXPOSURE
    if len(members) < min_size:
        return ISSUE_SINGLETON if len(members) == 1 else ISSUE_TOO_SMALL
    if len(members) > max_size:
        return ISSUE_OVERSIZE
    return None


def _finalise(brackets: list[Bracket], min_size: int, max_size: int) -> list[Bracket]:
    """(Re)classify every bracket and number the usable ones as positions 0.. in order."""
    pos = 0
    for b in brackets:
        b.issue = _classify(b.members, min_size, max_size)
        if b.ok:
            b.position_index = pos
            pos += 1
        else:
            b.position_index = -1
    return brackets


def group_brackets(
    infos: list[ExposureInfo], max_gap_s: float = 5.0, min_size: int = 2, max_size: int = 9
) -> list[Bracket]:
    """Split the images into brackets (see the module docstring); every image lands in exactly one Bracket.

    An image without an exposure time cannot be merged and says nothing about the sequence: it is kept out of the
    split (one stray file must not poison a bracket, nor the timestamp order) and reported as its own Bracket with
    the issue `no_exposure_time`, after the others."""
    timed = [i for i in infos if i.exposure_time is not None]
    untimed = sorted((i for i in infos if i.exposure_time is None), key=lambda i: i.name)
    ordered = capture_order(timed)
    groups: list[list[ExposureInfo]] = []
    current: list[ExposureInfo] = []
    for info in ordered:
        if current:
            prev = current[-1]
            gap = 0.0
            if info.timestamp is not None and prev.timestamp is not None:
                gap = (info.timestamp - prev.timestamp).total_seconds()
            repeat = any(same_time(info.exposure_time, m.exposure_time) for m in current)
            if gap > max_gap_s or repeat:
                groups.append(current)
                current = []
        current.append(info)
    if current:
        groups.append(current)
    groups.extend([i] for i in untimed)
    brackets = [Bracket(id=i, members=g, position_index=-1) for i, g in enumerate(groups)]
    _finalise(brackets, min_size, max_size)
    for b in brackets:
        if not b.ok:
            log.warning("bracket %d (%s): %s", b.id, ", ".join(b.names), b.issue)
    return brackets


def apply_overrides(
    brackets: list[Bracket], overrides: dict[str, int | None], min_size: int = 2, max_size: int = 9
) -> list[Bracket]:
    """Manual reassignment, `{"IMG_0007.jpg": 3}`: that file moves to bracket 3 (created if no bracket has that id).

    Keys are file names (or full paths); a value of None or a negative id drops the file from every bracket. Bracket
    ids are kept, emptied brackets disappear, brackets are re-sorted by their first frame and re-classified, and the
    usable ones renumbered as positions. Returns a new list; the input is not modified.
    """
    assign: dict[str, int | None] = {}
    by_name: dict[str, ExposureInfo] = {}
    for b in brackets:
        for m in b.members:
            assign[m.name] = b.id
            by_name[m.name] = m
    for key, target in (overrides or {}).items():
        name = os.path.basename(str(key))
        if name not in by_name:
            log.warning("override for %s: no such image, ignored", name)
            continue
        assign[name] = None if target is None or int(target) < 0 else int(target)
    groups: dict[int, list[ExposureInfo]] = {}
    for name, bid in assign.items():
        if bid is not None:
            groups.setdefault(bid, []).append(by_name[name])
    out: list[Bracket] = []
    for bid, members in groups.items():
        out.append(Bracket(id=bid, members=capture_order(members), position_index=-1))
    rank = {m.name: i for i, m in enumerate(capture_order(list(by_name.values())))}
    out.sort(key=lambda b: (rank[b.members[0].name], b.id))
    return _finalise(out, min_size, max_size)


def grouping_summary(brackets: list[Bracket]) -> dict[str, Any]:
    """Counts the UI and the report show: usable brackets, singletons, other issues."""
    return {
        "brackets": sum(1 for b in brackets if b.ok),
        "singletons": sum(1 for b in brackets if b.issue == ISSUE_SINGLETON),
        "issues": {b.id: b.issue for b in brackets if not b.ok},
        "sizes": sorted({len(b.members) for b in brackets if b.ok}),
    }


# ── JSON ──────────────────────────────────────────────────────────────────────────────────────────────────────────────


def brackets_to_json(brackets: list[Bracket]) -> str:
    """`{"brackets": [{"id", "position_index", "issue", "members": [ExposureInfo dicts]}]}` as a string."""
    return json.dumps(
        {
            "brackets": [
                {
                    "id": b.id,
                    "position_index": b.position_index,
                    "issue": b.issue,
                    "members": [m.to_dict() for m in b.members],
                }
                for b in brackets
            ]
        },
        indent=1,
    )


def brackets_from_json(text: str) -> list[Bracket]:
    data = json.loads(text)
    out = []
    for d in data["brackets"]:
        out.append(
            Bracket(
                id=int(d["id"]),
                members=[ExposureInfo.from_dict(m) for m in d["members"]],
                position_index=int(d.get("position_index", -1)),
                issue=d.get("issue"),
            )
        )
    return out


def exposure_steps_ev(b: Bracket) -> list[float]:
    """The bracket's EV offsets from its middle frame (log2 of exposure-time ratios), for display."""
    times = [m.exposure_time for m in b.by_exposure() if m.exposure_time]
    if not times:
        return []
    mid = times[len(times) // 2]
    return [math.log2(t / mid) for t in times]


#: The contract's short names.
to_json = brackets_to_json
from_json = brackets_from_json
