"""Shared fixtures: tiny synthetic bracket sets made by the repo's own synthetic camera (tools/synth/make_brackets.py)
from procedural EXRs in a tmp dir — 2–6 views of 24×32 to 96×128 px at −2/0/+2 EV, seconds to build, no Blender.

The procedural radiance is smooth (so JPEG keeps it) with a fine texture (so MTB has edges to lock on) and spans
9.5 stops: inside what ±2 EV through the synthetic CRF records, so a merge can be checked almost everywhere.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest

PLUGIN = Path(__file__).resolve().parents[1]
REPO = PLUGIN.parents[1]
for p in (PLUGIN, REPO / "tools" / "synth", REPO / "tools" / "eval"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import make_brackets as mb  # noqa: E402  (tools/synth: the synthetic camera)
from hdr_metrics import read_exr, write_exr  # noqa: E402  (tools/eval)


def procedural_radiance(h: int, w: int, seed: int, lo: float = -6.0, hi: float = 3.5) -> np.ndarray:
    """HxWx3 float32 linear radiance: a tilted log ramp + soft blobs + a fine texture, stretched to [lo, hi] stops
    (more shadow than highlight, like a room with a window: the CRF's toe gets samples), a per-view tint."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    a = rng.uniform(0, 2 * np.pi)
    f = np.cos(a) * (xx / w - 0.5) + np.sin(a) * (yy / h - 0.5)
    f = f / max(np.ptp(f), 1e-9) * 6.0
    for _ in range(4):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        r = rng.uniform(0.08, 0.2) * min(h, w)
        f += rng.uniform(-1.5, 1.5) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))
    tex = cv2.GaussianBlur(rng.normal(size=(h, w)), (0, 0), 1.3)
    f += tex / (tex.std() + 1e-9) * 0.4
    log2l = lo + (hi - lo) * (f - f.min()) / max(np.ptp(f), 1e-9)
    tint = np.array([1.0, rng.uniform(0.8, 1.0), rng.uniform(0.6, 0.9)])
    return (np.exp2(log2l)[..., None] * tint[None, None, :]).astype(np.float32)


@dataclass
class BracketSet:
    gt_dir: Path
    raw_dir: Path
    meta: dict
    crf_truth: dict
    bright_view: int = -1

    def images(self) -> list[dict]:
        return self.meta["images"]

    def view(self, vi: int) -> list[dict]:
        return [r for r in self.meta["images"] if r["view_index"] == vi]

    def truth(self, vi: int) -> np.ndarray:
        return read_exr(str(self.gt_dir / f"view_{vi:03d}.exr"))


def make_set(
    root: Path,
    n_views: int = 3,
    h: int = 64,
    w: int = 96,
    seed: int = 11,
    evs: tuple[float, ...] | None = (-2.0, 0.0, 2.0),
    radiance=None,
) -> BracketSet:
    """`evs=None` leaves make_brackets' own default (−4, −2, 0, +2, +4: what data/brackets_raw was shot with)."""
    gt = root / "gt"
    gt.mkdir(parents=True, exist_ok=True)
    for i in range(n_views):
        lin = radiance(i) if radiance is not None else procedural_radiance(h, w, seed * 100 + i)
        write_exr(str(gt / f"view_{i:03d}.exr"), lin)
    raw = root / "raw"
    logging.getLogger("conftest").info("make_brackets %s → %s", gt, raw)
    args = ["--gt", str(gt), "--out", str(raw), "--seed", str(seed)]
    if evs is not None:
        args += ["--evs", *[str(e) for e in evs]]
    assert mb.main(args) == 0
    meta = json.loads((raw / "exposures_gt.json").read_text())
    crf = json.loads((raw / "crf.json").read_text())
    return BracketSet(gt, raw, meta, crf)


@pytest.fixture(scope="session")
def set3(tmp_path_factory) -> BracketSet:
    """3 views, 64×96 — grouping, EXIF, pipeline."""
    return make_set(tmp_path_factory.mktemp("set3"), n_views=3, h=64, w=96, seed=11)


@pytest.fixture(scope="session")
def set6(tmp_path_factory) -> BracketSet:
    """6 views, 96×128 — CRF recovery and merge accuracy."""
    return make_set(tmp_path_factory.mktemp("set6"), n_views=6, h=96, w=128, seed=23)


@pytest.fixture(scope="session")
def set5ev(tmp_path_factory) -> BracketSet:
    """4 views, 96×128, at the camera's DEFAULT −4…+4 EV (the ±2 EV sets never clip a mid frame, which is how the
    (−31, −31) shifts of the 5-EV room went unnoticed): the +4 EV frames are mostly clipped, and view 3 is 2^4.5
    brighter than the others (the gain is set from view 0), so its 0 EV frame — the middle one — is clipped too."""
    bright = 3

    def radiance(i: int) -> np.ndarray:
        lin = procedural_radiance(96, 128, 4100 + i)
        return lin * np.float32(2.0**4.5) if i == bright else lin

    bs = make_set(tmp_path_factory.mktemp("set5ev"), n_views=4, h=96, w=128, seed=41, evs=None, radiance=radiance)
    bs.bright_view = bright
    return bs
