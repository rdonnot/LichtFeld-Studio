"""EXIF reading and bracket grouping against the synthetic camera's truth (exposures_gt.json)."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PIL.TiffImagePlugin import IFDRational

from hdr_brackets.exif import ExposureInfo, compute_ev, list_images, rational, read_exposure
from hdr_brackets.grouping import (
    ISSUE_OVERSIZE,
    ISSUE_SINGLETON,
    apply_overrides,
    brackets_from_json,
    brackets_to_json,
    group_brackets,
)


def _infos(bs) -> list[ExposureInfo]:
    return [read_exposure(p) for p in list_images(bs.raw_dir)]


# ── EXIF ──────────────────────────────────────────────────────────────────────────────────────────────────────────────


def test_read_exposure_matches_truth(set3) -> None:
    for r in set3.images():
        info = read_exposure(set3.raw_dir / r["file"])
        assert info.exposure_time == pytest.approx(r["exposure_time"], rel=1e-6)
        assert info.fnumber == pytest.approx(r["fnumber"], rel=1e-6)
        assert info.iso == pytest.approx(r["iso"])
        assert info.ev == pytest.approx(r["exif_ev"], abs=1e-6)  # LichtFeld's convention, as the truth wrote it
        assert info.timestamp == datetime.strptime(r["timestamp"], "%Y:%m:%d %H:%M:%S")
        assert (info.width, info.height) == (96, 64)


def test_missing_exif_is_not_an_error(tmp_path: Path) -> None:
    p = tmp_path / "bare.jpg"
    Image.fromarray(np.full((8, 8, 3), 128, np.uint8)).save(p)
    info = read_exposure(p)
    assert info.exposure_time is None and info.ev is None and info.timestamp is None
    assert (info.width, info.height) == (8, 8)


def test_rational_forms() -> None:
    assert rational(IFDRational(1, 250)) == pytest.approx(0.004)
    assert rational((1, 250)) == pytest.approx(0.004)
    assert rational((5,)) == 5.0
    assert rational(2.8) == 2.8
    assert rational((1, 0)) is None and rational("x") is None
    assert compute_ev(None, None, None) is None
    assert compute_ev(0.01, 4.0, 100) == pytest.approx(np.log2(0.01 / 16 * 100))
    assert compute_ev(0.01, None, None) == pytest.approx(np.log2(0.01))  # missing factors = 1, as exif.cpp


# ── Grouping ──────────────────────────────────────────────────────────────────────────────────────────────────────────


def _truth_groups(bs) -> list[list[str]]:
    views: dict[int, list[str]] = {}
    for r in bs.images():
        views.setdefault(r["view_index"], []).append(r["file"])
    return [sorted(v) for _, v in sorted(views.items())]


def test_grouping_exact(set3) -> None:
    brackets = group_brackets(_infos(set3))
    assert [sorted(b.names) for b in brackets] == _truth_groups(set3)
    assert all(b.ok for b in brackets)
    assert [b.position_index for b in brackets] == [0, 1, 2]


def test_grouping_splits_on_repeated_exposure(set3) -> None:
    """Brackets fired back to back (1 s apart, no gap) still split where the exposure sequence restarts."""
    infos = sorted(_infos(set3), key=lambda i: i.name)
    t0 = infos[0].timestamp
    tight = [dataclasses.replace(i, timestamp=t0 + timedelta(seconds=k)) for k, i in enumerate(infos)]
    assert [sorted(b.names) for b in group_brackets(tight)] == _truth_groups(set3)


def test_grouping_two_frame_bracket_and_singleton(set3) -> None:
    infos = sorted(_infos(set3), key=lambda i: i.name)
    # view 1 lost its +2 EV frame (IMG_0006) → a valid 2-frame bracket; a stray frame a minute later → a singleton
    infos = [i for i in infos if i.name != "IMG_0006.jpg"]
    stray = dataclasses.replace(
        infos[0], path=str(set3.raw_dir / "IMG_0099.jpg"), timestamp=infos[-1].timestamp + timedelta(seconds=60)
    )
    brackets = group_brackets(infos + [stray])
    assert [b.names for b in brackets] == [
        ["IMG_0001.jpg", "IMG_0002.jpg", "IMG_0003.jpg"],
        ["IMG_0004.jpg", "IMG_0005.jpg"],
        ["IMG_0007.jpg", "IMG_0008.jpg", "IMG_0009.jpg"],
        ["IMG_0099.jpg"],
    ]
    assert [b.issue for b in brackets] == [None, None, None, ISSUE_SINGLETON]
    assert [b.position_index for b in brackets] == [0, 1, 2, -1]


def test_grouping_oversize_is_reported(set3) -> None:
    base = read_exposure(set3.raw_dir / "IMG_0001.jpg")
    seq = [
        dataclasses.replace(
            base, path=f"/x/F_{k:02d}.jpg", exposure_time=0.001 * 2**k, timestamp=base.timestamp + timedelta(seconds=k)
        )
        for k in range(10)
    ]
    (b,) = group_brackets(seq, max_size=9)
    assert b.issue == ISSUE_OVERSIZE and b.position_index == -1


def test_grouping_without_timestamps_uses_names(set3) -> None:
    infos = [dataclasses.replace(i, timestamp=None) for i in _infos(set3)]
    assert [sorted(b.names) for b in group_brackets(infos[::-1])] == _truth_groups(set3)


def test_manual_override(set3) -> None:
    brackets = group_brackets(_infos(set3))
    moved = apply_overrides(brackets, {"IMG_0007.jpg": 1, "IMG_0009.jpg": None, "NOPE.jpg": 0})
    assert [b.names for b in moved] == [
        ["IMG_0001.jpg", "IMG_0002.jpg", "IMG_0003.jpg"],
        ["IMG_0004.jpg", "IMG_0005.jpg", "IMG_0006.jpg", "IMG_0007.jpg"],
        ["IMG_0008.jpg"],
    ]
    assert [b.id for b in moved] == [0, 1, 2]
    assert [b.issue for b in moved] == [None, None, ISSUE_SINGLETON]
    # a new id makes a new bracket; the input list is untouched
    split = apply_overrides(brackets, {"IMG_0003.jpg": 7})
    assert [b.id for b in split] == [0, 7, 1, 2] and split[1].names == ["IMG_0003.jpg"]
    assert len(brackets[0].members) == 3


def test_brackets_json_round_trip(set3) -> None:
    brackets = group_brackets(_infos(set3))
    back = brackets_from_json(brackets_to_json(brackets))
    assert [(b.id, b.position_index, b.issue, b.names) for b in back] == [
        (b.id, b.position_index, b.issue, b.names) for b in brackets
    ]
    assert back[0].members[0] == brackets[0].members[0]
