"""MTB alignment and the ghost score."""

from __future__ import annotations

import numpy as np

from hdr_brackets import align as align_mod
from hdr_brackets.align import (
    FLAG_THRESHOLD,
    align_bracket,
    align_bracket_detailed,
    bitmap_difference,
    search_failed,
    to_gray,
)
from hdr_brackets.pipeline import load_rgb8


def _bracket(bs, vi: int) -> list[np.ndarray]:
    return [load_rgb8(bs.raw_dir / r["file"]) for r in sorted(bs.view(vi), key=lambda r: r["exposure_time"])]


def test_tripod_bracket_is_unchanged(set6) -> None:
    for vi in range(3):
        imgs = _bracket(set6, vi)
        out, score = align_bracket(imgs)
        assert all(np.array_equal(a, b) for a, b in zip(out, imgs, strict=True))
        assert score < 0.01, score


def test_shifted_frame_is_realigned(set6) -> None:
    imgs = _bracket(set6, 0)  # grey medians 72 / 157 / 245: every frame searchable
    moved = [imgs[0], np.roll(imgs[1], 2, axis=0), np.roll(imgs[2], -2, axis=1)]
    before = bitmap_difference(to_gray(imgs[0]), to_gray(moved[2]))
    r = align_bracket_detailed(moved)
    assert r.reference == 0  # the best-exposed frame (most grey pixels in [5, 250]), not the middle one
    assert r.max_bits == 2  # 96 px: a search reaches ±3 px
    assert r.shifts == [(0, 0), (0, -2), (2, 0)] and r.status == ["reference", "mtb", "mtb"], (r.shifts, r.status)
    m = 4  # away from the wrapped / zero-filled border
    assert np.array_equal(r.images[1][m:-m, m:-m], imgs[1][m:-m, m:-m])
    assert np.array_equal(r.images[2][m:-m, m:-m], imgs[2][m:-m, m:-m])
    # a smooth frame's median bitmap has few contours: 2 px of drift flips only the bands along them
    assert r.score < 0.005 and before > 0.01, (r.score, before)
    # an explicit reference is honoured
    r1 = align_bracket_detailed(moved, reference=1)
    assert r1.reference == 1 and r1.shifts[0] == (0, 2) and r1.status[1] == "reference"


def test_search_at_its_limit_is_a_failure(set6) -> None:
    """A result at ±(2^bits − 1) cannot be told from a search that ran away: (0, 0), status "failed"."""
    imgs = _bracket(set6, 0)
    r = align_bracket_detailed([imgs[0], np.roll(imgs[1], 3, axis=0), imgs[2]])  # 3 px = the 96 px pyramid's limit
    assert r.status[1] == "failed" and r.shifts[1] == (0, 0) and r.failed == [1]
    assert search_failed((-31, -31), 5) and search_failed((0, 31), 5) and not search_failed((30, -30), 5)


def test_runaway_search_is_rejected(set6, monkeypatch) -> None:
    """Whatever calculateShift returns at the limit (the (−31, −31) of the 5-EV room) is never applied."""
    imgs = _bracket(set6, 0)
    monkeypatch.setattr(align_mod, "_calculate_shift", lambda mtb, a, b: (-3, -3))
    r = align_bracket_detailed(imgs)
    assert r.shifts == [(0, 0)] * 3 and r.status == ["reference", "failed", "failed"] and r.shift_max == 0


def test_clipped_frames_are_not_searched(set5ev) -> None:
    """−4…+4 EV (the camera's default): the +4 frame (median 255) and, on the bright view, the 0 EV frame too are
    never searched; the bright view's reference is not its clipped middle frame; every shift is (0, 0)."""
    for vi in range(len({r["view_index"] for r in set5ev.images()})):
        imgs = _bracket(set5ev, vi)
        r = align_bracket_detailed(imgs)
        meds = [float(np.median(to_gray(im))) for im in imgs]
        assert r.shifts == [(0, 0)] * 5, (vi, r.shifts, r.status, meds)
        assert r.status[r.reference] == "reference" and 5 < meds[r.reference] < 250
        for i, med in enumerate(meds):
            if not 5 < med < 250:
                assert r.status[i] == "clipped", (vi, i, med)
        assert r.status[4] == "clipped"  # +4 EV
    bright = _bracket(set5ev, set5ev.bright_view)
    assert np.median(to_gray(bright[2])) >= 250  # the mid (0 EV) frame of the bright view is clipped …
    r = align_bracket_detailed(bright)
    assert r.reference < 2 and r.status[2] == "clipped"  # … so it is neither the reference nor searched


def test_moving_object_is_flagged(set6) -> None:
    imgs = _bracket(set6, 2)
    ghost = imgs[2].copy()
    h, w = ghost.shape[:2]
    small = ghost.copy()
    small[h // 4 : h // 2, w // 3 : w // 3 + w // 4] = 255 - small[h // 4 : h // 2, w // 3 : w // 3 + w // 4]
    big = ghost.copy()
    big[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 255 - big[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    _, s_small = align_bracket([imgs[0], imgs[1], small])
    _, s_big = align_bracket([imgs[0], imgs[1], big])
    # the score is the share of the frame that changed: 1/16 of it stays under the flag, a quarter is flagged
    assert 0.03 < s_small < FLAG_THRESHOLD < s_big, (s_small, s_big)
