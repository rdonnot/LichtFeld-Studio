"""MTB alignment and the ghost score."""

from __future__ import annotations

import numpy as np

from hdr_brackets.align import FLAG_THRESHOLD, align_bracket, align_bracket_detailed, bitmap_difference, to_gray
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
    imgs = _bracket(set6, 1)
    moved = [np.roll(imgs[0], 3, axis=1), imgs[1], np.roll(imgs[2], -2, axis=0)]
    before = bitmap_difference(to_gray(imgs[1]), to_gray(moved[0]))
    r = align_bracket_detailed(moved)
    assert r.reference == 1
    assert r.shifts == [(-3, 0), (0, 0), (0, 2)]
    m = 4  # away from the wrapped / zero-filled border
    assert np.array_equal(r.images[0][m:-m, m:-m], imgs[0][m:-m, m:-m])
    assert np.array_equal(r.images[2][m:-m, m:-m], imgs[2][m:-m, m:-m])
    # a smooth frame's median bitmap has one contour: 3 px of drift flips only the band along it (≈ 4 % here)
    assert r.score < 0.005 and before > 0.02, (r.score, before)


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
