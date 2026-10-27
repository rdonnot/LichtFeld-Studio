"""Align the frames of one bracket (median threshold bitmaps, cv2 AlignMTB) and score the residual motion.

Why MTB: it compares frames through their median-threshold bitmaps (pixel > the frame's median), which a monotonic
camera response and a change of exposure leave unchanged, so frames 4 stops apart align without knowing the CRF. It
recovers integer translations only — handheld rotation is left to the ghost score and the deghosting merge.

The ghost score: after alignment, each frame's bitmap is compared with the reference's over the pixels both
exclusion bitmaps keep (|value − median| > EXCLUDE_RANGE: the noisy near-median pixels flip at random) and that the
shift did not fill; the score is the largest fraction of differing bits over the frames. A tripod bracket scores ≈ 0
(only clipping and noise flip bits); something that moved — a person, a branch, a handheld drift MTB cannot undo —
flips the bits it covers and along its edges, so the score reads roughly as the share of the frame that changed.
`FLAG_THRESHOLD = 0.08`: above 8 % differing bits the bracket is flagged as possibly ghosted (it is still merged; the
deghosting weights then matter). Measured on the synthetic tripod brackets at ±2 EV (tests/test_align.py): 0.0 after
alignment; an object changing 1/16 of the frame 6.4 % (not flagged), a quarter of it > 20 % (flagged); 3 px of
uncorrected drift on these smooth frames only ≈ 4 % (a median bitmap of a smooth frame has one contour) — on a
textured real frame the same drift flips far more.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)

#: Ghost score above which a bracket is flagged (fraction of differing median-threshold bits).
FLAG_THRESHOLD = 0.08
#: MTB's exclusion half-width around the median, in 8-bit levels (OpenCV's default).
EXCLUDE_RANGE = 4


@dataclass
class AlignResult:
    """`images` aligned to `images[reference]`; `shifts[i]` = (dx, dy) applied to frame i (pixels)."""

    images: list[np.ndarray]
    score: float
    shifts: list[tuple[int, int]]
    reference: int


def mtb_max_bits(height: int, width: int) -> int:
    """Pyramid depth for an image size: OpenCV's default 6 (shifts up to ±127 px) needs ≥ ~1000 px; on a 96 px image
    the coarsest level would be 1–2 px wide and the search wanders to its limit. Keeps ≥ 16 px at the top level."""
    side = max(1, min(int(height), int(width)))
    return int(min(6, max(1, math.floor(math.log2(max(side / 16.0, 1.0))))))


def to_gray(img: np.ndarray) -> np.ndarray:
    """uint8 RGB (or already grey) → uint8 grey."""
    a = np.asarray(img)
    if a.ndim == 2:
        return np.ascontiguousarray(a)
    return cv2.cvtColor(np.ascontiguousarray(a[..., :3]), cv2.COLOR_RGB2GRAY)


def _filled_mask(shape: tuple[int, int], shift: tuple[int, int]) -> np.ndarray:
    """True where a frame shifted by (dx, dy) still holds image content (shiftMat fills the rest with zeros)."""
    h, w = shape
    dx, dy = int(shift[0]), int(shift[1])
    m = np.ones((h, w), dtype=bool)
    if dx > 0:
        m[:, :dx] = False
    elif dx < 0:
        m[:, w + dx :] = False
    if dy > 0:
        m[:dy, :] = False
    elif dy < 0:
        m[h + dy :, :] = False
    return m


def bitmap_difference(
    gray_ref: np.ndarray, gray: np.ndarray, valid: np.ndarray | None = None, exclude_range: int = EXCLUDE_RANGE
) -> float:
    """Fraction of differing median-threshold bits over the pixels both exclusion bitmaps keep (and `valid`)."""
    mtb = cv2.createAlignMTB(1, int(exclude_range), True)
    tb0, eb0 = mtb.computeBitmaps(np.ascontiguousarray(gray_ref))
    tb1, eb1 = mtb.computeBitmaps(np.ascontiguousarray(gray))
    keep = (eb0 > 0) & (eb1 > 0)
    if valid is not None:
        keep &= valid
    n = int(keep.sum())
    if n == 0:
        return 0.0
    diff = ((tb0 > 0) != (tb1 > 0)) & keep
    return float(diff.sum()) / n


def align_bracket_detailed(
    images_uint8: list[np.ndarray], reference: int | None = None, max_bits: int | None = None
) -> AlignResult:
    """MTB alignment of every frame onto the reference (default: the middle one — pass the frames darkest first,
    so the middle is the mid exposure), the shifts and the ghost score. Frames must share one size."""
    imgs = [np.asarray(im) for im in images_uint8]
    if not imgs:
        raise ValueError("align_bracket: no images")
    shape = imgs[0].shape
    if any(im.shape != shape for im in imgs):
        raise ValueError(f"align_bracket: frames differ in size: {[im.shape for im in imgs]}")
    if any(im.dtype != np.uint8 for im in imgs):
        raise ValueError("align_bracket wants uint8 frames")
    ref = len(imgs) // 2 if reference is None else int(reference)
    h, w = shape[:2]
    bits = mtb_max_bits(h, w) if max_bits is None else int(max_bits)
    mtb = cv2.createAlignMTB(bits, EXCLUDE_RANGE, True)
    grays = [to_gray(im) for im in imgs]
    out: list[np.ndarray] = []
    shifts: list[tuple[int, int]] = []
    score = 0.0
    for i, (im, g) in enumerate(zip(imgs, grays, strict=True)):
        if i == ref:
            out.append(im)
            shifts.append((0, 0))
            continue
        sx, sy = (int(v) for v in mtb.calculateShift(grays[ref], g))
        if (sx, sy) == (0, 0):
            aligned, g_al = im, g
        else:
            aligned = mtb.shiftMat(np.ascontiguousarray(im), (sx, sy))
            g_al = mtb.shiftMat(g, (sx, sy))
        out.append(aligned)
        shifts.append((sx, sy))
        score = max(score, bitmap_difference(grays[ref], g_al, _filled_mask((h, w), (sx, sy))))
    return AlignResult(images=out, score=score, shifts=shifts, reference=ref)


def align_bracket(images_uint8: list[np.ndarray]) -> tuple[list[np.ndarray], float]:
    """Aligned frames (onto the middle one) and the ghost score (see FLAG_THRESHOLD)."""
    r = align_bracket_detailed(images_uint8)
    return r.images, r.score


def apply_shifts(images_uint8: list[np.ndarray], shifts: list[tuple[int, int]]) -> list[np.ndarray]:
    """Re-apply shifts found earlier (the pipeline aligns once, merges in a second pass)."""
    mtb = cv2.createAlignMTB()
    return [
        im if tuple(s) == (0, 0) else mtb.shiftMat(np.ascontiguousarray(im), (int(s[0]), int(s[1])))
        for im, s in zip(images_uint8, shifts, strict=True)
    ]
