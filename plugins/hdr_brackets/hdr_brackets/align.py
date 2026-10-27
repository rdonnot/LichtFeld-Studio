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

Clipped frames (review of 2026-10-27, measured on the 5-EV synthetic room): a frame whose grey median is ≥ 250 or ≤ 5
has a degenerate median bitmap — the exclusion band around the median swallows almost every pixel, so
`calculateShift` matches noise and wanders to its limit (2^bits − 1 px: (−31, −31) on 24/40 EV+4 frames of a
motionless tripod set). So:
  - the reference is the frame with the most grey pixels in [CLIP_LO, CLIP_HI] (not the middle exposure, which on a
    bright view is itself clipped — its median read 252 on pos_009);
  - a frame whose median is outside (CLIP_LO, CLIP_HI) is never searched: it takes the shift of its nearest measured
    neighbour in exposure order (Ward's adjacent-exposure idea — frames shot one after another moved together; on a
    tripod that is (0, 0)); status "clipped";
  - a search that ends with |dx| or |dy| ≥ 2^bits − 1 hit the pyramid's limit: that is a failed search, not a
    measurement — the frame keeps (0, 0), status "failed", and the caller reports it.
`AlignResult.status[i]` says which of "reference" / "mtb" / "clipped" / "failed" frame i is. The ghost score is taken
over the searched frames only (a clipped frame's bitmap carries no information).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

log = logging.getLogger(__name__)

#: Ghost score above which a bracket is flagged (fraction of differing median-threshold bits).
FLAG_THRESHOLD = 0.08
#: MTB's exclusion half-width around the median, in 8-bit levels (OpenCV's default).
EXCLUDE_RANGE = 4
#: Grey levels a frame must keep its median strictly between to be searched (and the in-range window of the reference).
CLIP_LO = 5
CLIP_HI = 250

STATUS_REFERENCE = "reference"
STATUS_MTB = "mtb"
STATUS_CLIPPED = "clipped"
STATUS_FAILED = "failed"


@dataclass
class AlignResult:
    """`images` aligned to `images[reference]`; `shifts[i]` = (dx, dy) applied to frame i (pixels); `status[i]` how
    that shift was obtained (module docstring); `medians[i]` / `in_range[i]` the grey median and the share of grey
    pixels in [CLIP_LO, CLIP_HI]; `max_bits` the pyramid depth used (a search limit of 2^max_bits − 1 px)."""

    images: list[np.ndarray]
    score: float
    shifts: list[tuple[int, int]]
    reference: int
    status: list[str] = field(default_factory=list)
    medians: list[float] = field(default_factory=list)
    in_range: list[float] = field(default_factory=list)
    max_bits: int = 0

    @property
    def failed(self) -> list[int]:
        return [i for i, st in enumerate(self.status) if st == STATUS_FAILED]

    @property
    def clipped(self) -> list[int]:
        return [i for i, st in enumerate(self.status) if st == STATUS_CLIPPED]

    @property
    def shift_max(self) -> int:
        return max((max(abs(int(dx)), abs(int(dy))) for dx, dy in self.shifts), default=0)


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


def _calculate_shift(mtb: cv2.AlignMTB, gray_ref: np.ndarray, gray: np.ndarray) -> tuple[int, int]:
    """One MTB search (a seam for tests)."""
    sx, sy = mtb.calculateShift(gray_ref, gray)
    return int(sx), int(sy)


def searchable(gray: np.ndarray, lo: int = CLIP_LO, hi: int = CLIP_HI) -> bool:
    """True when the frame's median bitmap means something: grey median strictly inside (lo, hi)."""
    m = float(np.median(gray))
    return lo < m < hi


def search_failed(shift: tuple[int, int], max_bits: int) -> bool:
    """A search that reached the pyramid's limit (2^bits − 1 px on either axis) found nothing."""
    limit = (1 << int(max_bits)) - 1
    return max(abs(int(shift[0])), abs(int(shift[1]))) >= limit


def best_exposed(grays: list[np.ndarray], lo: int = CLIP_LO, hi: int = CLIP_HI) -> int:
    """Index of the frame with the most grey pixels in [lo, hi] (the first on a tie)."""
    return int(np.argmax([float(((g >= lo) & (g <= hi)).mean()) for g in grays]))


def align_bracket_detailed(
    images_uint8: list[np.ndarray], reference: int | None = None, max_bits: int | None = None
) -> AlignResult:
    """MTB alignment of every frame onto the reference (default: the best-exposed frame, see `best_exposed`; pass
    the frames darkest first — the clipped frames inherit their neighbours' shifts in that order), the shifts, how
    each was obtained, and the ghost score. Frames must share one size. See the module docstring."""
    imgs = [np.asarray(im) for im in images_uint8]
    if not imgs:
        raise ValueError("align_bracket: no images")
    shape = imgs[0].shape
    if any(im.shape != shape for im in imgs):
        raise ValueError(f"align_bracket: frames differ in size: {[im.shape for im in imgs]}")
    if any(im.dtype != np.uint8 for im in imgs):
        raise ValueError("align_bracket wants uint8 frames")
    h, w = shape[:2]
    bits = mtb_max_bits(h, w) if max_bits is None else int(max_bits)
    mtb = cv2.createAlignMTB(bits, EXCLUDE_RANGE, True)
    grays = [to_gray(im) for im in imgs]
    ref = best_exposed(grays) if reference is None else int(reference)
    if not 0 <= ref < len(imgs):
        raise ValueError(f"align_bracket: reference {ref} out of range")
    n = len(imgs)
    shifts: list[tuple[int, int]] = [(0, 0)] * n
    status: list[str] = [STATUS_CLIPPED] * n
    status[ref] = STATUS_REFERENCE
    for i, g in enumerate(grays):
        if i == ref or not searchable(g):
            continue
        sh = _calculate_shift(mtb, grays[ref], g)
        if search_failed(sh, bits):
            log.info(
                "MTB search on frame %d reached its ±%d px limit (%s): failed, kept at (0, 0)", i, (1 << bits) - 1, sh
            )
            status[i] = STATUS_FAILED
        else:
            shifts[i] = sh
            status[i] = STATUS_MTB
    measured = [i for i, st in enumerate(status) if st in (STATUS_REFERENCE, STATUS_MTB)]
    for i in range(n):
        if status[i] == STATUS_CLIPPED:
            # nearest measured frame in exposure order; on a tie the one toward the reference
            j = min(measured, key=lambda k: (abs(k - i), abs(k - ref)))
            shifts[i] = shifts[j]
    out: list[np.ndarray] = []
    score = 0.0
    for i, (im, g) in enumerate(zip(imgs, grays, strict=True)):
        sh = shifts[i]
        if sh == (0, 0):
            aligned, g_al = im, g
        else:
            aligned = mtb.shiftMat(np.ascontiguousarray(im), sh)
            g_al = mtb.shiftMat(g, sh)
        out.append(aligned)
        if status[i] in (STATUS_MTB, STATUS_FAILED):
            score = max(score, bitmap_difference(grays[ref], g_al, _filled_mask((h, w), sh)))
    return AlignResult(
        images=out,
        score=score,
        shifts=shifts,
        reference=ref,
        status=status,
        medians=[float(np.median(g)) for g in grays],
        in_range=[float(((g >= CLIP_LO) & (g <= CLIP_HI)).mean()) for g in grays],
        max_bits=bits,
    )


def align_bracket(images_uint8: list[np.ndarray]) -> tuple[list[np.ndarray], float]:
    """Aligned frames (onto the best-exposed one) and the ghost score (see FLAG_THRESHOLD)."""
    r = align_bracket_detailed(images_uint8)
    return r.images, r.score


def apply_shifts(images_uint8: list[np.ndarray], shifts: list[tuple[int, int]]) -> list[np.ndarray]:
    """Re-apply shifts found earlier (the pipeline aligns once, merges in a second pass)."""
    mtb = cv2.createAlignMTB()
    return [
        im if tuple(s) == (0, 0) else mtb.shiftMat(np.ascontiguousarray(im), (int(s[0]), int(s[1])))
        for im, s in zip(images_uint8, shifts, strict=True)
    ]
