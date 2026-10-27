"""Merge one aligned bracket into linear radiance (HxWx3 float32), Debevec's weighted average in log space.

    ln E(p) = Σ_f w(Z_f(p)) · (g(Z_f(p)) − ln t_f) / Σ_f w(Z_f(p))

with w the hat (triangle) weight of the 8-bit level — OpenCV's own `triangleWeights` (5.0: z up to 127, 255 − z
above, so 0 at both ends) averaged over the three channels, so every frame gets one weight per pixel. Where every
frame is 0 or 255 in every channel the sum is 0/0 — OpenCV falls back to the unweighted mean of the frames'
estimates there (a fully saturated pixel then reads as if the middle exposure had just reached 255); both paths here
give such a pixel the bound its frames do prove instead: the darkest frame's estimate when saturated (radiance at
least that), the brightest frame's when black (at most that). The plain path is cv2.createMergeDebevec plus that
rule; the deghost path is the
same sum in numpy with one change: per pixel and frame, a sample
whose log-radiance estimate differs from the REFERENCE frame's estimate by more than `DEGHOST_LOG_THRESHOLD` (1.0 in
natural log, a factor e ≈ 2.7, any channel) is down-weighted by `GHOST_WEIGHT`, wherever the reference itself is
not clipped (all channels in [DEGHOST_CLIP_LO, DEGHOST_CLIP_HI]). Moving objects then take the reference frame's
content instead of a cross-fade, and clipped or crushed samples of the other frames (which the hat weight only
attenuates) stop biasing the result. Units are those of the CRF (g(128) = 0, see response.py).
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .response import CRF

log = logging.getLogger(__name__)

#: |ln E_f − ln E_ref| above which a sample is treated as a ghost (natural log).
DEGHOST_LOG_THRESHOLD = 1.0
#: Weight multiplier of a ghost sample (small, not 0: a ghost is down-weighted, the reference still dominates).
GHOST_WEIGHT = 0.02
#: The reference counts as unclipped where every channel is inside these 8-bit levels.
DEGHOST_CLIP_LO = 5
DEGHOST_CLIP_HI = 250


def hat_weights() -> np.ndarray:
    """OpenCV 5.0's triangle weights, (256,) float32: z for z ≤ 127, 255 − z above (0 at 0 and 255). Measured: the
    numpy sum with these reproduces cv2.createMergeDebevec to 1e-6 (z + 1 / 256 − z, an older form, is 3 % off)."""
    z = np.arange(256, dtype=np.float32)
    return np.where(z <= 127, z, 255.0 - z).astype(np.float32)


def _fill_unweighted(
    out: np.ndarray, den: np.ndarray, imgs: list[np.ndarray], times: list[float], crf: CRF
) -> np.ndarray:
    """Pixels with no weight in any frame (every channel 0 or 255 everywhere) → the bound the frames give."""
    bad = ~(den > 0) | ~np.all(np.isfinite(out), axis=2)
    if not bad.any():
        return out
    g = crf.log_irradiance
    order = np.argsort(np.asarray(times, dtype=np.float64))
    dark, bright = int(order[0]), int(order[-1])
    mean_level = np.mean([im.mean(axis=2) for im in imgs], axis=0)
    for src, sel in ((dark, bad & (mean_level >= 128)), (bright, bad & (mean_level < 128))):
        if sel.any():
            z = imgs[src][sel]
            est = np.stack([g[c][z[:, c]] for c in range(3)], axis=1) - np.log(times[src])
            out[sel] = np.exp(est).astype(np.float32)
    return out


def _check(images_uint8: list[np.ndarray], times: list[float]) -> list[np.ndarray]:
    imgs = [np.ascontiguousarray(np.asarray(im)) for im in images_uint8]
    if not imgs:
        raise ValueError("merge: no images")
    if len(imgs) != len(times):
        raise ValueError(f"merge: {len(imgs)} images but {len(times)} exposure times")
    shape = imgs[0].shape
    for im in imgs:
        if im.shape != shape or im.dtype != np.uint8 or im.ndim != 3 or im.shape[2] != 3:
            raise ValueError(f"merge wants uint8 HxWx3 frames of one size, got {im.shape} {im.dtype}")
    if any(not (t > 0) for t in times):
        raise ValueError(f"merge: exposure times must be > 0, got {times}")
    return imgs


def reference_index(times: list[float], reference: str | int = "middle") -> int:
    """'middle' (the median exposure time), 'darkest', 'brightest', or an index into the frames as given."""
    order = list(np.argsort(np.asarray(times, dtype=np.float64), kind="stable"))
    if isinstance(reference, (int, np.integer)):
        if not 0 <= int(reference) < len(times):
            raise ValueError(f"reference index {reference} out of range")
        return int(reference)
    if reference == "middle":
        return int(order[len(order) // 2])
    if reference == "darkest":
        return int(order[0])
    if reference == "brightest":
        return int(order[-1])
    raise ValueError(f"unknown reference {reference!r}")


def merge_debevec_numpy(
    images_uint8: list[np.ndarray],
    times: list[float],
    crf: CRF,
    deghost: bool = True,
    reference: str | int = "middle",
    threshold: float = DEGHOST_LOG_THRESHOLD,
) -> np.ndarray:
    """The Debevec weighted average in numpy (frame by frame, float32), with the deghost weights when asked."""
    imgs = _check(images_uint8, times)
    g = crf.log_irradiance.astype(np.float32)  # (3, 256)
    hat = hat_weights()
    log_t = np.log(np.asarray(times, dtype=np.float64)).astype(np.float32)

    def estimate(i: int) -> np.ndarray:
        z = imgs[i]
        est = np.empty(z.shape, dtype=np.float32)
        for c in range(3):
            est[..., c] = g[c][z[..., c]]
        est -= log_t[i]
        return est

    r = reference_index(list(times), reference)
    est_ref = estimate(r) if deghost else None
    ref_ok = None
    if deghost:
        zr = imgs[r]
        ref_ok = np.all((zr >= DEGHOST_CLIP_LO) & (zr <= DEGHOST_CLIP_HI), axis=2)
    num = np.zeros(imgs[0].shape, dtype=np.float32)
    den = np.zeros(imgs[0].shape[:2], dtype=np.float32)
    for i, z in enumerate(imgs):
        est = est_ref if (deghost and i == r) else estimate(i)
        w = hat[z].mean(axis=2)
        if deghost and i != r:
            d = np.max(np.abs(est - est_ref), axis=2)
            ghost = (d > threshold) & ref_ok
            w = np.where(ghost, w * GHOST_WEIGHT, w)
        num += w[..., None] * est
        den += w
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.exp(num / den[..., None]).astype(np.float32)
    return _fill_unweighted(out, den, imgs, list(times), crf)


def merge_bracket(
    images_uint8: list[np.ndarray],
    times: list[float],
    crf: CRF,
    deghost: bool = True,
    reference: str | int = "middle",
) -> np.ndarray:
    """Aligned uint8 RGB frames + exposure times (s) + CRF → linear radiance, HxWx3 float32 (see module docstring)."""
    if deghost:
        return merge_debevec_numpy(images_uint8, times, crf, deghost=True, reference=reference)
    imgs = _check(images_uint8, times)
    merger = cv2.createMergeDebevec()
    out = np.asarray(merger.process(imgs, np.asarray(times, dtype=np.float32), crf.response()), dtype=np.float32)
    hat = hat_weights()
    den = np.zeros(imgs[0].shape[:2], dtype=np.float32)
    for z in imgs:
        den += hat[z].mean(axis=2)
    return _fill_unweighted(out, den, imgs, list(times), crf)


def merge_stats(hdr: np.ndarray) -> dict[str, float | int]:
    """min / p1 / p50 / p99 / max over the finite values, and the NaN / inf / negative counts."""
    a = np.asarray(hdr, dtype=np.float32)
    finite = a[np.isfinite(a)]
    stats: dict[str, float | int] = {
        "nan": int(np.isnan(a).sum()),
        "inf": int(np.isinf(a).sum()),
        "negative": int((finite < 0).sum()),
    }
    if finite.size:
        p1, p50, p99 = np.percentile(finite, [1, 50, 99])
        stats.update(min=float(finite.min()), p1=float(p1), p50=float(p50), p99=float(p99), max=float(finite.max()))
    else:
        stats.update(min=float("nan"), p1=float("nan"), p50=float("nan"), p99=float("nan"), max=float("nan"))
    return stats
