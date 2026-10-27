"""Camera response (CRF) recovery from brackets: Debevec & Malik's least squares, OpenCV's formulation.

Why Debevec: it solves for g(z) = ln(E·t) — the log exposure that produced pixel value z — from the same pixels seen
at several known exposure times, with no assumption about the curve's shape (the synthetic camera's toe/shoulder curve
is deliberately not sRGB). One CRF per camera, from up to `max_brackets` brackets spread over the shoot.

Two engines solve the SAME problem — cv2.createCalibrateDebevec's equations: OpenCV's triangle weights w(z), the data
rows w(z_ij)·(g(z_ij) − x_i − ln t_j), the smoothness rows λ·w(z)·g''(z) for z = 1..254, the anchor g(128) = 0:
  - "numpy" (default): the per-sample unknowns x_i are eliminated in closed form (Schur complement), leaving a
    256 × 256 normal system, so thousands of samples cost milliseconds. Samples sit on OpenCV's own regular grid, per
    bracket; a location clipped (≤ 1 or ≥ 254) in every frame carries no information and is skipped. Each bracket
    keeps its own exposure times, so brackets need not share one exposure set.
  - "opencv": cv2.createCalibrateDebevec(samples, lam, random=False) on a mosaic of the brackets that share the most
    common exposure set (side by side, one mosaic per exposure). Its dense float SVD is slow — 16 s for 70 samples and
    70 s for 420 on this machine (2026-10-27, under build load) — so it is the cross-check, not the default.
Why the defaults differ from OpenCV's (70 samples, λ = 10): on the synthetic camera (shot + read noise, JPEG q95), 70
samples leave max |Δg| ≈ 0.15–0.25 over z ∈ [20, 235] against the truth (noise, not the solver: noise-free frames give
0.015); 1000 samples per bracket and λ = 30 reach 0.11 from 3 brackets and 0.078 from 6 (tests/test_response.py).

Scale: g is known only up to an additive constant (the absolute radiance scale is not observable from LDR frames).
Debevec's own anchor is kept — g(128) = 0 — so a merged EXR is in units where a pixel that reads 128 after a 1 s
exposure has radiance 1.0. Those are the merged EXR's units; anything comparing with a ground truth fits one global
scale (the synthetic truth `crf.json` is anchored at g(255) = 0 instead: compare after matching at z = 128).

File form: `CRF.to_json` writes exactly the tools/synth `crf.json` form — `{"domain": "ldr8", "log_irradiance":
[[256] × 3], …}` (natural log, channels in the order of the arrays fed in: RGB here) — so truth and recovery are the
same file type and `tools/eval/hdr_metrics.apply_crf` reads both.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

log = logging.getLogger(__name__)

ANCHOR_Z = 128
DOMAIN = "ldr8"


@dataclass
class CRF:
    """g(z) per channel, shape (3, 256), natural log, anchored g(128) = 0 (see the module docstring)."""

    log_irradiance: np.ndarray
    source: str = "debevec"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        g = np.asarray(self.log_irradiance, dtype=np.float64)
        if g.ndim == 1:
            g = np.tile(g, (3, 1))
        if g.shape != (3, 256):
            raise ValueError(f"CRF wants (3, 256) log irradiance, got {g.shape}")
        if not np.all(np.isfinite(g)):
            raise ValueError("CRF has non-finite values")
        self.log_irradiance = g

    def response(self) -> np.ndarray:
        """OpenCV's form: (256, 1, 3) float32 = exp(g)."""
        return np.exp(self.log_irradiance).T.reshape(256, 1, 3).astype(np.float32)

    def anchored(self, z: int = ANCHOR_Z) -> CRF:
        """The same curve shifted so g(z) = 0 per channel."""
        g = self.log_irradiance - self.log_irradiance[:, z : z + 1]
        return CRF(g, self.source, dict(self.meta, anchor=f"g({z})=0"))

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"domain": DOMAIN, "source": self.source}
        d.update({k: v for k, v in self.meta.items() if k not in ("domain", "log_irradiance", "source")})
        d["log_irradiance"] = self.log_irradiance.tolist()
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CRF:
        if d.get("domain", DOMAIN) != DOMAIN:
            raise ValueError(f"CRF domain {d.get('domain')!r} is not {DOMAIN!r}")
        meta = {k: v for k, v in d.items() if k not in ("domain", "log_irradiance", "source")}
        return cls(np.asarray(d["log_irradiance"], dtype=np.float64), str(d.get("source", "file")), meta)

    @classmethod
    def from_json(cls, text: str) -> CRF:
        return cls.from_dict(json.loads(text))

    def save(self, path: str | os.PathLike) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str | os.PathLike) -> CRF:
        return cls.from_json(Path(path).read_text())


# ── Calibration ───────────────────────────────────────────────────────────────────────────────────────────────────────


def _time_key(times: list[float]) -> tuple[int, ...]:
    """Exposure-time set rounded to 1/12 EV: brackets with the same key can share one mosaic."""
    return tuple(int(round(np.log2(t) * 12)) for t in sorted(times))


def _choose(items: list[int], k: int) -> list[int]:
    if len(items) <= k:
        return items
    idx = np.linspace(0, len(items) - 1, k).round().astype(int)
    return [items[i] for i in sorted(set(idx.tolist()))]


def triangle_weights() -> np.ndarray:
    """OpenCV 5.0's `triangleWeights` (z up to 127, 255 − z above; merge.hat_weights), float64."""
    z = np.arange(256, dtype=np.float64)
    return np.where(z <= 127, z, 255.0 - z)


def opencv_grid(height: int, width: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """The sample grid of CalibrateDebevec(random=False), exactly: (ys, xs) of ≈ `samples` points."""
    x_points = max(1, min(width, int(np.sqrt(samples * width / height))))
    y_points = max(1, min(height, samples // x_points))
    step_x, step_y = width // x_points, height // y_points
    xs = step_x // 2 + step_x * np.arange(x_points)
    ys = step_y // 2 + step_y * np.arange(y_points)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")  # OpenCV's loop: x outer, y inner
    keep = (xx < width) & (yy < height)
    return yy[keep].ravel(), xx[keep].ravel()


def _accumulate(H: np.ndarray, b: np.ndarray, Z: np.ndarray, log_t: np.ndarray, w: np.ndarray) -> None:
    """Add one bracket's data term, with every sample's x_i eliminated, to the normal system H g = b.

    Σ_ij W_ij (g[z_ij] − l_j − x_i)² with W = w², minimised over x_i, is g^T H g − 2 b^T g + const with
    H = Σ_ij W_ij e e^T − Σ_i a_i a_i^T / S_i,  b = Σ_ij W_ij l_j e − Σ_i c_i a_i / S_i
    (a_i = Σ_j W_ij e_{z_ij}, c_i = Σ_j W_ij l_j, S_i = Σ_j W_ij)."""
    Z = Z.astype(np.int64)
    W = w[Z] ** 2
    S = W.sum(axis=1)
    ok = S > 0  # a sample at 0 / 255 in every frame has no weight at all
    Z, W, S = Z[ok], W[ok], S[ok]
    c = W @ log_t
    P = Z.shape[1]
    for j in range(P):
        np.add.at(H, (Z[:, j], Z[:, j]), W[:, j])
        np.add.at(b, Z[:, j], W[:, j] * log_t[j] - W[:, j] * c / S)
        for k in range(P):
            np.add.at(H, (Z[:, j], Z[:, k]), -W[:, j] * W[:, k] / S)


def _solve(H: np.ndarray, b: np.ndarray, lam: float, w: np.ndarray, anchor: int = ANCHOR_Z) -> np.ndarray:
    """Add OpenCV's smoothness rows λ·w(z)·(g[z−1] − 2g[z] + g[z+1]), z = 1..254, fix g[anchor] = 0, solve."""
    i = np.arange(254)
    D = np.zeros((254, 256))
    D[i, i], D[i, i + 1], D[i, i + 2] = 1.0, -2.0, 1.0
    D *= (lam * w[1:255])[:, None]
    A = H + D.T @ D
    keep = np.r_[0:anchor, anchor + 1 : 256]
    g = np.zeros(256)
    g[keep] = np.linalg.lstsq(A[np.ix_(keep, keep)], b[keep], rcond=None)[0]
    return g


def debevec_numpy(
    brackets_images: list[list[np.ndarray]],
    times: list[list[float]],
    samples: int,
    lam: float,
    skip_clipped: bool = True,
) -> tuple[np.ndarray, int]:
    """g (3, 256) from brackets of uint8 RGB frames; `samples` grid points per bracket. Returns (g, samples used)."""
    w = triangle_weights()
    g = np.zeros((3, 256))
    used = 0
    sel: list[tuple[np.ndarray, np.ndarray]] = []
    for imgs in brackets_images:
        h, wd = imgs[0].shape[:2]
        ys, xs = opencv_grid(h, wd, samples)
        if skip_clipped:
            stack = np.stack([im[ys, xs] for im in imgs])  # (P, N, 3)
            informative = ~np.all((stack.max(axis=2) <= 1) | (stack.min(axis=2) >= 254), axis=0)
            ys, xs = ys[informative], xs[informative]
        sel.append((ys, xs))
        used += ys.size
    for ch in range(3):
        H = np.zeros((256, 256))
        b = np.zeros(256)
        for imgs, ts, (ys, xs) in zip(brackets_images, times, sel, strict=True):
            if ys.size == 0:
                continue
            Z = np.stack([im[ys, xs, ch] for im in imgs], axis=1)
            _accumulate(H, b, Z, np.log(np.asarray(ts, dtype=np.float64)), w)
        g[ch] = _solve(H, b, lam, w)
    return g, used


def debevec_opencv(
    brackets_images: list[list[np.ndarray]], times: list[list[float]], samples: int, lam: float
) -> np.ndarray:
    """g (3, 256) through cv2.createCalibrateDebevec on one mosaic per exposure (brackets already filtered to one
    exposure set and frame size); `samples` = OpenCV's grid over the whole mosaic."""
    n_frames = len(brackets_images[0])
    mosaics = [np.ascontiguousarray(np.hstack([b[k] for b in brackets_images])) for k in range(n_frames)]
    t_mean = np.array([np.mean([ts[k] for ts in times]) for k in range(n_frames)], dtype=np.float32)
    resp = cv2.createCalibrateDebevec(int(samples), float(lam), False).process(mosaics, t_mean)
    return np.log(np.maximum(np.asarray(resp, dtype=np.float64).reshape(256, 3), 1e-30)).T


def calibrate_crf(
    brackets_images: list[list[np.ndarray]],
    times: list[list[float]],
    samples: int = 1000,
    lam: float = 30.0,
    max_brackets: int = 12,
    method: str = "numpy",
) -> CRF:
    """Debevec CRF from several brackets (uint8 RGB frames + exposure times in seconds, any order within a bracket).

    `samples`: grid points per bracket ("numpy") or over the whole mosaic ("opencv", which also keeps only the
    brackets sharing the most common exposure set and frame size); `lam`: the smoothness weight; at most
    `max_brackets` brackets, spread over the list. Raises ValueError when no bracket has ≥ 2 frames.
    """
    if len(brackets_images) != len(times):
        raise ValueError("calibrate_crf: one time list per bracket")
    usable: list[tuple[list[np.ndarray], list[float]]] = []
    for imgs, ts in zip(brackets_images, times, strict=True):
        if len(imgs) != len(ts) or len(imgs) < 2:
            continue
        if any(np.asarray(im).shape != np.asarray(imgs[0]).shape for im in imgs):
            continue
        order = np.argsort(ts)
        usable.append(([np.asarray(imgs[i], dtype=np.uint8) for i in order], [float(ts[i]) for i in order]))
    if not usable:
        raise ValueError("calibrate_crf: no bracket with 2 or more frames of one size")
    if method == "opencv":
        keys = [(_time_key(ts), imgs[0].shape) for imgs, ts in usable]
        best_key, _ = Counter(keys).most_common(1)[0]
        members = [i for i, k in enumerate(keys) if k == best_key]
        if len(members) < len(usable):
            log.info("CRF: %d of %d brackets share the dominant exposure set; using those", len(members), len(usable))
        usable = [usable[i] for i in members]
    elif method != "numpy":
        raise ValueError(f"unknown CRF method {method!r}")
    chosen = [usable[i] for i in _choose(list(range(len(usable))), max(1, int(max_brackets)))]
    imgs_c = [c[0] for c in chosen]
    times_c = [c[1] for c in chosen]
    if method == "opencv":
        g = debevec_opencv(imgs_c, times_c, samples, lam)
        n_samples = int(samples)
    else:
        g, n_samples = debevec_numpy(imgs_c, times_c, samples, lam)
    meta: dict[str, Any] = {
        "anchor": f"g({ANCHOR_Z})=0",
        "method": method,
        "brackets_used": len(chosen),
        "frames_per_bracket": sorted({len(i) for i in imgs_c}),
        "exposure_times": times_c[0],
        "samples": n_samples,
        "lambda": float(lam),
        "channels": "RGB",
    }
    # Debevec leaves the extremes (few samples, weight → 0) free to bend back; a merge needs g non-decreasing.
    fixed = np.maximum.accumulate(g, axis=1)
    if not np.array_equal(fixed, g):
        meta["monotonic_fixed_levels"] = int(np.sum(fixed != g))
        g = fixed
    crf = CRF(g, "debevec", meta).anchored(ANCHOR_Z)
    log.info("CRF calibrated (%s) from %d bracket(s), %d samples", method, len(chosen), n_samples)
    return crf


def compare_crf(a: CRF, b: CRF, z_lo: int = 20, z_hi: int = 235, z_match: int = ANCHOR_Z) -> float:
    """Max |Δg| over z in [z_lo, z_hi] after matching both curves at z_match (natural log)."""
    ga = a.log_irradiance - a.log_irradiance[:, z_match : z_match + 1]
    gb = b.log_irradiance - b.log_irradiance[:, z_match : z_match + 1]
    return float(np.max(np.abs(ga - gb)[:, z_lo : z_hi + 1]))
