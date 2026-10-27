"""Tonemapped 8-bit sRGB proxy of a merged EXR: what SfM (COLMAP/GLOMAP) sees, and the LDR baseline M2 compares to.

Why a proxy: SfM needs one well-exposed 8-bit image per position; neither the darkest nor the brightest frame of a
bracket is that. Steps: auto key — scale the radiance so the geometric-mean luminance (Rec. 709 weights) maps to
0.18 × 2^ev (Reinhard's "key value") — then the operator per channel, then the sRGB OETF, rounded to 8 bit.
Operators: "reinhard" x / (1 + x); "aces" Narkowicz's 2015 fit of the ACES RRT+ODT, (x (2.51 x + 0.03)) /
(x (2.43 x + 0.59) + 0.14), applied to the keyed value as is (his ×0.6 pre-exposure is left to `ev`); "linear" a
clip to [0, 1]. The scale is returned by `proxy_exposure_value` and written to exposures.json, so a proxy pixel maps
back to the EXR's radiance. The pipeline uses ONE scale for the whole dataset by default (the median of the
positions' auto keys), so the proxies are photometrically consistent across views — a per-view key would make the
"LDR baseline" flicker and give SfM brightness jumps between neighbours.
"""

from __future__ import annotations

import numpy as np

OPERATORS = ("reinhard", "aces", "linear")
KEY = 0.18
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
_EPS = 1e-8


def luminance(hdr: np.ndarray) -> np.ndarray:
    return np.asarray(hdr, dtype=np.float64)[..., :3] @ LUMA


def proxy_exposure_value(hdr: np.ndarray, ev: float = 0.0) -> float:
    """The scale tonemap_proxy applies: KEY × 2^ev / geometric-mean luminance (finite, positive pixels only)."""
    y = luminance(hdr)
    y = y[np.isfinite(y) & (y > 0)]
    if y.size == 0:
        return 1.0
    gm = float(np.exp(np.mean(np.log(y + _EPS))))
    return float(KEY * (2.0 ** float(ev)) / max(gm, _EPS))


def srgb_oetf(x: np.ndarray) -> np.ndarray:
    """Linear [0, 1] → sRGB-encoded [0, 1] (IEC 61966-2-1 piecewise)."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def tone(x: np.ndarray, operator: str = "aces") -> np.ndarray:
    """The operator on keyed linear values → display-linear [0, 1]."""
    x = np.maximum(np.asarray(x, dtype=np.float64), 0.0)
    if operator == "reinhard":
        return x / (1.0 + x)
    if operator == "aces":
        return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)
    if operator == "linear":
        return np.clip(x, 0.0, 1.0)
    raise ValueError(f"unknown proxy operator {operator!r} (one of {', '.join(OPERATORS)})")


def tonemap_proxy(hdr: np.ndarray, ev: float = 0.0, operator: str = "aces", scale: float | None = None) -> np.ndarray:
    """HxWx3 linear radiance → HxWx3 uint8 sRGB. `scale`, when given, replaces the auto key (and already includes any
    EV offset: the pipeline passes its shared scale); otherwise the scale is `proxy_exposure_value(hdr, ev)`."""
    a = np.nan_to_num(np.asarray(hdr, dtype=np.float64)[..., :3], nan=0.0, posinf=0.0, neginf=0.0)
    s = proxy_exposure_value(a, ev) if scale is None else float(scale)
    enc = srgb_oetf(tone(a * s, operator))
    return np.clip(np.rint(enc * 255.0), 0, 255).astype(np.uint8)
