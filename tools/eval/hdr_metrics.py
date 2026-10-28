"""HDR metrics: mu-law PSNR in log-luminance, LDR PSNR at a photo's exposure+CRF, EXR I/O.

Pure numpy on HxWx3 arrays; EXR through OpenCV.
CLI:
  hdr_metrics.py mu-psnr a.exr b.exr            (b is the reference)
  hdr_metrics.py ldr-psnr render.exr photo.jpg --exposure 0.01 --crf crf.json [--gain 1.0] [--keep-saturated]
CRF json: {"log_irradiance": [256] or [3][256], "domain": "ldr8"} (Debevec g(z)).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Callable

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

Crf = "dict | Callable[[np.ndarray], np.ndarray]"


def mu_law(x: np.ndarray, mu: float = 5000.0) -> np.ndarray:
    x = np.maximum(np.asarray(x, dtype=np.float64), 0.0)
    return np.log1p(mu * x) / np.log1p(mu)


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    mse = float(np.mean(d * d))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(data_range**2 / mse))


def mu_law_psnr(hdr_a: np.ndarray, hdr_b: np.ndarray, mu: float = 5000.0, normalize: bool = True) -> float:
    a = np.asarray(hdr_a, dtype=np.float64)
    b = np.asarray(hdr_b, dtype=np.float64)
    if normalize:
        s = float(np.percentile(b, 99))
        if s > 0:
            a, b = a / s, b / s
    return psnr(mu_law(a, mu), mu_law(b, mu), 1.0)


def apply_exposure(hdr: np.ndarray, exposure_time_s: float, gain: float = 1.0) -> np.ndarray:
    return hdr * exposure_time_s * gain


def _quantise(y: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(np.asarray(y, dtype=np.float64) * 255.0), 0, 255).astype(np.uint8)


def apply_crf(linear: np.ndarray, crf) -> np.ndarray:
    """Linear exposed values -> uint8 through the CRF (inverse of Debevec g, or a callable)."""
    if callable(crf):
        return _quantise(crf(np.asarray(linear)))
    g = np.asarray(crf["log_irradiance"], dtype=np.float64)
    if g.ndim == 1:
        g = np.tile(g, (3, 1))
    logx = np.log(np.clip(np.asarray(linear, dtype=np.float64), 1e-8, None))
    z = np.arange(g.shape[1], dtype=np.float64)
    out = np.empty(logx.shape, dtype=np.uint8)
    for c in range(3):
        gc = g[c]
        if not np.all(np.diff(gc) > 0):
            gc = np.maximum.accumulate(gc)
            # np.interp needs non-decreasing xp; tiny ramp breaks exact ties deterministically
            gc = gc + np.arange(gc.size) * 1e-12
        out[..., c] = np.clip(np.rint(np.interp(logx[..., c], gc, z)), 0, 255).astype(np.uint8)
    return out


def ldr_psnr_at_exposure(
    render_hdr: np.ndarray,
    photo_ldr_uint8: np.ndarray,
    exposure_time_s: float,
    crf,
    gain: float = 1.0,
    ignore_saturated: bool = True,
) -> tuple[float, float]:
    pred = apply_crf(apply_exposure(render_hdr, exposure_time_s, gain), crf)
    photo = np.asarray(photo_ldr_uint8)
    if ignore_saturated:
        mask = np.all((photo < 250) & (photo > 5), axis=-1)
    else:
        mask = np.ones(photo.shape[:2], dtype=bool)
    frac = float(mask.mean())
    if not mask.any():
        return float("nan"), 0.0
    p = pred[mask].astype(np.float64) / 255.0
    q = photo[mask].astype(np.float64) / 255.0
    return psnr(p, q, 1.0), frac


# ── EXR I/O ──────────────────────────────────────────────────────────────────
# Through the OpenEXR Python bindings (the opencv-python-headless 5.x wheel is built WITHOUT OpenEXR).
# Float32 RGB, scanline, ZIP; reads any compression, RGB/RGBA interleaved or planar R,G,B, half or float.


def read_exr(path: str) -> np.ndarray:
    import OpenEXR

    with OpenEXR.File(str(path)) as f:
        ch = f.channels()
        if "RGB" in ch or "RGBA" in ch:
            px = ch["RGB" if "RGB" in ch else "RGBA"].pixels[..., :3]
        elif all(k in ch for k in "RGB"):
            px = np.stack([ch[k].pixels for k in "RGB"], axis=-1)
        else:
            first = next(iter(ch.values())).pixels
            px = np.stack([first] * 3, axis=-1)
    return np.ascontiguousarray(px, dtype=np.float32)


def write_exr(path: str, rgb_float32: np.ndarray) -> None:
    import OpenEXR

    rgb = np.ascontiguousarray(np.asarray(rgb_float32, dtype=np.float32))
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"write_exr wants HxWx3, got {rgb.shape}")
    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    # The working-space declaration (docs/hdr.md "Colour management"): Rec.709 primaries, D65, `lin_srgb`.
    try:
        header["chromaticities"] = OpenEXR.Chromaticities((0.64, 0.33), (0.30, 0.60), (0.15, 0.06), (0.3127, 0.3290))
    except Exception:
        pass
    header["lichtfeld_colorspace"] = "lin_srgb"
    OpenEXR.File(header, {"RGB": rgb}).write(str(path))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mu-psnr")
    m.add_argument("a")
    m.add_argument("b", help="reference")
    m.add_argument("--mu", type=float, default=5000.0)
    m.add_argument("--no-normalize", action="store_true")
    ldr = sub.add_parser("ldr-psnr")
    ldr.add_argument("render")
    ldr.add_argument("photo")
    ldr.add_argument("--exposure", type=float, required=True)
    ldr.add_argument("--crf", required=True)
    ldr.add_argument("--gain", type=float, default=1.0)
    ldr.add_argument("--keep-saturated", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "mu-psnr":
        v = mu_law_psnr(read_exr(args.a), read_exr(args.b), args.mu, not args.no_normalize)
        print(f"mu_psnr={v:.4f}")
    else:
        with open(args.crf) as f:
            crf = json.load(f)
        photo = cv2.imread(args.photo, cv2.IMREAD_COLOR)
        if photo is None:
            raise FileNotFoundError(args.photo)
        photo = photo[..., ::-1]
        v, frac = ldr_psnr_at_exposure(
            read_exr(args.render), photo, args.exposure, crf, args.gain, not args.keep_saturated
        )
        print(f"ldr_psnr={v:.4f} fraction_evaluated={frac:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
