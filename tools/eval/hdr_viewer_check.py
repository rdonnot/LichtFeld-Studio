"""tools/eval/hdr_viewer_check.py — M4: the viewer's HDR display against the training rasterizer.

The viewer (Vulkan) shades an HDR model as exp(SH) / s and displays radiance * 2^EV -> tone -> sRGB OETF -> 8 bit.
The training rasterizer (fastgs, color_activation_for(model)) wrote the same views as LINEAR EXR at `--eval`
(out/<run>/eval_step_N/<stem>.exr). This script applies the display chain to the EXR on the CPU and compares
it with the viewer's 8-bit render of the same dataset camera: PSNR per EV, plus the PNGs of both.

Usage (from tools/):
    uv run python eval/hdr_viewer_check.py --viewer-dir <dir of <stem>_ev±N.ppm> --exr-dir <eval_step_N> \
        --scale 2421.4 --stems pos_008 --evs -4 -2 0 2 4 [--tone none] [--out <png dir>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import OpenEXR
from PIL import Image


def read_exr(path: Path) -> np.ndarray:
    f = OpenEXR.File(str(path))
    ch = f.channels()
    if "RGB" in ch or "RGBA" in ch:  # OpenEXR 3.3 groups interleaved channels
        arr = np.asarray(ch["RGB" if "RGB" in ch else "RGBA"].pixels, dtype=np.float64)
        return arr[..., :3]
    if all(k in ch for k in ("R", "G", "B")):  # separate planes
        return np.stack([np.asarray(ch[k].pixels, dtype=np.float64) for k in ("R", "G", "B")], axis=-1)
    raise SystemExit(f"{path}: channels {list(ch.keys())} are not RGB")


def read_ppm(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)


def srgb_oetf(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def tone(x: np.ndarray, mode: str) -> np.ndarray:
    if mode in ("none", "linear"):
        return x
    if mode == "aces":
        return (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
    if mode == "filmic":
        def f(v):
            return ((v * (0.15 * v + 0.05) + 0.004) / (v * (0.15 * v + 0.50) + 0.06)) - 0.02 / 0.30
        return f(x) / f(11.2)
    raise SystemExit(f"tone {mode} not implemented on the CPU side")


def display_chain(radiance: np.ndarray, scale: float, ev: float, mode: str) -> np.ndarray:
    lin = radiance / scale * (2.0 ** ev)
    return np.round(srgb_oetf(tone(lin, mode)) * 255.0)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a / 255.0 - b / 255.0) ** 2)
    return float("inf") if mse == 0 else float(10.0 * np.log10(1.0 / mse))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer-dir", required=True, type=Path)
    ap.add_argument("--exr-dir", required=True, type=Path)
    ap.add_argument("--scale", required=True, type=float)
    ap.add_argument("--stems", nargs="+", required=True)
    ap.add_argument("--evs", nargs="+", type=float, required=True)
    ap.add_argument("--tone", default="none")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    rows = []
    for stem in args.stems:
        radiance = read_exr(args.exr_dir / f"{stem}.exr")
        for ev in args.evs:
            viewer_path = args.viewer_dir / f"{stem}_ev{ev:+.1f}.ppm"
            viewer = read_ppm(viewer_path)
            ref = display_chain(radiance, args.scale, ev, args.tone)
            if viewer.shape != ref.shape:
                raise SystemExit(f"{viewer_path}: viewer {viewer.shape} vs reference {ref.shape}")
            row = dict(stem=stem, ev=ev, psnr=psnr(viewer, ref),
                       mean_abs=float(np.mean(np.abs(viewer - ref))),
                       viewer_mean=float(viewer.mean()), ref_mean=float(ref.mean()),
                       viewer_sat=float((viewer.min(-1) == 255).mean()), ref_sat=float((ref.min(-1) == 255).mean()),
                       viewer_black=float((viewer.max(-1) == 0).mean()))
            rows.append(row)
            print(json.dumps(row))
            if args.out:
                args.out.mkdir(parents=True, exist_ok=True)
                Image.fromarray(viewer.astype(np.uint8)).save(args.out / f"{stem}_ev{ev:+.1f}_viewer.png")
                Image.fromarray(ref.astype(np.uint8)).save(args.out / f"{stem}_ev{ev:+.1f}_fastgs.png")
    worst = min(r["psnr"] for r in rows)
    print(f"worst PSNR {worst:.2f} dB over {len(rows)} renders")
    return 0 if worst > 30.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
