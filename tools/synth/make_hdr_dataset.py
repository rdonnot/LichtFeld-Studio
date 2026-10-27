"""tools/synth/make_hdr_dataset.py — a ground-truth HDR training set for the M2 smoke test.

The bracket plugin (M1) is what will normally produce `data/smoke_hdr`; until it exists, this lays the SYNTHETIC
truth out the way the loader expects a COLMAP dataset:

    <out>/images_hdr/view_NNN.exr     linear radiance (hard-linked, else copied, from the synthetic GT)
    <out>/images/view_NNN.png         an 8-bit ACES proxy of the same frame (for a proxy-LDR baseline / browsing)
    <out>/sparse/0/cameras.txt        the exact poses of make_scene.py, image names changed to the EXRs
    <out>/sparse/0/images.txt
    <out>/sparse/0/points3D.txt       header only: train with --random

Usage (from the repo root, the tools env):
    cd tools && uv run python synth/make_hdr_dataset.py --gt ../data/brackets_synth/gt --out ../data/smoke_hdr_gt
    ./build/LichtFeld-Studio --headless --train -d data/smoke_hdr_gt --images images_hdr --random \\
        -o out/m2_hdr --iter 3000 --test-every 8 --export ply --hdr
The proxy PNGs are named like the EXRs so `--images images` trains the same poses on the LDR proxy.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from eval.hdr_metrics import read_exr  # noqa: E402


def aces_proxy(rgb: np.ndarray, median_target: float = 0.18) -> np.ndarray:
    """Linear radiance -> 8-bit sRGB-ish ACES tone map, exposed so the scene median lands at median_target."""
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    median = float(np.median(lum[np.isfinite(lum) & (lum > 0)])) if np.any(lum > 0) else 1.0
    x = np.clip(rgb * (median_target / max(median, 1e-9)), 0.0, None)
    tone = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
    tone = np.clip(tone, 0.0, 1.0)
    srgb = np.where(tone <= 0.0031308, 12.92 * tone, 1.055 * np.power(tone, 1 / 2.4) - 0.055)
    return (np.clip(srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def rewrite_images_txt(src: Path, dst: Path, name_map: dict[str, str]) -> int:
    """images.txt: two lines per image; the NAME is the last field of the first line."""
    out_lines: list[str] = []
    renamed = 0
    for line in src.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            out_lines.append(line)
            continue
        fields = line.split()
        if len(fields) >= 10 and fields[-1] in name_map:
            fields[-1] = name_map[fields[-1]]
            renamed += 1
            out_lines.append(" ".join(fields))
        else:
            out_lines.append(line)
    dst.write_text("\n".join(out_lines) + "\n")
    return renamed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, help="data/brackets_synth/gt (view_NNN.exr + sparse_gt/)")
    ap.add_argument("--out", required=True, help="dataset folder to create")
    ap.add_argument("--sparse", default=None, help="COLMAP text model folder (default: <gt>/sparse_gt)")
    args = ap.parse_args(argv)

    gt = Path(args.gt)
    out = Path(args.out)
    sparse_src = Path(args.sparse) if args.sparse else gt / "sparse_gt"
    exrs = sorted(gt.glob("view_*.exr"))
    if not exrs:
        print(f"no view_*.exr in {gt}", file=sys.stderr)
        return 1
    for sub in ("images_hdr", "images", "sparse/0"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    from PIL import Image

    name_map: dict[str, str] = {}
    max_radiance = 0.0
    for exr in exrs:
        link_or_copy(exr, out / "images_hdr" / exr.name)
        rgb = read_exr(str(exr))
        max_radiance = max(max_radiance, float(np.nanmax(rgb)))
        Image.fromarray(aces_proxy(rgb), "RGB").save(out / "images" / (exr.stem + ".png"))
        for ext in (".jpg", ".jpeg", ".png", ".exr"):
            name_map[exr.stem + ext] = exr.stem + ".exr"

    shutil.copy2(sparse_src / "cameras.txt", out / "sparse" / "0" / "cameras.txt")
    renamed = rewrite_images_txt(sparse_src / "images.txt", out / "sparse" / "0" / "images.txt", name_map)
    (out / "sparse" / "0" / "points3D.txt").write_text(
        "# 3D point list with one line of data per point:\n"
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        "# Number of points: 0, mean track length: 0\n"
    )
    if renamed != len(exrs):
        print(f"warning: {renamed} image names rewritten for {len(exrs)} EXRs — check {sparse_src / 'images.txt'}",
              file=sys.stderr)
    print(f"{out}: {len(exrs)} views, max radiance {max_radiance:.3f}, {renamed} poses; "
          f"train with --images images_hdr --random --hdr")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
