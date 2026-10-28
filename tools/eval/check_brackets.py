"""check_brackets.py — is `<out>/brackets/` (the plugin's bracket training set, docs/hdr-m3-design.md §1) consistent?

    cd tools && uv run python eval/check_brackets.py ../data/smoke_hdr [--positions 8]

Structure (its own parser — the writer is not asked to grade itself): every frame's pose equals its position's,
image count = Σ frames (and = the JPEGs in brackets/images), every points3D track names an image that exists and is
a frame 0, every frame's JPEG opens and carries ExposureTime in its EXIF, `crf.json` has `exif_ev_ref` =
log2(iso_ref / fnumber_ref²). Units: for up to `--positions` positions, every frame's JPEG against
`hdr_metrics.apply_crf(EXR · 2^(exif_ev − exif_ev_ref), crf)` — the median |Δ| (in 8-bit codes) on the unsaturated
pixels (5 < z < 250, all channels), printed per EV step. PASS when the structure holds and the median |Δ| ≤ 3 on
every EV step whose frames keep ≥ 20 % unsaturated pixels (the +4 EV frames of the room are ~93 % clipped: what is
left sits in the LUT's toe, where the recovered curve is least constrained — reported, not judged). Exit 1 on FAIL.
A wrong anchor (design §8 risk 1) is a 2^2.64 = 6.25× exposure error and shows as a median |Δ| of tens of codes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hdr_metrics import apply_crf, read_exr  # noqa: E402

TAG_EXIF_IFD = 0x8769
TAG_EXPOSURE_TIME = 0x829A


def _lines(p: Path) -> list[str]:
    return [ln.rstrip("\r\n") for ln in p.read_text().splitlines() if not ln.startswith("#")]


def _is_image_line(t: list[str]) -> bool:
    if len(t) < 10:
        return False
    try:
        int(t[0]), [float(x) for x in t[1:8]], int(t[8])
    except ValueError:
        return False
    name = " ".join(t[9:])
    return "." in name and name[name.rfind(".") + 1 :][:1].isalpha()


def read_images_txt(p: Path) -> list[tuple[int, list[str], int, str, str]]:
    """[(image_id, pose tokens, camera_id, name, points2D line)]"""
    lines = _lines(p)
    out = []
    i = 0
    while i < len(lines):
        t = lines[i].split()
        if not _is_image_line(t):
            i += 1
            continue
        rec = [int(t[0]), t[1:8], int(t[8]), " ".join(t[9:]), ""]
        i += 1
        if i < len(lines) and not _is_image_line(lines[i].split()):
            rec[4] = lines[i].strip()
            i += 1
        out.append(tuple(rec))
    return out


def check(out_dir: Path, n_positions: int = 8, delta_max: float = 3.0, min_coverage: float = 0.2) -> dict:
    root = out_dir / "brackets"
    problems: list[str] = []
    sparse = root / "sparse" / "0"
    images = read_images_txt(sparse / "images.txt")
    cameras = {int(ln.split()[0]): ln for ln in _lines(sparse / "cameras.txt") if ln.strip()}
    ids = {im[0] for im in images}
    if len(ids) != len(images):
        problems.append("duplicate image ids in images.txt")
    by_pos: dict[str, list] = defaultdict(list)
    for im in images:
        by_pos[im[3].rpartition("_")[0]].append(im)
    for pos, frames in by_pos.items():
        poses = {" ".join(f[1]) for f in frames}
        if len(poses) != 1:
            problems.append(f"{pos}: {len(poses)} different poses among its frames")
        for f in frames:  # a shifted frame may carry its own camera (design §8 risk 7); it must exist
            if f[2] not in cameras:
                problems.append(f"{f[3]}: camera {f[2]} is not in cameras.txt")
        ks = sorted(int(f[3].rpartition("_f")[2].split(".")[0]) for f in frames)
        if ks != list(range(len(frames))):
            problems.append(f"{pos}: frame indices {ks}")
    idx = json.loads((root / "exposures.json").read_text())
    stride = int(idx.get("brackets", {}).get("id_stride", 10))
    jpgs = sorted(p.name for p in (root / "images").iterdir() if p.suffix.lower() == ".jpg")
    if len(jpgs) != len(images):
        problems.append(f"{len(jpgs)} JPEGs in images/ but {len(images)} records in images.txt")
    expected = sum(len(p["sources"]) for p in idx["positions"])
    if expected != len(images):
        problems.append(f"exposures.json lists {expected} frames, images.txt {len(images)}")
    n_tracks = n_track_elems = 0
    for ln in _lines(sparse / "points3D.txt"):
        t = ln.split()
        if len(t) < 8:
            continue
        n_tracks += 1
        for j in range(8, len(t) - 1, 2):
            n_track_elems += 1
            iid = int(t[j])
            if iid not in ids:
                problems.append(f"point {t[0]}: track names image {iid}, not in images.txt")
                break
            if iid % stride != 0:
                problems.append(f"point {t[0]}: track image {iid} is not a frame 0 (stride {stride})")
                break
    crf = json.loads((root / "crf.json").read_text())
    for key in ("exif_ev_ref", "iso_ref", "fnumber_ref", "units", "log_irradiance"):
        if key not in crf:
            problems.append(f"crf.json lacks {key}")
    if {"exif_ev_ref", "iso_ref", "fnumber_ref"} <= set(crf) and crf["iso_ref"] and crf["fnumber_ref"]:
        want = math.log2(crf["iso_ref"] / crf["fnumber_ref"] ** 2)
        if abs(want - crf["exif_ev_ref"]) > 1e-9:
            problems.append(f"exif_ev_ref {crf['exif_ev_ref']} ≠ log2(iso_ref / fnumber_ref²) = {want}")
    # EXIF through the link
    for name in jpgs[:: max(1, len(jpgs) // 10)]:
        with Image.open(root / "images" / name) as im:
            t = dict(im.getexif().get_ifd(TAG_EXIF_IFD)).get(TAG_EXPOSURE_TIME)
        if not t:
            problems.append(f"{name}: no ExposureTime in EXIF")
    # ── units ──
    ref = crf.get("exif_ev_ref", 0.0)
    per_ev: dict[float, list[tuple[float, float]]] = defaultdict(list)
    positions = idx["positions"]
    step = max(1, len(positions) // max(1, n_positions))
    for p in positions[::step][:n_positions]:
        hdr = read_exr(str(root / p["hdr"]))
        for s in p["sources"]:
            jpg = np.asarray(Image.open(root / "images" / s["frame"]).convert("RGB")).astype(int)
            mask = np.all((jpg > 5) & (jpg < 250), axis=2)
            if not mask.any():
                per_ev[round(s["ev_step"], 2)].append((float("nan"), 0.0))
                continue
            pred = apply_crf(hdr * 2.0 ** (s["exif_ev"] - ref), crf).astype(int)
            d = np.abs(pred - jpg)[mask]
            per_ev[round(s["ev_step"], 2)].append((float(np.median(d)), float(mask.mean())))
    rows = []
    for ev in sorted(per_ev):
        meds = [m for m, _ in per_ev[ev] if not math.isnan(m)]
        cov = [c for _, c in per_ev[ev]]
        med = float(np.median(meds)) if meds else float("nan")
        coverage = float(np.mean(cov)) if cov else 0.0
        judged = coverage >= min_coverage
        rows.append(
            {"ev_step": ev, "median_abs_delta": med, "coverage": coverage, "frames": len(per_ev[ev]), "judged": judged}
        )
        if judged and not (med <= delta_max):
            problems.append(f"EV {ev:+.1f}: median |Δ| {med:.1f} > {delta_max} on {coverage:.0%} of the pixels")
    return {
        "positions": len(by_pos),
        "frames": len(images),
        "jpegs": len(jpgs),
        "cameras": len(cameras),
        "points3D": n_tracks,
        "track_elements": n_track_elems,
        "id_stride": stride,
        "exif_ev_ref": ref,
        "iso_ref": crf.get("iso_ref"),
        "fnumber_ref": crf.get("fnumber_ref"),
        "omitted": idx.get("brackets", {}).get("omitted", []),
        "roundtrip": rows,
        "problems": problems,
        "pass": not problems,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path, help="the plugin's output dataset (holds brackets/)")
    ap.add_argument("--positions", type=int, default=8, help="positions to round-trip (spread over the set)")
    ap.add_argument("--delta-max", type=float, default=3.0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    r = check(a.out, a.positions, a.delta_max)
    if a.json:
        print(json.dumps(r, indent=1))
    else:
        print(
            f"brackets/: {r['positions']} positions, {r['frames']} frames ({r['jpegs']} JPEGs), {r['cameras']} camera(s), "
            f"{r['points3D']} points / {r['track_elements']} track elements, stride {r['id_stride']}, "
            f"exif_ev_ref {r['exif_ev_ref']:.4f} (ISO {r['iso_ref']}, f/{r['fnumber_ref']}), omitted {r['omitted']}"
        )
        print("LUT round-trip, EXR x 2^(exif_ev - exif_ev_ref) through crf.json vs the JPEG (unsaturated pixels):")
        for row in r["roundtrip"]:
            print(
                f"  EV {row['ev_step']:+5.1f}: median |d| = {row['median_abs_delta']:4.1f} /255 on {row['coverage']:6.1%} "
                f"of the pixels ({row['frames']} frames){'' if row['judged'] else '  [not judged: < 20 % coverage]'}"
            )
        for pb in r["problems"]:
            print("  PROBLEM:", pb)
        print("PASS" if r["pass"] else "FAIL")
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
