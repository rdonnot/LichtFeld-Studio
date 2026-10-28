"""tools/synth/make_brackets.py — exposure brackets with EXIF from the synthetic ground-truth EXRs.

Turns `data/brackets_synth/gt/view_NNN.exr` (linear scene radiance from make_scene.py) into what a camera on a
tripod would have shot: for each view, one JPEG per EV step through a KNOWN camera model

    pixel = CRF( clip( L * t * gain ) ) + noise, quantised to 8 bit

with EXIF ExposureTime / FNumber / ISOSpeedRatings / DateTimeOriginal, so the bracket plugin (M1) can be checked
against the truth: the recovered CRF against `crf.json`, the merged EXR against the EXR it came from, the grouping
against `exposures_gt.json`. LichtFeld seeds PPISP from the same three tags (src/core/exif.cpp: ev = log2(t / N² · ISO)).

Usage (from the repo root, the tools env):
    cd tools && uv run python synth/make_brackets.py --gt ../data/brackets_synth/gt --out ../data/brackets_raw \
        [--evs -2 0 2] [--base-time 0.01] [--iso 100] [--fnumber 4] [--seed 7] [--quality 95]

Output layout (what a camera dumps): `<out>/IMG_0001.jpg …` in shooting order (view 0 at EV −2, 0, +2, then view 1 …),
1 s apart inside a bracket and 20 s between brackets; `<out>/exposures_gt.json` (file → view, ev, exposure_time, iso,
fnumber, timestamp); `<out>/crf.json` (Debevec form: g[z] = ln E for z in 0..255, per channel, plus the closed form);
`<out>/gain.json` (the radiance → sensor scale, chosen so the 0 EV frame puts the scene median at 0.18).

Camera model: the CRF is a smooth toe/shoulder curve, deliberately NOT sRGB, so a recovery that only works for
sRGB-shaped curves is caught:  f(x) = (1 − exp(−c · x^g)) / (1 − exp(−c)), c = 3, g = 0.75 on x ∈ [0, 1].
Noise: Poisson shot noise at a full well of 20 000 e⁻ plus Gaussian read noise σ = 0.002 (in [0,1] units), before the
CRF, seeded per image.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.TiffImagePlugin import IFDRational

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from hdr_metrics import read_exr  # noqa: E402  (tools/eval: the one EXR reader)

# ── Camera model ──────────────────────────────────────────────────────────────────────────────────────────────────────

CRF_C = 3.0
CRF_G = 0.75
FULL_WELL = 20000.0
READ_NOISE = 0.002

TAG_EXPOSURE_TIME = 0x829A
TAG_FNUMBER = 0x829D
TAG_ISO = 0x8827
TAG_DATETIME_ORIGINAL = 0x9003
TAG_EXIF_IFD = 0x8769
TAG_MAKE = 0x010F
TAG_MODEL = 0x0110
TAG_DATETIME = 0x0132


def crf_forward(x: np.ndarray) -> np.ndarray:
    """Sensor-linear [0,1] → encoded [0,1]; monotonic, f(0)=0, f(1)=1."""
    x = np.clip(x, 0.0, 1.0)
    return (1.0 - np.exp(-CRF_C * np.power(x, CRF_G))) / (1.0 - math.exp(-CRF_C))


def crf_inverse(y: np.ndarray) -> np.ndarray:
    """Encoded [0,1] → sensor-linear [0,1] (closed form)."""
    y = np.clip(y, 0.0, 1.0)
    inner = 1.0 - y * (1.0 - math.exp(-CRF_C))
    return np.power(-np.log(np.maximum(inner, 1e-12)) / CRF_C, 1.0 / CRF_G)


def crf_debevec() -> list[list[float]]:
    """g[z] = ln(E) for z = 0..255, one list per channel (identical here: a grey CRF)."""
    z = np.arange(256, dtype=np.float64) / 255.0
    lin = crf_inverse(z)
    lin[0] = max(lin[0], 1e-6)
    g = np.log(lin).tolist()
    return [g, list(g), list(g)]


def expose(linear: np.ndarray, t: float, gain: float, rng: np.random.Generator) -> np.ndarray:
    """Radiance → noisy sensor-linear [0,1] (shot + read noise), float32."""
    s = np.clip(linear * (t * gain), 0.0, None).astype(np.float64)
    electrons = rng.poisson(np.clip(s, 0.0, 4.0) * FULL_WELL).astype(np.float64)
    s = electrons / FULL_WELL + rng.normal(0.0, READ_NOISE, size=s.shape)
    return np.clip(s, 0.0, 1.0).astype(np.float32)


def to_uint8(encoded: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(encoded * 255.0), 0, 255).astype(np.uint8)


# ── EXIF ──────────────────────────────────────────────────────────────────────────────────────────────────────────────


def _rational(x: float, denom: int = 1_000_000) -> IFDRational:
    """An EXIF RATIONAL (type 5). A plain (num, den) tuple makes Pillow write two LONGs/SHORTs (count 2) instead:
    Pillow reads that back, LichtFeld's `core/exif.cpp` does not — it took every ExposureTime as 1 s and seeded
    all 200 frames of the room at log2(ISO/N²) = 2.64 EV (found 2026-10-28 by the M3 bracket set's load check)."""
    return IFDRational(int(round(x * denom)), denom)


def save_jpeg_with_exif(path: Path, rgb: np.ndarray, t: float, fnumber: float, iso: int, when: datetime,
                        quality: int) -> None:
    im = Image.fromarray(rgb, "RGB")
    exif = Image.Exif()
    exif[TAG_MAKE] = "N'Brain"
    exif[TAG_MODEL] = "Synthetic bracket camera"
    exif[TAG_DATETIME] = when.strftime("%Y:%m:%d %H:%M:%S")
    ifd = exif.get_ifd(TAG_EXIF_IFD)
    ifd[TAG_EXPOSURE_TIME] = _rational(t)
    ifd[TAG_FNUMBER] = _rational(fnumber, 100)
    ifd[TAG_ISO] = int(iso)
    ifd[TAG_DATETIME_ORIGINAL] = when.strftime("%Y:%m:%d %H:%M:%S")
    im.save(path, "JPEG", quality=quality, subsampling=0, exif=exif.tobytes())


def exif_float(v) -> float:
    """A rational EXIF value as Pillow hands it back: IFDRational, (num, den) or a plain number."""
    if isinstance(v, tuple):
        return v[0] / v[1]
    return float(v)


def read_exif_exposure(path: Path) -> tuple[float, float, int, str]:
    with Image.open(path) as im:
        exif = im.getexif()
        ifd = exif.get_ifd(TAG_EXIF_IFD)
        t = exif_float(ifd[TAG_EXPOSURE_TIME])
        n = exif_float(ifd[TAG_FNUMBER])
        iso = int(ifd[TAG_ISO])
        dt = str(ifd[TAG_DATETIME_ORIGINAL])
    return t, n, iso, dt


# ── Main ──────────────────────────────────────────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gt", type=Path, required=True, help="folder with view_NNN.exr from make_scene.py")
    ap.add_argument("--out", type=Path, required=True, help="output folder (data/brackets_raw)")
    ap.add_argument("--evs", type=float, nargs="+", default=[-4.0, -2.0, 0.0, 2.0, 4.0])
    ap.add_argument("--base-time", type=float, default=0.01, help="exposure time at EV 0, seconds")
    ap.add_argument("--iso", type=int, default=100)
    ap.add_argument("--fnumber", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--start", default="2026:10:27 10:00:00", help="DateTimeOriginal of the first frame")
    args = ap.parse_args(argv)

    exrs = sorted(args.gt.glob("view_*.exr"))
    exrs = [p for p in exrs if "_preview" not in p.name]
    if not exrs:
        raise SystemExit(f"no view_*.exr in {args.gt}")
    args.out.mkdir(parents=True, exist_ok=True)

    # One gain for the whole set, from the first view: the 0 EV frame puts the scene median at 0.18.
    first = read_exr(exrs[0])
    median = float(np.median(first))
    gain = 0.18 / max(median * args.base_time, 1e-12)
    print(f"gain {gain:.4g}  (view_000 median radiance {median:.4g}, base time {args.base_time} s)")

    when = datetime.strptime(args.start, "%Y:%m:%d %H:%M:%S")
    records: list[dict] = []
    clip_stats: dict[float, list[float]] = {ev: [] for ev in args.evs}
    n = 0
    for vi, exr in enumerate(exrs):
        linear = read_exr(exr)
        for ev in args.evs:
            t = args.base_time * (2.0 ** ev)
            rng = np.random.default_rng([args.seed, vi, int(round(ev * 100)) + 10_000])
            sensor = expose(linear, t, gain, rng)
            rgb8 = to_uint8(crf_forward(sensor))
            n += 1
            name = f"IMG_{n:04d}.jpg"
            save_jpeg_with_exif(args.out / name, rgb8, t, args.fnumber, args.iso, when, args.quality)
            clip_stats[ev].append(float(np.mean(np.any(rgb8 >= 250, axis=2))))
            records.append({
                "file": name,
                "view": exr.stem,
                "view_index": vi,
                "ev": ev,
                "exposure_time": t,
                "iso": args.iso,
                "fnumber": args.fnumber,
                "exif_ev": math.log2(t / (args.fnumber ** 2) * args.iso),
                "timestamp": when.strftime("%Y:%m:%d %H:%M:%S"),
            })
            when += timedelta(seconds=1)
        when += timedelta(seconds=19)
        if vi % 10 == 0:
            print(f"  view {vi + 1}/{len(exrs)}")

    # Read the EXIF back: the plugin must see exactly what we wrote.
    for rec in records[:3] + records[-1:]:
        t, fno, iso, dt = read_exif_exposure(args.out / rec["file"])
        if abs(t - rec["exposure_time"]) > 1e-7 or abs(fno - rec["fnumber"]) > 1e-3 or iso != rec["iso"] \
                or dt != rec["timestamp"]:
            raise SystemExit(f"EXIF round-trip failed for {rec['file']}: {t} {fno} {iso} {dt} vs {rec}")

    (args.out / "exposures_gt.json").write_text(json.dumps({
        "source": str(args.gt),
        "gain": gain,
        "base_time": args.base_time,
        "evs": args.evs,
        "images": records,
    }, indent=1))
    (args.out / "crf.json").write_text(json.dumps({
        "domain": "ldr8",
        "formula": f"f(x) = (1 - exp(-{CRF_C} * x^{CRF_G})) / (1 - exp(-{CRF_C})), x = clip(L * t * gain, 0, 1)",
        "log_irradiance": crf_debevec(),
    }))
    (args.out / "gain.json").write_text(json.dumps({"gain": gain, "median_radiance_view_000": median}))

    print(f"wrote {n} JPEGs to {args.out}")
    for ev in args.evs:
        print(f"  EV {ev:+.0f}: mean clipped fraction {np.mean(clip_stats[ev]):.3%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
