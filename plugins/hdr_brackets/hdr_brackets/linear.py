"""Merge brackets of LINEAR frames (camera raws already developed to scene-linear EXR) into the M2 dataset.

The JPEG path (pipeline.py) has to recover a camera response first; a raw has none — its developed values are
already proportional to the light. So a position's radiance is a weighted mean of each frame over its exposure:

    E = Σ w_i · x_i / t_i  /  Σ w_i        t_i = exposure_time · 2^-nd_stops (an ND frame saw less light)
    w_i = t_i · h(x_i / white_i)            t: photon noise falls with exposure; h: 1 up to 80 % of white, 0 at 95 %

A pixel clipped in every frame keeps the shortest frame's x / t (a lower bound) and is counted as `saturated`.

Input: a capture folder written by N'Brain's camera tether (`tether_set.json`, "kind": "scan" or "bracket") — it
names every frame's file, exposure time and ND, so nothing is grouped from EXIF — and the developed EXRs, one per raw
at `<exr_dir>/<frame file with .exr>` (nd/ kept). The EXRs' colourspace is their `lichtfeld_colorspace` /
`colorspace` string attribute, else `input_space` (default ACEScg). It is carried to the merged EXRs unchanged: the
trainer converts a foreign linear space to its working space on load (OCIO 3×3, image_exr.cpp). Each frame's clip
level is its float attribute `white` when the developer wrote one, else measured: the image maximum, when at least
0.05 % of the pixels (and 16 or more) sit within 2 % of it (a plateau = the sensor clipped), else no clip.

Output (`<out>/`, the layout M2 trains from with `--images images_hdr --hdr`):
    images_hdr/pos_NNN.exr   merged radiance, float32 RGB, the input's colourspace tag
    images/pos_NNN.png       8-bit proxy for SfM (Rec.709 for display: an AP1 input goes through its 3×3 first)
    exposures.json           version 2 + "linear": true, "crf": null, "colorspace"
    sparse/0/                COLMAP model when SfM runs

    uv run python -m hdr_brackets.linear <capture_dir> <out> [--exr-dir D] [--max-side 3200] [--no-sfm]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import proxy as px
from .dataset import HDR_DIR, INDEX_FILE, INDEX_VERSION, PROXY_DIR, position_name

log = logging.getLogger(__name__)

SET_FILE = "tether_set.json"
EXR_DIR = "exr"
HAT_LO, HAT_HI = 0.80, 0.95
PLATEAU_FRACTION, PLATEAU_RTOL, PLATEAU_MIN = 5e-4, 0.02, 16

# ACEScg (AP1, D60) → linear Rec.709 (D65), Bradford — only for the proxy's display; the merged EXR keeps AP1.
AP1_TO_REC709 = np.array([[1.70505, -0.62179, -0.08326], [-0.13026, 1.14080, -0.01055], [-0.02400, -0.12897, 1.15297]])
AP1_NAMES = {"acescg", "aces - acescg", "lin_ap1", "linear ap1", "acescg (ap1)"}
REC709_NAMES = {"lin_srgb", "linear rec.709 (srgb)", "lin_rec709", "linear srgb", "scene-linear rec.709-srgb"}


# ── reading the capture ──────────────────────────────────────────────────────


@dataclass
class Frame:
    file: str           # relative to the capture folder, as tether_set.json lists it
    exr: Path
    exposure_time: float
    nd_stops: float = 0.0

    @property
    def t_eff(self) -> float:
        return self.exposure_time * 2.0 ** -self.nd_stops


@dataclass
class Position:
    index: int
    frames: list[Frame] = field(default_factory=list)


def read_capture(capture: Path, exr_dir: Path | None = None) -> tuple[dict, list[Position], list[str]]:
    """tether_set.json → positions with their EXRs; a frame with no file or no EXR is skipped and named."""
    rec = json.loads((capture / SET_FILE).read_text())
    exr_root = exr_dir or capture / EXR_DIR
    out, missing = [], []
    for p in rec.get("positions", []):
        pos = Position(int(p.get("index", len(out))))
        for f in p.get("frames", []):
            rel = str(f.get("file") or "")
            t = float(f.get("exposure_time") or 0)
            if not rel or t <= 0:
                missing.append(f"position {pos.index}: a frame with no file / exposure (card {f.get('card')})")
                continue
            exr = exr_root / Path(rel).with_suffix(".exr")
            if not exr.is_file():
                missing.append(f"{rel}: no developed EXR at {exr}")
                continue
            pos.frames.append(Frame(rel, exr, t, float(f.get("nd_stops") or 0)))
        if pos.frames:
            out.append(pos)
    return rec, out, missing


# ── EXR ──────────────────────────────────────────────────────────────────────


def read_frame(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    import OpenEXR

    with OpenEXR.File(str(path)) as f:
        header = dict(f.header())
        ch = f.channels()
        if "RGB" in ch or "RGBA" in ch:
            rgb = ch["RGB" if "RGB" in ch else "RGBA"].pixels[..., :3]
        else:
            rgb = np.stack([ch[k].pixels for k in ("R", "G", "B")], axis=-1)
    return np.ascontiguousarray(rgb, dtype=np.float32), header


def write_frame(path: Path, rgb: np.ndarray, colorspace: str) -> None:
    import OpenEXR

    path.parent.mkdir(parents=True, exist_ok=True)
    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage,
              "lichtfeld_colorspace": colorspace}
    OpenEXR.File(header, {"RGB": np.ascontiguousarray(rgb, dtype=np.float32)}).write(str(path))


# The standard `chromaticities` attribute, for a writer that names no colourspace (rawsmith): primaries → OCIO name.
_PRIMARIES = {
    "ACEScg": (0.713, 0.293, 0.165, 0.830, 0.128, 0.044),
    "lin_srgb": (0.64, 0.33, 0.30, 0.60, 0.15, 0.06),
    "Linear Rec.2020": (0.708, 0.292, 0.170, 0.797, 0.131, 0.046),
}


def header_colorspace(header: dict[str, Any]) -> str | None:
    """The EXR's declared colourspace: a string attribute, else its chromaticities' primaries (linear assumed —
    the developer's contract), else None."""
    for key in ("lichtfeld_colorspace", "colorspace", "oiio:ColorSpace"):
        v = header.get(key)
        if isinstance(v, bytes):
            v = v.decode(errors="replace")
        if isinstance(v, str) and v.strip():
            return v.strip()
    ch = header.get("chromaticities")
    if ch is not None:
        try:
            vals = [float(x) for x in (ch if isinstance(ch, (tuple, list)) else tuple(ch))][:6]
        except (TypeError, ValueError):
            vals = []
        for name, ref in _PRIMARIES.items():
            if len(vals) == 6 and all(abs(a - b) < 0.003 for a, b in zip(vals, ref, strict=True)):
                return name
    return None


def clip_level(rgb: np.ndarray, header: dict[str, Any]) -> float | None:
    """The value at which this frame clipped (see the module docstring), None when nothing clipped."""
    w = header.get("white")
    if isinstance(w, (int, float)) and w > 0:
        return float(w)
    sw = header.get("software")
    if isinstance(sw, bytes):
        sw = sw.decode(errors="replace")
    if isinstance(sw, str) and sw.startswith("rawsmith"):
        return 1.0                      # rawsmith's convention: 1.0 == raw white, clipped (its docs/ARCHITECTURE.md)
    peak = rgb.max(axis=-1)
    top = float(peak.max()) if peak.size else 0.0
    if top <= 0:
        return None
    near = np.count_nonzero(peak >= top * (1 - PLATEAU_RTOL))
    return top if near >= max(PLATEAU_FRACTION * peak.size, PLATEAU_MIN) else None


def resize(rgb: np.ndarray, max_side: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    if not max_side or max(h, w) <= max_side:
        return rgb
    import cv2

    s = max_side / max(h, w)
    return cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


# ── the merge ────────────────────────────────────────────────────────────────


def hat(r: np.ndarray) -> np.ndarray:
    """1 below HAT_LO of white, a ramp to 0 at HAT_HI."""
    return np.clip((HAT_HI - r) / (HAT_HI - HAT_LO), 0.0, 1.0)


def merge_frames(frames: list[tuple[np.ndarray, float, float | None]]) -> tuple[np.ndarray, int]:
    """[(rgb, t_eff, white)] of one position → (radiance, saturated pixel count). Streams: the caller may pass a
    generator, only the running sums stay in memory."""
    num = den = fallback = None
    best_t = None
    for rgb, t, white in frames:
        peak = rgb.max(axis=-1)
        w = (t * hat(peak / white) if white else np.full(peak.shape, t, np.float32)).astype(np.float32)
        if num is None:
            num = np.zeros_like(rgb)
            den = np.zeros(peak.shape, np.float32)
        num += rgb * (w / t)[..., None]
        den += w
        if best_t is None or t < best_t:       # the shortest frame: what a pixel clipped everywhere falls back on
            best_t, fallback = t, rgb / t
    if num is None:
        raise ValueError("no frames")
    ok = den > 0
    out = np.where(ok[..., None], num / np.maximum(den, 1e-30)[..., None], fallback)
    return np.maximum(out, 0.0).astype(np.float32), int(np.count_nonzero(~ok))


def to_display(rgb: np.ndarray, colorspace: str) -> np.ndarray:
    """The merged radiance in linear Rec.709 for the proxy (AP1 through its 3×3, Rec.709 as is, others as is)."""
    if colorspace.lower() in AP1_NAMES:
        return np.maximum(rgb @ AP1_TO_REC709.T.astype(np.float32), 0.0)
    return rgb


# ── the run ──────────────────────────────────────────────────────────────────


def run(capture: str | os.PathLike, out: str | os.PathLike, *, exr_dir: str | os.PathLike | None = None,
        input_space: str = "ACEScg", max_side: int = 0, proxy_operator: str = "aces", proxy_ev: float = 0.0,
        run_sfm: bool = True, camera_model: str = "SIMPLE_RADIAL",
        progress: Callable[[str, float], None] | None = None) -> dict[str, Any]:
    """The whole dataset; returns the report (also written to `<out>/linear_report.json`)."""
    t0 = time.time()
    capture, out = Path(capture), Path(out)
    tick = progress or (lambda _s, _f: None)
    rec, positions, warnings = read_capture(capture, Path(exr_dir) if exr_dir else None)
    if not positions:
        raise FileNotFoundError(f"{capture}: no position has a developed EXR ({'; '.join(warnings[:3])})")
    (out / HDR_DIR).mkdir(parents=True, exist_ok=True)
    (out / PROXY_DIR).mkdir(parents=True, exist_ok=True)
    space: str | None = None
    entries: list[dict[str, Any]] = []
    for k, pos in enumerate(positions):
        tick("merge", k / len(positions))
        whites: list[float | None] = []

        def frames(pos=pos, whites=whites):
            nonlocal space
            for f in pos.frames:
                rgb, header = read_frame(f.exr)
                cs = header_colorspace(header) or input_space
                if space is None:
                    space = cs
                elif cs != space:
                    raise ValueError(f"{f.exr}: colourspace {cs!r}, the others are {space!r}")
                white = clip_level(rgb, header)
                whites.append(white)
                yield resize(rgb, max_side), f.t_eff, white

        hdr, saturated = merge_frames(frames())
        name = position_name(k)
        write_frame(out / HDR_DIR / f"{name}.exr", hdr, space or input_space)
        y = px.luminance(to_display(hdr, space or input_space))
        entries.append({
            "name": name, "hdr": f"{HDR_DIR}/{name}.exr", "proxy": f"{PROXY_DIR}/{name}.png", "bracket_id": pos.index,
            "ghost_score": 0.0, "flagged": False, "saturated": saturated,
            "key": px.proxy_exposure_value(to_display(hdr, space or input_space), proxy_ev),
            "stats": {"min": float(y.min()), "p50": float(np.median(y)), "max": float(y.max())},
            "sources": [{"file": f.file, "exposure_time": f.exposure_time, "nd_stops": f.nd_stops,
                         "t_eff": f.t_eff, "white": w, "ev_step": float(np.log2(f.t_eff / _median_t(pos)))}
                        for f, w in zip(pos.frames, whites, strict=True)],
        })
        if saturated:
            warnings.append(f"{name}: {saturated} px clipped in every frame (radiance there is a lower bound)")
    # ONE proxy scale for the whole set (pipeline.py's rule): per-view keys would give SfM brightness jumps
    scale = float(np.median([e.pop("key") for e in entries]))
    for k, e in enumerate(entries):
        tick("proxy", k / len(entries))
        hdr, _ = read_frame(out / e["hdr"])
        from PIL import Image

        Image.fromarray(px.tonemap_proxy(to_display(hdr, space or input_space), operator=proxy_operator,
                                         scale=scale)).save(out / e["proxy"])
        e["proxy_scale"] = scale
    sfm: dict[str, Any] = {"ran": False}
    if run_sfm:
        tick("sfm", 0.0)
        from .sfm import run_sfm as _sfm

        r = _sfm(out, PROXY_DIR, camera_model=camera_model, single_camera=True,
                 progress=lambda label, f: tick(f"sfm: {label}", f))
        sfm = {"ran": True, "registered": r.registered, "total": len(entries),
               "sparse": str(r.sparse_dir) if r.sparse_dir else None}
        if r.registered < len(entries):
            warnings.append(f"SfM registered {r.registered} of {len(entries)} positions")
    index = {
        "version": INDEX_VERSION, "linear": True, "crf": None, "colorspace": space or input_space,
        "proxy_operator": proxy_operator, "proxy_ev": proxy_ev, "source_dir": str(capture),
        "units": "scene-linear per second of exposure at the shoot's ISO and f-number (the developer's scale)",
        "capture": {k: rec.get(k) for k in ("kind", "name", "camera", "lens", "aperture", "iso", "wb", "base")},
        "positions": entries,
    }
    (out / INDEX_FILE).write_text(json.dumps(index, indent=1))
    report = {"ok": True, "positions": len(entries), "frames": sum(len(p.frames) for p in positions),
              "colorspace": space or input_space, "proxy_scale": scale, "sfm": sfm, "warnings": warnings,
              "seconds": round(time.time() - t0, 1), "out": str(out)}
    (out / "linear_report.json").write_text(json.dumps(report, indent=1))
    tick("done", 1.0)
    return report


def _median_t(pos: Position) -> float:
    return float(np.median([f.t_eff for f in pos.frames]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="hdr_brackets.linear", description=__doc__.split("\n\n")[0])
    ap.add_argument("capture", help="folder with tether_set.json")
    ap.add_argument("out")
    ap.add_argument("--exr-dir", help="the developed EXRs (default <capture>/exr)")
    ap.add_argument("--input-space", default="ACEScg", help="colourspace of EXRs that do not declare one")
    ap.add_argument("--max-side", type=int, default=0, help="downscale so the long side is at most this (0 = full)")
    ap.add_argument("--proxy-operator", default="aces", choices=px.OPERATORS)
    ap.add_argument("--camera-model", default="SIMPLE_RADIAL")
    ap.add_argument("--no-sfm", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    def progress(stage: str, f: float) -> None:          # one JSON line per step: N'Brain reads them as progress
        print(json.dumps({"stage": stage, "progress": round(f, 3)}), flush=True)

    try:
        report = run(a.capture, a.out, exr_dir=a.exr_dir, input_space=a.input_space, max_side=a.max_side,
                     proxy_operator=a.proxy_operator, run_sfm=not a.no_sfm, camera_model=a.camera_model,
                     progress=progress)
    except (FileNotFoundError, ValueError) as e:
        print(json.dumps({"ok": False, "error": str(e)}), flush=True)
        return 2
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
