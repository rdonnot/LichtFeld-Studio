"""The bracket training set — `<out>/brackets/`, a self-contained dataset LichtFeld trains on with `--hdr-brackets`.

M2 trains on the merged EXRs; M3 (docs/hdr-m3-design.md §1) trains on the RAW JPEGs: every frame of a bracket is a
training image with its position's pose, the trainer seeds each frame's exposure from the JPEG's own EXIF and maps
radiance to the frame through the fixed Debevec LUT in crf.json. This module writes that set from a finished dataset's
own files (exposures.json, crf.json, sparse/0), so `python -m hdr_brackets.brackets_set <out>` regenerates it alone:

    <out>/brackets/
      images/pos_NNN_fK.jpg      symlink (fallback: copy) of position NNN's raw JPEG, K = 0..n−1 by exposure time
                                 ascending; the file's APP1 EXIF is untouched (the trainer reads it from the JPEG)
      sparse/0/cameras.txt       TEXT model (LichtFeld reads text when there is no cameras.bin): the SfM model's
              images.txt         shared camera; one image record per FRAME, IMAGE_ID = stride·p + K for proxy image
              points3D.txt       id p, the proxy's pose and camera verbatim, name pos_NNN_fK.jpg, followed by the
                                 proxy's POINTS2D line verbatim; every track image_id remapped p → stride·p (frame 0)
      crf.json                   <out>/crf.json + iso_ref, fnumber_ref (the shoot's medians), exif_ev_ref =
                                 log2(iso_ref / fnumber_ref²) and a `units` string
      exposures.json             <out>/exposures.json restricted to the written positions, each source with its
                                 `frame` name, plus a `brackets` block (stride, link mode, omitted positions)

Units (verified on data/smoke_hdr, 2026-10-28): the merge (merge.py) is `ln E = Σ w (g[z] − ln t) / Σ w`, so an EXR
pixel is `E = exp(g[z]) / t` — "exposure per second" at the ISO and f-number the shoot used, with g(128) = 0. A frame
shot for t_i seconds then reads `z = g⁻¹(ln(E · t_i))`, and with `exif_ev = log2(t·ISO/N²)` (exif.py, LichtFeld's
core/exif.cpp) `t_i = 2^(exif_ev_i − exif_ev_ref)` exactly when `exif_ev_ref = log2(ISO/N²)` of the shoot. That is
what the trainer computes: `x_i = v · 2^(exif_ev_i − exif_ev_ref)`, then the LUT. `tools/eval/check_brackets.py`
measures it: `hdr_metrics.apply_crf(EXR · 2^(exif_ev − exif_ev_ref), crf)` vs the JPEG is 1–2/255 median on the
unsaturated pixels of the −4…+2 EV frames; drop the anchor and it is off by 2^2.64 = 6.25× (design §8, risk 1).

Which positions: a position is written when its proxy is in the SfM model (a pose) and its bracket is not flagged
for ghosting; the others are omitted and named in the warnings. A frame MTB moved by (dx, dy) ≠ (0, 0) does not share
the reference's pose exactly (design §8, risk 7): it gets its own camera, the shared one with the principal point
moved by (−dx, −dy) — `cv2.AlignMTB.shiftMat` moves the raw frame by +shift onto the reference, so a scene point at
reference pixel u sits at u − shift in the raw frame. Only for camera models whose principal point this module
knows (see `_PRINCIPAL_POINT_INDEX`); otherwise the frame keeps the shared camera and a warning says so.

The id stride is `max(10, most frames in a position)`: with ≤ 10 frames the ids are 10·p + K, as the design reads
them; a 12-frame bracket set gets stride 12 (written to exposures.json `brackets.id_stride`). Ids must not collide.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import CRF_FILE, INDEX_FILE, PROXY_DIR

log = logging.getLogger(__name__)

BRACKETS_DIR = "brackets"
IMAGES_DIR = "images"
SPARSE_REL = "sparse/0"
MIN_STRIDE = 10
TEXT_FILES = ("cameras.txt", "images.txt", "points3D.txt")
BINARY_FILES = ("cameras.bin", "images.bin")
UNITS = (
    "x-axis of the LUT (exp(log_irradiance[z])): 1.0 = code 128 after 1 s at iso_ref, fnumber_ref; a frame's "
    "input is EXR radiance × 2^(exif_ev − exif_ev_ref) = radiance × exposure_time at the shoot's ISO and f-number"
)
#: Index of cx in PARAMS[] per COLMAP camera model (cy follows). Models not listed keep the shared camera.
_PRINCIPAL_POINT_INDEX = {
    "SIMPLE_PINHOLE": 1,
    "SIMPLE_RADIAL": 1,
    "SIMPLE_RADIAL_FISHEYE": 1,
    "RADIAL": 1,
    "RADIAL_FISHEYE": 1,
    "PINHOLE": 2,
    "OPENCV": 2,
    "OPENCV_FISHEYE": 2,
    "FULL_OPENCV": 2,
    "FOV": 2,
    "THIN_PRISM_FISHEYE": 2,
}


# ── COLMAP text model ─────────────────────────────────────────────────────────────────────────────────────────────────


@dataclass
class TextCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: list[str]  # verbatim tokens

    def line(self) -> str:
        return " ".join([str(self.camera_id), self.model, str(self.width), str(self.height), *self.params])


@dataclass
class TextImage:
    image_id: int
    pose: list[str]  # QW QX QY QZ TX TY TZ, verbatim tokens
    camera_id: int
    name: str
    points2d: str = ""  # the POINTS2D line, verbatim ("" when the image has none)


@dataclass
class TextModel:
    cameras: list[TextCamera]
    images: list[TextImage]
    points3d: list[str] = field(default_factory=list)  # verbatim lines

    def image_by_name(self) -> dict[str, TextImage]:
        return {im.name: im for im in self.images}


def _is_image_record(tokens: list[str]) -> bool:
    """COLMAP's image line: int id, 7 floats, int camera id, a name with an extension (LichtFeld's own test)."""
    if len(tokens) < 10:
        return False
    try:
        int(tokens[0])
        q = [float(t) for t in tokens[1:8]]
        int(tokens[8])
    except ValueError:
        return False
    name = " ".join(tokens[9:])
    dot = name.rfind(".")
    if dot < 0 or dot == len(name) - 1 or not name[dot + 1].isalpha():
        return False
    return all(math.isfinite(v) for v in q)


def _data_lines(path: Path) -> list[str]:
    return [ln.rstrip("\r\n") for ln in path.read_text().splitlines() if not ln.startswith("#")]


def read_text_model(sparse_dir: str | os.PathLike) -> TextModel:
    """cameras.txt + images.txt + points3D.txt → TextModel, every numeric token kept as written."""
    d = Path(sparse_dir)
    cameras: list[TextCamera] = []
    for ln in _data_lines(d / "cameras.txt"):
        t = ln.split()
        if len(t) < 4:
            continue
        cameras.append(TextCamera(int(t[0]), t[1], int(t[2]), int(t[3]), t[4:]))
    lines = _data_lines(d / "images.txt")
    images: list[TextImage] = []
    i = 0
    while i < len(lines):
        t = lines[i].split()
        if not _is_image_record(t):
            i += 1
            continue
        im = TextImage(int(t[0]), t[1:8], int(t[8]), " ".join(t[9:]))
        i += 1
        if i < len(lines) and not _is_image_record(lines[i].split()):
            im.points2d = lines[i].strip()
            i += 1
        images.append(im)
    points = [ln.strip() for ln in _data_lines(d / "points3D.txt") if ln.strip()]
    return TextModel(cameras, images, points)


def find_sparse_model(out_dir: str | os.PathLike) -> Path | None:
    """`<out>/sparse/0` (or `<out>/sparse`) when it holds a text or a binary model."""
    out = Path(out_dir)
    for d in (out / "sparse" / "0", out / "sparse"):
        if all((d / f).is_file() for f in TEXT_FILES[:2]) or all((d / f).is_file() for f in BINARY_FILES):
            return d
    return None


def load_model(sparse_dir: str | os.PathLike) -> TextModel:
    """A COLMAP model, text or binary (binary goes through pycolmap's `write_text` into a temp dir)."""
    d = Path(sparse_dir)
    if all((d / f).is_file() for f in TEXT_FILES[:2]):
        return read_text_model(d)
    import pycolmap  # noqa: PLC0415  (lazy: the heavy optional dependency, as in sfm.py)

    rec = pycolmap.Reconstruction(str(d))
    with tempfile.TemporaryDirectory(prefix="hdr_brackets_txt_") as tmp:
        rec.write_text(tmp)
        return read_text_model(tmp)


def write_text_model(sparse_dir: str | os.PathLike, model: TextModel) -> None:
    """COLMAP's text format: comment headers, one camera per line, two lines per image, one point per line."""
    d = Path(sparse_dir)
    d.mkdir(parents=True, exist_ok=True)
    with (d / "cameras.txt").open("w") as f:
        f.write("# Camera list with one line of data per camera:\n#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(model.cameras)}\n")
        for c in model.cameras:
            f.write(c.line() + "\n")
    n_obs = sum(len(im.points2d.split()) // 3 for im in model.images)
    with (d / "images.txt").open("w") as f:
        f.write(
            "# Image list with two lines of data per image:\n#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        )
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(
            f"# Number of images: {len(model.images)}, mean observations per image: {n_obs / max(1, len(model.images))}\n"
        )
        for im in model.images:
            f.write(" ".join([str(im.image_id), *im.pose, str(im.camera_id), im.name]) + "\n")
            f.write(im.points2d + "\n")
    with (d / "points3D.txt").open("w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(model.points3d)}\n")
        for ln in model.points3d:
            f.write(ln + "\n")


# ── The set ───────────────────────────────────────────────────────────────────────────────────────────────────────────


@dataclass
class BracketsResult:
    path: str
    positions: int
    frames: int
    id_stride: int
    link: str  # "symlink" | "copy"
    exif_ev_ref: float
    omitted: list[str] = field(default_factory=list)

    def summary(self) -> str:
        s = f"brackets/: {self.positions} positions, {self.frames} frames ({self.link}), exif_ev_ref {self.exif_ev_ref:.3f}"
        return s + (f", {len(self.omitted)} omitted" if self.omitted else "")


def frame_name(position: str, k: int) -> str:
    return f"{position}_f{k}.jpg"


def exif_ev_ref(iso: float | None, fnumber: float | None) -> float:
    """log2(iso / fnumber²), a missing factor = 1 (the same convention as exif.compute_ev): 2.644 at ISO 100 f/4."""
    return math.log2((iso if iso else 1.0) / ((fnumber if fnumber else 1.0) ** 2))


def _median(values: list[float | None]) -> float | None:
    v = [float(x) for x in values if x is not None and x > 0]
    return float(np.median(v)) if v else None


def _link_or_copy(src: Path, dst: Path, link: bool) -> str:
    dst.unlink(missing_ok=True)
    if link:
        try:
            os.symlink(os.path.relpath(src, dst.parent), dst)
            return "symlink"
        except OSError as e:  # no symlink privilege (Windows), a filesystem that refuses: copy keeps EXIF intact
            log.info("symlink %s failed (%s): copying", dst.name, e)
    shutil.copy2(src, dst)
    return "copy"


def _shifted_camera(base: TextCamera, camera_id: int, shift: tuple[int, int]) -> TextCamera | None:
    idx = _PRINCIPAL_POINT_INDEX.get(base.model)
    if idx is None or len(base.params) < idx + 2:
        return None
    params = list(base.params)
    params[idx] = repr(float(params[idx]) - float(shift[0]))
    params[idx + 1] = repr(float(params[idx + 1]) - float(shift[1]))
    return TextCamera(camera_id, base.model, base.width, base.height, params)


def write_brackets(
    out_dir: str | os.PathLike,
    *,
    link: bool = True,
    warnings: list[str] | None = None,
    index: dict[str, Any] | None = None,
    model: TextModel | None = None,
) -> BracketsResult | None:
    """`<out>/brackets/` from `<out>/exposures.json`, `<out>/crf.json` and `<out>/sparse/0` (see the module docstring).

    Returns None (with a warning) when there is no SfM model or no position to write. `warnings` collects what was
    omitted and why; `index` / `model` let a caller pass what it already has in memory."""
    out = Path(out_dir)
    warn_list = warnings if warnings is not None else []

    def warn(msg: str) -> None:
        log.warning(msg)
        warn_list.append(msg)

    if index is None:
        p = out / INDEX_FILE
        if not p.is_file():
            warn(f"brackets: no {INDEX_FILE} in {out}; nothing written")
            return None
        index = json.loads(p.read_text())
    if model is None:
        sparse = find_sparse_model(out)
        if sparse is None:
            warn("brackets: no SfM model in sparse/0 (poses are needed); nothing written")
            return None
        model = load_model(sparse)
    crf_path = out / CRF_FILE
    if not crf_path.is_file():
        warn(f"brackets: no {CRF_FILE} in {out}; nothing written")
        return None
    source_dir = Path(index.get("source_dir") or "")
    by_name = model.image_by_name()
    if not model.cameras:
        warn("brackets: the SfM model has no camera; nothing written")
        return None
    cameras = list(model.cameras)
    next_camera_id = max(c.camera_id for c in cameras) + 1
    camera_of = {c.camera_id: c for c in cameras}

    # ── which positions, which frames ──
    chosen: list[tuple[dict[str, Any], TextImage, list[dict[str, Any]]]] = []
    omitted: list[str] = []
    for pos in index.get("positions", []):
        name = pos["name"]
        proxy = os.path.basename(pos.get("proxy") or f"{PROXY_DIR}/{name}.png")
        im = by_name.get(proxy)
        if im is None:
            omitted.append(name)
            warn(f"brackets: {name} omitted — {proxy} is not registered in the SfM model (no pose)")
            continue
        if pos.get("flagged"):
            omitted.append(name)
            warn(
                f"brackets: {name} omitted — its bracket is flagged for ghosting (score {pos.get('ghost_score', 0):.3f})"
            )
            continue
        srcs = sorted(
            pos.get("sources", []), key=lambda s: (s.get("exposure_time") is None, s.get("exposure_time") or 0.0)
        )
        srcs = [s for s in srcs if s.get("exposure_time")]
        if not srcs:
            omitted.append(name)
            warn(f"brackets: {name} omitted — no source with an exposure time")
            continue
        chosen.append((pos, im, srcs))
    if not chosen:
        warn("brackets: no position with a pose; nothing written")
        return None
    stride = max(MIN_STRIDE, max(len(srcs) for _, _, srcs in chosen))

    # ── images/ ──
    root = out / BRACKETS_DIR
    img_dir = root / IMAGES_DIR
    if img_dir.is_dir():
        for p in img_dir.glob("pos_*_f*.jpg"):
            p.unlink()
    img_dir.mkdir(parents=True, exist_ok=True)
    link_modes: set[str] = set()
    images: list[TextImage] = []
    entries: list[dict[str, Any]] = []
    unknown_model: list[str] = []
    for pos, im, srcs in sorted(chosen, key=lambda c: c[0]["name"]):
        name = pos["name"]
        new_srcs = []
        for k, s in enumerate(srcs):
            fname = frame_name(name, k)
            src = source_dir / s["file"]
            if not src.is_file():
                warn(f"brackets: {name} frame {k}: {src} is missing; the set is incomplete")
                continue
            link_modes.add(_link_or_copy(src, img_dir / fname, link))
            cam_id = im.camera_id
            dx, dy = (int(v) for v in (s.get("shift") or [0, 0]))
            if (dx, dy) != (0, 0):
                cam = _shifted_camera(camera_of[im.camera_id], next_camera_id, (dx, dy))
                if cam is None:
                    unknown_model.append(fname)
                else:
                    cameras.append(cam)
                    cam_id = next_camera_id
                    next_camera_id += 1
            images.append(TextImage(stride * im.image_id + k, list(im.pose), cam_id, fname, im.points2d))
            new_srcs.append(dict(s, frame=fname))
        entry = dict(pos)
        for key in ("hdr", "proxy"):
            if isinstance(entry.get(key), str):
                entry[key] = f"../{entry[key]}"
        entry["sources"] = new_srcs
        entries.append(entry)
    if unknown_model:
        warn(
            f"brackets: {len(unknown_model)} shifted frame(s) kept the shared camera (principal point of camera model "
            f"{camera_of[chosen[0][1].camera_id].model} unknown): {', '.join(unknown_model[:6])}"
        )

    # ── sparse/0 ──
    written_ids = {im.image_id for im in images}
    points: list[str] = []
    for ln in model.points3d:
        t = ln.split()
        if len(t) < 8:
            continue
        track = t[8:]
        kept: list[str] = []
        for j in range(0, len(track) - 1, 2):
            new_id = stride * int(track[j])
            if new_id in written_ids:
                kept.extend((str(new_id), track[j + 1]))
        points.append(" ".join([*t[:8], *kept]))
    write_text_model(root / SPARSE_REL, TextModel(cameras, images, points))

    # ── crf.json ──
    iso_ref = _median([s.get("iso") for _, _, srcs in chosen for s in srcs])
    fn_ref = _median([s.get("fnumber") for _, _, srcs in chosen for s in srcs])
    if iso_ref is None or fn_ref is None:
        warn("brackets: ISO or f-number missing from every source EXIF; exif_ev_ref uses 1 for the missing factor")
    ev_ref = exif_ev_ref(iso_ref, fn_ref)
    crf = json.loads(crf_path.read_text())
    crf.update(iso_ref=iso_ref, fnumber_ref=fn_ref, exif_ev_ref=ev_ref, units=UNITS)
    (root / CRF_FILE).write_text(json.dumps(crf))

    # ── exposures.json ──
    mode = (
        "symlink" if link_modes == {"symlink"} else "copy" if link_modes == {"copy"} else "+".join(sorted(link_modes))
    )
    idx = dict(index)
    idx["positions"] = entries
    idx["crf"] = CRF_FILE
    idx["brackets"] = {
        "images": IMAGES_DIR,
        "sparse": SPARSE_REL,
        "id_stride": stride,
        "link": mode,
        "exif_ev_ref": ev_ref,
        "iso_ref": iso_ref,
        "fnumber_ref": fn_ref,
        "omitted": omitted,
        "frame_units": UNITS,
    }
    (root / INDEX_FILE).write_text(json.dumps(idx, indent=1))
    res = BracketsResult(str(root), len(entries), len(images), stride, mode, ev_ref, omitted)
    log.info(res.summary())
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hdr_brackets.brackets_set", description="(Re)write <out>/brackets/ from a finished dataset"
    )
    ap.add_argument("out", help="the plugin's output dataset (exposures.json, crf.json, sparse/0)")
    ap.add_argument("--copy", action="store_true", help="copy the JPEGs instead of symlinking them")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s")
    warnings: list[str] = []
    res = write_brackets(a.out, link=not a.copy, warnings=warnings)
    if res is None:
        return 1
    print(json.dumps({"brackets": res.__dict__, "warnings": warnings}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
