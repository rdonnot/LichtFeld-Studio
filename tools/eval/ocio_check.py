"""tools/eval/ocio_check.py — the OCIO display path end to end (docs/hdr.md "Colour management").

Renders one dataset camera of an HDR model headless with `--render-tonemap ocio` (the CUDA bake of the
baked 65^3 LUT), keeps the linear EXR of the same frame, pushes that EXR through PyOpenColorIO's own
DisplayViewTransform (exposure applied in linear first, as the app does) and reports the PSNR between the
app's 8-bit frame and OCIO's reference. It also checks the EXR header carries the working-space
declaration (chromaticities + lichtfeld_colorspace).

    cd tools && uv run python eval/ocio_check.py --binary ../build/LichtFeld-Studio \
        --ply ../../LichtFeld-HDR-m5/out/m5_roundtrip_m2_hdr7k/model_log.ply --dataset ../data/smoke_hdr \
        --ocio /opt/hfs22.0/packages/ocio/cg-config-v3.0.0_aces-v2.0_ocio-v2.5.ocio \
        --display "sRGB - Display" --view "ACES 2.0 - SDR 100 nits (Rec.709)" --out ../out/ocio_check

The config is a FILE on purpose: the app's OCIO (2.5.1, vcpkg) and PyOpenColorIO (2.6) ship different
versions of the built-in configs, and `ocio://studio-config-latest` would not name the same transform.
The video codec (crf 18) is the floor of the comparison, not the LUT: expect >= 40 dB.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


def quat_wxyz_from_matrix(m: np.ndarray) -> list[float]:
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    return [float(w), float(x), float(y), float(z)]


def camera_path_from_colmap(dataset: Path, image_name: str | None, out_json: Path) -> tuple[int, int]:
    """One-keyframe timeline at a dataset camera. COLMAP w2c (R, T) -> visualizer pose
    rot = F R^T F, pos = -F R^T T with F = diag(1, -1, -1) (coordinate_conventions.hpp
    dataWorldToCameraFromVisualizerPose), rotation stored [w, x, y, z]."""
    import pycolmap

    rec = pycolmap.Reconstruction(str(dataset / "sparse" / "0"))
    images = sorted(rec.images.values(), key=lambda im: im.name)
    image = next((im for im in images if image_name and im.name.startswith(image_name)), images[len(images) // 2])
    cam = rec.cameras[image.camera_id]
    pose = image.cam_from_world() if callable(getattr(image, "cam_from_world", None)) else image.cam_from_world
    R = np.asarray(pose.rotation.matrix(), dtype=np.float64)
    T = np.asarray(pose.translation, dtype=np.float64)
    F = np.diag([1.0, -1.0, -1.0])
    rot = F @ R.T @ F
    pos = -F @ (R.T @ T)
    fx = float(cam.focal_length_x) if hasattr(cam, "focal_length_x") else float(cam.params[0])
    width, height = int(cam.width), int(cam.height)
    hfov = 2.0 * math.atan(width / (2.0 * fx))
    focal_mm = 36.0 / (2.0 * math.tan(hfov / 2.0))
    timeline = {
        "version": 1,
        "clip_duration": 0.04,
        "keyframes": [
            {"time": 0.0, "position": [float(v) for v in pos], "rotation": quat_wxyz_from_matrix(rot),
             "focal_length_mm": focal_mm, "easing": 0},
        ],
    }
    out_json.write_text(json.dumps(timeline, indent=2))
    print(f"camera {image.name}: {width}x{height}, focal {focal_mm:.2f} mm -> {out_json}")
    return width, height


def read_ply_hdr_scale(ply: Path) -> float:
    with ply.open("rb") as f:
        header = f.read(1 << 16).decode("latin-1")
    m = re.search(r"comment hdr_scale ([0-9.eE+-]+)", header)
    return float(m.group(1)) if m else 1.0


def read_exr(path: Path) -> tuple[np.ndarray, dict]:
    import OpenEXR

    with OpenEXR.File(str(path)) as f:
        header = dict(f.header())
        ch = f.channels()
        if "RGB" in ch:
            rgb = np.asarray(ch["RGB"].pixels, dtype=np.float32)
        else:
            rgb = np.stack([np.asarray(ch[c].pixels, dtype=np.float32) for c in ("R", "G", "B")], axis=-1)
    return rgb, header


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 0 else 10.0 * math.log10(1.0 / mse)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--ply", required=True, help="an HDR PLY (lichtfeld_hdr linear|log)")
    ap.add_argument("--dataset", required=True, help="COLMAP dataset (sparse/0) the camera comes from")
    ap.add_argument("--image", default=None, help="dataset image name prefix (default: the middle one)")
    ap.add_argument("--ocio", required=True, help="OCIO config FILE (same transform in both libraries)")
    ap.add_argument("--display", default="sRGB - Display")
    ap.add_argument("--view", default="ACES 2.0 - SDR 100 nits (Rec.709)")
    ap.add_argument("--look", default="")
    ap.add_argument("--ev", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-psnr", type=float, default=40.0)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    width, height = camera_path_from_colmap(Path(args.dataset), args.image, out / "camera_path.json")
    exr_dir = out / "exr"
    video = out / "video.mp4"
    cmd = [args.binary, "--headless", "--render-camera-path", str(out / "camera_path.json"),
           "--render-load", args.ply, "--render-output", str(video),
           "--render-width", str(width), "--render-height", str(height), "--render-fps", "25", "--render-crf", "10",
           "--render-tonemap", "ocio", "--render-exposure", str(args.ev), "--render-exr-dir", str(exr_dir),
           "--ocio", args.ocio, "--ocio-display", args.display, "--ocio-view", args.view]
    if args.look:
        cmd += ["--ocio-look", args.look]
    print("+", " ".join(cmd))
    log = out / "render.log"
    with log.open("w") as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env={**os.environ, "OCIO": ""}).returncode
    if rc != 0:
        print(log.read_text()[-4000:])
        print(f"render failed (exit {rc})")
        return 2

    exr_path = exr_dir / "frame_000000.exr"
    linear, header = read_exr(exr_path)
    tag = header.get("lichtfeld_colorspace")
    chroma = header.get("chromaticities")
    print(f"EXR header: lichtfeld_colorspace={tag!r} chromaticities={chroma!r}")
    header_ok = tag == "lin_srgb" and chroma is not None

    # OCIO reference: exposure in linear (2^EV / s), then the display/view transform of the SAME config file.
    import PyOpenColorIO as ocio

    cfg = ocio.Config.CreateFromFile(args.ocio)
    scale = read_ply_hdr_scale(Path(args.ply))
    exposed = np.clip(linear * (2.0 ** args.ev / scale), 0.0, None).astype(np.float32)
    group = ocio.GroupTransform()
    if args.look:
        group.appendTransform(ocio.LookTransform(src="lin_srgb", dst="lin_srgb", looks=args.look))
    group.appendTransform(ocio.DisplayViewTransform(src="lin_srgb", display=args.display, view=args.view))
    proc = cfg.getProcessor(group).getDefaultCPUProcessor()
    ref = np.ascontiguousarray(exposed.reshape(-1, 3)).copy()
    proc.applyRGB(ref)
    ref = np.clip(ref.reshape(exposed.shape), 0.0, 1.0)

    import cv2

    cap = cv2.VideoCapture(str(video))
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        print("could not decode the video's first frame")
        return 2
    frame = frame_bgr[:, :, ::-1].astype(np.float32) / 255.0
    if frame.shape[:2] != ref.shape[:2]:
        print(f"frame {frame.shape} vs EXR {ref.shape}")
        return 2
    value = psnr(frame, ref)
    ref8 = np.round(ref * 255.0)
    diff = np.abs(frame * 255.0 - ref8)
    print(f"app video frame vs PyOpenColorIO reference ({args.display} / {args.view}, EV {args.ev:+.2f}, s = {scale}): "
          f"PSNR {value:.2f} dB, mean |d| {diff.mean():.3f}/255, p99 {np.quantile(diff, 0.99):.1f}/255, "
          f"frame mean {frame.mean() * 255:.1f}/255, ref mean {ref8.mean():.1f}/255")
    cv2.imwrite(str(out / "app_frame.png"), frame_bgr)
    cv2.imwrite(str(out / "ocio_reference.png"), (ref[:, :, ::-1] * 255.0 + 0.5).astype(np.uint8))
    if not header_ok:
        print("FAIL: the EXR header does not declare the working space")
        return 1
    if value < args.min_psnr:
        print(f"FAIL: {value:.2f} dB < {args.min_psnr} dB")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
