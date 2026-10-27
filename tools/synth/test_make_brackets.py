"""The synthetic camera is self-consistent: EXIF round-trips, the written CRF is the curve the pixels went through,
and tools/eval's apply_crf reproduces the JPEGs from the EXR + exposures_gt.json (the M1 measurement path)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_brackets as mb  # noqa: E402
from hdr_metrics import apply_crf, write_exr  # noqa: E402
from PIL import Image  # noqa: E402


def test_crf_monotonic_and_invertible() -> None:
    x = np.linspace(0, 1, 1001)
    y = mb.crf_forward(x)
    assert y[0] == pytest.approx(0.0) and y[-1] == pytest.approx(1.0)
    assert np.all(np.diff(y) > 0)
    assert np.allclose(mb.crf_inverse(y), x, atol=1e-6)
    g = np.array(mb.crf_debevec())
    assert g.shape == (3, 256) and np.all(np.diff(g, axis=1) > 0)


def test_apply_crf_matches_camera() -> None:
    sensor = np.random.default_rng(1).random((16, 16, 3)).astype(np.float32)
    expected = mb.to_uint8(mb.crf_forward(sensor))
    got = apply_crf(sensor, {"log_irradiance": mb.crf_debevec(), "domain": "ldr8"})
    assert np.abs(got.astype(int) - expected.astype(int)).max() <= 1


def test_brackets_end_to_end(tmp_path: Path) -> None:
    gt = tmp_path / "gt"
    gt.mkdir()
    rng = np.random.default_rng(3)
    for i in range(2):
        radiance = (np.exp(rng.normal(0.0, 2.0, (24, 32, 3))) * 0.5).astype(np.float32)  # ≈ 12 stops
        write_exr(str(gt / f"view_{i:03d}.exr"), radiance)
    out = tmp_path / "raw"
    assert mb.main(["--gt", str(gt), "--out", str(out), "--seed", "5"]) == 0

    meta = json.loads((out / "exposures_gt.json").read_text())
    assert len(meta["images"]) == 10
    assert [r["ev"] for r in meta["images"][:5]] == [-4.0, -2.0, 0.0, 2.0, 4.0]
    t = [r["exposure_time"] for r in meta["images"][:3]]
    assert t[1] / t[0] == pytest.approx(4.0) and t[2] / t[1] == pytest.approx(4.0)
    stamps = [r["timestamp"] for r in meta["images"]]
    assert stamps == sorted(stamps) and len(set(stamps)) == len(stamps)

    # EXIF as the plugin (and LichtFeld's exif.cpp) will read it.
    for r in meta["images"]:
        with Image.open(out / r["file"]) as im:
            ifd = im.getexif().get_ifd(mb.TAG_EXIF_IFD)
            assert mb.exif_float(ifd[mb.TAG_EXPOSURE_TIME]) == pytest.approx(r["exposure_time"], rel=1e-5)
            assert int(ifd[mb.TAG_ISO]) == r["iso"]
            assert str(ifd[mb.TAG_DATETIME_ORIGINAL]) == r["timestamp"]

    # The eval path: EXR → exposure → the written CRF → the JPEG (noise + JPEG only).
    crf = json.loads((out / "crf.json").read_text())
    from hdr_metrics import read_exr

    radiance = read_exr(str(gt / "view_000.exr"))
    for r in meta["images"][:5]:
        pred = apply_crf(radiance * r["exposure_time"] * meta["gain"], crf).astype(int)
        with Image.open(out / r["file"]) as im:
            photo = np.asarray(im.convert("RGB")).astype(int)
        ok = (photo > 5) & (photo < 250)
        err = np.abs(pred - photo)[ok]
        assert err.size > 0 and np.median(err) <= 3, f"EV {r['ev']}: median |Δ| {np.median(err)}"
