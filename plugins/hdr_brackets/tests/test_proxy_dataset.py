"""The tonemapped proxy and the dataset layout M2/M3 read."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from hdr_brackets._tools import read_exr
from hdr_brackets.dataset import PositionResult, read_proxy, write_dataset, write_proxy
from hdr_brackets.exif import ExposureInfo
from hdr_brackets.proxy import OPERATORS, proxy_exposure_value, srgb_oetf, tone, tonemap_proxy
from hdr_brackets.response import CRF

# ── Proxy ─────────────────────────────────────────────────────────────────────────────────────────────────────────────


def test_auto_key_maps_geometric_mean_to_018() -> None:
    hdr = np.full((4, 4, 3), 7.5, np.float32)
    assert proxy_exposure_value(hdr) == pytest.approx(0.18 / 7.5, rel=1e-5)
    assert proxy_exposure_value(hdr, ev=1.0) == pytest.approx(0.36 / 7.5, rel=1e-5)
    for op in OPERATORS:
        expect = int(np.rint(255 * srgb_oetf(tone(np.array(0.18), op))))
        assert int(tonemap_proxy(hdr, operator=op)[0, 0, 0]) == expect
    assert int(tonemap_proxy(hdr, operator="linear")[0, 0, 0]) == 118  # sRGB(0.18) = 0.461


def test_proxy_monotonic_and_ev_brightens() -> None:
    ramp = np.exp(np.linspace(-8, 8, 512)).astype(np.float32)
    hdr = np.stack([ramp] * 3, axis=-1)[None]
    for op in OPERATORS:
        p0 = tonemap_proxy(hdr, 0.0, op)
        assert p0.dtype == np.uint8 and p0.shape == hdr.shape
        assert np.all(np.diff(p0[0, :, 0].astype(int)) >= 0)
        assert np.all(tonemap_proxy(hdr, 1.0, op) >= p0)
    with pytest.raises(ValueError):
        tonemap_proxy(hdr, operator="agx")


def test_proxy_survives_nan_and_zero() -> None:
    hdr = np.zeros((3, 3, 3), np.float32)
    hdr[0, 0] = np.nan
    out = tonemap_proxy(hdr)
    assert out.dtype == np.uint8 and out.max() == 0


def test_proxy_png_round_trips_8bit(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    px = rng.integers(0, 256, (24, 32, 3), dtype=np.uint8)
    p = write_proxy(tmp_path, "pos_000", px)
    assert p.name == "pos_000.png" and p.parent.name == "images"
    back = read_proxy(p)
    assert back.dtype == np.uint8 and np.array_equal(back, px)


# ── Dataset ───────────────────────────────────────────────────────────────────────────────────────────────────────────


def _info(name: str, t: float) -> ExposureInfo:
    return ExposureInfo(
        path=f"/src/{name}",
        exposure_time=t,
        fnumber=4.0,
        iso=100.0,
        ev=float(np.log2(t / 16 * 100)),
        timestamp=datetime(2026, 10, 27, 10, 0, 0),
        width=32,
        height=24,
    )


def test_write_dataset_layout_and_schema(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    crf = CRF(np.tile(np.log(np.linspace(1e-3, 1, 256)), (3, 1)), "test").anchored()
    positions = []
    for k in range(2):
        srcs = [_info(f"IMG_{3 * k + j + 1:04d}.jpg", 0.01 * 4.0 ** (j - 1)) for j in range(3)]
        hdr = rng.lognormal(0, 1, (24, 32, 3)).astype(np.float32)
        positions.append(
            PositionResult(
                name=f"pos_{k:03d}",
                bracket_id=k + 10,
                sources=srcs,
                hdr=hdr,
                proxy=tonemap_proxy(hdr),
                proxy_scale=proxy_exposure_value(hdr),
                ghost_score=0.01 * k,
                flagged=bool(k),
                shifts=[(0, 0), (0, 0), (1, -2)],
            )
        )
    write_dataset(
        tmp_path, positions, crf, [s for p in positions for s in p.sources], proxy_operator="aces", source_dir="/src"
    )
    for k in range(2):
        assert np.array_equal(read_exr(tmp_path / f"images_hdr/pos_{k:03d}.exr"), positions[k].hdr)
        assert np.array_equal(read_proxy(tmp_path / f"images/pos_{k:03d}.png"), positions[k].proxy)
    assert CRF.load(tmp_path / "crf.json").source == "test"
    idx = json.loads((tmp_path / "exposures.json").read_text())
    assert idx["crf"] == "crf.json" and idx["proxy_operator"] == "aces" and idx["source_dir"] == "/src"
    assert [p["name"] for p in idx["positions"]] == ["pos_000", "pos_001"]
    p1 = idx["positions"][1]
    assert set(p1) >= {"name", "hdr", "proxy", "proxy_scale", "bracket_id", "ghost_score", "flagged", "sources"}
    assert p1["hdr"] == "images_hdr/pos_001.exr" and p1["proxy"] == "images/pos_001.png"
    assert p1["bracket_id"] == 11 and p1["flagged"] is True and p1["ghost_score"] == pytest.approx(0.01)
    s = p1["sources"][2]
    assert set(s) >= {"file", "exposure_time", "ev", "iso", "fnumber", "timestamp"}
    assert s["file"] == "IMG_0006.jpg" and s["exposure_time"] == pytest.approx(0.04) and s["shift"] == [1, -2]
    assert s["timestamp"] == "2026:10:27 10:00:00"
