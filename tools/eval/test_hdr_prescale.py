"""`hdr_ply.py prescale`: a linear HDR PLY divided by a white point, header rescaled, undoable (numpy only, no binary).

Why it exists (docs/hdr.md "Octane", "Houdini 22"): OctaneRender 2026.5 / 2027.1 clamps every splat's colour to
[0, 1] BEFORE its Intensity pin, Karma XPU clips at ≈ 346 — the file has to arrive already divided by the white the
renderer may show, and the renderer's gain (Octane Intensity = W) puts the radiance back.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hdr_ply  # noqa: E402

C0 = hdr_ply.C0


def make_linear_ply(path: Path, n: int = 500, seed: int = 3, scale: float = 2421.4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fields = [(k, "<f4") for k in ("x", "y", "z", "nx", "ny", "nz")]
    fields += [(f"f_dc_{i}", "<f4") for i in range(3)] + [(f"f_rest_{i}", "<f4") for i in range(45)]
    fields += [("opacity", "<f4")] + [(f"scale_{i}", "<f4") for i in range(3)] + [(f"rot_{i}", "<f4") for i in range(4)]
    data = np.zeros(n, dtype=np.dtype(fields))
    radiance = np.exp(rng.normal(4.0, 2.5, (n, 3)))  # 0.01 … 20 000, a lamp and a dark corner
    for i in range(3):
        data[f"f_dc_{i}"] = (radiance[:, i] - 0.5) / C0
    for i in range(45):
        data[f"f_rest_{i}"] = rng.normal(0, 50.0, n)
    hdr_ply.write_ply(path, data, ["lichtfeld_hdr linear", f"hdr_scale {scale}", f"hdr_color_max {1000 * scale}",
                                   f"hdr_display_exposure {1 / scale}", "lichtfeld_colorspace lin_srgb"])
    return data


@pytest.mark.parametrize("white", [2421.4, 100.0])
def test_prescale_divides_radiance_and_rescales_header(tmp_path: Path, white: float) -> None:
    src, dst = tmp_path / "lin.ply", tmp_path / "pre.ply"
    data = make_linear_ply(src)
    rep = hdr_ply.prescale(src, dst, white)
    lines, hdr, out = hdr_ply.read_ply(dst)
    assert hdr["mode"] == "linear" and hdr["scale"] == pytest.approx(2421.4 / white)
    assert hdr["color_max"] == pytest.approx(1000 * 2421.4 / white) and hdr["display_exposure"] == pytest.approx(white / 2421.4)
    assert any(ln == f"comment hdr_prescale {white!r}" for ln in lines)
    assert "comment lichtfeld_colorspace lin_srgb" in lines  # untouched comments survive
    for i in range(3):
        before = 0.5 + C0 * data[f"f_dc_{i}"].astype(np.float64)
        after = 0.5 + C0 * out[f"f_dc_{i}"].astype(np.float64)
        np.testing.assert_allclose(after, before / white, rtol=2e-6, atol=1e-6)
    for i in range(45):
        np.testing.assert_allclose(out[f"f_rest_{i}"], data[f"f_rest_{i}"] / white, rtol=2e-6)
    for k in ("x", "opacity", "scale_0", "rot_3"):
        np.testing.assert_array_equal(out[k], data[k])
    assert rep["dc_max_after"] == pytest.approx(rep["dc_max_before"] / white, rel=1e-5)


def test_prescale_white_keywords_and_refusals(tmp_path: Path) -> None:
    src = tmp_path / "lin.ply"
    data = make_linear_ply(src)
    assert hdr_ply.main(["prescale", str(src), str(tmp_path / "s.ply"), "--white", "scale"]) == 0
    _, hdr_s, _ = hdr_ply.read_ply(tmp_path / "s.ply")
    assert hdr_s["scale"] == pytest.approx(1.0) and hdr_s["display_exposure"] == pytest.approx(1.0)
    assert hdr_ply.main(["prescale", str(src), str(tmp_path / "m.ply"), "--white", "max"]) == 0
    _, _, out_m = hdr_ply.read_ply(tmp_path / "m.ply")
    dc_m = np.stack([0.5 + C0 * out_m[f"f_dc_{i}"].astype(np.float64) for i in range(3)], 1)
    assert dc_m.max() == pytest.approx(1.0, rel=1e-5)  # nothing above 1 any more
    # a second prescale composes: W1 then W2 == W1·W2, and the header remembers only the last W (the file's own state)
    assert hdr_ply.main(["prescale", str(tmp_path / "s.ply"), str(tmp_path / "s2.ply"), "--white", "2"]) == 0
    _, hdr_s2, out_s2 = hdr_ply.read_ply(tmp_path / "s2.ply")
    assert hdr_s2["scale"] == pytest.approx(0.5)
    np.testing.assert_allclose(0.5 + C0 * out_s2["f_dc_0"].astype(np.float64),
                               (0.5 + C0 * data["f_dc_0"].astype(np.float64)) / (2421.4 * 2), rtol=5e-6, atol=1e-6)
    # refusals: a log PLY, a non-HDR PLY, a bad white
    hdr_ply.write_ply(tmp_path / "log.ply", data, ["lichtfeld_hdr log", "hdr_scale 2421.4", "hdr_color_max 2421400"])
    with pytest.raises(ValueError, match="linear"):
        hdr_ply.prescale(tmp_path / "log.ply", tmp_path / "x.ply", 10.0)
    hdr_ply.write_ply(tmp_path / "ldr.ply", data, [])
    with pytest.raises(ValueError, match="linear"):
        hdr_ply.prescale(tmp_path / "ldr.ply", tmp_path / "x.ply", 10.0)
    with pytest.raises(ValueError, match="positive"):
        hdr_ply.prescale(src, tmp_path / "x.ply", 0.0)
