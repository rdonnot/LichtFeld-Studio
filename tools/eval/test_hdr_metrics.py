"""Tests for hdr_metrics, log_metrics and ldr_regression."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hdr_metrics as hm  # noqa: E402
import ldr_regression  # noqa: E402
import log_metrics  # noqa: E402


def synth_crf() -> tuple[dict, np.ndarray]:
    z = np.arange(256) / 255.0
    lin = 0.002 + z**2.2  # smooth, monotonic
    g = np.log(lin)
    return {"log_irradiance": g.tolist(), "domain": "ldr8"}, g


def test_mu_law_bounds_monotonic():
    x = np.linspace(0, 1, 101)
    y = hm.mu_law(x)
    assert y[0] == 0 and abs(y[-1] - 1) < 1e-12
    assert np.all(np.diff(y) > 0)
    assert hm.mu_law(np.array([-3.0]))[0] == 0


def test_psnr():
    a = np.random.default_rng(0).random((32, 32, 3))
    assert hm.psnr(a, a) == float("inf")
    assert hm.psnr(a, a + 0.1) == pytest.approx(20.0, abs=1e-6)


def test_mu_psnr_scale_invariant():
    rng = np.random.default_rng(1)
    b = rng.random((32, 32, 3)) * 50
    a = b * (1 + 0.05 * rng.standard_normal(b.shape))
    v1 = hm.mu_law_psnr(a, b)
    v2 = hm.mu_law_psnr(a * 7.3, b * 7.3)
    assert v1 == pytest.approx(v2, abs=1e-9)


def test_apply_crf_roundtrip():
    crf, g = synth_crf()
    z = np.tile(np.arange(10, 250, dtype=np.uint8), (4, 1))[..., None].repeat(3, axis=-1)
    lin = np.exp(g[z])
    back = hm.apply_crf(lin, crf)
    assert np.abs(back.astype(int) - z.astype(int)).max() <= 2
    # callable form and 3-channel form
    back2 = hm.apply_crf(lin, {"log_irradiance": [g.tolist()] * 3})
    assert np.array_equal(back, back2)
    assert hm.apply_crf(np.full((2, 2, 3), 0.5), lambda x: x).max() in (127, 128)


def test_ldr_psnr_at_exposure():
    crf, _ = synth_crf()
    rng = np.random.default_rng(2)
    render = (rng.random((48, 48, 3)) * 0.8 + 0.1).astype(np.float32) * 3.0
    t = 0.1
    photo = hm.apply_crf(hm.apply_exposure(render, t), crf)
    p, frac = hm.ldr_psnr_at_exposure(render, photo, t, crf)
    assert p > 50 and 0 < frac <= 1
    p_wrong, _ = hm.ldr_psnr_at_exposure(render, photo, t * 2, crf)
    assert p_wrong < p - 10


def test_exr_roundtrip(tmp_path):
    img = np.random.default_rng(3).random((8, 9, 3)).astype(np.float32)
    img[0, 0] = [350.0, 0.5, 1e-3]
    path = str(tmp_path / "x.exr")
    hm.write_exr(path, img)
    back = hm.read_exr(path)
    assert back.dtype == np.float32
    assert back[0, 0, 0] == np.float32(350.0)
    np.testing.assert_array_equal(back, img)


def test_log_metrics(tmp_path):
    p = tmp_path / "sub" / "m.csv"
    for run, v in [("baseline_a", 24.0), ("baseline_b", 24.2), ("other", 10.0)]:
        log_metrics.log_rows(p, "M0", "s", run, {"psnr": v, "ssim": 0.8}, commit="abc")
    rows = log_metrics.read_rows(p)
    assert len(rows) == 6 and list(rows[0]) == log_metrics.COLUMNS
    mean, std, n = log_metrics.baseline(p, "s", "psnr")
    assert n == 2 and mean == pytest.approx(24.1) and std == pytest.approx(0.1414213, abs=1e-5)
    assert log_metrics.baseline(p, "nope", "psnr")[2] == 0
    assert log_metrics.main(["--csv", str(p), "--milestone", "M1", "--scene", "s", "--run", "r", "psnr=1"]) == 0
    assert log_metrics.read_rows(p)[-1]["value"] == "1.0"


def fake_out(d: Path, psnr: float) -> Path:
    d.mkdir(parents=True)
    with (d / "metrics.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "psnr", "ssim", "lpips", "time_per_image", "num_gaussians"])
        w.writerow([1000, 20.0, 0.7, 0.3, 0.1, 5000])
        w.writerow([7000, psnr, 0.83, 0.2, 0.1, 100000])
    per = {
        f"img{i}.jpg": {"evaluations": [{"psnr": psnr + d_, "ssim": 0.8, "lpips": 0.2, "step": 7000, "split": "test"}]}
        for i, d_ in enumerate([-1, 1])
    }
    (d / "per_image_metrics.json").write_text(json.dumps(per))
    return d


def test_ldr_regression(tmp_path, capsys):
    csvp = str(tmp_path / "m.csv")
    base = ["--milestone", "M2", "--scene", "smoke", "--csv", csvp]
    out = fake_out(tmp_path / "o1", 24.0)
    assert ldr_regression.main([str(out), *base, "--run", "baseline_a"]) == 0
    assert "no baseline" in capsys.readouterr().out
    rows = {r["metric"]: r["value"] for r in log_metrics.read_rows(csvp)}
    assert float(rows["psnr"]) == 24.0 and float(rows["psnr_test_std"]) == pytest.approx(1.0)
    assert float(rows["num_gaussians"]) == 100000
    ok = fake_out(tmp_path / "o2", 24.05)
    assert ldr_regression.main([str(ok), *base, "--run", "hdr_off"]) == 0
    assert capsys.readouterr().out.startswith("PASS")
    bad = fake_out(tmp_path / "o3", 23.0)
    assert ldr_regression.main([str(bad), *base, "--run", "hdr_on"]) == 1
    assert capsys.readouterr().out.startswith("FAIL")
    assert ldr_regression.main([str(bad), *base, "--run", "x", "--no-judge"]) == 0
