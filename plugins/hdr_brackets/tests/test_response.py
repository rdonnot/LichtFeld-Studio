"""CRF recovery against the synthetic camera's truth (crf.json), the file form, and the two engines' agreement.

Numbers reached on set6 (6 views 96×128, ±2 EV, seed 23; max |Δg| over z ∈ [20, 235] after matching at z = 128,
natural log): 3 brackets 0.111, 6 brackets 0.078 with the defaults (1000 samples per bracket, λ = 30). OpenCV's own
defaults (70 samples, λ = 10) reach only 0.15–0.25 on these frames: the synthetic noise (shot + read + JPEG), not the
solver — noise-free frames give 0.015 with 70 samples.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest

from hdr_brackets.pipeline import load_rgb8
from hdr_brackets.response import CRF, calibrate_crf, compare_crf, debevec_numpy, debevec_opencv

log = logging.getLogger(__name__)


def _brackets(bs, n: int):
    imgs, times = [], []
    for vi in range(n):
        rows = sorted(bs.view(vi), key=lambda r: r["exposure_time"])
        imgs.append([load_rgb8(bs.raw_dir / r["file"]) for r in rows])
        times.append([r["exposure_time"] for r in rows])
    return imgs, times


@pytest.mark.parametrize("n_brackets", [3, 6])
def test_crf_recovery_matches_truth(set6, n_brackets: int) -> None:
    imgs, times = _brackets(set6, n_brackets)
    crf = calibrate_crf(imgs, times)
    err = compare_crf(crf, CRF.from_dict(set6.crf_truth))
    log.info("CRF from %d brackets: max |dg| over [20, 235] = %.4f", n_brackets, err)
    print(f"CRF from {n_brackets} brackets: max |dg| [20,235] = {err:.4f}")
    assert err < 0.15
    g = crf.log_irradiance
    assert g.shape == (3, 256) and np.allclose(g[:, 128], 0.0) and np.all(np.diff(g, axis=1) >= 0)


def test_numpy_engine_is_opencvs_least_squares(set6) -> None:
    """Same samples (OpenCV's grid), same equations: the curves agree wherever data constrains them."""
    imgs, times = _brackets(set6, 1)
    go = debevec_opencv(imgs, times, 30, 10.0)
    gn, _ = debevec_numpy(imgs, times, 30, 10.0, skip_clipped=False)
    go = go - go[:, 128:129]
    assert np.median(np.abs(go - gn)) < 0.01
    assert np.abs(go - gn)[:, 40:220].max() < 0.05


def test_crf_json_is_the_truth_form(set6, tmp_path) -> None:
    truth = CRF.from_dict(set6.crf_truth)  # the synthetic camera's file reads as a CRF
    assert truth.log_irradiance.shape == (3, 256)
    imgs, times = _brackets(set6, 3)
    crf = calibrate_crf(imgs, times)
    p = tmp_path / "crf.json"
    crf.save(p)
    d = json.loads(p.read_text())
    assert d["domain"] == "ldr8" and len(d["log_irradiance"]) == 3 and all(len(c) == 256 for c in d["log_irradiance"])
    back = CRF.load(p)
    assert np.allclose(back.log_irradiance, crf.log_irradiance)
    assert back.meta["anchor"] == "g(128)=0"
    assert crf.response().shape == (256, 1, 3) and crf.response().dtype == np.float32
    # tools/eval reads it: apply_crf(exp(g(z)) at t = 1) gives z back
    from hdr_metrics import apply_crf

    z = np.arange(30, 230)
    lin = np.exp(crf.log_irradiance[:, z]).T[None]
    assert np.abs(apply_crf(lin, d).astype(int)[0] - z[:, None]).max() <= 1


def test_crf_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        calibrate_crf([[np.zeros((4, 4, 3), np.uint8)]], [[0.01]])
    with pytest.raises(ValueError):
        CRF(np.zeros((2, 256)))
