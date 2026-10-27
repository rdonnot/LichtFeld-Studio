"""Merged radiance against the truth EXR, exposure consistency, the deghost path and the plain (OpenCV) path."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from hdr_brackets.align import align_bracket
from hdr_brackets.merge import merge_bracket, merge_debevec_numpy, merge_stats
from hdr_brackets.pipeline import load_rgb8
from hdr_brackets.response import CRF, calibrate_crf

log = logging.getLogger(__name__)


def _bracket(bs, vi: int):
    rows = sorted(bs.view(vi), key=lambda r: r["exposure_time"])
    return [load_rgb8(bs.raw_dir / r["file"]) for r in rows], [r["exposure_time"] for r in rows]


@pytest.fixture(scope="module")
def recovered(set6) -> CRF:
    data = [_bracket(set6, v) for v in range(6)]
    return calibrate_crf([d[0] for d in data], [d[1] for d in data])


def _errors(hdr: np.ndarray, truth: np.ndarray, imgs: list[np.ndarray]) -> tuple[float, float, float, float]:
    """One global scale fitted in log over the pixels the 0 EV frame records well ([10, 245] in every channel), then
    the relative error there (median, p99) and over every pixel the bracket covers (the −2 EV frame below 245 and the
    +2 EV frame above 10 in every channel: neither clipped nor crushed everywhere)."""
    z0 = imgs[1]
    fit = np.all((z0 >= 10) & (z0 <= 245), axis=2)
    covered = np.all((imgs[0] < 245) & (imgs[-1] > 10), axis=2)
    k = float(np.exp(np.mean(np.log(truth[fit]) - np.log(hdr[fit]))))
    rel = np.abs(hdr * k - truth) / truth
    return (
        float(np.median(rel[fit])),
        float(np.percentile(rel[fit], 99)),
        float(np.median(rel[covered])),
        float(np.percentile(rel[covered], 99)),
    )


@pytest.mark.parametrize("deghost", [True, False])
@pytest.mark.parametrize("which", ["recovered", "truth"])
def test_merge_matches_truth(set6, recovered, deghost: bool, which: str) -> None:
    crf = recovered if which == "recovered" else CRF.from_dict(set6.crf_truth)
    worst = (0.0, 0.0, 0.0, 0.0)
    for vi in range(6):
        imgs, times = _bracket(set6, vi)
        aligned, _ = align_bracket(imgs)
        hdr = merge_bracket(aligned, times, crf, deghost=deghost)
        assert hdr.dtype == np.float32 and hdr.shape == imgs[0].shape
        assert not np.isnan(hdr).any() and np.all(hdr > 0)
        e = _errors(hdr, set6.truth(vi), imgs)
        worst = tuple(max(a, b) for a, b in zip(worst, e, strict=True))
    log.info("merge %s deghost=%s: worst median %.4f p99 %.4f (covered: %.4f / %.4f)", which, deghost, *worst)
    print(
        f"merge {which} deghost={deghost}: worst median {worst[0]:.4f} p99 {worst[1]:.4f} "
        f"(covered {worst[2]:.4f} / {worst[3]:.4f})"
    )
    assert worst[0] < 0.05 and worst[1] < 0.25
    assert worst[2] < 0.05 and worst[3] < 0.25


@pytest.mark.parametrize("which", ["recovered", "truth"])
def test_merge_consistent_across_exposures(set6, recovered, which: str) -> None:
    """The −2 EV frames alone and the +2 EV frames alone give the same radiance on the mid-tones both record."""
    crf = recovered if which == "recovered" else CRF.from_dict(set6.crf_truth)
    import make_brackets as mb

    ratios = []
    for vi in range(6):
        imgs, times = _bracket(set6, vi)
        dark = merge_bracket([imgs[0]], [times[0]], crf, deghost=False)
        bright = merge_bracket([imgs[2]], [times[2]], crf, deghost=False)
        # mid-tones chosen from the noise-free truth, not from the frames' own (noisy) values: selecting on the
        # observed levels biases the dark frame up and the bright one down (a 7 % "error" that is selection bias)
        lin = set6.truth(vi) * set6.meta["gain"]
        z_dark = mb.to_uint8(mb.crf_forward(np.clip(lin * times[0], 0, 1)))
        z_bright = mb.to_uint8(mb.crf_forward(np.clip(lin * times[2], 0, 1)))
        mid = np.all((z_dark >= 25) & (z_bright <= 230), axis=2)
        if mid.sum() < 50:
            continue
        ratios.append(float(np.median(bright[mid] / dark[mid])))
    assert len(ratios) >= 3
    print(f"bright/dark radiance ratio on mid-tones ({which} CRF): {[round(r, 4) for r in ratios]}")
    # shot + read noise on these levels is a few % a pixel, a median over hundreds of pixels well under 1 %; JPEG adds
    # a little. Measured: 0.2–0.6 % with either CRF (the recovered one's Δg cancels between the two levels it links).
    assert all(abs(r - 1.0) < 0.02 for r in ratios), ratios


def test_numpy_plain_path_equals_opencv(set6, recovered) -> None:
    imgs, times = _bracket(set6, 0)
    cv = merge_bracket(imgs, times, recovered, deghost=False)
    npy = merge_debevec_numpy(imgs, times, recovered, deghost=False)
    assert np.allclose(cv, npy, rtol=1e-4, atol=0)


def test_unweighted_pixels_take_the_bound(recovered) -> None:
    """Saturated (or black) in every frame and channel: no weight anywhere; the darkest (brightest) frame bounds it."""
    sat = np.full((2, 2, 3), 255, np.uint8)
    black = np.zeros((2, 2, 3), np.uint8)
    times = [0.0025, 0.01, 0.04]
    g = recovered.log_irradiance
    for frames, t_bound, z in (([sat] * 3, times[0], 255), ([black] * 3, times[2], 0)):
        for deghost in (False, True):
            hdr = merge_bracket(frames, times, recovered, deghost=deghost)
            assert np.all(np.isfinite(hdr))
            assert np.allclose(hdr[0, 0], np.exp(g[:, z] - np.log(t_bound)), rtol=1e-5)


def test_deghost_takes_the_reference_where_something_moved(set6, recovered) -> None:
    imgs, times = _bracket(set6, 3)
    truth = set6.truth(3)
    ghost = imgs[0].copy()
    h, w = ghost.shape[:2]
    box = (slice(h // 4, h // 2), slice(w // 3, w // 3 + w // 4))
    ghost[box] = 255 - ghost[box]  # something moved through the −2 EV frame
    frames = [ghost, imgs[1], imgs[2]]
    plain = merge_bracket(frames, times, recovered, deghost=False)
    clean = merge_bracket(frames, times, recovered, deghost=True)
    ref_ok = np.all((imgs[1][box] >= 5) & (imgs[1][box] <= 250), axis=2)
    k = float(np.exp(np.median(np.log(truth) - np.log(merge_bracket(imgs, times, recovered)))))
    err_plain = np.median(np.abs(plain[box] * k - truth[box])[ref_ok] / truth[box][ref_ok])
    err_clean = np.median(np.abs(clean[box] * k - truth[box])[ref_ok] / truth[box][ref_ok])
    assert err_clean < 0.1 < err_plain, (err_clean, err_plain)


def test_merge_stats() -> None:
    a = np.ones((4, 4, 3), np.float32)
    a[0, 0, 0] = np.nan
    a[1, 1, 1] = 9.0
    s = merge_stats(a)
    assert s["nan"] == 1 and s["inf"] == 0 and s["max"] == 9.0 and s["min"] == 1.0 and s["p50"] == 1.0
    assert set(s) >= {"min", "p1", "p50", "p99", "max", "nan"}


def test_merge_rejects_mismatched_input(recovered) -> None:
    with pytest.raises(ValueError):
        merge_bracket([np.zeros((4, 4, 3), np.uint8)], [0.01, 0.02], recovered)
    with pytest.raises(ValueError):
        merge_bracket([np.zeros((4, 4, 3), np.uint8), np.zeros((4, 5, 3), np.uint8)], [0.01, 0.02], recovered)
