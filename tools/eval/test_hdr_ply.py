"""HDR PLY round trips through the built binary (PLAN.md M5 deliverable 6; docs/hdr.md).

Needs `build/LichtFeld-Studio` (or LFS_BINARY) — the tests are skipped without it. Each test writes a random
log-radiance PLY with numpy, converts it with `LichtFeld-Studio convert` and reads the result back with numpy:

  (a) log -> linear -> log: degree 0 exact (1e-5), degree 3 bounded relative radiance error over directions
  (b) the linear PLY's 0.5 + C0 f_dc is exp of the internal sh0 (degree 0: exactly; degree 3: the DC of the
      projected expansion, equal to exp of the log expansion's mean over the sphere within the projection error)
  (c) the HDR comments survive the round trip
  (d) a baked LDR PLY has no HDR header and its colours lie in [0, 1]
  (e) SPZ refuses an HDR model unless it is baked
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hdr_ply  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
BINARY = Path(os.environ.get("LFS_BINARY", REPO / "build" / "LichtFeld-Studio"))
pytestmark = pytest.mark.skipif(not BINARY.exists(), reason=f"no binary at {BINARY}")

SCALE = 2421.4
CAP = 1000.0 * SCALE


def make_log_ply(path: Path, n: int, degree: int, lobe_sigma: float, seed: int = 1) -> np.ndarray:
    """A random HDR model in LichtFeld's internal log convention, stamped `lichtfeld_hdr log`."""
    rng = np.random.default_rng(seed)
    K = {0: 0, 1: 3, 2: 8, 3: 15}[degree]
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
    fields += [(f"f_dc_{i}", "<f4") for i in range(3)]
    fields += [(f"f_rest_{i}", "<f4") for i in range(3 * K)]
    fields += [("opacity", "<f4")] + [(f"scale_{i}", "<f4") for i in range(3)] + [(f"rot_{i}", "<f4") for i in range(4)]
    data = np.zeros(n, dtype=np.dtype(fields))
    for c in "xyz":
        data[c] = rng.uniform(-1, 1, n)
    # log radiance of the DC: 0.1 .. 1000 x the scale's tenth, i.e. a dark room to a lamp
    log_r = rng.uniform(np.log(0.1), np.log(1000.0), (n, 3)) + np.log(SCALE / 10.0)
    dc = (log_r - 0.5) / hdr_ply.C0
    for i in range(3):
        data[f"f_dc_{i}"] = dc[:, i]
    if K:
        rest = rng.normal(0.0, lobe_sigma, (n, K, 3))  # log-space lobes, O(1) as PLAN.md M2 describes them
        flat = rest.transpose(0, 2, 1).reshape(n, 3 * K)
        for i in range(3 * K):
            data[f"f_rest_{i}"] = flat[:, i]
    data["opacity"] = rng.uniform(-2, 4, n)
    for i in range(3):
        data[f"scale_{i}"] = rng.uniform(-6, -3, n)
    data["rot_0"] = 1.0
    hdr_ply.write_ply(path, data, [f"lichtfeld_hdr log", f"hdr_scale {SCALE!r}", f"hdr_color_max {CAP!r}"])
    return data


def convert(src: Path, dst: Path, *extra: str) -> subprocess.CompletedProcess:
    cmd = [str(BINARY), "convert", str(src), str(dst), "-y", *extra]
    return subprocess.run(cmd, capture_output=True, text=True)


def expect_ok(res: subprocess.CompletedProcess) -> None:
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-2000:]


@pytest.mark.parametrize("degree", [0, 3])
def test_comments_round_trip(tmp_path: Path, degree: int):
    src = tmp_path / "log.ply"
    make_log_ply(src, 512, degree, 0.3)
    lin = tmp_path / "linear.ply"
    expect_ok(convert(src, lin))
    lines, hdr, _ = hdr_ply.read_ply(lin)
    assert hdr["mode"] == "linear"
    assert hdr["scale"] == pytest.approx(SCALE, rel=1e-6)
    assert hdr["color_max"] == pytest.approx(CAP, rel=1e-6)
    assert hdr["display_exposure"] == pytest.approx(1.0 / SCALE, rel=1e-5)
    back = tmp_path / "log_again.ply"
    expect_ok(convert(lin, back, "--hdr-export", "log"))
    _, hdr2, _ = hdr_ply.read_ply(back)
    assert hdr2["mode"] == "log"
    assert hdr2["scale"] == pytest.approx(SCALE, rel=1e-6)
    assert hdr2["color_max"] == pytest.approx(CAP, rel=1e-6)


def test_degree0_round_trip_exact(tmp_path: Path):
    src = tmp_path / "log.ply"
    data = make_log_ply(src, 2048, 0, 0.0)
    lin = tmp_path / "linear.ply"
    expect_ok(convert(src, lin))
    _, hdr, dlin = hdr_ply.read_ply(lin)
    dc_log = np.stack([data[f"f_dc_{i}"] for i in range(3)], -1).astype(np.float64)
    dc_lin = np.stack([dlin[f"f_dc_{i}"] for i in range(3)], -1).astype(np.float64)
    # (b): 0.5 + C0 f_dc(linear) == exp(0.5 + C0 sh0(log)), to float32 precision in relative terms
    radiance_lin = 0.5 + hdr_ply.C0 * dc_lin
    radiance_log = np.exp(np.minimum(0.5 + hdr_ply.C0 * dc_log, np.log(CAP)))
    rel = np.abs(radiance_lin - radiance_log) / radiance_log
    assert rel.max() < 1e-5, rel.max()
    # (a): back to log, the coefficients themselves agree to 1e-5 (absolute, log units are O(10))
    back = tmp_path / "log_again.ply"
    expect_ok(convert(lin, back, "--hdr-export", "log"))
    _, _, dback = hdr_ply.read_ply(back)
    dc_back = np.stack([dback[f"f_dc_{i}"] for i in range(3)], -1).astype(np.float64)
    assert np.abs(dc_back - dc_log).max() < 1e-5, np.abs(dc_back - dc_log).max()


@pytest.mark.parametrize("sigma,bound_p99,bound_median", [(0.1, 0.03, 0.006), (0.3, 0.30, 0.04)])
def test_degree3_round_trip_bounded(tmp_path: Path, sigma: float, bound_p99: float, bound_median: float):
    """exp of a degree-3 expansion is not degree 3: the projection loses what lies above degree 3. The loss grows
    with the lobe amplitude; measured 2026-10-28 on 2048 random splats x 1024 directions: sigma 0.1 -> median 0.29 %,
    p99 2.0 %; sigma 0.3 (15 independent lobes, i.e. exp swings of x3 over the sphere) -> median 2.6 %, p99 24 %.
    The bounds below are those numbers with margin; a real model (`hdr_roundtrip.py`) sits nearer the first."""
    src = tmp_path / "log.ply"
    make_log_ply(src, 2048, 3, sigma)
    lin = tmp_path / "linear.ply"
    expect_ok(convert(src, lin))
    _, hsrc, dsrc = hdr_ply.read_ply(src)
    _, hlin, dlin = hdr_ply.read_ply(lin)
    dirs = hdr_ply.random_directions(1024, 7)
    ra = hdr_ply.radiance(dsrc, hsrc, dirs)
    rb = hdr_ply.radiance(dlin, hlin, dirs)
    q = hdr_ply.rel_error_quantiles(ra, rb, 1e-3)
    print(f"degree 3, sigma {sigma}: linear-vs-log radiance relative error {q}")
    assert q["median"] < bound_median, q
    assert q["p99"] < bound_p99, q
    # DC of the linear file = mean radiance over the sphere of the log model (within the projection error)
    dc_lin = 0.5 + hdr_ply.C0 * np.stack([dlin[f"f_dc_{i}"] for i in range(3)], -1)
    mean_log = ra.mean(axis=1)
    rel_dc = np.abs(dc_lin - mean_log) / mean_log
    assert np.percentile(rel_dc, 99) < bound_p99, np.percentile(rel_dc, 99)
    # and back to log: the second projection must not drift further than the first
    back = tmp_path / "log_again.ply"
    expect_ok(convert(lin, back, "--hdr-export", "log"))
    _, hback, dback = hdr_ply.read_ply(back)
    rc = hdr_ply.radiance(dback, hback, dirs)
    q2 = hdr_ply.rel_error_quantiles(ra, rc, 1e-3)
    print(f"degree 3, sigma {sigma}: log->linear->log radiance relative error {q2}")
    assert q2["p99"] < 2.0 * bound_p99, q2


@pytest.mark.parametrize("degree", [0, 3])
def test_baked_ldr_in_unit_range(tmp_path: Path, degree: int):
    src = tmp_path / "log.ply"
    make_log_ply(src, 2048, degree, 0.3)
    for tone in ("linear", "aces", "neutral"):
        baked = tmp_path / f"baked_{tone}.ply"
        expect_ok(convert(src, baked, "--hdr-export", "baked", "--hdr-exposure", "0", "--hdr-tonemap", tone))
        lines, hdr, data = hdr_ply.read_ply(baked)
        assert hdr["mode"] is None, lines
        assert not any(ln.startswith("comment hdr_") for ln in lines)
        dc, rest, _ = hdr_ply.sh_columns(data)
        dc_col = 0.5 + hdr_ply.C0 * dc
        assert dc_col.min() >= -1e-4 and dc_col.max() <= 1.0 + 1e-4, (tone, dc_col.min(), dc_col.max())
        if degree:
            c = hdr_ply.expansion(dc, rest, hdr_ply.random_directions(256, 3))
            # The DC is in [0,1]; the degree-3 fit of a CLIPPED curve rings (Gibbs): measured 2026-10-28 with
            # sigma-0.3 log lobes, the hard-clipping `linear` operator overshoots to 1.12, aces / neutral less
            # (printed below). Viewers clamp the evaluated colour, so the overshoot costs nothing on screen.
            print(f"baked degree 3, {tone}: expansion range [{c.min():.4f}, {c.max():.4f}]")
            assert c.min() > -0.15 and c.max() < 1.15, (tone, c.min(), c.max())
    # a different exposure moves the colours
    darker = tmp_path / "baked_dark.ply"
    expect_ok(convert(src, darker, "--hdr-export", "baked", "--hdr-exposure", "-3"))
    _, _, d0 = hdr_ply.read_ply(tmp_path / "baked_aces.ply")
    _, _, d1 = hdr_ply.read_ply(darker)
    assert float(np.mean(d1["f_dc_0"])) < float(np.mean(d0["f_dc_0"]))


def test_spz_refuses_unless_baked(tmp_path: Path):
    src = tmp_path / "log.ply"
    make_log_ply(src, 256, 1, 0.2)
    res = convert(src, tmp_path / "out.spz")
    assert res.returncode != 0
    assert "bake" in (res.stdout + res.stderr).lower()
    expect_ok(convert(src, tmp_path / "out_baked.spz", "--hdr-export", "baked"))
    assert (tmp_path / "out_baked.spz").stat().st_size > 0


def test_ldr_ply_untouched(tmp_path: Path):
    """A PLY without the header keeps today's behaviour: converted bit for bit (no HDR comments, same values)."""
    src = tmp_path / "ldr.ply"
    data = make_log_ply(src, 256, 1, 0.2)
    hdr_ply.write_ply(src, data)  # no comments
    out = tmp_path / "ldr_out.ply"
    expect_ok(convert(src, out))
    lines, hdr, d = hdr_ply.read_ply(out)
    assert hdr["mode"] is None
    for name in ("f_dc_0", "f_rest_0", "opacity"):
        assert np.array_equal(d[name], data[name])
