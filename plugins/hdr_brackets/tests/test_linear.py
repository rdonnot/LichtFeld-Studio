"""hdr_brackets.linear: brackets of developed raws (linear EXR) merged without a CRF, ND frames by their stops."""

from __future__ import annotations

import json

import numpy as np
import pytest

from hdr_brackets import linear

H, W = 24, 32
TIMES = [1 / 1000, 1 / 60, 1 / 4]          # three plain frames
ND_T, ND_STOPS = 1 / 1000, 10              # the sun's frame: 1/1000 behind ND 10


def scene(seed: int) -> np.ndarray:
    """Radiance from 0.05 to ~3e5 (a 22-stop span), per channel."""
    rng = np.random.default_rng(seed)
    e = np.exp(rng.uniform(np.log(0.05), np.log(3e5), (H, W, 1))) * np.array([1.0, 0.8, 0.6])
    e[0, 0] = 1e9                           # clipped in every frame, ND included
    return e.astype(np.float32)


def write_capture(tmp, positions=2, white=1.0, with_attr=False, space="ACEScg"):
    cap = tmp / "cap"
    (cap / "exr" / "nd").mkdir(parents=True)
    rec = {"version": 1, "kind": "scan", "name": "t", "positions": []}
    truth = []
    for p in range(positions):
        e = scene(p)
        truth.append(e)
        frames = []
        for k, (t, nd) in enumerate([(t, 0) for t in TIMES] + [(ND_T, ND_STOPS)]):
            sub = "nd/" if nd else ""
            rel = f"{sub}p{p:03d}_f{k:02d}.cr3"
            x = np.minimum(e * t * 2.0 ** -nd, white).astype(np.float32)
            path = cap / "exr" / f"{sub}p{p:03d}_f{k:02d}.exr"
            linear.write_frame(path, x, space)
            if with_attr:
                import OpenEXR

                with OpenEXR.File(str(path)) as f:
                    hdr = dict(f.header()); px = f.channels()["RGB"].pixels
                hdr["white"] = float(white)
                OpenEXR.File(hdr, {"RGB": px}).write(str(path))
            frames.append({"file": rel, "exposure_time": t, "nd_stops": nd, "card": "/x"})
        rec["positions"].append({"index": p, "frames": frames})
    (cap / "tether_set.json").write_text(json.dumps(rec))
    return cap, truth


@pytest.mark.parametrize("with_attr", [False, True])
def test_merge_recovers_radiance_across_22_stops(tmp_path, with_attr):
    cap, truth = write_capture(tmp_path, with_attr=with_attr)
    out = tmp_path / "out"
    rep = linear.run(cap, out, run_sfm=False)
    assert rep["positions"] == 2 and rep["frames"] == 8 and rep["colorspace"] == "ACEScg"
    for p, e in enumerate(truth):
        got, header = linear.read_frame(out / "images_hdr" / f"pos_{p:03d}.exr")
        assert linear.header_colorspace(header) == "ACEScg"         # carried, converted by the trainer on load
        ok = np.ones((H, W), bool); ok[0, 0] = False
        rel = np.abs(got[ok] - e[ok]) / e[ok]
        assert float(rel.max()) < 1e-3                               # exact: the frames are noise-free
        assert got[0, 0, 0] > 1e6                                    # the clipped pixel is a lower bound, not black
    idx = json.loads((out / "exposures.json").read_text())
    assert idx["linear"] is True and idx["crf"] is None
    srcs = idx["positions"][0]["sources"]
    assert srcs[-1]["nd_stops"] == ND_STOPS and srcs[-1]["t_eff"] == pytest.approx(ND_T / 1024)
    # one clipped pixel is no plateau to measure: only a declared `white` can call it saturated
    assert idx["positions"][0]["saturated"] == (1 if with_attr else 0)
    assert (out / "images" / "pos_000.png").is_file()


@pytest.mark.parametrize("how", ["attr", "untagged"])
def test_aces2065_input_is_merged_then_written_as_acescg(tmp_path, how):
    """An AP0 developer (an in-house debayer writing ACES2065-1): the merge reads it, the dataset is ACEScg."""
    cap, truth = write_capture(tmp_path, positions=1, space="ACES2065-1")
    if how == "untagged":                       # no attribute at all: the caller says what the EXRs are
        import OpenEXR

        for path in (cap / "exr").rglob("*.exr"):
            with OpenEXR.File(str(path)) as f:
                hdr = {k: v for k, v in dict(f.header()).items() if k != "lichtfeld_colorspace"}
                px = f.channels()["RGB"].pixels
            OpenEXR.File(hdr, {"RGB": px}).write(str(path))
    out = tmp_path / "out"
    rep = linear.run(cap, out, run_sfm=False, input_space="ACES2065-1" if how == "untagged" else "ACEScg")
    assert rep["colorspace"] == "ACEScg" and rep["input_colorspace"].lower() == "aces2065-1"
    got, header = linear.read_frame(out / "images_hdr" / "pos_000.exr")
    assert linear.header_colorspace(header) == "ACEScg"
    want = np.maximum(truth[0] @ linear.AP0_TO_AP1.T, 0.0)
    ok = np.ones((H, W), bool); ok[0, 0] = False
    rel = np.abs(got[ok] - want[ok]) / np.maximum(want[ok], 1e-3 * want[ok].max())
    assert float(rel.max()) < 2e-3
    assert json.loads((out / "exposures.json").read_text())["colorspace"] == "ACEScg"


def test_ap0_is_known_by_its_chromaticities():
    ap0 = (0.7347, 0.2653, 0.0, 1.0, 0.0001, -0.077)
    assert linear.header_colorspace({"chromaticities": ap0}) == "ACES2065-1"
    assert linear.output_space("ACES2065-1") == "ACEScg" and linear.output_space("lin_srgb") == "lin_srgb"
    white = np.ones((1, 1, 3), np.float32)
    assert np.allclose(linear.to_output(white, "aces2065-1"), 1.0, atol=1e-6)   # same white: no adaptation


def test_missing_exr_is_named_not_fatal(tmp_path):
    cap, _ = write_capture(tmp_path)
    (cap / "exr" / "p001_f01.exr").unlink()
    rep = linear.run(cap, tmp_path / "out", run_sfm=False)
    assert rep["frames"] == 7
    assert any("p001_f01.cr3" in w for w in rep["warnings"])


def test_clip_level_is_measured_from_the_plateau():
    rgb = np.full((10, 10, 3), 0.3, np.float32)
    assert linear.clip_level(rgb, {}) is not None                     # flat = a plateau
    ramp = np.linspace(0, 0.5, 300, dtype=np.float32).reshape(10, 10, 3)
    assert linear.clip_level(ramp, {}) is None
    assert linear.clip_level(ramp, {"white": 0.9}) == pytest.approx(0.9)
