"""pipeline.run end to end (no SfM) on a generated 3-view set, its failure handling, and the CLI."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hdr_brackets import cli
from hdr_brackets._tools import read_exr
from hdr_brackets.dataset import read_proxy
from hdr_brackets.pipeline import STAGES, Options, run
from hdr_brackets.response import CRF, compare_crf


@pytest.fixture(scope="module")
def run3(set3, tmp_path_factory):
    out = tmp_path_factory.mktemp("out3")
    calls: list[tuple[str, float]] = []
    report = run(set3.raw_dir, out, Options(run_sfm=False), lambda label, f: calls.append((label, f)))
    return report, out, calls


def test_run_end_to_end(set3, run3) -> None:
    report, out, _ = run3
    assert report.n_images == 9 and report.n_brackets == 3 and report.n_singletons == 0
    assert report.n_positions == 3 and report.flagged == [] and report.warnings == []
    assert report.registered == 0 and report.total == 3 and report.out_dir == str(out)
    assert Path(report.crf_path) == out / "crf.json" and report.seconds > 0
    idx = json.loads((out / "exposures.json").read_text())
    truth_views = {r["file"]: r["view_index"] for r in set3.images()}
    for k, pos in enumerate(idx["positions"]):
        assert pos["name"] == f"pos_{k:03d}"
        assert {truth_views[s["file"]] for s in pos["sources"]} == {k}  # one view per position
        assert [s["exposure_time"] for s in pos["sources"]] == pytest.approx([0.0025, 0.01, 0.04])
        hdr = read_exr(out / pos["hdr"])
        proxy = read_proxy(out / pos["proxy"])
        assert hdr.shape == (64, 96, 3) and proxy.shape == (64, 96, 3) and proxy.dtype == np.uint8
        assert np.isfinite(hdr).all() and (hdr > 0).all()
        truth = set3.truth(k)
        rel = np.abs(hdr * np.exp(np.median(np.log(truth / hdr))) - truth) / truth
        assert np.median(rel) < 0.05
    # one shared proxy scale (photometrically consistent proxies)
    assert len({p["proxy_scale"] for p in idx["positions"]}) == 1
    assert json.loads((out / "brackets.json").read_text())["brackets"][0]["position_index"] == 0


def test_index_ev_words_and_saturation(set3, run3) -> None:
    """exif_ev / ev_step mean what exposures_gt.json's exif_ev / ev mean; saturated counts the shortest frame."""
    _, out, _ = run3
    idx = json.loads((out / "exposures.json").read_text())
    truth = {r["file"]: r for r in set3.images()}
    for pos in idx["positions"]:
        for s in pos["sources"]:
            assert s["exif_ev"] == pytest.approx(truth[s["file"]]["exif_ev"], abs=1e-6)
            assert s["ev_step"] == pytest.approx(truth[s["file"]]["ev"], abs=1e-6)
            assert s["align"] in ("reference", "mtb", "clipped", "failed")
        darkest = Image.open(set3.raw_dir / pos["sources"][0]["file"]).convert("RGB")
        assert pos["saturated"] == int(np.any(np.asarray(darkest) >= 250, axis=2).sum())
    assert idx["stats"]["saturated_total"] == sum(p["saturated"] for p in idx["positions"])
    assert idx["stats"]["shift_max"] == 0


def test_five_ev_set_end_to_end(set5ev, tmp_path: Path) -> None:
    """The camera's default −4…+4 EV with one view whose mid frame is clipped: no shift anywhere, the CRF and the
    merge stay accurate (on the 5-EV room the old code shifted 37 frames, up to 31 px, and the CRF was off 0.395)."""
    report = run(set5ev.raw_dir, tmp_path / "o", Options(run_sfm=False))
    assert report.n_positions == 4 and report.flagged == []
    assert report.stats["shift_max"] == 0 and report.stats["align_failed"] == 0
    assert report.stats["align_clipped"] >= 5  # every +4 EV frame, and the bright view's 0 EV frame
    assert report.stats["frames_clipped_over_90"] >= 1
    assert any("clipped" in w and "%" in w for w in report.warnings)
    assert not any("MTB search" in w for w in report.warnings)
    idx = json.loads((tmp_path / "o/exposures.json").read_text())
    assert all(s["shift"] == [0, 0] for p in idx["positions"] for s in p["sources"])
    crf = CRF.load(tmp_path / "o/crf.json")
    assert crf.meta["frames"] == "raw" and len(crf.meta["bracket_ids"]) == 4
    err = compare_crf(crf, CRF.from_dict(set5ev.crf_truth))
    assert err < 0.2, err
    gt = {r["file"]: r for r in set5ev.images()}
    for pos in idx["positions"]:
        vi = gt[pos["sources"][0]["file"]]["view_index"]
        zero = next(s for s in pos["sources"] if gt[s["file"]]["ev"] == 0.0)
        z0 = np.asarray(Image.open(set5ev.raw_dir / zero["file"]).convert("RGB"))
        mask = np.all((z0 > 5) & (z0 < 250), axis=2)
        if mask.mean() < 0.05:
            continue  # the bright view's 0 EV frame is clipped almost everywhere
        hdr, truth = read_exr(tmp_path / "o" / pos["hdr"])[mask], set5ev.truth(vi)[mask]
        rel = np.abs(hdr * np.exp(np.median(np.log(truth / hdr))) - truth) / truth
        assert np.median(rel) < 0.05, (pos["name"], np.median(rel))


def test_iso_or_aperture_change_is_warned(set3, tmp_path: Path) -> None:
    from hdr_brackets.exif import ExposureInfo
    from hdr_brackets.grouping import Bracket
    from hdr_brackets.pipeline import setting_warnings

    def info(name: str, iso: float, n: float) -> ExposureInfo:
        return ExposureInfo(f"/x/{name}", 0.01, n, iso, None, None, 8, 8)

    same = Bracket(0, [info("a", 100, 4.0), info("b", 100, 4.0)], 0)
    inside = Bracket(1, [info("c", 100, 4.0), info("d", 400, 4.0)], 1)
    other = Bracket(2, [info("e", 100, 5.6), info("f", 100, 5.6)], 2)
    assert setting_warnings([same]) == []
    w = setting_warnings([same, inside])
    assert len(w) == 1 and "ISO varies inside 1 bracket" in w[0] and "bracket 1: 100, 400" in w[0]
    w = setting_warnings([same, other])
    assert len(w) == 1 and "f-number varies across the shoot" in w[0] and "f/4, f/5.6" in w[0]


def test_progress_reported_per_stage_and_bracket(run3) -> None:
    _, _, calls = run3
    fracs = [f for _, f in calls]
    assert fracs == sorted(fracs) and fracs[-1] == pytest.approx(1.0) and all(0 <= f <= 1 for f in fracs)
    labels = [label for label, _ in calls]
    for stage in ("Reading EXIF", "Grouping brackets", "Calibrating CRF", "Proxies", "Writing dataset"):
        assert stage in labels
    assert "SfM skipped" in labels and set(STAGES) >= {"Aligning", "Merging"}
    for k in (1, 2, 3):
        assert f"Aligning bracket {k}/3" in labels and f"Merging bracket {k}/3" in labels


def test_bad_brackets_are_skipped_not_raised(set3, tmp_path: Path) -> None:
    src = tmp_path / "src"
    shutil.copytree(set3.raw_dir, src)
    (src / "IMG_9999.jpg").write_bytes(b"not a jpeg")  # unreadable file
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(src / "stray.png")  # no EXIF at all
    with Image.open(src / "IMG_0005.jpg") as im:  # bracket 1: a frame of another size
        exif = im.info["exif"]
        small = im.resize((48, 32))
    small.save(src / "IMG_0005.jpg", "JPEG", quality=95, exif=exif)
    report = run(src, tmp_path / "out", Options(run_sfm=False))
    assert report.n_positions == 2 and report.n_brackets == 3 and report.n_images == 10
    text = " | ".join(report.warnings)
    assert "IMG_9999.jpg" in text and "stray.png" in text and "differ in size" in text
    idx = json.loads((tmp_path / "out" / "exposures.json").read_text())
    assert [p["bracket_id"] for p in idx["positions"]] == [0, 2]


def test_saved_crf_is_reused(set3, run3, tmp_path: Path) -> None:
    _, out, _ = run3
    report = run(set3.raw_dir, tmp_path / "again", Options(run_sfm=False, crf_path=str(out / "crf.json")))
    assert report.n_positions == 3 and not report.warnings
    a = read_exr(out / "images_hdr/pos_001.exr")
    b = read_exr(tmp_path / "again/images_hdr/pos_001.exr")
    assert np.allclose(a, b, rtol=1e-6)


def test_missing_sfm_backend_is_a_warning(set3, tmp_path: Path, monkeypatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "pycolmap", None)
    report = run(set3.raw_dir, tmp_path / "o", Options(run_sfm=True))
    assert report.n_positions == 3 and report.registered == 0
    assert any("pycolmap" in w for w in report.warnings)


def test_cli(set3, tmp_path: Path, capsys) -> None:
    rc = cli.main([str(set3.raw_dir), str(tmp_path / "cli"), "--no-sfm", "--proxy-ev", "0.5", "--max-gap", "5"])
    assert rc == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["n_positions"] == 3 and rep["n_brackets"] == 3 and rep["registered"] == 0
    assert json.loads((tmp_path / "cli/exposures.json").read_text())["proxy_ev"] == 0.5
