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
