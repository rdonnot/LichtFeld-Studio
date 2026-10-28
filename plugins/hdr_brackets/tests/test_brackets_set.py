"""brackets/ — the bracket training set (docs/hdr-m3-design.md §1) written from a 3-view set and a hand-made text model.

The SfM model is written by hand (pycolmap cannot register a procedural planar set, test_sfm.py), with the image ids
deliberately NOT in name order (the real model has id 1 = pos_004.png), and the pipeline runs with `reuse_sfm` so
the whole path — reuse, write, report — is the one the real dataset takes."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from hdr_metrics import apply_crf  # tools/eval (conftest puts it on sys.path)
from PIL import Image

from hdr_brackets import brackets_set as bs
from hdr_brackets import cli
from hdr_brackets._tools import read_exr
from hdr_brackets.exif import read_exposure
from hdr_brackets.pipeline import Options, run

#: image id → proxy name: scrambled on purpose.
MODEL_IMAGES = {1: "pos_002.png", 2: "pos_000.png", 3: "pos_001.png"}
POSES = {
    1: "0.70710678118654757 0.70710678118654757 0 0 1.5 -0.25 3.0",
    2: "1 0 0 0 0.125 0.5 2.0",
    3: "0.5 0.5 0.5 0.5 -2.0 1.0 0.33333333333333331",
}
POINTS2D = {
    1: "10.5 20.25 1 30 40 -1 50.125 60 2",
    2: "11 21 1 31 41 -1",
    3: "12 22 1",
}
POINTS3D = ["1 0.1 0.2 3.0 200 100 50 0.4 1 0 2 0 3 0", "2 -1 0.5 2.5 10 20 30 0.9 1 2"]


def write_fake_model(sparse: Path, images: dict[int, str] | None = None) -> None:
    images = images or MODEL_IMAGES
    sparse.mkdir(parents=True, exist_ok=True)
    (sparse / "cameras.txt").write_text("# cameras\n1 SIMPLE_PINHOLE 96 64 80 48 32\n")
    lines = ["# images"]
    for i, name in images.items():
        lines += [f"{i} {POSES[i]} 1 {name}", POINTS2D[i]]
    (sparse / "images.txt").write_text("\n".join(lines) + "\n")
    (sparse / "points3D.txt").write_text("# points\n" + "\n".join(POINTS3D) + "\n")


@pytest.fixture(scope="module")
def run3b(set3, tmp_path_factory):
    out = tmp_path_factory.mktemp("out3b")
    write_fake_model(out / "sparse" / "0")
    calls: list[tuple[str, float]] = []
    report = run(set3.raw_dir, out, Options(run_sfm=True, reuse_sfm=True), lambda label, f: calls.append((label, f)))
    return report, out, calls


def _records(images_txt: Path) -> list[bs.TextImage]:
    return bs.read_text_model(images_txt.parent).images


# ── Pipeline path ─────────────────────────────────────────────────────────────────────────────────────────────────────


def test_reused_sfm_and_report(run3b) -> None:
    report, out, calls = run3b
    assert report.registered == 3 and report.total == 3 and report.stats["sfm_reused"] is True
    assert report.stats["sfm_points"] == 2 and not any("SfM" in w for w in report.warnings)
    assert (out / "sparse/0/images.txt").is_file()  # the reused model survived the output cleaning
    b = report.brackets
    assert b and b["positions"] == 3 and b["frames"] == 9 and b["id_stride"] == 10 and b["omitted"] == []
    assert b["link"] == "symlink" and b["exif_ev_ref"] == pytest.approx(np.log2(100 / 16))
    assert Path(b["path"]) == out / "brackets"
    labels = [label for label, _ in calls]
    assert "SfM reused" in labels and "Brackets" in labels
    fracs = [f for _, f in calls]
    assert fracs == sorted(fracs) and fracs[-1] == pytest.approx(1.0)
    assert json.loads(report.to_json())["brackets"]["frames"] == 9


def test_every_frame_has_its_positions_pose_and_points(run3b, set3) -> None:
    _, out, _ = run3b
    recs = _records(out / "brackets/sparse/0/images.txt")
    assert len(recs) == 9  # Σ frames
    proxy_by_name = {v: k for k, v in MODEL_IMAGES.items()}
    seen_ids = set()
    for r in recs:
        pos, _, fk = r.name.rpartition("_")
        k = int(fk[1:-4])
        p = proxy_by_name[f"{pos}.png"]
        assert r.image_id == 10 * p + k and r.image_id not in seen_ids
        seen_ids.add(r.image_id)
        assert " ".join(r.pose) == POSES[p] and r.camera_id == 1
        assert r.points2d == POINTS2D[p]
    assert sorted(r.name for r in recs) == [f"pos_{p:03d}_f{k}.jpg" for p in range(3) for k in range(3)]
    # frames are ordered by exposure time ascending
    for p in range(3):
        times = [read_exposure(out / f"brackets/images/pos_{p:03d}_f{k}.jpg").exposure_time for k in range(3)]
        assert times == sorted(times) == pytest.approx([0.0025, 0.01, 0.04])
    cams = bs.read_text_model(out / "brackets/sparse/0").cameras
    assert len(cams) == 1 and cams[0].line() == "1 SIMPLE_PINHOLE 96 64 80 48 32"


def test_tracks_remapped_to_frame_zero(run3b) -> None:
    _, out, _ = run3b
    model = bs.read_text_model(out / "brackets/sparse/0")
    ids = {im.image_id for im in model.images}
    assert len(model.points3d) == 2
    tracks = [ln.split()[8:] for ln in model.points3d]
    assert tracks[0] == ["10", "0", "20", "0", "30", "0"] and tracks[1] == ["10", "2"]
    for tr in tracks:
        for j in range(0, len(tr), 2):
            assert int(tr[j]) in ids and int(tr[j]) % 10 == 0
    # the geometry columns are untouched
    assert model.points3d[0].startswith("1 0.1 0.2 3.0 200 100 50 0.4 ")


def test_linked_jpeg_keeps_its_exif(run3b, set3) -> None:
    _, out, _ = run3b
    link = out / "brackets/images/pos_000_f2.jpg"
    assert link.is_symlink() and not os.path.isabs(os.readlink(link))
    idx = json.loads((out / "brackets/exposures.json").read_text())
    src = next(s for s in idx["positions"][0]["sources"] if s["frame"] == "pos_000_f2.jpg")
    assert link.resolve() == (set3.raw_dir / src["file"]).resolve()
    info = read_exposure(link)
    assert info.exposure_time == pytest.approx(src["exposure_time"]) and info.iso == 100 and info.fnumber == 4.0
    assert info.ev == pytest.approx(src["exif_ev"])
    with Image.open(link) as im:  # the APP1 segment is the raw file's, byte for byte
        assert im.info["exif"] == Image.open(set3.raw_dir / src["file"]).info["exif"]


def test_crf_json_carries_the_anchor(run3b) -> None:
    _, out, _ = run3b
    crf = json.loads((out / "brackets/crf.json").read_text())
    base = json.loads((out / "crf.json").read_text())
    assert crf["log_irradiance"] == base["log_irradiance"] and crf["domain"] == "ldr8"
    assert crf["iso_ref"] == 100.0 and crf["fnumber_ref"] == 4.0
    assert crf["exif_ev_ref"] == pytest.approx(2.643856, abs=1e-6)
    assert "code 128 after 1 s at iso_ref, fnumber_ref" in crf["units"]


def test_exposures_json_names_frames(run3b) -> None:
    _, out, _ = run3b
    idx = json.loads((out / "brackets/exposures.json").read_text())
    base = json.loads((out / "exposures.json").read_text())
    assert idx["version"] == base["version"] and idx["source_dir"] == base["source_dir"]
    assert [p["name"] for p in idx["positions"]] == ["pos_000", "pos_001", "pos_002"]
    for p in idx["positions"]:
        assert p["hdr"].startswith("../images_hdr/") and p["proxy"].startswith("../images/")
        assert [s["frame"] for s in p["sources"]] == [f"{p['name']}_f{k}.jpg" for k in range(3)]
        assert [s["exposure_time"] for s in p["sources"]] == sorted(s["exposure_time"] for s in p["sources"])
    b = idx["brackets"]
    assert b["id_stride"] == 10 and b["link"] == "symlink" and b["omitted"] == []
    assert b["exif_ev_ref"] == pytest.approx(np.log2(100 / 16)) and b["sparse"] == "sparse/0"


def test_lut_round_trips_each_frame(run3b, set3) -> None:
    """The unit check of design §8 risk 1: EXR × 2^(exif_ev − exif_ev_ref) through crf.json IS the JPEG (median
    |Δ| ≤ 3/255 on unsaturated pixels), and without the anchor it is 6.25× off and visibly wrong."""
    _, out, _ = run3b
    crf = json.loads((out / "brackets/crf.json").read_text())
    idx = json.loads((out / "brackets/exposures.json").read_text())
    ref = crf["exif_ev_ref"]
    checked = 0
    for p in idx["positions"]:
        hdr = read_exr(out / "brackets" / p["hdr"])
        for s in p["sources"]:
            jpg = np.asarray(Image.open(out / "brackets/images" / s["frame"]).convert("RGB")).astype(int)
            mask = np.all((jpg > 5) & (jpg < 250), axis=2)
            assert mask.mean() > 0.2, s["frame"]
            pred = apply_crf(hdr * 2.0 ** (s["exif_ev"] - ref), crf).astype(int)
            d = np.abs(pred - jpg)[mask]
            assert np.median(d) <= 3, (s["frame"], np.median(d))
            assert s["exif_ev"] - ref == pytest.approx(np.log2(s["exposure_time"]))
            wrong = apply_crf(hdr * 2.0 ** s["exif_ev"], crf).astype(int)  # anchor forgotten
            assert np.median(np.abs(wrong - jpg)[mask]) > 20
            checked += 1
    assert checked == 9


# ── write_brackets alone ──────────────────────────────────────────────────────────────────────────────────────────────


def _copy_dataset(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("brackets"), symlinks=True)
    return dst


def test_unregistered_and_flagged_positions_are_omitted(run3b, tmp_path: Path) -> None:
    _, out, _ = run3b
    d = _copy_dataset(out, tmp_path / "d")
    write_fake_model(d / "sparse" / "0", {1: "pos_002.png", 2: "pos_000.png"})  # pos_001 has no pose
    idx = json.loads((d / "exposures.json").read_text())
    idx["positions"][2]["flagged"] = True
    warnings: list[str] = []
    res = bs.write_brackets(d, warnings=warnings, index=idx)
    assert res is not None and res.positions == 1 and res.frames == 3 and res.omitted == ["pos_001", "pos_002"]
    assert any("pos_001" in w and "not registered" in w for w in warnings)
    assert any("pos_002" in w and "flagged" in w for w in warnings)
    recs = _records(d / "brackets/sparse/0/images.txt")
    assert sorted(r.image_id for r in recs) == [20, 21, 22]
    model = bs.read_text_model(d / "brackets/sparse/0")
    assert model.points3d[0].split()[8:] == ["20", "0"]  # the other views of the track are gone with their images
    assert model.points3d[1].split()[8:] == []
    assert json.loads((d / "brackets/exposures.json").read_text())["brackets"]["omitted"] == ["pos_001", "pos_002"]
    assert sorted(p.name for p in (d / "brackets/images").iterdir()) == [f"pos_000_f{k}.jpg" for k in range(3)]


def test_shifted_frame_gets_its_own_camera(run3b, tmp_path: Path) -> None:
    """MTB moved the raw frame by (dx, dy) onto the reference: its principal point is the shared one minus (dx, dy)."""
    _, out, _ = run3b
    d = _copy_dataset(out, tmp_path / "d")
    idx = json.loads((d / "exposures.json").read_text())
    idx["positions"][1]["sources"][2]["shift"] = [3, -2]
    warnings: list[str] = []
    res = bs.write_brackets(d, warnings=warnings, index=idx)
    assert res is not None and res.frames == 9 and warnings == []
    model = bs.read_text_model(d / "brackets/sparse/0")
    assert [c.line() for c in model.cameras] == [
        "1 SIMPLE_PINHOLE 96 64 80 48 32",
        "2 SIMPLE_PINHOLE 96 64 80 45.0 34.0",
    ]
    by_name = model.image_by_name()
    assert by_name["pos_001_f2.jpg"].camera_id == 2
    assert all(im.camera_id == 1 for n, im in by_name.items() if n != "pos_001_f2.jpg")
    assert by_name["pos_001_f2.jpg"].pose == by_name["pos_001_f0.jpg"].pose  # the pose stays the position's
    # an unknown camera model: the frame keeps the shared camera and the report says so
    (d / "sparse/0/cameras.txt").write_text("1 WEIRD_MODEL 96 64 80 48 32\n")
    warnings.clear()
    bs.write_brackets(d, warnings=warnings, index=idx)
    assert len(bs.read_text_model(d / "brackets/sparse/0").cameras) == 1
    assert any("pos_001_f2.jpg" in w and "WEIRD_MODEL" in w for w in warnings)


def test_stride_grows_with_more_than_ten_frames(run3b, tmp_path: Path) -> None:
    _, out, _ = run3b
    d = _copy_dataset(out, tmp_path / "d")
    idx = json.loads((d / "exposures.json").read_text())
    srcs = idx["positions"][0]["sources"]
    idx["positions"][0]["sources"] = [dict(srcs[k % 3], exposure_time=0.001 * (k + 1)) for k in range(12)]
    res = bs.write_brackets(d, index=idx)
    assert res is not None and res.id_stride == 12 and res.frames == 12 + 3 + 3
    recs = _records(d / "brackets/sparse/0/images.txt")
    ids = sorted(r.image_id for r in recs)
    assert len(ids) == len(set(ids)) == 18 and max(r.image_id for r in recs if r.name.startswith("pos_000")) == 24 + 11
    model = bs.read_text_model(d / "brackets/sparse/0")
    assert model.points3d[0].split()[8:] == ["12", "0", "24", "0", "36", "0"]
    assert json.loads((d / "brackets/exposures.json").read_text())["brackets"]["id_stride"] == 12


def test_copy_mode_and_cli_regenerate(run3b, tmp_path: Path, capsys) -> None:
    _, out, _ = run3b
    d = _copy_dataset(out, tmp_path / "d")
    assert bs.main([str(d), "--copy"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["brackets"]["link"] == "copy" and rep["brackets"]["frames"] == 9 and rep["warnings"] == []
    f = d / "brackets/images/pos_002_f0.jpg"
    assert f.is_file() and not f.is_symlink()
    idx = json.loads((d / "brackets/exposures.json").read_text())
    src = next(s for s in idx["positions"][2]["sources"] if s["frame"] == f.name)
    assert f.read_bytes() == (Path(idx["source_dir"]) / src["file"]).read_bytes()
    # a second write replaces the first (no stale frames)
    (d / "brackets/images/pos_009_f0.jpg").write_bytes(b"stale")
    assert bs.main([str(d)]) == 0
    assert not (d / "brackets/images/pos_009_f0.jpg").exists()
    assert (d / "brackets/images/pos_002_f0.jpg").is_symlink()


def test_binary_model_goes_through_pycolmap(run3b, tmp_path: Path) -> None:
    pycolmap = pytest.importorskip("pycolmap")
    _, out, _ = run3b
    d = _copy_dataset(out, tmp_path / "d")
    rec = pycolmap.Reconstruction(str(d / "sparse/0"))
    shutil.rmtree(d / "sparse/0")
    (d / "sparse/0").mkdir()
    rec.write_binary(str(d / "sparse/0"))
    assert (d / "sparse/0/images.bin").is_file() and not (d / "sparse/0/images.txt").exists()
    res = bs.write_brackets(d)
    assert res is not None and res.frames == 9
    a = _records(out / "brackets/sparse/0/images.txt")
    b = _records(d / "brackets/sparse/0/images.txt")
    assert [(r.image_id, r.name, r.camera_id) for r in a] == [(r.image_id, r.name, r.camera_id) for r in b]
    for ra, rb in zip(a, b, strict=True):  # COLMAP re-prints the numbers; the values are the same
        assert [float(x) for x in ra.pose] == pytest.approx([float(x) for x in rb.pose], abs=1e-12)
        assert [float(x) for x in ra.points2d.split()] == pytest.approx([float(x) for x in rb.points2d.split()])


def test_no_model_means_no_brackets_folder(set3, tmp_path: Path, monkeypatch) -> None:
    import sys

    report = run(set3.raw_dir, tmp_path / "o", Options(run_sfm=False))  # SfM off: the user knows there are no poses
    assert report.brackets is None and not (tmp_path / "o/brackets").exists() and report.warnings == []
    monkeypatch.setitem(sys.modules, "pycolmap", None)  # SfM wanted but it left no model: that is a warning
    report = run(set3.raw_dir, tmp_path / "o2", Options(run_sfm=True))
    assert report.brackets is None and any(w.startswith("brackets/: not written") for w in report.warnings)
    report = run(set3.raw_dir, tmp_path / "o3", Options(run_sfm=True, write_brackets=False))
    assert report.brackets is None and not any(w.startswith("brackets/") for w in report.warnings)


def test_cli_flags(set3, tmp_path: Path, capsys) -> None:
    write_fake_model(tmp_path / "c/sparse/0")
    rc = cli.main([str(set3.raw_dir), str(tmp_path / "c"), "--reuse-sfm"])
    rep = json.loads(capsys.readouterr().out)
    assert rc == 0 and rep["registered"] == 3 and rep["brackets"]["frames"] == 9
    rc = cli.main([str(set3.raw_dir), str(tmp_path / "c"), "--reuse-sfm", "--no-brackets"])
    rep = json.loads(capsys.readouterr().out)
    assert rc == 0 and rep["brackets"] is None
