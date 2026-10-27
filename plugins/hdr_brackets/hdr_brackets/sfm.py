"""Structure from motion on the proxies with pycolmap: SIFT → matching → incremental mapping → <dataset>/sparse/0.

Why pycolmap (not the colmap CLI): it is a pip wheel the plugin's uv env installs, so the plugin needs no system
COLMAP. The calls are the ones proven on this machine with pycolmap 4.2.1 (N'Brain's kits/splat/poses_pycolmap.py):
`extract_features` with one shared camera (a bracket set is one lens), `match_exhaustive` up to 150 images, else
`match_sequential` with overlap 10 and no loop detection (no vocabulary tree to download), `incremental_mapping`,
the reconstruction that registered the most images written as a binary model to `<dataset>/sparse/0` — the layout
LichtFeld trains from. pycolmap is imported lazily: this module imports without it, `run_sfm` raises SfmUnavailable.
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

EXHAUSTIVE_MAX = 150
SEQUENTIAL_OVERLAP = 10
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


class SfmUnavailable(RuntimeError):
    """pycolmap is not installed in this environment."""


@dataclass
class SfmResult:
    registered: int
    total: int
    sparse_dir: str | None
    points: int = 0
    models: int = 0
    matcher: str = ""
    registered_names: list[str] | None = None


def _import_pycolmap():
    try:
        import pycolmap  # noqa: PLC0415  (lazy: optional heavy dependency)
    except ImportError as e:
        raise SfmUnavailable(
            "pycolmap is not installed — `uv sync` in plugins/hdr_brackets (it is a dependency), or run with --no-sfm"
        ) from e
    return pycolmap


def run_sfm(
    dataset_dir: str | os.PathLike,
    images_subdir: str = "images",
    camera_model: str = "SIMPLE_PINHOLE",
    single_camera: bool = True,
    num_threads: int = -1,
    progress: Callable[[str, float], None] | None = None,
) -> SfmResult:
    """Register `<dataset>/<images_subdir>/*` and write the largest model to `<dataset>/sparse/0` (binary).

    Leaves `<dataset>/database.db`. `progress(label, fraction)` (0..1 of the SfM run) after each step. Raises
    SfmUnavailable without pycolmap, FileNotFoundError without images; a mapper that registers nothing is a result
    (registered = 0, sparse_dir None), not an exception."""
    pycolmap = _import_pycolmap()
    root = Path(dataset_dir)
    img_dir = root / images_subdir
    if not img_dir.is_dir():
        raise FileNotFoundError(f"no image folder {img_dir}")
    names = sorted(p.name for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if len(names) < 2:
        raise FileNotFoundError(f"{img_dir}: {len(names)} image(s), SfM needs 2 or more")

    def tick(label: str, f: float) -> None:
        if progress is not None:
            progress(label, f)

    db = root / "database.db"
    db.unlink(missing_ok=True)
    work = root / "sparse_work"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)

    # ── extract ──
    tick("SfM: features", 0.0)
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = camera_model
    ext = pycolmap.FeatureExtractionOptions()
    ext.num_threads = int(num_threads)
    mode = pycolmap.CameraMode.SINGLE if single_camera else pycolmap.CameraMode.AUTO
    pycolmap.extract_features(
        db,
        img_dir,
        image_names=names,
        camera_mode=mode,
        reader_options=reader,
        extraction_options=ext,
        device=pycolmap.Device.auto,
    )

    # ── match ──
    tick("SfM: matching", 0.3)
    mopt = pycolmap.FeatureMatchingOptions()
    mopt.num_threads = int(num_threads)
    if len(names) <= EXHAUSTIVE_MAX:
        matcher = "exhaustive"
        pycolmap.match_exhaustive(db, matching_options=mopt, device=pycolmap.Device.auto)
    else:
        matcher = "sequential"
        pair = pycolmap.SequentialPairingOptions()
        pair.overlap = SEQUENTIAL_OVERLAP
        pair.loop_detection = False
        pycolmap.match_sequential(db, matching_options=mopt, pairing_options=pair, device=pycolmap.Device.auto)

    # ── map ──
    tick("SfM: mapping", 0.6)
    iopt = pycolmap.IncrementalPipelineOptions()
    iopt.num_threads = int(num_threads)
    recs = pycolmap.incremental_mapping(db, img_dir, work, iopt)
    shutil.rmtree(work, ignore_errors=True)
    if not recs:
        log.warning("SfM registered no image of %d", len(names))
        tick("SfM: done", 1.0)
        return SfmResult(0, len(names), None, matcher=matcher)
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    sparse = root / "sparse" / "0"
    shutil.rmtree(root / "sparse", ignore_errors=True)
    sparse.mkdir(parents=True)
    if hasattr(rec, "write_binary"):
        rec.write_binary(str(sparse))
    else:  # older pycolmap: write() is binary
        rec.write(str(sparse))
    reg_names = sorted(im.name for im in rec.images.values() if getattr(im, "has_pose", True))
    res = SfmResult(
        registered=int(rec.num_reg_images()),
        total=len(names),
        sparse_dir=str(sparse),
        points=int(rec.num_points3D()),
        models=len(recs),
        matcher=matcher,
        registered_names=reg_names,
    )
    log.info("SfM: %d/%d registered, %d points, %d model(s)", res.registered, res.total, res.points, res.models)
    tick("SfM: done", 1.0)
    return res
