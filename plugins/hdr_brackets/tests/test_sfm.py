"""SfM: only the missing-backend path is tested here.

A real registration test needs ≥ 12 views of a textured scene with parallax; procedural planar textures in a tmp dir
do not give pycolmap a reconstruction it can solve reliably, so the real check is M0's validation on
data/brackets_raw (the Blender room) — `python -m hdr_brackets.cli data/brackets_raw data/smoke_hdr` and its
Report's `registered / total` (PLAN.md M1: ≥ 95 %)."""

from __future__ import annotations

import importlib
import sys

import pytest


def test_sfm_imports_and_fails_clearly_without_pycolmap(tmp_path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "pycolmap", None)  # `import pycolmap` now raises ImportError
    import hdr_brackets.sfm as sfm

    sfm = importlib.reload(sfm)  # the module itself imports fine without it
    (tmp_path / "images").mkdir()
    with pytest.raises(sfm.SfmUnavailable, match="pycolmap"):
        sfm.run_sfm(tmp_path)
