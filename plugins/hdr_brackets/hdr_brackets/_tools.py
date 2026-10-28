"""The one bridge to the repo's `tools/eval/hdr_metrics.py` (EXR I/O and HDR metrics).

Why a bridge: the plugin is installed by symlinking (or copying) `plugins/hdr_brackets/` into
`~/.lichtfeld/plugins/`. Through a symlink the repo's `tools/eval` is found from this file's resolved path and the
EXR reader/writer are the tools' own (one implementation for the eval path and the plugin). In a COPIED install the
repo is not there, so `read_exr` / `write_exr` fall back to a verbatim copy of the two functions (scanline float32 RGB,
ZIP — the opencv-python-headless 5.x wheel has no OpenEXR, hence the OpenEXR bindings). The metric math (`mu_law_psnr`,
`apply_crf` …) is never copied: `hdr_metrics()` returns the module, or None outside the repo.
Set `LFS_HDR_TOOLS=<repo>/tools/eval` to point at the tools explicitly.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

log = logging.getLogger(__name__)

_MODULE: ModuleType | None = None
_TRIED = False


def tools_eval_dir() -> Path | None:
    """`<repo>/tools/eval` when reachable (env override first, then this file's resolved repo)."""
    env = os.environ.get("LFS_HDR_TOOLS")
    candidates = [Path(env)] if env else []
    here = Path(__file__).resolve()
    # <repo>/plugins/hdr_brackets/hdr_brackets/_tools.py → parents[3] = <repo>
    if len(here.parents) > 3:
        candidates.append(here.parents[3] / "tools" / "eval")
    for c in candidates:
        if (c / "hdr_metrics.py").is_file():
            return c
    return None


def hdr_metrics() -> ModuleType | None:
    """The repo's `hdr_metrics` module, imported once; None when the repo's tools are not reachable."""
    global _MODULE, _TRIED
    if _TRIED:
        return _MODULE
    _TRIED = True
    d = tools_eval_dir()
    if d is None:
        log.info("tools/eval not reachable; using the bundled EXR reader/writer")
        return None
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    try:
        _MODULE = importlib.import_module("hdr_metrics")
    except Exception as e:  # a broken tools env must not break the plugin
        log.warning("could not import hdr_metrics from %s: %s", d, e)
        _MODULE = None
    return _MODULE


# ── EXR I/O (verbatim fallback of tools/eval/hdr_metrics.py) ──────────────────────────────────────────────────────────


def _read_exr_local(path: str) -> np.ndarray:
    import OpenEXR

    with OpenEXR.File(str(path)) as f:
        ch = f.channels()
        if "RGB" in ch or "RGBA" in ch:
            px = ch["RGB" if "RGB" in ch else "RGBA"].pixels[..., :3]
        elif all(k in ch for k in "RGB"):
            px = np.stack([ch[k].pixels for k in "RGB"], axis=-1)
        else:
            first = next(iter(ch.values())).pixels
            px = np.stack([first] * 3, axis=-1)
    return np.ascontiguousarray(px, dtype=np.float32)


def _write_exr_local(path: str, rgb_float32: np.ndarray) -> None:
    import OpenEXR

    rgb = np.ascontiguousarray(np.asarray(rgb_float32, dtype=np.float32))
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"write_exr wants HxWx3, got {rgb.shape}")
    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    header.update(exr_colorspace_header())
    OpenEXR.File(header, {"RGB": rgb}).write(str(path))


def exr_colorspace_header() -> dict:
    """The working-space declaration every LichtFeld EXR carries (docs/hdr.md "Colour management"):
    Rec.709 / sRGB primaries, D65 as the standard `chromaticities`, and `lichtfeld_colorspace lin_srgb`."""
    try:
        import OpenEXR

        chroma = OpenEXR.Chromaticities((0.64, 0.33), (0.30, 0.60), (0.15, 0.06), (0.3127, 0.3290))
    except Exception:  # an older binding without the type: the string attribute still says it
        return {"lichtfeld_colorspace": "lin_srgb"}
    return {"chromaticities": chroma, "lichtfeld_colorspace": "lin_srgb"}


def read_exr(path: str | os.PathLike) -> np.ndarray:
    """HxWx3 float32 from an EXR (any compression; RGB/RGBA interleaved or planar)."""
    m = hdr_metrics()
    return m.read_exr(str(path)) if m is not None else _read_exr_local(str(path))


def write_exr(path: str | os.PathLike, rgb_float32: np.ndarray) -> None:
    """HxWx3 float32 → scanline ZIP EXR."""
    m = hdr_metrics()
    if m is not None:
        m.write_exr(str(path), rgb_float32)
    else:
        _write_exr_local(str(path), rgb_float32)
