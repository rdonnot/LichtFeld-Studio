"""Write the merged dataset — the layout M2 trains from (EXR) and M3 trains from (the raw brackets + one pose each).

    <out>/images_hdr/pos_NNN.exr   linear radiance, float32 RGB, ZIP (tools/eval's write_exr)
    <out>/images/pos_NNN.png       the 8-bit sRGB proxy (what SfM registers; same stem as the EXR)
    <out>/crf.json                 the CRF, in tools/synth's crf.json form
    <out>/exposures.json           {"version": 2, "crf": "crf.json", "proxy_operator", "positions": [{name, hdr,
                                   proxy, proxy_scale, bracket_id, ghost_score, flagged, saturated, stats,
                                   sources: [{file, exposure_time, exif_ev, ev_step, iso, fnumber, timestamp,
                                   shift, align}]}], "stats": {…}, …}
    <out>/sparse/0/                COLMAP binary model (sfm.py), images named pos_NNN.png

`source_dir` + each source's `file` locate the raw JPEG; its `shift` = the (dx, dy) pixels MTB moved it onto the
bracket's reference, so M3 can train every raw frame with the bracket's pose, and `align` how that shift was
obtained ("reference", "mtb" = searched, "clipped" = too clipped to search, took its neighbour's, "failed" = the
search hit its limit, kept (0, 0); align.py). `proxy_scale` maps proxy pixels back to EXR radiance
(proxy ≈ OETF(op(EXR × proxy_scale))).

EV words mean what they mean in the synthetic truth (`exposures_gt.json`): `exif_ev` = the absolute EXIF value,
LichtFeld's log2(t / N² · ISO) (exif.py); `ev_step` = the frame's step inside its bracket, log2(t / t_median) with
t_median the bracket's median exposure time (0 for the middle frame of an odd bracket). Version 1 wrote the absolute
value as `ev`. `saturated` = pixels with a channel ≥ `saturated_level` (250) in the bracket's SHORTEST exposure: the
merged radiance there is only a lower bound.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ._tools import write_exr
from .exif import ExposureInfo, format_timestamp
from .response import CRF

log = logging.getLogger(__name__)

HDR_DIR = "images_hdr"
PROXY_DIR = "images"
CRF_FILE = "crf.json"
INDEX_FILE = "exposures.json"
INDEX_VERSION = 2


def position_name(index: int) -> str:
    return f"pos_{int(index):03d}"


@dataclass
class PositionResult:
    """One merged position. `hdr` / `proxy` may be None once written (the pipeline streams them to disk)."""

    name: str
    bracket_id: int
    sources: list[ExposureInfo]
    hdr: np.ndarray | None = None
    proxy: np.ndarray | None = None
    proxy_scale: float = 1.0
    ghost_score: float = 0.0
    flagged: bool = False
    shifts: list[tuple[int, int]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    saturated: int | None = None
    align_status: list[str] = field(default_factory=list)

    @property
    def hdr_rel(self) -> str:
        return f"{HDR_DIR}/{self.name}.exr"

    @property
    def proxy_rel(self) -> str:
        return f"{PROXY_DIR}/{self.name}.png"


# ── Files ─────────────────────────────────────────────────────────────────────────────────────────────────────────────


def write_hdr(out_dir: str | os.PathLike, name: str, hdr: np.ndarray) -> Path:
    p = Path(out_dir) / HDR_DIR / f"{name}.exr"
    p.parent.mkdir(parents=True, exist_ok=True)
    write_exr(p, np.asarray(hdr, dtype=np.float32))
    return p


def write_proxy(out_dir: str | os.PathLike, name: str, proxy_uint8: np.ndarray) -> Path:
    p = Path(out_dir) / PROXY_DIR / f"{name}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(proxy_uint8)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"proxy must be uint8 HxWx3, got {a.shape} {a.dtype}")
    Image.fromarray(a, "RGB").save(p, "PNG")
    return p


def read_proxy(path: str | os.PathLike) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def median_log2_time(infos: Sequence[ExposureInfo]) -> float | None:
    """log2 of the bracket's median exposure time (the median of log2 t: for an even count, the geometric middle)."""
    ts = [i.exposure_time for i in infos if i.exposure_time]
    return float(np.median(np.log2(ts))) if ts else None


def _source_entry(
    info: ExposureInfo, shift: tuple[int, int] | None, mid_log2_t: float | None, align: str | None = None
) -> dict[str, Any]:
    step = None
    if info.exposure_time and mid_log2_t is not None:
        step = float(np.log2(info.exposure_time) - mid_log2_t)
    d: dict[str, Any] = {
        "file": info.name,
        "exposure_time": info.exposure_time,
        "exif_ev": info.ev,
        "ev_step": step,
        "iso": info.iso,
        "fnumber": info.fnumber,
        "timestamp": format_timestamp(info.timestamp),
        "shift": [int(shift[0]), int(shift[1])] if shift is not None else [0, 0],
    }
    if align is not None:
        d["align"] = align
    return d


def _lookup(source_infos: Mapping[str, ExposureInfo] | Sequence[ExposureInfo] | None) -> dict[str, ExposureInfo]:
    if source_infos is None:
        return {}
    if isinstance(source_infos, Mapping):
        return {os.path.basename(k): v for k, v in source_infos.items()}
    return {i.name: i for i in source_infos}


def index_dict(
    positions: Sequence[PositionResult],
    source_infos: Mapping[str, ExposureInfo] | Sequence[ExposureInfo] | None = None,
    *,
    proxy_operator: str = "aces",
    proxy_ev: float = 0.0,
    source_dir: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """exposures.json as a dict (see the module docstring). A source's EXIF record comes from `source_infos` when it
    holds that file name (the freshest read), else from the position's own `sources`."""
    lookup = _lookup(source_infos)
    entries = []
    for pos in positions:
        infos = [lookup.get(info.name, info) for info in pos.sources]
        mid = median_log2_time(infos)
        srcs = []
        for k, info in enumerate(infos):
            srcs.append(
                _source_entry(
                    info,
                    pos.shifts[k] if k < len(pos.shifts) else None,
                    mid,
                    pos.align_status[k] if k < len(pos.align_status) else None,
                )
            )
        entry: dict[str, Any] = {
            "name": pos.name,
            "hdr": pos.hdr_rel,
            "proxy": pos.proxy_rel,
            "proxy_scale": float(pos.proxy_scale),
            "bracket_id": int(pos.bracket_id),
            "ghost_score": float(pos.ghost_score),
            "flagged": bool(pos.flagged),
        }
        if pos.saturated is not None:
            entry["saturated"] = int(pos.saturated)
        entry.update(stats=pos.stats, sources=srcs)
        entries.append(entry)
    d: dict[str, Any] = {
        "version": INDEX_VERSION,
        "crf": CRF_FILE,
        "proxy_operator": proxy_operator,
        "proxy_ev": float(proxy_ev),
        "units": "linear radiance in the CRF's units (Debevec anchor g(128)=0: mid-grey shot at 1 s reads 1.0)",
        "source_dir": source_dir,
        "positions": entries,
    }
    if extra:
        d.update(extra)
    return d


def write_index(
    out_dir: str | os.PathLike,
    positions: Sequence[PositionResult],
    crf: CRF,
    source_infos: Mapping[str, ExposureInfo] | Sequence[ExposureInfo] | None = None,
    **kwargs: Any,
) -> Path:
    """crf.json + exposures.json (the EXR / PNG files are written separately or by write_dataset)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    crf.save(out / CRF_FILE)
    p = out / INDEX_FILE
    p.write_text(json.dumps(index_dict(positions, source_infos, **kwargs), indent=1))
    return p


def write_dataset(
    out_dir: str | os.PathLike,
    positions: Sequence[PositionResult],
    crf: CRF,
    source_infos: Mapping[str, ExposureInfo] | Sequence[ExposureInfo] | None = None,
    **kwargs: Any,
) -> None:
    """Every position's EXR and proxy PNG (those still in memory), then crf.json and exposures.json.

    kwargs: proxy_operator, proxy_ev, source_dir, extra (see `index_dict`)."""
    for pos in positions:
        if pos.hdr is not None:
            write_hdr(out_dir, pos.name, pos.hdr)
        if pos.proxy is not None:
            write_proxy(out_dir, pos.name, pos.proxy)
    write_index(out_dir, positions, crf, source_infos, **kwargs)
    log.info("dataset written: %d positions in %s", len(positions), out_dir)
