"""One call from a folder of bracketed photos to an HDR dataset: `run(src_dir, out_dir, Options(), progress)`.

Stages, in order: read EXIF → group into brackets (+ manual overrides) → per bracket: load, align (MTB), flag
ghosting → calibrate the CRF on a sample of brackets (or load a saved one) → merge each bracket to a linear EXR →
tonemapped proxies → crf.json + exposures.json → SfM on the proxies → Report. The layout is dataset.py's.

The CRF is calibrated on RAW (unshifted) frames of brackets whose MTB shifts are all ≤ `Options.crf_max_shift` px:
a 1-px shift is harmless to the response, a wrong one is fatal — one bracket shifted by a failed search poisoned the
whole curve (review 2026-10-27: 0.395 nats from shifted frames, 0.161 from the same frames unshifted). Only when no
bracket qualifies are the least-shifted ones used, aligned, with a warning.

Warnings beyond a skipped bracket: ISO or f-number varying inside a bracket or across the shoot (the merge weighs
frames by exposure time only), an MTB search that hit its limit, frames more than `CLIPPED_WARN` clipped.

Contract with the LichtFeld UI layer (plugins/hdr_brackets/operators.py): `Options` fields and `Report` fields as
below; `progress(stage_label, fraction)` is called at least once per stage and once per bracket, from the calling
thread, with `fraction` the share of the WHOLE run done (0 → 1, non-decreasing) — a progress bar can show it as is.
An exception raised by `progress` propagates (that is how a caller cancels). A bad bracket never raises: it is
skipped and named in `Report.warnings`. Only a missing source folder raises (FileNotFoundError).

Memory: a bracket is loaded twice (align, then merge) rather than kept, the CRF brackets (≤ `crf_max_brackets`) a
third time, subsampled to ≤ 1024 px (nearest — 8-bit levels are preserved), and EXRs are written as they are merged,
so a few hundred 24 MP brackets fit.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from . import align as align_mod
from . import dataset as ds
from ._tools import read_exr
from .exif import ExposureInfo, list_images, read_exposure
from .grouping import Bracket, apply_overrides, brackets_to_json, group_brackets
from .merge import merge_bracket, merge_stats
from .proxy import proxy_exposure_value, tonemap_proxy
from .response import CRF, _time_key, calibrate_crf

LOG = logging.getLogger("hdr_brackets.pipeline")

#: Each stage's window of the overall progress fraction.
STAGES: dict[str, tuple[float, float]] = {
    "Reading EXIF": (0.00, 0.04),
    "Grouping brackets": (0.04, 0.05),
    "Aligning": (0.05, 0.30),
    "Calibrating CRF": (0.30, 0.36),
    "Merging": (0.36, 0.70),
    "Proxies": (0.70, 0.80),
    "Writing dataset": (0.80, 0.82),
    "SfM": (0.82, 1.00),
}
#: CRF sample brackets are kept with their longest side ≤ this (stride subsampling).
CRF_SAMPLE_MAX_SIDE = 1024
#: Below this registered share the report warns (PLAN.md M1: SfM registers ≥ 95 % of positions).
SFM_MIN_REGISTERED = 0.95
#: A frame with more than this share of its pixels clipped (any channel ≥ 250, or every channel ≤ 5) is reported.
CLIPPED_WARN = 0.90
#: 8-bit level from which a pixel counts as saturated (the merge's lower-bound pixels, `saturated` per position).
SATURATED_LEVEL = 250
#: ISO / f-number values closer than this (relative) are "the same" (EXIF rationals round).
SETTING_RTOL = 0.02
BRACKETS_FILE = "brackets.json"


@dataclass
class Options:
    max_gap_s: float = 5.0
    deghost: bool = True
    proxy_operator: str = "aces"
    proxy_ev: float = 0.0
    run_sfm: bool = True
    crf_path: str | None = None
    overrides: dict | None = None
    ghost_threshold: float = align_mod.FLAG_THRESHOLD
    # ── beyond the UI contract (defaults keep its behaviour) ──
    proxy_shared_scale: bool = True
    min_size: int = 2
    max_size: int = 9
    crf_samples: int = 1000  # per bracket (response.py: why not OpenCV's 70)
    crf_lambda: float = 100.0  # response.py: why not 30
    crf_max_brackets: int = 12
    crf_max_shift: int = 2  # a calibration bracket's largest MTB shift (px)


@dataclass
class Report:
    n_images: int
    n_brackets: int
    n_singletons: int
    flagged: list[str]
    crf_path: str | None
    registered: int
    total: int
    out_dir: str
    warnings: list[str]
    seconds: float
    # ── beyond the UI contract ──
    n_positions: int = 0
    skipped: list[str] = field(default_factory=list)
    proxy_scale: float | None = None
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=1)


@dataclass
class _Aligned:
    bracket: Bracket
    members: list[ExposureInfo]  # darkest first
    shifts: list[tuple[int, int]]
    score: float
    flagged: bool
    status: list[str] = field(default_factory=list)

    @property
    def shift_max(self) -> int:
        return max((max(abs(int(dx)), abs(int(dy))) for dx, dy in self.shifts), default=0)


# ── Helpers ───────────────────────────────────────────────────────────────────────────────────────────────────────────


class _Progress:
    def __init__(self, cb: Callable[[str, float], None] | None) -> None:
        self.cb = cb
        self.last = 0.0

    def __call__(self, stage: str, frac: float = 0.0, label: str | None = None) -> None:
        if self.cb is None:
            return
        lo, hi = STAGES[stage]
        f = max(self.last, lo + (hi - lo) * min(1.0, max(0.0, frac)))
        self.last = f
        self.cb(label or stage, f)


def load_rgb8(path: str | Path) -> np.ndarray:
    """An image file → uint8 HxWx3 RGB (16-bit inputs are reduced to 8 bit: the CRF domain is ldr8)."""
    with Image.open(path) as im:
        if im.mode in ("I;16", "I;16B", "I;16L", "I"):
            a = np.asarray(im, dtype=np.float64)
            a = np.clip(np.rint(a / 257.0), 0, 255).astype(np.uint8)
            return np.ascontiguousarray(np.dstack([a] * 3))
        return np.ascontiguousarray(np.asarray(im.convert("RGB")))


def _subsample(img: np.ndarray, max_side: int = CRF_SAMPLE_MAX_SIDE) -> np.ndarray:
    s = max(1, int(math.ceil(max(img.shape[:2]) / max_side)))
    return np.ascontiguousarray(img[::s, ::s]) if s > 1 else img


def _clean_outputs(out: Path, run_sfm: bool) -> None:
    """Remove this pipeline's previous products (only its own file patterns) so no stale position survives."""
    for sub, pat in ((ds.HDR_DIR, "pos_*.exr"), (ds.PROXY_DIR, "pos_*.png")):
        d = out / sub
        if d.is_dir():
            for p in d.glob(pat):
                p.unlink()
    if run_sfm:
        import shutil

        shutil.rmtree(out / "sparse", ignore_errors=True)


def read_infos(
    src_dir: str | Path, warnings: list[str] | None = None, tick: Callable[[float], None] | None = None
) -> list[ExposureInfo]:
    """EXIF of every image in the folder; unreadable files are skipped with a warning."""
    files = list_images(src_dir)
    infos: list[ExposureInfo] = []
    for k, p in enumerate(files):
        try:
            infos.append(read_exposure(p))
        except Exception as e:
            msg = f"{p.name}: unreadable image ({e}); skipped"
            LOG.warning(msg)
            if warnings is not None:
                warnings.append(msg)
        if tick is not None:
            tick((k + 1) / max(1, len(files)))
    return infos


def group_folder(src_dir: str | Path, options: Options | None = None) -> list[Bracket]:
    """EXIF + grouping (+ overrides) only — what a manual-override UI lists before running the whole pipeline."""
    o = options or Options()
    br = group_brackets(read_infos(src_dir), o.max_gap_s, o.min_size, o.max_size)
    if o.overrides:
        br = apply_overrides(br, o.overrides, o.min_size, o.max_size)
    return br


def clipped_fraction(img: np.ndarray) -> float:
    """Share of pixels clipped: any channel ≥ SATURATED_LEVEL, or every channel ≤ 5."""
    a = np.asarray(img)
    return float((np.any(a >= SATURATED_LEVEL, axis=2) | np.all(a <= 5, axis=2)).mean())


def saturated_count(img: np.ndarray) -> int:
    """Pixels with any channel ≥ SATURATED_LEVEL (in the shortest exposure: radiance known only as a lower bound)."""
    return int(np.any(np.asarray(img) >= SATURATED_LEVEL, axis=2).sum())


def _distinct(values: list[float | None], rtol: float = SETTING_RTOL) -> list[float]:
    """The distinct values (None ignored), two values within `rtol` of each other being one."""
    out: list[float] = []
    for v in sorted(float(x) for x in values if x is not None):
        if not out or abs(v - out[-1]) > rtol * max(abs(v), abs(out[-1])):
            out.append(v)
    return out


def setting_warnings(brackets: list[Bracket]) -> list[str]:
    """ISO / f-number that vary inside a bracket or across the shoot. The merge uses exposure time only, while the
    `exif_ev` written to exposures.json is log2(t·ISO/N²): a frame shot at another ISO or aperture is mis-weighted
    inside its bracket (wrong radiance), and positions shot at different settings differ in EXR scale."""
    out: list[str] = []
    for label, attr, fmt in (("ISO", "iso", "{:g}"), ("f-number", "fnumber", "f/{:g}")):
        inside = []
        for b in brackets:
            d = _distinct([getattr(m, attr) for m in b.members])
            if len(d) > 1:
                inside.append(f"bracket {b.id}: {', '.join(fmt.format(v) for v in d)}")
        if inside:
            out.append(
                f"{label} varies inside {len(inside)} bracket(s) ({'; '.join(inside[:4])}{' …' if len(inside) > 4 else ''}):"
                " the merge weighs frames by exposure time only, so those brackets' EXRs are wrong"
            )
        shoot = _distinct([getattr(m, attr) for b in brackets for m in b.members])
        if len(shoot) > 1 and not inside:
            out.append(
                f"{label} varies across the shoot ({', '.join(fmt.format(v) for v in shoot)}): the EXRs are scaled by "
                "exposure time only, so positions shot at different settings differ in brightness by that ratio"
            )
    return out


def _crf_sample_ids(brackets: list[Bracket], k: int) -> set[int]:
    """Bracket ids to keep for calibration: the dominant exposure-time set (and frame size), spread over the shoot."""
    keys = {
        b.id: (_time_key([m.exposure_time for m in b.members]), (b.members[0].width, b.members[0].height))
        for b in brackets
    }
    if not keys:
        return set()
    best, _ = Counter(keys.values()).most_common(1)[0]
    ids = [b.id for b in brackets if keys[b.id] == best]
    if len(ids) > k:
        idx = np.linspace(0, len(ids) - 1, k).round().astype(int)
        ids = [ids[i] for i in sorted(set(idx.tolist()))]
    return set(ids)


# ── Run ───────────────────────────────────────────────────────────────────────────────────────────────────────────────


def run(
    src_dir: str | Path,
    out_dir: str | Path,
    options: Options | None = None,
    progress: Callable[[str, float], None] | None = None,
    log: logging.Logger = LOG,
) -> Report:
    """Brackets in `src_dir` → HDR dataset in `out_dir` (see the module docstring)."""
    t0 = time.time()
    o = options if options is not None else Options()
    src, out = Path(src_dir), Path(out_dir)
    if not src.is_dir():
        raise FileNotFoundError(f"no source folder {src}")
    out.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    skipped: list[str] = []
    prog = _Progress(progress)

    def warn(msg: str) -> None:
        log.warning(msg)
        warnings.append(msg)

    def report(**kw: Any) -> Report:
        base = dict(
            n_images=0,
            n_brackets=0,
            n_singletons=0,
            flagged=[],
            crf_path=None,
            registered=0,
            total=0,
            out_dir=str(out),
            warnings=warnings,
            seconds=round(time.time() - t0, 2),
            skipped=skipped,
        )
        base.update(kw)
        return Report(**base)

    # ── 1 EXIF ──
    prog("Reading EXIF", 0.0)
    infos = read_infos(src, warnings, lambda f: prog("Reading EXIF", f))
    if not infos:
        warn(f"no images in {src}")
        return report()
    missing = [i.name for i in infos if i.exposure_time is None]
    if missing:
        warn(
            f"{len(missing)} image(s) without an exposure time: {', '.join(missing[:8])}"
            + (" …" if len(missing) > 8 else "")
        )

    # ── 2 Grouping ──
    prog("Grouping brackets", 0.0)
    brackets = group_brackets(infos, o.max_gap_s, o.min_size, o.max_size)
    if o.overrides:
        brackets = apply_overrides(brackets, o.overrides, o.min_size, o.max_size)
    (out / BRACKETS_FILE).write_text(brackets_to_json(brackets))
    good = [b for b in brackets if b.ok]
    n_singletons = sum(1 for b in brackets if b.issue == "singleton")
    for b in brackets:
        if not b.ok:
            skipped.append(f"bracket {b.id}")
            warn(f"bracket {b.id} ({', '.join(b.names)}): {b.issue}; skipped")
    sizes = Counter(len(b.members) for b in good)
    log.info("%d images → %d brackets (sizes %s), %d singleton(s)", len(infos), len(good), dict(sizes), n_singletons)
    for msg in setting_warnings(good):
        warn(msg)
    prog("Grouping brackets", 1.0)
    common = dict(n_images=len(infos), n_brackets=len(good), n_singletons=n_singletons)
    if not good:
        warn("no usable bracket")
        return report(**common)

    # ── 3 Align ──
    aligned: list[_Aligned] = []
    clipped_frames: list[tuple[str, float]] = []
    for k, b in enumerate(good):
        prog("Aligning", k / len(good), f"Aligning bracket {k + 1}/{len(good)}")
        members = b.by_exposure()
        try:
            imgs = [load_rgb8(m.path) for m in members]
            r = align_mod.align_bracket_detailed(imgs)
        except Exception as e:
            skipped.append(f"bracket {b.id}")
            warn(f"bracket {b.id} ({', '.join(b.names)}): {e}; skipped")
            continue
        for m, im in zip(members, imgs, strict=True):
            cf = clipped_fraction(im)
            if cf > CLIPPED_WARN:
                clipped_frames.append((m.name, cf))
        if r.failed:
            lim = (1 << r.max_bits) - 1
            warn(
                f"bracket {b.id}: MTB search reached its ±{lim} px limit on "
                + ", ".join(members[i].name for i in r.failed)
                + " (failed, not a measurement): kept unshifted"
            )
        flagged = r.score > o.ghost_threshold
        if flagged:
            log.info("bracket %d: ghost score %.3f > %.3f, flagged", b.id, r.score, o.ghost_threshold)
        aligned.append(_Aligned(b, members, r.shifts, r.score, flagged, r.status))
    prog("Aligning", 1.0)
    if clipped_frames:
        warn(
            f"{len(clipped_frames)} frame(s) more than {CLIPPED_WARN:.0%} clipped (they add little to the merge and "
            "cannot be aligned): "
            + ", ".join(f"{n} {100 * f:.1f} %" for n, f in clipped_frames[:8])
            + (" …" if len(clipped_frames) > 8 else "")
        )
    if not aligned:
        warn("every bracket failed to load or align")
        return report(**common)

    # ── 4 CRF ──
    prog("Calibrating CRF", 0.0)
    crf_file = out / ds.CRF_FILE
    crf_brackets: list[int] = []
    try:
        if o.crf_path:
            crf = CRF.load(o.crf_path)
            log.info("CRF loaded from %s", o.crf_path)
        else:
            steady = [a for a in aligned if a.shift_max <= o.crf_max_shift]
            calm = [a for a in steady if not a.flagged] or steady
            apply = False
            if not calm:  # every bracket moved: the least-moved ones, aligned
                calm = sorted(aligned, key=lambda a: a.shift_max)[: o.crf_max_brackets]
                apply = True
                warn(
                    f"CRF: no bracket has all its shifts ≤ {o.crf_max_shift} px; calibrated on the "
                    f"{len(calm)} least-shifted one(s), aligned (largest shift {calm[-1].shift_max} px)"
                )
            by_id = {a.bracket.id: a for a in calm}
            ids = _crf_sample_ids([a.bracket for a in calm], o.crf_max_brackets)
            chosen = [by_id[i] for i in sorted(ids)]
            samples_imgs: list[list[np.ndarray]] = []
            samples_times: list[list[float]] = []
            for j, a in enumerate(chosen):
                prog("Calibrating CRF", 0.8 * j / max(1, len(chosen)))
                imgs = [load_rgb8(m.path) for m in a.members]
                if apply:
                    imgs = align_mod.apply_shifts(imgs, a.shifts)
                samples_imgs.append([_subsample(im) for im in imgs])
                samples_times.append([float(m.exposure_time) for m in a.members])
                crf_brackets.append(a.bracket.id)
            crf = calibrate_crf(samples_imgs, samples_times, o.crf_samples, o.crf_lambda, o.crf_max_brackets)
            crf.meta["bracket_ids"] = crf_brackets
            crf.meta["frames"] = "aligned" if apply else "raw"
            samples_imgs.clear()
        crf.save(crf_file)
    except Exception as e:
        warn(f"CRF: {e}")
        return report(**common)
    prog("Calibrating CRF", 1.0)

    # ── 5 Merge ──
    _clean_outputs(out, o.run_sfm)
    positions: list[ds.PositionResult] = []
    keys: list[float] = []
    for k, a in enumerate(aligned):
        prog("Merging", k / len(aligned), f"Merging bracket {k + 1}/{len(aligned)}")
        name = ds.position_name(len(positions))
        try:
            raw = [load_rgb8(m.path) for m in a.members]
            saturated = saturated_count(raw[0])  # members are darkest first
            imgs = align_mod.apply_shifts(raw, a.shifts)
            del raw
            hdr = merge_bracket(imgs, [float(m.exposure_time) for m in a.members], crf, deghost=o.deghost)
            st = merge_stats(hdr)
            if st["nan"] or st["inf"]:
                warn(f"{name} (bracket {a.bracket.id}): {st['nan']} NaN / {st['inf']} inf values replaced by 0")
                hdr = np.nan_to_num(hdr, nan=0.0, posinf=0.0, neginf=0.0)
            ds.write_hdr(out, name, hdr)
            keys.append(proxy_exposure_value(hdr, o.proxy_ev))
        except Exception as e:
            skipped.append(f"bracket {a.bracket.id}")
            warn(f"bracket {a.bracket.id} ({', '.join(a.bracket.names)}): merge failed ({e}); skipped")
            continue
        positions.append(
            ds.PositionResult(
                name=name,
                bracket_id=a.bracket.id,
                sources=a.members,
                proxy_scale=keys[-1],
                ghost_score=a.score,
                flagged=a.flagged,
                shifts=a.shifts,
                stats=st,
                saturated=saturated,
                align_status=a.status,
            )
        )
    prog("Merging", 1.0)
    if not positions:
        warn("no bracket merged")
        return report(**common, crf_path=str(crf_file))

    # ── 6 Proxies ──
    shared = float(np.exp(np.median(np.log(keys)))) if o.proxy_shared_scale else None
    for k, pos in enumerate(positions):
        prog("Proxies", k / len(positions), f"Proxy {k + 1}/{len(positions)}")
        if shared is not None:
            pos.proxy_scale = shared
        try:
            hdr = read_exr(out / pos.hdr_rel)
            ds.write_proxy(out, pos.name, tonemap_proxy(hdr, o.proxy_ev, o.proxy_operator, scale=pos.proxy_scale))
        except Exception as e:
            warn(f"{pos.name}: proxy failed ({e})")
    prog("Proxies", 1.0)

    # ── 7 Dataset index ──
    flagged = [p.name for p in positions if p.flagged]
    p50 = [p.stats.get("p50", float("nan")) for p in positions]
    stats = {
        "ghost_score_max": float(max(p.ghost_score for p in positions)),
        "shift_max": int(max(a.shift_max for a in aligned)),
        "align_failed": int(sum(st == align_mod.STATUS_FAILED for a in aligned for st in a.status)),
        "align_clipped": int(sum(st == align_mod.STATUS_CLIPPED for a in aligned for st in a.status)),
        "frames_clipped_over_90": len(clipped_frames),
        "saturated_total": int(sum(p.saturated or 0 for p in positions)),
        "crf_brackets": crf_brackets,
        "radiance_p50_median": float(np.nanmedian(p50)) if p50 else None,
        "radiance_max": float(max(p.stats.get("max", 0.0) for p in positions)),
        "crf_source": crf.source,
    }

    prog("Writing dataset", 0.0)
    ds.write_index(
        out,
        positions,
        crf,
        infos,
        proxy_operator=o.proxy_operator,
        proxy_ev=o.proxy_ev,
        source_dir=str(src.resolve()),
        extra={
            "ghost_threshold": o.ghost_threshold,
            "deghost": o.deghost,
            "proxy_shared_scale": o.proxy_shared_scale,
            "saturated_level": SATURATED_LEVEL,
            "stats": dict(stats),
        },
    )
    prog("Writing dataset", 1.0)

    # ── 8 SfM ──
    registered, total = 0, len(positions)
    if o.run_sfm:
        prog("SfM", 0.0)
        try:
            from .sfm import run_sfm

            res = run_sfm(out, progress=lambda label, f: prog("SfM", f, label))
            registered, total = res.registered, res.total
            stats["sfm_points"] = res.points
            if total and registered / total < SFM_MIN_REGISTERED:
                warn(f"SfM registered {registered}/{total} positions (< {SFM_MIN_REGISTERED:.0%})")
        except Exception as e:
            warn(f"SfM: {e}")
        prog("SfM", 1.0)
    else:
        prog("SfM", 1.0, "SfM skipped")

    return report(
        **common,
        flagged=flagged,
        crf_path=str(crf_file),
        registered=registered,
        total=total,
        n_positions=len(positions),
        proxy_scale=shared,
        stats=stats,
    )
