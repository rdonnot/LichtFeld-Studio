"""Exposure EXIF of one image: exposure time, f-number, ISO, EV and capture time, read with Pillow.

Why: grouping needs the capture time and exposure time, merging needs the exposure time, and M3 seeds its per-image
exposure from the EV. The EV follows LichtFeld's own convention (src/core/exif.cpp, which seeds PPISP from the same
tags): ev = log2(t / N² · ISO), each missing factor taken as 1, with the same fallbacks — ShutterSpeedValue (APEX) for
a missing ExposureTime, ApertureValue (APEX) for a missing FNumber, and the ISO alternates (SOS, REI, ISOSpeed).
A file with no usable tags is not an error: `exposure_time=None` (and a logged warning); the grouping reports it.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

log = logging.getLogger(__name__)

# ── Tags ──────────────────────────────────────────────────────────────────────────────────────────────────────────────

TAG_EXIF_IFD = 0x8769
TAG_DATETIME = 0x0132
TAG_EXPOSURE_TIME = 0x829A
TAG_FNUMBER = 0x829D
TAG_ISO = 0x8827
TAG_SENSITIVITY_SOS = 0x8831
TAG_SENSITIVITY_REI = 0x8832
TAG_ISO_SPEED = 0x8833
TAG_DATETIME_ORIGINAL = 0x9003
TAG_SHUTTER_SPEED_APEX = 0x9201
TAG_APERTURE_APEX = 0x9202
TAG_SUBSEC_TIME_ORIGINAL = 0x9291

EXIF_TIME_FORMAT = "%Y:%m:%d %H:%M:%S"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


@dataclass
class ExposureInfo:
    """What one source image says about its exposure. `ev` is None only when no exposure tag at all was found."""

    path: str
    exposure_time: float | None
    fnumber: float | None
    iso: float | None
    ev: float | None
    timestamp: datetime | None
    width: int
    height: int

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = format_timestamp(self.timestamp)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExposureInfo:
        ts = d.get("timestamp")
        return cls(
            path=str(d["path"]),
            exposure_time=d.get("exposure_time"),
            fnumber=d.get("fnumber"),
            iso=d.get("iso"),
            ev=d.get("ev"),
            timestamp=parse_timestamp(ts) if ts else None,
            width=int(d.get("width", 0)),
            height=int(d.get("height", 0)),
        )


# ── Helpers ───────────────────────────────────────────────────────────────────────────────────────────────────────────


def rational(v: Any) -> float | None:
    """An EXIF number as Pillow hands it back (IFDRational, (num, den), a plain number or a 1-tuple) → float."""
    try:
        if isinstance(v, tuple):
            if len(v) == 2:
                return float(v[0]) / float(v[1]) if float(v[1]) != 0 else None
            if len(v) == 1:
                return rational(v[0])
            return None
        f = float(v)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return f if math.isfinite(f) else None


def _positive(v: float | None) -> float | None:
    return v if v is not None and v > 0 and math.isfinite(v) else None


def parse_timestamp(s: Any, subsec: Any = None) -> datetime | None:
    """EXIF `YYYY:MM:DD HH:MM:SS` (or ISO 8601) plus SubSecTimeOriginal digits → datetime; None when unreadable."""
    if s is None:
        return None
    text = str(s).strip().rstrip("\x00")
    ts: datetime | None = None
    try:
        ts = datetime.strptime(text[:19], EXIF_TIME_FORMAT)
    except ValueError:
        try:
            ts = datetime.fromisoformat(text)
        except ValueError:
            return None
    if subsec is not None:
        digits = "".join(ch for ch in str(subsec) if ch.isdigit())
        if digits:
            ts = ts.replace(microsecond=int((digits + "000000")[:6]))
    return ts


def format_timestamp(ts: datetime | None) -> str | None:
    """EXIF form when whole seconds (as the camera wrote it), ISO 8601 when it carries SubSecTime."""
    if ts is None:
        return None
    return ts.isoformat() if ts.microsecond else ts.strftime(EXIF_TIME_FORMAT)


def compute_ev(t: float | None, n: float | None, iso: float | None) -> float | None:
    """LichtFeld's EV: log2(t / N² · ISO), each missing factor = 1; None when all three are missing."""
    if t is None and n is None and iso is None:
        return None
    tt = t if t is not None else 1.0
    nn = n if n is not None else 1.0
    ss = iso if iso is not None else 1.0
    ev = math.log2(tt / (nn * nn) * ss)
    return ev if math.isfinite(ev) else None


# ── Reader ────────────────────────────────────────────────────────────────────────────────────────────────────────────


def read_exposure(path: str | os.PathLike) -> ExposureInfo:
    """Exposure tags of one image. Never raises for missing or broken EXIF (only for an unreadable image file)."""
    p = Path(path)
    with Image.open(p) as im:
        width, height = im.size
        try:
            exif = im.getexif()
        except Exception as e:  # a corrupt EXIF block is "no EXIF", not a crash
            log.warning("%s: unreadable EXIF (%s)", p.name, e)
            exif = None
        ifd: dict[int, Any] = {}
        base: dict[int, Any] = {}
        if exif is not None:
            base = dict(exif)
            try:
                ifd = dict(exif.get_ifd(TAG_EXIF_IFD))
            except Exception:
                ifd = {}

    def tag(key: int) -> Any:
        if key in ifd:
            return ifd[key]
        return base.get(key)

    t = _positive(rational(tag(TAG_EXPOSURE_TIME)))
    if t is None:
        tv = rational(tag(TAG_SHUTTER_SPEED_APEX))
        if tv is not None:
            t = _positive(2.0 ** (-tv))
    n = _positive(rational(tag(TAG_FNUMBER)))
    if n is None:
        av = rational(tag(TAG_APERTURE_APEX))
        if av is not None:
            n = _positive(2.0 ** (av / 2.0))
    iso = None
    for key in (TAG_ISO, TAG_SENSITIVITY_SOS, TAG_SENSITIVITY_REI, TAG_ISO_SPEED):
        iso = _positive(rational(tag(key)))
        if iso is not None:
            break
    ts = parse_timestamp(tag(TAG_DATETIME_ORIGINAL), tag(TAG_SUBSEC_TIME_ORIGINAL))
    if ts is None:
        ts = parse_timestamp(tag(TAG_DATETIME))
    if t is None:
        log.warning("%s: no exposure time in EXIF", p.name)
    if ts is None:
        log.info("%s: no capture time in EXIF; grouping falls back to the file order", p.name)
    return ExposureInfo(
        path=str(p),
        exposure_time=t,
        fnumber=n,
        iso=iso,
        ev=compute_ev(t, n, iso),
        timestamp=ts,
        width=int(width),
        height=int(height),
    )


def list_images(src_dir: str | os.PathLike) -> list[Path]:
    """Image files directly in `src_dir` (not recursive), sorted by name."""
    d = Path(src_dir)
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
