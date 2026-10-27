"""hdr_brackets — the bracket-merge core: exposure brackets → linear EXR + tonemapped proxies + SfM (milestone M1).

Plain Python, never imports `lichtfeld`, so it runs in the plugin's own uv env, in tests and from the CLI alike. The
LichtFeld UI layer (the plugin package one level up) drives `pipeline.run(src, out, Options(...), progress)` and reads
its `Report`.

Modules: `exif` (exposure tags) → `grouping` (brackets from timestamps + exposure times) → `align` (MTB + ghost score)
→ `response` (Debevec CRF) → `merge` (linear radiance, optional deghosting) → `proxy` (tonemapped 8-bit sRGB for SfM)
→ `dataset` (the on-disk layout M2/M3 read) → `sfm` (pycolmap) — `pipeline` chains them, `cli` exposes it.
"""

from __future__ import annotations

__version__ = "0.1.0"
