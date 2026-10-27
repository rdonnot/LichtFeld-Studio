"""HDR Brackets: shared settings, background job and the Merge operator (registered as lfs_plugins.hdr_brackets.operators.MergeOperator).

The panel and the operator share one ``SETTINGS`` / ``JOB`` pair so a script or an
MCP call that invokes the operator runs exactly what the panel would run.
Nothing heavy is imported here: the core package (``hdr_brackets.pipeline``) is
imported inside the worker thread.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import lichtfeld as lf
from lfs_plugins.types import Operator

PROXY_OPERATORS = ["aces", "reinhard", "linear"]


# ── Settings ──────────────────────────────────────────────────────────────────


@dataclass
class Settings:
    src_dir: str = ""
    out_dir: str = ""
    out_dir_custom: bool = False
    max_gap_s: float = 5.0
    deghost: bool = True
    proxy_operator: str = "aces"
    proxy_ev: float = 0.0
    run_sfm: bool = True
    overrides_text: str = ""

    def effective_out_dir(self) -> str:
        if self.out_dir_custom and self.out_dir:
            return self.out_dir
        src = self.src_dir.rstrip("/\\")
        return f"{src}_hdr" if src else ""


SETTINGS = Settings()


def parse_overrides(text: str) -> dict[str, int] | None:
    """Parse ``"IMG_0007.jpg=3, IMG_0008.jpg=3"`` into ``{file: bracket id}``.

    Separators: comma, semicolon or newline. Raises ValueError on a bad entry.
    """
    out: dict[str, int] = {}
    for raw in text.replace(";", ",").replace("\n", ",").split(","):
        item = raw.strip()
        if not item:
            continue
        name, sep, value = item.rpartition("=")
        if not sep or not name.strip():
            raise ValueError(f"override '{item}': expected FILE=NUMBER")
        try:
            out[name.strip()] = int(value.strip())
        except ValueError:
            raise ValueError(f"override '{item}': '{value.strip()}' is not an integer") from None
    return out or None


# ── Background job ────────────────────────────────────────────────────────────


@dataclass
class Job:
    lock: threading.Lock = field(default_factory=threading.Lock)
    running: bool = False
    cancel_requested: bool = False  # the core has no cancel yet: UI-only, the run finishes
    stage: str = ""
    fraction: float = 0.0
    report: Any = None
    error: str = ""
    out_dir: str = ""
    finished_at: float = 0.0
    generation: int = 0  # bumps on every finish so the panel can reset its preview


JOB = Job()


def _redraw() -> None:
    lf.ui.request_redraw()


def start_job(settings: Settings | None = None) -> tuple[bool, str]:
    """Start the pipeline on a worker thread. Returns (started, message)."""
    s = settings or SETTINGS
    if JOB.running:
        return False, "a merge is already running"
    src = s.src_dir.strip()
    if not src:
        return False, "choose a source folder first"
    try:
        overrides = parse_overrides(s.overrides_text)
    except ValueError as exc:
        return False, str(exc)
    out = s.effective_out_dir()
    opts = dict(
        max_gap_s=float(s.max_gap_s),
        deghost=bool(s.deghost),
        proxy_operator=s.proxy_operator,
        proxy_ev=float(s.proxy_ev),
        run_sfm=bool(s.run_sfm),
        overrides=overrides,
    )
    with JOB.lock:
        JOB.running = True
        JOB.cancel_requested = False
        JOB.stage = "starting"
        JOB.fraction = 0.0
        JOB.report = None
        JOB.error = ""
        JOB.out_dir = out
    threading.Thread(target=_worker, args=(src, out, opts), name="hdr-brackets", daemon=True).start()
    return True, f"merging {src} -> {out}"


def _worker(src: str, out: str, opts: dict) -> None:
    report, error = None, ""
    try:
        from hdr_brackets.pipeline import Options, run

        def progress(stage: str, fraction: float) -> None:
            with JOB.lock:
                JOB.stage, JOB.fraction = stage, float(fraction)
            lf.ui.schedule_on_ui_thread(_redraw)

        report = run(src, out, Options(**opts), progress)
    except Exception as exc:  # shown in the panel, never raised into the host
        lf.log.error(f"hdr_brackets: {exc}")
        error = f"{type(exc).__name__}: {exc}"

    def finish() -> None:
        with JOB.lock:
            JOB.running = False
            JOB.report, JOB.error = report, error
            JOB.finished_at = time.time()
            JOB.generation += 1
            if not error:
                JOB.stage, JOB.fraction = "done", 1.0
        _redraw()

    lf.ui.schedule_on_ui_thread(finish)


# ── Operator ──────────────────────────────────────────────────────────────────


class MergeOperator(Operator):
    label = "Merge Exposure Brackets"
    description = "Align, calibrate and merge exposure brackets to linear EXR using the HDR Brackets panel settings"

    # Registered id (the host derives it from module + qualname; it cannot be overridden):
    #   lfs_plugins.hdr_brackets.operators.MergeOperator
    @classmethod
    def poll(cls, context) -> bool:
        return not JOB.running

    def execute(self, context) -> set:
        ok, msg = start_job()
        if ok:
            lf.log.info(f"hdr_brackets: {msg}")
            return {"FINISHED"}
        lf.log.warn(f"hdr_brackets: {msg}")
        return {"CANCELLED"}
