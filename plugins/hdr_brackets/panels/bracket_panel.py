"""The "HDR Brackets" panel: pick a folder, merge brackets, look at the result.

Immediate-mode only (``draw(ui)``). The merge runs on a worker thread owned by
``..operators``; this panel only reads the shared ``JOB`` state.
"""

from __future__ import annotations

import sys
from pathlib import Path

import lichtfeld as lf

from ..operators import JOB, PROXY_OPERATORS, SETTINGS, parse_overrides, start_job

PREVIEW_MAX_W = 512


def _read_exr(path: str):
    try:
        from hdr_brackets._tools import read_exr
    except ImportError:  # fall back to the fork's eval tools
        root = Path(__file__).resolve().parents[3] / "tools" / "eval"
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from hdr_metrics import read_exr
    return read_exr(path)


def _downscale(img, max_w: int = PREVIEW_MAX_W):
    import cv2

    h, w = img.shape[:2]
    if w <= max_w:
        return img
    return cv2.resize(img, (max_w, max(1, round(h * max_w / w))), interpolation=cv2.INTER_AREA)


class BracketPanel(lf.ui.Panel):
    id = "hdr_brackets.main_panel"
    label = "HDR Brackets"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 220

    def __init__(self):
        self._texture = lf.ui.DynamicTexture()
        self._seen_generation = -1
        self._positions: list[str] = []  # EXR stems found in <out>/images_hdr
        self._pos_idx = 0
        self._use_exr = False
        self._ev = 0.0
        self._exr_cache: dict[str, object] = {}  # stem -> downscaled float32 HxWx3
        self._preview_key = None  # what the texture currently shows
        self._preview_dir = ""
        self._preview_open = False
        self._message = ""
        self._ov_error = ""

    # ── draw ──────────────────────────────────────────────────────────────────

    def draw(self, ui):
        s = SETTINGS
        running = JOB.running
        if JOB.generation != self._seen_generation:
            self._seen_generation = JOB.generation
            self._exr_cache.clear()
            self._preview_key = None

        ui.heading("Source")
        ui.begin_disabled(running)
        changed, s.src_dir = ui.path_input("Source folder", s.src_dir, True, "Bracketed images")
        ui.text_disabled("Folder of IMG_*.jpg (or png/tiff) shot as exposure brackets")
        ch, s.out_dir_custom = ui.checkbox("Custom output folder", s.out_dir_custom)
        if s.out_dir_custom:
            if not s.out_dir:
                s.out_dir = s.effective_out_dir()
            _, s.out_dir = ui.path_input("Output folder", s.out_dir, True, "Output folder")
        else:
            ui.text_disabled(f"Output: {s.effective_out_dir() or '<source>_hdr'}")

        ui.separator()
        ui.heading("Options")
        _, s.max_gap_s = ui.slider_float("Max gap between shots (s)", s.max_gap_s, 0.5, 30.0)
        _, s.deghost = ui.checkbox("Deghost moving content", s.deghost)
        _, s.run_sfm = ui.checkbox("Run SfM (COLMAP poses)", s.run_sfm)
        cur = PROXY_OPERATORS.index(s.proxy_operator) if s.proxy_operator in PROXY_OPERATORS else 0
        ch, cur = ui.combo("Proxy tonemap", cur, PROXY_OPERATORS)
        if ch:
            s.proxy_operator = PROXY_OPERATORS[cur]
        _, s.proxy_ev = ui.slider_float("Proxy EV", s.proxy_ev, -3.0, 3.0)

        if ui.collapsing_header("Manual bracket overrides", False):
            ui.text_disabled("FILE=BRACKET, comma separated: IMG_0007.jpg=3, IMG_0008.jpg=3")
            _, s.overrides_text = ui.input_text("Overrides", s.overrides_text)
            try:
                parse_overrides(s.overrides_text)
                self._ov_error = ""
            except ValueError as exc:
                self._ov_error = str(exc)
            if self._ov_error:
                ui.text_colored(self._ov_error, (1.0, 0.45, 0.35, 1.0))
        ui.end_disabled()

        ui.separator()
        self._draw_run(ui, running)
        if not running:
            self._draw_report(ui)

    def _draw_run(self, ui, running: bool):
        if not running:
            ui.begin_disabled(bool(self._ov_error))
            if ui.button_styled("Run", "primary"):
                ok, msg = start_job()
                self._message = "" if ok else msg
            ui.end_disabled()
            if self._message:
                ui.text_colored(self._message, (1.0, 0.45, 0.35, 1.0))
            return
        with JOB.lock:
            stage, frac, cancelled = JOB.stage, JOB.fraction, JOB.cancel_requested
        ui.label(f"Stage: {stage}")
        ui.progress_bar(max(0.0, min(1.0, frac)), f"{int(frac * 100)}%")
        ui.begin_disabled(cancelled)
        if ui.button("Cancel"):
            JOB.cancel_requested = True  # the core has no cancel yet: the run finishes
        ui.end_disabled()
        if cancelled:
            ui.text_disabled("Cancel is not supported by the core yet; the run will finish.")

    # ── report ────────────────────────────────────────────────────────────────

    def _draw_report(self, ui):
        if JOB.error:
            ui.separator()
            ui.text_colored(f"Failed: {JOB.error}", (1.0, 0.45, 0.35, 1.0))
            return
        r = JOB.report
        if r is None:
            return
        ui.separator()
        ui.heading("Result")
        ui.label(f"{r.n_images} images: {r.n_brackets} brackets, {r.n_singletons} singletons ({r.seconds:.1f} s)")
        if getattr(r, "registered", None) is not None:
            ui.label(f"SfM registered: {r.registered} / {r.total}")
        if getattr(r, "crf_path", None):
            ui.text_disabled(f"CRF: {r.crf_path}")
        if r.flagged:
            ui.text_colored(f"Ghosting flagged in {len(r.flagged)} file(s):", (1.0, 0.8, 0.3, 1.0))
            for name in r.flagged[:20]:
                ui.text_disabled(f"  {name}")
            if len(r.flagged) > 20:
                ui.text_disabled(f"  ... and {len(r.flagged) - 20} more")
        for w in r.warnings:
            ui.text_wrapped(f"Warning: {w}")

        out = str(r.out_dir or JOB.out_dir)
        if ui.button_styled("Load dataset", "success"):
            lf.scene.load_file(out, is_dataset=True, output_path=str(Path(out) / "train"))
        ui.same_line()
        if ui.button("Preview"):
            self._preview_open = True
            self._open_preview(out)
        if self._preview_open:
            self._draw_preview(ui, out)

    # ── preview ───────────────────────────────────────────────────────────────

    def _open_preview(self, out: str):
        self._preview_dir = out
        self._positions = sorted(p.stem for p in (Path(out) / "images_hdr").glob("*.exr"))
        if not self._positions:  # fall back to the proxies
            self._positions = sorted(p.stem for p in (Path(out) / "images").glob("*.png"))
        self._pos_idx = min(self._pos_idx, max(0, len(self._positions) - 1))
        self._preview_key = None

    def _draw_preview(self, ui, out: str):
        ui.separator()
        ui.heading("Preview")
        if not self._positions:
            ui.text_disabled("No images found in the output folder.")
            return
        ch, self._pos_idx = ui.combo("Position", self._pos_idx, self._positions)
        ch2, self._use_exr = ui.checkbox("Linear EXR (instead of proxy PNG)", self._use_exr)
        ch3 = False
        if self._use_exr:
            ch3, self._ev = ui.slider_float("Exposure (EV)", self._ev, -6.0, 6.0)
        key = (self._positions[self._pos_idx], self._use_exr, round(self._ev, 2) if self._use_exr else 0.0)
        if key != self._preview_key:
            self._preview_key = key
            try:
                self._render_preview(*key[:2], ev=key[2])
                self._message = ""
            except Exception as exc:
                self._message = f"preview: {type(exc).__name__}: {exc}"
        if self._message:
            ui.text_colored(self._message, (1.0, 0.45, 0.35, 1.0))
        if self._texture.valid:
            w, h = self._texture.width, self._texture.height
            ui.image(self._texture.id, (float(w), float(h)))

    def _render_preview(self, stem: str, use_exr: bool, ev: float):
        import numpy as np

        out = Path(self._preview_dir)
        if use_exr:
            from hdr_brackets.proxy import tonemap_proxy

            img = self._exr_cache.get(stem)
            if img is None:
                img = _downscale(np.asarray(_read_exr(str(out / "images_hdr" / f"{stem}.exr")), dtype=np.float32))
                self._exr_cache[stem] = img
            rgb = tonemap_proxy(img, ev=ev, operator=SETTINGS.proxy_operator)
        else:
            import cv2

            bgr = cv2.imread(str(out / "images" / f"{stem}.png"), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(f"images/{stem}.png")
            rgb = cv2.cvtColor(_downscale(bgr), cv2.COLOR_BGR2RGB)
        self._texture.update(lf.Tensor.from_numpy(np.ascontiguousarray(rgb, dtype=np.uint8)))

    def on_unmount(self, doc):
        self._texture.destroy()
