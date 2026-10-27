# HDR fork — milestone plan

Each milestone: Explore → plan (written here) → implement → build/test → `hdr-architect` review → commit.

---

## M0 — Baseline & code map
- Fork, clone, create branch `hdr`. Build upstream unchanged; run tests and `smoke_ldr`; record baseline metrics in `out/metrics.csv`.
- Fill the "Code map" in CLAUDE.md with real paths.
- Read the plugin API docs + one existing plugin; note how plugins add UI panels, run jobs, register undo, and hand data to training.
- Set up `tools/eval/` (render held-out views, PSNR/SSIM, CSV logging).

**Done when:** upstream builds, baseline metrics logged, code map complete.

---

## M1 — Bracket-merge plugin (Python)
Plugin folder: `plugins/hdr_brackets/` (own uv env).
1. **Group** images into brackets: by EXIF timestamp gaps + exposure time; manual override UI for odd cases.
2. **Align** within each bracket (`cv2.createAlignMTB`); flag brackets with large residual motion as possible ghosting.
3. **Response curve** once per camera: `cv2.createCalibrateDebevec` on a sample of brackets; save the CRF to the project.
4. **Merge** → linear float EXR per position (`cv2.createMergeDebevec`), optional deghosting weights.
5. **Proxy LDR** for SfM: tonemap (Reinhard/ACES-ish), 8-bit, same filenames → run COLMAP/GLOMAP on these.
6. **Output dataset** `smoke_hdr/`: `images_hdr/*.exr`, `images/*.png` (proxy), `sparse/`, and `exposures.json` (per source image: exposure time, EV, bracket id) for M3.
7. UI: folder picker, progress, preview (proxy vs. exposure slider on EXR).

**Done when:** `brackets_raw` → `smoke_hdr` in one click/command, SfM registers ≥95% of positions, EXR values are sane (no clipping, monotonic with exposure), unit tests for grouping + merge.

---

## M2 — Linear HDR color in the core
- Add `hdr.enabled` flag/CLI `--hdr`.
- Dataset loader: read EXR (float) when HDR is on; keep the 8-bit path untouched.
- Color representation: SH DC + rest stay, but the output activation becomes unbounded. Store log-radiance or use `exp`/softplus on the DC term (decide with `hdr-architect`; document why).
- Remove/branch any `[0,1]` clamps in the CUDA forward/backward and the loss when HDR is on.
- Loss in HDR mode: L1 + D-SSIM on a **tonemapped** (μ-law or log) version of render and target, so bright regions don't dominate.
- Densification thresholds: check whether gradient magnitudes change scale in HDR mode; re-tune if needed.

**Done when:** trains on `smoke_hdr` EXRs without NaNs; log-luminance PSNR beats a baseline trained on the proxy LDR; `smoke_ldr` regression within ±0.1 dB.

---

## M3 — Train directly from brackets (exposure model)
Alternative input path that skips merging (avoids ghosting):
- Use all bracket images as training views (shared pose per bracket, from SfM on the proxies).
- Per-image learnable exposure offset (init from EXIF EV) + a shared learnable CRF (small monotonic MLP or piecewise-linear, per HDR-GS); render HDR → exposure → CRF → compare with the raw LDR image in the LDR domain.
- Regularize: monotonic CRF, exposure offsets near EXIF, CRF fixed to M1's Debevec curve as an option.

**Done when:** held-out PSNR at each exposure ≥ the M2 (merged) approach, or a written explanation of why not.

---

## M4 — Viewer tone mapping (Vulkan)
- Render target in float (RGBA16F/32F) when HDR is on.
- Tone-mapping pass: exposure slider (EV), operators (linear-clamp, Reinhard, ACES-fit, AgX-like), gamma/sRGB out.
- Optional: false-color luminance view for debugging.
- Keep the LDR viewer path identical when HDR is off.

**Done when:** viewer shows smooth exposure sweeps on the HDR scene with no banding, and FPS within 10% of the LDR path.

---

## M5 — Export & interop
- PLY: write unbounded color (flag in header, e.g. `comment hdr_linear`); document the convention.
- Optional exports: baked LDR at a chosen exposure/operator (works in any viewer), EXR renders, camera-path video with exposure keyframes.
- Check SPZ/RAD paths: either support HDR or refuse clearly.

**Done when:** round-trip PLY load/save keeps values; baked LDR export opens correctly in a stock viewer.

---

## M6 — Polish & upstream-ability
- Tests: plugin unit tests, HDR training smoke test in CI (small iteration count), shader compile check.
- Docs: `docs/hdr.md` (capture guide: bracket spacing, tripod vs. handheld, white balance locked) + plugin README.
- Rebase on upstream; split into reviewable PRs if offering it back (plugin first, then core flag).
- Final `hdr-architect` review of the whole diff.

---

## Findings log
(Claude appends notes per milestone here.)

### M0 — 2026-10-27
**Setup.** Fork `rdonnot/LichtFeld-Studio`; branch `hdr` from `v0.5.4.2` as a worktree at
`~/Documents/Studio/apps/LichtFeld/LichtFeld-HDR` (the production `LichtFeld-Studio` checkout stays v0.5.4 for
N'Brain). Build traps: system CMake 3.28 < 3.30 (use `toolchain/cmake-3.31.12`); the shared vcpkg registry had to move
to master because `vcpkg.json` overrides pin versions (assimp 6.0.5, openssl 3.6.5, curl 8.23 …) absent from both the
old checkout and the `builtin-baseline`; the x64-linux triplet builds every port twice (dbg + rel).

**Upstream already has most of the HDR plumbing** (file:line in CLAUDE.md's code map), so the milestones shrink:
- Float EXR decoding exists; training just never asks for it (`decode_float` only in metrics, `.exr` not an image
  extension). M2 input work = three lines + skipping the undistort clamp.
- The rasterizer's only radiance ceiling is `max_blend_color = 4.0` (fastgs forward + backward mask); the trainer
  clamps the corrected image to [0,1] before the loss unless PPISP is on. No sigmoid anywhere. M2 colour work =
  flag-gated clamp removal, `dc_reg` off, SSIM constants / densification scale checked.
- **PPISP is M3's exposure model**: per-frame `exp2(EV)` gain seeded from the same EXIF tags our brackets carry,
  shared per-camera CRF (toe/shoulder/gamma) and vignetting, a scale anchor (`exposure_mean` + `project_mean`). What
  is missing: an exposure-only mode (its CRF clamps input to [0,1] and runs even with `train_crf=false`), an absolute
  seed `log2(ISO·t/N²)` instead of `0.5·(ev − mean)`, and "CRF fixed to M1's Debevec curve". No HDR-GS port needed.
- The viewer already tone-maps after float blending (`lfsDisplayTone`, 7 operators, exposure ×0.1–8) into an 8-bit
  target; M4 = an OETF after the tone map for linear scenes, an EV-range slider, false colour, banding check.
- PLY writes raw float SH (M5 = a header comment + a float image writer); SPZ/SOG quantise → refuse or bake.
- Eval already writes `metrics.csv` / `per_image_metrics.json` (PSNR with `data_range=1`, PNG renders) →
  `tools/eval/ldr_regression.py` wraps it; HDR eval needs EXR renders (M2).

**Data.** No bracketed capture on the machine; Romain chose synthetic-only for now. `tools/synth/make_scene.py`
(Blender 4.5, Cycles) renders the ground truth, `make_brackets.py` the EXIF'd JPEG brackets through a known CRF; the
tests in `tools/` prove the eval path reproduces the JPEGs from EXR + `exposures_gt.json` + `crf.json`.
`data/smoke_ldr` = the paris-metro-survivor G1 scan (122 PNGs, prior 24.04 dB @30k).

**Plugins.** No job API: a `threading.Thread` + `lf.ui.schedule_on_ui_thread`; progress bar, `DynamicTexture`
preview, `lf.undo.push/transaction`, `lf.scene.load_file(path, is_dataset=True)` to hand the merged dataset to
training; plugin deps in its own uv `.venv`. Bundled examples: `docs/plugins/examples/full_plugin/`.
