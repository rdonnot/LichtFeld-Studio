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
