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

### M2 design decision (hdr-architect, 2026-10-27) — implement exactly this, file:line read on v0.5.4.2

**Decision.** Store **log-radiance in the SH**: in HDR mode the full SH sum `b = 0.5 + SH_C0·sh0 + Σ Yᵢcᵢ`
(`kernel_utils.cuh:290-313`) becomes `c = exp(min(b, ln cmax))`, `cmax = 1e4`; the forward clamp becomes a runtime
`color_max` (4.0 unchanged for LDR, `kernels_forward.cuh:710`, `kernels_backward.cuh:935-939`); the lower clamp is a
no-op (exp > 0). The loss runs on **μ-law of the P99-normalised image**, `y = log1p(μ·x)/log1p(μ)`, `x = v/s`,
μ = 5000, applied to render and GT right before `compute_photometric_loss_with_gradient` (`trainer.cpp:1660`), the
gradient scaled back at `trainer.cpp:7883`. PPISP / bilateral grid are **refused** under `--hdr` (M3 brings the
exposure-only PPISP). Densification needs **no retune**. Eval writes EXR renders and reports μ-law PSNR; CRF/exposure
PSNR stays in `tools/eval`.

*Why log, not a raised clamp.* Adam's step on sh0 is bounded by `shs_lr = 0.0025` (`parameters.hpp:251`): with linear
colour `Δc ≈ SH_C0·lr ≈ 7e-4` radiance/step, a sun splat at x = 1000 needs ~1.4 M steps; in log space the same step is
×1.0007/step → ×1000 in ~10 k steps. The chain is self-conditioning: `dL/db = c·dL/dc = (μx/(1+μx))/ln(1+μ)·dL/dy`,
bounded in `[0, 0.117]` for every brightness — no explosion for radiance ≫ 1, no vanishing in shadows. Negative ShN lobes
become multiplicative darkening (no zero clamp needed). ShN is q16 with one (min,max) per 256 splats
(`kernel_utils.cuh:78-80`, `sh_value_codec.hpp:10`): linear 1e3 lobes would ruin a block's step (0.015), log lobes are
O(1). sh0 is float32 (`kernels_backward.cuh:250`); the f16 ShN swizzle is only a viewer bind (`kernel_utils.cuh:61`),
training storage is q16 (`training_setup.cpp:708`) → no 65504 overflow. **Cost / file format:** PLY `f_dc` becomes log
radiance (M5 writes `comment hdr_log_sh`; stock viewers show garbage until baked) and the viewer SH shader needs an
`exp` (M4). Init: `sh0 = (ln(clamp(p,1e-3,1)) − 0.5)/SH_C0` from the proxy colour p (`mrnf.cpp:2542`,
`splat_data.cpp:2889`). μ = 5000 over log1p/PQ: y′(0)/y′(1) ≈ 5000 matches the room's ~14 stops, PQ needs an absolute
nit anchor we lack, and `tools/eval/hdr_metrics.py` already uses this μ with P99 normalisation.

**Changes.**
- `core/include/core/parameters.hpp` (OptimizationParameters): `bool hdr = false; float hdr_mu = 5000.f;
  float hdr_scale = 0.f /*0 = P99 at load*/; float hdr_color_max = 1e4f;`. Follow `gut`: `argument_parser.cpp:102`
  `OptimizationCliBinding{"--hdr","hdr",Bool}`, `:914` `::args::Flag`, `:1615` capture, `:1800` `setFlag`, `:1911`
  `note_opt`; `optimization_properties.cpp:1027` `.bool_prop(&OptimizationParameters::hdr, …)` (JSON follows,
  `parameters.cpp:808,1323`); `py_params.cpp:1109` `def_prop_rw("hdr", …)`. Validation beside `parameters.cpp:1035`:
  `hdr && (use_exposure_correction || use_bilateral_grid || use_ppisp || gut)` → error (MRNF defaults set
  `use_exposure_correction = true`, `parameters.cpp:1251` — `--hdr` must clear it); `hdr` forces `dc_reg = 0`.
- Input: `trainer.cpp:8856` `pipelined_config.use_16bit_color |= hdr`; `dataset.hpp:546` add
  `request.params.decode_float = hdr; request.params.hdr_linear = hdr;` (new `LoadParams` field,
  `cache_image_loader.hpp:50`); `pipelined_image_loader.cpp:554` `if (!params.hdr_linear) tensor = tensor.clamp(0,1)`
  (keeps metrics' float path bit-identical); `filesystem_utils.hpp:352` add `".exr"`. EXR already probes as Float32
  (`image_io.cpp:928-938` → step 0 → `decodes_float`, loader `:443`) and the float path skips
  `quantize_rgb_to_u16_grid` (loader `:575`).
- Scale `s`: at dataset load when `hdr_scale == 0`: P99 of luminance over ≤ 16 training EXRs, written back into
  `params.optimization.hdr_scale` before `write_training_config` (`metrics.cpp:1329`) so a resume reuses it; log it.
- Rasterizer: `rasterization_api.h:21` `FastGSSettings`: `float color_max = 4.f; bool log_radiance = false;` threaded
  through `forward_raw`/`backward_raw` (`rasterization_api.cu:333,675`). `kernel_utils.cuh:313`:
  `if (log_radiance) result = expf(fminf(result, log_color_max));` (template bool on the preprocess kernels so LDR code
  is unchanged; `expf`, not `__expf`). `kernels_forward.cuh:710` / `kernels_backward.cuh:935-939`:
  `config::max_blend_color` → `color_max`. `kernels_backward.cuh:1085`: `dL_dcolor = blending_weight * grad_c *
  color_grad_factor * (LOG ? color : 1)` — `color` is the staged exp output in registers; the cap mask
  (`c ≤ color_max` ⇔ `b ≤ ln cmax`) works unchanged. `convert_sh_to_color_backward_grads` (`:1118`) receives `dL/db`
  and needs no edit.
- Loss: new `kernels/hdr_mulaw.cu`: `hdr_mulaw_forward(v, s, μ) → y, x_saved` (x = max(v,0)/s) and
  `hdr_mulaw_backward_(g, x_saved, s, μ)`: `g *= μ/((1+μx)·ln(1+μ)·s)`, 0 where v < 0. `trainer.cpp:6280`: after
  `stable_gt`/composite, `if (hdr) gt_image = stable_gt = mulaw(gt)` — feeds the edge map (`:5871`), thin-structure
  map, error map and loss. `trainer.cpp:7110/7132`: `else if (hdr) corrected_image = mulaw(output.image)` (keep
  `x_saved`), the clamp stays in the LDR branch. `trainer.cpp:7883`: `raster_grad = tile_grad; if (hdr)
  mulaw_backward_(raster_grad, x_saved)`. SSIM `C1/C2` (`ssim.cu:1977,2031…`) stay valid: both inputs in [0,1].
  `gradient_loss_weight` (`trainer.cpp:1645`), the L1 fallback error map (`:7775`) and the SSIM/CS error map
  (`:7711-7741`) read the transformed tensors — automatically μ-law. `raw_loss_input` (`:7137`) only exists with
  grid/PPISP, which HDR refuses.
- Densification: MRNF and MCMC accumulate `blending_weight · pixel_error` from the SSIM/CS error map
  (`kernels_backward.cuh:1082,1115`; `mrnf_kernels.cu:330`; `mcmc.cpp:739`) — dimensionless in [0,1] on μ-law images,
  so `growth_grad_threshold = 0.003` (`parameters.hpp:379`) keeps its meaning. The positional path
  (`length(dL_dmean2d·½wh)`, `:536`) runs only for `None` without an error map (`backward.cu:208`); the mean2d
  gradient (`:1108`) carries `(dL/dy)·μ(xᵢ−x_behind)/((1+μx)ln(1+μ))`, same order as LDR. No retune; verify by
  histogram of `_refine_weight_max` (`mrnf.cpp:1836`).
- Eval: `metrics.cpp`: in HDR force `bit_depth = 32`, replace `image.clamp(0,1)` (`:251`) by `mulaw(x/s)` for GT
  (`:656`) and render (`:1056`) → PSNR/SSIM at `data_range = 1` (`:1449`) are μ-law metrics (say so in metrics.csv).
  Before that, save the linear render: new `save_image_exr(path, CHW float)` in `core/image_exr.cpp` — OpenEXRCore is
  linked (`src/core/CMakeLists.txt:160`) and ships the write API (`exr_start_write` `openexr_context.h:467`,
  `exr_write_header` `:580`, `openexr_encode.h`): start_write → add_part(scanline) → required_attr_simple → channels
  R,G,B FLOAT → ZIP → write_header → per chunk `exr_write_scanline_chunk_info` + encoding init/default/run (~80 lines).
  Call beside the PNG save (`metrics.cpp:2117`) as `eval_dir/<stem>.exr`. LDR-at-source-exposure PSNR:
  `tools/eval/hdr_metrics.py` over `eval_step_N/*.exr` + `exposures_gt.json` + `crf.json` — not in core.

**Risks.** Log init far from truth (sky/lamp at ln 1 = 0 vs ln 1000): ~10 k Adam steps to climb; if highlights lag at
30 k add `hdr_sh0_lr_scale = 4` — watch max `exp(b)` per eval. `expf` overflow capped at `ln cmax`. `bg_color` is
radiance now: white = 1.0 after normalisation; default black. A resume with a different `s` silently changes the loss
→ `hdr_scale` written back. LOD colour decode (`bhatt_lod.cpp:146`, `octree_lod.cpp:135`) and the Vulkan SH shader
read linear → wrong until M4 (viewer shows log values meanwhile). `thin_structure`/mask paths assume `gt_tile` Float32
— true in HDR, check once.

**Verification.** `tests/test_hdr_color_chain.cpp` in `lichtfeld_tests` (list `tests/CMakeLists.txt:773`, labels
`fast;gpu`), gtest + LibTorch double reference as `test_fused_l1_ssim.cpp:13-80`: (a) μ-law kernel vs autograd on
x ∈ [−1, 50], rel < 1e-5, zero grad for x < 0; (b) a `__global__` wrapper around `convert_sh_to_color` (pattern
`tests/fastgs_tile_culling_boundary_cuda.cu`), 32 splats, SH3: forward = `exp(min(b, ln cmax))`, `dL/dsh0 =
SH_C0·c·g` and 0 above the cap, central differences h = 1e-3, rel < 1e-4; (c) `log_radiance = false` bit-equals the
old path (`memcmp`). `ctest -L fast`; `ldr_regression.py` on `smoke_ldr` ±0.1 dB and the LDR run's eval PNGs
`cmp`-identical to the M0 baseline at the same seed. `--hdr --iter 3000 --test-every 8` on `smoke_hdr`: finite loss
every step, no NaN in `eval_step_*/*.exr`, μ-law PSNR > the proxy-LDR baseline, `hdr_metrics.py` CRF PSNR per held-out
exposure logged; `_refine_weight_max` histogram overlaps the LDR run's.

### M2 implementation notes (hdr-m2, 2026-10-27) — what was built, what deviates, what is left

Built as designed, commit by commit on `hdr-m2`: the flag and parameters (`hdr`, `hdr_mu`, `hdr_scale`, `hdr_color_max`;
`--hdr` clears MRNF's `use_exposure_correction` and `dc_reg`, validation refuses the grid / PPISP / `gut` with the reason
in the message), the linear EXR input (`.exr` is an image, `LoadParams::hdr_linear` + `PipelinedLoaderConfig::hdr_linear`
skip the undistort clamp AND the derived JPEG/JPEG 2000 cache — that cache would have clipped radiance > 1 silently —
`use_16bit_color` forced, P99 `s` over ≤ 16 EXRs written back before the evaluator copies the params), the rasterizer
(`ColorActivation{color_max, log_radiance}` threaded `fast_rasterize_forward` → `forward_raw` → kernels, stored in the
context for the backward; `preprocess_cu<bool LOG_RADIANCE>`; `log_radiance_color` in `kernel_utils.cuh`; runtime blend cap;
`dL/db = c·dL/dc` in the blend backward), the μ-law loss (`kernels/hdr_mulaw.cu`; GT transformed once after the composite so
`source_gt`/`stable_gt`/`gt_image` all feed the edge, thin-structure, error maps and blob seeding in μ-law), the eval
(`save_image_exr` on OpenEXRCore, `eval_step_N/<stem>.exr` of the LINEAR render, PSNR/SSIM/LPIPS on `mulaw(x/s)`, the domain
named in `metrics_report.txt`), `tests/test_hdr_color_chain.cpp`, `tools/synth/make_hdr_dataset.py`.

**Deviations.** (1) *Cap semantics:* a channel at or above `ln cmax` is stored as exactly `color_max` and the backward's
cap mask is STRICT in HDR (`c < color_max`), so "0 above the cap" holds without the forward/backward disagreeing on
`expf(logf(cmax))` rounding; LDR keeps `<=`. (2) *Where the activation lives:* not on `FastGSSettings` (unused by any
caller) but as a trailing defaulted parameter of every entry point and a field of `FastRasterizeContext`; every training
render passes `color_activation_for(hdr, hdr_color_max)` — trainer, eval (`metrics.cpp`), timelapse, POPSpa scores.
`application.cpp`'s video render of a loaded model and the viewer do not (M4). (3) *Init in the trainer, not at the init
sites:* `Trainer::init_log_radiance_sh0` rewrites sh0 once on a fresh start (covers point cloud, `--random`, PLY); MRNF
blob seeds invert the μ-law before taking the log (`mrnf.cpp`). (4) *Loader:* `--images <folder>` resolves a same-stem
image of another extension (`resolve_image_extension`) — the M1 output names `pos_000.png` in its sparse model and
`pos_000.exr` in `images_hdr`. (5) *`hdr_color_max` is absolute* and the M1 merge is in sensor units (`s` = 2419 on
`smoke_hdr` vs 9.4 on the synthetic truth): the lamp reaches the 1e4 cap (max rendered radiance 9631). Either the cap
should scale with `s` (`cmax = k·s`) or the plugin should normalise its EXRs — decide in M3/M5 with the file format.
(6) *μ-law PSNR vs the proxy baseline* is not one number: the proxy model's renders live in the proxy's tone curve.
Reported side by side below. (7) `tests/image_codecs` is skipped when the OpenEXR C++ library is absent (core-only port).

**Measured (3000 iterations, `--test-every 8 --eval`, GPU shared with another training).**
- LDR bit-identity: `ldr_reeval.py` on the frozen M0 `baseline_a` project with this binary — "metrics: identical",
  "eval PNGs: 16 compared, identical". Fresh LDR smoke on `smoke_ldr`: 25.58 dB (stochastic; not a regression number).
- HDR on the M1 output `smoke_hdr --images images_hdr --hdr`: s = 2419.3, 27.51 dB μ-law in-app, `hdr_metrics.py`
  27.49 dB mean over the 5 held-out views (29.93 on `pos_008`) against the merged EXRs, no NaN, max radiance 9631.
- HDR on the synthetic truth `smoke_hdr_gt --images images_hdr --random --hdr`: s = 9.39, 32.98 dB μ-law, 26.5–35.5 per
  view, max radiance 458 (truth 404), no NaN. Loss finite every step (0.12 → 0.033).
- Proxy-LDR baseline on the same poses (`smoke_hdr --images images`): 29.26 dB PSNR in its own 8-bit domain.

**Left for "Done when".** A like-for-like proxy comparison (render the proxy model, invert the plugin's tone curve, μ-law
PSNR against the merged EXR — `tools/eval`), the `_refine_weight_max` histogram overlap check, the CRF/exposure PSNR per
held-out exposure through `hdr_metrics.py ldr-psnr`, and the cap-vs-scale decision in (5).

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
