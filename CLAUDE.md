# LichtFeld Studio — HDR fork

## Goal
1. **Bracket plugin** (Python, no core changes): align exposure brackets, recover camera response, merge to linear HDR (EXR), and emit a tonemapped LDR copy for COLMAP/GLOMAP.
2. **HDR training** (core fork, C++/CUDA/Vulkan): Gaussians store linear, unbounded radiance; training supports per-image exposure; viewer tone-maps; exports keep HDR values.

The work follows `PLAN.md`. Do one milestone at a time, in order. Don't start the next milestone until the current one meets its "Done when" criteria.

## Ground rules
- **Explore before editing.** At the start of each milestone, map the relevant code (use the Explore agent) and write a short findings section into `PLAN.md` under that milestone.
- **Keep the build green.** After every change set: build, run the tests, run the smoke test. Never commit a broken build.
- **Commit per milestone step** with clear messages (`hdr: ...`, `plugin: ...`). Work on branch `hdr`; never force-push `main`.
- **Stay rebasable on upstream.** Prefer additive changes behind a flag (`--hdr` / config `hdr.enabled`) over rewriting existing paths. LDR behavior must be unchanged when HDR is off.
- **License:** fork stays GPL-3.0. Don't copy code from incompatibly-licensed repos (HDR-GS reference code: check its license before porting; reimplement from the paper if unsure).
- **Delegate:**
  - `hdr-architect` (Fable): radiance/exposure math, CUDA kernel or loss changes, Vulkan tone-mapping design, and the review at the end of each milestone.
  - `fast-worker` (Sonnet): boilerplate, tests, UI wiring, formatting, docs.
  - Give subagents a tight brief: the goal, the exact files, the constraints, and what to return.
- **Ask the human** only for: missing test data, GPU/driver problems, or a design choice that changes the file format.

## Build & run (verified 2026-10-27 on v0.5.4.2, RTX A5000, CUDA 12.9, g++-14)
Fork home: `~/Documents/Studio/apps/LichtFeld/LichtFeld-HDR` — a git worktree of the production checkout
(`../LichtFeld-Studio`, v0.5.4, N'Brain's build; never touched). Remotes: `origin` = upstream, `fork` = rdonnot.
```bash
export PATH=$HOME/Documents/Studio/apps/LichtFeld/toolchain/cmake-3.31.12-linux-x86_64/bin:$PATH   # system cmake is 3.28; upstream needs ≥ 3.30
cmake -B build -G Ninja -DCMAKE_MAKE_PROGRAM=/usr/bin/ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=native \
  -DCMAKE_TOOLCHAIN_FILE=$HOME/Documents/Studio/apps/LichtFeld/vcpkg/scripts/buildsystems/vcpkg.cmake \
  -DLFS_DOWNLOAD_CACHE_DIR=$HOME/.cache/lichtfeld/downloads -DLFS_VCPKG_MAX_CONCURRENCY=6 \
  -DCMAKE_CXX_STANDARD_LIBRARIES=-lstdc++exp        # std::stacktrace link fix (0.5.4 needed it)
cmake --build build -j10                            # -j capped: this machine has crashed on RAM
./build/LichtFeld-Studio --version

# Python tests (no LibTorch): app module from build/ is cpython-312, so the venv must be 3.12
uv venv --python 3.12 build/pytest-venv && uv pip install --python build/pytest-venv/bin/python pytest pytest-xdist numpy pillow cryptography
LFS_BUILD_DIR=$PWD/build build/pytest-venv/bin/python -m pytest tests/python -q      # ~10 min, toolchain cmake on PATH
# C++ tests: separate build (its own vcpkg install: the `tests` feature adds GTest, so it cannot reuse build/vcpkg_installed),
# LibTorch 2.13.0+cu129 unpacked at external/libtorch/ — the SDK does not ship libcusparseLt.so.0 / libnvshmem_host.so.3
# that libtorch_cuda.so links: symlink them into external/libtorch/lib/ from the pip wheels
# nvidia-cusparselt-cu12 + nvidia-nvshmem-cu12 (installed in build/libtorch-deps), or the link fails with undefined cusparseLt*/nvshmem*
cmake -S . -B build/tests -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON <same compiler/toolchain flags as above> && cmake --build build/tests --target lichtfeld_tests tensor_hardening_tests -j6   # ~35 min
# vcpkg's bundled python3.12 runs the lichtfeld.python.* ctest entries: give it pytest first
uv pip install --python build/tests/vcpkg_installed/x64-linux/tools/python3/python3.12 --break-system-packages pytest pytest-xdist numpy pillow cryptography
ctest --test-dir build/tests --output-on-failure -L fast -E '^lichtfeld.fast$'   # 13 tests, all green on v0.5.4.2
ctest --test-dir build/tests -R '^lichtfeld.fast$'   # the gtest binary: a 60 s TIMEOUT property it cannot meet here (≈10 min, GPU shared);
                                                     # run build/tests/tests/lichtfeld_tests with its --gtest_filter directly. 27 Formats/Pipelined*Decode
                                                     # cases + PipelinedLoaderLedger need the MipNeRF360 scene at data/bicycle (TEST_DATA_DIR = <repo>/data), absent here.

# smoke-train (headless; `--test-every 8` holds out every 8th image, `--eval` is what writes <out>/metrics.csv)
./build/LichtFeld-Studio --headless --train -d data/smoke_ldr -o out/smoke_ldr --iter 3000 --test-every 8 --eval --export ply
./build/LichtFeld-Studio --headless --train -d data/smoke_hdr -o out/smoke_hdr --iter 3000 --test-every 8 --eval --export ply --hdr   # M2+
cd tools && uv run python eval/ldr_regression.py ../out/smoke_ldr --milestone M0 --scene smoke_ldr --run baseline_a
```
Traps: the shared `../vcpkg` registry checkout must be at least as new as the versions `vcpkg.json` overrides
(2026-10-27: moved to vcpkg master — "no version database entry for assimp at 6.0.5"); the x64-linux triplet also
builds debug variants of every port (upstream's `linux-release` preset uses a release-only triplet but a new triplet
name restores nothing from the binary cache). Tooling (eval, synthetic data) runs in `tools/` (a uv project:
`cd tools && uv run …`, `uv run pytest`), never in the app's embedded Python.

## Test data (`data/` and `out/` are gitignored)
- `data/smoke_ldr/` → symlink to `~/Documents/Projects/Pending/paris-metro-survivor/c2/sets/G1` (122 PNG 768×1344,
  `sparse/0`, no EXIF) — the LDR regression scene. Prior result on the production build: 24.04 dB @30k (mcmc).
- `data/brackets_synth/gt/` — SYNTHETIC ground truth (no real bracketed capture exists yet, Romain 2026-10-27):
  `tools/synth/make_scene.py` renders a Blender room (open wall to a sky HDRI, an emissive lamp, a dark alcove,
  ≈14 stops) from 40 cameras to linear EXR + `cameras.json` + `sparse_gt/` (COLMAP text model, exact poses).
- `data/brackets_raw/` — `tools/synth/make_brackets.py` turns those EXRs into IMG_NNNN.jpg at −2/0/+2 EV through a
  known toe/shoulder CRF (`crf.json`, Debevec form), shot + read noise, real EXIF; `exposures_gt.json` is the truth
  for grouping and exposure. A real capture drops in under the same layout later.
- `data/smoke_hdr/` — produced by the plugin from `brackets_raw` (M1 output).
- Hold out ~10% of positions for evaluation (`--test-every 8`). **Never train on held-out images.**

## Metrics (scripts in `tools/eval/`, rows in `out/metrics.csv`: timestamp,commit,milestone,scene,run,metric,value)
- **Training is not reproducible** (no seed flag; MRNF refinement is stochastic): five identical 7000-iteration runs
  on `smoke_ldr` gave PSNR 26.90–29.20 (mean 27.58, std 0.93 dB), SSIM 0.941–0.951, LPIPS 0.118–0.126. A ±0.1 dB
  criterion cannot be read off fresh trainings. So the LDR guarantee is two-tier:
  1. **Bit-exact** (every rasterizer / loader / eval change): `ldr_reeval.py ../out/baseline_a/project.licht
     --reference ../out/baseline_a` re-evaluates the frozen M0 reference project with the current binary (hdr off) —
     `metrics.csv` row and the 16 held-out eval PNGs must be pixel-identical (evaluation of a frozen project IS
     deterministic; verified twice). `out/baseline_a/` is the reference — do not delete it.
  2. **Statistical** (training-loop changes): `ldr_regression.py <out dir>` logs a fresh run and passes when |ΔPSNR|
     ≤ 0.1 dB + the baseline std (0.93 dB) against the mean of the `baseline_*` rows; several runs beat one.
- HDR: PSNR on held-out views **at each held-out exposure** (render HDR → that image's exposure + CRF → compare with
  the LDR photo: `hdr_metrics.py ldr-psnr`), and μ-law PSNR (μ = 5000, both images normalised by the reference's
  P99) against the merged EXR / the synthetic truth: `hdr_metrics.py mu-psnr`. HDR eval renders as EXR arrive with M2.
- Python suite of the app: `build/pytest-venv` (Python 3.12 = the embedded module's ABI; pytest, pytest-xdist, numpy,
  pillow, cryptography) with the toolchain CMake on PATH (one test configures CMake): 4225 passed, 277 skipped on
  v0.5.4.2 (2026-10-27). C++: `-L fast` minus the gtest binary = 13/13; the gtest binary passes everything except the real-data loader checks that need `data/bicycle` (MipNeRF360, not on this machine).

## Code map (v0.5.4.2, read 2026-10-27; line numbers drift — grep the quoted text)
- **Training loop / loss:** `src/training/trainer.cpp` — render → (PPISP or exposure grid) → `corrected_image.clamp_(0,1)`
  (~:7110, skipped when PPISP is on because its CRF clamps) → `compute_photometric_loss_with_gradient` (~:1660) →
  fused L1 + D-SSIM in `src/training/kernels/ssim.cu` (`C1 = 0.01², C2 = 0.03²` assume [0,1], ~:2031 and 7 more).
  Hook for a μ-law/log pre-loss transform: wrap render + GT right before that call, scale the returned gradient back.
- **Gaussian colour / SH:** fastgs `rasterization/include/kernel_utils.cuh:290` `0.5 + SH_C0·sh0` (no sigmoid), backward
  ~:1106; per-splat clamp `[0, max_blend_color = 4.0]` (`rasterization_config.h:38`) in `kernels_forward.cuh` and the
  mask in `kernels_backward.cuh` ~:935 — the ONLY radiance ceiling. gsplat/`gut` path `SphericalHarmonicsCUDA.cu`
  has no clamp. `dc_reg` (`kernels_backward.cuh` ~:244) is a hidden [0,1] prior on sh0 (0 in N'Brain's configs).
- **PPISP (per-camera ISP, NVIDIA, Apache-2 + GPL):** `src/training/components/ppisp.cpp`, `ppisp_controller.cu`,
  `kernels/ppisp.cu` (exposure `exp2(EV)` per frame → vignetting per camera → colour homography per frame → CRF per
  camera with input clamped [0,1]); EXIF seed `ppisp.cpp:243` `clamp(0.5·(ev − mean), ±16)` from `core/exif.cpp`
  (`ev = log2(t / N² · ISO)`, tags 0x829A/0x829D/0x8827). `--exposure-correction` = PPISP + bilateral grid, ON by
  default, exclusive with `--ppisp`. Anchor: `exposure_mean` reg + `project_mean()`.
- **Input:** float EXR decoder `src/core/image_exr.cpp` + `load_image_float` exist, but training takes the UInt16 path
  (`io/pipelined_image_loader.cpp`, `/65535`); `decode_float` is set only by metrics; `--use-16bit` only flips
  `output_uint8` (`dataset.hpp:546`); `.exr` is not in `is_image_file` (`io/include/io/filesystem_utils.hpp:352`);
  undistort clamps [0,1]. No sRGB→linear anywhere in training.
- **Vulkan viewer:** SH `rendering/rasterizer/vulkan/shader/src/slang/spherical_harmonics.slang` (+0.5, max 0);
  float blending (`alphablend_shader.slang`); tone map AFTER blending in
  `visualizer/rendering/resources/viewport/vksplat_compose.comp:125,133` `lfsDisplayTone(rgb, tonemapping, exposure)`
  → clamp → `R8G8B8A8_UNORM` (`vksplat_viewport_renderer.cpp` ~:5888); curves none/linear/filmic/hejl/aces/aces2/
  neutral in `slang/display_color.inc`; exposure 0.1–8 (`py_rendering.cpp:732`). No OETF after the tone map.
- **Export:** PLY `src/io/formats/ply.cpp` ~:3319 writes `f_dc_*` raw float32, header `comment` lines supported;
  SPZ (`external/spz/load-spz.cc:529`) and SOG (`sogs.cpp`) quantise; no float image writer (`core/image_io.cpp`
  jpg/png/tif, uint8). `io/hdr_tonemap.cpp` + `hdr_libplacebo.cpp` are HDR *video* decode, unrelated.
- **Python plugin API:** `docs/plugins/getting-started.md`, `api-reference.md`, `plugin-reference.md`; plugins in
  `~/.lichtfeld/plugins/<name>/` (`pyproject.toml` with `[tool.lichtfeld]`, `__init__.py` `on_load/on_unload`, own
  `.venv` via bundled Python 3.12 + `uv sync`). Panels `lf.ui.Panel.draw(ui)`; long jobs = `threading.Thread` +
  `lf.ui.schedule_on_ui_thread` (see `src/python/lfs_plugins/asset_index.py`, `bug_report_panel.py`); progress
  `ui.progress_bar`; preview `lf.ui.DynamicTexture` + `ui.image`; undo `lf.undo.push(name, undo, redo)` /
  `lf.undo.transaction`; hand a dataset to training `lf.scene.load_file(path, is_dataset=True, output_path=…)`;
  hooks `@lf.on_training_start / on_iteration_start / pre_optimizer_step / post_step / end`. Example:
  `docs/plugins/examples/full_plugin/`.
- **MCP server:** `.mcp.json` → `scripts/lichtfeld_mcp_bridge.py` (starts the app, proxies `http://localhost:45677/mcp`);
  `scene_load_dataset`, `training_start`, `runtime_job_wait`, `editor_run`; read `lichtfeld://runtime/catalog` first
  (repo `AGENTS.md`, `docs/docs/development/mcp/`).
- **Tests:** C++ `build/tests` (LibTorch), labels fast/slow/nightly/gpu; Python `tests/python` (`LFS_BUILD_DIR`);
  headless `--eval --test-every N` writes `metrics.csv`, `per_image_metrics.json`, `eval_step_N/` PNGs (8/16-bit).
