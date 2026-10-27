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

## Build & run (VERIFY against upstream README on first run, then fix this section)
```bash
# configure + build (Release)
cmake -B build -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build build -j

# tests
ctest --test-dir build --output-on-failure

# smoke-train (headless) — adjust binary name/flags to upstream CLI
./build/LichtFeld-Studio -d data/smoke_ldr -o out/smoke_ldr --iter 2000 --headless
./build/LichtFeld-Studio -d data/smoke_hdr -o out/smoke_hdr --iter 2000 --headless --hdr
```

## Test data
- `data/smoke_ldr/` — a small standard COLMAP scene (regression check).
- `data/brackets_raw/` — one small bracketed capture: ≥30 positions × 3–5 exposures (e.g. −2/0/+2 EV), with EXIF exposure times.
- `data/smoke_hdr/` — produced by the plugin from `brackets_raw` (M1 output).
- Hold out ~10% of positions for evaluation. **Never train on held-out images.**

## Metrics (scripts in `tools/eval/`)
- LDR regression: PSNR/SSIM on `smoke_ldr` within ±0.1 dB of upstream `main`.
- HDR: PSNR on held-out views **at each held-out exposure** (render HDR → apply that image's exposure + CRF → compare with the LDR photo). Also report PSNR in log-luminance (μ-law) against merged EXR.
- Log metrics to `out/metrics.csv` (commit hash, milestone, scene, metric, value).

## Code map (fill in during M0)
- Training loop / loss: TODO
- Gaussian color / SH storage: TODO
- CUDA rasterizer (forward/backward): TODO
- Vulkan viewer + shaders: TODO
- PLY/SPZ export: TODO
- Python plugin API + example plugin: TODO
- MCP server (for driving the app in tests): TODO
