# HDR Brackets

LichtFeld Studio plugin (HDR fork, milestone M1). It takes a folder of
exposure-bracketed photos and produces a COLMAP-ready dataset whose training
images are linear HDR:

1. groups the images into brackets (EXIF time gaps + exposure time, manual overrides),
2. aligns each bracket and recovers the camera response (CRF),
3. merges every bracket to a linear float EXR (with optional deghosting),
4. writes a tonemapped 8-bit proxy of each EXR for SfM,
5. runs SfM on the proxies (pycolmap) unless disabled.

## Output layout

```
<out>/
  images_hdr/*.exr    linear radiance, one per bracket (or singleton)
  images/*.png        tonemapped LDR proxies (what COLMAP sees)
  sparse/             COLMAP model (only when "Run SfM" is on)
  exposures.json      per position and source image: files, exposure, alignment (schema below)
  crf.json            the recovered (or supplied) camera response
  brackets.json       the grouping, every image (also the skipped ones)
```

### `exposures.json` (version 2)

```
{"version": 2, "crf": "crf.json", "proxy_operator": "aces", "proxy_ev": 0.0, "units": …, "source_dir": …,
 "ghost_threshold": 0.08, "deghost": true, "proxy_shared_scale": true, "saturated_level": 250,
 "stats": {"shift_max", "align_failed", "align_clipped", "frames_clipped_over_90", "saturated_total",
           "crf_brackets", "ghost_score_max", …},
 "positions": [{"name": "pos_000", "hdr": "images_hdr/pos_000.exr", "proxy": "images/pos_000.png",
                "proxy_scale", "bracket_id", "ghost_score", "flagged", "saturated", "stats": {min, p1, p50, …},
                "sources": [{"file", "exposure_time", "exif_ev", "ev_step", "iso", "fnumber", "timestamp",
                             "shift": [dx, dy], "align"}]}]}
```

- `exif_ev` — the absolute EXIF exposure value, LichtFeld's `log2(t / N² · ISO)` (what PPISP seeds from).
- `ev_step` — the frame's step inside its bracket, `log2(t / t_median)` (0 for the middle frame of an odd bracket).
  The same words as the synthetic truth `exposures_gt.json` (`exif_ev` absolute, `ev` the step). Version 1 wrote
  the absolute value as `ev`.
- `shift` — the pixels MTB moved the raw frame onto the bracket's reference; `align` how it was obtained:
  `reference`, `mtb` (searched), `clipped` (grey median ≥ 250 or ≤ 5: not searchable, took its nearest searched
  neighbour's shift), `failed` (the search reached its ±(2^bits − 1) px limit: kept at (0, 0), and warned).
- `saturated` — pixels with a channel ≥ 250 in the bracket's SHORTEST exposure: there the merged radiance is only a
  lower bound.

The report also warns when ISO or the f-number change inside a bracket or across the shoot (the merge weighs frames
by exposure time only), and names frames more than 90 % clipped.

## Checking against the synthetic truth

```bash
cd tools && uv run python eval/check_smoke_hdr.py ../data/smoke_hdr ../data/brackets_raw ../data/brackets_synth/gt
```

CRF vs the camera's truth, merged EXR vs the truth EXR (one global scale per position, over the 0 EV frame's
unclipped pixels), the shifts; PASS when CRF < 0.2 nats, median < 5 %, p99 < 15 % and no shift beyond 2 px.

"Load dataset" calls `lf.scene.load_file(<out>, is_dataset=True, output_path=<out>/train)`.

## Install

```bash
ln -sfn /home/romain/Documents/Studio/apps/LichtFeld/LichtFeld-HDR/plugins/hdr_brackets ~/.lichtfeld/plugins/hdr_brackets
```

The dependencies in `pyproject.toml` are installed into the plugin's venv by LichtFeld.

## Use

Open the "HDR Brackets" tab, pick the source folder (output defaults to
`<source>_hdr`), set options, press Run. When it finishes the report lists
brackets, singletons, files flagged for ghosting, SfM registered/total and
warnings. "Preview" shows any merged position as the proxy PNG or as the EXR
re-tonemapped with an EV slider (downscaled to 512 px wide).

Manual overrides for odd brackets: `IMG_0007.jpg=3, IMG_0008.jpg=3` forces those
files into bracket 3.

Scripts and MCP can run the same job with the operator `lfs_plugins.hdr_brackets.operators.MergeOperator`,
which uses the panel's current settings (`lf.ops.invoke("lfs_plugins.hdr_brackets.operators.MergeOperator")`).

### Loading at startup

LichtFeld 0.5.4.2 ignores the manifest's `auto_start`; it loads at startup only the plugins whose per-user setting
`load_on_startup` is true (`PluginManager.load_all`, `src/python/lfs_plugins/manager.py`). Tick "Load on startup" in
the Plugin Marketplace panel, or write `~/.lichtfeld/plugins/hdr_brackets/settings.json`:

```json
{"load_on_startup": true}
```

Otherwise load it once per session from the Python console: `lf.plugins.load("hdr_brackets")`.

## CLI (no app)

```bash
cd plugins/hdr_brackets && uv run python -m hdr_brackets.cli <src> <out>
```

## Limits

- Cancel only disables the button: the core has no cancellation yet, a started run finishes.
- One run at a time; the panel's settings are global to the app session.
- The preview is downscaled to 512 px wide and re-tonemaps in Python on the UI thread.
- Needs the CRF to be recoverable: a bracket of 2 exposures or less, or a scene with no
  shared overlap between exposures, falls back to what the core reports in the warnings.
