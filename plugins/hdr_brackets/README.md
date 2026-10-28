# HDR Brackets

LichtFeld Studio plugin (HDR fork, milestone M1). It takes a folder of
exposure-bracketed photos and produces a COLMAP-ready dataset whose training
images are linear HDR:

1. groups the images into brackets (EXIF time gaps + exposure time, manual overrides),
2. aligns each bracket and recovers the camera response (CRF),
3. merges every bracket to a linear float EXR (with optional deghosting),
4. writes a tonemapped 8-bit proxy of each EXR for SfM,
5. runs SfM on the proxies (pycolmap) unless disabled,
6. writes the bracket training set `brackets/` — the raw JPEGs, each with its position's pose — for M3's
   `--hdr-brackets` training (below).

## Output layout

```
<out>/
  images_hdr/*.exr    linear radiance, one per bracket (or singleton)
  images/*.png        tonemapped LDR proxies (what COLMAP sees)
  sparse/             COLMAP model (only when "Run SfM" is on)
  exposures.json      per position and source image: files, exposure, alignment (schema below)
  crf.json            the recovered (or supplied) camera response
  brackets.json       the grouping, every image (also the skipped ones)
  brackets/           the bracket training set (needs sparse/; "Write bracket training set", CLI --brackets)
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

### `brackets/` — the bracket training set (M3)

A second, self-contained dataset next to the merged one. LichtFeld trains on it directly with the raw JPEGs as
training images (`docs/hdr-m3-design.md` §1; the trainer side — `--hdr-brackets`, the position hold-out — is built
on another branch):

```
<out>/brackets/
  images/pos_NNN_fK.jpg    symlink (copy when symlinks are refused) to position NNN's raw JPEG, frame K = 0..n−1
                           by exposure time ascending; the APP1 EXIF is the raw file's, byte for byte — the trainer
                           seeds each frame's exposure from it
  sparse/0/cameras.txt     COLMAP TEXT model derived from <out>/sparse/0: the SfM camera; one image record per
           images.txt      FRAME, IMAGE_ID = stride·p + K for the proxy's id p (stride = max(10, frames in the
           points3D.txt    largest bracket), written to exposures.json `brackets.id_stride`), the proxy's pose and
                           its POINTS2D line verbatim, name pos_NNN_fK.jpg; every track image_id p → stride·p
  crf.json                 <out>/crf.json + iso_ref, fnumber_ref (the shoot's medians), exif_ev_ref =
                           log2(iso_ref / fnumber_ref²) (2.644 for ISO 100 f/4) and a `units` string
  exposures.json           <out>/exposures.json for the written positions, each source with its `frame`, plus a
                           `brackets` block: id_stride, link (symlink|copy), exif_ev_ref, omitted positions
```

```bash
./build/LichtFeld-Studio --headless --train -d <out>/brackets -o out/m3 --iter 7000 --test-every 8 --test-by-position \
    --eval --export ply --hdr --hdr-brackets --hdr-crf <out>/brackets/crf.json
```

**Units — the merge's convention, and what the trainer must compute.** The merge is Debevec's
`ln E = Σ w·(g[z] − ln t) / Σ w`, so a merged EXR pixel is `E = exp(g[z]) / t`: exposure per second at the ISO and
f-number the shoot used, with g(128) = 0. The frame shot for `t_i` seconds reads `z = g⁻¹(ln(E · t_i))`, and with
`exif_ev = log2(t · ISO / N²)` that is `t_i = 2^(exif_ev_i − exif_ev_ref)` exactly when `exif_ev_ref = log2(ISO / N²)`
of the shoot — the value in `brackets/crf.json`. So the trainer's input to the LUT is
`x_i = radiance × 2^(exif_ev_i − exif_ev_ref)`; the LUT's x-axis unit is "1.0 = code 128 after 1 s at iso_ref,
fnumber_ref". Forget the anchor and every frame is 2^2.64 = 6.25× off (design §8, risk 1). The check:

```bash
cd tools && uv run python eval/check_brackets.py ../data/smoke_hdr      # structure + the LUT round-trip per EV step
```

It parses the text model itself (poses equal per position, image count = Σ frames, every track on an existing frame 0,
EXIF through the links, the anchor) and prints, per EV step, the median |Δ| between each JPEG and
`hdr_metrics.apply_crf(EXR × 2^(exif_ev − exif_ev_ref), crf)` on the unsaturated pixels. On `data/smoke_hdr`
(2026-10-28): 1–2/255 at −4…+2 EV; the +4 EV frames are ~93 % clipped and the rest sits in the LUT's toe (reported,
not judged). Positions are written when their proxy is registered in the SfM model and their bracket is not flagged
for ghosting; the others are omitted and named in the report's warnings. A frame MTB shifted by (dx, dy) ≠ (0, 0)
gets its own camera (the shared one with the principal point moved by (−dx, −dy)) — on the tripod set every shift
is 0.

Regenerate the set alone, from a finished dataset: `cd plugins/hdr_brackets && uv run python -m hdr_brackets.brackets_set <out>`.
Regenerate the whole dataset without the SfM minutes: `… hdr_brackets.cli <src> <out> --reuse-sfm` keeps `<out>/sparse/0`
(valid as long as the positions come out the same — the report warns when the model names a proxy the run did not
write).

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
brackets, singletons, files flagged for ghosting, SfM registered/total, what went into
`brackets/` (positions, frames, link mode, `exif_ev_ref`) and warnings. "Preview" shows any merged position as the proxy PNG or as the EXR
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
cd plugins/hdr_brackets && uv run python -m hdr_brackets.cli <src> <out> [--no-sfm] [--reuse-sfm] [--no-brackets]
```

## Linear input: developed raws (`hdr_brackets.linear`)

For brackets shot RAW and developed to scene-linear EXR (no camera response to recover), grouped by N'Brain's
camera tether record instead of EXIF:

```bash
cd plugins/hdr_brackets && uv run python -m hdr_brackets.linear <capture_dir> <out> [--exr-dir D] [--max-side 3200] [--no-sfm]
```

`<capture_dir>/tether_set.json` names each position's frames, exposure times and ND stops; the developed EXRs are
`<capture_dir>/exr/<frame>.exr`. Per pixel `E = Σ w·x/t / Σ w`, `w = t · hat(x / white)`, ND frames by
`t · 2^-nd_stops`. The merged EXRs keep the input's colourspace tag (ACEScg by default) — the trainer converts on
load. Same `<out>` layout as above, `exposures.json` with `"linear": true, "crf": null`. Prints one JSON line per
step and the report last. Tests: `tests/test_linear.py`. N'Brain's side: `docs/hdr-scan.md` in ShowControl.

## Limits

- Cancel only disables the button: the core has no cancellation yet, a started run finishes.
- One run at a time; the panel's settings are global to the app session.
- The preview is downscaled to 512 px wide and re-tonemaps in Python on the UI thread.
- Needs the CRF to be recoverable: a bracket of 2 exposures or less, or a scene with no
  shared overlap between exposures, falls back to what the core reports in the warnings.
