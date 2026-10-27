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
  exposures.json      per image/bracket: files, exposure, grouping
  crf.json            the recovered (or supplied) camera response
```

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

Scripts and MCP can run the same job with the operator `hdr_brackets.merge`,
which uses the panel's current settings (`lf.ops.invoke("hdr_brackets.merge")`).

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
