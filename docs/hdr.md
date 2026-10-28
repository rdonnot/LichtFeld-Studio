# HDR Gaussian splats — the file convention and how to read it

LichtFeld's `--hdr` mode trains Gaussians whose colour is **linear radiance in absolute scene units**, unbounded
above 1 (PLAN.md M2). Inside the app the SH coefficients hold the *logarithm* of that radiance; a file never
has to: the exporters write the standard PLY convention and say in the header what the numbers mean.

## The PLY convention

Every Gaussian-splat tool reads colour as

    colour(w) = 0.5 + SH_C0 · f_dc  +  Σ_i Y_i(w) · f_rest_i          SH_C0 = 0.28209479…

with `f_rest_0 … f_rest_{3K-1}` channel-major (K reds, K greens, K blues; K = 3 / 8 / 15 for degree 1 / 2 / 3)
and the real SH basis in the usual Gaussian-splatting order. An HDR PLY written by LichtFeld keeps exactly that
and adds header comments:

```
comment lichtfeld_hdr linear
comment hdr_scale 2421.4
comment hdr_color_max 2421400.25
comment hdr_display_exposure 0.000412984
```

| comment | meaning |
|---|---|
| `lichtfeld_hdr linear` | `colour(w)` **is linear radiance**: 0 … thousands. No custom decoding; the only difference from an LDR file is that values are not confined to [0, 1]. |
| `hdr_scale <s>` | The scene's radiance scale: the 99th-percentile luminance of the training images. Mid-grey is around `s / 10`; `1/s` is the scene's natural display exposure. |
| `hdr_color_max <cap>` | The absolute radiance cap the model was trained with (`1000 · s` by default). No value in the file exceeds it. |
| `hdr_display_exposure <1/s>` | A hint for viewers: multiply colour by this to land the scene in [0, 1] before tone mapping. |
| `lichtfeld_hdr log` | LichtFeld's **internal** representation written verbatim (`--hdr-export log`): radiance = `exp(min(colour(w), ln cap))`. Lossless for LichtFeld; meaningless to any other reader. |

A PLY without `lichtfeld_hdr` is an ordinary LDR splat and is read exactly as before.

### What a viewer should do with a `linear` file

1. Evaluate the SH as usual; clamp the result at 0 (a band-limited radiance fit can ring slightly negative
   in some directions, as any linear-radiance splat can).
2. Multiply by an exposure: `2^EV · hdr_display_exposure` (EV = 0 shows the scene at its natural exposure).
3. Tone-map (ACES, filmic, neutral, …) and apply the sRGB transfer, then blend or display. Blending in linear
   radiance before the tone map is the physically right order; tone-mapping per splat is the cheap approximation.

Houdini's Gaussian-splat import, N'Brain's PLY decoder and NVIDIA's `vk_gaussian_splatting` all read step 1
with no change; whether they clamp before or after blending decides how bright pixels look, not whether the
file loads.

### Why the degree-3 values are a projection

Inside LichtFeld the per-direction radiance is `exp(b(w))` of a degree-3 log expansion `b`. The exponential of
an SH expansion is not an SH expansion, so the writer **projects** it: it samples `exp(min(b(w), ln cap))` on a
Fibonacci sphere (128 directions for degree 3, 64 for degree 2, 32 for degree 1 — at least twice the coefficient
count) and takes the least-squares degree-N fit (`src/core/hdr_sh.cpp`). Degree 0 is exact (`f_dc_lin =
(exp(0.5 + C0·f) − 0.5) / C0`). For degree 3 the loss is the energy above degree 3 of `exp(b)`; it grows with the
lobe amplitude. Measured (`tools/eval/test_hdr_ply.py`, PLAN.md M5 notes): random log lobes of σ = 0.1 give a
median relative radiance error well under 1 % over the sphere, σ = 0.3 a few per cent; on the real `smoke_hdr`
model the round trip costs a fraction of a dB of μ-law PSNR (numbers in PLAN.md).

Reading a `linear` file back applies the inverse projection (`ln(max(colour(w), 1e-6))` on the same sphere), so
export → import is not bit-exact above degree 0. For a lossless project round trip use the `.licht` project
(it stores the raw tensors) or `--hdr-export log`.

## Baked LDR — opens in any viewer

`--hdr-export baked --hdr-exposure <EV> --hdr-tonemap <op>` (CLI, `convert`), `hdr="baked"` (Python),
or the **HDR colour** row of the export panel writes a display-referred file: per direction

    srgb( tone( 2^EV / s · exp(min(b(w), ln cap)) ) )   clamped to [0, 1]

projected onto the same SH basis, with **no** HDR comments. The tone operators are the viewer's own
(`display_color.inc` compiled as a CUDA twin: `none`/`linear`, `filmic`, `hejl`, `aces` (default), `aces2`,
`neutral`), so the file shows what the viewer showed at that exposure. Every other export format (SPZ, GLB,
SOG, SSOG, HTML viewer, USD, NuRec USDZ, RAD) quantises colour to [0, 1]: an HDR model is **refused** there
unless `baked` is asked for, in which case the same bake runs first (`lfs::io::hdr_source_for_ldr_writer`).

## In the app

- `SplatData::hdr()` carries `{log_sh, scale, color_max}`; `is_hdr()` is what every consumer tests. The
  rasterizer takes `color_activation_for(model)`; the PLY reader sets the state from the header; a merge of
  HDR models keeps it; `extract_by_mask` / crop keep it.
- Loading a `linear` PLY as a training start (`--init`, or `LichtFeld-Studio eval <ply> -d <dataset> --hdr`) does
  not re-seed sh0 (the model already holds log radiance), and when `--hdr` is given with no `hdr_scale` /
  `hdr_color_max` the file's values are used. Without `--hdr` the trainer refuses an HDR model and says why.
- Headless video (`--render-camera-path`) of an HDR scene is tone-mapped with `--render-exposure` /
  `--render-tonemap`; `--render-exr-dir <dir>` also writes every frame as a linear EXR. `--eval` already
  writes `eval_step_N/<image>.exr` (linear) beside the μ-law PNG.
- `core::hdr::bake_hdr_to_ldr(splat, ev, tone)` returns a baked `SplatData` clone for any writer;
  `core::hdr::to_linear_radiance(splat)` the linear-radiance clone the PLY writer uses.
- Python: `lf.io.save_ply(..., hdr="auto|linear|baked|log", exposure_ev=0.0, tonemap="aces")` (same keywords
  on `save_spz`, `save_sog`, `save_ssog`, `save_usd`, `save_nurec_usdz`); `SplatData.is_hdr`, `.hdr_scale`,
  `.hdr_color_max`, `.bake_hdr_to_ldr(exposure_ev, tonemap)`; `lf.export_scene(..., hdr=, hdr_exposure_ev=,
  hdr_tonemap=)`.

## Tools

`tools/eval/hdr_ply.py` reads any of these files with numpy: `info`, `stamp` (add the comments to an M2-era
PLY that holds log values as `f_dc`), `compare` (radiance of two files on random directions, each decoded by its
own header), `range` (a baked file's expansion must stay in [0, 1]). `tools/eval/hdr_roundtrip.py` runs the full
measurement on a trained model (export → load → re-render → μ-law PSNR against the original eval EXRs).
