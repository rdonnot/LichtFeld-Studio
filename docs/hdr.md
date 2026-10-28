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
export → import is not bit-exact above degree 0. For a lossless round trip use the `.licht` project (it stores
the raw log tensors AND the HDR state — a training project's checkpoint chapter carries it in its parameter
JSON (`hdr`, `hdr_scale`, `hdr_color_max`; `apply_checkpoint_hdr_state`), a scene splat chapter in a 12-byte
trailer of the raw splat payload (flag `0x1` in the header, `src/io/project/splat_chapter.cpp`); older projects
read unchanged) or `--hdr-export log`.

Reading back floors each direction at **1 % of the splat's own DC radiance** (never below 1e-6) before the log:
a foreign `linear` file whose expansion goes negative over part of the sphere would otherwise drag the log fit
down to ln 1e-6 = −13.8 and ruin the bright side (measured on a degree-1 fixture with lobes of 1.3 × DC: the
positive side's median error fell from 97 % to 20 %; a log basis still cannot hold a clipped hemisphere).
Files LichtFeld writes are positive everywhere but for a vanishing fraction of samples, so the floor never
acts on them.

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
  rasterizer takes `color_activation_for(model)`; the PLY reader sets the state from the header; the `.licht`
  project stores it; a merge of HDR models keeps it (a mixed HDR + LDR merge is refused: the two readings of the
  SH cannot share one file); `extract_by_mask` / crop keep it; the simplifier and the Bhattacharyya / octree LOD
  builders keep it (their merged coefficients average LOG radiance, i.e. a geometric mean of radiance).
- The out-of-core PLY → RAD converter (`--tiles`, `--lod-builder octree`, very large files) parses colour only
  and never sees the header, so `convert` probes `is_hdr_ply` first: an HDR PLY is refused there unless
  `--hdr-export baked` (then it is baked in memory and written through `save_rad`); `--tiles` of an HDR PLY is
  refused outright.
- The sRGB transfer of the bake is `lfsSrgbEncode` in `display_color.inc`, the one definition the viewer's HDR
  path (M4) must use as well.
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

## Colour management (OpenColorIO)

"The HDR Gaussian has to be compatible with OCIO." It is, in three places: the files say which colorspace
their radiance is in, the viewer can show an HDR model through any OCIO display/view, and every bake to 8 bits
can run the same transform. OpenColorIO 2 is linked into `lfs_core` (BSD-3; `src/core/ocio.cpp` is the only
file that includes it) and needs no config file: the default is `$OCIO`, else the built-in ACES studio config
(`ocio://studio-config-latest`); `--ocio <file>` points at a studio's own (Houdini 22 ships three under
`/opt/hfs22.0/packages/ocio/`).

### The working space is `lin_srgb`

Inside LichtFeld an HDR model's radiance is **always** scene-linear with **Rec.709 / sRGB primaries and a D65
white** — OCIO's `Linear Rec.709 (sRGB)`, alias `lin_srgb` in the ACES configs, `scene_linear` role in most
others. That is what the synthetic data is, what a Debevec merge of sRGB JPEGs produces, and what the training
loss compares, so it is an invariant, not a setting: the `.licht` project stores nothing about it. Files
declare it for other readers:

| file | declaration |
|---|---|
| HDR PLY (`linear` and `log`) | `comment lichtfeld_colorspace lin_srgb` next to the `lichtfeld_hdr` comments (LDR PLYs are untouched: their colour is display-referred) |
| every EXR LichtFeld writes (`--eval`, `--render-exr-dir`, the brackets plugin's merge) | the standard `chromaticities` attribute (0.64/0.33, 0.30/0.60, 0.15/0.06, D65 0.3127/0.3290) **and** a string attribute `lichtfeld_colorspace = lin_srgb` |

**Reading a foreign declaration.** A PLY or EXR whose `lichtfeld_colorspace` names another colorspace of the
current config is converted on load *when that colorspace is linear* (OCIO's processor from it to `lin_srgb`
is probed on the basis vectors at two scales; the 3×3 is applied to every SH coefficient — `f_dc` through the
0.5 offset of the PLY convention, every `f_rest` directly — before the log projection, or per pixel for an
EXR). A non-linear declaration (ACEScct, sRGB-encoded, …), an unknown name, or a tag on a `lichtfeld_hdr log`
file is **refused** with the reason, never read as if it were `lin_srgb`. No tag = `lin_srgb`.

**What to assign in another application.** A LichtFeld HDR PLY or EXR is `Linear Rec.709 (sRGB)` in the
ACES 1.3 / 2.0 configs (`lin_srgb`), `scene_linear` / `linear` in a legacy config:

- *Houdini* (OCIO on, Karma / Solaris): set the file's colorspace to **Linear Rec.709 (sRGB)** when importing
  the EXR as a texture or the PLY's colour attribute; Houdini's own default config is the same ACES CG config,
  so an EXR with these chromaticities is picked up as linear Rec.709 automatically.
- *Nuke*: Read node **colorspace = Linear Rec.709 (sRGB)** (ACES configs) or `linear` (nuke-default); the
  `chromaticities` attribute does not drive Nuke's input transform, the Read node does.
- *DaVinci Resolve* (ACES or RCM): EXR input colour space **Linear**, gamut **Rec.709** (in ACES: "Linear
  Rec.709 (sRGB)" / "ACES → sRGB linear" IDT); in a Rec.709 timeline the ODT does the tone mapping.
- Anything that reads `hdr_display_exposure` (docs above): multiply by it to put the scene's P99 white at 1.0
  before the view transform.

### Display: tone mapping "OCIO" in the viewer

With an HDR model loaded, **Tone mapping → OCIO** in the rendering panel (Color & Appearance) replaces the
built-in curve by an OCIO **display / view / look** (three selects, filled from the current config; "Default"
= the config's own default display / view). The chain is

    radiance / s · 2^EV  (linear, the EV slider)  →  [look]  →  display/view transform  →  RGBA8

i.e. the exposure is applied in scene-linear light **before** the view transform, as OCIO expects. The
transform is baked once per selection into a **65³ float LUT over a per-channel log2 shaper** (−14 … +8
stops of normalised radiance; node 0 is exactly black) by `core::ocio::lut_for`, uploaded to the compose
pass as a storage buffer and sampled with trilinear interpolation (`display_color.inc` `lfsOcioLut`). Measured
against OCIO's own CPU processor on 20 000 log-uniform radiances (−12 … +10 stops, random chroma): 68 dB on
"ACES 2.0 - SDR 100 nits (Rec.709)", 71 dB "Un-tone-mapped", 60 dB "Raw" (unbounded, clipped at 1), 70 dB
on the PQ 1000-nit view — the test floor is 45 dB and ≤ 2/255 max error
(`tests/test_hdr_display_chain.cpp`, `HdrDisplayChainTest.Ocio*`). Settings: `color_tonemapping = "ocio"`,
`ocio_display`, `ocio_view`, `ocio_look` (Python `lf.get_render_settings()`, persisted with the session).

Notes: the viewer's output is RGBA8, so an HDR view (PQ, HLG) is shown clipped — the maths is right, the
swapchain is not HDR yet; "Raw" and "Un-tone-mapped" are not display encodings (linear, resp. sRGB-encoded
linear without a tone curve) and clip at 1.0 as they would anywhere; the standard-portal profile displays per
splat in the projection shader where no LUT exists and falls back to ACES with one warning; LDR models never
see tone 7.

### Bake: `--hdr-tonemap ocio` and `--render-tonemap ocio`

The same LUT, sampled by the same text in the CUDA kernel (`core/hdr_sh.cu`), drives every 8-bit bake:

    --hdr-export baked --hdr-tonemap ocio --ocio <config> --ocio-display "sRGB - Display" \
        --ocio-view "ACES 2.0 - SDR 100 nits (Rec.709)" [--ocio-look <look>] [--hdr-exposure <EV>]
    --render-camera-path … --render-tonemap ocio [--ocio* …]        # the video; --render-exr-dir stays linear
    lf.io.save_ply(..., hdr="baked", tonemap="ocio"); SplatData.bake_hdr_to_ldr(ev, "ocio")

`core::hdr::bake_hdr_to_ldr(splat, ev, ToneOperator::Ocio, selection)` and `display_image(..., Ocio,
selection)` take an explicit `ocio::Selection{config, display, view, look}`; **empty fields mean the
process-wide default** `core::ocio::current()`, which the `--ocio*` flags, the viewer (whenever it shows tone
"OCIO": what you see is what an export from that session bakes) and Python `lf.io.set_ocio(config=, display=,
view=, look=)` write. The CLI validates names before anything runs and lists the valid ones on a mistake.
`tests/test_hdr_display_chain.cpp` pins the CUDA bake to the C++ twin pixel for pixel on a sweep, and
`tools/eval/ocio_check.py` renders a real model headless and compares the video frame with the same EXR pushed
through PyOpenColorIO (expect ≥ 40 dB; the codec is the floor).

Python: `lf.io.ocio_available()`, `ocio_config_name()`, `ocio_working_space()`, `ocio_displays()`,
`ocio_views(display)`, `ocio_looks()`, `get_ocio()`, `set_ocio(...)`.

Trap: the app's OCIO (vcpkg 2.5.1) and PyOpenColorIO 2.6 ship *different versions* of the built-in configs
(`studio-config-v4` vs `v5`), so a cross-check between the two must use a config **file**, not `ocio://…`.

## Tools

`tools/eval/hdr_ply.py` reads any of these files with numpy: `info`, `stamp` (add the comments to an M2-era
PLY that holds log values as `f_dc`), `compare` (radiance of two files on random directions, each decoded by its
own header), `range` (a baked file's expansion must stay in [0, 1]). `tools/eval/hdr_roundtrip.py` runs the full
measurement on a trained model (export → load → re-render → μ-law PSNR against the original eval EXRs).
