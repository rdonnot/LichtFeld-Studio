# HDR Gaussian splats with LichtFeld — capture, train, view, export

This is the user guide to the HDR fork (`--hdr`): how to shoot a bracket set, the two ways to train from it, what
the viewer and the exporters do with a radiance model, and the PLY convention other tools can read. The engineering
log with every measurement is `PLAN.md`; the design of bracket training is `docs/hdr-m3-design.md`.

In one paragraph: LichtFeld's `--hdr` mode trains Gaussians whose colour is **linear radiance in absolute scene
units**, unbounded above 1. You feed it either linear EXRs merged from exposure brackets by the `hdr_brackets`
plugin (recommended), or the raw bracket JPEGs with their EXIF (`--hdr-brackets`). The viewer tone-maps live with
an exposure slider; the PLY export writes linear radiance any splat tool can read; the quantising formats bake at
an exposure you choose.

## 1. Shooting a bracket set

The plugin needs, per camera position, a few frames of the same view at different exposures. Everything below is
about making those frames mergeable; a set that breaks a rule still runs, but the report will name what it had to
work around.

- **Tripod, or a very fast bracket.** The frames of one bracket are aligned (median-threshold bitmaps, ±31 px,
  integer shifts), not warped: on a tripod every shift is 0 and the merge is exact; hand-held works when the
  camera's auto-bracketing fires the whole sequence in well under a second (burst mode, electronic shutter) and
  you hold still — expect a pixel or two of shift and a slightly softer merge. Hand-held brackets with more
  than ~5 px of drift, or anything moving in the frame (people, leaves, water), get a ghost score; a flagged
  bracket is merged with deghosting weights and listed in the report. Pick tripod for interiors and anything
  static; hand-held only outdoors in bright light where the shutter is short.
- **Spacing: 2 EV between frames, 3 or 5 frames** (−2/0/+2 or −4/−2/0/+2/+4). Two stops is where the Debevec
  calibration has enough overlap between neighbouring frames to recover the response curve, and 3–5 frames
  cover 12–16 stops of scene. Closer than 1 EV wastes frames; farther than 3 EV leaves the curve unconstrained in
  the gap. The synthetic reference set (`data/brackets_raw`, 40 positions × 5 frames at −4…+4 EV, a room
  of ≈ 14 stops with a lamp and a dark alcove) is what the numbers in PLAN.md were measured on.
- **Vary only the shutter.** ISO and aperture fixed for the whole shoot (manual mode, auto-ISO off, white balance
  fixed, no auto-exposure compensation). The merge weighs frames by exposure *time*; a changed ISO or f-number
  inside a bracket is warned about and treated as if it were time, which it is not exactly (noise and depth of
  field change). The same ISO and aperture across *all* positions is what makes the radiance units consistent
  from one position to the next — the trainer's unit anchor is `log2(ISO / N²)` of the shoot.
- **Exposure time, not stops of EV, is what the EXIF must carry.** Required tags: `ExposureTime` (0x829A) — or
  `ShutterSpeedValue` as the APEX fallback — and `DateTimeOriginal` (with `SubSecTimeOriginal` when the camera
  writes it). Read as well: `FNumber` / `ApertureValue`, `ISOSpeedRatings` (or the SOS/REI/ISOSpeed alternates).
  A frame without an exposure time is set aside and reported, never guessed. Check with
  `exiftool -ExposureTime -FNumber -ISO -DateTimeOriginal IMG_0001.jpg` before you shoot the whole set.
- **JPEG or RAW?** The plugin reads what Pillow reads: JPEG, PNG, TIFF. 8-bit camera JPEGs are fine — the
  response curve is recovered from them and the merge happens in linear float — provided the camera's picture
  style is neutral and constant (no HDR mode, no "auto lighting optimiser", no dynamic-range expansion, which
  change the curve from frame to frame). RAW gives a cleaner curve and two more stops per frame; develop every
  frame with the *same* linear-ish settings to 16-bit TIFF, keeping the EXIF, and feed those. Do not feed
  camera-made HDR JPEGs: their curve is not fixed.
- **Timing between brackets.** The grouping is by capture time: a gap longer than `Max gap between shots` (5 s
  by default) or a repeated exposure time starts a new bracket. Shoot each bracket as one burst, then take at
  least a few seconds to move to the next position. Where a camera restarts its sequence faster than that the
  repeated shutter time splits them anyway. Odd cases are fixed by hand in the panel (`IMG_0007.jpg=3`).
- **Coverage like any splat capture.** The merged positions are the views the splat is trained from, nothing
  more: overlap of 60–80 % between neighbouring positions, orbit the subject, several heights, no position
  closer to another than the baseline COLMAP can triangulate. Every position costs 3–5 frames, so plan for
  ~⅓ the number of positions you would shoot in LDR for the same time budget, and prefer more positions over more
  frames per position (5 frames help only on very high-contrast scenes).
- **The lamp / sun rule.** A light source that is still clipped in the *shortest* exposure is recorded as a
  *lower bound* of its radiance, in the merge and in the training. The report counts those pixels
  (`saturated`); the model will put the lamp at the cap (`hdr_color_max`, 1000 × the scene's white by
  default) and the false-colour view shows it in magenta. If the lamp matters, add a −6 EV frame (or an ND
  filter) so that it is unclipped somewhere; if the sun is in frame it will clip whatever you do — keep it
  out of frame or accept the bound. Conversely, the darkest frame must not be the only one that sees the
  shadows: the +EV frames carry them, and a +4 EV frame that is 90 % clipped is still useful for the alcove.

## 2. Two routes from brackets to a model

```
   brackets (JPEG + EXIF)
        │
        ▼ hdr_brackets plugin: group → align → CRF → merge → proxies → SfM
        ├──────────────► images_hdr/*.exr + images/*.png + sparse/     ── route A: --hdr (merge-then-train)
        └──────────────► brackets/ (raw frames, one pose per position) ── route B: --hdr --hdr-brackets
```

**Route A — merge, then train (recommended).** The plugin writes one linear EXR per position and an 8-bit
tone-mapped proxy of each for SfM; training reads the EXRs (`--images images_hdr`) and fits radiance directly
through a μ-law loss. This is the path with the best numbers: on the synthetic room, 7000 iterations give
**36.5 dB μ-law PSNR** against the merged EXRs and **32.2 dB** LDR PSNR at the five source exposures on
held-out views, +1.4 / +1.9 dB over a model trained on the proxies (PLAN.md M2). The merge averages a
position's frames, so the training sees one clean image per view.

**Route B — train from the raw brackets (`--hdr-brackets`).** No merge: every frame is a training view with
its position's pose, the render goes through the frame's exposure (from its EXIF, optionally refined) and the
calibrated response curve, and the loss is taken against the JPEG on the pixels it recorded unclipped. Use it
when the merge is the problem: hand-held brackets with ghosting, moving content, a bracket whose frames do not
align — the training never has to pick a reference frame, and a clipped or shifted frame costs only its own
pixels. Know what it costs: on the tripod synthetic set, where the merge has no ghosting to lose, route B is
**0.9 dB behind route A** at equal iterations (31.3 vs 32.2 dB LDR PSNR, PLAN.md M3 improvement round) — each
frame teaches only its unclipped band of a noisy JPEG where the merge had averaged five — and it needs
`brackets/` from the plugin (the proxies' poses are still what SfM gets).

## 3. The plugin: panel and CLI

Install once: `ln -sfn <repo>/plugins/hdr_brackets ~/.lichtfeld/plugins/hdr_brackets`, tick *Load on startup* in
the Plugin Marketplace (or `lf.plugins.load("hdr_brackets")` in the console; the README has the details).

**Panel — "HDR Brackets" tab.** Pick the source folder (the output defaults to `<source>_hdr`), then:
*Max gap between shots (s)* (5), *Deghost moving content* (on), *Run SfM (COLMAP poses)* (on), *Write bracket
training set (brackets/)* (on; route B's input), *Proxy tonemap* (`aces`, `reinhard`, `linear`) and *Proxy EV*
(the proxies are what COLMAP matches: leave ACES at 0 EV unless they come out dark), *Overrides* for the odd
bracket. **Run** reports brackets / singletons / flagged, SfM registered / total, the `brackets/` summary and
every warning; **Preview** shows any merged position as the proxy or as the EXR re-exposed with an EV slider;
**Load dataset** hands the folder to training (`lf.scene.load_file(<out>, is_dataset=True)`).

**CLI (no app):**

```bash
cd plugins/hdr_brackets && uv run python -m hdr_brackets.cli <src> <out> \
    [--no-sfm | --reuse-sfm] [--no-brackets] [--no-deghost] [--crf saved_crf.json] \
    [--max-gap 5] [--ghost-threshold 0.08] [--proxy-operator aces] [--proxy-ev 0]
```

Output: `images_hdr/*.exr`, `images/*.png`, `sparse/0`, `exposures.json` (every frame's exposure, EV, alignment
and the per-position saturation count), `crf.json`, `brackets.json`, and `brackets/` (README: "Output layout").
Radiance units: 1.0 = a pixel reading 128 after 1 s at the shoot's ISO and f-number; the trainer rescales by its
own P99 (`hdr_scale`, written to every export), so the absolute unit never has to be right.

## 4. Training

Route A:

```bash
./build/LichtFeld-Studio --headless --train -d <out> --images images_hdr -o out/run --iter 7000 \
    --test-every 8 --eval --export ply --hdr
```

Route B:

```bash
./build/LichtFeld-Studio --headless --train -d <out>/brackets -o out/run --iter 7000 \
    --test-every 8 --test-by-position --eval --export ply --hdr --hdr-brackets --hdr-crf <out>/brackets/crf.json
```

| flag | what it does |
|---|---|
| `--hdr` | HDR mode: float / 16-bit decode, log-radiance SH, no [0, 1] clamp, μ-law loss; switches the MRNF exposure correction and `dc_reg` off. Everything else is the LDR path, bit for bit, when it is absent. |
| `--images images_hdr` | train on the EXRs (the folder of proxies stays `images/` for SfM and browsing). |
| `--hdr-brackets` | route B; implies `--hdr`; the JPEGs are read on the LDR path and compared through exposure + CRF. |
| `--hdr-crf <crf.json>` | the response curve for route B — the plugin's `brackets/crf.json`, which also carries the unit anchor `exif_ev_ref`. Required with `--hdr-brackets`. |
| `--hdr-exposure exif\|learn` | route B: hold each frame's exposure at its EXIF value, or learn a per-frame offset from it (`--hdr-exposure-prior 1`, `--hdr-exposure-lr 2e-3`, `--hdr-exposure-max-delta 3` stops). With exact EXIF the two score the same; `learn` is for cameras whose tags are rounded. |
| `--hdr-bracket-bg 1` | route B: a random radiance background behind the model every step (default on; it was +0.8 dB). `--hdr-bracket-hat` and `--hdr-bracket-err-ema` are experiments that default to 0. |
| `--test-by-position` | hold out whole positions (every frame of every 8th position) rather than single frames; a bracket run does this whenever it holds out at all. |
| `--eval` | needed for any metric: writes `metrics.csv`, `per_image_metrics.json`, `eval_step_N/*.png` (μ-law) and `*.exr` (linear). |

Expect the μ-law PSNR in `metrics.csv` for an HDR run (the images are normalised by the reference's P99 and
μ-law compressed with μ = 5000 before the PSNR, so bright regions do not dominate); the LDR PSNR at each source
exposure is what `tools/eval/bracket_eval.py` computes after the fact, for either route. `hdr_scale` (the P99
white) and `hdr_color_max` (the cap) are measured from the data at the start and printed; a model started from an
HDR PLY keeps the file's.

## 5. In the viewer

An HDR model is tone-mapped live: blending happens in float radiance, then exposure, tone curve, sRGB. The
*Rendering ▸ Color & Appearance* panel shows, for an HDR model only:

- **Exposure (EV)**, −12 … +12, 0 EV puts the scene's white (`hdr_scale`) at 1.0. Sweep it: a lamp becomes a
  textured disc at −4 EV, the shadows open at +4 EV, no banding (the output is 8-bit from a float chain).
- **Tone curve**, the existing operators (`none` = a plain clamp, `linear`, `filmic`, `hejl`, `aces`, `aces2`,
  `neutral`), applied to exposed linear radiance.
- **False colour (stops)**: luminance in stops relative to the scene's white on a viridis ramp, −8 … +8 EV,
  magenta where the model sits at its cap (a clipped light source, §1). Independent of the exposure.
- **HDR display** (on): off shows the raw log values, i.e. what a viewer that does not know the convention
  would show — a quick way to see why the PLY carries a header.

Frame time is within 1 % of the LDR path for the same model (PLAN.md M4). Known limits: one activation per draw,
so a scene mixing HDR and LDR nodes is not supported (the merge refuses it); selection overlays are drawn in
normalised radiance at EV 0 and drift with the exposure.

## 6. Export matrix

| format | HDR kept? | what happens with an HDR model |
|---|---|---|
| PLY `--hdr-export linear` (default for an HDR model) | **yes** | linear radiance in the standard SH convention + the `lichtfeld_hdr` header below; any splat tool reads it, with values above 1. Degree ≥ 1 is a least-squares projection (≈ 0.001 dB on the room). |
| PLY `--hdr-export log` | yes (lossless) | LichtFeld's internal log SH, verbatim; meaningless to other readers. |
| `.licht` project | yes (lossless) | the raw log tensors and the HDR state. |
| PLY `--hdr-export baked --hdr-export-ev <EV> --hdr-tonemap <op>` | **bakes** | display-referred colours in [0, 1], no HDR header: opens in any viewer at that exposure. |
| SPZ, SOG, SSOG, HTML viewer, GLB, USD, NuRec USDZ, RAD (incl. the streaming PLY → RAD converter and `--tiles`) | **refuses** unless `baked` | these hold [0, 1] colour (quantised): "HDR model: this format holds LDR colour" — add `--hdr-export baked` and the same bake runs first. `--tiles` of an HDR PLY is refused outright. |
| Headless video `--render-camera-path` | tone-mapped | `--render-exposure <EV> --render-tonemap <op>`; `--render-exr-dir <dir>` also writes every frame as a linear EXR. |
| `--eval` renders | yes | `eval_step_N/*.exr` linear beside the μ-law PNG. |

CLI: the three `--hdr-export / --hdr-export-ev / --hdr-tonemap` flags on the training `--export` and on `convert`;
Python: `lf.io.save_ply(..., hdr="auto|linear|baked|log", exposure_ev=, tonemap=)` and the same keywords on the
other writers; the export panel shows an **HDR COLOUR** row when the selected splat is HDR.

## 7. Colour management

**Working space: linear sRGB / Rec.709 primaries, D65.** The merge keeps the camera JPEG's primaries (sRGB for
every camera's JPEG; a RAW developed to a wider space is simply carried through, unlabelled), the training and
the viewer treat the three channels as linear light in those primaries, and the bake and the viewer apply the
piecewise sRGB OETF (`display_color.inc`, one definition for both). No EXR is tagged with chromaticities and no
input is converted. A colour-managed path (OCIO configs, display transforms, ACEScg as the working space, tagged
EXRs) is being added on branch `hdr-ocio`; until it lands, keep the shoot in sRGB and treat the EXRs as linear
sRGB.

## 8. Testing

`tools/run_hdr_tests.sh` runs every HDR test and prints one summary (see PLAN.md M6): the `tools/` pytest (metrics,
the synthetic camera, the PLY round trip through the binary, the LDR bit-exact re-evaluation — the last two skip
when the binary or `out/baseline_a` is absent), the plugin's tests, and `ctest -L hdr` on the C++ test tree
(`-LE gpu` on a CPU-only machine). `.github/workflows/hdr.yml` runs the CPU part on every push to an `hdr*` branch.

---

# The file convention, in detail

LichtFeld's `--hdr` mode trains Gaussians whose colour is **linear radiance in absolute scene units**, unbounded
above 1 (PLAN.md M2). Inside the app the SH coefficients hold the *logarithm* of that radiance; a file never
has to: the exporters write the standard PLY convention and say in the header what the numbers mean.

## The PLY convention (reader contract)

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

`--hdr-export baked --hdr-export-ev <EV> --hdr-tonemap <op>` (CLI, `convert`), `hdr="baked"` (Python),
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
        --ocio-view "ACES 2.0 - SDR 100 nits (Rec.709)" [--ocio-look <look>] [--hdr-export-ev <EV>]
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
