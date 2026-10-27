"""tools/eval/compare_hdr_vs_proxy.py — M2 "Done when": does HDR training beat a model trained on the tonemapped
proxies, measured like-for-like?

Usage (from tools/):
    uv run python eval/compare_hdr_vs_proxy.py \
        --hdr-run ../../LichtFeld-HDR-m2/out/m2_hdr --proxy-run ../../LichtFeld-HDR-m2/out/m2_proxy \
        --dataset ../data/smoke_hdr [--alt-dataset DIR] [--step 3000] [--no-log]

Inputs:
  - HDR run: `eval_step_N/<pos>.exr` = LINEAR render of each held-out view (merge units).
  - Proxy run: `eval_step_N/<pos>.png` = LichtFeld's eval grid: a caption band (png_h − H rows) on top of
    Ground Truth | 4 px white gap | Render; the render is `[band:, W+4 : 2W+4]`. The GT half is checked against the
    dataset's proxy PNG: a mismatch means the dataset on disk is NOT the one the model was trained on (warned).
  - `--dataset`: the M1 output the models trained on (merged EXRs, ACES proxies, exposures.json with proxy_scale,
    crf.json). `--alt-dataset`: a second merge to also score against (e.g. a re-merge made after training).
  - Synthetic truth: `<raw>/exposures_gt.json` maps each source JPEG to `<truth>/view_NNN.exr`.

What it computes per held-out position:
  1. μ-law PSNR (μ = 5000, both normalised by the reference's P99, `hdr_metrics.mu_law_psnr`) of the HDR render vs
     the merged EXR, and vs the synthetic truth × k, k = exp(median log(merge / truth)) pooled over the held-out
     positions' pixels the 0-EV frame recorded unclipped (every channel in (5, 250); check_smoke_hdr's definition) —
     ONE global scale, because the merge is defined only up to one.
  2. Proxy render inverted: /255 → inverse sRGB OETF → inverse Narkowicz ACES (closed-form root of
     (2.51 − 2.43y)x² + (0.03 − 0.59y)x − 0.14y = 0; y ≥ 1, i.e. a channel at 255, is not invertible and is clipped
     to the root at y = 1, x ≈ 7.245) → / proxy_scale = radiance in merge units; then the same two μ-law PSNRs.
     Ceiling: the proxy GROUND-TRUTH PNG inverted the same way vs the merge (what tonemapping + 8 bit alone lose).
  3. LDR PSNR at each source exposure (`hdr_metrics.ldr_psnr_at_exposure`, pixels clipped in the photo ignored):
     render × t × gain → the dataset's CRF → vs the real JPEG. gain is fitted ONCE by least squares in log space on
     the merged EXR vs one JPEG (the first held-out position's reference frame): log gain = mean(g(z) − log(E·t))
     over its unclipped pixels. The merge itself is scored the same way as a reference row.
Logs the means to out/metrics.csv (milestone M2, scene smoke_hdr, run m2_compare) unless --no-log.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hdr_metrics import ldr_psnr_at_exposure, mu_law_psnr, read_exr  # noqa: E402
from log_metrics import DEFAULT_CSV, log_rows  # noqa: E402

TOOLS = Path(__file__).resolve().parents[1]
CLIP_LO, CLIP_HI = 5, 250
GAP = 4
MU = 5000.0


# ── Proxy inversion ──────────────────────────────────────────────────────────


def srgb_eotf(v: np.ndarray) -> np.ndarray:
    """sRGB-encoded [0, 1] → linear [0, 1] (inverse of proxy.srgb_oetf)."""
    v = np.clip(np.asarray(v, dtype=np.float64), 0.0, 1.0)
    return np.where(v <= 0.04045, v / 12.92, np.power((v + 0.055) / 1.055, 2.4))


def aces_inverse(y: np.ndarray) -> np.ndarray:
    """Display-linear y ∈ [0, 1] → keyed linear x ≥ 0 for Narkowicz's fit (monotonic on x ≥ 0). y ≥ 1 → the root at 1."""
    y = np.clip(np.asarray(y, dtype=np.float64), 0.0, 1.0)
    a = 2.51 - 2.43 * y
    b = 0.03 - 0.59 * y
    c = -0.14 * y
    return (-b + np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)


def aces_forward(x: np.ndarray) -> np.ndarray:
    x = np.maximum(np.asarray(x, dtype=np.float64), 0.0)
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)


def invert_proxy(png_u8: np.ndarray, proxy_scale: float) -> tuple[np.ndarray, float, float]:
    """uint8 sRGB ACES proxy → linear radiance in merge units; also the pixel / channel fraction at 255."""
    u = np.asarray(png_u8)
    x = aces_inverse(srgb_eotf(u.astype(np.float64) / 255.0))
    sat = u >= 255
    return x / proxy_scale, float(sat.any(axis=-1).mean()), float(sat.mean())


# ── I/O helpers ──────────────────────────────────────────────────────────────


def read_png(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def split_eval_png(path: Path, w: int, h: int) -> tuple[np.ndarray, np.ndarray, int]:
    """LichtFeld's eval PNG → (gt, render, caption band height). Render-only PNGs (width == w) pass through."""
    img = read_png(path)
    if img.shape[1] == w and img.shape[0] == h:
        return img, img, 0
    if img.shape[1] != 2 * w + GAP or img.shape[0] < h:
        raise ValueError(f"{path}: unexpected eval PNG size {img.shape[1]}x{img.shape[0]} for {w}x{h} views")
    band = img.shape[0] - h
    return img[band:, :w], img[band:, w + GAP: 2 * w + GAP], band


def jpeg(path: Path) -> np.ndarray:
    return read_png(path)


def crf_g(crf: dict) -> np.ndarray:
    g = np.asarray(crf["log_irradiance"], dtype=np.float64)
    return np.tile(g, (3, 1)) if g.ndim == 1 else g


def unclipped(photo: np.ndarray) -> np.ndarray:
    return np.all((photo > CLIP_LO) & (photo < CLIP_HI), axis=-1)


def fmt(v: float) -> str:
    return "   nan" if not math.isfinite(v) else f"{v:6.2f}"


# ── Main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    m2 = TOOLS.parents[1] / "LichtFeld-HDR-m2" / "out"
    ap.add_argument("--hdr-run", type=Path, default=m2 / "m2_hdr")
    ap.add_argument("--proxy-run", type=Path, default=m2 / "m2_proxy")
    ap.add_argument("--step", type=int, default=3000)
    ap.add_argument("--dataset", type=Path, default=TOOLS.parent / "data" / "smoke_hdr")
    ap.add_argument("--alt-dataset", type=Path, default=None, help="a second merge to also score against")
    ap.add_argument("--truth", type=Path, default=TOOLS.parent / "data" / "brackets_synth" / "gt")
    ap.add_argument("--raw-gt", type=Path, default=None, help="exposures_gt.json (default <source_dir>/exposures_gt.json)")
    ap.add_argument("--mu", type=float, default=MU)
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--run", default="m2_compare")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args(argv)
    mu = args.mu

    ds = args.dataset
    exp = json.loads((ds / "exposures.json").read_text())
    crf = json.loads((ds / exp.get("crf", "crf.json")).read_text())
    raw_dir = Path(exp["source_dir"])
    gt_json = json.loads((args.raw_gt or raw_dir / "exposures_gt.json").read_text())
    by_file = {im["file"]: im for im in gt_json["images"]}
    positions = {p["name"]: p for p in exp["positions"]}
    alt = None
    if args.alt_dataset:
        alt_exp = json.loads((args.alt_dataset / "exposures.json").read_text())
        alt = {p["name"]: p for p in alt_exp["positions"]}
    if exp.get("proxy_operator", "aces") != "aces" or float(exp.get("proxy_ev", 0.0)) != 0.0:
        raise SystemExit(f"proxy is {exp.get('proxy_operator')} at ev {exp.get('proxy_ev')}: only aces @ 0 is inverted")

    hdr_eval = args.hdr_run / f"eval_step_{args.step}"
    prx_eval = args.proxy_run / f"eval_step_{args.step}"
    names = sorted(p.stem for p in hdr_eval.glob("*.exr"))
    if not names:
        raise SystemExit(f"no EXR renders in {hdr_eval}")
    if names != sorted(p.stem for p in prx_eval.glob("*.png")):
        raise SystemExit("the two runs did not hold out the same views")

    # ── Load everything once ──
    views: dict[str, dict] = {}
    notes: list[str] = []
    for n in names:
        p = positions[n]
        merge = read_exr(ds / p["hdr"]).astype(np.float64)
        h, w = merge.shape[:2]
        files = [s["file"] for s in p["sources"]]
        view_ids = {by_file[f]["view"] for f in files}
        if len(view_ids) != 1:
            raise SystemExit(f"{n}: sources map to several views {view_ids}")
        view = view_ids.pop()
        hdr = read_exr(hdr_eval / f"{n}.exr").astype(np.float64)
        if hdr.shape != merge.shape:
            raise SystemExit(f"{n}: HDR render {hdr.shape} vs merge {merge.shape}")
        gt_half, prx_render, band = split_eval_png(prx_eval / f"{n}.png", w, h)
        _, hdr_png_render, hdr_band = split_eval_png(hdr_eval / f"{n}.png", w, h)
        proxy_gt = read_png(ds / p["proxy"])
        gt_diff = float(np.abs(gt_half.astype(np.float64) - proxy_gt).mean())
        if gt_diff > 0.5:
            notes.append(f"WARNING {n}: the proxy run's eval GT differs from {ds / p['proxy']} by {gt_diff:.2f}/255 "
                         "mean — this dataset is not the one the model was trained on")
        views[n] = dict(
            pos=p, merge=merge, view=view, hdr=hdr, prx_png=prx_render, proxy_gt=proxy_gt, gt_diff=gt_diff,
            band=band, hdr_band=hdr_band, hdr_png_render=hdr_png_render,
            truth=read_exr(args.truth / f"{view}.exr").astype(np.float64),
            jpgs=[(s, jpeg(raw_dir / s["file"])) for s in p["sources"]],
            alt=read_exr(args.alt_dataset / alt[n]["hdr"]).astype(np.float64) if alt else None,
        )

    # ── Global merge ↔ truth scale over 0-EV-unclipped pixels ──
    ratios = []
    for v in views.values():
        ref = min(v["jpgs"], key=lambda sj: abs(by_file[sj[0]["file"]]["ev"]))[1]
        m = unclipped(ref) & np.all(v["truth"] > 0, axis=-1) & np.all(v["merge"] > 0, axis=-1)
        ratios.append((np.log(v["merge"][m]) - np.log(v["truth"][m])).ravel())
    k_truth = float(np.exp(np.median(np.concatenate(ratios))))
    k_truth_alt = None
    if alt:
        ra = []
        for v in views.values():
            ref = min(v["jpgs"], key=lambda sj: abs(by_file[sj[0]["file"]]["ev"]))[1]
            m = unclipped(ref) & np.all(v["truth"] > 0, axis=-1) & np.all(v["alt"] > 0, axis=-1)
            ra.append((np.log(v["alt"][m]) - np.log(v["truth"][m])).ravel())
        k_truth_alt = float(np.exp(np.median(np.concatenate(ra))))

    # ── Exposure gain: least squares in log space, merged EXR vs one JPEG ──
    g = crf_g(crf)
    first = views[names[0]]
    ref_src = next((s for s in first["pos"]["sources"] if s.get("align") == "reference"), first["pos"]["sources"][0])
    ref_jpg = next(j for s, j in first["jpgs"] if s["file"] == ref_src["file"])
    mref = unclipped(ref_jpg) & np.all(first["merge"] > 0, axis=-1)
    lg = [g[c][ref_jpg[..., c][mref]] - np.log(first["merge"][..., c][mref] * ref_src["exposure_time"]) for c in range(3)]
    gain = float(np.exp(np.mean(np.concatenate(lg))))
    gain_check = []
    for v in views.values():
        for s, j in v["jpgs"]:
            mm = unclipped(j) & np.all(v["merge"] > 0, axis=-1)
            if mm.sum() < 1000:
                continue
            d = np.concatenate([g[c][j[..., c][mm]] - np.log(v["merge"][..., c][mm] * s["exposure_time"]) for c in range(3)])
            gain_check.append(float(np.exp(np.mean(d))))

    # ── Per-position metrics ──
    evs = sorted({by_file[s["file"]]["ev"] for v in views.values() for s, _ in v["jpgs"]})
    rows = []
    sat_render, sat_render_ch, sat_gt, sat_gt_ch = [], [], [], []
    for n in names:
        v = views[n]
        scale = v["pos"]["proxy_scale"]
        prx_lin, s_px, s_ch = invert_proxy(v["prx_png"], scale)
        gt_lin, g_px, g_ch = invert_proxy(v["proxy_gt"], scale)
        sat_render.append(s_px), sat_render_ch.append(s_ch), sat_gt.append(g_px), sat_gt_ch.append(g_ch)
        truth = v["truth"] * k_truth
        r = dict(
            name=n, view=v["view"],
            hdr_vs_merge=mu_law_psnr(v["hdr"], v["merge"], mu),
            proxy_vs_merge=mu_law_psnr(prx_lin, v["merge"], mu),
            proxy_gt_ceiling=mu_law_psnr(gt_lin, v["merge"], mu),
            hdr_vs_truth=mu_law_psnr(v["hdr"], truth, mu),
            proxy_vs_truth=mu_law_psnr(prx_lin, truth, mu),
            merge_vs_truth=mu_law_psnr(v["merge"], truth, mu),
        )
        if alt:
            r["hdr_vs_alt"] = mu_law_psnr(v["hdr"], v["alt"], mu)
            r["proxy_vs_alt"] = mu_law_psnr(prx_lin, v["alt"], mu)
        for s, j in v["jpgs"]:
            ev = by_file[s["file"]]["ev"]
            t = s["exposure_time"]
            r[("ldr_hdr", ev)] = ldr_psnr_at_exposure(v["hdr"], j, t, crf, gain)
            r[("ldr_proxy", ev)] = ldr_psnr_at_exposure(prx_lin, j, t, crf, gain)
            r[("ldr_merge", ev)] = ldr_psnr_at_exposure(v["merge"], j, t, crf, gain)
        rows.append(r)

    # ── Report ──
    print(f"dataset: {ds}   held-out: {', '.join(names)}   step {args.step}   mu {mu:g}")
    print(f"eval PNG layout: {2 * w + GAP}x{h + views[names[0]]['band']} = caption band {views[names[0]]['band']} px "
          f"on top, GT [0:{w}] | {GAP} px gap | render [{w + GAP}:{2 * w + GAP}] (HDR run's PNG: same, band "
          f"{views[names[0]]['hdr_band']}, halves in mu-law); EXR render-only {w}x{h}")
    print("proxy run's eval GT half vs dataset proxy PNG, mean |Δ|/255: "
          + ", ".join(f"{n} {views[n]['gt_diff']:.3f}" for n in names))
    alt_k = f"; alt merge k = {k_truth_alt:.4f}" if alt else ""
    print(f"merge/truth scale k = {k_truth:.4f} (exp median log ratio, 0-EV-unclipped pixels, pooled){alt_k}")
    print(f"exposure gain = {gain:.5f} (LSQ log, merge vs {ref_src['file']} t={ref_src['exposure_time']}); "
          f"same fit over every held-out JPEG: median {np.median(gain_check):.5f}, "
          f"range {min(gain_check):.5f}–{max(gain_check):.5f} ({len(gain_check)} frames)")
    print()
    cols = ["hdr_vs_merge", "proxy_vs_merge", "proxy_gt_ceiling", "hdr_vs_truth", "proxy_vs_truth", "merge_vs_truth"]
    if alt:
        cols += ["hdr_vs_alt", "proxy_vs_alt"]
    print("mu-law PSNR (dB)")
    print(f"{'position':9s} {'view':9s} " + " ".join(f"{c:>16s}" for c in cols))
    for r in rows:
        print(f"{r['name']:9s} {r['view']:9s} " + " ".join(f"{fmt(r[c]):>16s}" for c in cols))
    means = {c: float(np.mean([r[c] for r in rows])) for c in cols}
    print(f"{'mean':9s} {'':9s} " + " ".join(f"{fmt(means[c]):>16s}" for c in cols))
    print()
    print("LDR PSNR at each source exposure (dB; photo pixels in (5, 250) only; fraction evaluated in brackets)")
    print(f"{'position':9s} {'model':6s} " + " ".join(f"{f'EV {ev:+g}':>16s}" for ev in evs))
    for r in rows:
        for model in ("hdr", "proxy", "merge"):
            cells = [r[(f"ldr_{model}", ev)] for ev in evs]
            print(f"{r['name']:9s} {model:6s} " + " ".join(f"{fmt(c[0]):>9s} ({c[1]:.2f})" for c in cells))
    ldr_means = {}
    for model in ("hdr", "proxy", "merge"):
        for ev in evs:
            vals = [r[(f"ldr_{model}", ev)][0] for r in rows if math.isfinite(r[(f"ldr_{model}", ev)][0])]
            ldr_means[(model, ev)] = float(np.mean(vals)) if vals else float("nan")
        print(f"{'mean':9s} {model:6s} " + " ".join(f"{fmt(ldr_means[(model, ev)]):>16s}" for ev in evs))
    ldr_all = {m: float(np.nanmean([ldr_means[(m, ev)] for ev in evs])) for m in ("hdr", "proxy", "merge")}
    print(f"mean over EVs: hdr {ldr_all['hdr']:.2f}, proxy {ldr_all['proxy']:.2f}, merge {ldr_all['merge']:.2f}")
    print()
    print(f"proxy inversion saturation (a channel at 255 = keyed x ≥ 7.245, clipped): model render "
          f"{np.mean(sat_render) * 100:.3f}% of pixels ({np.mean(sat_render_ch) * 100:.3f}% of channels); proxy GT "
          f"{np.mean(sat_gt) * 100:.3f}% ({np.mean(sat_gt_ch) * 100:.3f}%)")
    for line in notes:
        print(line)
    d_merge = means["hdr_vs_merge"] - means["proxy_vs_merge"]
    d_truth = means["hdr_vs_truth"] - means["proxy_vs_truth"]
    verb = "beats" if d_merge > 0 and d_truth > 0 else "does NOT beat (Δ = HDR − proxy)"
    print(f"\nVERDICT: HDR {verb} proxy: {d_merge:+.2f} dB mu-law vs merge; {d_truth:+.2f} dB vs truth "
          f"(LDR at the source exposures: {ldr_all['hdr'] - ldr_all['proxy']:+.2f} dB mean over EVs)")

    if not args.no_log:
        metrics = {f"mu_psnr_{c}": means[c] for c in cols}
        for model in ("hdr", "proxy", "merge"):
            for ev in evs:
                metrics[f"ldr_psnr_{model}_ev{ev:+g}"] = ldr_means[(model, ev)]
            metrics[f"ldr_psnr_{model}_mean"] = ldr_all[model]
        metrics.update(
            mu_delta_vs_merge=d_merge, mu_delta_vs_truth=d_truth, merge_truth_scale=k_truth, exposure_gain=gain,
            proxy_render_saturated_frac=float(np.mean(sat_render)), proxy_gt_saturated_frac=float(np.mean(sat_gt)),
            iteration=args.step,
        )
        log_rows(args.csv, milestone="M2", scene="smoke_hdr", run=args.run, metrics=metrics)
        print(f"logged {len(metrics)} metrics to {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
