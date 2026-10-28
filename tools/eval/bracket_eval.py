"""tools/eval/bracket_eval.py — M3 "Done when": a model trained from the bracket JPEGs, scored per EV on the held-out
positions, next to M2's model trained from the merged EXRs.

Usage (from tools/):
    uv run python eval/bracket_eval.py --run ../out/m3_learn --step 7000 --dataset ../data/smoke_hdr \
        --truth ../data/brackets_synth/gt [--reference-run ../../LichtFeld-HDR/out/<m2 run>] [--no-log]

Inputs:
  - run: `eval_step_N/<stem>.exr`, the LINEAR render of each held-out frame (bracket run: pos_NNN_fK.exr, identical
    radiance per position — f0 is used; an M2 run: pos_NNN.exr). Same for --reference-run.
  - dataset: the M1 output; its `brackets/` set (exposures.json with each source's `frame`, crf.json with the
    Debevec LUT and exif_ev_ref, images/<frame>.jpg), `images_hdr/pos_NNN.exr` (the merge).
  - truth: the synthetic truth `view_NNN.exr`, joined through `<brackets_raw>/exposures_gt.json`.

Per held-out position and source frame: render × 2^(exif_ev − exif_ev_ref) (= × exposure_time at the shoot's
ISO/N) → the LUT (`hdr_metrics.apply_crf`) → PSNR against the JPEG on the pixels the JPEG recorded unclipped (every
channel in (5, 250)). The gain is NOT fitted: the EXIF is the anchor of the units. As a unit check the gain IS fitted
(least squares in log space on the position's reference frame, compare_hdr_vs_proxy.py's formula) and printed for
the render and for the merge — both must be ≈ 1.0. μ-law PSNR (μ = 5000, P99-normalised, so the merge/truth scale
cancels) against the merged EXR and the truth.
Logs ldr_psnr_ev{±n}, ldr_psnr_mean, mu_psnr_merge, mu_psnr_truth, gain_render, gain_merge to out/metrics.csv.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hdr_metrics import ldr_psnr_at_exposure, mu_law_psnr, read_exr  # noqa: E402
from log_metrics import DEFAULT_CSV, log_rows  # noqa: E402

CLIP_LO, CLIP_HI = 5, 250
STEM = re.compile(r"^(?P<pos>pos_\d{3})(?:_f(?P<frame>\d+))?$")


def crf_g(crf: dict) -> np.ndarray:
    g = np.asarray(crf["log_irradiance"], dtype=np.float64)
    return np.tile(g, (3, 1)) if g.ndim == 1 else g


def unclipped(jpg: np.ndarray) -> np.ndarray:
    return np.all((jpg > CLIP_LO) & (jpg < CLIP_HI), axis=-1)


def fit_gain(hdr: np.ndarray, jpg: np.ndarray, t: float, g: np.ndarray) -> float:
    """exp(mean(g[z] − log(E·t))) over the unclipped pixels: the factor the EXR would need to reproduce the JPEG."""
    m = unclipped(jpg) & np.all(hdr > 0, axis=-1)
    if m.sum() < 1000:
        return float("nan")
    d = np.concatenate([g[c][jpg[..., c][m]] - np.log(hdr[..., c][m] * t) for c in range(3)])
    return float(np.exp(np.mean(d)))


def held_out_renders(run: Path, step: int) -> dict[str, Path]:
    d = run / f"eval_step_{step}"
    if not d.is_dir():
        raise FileNotFoundError(f"{d} (train with --eval --test-every N; --iter {step} must be an eval step)")
    out: dict[str, Path] = {}
    for p in sorted(d.glob("*.exr")):
        m = STEM.match(p.stem)
        if not m:
            continue
        pos, frame = m.group("pos"), m.group("frame")
        if pos not in out or (frame is not None and int(frame) == 0):
            out[pos] = p
    if not out:
        raise FileNotFoundError(f"no pos_NNN[_fK].exr in {d}")
    return out


def score_run(run: Path, step: int, ds: Path, truth: Path | None, label: str) -> dict:
    br = ds / "brackets"
    exposures = json.load(open(br / "exposures.json"))
    crf = json.load(open(br / "crf.json"))
    if "exif_ev_ref" not in crf:
        raise SystemExit(f"{br / 'crf.json'} has no exif_ev_ref: not an M3 bracket set")
    ev_ref = float(crf["exif_ev_ref"])
    g = crf_g(crf)
    by_name = {p["name"]: p for p in exposures["positions"]}
    truth_of_file: dict[str, str] = {}
    if truth is not None:
        gt_index = json.load(open(Path(exposures["source_dir"]) / "exposures_gt.json"))
        truth_of_file = {im["file"]: im["view"] for im in gt_index["images"]}

    renders = held_out_renders(run, step)
    # The merge is defined only up to one global scale against the truth: k = exp(median log(merge / truth)) pooled
    # over the held-out positions' pixels the 0-EV frame recorded unclipped (compare_hdr_vs_proxy.py's definition),
    # so the truth column is like-for-like with M2's.
    k_truth = 1.0
    if truth is not None:
        ratios = []
        for pos in renders:
            info = by_name.get(pos)
            if info is None:
                continue
            ref = min(info["sources"], key=lambda s: abs(s.get("ev_step", s["exif_ev"])))
            jpg = np.asarray(Image.open(br / "images" / ref["frame"]).convert("RGB"))
            merge = read_exr(str(ds / "images_hdr" / f"{pos}.exr")).astype(np.float64)
            tr = read_exr(str(truth / f"{truth_of_file[info['sources'][0]['file']]}.exr")).astype(np.float64)
            m = unclipped(jpg) & np.all(tr > 0, axis=-1) & np.all(merge > 0, axis=-1)
            ratios.append((np.log(merge[m]) - np.log(tr[m])).ravel())
        k_truth = float(np.exp(np.median(np.concatenate(ratios))))
    per_ev: dict[float, list[float]] = {}
    frac_ev: dict[float, list[float]] = {}
    mu_merge, mu_truth, gains_render, gains_merge = [], [], [], []
    rows = []
    for pos, exr in renders.items():
        info = by_name.get(pos)
        if info is None:
            print(f"  {pos}: not in brackets/exposures.json (omitted position?) — skipped")
            continue
        render = read_exr(str(exr)).astype(np.float64)
        merge = read_exr(str(ds / "images_hdr" / f"{pos}.exr")).astype(np.float64)
        if render.shape != merge.shape:
            raise SystemExit(f"{exr}: {render.shape} vs merge {merge.shape} — the run was trained at another resolution")
        mu_merge.append(mu_law_psnr(render, merge))
        if truth is not None:
            view = truth_of_file[info["sources"][0]["file"]]
            mu_truth.append(mu_law_psnr(render, k_truth * read_exr(str(truth / f"{view}.exr")).astype(np.float64)))
        ref = min(info["sources"], key=lambda s: abs(s.get("ev_step", s["exif_ev"])))
        line = [f"  {pos:8s}"]
        for s in sorted(info["sources"], key=lambda s: s["exif_ev"]):
            jpg = np.asarray(Image.open(br / "images" / s["frame"]).convert("RGB"))
            t = 2.0 ** (float(s["exif_ev"]) - ev_ref)  # == exposure_time at the shoot's ISO / f-number
            ev = round(float(s.get("ev_step", s["exif_ev"] - ref["exif_ev"])), 2)
            psnr, frac = ldr_psnr_at_exposure(render, jpg, t, crf, gain=1.0)
            per_ev.setdefault(ev, []).append(psnr)
            frac_ev.setdefault(ev, []).append(frac)
            line.append(f"{ev:+5.1f}EV {psnr:6.2f}")
            if s is ref:
                gains_render.append(fit_gain(render, jpg, t, g))
                gains_merge.append(fit_gain(merge, jpg, t, g))
        rows.append(line)
        print(" ".join(line) + f"   mu-law vs merge {mu_merge[-1]:.2f}" + (f", vs truth {mu_truth[-1]:.2f}" if mu_truth else ""))

    evs = sorted(per_ev)
    means = {ev: float(np.nanmean(per_ev[ev])) for ev in evs}
    ldr_mean = float(np.nanmean(list(means.values())))
    out = {
        "label": label, "evs": evs, "ldr": means, "ldr_mean": ldr_mean,
        "frac": {ev: float(np.mean(frac_ev[ev])) for ev in evs},
        "mu_merge": float(np.mean(mu_merge)), "mu_truth": float(np.mean(mu_truth)) if mu_truth else float("nan"),
        "gain_render": float(np.nanmean(gains_render)), "gain_merge": float(np.nanmean(gains_merge)),
        "positions": len(rows), "k_truth": k_truth,
    }
    return out


def print_table(results: list[dict]) -> None:
    evs = results[0]["evs"]
    print()
    print(f"{'run':14s} " + " ".join(f"{ev:+5.1f}EV " for ev in evs) + "   mean   mu/merge mu/truth  gain(render) gain(merge)")
    for r in results:
        print(f"{r['label']:14s} " + " ".join(f"{r['ldr'].get(ev, float('nan')):7.2f} " for ev in evs)
              + f" {r['ldr_mean']:6.2f}   {r['mu_merge']:6.2f}   {r['mu_truth']:6.2f}   {r['gain_render']:8.3f}     {r['gain_merge']:8.3f}")
    print(f"merge/truth scale k = {results[0]['k_truth']:.3f} (pooled median; the truth column is scored against truth x k)")
    print("evaluated fraction (unclipped pixels) per EV: " + ", ".join(f"{ev:+g}: {results[0]['frac'][ev]*100:.0f}%" for ev in evs))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--step", type=int, default=7000)
    ap.add_argument("--dataset", required=True, type=Path, help="the M1 output with its brackets/ set")
    ap.add_argument("--truth", type=Path, default=None, help="synthetic truth dir (view_NNN.exr)")
    ap.add_argument("--reference-run", type=Path, default=None, help="an M2 run (pos_NNN.exr) scored the same way")
    ap.add_argument("--reference-step", type=int, default=None)
    ap.add_argument("--run-name", default=None, help="metrics.csv run column (default: the run folder's name)")
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args(argv)

    results = [score_run(args.run, args.step, args.dataset, args.truth, args.run.name)]
    if args.reference_run is not None:
        print()
        results.append(score_run(args.reference_run, args.reference_step or args.step, args.dataset, args.truth,
                                 "M2:" + args.reference_run.name))
    print_table(results)
    r = results[0]
    for key in ("gain_render", "gain_merge"):
        if not (0.8 < r[key] < 1.25):
            print(f"UNIT CHECK FAILED: {key} = {r[key]:.3f} (expected ≈ 1: the EXIF anchor exif_ev_ref is off, design §8 risk 1)")
    verdict = "PASS" if r["ldr_mean"] >= 32.15 else "FAIL"
    print(f"\nM3 Done-when: LDR mean {r['ldr_mean']:.2f} dB vs M2's 32.15 → {verdict}"
          + (f" (reference run rescored here: {results[1]['ldr_mean']:.2f})" if len(results) > 1 else ""))
    if not args.no_log:
        metrics = {f"ldr_psnr_ev{ev:+g}": r["ldr"][ev] for ev in r["evs"]}
        metrics.update(ldr_psnr_mean=r["ldr_mean"], mu_psnr_merge=r["mu_merge"], mu_psnr_truth=r["mu_truth"],
                       gain_render=r["gain_render"], gain_merge=r["gain_merge"], iteration=args.step)
        log_rows(args.csv, milestone="M3", scene="smoke_hdr_brackets", run=args.run_name or args.run.name, metrics=metrics)
        print(f"logged {len(metrics)} metrics to {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
