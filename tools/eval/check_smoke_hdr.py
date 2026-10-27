"""tools/eval/check_smoke_hdr.py — check a merged HDR dataset (the M1 plugin's output) against the synthetic truth.

Usage (from tools/):
    uv run python eval/check_smoke_hdr.py ../data/smoke_hdr ../data/brackets_raw ../data/brackets_synth/gt [--json out.json]

What it measures:
  - CRF: the dataset's `crf.json` against the synthetic camera's `<raw>/crf.json`, both matched at z = 128 (the truth
    is anchored at g(255) = 0, the recovery at g(128) = 0), max |Δg| over z ∈ [20, 235] in natural log, plus the
    count of levels the recovery's monotonic fix flattened (`monotonic_fixed_levels`, when recorded).
  - Merge: each `pos_NNN` is mapped to its view through `exposures.json` sources ↔ `exposures_gt.json`; ONE global
    log-scale per position (exp of the median log ratio truth / merged — the merged EXR's units are the CRF's,
    not the scene's) is fitted over the pixels the 0-EV frame recorded unclipped (every channel in (5, 250)), and
    the relative error |merged·s − truth| / truth over the same pixels is reported as median and p99, per position
    (the worst) and pooled over all positions (the overall).
  - Alignment: the per-source MTB shifts from exposures.json (the synthetic set is a motionless tripod: all zero).
Verdict: PASS when CRF < 0.2 nats, overall median < 5 %, overall p99 < 15 % and every |shift| ≤ 2 px.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hdr_metrics import read_exr  # noqa: E402

CRF_Z_LO, CRF_Z_HI, CRF_Z_MATCH = 20, 235, 128
CLIP_LO, CLIP_HI = 5, 250
LIMIT_CRF = 0.2
LIMIT_MEDIAN = 0.05
LIMIT_P99 = 0.15
LIMIT_SHIFT = 2
#: Pooled pixels per position for the overall statistics (a stride subsample keeps memory flat).
POOL_PER_POSITION = 200_000


def _g(d: dict) -> np.ndarray:
    g = np.asarray(d["log_irradiance"], dtype=np.float64)
    if g.ndim == 1:
        g = np.tile(g, (3, 1))
    return g - g[:, CRF_Z_MATCH : CRF_Z_MATCH + 1]


def crf_error(rec: dict, truth: dict) -> float:
    return float(np.max(np.abs(_g(rec) - _g(truth))[:, CRF_Z_LO : CRF_Z_HI + 1]))


def position_error(merged: np.ndarray, truth: np.ndarray, zero_ev: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Relative errors over the 0-EV frame's unclipped pixels after one global log-scale; (errors, scale, share)."""
    mask = np.all((zero_ev > CLIP_LO) & (zero_ev < CLIP_HI), axis=2)
    mask &= np.all((truth > 0) & (merged > 0) & np.isfinite(merged), axis=2)
    if not mask.any():
        return np.empty(0), float("nan"), 0.0
    m, t = merged[mask].astype(np.float64), truth[mask].astype(np.float64)
    s = float(np.exp(np.median(np.log(t / m))))
    rel = (np.abs(m * s - t) / t).ravel()
    return rel, s, float(mask.mean())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("smoke", type=Path, help="merged dataset (exposures.json, crf.json, images_hdr/)")
    ap.add_argument("raw", type=Path, help="the brackets (exposures_gt.json, crf.json, IMG_*.jpg)")
    ap.add_argument("gt", type=Path, help="the truth EXRs (view_NNN.exr)")
    ap.add_argument("--json", type=Path, default=None, help="also write the numbers here")
    a = ap.parse_args(argv)

    idx = json.loads((a.smoke / "exposures.json").read_text())
    gt = json.loads((a.raw / "exposures_gt.json").read_text())
    by_file = {r["file"]: r for r in gt["images"]}
    crf_rec = json.loads((a.smoke / idx.get("crf", "crf.json")).read_text())
    crf_truth = json.loads((a.raw / "crf.json").read_text())
    d_crf = crf_error(crf_rec, crf_truth)
    mono = crf_rec.get("monotonic_fixed_levels")

    rows = []
    pooled = []
    all_shifts = []
    for pos in idx["positions"]:
        srcs = pos["sources"]
        views = {by_file[s["file"]]["view"] for s in srcs if s["file"] in by_file}
        if len(views) != 1:
            print(f"{pos['name']}: sources map to views {sorted(views)}; skipped", file=sys.stderr)
            continue
        view = views.pop()
        zero = min(srcs, key=lambda s: abs(by_file[s["file"]]["ev"]))
        z0 = np.asarray(Image.open(a.raw / zero["file"]).convert("RGB"))
        merged = read_exr(str(a.smoke / pos["hdr"]))
        truth = read_exr(str(a.gt / f"{view}.exr"))
        rel, s, share = position_error(merged, truth, z0)
        shifts = [tuple(s_["shift"]) for s_ in srcs]
        all_shifts.extend(shifts)
        smax = max(max(abs(int(x)), abs(int(y))) for x, y in shifts)
        rows.append(
            {
                "name": pos["name"],
                "view": view,
                "median": float(np.median(rel)) if rel.size else float("nan"),
                "p99": float(np.percentile(rel, 99)) if rel.size else float("nan"),
                "scale": s,
                "share": share,
                "shift_max": smax,
                "shifted": sum(1 for sh in shifts if sh != (0, 0)),
                "saturated": pos.get("saturated"),
            }
        )
        if rel.size:
            step = max(1, rel.size // POOL_PER_POSITION)
            pooled.append(rel[::step])

    if not rows:
        print("no position could be checked")
        return 2
    pool = np.concatenate(pooled) if pooled else np.empty(0)
    overall_med = float(np.median(pool)) if pool.size else float("nan")
    overall_p99 = float(np.percentile(pool, 99)) if pool.size else float("nan")
    worst_med = max(rows, key=lambda r: r["median"])
    worst_p99 = max(rows, key=lambda r: r["p99"])
    shift_max = max(r["shift_max"] for r in rows)
    n_shifted = sum(1 for sh in all_shifts if sh != (0, 0))
    n_limit = sum(1 for sh in all_shifts if max(abs(sh[0]), abs(sh[1])) > LIMIT_SHIFT)

    print(f"{'pos':8} {'view':9} {'median':>7} {'p99':>7} {'unclip':>7} {'shift':>5} {'moved':>5} {'sat':>8}")
    for r in rows:
        sat = "" if r["saturated"] is None else str(r["saturated"])
        print(
            f"{r['name']:8} {r['view']:9} {100 * r['median']:6.2f}% {100 * r['p99']:6.2f}% {100 * r['share']:6.1f}% "
            f"{r['shift_max']:5d} {r['shifted']:5d} {sat:>8}"
        )
    print()
    print(f"positions checked     {len(rows)} / {len(idx['positions'])}")
    print(
        f"CRF max |dg| [{CRF_Z_LO},{CRF_Z_HI}]  {d_crf:.3f} nats"
        + (f"  (monotonic fix flattened {mono} levels)" if mono is not None else "  (no monotonic fix recorded)")
    )
    print(
        f"merge median rel err  overall {100 * overall_med:.2f}%  worst {100 * worst_med['median']:.2f}% ({worst_med['name']})"
    )
    print(
        f"merge p99 rel err     overall {100 * overall_p99:.2f}%  worst {100 * worst_p99['p99']:.2f}% ({worst_p99['name']})"
    )
    print(
        f"shifts                max |shift| {shift_max} px, {n_shifted}/{len(all_shifts)} frames moved, "
        f"{n_limit} beyond {LIMIT_SHIFT} px"
    )
    st = idx.get("stats") or {}
    if st:
        print(f"exposures.json stats  {json.dumps(st)}")

    checks = {
        f"CRF < {LIMIT_CRF}": d_crf < LIMIT_CRF,
        f"median < {LIMIT_MEDIAN:.0%}": overall_med < LIMIT_MEDIAN,
        f"p99 < {LIMIT_P99:.0%}": overall_p99 < LIMIT_P99,
        f"shift_max <= {LIMIT_SHIFT}": shift_max <= LIMIT_SHIFT,
    }
    ok = all(checks.values())
    failed = [k for k, v in checks.items() if not v]
    print(
        f"VERDICT: {'PASS' if ok else 'FAIL'} — CRF {d_crf:.3f} nats, median {100 * overall_med:.2f}%, "
        f"p99 {100 * overall_p99:.2f}%, shift_max {shift_max}" + (f" (failed: {', '.join(failed)})" if failed else "")
    )
    if a.json:
        a.json.write_text(
            json.dumps(
                {
                    "crf_max_dg": d_crf,
                    "monotonic_fixed_levels": mono,
                    "median": overall_med,
                    "p99": overall_p99,
                    "shift_max": shift_max,
                    "frames_moved": n_shifted,
                    "pass": ok,
                    "positions": rows,
                },
                indent=1,
            )
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
