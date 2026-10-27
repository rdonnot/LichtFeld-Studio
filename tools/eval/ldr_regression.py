"""Parse a LichtFeld output dir, log its metrics and judge PSNR against the logged baseline.

Reads <out>/metrics.csv (last row or --step N) and <out>/per_image_metrics.json, logs
psnr/ssim/lpips/num_gaussians/psnr_test_std via log_metrics, then prints a verdict:
PASS if |psnr - baseline mean| <= tolerance + baseline std, else FAIL (exit 1).
No baseline rows: prints "no baseline", exit 0.

CLI: ldr_regression.py <lichtfeld_out_dir> --milestone M2 --scene smoke_ldr --run hdr_off
         [--step N] [--baseline-prefix baseline] [--tolerance-db 0.1] [--no-judge] [--csv PATH]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import log_metrics  # noqa: E402


def _f(row: dict, key: str) -> float | None:
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return None


def pick_row(rows: list[dict], step: int | None) -> dict:
    if not rows:
        raise SystemExit("metrics.csv has no rows")
    if step is None:
        return rows[-1]
    for r in rows:
        if _f(r, "iteration") == step:
            return r
    raise SystemExit(f"no row with iteration {step}")


def per_image_test_psnrs(path: Path, step: int) -> list[float]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    out: list[float] = []
    for entry in data.values():
        for ev in entry.get("evaluations", []):
            if ev.get("step") == step and ev.get("split", "test") == "test" and "psnr" in ev:
                out.append(float(ev["psnr"]))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--milestone", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--baseline-prefix", default="baseline")
    ap.add_argument("--tolerance-db", type=float, default=0.1)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--csv", default=str(log_metrics.DEFAULT_CSV))
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    with (out / "metrics.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    row = pick_row(rows, args.step)
    step = int(_f(row, "iteration") or 0)

    metrics: dict[str, float] = {}
    for key in ("psnr", "ssim", "lpips", "num_gaussians"):
        v = _f(row, key)
        if v is not None:
            metrics[key] = v
    psnrs = per_image_test_psnrs(out / "per_image_metrics.json", step)
    if len(psnrs) > 1:
        metrics["psnr_test_std"] = statistics.pstdev(psnrs)
    if "psnr" not in metrics:
        raise SystemExit("no psnr in selected metrics row")

    # Baseline is read before logging so a baseline-named run does not judge itself.
    mean, std, n = log_metrics.baseline(args.csv, args.scene, "psnr", args.baseline_prefix)
    log_metrics.log_rows(args.csv, args.milestone, args.scene, args.run, metrics)

    desc = " ".join(f"{k}={v:.4g}" for k, v in metrics.items())
    if args.no_judge:
        print(f"{args.scene}/{args.run} step {step}: {desc} (logged, not judged)")
        return 0
    if n == 0 or math.isnan(mean):
        print(f"{args.scene}/{args.run} step {step}: {desc}; no baseline")
        return 0
    delta = metrics["psnr"] - mean
    budget = args.tolerance_db + std
    ok = abs(delta) <= budget
    print(
        f"{'PASS' if ok else 'FAIL'} {args.scene}/{args.run} step {step}: psnr {metrics['psnr']:.3f} vs "
        f"baseline {mean:.3f} (n={n}, std {std:.3f}) delta {delta:+.3f} dB, budget {budget:.3f} dB"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
