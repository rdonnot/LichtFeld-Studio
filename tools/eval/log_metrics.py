"""The one writer of out/metrics.csv (columns: timestamp,commit,milestone,scene,run,metric,value).

API: log_rows(), read_rows(), baseline().
CLI: log_metrics.py --milestone M0 --scene smoke_ldr --run baseline_a psnr=24.03 ssim=0.833 [--csv PATH]
"""

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

COLUMNS = ["timestamp", "commit", "milestone", "scene", "run", "metric", "value"]
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO_ROOT / "out" / "metrics.csv"


def git_commit(repo: Path = REPO_ROOT) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def log_rows(
    csv_path: str | Path,
    milestone: str,
    scene: str,
    run: str,
    metrics: dict[str, float],
    commit: str | None = None,
) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    commit = commit or git_commit()
    ts = datetime.now().astimezone().isoformat(timespec="seconds")
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(COLUMNS)
        for name, value in metrics.items():
            w.writerow([ts, commit, milestone, scene, run, name, repr(float(value))])


def read_rows(csv_path: str | Path) -> list[dict]:
    path = Path(csv_path)
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def baseline(
    csv_path: str | Path, scene: str, metric: str, run_prefix: str = "baseline"
) -> tuple[float, float, int]:
    """Mean, sample std (0 when n < 2) and count of matching rows; (nan, 0, 0) if none."""
    vals = [
        float(r["value"])
        for r in read_rows(csv_path)
        if r["scene"] == scene and r["metric"] == metric and r["run"].startswith(run_prefix)
    ]
    if not vals:
        return float("nan"), 0.0, 0
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return statistics.fmean(vals), std, len(vals)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--milestone", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("metrics", nargs="+", help="name=value pairs")
    args = ap.parse_args(argv)
    metrics: dict[str, float] = {}
    for kv in args.metrics:
        k, sep, v = kv.partition("=")
        if not sep:
            ap.error(f"expected name=value, got {kv!r}")
        metrics[k] = float(v)
    log_rows(args.csv, args.milestone, args.scene, args.run, metrics)
    print(f"logged {len(metrics)} metric(s) to {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
