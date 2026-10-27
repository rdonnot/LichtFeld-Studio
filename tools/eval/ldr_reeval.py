"""tools/eval/ldr_reeval.py — the BIT-EXACT LDR regression check: re-evaluate a frozen project with the current build.

Training is not reproducible run to run (M0 measured 27.2 vs 29.2 dB PSNR for two identical 7000-iteration runs on
smoke_ldr: MRNF's refinement is stochastic and there is no seed flag), so a ±0.1 dB criterion cannot be read off
fresh trainings. What IS deterministic is evaluation of a frozen project: `--resume project.licht --iter N --eval`
re-renders the held-out views and writes identical metrics every time (checked: two re-evals of the M0 baseline gave
PSNR 27.195011 both times). So the LDR-off guarantee for every rasterizer / loader / eval change is:

    re-evaluate the reference project with the new binary → metrics.csv row AND eval PNGs byte-identical to the
    reference evaluation made by the unmodified build.

Usage (from tools/):
    uv run python eval/ldr_reeval.py ../out/baseline_a/project.licht --reference ../out/baseline_a \
        [--binary ../build/LichtFeld-Studio] [--iter 7000] [--test-every 8] [--milestone M2 --run m2_reeval]

Exit 0 when the metrics row and every PNG match; 1 otherwise (prints what differs). The reference evaluation is the
`metrics.csv` + `eval_step_<iter>/` written by the training run that produced the project.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from log_metrics import DEFAULT_CSV, log_rows  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def last_row(csv_path: Path) -> dict[str, str]:
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{csv_path}: no rows")
    return rows[-1]


def sha256(path: Path) -> str:
    """Hash of the decoded pixels, not the file: LichtFeld stamps every PNG with a provenance comment."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as im:
        arr = np.ascontiguousarray(np.asarray(im))
    h = hashlib.sha256(str(arr.shape).encode() + str(arr.dtype).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("project", type=Path, help="frozen project.licht (the reference training's output)")
    ap.add_argument("--reference", type=Path, required=True, help="the reference training's output dir")
    ap.add_argument("--binary", type=Path, default=REPO / "build" / "LichtFeld-Studio")
    ap.add_argument("--iter", type=int, default=7000)
    ap.add_argument("--test-every", type=int, default=8)
    ap.add_argument("--keep", type=Path, default=None, help="keep the re-evaluation output here")
    ap.add_argument("--milestone", default=None)
    ap.add_argument("--run", default="reeval")
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = ap.parse_args(argv)

    out = args.keep or Path(tempfile.mkdtemp(prefix="lfs_reeval_"))
    if out.exists() and args.keep:
        shutil.rmtree(out)
    cmd = [str(args.binary), "--headless", "--resume", str(args.project), "-o", str(out), "--iter", str(args.iter),
           "--test-every", str(args.test_every), "--eval"]
    print("+", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout[-3000:], res.stderr[-3000:])
        return 1

    ref_row = last_row(args.reference / "metrics.csv")
    new_row = last_row(out / "metrics.csv")
    keys = ("iteration", "psnr", "ssim", "lpips", "num_gaussians")
    diffs = [f"{k}: {ref_row.get(k)} -> {new_row.get(k)}" for k in keys if ref_row.get(k) != new_row.get(k)]

    step = new_row.get("iteration", str(args.iter))
    ref_dir = args.reference / f"eval_step_{step}"
    new_dir = out / f"eval_step_{step}"
    png_diffs: list[str] = []
    ref_pngs = sorted(ref_dir.glob("*.png")) if ref_dir.exists() else []
    for p in ref_pngs:
        q = new_dir / p.name
        if not q.exists():
            png_diffs.append(f"missing {q.name}")
        elif sha256(p) != sha256(q):
            png_diffs.append(f"differs {q.name}")

    if args.milestone:
        log_rows(args.csv, args.milestone, "smoke_ldr", args.run, {
            "psnr": float(new_row["psnr"]), "ssim": float(new_row["ssim"]), "lpips": float(new_row["lpips"]),
            "reeval_metrics_identical": 0.0 if diffs else 1.0,
            "reeval_pngs_identical": 0.0 if png_diffs else 1.0,
        })

    print(f"metrics: {'identical' if not diffs else 'DIFFER: ' + '; '.join(diffs)}")
    print(f"eval PNGs: {len(ref_pngs)} compared, {'identical' if not png_diffs else 'DIFFER: ' + ', '.join(png_diffs[:8])}")
    if not args.keep:
        shutil.rmtree(out, ignore_errors=True)
    return 0 if not diffs and not png_diffs else 1


if __name__ == "__main__":
    sys.exit(main())
