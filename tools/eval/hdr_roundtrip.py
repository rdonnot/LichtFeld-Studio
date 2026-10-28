"""tools/eval/hdr_roundtrip.py — the M5 "Done when": an HDR model survives the PLY (PLAN.md M5, docs/hdr.md).

Takes a finished --hdr run (its training_config.json gives s and the cap; its eval_step_N/*.exr are the reference
renders of the held-out views) and:

  1. stamps the run's PLY as `lichtfeld_hdr log` when it predates M5 (log values written as f_dc), or uses it as is;
  2. `convert` -> linear-radiance PLY (the file other applications read), a degree-0 copy, a baked LDR PLY;
  3. `LichtFeld-Studio eval <ply> -d <dataset> --images <folder> --hdr --test-every N -o <dir>` on the log, linear
     and degree-0 files: each is loaded through the PLY reader and re-rendered on the held-out views;
  4. mu-law PSNR (tools/eval/hdr_metrics.py, mu = 5000, P99-normalised) between the original run's eval EXRs and
     each re-render's, plus each re-render's own metrics.csv PSNR against the ground truth;
  5. `hdr_ply.py compare` numbers (radiance relative error on random directions) and the file sizes.

Usage (from tools/):
    uv run python eval/hdr_roundtrip.py <run dir> --dataset ../data/smoke_hdr --images images_hdr \\
        [--binary ../build/LichtFeld-Studio] [--test-every 8] [--out <dir>] [--exposure 0 --tonemap aces]
Prints a JSON report and writes it to <out>/report.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hdr_metrics as hm  # noqa: E402
import hdr_ply  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def run(cmd: list[str], log: Path) -> None:
    print("+", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    log.write_text(res.stdout + "\n--- stderr ---\n" + res.stderr)
    if res.returncode != 0:
        print(res.stdout[-3000:], res.stderr[-3000:])
        raise SystemExit(f"command failed ({res.returncode}): {' '.join(cmd)}")


def last_metrics_row(out_dir: Path) -> dict:
    with (out_dir / "metrics.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}


def eval_dir(out_dir: Path) -> Path:
    dirs = sorted(out_dir.glob("eval_step_*"), key=lambda p: int(p.name.split("_")[-1]))
    if not dirs:
        raise SystemExit(f"{out_dir}: no eval_step_* folder")
    return dirs[-1]


def mu_psnr_between(ref_dir: Path, new_dir: Path) -> dict:
    per_view = {}
    for ref in sorted(ref_dir.glob("*.exr")):
        new = new_dir / ref.name
        if not new.exists():
            continue
        a = hm.read_exr(str(ref))
        b = hm.read_exr(str(new))
        per_view[ref.stem] = hm.mu_law_psnr(b, a)  # normalised by the reference's P99
    if not per_view:
        raise SystemExit(f"no matching EXRs between {ref_dir} and {new_dir}")
    return {"mean": float(np.mean(list(per_view.values()))), "per_view": per_view}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", type=Path, help="a finished --hdr run: training_config.json, splat_N.ply, eval_step_N/")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--images", default="images_hdr")
    ap.add_argument("--binary", type=Path, default=REPO / "build" / "LichtFeld-Studio")
    ap.add_argument("--test-every", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--exposure", type=float, default=0.0)
    ap.add_argument("--tonemap", default="aces")
    ap.add_argument("--skip-eval", action="store_true", help="only the conversions and the direction-space compare")
    args = ap.parse_args(argv)

    cfg = json.loads((args.run / "training_config.json").read_text())
    opt = cfg.get("optimization", cfg)
    scale = float(opt.get("hdr_scale", 0.0))
    cap = float(opt.get("hdr_color_max", 0.0))
    plys = sorted(args.run.glob("splat_*.ply"), key=lambda p: int(p.stem.split("_")[-1]))
    if not plys or scale <= 0:
        raise SystemExit(f"{args.run}: need splat_N.ply and an hdr_scale in training_config.json")
    src_ply = plys[-1]
    out = args.out or (REPO / "out" / f"m5_roundtrip_{args.run.name}")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    report: dict = {"run": str(args.run), "source_ply": str(src_ply), "hdr_scale": scale, "hdr_color_max": cap}

    # 1. the log PLY with its header
    _, src_hdr, _ = hdr_ply.read_ply(src_ply)
    log_ply = out / "model_log.ply"
    if src_hdr["mode"] == "log":
        shutil.copy(src_ply, log_ply)
    elif src_hdr["mode"] is None:
        hdr_ply.stamp(src_ply, log_ply, "log", scale, cap)
        report["stamped"] = "the run's PLY predates M5 (log values as f_dc): stamped lichtfeld_hdr log"
    else:
        raise SystemExit(f"{src_ply} is already {src_hdr['mode']}; give the run folder of a log/M2 export")

    # 2. conversions
    linear_ply = out / "model_linear.ply"
    linear0_ply = out / "model_linear_deg0.ply"
    log0_ply = out / "model_log_deg0.ply"
    baked_ply = out / "model_baked.ply"
    back_ply = out / "model_linear_back_to_log.ply"
    conv = [str(args.binary), "convert"]
    run(conv + [str(log_ply), str(linear_ply), "-y"], out / "convert_linear.log")
    run(conv + [str(log_ply), str(log0_ply), "-y", "--sh-degree", "0", "--hdr-export", "log"], out / "convert_log0.log")
    run(conv + [str(log0_ply), str(linear0_ply), "-y"], out / "convert_linear0.log")
    run(conv + [str(log_ply), str(baked_ply), "-y", "--hdr-export", "baked", "--hdr-exposure", str(args.exposure),
                "--hdr-tonemap", args.tonemap], out / "convert_baked.log")
    run(conv + [str(linear_ply), str(back_ply), "-y", "--hdr-export", "log"], out / "convert_back.log")
    report["files"] = {p.name: {"bytes": p.stat().st_size, "hdr": hdr_ply.read_ply(p)[1]}
                       for p in (log_ply, linear_ply, log0_ply, linear0_ply, baked_ply, back_ply)}

    # 5a. direction-space compare (numpy only)
    dirs = hdr_ply.random_directions(512, 0)
    _, h_log, d_log = hdr_ply.read_ply(log_ply)
    _, h_lin, d_lin = hdr_ply.read_ply(linear_ply)
    _, h_back, d_back = hdr_ply.read_ply(back_ply)
    n = min(d_log.shape[0], 20000)
    idx = np.random.default_rng(0).choice(d_log.shape[0], size=n, replace=False)
    r_log = hdr_ply.radiance(d_log[idx], h_log, dirs)
    r_lin = hdr_ply.radiance(d_lin[idx], h_lin, dirs)
    r_back = hdr_ply.radiance(d_back[idx], h_back, dirs)
    report["direction_space"] = {
        "splats": n, "directions": int(dirs.shape[0]),
        "log_vs_linear": hdr_ply.rel_error_quantiles(r_log, r_lin, 1e-3 * scale),
        "log_vs_linear_back_to_log": hdr_ply.rel_error_quantiles(r_log, r_back, 1e-3 * scale),
        "linear_negative_fraction": float((r_lin < 0).mean()),
    }
    _, h_b, d_b = hdr_ply.read_ply(baked_ply)
    dc_b, rest_b, _ = hdr_ply.sh_columns(d_b[idx])
    c_b = hdr_ply.expansion(dc_b, rest_b, dirs)
    report["baked"] = {"dc_min": float((0.5 + hdr_ply.C0 * dc_b).min()), "dc_max": float((0.5 + hdr_ply.C0 * dc_b).max()),
                       "expansion_min": float(c_b.min()), "expansion_max": float(c_b.max()),
                       "header_comments": [ln for ln in hdr_ply.read_ply(baked_ply)[0] if ln.startswith("comment")]}

    if args.skip_eval:
        print(json.dumps(report, indent=2))
        (out / "report.json").write_text(json.dumps(report, indent=2))
        return 0

    # 3./4. re-render through the PLY reader
    ref_dir = eval_dir(args.run)
    report["reference_eval"] = {"dir": str(ref_dir), "metrics": last_metrics_row(args.run)}
    report["reeval"] = {}
    for label, ply in (("log", log_ply), ("linear", linear_ply), ("log_deg0", log0_ply), ("linear_deg0", linear0_ply)):
        edir = out / f"eval_{label}"
        run([str(args.binary), "eval", str(ply), "-d", str(args.dataset), "--images", args.images, "--hdr",
             "--test-every", str(args.test_every), "-o", str(edir)], out / f"eval_{label}.log")
        entry = {"metrics_vs_gt": last_metrics_row(edir), "mu_psnr_vs_reference_renders": mu_psnr_between(ref_dir, eval_dir(edir))}
        report["reeval"][label] = entry
    # degree-0 copies: the linear one must match the log one exactly (up to float32)
    if "log_deg0" in report["reeval"] and "linear_deg0" in report["reeval"]:
        report["reeval"]["linear_deg0_vs_log_deg0"] = mu_psnr_between(eval_dir(out / "eval_log_deg0"), eval_dir(out / "eval_linear_deg0"))

    print(json.dumps(report, indent=2))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
