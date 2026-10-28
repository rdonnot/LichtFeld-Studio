"""Command line for the bracket merge: `python -m hdr_brackets.cli <src> <out> [options]`, Report as JSON on stdout.

Why: the same pipeline as the LichtFeld panel, scriptable (brackets_raw → smoke_hdr in one command) and testable
without the app. Logs go to stderr; the exit code is 0 when at least one position was written.
"""

from __future__ import annotations

import argparse
import logging
import sys

from .pipeline import Options, run


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="hdr_brackets", description="Merge exposure brackets to a linear-EXR dataset")
    ap.add_argument("src", help="folder of bracketed photos (IMG_*.jpg with EXIF)")
    ap.add_argument("out", help="output dataset folder (images_hdr/, images/, sparse/, exposures.json)")
    ap.add_argument("--no-sfm", action="store_true", help="skip pycolmap SfM on the proxies")
    ap.add_argument("--reuse-sfm", action="store_true", help="keep an existing <out>/sparse/0 instead of running SfM")
    ap.add_argument(
        "--brackets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write <out>/brackets/ (the raw JPEGs with their position's pose, for --hdr-brackets training)",
    )
    ap.add_argument("--no-deghost", action="store_true", help="plain Debevec merge (OpenCV), no ghost weights")
    ap.add_argument("--proxy-ev", type=float, default=0.0, help="EV offset of the proxies' auto key")
    ap.add_argument("--proxy-operator", default="aces", choices=["aces", "reinhard", "linear"])
    ap.add_argument("--crf", default=None, help="reuse a saved crf.json instead of calibrating")
    ap.add_argument("--max-gap", type=float, default=5.0, help="seconds between frames that start a new bracket")
    ap.add_argument("--ghost-threshold", type=float, default=Options.ghost_threshold)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        stream=sys.stderr,
        format="%(levelname)s %(name)s: %(message)s",
    )
    opts = Options(
        max_gap_s=a.max_gap,
        deghost=not a.no_deghost,
        proxy_operator=a.proxy_operator,
        proxy_ev=a.proxy_ev,
        run_sfm=not a.no_sfm,
        crf_path=a.crf,
        ghost_threshold=a.ghost_threshold,
        write_brackets=a.brackets,
        reuse_sfm=a.reuse_sfm,
    )
    last = [-1.0]

    def progress(label: str, f: float) -> None:
        if f - last[0] >= 0.05 or f >= 1.0:
            last[0] = f
            logging.getLogger("hdr_brackets.cli").info("[%3.0f%%] %s", 100 * f, label)

    report = run(a.src, a.out, opts, progress)
    sys.stdout.write(report.to_json() + "\n")
    return 0 if report.n_positions > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
