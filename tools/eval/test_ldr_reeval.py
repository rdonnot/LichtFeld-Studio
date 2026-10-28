"""The LDR bit-exact guarantee as a test (PLAN.md M6; CLAUDE.md "Metrics" tier 1).

`ldr_reeval.py` re-evaluates the frozen M0 reference project with the current binary (hdr off) and requires the
metrics row and the 16 held-out eval PNGs to be pixel-identical to the reference evaluation. The reference lives
outside git (`out/baseline_a`, ~1 GB), so this test SKIPS — never fails — when it or the binary is absent:

    LFS_BINARY          the LichtFeld-Studio binary (default <repo>/build/LichtFeld-Studio)
    LFS_LDR_REFERENCE   the reference run dir holding project.licht, metrics.csv, eval_step_7000/ (default
                        <repo>/out/baseline_a)
    LFS_HDR_SKIP_SLOW=1 skip it (tools/run_hdr_tests.sh --quick): the re-evaluation renders 16 views of a 1.1 M
                        splat model and takes a minute or two on a shared GPU.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ldr_reeval  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
BINARY = Path(os.environ.get("LFS_BINARY", REPO / "build" / "LichtFeld-Studio"))
REFERENCE = Path(os.environ.get("LFS_LDR_REFERENCE", REPO / "out" / "baseline_a"))
PROJECT = REFERENCE / "project.licht"


def _missing() -> str | None:
    if os.environ.get("LFS_HDR_SKIP_SLOW") == "1":
        return "LFS_HDR_SKIP_SLOW=1"
    if not BINARY.exists():
        return f"no binary at {BINARY}"
    if not PROJECT.exists() or not (REFERENCE / "metrics.csv").exists():
        return f"no frozen LDR reference at {REFERENCE} (project.licht + metrics.csv + eval_step_N/)"
    return None


@pytest.mark.ldr_reference
def test_ldr_reeval_is_bit_identical(tmp_path: Path) -> None:
    reason = _missing()
    if reason:
        pytest.skip(reason)
    rc = ldr_reeval.main([
        str(PROJECT), "--reference", str(REFERENCE), "--binary", str(BINARY),
        "--keep", str(tmp_path / "reeval"),
    ])
    assert rc == 0, "ldr_reeval: the LDR path changed (metrics row or eval PNGs differ) — see the output above"
