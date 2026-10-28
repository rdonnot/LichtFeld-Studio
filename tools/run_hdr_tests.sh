#!/usr/bin/env bash
# tools/run_hdr_tests.sh — every HDR-fork test from one command, one summary at the end (PLAN.md M6).
#
#   tools/run_hdr_tests.sh [--no-gpu] [--quick]
#
# Sections (each PASS / FAIL / SKIP with its reason; the exit code is 1 when any section FAILS, never for a SKIP):
#   tools      pytest in tools/ (eval + synth): hdr_metrics, log_metrics, ldr_regression, make_brackets, and
#              test_hdr_ply.py (needs the built binary → skipped without it) and test_ldr_reeval.py (needs the
#              binary AND the frozen LDR reference out/baseline_a → skipped without them; FAILS when the
#              re-evaluation is not bit-identical).
#   plugin     pytest of plugins/hdr_brackets/tests (pure Python; runs in the tools/ environment, the same
#              dependency set — the plugin's own .venv is LichtFeld's, created at load time).
#   ctest      the HDR gtests in the C++ test tree (`ctest -L hdr`): hdr_display_chain, hdr_eval_split,
#              hdr_parameters (CPU) and hdr_color_chain, hdr_bracket_stage (GPU; excluded with --no-gpu or when
#              no CUDA device is visible). Skipped when the test tree is not built.
#
# Environment:
#   LFS_BUILD_DIR        the app build (default <repo>/build) — supplies LFS_BINARY when that is unset
#   LFS_BINARY           the LichtFeld-Studio binary for the PLY round-trip and the LDR re-evaluation
#   LFS_TESTS_BUILD_DIR  the C++ test tree (default <repo>/build/tests; built with -DBUILD_TESTS=ON, CLAUDE.md)
#   LFS_LDR_REFERENCE    the frozen M0 reference run (default <repo>/out/baseline_a)
#   LFS_HDR_NO_GPU=1     same as --no-gpu
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${LFS_BUILD_DIR:-$REPO/build}"
TESTS_DIR="${LFS_TESTS_BUILD_DIR:-$REPO/build/tests}"
export LFS_BINARY="${LFS_BINARY:-$BUILD_DIR/LichtFeld-Studio}"
export LFS_LDR_REFERENCE="${LFS_LDR_REFERENCE:-$REPO/out/baseline_a}"
NO_GPU="${LFS_HDR_NO_GPU:-0}"
QUICK=0
for a in "$@"; do
    case "$a" in
        --no-gpu) NO_GPU=1 ;;
        --quick) QUICK=1 ;;
        -h|--help) sed -n 2,25p "$0"; exit 0 ;;
        *) echo "unknown option: $a" >&2; exit 2 ;;
    esac
done
if [ "$NO_GPU" = 0 ] && ! (command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1); then
    NO_GPU=1
    echo "no CUDA device visible: GPU-labelled tests are excluded"
fi
if [ "$QUICK" = 1 ]; then
    export LFS_HDR_SKIP_SLOW=1   # test_ldr_reeval.py honours it (a 7000-iteration project re-evaluation)
fi

names=(); results=(); notes=()
record() { names+=("$1"); results+=("$2"); notes+=("$3"); }
banner() { printf '\n══ %s ══════════════════════════════════════════════════\n' "$1"; }

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required (https://docs.astral.sh/uv/)" >&2
    exit 2
fi

# ── tools/ ────────────────────────────────────────────────────────────────────────────────────────────────────
banner "tools/ pytest (eval + synth)"
note=""
[ -x "$LFS_BINARY" ] || note="no binary at $LFS_BINARY → test_hdr_ply + test_ldr_reeval skipped"
[ -d "$LFS_LDR_REFERENCE" ] || note="${note:+$note; }no LDR reference at $LFS_LDR_REFERENCE → test_ldr_reeval skipped"
if (cd "$REPO/tools" && uv run --quiet python -m pytest -q -rs); then
    record "tools pytest" PASS "$note"
else
    record "tools pytest" FAIL "$note"
fi

# ── plugin ─────────────────────────────────────────────────────────────────────────────────────────────────────
banner "plugins/hdr_brackets pytest"
if (cd "$REPO/plugins/hdr_brackets" && uv run --quiet --project "$REPO/tools" python -m pytest -q -rs tests); then
    record "plugin pytest" PASS ""
else
    record "plugin pytest" FAIL ""
fi

# ── ctest ──────────────────────────────────────────────────────────────────────────────────────────────────────
banner "ctest -L hdr ($TESTS_DIR)"
if [ -f "$TESTS_DIR/CTestTestfile.cmake" ]; then
    label_args=(-L hdr)
    if [ "$NO_GPU" = 1 ]; then
        label_args+=(-LE gpu)
        gpu_note="GPU tests (hdr_color_chain, hdr_bracket_stage) excluded"
    else
        gpu_note=""
    fi
    if ctest --test-dir "$TESTS_DIR" "${label_args[@]}" --output-on-failure --no-tests=error; then
        record "ctest -L hdr" PASS "$gpu_note"
    else
        record "ctest -L hdr" FAIL "$gpu_note"
    fi
else
    record "ctest -L hdr" SKIP "no C++ test tree at $TESTS_DIR (cmake -DBUILD_TESTS=ON, CLAUDE.md 'Build & run')"
fi

# ── summary ────────────────────────────────────────────────────────────────────────────────────────────────────
banner "HDR tests — summary"
rc=0
for i in "${!names[@]}"; do
    printf '  %-16s %-5s %s\n' "${names[$i]}" "${results[$i]}" "${notes[$i]}"
    [ "${results[$i]}" = FAIL ] && rc=1
done
[ $rc = 0 ] && echo "  all sections passed or were skipped with a reason" || echo "  FAILURES above"
exit $rc
