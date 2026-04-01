#!/usr/bin/env bash
# Phase 4 — Exercise 4: Compile LLVM IR + runtime shim, run, verify output.
#
# Compiles build/phase4/matmul.ll with scripts/phase4/ex3_runtime_shim.c
# using system clang, then runs the executable.
#
# Usage: bash scripts/phase4/ex4_compile_and_run.sh

set -euo pipefail
trap 'echo "FAILED at line $LINENO — check the error above"; exit 1' ERR

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_P4="$REPO_ROOT/build/phase4"
SHIM="$REPO_ROOT/scripts/phase4/ex3_runtime_shim.c"
MATMUL_LL="$BUILD_P4/matmul.ll"
OUT_EXE="$BUILD_P4/run_matmul"

# --- Preflight checks ---
if [[ ! -f "$MATMUL_LL" ]]; then
  echo "ERROR: $MATMUL_LL not found"
  echo "Run first: bash scripts/phase4/ex2_lower_to_llvmir.sh"
  exit 1
fi
if [[ ! -f "$SHIM" ]]; then
  echo "ERROR: $SHIM not found"
  exit 1
fi

# --- Compile ---
echo "Compiling: clang -O2 matmul.ll ex3_runtime_shim.c -lm -o run_matmul"
clang -O2 "$MATMUL_LL" "$SHIM" -lm -o "$OUT_EXE"
echo "Compiled → $OUT_EXE"
echo ""

# --- Run ---
echo "Running: $OUT_EXE"
echo "---"
"$OUT_EXE"
