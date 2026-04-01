#!/bin/bash
# Phase 5 — Exercise 3: End-to-End Pipeline with Custom Dialect
#
# Lowers test_input.mlir (containing myml ops) all the way to LLVM IR
# by chaining the custom dialect plugins with the Phase 3 lowering passes.
#
# Pass ordering (and why each step is needed):
#   --ex1-load-myml               → registers myml dialect so mlir-opt can parse test_input.mlir
#   --ex2-lower-myml-to-linalg    → converts myml.relu/add → linalg.generic (our custom lowering)
#   --one-shot-bufferize          → tensors → memrefs (physical memory allocation)
#   --convert-linalg-to-affine-loops → explicit loop nests
#   --lower-affine                → affine → standard control flow
#   --convert-scf-to-cf           → structured control flow → basic blocks
#   --convert-cf-to-llvm          → control flow ops → LLVM dialect
#   --convert-arith-to-llvm       → arith ops → LLVM dialect
#   --finalize-memref-to-llvm     → memref ops → LLVM dialect
#   --convert-func-to-llvm        → func.func → llvm.func
#   --reconcile-unrealized-casts  → resolve any remaining type cast placeholders
#
# IMPORTANT: --ex1-load-myml MUST come before --ex2-lower-myml-to-linalg.
# The dialect must be registered (by ex1) before any pass can read myml ops.

set -e
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MLIR_OPT="$REPO_ROOT/build/bin/mlir-opt"
MLIR_TRANSLATE="$REPO_ROOT/build/bin/mlir-translate"
LLVM_AS="$REPO_ROOT/build/bin/llvm-as"

EX1_PLUGIN="$REPO_ROOT/build/passes/phase5/libEx1DefineDialect.so"
EX2_PLUGIN="$REPO_ROOT/build/passes/phase5/libEx2LowerToLinalg.so"

# On macOS, plugins may have .dylib extension
if [ ! -f "$EX1_PLUGIN" ]; then
  EX1_PLUGIN="${EX1_PLUGIN%.so}.dylib"
fi
if [ ! -f "$EX2_PLUGIN" ]; then
  EX2_PLUGIN="${EX2_PLUGIN%.so}.dylib"
fi

INPUT="$REPO_ROOT/scripts/phase5/test_input.mlir"
OUTPUT_DIR="$REPO_ROOT/build/phase5"
OUTPUT_LL="$OUTPUT_DIR/output.ll"

mkdir -p "$OUTPUT_DIR"

echo "==> Lowering $INPUT → $OUTPUT_LL"

# LLVM 23 note: plugin passes must be invoked via --pass-pipeline, not --passname.
# Built-in passes (one-shot-bufferize etc.) still accept both styles, but mixing
# plugin passes requires the pipeline string format.
#
# Two-step loading:
#   --load-dialect-plugin  → registers myml dialect BEFORE input parsing
#   --load-pass-plugin     → registers ex2-lower-myml-to-linalg BEFORE pipeline build
"$MLIR_OPT" \
  --load-dialect-plugin="$EX1_PLUGIN" \
  --load-pass-plugin="$EX2_PLUGIN" \
  --pass-pipeline="builtin.module(
    func.func(ex2-lower-myml-to-linalg),
    one-shot-bufferize{bufferize-function-boundaries=true},
    convert-linalg-to-affine-loops,
    lower-affine,
    convert-scf-to-cf,
    convert-cf-to-llvm,
    convert-arith-to-llvm,
    finalize-memref-to-llvm,
    convert-func-to-llvm,
    reconcile-unrealized-casts
  )" \
  "$INPUT" | \
"$MLIR_TRANSLATE" --mlir-to-llvmir -o "$OUTPUT_LL"

echo "==> Validating LLVM IR with llvm-as..."
"$LLVM_AS" "$OUTPUT_LL" -o /dev/null

echo "==> Success! LLVM IR written to $OUTPUT_LL"
echo "    View it: cat $OUTPUT_LL"
