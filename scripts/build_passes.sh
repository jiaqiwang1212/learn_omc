#!/bin/bash
# Build Phase 2 MLIR pass plugins
# Run from: /Users/jacob/workspace/learn_omc
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PASSES_SRC="$REPO_ROOT/passes"
PASSES_BUILD="$REPO_ROOT/build/passes"
MLIR_DIR="$REPO_ROOT/build/lib/cmake/mlir"

echo "==> Configuring passes CMake project..."
mkdir -p "$PASSES_BUILD"
cd "$PASSES_BUILD"

cmake -GNinja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DMLIR_DIR="$MLIR_DIR" \
  "$PASSES_SRC"

echo "==> Building pass plugins..."
ninja -j$(sysctl -n hw.logicalcpu) Ex1NoopPass Ex2CounterPass Ex3RewritePass

echo ""
echo "==> Built plugins:"
ls -la "$PASSES_BUILD/phase2/"*.dylib "$PASSES_BUILD/phase2/"*.so 2>/dev/null || \
  ls -la "$PASSES_BUILD/phase2/"lib*.* 2>/dev/null || true
echo ""
echo "==> Test with:"
MLIR_OPT="$REPO_ROOT/build/bin/mlir-opt"
PLUGIN_DIR="$PASSES_BUILD/phase2"
echo "  $MLIR_OPT --load-pass-plugin=$PLUGIN_DIR/libEx1NoopPass.dylib --ex1-noop $REPO_ROOT/scripts/phase2/test_input.mlir"
echo "  $MLIR_OPT --load-pass-plugin=$PLUGIN_DIR/libEx2CounterPass.dylib --ex2-counter $REPO_ROOT/scripts/phase2/test_input.mlir"
echo "  $MLIR_OPT --load-pass-plugin=$PLUGIN_DIR/libEx3RewritePass.dylib --ex3-rewrite --verify-each $REPO_ROOT/scripts/phase2/test_input.mlir"
