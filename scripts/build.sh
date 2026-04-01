#!/bin/bash
# Build torch-mlir (+ mlir-opt) from source
# Run from: /Users/jacob/workspace/learn_omc
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TORCH_MLIR_SRC="$REPO_ROOT/third_party/torch-mlir"
BUILD_DIR="$REPO_ROOT/build"
PYTHON_EXE="$REPO_ROOT/.venv/bin/python3"
PYTHON_ROOT="$("$PYTHON_EXE" -c "import sys; print(sys.exec_prefix)")"
PYTHON_INC="$("$PYTHON_EXE" -c "import sysconfig; print(sysconfig.get_path('include'))")"
PYTHON_LIB="$("$PYTHON_EXE" -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")/libpython3.11.dylib"

echo "==> Configuring cmake..."
echo "    Python root: $PYTHON_ROOT"
echo "    Python exe:  $PYTHON_EXE"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DPython3_ROOT_DIR="$PYTHON_ROOT" \
  -DPython_ROOT_DIR="$PYTHON_ROOT" \
  -DPython3_FIND_VIRTUALENV=ONLY \
  -DPython_FIND_VIRTUALENV=ONLY \
  -DPython3_FIND_STRATEGY=LOCATION \
  -DPython3_FIND_REGISTRY=NEVER \
  -DPython3_FIND_FRAMEWORK=NEVER \
  -DPython3_EXECUTABLE:FILEPATH="$PYTHON_EXE" \
  -DPython3_INCLUDE_DIR:PATH="$PYTHON_INC" \
  -DPython3_LIBRARY:FILEPATH="$PYTHON_LIB" \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS="torch-mlir" \
  -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="$TORCH_MLIR_SRC" \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_ZLIB=OFF \
  -DLLVM_ENABLE_ZSTD=OFF \
  "$TORCH_MLIR_SRC/externals/llvm-project/llvm"

echo "==> Building (this takes 45-90 min)..."
ninja -j$(sysctl -n hw.logicalcpu) \
  mlir-opt \
  mlir-translate \
  torch-mlir-opt \
  TorchMLIRPythonModules

echo "==> Done. Binaries in: $BUILD_DIR/bin/"
echo "    mlir-opt:          $BUILD_DIR/bin/mlir-opt"
echo "    torch-mlir-opt:    $BUILD_DIR/bin/torch-mlir-opt"
