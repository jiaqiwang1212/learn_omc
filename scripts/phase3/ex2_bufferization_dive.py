#!/usr/bin/env python3
"""Phase 3 — Exercise 2: Deep dive into bufferization.

Runs one-shot-bufferize on the linalg matmul input and highlights every
memref.alloc that appeared.  Compares tensor vs memref worlds.

Run with: uv run python scripts/phase3/ex2_bufferization_dive.py
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
OUT_DIR = REPO_ROOT / "build" / "phase3"
MLIR_OPT = REPO_ROOT / "build" / "bin" / "mlir-opt"
INPUT_FILE = REPO_ROOT / "scripts" / "phase2" / "test_input.mlir"
OUT_FILE = OUT_DIR / "step1_bufferized.mlir"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Read the original input
    input_text = INPUT_FILE.read_text()

    # Run bufferization
    cmd = [
        str(MLIR_OPT),
        str(INPUT_FILE),
        '--one-shot-bufferize=bufferize-function-boundaries',
        "--verify-each",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Bufferization FAILED:")
        print(result.stderr)
        sys.exit(1)

    output_text = result.stdout
    OUT_FILE.write_text(output_text)
    print(f"Saved bufferized IR to: {OUT_FILE}\n")

    # Show original
    print("=" * 70)
    print("ORIGINAL (tensor world):")
    print("=" * 70)
    print(input_text)

    # Show bufferized
    print("=" * 70)
    print("BUFFERIZED (memref world):")
    print("=" * 70)
    print(output_text)

    # Find memref.alloc lines with context
    print("=" * 70)
    print("memref.alloc insertions:")
    print("=" * 70)
    lines = output_text.splitlines()
    alloc_count = 0
    for i, line in enumerate(lines):
        if "memref.alloc" in line:
            alloc_count += 1
            start = max(0, i - 2)
            end = min(len(lines), i + 3)
            for j in range(start, end):
                marker = " >>>" if j == i else "    "
                print(f"{marker} {lines[j]}")
            print()

    print(f"Found {alloc_count} memref.alloc insertion(s)")
    print("""
READING GUIDE:
  Bufferization converts 'tensor' types (functional, no memory address) to
  'memref' types (physical memory with pointer, shape, strides).

  ML analogy: going from torch.Tensor (logical view, no address) to
  numpy.ndarray (physical memory layout with dtype, strides, and a pointer).

  EXERCISE:
    For each memref.alloc you see above:
      1. Which tensor operand did it come from?
         (Hint: look at the original linalg.matmul ins/outs)
      2. Why is an alloc needed here — what is the memory for?
      3. The 'outs' tensor in linalg.matmul becomes an alloc too.
         Why? (Hint: linalg needs somewhere to write results)

    Compare the function signature:
      Before: (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
      After:  what does it look like?

    Key insight: after bufferization, the function signature changes.
    The return type is gone — results are written into pre-allocated buffers.
""")


if __name__ == "__main__":
    main()
