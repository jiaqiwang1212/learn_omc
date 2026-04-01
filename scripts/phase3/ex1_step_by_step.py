#!/usr/bin/env python3
"""Phase 3 — Exercise 1: Step-by-step lowering from linalg to LLVM.

Applies each lowering pass one at a time, saving intermediate .mlir files
to build/phase3/. Each step uses --verify-each so errors surface early.

Run with: uv run python scripts/phase3/ex1_step_by_step.py
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
OUT_DIR = REPO_ROOT / "build" / "phase3"
MLIR_OPT = REPO_ROOT / "build" / "bin" / "mlir-opt"
INPUT_FILE = REPO_ROOT / "scripts" / "phase2" / "test_input.mlir"

STEPS = [
    {
        "flags": ['--one-shot-bufferize=bufferize-function-boundaries'],
        "output": "step1_bufferized.mlir",
        "desc": "tensors replaced by memrefs; memref.alloc inserted",
    },
    {
        "flags": ["--convert-linalg-to-affine-loops"],
        "output": "step2_affine.mlir",
        "desc": "linalg loops expanded to affine.for loop nests",
    },
    {
        "flags": ["--lower-affine"],
        "output": "step3_lowered_affine.mlir",
        "desc": "affine.for → scf.for; indexing expressions inlined",
    },
    {
        "flags": ["--convert-scf-to-cf"],
        "output": "step4_cf.mlir",
        "desc": "scf.for → cf.br basic block branches",
    },
    {
        "flags": ["--convert-cf-to-llvm", "--convert-arith-to-llvm"],
        "output": "step5_llvm_partial.mlir",
        "desc": "arith + cf ops lowered to llvm dialect equivalents",
    },
    {
        "flags": [
            "--finalize-memref-to-llvm",
            "--convert-func-to-llvm",
            "--reconcile-unrealized-casts",
        ],
        "output": "step6_llvm_full.mlir",
        "desc": "memref descriptors and func signatures lowered; unrealized casts resolved",
    },
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    prev_file = INPUT_FILE
    for i, step in enumerate(STEPS, start=1):
        out_file = OUT_DIR / step["output"]
        cmd = [
            str(MLIR_OPT),
            str(prev_file),
            *step["flags"],
            "--verify-each",
            "-o",
            str(out_file),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Step {i} FAILED: {' '.join(cmd)}")
            print(result.stderr)
            sys.exit(1)
        print(f"Step {i} → {step['output']}: {step['desc']}")
        prev_file = out_file

    print()
    print("All 6 steps completed. Files saved to build/phase3/")
    print("""
READING GUIDE:
  Each step saves one .mlir file in build/phase3/.
  Open them side-by-side and compare. Key questions:
  EXERCISE:
    1. In step1_bufferized.mlir, find every 'memref.alloc'.
       For each alloc, which tensor from the original did it come from?
    2. In step2_affine.mlir, find the 'affine.for' loops.
       How many loops are there? What do the bounds correspond to?
    3. Compare step5 and step6: what disappeared in the final step?
    4. After step6, could you feed this to mlir-translate? Try it:
       build/bin/mlir-translate --mlir-to-llvmir build/phase3/step6_llvm_full.mlir
""")


if __name__ == "__main__":
    main()
