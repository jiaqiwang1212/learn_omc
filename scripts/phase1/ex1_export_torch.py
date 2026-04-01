#!/usr/bin/env python3
"""Phase 1 — Exercise 1: Export matmul IR at TORCH output type.

Exports a 4x4 float32 matmul module using torch-mlir and prints the
TORCH dialect IR. Saves the IR to build/phase1/matmul_torch.mlir.

Run with: uv run python scripts/phase1/ex1_export_torch.py
"""
from pathlib import Path

import torch
import torch_mlir.fx as tmfx

REPO_ROOT = Path(__file__).parent.parent.parent
OUT_DIR = REPO_ROOT / "build" / "phase1"
OUT_FILE = OUT_DIR / "matmul_torch.mlir"


class Matmul(torch.nn.Module):
    def forward(self, a, b):
        return torch.matmul(a, b)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    module = tmfx.export_and_import(
        Matmul(),
        torch.ones(4, 4),
        torch.ones(4, 4),
        output_type=tmfx.OutputType.TORCH,
    )
    asm = module.operation.get_asm()

    OUT_FILE.write_text(asm)
    print(f"Saved to: {OUT_FILE}\n")

    print("=" * 70)
    print("TORCH dialect IR for matmul(a: 4x4xf32, b: 4x4xf32):")
    print("=" * 70)
    print(asm)
    print("=" * 70)

    # Reading hints
    print("""
READING GUIDE:
  - Lines starting with '%' define SSA values — each is produced exactly once
  - 'func.func @forward(...)' — the function; note the argument types in (...)
  - 'torch.aten.mm' or 'torch.aten.matmul' — the matmul op in the torch dialect
  - '!torch.vtensor<[4,4],f32>' — a torch virtual tensor type (4x4, float32)
  - '%arg0', '%arg1' — function arguments (SSA values defined by the function sig)

EXERCISE:
  For each '%N = <op>' line, note:
    1. What SSA value is being defined (%N)?
    2. What op is producing it?
    3. What are the operands (inputs to the op)?
  This is the def-use graph — the foundation of pass writing in Phase 2.
""")


if __name__ == "__main__":
    main()
