#!/usr/bin/env python3
"""Phase 4 — Exercise 1: Export 4x4 matmul at LINALG_ON_TENSORS.

Produces build/phase4/matmul.mlir — the starting point for the full
end-to-end pipeline in Phase 4.

Run with: uv run python scripts/phase4/ex1_export_matmul.py
"""
from pathlib import Path

import torch
import torch_mlir.fx as tmfx

REPO_ROOT = Path(__file__).parent.parent.parent
OUT_DIR = REPO_ROOT / "build" / "phase4"


class Matmul(torch.nn.Module):
    def forward(self, a, b):
        return torch.matmul(a, b)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    module = tmfx.export_and_import(
        Matmul(),
        torch.ones(4, 4),
        torch.ones(4, 4),
        output_type=tmfx.OutputType.LINALG_ON_TENSORS,
    )
    asm = module.operation.get_asm()
    # torch-mlir names the entry point @main; rename to @forward so it doesn't
    # conflict with C's main() when linking the runtime shim in Phase 4.
    asm = asm.replace("func.func @main(", "func.func @forward(")

    out_file = OUT_DIR / "matmul.mlir"
    out_file.write_text(asm)
    print(f"Saved: {out_file}")
    print()
    print(asm)


if __name__ == "__main__":
    main()
