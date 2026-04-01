#!/usr/bin/env python3
"""Phase 1 — Exercise 4: Compare TORCH vs LINALG_ON_TENSORS output types.

Exports the same 4x4 matmul at both output types, saves both to
build/phase1/, and prints a comparison showing:
  - Ops present in TORCH but gone in LINALG  (high-level → lowered away)
  - Ops present in LINALG but new in TORCH   (new lower-level ops)
  - Types that changed (vtensor → tensor)

This illustrates what the 'convert-torch-to-linalg' pass does.

Run with: uv run python scripts/phase1/ex4_compare_output_types.py
"""
import re
from pathlib import Path

import torch
import torch_mlir.fx as tmfx

REPO_ROOT = Path(__file__).parent.parent.parent
OUT_DIR = REPO_ROOT / "build" / "phase1"


class Matmul(torch.nn.Module):
    def forward(self, a, b):
        return torch.matmul(a, b)


def extract_ops(asm: str) -> set[str]:
    """Extract all dialect op names from IR text."""
    ops = set()
    for line in asm.splitlines():
        # Match ops that define values: %x = op.name
        m = re.search(r"=\s*([\w]+\.[\w\.]+)\s", line)
        if m:
            ops.add(m.group(1))
        # Match ops that don't define values: op.name ...
        m2 = re.match(r"\s+([\w]+\.[\w\.]+)\s", line)
        if m2:
            ops.add(m2.group(1))
    return ops


def extract_types(asm: str) -> set[str]:
    """Extract unique type strings from IR text."""
    return set(re.findall(r"![\w\.<>\[\],]+", asm))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mod_torch = tmfx.export_and_import(
        Matmul(),
        torch.ones(4, 4),
        torch.ones(4, 4),
        output_type=tmfx.OutputType.TORCH,
    )
    asm_torch = mod_torch.operation.get_asm()

    mod_linalg = tmfx.export_and_import(
        Matmul(),
        torch.ones(4, 4),
        torch.ones(4, 4),
        output_type=tmfx.OutputType.LINALG_ON_TENSORS,
    )
    asm_linalg = mod_linalg.operation.get_asm()

    (OUT_DIR / "matmul_torch.mlir").write_text(asm_torch)
    (OUT_DIR / "matmul_linalg.mlir").write_text(asm_linalg)
    print(f"Saved: {OUT_DIR}/matmul_torch.mlir")
    print(f"Saved: {OUT_DIR}/matmul_linalg.mlir\n")

    ops_torch = extract_ops(asm_torch)
    ops_linalg = extract_ops(asm_linalg)
    types_torch = extract_types(asm_torch)
    types_linalg = extract_types(asm_linalg)

    print("=" * 70)
    print("OUTPUT TYPE COMPARISON: TORCH  vs  LINALG_ON_TENSORS")
    print("=" * 70)

    removed = ops_torch - ops_linalg
    added = ops_linalg - ops_torch
    kept = ops_torch & ops_linalg

    print("\n── OPS REMOVED (torch dialect → lowered away) ───────────────────────")
    if removed:
        for op in sorted(removed):
            print(f"  - {op}")
    else:
        print("  (none)")

    print("\n── OPS ADDED (new lower-level ops) ──────────────────────────────────")
    if added:
        for op in sorted(added):
            print(f"  + {op}")
    else:
        print("  (none)")

    print("\n── OPS KEPT (present in both) ───────────────────────────────────────")
    for op in sorted(kept):
        print(f"    {op}")

    removed_types = types_torch - types_linalg
    added_types = types_linalg - types_torch
    print("\n── TYPES REMOVED ────────────────────────────────────────────────────")
    for t in sorted(removed_types)[:10]:
        print(f"  - {t}")

    print("\n── TYPES ADDED ──────────────────────────────────────────────────────")
    for t in sorted(added_types)[:10]:
        print(f"  + {t}")

    print("""
── WHAT THIS MEANS ──────────────────────────────────────────────────────
The 'convert-torch-to-linalg' pass (run internally by torch-mlir when you
select LINALG_ON_TENSORS) replaced torch dialect ops with linalg dialect ops:

  torch.aten.matmul  →  linalg.matmul
  !torch.vtensor     →  tensor<...>  (builtin tensor type, no vtensor wrapper)

The linalg dialect has structured semantics — named loops and indexing maps —
that enable downstream optimizations (vectorization, tiling, etc.)
This is why lowering to linalg is the first step in every torch-mlir pipeline.
""")


if __name__ == "__main__":
    main()
