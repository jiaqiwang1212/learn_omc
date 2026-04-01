#!/usr/bin/env python3
"""Phase 1 — Exercise 5: Relu at LINALG_ON_TENSORS — reading linalg.generic.

Exports torch.relu at LINALG_ON_TENSORS. Unlike matmul (which lowers to
linalg.matmul — a named op), most elementwise ops lower to linalg.generic
with explicit indexing maps describing the loop structure.

This script:
  1. Exports and saves relu_linalg.mlir
  2. Extracts the linalg.generic op
  3. Annotates its indexing maps with a human-readable explanation

Run with: uv run python scripts/phase1/ex5_relu_linalg_generic.py
"""
import re
from pathlib import Path

import torch
import torch_mlir.fx as tmfx

REPO_ROOT = Path(__file__).parent.parent.parent
OUT_DIR = REPO_ROOT / "build" / "phase1"


class Relu(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x)


def extract_linalg_generic_blocks(asm: str) -> list[str]:
    """Extract linalg.generic op blocks from IR text (simple brace matching)."""
    blocks = []
    lines = asm.splitlines()
    i = 0
    while i < len(lines):
        if "linalg.generic" in lines[i]:
            # Collect until closing } at same indent level
            depth = 0
            block_lines = []
            while i < len(lines):
                line = lines[i]
                block_lines.append(line)
                depth += line.count("{") - line.count("}")
                i += 1
                if depth <= 0 and len(block_lines) > 1:
                    break
            blocks.append("\n".join(block_lines))
        else:
            i += 1
    return blocks


def explain_indexing_map(map_str: str) -> str:
    """Give a plain-english explanation of an affine_map string."""
    # Common patterns
    if "d0" in map_str and "d1" in map_str:
        return "both input and output use (d0, d1) → elementwise over all dims"
    if "d0" in map_str and "d1" not in map_str:
        return "reduces over d1 (only d0 in output) → reduction along dim 1"
    return f"indexing map: {map_str}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    module = tmfx.export_and_import(
        Relu(),
        torch.ones(4, 4),
        output_type=tmfx.OutputType.LINALG_ON_TENSORS,
    )
    asm = module.operation.get_asm()

    out_file = OUT_DIR / "relu_linalg.mlir"
    out_file.write_text(asm)
    print(f"Saved: {out_file}\n")

    print("=" * 70)
    print("RELU at LINALG_ON_TENSORS")
    print("=" * 70)
    print(asm)
    print("=" * 70)

    # Locate linalg.generic blocks
    generic_blocks = extract_linalg_generic_blocks(asm)

    print(f"\n── FOUND {len(generic_blocks)} linalg.generic op(s) ─────────────────────────────")

    for idx, block in enumerate(generic_blocks):
        print(f"\n[linalg.generic #{idx + 1}]")
        print(block[:600])

        # Extract indexing_maps
        maps_match = re.search(r"indexing_maps\s*=\s*\[([^\]]+)\]", block)
        if maps_match:
            maps_raw = maps_match.group(1)
            individual_maps = re.findall(r"affine_map<([^>]+)>", maps_raw)
            print(f"\n  Indexing maps ({len(individual_maps)} total):")
            for i, m in enumerate(individual_maps):
                role = "input" if i < len(individual_maps) - 1 else "output"
                explanation = explain_indexing_map(m)
                print(f"    [{role}] affine_map<{m}>")
                print(f"           → {explanation}")

        # Extract iterator_types
        iters_match = re.search(r"iterator_types\s*=\s*\[([^\]]+)\]", block)
        if iters_match:
            iters = iters_match.group(1)
            print(f"\n  Iterator types: {iters}")
            if "parallel" in iters and "reduction" not in iters:
                print("    → all parallel = pure elementwise op (no reduction)")
            elif "reduction" in iters:
                print("    → has reduction dimension(s) = reduces along those axes")

    print("""
── CONTRAST WITH MATMUL ─────────────────────────────────────────────────
  linalg.matmul   — a *named* op with implicit semantics (C += A * B)
  linalg.generic  — an *explicit* op: you specify loops (iterator_types)
                    and indexing into each operand (indexing_maps)

Most ops that don't have a named linalg counterpart lower to linalg.generic.
When writing a pass in Phase 2, you may need to match linalg.generic and
inspect its indexing maps to understand what computation it performs.

KEY INSIGHT FOR AC5:
  relu(x) has all-parallel iterators + identity indexing map on both
  input and output → pure pointwise (every element independent).
  This pattern is the signature of elementwise ops in linalg.generic.
""")


if __name__ == "__main__":
    main()
