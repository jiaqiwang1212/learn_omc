#!/usr/bin/env python3
"""Phase 1 — Exercise 3: SSA def-use chain tracer.

Uses the torch-mlir Python IR bindings to walk the MLIR module and trace
the def-use chain for the matmul result value:
  - Which op *defines* the matmul result?
  - Which ops *use* it?
  - What would need to change to insert a new op between matmul and return?

This exercise directly prepares you for writing RewritePattern code in Phase 2.

Run with: uv run python scripts/phase1/ex3_def_use_chain.py
"""
import re

import torch
import torch_mlir.fx as tmfx


def find_matmul_op(module):
    """Walk all ops in the module and return the first matmul-like op."""
    for op in module.body.operations:
        # op is a func.func
        for region in op.regions:
            for block in region.blocks:
                for inner_op in block.operations:
                    name = inner_op.operation.name
                    if "matmul" in name or ".mm" in name:
                        return inner_op.operation
    return None


def main() -> None:
    # Re-export so we have live IR objects (not just text)
    class Matmul(torch.nn.Module):
        def forward(self, a, b):
            return torch.matmul(a, b)

    module = tmfx.export_and_import(
        Matmul(),
        torch.ones(4, 4),
        torch.ones(4, 4),
        output_type=tmfx.OutputType.TORCH,
    )

    matmul_op = find_matmul_op(module)

    print("=" * 70)
    print("SSA DEF-USE CHAIN TRACE: matmul result")
    print("=" * 70)

    if matmul_op is None:
        print("WARNING: Could not find matmul op via IR walk.")
        print("Falling back to text-based analysis.\n")
        asm = module.operation.get_asm()
        _text_fallback(asm)
        return

    # The op that defines the matmul result
    print(f"\n── DEFINES ──────────────────────────────────────────────────────────")
    print(f"  Op name  : {matmul_op.name}")
    print(f"  Num results: {len(matmul_op.results)}")
    for i, result in enumerate(matmul_op.results):
        print(f"  Result[{i}]  : type = {result.type}")

        # Uses of this result
        uses = list(result.uses)
        print(f"\n── USES of result[{i}] ({len(uses)} use(s)) ──────────────────────────────")
        if not uses:
            print("  (no uses — result is dead code)")
        for use in uses:
            user_op = use.owner
            print(f"  Used by op : {user_op.name}")
            print(f"  Operand #  : {use.operand_number}")

    print("""
── HOW TO INSERT A NEW OP BETWEEN MATMUL AND RETURN ─────────────────────
In Phase 2, when writing a RewritePattern, you would:

  1. Match the matmul op (e.g. via OpRewritePattern<torch::AtenMatmulOp>)
  2. Get its result:  Value matmulResult = op.getResult();
  3. Create the new op AFTER the matmul, consuming matmulResult:
       Value newResult = rewriter.create<MyNewOp>(loc, matmulResult);
  4. Replace all uses of matmulResult with newResult:
       rewriter.replaceAllUsesWith(matmulResult, newResult);
     or replace the matmul op itself:
       rewriter.replaceOpWithNewOp<MyNewOp>(op, ...);

Key rule (SSA): you never edit matmulResult in place — you create a new
SSA value and redirect uses to it. Violating this produces a verifier
error: "value defined multiple times".
""")


def _text_fallback(asm: str) -> None:
    """Simple text-based def-use analysis when IR walk is unavailable."""
    lines = asm.splitlines()

    matmul_result = None
    for line in lines:
        if "matmul" in line or ".mm" in line:
            m = re.match(r"\s*(%[\w\d]+)\s*=", line)
            if m:
                matmul_result = m.group(1)
                print(f"  Matmul op line : {line.strip()}")
                print(f"  Result SSA val : {matmul_result}")
                break

    if matmul_result:
        print(f"\nUses of {matmul_result}:")
        for line in lines:
            if matmul_result in line and "=" not in line.split(matmul_result)[0].replace(" ", ""):
                continue
            if matmul_result in line and "matmul" not in line and ".mm" not in line:
                print(f"  {line.strip()}")
    else:
        print("Could not locate matmul result SSA value in IR text.")


if __name__ == "__main__":
    main()
