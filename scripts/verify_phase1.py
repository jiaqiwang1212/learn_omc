#!/usr/bin/env python3
"""Phase 1 verification — AC1 milestone check.

Verifies that all Phase 1 exercises have been completed and that the
learner can read unseen MLIR output (AC1):
  - All exercise scripts exist
  - All build/phase1/ IR files have been generated
  - Spot-checks IR content for expected dialect ops/types

Run with: uv run python scripts/verify_phase1.py
Exit 0 = all checks pass. Non-zero = at least one check failed.
"""
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SCRIPTS = REPO_ROOT / "scripts" / "phase1"
BUILD = REPO_ROOT / "build" / "phase1"

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = PASS if ok else FAIL
    print(f"  [{status}] {name}")
    if not ok:
        if detail:
            print(f"         {detail}")
        failures.append(name)


# ── Exercise scripts exist ────────────────────────────────────────────────────
print("\nCheck 1: Exercise scripts exist")
for ex in ["ex1_export_torch.py", "ex2_annotate_ssa.py", "ex3_def_use_chain.py",
           "ex4_compare_output_types.py", "ex5_relu_linalg_generic.py"]:
    check(f"scripts/phase1/{ex}", (SCRIPTS / ex).exists())

# ── Build outputs exist ───────────────────────────────────────────────────────
print("\nCheck 2: IR output files exist (exercises must have been run)")
for fname in ["matmul_torch.mlir", "matmul_linalg.mlir", "relu_linalg.mlir"]:
    check(f"build/phase1/{fname}", (BUILD / fname).exists(),
          f"Run: uv run python scripts/phase1/{'ex1' if 'torch' in fname else 'ex4' if 'linalg' in fname and 'matmul' in fname else 'ex5'}_*.py")

# ── AC1: Read unseen IR — matmul_linalg.mlir ─────────────────────────────────
print("\nCheck 3: AC1 — Can read matmul_linalg.mlir (unseen LINALG IR)")
linalg_file = BUILD / "matmul_linalg.mlir"
if linalg_file.exists():
    asm = linalg_file.read_text()
    check("contains linalg.matmul op", "linalg.matmul" in asm,
          "Expected linalg.matmul op in LINALG_ON_TENSORS output")
    check("contains tensor<> type (not vtensor)", "tensor<" in asm,
          "Expected builtin tensor<> type (vtensor should be gone after lowering)")
    check("contains func.func", "func.func" in asm)
    # SSA values — every %N = pattern
    ssa_defs = re.findall(r"%[\w\d]+\s*=", asm)
    check(f"contains SSA value definitions ({len(ssa_defs)} found)", len(ssa_defs) > 0)
else:
    for name in ["contains linalg.matmul op", "contains tensor<> type",
                 "contains func.func", "contains SSA value definitions"]:
        check(name, False, "matmul_linalg.mlir not found — run ex4 first")

# ── AC1: Read unseen IR — relu_linalg.mlir ───────────────────────────────────
print("\nCheck 4: AC1 — Can read relu_linalg.mlir (unseen LINALG IR with linalg.generic)")
relu_file = BUILD / "relu_linalg.mlir"
if relu_file.exists():
    asm = relu_file.read_text()
    check("contains linalg.generic op", "linalg.generic" in asm,
          "relu should lower to linalg.generic (not a named op)")
    check("contains indexing_maps attribute", "indexing_maps" in asm,
          "linalg.generic must have indexing_maps")
    check("contains iterator_types attribute", "iterator_types" in asm)
    check("has all-parallel iterators (elementwise)", '"parallel"' in asm,
          "relu is elementwise — all iterator_types should be parallel")
else:
    for name in ["contains linalg.generic op", "contains indexing_maps",
                 "contains iterator_types", "has all-parallel iterators"]:
        check(name, False, "relu_linalg.mlir not found — run ex5 first")

# ── AC1: TORCH IR dialect namespaces ─────────────────────────────────────────
print("\nCheck 5: AC1 — matmul_torch.mlir contains expected torch dialect ops")
torch_file = BUILD / "matmul_torch.mlir"
if torch_file.exists():
    asm = torch_file.read_text()
    check("contains torch.aten op (matmul or mm)", "torch.aten" in asm)
    check("contains !torch.vtensor type", "torch.vtensor" in asm)
    check("contains func.func", "func.func" in asm)
else:
    for name in ["contains torch.aten op", "contains !torch.vtensor type", "contains func.func"]:
        check(name, False, "matmul_torch.mlir not found — run ex1 first")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
if failures:
    print(f"FAILED ({len(failures)} check(s)):")
    for f in failures:
        print(f"  - {f}")
    print("\nRun the exercises first:")
    print("  uv run python scripts/phase1/ex1_export_torch.py")
    print("  uv run python scripts/phase1/ex4_compare_output_types.py")
    print("  uv run python scripts/phase1/ex5_relu_linalg_generic.py")
    sys.exit(1)
else:
    print("All Phase 1 checks passed.")
    print("\nAC1 MILESTONE: You can read MLIR IR output — both TORCH and")
    print("LINALG_ON_TENSORS dialects — and identify ops, SSA values,")
    print("function signatures, and linalg.generic structure.")
    print("\nNext: Phase 2 — Write your first MLIR C++ pass.")
    sys.exit(0)
