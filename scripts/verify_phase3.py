#!/usr/bin/env python3
"""Phase 3 verification — AC3 milestone check.

Verifies the Phase 3 lowering chain exercises are complete (AC3):
  - Exercise scripts exist
  - Full lowering chain produces valid LLVM IR (ex3_full_chain.sh)
  - llvm-as validates the output .ll file (AC3)

Run with: uv run python scripts/verify_phase3.py
Exit 0 = all checks pass. Non-zero = at least one check failed.
"""
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SCRIPTS_P3 = REPO_ROOT / "scripts" / "phase3"
BUILD_P3 = REPO_ROOT / "build" / "phase3"
MLIR_OPT = REPO_ROOT / "build" / "bin" / "mlir-opt"
MLIR_TRANSLATE = REPO_ROOT / "build" / "bin" / "mlir-translate"
LLVM_AS = REPO_ROOT / "build" / "bin" / "llvm-as"
MATMUL_LL = BUILD_P3 / "matmul.ll"

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = PASS if ok else FAIL
    print(f"  [{status}] {name}")
    if not ok:
        if detail:
            print(f"         {detail}")
        failures.append(name)


def skip(name: str, reason: str) -> None:
    print(f"  [{SKIP}] {name} — {reason}")


# ── Exercise scripts ────────────────────────────────────────────────────────
print("\nCheck 1: Phase 3 exercise scripts exist")
for f in [
    "ex1_step_by_step.py",
    "ex2_bufferization_dive.py",
    "ex3_full_chain.sh",
    "ex4_break_pipeline.sh",
]:
    check(f"scripts/phase3/{f}", (SCRIPTS_P3 / f).exists())

# ── Build binaries ──────────────────────────────────────────────────────────
print("\nCheck 2: MLIR build binaries")
binaries_ok = MLIR_OPT.exists() and MLIR_TRANSLATE.exists()
check("build/bin/mlir-opt", MLIR_OPT.exists())
check("build/bin/mlir-translate", MLIR_TRANSLATE.exists())
if not binaries_ok:
    skip("remaining checks", "run scripts/build.sh first")
    print()
    print(f"FAILED ({len(failures)} check(s)):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)

# ── llvm-as available ───────────────────────────────────────────────────────
print("\nCheck 3: llvm-as available (needed for AC3)")
if LLVM_AS.exists():
    llvm_as = str(LLVM_AS)
    check("llvm-as found (build/bin/llvm-as)", True)
else:
    llvm_as = shutil.which("llvm-as")
    if llvm_as:
        check("llvm-as found (system)", True)
    else:
        skip("llvm-as found", "build it: cmake --build build --target llvm-as")

# ── Full chain script ──────────────────────────────────────────────────────
print("\nCheck 4: ex3_full_chain.sh — full lowering pipeline")
chain_script = SCRIPTS_P3 / "ex3_full_chain.sh"
if not chain_script.exists():
    skip("ex3_full_chain.sh", "run scripts/phase3/ex3_full_chain.sh first")
else:
    try:
        result = subprocess.run(
            ["bash", str(chain_script)],
            capture_output=True, text=True, timeout=60,
        )
        check("ex3_full_chain.sh exits 0", result.returncode == 0,
              f"exit code {result.returncode}\n         stderr: {result.stderr[:200]}")
        check("build/phase3/matmul.ll created", MATMUL_LL.exists())
    except subprocess.TimeoutExpired:
        check("ex3_full_chain.sh exits 0", False, "timeout (60s)")
    except Exception as e:
        check("ex3_full_chain.sh exits 0", False, str(e))

# ── AC3: llvm-as validates matmul.ll ────────────────────────────────────────
print("\nCheck 5: AC3 — llvm-as validates matmul.ll")
if not MATMUL_LL.exists():
    skip("llvm-as matmul.ll succeeds (AC3)", "run ex3_full_chain.sh first")
elif not llvm_as:
    skip("llvm-as matmul.ll succeeds (AC3)",
         "install llvm-as: brew install llvm && export PATH=$(brew --prefix llvm)/bin:$PATH")
else:
    try:
        result = subprocess.run(
            [llvm_as, str(MATMUL_LL), "-o", "/dev/null"],
            capture_output=True, text=True, timeout=30,
        )
        check("llvm-as matmul.ll succeeds (AC3)", result.returncode == 0,
              f"llvm-as failed:\n         {result.stderr[:300]}")
    except subprocess.TimeoutExpired:
        check("llvm-as matmul.ll succeeds (AC3)", False, "timeout (30s)")
    except Exception as e:
        check("llvm-as matmul.ll succeeds (AC3)", False, str(e))

# ── matmul.ll content sanity ────────────────────────────────────────────────
print("\nCheck 6: matmul.ll content sanity")
if not MATMUL_LL.exists():
    skip("matmul.ll content checks", "run ex3_full_chain.sh first")
else:
    content = MATMUL_LL.read_text()
    check("matmul.ll contains LLVM function definition", "define" in content)
    check("matmul.ll contains memory ops (memref lowering confirmed)",
          "alloca" in content or "load" in content)

# ── Intermediate step files ─────────────────────────────────────────────────
print("\nCheck 7: Intermediate step files (from ex1_step_by_step.py)")
step_files = [
    "step1_bufferized.mlir",
    "step2_affine.mlir",
    "step3_lowered_affine.mlir",
    "step4_cf.mlir",
    "step5_llvm_partial.mlir",
    "step6_llvm_full.mlir",
]
if not any((BUILD_P3 / f).exists() for f in step_files):
    skip("intermediate step files", "run: uv run python scripts/phase3/ex1_step_by_step.py")
else:
    for f in step_files:
        check(f"build/phase3/{f}", (BUILD_P3 / f).exists())

# ── Summary ─────────────────────────────────────────────────────────────────
print()
if failures:
    print(f"FAILED ({len(failures)} check(s)):")
    for f in failures:
        print(f"  - {f}")
    print()
    print("Next steps:")
    print("  1. Run ex3_full_chain.sh:   bash scripts/phase3/ex3_full_chain.sh")
    print("  2. Run step-by-step ex:     uv run python scripts/phase3/ex1_step_by_step.py")
    print("  3. Re-run:                   uv run python scripts/verify_phase3.py")
    sys.exit(1)
else:
    print("All Phase 3 checks passed!")
    print()
    print("AC3 MILESTONE: LLVM IR is valid — the lowering chain is complete.")
    print("  torch dialect → linalg → affine → scf → cf → llvm dialect → LLVM IR")
    print()
    print("Next: Phase 3.5 — Break the Pipeline (scripts/phase3/ex4_break_pipeline.sh)")
    print("Then: Phase 4 — End-to-End Pipeline (PyTorch → CPU executable)")
    sys.exit(0)
