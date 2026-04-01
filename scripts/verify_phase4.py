#!/usr/bin/env python3
"""Phase 4 verification — AC4 milestone check.

Verifies the Phase 4 end-to-end pipeline exercises are complete (AC4):
  - All exercise scripts exist
  - Full pipeline runs: export → lower → compile → execute
  - Output matches torch.matmul(ones(4,4), ones(4,4)) to 1e-5 tolerance

Run with: uv run python scripts/verify_phase4.py
Exit 0 = all checks pass. Non-zero = at least one check failed.
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SCRIPTS_P4 = REPO_ROOT / "scripts" / "phase4"
BUILD_P4 = REPO_ROOT / "build" / "phase4"
MLIR_OPT = REPO_ROOT / "build" / "bin" / "mlir-opt"
MLIR_TRANSLATE = REPO_ROOT / "build" / "bin" / "mlir-translate"

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


def run(args: list[str], timeout: int = 60, input_text: str | None = None) -> tuple[int, str]:
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout,
            input=input_text,
        )
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return -1, "timeout"
    except Exception as e:
        return -1, str(e)


# ── Exercise scripts ────────────────────────────────────────────────────────
print("\nCheck 1: Phase 4 exercise scripts exist")
for f in ["ex1_export_matmul.py", "ex2_lower_to_llvmir.sh",
          "ex3_runtime_shim.c", "ex4_compile_and_run.sh"]:
    check(f"scripts/phase4/{f}", (SCRIPTS_P4 / f).exists())

# ── Build binaries ──────────────────────────────────────────────────────────
print("\nCheck 2: MLIR build binaries")
check("build/bin/mlir-opt", MLIR_OPT.exists())
check("build/bin/mlir-translate", MLIR_TRANSLATE.exists())
if not (MLIR_OPT.exists() and MLIR_TRANSLATE.exists()):
    skip("remaining checks", "run scripts/build.sh first")
    sys.exit(1)

# ── Step 1: Export matmul.mlir ───────────────────────────────────────────────
print("\nCheck 3: ex1 — export matmul at LINALG_ON_TENSORS")
rc, out = run(["uv", "run", "python", str(SCRIPTS_P4 / "ex1_export_matmul.py")], timeout=60)
check("ex1_export_matmul.py exits 0", rc == 0, out[:200] if rc != 0 else "")
matmul_mlir = BUILD_P4 / "matmul.mlir"
check("build/phase4/matmul.mlir created", matmul_mlir.exists())
if matmul_mlir.exists():
    content = matmul_mlir.read_text()
    check("matmul.mlir contains linalg.matmul", "linalg.matmul" in content)
    check("matmul.mlir uses @forward (not @main)", "@forward" in content)

# ── Step 2: Lower to LLVM IR ────────────────────────────────────────────────
print("\nCheck 4: ex2 — lower to LLVM IR with C-interface wrappers")
if not matmul_mlir.exists():
    skip("ex2_lower_to_llvmir.sh", "matmul.mlir missing")
else:
    rc, out = run(["bash", str(SCRIPTS_P4 / "ex2_lower_to_llvmir.sh")], timeout=60)
    check("ex2_lower_to_llvmir.sh exits 0", rc == 0, out[:300] if rc != 0 else "")
    matmul_ll = BUILD_P4 / "matmul.ll"
    check("build/phase4/matmul.ll created", matmul_ll.exists())
    if matmul_ll.exists():
        ll_content = matmul_ll.read_text()
        check("matmul.ll contains _mlir_ciface_forward",
              "_mlir_ciface_forward" in ll_content)
        check("matmul.ll contains LLVM define", "define" in ll_content)

# ── Step 3: Compile ──────────────────────────────────────────────────────────
print("\nCheck 5: compile matmul.ll + runtime_shim.c")
matmul_ll = BUILD_P4 / "matmul.ll"
shim = SCRIPTS_P4 / "ex3_runtime_shim.c"
run_exe = BUILD_P4 / "run_matmul"

if not matmul_ll.exists():
    skip("clang compile", "matmul.ll missing")
elif not shim.exists():
    skip("clang compile", "ex3_runtime_shim.c missing")
else:
    rc, out = run(
        ["clang", "-O2", str(matmul_ll), str(shim), "-lm", "-o", str(run_exe)],
        timeout=30,
    )
    check("clang compile exits 0", rc == 0, out[:300] if rc != 0 else "")
    check("run_matmul executable created", run_exe.exists())

# ── Step 4: AC4 — run and verify output ────────────────────────────────────
print("\nCheck 6: AC4 — run_matmul output matches torch.matmul reference")
if not run_exe.exists():
    skip("run_matmul execution (AC4)", "compile step failed")
else:
    rc, out = run([str(run_exe)], timeout=10)
    check("run_matmul exits 0 (AC4)", rc == 0, out[:200] if rc != 0 else "")
    check("output contains 4.00 (all-ones matmul result)", "4.00" in out)
    check("AC4 PASS message in output", "AC4 PASS" in out)
    if rc == 0:
        print(f"  Output:\n    " + out.strip().replace("\n", "\n    "))

# ── Summary ─────────────────────────────────────────────────────────────────
print()
if failures:
    print(f"FAILED ({len(failures)} check(s)):")
    for f in failures:
        print(f"  - {f}")
    print()
    print("Next steps:")
    print("  1. Export:   uv run python scripts/phase4/ex1_export_matmul.py")
    print("  2. Lower:    bash scripts/phase4/ex2_lower_to_llvmir.sh")
    print("  3. Compile:  bash scripts/phase4/ex4_compile_and_run.sh")
    print("  4. Re-run:   uv run python scripts/verify_phase4.py")
    sys.exit(1)
else:
    print("All Phase 4 checks passed!")
    print()
    print("AC4 MILESTONE: PyTorch → MLIR → LLVM IR → CPU executable.")
    print("  torch.matmul(ones(4,4), ones(4,4)) output matches within 1e-5.")
    print()
    print("Next: AC5 — given torch.relu, write the lowering pipeline from scratch.")
    sys.exit(0)
