#!/usr/bin/env python3
"""Phase 5 verification — AC5 milestone check.

Verifies the Phase 5 end-to-end pipeline exercises are complete (AC5):
  - All exercise files exist (dialect header, pass plugins, test input, script)
  - Full pipeline runs: myml dialect → linalg → bufferize → LLVM IR
  - output.ll contains valid LLVM IR (define keyword)
  - llvm-as validates the output

Run with: uv run python scripts/verify_phase5.py
Exit 0 = all checks pass. Non-zero = at least one check failed.
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SCRIPTS_P5 = REPO_ROOT / "scripts" / "phase5"
BUILD_P5 = REPO_ROOT / "build" / "phase5"
BUILD_BIN = REPO_ROOT / "build" / "bin"
MLIR_OPT = BUILD_BIN / "mlir-opt"
MLIR_TRANSLATE = BUILD_BIN / "mlir-translate"
LLVM_AS = BUILD_BIN / "llvm-as"

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = PASS if ok else FAIL
    print(f"  [{'✓' if ok else '✗'} {status}] {name}")
    if not ok:
        if detail:
            print(f"         {detail}")
        failures.append(name)


def skip(name: str, reason: str) -> None:
    print(f"  [{SKIP}] {name} — {reason}")


def run(args: list[str], timeout: int = 60) -> tuple[int, str]:
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return -1, "timeout"
    except Exception as e:
        return -1, str(e)


def find_plugin(name: str) -> "Path | None":
    """Find a plugin by name, checking both .so and .dylib extensions."""
    for ext in (".so", ".dylib"):
        p = REPO_ROOT / "build" / "passes" / "phase5" / f"{name}{ext}"
        if p.exists():
            return p
    return None


# ── Exercise files ───────────────────────────────────────────────────────────
print("\nCheck 1: Phase 5 exercise files exist")
check("passes/phase5/MyMLDialect.h",
      (REPO_ROOT / "passes" / "phase5" / "MyMLDialect.h").exists())
check("passes/phase5/ex1_define_dialect.cpp",
      (REPO_ROOT / "passes" / "phase5" / "ex1_define_dialect.cpp").exists())
check("passes/phase5/ex2_lower_to_linalg.cpp",
      (REPO_ROOT / "passes" / "phase5" / "ex2_lower_to_linalg.cpp").exists())
check("scripts/phase5/test_input.mlir", (SCRIPTS_P5 / "test_input.mlir").exists())
check("scripts/phase5/ex3_end_to_end.sh", (SCRIPTS_P5 / "ex3_end_to_end.sh").exists())

# ── Build binaries ───────────────────────────────────────────────────────────
print("\nCheck 2: MLIR build binaries")
check("build/bin/mlir-opt", MLIR_OPT.exists())
check("build/bin/mlir-translate", MLIR_TRANSLATE.exists())
check("build/bin/llvm-as", LLVM_AS.exists())
if not (MLIR_OPT.exists() and MLIR_TRANSLATE.exists() and LLVM_AS.exists()):
    skip("remaining checks", "run scripts/build.sh first")
    sys.exit(1)

# ── Plugin files ─────────────────────────────────────────────────────────────
print("\nCheck 3: Phase 5 plugin files built")
ex1_plugin = find_plugin("libEx1DefineDialect")
ex2_plugin = find_plugin("libEx2LowerToLinalg")
check("libEx1DefineDialect (.so or .dylib)", ex1_plugin is not None)
check("libEx2LowerToLinalg (.so or .dylib)", ex2_plugin is not None)
if ex1_plugin is None or ex2_plugin is None:
    skip("remaining checks", "run scripts/build_passes.sh first")
    sys.exit(1)

# ── End-to-end pipeline ──────────────────────────────────────────────────────
print("\nCheck 4: End-to-end pipeline (myml → linalg → LLVM IR)")
BUILD_P5.mkdir(parents=True, exist_ok=True)
output_ll = BUILD_P5 / "output.ll"
input_mlir = SCRIPTS_P5 / "test_input.mlir"

pipeline = (
    "builtin.module("
    "func.func(ex2-lower-myml-to-linalg),"
    "one-shot-bufferize{bufferize-function-boundaries=true},"
    "convert-linalg-to-affine-loops,"
    "lower-affine,"
    "convert-scf-to-cf,"
    "convert-cf-to-llvm,"
    "convert-arith-to-llvm,"
    "finalize-memref-to-llvm,"
    "convert-func-to-llvm,"
    "reconcile-unrealized-casts"
    ")"
)
mlir_opt_args = [
    str(MLIR_OPT),
    f"--load-dialect-plugin={ex1_plugin}",  # registers myml dialect before parsing
    f"--load-pass-plugin={ex2_plugin}",     # registers lowering pass
    f"--pass-pipeline={pipeline}",
    str(input_mlir),
]

try:
    opt_result = subprocess.run(
        mlir_opt_args, capture_output=True, text=True, timeout=60,
    )
    opt_ok = opt_result.returncode == 0
    check("mlir-opt pipeline exits 0", opt_ok,
          (opt_result.stdout + opt_result.stderr)[:300] if not opt_ok else "")

    if opt_ok:
        translate_result = subprocess.run(
            [str(MLIR_TRANSLATE), "--mlir-to-llvmir", "-o", str(output_ll)],
            input=opt_result.stdout,
            capture_output=True, text=True, timeout=60,
        )
        translate_ok = translate_result.returncode == 0
        check("mlir-translate --mlir-to-llvmir exits 0", translate_ok,
              (translate_result.stdout + translate_result.stderr)[:300] if not translate_ok else "")
    else:
        skip("mlir-translate", "mlir-opt failed")

except subprocess.TimeoutExpired:
    check("mlir-opt pipeline exits 0", False, "timeout")

# ── Validate output.ll ───────────────────────────────────────────────────────
print("\nCheck 5: Validate output.ll")
if not output_ll.exists():
    skip("output.ll content checks", "output.ll not created")
else:
    ll_content = output_ll.read_text()
    check("output.ll contains 'define'", "define" in ll_content)
    check("output.ll contains test_relu", "test_relu" in ll_content)
    check("output.ll contains test_add", "test_add" in ll_content)

    rc, out = run([str(LLVM_AS), str(output_ll), "-o", "/dev/null"])
    check("llvm-as validates output.ll", rc == 0, out[:200] if rc != 0 else "")

# ── Summary ──────────────────────────────────────────────────────────────────
print()
if failures:
    print(f"FAILED ({len(failures)} check(s)):")
    for f in failures:
        print(f"  - {f}")
    print()
    print("Next steps:")
    print("  1. Build passes: bash scripts/build_passes.sh")
    print("  2. Run pipeline: bash scripts/phase5/ex3_end_to_end.sh")
    print("  3. Re-run:       uv run python scripts/verify_phase5.py")
    sys.exit(1)
else:
    print("All Phase 5 checks passed!")
    print()
    print("AC5 MILESTONE: Custom myml dialect → linalg.generic → LLVM IR.")
    print("  myml.relu and myml.add lower through the full MLIR pipeline.")
    print()
    print("Next: explore TableGen to auto-generate the dialect boilerplate.")
    sys.exit(0)
