#!/usr/bin/env python3
"""Phase 0 verification — confirms mlir-opt and torch-mlir are working.
Run with: uv run python scripts/verify_phase0.py
Exit code 0 = all checks pass. Non-zero = at least one check failed.
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
MLIR_OPT = REPO_ROOT / "build" / "bin" / "mlir-opt"
MLIR_TRANSLATE = REPO_ROOT / "build" / "bin" / "mlir-translate"
LLVM_AS = REPO_ROOT / "build" / "bin" / "llvm-as"

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

failures = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = PASS if ok else FAIL
    print(f"  [{status}] {name}")
    if not ok:
        if detail:
            print(f"         {detail}")
        failures.append(name)


# ── Check 1: mlir-opt binary exists ──────────────────────────────────────────
print("\nCheck 1: mlir-opt binary")
check("binary exists", MLIR_OPT.exists(), f"Expected at {MLIR_OPT}")

if MLIR_OPT.exists():
    result = subprocess.run([str(MLIR_OPT), "--version"], capture_output=True, text=True)
    check("--version exits 0", result.returncode == 0, result.stderr.strip())
    version_line = (result.stdout + result.stderr).splitlines()[0] if result.returncode == 0 else ""
    print(f"         {version_line}")

# ── Check 2: mlir-opt round-trip on a minimal .mlir file ─────────────────────
print("\nCheck 2: mlir-opt round-trip (arith.addf)")
SIMPLE_MLIR = """\
func.func @add(%a: f32, %b: f32) -> f32 {
  %c = arith.addf %a, %b : f32
  return %c : f32
}
"""
if MLIR_OPT.exists():
    result = subprocess.run(
        [str(MLIR_OPT), "--verify-each", "-"],
        input=SIMPLE_MLIR,
        capture_output=True,
        text=True,
    )
    ok = result.returncode == 0 and "func.func" in result.stdout
    check("round-trip succeeds", ok, result.stderr.strip()[:200] if not ok else "")
    check("output contains func.func", "func.func" in result.stdout)
else:
    check("round-trip succeeds", False, "mlir-opt not found — skipped")
    check("output contains func.func", False, "skipped")

# ── Check 3: torch-mlir Python module importable ─────────────────────────────
print("\nCheck 3: torch-mlir Python bindings")
try:
    import torch_mlir  # type: ignore[import]
    check("import torch_mlir", True)
    print(f"         path: {torch_mlir.__path__}")
except ImportError as e:
    check("import torch_mlir", False, str(e))

try:
    import torch  # type: ignore[import]
    check("import torch", True)
    print(f"         torch {torch.__version__}")
except ImportError as e:
    check("import torch", False, str(e))

# ── Check 4: torch-mlir compile a simple module ───────────────────────────────
# Note: torch_mlir.compile was removed; the current API is torch_mlir.fx.export_and_import
print("\nCheck 4: torch-mlir export_and_import (relu → TORCH dialect)")
try:
    import torch  # type: ignore[import]
    import torch_mlir.fx as tmfx  # type: ignore[import]
    import torch_mlir.ir as tmir  # type: ignore[import]

    class Relu(torch.nn.Module):
        def forward(self, x):
            return torch.relu(x)

    with tmir.Context():
        module = tmfx.export_and_import(Relu(), torch.ones(4), output_type=tmfx.OutputType.TORCH)
        asm = module.operation.get_asm()

    check("export_and_import succeeds", True)
    check("output contains torch.aten.relu", "torch.aten.relu" in asm,
          "unexpected ASM (first 300 chars): " + asm[:300])
    print(f"         IR preview: {asm.splitlines()[1].strip()[:80]}")
except Exception as e:
    check("export_and_import succeeds", False, str(e)[:300])
    check("output contains torch.aten.relu", False, "skipped")

# ── Check 5: Additional LLVM tools (on-demand, same build) ───────────────────
print("\nCheck 5: Additional LLVM tools (build/bin/)")
for tool, target in [(MLIR_TRANSLATE, "mlir-translate"), (LLVM_AS, "llvm-as")]:
    if tool.exists():
        check(f"{tool.name} built", True)
    else:
        print(f"  [INFO] {tool.name} not yet built — run when needed:")
        print(f"         cmake --build build --target {target}")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
if failures:
    print(f"FAILED ({len(failures)} check(s)):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("All Phase 0 checks passed. Ready for Phase 1.")
    sys.exit(0)
