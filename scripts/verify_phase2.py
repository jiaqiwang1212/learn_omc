#!/usr/bin/env python3
"""Phase 2 verification — AC2 milestone check.

Verifies that the Phase 2 pass plugin exercises are complete (AC2):
  - All C++ source files exist
  - CMakeLists.txt exists
  - build_passes.sh exists
  - Plugin .dylib/.so files exist in build/passes/phase2/
  - mlir-opt can load each plugin and the pass flag appears in --help
  - Ex2 counter pass actually counts matmul ops (output check)
  - Ex3 rewrite pass produces a func.call insertion (output check)

Run with: uv run python scripts/verify_phase2.py
Exit 0 = all checks pass. Non-zero = at least one check failed.
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
PASSES_SRC = REPO_ROOT / "passes" / "phase2"
PASSES_BUILD = REPO_ROOT / "build" / "passes" / "phase2"
MLIR_OPT = REPO_ROOT / "build" / "bin" / "mlir-opt"
TEST_INPUT = REPO_ROOT / "scripts" / "phase2" / "test_input.mlir"

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


def find_plugin(name: str) -> Path | None:
    """Find the built plugin (.so for MODULE libs on both macOS and Linux,
    falling back to .dylib for legacy SHARED builds on macOS)."""
    for ext in ["so", "dylib"]:
        p = PASSES_BUILD / f"lib{name}.{ext}"
        if p.exists():
            return p
    return None


# ── Source files ──────────────────────────────────────────────────────────────
print("\nCheck 1: C++ source files exist")
for f in ["ex1_noop_pass.cpp", "ex2_counter_pass.cpp", "ex3_rewrite_pass.cpp"]:
    check(f"passes/phase2/{f}", (PASSES_SRC / f).exists())

# ── CMake + build script ──────────────────────────────────────────────────────
print("\nCheck 2: CMake scaffolding exists")
check("passes/CMakeLists.txt", (REPO_ROOT / "passes" / "CMakeLists.txt").exists())
check("passes/phase2/CMakeLists.txt", (PASSES_SRC / "CMakeLists.txt").exists())
check("scripts/build_passes.sh", (REPO_ROOT / "scripts" / "build_passes.sh").exists())

# ── Test input ────────────────────────────────────────────────────────────────
print("\nCheck 3: Test input MLIR file exists")
check("scripts/phase2/test_input.mlir", TEST_INPUT.exists())
if TEST_INPUT.exists():
    content = TEST_INPUT.read_text()
    check("test_input.mlir contains linalg.matmul", "linalg.matmul" in content)

# ── Built plugins ─────────────────────────────────────────────────────────────
print("\nCheck 4: Plugin shared libraries built")
plugins = {}
for name in ["Ex1NoopPass", "Ex2CounterPass", "Ex3RewritePass"]:
    path = find_plugin(name)
    plugins[name] = path
    if path:
        check(f"lib{name}.dylib/.so exists", True)
    else:
        check(f"lib{name}.dylib/.so exists", False,
              f"Run: bash scripts/build_passes.sh")

# ── AC2: mlir-opt can load plugins ───────────────────────────────────────────
# NOTE: Dynamically loaded plugin passes cannot register CLI flags (e.g.
# --ex1-noop) because PassNameParser::initialize() snapshots the pass
# registry at startup, before --load-pass-plugin fires.  Instead we verify
# that the pass name appears in --help (via the PassPipelineCLParser) and
# that --pass-pipeline can resolve the pass name at runtime.
print("\nCheck 5: AC2 — mlir-opt loads each plugin (pass registered)")
if not MLIR_OPT.exists():
    skip("mlir-opt exists", "build/bin/mlir-opt not found — run scripts/build.sh first")
else:
    for pass_name, plugin_name, flag in [
        ("Ex1NoopPass", "Ex1NoopPass", "ex1-noop"),
        ("Ex2CounterPass", "Ex2CounterPass", "ex2-counter"),
        ("Ex3RewritePass", "Ex3RewritePass", "ex3-rewrite"),
    ]:
        plugin = plugins.get(plugin_name)
        if not plugin:
            skip(f"{flag} loads in mlir-opt", f"plugin not built yet")
            continue
        try:
            result = subprocess.run(
                [str(MLIR_OPT), f"--load-pass-plugin={plugin}", "--help"],
                capture_output=True, text=True, timeout=10
            )
            combined = result.stdout + result.stderr
            ok = flag in combined
            check(f"{flag} appears in mlir-opt --help", ok,
                  f"Plugin loaded but pass not registered — check mlirGetPassPluginInfo()")
        except subprocess.TimeoutExpired:
            check(f"{flag} mlir-opt --help", False, "timeout")
        except Exception as e:
            check(f"{flag} mlir-opt --help", False, str(e))

# ── AC2: Ex2 actually counts matmul ──────────────────────────────────────────
print("\nCheck 6: AC2 — Ex2 counter pass finds 1 linalg.matmul in test_input.mlir")
plugin = plugins.get("Ex2CounterPass")
if not plugin or not TEST_INPUT.exists():
    skip("Ex2 count check", "plugin or test_input.mlir not found")
else:
    try:
        result = subprocess.run(
            [str(MLIR_OPT), f"--load-pass-plugin={plugin}",
             "--pass-pipeline=builtin.module(func.func(ex2-counter))",
             str(TEST_INPUT)],
            capture_output=True, text=True, timeout=10
        )
        combined = result.stdout + result.stderr
        found = "1 linalg.matmul" in combined or "Found 1" in combined
        check("Ex2 reports 1 linalg.matmul op", found,
              "Either the pass hasn't been implemented yet, or the count output format doesn't match.\n"
              f"         Actual output: {combined[:200]}")
    except Exception as e:
        check("Ex2 count output", False, str(e))

# ── AC2: Ex3 inserts func.call ────────────────────────────────────────────────
print("\nCheck 7: AC2 — Ex3 rewrite pass inserts func.call @matmul_hook")
plugin = plugins.get("Ex3RewritePass")
if not plugin or not TEST_INPUT.exists():
    skip("Ex3 rewrite check", "plugin or test_input.mlir not found")
else:
    try:
        result = subprocess.run(
            [str(MLIR_OPT), f"--load-pass-plugin={plugin}",
             "--pass-pipeline=builtin.module(func.func(ex3-rewrite))",
             "--verify-each", str(TEST_INPUT)],
            capture_output=True, text=True, timeout=10
        )
        combined = result.stdout + result.stderr
        has_call = "matmul_hook" in combined or "func.call" in combined
        no_error = result.returncode == 0 or "error:" not in combined.lower()
        check("Ex3 output contains func.call @matmul_hook", has_call,
              "Either the pass hasn't been implemented yet, or the insertion failed.\n"
              f"         Return code: {result.returncode}")
        check("Ex3 --verify-each passes (no verifier errors)", no_error,
              f"Verifier error in output:\n{combined[:300]}")
    except Exception as e:
        check("Ex3 rewrite output", False, str(e))

# ── Summary ───────────────────────────────────────────────────────────────────
print()
if failures:
    print(f"FAILED ({len(failures)} check(s)):")
    for f in failures:
        print(f"  - {f}")
    print()
    print("Next steps:")
    print("  1. Build plugins:     bash scripts/build_passes.sh")
    print("  2. Implement TODOs in passes/phase2/ex1_noop_pass.cpp")
    print("                         passes/phase2/ex2_counter_pass.cpp")
    print("                         passes/phase2/ex3_rewrite_pass.cpp")
    print("  3. Re-run:            uv run python scripts/verify_phase2.py")
    sys.exit(1)
else:
    print("All Phase 2 checks passed!")
    print()
    print("AC2 MILESTONE: You can build, load, and run a custom MLIR C++ pass")
    print("as a plugin via mlir-opt --load-pass-plugin.")
    print()
    print("Next: Phase 3 — The Lowering Chain (dialects end-to-end).")
    sys.exit(0)
