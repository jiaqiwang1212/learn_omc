#!/usr/bin/env python3
"""Phase 1 — Exercise 2: SSA annotation helper.

Parses build/phase1/matmul_torch.mlir (produced by ex1) and prints a
structured annotation of the IR:
  - Every SSA value with its producer op
  - The function signature (argument types, return type)

Run with: uv run python scripts/phase1/ex2_annotate_ssa.py
(Run ex1 first to generate matmul_torch.mlir)
"""
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
MLIR_FILE = REPO_ROOT / "build" / "phase1" / "matmul_torch.mlir"


def parse_func_signature(lines: list[str]) -> tuple[list[str], str]:
    """Extract argument types and return type from func.func line."""
    sig_line = next((l for l in lines if "func.func" in l), "")
    # Arguments: (%arg0: <type>, %arg1: <type>, ...)
    args = re.findall(r"%arg\d+:\s*([\w!<>\[\],\s\.]+?)(?=[,)])", sig_line)
    # Return type: -> <type>
    ret = re.search(r"->\s*([\w!<>\[\],\s\.]+?)\s*\{", sig_line)
    return args, ret.group(1).strip() if ret else "unknown"


def parse_ssa_defs(lines: list[str]) -> list[tuple[str, str, str]]:
    """Extract (ssa_value, op_name, full_line) for each SSA definition."""
    results = []
    for line in lines:
        stripped = line.strip()
        # Match: %name = op.name ...  or  %name:N = op.name ...
        m = re.match(r"(%[\w\d:,\s]+?)\s*=\s*([\w\.]+)(.*)", stripped)
        if m:
            ssa_val = m.group(1).strip()
            op_name = m.group(2).strip()
            results.append((ssa_val, op_name, stripped))
    return results


def main() -> None:
    if not MLIR_FILE.exists():
        print(f"ERROR: {MLIR_FILE} not found.")
        print("Run ex1 first:  uv run python scripts/phase1/ex1_export_torch.py")
        sys.exit(1)

    text = MLIR_FILE.read_text()
    lines = text.splitlines()

    print("=" * 70)
    print(f"SSA ANNOTATION: {MLIR_FILE.name}")
    print("=" * 70)

    # Function signature
    args, ret_type = parse_func_signature(lines)
    print("\n── FUNCTION SIGNATURE ──────────────────────────────────────────────")
    for i, arg in enumerate(args):
        print(f"  %arg{i}  :  {arg.strip()}")
    print(f"  return :  {ret_type}")

    # SSA definitions
    defs = parse_ssa_defs(lines)
    print(f"\n── SSA DEFINITIONS ({len(defs)} values) ─────────────────────────────────")
    for ssa_val, op_name, full_line in defs:
        print(f"\n  VALUE:   {ssa_val}")
        print(f"  DEFINED BY:  {op_name}")
        print(f"  FULL LINE:   {full_line[:100]}")

    # Dialect summary
    ops_seen = {op for _, op, _ in defs}
    # Also grab ops that don't define values (e.g. return)
    for line in lines:
        m = re.match(r"([\w\.]+)\s+", line.strip())
        if m and "." in m.group(1) and not line.strip().startswith("%"):
            ops_seen.add(m.group(1))

    dialects: dict[str, list[str]] = {}
    for op in sorted(ops_seen):
        if "." in op:
            ns, _ = op.split(".", 1)
            dialects.setdefault(ns, []).append(op)

    print("\n── DIALECT SUMMARY ─────────────────────────────────────────────────")
    for ns, ops in sorted(dialects.items()):
        print(f"  {ns}:  {', '.join(ops)}")

    print("\nDone. Compare this with the raw IR in ex1 output.")


if __name__ == "__main__":
    main()
