<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-01 | Updated: 2026-04-01 -->

# scripts/phase2/

## Purpose
Contains the MLIR test input file used by all three Phase 2 pass plugins. The input encodes a `linalg.matmul` operation that the noop, counter, and rewrite passes operate on.

## Key Files
| File | Description |
|------|-------------|
| `test_input.mlir` | MLIR module containing a `linalg.matmul` — used as input to all Phase 2 pass invocations |

## For AI Agents

### Working In This Directory
- This directory is input-only; do not add generated files here
- To run a pass against this input:
  ```bash
  build/bin/mlir-opt --load-pass-plugin=build/passes/phase2/libEx3RewritePass.dylib \
                     --ex3-rewrite scripts/phase2/test_input.mlir
  ```

### Testing Requirements
- `uv run python scripts/verify_phase2.py` — passes load the file automatically

### Common Patterns
- MLIR text format: `func.func @name(%arg: tensor<...>) -> tensor<...> { ... }`
- `linalg.matmul` takes two input tensors and one output tensor via `ins`/`outs` syntax

## Dependencies

### Internal
- `passes/phase2/` — pass plugins that consume this file
- `scripts/verify_phase2.py` — verifier

<!-- MANUAL: -->
