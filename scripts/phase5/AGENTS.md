<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-01 | Updated: 2026-04-01 -->

# scripts/phase5/

## Purpose
End-to-end exercise and test input for the custom `myml` dialect (Phase 5 / AC5). The shell script drives the full lowering pipeline from `myml` IR through `linalg → affine → llvm` to a CPU-runnable binary, validating that the custom dialect integrates correctly with the MLIR lowering infrastructure.

## Key Files
| File | Description |
|------|-------------|
| `test_input.mlir` | MLIR module written in the `myml` dialect — input to the full lowering pipeline |
| `ex3_end_to_end.sh` | Full pipeline: load `myml` dialect plugin → lower to `linalg` → lower to LLVM IR → compile and run |

## For AI Agents

### Working In This Directory
- Run end-to-end test: `bash scripts/phase5/ex3_end_to_end.sh`
- The script loads both `libEx1DefineDialect.dylib` and `libEx2LowerToLinalg.dylib` from `build/passes/phase5/`
- Both pass plugins must be built before running: `bash scripts/build_passes.sh`

### Testing Requirements
- `uv run python scripts/verify_phase5.py` — must exit 0 (AC5)
- The pipeline must produce valid LLVM IR and run without error

### Common Patterns
- `test_input.mlir` uses `myml.matmul` op syntax defined in `passes/phase5/MyMLDialect.h`
- Load order matters: dialect plugin first, then lowering plugin
- After `myml → linalg` lowering, the standard Phase 3 pipeline applies

## Dependencies

### Internal
- `passes/phase5/` — `Ex1DefineDialect` and `Ex2LowerToLinalg` plugins
- `scripts/verify_phase5.py` — verifier
- `build/bin/mlir-opt`, `build/bin/llc` — required tools

<!-- MANUAL: -->
