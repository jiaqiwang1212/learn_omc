<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-01 | Updated: 2026-04-01 -->

# scripts/phase3/

## Purpose
Lowering chain exercises. The goal (AC3) is to lower a PyTorch matmul through the full MLIR dialect stack (`torch → linalg → affine/scf → llvm`) and produce valid LLVM IR verified by `llvm-as`. Also includes a "break the pipeline" exercise (Phase 3.5) that teaches diagnostic reasoning by intentionally injecting pipeline failures.

## Key Files
| File | Description |
|------|-------------|
| `ex1_step_by_step.py` | Lower matmul one dialect step at a time; print IR after each pass |
| `ex2_bufferization_dive.py` | Deep dive into the bufferization step — tensor-to-memref conversion |
| `ex3_full_chain.sh` | Full `mlir-opt` pipeline from torch IR to LLVM IR; pipe to `llvm-as` for validation |
| `ex4_break_pipeline.sh` | Intentionally broken pipeline invocations — diagnose and fix each failure |

## For AI Agents

### Working In This Directory
- Shell scripts reference `build/bin/mlir-opt` and `build/bin/llvm-as` — build must exist
- Python exercises: `uv run python scripts/phase3/exN_<name>.py`
- Shell exercises: `bash scripts/phase3/ex3_full_chain.sh`

### Testing Requirements
- `uv run python scripts/verify_phase3.py` — validates the full lowering chain produces valid LLVM IR
- `uv run python scripts/verify_phase3_5.py` — validates break-the-pipeline diagnostic reasoning

### Common Patterns
- Lowering order: `--convert-torch-to-linalg` → `--linalg-bufferize` → `--convert-linalg-to-affine-loops` → `--lower-affine` → `--convert-scf-to-cf` → `--convert-cf-to-llvm` → `--convert-func-to-llvm` → `--finalize-memref-to-llvm`
- Validate LLVM IR: pipe `mlir-opt ... --mlir-print-ir-after-all` output through `mlir-translate --mlir-to-llvmir | llvm-as`
- Each broken pipeline has a specific error message pattern to recognize

## Dependencies

### Internal
- `scripts/verify_phase3.py`, `scripts/verify_phase3_5.py` — verifiers
- `build/bin/mlir-opt`, `build/bin/llvm-as`, `build/bin/mlir-translate` — required tools

### External
- `torch_mlir` Python bindings (for Python exercises)

<!-- MANUAL: -->
