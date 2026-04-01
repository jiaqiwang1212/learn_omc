<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-01 | Updated: 2026-04-01 -->

# scripts/phase1/

## Purpose
IR literacy exercises. The goal (AC1) is to read unseen torch-mlir output — identifying all ops, SSA values, and function signature types — without documentation, within 5 minutes. Exercises cover exporting PyTorch models to torch-mlir IR, annotating SSA def-use chains, analyzing op types, and understanding output type representations.

## Key Files
| File | Description |
|------|-------------|
| `ex1_export_torch.py` | Export a simple PyTorch model to torch-mlir IR and inspect the output |
| `ex2_annotate_ssa.py` | Walk the IR and annotate every SSA value with its def and uses |
| `ex3_def_use_chain.py` | Trace def-use chains for a target value through the IR graph |
| `ex4_compare_output_types.py` | Compare ranked tensor types vs unranked, static vs dynamic shapes |
| `ex5_relu_linalg_generic.py` | Export a relu and observe how it maps to `linalg.generic` |

## For AI Agents

### Working In This Directory
- Run any exercise: `uv run python scripts/phase1/exN_<name>.py`
- Exercises use `torch-mlir` Python bindings; the package must be available via `uv sync`
- No build step required (pure Python)

### Testing Requirements
- `uv run python scripts/verify_phase1.py` from repo root — must exit 0

### Common Patterns
- Use `torch_mlir.compile(model, example_inputs, output_type=...)` to get IR text
- SSA values are `%name` tokens; every use traces back to a unique def
- `linalg.generic` encodes elementwise ops via `affine_map` indexing maps

## Dependencies

### Internal
- `scripts/verify_phase1.py` — acceptance verifier

### External
- `torch` — PyTorch
- `torch_mlir` — torch-mlir Python bindings

<!-- MANUAL: -->
