<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-01 | Updated: 2026-04-01 -->

# passes/phase5/

## Purpose
Implements a custom `myml` MLIR dialect and a lowering pass that converts `myml` ops to `linalg`. This is the capstone phase: designing a novel dialect, registering it, and writing a complete lowering pipeline so that `myml` ops flow through to CPU-executable LLVM IR.

## Key Files
| File | Description |
|------|-------------|
| `CMakeLists.txt` | Defines two plugin targets: `Ex1DefineDialect` and `Ex2LowerToLinalg` |
| `MyMLDialect.h` | Dialect declaration: op definitions, type system, and dialect registration for `myml` |
| `ex1_define_dialect.cpp` | Registers the `myml` dialect so `mlir-opt` can parse `myml.*` ops |
| `ex2_lower_to_linalg.cpp` | Conversion pass: lowers `myml` ops to equivalent `linalg` operations |

## For AI Agents

### Working In This Directory
- Build with `bash scripts/build_passes.sh` from repo root; outputs to `build/passes/phase5/`
- End-to-end test: `bash scripts/phase5/ex3_end_to_end.sh`
- Test input MLIR: `scripts/phase5/test_input.mlir`

### Testing Requirements
- `uv run python scripts/verify_phase5.py` — must exit 0 (AC5)
- The full lowering chain `myml → linalg → affine → llvm` must produce valid LLVM IR

### Common Patterns
- Dialect defined via `MyMLDialect` class inheriting `mlir::Dialect`
- Ops defined with `mlir::Op<>` CRTP template + ODS-style field declarations
- Lowering pass uses `mlir::ConversionTarget` + `mlir::RewritePatternSet` + `mlir::applyPartialConversion`
- `MyMLDialect.h` is the single source of truth for op signatures — keep it in sync with the `.cpp` lowering patterns

## Dependencies

### Internal
- `passes/CMakeLists.txt` — parent CMake
- `scripts/phase5/test_input.mlir` — `myml` dialect test input
- `scripts/phase5/ex3_end_to_end.sh` — full pipeline driver

### External
- MLIR dialect and conversion headers from `third_party/torch-mlir/externals/llvm-project/mlir/include/`

<!-- MANUAL: -->
