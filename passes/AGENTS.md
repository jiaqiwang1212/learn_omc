<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-01 | Updated: 2026-04-01 -->

# passes/

## Purpose
C++ source code for MLIR pass plugins, organized by curriculum phase. Each plugin is compiled to a shared library (`.dylib` on macOS) and loaded at runtime via `mlir-opt --load-pass-plugin`. Covers writing noop/counter/rewrite passes (Phase 2) and defining a custom `myml` dialect with a lowering pass to `linalg` (Phase 5).

## Key Files
| File | Description |
|------|-------------|
| `CMakeLists.txt` | Top-level CMake that adds `phase2` and `phase5` subdirectories |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `phase2/` | Three introductory C++ passes: noop, op-counter, rewrite (see `phase2/AGENTS.md`) |
| `phase5/` | Custom `myml` dialect definition and lowering pass to `linalg` (see `phase5/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- Build all passes with `bash scripts/build_passes.sh` from the repo root
- Output dylibs land in `build/passes/phaseN/lib<PassName>.dylib`
- Each phase subdirectory has its own `CMakeLists.txt` that registers the plugin targets
- The MLIR build must exist at `build/` before passes can be compiled (run `scripts/build.sh` first)

### Testing Requirements
- Phase 2 passes: `uv run python scripts/verify_phase2.py`
- Phase 5 passes: `uv run python scripts/verify_phase5.py`
- Manual smoke test: `build/bin/mlir-opt --load-pass-plugin=build/passes/phaseN/lib<Pass>.dylib --<pass-flag> <input.mlir>`

### Common Patterns
- Pass registration uses `PassPlugin::get()` + `PassRegistration<PassClass>` MLIR plugin API
- Pass class inherits from `mlir::PassWrapper<Derived, mlir::OperationPass<...>>`
- Each `.cpp` file defines one pass; header declarations live in the same directory

## Dependencies

### Internal
- `build/bin/mlir-opt` — required at link/load time
- `third_party/torch-mlir/externals/llvm-project/mlir/include/` — MLIR headers

### External
- LLVM/MLIR CMake infrastructure from the existing build

<!-- MANUAL: -->
