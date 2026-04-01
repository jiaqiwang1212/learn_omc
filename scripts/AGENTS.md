<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-01 | Updated: 2026-04-01 -->

# scripts/

## Purpose
Learning exercises and build/verification utilities organized by curriculum phase. Contains Python scripts for IR analysis, shell scripts for driving `mlir-opt` lowering pipelines, and per-phase verifier scripts that gate acceptance criteria. Root-level scripts handle the CMake build and pass compilation.

## Key Files
| File | Description |
|------|-------------|
| `build.sh` | CMake configure + ninja build for MLIR/torch-mlir (45–90 min first run) |
| `build_passes.sh` | Incremental build of C++ pass plugin dylibs |
| `verify_phase0.py` | AC0: confirms build tools exist at `build/bin/` |
| `verify_phase1.py` | AC1: IR literacy checks |
| `verify_phase2.py` | AC2: pass plugin load and execution |
| `verify_phase3.py` | AC3: lowering chain produces valid LLVM IR |
| `verify_phase3_5.py` | AC3.5: broken-pipeline diagnostic reasoning |
| `verify_phase4.py` | AC4: LLVM IR compiles and runs on CPU with correct numerical output |
| `verify_phase5.py` | AC5: custom `myml` dialect end-to-end lowering |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `phase1/` | IR literacy exercises — export, annotate, and analyze torch-mlir output (see `phase1/AGENTS.md`) |
| `phase2/` | Test MLIR inputs for phase 2 pass plugins (see `phase2/AGENTS.md`) |
| `phase3/` | Lowering chain exercises: step-by-step, bufferization, full chain, break-the-pipeline (see `phase3/AGENTS.md`) |
| `phase4/` | CPU execution exercises: export, lower to LLVM IR, runtime shim, compile and run (see `phase4/AGENTS.md`) |
| `phase5/` | Custom dialect end-to-end script and test MLIR input (see `phase5/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- Always invoke Python scripts via `uv run python scripts/<file>.py`
- Shell scripts assume `build/bin/` tools are on PATH or reference them by relative path
- Verifier scripts are the source of truth for acceptance criteria — pass = exit 0

### Testing Requirements
- Run the verifier for the phase being modified: `uv run python scripts/verify_phaseN.py`
- Do not mark an exercise complete until its verifier passes

### Common Patterns
- Exercise files are named `exN_<description>.<ext>` within each phase directory
- Python exercises use `torch-mlir` Python bindings or subprocess calls to `mlir-opt`
- Shell exercises chain `mlir-opt` invocations with `--convert-*` and `--lower-*` pass flags

## Dependencies

### Internal
- `build/bin/mlir-opt`, `build/bin/mlir-translate`, `build/bin/torch-mlir-opt` — required at runtime
- `build/passes/phase2/`, `build/passes/phase5/` — pass plugin dylibs for phases 2 and 5

### External
- `torch-mlir` Python bindings (via `uv` / `pyproject.toml`)
- `uv` — Python environment and runner

<!-- MANUAL: -->
