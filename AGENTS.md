<!-- Generated: 2026-04-01 | Updated: 2026-04-01 -->

# learn_omc

## Purpose
A hands-on MLIR learning curriculum that builds a complete end-to-end tensor lowering pipeline:
`PyTorch → torch-mlir → MLIR dialects (torch → linalg → affine → llvm) → LLVM IR → CPU executable`.
All 5 acceptance criteria are verified. Exercises are organized into 5 phases, from IR literacy through writing C++ pass plugins and defining a custom MLIR dialect.

## Key Files
| File | Description |
|------|-------------|
| `README.md` | Full curriculum overview, setup instructions, and verification commands |
| `pyproject.toml` | Python dependencies (managed with `uv`) |
| `main.py` | Entry point / scratch file |
| `scripts/build.sh` | CMake configure + ninja build for MLIR/torch-mlir tools |
| `scripts/build_passes.sh` | Builds C++ pass plugin `.dylib` files |
| `scripts/verify_phase*.py` | Per-phase acceptance criteria verifiers (exit 0 = pass) |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `passes/` | C++ MLIR pass plugin source code organized by phase (see `passes/AGENTS.md`) |
| `scripts/` | Python/shell learning exercises and build scripts organized by phase (see `scripts/AGENTS.md`) |
| `third_party/` | `torch-mlir` git submodule (includes `llvm-project` and `stablehlo`) |

## For AI Agents

### Working In This Directory
- **Never touch** `/Users/jacob/workspace/llvm-project` — it is unrelated; the correct llvm lives at `third_party/torch-mlir/externals/llvm-project`
- Built tools live in `build/bin/` (`mlir-opt`, `mlir-translate`, `torch-mlir-opt`, `llvm-as`, `llc`, `lli`)
- Pass plugin dylibs are built to `build/passes/phase2/` and `build/passes/phase5/`
- Always use `uv run` for Python scripts (e.g. `uv run python scripts/verify_phase1.py`)

### Testing Requirements
- Run the relevant phase verifier: `uv run python scripts/verify_phaseN.py`
- Exit code 0 means all checks pass for that phase
- Full suite: run all `verify_phase*.py` scripts sequentially

### Common Patterns
- Phase exercises are numbered `ex1_`, `ex2_`, … within each `scripts/phaseN/` directory
- C++ passes follow the MLIR pass plugin registration pattern (`PassPlugin::get()` + `PassRegistration`)
- Shell scripts (`.sh`) drive `mlir-opt` pipeline invocations; Python scripts analyze or verify IR

## Dependencies

### External
- `torch-mlir` submodule — provides `torch` dialect and lowering infrastructure
- `uv` — Python package and project manager
- `cmake` + `ninja` + `clang/clang++` — C++ build toolchain

<!-- MANUAL: -->
