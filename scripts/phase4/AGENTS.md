<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-01 | Updated: 2026-04-01 -->

# scripts/phase4/

## Purpose
CPU execution exercises. The goal (AC4) is to compile the LLVM IR produced in Phase 3 to a native binary, run it on CPU, and verify the numerical output matches `torch.matmul(a, b)` to within 1e-5. Covers export, lowering to LLVM IR, writing a C runtime shim, and compiling + running end-to-end.

## Key Files
| File | Description |
|------|-------------|
| `ex1_export_matmul.py` | Export a PyTorch matmul with concrete shapes for compilation |
| `ex2_lower_to_llvmir.sh` | Full `mlir-opt` pipeline that produces a `.ll` file ready for `llc` |
| `ex3_runtime_shim.c` | C file providing `matmul_hook` and memory allocation stubs called by the compiled IR |
| `ex4_compile_and_run.sh` | Compile `.ll` + shim with `llc`/`clang`, link, and run; print output tensor |

## For AI Agents

### Working In This Directory
- Python exercises: `uv run python scripts/phase4/exN_<name>.py`
- Shell exercises: `bash scripts/phase4/ex4_compile_and_run.sh`
- Requires `build/bin/llc` and `build/bin/lli` — build these with `cmake --build build --target llc lli`

### Testing Requirements
- `uv run python scripts/verify_phase4.py` — runs the compiled binary and checks numerical output vs PyTorch reference to within 1e-5

### Common Patterns
- `llc -filetype=obj` compiles `.ll` to object file; link with `clang` + runtime shim
- Memory for tensors is typically stack or `malloc`-allocated in the shim
- Compare output with `numpy.allclose(result, torch_reference, atol=1e-5)`

## Dependencies

### Internal
- `scripts/verify_phase4.py` — verifier
- `build/bin/llc`, `build/bin/lli` — compilation tools
- Phase 3 pipeline output (`.ll` file)

### External
- `torch`, `numpy` — for reference output and numerical comparison

<!-- MANUAL: -->
