<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-01 | Updated: 2026-04-01 -->

# passes/phase2/

## Purpose
Three introductory C++ MLIR pass plugins that teach the pass plugin API: a noop pass that walks and prints all op names, a counter pass that counts `linalg.matmul` ops, and a rewrite pass that inserts a `func.call @matmul_hook` before each matmul. All three are loaded as runtime plugins via `--load-pass-plugin`.

## Key Files
| File | Description |
|------|-------------|
| `CMakeLists.txt` | Defines three plugin targets: `Ex1NoopPass`, `Ex2CounterPass`, `Ex3RewritePass` |
| `ex1_noop_pass.cpp` | Walk all ops and print names — teaches op traversal |
| `ex2_counter_pass.cpp` | Count `linalg.matmul` occurrences — teaches op matching |
| `ex3_rewrite_pass.cpp` | Insert `func.call @matmul_hook` before each matmul — teaches IR rewriting with `OpBuilder` |

## For AI Agents

### Working In This Directory
- Build with `bash scripts/build_passes.sh` from repo root; outputs to `build/passes/phase2/`
- Test input MLIR is at `scripts/phase2/test_input.mlir`
- Load and run example:
  ```bash
  build/bin/mlir-opt --load-pass-plugin=build/passes/phase2/libEx1NoopPass.dylib \
                     --ex1-noop scripts/phase2/test_input.mlir
  ```

### Testing Requirements
- `uv run python scripts/verify_phase2.py` — must exit 0
- Each pass must load without error and produce expected output/IR transformations

### Common Patterns
- Pass class: `struct MyPass : mlir::PassWrapper<MyPass, mlir::OperationPass<mlir::ModuleOp>>`
- Registration: `llvm::PassPluginLibraryInfo getMyPassPluginInfo()` + `extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo()`
- Op matching: `op->getName().getStringRef() == "linalg.matmul"` or `mlir::isa<mlir::linalg::MatmulOp>(op)`

## Dependencies

### Internal
- `passes/CMakeLists.txt` — parent CMake
- `scripts/phase2/test_input.mlir` — test input
- `build/bin/mlir-opt` — runtime loader

### External
- MLIR pass plugin headers from `third_party/torch-mlir/externals/llvm-project/mlir/include/`

<!-- MANUAL: -->
