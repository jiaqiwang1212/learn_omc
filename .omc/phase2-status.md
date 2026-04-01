# Phase 2 Scaffold Status

_Last updated: 2026-04-01_

## Files Created

| File | Status |
|------|--------|
| `passes/CMakeLists.txt` | ✅ Created |
| `passes/phase2/CMakeLists.txt` | ✅ Created |
| `passes/phase2/ex1_noop_pass.cpp` | ✅ Created (TODO skeleton) |
| `passes/phase2/ex2_counter_pass.cpp` | ✅ Created (TODO skeleton) |
| `passes/phase2/ex3_rewrite_pass.cpp` | ✅ Created (TODO skeleton) |
| `scripts/build_passes.sh` | ✅ Created |
| `scripts/phase2/test_input.mlir` | ✅ Created |
| `scripts/verify_phase2.py` | ✅ Created |

## Build Status

All 3 plugins compile and link cleanly:

```
build/passes/phase2/libEx1NoopPass.dylib   ✅
build/passes/phase2/libEx2CounterPass.dylib ✅
build/passes/phase2/libEx3RewritePass.dylib ✅
```

### Fixes Applied During Scaffolding

1. **`MLIR_PLUGIN_API` → `LLVM_ATTRIBUTE_WEAK`** (all 3 files)
   - This MLIR version (23.0.0git) does not define `MLIR_PLUGIN_API`; the correct macro from `llvm/Support/Compiler.h` is `LLVM_ATTRIBUTE_WEAK`
   - The correct entry point signature is:
     ```cpp
     extern "C" ::mlir::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
     mlirGetPassPluginInfo() { ... }
     ```

2. **`applyPatternsAndFoldGreedily` → `applyPatternsGreedily`** (`ex3_rewrite_pass.cpp`)
   - Renamed in this MLIR version

## Known Blocker

**Pass flags not appearing in `mlir-opt --help` after `--load-pass-plugin`**

- `mlirGetPassPluginInfo` symbol _is_ exported (confirmed via `nm`)
- Plugin loads without error
- But `--ex1-noop`, `--ex2-counter`, `--ex3-rewrite` are not registered

**Root cause:** `mlir::PassRegistration<T>()` is called as a temporary in the lambda, which destructs immediately. The documented pattern requires a **static** instance:

```cpp
// Current (broken) — temporary destructs before registry is read:
[]() { mlir::PassRegistration<Ex1NoopPass>(); }

// Fix — static persists for the lifetime of the plugin:
[]() { static mlir::PassRegistration<Ex1NoopPass> reg; }
```

**Fix needed in:** all 3 `mlirGetPassPluginInfo()` lambdas (one line each).

## Next Action

Apply the static registration fix, rebuild, and run `uv run python scripts/verify_phase2.py`.

## IDE Diagnostics Note

clangd reports false-positive errors in the `.cpp` files because there is no `compile_commands.json` for the `passes/` directory. After running `bash scripts/build_passes.sh`, the file is generated at `build/passes/compile_commands.json`. To activate it in your IDE, either:
- Symlink: `ln -s build/passes/compile_commands.json passes/compile_commands.json`
- Or point clangd at the build directory with a `.clangd` config file
