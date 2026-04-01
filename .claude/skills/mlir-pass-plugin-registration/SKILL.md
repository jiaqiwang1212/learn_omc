---
name: mlir-pass-plugin-registration
description: |
  Fix for MLIR pass plugins that load without error but whose passes are never
  recognized by mlir-opt. Use when: (1) --load-pass-plugin loads cleanly but
  --pass-name gives "Unknown command line argument", (2) --pass-pipeline="pass-name"
  gives "'pass-name' does not refer to a registered pass", (3) plugin .dylib is
  suspiciously large (~35MB instead of ~64KB). Root cause: CMakeLists.txt uses
  add_library(SHARED) which statically links all MLIR libs into the plugin,
  creating a duplicate private pass registry. Fix: use MODULE + dynamic_lookup.
  Also covers PassNameParser snapshot issue and correct --pass-pipeline invocation
  for dynamically loaded passes.
author: Claude Code
version: 1.0.0
date: 2026-04-01
---

# MLIR Pass Plugin Registration

## Problem

A pass plugin builds and loads without error, but the passes it registers are
never recognized:

```
mlir-opt: Unknown command line argument '--ex1-noop'
```
or
```
error: 'ex1-noop' does not refer to a registered pass or pass pipeline
```

## Root Causes

### 1. Duplicate Registry (PRIMARY — causes both errors above)

`add_library(MyPass SHARED ...)` with direct MLIR library targets statically
links MLIR into the plugin. The plugin gets its own private copy of `passRegistry`.
`PassRegistration<T>` registers into the plugin's copy; `mlir-opt` never sees it.

**Diagnostic**: Check binary size.
```sh
ls -lh build/passes/phase2/libEx1NoopPass.dylib
# BAD:  ~35 MB  (entire MLIR linked in)
# GOOD: ~64 KB  (just plugin code)
```

### 2. `PassNameParser` Snapshot (prevents `--flag-name` syntax for any plugin pass)

`PassNameParser::initialize()` snapshots `passRegistry` at program startup, before
`--load-pass-plugin` callbacks fire. Even with the MODULE fix, plugin passes will
never appear as `--pass-name` CLI flags. Use `--pass-pipeline` instead.

### 3. Temporary `PassRegistration` (causes silent no-op registration)

```cpp
// WRONG — temporary destructs before registry is read:
[]() { mlir::PassRegistration<MyPass>(); }

// CORRECT — static persists for plugin lifetime:
[]() { static mlir::PassRegistration<MyPass> reg; }
```

## Solution

### Fix CMakeLists.txt — MODULE instead of SHARED

```cmake
# WRONG
add_library(Ex1NoopPass SHARED ex1_noop_pass.cpp)
target_link_libraries(Ex1NoopPass PRIVATE MLIRPass MLIRFuncDialect ...)

# CORRECT
add_library(Ex1NoopPass MODULE ex1_noop_pass.cpp)
# No direct MLIR library linkage — symbols resolve from mlir-opt at load time
if(APPLE)
  target_link_options(Ex1NoopPass PRIVATE -undefined dynamic_lookup)
endif()
```

### Fix plugin entry point — static registration

```cpp
extern "C" ::mlir::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "MyPass", "0.1", []() {
    static mlir::PassRegistration<MyPass> reg;  // static, not temporary
  }};
}
```

### Invoke via --pass-pipeline (not --flag-name)

```sh
# WRONG (never works for plugin passes)
mlir-opt --load-pass-plugin=libMyPass.so --my-pass input.mlir

# CORRECT
mlir-opt --load-pass-plugin=libMyPass.so \
  --pass-pipeline="builtin.module(func.func(my-pass))" \
  input.mlir
```

Match the pipeline nesting to the pass's anchor type:
- `OperationPass<func::FuncOp>` → `func.func(my-pass)`
- `OperationPass<ModuleOp>` → `builtin.module(my-pass)`

### Ex3-style passes: declare callee before inserting func.call

If your pass inserts a `func.call` to a symbol that doesn't exist in the module,
declare it first or `--verify-each` will fail:

```cpp
void runOnOperation() override {
  func::FuncOp func = getOperation();
  MLIRContext *ctx = &getContext();
  auto moduleOp = func->getParentOfType<ModuleOp>();
  if (moduleOp && !moduleOp.lookupSymbol<func::FuncOp>("my_hook")) {
    OpBuilder builder(moduleOp.getBodyRegion());
    builder.setInsertionPointToStart(moduleOp.getBody());
    auto hookType = FunctionType::get(ctx, {}, {});
    auto hookDecl = builder.create<func::FuncOp>(func.getLoc(), "my_hook", hookType);
    hookDecl.setPrivate();
  }
  // ... pattern rewrite ...
}
```

## Verification

```sh
# 1. Rebuild
bash scripts/build_passes.sh

# 2. Confirm small binary (MODULE, not SHARED)
ls -lh build/passes/phase2/libEx1NoopPass.so   # expect ~64KB

# 3. Run pass via pipeline
build/bin/mlir-opt \
  --load-pass-plugin=build/passes/phase2/libEx1NoopPass.so \
  --pass-pipeline="builtin.module(func.func(ex1-noop))" \
  scripts/phase2/test_input.mlir
```

## Environment

- MLIR version: 23.0.0git (torch-mlir submodule)
- macOS: requires `-undefined dynamic_lookup` linker flag
- Linux: no extra linker flags needed; MODULE produces `.so`

## Notes

- MLIR's own plugin example (`mlir/examples/standalone/standalone-plugin/CMakeLists.txt`)
  uses `MODULE` — check it as a reference.
- The `--pass-pipeline` text parser calls `PassInfo::lookup()` at parse time against the
  live `passRegistry`, so plugin-registered passes work correctly.
- `rewriter.create<OpTy>(...)` is deprecated in this MLIR version; prefer `OpTy::create(...)`.
