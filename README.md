# MLIR Tensor Lowering Pipeline

A hands-on MLIR learning curriculum that builds a complete end-to-end pipeline:

```
PyTorch → torch-mlir → MLIR dialects (torch → linalg → affine → llvm) → LLVM IR → CPU executable
```

All 5 acceptance criteria verified. ✅

---

## Acceptance Criteria

| AC | Description | Status |
|----|-------------|--------|
| AC1 | Read unseen torch-mlir output: identify all ops, SSA values, and function signature types within 5 minutes, without docs | ✅ |
| AC2 | Write a C++ MLIR pass that matches and replaces an op in `linalg` dialect, loaded via `--load-pass-plugin` | ✅ |
| AC3 | Lower a PyTorch matmul through MLIR dialects to valid LLVM IR; verified by `llvm-as` | ✅ |
| AC4 | Compile and run the LLVM IR on CPU; output matches `torch.matmul(a, b)` to within 1e-5 | ✅ |
| AC5 | Given a novel op (`torch.aten.relu`), predict the lowering path and write a correct `mlir-opt` pass sequence without docs | ✅ |

---

## Phases

| Phase | Goal | Key artifacts |
|-------|------|---------------|
| **0** — Environment Setup | Working MLIR + torch-mlir build | `scripts/build.sh`, `build/bin/mlir-opt` |
| **1** — IR Literacy | Read torch-mlir output fluently | `scripts/phase1/ex{1-5}_*.py` |
| **2** — First Pass | Write and load a C++ MLIR pass plugin | `passes/phase2/ex{1-3}_*.cpp` |
| **3** — Lowering Chain | Lower matmul through dialects to LLVM IR | `scripts/phase3/ex{1-4}_*` |
| **3.5** — Break the Pipeline | Diagnose intentionally broken pipelines | `scripts/phase3/ex4_break_pipeline.sh` |
| **4** — CPU Execution | Run LLVM IR on CPU, verify numerical output | `scripts/phase4/ex{1-4}_*` |
| **5** — Custom Dialect | Define `myml` dialect + lowering pass to `linalg` | `passes/phase5/`, `scripts/phase5/` |

---

## Prerequisites

- macOS or Linux (Apple Silicon tested)
- `uv` for Python dependency management
- `cmake` + `ninja` + `clang/clang++`

---

## Setup

### 1. Initialize submodules

```bash
git submodule update --init --depth 1 third_party/torch-mlir
cd third_party/torch-mlir
git submodule update --init --depth 1 externals/llvm-project externals/stablehlo
cd ../..
```

### 2. Install Python dependencies

```bash
uv sync
```

### 3. Build MLIR tools (45–90 min on Apple Silicon)

```bash
bash scripts/build.sh
```

Outputs: `build/bin/mlir-opt`, `build/bin/mlir-translate`, `build/bin/torch-mlir-opt`

### 4. Build additional LLVM tools (needed for Phase 3+)

```bash
cmake --build build --target llvm-as    # Phase 3: validate .ll files
cmake --build build --target llc        # Phase 4: compile .ll to native
cmake --build build --target lli        # Phase 4: run .ll directly
```

### 5. Build C++ pass plugins

```bash
bash scripts/build_passes.sh
```

Outputs: `build/passes/phase2/lib{Ex1NoopPass,Ex2CounterPass,Ex3RewritePass}.dylib`  
         `build/passes/phase5/lib{Ex1DefineDialect,Ex2LowerToLinalg}.dylib`

---

## Verification

Run each phase's verifier to confirm completion:

```bash
uv run python scripts/verify_phase0.py    # tools working
uv run python scripts/verify_phase1.py    # AC1: IR literacy
uv run python scripts/verify_phase2.py    # AC2: pass plugins
uv run python scripts/verify_phase3.py    # AC3: valid LLVM IR
uv run python scripts/verify_phase3_5.py  # diagnostic reasoning
uv run python scripts/verify_phase4.py    # AC4: CPU execution
uv run python scripts/verify_phase5.py    # AC5: custom dialect
```

Exit 0 = all checks pass.

---

## Pass Plugins

### Phase 2

```bash
MLIR_OPT=build/bin/mlir-opt
PLUGINS=build/passes/phase2

# Ex1: walk and print all op names
$MLIR_OPT --load-pass-plugin=$PLUGINS/libEx1NoopPass.dylib \
           --ex1-noop scripts/phase2/test_input.mlir

# Ex2: count linalg.matmul ops
$MLIR_OPT --load-pass-plugin=$PLUGINS/libEx2CounterPass.dylib \
           --ex2-counter scripts/phase2/test_input.mlir

# Ex3: insert func.call @matmul_hook before each matmul
$MLIR_OPT --load-pass-plugin=$PLUGINS/libEx3RewritePass.dylib \
           --ex3-rewrite --verify-each scripts/phase2/test_input.mlir
```

### Phase 5

```bash
PLUGINS=build/passes/phase5

# End-to-end: myml dialect → linalg → LLVM IR
bash scripts/phase5/ex3_end_to_end.sh
```

---

## Repository Layout

```
learn_omc/
├── passes/
│   ├── phase2/          # C++ pass plugins (noop, counter, rewrite)
│   └── phase5/          # Custom myml dialect + lowering pass
├── scripts/
│   ├── build.sh         # CMake configure + ninja build
│   ├── build_passes.sh  # Build pass plugins
│   ├── phase1/          # IR literacy exercises
│   ├── phase2/          # Pass plugin test inputs
│   ├── phase3/          # Lowering chain exercises
│   ├── phase4/          # CPU execution exercises
│   ├── phase5/          # Custom dialect exercises
│   └── verify_phase*.py # Per-phase verifiers
├── third_party/
│   └── torch-mlir/      # git submodule (llvm/torch-mlir)
│       └── externals/
│           ├── llvm-project/   # MLIR + LLVM tools
│           └── stablehlo/
└── pyproject.toml
```
