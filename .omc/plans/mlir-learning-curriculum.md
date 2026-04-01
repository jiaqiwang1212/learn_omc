# MLIR Learning Curriculum: ML Tensor Lowering Pipeline

## Metadata
- Source spec: `.omc/specs/deep-interview-mlir-learning.md`
- Created: 2026-04-01
- Revised: 2026-04-01 (v1.1 — Architect + Critic feedback incorporated)
- Target learner: Strong ML (PyTorch/TF), decent C++, weak compiler internals
- Milestone: Build a tiny end-to-end pipeline (PyTorch → torch-mlir → MLIR passes → LLVM IR → CPU executable)
- Estimated duration: 3–4 weeks of focused learning (conservative: 5–6 weeks if C++ build issues arise)

---

## Glossary

| Term | Meaning |
|------|---------|
| **SSA** | Static Single Assignment — every value defined exactly once; values cannot be mutated in place |
| **IR** | Intermediate Representation — the textual/in-memory form of a program inside the compiler |
| **Dialect** | A namespaced set of ops in MLIR (e.g., `linalg`, `arith`, `func`, `torch`) |
| **Op** | An operation in MLIR IR — the basic node (like a function call node in torch.fx) |
| **Pass** | A transformation that reads and rewrites IR |
| **Bufferization** | Converting tensor ops (functional) to memref ops (physical memory) |
| **Memref** | MLIR's typed pointer type — represents a buffer with shape, strides, and element type |
| **Legalization** | The process of converting ops from one dialect to another; "illegal op" means no conversion rule exists |
| **Unrealized cast** | A placeholder conversion op inserted when two dialects don't share a type — must be resolved before translation |
| **Lowering** | The process of translating high-level IR to progressively lower-level IR |

---

## RALPLAN-DR Summary

### Principles
1. **Learn by building** — every phase produces a working artifact, not just reading notes
2. **Anchor to ML knowledge** — map MLIR concepts to PyTorch analogies throughout, not just in Phase 1
3. **Goal-driven scope** — the end-to-end pipeline milestone IS the curriculum; every concept introduced must serve a phase
4. **Use proven entry points** — torch-mlir handles PyTorch → MLIR; learn by extending it, not reinventing it
5. **Verify with testable gates** — each phase ends with a concrete, objectively checkable criterion

### Decision Drivers (top 3)
1. **Strong ML background** → linalg dialect maps naturally to tensor ops; start there for fast conceptual anchoring
2. **Decent C++ (not expert)** → use Python bindings for exploration/prototyping, C++ for actual pass writing; scaffold with CMake templates
3. **First milestone = working pipeline** → curriculum must be project-shaped, not course-shaped; theory fills gaps as needed

### Viable Options

#### Option A: Theory-first
**Approach:** Read MLIR papers and dialect design docs before writing code
- Pros: Deep conceptual foundation; builds mental model for debugging novel pipelines; avoids confusion from IR details
- Cons: Delayed feedback loop; less motivating for a practical learner; conceptual understanding without hands-on reinforcement fades quickly

#### Option B: Tool-first / Reverse-engineer
**Approach:** Install torch-mlir, run examples, read output IR, extend by pattern matching
- Pros: Fast feedback; immediate hands-on experience; highly motivating
- Cons: Risk of surface-level understanding — learner can copy the pipeline without knowing why each pass is there; hits undocumented walls when pipeline breaks in novel ways

#### Option C: Project-first (Selected)
**Approach:** Define the end-to-end pipeline as the driving project; learn exactly what is needed to complete each phase
- Pros: Motivation stays high; every learning step has immediate purpose; matches "decent C++ / strong ML" profile
- Cons: **Inherits Option B's surface-level understanding risk** — learner can cargo-cult the shell script without understanding pass ordering invariants; mitigated by the Phase 3.5 "break the pipeline" checkpoint

**Selected:** Option C. The "break the pipeline" checkpoint (Phase 3.5) is the explicit mitigation for the surface-understanding risk inherited from Option B.

---

## Requirements Summary

**Goal:** Learn MLIR by building a working pipeline:
```
PyTorch function → torch-mlir → MLIR dialects (torch → linalg → affine/vector → llvm) → LLVM IR → CPU executable
```

**Learner profile:**
- Strong: PyTorch/TensorFlow, ML framework internals
- Decent: C++ (can write and read, not expert)
- Weak: compiler IR, passes, codegen, SSA form

**Out of scope:** GPU codegen, custom frontend (no torch-mlir replacement), upstream contributions, academic compiler theory

---

## Acceptance Criteria

- [ ] **AC1** — Given *unseen* torch-mlir output, can identify all ops, their dialect namespaces, SSA values, and function signature types within 5 minutes, without documentation
- [ ] **AC2** — Can write a simple MLIR C++ pass that matches and replaces at least one op in the `torch` or `linalg` dialect, and the pass runs successfully via `mlir-opt --load-pass-plugin`
- [ ] **AC3** — Can lower a small PyTorch function (matmul or elementwise op) through MLIR dialects to valid LLVM IR; verified by `llvm-as matmul.ll` succeeding without errors
- [ ] **AC4** — Can compile and run the resulting LLVM IR on CPU; output matches `torch.matmul(a, b).numpy()` to within 1e-5 tolerance
- [ ] **AC5** — Given a *novel op not covered in the curriculum* (e.g., `torch.aten.relu`), can predict the lowering path and write a correct `mlir-opt` pass sequence for it, without documentation

---

## Implementation Phases

### Phase 0: Environment Setup
**Duration:** 1 day
**Goal:** Working MLIR + torch-mlir installation with a verified hello-world

**Workspace layout:**
```
learn_omc/                        ← git repo root
├── third_party/
│   └── torch-mlir/               ← git submodule (llvm/torch-mlir)
│       └── externals/
│           ├── llvm-project/     ← sub-submodule (provides mlir-opt, mlir-translate)
│           └── stablehlo/        ← sub-submodule
├── build/                        ← cmake build output (gitignored)
├── scripts/
│   └── build.sh                  ← cmake configure + ninja build script
├── .venv/                        ← uv-managed Python 3.11 venv (gitignored)
└── pyproject.toml                ← uv project (torch 2.11.0, numpy)
```

**Steps (already completed):**
1. `git init` the workspace; add `.venv/` and `build/` to `.gitignore`
2. `uv init --python 3.11` + `uv add torch numpy`
3. `git submodule add --depth 1 https://github.com/llvm/torch-mlir.git third_party/torch-mlir`
4. `cd third_party/torch-mlir && git submodule update --init --depth 1 externals/llvm-project externals/stablehlo`
5. Run `scripts/build.sh` — CMake configures against torch-mlir's bundled llvm-project, builds `mlir-opt`, `mlir-translate`, `torch-mlir-opt`, and the Python bindings

**Build (run once submodules are initialized):**
```bash
./scripts/build.sh
# Takes 45–90 min on Apple Silicon (10 cores)
# Output: build/bin/mlir-opt, build/bin/torch-mlir-opt
```

**Why source build (not pip install):**
- `mlir-opt` and the Python bindings must be built from the same LLVM revision — torch-mlir's bundled `externals/llvm-project` submodule pins the exact compatible commit
- Avoids version mismatch between `mlir-opt` and `torch_mlir` Python module (a common failure mode with prebuilts)

**Verification (after build):**
```bash
uv run python scripts/verify_phase0.py
# Exit 0 = all checks pass. Checks:
#   1. build/bin/mlir-opt exists and --version exits 0
#   2. mlir-opt --verify-each round-trips a simple arith.addf function
#   3. import torch_mlir succeeds
#   4. torch_mlir.compile(Relu(), ..., output_type=TORCH) produces valid IR
```

**Resources:**
- torch-mlir build from source: https://github.com/llvm/torch-mlir/blob/main/docs/development.md
- MLIR getting started: https://mlir.llvm.org/getting_started/

---

### Phase 1: Read MLIR — IR Literacy
**Duration:** 3–4 days
**Goal:** Understand MLIR's structure well enough to read torch-mlir output fluently
**Unlocks:** AC1

**Prerequisite concepts to acquire (in order):**

1. **SSA form** — every value is defined exactly once; you cannot update a value in place.
   - Analogous to: tensor nodes in a `torch.fx` graph — each node produces a new tensor, never mutates an existing one
   - **Mechanical implication for pass writing**: when you write a `RewritePattern`, you create a *new op* with a new result value and call `replaceOpWithNewOp` — you never "edit" the existing op's result in place. Violating this produces a "value defined multiple times" verifier error.

2. **Operations, Attributes, Types** — every node in MLIR IR is an `op`; types annotate values; attributes are compile-time constants
3. **Regions and Blocks** — regions contain blocks; blocks contain ops; this is how control flow and function bodies are represented
4. **Dialects** — namespaced sets of ops; `linalg.matmul`, `arith.addf`, `func.func` are all dialect ops

**ML analogies:**
| MLIR concept | PyTorch analogy |
|--------------|----------------|
| SSA value | Tensor in a `torch.fx` graph node (immutable, one producer) |
| Operation | `torch.ops.aten.mm` call |
| Dialect | PyTorch operator namespace (`aten::`, `prims::`) |
| Pass | `torch.fx` graph transformation pass |
| Region | Function body / loop body |
| Type | Tensor dtype + shape (`f32`, `!torch.vtensor<[4,4],f32>`) |
| Bufferization | Going from `torch.Tensor` (logical view) to `numpy.ndarray` (physical memory with strides and pointer) |

**Exercises:**
1. Export a simple PyTorch matmul using torch-mlir:
   ```python
   import torch
   import torch_mlir

   class Matmul(torch.nn.Module):
       def forward(self, a, b):
           return torch.matmul(a, b)

   module = torch_mlir.compile(
       Matmul(),
       (torch.ones(4, 4), torch.ones(4, 4)),
       output_type=torch_mlir.OutputType.TORCH
   )
   print(module.operation.get_asm())
   ```
2. Annotate the output IR: label every SSA value (`%0`, `%1`, etc.), identify the `torch.operator` ops, find the function signature types
3. **SSA def-use chain tracing exercise** (critical for Phase 2 success):
   - Find the SSA value that holds the matmul result (e.g., `%3`)
   - Trace: which op *defines* it? Which ops *use* it? Draw the def-use chain for the 3 most important values
   - Ask yourself: "If I wanted to insert a new op between the matmul and the return, what would change in the IR?"
   - This exercise directly prepares you for writing `RewritePattern` code in Phase 2
4. Read the same function at `output_type=torch_mlir.OutputType.LINALG_ON_TENSORS` — compare what changed. Which ops are gone? What replaced them?
5. Export `torch.relu(x)` at `LINALG_ON_TENSORS` and compare with the matmul output. You will see `linalg.generic` with indexing maps instead of a named op. Read the indexing map — it describes the loop structure. (**Note:** most non-matmul ops lower to `linalg.generic`; getting comfortable reading it is important for AC5.)

**Milestone check (AC1):** Take a fresh, unseen torch-mlir output file. Without documentation:
- Identify every op and its dialect namespace
- Identify every SSA value and its producer op
- Identify the function signature (argument types, return type)

**Resources:**
- MLIR Language Reference: https://mlir.llvm.org/docs/LangRef/
- MLIR Toy Tutorial Ch. 1–2 (skim for IR reading): https://mlir.llvm.org/docs/Tutorials/Toy/Ch-1/
- torch-mlir dialect docs: https://github.com/llvm/torch-mlir/blob/main/docs/dialects.md

---

### Phase 2: First Pass — Write a Transformation
**Duration:** 3–4 days
**Goal:** Write a working MLIR C++ pass
**Unlocks:** AC2

**Prerequisite concepts:**
1. **Pass Manager** — pipeline of passes applied to IR; `mlir::PassManager` in C++; analogous to a `torch.fx` pass pipeline
2. **Pattern rewriting** — `RewritePattern` + `PatternRewriter` for matching and replacing ops; `replaceOpWithNewOp` creates a new SSA value and updates all uses
3. **Op interfaces** — ops implement typed interfaces (`LinalgOp`, `MemoryEffectsOpInterface`) for generic handling

> **Note on TableGen:** You will encounter TableGen when reading MLIR source code — it is a DSL for declaring ops and passes. You do *not* need to write TableGen for this milestone. See Appendix A for a brief explanation of when you'll need it.

**C++ scaffolding — use existing passes as templates (do not write from scratch):**
- Start from: `llvm-project/mlir/lib/Transforms/` for simple pass patterns
- Real pass examples from torch-mlir: `torch-mlir/lib/Conversion/TorchToLinalg/Linear.cpp`

**Exercises:**
1. **No-op pass**: Write a C++ pass that walks all ops in the `linalg` dialect and prints their names using `mlir::Operation::getName().getStringRef()`. Compile with CMake as a shared library.
2. **Counter pass**: Extend to count `linalg.matmul` ops. Print the count at pass end via `signalPassFailure()` (failure path) or `llvm::errs()`.
3. **Rewrite pass**: Use `RewritePattern` to match `linalg.matmul` and insert a `func.call` before it (to a stub function). Practice calling `rewriter.replaceOpWithNewOp<>()` and `rewriter.create<>()`.
4. **Test with `--verify-each`** (essential debugging tool):
   ```bash
   mlir-opt --load-pass-plugin=./libMyPass.so --my-pass --verify-each input.mlir
   ```
   `--verify-each` runs the MLIR verifier after every pass. If your pass produces invalid IR (e.g., a dangling use, a type mismatch), you get an error message *at that specific pass*, not a mysterious crash later. **Always use `--verify-each` when developing passes.**

**CMake skeleton:**
```cmake
find_package(MLIR REQUIRED CONFIG)
add_library(MyPass SHARED MyPass.cpp)
target_link_libraries(MyPass MLIRIR MLIRPass MLIRTransforms MLIRLinalgDialect)
```

**Milestone check (AC2):** Run `mlir-opt --load-pass-plugin=./libMyPass.so --my-pass input.mlir` and observe the expected transformation in the output.

**Resources:**
- MLIR Toy Tutorial Ch. 3 (writing passes): https://mlir.llvm.org/docs/Tutorials/Toy/Ch-3/
- MLIR Pass Infrastructure: https://mlir.llvm.org/docs/PassManagement/
- Pattern Rewriting guide: https://mlir.llvm.org/docs/PatternRewriter/
- torch-mlir source (real pass examples): `lib/Conversion/TorchToLinalg/Linear.cpp`

---

### Phase 3: The Lowering Chain — Dialects End-to-End
**Duration:** 5–7 days
**Goal:** Understand and manually execute each step of the lowering chain
**Unlocks:** AC3, AC5 (partial)

**The full chain to understand:**
```
torch dialect           (torch-mlir output)
    ↓  convert-torch-to-linalg
linalg on tensors       (named ops: linalg.matmul, linalg.generic)
    ↓  one-shot-bufferize
linalg on memrefs       (same ops but memref types; alloc/dealloc inserted)
    ↓  convert-linalg-to-affine-loops
affine loops            (explicit loop nests with affine indexing)
    ↓  lower-affine / convert-scf-to-cf
control flow            (basic block branches)
    ↓  convert-cf-to-llvm + convert-arith-to-llvm + finalize-memref-to-llvm + convert-func-to-llvm
llvm dialect            (MLIR's 1:1 representation of LLVM IR)
    ↓  --reconcile-unrealized-casts  ← explained below
    ↓  mlir-translate --mlir-to-llvmir
LLVM IR                 (.ll file)
```

**Why each step exists:**
| Step | Why it exists |
|------|--------------|
| torch → linalg | linalg has structured semantics (named loops, indexing maps) that enable optimization |
| tensors → memrefs (bufferization) | linalg on tensors is functional (no addresses); hardware needs memory pointers; bufferization inserts `memref.alloc`/`dealloc` |
| linalg → loops | explicit loop nests needed for LLVM codegen |
| loops → LLVM dialect | MLIR's LLVM dialect is a 1:1 mapping to LLVM IR instructions |
| LLVM dialect → LLVM IR | `mlir-translate` emits the textual `.ll` format |

**What is `--reconcile-unrealized-casts` and why is it needed?**

When you convert multiple dialects to LLVM dialect (e.g., `convert-arith-to-llvm`, `convert-func-to-llvm`), each conversion replaces ops from one dialect but may leave *type mismatches* at the boundaries between converted and not-yet-converted regions. MLIR handles these mismatches by inserting `builtin.unrealized_conversion_cast` placeholder ops — they say "this type needs to become that type, but I don't know how yet."

`--reconcile-unrealized-casts` is the final cleanup pass: it resolves all these placeholders. If it finds a cast that cannot be resolved (e.g., two types that have no valid conversion), it errors out with a clear message. **This pass must always come last, after all other dialect conversions.** If `mlir-translate` fails with `"failed to lower: builtin.unrealized_conversion_cast"`, you are missing `--reconcile-unrealized-casts` or have applied it too early.

**ML analogy for bufferization:** Going from `torch.Tensor` (logical tensor, no memory address, functional) to `numpy.ndarray` (physical memory layout with dtype, strides, and a pointer). Bufferization is that transition.

**Exercises:**
1. **Step-by-step with `mlir-opt` + `--verify-each` + `--debug-only`**:
   Start from `LINALG_ON_TENSORS` output and apply one pass at a time, saving intermediate files. Use these flags:
   ```bash
   mlir-opt input.mlir --one-shot-bufferize="bufferize-function-boundaries" \
     --verify-each \
     --debug-only=dialect-conversion \
     -o step1_bufferized.mlir
   ```
   `--debug-only=dialect-conversion` prints which conversion patterns fire and in what order — far more useful than `--debug` (which floods the terminal with everything).
2. **Bufferization deep dive**: Compare `input.mlir` (tensors) with `step1_bufferized.mlir` (memrefs). Find every `memref.alloc`. Understand why each one was inserted. Where did the tensor "become" an allocation?
3. **Full chain script** — build this incrementally, one pass at a time:
   ```bash
   mlir-opt input.mlir \
     --one-shot-bufferize="bufferize-function-boundaries" \
     --convert-linalg-to-affine-loops \
     --lower-affine \
     --convert-scf-to-cf \
     --convert-cf-to-llvm \
     --convert-arith-to-llvm \
     --finalize-memref-to-llvm \
     --convert-func-to-llvm \
     --reconcile-unrealized-casts \
   | mlir-translate --mlir-to-llvmir -o output.ll
   ```

**Milestone check (AC3, AC5 partial):** Given a fresh `linalg.matmul` IR, trace which pass handles each op and explain why that pass is in that specific position in the sequence.

**Resources:**
- MLIR Conversion passes reference: https://mlir.llvm.org/docs/Passes/
- One-shot bufferization: https://mlir.llvm.org/docs/Bufferization/
- linalg dialect: https://mlir.llvm.org/docs/Dialects/Linalg/

---

### Phase 3.5: Break the Pipeline — Diagnostic Reasoning
**Duration:** 1 day
**Goal:** Build genuine understanding of pass ordering invariants by deliberately breaking the pipeline
**Purpose:** Prevents cargo-culting the shell script without understanding why it works

> **Why this phase exists:** After Phase 3, you can copy the shell script and it runs. That is necessary but not sufficient. If any pass is removed, reordered, or if a new op is added to your pipeline, you must be able to diagnose what fails and why. This phase builds that diagnostic muscle.

**Exercises:**
1. **Remove `--reconcile-unrealized-casts`** from the script. Run it. Read the error from `mlir-translate`. Explain in one sentence what went wrong.
2. **Swap the order of bufferization and `--convert-linalg-to-affine-loops`**. Run with `--verify-each`. Read the verifier error. Explain why bufferization must precede loop lowering (hint: what types do the linalg ops have after bufferization vs. before?).
3. **Remove `--convert-func-to-llvm`**. Run the full chain. What happens during `mlir-translate`? Explain why `func.func` must be converted to LLVM before translation.
4. **(Bonus)** Export `torch.relu(x)` instead of `torch.matmul`. Does your lowering script work unchanged? If not, diagnose why — which pass doesn't handle the new ops? Use `--debug-only=dialect-conversion` to find the gap.

**Milestone check:** Given an error output from a broken pipeline, can you identify which pass is failing, why, and what change fixes it?

---

### Phase 4: End-to-End Pipeline — Run on CPU
**Duration:** 5–7 days
**Goal:** PyTorch function → CPU executable; output matches PyTorch reference
**Unlocks:** AC4 (full milestone)

**Pipeline to assemble:**
```
PyTorch module (Python)
    ↓  torch_mlir.compile(..., output_type=LINALG_ON_TENSORS)
linalg on tensors IR (.mlir)
    ↓  shell script from Phase 3
LLVM IR (.ll)
    ↓  clang -O2 -o output output.ll runtime_shim.c
CPU executable
    ↓  ./output | compare against Python reference
```

**Steps:**
1. **Choose the target function**: `def f(a, b): return torch.matmul(a, b)` with 4×4 float32 inputs
2. **Export with torch-mlir**: Use `output_type=LINALG_ON_TENSORS`. Save to `matmul.mlir`
3. **Run the lowering script** from Phase 3. Produce `matmul.ll`
4. **Write the runtime shim** — see calling convention below
5. **Compile and run**: `clang -O2 matmul.ll runtime_shim.c -o run && ./run`
6. **Verify**: Compare output against `torch.matmul(a, b).numpy()`

**Critical: The `memref` descriptor calling convention**

MLIR-compiled functions do not accept raw `float*` pointers. They expect `memref` descriptors — C structs with this layout:

```c
// From mlir/include/mlir/ExecutionEngine/CRunnerUtils.h
// For a 2D memref (e.g., a 4x4 matrix):
typedef struct {
    float *allocated;   // pointer to the original allocation (for dealloc)
    float *aligned;     // pointer to the aligned data start (what you index into)
    int64_t offset;     // offset from aligned to first element (usually 0)
    int64_t sizes[2];   // shape: {4, 4}
    int64_t strides[2]; // strides: {4, 1} for row-major
} MemRef2D;
```

The MLIR-compiled function for `matmul(a: memref<4x4xf32>, b: memref<4x4xf32>) -> memref<4x4xf32>` will be called with *pointers to* these structs (passed by reference), not by value.

**Runtime shim skeleton (`runtime_shim.c`):**
```c
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    float *allocated;
    float *aligned;
    int64_t offset;
    int64_t sizes[2];
    int64_t strides[2];
} MemRef2D;

// MLIR-compiled function signature (check your matmul.ll to confirm the exact name)
void _mlir_ciface_forward(MemRef2D *result, MemRef2D *a, MemRef2D *b);

int main() {
    // Allocate 4x4 float arrays
    float a_data[16], b_data[16], out_data[16];
    for (int i = 0; i < 16; i++) { a_data[i] = 1.0f; b_data[i] = 1.0f; }

    MemRef2D a = { a_data, a_data, 0, {4, 4}, {4, 1} };
    MemRef2D b = { b_data, b_data, 0, {4, 4}, {4, 1} };
    MemRef2D result = { out_data, out_data, 0, {4, 4}, {4, 1} };

    _mlir_ciface_forward(&result, &a, &b);

    // Print result — should be all 4.0 (matmul of 1s on 4x4)
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++)
            printf("%.1f ", result.aligned[i * 4 + j]);
        printf("\n");
    }
    return 0;
}
```

> **Finding the function name**: Look at `matmul.ll` for the function name. MLIR adds the `_mlir_ciface_` prefix for the C-compatible interface. The full name is in the `.ll` file.

**Alternative if manual shim is too painful:** Use `mlir-cpu-runner`:
```bash
mlir-opt input.mlir [passes...] | \
  mlir-cpu-runner --shared-libs=/path/to/mlir_runner_utils.so --entry-point-result=void
```
`mlir-cpu-runner` handles the calling convention for you. Use it to validate correctness first, then tackle the manual shim.

**Debugging tips:**
- If output values are wrong: add `--debug-only=dialect-conversion` to the lowering script to check if any ops fell through
- If the executable segfaults: the `aligned` pointer in your `MemRef2D` may be wrong; verify it equals `allocated` for simple cases
- If `clang` can't link: try `clang matmul.ll runtime_shim.c -lm -o run`

**Milestone check (AC4):** `./run` prints matrix values that match `torch.matmul(a, b).numpy()` to within float32 precision (1e-5). AC5 final check: export `torch.relu(x)` and write the lowering script for it from scratch.

**Resources:**
- `CRunnerUtils.h` (memref descriptor structs): `mlir/include/mlir/ExecutionEngine/CRunnerUtils.h` in the LLVM monorepo
- mlir-cpu-runner: https://mlir.llvm.org/docs/Tools/MLIRCPURunner/
- MLIR Python bindings (ExecutionEngine JIT): https://mlir.llvm.org/docs/Bindings/Python/
- MLIR integration tests (real end-to-end examples): `mlir/test/Integration/` in the LLVM monorepo

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| torch-mlir version mismatch with LLVM | High | Blocks Phase 1 | Pin torch-mlir to a specific LLVM version; check compatibility matrix at torch-mlir releases page |
| Learner cannot interpret MLIR error messages | High | Blocks at any phase | Use `--verify-each` from Phase 2 onward; use `--debug-only=dialect-conversion` in Phase 3; see Phase 3.5 for deliberate error-reading practice |
| Learner cargo-cults the lowering script | Medium | Produces AC5 failure | Addressed directly by Phase 3.5 "break the pipeline" exercises |
| CMake build complexity for custom passes | Medium | Blocks Phase 2 | Use `mlir-opt` plugin loading (shared library) instead of full CMake integration initially |
| `memref` calling convention mismatch at runtime | Medium | Blocks Phase 4 | Use `mlir-cpu-runner` for first execution; refer to `CRunnerUtils.h` for struct layout; see Phase 4 calling convention section |
| torch-mlir Python API changes between versions | Medium | Breaks Phase 1 exercises | Pin torch-mlir version in Phase 0; check `torch_mlir.compile()` signature in the installed version before writing exercises |
| Bufferization producing incorrect memory layouts | Medium | Blocks Phase 3–4 | Test each intermediate IR step independently; use `--verify-each` |
| C++ compilation errors in pass code | Low | Slows Phase 2 | Copy an existing torch-mlir pass as template; modify incrementally |

---

## Verification Steps (per AC)

| Criterion | Verification |
|-----------|-------------|
| AC1 | Open unseen torch-mlir output; without docs: identify all ops with dialect namespaces, all SSA value producers, and the function signature — in under 5 minutes |
| AC2 | `mlir-opt --load-pass-plugin=./libMyPass.so --my-pass --verify-each input.mlir` produces expected transformation |
| AC3 | `llvm-as matmul.ll` succeeds (valid LLVM IR) |
| AC4 | `./run` output matches `torch.matmul(a,b).flatten().tolist()` to 1e-5 tolerance |
| AC5 | Given `torch.aten.relu`, write the `mlir-opt` lowering pipeline from scratch and confirm it produces valid LLVM IR |

---

## ADR: Learning Path Architecture Decision

### Decision
Use **Project-first curriculum** (Option C) structured around the end-to-end pipeline milestone, with torch-mlir as the ingestion layer, and a mandatory "break the pipeline" checkpoint after Phase 3.

### Drivers
- Strong ML intuition → linalg as conceptual bridge is most efficient
- Practical (not academic) learner → theory introduced only when it unblocks a phase
- C++ comfort is "decent not expert" → Python bindings for exploration, C++ for pass writing

### Alternatives Considered
- **Option A (Theory-first)**: Would build a stronger conceptual foundation for debugging novel pipelines, but delays the first working artifact and risks demotivation for a learner who processes concepts through hands-on work
- **Option B (Tool-first only)**: Fastest path to a running pipeline, but produces shallow understanding; the learner can copy the pipeline without knowing *why* each pass is required — this is also a risk in Option C (mitigated by Phase 3.5)

### Why Chosen
Option C keeps motivation high (each phase has a runnable output), minimizes scope (CPU only, torch-mlir as frontend), and directly maps to the 5 acceptance criteria. Phase 3.5 is the explicit structural answer to the Option B / Option C surface-understanding risk.

### Consequences
- Learner may not deeply understand some MLIR concepts (TableGen interface inheritance, trait system) — acceptable for milestone 1
- If the learner later wants to build a custom frontend or target GPU, they will need Phase 5+ work not covered here

### Follow-ups
- Phase 5 (future): Add a custom MLIR dialect and lower it into the existing pipeline
- Phase 6 (future): Target GPU via `convert-linalg-to-gpu` / SPIR-V lowering
- Phase 7 (future): Contribute a pass or bugfix to upstream torch-mlir

---

## Appendix A: TableGen — What It Is and When You'll Need It

TableGen is a DSL embedded in the LLVM/MLIR build system for declaring ops, passes, and interfaces. You will encounter it when reading MLIR source code (`.td` files). You do **not** need to write TableGen for this curriculum.

**What it does:** Generates C++ boilerplate for op definitions (accessors, verifiers, builders) from declarative `.td` file definitions. The generated C++ is what you include when writing passes.

**When you'll need it:**
- Phase 5 (future): Defining a custom dialect with custom ops requires writing `.td` files
- Reading existing MLIR source: `.td` files define the ops you'll be pattern-matching against

**Reference:** https://mlir.llvm.org/docs/DefiningDialects/ — read when you start Phase 5.

---

## Changelog
- v1.0 (2026-04-01): Initial draft from deep-interview spec. Planner pass.
- v1.2 (2026-04-01): Phase 0 rewritten to reflect actual source-build arrangement: git submodule for torch-mlir, uv Python env, `scripts/build.sh`, workspace layout documented.
- v1.1 (2026-04-01): Incorporated Architect + Critic feedback. 7 required changes applied:
  1. AC1 and AC5 rewritten to be objectively testable (no more "without confusion")
  2. SSA def-use chain tracing exercise added to Phase 1
  3. TableGen moved from Phase 2 prerequisites to Appendix A
  4. `--verify-each` and `--debug-only=dialect-conversion` promoted into Phase 2 and Phase 3 exercises
  5. `--reconcile-unrealized-casts` explained with mechanism and failure modes
  6. Phase 3.5 "break the pipeline" checkpoint added
  7. `memref` descriptor calling convention explained in Phase 4 with C code example
  - Also fixed: `pip install` → `uv pip install`; version pinning added to Phase 0; torch-mlir API version mismatch added to risk table; Options A/B analysis strengthened (removed fabricated "2-3x" claim); `linalg.generic` introduced in Phase 1 Exercise 5; ML analogy table extended to include bufferization; glossary added
