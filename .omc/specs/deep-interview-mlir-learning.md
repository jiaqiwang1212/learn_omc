# Deep Interview Spec: Learning MLIR for ML Tensor Lowering

## Metadata
- Interview ID: mlir-learning-2026-04-01
- Rounds: 8
- Final Ambiguity Score: 15%
- Type: greenfield
- Generated: 2026-04-01
- Threshold: 20%
- Status: PASSED

## Clarity Breakdown
| Dimension | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Goal Clarity | 0.92 | 40% | 0.368 |
| Constraint Clarity | 0.80 | 30% | 0.240 |
| Success Criteria | 0.82 | 30% | 0.246 |
| **Total Clarity** | | | **0.854** |
| **Ambiguity** | | | **15%** |

## Goal
Learn MLIR by building a tiny end-to-end ML tensor lowering pipeline: take a small PyTorch computation, ingest it via **torch-mlir**, write custom MLIR passes to lower through dialects, and emit an LLVM IR / CPU-executable result.

The focus is on understanding and controlling the **lowering** step — torch.compile handles the PyTorch frontend adequately; MLIR gives the ability to customize how tensor ops map to hardware (starting with CPU/LLVM).

## Constraints
- **Background:** Strong ML (PyTorch/TensorFlow fluent), weak compiler internals (IR, passes, codegen are new territory)
- **C++ level:** Decent — has written C++ before, can get things done, but not their strongest language
- **Entry point:** Use torch-mlir as the PyTorch → MLIR bridge; focus energy on writing passes *below* that layer
- **Target hardware:** CPU first (via LLVM backend); GPU/accelerator are out of scope for the initial milestone
- **Not learning MLIR for academic purposes** — goal is practical ability to build and extend lowering pipelines

## Non-Goals
- Writing a PyTorch → MLIR frontend from scratch (torch-mlir handles this)
- GPU codegen (CUDA, ROCm, SPIR-V) — out of scope for milestone 1
- Contributing to upstream torch-mlir or MLIR itself (may come later)
- Compiler theory depth for its own sake — theory is a means, not the goal

## Acceptance Criteria
- [ ] Can read and navigate torch-mlir's output IR without confusion
- [ ] Can write a simple MLIR C++ pass that transforms operations in the `torch` or `linalg` dialect
- [ ] Can lower a small PyTorch function (e.g., a matmul or elementwise op) through MLIR dialects to LLVM IR
- [ ] Can compile and run the resulting LLVM IR on CPU and verify correctness against PyTorch reference output
- [ ] Can explain which dialects are involved in the lowering chain (torch → linalg → affine/vector → llvm) and why

## Assumptions Exposed & Resolved
| Assumption | Challenge | Resolution |
|------------|-----------|------------|
| "I need MLIR" | torch.compile already gives a pipeline without MLIR | User confirmed: needs more control over lowering than torch.compile exposes |
| "Build it myself" | Building a custom frontend is much harder than extending torch-mlir | User chose: use torch-mlir as entry, focus on passes below it |
| "End-to-end means GPU" | Could mean many things | User confirmed: CPU/LLVM is the initial target |

## Technical Context

**Key tools/projects to learn:**
1. **torch-mlir** — converts PyTorch ops to MLIR's `torch` dialect; this is the entry point
2. **MLIR core** — the IR infrastructure, dialect system, pass manager, TableGen patterns
3. **linalg dialect** — key intermediate for tensor ops; most torch ops lower here first
4. **llvm dialect + translation** — final lowering to LLVM IR for CPU codegen
5. **mlir-opt** — the MLIR optimizer tool; essential for experimenting with pass pipelines

**Recommended learning path given background (strong ML, decent C++):**
1. Start with MLIR's official getting-started docs and the "Toy Tutorial" (C++, builds a simple language end-to-end)
2. Study torch-mlir's `TorchToLinalg` conversion passes to understand the op lowering patterns
3. Write a custom pass in C++ that transforms one dialect to another (start with a no-op pass, then a real transformation)
4. Build the full pipeline: Python script → torch-mlir ingestion → custom pass → `mlir-translate` → LLVM IR → `clang`/`lli` → run

## Ontology (Key Entities)

| Entity | Type | Fields | Relationships |
|--------|------|--------|---------------|
| Learner | core domain | ML background, decent C++, learning goal | drives the Pipeline |
| MLIR | core domain | dialects, passes, pass manager, TableGen | backbone of Pipeline |
| TorchMLIR | supporting | torch dialect, op conversion, Python bindings | entry point into MLIR |
| LoweringPipeline | core domain | stages, passes, target | connects TorchMLIR → LLVMBackend |
| MLPass | supporting | transformation logic, C++ API | steps within Pipeline |
| LLVMBackend | supporting | LLVM IR, mlir-translate, clang/lli | bottom of Pipeline |
| CPUTarget | external system | x86/ARM, executable binary | output of LLVMBackend |

## Ontology Convergence
| Round | Entity Count | New | Changed | Stable | Stability Ratio |
|-------|-------------|-----|---------|--------|----------------|
| 1 | 2 | 2 | - | - | N/A |
| 2 | 4 | 2 | 0 | 2 | N/A (round 2) |
| 3 | 5 | 1 | 0 | 4 | building |
| 4 | 6 | 1 | 0 | 5 | 83% |
| 5 | 7 | 1 | 0 | 6 | 86% |
| 6 | 7 | 0 | 0 | 7 | 100% |
| 7 | 7 | 0 | 0 | 7 | 100% |
| 8 | 7 | 0 | 0 | 7 | 100% |

Ontology converged at round 6 and remained stable through completion.

## Interview Transcript
<details>
<summary>Full Q&A (8 rounds)</summary>

### Round 1
**Q:** When you imagine yourself having 'learned MLIR', what are you actually doing with it?
**A:** Build a compiler or DSL
**Ambiguity:** 88% (Goal: 0.30, Constraints: 0.00, Criteria: 0.00)

### Round 2
**Q:** What domain is the compiler or DSL for? What problem would it solve?
**A:** ML / tensor computation
**Ambiguity:** 74% (Goal: 0.50, Constraints: 0.10, Criteria: 0.10)

### Round 3
**Q:** What's your programming background?
**A:** Strong ML, weak compilers (knows PyTorch/TensorFlow well; compiler internals are new)
**Ambiguity:** 61% (Goal: 0.55, Constraints: 0.40, Criteria: 0.15)

### Round 4
**Q:** What would your first concrete milestone look like?
**A:** Build a tiny end-to-end pipeline
**Ambiguity:** 47% (Goal: 0.65, Constraints: 0.45, Criteria: 0.45)

### Round 5 — Contrarian
**Q:** PyTorch 2.0 + torch.compile already gives you a tensor pipeline. What does MLIR give you that torch.compile doesn't?
**A:** Need more control over lowering; torch.compile is the frontend
**Ambiguity:** 38% (Goal: 0.75, Constraints: 0.50, Criteria: 0.55)

### Round 6 — Simplifier
**Q:** How comfortable are you writing and reading C++?
**A:** Decent C++, not expert
**Ambiguity:** 33% (Goal: 0.78, Constraints: 0.65, Criteria: 0.55)

### Round 7 — Simplifier
**Q:** Your tiny end-to-end pipeline lowers ML computation down to...?
**A:** CPU (LLVM IR / executable)
**Ambiguity:** 21% (Goal: 0.85, Constraints: 0.75, Criteria: 0.75)

### Round 8
**Q:** For the PyTorch → MLIR entry point: use torch-mlir or build it yourself?
**A:** Use torch-mlir (learn by extending it)
**Ambiguity:** 15% — threshold met ✅

</details>
