# Tinkex Cookbook Core Foundation - Inventory, Remaining Python Libs, Plan

**Date:** 2025-12-22
**Updated:** 2025-12-23
**Context:** Full North-Shore-AI ecosystem now integrated. This document captures
the current doc inventory, remaining Python libraries to consider, and the core foundation plan.

NOTE: Superseded by the corrected ownership model. Adapter implementations now
live in `crucible_kitchen`; references to tinkex_cookbook adapters are historical.

> **2025-12-23 Update:** Added crucible_harness, eval_ex from Hex for inspect-ai parity.
> Two separate concerns now identified: Training recipes (Phase 1) and Evaluation harness (new).

---

## 1) Document Inventory (as of today)

**docs/20251220/**
- `docs/20251220/00_MASTER_SUMMARY.md` - Tinker-Cookbook Porting Feasibility Master Summary
- `docs/20251220/01_chz_library.md` - CHZ Library Analysis Report
- `docs/20251220/02_textarena_library.md` - TextArena Library Research Report
- `docs/20251220/03_math_verify_library.md` - Math-Verify Library Research Report
- `docs/20251220/04_inspect_ai_library.md` - Inspect AI Library Research Report
- `docs/20251220/05_pylatexenc_library.md` - pylatexenc Library Research Report
- `docs/20251220/06_sympy_symbolic_math.md` - SymPy Symbolic Mathematics: Feasibility Assessment
- `docs/20251220/07_scipy_nx_mapping.md` - SciPy to Nx/Scholar Mapping Analysis
- `docs/20251220/08_torch_transformers_axon_mapping.md` - PyTorch/HF to Nx/Axon/Bumblebee Mapping
- `docs/20251220/09_datasets_blobfile_mapping.md` - HuggingFace Datasets & Blobfile Mapping Analysis
- `docs/20251220/10_verifiers_library.md` - Verifiers Library Research Report
- `docs/20251220/11_tinker_to_tinkex.md` - Python tinker -> Elixir tinkex Mapping
- `docs/20251220/12_torch_transformers_actual_usage.md` - Torch & Transformers Actual Usage
- `docs/20251220/12_torch_transformers_actual_usage_SUMMARY.txt` - Torch & Transformers Summary
- `docs/20251220/12a_porting_cheatsheet.md` - Torch -> Nx Porting Cheatsheet
- `docs/20251220/12b_tensor_data_implementation.ex` - TensorData Implementation Example
- `docs/20251220/13_numpy_scipy_actual_usage.md` - NumPy/SciPy Actual Usage
- `docs/20251220/14_datasets_blobfile_actual_usage.md` - Datasets/Blobfile Actual Usage
- `docs/20251220/15_utility_libs.md` - Utility Library Usage Analysis
- `docs/20251220/16_COMPLETE_DEPENDENCY_MAPPING.md` - Complete Dependency Mapping
- `docs/20251220/17_RECIPE_DEPENDENCY_MAPPING.md` - Recipe -> Dependency Mapping
- `docs/20251220/18_DATASETS_GAP_ANALYSIS.md` - Datasets Gap Analysis
- `docs/20251220/19_CRUCIBLE_DATASETS_DEEP_DIVE.md` - crucible_datasets Deep Dive
- `docs/20251220/README.md` - Torch & Transformers Usage Analysis Overview
- `docs/20251220/VERIFICATION_SUMMARY.md` - Verification Summary: TextArena & Verifiers

**docs/20251221/**
- `docs/20251221/REMAINING_WORK.md` - Remaining Work After Datasets
- `docs/20251221/DEPENDENCY_USAGE_TABLE.md` - Dependency usage table (by import scan)
- `docs/20251223/PORTS_AND_ADAPTERS.md` - External services ports/adapters design
- `docs/20251221/PHASE1_FOUNDATION_SLICE.md` - Phase 1 plan (reference experiment)
- `docs/20251221/PHASE1_AGENT_PROMPT.md` - Phase 1 agent prompt (TDD)
- `docs/20251221/PHASE2_LIBS_AND_HALF_RECIPES.md` - Phase 2 plan (core libs + ~half)
- `docs/20251221/PHASE3_REMAINING_EXAMPLES.md` - Phase 3 placeholder plan
- `docs/20251221/SNAKEBRIDGE_LIBS.md` - SnakeBridge Python library targets
- `docs/20251221/COOKBOOK_CORE_FOUNDATION.md` - (this file)

---

## 2) Remaining Python Libraries To Consider

This list is derived from the upstream Python `tinker_cookbook` dependencies after
subtracting what is already solved (crucible_datasets and chz_ex) and factoring in
current Elixir equivalents.

### A) Required for Math RL (Snakebridge candidates)
- **sympy** - symbolic math validation
- **pylatexenc** - LaTeX parsing for math grading
- **math-verify** - equivalence checks for math answers

### B) Optional / Recipe-specific (Snakebridge candidates if needed)
- **textarena** - multiplayer RL environments
- **verifiers** - verifier-based RL (plus **openai** client)
- **chromadb** - vector search backend (tool-use/search; use `chroma`)
- **google-genai** - tool-use/search integration
- **huggingface_hub** - optional vector-search dependency path
- **inspect-ai** - likely replace with native crucible_harness, keep optional

### C) Native Elixir Ports (no Python bridge planned)
- **numpy** -> Nx (vectorized math)
- **scipy** -> native Elixir for `signal.lfilter` (discounted rewards)
- **torch/torchvision** -> Nx + image utilities (no training, data prep only)
- **transformers** -> tokenizers (already in tinkex) + minimal metadata helpers
- **pillow** -> image decoding via existing Elixir image libs (or Nx + image codecs)
- **rich/termcolor** -> IO.ANSI or TableRex for terminal formatting
- **blobfile** -> File + ExAws.S3 (JSONL, blob reads)
- **anyio/asyncio** -> Task/Task.async_stream (concurrency primitives)
- **cloudpickle** -> Elixir serialization; verify actual usage first

### D) Optional Tracking/Telemetry Integrations
- **wandb** - optional experiment tracking
- **neptune-scale** - optional tracking
- **trackio** - optional tracking

---

## 3) Dependency Pins + Doc Notes

**mix.exs pins (updated 2025-12-23)**

Core training dependencies:
- `{:tinkex, "~> 0.3.2"}` - Tinker API client (training + sampling)
- `{:chz_ex, "~> 0.1.2"}` - Configuration schemas + CLI + blueprint pipeline
- `{:snakebridge, "~> 0.3.0"}` - Python interop for math grading

Dataset + evaluation ecosystem:
- `{:crucible_datasets, "~> 0.5.1"}` - Dataset management, loaders, caching
- `{:crucible_harness, "~> 0.3.1"}` - Experiment orchestration, Solver/Generate/TaskState
- `{:eval_ex, "~> 0.1.1"}` - Evaluation framework, Task/Sample/Scorer

Supporting libraries:
- `{:hf_datasets_ex, "~> 0.1"}` - HuggingFace datasets loading
- `{:hf_hub, "~> 0.1"}` - HuggingFace Hub API
- `{:nx, "~> 0.9"}` - Tensor operations (replaces numpy/torch)
- `{:table_rex, "~> 4.0"}` - Terminal tables (replaces rich)

**SnakeBridge doc notes (v0.3.0)**
- Built-in manifests for `sympy`, `pylatexenc`, and `math_verify`
- Requires Snakepit runtime; docs show `{:snakepit, "~> 0.7.0"}` alongside SnakeBridge
- Python setup via `mix snakebridge.setup --venv .venv`
- Config in `config/config.exs` (load manifests, set `python_path`, pool size)
- Mix tasks: `snakebridge.manifests`, `snakebridge.manifest.compile`, `snakebridge.manifest.check`

## 4) Two Separate Concerns (Training vs Evaluation)

**Key Insight (2025-12-23):** tinker-cookbook has two distinct LLM interaction patterns:

| Component | Purpose | LLM Calls Via | Dependencies |
|-----------|---------|---------------|--------------|
| **Recipes** (95%) | Training, RL, fine-tuning | `Tinkex.SamplingClient` directly | tinkex, chz_ex |
| **Eval** (5%) | Model evaluation | inspect-ai patterns | crucible_harness, eval_ex |

### Training Stack (Phase 1 Focus)
```
SlBasic/RlBasic/etc. (recipes)
    |
    +-- Renderer (message -> tokens + weights)
    |
    +-- Tinkex.TrainingClient (forward_backward, optim_step)
    |
    +-- Tinkex.SamplingClient (generate responses)
```

### Evaluation Stack (Phase 1.5 / Separate Track)
```
TinkexCookbook.Eval.Runner
    |
    +-- CrucibleHarness.Solver.Generate
    |       |
    |       +-- CrucibleHarness.Generate protocol
    |               |
    |               +-- TinkexCookbook.Eval.TinkexGenerate (adapter, ~100 LOC)
    |                       |
    |                       +-- Tinkex.SamplingClient
    |
    +-- EvalEx.Task + EvalEx.Scorer
    |
    +-- CrucibleDatasets.MemoryDataset
```

---

## 5) Core Foundation Plan (Tinkex Cookbook)

This is the current plan to build the cookbook core (common plumbing) before
recipe-specific ports. The renderer abstraction is called out as a first-class
foundation artifact.

### Current Implementation Status (2025-12-23)

| Component | Status | Location |
|-----------|--------|----------|
| TrainOnWhat enum | DONE | `lib/tinkex_cookbook/renderers/train_on_what.ex` |
| Renderer behaviour | DONE | `lib/tinkex_cookbook/renderers/renderer.ex` |
| Types (ModelInput, Datum, TensorData) | DONE | `lib/tinkex_cookbook/types/` |
| SlBasic recipe skeleton | DONE | `lib/tinkex_cookbook/recipes/sl_basic.ex` |
| CLI utilities (CliUtils, MlLog) | DONE | `lib/tinkex_cookbook/utils/` |
| Ports/Adapters pattern | DONE | `lib/tinkex_cookbook/ports/`, `adapters/` |
| Concrete renderer (llama3) | TODO | Need to port from Python |
| NoRobots dataset builder | TODO | Need crucible_datasets wiring |
| Tinkex training integration | TODO | Currently stub in SlBasic |
| Renderer parity tests | TODO | Mirror test_renderers.py |
| TinkexGenerate adapter (eval) | TODO | ~100 LOC for eval stack |

### Phase 0: Audit and Scaffolding [COMPLETE]
- Inventory config classes, renderers, dataset builders, and shared utils from
  `tinker_cookbook/` (Python) and map to Elixir modules.
- Extract stable interfaces: Renderer, TrainOnWhat, ModelInput, Datum, TensorData.
- Capture upstream tests that will anchor parity (renderers + utils tests).

### Phase 1: Configuration + Serialization [COMPLETE]
- Define config modules with `use ChzEx.Schema` and typed fields.
- Standardize `ChzEx.asdict/1` and `ChzEx.make/2` for serialization and construction.
- Add CLI entrypoints via `ChzEx.entrypoint/2` where Python uses chz CLI.

### Phase 2: Renderer Core [IN PROGRESS]
- Port `renderers.py` with focus on:
  - `TrainOnWhat` enum (keep values identical) [DONE]
  - `Renderer` protocol and concrete renderers [BEHAVIOUR DONE, IMPLEMENTATIONS TODO]
  - Message/part structures (text, image, tool calls) [DONE]
  - Loss-weighting logic (avoid mixing NLL/cross-entropy in the same module) [DONE]
- Build parity tests that mirror `tinker_cookbook/tests/test_renderers.py`. [TODO]

### Phase 2 Addendum (2025-12-24)

- VL renderers (Qwen3VL/Qwen3VLInstruct) deferred to Phase 3.
- Tool calling should be implemented as a shared framework module and reused by Qwen3/KimiK2.
- RL training should ship sync + async paths together using shared core abstractions.
- See `docs/20251224/phase2_prerequisites/PHASE2_PART1_INFRASTRUCTURE.md` for the detailed infra plan.

### Phase 3: Tensor/Datum Plumbing [COMPLETE]
- Implement `Tinkex.TensorData` (use `docs/20251220/12b_tensor_data_implementation.ex`).
- Build ModelInput/Dataset datum helpers used by supervised/rl/preference flows.
- Ensure `Nx`-based ops match Python tensor semantics.

### Phase 4: Dataset Builders (crucible_datasets) [TODO]
- Replace Python `datasets.load_dataset` usage with `CrucibleDatasets.*`.
- Map dataset builder configs to ChzEx and connect to shared tokenizer/renderers.

### Phase 5: Training Orchestration (Tinkex clients) [TODO]
- Supervised: port `supervised/train.py` + `supervised/common.py`.
- RL: port `rl/train.py`, `rl/data_processing.py`, `rl/metrics.py`.
- Preference: port `preference/train_dpo.py`.
- Wire checkpoint + logging utilities (`checkpoint_utils.py`, `utils/ml_log.py`).

### Phase 6: Utilities + CLI/Display [PARTIAL]
- Terminal display (`display.py`, `format_colorized.py`) using IO.ANSI. [PARTIAL]
- Logging + tracing (`utils/logtree.py`, `utils/trace.py`). [DONE - MlLog]
- File/system utilities (`utils/file_utils.py`, `utils/misc_utils.py`). [DONE - CliUtils]

### Phase 1.5: Evaluation Stack (NEW)
- Wire `TinkexCookbook.Eval.TinkexGenerate` adapter implementing `CrucibleHarness.Generate`
- Create `TinkexCookbook.Eval.Runner` to orchestrate EvalEx.Task + CrucibleHarness.Solver
- Total: ~100-150 LOC of wiring code

---

## 6) Testing Strategy (Supertester Principles)

All core modules should be built with deterministic tests, using Supertester to
avoid flaky concurrency and sleeps.

- Use `Supertester.ExUnitFoundation` with full isolation for async-safe tests.
- Prefer `cast_and_sync/2` and OTP-aware assertions for GenServer interactions.
- Mock Tinkex HTTP clients and external services (no live network).
- Add property-based tests for renderer invariants (tokenization, weights, loss masks).
- Mirror Python tests where they exist; add new tests for parity gaps.

---

## 7) Notes from External Signal (Renderer Emphasis)

The renderer abstraction is a first-class product surface. The core port should:
- Preserve the original `Renderer` semantics and `TrainOnWhat` enum exactly.
- Keep loss computation modular (separate NLL vs cross-entropy helpers).
- Make renderer outputs inspectable for debugging and evaluation tooling.

---

## 8) Snakebridge Target List + Dependency Table Reference

**Dependency inventory:** `docs/20251221/DEPENDENCY_USAGE_TABLE.md`

**Snakebridge (keep Python) targets:**
- **Required (math_rl):** `sympy`, `pylatexenc`, `math_verify`
- **Optional/recipe-specific:** `textarena`, `verifiers` (+ `openai`), `chromadb` (`chroma`),
  `google-genai`, `huggingface_hub`, `inspect-ai`

Everything else is planned as native Elixir (Nx, IO.ANSI/TableRex, File/ExAws, etc.)
or is optional and currently unused in code.
