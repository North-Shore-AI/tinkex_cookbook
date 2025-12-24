# Tinkex Cookbook Agent Guide

**Updated:** 2025-12-23

This repo's core plumbing is built on these libraries:
- **ChzEx** - Configuration + CLI
- **HfDatasetsEx** + **HfHub** - HuggingFace datasets/hub
- **SnakeBridge** - Python interop (math grading)
- **CrucibleHarness** + **EvalEx** + **CrucibleDatasets** - Evaluation ecosystem

Use this guide to stay consistent when porting the Python tinker-cookbook into Elixir.

> **Phase 1 Completion:** See `docs/20251223/PHASE1_COMPLETION_AGENT_PROMPT.md` for detailed
> instructions on completing the remaining Phase 1 work.

---

## 1) ChzEx (Configuration + CLI)

**Purpose:** typed config schemas, defaults, validation, CLI parsing.

**Do:**
- Define configs with `use ChzEx.Schema` + `chz_schema`.
- Use `field/3` with explicit types and defaults.
- Use `ChzEx.make/2` for construction from maps.
- Use `ChzEx.asdict/1` for serialization.
- Use `ChzEx.entrypoint/2` for CLI entrypoints.

**Don't:**
- Don't create atoms from user input; keep CLI keys as strings.
- Don't bypass `changeset/2` for validation-heavy configs.

**Pin:** `{:chz_ex, "~> 0.1.2"}`

---

## 2) HfDatasetsEx (HuggingFace datasets for Elixir)

**Purpose:** replace Python `datasets` library for dataset loading and operations.

**Do:**
- Use `HfDatasetsEx.load_dataset/2` for loading by repo_id.
- Use `DatasetDict["train"]` / `DatasetDict["test"]` for split access.
- Use dataset operations: `shuffle`, `take`, `skip`, `select`, `map`, `filter`.
- Use `Dataset.to_list/1` to convert to list of maps.
- Use `HfDatasetsEx.Dataset.from_list/1` for JSONL data.
- Use `HF_TOKEN` for gated datasets.

**Don't:**
- Don't reintroduce Python `datasets` in Elixir code.

**Pin:** `{:hf_datasets_ex, "~> 0.1"}` or path dependency

---

## 3) SnakeBridge (Python libs via Snakepit)

**Purpose:** manifest-driven, curated Python calls for libs we keep in Python.

**Built-in manifests:** `sympy`, `pylatexenc`, `math_verify`

**Do:**
- Configure manifests in `config/config.exs`.
- Use `mix snakebridge.setup --venv .venv` for Python environment.
- Keep `allow_unsafe: false` unless explicitly approved.
- Prefer compile-time generation for stable APIs.

**Don't:**
- Don't auto-expose large Python modules; use minimal manifests.
- Don't call Python directly without a manifest.

**Pins:**
- `{:snakebridge, "~> 0.3.0"}`
- Ensure `:snakepit` runtime is available per SnakeBridge docs.

---

## 4) Ports & Adapters (External Services)

**Purpose:** standardize external service access behind ports for easy swapping.

**Do:**
- Use `TinkexCookbook.Ports` to resolve adapters at the composition root.
- Call services through ports (`Ports.VectorStore`, `Ports.EmbeddingClient`, etc.).
- Provide adapters in `lib/tinkex_cookbook/adapters/*`.

**Don't:**
- Don't call external clients (OpenAI/Gemini/Chroma/HF) directly in recipes.
- Don't hardcode adapters; wire them via `Ports.new/1` overrides or app config.

**Vector DB:** prefer `{:chroma, "~> 0.1.2"}` for ChromaDB.

**LLM Adapters:** prefer `Adapters.LLMClient.Codex` and `Adapters.LLMClient.ClaudeAgent`
for CLI-backed agents; use `output_schema` for structured outputs.

---

## 5) Evaluation Ecosystem (inspect-ai parity)

**Purpose:** Provide evaluation infrastructure matching inspect-ai patterns.

**Libraries:**
- `CrucibleHarness` - Solver/Generate/TaskState for composable execution
- `EvalEx` - Task/Sample/Scorer for evaluation framework
- `CrucibleDatasets` - MemoryDataset and dataset operations

**Do:**
- Use `CrucibleHarness.Generate` behaviour for LLM backends.
- Use `CrucibleHarness.Solver` behaviour for composable steps.
- Use `CrucibleHarness.TaskState` for state threading.
- Use `EvalEx.Task` for evaluation task definitions.
- Use `EvalEx.Scorer` for output scoring.
- Use `CrucibleDatasets.MemoryDataset` for in-memory datasets.

**Don't:**
- Don't call LLM APIs directly in eval code; go through Generate adapter.
- Don't bypass the Solver pattern for multi-step evaluations.

**Pins:**
- `{:crucible_harness, "~> 0.3.1"}`
- `{:eval_ex, "~> 0.1.1"}`
- `{:crucible_datasets, "~> 0.5.1"}`

**Eval Stack Architecture:**
```
TinkexCookbook.Eval.Runner
    |
    +-- CrucibleHarness.Solver.Generate
    |       |
    |       +-- TinkexCookbook.Eval.TinkexGenerate (adapter)
    |               |
    |               +-- Tinkex.SamplingClient
    |
    +-- EvalEx.Task + EvalEx.Scorer
    |
    +-- CrucibleDatasets.MemoryDataset
```

---

## Testing Principles

All core plumbing must be TDD:
- Write failing tests first, then implement minimal code to pass.
- Avoid sleeps; use deterministic sync helpers.
- Mock Tinkex clients and Python calls (no network in tests).
- Add property tests for renderer invariants.

---

## Phase 1 Status: COMPLETE

Phase 1 delivers a working `sl_basic` recipe in Elixir with full test coverage.

**Completed Components:**

| Component | Location | Tests |
|-----------|----------|-------|
| TrainOnWhat enum | `lib/tinkex_cookbook/renderers/train_on_what.ex` | 17 tests |
| Renderer behaviour | `lib/tinkex_cookbook/renderers/renderer.ex` | 15 tests |
| Llama3 renderer | `lib/tinkex_cookbook/renderers/llama3.ex` | 19 tests |
| Types (ModelInput, Datum, TensorData) | `lib/tinkex_cookbook/types/*.ex` | 21 tests |
| SlBasic recipe | `lib/tinkex_cookbook/recipes/sl_basic.ex` | 5 tests |
| CLI utilities | `lib/tinkex_cookbook/utils/*.ex` | 18 tests |
| NoRobots dataset | `lib/tinkex_cookbook/datasets/no_robots.ex` | 10 tests |
| Supervised training | `lib/tinkex_cookbook/supervised/train.ex` | 12 tests |
| TinkexGenerate adapter | `lib/tinkex_cookbook/eval/tinkex_generate.ex` | 6 tests |
| Eval Runner | `lib/tinkex_cookbook/eval/runner.ex` | 11 tests |

**Quality Gates:**
- 165 tests passing
- Zero compiler warnings
- Credo strict: no issues
- Dialyzer: no type errors

---

## Phase 1 Implementation Notes

### Llama3 Renderer

The Llama3 renderer (`lib/tinkex_cookbook/renderers/llama3.ex`) implements the full Llama 3 chat template:

```
<|begin_of_text|><|start_header_id|>{role}<|end_header_id|>

{content}<|eot_id|>
```

Key callbacks:
- `init/1` - Initialize with tokenizer
- `bos_tokens/1` - Returns `<|begin_of_text|>` tokens
- `stop_sequences/1` - Returns `<|eot_id|>` tokens
- `render_message/4` - Renders a single message
- `parse_response/2` - Parses sampling response

### NoRobots Dataset Builder

The NoRobots dataset builder (`lib/tinkex_cookbook/datasets/no_robots.ex`) provides:
- `load/1` - Load dataset from HuggingFace
- `sample_to_messages/1` - Extract messages from samples
- `build_datum/4` - Convert sample to training Datum
- `build_datums/4` - Batch conversion
- `create_supervised_dataset/2` - Create SupervisedDataset

### Training Module

The training module (`lib/tinkex_cookbook/supervised/train.ex`) provides:
- `TrainConfig` - Configuration struct with defaults
- `batch_datums/2` - Batch datums for training
- `training_step/3` - Single forward_backward + optim_step
- `compute_lr/4` - Learning rate scheduling (linear, constant, cosine)
- `run_epoch/4` - Run single training epoch
- `run/3` - Full training loop

### TinkexGenerate Adapter

The TinkexGenerate adapter (`lib/tinkex_cookbook/eval/tinkex_generate.ex`) implements
`CrucibleHarness.Generate` behaviour for Tinkex sampling:
- `generate/2` - Generate text from messages
- `build_model_input/3` - Build ModelInput from messages
- `parse_response/3` - Parse sample response

### Eval Runner

The evaluation runner (`lib/tinkex_cookbook/eval/runner.ex`) provides:
- `run/2` - Run evaluation on samples
- `run_sample/2` - Evaluate single sample
- `create_messages/1` - Create messages from sample
- `score_results/2` - Score with exact_match/contains
- `compute_metrics/1` - Compute accuracy metrics
- `run_task/2` - Run EvalEx task

---

## Test Support Modules

Mock modules for testing without network access:
- `test/support/mock_tokenizer.ex` - Deterministic tokenizer
- `test/support/mock_tinkex.ex` - Mock TrainingClient/SamplingClient
