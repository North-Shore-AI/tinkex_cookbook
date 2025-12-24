# inspect-ai → Elixir Architecture

**Date:** 2025-12-23
**Status:** Master Architecture Document
**Purpose:** Define how inspect-ai functionality maps to North-Shore-AI Elixir ecosystem

---

## Key Insight: Two Separate Concerns

tinker-cookbook has TWO distinct LLM interaction patterns:

| Component | Purpose | LLM Calls Via | Uses inspect-ai? |
|-----------|---------|---------------|------------------|
| **Recipes** (95%) | Training, RL, fine-tuning | `tinker.SamplingClient` directly | NO |
| **Eval** (5%) | Model evaluation | inspect-ai's `ModelAPI` wrapping tinker | YES |

---

## What tinker-cookbook/eval Needs from inspect-ai

Only the `eval/` directory uses inspect-ai patterns:

| inspect-ai Component | Purpose | Used For |
|---------------------|---------|----------|
| `Task` + `@task` | Evaluation task definition | Defining eval benchmarks |
| `Solver` + `generate()` | Model interaction protocol | Calling models during eval |
| `Scorer` + `model_graded_qa` | Output evaluation | Grading model outputs |
| `Sample` / `Dataset` | Data structures | Loading eval samples |
| `eval_async()` | Execution orchestration | Running evaluations |
| `ModelAPI` | Model provider abstraction | Wrapping tinkex for eval |

---

## Reusability Principle

**Goal:** Make inspect-ai patterns reusable across the ecosystem, not just for tinkex_cookbook.

**Approach:** Distribute abstractions to existing libs; keep tinkex_cookbook thin.

---

## Where Each inspect-ai Component Lives

### crucible_harness (Experiment Orchestration)

| inspect-ai | Elixir Module | Why Here |
|------------|---------------|----------|
| `Solver` protocol | `CrucibleHarness.Solver` | Experiment step abstraction |
| `chain()` composition | `CrucibleHarness.Solver.Chain` | Sequential execution |
| `TaskState` | `CrucibleHarness.TaskState` | State threading |
| `Generate` protocol | `CrucibleHarness.Generate` | Abstract LLM interface |
| `generate()` solver | `CrucibleHarness.Solver.Generate` | Built-in solver |

**Reusable by:** Any ML experiment, A/B testing, ablation studies

### eval_ex (Evaluation Framework)

| inspect-ai | Elixir Module | Why Here |
|------------|---------------|----------|
| `Task` class | `EvalEx.Task` | Evaluation task definition |
| `@task` decorator | `EvalEx.Task.Registry` | Task discovery |
| `Sample` | `EvalEx.Sample` | Rich sample with metadata |
| `Scorer` protocol | `EvalEx.Scorer` | Scoring abstraction |
| `exact_match()` | `EvalEx.Scorer.ExactMatch` | Built-in scorer |
| `model_graded_qa()` | `EvalEx.Scorer.LLMJudge` | LLM-as-judge |
| Error categorization | `EvalEx.Error` | Error analysis |

**Reusable by:** Any evaluation harness, not just inspect-ai patterns

### crucible_datasets (Dataset Management)

| inspect-ai | Elixir Module | Why Here |
|------------|---------------|----------|
| `MemoryDataset` | `CrucibleDatasets.MemoryDataset` | In-memory datasets |
| `Dataset.filter()` | `CrucibleDatasets.Dataset.filter/2` | Dataset operations |
| `shuffle_choices()` | `CrucibleDatasets.Dataset.shuffle_choices/2` | MC shuffling |
| `FieldSpec` | `CrucibleDatasets.FieldMapping` | Field mapping |

**Reusable by:** Any data pipeline, not just evaluations

### tinkex (Pure Tinker Port)

| tinker | Elixir Module | Changes |
|--------|---------------|---------|
| `SamplingClient` | `Tinkex.SamplingClient` | NONE - stays pure |
| `TrainingClient` | `Tinkex.TrainingClient` | NONE - stays pure |

**No inspect-ai code here.** tinkex remains a 1:1 port.

### tinkex_cookbook (Thin Integration Layer)

| Component | Elixir Module | Purpose |
|-----------|---------------|---------|
| Recipe ports | `TinkexCookbook.Recipes.*` | Port of training recipes |
| Eval runner | `TinkexCookbook.Eval.Runner` | Wires up eval components |
| Tinkex adapter | `TinkexCookbook.Eval.TinkexGenerate` | Implements `CrucibleHarness.Generate` using tinkex |

**This is THIN** - mostly wiring, minimal logic.

---

## Visual Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      tinkex_cookbook                            │
│  ┌─────────────────────────┐  ┌──────────────────────────────┐  │
│  │  RECIPES (95%)          │  │  EVAL (5%)                   │  │
│  │  ───────────────        │  │  ─────────                   │  │
│  │  • rl_basic             │  │  • TinkexGenerate adapter    │  │
│  │  • sl_loop              │  │  • Wiring code               │  │
│  │  • math_rl              │  │                              │  │
│  │  • preference/dpo       │  │  Uses abstractions from:     │  │
│  │                         │  │  ├─ crucible_harness         │  │
│  │  Uses: tinkex directly  │  │  ├─ eval_ex                  │  │
│  │                         │  │  └─ crucible_datasets        │  │
│  └───────────┬─────────────┘  └──────────────┬───────────────┘  │
└──────────────┼───────────────────────────────┼──────────────────┘
               │                               │
               ▼                               ▼
        ┌──────────────┐            ┌─────────────────────────────┐
        │   tinkex     │            │  North-Shore-AI Ecosystem   │
        │  (pure port) │            │  ─────────────────────────  │
        │              │            │  crucible_harness:          │
        │  • Sampling  │◄───────────│    Solver, TaskState,       │
        │  • Training  │            │    Generate protocol        │
        │  • Tokenizer │            │                             │
        └──────────────┘            │  eval_ex:                   │
                                    │    Task, Sample, Scorer     │
                                    │                             │
                                    │  crucible_datasets:         │
                                    │    MemoryDataset, Filter    │
                                    └─────────────────────────────┘
```

---

## LLM Call Flow

### Recipe LLM Calls (No inspect-ai)

```
Recipe (rl_basic.ex)
    │
    └──► Tinkex.SamplingClient.sample(prompt, params)
            │
            └──► Tinker API
```

### Eval LLM Calls (inspect-ai pattern)

```
EvalEx.Runner
    │
    └──► CrucibleHarness.Solver.Generate
            │
            └──► CrucibleHarness.Generate protocol
                    │
                    └──► TinkexCookbook.Eval.TinkexGenerate (adapter)
                            │
                            └──► Tinkex.SamplingClient.sample()
                                    │
                                    └──► Tinker API
```

---

## tinkex_cookbook Stays Thin

The cookbook's eval/ directory is just:

```elixir
# lib/tinkex_cookbook/eval/tinkex_generate.ex
defmodule TinkexCookbook.Eval.TinkexGenerate do
  @behaviour CrucibleHarness.Generate

  @impl true
  def generate(messages, config) do
    # ~20 lines: convert messages, call tinkex, format response
    Tinkex.SamplingClient.sample(client, prompt, params)
  end
end
```

```elixir
# lib/tinkex_cookbook/eval/runner.ex
defmodule TinkexCookbook.Eval.Runner do
  def run(task, opts) do
    # ~30 lines: wire up EvalEx.Task + CrucibleHarness.Solver
    # All logic lives in the ecosystem libs
  end
end
```

**Total tinkex_cookbook eval code:** ~100 LOC (wiring only)

---

## Implementation Summary

| Library | What to Add | LOC | Reusable By |
|---------|-------------|-----|-------------|
| crucible_harness | Solver, TaskState, Generate | ~400 | Any experiment |
| eval_ex | Task, Sample, Scorer | ~500 | Any evaluation |
| crucible_datasets | MemoryDataset, Filter | ~300 | Any data pipeline |
| tinkex | NOTHING | 0 | N/A (stays pure) |
| tinkex_cookbook | TinkexGenerate adapter, wiring | ~100 | Just this project |
| **Total** | | **~1,300** | |

---

## Comparison: Before vs After

**Before (my incorrect plan):**
- Put ModelAPI in tinkex (wrong - tinkex should stay pure)
- ~2,600 LOC across 4 libs
- Duplicated concerns

**After (correct plan):**
- Distribute abstractions to existing libs
- tinkex_cookbook is thin adapter
- ~1,300 LOC (50% reduction)
- Clean separation of concerns

---

**Document Status:** Complete
**Last Updated:** 2025-12-23
