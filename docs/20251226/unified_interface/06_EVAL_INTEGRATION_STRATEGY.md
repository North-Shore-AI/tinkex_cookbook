# Evaluation Integration Strategy

Date: 2025-12-26
Status: Draft
Owner: North-Shore-AI

## 1) Purpose

Define how evaluation flows integrate with the unified facade while keeping `tinkex_cookbook` thin and leveraging EvalEx and CrucibleHarness.

## 2) Scope from Python Cookbook

Python `tinker-cookbook` uses inspect-ai only in the eval path:
- `tinker_cookbook/eval/*`
- Inline evals in `recipes/chat_sl/train.py`

No inspect-ai usage exists in core training loops.

## 3) Elixir Mapping

- `crucible_harness` provides Solver and Generate behaviour.
- `eval_ex` provides Task, Sample, Scorer abstractions.
- `crucible_datasets` provides MemoryDataset and dataset utilities.
- `tinkex_cookbook` provides only a thin adapter and wiring.

## 4) Thin Adapter Contract

Required modules in `tinkex_cookbook`:

- `TinkexCookbook.Eval.TinkexGenerate`
  - Implements `CrucibleHarness.Generate` using `Tinkex.SamplingClient`.

- `TinkexCookbook.Eval.Runner`
  - Wires `EvalEx.Task` + `CrucibleHarness.Solver.Generate`.
  - Runs tasks sequentially (async optional later).

## 5) Facade Integration

The unified facade should support optional evaluation:

```elixir
TinkexCookbook.Runtime.eval(recipe_module, opts)
  -> build EvalEx.Task
  -> run via TinkexCookbook.Eval.Runner
```

Inline evals (chat_sl) should map to an opt-in facade flag.

## 6) Parity Notes

Known gaps vs inspect-ai:
- No async eval runner (currently sequential).
- No inspect-evals task loader.
- No tool calling support in eval adapter.

These are acceptable for current cookbook usage and can be added later.

## 7) Acceptance Criteria

- Eval wiring stays within `tinkex_cookbook/eval`.
- Eval code remains thin (<200 LOC).
- No inspect-ai dependency required.

