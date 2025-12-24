# Phase 1 Agent Prompt (Full-Slice, TDD, No Warnings)

**Updated:** 2025-12-23

You are an implementation agent. Your mission is to complete **Phase 1** for
`tinkex_cookbook` with a full-slice reference recipe port. All work must be
TDD, all tests pass, and the phase ends with a runnable `sl_basic` equivalent.

This prompt is the source of truth for what to read, build, and verify.

> **Status Update (2025-12-23):** Much of Phase 1 is complete. Focus on:
> 1. Concrete Llama3 renderer implementation
> 2. NoRobots dataset builder wiring
> 3. Tinkex training integration (replace stubs)
> 4. Renderer parity tests
> 5. TinkexGenerate adapter for eval stack (~100 LOC)

---

## Goal

Deliver a **working Elixir port** of the `sl_basic` recipe with full plumbing:
configs, renderers, dataset builder, training orchestration, and logging.

**Quality bar:**
- All tests pass.
- No warnings (compiler, Credo, Dialyzer).
- No runtime errors.
- Deterministic tests (no sleeps).

---

## Required Reading (Do This First)

### Repo Instructions
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/AGENTS.md`
- `/home/home/p/g/n/supertester/README.md`

### Phase Plans + Inventory
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251221/COOKBOOK_CORE_FOUNDATION.md`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251221/PHASE1_FOUNDATION_SLICE.md`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251221/DEPENDENCY_USAGE_TABLE.md`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251221/REMAINING_WORK.md`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251221/SNAKEBRIDGE_LIBS.md`

### Core Porting References
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251220/README.md`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251220/12_torch_transformers_actual_usage.md`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251220/12b_tensor_data_implementation.ex`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251220/13_numpy_scipy_actual_usage.md`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251220/15_utility_libs.md`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251220/11_tinker_to_tinkex.md`

### Python Source (Behavior Spec)
Read these files in `tinker-cookbook` (Python) to mirror behavior:
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/tinker-cookbook/tinker_cookbook/recipes/sl_basic.py`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/tinker-cookbook/tinker_cookbook/recipes/chat_sl/chat_datasets.py`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/tinker-cookbook/tinker_cookbook/supervised/train.py`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/tinker-cookbook/tinker_cookbook/supervised/data.py`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/tinker-cookbook/tinker_cookbook/supervised/types.py`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/tinker-cookbook/tinker_cookbook/supervised/common.py`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/tinker-cookbook/tinker_cookbook/renderers.py`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/tinker-cookbook/tinker_cookbook/tokenizer_utils.py`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/tinker-cookbook/tinker_cookbook/image_processing_utils.py`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/tinker-cookbook/tinker_cookbook/utils/ml_log.py`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/tinker-cookbook/tinker_cookbook/display.py`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/tinker-cookbook/tinker_cookbook/cli_utils.py`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/tinker-cookbook/tinker_cookbook/model_info.py`

### Python Tests to Mirror
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/tinker-cookbook/tinker_cookbook/tests/test_renderers.py`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/tinker-cookbook/tinker_cookbook/tests/test_utils.py`

---

## Constraints (Must Follow)

- Use **ChzEx** for config schemas (no ad-hoc structs).
- Use **CrucibleDatasets** for dataset loading (no Python datasets).
- Use **CrucibleHarness** for eval solver/generate patterns (inspect-ai parity).
- Use **EvalEx** for eval task/sample/scorer patterns.
- Use **SnakeBridge** only for Python libs when explicitly needed (Phase 1 likely not).
- No atom creation from user input (CLI keys stay strings).
- Use **Nx** for numeric ops instead of torch.
- Tests must be deterministic (no sleeps, use Supertester sync helpers).
- Keep all edits ASCII only.

---

## Phase 1 Deliverables

1. **Runnable `sl_basic` equivalent in Elixir** [SKELETON DONE]
   - CLI entrypoint using ChzEx [DONE]
   - Dataset builder using CrucibleDatasets [TODO]
   - Renderer pipeline with `TrainOnWhat` semantics [DONE]
   - Training loop calling Tinkex clients [TODO - stub exists]

2. **Renderer Core + Tests** [PARTIAL]
   - `TrainOnWhat` enum matches Python values exactly [DONE]
   - Renderer protocol and at least one concrete renderer required for the chosen model [BEHAVIOUR DONE, llama3 TODO]
   - Loss weighting logic (keep NLL and cross-entropy logic cleanly separated) [DONE]
   - Tests derived from Python `test_renderers.py` (use deterministic tokenizer stubs) [TODO]

3. **Config + Serialization** [DONE]
   - `ChzEx.Schema` modules for supervised configs and dataset builders [DONE]
   - `ChzEx.make/2` and `ChzEx.asdict/1` used consistently [DONE]

4. **Logging/CLI Utilities** [DONE]
   - Port minimal `cli_utils` behavior for log dir handling [DONE]
   - Port `utils/ml_log.py` behavior with IO.ANSI/TableRex [DONE]

5. **Evaluation Stack Wiring (NEW)** [TODO]
   - `TinkexCookbook.Eval.TinkexGenerate` implementing `CrucibleHarness.Generate`
   - `TinkexCookbook.Eval.Runner` wiring EvalEx.Task + CrucibleHarness.Solver
   - Total: ~100-150 LOC

---

## TDD Workflow (Required)

Follow this exact loop for each module:
1. Write tests first (mirror Python tests or specs).
2. Implement minimal code to pass tests.
3. Refactor with guardrails (no behavior drift).

---

## Recommended Build Order (Updated 2025-12-23)

> Items marked [DONE] are complete. Focus on remaining items.

1. **Renderer primitives** [MOSTLY DONE]
   - `TrainOnWhat` enum [DONE]
   - message/part structs (text/image/tool calls) [DONE]
   - base renderer protocol + minimal concrete renderer [BEHAVIOUR DONE, llama3 TODO]
   - tests for tokenization + loss masks [TODO]

2. **TensorData + datum helpers** [DONE]
   - Use `docs/20251220/12b_tensor_data_implementation.ex` as the base

3. **Dataset builder** [TODO]
   - `NoRobotsBuilder` from `chat_sl/chat_datasets.py`
   - `ChatDatasetBuilderCommonConfig`
   - Wire to CrucibleDatasets

4. **Supervised train orchestration** [TODO]
   - `supervised/train.py` behavior mapped to Elixir
   - Make training client calls mockable
   - Replace stubs with actual Tinkex client calls

5. **sl_basic recipe** [SKELETON DONE]
   - Blueprint config builder [DONE]
   - CLI entrypoint and log path handling [DONE]
   - Complete training loop [TODO]

6. **Eval stack wiring (NEW)** [TODO]
   - `TinkexCookbook.Eval.TinkexGenerate` adapter
   - `TinkexCookbook.Eval.Runner` orchestration

---

## Test Guidance

- Use **Supertester.ExUnitFoundation** for isolation.
- Mock Tinkex HTTP clients; no real network.
- Stub tokenizers to avoid external model downloads.
- Keep tests small and deterministic.

---

## Quality Gates (Non-Negotiable)

Run these before declaring Phase 1 complete:

```bash
mix format
mix test
mix credo --strict
mix dialyzer
```

No warnings, no errors.

---

## Forbidden Actions

- Do not add new dependencies beyond the approved set:
  `crucible_datasets`, `crucible_harness`, `eval_ex`, `chz_ex`, `snakebridge`, `tinkex`
  (unless explicitly approved).
- Do not run network calls in tests.
- Do not introduce Python runtime calls for Phase 1.
