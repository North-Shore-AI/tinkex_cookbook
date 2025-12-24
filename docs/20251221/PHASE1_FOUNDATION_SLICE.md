# Phase 1 - Foundation + Full-Slice Reference Experiment

**Goal:** establish the core plumbing with one end-to-end experiment ported
from `tinker_cookbook` as the reference slice for all other recipes.

**Reference experiment:** `tinker_cookbook/recipes/sl_basic.py`

**Updated:** 2025-12-23

Why `sl_basic`:
- Minimal surface area but still exercises **configs**, **datasets**, **renderers**,
  **training loop**, and **logging**.
- Uses `TrainOnWhat` and renderer flow (the abstraction called out as critical).
- Gives a clean template for CLI + blueprint workflows.

---

## Current Status (2025-12-23)

| Component | Status | Notes |
|-----------|--------|-------|
| TrainOnWhat enum | DONE | Exact parity with Python |
| Renderer behaviour | DONE | `build_supervised_example`, `build_generation_prompt` |
| Types (ModelInput, Datum, TensorData) | DONE | Core structs implemented |
| SlBasic recipe skeleton | DONE | ChzEx config wired, training is stub |
| CLI utilities | DONE | CliUtils, MlLog implemented |
| Ports/Adapters | DONE | HubClient, DatasetStore, etc. |
| Eval ecosystem deps | DONE | crucible_harness, eval_ex, crucible_datasets added |
| Concrete renderer (llama3) | TODO | Only behaviour exists |
| NoRobots dataset builder | TODO | Needs crucible_datasets wiring |
| Tinkex training integration | TODO | Currently just logs |
| Renderer parity tests | TODO | Need tests from test_renderers.py |
| TinkexGenerate adapter | TODO | ~100 LOC for eval stack |

---

## Scope (Phase 1)

### A) Core Plumbing (must ship)
- Config schemas using **ChzEx** for:
  - `supervised/train.py` config [DONE]
  - `supervised/types.py` (dataset builder configs) [DONE]
  - `chat_sl/chat_datasets.py` (NoRobotsBuilder) [PARTIAL]
- Renderer core:
  - `TrainOnWhat` enum (match values exactly) [DONE]
  - Base renderer protocol [DONE]
  - One concrete renderer needed for the chosen model [TODO - llama3]
  - Loss masking and weights with clean NLL vs cross-entropy separation [DONE]
- Dataset path:
  - `CrucibleDatasets` integration for the NoRobots dataset [TODO]
  - Minimal dataset builder pipeline [TODO]
- Training orchestration:
  - `supervised/train.py` orchestration calling Tinkex clients [TODO - stub exists]
  - Logging/metrics via `utils/ml_log.py` and `display.py` [DONE]

### B) Reference CLI [DONE]
- `sl_basic` equivalent in Elixir:
  - blueprint builder [DONE]
  - `ChzEx.entrypoint/2` or `ChzEx.Blueprint` pipeline [DONE]
  - explicit log dir handling (ask/confirm) [DONE]

### C) Evaluation Stack (NEW - Phase 1.5)
- Dependencies added: crucible_harness, eval_ex, crucible_datasets [DONE]
- `TinkexCookbook.Eval.TinkexGenerate` adapter [TODO]
- `TinkexCookbook.Eval.Runner` wiring [TODO]

---

## Deliverables

- One runnable Elixir example equivalent to `sl_basic` [SKELETON DONE]
- Core renderer module with parity tests (ported from Python test_renderers) [BEHAVIOUR DONE, TESTS TODO]
- Dataset builder + tokenizer integration using `CrucibleDatasets` [TODO]
- Supervised training orchestrator using Tinkex clients [STUB EXISTS]
- Deterministic tests using Supertester (no sleeps) [TODO]
- Eval stack wiring (TinkexGenerate adapter) [TODO - NEW]

---

## TDD + Supertester Notes

- Use `Supertester.ExUnitFoundation` for async-safe isolation
- Replace any `Process.sleep/1` with deterministic sync
- Mock Tinkex clients (no real network)
- Add property tests for renderer invariants:
  - token length, mask length, weights, and stop tokens

---

## Acceptance Criteria

- `sl_basic` runs end-to-end (no errors, deterministic outputs)
- Renderer tests pass with parity vs Python reference
- All unit tests are async-safe and deterministic

