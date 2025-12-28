# Testing, Parity, and Migration Plan

Date: 2025-12-26
Status: Draft
Owner: North-Shore-AI

## 1) Purpose

Define the migration steps for the ground-up rewrite and the parity strategy to validate it against Python `tinker-cookbook`.

## 2) Migration Phases

### Phase 0: Scaffold
- Add facade modules and recipe behaviour.
- Add CrucibleTrain dependency and remove duplicated training infra.
- Introduce adapter skeletons for Crucible ports.

### Phase 1: Core Recipes
- Rebuild `sl_basic` via IR spec + pipeline.
- Implement parity harness for sl_basic.
- Validate dataset ordering and renderer parity.

### Phase 2: Expand Recipes
- Rebuild `rl_basic`, `chat_sl`, `dpo`, `code_rl` via facade.
- Add recipe parity harness scripts as needed.

### Phase 3: Optional Recipes
- Port recipes that require optional Python deps (textarena, verifiers, tool_use/search).

## 3) Parity Protocol

Use the existing protocol in:
- `docs/20251224/recipe_parity/RECIPE_PARITY_PROTOCOL.md`

Key requirements:
- Config parity (ChzEx vs chz).
- Dataset ordering parity (deterministic shuffles).
- Renderer/tokenizer parity.
- Datum construction parity.
- Training step and metrics parity (tolerance allowed).

## 4) Required Artifacts

- `config.json`
- `metrics.jsonl`
- dataset snapshot
- rendered samples
- first batch payload hash

## 5) Test Strategy

- Unit tests for facade, recipe specs, and adapter wiring.
- Mox mocks for all external calls.
- Deterministic tests, no sleeps.

## 6) Acceptance Criteria

- All recipes run through facade.
- Parity harness passes for core recipes.
- No warnings or errors in quality gates.

