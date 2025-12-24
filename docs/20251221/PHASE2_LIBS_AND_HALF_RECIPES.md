# Phase 2 - Core Libs + ~Half the Recipes

**Goal:** build the core library stack with ChzEx + HfDatasetsEx + HfHub + SnakeBridge,
then port roughly half of the cookbook recipes.

**Phase 1 output** is assumed complete and used as the template.

---

## A) Library Integration (Phase 2 core)

### 1) ChzEx
- Port remaining config schemas across supervised, RL, preference, distillation.
- Standardize `ChzEx.asdict/1` and `ChzEx.make/2` as the serialization layer.

### 2) HfDatasetsEx + HfHub (HF)
- Replace `datasets.load_dataset(...)` with `HfDatasetsEx.load_dataset/2`.
- Use `HfHub` for repo metadata and file downloads when needed.
- Wire `HF_TOKEN` for gated datasets.

### 3) SnakeBridge (Python libs)
- Use built-in manifests for:
  - `sympy`
  - `pylatexenc`
  - `math_verify`
- Add custom manifests for optional libs as needed (see `SNAKEBRIDGE_LIBS.md`).

---

## B) Recipe Targets (aim ~half)

**Phase 1 already ports:** `sl_basic`

**Phase 2 target set (8-9 recipes total including Phase 1):**
- `sl_loop`
- `rl_basic`
- `rl_loop`
- `chat_sl`
- `preference/dpo`
- `code_rl`
- `prompt_distillation`
- `distillation/on_policy_distillation`
- (optional stretch) `distillation/off_policy_reasoning`

**Rationale:** these are core, broadly used, and avoid optional Python libs
(`textarena`, `verifiers`, `chromadb`, `google-genai`, `openai`).

---

## C) Common Plumbing to Finish in Phase 2

- Dataset builders for chat + RL + preference
- Tokenizer utilities + renderer coverage for major model families (Llama3, Qwen3, DeepSeek)
  - This is formatting + tokenization only; no local inference (no llama.cpp needed).
- `TensorData` + Datum builders for training + sampling flows
- Logging, checkpointing, and trace utilities
  - Defer `chromadb` until tool_use/search is in scope; prefer `chroma` (HTTP client).

---

## D) Testing Commitments

- Full TDD with Supertester:
  - `Supertester.ExUnitFoundation` + isolation
  - deterministic sync helpers, no sleeps
- Mock all Tinkex client calls
- Port Python tests where possible (`test_renderers`, `test_utils`)

---

## Phase 2 Exit Criteria

- ~Half of recipes runnable (including Phase 1)
- Configs + renderers + datasets + training flows stable
- SnakeBridge wired for math libs (even if math_rl is deferred)
