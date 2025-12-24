# Recipe → Dependency Mapping

**Date:** 2025-12-20
**Purpose:** Map each cookbook recipe to its dependencies and porting requirements

---

## Quick Reference Matrix

| Recipe | DONE | TRIVIAL | PORT | WRAP | SKIP |
|--------|------|---------|------|------|------|
| sl_basic | tinker | - | chz | - | - |
| sl_loop | tinker, tokenizers | torch→Nx | chz | datasets (crucible_datasets) | - |
| rl_basic | tinker | - | chz | - | - |
| rl_loop | tinker, tokenizers | torch→Nx, scipy | chz | datasets (crucible_datasets) | - |
| chat_sl | tinker, tokenizers | - | chz | datasets (crucible_datasets) | - |
| math_rl | tinker | scipy | chz | sympy, pylatexenc, math-verify | - |
| preference/dpo | tinker | - | chz | - | - |
| tool_use/search | tinker | - | chz | google.genai, chromadb (`chroma`) | - |
| multiplayer_rl | tinker | - | chz | - | textarena |
| code_rl | tinker | - | chz | - | - |
| verifiers_rl | tinker | - | chz | - | verifiers |

---

## Detailed Recipe Breakdown

### 1. sl_basic.py (Supervised Learning Basic)

**Location:** `recipes/sl_basic.py`
**Purpose:** Introduction to supervised learning

| Dependency | Category | Elixir Mapping | Notes |
|------------|----------|----------------|-------|
| `chz` | PORT | Elixir structs | @chz.chz decorator |
| `asyncio` | TRIVIAL | `Task.async_stream/2` | Built-in |
| `tinker` | DONE | tinkex | 1:1 parity |

**Effort:** Trivial once chz ported

---

### 2. sl_loop.py (Supervised Learning Loop)

**Location:** `recipes/sl_loop.py`
**Purpose:** Full supervised learning training loop

| Dependency | Category | Elixir Mapping | Notes |
|------------|----------|----------------|-------|
| `chz` | PORT | Elixir structs | Config management |
| `datasets` | DONE | `crucible_datasets` v0.4.1 | HF Hub + local ops |
| `tinker` | DONE | tinkex | Training API |
| `torch` | TRIVIAL | Nx | Tensor ops only |
| `logging` | TRIVIAL | Logger | Built-in |

**Effort:** Medium

---

### 3. rl_basic.py (Reinforcement Learning Basic)

**Location:** `recipes/rl_basic.py`
**Purpose:** Introduction to RL training

| Dependency | Category | Elixir Mapping | Notes |
|------------|----------|----------------|-------|
| `chz` | PORT | Elixir structs | @chz.chz decorator |
| `asyncio` | TRIVIAL | Task | Built-in |
| `tinker` | DONE | tinkex | 1:1 parity |

**Effort:** Trivial once chz ported

---

### 4. rl_loop.py (Reinforcement Learning Loop)

**Location:** `recipes/rl_loop.py`
**Purpose:** Full RL training loop with grading

| Dependency | Category | Elixir Mapping | Notes |
|------------|----------|----------------|-------|
| `chz` | PORT | Elixir structs | Config management |
| `datasets` | DONE | `crucible_datasets` v0.4.1 | HF Hub + local ops |
| `tinker` | DONE | tinkex | Training API |
| `torch` | TRIVIAL | Nx | Tensor ops |
| `scipy.signal.lfilter` | TRIVIAL | Enum.reduce | Discounted rewards |

**Effort:** Medium

---

### 5. chat_sl/ (Chat Supervised Learning)

**Location:** `recipes/chat_sl/train.py`, `chat_datasets.py`
**Purpose:** Chat-based supervised fine-tuning

| Dependency | Category | Elixir Mapping | Notes |
|------------|----------|----------------|-------|
| `chz` | PORT | Elixir structs | Config |
| `datasets` | DONE | `crucible_datasets` v0.4.1 | HF Hub + streaming |
| `tinker` | DONE | tinkex | Training API |
| `tokenizers` | DONE | {:tokenizers} | In tinkex |

**HuggingFace Datasets Used:**
- `allenai/tulu-3-sft-mixture`
- `HuggingFaceH4/MATH-500`
- `openai/gsm8k`
- `Anthropic/hh-rlhf`
- And more...

**Effort:** Medium (HF Hub wrapper needed)

---

### 6. math_rl/ (Math Reinforcement Learning)

**Location:** `recipes/math_rl/train.py`, `math_grading.py`
**Purpose:** Math problem solving via RL

| Dependency | Category | Elixir Mapping | Notes |
|------------|----------|----------------|-------|
| `chz` | PORT | Elixir structs | Config |
| `sympy` | WRAP | Snakepit | Symbolic math |
| `pylatexenc` | WRAP | Pythonx/Snakepit | LaTeX parsing |
| `math_verify` | WRAP | Snakepit | Answer verification |
| `scipy.signal.lfilter` | TRIVIAL | Enum.reduce | Discounted rewards |
| `tinker` | DONE | tinkex | Training API |

**Effort:** High (requires Snakepit math stack)

---

### 7. preference/dpo/ (Direct Preference Optimization)

**Location:** `recipes/preference/dpo/train.py`
**Purpose:** DPO training for preference learning

| Dependency | Category | Elixir Mapping | Notes |
|------------|----------|----------------|-------|
| `chz` | PORT | Elixir structs | Config |
| `tinker` | DONE | tinkex | Training API |

**Effort:** Trivial once chz ported (training is server-side)

---

### 8. tool_use/search/ (Tool Use with Search)

**Location:** `recipes/tool_use/search/train.py`, `embedding.py`, `tools.py`
**Purpose:** Training models to use search tools

| Dependency | Category | Elixir Mapping | Notes |
|------------|----------|----------------|-------|
| `chz` | PORT | Elixir structs | Config |
| `google.genai` | WRAP | HTTP client (Req) | Gemini embeddings |
| `chromadb` | WRAP | `chroma` (HTTP client) | Vector DB |
| `tinker` | DONE | tinkex | Training API |

**Effort:** Medium (HTTP client wrappers)

---

### 9. multiplayer_rl/text_arena/ (Multiplayer RL)

**Location:** `recipes/multiplayer_rl/text_arena/train.py`, `env.py`
**Purpose:** Multi-agent game training (TicTacToe)

| Dependency | Category | Elixir Mapping | Notes |
|------------|----------|----------------|-------|
| `chz` | PORT | Elixir structs | Config |
| `textarena` | SKIP | - | Optional, TicTacToe only |
| `tinker` | DONE | tinkex | Training API |

**Effort:** Trivial if skipping textarena, Medium if wrapping

---

### 10. code_rl/ (Code RL)

**Location:** `recipes/code_rl/train.py`
**Purpose:** Code generation via RL

| Dependency | Category | Elixir Mapping | Notes |
|------------|----------|----------------|-------|
| `chz` | PORT | Elixir structs | Config |
| `tinker` | DONE | tinkex | Training API |

**Effort:** Trivial once chz ported

---

### 11. verifiers_rl/ (Verifiers RL)

**Location:** `recipes/verifiers_rl/train.py`
**Purpose:** RL with verifier environments

| Dependency | Category | Elixir Mapping | Notes |
|------------|----------|----------------|-------|
| `chz` | PORT | Elixir structs | Config |
| `verifiers` | SKIP | - | Optional feature |
| `tinker` | DONE | tinkex | Training API |

**Effort:** Trivial if skipping verifiers

---

## Dependency Summary by Category

### DONE (tinkex provides) - 0 days

| Dependency | Recipes Using |
|------------|---------------|
| `tinker` | All 11 recipes |
| `tokenizers` | sl_loop, rl_loop, chat_sl |
| `tiktoken` | Various tokenization |

### TRIVIAL (Nx/Native Elixir) - 1-2 days

| Dependency | Elixir | Recipes Using |
|------------|--------|---------------|
| `numpy` (18 funcs) | `Nx` | 3 recipes |
| `scipy.signal.lfilter` | `Enum.reduce` | rl_loop, math_rl |
| `torch` (tensors) | `Nx` | sl_loop, rl_loop |
| `asyncio` | `Task` | sl_basic, rl_basic |
| `logging` | `Logger` | All |

### PORT (Native Elixir) - 3-5 days

| Dependency | Elixir | Recipes Using |
|------------|--------|---------------|
| `chz` | Elixir structs + Ecto.Changeset | **All 11 recipes** |
| `datasets` (HF + local) | `crucible_datasets` v0.4.1 | sl_loop, rl_loop, chat_sl |

### WRAP (Pythonx/Snakepit) - 2-3 weeks

| Dependency | Strategy | Recipes Using |
|------------|----------|---------------|
| `sympy` | Snakepit | math_rl only |
| `pylatexenc` | Snakepit | math_rl only |
| `math_verify` | Snakepit | math_rl only |
| `google.genai` | HTTP client | tool_use/search |
| `chromadb` | `chroma` HTTP client | tool_use/search |

### SKIP (Optional) - 0 days

| Dependency | Reason | Recipe |
|------------|--------|--------|
| `textarena` | Only for multiplayer TicTacToe | multiplayer_rl |
| `verifiers` | Optional install (`[verifiers]`) | verifiers_rl |

---

## Implementation Priority Order

### Phase 1: Enable 9 of 11 Recipes (Weeks 1-2)

Port `chz` to Elixir structs → unlocks:
1. sl_basic ✓
2. sl_loop ✓ (+ Nx for torch)
3. rl_basic ✓
4. rl_loop ✓ (+ scipy replacement)
5. preference/dpo ✓
6. code_rl ✓
7. multiplayer_rl (without textarena) ✓
8. verifiers_rl (without verifiers) ✓
9. chat_sl ✓ (datasets already covered by crucible_datasets)

### Phase 2: Enable math_rl (Week 3)

Set up Snakepit math stack → unlocks:
9. math_rl ✓

### Phase 3: Enable tool_use (Week 4)

Add HTTP client wrappers → unlocks:
10. tool_use/search ✓

---

## Blocking Dependencies

| Dependency | Blocks | Workaround |
|------------|--------|------------|
| `chz` | All 11 recipes | Port first |
| `sympy` stack | math_rl only | Other recipes work without |
| `textarena` | multiplayer_rl | Can implement TicTacToe natively |
| `verifiers` | verifiers_rl | Recipe can be skipped |

---

**Document Status:** Complete
**Last Updated:** 2025-12-20
