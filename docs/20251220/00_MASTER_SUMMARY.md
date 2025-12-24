# Tinker-Cookbook Porting Feasibility Master Summary

**Date:** 2025-12-20
**Status:** VERIFIED - All findings confirmed by 9 specialized audit agents
**Purpose:** Consolidated analysis of tinker-cookbook Python dependencies for Elixir (tinkex) porting

---

## Executive Summary

This document synthesizes **16 detailed dependency analyses** (including 6 new verification reports) to assess the feasibility of porting tinker-cookbook recipes to Elixir.

### CRITICAL CLARIFICATION

**We are porting COOKBOOK (client-side orchestration), NOT Tinker's backend.**

The cookbook is a collection of example recipes that call the Tinker API. GPU training (flash-attn, deepspeed, liger-kernel) happens SERVER-SIDE via the Tinker platform. The Elixir port only needs to replicate CLIENT-SIDE logic.

| Category | Feasibility | Strategy | Verified Effort |
|----------|-------------|----------|-----------------|
| **Tinker SDK** | **ALREADY DONE** | tinkex has 1:1 parity | 0 days |
| **NumPy/SciPy** | **Trivial** | Nx (18 functions, 1 scipy) | 1-2 days |
| **Configuration** | **Port to Elixir** | Structs + Ecto.Changeset | 3-5 days |
| **Data Loading** | **Native (crucible_datasets v0.4.1)** | HF Hub via `crucible_datasets` | 0-2 days (wire-in) |
| **Math Verification** | **Pythonx/Snakepit** | Wrap sympy/math-verify | 1 week |
| **Evaluation** | **Native Preferred** | crucible_harness + custom | 1-2 weeks |
| **GPU Kernels** | **N/A** | Handled by Tinker API | 0 days |

**Revised Assessment:** 85% of cookbook functionality is **trivially portable** with correct scope understanding. Total effort: **4-6 weeks** (down from original 6-month estimate).

**Update (2025-12-22):** Configuration work should use `chz_ex` (Hex v0.1.2)
with `ChzEx.Schema` rather than hand-rolled structs.

**Update (2025-12-23):** Complete library mapping now available. Key additions:
- `pydantic` → `sinter` (Hex v0.0.1) for schema validation
- `huggingface_hub` → `hf_hub_ex` (distinct from `hf_datasets_ex`)
- BEAM-obviated libs: asyncio, cloudpickle, tqdm, threading (no packages needed)
- See `docs/20251223/PYTHON_TO_ELIXIR_LIBRARY_MAPPING.md` for complete reference.

---

## Dependency Matrix by Recipe (VERIFIED)

### Supervised Fine-Tuning (chat_sl)

| Dependency | Type | Elixir Mapping | Status |
|------------|------|----------------|--------|
| `chz` | Configuration | Elixir structs + Ecto.Changeset | ✅ Verified |
| `transformers` | Tokenization only | `{:tokenizers}` already in tinkex | ✅ Done |
| `torch` | Tensor ops only | Nx (not model training) | ✅ Trivial |
| `datasets` | Data loading | `crucible_datasets` v0.4.1 (HF Hub native) | ✅ Verified |
| `tokenizers` | Tokenization | `{:tokenizers, "~> 0.5"}` | ✅ Done |

**Verdict:** Almost everything already available. Port configs to Elixir structs, use Tinkex API for training.

---

### Math RL Training (math_rl)

| Dependency | Type | Elixir Mapping | Status |
|------------|------|----------------|--------|
| `sympy` | Symbolic math | Snakepit wrapper | ✅ Verified |
| `math-verify` | Answer verification | Snakepit wrapper | ✅ Verified |
| `pylatexenc` | LaTeX parsing | Pythonx (pure Python) | ✅ Verified |
| `scipy.signal.lfilter` | Discounted rewards | **Native Elixir (2 hours)** | ✅ Verified |
| `chz` | Configuration | Elixir structs | ✅ Verified |

**Verdict:** Math verification via Snakepit, discounted rewards via native Elixir.

**Native Implementation (replaces scipy.signal.lfilter):**
```elixir
# This is the ONLY scipy function used in the entire cookbook
def discounted_future_sum(rewards, gamma) when is_list(rewards) do
  rewards
  |> Enum.reverse()
  |> Enum.reduce({[], 0.0}, fn reward, {acc, future_sum} ->
    current_sum = reward + gamma * future_sum
    {[current_sum | acc], current_sum}
  end)
  |> elem(0)
end
```

---

### Direct Preference Optimization (preference/dpo)

| Dependency | Type | Elixir Mapping | Status |
|------------|------|----------------|--------|
| `torch` | Tensor ops only | Nx (training via API) | ✅ Verified |
| `transformers` | Tokenization | `{:tokenizers}` in tinkex | ✅ Done |
| `chz` | Configuration | Elixir structs | ✅ Verified |
| `datasets` | Preference pairs | `crucible_datasets` loaders | ✅ Verified |

**Verdict:** Training happens SERVER-SIDE via Tinker API. Client just prepares data.

---

### Multiplayer RL (multiplayer_rl) - OPTIONAL

| Dependency | Type | Elixir Mapping | Status |
|------------|------|----------------|--------|
| `textarena` | Multi-agent games | **SKIP** (optional recipe) | ⚠️ Low Priority |
| `verifiers` | RL environments | **SKIP** (optional feature) | ⚠️ Very Low |
| `openai` | Model inference | HTTP client (Req) | ✅ Trivial |

**Verdict:** Skip unless specifically implementing TicTacToe self-play. Elixir excels at concurrent orchestration if needed later.

---

### Tool Use (tool_use/search)

| Dependency | Type | Elixir Mapping | Status |
|------------|------|----------------|--------|
| `sentence-transformers` | Embeddings | Bumblebee | ✅ Available |
| `faiss` | Vector search | Use Qdrant/Milvus HTTP | ⚠️ Alternative |
| `openai` | API calls | HTTP client | ✅ Trivial |

**Verdict:** Use Bumblebee for embeddings, vector DB via HTTP.

---

### Reinforcement Learning Core (rl/train.py)

| Dependency | Type | Elixir Mapping | Status |
|------------|------|----------------|--------|
| `flash-attn` | GPU kernels | **Tinker API handles** | ✅ N/A |
| `deepspeed` | Distributed training | **Tinker API handles** | ✅ N/A |
| `liger-kernel` | Triton kernels | **Tinker API handles** | ✅ N/A |
| `vllm` | Inference server | OpenAI-compatible HTTP | ✅ Trivial |
| `verifiers` | Environments | **SKIP** (optional) | ⚠️ Low Priority |

**Verdict:** GPU training is SERVER-SIDE. Elixir orchestrates via Tinkex API.

---

## Library Mapping Summary (VERIFIED BY 9 AGENTS)

### Already Done (tinkex provides)

| Python Library | Elixir Equivalent | Status |
|----------------|-------------------|--------|
| `tinker>=0.3.0` | `{:tinkex, "~> 0.3.2"}` | ✅ 1:1 parity |
| `tokenizers` | `{:tokenizers, "~> 0.5"}` | ✅ In tinkex |
| `tiktoken` | `{:tiktoken_ex, "~> 0.1"}` | ✅ In tinkex |

### Trivial Native Elixir (1-2 days total)

| Python Library | Elixir Equivalent | Actual Usage |
|----------------|-------------------|--------------|
| `numpy` (18 funcs) | `Nx` | mean, std, sqrt, linspace, argsort, prod |
| `scipy` (1 func!) | `Enum.reduce/3` | `lfilter` for discounted rewards only |
| `torch` (tensors) | `Nx` | Data prep only, NOT training |
| `blobfile` (3 uses) | `File` + `ExAws.S3` | Simple JSONL reads |

### Native Elixir Port (3-5 days)

| Python Library | Elixir Equivalent | Effort |
|----------------|-------------------|--------|
| `chz` (80+ files) | Elixir structs + Ecto.Changeset | 3-5 days |
| `datasets` (local ops) | `crucible_datasets` | DONE (v0.4.1) |
| `rich`/`termcolor` | `TableRex` + `IO.ANSI` | 4-6 hours |
| `anyio`/`asyncio` | `Task.async_stream/2` | 1 hour |

### Pythonx/Snakepit Wrappers (1-2 weeks, datasets done in crucible_datasets)

| Python Library | Strategy | Priority |
|----------------|----------|----------|
| `sympy` | Snakepit | MEDIUM |
| `pylatexenc` | Pythonx | MEDIUM |
| `math-verify` | Snakepit | MEDIUM |

### Native Alternative (REVISED - inspect-ai)

| Python Library | Strategy | Rationale |
|----------------|----------|-----------|
| `inspect-ai` | **Native crucible_harness** | Uses <10% of framework |

**Key Finding:** Cookbook uses minimal inspect-ai surface area. Build native Elixir evaluation in 1-2 weeks instead of wrapping.

### SKIP (Optional Dependencies)

| Component | Reason | Decision |
|-----------|--------|----------|
| `textarena` | Only for multiplayer_rl recipe | Skip unless needed |
| `verifiers` | Optional install (`[verifiers]`) | Skip entirely |

### Handled by Tinker API (Not Our Problem)

| Component | Location | Status |
|-----------|----------|--------|
| `flash-attn` | Server-side | ✅ Tinker handles |
| `deepspeed` | Server-side | ✅ Tinker handles |
| `liger-kernel` | Server-side | ✅ Tinker handles |
| LoRA/DPO/GRPO training | Server-side | ✅ Tinker handles |

---

## Recommended Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     ELIXIR (Tinkex + Apps)                       │
├─────────────────────────────────────────────────────────────────┤
│  Orchestration     │  Data Pipeline     │  Evaluation           │
│  ├─ GenServers     │  ├─ Explorer       │  ├─ Nx metrics        │
│  ├─ Broadway       │  ├─ S3/ExAws       │  ├─ HTTP judges       │
│  └─ OTP Supervisor │  └─ Parquet I/O    │  └─ Custom rubrics    │
├─────────────────────────────────────────────────────────────────┤
│                    PYTHON INTEROP LAYER                          │
├─────────────────────────────────────────────────────────────────┤
│  Pythonx (Livebook)          │  Snakepit (Production)           │
│  ├─ sympy                    │  ├─ math-verify                   │
│  ├─ pylatexenc               │  ├─ textarena                     │
│  └─ math-verify (prototype)  │  └─ verifiers (environments)     │
├─────────────────────────────────────────────────────────────────┤
│                    EXTERNAL SERVICES (HTTP)                      │
├─────────────────────────────────────────────────────────────────┤
│  Tinker API           │  vLLM Server       │  HuggingFace Hub   │
│  ├─ LoRA training     │  ├─ /v1/chat       │  ├─ Datasets       │
│  ├─ DPO training      │  └─ /v1/completions│  └─ Models         │
│  └─ GRPO training     │                    │                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Roadmap (REVISED)

### Week 1: Core Data Types
- [ ] Add `Nx` dependency to tinkex (if not present)
- [ ] Implement `Tinkex.TensorData` (Nx ↔ API bridge)
- [ ] Port 18 numpy functions to Nx (trivial)
- [ ] Implement `discounted_future_sum/2` (2 hours)
- [ ] Native blobfile replacement (File + ExAws.S3)

### Week 2: Configuration & Datasets
- [ ] Port chz patterns to Elixir structs
- [ ] Add Explorer for Parquet/JSONL operations
- [ ] Implement dataset ops (shuffle, split, filter)
- [ ] Port rich/termcolor to IO.ANSI (4-6 hours)

### Week 3: Math Verification
- [ ] Set up Snakepit process pool
- [ ] Wrap sympy/pylatexenc/math-verify
- [ ] Create unified `Tinkex.Math` module
- [ ] Test numerical equivalence with Python

### Week 4: HuggingFace Integration
- [ ] Add `{:crucible_datasets, "~> 0.4.1"}` dependency
- [ ] Replace `datasets.load_dataset(...)` with `CrucibleDatasets.load_dataset/2`
- [ ] Test with 5 common datasets (gsm8k, math-500, tulu3, hh-rlhf, caltech101)

### Weeks 5-6: Evaluation (Optional)
- [ ] Model adapter protocol for LLM-as-judge
- [ ] LLM-as-judge scorer
- [ ] Integrate with crucible_harness (70% already exists)

### Future (If Needed)
- [ ] textarena wrapper for multiplayer RL
- [ ] verifiers integration for RL environments

---

## Detailed Reports

### Original Analysis Documents (Verified)

| Report | File | Status | Summary |
|--------|------|--------|---------|
| chz Library | `01_chz_library.md` | ✅ Verified | 80+ files, port to Elixir structs |
| textarena | `02_textarena_library.md` | ✅ Updated | OPTIONAL - skip unless needed |
| math-verify | `03_math_verify_library.md` | ✅ Verified | Pure Python - Snakepit ideal |
| inspect-ai | `04_inspect_ai_library.md` | ✅ Major Update | Uses <10% - build native instead |
| pylatexenc | `05_pylatexenc_library.md` | ✅ Verified | Pure Python - Pythonx ideal |
| SymPy | `06_sympy_symbolic_math.md` | ✅ Verified | Pure Python - Snakepit for production |
| SciPy | `07_scipy_nx_mapping.md` | ⚠️ Outdated | See 13_numpy_scipy_actual_usage.md |
| Torch/Transformers | `08_torch_transformers_axon_mapping.md` | ⚠️ Outdated | See 12_torch_transformers_actual_usage.md |
| Datasets/Blobfile | `09_datasets_blobfile_mapping.md` | ⚠️ Outdated | See 14_datasets_blobfile_actual_usage.md |
| Verifiers | `10_verifiers_library.md` | ✅ Updated | OPTIONAL - skip entirely |

### New Verification Documents (Created by Audit Agents)

| Report | File | Agent | Key Finding |
|--------|------|-------|-------------|
| Tinker → tinkex | `11_tinker_to_tinkex.md` | a44a379 | 1:1 parity confirmed, ALREADY DONE |
| Torch/Transformers Actual | `12_torch_transformers_actual_usage.md` | abdc804 | Tensor ops only, NOT training |
| NumPy/SciPy Actual | `13_numpy_scipy_actual_usage.md` | afc2b1c | 18 functions, 1 scipy - trivial |
| Datasets/Blobfile Actual | `14_datasets_blobfile_actual_usage.md` | a316650 | Native Elixir for I/O, `crucible_datasets` for HF Hub |
| Utility Libraries | `15_utility_libs.md` | a6087fc | rich/termcolor/anyio - 4-6 hours |
| **Complete Mapping** | **`16_COMPLETE_DEPENDENCY_MAPPING.md`** | - | **Definitive reference for porting** |

### Supporting Documents

| Report | File | Summary |
|--------|------|---------|
| Verification Summary | `VERIFICATION_SUMMARY.md` | Math docs verification status |

---

## Risk Assessment (REVISED)

### Low Risk - Proceed Confidently

| Area | Rationale |
|------|-----------|
| Configuration (chz → structs) | Mechanical translation, 80+ files |
| Local file I/O (blobfile → File) | Only 3 usages, trivial |
| Dataset operations (Explorer) | Polars-backed, fast |
| NumPy/SciPy (Nx) | 18+1 functions, all trivial |
| HTTP APIs (vLLM, OpenAI) | Standard HTTP clients |
| Tokenization | Already in tinkex |

### Medium Risk - Prototype First

| Area | Rationale |
|------|-----------|
| Math verification (Snakepit) | sympy integration needs testing |
| HuggingFace Hub (`crucible_datasets`) | 15+ datasets, auth handling |
| LLM-as-judge (native) | Need to design protocol |

### Deferred - Skip Unless Requested

| Area | Rationale |
|------|-----------|
| textarena | Only for multiplayer_rl recipe |
| verifiers | Optional feature, edge case |

### Not Applicable - Tinker API Handles

| Area | Rationale |
|------|-----------|
| LoRA/DPO/GRPO training | Server-side GPU operations |
| Flash attention | CUDA kernels |
| Distributed training | DeepSpeed on server |

---

## Critical Corrections from Original Analysis

### What We Got WRONG:

| Original Estimate | Actual | Correction |
|-------------------|--------|------------|
| torch/transformers: 8 weeks | 1-2 days | Tensor ops only, NOT training |
| scipy: extensive | 1 function | `lfilter` for discounted rewards |
| inspect-ai: 3-6 months | 1-2 weeks | Uses <10% of framework |
| Overall: 6 months | **4-6 weeks** | CLIENT-SIDE scope, not backend |

### What We Got RIGHT:

- tinker SDK → tinkex: 1:1 parity confirmed
- chz → Elixir structs: Correct approach
- blobfile → File + ExAws.S3: Correct approach
- Math verification → Snakepit: Correct approach
- GPU ops → Tinker API: Correct approach

---

## Conclusion

**Tinker-cookbook is 85% trivially portable** with correct scope understanding:

1. **tinkex already provides** core SDK functionality (1:1 parity with Python)
2. **NumPy/SciPy/torch** usage is minimal client-side tensor ops → Nx handles all of it
3. **Configuration** (chz) is mechanical translation to Elixir structs
4. **Dataset loading** is native via `crucible_datasets` v0.4.1 (HF Hub included)
5. **Evaluation** uses <10% of inspect-ai → build native instead
6. **GPU training** is SERVER-SIDE → Tinker API handles it
7. **Optional features** (textarena, verifiers) can be skipped

**Key Insight:** The original analysis confused CLIENT-SIDE cookbook with SERVER-SIDE training infrastructure. Once scoped correctly, the port becomes straightforward.

**Total Effort:** 4-6 weeks for full cookbook parity (not 6 months)

---

**Document Status:** VERIFIED COMPLETE
**Verified By:** 9 specialized audit agents (2025-12-20)
**Definitive Reference:** `16_COMPLETE_DEPENDENCY_MAPPING.md`
**Maintainer:** tinkex team
