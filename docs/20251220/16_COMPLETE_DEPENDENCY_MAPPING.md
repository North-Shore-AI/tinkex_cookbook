# Complete Dependency Mapping: tinker-cookbook → tinkex_cookbook

**Date:** 2025-12-20
**Status:** VERIFIED - All mappings confirmed against actual cookbook codebase
**Purpose:** Definitive reference for porting decisions

---

## Executive Summary

After thorough analysis by 9 specialized agents, here is the complete dependency mapping:

| Category | Count | Strategy | Effort |
|----------|-------|----------|--------|
| **Already Replaced** | 1 | tinkex handles | 0 days |
| **Trivial Nx Ports** | 18 functions | Native Elixir | 1-2 days |
| **Native Elixir** | 5 libs | Port directly | 1-2 weeks |
| **Pythonx/Snakepit** | 4 libs | Wrap Python | 2-3 weeks |
| **Optional/Skip** | 3 libs | Don't port | 0 days |
| **HTTP API** | 1 service | HTTP client | 1 day |

**Total Porting Effort:** 4-6 weeks for full cookbook parity

---

## 1. ALREADY REPLACED BY TINKEX

### Python `tinker>=0.3.0` → Elixir `{:tinkex, "~> 0.3.2"}`

**Status:** 100% COMPLETE - No porting needed

| Python | Elixir tinkex | Parity |
|--------|---------------|--------|
| `ServiceClient` | `Tinkex.ServiceClient` | 1:1 |
| `TrainingClient` | `Tinkex.TrainingClient` | 1:1 |
| `SamplingClient` | `Tinkex.SamplingClient` | 1:1 |
| `RestClient` | `Tinkex.RestClient` | 1:1 |
| `forward_backward()` | `TrainingClient.forward_backward/3` | 1:1 |
| `sample()` | `SamplingClient.sample/4` | 1:1 |
| Tokenizers (HF) | `{:tokenizers, "~> 0.5"}` | 1:1 |
| Tokenizers (TikToken) | `{:tiktoken_ex, "~> 0.1"}` | 1:1 |

**tinkex also provides ENHANCEMENTS not in Python:**
- Automatic recovery from checkpoint corruption
- Multi-tenant rate limiting
- Composable regularizer pipelines
- Built-in telemetry/metrics

---

## 2. TRIVIAL Nx REPLACEMENTS (18 NumPy Functions)

### Statistical Operations (6 usages)
| NumPy | Nx | Files |
|-------|-----|-------|
| `np.mean()` | `Nx.mean/1` | 3 files |
| `np.std()` | `Nx.standard_deviation/1` | 2 files |
| `np.sqrt()` | `Nx.sqrt/1` | 1 file |

### Array Creation (4 usages)
| NumPy | Nx | Files |
|-------|-----|-------|
| `np.linspace()` | `Nx.linspace/3` | 2 files |
| `np.argsort()` | `Nx.argsort/1` | 1 file |
| `np.prod()` | `Nx.product/1` | 1 file |

### Random Numbers (2 usages)
| NumPy | Nx | Files |
|-------|-----|-------|
| `np.random.RandomState()` | `Nx.Random.key/1` | 1 file |
| `rng.randint()` | `Nx.Random.randint/3` | 1 file |

### Type Operations (4 usages)
| NumPy | Nx | Files |
|-------|-----|-------|
| `isinstance(x, np.ndarray)` | `is_struct(x, Nx.Tensor)` | 1 file |
| `.tolist()` | `Nx.to_flat_list/1` | 1 file |
| `torch.tensor()` | `Nx.tensor/1` | 12 files |
| `torch.cat()` | `Nx.concatenate/1` | 4 files |

### SciPy (1 function - ONLY scipy usage!)
| SciPy | Native Elixir | Location |
|-------|---------------|----------|
| `scipy.signal.lfilter` (discounted rewards) | Custom `Enum.reduce/3` | `rl/metrics.py:147` |

**Elixir Implementation:**
```elixir
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

## 3. NATIVE ELIXIR PORTS

### A. Configuration: `chz` → Elixir Structs

**Files Using:** 80+ files with `@chz.chz` decorator

**Mapping:**
```
Python @chz.chz                  →  defstruct + @enforce_keys
Python chz.field(default=x)      →  field :name, default: x
Python chz.field(munger=fn)      →  Ecto.Changeset.update_change
Python chz.asdict()              →  Map.from_struct/1
```

**Effort:** 3-5 days (mostly mechanical translation)

### B. Data Loading: `blobfile` → Native Elixir

**Files Using:** 3 locations (simple JSONL reads)

**Mapping:**
```
Python blobfile.BlobFile(path)   →  File.stream!/1 (local)
Python blobfile with s3://       →  ExAws.S3.get_object/2
```

**Implementation:**
```elixir
def read_jsonl(path) when is_binary(path) do
  cond do
    String.starts_with?(path, "s3://") -> read_jsonl_s3(path)
    true -> File.stream!(path) |> Stream.map(&Jason.decode!/1)
  end
end
```

**Effort:** 1-2 days

### C. Dataset Operations: `datasets` (local ops) → Explorer

**Operations Used:**
| Python datasets | Explorer | Status |
|-----------------|----------|--------|
| `shuffle(seed=0)` | `DataFrame.shuffle(seed: 0)` | Direct |
| `take(N)` | `DataFrame.slice(0, N)` | Direct |
| `skip(N)` | `DataFrame.slice(N, -1)` | Direct |
| `filter(fn)` | `DataFrame.filter(condition)` | Direct |
| `concatenate_datasets()` | `DataFrame.concat_rows()` | Direct |
| `Dataset.from_list()` | `DataFrame.new()` | Direct |

**Note:** HuggingFace Hub downloads are now handled by `crucible_datasets` v0.4.1
via `hf_hub` (no Pythonx required for cookbook datasets).

**Effort:** 3-5 days

### D. Utility Libraries → Native Elixir

| Python | Elixir | Effort |
|--------|--------|--------|
| `rich` (tables) | `TableRex` or `IO.ANSI` | 1-2 hours |
| `termcolor` | `IO.ANSI` (built-in) | 15 min |
| `anyio`/`asyncio` | `Task.async_stream/2` | 1 hour |

**Total Effort:** 4-6 hours

### E. Evaluation: `inspect-ai` → Native Crucible

**Critical Finding:** Cookbook uses <10% of inspect-ai surface area:
- Model adapter pattern (`ModelAPI`)
- Async eval runner (`eval_async`)
- Task definition (`@task`)
- LLM-as-judge (`model_graded_qa`)

**Crucible Already Has:**
- `crucible_harness` - experiment orchestration
- `crucible_telemetry` - metrics
- `crucible_datasets` v0.4.1 - full tinker dataset coverage + HF Hub

**Need to Build:**
- Model adapter protocol (1-2 days)
- LLM-as-judge scorer (2-3 days)
- Task definition DSL (2-3 days)

**Effort:** 1-2 weeks (native, not wrapping)

---

## 4. PYTHONX/SNAKEPIT WRAPPERS

### A. Math Verification Stack

| Library | Wrapper | Complexity | Reason |
|---------|---------|------------|--------|
| `sympy` | Snakepit | MEDIUM | Symbolic math, pure Python |
| `pylatexenc` | Pythonx | LOW | LaTeX parsing, pure Python |
| `math-verify` | Snakepit | LOW | Answer verification, pure Python |

**Usage Location:** `recipes/math_rl/math_grading.py` (single file)

**Wrapper Pattern:**
```elixir
defmodule Tinkex.Math.Verify do
  use Snakepit

  def grade_answer(given, expected) do
    Snakepit.call(:math_verify, :grade, [given, expected])
  end
end
```

**Effort:** 1 week total

### B. HuggingFace Datasets (Hub Downloads)

**15+ datasets require HuggingFace Hub API:**
- `allenai/tulu-3-sft-mixture`
- `HuggingFaceH4/MATH-500`
- `openai/gsm8k`
- `Anthropic/hh-rlhf`
- etc.

**Implementation (native):**
```elixir
# mix.exs
{:crucible_datasets, "~> 0.4.1"}

# usage
{:ok, ds} = CrucibleDatasets.load_dataset("openai/gsm8k", split: "train")
{:ok, stream} =
  CrucibleDatasets.load_dataset("openai/gsm8k", split: "train", streaming: true)
```

**Effort:** 0-2 days (wire into cookbook)

---

## 5. OPTIONAL - SKIP UNLESS NEEDED

### A. `textarena` (Multiplayer RL)

**Status:** CORE dependency, but only for `recipes/multiplayer_rl/`

**Decision:** Skip unless implementing TicTacToe self-play training

**If Needed:**
- Pythonx wrapper: 3-5 days
- Native Elixir (TicTacToe only): 1 week

### B. `verifiers` (RL Environments)

**Status:** OPTIONAL dependency (`pip install tinker-cookbook[verifiers]`)

**Decision:** Skip entirely - edge case feature

**If Needed:**
- Python subprocess via Port: 5-7 days
- Heavy native dependencies make pure port impractical

### C. GPU Training Libraries

**Status:** NOT PORTABLE - handled by Tinker API

| Library | Purpose | tinkex Handles |
|---------|---------|----------------|
| `flash-attn` | CUDA attention | Via API |
| `deepspeed` | Distributed training | Via API |
| `liger-kernel` | Triton kernels | Via API |
| `vllm` | Inference server | HTTP client |

**Decision:** Use Tinker API for all GPU operations

---

## 6. HTTP API INTEGRATIONS

### vLLM Inference Server

**Pattern:** OpenAI-compatible HTTP API

```elixir
defmodule Tinkex.Inference.VLLM do
  def chat_completion(endpoint, messages, opts \\ []) do
    Req.post!(endpoint <> "/v1/chat/completions",
      json: %{messages: messages, model: opts[:model]}
    )
  end
end
```

**Effort:** 1 day (straightforward HTTP client)

---

## Complete Dependency Summary Table

| Dependency | Usage Count | Strategy | Priority | Effort |
|------------|-------------|----------|----------|--------|
| `tinker` | Core | **ALREADY DONE** (tinkex) | - | 0 |
| `numpy` | 18 functions | **Nx** (trivial) | HIGH | 1-2 days |
| `scipy.signal` | 1 function | **Native Elixir** | HIGH | 2 hours |
| `torch` (tensors) | 12 files | **Nx** (trivial) | HIGH | 1-2 days |
| `transformers` (tokenizer) | 3 files | **Already in tinkex** | - | 0 |
| `chz` | 80+ files | **Elixir structs** | HIGH | 3-5 days |
| `datasets` (local) | 8 files | **crucible_datasets v0.4.1** | DONE | 0-2 days (wire-in) |
| `datasets` (HF Hub) | 15+ datasets | **crucible_datasets v0.4.1** | DONE | 0-2 days (wire-in) |
| `blobfile` | 3 files | **Native Elixir** | HIGH | 1-2 days |
| `rich`/`termcolor` | 4 files | **IO.ANSI** | LOW | 4-6 hours |
| `sympy`/`pylatexenc` | 1 file | **Snakepit** | MEDIUM | 1 week |
| `math-verify` | 1 file | **Snakepit** | MEDIUM | 2 days |
| `inspect-ai` | 4 files | **Native Crucible** | MEDIUM | 1-2 weeks |
| `textarena` | 1 recipe | **SKIP** (optional) | LOW | - |
| `verifiers` | 1 recipe | **SKIP** (optional) | VERY LOW | - |
| GPU libs | N/A | **Tinker API** | - | 0 |

---

## Implementation Roadmap

### Week 1: Core Data Types
- [ ] Add `Nx` dependency to tinkex
- [ ] Implement `Tinkex.TensorData` (Nx ↔ API bridge)
- [ ] Port numpy/scipy functions to Nx
- [ ] Native blobfile replacement

### Week 2: Configuration & Datasets
- [ ] Port chz patterns to Elixir structs
- [ ] Add Explorer for Parquet/JSONL
- [ ] Add `crucible_datasets ~> 0.4.1` and wire dataset operations

### Week 3: Math Verification
- [ ] Set up Snakepit process pool
- [ ] Wrap sympy/pylatexenc/math-verify
- [ ] Create unified `Tinkex.Math` module

### Week 4: HuggingFace Integration
- [ ] Replace `datasets.load_dataset` calls with `CrucibleDatasets.load_dataset/2`
- [ ] Test with 5 common datasets

### Week 5-6: Evaluation (Optional)
- [ ] Model adapter protocol
- [ ] LLM-as-judge scorer
- [ ] Integrate with crucible_harness

---

## Files Created/Updated by Audit Agents

| File | Agent | Status |
|------|-------|--------|
| `11_tinker_to_tinkex.md` | a44a379 | Created |
| `12_torch_transformers_actual_usage.md` | abdc804 | Created |
| `13_numpy_scipy_actual_usage.md` | afc2b1c | Created |
| `14_datasets_blobfile_actual_usage.md` | a316650 | Created |
| `15_utility_libs.md` | a6087fc | Created |
| `01_chz_library.md` | a70818e | Verified |
| `03_math_verify_library.md` | a71e1fe | Verified |
| `05_pylatexenc_library.md` | a71e1fe | Verified |
| `06_sympy_symbolic_math.md` | a71e1fe | Verified |
| `04_inspect_ai_library.md` | a351f72 | Extensively Updated |
| `02_textarena_library.md` | a3761dd | Updated |
| `10_verifiers_library.md` | a3761dd | Updated |
| `VERIFICATION_SUMMARY.md` | a71e1fe | Created |

---

## Critical Corrections from Original Analysis

### What We Got WRONG Initially:

1. **Scope Confusion**: Original analysis assumed we were porting Tinker's BACKEND (flash-attn, deepspeed, etc.). We're actually porting COOKBOOK (client-side examples).

2. **torch/transformers**: Originally estimated 8 weeks. Actual usage is just tensor ops and tokenization - already handled by Nx and `{:tokenizers}`.

3. **scipy**: Originally assumed extensive usage. Actual usage is ONE function (`lfilter` for discounted rewards) - trivial to implement in Elixir.

4. **inspect-ai**: Originally estimated 3-6 months. Actual usage is <10% of framework - can be replaced by crucible_harness in 1-2 weeks.

### What We Got RIGHT:

1. **tinker SDK** → tinkex: 1:1 parity confirmed
2. **chz** → Elixir structs: Correct approach
3. **blobfile** → File + ExAws.S3: Correct approach
4. **Math verification** → Snakepit: Correct approach
5. **GPU ops** → Tinker API: Correct approach

---

**Document Status:** Complete
**Last Updated:** 2025-12-20
**Verified By:** 9 specialized audit agents
