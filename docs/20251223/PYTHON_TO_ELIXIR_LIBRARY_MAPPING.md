# Python to Elixir Library Mapping (Complete Reference)

**Date:** 2025-12-23
**Status:** Authoritative mapping for tinkex_cookbook
**Purpose:** Single source of truth for all Python → Elixir library decisions

---

## Quick Reference: Key Mappings

| Python Library | Elixir Equivalent | Status |
|----------------|-------------------|--------|
| `tinker` | `tinkex` | Done (Hex ~> 0.3.2) |
| `chz` | `chz_ex` | Done (Hex ~> 0.1.2) |
| `pydantic` | `sinter` | Done (Hex ~> 0.0.1) |
| `datasets` | `hf_datasets_ex` | Done (Hex ~> 0.1) |
| `huggingface_hub` | `hf_hub` | Done (Hex ~> 0.1) |
| `google-genai` | `gemini_ex` | Done (Hex ~> 0.8) |
| `openai` | `openai_ex` | Done (Hex ~> 0.8) |
| `chromadb` | `chroma` | Planned (Hex ~> 0.1.2) |
| `sympy` / `pylatexenc` / `math_verify` | `snakebridge` | Done (Hex ~> 0.3.0) |

---

## 1. BEAM-Obviated Libraries (Native Elixir - No Dependency Needed)

These Python libraries are **completely replaced** by BEAM/OTP primitives:

| Python Library | BEAM Replacement | Scope | Notes |
|----------------|------------------|-------|-------|
| `asyncio` / `anyio` | `Task`, `GenServer`, OTP | Core | BEAM processes are lightweight actors |
| `cloudpickle` | `:erlang.term_to_binary/1` | xmux | ETF serialization is native |
| `tqdm` | `ProgressBar` or Logger | prompt_distillation | Optional; progress via Telemetry |
| `threading` / `multiprocessing` | `Task.async_stream/2` | Core | BEAM handles concurrency |
| `queue` | `GenStage`, `:queue` | Core | Built-in message passing |

**Why BEAM obviates these:**
- Elixir processes are ~2KB each (vs Python threads at ~8MB)
- No GIL - true parallelism on all cores
- Built-in message passing, supervision, and fault tolerance
- `:erlang.term_to_binary/1` handles any Elixir term (replaces pickle/cloudpickle)

---

## 2. Native Elixir Replacements (Simple Ports)

These require Elixir packages but no Python interop:

### Terminal/UI
| Python | Elixir | Hex Package | Scope | Usage Count |
|--------|--------|-------------|-------|-------------|
| `rich` | `TableRex` + `IO.ANSI` | `{:table_rex, "~> 4.0"}` | Core | 1 file (`ml_log.py`) |
| `termcolor` | `IO.ANSI` (stdlib) | None needed | Core | 4 files |

### Tensor/Numeric
| Python | Elixir | Hex Package | Scope | Usage Count |
|--------|--------|-------------|-------|-------------|
| `numpy` | `Nx` | `{:nx, "~> 0.9"}` | Core + RL | 18 functions total |
| `torch` (tensors only) | `Nx` | `{:nx, "~> 0.9"}` | Core + RL | 12 files (NOT training) |
| `scipy.signal.lfilter` | Native `Enum.reduce` | None needed | RL | 1 function only |

**scipy implementation:**
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

### Image Processing
| Python | Elixir | Hex Package | Scope | Notes |
|--------|--------|-------------|-------|-------|
| `Pillow` (PIL) | `Image` | `{:image, "~> 0.47"}` | multimodal | Or use `StbImage` |

### File I/O
| Python | Elixir | Hex Package | Scope | Notes |
|--------|--------|-------------|-------|-------|
| `blobfile` (local) | `File` (stdlib) | None needed | datasets | JSONL reads |
| `blobfile` (s3://) | `ExAws.S3` | `{:ex_aws_s3, "~> 2.5"}` | datasets | Cloud storage |

---

## 3. Direct Elixir Package Mappings

These Python libraries have direct Elixir equivalents on Hex:

| Python | Elixir | Hex Package | Scope | Status |
|--------|--------|-------------|-------|--------|
| `tinker` | `Tinkex` | `{:tinkex, "~> 0.3.2"}` | Core | 1:1 parity |
| `chz` | `ChzEx` | `{:chz_ex, "~> 0.1.2"}` | Core (80+ files) | Ready |
| `pydantic` | `Sinter` | `{:sinter, "~> 0.0.1"}` | renderers, xmux | Ready |
| `transformers` (tokenizers) | `Tokenizers` | via `tinkex` | Core | Already included |
| `tiktoken` | `TiktokenEx` | via `tinkex` | Core | Already included |

---

## 4. HuggingFace Ecosystem Clarification

**You have TWO separate Elixir libraries:**

| Python | Elixir | Purpose | Hex Package |
|--------|--------|---------|-------------|
| `datasets` | `hf_datasets_ex` | Dataset loading, streaming, splits | `{:hf_datasets_ex, "~> 0.1"}` |
| `huggingface_hub` | `hf_hub` | Model downloads, repo metadata, Hub API | `{:hf_hub, "~> 0.1"}` |

**Usage in cookbook:**

| Location | Python Import | Elixir Replacement |
|----------|---------------|-------------------|
| `supervised/data.py` | `from datasets import load_dataset` | `HfDatasetsEx.load_dataset/2` |
| `hyperparam_utils.py` | `from huggingface_hub import hf_hub_download` | `HfHub.hf_hub_download/3` |
| `tool_use/search/search_env.py` | `from huggingface_hub import ...` | `HfHub` |

**Summary:**
- `hf_datasets_ex` → replaces `datasets` Python package (load_dataset, IterableDataset, etc.)
- `hf_hub` → replaces `huggingface_hub` Python package (model downloads, API calls)

---

## 5. SnakeBridge Libraries (Keep in Python)

These remain in Python via SnakeBridge v0.3.0 (built-in manifests):

| Python Library | Scope | Why Keep Python |
|----------------|-------|-----------------|
| `sympy` | math_rl only | Complex symbolic math engine |
| `pylatexenc` | math_rl only | LaTeX parsing, pure Python |
| `math_verify` | math_rl only | Answer equivalence, pure Python |

**These are the ONLY three libraries using SnakeBridge for core cookbook.**

---

## 6. Recipe-Specific Dependencies

Libraries only needed for specific recipes (not core):

### math_rl Recipe
| Python | Strategy | Notes |
|--------|----------|-------|
| `sympy` | SnakeBridge | Built-in manifest |
| `pylatexenc` | SnakeBridge | Built-in manifest |
| `math_verify` | SnakeBridge | Built-in manifest |

### multiplayer_rl Recipe (OPTIONAL)
| Python | Strategy | Notes |
|--------|----------|-------|
| `textarena` | SnakeBridge (if needed) | Skip unless implementing |

### verifiers_rl Recipe (OPTIONAL)
| Python | Strategy | Notes |
|--------|----------|-------|
| `verifiers` | SnakeBridge (if needed) | Skip unless implementing |
| `openai` | HTTP client (`Req`) | Simple REST API |

### tool_use/search Recipe (OPTIONAL)
| Python | Strategy | Notes |
|--------|----------|-------|
| `chromadb` | HTTP client | Vector DB has REST API |
| `google-genai` | `gemini_ex` | Already have this |

### eval (OPTIONAL)
| Python | Strategy | Notes |
|--------|----------|-------|
| `inspect-ai` | Native `crucible_harness` | Uses <10% of framework |

---

## 7. Complete Dependency Matrix by Scope

### CORE (used everywhere)
| Python | Elixir | Coverage |
|--------|--------|----------|
| `tinker` | `tinkex` | 100% |
| `chz` | `chz_ex` | 100% |
| `numpy` | `Nx` | 100% |
| `asyncio` | BEAM | 100% |

### CORE + SPECIFIC RECIPES
| Python | Elixir | Core | RL | Preference | Supervised |
|--------|--------|------|-----|------------|------------|
| `torch` (tensors) | `Nx` | Y | Y | Y | Y |
| `datasets` | `hf_datasets_ex` | - | Y | Y | Y |
| `transformers` | `tokenizers` | Y | - | - | - |

### RECIPE-ONLY
| Python | Elixir | Recipe |
|--------|--------|--------|
| `sympy` | SnakeBridge | math_rl |
| `pylatexenc` | SnakeBridge | math_rl |
| `math_verify` | SnakeBridge | math_rl |
| `scipy` | Native Elixir | rl (1 function) |
| `PIL` | `Image` | vlm_classifier |
| `tqdm` | Logger/optional | prompt_distillation |
| `pydantic` | `sinter` | renderers, xmux |

### OPTIONAL (skip unless needed)
| Python | Elixir | Recipe |
|--------|--------|--------|
| `textarena` | SnakeBridge | multiplayer_rl |
| `verifiers` | SnakeBridge | verifiers_rl |
| `chromadb` | HTTP client | tool_use/search |
| `inspect-ai` | Native harness | eval |

---

## 8. mix.exs Dependencies Summary

Current `mix.exs` includes:

```elixir
defp deps do
  [
    # HuggingFace ecosystem
    {:hf_datasets_ex, "~> 0.1"},        # datasets
    {:hf_hub, "~> 0.1"},                # huggingface_hub

    # LLM APIs
    {:gemini_ex, "~> 0.8"},             # google-genai
    {:openai_ex, "~> 0.8"},             # openai

    # LLM agent SDKs (CLI-backed)
    {:claude_agent_sdk, "~> 0.6.8"},    # Claude Code CLI
    {:codex_sdk, "~> 0.4.2"},           # Codex CLI

    # Vector store
    {:chroma, "~> 0.1.2"},              # chromadb

    # Core mappings
    {:tinkex, "~> 0.3.2"},              # tinker
    {:chz_ex, "~> 0.1.2"},              # chz
    {:sinter, "~> 0.0.1"},              # pydantic
    {:nx, "~> 0.9"},                    # numpy/torch tensors

    # Python interop (math_rl only)
    {:snakebridge, "~> 0.3.0"},         # sympy, pylatexenc, math_verify

    # Utility replacements
    {:table_rex, "~> 4.0"},             # rich
    {:ex_aws, "~> 2.5"},                # blobfile (cloud)
    {:ex_aws_s3, "~> 2.5"},             # blobfile (s3://)

    # Image processing (if vlm_classifier needed)
    # {:image, "~> 0.47"},              # PIL
  ]
end
```

---

## 9. Inference

**No separate inference library needed** - `tinkex` handles it via `Tinkex.SamplingClient`.

Tinker provides OpenAI-compatible inference server-side. No need for llama.cpp, vLLM, or Ollama.

---

## 9a. Remaining Python Libs (Recipe-Specific)

| Python Lib | Recipe | Strategy | Spec |
|------------|--------|----------|------|
| `inspect-ai` | eval/* | **Native Elixir** (crucible_harness + ~600 LOC) | [INSPECT_AI_PARITY_SPEC.md](inspect_ai_parity/INSPECT_AI_PARITY_SPEC.md) |
| `verifiers` | verifiers_rl | SnakeBridge if needed | Optional |
| `textarena` | multiplayer_rl/text_arena | SnakeBridge if needed | Optional |

### inspect-ai Coverage via Existing Libs

| inspect-ai Feature | North-Shore-AI Equivalent |
|--------------------|---------------------------|
| Task orchestration | `crucible_harness` |
| Statistical analysis | `crucible_bench` |
| Dataset loading | `crucible_datasets`, `hf_datasets_ex` |
| Report generation | `crucible_harness` (Markdown/LaTeX/HTML/Jupyter) |
| Model adapter | `tinkex` (SamplingClient) |

**Only need to build:** Task DSL, LLM-as-judge scorer, ModelAPI behaviour (~600 LOC total)

---

## 10. Researcher Breadcrumbs (Future Expansion)

These are in `pyproject.toml` but NOT imported - intentional hooks for researchers:

| Python Library | Purpose | Elixir Consideration |
|----------------|---------|---------------------|
| `torchvision` | Vision transforms, augmentation | `Image` + `Nx` for transforms |
| `anyio` | Async abstraction | BEAM handles natively |
| `wandb` | Experiment tracking | `crucible_telemetry` or HTTP API |
| `neptune-scale` | ML monitoring | HTTP API |
| `trackio` | Training I/O tracking | Custom telemetry |

**These are NOT blocking** - just hints for future capability expansion.

---

## 11. Migration Checklist

### Already Done
- [x] `tinker` → `tinkex`
- [x] `chz` → `chz_ex`
- [x] `datasets` → `hf_datasets_ex`
- [x] `huggingface_hub` → `hf_hub_ex`
- [x] `pydantic` → `sinter`
- [x] SnakeBridge for math libs

### BEAM Handles (No Action)
- [x] `asyncio` → BEAM processes
- [x] `cloudpickle` → ETF
- [x] `threading` → Tasks

### Need to Wire Up
- [ ] `numpy` → Nx usage in cookbook
- [ ] `torch` → Nx tensor ops
- [ ] `rich` → TableRex for CLI output
- [ ] `blobfile` → File/ExAws usage
- [ ] `scipy.signal.lfilter` → native implementation

### Optional (Skip for Now)
- [ ] `textarena` → only if multiplayer_rl needed
- [ ] `verifiers` → only if verifiers_rl needed
- [ ] `chromadb` → only if tool_use/search needed
- [ ] `inspect-ai` → prefer native crucible_harness

---

**Document Status:** Complete
**Last Updated:** 2025-12-23
