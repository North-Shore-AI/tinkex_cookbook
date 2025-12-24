# Tinker-Cookbook Dependency Usage Table (2025-12-23)

This table is based on direct import scans of
`tinker-cookbook/tinker_cookbook/*.py` plus the upstream `pyproject.toml`.
It answers: **is the lib actually used, where, and what's the plan**.

**Snakebridge:** v0.3.0 (built-in manifests for sympy/pylatexenc/math_verify).

## Key Mappings

| Python | Elixir | Status |
|--------|--------|--------|
| `tinker` | `tinkex` ~> 0.3.2 | Done |
| `chz` | `chz_ex` ~> 0.1.2 | Done |
| `pydantic` | `sinter` ~> 0.0.1 | Done |
| `datasets` | `hf_datasets_ex` | Done |
| `huggingface_hub` | `hf_hub_ex` | Done |

## BEAM-Obviated (No Elixir Package Needed)

| Python | BEAM Replacement | Notes |
|--------|------------------|-------|
| `asyncio` / `anyio` | `Task`, `GenServer` | Native concurrency |
| `cloudpickle` | `:erlang.term_to_binary` | ETF serialization |
| `tqdm` | Logger / Telemetry | Optional progress |
| `threading` | `Task.async_stream` | Native parallelism |

---

## Core + Recipe Dependencies

| Python lib | Used? | Where (examples) | Scope | Plan |
|---|---|---|---|---|
| **sympy** | Yes | `tinker_cookbook/recipes/math_rl/math_grading.py` | math_rl | Snakebridge (keep Python) |
| **pylatexenc** | Yes | `tinker_cookbook/recipes/math_rl/math_grading.py` | math_rl | Snakebridge (keep Python) |
| **math_verify** | Yes (dynamic import) | `tinker_cookbook/recipes/math_rl/math_grading.py` | math_rl | Snakebridge (keep Python) |
| **scipy** | Yes (dynamic import) | `tinker_cookbook/rl/metrics.py` | rl | Native Elixir (discounted rewards) |
| **numpy** | Yes | `tinker_cookbook/rl/train.py`, `tinker_cookbook/utils/misc_utils.py`, etc. | core + rl + preference | Native Elixir (Nx) |
| **torch** | Yes | `tinker_cookbook/renderers.py`, `tinker_cookbook/supervised/common.py`, `tinker_cookbook/rl/*`, `tinker_cookbook/preference/*` | core + rl + preference | Native Elixir (Nx tensors, no training) |
| **transformers** | Yes (limited) | `tinker_cookbook/hyperparam_utils.py`, `tinker_cookbook/tests/test_renderers.py` | config/tests | Minimal metadata or stub, tokenizers in Elixir |
| **Pillow (PIL)** | Yes | `tinker_cookbook/renderers.py`, `tinker_cookbook/image_processing_utils.py` | multimodal | Native Elixir image decode |
| **rich** | Yes | `tinker_cookbook/utils/ml_log.py` | logging | Native Elixir (TableRex/IO.ANSI) |
| **termcolor** | Yes | `tinker_cookbook/display.py`, `tinker_cookbook/utils/format_colorized.py` | logging/UI | Native Elixir (IO.ANSI) |
| **blobfile** | Yes | `tinker_cookbook/supervised/data.py` | datasets | Native Elixir (File + ExAws) |
| **asyncio** | Yes (stdlib) | many files (train loops, evals) | core | Native Elixir concurrency |
| **cloudpickle** | Yes | `tinker_cookbook/xmux/core.py` | xmux tooling | Native Elixir serialization or skip |
| **pydantic** | Yes | `tinker_cookbook/renderers.py`, `tinker_cookbook/xmux/*` | renderers/xmux | `sinter` ~> 0.0.1 (Hex) |
| **tqdm** | Yes | `tinker_cookbook/recipes/prompt_distillation/create_data.py` | prompt_distillation | Native Elixir progress or optional |

---

## Optional / Recipe-Specific Dependencies

| Python lib | Used? | Where (examples) | Scope | Plan |
|---|---|---|---|---|
| **textarena** | Yes | `tinker_cookbook/recipes/multiplayer_rl/text_arena/env.py` | multiplayer_rl | Optional; snakebridge if needed |
| **verifiers** | Yes | `tinker_cookbook/recipes/verifiers_rl/*` | verifiers_rl | Optional; snakebridge if needed |
| **openai** | Yes | `tinker_cookbook/recipes/verifiers_rl/tinker_openai.py` | verifiers_rl | Optional; HTTP client or snakebridge |
| **chromadb** | Yes | `tinker_cookbook/recipes/tool_use/search/tools.py` | tool_use/search | Optional; `chroma` HTTP client |
| **google-genai** | Yes (`google.genai`) | `tinker_cookbook/recipes/tool_use/search/embedding.py` | tool_use/search | Optional; HTTP client or snakebridge |
| **huggingface_hub** | Yes | `tinker_cookbook/recipes/tool_use/search/search_env.py`, `tinker_cookbook/hyperparam_utils.py` | tool_use/search | Optional; may be replaced |
| **inspect-ai** | Yes | `tinker_cookbook/eval/*` | evals | Optional; prefer native crucible_harness |

---

## Listed in pyproject.toml but **not found** in code imports

| Python lib | Used? | Notes |
|---|---|---|
| **torchvision** | No | Not imported in `tinker_cookbook` |
| **anyio** | No | Not imported; asyncio is used directly |
| **wandb** | No | Optional dependency only |
| **neptune-scale** | No | Optional dependency only |
| **trackio** | No | Optional dependency only |
