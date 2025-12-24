# SnakeBridge Python Libraries (Complete List for Cookbook)

**Updated:** 2025-12-23

This document lists **all Python libraries intended for SnakeBridge integration**
based on actual import usage in `tinker_cookbook`. These are the only libs that
should be kept in Python; everything else is planned as native Elixir.

---

## Required (math_rl recipe only)

| Python lib | Where used | Why SnakeBridge |
|---|---|---|
| **sympy** | `recipes/math_rl/math_grading.py` | Symbolic math comparison |
| **pylatexenc** | `recipes/math_rl/math_grading.py` | LaTeX to text parsing |
| **math_verify** | `recipes/math_rl/math_grading.py` | Equivalence verification |

**Note:** These are built-in SnakeBridge manifests (v0.3.0). These are the ONLY
three libraries that require Python for core cookbook functionality.

---

## Optional / Recipe-Specific (Skip Unless Needed)

| Python lib | Where used | Elixir Alternative |
|---|---|---|
| **textarena** | `recipes/multiplayer_rl/text_arena/env.py` | SnakeBridge if needed |
| **verifiers** | `recipes/verifiers_rl/*` | SnakeBridge if needed |
| **openai** | `recipes/verifiers_rl/tinker_openai.py` | `openai_ex` (~> 0.8) |
| **chromadb** | `recipes/tool_use/search/tools.py` | `chroma` (~> 0.1.2) |
| **google-genai** | `recipes/tool_use/search/embedding.py` | `gemini_ex` (~> 0.8) |
| **huggingface_hub** | `hyperparam_utils.py` | `hf_hub` (~> 0.1) |
| **inspect-ai** | `eval/*` | Native `crucible_harness` preferred |

---

## BEAM-Obviated (No Package Needed)

These Python libs are replaced by BEAM/OTP primitives - no Elixir package needed:

| Python lib | BEAM Replacement | Notes |
|---|---|---|
| `asyncio` / `anyio` | `Task`, `GenServer` | Native lightweight processes |
| `cloudpickle` | `:erlang.term_to_binary` | ETF serialization |
| `tqdm` | `Logger` / Telemetry | Progress via events |
| `threading` | `Task.async_stream` | Native parallelism |

---

## Native Elixir Replacements (Hex Packages)

| Python lib | Elixir Package | Hex |
|---|---|---|
| `numpy` | `Nx` | `{:nx, "~> 0.9"}` |
| `torch` (tensors) | `Nx` | `{:nx, "~> 0.9"}` |
| `scipy` | Native Elixir | 1 function only |
| `PIL` | `Image` | `{:image, "~> 0.47"}` |
| `rich` | `TableRex` | `{:table_rex, "~> 4.0"}` |
| `termcolor` | `IO.ANSI` | Built-in |
| `blobfile` | `File` + `ExAws.S3` | `{:ex_aws_s3, "~> 2.5"}` |
| `pydantic` | `Sinter` | `{:sinter, "~> 0.0.1"}` |
| `transformers` | `Tokenizers` | via `tinkex` |

---

## Direct Elixir Mappings (Already Done)

| Python lib | Elixir Package | Status |
|---|---|---|
| `tinker` | `tinkex` | Done (~> 0.3.2) |
| `chz` | `chz_ex` | Done (~> 0.1.2) |
| `datasets` | `hf_datasets_ex` | Done (~> 0.1) |
| `huggingface_hub` | `hf_hub` | Done (~> 0.1) |
| `google-genai` | `gemini_ex` | Done (~> 0.8) |

---

## Manifest Notes

- SnakeBridge v0.3.0 includes built-in manifests for `sympy`, `pylatexenc`,
  and `math_verify`.
- Optional libs should be added with **curated manifests only** (never full APIs).
- Keep `allow_unsafe: false` unless explicitly approved for exploration.
- See `docs/20251223/PYTHON_TO_ELIXIR_LIBRARY_MAPPING.md` for complete reference.
