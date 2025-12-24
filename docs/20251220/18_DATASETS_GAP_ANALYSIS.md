# Datasets Gap Analysis: The Real Scope

**Date:** 2025-12-20
**Status:** CRITICAL FINDING - This was the major undertaking (historical)
**Effort:** 12-17 weeks (3-4 months) for full parity (historical)

---

## Status Update (2025-12-22)

This gap is now closed. `crucible_datasets` v0.4.1 (Hex) implements HuggingFace Hub
loading, `load_dataset/2` parity, streaming, DatasetDict/IterableDataset, features,
and all 22 tinker datasets. The Pythonx wrapper recommendation below is no longer
required for the cookbook's required datasets.

**Readiness:** The cookbook can now depend on `{:crucible_datasets, "~> 0.4.1"}` and
replace `datasets.load_dataset(...)` calls with `CrucibleDatasets.load_dataset/2`
or the dataset-specific loaders.

**Known limits:** Parquet streaming remains batch-based; HF gated datasets require
`HF_TOKEN`; Explorer/Polars requires `rechunk: true` (already applied in
crucible_datasets).

---

## Executive Summary

The `datasets` dependency is NOT a simple wrapper. It's a **3-4 month project** that was already partially started as `crucible_datasets` but never completed.

| What We Thought | What It Actually Is |
|-----------------|---------------------|
| "Wrap HuggingFace datasets" | Build complete data pipeline infrastructure |
| 1-2 weeks | **12-17 weeks** |
| Simple HTTP wrapper | Full-featured dataset library |

---

## Part 1: tinker-cookbook Dataset Usage

### 20+ HuggingFace Datasets Used

| Dataset | Domain | Usage | Size |
|---------|--------|-------|------|
| `openai/gsm8k` | Math (Grade School) | RL training/eval | 8.5K |
| `HuggingFaceH4/MATH-500` | Math (Competition) | Eval | 500 |
| `EleutherAI/hendrycks_math` | Math (Competition) | Training | 12.5K |
| `zwhe99/DeepMath-103K` | Math (Reasoning) | Training | 103K |
| `POLARIS-Project/Polaris-Dataset-53K` | Math | Training | 53K |
| `agentica-org/DeepCoder-Preview-Dataset` | Code | RL training/eval | Large |
| `allenai/tulu-3-sft-mixture` | Chat/Instruction | SFT training | 326K |
| `HuggingFaceH4/no_robots` | Chat/Instruction | SFT training | 10K |
| `open-thoughts/OpenThoughts3-1.2M` | Reasoning | Distillation | 1.2M |
| `Anthropic/hh-rlhf` | Preference | DPO/RLHF | 170K |
| `nvidia/HelpSteer3` | Preference | DPO | 40K |
| `nvidia/HelpSteer2` | Preference | DPO | 37K |
| `argilla/ultrafeedback-binarized-preferences` | Preference | DPO | 61K |
| `lmarena-ai/arena-human-preference-140k` | Preference | DPO | 140K |
| `allenai/llama-3.1-tulu-3-8b-preference-mixture` | Preference | DPO | Large |
| `prometheus-eval/Feedback-Collection` | Rubric/Eval | Rubric training | 100K |
| `dpdl-benchmark/caltech101` | Vision | VLM classifier | 9K |
| `dpdl-benchmark/oxford_flowers102` | Vision | VLM classifier | 8K |
| `dpdl-benchmark/oxford_iiit_pet` | Vision | VLM classifier | 7K |
| `tanganke/stanford_cars` | Vision | VLM classifier | 16K |

### Processing Operations Used

```python
# Loading
datasets.load_dataset(name, config, split)
datasets.load_dataset(..., streaming=True)  # For 1M+ datasets

# Transformations
.shuffle(seed=N)
.filter(lambda x: condition)
.select(range(start, end))
.take(n) / .skip(n)
.batch(batch_size, drop_last_batch=True)
.map(transform_fn)

# Combining
concatenate_datasets([ds1, ds2, ...])
get_dataset_config_names(name)
```

### Files Using Datasets

| File | Datasets | Operations |
|------|----------|------------|
| `distillation/datasets.py` | DeepMath-103K, tulu-3-sft | Prompt extraction |
| `recipes/preference/datasets.py` | 6 preference datasets | Comparison building |
| `recipes/chat_sl/chat_datasets.py` | tulu-3-sft, no_robots | Conversation processing |
| `recipes/math_rl/math_env.py` | 5 math datasets | Concatenation, filtering |
| `recipes/code_rl/code_env.py` | DeepCoder (4 configs) | Multi-config loading |
| `recipes/vlm_classifier/data.py` | 4 vision datasets | Few-shot sampling |
| `supervised/data.py` | Generic wrapper | Streaming, batching |
| `preference/preference_datasets.py` | Generic wrapper | JSONL loading |
| `recipes/sl_loop.py` | no_robots | Basic loading |
| `recipes/rl_loop.py` | gsm8k | Basic loading |
| `recipes/distillation/off_policy_reasoning.py` | OpenThoughts-1.2M | Streaming |
| `recipes/rubric/data.py` | Feedback-Collection | Rubric extraction |

---

## Part 2: crucible_datasets Current State

### Location: `/home/home/p/g/North-Shore-AI/crucible_datasets/`

### What Exists (Partial Implementation)

```
CrucibleDatasets/
├── Dataset           # Schema struct ✓
├── Loader            # Unified loader interface ✓
│   ├── MMLU          # SYNTHETIC ONLY (placeholder)
│   ├── HumanEval     # SYNTHETIC ONLY (placeholder)
│   └── GSM8K         # SYNTHETIC ONLY (placeholder)
├── Cache             # Local caching ✓
├── Sampler           # Sampling utilities ✓
│   ├── random/2      ✓
│   ├── stratified/2  ✓
│   ├── k_fold/2      ✓
│   └── train_test_split/2 ✓
├── Evaluator         # Evaluation metrics ✓
│   ├── ExactMatch    ✓
│   ├── F1            ✓
│   ├── BLEU          ✓
│   └── ROUGE         ✓
├── Registry          # Dataset registry ✓
├── ResultStore       # Result persistence ✓
└── Exporter          # CSV, JSONL, MD, HTML ✓
```

### Critical Gap: Loaders Are Synthetic

```elixir
# Current GSM8K loader - GENERATES FAKE DATA
defmodule CrucibleDatasets.Loader.GSM8K do
  def load(_opts \\ []) do
    # Returns synthetic sample problems, NOT real GSM8K data
    %Dataset{
      name: "gsm8k",
      items: [
        %{question: "If John has 5 apples...", answer: "3"}
        # Synthetic, not from HuggingFace
      ]
    }
  end
end
```

### What Works Well

| Feature | Status | Notes |
|---------|--------|-------|
| Sampling (random, stratified, k-fold) | ✅ Complete | Better than Python |
| Evaluation metrics (EM, F1, BLEU, ROUGE) | ✅ Complete | Good coverage |
| Result persistence | ✅ Complete | JSONL storage |
| Export (CSV, JSONL, MD, HTML) | ✅ Complete | Multiple formats |
| Caching | ✅ Complete | ~/.elixir_ai_research/ |
| Dataset schema | ✅ Complete | Validated struct |

---

## Part 3: Gap Analysis

### Loader Gaps (20+ datasets missing)

| Dataset | crucible_datasets | Status |
|---------|-------------------|--------|
| `openai/gsm8k` | Synthetic placeholder | ❌ Needs real HF fetch |
| `HuggingFaceH4/MATH-500` | Not implemented | ❌ New loader needed |
| `EleutherAI/hendrycks_math` | Not implemented | ❌ New loader needed |
| `zwhe99/DeepMath-103K` | Not implemented | ❌ New loader needed |
| `allenai/tulu-3-sft-mixture` | Not implemented | ❌ New loader needed |
| `HuggingFaceH4/no_robots` | Not implemented | ❌ New loader needed |
| 6 preference datasets | Not implemented | ❌ New loaders needed |
| 4 vision datasets | Not implemented | ❌ + Image handling |
| `open-thoughts/OpenThoughts3-1.2M` | Not implemented | ❌ + Streaming |
| MMLU | Synthetic placeholder | ❌ Needs real HF fetch |
| HumanEval | Synthetic placeholder | ❌ Needs GitHub fetch |

### Processing Operation Gaps

| Operation | Python | Elixir | Gap |
|-----------|--------|--------|-----|
| `load_dataset()` | Full HF integration | Synthetic only | ❌ **MAJOR** |
| `.shuffle(seed)` | Native | Via Sampler | ✅ OK |
| `.select(range)` | Native | Via Enum | ✅ OK |
| `.filter(fn)` | Native | Via Enum | ✅ OK |
| `.take(n)/.skip(n)` | Native | Via Enum | ✅ OK |
| `concatenate_datasets` | Native | Not implemented | ❌ Need |
| `.batch()` | Native | Not implemented | ❌ Need |
| Streaming | Native | Not implemented | ❌ **MAJOR** |
| Config selection | Native | Basic | ⚠️ Improve |

### Data Type Gaps

| Type | Python | Elixir | Gap |
|------|--------|--------|-----|
| SFT messages format | Full support | Not implemented | ❌ Need Message type |
| Preference pairs | Full support | Not implemented | ❌ Need Comparison type |
| Conversations | Full support | Not implemented | ❌ Need Conversation type |
| Image data | Full support | Not implemented | ❌ Need image handling |

### Grading/Evaluation Gaps

| Feature | Python | Elixir | Gap |
|---------|--------|--------|-----|
| Math grading (sympy) | `grade_answer` | Not implemented | ❌ Via Snakepit |
| Boxed answer extraction | `extract_boxed` | Not implemented | ❌ Need regex |
| GSM8K answer extraction | Full | Partial | ⚠️ Improve |
| Code sandbox execution | `sandbox_check_correctness` | Not implemented | ❌ **MAJOR** |

---

## Part 4: Effort Estimates

### Detailed Breakdown

| Component | Effort | Notes |
|-----------|--------|-------|
| **HuggingFace API Client** | 2-3 weeks | HTTP client, auth, parquet/JSON parsing |
| **Math Datasets (5)** | 2 weeks | GSM8K, MATH-500, hendrycks, DeepMath, POLARIS |
| **Chat/SFT Datasets (2)** | 1 week | tulu-3, no_robots + message types |
| **Preference Datasets (6)** | 2 weeks | All comparison datasets + types |
| **Code Datasets (1)** | 1 week | DeepCoder + multi-config |
| **Vision Datasets (4)** | 2 weeks | Image handling required |
| **Streaming Support** | 1-2 weeks | For 1M+ datasets |
| **Data Processing Ops** | 1 week | concatenate, batch, etc. |
| **Message/Conversation Types** | 1 week | Schema + conversions |
| **Preference Data Types** | 1 week | Comparison builders |
| **Math Grading** | 1-2 weeks | Sympy integration |
| **Testing & Documentation** | 2-3 weeks | Comprehensive coverage |

### Total: **12-17 weeks (3-4 months)**

---

## Part 5: Recommended Strategy

### Option A: Full Native Port (12-17 weeks)

Build complete `crucible_datasets` with real HuggingFace integration.

**Pros:**
- Pure Elixir, no Python dependency
- Full control over implementation
- Integrates with existing crucible ecosystem

**Cons:**
- 3-4 months of work
- Duplicating mature Python library
- Ongoing maintenance burden

### Option B: Pythonx Wrapper (3-4 weeks)

Wrap Python `datasets` library via Pythonx/Snakepit.

```elixir
defmodule Tinkex.Datasets do
  use Pythonx

  def load_dataset(name, opts \\ []) do
    Pythonx.call(:datasets, :load_dataset, [name, opts])
    |> to_explorer_dataframe()
  end
end
```

**Pros:**
- 3-4 weeks instead of 3-4 months
- Full HuggingFace compatibility
- Access to all 20+ datasets immediately

**Cons:**
- Python dependency
- Serialization overhead
- Less "pure Elixir"

### Option C: Hybrid (6-8 weeks)

1. Use Pythonx for HuggingFace downloads only
2. Convert to Explorer DataFrames
3. All processing in native Elixir
4. Keep existing crucible_datasets evaluation infrastructure

```elixir
# Download via Pythonx
raw = Tinkex.Datasets.HuggingFace.download("openai/gsm8k", split: "train")

# Convert to Explorer
df = Explorer.DataFrame.new(raw)

# Process natively
df
|> DataFrame.shuffle(seed: 42)
|> DataFrame.slice(0, 1000)
|> DataFrame.to_rows()
```

**Pros:**
- Best of both worlds
- Native processing performance
- Leverage existing crucible_datasets
- Reasonable timeline

**Cons:**
- Still needs Python for downloads
- Some complexity in conversion layer

### Recommendation: **Option C (Hybrid)**

| Phase | Weeks | Deliverable |
|-------|-------|-------------|
| 1 | 1-2 | Pythonx HuggingFace download wrapper |
| 2 | 1-2 | HF → Explorer DataFrame conversion |
| 3 | 1-2 | Native processing ops (concat, batch, stream) |
| 4 | 1-2 | Message/Conversation/Comparison types |
| 5 | 1-2 | Integrate with crucible_datasets evaluators |
| **Total** | **5-8 weeks** | Full cookbook dataset support |

---

## Part 6: What crucible_datasets Already Does Well

Don't rebuild these - they're done:

| Feature | Module | Status |
|---------|--------|--------|
| Random sampling | `Sampler.random/2` | ✅ Production ready |
| Stratified sampling | `Sampler.stratified/2` | ✅ Production ready |
| K-fold CV | `Sampler.k_fold/2` | ✅ Production ready |
| Train/test split | `Sampler.train_test_split/2` | ✅ Production ready |
| Exact match eval | `Evaluator.ExactMatch` | ✅ Production ready |
| F1 scoring | `Evaluator.F1` | ✅ Production ready |
| BLEU scoring | `Evaluator.BLEU` | ✅ Production ready |
| ROUGE scoring | `Evaluator.ROUGE` | ✅ Production ready |
| Result persistence | `ResultStore` | ✅ Production ready |
| Multi-format export | `Exporter` | ✅ Production ready |
| Local caching | `Cache` | ✅ Production ready |

---

## Conclusion

The `datasets` gap is the **single largest piece of work** for the tinkex_cookbook port:

| Original Estimate | Actual (Full Native) | Actual (Hybrid) |
|-------------------|----------------------|-----------------|
| 1-2 weeks | **12-17 weeks** | **5-8 weeks** |

**The hybrid approach (Pythonx for downloads, native for processing) cuts the timeline in half while maintaining Elixir-native data processing.**

---

**Document Status:** Complete
**Last Updated:** 2025-12-20
