# crucible_datasets Deep Dive: What Actually Exists

**Date:** 2025-12-20
**Location:** `/home/home/p/g/North-Shore-AI/crucible_datasets/`
**Status:** IMPLEMENTED (v0.4.1) - This doc is historical

---

## Executive Summary

**Update (2025-12-22):** `crucible_datasets` v0.4.1 now includes HuggingFace Hub
fetching via `hf_hub` (list_repo_tree/dataset_splits/download), real loaders for
all tinker datasets, DatasetDict/IterableDataset, streaming, and image features.
Use this doc as historical context; current implementation status lives in
`crucible_datasets` docs (`docs/20251220/implementation_status.md`).

`crucible_datasets` is a **well-architected but incomplete** dataset library:

| Component | Status | Notes |
|-----------|--------|-------|
| Dataset struct/schema | ✅ Complete | Validation, checksums |
| Caching infrastructure | ✅ Complete | TTL, versioning, manifest |
| Sampling utilities | ✅ Complete | Random, stratified, k-fold |
| Evaluation metrics | ✅ Complete | ExactMatch, F1, BLEU, ROUGE |
| Result storage | ✅ Complete | Persistence, querying |
| Export formats | ✅ Complete | CSV, JSONL, MD, HTML |
| **Remote fetching** | ❌ **NOT IMPLEMENTED** | Generates synthetic data |
| **Data parsers** | ⚠️ Written but unused | Never called |

---

## Dependencies (mix.exs)

### Current Production Dependencies

```elixir
defp deps do
  [
    {:jason, "~> 1.4"},           # JSON parsing
    {:telemetry, "~> 1.3"},       # Instrumentation
    {:crucible_ir, "~> 0.1.1"},   # DatasetRef integration
    # Dev-only below
    {:dialyxir, "~> 1.4", only: [:dev], runtime: false},
    {:ex_doc, "~> 0.38", only: :dev, runtime: false}
  ]
end
```

### What's MISSING

| Dependency | Purpose | Status |
|------------|---------|--------|
| `{:req, "~> 0.5"}` | HTTP client | **REMOVED** (was in legacy version) |
| `{:explorer, "~> 0.10"}` | DataFrames | Never added |
| `{:nx, "~> 0.9"}` | Tensors | Never added |
| `{:pythonx, ...}` | Python interop | Never added |

### Evidence: Legacy Version HAD Req

```elixir
# From dataset_manager-0.1.0/mix.exs (OLD VERSION)
defp deps do
  [
    {:req, "~> 0.5"},  # <-- HTTP CLIENT WAS HERE
    {:jason, "~> 1.4"},
    {:ex_doc, "~> 0.31", only: :dev, runtime: false}
  ]
end
```

The HTTP client was **removed** before fetching was implemented.

---

## Remote Fetching: ZERO Implementation

### Search Results

| Pattern | Files Found | Actual HTTP Calls |
|---------|-------------|-------------------|
| `Req.get` | 0 | None |
| `Req.post` | 0 | None |
| `HTTPoison` | 0 | None |
| `Tesla` | 0 | None |
| `Finch` | 0 | None |
| `:httpc` | 0 | None |

**There are NO HTTP calls anywhere in the codebase.**

### Intended Data Sources (Declared but Not Used)

```elixir
# From loader.ex - THIS MAP IS NEVER USED FOR FETCHING
@dataset_sources %{
  mmlu: {:huggingface, "cais/mmlu", "all"},
  mmlu_stem: {:huggingface, "cais/mmlu", "stem"},
  humaneval: {:github, "openai/human-eval", "data/HumanEval.jsonl.gz"},
  gsm8k: {:huggingface, "gsm8k", "main"}
}
```

### What Actually Happens

```elixir
# loader.ex - Dispatches to individual loaders
defp fetch_and_parse({dataset_name, source_spec}, _name, opts) do
  case dataset_name do
    name when name in [:mmlu, :mmlu_stem] -> MMLU.load(dataset_name, opts)
    :humaneval -> HumanEval.load(opts)
    :gsm8k -> GSM8K.load(opts)
    _ -> load_custom(dataset_name, source_spec, opts)
  end
end

# But MMLU.load, HumanEval.load, GSM8K.load all GENERATE SYNTHETIC DATA
```

---

## Loader Implementations: All Synthetic

### GSM8K Loader

**File:** `lib/dataset_manager/loader/gsm8k.ex`

```elixir
@doc """
Load GSM8K dataset.

For demo purposes, generates synthetic data.
In production, would fetch from HuggingFace.
"""
def load(opts \\ []) do
  items = generate_sample_items(opts)  # <-- SYNTHETIC

  dataset = Dataset.new(
    "gsm8k", "1.0", items,
    %{source: "huggingface:gsm8k", license: "MIT", domain: "math_word_problems"}
  )
  {:ok, dataset}
end
```

**Synthetic Data (10 hardcoded problems):**

```elixir
problems = [
  {"Natalie sold clips to 48 of her friends in April...", 72, 2},
  {"Weng earns $12 an hour for babysitting...", 10, 2},
  {"Betty is saving money for a new wallet...", 50, 3},
  {"Julie is reading a 120-page book...", 12, 2},
  {"James writes a 3-page letter to 2 different friends...", 24, 3},
  {"Mark has a garden with flowers...", 90, 3},
  {"Albert is wondering how much pizza he can eat...", 14, 4},
  {"Ken created a care package to send...", 162, 5},
  {"Alexis is applying for a new job...", 42, 6},
  {"Tina makes $18.00 an hour...", 111, 4}
]
```

**Parser EXISTS but is NEVER CALLED:**

```elixir
def parse_jsonl(content) do
  content
  |> String.split("\n", trim: true)
  |> Enum.with_index()
  |> Enum.map(fn {line, idx} ->
    case Jason.decode(line) do
      {:ok, data} ->
        answer = extract_numerical_answer(data["answer"])
        %{
          id: "gsm8k_#{idx}",
          input: %{question: data["question"]},
          expected: answer,
          metadata: %{raw_answer: data["answer"]}
        }
      {:error, _} -> nil
    end
  end)
  |> Enum.reject(&is_nil/1)
end

def extract_numerical_answer(answer_text) do
  # Parses "#### 42" format from real GSM8K
  case Regex.run(~r/####\s*([0-9,]+(?:\.[0-9]+)?)/, answer_text || "") do
    [_, number_str] -> # ... conversion logic
    _ -> nil
  end
end
```

### MMLU Loader

**File:** `lib/dataset_manager/loader/mmlu.ex`

```elixir
def load(dataset_name, opts \\ []) do
  # In production, this would fetch from HuggingFace:
  # url = "https://huggingface.co/datasets/cais/mmlu"
  # For now, generate synthetic data for testing

  subjects = case dataset_name do
    :mmlu_stem -> @stem_subjects  # ["physics", "chemistry", "biology", ...]
    :mmlu -> @stem_subjects ++ ["history", "philosophy", "law"]
  end

  items = generate_sample_items(subjects, opts)  # <-- SYNTHETIC
```

**Synthetic Data:**

```elixir
%{
  id: "mmlu_#{subject}_#{i}",
  input: %{
    question: "Sample #{subject} question #{i}?",  # <-- FAKE
    choices: ["Option A", "Option B", "Option C", "Option D"]
  },
  expected: correct_answer,
  metadata: %{subject: subject, difficulty: Enum.random(["easy", "medium", "hard"])}
}
```

**Parser EXISTS but NEVER CALLED:**

```elixir
def parse_csv(content, subject) do
  content
  |> String.split("\n", trim: true)
  |> Enum.with_index()
  |> Enum.map(fn {line, idx} ->
    case String.split(line, ",") do
      [question, a, b, c, d, answer] ->
        %{
          id: "mmlu_#{subject}_#{idx}",
          input: %{question: question, choices: [a, b, c, d]},
          expected: answer,
          metadata: %{subject: subject}
        }
      _ -> nil
    end
  end)
  |> Enum.reject(&is_nil/1)
end
```

### HumanEval Loader

**File:** `lib/dataset_manager/loader/human_eval.ex`

```elixir
def load(opts \\ []) do
  # In production, would fetch from:
  # https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz

  items = generate_sample_items(opts)  # <-- SYNTHETIC
```

**Synthetic Data (10 hardcoded problems):**

```elixir
problems = [
  {"has_close_elements", "list of numbers", "Check if any two numbers are closer than threshold"},
  {"separate_paren_groups", "string", "Separate nested parentheses groups"},
  {"truncate_number", "float", "Return the decimal part of a number"},
  {"below_zero", "list of operations", "Check if balance goes below zero"},
  {"mean_absolute_deviation", "list of numbers", "Calculate mean absolute deviation"},
  {"intersperse", "list and delimiter", "Insert delimiter between elements"},
  {"parse_nested_parens", "string", "Find max nesting depth of parentheses"},
  {"filter_by_substring", "list and substring", "Filter strings containing substring"},
  {"sum_product", "list of integers", "Return sum and product"},
  {"rolling_max", "list of integers", "Return rolling maximum"}
]
```

---

## Registry: Complete Metadata

**File:** `lib/dataset_manager/registry.ex`

The registry has **complete metadata** for intended datasets:

```elixir
@datasets %{
  mmlu: %{
    name: :mmlu,
    description: "Massive Multitask Language Understanding benchmark",
    version: "1.0",
    loader: MMLU,
    domain: "general_knowledge",
    num_items: 15908,
    splits: [:test, :dev, :train],
    source_url: "https://huggingface.co/datasets/cais/mmlu",
    citation: "Hendrycks et al., 2021",
    subjects: [
      "abstract_algebra", "anatomy", "astronomy", "business_ethics",
      # ... 50+ subjects
    ]
  },

  humaneval: %{
    name: :humaneval,
    description: "OpenAI's HumanEval code generation benchmark",
    version: "1.0",
    loader: HumanEval,
    domain: "code_generation",
    num_items: 164,
    language: "python",
    source_url: "https://github.com/openai/human-eval",
    citation: "Chen et al., 2021"
  },

  gsm8k: %{
    name: :gsm8k,
    description: "Grade School Math 8K dataset",
    version: "1.0",
    loader: GSM8K,
    domain: "math_word_problems",
    num_items: 8500,
    difficulty: "grade_school",
    source_url: "https://huggingface.co/datasets/gsm8k",
    citation: "Cobbe et al., 2021"
  }
}
```

---

## What's Actually Complete

### Production-Ready Modules

| Module | Lines | Status | Description |
|--------|-------|--------|-------------|
| `CrucibleDatasets` | 150+ | ✅ | Main API facade |
| `Dataset` | 100+ | ✅ | Struct with validation, checksums |
| `Cache` | 200+ | ✅ | TTL caching, versioning, manifest |
| `Evaluator` | 100+ | ✅ | Evaluation orchestration |
| `Evaluator.ExactMatch` | 80+ | ✅ | Multi-type comparison |
| `Evaluator.F1` | 100+ | ✅ | Token-level precision/recall |
| `Evaluator.BLEU` | 150+ | ✅ | Full BLEU with smoothing |
| `Evaluator.ROUGE` | 150+ | ✅ | ROUGE-1, ROUGE-2, ROUGE-L |
| `EvaluationResult` | 50+ | ✅ | Result struct + JSON |
| `Sampler` | 150+ | ✅ | Random, stratified, k-fold |
| `Registry` | 100+ | ✅ | Dataset discovery |
| `ResultStore` | 150+ | ✅ | Persistent results |
| `Exporter` | 200+ | ✅ | CSV, JSONL, MD, HTML |

### Placeholder/Stub Modules

| Module | Status | What's Missing |
|--------|--------|----------------|
| `Loader.GSM8K` | Stub | Real HuggingFace fetch |
| `Loader.MMLU` | Stub | Real HuggingFace fetch |
| `Loader.HumanEval` | Stub | Real GitHub fetch |

---

## To Complete crucible_datasets

### Step 1: Add HTTP Client

```elixir
# mix.exs
defp deps do
  [
    {:req, "~> 0.5"},  # Add back the HTTP client
    {:jason, "~> 1.4"},
    {:telemetry, "~> 1.3"},
    {:crucible_ir, "~> 0.1.1"}
  ]
end
```

### Step 2: Implement HuggingFace Fetcher

```elixir
defmodule CrucibleDatasets.Fetcher.HuggingFace do
  @base_url "https://huggingface.co/datasets"

  def fetch(repo_id, opts \\ []) do
    split = Keyword.get(opts, :split, "train")
    config = Keyword.get(opts, :config, "default")

    # HuggingFace datasets API uses parquet files
    url = "#{@base_url}/#{repo_id}/resolve/main/#{config}/#{split}-00000-of-00001.parquet"

    case Req.get(url) do
      {:ok, %{status: 200, body: body}} ->
        {:ok, parse_parquet(body)}
      {:ok, %{status: status}} ->
        {:error, "HTTP #{status}"}
      {:error, reason} ->
        {:error, reason}
    end
  end
end
```

### Step 3: Wire Up Existing Parsers

```elixir
# In gsm8k.ex - change from:
def load(opts \\ []) do
  items = generate_sample_items(opts)  # Remove this
  # ...
end

# To:
def load(opts \\ []) do
  with {:ok, content} <- Fetcher.HuggingFace.fetch("gsm8k", opts),
       items <- parse_jsonl(content) do  # USE EXISTING PARSER
    dataset = Dataset.new("gsm8k", "1.0", items, metadata())
    {:ok, dataset}
  end
end
```

---

## Effort to Complete

| Task | Effort | Notes |
|------|--------|-------|
| Add Req dependency | 5 min | Just add to mix.exs |
| HuggingFace fetcher | 2-3 days | Parquet/JSON handling |
| GitHub fetcher | 1 day | Simpler, just JSONL |
| Wire GSM8K | 2 hours | Parser already exists |
| Wire MMLU | 2 hours | Parser already exists |
| Wire HumanEval | 2 hours | Parser already exists |
| Add new datasets (17+) | 2-3 weeks | Per tinker-cookbook needs |
| **Total** | **3-4 weeks** | To match cookbook requirements |

---

## Key Insight

**crucible_datasets is 70% complete** - the hard parts (evaluation, caching, sampling) are done. What's missing is:

1. HTTP client dependency (removed, needs re-adding)
2. Actual fetch implementation (never written)
3. Wiring parsers to fetchers (parsers exist but unused)

This is **NOT a 12-17 week project** if we build on crucible_datasets. It's **3-4 weeks** to:
- Re-add Req
- Implement fetchers
- Wire up existing parsers
- Add the specific datasets cookbook needs

---

**Document Status:** Complete
**Last Updated:** 2025-12-20
