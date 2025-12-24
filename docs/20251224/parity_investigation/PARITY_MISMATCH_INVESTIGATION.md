# Parity Mismatch Investigation: sl_basic Recipe

**Date:** 2024-12-24
**Status:** Investigation Required
**Priority:** High

## Executive Summary

The parity harness comparing Python and Elixir implementations of the `sl_basic` recipe reveals systematic mismatches. This document analyzes the root causes and provides isolated test cases for each sub-library where deviations occur.

## Observed Mismatches

| Artifact | Status | Evidence |
|----------|--------|----------|
| config.json | MATCH | Both use llama3 renderer, same hyperparams |
| dataset_snapshot.json | MISMATCH | Different samples for same indices |
| rendered_samples.json | MISMATCH | Different token sequences |
| first_batch_payload.json | MISMATCH | Different datum hashes |
| metrics.jsonl | MISMATCH | Format differences (status lines) |

## Root Cause Analysis

### 1. Dataset Ordering Discrepancy (PRIMARY CAUSE)

**Libraries involved:**
- Python: `datasets` (HuggingFace)
- Elixir: `hf_datasets_ex`

**Evidence:**

Python (`tinker-cookbook/tinker_cookbook/recipes/chat_sl/chat_datasets.py:60`):
```python
train_dataset = train_dataset.shuffle(seed=0)
```

Elixir (`lib/tinkex_cookbook/datasets/no_robots.ex:65-70`):
```elixir
case HfDatasetsEx.load_dataset(@dataset_name, split: split) do
  {:ok, dataset} ->
    samples =
      dataset
      |> maybe_limit(limit)  # No shuffle!
      |> Enum.to_list()
```

**Impact:** The first 4 samples in Python (post-shuffle) differ from the first 4 samples in Elixir (pre-shuffle, original order).

**Observed in artifacts:**
- Elixir dataset_snapshot shows sample 0 content_hash: `ddb89ca8bdec...`
- Python would show different hash due to shuffle reordering

#### Isolated Test Case #1: Dataset Ordering

```bash
# Test: Verify dataset ordering differs between implementations

# Python test
cd tinker-cookbook
python3 -c "
import datasets
ds = datasets.load_dataset('HuggingFaceH4/no_robots')['train']

# Without shuffle
print('=== Python: No shuffle (first 3 samples) ===')
for i in range(3):
    msg = ds[i]['messages'][0]['content'][:80]
    print(f'{i}: {msg}...')

# With shuffle(seed=0) - this is what sl_basic does
ds_shuffled = ds.shuffle(seed=0)
print()
print('=== Python: With shuffle(seed=0) (first 3 samples) ===')
for i in range(3):
    msg = ds_shuffled[i]['messages'][0]['content'][:80]
    print(f'{i}: {msg}...')
"

# Elixir test (from tinkex_cookbook root)
cd ..
mix run -e "
{:ok, samples} = HfDatasetsEx.load_dataset(\"HuggingFaceH4/no_robots\", split: \"train\")
samples = samples |> Enum.take(3) |> Enum.to_list()

IO.puts(\"=== Elixir: No shuffle (first 3 samples) ===\")
Enum.with_index(samples, fn sample, i ->
  msg = sample[\"messages\"] |> hd() |> Map.get(\"content\") |> String.slice(0, 80)
  IO.puts(\"\#{i}: \#{msg}...\")
end)
"
```

**Expected outcome:** Python shuffled samples differ from Elixir samples, but Python unshuffled should match Elixir.

---

### 2. Renderer Token Sequences (SECONDARY)

**Libraries involved:**
- Python: `tinker_cookbook.renderers.Llama3Renderer`
- Elixir: `TinkexCookbook.Renderers.Llama3`

**Status:** Likely MATCHING (pending dataset alignment)

Both implementations use identical Llama3 chat template:
```
<|begin_of_text|><|start_header_id|>{role}<|end_header_id|>

{content}<|eot_id|>
```

**Evidence from code:**

Python (`tinker-cookbook/tinker_cookbook/renderers.py:531-533`):
```python
ob_str = f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n"
ac_str = f"{message['content']}<|eot_id|>"
```

Elixir (`lib/tinkex_cookbook/renderers/llama3.ex:106-110`):
```elixir
prefix_str = "#{@start_header}#{role}#{@end_header}\n\n"
content_str = "#{content}#{@eot_id}"
```

**Key tokens (Llama 3.1-8B):**
- BOS: 128000 (`<|begin_of_text|>`)
- start_header_id: 128006
- end_header_id: 128007
- eot_id: 128009

#### Isolated Test Case #2: Renderer Token Parity

```bash
# Test: Compare rendered tokens for identical input message

# Create shared test message
TEST_MESSAGE='{"role": "user", "content": "Hello, world!"}'

# Python test
cd tinker-cookbook
python3 -c "
from transformers import AutoTokenizer
from tinker_cookbook.renderers import Llama3Renderer

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B')
renderer = Llama3Renderer(tokenizer)

message = {'role': 'user', 'content': 'Hello, world!'}
rendered = renderer.render_message(0, message)

print('=== Python Llama3Renderer ===')
print(f'prefix tokens: {rendered.prefix.tokens}')
print(f'content tokens: {rendered.content[0].tokens}')
print(f'bos tokens: {renderer._bos_tokens}')
"

# Elixir test
cd ..
mix run -e "
alias TinkexCookbook.Renderers.Llama3
alias Tinkex.Tokenizer

{:ok, tokenizer} = Tokenizer.from_pretrained(\"meta-llama/Llama-3.1-8B\")
{:ok, state} = Llama3.init(tokenizer: tokenizer)

message = TinkexCookbook.Renderers.Types.message(\"user\", \"Hello, world!\")
{rendered, _state} = Llama3.render_message(0, message, false, state)

IO.puts(\"=== Elixir Llama3 Renderer ===\")
IO.puts(\"prefix tokens: \#{inspect(rendered.prefix.tokens)}\")
IO.puts(\"content tokens: \#{inspect(hd(rendered.content).tokens)}\")
IO.puts(\"bos tokens: \#{inspect(Llama3.bos_tokens(state))}\")
"
```

**Expected outcome:** Identical token sequences for identical input.

---

### 3. Tokenizer Implementation Differences (POTENTIAL)

**Libraries involved:**
- Python: `transformers.AutoTokenizer`
- Elixir: `Tinkex.Tokenizer` (wraps tokenizers-elixir or calls Python)

**Risk:** Different tokenizer implementations could produce different token IDs for identical strings.

#### Isolated Test Case #3: Tokenizer Parity

```bash
# Test: Compare tokenizer output for edge cases

TEST_STRINGS=(
    "Hello, world!"
    "The quick brown fox"
    "123 + 456 = 579"
    "<|start_header_id|>user<|end_header_id|>"
    "Newlines\n\nand tabs\t\there"
)

# Python test
cd tinker-cookbook
python3 -c "
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B')

test_strings = [
    'Hello, world!',
    'The quick brown fox',
    '123 + 456 = 579',
    '<|start_header_id|>user<|end_header_id|>',
    'Newlines\n\nand tabs\t\there'
]

print('=== Python Tokenizer ===')
for s in test_strings:
    tokens = tokenizer.encode(s, add_special_tokens=False)
    print(f'{repr(s)}: {tokens}')
"

# Elixir test
cd ..
mix run -e "
alias Tinkex.Tokenizer

{:ok, tokenizer} = Tokenizer.from_pretrained(\"meta-llama/Llama-3.1-8B\")

test_strings = [
    \"Hello, world!\",
    \"The quick brown fox\",
    \"123 + 456 = 579\",
    \"<|start_header_id|>user<|end_header_id|>\",
    \"Newlines\\n\\nand tabs\\t\\there\"
]

IO.puts(\"=== Elixir Tokenizer ===\")
Enum.each(test_strings, fn s ->
    tokens = Tokenizer.encode(tokenizer, s, add_special_tokens: false)
    IO.puts(\"\#{inspect(s)}: \#{inspect(tokens)}\")
end)
"
```

**Expected outcome:** Identical token IDs for identical strings.

---

### 4. Datum Construction Pipeline

**Libraries involved:**
- Python: `tinker_cookbook.supervised.data.conversation_to_datum`
- Elixir: `TinkexCookbook.Supervised.Common.datum_from_model_input_weights`

Both implement the right-shift/left-shift transformation for next-token prediction:
- Input: tokens[:-1]
- Target: tokens[1:]
- Weights: shifted to align with targets

#### Isolated Test Case #4: Datum Construction Parity

```bash
# Test: Compare datum construction for identical rendered tokens

# Python test
cd tinker-cookbook
python3 -c "
from transformers import AutoTokenizer
from tinker_cookbook.renderers import Llama3Renderer, TrainOnWhat
from tinker_cookbook.supervised.data import conversation_to_datum
import json

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B')
renderer = Llama3Renderer(tokenizer)

messages = [
    {'role': 'user', 'content': 'Hello'},
    {'role': 'assistant', 'content': 'Hi there!'}
]

datum = conversation_to_datum(
    messages,
    renderer,
    max_length=256,
    train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES
)

print('=== Python Datum ===')
print(f'model_input length: {datum.model_input.length}')
print(f'target_tokens shape: {datum.loss_fn_inputs[\"target_tokens\"].shape}')
print(f'weights shape: {datum.loss_fn_inputs[\"weights\"].shape}')
print(f'first 10 target tokens: {datum.loss_fn_inputs[\"target_tokens\"].data[:10]}')
print(f'first 10 weights: {datum.loss_fn_inputs[\"weights\"].data[:10]}')
"

# Elixir test
cd ..
mix run -e "
alias TinkexCookbook.Renderers.{Llama3, Renderer, TrainOnWhat}
alias TinkexCookbook.Supervised.Common
alias Tinkex.Tokenizer

{:ok, tokenizer} = Tokenizer.from_pretrained(\"meta-llama/Llama-3.1-8B\")
{:ok, state} = Llama3.init(tokenizer: tokenizer)

messages = [
    TinkexCookbook.Renderers.Types.message(\"user\", \"Hello\"),
    TinkexCookbook.Renderers.Types.message(\"assistant\", \"Hi there!\")
]

{model_input, weights} = Renderer.build_supervised_example(
    Llama3,
    messages,
    TrainOnWhat.all_assistant_messages(),
    state
)

datum = Common.datum_from_model_input_weights(model_input, weights, 256)

IO.puts(\"=== Elixir Datum ===\")
IO.puts(\"model_input length: \#{datum.model_input.length}\")
IO.puts(\"target_tokens shape: \#{inspect(datum.loss_fn_inputs.target_tokens.shape)}\")
IO.puts(\"weights shape: \#{inspect(datum.loss_fn_inputs.weights.shape)}\")
IO.puts(\"first 10 target tokens: \#{inspect(Enum.take(datum.loss_fn_inputs.target_tokens.data, 10))}\")
IO.puts(\"first 10 weights: \#{inspect(Enum.take(datum.loss_fn_inputs.weights.data, 10))}\")
"
```

---

### 5. Metrics Logging Format

**Status:** Non-critical format difference

**Evidence:**
- Elixir: Logs status lines with human-readable progress
- Python: Logs structured metrics only

This is a display/logging difference, not a training difference.

---

## Recommended Fix Priority

1. **CRITICAL:** Align dataset shuffle behavior
   - Either: Add `shuffle(seed: 0)` to Elixir HfDatasetsEx loading
   - Or: Remove shuffle from Python for parity testing only
   - Recommended: Make shuffle configurable with matching defaults

2. **HIGH:** Verify tokenizer parity with Test Case #3

3. **MEDIUM:** Verify renderer parity with Test Case #2

4. **LOW:** Verify datum construction with Test Case #4

## Implementation Plan for Fix

### Option A: Add Shuffle to Elixir (Preferred)

```elixir
# lib/tinkex_cookbook/datasets/no_robots.ex

def load(opts \\ []) do
  split = Keyword.get(opts, :split, "train")
  limit = Keyword.get(opts, :limit)
  shuffle_seed = Keyword.get(opts, :shuffle_seed, 0)  # Default to match Python

  case HfDatasetsEx.load_dataset(@dataset_name, split: split) do
    {:ok, dataset} ->
      samples =
        dataset
        |> Enum.to_list()
        |> maybe_shuffle(shuffle_seed)  # Add shuffle step
        |> maybe_limit(limit)

      {:ok, samples}
    ...
  end
end

defp maybe_shuffle(samples, nil), do: samples
defp maybe_shuffle(samples, seed) do
  # Use deterministic shuffle matching Python's algorithm
  :rand.seed(:exsss, {seed, seed, seed})
  Enum.shuffle(samples)
end
```

**WARNING:** Elixir's `Enum.shuffle/1` and Python's `datasets.shuffle()` use different algorithms. This needs further investigation to ensure identical ordering.

### Option B: Bypass Shuffle for Parity Testing

Modify `scripts/parity/run_sl_basic_python.sh` to disable shuffle:

```python
# tinker_cookbook/recipes/sl_basic_parity.py
# Add shuffle_enabled=False for parity mode
```

---

## Investigation Checklist for Agent

- [ ] Run Test Case #1 to confirm dataset ordering hypothesis
- [ ] Run Test Case #2 to verify renderer token parity
- [ ] Run Test Case #3 to verify tokenizer implementation parity
- [ ] Run Test Case #4 to verify datum construction parity
- [ ] Document exact shuffle algorithm used by HuggingFace datasets
- [ ] Implement matching shuffle in Elixir (if needed)
- [ ] Re-run full parity harness after fix
- [ ] Verify MATCH status on all artifacts

---

## Files Modified/Created by Parity Harness

| File | Purpose |
|------|---------|
| `scripts/parity/run_sl_basic_parity.sh` | Combined runner |
| `scripts/parity/run_sl_basic_python.sh` | Python runner |
| `scripts/parity/run_sl_basic_elixir.sh` | Elixir runner |
| `scripts/parity/compare_artifacts.py` | Comparison script |
| `tinker-cookbook/tinker_cookbook/utils/parity.py` | Python parity helpers |
| `lib/tinkex_cookbook/utils/parity.ex` | Elixir parity helpers |
| `tinker-cookbook/tinker_cookbook/recipes/sl_basic_parity.py` | Python parity entry point |

## Artifact Locations

- Python: `/tmp/parity/sl_basic/python/`
- Elixir: `/tmp/parity/sl_basic/elixir/`

---

## References

- Python HuggingFace datasets shuffle: https://huggingface.co/docs/datasets/process#shuffle
- Elixir random seeding: https://hexdocs.pm/elixir/Enum.html#shuffle/1
- Llama 3 chat template: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

---

*Generated by parity investigation agent, 2024-12-24*
