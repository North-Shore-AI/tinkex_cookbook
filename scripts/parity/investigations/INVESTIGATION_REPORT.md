# Parity Investigation Report

Date: 2024-12-24

## Executive Summary

Investigation of parity issues between Python (`tinker-cookbook`) and Elixir (`tinkex_cookbook`) implementations of the `sl_basic` supervised learning recipe revealed **two root causes**:

1. **Issue 2 (BOS Token Count)**: Elixir tokenizer wrapper ignores `add_special_tokens: false` option
2. **Issue 1 (Batch Ordering)**: Different shuffling approaches at different abstraction levels

Issue 3 (Token Length Differences) is a consequence of Issues 1 and 2 combined.

---

## Issue 2: BOS Token Count Mismatch

### THEORY
The Elixir tokenizer wrapper ignores the `add_special_tokens: false` option, causing HuggingFace Tokenizers to add BOS tokens to every encoded string.

### EVIDENCE

**Elixir tokenizer wrapper** (`lib/tinkex_cookbook/recipes/sl_basic.ex:296-311`):
```elixir
defp create_tokenizer_wrapper(tokenizer_handle) do
  %{
    encode: fn text, _opts ->  # <-- _opts is IGNORED!
      case Tokenizers.Tokenizer.encode(tokenizer_handle, text) do
        {:ok, encoding} -> Tokenizers.Encoding.get_ids(encoding)
        {:error, _} -> []
      end
    end,
    ...
  }
end
```

The `_opts` parameter (which contains `add_special_tokens: false`) is never passed to `Tokenizers.Tokenizer.encode/3`.

**Python tokenizer usage** (`tinker-cookbook/tinker_cookbook/renderers.py:544-546`):
```python
@property
def _bos_tokens(self) -> list[int]:
    return self.tokenizer.encode("<|begin_of_text|>", add_special_tokens=False)
```

Python explicitly passes `add_special_tokens=False`, preventing duplicate BOS tokens.

**Observed in parity artifacts**:

Python first datum model_input chunks:
```json
{"type": "encoded_text", "tokens": [128000]},           // 1 BOS token
{"type": "encoded_text", "tokens": [128006, 882, ...]}, // Header (no BOS)
```

Elixir first datum model_input chunks:
```json
{"type": "encoded_text", "tokens": [128000, 128000]},           // 2 BOS tokens!
{"type": "encoded_text", "tokens": [128000, 128006, 882, ...]}, // 1 BOS + Header
{"type": "encoded_text", "tokens": [128000, 3923, ...]},        // 1 BOS + Content
```

Every Elixir chunk has an extra BOS token (128000) prepended.

### CONCLUSION
**ROOT CAUSE CONFIRMED**: Elixir ignores `add_special_tokens: false`, causing BOS to be added to every encoded string.

### FIX
Modify the tokenizer wrapper to pass the option:

```elixir
defp create_tokenizer_wrapper(tokenizer_handle) do
  %{
    encode: fn text, opts ->
      add_special = Keyword.get(opts, :add_special_tokens, true)
      case Tokenizers.Tokenizer.encode(tokenizer_handle, text, add_special_tokens: add_special) do
        {:ok, encoding} -> Tokenizers.Encoding.get_ids(encoding)
        {:error, _} -> []
      end
    end,
    ...
  }
end
```

---

## Issue 1: Batch Ordering Mismatch

### THEORY
Python and Elixir apply shuffling at different abstraction levels with different algorithms, causing samples to appear in different positions within batches.

### EVIDENCE

**Python flow** (`SupervisedDatasetFromHFDataset`):
1. `NoRobotsBuilder.__call__()`: Load HF dataset, shuffle with seed=0
2. Store reference to shuffled dataset
3. `set_epoch(0)`: Re-shuffle the already-shuffled dataset with seed=0
4. `get_batch(0)`: Build datums lazily from the re-shuffled dataset

**Elixir flow** (`SupervisedDatasetFromList`):
1. `NoRobots.load(shuffle_seed: 0)`: Load and shuffle HF dataset
2. `create_supervised_dataset()`: Build ALL datums immediately into a list
3. `set_epoch(0)`: Re-shuffle the **datums list** using `phash2({seed, idx})`
4. `get_batch(0)`: Return slice of re-shuffled datums

**Key differences**:

| Aspect | Python | Elixir |
|--------|--------|--------|
| Datum building | Lazy (during get_batch) | Eager (during dataset creation) |
| set_epoch target | HF dataset | Datum list |
| Shuffle algorithm | HF's random shuffle | Erlang's phash2 |
| Shuffle input | Already-shuffled dataset | Pre-built datums |

**Observed behavior**:
- Python datum 0: "Compose the lyrics..." (original index 2)
- Elixir datum 0: "What would be some good animals..." (original index 1)

Both have the same 4 samples after initial shuffle, but the re-shuffle at epoch 0 produces different orderings because:
1. Different shuffle algorithms (HF shuffle vs phash2)
2. Applied to different data structures (HF dataset rows vs Elixir datum structs)

### CONCLUSION
**ROOT CAUSE CONFIRMED**: Shuffling is applied at different levels with different algorithms.

### FIX OPTIONS

**Option A (Recommended)**: Match Python's lazy evaluation approach
- Elixir should store samples, not datums
- Build datums lazily during `get_batch`
- Apply shuffle to samples before datum building

**Option B**: Match shuffling exactly
- Use the same PRNG (PCG64) and shuffle algorithm
- Ensure shuffling is applied to the same abstraction level

---

## Issue 3: Token Length Differences

### THEORY
Token length differences are a consequence of Issues 1 and 2.

### EVIDENCE

| Metric | Python | Elixir | Difference |
|--------|--------|--------|------------|
| Datum 0 length | 203 tokens | 255 tokens | +52 |
| First sample tokens | 97 | 231 | +134 |

The 52-token difference in datum 0 is explained by:
1. **Different samples** due to batch ordering (Issue 1): Different content = different length
2. **Extra BOS tokens** (Issue 2): ~5 chunks × 1 extra BOS = ~5 tokens
3. **Different message content**: Elixir has "animals to draw" (longer), Python has "country song lyrics" (shorter)

### CONCLUSION
Issue 3 is a **symptom**, not a root cause. Fixing Issues 1 and 2 will resolve Issue 3.

---

## Reproduction Scripts

See companion scripts in this directory:
- `bos_token_test.exs` - Verify BOS token behavior
- `batch_order_test.exs` - Trace sample-to-batch mapping
- `compare_shuffle.py` - Compare shuffle algorithms

---

## Additional Finding: Unicode Encoding Difference

During investigation, I noticed that the dataset content hashes differ between Python and Elixir even though the message text appears identical:

**Python dataset_snapshot.json**:
```
"Substitute \"in\" with an apostrophe for any words ending in \u201cing.\u201d"
```

**Elixir dataset_snapshot.json**:
```
"Substitute \"in\" with an apostrophe for any words ending in "ing."
```

Python uses Unicode fancy quotes (`\u201c` = " and `\u201d` = ") while Elixir shows ASCII quotes. This may be a JSON serialization difference in how the parity artifacts are saved, not an actual difference in the underlying data.

**Recommendation**: Verify that both implementations use the same JSON serialization settings when writing parity artifacts, specifically for Unicode escaping.

---

## Recommended Action Plan

### Priority 1: Fix BOS Token Issue (Issue 2)
**File**: `lib/tinkex_cookbook/recipes/sl_basic.ex`
**Change**: Pass `add_special_tokens` option to tokenizer

### Priority 2: Fix Batch Ordering (Issue 1)
**Files**:
- `lib/tinkex_cookbook/supervised/dataset.ex`
- `lib/tinkex_cookbook/datasets/no_robots.ex`
**Change**: Match Python's lazy datum building approach OR use identical shuffle algorithm

### Verification
After fixes, run:
```bash
./scripts/parity/run_sl_basic_parity.sh
```

Expected result: All parity checks should pass (dataset_snapshot, rendered_samples, first_batch_payload).
