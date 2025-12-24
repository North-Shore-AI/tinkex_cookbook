# Agent Investigation Prompt: Batch Ordering and BOS Token Parity Issues

## Context

You are investigating parity issues between two implementations of the `sl_basic` supervised learning recipe:

- **Python**: `tinker-cookbook/tinker_cookbook/recipes/sl_basic.py` (and `sl_basic_parity.py`)
- **Elixir**: `tinkex_cookbook/lib/tinkex_cookbook/recipes/sl_basic.ex`

Both implementations train a Llama-3.1-8B model on the HuggingFace `no_robots` dataset using supervised learning.

### What Has Been Verified Working

1. **Dataset shuffle parity** - CONFIRMED WORKING
   - Both implementations use PCG64 PRNG with `seed=0`
   - Elixir's `HfDatasetsEx.Dataset.shuffle(dataset, seed: 0)` produces identical ordering to Python's `dataset.shuffle(seed=0)`
   - All 4 sample content hashes match exactly between implementations

2. **Renderer selection** - CONFIRMED ALIGNED
   - Both now use `llama3` renderer (not `role_colon`)

### Outstanding Issues to Investigate

**Issue 1: Batch Ordering Mismatch**
- Python datum 0 contains "Compose the lyrics..." (shuffled sample index 2)
- Elixir datum 0 contains "What would be some good animals..." (shuffled sample index 1)
- The samples are identical, but they appear in different datums within the batch

**Issue 2: BOS Token Count Mismatch**
- Python first datum starts with: `[128000, 128006, 882, 128007, ...]` (1 BOS token)
- Elixir first datum starts with: `[128000, 128000, 128000, 128006, 882, ...]` (3 BOS tokens)
- Token 128000 is `<|begin_of_text|>` (BOS) for Llama 3

**Issue 3: Total Token Length Differences**
- Python datum 0: 203 tokens
- Elixir datum 0: 255 tokens
- This ~50 token difference may be explained by BOS tokens and/or different content

## Your Mission

Investigate these issues systematically. **DO NOT proceed under assumptions.** For each issue:

1. **Formulate a theory** explaining what might cause the discrepancy
2. **Design an experiment** to test the theory
3. **Create reproducible scripts** that can be run independently
4. **Gather evidence** by running the experiments
5. **Report findings** with concrete data, not speculation

## Investigation Guidelines

### Methodology

For each issue, follow this structure:

```
THEORY: [Clear statement of what you believe causes the issue]
PREDICTION: [What specific behavior would confirm/refute this theory]
EXPERIMENT: [Concrete steps to test]
EVIDENCE: [Actual output from running the experiment]
CONCLUSION: [Whether theory was confirmed/refuted, and next steps]
```

### Do Not

- Make changes to fix issues before understanding root cause
- Assume behavior without testing
- Skip creating reproducible test scripts
- Conflate multiple issues in one experiment
- Draw conclusions without evidence

### Do

- Create minimal, isolated test cases
- Save test scripts to `scripts/parity/investigations/`
- Compare equivalent code paths side-by-side
- Document exact file paths and line numbers for relevant code
- Test one variable at a time

## Starting Points for Investigation

### Issue 1: Batch Ordering

Relevant code locations to examine:

**Python:**
- `tinker-cookbook/tinker_cookbook/supervised/data.py` - `SupervisedDatasetFromHFDataset` class
- `tinker-cookbook/tinker_cookbook/recipes/chat_sl/chat_datasets.py` - `NoRobotsBuilder`
- How does the HF dataset iterator work? Does it preserve order?

**Elixir:**
- `lib/tinkex_cookbook/datasets/no_robots.ex` - dataset loading
- `lib/tinkex_cookbook/supervised/common.ex` - batch creation
- How are samples grouped into batches?

**Theories to consider:**
1. Different iteration order over the dataset
2. Different batch creation logic (e.g., chunking vs streaming)
3. Parallel processing causing non-deterministic order
4. Different handling of dataset indexing

**Experiment template:**
```bash
# Create script: scripts/parity/investigations/batch_order_test.sh

# Step 1: Print the exact sample indices in each batch for Python
# Step 2: Print the exact sample indices in each batch for Elixir
# Step 3: Compare the mapping of sample -> batch -> position
```

### Issue 2: BOS Token Count

Relevant code locations:

**Python:**
- `tinker-cookbook/tinker_cookbook/renderers.py` - `Llama3Renderer` class
- Look for `_bos_tokens` property and where BOS is added

**Elixir:**
- `lib/tinkex_cookbook/renderers/llama3.ex` - `bos_tokens/1` function
- Look for where BOS tokens are prepended to the model input

**Theories to consider:**
1. Different `bos_tokens` implementation (one returns 1 token, other returns 3)
2. BOS added at different stages (renderer vs datum construction)
3. Multiple BOS additions (e.g., renderer adds one, datum builder adds another)
4. Different handling of the first message vs subsequent messages

**Experiment template:**
```elixir
# Test in Elixir: What does Llama3.bos_tokens(state) return?
# Compare to Python: What does renderer._bos_tokens return?
```

### Issue 3: Token Length Difference

This may be a consequence of Issues 1 and 2, but verify independently:

**Theories to consider:**
1. BOS token difference accounts for it (3 - 1 = 2 extra tokens, but diff is ~50)
2. Different max_length truncation behavior
3. Different samples in the datums (due to batch ordering)
4. Different tokenization of the same content

## Deliverables

Create the following artifacts in `scripts/parity/investigations/`:

1. `batch_order_investigation.sh` - Script to trace sample-to-batch mapping
2. `bos_token_investigation.sh` - Script to compare BOS token generation
3. `token_length_investigation.sh` - Script to compare tokenization of identical content
4. `INVESTIGATION_REPORT.md` - Your findings with evidence

## Parity Artifacts Available

Recent parity run artifacts are in `/tmp/parity/sl_basic/`:
- `python/parity/dataset_snapshot.json` - Python's first 10 samples with hashes
- `python/parity/first_batch_payload.json` - Python's first batch datums
- `python/parity/rendered_samples.json` - Python's rendered tokens
- `elixir/parity/dataset_snapshot.json` - Elixir's first 10 samples with hashes
- `elixir/parity/first_batch_payload.json` - Elixir's first batch datums
- `elixir/parity/rendered_samples.json` - Elixir's rendered tokens

You can re-run the parity test with:
```bash
cd /home/home/p/g/North-Shore-AI/tinkex_cookbook
./scripts/parity/run_sl_basic_parity.sh
```

## Environment

- Repository root: `/home/home/p/g/North-Shore-AI/tinkex_cookbook`
- Python cookbook: `./tinker-cookbook/`
- Python venv: `./tinker-cookbook/.venv/`
- Elixir code: `./lib/tinkex_cookbook/`
- Model: `meta-llama/Llama-3.1-8B`
- Dataset: `HuggingFaceH4/no_robots`
- Renderer: `llama3` (both implementations)
- Batch size: 2
- Samples: 4 (limited for parity testing)

## Success Criteria

Your investigation is complete when you can answer:

1. **Why** do samples appear in different batch positions?
2. **Where** in the code does the divergence occur?
3. **Why** does Elixir produce 3 BOS tokens vs Python's 1?
4. **Where** in the code are the extra BOS tokens added?
5. **What** specific code changes would achieve parity?

Provide evidence for each answer, not speculation.
