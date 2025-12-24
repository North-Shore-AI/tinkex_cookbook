# Recipe Parity Verification Protocol

This document defines a repeatable, comprehensive procedure to compare recipe
behavior between (run from the `tinkex_cookbook` repo root):

- Python: `./tinker-cookbook`
- Elixir: repo root (`./`)

The goal is functional equivalence for all recipes, with systematic tracing of
differences down to sub-components (tinkex vs tinker, chz_ex vs chz,
hf_datasets_ex vs datasets, snakebridge vs python libs, crucible/eval_ex vs
inspect-ai).

This procedure is written for `sl_basic` first, but is intended to apply to all
recipes.

---

## 1) Parity levels and acceptance criteria

Define the parity target for each recipe before comparing:

1) Functional parity (required)
   - Same steps complete without errors.
   - Same artifacts are produced (checkpoints, metrics files, etc).

2) Behavioral parity (required)
   - Same configuration semantics and CLI behavior.
   - Same dataset splits and example counts.
   - Same renderer behavior (prompt templates, stop tokens).

3) Numerical parity (nice-to-have, not always exact)
   - Loss trends should be in the same ballpark for small runs.
   - Exact match only expected for deterministic paths (fixed seed, no sampling).

Acceptance is achieved when functional + behavioral parity hold, and any
numerical differences are explained by known sources of nondeterminism.

---

## 2) Environment alignment (must-do)

Before running any comparison, align the runtime environment:

- Use the same `TINKER_BASE_URL` and the same `TINKER_API_KEY`.
- Pin model name and version; record it in both configs.
- Use the same dataset revision and split.
- Clear or isolate caches so both runs fetch identical data.
- Set the same seeds for shuffling and sampling.
- Ensure tokenizers match (same model, same tokenizer backend).

Suggested baseline env vars:

```
export TINKER_BASE_URL="https://tinker.thinkingmachines.dev/services/tinker-prod"
export TINKER_API_KEY="..."
export HF_TOKEN="..."  # if dataset is gated
```

If HTTP/2 flow control causes failures in Elixir, force HTTP/1 for training:

```
export TINKEX_HTTP_PROTOCOL=http1
```

Transport choice should not affect training semantics, but must be recorded in
the run metadata.

---

## 3) Standard run modes

Use consistent run modes across recipes to reduce noise:

1) Smoke run
   - Very small sample count (e.g., 4-16).
   - Small batch size (e.g., 2-8).
   - One epoch.

2) Parity run
   - Enough samples to exercise batching and checkpoints.
   - Fixed seed for shuffling.
   - Deterministic options where possible.

3) Full run
   - Default settings for the recipe.
   - Used only after smoke/parity runs match.

Record which run mode you are comparing.

---

## 4) Required artifacts (collect from both runs)

Collect and store the following artifacts for each run in a comparable layout.
Use a per-run directory with a clear name. All paths below are relative to the
`tinkex_cookbook` repo root unless stated otherwise.

Required:

- Config dump
  - Python: `config.json` from `ml_log`
  - Elixir: `config.json` from `MlLog`
- Metrics log
  - Python: `metrics.jsonl`
  - Elixir: `metrics.jsonl`
- Dataset snapshot
  - Sample IDs or hashes of the first N samples.
  - Counts per split.
- Renderer output
  - Rendered prompt text for the first N samples.
  - Token IDs for the first N samples.
- Datum payloads
  - JSON hash of the first batch payload sent to Tinker.
- Checkpoint list
  - Paths of saved checkpoints and final weights.

Optional:

- HTTP request size logs (for debugging).
- Sampling outputs for a fixed prompt.

---

## 5) Layer-by-layer comparison procedure

Run comparisons from top to bottom to isolate differences quickly.

### 5.1 Config and CLI parity (chz vs chz_ex)

Goal: identical config values given the same CLI args.

Steps:

1) Run both with the same CLI args.
2) Compare `config.json` files:
   - Same keys and values.
   - Same defaults resolved.
3) Confirm CLI features:
   - `key=value` parsing.
   - Nested keys (if any).
   - Hyphen allowance matches Python behavior.

If differences appear:
- Check schema defaults and mungers.
- Check validators and enum values.

### 5.2 Dataset loading parity (datasets vs hf_datasets_ex)

Goal: identical data samples in the same order.

Steps:

1) Log dataset split sizes and limits.
2) Record the first N sample IDs and content hashes.
3) Compare:
   - Count of samples.
   - Order after shuffling.

If differences appear:
- Ensure same dataset revision.
- Confirm shuffling seed and implementation.
- Confirm truncation and filtering behavior.

### 5.3 Renderer and tokenizer parity

Goal: identical rendered prompt text and token IDs.

Steps:

1) Capture rendered prompt text for the first N samples.
2) Capture token IDs for the same samples.
3) Compare text + token hashes.

If differences appear:
- Check renderer template (role names, separators, BOS/EOS tokens).
- Check tokenizer backend and model version.

### 5.4 Datum building parity

Goal: identical (or logically equivalent) datums passed to training.

Steps:

1) Log the first batch of datums as JSON.
2) Compare:
   - Model input chunk types and token lengths.
   - Loss weights and targets.
   - Max-length truncation rules.

If differences appear:
- Verify train_on_what selection.
- Verify truncation or masking logic.

### 5.5 Training loop parity (tinker vs tinkex)

Goal: same number of steps and similar metrics.

Steps:

1) Verify number of batches and steps.
2) Compare checkpoint cadence.
3) Compare loss trends with tolerance.
4) Compare final weights save path format.

If differences appear:
- Check batch size calculations.
- Check request chunking and payload splitting.
- Check learning rate schedule implementation.

### 5.6 Evaluation parity (inspect-ai vs crucible/eval_ex)

If the recipe includes evaluation:

1) Run evaluation with a fixed sample set.
2) Compare raw outputs and scoring.
3) Compare aggregate metrics.

If differences appear:
- Check prompt formatting and stop sequences.
- Check scoring functions and normalization.

### 5.7 Python interop parity (snakebridge)

If a recipe uses Python-only tooling:

1) Capture inputs and outputs to the Python bridge.
2) Compare to Python cookbook outputs.
3) Ensure manifest selection and versions match.

---

## 6) Comparison workflow (repeatable)

This workflow is meant to be repeated per recipe.

1) Run the Python recipe in smoke mode.
2) Run the Elixir recipe in smoke mode.
3) Compare artifacts layer-by-layer (Section 5).
4) Fix divergences at the lowest layer first.
5) Repeat smoke mode until parity holds.
6) Repeat for parity mode.
7) Only then attempt full mode.

Use a simple checklist per recipe:

- [ ] Config parity
- [ ] Dataset parity
- [ ] Renderer/tokenizer parity
- [ ] Datum parity
- [ ] Training loop parity
- [ ] Evaluation parity (if present)
- [ ] Final artifact parity

---

## 7) Example: sl_basic parity run

Recommended smoke settings (run from repo root):

Python:

```
PYTHONPATH=./tinker-cookbook python -m tinker_cookbook.recipes.sl_basic \
  log_path=/tmp/tinker-sl-basic-smoke \
  batch_size=2 \
  num_epochs=1 \
  max_length=256
```

Elixir:

```
TINKEX_HTTP_PROTOCOL=http1 mix run -e "TinkexCookbook.Recipes.SlBasic.main()" -- \
  log_path=/tmp/tinkex-sl-basic-smoke \
  batch_size=2 \
  num_epochs=1 \
  max_length=256 \
  n_train_samples=4
```

Artifacts to compare:

- `config.json`
- `metrics.jsonl`
- First 4 samples (IDs and rendered prompts)
- First batch datum payload hash

---

## 8) Troubleshooting map

If a mismatch is detected, use this table to locate the likely layer:

- Config mismatch -> chz/chz_ex schema defaults or mungers.
- Dataset mismatch -> datasets/hf_datasets_ex load/split/shuffle behavior.
- Token mismatch -> renderer template or tokenizer backend/version.
- Loss weights mismatch -> train_on_what or truncation logic.
- Step count mismatch -> batch size or dataset length calculation.
- Checkpoint mismatch -> save cadence or step indexing.
- Eval mismatch -> generate adapter or scorer logic.

---

## 9) Run metadata to record

Every parity run should record:

- Git commit SHA for both repos.
- Dataset revision and split.
- Model name and tokenizer version.
- Environment variables used.
- Transport mode (HTTP/1 vs HTTP/2).
- Run mode (smoke/parity/full).

---

## 10) Extending to all recipes

This protocol scales across recipes by:

- Keeping the same layer order.
- Reusing the artifact checklist.
- Adding recipe-specific invariants (e.g., RL reward stats, eval metrics).

When a new recipe is ported, create a small "parity harness" for that recipe
to emit the artifacts listed here.

---

## 11) Automated Parity Harness

An automated parity harness is available for sl_basic (and can be extended to
other recipes). The harness collects comparable artifacts from both
implementations and produces a diff-friendly report.

### 11.1 Enabling Parity Mode

Both Python and Elixir implementations support opt-in instrumentation via an
environment variable:

```bash
export PARITY_MODE=1
```

When enabled, additional artifacts are written to a `parity/` subdirectory
within the log path:

- `dataset_snapshot.json` - First N sample IDs and content hashes
- `rendered_samples.json` - Rendered prompts and token IDs for first N samples
- `first_batch_payload.json` - Hash and structure of the first training batch

These artifacts are not written unless PARITY_MODE=1, so normal training
semantics are unchanged.

### 11.2 Run Scripts

Convenience scripts are provided in `scripts/parity/`:

**Run Python sl_basic with parity artifacts:**

```bash
./scripts/parity/run_sl_basic_python.sh
```

Output: `/tmp/parity/sl_basic/python/`

**Run Elixir sl_basic with parity artifacts:**

```bash
./scripts/parity/run_sl_basic_elixir.sh
```

Output: `/tmp/parity/sl_basic/elixir/`

**Run both and compare (full pipeline):**

```bash
./scripts/parity/run_sl_basic_parity.sh
```

Output:
- Python artifacts: `/tmp/parity/sl_basic/python/`
- Elixir artifacts: `/tmp/parity/sl_basic/elixir/`
- Comparison report: `/tmp/parity/sl_basic/report.json`

### 11.3 Comparison Script

The comparison script (`scripts/parity/compare_artifacts.py`) reads artifacts
from both directories and produces:

1. A human-readable summary printed to stdout
2. A JSON report with detailed comparison results

Usage:

```bash
python scripts/parity/compare_artifacts.py [python_dir] [elixir_dir]

# With custom paths
python scripts/parity/compare_artifacts.py \
    /tmp/my-python-run \
    /tmp/my-elixir-run \
    --output /tmp/my-report.json

# JSON output only
python scripts/parity/compare_artifacts.py --json
```

The script compares:
- **Config parity**: Key configuration values
- **Metrics parity**: Step counts and line counts
- **Dataset snapshot parity**: Sample content hashes
- **Rendered samples parity**: Token IDs and weights
- **First batch payload parity**: Overall payload hash

### 11.4 Smoke Run Parameters

The run scripts use minimal smoke settings by default:

| Parameter | Value |
|-----------|-------|
| batch_size | 2 |
| num_epochs | 1 |
| max_length | 256 |
| n_train_samples | 4 (Elixir only) |

To use different parameters, modify the run scripts or call the recipes
directly with custom CLI args.

### 11.5 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| TINKER_API_KEY | Required. Your Tinker API key. | - |
| TINKER_BASE_URL | Tinker API endpoint. | Production URL |
| PARITY_MODE | Enable parity artifact collection. | 0 (disabled) |
| TINKEX_HTTP_PROTOCOL | HTTP protocol for Elixir. | http1 |
| PARITY_OUTPUT_DIR | Override artifact output directory. | /tmp/parity/sl_basic/{lang}/ |
| SKIP_PYTHON | Skip Python run in combined script. | 0 |
| SKIP_ELIXIR | Skip Elixir run in combined script. | 0 |

### 11.6 Extending to Other Recipes

To add parity support for a new recipe:

1. **Python side:**
   - Import the parity helpers:
     ```python
     from tinker_cookbook.utils.parity import (
         init_parity_logger,
         log_dataset_snapshot,
         log_rendered_sample,
         log_first_batch_payload,
         flush_parity_logger,
     )
     ```
   - Call `init_parity_logger(log_path)` at training start
   - Call `log_dataset_snapshot(samples)` after loading data
   - Call `log_rendered_sample(...)` in datum creation
   - Call `log_first_batch_payload(batch)` before first training step
   - Call `flush_parity_logger()` at training end

2. **Elixir side:**
   - Import the Parity module:
     ```elixir
     alias TinkexCookbook.Utils.Parity
     ```
   - Call `Parity.init_logger(log_path)` at training start
   - Call `Parity.log_dataset_snapshot(samples)` after loading data
   - Call `Parity.log_rendered_sample(...)` in datum creation
   - Call `Parity.log_first_batch_payload(batch)` before first step
   - Call `Parity.flush_logger()` and `Parity.stop_logger()` at training end

3. **Create run scripts:**
   - Copy `run_sl_basic_python.sh` and `run_sl_basic_elixir.sh`
   - Update recipe name and CLI args for the new recipe
   - Optionally create a combined `run_{recipe}_parity.sh`

4. **Extend comparison script (if needed):**
   - The current `compare_artifacts.py` should work for any recipe
   - Add recipe-specific checks if the recipe has unique artifacts

---

End of document.
