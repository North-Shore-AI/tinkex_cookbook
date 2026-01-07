# DPO Refactor: SamplingClient Logprobs + forward_backward_custom

Date: 2026-01-06

## Scope

Refactor the DPO stages in `crucible_kitchen` so they:

- Compute reference logprobs via `CrucibleTrain.Ports.SamplingClient.compute_logprobs/4`
- Use `CrucibleTrain.Ports.TrainingClient.forward_backward_custom/5` with a real DPO loss
- Emit deterministic metrics matching Python parity

Target files:

- `crucible_kitchen/lib/crucible_kitchen/stages/compute_reference_logprobs.ex`
- `crucible_kitchen/lib/crucible_kitchen/stages/dpo_forward_backward.ex`
- Supporting tests in `crucible_kitchen/test/crucible_kitchen/stages/*`

Python parity reference:

- `tinker-cookbook/tinker_cookbook/preference/train_dpo.py`

## Current State

- `ComputeReferenceLogprobs` calls `training_client.compute_logprobs` (non-port API).
- `DPOForwardBackward` calls `training_client.dpo_forward_backward` (placeholder) and does not compute real DPO loss.

## Target Behavior

1) Reference logprobs are computed using the **sampling client** (not training client).
2) DPO loss is computed inside `forward_backward_custom` using the same algorithm as Python:

```
loss = -log(sigmoid(beta * (log_pi_chosen - log_pi_rejected - log_ref_chosen + log_ref_rejected)))
```

3) Metrics are consistent with the Python reference:

- `dpo_loss`
- `accuracy`
- `margin`
- `chosen_reward`
- `rejected_reward`
- `num_pairs`

## Implementation Plan (TDD)

### 1) Tests First

Add stage tests that fail under current behavior:

- `ComputeReferenceLogprobs`:
  - Uses `sampling_client` port
  - Builds **full sequence** `ModelInput` (prompt + completion + last target token)
  - Calls `SamplingClient.compute_logprobs` per sequence
  - Stores `ref_chosen_logprobs` and `ref_rejected_logprobs` in state

- `DPOForwardBackward`:
  - Uses `TrainingClient.forward_backward_custom`
  - Loss function consumes logprobs and returns metrics
  - `:dpo_future` stored and `AwaitFuture` yields metrics

Recommended test location:

- `crucible_kitchen/test/crucible_kitchen/stages/dpo_stages_test.exs`

Mocks:

- Use `Mox` for `CrucibleTrain.Ports.SamplingClient` and `TrainingClient`
- Ensure no network in tests

### 2) ComputeReferenceLogprobs Stage

**Key change:** Replace `training_client` usage with `sampling_client` port.

Algorithm (match Python):

- Build `full_sequence_inputs` from each datum or pair:
  - Use the `datum.model_input`
  - Append the last target token (if present)
- Call `SamplingClient.compute_logprobs` for each sequence
- Await results and strip the first logprob (prompt prefill)

State updates:

- `:ref_chosen_logprobs` (list of logprob tensors)
- `:ref_rejected_logprobs`

Notes:

- If `ref_session` is not already present, create it by calling `SamplingClient.start_session` using `reference_model` (or base model).
- Prefer a short `sampling_params` for compute_logprobs (max_tokens=1, include_prompt_logprobs=true) if needed by adapter.

### 3) DPOForwardBackward Stage

**Key change:** Use `TrainingClient.forward_backward_custom` and a real DPO loss.

Loss function shape (Elixir):

- Inputs: `datums`, `logprobs_list`
- Output: `{loss_tensor, metrics_map}`

Steps:

- Split `logprobs_list` into chosen/rejected (even/odd indices)
- Build `chosen_logprob` and `rejected_logprob` as weighted dot products
- Do the same for reference logprobs
- Compute DPO loss and metrics

Match the Python implementation in:

- `tinker-cookbook/tinker_cookbook/preference/train_dpo.py`

**Important:** Use the same weighting behavior as Python
(`loss_fn_inputs["weights"]` aligned to token logprobs).

### 4) Telemetry + Schema

Add/verify `describe/1` for both stages:

- Required state keys
- Config keys (`dpo_beta`, `reference_model`)

Emit telemetry events:

- `[:crucible_kitchen, :dpo, :ref_logprobs]`
- `[:crucible_kitchen, :dpo, :forward_backward]`

### 5) Parity Test

Add a parity fixture test:

- Build a tiny DPO batch (2 pairs) with deterministic tokens
- Precompute Python DPO loss/metrics
- Assert Elixir loss and metrics within tolerance

## Supporting Changes (if needed)

- Ensure `sampling_client` is wired in `tinkex_cookbook` manifests and `crucible_kitchen` adapters.
- If the preference dataset does not currently surface datums, build datums within `DPOForwardBackward` using the renderer and stored tokenizer.

## Done When

- Stage tests pass with `SamplingClient` mocks
- DPO loss/metrics match Python reference
- No use of training-client-only logprob helpers
