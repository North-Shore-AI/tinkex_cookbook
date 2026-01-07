# Distillation Workflow: Teacher Sampling + Distill Datums + Metrics

Date: 2026-01-06

## Scope

Implement the missing distillation stages in `crucible_kitchen` using the new
`CrucibleTrain.Ports.SamplingClient` port. The workflow must support teacher
sampling, distillation datums, forward/backward, and metric logging.

Target files:

- `crucible_kitchen/lib/crucible_kitchen/workflows/distillation.ex`
- New stages under `crucible_kitchen/lib/crucible_kitchen/stages/`
  - `init_teacher.ex`
  - `get_distillation_batch.ex`
  - `teacher_inference.ex`
  - `build_distill_datums.ex`
  - `distillation_forward_backward.ex`
  - `log_distillation_metrics.ex`
  - `cleanup_teacher.ex`

Python parity references:

- `tinker-cookbook/tinker_cookbook/distillation/train_on_policy.py`
- `tinker-cookbook/tinker_cookbook/distillation/datasets.py`

## Target Stage Responsibilities

### InitTeacher

- Starts a teacher sampling session with `SamplingClient.start_session/2`.
- Uses `teacher_model` or `teacher_checkpoint_path` config.
- Stores `:teacher_session` in state.

### BuildDistillationDataset

- Uses `DatasetStore.to_list` to load samples.
- Builds a prompt-only dataset (strings or messages) with batch size + max length.
- Stores:
  - `:distillation_dataset`
  - `:num_distillation_batches`

### GetDistillationBatch

- Returns the next batch of prompts/messages.
- Stores `:distillation_batch` and `:batch_index`.

### TeacherInference

- Uses `SamplingClient.sample/5` (or `sample_stream/5`) to generate teacher outputs.
- Uses renderer to build prompts (`ModelInput`) and stop sequences.
- Stores:
  - `:teacher_responses`
  - `:teacher_logprobs` (optional for KL distillation)

### BuildDistillDatums

- Converts `distillation_batch` + `teacher_responses` into `CrucibleTrain.Types.Datum`.
- Uses `CrucibleTrain.Renderers.Renderer.build_supervised_example` to create
  `ModelInput` and weights.
- Stores `:distill_datums`.

### DistillationForwardBackward

- Calls `TrainingClient.forward_backward/4` with `loss_fn: :cross_entropy`
  (or a custom loss if using teacher logprobs).
- Stores `:distillation_future`.

### LogDistillationMetrics

- Consumes `fb_result.metrics` and teacher sampling metrics.
- Emits:
  - `distill_loss`
  - `num_prompts`
  - `num_tokens`
  - `teacher_latency_ms`

### CleanupTeacher

- Closes `SamplingClient` session.

## Implementation Plan (TDD)

1) **Stage tests** (mock SamplingClient + TrainingClient):
   - Ensure teacher session lifecycle
   - Ensure prompt rendering uses correct renderer and tokenizer
   - Ensure datums are built deterministically
   - Ensure forward/backward is invoked with correct loss

2) **Stage implementation** (use Kitchen stage helpers):
   - Keep state keys stable for workflow wiring
   - Use `Context.get_train_ports/1` for `SamplingClient` port
   - Add `describe/1` with required/optional keys

3) **Telemetry**
   - Emit stage events:
     - `[:crucible_kitchen, :distillation, :teacher_inference]`
     - `[:crucible_kitchen, :distillation, :forward_backward]`

## Configuration Keys

Recommended config keys for the recipe and workflow:

- `teacher_model` (required)
- `teacher_checkpoint_path` (optional)
- `max_tokens` (teacher sampling)
- `temperature`
- `top_k` / `top_p`
- `batch_size`
- `max_length`
- `distill_alpha` (if mixing hard+soft loss)
- `distill_temperature` (if using KL)

## Determinism

- Seed all sampling when possible; otherwise mark teacher sampling as nondeterministic
- Use deterministic dataset ordering (PCG64)
- Use `add_special_tokens: false` in tokenizer

## Done When

- Distillation workflow executes end-to-end with mocks
- Stage tests pass and telemetry is emitted
- Teacher session is created and cleaned up reliably
