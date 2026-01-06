# Crucible Kitchen Implementation Plan

Date: 2026-01-06

## Objective

Make crucible_kitchen the thick orchestration core with complete workflows, adapter ownership, and run lifecycle support for the five focus recipes.

## K0: Ownership Alignment and Adapter Registry

### Tasks

- Move all noop adapters for LLMClient, EmbeddingClient, VectorStore into crucible_kitchen.
- Remove any residual adapter references in tinkex_cookbook (including LLM and vector store adapters).
- Normalize adapter naming and module paths under `CrucibleKitchen.Adapters.*`.
- Update adapter manifests to reference only kitchen-owned adapters.
- Add adapter contract tests for all ports (TrainingClient, DatasetStore, HubClient, BlobStore, LLMClient, EmbeddingClient, VectorStore).

### Acceptance Criteria

- manifests.ex points only to `CrucibleKitchen.Adapters.*`.
- All port implementations compile without cross-repo adapter references.
- Adapter contract tests pass in isolation.

## K1: Workflow Completion

### Supervised Workflow (confirm)

- Stages: LoadDataset, InitTokenizer, BuildDatums, ForwardBackward, OptimStep, LogMetrics.
- Add ChatDatumBuilder stage to support chat_sl.

### Preference Workflow (DPO)

- Stages: LoadDataset, BuildComparisons, ForwardBackwardCustom, OptimStep, LogMetrics.
- Ensure comparison datum types exist in crucible_train and are used consistently.

### RL Workflow (math_rl)

- Stages: LoadDataset, BuildEnvGroup, Rollout, ComputeAdvantages, ForwardBackward, OptimStep, LogMetrics.
- Provide reward evaluation callback using snakebridge math_verify.

### Distillation Workflow

- Stages: LoadTeacherSampler, SampleTeacher, BuildDistillDatums, ForwardBackward, OptimStep, LogMetrics.
- Require SamplingClient adapter.

### Stage Requirements

- Each stage defines `describe/1` with schema.
- Stage emits telemetry events with run_id, recipe, dataset.
- Stage is deterministic with explicit seed passed via context.

### Acceptance Criteria

- All four workflows exist and can run with mocked ports.
- Stage schemas registered in CrucibleFramework.
- Telemetry events emitted for each stage.

## K2: Adapter and SDK Integration

### Tinkex Adapters

- TrainingClient adapter: start_session, forward_backward, optim_step, save_checkpoint, close_session.
- SamplingClient adapter: sampling with checkpoint_id, streaming, and retry policies.
- Tokenizer adapter: consistent tokenizer construction with `add_special_tokens: false` and explicit BOS/EOS control.

### HF Adapters

- DatasetStore adapter: `load_dataset`, split selection, streaming, shuffle.
- HubClient adapter: download/upload, repo metadata.

### LLM and Embedding Adapters

- LLMClient: claude_agent_sdk, codex_sdk, gemini_ex wrappers with uniform response shape.
- EmbeddingClient: gemini_ex and embed_ex wrappers with consistent vector schema.
- VectorStore: chroma adapter + noop.

### Python Bridge

- snakebridge adapter: manifest loader, session lifecycle, error wrapping.
- Provide manifests for `math_verify`, `sympy`, `pylatexenc`.

### Acceptance Criteria

- All adapters compile and are exercised in smoke tests with mocks.
- LLMClient returns consistent `{:ok, %{content: ..., meta: ...}}` shape across providers.
- snakebridge manifests load in dev environment.

## K3: Evaluation and Telemetry

- Implement Evaluate stage for eval_ex tasks.
- Add harness bridge for batch evaluations.
- Emit metrics to crucible_telemetry MetricsStore.
- Standardize metrics naming and units.

Acceptance Criteria:
- Evaluate stage produces EvalEx.Sample results with scores.
- Metrics stored via MetricsStore and queryable.

## K4: MLOps and Run Lifecycle

- Register checkpoints and artifacts in crucible_model_registry.
- Provide optional deployment stage via crucible_deployment.
- Add feedback ingestion hooks via crucible_feedback.
- Expose run lifecycle events (created, running, completed, failed).

Acceptance Criteria:
- Each run produces a registry entry with lineage.
- Run lifecycle events are visible to control plane.

## Deliverable Checklist

- Full workflow registry with all stages.
- Adapter registry with kitchen-owned implementations.
- Telemetry coverage for every stage.
- Documentation for each workflow and adapter.
