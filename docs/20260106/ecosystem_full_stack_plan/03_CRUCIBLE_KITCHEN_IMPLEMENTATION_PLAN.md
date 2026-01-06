# Crucible Kitchen Implementation Plan

Date: 2026-01-06

## Guiding Constraints

- Follow corrected ownership model: crucible_kitchen owns orchestration + adapters only.
- Training ports, types, renderers live in crucible_train.
- External SDKs remain independent (tinkex, hf_*_ex, claude_agent_sdk, codex_sdk, gemini_ex).

## Phase K0: Ownership Alignment

- Remove any port duplication in crucible_kitchen that overlaps crucible_train.
- Keep a single kitchen-owned port only when workflow-specific (e.g., Completer).
- Move or re-home adapters to crucible_kitchen (Tinkex, HF, LLMs, snakebridge).
- Ensure all adapters implement crucible_train ports.

Deliverables:
- Adapter modules in crucible_kitchen for: TrainingClient, DatasetStore, HubClient, BlobStore, LLMClient, EmbeddingClient, VectorStore, PythonBridge.
- Manifest-driven adapter resolution (dev/test/prod).

## Phase K1: Workflow Completion

Implement full workflows with stage definitions and defaults:

- Supervised (existing, confirm parity)
  - LoadDataset -> InitTokenizer -> BuildDatums -> ForwardBackward -> OptimStep -> LogMetrics

- Preference (DPO)
  - LoadDataset -> BuildComparisons -> ForwardBackwardCustom -> OptimStep -> LogMetrics

- RL (math_rl)
  - LoadDataset -> BuildEnvGroup -> Rollout -> ComputeAdvantages -> ForwardBackward -> OptimStep -> LogMetrics

- Distillation
  - LoadTeacherSampler -> SampleTeacher -> BuildDistillDatums -> ForwardBackward -> OptimStep -> LogMetrics

Deliverables:
- Stage modules for each step with schema and telemetry.
- Workflow registry in crucible_kitchen.

## Phase K2: Adapter and SDK Integration

- Tinkex adapter: full TrainingClient, SamplingClient, Tokenizer access.
- HF adapters: HfDatasetsEx and HfHubEx integration, including gated datasets.
- LLM adapters: claude_agent_sdk, codex_sdk, gemini_ex via LLMClient port.
- Python bridge: snakebridge manifests for math_verify, sympy, pylatexenc.

Deliverables:
- Adapter suites with test doubles.
- Standardized error and retry policies.

## Phase K3: Evaluation and Telemetry

- Hook EvalEx tasks to crucible_kitchen Evaluate stage.
- Use CrucibleHarness for batch experiments.
- Emit telemetry events for every stage, with run ids and dataset provenance.
- Store metrics via crucible_telemetry MetricsStore port.

Deliverables:
- Evaluate stage + metrics rollup.
- Default telemetry pipeline for kitchen runs.

## Phase K4: MLOps and Control Plane

- Integrate model registry for checkpoints and artifacts.
- Add deployment stages (optional per workflow).
- Wire feedback ingestion for continual learning.
- Provide run lifecycle APIs for nsai_gateway.

Deliverables:
- Run lifecycle events: created, running, completed, failed.
- Artifacts and lineage recorded in crucible_model_registry.

## Acceptance Criteria

- All five focus recipes execute end-to-end through crucible_kitchen.
- Adapter wiring is purely in crucible_kitchen.
- Deterministic runs with parity tests for the selected recipes.
- Telemetry and artifact lineage are present for each run.
