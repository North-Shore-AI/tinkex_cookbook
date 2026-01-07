# Dependency Audit and Execution Plan

Date: 2026-01-06

## Purpose

Confirm which core dependencies are real implementations versus placeholders, and
plan the work needed to make crucible_kitchen and tinkex_cookbook fully integrated
with no noop-only paths in production.

This plan also includes the immediate workstream tasks:

1) DPO refactor (SamplingClient logprobs + forward_backward_custom)
2) Distillation workflow stages with SamplingClient + telemetry
3) chat_sl, math_rl, distillation recipes + parity/property tests

## Audit Summary

The dependencies below are the required core stack for kitchen and cookbook. The
question is not only whether the repos exist, but whether kitchen actually uses
real integrations (not just noops or placeholders).

Status legend:

- Implemented: repo exists locally with lib/test and real code
- Integrated: kitchen/cookbook uses real adapters or types from the repo
- Placeholder risk: dependency listed but unused or only noop adapters exist

### Repository Presence (in ~/p/g/North-Shore-AI)

Present with code:

- crucible_train
- crucible_framework
- crucible_ir
- crucible_telemetry
- crucible_harness
- crucible_datasets
- eval_ex
- tinkex
- hf_datasets_ex
- hf_hub_ex
- tiktoken_ex

Missing locally:

- snakebridge (used as hex dependency in crucible_kitchen; repo not in workspace)

### Integration Status (kitchen/cookbook)

| Dependency | Repo present | Integrated now | Placeholder risk | Notes |
| --- | --- | --- | --- | --- |
| crucible_train | yes | yes | low | Core types/ports/renderers in kitchen stages. |
| crucible_framework | yes | partial | medium | Used in cookbook runtime, not in kitchen orchestration. |
| crucible_ir | yes | partial | medium | Used in cookbook recipe specs, not in kitchen workflows. |
| crucible_telemetry | yes | no | high | MetricsStore adapter is noop only; telemetry not wired to store. |
| crucible_harness | yes | no | high | No kitchen usage yet. |
| crucible_datasets | yes | no | high | No kitchen usage yet. |
| eval_ex | yes | placeholder | high | Eval adapter computes metrics locally, no EvalEx tasks. |
| tinkex | yes | yes | low | TrainingClient + SamplingClient adapters call real SDK. |
| hf_datasets_ex | yes | yes | low | DatasetStore adapter wraps hf_datasets_ex. |
| hf_hub_ex | yes | yes | low | HubClient adapter wraps hf_hub_ex. |
| snakebridge | no (hex only) | partial | high | Adapter exists, but local repo missing and math_rl not wired. |
| tiktoken_ex | yes | no | high | Dependency included but no tokenizer adapter usage. |

## Decisions From Audit

1) Keep cookbook thin
- Removed cookbook adapters and SDK dependencies are aligned with the ownership model.
- All adapters should live in crucible_kitchen.

2) Noop-only paths are allowed only in tests
- Production manifests must not point at noop adapters.
- Add manifest validation to enforce this.

3) Missing integrations must be implemented before parity gates
- eval_ex, crucible_harness, crucible_telemetry, crucible_datasets, tiktoken_ex,
  snakebridge require explicit integration work.

## Execution Plan

### Phase A: No-Placeholder Integration Work

1) crucible_telemetry
- Add a real MetricsStore adapter (JSONL or in-memory) and wire kitchen telemetry
  to emit metrics into MetricsStore.
- Add tests for MetricsStore writes on stage completion.

2) eval_ex + crucible_harness
- Replace EvalClient placeholder with EvalEx Task/Scorer integration.
- Add an Evaluate stage that uses CrucibleHarness for batch eval runs.
- Add adapter tests and stage tests (no network).

3) crucible_datasets
- Use CrucibleDatasets.MemoryDataset for evaluation and parity tests.
- Add dataset determinism tests (PCG64 seed).

4) tiktoken_ex
- Add a tokenizer adapter and wire InitTokenizer to use it when configured.
- Add tokenization determinism tests (add_special_tokens: false).

5) snakebridge
- Decide repo strategy: local repo or hex-only.
- Add config and tests for math_verify manifest using Snakebridge runtime stubs.
- Wire math_rl to use math_verify for rewards.

6) crucible_framework + crucible_ir
- Decide on a single orchestration path:
  - Option A: keep kitchen native runner, emit CrucibleIR specs for registry/telemetry
  - Option B: kitchen builds CrucibleIR and executes via CrucibleFramework
- Add one integration test to confirm the selected path.

### Phase B: Required Workstreams (TDD)

1) DPO refactor
- Write failing tests for ComputeReferenceLogprobs and DPOForwardBackward.
- Use SamplingClient.compute_logprobs for reference logprobs.
- Use TrainingClient.forward_backward_custom with real DPO loss.
- Add parity test vs Python DPO metrics.

2) Distillation stages
- Implement InitTeacher, TeacherInference, BuildDistillDatums, DistillationForwardBackward,
  LogDistillationMetrics, CleanupTeacher.
- Use SamplingClient for teacher sampling.
- Add telemetry tests and stage contract tests.

3) Missing recipes + parity/property tests
- Implement chat_sl, math_rl, distillation recipes in tinkex_cookbook.
- Add parity tests for all five recipes.
- Add property tests for renderer invariants and dataset determinism.

### Phase C: Enforcement and Gates

- Add manifest validation that disallows noop adapters in prod manifests.
- Add CI check that asserts no placeholder adapters are used by default.
- Track gating criteria:
  - Gate A: ownership alignment
  - Gate B: workflow completion
  - Gate C: focus recipes runnable
  - Gate D: parity tests pass
  - Gate E: eval + telemetry integrated
  - Gate F: registry and run lifecycle visible

## Deliverables

- Dependency audit matrix updated with evidence.
- Noop-free production manifests.
- Integrated eval + telemetry pipelines.
- DPO, distillation, chat_sl, math_rl implemented and tested.

## Verification Commands

- `mix test` in crucible_kitchen, crucible_train, tinkex_cookbook
- `mix credo --strict` and `mix dialyzer` where configured
- Parity tests under `tinkex_cookbook/test/parity/`
