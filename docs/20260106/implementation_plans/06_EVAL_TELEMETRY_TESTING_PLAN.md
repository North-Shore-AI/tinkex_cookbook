# Evaluation, Telemetry, and Testing Plan

Date: 2026-01-06

## Evaluation Integration

### Tasks

- Implement Evaluate stage in crucible_kitchen using eval_ex tasks.
- Provide dataset adapters from crucible_datasets and MemoryDataset.
- Add scorers for exact_match, contains, and DPO-specific metrics as needed.
- Integrate crucible_harness for batch evaluation runs.

### Acceptance

- Evaluate stage produces EvalEx.Sample results with scores.
- Batch evaluation can be run from CLI for focus recipes.

## Telemetry Integration

- Emit stage-level telemetry events for all workflows.
- Standardize metric names and units across recipes.
- Send metrics to crucible_telemetry MetricsStore.
- Add trace spans for key run steps.

## Testing Strategy

### TDD Requirement

- All new plumbing changes are TDD: failing tests first, minimal implementation next.

### Unit Tests

- Stage unit tests with mocked ports.
- Adapter contract tests for each port.
- Dataset builder unit tests with deterministic seeds.

### Parity Tests

- Token parity for sl_basic and chat_sl.
- DPO loss parity against Python reference.
- RL metrics parity (KL, entropy, advantage stats).

### Integration Tests

- End-to-end recipe runs using mocked clients.
- Optional smoke tests against real Tinkex server when available.

### Property Tests

- Renderer invariants (e.g., BOS/EOS handling, token count bounds).
- Dataset shuffling determinism with fixed seeds.

## Acceptance Criteria

- Tests cover each new stage and adapter.
- Parity tests pass for all five focus recipes.
- Telemetry events are emitted and stored for each run.
