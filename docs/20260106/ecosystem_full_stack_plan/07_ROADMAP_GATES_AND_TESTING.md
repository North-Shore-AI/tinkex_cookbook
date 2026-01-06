# Roadmap, Gates, and Testing

Date: 2026-01-06

## Phased Roadmap

### Phase 0: Ownership Alignment
- Complete adapter migration into crucible_kitchen.
- Remove port duplication and update manifests.

Gate:
- Static checks confirm no adapters or ports are duplicated across repos.

### Phase 1: Kitchen Workflow Completion
- Implement RL, DPO, and distillation workflows.
- Add missing stages with schemas and telemetry.

Gate:
- Unit tests for each stage.
- Workflow dry-run passes with mocked ports.

### Phase 2: Focus Recipe Enablement
- Wire 5 recipes through crucible_kitchen.
- Confirm dataset field mappings and tokenizer behavior.

Gate:
- Parity tests for the 5 recipes against Python tinker-cookbook.
- Deterministic dataset shuffling and tokenizer parity tests.

### Phase 3: Evaluation and MLOps Integration
- Hook eval_ex and crucible_harness into kitchen Evaluate stage.
- Register artifacts in crucible_model_registry.

Gate:
- Evaluation metrics recorded in telemetry.
- Artifact lineage recorded for each run.

### Phase 4: Control Plane Integration
- Expose run lifecycle APIs via nsai_gateway.
- Integrate command approvals and synapse orchestration signals.

Gate:
- End-to-end run triggered from gateway.
- Run lifecycle events visible to command and synapse.

## Testing Strategy

### Unit Tests
- Stage logic validation with mocked ports.
- Adapter contract tests for each port.

### Parity Tests
- Token parity for supervised and chat recipes.
- Metrics parity for NLL, KL, and DPO loss.

### Integration Tests
- Run end-to-end on local tinkex test server or mocked clients.
- Validate telemetry, checkpoints, and artifacts are produced.

### Determinism
- Fixed seed tests for dataset shuffling and batching.
- Snapshot tests for rendered message tokens.

## Risk Mitigation

- Keep recipes limited to 5 to avoid scope creep.
- Enforce adapter ownership through lint checks and CI rules.
- Use strong mocks for networkless tests.
- Document all run inputs and outputs for reproducibility.
