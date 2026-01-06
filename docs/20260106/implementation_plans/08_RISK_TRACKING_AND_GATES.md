# Risk Tracking and Gates

Date: 2026-01-06

## Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Adapter ownership drift | Breaks corrected model | Enforce manifest validation in CI |
| Missing tinkex endpoints | Blocks DPO and model lifecycle | Implement endpoints before parity tests |
| Snakebridge instability | Blocks math_rl | Add deterministic tests + retries |
| Non-deterministic datasets | Breaks parity | Enforce PCG64 shuffling + seed checks |
| Telemetry gaps | Hidden failures | Emit events per stage and assert in tests |
| Over-scope beyond 5 recipes | Delivery delay | Lock scope to focus recipes |

## Gates

### Gate A: Ownership Alignment

- manifests reference only kitchen adapters.
- No adapter modules in tinkex_cookbook.

### Gate B: Workflow Completion

- RL, DPO, distillation workflows exist with stage schemas.
- Stage unit tests pass.

### Gate C: Focus Recipe Enablement

- chat_sl, math_rl, distillation recipes run end-to-end.

### Gate D: Parity

- Token parity for sl_basic, chat_sl.
- Loss and metric parity for DPO and RL.

### Gate E: Evaluation and Telemetry

- Evaluate stage produces metrics in telemetry store.

### Gate F: MLOps and Control Plane

- Registry entries created for runs.
- Gateway endpoints functional.

## Exit Criteria

- All five focus recipes stable with parity tests.
- Kitchen workflows and adapters are documented and tested.
- Run lifecycle, telemetry, and artifacts are wired end-to-end.
