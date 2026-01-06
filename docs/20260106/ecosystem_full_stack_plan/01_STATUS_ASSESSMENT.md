# Status Assessment

Date: 2026-01-06

## Ecosystem Status (Snapshot)

### Orchestration and Core

| Repo | Status | Notes |
| --- | --- | --- |
| crucible_kitchen | MVP complete | Supervised workflow works; RL/DPO/distill workflows are placeholders. Uses corrected ownership model docs. |
| crucible_framework | Stable | Thin pipeline runner with stage schema validation. |
| crucible_ir | Stable | Experiment and stage specs, used as system contract. |

### Training Stack

| Repo | Status | Notes |
| --- | --- | --- |
| crucible_train | Production-ready | Renderers, training loops, ports, logging, stages. |
| tinkex | Mature but has gaps | Missing service endpoints and typed responses in some areas. |
| tinkex_cookbook | Phase 1 complete | sl_basic parity verified; currently not thin enough. |

### Data and Datasets

| Repo | Status | Notes |
| --- | --- | --- |
| hf_datasets_ex | Mature | HF datasets parity with streaming and PCG64 shuffle. |
| hf_hub_ex | Mature | Hub access and model/dataset operations. |
| crucible_datasets | Mature | Benchmark datasets + MemoryDataset; inspect-ai parity features. |
| datasets_ex | Optional | Internal dataset system; keep separate from HF stack. |

### Evaluation and Experimentation

| Repo | Status | Notes |
| --- | --- | --- |
| eval_ex | Mature | inspect-ai inspired tasks and scorers. |
| crucible_harness | Mature | Experiment runner with solver pipeline. |
| crucible_bench | Available | Statistical testing and comparisons. |

### Observability and MLOps

| Repo | Status | Notes |
| --- | --- | --- |
| crucible_telemetry | Mature | Metrics storage and streaming events. |
| crucible_trace | Mature | Causal traces. |
| crucible_model_registry | Production-ready | Artifact storage and lineage. |
| crucible_deployment | Production-ready | Deployment targets + rollout strategies. |
| crucible_feedback | Production-ready | Production feedback loops. |

### Reliability Layer

| Repo | Status | Notes |
| --- | --- | --- |
| crucible_ensemble | Available | Ensemble voting strategies. |
| crucible_hedging | Available | Request hedging patterns. |
| crucible_adversary | Available | Adversarial testing. |
| crucible_xai | Available | Explainability tools. |

### Control Plane and Agents

| Repo | Status | Notes |
| --- | --- | --- |
| nsai_gateway | Existing | Gateway, auth, routing, telemetry. |
| nsai_registry | Existing | Service discovery and health. |
| nsai_sites | Existing | Public surface for NSAI. |
| pilot | Existing | CLI and operator shell. |
| command | Mature | Agent orchestration with approvals and RAG (portfolio integration recommended). |
| synapse | Mature | Signal-driven multi-agent orchestration. |
| flowstone | Mature | Asset-first data orchestration for BEAM. |

### LLM SDKs

| Repo | Status | Notes |
| --- | --- | --- |
| claude_agent_sdk | Mature | Claude Code CLI SDK with streaming and MCP tools. |
| codex_sdk | Mature | Codex CLI SDK with streaming and app-server transport. |
| gemini_ex | Mature | Gemini client with tools, streaming, and tuning. |

## Key Gaps

- crucible_kitchen lacks full RL, DPO, and distillation workflows.
- Adapter ownership and wiring must match corrected ownership model.
- tinkex SDK missing some service endpoints and typed responses; impacts parity.
- Evaluation plumbing between crucible_kitchen, eval_ex, and crucible_harness is partial.
- Recipe coverage beyond sl_basic is not yet wired to kitchen workflows.
- Control plane integration (nsai_gateway -> crucible_kitchen) is not formalized.
- Command and synapse integration is not standardized with kitchen run lifecycle.

## Immediate Risks

- Divergent ownership of ports and adapters across repos.
- Recipe parity testing without stable adapter wiring.
- Missing run lifecycle events, causing incomplete telemetry and lineage.
- Non-deterministic dataset ordering or tokenization when swapping adapters.
