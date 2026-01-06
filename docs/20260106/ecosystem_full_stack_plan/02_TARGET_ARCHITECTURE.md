# Target Architecture

Date: 2026-01-06

## Architecture Goals

- crucible_kitchen is the thick orchestration core.
- tinkex_cookbook is a thin recipe layer.
- All external SDKs are wrapped by adapters in crucible_kitchen.
- crucible_train owns training ports, types, and renderers.
- crucible_ir is the contract for experiment specs.
- NSAI control plane exposes the kitchen as a service.

## Macro Layer Diagram

```
+--------------------------------------------------------------------+
| Public Interfaces                                                  |
| nsai_sites -> nsai_gateway -> nsai_registry -> pilot               |
+--------------------------------------------------------------------+
                               |
                               v
+--------------------------------------------------------------------+
| Orchestration Core                                                 |
| crucible_kitchen (workflows, stages, adapters, run lifecycle)      |
+--------------------------------------------------------------------+
        |                      |                      |
        v                      v                      v
+---------------+      +-----------------+      +--------------------+
| Training Core |      | Evaluation Core |      | Observability Core |
| crucible_train|      | eval_ex +       |      | crucible_telemetry |
| (types, ports)|      | crucible_harness|      | crucible_trace     |
+---------------+      +-----------------+      +--------------------+
        |                      |                      |
        v                      v                      v
+----------------+  +------------------+   +-------------------------+
| Data + Hub      |  | MLOps            |   | Reliability             |
| hf_datasets_ex  |  | model_registry   |   | ensemble/hedging/xai    |
| hf_hub_ex       |  | deployment       |   | adversary              |
| crucible_datasets| | feedback         |   |                        |
+----------------+  +------------------+   +-------------------------+
        |
        v
+--------------------------------------------------------------------+
| External SDKs and Services                                         |
| tinkex, claude_agent_sdk, codex_sdk, gemini_ex, snakebridge         |
+--------------------------------------------------------------------+
```

## Control Plane and Agent Layer

```
nsai_gateway
  -> routes experiment run requests
  -> authenticates, rate limits, telemetry

command
  -> session + approvals + tool tracking
  -> triggers crucible_kitchen runs

synapse
  -> signal bus for multi-agent workflows
  -> subscribes to run lifecycle events

flowstone
  -> data pipeline orchestration
  -> produces dataset artifacts for training
```

## Contract Boundaries

- Experiment spec: CrucibleIR.Experiment and StageDef are the system contract.
- Training ports: CrucibleTrain.Ports.* define all training interfaces.
- Adapters: live in crucible_kitchen; wrap external SDKs.
- Recipes: live in tinkex_cookbook; configure and call crucible_kitchen.

## Entry Points

- CrucibleKitchen.run(recipe, config, adapters: map)
- TinkexCookbook.run(recipe, config, overrides)
- nsai_gateway endpoints map to kitchen run lifecycle

## Key Flows

1) Recipe execution
   - tinkex_cookbook builds config
   - crucible_kitchen resolves adapters
   - crucible_framework runs stages
   - crucible_train ports execute training ops

2) Evaluation
   - eval_ex tasks over crucible_datasets
   - crucible_harness runs batch experiments
   - results logged via crucible_telemetry

3) MLOps
   - checkpoints and artifacts stored in crucible_model_registry
   - deployments via crucible_deployment
   - feedback ingestion via crucible_feedback
