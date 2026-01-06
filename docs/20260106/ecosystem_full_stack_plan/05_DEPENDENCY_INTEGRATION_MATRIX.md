# Dependency Integration Matrix

Date: 2026-01-06

## Port to Adapter Mapping

| Port (Owner) | Adapter (Repo) | Wraps | Notes |
| --- | --- | --- | --- |
| TrainingClient (crucible_train) | CrucibleKitchen.Adapters.Tinkex.TrainingClient | tinkex | Core training calls, forward_backward, optim_step, checkpoints |
| SamplingClient (crucible_train) | CrucibleKitchen.Adapters.Tinkex.SamplingClient | tinkex | Used for distillation and evaluation sampling |
| DatasetStore (crucible_train) | CrucibleKitchen.Adapters.HfDatasets.DatasetStore | hf_datasets_ex | Dataset loading, streaming, shuffle |
| HubClient (crucible_train) | CrucibleKitchen.Adapters.HfHub.Client | hf_hub_ex | Model and dataset hub access |
| BlobStore (crucible_train) | CrucibleKitchen.Adapters.BlobStore.{Local,S3,Hf} | crucible_model_registry | Artifact storage and lineage |
| LLMClient (crucible_train) | CrucibleKitchen.Adapters.LLMClient.{Claude,Codex,Gemini} | claude_agent_sdk, codex_sdk, gemini_ex | For tool use, eval, and orchestration |
| EmbeddingClient (crucible_train) | CrucibleKitchen.Adapters.EmbeddingClient.* | gemini_ex, embed_ex | Embeddings for retrieval workflows |
| VectorStore (crucible_train) | CrucibleKitchen.Adapters.VectorStore.* | chroma, portfolio_index | Retrieval store for tool_use (out of scope recipe) |
| MetricsStore (crucible_telemetry) | CrucibleTelemetry.Adapters.* | telemetry | Run metrics and aggregates |
| PythonBridge (crucible_kitchen) | CrucibleKitchen.Adapters.Snakebridge | snakebridge | math_verify, sympy, pylatexenc |

## Repo Integration Tasks

### crucible_kitchen
- Implement adapter registry and manifests.
- Provide workflow registry for SL, RL, DPO, distill.
- Emit telemetry events per stage.

### crucible_train
- Expose stable ports and types for kitchen usage.
- Ensure training stages are compatible with kitchen workflow execution.

### tinkex
- Add missing service endpoints (capabilities, health, model info).
- Add typed responses for training run endpoints.
- Validate forward_backward_custom support for DPO.

### hf_datasets_ex + crucible_datasets
- Confirm dataset coverage for Tulu3, GSM8K, DPO datasets.
- Validate PCG64 shuffling parity for deterministic batches.

### eval_ex + crucible_harness
- Provide Evaluate stage integration for kitchen.
- Standardize sample conversion from crucible_datasets.

### crucible_model_registry + crucible_deployment + crucible_feedback
- Standardize artifact and run metadata across kitchen runs.
- Provide deployment hooks for completed runs.

### flowstone
- Define dataset build pipelines that output DatasetRef for kitchen.
- Store dataset lineage metadata for experiments.

### command + synapse
- Define run lifecycle events and event schemas.
- Allow command sessions to trigger kitchen runs with approvals.
- Allow synapse signals to orchestrate multi-agent experiment flows.

### LLM SDKs
- Standardize LLMClient adapter options (model, cost tracking, streaming).
- Use unified error and retry policies in adapters.
