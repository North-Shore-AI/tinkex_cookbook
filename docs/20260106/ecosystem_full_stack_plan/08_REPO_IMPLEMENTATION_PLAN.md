# Repo Implementation Plan

Date: 2026-01-06

## Core Orchestration

### crucible_kitchen
- Implement full workflow registry (supervised, RL, DPO, distill).
- Own adapter layer for external SDKs.
- Emit run lifecycle events and telemetry.
- Provide run API for control plane.

### crucible_framework
- Ensure stage schema validation is enabled for kitchen workflows.
- Provide optional persistence hooks for run history.

### crucible_ir
- Keep Experiment and StageDef schemas as the contract.
- Add any missing fields required for MLOps integration (run ids, artifact refs).

## Training and SDKs

### crucible_train
- Ensure ports are stable and well-documented.
- Provide comparison datum types and DPO metrics if missing.
- Provide RL metrics and advantage computation helpers.

### tinkex
- Add missing service endpoints and typed responses.
- Verify forward_backward_custom or equivalent for DPO.
- Ensure tokenizer and sampling APIs are stable for kitchen adapters.

### tinkex_cookbook
- Thin to recipes and configs only.
- Use crucible_kitchen for all execution.
- Keep parity tooling for the five focus recipes.

## Data and Datasets

### hf_datasets_ex
- Validate schema mappings for chat and preference datasets.
- Ensure deterministic shuffling (PCG64) in all dataset builders.

### hf_hub_ex
- Verify model and dataset download flows for kitchen adapters.
- Support gated datasets via HF_TOKEN.

### crucible_datasets
- Provide dataset references for GSM8K, MMLU, HumanEval, NoRobots.
- Provide MemoryDataset conversions for eval_ex tasks.

### datasets_ex
- Keep optional internal datasets system; avoid overlap with HF.

### flowstone
- Provide dataset build pipelines that output DatasetRef artifacts.
- Record dataset lineage metadata for experiments.

## Evaluation and Experimentation

### eval_ex
- Provide task registry for kitchen Evaluate stage.
- Expand scorers as needed for DPO and RL metrics.

### crucible_harness
- Provide batch runs for recipe regressions.
- Integrate with kitchen runs for large sweeps.

### crucible_bench
- Use for statistical comparisons of experiment variants.

## Observability

### crucible_telemetry
- Define metrics schemas for training and evaluation stages.
- Provide MetricsStore adapter used by kitchen.

### crucible_trace
- Provide trace events for stage-level diagnostics.

## MLOps

### crucible_model_registry
- Register checkpoints and model artifacts from kitchen runs.
- Store lineage linking to dataset and config hashes.

### crucible_deployment
- Support deployment of registry artifacts (optional phase).

### crucible_feedback
- Ingest production signals back into dataset pipelines.

## Control Plane and Agents

### nsai_gateway
- Expose REST endpoints for run create, status, cancel, and artifact fetch.

### nsai_registry
- Track service health for kitchen, telemetry, and registry services.

### nsai_sites
- Surface run status and experiment dashboards.

### pilot
- Provide CLI tools for triggering kitchen runs.

### command
- Use as approval and session system for human-in-the-loop runs.
- Trigger kitchen runs and store run metadata.

### synapse
- Orchestrate multi-agent research loops via run lifecycle signals.

## LLM SDKs and Python Bridge

### claude_agent_sdk, codex_sdk, gemini_ex
- Provide LLMClient adapters in kitchen.
- Standardize error, retry, and cost tracking policies.

### snakebridge + snakepit
- Provide math_verify, sympy, pylatexenc manifests.
- Use in math_rl and evaluation recipes.

## Config and Tokenization

### chz_ex
- Define schema for all recipe configs.
- Maintain parity with Python defaults.

### tiktoken_ex
- Provide tokenizer access where needed by adapters.
