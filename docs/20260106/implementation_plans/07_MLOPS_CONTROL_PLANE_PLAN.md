# MLOps and Control Plane Plan

Date: 2026-01-06

## Goals

- Wire kitchen runs into model registry, deployment, and feedback loops.
- Expose run lifecycle through NSAI control plane.
- Integrate command and synapse for HITL and multi-agent orchestration.

## Model Registry Integration

### Tasks

- Register checkpoints and artifacts on run completion.
- Record dataset refs, config hashes, and run metadata.
- Store lineage from teacher to student for distillation.

### Acceptance

- Registry entries exist for every completed run.
- Lineage graph links datasets and parents.

## Deployment Integration (Optional per workflow)

- Add deploy stage using crucible_deployment.
- Support canary and blue-green deployment strategies.
- Emit deployment status to telemetry.

## Feedback Integration

- Add feedback ingestion hooks from crucible_feedback.
- Feed curated feedback into dataset pipeline for retraining.

## Control Plane APIs

### nsai_gateway

- Endpoints:
  - POST /experiments/run
  - GET /experiments/:id/status
  - POST /experiments/:id/cancel
  - GET /experiments/:id/artifacts

### nsai_registry

- Register kitchen service health and capabilities.

### nsai_sites

- Surface run status and latest experiment metrics.

## Command and Synapse Integration

- command:
  - Use approvals for high-cost runs.
  - Store run metadata in sessions.

- synapse:
  - Publish run lifecycle signals.
  - Enable multi-agent experiment loops (analysis, verification, rollout).

## Flowstone Integration

- Define dataset build pipelines that output DatasetRef artifacts.
- Record dataset lineage for downstream runs.

## Acceptance Criteria

- Run lifecycle visible through gateway and registry.
- Artifacts accessible via registry and gateway endpoints.
- Feedback ingestion round-trip validated with a sample dataset.
