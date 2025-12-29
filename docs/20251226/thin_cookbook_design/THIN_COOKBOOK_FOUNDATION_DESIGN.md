# Thin TinkexCookbook Foundation Design

Date: 2025-12-26
Status: Draft
Owner: North-Shore-AI

NOTE: Superseded by the corrected ownership model. Adapter implementations now
live in `crucible_kitchen`; TinkexCookbook provides recipes/config only.

## 1) Purpose

Rebuild `tinkex_cookbook` as a thin recipe and adapter layer that delegates all training and core ML infrastructure to the Crucible ecosystem. This document defines the architectural boundaries, module layout, integration points, and migration path to make the cookbook the canonical foundation for Tinker recipes without duplicating training infrastructure.

Related docs:
- `docs/20251226/thin_cookbook_design/THIN_COOKBOOK_HEXAGONAL_ARCHITECTURE.md`

## 2) Goals

- Make `tinkex_cookbook` a thin layer that:
  - Defines recipes and CLI entrypoints.
  - Provides Tinker-specific adapters for Crucible ports.
  - Produces `CrucibleIR` specs and runs them via `crucible_framework`.
- Provide a unified facade (`TinkexCookbook.Runtime`) so recipes call one entrypoint.
- Use a minimal manifest system for environment-specific port wiring.
- Eliminate duplicated training infrastructure in `tinkex_cookbook`.
- Use `crucible_train` as the single source of truth for training loops, renderers, types, and logging.
- Preserve inspect-ai parity for evaluation by keeping EvalEx/CrucibleHarness where needed.
- Provide a clean, testable integration surface with ports and adapters.

## 3) Non-Goals

- Re-implement training loops in `tinkex_cookbook`.
- Move core training types/renderers into the cookbook.
- Build production adapters inside `crucible_train` (these belong in the cookbook or external apps).
- Make `crucible_ir` execute or orchestrate runs.

## 4) Architectural Decisions (Non-Negotiable)

1) Training infrastructure lives in `crucible_train`.
2) Execution/orchestration lives in `crucible_framework`.
3) Specs live in `crucible_ir` (data only; no execution).
4) `tinkex_cookbook` is recipes + adapters only.
5) All external integrations go through ports/adapters.
6) The facade (`TinkexCookbook.Runtime`) is the single entrypoint for recipes.
7) `TinkexCookbook.Runtime` is the composition root; `TinkexCookbook.Ports` is the adapter resolver.
8) Manifests are minimal, named port maps (no dynamic module loading).

## 5) Layering Model

```
crucible_ir           (specs, serialization)
crucible_framework    (runner, pipeline, context)
crucible_train        (training loops, renderers, types, logging)
crucible_* services   (model registry, deployment, feedback)
----------------------------------------------
tinkex_cookbook       (runtime + recipes + adapters + CLI)
```

## 6) Responsibilities by Repo

### crucible_ir
- Data structures for experiments, training config, deployment config, feedback.
- JSON serialization and validation utilities.

### crucible_framework
- Pipeline runner and stage orchestration.
- Context, persistence, registry, tracing integration.

### crucible_train
- Training loops (SL, RL, DPO, distill).
- Renderers, types, logging, evaluation helpers.
- Stage implementations for training.

### crucible_* (model_registry, deployment, feedback)
- Domain logic, storage, and stage implementations.

### tinkex_cookbook
- Recipes that emit `CrucibleIR.Experiment` and/or call `CrucibleFramework.run/2`.
- Unified facade (`TinkexCookbook.Runtime`) as the single entrypoint.
- Tinker adapters for Crucible ports (TrainingClient, DatasetStore, BlobStore, etc.).
- CLI entrypoints and recipe configuration (ChzEx).
- Optional EvalEx/Harness integration for inspect-ai parity.

## 7) Target Module Layout (Cookbook)

```
lib/tinkex_cookbook/
  recipes/                 # Recipe modules (thin orchestration only)
  runtime/                 # Unified facade + composition root
  cli/                     # CLI entrypoints if not using Mix.Tasks
  adapters/                # Tinker-specific adapters for Crucible ports
  ports/                   # Adapter resolver (ports registry)
  datasets/                # Recipe-specific dataset loaders (thin)
  eval/                    # Inspect-ai parity eval wiring (optional)
  config/                  # ChzEx schemas for recipe configs
  utils/                   # Small helpers only
```

### What Moves Out (or is deleted)
- Renderers, types, training loops, logging implementations.
- Training dataset logic that duplicates `crucible_train`.

### What Stays
- Recipe definitions (e.g., `sl_basic`) but rewritten to call into `crucible_train`.
- Tinker adapters (TrainingClient, SamplingClient, DatasetStore, BlobStore).
- CLI entrypoints and configuration schemas.
- Inspect-ai evaluation wiring where needed.

## 8) Recipe Execution Model

### Canonical run surface
- Recipes produce `CrucibleIR.Experiment` (pipeline spec).
- Execution uses `CrucibleFramework.run/2`.
- Training stages in pipeline delegate to `CrucibleTrain`.

### Example flow
1) `tinkex_cookbook` recipe builds `CrucibleIR.Training.Config`.
2) Recipe builds `CrucibleIR.Experiment` pipeline:
   - `CrucibleTrain.Stages.SupervisedTrain`
   - Optional `CrucibleModelRegistry.Stages.Register`
   - Optional `CrucibleDeployment.Stages.Deploy`
   - Optional `CrucibleFeedback.Stages.CheckTriggers`
3) `CrucibleFramework.run/2` executes pipeline.

## 9) Adapter Contracts (Tinker-Specific)

`tinkex_cookbook` must provide adapters for Crucible ports:

- `CrucibleTrain.Ports.TrainingClient` -> Tinker training API
- `CrucibleTrain.Ports.DatasetStore` -> dataset retrieval (HF + Tinker)
- `CrucibleTrain.Ports.BlobStore` -> artifact storage (S3/Tinker)
- `CrucibleTrain.Ports.EmbeddingClient` -> embedding service (optional)
- `CrucibleTrain.Ports.HubClient` -> registry/hub integration

Adapters are resolved by `TinkexCookbook.Runtime` via `TinkexCookbook.Ports.new/1` and injected into stages. Recipes never resolve ports directly.

## 10) Manifest System (Minimal)

Manifests are small, named port override maps resolved by the facade:

- Default ports live in `TinkexCookbook.Ports`.
- A manifest provides environment-specific overrides (local, dev, prod, test).
- The facade merges: defaults < manifest < explicit overrides.
- Manifest selection uses strings (no atom creation from CLI input).

## 11) Evaluation Strategy

- Use EvalEx + CrucibleHarness for inspect-ai parity tasks.
- Use `CrucibleTrain.Eval` for training-time evaluation if needed.
- Keep evaluation wiring in the cookbook; do not re-implement scoring in recipes.

## 12) Migration Plan

Phase 0: Foundation
- Add `crucible_train` dependency and remove duplicate modules.
- Introduce adapter modules for Tinker ports.
- Update recipe configs to use `CrucibleTrain` types.

Phase 1: Recipe Rewrite
- Rebuild `sl_basic` to emit `CrucibleIR.Experiment` and run via framework.
- Ensure parity tests with previous outputs (token-level parity where applicable).

Phase 2: Expand Recipes
- Port remaining recipes by delegating to `crucible_train` and adding adapters.
- Add EvalEx-based evaluation paths as needed.

Phase 3: MLOps Pipeline Recipes
- Add recipe templates that include registry/deployment/feedback stages.

## 13) Acceptance Criteria

- `tinkex_cookbook` no longer owns training loops, renderers, or core types.
- All recipes delegate to `crucible_train` for training logic.
- Recipes output `CrucibleIR.Experiment` and run via `CrucibleFramework.run/2`.
- Recipes call the unified facade only; no direct ports or client calls.
- Manifests provide environment-specific port wiring with minimal complexity.
- Ports/adapters provide Tinker-specific integration points.
- Tests pass with no warnings; parity checks documented where applicable.

## 14) Open Questions

- Which recipes must retain EvalEx parity vs migrate to `CrucibleTrain.Eval`?
- Should the cookbook expose a minimal public API for recipe composition beyond CLI?
- How much of the existing dataset logic should be preserved vs moved to `crucible_train`?
