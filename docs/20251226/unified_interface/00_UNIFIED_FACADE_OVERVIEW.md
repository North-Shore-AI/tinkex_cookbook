# Unified Facade Overview for TinkexCookbook

Date: 2025-12-26
Status: Draft
Owner: North-Shore-AI

NOTE: Superseded by the corrected ownership model. Adapter implementations now
live in `crucible_kitchen`; TinkexCookbook provides recipes/config only.

## 1) Purpose

Define a unified facade for a ground-up rewrite of `tinkex_cookbook` so recipes consume a single stable API while all training infrastructure lives in the Crucible ecosystem. The facade hides internal complexity but preserves the layered architecture.

## 2) Sources Used

This design is based on the following local docs and Python reference:

- `docs/20251221/COOKBOOK_CORE_FOUNDATION.md`
- `docs/20251221/PHASE1_FOUNDATION_SLICE.md`
- `docs/20251221/REMAINING_WORK.md`
- `docs/20251221/DEPENDENCY_USAGE_TABLE.md`
- `docs/20251223/INSPECT_AI_ELIXIR_ARCHITECTURE.md`
- `docs/20251223/PYTHON_TO_ELIXIR_LIBRARY_MAPPING.md`
- `docs/20251223/PORTS_AND_ADAPTERS.md`
- `docs/20251224/inspect_ai_scope_mapping_v2/*`
- `docs/20251224/recipe_parity/RECIPE_PARITY_PROTOCOL.md`
- `docs/20251226/thin_cookbook_design/THIN_COOKBOOK_FOUNDATION_DESIGN.md`
- `docs/20251226/thin_cookbook_design/THIN_COOKBOOK_HEXAGONAL_ARCHITECTURE.md`
- Python cookbook: `tinker-cookbook/tinker_cookbook/recipes/*` and `tinker_cookbook/model_info.py`

## 3) Guiding Principles

1) Thin cookbook: recipes and adapters only.
2) Unified facade: one entrypoint for running recipes.
3) Spec driven: recipes build `CrucibleIR.Experiment`.
4) Clear ownership: training in `crucible_train`, orchestration in `crucible_framework`, specs in `crucible_ir`.
5) Ports and adapters: all external integrations are swappable.
6) Minimal manifests: named port maps for environment-specific wiring.

## 4) What Changes

- `tinkex_cookbook` stops shipping renderers, training loops, and core types.
- The cookbook provides a facade module (single API) and Tinker-specific adapters.
- Recipes call the facade only.
- All training and evaluation logic delegates to Crucible libraries.
- The facade lives in `tinkex_cookbook` (no new repo).

## 5) Target Architecture

```
recipes -> TinkexCookbook.Runtime (facade)
         -> CrucibleIR.Experiment
         -> CrucibleFramework.run/2
         -> CrucibleTrain stages
         -> Crucible services (registry, deployment, feedback) optional
```

## 6) Facade Responsibilities

- Build a spec from recipe config.
- Resolve ports and adapters (Tinker specific) using manifests and overrides.
- Execute via `CrucibleFramework.run/2`.
- Provide consistent logging, run ids, and artifact layout.
- Optionally run evals via EvalEx and CrucibleHarness.

## 7) Non-Goals

- Do not move execution into `crucible_ir`.
- Do not duplicate training infrastructure in the cookbook.
- Do not add cross-layer dependencies that violate the ecosystem boundaries.

## 8) Primary Deliverables

- A facade module (`TinkexCookbook.Runtime` or `TinkexCookbook.Platform`).
- A recipe behaviour with a spec builder.
- Tinker adapters for CrucibleTrain ports.
- A minimal manifest system for port wiring (local/dev/prod/test).
- Updated CLI and configs to route through the facade.
- A migration plan and parity validation for the rewrite.

## 9) Open Questions

- Do recipes expose a public API beyond CLI (library usage)?
- Do we need a dedicated `RecipeSpec` struct in the cookbook or direct IR usage only?
- Which eval paths must retain inspect-ai parity vs migrate to `CrucibleTrain.Eval`?
