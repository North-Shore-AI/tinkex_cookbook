# Prompt: Implement Unified Facade Rewrite

Date: 2025-12-26

## Goal

Implement the unified facade architecture described in:
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251226/unified_interface/00_UNIFIED_FACADE_OVERVIEW.md`

## Required Reading (Full Paths)

- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251226/unified_interface/00_UNIFIED_FACADE_OVERVIEW.md`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251226/thin_cookbook_design/THIN_COOKBOOK_FOUNDATION_DESIGN.md`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251226/thin_cookbook_design/THIN_COOKBOOK_HEXAGONAL_ARCHITECTURE.md`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251223/INSPECT_AI_ELIXIR_ARCHITECTURE.md`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251223/PORTS_AND_ADAPTERS.md`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251224/recipe_parity/RECIPE_PARITY_PROTOCOL.md`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/AGENTS.md`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/mix.exs`

## Context Summary

`tinkex_cookbook` must become a thin recipe and adapter layer. Training, renderers, and core types move to `crucible_train`. Orchestration uses `crucible_framework`, and specs use `crucible_ir`. Recipes must call a single facade API implemented in the cookbook, with ports resolved via a minimal manifest system.

## Implementation Requirements

1) Add a unified facade module (`TinkexCookbook.Runtime`) and route all recipes through it.
2) Remove duplicated training infra from `tinkex_cookbook`.
3) Add Tinker adapters for CrucibleTrain ports.
4) Add a minimal manifest system for environment-specific port wiring.
5) Update recipes to emit `CrucibleIR.Experiment` and run via `CrucibleFramework.run/2`.
6) Keep eval wiring thin and optional.

## TDD and Quality Gates

- Write tests first.
- `mix test` must pass.
- `mix compile --warnings-as-errors` must be clean.
- `mix format` must be clean.
- `mix credo --strict` must be clean.
- `mix dialyzer` must be clean.

## Version Bump (Required)

- Bump version `0.x.y` in `/home/home/p/g/North-Shore-AI/tinkex_cookbook/mix.exs`.
- Update `/home/home/p/g/North-Shore-AI/tinkex_cookbook/README.md` to reflect the new version.
- Add a 2025-12-26 entry to `/home/home/p/g/North-Shore-AI/tinkex_cookbook/CHANGELOG.md`.
