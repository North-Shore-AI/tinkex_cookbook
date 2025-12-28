# Prompt: Implement Thin TinkexCookbook Foundation

Date: 2025-12-26

## Goal

Implement the architecture described in:
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251226/thin_cookbook_design/THIN_COOKBOOK_FOUNDATION_DESIGN.md`

Convert `tinkex_cookbook` into a thin recipe + adapter layer that delegates training to `crucible_train` and execution to `crucible_framework`.

## Required Reading (Full Paths)

### Repo Guidance
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/AGENTS.md`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/README.md`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/CHANGELOG.md`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/mix.exs`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251226/thin_cookbook_design/THIN_COOKBOOK_HEXAGONAL_ARCHITECTURE.md`

### Cookbook Source (current state)
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/lib/tinkex_cookbook.ex`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/lib/tinkex_cookbook/recipes/sl_basic.ex`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/lib/tinkex_cookbook/supervised/train.ex`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/lib/tinkex_cookbook/renderers/`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/lib/tinkex_cookbook/types/`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/lib/tinkex_cookbook/ports/`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/lib/tinkex_cookbook/adapters/`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/lib/tinkex_cookbook/eval/`

### Crucible Ecosystem Interfaces
- `/home/home/p/g/North-Shore-AI/crucible_train/README.md`
- `/home/home/p/g/North-Shore-AI/crucible_train/lib/crucible_train/crucible_train.ex`
- `/home/home/p/g/North-Shore-AI/crucible_train/lib/crucible_train/supervised/config.ex`
- `/home/home/p/g/North-Shore-AI/crucible_train/lib/crucible_train/stages/supervised_train.ex`
- `/home/home/p/g/North-Shore-AI/crucible_framework/README.md`
- `/home/home/p/g/North-Shore-AI/crucible_framework/lib/crucible_framework.ex`
- `/home/home/p/g/North-Shore-AI/crucible_framework/lib/crucible/pipeline/runner.ex`
- `/home/home/p/g/North-Shore-AI/crucible_ir/lib/crucible_ir/experiment.ex`
- `/home/home/p/g/North-Shore-AI/crucible_ir/lib/crucible_ir/training/config.ex`

## Context Summary

`tinkex_cookbook` will become a thin recipe/adapter layer with a unified facade. Training loops, renderers, and core types must live in `crucible_train`. Recipes should emit `CrucibleIR.Experiment` and run via `CrucibleFramework.run/2` through `TinkexCookbook.Runtime`. Tinker-specific integrations must be implemented via ports/adapters in the cookbook, resolved by the facade using a minimal manifest system.

## Implementation Requirements

1) Remove duplicated training infra from `tinkex_cookbook` (renderers, types, training loops).
2) Add `crucible_train` as the single training dependency and delegate training to it.
3) Add a unified facade (`TinkexCookbook.Runtime`) as the composition root for recipes.
4) Rebuild recipes (start with `sl_basic`) to output `CrucibleIR.Experiment` and call `CrucibleFramework.run/2` via the facade.
5) Implement Tinker adapters for required `CrucibleTrain.Ports`.
6) Keep EvalEx/CrucibleHarness wiring where inspect-ai parity is required.
7) Update CLI entrypoints and configs to reference the new spec-driven flow.
8) Add a minimal manifest system for environment-specific port wiring.
9) Maintain ports/adapters and avoid direct external calls from recipes.

## TDD and Quality Gates

- Write tests first (ExUnit + Mox as needed).
- All tests pass: `mix test`.
- No warnings: `mix compile --warnings-as-errors`.
- Format: `mix format`.
- Credo strict: `mix credo --strict`.
- Dialyzer: `mix dialyzer`.

## Version Bump (Required)

- Bump version `0.x.y` in `/home/home/p/g/North-Shore-AI/tinkex_cookbook/mix.exs`.
- Update `/home/home/p/g/North-Shore-AI/tinkex_cookbook/README.md` to reflect the new version.
- Add a 2025-12-26 entry to `/home/home/p/g/North-Shore-AI/tinkex_cookbook/CHANGELOG.md`.

## Acceptance Criteria

- `tinkex_cookbook` is thin (recipes + adapters only).
- Training, renderers, and types are delegated to `crucible_train`.
- Recipes run via `CrucibleFramework.run/2` using `CrucibleIR.Experiment`.
- All quality gates pass with no warnings/errors.
