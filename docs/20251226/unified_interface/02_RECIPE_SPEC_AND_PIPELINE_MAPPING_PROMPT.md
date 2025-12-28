# Prompt: Implement Recipe Spec and Pipeline Mapping

Date: 2025-12-26

## Goal

Implement the spec mapping described in:
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251226/unified_interface/02_RECIPE_SPEC_AND_PIPELINE_MAPPING.md`

## Required Reading (Full Paths)

- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251226/unified_interface/02_RECIPE_SPEC_AND_PIPELINE_MAPPING.md`
- `/home/home/p/g/North-Shore-AI/crucible_ir/lib/crucible_ir/experiment.ex`
- `/home/home/p/g/North-Shore-AI/crucible_ir/lib/crucible_ir/stage_def.ex`
- `/home/home/p/g/North-Shore-AI/crucible_ir/lib/crucible_ir/training/config.ex`
- `/home/home/p/g/North-Shore-AI/crucible_train/lib/crucible_train/stages/`
- `/home/home/p/g/North-Shore-AI/crucible_framework/lib/crucible/pipeline/runner.ex`

## Context Summary

Recipes must emit a `CrucibleIR.Experiment` spec that uses `CrucibleTrain` stages. Optional registry, deployment, and feedback stages can be appended.

## Implementation Requirements

1) Build specs for core recipes (sl_basic, rl_basic, dpo, distillation).
2) Ensure stage options include `training_config` and `ports`.
3) Add tests to validate spec shape and stage ordering.

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

