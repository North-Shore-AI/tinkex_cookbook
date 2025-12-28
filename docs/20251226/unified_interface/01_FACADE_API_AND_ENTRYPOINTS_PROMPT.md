# Prompt: Implement Facade API and Entrypoints

Date: 2025-12-26

## Goal

Implement the facade API described in:
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251226/unified_interface/01_FACADE_API_AND_ENTRYPOINTS.md`

## Required Reading (Full Paths)

- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251226/unified_interface/01_FACADE_API_AND_ENTRYPOINTS.md`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/lib/tinkex_cookbook/recipes/`
- `/home/home/p/g/North-Shore-AI/crucible_framework/lib/crucible_framework.ex`
- `/home/home/p/g/North-Shore-AI/crucible_framework/lib/crucible/pipeline/runner.ex`
- `/home/home/p/g/North-Shore-AI/crucible_ir/lib/crucible_ir/experiment.ex`

## Context Summary

Recipes must call a single facade module. The facade builds an IR spec, resolves ports via manifests, and runs it via `CrucibleFramework.run/2`. CLI entrypoints should route through this facade.

## Implementation Requirements

1) Add `TinkexCookbook.Runtime` (or equivalent) with `build_spec/2`, `run/2`, `run_spec/2`, and `build_ports/1`.
2) Define a recipe behaviour to standardize `build_spec/1`.
3) Add minimal manifest selection for port wiring.
4) Update recipe entrypoints to call the facade.
5) Add tests for facade and recipe interface.

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
