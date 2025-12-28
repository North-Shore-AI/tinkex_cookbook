# Prompt: Implement Migration and Parity Plan

Date: 2025-12-26

## Goal

Implement the migration and parity plan described in:
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251226/unified_interface/05_TESTING_PARITY_AND_MIGRATION_PLAN.md`

## Required Reading (Full Paths)

- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251226/unified_interface/05_TESTING_PARITY_AND_MIGRATION_PLAN.md`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251224/recipe_parity/RECIPE_PARITY_PROTOCOL.md`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251224/parity_investigation/PARITY_MISMATCH_INVESTIGATION.md`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/scripts/parity/`

## Context Summary

The rewrite must be validated against Python behavior using the parity harness. Migration should be phased and aligned with parity checks.

## Implementation Requirements

1) Add or update parity scripts for new facade-based recipes.
2) Ensure artifact logging matches the protocol.
3) Add tests for parity helpers in Elixir.

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

