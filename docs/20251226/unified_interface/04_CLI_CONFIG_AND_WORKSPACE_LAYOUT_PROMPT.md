# Prompt: Implement CLI and Config Layout

Date: 2025-12-26

## Goal

Implement the CLI and config layout described in:
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251226/unified_interface/04_CLI_CONFIG_AND_WORKSPACE_LAYOUT.md`

## Required Reading (Full Paths)

- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251226/unified_interface/04_CLI_CONFIG_AND_WORKSPACE_LAYOUT.md`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251221/PHASE1_FOUNDATION_SLICE.md`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/tinker-cookbook/tinker_cookbook/recipes/sl_basic.py`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/tinker-cookbook/tinker_cookbook/recipes/chat_sl/train.py`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/lib/tinkex_cookbook/utils/cli_utils.ex`

## Context Summary

CLI entrypoints must use ChzEx and route through the facade. Log path handling and run naming should match Python cookbook behavior. Manifest selection should be exposed via CLI or env config.

## Implementation Requirements

1) Define ChzEx config schemas for core recipes.
2) Provide CLI entrypoints that call the facade.
3) Add CLI support for manifest selection.
4) Implement log path handling consistent with Python.
5) Update workspace layout to match the new module structure.

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
