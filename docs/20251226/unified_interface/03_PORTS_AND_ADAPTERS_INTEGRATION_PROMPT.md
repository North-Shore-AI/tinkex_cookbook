# Prompt: Implement Ports and Adapters Integration

Date: 2025-12-26

NOTE: Superseded by the corrected ownership model. Adapter implementations now
live in `crucible_kitchen`; TinkexCookbook provides recipes/config only.

## Goal

Implement the ports and adapters integration described in:
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251226/unified_interface/03_PORTS_AND_ADAPTERS_INTEGRATION.md`

## Required Reading (Full Paths)

- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251226/unified_interface/03_PORTS_AND_ADAPTERS_INTEGRATION.md`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251223/PORTS_AND_ADAPTERS.md`
- `/home/home/p/g/North-Shore-AI/crucible_train/lib/crucible_train/ports/ports.ex`
- `/home/home/p/g/North-Shore-AI/crucible_train/lib/crucible_train/ports/training_client.ex`
- `/home/home/p/g/North-Shore-AI/crucible_train/lib/crucible_train/ports/dataset_store.ex`
- `/home/home/p/g/North-Shore-AI/crucible_train/lib/crucible_train/ports/blob_store.ex`
- `/home/home/p/g/North-Shore-AI/crucible_train/lib/crucible_train/ports/hub_client.ex`

## Context Summary

`tinkex_cookbook` must provide Tinker-specific adapters for CrucibleTrain ports and resolve them through a single composition root (`TinkexCookbook.Runtime`). Recipes must not call external clients directly.

## Implementation Requirements

1) Implement adapters for TrainingClient, DatasetStore, HubClient, BlobStore.
2) Add a minimal manifest system for port wiring.
3) Wire adapters via `TinkexCookbook.Ports.new/1` from the facade.
4) Normalize errors to `CrucibleTrain.Ports.Error`.
5) Add adapter tests with Mox.

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
