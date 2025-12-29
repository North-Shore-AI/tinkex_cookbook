# Tinkex Cookbook Developer Guide

This guide captures the core architecture, dependency strategy, and development
rules for the Elixir port of tinker-cookbook. It is written for contributors
who need to extend recipes or configuration in this repo, with adapters living
in `crucible_kitchen` and core types/renderers in `crucible_train`.

---

## 1) Goals and Non-Goals

**Goals**
- Keep external dependencies swappable and testable.
- Preserve Python behavior where it matters (rendering, datasets, config).
- Avoid runtime Python unless explicitly required (SnakeBridge only).
- Keep recipe modules thin; push infra into ports and shared modules.

**Non-goals**
- Full parity with every Python recipe in Phase 1.
- Embedding or LLM calls hardcoded in recipes.

---

## 2) Repo Map (Key Modules)

Recipes and runtime:
- `lib/tinkex_cookbook/recipes/*` (entrypoints and orchestration)
- `lib/tinkex_cookbook/runtime/*` (composition root and runners)
- `lib/tinkex_cookbook/datasets/*` (dataset helpers)
- `lib/tinkex_cookbook/eval/*` (evaluation helpers)
- `lib/tinkex_cookbook/utils/*` and `lib/tinkex_cookbook/tokenizer_utils.ex` (shared utilities)

Legacy (deprecated; do not extend):
- `lib/tinkex_cookbook/adapters/*` (superseded by `crucible_kitchen` adapters)

External (authoritative):
- `crucible_kitchen/lib/crucible_kitchen/adapters/*` (adapter implementations)
- `crucible_train/lib/crucible_train/types/*` and `crucible_train/lib/crucible_train/renderers/*`

---

## 3) Adapter Ownership (Corrected Model)

TinkexCookbook recipes call CrucibleKitchen workflows and supply adapter maps.
Adapters implementing CrucibleTrain ports live in `crucible_kitchen`, not here.

Example adapter map:
```elixir
adapters = %{
  training_client: {CrucibleKitchen.Adapters.Tinkex.TrainingClient, []},
  dataset_store: {CrucibleKitchen.Adapters.HfDatasets.DatasetStore, []},
  hub_client: {CrucibleKitchen.Adapters.HfHub.HubClient, []},
  blob_store: {CrucibleKitchen.Adapters.Noop.BlobStore, []}
}

CrucibleKitchen.run(TinkexCookbook.Recipes.SlBasicV2, config, adapters: adapters)
```

Legacy note: modules under `lib/tinkex_cookbook/adapters` are deprecated and
will be removed. Do not add new adapters here.

---

## 4) Legacy LLM Adapters (Deprecated)

LLM adapter modules under `lib/tinkex_cookbook/adapters/llm_client` are legacy.
Do not extend them; prefer external integration via `crucible_kitchen` adapters
or application-specific tooling.

---

## 5) Dataset + Hub Stack

- Dataset adapters in `crucible_kitchen` use `hf_datasets_ex`.
- Hub adapters in `crucible_kitchen` use `hf_hub_ex`.
- Do not reintroduce Python datasets or CrucibleDatasets.

DatasetStore port supports:
- `load_dataset/3`
- `get_split/3`
- `shuffle/3`
- `take/3`
- `skip/3`
- `select/3`
- `to_list/2`

---

## 6) SnakeBridge (Python Interop)

Use SnakeBridge for the Python-only math stack:
- `sympy`
- `pylatexenc`
- `math_verify`

Rules:
- Use minimal manifests only.
- No ad-hoc Python calls from Elixir.
- Keep `allow_unsafe: false` unless explicitly approved.

---

## 7) Config and CLI

ChzEx is the standard schema/CLI layer:
- Keep CLI keys as strings until matched to schema fields.
- Never create atoms from user input.
- Use `ChzEx.entrypoint/2` for CLI entrypoints.

---

## 8) Testing and Quality

Testing policy:
- TDD for all core changes.
- Use Mox mocks for ports and external clients.
- No network calls in tests.
- Avoid `Process.sleep`; use deterministic sync helpers.

Quality gates:
- `mix format`
- `mix test`
- `mix credo --strict`
- `mix dialyzer`

---

## 9) Adding a New Adapter

Checklist:
1. Add/update a port behaviour in `crucible_train` or `crucible_telemetry` if needed.
2. Implement adapter under `crucible_kitchen/lib/crucible_kitchen/adapters`.
3. Add tests in `crucible_kitchen`.
4. Update corrected ownership docs (`crucible_kitchen/docs/corrected_ownership_model/*`).
5. Do not add new adapters to `tinkex_cookbook` (legacy modules are deprecated).

---

## 10) Adding a New Recipe

Checklist:
1. Define config with ChzEx.
2. Use ports for all external calls.
3. Keep orchestration in recipe; move logic to shared modules.
4. Write tests for dataset, renderer, and CLI behavior.

---

## 11) Reference Docs

- `crucible_kitchen/docs/corrected_ownership_model/00-canonical-architecture.md`
- `docs/20251223/PORTS_AND_ADAPTERS.md` (historical)
- `docs/20251223/PYTHON_TO_ELIXIR_LIBRARY_MAPPING.md`
- `docs/20251221/COOKBOOK_CORE_FOUNDATION.md`
