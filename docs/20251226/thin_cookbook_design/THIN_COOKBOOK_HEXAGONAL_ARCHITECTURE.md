# Thin TinkexCookbook Hexagonal Architecture

Date: 2025-12-26
Status: Draft
Owner: North-Shore-AI

NOTE: Superseded by the corrected ownership model. Adapter implementations now
live in `crucible_kitchen`; TinkexCookbook provides recipes/config only.

## 1) Purpose

Define the hexagonal (ports and adapters) architecture for the thin
`tinkex_cookbook` rewrite, with a unified facade as the composition root.

## 2) Core Rule

- Recipes and CLI entrypoints call `TinkexCookbook.Runtime` only.
- The facade resolves ports, builds the spec, and runs the pipeline.
- All external systems are accessed through ports and adapters.

## 3) Hexagonal Layout

```
Inbound
  recipes/* + CLI
        |
        v
TinkexCookbook.Runtime (facade + composition root)
        |
        v
CrucibleIR.Experiment
        |
        v
CrucibleFramework.run/2
        |
        v
CrucibleTrain stages
        |
        v
Outbound ports (CrucibleTrain.Ports.*)
        |
        v
TinkexCookbook.Adapters.*  --->  External systems (Tinker API, HF, S3, etc)
```

## 4) Composition Root

`TinkexCookbook.Runtime` is the composition root. It:

- Resolves ports via `TinkexCookbook.Ports.new/1`.
- Injects ports into stage options or context assigns.
- Owns manifest selection and overrides.

Recipes never call `TinkexCookbook.Ports.new/1` directly.

## 5) Manifest System (Minimal)

Manifests are named port override maps for environment-specific wiring.

- String-keyed names (no atom creation).
- Merge order: defaults < manifest < explicit overrides.
- Lives in `TinkexCookbook.Runtime` config.

Example:

```elixir
config :tinkex_cookbook, TinkexCookbook.Runtime,
  default_manifest: "local",
  manifests: %{
    "local" => [
      blob_store: {TinkexCookbook.Adapters.BlobStore.Local, []}
    ],
    "prod" => [
      blob_store: {TinkexCookbook.Adapters.BlobStore.S3, bucket: "tinkex-prod"}
    ]
  }
```

## 6) Invariants (Non-Negotiable)

- Recipes are orchestration only; no direct client calls.
- All external integrations go through ports/adapters.
- The facade is the only entrypoint for execution.
- Training logic lives in `crucible_train`, not the cookbook.

## 7) Execution Flow (Happy Path)

1) CLI parses recipe config with ChzEx.
2) `TinkexCookbook.Runtime.run/2` selects a manifest and builds `ports`.
3) The recipe builds a `CrucibleIR.Experiment`.
4) The facade runs the pipeline via `CrucibleFramework.run/2`.
5) Stages call outbound ports implemented by adapters.
