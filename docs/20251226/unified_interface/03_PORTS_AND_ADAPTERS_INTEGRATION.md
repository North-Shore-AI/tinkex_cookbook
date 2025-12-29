# Ports and Adapters Integration (Tinker)

Date: 2025-12-26
Status: Draft
Owner: North-Shore-AI

NOTE: Superseded by the corrected ownership model. Adapter implementations now
live in `crucible_kitchen`; TinkexCookbook provides recipes/config only. The
examples below are retained for historical context.

## 1) Purpose

Define how `tinkex_cookbook` implements adapters for Crucible ports and how the facade wires them. All external integrations must go through ports.

## 2) Required CrucibleTrain Ports

`crucible_train` defines these ports (behaviours):

- `CrucibleTrain.Ports.TrainingClient`
- `CrucibleTrain.Ports.DatasetStore`
- `CrucibleTrain.Ports.BlobStore`
- `CrucibleTrain.Ports.HubClient`
- `CrucibleTrain.Ports.EmbeddingClient`
- `CrucibleTrain.Ports.VectorStore` (optional)

## 3) Tinker Adapters

### 3.1 TrainingClient

Adapter: `TinkexCookbook.Adapters.TrainingClient.Tinkex`

Responsibilities:
- Start training session via `Tinkex.TrainingClient`.
- Implement `forward_backward`, `optim_step`, `save_state`, `save_weights_for_sampler`.
- Return Task-based futures where required.

### 3.2 DatasetStore

Adapter: `TinkexCookbook.Adapters.DatasetStore.HfDatasets`

Responsibilities:
- Load HuggingFace datasets using `HfDatasetsEx`.
- Provide sample iteration with deterministic shuffling (seed parity).
- Convert to `CrucibleTrain` dataset types where needed.

### 3.3 HubClient

Adapter: `TinkexCookbook.Adapters.HubClient.HfHub`

Responsibilities:
- Model and file downloads via `HfHub`.
- Access to repo metadata if needed for recipes.

### 3.4 BlobStore

Adapters:
- `TinkexCookbook.Adapters.BlobStore.Local`
- `TinkexCookbook.Adapters.BlobStore.S3`

Responsibilities:
- Save and load artifacts (checkpoints, configs, metrics).
- Use `ExAws.S3` for cloud paths.

### 3.5 EmbeddingClient

Adapter: `TinkexCookbook.Adapters.EmbeddingClient.Noop` (default)

Optional future adapters:
- Tinker embedding API
- External embedding service

## 4) Manifest System (Minimal)

Manifests are named port override maps resolved by the facade:

- Defaults live in `TinkexCookbook.Ports`.
- Manifests provide environment-specific overrides.
- The facade merges: defaults < manifest < explicit overrides.
- Manifest selection uses strings (no atom creation).

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

## 5) Ports Composition

The facade should expose a single composition root:

```elixir
ports = TinkexCookbook.Ports.new(
  ports: [
    training_client: {TinkexCookbook.Adapters.TrainingClient.Tinkex, []},
    dataset_store: {TinkexCookbook.Adapters.DatasetStore.HfDatasets, []},
    blob_store: {TinkexCookbook.Adapters.BlobStore.Local, []},
    hub_client: {TinkexCookbook.Adapters.HubClient.HfHub, []}
  ]
)
```

`TinkexCookbook.Runtime` is the composition root; it builds `ports` and injects them into stage options or context assigns. Recipes never call `TinkexCookbook.Ports.new/1` directly.

## 6) Error Normalization

All adapters should normalize external errors to a common error shape:
- `{:error, %CrucibleTrain.Ports.Error{}}`

This ensures consistent handling across recipes and stages.

## 7) Acceptance Criteria

- No direct client calls in recipes.
- All external integrations resolved through ports.
- Adapters are testable with Mox.
