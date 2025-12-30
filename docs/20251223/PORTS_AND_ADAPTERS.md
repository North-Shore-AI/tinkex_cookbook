# Ports and Adapters (Core Architecture)

**Purpose:** keep external services swappable and testable without touching recipe logic.

> NOTE: This document predates the corrected ownership model. Adapter
> implementations now live in `crucible_kitchen`; TinkexCookbook provides
> recipes/config only. The examples below are historical.

This repo uses a lightweight Ports & Adapters design:

- **Ports** define the cookbook-facing interfaces (behaviours).
- **Adapters** implement those ports for specific clients/libraries.
- **Ports.new/1** wires everything in one place.

---

## Ports (Behaviours)

| Port | Responsibility |
|---|---|
| `Ports.VectorStore` | Vector DB operations (ChromaDB) |
| `Ports.EmbeddingClient` | Embedding generation |
| `Ports.LLMClient` | Chat/completion inference |
| `Ports.DatasetStore` | Dataset loading + basic ops |
| `Ports.HubClient` | HuggingFace Hub downloads |
| `Ports.BlobStore` | File/blob access |

---

## Adapter Wiring

Adapters are resolved via `TinkexCookbook.Ports`:

```elixir
ports =
  TinkexCookbook.Ports.new(
    ports: [
      vector_store: {YourApp.Adapters.VectorStore.Chroma, []},
      dataset_store: {CrucibleKitchen.Adapters.HfDatasets.DatasetStore, []}
    ]
  )
```

You can also set app-level defaults:

```elixir
config :tinkex_cookbook, TinkexCookbook.Ports,
  vector_store: {YourApp.Adapters.VectorStore.Chroma, []},
  dataset_store: {CrucibleKitchen.Adapters.HfDatasets.DatasetStore, []}
```

---

## Usage Pattern

Recipes should **never** call external clients directly. Use ports:

```elixir
{:ok, collection} =
  TinkexCookbook.Ports.VectorStore.get_or_create_collection(ports, "docs", %{})

{:ok, dataset} =
  TinkexCookbook.Ports.DatasetStore.load_dataset(ports, "allenai/tulu-3-sft")

{:ok, response} =
  TinkexCookbook.Ports.LLMClient.chat(ports, [%{role: "user", content: "Summarize this"}],
    output_schema: %{"type" => "object", "properties" => %{"summary" => %{"type" => "string"}}}
  )
```

---

## Current Adapters

| Port | Adapter | Status |
|---|---|---|
| `VectorStore` | `Adapters.VectorStore.Chroma` | Ready |
| `DatasetStore` | `Adapters.DatasetStore.HfDatasets` | Ready |
| `HubClient` | `Adapters.HubClient.HfHub` | Ready |
| `BlobStore` | `Adapters.BlobStore.Local` | Ready |
| `EmbeddingClient` | `Adapters.EmbeddingClient.Noop` | Placeholder |
| `LLMClient` | `Adapters.LLMClient.Noop` | Placeholder |
| `LLMClient` | `Adapters.LLMClient.Codex` | Ready |
| `LLMClient` | `Adapters.LLMClient.ClaudeAgent` | Ready |

---

## Testing

Ports are mocked via Mox. Example:

```elixir
TinkexCookbook.Ports.VectorStoreMock
|> expect(:get_or_create_collection, fn _opts, "docs", %{} -> {:ok, :collection} end)
```

This keeps tests deterministic and avoids network calls.

---

## Guidelines

- Keep port interfaces minimal and cookbook-centric.
- Adapters normalize errors to `Ports.Error`.
- Recipes only depend on ports, not on client-specific modules.
- LLM structured output:
  - Codex adapter uses `output_schema` (passed to `Codex.Thread.run/3`).
  - Claude adapter maps `output_schema` to `Options.output_format`.
