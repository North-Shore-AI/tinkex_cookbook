# Tinkex Cookbook Developer Guide

This guide captures the core architecture, dependency strategy, and development
rules for the Elixir port of tinker-cookbook. It is written for contributors
who need to extend recipes, adapters, or core data types.

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

Core types:
- `lib/tinkex_cookbook/types/*` (TensorData, ModelInput, Datum)
- `lib/tinkex_cookbook/types.ex` (type aliases)

Renderers:
- `lib/tinkex_cookbook/renderers/*` (TrainOnWhat, Renderer behavior, types)

Supervised:
- `lib/tinkex_cookbook/supervised/*` (configs, dataset behavior, helpers)

Recipes:
- `lib/tinkex_cookbook/recipes/*` (entrypoints and orchestration)

Ports and adapters:
- `lib/tinkex_cookbook/ports.ex`
- `lib/tinkex_cookbook/ports/*`
- `lib/tinkex_cookbook/adapters/*`

---

## 3) Ports and Adapters (Core Architecture)

**Ports** are the cookbook-facing interfaces (behaviours).
**Adapters** implement those ports for specific clients/libraries.

Ports:
- `Ports.VectorStore` (vector DB ops)
- `Ports.EmbeddingClient` (embedding generation)
- `Ports.LLMClient` (chat/completion inference)
- `Ports.DatasetStore` (dataset loading + ops)
- `Ports.HubClient` (HuggingFace Hub)
- `Ports.BlobStore` (local or cloud file access)

Adapters:
- VectorStore: `Adapters.VectorStore.Chroma`, `Adapters.VectorStore.Noop`
- DatasetStore: `Adapters.DatasetStore.HfDatasets`, `Adapters.DatasetStore.Noop`
- HubClient: `Adapters.HubClient.HfHub`, `Adapters.HubClient.Noop`
- BlobStore: `Adapters.BlobStore.Local`, `Adapters.BlobStore.Noop`
- LLMClient: `Adapters.LLMClient.Codex`, `Adapters.LLMClient.ClaudeAgent`,
  `Adapters.LLMClient.Noop`
- EmbeddingClient: `Adapters.EmbeddingClient.Noop`

Composition root:
```elixir
ports =
  TinkexCookbook.Ports.new(
    ports: [
      vector_store: {TinkexCookbook.Adapters.VectorStore.Chroma, []},
      llm_client: {TinkexCookbook.Adapters.LLMClient.Codex, []}
    ]
  )
```

App-level config:
```elixir
config :tinkex_cookbook, TinkexCookbook.Ports,
  vector_store: {TinkexCookbook.Adapters.VectorStore.Chroma, []},
  llm_client: {TinkexCookbook.Adapters.LLMClient.ClaudeAgent, []}
```

**Rule:** recipes call ports only, never external clients directly.

---

## 4) LLM Adapters

### 4.1 Codex Adapter (`Adapters.LLMClient.Codex`)

Uses the Codex CLI via `codex_sdk` to run a turn on a thread.

Inputs:
- `messages` list (chat-style, `%{role, content}`)
- `output_schema` (map) for structured output
- Optional `codex_opts`, `thread_opts`, `turn_opts`

Structured output:
- `output_schema` is passed to `Codex.Thread.run/3`
- Response includes `structured_output` if the model produced valid JSON

CLI requirements:
- `codex` executable installed
- Auth via `CODEX_API_KEY` or CLI login

### 4.2 Claude Adapter (`Adapters.LLMClient.ClaudeAgent`)

Uses the Claude Code CLI via `claude_agent_sdk` and streams messages.

Inputs:
- `messages` list (chat-style, `%{role, content}`)
- `output_schema` (map), mapped to `Options.output_format`
- Optional `options` (ClaudeAgentSDK.Options)

Structured output:
- `output_schema` is mapped to `{:json_schema, schema}`
- Response includes `structured_output` from result frames

CLI requirements:
- `claude` / `claude-code` executable installed
- Auth via OAuth token or CLI login

---

## 5) Dataset + Hub Stack

- Datasets use `hf_datasets_ex`.
- Hub access uses `hf_hub`.
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
1. Add a port behaviour if needed.
2. Implement adapter under `lib/tinkex_cookbook/adapters`.
3. Normalize errors to `TinkexCookbook.Ports.Error`.
4. Add tests with Mox or stub modules.
5. Update `docs/20251223/PORTS_AND_ADAPTERS.md`.

---

## 10) Adding a New Recipe

Checklist:
1. Define config with ChzEx.
2. Use ports for all external calls.
3. Keep orchestration in recipe; move logic to shared modules.
4. Write tests for dataset, renderer, and CLI behavior.

---

## 11) Reference Docs

- `docs/20251223/PORTS_AND_ADAPTERS.md`
- `docs/20251223/PYTHON_TO_ELIXIR_LIBRARY_MAPPING.md`
- `docs/20251221/COOKBOOK_CORE_FOUNDATION.md`
