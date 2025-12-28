# CLI, Config, and Workspace Layout

Date: 2025-12-26
Status: Draft
Owner: North-Shore-AI

## 1) Purpose

Define how recipe configs, CLI entrypoints, and workspace layout are structured for the unified facade rewrite.

## 2) Config Conventions

- All recipe configs use `ChzEx.Schema`.
- Config defaults mirror Python `chz` defaults.
- CLI entrypoints use `ChzEx.entrypoint/2` or `ChzEx.Blueprint`.
  - Facade options (manifest, ports overrides) are handled outside recipe schemas.

### Example (sl_basic)

```elixir
use ChzEx.Schema

chz_schema "sl_basic" do
  field :model_name, :string, default: "meta-llama/Llama-3.1-8B"
  field :log_path, :string
  field :learning_rate, :float, default: 2.0e-4
  field :num_epochs, :integer, default: 1
end
```

## 3) Log Path Handling

Follow Python behavior from `cli_utils.check_log_dir`:
- `ask` default: prompt before overwrite.
- `overwrite`: delete and recreate.
- `abort`: fail if exists.

The facade should centralize this logic.

## 4) Run Naming

When `log_path` is not provided, derive a run name that mirrors Python:

```
{dataset}-{model}-{lr}lr-{batch}batch-{timestamp}
```

## 5) Manifest Selection

- CLI should accept `--manifest` to select a port manifest by name.
- Manifests are string-keyed to avoid atom creation from user input.
- Facade merges defaults < manifest < explicit `ports` overrides.

## 6) Workspace Layout

Target layout in `tinkex_cookbook`:

```
lib/tinkex_cookbook/
  runtime/           # facade implementation
  recipes/           # recipe modules and configs
  adapters/          # Tinker-specific adapters
  ports/             # adapter resolver
  eval/              # inspect-ai parity wiring
  datasets/          # thin dataset helpers
  utils/             # CLI and log helpers only
```

## 7) Environment Variables

- `TINKER_API_KEY` and `TINKER_BASE_URL` for Tinker API.
- `HF_TOKEN` for gated datasets.
- `TINKEX_HTTP_PROTOCOL` for transport control in parity runs.

## 8) Acceptance Criteria

- All recipes use ChzEx configs.
- CLI entrypoints call the facade only.
- Log path handling matches Python behavior.
