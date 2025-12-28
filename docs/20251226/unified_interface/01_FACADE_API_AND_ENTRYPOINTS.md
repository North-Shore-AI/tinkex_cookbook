# Facade API and Entrypoints

Date: 2025-12-26
Status: Draft
Owner: North-Shore-AI

## 1) Purpose

Define the unified facade API that recipes and CLI entrypoints use. This is the single stable surface for running cookbook workflows.

## 2) Facade Module

Module name (recommended):
- `TinkexCookbook.Runtime`

### Core Functions

```elixir
@spec build_spec(module(), map() | keyword()) :: CrucibleIR.Experiment.t()
@spec run(module(), map() | keyword()) :: {:ok, Crucible.Context.t()} | {:error, term()}
@spec run_spec(CrucibleIR.Experiment.t(), keyword()) :: {:ok, Crucible.Context.t()} | {:error, term()}
@spec build_ports(keyword()) :: TinkexCookbook.Ports.t()
@spec train(module(), map() | keyword()) :: {:ok, Crucible.Context.t()} | {:error, term()}
@spec eval(module(), map() | keyword()) :: {:ok, map()} | {:error, term()}
```

### Semantics

- `build_spec/2` calls the recipe module and returns a `CrucibleIR.Experiment`.
- `run/2` builds the spec and runs it through `CrucibleFramework.run/2`.
- `run_spec/2` is a low level entrypoint for external callers.
- `build_ports/1` resolves ports via `TinkexCookbook.Ports` using manifests and overrides.
- `train/2` is a convenience wrapper for train only pipelines.
- `eval/2` optionally runs EvalEx or other evaluation steps.

## 3) Recipe Behaviour

Each recipe implements a minimal behaviour so the facade can call it.

```elixir
defmodule TinkexCookbook.Recipe do
  @callback name() :: String.t()
  @callback description() :: String.t()
  @callback config_schema() :: module()
  @callback build_spec(config :: struct() | map()) :: CrucibleIR.Experiment.t()
  @callback default_config() :: map()
end
```

## 4) CLI Entrypoints

The CLI should use ChzEx and route into the facade.

```elixir
TinkexCookbook.CLI.run(recipe_module, argv)
  -> config = ChzEx entrypoint
  -> spec = TinkexCookbook.Runtime.build_spec(recipe_module, config)
  -> TinkexCookbook.Runtime.run_spec(spec, cli_opts)
```

### Log Path Handling

Match Python behavior:
- Use `cli_utils.check_log_dir` semantics (ask/abort/overwrite).
- Derive a run name if `log_path` is not provided.

## 5) Config Model

- All recipes use `ChzEx.Schema` for configs.
- `default_config/0` returns a map of the same fields as Python defaults.
- A `Blueprint` helper should be provided for common patterns (see `tinker_cookbook/recipes/sl_basic.py`).

## 6) Manifest Selection (Minimal)

- `run/2` and `run_spec/2` accept `manifest: "local" | "dev" | "prod" | "test"`.
- Manifest lookup is string-keyed (no atom creation).
- Port resolution merges: defaults < manifest < explicit `ports` override.

## 7) Errors and Result Shape

- Facade returns `{:error, reason}` for all errors.
- No exceptions should escape the facade in standard usage.
- For `run/2`, the success return is `{:ok, %Crucible.Context{}}`.

## 8) Integration Points

- The facade injects `ports` into stage options and context assigns.
- `run_spec/2` can accept `assigns` and `run_id` for external orchestration.
