defmodule TinkexCookbook.Runtime do
  @moduledoc """
  Unified facade for TinkexCookbook recipe execution.

  This is the primary entrypoint for running recipes. All recipes
  delegate to this module, which handles port resolution,
  experiment building, and execution via CrucibleFramework.

  ## Usage

      # Run a recipe with default manifest
      TinkexCookbook.Runtime.run(TinkexCookbook.Recipes.SlBasic, config)

      # Run with a specific manifest
      TinkexCookbook.Runtime.run(TinkexCookbook.Recipes.SlBasic, config, manifest: :prod)

      # Run with port overrides
      TinkexCookbook.Runtime.run(TinkexCookbook.Recipes.SlBasic, config,
        ports: %{"training_client" => MyCustomAdapter}
      )
  """

  alias TinkexCookbook.Runtime.Manifests

  @type manifest_name :: :default | :local | :dev | :prod | :test | atom()
  @type run_opts :: [
          manifest: manifest_name(),
          ports: map(),
          dry_run: boolean()
        ]

  @doc """
  Run a recipe with the given config.

  This is the primary entrypoint. It:
  1. Resolves ports via manifest + overrides
  2. Builds the experiment spec via recipe.build_spec/1
  3. Executes via CrucibleFramework.run/2

  ## Options

  - `:manifest` - Named manifest for port wiring (default: `:default`)
  - `:ports` - Map of port overrides
  - `:dry_run` - If true, builds spec but doesn't execute
  """
  @spec run(module(), map() | struct(), run_opts()) ::
          {:ok, map()} | {:error, term()}
  def run(recipe_module, config, opts \\ []) do
    manifest = Keyword.get(opts, :manifest, :default)
    port_overrides = Keyword.get(opts, :ports, %{})
    dry_run = Keyword.get(opts, :dry_run, false)

    with {:ok, ports} <- build_ports(manifest, port_overrides),
         experiment <- recipe_module.build_spec(config) do
      if dry_run do
        {:ok, %{experiment: experiment, ports: ports}}
      else
        CrucibleFramework.run(experiment, ports: ports)
      end
    end
  end

  @doc """
  Low-level entrypoint for external orchestration.

  Accepts a pre-built CrucibleIR.Experiment and runs it.
  """
  @spec run_spec(CrucibleIR.Experiment.t(), run_opts()) ::
          {:ok, map()} | {:error, term()}
  def run_spec(experiment, opts \\ []) do
    manifest = Keyword.get(opts, :manifest, :default)
    port_overrides = Keyword.get(opts, :ports, %{})

    with {:ok, ports} <- build_ports(manifest, port_overrides) do
      CrucibleFramework.run(experiment, ports: ports)
    end
  end

  @doc """
  Build a CrucibleIR.Experiment from a recipe and config.

  The recipe must implement the Recipe behaviour.
  """
  @spec build_spec(module(), map() | struct()) :: CrucibleIR.Experiment.t()
  def build_spec(recipe_module, config) do
    recipe_module.build_spec(config)
  end

  @doc """
  Build ports map from manifest and overrides.

  Merges defaults < manifest < overrides.
  """
  @spec build_ports(manifest_name(), map()) :: {:ok, map()}
  def build_ports(manifest_name, overrides \\ %{}) do
    ports =
      Manifests.defaults()
      |> Map.merge(Manifests.get(manifest_name))
      |> Map.merge(normalize_overrides(overrides))

    {:ok, ports}
  end

  @doc """
  Convenience wrapper for training-only pipelines.
  """
  @spec train(module(), map(), run_opts()) ::
          {:ok, map()} | {:error, term()}
  def train(recipe_module, config, opts \\ []) do
    run(recipe_module, config, opts)
  end

  @doc """
  Convenience wrapper for evaluation pipelines.
  """
  @spec eval(module(), map(), run_opts()) ::
          {:ok, map()} | {:error, term()}
  def eval(recipe_module, config, opts \\ []) do
    run(recipe_module, config, opts)
  end

  # Normalize string keys to match manifests
  defp normalize_overrides(overrides) when is_map(overrides) do
    Map.new(overrides, fn
      {k, v} when is_atom(k) -> {Atom.to_string(k), v}
      {k, v} -> {k, v}
    end)
  end
end
