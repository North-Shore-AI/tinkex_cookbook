defmodule TinkexCookbook.Recipes.SlBasicV2 do
  @moduledoc """
  Supervised Learning Basic recipe - V2 using CrucibleKitchen.

  This is the refactored version that uses CrucibleKitchen as the core
  orchestration engine. It provides the same functionality as sl_basic
  but with the benefits of the crucible_kitchen architecture:

  - Hexagonal architecture with pluggable adapters
  - Built-in telemetry and metrics
  - Standardized workflow stages
  - Better testing and composability

  ## Usage

      # Using CrucibleKitchen.run
      adapters = TinkexCookbook.Recipes.SlBasicV2.default_adapters()
      CrucibleKitchen.run(TinkexCookbook.Recipes.SlBasicV2, %{
        model: "meta-llama/Llama-3.1-8B",
        epochs: 1
      }, adapters: adapters)

      # Using the convenience function
      TinkexCookbook.Recipes.SlBasicV2.run(%{
        model: "meta-llama/Llama-3.1-8B",
        dataset: :no_robots
      })
  """

  use CrucibleKitchen.Recipe

  alias CrucibleKitchen.Adapters.HfDatasets.DatasetStore, as: HfDatasetsAdapter
  alias CrucibleKitchen.Adapters.Tinkex.TrainingClient, as: TinkexAdapter
  alias CrucibleKitchen.Workflows.Supervised, as: SupervisedWorkflow

  require Logger

  @impl true
  def name, do: :sl_basic_v2

  @impl true
  def description do
    "Supervised fine-tuning using CrucibleKitchen with Tinkex backend"
  end

  @impl true
  def default_config do
    %{
      # Model config
      model: "meta-llama/Llama-3.1-8B",
      lora_rank: 32,

      # Dataset config
      dataset: :no_robots,
      split: "train",

      # Training config
      epochs: 1,
      batch_size: 128,
      learning_rate: 2.0e-4,
      lr_schedule: :linear,
      max_length: 32_768,

      # Checkpoint config
      save_every: 20,
      eval_every: 0,

      # Logging config
      log_every: 10
    }
  end

  @impl true
  def required_adapters do
    [:training_client, :dataset_store]
  end

  @impl true
  def optional_adapters do
    [:blob_store, :metrics_store, :hub_client]
  end

  @impl true
  def workflow do
    SupervisedWorkflow.__workflow__()
  end

  @impl true
  def validate_config(config) do
    cond do
      is_nil(config[:model]) or config[:model] == "" ->
        {:error, "model is required"}

      is_nil(config[:dataset]) ->
        {:error, "dataset is required"}

      config[:epochs] < 1 ->
        {:error, "epochs must be >= 1"}

      config[:batch_size] < 1 ->
        {:error, "batch_size must be >= 1"}

      true ->
        :ok
    end
  end

  # ===========================================================================
  # Convenience Functions
  # ===========================================================================

  @doc """
  Run the recipe with default Tinkex adapters.

  ## Options

  - `:api_key` - Override TINKER_API_KEY
  - `:base_url` - Override TINKER_BASE_URL

  ## Example

      TinkexCookbook.Recipes.SlBasicV2.run(%{
        model: "meta-llama/Llama-3.1-8B",
        epochs: 2
      })
  """
  @spec run(map(), keyword()) :: {:ok, map()} | {:error, term()}
  def run(config, opts \\ []) do
    adapters = build_adapters(opts)
    CrucibleKitchen.run(__MODULE__, config, adapters: adapters)
  end

  @doc """
  Get the default adapter configuration for Tinkex backend.
  """
  @spec default_adapters(keyword()) :: map()
  def default_adapters(opts \\ []) do
    build_adapters(opts)
  end

  defp build_adapters(opts) do
    api_key = Keyword.get(opts, :api_key)
    base_url = Keyword.get(opts, :base_url)

    training_opts =
      []
      |> maybe_add(:api_key, api_key)
      |> maybe_add(:base_url, base_url)

    %{
      training_client: {TinkexAdapter, training_opts},
      dataset_store: {HfDatasetsAdapter, []}
    }
  end

  defp maybe_add(opts, _key, nil), do: opts
  defp maybe_add(opts, key, value), do: Keyword.put(opts, key, value)

  # ===========================================================================
  # Legacy Compatibility
  # ===========================================================================

  @doc """
  Backward compatible entry point for CLI execution.

  Delegates to run/2 with parsed arguments.
  """
  @spec main(list(String.t())) :: :ok | {:error, term()}
  def main(argv \\ System.argv()) do
    config =
      argv
      |> Enum.filter(&String.contains?(&1, "="))
      |> Enum.map(fn arg ->
        [key, value] = String.split(arg, "=", parts: 2)
        {String.to_atom(key), parse_value(value)}
      end)
      |> Map.new()

    case run(config) do
      {:ok, result} ->
        Logger.info("Training completed successfully")
        Logger.info("Final step: #{result.state[:global_step]}")
        :ok

      {:error, reason} ->
        Logger.error("Training failed: #{inspect(reason)}")
        {:error, reason}
    end
  end

  defp parse_value(value) do
    cond do
      value =~ ~r/^\d+$/ -> String.to_integer(value)
      value =~ ~r/^\d+\.\d+$/ -> String.to_float(value)
      value =~ ~r/^\d+e-?\d+$/i -> String.to_float(value)
      value =~ ~r/^\d+\.\d+e-?\d+$/i -> String.to_float(value)
      true -> value
    end
  end
end
