defmodule TinkexCookbook.Recipes.ChatSl do
  @moduledoc """
  Chat supervised learning recipe.

  Trains a model on chat-formatted datasets such as NoRobots or Tulu3,
  using CrucibleKitchen's supervised workflow with the Tinkex backend.
  """

  use CrucibleKitchen.Recipe

  alias CrucibleKitchen.Adapters.HfDatasets.DatasetStore, as: HfDatasetsAdapter
  alias CrucibleKitchen.Adapters.Noop.Evaluator, as: NoopEvaluator
  alias CrucibleKitchen.Adapters.Noop.ModelRegistry, as: NoopModelRegistry
  alias CrucibleKitchen.Adapters.Tinkex.TrainingClient, as: TinkexTrainingAdapter
  alias CrucibleKitchen.Workflows.Supervised, as: SupervisedWorkflow

  require Logger

  @impl true
  def name, do: :chat_sl

  @impl true
  def description do
    "Chat supervised learning with CrucibleKitchen and Tinkex backend"
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
      train_on: :all_assistant_messages,

      # Training config
      epochs: 1,
      batch_size: 256,
      learning_rate: 1.0e-4,
      lr_schedule: :linear,
      max_length: 16_384,

      # Checkpoint config
      save_every: 20,
      eval_every: 20,

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
    [:blob_store, :metrics_store, :hub_client, :evaluator, :model_registry]
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
  Run the chat_sl recipe with default Tinkex adapters.

  ## Options

  - `:api_key` - Override TINKER_API_KEY
  - `:base_url` - Override TINKER_BASE_URL
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
      training_client: {TinkexTrainingAdapter, training_opts},
      dataset_store: {HfDatasetsAdapter, []},
      evaluator: {NoopEvaluator, []},
      model_registry: {NoopModelRegistry, []}
    }
  end

  defp maybe_add(opts, _key, nil), do: opts
  defp maybe_add(opts, key, value), do: Keyword.put(opts, key, value)

  # ===========================================================================
  # CLI Entry Point
  # ===========================================================================

  @doc """
  CLI entry point for mix kitchen.run :chat_sl.
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
        Logger.info("chat_sl training completed successfully")

        if eval_results = result.context.state[:eval_results] do
          Logger.info("Evaluation results:")
          Logger.info("  Accuracy: #{Map.get(eval_results, :accuracy, "N/A")}")
        end

        :ok

      {:error, reason} ->
        Logger.error("chat_sl training failed: #{inspect(reason)}")
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
