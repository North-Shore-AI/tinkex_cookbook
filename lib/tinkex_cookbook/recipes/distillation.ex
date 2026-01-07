defmodule TinkexCookbook.Recipes.Distillation do
  @moduledoc """
  Knowledge distillation recipe.

  Runs a teacher sampler to generate responses and trains a student model
  using CrucibleKitchen's distillation workflow.
  """

  use CrucibleKitchen.Recipe

  alias CrucibleKitchen.Adapters.HfDatasets.DatasetStore, as: HfDatasetsAdapter
  alias CrucibleKitchen.Adapters.Noop.Evaluator, as: NoopEvaluator
  alias CrucibleKitchen.Adapters.Noop.ModelRegistry, as: NoopModelRegistry
  alias CrucibleKitchen.Adapters.Tinkex.SamplingClient, as: TinkexSamplingAdapter
  alias CrucibleKitchen.Adapters.Tinkex.TrainingClient, as: TinkexTrainingAdapter
  alias CrucibleKitchen.Workflows.Distillation, as: DistillationWorkflow

  require Logger

  @impl true
  def name, do: :distillation

  @impl true
  def description do
    "Knowledge distillation with teacher sampling via CrucibleKitchen"
  end

  @impl true
  def default_config do
    %{
      # Student model config
      model: "Qwen/Qwen3-8B-Base",
      lora_rank: 128,

      # Teacher config
      teacher_model: "Qwen/Qwen3-8B",
      teacher_checkpoint_path: nil,

      # Dataset config
      dataset: :deepmath,
      split: "train",

      # Training config
      epochs: 1,
      batch_size: 1024,
      learning_rate: 1.0e-4,
      lr_schedule: :linear,
      max_length: 8192,

      # Teacher sampling
      max_tokens: 4096,
      temperature: 1.0,

      # Distillation loss
      distill_loss_fn: :cross_entropy,
      distill_loss_fn_config: nil,

      # Checkpoint config
      save_every: 20,
      eval_every: 20,

      # Logging config
      log_every: 10
    }
  end

  @impl true
  def required_adapters do
    [:training_client, :sampling_client, :dataset_store]
  end

  @impl true
  def optional_adapters do
    [:blob_store, :metrics_store, :hub_client, :evaluator, :model_registry]
  end

  @impl true
  def workflow do
    DistillationWorkflow.__workflow__()
  end

  @impl true
  def validate_config(config) do
    cond do
      is_nil(config[:model]) or config[:model] == "" ->
        {:error, "model is required"}

      is_nil(config[:teacher_model]) or config[:teacher_model] == "" ->
        {:error, "teacher_model is required"}

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
  Run the distillation recipe with default Tinkex adapters.
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

    client_opts =
      []
      |> maybe_add(:api_key, api_key)
      |> maybe_add(:base_url, base_url)

    %{
      training_client: {TinkexTrainingAdapter, client_opts},
      sampling_client: {TinkexSamplingAdapter, client_opts},
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
  CLI entry point for mix kitchen.run :distillation.
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
        Logger.info("Distillation training completed successfully")

        if metrics = result.context.state[:distillation_metrics] do
          Logger.info("Distillation metrics:")
          Logger.info("  Loss: #{Map.get(metrics, :loss, "N/A")}")
        end

        :ok

      {:error, reason} ->
        Logger.error("Distillation training failed: #{inspect(reason)}")
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
