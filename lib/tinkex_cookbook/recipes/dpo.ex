defmodule TinkexCookbook.Recipes.DPO do
  @moduledoc """
  Direct Preference Optimization (DPO) recipe.

  Implements DPO training for aligning language models with human preferences.
  Uses CrucibleKitchen's Preference workflow with Tinkex backend.

  ## DPO Overview

  DPO optimizes the model to prefer "chosen" responses over "rejected" responses:

      L = -log(sigmoid(β * (log π(y_w|x) - log π(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x))))

  Where:
  - β (beta) controls the strength of the preference constraint
  - π is the policy being trained
  - π_ref is the frozen reference model (usually the initial model)
  - y_w is the chosen (winning) response
  - y_l is the rejected (losing) response

  ## Supported Datasets

  - `:hhh` - HHH (Helpful, Harmless, Honest) dataset
  - `:ultrafeedback` - UltraFeedback dataset
  - `:helpsteer3` - HelpSteer3 dataset

  ## Usage

      # Run with defaults
      TinkexCookbook.Recipes.DPO.run(%{
        model: "meta-llama/Llama-3.2-1B",
        dataset: :hhh
      })

      # Custom configuration
      TinkexCookbook.Recipes.DPO.run(%{
        model: "meta-llama/Llama-3.1-8B",
        dataset: :ultrafeedback,
        dpo_beta: 0.05,
        learning_rate: 5.0e-6,
        batch_size: 128
      })

  ## Results

  After training completes, you can access:
  - `result.context.state.dpo_metrics` - DPO training metrics
  - `result.context.state.eval_results` - Evaluation results (if enabled)
  - `result.context.state.registered_model` - Registered model record

  ## References

  - Rafailov et al., "Direct Preference Optimization" (2023)
  """

  use CrucibleKitchen.Recipe

  alias CrucibleKitchen.Adapters.HfDatasets.DatasetStore, as: HfDatasetsAdapter
  alias CrucibleKitchen.Adapters.Noop.Evaluator, as: NoopEvaluator
  alias CrucibleKitchen.Adapters.Noop.ModelRegistry, as: NoopModelRegistry
  alias CrucibleKitchen.Adapters.Tinkex.SamplingClient, as: TinkexSamplingAdapter
  alias CrucibleKitchen.Adapters.Tinkex.TrainingClient, as: TinkexAdapter
  alias CrucibleKitchen.Workflows.Preference, as: PreferenceWorkflow

  require Logger

  @impl true
  def name, do: :dpo

  @impl true
  def description do
    "Direct Preference Optimization training with Tinkex backend"
  end

  @impl true
  def default_config do
    %{
      # Model config
      model: "meta-llama/Llama-3.2-1B",
      reference_model: nil,
      lora_rank: 32,

      # Dataset config
      dataset: :hhh,
      split: "train",

      # DPO-specific config
      dpo_beta: 0.1,

      # Training config
      epochs: 1,
      batch_size: 256,
      learning_rate: 1.0e-5,
      lr_schedule: :linear,
      max_length: 8192,

      # Checkpoint config
      save_every: 100,
      eval_every: 0,

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
    PreferenceWorkflow.__workflow__()
  end

  @impl true
  def validate_config(config) do
    with :ok <- validate_model(config),
         :ok <- validate_dataset(config) do
      validate_training_params(config)
    end
  end

  defp validate_model(config) do
    if is_nil(config[:model]) or config[:model] == "" do
      {:error, "model is required"}
    else
      :ok
    end
  end

  defp validate_dataset(config) do
    cond do
      is_nil(config[:dataset]) ->
        {:error, "dataset is required"}

      config[:dataset] not in [:hhh, :ultrafeedback, :helpsteer3] and
          not is_binary(config[:dataset]) ->
        {:error, "dataset must be :hhh, :ultrafeedback, :helpsteer3, or a path"}

      true ->
        :ok
    end
  end

  defp validate_training_params(config) do
    cond do
      config[:epochs] < 1 -> {:error, "epochs must be >= 1"}
      config[:batch_size] < 1 -> {:error, "batch_size must be >= 1"}
      config[:dpo_beta] <= 0 -> {:error, "dpo_beta must be positive"}
      true -> :ok
    end
  end

  # ===========================================================================
  # Convenience Functions
  # ===========================================================================

  @doc """
  Run the DPO recipe with default Tinkex adapters.

  ## Options

  - `:api_key` - Override TINKER_API_KEY
  - `:base_url` - Override TINKER_BASE_URL

  ## Example

      TinkexCookbook.Recipes.DPO.run(%{
        model: "meta-llama/Llama-3.2-1B",
        dataset: :hhh,
        dpo_beta: 0.1
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

    client_opts =
      []
      |> maybe_add(:api_key, api_key)
      |> maybe_add(:base_url, base_url)

    %{
      training_client: {TinkexAdapter, client_opts},
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
  CLI entry point for mix kitchen.run :dpo.
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
        Logger.info("DPO training completed successfully")
        Logger.info("Final step: #{result.context.state[:global_step]}")

        if dpo_metrics = result.context.state[:dpo_metrics] do
          Logger.info("DPO metrics:")
          Logger.info("  Loss: #{Map.get(dpo_metrics, :loss, "N/A")}")
          Logger.info("  Accuracy: #{Map.get(dpo_metrics, :accuracy, "N/A")}")
          Logger.info("  Margin: #{Map.get(dpo_metrics, :margin, "N/A")}")
        end

        if model = result.context.state[:registered_model] do
          Logger.info("Model registered: #{model.name} v#{model.version}")
        end

        :ok

      {:error, reason} ->
        Logger.error("DPO training failed: #{inspect(reason)}")
        {:error, reason}
    end
  end

  defp parse_value(value) do
    cond do
      value =~ ~r/^\d+$/ -> String.to_integer(value)
      value =~ ~r/^\d+\.\d+$/ -> String.to_float(value)
      value =~ ~r/^\d+e-?\d+$/i -> String.to_float(value)
      value =~ ~r/^\d+\.\d+e-?\d+$/i -> String.to_float(value)
      value == "true" -> true
      value == "false" -> false
      true -> String.to_atom(value)
    end
  end
end
