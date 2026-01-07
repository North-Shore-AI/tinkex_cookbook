defmodule TinkexCookbook.Recipes.MathRl do
  @moduledoc """
  Math reasoning via Reinforcement Learning.

  Uses GSM8K-style problems with \\boxed{} answers and Snakebridge math_verify
  for reward computation. Orchestrated by CrucibleKitchen's RL workflow.
  """

  use CrucibleKitchen.Recipe

  alias CrucibleKitchen.Adapters.HfDatasets.DatasetStore, as: HfDatasetsAdapter
  alias CrucibleKitchen.Adapters.Noop.Evaluator, as: NoopEvaluator
  alias CrucibleKitchen.Adapters.Noop.ModelRegistry, as: NoopModelRegistry
  alias CrucibleKitchen.Adapters.Tinkex.SamplingClient, as: TinkexSamplingAdapter
  alias CrucibleKitchen.Adapters.Tinkex.TrainingClient, as: TinkexTrainingAdapter
  alias CrucibleKitchen.RL.MathEnvBuilder
  alias CrucibleKitchen.Workflows.Reinforcement, as: ReinforcementWorkflow

  require Logger

  @impl true
  def name, do: :math_rl

  @impl true
  def description do
    "Math reasoning via Reinforcement Learning with GSM8K environment"
  end

  @impl true
  def default_config do
    %{
      # Model config
      model: "meta-llama/Llama-3.1-8B-Instruct",
      lora_rank: 32,

      # Dataset / environment config
      dataset: :gsm8k,
      split: "train",
      env: :gsm8k,
      env_group_builder: &MathEnvBuilder.build/1,
      convo_prefix: :standard,
      seed: 0,
      format_coef: 0.1,
      math_verify_module: CrucibleKitchen.Adapters.Snakebridge.MathVerify,
      math_verify_timeout_ms: 1000,

      # RL config
      group_size: 4,
      groups_per_batch: 100,
      num_rollouts: 100,
      max_tokens: 5,
      temperature: 1.0,

      # PPO config
      gamma: 0.99,
      gae_lambda: 0.95,
      clip_epsilon: 0.2,
      ppo_epochs: 4,
      kl_penalty_coef: 0.0,

      # Training config
      learning_rate: 1.0e-5,
      lr_schedule: :linear,

      # Checkpoint config
      save_every: 20,
      eval_every: 20,

      # Logging config
      log_every: 1
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
    ReinforcementWorkflow.__workflow__()
  end

  @impl true
  def validate_config(config) do
    with :ok <- validate_model(config),
         :ok <- validate_rl_params(config) do
      validate_ppo_params(config)
    end
  end

  defp validate_model(config) do
    if is_nil(config[:model]) or config[:model] == "" do
      {:error, "model is required"}
    else
      :ok
    end
  end

  defp validate_rl_params(config) do
    cond do
      config[:group_size] < 1 -> {:error, "group_size must be >= 1"}
      config[:groups_per_batch] < 1 -> {:error, "groups_per_batch must be >= 1"}
      config[:num_rollouts] < 1 -> {:error, "num_rollouts must be >= 1"}
      true -> :ok
    end
  end

  defp validate_ppo_params(config) do
    cond do
      config[:gamma] <= 0 or config[:gamma] > 1 -> {:error, "gamma must be in (0, 1]"}
      config[:clip_epsilon] <= 0 -> {:error, "clip_epsilon must be positive"}
      config[:ppo_epochs] < 1 -> {:error, "ppo_epochs must be >= 1"}
      true -> :ok
    end
  end

  # ===========================================================================
  # Convenience Functions
  # ===========================================================================

  @doc """
  Run the MathRL recipe with default Tinkex adapters.
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
  CLI entry point for mix kitchen.run :math_rl.
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
        Logger.info("MathRL training completed successfully")

        if rollout_metrics = result.context.state[:rollout_metrics] do
          Logger.info("Rollout metrics:")
          Logger.info("  Reward mean: #{Map.get(rollout_metrics, :reward_mean, "N/A")}")
        end

        :ok

      {:error, reason} ->
        Logger.error("MathRL training failed: #{inspect(reason)}")
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
