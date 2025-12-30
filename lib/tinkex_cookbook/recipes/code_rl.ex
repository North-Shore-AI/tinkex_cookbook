defmodule TinkexCookbook.Recipes.CodeRL do
  @moduledoc """
  Code Generation via Reinforcement Learning recipe.

  Trains language models to generate correct code using RL with execution-based
  rewards. Uses CrucibleKitchen's Reinforcement workflow with Tinkex backend.

  ## Overview

  This recipe implements GRPO-style RL training for code generation:

  1. **Rollout**: Generate code solutions for programming problems
  2. **Reward**: Execute code and score based on test case pass rate
  3. **Advantage**: Compute advantages using GAE
  4. **Update**: PPO-style policy gradient with clipping

  ## Supported Environments

  - `:deepcoder` - DeepCoder-style programming problems
  - `:humaneval` - HumanEval benchmark problems (evaluation only)

  ## Usage

      # Run with defaults
      TinkexCookbook.Recipes.CodeRL.run(%{
        model: "meta-llama/Llama-3.1-8B-Instruct"
      })

      # Custom configuration
      TinkexCookbook.Recipes.CodeRL.run(%{
        model: "Qwen/Qwen2.5-Coder-7B",
        env: :deepcoder,
        group_size: 8,
        groups_per_batch: 50,
        learning_rate: 5.0e-6,
        ppo_epochs: 2
      })

  ## Results

  After training completes, you can access:
  - `result.context.state.rollout_metrics` - Rollout statistics
  - `result.context.state.ppo_metrics` - PPO training metrics
  - `result.context.state.registered_model` - Registered model record

  ## References

  - Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning" (2024)
  - Chen et al., "Evaluating Large Language Models Trained on Code" (2021)
  """

  use CrucibleKitchen.Recipe

  alias CrucibleKitchen.Adapters.HfDatasets.DatasetStore, as: HfDatasetsAdapter
  alias CrucibleKitchen.Adapters.Noop.Evaluator, as: NoopEvaluator
  alias CrucibleKitchen.Adapters.Noop.ModelRegistry, as: NoopModelRegistry
  alias CrucibleKitchen.Adapters.Tinkex.TrainingClient, as: TinkexAdapter
  alias CrucibleKitchen.Workflows.Reinforcement, as: ReinforcementWorkflow

  require Logger

  @impl true
  def name, do: :code_rl

  @impl true
  def description do
    "Code generation via Reinforcement Learning with Tinkex backend"
  end

  @impl true
  def default_config do
    %{
      # Model config
      model: "meta-llama/Llama-3.1-8B-Instruct",
      lora_rank: 32,

      # Environment config
      env: :deepcoder,

      # RL config
      group_size: 4,
      groups_per_batch: 100,
      num_rollouts: 100,
      max_tokens: 512,

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
    [:training_client, :dataset_store]
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
  Run the CodeRL recipe with default Tinkex adapters.

  ## Options

  - `:api_key` - Override TINKER_API_KEY
  - `:base_url` - Override TINKER_BASE_URL

  ## Example

      TinkexCookbook.Recipes.CodeRL.run(%{
        model: "meta-llama/Llama-3.1-8B-Instruct",
        env: :deepcoder,
        num_rollouts: 50
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
  CLI entry point for mix kitchen.run :code_rl.
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
        Logger.info("CodeRL training completed successfully")

        if rollout_metrics = result.context.state[:rollout_metrics] do
          Logger.info("Rollout metrics:")
          Logger.info("  Reward mean: #{Map.get(rollout_metrics, :reward_mean, "N/A")}")
          Logger.info("  Reward std: #{Map.get(rollout_metrics, :reward_std, "N/A")}")
        end

        if ppo_metrics = result.context.state[:ppo_metrics] do
          Logger.info("PPO metrics:")
          Logger.info("  Policy loss: #{Map.get(ppo_metrics, :policy_loss, "N/A")}")
          Logger.info("  Entropy: #{Map.get(ppo_metrics, :entropy, "N/A")}")
          Logger.info("  Clip fraction: #{Map.get(ppo_metrics, :clip_fraction, "N/A")}")
        end

        if model = result.context.state[:registered_model] do
          Logger.info("Model registered: #{model.name} v#{model.version}")
        end

        :ok

      {:error, reason} ->
        Logger.error("CodeRL training failed: #{inspect(reason)}")
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
