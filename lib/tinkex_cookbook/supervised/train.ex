defmodule TinkexCookbook.Supervised.Train do
  @moduledoc """
  Supervised training orchestration using Tinkex clients.

  This module implements the core training loop for supervised fine-tuning.
  It handles batching, learning rate scheduling, and metric collection.

  ## Architecture

  The training flow follows the pipelining pattern from the Python implementation:

  1. Load dataset and build datums
  2. Initialize training client
  3. For each epoch:
     - Shuffle dataset
     - For each batch:
       - Compute learning rate
       - Execute forward_backward
       - Execute optim_step
       - Log metrics
  4. Save final checkpoint

  ## Usage

      config = Train.TrainConfig.new(
        model_name: "meta-llama/Llama-3.1-8B",
        log_path: "/tmp/my_training",
        learning_rate: 2.0e-4,
        num_epochs: 1
      )

      {:ok, results} = Train.run(config, training_client, dataset)

  See also: `TinkexCookbook.Recipes.SlBasic` for a complete example.
  """

  require Logger

  alias TinkexCookbook.Supervised.SupervisedDataset
  alias TinkexCookbook.Types.Datum

  defmodule TrainConfig do
    @moduledoc """
    Configuration for supervised training.
    """

    @type lr_schedule :: :linear | :constant | :cosine

    @type t :: %__MODULE__{
            model_name: String.t(),
            log_path: String.t(),
            learning_rate: float(),
            lr_schedule: lr_schedule(),
            num_epochs: pos_integer(),
            adam_beta1: float(),
            adam_beta2: float(),
            adam_eps: float(),
            eval_every: non_neg_integer(),
            save_every: non_neg_integer()
          }

    defstruct [
      :model_name,
      :log_path,
      learning_rate: 1.0e-4,
      lr_schedule: :linear,
      num_epochs: 1,
      adam_beta1: 0.9,
      adam_beta2: 0.95,
      adam_eps: 1.0e-8,
      eval_every: 10,
      save_every: 20
    ]

    @doc """
    Creates a new TrainConfig with the given options.
    """
    @spec new(keyword()) :: t()
    def new(opts) do
      struct!(__MODULE__, opts)
    end
  end

  @doc """
  Batches datums into groups of the specified batch size.

  Any remaining datums that don't fill a complete batch are dropped.
  """
  @spec batch_datums([Datum.t()], pos_integer()) :: [[Datum.t()]]
  def batch_datums(datums, batch_size) when batch_size > 0 do
    datums
    |> Enum.chunk_every(batch_size)
    |> Enum.filter(fn batch -> length(batch) == batch_size end)
  end

  @doc """
  Executes a single training step: forward_backward + optim_step.

  Returns the metrics from the forward_backward pass.
  """
  @spec training_step(struct(), [Datum.t()], map()) :: {:ok, map()} | {:error, term()}
  def training_step(training_client, batch, adam_params) do
    case training_client.__struct__.forward_backward(training_client, batch) do
      {:ok, fwd_bwd_result} ->
        :ok = training_client.__struct__.optim_step(training_client, adam_params)

        metrics = fwd_bwd_result[:metrics] || fwd_bwd_result["metrics"] || %{}
        {:ok, metrics}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Computes the learning rate for a given step using the specified schedule.

  ## Schedules

  - `:constant` - Learning rate stays fixed
  - `:linear` - Linear decay from base_lr to 0
  - `:cosine` - Cosine annealing from base_lr to 0
  """
  @spec compute_lr(float(), non_neg_integer(), pos_integer(), TrainConfig.lr_schedule()) ::
          float()
  def compute_lr(base_lr, step, total_steps, schedule) do
    progress = step / max(total_steps - 1, 1)

    case schedule do
      :constant ->
        base_lr

      :linear ->
        base_lr * (1.0 - progress)

      :cosine ->
        base_lr * 0.5 * (1.0 + :math.cos(:math.pi() * progress))
    end
  end

  @doc """
  Runs a single epoch of training.

  Returns a list of metrics for each step in the epoch.
  """
  @spec run_epoch(struct(), SupervisedDataset.t(), non_neg_integer(), map()) ::
          {:ok, [map()]} | {:error, term()}
  def run_epoch(training_client, dataset, epoch_idx, config) do
    n_batches = SupervisedDataset.length(dataset)
    dataset = SupervisedDataset.set_epoch(dataset, epoch_idx)

    metrics_list =
      for batch_idx <- 0..(n_batches - 1) do
        batch = SupervisedDataset.get_batch(dataset, batch_idx)
        step = epoch_idx * n_batches + batch_idx

        lr =
          compute_lr(
            config.learning_rate,
            step,
            config.total_steps,
            config.lr_schedule
          )

        adam_params = %{
          learning_rate: lr,
          beta1: config.adam_beta1,
          beta2: config.adam_beta2,
          eps: config.adam_eps
        }

        case training_step(training_client, batch, adam_params) do
          {:ok, step_metrics} ->
            Map.merge(step_metrics, %{
              "step" => step,
              "epoch" => epoch_idx,
              "batch" => batch_idx,
              "learning_rate" => lr
            })

          {:error, reason} ->
            Logger.error("Training step #{step} failed: #{inspect(reason)}")
            %{"step" => step, "error" => inspect(reason)}
        end
      end

    {:ok, metrics_list}
  end

  @doc """
  Runs the full training loop.

  ## Parameters

  - `config` - TrainConfig with training parameters
  - `training_client` - Initialized Tinkex TrainingClient
  - `dataset` - SupervisedDataset with training data

  ## Returns

  - `{:ok, all_metrics}` - List of metrics from all steps
  - `{:error, reason}` - If training fails
  """
  @spec run(TrainConfig.t(), struct(), SupervisedDataset.t()) :: {:ok, [map()]} | {:error, term()}
  def run(%TrainConfig{} = config, training_client, dataset) do
    Logger.info("Starting supervised training...")
    Logger.info("  Model: #{config.model_name}")
    Logger.info("  Epochs: #{config.num_epochs}")
    Logger.info("  Learning rate: #{config.learning_rate}")

    n_batches = SupervisedDataset.length(dataset)
    total_steps = n_batches * config.num_epochs

    Logger.info("  Total steps: #{total_steps}")

    run_config = Map.merge(Map.from_struct(config), %{total_steps: total_steps})

    all_metrics =
      Enum.flat_map(0..(config.num_epochs - 1), fn epoch_idx ->
        Logger.info("Starting epoch #{epoch_idx + 1}/#{config.num_epochs}")

        {:ok, epoch_metrics} = run_epoch(training_client, dataset, epoch_idx, run_config)
        epoch_metrics
      end)

    Logger.info("Training complete. Total steps: #{length(all_metrics)}")

    {:ok, all_metrics}
  end
end
