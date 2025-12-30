defmodule TinkexCookbook.Recipes.SlBasic do
  @moduledoc """
  Basic supervised learning recipe.

  This recipe provides supervised fine-tuning on the NoRobots dataset
  using the Tinker ML platform via the TinkexCookbook.Runtime facade.

  ## Usage

      # Via Runtime facade
      config = %{model: "meta-llama/Llama-3.1-8B", num_epochs: 1}
      TinkexCookbook.Runtime.run(TinkexCookbook.Recipes.SlBasic, config)

      # Via mix task
      mix tinkex.sl_basic --model meta-llama/Llama-3.1-8B
  """

  @behaviour TinkexCookbook.Recipe

  alias CrucibleIR.{BackendRef, DatasetRef, Experiment, ModelRef, StageDef}
  alias CrucibleIR.Training
  alias CrucibleTrain.Renderers.TrainOnWhat
  alias CrucibleTrain.Supervised.Dataset, as: SupervisedDataset
  alias CrucibleKitchen.Adapters.Tinkex.TrainingClient, as: TinkexAdapter
  alias TinkexCookbook.Datasets.NoRobots

  require Logger

  @impl true
  def name, do: "sl_basic"

  @impl true
  def description, do: "Basic supervised fine-tuning with NoRobots dataset"

  @impl true
  def config_schema, do: CliConfig

  @impl true
  def default_config do
    %{
      model: "meta-llama/Llama-3.1-8B",
      dataset: "no_robots",
      num_epochs: 1,
      learning_rate: 2.0e-4,
      lr_schedule: "linear",
      batch_size: 128,
      max_length: 32_768,
      lora_rank: 32,
      train_on: TrainOnWhat.all_assistant_messages(),
      eval_every: 8,
      save_every: 20,
      log_path: "/tmp/tinkex-examples/sl_basic"
    }
  end

  @impl true
  def build_spec(config) do
    config = Map.merge(default_config(), normalize_config(config))

    # Build the experiment ID from recipe + model
    experiment_id = :"sl_basic_#{String.replace(config.model, "/", "_")}"

    %Experiment{
      id: experiment_id,
      description: description(),
      experiment_type: :training,
      backend: %BackendRef{
        id: :tinker,
        model_version: config.model
      },
      metadata: %{
        recipe: name(),
        model: config.model,
        dataset: config.dataset
      },
      dataset: %DatasetRef{name: String.to_atom(config.dataset)},
      training_config: %Training.Config{
        id: :"#{experiment_id}_config",
        model_ref: %ModelRef{id: :base_model, name: config.model},
        dataset_ref: %DatasetRef{name: String.to_atom(config.dataset)},
        epochs: config.num_epochs,
        batch_size: config.batch_size,
        learning_rate: config.learning_rate,
        options: %{
          lr_schedule: parse_lr_schedule(config.lr_schedule),
          max_length: config.max_length,
          lora_rank: config.lora_rank,
          train_on: config.train_on,
          renderer: get_renderer(config.model)
        }
      },
      pipeline: [
        %StageDef{
          name: :supervised_train,
          module: CrucibleTrain.Stages.SupervisedTrain,
          options: %{
            model: config.model,
            dataset: config.dataset,
            num_epochs: config.num_epochs,
            learning_rate: config.learning_rate,
            lr_schedule: parse_lr_schedule(config.lr_schedule),
            batch_size: config.batch_size,
            max_length: config.max_length,
            lora_rank: config.lora_rank,
            train_on: config.train_on,
            renderer: get_renderer(config.model)
          }
        }
      ]
    }
  end

  @doc """
  Main entry point for CLI execution.

  This function is called by `mix sl_basic` and provides
  backward compatibility with the original implementation.

  ## Arguments

  Accepts key=value pairs:
  - `n_train_samples=100` - Limit number of training samples
  - `model_name=...` - Model to train
  - `learning_rate=0.0002` - Learning rate
  - etc.
  """
  @spec main(list(String.t())) :: :ok | {:error, term()}
  def main(argv \\ System.argv()) do
    # Simple argv parsing: key=value pairs
    config =
      argv
      |> Enum.filter(&String.contains?(&1, "="))
      |> Enum.map(fn arg ->
        [key, value] = String.split(arg, "=", parts: 2)
        {String.to_atom(key), parse_value(value)}
      end)
      |> Map.new()

    # Extract n_train_samples for run_training opts
    {n_train_samples, config} = Map.pop(config, :n_train_samples)
    opts = if n_train_samples, do: [n_train_samples: n_train_samples], else: []

    run_training(config, opts)
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

  @doc """
  Run the supervised training with direct Tinkex calls.

  This maintains backward compatibility while the crucible_train stages
  are being developed to support Tinkex.
  """
  @spec run_training(map(), keyword()) :: :ok | {:error, term()}
  def run_training(config, opts \\ []) do
    config = Map.merge(default_config(), normalize_config(config))
    n_train_samples = Keyword.get(opts, :n_train_samples)

    Logger.info("Starting sl_basic training...")
    Logger.info("  Model: #{config.model}")
    Logger.info("  Learning rate: #{config.learning_rate}")
    Logger.info("  Epochs: #{config.num_epochs}")
    Logger.info("  Batch size: #{config.batch_size}")
    Logger.info("  LoRA rank: #{config.lora_rank}")

    # Start training session via adapter
    session_config = %{
      model: config.model,
      lora_rank: config.lora_rank,
      learning_rate: config.learning_rate
    }

    case TinkexAdapter.start_session([], session_config) do
      {:ok, session} ->
        result = run_training_loop(session, config, n_train_samples)
        TinkexAdapter.close_session([], session)
        result

      {:error, reason} ->
        Logger.error("Failed to start training session: #{inspect(reason)}")
        {:error, reason}
    end
  end

  # Private training implementation
  defp run_training_loop(session, config, n_train_samples) do
    # Get tokenizer for rendering
    case TinkexAdapter.get_tokenizer([], session) do
      {:ok, tokenizer} ->
        run_with_tokenizer(session, config, tokenizer, n_train_samples)

      {:error, reason} ->
        {:error, {:tokenizer_failed, reason}}
    end
  end

  defp run_with_tokenizer(session, config, tokenizer, n_train_samples) do
    # Load dataset
    load_opts =
      if n_train_samples do
        [split: "train", shuffle_seed: 0, limit: n_train_samples]
      else
        [split: "train", shuffle_seed: 0]
      end

    case NoRobots.load(load_opts) do
      {:ok, samples} ->
        Logger.info("Loaded #{length(samples)} samples")
        run_epochs(session, config, samples, tokenizer)

      {:error, reason} ->
        {:error, {:dataset_load_failed, reason}}
    end
  end

  defp run_epochs(session, config, samples, tokenizer) do
    # Get renderer
    renderer = get_renderer_module(config.model)
    tokenizer_wrapper = create_tokenizer_wrapper(tokenizer)
    {:ok, renderer_state} = renderer.init(tokenizer: tokenizer_wrapper)

    # Build dataset
    train_on = config.train_on
    batch_size = config.batch_size
    max_length = config.max_length

    dataset =
      NoRobots.create_supervised_dataset(
        samples,
        renderer_module: renderer,
        renderer_state: renderer_state,
        train_on_what: train_on,
        batch_size: batch_size,
        max_length: max_length
      )

    n_batches = SupervisedDataset.length(dataset)
    total_steps = n_batches * config.num_epochs

    Logger.info("Dataset ready: #{length(samples)} samples, #{n_batches} batches")
    Logger.info("Total training steps: #{total_steps}")

    # Run training epochs
    run_training_epochs(session, config, dataset, total_steps)
  end

  defp run_training_epochs(session, config, dataset, total_steps) do
    lr_schedule = parse_lr_schedule(config.lr_schedule)
    n_batches = SupervisedDataset.length(dataset)

    result =
      Enum.reduce_while(0..(config.num_epochs - 1), {:ok, 0}, fn epoch_idx, {:ok, step_offset} ->
        Logger.info("Epoch #{epoch_idx + 1}/#{config.num_epochs}")
        dataset = SupervisedDataset.set_epoch(dataset, epoch_idx)

        case run_epoch(
               session,
               config,
               dataset,
               epoch_idx,
               step_offset,
               total_steps,
               lr_schedule,
               n_batches
             ) do
          {:ok, new_step_offset} -> {:cont, {:ok, new_step_offset}}
          {:error, reason} -> {:halt, {:error, reason}}
        end
      end)

    case result do
      {:ok, _} ->
        Logger.info("Training complete!")
        save_final_weights(session)
        :ok

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp run_epoch(
         session,
         config,
         dataset,
         _epoch_idx,
         step_offset,
         total_steps,
         lr_schedule,
         n_batches
       ) do
    Enum.reduce_while(0..(n_batches - 1), {:ok, step_offset}, fn batch_idx, {:ok, current_step} ->
      global_step = step_offset + batch_idx
      batch = SupervisedDataset.get_batch(dataset, batch_idx)

      # Compute learning rate
      lr = compute_lr(config.learning_rate, global_step, total_steps, lr_schedule)

      # Run training step
      case run_training_step(session, batch, lr) do
        {:ok, _metrics} ->
          log_progress(global_step, total_steps, lr)
          maybe_save_checkpoint(session, config, global_step)
          {:cont, {:ok, current_step + 1}}

        {:error, reason} ->
          Logger.error("Training step #{global_step} failed: #{inspect(reason)}")
          {:halt, {:error, reason}}
      end
    end)
  end

  defp run_training_step(session, batch, lr) do
    # Forward-backward pass
    fb_future = TinkexAdapter.forward_backward([], session, batch)

    case TinkexAdapter.await([], fb_future) do
      {:ok, _fb_result} ->
        optim_future = TinkexAdapter.optim_step([], session, lr)
        TinkexAdapter.await([], optim_future)

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp log_progress(global_step, total_steps, lr) do
    if rem(global_step + 1, 10) == 0 do
      Logger.info("Step #{global_step + 1}/#{total_steps} | lr=#{Float.round(lr, 8)}")
    end
  end

  defp maybe_save_checkpoint(session, config, global_step) do
    if config.save_every > 0 and rem(global_step + 1, config.save_every) == 0 do
      checkpoint_name = "step_#{String.pad_leading(Integer.to_string(global_step), 6, "0")}"
      Logger.info("Saving checkpoint: #{checkpoint_name}")
      TinkexAdapter.save_checkpoint([], session, checkpoint_name)
    end
  end

  defp save_final_weights(session) do
    Logger.info("Saving final weights...")
    TinkexAdapter.save_weights_for_sampler(session, "final_weights")
  end

  # Helper functions
  defp normalize_config(config) when is_struct(config), do: Map.from_struct(config)
  defp normalize_config(config) when is_map(config), do: config

  defp parse_lr_schedule("linear"), do: :linear
  defp parse_lr_schedule("constant"), do: :constant
  defp parse_lr_schedule("cosine"), do: :cosine
  defp parse_lr_schedule(other) when is_atom(other), do: other
  defp parse_lr_schedule(_), do: :linear

  defp compute_lr(base_lr, step, total_steps, schedule) do
    progress = step / max(total_steps - 1, 1)

    case schedule do
      :constant -> base_lr
      :linear -> base_lr * (1.0 - progress)
      :cosine -> base_lr * 0.5 * (1.0 + :math.cos(:math.pi() * progress))
    end
  end

  defp get_renderer(model_name) do
    cond do
      String.contains?(model_name, "Llama-3") -> CrucibleTrain.Renderers.Llama3
      String.contains?(model_name, "Qwen") -> CrucibleTrain.Renderers.Qwen3
      String.contains?(model_name, "DeepSeek") -> CrucibleTrain.Renderers.DeepSeekV3
      true -> CrucibleTrain.Renderers.RoleColon
    end
  end

  defp get_renderer_module(model_name) do
    get_renderer(model_name)
  end

  defp create_tokenizer_wrapper(tokenizer_handle) do
    %{
      encode: fn text, opts ->
        add_special = Keyword.get(opts, :add_special_tokens, true)

        case Tokenizers.Tokenizer.encode(tokenizer_handle, text, add_special_tokens: add_special) do
          {:ok, encoding} -> Tokenizers.Encoding.get_ids(encoding)
          {:error, _} -> []
        end
      end,
      decode: fn tokens ->
        case Tokenizers.Tokenizer.decode(tokenizer_handle, tokens) do
          {:ok, text} -> text
          {:error, _} -> ""
        end
      end
    }
  end
end
