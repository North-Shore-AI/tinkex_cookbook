defmodule TinkexCookbook.Recipes.SlBasic do
  @moduledoc """
  Basic supervised learning recipe.

  This is the Elixir port of Python's `tinker_cookbook/recipes/sl_basic.py`.
  It demonstrates a minimal supervised fine-tuning workflow using the NoRobots dataset.

  ## Usage

      # From command line
      mix run -e "TinkexCookbook.Recipes.SlBasic.main()"

      # With custom args
      mix run -e "TinkexCookbook.Recipes.SlBasic.main()" -- \\
        log_path=/tmp/my_run \\
        learning_rate=0.0002 \\
        num_epochs=2

  ## Environment Variables

      TINKER_API_KEY - Required. Your Tinker API key.
      TINKER_BASE_URL - Optional. Tinker API base URL.
      TINKEX_HTTP_PROTOCOL - Optional. Set to "http2" to use HTTP/2 (default uses HTTP/1 for training).
  """

  alias TinkexCookbook.Datasets.NoRobots
  alias TinkexCookbook.Renderers.{Llama3, TrainOnWhat}

  alias TinkexCookbook.Supervised.{
    ChatDatasetBuilderCommonConfig,
    Config,
    SupervisedDataset
  }

  alias TinkexCookbook.Utils.{CliUtils, MlLog}

  # CliConfig is defined in sl_basic_config.ex to work around ChzEx macro issues
  alias __MODULE__.CliConfig

  require Logger

  @default_base_url "https://tinker.thinkingmachines.dev/services/tinker-prod"

  @doc """
  Builds the default configuration blueprint for sl_basic.

  Returns a Config struct with sensible defaults for the NoRobots dataset.
  """
  @spec build_config(keyword()) :: Config.t()
  def build_config(overrides \\ []) do
    model_name = Keyword.get(overrides, :model_name, "meta-llama/Llama-3.1-8B")
    renderer_name = get_recommended_renderer_name(model_name)

    common_config = %ChatDatasetBuilderCommonConfig{
      model_name_for_tokenizer: model_name,
      renderer_name: renderer_name,
      max_length: Keyword.get(overrides, :max_length, 32_768),
      batch_size: Keyword.get(overrides, :batch_size, 128),
      train_on_what: Keyword.get(overrides, :train_on_what, "all_assistant_messages")
    }

    %Config{
      log_path: Keyword.get(overrides, :log_path, "/tmp/tinkex-examples/sl_basic"),
      model_name: model_name,
      dataset_builder: {:no_robots, common_config},
      learning_rate: Keyword.get(overrides, :learning_rate, 2.0e-4),
      lr_schedule: Keyword.get(overrides, :lr_schedule, "linear"),
      num_epochs: Keyword.get(overrides, :num_epochs, 1),
      lora_rank: Keyword.get(overrides, :lora_rank, 32),
      eval_every: Keyword.get(overrides, :eval_every, 8),
      save_every: Keyword.get(overrides, :save_every, 20)
    }
  end

  @doc """
  Returns the recommended renderer name for a given model.
  """
  @spec get_recommended_renderer_name(String.t()) :: String.t()
  def get_recommended_renderer_name(model_name) do
    cond do
      String.contains?(model_name, "Llama-3") -> "llama3"
      String.contains?(model_name, "Qwen") -> "qwen3"
      String.contains?(model_name, "DeepSeek") -> "deepseekv3"
      true -> "role_colon"
    end
  end

  @doc """
  Main entry point for the sl_basic recipe.

  Parses command line arguments and runs the training loop.
  """
  @spec main(list(String.t())) :: :ok | {:error, term()}
  def main(argv \\ System.argv()) do
    cli_config =
      case ChzEx.entrypoint(CliConfig, argv) do
        {:ok, config} -> config
        {:error, error} -> raise ChzEx.ConfigError, errors: error
      end

    overrides = cli_overrides(cli_config)
    config = build_config(overrides)

    # Expand log path
    config = Config.expand_log_path(config)

    # Check log directory
    behavior = behavior_atom(cli_config.behavior_if_exists)
    result = CliUtils.check_log_dir(config.log_path, behavior)

    # Get optional sample limit
    n_train_samples = cli_config.n_train_samples

    case result do
      :ok ->
        run_training(config, n_train_samples: n_train_samples)

      :resume ->
        Logger.info("Resuming training from checkpoint...")
        run_training(config, n_train_samples: n_train_samples, resume: true)
    end
  end

  @doc """
  Runs the training loop with the given configuration.

  This is the main training function that:
  1. Creates Tinkex clients
  2. Loads and processes the NoRobots dataset
  3. Runs the training loop with forward_backward and optim_step
  4. Saves checkpoints periodically

  ## Options

  - `:n_train_samples` - Limit the number of training samples (for testing)
  - `:resume` - Whether resuming from a checkpoint
  """
  @spec run_training(Config.t(), keyword()) :: :ok | {:error, term()}
  def run_training(config, opts \\ []) do
    n_train_samples = Keyword.get(opts, :n_train_samples)
    _resume = Keyword.get(opts, :resume, false)

    Logger.info("Starting sl_basic training...")
    Logger.info("  Model: #{config.model_name}")
    Logger.info("  Log path: #{config.log_path}")
    Logger.info("  Learning rate: #{config.learning_rate}")
    Logger.info("  LR schedule: #{config.lr_schedule}")
    Logger.info("  Epochs: #{config.num_epochs}")
    Logger.info("  Batch size: #{get_batch_size(config)}")
    Logger.info("  LoRA rank: #{config.lora_rank}")

    # Setup logging
    ml_logger = MlLog.setup_logging(config.log_path, config: config)

    # Get API key
    api_key = System.get_env("TINKER_API_KEY")

    if api_key do
      # Run the training pipeline
      result = run_training_pipeline(config, ml_logger, api_key, n_train_samples)

      # Close logger
      MlLog.close(ml_logger)

      result
    else
      Logger.error("TINKER_API_KEY environment variable is required")
      {:error, :missing_api_key}
    end
  end

  # Private training pipeline
  defp run_training_pipeline(config, ml_logger, api_key, n_train_samples) do
    base_url = System.get_env("TINKER_BASE_URL", @default_base_url)

    Logger.info("Connecting to Tinker at #{base_url}...")

    configure_tinkex_http()

    # Create Tinkex config and clients
    tinkex_config = Tinkex.Config.new(api_key: api_key, base_url: base_url)

    with {:ok, service_client} <- create_service_client(tinkex_config),
         {:ok, training_client} <- create_training_client(service_client, config),
         {:ok, tokenizer} <- get_tokenizer(training_client),
         {:ok, dataset} <- build_dataset(config, tokenizer, n_train_samples) do
      # Run the training loop
      run_training_loop(config, training_client, dataset, ml_logger)
    else
      {:error, reason} ->
        Logger.error("Training failed: #{inspect(reason)}")
        {:error, reason}
    end
  end

  defp create_service_client(tinkex_config) do
    Logger.info("Creating ServiceClient...")

    case Tinkex.ServiceClient.start_link(config: tinkex_config) do
      {:ok, pid} ->
        Logger.info("ServiceClient created")
        {:ok, pid}

      {:error, reason} ->
        {:error, {:service_client_failed, reason}}
    end
  end

  defp create_training_client(service_client, config) do
    Logger.info("Creating TrainingClient (LoRA rank=#{config.lora_rank})...")
    Logger.info("  [note] This may take 30-120s on first run (model loading)...")

    lora_config = %Tinkex.Types.LoraConfig{rank: config.lora_rank}

    case Tinkex.ServiceClient.create_lora_training_client(
           service_client,
           config.model_name,
           lora_config: lora_config,
           call_timeout: :infinity
         ) do
      {:ok, pid} ->
        Logger.info("TrainingClient created")
        {:ok, pid}

      {:error, reason} ->
        {:error, {:training_client_failed, reason}}
    end
  end

  defp get_tokenizer(training_client) do
    Logger.info("Getting tokenizer...")

    case Tinkex.TrainingClient.get_tokenizer(training_client) do
      {:ok, tokenizer} ->
        Logger.info("Tokenizer loaded")
        {:ok, tokenizer}

      {:error, reason} ->
        {:error, {:tokenizer_failed, reason}}
    end
  end

  defp build_dataset(config, tokenizer, n_train_samples) do
    {:no_robots, common_config} = config.dataset_builder

    Logger.info("Loading NoRobots dataset...")

    load_opts =
      if n_train_samples do
        [split: "train", limit: n_train_samples]
      else
        [split: "train"]
      end

    case NoRobots.load(load_opts) do
      {:ok, samples} ->
        Logger.info("Loaded #{length(samples)} samples")
        Logger.info("Building supervised dataset...")

        # Create renderer state with the tokenizer wrapper
        tokenizer_wrapper = create_tokenizer_wrapper(tokenizer)
        {:ok, renderer_state} = Llama3.init(tokenizer: tokenizer_wrapper)

        train_on_what = common_config.train_on_what || TrainOnWhat.all_assistant_messages()

        dataset =
          NoRobots.create_supervised_dataset(
            samples,
            renderer_module: Llama3,
            renderer_state: renderer_state,
            train_on_what: train_on_what,
            batch_size: common_config.batch_size,
            max_length: common_config.max_length
          )

        n_batches = SupervisedDataset.length(dataset)
        Logger.info("Dataset ready: #{length(samples)} samples, #{n_batches} batches")

        {:ok, dataset}

      {:error, reason} ->
        {:error, {:dataset_load_failed, reason}}
    end
  end

  # Wraps a Tinkex tokenizer handle to provide the interface expected by our renderers
  # The tokenizer handle is a Tokenizers.Tokenizer.t() (NIF-backed)
  defp create_tokenizer_wrapper(tokenizer_handle) do
    %{
      encode: fn text, _opts ->
        # Use Tokenizers NIF directly
        case Tokenizers.Tokenizer.encode(tokenizer_handle, text) do
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

  defp run_training_loop(config, training_client, dataset, ml_logger) do
    n_batches = SupervisedDataset.length(dataset)
    total_steps = n_batches * config.num_epochs
    lr_schedule = parse_lr_schedule(config.lr_schedule)

    Logger.info("Starting training loop...")
    Logger.info("  Total steps: #{total_steps}")

    MlLog.log_metrics(ml_logger, %{status: "training_started", total_steps: total_steps}, 0)

    # Run epochs
    result =
      Enum.reduce_while(0..(config.num_epochs - 1), {:ok, 0}, fn epoch_idx, {:ok, step_offset} ->
        Logger.info("Starting epoch #{epoch_idx + 1}/#{config.num_epochs}")

        # Shuffle dataset for this epoch
        dataset = SupervisedDataset.set_epoch(dataset, epoch_idx)

        case run_epoch(
               config,
               training_client,
               dataset,
               epoch_idx,
               step_offset,
               total_steps,
               lr_schedule,
               ml_logger
             ) do
          {:ok, new_step_offset} ->
            {:cont, {:ok, new_step_offset}}

          {:error, reason} ->
            {:halt, {:error, reason}}
        end
      end)

    case result do
      {:ok, _final_step} ->
        Logger.info("Training complete!")
        MlLog.log_metrics(ml_logger, %{status: "training_complete"}, total_steps)

        # Save final weights
        save_final_weights(training_client, config)

        :ok

      {:error, reason} ->
        Logger.error("Training failed: #{inspect(reason)}")
        {:error, reason}
    end
  end

  defp run_epoch(
         config,
         training_client,
         dataset,
         epoch_idx,
         step_offset,
         total_steps,
         lr_schedule,
         ml_logger
       ) do
    n_batches = SupervisedDataset.length(dataset)

    Enum.reduce_while(0..(n_batches - 1), {:ok, step_offset}, fn batch_idx, {:ok, current_step} ->
      global_step = step_offset + batch_idx

      # Get batch
      batch = SupervisedDataset.get_batch(dataset, batch_idx)

      # Convert our Datum structs to Tinkex format
      tinkex_data = Enum.map(batch, &datum_to_tinkex/1)

      # Compute learning rate
      lr = compute_lr(config.learning_rate, global_step, total_steps, lr_schedule)

      # Create Adam params
      adam_params = %Tinkex.Types.AdamParams{
        learning_rate: lr,
        beta1: config.adam_beta1,
        beta2: config.adam_beta2,
        eps: config.adam_eps
      }

      # Run training step and process result
      step_context = %{
        global_step: global_step,
        epoch_idx: epoch_idx,
        batch_idx: batch_idx,
        lr: lr,
        total_steps: total_steps,
        config: config,
        training_client: training_client,
        ml_logger: ml_logger,
        current_step: current_step
      }

      process_training_step(training_client, tinkex_data, adam_params, step_context)
    end)
  end

  defp process_training_step(training_client, tinkex_data, adam_params, ctx) do
    case training_step(training_client, tinkex_data, adam_params) do
      {:ok, metrics} ->
        handle_step_success(metrics, ctx)

      {:error, reason} ->
        Logger.error("Training step #{ctx.global_step} failed: #{inspect(reason)}")
        {:halt, {:error, reason}}
    end
  end

  defp handle_step_success(metrics, ctx) do
    # Log metrics
    step_metrics =
      Map.merge(metrics, %{
        step: ctx.global_step,
        epoch: ctx.epoch_idx,
        batch: ctx.batch_idx,
        learning_rate: ctx.lr
      })

    MlLog.log_metrics(ctx.ml_logger, step_metrics, ctx.global_step)

    # Periodic logging
    log_step_progress(metrics, ctx)

    # Save checkpoint if needed
    maybe_save_checkpoint(ctx)

    {:cont, {:ok, ctx.current_step + 1}}
  end

  defp log_step_progress(metrics, ctx) do
    if rem(ctx.global_step + 1, 10) == 0 do
      Logger.info(
        "Step #{ctx.global_step + 1}/#{ctx.total_steps} | " <>
          "loss=#{format_float(metrics["loss"])} | " <>
          "lr=#{format_float(ctx.lr)}"
      )
    end
  end

  defp maybe_save_checkpoint(ctx) do
    if ctx.config.save_every > 0 and rem(ctx.global_step + 1, ctx.config.save_every) == 0 do
      save_checkpoint(ctx.training_client, ctx.config, ctx.global_step)
    end
  end

  defp training_step(training_client, data, adam_params) do
    # Forward-backward pass - always returns {:ok, task}
    {:ok, fb_task} =
      Tinkex.TrainingClient.forward_backward(training_client, data, :cross_entropy)

    case Task.await(fb_task, :infinity) do
      {:ok, fb_output} ->
        # Optimizer step - always returns {:ok, task}
        {:ok, optim_task} = Tinkex.TrainingClient.optim_step(training_client, adam_params)

        case Task.await(optim_task, :infinity) do
          {:ok, _optim_output} ->
            {:ok, fb_output.metrics || %{}}

          {:error, reason} ->
            {:error, {:optim_step_failed, reason}}
        end

      {:error, reason} ->
        {:error, {:forward_backward_failed, reason}}
    end
  end

  defp save_checkpoint(training_client, _config, step) do
    checkpoint_name = "step_#{String.pad_leading(Integer.to_string(step), 6, "0")}"
    Logger.info("Saving checkpoint: #{checkpoint_name}")

    # save_state always returns {:ok, task}
    {:ok, task} = Tinkex.TrainingClient.save_state(training_client, checkpoint_name)

    case Task.await(task, :infinity) do
      {:ok, _} ->
        Logger.info("Checkpoint saved: #{checkpoint_name}")
        :ok

      {:error, reason} ->
        Logger.warning("Failed to save checkpoint: #{inspect(reason)}")
        :error
    end
  end

  defp save_final_weights(training_client, _config) do
    weights_name = "final_weights"
    Logger.info("Saving final weights: #{weights_name}")

    # save_weights_for_sampler always returns {:ok, task}
    {:ok, task} = Tinkex.TrainingClient.save_weights_for_sampler(training_client, weights_name)

    case Task.await(task, :infinity) do
      {:ok, result} ->
        Logger.info("Final weights saved: #{inspect(result)}")
        {:ok, result}

      {:error, reason} ->
        Logger.warning("Failed to save final weights: #{inspect(reason)}")
        {:error, reason}
    end
  end

  # Convert our Datum struct to Tinkex format
  defp datum_to_tinkex(%TinkexCookbook.Types.Datum{} = datum) do
    # Convert our ModelInput to Tinkex ModelInput
    tinkex_model_input = model_input_to_tinkex(datum.model_input)

    # Convert loss_fn_inputs
    tinkex_loss_fn_inputs =
      datum.loss_fn_inputs
      |> Enum.map(fn {key, tensor_data} ->
        {key, tensor_data_to_tinkex(tensor_data)}
      end)
      |> Map.new()

    %{
      model_input: tinkex_model_input,
      loss_fn_inputs: tinkex_loss_fn_inputs
    }
  end

  defp model_input_to_tinkex(%TinkexCookbook.Types.ModelInput{chunks: chunks}) do
    tinkex_chunks =
      Enum.map(chunks, fn
        %TinkexCookbook.Types.EncodedTextChunk{tokens: tokens} ->
          %Tinkex.Types.EncodedTextChunk{tokens: tokens, type: "encoded_text"}
      end)

    %Tinkex.Types.ModelInput{chunks: tinkex_chunks}
  end

  defp tensor_data_to_tinkex(%TinkexCookbook.Types.TensorData{} = td) do
    %Tinkex.Types.TensorData{
      data: td.data,
      dtype: td.dtype,
      shape: td.shape
    }
  end

  # Learning rate computation
  defp compute_lr(base_lr, step, total_steps, schedule) do
    progress = step / max(total_steps - 1, 1)

    case schedule do
      :constant -> base_lr
      :linear -> base_lr * (1.0 - progress)
      :cosine -> base_lr * 0.5 * (1.0 + :math.cos(:math.pi() * progress))
    end
  end

  defp parse_lr_schedule("linear"), do: :linear
  defp parse_lr_schedule("constant"), do: :constant
  defp parse_lr_schedule("cosine"), do: :cosine
  defp parse_lr_schedule(other), do: raise(ArgumentError, "Unknown lr_schedule: #{other}")

  defp get_batch_size(%Config{dataset_builder: {:no_robots, common_config}}) do
    common_config.batch_size
  end

  defp get_batch_size(_), do: 128

  defp format_float(nil), do: "N/A"
  defp format_float(f) when is_float(f), do: :erlang.float_to_binary(f, decimals: 6)
  defp format_float(other), do: inspect(other)

  # Private helpers
  defp cli_overrides(%CliConfig{} = config) do
    [
      log_path: config.log_path,
      model_name: config.model_name,
      learning_rate: config.learning_rate,
      lr_schedule: config.lr_schedule,
      num_epochs: config.num_epochs,
      eval_every: config.eval_every,
      save_every: config.save_every,
      batch_size: config.batch_size,
      max_length: config.max_length,
      lora_rank: config.lora_rank,
      train_on_what: config.train_on_what
    ]
  end

  defp behavior_atom(value) when is_atom(value), do: value
  defp behavior_atom("delete"), do: :delete
  defp behavior_atom("resume"), do: :resume
  defp behavior_atom("ask"), do: :ask
  defp behavior_atom("raise"), do: :raise

  defp behavior_atom(value) do
    raise ArgumentError, "Invalid behavior_if_exists: #{inspect(value)}"
  end

  @doc false
  def configure_tinkex_http do
    case System.get_env("TINKEX_HTTP_PROTOCOL") do
      "http2" ->
        :ok

      "http1" ->
        force_http1_training_pool()
        maybe_restart_tinkex_app()

      nil ->
        force_http1_training_pool()
        maybe_restart_tinkex_app()

      other ->
        Logger.warning(
          "Unknown TINKEX_HTTP_PROTOCOL=#{inspect(other)}; defaulting to HTTP/1 for training"
        )

        force_http1_training_pool()
        maybe_restart_tinkex_app()
    end
  end

  defp force_http1_training_pool do
    overrides =
      case Application.get_env(:tinkex, :pool_overrides) do
        value when is_map(value) -> value
        _ -> %{}
      end

    training_overrides =
      overrides
      |> Map.get(:training, [])
      |> normalize_overrides()
      |> Keyword.put(:protocols, [:http1])

    Application.put_env(
      :tinkex,
      :pool_overrides,
      Map.put(overrides, :training, training_overrides)
    )
  end

  defp maybe_restart_tinkex_app do
    if mix_env() == :test do
      :ok
    else
      if tinkex_started?() do
        Logger.info("Restarting Tinkex to apply HTTP pool overrides")
        _ = Application.stop(:tinkex)
        _ = Application.ensure_all_started(:tinkex)
      end

      :ok
    end
  end

  defp tinkex_started? do
    Enum.any?(Application.started_applications(), fn {app, _desc, _vsn} -> app == :tinkex end)
  end

  defp mix_env do
    if Code.ensure_loaded?(Mix) do
      Mix.env()
    else
      :prod
    end
  end

  defp normalize_overrides(overrides) when is_list(overrides), do: overrides
  defp normalize_overrides(overrides) when is_map(overrides), do: Map.to_list(overrides)
  defp normalize_overrides(_), do: []
end
