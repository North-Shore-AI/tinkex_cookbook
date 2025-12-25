defmodule TinkexCookbook.Preference.TrainDpo do
  @moduledoc """
  Direct Preference Optimization (DPO) training loop.
  """

  require Logger

  alias Tinkex.Types.AdamParams
  alias TinkexCookbook.Display
  alias TinkexCookbook.Supervised.SupervisedDataset
  alias TinkexCookbook.TokenizerUtils
  alias TinkexCookbook.Types.{Datum, ModelInput, TensorData}
  alias TinkexCookbook.Utils.{Checkpoint, LRScheduling, MiscUtils, TinkexConvert}
  alias TinkexCookbook.Utils.MlLog

  @default_base_url "https://tinker.thinkingmachines.dev/services/tinker-prod"

  # Forward alias declarations for nested modules (defined below)
  alias __MODULE__.{Config, UpdateContext}

  defmodule Config do
    @moduledoc """
    Configuration for DPO training.
    """

    use ChzEx.Schema

    chz_schema do
      field(:log_path, :string)
      field(:model_name, :string)
      field(:dataset_builder, :any, virtual: true)
      field(:load_checkpoint_path, :string, default: nil)
      field(:learning_rate, :float, default: 1.0e-5)
      field(:lr_schedule, :string, default: "linear")
      field(:num_epochs, :integer, default: 1)
      field(:dpo_beta, :float, default: 0.1)
      field(:lora_rank, :integer, default: 32)
      field(:num_replicas, :integer, default: 8)
      field(:base_url, :string, default: nil)
      field(:evaluator_builders, :any, default: [], virtual: true)
      field(:infrequent_evaluator_builders, :any, default: [], virtual: true)
      field(:save_every, :integer, default: 20)
      field(:eval_every, :integer, default: 10)
      field(:infrequent_eval_every, :integer, default: 100)
      field(:adam_beta1, :float, default: 0.9)
      field(:adam_beta2, :float, default: 0.95)
      field(:adam_eps, :float, default: 1.0e-8)
      field(:wandb_project, :string, default: nil)
      field(:wandb_name, :string, default: nil)
      field(:reference_model_name, :string, default: nil)
    end

    @type t() :: %__MODULE__{
            log_path: String.t(),
            model_name: String.t(),
            dataset_builder: any(),
            load_checkpoint_path: String.t() | nil,
            learning_rate: float(),
            lr_schedule: String.t(),
            num_epochs: integer(),
            dpo_beta: float(),
            lora_rank: integer(),
            num_replicas: integer(),
            base_url: String.t() | nil,
            evaluator_builders: list(),
            infrequent_evaluator_builders: list(),
            save_every: integer(),
            eval_every: integer(),
            infrequent_eval_every: integer(),
            adam_beta1: float(),
            adam_beta2: float(),
            adam_eps: float(),
            wandb_project: String.t() | nil,
            wandb_name: String.t() | nil,
            reference_model_name: String.t() | nil
          }

    @spec expand_log_path(t()) :: t()
    def expand_log_path(%__MODULE__{log_path: log_path} = cfg) do
      %{cfg | log_path: Path.expand(log_path)}
    end
  end

  defmodule UpdateContext do
    @moduledoc false
    defstruct [
      :cfg,
      :training_client,
      :reference_client,
      :evaluators,
      :infrequent_evaluators,
      :ml_logger,
      :log_path,
      :tokenizer
    ]

    @type t() :: %__MODULE__{
            cfg: TinkexCookbook.Preference.TrainDpo.Config.t(),
            training_client: struct(),
            reference_client: struct(),
            evaluators: list(),
            infrequent_evaluators: list(),
            ml_logger: pid(),
            log_path: String.t(),
            tokenizer: any()
          }
  end

  @spec create_dpo_clients(Config.t(), map() | nil, struct() | pid()) :: {struct(), struct()}
  def create_dpo_clients(cfg, resume_info, service_client) do
    {:ok, training_client} =
      service_client_module(service_client).create_lora_training_client(
        service_client,
        cfg.model_name,
        rank: cfg.lora_rank
      )

    cond do
      resume_info != nil ->
        path = resume_info["state_path"] || resume_info[:state_path]

        if path do
          _ =
            await_task!(
              training_client_module(training_client).load_state_with_optimizer(
                training_client,
                path
              )
            )

          Logger.info("Resumed DPO training from #{path}")
        end

      cfg.load_checkpoint_path != nil ->
        _ =
          await_task!(
            training_client_module(training_client).load_state(
              training_client,
              cfg.load_checkpoint_path
            )
          )

        Logger.info("Loaded weights from #{cfg.load_checkpoint_path}")

      true ->
        :ok
    end

    reference_client =
      await_task!(
        training_client_module(training_client).save_weights_and_get_sampling_client(
          training_client,
          name: "reference"
        )
      )

    {training_client, reference_client}
  end

  @spec compute_dpo_loss(
          [
            Nx.Tensor.t()
          ],
          [Nx.Tensor.t()],
          [Nx.Tensor.t()],
          [Nx.Tensor.t()],
          float()
        ) ::
          {Nx.Tensor.t(), map()}
  def compute_dpo_loss(
        chosen_logprobs,
        rejected_logprobs,
        chosen_ref_logprobs,
        rejected_ref_logprobs,
        dpo_beta
      ) do
    chosen_log_ratio = Nx.subtract(Nx.stack(chosen_logprobs), Nx.stack(chosen_ref_logprobs))
    rejected_log_ratio = Nx.subtract(Nx.stack(rejected_logprobs), Nx.stack(rejected_ref_logprobs))

    diff = Nx.subtract(chosen_log_ratio, rejected_log_ratio)
    logits = Nx.multiply(dpo_beta, diff)
    losses = Nx.negate(Nx.log(Nx.sigmoid(logits)))
    loss = Nx.mean(losses)

    accuracy = Nx.mean(Nx.greater(chosen_log_ratio, rejected_log_ratio))
    chosen_rewards = Nx.multiply(dpo_beta, chosen_log_ratio)
    rejected_rewards = Nx.multiply(dpo_beta, rejected_log_ratio)
    margin = Nx.mean(Nx.subtract(chosen_rewards, rejected_rewards))

    metrics = %{
      "dpo_loss" => Nx.to_number(loss),
      "accuracy" => Nx.to_number(accuracy),
      "margin" => Nx.to_number(margin),
      "chosen_reward" => Nx.to_number(Nx.mean(chosen_rewards)),
      "rejected_reward" => Nx.to_number(Nx.mean(rejected_rewards))
    }

    {loss, metrics}
  end

  @spec main(Config.t()) :: :ok | {:error, term()}
  def main(cfg) do
    cfg = Config.expand_log_path(cfg)

    ml_logger =
      MlLog.setup_logging(cfg.log_path,
        wandb_project: cfg.wandb_project,
        wandb_name: cfg.wandb_name,
        config: cfg
      )

    resume_info = Checkpoint.get_last_checkpoint(cfg.log_path)
    {start_epoch, start_batch} = resume_epoch_batch(resume_info)

    with {:ok, api_key} <- get_api_key(),
         {:ok, service_client} <- create_service_client(api_key, cfg.base_url) do
      run_training(cfg, ml_logger, resume_info, start_epoch, start_batch, service_client)
    end
  end

  defp get_api_key do
    case System.get_env("TINKER_API_KEY") do
      nil ->
        Logger.error("TINKER_API_KEY environment variable is required")
        {:error, :missing_api_key}

      api_key ->
        {:ok, api_key}
    end
  end

  defp create_service_client(api_key, base_url_override) do
    base_url = base_url_override || System.get_env("TINKER_BASE_URL", @default_base_url)
    tinkex_config = Tinkex.Config.new(api_key: api_key, base_url: base_url)
    Tinkex.ServiceClient.start_link(config: tinkex_config)
  end

  defp run_training(cfg, ml_logger, resume_info, start_epoch, start_batch, service_client) do
    {training_client, reference_client} = create_dpo_clients(cfg, resume_info, service_client)
    tokenizer = get_tokenizer(training_client, cfg.model_name)

    {dataset, _maybe_test_dataset} = cfg.dataset_builder.__struct__.build(cfg.dataset_builder)
    n_batches = SupervisedDataset.length(dataset)
    total_steps = n_batches * cfg.num_epochs

    evaluators = Enum.map(cfg.evaluator_builders, fn builder -> builder.() end)

    infrequent_evaluators =
      Enum.map(cfg.infrequent_evaluator_builders, fn builder -> builder.() end)

    Logger.info(
      "Training for #{n_batches} batches x #{cfg.num_epochs} epochs = #{total_steps} steps"
    )

    ctx = %UpdateContext{
      cfg: cfg,
      training_client: training_client,
      reference_client: reference_client,
      evaluators: evaluators,
      infrequent_evaluators: infrequent_evaluators,
      ml_logger: ml_logger,
      log_path: cfg.log_path,
      tokenizer: tokenizer
    }

    _final_dataset = run_epochs(ctx, dataset, start_epoch, start_batch, n_batches, total_steps)

    save_final_checkpoint(training_client, cfg, start_epoch, n_batches)
    MlLog.close(ml_logger)
    Logger.info("DPO training completed successfully")
    :ok
  end

  defp run_epochs(ctx, dataset, start_epoch, start_batch, n_batches, total_steps) do
    Enum.reduce(start_epoch..(ctx.cfg.num_epochs - 1), dataset, fn epoch_idx, dataset_acc ->
      Logger.info("Starting epoch #{epoch_idx}")
      dataset_epoch = SupervisedDataset.set_epoch(dataset_acc, epoch_idx)
      batch_start = if epoch_idx == start_epoch, do: start_batch, else: 0

      Enum.each(batch_start..(n_batches - 1), fn batch_idx ->
        do_update(ctx, epoch_idx, batch_idx, n_batches, total_steps, dataset_epoch)
      end)

      dataset_epoch
    end)
  end

  defp save_final_checkpoint(training_client, cfg, start_epoch, n_batches) do
    if start_epoch < cfg.num_epochs do
      _ =
        Checkpoint.save_checkpoint(
          training_client,
          "final",
          cfg.log_path,
          %{epoch: cfg.num_epochs, batch: n_batches},
          :both
        )
    else
      Logger.info("Training was already complete; nothing to do")
    end
  end

  defp do_update(%UpdateContext{} = ctx, epoch_idx, batch_idx, n_batches, total_steps, dataset) do
    start_time = System.monotonic_time()
    step = epoch_idx * n_batches + batch_idx

    # Initialize metrics and handle checkpointing
    metrics = maybe_save_checkpoint(ctx, step, epoch_idx, batch_idx)

    # Compute learning rate and Adam params
    learning_rate = compute_learning_rate(ctx.cfg, step, total_steps)
    adam_params = build_adam_params(ctx.cfg, learning_rate)

    # Run evaluations
    metrics = maybe_run_evaluations(ctx, step, metrics)

    # Get batch data
    {data, metrics} =
      MiscUtils.timed("get_batch", metrics, fn ->
        SupervisedDataset.get_batch(dataset, batch_idx)
      end)

    {chosen_data, rejected_data} = split_even_odd(data)

    # Log examples for first step
    maybe_log_examples(step, chosen_data, rejected_data, ctx.tokenizer)

    # Compute reference logprobs
    {chosen_ref_logprob_seqs, rejected_ref_logprob_seqs} =
      compute_reference_logprobs(ctx.reference_client, data)

    # Build DPO loss function
    dpo_loss_fn =
      build_dpo_loss_fn(
        chosen_data,
        rejected_data,
        chosen_ref_logprob_seqs,
        rejected_ref_logprob_seqs,
        ctx.cfg.dpo_beta
      )

    # Execute training step
    {dpo_metrics, metrics} =
      execute_training_step(ctx.training_client, data, dpo_loss_fn, adam_params, metrics)

    # Compute final metrics
    num_tokens =
      Enum.reduce(data, 0, fn datum, acc -> acc + ModelInput.length(datum.model_input) end)

    metrics =
      finalize_metrics(
        metrics,
        dpo_metrics,
        chosen_data,
        num_tokens,
        learning_rate,
        step,
        total_steps,
        start_time
      )

    MlLog.log_metrics(ctx.ml_logger, metrics, step)
  end

  defp maybe_save_checkpoint(%UpdateContext{} = ctx, step, epoch_idx, batch_idx) do
    metrics = %{"epoch" => epoch_idx}

    if ctx.cfg.save_every > 0 and step > 0 and rem(step, ctx.cfg.save_every) == 0 do
      {save_result, metrics} =
        MiscUtils.timed("save_checkpoint", metrics, fn ->
          Checkpoint.save_checkpoint(
            ctx.training_client,
            pad_step(step),
            ctx.log_path,
            %{epoch: epoch_idx, batch: batch_idx},
            :both
          )
        end)

      case save_result do
        %{"state_path" => path} -> Map.put(metrics, "state_path", path)
        %{state_path: path} -> Map.put(metrics, "state_path", path)
        _ -> metrics
      end
    else
      metrics
    end
  end

  defp compute_learning_rate(cfg, step, total_steps) do
    lr_schedule = String.to_existing_atom(cfg.lr_schedule)

    cfg.learning_rate *
      LRScheduling.compute_schedule_lr_multiplier(lr_schedule, step, total_steps)
  end

  defp build_adam_params(cfg, learning_rate) do
    %AdamParams{
      learning_rate: learning_rate,
      beta1: cfg.adam_beta1,
      beta2: cfg.adam_beta2,
      eps: cfg.adam_eps
    }
  end

  defp maybe_run_evaluations(%UpdateContext{} = ctx, step, metrics) do
    metrics = maybe_run_regular_evaluations(ctx, step, metrics)
    maybe_run_infrequent_evaluations(ctx, step, metrics)
  end

  defp maybe_run_regular_evaluations(%UpdateContext{} = ctx, step, metrics) do
    if ctx.cfg.eval_every > 0 and rem(step, ctx.cfg.eval_every) == 0 do
      {eval_metrics, metrics} =
        MiscUtils.timed("evals", metrics, fn ->
          run_evaluations(ctx.evaluators, ctx.training_client)
        end)

      Map.merge(metrics, prefix_keys(eval_metrics, "test/"))
    else
      metrics
    end
  end

  defp maybe_run_infrequent_evaluations(%UpdateContext{} = ctx, step, metrics) do
    if ctx.cfg.infrequent_eval_every > 0 and rem(step, ctx.cfg.infrequent_eval_every) == 0 do
      {eval_metrics, metrics} =
        MiscUtils.timed("infrequent_evals", metrics, fn ->
          run_evaluations(ctx.infrequent_evaluators, ctx.training_client)
        end)

      Map.merge(metrics, prefix_keys(eval_metrics, "test/"))
    else
      metrics
    end
  end

  defp maybe_log_examples(step, chosen_data, rejected_data, tokenizer) do
    if step == 0 and not Enum.empty?(chosen_data) do
      max_examples = min(10, length(chosen_data))
      log_example_pairs(chosen_data, rejected_data, tokenizer, max_examples)
    end
  end

  defp log_example_pairs(chosen_data, rejected_data, tokenizer, max_examples) do
    Enum.each(0..(max_examples - 1), fn idx ->
      Logger.info(Display.colorize_example(Enum.at(chosen_data, idx), tokenizer, "weights"))
      Logger.info(Display.colorize_example(Enum.at(rejected_data, idx), tokenizer, "weights"))
    end)
  end

  defp build_dpo_loss_fn(
         chosen_data,
         rejected_data,
         chosen_ref_logprob_seqs,
         rejected_ref_logprob_seqs,
         dpo_beta
       ) do
    fn _data, logprobs_list ->
      {chosen_logprob_seqs, rejected_logprob_seqs} = split_even_odd(logprobs_list)

      chosen_logprobs = compute_weighted_logprobs(chosen_logprob_seqs, chosen_data)
      rejected_logprobs = compute_weighted_logprobs(rejected_logprob_seqs, rejected_data)
      chosen_ref_logprobs = compute_weighted_logprobs_tensor(chosen_ref_logprob_seqs, chosen_data)

      rejected_ref_logprobs =
        compute_weighted_logprobs_tensor(rejected_ref_logprob_seqs, rejected_data)

      compute_dpo_loss(
        chosen_logprobs,
        rejected_logprobs,
        chosen_ref_logprobs,
        rejected_ref_logprobs,
        dpo_beta
      )
    end
  end

  defp compute_weighted_logprobs(logprob_seqs, data) do
    Enum.zip(logprob_seqs, data)
    |> Enum.map(fn {logprob_seq, datum} ->
      weights = TensorData.to_list(datum.loss_fn_inputs["weights"])
      weighted_logprob(logprob_seq, weights)
    end)
  end

  defp compute_weighted_logprobs_tensor(logprob_seqs, data) do
    Enum.zip(logprob_seqs, data)
    |> Enum.map(fn {logprob_seq, datum} ->
      weights = TensorData.to_list(datum.loss_fn_inputs["weights"])
      weighted_logprob(Nx.tensor(logprob_seq), weights)
    end)
  end

  defp execute_training_step(training_client, data, dpo_loss_fn, adam_params, metrics) do
    data_tinkex = Enum.map(data, &TinkexConvert.datum_to_tinkex/1)

    MiscUtils.timed("step", metrics, fn ->
      backward_result =
        await_task!(
          training_client_module(training_client).forward_backward_custom(
            training_client,
            data_tinkex,
            dpo_loss_fn
          )
        )

      dpo_metrics = extract_metrics(backward_result)

      _ =
        await_task!(
          training_client_module(training_client).optim_step(training_client, adam_params)
        )

      dpo_metrics
    end)
  end

  defp finalize_metrics(
         metrics,
         dpo_metrics,
         chosen_data,
         num_tokens,
         learning_rate,
         step,
         total_steps,
         start_time
       ) do
    total_time = System.monotonic_time() - start_time

    metrics
    |> Map.merge(dpo_metrics)
    |> Map.merge(%{
      "num_pairs" => length(chosen_data),
      "num_tokens" => num_tokens,
      "learning_rate" => learning_rate,
      "progress" => progress(step, total_steps),
      "time/total" => System.convert_time_unit(total_time, :native, :millisecond) / 1000
    })
  end

  defp compute_reference_logprobs(reference_client, data) do
    full_sequences = Enum.map(data, &build_full_sequence/1)

    logprobs =
      full_sequences
      |> Enum.map(&TinkexConvert.model_input_to_tinkex/1)
      |> compute_logprobs(reference_client)
      |> Enum.map(&Enum.drop(&1, 1))

    split_even_odd(logprobs)
  end

  defp build_full_sequence(%Datum{} = datum) do
    target_tokens = TensorData.to_list(datum.loss_fn_inputs["target_tokens"])

    case target_tokens do
      [] -> datum.model_input
      _ -> ModelInput.append_int(datum.model_input, List.last(target_tokens))
    end
  end

  defp compute_logprobs(model_inputs, sampling_client) do
    tasks =
      Enum.map(model_inputs, fn model_input ->
        sampling_client_module(sampling_client).compute_logprobs(sampling_client, model_input)
      end)

    tasks
    |> Enum.map(fn
      {:ok, task} -> await_task!(task)
      {:error, reason} -> raise "compute_logprobs failed: #{inspect(reason)}"
    end)
    |> Enum.map(fn
      {:ok, logprobs} -> logprobs
      {:error, reason} -> raise "compute_logprobs failed: #{inspect(reason)}"
      other -> raise "Unexpected compute_logprobs response: #{inspect(other)}"
    end)
  end

  defp weighted_logprob(logprob_seq, weights) do
    weights_tensor = Nx.tensor(weights, type: {:f, 32})
    Nx.dot(logprob_seq, weights_tensor)
  end

  defp split_even_odd(list) do
    Enum.with_index(list)
    |> Enum.reduce({[], []}, fn {item, idx}, {even, odd} ->
      if rem(idx, 2) == 0 do
        {[item | even], odd}
      else
        {even, [item | odd]}
      end
    end)
    |> then(fn {even, odd} -> {Enum.reverse(even), Enum.reverse(odd)} end)
  end

  defp run_evaluations(evaluators, training_client) do
    evaluators
    |> Enum.map(fn evaluator ->
      Task.async(fn ->
        evaluator.__struct__.evaluate(evaluator, training_client)
      end)
    end)
    |> Enum.reduce(%{}, fn task, acc ->
      case Task.await(task, :infinity) do
        {:ok, metrics} -> Map.merge(acc, metrics)
        {:error, reason} -> raise "Evaluation failed: #{inspect(reason)}"
        other -> raise "Unexpected evaluator response: #{inspect(other)}"
      end
    end)
  end

  defp extract_metrics(result) when is_map(result) do
    Map.get(result, :metrics) || Map.get(result, "metrics") || %{}
  end

  defp progress(step, total_steps) when total_steps > 0 do
    step / total_steps
  end

  defp progress(_step, _total_steps), do: 0.0

  defp resume_epoch_batch(nil), do: {0, 0}

  defp resume_epoch_batch(resume_info) do
    epoch = Map.get(resume_info, "epoch") || Map.get(resume_info, :epoch) || 0
    batch = Map.get(resume_info, "batch") || Map.get(resume_info, :batch) || 0
    {epoch, batch}
  end

  defp pad_step(step), do: step |> Integer.to_string() |> String.pad_leading(6, "0")

  defp prefix_keys(metrics, prefix) do
    Map.new(metrics, fn {key, value} -> {prefix <> to_string(key), value} end)
  end

  defp get_tokenizer(training_client, model_name) do
    case training_client_module(training_client).get_tokenizer(training_client) do
      {:ok, tokenizer} ->
        tokenizer

      {:error, _} ->
        case TokenizerUtils.get_tokenizer(model_name) do
          {:ok, tokenizer} -> tokenizer
          {:error, reason} -> raise "Failed to load tokenizer: #{inspect(reason)}"
        end
    end
  end

  defp await_task!(task) do
    case task do
      %Task{} = task ->
        Task.await(task, :infinity)

      {:ok, %Task{} = inner} ->
        await_task!(inner)

      {:ok, value} ->
        value

      {:error, reason} ->
        raise "Task failed: #{inspect(reason)}"

      other ->
        other
    end
  end

  defp training_client_module(%module{}), do: module

  defp sampling_client_module(%module{}), do: module

  defp service_client_module(client) when is_pid(client), do: Tinkex.ServiceClient
  defp service_client_module(%module{}), do: module
end
