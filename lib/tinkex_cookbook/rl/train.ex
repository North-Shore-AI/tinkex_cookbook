# Training loop code is inherently complex with many parameters
# credo:disable-for-this-file Credo.Check.Refactor.FunctionArity
# credo:disable-for-this-file Credo.Check.Refactor.Nesting
# credo:disable-for-this-file Credo.Check.Refactor.CyclomaticComplexity
defmodule TinkexCookbook.RL.Train do
  @moduledoc """
  RL training loop with sync, async, and stream-minibatch modes.
  """

  require Logger

  alias Tinkex.Types.AdamParams
  alias TinkexCookbook.Completers.TinkexTokenCompleter
  alias TinkexCookbook.Display

  alias TinkexCookbook.RL.{
    DataProcessing,
    EnvGroupBuilder,
    Metrics,
    MetricUtil,
    Rollouts,
    TrajectoryGroup
  }

  alias TinkexCookbook.RL.MetricUtil.RLTestSetEvaluator
  alias TinkexCookbook.Types.Datum

  alias TinkexCookbook.Utils.{
    BlockingQueue,
    Checkpoint,
    Logtree,
    MiscUtils,
    MlLog,
    TinkexConvert,
    Trace
  }

  @default_base_url "https://tinker.thinkingmachines.dev/services/tinker-prod"

  require Logtree

  defmodule StreamMinibatchConfig do
    @moduledoc """
    Configuration for training with minibatch streaming.
    """

    use ChzEx.Schema

    chz_schema do
      field(:groups_per_batch, :integer)
      field(:num_minibatches, :integer)
    end
  end

  defmodule AsyncConfig do
    @moduledoc """
    Configuration for async RL training.
    """

    use ChzEx.Schema

    chz_schema do
      field(:max_steps_off_policy, :integer)
      field(:groups_per_batch, :integer)
    end
  end

  defmodule Config do
    @moduledoc """
    RL training configuration.
    """

    use ChzEx.Schema

    chz_schema do
      field(:learning_rate, :float)
      field(:dataset_builder, :any, virtual: true)
      field(:model_name, :string)
      field(:max_tokens, :integer)
      field(:temperature, :float, default: 1.0)
      field(:compute_post_kl, :boolean, default: false)
      field(:evaluator_builders, :any, default: [], virtual: true)
      field(:lora_rank, :integer, default: 32)
      field(:kl_penalty_coef, :float, default: 0.0)
      field(:kl_discount_factor, :float, default: 0.0)
      field(:loss_fn, :string, default: "importance_sampling")
      field(:num_substeps, :integer, default: 1)
      field(:wandb_project, :string, default: nil)
      field(:wandb_name, :string, default: nil)
      field(:log_path, :string)
      field(:base_url, :string, default: nil)
      field(:enable_trace, :boolean, default: false)
      field(:remove_constant_reward_groups, :boolean, default: false)
      field(:eval_every, :integer, default: 20)
      field(:save_every, :integer, default: 20)
      field(:load_checkpoint_path, :string, default: nil)
      field(:async_config, :any, default: nil, virtual: true)
      field(:stream_minibatch_config, :any, default: nil, virtual: true)
      field(:num_groups_to_log, :integer, default: 4)
    end

    @type t() :: %__MODULE__{}

    @spec expand_log_path(t()) :: t()
    def expand_log_path(%__MODULE__{log_path: log_path} = config) do
      %{config | log_path: Path.expand(log_path)}
    end
  end

  defmodule WrappedTrajectoryGroup do
    @moduledoc """
    Wrapper for trajectory groups with sampling metadata.
    """

    defstruct [:trajectory_group, :env_group_builder, :sampling_client_step, metrics: %{}]
  end

  @spec train_step([Datum.t()], struct(), float(), pos_integer(), String.t()) :: [[float()]]
  def train_step(data, training_client, learning_rate, num_substeps, loss_fn) do
    batches =
      case data do
        [] -> []
        _ -> MiscUtils.split_list(data, min(num_substeps, length(data)))
      end

    if batches == [] do
      []
    else
      adam_params = %AdamParams{
        learning_rate: learning_rate,
        beta1: 0.9,
        beta2: 0.95,
        eps: 1.0e-8
      }

      training_logprobs = []

      fwd_task =
        enqueue_forward_backward(
          training_client,
          Enum.at(batches, 0)
          |> Enum.map(&remove_mask/1)
          |> Enum.map(&TinkexConvert.datum_to_tinkex/1),
          loss_fn
        )

      optim_task = enqueue_optim_step(training_client, adam_params)

      {training_logprobs, _} =
        Enum.reduce(0..(length(batches) - 1), {training_logprobs, {fwd_task, optim_task}}, fn
          i, {logprobs_acc, {current_fwd, current_optim}} ->
            {next_fwd, next_optim} =
              if i + 1 < length(batches) do
                next_batch =
                  Enum.at(batches, i + 1)
                  |> Enum.map(&remove_mask/1)
                  |> Enum.map(&TinkexConvert.datum_to_tinkex/1)

                {enqueue_forward_backward(training_client, next_batch, loss_fn),
                 enqueue_optim_step(training_client, adam_params)}
              else
                {nil, nil}
              end

            fwd_result = await_task!(current_fwd)
            logprobs_acc = logprobs_acc ++ training_logprobs_from_fwd_bwd(fwd_result)
            _ = await_task!(current_optim)

            {logprobs_acc, {next_fwd, next_optim}}
        end)

      training_logprobs
    end
  end

  @spec do_sync_training(
          non_neg_integer(),
          non_neg_integer(),
          non_neg_integer(),
          Config.t(),
          term(),
          term(),
          [struct()],
          struct(),
          MlLog.Logger.t(),
          term()
        ) :: :ok
  def do_sync_training(
        start_batch,
        end_batch,
        num_batches,
        cfg,
        training_client,
        service_client,
        evaluators,
        dataset,
        ml_logger,
        tokenizer
      ) do
    {sampling_client, _} =
      save_checkpoint_and_get_sampling_client(
        training_client,
        start_batch,
        cfg.log_path,
        cfg.save_every,
        start_batch
      )

    _final_sampling_client =
      Enum.reduce(start_batch..(end_batch - 1), sampling_client, fn i_batch,
                                                                    sampling_client_acc ->
        metrics = %{
          "progress/batch" => i_batch,
          "optim/lr" => cfg.learning_rate,
          "progress/done_frac" => (i_batch + 1) / num_batches
        }

        t_start = System.monotonic_time(:millisecond)

        {metrics, sampling_client_acc} =
          if cfg.eval_every > 0 and rem(i_batch, cfg.eval_every) == 0 do
            {eval_metrics, updated_metrics} =
              MiscUtils.timed("run_evals", metrics, fn ->
                run_evaluations_parallel(evaluators, sampling_client_acc, cfg, i_batch)
              end)

            {Map.merge(updated_metrics, eval_metrics), sampling_client_acc}
          else
            {metrics, sampling_client_acc}
          end

        env_group_builders = dataset.__struct__.get_batch(dataset, i_batch)

        trajectory_groups =
          with_logtree_scope(
            cfg.log_path,
            cfg.num_groups_to_log,
            "train_iteration_#{pad_batch(i_batch)}",
            "RL Iteration #{i_batch}",
            fn ->
              env_group_builders
              |> Enum.with_index()
              |> Task.async_stream(
                fn {builder, idx} ->
                  do_group_rollout_and_filter_constant_reward(
                    sampling_client_acc,
                    builder,
                    cfg.max_tokens,
                    cfg.temperature,
                    false,
                    idx < cfg.num_groups_to_log
                  )
                end,
                ordered: true,
                timeout: :infinity
              )
              |> Enum.map(fn {:ok, group} -> group end)
            end
          )

        trajectory_groups =
          if cfg.remove_constant_reward_groups do
            DataProcessing.remove_constant_reward_groups(trajectory_groups)
          else
            trajectory_groups
          end

        {sampling_client_next, train_step_metrics} =
          do_train_step_and_get_sampling_client(
            cfg,
            i_batch,
            training_client,
            service_client,
            tokenizer,
            env_group_builders,
            trajectory_groups
          )

        metrics = Map.merge(metrics, train_step_metrics)

        metrics =
          Map.put(metrics, "time/total", (System.monotonic_time(:millisecond) - t_start) / 1000)

        MlLog.log_metrics(ml_logger, metrics, i_batch)

        sampling_client_next
      end)

    :ok
  end

  @spec do_sync_training_with_stream_minibatch(
          non_neg_integer(),
          non_neg_integer(),
          non_neg_integer(),
          Config.t(),
          term(),
          term(),
          [struct()],
          struct(),
          MlLog.Logger.t(),
          term()
        ) :: :ok
  def do_sync_training_with_stream_minibatch(
        start_batch,
        end_batch,
        num_batches,
        cfg,
        training_client,
        service_client,
        evaluators,
        dataset,
        ml_logger,
        tokenizer
      ) do
    {sampling_client, _} =
      save_checkpoint_and_get_sampling_client(
        training_client,
        start_batch,
        cfg.log_path,
        cfg.save_every,
        start_batch
      )

    _final_sampling_client =
      Enum.reduce(start_batch..(end_batch - 1), sampling_client, fn i_batch,
                                                                    sampling_client_acc ->
        metrics = %{
          "progress/batch" => i_batch,
          "optim/lr" => cfg.learning_rate,
          "progress/done_frac" => (i_batch + 1) / num_batches
        }

        t_start = System.monotonic_time(:millisecond)

        {metrics, sampling_client_acc} =
          if (cfg.eval_every > 0 and rem(i_batch, cfg.eval_every) == 0) or
               i_batch == end_batch - 1 do
            {eval_metrics, updated_metrics} =
              MiscUtils.timed("run_evals", metrics, fn ->
                run_evaluations_parallel(evaluators, sampling_client_acc, cfg, i_batch)
              end)

            {Map.merge(updated_metrics, eval_metrics), sampling_client_acc}
          else
            {metrics, sampling_client_acc}
          end

        trajectory_groups_queue = start_queue()
        env_group_builders = dataset.__struct__.get_batch(dataset, i_batch)

        with_logtree_scope(
          cfg.log_path,
          cfg.num_groups_to_log,
          "train_iteration_#{pad_batch(i_batch)}",
          "RL Iteration #{i_batch}",
          fn ->
            Enum.each(Enum.with_index(env_group_builders), fn {builder, idx} ->
              Task.start(fn ->
                t_start = System.monotonic_time(:millisecond)

                trajectory_group =
                  do_group_rollout_and_filter_constant_reward(
                    sampling_client_acc,
                    builder,
                    cfg.max_tokens,
                    cfg.temperature,
                    cfg.remove_constant_reward_groups,
                    idx < cfg.num_groups_to_log
                  )

                metrics = %{
                  "time/trajectory_group_worker_loop/total" =>
                    (System.monotonic_time(:millisecond) - t_start) / 1000
                }

                if trajectory_group != nil do
                  BlockingQueue.push(
                    trajectory_groups_queue,
                    %WrappedTrajectoryGroup{
                      trajectory_group: trajectory_group,
                      env_group_builder: builder,
                      sampling_client_step: i_batch,
                      metrics: metrics
                    }
                  )
                else
                  BlockingQueue.push(trajectory_groups_queue, nil)
                end
              end)
            end)
          end
        )

        {sampling_client_next, full_batch_metrics} =
          do_train_step_streaming_and_get_sampling_client(
            cfg,
            i_batch,
            trajectory_groups_queue,
            training_client,
            service_client,
            tokenizer
          )

        metrics = Map.merge(metrics, full_batch_metrics)

        metrics =
          Map.put(metrics, "time/total", (System.monotonic_time(:millisecond) - t_start) / 1000)

        MlLog.log_metrics(ml_logger, metrics, i_batch)
        sampling_client_next
      end)

    :ok
  end

  @spec do_async_training(
          non_neg_integer(),
          non_neg_integer(),
          non_neg_integer(),
          Config.t(),
          term(),
          term(),
          [struct()],
          struct(),
          MlLog.Logger.t(),
          term()
        ) :: :ok
  def do_async_training(
        start_batch,
        end_batch,
        num_batches,
        cfg,
        training_client,
        service_client,
        evaluators,
        dataset,
        ml_logger,
        tokenizer
      ) do
    async_cfg = cfg.async_config

    env_group_builders_queue = start_queue()
    trajectory_groups_queue = start_queue()
    sampling_updates_queue = start_queue()

    path_dict =
      await_task!(
        Checkpoint.save_checkpoint_async(
          training_client,
          pad_batch(start_batch),
          cfg.log_path,
          %{batch: start_batch},
          :both
        )
      )

    sampling_client =
      await_task!(
        training_client_module(training_client).create_sampling_client_async(
          training_client,
          path_dict["sampler_path"]
        )
      )

    {:ok, sampling_state} =
      Agent.start_link(fn ->
        %{client: sampling_client, step: start_batch}
      end)

    shutdown = fn ->
      Enum.each(1..async_cfg.groups_per_batch, fn _ ->
        BlockingQueue.push(env_group_builders_queue, :shutdown)
      end)

      BlockingQueue.push(sampling_updates_queue, :shutdown)
    end

    dataloader_task =
      Task.async(fn ->
        Enum.each(start_batch..(end_batch - 1), fn i_batch ->
          env_group_builders = dataset.__struct__.get_batch(dataset, i_batch)
          Enum.each(env_group_builders, &BlockingQueue.push(env_group_builders_queue, &1))
        end)
      end)

    worker_tasks =
      Enum.map(1..async_cfg.groups_per_batch, fn _ ->
        Task.async(fn ->
          loop_trajectory_workers(
            env_group_builders_queue,
            trajectory_groups_queue,
            sampling_state,
            cfg
          )
        end)
      end)

    training_task =
      Task.async(fn ->
        loop_training(
          start_batch,
          end_batch,
          num_batches,
          cfg,
          training_client,
          service_client,
          tokenizer,
          env_group_builders_queue,
          trajectory_groups_queue,
          sampling_updates_queue,
          sampling_state,
          ml_logger,
          shutdown
        )
      end)

    evaluation_task =
      Task.async(fn ->
        loop_evaluations(
          evaluators,
          cfg,
          sampling_updates_queue,
          ml_logger
        )
      end)

    Task.await(dataloader_task, :infinity)
    Enum.each(worker_tasks, &Task.await(&1, :infinity))
    Task.await(training_task, :infinity)
    Task.await(evaluation_task, :infinity)

    Agent.stop(sampling_state)
    :ok
  end

  @spec main(struct()) :: :ok | {:error, term()}
  def main(cfg) do
    cfg = Config.expand_log_path(cfg)

    ml_logger =
      MlLog.setup_logging(cfg.log_path,
        wandb_project: cfg.wandb_project,
        config: cfg,
        wandb_name: cfg.wandb_name
      )

    if cfg.enable_trace do
      trace_path = Path.join(cfg.log_path, "trace_events.jsonl")
      Logger.info("Tracing is enabled. Trace events will be saved to #{trace_path}")
      Trace.trace_init(output_file: trace_path)
    end

    api_key = System.get_env("TINKER_API_KEY")

    if is_nil(api_key) do
      Logger.error("TINKER_API_KEY environment variable is required")
      {:error, :missing_api_key}
    else
      base_url = cfg.base_url || System.get_env("TINKER_BASE_URL", @default_base_url)
      tinkex_config = Tinkex.Config.new(api_key: api_key, base_url: base_url)

      with {:ok, service_client} <- Tinkex.ServiceClient.start_link(config: tinkex_config),
           {:ok, training_client} <- create_training_client(service_client, cfg),
           {:ok, tokenizer} <- Tinkex.TrainingClient.get_tokenizer(training_client) do
        {dataset, maybe_test_dataset} = cfg.dataset_builder.__struct__.build(cfg.dataset_builder)

        evaluators =
          Enum.map(cfg.evaluator_builders, fn builder ->
            builder.()
          end)

        evaluators =
          if maybe_test_dataset != nil do
            evaluators ++ [RLTestSetEvaluator.new(maybe_test_dataset, cfg.max_tokens)]
          else
            evaluators
          end

        num_batches = dataset.__struct__.length(dataset)

        training_fun =
          cond do
            cfg.async_config != nil -> &do_async_training/10
            cfg.stream_minibatch_config != nil -> &do_sync_training_with_stream_minibatch/10
            true -> &do_sync_training/10
          end

        training_fun.(
          start_batch(cfg),
          num_batches,
          num_batches,
          cfg,
          training_client,
          service_client,
          evaluators,
          dataset,
          ml_logger,
          tokenizer
        )

        if start_batch(cfg) < num_batches do
          _ =
            Checkpoint.save_checkpoint(
              training_client,
              "final",
              cfg.log_path,
              %{batch: num_batches},
              :both
            )
        else
          Logger.info("Training was already complete; nothing to do")
        end

        MlLog.close(ml_logger)
        Logger.info("Training completed successfully")
        :ok
      else
        {:error, reason} ->
          Logger.error("Training failed: #{inspect(reason)}")
          {:error, reason}
      end
    end
  end

  # Helpers

  defp start_batch(cfg) do
    case Checkpoint.get_last_checkpoint(cfg.log_path) do
      nil -> 0
      %{"batch" => batch} -> batch
      %{batch: batch} -> batch
      _ -> 0
    end
  end

  defp create_training_client(service_client, cfg) do
    resume_info = Checkpoint.get_last_checkpoint(cfg.log_path)

    cond do
      resume_info != nil ->
        path = resume_info["state_path"] || resume_info[:state_path]

        await_task!(
          Tinkex.ServiceClient.create_training_client_from_state_with_optimizer_async(
            service_client,
            path
          )
        )

      cfg.load_checkpoint_path != nil ->
        await_task!(
          Tinkex.ServiceClient.create_training_client_from_state_async(
            service_client,
            cfg.load_checkpoint_path
          )
        )

      true ->
        lora_config = %Tinkex.Types.LoraConfig{rank: cfg.lora_rank}

        await_task!(
          Tinkex.ServiceClient.create_lora_training_client_async(
            service_client,
            cfg.model_name,
            lora_config: lora_config,
            call_timeout: :infinity
          )
        )
    end
    |> normalize_ok_tuple()
  end

  defp run_evaluations_parallel(evaluators, sampling_client, cfg, i_batch) do
    evaluators
    |> Task.async_stream(
      fn evaluator ->
        _ev_name = get_evaluator_name(evaluator)
        run_single_evaluation(evaluator, cfg, i_batch, sampling_client)
      end,
      ordered: true,
      timeout: :infinity
    )
    |> Enum.reduce(%{}, fn {:ok, metrics}, acc -> Map.merge(acc, metrics) end)
  end

  defp run_single_evaluation(evaluator, cfg, i_batch, sampling_client) do
    ev_name = get_evaluator_name(evaluator)

    with_logtree_scope(
      cfg.log_path,
      cfg.num_groups_to_log,
      "eval_#{ev_name}_iteration_#{pad_batch(i_batch)}",
      "Running evaluation #{ev_name} #{i_batch}",
      fn ->
        case evaluator.__struct__.evaluate(evaluator, sampling_client) do
          {:ok, metrics} -> metrics
          {:error, reason} -> %{"error" => inspect(reason)}
        end
      end
    )
  end

  defp get_evaluator_name(evaluator) do
    case Map.fetch(evaluator, :name) do
      {:ok, name} when is_binary(name) -> name
      _ -> ""
    end
  end

  # Suppress dialyzer warning about dead code - the num_inds == 1 check prevents division by zero
  # even though it's not currently exercised by any call site
  @dialyzer {:nowarn_function, select_representative_inds: 2}
  defp select_representative_inds(scores, num_inds) do
    sorted_inds =
      scores
      |> Enum.with_index()
      |> Enum.sort_by(fn {score, _idx} -> score end)
      |> Enum.map(fn {_score, idx} -> idx end)

    positions =
      if num_inds == 1 do
        [0]
      else
        Enum.map(0..(num_inds - 1), fn i ->
          trunc(i * (length(sorted_inds) - 1) / (num_inds - 1))
        end)
      end

    Enum.map(positions, &Enum.at(sorted_inds, &1))
  end

  defp print_group(traj_group, tokenizer) do
    max_trajs_to_print = 4

    traj_group =
      if length(traj_group.trajectories_G) > max_trajs_to_print do
        inds =
          select_representative_inds(
            TrajectoryGroup.get_total_rewards(traj_group),
            max_trajs_to_print
          )

        %TrajectoryGroup{
          trajectories_G: Enum.map(inds, &Enum.at(traj_group.trajectories_G, &1)),
          final_rewards_G: Enum.map(inds, &Enum.at(traj_group.final_rewards_G, &1)),
          metrics_G: Enum.map(inds, &Enum.at(traj_group.metrics_G, &1))
        }
      else
        traj_group
      end

    rewards = TrajectoryGroup.get_total_rewards(traj_group)
    advantages = DataProcessing.compute_advantages([traj_group])
    {data, metadata} = DataProcessing.assemble_training_data([traj_group], advantages)

    buf =
      data
      |> Enum.zip(metadata)
      |> Enum.reduce({"", nil}, fn {datum, meta}, {acc, last_meta} ->
        idx = meta[:traj_idx]

        acc =
          if meta != last_meta do
            acc <>
              "\n****** trajectory idx=#{idx}, reward=#{Float.round(Enum.at(rewards, idx), 3)} ******\n"
          else
            acc
          end

        acc = acc <> "---- datum ----\n"
        acc = acc <> Display.colorize_example(datum, tokenizer, "advantages") <> "\n"
        {acc, meta}
      end)
      |> elem(0)

    Logger.info("\n====== Trajectory Group ======\n#{buf}====== End Trajectory Group ======")
  end

  defp remove_mask(%Datum{} = datum) do
    updated_inputs = Map.delete(datum.loss_fn_inputs, "mask")
    %Datum{datum | loss_fn_inputs: updated_inputs}
  end

  defp training_logprobs_from_fwd_bwd(fwd_bwd_result) do
    outputs =
      fwd_bwd_result.loss_fn_outputs ||
        fwd_bwd_result[:loss_fn_outputs] ||
        []

    Enum.map(outputs, fn output ->
      logprobs = output["logprobs"] || output[:logprobs]

      cond do
        match?(%Tinkex.Types.TensorData{}, logprobs) ->
          Tinkex.Types.TensorData.tolist(logprobs)

        match?(%TinkexCookbook.Types.TensorData{}, logprobs) ->
          TinkexCookbook.Types.TensorData.to_list(logprobs)

        is_list(logprobs) ->
          logprobs

        true ->
          []
      end
    end)
  end

  @doc false
  def do_group_rollout_and_filter_constant_reward(
        sampling_client,
        env_group_builder,
        max_tokens,
        temperature,
        remove_constant,
        enable_logging \\ true
      ) do
    policy =
      TinkexTokenCompleter.new(
        sampling_client: sampling_client,
        max_tokens: max_tokens,
        temperature: temperature
      )

    trajectory_group =
      Logtree.optional_enable_logging enable: enable_logging do
        Rollouts.do_group_rollout(env_group_builder, policy)
      end

    if remove_constant and MiscUtils.all_same(TrajectoryGroup.get_total_rewards(trajectory_group)) do
      nil
    else
      trajectory_group
    end
  end

  @doc false
  def save_checkpoint_and_get_sampling_client(
        training_client,
        i_batch,
        log_path,
        save_every,
        start_batch \\ 0
      ) do
    metrics = %{}

    {sampling_client, metrics} =
      if save_every > 0 and i_batch > start_batch and rem(i_batch, save_every) == 0 do
        {path_dict, metrics} =
          MiscUtils.timed("save_checkpoint", metrics, fn ->
            await_task!(
              Checkpoint.save_checkpoint_async(
                training_client,
                pad_batch(i_batch),
                log_path,
                %{batch: i_batch},
                :both
              )
            )
          end)

        {await_task!(
           training_client_module(training_client).create_sampling_client_async(
             training_client,
             path_dict["sampler_path"]
           )
         ), metrics}
      else
        {await_task!(
           training_client_module(training_client).save_weights_and_get_sampling_client(
             training_client
           )
         ), metrics}
      end

    {sampling_client, metrics}
  end

  defp prepare_minibatch(
         env_group_builders,
         trajectory_groups,
         tokenizer,
         service_client,
         model_name,
         kl_penalty_coef,
         kl_discount_factor
       ) do
    metrics = %{}
    taglists = Enum.map(env_group_builders, &EnvGroupBuilder.logging_tags/1)

    metrics =
      Map.merge(metrics, MetricUtil.compute_trajectory_metrics(trajectory_groups, taglists))

    Enum.each(Enum.take(trajectory_groups, 2), &print_group(&1, tokenizer))

    {data, metrics} =
      MiscUtils.timed("assemble_training_data", metrics, fn ->
        advantages = DataProcessing.compute_advantages(trajectory_groups)
        {data, _metadata} = DataProcessing.assemble_training_data(trajectory_groups, advantages)
        data
      end)

    {data, metrics} =
      if kl_penalty_coef > 0 do
        {kl_result, metrics} =
          MiscUtils.timed("kl_vs_base", metrics, fn ->
            base_sampling_client =
              await_task!(
                service_client_module(service_client).create_sampling_client_async(service_client,
                  base_model: model_name
                )
              )

            Metrics.incorporate_kl_penalty(
              data,
              base_sampling_client,
              kl_penalty_coef,
              kl_discount_factor
            )
          end)

        {updated_data, kl_metrics} = kl_result
        {updated_data, Map.merge(metrics, kl_metrics)}
      else
        {data, metrics}
      end

    {data, metrics}
  end

  @doc false
  def compute_full_batch_metrics_and_get_sampling_client(
        training_client,
        i_batch,
        data,
        training_logprobs,
        log_path,
        save_every,
        do_compute_post_kl
      ) do
    metrics = %{}

    {kl_metrics, metrics} =
      MiscUtils.timed("compute_kl_sample_train", metrics, fn ->
        Metrics.compute_kl_sample_train(data, training_logprobs)
      end)

    metrics = Map.merge(metrics, kl_metrics)

    {sampling_client, checkpoint_metrics} =
      save_checkpoint_and_get_sampling_client(training_client, i_batch, log_path, save_every)

    metrics = Map.merge(metrics, checkpoint_metrics)

    metrics =
      if do_compute_post_kl do
        {post_metrics, metrics} =
          MiscUtils.timed("compute_post_kl", metrics, fn ->
            Metrics.compute_post_kl(data, sampling_client)
          end)

        Map.merge(metrics, post_metrics)
      else
        metrics
      end

    {sampling_client, metrics}
  end

  defp do_train_step_streaming_and_get_sampling_client(
         cfg,
         i_batch,
         trajectory_groups_queue,
         training_client,
         service_client,
         tokenizer,
         trajectory_group_filter \\ fn _ -> true end
       ) do
    stream_cfg = cfg.stream_minibatch_config

    if rem(stream_cfg.groups_per_batch, cfg.num_substeps) != 0 do
      raise ArgumentError,
            "stream_minibatch groups_per_batch must be divisible by num_substeps"
    end

    groups_per_substep = div(stream_cfg.groups_per_batch, cfg.num_substeps)

    if rem(groups_per_substep, stream_cfg.num_minibatches) != 0 do
      raise ArgumentError,
            "groups_per_substep must be divisible by num_minibatches"
    end

    groups_per_minibatch = div(groups_per_substep, stream_cfg.num_minibatches)

    Trace.update_scope_context(%{"step" => i_batch})

    metrics = %{}
    all_data = []
    all_training_logprobs = []
    all_wrapped_groups = []

    {all_data, all_training_logprobs, all_wrapped_groups, metrics} =
      Enum.reduce(
        0..(cfg.num_substeps - 1),
        {all_data, all_training_logprobs, all_wrapped_groups, metrics},
        fn
          _i_substep, {data_acc, logprobs_acc, groups_acc, metrics_acc} ->
            {data_acc, logprobs_acc, groups_acc, metrics_acc, forward_tasks} =
              Enum.reduce(
                0..(stream_cfg.num_minibatches - 1),
                {data_acc, logprobs_acc, groups_acc, metrics_acc, []},
                fn
                  _i_mb, {d_acc, l_acc, g_acc, m_acc, tasks_acc} ->
                    wrapped_groups =
                      collect_groups(
                        trajectory_groups_queue,
                        groups_per_minibatch,
                        trajectory_group_filter
                      )

                    wrapped_groups = Enum.filter(wrapped_groups, &(&1 != nil))

                    if wrapped_groups == [] do
                      {d_acc, l_acc, g_acc, m_acc, tasks_acc}
                    else
                      {data, prep_metrics} =
                        prepare_minibatch(
                          Enum.map(wrapped_groups, & &1.env_group_builder),
                          Enum.map(wrapped_groups, & &1.trajectory_group),
                          tokenizer,
                          service_client,
                          cfg.model_name,
                          cfg.kl_penalty_coef,
                          cfg.kl_discount_factor
                        )

                      m_acc = Map.merge(m_acc, prep_metrics)

                      tinkex_data =
                        data
                        |> Enum.map(&remove_mask/1)
                        |> Enum.map(&TinkexConvert.datum_to_tinkex/1)

                      fwd_task =
                        enqueue_forward_backward(training_client, tinkex_data, cfg.loss_fn)

                      {d_acc ++ data, l_acc, g_acc ++ wrapped_groups, m_acc,
                       tasks_acc ++ [fwd_task]}
                    end
                end
              )

            optim_task =
              enqueue_optim_step(
                training_client,
                %AdamParams{
                  learning_rate: cfg.learning_rate,
                  beta1: 0.9,
                  beta2: 0.95,
                  eps: 1.0e-8
                }
              )

            logprobs_acc =
              logprobs_acc ++
                Enum.flat_map(forward_tasks, fn task ->
                  training_logprobs_from_fwd_bwd(await_task!(task))
                end)

            _ = await_task!(optim_task)

            {data_acc, logprobs_acc, groups_acc, metrics_acc}
        end
      )

    metrics =
      metrics
      |> Map.merge(Metrics.compute_sampling_client_metrics(all_wrapped_groups))
      |> Map.merge(
        MetricUtil.compute_trajectory_metrics(
          Enum.map(all_wrapped_groups, & &1.trajectory_group),
          Enum.map(all_wrapped_groups, fn group ->
            EnvGroupBuilder.logging_tags(group.env_group_builder)
          end)
        )
      )

    {sampling_client, full_batch_metrics} =
      compute_full_batch_metrics_and_get_sampling_client(
        training_client,
        i_batch + 1,
        all_data,
        all_training_logprobs,
        cfg.log_path,
        cfg.save_every,
        cfg.compute_post_kl
      )

    {sampling_client, Map.merge(metrics, full_batch_metrics)}
  end

  defp do_train_step_and_get_sampling_client(
         cfg,
         i_batch,
         training_client,
         service_client,
         tokenizer,
         env_group_builders,
         trajectory_groups
       ) do
    Trace.update_scope_context(%{"step" => i_batch})

    {data, prep_metrics} =
      prepare_minibatch(
        env_group_builders,
        trajectory_groups,
        tokenizer,
        service_client,
        cfg.model_name,
        cfg.kl_penalty_coef,
        cfg.kl_discount_factor
      )

    {training_logprobs, metrics} =
      MiscUtils.timed("train", prep_metrics, fn ->
        train_step(
          data,
          training_client,
          cfg.learning_rate,
          cfg.num_substeps,
          cfg.loss_fn
        )
      end)

    {sampling_client, full_metrics} =
      compute_full_batch_metrics_and_get_sampling_client(
        training_client,
        i_batch + 1,
        data,
        training_logprobs,
        cfg.log_path,
        cfg.save_every,
        cfg.compute_post_kl
      )

    {sampling_client, Map.merge(metrics, full_metrics)}
  end

  defp loop_trajectory_workers(
         env_group_builders_queue,
         trajectory_groups_queue,
         sampling_state,
         cfg
       ) do
    case BlockingQueue.pop(env_group_builders_queue) do
      :shutdown ->
        :ok

      env_group_builder ->
        %{client: sampling_client, step: sampling_client_step} = Agent.get(sampling_state, & &1)

        t_start = System.monotonic_time(:millisecond)

        trajectory_group =
          do_group_rollout_and_filter_constant_reward(
            sampling_client,
            env_group_builder,
            cfg.max_tokens,
            cfg.temperature,
            cfg.remove_constant_reward_groups
          )

        metrics = %{
          "time/trajectory_group_worker_loop/total" =>
            (System.monotonic_time(:millisecond) - t_start) / 1000
        }

        if trajectory_group != nil do
          BlockingQueue.push(
            trajectory_groups_queue,
            %WrappedTrajectoryGroup{
              trajectory_group: trajectory_group,
              env_group_builder: env_group_builder,
              sampling_client_step: sampling_client_step,
              metrics: metrics
            }
          )
        else
          BlockingQueue.push(trajectory_groups_queue, nil)
        end

        loop_trajectory_workers(
          env_group_builders_queue,
          trajectory_groups_queue,
          sampling_state,
          cfg
        )
    end
  end

  defp loop_training(
         start_batch,
         end_batch,
         num_batches,
         cfg,
         training_client,
         service_client,
         tokenizer,
         env_group_builders_queue,
         trajectory_groups_queue,
         sampling_updates_queue,
         sampling_state,
         ml_logger,
         shutdown_fun
       ) do
    loop_training_inner(
      start_batch,
      end_batch,
      num_batches,
      cfg,
      training_client,
      service_client,
      tokenizer,
      env_group_builders_queue,
      trajectory_groups_queue,
      sampling_updates_queue,
      sampling_state,
      ml_logger,
      shutdown_fun,
      []
    )
  end

  defp loop_training_inner(
         i_batch,
         end_batch,
         num_batches,
         cfg,
         training_client,
         service_client,
         tokenizer,
         env_group_builders_queue,
         trajectory_groups_queue,
         sampling_updates_queue,
         sampling_state,
         ml_logger,
         shutdown_fun,
         wrapped_groups
       ) do
    async_cfg = cfg.async_config

    if i_batch >= end_batch do
      shutdown_fun.()
      :ok
    else
      wrapped_group = BlockingQueue.pop(trajectory_groups_queue)

      wrapped_groups =
        if wrapped_group == nil do
          wrapped_groups
        else
          wrapped_groups ++ [wrapped_group]
        end

      wrapped_groups =
        Enum.filter(wrapped_groups, fn group ->
          if group == nil do
            false
          else
            if i_batch - group.sampling_client_step > async_cfg.max_steps_off_policy do
              Logger.info("[training_loop] Step #{i_batch}: Samples are too stale, skipping")
              BlockingQueue.push(env_group_builders_queue, group.env_group_builder)
              false
            else
              true
            end
          end
        end)

      if cfg.stream_minibatch_config != nil do
        t_start = System.monotonic_time(:millisecond)

        if wrapped_group != nil do
          BlockingQueue.push(trajectory_groups_queue, wrapped_group)
        end

        trajectory_group_filter = fn group ->
          if group == nil do
            false
          else
            if i_batch - group.sampling_client_step > async_cfg.max_steps_off_policy do
              Logger.info("[training_loop] Step #{i_batch}: Samples are too stale, skipping")
              BlockingQueue.push(env_group_builders_queue, group.env_group_builder)
              false
            else
              true
            end
          end
        end

        {sampling_client, train_metrics} =
          do_train_step_streaming_and_get_sampling_client(
            cfg,
            i_batch,
            trajectory_groups_queue,
            training_client,
            service_client,
            tokenizer,
            trajectory_group_filter
          )

        Agent.update(sampling_state, fn _ -> %{client: sampling_client, step: i_batch + 1} end)
        BlockingQueue.push(sampling_updates_queue, %{client: sampling_client, step: i_batch + 1})

        metrics =
          %{
            "training_client/step" => i_batch,
            "optim/lr" => cfg.learning_rate,
            "progress/done_frac" => (i_batch + 1) / num_batches
          }
          |> Map.merge(train_metrics)
          |> Map.put(
            "time/training_loop/total",
            (System.monotonic_time(:millisecond) - t_start) / 1000
          )

        MlLog.log_metrics(ml_logger, metrics, i_batch)

        loop_training_inner(
          i_batch + 1,
          end_batch,
          num_batches,
          cfg,
          training_client,
          service_client,
          tokenizer,
          env_group_builders_queue,
          trajectory_groups_queue,
          sampling_updates_queue,
          sampling_state,
          ml_logger,
          shutdown_fun,
          []
        )
      else
        if length(wrapped_groups) < async_cfg.groups_per_batch do
          loop_training_inner(
            i_batch,
            end_batch,
            num_batches,
            cfg,
            training_client,
            service_client,
            tokenizer,
            env_group_builders_queue,
            trajectory_groups_queue,
            sampling_updates_queue,
            sampling_state,
            ml_logger,
            shutdown_fun,
            wrapped_groups
          )
        else
          t_start = System.monotonic_time(:millisecond)

          metrics =
            %{
              "training_client/step" => i_batch,
              "optim/lr" => cfg.learning_rate,
              "progress/done_frac" => (i_batch + 1) / num_batches
            }
            |> Map.merge(Metrics.compute_sampling_client_metrics(wrapped_groups))

          {sampling_client, train_metrics} =
            do_train_step_and_get_sampling_client(
              cfg,
              i_batch,
              training_client,
              service_client,
              tokenizer,
              Enum.map(wrapped_groups, & &1.env_group_builder),
              Enum.map(wrapped_groups, & &1.trajectory_group)
            )

          Agent.update(sampling_state, fn _ -> %{client: sampling_client, step: i_batch + 1} end)

          BlockingQueue.push(sampling_updates_queue, %{client: sampling_client, step: i_batch + 1})

          metrics = Map.merge(metrics, train_metrics)

          metrics =
            Map.put(
              metrics,
              "time/training_loop/total",
              (System.monotonic_time(:millisecond) - t_start) / 1000
            )

          MlLog.log_metrics(ml_logger, metrics, i_batch)

          loop_training_inner(
            i_batch + 1,
            end_batch,
            num_batches,
            cfg,
            training_client,
            service_client,
            tokenizer,
            env_group_builders_queue,
            trajectory_groups_queue,
            sampling_updates_queue,
            sampling_state,
            ml_logger,
            shutdown_fun,
            []
          )
        end
      end
    end
  end

  defp loop_evaluations(evaluators, cfg, sampling_updates_queue, ml_logger) do
    if evaluators == [] or cfg.eval_every == 0 do
      :ok
    else
      loop_evaluations_inner(evaluators, cfg, sampling_updates_queue, ml_logger)
    end
  end

  defp loop_evaluations_inner(evaluators, cfg, sampling_updates_queue, ml_logger) do
    case BlockingQueue.pop(sampling_updates_queue) do
      :shutdown ->
        :ok

      %{client: sampling_client, step: step} ->
        if cfg.eval_every > 0 and rem(step, cfg.eval_every) == 0 do
          metrics = %{}

          {eval_metrics, metrics} =
            MiscUtils.timed("run_evals", metrics, fn ->
              Enum.reduce(evaluators, %{}, fn evaluator, acc ->
                case evaluator.__struct__.evaluate(evaluator, sampling_client) do
                  {:ok, metrics} ->
                    Map.merge(acc, Map.new(metrics, fn {k, v} -> {"test/#{k}", v} end))

                  {:error, _} ->
                    acc
                end
              end)
            end)

          metrics = Map.merge(metrics, eval_metrics)
          MlLog.log_metrics(ml_logger, metrics, step)
        end

        loop_evaluations_inner(evaluators, cfg, sampling_updates_queue, ml_logger)
    end
  end

  defp enqueue_forward_backward(training_client, data, loss_fn) do
    case training_client_module(training_client).forward_backward(training_client, data, loss_fn) do
      {:ok, task} -> task
      {:error, reason} -> raise "forward_backward failed: #{inspect(reason)}"
    end
  end

  defp enqueue_optim_step(training_client, adam_params) do
    case training_client_module(training_client).optim_step(training_client, adam_params) do
      {:ok, task} -> task
      {:error, reason} -> raise "optim_step failed: #{inspect(reason)}"
    end
  end

  defp collect_groups(queue, count, filter_fun, acc \\ [])

  defp collect_groups(_queue, count, _filter_fun, acc) when length(acc) >= count do
    acc
  end

  defp collect_groups(queue, count, filter_fun, acc) do
    item = BlockingQueue.pop(queue)

    if filter_fun.(item) do
      collect_groups(queue, count, filter_fun, acc ++ [item])
    else
      collect_groups(queue, count, filter_fun, acc)
    end
  end

  defp await_task!({:ok, task}) do
    case Task.await(task, :infinity) do
      {:ok, result} -> result
      {:error, reason} -> raise "Task failed: #{inspect(reason)}"
      other -> other
    end
  end

  defp await_task!(task) when is_struct(task, Task), do: await_task!({:ok, task})

  defp normalize_ok_tuple({:ok, result}), do: {:ok, result}
  defp normalize_ok_tuple({:error, _} = error), do: error
  defp normalize_ok_tuple(result), do: {:ok, result}

  defp with_logtree_scope(log_path, num_groups_to_log, file_name, scope_name, fun) do
    if log_path != nil and num_groups_to_log > 0 do
      logtree_path = Path.join(log_path, "#{file_name}.html")

      Logtree.init_trace scope_name, path: logtree_path do
        fun.()
      end
    else
      fun.()
    end
  end

  defp start_queue do
    {:ok, pid} = BlockingQueue.start_link()
    pid
  end

  defp pad_batch(i_batch) do
    i_batch |> Integer.to_string() |> String.pad_leading(6, "0")
  end

  defp training_client_module(%module{}), do: module
  defp training_client_module(_), do: Tinkex.TrainingClient

  defp service_client_module(%module{}), do: module
  defp service_client_module(_), do: Tinkex.ServiceClient
end
