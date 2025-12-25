# Training loop code is inherently complex with many parameters
# credo:disable-for-this-file Credo.Check.Refactor.FunctionArity
# credo:disable-for-this-file Credo.Check.Refactor.Nesting
defmodule TinkexCookbook.Distillation.TrainOnPolicy do
  @moduledoc """
  On-policy distillation training loop.
  """

  require Logger

  alias TinkexCookbook.Display
  alias TinkexCookbook.Distillation.CompositeDataset
  alias TinkexCookbook.RL.{DataProcessing, EnvGroupBuilder, Metrics, Train, TrajectoryGroup}
  alias TinkexCookbook.RL.MetricUtil
  alias TinkexCookbook.RL.MetricUtil.RLTestSetEvaluator
  alias TinkexCookbook.TokenizerUtils
  alias TinkexCookbook.Types.{Datum, ModelInput, TensorData}
  alias TinkexCookbook.Utils.{Checkpoint, MiscUtils, TinkexConvert, Trace}
  alias TinkexCookbook.Utils.MlLog

  @default_base_url "https://tinker.thinkingmachines.dev/services/tinker-prod"

  defmodule Config do
    @moduledoc """
    Configuration for on-policy distillation.
    """

    use ChzEx.Schema

    @type t() :: %__MODULE__{}

    chz_schema do
      field(:learning_rate, :float)
      field(:dataset_configs, :any, virtual: true)
      field(:model_name, :string)
      field(:max_tokens, :integer)
      field(:temperature, :float, default: 1.0)
      field(:compute_post_kl, :boolean, default: false)
      field(:evaluator_builders, :any, default: [], virtual: true)
      field(:lora_rank, :integer, default: 32)
      field(:kl_penalty_coef, :float, default: 1.0)
      field(:kl_discount_factor, :float, default: 0.0)
      field(:loss_fn, :string, default: "importance_sampling")
      field(:num_substeps, :integer, default: 1)
      field(:wandb_project, :string, default: nil)
      field(:wandb_name, :string, default: nil)
      field(:log_path, :string)
      field(:base_url, :string, default: nil)
      field(:enable_trace, :boolean, default: false)
      field(:eval_every, :integer, default: 20)
      field(:save_every, :integer, default: 20)
      field(:load_checkpoint_path, :string, default: nil)
    end

    @spec expand_log_path(struct()) :: struct()
    def expand_log_path(%__MODULE__{log_path: log_path} = cfg) do
      %{cfg | log_path: Path.expand(log_path)}
    end
  end

  @spec incorporate_kl_penalty(
          [Datum.t()],
          [struct()],
          [non_neg_integer()],
          float(),
          float()
        ) :: {list(Datum.t()), map()}
  def incorporate_kl_penalty(
        data,
        teacher_clients,
        dataset_indices,
        kl_penalty_coef,
        kl_discount_factor
      ) do
    full_sequence_inputs = Enum.map(data, &build_full_sequence/1)

    teacher_logprobs =
      compute_logprobs_for_clients(teacher_clients, full_sequence_inputs)
      |> Enum.map(&Enum.drop(&1, 1))

    sampled_logprobs =
      Enum.map(data, fn datum -> TensorData.to_list(datum.loss_fn_inputs["logprobs"]) end)

    float_masks = Enum.map(data, fn datum -> TensorData.to_list(datum.loss_fn_inputs["mask"]) end)

    reverse_kl =
      Enum.zip([sampled_logprobs, teacher_logprobs, float_masks])
      |> Enum.map(fn {sampled_lp, teacher_lp, mask} ->
        Enum.zip([sampled_lp, teacher_lp, mask])
        |> Enum.map(fn {sampled, teacher, m} -> (sampled - teacher) * m end)
      end)

    total_mask_sum = Enum.reduce(float_masks, 0.0, fn mask, acc -> acc + Enum.sum(mask) end)

    if total_mask_sum == 0.0 do
      raise ArgumentError, "KL penalty requires at least one masked token"
    end

    total_diff_sum = Enum.reduce(reverse_kl, 0.0, fn diffs, acc -> acc + Enum.sum(diffs) end)
    avg_logp_diff = total_diff_sum / total_mask_sum

    per_dataset =
      Enum.with_index(data)
      |> Enum.reduce(%{}, fn {_datum, idx}, acc ->
        dataset_idx = Enum.at(dataset_indices, idx)
        diffs = Enum.at(reverse_kl, idx)
        mask = Enum.at(float_masks, idx)
        sum_diff = Enum.sum(diffs)
        sum_mask = Enum.sum(mask)

        {prev_diff, prev_mask} = Map.get(acc, dataset_idx, {0.0, 0.0})
        Map.put(acc, dataset_idx, {prev_diff + sum_diff, prev_mask + sum_mask})
      end)

    updated_data =
      Enum.zip([data, reverse_kl, float_masks])
      |> Enum.map(fn {datum, diffs, mask} ->
        kl_advantages =
          diffs
          |> Enum.zip(mask)
          |> Enum.map(fn {diff, m} -> -kl_penalty_coef * m * diff end)

        kl_advantages =
          if kl_discount_factor > 0 do
            Metrics.discounted_future_sum_vectorized(kl_advantages, kl_discount_factor)
          else
            kl_advantages
          end

        advantages = TensorData.to_list(datum.loss_fn_inputs["advantages"])

        updated_advantages =
          Enum.zip(advantages, kl_advantages)
          |> Enum.map(fn {adv, kl_adv} -> adv + kl_adv end)

        updated_inputs =
          Map.put(
            datum.loss_fn_inputs,
            "advantages",
            TensorData.from_list(updated_advantages, :float32)
          )

        %Datum{datum | loss_fn_inputs: updated_inputs}
      end)

    metrics = %{"teacher_kl" => avg_logp_diff}

    metrics =
      Enum.reduce(per_dataset, metrics, fn {dataset_idx, {sum_diff, sum_mask}}, acc ->
        if sum_mask > 0 do
          Map.put(acc, "teacher_kl/dataset_#{dataset_idx}", sum_diff / sum_mask)
        else
          acc
        end
      end)

    {updated_data, metrics}
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

    if cfg.enable_trace do
      trace_path = Path.join(cfg.log_path, "trace_events.jsonl")
      Logger.info("Tracing is enabled. Trace events will be saved to #{trace_path}")
      Trace.trace_init(output_file: trace_path)
    end

    resume_info = Checkpoint.get_last_checkpoint(cfg.log_path)
    start_batch = if resume_info, do: Map.get(resume_info, "batch") || 0, else: 0

    api_key = System.get_env("TINKER_API_KEY")

    if is_nil(api_key) do
      Logger.error("TINKER_API_KEY environment variable is required")
      {:error, :missing_api_key}
    else
      base_url = cfg.base_url || System.get_env("TINKER_BASE_URL", @default_base_url)
      tinkex_config = Tinkex.Config.new(api_key: api_key, base_url: base_url)

      with {:ok, service_client} <- Tinkex.ServiceClient.start_link(config: tinkex_config) do
        training_client =
          await_task!(
            service_client_module(service_client).create_lora_training_client_async(
              service_client,
              cfg.model_name,
              rank: cfg.lora_rank
            )
          )

        load_state_path =
          if resume_info do
            Map.get(resume_info, "state_path")
          else
            cfg.load_checkpoint_path
          end

        if load_state_path do
          _ =
            await_task!(
              training_client_module(training_client).load_state_with_optimizer(
                training_client,
                load_state_path
              )
            )

          Logger.info("Loaded state from #{load_state_path}")
        end

        tokenizer = get_tokenizer(training_client, cfg.model_name)

        {datasets, teacher_clients, groups_per_batch_list, evaluators} =
          build_datasets_and_teachers(cfg.dataset_configs, service_client, cfg.max_tokens)

        composite_dataset = CompositeDataset.new(datasets, groups_per_batch_list)
        num_batches = CompositeDataset.length(composite_dataset)
        Logger.info("Will train on #{num_batches} batches")

        evaluators = evaluators ++ Enum.map(cfg.evaluator_builders, & &1.())

        do_sync_training(
          start_batch,
          num_batches,
          num_batches,
          cfg,
          training_client,
          service_client,
          evaluators,
          composite_dataset,
          teacher_clients,
          ml_logger,
          tokenizer
        )

        if start_batch < num_batches do
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
      end
    end
  end

  defp do_sync_training(
         start_batch,
         end_batch,
         num_batches,
         cfg,
         training_client,
         _service_client,
         evaluators,
         dataset,
         teacher_clients,
         ml_logger,
         tokenizer
       ) do
    {initial_sampling_client, _} =
      Train.save_checkpoint_and_get_sampling_client(
        training_client,
        start_batch,
        cfg.log_path,
        cfg.save_every
      )

    Enum.reduce(start_batch..(end_batch - 1), initial_sampling_client, fn i_batch,
                                                                          sampling_client ->
      metrics = %{
        "progress/batch" => i_batch,
        "optim/lr" => cfg.learning_rate,
        "progress/done_frac" => (i_batch + 1) / num_batches
      }

      start_time = System.monotonic_time()

      metrics =
        if cfg.eval_every > 0 and rem(i_batch, cfg.eval_every) == 0 do
          {eval_metrics, metrics} =
            MiscUtils.timed("run_evals", metrics, fn ->
              run_sampling_evaluations(evaluators, sampling_client)
            end)

          Map.merge(metrics, prefix_keys(eval_metrics, "test/"))
        else
          metrics
        end

      {env_group_builders, dataset_indices} = CompositeDataset.get_batch(dataset, i_batch)

      {trajectory_groups, metrics} =
        MiscUtils.timed("sample", metrics, fn ->
          env_group_builders
          |> Task.async_stream(
            fn builder ->
              Train.do_group_rollout_and_filter_constant_reward(
                sampling_client,
                builder,
                cfg.max_tokens,
                cfg.temperature,
                false
              )
            end,
            ordered: true,
            timeout: :infinity
          )
          |> Enum.map(fn {:ok, group} -> group end)
          |> Enum.filter(& &1)
        end)

      {sampling_client, train_metrics} =
        do_train_step_and_get_sampling_client(
          cfg,
          i_batch,
          training_client,
          tokenizer,
          env_group_builders,
          trajectory_groups,
          dataset_indices,
          teacher_clients
        )

      metrics = Map.merge(metrics, train_metrics)
      total_time = System.monotonic_time() - start_time

      metrics =
        Map.put(
          metrics,
          "time/total",
          System.convert_time_unit(total_time, :native, :millisecond) / 1000
        )

      MlLog.log_metrics(ml_logger, metrics, i_batch)

      sampling_client
    end)
  end

  defp do_train_step_and_get_sampling_client(
         cfg,
         i_batch,
         training_client,
         tokenizer,
         env_group_builders,
         trajectory_groups,
         dataset_indices,
         teacher_clients
       ) do
    {data, prepare_metrics} =
      prepare_minibatch(
        env_group_builders,
        trajectory_groups,
        tokenizer,
        dataset_indices,
        teacher_clients,
        cfg.kl_penalty_coef,
        cfg.kl_discount_factor
      )

    {training_logprobs, train_metrics} =
      MiscUtils.timed("train", prepare_metrics, fn ->
        Train.train_step(data, training_client, cfg.learning_rate, cfg.num_substeps, cfg.loss_fn)
      end)

    {sampling_client, full_metrics} =
      Train.compute_full_batch_metrics_and_get_sampling_client(
        training_client,
        i_batch + 1,
        data,
        training_logprobs,
        cfg.log_path,
        cfg.save_every,
        cfg.compute_post_kl
      )

    {sampling_client, Map.merge(train_metrics, full_metrics)}
  end

  defp prepare_minibatch(
         env_group_builders,
         trajectory_groups,
         tokenizer,
         dataset_indices,
         teacher_clients,
         kl_penalty_coef,
         kl_discount_factor
       ) do
    taglists = Enum.map(env_group_builders, &EnvGroupBuilder.logging_tags/1)
    metrics = MetricUtil.compute_trajectory_metrics(trajectory_groups, taglists)

    Enum.each(Enum.take(trajectory_groups, 2), &print_group(&1, tokenizer))

    {data, metrics} =
      MiscUtils.timed("assemble_training_data", metrics, fn ->
        advantages = DataProcessing.compute_advantages(trajectory_groups)
        DataProcessing.assemble_training_data(trajectory_groups, advantages)
      end)

    {data, metadata} = data

    printed = MapSet.new()

    _printed =
      Enum.zip(data, metadata)
      |> Enum.reduce(printed, fn {datum, meta}, printed_acc ->
        dataset_idx = Enum.at(dataset_indices, meta.group_idx)

        if MapSet.member?(printed_acc, dataset_idx) do
          printed_acc
        else
          Logger.info(Display.colorize_example(datum, tokenizer, "mask"))
          MapSet.put(printed_acc, dataset_idx)
        end
      end)

    if kl_penalty_coef > 0 do
      {kl_result, metrics} =
        MiscUtils.timed("compute_kl_penalty", metrics, fn ->
          teacher_clients_d =
            Enum.map(metadata, fn meta ->
              Enum.at(teacher_clients, Enum.at(dataset_indices, meta.group_idx))
            end)

          dataset_indices_d =
            Enum.map(metadata, fn meta -> Enum.at(dataset_indices, meta.group_idx) end)

          incorporate_kl_penalty(
            data,
            teacher_clients_d,
            dataset_indices_d,
            kl_penalty_coef,
            kl_discount_factor
          )
        end)

      {updated_data, kl_metrics} = kl_result
      {updated_data, Map.merge(metrics, kl_metrics)}
    else
      {data, metrics}
    end
  end

  defp print_group(%TrajectoryGroup{} = group, tokenizer) do
    Enum.each(group.trajectories_G, fn traj ->
      Logger.info(Display.format_trajectory(traj, tokenizer))
    end)
  end

  defp compute_logprobs_for_clients(teacher_clients, model_inputs) do
    teacher_clients
    |> Enum.zip(model_inputs)
    |> Enum.map(fn {client, model_input} ->
      await_task!(
        sampling_client_module(client).compute_logprobs(
          client,
          TinkexConvert.model_input_to_tinkex(model_input)
        )
      )
    end)
    |> Enum.map(fn
      {:ok, logprobs} -> logprobs
      {:error, reason} -> raise "compute_logprobs failed: #{inspect(reason)}"
      other -> raise "Unexpected compute_logprobs response: #{inspect(other)}"
    end)
  end

  defp build_full_sequence(%Datum{} = datum) do
    target_tokens = TensorData.to_list(datum.loss_fn_inputs["target_tokens"])

    case target_tokens do
      [] -> datum.model_input
      _ -> ModelInput.append_int(datum.model_input, List.last(target_tokens))
    end
  end

  defp run_sampling_evaluations(evaluators, sampling_client) do
    evaluators
    |> Enum.map(fn evaluator ->
      Task.async(fn -> evaluator.__struct__.evaluate(evaluator, sampling_client) end)
    end)
    |> Enum.reduce(%{}, fn task, acc ->
      case Task.await(task, :infinity) do
        {:ok, metrics} -> Map.merge(acc, metrics)
        {:error, reason} -> raise "Evaluation failed: #{inspect(reason)}"
        other -> raise "Unexpected evaluator response: #{inspect(other)}"
      end
    end)
  end

  defp prefix_keys(metrics, prefix) do
    Map.new(metrics, fn {key, value} -> {prefix <> to_string(key), value} end)
  end

  defp build_datasets_and_teachers(dataset_configs, service_client, max_tokens) do
    dataset_configs
    |> Enum.reduce({[], [], [], []}, fn dataset_config, {datasets, teachers, groups, evals} ->
      {dataset, maybe_test_dataset} =
        dataset_config.dataset_builder.__struct__.build(dataset_config.dataset_builder)

      evals =
        if maybe_test_dataset != nil do
          evals ++ [RLTestSetEvaluator.new(maybe_test_dataset, max_tokens)]
        else
          evals
        end

      teacher_config = dataset_config.teacher_config

      teacher_client =
        if teacher_config.load_checkpoint_path != nil do
          service_client_module(service_client).create_sampling_client(
            service_client,
            base_model: teacher_config.base_model,
            model_path: teacher_config.load_checkpoint_path
          )
        else
          service_client_module(service_client).create_sampling_client(
            service_client,
            base_model: teacher_config.base_model
          )
        end

      teacher_client =
        case teacher_client do
          {:ok, client} -> client
          {:error, reason} -> raise "Failed to create teacher sampling client: #{inspect(reason)}"
        end

      {
        datasets ++ [dataset],
        teachers ++ [teacher_client],
        groups ++ [dataset_config.groups_per_batch],
        evals
      }
    end)
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

  @spec training_client_module(pid() | struct()) :: module()
  defp training_client_module(client) when is_pid(client), do: Tinkex.TrainingClient
  defp training_client_module(%module{}), do: module

  @spec sampling_client_module(pid() | struct()) :: module()
  defp sampling_client_module(client) when is_pid(client), do: Tinkex.SamplingClient
  defp sampling_client_module(%module{}), do: module

  @spec service_client_module(pid()) :: module()
  defp service_client_module(client) when is_pid(client), do: Tinkex.ServiceClient
end
