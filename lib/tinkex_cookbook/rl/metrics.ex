defmodule TinkexCookbook.RL.Metrics do
  @moduledoc """
  Metrics and KL computation helpers for RL training.
  """

  alias TinkexCookbook.Types.{Datum, ModelInput, TensorData}
  alias TinkexCookbook.Utils.{MiscUtils, TinkexConvert}

  @spec compute_kl_sample_train([Datum.t()], [[float()]]) :: map()
  def compute_kl_sample_train(data, training_logprobs) do
    {all_diffs, all_sampling} =
      MiscUtils.safezip(data, training_logprobs)
      |> Enum.reduce({[], []}, fn {datum, train_logprobs}, {diffs_acc, samp_acc} ->
        sampling_logprobs = TensorData.to_list(datum.loss_fn_inputs["logprobs"])
        mask = TensorData.to_list(datum.loss_fn_inputs["mask"])

        indices = Enum.with_index(mask) |> Enum.filter(fn {m, _} -> m > 0 end)

        sampling_actions = Enum.map(indices, fn {_m, idx} -> Enum.at(sampling_logprobs, idx) end)
        training_actions = Enum.map(indices, fn {_m, idx} -> Enum.at(train_logprobs, idx) end)

        if sampling_actions == [] do
          {diffs_acc, samp_acc}
        else
          diffs = Enum.zip_with(sampling_actions, training_actions, &(&1 - &2))
          {diffs_acc ++ diffs, samp_acc ++ sampling_actions}
        end
      end)

    if all_diffs == [] do
      raise ArgumentError, "No action logprobs found for KL computation"
    end

    kl_sample_train_v1 = mean(all_diffs)
    kl_sample_train_v2 = 0.5 * mean(Enum.map(all_diffs, &(&1 * &1)))
    entropy_sample = -mean(all_sampling)

    %{
      "optim/kl_sample_train_v1" => kl_sample_train_v1,
      "optim/kl_sample_train_v2" => kl_sample_train_v2,
      "optim/entropy" => entropy_sample
    }
  end

  @spec compute_post_kl([Datum.t()], struct()) :: map()
  def compute_post_kl(data, post_sampling_client) do
    full_sequence_inputs =
      Enum.map(data, fn datum ->
        last_target =
          datum.loss_fn_inputs["target_tokens"]
          |> TensorData.to_list()
          |> List.last()

        ModelInput.append_int(datum.model_input, last_target)
      end)

    new_logprobs =
      full_sequence_inputs
      |> Enum.map(&TinkexConvert.model_input_to_tinkex/1)
      |> compute_logprobs(post_sampling_client)

    prev_logprobs =
      Enum.map(data, fn datum -> TensorData.to_list(datum.loss_fn_inputs["logprobs"]) end)

    action_masks =
      Enum.map(data, fn datum -> TensorData.to_list(datum.loss_fn_inputs["mask"]) end)

    flat_diffs =
      MiscUtils.safezip([new_logprobs, prev_logprobs, action_masks])
      |> Enum.flat_map(fn {new_lp, prev_lp, mask} ->
        aligned_new = Enum.drop(new_lp, 1)

        prev_lp
        |> Enum.zip(aligned_new)
        |> Enum.zip(mask)
        |> Enum.filter(fn {_pair, m} -> m > 0 end)
        |> Enum.map(fn {{prev_val, new_val}, _m} -> prev_val - new_val end)
      end)

    kl_post_v1 = mean(flat_diffs)
    kl_post_v2 = 0.5 * mean(Enum.map(flat_diffs, &(&1 * &1)))

    %{"kl_pre_post_v1" => kl_post_v1, "kl_pre_post_v2" => kl_post_v2}
  end

  @spec incorporate_kl_penalty([Datum.t()], struct(), float(), float()) ::
          {[Datum.t()], map()}
  def incorporate_kl_penalty(data, base_sampling_client, kl_penalty_coef, kl_discount_factor) do
    full_sequence_inputs =
      Enum.map(data, fn datum ->
        last_target =
          datum.loss_fn_inputs["target_tokens"]
          |> TensorData.to_list()
          |> List.last()

        ModelInput.append_int(datum.model_input, last_target)
      end)

    base_logprobs =
      full_sequence_inputs
      |> Enum.map(&TinkexConvert.model_input_to_tinkex/1)
      |> compute_logprobs(base_sampling_client)

    sampled_logprobs =
      Enum.map(data, fn datum -> TensorData.to_list(datum.loss_fn_inputs["logprobs"]) end)

    float_masks = Enum.map(data, fn datum -> TensorData.to_list(datum.loss_fn_inputs["mask"]) end)

    logprob_diffs =
      MiscUtils.safezip([base_logprobs, sampled_logprobs, float_masks])
      |> Enum.map(fn {base_lp, sampled_lp, mask} ->
        aligned_base = Enum.drop(base_lp, 1)

        sampled_lp
        |> Enum.zip(aligned_base)
        |> Enum.zip(mask)
        |> Enum.map(fn {{sampled, base}, m} -> (sampled - base) * m end)
      end)

    total_mask_sum = Enum.reduce(float_masks, 0.0, fn mask, acc -> acc + Enum.sum(mask) end)

    if total_mask_sum == 0.0 do
      raise ArgumentError, "KL penalty requires at least one masked token"
    end

    total_diff_sum = Enum.reduce(logprob_diffs, 0.0, fn diffs, acc -> acc + Enum.sum(diffs) end)
    avg_logp_diff = total_diff_sum / total_mask_sum

    updated_data =
      MiscUtils.safezip([data, logprob_diffs, float_masks])
      |> Enum.map(fn {datum, diffs, mask} ->
        kl_advantages =
          diffs
          |> Enum.zip(mask)
          |> Enum.map(fn {diff, m} -> kl_penalty_coef * m * (avg_logp_diff - diff) end)

        kl_advantages =
          if kl_discount_factor > 0 do
            discounted_future_sum_vectorized(kl_advantages, kl_discount_factor)
          else
            kl_advantages
          end

        advantages = TensorData.to_list(datum.loss_fn_inputs["advantages"])

        updated_advantages =
          advantages
          |> Enum.zip(kl_advantages)
          |> Enum.map(fn {adv, kl_adv} -> adv + kl_adv end)

        updated_inputs =
          Map.put(
            datum.loss_fn_inputs,
            "advantages",
            TensorData.from_list(updated_advantages, :float32)
          )

        %Datum{datum | loss_fn_inputs: updated_inputs}
      end)

    {updated_data, %{"kl_policy_base" => avg_logp_diff}}
  end

  @spec discounted_future_sum_vectorized([float()], float()) :: [float()]
  def discounted_future_sum_vectorized(values, gamma) do
    values
    |> Enum.reverse()
    |> Enum.reduce({[], 0.0}, fn value, {acc, running} ->
      updated = value + gamma * running
      {[updated | acc], updated}
    end)
    |> elem(0)
  end

  @spec compute_sampling_client_metrics([map()]) :: map()
  def compute_sampling_client_metrics(wrapped_trajectory_groups) do
    steps = Enum.map(wrapped_trajectory_groups, & &1.sampling_client_step)

    sample_times =
      Enum.map(wrapped_trajectory_groups, fn group ->
        Map.get(group.metrics, "time/trajectory_group_worker_loop/total", 0.0)
      end)

    %{
      "sampling_client/step_max" => Enum.max(steps),
      "sampling_client/step_min" => Enum.min(steps),
      "sampling_client/step_mean" => mean(steps),
      "time/sampling_time_max" => Enum.max(sample_times),
      "time/sampling_time_min" => Enum.min(sample_times),
      "time/sampling_time_mean" => mean(sample_times)
    }
  end

  defp compute_logprobs(model_inputs, sampling_client) do
    tasks =
      Enum.map(model_inputs, fn model_input ->
        case sampling_client.__struct__.compute_logprobs(sampling_client, model_input) do
          {:ok, task} -> task
          {:error, reason} -> raise "compute_logprobs failed: #{inspect(reason)}"
        end
      end)

    Enum.map(tasks, fn task ->
      case Task.await(task, :infinity) do
        {:ok, logprobs} -> logprobs
        {:error, reason} -> raise "compute_logprobs failed: #{inspect(reason)}"
        other -> raise "Unexpected compute_logprobs response: #{inspect(other)}"
      end
    end)
  end

  defp mean(values) do
    Enum.sum(values) / max(length(values), 1)
  end
end
