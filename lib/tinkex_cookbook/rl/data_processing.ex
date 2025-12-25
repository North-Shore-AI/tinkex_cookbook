defmodule TinkexCookbook.RL.DataProcessing do
  @moduledoc """
  Data processing helpers for RL training.
  """

  require Logger

  alias TinkexCookbook.Completers.TokensWithLogprobs
  alias TinkexCookbook.RL.{Trajectory, TrajectoryGroup}
  alias TinkexCookbook.Supervised.Common
  alias TinkexCookbook.Types.{Datum, EncodedTextChunk, ImageChunk, ModelInput, TensorData}
  alias TinkexCookbook.Utils.MiscUtils

  @spec compute_advantages([TrajectoryGroup.t()]) :: [[float()]]
  def compute_advantages(trajectory_groups) do
    Enum.map(trajectory_groups, fn group ->
      rewards = TrajectoryGroup.get_total_rewards(group)
      mean = Enum.sum(rewards) / max(length(rewards), 1)
      Enum.map(rewards, &(&1 - mean))
    end)
  end

  @spec trajectory_to_data(Trajectory.t(), float()) :: [Datum.t()]
  def trajectory_to_data(traj, traj_advantage) do
    acc = %{
      full_sequence: [],
      sampled_logprobs: [],
      advantages: [],
      mask: []
    }

    {data, acc} =
      Enum.reduce(traj.transitions, {[], acc}, fn transition, {data_acc, acc_state} ->
        ob_flat = flatten_chunks(transition.ob.chunks)
        ac = transition.ac

        {acc_state, delta_ob_flat, data_acc} =
          cond do
            acc_state.full_sequence == [] ->
              {acc_state, ob_flat, data_acc}

            prefix?(acc_state.full_sequence, ob_flat) ->
              delta = Enum.drop(ob_flat, length(acc_state.full_sequence))
              {acc_state, delta, data_acc}

            true ->
              datum = make_datum_from_state(acc_state)
              {clear_state(), ob_flat, data_acc ++ [datum]}
          end

        delta_ob_len = flat_ob_token_len(delta_ob_flat)
        logprobs = TokensWithLogprobs.logprobs!(ac)
        ac_len = length(ac.tokens)

        acc_state = %{
          full_sequence: acc_state.full_sequence ++ delta_ob_flat ++ ac.tokens,
          sampled_logprobs:
            acc_state.sampled_logprobs ++
              List.duplicate(0.0, delta_ob_len) ++ logprobs,
          advantages:
            acc_state.advantages ++
              List.duplicate(0.0, delta_ob_len) ++ List.duplicate(traj_advantage, ac_len),
          mask:
            acc_state.mask ++
              List.duplicate(0.0, delta_ob_len) ++ List.duplicate(1.0, ac_len)
        }

        {data_acc, acc_state}
      end)

    data =
      if acc.full_sequence != [] do
        data ++ [make_datum_from_state(acc)]
      else
        data
      end

    data
  end

  @spec assemble_training_data([TrajectoryGroup.t()], [[float()]]) ::
          {[Datum.t()], [map()]}
  def assemble_training_data(trajectory_groups, advantages_per_group) do
    data = []
    metadata = []

    {data, metadata} =
      MiscUtils.safezip(trajectory_groups, advantages_per_group)
      |> Enum.with_index()
      |> Enum.reduce({data, metadata}, fn {{group, advantages_g}, group_idx},
                                          {data_acc, meta_acc} ->
        {data_acc, meta_acc} =
          MiscUtils.safezip(group.trajectories_G, advantages_g)
          |> Enum.with_index()
          |> Enum.reduce({data_acc, meta_acc}, fn {{traj, traj_advantage}, traj_idx},
                                                  {d_acc, m_acc} ->
            new_data = trajectory_to_data(traj, traj_advantage)

            new_meta =
              List.duplicate(%{group_idx: group_idx, traj_idx: traj_idx}, length(new_data))

            {d_acc ++ new_data, m_acc ++ new_meta}
          end)

        {data_acc, meta_acc}
      end)

    {data, metadata}
  end

  @spec remove_constant_reward_groups([TrajectoryGroup.t()]) :: [TrajectoryGroup.t()]
  def remove_constant_reward_groups(groups) do
    new_groups =
      Enum.filter(groups, fn group ->
        not MiscUtils.all_same(TrajectoryGroup.get_total_rewards(group))
      end)

    if new_groups == [] do
      Logger.warning("All rewards are uniform. There will be no gradient")
      Enum.take(groups, 1)
    else
      new_groups
    end
  end

  defp clear_state do
    %{
      full_sequence: [],
      sampled_logprobs: [],
      advantages: [],
      mask: []
    }
  end

  defp make_datum_from_state(state) do
    all_tokens = flat_ob_to_model_input(state.full_sequence)

    {input_tokens, target_tokens} =
      Common.create_rightshifted_model_input_and_leftshifted_targets(all_tokens.chunks)

    sampled_logprobs = Enum.drop(state.sampled_logprobs, 1)
    advantages = Enum.drop(state.advantages, 1)
    mask = Enum.drop(state.mask, 1)

    input_len = ModelInput.length(input_tokens)

    if input_len != length(target_tokens) or
         input_len != length(sampled_logprobs) or
         input_len != length(advantages) or
         input_len != length(mask) do
      raise ArgumentError, "RL datum lengths do not match"
    end

    Datum.new(input_tokens, %{
      "target_tokens" => TensorData.from_list(target_tokens, :int64),
      "logprobs" => TensorData.from_list(sampled_logprobs, :float32),
      "advantages" => TensorData.from_list(advantages, :float32),
      "mask" => TensorData.from_list(mask, :float32)
    })
  end

  defp prefix?(seq1, seq2) do
    length(seq1) <= length(seq2) and Enum.take(seq2, length(seq1)) == seq1
  end

  defp flat_ob_token_len(flat_ob) do
    Enum.reduce(flat_ob, 0, fn elem, acc ->
      cond do
        is_integer(elem) -> acc + 1
        match?(%ImageChunk{}, elem) -> acc + elem.expected_tokens
        match?(%EncodedTextChunk{}, elem) -> acc + length(elem.tokens)
        true -> acc
      end
    end)
  end

  defp flat_ob_to_model_input(flat_ob) do
    {chunks, current_text} =
      Enum.reduce(flat_ob, {[], []}, fn elem, acc ->
        process_flat_ob_element(elem, acc)
      end)

    chunks = finalize_chunks(chunks, current_text)
    ModelInput.new(chunks)
  end

  defp process_flat_ob_element(elem, {chunks_acc, text_acc}) when is_integer(elem) do
    {chunks_acc, text_acc ++ [elem]}
  end

  defp process_flat_ob_element(%EncodedTextChunk{} = elem, {chunks_acc, text_acc}) do
    {(chunks_acc ++ [flush_text_chunk(text_acc) | [elem]]) |> List.flatten(), []}
  end

  defp process_flat_ob_element(elem, {chunks_acc, text_acc}) do
    new_chunks = append_flushed_chunk(chunks_acc, text_acc)
    {new_chunks ++ [elem], []}
  end

  defp append_flushed_chunk(chunks_acc, text_acc) do
    case flush_text_chunk(text_acc) do
      nil -> chunks_acc
      chunk -> chunks_acc ++ [chunk]
    end
  end

  defp finalize_chunks(chunks, current_text) do
    case flush_text_chunk(current_text) do
      nil -> chunks
      chunk -> chunks ++ [chunk]
    end
  end

  defp flush_text_chunk([]), do: nil
  defp flush_text_chunk(tokens), do: %EncodedTextChunk{tokens: tokens}

  defp flatten_chunks(chunks) do
    Enum.flat_map(chunks, fn
      %EncodedTextChunk{tokens: tokens} -> tokens
      %ImageChunk{} = image -> [image]
    end)
  end
end
