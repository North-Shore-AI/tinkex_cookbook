defmodule TinkexCookbook.Supervised.Common do
  @moduledoc """
  Common utilities for supervised learning.

  This module provides:
  - `datum_from_model_input_weights/3` - Creates a training Datum from a ModelInput
  - `compute_mean_nll/2` - Computes weighted mean negative log likelihood
  """

  alias TinkexCookbook.Types.{Datum, EncodedTextChunk, ImageChunk, ModelInput, TensorData}

  require Logger

  @doc """
  Creates a Datum from a ModelInput and weights tensor.

  Performs max_length truncation and next-token slicing to create input and target.
  Text chunks can be truncated, but image chunks must be wholly discarded to stay
  within max_length.

  ## Arguments

  - `model_input` - The model input containing a sequence of text and/or image chunks
  - `weights` - The weights list aligned with the model_input length
  - `max_length` - Optional maximum sequence length. If provided, truncates to this length.

  ## Returns

  A Datum with model_input (input tokens) and loss_fn_inputs (target tokens and weights)
  """
  @spec datum_from_model_input_weights(ModelInput.t(), [number()], pos_integer() | nil) ::
          Datum.t()
  def datum_from_model_input_weights(model_input, weights, max_length \\ nil) do
    chunks = model_input.chunks

    # Truncate to max_length by popping from end
    {truncated_chunks, _} = truncate_chunks(chunks, max_length)

    # Remove trailing images (no text to predict after them)
    cleaned_chunks = remove_trailing_images(truncated_chunks)

    # Create right-shifted inputs and left-shifted targets
    {input_model_input, target_tokens} =
      create_rightshifted_model_input_and_leftshifted_targets(cleaned_chunks)

    # Slice weights to match targets (skip first, align with targets)
    sliced_weights = Enum.slice(weights, 1, length(target_tokens))

    Datum.new(input_model_input, %{
      "weights" => TensorData.from_list(sliced_weights, :float32),
      "target_tokens" => TensorData.from_list(target_tokens, :int64)
    })
  end

  @doc """
  Computes weighted mean negative log likelihood.
  """
  @spec compute_mean_nll([TensorData.t()], [TensorData.t()]) :: float()
  def compute_mean_nll(logprobs_list, weights_list) do
    {total_weighted_logprobs, total_weights} =
      Enum.zip(logprobs_list, weights_list)
      |> Enum.reduce({0.0, 0.0}, fn {logprobs, weights}, {acc_logprobs, acc_weights} ->
        logprobs_data = TensorData.to_list(logprobs)
        weights_data = TensorData.to_list(weights)

        weighted_sum =
          Enum.zip(logprobs_data, weights_data)
          |> Enum.reduce(0.0, fn {lp, w}, acc -> acc + lp * w end)

        weight_sum = Enum.sum(weights_data)

        {acc_logprobs + weighted_sum, acc_weights + weight_sum}
      end)

    if total_weights == 0.0 do
      Logger.warning("No valid weights found for NLL computation")
      :nan
    else
      -total_weighted_logprobs / total_weights
    end
  end

  # Private helpers

  defp truncate_chunks(chunks, nil), do: {chunks, nil}

  defp truncate_chunks(chunks, max_length) do
    total_length = Enum.reduce(chunks, 0, fn chunk, acc -> acc + chunk_length(chunk) end)

    if total_length <= max_length do
      {chunks, total_length}
    else
      do_truncate(Enum.reverse(chunks), total_length, max_length)
    end
  end

  defp do_truncate([], total, _max), do: {[], total}

  defp do_truncate(rev_chunks, total, max) when total <= max do
    {Enum.reverse(rev_chunks), total}
  end

  defp do_truncate([%ImageChunk{expected_tokens: n} | rest], total, max) do
    # Image chunks must be removed entirely
    do_truncate(rest, total - n, max)
  end

  defp do_truncate([%EncodedTextChunk{tokens: tokens} | rest], total, max) do
    chunk_len = length(tokens)
    overflow = total - max

    if overflow < chunk_len do
      # Partial truncation
      new_tokens = Enum.take(tokens, chunk_len - overflow)
      new_chunk = %EncodedTextChunk{tokens: new_tokens}
      {Enum.reverse([new_chunk | rest]), max}
    else
      # Remove entire chunk
      do_truncate(rest, total - chunk_len, max)
    end
  end

  defp remove_trailing_images(chunks) do
    chunks
    |> Enum.reverse()
    |> Enum.drop_while(fn chunk -> match?(%ImageChunk{}, chunk) end)
    |> Enum.reverse()
  end

  @doc """
  Builds a right-shifted ModelInput and left-shifted target token list.

  Used by supervised and RL data processing to align inputs/targets.
  """
  @spec create_rightshifted_model_input_and_leftshifted_targets([ModelInput.chunk()]) ::
          {ModelInput.t(), [non_neg_integer()]}
  def create_rightshifted_model_input_and_leftshifted_targets(chunks) do
    if chunks == [] do
      raise ArgumentError, "must have at least one chunk"
    end

    last_chunk = List.last(chunks)

    unless match?(%EncodedTextChunk{}, last_chunk) do
      raise ArgumentError,
            "The last chunk must be a text chunk. Images are 0-loss anyways, so remove them beforehand."
    end

    total_length = Enum.reduce(chunks, 0, fn chunk, acc -> acc + chunk_length(chunk) end)

    if total_length < 2 do
      raise ArgumentError, "need at least 2 tokens for input/target split"
    end

    # Build input chunks: all but last token
    input_chunks = build_input_chunks(chunks)

    # Build target tokens: all tokens, then slice off first
    all_tokens = collect_all_tokens(chunks)
    target_tokens = Enum.drop(all_tokens, 1)

    {%ModelInput{chunks: input_chunks}, target_tokens}
  end

  defp build_input_chunks(chunks) do
    {init_chunks, [last]} = Enum.split(chunks, -1)

    case last do
      %EncodedTextChunk{tokens: tokens} when length(tokens) > 1 ->
        truncated_last = %EncodedTextChunk{tokens: Enum.drop(tokens, -1)}
        init_chunks ++ [truncated_last]

      %EncodedTextChunk{tokens: [_single]} ->
        # Last chunk has only one token, drop it entirely
        init_chunks

      _ ->
        init_chunks
    end
  end

  defp collect_all_tokens(chunks) do
    Enum.flat_map(chunks, fn
      %EncodedTextChunk{tokens: tokens} -> tokens
      %ImageChunk{expected_tokens: n} -> List.duplicate(0, n)
    end)
  end

  defp chunk_length(%EncodedTextChunk{tokens: tokens}), do: length(tokens)
  defp chunk_length(%ImageChunk{expected_tokens: n}), do: n
end
