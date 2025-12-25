defmodule TinkexCookbook.Renderers.Helpers do
  @moduledoc """
  Shared encoding/decoding helpers for renderers.
  """

  alias TinkexCookbook.Renderers.Types
  alias TinkexCookbook.Types.TensorData

  @spec encode(module() | map(), String.t(), keyword()) :: [non_neg_integer()]
  def encode(tokenizer, text, opts \\ [])

  def encode(tokenizer, text, opts) when is_atom(tokenizer) do
    tokenizer.encode(text, opts)
  end

  def encode(%{encode: encode_fn}, text, opts) when is_function(encode_fn, 2) do
    encode_fn.(text, opts)
  end

  @spec decode(module() | map(), [non_neg_integer()]) :: String.t()
  def decode(tokenizer, tokens) when is_atom(tokenizer) do
    tokenizer.decode(tokens)
  end

  def decode(%{decode: decode_fn}, tokens) when is_function(decode_fn, 1) do
    decode_fn.(tokens)
  end

  @spec bos_token_string(module() | map()) :: String.t() | nil
  def bos_token_string(tokenizer) when is_atom(tokenizer) do
    if function_exported?(tokenizer, :bos_token, 0) do
      tokenizer.bos_token()
    else
      nil
    end
  end

  def bos_token_string(%{bos_token: bos_token}) when is_binary(bos_token), do: bos_token

  def bos_token_string(%{bos_token: bos_token_fn}) when is_function(bos_token_fn, 0),
    do: bos_token_fn.()

  def bos_token_string(_tokenizer), do: nil

  @spec parse_response_for_stop_token([non_neg_integer()], module() | map(), non_neg_integer()) ::
          {Types.Message.t(), boolean()}
  def parse_response_for_stop_token(tokens, tokenizer, stop_token) do
    count = Enum.count(tokens, &(&1 == stop_token))

    cond do
      count == 0 ->
        content = decode(tokenizer, tokens)
        {Types.message("assistant", content), false}

      count == 1 ->
        idx = Enum.find_index(tokens, &(&1 == stop_token))
        content = decode(tokenizer, Enum.take(tokens, idx))
        {Types.message("assistant", content), true}

      true ->
        raise RuntimeError,
              "When parsing response, expected to split into 1 or 2 pieces using stop tokens, " <>
                "but got #{count}. You probably are using the wrong stop tokens when sampling"
    end
  end

  @spec tokens_weights_from_strings_weights(
          [{String.t(), number()}],
          module() | map()
        ) :: {TensorData.t(), TensorData.t()}
  def tokens_weights_from_strings_weights(strings_weights, tokenizer)
      when is_list(strings_weights) do
    {strings, weights} = Enum.unzip(strings_weights)

    token_chunks =
      strings
      |> Enum.with_index()
      |> Enum.map(fn {string, idx} ->
        encode(tokenizer, string, add_special_tokens: idx == 0)
      end)

    tokens = List.flatten(token_chunks)

    token_weights =
      Enum.zip(weights, token_chunks)
      |> Enum.flat_map(fn {weight, chunk} ->
        List.duplicate(weight, length(chunk))
      end)

    {
      TensorData.from_list(tokens, :int64),
      TensorData.from_list(token_weights, :float32)
    }
  end
end
