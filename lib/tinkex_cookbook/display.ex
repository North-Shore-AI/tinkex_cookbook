defmodule TinkexCookbook.Display do
  @moduledoc """
  Display helpers for colorized output and trajectory formatting.
  """

  alias TinkexCookbook.Types.{Datum, EncodedTextChunk, ImageChunk, ModelInput, TensorData}
  alias TinkexCookbook.Utils.FormatColorized

  @spec colorize_example(Datum.t(), module() | map(), String.t()) :: String.t()
  def colorize_example(%Datum{} = datum, tokenizer, key \\ "weights") do
    int_tokens =
      datum.model_input.chunks
      |> Enum.flat_map(&to_ints(&1, tokenizer))
      |> Kernel.++(last_target_token(datum))

    weights = [0.0] ++ TensorData.to_list(datum.loss_fn_inputs[key])
    FormatColorized.format_colorized(int_tokens, weights, tokenizer)
  end

  @spec format_trajectory(map(), module() | map()) :: String.t()
  def format_trajectory(%{transitions: transitions} = _trajectory, tokenizer) do
    header = String.duplicate("=", 60)

    transitions_text =
      transitions
      |> Enum.with_index()
      |> Enum.map_join("\n", fn {transition, idx} ->
        ob_tokens = ModelInput.all_tokens(transition.ob)
        action_tokens = transition.ac.tokens

        [
          "------ Transition #{idx} ------",
          "#{colorize_label("Observation:")} #{decode(tokenizer, ob_tokens)}",
          "#{colorize_label("Action:")} #{decode(tokenizer, action_tokens)}",
          "#{colorize_label("Reward:")} #{transition.reward}",
          "#{colorize_label("Episode done:")} #{transition.episode_done}",
          "#{colorize_label("Metrics:")} #{inspect(transition.metrics)}",
          String.duplicate("-", 60)
        ]
        |> Enum.join("\n")
      end)

    [header, transitions_text, header]
    |> Enum.join("\n")
  end

  defp to_ints(%EncodedTextChunk{tokens: tokens}, _tokenizer), do: tokens

  defp to_ints(%ImageChunk{expected_tokens: n}, tokenizer) do
    at_token =
      case encode(tokenizer, "@", add_special_tokens: false) do
        [token] -> token
        _ -> ?@
      end

    List.duplicate(at_token, n)
  end

  defp last_target_token(%Datum{loss_fn_inputs: inputs}) do
    case Map.get(inputs, "target_tokens") do
      %TensorData{} = data ->
        case TensorData.to_list(data) do
          [] -> []
          list -> [List.last(list)]
        end

      _ ->
        []
    end
  end

  defp colorize_label(label) do
    IO.ANSI.format([:green, :bright, label, :reset]) |> IO.iodata_to_binary()
  end

  defp encode(tokenizer, text, opts) when is_atom(tokenizer) do
    tokenizer.encode(text, opts)
  end

  defp encode(%Tokenizers.Tokenizer{} = tokenizer, text, _opts) do
    case Tokenizers.Tokenizer.encode(tokenizer, text) do
      {:ok, encoding} -> Tokenizers.Encoding.get_ids(encoding)
      {:error, _} -> []
    end
  end

  defp encode(%TiktokenEx.Encoding{} = encoding, text, opts) do
    TiktokenEx.Encoding.encode(encoding, text, opts)
  end

  defp encode(%{encode: encode_fn}, text, opts) when is_function(encode_fn, 2) do
    encode_fn.(text, opts)
  end

  defp decode(tokenizer, tokens) when is_atom(tokenizer) do
    tokenizer.decode(tokens)
  end

  defp decode(%Tokenizers.Tokenizer{} = tokenizer, tokens) do
    case Tokenizers.Tokenizer.decode(tokenizer, tokens) do
      {:ok, text} -> text
      {:error, _} -> inspect(tokens)
    end
  end

  defp decode(%TiktokenEx.Encoding{} = encoding, tokens) do
    TiktokenEx.Encoding.decode(encoding, tokens)
  end

  defp decode(%{decode: decode_fn}, tokens) when is_function(decode_fn, 1) do
    decode_fn.(tokens)
  end
end
