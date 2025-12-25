defmodule TinkexCookbook.Utils.FormatColorized do
  @moduledoc """
  Colorize decoded text based on per-token weights.
  """

  @newline_arrow <<0x21B5::utf8>>

  @spec format_colorized([integer()], [float()], module() | map(), boolean()) :: String.t()
  def format_colorized(tokens, weights, tokenizer, draw_newline_arrow \\ false) do
    if length(tokens) != length(weights) do
      raise ArgumentError, "`tokens` and `weights` must be the same length."
    end

    {chunks, current_ids, current_color} =
      Enum.zip(tokens, weights)
      |> Enum.reduce({[], [], nil}, fn {token, weight}, {acc, ids, color} ->
        next_color =
          cond do
            weight < 0 -> :red
            weight == 0 -> :yellow
            true -> :green
          end

        if next_color != color and ids != [] do
          {acc ++ [format_run(ids, color, tokenizer, draw_newline_arrow)], [token], next_color}
        else
          {acc, ids ++ [token], next_color}
        end
      end)

    chunks =
      if current_ids != [] do
        chunks ++ [format_run(current_ids, current_color, tokenizer, draw_newline_arrow)]
      else
        chunks
      end

    Enum.join(chunks, "")
  end

  defp format_run(tokens, color, tokenizer, draw_newline_arrow) do
    text = decode(tokenizer, tokens)
    parts = split_lines_keepends(text)

    Enum.map_join(parts, "", fn part ->
      part =
        if draw_newline_arrow do
          String.replace(part, "\n", @newline_arrow <> "\n")
        else
          part
        end

      IO.ANSI.format([color, part, :reset]) |> IO.iodata_to_binary()
    end)
  end

  defp split_lines_keepends(text) do
    parts = String.split(text, "\n", trim: false)

    case parts do
      [] -> []
      [single] -> [single]
      _ -> append_newlines_except_last(parts)
    end
  end

  defp append_newlines_except_last(parts) do
    parts
    |> Enum.with_index()
    |> Enum.map(fn {part, idx} ->
      if idx < length(parts) - 1 do
        part <> "\n"
      else
        part
      end
    end)
  end

  defp decode(tokenizer, tokens) when is_atom(tokenizer) do
    tokenizer.decode(tokens)
  end

  defp decode(%{decode: decode_fn}, tokens) when is_function(decode_fn, 1) do
    decode_fn.(tokens)
  end
end
