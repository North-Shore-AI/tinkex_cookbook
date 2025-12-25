defmodule TinkexCookbook.Test.SpecialTokenizer do
  @moduledoc """
  Tokenizer that treats known special tokens as single IDs for renderer tests.
  """

  @fullwidth_pipe <<0xFF5C::utf8>>
  @word_sep <<0x2581::utf8>>

  @deepseek_begin "<" <>
                    @fullwidth_pipe <>
                    "begin" <>
                    @word_sep <>
                    "of" <>
                    @word_sep <>
                    "sentence" <> @fullwidth_pipe <> ">"
  @deepseek_end "<" <>
                  @fullwidth_pipe <>
                  "end" <>
                  @word_sep <>
                  "of" <>
                  @word_sep <>
                  "sentence" <> @fullwidth_pipe <> ">"
  @deepseek_user "<" <> @fullwidth_pipe <> "User" <> @fullwidth_pipe <> ">"
  @deepseek_assistant "<" <> @fullwidth_pipe <> "Assistant" <> @fullwidth_pipe <> ">"

  @special_tokens [
    {"<|im_end|>", 10_001},
    {"<|im_start|>", 10_002},
    {"<|return|>", 10_003},
    {"<|eot_id|>", 10_004},
    {"<|begin_of_text|>", 10_005},
    {"<|start_header_id|>", 10_006},
    {"<|end_header_id|>", 10_007},
    {@deepseek_begin, 10_101},
    {@deepseek_end, 10_102},
    {@deepseek_user, 10_103},
    {@deepseek_assistant, 10_104}
  ]

  @sorted_tokens Enum.sort_by(@special_tokens, fn {token, _id} -> -String.length(token) end)

  def encode(text, _opts \\ []) do
    do_encode(text, [])
  end

  def decode(tokens) when is_list(tokens) do
    Enum.map_join(tokens, "", &decode_token/1)
  end

  def bos_token, do: "<s>"
  def eos_token, do: "</s>"

  defp do_encode("", acc), do: Enum.reverse(acc)

  defp do_encode(text, acc) do
    case Enum.find(@sorted_tokens, fn {token, _id} -> String.starts_with?(text, token) end) do
      {token, id} ->
        {_prefix, rest} = String.split_at(text, String.length(token))
        do_encode(rest, [id | acc])

      nil ->
        <<char::utf8, rest::binary>> = text
        do_encode(rest, [char | acc])
    end
  end

  defp decode_token(token) do
    case Enum.find(@special_tokens, fn {_str, id} -> id == token end) do
      {str, _} -> str
      nil -> <<token::utf8>>
    end
  end
end
