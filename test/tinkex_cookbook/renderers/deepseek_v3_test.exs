defmodule TinkexCookbook.Renderers.DeepSeekV3Test do
  use ExUnit.Case, async: true

  alias TinkexCookbook.Renderers.{DeepSeekV3, DeepSeekV3DisableThinking, Renderer, Types}
  alias TinkexCookbook.Test.SpecialTokenizer
  alias TinkexCookbook.Types.EncodedTextChunk

  defp fullwidth_pipe, do: <<0xFF5C::utf8>>
  defp word_sep, do: <<0x2581::utf8>>

  defp special_token(name) do
    "<" <> fullwidth_pipe() <> name <> fullwidth_pipe() <> ">"
  end

  test "bos_tokens uses begin of sentence token" do
    {:ok, state} = DeepSeekV3.init(tokenizer: SpecialTokenizer)
    bos = DeepSeekV3.bos_tokens(state)

    expected =
      special_token("begin" <> word_sep() <> "of" <> word_sep() <> "sentence")

    assert SpecialTokenizer.decode(bos) == expected
  end

  test "stop_sequences uses end of sentence token" do
    {:ok, state} = DeepSeekV3.init(tokenizer: SpecialTokenizer)
    [stop] = DeepSeekV3.stop_sequences(state)

    expected =
      special_token("end" <> word_sep() <> "of" <> word_sep() <> "sentence")

    assert SpecialTokenizer.decode([stop]) == expected
  end

  test "assistant messages include end token" do
    {:ok, state} = DeepSeekV3.init(tokenizer: SpecialTokenizer)
    message = Types.message("assistant", "Hi")

    {rendered, _} = DeepSeekV3.render_message(0, message, true, state)
    [%EncodedTextChunk{tokens: content_tokens}] = rendered.content

    expected =
      "Hi" <>
        special_token("end" <> word_sep() <> "of" <> word_sep() <> "sentence")

    assert SpecialTokenizer.decode(content_tokens) == expected
  end

  test "user messages omit end token" do
    {:ok, state} = DeepSeekV3.init(tokenizer: SpecialTokenizer)
    message = Types.message("user", "Hi")

    {rendered, _} = DeepSeekV3.render_message(0, message, true, state)
    [%EncodedTextChunk{tokens: content_tokens}] = rendered.content

    assert SpecialTokenizer.decode(content_tokens) == "Hi"
  end

  test "system role raises unless system_role_as_user is true" do
    {:ok, state} = DeepSeekV3.init(tokenizer: SpecialTokenizer)
    message = Types.message("system", "Hi")

    assert_raise ArgumentError, ~r/Unsupported role/, fn ->
      DeepSeekV3.render_message(0, message, true, state)
    end
  end

  test "system role works when system_role_as_user is enabled" do
    {:ok, state} = DeepSeekV3.init(tokenizer: SpecialTokenizer, system_role_as_user: true)
    message = Types.message("system", "Hi")

    {rendered, _} = DeepSeekV3.render_message(0, message, true, state)
    [%EncodedTextChunk{tokens: content_tokens}] = rendered.content

    assert SpecialTokenizer.decode(content_tokens) == "Hi"
  end

  test "disable thinking prefixes </think>" do
    {:ok, state} = DeepSeekV3DisableThinking.init(tokenizer: SpecialTokenizer)
    message = Types.message("assistant", "Answer")

    {rendered, _} = DeepSeekV3DisableThinking.render_message(0, message, true, state)
    [%EncodedTextChunk{tokens: content_tokens}] = rendered.content
    content = SpecialTokenizer.decode(content_tokens)

    assert String.starts_with?(content, "</think>")
  end

  test "disable thinking generation prompt includes </think> prefill" do
    {:ok, state} = DeepSeekV3DisableThinking.init(tokenizer: SpecialTokenizer)
    messages = [Types.message("user", "Hello")]

    {model_input, _} =
      Renderer.build_generation_prompt(
        DeepSeekV3DisableThinking,
        messages,
        "assistant",
        nil,
        state
      )

    %EncodedTextChunk{tokens: prefill_tokens} = List.last(model_input.chunks)
    assert SpecialTokenizer.decode(prefill_tokens) == "</think>"
  end
end
