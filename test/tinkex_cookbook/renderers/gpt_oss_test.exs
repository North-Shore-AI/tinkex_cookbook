defmodule TinkexCookbook.Renderers.GptOssTest do
  use ExUnit.Case, async: true

  alias TinkexCookbook.Renderers.{GptOss, Types}
  alias TinkexCookbook.Test.SpecialTokenizer
  alias TinkexCookbook.Types.EncodedTextChunk

  test "init validates system prompt options" do
    assert_raise ArgumentError, ~r/Reasoning effort must be set/, fn ->
      GptOss.init(tokenizer: SpecialTokenizer, use_system_prompt: true)
    end

    assert_raise ArgumentError, ~r/Reasoning effort must be set/, fn ->
      GptOss.init(tokenizer: SpecialTokenizer, use_system_prompt: false, reasoning_effort: "low")
    end
  end

  test "bos_tokens includes system prompt when enabled" do
    {:ok, state} =
      GptOss.init(
        tokenizer: SpecialTokenizer,
        use_system_prompt: true,
        reasoning_effort: "low",
        current_date: "2025-01-01"
      )

    tokens = GptOss.bos_tokens(state)
    text = SpecialTokenizer.decode(tokens)

    assert String.contains?(text, "Reasoning: low")
    assert String.contains?(text, "Current date: 2025-01-01")
  end

  test "assistant render includes analysis channel only for last message" do
    {:ok, state} = GptOss.init(tokenizer: SpecialTokenizer, use_system_prompt: false)
    message = Types.message("assistant", "Answer", thinking: "Reasoning")

    {rendered_last, _} = GptOss.render_message(0, message, true, state)
    [%EncodedTextChunk{tokens: last_tokens}] = rendered_last.content
    last_content = SpecialTokenizer.decode(last_tokens)

    assert String.contains?(last_content, "<|channel|>analysis")
    assert String.contains?(last_content, "<|channel|>final")

    {rendered_hist, _} = GptOss.render_message(0, message, false, state)
    [%EncodedTextChunk{tokens: hist_tokens}] = rendered_hist.content
    hist_content = SpecialTokenizer.decode(hist_tokens)

    refute String.contains?(hist_content, "<|channel|>analysis")
    assert String.contains?(hist_content, "<|channel|>final")
  end

  test "stop_sequences returns return token" do
    {:ok, state} = GptOss.init(tokenizer: SpecialTokenizer, use_system_prompt: false)
    assert GptOss.stop_sequences(state) == SpecialTokenizer.encode("<|return|>")
  end

  test "parse_response splits on return token" do
    {:ok, state} = GptOss.init(tokenizer: SpecialTokenizer, use_system_prompt: false)
    tokens = SpecialTokenizer.encode("Hi<|return|>")

    {message, format_ok} = GptOss.parse_response(tokens, state)

    assert format_ok == true
    assert message.content == "Hi"
  end
end
