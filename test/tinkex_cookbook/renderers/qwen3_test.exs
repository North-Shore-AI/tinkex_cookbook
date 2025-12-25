defmodule TinkexCookbook.Renderers.Qwen3Test do
  use ExUnit.Case, async: true

  alias TinkexCookbook.Renderers.{Qwen3, Qwen3DisableThinking, Qwen3Instruct, Renderer, Types}
  alias TinkexCookbook.Test.SpecialTokenizer
  alias TinkexCookbook.Types.EncodedTextChunk

  describe "Qwen3.init/1" do
    test "defaults strip_thinking_from_history to true" do
      assert {:ok, state} = Qwen3.init(tokenizer: SpecialTokenizer)
      assert state.strip_thinking_from_history == true
    end
  end

  describe "Qwen3.bos_tokens/1" do
    test "returns empty list" do
      {:ok, state} = Qwen3.init(tokenizer: SpecialTokenizer)
      assert Qwen3.bos_tokens(state) == []
    end
  end

  describe "Qwen3.stop_sequences/1" do
    test "returns <|im_end|> token" do
      {:ok, state} = Qwen3.init(tokenizer: SpecialTokenizer)
      assert Qwen3.stop_sequences(state) == SpecialTokenizer.encode("<|im_end|>")
    end
  end

  describe "Qwen3.render_message/4" do
    setup do
      {:ok, state} = Qwen3.init(tokenizer: SpecialTokenizer)
      %{state: state}
    end

    test "adds <think> prefix for last assistant message without thinking", %{state: state} do
      message = Types.message("assistant", "Hello")
      {rendered, _} = Qwen3.render_message(0, message, true, state)

      %EncodedTextChunk{tokens: prefix_tokens} = rendered.prefix
      prefix = SpecialTokenizer.decode(prefix_tokens)

      assert String.contains?(prefix, "<|im_start|>assistant")
      assert String.contains?(prefix, "<think>\n")
    end

    test "strips thinking from historical assistant messages", %{state: state} do
      message = Types.message("assistant", "<think>\nReason\n</think>\n\nAnswer")
      {rendered, _} = Qwen3.render_message(0, message, false, state)

      [%EncodedTextChunk{tokens: content_tokens}] = rendered.content
      content = SpecialTokenizer.decode(content_tokens)

      refute String.contains?(content, "<think>")
      assert String.contains?(content, "Answer")
    end

    test "wraps tool responses", %{state: state} do
      message = Types.message("tool", "tool output")
      {rendered, _} = Qwen3.render_message(0, message, false, state)

      [%EncodedTextChunk{tokens: content_tokens}] = rendered.content
      content = SpecialTokenizer.decode(content_tokens)

      assert String.contains?(content, "<tool_response>")
      assert String.contains?(content, "tool output")
      assert String.contains?(content, "</tool_response>")
    end

    test "encodes tool_calls blocks for assistant", %{state: state} do
      tool_call = Types.tool_call("search", ~s({"query":"elixir"}))
      message = Types.message("assistant", "Result", tool_calls: [tool_call])

      {rendered, _} = Qwen3.render_message(0, message, true, state)
      [%EncodedTextChunk{tokens: content_tokens}] = rendered.content
      content = SpecialTokenizer.decode(content_tokens)

      assert String.contains?(content, "<tool_call>")
      assert String.contains?(content, "\"name\":\"search\"")
    end
  end

  describe "Qwen3.parse_response/2" do
    test "extracts tool calls and strips blocks" do
      {:ok, state} = Qwen3.init(tokenizer: SpecialTokenizer)

      payload = Jason.encode!(%{"name" => "search", "arguments" => %{"query" => "elixir"}})

      response =
        "Hi<tool_call>\n" <> payload <> "\n</tool_call><|im_end|>"

      tokens = SpecialTokenizer.encode(response)

      {message, format_ok} = Qwen3.parse_response(tokens, state)

      assert format_ok == true
      assert message.content == "Hi"
      assert [%Types.ToolCall{} = tool_call] = message.tool_calls
      assert tool_call.function.name == "search"
    end

    test "returns format false on invalid tool call JSON" do
      {:ok, state} = Qwen3.init(tokenizer: SpecialTokenizer)
      tokens = SpecialTokenizer.encode("Hi<tool_call>{bad}</tool_call><|im_end|>")

      {message, format_ok} = Qwen3.parse_response(tokens, state)

      assert format_ok == false
      assert message.content =~ "Hi"
    end
  end

  describe "Qwen3DisableThinking" do
    test "adds empty thinking block for assistant" do
      {:ok, state} = Qwen3DisableThinking.init(tokenizer: SpecialTokenizer)
      message = Types.message("assistant", "Answer")

      {rendered, _} = Qwen3DisableThinking.render_message(0, message, true, state)
      [%EncodedTextChunk{tokens: content_tokens}] = rendered.content
      content = SpecialTokenizer.decode(content_tokens)

      assert String.contains?(content, "<think>\n\n</think>\n\n")
    end

    test "generation prompt prefixes empty thinking block" do
      {:ok, state} = Qwen3DisableThinking.init(tokenizer: SpecialTokenizer)
      messages = [Types.message("user", "Hello")]

      {model_input, _} =
        Renderer.build_generation_prompt(Qwen3DisableThinking, messages, "assistant", nil, state)

      %EncodedTextChunk{tokens: prefill_tokens} = List.last(model_input.chunks)
      assert SpecialTokenizer.decode(prefill_tokens) == "<think>\n\n</think>\n\n"
    end
  end

  describe "Qwen3Instruct" do
    test "does not inject <think> for assistant" do
      {:ok, state} = Qwen3Instruct.init(tokenizer: SpecialTokenizer)
      message = Types.message("assistant", "Hello")

      {rendered, _} = Qwen3Instruct.render_message(0, message, true, state)
      %EncodedTextChunk{tokens: prefix_tokens} = rendered.prefix
      prefix = SpecialTokenizer.decode(prefix_tokens)

      refute String.contains?(prefix, "<think>")
    end
  end
end
