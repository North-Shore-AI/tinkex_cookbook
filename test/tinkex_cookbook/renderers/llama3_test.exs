defmodule TinkexCookbook.Renderers.Llama3Test do
  @moduledoc """
  Tests for the Llama3 renderer.

  These tests verify that the Llama3 renderer produces the correct token sequences
  matching the Llama 3 chat template format.
  """
  use ExUnit.Case, async: true

  alias TinkexCookbook.Renderers.{Llama3, Renderer, TrainOnWhat, Types}
  alias TinkexCookbook.Test.MockTokenizer
  alias TinkexCookbook.Types.{EncodedTextChunk, ModelInput}

  describe "init/1" do
    test "returns {:ok, state} with tokenizer" do
      assert {:ok, state} = Llama3.init(tokenizer: MockTokenizer)
      assert state.tokenizer == MockTokenizer
    end

    test "raises when tokenizer is missing" do
      assert_raise KeyError, fn ->
        Llama3.init([])
      end
    end
  end

  describe "bos_tokens/1" do
    test "returns Llama3 BOS tokens for <|begin_of_text|>" do
      {:ok, state} = Llama3.init(tokenizer: MockTokenizer)
      bos = Llama3.bos_tokens(state)

      assert is_list(bos)
      # Should encode "<|begin_of_text|>"
      expected = MockTokenizer.encode("<|begin_of_text|>")
      assert bos == expected
    end
  end

  describe "stop_sequences/1" do
    test "returns eot_id token" do
      {:ok, state} = Llama3.init(tokenizer: MockTokenizer)
      stops = Llama3.stop_sequences(state)

      assert is_list(stops)
      # Should return the encoded <|eot_id|> token
      expected = MockTokenizer.encode("<|eot_id|>")
      assert stops == expected
    end
  end

  describe "render_message/4" do
    setup do
      {:ok, state} = Llama3.init(tokenizer: MockTokenizer)
      %{state: state}
    end

    test "renders user message with correct prefix format", %{state: state} do
      message = Types.message("user", "Hello!")

      {rendered, _new_state} = Llama3.render_message(0, message, false, state)

      assert %Types.RenderedMessage{} = rendered
      assert %EncodedTextChunk{tokens: prefix_tokens} = rendered.prefix

      # Prefix should be: <|start_header_id|>user<|end_header_id|>\n\n
      expected_prefix = MockTokenizer.encode("<|start_header_id|>user<|end_header_id|>\n\n")
      assert prefix_tokens == expected_prefix
    end

    test "renders assistant message with correct prefix format", %{state: state} do
      message = Types.message("assistant", "I'm fine!")

      {rendered, _new_state} = Llama3.render_message(0, message, false, state)

      assert %EncodedTextChunk{tokens: prefix_tokens} = rendered.prefix

      expected_prefix = MockTokenizer.encode("<|start_header_id|>assistant<|end_header_id|>\n\n")
      assert prefix_tokens == expected_prefix
    end

    test "renders system message with correct prefix format", %{state: state} do
      message = Types.message("system", "You are helpful.")

      {rendered, _new_state} = Llama3.render_message(0, message, false, state)

      assert %EncodedTextChunk{tokens: prefix_tokens} = rendered.prefix

      expected_prefix = MockTokenizer.encode("<|start_header_id|>system<|end_header_id|>\n\n")
      assert prefix_tokens == expected_prefix
    end

    test "content includes message text with eot_id", %{state: state} do
      message = Types.message("user", "Hello!")

      {rendered, _new_state} = Llama3.render_message(0, message, false, state)

      assert is_list(rendered.content)
      refute Enum.empty?(rendered.content)

      # Content should be: Hello!<|eot_id|>
      [%EncodedTextChunk{tokens: content_tokens} | _] = rendered.content
      expected_content = MockTokenizer.encode("Hello!<|eot_id|>")
      assert content_tokens == expected_content
    end

    test "does not include suffix (Llama3 format has no suffix)", %{state: state} do
      message = Types.message("user", "Hello!")

      {rendered, _new_state} = Llama3.render_message(0, message, false, state)

      # Llama3 doesn't use suffix - eot_id is part of content
      assert rendered.suffix == nil
    end

    test "preserves state through multiple renders", %{state: state} do
      message1 = Types.message("user", "First")
      message2 = Types.message("assistant", "Second")

      {_rendered1, state1} = Llama3.render_message(0, message1, false, state)
      {_rendered2, state2} = Llama3.render_message(1, message2, true, state1)

      assert state2.tokenizer == MockTokenizer
    end
  end

  describe "parse_response/2" do
    setup do
      {:ok, state} = Llama3.init(tokenizer: MockTokenizer)
      %{state: state}
    end

    test "parses response with single eot_id correctly", %{state: state} do
      response_text = "The answer is 42.<|eot_id|>"
      tokens = MockTokenizer.encode(response_text)

      {message, is_complete} = Llama3.parse_response(tokens, state)

      assert message.role == "assistant"
      assert message.content == "The answer is 42."
      assert is_complete == true
    end

    test "handles response without eot_id (incomplete)", %{state: state} do
      response_text = "Partial response"
      tokens = MockTokenizer.encode(response_text)

      {message, is_complete} = Llama3.parse_response(tokens, state)

      assert message.role == "assistant"
      assert message.content == "Partial response"
      assert is_complete == false
    end

    test "raises on double eot_id", %{state: state} do
      response_text = "Text<|eot_id|>More<|eot_id|>"
      tokens = MockTokenizer.encode(response_text)

      assert_raise RuntimeError, ~r/expected to split into 1 or 2/, fn ->
        Llama3.parse_response(tokens, state)
      end
    end
  end

  describe "integration with Renderer module" do
    setup do
      {:ok, state} = Llama3.init(tokenizer: MockTokenizer)
      %{state: state}
    end

    test "build_supervised_example produces valid output", %{state: state} do
      messages = [
        Types.message("user", "Hello!"),
        Types.message("assistant", "Hi there!")
      ]

      {model_input, weights} =
        Renderer.build_supervised_example(
          Llama3,
          messages,
          TrainOnWhat.all_assistant_messages(),
          state
        )

      assert %ModelInput{} = model_input
      assert is_list(weights)
      assert ModelInput.length(model_input) == length(weights)
    end

    test "build_supervised_example with last_assistant_message weights correctly", %{state: state} do
      messages = [
        Types.message("user", "Question 1?"),
        Types.message("assistant", "Answer 1"),
        Types.message("user", "Question 2?"),
        Types.message("assistant", "Answer 2")
      ]

      {_model_input, weights} =
        Renderer.build_supervised_example(
          Llama3,
          messages,
          TrainOnWhat.last_assistant_message(),
          state
        )

      # Only the last assistant message content should have weight 1.0
      weight_sum = Enum.sum(weights)
      assert weight_sum > 0

      # Should have some 0.0 weights (for non-last messages)
      assert Enum.any?(weights, fn w -> w == 0.0 end)
      # Should have some 1.0 weights (for last assistant)
      assert Enum.any?(weights, fn w -> w == 1.0 end)
    end

    test "build_supervised_example with all_tokens weights everything", %{state: state} do
      messages = [
        Types.message("user", "Hi"),
        Types.message("assistant", "Hello!")
      ]

      {model_input, weights} =
        Renderer.build_supervised_example(
          Llama3,
          messages,
          TrainOnWhat.all_tokens(),
          state
        )

      # All weights should be 1.0
      assert Enum.all?(weights, fn w -> w == 1.0 end)
      assert length(weights) == ModelInput.length(model_input)
    end

    test "build_generation_prompt produces valid output", %{state: state} do
      messages = [
        Types.message("user", "What is 2+2?")
      ]

      {model_input, _final_state} =
        Renderer.build_generation_prompt(
          Llama3,
          messages,
          "assistant",
          nil,
          state
        )

      assert %ModelInput{} = model_input
      # Should have BOS + user message + partial assistant prefix
      assert length(model_input.chunks) >= 2
    end

    test "build_generation_prompt with prefill adds prefill tokens", %{state: state} do
      messages = [
        Types.message("user", "What is 2+2?")
      ]

      {model_input, _final_state} =
        Renderer.build_generation_prompt(
          Llama3,
          messages,
          "assistant",
          "The answer is",
          state
        )

      # Last chunk should be the prefill
      %EncodedTextChunk{tokens: prefill_tokens} = List.last(model_input.chunks)
      expected_prefill = MockTokenizer.encode("The answer is")
      assert prefill_tokens == expected_prefill
    end
  end

  describe "Llama3 format compliance" do
    setup do
      {:ok, state} = Llama3.init(tokenizer: MockTokenizer)
      %{state: state}
    end

    test "full conversation renders in correct Llama3 format", %{state: state} do
      messages = [
        Types.message("system", "You are a helpful assistant."),
        Types.message("user", "Hello"),
        Types.message("assistant", "Hi there!")
      ]

      {model_input, _weights} =
        Renderer.build_supervised_example(
          Llama3,
          messages,
          TrainOnWhat.all_messages(),
          state
        )

      # Get all tokens
      all_tokens = ModelInput.all_tokens(model_input)
      decoded = MockTokenizer.decode(all_tokens)

      # Should contain BOS
      assert String.contains?(decoded, "<|begin_of_text|>")

      # Should contain all role headers
      assert String.contains?(decoded, "<|start_header_id|>system<|end_header_id|>")
      assert String.contains?(decoded, "<|start_header_id|>user<|end_header_id|>")
      assert String.contains?(decoded, "<|start_header_id|>assistant<|end_header_id|>")

      # Should contain all content
      assert String.contains?(decoded, "You are a helpful assistant.")
      assert String.contains?(decoded, "Hello")
      assert String.contains?(decoded, "Hi there!")

      # Should contain eot_id tokens
      assert String.contains?(decoded, "<|eot_id|>")
    end
  end
end
