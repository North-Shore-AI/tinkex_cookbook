defmodule TinkexCookbook.Renderers.RendererTest do
  use ExUnit.Case, async: true

  alias TinkexCookbook.Renderers.{Renderer, TrainOnWhat}
  alias TinkexCookbook.Renderers.Types
  alias TinkexCookbook.Types.{EncodedTextChunk, ModelInput}

  # Mock tokenizer for testing
  defmodule MockTokenizer do
    @doc "Mock encode function that returns predictable token sequences"
    def encode(text, _opts \\ []) do
      # Simple encoding: each character becomes a token (ASCII value)
      text |> String.to_charlist()
    end

    def decode(tokens) do
      tokens |> List.to_string()
    end

    def bos_token, do: "<s>"
    def eos_token, do: "</s>"
  end

  # Test implementation of Renderer for Llama3
  defmodule TestLlama3Renderer do
    @behaviour TinkexCookbook.Renderers.Renderer

    @impl true
    def init(opts) do
      {:ok, %{tokenizer: Keyword.fetch!(opts, :tokenizer)}}
    end

    @impl true
    def render_message(_idx, message, _is_last, state) do
      tokenizer = state.tokenizer
      role = message.role

      prefix_str = "<|start_header_id|>#{role}<|end_header_id|>\n\n"
      content_str = "#{Types.ensure_text(message.content)}<|eot_id|>"

      prefix_tokens = tokenizer.encode(prefix_str)
      content_tokens = tokenizer.encode(content_str)

      rendered = %Types.RenderedMessage{
        prefix: %EncodedTextChunk{tokens: prefix_tokens},
        content: [%EncodedTextChunk{tokens: content_tokens}],
        suffix: nil
      }

      {rendered, state}
    end

    @impl true
    def bos_tokens(state) do
      state.tokenizer.encode("<|begin_of_text|>")
    end

    @impl true
    def stop_sequences(_state), do: ["<|eot_id|>"]
  end

  describe "Renderer behaviour" do
    test "init/1 returns {:ok, state}" do
      assert {:ok, state} = TestLlama3Renderer.init(tokenizer: MockTokenizer)
      assert state.tokenizer == MockTokenizer
    end

    test "render_message/4 returns RenderedMessage and state" do
      {:ok, state} = TestLlama3Renderer.init(tokenizer: MockTokenizer)
      message = Types.message("user", "Hello!")

      {rendered, _new_state} = TestLlama3Renderer.render_message(0, message, false, state)

      assert %Types.RenderedMessage{} = rendered
      assert rendered.prefix != nil
      assert is_list(rendered.content)
    end

    test "bos_tokens/1 returns BOS token sequence" do
      {:ok, state} = TestLlama3Renderer.init(tokenizer: MockTokenizer)
      bos = TestLlama3Renderer.bos_tokens(state)

      assert is_list(bos)
      assert bos != []
    end

    test "stop_sequences/1 returns stop sequences" do
      {:ok, state} = TestLlama3Renderer.init(tokenizer: MockTokenizer)
      stops = TestLlama3Renderer.stop_sequences(state)

      assert is_list(stops)
      assert "<|eot_id|>" in stops
    end
  end

  describe "Renderer.build_supervised_example/4" do
    setup do
      {:ok, state} = TestLlama3Renderer.init(tokenizer: MockTokenizer)
      %{state: state}
    end

    test "returns ModelInput and weights for single user message", %{state: state} do
      messages = [Types.message("user", "Hello!")]

      {model_input, weights} =
        Renderer.build_supervised_example(
          TestLlama3Renderer,
          messages,
          TrainOnWhat.all_messages(),
          state
        )

      assert %ModelInput{} = model_input
      assert is_list(weights)
      assert ModelInput.length(model_input) == length(weights)
    end

    test "assigns weight=1 to assistant content with all_assistant_messages", %{state: state} do
      messages = [
        Types.message("user", "Hi"),
        Types.message("assistant", "Hello!")
      ]

      {_model_input, weights} =
        Renderer.build_supervised_example(
          TestLlama3Renderer,
          messages,
          TrainOnWhat.all_assistant_messages(),
          state
        )

      # Some weights should be 1.0 (assistant content)
      assert Enum.any?(weights, fn w -> w == 1.0 end)
      # Some weights should be 0.0 (user content and prefixes)
      assert Enum.any?(weights, fn w -> w == 0.0 end)
    end

    test "assigns weight=1 only to last assistant with last_assistant_message", %{state: state} do
      messages = [
        Types.message("user", "Hi"),
        Types.message("assistant", "Hello!"),
        Types.message("user", "How are you?"),
        Types.message("assistant", "I'm good!")
      ]

      {_model_input, weights} =
        Renderer.build_supervised_example(
          TestLlama3Renderer,
          messages,
          TrainOnWhat.last_assistant_message(),
          state
        )

      # Only the last assistant message should have weight
      weight_sum = Enum.sum(weights)
      assert weight_sum > 0
    end

    test "assigns weight=1 to all tokens with all_tokens", %{state: state} do
      messages = [
        Types.message("user", "Hi"),
        Types.message("assistant", "Hello!")
      ]

      {model_input, weights} =
        Renderer.build_supervised_example(
          TestLlama3Renderer,
          messages,
          TrainOnWhat.all_tokens(),
          state
        )

      # All weights should be 1.0
      assert Enum.all?(weights, fn w -> w == 1.0 end)
      assert length(weights) == ModelInput.length(model_input)
    end

    test "requires trainable field for customized", %{state: state} do
      messages = [
        Types.message("user", "Hi"),
        Types.message("assistant", "Hello!")
      ]

      assert_raise ArgumentError, ~r/trainable field/, fn ->
        Renderer.build_supervised_example(
          TestLlama3Renderer,
          messages,
          TrainOnWhat.customized(),
          state
        )
      end
    end

    test "rejects trainable field for non-customized", %{state: state} do
      messages = [
        Types.message("user", "Hi", trainable: false),
        Types.message("assistant", "Hello!", trainable: true)
      ]

      assert_raise ArgumentError, ~r/trainable field/, fn ->
        Renderer.build_supervised_example(
          TestLlama3Renderer,
          messages,
          TrainOnWhat.all_messages(),
          state
        )
      end
    end

    test "uses per-message trainable flags for customized", %{state: state} do
      messages = [
        Types.message("user", "Hi", trainable: false),
        Types.message("assistant", "Hello!", trainable: true)
      ]

      {_model_input, weights} =
        Renderer.build_supervised_example(
          TestLlama3Renderer,
          messages,
          TrainOnWhat.customized(),
          state
        )

      assert Enum.any?(weights, fn w -> w == 1.0 end)
      assert Enum.any?(weights, fn w -> w == 0.0 end)
    end

    test "assigns weights to user/system messages for all_user_and_system_messages",
         %{state: state} do
      messages = [
        Types.message("user", "Hi"),
        Types.message("assistant", "Hello!")
      ]

      {_model_input, weights} =
        Renderer.build_supervised_example(
          TestLlama3Renderer,
          messages,
          TrainOnWhat.all_user_and_system_messages(),
          state
        )

      assert Enum.any?(weights, fn w -> w == 1.0 end)
      assert Enum.any?(weights, fn w -> w == 0.0 end)
    end
  end

  describe "Renderer.build_generation_prompt/5" do
    setup do
      {:ok, state} = TestLlama3Renderer.init(tokenizer: MockTokenizer)
      %{state: state}
    end

    test "appends prefill tokens when provided", %{state: state} do
      messages = [Types.message("user", "Hi")]

      {model_input, _state} =
        Renderer.build_generation_prompt(
          TestLlama3Renderer,
          messages,
          "assistant",
          "OK",
          state
        )

      assert %EncodedTextChunk{tokens: tokens} = List.last(model_input.chunks)
      assert tokens == MockTokenizer.encode("OK")
    end
  end
end
