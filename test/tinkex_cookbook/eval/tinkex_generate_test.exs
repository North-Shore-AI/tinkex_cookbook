defmodule TinkexCookbook.Eval.TinkexGenerateTest do
  @moduledoc """
  Tests for the TinkexGenerate adapter.

  These tests verify that TinkexGenerate correctly implements the
  CrucibleHarness.Generate behaviour using Tinkex sampling.
  """
  use ExUnit.Case, async: true

  alias TinkexCookbook.Eval.TinkexGenerate
  alias TinkexCookbook.Test.MockTinkex

  describe "generate/2" do
    setup do
      sampling_client = MockTinkex.SamplingClient.new()

      config = %{
        model: "test-model",
        temperature: 0.7,
        max_tokens: 100,
        stop: ["<|eot_id|>"],
        sampling_client: sampling_client
      }

      %{config: config}
    end

    test "returns {:ok, response} with content", %{config: config} do
      messages = [
        %{role: "user", content: "Hello!"}
      ]

      result = TinkexGenerate.generate(messages, config)

      assert {:ok, response} = result
      assert is_binary(response.content)
      assert response.content == "Hello!"
    end

    test "response includes finish_reason", %{config: config} do
      messages = [
        %{role: "user", content: "Test"}
      ]

      {:ok, response} = TinkexGenerate.generate(messages, config)

      assert Map.has_key?(response, :finish_reason)
      assert is_binary(response.finish_reason)
    end

    test "response includes usage statistics", %{config: config} do
      messages = [
        %{role: "user", content: "Test"}
      ]

      {:ok, response} = TinkexGenerate.generate(messages, config)

      assert Map.has_key?(response, :usage)
      assert is_map(response.usage)
    end

    test "handles multi-turn conversations", %{config: config} do
      messages = [
        %{role: "system", content: "You are helpful."},
        %{role: "user", content: "What is 2+2?"},
        %{role: "assistant", content: "4"},
        %{role: "user", content: "And 3+3?"}
      ]

      result = TinkexGenerate.generate(messages, config)

      assert {:ok, _response} = result
    end
  end

  describe "build_model_input/3" do
    setup do
      %{
        renderer_module: TinkexCookbook.Renderers.Llama3,
        tokenizer: TinkexCookbook.Test.MockTokenizer
      }
    end

    test "converts messages to model input", %{
      renderer_module: renderer_module,
      tokenizer: tokenizer
    } do
      {:ok, state} = renderer_module.init(tokenizer: tokenizer)

      messages = [
        %{role: "user", content: "Hello"}
      ]

      {model_input, _state} =
        TinkexGenerate.build_model_input(messages, renderer_module, state)

      assert %TinkexCookbook.Types.ModelInput{} = model_input
    end
  end

  describe "parse_response/3" do
    setup do
      %{
        renderer_module: TinkexCookbook.Renderers.Llama3,
        tokenizer: TinkexCookbook.Test.MockTokenizer
      }
    end

    test "parses response tokens into content", %{
      renderer_module: renderer_module,
      tokenizer: tokenizer
    } do
      {:ok, state} = renderer_module.init(tokenizer: tokenizer)

      sample_response = %{
        sequences: [
          %{
            tokens: String.to_charlist("The answer is 4.<|eot_id|>"),
            text: "The answer is 4.<|eot_id|>",
            stop_reason: "end_of_turn",
            logprobs: []
          }
        ]
      }

      {message, is_complete} =
        TinkexGenerate.parse_response(sample_response, renderer_module, state)

      assert message.role == "assistant"
      assert is_binary(message.content)
      assert is_complete == true or is_complete == false
    end
  end
end
