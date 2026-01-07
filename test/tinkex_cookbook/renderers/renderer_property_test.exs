defmodule TinkexCookbook.Renderers.RendererPropertyTest do
  use ExUnit.Case, async: true

  alias CrucibleTrain.Renderers.{Renderer, TrainOnWhat, Types}
  alias CrucibleTrain.Types.ModelInput
  alias TinkexCookbook.Test.MockTokenizer

  @renderer CrucibleTrain.Renderers.Llama3

  test "generation prompt preserves message order (property-style)" do
    generator =
      StreamData.tuple({
        StreamData.string(:alphanumeric, min_length: 1, max_length: 12),
        StreamData.string(:alphanumeric, min_length: 1, max_length: 12)
      })

    generator
    |> Enum.take(50)
    |> Enum.each(fn {user, assistant} ->
      user = "USER_" <> user
      assistant = "ASSISTANT_" <> assistant
      {:ok, state} = @renderer.init(tokenizer: MockTokenizer)

      messages = [
        Types.message("user", user),
        Types.message("assistant", assistant)
      ]

      {model_input, _} =
        Renderer.build_generation_prompt(@renderer, messages, "assistant", nil, state)

      decoded = MockTokenizer.decode(ModelInput.all_tokens(model_input))

      user_idx = index_of(decoded, user)
      assistant_idx = index_of(decoded, assistant)

      assert user_idx != nil
      assert assistant_idx != nil
      assert user_idx < assistant_idx
    end)
  end

  test "supervised example weights align with model input length (property-style)" do
    generator =
      StreamData.tuple({
        StreamData.string(:alphanumeric, min_length: 1, max_length: 12),
        StreamData.string(:alphanumeric, min_length: 1, max_length: 12)
      })

    generator
    |> Enum.take(50)
    |> Enum.each(fn {user, assistant} ->
      user = "USER_" <> user
      assistant = "ASSISTANT_" <> assistant
      {:ok, state} = @renderer.init(tokenizer: MockTokenizer)

      messages = [
        Types.message("user", user),
        Types.message("assistant", assistant)
      ]

      {model_input, weights} =
        Renderer.build_supervised_example(
          @renderer,
          messages,
          TrainOnWhat.last_assistant_message(),
          state
        )

      assert length(weights) == ModelInput.length(model_input)
      assert ModelInput.length(model_input) > 0
    end)
  end

  defp index_of(haystack, needle) do
    case :binary.match(haystack, needle) do
      {idx, _} -> idx
      :nomatch -> nil
    end
  end
end
