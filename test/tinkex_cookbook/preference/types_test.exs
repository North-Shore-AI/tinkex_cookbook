defmodule TinkexCookbook.Preference.TypesTest do
  use ExUnit.Case, async: true

  import ExUnit.CaptureLog

  alias TinkexCookbook.Preference.{
    Comparison,
    ComparisonRendererFromChatRenderer,
    LabeledComparison,
    PreferenceModel,
    PreferenceModelFromChatRenderer
  }

  alias TinkexCookbook.Renderers.{Helpers, RoleColon, Types}
  alias TinkexCookbook.Test.MockTinkex
  alias TinkexCookbook.Test.MockTokenizer
  alias TinkexCookbook.Types.ModelInput

  defp renderer do
    {:ok, state} = RoleColon.init(tokenizer: MockTokenizer)
    %ComparisonRendererFromChatRenderer{renderer_module: RoleColon, renderer_state: state}
  end

  defp comparison do
    %Comparison{
      prompt_conversation: [Types.message("user", "Hello")],
      completion_a: [Types.message("assistant", "A")],
      completion_b: [Types.message("assistant", "B")]
    }
  end

  test "comparison swap flips completions" do
    comparison = comparison()

    swapped = Comparison.swap(comparison)

    assert swapped.completion_a == comparison.completion_b
    assert swapped.completion_b == comparison.completion_a
  end

  test "labeled comparison swap flips label" do
    labeled = %LabeledComparison{comparison: comparison(), label: "A"}

    swapped = LabeledComparison.swap(labeled)

    assert swapped.label == "B"
    assert swapped.comparison.completion_a == labeled.comparison.completion_b
  end

  test "comparison renderer includes completion separators in prompt" do
    model_input =
      ComparisonRendererFromChatRenderer.build_generation_prompt(renderer(), comparison())

    decoded = Helpers.decode(MockTokenizer, ModelInput.all_tokens(model_input))

    assert decoded =~ "==== Completion A ===="
    assert decoded =~ "==== Completion B ===="
    assert decoded =~ "==== Preference ===="
  end

  test "comparison renderer truncates to first weight token" do
    labeled = %LabeledComparison{comparison: comparison(), label: "A"}

    {model_input, weights} =
      ComparisonRendererFromChatRenderer.to_model_input_weights(renderer(), labeled)

    assert length(weights) == ModelInput.length(model_input)
    assert List.last(weights) == 1.0
    assert Enum.count(weights, &(&1 == 1.0)) == 1
  end

  test "preference model from chat renderer scores outputs" do
    pref_renderer = renderer()

    comparison = comparison()

    assert -1.0 ==
             PreferenceModel.score(
               %PreferenceModelFromChatRenderer{
                 comparison_renderer: pref_renderer,
                 sampling_client: MockTinkex.SamplingClient.new(response_tokens: ~c"A")
               },
               comparison
             )

    assert 1.0 ==
             PreferenceModel.score(
               %PreferenceModelFromChatRenderer{
                 comparison_renderer: pref_renderer,
                 sampling_client: MockTinkex.SamplingClient.new(response_tokens: ~c"B")
               },
               comparison
             )

    assert 0.0 ==
             PreferenceModel.score(
               %PreferenceModelFromChatRenderer{
                 comparison_renderer: pref_renderer,
                 sampling_client: MockTinkex.SamplingClient.new(response_tokens: ~c"Tie")
               },
               comparison
             )

    log =
      capture_log(fn ->
        assert 0.0 ==
                 PreferenceModel.score(
                   %PreferenceModelFromChatRenderer{
                     comparison_renderer: pref_renderer,
                     sampling_client: MockTinkex.SamplingClient.new(response_tokens: ~c"Unknown")
                   },
                   comparison
                 )
      end)

    assert log =~ "Invalid preference model output"
    assert log =~ "Unknown"
  end
end
