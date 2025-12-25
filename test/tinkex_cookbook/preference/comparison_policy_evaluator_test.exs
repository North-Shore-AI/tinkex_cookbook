defmodule TinkexCookbook.Preference.ComparisonPolicyEvaluatorTest do
  use ExUnit.Case, async: true

  alias TinkexCookbook.Preference.{Comparison, ComparisonPolicyEvaluator}
  alias TinkexCookbook.Renderers.{RoleColon, Types}
  alias TinkexCookbook.Test.MockTokenizer

  # Mock preference model that returns fixed scores
  defmodule MockPreferenceModel do
    defstruct [:score_value]

    def score(%__MODULE__{score_value: value}, %Comparison{}), do: value
  end

  # Mock sampling client that returns fixed response
  defmodule MockSamplingClient do
    defstruct [:response_tokens]

    def sample(%__MODULE__{response_tokens: tokens}, _prompt, _params, _opts \\ []) do
      response = %{
        sequences: [
          %{tokens: tokens, logprobs: nil}
        ]
      }

      {:ok, Task.async(fn -> {:ok, response} end)}
    end
  end

  describe "evaluate/2" do
    setup do
      {:ok, renderer_state} = RoleColon.init(tokenizer: MockTokenizer)

      comparison = %Comparison{
        prompt_conversation: [Types.message("user", "What is 2+2?")],
        completion_a: [Types.message("assistant", "4")],
        completion_b: nil
      }

      {:ok, renderer_state: renderer_state, comparison: comparison}
    end

    test "returns win_rate and stderr metrics", %{
      renderer_state: renderer_state,
      comparison: comparison
    } do
      # Preference model that prefers B (the policy completion) over A
      preference_model_builder = fn ->
        %MockPreferenceModel{score_value: 0.8}
      end

      # Response tokens that decode to "4" with end delimiter
      response_tokens = String.to_charlist("4\n\nUser:")

      sampling_client = %MockSamplingClient{response_tokens: response_tokens}

      evaluator = %ComparisonPolicyEvaluator{
        preference_model_builder: preference_model_builder,
        comparisons: [comparison],
        renderer_module: RoleColon,
        renderer_state: renderer_state,
        max_tokens: 100,
        both_ways: true,
        content_preprocessor: nil
      }

      {:ok, metrics} = ComparisonPolicyEvaluator.evaluate(evaluator, sampling_client)

      assert Map.has_key?(metrics, "win_rate")
      assert Map.has_key?(metrics, "stderr")
      assert is_number(metrics["win_rate"])
      assert is_number(metrics["stderr"])
    end

    test "computes correct normalized win_rate", %{
      renderer_state: renderer_state,
      comparison: comparison
    } do
      # r_0 = 1.0 (prefers new completion)
      # r_1 = 1.0 (after swap, same preference)
      # (r_0 - r_1 + 2) / 4.0 = (1.0 - 1.0 + 2) / 4.0 = 0.5
      preference_model_builder = fn ->
        %MockPreferenceModel{score_value: 1.0}
      end

      response_tokens = String.to_charlist("4\n\nUser:")
      sampling_client = %MockSamplingClient{response_tokens: response_tokens}

      evaluator = %ComparisonPolicyEvaluator{
        preference_model_builder: preference_model_builder,
        comparisons: [comparison],
        renderer_module: RoleColon,
        renderer_state: renderer_state,
        max_tokens: 100
      }

      {:ok, metrics} = ComparisonPolicyEvaluator.evaluate(evaluator, sampling_client)

      # With symmetric preference model, win_rate should be 0.5
      assert_in_delta metrics["win_rate"], 0.5, 0.01
    end

    test "applies content_preprocessor to completions", %{
      renderer_state: renderer_state,
      comparison: comparison
    } do
      # Track if preprocessor was called
      test_pid = self()

      preprocessor = fn content ->
        send(test_pid, {:preprocessor_called, content})
        String.upcase(content)
      end

      preference_model_builder = fn ->
        %MockPreferenceModel{score_value: 0.0}
      end

      response_tokens = String.to_charlist("hello\n\nUser:")
      sampling_client = %MockSamplingClient{response_tokens: response_tokens}

      evaluator = %ComparisonPolicyEvaluator{
        preference_model_builder: preference_model_builder,
        comparisons: [comparison],
        renderer_module: RoleColon,
        renderer_state: renderer_state,
        max_tokens: 100,
        content_preprocessor: preprocessor
      }

      {:ok, _metrics} = ComparisonPolicyEvaluator.evaluate(evaluator, sampling_client)

      assert_received {:preprocessor_called, "hello"}
    end

    test "processes multiple comparisons concurrently", %{
      renderer_state: renderer_state
    } do
      comparisons =
        for i <- 1..5 do
          %Comparison{
            prompt_conversation: [Types.message("user", "Question #{i}")],
            completion_a: [Types.message("assistant", "Answer #{i}")],
            completion_b: nil
          }
        end

      preference_model_builder = fn ->
        %MockPreferenceModel{score_value: 0.5}
      end

      response_tokens = String.to_charlist("answer\n\nUser:")
      sampling_client = %MockSamplingClient{response_tokens: response_tokens}

      evaluator = %ComparisonPolicyEvaluator{
        preference_model_builder: preference_model_builder,
        comparisons: comparisons,
        renderer_module: RoleColon,
        renderer_state: renderer_state,
        max_tokens: 100
      }

      {:ok, metrics} = ComparisonPolicyEvaluator.evaluate(evaluator, sampling_client)

      # With 5 comparisons, stderr should be computed
      assert is_number(metrics["stderr"])
      assert metrics["stderr"] >= 0.0
    end
  end
end
