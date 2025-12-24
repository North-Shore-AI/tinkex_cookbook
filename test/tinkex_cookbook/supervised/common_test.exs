defmodule TinkexCookbook.Supervised.CommonTest do
  use ExUnit.Case, async: true

  alias TinkexCookbook.Supervised.Common
  alias TinkexCookbook.Types.{EncodedTextChunk, ImageChunk, ModelInput, TensorData}

  describe "datum_from_model_input_weights/3" do
    test "creates datum with right-shifted inputs and left-shifted targets" do
      # Input: [1, 2, 3, 4, 5]
      # Expected input: [1, 2, 3, 4] (last token removed)
      # Expected targets: [2, 3, 4, 5] (first token removed)
      model_input = ModelInput.from_ints([1, 2, 3, 4, 5])
      weights = [0.0, 1.0, 1.0, 1.0, 1.0]

      datum = Common.datum_from_model_input_weights(model_input, weights)

      assert ModelInput.all_tokens(datum.model_input) == [1, 2, 3, 4]
      assert TensorData.to_list(datum.loss_fn_inputs["target_tokens"]) == [2, 3, 4, 5]
      # Weights are sliced: skip first, take 4
      assert TensorData.to_list(datum.loss_fn_inputs["weights"]) == [1.0, 1.0, 1.0, 1.0]
    end

    test "truncates to max_length" do
      model_input = ModelInput.from_ints([1, 2, 3, 4, 5, 6, 7, 8])
      weights = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

      datum = Common.datum_from_model_input_weights(model_input, weights, 5)

      # Truncated to 5 tokens, then right-shifted to 4
      assert ModelInput.all_tokens(datum.model_input) == [1, 2, 3, 4]
      assert TensorData.to_list(datum.loss_fn_inputs["target_tokens"]) == [2, 3, 4, 5]
    end

    test "preserves chunk order when truncating across multiple chunks" do
      chunks = [
        EncodedTextChunk.new([1, 2, 3]),
        EncodedTextChunk.new([4, 5, 6]),
        EncodedTextChunk.new([7, 8, 9])
      ]

      model_input = ModelInput.new(chunks)
      weights = List.duplicate(1.0, 9)

      datum = Common.datum_from_model_input_weights(model_input, weights, 7)

      assert ModelInput.all_tokens(datum.model_input) == [1, 2, 3, 4, 5, 6]
      assert TensorData.to_list(datum.loss_fn_inputs["target_tokens"]) == [2, 3, 4, 5, 6, 7]
    end

    test "drops trailing image chunks before shifting" do
      chunks = [
        EncodedTextChunk.new([1, 2, 3]),
        ImageChunk.new(<<1, 2>>, "jpeg", 2)
      ]

      model_input = ModelInput.new(chunks)
      weights = [0.0, 1.0, 1.0, 1.0, 1.0]

      datum = Common.datum_from_model_input_weights(model_input, weights)

      assert ModelInput.all_tokens(datum.model_input) == [1, 2]
      assert TensorData.to_list(datum.loss_fn_inputs["target_tokens"]) == [2, 3]
      assert TensorData.to_list(datum.loss_fn_inputs["weights"]) == [1.0, 1.0]
    end

    test "handles minimal 2-token input" do
      model_input = ModelInput.from_ints([1, 2])
      weights = [0.0, 1.0]

      datum = Common.datum_from_model_input_weights(model_input, weights)

      assert ModelInput.all_tokens(datum.model_input) == [1]
      assert TensorData.to_list(datum.loss_fn_inputs["target_tokens"]) == [2]
      assert TensorData.to_list(datum.loss_fn_inputs["weights"]) == [1.0]
    end

    test "raises for single token input" do
      model_input = ModelInput.from_ints([1])
      weights = [1.0]

      assert_raise ArgumentError, ~r/need at least 2 tokens/, fn ->
        Common.datum_from_model_input_weights(model_input, weights)
      end
    end
  end

  describe "compute_mean_nll/2" do
    test "computes weighted mean negative log likelihood" do
      logprobs = [TensorData.from_list([-1.0, -2.0, -3.0], :float32)]
      weights = [TensorData.from_list([1.0, 1.0, 1.0], :float32)]

      nll = Common.compute_mean_nll(logprobs, weights)

      # -(-1.0 * 1.0 + -2.0 * 1.0 + -3.0 * 1.0) / 3.0 = 6.0 / 3.0 = 2.0
      assert_in_delta nll, 2.0, 0.001
    end

    test "handles zero weights" do
      logprobs = [TensorData.from_list([-1.0, -2.0], :float32)]
      weights = [TensorData.from_list([0.0, 0.0], :float32)]

      nll = Common.compute_mean_nll(logprobs, weights)

      assert nll == :nan
    end

    test "handles multiple batches" do
      logprobs = [
        TensorData.from_list([-1.0, -2.0], :float32),
        TensorData.from_list([-3.0, -4.0], :float32)
      ]

      weights = [
        TensorData.from_list([1.0, 1.0], :float32),
        TensorData.from_list([1.0, 1.0], :float32)
      ]

      nll = Common.compute_mean_nll(logprobs, weights)

      # -(-1.0 - 2.0 - 3.0 - 4.0) / 4.0 = 10.0 / 4.0 = 2.5
      assert_in_delta nll, 2.5, 0.001
    end
  end
end
