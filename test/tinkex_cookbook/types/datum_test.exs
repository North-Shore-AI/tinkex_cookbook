defmodule TinkexCookbook.Types.DatumTest do
  use ExUnit.Case, async: true

  alias TinkexCookbook.Types.{Datum, ModelInput, TensorData}

  describe "new/2" do
    test "creates Datum with model_input and loss_fn_inputs" do
      model_input = ModelInput.from_ints([1, 2, 3])
      weights = TensorData.from_list([1.0, 1.0], :float32)
      target_tokens = TensorData.from_list([2, 3], :int64)

      datum =
        Datum.new(model_input, %{
          "weights" => weights,
          "target_tokens" => target_tokens
        })

      assert datum.model_input == model_input
      assert datum.loss_fn_inputs["weights"] == weights
      assert datum.loss_fn_inputs["target_tokens"] == target_tokens
    end
  end

  describe "get_weights/1" do
    test "returns weights TensorData" do
      model_input = ModelInput.from_ints([1, 2, 3])
      weights = TensorData.from_list([1.0, 1.0], :float32)
      target_tokens = TensorData.from_list([2, 3], :int64)

      datum =
        Datum.new(model_input, %{
          "weights" => weights,
          "target_tokens" => target_tokens
        })

      assert Datum.get_weights(datum) == weights
    end
  end

  describe "get_target_tokens/1" do
    test "returns target_tokens TensorData" do
      model_input = ModelInput.from_ints([1, 2, 3])
      weights = TensorData.from_list([1.0, 1.0], :float32)
      target_tokens = TensorData.from_list([2, 3], :int64)

      datum =
        Datum.new(model_input, %{
          "weights" => weights,
          "target_tokens" => target_tokens
        })

      assert Datum.get_target_tokens(datum) == target_tokens
    end
  end

  describe "num_loss_tokens/1" do
    test "returns sum of weights" do
      model_input = ModelInput.from_ints([1, 2, 3, 4, 5])
      weights = TensorData.from_list([0.0, 1.0, 1.0, 1.0], :float32)
      target_tokens = TensorData.from_list([2, 3, 4, 5], :int64)

      datum =
        Datum.new(model_input, %{
          "weights" => weights,
          "target_tokens" => target_tokens
        })

      assert Datum.num_loss_tokens(datum) == 3.0
    end
  end
end
