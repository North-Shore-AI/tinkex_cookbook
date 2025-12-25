defmodule TinkexCookbook.Distillation.TrainOnPolicyTest do
  use ExUnit.Case, async: true

  alias TinkexCookbook.Distillation.TrainOnPolicy
  alias TinkexCookbook.Test.MockTinkex
  alias TinkexCookbook.Types.{Datum, ModelInput, TensorData}

  test "incorporate_kl_penalty adjusts advantages and returns metrics" do
    datum = %Datum{
      model_input: ModelInput.from_ints([1, 2]),
      loss_fn_inputs: %{
        "logprobs" => TensorData.from_list([0.0, 0.0], :float32),
        "mask" => TensorData.from_list([1.0, 1.0], :float32),
        "advantages" => TensorData.from_list([0.0, 0.0], :float32),
        "target_tokens" => TensorData.from_list([2, 3], :int64)
      }
    }

    teacher_client = MockTinkex.SamplingClient.new()

    {updated, metrics} =
      TrainOnPolicy.incorporate_kl_penalty([datum], [teacher_client], [0], 1.0, 0.0)

    [updated_datum] = updated

    assert TensorData.to_list(updated_datum.loss_fn_inputs["advantages"]) == [-1.0, -1.0]
    assert metrics["teacher_kl"] == 1.0
    assert metrics["teacher_kl/dataset_0"] == 1.0
  end
end
