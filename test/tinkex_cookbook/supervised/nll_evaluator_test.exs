# Mock dataset for testing - defined before test module
defmodule TinkexCookbook.Supervised.NLLEvaluatorTest.MockDataset do
  @behaviour TinkexCookbook.Supervised.SupervisedDataset

  defstruct [:batches]

  @impl true
  def get_batch(%{batches: batches}, index), do: Enum.at(batches, index)

  @impl true
  def length(%{batches: batches}), do: Kernel.length(batches)

  @impl true
  def set_epoch(dataset, _seed), do: dataset
end

defmodule TinkexCookbook.Supervised.NLLEvaluatorTest do
  use ExUnit.Case, async: true

  alias TinkexCookbook.Supervised.NLLEvaluator
  alias TinkexCookbook.Supervised.NLLEvaluatorTest.MockDataset
  alias TinkexCookbook.Types.{Datum, EncodedTextChunk, ModelInput, TensorData}

  describe "new/2" do
    test "creates evaluator with default name" do
      datums = [make_datum()]
      evaluator = NLLEvaluator.new(datums)

      assert evaluator.name == "test"
      assert evaluator.data == datums
    end

    test "creates evaluator with custom name" do
      datums = [make_datum()]
      evaluator = NLLEvaluator.new(datums, name: "validation")

      assert evaluator.name == "validation"
    end
  end

  describe "from_dataset/2" do
    test "collects all batches from dataset" do
      dataset = %MockDataset{batches: [[make_datum()], [make_datum(), make_datum()]]}
      evaluator = NLLEvaluator.from_dataset(dataset, name: "test")

      assert length(evaluator.data) == 3
    end
  end

  # Helper to create a minimal datum
  defp make_datum do
    model_input = %ModelInput{
      chunks: [%EncodedTextChunk{tokens: [1, 2, 3]}]
    }

    Datum.new(model_input, %{
      "weights" => TensorData.from_list([1.0, 1.0], :float32),
      "target_tokens" => TensorData.from_list([2, 3], :int64)
    })
  end
end
