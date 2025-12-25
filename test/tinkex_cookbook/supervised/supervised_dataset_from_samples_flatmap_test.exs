defmodule TinkexCookbook.Supervised.SupervisedDatasetFromSamplesFlatMapTest do
  use ExUnit.Case, async: true

  alias TinkexCookbook.Supervised.{Common, SupervisedDataset, SupervisedDatasetFromSamplesFlatMap}
  alias TinkexCookbook.Types.ModelInput

  defp datum_for(value) do
    model_input = ModelInput.from_ints([value, value + 1])
    Common.datum_from_model_input_weights(model_input, [0.0, 1.0])
  end

  test "flatmap dataset builds batches with multiple datums per sample" do
    samples = [%{"value" => 1}, %{"value" => 2}]

    dataset =
      SupervisedDatasetFromSamplesFlatMap.new(samples, 1, fn sample ->
        value = sample["value"]
        [datum_for(value), datum_for(value + 10)]
      end)

    batch0 = SupervisedDataset.get_batch(dataset, 0)
    batch1 = SupervisedDataset.get_batch(dataset, 1)

    assert length(batch0) == 2
    assert length(batch1) == 2
  end
end
