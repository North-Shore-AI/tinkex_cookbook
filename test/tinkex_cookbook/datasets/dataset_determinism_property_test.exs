defmodule TinkexCookbook.Datasets.DatasetDeterminismPropertyTest do
  use ExUnit.Case, async: true

  alias CrucibleTrain.Supervised.DatasetFromSamples

  test "pcg64 shuffle is deterministic for the same seed (property-style)" do
    generator =
      StreamData.tuple({
        StreamData.list_of(StreamData.positive_integer(), min_length: 2, max_length: 20),
        StreamData.positive_integer()
      })

    generator
    |> Enum.take(50)
    |> Enum.each(fn {samples, seed} ->
      samples = Enum.uniq(samples)

      dataset =
        DatasetFromSamples.new(
          Enum.map(samples, &%{id: &1}),
          2,
          fn sample -> sample end,
          shuffle: :pcg64
        )

      shuffled_a = DatasetFromSamples.set_epoch(dataset, seed).shuffled_samples
      shuffled_b = DatasetFromSamples.set_epoch(dataset, seed).shuffled_samples

      assert shuffled_a == shuffled_b
    end)
  end
end
