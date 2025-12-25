defmodule TinkexCookbook.Preference.PreferenceDatasetsTest do
  use ExUnit.Case, async: true

  alias TinkexCookbook.Preference.{
    Comparison,
    ComparisonBuilderFromJsonl,
    ComparisonDatasetBuilder,
    LabeledComparison
  }

  defp write_jsonl!(path, lines) do
    File.write!(path, Enum.join(lines, "\n") <> "\n")
  end

  test "comparison builder from jsonl loads labeled comparisons" do
    tmp_dir = System.tmp_dir!()
    path = Path.join(tmp_dir, "comparisons_test.jsonl")

    comparison = %{
      "comparison" => %{
        "prompt_conversation" => [%{"role" => "user", "content" => "Hi"}],
        "completion_A" => [%{"role" => "assistant", "content" => "A"}],
        "completion_B" => [%{"role" => "assistant", "content" => "B"}]
      },
      "label" => "A"
    }

    write_jsonl!(path, [Jason.encode!(comparison)])

    builder = %ComparisonBuilderFromJsonl{train_path: path, test_path: nil}

    {train, test} = ComparisonDatasetBuilder.get_train_and_test_datasets(builder)
    assert length(train) == 1
    assert test == nil

    [example] = train
    labeled = ComparisonDatasetBuilder.example_to_labeled_comparison(builder, example)

    assert %LabeledComparison{label: "A", comparison: %Comparison{}} = labeled
    assert labeled.comparison.completion_a == [%{"role" => "assistant", "content" => "A"}]
    assert labeled.comparison.completion_b == [%{"role" => "assistant", "content" => "B"}]

    {train_comparisons, _test_comparisons} =
      ComparisonDatasetBuilder.get_labeled_comparisons(builder)

    assert length(train_comparisons) == 1
  end
end
