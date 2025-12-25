defmodule TinkexCookbook.Preference.DPODatasetsTest do
  use ExUnit.Case, async: true

  alias TinkexCookbook.Preference.{
    Comparison,
    DPODatasetBuilderFromComparisons,
    LabeledComparison
  }

  alias TinkexCookbook.Renderers.{Helpers, RoleColon, Types}
  alias TinkexCookbook.Supervised.{ChatDatasetBuilderCommonConfig, SupervisedDataset}
  alias TinkexCookbook.Test.MockTokenizer
  alias TinkexCookbook.Types.{ModelInput, TensorData}

  defmodule TestComparisonBuilder do
    defstruct [:comparisons]

    def get_train_and_test_datasets(%__MODULE__{comparisons: comparisons}), do: {comparisons, nil}

    def example_to_labeled_comparison(_builder, %LabeledComparison{} = labeled), do: labeled
  end

  test "dpo dataset builder yields chosen/rejected datum pairs" do
    {:ok, renderer_state} = RoleColon.init(tokenizer: MockTokenizer)

    common_config = %ChatDatasetBuilderCommonConfig{
      model_name_for_tokenizer: "mock",
      renderer_name: "role_colon",
      batch_size: 1
    }

    comparison = %Comparison{
      prompt_conversation: [Types.message("user", "Prompt")],
      completion_a: [Types.message("assistant", "COMPLETION_A")],
      completion_b: [Types.message("assistant", "COMPLETION_B")]
    }

    labeled = %LabeledComparison{comparison: comparison, label: "A"}

    builder = %DPODatasetBuilderFromComparisons{
      comparison_builder: %TestComparisonBuilder{comparisons: [labeled]},
      common_config: common_config,
      renderer_module: RoleColon,
      renderer_state: renderer_state
    }

    {dataset, _} = DPODatasetBuilderFromComparisons.build(builder)
    [chosen, rejected] = SupervisedDataset.get_batch(dataset, 0)

    assert decode_full_sequence(chosen) =~ "COMPLETION_A"
    assert decode_full_sequence(rejected) =~ "COMPLETION_B"
  end

  defp decode_full_sequence(datum) do
    last_target =
      datum.loss_fn_inputs["target_tokens"]
      |> TensorData.to_list()
      |> List.last()

    full_input = ModelInput.append_int(datum.model_input, last_target)
    Helpers.decode(MockTokenizer, ModelInput.all_tokens(full_input))
  end
end
