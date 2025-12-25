defmodule TinkexCookbook.Preference.DPODatasetBuilderFromComparisons do
  @moduledoc """
  DPO dataset builder that produces chosen/rejected datum pairs.
  """

  use ChzEx.Schema

  alias TinkexCookbook.Preference.{ComparisonDatasetBuilder, LabeledComparison}
  alias TinkexCookbook.Renderers
  alias TinkexCookbook.Renderers.{Renderer, TrainOnWhat}
  alias TinkexCookbook.Supervised.ChatDatasetBuilderCommonConfig
  alias TinkexCookbook.Supervised.{Common, SupervisedDatasetFromSamplesFlatMap}
  alias TinkexCookbook.TokenizerUtils

  chz_schema do
    field(:comparison_builder, :any, virtual: true)
    field(:common_config, :any, virtual: true)
    field(:renderer_module, :any, default: nil, virtual: true)
    field(:renderer_state, :any, default: nil, virtual: true)
  end

  @spec build(struct()) :: {struct(), struct() | nil}
  def build(%__MODULE__{} = builder) do
    common_config = %ChatDatasetBuilderCommonConfig{} = builder.common_config

    {renderer_module, renderer_state} =
      case {builder.renderer_module, builder.renderer_state} do
        {module, state} when not is_nil(module) and not is_nil(state) ->
          {module, state}

        _ ->
          {:ok, tokenizer} = TokenizerUtils.get_tokenizer(common_config.model_name_for_tokenizer)
          {:ok, module, extra_opts} = Renderers.lookup(common_config.renderer_name)
          {:ok, state} = module.init([{:tokenizer, tokenizer} | extra_opts])
          {module, state}
      end

    {train_dataset, test_dataset} =
      ComparisonDatasetBuilder.get_train_and_test_datasets(builder.comparison_builder)

    comparison_to_datums = fn %LabeledComparison{} = labeled ->
      chosen_completion =
        if labeled.label == "A" do
          labeled.comparison.completion_a
        else
          labeled.comparison.completion_b
        end

      rejected_completion =
        if labeled.label == "A" do
          labeled.comparison.completion_b
        else
          labeled.comparison.completion_a
        end

      chosen_convo = labeled.comparison.prompt_conversation ++ chosen_completion
      rejected_convo = labeled.comparison.prompt_conversation ++ rejected_completion

      {chosen_input, chosen_weights} =
        Renderer.build_supervised_example(
          renderer_module,
          chosen_convo,
          TrainOnWhat.last_assistant_message(),
          renderer_state
        )

      {rejected_input, rejected_weights} =
        Renderer.build_supervised_example(
          renderer_module,
          rejected_convo,
          TrainOnWhat.last_assistant_message(),
          renderer_state
        )

      [
        Common.datum_from_model_input_weights(
          chosen_input,
          chosen_weights,
          common_config.max_length
        ),
        Common.datum_from_model_input_weights(
          rejected_input,
          rejected_weights,
          common_config.max_length
        )
      ]
    end

    example_to_data = fn example ->
      case ComparisonDatasetBuilder.example_to_labeled_comparison(
             builder.comparison_builder,
             example
           ) do
        nil -> []
        labeled -> comparison_to_datums.(labeled)
      end
    end

    train =
      SupervisedDatasetFromSamplesFlatMap.new(
        train_dataset,
        common_config.batch_size,
        example_to_data
      )

    test =
      if test_dataset != nil do
        SupervisedDatasetFromSamplesFlatMap.new(
          test_dataset,
          length(test_dataset),
          example_to_data
        )
      else
        nil
      end

    {train, test}
  end
end
