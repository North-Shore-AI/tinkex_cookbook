defmodule TinkexCookbook.Preference.ComparisonDatasetBuilder do
  @moduledoc """
  Behaviour for building datasets of labeled comparisons.
  """

  alias TinkexCookbook.Preference.LabeledComparison

  @callback get_train_and_test_datasets(struct()) :: {list(map()), list(map()) | nil}
  @callback example_to_labeled_comparison(struct(), map()) :: LabeledComparison.t() | nil

  @spec get_train_and_test_datasets(struct()) :: {list(map()), list(map()) | nil}
  def get_train_and_test_datasets(%module{} = builder) do
    {train, test} = module.get_train_and_test_datasets(builder)
    {dataset_to_list(train), dataset_to_list(test)}
  end

  @spec example_to_labeled_comparison(struct(), map()) :: LabeledComparison.t() | nil
  def example_to_labeled_comparison(%module{} = builder, example) do
    module.example_to_labeled_comparison(builder, example)
  end

  @spec get_labeled_comparisons(struct()) ::
          {[LabeledComparison.t()], [LabeledComparison.t()] | nil}
  def get_labeled_comparisons(builder) do
    {train_dataset, test_dataset} = get_train_and_test_datasets(builder)

    train = process_labeled_comparisons(builder, train_dataset)
    test = if test_dataset, do: process_labeled_comparisons(builder, test_dataset), else: nil

    {train, test}
  end

  defp process_labeled_comparisons(builder, dataset) do
    Enum.reduce(dataset, [], fn example, acc ->
      case example_to_labeled_comparison(builder, example) do
        nil -> acc
        labeled -> [labeled | acc]
      end
    end)
    |> Enum.reverse()
  end

  defp dataset_to_list(nil), do: nil
  defp dataset_to_list(list) when is_list(list), do: list

  defp dataset_to_list(%{items: items}) when is_list(items), do: items

  defp dataset_to_list(dataset) do
    if function_exported?(HfDatasetsEx.Dataset, :to_list, 1) do
      HfDatasetsEx.Dataset.to_list(dataset)
    else
      dataset
    end
  end
end

defmodule TinkexCookbook.Preference.ChatDatasetBuilderFromComparisons do
  @moduledoc """
  Chat dataset builder that derives datums from labeled comparisons.
  """

  use ChzEx.Schema

  alias TinkexCookbook.Preference.{
    ComparisonDatasetBuilder,
    ComparisonRendererFromChatRenderer,
    LabeledComparison
  }

  alias TinkexCookbook.Renderers
  alias TinkexCookbook.Supervised.ChatDatasetBuilderCommonConfig
  alias TinkexCookbook.Supervised.{Common, SupervisedDatasetFromSamplesFlatMap}
  alias TinkexCookbook.TokenizerUtils

  chz_schema do
    field(:comparison_builder, :any, virtual: true)
    field(:common_config, :any, virtual: true)
    field(:swap, :boolean, default: false)
  end

  @spec build(struct()) :: {struct(), struct() | nil}
  def build(%__MODULE__{} = builder) do
    common_config = %ChatDatasetBuilderCommonConfig{} = builder.common_config

    {:ok, tokenizer} = TokenizerUtils.get_tokenizer(common_config.model_name_for_tokenizer)
    {:ok, renderer_module, extra_opts} = Renderers.lookup(common_config.renderer_name)
    {:ok, renderer_state} = renderer_module.init([{:tokenizer, tokenizer} | extra_opts])

    comparison_renderer = %ComparisonRendererFromChatRenderer{
      renderer_module: renderer_module,
      renderer_state: renderer_state
    }

    {train_dataset, test_dataset} =
      ComparisonDatasetBuilder.get_train_and_test_datasets(builder.comparison_builder)

    comparison_to_datum = fn %LabeledComparison{} = labeled ->
      {model_input, weights} =
        ComparisonRendererFromChatRenderer.to_model_input_weights(comparison_renderer, labeled)

      Common.datum_from_model_input_weights(model_input, weights, common_config.max_length)
    end

    example_to_data = fn example ->
      labeled =
        ComparisonDatasetBuilder.example_to_labeled_comparison(
          builder.comparison_builder,
          example
        )

      cond do
        labeled == nil ->
          []

        builder.swap ->
          [comparison_to_datum.(labeled), comparison_to_datum.(LabeledComparison.swap(labeled))]

        random_swap?(example) ->
          [comparison_to_datum.(LabeledComparison.swap(labeled))]

        true ->
          [comparison_to_datum.(labeled)]
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

  defp random_swap?(example) do
    rem(:erlang.phash2(example), 2) == 0
  end
end

defmodule TinkexCookbook.Preference.ComparisonBuilderFromJsonl do
  @moduledoc """
  Load labeled comparisons from JSONL files.
  """

  use ChzEx.Schema

  alias TinkexCookbook.Preference.{Comparison, LabeledComparison}

  chz_schema do
    field(:train_path, :string)
    field(:test_path, :string, default: nil)
  end

  @spec get_train_and_test_datasets(struct()) :: {list(map()), list(map()) | nil}
  def get_train_and_test_datasets(%__MODULE__{} = builder) do
    train = load_jsonl(builder.train_path)
    test = if builder.test_path, do: load_jsonl(builder.test_path), else: nil
    {train, test}
  end

  @spec example_to_labeled_comparison(struct(), map()) :: LabeledComparison.t() | nil
  def example_to_labeled_comparison(%__MODULE__{}, example) when is_map(example) do
    case example do
      %{"comparison" => comparison_map, "label" => label} ->
        %LabeledComparison{
          comparison: %Comparison{
            prompt_conversation: Map.get(comparison_map, "prompt_conversation", []),
            completion_a: Map.get(comparison_map, "completion_A", []),
            completion_b: Map.get(comparison_map, "completion_B", [])
          },
          label: label
        }

      _ ->
        nil
    end
  end

  defp load_jsonl(path) do
    path
    |> Path.expand()
    |> File.stream!()
    |> Enum.reduce([], fn line, acc ->
      case String.trim(line) do
        "" -> acc
        trimmed -> [Jason.decode!(trimmed) | acc]
      end
    end)
    |> Enum.reverse()
  end
end
