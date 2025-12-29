defmodule TinkexCookbook.Datasets.NoRobots do
  @moduledoc """
  NoRobots dataset builder for supervised chat fine-tuning.

  This module provides utilities for loading and processing the NoRobots dataset
  (HuggingFaceH4/no_robots) for supervised learning.

  ## Dataset Format

  Each example in the NoRobots dataset contains a `messages` field with conversation
  turns in the format:

      %{
        "messages" => [
          %{"role" => "user", "content" => "Hello!"},
          %{"role" => "assistant", "content" => "Hi there!"}
        ]
      }

  ## Usage

      # Load dataset from HuggingFace
      {:ok, samples} = NoRobots.load(split: "train", limit: 100)

      # Create a supervised dataset
      dataset = NoRobots.create_supervised_dataset(samples,
        renderer_module: CrucibleTrain.Renderers.Llama3,
        renderer_state: state,
        train_on_what: :all_assistant_messages,
        batch_size: 32
      )
  """

  alias CrucibleTrain.Renderers.{Renderer, Types}
  alias CrucibleTrain.Supervised.{Common, Dataset, DatasetFromSamples}
  alias CrucibleTrain.Types.Datum

  @dataset_name "HuggingFaceH4/no_robots"

  @doc """
  Loads the NoRobots dataset from HuggingFace.

  ## Options

  - `:split` - The dataset split to load ("train" or "test"). Defaults to "train".
  - `:limit` - Maximum number of samples to load. Defaults to nil (all).
  - `:shuffle_seed` - Seed for shuffling. Defaults to nil (no shuffle).

  ## Returns

  - `{:ok, samples}` - List of sample maps
  - `{:error, reason}` - Error loading the dataset
  """
  @spec load(keyword()) :: {:ok, [map()]} | {:error, term()}
  def load(opts \\ []) do
    split = Keyword.get(opts, :split, "train")
    limit = Keyword.get(opts, :limit)
    shuffle_seed = Keyword.get(opts, :shuffle_seed)

    case HfDatasetsEx.load_dataset(@dataset_name, split: split) do
      {:ok, dataset} ->
        # Use hf_datasets_ex's shuffle if seed provided
        dataset = maybe_shuffle_dataset(dataset, shuffle_seed)

        samples =
          dataset.items
          |> maybe_limit(limit)

        {:ok, samples}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Extracts messages from a dataset sample.

  Converts the raw message maps to Message structs.
  """
  @spec sample_to_messages(map()) :: [Types.Message.t()]
  def sample_to_messages(%{"messages" => messages}) when is_list(messages) do
    Enum.map(messages, fn msg ->
      Types.message(msg["role"], msg["content"])
    end)
  end

  def sample_to_messages(_), do: []

  @doc """
  Builds a single Datum from a dataset sample.

  Uses the renderer to convert messages into a ModelInput with
  properly aligned weights, then applies the right-shift/left-shift
  transformation required for next-token prediction training.

  ## Parameters

  - `sample` - A dataset sample with a "messages" field
  - `renderer_module` - The renderer module to use (e.g., Llama3)
  - `renderer_state` - The renderer state from init/1
  - `train_on_what` - Training weight strategy (atom: :all_assistant_messages, etc.)
  - `max_length` - Optional maximum sequence length (default: nil)

  ## Returns

  A Datum struct ready for training with properly shifted inputs/targets.
  """
  @spec build_datum(
          map(),
          module(),
          map(),
          CrucibleTrain.Renderers.TrainOnWhat.t(),
          pos_integer() | nil
        ) ::
          Datum.t()
  def build_datum(sample, renderer_module, renderer_state, train_on_what, max_length \\ nil) do
    messages = sample_to_messages(sample)

    {model_input, weights} =
      Renderer.build_supervised_example(
        renderer_module,
        messages,
        train_on_what,
        renderer_state
      )

    # Use the common function that handles right-shift/left-shift
    Common.datum_from_model_input_weights(model_input, weights, max_length)
  end

  @doc """
  Builds datums for a list of samples.

  ## Parameters

  - `samples` - List of dataset samples
  - `renderer_module` - The renderer module to use
  - `renderer_state` - The renderer state from init/1
  - `train_on_what` - Training weight strategy
  - `max_length` - Optional maximum sequence length (default: nil)

  ## Returns

  List of Datum structs.
  """
  @spec build_datums(
          [map()],
          module(),
          map(),
          CrucibleTrain.Renderers.TrainOnWhat.t(),
          pos_integer() | nil
        ) :: [Datum.t()]
  def build_datums(samples, renderer_module, renderer_state, train_on_what, max_length \\ nil) do
    Enum.map(samples, fn sample ->
      build_datum(sample, renderer_module, renderer_state, train_on_what, max_length)
    end)
  end

  @doc """
  Creates a SupervisedDataset from a list of samples.

  Uses lazy datum building:
  - Stores samples (not datums)
  - Shuffles samples on `set_epoch/2`
  - Builds datums lazily during `get_batch/2`

  ## Options

  - `:renderer_module` - Required. The renderer module to use.
  - `:renderer_state` - Required. The renderer state from init/1.
  - `:train_on_what` - Training weight strategy. Defaults to :all_assistant_messages.
  - `:batch_size` - Batch size. Defaults to 32.
  - `:max_length` - Maximum sequence length. Defaults to nil (no limit).

  ## Returns

  A Dataset struct.
  """
  @spec create_supervised_dataset([map()], keyword()) :: Dataset.t() | DatasetFromSamples.t()
  def create_supervised_dataset(samples, opts) do
    renderer_module = Keyword.fetch!(opts, :renderer_module)
    renderer_state = Keyword.fetch!(opts, :renderer_state)
    train_on_what = Keyword.get(opts, :train_on_what, :all_assistant_messages)
    batch_size = Keyword.get(opts, :batch_size, 32)
    max_length = Keyword.get(opts, :max_length)

    # Create datum builder function for lazy evaluation
    datum_builder = fn sample ->
      build_datum(sample, renderer_module, renderer_state, train_on_what, max_length)
    end

    DatasetFromSamples.new(samples, batch_size, datum_builder)
  end

  # Private helpers

  defp maybe_limit(dataset, nil), do: dataset
  defp maybe_limit(dataset, limit) when is_integer(limit), do: Enum.take(dataset, limit)

  defp maybe_shuffle_dataset(dataset, nil), do: dataset

  defp maybe_shuffle_dataset(dataset, seed) when is_integer(seed) do
    HfDatasetsEx.Dataset.shuffle(dataset, seed: seed)
  end
end
