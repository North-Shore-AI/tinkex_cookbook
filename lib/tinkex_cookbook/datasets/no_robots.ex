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

      # Build datums for training
      datums = NoRobots.build_datums(samples, Llama3, state, TrainOnWhat.all_assistant_messages())

      # Create a supervised dataset
      dataset = NoRobots.create_supervised_dataset(samples,
        renderer_module: Llama3,
        renderer_state: state,
        train_on_what: TrainOnWhat.all_assistant_messages(),
        batch_size: 32
      )
  """

  alias TinkexCookbook.Renderers.{Renderer, TrainOnWhat, Types}
  alias TinkexCookbook.Supervised.{Common, SupervisedDatasetFromSamples}
  alias TinkexCookbook.Types.Datum
  alias TinkexCookbook.Utils.Parity

  @dataset_name "HuggingFaceH4/no_robots"

  # Agent for tracking rendered sample count in parity mode
  @rendered_sample_counter __MODULE__.RenderedSampleCounter

  @doc """
  Loads the NoRobots dataset from HuggingFace.

  ## Options

  - `:split` - The dataset split to load ("train" or "test"). Defaults to "train".
  - `:limit` - Maximum number of samples to load. Defaults to nil (all).

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

  Converts the raw message maps to Types.Message structs.
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
  - `train_on_what` - Training weight strategy (see TrainOnWhat)
  - `max_length` - Optional maximum sequence length (default: nil)

  ## Returns

  A Datum struct ready for training with properly shifted inputs/targets.
  """
  @spec build_datum(map(), module(), map(), TrainOnWhat.t(), pos_integer() | nil) :: Datum.t()
  def build_datum(sample, renderer_module, renderer_state, train_on_what, max_length \\ nil) do
    messages = sample_to_messages(sample)

    {model_input, weights} =
      Renderer.build_supervised_example(
        renderer_module,
        messages,
        train_on_what,
        renderer_state
      )

    # Log rendered sample for parity comparison (if PARITY_MODE=1, first 10 samples)
    sample_index = get_and_increment_rendered_sample_count()
    maybe_log_rendered_sample(sample_index, sample, model_input, weights)

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
  @spec build_datums([map()], module(), map(), TrainOnWhat.t(), pos_integer() | nil) ::
          [Datum.t()]
  def build_datums(samples, renderer_module, renderer_state, train_on_what, max_length \\ nil) do
    Enum.map(samples, fn sample ->
      build_datum(sample, renderer_module, renderer_state, train_on_what, max_length)
    end)
  end

  @doc """
  Creates a SupervisedDataset from a list of samples.

  Uses lazy datum building to match Python's `SupervisedDatasetFromHFDataset` behavior:
  - Stores samples (not datums)
  - Shuffles samples on `set_epoch/2` using HfDatasetsEx (PCG64-based)
  - Builds datums lazily during `get_batch/2`

  This ensures parity with Python's training loop.

  ## Options

  - `:renderer_module` - Required. The renderer module to use.
  - `:renderer_state` - Required. The renderer state from init/1.
  - `:train_on_what` - Training weight strategy. Defaults to all_assistant_messages.
  - `:batch_size` - Batch size. Defaults to 32.
  - `:max_length` - Maximum sequence length. Defaults to nil (no limit).

  ## Returns

  A SupervisedDatasetFromSamples struct.
  """
  @spec create_supervised_dataset([map()], keyword()) :: SupervisedDatasetFromSamples.t()
  def create_supervised_dataset(samples, opts) do
    renderer_module = Keyword.fetch!(opts, :renderer_module)
    renderer_state = Keyword.fetch!(opts, :renderer_state)
    train_on_what = Keyword.get(opts, :train_on_what, TrainOnWhat.all_assistant_messages())
    batch_size = Keyword.get(opts, :batch_size, 32)
    max_length = Keyword.get(opts, :max_length)

    # Log dataset snapshot for parity comparison (if PARITY_MODE=1)
    Parity.log_dataset_snapshot(samples, 10)

    # Start rendered sample counter for parity mode (will be used during lazy datum building)
    start_rendered_sample_counter()

    # Create datum builder function for lazy evaluation
    # This matches Python's map_fn that's called during get_batch
    datum_builder = fn sample ->
      build_datum(sample, renderer_module, renderer_state, train_on_what, max_length)
    end

    SupervisedDatasetFromSamples.new(samples, batch_size, datum_builder)
  end

  # Private helpers

  defp maybe_limit(dataset, nil), do: dataset
  defp maybe_limit(dataset, limit) when is_integer(limit), do: Enum.take(dataset, limit)

  defp maybe_shuffle_dataset(dataset, nil), do: dataset

  defp maybe_shuffle_dataset(dataset, seed) when is_integer(seed) do
    HfDatasetsEx.Dataset.shuffle(dataset, seed: seed)
  end

  # Parity mode helpers

  @doc false
  def start_rendered_sample_counter do
    if Parity.parity_mode?() do
      Agent.start_link(fn -> 0 end, name: @rendered_sample_counter)
    end
  end

  @doc false
  def stop_rendered_sample_counter do
    if Process.whereis(@rendered_sample_counter) do
      Agent.stop(@rendered_sample_counter)
    end
  end

  defp get_and_increment_rendered_sample_count do
    if Process.whereis(@rendered_sample_counter) do
      Agent.get_and_update(@rendered_sample_counter, fn count -> {count, count + 1} end)
    else
      nil
    end
  end

  defp maybe_log_rendered_sample(sample_index, sample, model_input, weights) do
    if Parity.parity_mode?() && sample_index != nil && sample_index < 10 do
      messages =
        sample
        |> sample_to_messages()
        |> Enum.map(fn msg -> %{role: msg.role, content: msg.content} end)

      Parity.log_rendered_sample(sample_index, messages, model_input, weights)
    end
  end
end
