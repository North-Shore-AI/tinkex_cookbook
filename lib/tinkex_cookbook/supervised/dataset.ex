defmodule TinkexCookbook.Supervised.SupervisedDataset do
  @moduledoc """
  Behaviour for supervised learning datasets.

  A supervised dataset provides batched access to training data.
  Implementations should handle shuffling and epoch management.
  """

  alias TinkexCookbook.Types.Datum

  @type t :: struct()

  @doc """
  Returns a batch of datums at the given index.
  """
  @callback get_batch(dataset :: t(), index :: non_neg_integer()) :: [Datum.t()]

  @doc """
  Returns the number of batches in the dataset.
  """
  @callback length(dataset :: t()) :: non_neg_integer()

  @doc """
  Sets the epoch, which may trigger reshuffling.
  """
  @callback set_epoch(dataset :: t(), seed :: non_neg_integer()) :: t()

  @doc """
  Dispatches get_batch to the dataset implementation.
  """
  @spec get_batch(t(), non_neg_integer()) :: [Datum.t()]
  def get_batch(%module{} = dataset, index) do
    module.get_batch(dataset, index)
  end

  @doc """
  Dispatches length to the dataset implementation.
  """
  @spec length(t()) :: non_neg_integer()
  def length(%module{} = dataset) do
    module.length(dataset)
  end

  @doc """
  Dispatches set_epoch to the dataset implementation.
  """
  @spec set_epoch(t(), non_neg_integer()) :: t()
  def set_epoch(%module{} = dataset, seed) do
    module.set_epoch(dataset, seed)
  end
end

defmodule TinkexCookbook.Supervised.SupervisedDatasetFromList do
  @moduledoc """
  A supervised dataset backed by a list of datums.

  This is a simple implementation useful for testing and small datasets.
  For larger datasets, use HuggingFace-backed implementations.
  """

  @behaviour TinkexCookbook.Supervised.SupervisedDataset

  alias TinkexCookbook.Types.Datum

  @type t :: %__MODULE__{
          data: [Datum.t()],
          batch_size: pos_integer(),
          shuffled_data: [Datum.t()] | nil
        }

  defstruct [:data, :batch_size, :shuffled_data]

  @doc """
  Creates a new dataset from a list of datums.
  """
  @spec new([Datum.t()], pos_integer()) :: t()
  def new(data, batch_size) when is_list(data) and batch_size > 0 do
    %__MODULE__{
      data: data,
      batch_size: batch_size,
      shuffled_data: nil
    }
  end

  @impl true
  @spec get_batch(t(), non_neg_integer()) :: [Datum.t()]
  def get_batch(%__MODULE__{batch_size: batch_size} = dataset, index) do
    data = dataset.shuffled_data || dataset.data
    start = index * batch_size
    Enum.slice(data, start, batch_size)
  end

  @impl true
  @spec length(t()) :: non_neg_integer()
  def length(%__MODULE__{data: data, batch_size: batch_size}) do
    div(Enum.count(data), batch_size)
  end

  @impl true
  @spec set_epoch(t(), non_neg_integer()) :: t()
  def set_epoch(%__MODULE__{data: data} = dataset, seed) do
    # Use the seed for deterministic shuffling
    shuffled =
      data
      |> Enum.with_index()
      |> Enum.sort_by(fn {_item, idx} ->
        # Simple deterministic shuffle based on seed and index
        :erlang.phash2({seed, idx})
      end)
      |> Enum.map(fn {item, _idx} -> item end)

    %{dataset | shuffled_data: shuffled}
  end
end
