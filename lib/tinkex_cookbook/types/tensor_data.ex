defmodule TinkexCookbook.Types.TensorData do
  @moduledoc """
  Wrapper for tensor data with shape and dtype information.

  Mirrors the Python `tinker.TensorData` type. This is used to represent
  tensor data in a format compatible with the Tinker API.
  """

  @type dtype :: :int64 | :float32 | :float64 | :int32

  @type t :: %__MODULE__{
          data: list(number()),
          dtype: dtype(),
          shape: [non_neg_integer()]
        }

  @enforce_keys [:data, :dtype, :shape]
  defstruct [:data, :dtype, :shape]

  @doc """
  Creates a new TensorData with explicit data, dtype, and shape.
  """
  @spec new(list(number()), dtype(), [non_neg_integer()]) :: t()
  def new(data, dtype, shape) when is_list(data) and is_list(shape) do
    %__MODULE__{
      data: data,
      dtype: dtype,
      shape: shape
    }
  end

  @doc """
  Creates a TensorData from a flat list, inferring 1D shape.
  """
  @spec from_list(list(number()), dtype()) :: t()
  def from_list(data, dtype) when is_list(data) do
    new(data, dtype, [length(data)])
  end

  @doc """
  Returns the underlying data as a list.
  """
  @spec to_list(t()) :: list(number())
  def to_list(%__MODULE__{data: data}), do: data

  @doc """
  Returns the total number of elements.
  """
  @spec size(t()) :: non_neg_integer()
  def size(%__MODULE__{data: data}), do: Kernel.length(data)

  @doc """
  Returns the sum of all elements.
  """
  @spec sum(t()) :: number()
  def sum(%__MODULE__{data: data}), do: Enum.sum(data)

  @doc """
  Slices the data from start (inclusive) to stop (exclusive).
  """
  @spec slice(t(), non_neg_integer(), non_neg_integer()) :: t()
  def slice(%__MODULE__{dtype: dtype, data: data}, start, stop) do
    sliced_data = Enum.slice(data, start, stop - start)
    new(sliced_data, dtype, [Kernel.length(sliced_data)])
  end
end
