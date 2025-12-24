# Implementation Example: Tinkex.TensorData Module
#
# This is the ONLY missing piece needed to replace torch in tinkex_cookbook.
# Place this in tinkex library at: lib/tinkex/tensor_data.ex

defmodule Tinkex.TensorData do
  @moduledoc """
  Bridge between Nx tensors and Tinker API JSON payloads.

  The Tinker API expects/returns tensor data in this JSON structure:
  ```json
  {
    "data": [1.0, 2.0, 3.0, ...],  // Flattened list of numbers
    "dtype": "float32",             // Type string
    "shape": [2, 3]                 // Dimensions
  }
  ```

  This module converts between Nx.Tensor and this JSON-friendly map structure.

  ## Examples

      # Creating TensorData from Nx tensor
      iex> tensor = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      iex> data = Tinkex.TensorData.from_nx(tensor)
      %Tinkex.TensorData{
        data: [1.0, 2.0, 3.0, 4.0],
        dtype: "float32",
        shape: [2, 2]
      }

      # Converting back to Nx tensor
      iex> Tinkex.TensorData.to_nx(data)
      #Nx.Tensor<
        f32[2][2]
        [
          [1.0, 2.0],
          [3.0, 4.0]
        ]
      >

      # Use in API payloads
      iex> datum = %Tinkex.Datum{
      ...>   model_input: input,
      ...>   loss_fn_inputs: %{
      ...>     "weights" => Tinkex.TensorData.from_nx(weights),
      ...>     "advantages" => Tinkex.TensorData.from_nx(advantages)
      ...>   }
      ...> }
  """

  @enforce_keys [:data, :dtype, :shape]
  defstruct [:data, :dtype, :shape]

  @type t :: %__MODULE__{
          data: list(number()),
          dtype: String.t(),
          shape: list(non_neg_integer())
        }

  # Nx dtype -> Tinker API dtype string
  @dtype_map %{
    {:f, 16} => "float16",
    {:f, 32} => "float32",
    {:f, 64} => "float64",
    {:bf, 16} => "bfloat16",
    {:s, 8} => "int8",
    {:s, 16} => "int16",
    {:s, 32} => "int32",
    {:s, 64} => "int64",
    {:u, 8} => "uint8",
    {:u, 16} => "uint16",
    {:u, 32} => "uint32",
    {:u, 64} => "uint64"
  }

  @inverse_dtype_map Map.new(@dtype_map, fn {k, v} -> {v, k} end)

  @doc """
  Convert an Nx tensor to TensorData for API transmission.

  ## Examples

      iex> tensor = Nx.tensor([1, 2, 3], type: :s64)
      iex> Tinkex.TensorData.from_nx(tensor)
      %Tinkex.TensorData{
        data: [1, 2, 3],
        dtype: "int64",
        shape: [3]
      }

      iex> tensor = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      iex> data = Tinkex.TensorData.from_nx(tensor)
      iex> data.shape
      [2, 2]
  """
  @spec from_nx(Nx.Tensor.t()) :: t()
  def from_nx(%Nx.Tensor{} = tensor) do
    %__MODULE__{
      data: tensor |> Nx.flatten() |> Nx.to_flat_list(),
      dtype: dtype_to_string(Nx.type(tensor)),
      shape: tensor |> Nx.shape() |> Tuple.to_list()
    }
  end

  @doc """
  Convert TensorData from API response to Nx tensor.

  ## Examples

      iex> data = %Tinkex.TensorData{
      ...>   data: [1.0, 2.0, 3.0, 4.0],
      ...>   dtype: "float32",
      ...>   shape: [2, 2]
      ...> }
      iex> Tinkex.TensorData.to_nx(data)
      #Nx.Tensor<
        f32[2][2]
        [
          [1.0, 2.0],
          [3.0, 4.0]
        ]
      >

      iex> data = %Tinkex.TensorData{
      ...>   data: [1, 2, 3],
      ...>   dtype: "int64",
      ...>   shape: [3]
      ...> }
      iex> tensor = Tinkex.TensorData.to_nx(data)
      iex> Nx.type(tensor)
      {:s, 64}
  """
  @spec to_nx(t()) :: Nx.Tensor.t()
  def to_nx(%__MODULE__{data: data, dtype: dtype, shape: shape}) do
    data
    |> Nx.tensor(type: string_to_dtype(dtype))
    |> Nx.reshape(List.to_tuple(shape))
  end

  @doc """
  Encode TensorData to a map for JSON serialization.

  Used internally by Tinkex.Client when sending requests.

  ## Examples

      iex> tensor = Nx.tensor([1.0, 2.0, 3.0])
      iex> data = Tinkex.TensorData.from_nx(tensor)
      iex> Tinkex.TensorData.to_map(data)
      %{
        "data" => [1.0, 2.0, 3.0],
        "dtype" => "float32",
        "shape" => [3]
      }
  """
  @spec to_map(t()) :: map()
  def to_map(%__MODULE__{data: data, dtype: dtype, shape: shape}) do
    %{
      "data" => data,
      "dtype" => dtype,
      "shape" => shape
    }
  end

  @doc """
  Decode a map (from JSON response) to TensorData.

  Used internally by Tinkex.Client when parsing responses.

  ## Examples

      iex> map = %{
      ...>   "data" => [1.0, 2.0, 3.0],
      ...>   "dtype" => "float32",
      ...>   "shape" => [3]
      ...> }
      iex> Tinkex.TensorData.from_map(map)
      %Tinkex.TensorData{
        data: [1.0, 2.0, 3.0],
        dtype: "float32",
        shape: [3]
      }
  """
  @spec from_map(map()) :: t()
  def from_map(%{"data" => data, "dtype" => dtype, "shape" => shape}) do
    %__MODULE__{
      data: data,
      dtype: dtype,
      shape: shape
    }
  end

  # Private helpers

  defp dtype_to_string(dtype) when is_map_key(@dtype_map, dtype) do
    @dtype_map[dtype]
  end

  defp dtype_to_string(dtype) do
    raise ArgumentError, """
    Unsupported Nx dtype: #{inspect(dtype)}

    Supported types:
    - Floats: :f16, :f32, :f64, :bf16
    - Signed ints: :s8, :s16, :s32, :s64
    - Unsigned ints: :u8, :u16, :u32, :u64
    """
  end

  defp string_to_dtype(dtype_str) when is_map_key(@inverse_dtype_map, dtype_str) do
    @inverse_dtype_map[dtype_str]
  end

  defp string_to_dtype(dtype_str) do
    raise ArgumentError, """
    Unknown dtype string from API: #{inspect(dtype_str)}

    Expected one of: #{Enum.join(Map.keys(@inverse_dtype_map), ", ")}
    """
  end
end

# Usage Example in Datum Construction
# ------------------------------------
# This shows how TensorData replaces torch.TensorData in training code.

defmodule Tinkex.Datum do
  @moduledoc """
  Training datum containing model input and loss function inputs.

  Matches the Python tinker.Datum structure.
  """

  defstruct [:model_input, :loss_fn_inputs]

  @type t :: %__MODULE__{
          model_input: Tinkex.ModelInput.t(),
          loss_fn_inputs: %{String.t() => Tinkex.TensorData.t()}
        }

  @doc """
  Create a supervised learning datum from tokens and weights.

  ## Example

      iex> tokens = Nx.tensor([1, 2, 3, 4, 5], type: :s64)
      iex> weights = Nx.tensor([0.0, 1.0, 1.0, 1.0, 1.0], type: :f32)
      iex>
      iex> # Shift for next-token prediction
      iex> input_tokens = tokens[0..-2//1]  # [1, 2, 3, 4]
      iex> target_tokens = tokens[1..-1//1]  # [2, 3, 4, 5]
      iex> target_weights = weights[1..-1//1]  # [1.0, 1.0, 1.0, 1.0]
      iex>
      iex> datum = %Tinkex.Datum{
      ...>   model_input: Tinkex.ModelInput.from_ints(Nx.to_flat_list(input_tokens)),
      ...>   loss_fn_inputs: %{
      ...>     "target_tokens" => Tinkex.TensorData.from_nx(target_tokens),
      ...>     "weights" => Tinkex.TensorData.from_nx(target_weights)
      ...>   }
      ...> }
  """
end

# Comparison: Python vs Elixir
# -----------------------------

"""
Python (tinker-cookbook):
    import torch
    import tinker

    tokens = torch.tensor([1, 2, 3, 4, 5])
    weights = torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0])

    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens[:-1].tolist()),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_torch(tokens[1:]),
            "weights": tinker.TensorData.from_torch(weights[1:]),
        }
    )

Elixir (tinkex_cookbook):
    alias Tinkex.{Datum, ModelInput, TensorData}

    tokens = Nx.tensor([1, 2, 3, 4, 5], type: :s64)
    weights = Nx.tensor([0.0, 1.0, 1.0, 1.0, 1.0], type: :f32)

    datum = %Datum{
      model_input: ModelInput.from_ints(Nx.to_flat_list(tokens[0..-2//1])),
      loss_fn_inputs: %{
        "target_tokens" => TensorData.from_nx(tokens[1..-1//1]),
        "weights" => TensorData.from_nx(weights[1..-1//1])
      }
    }

Nearly identical! The main differences:
1. torch.tensor() → Nx.tensor()
2. .tolist() → Nx.to_flat_list()
3. Python slicing [:-1] → Elixir slicing [0..-2//1]
4. from_torch() → from_nx()
"""

# Test Example
# ------------

defmodule Tinkex.TensorDataTest do
  use ExUnit.Case, async: true

  alias Tinkex.TensorData

  describe "from_nx/1 and to_nx/1" do
    test "roundtrip float tensor" do
      original = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], type: :f32)

      data = TensorData.from_nx(original)
      roundtrip = TensorData.to_nx(data)

      assert Nx.shape(roundtrip) == {2, 2}
      assert Nx.type(roundtrip) == {:f, 32}
      assert Nx.to_flat_list(roundtrip) == [1.0, 2.0, 3.0, 4.0]
    end

    test "roundtrip int tensor" do
      original = Nx.tensor([1, 2, 3], type: :s64)

      data = TensorData.from_nx(original)
      roundtrip = TensorData.to_nx(data)

      assert Nx.shape(roundtrip) == {3}
      assert Nx.type(roundtrip) == {:s, 64}
      assert Nx.to_flat_list(roundtrip) == [1, 2, 3]
    end

    test "preserves shape for multidimensional tensors" do
      original = Nx.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], type: :s32)

      data = TensorData.from_nx(original)

      assert data.shape == [2, 2, 2]
      assert data.dtype == "int32"

      roundtrip = TensorData.to_nx(data)
      assert Nx.shape(roundtrip) == {2, 2, 2}
    end
  end

  describe "to_map/1 and from_map/1" do
    test "roundtrip through map (simulates JSON)" do
      tensor = Nx.tensor([1.0, 2.0, 3.0], type: :f32)

      data = TensorData.from_nx(tensor)
      map = TensorData.to_map(data)

      assert map == %{
               "data" => [1.0, 2.0, 3.0],
               "dtype" => "float32",
               "shape" => [3]
             }

      roundtrip_data = TensorData.from_map(map)
      roundtrip_tensor = TensorData.to_nx(roundtrip_data)

      assert Nx.to_flat_list(roundtrip_tensor) == [1.0, 2.0, 3.0]
    end
  end

  describe "error handling" do
    test "raises on unsupported dtype" do
      # Nx doesn't have complex numbers, so this test is hypothetical
      # Just showing the error handling pattern
      assert_raise ArgumentError, ~r/Unsupported Nx dtype/, fn ->
        # This would fail if we somehow created an unsupported type
        TensorData.dtype_to_string({:unknown, 32})
      end
    end

    test "raises on unknown dtype string from API" do
      assert_raise ArgumentError, ~r/Unknown dtype string/, fn ->
        TensorData.from_map(%{
          "data" => [1, 2, 3],
          "dtype" => "complex128",
          "shape" => [3]
        })
      end
    end
  end
end
