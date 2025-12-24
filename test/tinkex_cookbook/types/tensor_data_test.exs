defmodule TinkexCookbook.Types.TensorDataTest do
  use ExUnit.Case, async: true

  alias TinkexCookbook.Types.TensorData

  describe "new/3" do
    test "creates TensorData with data, dtype, and shape" do
      data = [1, 2, 3, 4]
      td = TensorData.new(data, :int64, [4])

      assert td.data == data
      assert td.dtype == :int64
      assert td.shape == [4]
    end

    test "creates TensorData with float32 dtype" do
      data = [1.0, 2.0, 3.0]
      td = TensorData.new(data, :float32, [3])

      assert td.data == data
      assert td.dtype == :float32
      assert td.shape == [3]
    end

    test "creates TensorData with 2D shape" do
      data = [1, 2, 3, 4, 5, 6]
      td = TensorData.new(data, :int64, [2, 3])

      assert td.shape == [2, 3]
    end
  end

  describe "from_list/2" do
    test "creates TensorData from integer list" do
      data = [1, 2, 3]
      td = TensorData.from_list(data, :int64)

      assert td.data == data
      assert td.dtype == :int64
      assert td.shape == [3]
    end

    test "creates TensorData from float list" do
      data = [1.0, 2.0, 3.0]
      td = TensorData.from_list(data, :float32)

      assert td.data == data
      assert td.dtype == :float32
      assert td.shape == [3]
    end
  end

  describe "to_list/1" do
    test "returns the underlying data" do
      data = [1, 2, 3, 4]
      td = TensorData.new(data, :int64, [4])

      assert TensorData.to_list(td) == data
    end
  end

  describe "size/1" do
    test "returns total element count" do
      td1 = TensorData.new([1, 2, 3, 4], :int64, [4])
      assert TensorData.size(td1) == 4

      td2 = TensorData.new([1, 2, 3, 4, 5, 6], :int64, [2, 3])
      assert TensorData.size(td2) == 6
    end
  end

  describe "sum/1" do
    test "sums numeric data" do
      td = TensorData.new([1.0, 2.0, 3.0, 4.0], :float32, [4])
      assert TensorData.sum(td) == 10.0
    end

    test "sums integer data" do
      td = TensorData.new([1, 2, 3, 4], :int64, [4])
      assert TensorData.sum(td) == 10
    end
  end

  describe "slice/3" do
    test "slices data from start to end" do
      td = TensorData.new([1, 2, 3, 4, 5], :int64, [5])
      sliced = TensorData.slice(td, 1, 4)

      assert sliced.data == [2, 3, 4]
      assert sliced.shape == [3]
    end
  end
end
