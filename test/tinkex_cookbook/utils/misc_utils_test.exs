defmodule TinkexCookbook.Utils.MiscUtilsTest do
  use ExUnit.Case, async: true

  alias TinkexCookbook.Utils.MiscUtils

  test "safezip zips equal length lists" do
    assert MiscUtils.safezip([1, 2], [:a, :b]) == [{1, :a}, {2, :b}]
  end

  test "safezip raises on length mismatch" do
    assert_raise ArgumentError, fn ->
      MiscUtils.safezip([1], [1, 2])
    end
  end

  test "split_list distributes evenly" do
    assert MiscUtils.split_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4, 5]]
  end

  test "all_same works for matching values" do
    assert MiscUtils.all_same([:a, :a, :a]) == true
    assert MiscUtils.all_same([:a, :b]) == false
  end

  test "all_same raises for empty list" do
    assert_raise ArgumentError, fn ->
      MiscUtils.all_same([])
    end
  end

  test "dict_mean computes per-key means" do
    assert MiscUtils.dict_mean([%{"a" => 1, "b" => 3}, %{"a" => 3}]) == %{"a" => 2.0, "b" => 3.0}
  end

  test "concat_lists flattens nested lists" do
    assert MiscUtils.concat_lists([[1], [2, 3], []]) == [1, 2, 3]
  end

  test "not_none passes through values and raises on nil" do
    assert MiscUtils.not_none(1) == 1

    assert_raise ArgumentError, fn ->
      MiscUtils.not_none(nil)
    end
  end

  test "lookup_func resolves module and function" do
    fun = MiscUtils.lookup_func("String:upcase")
    assert fun.(["hi"]) == "HI"

    fun2 = MiscUtils.lookup_func("downcase", "String")
    assert fun2.(["Hi"]) == "hi"
  end

  test "timed returns updated metrics" do
    metrics = %{}
    {_result, updated} = MiscUtils.timed("work", metrics, fn -> :ok end)
    assert Map.has_key?(updated, "time/work")
  end
end
