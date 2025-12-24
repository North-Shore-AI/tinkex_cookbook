defmodule TinkexCookbookTest do
  use ExUnit.Case
  doctest TinkexCookbook

  test "greets the world" do
    assert TinkexCookbook.hello() == :world
  end
end
