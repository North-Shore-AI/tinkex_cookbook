defmodule TinkexCookbook.TokenizerUtilsTest do
  use ExUnit.Case, async: true

  alias TinkexCookbook.TokenizerUtils

  test "get_tokenizer delegates to Tinkex tokenizer loader" do
    {:ok, tokenizer} =
      TokenizerUtils.get_tokenizer("test-model",
        load_fun: fn _id, _opts -> {:ok, :dummy_tokenizer} end
      )

    assert tokenizer == :dummy_tokenizer
  end
end
