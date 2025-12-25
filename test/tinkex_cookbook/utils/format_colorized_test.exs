defmodule TinkexCookbook.Utils.FormatColorizedTest do
  use ExUnit.Case, async: true

  alias TinkexCookbook.Test.MockTokenizer
  alias TinkexCookbook.Utils.FormatColorized

  test "format_colorized includes decoded text" do
    tokens = MockTokenizer.encode("Hi")
    weights = [1.0, 1.0]

    output = FormatColorized.format_colorized(tokens, weights, MockTokenizer)

    assert String.contains?(output, "Hi")
  end

  test "format_colorized raises on length mismatch" do
    tokens = MockTokenizer.encode("Hi")
    weights = [1.0]

    assert_raise ArgumentError, ~r/`tokens` and `weights` must be the same length/, fn ->
      FormatColorized.format_colorized(tokens, weights, MockTokenizer)
    end
  end
end
