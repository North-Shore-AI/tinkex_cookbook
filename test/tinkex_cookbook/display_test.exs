defmodule TinkexCookbook.DisplayTest do
  use ExUnit.Case, async: true

  alias CrucibleTrain.Types.{Datum, ModelInput, TensorData}
  alias TinkexCookbook.Display
  alias TinkexCookbook.Test.MockTokenizer

  test "colorize_example returns formatted output" do
    model_input = ModelInput.from_ints(MockTokenizer.encode("Hi"))

    loss_fn_inputs = %{
      "weights" => TensorData.from_list([1.0, 1.0], :float32),
      "target_tokens" => TensorData.from_list(MockTokenizer.encode("i!"), :int64)
    }

    datum = Datum.new(model_input, loss_fn_inputs)

    output = Display.colorize_example(datum, MockTokenizer)

    # Output contains ANSI color codes that split characters.
    # Strip ANSI codes and check the content.
    stripped = Regex.replace(~r/\e\[[0-9;]*m/, output, "")
    assert String.contains?(stripped, "H")
    assert String.contains?(stripped, "i")
  end
end
