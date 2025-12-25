defmodule TinkexCookbook.Completers.TokenCompleterTest do
  use ExUnit.Case, async: true

  alias TinkexCookbook.Completers.{TinkexTokenCompleter, TokensWithLogprobs}
  alias TinkexCookbook.Types.ModelInput

  defmodule SamplingClientStub do
    defstruct [:response]

    def new(response), do: %__MODULE__{response: response}

    def sample(%__MODULE__{response: response}, _prompt, _params, _opts \\ []) do
      {:ok, Task.async(fn -> {:ok, response} end)}
    end
  end

  test "complete returns tokens with logprobs" do
    response = %{sequences: [%{tokens: [1, 2, 3], logprobs: [-0.1, -0.2, -0.3]}]}
    client = SamplingClientStub.new(response)
    completer = TinkexTokenCompleter.new(sampling_client: client, max_tokens: 4, temperature: 1.0)

    model_input = ModelInput.from_ints([1, 2])

    assert {:ok, %TokensWithLogprobs{} = result} =
             TinkexTokenCompleter.complete(completer, model_input, ["<|stop|>"])

    assert result.tokens == [1, 2, 3]
    assert result.maybe_logprobs == [-0.1, -0.2, -0.3]
  end

  test "complete returns error when logprobs are missing" do
    response = %{sequences: [%{tokens: [1, 2, 3], logprobs: nil}]}
    client = SamplingClientStub.new(response)
    completer = TinkexTokenCompleter.new(sampling_client: client, max_tokens: 4, temperature: 1.0)

    model_input = ModelInput.from_ints([1, 2])

    assert {:error, :logprobs_missing} =
             TinkexTokenCompleter.complete(completer, model_input, ["<|stop|>"])
  end

  test "TokensWithLogprobs.logprobs!/1 raises when missing" do
    result = %TokensWithLogprobs{tokens: [1], maybe_logprobs: nil}

    assert_raise ArgumentError, ~r/Logprobs are not available/, fn ->
      TokensWithLogprobs.logprobs!(result)
    end
  end
end
