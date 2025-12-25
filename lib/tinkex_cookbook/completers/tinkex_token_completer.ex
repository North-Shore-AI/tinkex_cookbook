# credo:disable-for-this-file Credo.Check.Refactor.CyclomaticComplexity
defmodule TinkexCookbook.Completers.TinkexTokenCompleter do
  @moduledoc """
  Token completer backed by a Tinkex sampling client.
  """

  @behaviour TinkexCookbook.Completers.TokenCompleter

  alias TinkexCookbook.Completers.TokensWithLogprobs

  defstruct [:sampling_client, :max_tokens, temperature: 1.0]

  @type t :: %__MODULE__{
          sampling_client: struct(),
          max_tokens: pos_integer(),
          temperature: float()
        }

  @spec new(keyword()) :: t()
  def new(opts) do
    struct!(__MODULE__, opts)
  end

  @impl true
  def complete(%__MODULE__{} = completer, model_input, stop) do
    sampling_params = %{
      stop: stop,
      max_tokens: completer.max_tokens,
      temperature: completer.temperature
    }

    sampling_client = completer.sampling_client

    with {:ok, task} <-
           sampling_client.__struct__.sample(
             sampling_client,
             model_input,
             sampling_params,
             num_samples: 1
           ),
         {:ok, response} <- Task.await(task, :infinity),
         {:ok, result} <- extract_tokens_with_logprobs(response) do
      {:ok, result}
    else
      {:error, reason} -> {:error, reason}
    end
  end

  defp extract_tokens_with_logprobs(response) do
    sequences = response.sequences || response[:sequences] || []

    case sequences do
      [sequence | _] ->
        tokens = sequence.tokens || sequence[:tokens]
        logprobs = sequence.logprobs || sequence[:logprobs]

        cond do
          not is_list(tokens) ->
            {:error, :invalid_tokens}

          is_list(logprobs) ->
            {:ok, %TokensWithLogprobs{tokens: tokens, maybe_logprobs: logprobs}}

          true ->
            {:error, :logprobs_missing}
        end

      [] ->
        {:error, :no_sequences}
    end
  end
end
