defmodule TinkexCookbook.Completers.TinkexMessageCompleter do
  @moduledoc """
  Message completer backed by a renderer and a Tinkex sampling client.
  """

  @behaviour TinkexCookbook.Completers.MessageCompleter

  alias TinkexCookbook.Renderers.{Renderer, Types}

  defstruct [:sampling_client, :renderer_module, :renderer_state, :max_tokens, :stop_condition]

  @type t :: %__MODULE__{
          sampling_client: struct(),
          renderer_module: module(),
          renderer_state: map(),
          max_tokens: pos_integer(),
          stop_condition: [String.t()] | [integer()] | nil
        }

  @spec new(keyword()) :: t()
  def new(opts) do
    struct!(__MODULE__, opts)
  end

  @impl true
  def complete(%__MODULE__{} = completer, messages) do
    stop_condition =
      completer.stop_condition ||
        completer.renderer_module.stop_sequences(completer.renderer_state)

    {model_input, renderer_state} =
      Renderer.build_generation_prompt(
        completer.renderer_module,
        messages,
        "assistant",
        nil,
        completer.renderer_state
      )

    sampling_params = %{
      stop: stop_condition,
      max_tokens: completer.max_tokens,
      temperature: 1.0
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
         {:ok, message} <- parse_response(response, completer.renderer_module, renderer_state) do
      {:ok, message}
    else
      {:error, reason} -> {:error, reason}
    end
  end

  defp parse_response(response, renderer_module, renderer_state) do
    sequences = response.sequences || response[:sequences] || []

    case sequences do
      [sequence | _] ->
        tokens = sequence.tokens || sequence[:tokens]
        {parsed, _} = renderer_module.parse_response(tokens, renderer_state)
        {:ok, Types.message("assistant", parsed.content)}

      [] ->
        {:error, :no_sequences}
    end
  end
end
