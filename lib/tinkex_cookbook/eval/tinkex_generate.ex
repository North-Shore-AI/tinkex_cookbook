defmodule TinkexCookbook.Eval.TinkexGenerate do
  @moduledoc """
  Implements CrucibleHarness.Generate using Tinkex sampling.

  This adapter bridges the CrucibleHarness evaluation framework to Tinkex
  for generating responses from fine-tuned models.

  ## Configuration

  The config map should include:

    * `:sampling_client` - A Tinkex.SamplingClient or mock
    * `:model` - Model identifier (for logging/config)
    * `:temperature` - Sampling temperature
    * `:max_tokens` - Maximum tokens to generate
    * `:stop` - Stop sequences
    * `:renderer_module` - Optional. Renderer module for building prompts
    * `:renderer_state` - Optional. Renderer state

  ## Usage

      config = %{
        sampling_client: sampling_client,
        model: "meta-llama/Llama-3.1-8B",
        temperature: 0.7,
        max_tokens: 1024,
        stop: ["<|eot_id|>"],
        renderer_module: TinkexCookbook.Renderers.Llama3,
        renderer_state: renderer_state
      }

      messages = [%{role: "user", content: "Hello!"}]
      {:ok, response} = TinkexGenerate.generate(messages, config)

  ## With CrucibleHarness Solver

      solver = CrucibleHarness.Solver.Generate.new(config)
      generate_fn = &TinkexGenerate.generate/2
      {:ok, result} = CrucibleHarness.Solver.solve(solver, state, generate_fn)
  """

  @behaviour CrucibleHarness.Generate

  alias TinkexCookbook.Renderers.{Renderer, Types}
  alias TinkexCookbook.Types.ModelInput

  @impl true
  @doc """
  Generates text using a Tinkex sampling client.

  ## Parameters

    * `messages` - List of message maps with `:role` and `:content` keys
    * `config` - Configuration map with sampling parameters

  ## Returns

    * `{:ok, response}` - Response with `:content`, `:finish_reason`, and `:usage`
    * `{:error, reason}` - If generation fails
  """
  @spec generate([map()], map()) :: {:ok, map()} | {:error, term()}
  def generate(messages, config) do
    sampling_client = Map.fetch!(config, :sampling_client)

    # Build sampling params
    sampling_params = build_sampling_params(config)

    # If renderer is provided, build proper model input
    model_input =
      if config[:renderer_module] && config[:renderer_state] do
        {input, _state} =
          build_model_input(messages, config.renderer_module, config.renderer_state)

        input
      else
        # Simple fallback: just encode messages as text
        build_simple_input(messages)
      end

    # Call sampling client
    case sampling_client.__struct__.sample(sampling_client, model_input, sampling_params) do
      {:ok, %Task{} = task} ->
        with {:ok, sample_response} <- Task.await(task, :infinity) do
          response = format_response(sample_response, config)
          {:ok, response}
        end

      {:ok, sample_response} ->
        response = format_response(sample_response, config)
        {:ok, response}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Builds a ModelInput from messages using a renderer.
  """
  @spec build_model_input([map()], module(), map()) :: {ModelInput.t(), map()}
  def build_model_input(messages, renderer_module, renderer_state) do
    # Convert raw maps to Types.Message structs
    typed_messages = Enum.map(messages, &to_typed_message/1)

    # Build generation prompt
    Renderer.build_generation_prompt(
      renderer_module,
      typed_messages,
      "assistant",
      nil,
      renderer_state
    )
  end

  @doc """
  Parses a sample response back into a message.
  """
  @spec parse_response(map(), module(), map()) :: {Types.Message.t(), boolean()}
  def parse_response(sample_response, renderer_module, renderer_state) do
    # Get the first sequence
    [first_sequence | _] = sample_response.sequences

    # Parse using renderer
    renderer_module.parse_response(first_sequence.tokens, renderer_state)
  end

  # Private helpers

  defp build_sampling_params(config) do
    [
      temperature: Map.get(config, :temperature, 0.7),
      max_tokens: Map.get(config, :max_tokens, 1024),
      stop: Map.get(config, :stop, [])
    ]
  end

  defp build_simple_input(messages) do
    # Convert messages to a simple text prompt
    text =
      Enum.map_join(messages, "\n", fn msg ->
        role = get_field(msg, :role) || get_field(msg, "role")
        content = get_field(msg, :content) || get_field(msg, "content")
        "#{role}: #{content}"
      end)

    # Create a simple model input with ASCII tokens
    tokens = String.to_charlist(text)
    ModelInput.from_ints(tokens)
  end

  # Get field from either struct or map
  defp get_field(%{__struct__: _} = struct, field) when is_atom(field) do
    Map.get(struct, field)
  end

  defp get_field(map, field) when is_map(map) do
    Map.get(map, field)
  end

  defp to_typed_message(msg) when is_map(msg) do
    role = msg[:role] || msg["role"]
    content = msg[:content] || msg["content"]
    Types.message(role, content)
  end

  defp format_response(sample_response, _config) do
    # Extract from first sequence
    [first_sequence | _] = sample_response.sequences || sample_response[:sequences]

    content = first_sequence.text || first_sequence[:text] || ""
    stop_reason = first_sequence.stop_reason || first_sequence[:stop_reason] || "stop"

    # Clean up content (remove stop tokens)
    cleaned_content = clean_content(content)

    %{
      content: cleaned_content,
      finish_reason: normalize_finish_reason(stop_reason),
      usage: %{
        prompt_tokens: 0,
        completion_tokens: String.length(cleaned_content),
        total_tokens: String.length(cleaned_content)
      }
    }
  end

  defp clean_content(content) do
    # Remove common stop tokens from the end
    content
    |> String.replace(~r/<\|eot_id\|>$/, "")
    |> String.replace(~r/<\|end_of_text\|>$/, "")
    |> String.replace(~r/<\|im_end\|>$/, "")
    |> String.trim()
  end

  defp normalize_finish_reason(reason) when is_binary(reason), do: reason
  defp normalize_finish_reason(:end_of_turn), do: "stop"
  defp normalize_finish_reason(:length), do: "length"
  defp normalize_finish_reason(:stop), do: "stop"
  defp normalize_finish_reason(_), do: "stop"
end
