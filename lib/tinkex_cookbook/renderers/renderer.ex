defmodule TinkexCookbook.Renderers.Renderer do
  @moduledoc """
  Behaviour for rendering conversation messages into token sequences.

  A renderer converts a list of chat messages into the format expected by
  a specific model family (Llama3, Qwen3, etc.). It handles:

  - Message formatting with role tokens and delimiters
  - Weight assignment for supervised training (TrainOnWhat)
  - Generation prompt construction for sampling
  - Response parsing

  ## Implementing a Renderer

      defmodule MyRenderer do
        @behaviour TinkexCookbook.Renderers.Renderer

        @impl true
        def init(opts) do
          {:ok, %{tokenizer: Keyword.fetch!(opts, :tokenizer)}}
        end

        @impl true
        def render_message(idx, message, is_last, state) do
          # Format message and return RenderedMessage
          {rendered_message, state}
        end

        @impl true
        def bos_tokens(state), do: []

        @impl true
        def stop_sequences(state), do: ["<|stop|>"]
      end
  """

  alias TinkexCookbook.Renderers.{TrainOnWhat, Types}
  alias TinkexCookbook.Types.{EncodedTextChunk, ModelInput}

  @type state :: map()
  @type chunk :: EncodedTextChunk.t()

  @doc """
  Initialize the renderer with options (typically including a tokenizer).
  """
  @callback init(opts :: keyword()) :: {:ok, state()} | {:error, term()}

  @doc """
  Render a single message into token chunks.

  Returns a RenderedMessage with prefix, content, and optional suffix chunks,
  along with the updated state.
  """
  @callback render_message(
              idx :: non_neg_integer(),
              message :: Types.Message.t(),
              is_last :: boolean(),
              state :: state()
            ) :: {Types.RenderedMessage.t(), state()}

  @doc """
  Returns the beginning-of-sequence tokens for this renderer.
  """
  @callback bos_tokens(state()) :: [non_neg_integer()]

  @doc """
  Returns the stop sequences for sampling.
  """
  @callback stop_sequences(state()) :: [String.t()] | [non_neg_integer()]

  @doc """
  Optional callback to parse a response into a Message.
  """
  @callback parse_response(response :: [non_neg_integer()], state()) ::
              {Types.Message.t(), boolean()}

  @optional_callbacks [parse_response: 2]

  # ===========================================================================
  # Public API
  # ===========================================================================

  @doc """
  Build a supervised training example from a list of messages.

  Returns a ModelInput and a list of weights aligned with each token.
  The weights are determined by the `train_on_what` parameter.

  ## Parameters

  - `renderer_module` - The renderer module implementing the Renderer behaviour
  - `messages` - List of conversation messages
  - `train_on_what` - Strategy for assigning loss weights (see TrainOnWhat)
  - `state` - The renderer state from init/1

  ## Returns

  A tuple of `{model_input, weights}` where:
  - `model_input` is a ModelInput struct with all token chunks
  - `weights` is a list of floats (0.0 or 1.0) aligned with each token
  """
  @spec build_supervised_example(
          module(),
          [Types.Message.t()],
          TrainOnWhat.t(),
          state()
        ) :: {ModelInput.t(), [float()]}
  def build_supervised_example(renderer_module, messages, train_on_what, state) do
    if function_exported?(renderer_module, :build_supervised_example, 3) do
      renderer_module.build_supervised_example(messages, train_on_what, state)
    else
      build_supervised_example_default(renderer_module, messages, train_on_what, state)
    end
  end

  @doc """
  Default implementation for supervised example building.
  """
  @spec build_supervised_example_default(
          module(),
          [Types.Message.t()],
          TrainOnWhat.t(),
          state()
        ) :: {ModelInput.t(), [float()]}
  def build_supervised_example_default(renderer_module, messages, train_on_what, state) do
    # Validate train_on_what usage
    validate_train_on_what!(messages, train_on_what)

    message_count = length(messages)

    # Start with BOS tokens (weight is always 0.0, matching Python)
    bos = renderer_module.bos_tokens(state)
    bos_weight = 0.0

    initial_chunks_weights =
      if bos != [], do: [{%EncodedTextChunk{tokens: bos}, bos_weight}], else: []

    # Render each message
    {chunks_weights, _final_state} =
      messages
      |> Enum.with_index()
      |> Enum.reduce({initial_chunks_weights, state}, fn {message, idx}, {acc, st} ->
        is_last = idx == message_count - 1
        {rendered, new_state} = renderer_module.render_message(idx, message, is_last, st)

        # Calculate weights for this message
        ob_weight = if train_on_what == TrainOnWhat.all_tokens(), do: 1.0, else: 0.0
        action_weight = compute_action_weight(message, idx, message_count, train_on_what)

        acc =
          acc
          |> append_weighted_chunk(rendered.prefix, ob_weight)
          |> append_weighted_content(rendered.content, action_weight)
          |> append_weighted_suffix(rendered.suffix, action_weight, is_last)

        {acc, new_state}
      end)

    # Convert chunks_weights to ModelInput and weights list
    chunks = Enum.map(chunks_weights, fn {chunk, _w} -> chunk end)

    weights =
      Enum.flat_map(chunks_weights, fn {chunk, w} ->
        List.duplicate(w, chunk_length(chunk))
      end)

    {ModelInput.new(chunks), weights}
  end

  @doc """
  Build a generation prompt for sampling from the model.

  Returns a ModelInput containing all the tokens needed to start generation.
  """
  @spec build_generation_prompt(
          module(),
          [Types.Message.t()],
          String.t(),
          String.t() | nil,
          state()
        ) :: {ModelInput.t(), state()}
  def build_generation_prompt(
        renderer_module,
        messages,
        role \\ "assistant",
        prefill \\ nil,
        state
      ) do
    if function_exported?(renderer_module, :build_generation_prompt, 4) do
      renderer_module.build_generation_prompt(messages, role, prefill, state)
    else
      build_generation_prompt_default(renderer_module, messages, role, prefill, state)
    end
  end

  @doc """
  Default implementation for generation prompt building.
  """
  @spec build_generation_prompt_default(
          module(),
          [Types.Message.t()],
          String.t(),
          String.t() | nil,
          state()
        ) :: {ModelInput.t(), state()}
  def build_generation_prompt_default(
        renderer_module,
        messages,
        role,
        prefill,
        state
      ) do
    # Start with BOS tokens
    bos = renderer_module.bos_tokens(state)
    initial_chunks = if bos != [], do: [%EncodedTextChunk{tokens: bos}], else: []

    # Render all existing messages
    {chunks, state} =
      messages
      |> Enum.with_index()
      |> Enum.reduce({initial_chunks, state}, fn {message, idx}, {acc, st} ->
        {rendered, new_state} = renderer_module.render_message(idx, message, false, st)

        acc_with_parts =
          acc ++
            if(rendered.prefix, do: [rendered.prefix], else: []) ++
            Enum.filter(rendered.content, & &1)

        {acc_with_parts, new_state}
      end)

    # Add partial message for the new response
    partial_message = Types.message(role, "")

    {rendered, final_state} =
      renderer_module.render_message(length(messages), partial_message, false, state)

    chunks_with_partial =
      chunks ++ if rendered.prefix, do: [rendered.prefix], else: []

    # Add prefill if provided
    final_chunks =
      if prefill do
        tokenizer =
          case Map.fetch(final_state, :tokenizer) do
            {:ok, tokenizer} ->
              tokenizer

            :error ->
              raise ArgumentError, "Renderer state missing :tokenizer for prefill encoding"
          end

        prefill_tokens = tokenizer.encode(prefill, add_special_tokens: false)
        chunks_with_partial ++ [%EncodedTextChunk{tokens: prefill_tokens}]
      else
        chunks_with_partial
      end

    {ModelInput.new(final_chunks), final_state}
  end

  # ===========================================================================
  # Private Helpers
  # ===========================================================================

  defp validate_train_on_what!(messages, train_on_what) do
    if train_on_what == TrainOnWhat.customized() do
      for msg <- messages do
        unless Map.has_key?(msg, :trainable) && msg.trainable != nil do
          raise ArgumentError,
                "When using CUSTOMIZED train_on_what, each message must have a trainable field"
        end
      end
    else
      for msg <- messages do
        if Map.get(msg, :trainable) != nil do
          raise ArgumentError,
                "When using non-CUSTOMIZED train_on_what, messages must not have trainable field"
        end
      end
    end

    :ok
  end

  defp compute_action_weight(message, idx, num_messages, "last_assistant_message") do
    is_last_assistant = idx == num_messages - 1 && message.role == "assistant"
    if is_last_assistant, do: 1.0, else: 0.0
  end

  defp compute_action_weight(message, _idx, _num_messages, "all_assistant_messages") do
    if message.role == "assistant", do: 1.0, else: 0.0
  end

  defp compute_action_weight(_message, _idx, _num_messages, "all_messages"), do: 1.0
  defp compute_action_weight(_message, _idx, _num_messages, "all_tokens"), do: 1.0

  defp compute_action_weight(message, _idx, _num_messages, "all_user_and_system_messages") do
    if message.role in ["user", "system"], do: 1.0, else: 0.0
  end

  defp compute_action_weight(message, _idx, _num_messages, "customized") do
    if message.trainable, do: 1.0, else: 0.0
  end

  defp compute_action_weight(_message, _idx, _num_messages, train_on_what) do
    raise ArgumentError, "Unknown train_on_what: #{inspect(train_on_what)}"
  end

  defp chunk_length(%EncodedTextChunk{tokens: tokens}), do: length(tokens)
  defp chunk_length(%{expected_tokens: n}), do: n

  defp append_weighted_chunk(acc, nil, _weight), do: acc
  defp append_weighted_chunk(acc, chunk, weight), do: acc ++ [{chunk, weight}]

  defp append_weighted_content(acc, content, weight) do
    Enum.reduce(content, acc, fn
      nil, acc -> acc
      chunk, acc -> acc ++ [{chunk, weight}]
    end)
  end

  defp append_weighted_suffix(acc, _suffix, _weight, false), do: acc
  defp append_weighted_suffix(acc, nil, _weight, true), do: acc
  defp append_weighted_suffix(acc, suffix, weight, true), do: acc ++ [{suffix, weight}]
end
