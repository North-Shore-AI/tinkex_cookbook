defmodule TinkexCookbook.Renderers.KimiK2 do
  @moduledoc """
  Renderer for Kimi K2 models.
  """

  @behaviour TinkexCookbook.Renderers.Renderer

  alias TinkexCookbook.Renderers.{Helpers, ToolCalls, TrainOnWhat, Types}
  alias TinkexCookbook.Types.{EncodedTextChunk, ModelInput}

  @type state :: %{tokenizer: module() | map()}

  @impl true
  def init(opts) do
    tokenizer = Keyword.fetch!(opts, :tokenizer)
    {:ok, %{tokenizer: tokenizer}}
  end

  @impl true
  def bos_tokens(_state), do: []

  @impl true
  def stop_sequences(%{tokenizer: tokenizer}) do
    tokens = Helpers.encode(tokenizer, "<|im_end|>")

    if length(tokens) != 1 do
      raise ArgumentError, "Expected single token for <|im_end|>, got #{length(tokens)}"
    end

    tokens
  end

  @impl true
  def render_message(_idx, message, is_last, %{tokenizer: tokenizer} = state) do
    unless is_binary(message.content) do
      raise ArgumentError, "KimiK2Renderer only supports message with string content"
    end

    content = Types.ensure_text(message.content)
    role = message.role
    role_name = message.name || role

    ob_str = build_observation_string(role, role_name, message)
    ac_str = build_action_string(role, message, is_last, content)
    ac_str = ac_str <> "<|im_end|>"

    rendered = %Types.RenderedMessage{
      prefix: %EncodedTextChunk{tokens: Helpers.encode(tokenizer, ob_str)},
      content: [%EncodedTextChunk{tokens: Helpers.encode(tokenizer, ac_str)}],
      suffix: nil
    }

    {rendered, state}
  end

  defp build_observation_string(role, role_name, message) do
    cond do
      role == "user" ->
        "<|im_user|>#{role_name}<|im_middle|>"

      role == "assistant" ->
        "<|im_assistant|>#{role_name}<|im_middle|>"

      role == "system" ->
        "<|im_system|>#{role_name}<|im_middle|>"

      role == "tool" ->
        tool_call_id = message.tool_call_id || ""
        "<|im_system|>#{role_name}<|im_middle|>## Return of #{tool_call_id}\n"

      true ->
        "<|im_system|>#{role_name}<|im_middle|>"
    end
  end

  defp build_action_string(role, message, is_last, content) do
    if role == "assistant" do
      thinking = message.thinking || ""

      thinking_block =
        if is_last and thinking != "" do
          "<think>#{thinking}</think>"
        else
          "<think></think>"
        end

      base = thinking_block <> content

      if is_list(message.tool_calls) and message.tool_calls != [] do
        base <> ToolCalls.encode_kimi_k2(message.tool_calls)
      else
        base
      end
    else
      content
    end
  end

  @impl true
  def parse_response(tokens, %{tokenizer: tokenizer} = state) do
    [stop_token] = stop_sequences(state)
    {message, format_ok} = Helpers.parse_response_for_stop_token(tokens, tokenizer, stop_token)

    if format_ok do
      parse_kimi_response_with_thinking_and_tools(message)
    else
      {message, false}
    end
  end

  defp parse_kimi_response_with_thinking_and_tools(message) do
    content = Types.ensure_text(message.content)

    {content, message} =
      case extract_thinking(content) do
        nil ->
          {content, message}

        {thinking, trimmed} ->
          {trimmed, %{message | thinking: thinking, content: trimmed}}
      end

    case ToolCalls.decode_kimi_k2(content) do
      {:ok, %{content: cleaned, tool_calls: []}} ->
        {%{message | content: cleaned}, true}

      {:ok, %{content: cleaned, tool_calls: tool_calls}} ->
        {%{message | content: cleaned, tool_calls: tool_calls}, true}

      {:error, _} ->
        {message, false}
    end
  end

  @spec build_generation_prompt([Types.Message.t()], String.t(), String.t() | nil, map()) ::
          {ModelInput.t(), map()}
  def build_generation_prompt(messages, role, prefill, %{tokenizer: tokenizer} = state) do
    chunks = []

    chunks =
      if messages == [] or hd(messages).role != "system" do
        chunks ++ [default_system_chunk(tokenizer)]
      else
        chunks
      end

    chunks =
      messages
      |> Enum.with_index()
      |> Enum.reduce(chunks, fn {message, idx}, acc ->
        {rendered, _} = render_message(idx, message, false, state)

        acc = acc ++ [rendered.prefix]
        acc ++ Enum.filter(rendered.content, & &1)
      end)

    gen_prompt = "<|im_assistant|>#{role}<|im_middle|>"
    chunks = chunks ++ [%EncodedTextChunk{tokens: Helpers.encode(tokenizer, gen_prompt)}]

    chunks =
      if prefill do
        chunks ++ [%EncodedTextChunk{tokens: Helpers.encode(tokenizer, prefill)}]
      else
        chunks
      end

    {ModelInput.new(chunks), state}
  end

  @spec build_supervised_example([Types.Message.t()], TrainOnWhat.t(), map()) ::
          {ModelInput.t(), [float()]}
  def build_supervised_example(messages, train_on_what, %{tokenizer: tokenizer} = state) do
    validate_train_on_what!(messages, train_on_what)

    last_assistant_idx = find_last_assistant_index(messages)
    initial_chunks = maybe_add_default_system_chunk(messages, tokenizer)

    {chunks_weights, _} =
      messages
      |> Enum.with_index()
      |> Enum.reduce({initial_chunks, state}, fn {message, idx}, {acc, st} ->
        process_message_for_supervised_example(
          message,
          idx,
          length(messages),
          last_assistant_idx,
          train_on_what,
          acc,
          st
        )
      end)

    chunks = Enum.map(chunks_weights, fn {chunk, _weight} -> chunk end)

    weights =
      Enum.flat_map(chunks_weights, fn {chunk, weight} ->
        List.duplicate(weight, chunk_length(chunk))
      end)

    {ModelInput.new(chunks), weights}
  end

  defp find_last_assistant_index(messages) do
    messages
    |> Enum.with_index()
    |> Enum.reverse()
    |> Enum.find_value(-1, fn {msg, idx} ->
      if msg.role == "assistant" and (msg.tool_calls == nil or msg.tool_calls == []) do
        idx
      else
        nil
      end
    end)
  end

  defp maybe_add_default_system_chunk(messages, tokenizer) do
    if messages == [] or hd(messages).role != "system" do
      [{default_system_chunk(tokenizer), 0.0}]
    else
      []
    end
  end

  defp process_message_for_supervised_example(
         message,
         idx,
         total_messages,
         last_assistant_idx,
         train_on_what,
         acc,
         state
       ) do
    is_last_message = idx == total_messages - 1
    is_assistant = message.role == "assistant"
    is_user_or_system = message.role in ["user", "system"]
    is_last_assistant = idx >= last_assistant_idx and is_assistant

    {rendered, new_state} = render_message(idx, message, is_last_assistant, state)

    ob_weight = if train_on_what == TrainOnWhat.all_tokens(), do: 1.0, else: 0.0
    acc = acc ++ [{rendered.prefix, ob_weight}]

    action_weight =
      calculate_action_weight(
        train_on_what,
        is_last_message,
        is_assistant,
        is_user_or_system,
        message
      )

    acc =
      acc ++
        Enum.flat_map(rendered.content, fn
          nil -> []
          chunk -> [{chunk, action_weight}]
        end)

    acc =
      if is_last_message and rendered.suffix do
        acc ++ [{rendered.suffix, action_weight}]
      else
        acc
      end

    {acc, new_state}
  end

  defp calculate_action_weight(
         train_on_what,
         is_last_message,
         is_assistant,
         is_user_or_system,
         message
       ) do
    has_weight =
      has_action_weight?(train_on_what, is_last_message, is_assistant, is_user_or_system, message)

    if has_weight, do: 1.0, else: 0.0
  end

  defp has_action_weight?(
         "last_assistant_message",
         is_last_message,
         is_assistant,
         _is_user_or_system,
         _message
       ) do
    is_last_message and is_assistant
  end

  defp has_action_weight?(
         "all_assistant_messages",
         _is_last_message,
         is_assistant,
         _is_user_or_system,
         _message
       ) do
    is_assistant
  end

  defp has_action_weight?(
         "all_messages",
         _is_last_message,
         _is_assistant,
         _is_user_or_system,
         _message
       ) do
    true
  end

  defp has_action_weight?(
         "all_tokens",
         _is_last_message,
         _is_assistant,
         _is_user_or_system,
         _message
       ) do
    true
  end

  defp has_action_weight?(
         "all_user_and_system_messages",
         _is_last_message,
         _is_assistant,
         is_user_or_system,
         _message
       ) do
    is_user_or_system
  end

  defp has_action_weight?(
         "customized",
         _is_last_message,
         _is_assistant,
         _is_user_or_system,
         message
       ) do
    message.trainable == true
  end

  defp has_action_weight?(other, _is_last_message, _is_assistant, _is_user_or_system, _message) do
    raise ArgumentError, "Unknown train_on_what: #{inspect(other)}"
  end

  defp default_system_chunk(tokenizer) do
    system_str =
      "<|im_system|>system<|im_middle|>You are Kimi, an AI assistant created by Moonshot AI.<|im_end|>"

    %EncodedTextChunk{tokens: Helpers.encode(tokenizer, system_str)}
  end

  defp extract_thinking(content) do
    regex = ~r/<think>(.*?)<\/think>/s

    case Regex.run(regex, content, return: :index) do
      nil ->
        nil

      [{start, len} | _] ->
        [thinking] = Regex.run(regex, content, capture: :all_but_first)
        rest = binary_part(content, start + len, byte_size(content) - start - len)
        {thinking, String.trim_leading(rest)}
    end
  end

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
  end

  defp chunk_length(%EncodedTextChunk{tokens: tokens}), do: length(tokens)
  defp chunk_length(%{expected_tokens: n}), do: n
end
