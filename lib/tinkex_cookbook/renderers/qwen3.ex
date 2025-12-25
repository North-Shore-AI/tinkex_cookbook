defmodule TinkexCookbook.Renderers.Qwen3 do
  @moduledoc """
  Renderer for Qwen3 models with thinking enabled.
  """

  @behaviour TinkexCookbook.Renderers.Renderer

  alias TinkexCookbook.Renderers.{Helpers, ToolCalls, Types}
  alias TinkexCookbook.Types.EncodedTextChunk

  @type state :: %{
          tokenizer: module() | map(),
          strip_thinking_from_history: boolean()
        }

  @impl true
  def init(opts) do
    tokenizer = Keyword.fetch!(opts, :tokenizer)
    strip_thinking = Keyword.get(opts, :strip_thinking_from_history, true)

    {:ok,
     %{
       tokenizer: tokenizer,
       strip_thinking_from_history: strip_thinking
     }}
  end

  @impl true
  def bos_tokens(_state), do: []

  @impl true
  def stop_sequences(%{tokenizer: tokenizer}) do
    tokens = Helpers.encode(tokenizer, "<|im_end|>", add_special_tokens: false)

    if length(tokens) != 1 do
      raise ArgumentError, "Expected single token for <|im_end|>, got #{length(tokens)}"
    end

    tokens
  end

  @impl true
  def render_message(idx, message, is_last, %{tokenizer: tokenizer} = state) do
    validate_qwen3_message!(message)

    content = Types.ensure_text(message.content)
    maybe_newline = if idx > 0, do: "\n", else: ""

    role = qwen_role(message)
    ob_str = "#{maybe_newline}<|im_start|>#{role}\n"

    ac_content = prepare_action_content(message, content)
    {ac_content, ob_str} = apply_thinking_rules(ac_content, ob_str, message, is_last, state)
    ac_content = append_tool_calls(ac_content, message)
    ac_content = ac_content <> "<|im_end|>"

    rendered = %Types.RenderedMessage{
      prefix: %EncodedTextChunk{
        tokens: Helpers.encode(tokenizer, ob_str, add_special_tokens: false)
      },
      content: [
        %EncodedTextChunk{
          tokens: Helpers.encode(tokenizer, ac_content, add_special_tokens: false)
        }
      ],
      suffix: nil
    }

    {rendered, state}
  end

  defp validate_qwen3_message!(message) do
    if message.thinking != nil do
      raise ArgumentError, "Thinking tokens not supported in Qwen3 renderer"
    end

    unless is_binary(message.content) do
      raise ArgumentError, "Qwen3Renderer only supports message with string content"
    end
  end

  defp prepare_action_content(message, content) do
    if message.role == "tool" do
      wrap_tool_response(content)
    else
      content
    end
  end

  defp apply_thinking_rules(ac_content, ob_str, message, is_last, state) do
    cond do
      should_strip_thinking?(state, message, ac_content, is_last) ->
        [_before, rest] = String.split(ac_content, "</think>", parts: 2)
        {String.trim_leading(rest), ob_str}

      should_add_thinking?(message, ac_content, is_last) ->
        {ac_content, ob_str <> "<think>\n"}

      true ->
        {ac_content, ob_str}
    end
  end

  defp should_strip_thinking?(state, message, ac_content, is_last) do
    state.strip_thinking_from_history and message.role == "assistant" and
      String.contains?(ac_content, "</think>") and not is_last
  end

  defp should_add_thinking?(message, ac_content, is_last) do
    message.role == "assistant" and not String.contains?(ac_content, "<think>") and is_last
  end

  defp append_tool_calls(ac_content, message) do
    if is_list(message.tool_calls) and message.tool_calls != [] do
      ac_content <> ToolCalls.encode_qwen3(message.tool_calls)
    else
      ac_content
    end
  end

  @impl true
  def parse_response(tokens, %{tokenizer: tokenizer} = state) do
    [stop_token] = stop_sequences(state)
    {message, format_ok} = Helpers.parse_response_for_stop_token(tokens, tokenizer, stop_token)

    if format_ok do
      parse_response_with_tool_calls(message)
    else
      {message, false}
    end
  end

  defp parse_response_with_tool_calls(message) do
    content = Types.ensure_text(message.content)

    case ToolCalls.decode_qwen3(content) do
      {:ok, %{content: cleaned, tool_calls: []}} ->
        {%{message | content: cleaned}, true}

      {:ok, %{content: cleaned, tool_calls: tool_calls}} ->
        {%{message | content: cleaned, tool_calls: tool_calls}, true}

      {:error, _} ->
        {message, false}
    end
  end

  @doc false
  def qwen_role(%Types.Message{role: "tool"}), do: "user"
  def qwen_role(%Types.Message{role: role}), do: role

  @doc false
  def wrap_tool_response(content) when is_binary(content) do
    "<tool_response>\n" <> content <> "\n</tool_response>"
  end
end

defmodule TinkexCookbook.Renderers.Qwen3DisableThinking do
  @moduledoc """
  Renderer for Qwen3 models with thinking disabled.
  """

  @behaviour TinkexCookbook.Renderers.Renderer

  alias TinkexCookbook.Renderers.{Qwen3, Renderer, Types}

  @impl true
  def init(opts), do: Qwen3.init(opts)

  @impl true
  def bos_tokens(state), do: Qwen3.bos_tokens(state)

  @impl true
  def stop_sequences(state), do: Qwen3.stop_sequences(state)

  @impl true
  def render_message(idx, message, is_last, state) do
    message =
      if message.role == "assistant" and is_binary(message.content) and
           not String.contains?(message.content, "<think>") do
        %{message | content: "<think>\n\n</think>\n\n" <> message.content}
      else
        message
      end

    Qwen3.render_message(idx, message, is_last, state)
  end

  @impl true
  def parse_response(tokens, state), do: Qwen3.parse_response(tokens, state)

  @spec build_generation_prompt([Types.Message.t()], String.t(), String.t() | nil, map()) ::
          {TinkexCookbook.Types.ModelInput.t(), map()}
  def build_generation_prompt(messages, role, prefill, state) do
    prefill = "<think>\n\n</think>\n\n" <> (prefill || "")
    Renderer.build_generation_prompt_default(__MODULE__, messages, role, prefill, state)
  end
end

defmodule TinkexCookbook.Renderers.Qwen3Instruct do
  @moduledoc """
  Renderer for Qwen3 instruct models (no <think> tags).
  """

  @behaviour TinkexCookbook.Renderers.Renderer

  alias TinkexCookbook.Renderers.{Helpers, ToolCalls, Types}
  alias TinkexCookbook.Types.EncodedTextChunk

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
    tokens = Helpers.encode(tokenizer, "<|im_end|>", add_special_tokens: false)

    if length(tokens) != 1 do
      raise ArgumentError, "Expected single token for <|im_end|>, got #{length(tokens)}"
    end

    tokens
  end

  @impl true
  def render_message(idx, message, _is_last, %{tokenizer: tokenizer} = state) do
    if message.thinking != nil do
      raise ArgumentError, "Thinking tokens not supported in Qwen3 instruct renderer"
    end

    unless is_binary(message.content) do
      raise ArgumentError, "Qwen3InstructRenderer only supports message with string content"
    end

    content = Types.ensure_text(message.content)
    maybe_newline = if idx > 0, do: "\n", else: ""
    role = qwen_role(message)
    ob_str = "#{maybe_newline}<|im_start|>#{role}\n"

    ac_content =
      if message.role == "tool" do
        wrap_tool_response(content)
      else
        content
      end

    ac_content =
      if is_list(message.tool_calls) and message.tool_calls != [] do
        ac_content <> ToolCalls.encode_qwen3(message.tool_calls)
      else
        ac_content
      end

    ac_content = ac_content <> "<|im_end|>"

    rendered = %Types.RenderedMessage{
      prefix: %EncodedTextChunk{
        tokens: Helpers.encode(tokenizer, ob_str, add_special_tokens: false)
      },
      content: [
        %EncodedTextChunk{
          tokens: Helpers.encode(tokenizer, ac_content, add_special_tokens: false)
        }
      ],
      suffix: nil
    }

    {rendered, state}
  end

  @impl true
  def parse_response(tokens, %{tokenizer: tokenizer} = state) do
    [stop_token] = stop_sequences(state)
    {message, format_ok} = Helpers.parse_response_for_stop_token(tokens, tokenizer, stop_token)

    if format_ok do
      parse_instruct_response_with_tool_calls(message)
    else
      {message, false}
    end
  end

  defp parse_instruct_response_with_tool_calls(message) do
    content = Types.ensure_text(message.content)

    case ToolCalls.decode_qwen3(content) do
      {:ok, %{content: cleaned, tool_calls: []}} ->
        {%{message | content: cleaned}, true}

      {:ok, %{content: cleaned, tool_calls: tool_calls}} ->
        {%{message | content: cleaned, tool_calls: tool_calls}, true}

      {:error, _} ->
        {message, false}
    end
  end

  defp qwen_role(%Types.Message{role: "tool"}), do: "user"
  defp qwen_role(%Types.Message{role: role}), do: role

  defp wrap_tool_response(content) when is_binary(content) do
    "<tool_response>\n" <> content <> "\n</tool_response>"
  end
end
