defmodule TinkexCookbook.Renderers.Qwen3VL do
  @moduledoc """
  Renderer for Qwen3-VL models.
  """

  @behaviour TinkexCookbook.Renderers.Renderer

  alias TinkexCookbook.Renderers.{Helpers, Qwen3, ToolCalls, Types, Vision}
  alias TinkexCookbook.Types.EncodedTextChunk

  @type state :: %{
          tokenizer: module() | map(),
          image_processor: Vision.image_processor()
        }

  @impl true
  def init(opts) do
    tokenizer = Keyword.fetch!(opts, :tokenizer)
    image_processor = Keyword.fetch!(opts, :image_processor)
    {:ok, %{tokenizer: tokenizer, image_processor: image_processor}}
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
      raise ArgumentError, "Thinking tokens not supported in Qwen3 VL renderer"
    end

    maybe_newline = if idx > 0, do: "\n", else: ""
    role = Qwen3.qwen_role(message)
    ob_str = "#{maybe_newline}<|im_start|>#{role}\n"

    parts =
      message.content
      |> Types.ensure_parts()
      |> wrap_vision_parts()

    parts =
      if message.role == "tool" do
        wrap_tool_response_parts(parts)
      else
        parts
      end

    contains_think =
      Enum.any?(parts, fn
        %Types.TextPart{text: text} -> String.contains?(text, "<think>")
        _ -> false
      end)

    ob_str =
      if message.role == "assistant" and not contains_think do
        ob_str <> "<think>\n"
      else
        ob_str
      end

    parts =
      if is_list(message.tool_calls) do
        parts ++ [Types.text_part(ToolCalls.encode_qwen3(message.tool_calls))]
      else
        parts
      end

    parts = parts ++ [Types.text_part("<|im_end|>")]

    rendered = %Types.RenderedMessage{
      prefix: %EncodedTextChunk{
        tokens: Helpers.encode(tokenizer, ob_str, add_special_tokens: false)
      },
      content: encode_parts(parts, tokenizer, state.image_processor),
      suffix: nil
    }

    {rendered, state}
  end

  @impl true
  def parse_response(tokens, state), do: Qwen3.parse_response(tokens, state)

  @doc false
  def wrap_vision_parts(parts) do
    Enum.flat_map(parts, fn
      %Types.ImagePart{} = part ->
        [
          Types.text_part("<|vision_start|>"),
          part,
          Types.text_part("<|vision_end|>")
        ]

      part ->
        [part]
    end)
  end

  @doc false
  def wrap_tool_response_parts(parts) do
    [Types.text_part("<tool_response>\n")] ++ parts ++ [Types.text_part("\n</tool_response>")]
  end

  @doc false
  def encode_parts(parts, tokenizer, image_processor) do
    Enum.map(parts, fn
      %Types.ImagePart{image: image} ->
        Vision.image_to_chunk(image, image_processor)

      %Types.TextPart{text: text} ->
        %EncodedTextChunk{tokens: Helpers.encode(tokenizer, text, add_special_tokens: false)}
    end)
  end
end

defmodule TinkexCookbook.Renderers.Qwen3VLInstruct do
  @moduledoc """
  Renderer for Qwen3-VL Instruct models (no <think> tags).
  """

  @behaviour TinkexCookbook.Renderers.Renderer

  alias TinkexCookbook.Renderers.{Helpers, Qwen3, Qwen3VL, ToolCalls, Types}
  alias TinkexCookbook.Types.EncodedTextChunk

  @type state :: Qwen3VL.state()

  @impl true
  def init(opts), do: Qwen3VL.init(opts)

  @impl true
  def bos_tokens(state), do: Qwen3VL.bos_tokens(state)

  @impl true
  def stop_sequences(state), do: Qwen3VL.stop_sequences(state)

  @impl true
  def render_message(idx, message, _is_last, %{tokenizer: tokenizer} = state) do
    if message.thinking != nil do
      raise ArgumentError, "Thinking tokens not supported in Qwen3 VL instruct renderer"
    end

    maybe_newline = if idx > 0, do: "\n", else: ""
    role = Qwen3.qwen_role(message)
    ob_str = "#{maybe_newline}<|im_start|>#{role}\n"

    parts =
      message.content
      |> Types.ensure_parts()
      |> Qwen3VL.wrap_vision_parts()

    parts =
      if message.role == "tool" do
        Qwen3VL.wrap_tool_response_parts(parts)
      else
        parts
      end

    parts =
      if is_list(message.tool_calls) do
        parts ++ [Types.text_part(ToolCalls.encode_qwen3(message.tool_calls))]
      else
        parts
      end

    parts = parts ++ [Types.text_part("<|im_end|>")]

    rendered = %Types.RenderedMessage{
      prefix: %EncodedTextChunk{
        tokens: Helpers.encode(tokenizer, ob_str, add_special_tokens: false)
      },
      content: Qwen3VL.encode_parts(parts, tokenizer, state.image_processor),
      suffix: nil
    }

    {rendered, state}
  end

  @impl true
  def parse_response(tokens, state), do: Qwen3.parse_response(tokens, state)
end
