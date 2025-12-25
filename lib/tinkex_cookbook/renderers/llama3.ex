defmodule TinkexCookbook.Renderers.Llama3 do
  @moduledoc """
  Llama 3 family renderer.

  Formats messages using Llama 3's chat template:

      <|begin_of_text|><|start_header_id|>system<|end_header_id|>

      You are a helpful AI assistant for travel tips and recommendations<|eot_id|><|start_header_id|>user<|end_header_id|>

      What can you help me with?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

  ## Token Format

  - BOS: `<|begin_of_text|>`
  - Role header: `<|start_header_id|>{role}<|end_header_id|>\\n\\n`
  - Content ends with: `<|eot_id|>`
  - Stop sequence: `<|eot_id|>`

  ## Usage

      {:ok, state} = Llama3.init(tokenizer: MyTokenizer)
      {rendered, state} = Llama3.render_message(0, message, false, state)

  ## Tokenizer Interface

  The tokenizer can be either:
  - A module with `encode/2` and `decode/1` functions
  - A map with `:encode` and `:decode` keys containing anonymous functions

  """

  @behaviour TinkexCookbook.Renderers.Renderer

  alias TinkexCookbook.Renderers.{Helpers, Types}
  alias TinkexCookbook.Types.EncodedTextChunk

  @type state :: %{tokenizer: module() | map()}

  # Llama 3 special token strings
  @bos_str "<|begin_of_text|>"
  @start_header "<|start_header_id|>"
  @end_header "<|end_header_id|>"
  @eot_id "<|eot_id|>"

  @impl true
  @doc """
  Initialize the renderer with a tokenizer.

  ## Options

  - `:tokenizer` - Required. The tokenizer module to use for encoding.

  ## Examples

      {:ok, state} = Llama3.init(tokenizer: MyTokenizer)
  """
  @spec init(keyword()) :: {:ok, state()}
  def init(opts) do
    tokenizer = Keyword.fetch!(opts, :tokenizer)
    {:ok, %{tokenizer: tokenizer}}
  end

  @impl true
  @doc """
  Returns the BOS (beginning of sequence) tokens.

  For Llama 3, this is the encoded `<|begin_of_text|>` token.
  """
  @spec bos_tokens(state()) :: [non_neg_integer()]
  def bos_tokens(%{tokenizer: tokenizer}) do
    encode(tokenizer, @bos_str, add_special_tokens: false)
  end

  @impl true
  @doc """
  Returns the stop sequences for sampling.

  For Llama 3, this is the `<|eot_id|>` token.
  """
  @spec stop_sequences(state()) :: [non_neg_integer()]
  def stop_sequences(%{tokenizer: tokenizer}) do
    tokens = encode(tokenizer, @eot_id, add_special_tokens: false)

    case tokens do
      [token] ->
        [token]

      _ ->
        raise ArgumentError, "Expected single token for <|eot_id|>, got #{length(tokens)}"
    end
  end

  @impl true
  @doc """
  Render a single message into token chunks.

  Formats the message according to Llama 3's chat template:
  - Prefix: `<|start_header_id|>{role}<|end_header_id|>\\n\\n`
  - Content: `{content}<|eot_id|>`
  - Suffix: nil (Llama 3 doesn't use separate suffix)
  """
  @spec render_message(
          non_neg_integer(),
          Types.Message.t(),
          boolean(),
          state()
        ) :: {Types.RenderedMessage.t(), state()}
  def render_message(_idx, message, _is_last, %{tokenizer: tokenizer} = state) do
    role = message.role

    # Llama3 does not support thinking/CoT tokens
    if message.thinking != nil do
      raise ArgumentError, "CoT tokens not supported in Llama3"
    end

    unless is_binary(message.content) do
      raise ArgumentError, "Llama3Renderer only supports message with string content"
    end

    content = Types.ensure_text(message.content)

    # Build prefix: <|start_header_id|>{role}<|end_header_id|>\n\n
    prefix_str = "#{@start_header}#{role}#{@end_header}\n\n"
    prefix_tokens = encode(tokenizer, prefix_str, add_special_tokens: false)

    # Build content: {content}<|eot_id|>
    content_str = "#{content}#{@eot_id}"
    content_tokens = encode(tokenizer, content_str, add_special_tokens: false)

    rendered = %Types.RenderedMessage{
      prefix: %EncodedTextChunk{tokens: prefix_tokens},
      content: [%EncodedTextChunk{tokens: content_tokens}],
      suffix: nil
    }

    {rendered, state}
  end

  @impl true
  @doc """
  Parse a response from token sequence into a Message.

  Extracts the content before the `<|eot_id|>` marker and returns
  the message along with a boolean indicating if the response is complete.
  """
  @spec parse_response([non_neg_integer()], state()) :: {Types.Message.t(), boolean()}
  def parse_response(tokens, %{tokenizer: tokenizer} = state) do
    [stop_token] = stop_sequences(state)
    Helpers.parse_response_for_stop_token(tokens, tokenizer, stop_token)
  end

  # Helper to encode text with either a module or a map tokenizer
  defp encode(tokenizer, text, opts) when is_atom(tokenizer) do
    tokenizer.encode(text, opts)
  end

  defp encode(%{encode: encode_fn}, text, opts) when is_function(encode_fn, 2) do
    encode_fn.(text, opts)
  end
end
