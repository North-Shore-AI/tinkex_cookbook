defmodule TinkexCookbook.Renderers.RoleColon do
  @moduledoc """
  Role-colon renderer (User: / Assistant:).
  """

  @behaviour TinkexCookbook.Renderers.Renderer

  alias TinkexCookbook.Renderers.{Helpers, Types}
  alias TinkexCookbook.Types.EncodedTextChunk

  @type state :: %{tokenizer: module() | map()}

  @impl true
  def init(opts) do
    tokenizer = Keyword.fetch!(opts, :tokenizer)
    {:ok, %{tokenizer: tokenizer}}
  end

  @impl true
  def bos_tokens(%{tokenizer: tokenizer}) do
    case Helpers.bos_token_string(tokenizer) do
      nil -> []
      token_str -> Helpers.encode(tokenizer, token_str, add_special_tokens: false)
    end
  end

  @impl true
  def stop_sequences(_state), do: ["\n\nUser:"]

  @impl true
  def render_message(_idx, message, _is_last, %{tokenizer: tokenizer} = state) do
    if message.thinking != nil do
      raise ArgumentError, "Thinking tokens not supported in RoleColonRenderer"
    end

    unless is_binary(message.content) do
      raise ArgumentError, "RoleColonRenderer only supports message with string content"
    end

    content = Types.ensure_text(message.content)
    ob_str = String.capitalize(message.role) <> ":"
    ac_str = " " <> content <> "\n\n"
    ac_tail_str = if message.role == "assistant", do: "User:", else: "<UNUSED>"

    rendered = %Types.RenderedMessage{
      prefix: %EncodedTextChunk{
        tokens: Helpers.encode(tokenizer, ob_str, add_special_tokens: false)
      },
      content: [
        %EncodedTextChunk{
          tokens: Helpers.encode(tokenizer, ac_str, add_special_tokens: false)
        }
      ],
      suffix: %EncodedTextChunk{
        tokens: Helpers.encode(tokenizer, ac_tail_str, add_special_tokens: false)
      }
    }

    {rendered, state}
  end

  @impl true
  def parse_response(tokens, %{tokenizer: tokenizer}) do
    text = Helpers.decode(tokenizer, tokens)
    parts = String.split(text, "\n\nUser:")

    case parts do
      [content] ->
        {Types.message("assistant", String.trim(content)), false}

      [content, _after] ->
        {Types.message("assistant", String.trim(content)), true}

      _ ->
        raise RuntimeError,
              "When parsing response, expected to split into 1 or 2 pieces using stop tokens, " <>
                "but got #{length(parts)}. You probably are using the wrong stop tokens when sampling"
    end
  end
end
