defmodule TinkexCookbook.Renderers.DeepSeekV3 do
  @moduledoc """
  Renderer for DeepSeek V3 models.
  """

  @behaviour TinkexCookbook.Renderers.Renderer

  alias TinkexCookbook.Renderers.{Helpers, Types}
  alias TinkexCookbook.Types.EncodedTextChunk

  @type state :: %{
          tokenizer: module() | map(),
          system_role_as_user: boolean()
        }

  @impl true
  def init(opts) do
    tokenizer = Keyword.fetch!(opts, :tokenizer)
    system_role_as_user = Keyword.get(opts, :system_role_as_user, false)

    {:ok,
     %{
       tokenizer: tokenizer,
       system_role_as_user: system_role_as_user
     }}
  end

  @impl true
  def bos_tokens(%{tokenizer: tokenizer}) do
    [special_token(tokenizer, "begin" <> word_sep() <> "of" <> word_sep() <> "sentence")]
  end

  @impl true
  def stop_sequences(%{tokenizer: tokenizer}) do
    [special_token(tokenizer, "end" <> word_sep() <> "of" <> word_sep() <> "sentence")]
  end

  @impl true
  def render_message(_idx, message, _is_last, %{tokenizer: tokenizer} = state) do
    if message.thinking != nil do
      raise ArgumentError, "Thinking tokens not supported in DeepSeekV3 renderer"
    end

    unless is_binary(message.content) do
      raise ArgumentError, "DeepSeekV3Renderer only supports message with string content"
    end

    content = Types.ensure_text(message.content)

    role_token =
      cond do
        message.role == "user" ->
          special_token(tokenizer, "User")

        message.role == "system" and state.system_role_as_user ->
          special_token(tokenizer, "User")

        message.role == "assistant" ->
          special_token(tokenizer, "Assistant")

        true ->
          raise ArgumentError, "Unsupported role: #{message.role}"
      end

    ob_tokens = [role_token]
    ac_tokens = Helpers.encode(tokenizer, content, add_special_tokens: false)

    ac_tokens =
      if message.role == "assistant" do
        ac_tokens ++ stop_sequences(state)
      else
        ac_tokens
      end

    rendered = %Types.RenderedMessage{
      prefix: %EncodedTextChunk{tokens: ob_tokens},
      content: [%EncodedTextChunk{tokens: ac_tokens}],
      suffix: nil
    }

    {rendered, state}
  end

  @impl true
  def parse_response(tokens, %{tokenizer: tokenizer} = state) do
    [stop_token] = stop_sequences(state)
    Helpers.parse_response_for_stop_token(tokens, tokenizer, stop_token)
  end

  defp special_token(tokenizer, name) do
    sep = <<0xFF5C::utf8>>
    token_str = "<" <> sep <> name <> sep <> ">"
    tokens = Helpers.encode(tokenizer, token_str, add_special_tokens: false)

    if length(tokens) != 1 do
      raise ArgumentError, "Expected single token for #{token_str}, got #{inspect(tokens)}"
    end

    hd(tokens)
  end

  defp word_sep, do: <<0x2581::utf8>>
end

defmodule TinkexCookbook.Renderers.DeepSeekV3DisableThinking do
  @moduledoc """
  Renderer for DeepSeek V3 models with thinking disabled.
  """

  @behaviour TinkexCookbook.Renderers.Renderer

  alias TinkexCookbook.Renderers.{DeepSeekV3, Renderer}

  @impl true
  def init(opts), do: DeepSeekV3.init(opts)

  @impl true
  def bos_tokens(state), do: DeepSeekV3.bos_tokens(state)

  @impl true
  def stop_sequences(state), do: DeepSeekV3.stop_sequences(state)

  @impl true
  def render_message(idx, message, is_last, state) do
    message =
      if message.role == "assistant" and is_binary(message.content) and
           not String.starts_with?(message.content, "<think>") and
           not String.starts_with?(message.content, "</think>") do
        %{message | content: "</think>" <> message.content}
      else
        message
      end

    DeepSeekV3.render_message(idx, message, is_last, state)
  end

  @impl true
  def parse_response(tokens, state), do: DeepSeekV3.parse_response(tokens, state)

  @spec build_generation_prompt(
          [TinkexCookbook.Renderers.Types.Message.t()],
          String.t(),
          String.t() | nil,
          map()
        ) ::
          {TinkexCookbook.Types.ModelInput.t(), map()}
  def build_generation_prompt(messages, role, prefill, state) do
    prefill = "</think>" <> (prefill || "")
    Renderer.build_generation_prompt_default(__MODULE__, messages, role, prefill, state)
  end
end
