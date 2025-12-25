defmodule TinkexCookbook.Renderers.GptOss do
  @moduledoc """
  Renderer for gpt-oss models.
  """

  @behaviour TinkexCookbook.Renderers.Renderer

  alias TinkexCookbook.Renderers.{Helpers, Types}
  alias TinkexCookbook.Types.EncodedTextChunk

  @system_prompt "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n" <>
                   "Knowledge cutoff: 2024-06\nCurrent date: {current_date}\n\nReasoning: {reasoning_effort}\n\n" <>
                   "# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"

  @type state :: %{
          tokenizer: module() | map(),
          use_system_prompt: boolean(),
          reasoning_effort: String.t() | nil,
          current_date: String.t() | nil
        }

  @impl true
  def init(opts) do
    tokenizer = Keyword.fetch!(opts, :tokenizer)
    use_system_prompt = Keyword.get(opts, :use_system_prompt, false)
    reasoning_effort = Keyword.get(opts, :reasoning_effort)
    current_date = Keyword.get(opts, :current_date)

    if use_system_prompt != (reasoning_effort != nil) do
      raise ArgumentError, "Reasoning effort must be set iff using system prompt"
    end

    {:ok,
     %{
       tokenizer: tokenizer,
       use_system_prompt: use_system_prompt,
       reasoning_effort: reasoning_effort,
       current_date: current_date
     }}
  end

  @impl true
  def bos_tokens(%{tokenizer: tokenizer} = state) do
    if state.use_system_prompt do
      Helpers.encode(tokenizer, build_system_prompt(state), add_special_tokens: false)
    else
      []
    end
  end

  @impl true
  def stop_sequences(%{tokenizer: tokenizer}) do
    tokens = Helpers.encode(tokenizer, "<|return|>", add_special_tokens: false)

    if length(tokens) != 1 do
      raise ArgumentError, "Expected single token for <|return|>, got #{length(tokens)}"
    end

    tokens
  end

  @impl true
  def render_message(_idx, message, is_last, %{tokenizer: tokenizer} = state) do
    if message.tool_calls != nil do
      raise ArgumentError, "Tool calls not supported in GptOss renderer"
    end

    unless is_binary(message.content) do
      raise ArgumentError, "GptOssRenderer only supports message with string content"
    end

    content = Types.ensure_text(message.content)
    ob_str = "<|start|>#{message.role}"

    ac_str =
      if message.role == "assistant" do
        thinking = message.thinking
        message_content = content

        analysis =
          if thinking && is_last do
            "<|channel|>analysis<|message|>" <>
              thinking <> "<|end|><|start|>assistant"
          else
            ""
          end

        analysis <> "<|channel|>final<|message|>" <> message_content
      else
        if message.thinking != nil do
          raise ArgumentError, "Thinking is only allowed for assistant messages"
        end

        "<|message|>" <> content
      end

    ac_str =
      if is_last do
        ac_str <> "<|return|>"
      else
        ac_str <> "<|end|>"
      end

    rendered = %Types.RenderedMessage{
      prefix: %EncodedTextChunk{
        tokens: Helpers.encode(tokenizer, ob_str, add_special_tokens: false)
      },
      content: [
        %EncodedTextChunk{tokens: Helpers.encode(tokenizer, ac_str, add_special_tokens: false)}
      ],
      suffix: nil
    }

    {rendered, state}
  end

  @impl true
  def parse_response(tokens, %{tokenizer: tokenizer} = state) do
    [stop_token] = stop_sequences(state)
    Helpers.parse_response_for_stop_token(tokens, tokenizer, stop_token)
  end

  defp build_system_prompt(state) do
    current_date =
      state.current_date ||
        Date.utc_today()
        |> Date.to_string()

    String.replace(@system_prompt, "{current_date}", current_date)
    |> String.replace("{reasoning_effort}", state.reasoning_effort)
  end
end
