defmodule TinkexCookbook.Renderers.KimiK2Test do
  use ExUnit.Case, async: true

  alias TinkexCookbook.Renderers.{KimiK2, Renderer, TrainOnWhat, Types}
  alias TinkexCookbook.Test.SpecialTokenizer
  alias TinkexCookbook.Types.{EncodedTextChunk, ModelInput}

  @default_system_prompt "<|im_system|>system<|im_middle|>You are Kimi, an AI assistant created by Moonshot AI.<|im_end|>"

  test "assistant rendering preserves thinking only for last message" do
    {:ok, state} = KimiK2.init(tokenizer: SpecialTokenizer)
    message = Types.message("assistant", "Answer", thinking: "Reasoning")

    {rendered_last, _} = KimiK2.render_message(0, message, true, state)
    [%EncodedTextChunk{tokens: last_tokens}] = rendered_last.content
    last_content = SpecialTokenizer.decode(last_tokens)

    assert String.contains?(last_content, "<think>Reasoning</think>")

    {rendered_hist, _} = KimiK2.render_message(0, message, false, state)
    [%EncodedTextChunk{tokens: hist_tokens}] = rendered_hist.content
    hist_content = SpecialTokenizer.decode(hist_tokens)

    assert String.contains?(hist_content, "<think></think>")
  end

  test "assistant tool_calls section is included" do
    {:ok, state} = KimiK2.init(tokenizer: SpecialTokenizer)
    tool_call = Types.tool_call("get_weather", ~s({"city":"SF"}), id: "functions.get_weather:0")
    message = Types.message("assistant", "Answer", tool_calls: [tool_call])

    {rendered, _} = KimiK2.render_message(0, message, true, state)
    [%EncodedTextChunk{tokens: content_tokens}] = rendered.content
    content = SpecialTokenizer.decode(content_tokens)

    assert String.contains?(content, "<|tool_calls_section_begin|>")
    assert String.contains?(content, "<|tool_calls_section_end|>")
  end

  test "build_generation_prompt adds default system prompt when missing" do
    {:ok, state} = KimiK2.init(tokenizer: SpecialTokenizer)
    messages = [Types.message("user", "Hello")]

    {model_input, _} = Renderer.build_generation_prompt(KimiK2, messages, "assistant", nil, state)

    %EncodedTextChunk{tokens: first_tokens} = hd(model_input.chunks)
    assert SpecialTokenizer.decode(first_tokens) == @default_system_prompt
  end

  test "parse_response extracts thinking and tool_calls" do
    {:ok, state} = KimiK2.init(tokenizer: SpecialTokenizer)

    tool_section =
      "<|tool_calls_section_begin|>" <>
        "<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>" <>
        ~s({"city":"SF"}) <>
        "<|tool_call_end|><|tool_calls_section_end|>"

    response =
      "<think>Reasoning</think>Answer" <> tool_section <> "<|im_end|>"

    tokens = SpecialTokenizer.encode(response)
    {message, format_ok} = KimiK2.parse_response(tokens, state)

    assert format_ok == true
    assert message.thinking == "Reasoning"
    assert message.content == "Answer"
    assert [%Types.ToolCall{} = tool_call] = message.tool_calls
    assert tool_call.function.name == "get_weather"
  end

  test "build_supervised_example injects default system prompt" do
    {:ok, state} = KimiK2.init(tokenizer: SpecialTokenizer)
    messages = [Types.message("user", "Hello")]

    {model_input, weights} =
      Renderer.build_supervised_example(
        KimiK2,
        messages,
        TrainOnWhat.all_messages(),
        state
      )

    %EncodedTextChunk{tokens: first_tokens} = hd(model_input.chunks)
    assert SpecialTokenizer.decode(first_tokens) == @default_system_prompt
    assert length(weights) == ModelInput.length(model_input)
  end
end
