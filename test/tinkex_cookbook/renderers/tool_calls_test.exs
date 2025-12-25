defmodule TinkexCookbook.Renderers.ToolCallsTest do
  use ExUnit.Case, async: true

  alias TinkexCookbook.Renderers.ToolCalls
  alias TinkexCookbook.Renderers.Types

  describe "encode_qwen3/1 and decode_qwen3/1" do
    test "round-trips tool calls and strips blocks from content" do
      tool_call = Types.tool_call("search", ~s({"query":"elixir"}), id: "call_1")
      encoded = ToolCalls.encode_qwen3([tool_call])

      assert String.contains?(encoded, "<tool_call>")
      assert String.contains?(encoded, "</tool_call>")

      content = "Result\n" <> encoded

      assert {:ok, %{content: cleaned, tool_calls: [parsed]}} =
               ToolCalls.decode_qwen3(content)

      assert cleaned == "Result"
      assert parsed.function.name == "search"
      assert parsed.function.arguments == ~s({"query":"elixir"})
      assert parsed.id == nil
    end

    test "returns error on invalid tool call JSON" do
      content = "Result<tool_call>{not json}</tool_call>"

      assert {:error, :invalid_tool_call} = ToolCalls.decode_qwen3(content)
    end
  end

  describe "encode_kimi_k2/1 and decode_kimi_k2/1" do
    test "round-trips tool calls and strips tool section from content" do
      tool_call =
        Types.tool_call("get_weather", ~s({"city":"SF"}), id: "functions.get_weather:0")

      encoded = ToolCalls.encode_kimi_k2([tool_call])

      assert String.contains?(encoded, "<|tool_calls_section_begin|>")
      assert String.contains?(encoded, "<|tool_calls_section_end|>")

      content = "Answer" <> encoded

      assert {:ok, %{content: cleaned, tool_calls: [parsed]}} =
               ToolCalls.decode_kimi_k2(content)

      assert cleaned == "Answer"
      assert parsed.id == "functions.get_weather:0"
      assert parsed.function.name == "get_weather"
      assert parsed.function.arguments == ~s({"city":"SF"})
    end

    test "returns error on invalid tool call arguments" do
      content =
        "Answer<|tool_calls_section_begin|>" <>
          "<|tool_call_begin|>functions.bad:0<|tool_call_argument_begin|>{bad}<|tool_call_end|>" <>
          "<|tool_calls_section_end|>"

      assert {:error, :invalid_tool_call} = ToolCalls.decode_kimi_k2(content)
    end
  end
end
