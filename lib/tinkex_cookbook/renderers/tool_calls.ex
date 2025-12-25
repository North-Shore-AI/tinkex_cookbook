defmodule TinkexCookbook.Renderers.ToolCalls do
  @moduledoc """
  Shared helpers for encoding and decoding tool calls across renderers.
  """

  alias TinkexCookbook.Renderers.Types

  @type decode_result :: %{content: String.t(), tool_calls: [Types.ToolCall.t()]}

  @spec encode_qwen3([Types.ToolCall.t()]) :: String.t()
  def encode_qwen3(tool_calls) when is_list(tool_calls) do
    Enum.map_join(tool_calls, "\n", &encode_qwen3_tool_call!/1)
  end

  @spec decode_qwen3(String.t()) :: {:ok, decode_result()} | {:error, :invalid_tool_call}
  def decode_qwen3(content) when is_binary(content) do
    matches = Regex.scan(~r/<tool_call>(.*?)<\/tool_call>/s, content, capture: :all_but_first)

    if matches == [] do
      {:ok, %{content: content, tool_calls: []}}
    else
      with {:ok, tool_calls} <- decode_qwen3_matches(matches) do
        cleaned =
          Regex.replace(~r/\n?<tool_call>.*?<\/tool_call>/s, content, "")
          |> String.trim()

        {:ok, %{content: cleaned, tool_calls: tool_calls}}
      end
    end
  end

  @spec encode_kimi_k2([Types.ToolCall.t()]) :: String.t()
  def encode_kimi_k2(tool_calls) when is_list(tool_calls) do
    if Enum.empty?(tool_calls) do
      ""
    else
      calls =
        Enum.map_join(tool_calls, "", fn tool_call ->
          tool_id = tool_call.id || ""
          args = tool_call.function.arguments

          "<|tool_call_begin|>" <>
            tool_id <>
            "<|tool_call_argument_begin|>" <>
            args <>
            "<|tool_call_end|>"
        end)

      "<|tool_calls_section_begin|>" <> calls <> "<|tool_calls_section_end|>"
    end
  end

  @spec decode_kimi_k2(String.t()) :: {:ok, decode_result()} | {:error, :invalid_tool_call}
  def decode_kimi_k2(content) when is_binary(content) do
    if String.contains?(content, "<|tool_calls_section_begin|>") do
      decode_kimi_k2_with_section(content)
    else
      {:ok, %{content: content, tool_calls: []}}
    end
  end

  defp decode_kimi_k2_with_section(content) do
    case Regex.run(
           ~r/<\|tool_calls_section_begin\|>(.*?)<\|tool_calls_section_end\|>/s,
           content,
           capture: :all_but_first
         ) do
      nil ->
        {:ok, %{content: content, tool_calls: []}}

      [section] ->
        with {:ok, tool_calls} <- decode_kimi_k2_section(section) do
          handle_decoded_tool_calls(content, tool_calls)
        end
    end
  end

  defp handle_decoded_tool_calls(content, []), do: {:ok, %{content: content, tool_calls: []}}

  defp handle_decoded_tool_calls(content, tool_calls) do
    [prefix | _] = String.split(content, "<|tool_calls_section_begin|>", parts: 2)
    {:ok, %{content: prefix, tool_calls: tool_calls}}
  end

  defp encode_qwen3_tool_call!(%Types.ToolCall{} = tool_call) do
    %Types.FunctionBody{name: name, arguments: arguments} = tool_call.function

    payload =
      case Jason.decode(arguments) do
        {:ok, decoded} when is_map(decoded) ->
          %{"name" => name, "arguments" => decoded}

        {:ok, _} ->
          raise ArgumentError, "Qwen3 tool call arguments must decode to a JSON object"

        {:error, _} ->
          raise ArgumentError, "Qwen3 tool call arguments must be valid JSON"
      end

    "<tool_call>\n" <> Jason.encode!(payload) <> "\n</tool_call>"
  end

  defp decode_qwen3_matches(matches) do
    Enum.reduce_while(matches, {:ok, []}, fn [match], {:ok, acc} ->
      case parse_qwen3_tool_call(match) do
        {:ok, tool_call} -> {:cont, {:ok, acc ++ [tool_call]}}
        {:error, _} -> {:halt, {:error, :invalid_tool_call}}
      end
    end)
  end

  defp parse_qwen3_tool_call(tool_call_str) do
    with {:ok, decoded} <- Jason.decode(String.trim(tool_call_str)),
         true <- is_map(decoded),
         name when is_binary(name) <- Map.get(decoded, "name"),
         args when is_map(args) <- Map.get(decoded, "arguments") do
      id =
        case Map.get(decoded, "id") do
          value when is_binary(value) -> value
          _ -> nil
        end

      {:ok, Types.tool_call(name, Jason.encode!(args), id: id)}
    else
      _ -> {:error, :invalid_tool_call}
    end
  end

  defp decode_kimi_k2_section(section) do
    pattern =
      ~r/<\|tool_call_begin\|>(.*?)<\|tool_call_argument_begin\|>(.*?)<\|tool_call_end\|>/s

    matches = Regex.scan(pattern, section, capture: :all_but_first)

    Enum.reduce_while(matches, {:ok, []}, fn match, {:ok, acc} ->
      process_kimi_tool_call_match(match, acc)
    end)
  end

  defp process_kimi_tool_call_match([tool_id, args_str], acc) do
    tool_id = String.trim(tool_id)
    args_str = String.trim(args_str)

    case Jason.decode(args_str) do
      {:ok, _} ->
        tool_call = build_kimi_tool_call(tool_id, args_str)
        {:cont, {:ok, acc ++ [tool_call]}}

      {:error, _} ->
        {:halt, {:error, :invalid_tool_call}}
    end
  end

  defp build_kimi_tool_call(tool_id, args_str) do
    func_name = extract_kimi_k2_function_name(tool_id)
    id = if tool_id == "", do: nil, else: tool_id
    Types.tool_call(func_name, args_str, id: id)
  end

  defp extract_kimi_k2_function_name(tool_id) do
    cond do
      tool_id == "" ->
        ""

      not String.contains?(tool_id, ".") ->
        ""

      true ->
        [_prefix, rest] = String.split(tool_id, ".", parts: 2)
        extract_function_name_from_rest(rest)
    end
  end

  defp extract_function_name_from_rest(rest) do
    if String.contains?(rest, ":") do
      [name | _] = String.split(rest, ":", parts: 2)
      name
    else
      rest
    end
  end
end
