defmodule TinkexCookbook.Renderers.RoleColonTest do
  use ExUnit.Case, async: true

  alias TinkexCookbook.Renderers.{RoleColon, Types}
  alias TinkexCookbook.Test.MockTokenizer
  alias TinkexCookbook.Types.EncodedTextChunk

  test "stop_sequences returns user delimiter" do
    {:ok, state} = RoleColon.init(tokenizer: MockTokenizer)
    assert RoleColon.stop_sequences(state) == ["\n\nUser:"]
  end

  test "renders assistant message with suffix user marker" do
    {:ok, state} = RoleColon.init(tokenizer: MockTokenizer)
    message = Types.message("assistant", "Hello")

    {rendered, _} = RoleColon.render_message(0, message, true, state)

    %EncodedTextChunk{tokens: prefix_tokens} = rendered.prefix
    %EncodedTextChunk{tokens: suffix_tokens} = rendered.suffix

    assert MockTokenizer.decode(prefix_tokens) == "Assistant:"
    assert MockTokenizer.decode(suffix_tokens) == "User:"
  end

  test "parse_response splits on user delimiter" do
    {:ok, state} = RoleColon.init(tokenizer: MockTokenizer)
    tokens = MockTokenizer.encode("Hi there\n\nUser:")

    {message, format_ok} = RoleColon.parse_response(tokens, state)

    assert message.role == "assistant"
    assert message.content == "Hi there"
    assert format_ok == true
  end

  test "parse_response returns format false without delimiter" do
    {:ok, state} = RoleColon.init(tokenizer: MockTokenizer)
    tokens = MockTokenizer.encode("Hi there")

    {message, format_ok} = RoleColon.parse_response(tokens, state)

    assert message.content == "Hi there"
    assert format_ok == false
  end

  test "parse_response raises on multiple delimiters" do
    {:ok, state} = RoleColon.init(tokenizer: MockTokenizer)
    tokens = MockTokenizer.encode("A\n\nUser:B\n\nUser:")

    assert_raise RuntimeError, ~r/expected to split into 1 or 2/, fn ->
      RoleColon.parse_response(tokens, state)
    end
  end
end
