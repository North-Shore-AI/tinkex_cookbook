defmodule TinkexCookbook.Renderers.TypesTest do
  use ExUnit.Case, async: true

  alias TinkexCookbook.Renderers.Types

  describe "TextPart" do
    test "creates a text part with required fields" do
      part = Types.text_part("Hello, world!")

      assert part.type == "text"
      assert part.text == "Hello, world!"
    end

    test "new/1 creates a text part struct" do
      part = %Types.TextPart{type: "text", text: "test"}

      assert part.type == "text"
      assert part.text == "test"
    end
  end

  describe "ImagePart" do
    test "creates an image part with string URL" do
      part = Types.image_part("https://example.com/image.jpg")

      assert part.type == "image"
      assert part.image == "https://example.com/image.jpg"
    end

    test "creates an image part with binary data" do
      binary_data = <<0xFF, 0xD8, 0xFF>>
      part = Types.image_part(binary_data)

      assert part.type == "image"
      assert part.image == binary_data
    end
  end

  describe "Message" do
    test "creates a basic user message" do
      msg = Types.message("user", "Hello!")

      assert msg.role == "user"
      assert msg.content == "Hello!"
      assert msg.tool_calls == nil
      assert msg.thinking == nil
      assert msg.trainable == nil
    end

    test "creates an assistant message with thinking" do
      msg = Types.message("assistant", "The answer is 42.", thinking: "Let me think...")

      assert msg.role == "assistant"
      assert msg.content == "The answer is 42."
      assert msg.thinking == "Let me think..."
    end

    test "creates a message with trainable flag" do
      msg = Types.message("assistant", "Response", trainable: true)

      assert msg.trainable == true
    end

    test "creates a message with multimodal content" do
      content = [
        Types.text_part("Here is an image:"),
        Types.image_part("https://example.com/img.jpg")
      ]

      msg = Types.message("user", content)

      assert msg.role == "user"
      assert is_list(msg.content)
      assert length(msg.content) == 2
    end
  end

  describe "ToolCall" do
    test "creates a tool call with function body" do
      tool_call =
        Types.tool_call("search", ~s({"query": "elixir"}), id: "call_123")

      assert tool_call.type == "function"
      assert tool_call.id == "call_123"
      assert tool_call.function.name == "search"
      assert tool_call.function.arguments == ~s({"query": "elixir"})
    end

    test "creates a tool call without id" do
      tool_call = Types.tool_call("calculate", ~s({"x": 1}))

      assert tool_call.type == "function"
      assert tool_call.id == nil
      assert tool_call.function.name == "calculate"
    end
  end

  describe "ensure_text/1" do
    test "returns string content directly" do
      assert Types.ensure_text("Hello") == "Hello"
    end

    test "extracts text from single text part list" do
      content = [Types.text_part("Hello")]

      assert Types.ensure_text(content) == "Hello"
    end

    test "extracts text from single text part map" do
      content = [%{"type" => "text", "text" => "Hello"}]

      assert Types.ensure_text(content) == "Hello"
    end

    test "raises for multimodal content" do
      content = [
        Types.text_part("Hello"),
        Types.image_part("https://example.com/img.jpg")
      ]

      assert_raise ArgumentError, ~r/Expected text content/, fn ->
        Types.ensure_text(content)
      end
    end

    test "raises for multiple text parts" do
      content = [
        Types.text_part("Hello"),
        Types.text_part("World")
      ]

      assert_raise ArgumentError, ~r/Expected text content/, fn ->
        Types.ensure_text(content)
      end
    end
  end

  describe "ensure_parts/1" do
    test "wraps string content into a single text part" do
      [part] = Types.ensure_parts("Hello")
      assert part.type == "text"
      assert part.text == "Hello"
    end

    test "accepts text and image part structs" do
      parts = [Types.text_part("Hello"), Types.image_part("https://example.com/img.jpg")]
      assert Types.ensure_parts(parts) == parts
    end

    test "accepts map content parts" do
      parts = [
        %{"type" => "text", "text" => "Hello"},
        %{"type" => "image", "image" => "https://example.com/img.jpg"}
      ]

      normalized = Types.ensure_parts(parts)
      assert [%Types.TextPart{}, %Types.ImagePart{}] = normalized
    end
  end

  describe "RenderedMessage" do
    test "creates a rendered message with all fields" do
      prefix = %{tokens: [1, 2, 3]}
      content = [%{tokens: [4, 5, 6]}]
      suffix = %{tokens: [7, 8]}

      rendered = %Types.RenderedMessage{
        prefix: prefix,
        content: content,
        suffix: suffix
      }

      assert rendered.prefix == prefix
      assert rendered.content == content
      assert rendered.suffix == suffix
    end

    test "creates a rendered message with only content" do
      content = [%{tokens: [1, 2, 3]}]

      rendered = %Types.RenderedMessage{content: content}

      assert rendered.prefix == nil
      assert rendered.content == content
      assert rendered.suffix == nil
    end
  end
end
