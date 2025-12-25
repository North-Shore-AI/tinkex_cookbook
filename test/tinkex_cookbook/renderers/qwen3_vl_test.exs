defmodule TinkexCookbook.Renderers.Qwen3VLTest do
  use ExUnit.Case, async: true

  alias TinkexCookbook.Renderers.{Qwen3VL, Qwen3VLInstruct}
  alias TinkexCookbook.Renderers.Types
  alias TinkexCookbook.Test.MockTokenizer
  alias TinkexCookbook.Types.{EncodedTextChunk, ImageChunk}
  alias Vix.Vips.Image, as: VipsImage

  defmodule MockImageProcessor do
    def merge_size, do: 2
    def get_number_of_image_patches(height, width, _images_kwargs), do: height * width * 4
  end

  setup do
    image = VipsImage.build_image!(2, 3, [0, 0, 0])
    {:ok, bytes} = VipsImage.write_to_buffer(image, ".jpg")
    {:ok, state} = Qwen3VL.init(tokenizer: MockTokenizer, image_processor: MockImageProcessor)

    %{state: state, image_bytes: bytes}
  end

  test "render_message wraps image parts with vision markers", %{state: state, image_bytes: bytes} do
    message = Types.message("user", [Types.image_part(bytes)])

    {rendered, _state} = Qwen3VL.render_message(0, message, false, state)

    assert [
             %EncodedTextChunk{} = vision_start,
             %ImageChunk{} = image_chunk,
             %EncodedTextChunk{} = vision_end,
             %EncodedTextChunk{} = im_end
           ] = rendered.content

    assert MockTokenizer.decode(vision_start.tokens) == "<|vision_start|>"
    assert MockTokenizer.decode(vision_end.tokens) == "<|vision_end|>"
    assert MockTokenizer.decode(im_end.tokens) == "<|im_end|>"
    assert image_chunk.format == "jpeg"
    assert image_chunk.expected_tokens == 6
  end

  test "assistant without think adds think prefix", %{state: state} do
    message = Types.message("assistant", [Types.text_part("Hello")])

    {rendered, _state} = Qwen3VL.render_message(0, message, false, state)

    prefix_text = MockTokenizer.decode(rendered.prefix.tokens)
    assert String.contains?(prefix_text, "<think>\n")
  end

  test "instruct renderer does not add think prefix", %{state: state} do
    message = Types.message("assistant", [Types.text_part("Hello")])

    {rendered, _state} = Qwen3VLInstruct.render_message(0, message, false, state)

    prefix_text = MockTokenizer.decode(rendered.prefix.tokens)
    refute String.contains?(prefix_text, "<think>\n")
  end
end
