defmodule TinkexCookbook.Types.ModelInputTest do
  use ExUnit.Case, async: true

  alias TinkexCookbook.Types.{EncodedTextChunk, ImageChunk, ModelInput}

  describe "EncodedTextChunk" do
    test "creates chunk with tokens" do
      chunk = EncodedTextChunk.new([1, 2, 3, 4])
      assert chunk.tokens == [1, 2, 3, 4]
    end

    test "length/1 returns token count" do
      chunk = EncodedTextChunk.new([1, 2, 3])
      assert EncodedTextChunk.length(chunk) == 3
    end
  end

  describe "ImageChunk" do
    test "creates chunk with data and expected_tokens" do
      chunk = ImageChunk.new(<<1, 2, 3>>, "jpeg", 100)
      assert chunk.data == <<1, 2, 3>>
      assert chunk.format == "jpeg"
      assert chunk.expected_tokens == 100
    end

    test "length/1 returns expected_tokens" do
      chunk = ImageChunk.new(<<>>, "jpeg", 256)
      assert ImageChunk.length(chunk) == 256
    end
  end

  describe "ModelInput" do
    test "creates with list of chunks" do
      chunk1 = EncodedTextChunk.new([1, 2, 3])
      chunk2 = EncodedTextChunk.new([4, 5])
      model_input = ModelInput.new([chunk1, chunk2])

      assert length(model_input.chunks) == 2
    end

    test "length/1 returns total length across all chunks" do
      chunk1 = EncodedTextChunk.new([1, 2, 3])
      chunk2 = EncodedTextChunk.new([4, 5])
      model_input = ModelInput.new([chunk1, chunk2])

      assert ModelInput.length(model_input) == 5
    end

    test "from_ints/1 creates single-chunk ModelInput" do
      model_input = ModelInput.from_ints([1, 2, 3, 4, 5])

      assert length(model_input.chunks) == 1
      assert ModelInput.length(model_input) == 5
    end

    test "empty/0 creates a ModelInput with no chunks" do
      model_input = ModelInput.empty()
      assert model_input.chunks == []
    end

    test "append_int/2 appends tokens to the last text chunk" do
      model_input = ModelInput.from_ints([1, 2])
      appended = ModelInput.append_int(model_input, 3)

      assert ModelInput.all_tokens(appended) == [1, 2, 3]
    end

    test "all_tokens/1 extracts all tokens from text chunks" do
      chunk1 = EncodedTextChunk.new([1, 2, 3])
      chunk2 = EncodedTextChunk.new([4, 5])
      model_input = ModelInput.new([chunk1, chunk2])

      assert ModelInput.all_tokens(model_input) == [1, 2, 3, 4, 5]
    end

    test "all_tokens/1 returns placeholder zeros for image chunks" do
      text_chunk = EncodedTextChunk.new([1, 2, 3])
      image_chunk = ImageChunk.new(<<>>, "jpeg", 4)
      model_input = ModelInput.new([text_chunk, image_chunk])

      tokens = ModelInput.all_tokens(model_input)
      assert tokens == [1, 2, 3, 0, 0, 0, 0]
    end
  end
end
