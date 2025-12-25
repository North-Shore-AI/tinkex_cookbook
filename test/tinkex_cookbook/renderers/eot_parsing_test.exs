defmodule TinkexCookbook.Renderers.EotParsingTest do
  @moduledoc """
  Tests for EOT (End of Turn) token parsing across all renderers.

  Verifies:
  1. Normal case with single EOT - parses correctly with format_ok=true
  2. No EOT token - returns content with format_ok=false
  3. Double EOT token - raises RuntimeError
  """

  use ExUnit.Case, async: true

  alias TinkexCookbook.Renderers.{Llama3, Qwen3, RoleColon}

  # Test content and its mock tokens
  @test_content "53 + 18 = 71"
  @content_tokens [4331, 489, 220, 972, 284, 220, 6028]

  # EOT token IDs for each renderer family
  @llama3_eot_id 128_009
  @qwen3_eot_id 151_645

  describe "Llama3 EOT parsing" do
    setup do
      tokenizer = mock_tokenizer(@llama3_eot_id, "<|eot_id|>")
      {:ok, state} = Llama3.init(tokenizer: tokenizer)
      {:ok, state: state}
    end

    test "parses response with EOT correctly", %{state: state} do
      # Content tokens + eot_id
      tokens = @content_tokens ++ [@llama3_eot_id]

      {message, format_ok} = Llama3.parse_response(tokens, state)

      assert message.role == "assistant"
      assert message.content == @test_content
      assert format_ok == true
    end

    test "returns format_ok=false when no EOT", %{state: state} do
      # Content tokens without eot_id
      tokens = @content_tokens

      {message, format_ok} = Llama3.parse_response(tokens, state)

      assert message.role == "assistant"
      assert message.content == @test_content
      assert format_ok == false
    end

    test "raises RuntimeError on double EOT", %{state: state} do
      # Content tokens + eot_id + eot_id
      tokens = @content_tokens ++ [@llama3_eot_id, @llama3_eot_id]

      assert_raise RuntimeError, ~r/expected to split into 1 or 2 pieces/, fn ->
        Llama3.parse_response(tokens, state)
      end
    end
  end

  describe "Qwen3 EOT parsing" do
    setup do
      tokenizer = mock_tokenizer(@qwen3_eot_id, "<|im_end|>")
      {:ok, state} = Qwen3.init(tokenizer: tokenizer)
      {:ok, state: state}
    end

    test "parses response with EOT correctly", %{state: state} do
      tokens = @content_tokens ++ [@qwen3_eot_id]

      {message, format_ok} = Qwen3.parse_response(tokens, state)

      assert message.role == "assistant"
      assert message.content == @test_content
      assert format_ok == true
    end

    test "returns format_ok=false when no EOT", %{state: state} do
      tokens = @content_tokens

      {message, format_ok} = Qwen3.parse_response(tokens, state)

      assert message.role == "assistant"
      assert message.content == @test_content
      assert format_ok == false
    end

    test "raises RuntimeError on double EOT", %{state: state} do
      tokens = @content_tokens ++ [@qwen3_eot_id, @qwen3_eot_id]

      assert_raise RuntimeError, ~r/expected to split into 1 or 2 pieces/, fn ->
        Qwen3.parse_response(tokens, state)
      end
    end
  end

  describe "RoleColon EOT parsing" do
    setup do
      # RoleColon uses text-based delimiter, not token ID
      tokenizer = mock_role_colon_tokenizer()
      {:ok, state} = RoleColon.init(tokenizer: tokenizer)
      {:ok, state: state}
    end

    test "parses response with delimiter correctly", %{state: state} do
      # RoleColon uses a different parsing mechanism - uses string delimiter
      tokens = @content_tokens

      {message, _format_ok} = RoleColon.parse_response(tokens, state)

      assert message.role == "assistant"
    end
  end

  # Creates a mock tokenizer that decodes content tokens to the test content
  defp mock_tokenizer(eot_id, eot_string) do
    %{
      encode: fn text, _opts ->
        cond do
          text == eot_string -> [eot_id]
          text == @test_content -> @content_tokens
          true -> String.to_charlist(text) |> Enum.map(& &1)
        end
      end,
      decode: fn tokens ->
        # Filter out EOT tokens and decode
        content_only = Enum.reject(tokens, &(&1 == eot_id))

        if content_only == @content_tokens do
          @test_content
        else
          # Fallback for other tokens
          Enum.map_join(content_only, "", fn t -> "#{t}" end)
        end
      end
    }
  end

  # RoleColon uses text-based parsing, not token-based
  defp mock_role_colon_tokenizer do
    %{
      encode: fn text, _opts ->
        cond do
          text == @test_content -> @content_tokens
          # Mock delimiter tokens
          text == "\n\nUser:" -> [10, 10, 12_336]
          true -> String.to_charlist(text) |> Enum.map(& &1)
        end
      end,
      decode: fn tokens ->
        if tokens == @content_tokens do
          @test_content
        else
          Enum.map_join(tokens, "", fn t -> "#{t}" end)
        end
      end
    }
  end
end
