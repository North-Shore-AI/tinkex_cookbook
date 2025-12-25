defmodule TinkexCookbook.Renderers.HfParityTest do
  @moduledoc """
  Tests that verify Elixir renderers produce identical tokens to HuggingFace.

  These tests use pre-generated fixtures from Python's transformers library.
  To regenerate fixtures:

      python scripts/generate_hf_parity_fixtures.py

  The fixtures contain expected token sequences from HF's apply_chat_template().
  """

  use ExUnit.Case, async: true

  alias TinkexCookbook.Renderers.{Llama3, Qwen3, Renderer, Types}

  @fixtures_path "test/fixtures/hf_parity.json"

  # These tests require real tokenizers (not available in CI by default)
  # Run with: mix test --include integration
  @moduletag :integration
  @moduletag :hf_parity

  setup_all do
    if File.exists?(@fixtures_path) do
      fixtures = @fixtures_path |> File.read!() |> Jason.decode!()
      {:ok, fixtures: fixtures}
    else
      IO.puts("\nWARNING: #{@fixtures_path} not found. Run:")
      IO.puts("  python scripts/generate_hf_parity_fixtures.py\n")
      :ok
    end
  end

  describe "Llama3 renderer" do
    @tag :llama3
    test "generation matches HF chat template", context do
      fixtures = Map.get(context, :fixtures, %{})
      raw_fixture = fixtures["meta-llama__Llama-3.2-1B-Instruct"]

      case check_fixture(raw_fixture, "meta-llama/Llama-3.2-1B-Instruct") do
        :skip ->
          :ok

        {:ok, fixture} ->
          tokenizer = mock_llama3_tokenizer()
          {:ok, state} = Llama3.init(tokenizer: tokenizer)

          messages =
            fixture["generation"]["augmented_messages"]
            |> Enum.map(&to_message/1)

          # Build generation prompt (module, messages, role, prefill, state)
          model_input =
            Renderer.build_generation_prompt(Llama3, messages, "assistant", nil, state)

          actual_tokens = model_input_to_tokens(model_input)

          expected_tokens = fixture["generation"]["expected_tokens"]

          assert actual_tokens == expected_tokens,
                 """
                 Token mismatch for Llama3 generation prompt.

                 Expected (#{length(expected_tokens)} tokens):
                 #{inspect(expected_tokens)}

                 Actual (#{length(actual_tokens)} tokens):
                 #{inspect(actual_tokens)}

                 Expected decoded:
                 #{fixture["generation"]["decoded"]}
                 """
      end
    end

    @tag :llama3
    test "supervised example matches HF chat template", context do
      fixtures = Map.get(context, :fixtures, %{})
      raw_fixture = fixtures["meta-llama__Llama-3.2-1B-Instruct"]

      case check_fixture(raw_fixture, "meta-llama/Llama-3.2-1B-Instruct") do
        :skip ->
          :ok

        {:ok, fixture} ->
          tokenizer = mock_llama3_tokenizer()
          {:ok, state} = Llama3.init(tokenizer: tokenizer)

          messages =
            fixture["supervised"]["augmented_messages"]
            |> Enum.map(&to_message/1)

          # Build supervised example
          {model_input, _weights} =
            Renderer.build_supervised_example(
              Llama3,
              messages,
              state,
              "all_assistant_messages"
            )

          actual_tokens = model_input_to_tokens(model_input)

          expected_tokens = fixture["supervised"]["expected_tokens"]

          assert actual_tokens == expected_tokens,
                 """
                 Token mismatch for Llama3 supervised example.

                 Expected (#{length(expected_tokens)} tokens):
                 #{inspect(expected_tokens)}

                 Actual (#{length(actual_tokens)} tokens):
                 #{inspect(actual_tokens)}
                 """
      end
    end
  end

  describe "Qwen3 renderer" do
    @tag :qwen3
    test "generation matches HF chat template", context do
      fixtures = Map.get(context, :fixtures, %{})
      raw_fixture = fixtures["Qwen__Qwen3-0.6B"]

      case check_fixture(raw_fixture, "Qwen/Qwen3-0.6B") do
        :skip ->
          :ok

        {:ok, fixture} ->
          tokenizer = mock_qwen3_tokenizer()
          {:ok, state} = Qwen3.init(tokenizer: tokenizer)

          messages =
            fixture["generation"]["messages"]
            |> Enum.map(&to_message/1)

          # Build generation prompt (module, messages, role, prefill, state)
          model_input = Renderer.build_generation_prompt(Qwen3, messages, "assistant", nil, state)
          actual_tokens = model_input_to_tokens(model_input)

          expected_tokens = fixture["generation"]["expected_tokens"]

          assert actual_tokens == expected_tokens,
                 """
                 Token mismatch for Qwen3 generation prompt.

                 Expected: #{inspect(expected_tokens)}
                 Actual: #{inspect(actual_tokens)}
                 """
      end
    end
  end

  # Helpers

  # Returns :skip if fixture is missing, allowing tests to gracefully skip
  defp check_fixture(nil, model_name) do
    IO.puts("Skipping: Fixture not found for #{model_name}")
    :skip
  end

  defp check_fixture(fixture, _model_name), do: {:ok, fixture}

  defp to_message(%{"role" => role, "content" => content}) do
    %Types.Message{role: role, content: content}
  end

  defp model_input_to_tokens(model_input) do
    model_input.chunks
    |> Enum.flat_map(fn
      %TinkexCookbook.Types.EncodedTextChunk{tokens: tokens} -> tokens
      _ -> []
    end)
  end

  # Mock tokenizers that return the same tokens as HF
  # These need to be replaced with actual tokenizer calls for real testing

  defp mock_llama3_tokenizer do
    # For now, return a mock that will fail - users need real tokenizer
    %{
      encode: fn _text, _opts ->
        raise "Real tokenizer needed. Load with: Tokenizers.Tokenizer.from_pretrained(\"meta-llama/Llama-3.2-1B-Instruct\")"
      end,
      decode: fn _tokens ->
        raise "Real tokenizer needed"
      end
    }
  end

  defp mock_qwen3_tokenizer do
    %{
      encode: fn _text, _opts ->
        raise "Real tokenizer needed. Load with: Tokenizers.Tokenizer.from_pretrained(\"Qwen/Qwen3-0.6B\")"
      end,
      decode: fn _tokens ->
        raise "Real tokenizer needed"
      end
    }
  end
end
