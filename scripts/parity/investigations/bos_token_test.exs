#!/usr/bin/env elixir
# Script to verify BOS token behavior in Elixir tokenizer
#
# This script demonstrates that the tokenizer adds BOS tokens when
# add_special_tokens is not explicitly set to false.
#
# Run with: elixir scripts/parity/investigations/bos_token_test.exs

Mix.install([
  {:tokenizers, "~> 0.5"}
])

# Load Llama 3 tokenizer
IO.puts("Loading Llama 3 tokenizer...")
{:ok, tokenizer} = Tokenizers.Tokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

# Test text (role header from Llama3 renderer)
test_text = "<|start_header_id|>user<|end_header_id|>\n\n"

# Token 128000 is <|begin_of_text|> (BOS)
bos_token = 128_000

IO.puts("\n=== Test 1: Default encoding (add_special_tokens not specified) ===")
{:ok, encoding} = Tokenizers.Tokenizer.encode(tokenizer, test_text)
tokens = Tokenizers.Encoding.get_ids(encoding)
IO.puts("Tokens: #{inspect(tokens)}")
IO.puts("First token is BOS (#{bos_token})? #{List.first(tokens) == bos_token}")

IO.puts("\n=== Test 2: With add_special_tokens: true ===")
{:ok, encoding} = Tokenizers.Tokenizer.encode(tokenizer, test_text, add_special_tokens: true)
tokens = Tokenizers.Encoding.get_ids(encoding)
IO.puts("Tokens: #{inspect(tokens)}")
IO.puts("First token is BOS (#{bos_token})? #{List.first(tokens) == bos_token}")

IO.puts("\n=== Test 3: With add_special_tokens: false ===")
{:ok, encoding} = Tokenizers.Tokenizer.encode(tokenizer, test_text, add_special_tokens: false)
tokens = Tokenizers.Encoding.get_ids(encoding)
IO.puts("Tokens: #{inspect(tokens)}")
IO.puts("First token is BOS (#{bos_token})? #{List.first(tokens) == bos_token}")

IO.puts("\n=== Test 4: BOS token string encoding ===")
bos_str = "<|begin_of_text|>"
{:ok, encoding} = Tokenizers.Tokenizer.encode(tokenizer, bos_str, add_special_tokens: false)
tokens = Tokenizers.Encoding.get_ids(encoding)
IO.puts("BOS string tokens: #{inspect(tokens)}")
IO.puts("Length: #{length(tokens)} (expected: 1)")

IO.puts("\n=== Summary ===")

IO.puts("""
If Test 1 and Test 2 show BOS at the start, but Test 3 does NOT,
then the fix is to pass add_special_tokens: false to the tokenizer.

The current Elixir tokenizer wrapper ignores the add_special_tokens option,
causing unwanted BOS tokens to be added to every encoded chunk.
""")
