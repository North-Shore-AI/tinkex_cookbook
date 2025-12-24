#!/usr/bin/env python3
"""
Script to verify BOS token behavior in Python tokenizer.

This script demonstrates that Python explicitly uses add_special_tokens=False
to prevent duplicate BOS tokens.

Run with: python scripts/parity/investigations/bos_token_test.py
"""

from transformers import AutoTokenizer

print("Loading Llama 3 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

# Test text (role header from Llama3 renderer)
test_text = "<|start_header_id|>user<|end_header_id|>\n\n"

# Token 128000 is <|begin_of_text|> (BOS)
bos_token = 128000

print("\n=== Test 1: Default encoding ===")
tokens = tokenizer.encode(test_text)
print(f"Tokens: {tokens}")
print(f"First token is BOS ({bos_token})? {tokens[0] == bos_token if tokens else False}")

print("\n=== Test 2: With add_special_tokens=True ===")
tokens = tokenizer.encode(test_text, add_special_tokens=True)
print(f"Tokens: {tokens}")
print(f"First token is BOS ({bos_token})? {tokens[0] == bos_token if tokens else False}")

print("\n=== Test 3: With add_special_tokens=False ===")
tokens = tokenizer.encode(test_text, add_special_tokens=False)
print(f"Tokens: {tokens}")
print(f"First token is BOS ({bos_token})? {tokens[0] == bos_token if tokens else False}")

print("\n=== Test 4: BOS token string encoding ===")
bos_str = "<|begin_of_text|>"
tokens = tokenizer.encode(bos_str, add_special_tokens=False)
print(f"BOS string tokens: {tokens}")
print(f"Length: {len(tokens)} (expected: 1)")

print("\n=== Python Renderer BOS usage ===")
print("""
In tinker_cookbook/renderers.py, Llama3Renderer uses:

    @property
    def _bos_tokens(self) -> list[int]:
        return self.tokenizer.encode("<|begin_of_text|>", add_special_tokens=False)

This ensures only ONE BOS token at the start of the sequence.
""")

print("=== Summary ===")
print("""
Python explicitly uses add_special_tokens=False throughout the renderer.
The Elixir tokenizer wrapper must do the same to achieve parity.
""")
