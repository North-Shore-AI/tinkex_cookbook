defmodule TinkexCookbook.Test.MockTokenizer do
  @moduledoc """
  Deterministic mock tokenizer for testing renderers.

  Provides predictable encoding/decoding using ASCII character codes.
  This allows tests to verify token sequences without relying on
  real tokenizers or network access.

  ## Examples

      iex> MockTokenizer.encode("Hi")
      [72, 105]

      iex> MockTokenizer.decode([72, 105])
      "Hi"
  """

  @doc """
  Encodes text into a list of token integers.

  Uses ASCII character codes for predictable, deterministic encoding.
  The `add_special_tokens` option is ignored in this mock.
  """
  @spec encode(String.t(), keyword()) :: [non_neg_integer()]
  def encode(text, _opts \\ []) do
    String.to_charlist(text)
  end

  @doc """
  Decodes a list of token integers back into text.
  """
  @spec decode([non_neg_integer()]) :: String.t()
  def decode(tokens) when is_list(tokens) do
    List.to_string(tokens)
  end

  @doc """
  Returns the BOS (beginning of sequence) token string.
  """
  @spec bos_token() :: String.t()
  def bos_token, do: "<s>"

  @doc """
  Returns the EOS (end of sequence) token string.
  """
  @spec eos_token() :: String.t()
  def eos_token, do: "</s>"
end
