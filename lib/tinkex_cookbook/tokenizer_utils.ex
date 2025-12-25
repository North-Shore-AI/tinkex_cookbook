defmodule TinkexCookbook.TokenizerUtils do
  @moduledoc """
  Tokenizer helpers mirroring Python tokenizer_utils.get_tokenizer.
  """

  @spec get_tokenizer(String.t(), keyword()) :: {:ok, term()} | {:error, term()}
  def get_tokenizer(model_name, opts \\ []) when is_binary(model_name) do
    tokenizer_id =
      Tinkex.Tokenizer.get_tokenizer_id(model_name, Keyword.get(opts, :training_client), opts)

    Tinkex.Tokenizer.get_or_load_tokenizer(tokenizer_id, opts)
  end
end
