defmodule TinkexCookbook.Adapters.EmbeddingClient.Noop do
  @moduledoc """
  No-op adapter for embedding generation.
  """

  @behaviour TinkexCookbook.Ports.EmbeddingClient

  alias TinkexCookbook.Ports.Error

  @impl true
  def embed_texts(_opts, _texts, _opts2) do
    {:error, Error.new(:embedding_client, __MODULE__, "Embedding adapter is not configured")}
  end
end
