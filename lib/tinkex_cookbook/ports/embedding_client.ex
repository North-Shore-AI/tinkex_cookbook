defmodule TinkexCookbook.Ports.EmbeddingClient do
  @moduledoc """
  Port for embedding generation services.
  """

  alias TinkexCookbook.Ports

  @type adapter_opts :: keyword()
  @type embedding :: [number()]

  @callback embed_texts(adapter_opts(), [String.t()], keyword()) ::
              {:ok, [embedding()]} | {:error, term()}

  @spec embed_texts(Ports.t(), [String.t()], keyword()) ::
          {:ok, [embedding()]} | {:error, term()}
  def embed_texts(%Ports{} = ports, texts, opts \\ []) do
    {module, adapter_opts} = Ports.resolve(ports, :embedding_client)
    module.embed_texts(adapter_opts, texts, opts)
  end
end
