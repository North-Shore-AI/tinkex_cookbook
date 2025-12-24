defmodule TinkexCookbook.Ports.VectorStore do
  @moduledoc """
  Port for vector database operations (e.g., ChromaDB).
  """

  alias TinkexCookbook.Ports

  @type adapter_opts :: keyword()
  @type collection :: term()
  @type embeddings :: [[number()]]
  @type query_opts :: keyword()

  @callback create_collection(adapter_opts(), String.t(), map()) ::
              {:ok, collection()} | {:error, term()}

  @callback get_collection(adapter_opts(), String.t()) ::
              {:ok, collection()} | {:error, term()}

  @callback get_or_create_collection(adapter_opts(), String.t(), map()) ::
              {:ok, collection()} | {:error, term()}

  @callback delete_collection(adapter_opts(), collection() | String.t()) ::
              :ok | {:error, term()}

  @callback add(adapter_opts(), collection(), map()) :: :ok | {:error, term()}

  @callback query(adapter_opts(), collection(), embeddings(), query_opts()) ::
              {:ok, term()} | {:error, term()}

  @spec create_collection(Ports.t(), String.t(), map()) ::
          {:ok, collection()} | {:error, term()}
  def create_collection(%Ports{} = ports, name, metadata \\ %{}) do
    {module, opts} = Ports.resolve(ports, :vector_store)
    module.create_collection(opts, name, metadata)
  end

  @spec get_collection(Ports.t(), String.t()) ::
          {:ok, collection()} | {:error, term()}
  def get_collection(%Ports{} = ports, name) do
    {module, opts} = Ports.resolve(ports, :vector_store)
    module.get_collection(opts, name)
  end

  @spec get_or_create_collection(Ports.t(), String.t(), map()) ::
          {:ok, collection()} | {:error, term()}
  def get_or_create_collection(%Ports{} = ports, name, metadata \\ %{}) do
    {module, opts} = Ports.resolve(ports, :vector_store)
    module.get_or_create_collection(opts, name, metadata)
  end

  @spec delete_collection(Ports.t(), collection() | String.t()) :: :ok | {:error, term()}
  def delete_collection(%Ports{} = ports, collection_or_name) do
    {module, opts} = Ports.resolve(ports, :vector_store)
    module.delete_collection(opts, collection_or_name)
  end

  @spec add(Ports.t(), collection(), map()) :: :ok | {:error, term()}
  def add(%Ports{} = ports, collection, payload) do
    {module, opts} = Ports.resolve(ports, :vector_store)
    module.add(opts, collection, payload)
  end

  @spec query(Ports.t(), collection(), embeddings(), query_opts()) ::
          {:ok, term()} | {:error, term()}
  def query(%Ports{} = ports, collection, embeddings, opts \\ []) do
    {module, adapter_opts} = Ports.resolve(ports, :vector_store)
    module.query(adapter_opts, collection, embeddings, opts)
  end
end
