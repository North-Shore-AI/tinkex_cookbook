defmodule TinkexCookbook.Runtime.Manifests do
  @moduledoc """
  Environment-specific port wiring manifests.

  Manifests are named configurations that wire ports to specific
  adapter implementations. This enables environment-specific behavior
  without changing recipe code.

  ## Available Manifests

  - `:default` - Uses production-ready adapters
  - `:local` - Local development (local blob storage)
  - `:prod` - Production environment
  - `:test` - All noop adapters for testing
  """

  alias TinkexCookbook.Adapters

  @doc """
  Returns default port implementations.

  These are the baseline adapters used when no manifest is specified.
  """
  @spec defaults() :: map()
  def defaults do
    %{
      "training_client" => Adapters.TrainingClient.Tinkex,
      "dataset_store" => Adapters.DatasetStore.HfDatasets,
      "blob_store" => Adapters.BlobStore.Local,
      "hub_client" => Adapters.HubClient.HfHub,
      "llm_client" => Adapters.LLMClient.Noop,
      "embedding_client" => Adapters.EmbeddingClient.Noop,
      "vector_store" => Adapters.VectorStore.Noop
    }
  end

  @doc """
  Get a named manifest's port overrides.
  """
  @spec get(atom()) :: map()
  def get(:default), do: %{}

  def get(:local) do
    %{
      "blob_store" => Adapters.BlobStore.Local
    }
  end

  def get(:dev) do
    %{
      "blob_store" => Adapters.BlobStore.Local
    }
  end

  def get(:prod) do
    %{
      "blob_store" => Adapters.BlobStore.S3,
      "llm_client" => Adapters.LLMClient.ClaudeAgent
    }
  end

  def get(:test) do
    %{
      "training_client" => Adapters.TrainingClient.Noop,
      "dataset_store" => Adapters.DatasetStore.Noop,
      "blob_store" => Adapters.BlobStore.Noop,
      "hub_client" => Adapters.HubClient.Noop,
      "llm_client" => Adapters.LLMClient.Noop,
      "embedding_client" => Adapters.EmbeddingClient.Noop,
      "vector_store" => Adapters.VectorStore.Noop
    }
  end

  def get(name) do
    raise ArgumentError, "Unknown manifest: #{inspect(name)}"
  end
end
