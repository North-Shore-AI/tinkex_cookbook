defmodule TinkexCookbook.Runtime.Manifests do
  @moduledoc """
  Environment-specific port wiring manifests.

  Manifests are named configurations that wire ports to specific
  adapter implementations. This enables environment-specific behavior
  without changing recipe code.

  NOTE: This module is retained for backward compatibility. Adapters
  now live in `crucible_kitchen` (or in your application). Prefer
  passing adapter maps directly when invoking CrucibleKitchen.

  ## Available Manifests

  - `:default` - Uses production-ready adapters
  - `:local` - Local development (local blob storage)
  - `:prod` - Production environment
  - `:test` - All noop adapters for testing
  """

  alias CrucibleKitchen.Adapters, as: KitchenAdapters
  alias CrucibleTrain.Adapters, as: TrainAdapters

  @doc """
  Returns default port implementations.

  These are the baseline adapters used when no manifest is specified.
  """
  @spec defaults() :: map()
  def defaults do
    %{
      training_client: KitchenAdapters.Tinkex.TrainingClient,
      dataset_store: KitchenAdapters.HfDatasets.DatasetStore,
      blob_store: KitchenAdapters.Noop.BlobStore,
      hub_client: KitchenAdapters.HfHub.HubClient,
      llm_client: TrainAdapters.Noop.LLMClient,
      embedding_client: TrainAdapters.Noop.EmbeddingClient,
      vector_store: TrainAdapters.Noop.VectorStore
    }
  end

  @doc """
  Get a named manifest's port overrides.
  """
  @spec get(atom()) :: map()
  def get(:default), do: %{}

  def get(:local), do: %{}

  def get(:dev), do: %{}

  def get(:prod), do: %{}

  def get(:test) do
    %{
      training_client: KitchenAdapters.Noop.TrainingClient,
      dataset_store: KitchenAdapters.Noop.DatasetStore,
      blob_store: KitchenAdapters.Noop.BlobStore,
      hub_client: KitchenAdapters.Noop.HubClient,
      llm_client: TrainAdapters.Noop.LLMClient,
      embedding_client: TrainAdapters.Noop.EmbeddingClient,
      vector_store: TrainAdapters.Noop.VectorStore
    }
  end

  def get(name) do
    raise ArgumentError, "Unknown manifest: #{inspect(name)}"
  end
end
