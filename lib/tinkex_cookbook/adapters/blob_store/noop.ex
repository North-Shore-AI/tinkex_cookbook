defmodule TinkexCookbook.Adapters.BlobStore.Noop do
  @moduledoc """
  No-op adapter for blob access.
  """

  @behaviour CrucibleTrain.Ports.BlobStore

  alias CrucibleTrain.Ports.Error

  defp error do
    Error.new(:blob_store, __MODULE__, "Blob store adapter is not configured")
  end

  @impl true
  def read(_opts, _path), do: {:error, error()}

  @impl true
  def stream(_opts, _path), do: {:error, error()}

  @impl true
  def write(_opts, _path, _data), do: {:error, error()}

  @impl true
  def exists?(_opts, _path), do: false
end
