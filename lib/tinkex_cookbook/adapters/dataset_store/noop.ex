defmodule TinkexCookbook.Adapters.DatasetStore.Noop do
  @moduledoc """
  No-op adapter for dataset store operations.
  """

  @behaviour TinkexCookbook.Ports.DatasetStore

  alias TinkexCookbook.Ports.Error

  defp error do
    Error.new(:dataset_store, __MODULE__, "Dataset adapter is not configured")
  end

  @impl true
  def load_dataset(_opts, _repo_id, _opts2), do: {:error, error()}

  @impl true
  def get_split(_opts, _dataset, _split), do: {:error, error()}

  @impl true
  def shuffle(_opts, _dataset, _opts2), do: {:error, error()}

  @impl true
  def take(_opts, _dataset, _count), do: {:error, error()}

  @impl true
  def skip(_opts, _dataset, _count), do: {:error, error()}

  @impl true
  def select(_opts, _dataset, _selection), do: {:error, error()}

  @impl true
  def to_list(_opts, _dataset), do: {:error, error()}
end
