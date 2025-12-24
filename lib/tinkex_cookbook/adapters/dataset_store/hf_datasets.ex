defmodule TinkexCookbook.Adapters.DatasetStore.HfDatasets do
  @moduledoc """
  Adapter for HfDatasetsEx dataset operations.
  """

  @behaviour TinkexCookbook.Ports.DatasetStore

  alias TinkexCookbook.Ports.Error

  @impl true
  def load_dataset(_opts, repo_id, opts) do
    HfDatasetsEx.load_dataset(repo_id, opts)
  rescue
    exception ->
      {:error, Error.new(:dataset_store, __MODULE__, Exception.message(exception), exception)}
  end

  @impl true
  def get_split(_opts, %HfDatasetsEx.DatasetDict{} = dataset_dict, split) do
    case HfDatasetsEx.DatasetDict.get(dataset_dict, split) do
      nil ->
        {:error,
         Error.new(
           :dataset_store,
           __MODULE__,
           "Split not found: #{inspect(split)}",
           split
         )}

      dataset ->
        {:ok, dataset}
    end
  end

  def get_split(_opts, _dataset, split) do
    {:error, Error.new(:dataset_store, __MODULE__, "Dataset does not support splits", split)}
  end

  @impl true
  def shuffle(_opts, dataset, opts) do
    dataset
    |> dispatch!(:shuffle, [opts])
    |> ok()
  rescue
    exception ->
      {:error, Error.new(:dataset_store, __MODULE__, Exception.message(exception), exception)}
  end

  @impl true
  def take(_opts, dataset, count) do
    dataset
    |> dispatch!(:take, [count])
    |> ok()
  rescue
    exception ->
      {:error, Error.new(:dataset_store, __MODULE__, Exception.message(exception), exception)}
  end

  @impl true
  def skip(_opts, dataset, count) do
    dataset
    |> dispatch!(:skip, [count])
    |> ok()
  rescue
    exception ->
      {:error, Error.new(:dataset_store, __MODULE__, Exception.message(exception), exception)}
  end

  @impl true
  def select(_opts, dataset, selection) do
    dataset
    |> dispatch!(:select, [selection])
    |> ok()
  rescue
    exception ->
      {:error, Error.new(:dataset_store, __MODULE__, Exception.message(exception), exception)}
  end

  @impl true
  def to_list(_opts, dataset) do
    dataset
    |> dispatch!(:to_list, [])
    |> ok()
  rescue
    exception ->
      {:error, Error.new(:dataset_store, __MODULE__, Exception.message(exception), exception)}
  end

  defp ok(result), do: {:ok, result}

  defp dispatch!(%HfDatasetsEx.Dataset{} = dataset, fun, args) do
    apply(HfDatasetsEx.Dataset, fun, [dataset | args])
  end

  defp dispatch!(%HfDatasetsEx.IterableDataset{} = dataset, fun, args) do
    apply(HfDatasetsEx.IterableDataset, fun, [dataset | args])
  end

  defp dispatch!(%HfDatasetsEx.DatasetDict{} = dataset_dict, fun, args) do
    apply(HfDatasetsEx.DatasetDict, fun, [dataset_dict | args])
  end

  defp dispatch!(_dataset, fun, _args) do
    raise ArgumentError, "Unsupported dataset type for #{fun}"
  end
end
