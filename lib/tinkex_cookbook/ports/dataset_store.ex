defmodule TinkexCookbook.Ports.DatasetStore do
  @moduledoc """
  Port for dataset loading and common dataset operations.
  """

  alias TinkexCookbook.Ports

  @type adapter_opts :: keyword()
  @type dataset :: term()

  @callback load_dataset(adapter_opts(), String.t(), keyword()) ::
              {:ok, dataset()} | {:error, term()}

  @callback get_split(adapter_opts(), dataset(), String.t() | atom()) ::
              {:ok, dataset()} | {:error, term()}

  @callback shuffle(adapter_opts(), dataset(), keyword()) ::
              {:ok, dataset()} | {:error, term()}

  @callback take(adapter_opts(), dataset(), non_neg_integer()) ::
              {:ok, dataset()} | {:error, term()}

  @callback skip(adapter_opts(), dataset(), non_neg_integer()) ::
              {:ok, dataset()} | {:error, term()}

  @callback select(adapter_opts(), dataset(), Range.t() | [non_neg_integer()]) ::
              {:ok, dataset()} | {:error, term()}

  @callback to_list(adapter_opts(), dataset()) ::
              {:ok, [map()]} | {:error, term()}

  @spec load_dataset(Ports.t(), String.t(), keyword()) ::
          {:ok, dataset()} | {:error, term()}
  def load_dataset(%Ports{} = ports, repo_id, opts \\ []) do
    {module, adapter_opts} = Ports.resolve(ports, :dataset_store)
    module.load_dataset(adapter_opts, repo_id, opts)
  end

  @spec get_split(Ports.t(), dataset(), String.t() | atom()) ::
          {:ok, dataset()} | {:error, term()}
  def get_split(%Ports{} = ports, dataset_dict, split) do
    {module, adapter_opts} = Ports.resolve(ports, :dataset_store)
    module.get_split(adapter_opts, dataset_dict, split)
  end

  @spec shuffle(Ports.t(), dataset(), keyword()) ::
          {:ok, dataset()} | {:error, term()}
  def shuffle(%Ports{} = ports, dataset, opts \\ []) do
    {module, adapter_opts} = Ports.resolve(ports, :dataset_store)
    module.shuffle(adapter_opts, dataset, opts)
  end

  @spec take(Ports.t(), dataset(), non_neg_integer()) ::
          {:ok, dataset()} | {:error, term()}
  def take(%Ports{} = ports, dataset, count) do
    {module, adapter_opts} = Ports.resolve(ports, :dataset_store)
    module.take(adapter_opts, dataset, count)
  end

  @spec skip(Ports.t(), dataset(), non_neg_integer()) ::
          {:ok, dataset()} | {:error, term()}
  def skip(%Ports{} = ports, dataset, count) do
    {module, adapter_opts} = Ports.resolve(ports, :dataset_store)
    module.skip(adapter_opts, dataset, count)
  end

  @spec select(Ports.t(), dataset(), Range.t() | [non_neg_integer()]) ::
          {:ok, dataset()} | {:error, term()}
  def select(%Ports{} = ports, dataset, selection) do
    {module, adapter_opts} = Ports.resolve(ports, :dataset_store)
    module.select(adapter_opts, dataset, selection)
  end

  @spec to_list(Ports.t(), dataset()) :: {:ok, [map()]} | {:error, term()}
  def to_list(%Ports{} = ports, dataset) do
    {module, adapter_opts} = Ports.resolve(ports, :dataset_store)
    module.to_list(adapter_opts, dataset)
  end
end
