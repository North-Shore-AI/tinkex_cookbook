defmodule TinkexCookbook.Ports.BlobStore do
  @moduledoc """
  Port for file/blob access (local or remote).
  """

  alias TinkexCookbook.Ports

  @type adapter_opts :: keyword()
  @type path :: String.t()

  @callback read(adapter_opts(), path()) :: {:ok, binary()} | {:error, term()}
  @callback stream(adapter_opts(), path()) :: {:ok, Enumerable.t()} | {:error, term()}
  @callback write(adapter_opts(), path(), iodata()) :: :ok | {:error, term()}
  @callback exists?(adapter_opts(), path()) :: boolean()

  @spec read(Ports.t(), path()) :: {:ok, binary()} | {:error, term()}
  def read(%Ports{} = ports, path) do
    {module, adapter_opts} = Ports.resolve(ports, :blob_store)
    module.read(adapter_opts, path)
  end

  @spec stream(Ports.t(), path()) :: {:ok, Enumerable.t()} | {:error, term()}
  def stream(%Ports{} = ports, path) do
    {module, adapter_opts} = Ports.resolve(ports, :blob_store)
    module.stream(adapter_opts, path)
  end

  @spec write(Ports.t(), path(), iodata()) :: :ok | {:error, term()}
  def write(%Ports{} = ports, path, data) do
    {module, adapter_opts} = Ports.resolve(ports, :blob_store)
    module.write(adapter_opts, path, data)
  end

  @spec exists?(Ports.t(), path()) :: boolean()
  def exists?(%Ports{} = ports, path) do
    {module, adapter_opts} = Ports.resolve(ports, :blob_store)
    module.exists?(adapter_opts, path)
  end
end
