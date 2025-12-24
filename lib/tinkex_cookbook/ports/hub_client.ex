defmodule TinkexCookbook.Ports.HubClient do
  @moduledoc """
  Port for HuggingFace Hub operations.
  """

  alias TinkexCookbook.Ports

  @type adapter_opts :: keyword()

  @callback download(adapter_opts(), keyword()) :: {:ok, Path.t()} | {:error, term()}
  @callback snapshot(adapter_opts(), keyword()) :: {:ok, Path.t()} | {:error, term()}
  @callback list_files(adapter_opts(), String.t(), keyword()) ::
              {:ok, [String.t()]} | {:error, term()}

  @spec download(Ports.t(), keyword()) :: {:ok, Path.t()} | {:error, term()}
  def download(%Ports{} = ports, opts) do
    {module, adapter_opts} = Ports.resolve(ports, :hub_client)
    module.download(adapter_opts, opts)
  end

  @spec snapshot(Ports.t(), keyword()) :: {:ok, Path.t()} | {:error, term()}
  def snapshot(%Ports{} = ports, opts) do
    {module, adapter_opts} = Ports.resolve(ports, :hub_client)
    module.snapshot(adapter_opts, opts)
  end

  @spec list_files(Ports.t(), String.t(), keyword()) :: {:ok, [String.t()]} | {:error, term()}
  def list_files(%Ports{} = ports, repo_id, opts \\ []) do
    {module, adapter_opts} = Ports.resolve(ports, :hub_client)
    module.list_files(adapter_opts, repo_id, opts)
  end
end
