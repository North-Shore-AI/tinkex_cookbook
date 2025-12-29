defmodule TinkexCookbook.Adapters.HubClient.HfHub do
  @moduledoc """
  Adapter for HuggingFace Hub operations via HfHub.
  """

  @behaviour CrucibleTrain.Ports.HubClient

  alias CrucibleTrain.Ports.Error

  @impl true
  def download(adapter_opts, opts) do
    merged = Keyword.merge(adapter_opts, opts)

    HfHub.Download.hf_hub_download(merged)
  rescue
    exception ->
      {:error, Error.new(:hub_client, __MODULE__, Exception.message(exception), exception)}
  end

  @impl true
  def snapshot(adapter_opts, opts) do
    merged = Keyword.merge(adapter_opts, opts)

    HfHub.Download.snapshot_download(merged)
  rescue
    exception ->
      {:error, Error.new(:hub_client, __MODULE__, Exception.message(exception), exception)}
  end

  @impl true
  def list_files(adapter_opts, repo_id, opts) do
    merged = Keyword.merge(adapter_opts, opts)

    HfHub.Api.list_files(repo_id, merged)
  rescue
    exception ->
      {:error, Error.new(:hub_client, __MODULE__, Exception.message(exception), exception)}
  end
end
