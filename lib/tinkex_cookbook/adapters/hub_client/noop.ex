defmodule TinkexCookbook.Adapters.HubClient.Noop do
  @moduledoc """
  No-op adapter for HuggingFace Hub operations.
  """

  @behaviour TinkexCookbook.Ports.HubClient

  alias TinkexCookbook.Ports.Error

  defp error do
    Error.new(:hub_client, __MODULE__, "Hub adapter is not configured")
  end

  @impl true
  def download(_opts, _download_opts), do: {:error, error()}

  @impl true
  def snapshot(_opts, _snapshot_opts), do: {:error, error()}

  @impl true
  def list_files(_opts, _repo_id, _list_opts), do: {:error, error()}
end
