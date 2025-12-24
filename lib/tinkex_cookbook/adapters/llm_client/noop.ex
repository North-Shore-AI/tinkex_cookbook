defmodule TinkexCookbook.Adapters.LLMClient.Noop do
  @moduledoc """
  No-op adapter for LLM inference.
  """

  @behaviour TinkexCookbook.Ports.LLMClient

  alias TinkexCookbook.Ports.Error

  @impl true
  def chat(_opts, _messages, _opts2) do
    {:error, Error.new(:llm_client, __MODULE__, "LLM adapter is not configured")}
  end
end
