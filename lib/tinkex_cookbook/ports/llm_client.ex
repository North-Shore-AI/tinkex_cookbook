defmodule TinkexCookbook.Ports.LLMClient do
  @moduledoc """
  Port for LLM inference services (chat/completions).
  """

  alias TinkexCookbook.Ports

  @type adapter_opts :: keyword()
  @type message :: map()

  @callback chat(adapter_opts(), [message()], keyword()) :: {:ok, term()} | {:error, term()}

  @spec chat(Ports.t(), [message()], keyword()) :: {:ok, term()} | {:error, term()}
  def chat(%Ports{} = ports, messages, opts \\ []) do
    {module, adapter_opts} = Ports.resolve(ports, :llm_client)
    module.chat(adapter_opts, messages, opts)
  end
end
