defmodule TinkexCookbook.Completers.MessageCompleter do
  @moduledoc """
  Behaviour for message-based completers.
  """

  alias TinkexCookbook.Renderers.Types.Message

  @callback complete(struct(), [Message.t()]) :: {:ok, Message.t()} | {:error, term()}
end
