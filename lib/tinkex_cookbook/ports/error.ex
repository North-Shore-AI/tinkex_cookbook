defmodule TinkexCookbook.Ports.Error do
  @moduledoc """
  Standard error type for port/adapter failures.
  """

  defexception [:port, :adapter, :message, :details]

  @type t :: %__MODULE__{
          port: atom(),
          adapter: module(),
          message: String.t(),
          details: term() | nil
        }

  @doc """
  Build a ports error with consistent metadata.
  """
  @spec new(atom(), module(), String.t(), term() | nil) :: t()
  def new(port, adapter, message, details \\ nil) do
    %__MODULE__{
      port: port,
      adapter: adapter,
      message: message,
      details: details
    }
  end
end
