defmodule TinkexCookbook.Utils.Trace do
  @moduledoc """
  Minimal trace hooks for compatibility with Python trace utilities.
  """

  @spec scope((-> result)) :: result when result: var
  def scope(fun) when is_function(fun, 0), do: fun.()

  @spec update_scope_context(map()) :: :ok
  def update_scope_context(_attrs), do: :ok

  @spec trace_init(keyword()) :: :ok
  def trace_init(_opts \\ []), do: :ok
end
