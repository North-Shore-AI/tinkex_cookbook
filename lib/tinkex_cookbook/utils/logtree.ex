defmodule TinkexCookbook.Utils.Logtree do
  @moduledoc """
  Minimal logtree-compatible helpers (no-op or Logger-backed).
  """

  require Logger

  alias TinkexCookbook.Utils.Logtree

  defmacro init_trace(title, opts \\ [], do: block) do
    quote do
      Logtree.run_init_trace(unquote(title), unquote(opts), fn ->
        unquote(block)
      end)
    end
  end

  def run_init_trace(_title, _opts, fun) when is_function(fun, 0), do: fun.()

  defmacro scope_header(title, opts \\ [], do: block) do
    quote do
      Logtree.run_scope_header(unquote(title), unquote(opts), fn ->
        unquote(block)
      end)
    end
  end

  def run_scope_header(_title, _opts, fun) when is_function(fun, 0), do: fun.()

  defmacro scope_div(opts \\ [], do: block) do
    quote do
      Logtree.run_scope_div(unquote(opts), fn ->
        unquote(block)
      end)
    end
  end

  def run_scope_div(_opts, fun) when is_function(fun, 0), do: fun.()

  defmacro scope_details(summary, opts \\ [], do: block) do
    quote do
      Logtree.run_scope_details(unquote(summary), unquote(opts), fn ->
        unquote(block)
      end)
    end
  end

  def run_scope_details(_summary, _opts, fun) when is_function(fun, 0), do: fun.()

  defmacro scope_disable(do: block) do
    quote do
      Logtree.run_scope_disable(fn -> unquote(block) end)
    end
  end

  def run_scope_disable(fun) when is_function(fun, 0), do: fun.()

  defmacro optional_enable_logging(enable, do: block) do
    quote do
      Logtree.run_optional_enable_logging(unquote(enable), fn ->
        unquote(block)
      end)
    end
  end

  def run_optional_enable_logging(_enable, fun) when is_function(fun, 0), do: fun.()

  def scope_header_decorator(fun, title \\ nil) when is_function(fun) do
    fn args ->
      run_scope_header(title || "scope", [], fn -> apply(fun, args) end)
    end
  end

  def log_text(text, _opts \\ []) when is_binary(text) do
    Logger.info(text)
    :ok
  end

  def log_html(html, _opts \\ []) when is_binary(html) do
    Logger.debug(html)
    :ok
  end

  def log_formatter(formatter, _opts \\ []) do
    text =
      cond do
        function_exported?(formatter.__struct__, :to_html, 1) ->
          formatter.__struct__.to_html(formatter)

        function_exported?(formatter.__struct__, :to_text, 1) ->
          formatter.__struct__.to_text(formatter)

        true ->
          inspect(formatter)
      end

    Logger.info(text)
    :ok
  end

  def header(text, _opts \\ []) when is_binary(text) do
    Logger.info(text)
    :ok
  end

  def details(_text, _opts \\ []), do: :ok
  def table(_rows, _opts \\ []), do: :ok
  def table_from_dict(_map, _opts \\ []), do: :ok
  def table_from_dict_of_lists(_map, _opts \\ []), do: :ok

  def write_html_with_default_style(_body, _path, _opts \\ []), do: :ok
end
