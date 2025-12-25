defmodule TinkexCookbook.Utils.MiscUtils do
  @moduledoc """
  Small utility helpers used across training and RL code.
  """

  @spec timed(String.t(), map(), (-> result)) :: {result, map()} when result: var
  def timed(key, metrics, fun) when is_function(fun, 0) and is_binary(key) do
    start = System.monotonic_time()
    result = fun.()
    elapsed = System.monotonic_time() - start
    seconds = System.convert_time_unit(elapsed, :native, :millisecond) / 1000
    updated = Map.put(metrics, "time/#{key}", seconds)
    {result, updated}
  end

  @spec safezip([list()]) :: list(tuple())
  def safezip(lists) when is_list(lists) do
    lengths = Enum.map(lists, &length/1)

    case lengths do
      [] ->
        []

      [first | rest] ->
        if Enum.any?(rest, &(&1 != first)) do
          raise ArgumentError, "safezip expects lists of equal length"
        end

        Enum.zip(lists)
    end
  end

  @spec safezip(list(), list()) :: list(tuple())
  def safezip(list1, list2), do: safezip([list1, list2])

  @spec dict_mean([map()]) :: map()
  def dict_mean(list_of_dicts) when is_list(list_of_dicts) do
    list_of_dicts
    |> Enum.reduce(%{}, fn dict, acc ->
      Enum.reduce(dict, acc, fn {key, value}, acc2 ->
        Map.update(acc2, key, [value], fn values -> [value | values] end)
      end)
    end)
    |> Enum.map(fn {key, values} ->
      {key, Enum.sum(values) / max(length(values), 1)}
    end)
    |> Map.new()
  end

  @spec split_list(list(), pos_integer()) :: [list()]
  def split_list(list, num_splits) when is_list(list) do
    len = length(list)

    cond do
      num_splits <= 0 ->
        raise ArgumentError, "num_splits must be positive, got #{num_splits}"

      num_splits > len ->
        raise ArgumentError, "Cannot split list of length #{len} into #{num_splits} parts"

      true ->
        edges =
          Enum.map(0..num_splits, fn idx ->
            trunc(idx * len / num_splits)
          end)

        Enum.map(0..(num_splits - 1), fn idx ->
          start_idx = Enum.at(edges, idx)
          end_idx = Enum.at(edges, idx + 1)
          Enum.slice(list, start_idx, end_idx - start_idx)
        end)
    end
  end

  @spec all_same(list()) :: boolean()
  def all_same([]), do: raise(ArgumentError, "all_same expects a non-empty list")

  def all_same([first | rest]) do
    Enum.all?(rest, &(&1 == first))
  end

  @spec concat_lists([[term()]]) :: [term()]
  def concat_lists(list_of_lists) when is_list(list_of_lists) do
    Enum.concat(list_of_lists)
  end

  @spec not_none(term()) :: term()
  def not_none(nil), do: raise(ArgumentError, "value must not be nil")
  def not_none(value), do: value

  @spec lookup_func(String.t(), String.t() | module() | nil) :: (list() -> term())
  def lookup_func(path_to_func, default_module \\ nil) when is_binary(path_to_func) do
    {module_name, func_name} =
      case String.split(path_to_func, ":") do
        [func] when is_binary(default_module) or is_atom(default_module) ->
          {default_module, func}

        [module, func] ->
          {module, func}

        _ ->
          raise ArgumentError, "Invalid path: #{path_to_func}"
      end

    module = resolve_module(module_name)
    func_atom = resolve_existing_atom(func_name, :function)

    exports = module.__info__(:functions)

    unless Enum.any?(exports, fn {name, _arity} -> name == func_atom end) do
      raise ArgumentError, "Function #{func_name} not found in #{inspect(module)}"
    end

    fn args when is_list(args) -> apply(module, func_atom, args) end
  end

  defp resolve_module(module) when is_atom(module), do: module

  defp resolve_module(module) when is_binary(module) do
    parts = String.split(module, ".")
    atoms = Enum.map(parts, &resolve_existing_atom(&1, :module))
    Module.safe_concat(atoms)
  end

  defp resolve_existing_atom(value, _kind) when is_atom(value), do: value

  defp resolve_existing_atom(value, kind) when is_binary(value) do
    String.to_existing_atom(value)
  rescue
    _e in ArgumentError ->
      reraise ArgumentError, [message: "Unknown #{kind} atom: #{value}"], __STACKTRACE__
  end
end
