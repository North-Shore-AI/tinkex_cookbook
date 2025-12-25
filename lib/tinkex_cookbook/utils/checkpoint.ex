defmodule TinkexCookbook.Utils.Checkpoint do
  @moduledoc """
  Checkpoint utilities for saving and resuming training runs.
  """

  require Logger

  @checkpoints_filename "checkpoints.jsonl"

  @type checkpoint_kind :: :sampler | :state | :both
  @type loop_state :: %{optional(:epoch) => integer(), optional(:batch) => integer()}
  @type checkpoint_result :: %{
          optional(String.t()) => String.t()
        }

  @spec load_checkpoints_file(String.t()) :: [map()]
  def load_checkpoints_file(log_dir) when is_binary(log_dir) do
    path = Path.join(log_dir, @checkpoints_filename)

    if File.exists?(path) do
      Logger.info("Reading checkpoints from #{path}")

      path
      |> File.read!()
      |> String.split("\n", trim: true)
      |> Enum.map(&Jason.decode!/1)
    else
      Logger.info("No checkpoints found at #{path}")
      []
    end
  end

  @spec get_last_checkpoint(String.t(), String.t()) :: map() | nil
  def get_last_checkpoint(log_dir, required_key \\ "state_path") when is_binary(required_key) do
    checkpoints = load_checkpoints_file(log_dir)
    checkpoints_with_key = Enum.filter(checkpoints, &Map.has_key?(&1, required_key))

    case checkpoints_with_key do
      [] ->
        Logger.info("No checkpoints found with key #{required_key} in #{log_dir}")
        nil

      list ->
        last = List.last(list)

        Logger.info(
          "Found #{length(list)} valid checkpoints with key '#{required_key}' in #{log_dir}"
        )

        Logger.info("Using last checkpoint: #{inspect(last)}")
        last
    end
  end

  @spec save_checkpoint_async(term(), String.t(), String.t(), loop_state(), checkpoint_kind()) ::
          Task.t()
  def save_checkpoint_async(training_client, name, log_path, loop_state, kind \\ :state) do
    Task.async(fn ->
      save_checkpoint_internal(training_client, name, log_path, loop_state, kind)
    end)
  end

  @spec save_checkpoint(term(), String.t(), String.t(), loop_state(), checkpoint_kind()) ::
          checkpoint_result()
  def save_checkpoint(training_client, name, log_path, loop_state, kind \\ :state) do
    save_checkpoint_async(training_client, name, log_path, loop_state, kind)
    |> Task.await(:infinity)
  end

  defp save_checkpoint_internal(training_client, name, log_path, loop_state, kind) do
    unless kind in [:state, :sampler, :both] do
      raise ArgumentError, "Unknown checkpoint kind: #{inspect(kind)}"
    end

    module = training_client_module(training_client)

    tasks =
      %{}
      |> maybe_put_task(:state, kind, fn ->
        module.save_state(training_client, name)
      end)
      |> maybe_put_task(:sampler, kind, fn ->
        module.save_weights_for_sampler(training_client, name)
      end)

    results =
      Enum.reduce(tasks, %{}, fn {key, task}, acc ->
        case Task.await(task, :infinity) do
          {:ok, result} -> Map.put(acc, key, result)
          {:error, reason} -> raise "Checkpoint save failed: #{inspect(reason)}"
        end
      end)

    paths =
      results
      |> Enum.reduce(%{}, fn {key, result}, acc ->
        path = extract_path!(result)
        Map.put(acc, "#{key}_path", path)
      end)

    full_entry =
      %{"name" => name}
      |> Map.merge(loop_state)
      |> Map.merge(paths)

    File.mkdir_p!(log_path)

    File.write!(Path.join(log_path, @checkpoints_filename), Jason.encode!(full_entry) <> "\n", [
      :append
    ])

    paths
  end

  defp maybe_put_task(acc, key, kind, fun) do
    if kind in [:both, key] do
      case fun.() do
        {:ok, task} -> Map.put(acc, key, task)
        {:error, reason} -> raise "Checkpoint save failed: #{inspect(reason)}"
      end
    else
      acc
    end
  end

  defp extract_path!(%{path: path}) when is_binary(path), do: path
  defp extract_path!(%{"path" => path}) when is_binary(path), do: path

  defp extract_path!(other) do
    raise "Checkpoint save returned invalid response: #{inspect(other)}"
  end

  defp training_client_module(%module{}), do: module
  defp training_client_module(_), do: Tinkex.TrainingClient
end
