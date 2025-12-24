defmodule TinkexCookbook.Utils.CliUtils do
  @moduledoc """
  CLI utilities for training scripts.

  Provides helpers for log directory management and CLI interaction.
  """

  require Logger

  @type logdir_behavior :: :delete | :resume | :ask | :raise

  @doc """
  Checks if a log directory exists and handles it according to the specified behavior.

  This should be called at the beginning of CLI entrypoint to training scripts.
  It handles cases where we're trying to log to a directory that already exists.

  ## Behaviors

  - `:delete` - Delete the existing log directory and start fresh
  - `:resume` - Continue to the training loop (will try to resume from last checkpoint)
  - `:ask` - Prompt the user for what to do
  - `:raise` - Raise an error if the directory already exists

  ## Examples

      TinkexCookbook.Utils.CliUtils.check_log_dir("/tmp/experiment1", :ask)
  """
  @spec check_log_dir(String.t(), logdir_behavior()) :: :ok | :resume
  def check_log_dir(log_dir, behavior_if_exists) do
    if File.exists?(log_dir) do
      handle_existing_dir(log_dir, behavior_if_exists)
    else
      Logger.info(
        "Log directory #{log_dir} does not exist. Will create it and start logging there."
      )

      :ok
    end
  end

  defp handle_existing_dir(log_dir, :delete) do
    Logger.info(
      "Log directory #{log_dir} already exists. Will delete it and start logging there."
    )

    File.rm_rf!(log_dir)
    :ok
  end

  defp handle_existing_dir(log_dir, :resume) do
    Logger.info("Log directory #{log_dir} exists. Resuming from last checkpoint.")
    :resume
  end

  defp handle_existing_dir(log_dir, :raise) do
    raise "Log directory #{log_dir} already exists. Will not delete it."
  end

  defp handle_existing_dir(log_dir, :ask) do
    prompt_user(log_dir)
  end

  defp handle_existing_dir(_log_dir, behavior) do
    raise ArgumentError, "Invalid behavior_if_exists: #{inspect(behavior)}"
  end

  defp prompt_user(log_dir) do
    IO.puts("Log directory #{log_dir} already exists. What do you want to do?")
    IO.puts("  [d]elete - Delete the directory and start fresh")
    IO.puts("  [r]esume - Resume from last checkpoint")
    IO.puts("  [e]xit   - Exit without doing anything")
    IO.write("> ")

    case IO.gets("") |> String.trim() |> String.downcase() do
      input when input in ["d", "delete"] ->
        File.rm_rf!(log_dir)
        :ok

      input when input in ["r", "resume"] ->
        :resume

      input when input in ["e", "exit"] ->
        System.halt(0)

      _ ->
        IO.puts("Invalid input. Please enter 'd', 'r', or 'e'.")
        prompt_user(log_dir)
    end
  end

  @doc """
  Expands a path, handling ~ for home directory.
  """
  @spec expand_path(String.t()) :: String.t()
  def expand_path(path) do
    Path.expand(path)
  end

  @doc """
  Ensures the parent directory of a path exists.
  """
  @spec ensure_parent_dir(String.t()) :: :ok
  def ensure_parent_dir(path) do
    path
    |> Path.dirname()
    |> File.mkdir_p!()

    :ok
  end
end
