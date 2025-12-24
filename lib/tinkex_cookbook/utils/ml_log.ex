defmodule TinkexCookbook.Utils.MlLog do
  @moduledoc """
  ML logging utilities for training runs.

  Provides structured logging of metrics, hyperparameters, and configurations
  to JSONL files. Optionally integrates with external logging services.
  """

  require Logger

  defmodule Logger do
    @moduledoc """
    A logger instance that writes metrics to a log directory.
    """

    defstruct [:log_dir, :metrics_file, :config_file, :logged_hparams?]

    @type t :: %__MODULE__{
            log_dir: String.t(),
            metrics_file: Path.t(),
            config_file: Path.t(),
            logged_hparams?: boolean()
          }
  end

  @doc """
  Sets up logging infrastructure.

  Creates the log directory and initializes file handles for metrics logging.

  ## Options

  - `:wandb_project` - Optional W&B project name (not implemented in Phase 1)
  - `:wandb_name` - Optional W&B run name
  - `:config` - Configuration to log
  - `:do_configure_logging_module` - Whether to configure Elixir's Logger

  ## Returns

  A Logger struct that can be used with `log_metrics/3` and `close/1`.
  """
  @spec setup_logging(String.t(), keyword()) :: Logger.t()
  def setup_logging(log_dir, opts \\ []) do
    log_dir = Path.expand(log_dir)
    File.mkdir_p!(log_dir)

    metrics_file = Path.join(log_dir, "metrics.jsonl")
    config_file = Path.join(log_dir, "config.json")

    logger = %Logger{
      log_dir: log_dir,
      metrics_file: metrics_file,
      config_file: config_file,
      logged_hparams?: false
    }

    # Log config if provided
    config = Keyword.get(opts, :config)

    if config do
      log_hparams(logger, config)
    else
      logger
    end
  end

  @doc """
  Logs hyperparameters/configuration to a JSON file.
  """
  @spec log_hparams(Logger.t(), term()) :: Logger.t()
  def log_hparams(%Logger{logged_hparams?: true} = logger, _config), do: logger

  def log_hparams(%Logger{config_file: config_file} = logger, config) do
    config_map = dump_config(config)
    json = Jason.encode!(config_map, pretty: true)
    File.write!(config_file, json)

    Elixir.Logger.info("Logged config to #{config_file}")

    %{logger | logged_hparams?: true}
  end

  @doc """
  Logs metrics for a training step.

  Appends a JSON line to the metrics file with the step number and all metrics.
  """
  @spec log_metrics(Logger.t(), map(), non_neg_integer() | nil) :: :ok
  def log_metrics(%Logger{metrics_file: metrics_file}, metrics, step \\ nil) do
    entry =
      if step do
        Map.put(metrics, :step, step)
      else
        metrics
      end

    json_line = Jason.encode!(entry) <> "\n"
    File.write!(metrics_file, json_line, [:append])

    # Also log to console with pretty formatting
    log_metrics_to_console(metrics, step)

    :ok
  end

  @doc """
  Closes the logger and performs cleanup.
  """
  @spec close(Logger.t()) :: :ok
  def close(%Logger{log_dir: log_dir}) do
    Elixir.Logger.info("Closing logger for #{log_dir}")
    :ok
  end

  @doc """
  Returns the URL for the logger (e.g., W&B run URL).

  Returns nil for file-only logging.
  """
  @spec get_logger_url(Logger.t()) :: String.t() | nil
  def get_logger_url(%Logger{}), do: nil

  # Private helpers

  defp dump_config(config) when is_struct(config) do
    config
    |> Map.from_struct()
    |> dump_config()
  end

  defp dump_config(config) when is_map(config) do
    config
    |> Enum.map(fn {k, v} -> {to_string(k), dump_config(v)} end)
    |> Map.new()
  end

  defp dump_config(config) when is_list(config) do
    Enum.map(config, &dump_config/1)
  end

  defp dump_config(config) when is_tuple(config) do
    config
    |> Tuple.to_list()
    |> dump_config()
  end

  defp dump_config(config) when is_atom(config), do: to_string(config)
  defp dump_config(config), do: config

  defp log_metrics_to_console(metrics, step) do
    step_str = if step, do: "Step #{step}: ", else: ""

    # Format key metrics for display
    formatted =
      metrics
      |> Enum.sort_by(fn {k, _v} -> to_string(k) end)
      |> Enum.map_join(" | ", fn {k, v} -> format_metric(k, v) end)

    Elixir.Logger.info("#{step_str}#{formatted}")
  end

  defp format_metric(key, value) when is_float(value) do
    "#{key}=#{Float.round(value, 6)}"
  end

  defp format_metric(key, value) do
    "#{key}=#{inspect(value)}"
  end
end
