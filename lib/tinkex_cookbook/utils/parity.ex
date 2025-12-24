defmodule TinkexCookbook.Utils.Parity do
  @moduledoc """
  Parity instrumentation for Elixir tinkex_cookbook.

  This module provides opt-in instrumentation for collecting parity artifacts
  when comparing Python and Elixir implementations. Enable by setting the
  environment variable PARITY_MODE=1.

  Artifacts are written to the log_path directory under a `parity/` subdirectory.
  """

  alias TinkexCookbook.Types.{Datum, ModelInput, TensorData, EncodedTextChunk}

  require Logger

  @doc """
  Check if parity mode is enabled via environment variable.
  """
  @spec parity_mode?() :: boolean()
  def parity_mode? do
    case System.get_env("PARITY_MODE") do
      val when val in ["1", "true", "yes", "TRUE", "YES"] -> true
      _ -> false
    end
  end

  @doc """
  Get the parity artifacts directory for the current run.
  Creates the directory if in parity mode.
  """
  @spec get_parity_dir(String.t()) :: String.t()
  def get_parity_dir(log_path) do
    parity_dir = Path.join(log_path, "parity")

    if parity_mode?() do
      File.mkdir_p!(parity_dir)
    end

    parity_dir
  end

  @doc """
  Create a SHA256 hash of content.
  """
  @spec hash_content(term()) :: String.t()
  def hash_content(content) when is_binary(content) do
    :crypto.hash(:sha256, content)
    |> Base.encode16(case: :lower)
  end

  def hash_content(content) do
    content
    |> Jason.encode!(pretty: false)
    |> hash_content()
  end

  @doc """
  Serialize a Datum to a JSON-friendly map for comparison.
  """
  @spec serialize_datum(Datum.t()) :: map()
  def serialize_datum(%Datum{} = datum) do
    %{
      model_input: serialize_model_input(datum.model_input),
      loss_fn_inputs:
        datum.loss_fn_inputs
        |> Enum.map(fn {key, tensor_data} ->
          {key, serialize_tensor_data(tensor_data)}
        end)
        |> Map.new()
    }
  end

  @doc """
  Serialize ModelInput to map.
  """
  @spec serialize_model_input(ModelInput.t()) :: map()
  def serialize_model_input(%ModelInput{chunks: chunks}) do
    serialized_chunks =
      Enum.map(chunks, fn
        %EncodedTextChunk{tokens: tokens} ->
          %{type: "encoded_text", tokens: tokens}

        other ->
          %{type: "unknown", value: inspect(other)}
      end)

    length =
      chunks
      |> Enum.reduce(0, fn
        %EncodedTextChunk{tokens: tokens}, acc -> acc + length(tokens)
        _, acc -> acc
      end)

    %{
      chunks: serialized_chunks,
      length: length
    }
  end

  @doc """
  Serialize TensorData to map.
  """
  @spec serialize_tensor_data(TensorData.t()) :: map()
  def serialize_tensor_data(%TensorData{} = td) do
    %{
      data: td.data,
      dtype: to_string(td.dtype),
      shape: td.shape
    }
  end

  defmodule Logger do
    @moduledoc """
    Parity logger for collecting and writing parity artifacts.
    """

    use Agent

    alias TinkexCookbook.Utils.Parity

    defstruct [
      :log_path,
      :parity_dir,
      :enabled,
      dataset_snapshot: nil,
      rendered_samples: [],
      first_batch_payload: nil
    ]

    @type t :: %__MODULE__{
            log_path: String.t(),
            parity_dir: String.t(),
            enabled: boolean(),
            dataset_snapshot: map() | nil,
            rendered_samples: list(map()),
            first_batch_payload: map() | nil
          }

    @doc """
    Start a new parity logger agent.
    """
    @spec start_link(String.t()) :: {:ok, pid()} | {:error, term()}
    def start_link(log_path) do
      initial_state = %__MODULE__{
        log_path: log_path,
        parity_dir: Parity.get_parity_dir(log_path),
        enabled: Parity.parity_mode?()
      }

      Agent.start_link(fn -> initial_state end, name: __MODULE__)
    end

    @doc """
    Stop the parity logger.
    """
    @spec stop() :: :ok
    def stop do
      if Process.whereis(__MODULE__) do
        Agent.stop(__MODULE__)
      end

      :ok
    end

    @doc """
    Log the first N samples of the dataset.
    """
    @spec log_dataset_snapshot(list(), non_neg_integer(), String.t() | nil) :: :ok
    def log_dataset_snapshot(samples, n_samples \\ 10, id_key \\ nil) do
      if Process.whereis(__MODULE__) && enabled?() do
        Agent.update(__MODULE__, fn state ->
          snapshot = build_dataset_snapshot(samples, n_samples, id_key)
          write_artifact(state, "dataset_snapshot.json", snapshot)
          %{state | dataset_snapshot: snapshot}
        end)
      end

      :ok
    end

    @doc """
    Log a rendered sample for comparison.
    """
    @spec log_rendered_sample(
            non_neg_integer(),
            list(),
            ModelInput.t(),
            list(),
            String.t() | nil
          ) :: :ok
    def log_rendered_sample(sample_index, messages, model_input, weights, prompt_text \\ nil) do
      if Process.whereis(__MODULE__) && enabled?() do
        Agent.update(__MODULE__, fn state ->
          rendered =
            build_rendered_sample(sample_index, messages, model_input, weights, prompt_text)

          new_samples = state.rendered_samples ++ [rendered]

          # Write after collecting a few samples
          if length(new_samples) >= 10 do
            flush_rendered_samples(state, new_samples)
          end

          %{state | rendered_samples: new_samples}
        end)
      end

      :ok
    end

    @doc """
    Log the first batch payload for comparison.
    """
    @spec log_first_batch_payload(list(Datum.t())) :: :ok
    def log_first_batch_payload(batch) do
      if Process.whereis(__MODULE__) && enabled?() do
        Agent.update(__MODULE__, fn state ->
          # Only log the first batch
          if state.first_batch_payload == nil do
            payload = build_batch_payload(batch)
            write_artifact(state, "first_batch_payload.json", payload)
            %{state | first_batch_payload: payload}
          else
            state
          end
        end)
      end

      :ok
    end

    @doc """
    Log checkpoint paths.
    """
    @spec log_checkpoint_paths(list(String.t())) :: :ok
    def log_checkpoint_paths(paths) do
      if Process.whereis(__MODULE__) && enabled?() do
        Agent.get(__MODULE__, fn state ->
          write_artifact(state, "checkpoint_paths.json", %{paths: paths})
        end)
      end

      :ok
    end

    @doc """
    Log the final weights path.
    """
    @spec log_final_weights_path(String.t()) :: :ok
    def log_final_weights_path(path) do
      if Process.whereis(__MODULE__) && enabled?() do
        Agent.get(__MODULE__, fn state ->
          write_artifact(state, "final_weights_path.json", %{path: path})
        end)
      end

      :ok
    end

    @doc """
    Flush all pending artifacts to disk.
    """
    @spec flush() :: :ok
    def flush do
      if Process.whereis(__MODULE__) && enabled?() do
        Agent.update(__MODULE__, fn state ->
          flush_rendered_samples(state, state.rendered_samples)
          state
        end)
      end

      :ok
    end

    @doc """
    Check if parity mode is enabled.
    """
    @spec enabled?() :: boolean()
    def enabled? do
      if Process.whereis(__MODULE__) do
        Agent.get(__MODULE__, & &1.enabled)
      else
        false
      end
    end

    # Private helpers

    defp build_dataset_snapshot(samples, n_samples, id_key) do
      snapshot_samples =
        samples
        |> Enum.take(n_samples)
        |> Enum.with_index()
        |> Enum.map(fn {sample, index} ->
          sample_info = %{
            index: index,
            content_hash: Parity.hash_content(sample)
          }

          sample_info =
            if id_key && is_map(sample) && Map.has_key?(sample, id_key) do
              Map.put(sample_info, :id, Map.get(sample, id_key))
            else
              sample_info
            end

          # Extract messages if available
          if is_map(sample) && Map.has_key?(sample, "messages") do
            Map.put(sample_info, :messages, Map.get(sample, "messages"))
          else
            sample_info
          end
        end)

      %{
        total_count: length(samples),
        n_snapshot: min(n_samples, length(samples)),
        samples: snapshot_samples
      }
    end

    defp build_rendered_sample(sample_index, messages, model_input, weights, prompt_text) do
      weights_list = if is_list(weights), do: weights, else: [weights]
      weights_sum = Enum.sum(weights_list)

      all_tokens =
        model_input.chunks
        |> Enum.flat_map(fn
          %EncodedTextChunk{tokens: tokens} -> tokens
          _ -> []
        end)

      rendered = %{
        sample_index: sample_index,
        messages: messages,
        model_input: Parity.serialize_model_input(model_input),
        weights: weights_list,
        weights_sum: weights_sum,
        all_tokens: all_tokens,
        token_count: length(all_tokens)
      }

      if prompt_text do
        rendered
        |> Map.put(:prompt_text, prompt_text)
        |> Map.put(:prompt_hash, Parity.hash_content(prompt_text))
      else
        rendered
      end
    end

    defp build_batch_payload(batch) do
      datums =
        batch
        |> Enum.with_index()
        |> Enum.map(fn {datum, index} ->
          datum_info = Parity.serialize_datum(datum)
          Map.put(datum_info, :datum_index, index)
        end)

      payload_hash = Parity.hash_content(datums)

      %{
        batch_size: length(batch),
        datums: datums,
        payload_hash: payload_hash
      }
    end

    defp flush_rendered_samples(state, samples) do
      if samples != [] do
        write_artifact(state, "rendered_samples.json", %{
          count: length(samples),
          samples: samples
        })
      end
    end

    defp write_artifact(state, filename, data) do
      if state.enabled do
        filepath = Path.join(state.parity_dir, filename)

        case Jason.encode(data, pretty: true) do
          {:ok, json} ->
            File.write!(filepath, json)

          {:error, reason} ->
            Elixir.Logger.warning(
              "Failed to encode parity artifact #{filename}: #{inspect(reason)}"
            )
        end
      end
    end
  end

  @doc """
  Initialize the parity logger.
  """
  @spec init_logger(String.t()) :: {:ok, pid()} | {:error, term()}
  def init_logger(log_path) do
    Logger.start_link(log_path)
  end

  @doc """
  Stop the parity logger.
  """
  @spec stop_logger() :: :ok
  def stop_logger do
    Logger.stop()
  end

  @doc """
  Log dataset snapshot if parity mode is enabled.
  """
  @spec log_dataset_snapshot(list(), non_neg_integer(), String.t() | nil) :: :ok
  def log_dataset_snapshot(samples, n_samples \\ 10, id_key \\ nil) do
    Logger.log_dataset_snapshot(samples, n_samples, id_key)
  end

  @doc """
  Log a rendered sample if parity mode is enabled.
  """
  @spec log_rendered_sample(
          non_neg_integer(),
          list(),
          ModelInput.t(),
          list(),
          String.t() | nil
        ) :: :ok
  def log_rendered_sample(sample_index, messages, model_input, weights, prompt_text \\ nil) do
    Logger.log_rendered_sample(sample_index, messages, model_input, weights, prompt_text)
  end

  @doc """
  Log the first batch payload if parity mode is enabled.
  """
  @spec log_first_batch_payload(list(Datum.t())) :: :ok
  def log_first_batch_payload(batch) do
    Logger.log_first_batch_payload(batch)
  end

  @doc """
  Flush pending parity artifacts.
  """
  @spec flush_logger() :: :ok
  def flush_logger do
    Logger.flush()
  end
end
