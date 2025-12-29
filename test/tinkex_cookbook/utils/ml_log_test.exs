defmodule TinkexCookbook.Utils.MlLogTest do
  # async: false because tests need to capture Logger output which requires
  # changing the global Logger level
  use ExUnit.Case, async: false

  import ExUnit.CaptureLog

  require Logger

  alias TinkexCookbook.Utils.MlLog

  setup do
    temp_dir = Path.join(System.tmp_dir!(), "ml_log_test_#{:rand.uniform(100_000)}")
    File.rm_rf(temp_dir)

    # Save original logger level and set to :info for these tests
    original_level = Logger.level()
    Logger.configure(level: :info)

    on_exit(fn ->
      Logger.configure(level: original_level)
      File.rm_rf(temp_dir)
    end)

    {:ok, temp_dir: temp_dir}
  end

  describe "setup_logging/2" do
    test "creates log directory and returns logger", %{temp_dir: temp_dir} do
      logger = MlLog.setup_logging(temp_dir)

      assert File.dir?(temp_dir)
      assert logger.log_dir == temp_dir
      assert logger.metrics_file == Path.join(temp_dir, "metrics.jsonl")
      assert logger.config_file == Path.join(temp_dir, "config.json")
    end

    test "logs config when provided", %{temp_dir: temp_dir} do
      config = %{learning_rate: 0.001, batch_size: 32}

      log =
        capture_log([level: :info], fn ->
          logger = MlLog.setup_logging(temp_dir, config: config)

          assert logger.logged_hparams? == true
          assert File.exists?(logger.config_file)

          content = File.read!(logger.config_file)
          decoded = Jason.decode!(content)

          assert decoded["learning_rate"] == 0.001
          assert decoded["batch_size"] == 32
        end)

      assert log =~ "Logged config to"
    end
  end

  describe "log_metrics/3" do
    test "appends metrics to JSONL file", %{temp_dir: temp_dir} do
      logger = MlLog.setup_logging(temp_dir)

      log =
        capture_log([level: :info], fn ->
          MlLog.log_metrics(logger, %{loss: 0.5, accuracy: 0.9}, 1)
          MlLog.log_metrics(logger, %{loss: 0.3, accuracy: 0.95}, 2)
        end)

      assert log =~ "Step 1:"
      assert log =~ "Step 2:"

      content = File.read!(logger.metrics_file)
      lines = String.split(content, "\n", trim: true)

      assert length(lines) == 2

      [line1, line2] = lines
      decoded1 = Jason.decode!(line1)
      decoded2 = Jason.decode!(line2)

      assert decoded1["step"] == 1
      assert decoded1["loss"] == 0.5
      assert decoded2["step"] == 2
      assert decoded2["accuracy"] == 0.95
    end

    test "handles metrics without step", %{temp_dir: temp_dir} do
      logger = MlLog.setup_logging(temp_dir)

      log =
        capture_log([level: :info], fn ->
          MlLog.log_metrics(logger, %{final_loss: 0.1})
        end)

      assert log =~ "final_loss=0.1"

      content = File.read!(logger.metrics_file)
      decoded = Jason.decode!(String.trim(content))

      refute Map.has_key?(decoded, "step")
      assert decoded["final_loss"] == 0.1
    end
  end

  describe "log_hparams/2" do
    test "only logs once", %{temp_dir: temp_dir} do
      logger = MlLog.setup_logging(temp_dir)

      log =
        capture_log([level: :info], fn ->
          logger = MlLog.log_hparams(logger, %{lr: 0.01})
          assert logger.logged_hparams? == true

          # Second call should be no-op
          logger = MlLog.log_hparams(logger, %{lr: 0.02})
          assert logger.logged_hparams? == true
        end)

      assert log =~ "Logged config to"

      # File should still have original value
      content = File.read!(logger.config_file)
      decoded = Jason.decode!(content)

      assert decoded["lr"] == 0.01
    end
  end

  describe "close/1" do
    test "returns :ok", %{temp_dir: temp_dir} do
      logger = MlLog.setup_logging(temp_dir)

      log =
        capture_log([level: :info], fn ->
          assert MlLog.close(logger) == :ok
        end)

      assert log =~ "Closing logger for"
    end
  end

  describe "get_logger_url/1" do
    test "returns nil for file logger", %{temp_dir: temp_dir} do
      logger = MlLog.setup_logging(temp_dir)

      assert MlLog.get_logger_url(logger) == nil
    end
  end
end
