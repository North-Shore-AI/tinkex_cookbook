defmodule TinkexCookbook.Utils.CliUtilsTest do
  use ExUnit.Case, async: true

  import ExUnit.CaptureLog

  alias TinkexCookbook.Utils.CliUtils

  describe "check_log_dir/2" do
    test "returns :ok for non-existent directory" do
      temp_dir = Path.join(System.tmp_dir!(), "test_cli_utils_#{:rand.uniform(100_000)}")

      # Ensure it doesn't exist
      File.rm_rf(temp_dir)

      log =
        capture_log(fn ->
          assert CliUtils.check_log_dir(temp_dir, :delete) == :ok
        end)

      assert log =~ "does not exist"
      assert log =~ "Will create it"
    end

    test "deletes existing directory with :delete behavior" do
      temp_dir = Path.join(System.tmp_dir!(), "test_cli_delete_#{:rand.uniform(100_000)}")

      # Create the directory with a file
      File.mkdir_p!(temp_dir)
      File.write!(Path.join(temp_dir, "test.txt"), "test")

      log =
        capture_log(fn ->
          assert CliUtils.check_log_dir(temp_dir, :delete) == :ok
        end)

      refute File.exists?(temp_dir)
      assert log =~ "already exists"
      assert log =~ "Will delete it"
    end

    test "returns :resume for existing directory with :resume behavior" do
      temp_dir = Path.join(System.tmp_dir!(), "test_cli_resume_#{:rand.uniform(100_000)}")

      # Create the directory
      File.mkdir_p!(temp_dir)

      log =
        capture_log(fn ->
          assert CliUtils.check_log_dir(temp_dir, :resume) == :resume
        end)

      assert log =~ "exists"
      assert log =~ "Resuming from last checkpoint"

      # Cleanup
      File.rm_rf(temp_dir)
    end

    test "raises for existing directory with :raise behavior" do
      temp_dir = Path.join(System.tmp_dir!(), "test_cli_raise_#{:rand.uniform(100_000)}")

      # Create the directory
      File.mkdir_p!(temp_dir)

      assert_raise RuntimeError, ~r/already exists/, fn ->
        CliUtils.check_log_dir(temp_dir, :raise)
      end

      # Cleanup
      File.rm_rf(temp_dir)
    end

    test "raises for invalid behavior" do
      temp_dir = Path.join(System.tmp_dir!(), "test_cli_invalid_#{:rand.uniform(100_000)}")

      File.mkdir_p!(temp_dir)

      assert_raise ArgumentError, ~r/Invalid behavior_if_exists/, fn ->
        CliUtils.check_log_dir(temp_dir, :invalid)
      end

      File.rm_rf(temp_dir)
    end
  end

  describe "expand_path/1" do
    test "expands ~ to home directory" do
      path = CliUtils.expand_path("~/test")

      assert String.starts_with?(path, "/")
      refute String.contains?(path, "~")
    end

    test "handles absolute paths" do
      path = CliUtils.expand_path("/tmp/test")

      assert path == "/tmp/test"
    end
  end

  describe "ensure_parent_dir/1" do
    test "creates parent directory" do
      temp_dir = Path.join(System.tmp_dir!(), "test_parent_#{:rand.uniform(100_000)}")
      file_path = Path.join(temp_dir, "nested/file.txt")

      # Ensure parent doesn't exist
      File.rm_rf(temp_dir)

      assert CliUtils.ensure_parent_dir(file_path) == :ok
      assert File.dir?(Path.dirname(file_path))

      # Cleanup
      File.rm_rf(temp_dir)
    end
  end
end
