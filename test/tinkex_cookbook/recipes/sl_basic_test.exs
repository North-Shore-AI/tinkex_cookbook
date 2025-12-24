defmodule TinkexCookbook.Recipes.SlBasicTest do
  use ExUnit.Case, async: true

  alias TinkexCookbook.Recipes.SlBasic
  alias TinkexCookbook.Supervised.Config

  describe "build_config/1" do
    test "returns default config with no overrides" do
      config = SlBasic.build_config()

      assert %Config{} = config
      assert config.model_name == "meta-llama/Llama-3.1-8B"
      assert config.learning_rate == 2.0e-4
      assert config.num_epochs == 1
      assert config.lr_schedule == "linear"
    end

    test "applies overrides" do
      config =
        SlBasic.build_config(
          model_name: "Qwen/Qwen3-8B",
          learning_rate: 1.0e-4,
          num_epochs: 3
        )

      assert config.model_name == "Qwen/Qwen3-8B"
      assert config.learning_rate == 1.0e-4
      assert config.num_epochs == 3
    end

    test "includes dataset builder with common config" do
      config = SlBasic.build_config()

      assert {:no_robots, common_config} = config.dataset_builder
      assert common_config.model_name_for_tokenizer == "meta-llama/Llama-3.1-8B"
      assert common_config.renderer_name == "llama3"
      assert common_config.batch_size == 128
      assert common_config.train_on_what == "all_assistant_messages"
    end
  end

  describe "get_recommended_renderer_name/1" do
    test "returns llama3 for Llama models" do
      assert SlBasic.get_recommended_renderer_name("meta-llama/Llama-3.1-8B") == "llama3"
      assert SlBasic.get_recommended_renderer_name("meta-llama/Llama-3.2-1B-Instruct") == "llama3"
    end

    test "returns qwen3 for Qwen models" do
      assert SlBasic.get_recommended_renderer_name("Qwen/Qwen3-8B") == "qwen3"
      assert SlBasic.get_recommended_renderer_name("Qwen/Qwen3-30B-A3B") == "qwen3"
    end

    test "returns deepseekv3 for DeepSeek models" do
      assert SlBasic.get_recommended_renderer_name("deepseek-ai/DeepSeek-V3.1") == "deepseekv3"
    end

    test "returns role_colon for unknown models" do
      assert SlBasic.get_recommended_renderer_name("unknown/model") == "role_colon"
    end
  end

  describe "run_training/1" do
    setup do
      temp_dir = Path.join(System.tmp_dir!(), "sl_basic_test_#{:rand.uniform(100_000)}")
      File.rm_rf(temp_dir)

      on_exit(fn -> File.rm_rf(temp_dir) end)

      {:ok, temp_dir: temp_dir}
    end

    test "returns error when TINKER_API_KEY is not set", %{temp_dir: temp_dir} do
      # Temporarily unset API key to test the error path
      original_key = System.get_env("TINKER_API_KEY")
      System.delete_env("TINKER_API_KEY")

      config = SlBasic.build_config(log_path: temp_dir)
      config = Config.expand_log_path(config)

      result = SlBasic.run_training(config)
      assert result == {:error, :missing_api_key}

      # Restore original key if it existed
      if original_key, do: System.put_env("TINKER_API_KEY", original_key)
    end

    @tag :integration
    @tag timeout: :infinity
    test "creates log directory and runs training (requires TINKER_API_KEY)", %{
      temp_dir: temp_dir
    } do
      if System.get_env("TINKER_API_KEY") do
        config =
          SlBasic.build_config(
            log_path: temp_dir,
            # Use small config for testing
            batch_size: 2,
            num_epochs: 1
          )

        config = Config.expand_log_path(config)

        # Run with limited samples for testing
        assert SlBasic.run_training(config, n_train_samples: 4) == :ok

        # Check that log files were created
        assert File.exists?(Path.join(temp_dir, "config.json"))
        assert File.exists?(Path.join(temp_dir, "metrics.jsonl"))
      else
        IO.puts("Skipping integration test: TINKER_API_KEY not set")
        :ok
      end
    end
  end

  describe "configure_tinkex_http/0" do
    setup do
      original_protocol = System.get_env("TINKEX_HTTP_PROTOCOL")
      original_overrides = Application.get_env(:tinkex, :pool_overrides)

      on_exit(fn ->
        if original_protocol do
          System.put_env("TINKEX_HTTP_PROTOCOL", original_protocol)
        else
          System.delete_env("TINKEX_HTTP_PROTOCOL")
        end

        if original_overrides do
          Application.put_env(:tinkex, :pool_overrides, original_overrides)
        else
          Application.delete_env(:tinkex, :pool_overrides)
        end
      end)

      :ok
    end

    test "defaults to HTTP/1 for training when protocol is unset" do
      System.delete_env("TINKEX_HTTP_PROTOCOL")
      Application.delete_env(:tinkex, :pool_overrides)

      SlBasic.configure_tinkex_http()

      overrides = Application.get_env(:tinkex, :pool_overrides)
      assert is_map(overrides)
      assert Keyword.get(overrides[:training], :protocols) == [:http1]
    end

    test "keeps existing overrides when protocol is http2" do
      System.put_env("TINKEX_HTTP_PROTOCOL", "http2")
      Application.put_env(:tinkex, :pool_overrides, %{training: [size: 3]})

      SlBasic.configure_tinkex_http()

      assert Application.get_env(:tinkex, :pool_overrides) == %{training: [size: 3]}
    end

    test "merges HTTP/1 override with existing training overrides" do
      System.put_env("TINKEX_HTTP_PROTOCOL", "http1")
      Application.put_env(:tinkex, :pool_overrides, %{training: [size: 3]})

      SlBasic.configure_tinkex_http()

      overrides = Application.get_env(:tinkex, :pool_overrides)
      assert Keyword.get(overrides[:training], :size) == 3
      assert Keyword.get(overrides[:training], :protocols) == [:http1]
    end
  end
end
