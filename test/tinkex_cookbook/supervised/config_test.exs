defmodule TinkexCookbook.Supervised.ConfigTest do
  use ExUnit.Case, async: true

  alias TinkexCookbook.Supervised.{ChatDatasetBuilderCommonConfig, Config}

  describe "ChatDatasetBuilderCommonConfig" do
    test "creates config with required fields" do
      config = %ChatDatasetBuilderCommonConfig{
        model_name_for_tokenizer: "meta-llama/Llama-3.1-8B",
        renderer_name: "llama3",
        max_length: 4096,
        batch_size: 32
      }

      assert config.model_name_for_tokenizer == "meta-llama/Llama-3.1-8B"
      assert config.renderer_name == "llama3"
      assert config.max_length == 4096
      assert config.batch_size == 32
      assert config.train_on_what == nil
    end

    test "creates config with train_on_what" do
      config = %ChatDatasetBuilderCommonConfig{
        model_name_for_tokenizer: "meta-llama/Llama-3.1-8B",
        renderer_name: "llama3",
        max_length: 4096,
        batch_size: 32,
        train_on_what: "all_assistant_messages"
      }

      assert config.train_on_what == "all_assistant_messages"
    end
  end

  describe "Config" do
    test "creates config with required fields" do
      common_config = %ChatDatasetBuilderCommonConfig{
        model_name_for_tokenizer: "meta-llama/Llama-3.1-8B",
        renderer_name: "llama3",
        max_length: 4096,
        batch_size: 32
      }

      config = %Config{
        log_path: "/tmp/test",
        model_name: "meta-llama/Llama-3.1-8B",
        dataset_builder: {:no_robots, common_config}
      }

      assert config.log_path == "/tmp/test"
      assert config.model_name == "meta-llama/Llama-3.1-8B"
      assert config.learning_rate == 1.0e-4
      assert config.lr_schedule == "linear"
      assert config.num_epochs == 1
      assert config.lora_rank == 32
    end

    test "has default values for training parameters" do
      common_config = %ChatDatasetBuilderCommonConfig{
        model_name_for_tokenizer: "meta-llama/Llama-3.1-8B",
        renderer_name: "llama3",
        max_length: 4096,
        batch_size: 32
      }

      config = %Config{
        log_path: "/tmp/test",
        model_name: "meta-llama/Llama-3.1-8B",
        dataset_builder: {:no_robots, common_config}
      }

      assert config.eval_every == 10
      assert config.save_every == 20
      assert config.adam_beta1 == 0.9
      assert config.adam_beta2 == 0.95
      assert config.adam_eps == 1.0e-8
    end

    test "allows overriding default values" do
      common_config = %ChatDatasetBuilderCommonConfig{
        model_name_for_tokenizer: "meta-llama/Llama-3.1-8B",
        renderer_name: "llama3",
        max_length: 4096,
        batch_size: 32
      }

      config = %Config{
        log_path: "/tmp/test",
        model_name: "meta-llama/Llama-3.1-8B",
        dataset_builder: {:no_robots, common_config},
        learning_rate: 2.0e-4,
        num_epochs: 3,
        eval_every: 8
      }

      assert config.learning_rate == 2.0e-4
      assert config.num_epochs == 3
      assert config.eval_every == 8
    end
  end
end
