defmodule TinkexCookbook.Recipes.ChatSlTest do
  use ExUnit.Case, async: true

  alias CrucibleKitchen.Workflows.Supervised
  alias TinkexCookbook.Recipes.ChatSl

  describe "recipe metadata" do
    test "name and description" do
      assert ChatSl.name() == :chat_sl
      assert is_binary(ChatSl.description())
    end

    test "uses supervised workflow" do
      assert ChatSl.workflow() == Supervised.__workflow__()
    end

    test "requires training and dataset adapters" do
      assert :training_client in ChatSl.required_adapters()
      assert :dataset_store in ChatSl.required_adapters()
    end
  end

  describe "defaults" do
    test "default_config matches chat_sl defaults" do
      config = ChatSl.default_config()

      assert config.model == "meta-llama/Llama-3.1-8B"
      assert config.dataset == :no_robots
      assert config.learning_rate == 1.0e-4
      assert config.batch_size == 256
      assert config.max_length == 16_384
      assert config.lr_schedule == :linear
    end
  end
end
