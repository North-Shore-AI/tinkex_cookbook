defmodule TinkexCookbook.Recipes.MathRlTest do
  use ExUnit.Case, async: true

  alias CrucibleKitchen.Workflows.Reinforcement
  alias TinkexCookbook.Recipes.MathRl

  describe "recipe metadata" do
    test "name and description" do
      assert MathRl.name() == :math_rl
      assert is_binary(MathRl.description())
    end

    test "uses reinforcement workflow" do
      assert MathRl.workflow() == Reinforcement.__workflow__()
    end

    test "requires training, dataset, and sampling adapters" do
      required = MathRl.required_adapters()
      assert :training_client in required
      assert :dataset_store in required
      assert :sampling_client in required
    end
  end

  describe "defaults" do
    test "default_config uses gsm8k and math env builder" do
      config = MathRl.default_config()

      assert config.model == "meta-llama/Llama-3.1-8B-Instruct"
      assert config.dataset == :gsm8k
      assert config.group_size == 4
      assert config.groups_per_batch == 100
      assert is_function(config.env_group_builder, 1)
    end
  end
end
