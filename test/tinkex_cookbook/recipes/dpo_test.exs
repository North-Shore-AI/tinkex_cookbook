defmodule TinkexCookbook.Recipes.DPOTest do
  use ExUnit.Case, async: true

  alias CrucibleKitchen.Workflows.Preference
  alias TinkexCookbook.Recipes.DPO

  describe "recipe metadata" do
    test "name and description" do
      assert DPO.name() == :dpo
      assert is_binary(DPO.description())
    end

    test "uses preference workflow" do
      assert DPO.workflow() == Preference.__workflow__()
    end

    test "requires training, dataset, and sampling adapters" do
      required = DPO.required_adapters()
      assert :training_client in required
      assert :dataset_store in required
      assert :sampling_client in required
    end
  end

  describe "defaults" do
    test "default_config matches DPO defaults" do
      config = DPO.default_config()

      assert config.model == "meta-llama/Llama-3.2-1B"
      assert config.dataset == :hhh
      assert config.dpo_beta == 0.1
      assert config.batch_size == 256
    end
  end
end
