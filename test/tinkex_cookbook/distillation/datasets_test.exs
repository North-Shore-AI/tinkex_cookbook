defmodule TinkexCookbook.Distillation.DatasetsTest do
  use ExUnit.Case, async: true

  alias TinkexCookbook.Distillation.{CompositeDataset, PromptOnlyDataset}
  alias TinkexCookbook.Renderers.RoleColon
  alias TinkexCookbook.Test.MockTokenizer

  defmodule FakeDataset do
    defstruct [:batch]

    def get_batch(%__MODULE__{batch: batch}, _index), do: batch
    def length(_), do: 1
  end

  test "composite dataset concatenates batches and indices" do
    dataset_a = %FakeDataset{batch: [:a1, :a2]}
    dataset_b = %FakeDataset{batch: [:b1]}

    composite = CompositeDataset.new([dataset_a, dataset_b], [2, 1])

    {builders, indices} = CompositeDataset.get_batch(composite, 0)

    assert builders == [:a1, :a2, :b1]
    assert indices == [0, 0, 1]
  end

  test "prompt-only dataset truncates prompts by max tokens" do
    {:ok, renderer_state} = RoleColon.init(tokenizer: MockTokenizer)

    dataset = %PromptOnlyDataset{
      prompts: ["abcd"],
      batch_size: 1,
      group_size: 1,
      renderer_module: RoleColon,
      renderer_state: renderer_state,
      tokenizer: MockTokenizer,
      max_prompt_tokens: 2,
      convo_prefix: nil,
      dataset_name: "test"
    }

    [builder] = PromptOnlyDataset.get_batch(dataset, 0)
    env = builder.env_thunk.()

    assert env.prompt == "ab"
  end
end
