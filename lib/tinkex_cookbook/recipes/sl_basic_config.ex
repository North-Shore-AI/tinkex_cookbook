defmodule TinkexCookbook.Recipes.SlBasic.CliConfig do
  @moduledoc """
  CLI configuration schema for the sl_basic recipe.
  """

  use ChzEx.Schema

  alias CrucibleTrain.Renderers.TrainOnWhat

  chz_schema do
    field(:log_path, :string, default: "/tmp/tinkex-examples/sl_basic")
    field(:model_name, :string, default: "meta-llama/Llama-3.1-8B")
    field(:learning_rate, :float, default: 2.0e-4)
    field(:lr_schedule, :string, default: "linear")
    field(:num_epochs, :integer, default: 1)
    field(:eval_every, :integer, default: 8)
    field(:save_every, :integer, default: 20)
    field(:batch_size, :integer, default: 128)
    field(:max_length, :integer, default: 32_768)
    field(:lora_rank, :integer, default: 32)
    field(:n_train_samples, :integer, munger: ChzEx.Munger.default(nil))

    field(:train_on_what, :string,
      default: TrainOnWhat.all_assistant_messages(),
      validator: ChzEx.Validator.one_of(TrainOnWhat.values())
    )

    field(:behavior_if_exists, :string,
      default: "ask",
      validator: ChzEx.Validator.one_of(["delete", "resume", "ask", "raise"])
    )
  end
end
