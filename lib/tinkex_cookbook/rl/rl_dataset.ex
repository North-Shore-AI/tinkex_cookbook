defmodule TinkexCookbook.RL.RLDataset do
  @moduledoc """
  Behaviour for datasets that produce EnvGroupBuilder batches.
  """

  @type t :: struct()

  @callback get_batch(dataset :: struct(), index :: non_neg_integer()) :: [struct()]
  @callback length(dataset :: struct()) :: non_neg_integer()
end
