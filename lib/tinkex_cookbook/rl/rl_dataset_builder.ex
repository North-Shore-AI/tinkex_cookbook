defmodule TinkexCookbook.RL.RLDatasetBuilder do
  @moduledoc """
  Behaviour for building RL datasets.
  """

  alias TinkexCookbook.RL.RLDataset

  @callback build(builder :: struct()) :: {RLDataset.t(), RLDataset.t() | nil}
end
