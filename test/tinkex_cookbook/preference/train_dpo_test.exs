defmodule TinkexCookbook.Preference.TrainDpoTest do
  use ExUnit.Case, async: true

  alias TinkexCookbook.Preference.TrainDpo

  test "compute_dpo_loss returns expected metrics" do
    chosen = [Nx.tensor(1.0), Nx.tensor(0.0)]
    rejected = [Nx.tensor(0.0), Nx.tensor(1.0)]
    chosen_ref = [Nx.tensor(0.0), Nx.tensor(0.0)]
    rejected_ref = [Nx.tensor(0.0), Nx.tensor(0.0)]

    {loss, metrics} =
      TrainDpo.compute_dpo_loss(chosen, rejected, chosen_ref, rejected_ref, 1.0)

    assert_in_delta Nx.to_number(loss), 0.8132616, 1.0e-4
    assert_in_delta metrics["dpo_loss"], 0.8132616, 1.0e-4
    assert_in_delta metrics["accuracy"], 0.5, 1.0e-6
    assert_in_delta metrics["margin"], 0.0, 1.0e-6
    assert_in_delta metrics["chosen_reward"], 0.5, 1.0e-6
    assert_in_delta metrics["rejected_reward"], 0.5, 1.0e-6
  end
end
