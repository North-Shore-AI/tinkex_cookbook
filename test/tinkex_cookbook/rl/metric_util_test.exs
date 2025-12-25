defmodule TinkexCookbook.RL.MetricUtilTest do
  use ExUnit.Case, async: true

  alias TinkexCookbook.Completers.TokensWithLogprobs
  alias TinkexCookbook.RL.{MetricUtil, Trajectory, TrajectoryGroup, Transition}
  alias TinkexCookbook.Types.ModelInput

  defp transition(ob_len, ac_len, reward) do
    %Transition{
      ob: ModelInput.from_ints(Enum.to_list(1..ob_len)),
      ac: %TokensWithLogprobs{tokens: Enum.to_list(1..ac_len), maybe_logprobs: []},
      reward: reward,
      episode_done: false,
      metrics: %{}
    }
  end

  defp trajectory(transitions) do
    %Trajectory{transitions: transitions, final_ob: ModelInput.from_ints([1])}
  end

  defp group(trajectories, final_rewards) do
    %TrajectoryGroup{
      trajectories_G: trajectories,
      final_rewards_G: final_rewards,
      metrics_G: Enum.map(trajectories, fn _ -> %{} end)
    }
  end

  test "compute_trajectory_metrics aggregates turn and reward statistics" do
    traj1 = trajectory([transition(2, 1, 1.0), transition(3, 2, 0.0)])
    traj2 = trajectory([transition(1, 1, 2.0)])
    traj3 = trajectory([transition(4, 1, 0.0)])

    group1 = group([traj1, traj2], [0.0, 0.0])
    group2 = group([traj3], [0.0])

    metrics =
      MetricUtil.compute_trajectory_metrics([group1, group2], [["math"], ["code"]])

    assert metrics["env/all/total_episodes"] == 3
    assert metrics["env/all/total_turns"] == 4
    assert_in_delta metrics["env/all/ac_tokens_per_turn"], 1.25, 1.0e-12
    assert_in_delta metrics["env/all/ob_tokens_per_turn"], 2.5, 1.0e-12
    assert_in_delta metrics["env/all/turns_per_episode"], 4 / 3, 1.0e-12
    assert metrics["env/all/reward/total"] == 1.0
    assert metrics["env/all/by_group/frac_mixed"] == 0.5
    assert metrics["env/all/by_group/frac_all_bad"] == 0.5
    assert metrics["env/all/by_group/frac_all_good"] == 0.0

    assert Map.has_key?(metrics, "env/math/total_episodes")
    assert Map.has_key?(metrics, "env/code/total_episodes")
  end

  defmodule DummyDataset do
    defstruct [:batches]

    def get_batch(%__MODULE__{batches: batches}, index), do: Enum.at(batches, index)
    def length(%__MODULE__{batches: batches}), do: Kernel.length(batches)
  end

  test "dataset_to_env_group_builders flattens batches" do
    dataset = %DummyDataset{batches: [[1, 2], [3]]}
    assert MetricUtil.dataset_to_env_group_builders(dataset) == [1, 2, 3]
  end
end
