defmodule TinkexCookbook.RL.DataProcessingTest do
  use ExUnit.Case, async: true

  alias TinkexCookbook.Completers.TokensWithLogprobs
  alias TinkexCookbook.RL.{DataProcessing, Trajectory, TrajectoryGroup, Transition}
  alias TinkexCookbook.Types.{ModelInput, TensorData}

  defp transition(ob_tokens, ac_tokens, logprobs, reward \\ 0.0) do
    %Transition{
      ob: ModelInput.from_ints(ob_tokens),
      ac: %TokensWithLogprobs{tokens: ac_tokens, maybe_logprobs: logprobs},
      reward: reward,
      episode_done: false
    }
  end

  defp trajectory(transitions, final_tokens) do
    %Trajectory{
      transitions: transitions,
      final_ob: ModelInput.from_ints(final_tokens)
    }
  end

  defp group(trajectories, final_rewards) do
    %TrajectoryGroup{
      trajectories_G: trajectories,
      final_rewards_G: final_rewards,
      metrics_G: Enum.map(trajectories, fn _ -> %{} end)
    }
  end

  test "compute_advantages centers rewards within each group" do
    traj1 = trajectory([transition([1], [2], [0.1], 3.0)], [1, 2])
    traj2 = trajectory([transition([1], [3], [0.1], 1.0)], [1, 3])
    group = group([traj1, traj2], [0.0, 0.0])

    assert DataProcessing.compute_advantages([group]) == [[1.0, -1.0]]
  end

  test "trajectory_to_data merges prefix observations into a single datum" do
    transitions = [
      transition([1, 2], [3], [0.1]),
      transition([1, 2, 3, 4], [5, 6], [0.2, 0.3])
    ]

    traj = trajectory(transitions, [1, 2, 3, 4, 5, 6])
    [datum] = DataProcessing.trajectory_to_data(traj, 2.0)

    assert ModelInput.all_tokens(datum.model_input) == [1, 2, 3, 4, 5]
    assert TensorData.to_list(datum.loss_fn_inputs["target_tokens"]) == [2, 3, 4, 5, 6]
    assert TensorData.to_list(datum.loss_fn_inputs["logprobs"]) == [0.0, 0.1, 0.0, 0.2, 0.3]
    assert TensorData.to_list(datum.loss_fn_inputs["advantages"]) == [0.0, 2.0, 0.0, 2.0, 2.0]
    assert TensorData.to_list(datum.loss_fn_inputs["mask"]) == [0.0, 1.0, 0.0, 1.0, 1.0]
  end

  test "trajectory_to_data splits when observations are not prefix-extended" do
    transitions = [
      transition([1, 2], [3], [0.1]),
      transition([9], [10], [0.2])
    ]

    traj = trajectory(transitions, [9, 10])
    assert length(DataProcessing.trajectory_to_data(traj, 1.0)) == 2
  end

  test "assemble_training_data returns data and metadata aligned to trajectories" do
    traj1 = trajectory([transition([1], [2], [0.1])], [1, 2])
    traj2 = trajectory([transition([3], [4], [0.2])], [3, 4])
    group = group([traj1, traj2], [0.0, 0.0])

    {data, metadata} = DataProcessing.assemble_training_data([group], [[1.0, -1.0]])

    assert length(data) == 2
    assert metadata == [%{group_idx: 0, traj_idx: 0}, %{group_idx: 0, traj_idx: 1}]
  end

  test "remove_constant_reward_groups keeps only non-uniform groups or a singleton fallback" do
    uniform_group =
      group(
        [
          trajectory([transition([1], [2], [0.1], 1.0)], [1, 2]),
          trajectory([transition([3], [4], [0.1], 1.0)], [3, 4])
        ],
        [0.0, 0.0]
      )

    mixed_group =
      group(
        [
          trajectory([transition([1], [2], [0.1], 1.0)], [1, 2]),
          trajectory([transition([3], [4], [0.1], 2.0)], [3, 4])
        ],
        [0.0, 0.0]
      )

    assert DataProcessing.remove_constant_reward_groups([uniform_group, mixed_group]) ==
             [mixed_group]

    assert DataProcessing.remove_constant_reward_groups([uniform_group]) == [uniform_group]
  end
end
