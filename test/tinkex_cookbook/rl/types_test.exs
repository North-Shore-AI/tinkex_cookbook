defmodule TinkexCookbook.RL.TypesTest do
  use ExUnit.Case, async: true

  alias TinkexCookbook.RL.{Trajectory, TrajectoryGroup, Transition}
  alias TinkexCookbook.Types.ModelInput

  test "trajectory group totals include step rewards and final rewards" do
    traj1 =
      %Trajectory{
        transitions: [
          %Transition{ob: ModelInput.from_ints([1]), ac: nil, reward: 1.0, episode_done: false},
          %Transition{ob: ModelInput.from_ints([2]), ac: nil, reward: 2.0, episode_done: true}
        ],
        final_ob: ModelInput.from_ints([3])
      }

    traj2 =
      %Trajectory{
        transitions: [
          %Transition{ob: ModelInput.from_ints([4]), ac: nil, reward: 0.0, episode_done: true}
        ],
        final_ob: ModelInput.from_ints([5])
      }

    group = %TrajectoryGroup{
      trajectories_G: [traj1, traj2],
      final_rewards_G: [0.5, 1.0],
      metrics_G: [%{}, %{}]
    }

    assert TrajectoryGroup.get_total_rewards(group) == [3.5, 1.0]
  end
end
