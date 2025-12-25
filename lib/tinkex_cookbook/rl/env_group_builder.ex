defmodule TinkexCookbook.RL.EnvGroupBuilder do
  @moduledoc """
  Behaviour for building groups of environments.
  """

  alias TinkexCookbook.RL.{Trajectory, Types}

  @callback make_envs(builder :: struct()) :: [struct()]
  @callback compute_group_rewards(
              builder :: struct(),
              trajectories :: [Trajectory.t()],
              envs :: [struct()]
            ) :: [{float(), Types.metrics()}]
  @callback logging_tags(builder :: struct()) :: [String.t()]

  @optional_callbacks compute_group_rewards: 3, logging_tags: 1

  @spec compute_group_rewards(struct(), [Trajectory.t()], [struct()]) ::
          [{float(), Types.metrics()}]
  def compute_group_rewards(builder, trajectories, envs) do
    if function_exported?(builder.__struct__, :compute_group_rewards, 3) do
      builder.__struct__.compute_group_rewards(builder, trajectories, envs)
    else
      Enum.map(trajectories, fn _ -> {0.0, %{}} end)
    end
  end

  @spec logging_tags(struct()) :: [String.t()]
  def logging_tags(builder) do
    if function_exported?(builder.__struct__, :logging_tags, 1) do
      builder.__struct__.logging_tags(builder)
    else
      []
    end
  end
end
