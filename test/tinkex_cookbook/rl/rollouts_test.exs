defmodule TinkexCookbook.RL.RolloutsTest do
  use ExUnit.Case, async: true

  alias TinkexCookbook.Completers.TokensWithLogprobs
  alias TinkexCookbook.RL.{Rollouts, StepResult}
  alias TinkexCookbook.Types.ModelInput

  defmodule DeterministicPolicy do
    defstruct []

    def complete(_policy, ob, _stop_condition) do
      token = ob |> ModelInput.all_tokens() |> Enum.sum()
      {:ok, %TokensWithLogprobs{tokens: [token], maybe_logprobs: [0.0]}}
    end
  end

  defmodule TestEnv do
    defstruct [:pid, :initial_tokens]

    def start(initial_tokens, step_results) do
      {:ok, pid} = Agent.start(fn -> step_results end)
      %__MODULE__{pid: pid, initial_tokens: initial_tokens}
    end

    def initial_observation(%__MODULE__{initial_tokens: tokens}) do
      {ModelInput.from_ints(tokens), [0]}
    end

    def step(%__MODULE__{pid: pid}, _action) do
      Agent.get_and_update(pid, fn
        [result | rest] -> {result, rest}
      end)
    end
  end

  defmodule TestEnvGroupBuilder do
    defstruct [:envs, :final_rewards, :metrics]

    def make_envs(%__MODULE__{envs: envs}), do: envs

    def compute_group_rewards(%__MODULE__{} = builder, _trajectories, _envs) do
      Enum.zip(builder.final_rewards, builder.metrics)
    end
  end

  test "do_single_rollout collects transitions until episode_done" do
    env =
      TestEnv.start(
        [1],
        [
          %StepResult{
            reward: 1.0,
            episode_done: false,
            next_observation: ModelInput.from_ints([2]),
            next_stop_condition: [0],
            metrics: %{"step" => 1}
          },
          %StepResult{
            reward: 2.0,
            episode_done: true,
            next_observation: ModelInput.from_ints([3]),
            next_stop_condition: [0],
            metrics: %{"step" => 2}
          }
        ]
      )

    on_exit(fn ->
      if Process.alive?(env.pid) do
        Agent.stop(env.pid)
      end
    end)

    traj = Rollouts.do_single_rollout(%DeterministicPolicy{}, env)

    assert length(traj.transitions) == 2
    [first, second] = traj.transitions
    assert ModelInput.all_tokens(first.ob) == [1]
    assert first.ac.tokens == [1]
    assert first.reward == 1.0
    assert ModelInput.all_tokens(second.ob) == [2]
    assert second.ac.tokens == [2]
    assert second.reward == 2.0
    assert ModelInput.all_tokens(traj.final_ob) == [3]
  end

  test "do_group_rollout returns trajectories and group rewards in order" do
    env1 =
      TestEnv.start(
        [1],
        [
          %StepResult{
            reward: 1.0,
            episode_done: true,
            next_observation: ModelInput.from_ints([2]),
            next_stop_condition: [0],
            metrics: %{}
          }
        ]
      )

    env2 =
      TestEnv.start(
        [10],
        [
          %StepResult{
            reward: 2.0,
            episode_done: true,
            next_observation: ModelInput.from_ints([11]),
            next_stop_condition: [0],
            metrics: %{}
          }
        ]
      )

    on_exit(fn ->
      if Process.alive?(env1.pid), do: Agent.stop(env1.pid)
      if Process.alive?(env2.pid), do: Agent.stop(env2.pid)
    end)

    builder = %TestEnvGroupBuilder{
      envs: [env1, env2],
      final_rewards: [0.5, 1.5],
      metrics: [%{"env" => "a"}, %{"env" => "b"}]
    }

    traj_group = Rollouts.do_group_rollout(builder, %DeterministicPolicy{})

    assert length(traj_group.trajectories_G) == 2
    assert Enum.map(traj_group.final_rewards_G, & &1) == [0.5, 1.5]
    assert traj_group.metrics_G == [%{"env" => "a"}, %{"env" => "b"}]

    [traj1, traj2] = traj_group.trajectories_G
    assert ModelInput.all_tokens(traj1.final_ob) == [2]
    assert ModelInput.all_tokens(traj2.final_ob) == [11]
  end
end
