defmodule TinkexCookbook.RL.Rollouts do
  @moduledoc """
  Rollout helpers for RL environments.
  """

  alias TinkexCookbook.RL.{EnvGroupBuilder, Trajectory, TrajectoryGroup, Transition}
  alias TinkexCookbook.Types.ModelInput
  alias TinkexCookbook.Utils.Logtree

  require Logtree

  @spec do_single_rollout(struct(), struct()) :: Trajectory.t()
  def do_single_rollout(policy, env) do
    {ob, stop_condition} = env.__struct__.initial_observation(env)

    {transitions, final_ob} =
      do_rollout(policy, env, ob, stop_condition, [])

    %Trajectory{transitions: Enum.reverse(transitions), final_ob: final_ob}
  end

  defp do_rollout(policy, env, ob, stop_condition, acc) do
    case policy.__struct__.complete(policy, ob, stop_condition) do
      {:ok, ac_with_logprobs} ->
        step_result = env.__struct__.step(env, ac_with_logprobs.tokens)

        transition = %Transition{
          ob: ob,
          ac: ac_with_logprobs,
          reward: step_result.reward,
          episode_done: step_result.episode_done,
          metrics: step_result.metrics
        }

        if step_result.episode_done do
          {[transition | acc], step_result.next_observation}
        else
          do_rollout(
            policy,
            env,
            step_result.next_observation,
            step_result.next_stop_condition,
            [transition | acc]
          )
        end

      {:error, reason} ->
        raise "Token completer failed: #{inspect(reason)}"
    end
  end

  @spec do_group_rollout(struct(), struct()) :: TrajectoryGroup.t()
  def do_group_rollout(env_group_builder, policy) do
    envs = env_group_builder.__struct__.make_envs(env_group_builder)

    trajectories =
      envs
      |> Task.async_stream(fn env -> do_single_rollout(policy, env) end,
        ordered: true,
        timeout: :infinity
      )
      |> Enum.map(fn {:ok, traj} -> traj end)

    rewards_and_metrics =
      EnvGroupBuilder.compute_group_rewards(env_group_builder, trajectories, envs)

    {final_rewards, metrics} = Enum.unzip(rewards_and_metrics)

    Logtree.scope_header "Trajectory Summary" do
      Enum.with_index(trajectories)
      |> Enum.each(fn {traj, idx} ->
        step_reward_sum = Enum.reduce(traj.transitions, 0.0, fn t, acc -> acc + t.reward end)

        rows =
          traj.transitions
          |> Enum.with_index()
          |> Enum.map(fn {t, t_idx} ->
            %{
              "step" => t_idx,
              "ob_len" => ModelInput.length(t.ob),
              "ac_len" => length(t.ac.tokens),
              "reward" => format_reward(t.reward)
            }
          end)
          |> Kernel.++([
            %{
              "step" => "final",
              "ob_len" => ModelInput.length(traj.final_ob),
              "ac_len" => "-",
              "reward" => format_reward(Enum.at(final_rewards, idx))
            },
            %{
              "step" => "total",
              "ob_len" => "-",
              "ac_len" => "-",
              "reward" => format_reward(step_reward_sum + Enum.at(final_rewards, idx))
            }
          ])

        Logtree.table(rows, caption: "Trajectory #{idx}")
      end)
    end

    %TrajectoryGroup{
      trajectories_G: trajectories,
      final_rewards_G: final_rewards,
      metrics_G: metrics
    }
  end

  defp format_reward(reward) when is_float(reward) do
    :io_lib.format("~.3f", [reward]) |> IO.iodata_to_binary()
  end

  defp format_reward(reward), do: to_string(reward)
end
