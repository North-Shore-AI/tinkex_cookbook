# credo:disable-for-this-file Credo.Check.Refactor.Nesting
defmodule TinkexCookbook.RL.MetricUtil do
  @moduledoc """
  Metric utilities for RL training and evaluation.
  """

  alias TinkexCookbook.RL.{Rollouts, TrajectoryGroup}
  alias TinkexCookbook.Types.ModelInput
  alias TinkexCookbook.Utils.MiscUtils

  @spec compute_trajectory_metrics([TrajectoryGroup.t()], [[String.t()]]) :: map()
  def compute_trajectory_metrics(trajectory_groups, taglists) do
    tag2groups =
      Enum.zip(taglists, trajectory_groups)
      |> Enum.reduce(%{}, fn {taglist, group}, acc ->
        Enum.reduce(taglist, acc, fn tag, acc2 ->
          Map.update(acc2, tag, [group], fn groups -> [group | groups] end)
        end)
      end)

    have_nontrivial_tags =
      Enum.any?(tag2groups, fn {_tag, groups} -> length(groups) < length(trajectory_groups) end)

    metrics =
      if have_nontrivial_tags do
        Enum.reduce(tag2groups, %{}, fn {tag, groups}, acc ->
          prefixed =
            _compute_trajectory_metrics(groups)
            |> Enum.map(fn {k, v} -> {"env/#{tag}/#{k}", v} end)
            |> Map.new()

          Map.merge(acc, prefixed)
        end)
      else
        %{}
      end

    Map.merge(metrics, prefix_metrics("env/all", _compute_trajectory_metrics(trajectory_groups)))
  end

  @spec dataset_to_env_group_builders(struct()) :: [struct()]
  def dataset_to_env_group_builders(dataset) do
    Enum.flat_map(0..(dataset.__struct__.length(dataset) - 1), fn idx ->
      dataset.__struct__.get_batch(dataset, idx)
    end)
  end

  defp prefix_metrics(prefix, metrics) do
    Enum.map(metrics, fn {k, v} -> {"#{prefix}/#{k}", v} end) |> Map.new()
  end

  defp _compute_trajectory_metrics(trajectory_groups) do
    flat_trajs = Enum.flat_map(trajectory_groups, & &1.trajectories_G)

    ac_tokens_by_turn =
      for traj <- flat_trajs, transition <- traj.transitions do
        length(transition.ac.tokens)
      end

    ob_tokens_by_turn =
      for traj <- flat_trajs, transition <- traj.transitions do
        ModelInput.length(transition.ob)
      end

    turns_by_trajectory = Enum.map(flat_trajs, &length(&1.transitions))

    total_turns = Enum.sum(turns_by_trajectory)
    total_episodes = length(flat_trajs)

    metrics = %{
      "ac_tokens_per_turn" => safe_div(Enum.sum(ac_tokens_by_turn), total_turns),
      "ob_tokens_per_turn" => safe_div(Enum.sum(ob_tokens_by_turn), total_turns),
      "turns_per_episode" => safe_div(total_turns, total_episodes),
      "total_episodes" => total_episodes,
      "total_turns" => total_turns,
      "total_ac_tokens" => Enum.sum(ac_tokens_by_turn),
      "total_ob_tokens" => Enum.sum(ob_tokens_by_turn)
    }

    reward_mean =
      trajectory_groups
      |> Enum.flat_map(&TrajectoryGroup.get_total_rewards/1)
      |> mean()

    metrics = Map.put(metrics, "reward/total", reward_mean)

    transition_metrics =
      for tg <- trajectory_groups,
          traj <- tg.trajectories_G,
          transition <- traj.transitions,
          transition.metrics != %{} do
        transition.metrics
      end

    traj_metrics = Enum.flat_map(trajectory_groups, & &1.metrics_G)

    metrics
    |> Map.merge(MiscUtils.dict_mean(transition_metrics ++ traj_metrics))
    |> Map.merge(_compute_by_group_metrics(trajectory_groups))
  end

  defp _compute_by_group_metrics(trajectory_groups, good_thresh \\ 0.5) do
    n_groups = length(trajectory_groups)

    {n_mixed, n_good, n_bad} =
      Enum.reduce(trajectory_groups, {0, 0, 0}, fn group, {mixed, good, bad} ->
        rewards = TrajectoryGroup.get_total_rewards(group)

        if MiscUtils.all_same(rewards) do
          if hd(rewards) >= good_thresh do
            {mixed, good + 1, bad}
          else
            {mixed, good, bad + 1}
          end
        else
          {mixed + 1, good, bad}
        end
      end)

    %{
      "by_group/frac_mixed" => safe_div(n_mixed, n_groups),
      "by_group/frac_all_good" => safe_div(n_good, n_groups),
      "by_group/frac_all_bad" => safe_div(n_bad, n_groups)
    }
  end

  defp mean(values) do
    safe_div(Enum.sum(values), length(values))
  end

  defp safe_div(_num, 0), do: 0.0
  defp safe_div(num, den), do: num / den
end

defmodule TinkexCookbook.RL.MetricUtil.RLTestSetEvaluator do
  @moduledoc """
  Evaluator that runs RL rollouts on a test set.
  """

  alias TinkexCookbook.Completers.TinkexTokenCompleter
  alias TinkexCookbook.RL.{EnvGroupBuilder, MetricUtil, Rollouts}
  alias TinkexCookbook.Utils.Logtree

  @behaviour TinkexCookbook.Eval.Evaluators.SamplingClientEvaluator

  require Logtree

  defstruct [:env_group_builders, :max_tokens, name: "test", num_groups_to_log: 4]

  @type t :: %__MODULE__{
          env_group_builders: [struct()],
          max_tokens: pos_integer(),
          name: String.t(),
          num_groups_to_log: non_neg_integer()
        }

  @spec new(struct(), pos_integer(), keyword()) :: t()
  def new(dataset, max_tokens, opts \\ []) do
    %__MODULE__{
      env_group_builders: MetricUtil.dataset_to_env_group_builders(dataset),
      max_tokens: max_tokens,
      name: Keyword.get(opts, :name, "test"),
      num_groups_to_log: Keyword.get(opts, :num_groups_to_log, 4)
    }
  end

  @impl true
  def evaluate(%__MODULE__{} = evaluator, sampling_client) do
    policy =
      TinkexTokenCompleter.new(
        sampling_client: sampling_client,
        max_tokens: evaluator.max_tokens
      )

    trajectory_groups =
      evaluator.env_group_builders
      |> Enum.with_index()
      |> Task.async_stream(
        fn {builder, idx} ->
          Logtree.optional_enable_logging enable: idx < evaluator.num_groups_to_log do
            Rollouts.do_group_rollout(builder, policy)
          end
        end,
        ordered: true,
        timeout: :infinity
      )
      |> Enum.map(fn {:ok, group} -> group end)

    taglists = Enum.map(evaluator.env_group_builders, &EnvGroupBuilder.logging_tags/1)

    metrics =
      MetricUtil.compute_trajectory_metrics(trajectory_groups, taglists)
      |> Enum.map(fn {k, v} -> {"#{evaluator.name}/#{k}", v} end)
      |> Map.new()

    {:ok, metrics}
  end
end
