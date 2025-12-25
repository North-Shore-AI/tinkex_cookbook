defmodule TinkexCookbook.RL.PreferenceEnvsTest do
  use ExUnit.Case, async: true

  import ExUnit.CaptureLog

  alias TinkexCookbook.Completers.TokensWithLogprobs
  alias TinkexCookbook.Preference.Comparison
  alias TinkexCookbook.Renderers.{RoleColon, Types}
  alias TinkexCookbook.RL.{PreferenceEnvs, Trajectory, Transition}
  alias TinkexCookbook.RL.PreferenceEnvs.PairwisePreferenceGroupBuilder
  alias TinkexCookbook.Test.MockTokenizer
  alias TinkexCookbook.Types.ModelInput

  defmodule AlwaysPreferSecond do
    defstruct []

    def score(_model, %Comparison{}), do: 1.0
  end

  test "get_pairs produces expected combinations" do
    assert PreferenceEnvs.get_pairs(3, "all_pairs_one_way") == [{0, 1}, {0, 2}, {1, 2}]

    assert PreferenceEnvs.get_pairs(3, "all_pairs_both_ways") ==
             [{0, 1}, {0, 2}, {1, 0}, {1, 2}, {2, 0}, {2, 1}]
  end

  test "get_pairs_chunked splits by chunk size" do
    pairs = PreferenceEnvs.get_pairs_chunked(6, "all_pairs_one_way", 3)

    assert pairs == [{0, 1}, {0, 2}, {1, 2}, {3, 4}, {3, 5}, {4, 5}]
  end

  test "pairwise preference group computes win/loss rewards" do
    {:ok, renderer_state} = RoleColon.init(tokenizer: MockTokenizer)

    builder = %PairwisePreferenceGroupBuilder{
      convo_prefix: [Types.message("user", "Prompt")],
      policy_renderer_module: RoleColon,
      policy_renderer_state: renderer_state,
      tournament_pattern: "all_pairs_one_way",
      preference_model: %AlwaysPreferSecond{},
      num_envs: 2,
      content_preprocessor: nil,
      matchup_group_size: 4
    }

    trajectories = [
      trajectory_with_response("A\n\nUser:"),
      trajectory_with_response("B\n\nUser:")
    ]

    log =
      capture_log(fn ->
        rewards = PairwisePreferenceGroupBuilder.compute_group_rewards(builder, trajectories, [])

        assert [
                 {-1.0, %{"format" => true, "win_minus_loss" => -1.0}},
                 {1.0, %{"format" => true, "win_minus_loss" => 1.0}}
               ] = rewards
      end)

    assert log =~ "Got 2 trajectories, doing 1 pairwise matchups"
    assert log =~ "Matchup (0 vs 1)"
    assert log =~ "Valid format: true"
  end

  defp trajectory_with_response(text) do
    tokens = String.to_charlist(text)

    transition = %Transition{
      ob: ModelInput.from_ints([1]),
      ac: %TokensWithLogprobs{tokens: tokens, maybe_logprobs: []},
      reward: 0.0,
      episode_done: true,
      metrics: %{}
    }

    %Trajectory{transitions: [transition], final_ob: ModelInput.from_ints([1])}
  end
end
