defmodule TinkexCookbook.RL.PreferenceEnvs do
  @moduledoc """
  Preference-based RL environments and dataset builders.
  """

  require Logger

  alias TinkexCookbook.Preference.{
    Comparison,
    ComparisonDatasetBuilder,
    LabeledComparison,
    PreferenceModel
  }

  alias TinkexCookbook.Renderers
  alias TinkexCookbook.Renderers.{Renderer, Types}
  alias TinkexCookbook.RL.{Env, EnvGroupBuilder, RLDataset, StepResult, Trajectory}
  alias TinkexCookbook.TokenizerUtils
  alias TinkexCookbook.Types.ModelInput
  alias TinkexCookbook.Utils.{Logtree, LogtreeFormatters}

  defmodule PreferenceEnv do
    @moduledoc """
    Single-step environment used for preference rollouts.
    """

    @behaviour Env

    defstruct [:convo_prefix, :policy_renderer_module, :policy_renderer_state]

    @type t :: %__MODULE__{
            convo_prefix: [Types.Message.t()],
            policy_renderer_module: module(),
            policy_renderer_state: map()
          }

    @impl true
    def initial_observation(%__MODULE__{} = env) do
      stop_condition = env.policy_renderer_module.stop_sequences(env.policy_renderer_state)

      {model_input, _state} =
        Renderer.build_generation_prompt(
          env.policy_renderer_module,
          env.convo_prefix,
          "assistant",
          nil,
          env.policy_renderer_state
        )

      {model_input, stop_condition}
    end

    @impl true
    def step(%__MODULE__{} = env, _action) do
      stop_condition = env.policy_renderer_module.stop_sequences(env.policy_renderer_state)

      %StepResult{
        reward: 0.0,
        episode_done: true,
        next_observation: ModelInput.empty(),
        next_stop_condition: stop_condition,
        metrics: %{}
      }
    end
  end

  @spec get_pairs_chunked(non_neg_integer(), String.t(), pos_integer()) :: [
          {non_neg_integer(), non_neg_integer()}
        ]
  def get_pairs_chunked(n, pattern, chunk_size) do
    0..(n - 1)
    |> Enum.chunk_every(chunk_size)
    |> Enum.flat_map(fn chunk ->
      chunk_start = List.first(chunk)
      chunk_size = length(chunk)

      get_pairs(chunk_size, pattern)
      |> Enum.map(fn {i, j} -> {chunk_start + i, chunk_start + j} end)
    end)
  end

  @spec get_pairs(non_neg_integer(), String.t()) :: [{non_neg_integer(), non_neg_integer()}]
  def get_pairs(n, pattern) do
    case pattern do
      "all_pairs_both_ways" ->
        for i <- 0..(n - 1), j <- 0..(n - 1), i != j, do: {i, j}

      "all_pairs_one_way" ->
        if n < 2 do
          []
        else
          for i <- 0..(n - 2), j <- (i + 1)..(n - 1), do: {i, j}
        end

      _ ->
        raise ArgumentError, "Invalid tournament pattern: #{inspect(pattern)}"
    end
  end

  defmodule PairwisePreferenceGroupBuilder do
    @moduledoc """
    Env group builder for pairwise preference comparisons.
    """

    @behaviour EnvGroupBuilder

    alias TinkexCookbook.RL.PreferenceEnvs

    require Logtree

    defstruct [
      :convo_prefix,
      :policy_renderer_module,
      :policy_renderer_state,
      :tournament_pattern,
      :preference_model,
      :num_envs,
      :content_preprocessor,
      matchup_group_size: 4,
      eval_target_completion_a: nil
    ]

    @type t :: %__MODULE__{
            convo_prefix: [Types.Message.t()],
            policy_renderer_module: module(),
            policy_renderer_state: map(),
            tournament_pattern: String.t(),
            preference_model: struct(),
            num_envs: pos_integer(),
            content_preprocessor: (String.t() -> String.t()) | nil,
            matchup_group_size: pos_integer(),
            eval_target_completion_a: [Types.Message.t()] | nil
          }

    @impl true
    def make_envs(%__MODULE__{} = builder) do
      Enum.map(1..builder.num_envs, fn _ ->
        %PreferenceEnv{
          convo_prefix: builder.convo_prefix,
          policy_renderer_module: builder.policy_renderer_module,
          policy_renderer_state: builder.policy_renderer_state
        }
      end)
    end

    @impl true
    def compute_group_rewards(%__MODULE__{} = builder, trajectory_group, _env_group) do
      response_tuples = Enum.map(trajectory_group, &get_response_message(builder, &1))
      {response_messages, is_valid_list} = Enum.unzip(response_tuples)

      Logtree.scope_header "Prompt" do
        Logtree.log_formatter(%LogtreeFormatters.ConversationFormatter{
          messages: builder.convo_prefix
        })
      end

      Enum.with_index(response_messages)
      |> Enum.each(fn {messages, idx} ->
        Logtree.scope_header "Completion #{idx}" do
          Logtree.log_formatter(%LogtreeFormatters.ConversationFormatter{messages: messages})
          Logtree.log_text("Valid format: #{Enum.at(is_valid_list, idx)}")
        end
      end)

      comparison_indices_pairs =
        PreferenceEnvs.get_pairs_chunked(
          length(response_messages),
          builder.tournament_pattern,
          builder.matchup_group_size
        )

      Logtree.log_text(
        "Got #{length(trajectory_group)} trajectories, doing #{length(comparison_indices_pairs)} pairwise matchups."
      )

      comparisons =
        Enum.map(comparison_indices_pairs, fn {i, j} ->
          comparison_reward_for_second_messages(
            builder,
            Enum.at(response_messages, i),
            Enum.at(response_messages, j)
          )
        end)

      rewards = Enum.map(comparisons, &PreferenceModel.score(builder.preference_model, &1))

      Logtree.scope_header "Pairwise Comparisons" do
        Enum.each(Enum.zip(comparison_indices_pairs, rewards), fn {{i, j}, reward} ->
          Logtree.log_text("Matchup (#{i} vs #{j}) — Reward: #{Float.round(reward, 2)}")
        end)
      end

      win_minus_loss_list = List.duplicate(0.0, length(response_messages))
      matchup_count = List.duplicate(0, length(response_messages))

      {win_minus_loss_list, matchup_count} =
        Enum.zip(comparison_indices_pairs, rewards)
        |> Enum.reduce({win_minus_loss_list, matchup_count}, fn {{i, j}, reward},
                                                                {wins, counts} ->
          wins = List.update_at(wins, j, &(&1 + reward))
          wins = List.update_at(wins, i, &(&1 - reward))
          counts = List.update_at(counts, j, &(&1 + 1))
          counts = List.update_at(counts, i, &(&1 + 1))
          {wins, counts}
        end)

      format_coef = 1.0

      Enum.zip([win_minus_loss_list, is_valid_list, matchup_count])
      |> Enum.map(fn {win_minus_loss, is_valid, count} ->
        count = max(count, 1)
        wml = win_minus_loss / count
        reward = wml + format_coef * (bool_to_float(is_valid) - 1.0)
        {reward, %{"win_minus_loss" => wml, "format" => is_valid}}
      end)
    end

    @impl true
    def logging_tags(_builder), do: ["pair_pref"]

    defp get_response_message(%__MODULE__{} = builder, %Trajectory{transitions: [transition | _]}) do
      {message, is_valid} =
        builder.policy_renderer_module.parse_response(
          transition.ac.tokens,
          builder.policy_renderer_state
        )

      {[message], is_valid}
    end

    defp comparison_reward_for_second_messages(%__MODULE__{} = builder, message_i, message_j) do
      %Comparison{
        prompt_conversation: builder.convo_prefix,
        completion_a: Enum.map(message_i, &preprocess_message(builder, &1)),
        completion_b: Enum.map(message_j, &preprocess_message(builder, &1))
      }
    end

    defp preprocess_message(%__MODULE__{content_preprocessor: nil}, message), do: message

    defp preprocess_message(%__MODULE__{content_preprocessor: preprocessor}, message) do
      content = Types.ensure_text(message.content)
      %{message | content: preprocessor.(content)}
    end

    defp bool_to_float(true), do: 1.0
    defp bool_to_float(false), do: 0.0
  end

  defmodule PairwisePreferenceDataset do
    @moduledoc """
    Dataset that yields pairwise preference env group builders.
    """

    @behaviour RLDataset

    defstruct [
      :comparison_builder,
      :renderer_module,
      :renderer_state,
      :batch_size,
      :preference_model,
      :train_dataset,
      :tournament_pattern,
      :group_size,
      :content_preprocessor
    ]

    @type t :: %__MODULE__{
            comparison_builder: struct(),
            renderer_module: module(),
            renderer_state: map(),
            batch_size: pos_integer(),
            preference_model: struct(),
            train_dataset: [map()],
            tournament_pattern: String.t(),
            group_size: pos_integer(),
            content_preprocessor: (String.t() -> String.t()) | nil
          }

    @impl true
    def get_batch(%__MODULE__{} = dataset, index) do
      start_idx = index * dataset.batch_size
      examples = Enum.slice(dataset.train_dataset, start_idx, dataset.batch_size)

      examples
      |> Enum.map(
        &ComparisonDatasetBuilder.example_to_labeled_comparison(dataset.comparison_builder, &1)
      )
      |> Enum.filter(&(&1 != nil))
      |> Enum.map(&labeled_comparison_to_env_group(dataset, &1))
    end

    @impl true
    def length(%__MODULE__{} = dataset) do
      div(Kernel.length(dataset.train_dataset), dataset.batch_size)
    end

    defp labeled_comparison_to_env_group(%__MODULE__{} = dataset, %LabeledComparison{} = labeled) do
      %PairwisePreferenceGroupBuilder{
        convo_prefix: labeled.comparison.prompt_conversation,
        policy_renderer_module: dataset.renderer_module,
        policy_renderer_state: dataset.renderer_state,
        preference_model: dataset.preference_model,
        tournament_pattern: dataset.tournament_pattern,
        num_envs: dataset.group_size,
        content_preprocessor: dataset.content_preprocessor
      }
    end
  end

  defmodule PairwisePreferenceRLDatasetBuilder do
    @moduledoc """
    Builder for pairwise preference RL datasets.
    """

    use ChzEx.Schema

    chz_schema do
      field(:comparison_builder, :any, virtual: true)
      field(:batch_size, :integer)
      field(:policy_renderer_name, :string)
      field(:policy_model_name, :string)
      field(:tournament_pattern, :string, default: "all_pairs_both_ways")
      field(:group_size, :integer)
      field(:content_preprocessor, :any, default: nil, virtual: true)
      field(:preference_model_builder, :any, virtual: true)
    end

    @spec build(struct()) :: {PairwisePreferenceDataset.t(), nil}
    def build(%__MODULE__{} = builder) do
      {:ok, tokenizer} = TokenizerUtils.get_tokenizer(builder.policy_model_name)
      {:ok, renderer_module, extra_opts} = Renderers.lookup(builder.policy_renderer_name)
      {:ok, renderer_state} = renderer_module.init([{:tokenizer, tokenizer} | extra_opts])

      preference_model =
        case builder.preference_model_builder do
          %module{} = pref_builder -> module.build(pref_builder)
          fun when is_function(fun, 0) -> fun.()
        end

      {train_dataset, _} =
        ComparisonDatasetBuilder.get_train_and_test_datasets(builder.comparison_builder)

      dataset = %PairwisePreferenceDataset{
        comparison_builder: builder.comparison_builder,
        renderer_module: renderer_module,
        renderer_state: renderer_state,
        batch_size: builder.batch_size,
        preference_model: preference_model,
        train_dataset: train_dataset,
        tournament_pattern: builder.tournament_pattern,
        group_size: builder.group_size,
        content_preprocessor: builder.content_preprocessor
      }

      {dataset, nil}
    end
  end
end
