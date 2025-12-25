defmodule TinkexCookbook.RL.ProblemEnv do
  @moduledoc """
  Shared environment logic for single-step problem solving tasks.
  """

  alias TinkexCookbook.Renderers.{Helpers, Renderer, Types}
  alias TinkexCookbook.RL.StepResult
  alias TinkexCookbook.Types.ModelInput
  alias TinkexCookbook.Utils.Logtree

  @callback get_question(env :: struct()) :: String.t()
  @callback check_answer(env :: struct(), sample_str :: String.t()) :: boolean()
  @callback check_format(env :: struct(), sample_str :: String.t()) :: boolean()
  @callback get_reference_answer(env :: struct()) :: String.t()

  defmacro __using__(_opts) do
    quote do
      alias TinkexCookbook.RL.{Env, ProblemEnv}

      @behaviour Env
      @behaviour ProblemEnv

      def initial_observation(env), do: ProblemEnv.initial_observation(env)
      def step(env, action), do: ProblemEnv.step(env, action)
    end
  end

  @spec initial_observation(struct()) :: {ModelInput.t(), [String.t()] | [non_neg_integer()]}
  def initial_observation(env) do
    renderer_module = env.renderer_module
    renderer_state = env.renderer_state
    convo_prefix = env.convo_prefix || []

    messages =
      convo_prefix ++
        [Types.message("user", env.__struct__.get_question(env))]

    {model_input, _state} =
      Renderer.build_generation_prompt(
        renderer_module,
        messages,
        "assistant",
        nil,
        renderer_state
      )

    {model_input, renderer_module.stop_sequences(renderer_state)}
  end

  @spec step(struct(), [non_neg_integer()]) :: StepResult.t()
  def step(env, action) do
    renderer_module = env.renderer_module
    renderer_state = env.renderer_state
    format_coef = Map.get(env, :format_coef, 0.1)

    {message, parse_success} =
      if function_exported?(renderer_module, :parse_response, 2) do
        renderer_module.parse_response(action, renderer_state)
      else
        content = Helpers.decode(renderer_state.tokenizer, action)
        {Types.message("assistant", content), false}
      end

    content = Types.ensure_text(message.content)

    correct_format =
      if parse_success and env.__struct__.check_format(env, content), do: 1.0, else: 0.0

    correct_answer = if env.__struct__.check_answer(env, content), do: 1.0, else: 0.0
    total_reward = format_coef * (correct_format - 1.0) + correct_answer

    Logtree.log_text("Problem: #{env.__struct__.get_question(env)}")
    Logtree.log_text("Response: #{message.content}")
    Logtree.log_text("Reference Answer: #{env.__struct__.get_reference_answer(env)}")

    Logtree.log_text(
      "Format Valid: #{format_status(correct_format)}, Correct: #{format_status(correct_answer)}, Reward: #{format_reward(total_reward)}"
    )

    %StepResult{
      reward: total_reward,
      episode_done: true,
      next_observation: ModelInput.empty(),
      next_stop_condition: renderer_module.stop_sequences(renderer_state),
      metrics: %{
        "format" => correct_format,
        "correct" => correct_answer
      }
    }
  end

  defp format_status(1.0), do: "✓"
  defp format_status(_), do: "✗"

  defp format_reward(reward) when is_float(reward) do
    :io_lib.format("~.2f", [reward]) |> IO.iodata_to_binary()
  end
end

defmodule TinkexCookbook.RL.ProblemGroupBuilder do
  @moduledoc """
  EnvGroupBuilder for problem environments.
  """

  @behaviour TinkexCookbook.RL.EnvGroupBuilder

  defstruct [:env_thunk, :num_envs, dataset_name: "problems"]

  @spec make_envs(struct()) :: [struct()]
  def make_envs(%__MODULE__{env_thunk: env_thunk, num_envs: num_envs})
      when is_function(env_thunk, 0) do
    Enum.map(1..num_envs, fn _ -> env_thunk.() end)
  end

  @spec compute_group_rewards(struct(), list(), list()) :: [{float(), map()}]
  def compute_group_rewards(_builder, trajectories, _envs) do
    Enum.map(trajectories, fn _ -> {0.0, %{}} end)
  end

  @spec logging_tags(struct()) :: [String.t()]
  def logging_tags(%__MODULE__{dataset_name: dataset_name}) do
    [dataset_name]
  end
end
