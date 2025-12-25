defmodule TinkexCookbook.RL.ProblemEnvTest do
  use ExUnit.Case, async: true

  alias TinkexCookbook.Renderers.Llama3
  alias TinkexCookbook.RL.StepResult
  alias TinkexCookbook.Test.SpecialTokenizer
  alias TinkexCookbook.Types.ModelInput

  defmodule DummyProblemEnv do
    use TinkexCookbook.RL.ProblemEnv

    defstruct [:renderer_module, :renderer_state, :convo_prefix, :format_coef]

    def get_question(_env), do: "2+2?"
    def check_answer(_env, sample_str), do: sample_str == "4"
    def check_format(_env, sample_str), do: sample_str != ""
    def get_reference_answer(_env), do: "4"
  end

  test "initial_observation builds prompt and stop condition" do
    {:ok, state} = Llama3.init(tokenizer: SpecialTokenizer)

    env = %DummyProblemEnv{
      renderer_module: Llama3,
      renderer_state: state,
      convo_prefix: [],
      format_coef: 0.1
    }

    {model_input, stop} = DummyProblemEnv.initial_observation(env)
    assert %ModelInput{} = model_input
    assert stop == Llama3.stop_sequences(state)
  end

  test "step computes reward and metrics from parsed response" do
    {:ok, state} = Llama3.init(tokenizer: SpecialTokenizer)

    env = %DummyProblemEnv{
      renderer_module: Llama3,
      renderer_state: state,
      convo_prefix: [],
      format_coef: 0.1
    }

    action_tokens = SpecialTokenizer.encode("4<|eot_id|>")
    %StepResult{} = step = DummyProblemEnv.step(env, action_tokens)

    assert step.reward == 1.0
    assert step.metrics["format"] == 1.0
    assert step.metrics["correct"] == 1.0
  end
end
