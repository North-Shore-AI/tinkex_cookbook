defmodule TinkexCookbook.Eval.Evaluators do
  @moduledoc """
  Evaluator behaviours for training and sampling clients.
  """

  defmodule TrainingClientEvaluator do
    @moduledoc false
    @callback evaluate(struct(), pid()) :: {:ok, map()} | {:error, term()}
  end

  defmodule SamplingClientEvaluator do
    @moduledoc false
    @callback evaluate(struct(), pid()) :: {:ok, map()} | {:error, term()}
  end

  @type evaluator_builder :: (-> struct())
  @type sampling_client_evaluator_builder :: (-> struct())
end
