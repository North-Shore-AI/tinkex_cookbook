defmodule TinkexCookbook.RL.Env do
  @moduledoc """
  Behaviour for a single-use RL environment.
  """

  alias TinkexCookbook.Completers.TokenCompleter
  alias TinkexCookbook.RL.{StepResult, Types}

  @callback initial_observation(env :: struct()) ::
              {Types.observation(), TokenCompleter.stop_condition()}
  @callback step(env :: struct(), Types.action()) :: StepResult.t()
end
