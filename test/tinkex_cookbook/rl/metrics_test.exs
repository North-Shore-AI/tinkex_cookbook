defmodule TinkexCookbook.RL.MetricsTest do
  use ExUnit.Case, async: true

  alias TinkexCookbook.RL.Metrics
  alias TinkexCookbook.Types.{Datum, ModelInput, TensorData}

  defmodule TestSamplingClient do
    defstruct [:logprobs]

    def compute_logprobs(%__MODULE__{logprobs: logprobs}, _model_input) do
      {:ok, Task.async(fn -> {:ok, logprobs} end)}
    end
  end

  defp datum(logprobs, mask, advantages \\ [0.0, 0.0], target_tokens \\ [2, 3]) do
    %Datum{
      model_input: ModelInput.from_ints([1, 2]),
      loss_fn_inputs: %{
        "logprobs" => TensorData.from_list(logprobs, :float32),
        "mask" => TensorData.from_list(mask, :float32),
        "advantages" => TensorData.from_list(advantages, :float32),
        "target_tokens" => TensorData.from_list(target_tokens, :int64)
      }
    }
  end

  test "compute_kl_sample_train computes KL and entropy on action tokens" do
    datum = datum([1.0, 0.0, -1.0], [0.0, 1.0, 1.0])
    training_logprobs = [[0.5, -0.5, -1.5]]

    metrics = Metrics.compute_kl_sample_train([datum], training_logprobs)

    assert metrics["optim/kl_sample_train_v1"] == 0.5
    assert metrics["optim/kl_sample_train_v2"] == 0.125
    assert metrics["optim/entropy"] == 0.5
  end

  test "compute_post_kl uses aligned logprobs and mask" do
    datum = datum([0.1, 0.2], [1.0, 1.0], [0.0, 0.0], [2, 3])
    sampling_client = %TestSamplingClient{logprobs: [0.0, -0.5, -1.0]}

    metrics = Metrics.compute_post_kl([datum], sampling_client)

    assert_in_delta metrics["kl_pre_post_v1"], 0.9, 1.0e-12
    assert_in_delta metrics["kl_pre_post_v2"], 0.45, 1.0e-12
  end

  test "incorporate_kl_penalty adjusts advantages and returns base KL metric" do
    datum = datum([0.0, -1.0], [1.0, 1.0], [0.5, 0.5], [2, 3])
    sampling_client = %TestSamplingClient{logprobs: [0.0, -1.0, -1.0]}

    {updated, metrics} = Metrics.incorporate_kl_penalty([datum], sampling_client, 2.0, 0.0)

    assert metrics["kl_policy_base"] == 0.5

    [updated_datum] = updated
    assert TensorData.to_list(updated_datum.loss_fn_inputs["advantages"]) == [-0.5, 1.5]
  end

  test "discounted_future_sum_vectorized computes discounted sums" do
    assert Metrics.discounted_future_sum_vectorized([1.0, 1.0, 1.0], 0.5) == [1.75, 1.5, 1.0]
  end
end
