defmodule TinkexCookbook.Eval.RunnerTest do
  @moduledoc """
  Tests for the evaluation runner.

  These tests verify that the Runner correctly orchestrates
  evaluation using EvalEx tasks and TinkexGenerate.
  """
  use ExUnit.Case, async: true

  alias TinkexCookbook.Eval.Runner
  alias TinkexCookbook.Test.MockTinkex

  describe "run/2" do
    setup do
      sampling_client = MockTinkex.SamplingClient.new()

      config = %{
        sampling_client: sampling_client,
        model: "test-model",
        temperature: 0.7,
        max_tokens: 100,
        stop: []
      }

      samples = [
        %{id: "1", input: "What is 2+2?", target: "4"},
        %{id: "2", input: "What is 3+3?", target: "6"}
      ]

      %{config: config, samples: samples}
    end

    test "returns results for all samples", %{config: config, samples: samples} do
      {:ok, results} = Runner.run(samples, config)

      assert length(results) == 2
    end

    test "each result has required fields", %{config: config, samples: samples} do
      {:ok, results} = Runner.run(samples, config)

      Enum.each(results, fn result ->
        assert Map.has_key?(result, :id)
        assert Map.has_key?(result, :input)
        assert Map.has_key?(result, :output)
        assert Map.has_key?(result, :target)
      end)
    end

    test "preserves sample id in results", %{config: config, samples: samples} do
      {:ok, results} = Runner.run(samples, config)

      result_ids = Enum.map(results, & &1.id)
      sample_ids = Enum.map(samples, & &1.id)

      assert result_ids == sample_ids
    end
  end

  describe "run_sample/2" do
    setup do
      sampling_client = MockTinkex.SamplingClient.new()

      config = %{
        sampling_client: sampling_client,
        model: "test-model",
        temperature: 0.7,
        max_tokens: 100,
        stop: []
      }

      %{config: config}
    end

    test "generates output for a sample", %{config: config} do
      sample = %{id: "test", input: "Hello!", target: "Hi!"}

      {:ok, result} = Runner.run_sample(sample, config)

      assert result.id == "test"
      assert is_binary(result.output)
    end

    test "preserves target in result", %{config: config} do
      sample = %{id: "test", input: "Hello!", target: "Hi!"}

      {:ok, result} = Runner.run_sample(sample, config)

      assert result.target == "Hi!"
    end
  end

  describe "create_messages/1" do
    test "creates user message from input string" do
      sample = %{id: "1", input: "What is 2+2?", target: "4"}

      messages = Runner.create_messages(sample)

      assert length(messages) == 1
      assert hd(messages).role == "user"
      assert hd(messages).content == "What is 2+2?"
    end

    test "handles sample with system prompt" do
      sample = %{
        id: "1",
        input: "What is 2+2?",
        target: "4",
        system_prompt: "You are a math teacher."
      }

      messages = Runner.create_messages(sample)

      assert length(messages) == 2
      assert hd(messages).role == "system"
    end
  end

  describe "score_results/2" do
    test "scores results with exact match" do
      results = [
        %{id: "1", input: "q1", output: "4", target: "4"},
        %{id: "2", input: "q2", output: "5", target: "6"}
      ]

      scored = Runner.score_results(results, :exact_match)

      assert length(scored) == 2

      [first, second] = scored
      assert first.score == 1.0
      assert second.score == 0.0
    end

    test "scores results with contains check" do
      results = [
        %{id: "1", input: "q1", output: "The answer is 4.", target: "4"},
        %{id: "2", input: "q2", output: "I think 5.", target: "6"}
      ]

      scored = Runner.score_results(results, :contains)

      [first, second] = scored
      assert first.score == 1.0
      assert second.score == 0.0
    end
  end

  describe "compute_metrics/1" do
    test "computes accuracy from scored results" do
      scored_results = [
        %{id: "1", score: 1.0},
        %{id: "2", score: 0.0},
        %{id: "3", score: 1.0},
        %{id: "4", score: 1.0}
      ]

      metrics = Runner.compute_metrics(scored_results)

      assert metrics.accuracy == 0.75
      assert metrics.total == 4
      assert metrics.correct == 3
    end

    test "handles empty results" do
      metrics = Runner.compute_metrics([])

      assert metrics.accuracy == 0.0
      assert metrics.total == 0
      assert metrics.correct == 0
    end
  end
end
