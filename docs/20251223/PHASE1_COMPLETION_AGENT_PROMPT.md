# Phase 1 Completion Agent Prompt

**Date:** 2025-12-23
**Mission:** Complete Phase 1 of tinkex_cookbook with TDD, zero warnings, all tests passing
**Estimated LOC:** ~750 new code + tests

---

## Executive Summary

You are completing the Elixir port of `tinker-cookbook` (Python) to `tinkex_cookbook` (Elixir).
Phase 1 establishes core plumbing with `sl_basic` as the reference recipe.

**What's DONE:**
- TrainOnWhat enum, Renderer behaviour, Types (ModelInput, Datum, TensorData)
- SlBasic recipe skeleton with ChzEx config
- CLI utilities (CliUtils, MlLog)
- Ports/Adapters pattern
- Dependencies: crucible_harness, eval_ex, crucible_datasets added to mix.exs

**What's TODO (your mission):**
1. Concrete Llama3 renderer implementation
2. Renderer parity tests (from Python test_renderers.py)
3. NoRobots dataset builder wiring with CrucibleDatasets
4. Tinkex training integration (replace stubs in SlBasic)
5. TinkexGenerate adapter for eval stack (~100 LOC)
6. Update documentation (README.md, AGENTS.md)

---

## Required Reading (Do This First)

### 1. Project Instructions
```
/home/home/p/g/North-Shore-AI/tinkex_cookbook/AGENTS.md
/home/home/p/g/North-Shore-AI/tinkex_cookbook/README.md
/home/home/p/g/North-Shore-AI/CLAUDE.md
```

### 2. Phase 1 Planning Docs
```
/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251221/COOKBOOK_CORE_FOUNDATION.md
/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251221/PHASE1_FOUNDATION_SLICE.md
/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251221/REMAINING_WORK.md
/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251221/DEPENDENCY_USAGE_TABLE.md
```

### 3. Architecture Docs
```
/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251223/INSPECT_AI_ELIXIR_ARCHITECTURE.md
/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251223/PYTHON_TO_ELIXIR_LIBRARY_MAPPING.md
/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251220/11_tinker_to_tinkex.md
/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251220/12a_porting_cheatsheet.md
```

### 4. Existing Elixir Implementation (READ THESE)
```
/home/home/p/g/North-Shore-AI/tinkex_cookbook/lib/tinkex_cookbook/renderers/train_on_what.ex
/home/home/p/g/North-Shore-AI/tinkex_cookbook/lib/tinkex_cookbook/renderers/renderer.ex
/home/home/p/g/North-Shore-AI/tinkex_cookbook/lib/tinkex_cookbook/renderers/types.ex
/home/home/p/g/North-Shore-AI/tinkex_cookbook/lib/tinkex_cookbook/types/model_input.ex
/home/home/p/g/North-Shore-AI/tinkex_cookbook/lib/tinkex_cookbook/types/datum.ex
/home/home/p/g/North-Shore-AI/tinkex_cookbook/lib/tinkex_cookbook/types/tensor_data.ex
/home/home/p/g/North-Shore-AI/tinkex_cookbook/lib/tinkex_cookbook/recipes/sl_basic.ex
/home/home/p/g/North-Shore-AI/tinkex_cookbook/lib/tinkex_cookbook/supervised/config.ex
/home/home/p/g/North-Shore-AI/tinkex_cookbook/lib/tinkex_cookbook/supervised/common.ex
/home/home/p/g/North-Shore-AI/tinkex_cookbook/lib/tinkex_cookbook/supervised/dataset.ex
/home/home/p/g/North-Shore-AI/tinkex_cookbook/lib/tinkex_cookbook/utils/ml_log.ex
/home/home/p/g/North-Shore-AI/tinkex_cookbook/lib/tinkex_cookbook/utils/cli_utils.ex
/home/home/p/g/North-Shore-AI/tinkex_cookbook/lib/tinkex_cookbook/ports.ex
/home/home/p/g/North-Shore-AI/tinkex_cookbook/mix.exs
```

### 5. Python Source (Behavior Spec) - PORT FROM THESE
```
/home/home/p/g/North-Shore-AI/tinkex_cookbook/tinker-cookbook/tinker_cookbook/renderers.py
/home/home/p/g/North-Shore-AI/tinkex_cookbook/tinker-cookbook/tinker_cookbook/recipes/sl_basic.py
/home/home/p/g/North-Shore-AI/tinkex_cookbook/tinker-cookbook/tinker_cookbook/recipes/chat_sl/chat_datasets.py
/home/home/p/g/North-Shore-AI/tinkex_cookbook/tinker-cookbook/tinker_cookbook/supervised/train.py
/home/home/p/g/North-Shore-AI/tinkex_cookbook/tinker-cookbook/tinker_cookbook/supervised/data.py
/home/home/p/g/North-Shore-AI/tinkex_cookbook/tinker-cookbook/tinker_cookbook/supervised/types.py
/home/home/p/g/North-Shore-AI/tinkex_cookbook/tinker-cookbook/tinker_cookbook/supervised/common.py
/home/home/p/g/North-Shore-AI/tinkex_cookbook/tinker-cookbook/tinker_cookbook/tokenizer_utils.py
/home/home/p/g/North-Shore-AI/tinkex_cookbook/tinker-cookbook/tinker_cookbook/eval/inspect_utils.py
/home/home/p/g/North-Shore-AI/tinkex_cookbook/tinker-cookbook/tinker_cookbook/eval/inspect_evaluators.py
```

### 6. Python Tests (Mirror These)
```
/home/home/p/g/North-Shore-AI/tinkex_cookbook/tinker-cookbook/tinker_cookbook/tests/test_renderers.py
/home/home/p/g/North-Shore-AI/tinkex_cookbook/tinker-cookbook/tinker_cookbook/tests/test_utils.py
```

### 7. Ecosystem Libraries (Understand APIs)
```
/home/home/p/g/North-Shore-AI/crucible_harness/lib/research_harness/solver.ex
/home/home/p/g/North-Shore-AI/crucible_harness/lib/research_harness/generate.ex
/home/home/p/g/North-Shore-AI/crucible_harness/lib/research_harness/task_state.ex
/home/home/p/g/North-Shore-AI/eval_ex/lib/eval_ex/task.ex
/home/home/p/g/North-Shore-AI/eval_ex/lib/eval_ex/sample.ex
/home/home/p/g/North-Shore-AI/eval_ex/lib/eval_ex/scorer.ex
/home/home/p/g/North-Shore-AI/crucible_datasets/lib/dataset_manager/memory_dataset.ex
```

---

## Task 1: Concrete Llama3 Renderer (~200 LOC)

### Context
The `Renderer` behaviour exists but no concrete implementation. Port the Llama3 renderer from Python.

### Files to Create
```
lib/tinkex_cookbook/renderers/llama3.ex
```

### Python Reference
Read `tinker-cookbook/tinker_cookbook/renderers.py`, specifically:
- `Llama3Renderer` class
- BOS/EOS tokens
- Role formatting (`<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>`)
- Stop sequences

### Implementation Spec

```elixir
defmodule TinkexCookbook.Renderers.Llama3 do
  @moduledoc """
  Llama 3 family renderer.

  Formats messages using Llama 3's chat template:
  - BOS token: 128000
  - Role header: <|start_header_id|>{role}<|end_header_id|>
  - Content follows header
  - Message ends with <|eot_id|> (128009)
  """

  @behaviour TinkexCookbook.Renderers.Renderer

  # Token IDs (from Llama 3 tokenizer)
  @bos_token 128000
  @eos_token 128001
  @start_header_id 128006
  @end_header_id 128007
  @eot_id 128009

  @impl true
  def init(opts) do
    tokenizer = Keyword.fetch!(opts, :tokenizer)
    {:ok, %{tokenizer: tokenizer}}
  end

  @impl true
  def bos_tokens(_state), do: [@bos_token]

  @impl true
  def stop_sequences(_state), do: [@eot_id, @eos_token]

  @impl true
  def render_message(idx, message, is_last, state) do
    # Build prefix: <|start_header_id|>{role}<|end_header_id|>\n\n
    # Build content: message.content tokens
    # Build suffix: <|eot_id|> (only if not generating)
    # Return {%RenderedMessage{prefix, content, suffix}, state}
  end

  @impl true
  def parse_response(tokens, state) do
    # Decode tokens, strip EOS markers
    # Return {%Message{role: "assistant", content: text}, is_complete}
  end
end
```

### TDD Steps
1. Write test file `test/tinkex_cookbook/renderers/llama3_test.exs`
2. Test `init/1` returns state with tokenizer
3. Test `bos_tokens/1` returns `[128000]`
4. Test `stop_sequences/1` returns correct tokens
5. Test `render_message/4` for user, assistant, system roles
6. Test full `build_supervised_example/4` integration via Renderer module
7. Implement to pass tests

---

## Task 2: Renderer Parity Tests (~150 LOC)

### Context
Port tests from Python `test_renderers.py` to verify exact behavior parity.

### Files to Create
```
test/tinkex_cookbook/renderers/renderer_test.exs
test/tinkex_cookbook/renderers/train_on_what_test.exs
test/support/mock_tokenizer.ex
```

### Mock Tokenizer
Create a deterministic mock tokenizer for testing:

```elixir
defmodule TinkexCookbook.Test.MockTokenizer do
  @moduledoc "Deterministic tokenizer for testing"

  def encode(text, _opts \\ []) do
    # Simple: each character becomes its codepoint
    # Or use a fixed mapping for predictable tests
    String.to_charlist(text)
  end

  def decode(tokens) do
    List.to_string(tokens)
  end
end
```

### Tests to Port
From `test_renderers.py`:
1. `test_train_on_what_values` - Verify enum values match Python exactly
2. `test_build_supervised_example_weights` - Weight assignment per TrainOnWhat
3. `test_last_assistant_message_weights` - Only last assistant gets weight 1.0
4. `test_all_assistant_messages_weights` - All assistant messages get weight 1.0
5. `test_all_tokens_weights` - Everything gets weight 1.0
6. `test_customized_weights` - Per-message trainable field
7. `test_generation_prompt` - Prompt building for sampling

---

## Task 3: NoRobots Dataset Builder (~100 LOC)

### Context
Wire the NoRobots dataset loading using CrucibleDatasets.

### Files to Create/Modify
```
lib/tinkex_cookbook/datasets/no_robots.ex
lib/tinkex_cookbook/supervised/dataset.ex (modify)
```

### Python Reference
Read `chat_sl/chat_datasets.py`:
- `NoRobotsBuilder` class
- HuggingFace dataset: `HuggingFaceH4/no_robots`
- Field mapping: `messages` field contains conversation

### Implementation Spec

```elixir
defmodule TinkexCookbook.Datasets.NoRobots do
  @moduledoc """
  NoRobots dataset builder for supervised chat fine-tuning.

  Dataset: HuggingFaceH4/no_robots
  Format: Each example has a `messages` field with conversation turns.
  """

  alias CrucibleDatasets.MemoryDataset

  @dataset_name "HuggingFaceH4/no_robots"

  def load(opts \\ []) do
    split = Keyword.get(opts, :split, "train")
    limit = Keyword.get(opts, :limit, nil)

    # Use HfDatasetsEx or CrucibleDatasets to load
    # Transform to list of %{messages: [...]}
    # Return {:ok, samples} or {:error, reason}
  end

  def build_datums(samples, renderer_module, renderer_state, config) do
    # For each sample:
    # 1. Extract messages
    # 2. Call Renderer.build_supervised_example/4
    # 3. Convert to Datum with TensorData weights
    # Return list of Datum structs
  end
end
```

### TDD Steps
1. Write `test/tinkex_cookbook/datasets/no_robots_test.exs`
2. Test `load/1` with mock data (don't hit network)
3. Test `build_datums/4` produces correct Datum structs
4. Test weight alignment with token count

---

## Task 4: Tinkex Training Integration (~200 LOC)

### Context
Replace stubs in SlBasic with actual Tinkex client calls.

### Files to Modify
```
lib/tinkex_cookbook/recipes/sl_basic.ex
lib/tinkex_cookbook/supervised/train.ex (create)
```

### Python Reference
Read `supervised/train.py`:
- `run_supervised_training()` function
- Training loop structure
- `forward_backward` + `optim_step` pattern
- Checkpoint saving

### Implementation Spec

```elixir
defmodule TinkexCookbook.Supervised.Train do
  @moduledoc """
  Supervised training orchestration using Tinkex clients.
  """

  require Logger

  def run(config, opts \\ []) do
    # 1. Create Tinkex.TrainingClient
    # 2. Load dataset using NoRobots.load/1
    # 3. Build datums using NoRobots.build_datums/4
    # 4. Training loop:
    #    - Batch datums
    #    - Call training_client.forward_backward(batch)
    #    - Call training_client.optim_step()
    #    - Log metrics
    #    - Checkpoint periodically
    # 5. Save final weights
  end

  defp create_training_client(config) do
    # Use Tinkex.ServiceClient to create TrainingClient
  end

  defp training_loop(client, batches, config) do
    # Iterate batches, call forward_backward + optim_step
  end
end
```

### Mocking for Tests
Create mock Tinkex clients:

```elixir
# test/support/mock_tinkex.ex
defmodule TinkexCookbook.Test.MockTinkex do
  defmodule TrainingClient do
    def forward_backward(_batch), do: {:ok, %{loss: 0.5}}
    def optim_step(), do: :ok
  end

  defmodule SamplingClient do
    def sample(_prompt, _params), do: {:ok, %{tokens: [1, 2, 3], text: "response"}}
  end
end
```

### TDD Steps
1. Write `test/tinkex_cookbook/supervised/train_test.exs`
2. Test training loop with mock client
3. Test batching logic
4. Test checkpoint creation
5. Test metric logging

---

## Task 5: TinkexGenerate Eval Adapter (~100 LOC)

### Context
Wire the evaluation stack by implementing `CrucibleHarness.Generate` behaviour.

### Files to Create
```
lib/tinkex_cookbook/eval/tinkex_generate.ex
lib/tinkex_cookbook/eval/runner.ex
```

### Python Reference
Read `eval/inspect_utils.py`:
- `InspectAPIFromTinkerSampling` class
- Message conversion
- Response parsing

### Implementation Spec

```elixir
defmodule TinkexCookbook.Eval.TinkexGenerate do
  @moduledoc """
  Implements CrucibleHarness.Generate using Tinkex.SamplingClient.

  This is the adapter that bridges the evaluation framework to Tinkex.
  """

  @behaviour CrucibleHarness.Generate

  @impl true
  def generate(messages, config) do
    # 1. Get or create SamplingClient from config
    # 2. Convert messages to prompt using renderer
    # 3. Call sampling_client.sample(prompt, params)
    # 4. Parse response
    # 5. Return {:ok, %{content: text, finish_reason: reason, usage: usage}}
  end
end
```

```elixir
defmodule TinkexCookbook.Eval.Runner do
  @moduledoc """
  Orchestrates evaluation using EvalEx and CrucibleHarness.
  """

  alias CrucibleHarness.{Solver, TaskState}
  alias CrucibleHarness.Solver.{Chain, Generate}
  alias EvalEx.{Task, Sample, Scorer}

  def run(task, opts \\ []) do
    # 1. Load samples from task
    # 2. Create generate_fn using TinkexGenerate
    # 3. Create solver chain
    # 4. Run each sample through solver
    # 5. Score outputs
    # 6. Return results
  end
end
```

### TDD Steps
1. Write `test/tinkex_cookbook/eval/tinkex_generate_test.exs`
2. Test `generate/2` with mock SamplingClient
3. Test message conversion
4. Test response parsing
5. Write `test/tinkex_cookbook/eval/runner_test.exs`
6. Test full eval flow with mocks

---

## Task 6: Documentation Updates

### README.md Updates
Add:
- Phase 1 completion status
- Quick start for sl_basic
- Evaluation stack usage
- Dependency list

### AGENTS.md Updates
Add:
- Llama3 renderer documentation
- Dataset builder usage
- Training integration guide
- Eval adapter usage

### Files to Update
```
/home/home/p/g/North-Shore-AI/tinkex_cookbook/README.md
/home/home/p/g/North-Shore-AI/tinkex_cookbook/AGENTS.md
/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251221/PHASE1_FOUNDATION_SLICE.md
/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251221/COOKBOOK_CORE_FOUNDATION.md
```

---

## TDD Workflow (Required)

For EACH task, follow this exact loop:

```
1. Create test file first
2. Write failing tests that specify behavior
3. Run tests - verify they fail for the right reason
4. Implement minimal code to pass tests
5. Run tests - verify they pass
6. Refactor if needed (tests must still pass)
7. Run full quality check before moving on
```

### Quality Check Command
Run after EACH task:

```bash
cd /home/home/p/g/North-Shore-AI/tinkex_cookbook

# Format
mix format

# Compile with warnings as errors
mix compile --warnings-as-errors

# Run tests
mix test

# Credo
mix credo --strict

# Dialyzer
mix dialyzer
```

**ALL must pass with ZERO warnings before proceeding.**

---

## Constraints (Must Follow)

1. **No ad-hoc structs** - Use ChzEx.Schema for configs
2. **No network in tests** - Mock all external calls
3. **No atom from user input** - CLI keys stay strings
4. **No sleeps in tests** - Use deterministic sync
5. **ASCII only** - No unicode in code
6. **No new deps** - Only use what's in mix.exs

### Approved Dependencies
```elixir
{:tinkex, "~> 0.3.2"}
{:chz_ex, "~> 0.1.2"}
{:crucible_harness, "~> 0.3.1"}
{:eval_ex, "~> 0.1.1"}
{:crucible_datasets, "~> 0.5.1"}
{:hf_datasets_ex, "~> 0.1"}
{:hf_hub, "~> 0.1"}
{:nx, "~> 0.9"}
{:snakebridge, "~> 0.3.0"}
```

---

## File Creation Order

Execute in this order to minimize circular dependencies:

```
1. test/support/mock_tokenizer.ex
2. test/support/mock_tinkex.ex
3. test/tinkex_cookbook/renderers/train_on_what_test.exs
4. test/tinkex_cookbook/renderers/llama3_test.exs
5. lib/tinkex_cookbook/renderers/llama3.ex
6. test/tinkex_cookbook/renderers/renderer_test.exs
7. test/tinkex_cookbook/datasets/no_robots_test.exs
8. lib/tinkex_cookbook/datasets/no_robots.ex
9. test/tinkex_cookbook/supervised/train_test.exs
10. lib/tinkex_cookbook/supervised/train.ex
11. test/tinkex_cookbook/eval/tinkex_generate_test.exs
12. lib/tinkex_cookbook/eval/tinkex_generate.ex
13. test/tinkex_cookbook/eval/runner_test.exs
14. lib/tinkex_cookbook/eval/runner.ex
15. Update lib/tinkex_cookbook/recipes/sl_basic.ex
16. Update README.md
17. Update AGENTS.md
18. Update Phase 1 docs
```

---

## Acceptance Criteria

Phase 1 is complete when:

- [ ] `mix format` - no changes needed
- [ ] `mix compile --warnings-as-errors` - zero warnings
- [ ] `mix test` - all tests pass
- [ ] `mix credo --strict` - no issues
- [ ] `mix dialyzer` - no warnings
- [ ] Llama3 renderer implemented and tested
- [ ] Renderer parity tests pass
- [ ] NoRobots dataset builder works
- [ ] SlBasic can run (with mock clients)
- [ ] TinkexGenerate adapter implemented
- [ ] Eval.Runner can orchestrate evaluations
- [ ] README.md updated with usage
- [ ] AGENTS.md updated with implementation notes
- [ ] Phase 1 docs marked complete

---

## Example Test Structure

```elixir
# test/tinkex_cookbook/renderers/llama3_test.exs
defmodule TinkexCookbook.Renderers.Llama3Test do
  use ExUnit.Case, async: true

  alias TinkexCookbook.Renderers.Llama3
  alias TinkexCookbook.Test.MockTokenizer

  describe "init/1" do
    test "returns state with tokenizer" do
      {:ok, state} = Llama3.init(tokenizer: MockTokenizer)
      assert state.tokenizer == MockTokenizer
    end
  end

  describe "bos_tokens/1" do
    test "returns Llama3 BOS token" do
      {:ok, state} = Llama3.init(tokenizer: MockTokenizer)
      assert Llama3.bos_tokens(state) == [128000]
    end
  end

  describe "render_message/4" do
    test "renders user message with correct format" do
      {:ok, state} = Llama3.init(tokenizer: MockTokenizer)
      message = %{role: "user", content: "Hello"}

      {rendered, _state} = Llama3.render_message(0, message, false, state)

      # Verify prefix contains header tokens
      assert rendered.prefix != nil
      # Verify content contains message
      assert length(rendered.content) > 0
    end
  end
end
```

---

## Forbidden Actions

- Do NOT skip tests
- Do NOT ignore warnings
- Do NOT add dependencies
- Do NOT use Process.sleep in tests
- Do NOT make network calls in tests
- Do NOT leave TODOs in final code
- Do NOT commit code that doesn't compile
- Do NOT commit code with failing tests

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Compiler warnings | 0 |
| Test failures | 0 |
| Credo issues | 0 |
| Dialyzer warnings | 0 |
| Code coverage | >80% |
| New LOC | ~750 |
| Test LOC | ~500 |

---

**Document Status:** Complete
**Created:** 2025-12-23
**For:** Phase 1 Completion Agent
