# Phase 2 Part 1: Infrastructure Prerequisites

**Created:** 2025-12-24
**Status:** IMPLEMENTED (2025-12-24)
**Scope:** Core infrastructure needed before porting Phase 2 recipes

---

## Executive Summary

Phase 2 targets porting ~8 additional recipes from Python `tinker-cookbook` to Elixir `tinkex_cookbook`. Before the recipes themselves can be ported, shared infrastructure must be built. This document specifies the **prerequisite infrastructure** (Phase 2 Part 1) that unblocks recipe work.

### Implementation Status (Elixir)

Phase 2 Part 1 infrastructure has been implemented with tests in place. Highlights:
- Tool-calling framework (`renderers/tool_calls.ex`) with encode/decode tests
- Renderers: Qwen3 (+ disable thinking, instruct), DeepSeekV3 (+ disable thinking), KimiK2, GptOss, RoleColon
- Completers (token/message) with Tinkex-backed implementations
- Checkpoint utilities, LR scheduling, misc utils, logtree/trace/display helpers
- RL core: types, envs, rollouts, data_processing, metrics/metric_util, sync+async training
- Preference + DPO: comparison types, preference datasets/envs, DPO datasets + training loop
- Distillation: prompt-only datasets + on-policy distillation training with teacher KL penalty
<!-- REVIEW: Updated status to reflect completed infrastructure implementation -->

### Phase 2 Recipe Targets

| Recipe | Python Location | Dependencies |
|--------|-----------------|--------------|
| sl_loop | `recipes/sl_loop.py` | Renderers, Checkpoint |
| rl_basic | `recipes/rl_basic.py` | RL Types, Rollouts, RL Train |
| rl_loop | `recipes/rl_loop.py` | RL Types, Rollouts, RL Train |
| chat_sl | `recipes/chat_sl/train.py` | Renderers, Chat Datasets |
| preference/dpo | `recipes/preference/dpo/train.py` | DPO Types, DPO Train |
| code_rl | `recipes/code_rl/train.py` | RL Types, Code Env |
| prompt_distillation | `recipes/prompt_distillation/train.py` | Distillation infra |
| on_policy_distillation | `recipes/distillation/on_policy_distillation.py` | Distillation infra |

---

## Source Analysis

### What Lives Where

**In tinker-cookbook (must port):**
- All renderers (`renderers.py` - 1,481 LOC)
- All RL infrastructure (`rl/*.py` - 2,390 LOC, incl. `problem_env.py`, `preference_envs.py`, `metric_util.py`, `play_w_env.py`)
- All preference/DPO logic (`preference/*.py` - 867 LOC, incl. `preference_datasets.py`, `comparison_policy_evaluator.py`)
- All distillation logic (`distillation/*.py` - 739 LOC)
- Completers (`completers.py` - 118 LOC)
- Checkpoint utilities (`checkpoint_utils.py` - 109 LOC)
- Display utilities (`display.py` - 46 LOC, `utils/format_colorized.py` - 50 LOC)
<!-- REVIEW: Updated LOC counts and module list based on wc -l across Python sources -->

**External dependencies (already have Elixir equivalents):**
- `chz` (config) -> ChzEx
- `tinker` (SDK) -> Tinkex (100%+ parity)
- `datasets` (HuggingFace) -> HfDatasetsEx
- `sympy`, `math-verify`, `pylatexenc` -> SnakeBridge (wrapped)

### Existing Elixir Patterns to Reuse

- `lib/tinkex_cookbook/renderers/renderer.ex` and `lib/tinkex_cookbook/renderers/llama3.ex` define the stateful renderer pattern
- `lib/tinkex_cookbook/supervised/common.ex` provides rightshift/leftshift helpers used by RL data processing and DPO
- `lib/tinkex_cookbook/supervised/dataset.ex` already implements PCG64 shuffle (Python parity)
- `lib/tinkex_cookbook/types/*.ex` (ModelInput, Datum, TensorData) can be reused across RL/DPO/distillation
- `lib/tinkex_cookbook/recipes/sl_basic.ex` shows Task-based `forward_backward` + `optim_step` usage
- `lib/tinkex_cookbook/eval/tinkex_generate.ex` shows sampling client wiring for message prompts
<!-- REVIEW: Existing Elixir patterns verified in lib/tinkex_cookbook -->

---

## Infrastructure Components

### Tier 1: Core Infrastructure (All Recipes)

#### 1.1 Additional Renderers

**Python source:** `tinker_cookbook/renderers.py`

**Current state:** Only `Llama3Renderer` exists in Elixir.

**Required renderers (Phase 2 Part 1):**

| Renderer | Python LOC | Priority | Complexity |
|----------|-----------|----------|------------|
| `Qwen3Renderer` | 168 | HIGH | Medium (thinking + tool calls) |
| `Qwen3DisableThinkingRenderer` | 32 | HIGH | Low (extends Qwen3) |
| `Qwen3InstructRenderer` | 93 | MEDIUM | Low (no `<think>` tags) |
| `DeepSeekV3Renderer` | 55 | MEDIUM | Low (role tokens, no system) |
| `DeepSeekV3DisableThinkingRenderer` | 24 | MEDIUM | Low |
| `RoleColonRenderer` | 62 | LOW | Low |
| `KimiK2Renderer` | 246 | LOW | Medium (tool calls + thinking) |
| `GptOssRenderer` | 143 | LOW | Medium (system prompt + channels) |
<!-- REVIEW: Updated per-renderer LOC from renderers.py -->

**Deferred renderers (Phase 3):**
- `Qwen3VLRenderer` (109 LOC) - requires image processor + image chunk handling
- `Qwen3VLInstructRenderer` (52 LOC) - VL variant without `<think>`
<!-- REVIEW: VL renderers deferred per Phase 2 decision -->

**Elixir location:** `lib/tinkex_cookbook/renderers/`

**Implementation notes:**
- Follow `TinkexCookbook.Renderers.Renderer` (stateful `init/1`, `render_message/4`, `bos_tokens/1`, `stop_sequences/1`, optional `parse_response/2`)
- Shared helpers to extract: `parse_response_for_stop_token`, `tokens_weights_from_strings_weights`, tool call encoder/decoder (framework-level)
- `model_info.get_recommended_renderer_name` returns `qwen3_vl`/`qwen3_vl_instruct` for VL models; Phase 2 defers VL renderers, so Phase 2 recipes should avoid VL models or document limitations
- Qwen3: `strip_thinking_from_history` defaults to true (strip `<think>...</think>` for historical assistant turns); adds `<think>\n` for last assistant if missing; tool role renders as `user`, tool responses wrapped in `<tool_response>`; `<tool_call>...</tool_call>` blocks are parsed into `tool_calls`
- Qwen3DisableThinking: injects `<think>\n\n</think>\n\n` and prepends same to generation prefill; Qwen3Instruct omits `<think>` entirely
- Qwen3VL: deferred to Phase 3 (requires `image_processor` + `ImageChunk` handling)
- DeepSeek: special separator is fullwidth vertical line `chr(65372)` (U+FF5C), system role is unsupported unless `system_role_as_user`, end token only appended for assistant messages
- RoleColon stop sequences are strings (`"\n\nUser:"`), not token IDs
- KimiK2: adds default system prompt if missing, encodes tool calls in `<|tool_calls_section_begin|>`; parses `<think>` and tool calls from response
- GptOss: uses `<|return|>` as stop token and `<|end|>` for non-last messages; optional system prompt with `reasoning_effort`; analysis channel only included when `thinking` is present and `is_last`
<!-- REVIEW: Expanded renderer behavior notes based on renderers.py details -->

**Registry pattern:**
```elixir
# lib/tinkex_cookbook/renderers.ex
defmodule TinkexCookbook.Renderers do
  @renderers %{
    "llama3" => TinkexCookbook.Renderers.Llama3,
    "qwen3" => TinkexCookbook.Renderers.Qwen3,
    "qwen3_disable_thinking" => TinkexCookbook.Renderers.Qwen3DisableThinking,
    "qwen3_instruct" => TinkexCookbook.Renderers.Qwen3Instruct,
    # Deferred to Phase 3:
    # "qwen3_vl" => TinkexCookbook.Renderers.Qwen3VL,
    # "qwen3_vl_instruct" => TinkexCookbook.Renderers.Qwen3VLInstruct,
    "deepseekv3" => TinkexCookbook.Renderers.DeepSeekV3,
    "deepseekv3_disable_thinking" => TinkexCookbook.Renderers.DeepSeekV3DisableThinking,
    "kimi_k2" => TinkexCookbook.Renderers.KimiK2,
    "gpt_oss_no_sysprompt" => {TinkexCookbook.Renderers.GptOss, [use_system_prompt: false]},
    "gpt_oss_low_reasoning" => {TinkexCookbook.Renderers.GptOss,
                                [use_system_prompt: true, reasoning_effort: "low"]},
    "gpt_oss_medium_reasoning" => {TinkexCookbook.Renderers.GptOss,
                                   [use_system_prompt: true, reasoning_effort: "medium"]},
    "gpt_oss_high_reasoning" => {TinkexCookbook.Renderers.GptOss,
                                 [use_system_prompt: true, reasoning_effort: "high"]},
    "role_colon" => TinkexCookbook.Renderers.RoleColon
  }

  @spec get(String.t(), map(), keyword()) :: {:ok, Renderer.state()} | {:error, term()}
  def get(name, tokenizer, opts \\ []) do
    case Map.fetch(@renderers, name) do
      {:ok, {module, extra_opts}} ->
        module.init([{:tokenizer, tokenizer} | extra_opts] ++ opts)

      {:ok, module} ->
        module.init([{:tokenizer, tokenizer} | opts])

      :error ->
        {:error, {:unknown_renderer, name}}
    end
  end

  def supported_renderers, do: Map.keys(@renderers)
end
```

#### 1.2 Completers Module

**Python source:** `tinker_cookbook/completers.py`

**Purpose:** Abstraction over sampling - TokenCompleter works on tokens (for RL), MessageCompleter works on messages (for evals).

**Types:**
```python
StopCondition = list[str] | list[int]

@dataclass
class TokensWithLogprobs:
    tokens: list[int]
    maybe_logprobs: list[float] | None
```
**Note:** `TokensWithLogprobs.logprobs` raises if `maybe_logprobs` is `None` (RL requires logprobs).

**Behaviours to implement:**

```elixir
# lib/tinkex_cookbook/completers/token_completer.ex
defmodule TinkexCookbook.Completers.TokenCompleter do
  @type stop_condition :: [String.t()] | [integer()]
  @type tokens_with_logprobs :: %{tokens: [integer()], maybe_logprobs: [float()] | nil}

  @callback complete(model_input :: ModelInput.t(), stop :: stop_condition()) ::
    {:ok, tokens_with_logprobs()} | {:error, term()}
end

# lib/tinkex_cookbook/completers/message_completer.ex
defmodule TinkexCookbook.Completers.MessageCompleter do
  @callback complete(messages :: [Message.t()]) :: {:ok, Message.t()} | {:error, term()}
end
```

**Concurrency note:** Python completers are `async def __call__`. In Elixir, either return `{:ok, Task.t()}` and let callers `Task.await/2`, or make `complete/2` synchronous and await internally to match current `sl_basic` usage patterns.

**Implementations:**
- `TinkexTokenCompleter` - Uses `Tinkex.SamplingClient.sample/3` (awaits Task; asserts logprobs present for RL)
- `TinkexMessageCompleter` - Uses renderer + sampling client; defaults stop to renderer; returns assistant content only
<!-- REVIEW: TokensWithLogprobs uses maybe_logprobs in Python; MessageCompleter drops tool_calls -->

#### 1.3 Checkpoint Utilities

**Python source:** `tinker_cookbook/checkpoint_utils.py`

**Functions to port:**

| Function | Purpose |
|----------|---------|
| `load_checkpoints_file` | Read `checkpoints.jsonl` entries |
| `get_last_checkpoint` | Find most recent checkpoint in log_path (filters by `required_key`) |
| `save_checkpoint_async` | Async save of state/sampler (writes `checkpoints.jsonl`) |
| `save_checkpoint` | Sync wrapper (uses `asyncio.run`) |
<!-- REVIEW: Python checkpoint_utils has no load_checkpoint; added load_checkpoints_file -->

**Elixir location:** `lib/tinkex_cookbook/utils/checkpoint.ex`

**Key types:**
```elixir
@type checkpoint_kind :: :sampler | :state | :both
@type loop_state :: %{optional(:epoch) => integer(), optional(:batch) => integer()}
@type checkpoint_result :: %{
  optional(:sampler_path) => String.t(),
  optional(:state_path) => String.t()
}
```

**Integration with Tinkex:**
```elixir
# Uses Tinkex.TrainingClient methods (Task-based in Elixir):
# - save_weights_for_sampler/3
# - save_state/3
# - load_state/3
# - load_state_with_optimizer/3
```

#### 1.4 LR Scheduling

**Python source:** `tinker_cookbook/utils/lr_scheduling.py`

**Schedules to implement:**

| Schedule | Formula |
|----------|---------|
| `constant` | `lr` |
| `linear` | `lr * (1 - step/total_steps)` |
| `cosine` | `lr * 0.5 * (1 + cos(pi * step/total_steps))` |
<!-- REVIEW: Python only implements linear/cosine/constant in utils/lr_scheduling.py -->

**Elixir location:** `lib/tinkex_cookbook/utils/lr_scheduling.ex`

**Note:** We partially have this in `supervised/train.ex` as `compute_lr/4`. In Python, DPO uses `compute_schedule_lr_multiplier`, so the Elixir port should expose an equivalent multiplier function.

#### 1.5 Tool Calling Framework

**Purpose:** shared encode/decode helpers for tool calls that multiple renderers can use.

**Python sources:** `tinker_cookbook/renderers.py` (Qwen3 + KimiK2 tool tags)

**Scope:**
- Encode tool call payloads to tagged blocks for Qwen3 (`<tool_call>...</tool_call>`)
- Decode tool call blocks to `ToolCall` structs with JSON arguments
- Reusable helpers for KimiK2 tool call sections
- Unit tests for round-trip correctness and invalid JSON handling

#### 1.6 Tinkex API Surface (Verified)

**Verified in `deps/tinkex`:**
- `Tinkex.TrainingClient.forward_backward/4` -> `{:ok, Task.t()}`
- `Tinkex.TrainingClient.forward_backward_custom/4` -> `{:ok, Task.t()}`
- `Tinkex.TrainingClient.optim_step/3` -> `{:ok, Task.t()}`
- `Tinkex.TrainingClient.save_weights_for_sampler/3` -> `{:ok, Task.t()}`
- `Tinkex.TrainingClient.save_weights_and_get_sampling_client/2` -> `{:ok, Task.t()}` (returns sampling client)
- `Tinkex.TrainingClient.save_state/3`, `load_state/3`, `load_state_with_optimizer/3` -> `{:ok, Task.t()}`
- `Tinkex.SamplingClient.compute_logprobs/3` -> `{:ok, Task.t()}`
<!-- REVIEW: Verified function existence in deps/tinkex/lib/tinkex/*.ex -->

---

### Tier 2: RL Infrastructure

**Required by:** `rl_basic`, `rl_loop`, `code_rl`

#### 2.1 RL Types

**Python source:** `tinker_cookbook/rl/types.py`

**Core types to port (Python semantics):**

```elixir
# lib/tinkex_cookbook/rl/types.ex

defmodule TinkexCookbook.RL.Types do
  @type action :: [integer()]
  @type observation :: ModelInput.t()
  @type logprobs :: [float()]
  @type metrics :: %{String.t() => float() | integer()}
end

defmodule TinkexCookbook.RL.StepResult do
  @type t :: %__MODULE__{
    reward: float(),
    episode_done: boolean(),
    next_observation: ModelInput.t(),
    next_stop_condition: stop_condition(),
    metrics: map()
  }
  defstruct [:reward, :episode_done, :next_observation, :next_stop_condition, metrics: %{}]
end

defmodule TinkexCookbook.RL.Transition do
  @type t :: %__MODULE__{
    ob: ModelInput.t(),
    ac: tokens_with_logprobs(),
    reward: float(),
    episode_done: boolean(),
    metrics: map()
  }
  defstruct [:ob, :ac, :reward, :episode_done, metrics: %{}]
end

defmodule TinkexCookbook.RL.Trajectory do
  @type t :: %__MODULE__{
    transitions: [Transition.t()],
    final_ob: ModelInput.t()
  }
  defstruct [:transitions, :final_ob]
end

defmodule TinkexCookbook.RL.TrajectoryGroup do
  @type t :: %__MODULE__{
    trajectories_G: [Trajectory.t()],
    final_rewards_G: [float()],
    metrics_G: [map()]
  }
  defstruct [:trajectories_G, :final_rewards_G, :metrics_G]

  def get_total_rewards(%__MODULE__{} = group) do
    # Python uses strict zip (safezip) over trajectories_G and final_rewards_G.
    Enum.zip(group.trajectories_G, group.final_rewards_G)
    |> Enum.map(fn {traj, final_reward} ->
      step_rewards = Enum.reduce(traj.transitions, 0.0, & &1.reward + &2)
      step_rewards + final_reward
    end)
  end
end
```
<!-- REVIEW: TrajectoryGroup fields in Python are trajectories_G/final_rewards_G/metrics_G -->

**Behaviours:**

```elixir
# lib/tinkex_cookbook/rl/env.ex
defmodule TinkexCookbook.RL.Env do
  @doc "Single-use environment that agent interacts with."

  @callback initial_observation() :: {observation(), stop_condition()}
  @callback step(action()) :: StepResult.t()
end

# lib/tinkex_cookbook/rl/env_group_builder.ex
defmodule TinkexCookbook.RL.EnvGroupBuilder do
  @doc "Builds a group of environments for batch RL (GRPO reward centering)."

  @callback make_envs() :: [env :: struct()]
  @callback compute_group_rewards(trajectories :: [Trajectory.t()], envs :: [struct()]) ::
    [{float(), map()}]
  @callback logging_tags() :: [String.t()]
end

# lib/tinkex_cookbook/rl/rl_dataset.ex
defmodule TinkexCookbook.RL.RLDataset do
  @callback get_batch(index :: non_neg_integer()) :: [EnvGroupBuilder.t()]
  @callback length() :: non_neg_integer()
end
```
**Notes:**
- `compute_group_rewards/2` is optional in Python (`EnvGroupBuilder` provides a default of zero rewards).
- Python includes `RLDatasetBuilder` (chz class) that returns `{train_dataset, maybe_test_dataset}`; mirror this in Elixir with `ChzEx` configs.
<!-- REVIEW: Env/EnvGroupBuilder are async in Python and return bare values, not {:ok, ...} tuples -->

#### 2.2 Rollouts

**Python source:** `tinker_cookbook/rl/rollouts.py`

**Functions:**

```elixir
# lib/tinkex_cookbook/rl/rollouts.ex

@doc "Run a single rollout in an environment until episode_done."
@spec do_single_rollout(TokenCompleter.t(), Env.t()) :: Trajectory.t()

@doc "Run rollouts for all envs in a group, then compute group rewards."
@spec do_group_rollout(EnvGroupBuilder.t(), TokenCompleter.t()) :: TrajectoryGroup.t()
```

**Async strategy (Python parity):**
```elixir
def do_group_rollout(builder, policy) do
  envs = builder.make_envs()

  # Parallel rollouts using Task.async_stream
  trajectories =
    envs
    |> Task.async_stream(&do_single_rollout(policy, &1), ordered: true, timeout: :infinity)
    |> Enum.map(fn {:ok, traj} -> traj end)

  # Compute group rewards
  rewards_and_metrics = builder.compute_group_rewards(trajectories, envs)
  {final_rewards, metrics} = Enum.unzip(rewards_and_metrics)

  %TrajectoryGroup{
    trajectories_G: trajectories,
    final_rewards_G: final_rewards,
    metrics_G: metrics
  }}
end
```
**Logging note:** Python `do_group_rollout` uses `logtree` to emit per-trajectory tables with final rewards; Elixir port should either implement a minimal logtree or stub these logs behind a feature flag.
<!-- REVIEW: Python rollouts are async and return bare Trajectory/TrajectoryGroup; update Elixir signatures accordingly -->

#### 2.3 RL Data Processing

**Python source:** `tinker_cookbook/rl/data_processing.py`

**Functions:**

| Function | Purpose |
|----------|---------|
| `compute_advantages` | Center rewards within groups (GRPO) |
| `assemble_training_data` | Convert trajectories to Datums |
| `remove_constant_reward_groups` | Filter groups where all rewards equal |

**Data shape notes:**
- `trajectory_to_data` builds `tinker.Datum` with `loss_fn_inputs`: `target_tokens`, `logprobs`, `advantages`, `mask`
- Uses `create_rightshifted_model_input_and_leftshifted_targets` from `supervised.common`
- `assemble_training_data` returns both `data_D` and `metadata_D` (with `group_idx`, `traj_idx`)
- `remove_constant_reward_groups` returns a singleton list if all rewards are uniform (avoids empty batch)

**Key algorithm - advantage computation:**
```python
# Per-group advantage centering (GRPO) - mean only
for group in trajectory_groups:
    rewards = torch.tensor(group.get_total_rewards())
    advantages = rewards - rewards.mean()
```
<!-- REVIEW: Python data_processing.compute_advantages uses mean-centering only (no std) -->

#### 2.4 RL Training Core

**Python source:** `tinker_cookbook/rl/train.py` (1,140 LOC)

**Core functions (Python semantics):**

```elixir
# lib/tinkex_cookbook/rl/train.ex

@doc "Single training step on collected trajectories (pipelines forward_backward + optim_step)."
@spec train_step(
  data :: [Datum.t()],
  training_client :: Tinkex.TrainingClient.t(),
  learning_rate :: float(),
  num_substeps :: pos_integer(),
  loss_fn :: atom()
) :: [Nx.Tensor.t()]

@doc "Sync on-policy training loop."
@spec do_sync_training(config :: Config.t(), ...) :: :ok

@doc "Sync training with minibatch streaming."
@spec do_sync_training_with_stream_minibatch(config :: Config.t(), ...) :: :ok

@doc "Async off-policy training (max_steps_off_policy)."
@spec do_async_training(config :: Config.t(), ...) :: :ok
```

**Additional helpers used by recipes and distillation:**
- `prepare_minibatch/6` (uses `compute_trajectory_metrics`, `compute_advantages`, `assemble_training_data`)
- `compute_full_batch_metrics_and_get_sampling_client/6` (uses `rl.metrics` for KL)
- `do_group_rollout_and_filter_constant_reward/5`
- `save_checkpoint_and_get_sampling_client/5`
<!-- REVIEW: rl/train.py is 1,140 LOC and includes sync, async, and stream-minibatch flows -->

**Loss functions used:**
- `:importance_sampling` - Default for RL
- `:ppo` - PPO clipped objective

**Notes:**
- `train_step` pipelines `forward_backward_async` and `optim_step_async` and returns training logprobs per datum.
- `Config` includes `async_config` and `stream_minibatch_config` for async/streaming; default path is `do_sync_training`.
- `rl.metrics` and `rl.metric_util` are required for KL penalties and trajectory metrics in training and distillation.

#### 2.5 RL Metrics + Evaluators

**Python sources:** `tinker_cookbook/rl/metrics.py`, `tinker_cookbook/rl/metric_util.py`

**Key functions/classes:**
- `compute_kl_sample_train` (sampling vs training logprobs)
- `compute_post_kl` (post-update KL via `compute_logprobs_async`)
- `incorporate_kl_penalty` (adjust advantages in-place)
- `compute_sampling_client_metrics` (async training sampling staleness)
- `compute_trajectory_metrics` + `RLTestSetEvaluator` (test-set metrics for RL/distillation)

#### 2.6 RL Environment Bases

**Python sources:** `tinker_cookbook/rl/problem_env.py`, `tinker_cookbook/rl/preference_envs.py`

**Key components:**
- `ProblemEnv` / `ProblemGroupBuilder` (base env used by math/code RL recipes)
- `PreferenceEnv` / `PairwisePreferenceGroupBuilder` (pairwise preference rewards for RLHF-style recipes)
- `play_w_env.py` (interactive debug runner; optional for Phase 2)
<!-- REVIEW: RL utilities beyond types/rollouts/train are required for recipes and distillation -->

---

### Tier 3: Preference/DPO Infrastructure

**Required by:** `preference/dpo`

#### 3.1 DPO Types

**Python source:** `tinker_cookbook/preference/types.py`, `preference/dpo_datasets.py`

**Key types (Python):**
- `Comparison` (prompt_conversation + completion_A + completion_B)
- `LabeledComparison` (label: "A" | "B" | "Tie", with swap helpers)
- `ComparisonRenderer` / `ComparisonRendererFromChatRenderer` (builds model inputs and weights)
- `PreferenceModel` / `PreferenceModelBuilder` (preference scorer interface)
- `PreferenceModelBuilderFromChatRenderer` (builds a sampling-based preference model)

**Dataset builder:**
- `DPODatasetBuilderFromComparisons` (wraps `ComparisonDatasetBuilder`, yields chosen/rejected datums interleaved)
<!-- REVIEW: preference/types.py defines comparison and preference model interfaces, not just pair tuples -->

#### 3.2 DPO Training

**Python source:** `tinker_cookbook/preference/train_dpo.py`

**Core function:**
```elixir
@doc "Compute DPO loss from log probability ratios."
@spec compute_dpo_loss(
  chosen_logprobs :: [Nx.Tensor.t()],
  rejected_logprobs :: [Nx.Tensor.t()],
  chosen_ref_logprobs :: [Nx.Tensor.t()],
  rejected_ref_logprobs :: [Nx.Tensor.t()],
  dpo_beta :: float()
) :: {Nx.Tensor.t(), metrics :: map()}
```

**DPO loss formula:**
```
loss = -log(sigmoid(beta * (chosen_log_ratio - rejected_log_ratio)))
where log_ratio = policy_logprob - reference_logprob
```

**Schedule note:** DPO uses `compute_schedule_lr_multiplier` with `lr_schedule` in `{linear, cosine, constant}`.

**Reference model handling:**
- `create_dpo_clients` builds a training client and a frozen reference sampling client via `save_weights_and_get_sampling_client("reference")`
- Uses `SamplingClient.compute_logprobs_async` on reconstructed full sequences (append last target token)
- Computes weighted logprobs using `datum.loss_fn_inputs["weights"]`
- DPO step uses `training_client.forward_backward_custom` with a `dpo_loss_fn`, then `optim_step`
<!-- REVIEW: DPO in Python uses forward_backward_custom + reference sampling client from save_weights_and_get_sampling_client -->

**Config notes:**
- `reference_model_name` and `num_replicas` exist in Python config but are unused in `train_dpo.py`

---

### Tier 4: Distillation Infrastructure

**Required by:** `prompt_distillation`, `on_policy_distillation`

#### 4.1 Distillation Types

**Python source:** `tinker_cookbook/distillation/datasets.py`

**Key concept:** On-policy distillation reuses RL infrastructure; no dedicated distillation datum type exists in Python.

**Key types/modules (Python):**
- `TeacherConfig` / `DistillationDatasetConfig`
- `CompositeDataset` (mixes multiple RL datasets by `groups_per_batch`)
- `PromptOnlyEnv`, `PromptOnlyDataset`, `PromptOnlyDatasetBuilder` (prompt-only RL envs)
<!-- REVIEW: distillation/datasets.py defines dataset configs and envs, not a distillation datum -->

#### 4.2 On-Policy Distillation

**Python source:** `tinker_cookbook/distillation/train_on_policy.py`

**Flow (Python):**
1. Sample trajectories with the student policy (RL rollouts)
2. Convert trajectories to datums via `assemble_training_data`
3. Apply KL penalty against teacher logprobs (reverse KL, optional discount)
4. Train student via `train_step` (importance_sampling by default)
5. Repeat with fresh rollouts each batch

**Teacher sampling details:**
- One sampling client per dataset config (`TeacherConfig`)
- `compute_logprobs_async` called per datum to compute reverse KL

---

## Async Patterns

### Python Pattern -> Elixir Equivalent

| Python | Elixir |
|--------|--------|
| `asyncio.gather(*tasks)` | `Task.async_stream(ordered: true)` or `Task.await_many` |
| `asyncio.create_task(coro, name=...)` | `Task.async` or `Task.Supervisor.async_nolink` |
| `asyncio.Queue()` | GenServer with `:queue` or `:queue` in a receive loop |
| `await future.result_async()` | `Task.await(task)` |

### Example: Parallel Rollouts

**Python:**
```python
trajectory_groups_P = await asyncio.gather(
    *[asyncio.create_task(do_group_rollout(...)) for builder in builders]
)
```

**Elixir:**
```elixir
trajectory_groups =
  builders
  |> Task.async_stream(&do_group_rollout(&1, sampling_client),
    ordered: true,
    timeout: :infinity
  )
  |> Enum.map(fn {:ok, result} -> result end)
```
<!-- REVIEW: Python uses asyncio.gather/create_task; keep ordered results for trajectory alignment -->

---

## Implementation Order

```
Week 1-2: Tier 1 (Core)
├── 1.1 Qwen3Renderer + variants (incl. Qwen3Instruct; VL deferred)
├── 1.2 DeepSeekV3Renderer + variants
├── 1.3 RoleColonRenderer
├── 1.4 KimiK2Renderer
├── 1.5 GptOssRenderer variants
├── 1.6 Tool calling framework (shared encode/decode)
├── 1.7 Renderer registry module
├── 1.8 Completers (TokenCompleter, MessageCompleter)
├── 1.9 Checkpoint utilities
└── 1.10 LR scheduling (multiplier function)

Week 3-4: Tier 2 (RL)
├── 2.1 RL Types (structs + behaviours)
├── 2.2 Rollouts module
├── 2.3 RL Data Processing
├── 2.4 RL Metrics + Metric Util
├── 2.5 RL Training Core (sync + async together)
├── 2.6 ProblemEnv + GroupBuilder (shared env base)
└── 2.7 PreferenceEnv + PairwisePreferenceGroupBuilder (used by RLHF-style recipes)

Week 5: Tier 3 (DPO)
├── 3.1 Preference Types + Preference Datasets
├── 3.2 DPO Dataset Builder
└── 3.3 DPO Training

Week 6: Tier 4 (Distillation)
├── 4.1 Distillation Dataset Configs
└── 4.2 On-Policy Distillation
```

---

## Testing Strategy

### Unit Tests (No Network)

- Renderer tests: Port from `test_renderers.py`
- Type tests: Struct construction, validation
- Data processing tests: Advantage computation, datum assembly
- Property tests for renderer invariants (stop token parsing, round-trip content)
- Tool calling tests: JSON payload encoding/decoding + renderer-specific tool tag parsing
- Use `test/support/mock_tokenizer.ex` for deterministic tokenization

### Integration Tests (Mock Tinkex)

- Rollout tests with mock TokenCompleter
- Training step tests with mock TrainingClient
- Checkpoint save/load roundtrip
- Extend `test/support/mock_tinkex.ex` for `compute_logprobs`, `forward_backward_custom`, and Task-returning APIs
<!-- REVIEW: Testing strategy aligned with Phase 1 mocks and TDD requirements -->

### Parity Tests

For each component, create parity scripts similar to Phase 1:
- Dump Python outputs to JSON
- Compare Elixir outputs
- Verify exact match (or acceptable tolerance for floats)

---

## Critical Questions (Answers)

1. **Minimal infrastructure for `rl_basic`:** RL types, rollouts, data_processing, rl/train (sync path + train_step), `rl/problem_env` base, `rl/metric_util` for metrics, TokenCompleter, renderer registry, checkpoint utils, LR scheduling, and misc utilities (`safezip`, `split_list`, `timed`) plus a minimal logtree/trace stub.
2. **Shared vs recipe-specific:** Shared = renderers, completers, checkpoint/lr scheduling, RL core (types/rollouts/data_processing/train/metrics/metric_util/problem_env), preference types/datasets, distillation dataset configs; Recipe-specific = math/code envs, chat dataset builders, specific datasets and prompts.
3. **Async patterns:** Implement sync and async together using shared core functions; use `Task.async_stream(ordered: true)` for rollouts and a GenServer with `:queue` for queue patterns.
4. **Testing infrastructure:** Extend existing mocks (`test/support/mock_tokenizer.ex`, `test/support/mock_tinkex.ex`) to cover `compute_logprobs` and `forward_backward_custom`; add property tests for renderer invariants.
5. **Parity testing approach:** For each module, dump Python outputs (advantages, datums, KL metrics, DPO loss, renderer tokens) and compare against Elixir with deterministic seeds; allow float tolerances where needed.

---

## Decisions (2025-12-24)

1. **VL (Vision-Language) Renderers:** Defer Qwen3VL/Qwen3VLInstruct to Phase 3; Phase 2 recipes should avoid VL models.
2. **Tool Calling:** Implement tool calling at the framework level and reuse it across renderers (Qwen3, KimiK2, future tool-use recipes).
3. **RL Training:** Implement sync and async training together with shared core abstractions; do not phase async later.

---

## File Manifest

### To Create

```
lib/tinkex_cookbook/
├── tokenizer_utils.ex              # get_tokenizer helpers (model_name -> tokenizer)
├── renderers/
│   ├── qwen3.ex                    # Qwen3Renderer + DisableThinking + Instruct
│   ├── deepseek_v3.ex              # DeepSeekV3Renderer + DisableThinking
│   ├── kimi_k2.ex                  # KimiK2Renderer
│   ├── gpt_oss.ex                  # GptOssRenderer (variants via opts)
│   ├── role_colon.ex               # RoleColonRenderer
│   └── tool_calls.ex               # Shared tool call encode/decode helpers
├── renderers.ex                     # Registry module
├── completers/
│   ├── token_completer.ex          # Behaviour
│   ├── message_completer.ex        # Behaviour
│   ├── tinkex_token_completer.ex   # Implementation
│   └── tinkex_message_completer.ex # Implementation
├── utils/
│   ├── checkpoint.ex               # Checkpoint utilities
│   ├── lr_scheduling.ex            # LR schedules (extend existing)
│   ├── misc_utils.ex               # safezip, split_list, timed, all_same
│   ├── logtree.ex                  # Minimal logging helpers (optional stub)
│   ├── logtree_formatters.ex       # ConversationFormatter helpers
│   └── trace.ex                    # Trace helpers (optional stub)
├── rl/
│   ├── types.ex                    # All RL types
│   ├── env.ex                      # Env behaviour
│   ├── env_group_builder.ex        # EnvGroupBuilder behaviour
│   ├── rl_dataset.ex               # RLDataset behaviour
│   ├── rollouts.ex                 # Rollout functions
│   ├── data_processing.ex          # Advantage computation, etc.
│   ├── metrics.ex                  # KL + trajectory metrics helpers
│   ├── metric_util.ex              # RLTestSetEvaluator, compute_trajectory_metrics
│   ├── problem_env.ex              # Base RL env for problem-style tasks
│   ├── preference_envs.ex          # Pairwise preference envs (RLHF-style)
│   └── train.ex                    # RL training core
├── preference/
│   ├── types.ex                    # Preference types
│   ├── preference_datasets.ex      # ComparisonDatasetBuilder
│   ├── dpo_datasets.ex             # DPO dataset builder
│   ├── comparison_policy_evaluator.ex # Preference evaluators
│   └── train_dpo.ex                # DPO training
└── distillation/
    ├── datasets.ex                 # Distillation dataset configs/envs
    └── train_on_policy.ex          # On-policy distillation
```

**Deferred to Phase 3 (not in Part 1 deliverables):**
- `lib/tinkex_cookbook/renderers/qwen3_vl.ex` (Qwen3VLRenderer + Qwen3VLInstructRenderer)

### Python Source Reference

```
tinker-cookbook/tinker_cookbook/
├── renderers.py                    # 1,481 LOC - All renderers
├── completers.py                   # 118 LOC - Completers
├── checkpoint_utils.py             # 109 LOC - Checkpointing
├── tokenizer_utils.py              # 36 LOC - get_tokenizer helpers
├── display.py                      # 46 LOC - Colorized display helpers
├── utils/
│   ├── format_colorized.py         # 50 LOC
│   ├── lr_scheduling.py            # 23 LOC
│   ├── misc_utils.py               # 94 LOC
│   ├── logtree.py                  # 1,017 LOC
│   ├── logtree_formatters.py       # 108 LOC
│   ├── ml_log.py                   # 519 LOC
│   └── trace.py                    # 443 LOC
├── rl/
│   ├── types.py                    # 159 LOC - RL types
│   ├── rollouts.py                 # 81 LOC - Rollout logic
│   ├── data_processing.py          # 207 LOC - Data processing
│   ├── train.py                    # 1,140 LOC - Training loops
│   ├── metrics.py                  # 169 LOC - RL metrics
│   ├── metric_util.py              # 136 LOC - Trajectory metrics + test evaluator
│   ├── problem_env.py              # 103 LOC - Base env
│   ├── preference_envs.py          # 283 LOC - Pairwise preference envs
│   └── play_w_env.py               # 112 LOC - Debug runner
├── preference/
│   ├── types.py                    # 154 LOC
│   ├── preference_datasets.py      # 172 LOC
│   ├── comparison_policy_evaluator.py # 67 LOC
│   ├── dpo_datasets.py             # 77 LOC
│   └── train_dpo.py                # 397 LOC
└── distillation/
    ├── datasets.py                 # 278 LOC
    └── train_on_policy.py          # 461 LOC
```
<!-- REVIEW: Updated Python LOC counts and added missing RL/preference modules -->

---

## Acceptance Criteria

Phase 2 Part 1 is complete when:

1. **Renderers:** Qwen3 (+ variants), DeepSeek, RoleColon, KimiK2, GptOss pass parity tests
2. **Completers:** TokenCompleter and MessageCompleter work with Task-based Tinkex APIs
3. **Checkpoint:** Save/resume via `checkpoints.jsonl` works for training loops
4. **RL Types:** All structs/behaviours + RLDatasetBuilder defined
5. **Rollouts:** `do_group_rollout` produces correct trajectories with ordered results
6. **RL Train:** Sync training loop + KL metrics path run end-to-end
7. **Tool Calling:** Shared tool call encoder/decoder covered by unit tests and used by Qwen3/KimiK2
8. **DPO:** `compute_dpo_loss` + `forward_backward_custom` parity verified
9. **Distillation:** On-policy distillation runs with teacher KL penalty
10. **Tests:** 50+ new tests, all passing

---

## Review Notes (2025-12-24)

### Corrections Made

1. **Status + Scope:** Marked Phase 2 Part 1 as implemented and added a concise infra summary.
2. **Renderer/Tooling Status:** Confirmed tool-call framework + renderer set as implemented (Qwen3/DeepSeek/KimiK2/GptOss/RoleColon).
3. **RL/Preference/Distillation:** Updated to reflect completed RL core + preference/DPO + distillation infra.

### Gaps Identified

1. **Optional RL utility:** `rl/play_w_env` remains unported (optional in Phase 2 Part 1).
2. **Preference evaluator:** `preference/comparison_policy_evaluator` not yet ported (recipe/eval add-on).

### Recommendations

1. Run full quality gates (`mix test`, `mix credo --strict`, `mix dialyzer`) before porting recipes.
2. Revisit `rl/play_w_env` if any Phase 2 recipe requires interactive rollouts.
3. Port comparison-policy evaluator if preference evaluation is needed for DPO runs.
