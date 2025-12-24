# Phase 2 Part 1: Infrastructure Prerequisites

**Created:** 2025-12-24
**Status:** DRAFT - Pending Review
**Scope:** Core infrastructure needed before porting Phase 2 recipes

---

## Executive Summary

Phase 2 targets porting ~8 additional recipes from Python `tinker-cookbook` to Elixir `tinkex_cookbook`. Before the recipes themselves can be ported, shared infrastructure must be built. This document specifies the **prerequisite infrastructure** (Phase 2 Part 1) that unblocks recipe work.

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
- All renderers (`renderers.py` - 1,482 LOC)
- All RL infrastructure (`rl/*.py` - ~2,000 LOC)
- All preference/DPO logic (`preference/*.py` - ~800 LOC)
- All distillation logic (`distillation/*.py` - ~500 LOC)
- Completers (`completers.py` - 119 LOC)
- Checkpoint utilities (`checkpoint_utils.py` - ~200 LOC)
- Display utilities (`display.py`, `utils/format_colorized.py`)

**External dependencies (already have Elixir equivalents):**
- `chz` (config) → ChzEx
- `tinker` (SDK) → Tinkex (100%+ parity)
- `datasets` (HuggingFace) → HfDatasetsEx
- `sympy`, `math-verify`, `pylatexenc` → SnakeBridge (wrapped)

---

## Infrastructure Components

### Tier 1: Core Infrastructure (All Recipes)

#### 1.1 Additional Renderers

**Python source:** `tinker_cookbook/renderers.py`

**Current state:** Only `Llama3Renderer` exists in Elixir.

**Required renderers:**

| Renderer | Python LOC | Priority | Complexity |
|----------|-----------|----------|------------|
| `Qwen3Renderer` | ~170 | HIGH | Medium (thinking modes) |
| `Qwen3DisableThinkingRenderer` | ~30 | HIGH | Low (extends Qwen3) |
| `Qwen3InstructRenderer` | ~40 | MEDIUM | Low |
| `DeepSeekV3Renderer` | ~80 | MEDIUM | Low |
| `DeepSeekV3DisableThinkingRenderer` | ~20 | MEDIUM | Low |
| `RoleColonRenderer` | ~60 | LOW | Low |
| `KimiK2Renderer` | ~240 | LOW | Medium (tool calls) |
| `GptOssRenderer` | ~100 | LOW | Low |

**Elixir location:** `lib/tinkex_cookbook/renderers/`

**Implementation notes:**
- Follow existing `Llama3` pattern with `Renderer` behaviour
- Each renderer implements: `init/1`, `render_message/4`, `bos_tokens/1`, `stop_sequences/1`, `parse_response/2`
- Qwen3 needs thinking block handling (`<think>...</think>`)
- DeepSeek uses special unicode separators (`|` = chr(65372))

**Registry pattern:**
```elixir
# lib/tinkex_cookbook/renderers.ex
defmodule TinkexCookbook.Renderers do
  @renderers %{
    "llama3" => TinkexCookbook.Renderers.Llama3,
    "qwen3" => TinkexCookbook.Renderers.Qwen3,
    "qwen3_disable_thinking" => TinkexCookbook.Renderers.Qwen3DisableThinking,
    "deepseekv3" => TinkexCookbook.Renderers.DeepSeekV3,
    "role_colon" => TinkexCookbook.Renderers.RoleColon
  }

  @spec get(String.t(), map(), keyword()) :: {:ok, Renderer.state()} | {:error, term()}
  def get(name, tokenizer, opts \\ []) do
    case Map.fetch(@renderers, name) do
      {:ok, module} -> module.init(Keyword.put(opts, :tokenizer, tokenizer))
      :error -> {:error, {:unknown_renderer, name}}
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

**Behaviours to implement:**

```elixir
# lib/tinkex_cookbook/completers/token_completer.ex
defmodule TinkexCookbook.Completers.TokenCompleter do
  @type stop_condition :: [String.t()] | [integer()]
  @type tokens_with_logprobs :: %{tokens: [integer()], logprobs: [float()] | nil}

  @callback complete(model_input :: ModelInput.t(), stop :: stop_condition()) ::
    {:ok, tokens_with_logprobs()} | {:error, term()}
end

# lib/tinkex_cookbook/completers/message_completer.ex
defmodule TinkexCookbook.Completers.MessageCompleter do
  @callback complete(messages :: [Message.t()]) :: {:ok, Message.t()} | {:error, term()}
end
```

**Implementations:**
- `TinkexTokenCompleter` - Uses `Tinkex.SamplingClient.sample/3`
- `TinkexMessageCompleter` - Uses renderer + sampling client

#### 1.3 Checkpoint Utilities

**Python source:** `tinker_cookbook/checkpoint_utils.py`

**Functions to port:**

| Function | Purpose |
|----------|---------|
| `save_checkpoint` | Save training state to disk |
| `save_checkpoint_async` | Async version |
| `get_last_checkpoint` | Find most recent checkpoint in log_path |
| `load_checkpoint` | Load training state |

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
# Uses Tinkex.TrainingClient methods:
# - save_weights_for_sampler/2
# - save_state/2
# - load_state/2
# - load_state_with_optimizer/2
```

#### 1.4 LR Scheduling

**Python source:** `tinker_cookbook/utils/lr_scheduling.py`

**Schedules to implement:**

| Schedule | Formula |
|----------|---------|
| `constant` | `lr` |
| `linear` | `lr * (1 - step/total_steps)` |
| `cosine` | `lr * 0.5 * (1 + cos(pi * step/total_steps))` |
| `warmup_linear` | Warmup then linear decay |
| `warmup_cosine` | Warmup then cosine decay |

**Elixir location:** `lib/tinkex_cookbook/utils/lr_scheduling.ex`

**Note:** We partially have this in `supervised/train.ex` as `compute_lr/4`. May need to extract and extend.

---

### Tier 2: RL Infrastructure

**Required by:** `rl_basic`, `rl_loop`, `code_rl`

#### 2.1 RL Types

**Python source:** `tinker_cookbook/rl/types.py`

**Core types to port:**

```elixir
# lib/tinkex_cookbook/rl/types.ex

defmodule TinkexCookbook.RL.Types do
  @type action :: [integer()]
  @type observation :: ModelInput.t()
  @type logprobs :: [float()]
  @type metrics :: %{String.t() => number()}
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
    trajectories: [Trajectory.t()],
    final_rewards: [float()],
    metrics: [map()]
  }
  defstruct [:trajectories, :final_rewards, :metrics]

  def get_total_rewards(%__MODULE__{} = group) do
    Enum.zip(group.trajectories, group.final_rewards)
    |> Enum.map(fn {traj, final_reward} ->
      step_rewards = Enum.reduce(traj.transitions, 0.0, & &1.reward + &2)
      step_rewards + final_reward
    end)
  end
end
```

**Behaviours:**

```elixir
# lib/tinkex_cookbook/rl/env.ex
defmodule TinkexCookbook.RL.Env do
  @doc "Single-use environment that agent interacts with."

  @callback initial_observation() :: {:ok, {observation(), stop_condition()}} | {:error, term()}
  @callback step(action()) :: {:ok, StepResult.t()} | {:error, term()}
end

# lib/tinkex_cookbook/rl/env_group_builder.ex
defmodule TinkexCookbook.RL.EnvGroupBuilder do
  @doc "Builds a group of environments for batch RL (GRPO reward centering)."

  @callback make_envs() :: {:ok, [env :: struct()]} | {:error, term()}
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

#### 2.2 Rollouts

**Python source:** `tinker_cookbook/rl/rollouts.py`

**Functions:**

```elixir
# lib/tinkex_cookbook/rl/rollouts.ex

@doc "Run a single rollout in an environment until episode_done."
@spec do_single_rollout(TokenCompleter.t(), Env.t()) :: {:ok, Trajectory.t()} | {:error, term()}

@doc "Run rollouts for all envs in a group, then compute group rewards."
@spec do_group_rollout(EnvGroupBuilder.t(), TokenCompleter.t()) ::
  {:ok, TrajectoryGroup.t()} | {:error, term()}
```

**Async strategy:**
```elixir
def do_group_rollout(builder, policy) do
  {:ok, envs} = builder.make_envs()

  # Parallel rollouts using Task.async_stream
  trajectories =
    envs
    |> Task.async_stream(&do_single_rollout(policy, &1), timeout: :infinity)
    |> Enum.map(fn {:ok, {:ok, traj}} -> traj end)

  # Compute group rewards
  rewards_and_metrics = builder.compute_group_rewards(trajectories, envs)
  {final_rewards, metrics} = Enum.unzip(rewards_and_metrics)

  {:ok, %TrajectoryGroup{
    trajectories: trajectories,
    final_rewards: final_rewards,
    metrics: metrics
  }}
end
```

#### 2.3 RL Data Processing

**Python source:** `tinker_cookbook/rl/data_processing.py`

**Functions:**

| Function | Purpose |
|----------|---------|
| `compute_advantages` | Center rewards within groups (GRPO) |
| `assemble_training_data` | Convert trajectories to Datums |
| `remove_constant_reward_groups` | Filter groups where all rewards equal |

**Key algorithm - advantage computation:**
```python
# Per-group advantage centering (GRPO)
for group in trajectory_groups:
    rewards = group.get_total_rewards()
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards) + 1e-8
    advantages = [(r - mean_reward) / std_reward for r in rewards]
```

#### 2.4 RL Training Core

**Python source:** `tinker_cookbook/rl/train.py` (partial - ~400 LOC)

**Core functions:**

```elixir
# lib/tinkex_cookbook/rl/train.ex

@doc "Single training step on collected trajectories."
@spec train_step(
  data :: [Datum.t()],
  training_client :: Tinkex.TrainingClient.t(),
  learning_rate :: float(),
  num_substeps :: integer(),
  loss_fn :: atom()
) :: {:ok, [Nx.Tensor.t()]} | {:error, term()}

@doc "Sync on-policy training loop."
@spec do_sync_training(config :: Config.t(), ...) :: :ok | {:error, term()}
```

**Loss functions used:**
- `:importance_sampling` - Default for RL
- `:ppo` - PPO clipped objective

---

### Tier 3: Preference/DPO Infrastructure

**Required by:** `preference/dpo`

#### 3.1 DPO Types

**Python source:** `tinker_cookbook/preference/types.py`, `preference/dpo_datasets.py`

**Key types:**
```elixir
# Preference pairs: chosen/rejected datums interleaved
# data[0] = chosen, data[1] = rejected, data[2] = chosen, ...

defmodule TinkexCookbook.Preference.Types do
  @type preference_pair :: {chosen :: Datum.t(), rejected :: Datum.t()}
end
```

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

**Reference model handling:**
- Create reference sampling client from initial weights
- Use `compute_logprobs/1` to get reference log probs
- Reference client is frozen during training

---

### Tier 4: Distillation Infrastructure

**Required by:** `prompt_distillation`, `on_policy_distillation`

#### 4.1 Distillation Types

**Python source:** `tinker_cookbook/distillation/datasets.py`

**Key concept:** Teacher model generates outputs, student learns to match.

```elixir
defmodule TinkexCookbook.Distillation.Types do
  @type distillation_datum :: %{
    prompt: ModelInput.t(),
    teacher_output: [integer()],
    teacher_logprobs: [float()]
  }
end
```

#### 4.2 On-Policy Distillation

**Python source:** `tinker_cookbook/distillation/train_on_policy.py`

**Flow:**
1. Sample from teacher to get outputs + logprobs
2. Create datums with teacher outputs as targets
3. Train student on these datums
4. Repeat (on-policy = fresh teacher samples each batch)

---

## Async Patterns

### Python Pattern → Elixir Equivalent

| Python | Elixir |
|--------|--------|
| `asyncio.gather(*tasks)` | `Task.async_stream` or `Task.await_many` |
| `asyncio.create_task(coro, name=...)` | `Task.async(fn -> ... end)` |
| `asyncio.Queue()` | GenServer with `:queue` or simple list |
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
  |> Task.async_stream(&do_group_rollout(&1, sampling_client), timeout: :infinity)
  |> Enum.map(fn {:ok, result} -> result end)
```

---

## Implementation Order

```
Week 1-2: Tier 1 (Core)
├── 1.1 Qwen3Renderer + variants
├── 1.2 DeepSeekV3Renderer + variants
├── 1.3 RoleColonRenderer
├── 1.4 Renderer registry module
├── 1.5 Completers (TokenCompleter, MessageCompleter)
└── 1.6 Checkpoint utilities

Week 3-4: Tier 2 (RL)
├── 2.1 RL Types (structs + behaviours)
├── 2.2 Rollouts module
├── 2.3 RL Data Processing
└── 2.4 RL Training Core

Week 5: Tier 3 (DPO)
├── 3.1 DPO Types
└── 3.2 DPO Training

Week 6: Tier 4 (Distillation)
├── 4.1 Distillation Types
└── 4.2 On-Policy Distillation
```

---

## Testing Strategy

### Unit Tests (No Network)

- Renderer tests: Port from `test_renderers.py`
- Type tests: Struct construction, validation
- Data processing tests: Advantage computation, datum assembly

### Integration Tests (Mock Tinkex)

- Rollout tests with mock TokenCompleter
- Training step tests with mock TrainingClient
- Checkpoint save/load roundtrip

### Parity Tests

For each component, create parity scripts similar to Phase 1:
- Dump Python outputs to JSON
- Compare Elixir outputs
- Verify exact match (or acceptable tolerance for floats)

---

## Open Questions

1. **VL (Vision-Language) Renderers:** Should we port `Qwen3VLRenderer` now or defer to Phase 3?

2. **Tool Calling:** `KimiK2Renderer` has complex tool call parsing. Priority?

3. **Async RL Training:** The streaming minibatch pattern in `rl/train.py` is complex. Start with sync-only?

4. **Tinkex `forward_backward_custom`:** DPO uses custom loss functions. Verify Tinkex supports this.

---

## File Manifest

### To Create

```
lib/tinkex_cookbook/
├── renderers/
│   ├── qwen3.ex                    # Qwen3Renderer + variants
│   ├── deepseek_v3.ex              # DeepSeekV3Renderer + variants
│   └── role_colon.ex               # RoleColonRenderer
├── renderers.ex                     # Registry module
├── completers/
│   ├── token_completer.ex          # Behaviour
│   ├── message_completer.ex        # Behaviour
│   ├── tinkex_token_completer.ex   # Implementation
│   └── tinkex_message_completer.ex # Implementation
├── utils/
│   ├── checkpoint.ex               # Checkpoint utilities
│   └── lr_scheduling.ex            # LR schedules (extend existing)
├── rl/
│   ├── types.ex                    # All RL types
│   ├── env.ex                      # Env behaviour
│   ├── env_group_builder.ex        # EnvGroupBuilder behaviour
│   ├── rl_dataset.ex               # RLDataset behaviour
│   ├── rollouts.ex                 # Rollout functions
│   ├── data_processing.ex          # Advantage computation, etc.
│   └── train.ex                    # RL training core
├── preference/
│   ├── types.ex                    # Preference types
│   └── dpo_train.ex                # DPO training
└── distillation/
    ├── types.ex                    # Distillation types
    └── on_policy.ex                # On-policy distillation
```

### Python Source Reference

```
tinker-cookbook/tinker_cookbook/
├── renderers.py                    # 1,482 LOC - All renderers
├── completers.py                   # 119 LOC - Completers
├── checkpoint_utils.py             # ~200 LOC - Checkpointing
├── rl/
│   ├── types.py                    # 160 LOC - RL types
│   ├── rollouts.py                 # ~150 LOC - Rollout logic
│   ├── data_processing.py          # ~200 LOC - Data processing
│   ├── train.py                    # ~1,140 LOC - Training loops
│   └── metrics.py                  # ~150 LOC - RL metrics
├── preference/
│   ├── types.py                    # ~50 LOC
│   ├── dpo_datasets.py             # ~200 LOC
│   └── train_dpo.py                # ~400 LOC
└── distillation/
    ├── datasets.py                 # ~150 LOC
    └── train_on_policy.py          # ~350 LOC
```

---

## Acceptance Criteria

Phase 2 Part 1 is complete when:

1. **Renderers:** Qwen3, DeepSeek, RoleColon pass parity tests
2. **Completers:** TokenCompleter and MessageCompleter work with Tinkex
3. **Checkpoint:** Save/load/resume works for training loops
4. **RL Types:** All structs and behaviours defined
5. **Rollouts:** `do_group_rollout` produces correct trajectories
6. **RL Train:** Basic sync training loop runs
7. **DPO:** `compute_dpo_loss` matches Python output
8. **Tests:** 50+ new tests, all passing
