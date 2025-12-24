# Torch & Transformers: Actual Usage in Tinker-Cookbook

**Date**: 2025-12-20
**Context**: Corrects 08_torch_transformers_axon_mapping.md which incorrectly assumed we were porting model training code.
**Key Finding**: The cookbook is a **CLIENT** library that calls the Tinker API for training. torch/transformers are used for **data preparation and local tensor ops**, NOT model training.

---

## Executive Summary

After analyzing all imports and usages of `torch` and `transformers` across the tinker-cookbook codebase:

**Do we need torch/transformers wrappers in tinkex_cookbook?**

**NO for transformers**: We only need the tokenizer, which tinkex already has via `{:tokenizers, "~> 0.5"}`.

**MINIMAL for torch**: We need basic tensor operations (creation, concatenation, arithmetic) for data preparation. Nx can handle this trivially.

**Training happens server-side via Tinker API**. The cookbook never implements backprop, optimizers, or model weights.

---

## Complete Import Analysis

### Files Importing torch (11 files)

1. `supervised/common.py` - Data conversion utilities
2. `supervised/train.py` - Training orchestration (calls API)
3. `preference/train_dpo.py` - DPO training orchestration (calls API)
4. `preference/types.py` - Data type definitions
5. `distillation/train_on_policy.py` - Distillation orchestration (calls API)
6. `recipes/rl_loop.py` - RL training orchestration (calls API)
7. `renderers.py` - Token/weight tensor creation for prompts
8. `tests/compare_sampling_training_logprobs.py` - Testing utilities
9. `tests/test_renderers.py` - Renderer tests
10. `rl/data_processing.py` - RL data preparation
11. `rl/metrics.py` - RL metric computation
12. `rl/train.py` - RL training orchestration (calls API)

### Files Importing transformers (2 files)

1. `tokenizer_utils.py` - **ONLY** imports `AutoTokenizer` (lazy-loaded)
2. `hyperparam_utils.py` - Imports `AutoConfig` to read model metadata
3. `tests/test_renderers.py` - Test file using tokenizer

---

## Usage Categories

### Category 1: Tokenization (100% replaceable)

**Files**: `tokenizer_utils.py`, `tests/test_renderers.py`

**Python Usage**:
```python
from transformers.models.auto.tokenization_auto import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokens = tokenizer.encode(text, add_special_tokens=False)
text = tokenizer.decode(tokens)
bos_token_str = tokenizer.bos_token
```

**Elixir Equivalent** (already exists in tinkex):
```elixir
# via {:tokenizers, "~> 0.5"}
{:ok, tokenizer} = Tokenizers.Tokenizer.from_pretrained(model_name)
{:ok, encoding} = Tokenizers.encode(tokenizer, text)
tokens = Tokenizers.Encoding.get_ids(encoding)
text = Tokenizers.decode(tokenizer, tokens)
```

**Status**: ✅ Already handled by tinkex's tokenizer dependency

---

### Category 2: Model Metadata (Optional, low priority)

**Files**: `hyperparam_utils.py`

**Python Usage**:
```python
from transformers import AutoConfig
config = AutoConfig.from_pretrained(model_name)
hidden_size = config.hidden_size
```

**Purpose**: Read model config to compute LoRA hyperparameters (learning rate scaling, param counts)

**Elixir Options**:
1. **Option A** (recommended): Hardcode known model configs (like they do for Llama-3 models in the Python code)
2. **Option B**: HTTP fetch HF config.json and parse with Jason
3. **Option C**: Skip entirely - these are heuristics, not requirements

**Status**: ⚠️ Low priority - can hardcode or skip for MVP

---

### Category 3: Local Tensor Operations (Simple Nx ops)

**Files**: All training/RL modules

**Python torch operations used**:

#### 3a. Tensor Creation
```python
# Python
torch.tensor([1, 2, 3])
torch.ones_like(tokens)
torch.full((len(chunk),), weight_value)
```

```elixir
# Elixir/Nx equivalent
Nx.tensor([1, 2, 3])
Nx.broadcast(1, Nx.shape(tokens))
Nx.broadcast(weight_value, {Nx.size(chunk)})
```

#### 3b. Tensor Concatenation
```python
# Python
torch.cat([tensor1, tensor2])
torch.stack([tensor1, tensor2])
```

```elixir
# Elixir/Nx equivalent
Nx.concatenate([tensor1, tensor2])
Nx.stack([tensor1, tensor2])
```

#### 3c. Arithmetic/Reduction
```python
# Python
logprobs.dot(weights)
weights.sum()
rewards.mean()
advantages = rewards - rewards.mean()
```

```elixir
# Elixir/Nx equivalent
Nx.dot(logprobs, weights)
Nx.sum(weights)
Nx.mean(rewards)
advantages = Nx.subtract(rewards, Nx.mean(rewards))
```

#### 3d. Type Assertions
```python
# Python
assert tokens.dtype == torch.int64
```

```elixir
# Elixir/Nx equivalent
# Type is explicit: Nx.tensor([1,2,3], type: :s64)
# Or check: Nx.type(tokens) == {:s, 64}
```

**Status**: ✅ Trivial Nx equivalents

---

### Category 4: API Bridge (tinker.TensorData interop)

**Critical Pattern**: Converting between cookbook tensors and Tinker API payloads

**Python Implementation**:
```python
import tinker
import torch

# Sending data TO Tinker API
tinker.TensorData.from_torch(torch.tensor([1.0, 2.0, 3.0]))
# Returns: TensorData(data=[1.0, 2.0, 3.0], dtype="float32", shape=[3])

# Receiving data FROM Tinker API
logprobs_tensor = tinker_response.loss_fn_inputs["logprobs"].to_torch()
# Returns: torch.Tensor
```

**What is TensorData?** (From Python SDK)
```python
@dataclass
class TensorData:
    data: list  # Flattened list of numbers
    dtype: str  # "float32", "int64", etc.
    shape: list[int]  # [batch, seq_len, ...]

    @classmethod
    def from_torch(cls, tensor: torch.Tensor):
        return cls(
            data=tensor.flatten().tolist(),
            dtype=str(tensor.dtype).replace("torch.", ""),
            shape=list(tensor.shape)
        )

    def to_torch(self) -> torch.Tensor:
        return torch.tensor(self.data, dtype=...).reshape(self.shape)
```

**Elixir Equivalent** (for tinkex):
```elixir
defmodule Tinkex.TensorData do
  @moduledoc """
  Bridge between Nx tensors and Tinker API JSON payloads.
  Matches Python tinker.TensorData structure.
  """

  defstruct [:data, :dtype, :shape]

  @doc """
  Convert Nx tensor to TensorData for API transmission.
  """
  def from_nx(tensor) do
    %__MODULE__{
      data: tensor |> Nx.flatten() |> Nx.to_flat_list(),
      dtype: dtype_to_string(Nx.type(tensor)),
      shape: Tuple.to_list(Nx.shape(tensor))
    }
  end

  @doc """
  Convert TensorData from API response to Nx tensor.
  """
  def to_nx(%__MODULE__{data: data, dtype: dtype, shape: shape}) do
    data
    |> Nx.tensor(type: string_to_dtype(dtype))
    |> Nx.reshape(List.to_tuple(shape))
  end

  defp dtype_to_string({:f, 32}), do: "float32"
  defp dtype_to_string({:s, 64}), do: "int64"
  # ... other types

  defp string_to_dtype("float32"), do: {:f, 32}
  defp string_to_dtype("int64"), do: {:s, 64}
  # ... other types
end
```

**Usage in tinkex_cookbook**:
```elixir
# Creating training data
datum = %Tinkex.Datum{
  model_input: %Tinkex.ModelInput{chunks: [...]},
  loss_fn_inputs: %{
    "target_tokens" => Tinkex.TensorData.from_nx(Nx.tensor(target_tokens)),
    "weights" => Tinkex.TensorData.from_nx(Nx.tensor(weights)),
    "advantages" => Tinkex.TensorData.from_nx(Nx.tensor(advantages))
  }
}

# Processing API responses
logprobs = response.loss_fn_outputs["logprobs"]
  |> Tinkex.TensorData.to_nx()
  |> Nx.dot(weights)
```

**Status**: 🔧 Need to implement `Tinkex.TensorData` module

---

## Detailed Usage Breakdown by Module

### renderers.py (Chat Template Rendering)

**Purpose**: Convert chat messages to token sequences with training weights

**torch usage**:
```python
def tokens_weights_from_strings_weights(
    strings_weights: list[tuple[str, float]],
    tokenizer: Tokenizer,
) -> tuple[torch.Tensor, torch.Tensor]:
    strings, weights = zip(*strings_weights, strict=True)
    token_chunks = [tokenizer.encode(s, ...) for s in strings]

    # Create weight tensor matching token lengths
    weights = torch.cat([
        torch.full((len(chunk),), w)
        for chunk, w in zip(token_chunks, weights)
    ])

    # Create token tensor
    tokens = torch.cat([torch.tensor(chunk) for chunk in token_chunks])
    return tokens, weights
```

**Elixir equivalent**:
```elixir
def tokens_weights_from_strings_weights(strings_weights, tokenizer) do
  {strings, weights} = Enum.unzip(strings_weights)

  token_chunks = Enum.map(strings, &Tokenizers.encode(tokenizer, &1))

  # Create weight tensor
  weights =
    Enum.zip(token_chunks, weights)
    |> Enum.flat_map(fn {chunk, w} ->
      List.duplicate(w, length(chunk.ids))
    end)
    |> Nx.tensor()

  # Create token tensor
  tokens =
    token_chunks
    |> Enum.flat_map(& &1.ids)
    |> Nx.tensor(type: :s64)

  {tokens, weights}
end
```

**Complexity**: Low - just list operations + Nx.tensor

---

### supervised/common.py (Loss Computation Helpers)

**Purpose**: Compute weighted negative log-likelihood from API responses

**torch usage**:
```python
def compute_mean_nll(
    logprobs_list: list[tinker.TensorData],
    weights_list: list[tinker.TensorData]
) -> float:
    total_weighted_logprobs = 0.0
    total_weights = 0.0

    for logprobs, weights in zip(logprobs_list, weights_list):
        logprobs_torch = logprobs.to_torch()
        weights_torch = weights.to_torch()
        total_weighted_logprobs += logprobs_torch.dot(weights_torch)
        total_weights += weights_torch.sum()

    return float(-total_weighted_logprobs / total_weights)
```

**Elixir equivalent**:
```elixir
def compute_mean_nll(logprobs_list, weights_list) do
  {total_weighted, total_weights} =
    Enum.zip(logprobs_list, weights_list)
    |> Enum.reduce({0.0, 0.0}, fn {logprobs, weights}, {tw, wsum} ->
      logprobs_nx = Tinkex.TensorData.to_nx(logprobs)
      weights_nx = Tinkex.TensorData.to_nx(weights)

      {
        tw + Nx.to_number(Nx.dot(logprobs_nx, weights_nx)),
        wsum + Nx.to_number(Nx.sum(weights_nx))
      }
    end)

  -total_weighted / total_weights
end
```

**Complexity**: Low - basic Nx arithmetic

---

### rl/data_processing.py (RL Data Preparation)

**Purpose**: Convert trajectories to training batches with advantages

**torch usage**:
```python
def compute_advantages(trajectory_groups) -> List[torch.Tensor]:
    advantages = []
    for traj_group in trajectory_groups:
        rewards = torch.tensor(traj_group.get_total_rewards())
        # Center advantages within group
        advantages.append(rewards - rewards.mean())
    return advantages

def trajectory_to_data(traj, traj_advantage) -> list[tinker.Datum]:
    # ... construct sequences ...
    return tinker.Datum(
        model_input=...,
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
            "logprobs": TensorData.from_torch(torch.tensor(logprobs)),
            "advantages": TensorData.from_torch(torch.tensor(advantages)),
            "mask": TensorData.from_torch(torch.tensor(mask)),
        }
    )
```

**Elixir equivalent**:
```elixir
def compute_advantages(trajectory_groups) do
  Enum.map(trajectory_groups, fn traj_group ->
    rewards = Nx.tensor(traj_group.get_total_rewards())
    Nx.subtract(rewards, Nx.mean(rewards))
  end)
end

def trajectory_to_data(traj, traj_advantage) do
  # ... construct sequences ...
  %Tinkex.Datum{
    model_input: ...,
    loss_fn_inputs: %{
      "target_tokens" => Tinkex.TensorData.from_nx(Nx.tensor(target_tokens)),
      "logprobs" => Tinkex.TensorData.from_nx(Nx.tensor(logprobs)),
      "advantages" => Tinkex.TensorData.from_nx(Nx.tensor(advantages)),
      "mask" => Tinkex.TensorData.from_nx(Nx.tensor(mask))
    }
  }
end
```

**Complexity**: Low - Nx operations + struct construction

---

### preference/train_dpo.py (DPO Loss Computation)

**Purpose**: Compute Direct Preference Optimization loss locally before sending to API

**torch usage**:
```python
def compute_dpo_loss(
    chosen_logprobs: list[torch.Tensor],
    rejected_logprobs: list[torch.Tensor],
    chosen_ref_logprobs: list[torch.Tensor],
    rejected_ref_logprobs: list[torch.Tensor],
    dpo_beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    # Compute log ratios
    chosen_log_ratio = torch.stack([
        (c - c_ref).sum()
        for c, c_ref in zip(chosen_logprobs, chosen_ref_logprobs)
    ])

    rejected_log_ratio = torch.stack([
        (r - r_ref).sum()
        for r, r_ref in zip(rejected_logprobs, rejected_ref_logprobs)
    ])

    # DPO loss
    losses = -torch.log(torch.sigmoid(
        dpo_beta * (chosen_log_ratio - rejected_log_ratio)
    ))

    return losses, {
        "dpo_loss": losses.mean().item(),
        "chosen_rewards": (dpo_beta * chosen_log_ratio).mean().item(),
        # ... more metrics
    }
```

**Elixir equivalent**:
```elixir
def compute_dpo_loss(chosen_logprobs, rejected_logprobs,
                     chosen_ref_logprobs, rejected_ref_logprobs,
                     dpo_beta) do
  # Compute log ratios
  chosen_log_ratio =
    Enum.zip(chosen_logprobs, chosen_ref_logprobs)
    |> Enum.map(fn {c, c_ref} ->
      Nx.subtract(c, c_ref) |> Nx.sum()
    end)
    |> Nx.stack()

  rejected_log_ratio =
    Enum.zip(rejected_logprobs, rejected_ref_logprobs)
    |> Enum.map(fn {r, r_ref} ->
      Nx.subtract(r, r_ref) |> Nx.sum()
    end)
    |> Nx.stack()

  # DPO loss
  diff = Nx.subtract(chosen_log_ratio, rejected_log_ratio)
  sigmoid = Nx.divide(1, Nx.add(1, Nx.exp(Nx.negate(Nx.multiply(dpo_beta, diff)))))
  losses = Nx.negate(Nx.log(sigmoid))

  {losses, %{
    "dpo_loss" => Nx.to_number(Nx.mean(losses)),
    "chosen_rewards" => Nx.to_number(Nx.mean(Nx.multiply(dpo_beta, chosen_log_ratio)))
    # ... more metrics
  }}
end
```

**Complexity**: Medium - requires sigmoid, log, but all available in Nx

---

## Training Architecture (Why No Model Code)

### Python Flow
```
┌─────────────────────────────────────────────────────────────┐
│ tinker-cookbook (CLIENT)                                    │
├─────────────────────────────────────────────────────────────┤
│ 1. Prepare data with torch (tokenize, create tensors)      │
│ 2. Create tinker.Datum with TensorData payloads            │
│ 3. Call training_client.forward_backward_async(data)       │
│    └─> HTTP POST to Tinker API                             │
│ 4. Receive loss/gradients from API                         │
│ 5. Call training_client.optim_step_async()                 │
│    └─> HTTP POST to Tinker API                             │
│ 6. Repeat                                                   │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ HTTP/REST
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Tinker API (SERVER - we never touch this)                  │
├─────────────────────────────────────────────────────────────┤
│ • Hosts actual PyTorch models in GPU memory                │
│ • Implements backprop, LoRA adapters, optimizers           │
│ • Manages multi-GPU training, model weights                │
│ • Returns logprobs, loss, gradients via API                │
└─────────────────────────────────────────────────────────────┘
```

### Elixir Equivalent
```
┌─────────────────────────────────────────────────────────────┐
│ tinkex_cookbook (CLIENT)                                    │
├─────────────────────────────────────────────────────────────┤
│ 1. Prepare data with Nx (tokenize, create tensors)         │
│ 2. Create Tinkex.Datum with TensorData structs             │
│ 3. Call Tinkex.TrainingClient.forward_backward(data)       │
│    └─> HTTP POST to Tinker API (via tinkex)                │
│ 4. Receive loss/gradients from API                         │
│ 5. Call Tinkex.TrainingClient.optim_step()                 │
│    └─> HTTP POST to Tinker API (via tinkex)                │
│ 6. Repeat                                                   │
└─────────────────────────────────────────────────────────────┘
```

**Key Insight**: The cookbook is just a workflow orchestrator. All model operations happen server-side.

---

## Implementation Plan for tinkex_cookbook

### Phase 1: Core Data Types (tinkex)
```elixir
# lib/tinkex/tensor_data.ex
defmodule Tinkex.TensorData do
  defstruct [:data, :dtype, :shape]

  def from_nx(tensor)
  def to_nx(tensor_data)
end

# lib/tinkex/datum.ex
defmodule Tinkex.Datum do
  defstruct [:model_input, :loss_fn_inputs]
end
```

### Phase 2: Renderers (tinkex_cookbook)
```elixir
# lib/tinkex_cookbook/renderers/llama3.ex
defmodule TinkexCookbook.Renderers.Llama3 do
  def build_supervised_example(messages, tokenizer)
  def build_generation_prompt(messages, tokenizer)
  def get_stop_sequences()
  def parse_response(tokens, tokenizer)
end
```

### Phase 3: Training Orchestration (tinkex_cookbook)
```elixir
# lib/tinkex_cookbook/supervised/trainer.ex
defmodule TinkexCookbook.Supervised.Trainer do
  def train(config, dataset) do
    # Replicate supervised/train.py logic
    # Uses Tinkex.TrainingClient, not local training
  end
end
```

### Phase 4: RL Support (tinkex_cookbook)
```elixir
# lib/tinkex_cookbook/rl/data_processing.ex
defmodule TinkexCookbook.RL.DataProcessing do
  def compute_advantages(trajectory_groups)
  def trajectory_to_data(trajectory, advantage)
end
```

---

## Dependencies Required

### For tinkex (base library)
```elixir
# mix.exs
defp deps do
  [
    {:req, "~> 0.5"},          # HTTP client (already have)
    {:jason, "~> 1.4"},        # JSON (already have)
    {:nx, "~> 0.9"},           # Tensor ops (NEW - replaces torch)
    {:tokenizers, "~> 0.5"}    # HF tokenizers (already have)
  ]
end
```

### For tinkex_cookbook (recipes library)
```elixir
# mix.exs
defp deps do
  [
    {:tinkex, path: "../tinkex"},  # Base client
    {:nx, "~> 0.9"},               # Inherit from tinkex
    # No torch, no transformers!
  ]
end
```

---

## Migration Complexity Assessment

### Easy (Existing Elixir equivalents)
- ✅ Tokenization: `{:tokenizers, "~> 0.5"}`
- ✅ JSON serialization: `Jason`
- ✅ HTTP client: `Req`
- ✅ Async patterns: `Task.async` / `Task.await`

### Medium (Simple Nx wrappers)
- 🟡 `torch.tensor()` → `Nx.tensor()`
- 🟡 `torch.cat()` → `Nx.concatenate()`
- 🟡 `torch.stack()` → `Nx.stack()`
- 🟡 `tensor.mean()` → `Nx.mean()`
- 🟡 `tensor.sum()` → `Nx.sum()`
- 🟡 `tensor.dot()` → `Nx.dot()`

### Hard (Need custom implementation)
- 🔴 `TensorData.from_torch() / to_torch()` → Need `Tinkex.TensorData.from_nx() / to_nx()`
- 🔴 Chat template renderers (7 classes) → Port logic to Elixir modules
- 🔴 RL trajectory processing → Port to functional style

### Not Needed
- ❌ PyTorch autograd (server-side)
- ❌ Model classes (server-side)
- ❌ Optimizer implementations (server-side)
- ❌ GPU memory management (server-side)
- ❌ LoRA adapters (server-side)

---

## Conclusion

**Q: Do we need torch/transformers in tinkex_cookbook?**

**A: NO.**

- **transformers**: Only used for tokenization → Already have `{:tokenizers, "~> 0.5"}`
- **torch**: Only used for simple tensor ops → `Nx` handles all cases trivially

**What we DO need**:
1. `Tinkex.TensorData` module to bridge Nx ↔ API JSON payloads
2. Nx dependency in both tinkex and tinkex_cookbook
3. Port chat renderers to Elixir (medium effort, pure logic)
4. Port training orchestration logic (mostly async HTTP calls + data prep)

**Effort Estimate**:
- Tinkex.TensorData: 1-2 days
- Renderers (7 variants): 3-5 days
- Training orchestration: 5-7 days
- RL support: 3-5 days

**Total**: ~2-3 weeks for full feature parity with tinker-cookbook.

---

## Appendix: Complete File-by-File torch/transformers Usage

### torch imports (12 files)

| File | Usage | Nx Equivalent |
|------|-------|---------------|
| `supervised/common.py` | `tensor.dot()`, `tensor.sum()` | `Nx.dot()`, `Nx.sum()` |
| `supervised/train.py` | None (just imports) | N/A |
| `preference/train_dpo.py` | `torch.stack()`, `torch.log()`, `torch.sigmoid()` | `Nx.stack()`, `Nx.log()`, `Nx.sigmoid()` |
| `preference/types.py` | `torch.nonzero()` | `Nx.argmax()` or custom |
| `distillation/train_on_policy.py` | `torch.tensor()` for conversion | `Nx.tensor()` |
| `recipes/rl_loop.py` | `TensorData.from_torch(torch.tensor(...))` | `TensorData.from_nx(Nx.tensor(...))` |
| `renderers.py` | `torch.tensor()`, `torch.cat()`, `torch.full()` | `Nx.tensor()`, `Nx.concatenate()`, `Nx.broadcast()` |
| `tests/compare_sampling_training_logprobs.py` | `torch.tensor()`, `torch.ones_like()` | `Nx.tensor()`, `Nx.broadcast()` |
| `tests/test_renderers.py` | None (just imports) | N/A |
| `rl/data_processing.py` | `torch.tensor()`, centering ops | `Nx.tensor()`, `Nx.subtract()` |
| `rl/metrics.py` | `torch.cat()`, arithmetic, `torch.tensor()` | `Nx.concatenate()`, Nx arithmetic |
| `rl/train.py` | `.to_torch()` on API responses | `.to_nx()` on API responses |

### transformers imports (3 files)

| File | Usage | Elixir Equivalent |
|------|-------|-------------------|
| `tokenizer_utils.py` | `AutoTokenizer.from_pretrained()` | `Tokenizers.Tokenizer.from_pretrained()` |
| `hyperparam_utils.py` | `AutoConfig.from_pretrained()` | HTTP fetch config.json or hardcode |
| `tests/test_renderers.py` | `AutoTokenizer.from_pretrained()` | `Tokenizers.Tokenizer.from_pretrained()` |

**NO files import**:
- `torch.nn` (neural networks)
- `torch.optim` (optimizers)
- `torch.autograd` (gradients)
- `transformers.models` (model classes)
- `transformers.Trainer` (training loops)

This confirms: **The cookbook is a pure client library. All ML happens server-side.**
