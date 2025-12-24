# Actual NumPy and SciPy Usage Analysis in tinker-cookbook

**Date:** 2025-12-20
**Source:** `/home/home/p/g/North-Shore-AI/tinkerer/thinking-machines-labs/tinker-cookbook/`
**Purpose:** Document ACTUAL numpy/scipy usage (not theoretical) to guide Elixir/Nx port

---

## Executive Summary

**NumPy Usage:** 18 distinct function calls across 9 files
**SciPy Usage:** 1 function (`scipy.signal.lfilter`) in 1 location

All NumPy operations have direct Nx equivalents. The single SciPy usage (IIR filter for discounted rewards) can be replaced with a simple recursive Elixir implementation.

---

## 1. Complete NumPy Function Inventory

### 1.1 Statistical Operations (6 usages)

| Function | File | Line | Context | Nx Equivalent |
|----------|------|------|---------|---------------|
| `np.mean()` | `preference/comparison_policy_evaluator.py` | 65 | Win rate calculation | `Nx.mean(tensor)` |
| `np.mean()` | `utils/misc_utils.py` | 35 | Dictionary averaging | `Nx.mean(tensor)` |
| `np.mean()` | `rl/metric_util.py` | 79 | Reward averaging | `Nx.mean(tensor)` |
| `np.std()` | `preference/comparison_policy_evaluator.py` | 66 | Standard error calculation | `Nx.standard_deviation(tensor)` |
| `np.std()` | `recipes/verifiers_rl/evaluate.py` | 41, 52 | Reward statistics (2×) | `Nx.standard_deviation(tensor)` |
| `np.sqrt()` | `preference/comparison_policy_evaluator.py` | 66 | Standard error denominator | `Nx.sqrt(tensor)` |

**Code Example 1: Win Rate Statistics**
```python
# tinker-cookbook (Python)
return {
    "win_rate": np.mean(results).item(),
    "stderr": np.std(results).item() / np.sqrt(len(results)),
}
```

**Nx Equivalent:**
```elixir
# tinkex_cookbook (Elixir)
results_tensor = Nx.tensor(results)
n = Nx.size(results_tensor)

%{
  win_rate: Nx.to_number(Nx.mean(results_tensor)),
  stderr: Nx.to_number(Nx.standard_deviation(results_tensor) / Nx.sqrt(n))
}
```

---

### 1.2 Array Creation & Manipulation (4 usages)

| Function | File | Line | Context | Nx Equivalent |
|----------|------|------|---------|---------------|
| `np.linspace()` | `utils/misc_utils.py` | 84 | Split list into equal chunks | `Nx.linspace(start, stop, n: count)` |
| `np.linspace()` | `rl/train.py` | 78 | Select representative indices | `Nx.linspace(start, stop, n: count)` |
| `np.argsort()` | `rl/train.py` | 77 | Sort scores for sampling | `Nx.argsort(tensor)` |
| `np.prod()` | `hyperparam_utils.py` | 165 | Count model parameters | `Nx.product(tensor)` |

**Code Example 2: Split List into Equal Chunks**
```python
# tinker-cookbook (Python)
def split_list(lst: Sequence[T], num_splits: int) -> list[list[T]]:
    edges = np.linspace(0, len(lst), num_splits + 1).astype(int)
    return [list(lst[edges[i] : edges[i + 1]]) for i in range(num_splits)]
```

**Nx Equivalent:**
```elixir
# tinkex_cookbook (Elixir)
def split_list(lst, num_splits) do
  n = length(lst)
  edges =
    Nx.linspace(0, n, n: num_splits + 1)
    |> Nx.as_type(:s64)
    |> Nx.to_flat_list()

  Enum.chunk_every(lst, edges, fn i ->
    {start_idx, end_idx} = {Enum.at(edges, i), Enum.at(edges, i + 1)}
    Enum.slice(lst, start_idx, end_idx - start_idx)
  end)
end
```

**Code Example 3: Representative Index Selection**
```python
# tinker-cookbook (Python)
def _select_representative_inds(scores: list[float], num_inds: int) -> list[int]:
    sorted_inds = np.argsort(scores)
    uniform_inds = np.linspace(0, len(sorted_inds) - 1, num_inds).astype(int)
    return [int(sorted_inds[i]) for i in uniform_inds]
```

**Nx Equivalent:**
```elixir
# tinkex_cookbook (Elixir)
def select_representative_inds(scores, num_inds) do
  scores_tensor = Nx.tensor(scores)
  sorted_inds = Nx.argsort(scores_tensor)

  n = Nx.size(sorted_inds)
  uniform_inds =
    Nx.linspace(0, n - 1, n: num_inds)
    |> Nx.as_type(:s64)

  Nx.take(sorted_inds, uniform_inds)
  |> Nx.to_flat_list()
end
```

---

### 1.3 Type Checking (2 usages)

| Function | File | Line | Context | Elixir Equivalent |
|----------|------|------|---------|-------------------|
| `isinstance(x, np.ndarray)` | `recipes/tool_use/search/search_env.py` | 255 | Type guard for ground truth | `is_struct(x, Nx.Tensor)` |
| `.tolist()` | `recipes/tool_use/search/search_env.py` | 256 | Convert ndarray → list | `Nx.to_list(tensor)` |

**Code Example 4: Type Conversion**
```python
# tinker-cookbook (Python)
if isinstance(ground_truth, np.ndarray):
    ground_truth = ground_truth.tolist()
```

**Nx Equivalent:**
```elixir
# tinkex_cookbook (Elixir)
ground_truth =
  if is_struct(ground_truth, Nx.Tensor) do
    Nx.to_list(ground_truth)
  else
    ground_truth
  end
```

---

### 1.4 Random Number Generation (2 usages)

| Function | File | Line | Context | Elixir Equivalent |
|----------|------|------|---------|-------------------|
| `np.random.RandomState(seed)` | `recipes/math_rl/arithmetic_env.py` | 62 | RNG initialization | `:rand.seed(:exsss, seed)` or `Nx.Random.key(seed)` |
| `rng.randint(low, high)` | `recipes/math_rl/arithmetic_env.py` | 74, 75 | Sample random integers | `Nx.Random.randint(key, low, high)` |

**Code Example 5: Random Number Generation**
```python
# tinker-cookbook (Python)
class ArithmeticDataset:
    def __init__(self):
        self._rng = np.random.RandomState(None)

    def get_batch(self, index: int):
        self._rng.seed(index)
        x = self._rng.randint(0, 101)
        y = self._rng.randint(0, 101)
```

**Nx Equivalent:**
```elixir
# tinkex_cookbook (Elixir)
defmodule ArithmeticDataset do
  defstruct [:rng_key]

  def new() do
    %__MODULE__{rng_key: Nx.Random.key(System.system_time())}
  end

  def get_batch(%__MODULE__{} = dataset, index) do
    key = Nx.Random.key(index)
    {x, key} = Nx.Random.randint(key, 0, 101)
    {y, _key} = Nx.Random.randint(key, 0, 101)
    {Nx.to_number(x), Nx.to_number(y)}
  end
end
```

---

### 1.5 Array Attribute Access (4 usages - llms-full.txt only)

**Note:** These appear only in `llms-full.txt` (documentation/examples), not in runtime code:

| Function | Line | Context |
|----------|------|---------|
| `np.concatenate()` | 848, 849 | Concatenate logprobs/weights arrays |
| `np.dot()` | 850 | Loss per token calculation |
| `np.array()` | 2488 | Convert Python list to array |
| `np.float32`, `np.int64` | 2511, 2513 | Dtype constants |

**These are NOT used in actual runtime code** and can be ignored for the port.

---

## 2. Complete SciPy Function Inventory

### 2.1 ONLY Usage: `scipy.signal.lfilter`

**File:** `tinker_cookbook/rl/metrics.py`
**Lines:** 145-147
**Function:** `discounted_future_sum_vectorized(x: np.ndarray, gamma: float)`

**Code:**
```python
def discounted_future_sum_vectorized(x: np.ndarray, gamma: float) -> np.ndarray:
    """
    Compute discounted sum of future values for each position using a vectorized approach.

    Args:
        x (np.ndarray): 1D array of rewards.
        gamma (float): Discount factor.

    Returns:
        np.ndarray: discounted sum of future values.
    """
    # Reverse x so lfilter processes from end to start
    import scipy.signal

    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1].astype(x.dtype)
```

---

### 2.2 What `lfilter` Does

**Mathematical Operation:**
```
For rewards [r1, r2, r3, r4] with gamma=0.9:

Output[4] = r4
Output[3] = r3 + gamma * Output[4] = r3 + 0.9*r4
Output[2] = r2 + gamma * Output[3] = r2 + 0.9*(r3 + 0.9*r4)
Output[1] = r1 + gamma * Output[2] = r1 + 0.9*(r2 + 0.9*(r3 + 0.9*r4))
```

This is the **discounted cumulative reward** used in RL to compute advantages.

**How `lfilter` Implements This:**
- `lfilter([1], [1, -gamma], x)` is an IIR (Infinite Impulse Response) filter
- Transfer function: `H(z) = 1 / (1 - gamma*z^-1)`
- In time domain: `y[n] = x[n] + gamma * y[n-1]`
- The reversal (`x[::-1]` and `[::-1]`) makes it compute future sums instead of past sums

---

### 2.3 Nx Equivalent (Pure Elixir Implementation)

**Option 1: Recursive Elixir (Recommended)**
```elixir
defmodule RL.Metrics do
  @doc """
  Compute discounted sum of future rewards.

  ## Examples
      iex> discounted_future_sum([1.0, 2.0, 3.0, 4.0], 0.9)
      [1.0 + 0.9*(2.0 + 0.9*(3.0 + 0.9*4.0)),
       2.0 + 0.9*(3.0 + 0.9*4.0),
       3.0 + 0.9*4.0,
       4.0]
  """
  def discounted_future_sum(rewards, gamma) when is_list(rewards) do
    rewards
    |> Enum.reverse()
    |> Enum.reduce({[], 0.0}, fn reward, {acc, future_sum} ->
      current_sum = reward + gamma * future_sum
      {[current_sum | acc], current_sum}
    end)
    |> elem(0)
  end

  # Tensor version
  def discounted_future_sum_vectorized(rewards_tensor, gamma) do
    rewards_list = Nx.to_flat_list(rewards_tensor)
    result = discounted_future_sum(rewards_list, gamma)
    Nx.tensor(result)
  end
end
```

**Option 2: Nx Scan (More Functional)**
```elixir
def discounted_future_sum_nx(rewards_tensor, gamma) do
  n = Nx.size(rewards_tensor)
  reversed = Nx.reverse(rewards_tensor)

  # Scan accumulates: acc[i] = rewards[i] + gamma * acc[i-1]
  {_final, cumulative} =
    Nx.reduce(reversed, {0.0, Nx.broadcast(0.0, {n})}, fn reward, {acc, output}, idx ->
      new_acc = reward + gamma * acc
      new_output = Nx.indexed_put(output, [[idx]], new_acc)
      {new_acc, new_output}
    end)

  Nx.reverse(cumulative)
end
```

**Option 3: Direct Implementation (Most Efficient)**
```elixir
def discounted_future_sum_defn(rewards, gamma) do
  import Nx.Defn

  defn compute_discounted(rewards, gamma) do
    n = Nx.size(rewards)
    reversed = Nx.reverse(rewards)

    {_acc, result} = while {acc = 0.0, result = Nx.broadcast(0.0, {n})},
                           i <- 0..(n - 1) do
      reward = reversed[i]
      new_acc = reward + gamma * acc
      new_result = Nx.indexed_put(result, [[i]], new_acc)
      {new_acc, new_result}
    end

    Nx.reverse(result)
  end

  compute_discounted(rewards, gamma)
end
```

---

### 2.4 Usage Context in tinker-cookbook

**Where it's called:**
```python
# tinker_cookbook/rl/metrics.py, line 124
kl_advantages = torch.tensor(
    discounted_future_sum_vectorized(kl_advantages.numpy(), kl_discount_factor)
)
```

**Context:** Compute discounted KL penalty advantages for RL training (optional feature controlled by `kl_discount_factor` config parameter).

**Port Strategy:**
1. Replace with `RL.Metrics.discounted_future_sum_vectorized/2`
2. No need for torch→numpy→torch round-trip (work directly with Nx tensors)

---

## 3. File-by-File Summary

### Files with NumPy Usage

| File | NumPy Functions | Primary Purpose | Port Difficulty |
|------|-----------------|-----------------|-----------------|
| `hyperparam_utils.py` | `np.prod()` | Count model parameters | ⭐ Trivial |
| `preference/comparison_policy_evaluator.py` | `np.mean()`, `np.std()`, `np.sqrt()` | Win rate statistics | ⭐ Trivial |
| `utils/misc_utils.py` | `np.mean()`, `np.linspace()` | Utilities (averaging, splitting) | ⭐ Trivial |
| `recipes/verifiers_rl/evaluate.py` | `np.std()` | Reward statistics | ⭐ Trivial |
| `recipes/tool_use/search/search_env.py` | `isinstance(np.ndarray)`, `.tolist()` | Type checking | ⭐ Trivial |
| `recipes/math_rl/arithmetic_env.py` | `np.random.RandomState`, `.randint()` | RNG for toy problems | ⭐⭐ Easy |
| `rl/metrics.py` | `scipy.signal.lfilter()` | Discounted rewards | ⭐⭐ Easy |
| `rl/metric_util.py` | `np.mean()` | Reward averaging | ⭐ Trivial |
| `rl/train.py` | `np.argsort()`, `np.linspace()` | Sampling/indexing | ⭐ Trivial |

**Port Difficulty Legend:**
- ⭐ Trivial: Direct 1:1 Nx function replacement
- ⭐⭐ Easy: Requires simple refactoring but straightforward

---

## 4. Critical Finding: No Complex SciPy Dependencies

**CONFIRMED:** The earlier analysis concern about extensive SciPy usage was **incorrect**.

**Actual SciPy Usage:**
- 1 function (`scipy.signal.lfilter`)
- 1 file (`rl/metrics.py`)
- 1 use case (discounted rewards for RL advantage calculation)

**Why Previous Analysis Was Misleading:**
- `07_scipy_nx_mapping.md` listed theoretical SciPy functions that COULD be needed
- Actual inspection shows `lfilter` is the ONLY scipy function used in runtime code

---

## 5. Nx Replacement Strategy

### 5.1 NumPy → Nx Mapping Table

| NumPy | Nx | Notes |
|-------|-----|-------|
| `np.mean(arr)` | `Nx.mean(tensor)` | Direct replacement |
| `np.std(arr)` | `Nx.standard_deviation(tensor)` | Uses population std by default |
| `np.sqrt(arr)` | `Nx.sqrt(tensor)` | Element-wise square root |
| `np.linspace(a, b, n)` | `Nx.linspace(a, b, n: n)` | Note keyword argument |
| `np.argsort(arr)` | `Nx.argsort(tensor)` | Returns indices |
| `np.prod(arr)` | `Nx.product(tensor)` | Product of all elements |
| `arr.astype(int)` | `Nx.as_type(tensor, :s64)` | Type casting |
| `np.array(list)` | `Nx.tensor(list)` | List to tensor |
| `.tolist()` | `Nx.to_flat_list(tensor)` | Tensor to list |
| `isinstance(x, np.ndarray)` | `is_struct(x, Nx.Tensor)` | Type checking |
| `len(arr)` | `Nx.size(tensor)` | Number of elements |

### 5.2 SciPy → Elixir Mapping

| SciPy | Elixir Equivalent | Implementation |
|-------|-------------------|----------------|
| `scipy.signal.lfilter([1], [1, -gamma], x[::-1])[::-1]` | `RL.Metrics.discounted_future_sum/2` | See Section 2.3 |

---

## 6. Implementation Checklist for tinkex_cookbook

### Phase 1: Core Utilities (Week 1)
- [ ] Port `utils/misc_utils.py` statistics functions
  - [ ] `dict_mean/1` using `Nx.mean/1`
  - [ ] `split_list/2` using `Nx.linspace/2`
- [ ] Port `hyperparam_utils.py` parameter counting
  - [ ] `get_full_finetune_param_count/1` using `Nx.product/1`

### Phase 2: RL Metrics (Week 2)
- [ ] Port `rl/metrics.py`
  - [ ] `discounted_future_sum_vectorized/2` (custom Elixir implementation)
  - [ ] Test against Python reference implementation
- [ ] Port `rl/metric_util.py`
  - [ ] Reward averaging using `Nx.mean/1`
- [ ] Port `rl/train.py` sampling utilities
  - [ ] `_select_representative_inds/2` using `Nx.argsort/1` + `Nx.linspace/2`

### Phase 3: Evaluation & Environments (Week 3)
- [ ] Port `preference/comparison_policy_evaluator.py`
  - [ ] Win rate statistics
- [ ] Port `recipes/math_rl/arithmetic_env.py`
  - [ ] Replace `np.random` with `Nx.Random`
- [ ] Port `recipes/verifiers_rl/evaluate.py`
  - [ ] Standard deviation calculations

### Phase 4: Edge Cases (Week 4)
- [ ] Port `recipes/tool_use/search/search_env.py`
  - [ ] Type checking (`isinstance(x, np.ndarray)`)
  - [ ] Array conversions (`.tolist()`)

---

## 7. Testing Strategy

### 7.1 Unit Tests for Discounted Rewards (Critical Path)

**Test Case 1: Basic Functionality**
```elixir
defmodule RL.MetricsTest do
  use ExUnit.Case

  test "discounted_future_sum matches scipy.signal.lfilter" do
    # Reference from Python:
    # scipy.signal.lfilter([1], [1, -0.9], [1,2,3,4][::-1])[::-1]
    # => [6.859, 6.51, 5.9, 4.0]

    rewards = Nx.tensor([1.0, 2.0, 3.0, 4.0])
    gamma = 0.9

    result = RL.Metrics.discounted_future_sum_vectorized(rewards, gamma)
    expected = Nx.tensor([6.859, 6.51, 5.9, 4.0])

    assert_all_close(result, expected, atol: 1.0e-3)
  end

  test "handles zero gamma (no discounting)" do
    rewards = Nx.tensor([1.0, 2.0, 3.0])
    result = RL.Metrics.discounted_future_sum_vectorized(rewards, 0.0)

    assert result == rewards  # No discounting
  end

  test "handles gamma = 1.0 (full accumulation)" do
    rewards = Nx.tensor([1.0, 1.0, 1.0, 1.0])
    result = RL.Metrics.discounted_future_sum_vectorized(rewards, 1.0)

    expected = Nx.tensor([4.0, 3.0, 2.0, 1.0])
    assert result == expected
  end
end
```

### 7.2 Cross-Validation Against Python

**Create reference output file:**
```python
# scripts/generate_test_references.py
import numpy as np
import scipy.signal
import json

def generate_discounted_rewards_references():
    test_cases = [
        {"rewards": [1, 2, 3, 4], "gamma": 0.9},
        {"rewards": [0.5, 0.5, 1.0, 0.0], "gamma": 0.95},
        {"rewards": list(range(10)), "gamma": 0.8},
    ]

    results = []
    for case in test_cases:
        x = np.array(case["rewards"])
        gamma = case["gamma"]
        output = scipy.signal.lfilter([1], [1, -gamma], x[::-1])[::-1].tolist()
        results.append({**case, "output": output})

    with open("test/fixtures/discounted_rewards.json", "w") as f:
        json.dump(results, f, indent=2)

generate_discounted_rewards_references()
```

**Elixir validation test:**
```elixir
test "matches Python reference outputs" do
  fixtures =
    "test/fixtures/discounted_rewards.json"
    |> File.read!()
    |> Jason.decode!()

  for fixture <- fixtures do
    rewards = Nx.tensor(fixture["rewards"])
    gamma = fixture["gamma"]
    expected = Nx.tensor(fixture["output"])

    result = RL.Metrics.discounted_future_sum_vectorized(rewards, gamma)

    assert_all_close(result, expected, atol: 1.0e-6)
  end
end
```

---

## 8. Performance Considerations

### 8.1 NumPy → Nx Performance Expectations

**Operations with IDENTICAL performance:**
- `np.mean`, `np.std`, `np.sqrt` (all parallelizable element-wise ops)
- `np.argsort`, `np.linspace` (well-optimized in EXLA backend)

**Operations potentially FASTER in Nx:**
- Chained operations (Nx lazy evaluation optimizes computation graph)
- Example: `Nx.mean(Nx.sqrt(tensor))` fuses into single kernel

**Operations potentially SLOWER in Nx:**
- Small arrays (<1000 elements) due to EXLA overhead
- Mitigation: Use Nx.BinaryBackend for small tensors

### 8.2 SciPy `lfilter` vs. Elixir Recursive Implementation

**Benchmark Setup:**
```elixir
defmodule Benchmarks.DiscountedRewards do
  def run() do
    sizes = [10, 100, 1000, 10_000]

    for size <- sizes do
      rewards = Nx.random_uniform({size})
      gamma = 0.9

      Benchee.run(%{
        "recursive_elixir" => fn ->
          RL.Metrics.discounted_future_sum(Nx.to_flat_list(rewards), gamma)
        end,
        "nx_scan" => fn ->
          RL.Metrics.discounted_future_sum_nx(rewards, gamma)
        end,
        "nx_defn" => fn ->
          RL.Metrics.discounted_future_sum_defn(rewards, gamma)
        end
      })
    end
  end
end
```

**Expected Results:**
- **n < 100:** Recursive Elixir likely fastest (no tensor overhead)
- **n = 100-1000:** Nx scan competitive
- **n > 1000:** Nx defn (JIT compiled) should dominate

**Recommendation:** Use recursive Elixir for typical RL episode lengths (50-200 timesteps).

---

## 9. Migration Risks & Mitigations

### Risk 1: Floating Point Precision Differences

**Issue:** NumPy default float64 vs. Nx default float32

**Mitigation:**
```elixir
# Force float64 for numerical stability in RL algorithms
rewards = Nx.tensor(rewards_list, type: :f64)
result = RL.Metrics.discounted_future_sum_vectorized(rewards, gamma)
```

**Test:**
```elixir
test "maintains float64 precision" do
  rewards = Nx.tensor([1.0, 2.0, 3.0], type: :f64)
  result = RL.Metrics.discounted_future_sum_vectorized(rewards, 0.999)

  assert Nx.type(result) == {:f, 64}
  # Verify no catastrophic cancellation errors
end
```

### Risk 2: Random Number Generator Reproducibility

**Issue:** `np.random.RandomState` seed behavior vs. `Nx.Random` key-based PRNG

**Mitigation:**
```elixir
# Explicit key management for reproducibility
defmodule ArithmeticDataset do
  def get_batch(index) do
    # Use index as seed for deterministic batch generation
    key = Nx.Random.key(index)
    {x, key} = Nx.Random.randint(key, 0, 101)
    {y, _key} = Nx.Random.randint(key, 0, 101)
    {Nx.to_number(x), Nx.to_number(y)}
  end
end
```

### Risk 3: `argsort` Stability

**Issue:** NumPy's `argsort` is stable (preserves order of equal elements), Nx's may not be

**Mitigation:**
```elixir
# Add secondary sort key if stability required
def stable_argsort(tensor) do
  n = Nx.size(tensor)
  indices = Nx.iota({n})

  # Sort by (value, index) pairs
  Nx.argsort(tensor, stable: true)  # If Nx supports stable flag
end
```

**Test:**
```elixir
test "argsort handles ties correctly" do
  scores = Nx.tensor([1.0, 2.0, 2.0, 3.0])
  indices = Nx.argsort(scores)

  # Should preserve original order for equal elements
  assert Nx.to_flat_list(indices) == [0, 1, 2, 3]
end
```

---

## 10. Conclusion

### Key Findings

1. **NumPy usage is minimal and straightforward** (18 function calls, all trivial)
2. **SciPy usage is SINGULAR** (`scipy.signal.lfilter` for discounted rewards only)
3. **No blockers for Elixir/Nx port** (all operations have direct equivalents)

### Recommended Port Priority

**Week 1 (Critical Path):**
1. `RL.Metrics.discounted_future_sum/2` (replaces scipy.signal.lfilter)
2. Basic statistics (`Nx.mean`, `Nx.std`, `Nx.sqrt`)

**Week 2 (High Value):**
3. Array manipulation (`Nx.linspace`, `Nx.argsort`)
4. Random number generation (`Nx.Random`)

**Week 3 (Nice to Have):**
5. Type checking utilities
6. Parameter counting

### Success Criteria

- [ ] All unit tests pass with <1e-6 numerical difference from Python
- [ ] Benchmark shows <10% performance regression on typical workloads
- [ ] Zero runtime scipy dependencies in production code

---

## Appendix A: Complete Function Call Locations

### NumPy Calls (Sorted by File)

```
tinker_cookbook/hyperparam_utils.py:165:        count += np.prod(shape)

tinker_cookbook/preference/comparison_policy_evaluator.py:65:            "win_rate": np.mean(results).item(),
tinker_cookbook/preference/comparison_policy_evaluator.py:66:            "stderr": np.std(results).item() / np.sqrt(len(results)),

tinker_cookbook/utils/misc_utils.py:35:    return {k: float(np.mean(values)) for k, values in key2values.items()}
tinker_cookbook/utils/misc_utils.py:84:    edges = np.linspace(0, len(lst), num_splits + 1).astype(int)

tinker_cookbook/recipes/verifiers_rl/evaluate.py:41:        f"reward: avg - {sum(results.reward) / len(results.reward):.3f}, std - {np.std(results.reward):.3f}"
tinker_cookbook/recipes/verifiers_rl/evaluate.py:52:        print(f"{k}: avg - {sum(v) / len(v):.3f}, std - {np.std(v):.3f}")

tinker_cookbook/recipes/tool_use/search/search_env.py:255:    if isinstance(ground_truth, np.ndarray):
tinker_cookbook/recipes/tool_use/search/search_env.py:256:        ground_truth = ground_truth.tolist()

tinker_cookbook/recipes/math_rl/arithmetic_env.py:62:        self._rng = np.random.RandomState(None)
tinker_cookbook/recipes/math_rl/arithmetic_env.py:73:    def _make_env_group_builder(self, rng: np.random.RandomState) -> ProblemGroupBuilder:

tinker_cookbook/rl/metrics.py:133:def discounted_future_sum_vectorized(x: np.ndarray, gamma: float) -> np.ndarray:
tinker_cookbook/rl/metrics.py:147:    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1].astype(x.dtype)

tinker_cookbook/rl/metric_util.py:79:    metrics["reward/total"] = np.mean(

tinker_cookbook/rl/train.py:77:    sorted_inds = np.argsort(scores)
tinker_cookbook/rl/train.py:78:    uniform_inds = np.linspace(0, len(sorted_inds) - 1, num_inds).astype(int)
```

### SciPy Calls (Complete List)

```
tinker_cookbook/rl/metrics.py:145:    import scipy.signal
tinker_cookbook/rl/metrics.py:147:    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1].astype(x.dtype)
```

---

## Appendix B: Nx API Quick Reference

```elixir
# Statistics
Nx.mean(tensor)                    # Average
Nx.standard_deviation(tensor)      # Standard deviation
Nx.variance(tensor)                # Variance
Nx.sum(tensor)                     # Sum all elements
Nx.product(tensor)                 # Product of all elements

# Array Creation
Nx.tensor([1, 2, 3])               # List → Tensor
Nx.linspace(0, 10, n: 5)           # [0.0, 2.5, 5.0, 7.5, 10.0]
Nx.iota({5})                       # [0, 1, 2, 3, 4]
Nx.broadcast(0.0, {3, 3})          # 3×3 tensor of zeros

# Indexing & Sorting
Nx.argsort(tensor)                 # Indices that sort tensor
Nx.take(tensor, indices)           # Gather elements at indices
Nx.reverse(tensor)                 # Reverse along first axis
tensor[index]                      # Element access

# Math Operations
Nx.sqrt(tensor)                    # Square root
Nx.exp(tensor)                     # Exponential
Nx.log(tensor)                     # Natural log
Nx.pow(tensor, 2)                  # Power

# Type Operations
Nx.as_type(tensor, :f64)           # Cast to float64
Nx.as_type(tensor, :s64)           # Cast to int64
Nx.type(tensor)                    # Get dtype
Nx.size(tensor)                    # Number of elements

# Conversions
Nx.to_flat_list(tensor)            # Tensor → List
Nx.to_number(tensor)               # Scalar tensor → Number
Nx.to_binary(tensor)               # Tensor → Binary

# Random
Nx.Random.key(seed)                # Initialize PRNG
Nx.Random.randint(key, min, max)   # Random integers
Nx.Random.uniform(key, min, max)   # Random floats
```

---

**End of Analysis**
