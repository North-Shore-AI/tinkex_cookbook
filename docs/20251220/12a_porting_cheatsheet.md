# Torch → Nx Porting Cheatsheet

Quick reference for porting tinker-cookbook code from Python/torch to Elixir/Nx.

## Tensor Creation

```python
# Python
import torch

torch.tensor([1, 2, 3])
torch.tensor([1.0, 2.0, 3.0])
torch.ones_like(tokens)
torch.full((10,), 0.5)
torch.zeros(5)
```

```elixir
# Elixir
import Nx

Nx.tensor([1, 2, 3], type: :s64)
Nx.tensor([1.0, 2.0, 3.0], type: :f32)
Nx.broadcast(1, Nx.shape(tokens))
Nx.broadcast(0.5, {10})
Nx.broadcast(0, {5})
```

## Tensor Operations

```python
# Python
torch.cat([t1, t2])
torch.stack([t1, t2, t3])
t1 + t2
t1 - t2
t1 * t2
t1 / t2
```

```elixir
# Elixir
Nx.concatenate([t1, t2])
Nx.stack([t1, t2, t3])
Nx.add(t1, t2)
Nx.subtract(t1, t2)
Nx.multiply(t1, t2)
Nx.divide(t1, t2)
```

## Reductions

```python
# Python
tensor.sum()
tensor.mean()
tensor.max()
tensor.min()
```

```elixir
# Elixir
Nx.sum(tensor)
Nx.mean(tensor)
Nx.reduce_max(tensor)
Nx.reduce_min(tensor)
```

## Math Functions

```python
# Python
torch.log(tensor)
torch.exp(tensor)
torch.sigmoid(tensor)
torch.dot(t1, t2)
```

```elixir
# Elixir
Nx.log(tensor)
Nx.exp(tensor)
Nx.sigmoid(tensor)
Nx.dot(t1, t2)
```

## Shape Operations

```python
# Python
tensor.reshape([2, 3])
tensor.flatten()
tensor.squeeze()
tensor[1:]  # slice
```

```elixir
# Elixir
Nx.reshape(tensor, {2, 3})
Nx.flatten(tensor)
Nx.squeeze(tensor)
tensor[1..-1//1]  # slice
```

## Type Conversions

```python
# Python
tensor.float()
tensor.long()
tensor.dtype
tensor.tolist()
tensor.item()  # scalar to Python number
```

```elixir
# Elixir
Nx.as_type(tensor, :f32)
Nx.as_type(tensor, :s64)
Nx.type(tensor)
Nx.to_flat_list(tensor)
Nx.to_number(tensor)  # scalar to Elixir number
```

## TensorData Bridge (NEW - implement this)

```python
# Python
import tinker

# To API
tinker.TensorData.from_torch(torch.tensor([1.0, 2.0, 3.0]))
# Returns: TensorData(data=[1.0, 2.0, 3.0], dtype="float32", shape=[3])

# From API
response.loss_fn_outputs["logprobs"].to_torch()
# Returns: torch.Tensor
```

```elixir
# Elixir (implement in tinkex)
alias Tinkex.TensorData

# To API
TensorData.from_nx(Nx.tensor([1.0, 2.0, 3.0]))
# Returns: %TensorData{data: [1.0, 2.0, 3.0], dtype: "float32", shape: [3]}

# From API
response.loss_fn_outputs["logprobs"]
|> TensorData.to_nx()
# Returns: Nx.Tensor
```

## Common Patterns in Cookbook

### Pattern 1: Compute Advantages (RL)

```python
# Python
rewards = torch.tensor(traj_group.get_total_rewards())
advantages = rewards - rewards.mean()
```

```elixir
# Elixir
rewards = Nx.tensor(traj_group.get_total_rewards())
advantages = Nx.subtract(rewards, Nx.mean(rewards))
```

### Pattern 2: Weighted Dot Product (Loss)

```python
# Python
logprobs = logprobs_data.to_torch()
weights = weights_data.to_torch()
weighted_sum = logprobs.dot(weights)
```

```elixir
# Elixir
logprobs = TensorData.to_nx(logprobs_data)
weights = TensorData.to_nx(weights_data)
weighted_sum = Nx.dot(logprobs, weights)
```

### Pattern 3: Create Training Datum

```python
# Python
tinker.Datum(
    model_input=tinker.ModelInput.from_ints(tokens.tolist()),
    loss_fn_inputs={
        "target_tokens": tinker.TensorData.from_torch(torch.tensor(targets)),
        "weights": tinker.TensorData.from_torch(torch.tensor(weights)),
        "advantages": tinker.TensorData.from_torch(torch.tensor(advantages)),
    }
)
```

```elixir
# Elixir
%Tinkex.Datum{
  model_input: Tinkex.ModelInput.from_ints(Nx.to_flat_list(tokens)),
  loss_fn_inputs: %{
    "target_tokens" => Tinkex.TensorData.from_nx(Nx.tensor(targets)),
    "weights" => Tinkex.TensorData.from_nx(Nx.tensor(weights)),
    "advantages" => Tinkex.TensorData.from_nx(Nx.tensor(advantages))
  }
}
```

### Pattern 4: DPO Loss Computation

```python
# Python
chosen_log_ratio = torch.stack([
    (c - c_ref).sum()
    for c, c_ref in zip(chosen, chosen_ref)
])
losses = -torch.log(torch.sigmoid(beta * (chosen_log_ratio - rejected_log_ratio)))
```

```elixir
# Elixir
chosen_log_ratio =
  Enum.zip(chosen, chosen_ref)
  |> Enum.map(fn {c, c_ref} ->
    Nx.subtract(c, c_ref) |> Nx.sum()
  end)
  |> Nx.stack()

diff = Nx.subtract(chosen_log_ratio, rejected_log_ratio)
sigmoid = Nx.sigmoid(Nx.multiply(beta, diff))
losses = Nx.negate(Nx.log(sigmoid))
```

### Pattern 5: Concatenate Token Chunks (Renderers)

```python
# Python
token_chunks = [tokenizer.encode(s) for s in strings]
weights = torch.cat([
    torch.full((len(chunk),), w)
    for chunk, w in zip(token_chunks, weights)
])
tokens = torch.cat([torch.tensor(chunk) for chunk in token_chunks])
```

```elixir
# Elixir
token_chunks = Enum.map(strings, &Tokenizers.encode(tokenizer, &1))

weights =
  Enum.zip(token_chunks, weights)
  |> Enum.flat_map(fn {chunk, w} ->
    List.duplicate(w, length(chunk.ids))
  end)
  |> Nx.tensor()

tokens =
  Enum.flat_map(token_chunks, & &1.ids)
  |> Nx.tensor(type: :s64)
```

## List/Enumerable Patterns

```python
# Python
zip(list1, list2, strict=True)
[x for x in items if condition]
sum(values)
len(items)
```

```elixir
# Elixir
Enum.zip(list1, list2)  # Note: Elixir zip doesn't error on length mismatch by default
Enum.filter(items, fn x -> condition end)
Enum.sum(values)
length(items)
```

## Async Patterns

```python
# Python
import asyncio

results = await asyncio.gather(*[
    async_function(x) for x in items
])
```

```elixir
# Elixir
results =
  items
  |> Enum.map(&Task.async(fn -> async_function(&1) end))
  |> Enum.map(&Task.await(&1))

# Or more concisely:
results =
  Task.async_stream(items, &async_function/1, ordered: true)
  |> Enum.map(fn {:ok, result} -> result end)
```

## Error Handling

```python
# Python
try:
    result = risky_operation()
except ValueError as e:
    logger.error(f"Error: {e}")
    result = None
```

```elixir
# Elixir
result =
  case risky_operation() do
    {:ok, value} -> value
    {:error, reason} ->
      Logger.error("Error: #{inspect(reason)}")
      nil
  end

# Or with try/rescue for exceptions:
result =
  try do
    risky_operation()
  rescue
    e in ArgumentError ->
      Logger.error("Error: #{Exception.message(e)}")
      nil
  end
```

## Logging

```python
# Python
import logging
logger = logging.getLogger(__name__)
logger.info(f"Epoch {epoch}, Loss: {loss:.4f}")
```

```elixir
# Elixir
require Logger
Logger.info("Epoch #{epoch}, Loss: #{Float.round(loss, 4)}")
```

## NOT Needed

These PyTorch features are NOT used in tinker-cookbook:

```python
# DON'T PORT THESE - They don't exist in the cookbook
import torch.nn
import torch.optim
from torch import autograd

model = torch.nn.Linear(10, 5)
optimizer = torch.optim.Adam(model.parameters())
loss.backward()
optimizer.step()
```

The cookbook **never** implements model training code. All training happens server-side via Tinker API.
