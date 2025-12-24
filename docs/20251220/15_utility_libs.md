# Utility Library Usage Analysis

**Date:** 2025-12-20
**Source:** tinker-cookbook Python codebase
**Status:** TRIVIAL - Native Elixir ports, no Python wrapping needed

---

## Summary

tinker-cookbook uses three standard Python utility libraries for terminal formatting and async operations. All have straightforward Elixir equivalents requiring no FFI or Python bridging.

---

## 1. `rich` - Terminal Formatting & Tables

### Python Usage

**Files:** `tinker_cookbook/utils/ml_log.py`

```python
from rich.console import Console
from rich.table import Table

# Pretty-print logger with styled tables
class PrettyPrintLogger(Logger):
    def __init__(self):
        self.console = Console()

    def log_metrics(self, metrics: Dict[str, Any], step: int | None = None):
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=30)
        table.add_column("Value", style="green")

        if step is not None:
            table.title = f"Step {step}"

        for key, value in sorted(metrics.items()):
            table.add_row(key, f"{value:.6f}")

        self.console.print(table)
```

### Elixir Equivalents

**Option 1: TableRex** (for rich tables)
```elixir
# mix.exs
{:table_rex, "~> 4.0"}

# In code
alias TableRex.Table

def log_metrics(metrics, step \\ nil) do
  title = if step, do: "Step #{step}", else: nil

  rows =
    metrics
    |> Enum.sort()
    |> Enum.map(fn {key, value} ->
      [to_string(key), format_value(value)]
    end)

  Table.new(rows, ["Metric", "Value"], title)
  |> Table.put_column_meta(:all, align: :left)
  |> Table.render!()
  |> IO.puts()
end
```

**Option 2: IO.ANSI** (for simple colored output)
```elixir
def log_metrics(metrics, step \\ nil) do
  if step, do: IO.puts(IO.ANSI.cyan() <> "Step #{step}" <> IO.ANSI.reset())

  metrics
  |> Enum.sort()
  |> Enum.each(fn {key, value} ->
    IO.puts([
      IO.ANSI.cyan(), String.pad_trailing(to_string(key), 30), IO.ANSI.reset(),
      IO.ANSI.green(), format_value(value), IO.ANSI.reset()
    ])
  end)
end
```

**Option 3: Logger** (for production)
```elixir
# config/config.exs
config :logger, :console,
  format: "$metadata[$level] $message\n",
  metadata: [:step, :metric]

# In code
Logger.info("Training metrics",
  step: step,
  metric: :loss,
  value: loss
)
```

**Recommendation:** Use `Logger` for production telemetry, `TableRex` for CLI tools/debugging.

---

## 2. `termcolor` - Colored Terminal Output

### Python Usage

**Files:**
- `tinker_cookbook/display.py`
- `tinker_cookbook/utils/format_colorized.py`
- `tinker_cookbook/eval/inspect_utils.py`
- `tinker_cookbook/rl/play_w_env.py`

```python
from termcolor import colored

# Simple coloring
colored("Error message", "red")
colored("Success", "green", attrs=["bold"])

# Token weight visualization
def format_colorized(tokens, weights, tokenizer):
    chunks = []
    for tok_id, w in zip(tokens, weights):
        if w < 0:
            color = "red"
        elif w == 0:
            color = "yellow"
        else:
            color = "green"
        decoded = tokenizer.decode([tok_id])
        chunks.append(colored(decoded, color))
    return "".join(chunks)
```

### Elixir Equivalent

**IO.ANSI** (built-in)
```elixir
# Simple coloring
IO.ANSI.red() <> "Error message" <> IO.ANSI.reset()
IO.ANSI.green() <> IO.ANSI.bright() <> "Success" <> IO.ANSI.reset()

# Token weight visualization
def format_colorized(tokens, weights, tokenizer) do
  tokens
  |> Enum.zip(weights)
  |> Enum.map(fn {tok_id, w} ->
    color = case w do
      w when w < 0 -> IO.ANSI.red()
      0 -> IO.ANSI.yellow()
      _ -> IO.ANSI.green()
    end

    decoded = Tokenizer.decode(tokenizer, [tok_id])
    [color, decoded, IO.ANSI.reset()]
  end)
  |> IO.iodata_to_binary()
end
```

**Available Colors/Styles:**
- Colors: `black/0`, `red/0`, `green/0`, `yellow/0`, `blue/0`, `magenta/0`, `cyan/0`, `white/0`
- Styles: `bright/0`, `faint/0`, `italic/0`, `underline/0`, `blink_slow/0`, `inverse/0`
- Control: `reset/0`, `clear/0`

---

## 3. `anyio` - Async Utilities

### Python Usage

**Status:** NOT DIRECTLY IMPORTED in tinker-cookbook.

Used implicitly via standard `asyncio` library:

```python
import asyncio

# Async rollouts
async def do_single_rollout(policy, env):
    ob, stop_condition = await env.initial_observation()
    while True:
        ac = await policy(ob, stop_condition)
        result = await env.step(ac.tokens)
        if result.episode_done:
            break
    return trajectory

# Gather parallel operations
trajectories = await asyncio.gather(*[
    do_single_rollout(policy, env) for env in envs
])

# Interactive input
async def get_async_input(prompt):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, input, prompt)
```

### Elixir Equivalent

**Native Task/GenServer** (no libraries needed)

```elixir
# Sequential async operations (like Python's await)
def do_single_rollout(policy, env) do
  {ob, stop_condition} = Env.initial_observation(env)

  Stream.iterate({ob, stop_condition}, fn {ob, stop} ->
    ac = Policy.call(policy, ob, stop)
    Env.step(env, ac.tokens)
  end)
  |> Enum.take_while(fn result -> not result.episode_done end)
  |> build_trajectory()
end

# Parallel operations (like asyncio.gather)
trajectories =
  envs
  |> Enum.map(fn env ->
    Task.async(fn -> do_single_rollout(policy, env) end)
  end)
  |> Task.await_many()

# Interactive input (blocking I/O)
def get_input(prompt) do
  IO.gets(prompt)
end

# Non-blocking input with Task
def get_async_input(prompt) do
  Task.async(fn -> IO.gets(prompt) end)
  |> Task.await()
end
```

**Key Differences:**
- Elixir processes are lightweight (millions per VM) vs Python coroutines
- Use `Task.async/await` for parallel work
- Use `GenServer` for stateful async workers
- No event loop required - BEAM scheduler handles it

**Example: Parallel Training Loop**
```elixir
defmodule TrainingLoop do
  def train_parallel(envs, policy, num_rollouts) do
    1..num_rollouts
    |> Task.async_stream(fn _ ->
      env = Enum.random(envs)
      do_single_rollout(policy, env)
    end, max_concurrency: System.schedulers_online() * 2)
    |> Enum.to_list()
  end
end
```

---

## Port Effort Assessment

| Library | Python LOC | Elixir LOC | Complexity | Effort |
|---------|-----------|------------|------------|--------|
| `rich` tables | ~50 | ~30 (TableRex) | Low | 1-2 hours |
| `termcolor` | ~10 per use | ~8 (IO.ANSI) | Trivial | 15 min |
| `anyio` patterns | ~20 per pattern | ~15 (Task) | Low | 1 hour |

**Total estimated effort:** 4-6 hours to port all utility patterns.

---

## Implementation Notes

### 1. Logging Strategy for tinkex_cookbook

```elixir
# config/config.exs
config :logger,
  backends: [:console, {LoggerFileBackend, :file_log}]

config :logger, :console,
  format: "[$level] $metadata$message\n",
  metadata: [:step, :loss, :accuracy]

config :logger, :file_log,
  path: "logs/training.log",
  level: :info

# lib/tinkex_cookbook/logger.ex
defmodule TinkexCookbook.Logger do
  require Logger

  def log_metrics(metrics, step) do
    # Telemetry event for structured logging
    :telemetry.execute(
      [:tinkex_cookbook, :training, :metrics],
      metrics,
      %{step: step}
    )

    # Pretty console output
    if Application.get_env(:tinkex_cookbook, :pretty_logs, false) do
      print_table(metrics, step)
    else
      Logger.info("Step #{step} metrics", metrics: metrics, step: step)
    end
  end

  defp print_table(metrics, step) do
    # Use TableRex or IO.ANSI as needed
  end
end
```

### 2. Async Pattern Migration

**Python:**
```python
async def train_loop():
    tasks = [train_step(i) for i in range(100)]
    results = await asyncio.gather(*tasks)
    return results
```

**Elixir:**
```elixir
def train_loop do
  1..100
  |> Task.async_stream(&train_step/1,
       max_concurrency: 10,
       timeout: 60_000)
  |> Enum.map(fn {:ok, result} -> result end)
end
```

### 3. No Python Dependencies Required

All three libraries have native Elixir equivalents:
- **No** need for Rustler/NIFs
- **No** need for Python interop
- **No** need for system calls

Standard Elixir + optional `table_rex` package is sufficient.

---

## Conclusion

**Status:** READY TO PORT

- `rich` → `TableRex` or `IO.ANSI` (trivial)
- `termcolor` → `IO.ANSI` (drop-in replacement)
- `anyio` → `Task`/`GenServer` (native async)

No architectural complexity. Can be ported incrementally as needed for CLI tools and logging infrastructure.

**Next steps:** Implement logging module in `tinkex_cookbook` using these patterns when porting training scripts.
