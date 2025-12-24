# Python tinker → Elixir tinkex: Feature Mapping

**Date:** 2025-12-20
**Purpose:** Confirm that the Python `tinker>=0.3.0` dependency is fully replaced by the existing `tinkex` Elixir library

---

## Executive Summary

**The `tinker>=0.3.0` dependency is ALREADY HANDLED by tinkex.** This document confirms feature parity between the Python tinker SDK (v0.3.0+, current v0.7.0) and the Elixir tinkex library (v0.3.2).

---

## Python tinker SDK Features (v0.3.0+)

Based on official documentation and the tinker-cookbook repository, the Python SDK provides:

### Core API Clients

| Python Class | Purpose |
|--------------|---------|
| `ServiceClient` | Entry point for service operations |
| `TrainingClient` | LoRA fine-tuning with gradient operations |
| `SamplingClient` | Text generation and inference |
| `RestClient` | Session and checkpoint management |
| `APIFuture` | Asynchronous operation handling |

### Primary Training Methods

| Python Method | Purpose |
|---------------|---------|
| `forward_backward()` | Compute gradients from loss functions |
| `forward_backward_custom()` | Custom loss with per-datum logprobs |
| `forward()` | Forward-only inference (no gradients) |
| `optim_step()` | Apply optimizer updates |
| `save_state()` / `load_state()` | Checkpoint persistence |
| `save_weights_and_get_sampling_client()` | Export trained weights |

### Sampling Operations

| Python Method | Purpose |
|---------------|---------|
| `sample()` | Generate text completions |
| `create_sampling_client()` | Initialize sampling session |

### Session Management

| Python Method | Purpose |
|---------------|---------|
| `create_lora_training_client()` | Start LoRA training session |
| `list_sessions()` | Enumerate active sessions |
| `get_session()` | Retrieve session details |
| `session_heartbeat()` | Keep session alive |

### Checkpoint Operations

| Python Method | Purpose |
|---------------|---------|
| `list_user_checkpoints()` | List saved checkpoints |
| `get_checkpoint_archive_url()` | Download checkpoint artifacts |
| `delete_checkpoint()` | Remove saved checkpoints |

### Additional Features

- **Tokenization**: HuggingFace tokenizers + TikToken (Kimi K2)
- **Multimodal Input**: Vision-language model support (text + images)
- **Telemetry**: Event logging and metrics collection
- **Retry Logic**: Configurable backoff strategies
- **HTTP/2**: Connection pooling and streaming

---

## Elixir tinkex Mapping (v0.3.2)

### Client Modules (1:1 Parity)

| Python tinker | Elixir tinkex | Status |
|---------------|---------------|--------|
| `ServiceClient` | `Tinkex.ServiceClient` | COMPLETE |
| `TrainingClient` | `Tinkex.TrainingClient` | COMPLETE |
| `SamplingClient` | `Tinkex.SamplingClient` | COMPLETE |
| `RestClient` | `Tinkex.RestClient` | COMPLETE |
| `APIFuture` | `Tinkex.Future` + Elixir Tasks | COMPLETE |

### Training Operations (1:1 Parity)

| Python Method | Elixir tinkex | Status |
|---------------|---------------|--------|
| `forward_backward(data, loss_fn)` | `TrainingClient.forward_backward/3` | COMPLETE |
| `forward_backward_custom(data, loss_fn)` | `TrainingClient.forward_backward_custom/4` | COMPLETE |
| `forward(data)` | `TrainingClient.forward/4` | COMPLETE |
| `optim_step(params)` | `TrainingClient.optim_step/2` | COMPLETE |
| `save_state(path)` | `TrainingClient.save_state/3` | COMPLETE |
| `load_state(path)` | `TrainingClient.load_state/3` | COMPLETE |
| `load_state_with_optimizer()` | `TrainingClient.load_state_with_optimizer/3` | COMPLETE |
| `save_weights_and_get_sampling_client()` | `TrainingClient.save_weights_and_get_sampling_client/2` | COMPLETE |

### Sampling Operations (1:1 Parity)

| Python Method | Elixir tinkex | Status |
|---------------|---------------|--------|
| `sample(prompt, params)` | `SamplingClient.sample/4` | COMPLETE |
| `create_sampling_client()` | `ServiceClient.create_sampling_client/2` | COMPLETE |

### Session Management (1:1 Parity)

| Python Method | Elixir tinkex | Status |
|---------------|---------------|--------|
| `create_lora_training_client()` | `ServiceClient.create_lora_training_client/3` | COMPLETE |
| `list_sessions()` | `RestClient.list_sessions/2` | COMPLETE |
| `get_session()` | `RestClient.get_session/2` | COMPLETE |
| `session_heartbeat()` | Automatic via `SessionManager` | ENHANCED |

### Checkpoint Operations (1:1 Parity)

| Python Method | Elixir tinkex | Status |
|---------------|---------------|--------|
| `list_user_checkpoints()` | `RestClient.list_user_checkpoints/2` | COMPLETE |
| `get_checkpoint_archive_url()` | `RestClient.get_checkpoint_archive_url/2` | COMPLETE |
| `delete_checkpoint()` | `RestClient.delete_checkpoint/2` | COMPLETE |
| (download checkpoint) | `CheckpointDownload.download/3` | ENHANCED |

### Type System (Equivalent Structures)

| Python tinker | Elixir tinkex | Notes |
|---------------|---------------|-------|
| `Datum` | `Tinkex.Types.Datum` | Training data structure |
| `ModelInput` | `Tinkex.Types.ModelInput` | Text/image chunks |
| `SamplingParams` | `Tinkex.Types.SamplingParams` | Generation parameters |
| `AdamParams` | `Tinkex.Types.AdamParams` | Optimizer configuration |
| `ForwardBackwardOutput` | `Tinkex.Types.ForwardBackwardOutput` | Training results |
| `ImageChunk` | `Tinkex.Types.ImageChunk` | Multimodal input |

### Tokenization (Full Parity)

| Python tinker | Elixir tinkex | Status |
|---------------|---------------|--------|
| HuggingFace tokenizers | `{:tokenizers, "~> 0.5"}` | COMPLETE |
| TikToken (Kimi K2) | `{:tiktoken_ex, "~> 0.1"}` | COMPLETE |

### Advanced Features (Elixir Enhancements)

| Feature | Python tinker | Elixir tinkex | Notes |
|---------|---------------|---------------|-------|
| **Concurrency** | `asyncio` | Elixir Tasks + OTP | Native BEAM concurrency |
| **Telemetry** | Basic logging | `:telemetry` ecosystem | Production-grade observability |
| **Retry Logic** | Basic exponential backoff | `Tinkex.RetryConfig` + handlers | Configurable, composable |
| **Recovery** | Manual | `Tinkex.Recovery.*` (opt-in) | Automatic restart from checkpoints |
| **Rate Limiting** | Per-client | Shared `RateLimiter` (multi-tenant) | Bucket isolation per API key |
| **Metrics** | Manual collection | `Tinkex.Metrics` (snapshot/export) | Built-in aggregation |
| **Streaming Downloads** | In-memory | `CheckpointDownload` (O(1) memory) | Progress callbacks |
| **Multipart Uploads** | Built-in | `Tinkex.Multipart.*` | Automatic form-data encoding |
| **Regularizers** | External | `Tinkex.Regularizer.Pipeline` | Composable penalty functions |
| **CLI** | `tinker` command | `tinkex` escript | Checkpoint/sampling commands |

---

## Gap Analysis

### Features in Python tinker NOT in tinkex

**NONE.** All core functionality from Python tinker >=0.3.0 is implemented in tinkex 0.3.2.

### Features in tinkex NOT in Python tinker

| Feature | tinkex Module | Benefit |
|---------|---------------|---------|
| Automatic recovery from corruption | `Tinkex.Recovery.*` | Restart training runs from checkpoints |
| Composable regularizers | `Tinkex.Regularizer.*` | L1/L2/custom penalties in Nx |
| Multi-tenant rate limiting | `Tinkex.RateLimiter` | Isolated buckets per API key |
| Telemetry capture macros | `Tinkex.Telemetry.Capture` | Exception logging |
| Metrics snapshot/export | `Tinkex.Metrics` | Counters + histograms |
| Queue observability | `QueueStateObserver` | Debounced warnings |
| Escript CLI | `tinkex` binary | Self-contained deployment |

---

## Dependency Replacement Confirmation

### tinker-cookbook Dependencies

The Python tinker-cookbook specifies:

```python
dependencies = [
    "tinker>=0.3.0",  # Core training API
    ...
]
```

### tinkex_cookbook Dependencies (Elixir)

```elixir
def deps do
  [
    {:tinkex, "~> 0.3.2"},  # REPLACES tinker>=0.3.0
    ...
  ]
end
```

**CONFIRMED:** The `tinker>=0.3.0` dependency is **fully replaced** by `{:tinkex, "~> 0.3.2"}`.

---

## Implementation Notes for tinkex_cookbook

### Use tinkex for ALL Tinker API Operations

| Python tinker-cookbook | Elixir tinkex_cookbook | Module |
|-------------------------|------------------------|--------|
| `tinker.ServiceClient()` | `Tinkex.ServiceClient.new()` | Service initialization |
| `client.forward_backward()` | `TrainingClient.forward_backward/3` | Training loop |
| `client.sample()` | `SamplingClient.sample/4` | Text generation |
| `client.save_state()` | `TrainingClient.save_state/3` | Checkpointing |

### Leverage Elixir-Specific Enhancements

1. **Use Tasks for Concurrency**
   ```elixir
   # Fan out 10 sampling requests in parallel
   tasks = for i <- 1..10 do
     {:ok, task} = SamplingClient.sample(sampler, prompt, params)
     task
   end
   results = Task.await_many(tasks, 30_000)
   ```

2. **Enable Telemetry for Observability**
   ```elixir
   handler = Tinkex.Telemetry.attach_logger(level: :info)
   # ... run training ...
   Tinkex.Telemetry.detach(handler)
   ```

3. **Collect Metrics Post-Run**
   ```elixir
   :ok = Tinkex.Metrics.flush()
   snapshot = Tinkex.Metrics.snapshot()
   IO.inspect(snapshot.counters, label: "request counts")
   ```

4. **Opt-In to Recovery (Production)**
   ```elixir
   policy = Tinkex.Recovery.Policy.new(enabled: true, checkpoint_strategy: :latest)
   {:ok, monitor} = Tinkex.Recovery.Monitor.start_link(executor: executor, policy: policy)
   ```

---

## Version Mapping

| Python tinker | Elixir tinkex | Notes |
|---------------|---------------|-------|
| 0.3.0 (cookbook min) | 0.3.2 (current) | Full parity |
| 0.7.0 (current Python) | 0.3.2 (reports as 0.7.0 to server) | Server feature gating aligned |

The Elixir tinkex library internally reports `sdk_version: "0.7.0"` to the Tinker backend for feature gating compatibility, while maintaining its own version (0.3.2) for Elixir package management.

---

## Conclusion

**The Python `tinker>=0.3.0` dependency is 100% REPLACED by `{:tinkex, "~> 0.3.2"}`.**

All core training, sampling, checkpoint, and session management features are implemented with 1:1 API parity. Tinkex additionally provides Elixir-native enhancements (OTP concurrency, telemetry, recovery, metrics) that surpass the Python SDK's capabilities.

**No gaps exist.** Proceed with porting tinker-cookbook recipes using tinkex as the direct replacement.

---

**Document Status:** Complete
**Next Action:** Port tinker-cookbook recipes to Elixir using tinkex API
**Maintainer:** tinkex_cookbook team
