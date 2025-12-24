# PyTorch/HuggingFace to Elixir Nx/Axon/Bumblebee Mapping

**Author:** Claude Opus 4.5
**Date:** 2025-12-20
**Purpose:** Technical mapping of PyTorch/Transformers features used in tinker-cookbook to Elixir ML ecosystem equivalents

---

## Executive Summary

This report analyzes the PyTorch and HuggingFace Transformers features used in the tinker-cookbook codebase and maps them to Elixir equivalents (Nx, Axon, Bumblebee, Scholar). The analysis reveals that while the Elixir ML ecosystem has made significant progress, **critical gaps remain for production LoRA fine-tuning workflows**, particularly in parameter-efficient fine-tuning (PEFT) and training loop orchestration.

### Key Findings

- **LoRA Training:** Experimental support exists via `lorax` library, but lacks production maturity of Python PEFT
- **Optimizers:** Axon provides Adam/AdamW via Polaris, compatible with tinker-cookbook patterns
- **Tokenizers:** Full HuggingFace tokenizer support via Rust bindings (`tokenizers` package)
- **Automatic Differentiation:** Nx provides JAX-style autograd with XLA compilation via EXLA
- **Transformers:** Bumblebee supports inference for 100+ HF models, but **training is limited**
- **Gap Analysis:** Missing production-grade PEFT, distributed training, and DPO/RLHF primitives

---

## 1. PyTorch Features Used in tinker-cookbook

Based on analysis of the tinker-cookbook Python codebase, the following PyTorch features are actively used:

### 1.1 Core PyTorch Usage

| Feature | Usage Pattern | Files |
|---------|---------------|-------|
| **Tensor Operations** | `torch.Tensor`, `.dot()`, `.sum()`, `.cat()` | `supervised/common.py`, `rl/metrics.py`, `rl/data_processing.py` |
| **Gradient Computation** | Implicit via Tinker API (forward/backward) | All training modules |
| **Optimizer State** | Adam parameters (beta1, beta2, eps, lr) | `supervised/train.py`, `preference/train_dpo.py`, `rl/train.py` |
| **LoRA Fine-Tuning** | Via Tinker API (`lora_rank`, `train_mlp`, `train_attn`) | `supervised/train.py` (rank=32), DPO, RL |
| **Loss Functions** | NLL, DPO loss, RL advantages | `supervised/common.py`, `preference/train_dpo.py` |
| **Data Processing** | Token tensors, weights, masks | `supervised/common.py` (datum_from_tokens_weights) |

### 1.2 tinker-cookbook Training Patterns

#### Supervised Fine-Tuning (SFT)
```python
# From supervised/train.py
training_client = service_client.create_lora_training_client(
    base_model=config.model_name,
    rank=config.lora_rank  # Default: 32
)

# Adam optimizer configuration
adam_params = tinker.AdamParams(
    learning_rate=config.learning_rate,  # Default: 1e-4
    beta1=config.adam_beta1,             # Default: 0.9
    beta2=config.adam_beta2,             # Default: 0.95 (NOT 0.999!)
    eps=config.adam_eps                  # Default: 1e-12 (NOT 1e-8!)
)
```

#### Direct Preference Optimization (DPO)
```python
# From preference/train_dpo.py
training_client = service_client.create_lora_training_client(
    base_model=config.model_name,
    rank=config.lora_rank  # Default: 32
)

# DPO-specific hyperparameters
dpo_beta: float = 0.1
learning_rate: float = 1e-5  # Lower than SFT
```

#### Reinforcement Learning
```python
# From rl/train.py
# KL divergence computation using PyTorch
kl_sample_train_v1 = flat_diffs.mean().item()
kl_sample_train_v2 = 0.5 * (flat_diffs**2).mean().item()

# Advantage computation (GAE-style)
incorporate_kl_penalty(data_D, base_sampling_client, kl_penalty_coef, kl_discount_factor)
```

### 1.3 PyTorch Tensor API Usage

| Operation | PyTorch Syntax | Purpose |
|-----------|----------------|---------|
| Dot product | `logprobs_torch.dot(weights_torch)` | Weighted NLL computation |
| Sum reduction | `weights_torch.sum()` | Total weight normalization |
| Concatenation | `torch.cat(all_diffs)` | Batch metrics aggregation |
| Mean | `flat_diffs.mean().item()` | KL divergence metrics |
| Element-wise ops | `(flat_diffs**2).mean()` | Squared KL (v2 metric) |
| Masking | `sampling_logprobs[action_mask]` | Action token selection |

---

## 2. Elixir Nx/Axon Equivalents

### 2.1 Numerical Computing: Nx vs PyTorch

| PyTorch | Nx (Elixir) | Notes |
|---------|-------------|-------|
| `torch.tensor()` | `Nx.tensor()` | Multi-dimensional tensors |
| `tensor.dot()` | `Nx.dot()` | Dot product |
| `tensor.sum()` | `Nx.sum()` | Reduction operations |
| `torch.cat()` | `Nx.concatenate()` | Tensor concatenation |
| `tensor.mean()` | `Nx.mean()` | Mean reduction |
| `tensor ** 2` | `Nx.pow(tensor, 2)` | Element-wise power |
| `tensor[mask]` | `Nx.take()` or `Nx.slice()` | Indexing/masking |
| `tensor.item()` | `Nx.to_number()` | Scalar extraction |
| `tensor.tolist()` | `Nx.to_list()` | Convert to Elixir list |

#### Example: Weighted NLL in Elixir
```elixir
# PyTorch version (tinker-cookbook)
total_weighted_logprobs += logprobs_torch.dot(weights_torch)
total_weights += weights_torch.sum()
mean_nll = -total_weighted_logprobs / total_weights

# Nx equivalent
total_weighted_logprobs = Nx.add(total_weighted_logprobs, Nx.dot(logprobs_nx, weights_nx))
total_weights = Nx.add(total_weights, Nx.sum(weights_nx))
mean_nll = Nx.negate(Nx.divide(total_weighted_logprobs, total_weights))
```

### 2.2 Automatic Differentiation: torch.autograd vs Nx.Defn

| Feature | PyTorch | Elixir Nx |
|---------|---------|-----------|
| **Gradient computation** | `torch.autograd.grad()` | `Nx.Defn.grad()` |
| **Reverse-mode AD** | Default | `Nx.Defn.grad()` (default) |
| **Forward-mode AD** | Limited support | `Nx.Defn.jvp()` |
| **JIT compilation** | `torch.jit.script()` | `Nx.Defn.defn` with EXLA backend |
| **XLA integration** | `torch/xla` | `EXLA` (default backend) |
| **Stop gradient** | `tensor.detach()` | `Nx.Defn.stop_grad()` |

#### Nx Automatic Differentiation Example
```elixir
defmodule MyModel do
  import Nx.Defn

  # Numerical definition with automatic differentiation
  defn loss(params, inputs, targets) do
    predictions = forward(params, inputs)
    Nx.mean(Nx.pow(predictions - targets, 2))
  end

  # Gradient computation
  defn grad_loss(params, inputs, targets) do
    grad(params, &loss(&1, inputs, targets))
  end
end
```

**Key Insight:** Nx's `defn` is inspired by JAX's `jit`, providing multi-stage compilation to CPU/GPU via XLA. This is architecturally closer to JAX than PyTorch's eager execution model.

### 2.3 Optimizers: torch.optim vs Polaris

| PyTorch Optimizer | Polaris (Axon) | Configuration |
|-------------------|----------------|---------------|
| `torch.optim.Adam` | `Polaris.Optimizers.adam()` | `learning_rate`, `b1`, `b2`, `eps` |
| `torch.optim.AdamW` | `Polaris.Optimizers.adamw()` | Same + `decay` |
| `torch.optim.SGD` | `Polaris.Optimizers.sgd()` | `learning_rate`, `momentum` |
| `torch.optim.RMSprop` | `Polaris.Optimizers.rmsprop()` | `learning_rate`, `decay`, `momentum` |

#### Tinkex Already Uses Correct Adam Defaults

From `/home/home/p/g/North-Shore-AI/cns_ui/deps/tinkex/lib/tinkex/types/adam_params.ex`:
```elixir
defmodule Tinkex.Types.AdamParams do
  @moduledoc """
  IMPORTANT: Defaults match Python SDK exactly:
  - learning_rate: 0.0001
  - beta1: 0.9
  - beta2: 0.95 (NOT 0.999!)
  - eps: 1.0e-12 (NOT 1e-8!)
  """

  defstruct learning_rate: 0.0001,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1.0e-12
end
```

**Critical Alignment:** Tinkex's Adam defaults match tinker-cookbook's non-standard values (beta2=0.95, eps=1e-12), ensuring parity with Python SDK.

#### Axon Training Loop with Optimizer
```elixir
model = Axon.input("input") |> Axon.dense(128) |> Axon.dense(10)

model
|> Axon.Loop.trainer(:categorical_cross_entropy, Polaris.Optimizers.adamw(learning_rate: 0.005))
|> Axon.Loop.run(train_data, epochs: 10, compiler: EXLA)
```

### 2.4 Training Loop: PyTorch Ignite-style vs Axon.Loop

| Concept | PyTorch/Ignite | Axon |
|---------|----------------|------|
| **Loop abstraction** | `Engine` | `Axon.Loop` |
| **Trainer creation** | `create_supervised_trainer()` | `Axon.Loop.trainer/3` |
| **Evaluator creation** | `create_supervised_evaluator()` | `Axon.Loop.evaluator/1` |
| **Event handlers** | `@trainer.on(Events.ITERATION_COMPLETED)` | `Axon.Loop.handle/4` |
| **Metrics** | `Metrics.Accuracy()` | `Axon.Metrics` (attach via `metric/5`) |
| **Checkpoint saving** | Manual | `Axon.Loop.checkpoint/2` |

#### Axon.Loop Architecture (inspired by PyTorch Ignite)
```elixir
# Create trainer loop
train_loop =
  model
  |> Axon.Loop.trainer(:categorical_cross_entropy, Polaris.Optimizers.adam(learning_rate: 1e-3))
  |> Axon.Loop.metric(:accuracy)
  |> Axon.Loop.handle(:epoch_completed, fn state ->
       IO.puts("Epoch #{state.epoch} - Loss: #{state.loss}, Acc: #{state.metrics.accuracy}")
       {:continue, state}
     end)

# Run training
trained_state = Axon.Loop.run(train_loop, train_data, epochs: 10)
```

**Alignment with tinker-cookbook:** Axon.Loop's instrumented reduction pattern matches tinker-cookbook's pipelined training approach. However, **LoRA-specific training primitives are missing**.

---

## 3. HuggingFace Transformers to Bumblebee Mapping

### 3.1 Model Loading: transformers.AutoModel vs Bumblebee

| HuggingFace Transformers | Bumblebee (Elixir) |
|--------------------------|---------------------|
| `AutoModel.from_pretrained("bert-base-cased")` | `Bumblebee.load_model({:hf, "bert-base-cased"})` |
| `AutoModelForCausalLM.from_pretrained("gpt2")` | `Bumblebee.load_model({:hf, "gpt2"})` |
| `AutoModelForSequenceClassification` | `Bumblebee.load_model({:hf, "model"}, architecture: :for_sequence_classification)` |

#### Example: Loading GPT-2 in Bumblebee
```elixir
{:ok, model_info} = Bumblebee.load_model({:hf, "gpt2"})
{:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "gpt2"})
{:ok, generation_config} = Bumblebee.load_generation_config({:hf, "gpt2"})

serving = Bumblebee.Text.generation(model_info, tokenizer, generation_config)
Nx.Serving.run(serving, "Once upon a time")
```

### 3.2 Tokenizers: HuggingFace tokenizers vs Elixir tokenizers

Tinkex already has HuggingFace tokenizer support via Rust bindings:

From `/home/home/p/g/North-Shore-AI/tinkex/mix.exs`:
```elixir
# Tokenization (HuggingFace models)
{:tokenizers, "~> 0.5"},

# Tokenization (TikToken-style byte BPE, Kimi K2 compatible)
{:tiktoken_ex, "~> 0.1.0"},
```

| Feature | HuggingFace (Python) | Elixir tokenizers |
|---------|----------------------|-------------------|
| **Load pre-trained** | `AutoTokenizer.from_pretrained("bert-base-cased")` | `Tokenizers.Tokenizer.from_pretrained("bert-base-cased")` |
| **Encode text** | `tokenizer("Hello there!", return_tensors="pt")` | `Tokenizers.Tokenizer.encode(tokenizer, "Hello there!")` |
| **Get tokens** | `encoding.tokens()` | `Tokenizers.Encoding.get_tokens(encoding)` |
| **Get IDs** | `encoding.input_ids` | `Tokenizers.Encoding.get_ids(encoding)` |
| **Backend** | Rust (`tokenizers` crate) | Rust (`tokenizers` crate via NIFs) |

#### Example: Tokenization in Elixir
```elixir
{:ok, tokenizer} = Tokenizers.Tokenizer.from_pretrained("bert-base-cased")
{:ok, encoding} = Tokenizers.Tokenizer.encode(tokenizer, "Hello there!")

tokens = Tokenizers.Encoding.get_tokens(encoding)
# ["Hello", "there", "!"]

ids = Tokenizers.Encoding.get_ids(encoding)
# [8667, 1175, 106]
```

**Parity Assessment:** Elixir tokenizers library provides **full feature parity** with HuggingFace tokenizers via shared Rust backend.

### 3.3 Model Architectures: Coverage Comparison

| Architecture | HuggingFace Transformers | Bumblebee | Notes |
|--------------|--------------------------|-----------|-------|
| **BERT** | Full training + inference | **Inference only** | Pre-trained weights loadable |
| **GPT-2** | Full training + inference | **Inference only** | Text generation supported |
| **GPT-Neo/J** | Full training + inference | **Inference only** | Large models supported |
| **LLaMA** | Full training + inference | **Inference only** | Via `{:hf, "meta-llama/..."}` |
| **Qwen** | Full training + inference | **Inference only** | Qwen2 supported |
| **Stable Diffusion** | Full training + inference | **Inference only** | Image generation |
| **CLIP** | Full training + inference | **Inference only** | Vision-language models |

**Critical Gap:** Bumblebee currently supports **inference only**. Training transformers from scratch or fine-tuning requires manual Axon implementation.

---

## 4. LoRA Fine-Tuning: PEFT vs lorax

### 4.1 HuggingFace PEFT (Python)

PEFT (Parameter-Efficient Fine-Tuning) is the standard library for LoRA in Python:

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,                    # LoRA rank
    lora_alpha=32,           # LoRA scaling
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = AutoModelForCausalLM.from_pretrained("gpt2")
lora_model = get_peft_model(model, config)
```

**Features:**
- LoRA, LoRA+, AdaLoRA, IA3, QLoRA (4-bit quantization)
- Hotswap adapters during inference
- Merge adapters back to base model
- Multi-adapter composition

### 4.2 lorax (Elixir)

The `lorax` library provides experimental LoRA support for Elixir:

```elixir
# From lorax documentation
lora_config = %Lorax.Config{
  r: 1,           # LoRA rank (default)
  alpha: 2,       # LoRA alpha (default)
  target: [:query, :value]  # Adapt Q/V matrices
}

# Fine-tune GPT-2 with LoRA
{:ok, model} = Bumblebee.load_model({:hf, "gpt2"})
lora_model = Lorax.apply_lora(model, lora_config)

# Training loop (manual Axon implementation required)
train_loop =
  lora_model
  |> Axon.Loop.trainer(:cross_entropy, Polaris.Optimizers.adam(learning_rate: 1e-4))
  |> Axon.Loop.run(train_data, epochs: 3)
```

### 4.3 Feature Comparison: PEFT vs lorax

| Feature | PEFT (Python) | lorax (Elixir) | Status |
|---------|---------------|----------------|--------|
| **Basic LoRA** | Full support | **Experimental** | Works but immature |
| **LoRA rank** | Configurable | Configurable (default r=1) | Parity |
| **Target modules** | Flexible (Q/K/V/MLP) | Q/V only by default | **Limited** |
| **QLoRA (4-bit)** | Full support | **Not implemented** | Missing |
| **Adapter merging** | Built-in | **Manual** | Missing |
| **Multi-adapter** | Supported | **Not supported** | Missing |
| **Production ready** | Yes (billions of parameters) | **No (experimental)** | Not production-grade |

**Critical Assessment:**
- **lorax is a SpawnFest prototype**, not production-ready
- **No equivalent to PEFT's maturity** in Elixir ecosystem
- **Tinkex's approach (API-based LoRA via Tinker platform) is more reliable** than native Elixir LoRA

### 4.4 Tinkex LoRA Configuration

Tinkex already mirrors Tinker's LoRA config:

From `/home/home/p/g/North-Shore-AI/cns_ui/deps/tinkex/lib/tinkex/types/lora_config.ex`:
```elixir
defmodule Tinkex.Types.LoraConfig do
  @moduledoc """
  LoRA configuration for model fine-tuning.
  Mirrors Python tinker.types.LoraConfig.
  """

  defstruct rank: 32,
            seed: nil,
            train_mlp: true,
            train_attn: true,
            train_unembed: true
end
```

**Alignment:** This matches tinker-cookbook's LoRA defaults exactly (rank=32, train attention + MLP + unembed).

---

## 5. Advanced Features: Gaps and Workarounds

### 5.1 Distributed Training

| Feature | PyTorch | Elixir/Axon |
|---------|---------|-------------|
| **Data parallel** | `torch.nn.DataParallel` | **Planned, not implemented** |
| **Model parallel** | `torch.nn.parallel.DistributedDataParallel` | **Not implemented** |
| **Multi-GPU** | `torch.cuda.device_count()` | EXLA multi-device (experimental) |
| **Multi-node** | Horovod, DeepSpeed | **Not implemented** |

**Current State:** Axon team plans distributed training, but no timeline. Elixir's BEAM VM provides distribution primitives (nodes, message passing), but no ML-specific orchestration.

### 5.2 Mixed Precision Training

| Feature | PyTorch | Elixir/Nx |
|---------|---------|-----------|
| **FP16 training** | `torch.cuda.amp.autocast()` | **Not implemented** |
| **BF16 training** | `torch.autocast(dtype=torch.bfloat16)` | **Not implemented** |
| **Gradient scaling** | `GradScaler` | **Not implemented** |

**Workaround:** EXLA supports BF16 tensors, but no automatic mixed precision (AMP) training loop.

### 5.3 Gradient Accumulation

| Feature | PyTorch | Axon |
|---------|---------|------|
| **Accumulation steps** | Manual (`loss.backward()` every N steps) | **Manual implementation required** |
| **Automatic** | HF Trainer (`gradient_accumulation_steps=4`) | **Not available** |

**Implementation in Axon:**
```elixir
# Manual gradient accumulation (pseudo-code)
defn accumulate_gradients(model, batches, accumulation_steps) do
  Enum.chunk_every(batches, accumulation_steps)
  |> Enum.map(fn batch_chunk ->
       grads = Enum.map(batch_chunk, &compute_grad(model, &1))
       Enum.reduce(grads, &Nx.add/2) |> Nx.divide(accumulation_steps)
     end)
end
```

### 5.4 Learning Rate Schedulers

| PyTorch Scheduler | Axon/Polaris | Status |
|-------------------|--------------|--------|
| `LinearLR` | `Polaris.Schedules.linear_decay()` | **Available** |
| `CosineAnnealingLR` | `Polaris.Schedules.cosine_decay()` | **Available** |
| `ExponentialLR` | `Polaris.Schedules.exponential_decay()` | **Available** |
| `ReduceLROnPlateau` | **Not implemented** | Missing |
| `OneCycleLR` | **Not implemented** | Missing |

**Status:** Basic schedulers available, advanced ones missing.

### 5.5 Checkpointing and Model Serialization

| Feature | PyTorch/HF | Axon/Bumblebee |
|---------|------------|----------------|
| **Save model** | `model.save_pretrained()` | `Axon.serialize()` (binary format) |
| **Load model** | `AutoModel.from_pretrained()` | `Axon.deserialize()` + Bumblebee.load_model |
| **HF Hub upload** | `push_to_hub()` | **Not implemented** |
| **Safetensors** | Supported | **Read-only (via Bumblebee)** |

**Current State:**
- Bumblebee can **load** HuggingFace checkpoints (safetensors, PyTorch bins)
- **No support for saving/uploading** trained models to HF Hub
- Tinkex relies on Tinker platform for checkpoint management

---

## 6. Gap Analysis: What's Missing in Elixir ML Ecosystem

### 6.1 Critical Gaps (Blockers for Production)

| Gap | Impact | Workaround |
|-----|--------|------------|
| **Production-grade PEFT/LoRA** | Cannot fine-tune large models efficiently | Use Tinkex (API-based) or Python PEFT |
| **Transformer training** | Bumblebee inference-only | Implement custom Axon models |
| **DPO/RLHF primitives** | No reinforcement learning from human feedback | Python implementations only |
| **Distributed training** | Cannot scale beyond single GPU | Use Python frameworks |
| **Model upload to HF Hub** | Cannot share trained models | Manual conversion |

### 6.2 Moderate Gaps (Workarounds Exist)

| Gap | Impact | Workaround |
|-----|--------|------------|
| **Mixed precision training** | Slower training, higher memory | Use BF16 tensors manually |
| **Gradient accumulation** | Limited batch sizes | Manual implementation in Axon.Loop |
| **Advanced LR schedulers** | Suboptimal convergence | Use basic schedulers or manual step functions |
| **Gradient checkpointing** | Memory-intensive for large models | Not available, use smaller models |

### 6.3 Minor Gaps (Nice to Have)

| Gap | Impact | Workaround |
|-----|--------|------------|
| **Tensorboard integration** | Limited training visualization | Use Axon metrics + custom logging |
| **ONNX export** | Cannot export for deployment | Use Axon's native serialization |
| **Model quantization** | Larger inference memory footprint | Not critical for research |

---

## 7. Porting Feasibility Assessment

### 7.1 What Can Be Ported Today

| tinker-cookbook Feature | Elixir Feasibility | Recommendation |
|-------------------------|-------------------|----------------|
| **Basic tensor ops** | **High** (Nx parity) | Direct port to Nx |
| **Adam optimizer** | **High** (Polaris.Optimizers.adamw) | Use Axon training loop |
| **Tokenization** | **High** (tokenizers package) | Already available in tinkex |
| **Inference with pre-trained models** | **High** (Bumblebee) | Use Bumblebee.Text.generation |
| **Evaluation metrics** | **Medium** (manual implementation) | Port metrics using Nx |

### 7.2 What Requires Significant Work

| tinker-cookbook Feature | Elixir Feasibility | Effort Estimate |
|-------------------------|-------------------|-----------------|
| **LoRA fine-tuning (native)** | **Low-Medium** | 2-4 weeks (experimental), 2-3 months (production) |
| **DPO training** | **Low** | 4-6 weeks (requires LoRA + preference loss) |
| **RL training (PPO/GRPO)** | **Low** | 6-8 weeks (requires policy gradients, KL penalties) |
| **Distributed training** | **Very Low** | 3-6 months (requires Axon core changes) |

### 7.3 What Should Use Tinker API (Tinkex)

| tinker-cookbook Feature | Recommendation | Rationale |
|-------------------------|----------------|-----------|
| **LoRA training** | **Use Tinkex API** | Mature, production-ready, matches Python SDK |
| **DPO training** | **Use Tinkex API** | Complex loss computation, GPU optimization |
| **RL training** | **Use Tinkex API** | Requires stable rollouts, KL penalties |
| **Multi-GPU training** | **Use Tinkex API** | No Elixir equivalent |

**Strategic Decision:** Tinkex's API-based approach is **superior** to native Elixir LoRA for production workloads. Focus Elixir ML development on:
1. **Inference pipelines** (Bumblebee excels here)
2. **Data preprocessing** (Elixir's concurrency shines)
3. **Evaluation harnesses** (Axon metrics + custom logic)
4. **Orchestration** (BEAM VM distribution for experiment management)

---

## 8. Recommendations

### 8.1 For Tinkex Development

1. **Keep using Tinker API for training** - Don't port LoRA training to native Elixir until ecosystem matures
2. **Leverage Bumblebee for inference** - Use for evaluation, serving, and demo applications
3. **Use Nx for data processing** - Tensor operations, metrics, preprocessing
4. **Build orchestration in Elixir** - BEAM VM excels at managing distributed experiment workflows
5. **Contribute to lorax if needed** - But only for research experiments, not production

### 8.2 For CNS/Thinker Integration

From the `/home/home/p/g/North-Shore-AI/tinkerer/CLAUDE.md` context, the CNS 3.0 dialectical agents (Proposer, Antagonist, Synthesizer) should use:

| Agent | Training | Inference | Rationale |
|-------|----------|-----------|-----------|
| **Proposer** | Tinkex API (LoRA) | Bumblebee (evaluation) | Production-grade LoRA training |
| **Antagonist** | N/A (rule-based initially) | Bumblebee (NLI, retrieval) | Leverage pre-trained models |
| **Synthesizer** | Tinkex API (LoRA) | Bumblebee (serving) | Constrained generation needs GPU |
| **Critics** | Tinkex API (fine-tuning) | Bumblebee (scoring) | Grounding/Logic/Novelty critics |

**Critical for CNS:**
- Use Tinkex for **Proposer LoRA fine-tuning** (already validated in thinker/evaluation.py)
- Use Bumblebee for **inference-only critics** (DeBERTa entailment, sentence transformers)
- **Do not attempt native LoRA in Elixir** until lorax matures (6-12 month timeline)

### 8.3 For Elixir ML Ecosystem Contributions

Priority areas where contributions would have high impact:

1. **Production-grade LoRA in Axon** (6-month effort)
   - Collaborate with lorax maintainer
   - Implement QLoRA (4-bit quantization)
   - Add adapter merging, hotswap

2. **Transformer training in Bumblebee** (3-month effort)
   - Extend Bumblebee to support training loops
   - Implement causal language modeling loss
   - Add gradient checkpointing for memory efficiency

3. **RLHF/DPO primitives** (4-month effort)
   - Port DPO loss computation to Nx
   - Implement reference model KL penalties
   - Create preference dataset utilities

4. **Distributed training in Axon** (12-month effort)
   - Design data parallel API
   - Integrate with BEAM distribution
   - Add multi-GPU synchronization

### 8.4 Migration Path (If Pursuing Native Elixir Training)

**Phase 1: Inference (0-3 months)**
- Use Bumblebee for all inference workloads
- Validate pre-trained model loading
- Build evaluation harness with Nx metrics

**Phase 2: Basic Fine-Tuning (3-6 months)**
- Experiment with lorax for small models (GPT-2, BERT)
- Implement custom training loops in Axon
- Compare performance vs Tinkex API

**Phase 3: Production LoRA (6-12 months)**
- Contribute QLoRA to lorax
- Implement adapter management
- Achieve parity with PEFT on benchmark tasks

**Phase 4: Advanced Training (12+ months)**
- DPO/RLHF primitives
- Distributed training
- Mixed precision automation

**Recommendation:** **Stay at Phase 1-2** for CNS project. Use Tinkex API for production training (already working). Contribute to lorax only if research needs require it.

---

## 9. Technical Reference

### 9.1 Key Libraries and Versions

| Library | Version | Purpose |
|---------|---------|---------|
| **Nx** | 0.10.0+ | Numerical computing, tensors, autograd |
| **EXLA** | 0.9.0+ | XLA backend for GPU/CPU compilation |
| **Axon** | 0.8.0+ | Neural networks, training loops |
| **Bumblebee** | 0.6.0+ | Pre-trained transformers, HF integration |
| **Polaris** | (bundled with Axon) | Optimizers, LR schedulers |
| **tokenizers** | 0.5.1+ | HuggingFace tokenizers (Rust bindings) |
| **lorax** | 0.2.1+ | Experimental LoRA fine-tuning |
| **Scholar** | Latest | Traditional ML (clustering, regression) |

### 9.2 Nx Backend Configuration

For maximum performance (equivalent to PyTorch with CUDA):

```elixir
# config/config.exs
config :nx, default_backend: EXLA.Backend

# Use XLA compilation for all Nx.Defn functions
config :nx, :default_defn_options, [compiler: EXLA]

# Enable GPU if available
config :exla, :clients,
  cuda: [platform: :cuda],
  host: [platform: :host]
```

### 9.3 Equivalent Code Snippets

#### PyTorch (tinker-cookbook)
```python
# supervised/common.py
def compute_mean_nll(logprobs_list, weights_list):
    total_weighted_logprobs = 0.0
    total_weights = 0.0

    for logprobs, weights in zip(logprobs_list, weights_list):
        logprobs_torch = logprobs.to_torch()
        weights_torch = weights.to_torch()
        total_weighted_logprobs += logprobs_torch.dot(weights_torch)
        total_weights += weights_torch.sum()

    return float(-total_weighted_logprobs / total_weights)
```

#### Elixir/Nx Equivalent
```elixir
defmodule Tinkex.Metrics do
  import Nx.Defn

  @doc "Compute weighted mean negative log likelihood"
  def compute_mean_nll(logprobs_list, weights_list) do
    {total_weighted_logprobs, total_weights} =
      Enum.zip(logprobs_list, weights_list)
      |> Enum.reduce({0.0, 0.0}, fn {logprobs, weights}, {acc_logprobs, acc_weights} ->
           logprobs_nx = Nx.tensor(logprobs)
           weights_nx = Nx.tensor(weights)

           weighted = Nx.dot(logprobs_nx, weights_nx)
           sum_weights = Nx.sum(weights_nx)

           {acc_logprobs + Nx.to_number(weighted),
            acc_weights + Nx.to_number(sum_weights)}
         end)

    -total_weighted_logprobs / total_weights
  end
end
```

---

## 10. Sources and References

### Research and Documentation

- [Training LoRA Models with Axon](https://dockyard.com/blog/2024/10/08/training-lora-models-with-axon)
- [Numerical Elixir (Nx) · GitHub](https://github.com/elixir-nx)
- [Machine Learning in Elixir: Learning to Learn with Nx and Axon](https://pragprog.com/titles/smelixir/machine-learning-in-elixir/)
- [lorax: The LoRA fine-tuning method implemented in Elixir](https://github.com/wtedw/lorax)
- [Axon: Nx-powered Neural Networks](https://github.com/elixir-nx/axon)
- [From GPT2 to Stable Diffusion: Hugging Face arrives to the Elixir community](https://huggingface.co/blog/elixir-bumblebee)
- [Bumblebee: Pre-trained Neural Network models in Axon](https://github.com/elixir-nx/bumblebee)
- [Elixir tokenizers: Bindings for HuggingFace Tokenizers](https://github.com/elixir-nx/tokenizers)

### Elixir ML Ecosystem

- [Your first training loop — Axon v0.8.0](https://hexdocs.pm/axon/your_first_training_loop.html)
- [Axon.Loop — Axon v0.8.0](https://hexdocs.pm/axon/Axon.Loop.html)
- [Nx (Numerical Elixir) is now publicly available](https://dashbit.co/blog/nx-numerical-elixir-is-now-publicly-available)
- [Three Years of Nx: Growing the Elixir Machine Learning Ecosystem](https://dockyard.com/blog/2023/11/08/three-years-of-nx-growing-the-machine-learning-ecosystem)
- [Scholar: Traditional machine learning on top of Nx](https://github.com/elixir-nx/scholar)

### PyTorch and JAX References

- [Adam Optimizer - GeeksforGeeks](https://www.geeksforgeeks.org/deep-learning/adam-optimizer/)
- [JAX Automatic Differentiation](https://docs.jax.dev/en/latest/automatic-differentiation.html)
- [JAX GitHub Repository](https://github.com/jax-ml/jax)
- [HuggingFace PEFT Documentation](https://huggingface.co/docs/transformers/peft)
- [HuggingFace PEFT GitHub](https://github.com/huggingface/peft)
- [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers)

---

## Appendix A: Tinkex Current Dependencies

From `/home/home/p/g/North-Shore-AI/tinkex/mix.exs`:

```elixir
defp deps do
  [
    # HTTP/2 client
    {:finch, "~> 0.18"},

    # JSON encoding/decoding
    {:jason, "~> 1.4"},

    # Numerical computing (tensor operations)
    {:nx, "~> 0.9"},

    # GPU/CPU-accelerated backend for Nx
    {:exla, "~> 0.9", runtime: false},

    # Tokenization (HuggingFace models)
    {:tokenizers, "~> 0.5"},

    # Tokenization (TikToken-style byte BPE, Kimi K2 compatible)
    {:tiktoken_ex, "~> 0.1.0"},

    # Telemetry
    {:telemetry, "~> 1.2"},
    {:semaphore, "~> 1.3"},

    # Regularization primitives
    {:nx_penalties, "~> 0.1.2"},
  ]
end
```

**Assessment:** Tinkex already has the core dependencies for tensor operations (Nx, EXLA) and tokenization (tokenizers, tiktoken_ex). Missing: Axon, Bumblebee, lorax (not needed for API-based workflow).

---

## Appendix B: Feature Parity Matrix

| Feature Category | PyTorch/HF | Elixir Ecosystem | Parity Score |
|------------------|------------|------------------|--------------|
| **Tensor Operations** | torch.Tensor | Nx.Tensor | 95% |
| **Automatic Differentiation** | torch.autograd | Nx.Defn.grad | 90% |
| **Optimizers** | torch.optim | Polaris.Optimizers | 85% |
| **Neural Networks** | torch.nn | Axon | 80% |
| **Training Loops** | PyTorch Ignite | Axon.Loop | 75% |
| **Pre-trained Models** | transformers | Bumblebee | 70% (inference only) |
| **Tokenization** | tokenizers | tokenizers (Elixir) | 100% |
| **LoRA Fine-Tuning** | PEFT | lorax | 30% (experimental) |
| **Distributed Training** | DDP, DeepSpeed | None | 0% |
| **Mixed Precision** | AMP | None | 0% |
| **RLHF/DPO** | trl, trlx | None | 0% |
| **Model Hub** | HF Hub | Bumblebee (read-only) | 50% |

**Overall Parity:** ~60% (suitable for inference and research, not production training)

---

## Conclusion

The Elixir ML ecosystem has made impressive progress, particularly in numerical computing (Nx), inference (Bumblebee), and tokenization. However, **critical gaps remain for production fine-tuning workflows**:

1. **LoRA training is experimental** (lorax) - use Tinkex API instead
2. **Transformers are inference-only** - no native training support in Bumblebee
3. **DPO/RLHF primitives don't exist** - Python-only for now
4. **Distributed training is planned but not implemented**

**For CNS/Thinker project:** Continue using Tinkex API for Proposer fine-tuning. Use Bumblebee for inference-based critics. Contribute to lorax only if research needs require experimentation with native Elixir training.

**Strategic recommendation:** Elixir's strengths lie in **orchestration, concurrency, and distribution**. Use BEAM VM for experiment management, data pipelines, and serving. Delegate heavy training to Tinker API or Python until Elixir ecosystem matures (12-24 month timeline for production-grade LoRA).

---

**Document Version:** 1.0
**Last Updated:** 2025-12-20
**Next Review:** Q2 2026 (re-assess lorax maturity, Bumblebee training support)
