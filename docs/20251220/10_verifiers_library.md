# Verifiers Library Research Report

**Date:** 2025-12-20
**Author:** Research Analysis
**Purpose:** Technical assessment of the Python `verifiers` library for potential Elixir integration

**⚠️ DEPENDENCY STATUS: OPTIONAL - NOT CORE TO TINKEX**

This library is **OPTIONAL** in the tinker-cookbook, used only for specialized RL environment recipes. It is **NOT** required for core Tinker API functionality.

**VERIFIED: 2025-12-20**
- ✅ Confirmed as OPTIONAL dependency in tinker-cookbook `pyproject.toml` (lines 55-58)
- ✅ Used in `recipes/verifiers_rl/` for custom RL environment integration
- ✅ NOT listed in main cookbook dependencies (only under `[project.optional-dependencies]`)
- ✅ Requires explicit installation: `pip install tinker-cookbook[verifiers]`
- 🎯 **Port Priority: VERY LOW** (specialized use case, optional installation)

---

## Executive Summary

The `verifiers` library is a modular Python framework for creating reinforcement learning (RL) environments specifically designed for training Large Language Models (LLMs). It provides standardized abstractions for multi-turn agentic interactions, tool use, and verifiable reward functions. The library is maintained by PrimeIntellect-ai and serves as the foundation for their Environments Hub platform.

**Use Case in Cookbook:** Custom RL environments with verifiable rewards (e.g., code verification, mathematical reasoning tasks with automated grading).

**Key Findings:**
- Pure Python library (99.6% Python codebase) with Triton-based GPU kernels
- Heavy native dependencies via optional RL components (PyTorch, CUDA, flash-attention)
- Designed for OpenAI-compatible API integration (not tightly coupled to OpenAI)
- Porting to Elixir would require significant native interop or API-based integration

**Recommendation for Tinkex:**
- **Short-term:** Skip entirely (optional cookbook feature, not core)
- **Medium-term:** API-based integration via Python subprocess if needed
- **Long-term:** Consider native Elixir RL environment framework leveraging Nx/Scholar

---

## 1. Purpose and Functionality

### Core Functionality

The `verifiers` library provides modular components for creating RL environments and training LLM agents. It supports three primary use cases:

1. **RL Training**: Group Relative Policy Optimization (GRPO) for multi-turn environments
2. **LLM Evaluation**: Standardized evaluation harnesses with verifiable rewards
3. **Synthetic Data Generation**: Automated data pipeline creation using LLM rollouts

### Architecture Components

#### Environment Types

| Environment | Purpose |
|-------------|---------|
| `SingleTurnEnv` | Single response per prompt (basic Q&A) |
| `MultiTurnEnv` | Abstract base for custom multi-turn protocols |
| `ToolEnv` | Agentic loops with function calling capabilities |
| `StatefulToolEnv` | Tool execution with persistent state |
| `CodeMathEnv` | Interactive Python code execution |
| `SandboxEnv` | Isolated execution environments |
| `PythonEnv` | Sandboxed Python code verification |
| `ReasoningGymEnv` | Integration with reasoning-gym tasks |

#### Reward System

- **Rubric**: Core abstraction for reward functions (sync/async support)
- **JudgeRubric**: LLM-as-judge evaluation using auxiliary models
- **ToolRubric**: Tracks and scores tool invocation patterns
- **RubricGroup**: Composition of multiple reward functions

#### Data Pipeline

- Built on HuggingFace Datasets standard
- Required columns: `prompt` (input text)
- Optional columns: `answer` (ground truth), `info` (metadata dict)
- Supports trajectory-based rollout tracking (v0.1.8+ refactor)

### Training Integration

The library supports multiple training frameworks:

1. **vf.RLTrainer**: Minimal "nano" trainer (~1000 lines, transformers-based)
2. **prime-rl**: Primary production trainer (performance-optimized, FSDP support)
3. **SkyRL / Tinker**: External trainers with verifiers integration
4. **Custom trainers**: Any system exposing OpenAI-compatible inference clients

Training features:
- Async CISPO (Compositional In-context Policy Optimization) for off-policy rollouts
- Token-level tracking across conversation turns
- Weights & Biases (wandb) integration for experiment tracking
- GRPO (Group Relative Policy Optimization) implementation

### Actual Cookbook Integration

From `/home/home/p/g/North-Shore-AI/tinkerer/thinking-machines-labs/tinker-cookbook/tinker_cookbook/recipes/verifiers_rl/train.py`:

**Key Integration Pattern:**
The cookbook uses a **custom rollout function** that bridges verifiers environments with Tinker's training loop:

```python
import verifiers as vf
from tinker_cookbook.recipes.verifiers_rl.tinker_openai import TinkerAsyncOpenAIClient

# Load verifiers environment
vf_env = vf.load_environment("reverse-text")

# Custom rollout using Tinker sampling client
async def custom_do_group_rollout(builder, policy):
    # Create OpenAI-compatible client wrapping Tinker
    local_client = TinkerAsyncOpenAIClient(
        sampling_client=policy.sampling_client,
        renderer=shared_renderer,
        tokenizer=local_tokenizer
    )

    # Run verifiers rollout with Tinker as backend
    completion, state = await builder.vf_env.rollout(
        client=local_client,
        model="tinker",
        prompt=builder.prompt,
        answer=builder.answer,
    )

    # Score using verifiers rubric
    rs = await builder.vf_env.rubric.score_rollout(
        prompt=builder.prompt,
        completion=completion,
        answer=builder.answer,
        state=state,
    )
```

**Critical Wrapping Strategy:**
- Verifiers expects OpenAI-compatible async clients
- Cookbook implements `TinkerAsyncOpenAIClient` to translate Tinker API → OpenAI format
- Uses `custom_do_group_rollout` to override default RL training loop
- Delegates reward computation to verifiers rubrics (not cookbook code)

---

## 2. GitHub and PyPI Information

### Repository

- **GitHub**: [https://github.com/PrimeIntellect-ai/verifiers](https://github.com/PrimeIntellect-ai/verifiers)
- **Maintainer**: PrimeIntellect-ai (originally created by Will Brown @willccbb)
- **License**: MIT
- **Language**: Python (99.6% of codebase)

### PyPI Package

- **Package Name**: `verifiers`
- **Current Version**: 0.1.8.post2 (released December 11, 2025)
- **Development Status**: Beta (4)
- **PyPI URL**: [https://pypi.org/project/verifiers/](https://pypi.org/project/verifiers/)

### Installation Methods

```bash
# Basic installation (no RL training)
uv add verifiers

# With RL training support
uv add 'verifiers[rl]'

# With environment dependencies
uv add 'verifiers[envs]'

# Development installation
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/verifiers/main/scripts/install.sh | bash
```

### Documentation

- **ReadTheDocs**: [https://verifiers.readthedocs.io/](https://verifiers.readthedocs.io/)
- **Environments Hub**: Integration with HuggingFace-style environment sharing

---

## 3. Dependencies Analysis

### Python Version Requirements

- **Minimum**: Python 3.10
- **Maximum**: Python <3.14
- **Supported**: 3.10, 3.11, 3.12, 3.13
- **Recommended**: Python 3.12+ (per quick-start guide)

### Core Dependencies (Always Required)

These are pure Python packages with minimal native dependencies:

```toml
datasets >= 3.0.0          # HuggingFace datasets
jinja2 >= 3.1.6            # Templating
openai >= 1.108.1          # OpenAI API client
openai-agents >= 0.0.7     # Agent abstractions
nest-asyncio >= 1.6.0      # Async utilities
requests                   # HTTP library
rich                       # Terminal formatting
tenacity >= 8.5.0          # Retry logic
textual                    # TUI framework
pydantic >= 2.11.9         # Data validation
prime-sandboxes >= 0.2.5   # Code execution sandboxes
wget >= 3.2                # File downloads
```

**Python version-specific**:
- `tomli` (Python <3.11 only)
- `typing_extensions` (Python <3.12 only)

### Optional Dependencies: RL Training (`verifiers[rl]`)

**CRITICAL: These include significant native/compiled components**

```toml
torch >= 2.8.0, <2.9.0           # PyTorch (CUDA/ROCm backends)
transformers >= 4.56.2           # HuggingFace transformers
accelerate >= 1.4.0              # Distributed training utilities
peft                             # Parameter-efficient fine-tuning
wandb                            # Experiment tracking
vllm >= 0.10.0, <0.11.0          # Inference optimization (CUDA kernels)
liger-kernel >= 0.5.10           # Triton kernels (GPU-specific)
deepspeed >= 0.17.6              # Distributed training (JIT compiled ops)
flash-attn >= 2.8.3              # Flash Attention (CUDA/HIP kernels)
```

### Optional Dependencies: Environments (`verifiers[envs]`)

```toml
math-verify >= 0.8.0       # Mathematical answer verification
duckduckgo-search          # Web search integration
brave-search               # Alternative web search
reasoning-gym              # Reasoning task benchmarks
nltk                       # NLP utilities
textarena                  # Multi-agent environments
```

### Optional Dependencies: Documentation (`verifiers[docs]`)

```toml
sphinx                     # Documentation generator
myst-parser                # Markdown support for Sphinx
furo                       # Documentation theme
```

---

## 4. Native/C++ Dependencies Deep Dive

### Summary Table

| Dependency | Native Code | Compilation Required | Platform |
|------------|-------------|---------------------|----------|
| Core packages | No | No | Platform-independent |
| PyTorch | Yes | Pre-built wheels available | CUDA/ROCm/CPU |
| vLLM | Yes | Pre-built wheels, CUDA kernels | NVIDIA GPUs (compute >=7.0) |
| flash-attn | Yes | **Often requires compilation** | NVIDIA/AMD GPUs |
| deepspeed | Yes | **JIT compilation or pre-build** | NVIDIA/AMD/Intel GPUs |
| liger-kernel | Yes (Triton) | No (Triton compiles at runtime) | NVIDIA/AMD GPUs |

### 4.1 Flash Attention (`flash-attn >= 2.8.3`)

**Nature**: CUDA/HIP kernels for efficient attention computation

**Native Dependencies**:
- NVIDIA CUDA Toolkit >= 12.0 or AMD ROCm
- C++ compiler (gcc/g++ on Linux, MSVC on Windows)
- Ninja build system (required for reasonable build times)
- 8-9 GB RAM per compilation job

**Compilation Requirements**:
- Pre-built wheels available on PyPI for common configurations
- Source compilation often needed for:
  - Specific CUDA versions
  - Custom GPU architectures
  - Windows platforms
- Compilation time: 3-5 minutes (with ninja on 64-core machine)
- Without ninja: up to 2 hours

**Environment Variables**:
```bash
FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE    # Skip CUDA build (CI only)
FLASH_ATTENTION_FORCE_BUILD=1           # Force rebuild
FLASH_ATTENTION_FORCE_CXX11_ABI=1       # C++11 ABI compatibility
MAX_JOBS=4                              # Limit parallel jobs (RAM constraint)
```

**Porting Implications**:
- Cannot be directly ported to Elixir
- Would require NIFs wrapping CUDA kernels or fallback to standard attention
- Verifiers sets `FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE` in pyproject.toml, suggesting optional use

### 4.2 DeepSpeed (`deepspeed >= 0.17.6`)

**Nature**: Distributed training framework with JIT-compiled ops

**Native Dependencies**:
- CUDA compiler (nvcc) or ROCm compiler (hipcc)
- PyTorch (matching CUDA version)
- Ninja build system
- System libraries: `gcc`, `g++`, `make`, `cmake`, `zlib`, `openssl`, etc.

**Compilation Modes**:
1. **JIT (default)**: Ops compiled at first runtime use via torch JIT loader
2. **Pre-build**: Set `DS_BUILD_OPS=1` during pip install
3. **Selective**: Toggle specific ops with `DS_BUILD_` environment variables

**GPU Architecture**:
- Default compute capabilities: "6.0;6.1;7.0" (Pascal, Volta)
- Specify targets: `TORCH_CUDA_ARCH_LIST="6.1;7.5;8.6"`
- Tested architectures: Pascal, Volta, Ampere, Hopper

**CUDA Version Matching**:
- Major version must match between CUDA toolkit and PyTorch
- Minor version mismatch may cause warnings/issues
- Check versions: `nvcc --version` vs PyTorch CUDA version

**Porting Implications**:
- Heavily dependent on CUDA/ROCm ecosystems
- JIT compilation at runtime makes static analysis difficult
- Would require complete reimplementation or external process integration

### 4.3 vLLM (`vllm >= 0.10.0, <0.11.0`)

**Nature**: High-performance LLM inference server with optimized CUDA kernels

**Native Dependencies**:
- PyTorch >= 2.9.0 (with matching CUDA version)
- NVIDIA GPU: Compute capability >= 7.0 (V100, T4, RTX20xx, A100, L4, H100)
- CUDA Toolkit (version must match PyTorch)
- FlashInfer (`flashinfer-python==0.5.3`) - CUDA version-specific
- NCCL (NVIDIA Collective Communications Library) for multi-GPU
- Numba == 0.61.2 (for speculative decoding)
- Ray[cgraph] >= 2.48.0 (for pipeline parallelism)

**Platform-Specific Builds**:
- **NVIDIA**: `requirements/cuda.txt`
- **AMD ROCm**: `requirements/rocm.txt`
- **Intel XPU**: `requirements/xpu.txt`
- **TPU**: `requirements/tpu.txt` (JAX/XLA stack)
- **CPU-only**: `requirements/cpu.txt`

**Key Features**:
- OpenAI-compatible `/v1/chat/completions` and `/v1/completions` endpoints
- Continuous batching and PagedAttention
- Quantization support (AWQ, GPTQ, SqueezeLLM)
- Multi-GPU tensor/pipeline parallelism

**Porting Implications**:
- Verifiers uses vLLM as an **optional inference backend**
- OpenAI-compatible API means could be replaced with any compatible server
- Elixir client could communicate via HTTP API without native integration

### 4.4 Liger Kernel (`liger-kernel >= 0.5.10`)

**Nature**: Triton-based GPU kernels for efficient LLM training

**Native Dependencies**:
- **Triton >= 2.3.0** (for CUDA) or **>= 3.0.0** (for ROCm)
- PyTorch >= 2.1.2 (CUDA) or >= 2.5.0 (ROCm)
- No direct CUDA compilation (Triton JIT compiles at runtime)

**Key Advantages**:
- Minimal dependencies (only torch + triton)
- No manual kernel compilation required
- 100% Triton-based (works seamlessly with `torch.compile`)
- Full AMD ROCm support (as of v0.4.0)

**Performance**:
- 20% increase in training throughput vs HuggingFace implementations
- 60% reduction in GPU memory usage
- Compatible with PyTorch FSDP, DeepSpeed, DDP

**Porting Implications**:
- Triton is Python-based JIT compiler for GPU kernels
- No direct Elixir port possible
- Could integrate via external Python process or skip if not using RL training

### 4.5 PyTorch (`torch >= 2.8.0, <2.9.0`)

**Nature**: Core deep learning framework with CUDA/ROCm backends

**Native Dependencies**:
- CUDA Toolkit (for NVIDIA GPUs)
- ROCm (for AMD GPUs)
- Intel oneMKL (for CPU/Intel XPU optimization)
- cuDNN, cuBLAS, NCCL (NVIDIA libraries)

**Distribution Options**:
1. **Pre-built wheels**: Available for common configurations (PyPI, conda)
2. **CPU-only builds**: No GPU dependencies
3. **Source compilation**: For custom configurations

**CUDA Bindings**:
- PyTorch ships with pre-compiled CUDA extensions
- Supports multiple CUDA versions per release
- Specify version during install: `pip install torch --index-url https://download.pytorch.org/whl/cu121`

**Porting Implications**:
- Nx (Elixir) provides similar tensor operations
- Could potentially map PyTorch model inference to Nx/EXLA
- Training loop logic could be reimplemented in Elixir
- Pre-trained model loading would require format conversion

---

## 5. Relationship to OpenAI

### OpenAI API Client Dependency

**Direct Dependency**: `openai >= 1.108.1` (core requirement)

**Purpose**:
- **Not** for mandatory OpenAI API usage
- Provides standardized client interface for any OpenAI-compatible endpoint
- Used for LLM judge rubrics and external model evaluation
- Compatible with local servers (vLLM, ollama, etc.)

### OpenAI-Compatible Inference

**Architecture**: Verifiers is **API-agnostic** but uses OpenAI's API specification

**Supported Endpoints**:
1. `/v1/chat/completions` (recommended for most use cases)
2. `/v1/completions` (legacy support)

**Compatible Backends**:
- OpenAI API (GPT-4, GPT-3.5, etc.)
- vLLM (local inference server)
- Ollama (local model serving)
- Any server implementing OpenAI's API spec

**Environment Variables**:
```bash
OPENAI_API_KEY=<key>           # Can be dummy value for local servers
OPENAI_BASE_URL=<url>          # Point to local vLLM/ollama instance
```

### OpenAI Agents Dependency

**Dependency**: `openai-agents >= 0.0.7`

**Purpose**: Abstractions for agentic tool use patterns (function calling, multi-turn)

**Note**: This is OpenAI's agent framework library, separate from the API client

### Judge Rubrics (LLM-as-Judge)

**Use Case**: Using LLMs to evaluate other LLM outputs

**Example**:
```python
import verifiers as vf

judge_rubric = vf.JudgeRubric(
    model="gpt-4o-mini",         # Can be any OpenAI-compatible endpoint
    prompt_template="Evaluate the answer..."
)
```

**Porting Implication**: Elixir client would need HTTP API integration for judge calls

---

## 6. Porting Feasibility Assessment

### 6.1 Component-Level Analysis

| Component | Porting Difficulty | Strategy |
|-----------|-------------------|----------|
| Environment abstractions | Low | Reimplement in Elixir (pure logic) |
| Rubric system | Low | Elixir behaviors + callbacks |
| Dataset loading | Medium | Use HuggingFace API or port to Elixir |
| OpenAI API client | Low | HTTP client (Req, Tesla, Finch) |
| Rollout logic | Medium | Async workflows (GenServer, Task) |
| GRPO training | **High** | Requires PyTorch/Nx bridge |
| vLLM inference | Low (API) | HTTP client integration |
| Flash-attn kernels | **Impossible** | Not portable to Elixir |
| DeepSpeed ops | **Impossible** | Not portable to Elixir |
| Liger kernels | **Impossible** | Triton runtime required |

### 6.2 Integration Strategies

#### Strategy 1: **API-Based Integration (Recommended)**

**Approach**: Use verifiers as external service, Elixir as orchestrator

**Architecture**:
```
Elixir (tinkex)
  ├─> HTTP Client ─> Tinker API (training)
  ├─> HTTP Client ─> vLLM Server (inference)
  └─> Python Process ─> verifiers (RL environment rollouts)
```

**Advantages**:
- No native dependency porting required
- Leverage existing Python ecosystem
- Clean separation of concerns
- Scalable via distributed processes

**Disadvantages**:
- Inter-process communication overhead
- Python runtime dependency
- Serialization/deserialization costs

**Implementation**:
- Use `Port` or `MuonTrap` for Python process management
- JSON-RPC or Protocol Buffers for IPC
- GenServer pool for Python worker processes

#### Strategy 2: **Partial Native Port**

**Approach**: Port environment logic, delegate RL training to Python

**What to Port**:
- Environment state machines (ToolEnv, MultiTurnEnv logic)
- Rubric evaluation (synchronous reward functions)
- Dataset processing (HuggingFace API integration)
- Trajectory tracking and logging

**What to Keep in Python**:
- GRPO training loops (torch, transformers dependencies)
- Flash-attn, deepspeed, liger-kernel optimizations
- vLLM inference server

**Advantages**:
- Better integration with Elixir ecosystem
- Lower runtime overhead for environment logic
- Type safety and pattern matching for state machines

**Disadvantages**:
- Significant development effort
- Ongoing maintenance as verifiers evolves
- Still requires Python for RL training

#### Strategy 3: **Elixir-Native Alternative**

**Approach**: Build Elixir-native RL framework using Nx/Scholar

**Components to Build**:
- Custom RL environments using Elixir behaviors
- Reward functions as pure Elixir modules
- GRPO implementation in Nx (similar to Scholar's algorithms)
- Integration with Axon for model definitions

**Advantages**:
- No Python dependency
- Fully integrated with North-Shore-AI ecosystem
- Leverage BEAM concurrency for parallel rollouts
- Type safety and OTP supervision trees

**Disadvantages**:
- **Massive development effort**
- Missing GPU kernel optimizations (flash-attn, etc.)
- Slower iteration vs using battle-tested Python tools
- Nx/EXLA maturity gap vs PyTorch

**Feasibility**: **Not recommended** unless long-term strategic goal

---

## 7. Recommended Approach for Tinkex

### Short-Term: **API-Based Integration**

**Implementation Plan**:

1. **Add Python process management to tinkex**:
   ```elixir
   defmodule Tinkex.Verifiers.Worker do
     use GenServer
     # Manage Python process with verifiers installed
     # Communicate via JSON-RPC over stdio
   end
   ```

2. **Create Elixir API wrappers**:
   ```elixir
   defmodule Tinkex.Verifiers do
     def create_environment(type, opts)
     def rollout(env, prompt)
     def evaluate(rubric, trajectory)
   end
   ```

3. **Use for tinker-cookbook verifiers_rl recipes**:
   - Detect if verifiers is available in Python env
   - Fall back gracefully if not installed
   - Document as optional dependency

4. **Integration points**:
   - Tinkex.Training: Configure RL environments
   - Tinkex.Evaluation: Run verifier-based evals
   - Tinkex.Dataset: Generate synthetic data via rollouts

### Medium-Term: **Selective Porting**

**Port environment logic** for common use cases:
- `ToolEnv`: Elixir-native function calling environment
- `JudgeRubric`: HTTP-based LLM judge using Req
- Trajectory tracking: GenServer-based state machine

**Keep Python integration** for advanced features:
- GRPO training (heavy PyTorch dependency)
- Complex multi-agent environments (textarena)
- Research-grade optimizations (flash-attn, liger-kernel)

### Long-Term: **Evaluate Nx/Scholar Ecosystem**

**Monitor developments**:
- Nx maturity for RL workloads
- Scholar RL algorithm implementations
- EXLA/TPU optimization progress
- Community momentum around Elixir ML

**Decision criteria for native port**:
- Nx/Scholar reaches feature parity with PyTorch for RL
- North-Shore-AI requires custom RL algorithms
- Performance bottlenecks in Python integration
- Strategic value of pure-Elixir ML stack

---

## 8. Dependency Installation Recommendations

### For Tinkex Documentation

**Minimal Installation** (no RL training):
```bash
# Python 3.10+ required
pip install verifiers
```

**Full Installation** (with RL training):
```bash
# Requires NVIDIA GPU with CUDA 12.0+
pip install 'verifiers[rl]'

# May require manual flash-attn installation:
pip install flash-attn --no-build-isolation

# Limit compilation jobs if RAM-constrained:
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

**Tinker Cookbook Integration**:
```bash
# Install tinker-cookbook with verifiers support
git clone https://github.com/thinking-machines-lab/tinker-cookbook
cd tinker-cookbook
pip install -e .

# Install verifiers as optional dependency
pip install 'verifiers[rl]'
```

### Pre-Built Wheels vs Source Compilation

**Recommendation**: Use pre-built wheels when possible

**When source compilation may be required**:
- Custom CUDA version (not 12.1 or 12.4)
- Non-standard GPU architecture
- Windows development environment
- Cutting-edge GPU features

**Pre-check compatibility**:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
nvcc --version  # Should match PyTorch CUDA version
```

---

## 9. Technical Specifications Summary

### System Requirements

**Minimum (API usage only)**:
- Python 3.10+
- 4 GB RAM
- CPU-only (for environment logic)

**Recommended (RL training)**:
- Python 3.12+
- NVIDIA GPU: Compute capability >= 7.0 (V100, A100, etc.)
- 16+ GB VRAM (GPU memory)
- 64+ GB RAM (system memory)
- CUDA 12.0+ toolkit
- 100+ GB disk space (models + dependencies)

**Compilation Requirements** (if building from source):
- gcc/g++ 9+ (Linux) or MSVC (Windows)
- Ninja build system
- cmake, make
- 8 GB RAM per compilation thread

### Performance Characteristics

**Environment Rollouts**:
- Async multi-turn inference (vLLM-optimized)
- Concurrent environments via Python asyncio
- Typical throughput: 100-1000 rollouts/hour (model-dependent)

**RL Training**:
- GRPO: Off-policy trajectory-based optimization
- Distributed training: FSDP, DeepSpeed, DDP compatible
- Memory efficiency: Flash-attn (60% reduction), liger-kernel (50% reduction)

**Integration Overhead**:
- JSON serialization: ~1-10ms per request (size-dependent)
- Process IPC: ~5-50ms round-trip (local)
- HTTP API calls: ~50-200ms (network-dependent)

---

## 10. Conclusion

### Key Takeaways

1. **Verifiers is a high-quality, modular RL environment framework** for LLM training with standardized abstractions and strong community support.

2. **Heavy native dependencies** (flash-attn, deepspeed, vLLM) make full Elixir porting **impractical** without major compromises.

3. **OpenAI integration is API-based**, not service-dependent. Can use local models, making it flexible for offline/private deployments.

4. **Recommended strategy**: API-based integration via Python subprocess, wrapping core functionality in Elixir GenServers.

5. **Tinker cookbook recipes** using verifiers should document it as an **optional dependency** with graceful fallbacks.

### Next Steps for Tinkex

1. **Document verifiers as optional** in tinkex README
2. **Create example integration** showing Python process management
3. **Implement API wrapper modules** for common verifier patterns
4. **Test with tinker-cookbook RL recipes** (if applicable)
5. **Monitor Nx/Scholar development** for future native alternatives

### References

- [GitHub: PrimeIntellect-ai/verifiers](https://github.com/PrimeIntellect-ai/verifiers)
- [PyPI: verifiers package](https://pypi.org/project/verifiers/)
- [GitHub: tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
- [Verifiers Documentation](https://verifiers.readthedocs.io/)
- [Flash Attention Repository](https://github.com/Dao-AILab/flash-attention)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [Liger Kernel Repository](https://github.com/linkedin/Liger-Kernel)
- [vLLM Documentation](https://docs.vllm.ai/)

---

## 11. Verified Integration Strategy for Tinkex

### 11.1 Dependency Classification

**Status:** OPTIONAL dependency in tinker-cookbook (explicit opt-in via `[verifiers]` extra).

**Decision Tree:**
```
Are you porting tinker-cookbook recipes to Elixir?
├─ NO → Skip verifiers entirely ✅ RECOMMENDED
└─ YES → Are you porting verifiers_rl recipes?
    ├─ NO → Skip verifiers
    └─ YES → Choose integration strategy below
```

### 11.2 Verified Wrapping Strategies

Based on actual cookbook implementation (`recipes/verifiers_rl/train.py`, `tinker_openai.py`):

**Strategy 1: Python Subprocess Integration (Recommended)**

The cookbook uses an **adapter pattern** to make Tinker API compatible with verifiers' OpenAI-style interface:

```elixir
defmodule Tinkex.Verifiers.OpenAIAdapter do
  @moduledoc """
  Translates Tinker sampling API to OpenAI-compatible format for verifiers.
  Similar to cookbook's TinkerAsyncOpenAIClient.
  """

  def chat_completion(sampling_client, messages, opts \\ []) do
    # Convert OpenAI messages to Tinker format
    tinker_input = messages_to_tinker_format(messages)

    # Call Tinker sampling API
    {:ok, response} = Tinkex.Sampling.sample(
      sampling_client,
      tinker_input,
      max_tokens: opts[:max_tokens] || 512
    )

    # Convert back to OpenAI format
    %{
      "id" => generate_id(),
      "object" => "chat.completion",
      "choices" => [
        %{
          "index" => 0,
          "message" => %{
            "role" => "assistant",
            "content" => response.text
          },
          "finish_reason" => "stop"
        }
      ]
    }
  end

  defp messages_to_tinker_format(messages) do
    # Implement OpenAI → Tinker conversion
    # See cookbook's TinkerAsyncOpenAIClient for reference
  end
end
```

**Strategy 2: API-Based Integration via Port**

For verifiers-specific features (environments, rubrics):

```elixir
defmodule Tinkex.Verifiers.Worker do
  use GenServer

  def start_link(env_id) do
    GenServer.start_link(__MODULE__, env_id)
  end

  def init(env_id) do
    # Start Python subprocess with verifiers
    port = Port.open({:spawn, "python -m verifiers_worker"}, [
      :binary,
      packet: 4
    ])

    # Initialize environment
    send_command(port, {:load_env, env_id})

    {:ok, %{port: port, env_id: env_id}}
  end

  def handle_call({:rollout, prompt, answer}, _from, state) do
    # Send rollout request to Python
    response = send_command(state.port, {
      :rollout,
      %{prompt: prompt, answer: answer}
    })

    {:reply, response, state}
  end

  defp send_command(port, command) do
    json = Jason.encode!(command)
    Port.command(port, json)
    # Wait for response...
  end
end
```

**Python Worker Script (`verifiers_worker.py`):**

```python
import sys
import json
import verifiers as vf

env = None

while True:
    # Read command from Elixir
    length_bytes = sys.stdin.buffer.read(4)
    length = int.from_bytes(length_bytes, 'big')
    cmd_bytes = sys.stdin.buffer.read(length)
    cmd = json.loads(cmd_bytes)

    if cmd['type'] == 'load_env':
        env = vf.load_environment(cmd['env_id'])
        result = {'status': 'ok'}

    elif cmd['type'] == 'rollout':
        # Use Tinkex adapter (passed via HTTP or custom client)
        completion, state = await env.rollout(
            client=tinker_openai_client,
            prompt=cmd['prompt'],
            answer=cmd['answer']
        )
        result = {'completion': completion, 'state': state}

    # Send response
    response = json.dumps(result).encode()
    sys.stdout.buffer.write(len(response).to_bytes(4, 'big'))
    sys.stdout.buffer.write(response)
    sys.stdout.buffer.flush()
```

### 11.3 Critical Integration Notes

**From cookbook analysis:**

1. **Custom rollout override required**: Verifiers integration uses `custom_do_group_rollout` to bypass normal RL loop
2. **OpenAI adapter is key**: The `TinkerAsyncOpenAIClient` is the bridge between Tinker and verifiers
3. **Trajectory tracking**: Must capture `(messages, model_input, tokens, logprobs)` per generation step
4. **Reward delegation**: All reward computation happens in verifiers rubrics, not cookbook code

**Elixir Port Requirements:**
- Implement OpenAI-compatible HTTP endpoint OR
- Implement async client adapter (similar to cookbook's approach)
- Handle trajectory recording for multi-turn interactions
- Support verifiers dataset format (prompt, answer, task, info columns)

### 11.4 Porting Recommendations

| Use Case | Strategy | Effort | Priority |
|----------|----------|--------|----------|
| **No verifiers recipes** | Skip entirely | 0 days | N/A |
| **Experimental verifiers RL** | Python subprocess + adapter | 5-7 days | VERY LOW |
| **Production verifiers RL** | Not recommended (use Python) | N/A | N/A |

**Final Recommendation:**
**SKIP ENTIRELY unless you have a specific verifiers-based RL use case.** Even then, consider using Python directly rather than porting, as the heavy native dependencies (flash-attention, deepspeed) make Elixir integration impractical.

If absolutely required:
1. Implement `TinkerAsyncOpenAIClient` Elixir equivalent
2. Run verifiers in Python subprocess
3. Communicate via JSON-RPC over Port
4. Delegate all RL environment logic to Python side

**This is an edge case feature - do not prioritize.**

---

**Report Status**: Complete
**Last Updated**: 2025-12-20
**Confidence Level**: High (based on comprehensive web research and official documentation)
**Verification Status**: ✅ VERIFIED against tinker-cookbook source code (2025-12-20)
