# Math-Verify Library Research Report

**Date:** 2025-12-20
**Prepared for:** North-Shore-AI / Tinkex Project
**Purpose:** Assess math-verify library for potential Elixir integration

**VERIFIED: 2025-12-20**

## Executive Summary

`math-verify` is a pure Python library developed by HuggingFace for verifying mathematical equivalence between answers, designed specifically for evaluating Large Language Model (LLM) outputs in mathematical reasoning tasks. The library has **no C++ or native dependencies** and could be integrated with Elixir using existing Python interop tools (Pythonx, Snakepit). However, native Elixir alternatives exist that may be more performant and maintainable for long-term use.

---

## 1. Purpose and Functionality

### Overview
`math-verify` is a HuggingFace library that verifies whether two mathematical expressions are equivalent. It's used extensively in the tinker-cookbook for grading mathematical answers in reinforcement learning environments.

### Core Capabilities

**Answer Extraction:** The library supports three extraction configurations:
- **LatexExtractionConfig**: Extracts LaTeX expressions (e.g., `\sqrt{2}`)
- **ExprExtractionConfig**: Extracts plain mathematical expressions (e.g., `1/2`)
- **StringExtractionConfig**: Extracts literal strings (e.g., `A`)

**Verification Process:** Three-step algorithm:
1. **Answer Extraction** - Parse and extract mathematical content from text
2. **Common Representation Conversion** - Convert to SymPy expressions
3. **Gold Comparison** - Compare normalized expressions for equivalence

**Special Features:**
- Asymmetric inequality/interval handling (1 < x < 2 vs (1,2))
- Solution chain verification (accepts "a+2z = 2z + a = 101" when gold is "101")
- Set operations (union, intersection)
- Handles LaTeX, plain text, and mixed formats

### Usage in Tinker-Cookbook

Located in `tinker_cookbook/recipes/math_rl/math_grading.py` (lines 489-506):

```python
def grade_answer_math_verify(given_answer: str, ground_truth: str) -> bool:
    """
    Use the math_verify package to verify the answer.
    """
    from math_verify import parse, verify

    # Make sure the answer is wrapped in $ if it already isn't otherwise it is not parsed correctly
    if not given_answer.startswith("$") and not given_answer.endswith("$"):
        given_answer = f"${given_answer}$"
    if not ground_truth.startswith("$") and not ground_truth.endswith("$"):
        ground_truth = f"${ground_truth}$"

    given_answer_parsed = parse(given_answer)
    ground_truth_parsed = parse(ground_truth)

    is_correct = verify(given_answer_parsed, ground_truth_parsed)

    return is_correct
```

**Actual Usage:** The library is used as an alternative to the built-in `grade_answer()` function that uses sympy directly. Currently used **directly in Python** with no Elixir wrapper.

---

## 2. Repository and Package Information

### Official Sources
- **GitHub Repository**: [huggingface/Math-Verify](https://github.com/huggingface/Math-Verify)
- **PyPI Package**: [math-verify](https://pypi.org/project/math-verify/)
- **Maintainer**: Hynek Kydlíček (HuggingFace)
- **License**: Apache Software License 2.0
- **Current Version**: 0.8.0 (as of December 2025)

### Installation

```bash
# Basic installation
pip install math-verify

# Recommended: specify ANTLR version to avoid issues
pip install math-verify[antlr4_13_2]

# With inference capabilities
pip install 'math-verify[inference]'
```

### Platform Support
- **OS**: Platform-independent (Windows, macOS, Linux)
- **Python Version**: Requires Python >=3.10 (supports 3.10, 3.11, 3.12)

---

## 3. Dependencies Analysis

### Core Dependencies

**Primary Dependency:**
- `latex2sympy2_extended==1.10.2` - LaTeX to SymPy converter

**Transitive Dependencies (via latex2sympy2_extended):**
- `sympy` - Symbolic mathematics library
- `antlr4-python3-runtime` (version >=4.9.3 and <=4.13.2) - Parser runtime

### Optional Dependencies

**antlr4 version-specific:**
- `antlr4_9_3`: antlr4-python3-runtime==4.9.3
- `antlr4_11_0`: antlr4-python3-runtime==4.11.0
- `antlr4_13_2`: antlr4-python3-runtime==4.13.2

**Development:**
- `pytest` (testing)
- `ruff` (formatting)

**Inference:**
- `lighteval[math]` (for model evaluation scripts)

### C++ and Native Code Analysis

**CRITICAL FINDING: NO C++ DEPENDENCIES**

After thorough analysis of all dependencies:

1. **math-verify**: Pure Python library
2. **latex2sympy2_extended**: Pure Python library using Erlang's yecc parser and Elixir transformations
3. **sympy**: Pure Python library (no C++ required for core functionality)
   - Optional performance accelerators exist (gmpy2, Cython) but are NOT required
   - Can run on standard Python installation without compilation
4. **antlr4-python3-runtime**: Pure Python library
   - Distributed as `py3-none-any` wheel (platform-independent)
   - Separate from C++ ANTLR runtime (which is used for different purposes)

**SymPy Optional Native Extensions (NOT required):**
- `gmpy2`: Speeds up arithmetic using GMP (C library) - optional for performance
- `Cython`: Can compile parts of SymPy to C for speed - optional
- `SymEngine`: Standalone C++ symbolic manipulation library - optional alternative to SymPy

**Conclusion:** The entire math-verify dependency chain is pure Python with no mandatory native code compilation or C++ dependencies.

### Build System
- **Build backend**: setuptools
- **Package format**: Pure Python wheel (no compilation needed)
- **Source layout**: `src/` directory structure

---

## 4. Elixir Integration Options

### Option A: Python Interop via Pythonx (Recommended for Quick Integration)

**Library:** [livebook-dev/pythonx](https://github.com/livebook-dev/pythonx)

**Approach:**
- Embeds Python interpreter in Elixir using Erlang NIFs
- Python runs in same OS process as BEAM
- Direct function calls with automatic type conversion

**Pros:**
- Immediate access to math-verify without porting
- Tight integration with automatic data marshalling
- Well-maintained by Livebook team
- Suitable for Livebook-based experimentation

**Cons:**
- Python Global Interpreter Lock (GIL) prevents true concurrency
- Can be bottleneck when called from multiple Elixir processes
- Runtime Python dependency required
- Memory overhead from embedded interpreter

**GIL Considerations:**
- SymPy is pure Python and holds GIL during computation
- No NumPy-style GIL release for CPU-intensive operations
- Sequential execution even with concurrent Elixir processes
- Acceptable for occasional grading; problematic for high-throughput

**Example Implementation:**

```elixir
# In mix.exs
defp deps do
  [
    {:pythonx, "~> 0.4.7"}
  ]
end

# Math verifier module
defmodule Tinkex.MathVerifier do
  def verify_answer(given_answer, ground_truth) do
    Pythonx.eval("""
    from math_verify import parse, verify

    given = "${given_answer}$" if not "${given_answer}".startswith("$") else "${given_answer}"
    truth = "${ground_truth}$" if not "${ground_truth}".startswith("$") else "${ground_truth}"

    result = verify(parse(given), parse(truth))
    """)
    |> case do
      {:ok, result} -> {:ok, result}
      {:error, reason} -> {:error, reason}
    end
  end
end
```

### Option B: Python Interop via Snakepit (Recommended for Production)

**Library:** [nshkrdotcom/snakepit](https://github.com/nshkrdotcom/snakepit)

**Approach:**
- Process pooler with external Python workers
- gRPC-based communication
- Supervised worker pools

**Pros:**
- True process isolation and concurrency
- Better fault tolerance via supervision
- Can scale with multiple Python workers
- GIL limitation scoped per worker
- Heartbeat monitoring and auto-recovery

**Cons:**
- More complex setup than Pythonx
- IPC overhead for each call
- Requires Python runtime on deployment system
- Network serialization costs

**Use Case:** High-throughput production environments where grading happens frequently and concurrently.

### Option C: Native Elixir Port (Recommended for Long-term)

**Approach:** Reimplement math-verify functionality in pure Elixir

**Why Feasible:**
1. Math-verify logic is straightforward parsing + normalization
2. No complex C++ code to port
3. Elixir has good parsing libraries (leex, yecc)
4. SymPy operations are mostly algebraic simplification

**Implementation Strategy:**

**Phase 1: LaTeX/Expression Parser**
- Use `Abacus` or `Leibniz` for expression parsing
- Extend with LaTeX support using leex/yecc
- Normalize to canonical form

**Phase 2: Symbolic Math Engine**
- Use `Exun` for symbolic mathematics
- Implement algebraic simplification rules
- Add interval/set comparison logic

**Phase 3: Equivalence Checking**
- Implement string normalization rules from math-verify
- Add SymPy-style difference checking
- Handle edge cases (tuples, fractions, inequalities)

**Pros:**
- No external runtime dependencies
- Full control over behavior
- Native BEAM concurrency
- Better performance for high-throughput
- Easier to debug and maintain

**Cons:**
- Significant development effort (2-4 weeks)
- Need to maintain parity with math-verify behavior
- Symbolic math in Elixir less mature than Python
- May miss edge cases initially

---

## 5. Elixir Symbolic Math Libraries

### Exun (Most Feature-Complete)

**Repository:** [Nazari/exun](https://github.com/Nazari/exun)

**Status:** Beta

**Features:**
- Symbolic math with pattern matching
- Derivatives and integration (polynomial, trig, logarithmic)
- Vector and matrix types with symbolic elements
- Unit support and dimensional analysis
- Numerator/denominator representation for accuracy
- AST built with Erlang's yecc parser

**Limitations:**
- Beta maturity level
- Limited documentation
- Smaller ecosystem than SymPy

**Suitability for Math-Verify:**
- Good foundation for symbolic equivalence checking
- Needs extension for LaTeX parsing
- Derivative/integral features not needed for grading

### SymMath

**Repository:** [genericsoma/SymMath](https://github.com/genericsoma/SymMath)

**Features:**
- Symbolic manipulation using Elixir macros
- Lightweight approach

**Limitations:**
- Less complete than Exun
- Minimal maintenance
- Limited documentation

**Suitability:** Not recommended as primary foundation; could be used for specific transformations.

### Expression Parsers

**Abacus** ([narrowtux/abacus](https://github.com/narrowtux/abacus))
- Math expression parser and evaluator
- Supports arithmetic, exponentials, factorials, bitwise/boolean ops
- Inspired by math.js
- Good for parsing plain expressions
- No LaTeX support

**Leibniz** ([Leibniz](https://elixirstatus.com/p/Z26H-leibniz-math-expression-parser-and-evaluator))
- Pure Elixir parser using leex and yecc
- Math expression evaluation
- Good learning example for parser implementation

**expr** ([Rob-bie/expr](https://github.com/Rob-bie/expr))
- Basic expression parsing
- No error checking in v0.1.0
- Minimal feature set

**Recommendation:** Combine Exun (symbolic math) + Abacus (expression parsing) + custom LaTeX parser (leex/yecc) for native implementation.

---

## 6. Comparative Analysis

### Python (math-verify) vs Elixir Native

| Aspect | math-verify + Pythonx/Snakepit | Native Elixir Port |
|--------|--------------------------------|-------------------|
| **Development Time** | Immediate (Pythonx) to 1-2 days (Snakepit) | 2-4 weeks |
| **Runtime Dependencies** | Python 3.10+, math-verify, sympy | None (pure Elixir) |
| **Concurrency** | Limited by GIL (Pythonx) or worker pool (Snakepit) | Full BEAM concurrency |
| **Performance** | Good for single-threaded, IPC overhead for Snakepit | Excellent for concurrent workloads |
| **Maintenance** | Track upstream math-verify changes | Full control, self-maintained |
| **Deployment** | Requires Python on target system | Elixir release only |
| **Debugging** | Cross-language complexity | Native Elixir tools |
| **Testing** | Harder to test Python integration | Standard ExUnit tests |
| **Ecosystem Maturity** | Very mature (SymPy 15+ years) | Limited (Exun in beta) |
| **Edge Cases** | Inherits math-verify's comprehensive handling | Need to discover and implement |

---

## 7. Recommendations

**CURRENT STATE (2025-12-20):** The tinker-cookbook uses math-verify **directly in Python** with no Elixir integration. The Pythonx/Snakepit recommendations below are for **future tinkex integration** only.

### Short-term (Immediate to 1 month)

**Use Pythonx for experimentation and prototyping:**

```elixir
# Quick integration for tinkex development
defmodule Tinkex.Evaluation.MathGrader do
  @moduledoc """
  Mathematical answer verification using math-verify via Pythonx.
  Suitable for experimentation and low-throughput use cases.
  """

  def verify(given_answer, ground_truth, opts \\ []) do
    timeout = Keyword.get(opts, :timeout, 5000)

    python_code = """
    from math_verify import parse, verify

    def wrap_math(s):
        return s if s.startswith("$") else f"${s}$"

    given_parsed = parse(wrap_math("""#{escape_python(given_answer)}"""))
    truth_parsed = parse(wrap_math("""#{escape_python(ground_truth)}"""))
    result = verify(given_parsed, truth_parsed)
    """

    case Pythonx.eval(python_code, timeout: timeout) do
      {:ok, true} -> {:ok, :correct}
      {:ok, false} -> {:ok, :incorrect}
      {:error, reason} -> {:error, {:verification_failed, reason}}
    end
  end

  defp escape_python(str) do
    String.replace(str, "\"", "\\\"")
  end
end
```

**Rationale:**
- Fastest path to integration
- Acceptable for dataset evaluation (batch processing)
- Can validate Elixir implementation against it later

### Medium-term (1-3 months)

**Migrate to Snakepit for production workloads:**

```elixir
# Production-ready with worker pool
defmodule Tinkex.Evaluation.MathGrader.Pool do
  use Snakepit,
    language: :python,
    workers: 4,
    protocol: :grpc

  def verify(given_answer, ground_truth) do
    # Route to available worker
    Snakepit.call(__MODULE__, {:verify, given_answer, ground_truth})
  end

  # Python worker implementation handles math-verify calls
end
```

**Rationale:**
- Better concurrency for training loops
- Fault tolerance via supervision
- Scales with workload
- Still leverages mature math-verify

### Long-term (3-6 months)

**Develop native Elixir implementation:**

**Phase 1 (Weeks 1-2): Parser and Normalizer**
```elixir
defmodule Tinkex.MathVerify.Parser do
  @moduledoc "Parse mathematical expressions from LaTeX and plain text"
  # Use Abacus + custom LaTeX parser
end

defmodule Tinkex.MathVerify.Normalizer do
  @moduledoc "Normalize expressions to canonical form"
  # Implement normalization rules from math-verify
end
```

**Phase 2 (Weeks 3-4): Symbolic Comparison**
```elixir
defmodule Tinkex.MathVerify.Verifier do
  @moduledoc "Verify equivalence of mathematical expressions"
  # Use Exun for symbolic operations
  # Implement difference-based checking
  # Handle tuples, sets, intervals, inequalities
end
```

**Phase 3 (Weeks 5-6): Testing and Edge Cases**
- Comprehensive test suite against math-verify
- Edge case handling
- Performance optimization

**Rationale:**
- Eliminates external dependencies
- Full control over grading logic
- Better long-term maintainability
- Native BEAM performance

### Hybrid Approach (Recommended)

**Use all three in parallel:**

1. **Pythonx**: Immediate experiments and validation
2. **Snakepit**: Production RL training pipelines
3. **Native**: Gradual rollout starting with simple cases

**Implementation:**

```elixir
defmodule Tinkex.Evaluation.MathGrader do
  @behaviour Tinkex.Evaluation.Grader

  def verify(given_answer, ground_truth, opts \\ []) do
    strategy = Keyword.get(opts, :strategy, :auto)

    case strategy do
      :python -> verify_python(given_answer, ground_truth, opts)
      :native -> verify_native(given_answer, ground_truth, opts)
      :auto -> verify_auto(given_answer, ground_truth, opts)
    end
  end

  defp verify_auto(given, truth, opts) do
    # Use native for simple cases, fall back to Python for complex
    case verify_native(given, truth, opts) do
      {:ok, result} -> {:ok, result}
      {:error, :unsupported} -> verify_python(given, truth, opts)
      error -> error
    end
  end
end
```

---

## 8. Risk Assessment

### Python Interop Risks

**Technical Risks:**
- **GIL bottleneck**: Limits concurrent grading throughput
- **Error handling**: Python exceptions need careful translation
- **Version compatibility**: Python/package version mismatches
- **Memory leaks**: Long-running Python interpreters may leak

**Mitigation:**
- Use Snakepit worker pools instead of Pythonx for production
- Implement circuit breakers and timeouts
- Version lock Python dependencies
- Restart workers periodically

**Operational Risks:**
- **Deployment complexity**: Need Python runtime on all nodes
- **Debugging difficulty**: Cross-language stack traces
- **Dependency management**: Two package ecosystems

**Mitigation:**
- Use Docker with frozen Python environment
- Comprehensive logging at language boundary
- Document Python setup clearly

### Native Port Risks

**Technical Risks:**
- **Incomplete symbolic math**: Exun less mature than SymPy
- **Edge case divergence**: May not match math-verify behavior
- **LaTeX parsing complexity**: Need to handle full LaTeX math spec

**Mitigation:**
- Extensive test suite comparing with math-verify
- Gradual rollout with validation
- Focus on most common expression types first

**Resource Risks:**
- **Development time**: 2-4 weeks of focused work
- **Maintenance burden**: Ongoing updates as requirements evolve

**Mitigation:**
- Start with minimal viable implementation
- Expand coverage based on actual usage patterns
- Keep Python fallback for edge cases

---

## 9. Implementation Checklist

### Pythonx Quick Start (1 day)

- [ ] Add `{:pythonx, "~> 0.4.7"}` to mix.exs
- [ ] Install math-verify in Python environment
- [ ] Create `Tinkex.Evaluation.MathGrader` module
- [ ] Write basic test suite
- [ ] Document Python setup requirements
- [ ] Add error handling and timeouts

### Snakepit Production (1 week)

- [ ] Add `{:snakepit, "~> 0.6.3"}` to mix.exs
- [ ] Implement Python worker script with math-verify
- [ ] Configure worker pool (size, heartbeat, restart strategy)
- [ ] Set up gRPC communication
- [ ] Write integration tests
- [ ] Add telemetry and monitoring
- [ ] Create deployment guide

### Native Elixir Port (4 weeks)

**Week 1: Foundation**
- [ ] Evaluate Exun, Abacus, Leibniz
- [ ] Design AST structure for expressions
- [ ] Implement basic expression parser
- [ ] Set up test harness against math-verify

**Week 2: LaTeX Support**
- [ ] Implement LaTeX lexer (leex)
- [ ] Implement LaTeX parser (yecc)
- [ ] Add normalization rules
- [ ] Test against LaTeX examples

**Week 3: Symbolic Comparison**
- [ ] Integrate Exun for symbolic operations
- [ ] Implement difference-based checking
- [ ] Handle sets, intervals, inequalities
- [ ] Add tuple/multi-element support

**Week 4: Polish**
- [ ] Comprehensive test suite
- [ ] Performance benchmarks
- [ ] Edge case handling
- [ ] Documentation and examples

---

## 10. Conclusion

The `math-verify` library is a well-designed, pure Python solution with no native dependencies, making it straightforward to integrate with Elixir via Python interop tools. For the North-Shore-AI/Tinkex project, I recommend:

1. **Immediate**: Use Pythonx for experimentation (0-1 month)
2. **Near-term**: Deploy Snakepit for production RL pipelines (1-3 months)
3. **Long-term**: Develop native Elixir implementation (3-6 months)

This phased approach balances speed to market with long-term sustainability and allows validation of the native implementation against the proven Python solution.

**Key Takeaway:** No C++ barriers to Python interop, but native Elixir port is feasible and recommended for production maturity.

---

## References

### Primary Sources
- [math-verify PyPI](https://pypi.org/project/math-verify/)
- [HuggingFace Math-Verify GitHub](https://github.com/huggingface/Math-Verify)
- [latex2sympy2_extended GitHub](https://github.com/huggingface/latex2sympy2_extended)

### Dependencies
- [SymPy Documentation - Dependencies](https://docs.sympy.org/latest/contributing/dependencies.html)
- [antlr4-python3-runtime PyPI](https://pypi.org/project/antlr4-python3-runtime/)
- [latex2sympy2_extended Dependencies](https://socket.dev/pypi/package/latex2sympy2-extended/dependencies/1.10.1/tar-gz)

### Elixir Integration
- [Pythonx GitHub](https://github.com/livebook-dev/pythonx)
- [Snakepit GitHub](https://github.com/nshkrdotcom/snakepit)
- [Dashbit - Embedding Python in Elixir](https://dashbit.co/blog/running-python-in-elixir-its-fine)
- [Curiosum - Python Libraries in Elixir](https://www.curiosum.com/blog/borrowing-libs-from-python-in-elixir)

### Elixir Math Libraries
- [Exun GitHub](https://github.com/Nazari/exun)
- [SymMath GitHub](https://github.com/genericsoma/SymMath)
- [Abacus GitHub](https://github.com/narrowtux/abacus)
- [expr GitHub](https://github.com/Rob-bie/expr)
- [Leibniz on ElixirStatus](https://elixirstatus.com/p/Z26H-leibniz-math-expression-parser-and-evaluator)

### Technical Background
- [SymPy Wikipedia](https://en.wikipedia.org/wiki/SymPy)
- [List of Computer Algebra Systems](https://en.wikipedia.org/wiki/List_of_computer_algebra_systems)
- [ANTLR Download](https://www.antlr.org/download.html)
