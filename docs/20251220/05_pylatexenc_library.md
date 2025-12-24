# pylatexenc Library Research Report

**Date:** 2025-12-20
**Project:** tinkex / tinker-cookbook integration analysis

**VERIFIED: 2025-12-20**

## Executive Summary

`pylatexenc` is a pure Python library for parsing LaTeX code and providing bidirectional conversion between LaTeX and Unicode. It is **highly suitable for wrapping with Elixir** via pythonx or similar Python interop libraries, as it has no C++ or native dependencies and implements a well-defined parsing functionality that can be easily called from Elixir.

---

## 1. Library Purpose and Functionality

### Overview

`pylatexenc` is a simple LaTeX parser providing latex-to-unicode and unicode-to-latex conversion. Rather than being a full (La)TeX engine, it treats LaTeX as markup code, parsing its structural elements into logical object representations.

### Core Modules

1. **`pylatexenc.latexencode`**
   - Function: `unicode_to_latex()` converts Unicode strings into LaTeX text and escape sequences
   - Recognizes accented characters and most math symbols
   - Handles special character encoding for LaTeX compatibility

2. **`pylatexenc.latexwalker`**
   - Parses the LaTeX structure of given LaTeX code
   - Returns a logical structure of objects suitable for conversion to other formats
   - Provides the foundation for other modules in the library

3. **`pylatexenc.latex2text`**
   - Built on top of `latexwalker`
   - Converts LaTeX code to plain text with Unicode characters
   - Command-line tool available: `latex2text`

### Command-Line Tools

The functionality is exposed via three command-line scripts:
- `latexencode` - Encode Unicode to LaTeX
- `latex2text` - Convert LaTeX to plain text
- `latexwalker` - Parse LaTeX structure

### Usage in tinker-cookbook

In the tinker-cookbook codebase, `pylatexenc` is used specifically for mathematical answer processing:

**File:** `tinker_cookbook/recipes/math_rl/math_grading.py` (lines 233-248)

```python
from pylatexenc import latex2text

def _parse_latex(expr: str) -> str:
    """Attempts to parse latex to an expression sympy can read."""
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)

    # Replace the specific characters that this parser uses.
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")

    return expr.strip()
```

**Actual Usage:** The library normalizes LaTeX mathematical expressions so they can be evaluated with SymPy for answer grading in reinforcement learning training for math problems. Currently used **directly in Python** with no Elixir wrapper.

**File:** `tinker_cookbook/rl/train.py` (line 1049)

```python
# Logging configuration to reduce noise during training
logging.getLogger("pylatexenc").setLevel(logging.WARNING)
```

---

## 2. Repository and Package Information

### GitHub Repository
- **URL:** https://github.com/phfaist/pylatexenc
- **Description:** "Simple LaTeX parser providing latex-to-unicode and unicode-to-latex conversion"
- **License:** MIT License
- **Contributors:** 10 or fewer contributors
- **Stars:** 375+ (as of research date)

### PyPI Package
- **URL:** https://pypi.org/project/pylatexenc/
- **Installation:** `pip install pylatexenc`
- **Python Compatibility:** Python ≥ 3.4 or ≥ 2.7
- **Current Version:** 3.0alpha (pre-release as of April 2023)
  - Note: Version 3.0alpha introduces new features and major changes
  - Documentation is still incomplete for v3.0
  - New APIs are subject to changes
- **Stable Version:** 2.10 (widely available on distributions)

### Documentation
- **URL:** https://pylatexenc.readthedocs.io/
- **Status:** Comprehensive documentation for v2.x; v3.0 documentation incomplete

### Distribution Availability
- **Conda-forge:** Available as `noarch` package (confirms no architecture-specific code)
- **PyPI:** Standard Python package
- **Linux Distributions:**
  - Debian/Ubuntu: `python3-pylatexenc`
  - Arch Linux: `python-pylatexenc`
  - Gentoo: `dev-python/pylatexenc`
  - PureOS: Available (architecture: `all`)

---

## 3. Dependencies and C++ Analysis

### Pure Python Implementation

**Conclusion: pylatexenc has NO C++ or native dependencies.**

**Evidence:**

1. **Package Type:**
   - Conda-forge lists it as `noarch`, meaning no platform-specific compiled code
   - Debian/Ubuntu package specifies "Architecture: all"
   - Available through piwheels (pure Python wheel distribution)

2. **Installation:**
   - Simple pip installation with no compiler requirements
   - No mention of build tools (gcc, clang, etc.) in documentation
   - Works on all platforms without platform-specific wheels

3. **External References:**
   - Qiskit issue #2417 noted: "pylatexenc has no other external dependencies and just returns a string"
   - The library is described as a straightforward Python parser

4. **Repository Structure:**
   - No C extension files (.c, .cpp, .pyx)
   - Uses Poetry for build system (pure Python build)
   - JavaScript transcription exists (in `js-transcrypt/` folder), proving the logic is language-agnostic

### Minimal Dependencies

From the dependency analysis:
- **Runtime:** Pure Python standard library only
- **Optional:** Development dependencies for documentation (Sphinx, etc.)
- **No external libraries required** for core functionality

### Python Version Support

The library is designed to be backwards-compatible:
- Supports Python 2.7 (legacy)
- Supports Python 3.4+
- Well-tested across multiple Python versions

---

## 4. Elixir Wrapping Feasibility Assessment

### High Suitability for Elixir Integration

**Overall Assessment: EXCELLENT candidate for Elixir wrapping**

### Wrapping Options

#### Option 1: Pythonx (Recommended for Simple Use Cases)

**Description:** Pythonx embeds a Python interpreter directly in the BEAM VM using Erlang NIFs.

**Advantages:**
- Direct in-process execution (low latency)
- Transparent data type conversion between Elixir and Python
- Well-integrated with Livebook
- Suitable for computational workloads (releases GIL for CPU-intensive operations)

**Disadvantages:**
- Global Interpreter Lock (GIL) limits concurrency
- Calling from multiple Elixir processes doesn't provide true parallelism
- Can be a bottleneck if called frequently from many processes
- Shared process memory means Python crashes can crash BEAM

**Best for:**
- Single-process or low-concurrency use cases
- Quick prototyping and integration
- Livebook notebooks
- Batch processing where latency isn't critical

**Example Usage Pattern:**
```elixir
# Using Pythonx to call pylatexenc
defmodule TinkexLatex do
  def latex_to_text(latex_string) do
    Pythonx.eval("""
    from pylatexenc.latex2text import LatexNodes2Text
    LatexNodes2Text().latex_to_text(#{inspect(latex_string)})
    """)
  end
end
```

**References:**
- GitHub: https://github.com/livebook-dev/pythonx
- Docs: https://hexdocs.pm/pythonx/

#### Option 2: Snakepit (Recommended for Production Use)

**Description:** High-performance process pooler and session manager for external language integrations.

**Advantages:**
- Battle-tested for ML/AI integrations
- True process isolation (Python crashes don't crash BEAM)
- Lightning-fast concurrent initialization (1000x faster than sequential)
- Session-based execution with automatic worker affinity
- gRPC-based communication with HTTP/2 streaming support
- Native streaming support for real-time progress updates
- Supports multiple external languages (Python, Node.js, Ruby, R)

**Disadvantages:**
- More complex setup than Pythonx
- Additional overhead from process communication
- Requires gRPC infrastructure

**Best for:**
- Production systems requiring high reliability
- High-concurrency scenarios
- Long-running Python sessions
- When process isolation is critical
- Streaming or real-time processing needs

**References:**
- GitHub: https://github.com/nshkrdotcom/snakepit

#### Option 3: ErlPort (Legacy Option)

**Description:** Erlang library for connecting to multiple programming languages via port protocol.

**Advantages:**
- Mature and stable
- Part of Erlang ecosystem
- Process-based isolation

**Disadvantages:**
- Slower than modern alternatives
- Less ergonomic API
- Limited type conversion
- No modern streaming capabilities

**Best for:**
- Legacy systems already using ErlPort
- Simple, low-frequency calls

#### Option 4: HTTP/REST API (Alternative Approach)

**Description:** Wrap pylatexenc in a lightweight Python web service (Flask, FastAPI).

**Advantages:**
- Language-agnostic
- Easy horizontal scaling
- Complete process isolation
- Simple deployment model

**Disadvantages:**
- Network latency overhead
- Additional infrastructure to manage
- Serialization/deserialization costs

**Best for:**
- Microservices architecture
- When pylatexenc is one of many external dependencies
- Multi-language teams

### Complexity Assessment

**Wrapping Complexity: LOW**

Reasons:
1. **Pure Python:** No compilation or native dependencies
2. **Simple API:** Clear function-based interface
3. **Stateless Operations:** Most operations are pure functions
4. **No Global State:** Each parse operation is independent
5. **Good Error Handling:** Exceptions are well-defined

### Performance Considerations

For the specific use case in tinker-cookbook (math answer grading):

**Current Usage Pattern:**
```python
# One-off latex-to-text conversions
expr = latex2text.LatexNodes2Text().latex_to_text(expr)
```

**Elixir Wrapping Performance:**
- **Pythonx:** ~1-10ms overhead per call (acceptable for grading)
- **Snakepit:** ~5-20ms overhead per call (includes gRPC round-trip)
- **HTTP API:** ~10-50ms overhead per call (includes network)

**Recommendation:** For tinker-cookbook's math grading use case, **Pythonx** is sufficient. The calls are infrequent (once per answer), and the GIL limitation is not a concern.

### Recommended Architecture for tinkex

```
┌─────────────────────────────────────────┐
│  Tinkex (Elixir Application)            │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │  TinkexLatex Module                │ │
│  │  (Elixir wrapper for pylatexenc)   │ │
│  │                                    │ │
│  │  - latex_to_text/1                │ │
│  │  - unicode_to_latex/1             │ │
│  └─────────────┬──────────────────────┘ │
│                │                         │
│                │ (Pythonx NIF)          │
│                ▼                         │
│  ┌────────────────────────────────────┐ │
│  │  Embedded Python Interpreter       │ │
│  │  (pylatexenc library loaded)       │ │
│  └────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

---

## 5. Elixir Alternatives for LaTeX Processing

### Native Elixir Options

After extensive research, **no mature native Elixir libraries for LaTeX parsing exist**.

**Available Elixir Libraries (Related but not LaTeX-specific):**

1. **unicode** (elixir-unicode/unicode)
   - Purpose: Unicode codepoint introspection and detection
   - Not LaTeX-specific but provides Unicode utilities
   - Could be used for character normalization
   - URL: https://github.com/elixir-unicode/unicode

2. **unicode_set** (elixir-unicode/unicode_set)
   - Purpose: Unicode set parsing, expansion, macros and guards
   - Not LaTeX-specific
   - URL: https://github.com/elixir-unicode/unicode_set

3. **text** (kipcole9/text)
   - Purpose: Text segmentation (graphemes, words, sentences)
   - Based on Unicode standard and CLDR data
   - Not LaTeX-specific
   - URL: https://github.com/kipcole9/text

4. **TextParser** (Elixir)
   - Purpose: Extract and validate structured tokens from text
   - Handles emoji and Unicode characters
   - Not LaTeX-specific but could be extended
   - URL: Announced at https://solnic.dev/posts/announcing-textparser-for-elixir/

**Analysis:** None of these libraries handle LaTeX syntax. Building a LaTeX parser from scratch in Elixir would be a significant undertaking.

### Alternative Approaches

#### Option A: Port pylatexenc to Elixir (Not Recommended)

**Effort Estimate:** 4-8 weeks for full port

**Pros:**
- Pure Elixir solution
- No Python dependency
- Full control over implementation

**Cons:**
- High development and maintenance cost
- Reinventing a well-tested wheel
- LaTeX specification is complex
- Ongoing sync with LaTeX standard changes
- Limited Elixir community interest in LaTeX

**Verdict:** Not worth the effort given pylatexenc's simplicity and Python interop options.

#### Option B: Use Existing LaTeX Tools via Ports

**Options:**
1. **pandoc** - Universal document converter (includes LaTeX)
   - Could call via System.cmd/3
   - Heavy dependency (Haskell-based)
   - Overkill for simple LaTeX-to-text conversion

2. **LaTeX/TeX binaries** - Full TeX distribution
   - Call `latex`, `pdflatex`, etc. via ports
   - Extremely heavy dependency
   - Not suitable for parsing/conversion only

**Verdict:** These tools are too heavyweight for the simple conversion use case.

#### Option C: Embrace Python Interop (Recommended)

**Rationale:**
- Python ML/AI ecosystem integration is already a goal for tinkex
- Many ML libraries (numpy, scipy, transformers) require Python
- Building a robust Python interop layer benefits the entire project
- pylatexenc is just one of many Python dependencies in tinker-cookbook

**Strategic Benefits:**
1. Enables use of Python ML libraries in Elixir
2. Pythonx/Snakepit can handle multiple Python dependencies
3. Elixir handles orchestration, Python handles specialized computation
4. Best-of-both-worlds architecture

---

## 6. Recommendations

**CURRENT STATE (2025-12-20):** The tinker-cookbook uses pylatexenc **directly in Python** with no Elixir integration. The Pythonx/Snakepit recommendations below are for **future tinkex integration** only.

### For tinkex Integration

**Primary Recommendation: Use Pythonx wrapper for pylatexenc**

**Implementation Plan:**

1. **Phase 1: Basic Wrapper (Week 1)**
   - Add pythonx dependency to tinkex
   - Create `Tinkex.LaTeX` module with core functions:
     - `latex_to_text/1`
     - `unicode_to_latex/1`
   - Write unit tests using tinker-cookbook math examples
   - Document API in ExDoc

2. **Phase 2: Error Handling (Week 2)**
   - Add proper error handling for malformed LaTeX
   - Add timeout protection (using `Task.await/2` with timeout)
   - Add input validation
   - Add telemetry for monitoring

3. **Phase 3: Optimization (Optional)**
   - Profile performance for typical workloads
   - Consider caching for repeated conversions
   - If performance issues arise, evaluate Snakepit migration

**Example Implementation:**

```elixir
defmodule Tinkex.LaTeX do
  @moduledoc """
  LaTeX parsing and conversion utilities using pylatexenc.
  """

  @doc """
  Converts LaTeX markup to Unicode plain text.

  ## Examples

      iex> Tinkex.LaTeX.latex_to_text("\\\\frac{1}{2}")
      {:ok, "1/2"}

      iex> Tinkex.LaTeX.latex_to_text("\\\\sqrt{2}")
      {:ok, "√2"}
  """
  @spec latex_to_text(String.t()) :: {:ok, String.t()} | {:error, term()}
  def latex_to_text(latex) when is_binary(latex) do
    python_code = """
    from pylatexenc.latex2text import LatexNodes2Text
    result = LatexNodes2Text().latex_to_text(#{inspect(latex)})
    result
    """

    case Pythonx.eval(python_code) do
      {:ok, result} -> {:ok, result}
      {:error, _} = error -> error
    end
  rescue
    e -> {:error, Exception.message(e)}
  end

  @doc """
  Converts Unicode text to LaTeX markup.

  ## Examples

      iex> Tinkex.LaTeX.unicode_to_latex("π")
      {:ok, "\\\\pi"}
  """
  @spec unicode_to_latex(String.t()) :: {:ok, String.t()} | {:error, term()}
  def unicode_to_latex(text) when is_binary(text) do
    python_code = """
    from pylatexenc.latexencode import unicode_to_latex
    result = unicode_to_latex(#{inspect(text)})
    result
    """

    case Pythonx.eval(python_code) do
      {:ok, result} -> {:ok, result}
      {:error, _} = error -> error
    end
  rescue
    e -> {:error, Exception.message(e)}
  end
end
```

### For Production Systems

If tinkex evolves to require high-throughput LaTeX processing:

1. **Benchmark** current Pythonx performance
2. **Profile** bottlenecks in the processing pipeline
3. **Consider** Snakepit if:
   - Processing >100 LaTeX strings/second
   - Multiple concurrent Elixir processes need LaTeX parsing
   - Python crashes are causing system instability
4. **Implement** connection pooling for Snakepit workers
5. **Monitor** processing latency and error rates

### Documentation Requirements

Create the following documentation:

1. **Installation Guide**
   - How to install pythonx
   - How to install pylatexenc in the Python environment
   - Virtual environment setup for development

2. **Usage Examples**
   - Common math notation conversions
   - Error handling patterns
   - Integration with math grading workflows

3. **Troubleshooting**
   - Common LaTeX parsing errors
   - Python environment issues
   - Performance tuning tips

### Testing Strategy

1. **Unit Tests**
   - Test common LaTeX expressions from tinker-cookbook
   - Test Unicode character roundtrips
   - Test error cases (malformed LaTeX)

2. **Integration Tests**
   - Test with actual math problem datasets
   - Test with sympy integration (if applicable)
   - Test timeout behavior

3. **Property Tests**
   - LaTeX-to-text-to-latex roundtrip properties
   - Character preservation properties

---

## 7. Conclusion

**pylatexenc is an ideal candidate for Elixir integration:**

- ✅ Pure Python (no C++ dependencies)
- ✅ Simple, well-defined API
- ✅ Stateless operations
- ✅ MIT licensed
- ✅ Well-maintained and documented
- ✅ Widely used in the Python ecosystem

**Recommended approach:**

1. Use **Pythonx** for initial integration (simple, low overhead)
2. Wrap in a clean Elixir API (`Tinkex.LaTeX` module)
3. Add proper error handling and monitoring
4. Migrate to **Snakepit** only if performance or isolation becomes an issue

**No need to build a native Elixir LaTeX parser** - the effort would not be justified given excellent Python interop options and the specialized nature of LaTeX parsing.

---

## References

### pylatexenc Resources
- [GitHub Repository](https://github.com/phfaist/pylatexenc)
- [PyPI Package](https://pypi.org/project/pylatexenc/)
- [Documentation](https://pylatexenc.readthedocs.io/)

### Elixir Python Interop
- [Pythonx GitHub](https://github.com/livebook-dev/pythonx)
- [Pythonx Documentation](https://hexdocs.pm/pythonx/)
- [Snakepit GitHub](https://github.com/nshkrdotcom/snakepit)
- [Dashbit Blog: Embedding Python in Elixir](https://dashbit.co/blog/running-python-in-elixir-its-fine)
- [Curiosum: Python Libraries in Elixir](https://www.curiosum.com/blog/borrowing-libs-from-python-in-elixir)
- [Medium: How we use Python within Elixir](https://medium.com/stuart-engineering/how-we-use-python-within-elixir-486eb4d266f9)

### Elixir Unicode Libraries
- [elixir-unicode/unicode](https://github.com/elixir-unicode/unicode)
- [elixir-unicode/unicode_set](https://github.com/elixir-unicode/unicode_set)
- [kipcole9/text](https://github.com/kipcole9/text)

---

**Report compiled:** 2025-12-20
**Next steps:** Implement Tinkex.LaTeX wrapper module using Pythonx
