# HuggingFace Datasets & Blobfile: Actual Usage Analysis

**Date**: 2025-12-20
**Analysis**: Real-world usage patterns in tinker-cookbook (CLIENT-SIDE)
**Purpose**: Determine migration strategy for tinkex_cookbook

---

**Update (2025-12-22):** `crucible_datasets` v0.4.1 (Hex) now provides native
HuggingFace Hub loading, streaming, and all required tinker datasets. The
Pythonx wrapper recommendation in this document is historical; use
`CrucibleDatasets.load_dataset/2` or dataset-specific loaders for cookbook
integration.

## Executive Summary

**Critical Finding**: The cookbook is **CLIENT-SIDE** data loading code. It loads datasets locally, processes them into training examples, and sends them to Tinker API for training. There is NO server-side dataset management.

### Usage Breakdown

| Library | Usage Count | Primary Purpose | Migration Complexity |
|---------|-------------|-----------------|---------------------|
| `datasets` | 21 instances | HuggingFace Hub downloads + in-memory operations | **MEDIUM** |
| `blobfile` | 3 instances | Read local/cloud JSONL files | **LOW** |

### Recommendation: **HYBRID APPROACH**

1. **For HuggingFace datasets**: Use Pythonx wrapper (unavoidable Python dependency)
2. **For blobfile**: Replace with native Elixir (File/ExAws.S3)
3. **For local processing**: Implement in pure Elixir where feasible

---

## Part 1: HuggingFace `datasets` Library Usage

### 1.1 Loading Operations (21 total instances across 8 files)

#### A. Direct HuggingFace Hub Downloads

**Pattern**: `datasets.load_dataset("org/repo", split="train")`

```python
# File: tinker_cookbook/recipes/chat_sl/chat_datasets.py
dataset = datasets.load_dataset("allenai/tulu-3-sft-mixture")  # Line 25
dataset = datasets.load_dataset("HuggingFaceH4/no_robots")     # Line 55

# File: tinker_cookbook/recipes/math_rl/math_env.py
ds = load_dataset("HuggingFaceH4/MATH-500", name="default", split="test")  # Line 112
ds = load_dataset("EleutherAI/hendrycks_math", name=cfg, split=split)     # Line 133
ds = load_dataset("allenai/tulu-3-sft-mixture", split="train")            # Line 202
ds = load_dataset("POLARIS-Project/Polaris-Dataset-53K", split="train")   # Line 229
ds = load_dataset("zwhe99/DeepMath-103K", split="train")                  # Line 282
ds = load_dataset("openai/gsm8k", name="main", split=split)               # Line 335

# File: tinker_cookbook/recipes/preference/datasets.py
dataset = datasets.load_dataset("allenai/llama-3.1-tulu-3-8b-preference-mixture", split="train")  # Line 108
dataset = datasets.load_dataset("Anthropic/hh-rlhf")                                             # Line 139
dataset = datasets.load_dataset("nvidia/HelpSteer3", "preference")                               # Line 154
dataset = datasets.load_dataset("argilla/ultrafeedback-binarized-preferences", split="train")    # Line 187
dataset = datasets.load_dataset("lmarena-ai/arena-human-preference-140k", split="train")         # Line 215
dataset = datasets.load_dataset("nvidia/HelpSteer2", split="train")                              # Line 264

# File: tinker_cookbook/distillation/datasets.py
ds = load_dataset("zwhe99/DeepMath-103K", split=split)          # Line 185
ds = load_dataset("allenai/tulu-3-sft-mixture", split="train")  # Line 202
```

**Analysis**:
- **15+ distinct HuggingFace datasets** downloaded from Hub
- Requires HuggingFace Hub API authentication
- Datasets are cached locally by `datasets` library (typically in `~/.cache/huggingface/datasets/`)

#### B. Dataset Construction from Local Data

```python
# File: tinker_cookbook/supervised/data.py (FromConversationFileBuilder)
# Lines 117-129
conversations = []
with blobfile.BlobFile(self.file_path, "r", streaming=False) as f:
    for line in f:
        data = json.loads(line.strip())
        conversations.append(data)

dataset = datasets.Dataset.from_list(conversations)  # <-- CREATE IN-MEMORY DATASET
```

```python
# File: tinker_cookbook/preference/preference_datasets.py (ComparisonBuilderFromJsonl)
# Lines 136-141
train_data = []
with blobfile.BlobFile(self.train_path, "r", streaming=False) as f:
    for line in f:
        train_data.append(json.loads(line.strip()))

train_dataset = datasets.Dataset.from_list(train_data)  # <-- CREATE IN-MEMORY DATASET
```

```python
# File: tinker_cookbook/recipes/preference/datasets.py (HelpSteer2ComparisonBuilder)
# Lines 294-296
df = pd.DataFrame(comparisons)
dataset = datasets.Dataset.from_pandas(df)  # <-- FROM PANDAS DATAFRAME
```

**Analysis**:
- Loads JSONL → Python list → HuggingFace Dataset (in-memory)
- Could be replaced with Explorer DataFrames + Parquet

### 1.2 Dataset Operations (In-Memory Transformations)

#### A. Shuffling & Splitting

```python
# Pattern: shuffle(seed) → take(N) / skip(N)
dataset = dataset.shuffle(seed=0)      # Randomize order
test_ds = dataset.take(1024)           # First 1024 examples
train_ds = dataset.skip(1024)          # Rest of examples
```

**Usage locations**:
- `chat_datasets.py`: Lines 28-30 (Tulu3), Lines 59 (NoRobots)
- `preference/datasets.py`: Lines 112-114, 142, 156-157, 189-192, 218-220, 297
- `math_env.py`: Lines 152, 282, 337
- `sl_loop.py`: Line 81

**Equivalent in Explorer/Elixir**:
```elixir
df = Explorer.DataFrame.shuffle(df, seed: 0)
{train_df, test_df} = Explorer.DataFrame.split(df, 0.9)
# OR
test_df = Explorer.DataFrame.slice(df, 0, 1024)
train_df = Explorer.DataFrame.slice(df, 1024, -1)
```

#### B. Row Selection (Batching)

```python
# Pattern: select(range(start, end))
batch_rows = train_dataset.select(range(batch_start, batch_end))
```

**Usage locations**:
- `math_env.py`: Line 166, 353
- `sl_loop.py`: Line 106
- `supervised/data.py`: Lines 50-52 (SupervisedDatasetFromHFDataset)

**Equivalent in Explorer**:
```elixir
batch_df = Explorer.DataFrame.slice(df, batch_start, batch_end - batch_start)
```

#### C. Filtering

```python
# Pattern: filter(lambda example: condition)
ds = ds.filter(lambda example: example["problem"] not in test_problems)
```

**Usage location**: `math_env.py`: Line 134

**Equivalent in Explorer**:
```elixir
df = Explorer.DataFrame.filter(df, not problem in ^test_problems)
```

#### D. Dataset Concatenation

```python
# Pattern: concatenate_datasets([ds1, ds2, ...])
full_dataset = concatenate_datasets(pieces)
```

**Usage location**: `math_env.py`: Line 136

**Equivalent in Explorer**:
```elixir
df = Explorer.DataFrame.concat_rows([df1, df2, df3])
```

### 1.3 Iteration Patterns

#### A. Batch Iteration (Standard)

```python
# Pattern 1: Select range → Convert to list of dicts
rows = self.shuffle_dataset.select(range(index * batch_size, (index + 1) * batch_size))
return [self.map_fn(row) for row in rows.to_list()]
```

**Usage**: `supervised/data.py`: Lines 49-57 (SupervisedDatasetFromHFDataset)

#### B. Streaming Iteration

```python
# Pattern 2: IterableDataset with shuffle buffer
self.hf_dataset = hf_dataset.shuffle(seed=0, buffer_size=10_000).batch(
    batch_size=batch_size, drop_last_batch=True
)
self.dataset_iterator = iter(self.hf_dataset)
batch = next(self.dataset_iterator)
```

**Usage**: `supervised/data.py`: Lines 77-99 (StreamingSupervisedDatasetFromHFDataset)

**Analysis**: Streaming mode used for very large datasets that don't fit in memory.

### 1.4 Custom Classes Wrapping HF Datasets

| Class | Purpose | HF Dataset Type |
|-------|---------|-----------------|
| `SupervisedDatasetFromHFDataset` | In-memory dataset with map/flatmap | `datasets.Dataset` |
| `StreamingSupervisedDatasetFromHFDataset` | Streaming dataset for large data | `datasets.IterableDataset` |
| `MathDataset` | Math problem environment wrapper | `datasets.Dataset` |
| `Gsm8kDataset` | GSM8K math problems | `datasets.Dataset` |
| `PolarisDataset` | Polaris math problems | `datasets.Dataset` |
| `DeepMathDataset` | DeepMath problems | `datasets.Dataset` |
| `PromptOnlyDataset` | Distillation prompts | `datasets.Dataset` |

---

## Part 2: Blobfile Library Usage

### 2.1 All Blobfile Instances (3 total)

#### Instance 1: Supervised Data Loading

```python
# File: tinker_cookbook/supervised/data.py
# Lines 119-126

with blobfile.BlobFile(self.file_path, "r", streaming=False) as f:
    for line in f:
        data = json.loads(line.strip())
        if "messages" not in data:
            raise ValueError(f"Each line must contain 'messages' field. Got: {data.keys()}")
        conversations.append(data)
```

**Analysis**:
- Opens JSONL file for reading
- `streaming=False` → Read entire file into memory
- Could be local file path or cloud URL (e.g., `s3://bucket/file.jsonl`)

#### Instance 2 & 3: Preference Data Loading

```python
# File: tinker_cookbook/preference/preference_datasets.py
# Lines 137-139 (train dataset)

train_data = []
with blobfile.BlobFile(self.train_path, "r", streaming=False) as f:
    for line in f:
        train_data.append(json.loads(line.strip()))

# Lines 147-149 (test dataset)
test_data = []
with blobfile.BlobFile(self.test_path, "r", streaming=False) as f:
    for line in f:
        test_data.append(json.loads(line.strip()))
```

**Analysis**:
- Same pattern: read JSONL line-by-line
- Two separate files (train/test splits)

### 2.2 Blobfile Feature Usage

| Feature | Used? | Replacement |
|---------|-------|-------------|
| Read text files | ✅ Yes | `File.stream!/1` (local), `ExAws.S3.download_file/4` (cloud) |
| Write files | ❌ No | N/A |
| List directories | ❌ No | N/A |
| Copy/move files | ❌ No | N/A |
| Cloud storage (S3/GCS) | ⚠️ Implicit | `ExAws.S3` |

**Key Observation**: Blobfile is only used for **simple file reading**. The library's main value is abstracting local vs. cloud paths, but the cookbook doesn't explicitly demonstrate cloud usage.

---

## Part 3: Data Flow Architecture

### 3.1 CLIENT-SIDE Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     CLIENT (tinker-cookbook)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. DOWNLOAD DATASET                                            │
│     ├─ HuggingFace Hub (datasets.load_dataset)                 │
│     ├─ Local JSONL (blobfile.BlobFile)                         │
│     └─ Cloud storage (blobfile with s3://)                     │
│                                                                 │
│  2. IN-MEMORY PROCESSING                                        │
│     ├─ Shuffle (dataset.shuffle)                               │
│     ├─ Split (dataset.take/skip)                               │
│     ├─ Filter (dataset.filter)                                 │
│     └─ Batch (dataset.select)                                  │
│                                                                 │
│  3. CONVERT TO TRAINING FORMAT                                  │
│     ├─ Tokenize conversations (renderer.build_supervised_example)│
│     ├─ Create tinker.Datum objects                             │
│     └─ Apply loss masks (weights)                              │
│                                                                 │
│  4. SEND TO TINKER API                                          │
│     └─ training_client.forward_backward(batch)                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                               │
                               │ HTTPS (tinker.Datum[] serialized)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TINKER API (server-side)                     │
├─────────────────────────────────────────────────────────────────┤
│  - Distributed training                                         │
│  - GPU orchestration                                            │
│  - Checkpoint management                                        │
└─────────────────────────────────────────────────────────────────┘
```

**Key Insight**: The cookbook never sends raw datasets to Tinker. It processes data locally and sends tokenized batches.

### 3.2 Dataset Types & Processing

| Dataset Type | Example | Processing Strategy |
|-------------|---------|---------------------|
| **Chat/SFT** | Tulu3, NoRobots | Extract `messages` field → Tokenize conversation → Mask prompts |
| **Preference** | HHH, UltraFeedback | Parse chosen/rejected pairs → Create comparison prompts |
| **Math** | MATH-500, GSM8K | Extract problem/answer → Format with `\boxed{}` → Verify format |
| **Distillation** | DeepMath prompts | Extract prompt only → No answer checking |

---

## Part 4: Can We Use Explorer + Parquet Instead of HuggingFace Datasets?

### 4.1 Feasibility Analysis

#### What We CAN Replace (Local Processing)

✅ **In-memory operations**:
- `dataset.shuffle(seed=0)` → `Explorer.DataFrame.shuffle(df, seed: 0)`
- `dataset.take(N)` → `Explorer.DataFrame.slice(df, 0, N)`
- `dataset.skip(N)` → `Explorer.DataFrame.slice(df, N, -1)`
- `dataset.select(range(i, j))` → `Explorer.DataFrame.slice(df, i, j-i)`
- `dataset.filter(fn)` → `Explorer.DataFrame.filter(df, condition)`
- `concatenate_datasets([...])` → `Explorer.DataFrame.concat_rows([...])`

✅ **Local file loading**:
- `Dataset.from_list(json_data)` → `Explorer.DataFrame.new(map_data)`
- JSONL parsing → `File.stream! → Jason.decode! → DataFrame.new`

✅ **Iteration**:
- `for row in dataset` → `Explorer.DataFrame.to_rows_stream(df)`

#### What We CANNOT Replace (HuggingFace Hub)

❌ **HuggingFace Hub downloads**:
- `load_dataset("allenai/tulu-3-sft-mixture")` requires HF API
- Dataset metadata (splits, configs, features) managed by HF
- Authentication/credentials for private datasets
- Streaming from HF Hub for large datasets

❌ **Complex dataset formats**:
- Parquet files with nested structures (e.g., lists of messages)
- Multi-file datasets with sharding
- Dataset cards and metadata

### 4.2 Parquet Compatibility

**Explorer supports Parquet**, but HuggingFace datasets use specific schema conventions:

```python
# HuggingFace dataset structure (example)
{
  "messages": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi!"}
  ],
  "source": "tulu3",
  "id": "12345"
}
```

Explorer can read this if:
1. Parquet file uses compatible types (nested lists → Arrow list types)
2. Schema is known upfront
3. Files are available locally or via HTTP/S3

**Challenge**: HuggingFace datasets are NOT just Parquet files. They include:
- Multiple splits (train/test/validation)
- Configuration variants (e.g., `hendrycks_math` has 8+ configs)
- Streaming capabilities
- Lazy loading

---

## Part 5: Migration Strategies for Tinkex_Cookbook

### 5.1 Strategy A: Pure Pythonx Wrapper (Simplest)

**Approach**: Wrap entire `datasets` + `blobfile` usage in Pythonx modules.

```elixir
# lib/tinkex_cookbook/datasets/loader.ex
defmodule TinkexCookbook.Datasets.Loader do
  use Pythonx

  defpymodule DatasetLoader do
    """
    import datasets
    import blobfile
    import json

    def load_hf_dataset(name, split=None):
        return datasets.load_dataset(name, split=split)

    def load_jsonl(path):
        data = []
        with blobfile.BlobFile(path, "r", streaming=False) as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    """
  end

  def load_huggingface(name, split \\ nil) do
    DatasetLoader.load_hf_dataset(name, split)
    |> pythonx_to_elixir()
  end

  def load_jsonl(path) do
    DatasetLoader.load_jsonl(path)
    |> pythonx_to_elixir()
  end
end
```

**Pros**:
- ✅ Zero reimplementation effort
- ✅ Maintains compatibility with all HF datasets
- ✅ Handles cloud storage via blobfile

**Cons**:
- ❌ Python runtime dependency
- ❌ Pythonx conversion overhead (Python dicts → Elixir maps)
- ❌ No type safety
- ❌ Debugging across language boundary

### 5.2 Strategy B: Hybrid Approach (Recommended)

**Approach**: Native Elixir for local files, Pythonx for HuggingFace Hub.

```elixir
# lib/tinkex_cookbook/datasets.ex
defmodule TinkexCookbook.Datasets do
  @moduledoc """
  Dataset loading with automatic source detection.
  """

  # Native Elixir for local JSONL
  def load("file://" <> path), do: load_jsonl_native(path)
  def load(path) when is_binary(path) and path =~ ~r/\.jsonl$/, do: load_jsonl_native(path)

  # Pythonx for HuggingFace datasets
  def load(repo_id) when is_binary(repo_id) do
    if String.contains?(repo_id, "/") do
      HuggingFace.load_dataset(repo_id)
    else
      raise ArgumentError, "Invalid dataset identifier: #{repo_id}"
    end
  end

  # Native Elixir JSONL loader
  defp load_jsonl_native(path) do
    path
    |> File.stream!()
    |> Stream.map(&Jason.decode!/1)
    |> Enum.to_list()
    |> Explorer.DataFrame.new()
  end
end

# lib/tinkex_cookbook/datasets/huggingface.ex
defmodule TinkexCookbook.Datasets.HuggingFace do
  use Pythonx

  defpymodule HFLoader do
    "import datasets; ..."
  end

  def load_dataset(repo_id, split \\ "train") do
    HFLoader.load_hf_dataset(repo_id, split)
    |> convert_to_dataframe()
  end
end

# lib/tinkex_cookbook/datasets/operations.ex
defmodule TinkexCookbook.Datasets.Operations do
  @doc "Shuffle dataset with optional seed"
  def shuffle(df, opts \\ []) do
    seed = Keyword.get(opts, :seed, :rand.uniform(1_000_000))
    Explorer.DataFrame.shuffle(df, seed: seed)
  end

  @doc "Split dataset into train/test"
  def train_test_split(df, test_size) when is_integer(test_size) do
    test = Explorer.DataFrame.slice(df, 0, test_size)
    train = Explorer.DataFrame.slice(df, test_size, Explorer.DataFrame.n_rows(df) - test_size)
    {train, test}
  end

  def train_test_split(df, test_fraction) when is_float(test_fraction) do
    n_rows = Explorer.DataFrame.n_rows(df)
    test_size = round(n_rows * test_fraction)
    train_test_split(df, test_size)
  end
end
```

**Pros**:
- ✅ Best of both worlds
- ✅ Native performance for local files
- ✅ HuggingFace Hub compatibility
- ✅ Type-safe Elixir operations

**Cons**:
- ⚠️ Still requires Python for HF datasets
- ⚠️ Need conversion layer (Pythonx → Explorer)

### 5.3 Strategy C: Pure Elixir (Most Ambitious)

**Approach**: Reimplement HuggingFace Hub client in Elixir.

**What's needed**:
1. HTTP client to HuggingFace Hub API
2. Parquet reader (use Explorer)
3. Dataset metadata parser (dataset card YAML)
4. Authentication (HF API tokens)
5. Caching layer (mimic `~/.cache/huggingface/`)

**Pros**:
- ✅ No Python dependency
- ✅ Full type safety
- ✅ Native Elixir performance

**Cons**:
- ❌ Massive reimplementation effort (1000+ LOC)
- ❌ Maintenance burden (HF API changes)
- ❌ Missing features (streaming, complex datasets)
- ❌ Not worth it for 15 datasets

---

## Part 6: Blobfile Migration (Simple)

### 6.1 Native Elixir Replacement

```elixir
# lib/tinkex_cookbook/storage.ex
defmodule TinkexCookbook.Storage do
  @moduledoc """
  Unified file I/O for local and cloud storage.
  Replaces Python's blobfile library.
  """

  @doc "Read JSONL file line-by-line"
  def read_jsonl("s3://" <> _ = path), do: read_jsonl_s3(path)
  def read_jsonl("gs://" <> _ = path), do: read_jsonl_gcs(path)
  def read_jsonl(path), do: read_jsonl_local(path)

  # Local file
  defp read_jsonl_local(path) do
    path
    |> File.stream!()
    |> Stream.map(&String.trim/1)
    |> Stream.reject(&(&1 == ""))
    |> Stream.map(&Jason.decode!/1)
    |> Enum.to_list()
  end

  # S3 file
  defp read_jsonl_s3("s3://" <> rest) do
    [bucket | path_parts] = String.split(rest, "/", parts: 2)
    key = Enum.join(path_parts, "/")

    case ExAws.S3.get_object(bucket, key) |> ExAws.request() do
      {:ok, %{body: body}} ->
        body
        |> String.split("\n")
        |> Enum.map(&String.trim/1)
        |> Enum.reject(&(&1 == ""))
        |> Enum.map(&Jason.decode!/1)

      {:error, reason} ->
        raise "Failed to read S3 file: #{inspect(reason)}"
    end
  end

  # GCS file (similar pattern)
  defp read_jsonl_gcs(_path) do
    raise "GCS not yet implemented"
  end
end
```

**Usage**:
```elixir
# Instead of:
# with blobfile.BlobFile(self.file_path, "r") as f:
#     for line in f:
#         data.append(json.loads(line))

# Use:
data = TinkexCookbook.Storage.read_jsonl("/path/to/file.jsonl")
# OR
data = TinkexCookbook.Storage.read_jsonl("s3://bucket/file.jsonl")
```

### 6.2 Alternative: ExAws.S3 Directly

If all cloud files are S3:

```elixir
defmodule TinkexCookbook.Storage.S3 do
  def download_to_temp(bucket, key) do
    temp_path = Path.join(System.tmp_dir!(), Path.basename(key))

    case ExAws.S3.download_file(bucket, key, temp_path) |> ExAws.request() do
      :ok -> {:ok, temp_path}
      error -> error
    end
  end

  def read_jsonl_from_s3(bucket, key) do
    {:ok, temp_path} = download_to_temp(bucket, key)

    data =
      temp_path
      |> File.stream!()
      |> Stream.map(&Jason.decode!/1)
      |> Enum.to_list()

    File.rm!(temp_path)
    data
  end
end
```

---

## Part 7: Concrete Recommendations

### 7.1 For HuggingFace Datasets

**Use Pythonx wrapper** because:
1. Cookbook uses **15+ distinct HF datasets** from Hub
2. Reimplementing HF Hub client is not cost-effective
3. Datasets library handles:
   - Authentication
   - Multi-split loading
   - Streaming large datasets
   - Caching
   - Metadata management

**Implementation path**:
```elixir
# lib/tinkex_cookbook/datasets/huggingface.ex
defmodule TinkexCookbook.Datasets.HuggingFace do
  use Pythonx

  defpymodule HFDatasets do
    """
    import datasets

    class DatasetWrapper:
        def __init__(self, ds):
            self.ds = ds

        def to_list(self):
            return list(self.ds)

        def shuffle(self, seed=None):
            return DatasetWrapper(self.ds.shuffle(seed=seed))

        def take(self, n):
            return DatasetWrapper(self.ds.take(n))

        def skip(self, n):
            return DatasetWrapper(self.ds.skip(n))

    def load_dataset(name, split=None):
        ds = datasets.load_dataset(name, split=split)
        if isinstance(ds, dict):
            return {k: DatasetWrapper(v) for k, v in ds.items()}
        return DatasetWrapper(ds)
    """
  end

  def load(repo_id, opts \\ []) do
    split = Keyword.get(opts, :split)
    HFDatasets.load_dataset(repo_id, split)
  end

  def to_dataframe(dataset_wrapper) do
    dataset_wrapper
    |> Pythonx.call(:to_list, [])
    |> Enum.map(&pythonx_dict_to_map/1)
    |> Explorer.DataFrame.new()
  end
end
```

**Usage**:
```elixir
# Load dataset
dataset = TinkexCookbook.Datasets.HuggingFace.load("allenai/tulu-3-sft-mixture", split: "train")

# Convert to DataFrame for Elixir processing
df = TinkexCookbook.Datasets.HuggingFace.to_dataframe(dataset)

# Now use Explorer operations
df = Explorer.DataFrame.shuffle(df, seed: 0)
{train_df, test_df} = TinkexCookbook.Datasets.Operations.train_test_split(df, 1024)
```

### 7.2 For Blobfile

**Replace with native Elixir** because:
1. Only **3 usage instances** (all simple JSONL reads)
2. No complex features used
3. Easy to implement with File + ExAws.S3

**Implementation**:
```elixir
# lib/tinkex_cookbook/storage.ex
defmodule TinkexCookbook.Storage do
  def read_jsonl(path) when is_binary(path) do
    cond do
      String.starts_with?(path, "s3://") -> read_jsonl_s3(path)
      String.starts_with?(path, "gs://") -> read_jsonl_gcs(path)
      true -> read_jsonl_local(path)
    end
  end

  defp read_jsonl_local(path) do
    File.stream!(path)
    |> Stream.map(&String.trim/1)
    |> Stream.reject(&(&1 == ""))
    |> Stream.map(&Jason.decode!/1)
    |> Enum.to_list()
  end

  defp read_jsonl_s3(uri) do
    %{host: bucket, path: "/" <> key} = URI.parse(uri)

    ExAws.S3.get_object(bucket, key)
    |> ExAws.request!()
    |> Map.fetch!(:body)
    |> String.split("\n")
    |> Enum.map(&String.trim/1)
    |> Enum.reject(&(&1 == ""))
    |> Enum.map(&Jason.decode!/1)
  end
end
```

### 7.3 For Dataset Operations

**Use Explorer directly** because:
1. All operations have native Explorer equivalents
2. Better performance (native Rust via Polars)
3. Type safety
4. Composability

**Mapping table**:

| Python (datasets) | Elixir (Explorer) | Notes |
|-------------------|-------------------|-------|
| `ds.shuffle(seed=0)` | `DataFrame.shuffle(df, seed: 0)` | ✅ Direct replacement |
| `ds.take(N)` | `DataFrame.slice(df, 0, N)` | ✅ |
| `ds.skip(N)` | `DataFrame.slice(df, N, DataFrame.n_rows(df) - N)` | ✅ |
| `ds.select(range(i, j))` | `DataFrame.slice(df, i, j - i)` | ✅ |
| `ds.filter(fn)` | `DataFrame.filter(df, condition)` | ✅ |
| `concatenate_datasets([...])` | `DataFrame.concat_rows([...])` | ✅ |
| `Dataset.from_list(data)` | `DataFrame.new(data)` | ✅ |
| `Dataset.from_pandas(df)` | N/A (no Pandas) | ⚠️ Convert via JSON |

---

## Part 8: Implementation Checklist

### Phase 1: Blobfile Replacement (Week 1)

- [ ] Create `TinkexCookbook.Storage` module
- [ ] Implement `read_jsonl/1` for local files
- [ ] Implement `read_jsonl/1` for S3 URIs
- [ ] Add ExAws.S3 dependency to `mix.exs`
- [ ] Write tests for local and S3 JSONL reading
- [ ] Document S3 credential setup (AWS_ACCESS_KEY_ID, etc.)

### Phase 2: Dataset Operations (Week 2)

- [ ] Create `TinkexCookbook.Datasets.Operations` module
- [ ] Implement `shuffle/2`
- [ ] Implement `train_test_split/2`
- [ ] Implement `batch_iterator/2`
- [ ] Write tests for all operations
- [ ] Benchmark vs. Python datasets library

### Phase 3: HuggingFace Integration (Week 3)

- [ ] Create `TinkexCookbook.Datasets.HuggingFace` module
- [ ] Set up Pythonx wrapper for `datasets.load_dataset`
- [ ] Implement conversion from Pythonx dataset to Explorer DataFrame
- [ ] Handle nested structures (messages lists)
- [ ] Test with 3-5 common datasets (Tulu3, MATH-500, etc.)
- [ ] Document HuggingFace Hub authentication

### Phase 4: Migration & Testing (Week 4)

- [ ] Port `FromConversationFileBuilder` to use `Storage.read_jsonl`
- [ ] Port `ComparisonBuilderFromJsonl` to use `Storage.read_jsonl`
- [ ] Port at least one HF dataset loader (e.g., Tulu3)
- [ ] End-to-end test: Load dataset → Process → Send to Tinker API
- [ ] Performance comparison (Python vs. Elixir)
- [ ] Update documentation

---

## Part 9: Open Questions

1. **How to handle nested data structures?**
   - Example: `{"messages": [{"role": "user", "content": "..."}]}`
   - Explorer supports nested types, but need to test JSON → DataFrame conversion

2. **Streaming for large datasets?**
   - Python `datasets` supports streaming from HF Hub
   - Elixir equivalent: Stream from HTTP → Parse JSONL → Process incrementally
   - May need custom streaming implementation

3. **Caching strategy?**
   - HuggingFace datasets cache to `~/.cache/huggingface/datasets/`
   - Should we cache in Elixir? Where?
   - Use Parquet for cached datasets?

4. **Type conversions?**
   - Pythonx dict → Elixir map → Explorer DataFrame
   - Need robust conversion for nested structures
   - Error handling for type mismatches

---

## Part 10: Migration Complexity Estimate

### Low Complexity (Blobfile)
- **Effort**: 1-2 days
- **Lines of code**: ~100
- **Risk**: Low (simple file I/O)

### Medium Complexity (Dataset Operations)
- **Effort**: 3-5 days
- **Lines of code**: ~200-300
- **Risk**: Medium (need to handle edge cases)

### High Complexity (HuggingFace Integration)
- **Effort**: 1-2 weeks
- **Lines of code**: ~500-800
- **Risk**: Medium-High (Pythonx bridge, type conversions, testing)

**Total Estimate**: 3-4 weeks for full migration

---

## Conclusion

**Recommended Architecture**:

```
┌─────────────────────────────────────────────────────────────┐
│                    Tinkex Cookbook (Elixir)                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Dataset Loading:                                           │
│  ├─ Local JSONL ────→ TinkexCookbook.Storage (native)     │
│  ├─ S3/GCS ─────────→ ExAws.S3 (native)                   │
│  └─ HuggingFace ────→ Pythonx wrapper (unavoidable)       │
│                                                             │
│  Processing:                                                │
│  ├─ Shuffle/Split ──→ Explorer.DataFrame (native)         │
│  ├─ Filter ─────────→ Explorer.DataFrame (native)         │
│  └─ Batch ──────────→ Explorer.DataFrame (native)         │
│                                                             │
│  Tokenization & Training:                                   │
│  └─ (Pure Elixir, separate analysis)                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key Takeaways**:

1. **Blobfile**: Replace entirely with native Elixir (File + ExAws.S3)
2. **Datasets (HuggingFace Hub)**: Use Pythonx wrapper (no viable pure-Elixir alternative)
3. **Dataset operations**: Use Explorer (shuffle, split, filter, batch)
4. **Local JSONL processing**: Pure Elixir (File.stream! + Jason)
5. **Performance**: Native Elixir for I/O, Explorer for data ops, Pythonx only for HF Hub

This corrects the preliminary analysis in `09_datasets_blobfile_mapping.md` with actual usage data from the codebase.
