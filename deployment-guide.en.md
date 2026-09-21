# Medical Knowledge Graph Diagnostic Engine — Cross-Platform Deployment Guide

**Version**: v2.0
**Last updated**: 2025-10-07
**Target environments**: Windows x86 + Blackwell | ARM + Blackwell (DGX Spark)

> **English edition.** The Traditional Chinese original is
> [`deployment-guide.md`](deployment-guide.md), and it is the one `deploy.cmd`
> points operators at. The two are kept in step; where they disagree, the code
> settles it.

---

> **⚠️ Phase C update notice (2026-05)**
>
> Deployment has moved to [uv](https://docs.astral.sh/uv/) as the core package
> manager. **Use `deploy.sh` (Linux) / `deploy.cmd` (Windows) as the primary
> deployment entry point.**
>
> - PyTorch (torch/torchvision/torchaudio) is now managed by uv through
>   `[tool.uv.sources]` in `pyproject.toml` and pinned in `uv.lock`, so versions
>   are identical across platforms.
> - The PyG native extensions (pyg-lib, torch-scatter/sparse/cluster) are still
>   installed by the deploy scripts post-sync, with a graceful skip when no wheel
>   exists and a fallback to `torch.scatter_reduce`.
> - The `pip install` / `python -m venv` examples embedded below (for instance
>   the "DGX Spark setup script" section) are original v2.0 content and are
>   **kept for understanding the mechanics only**. Run `deploy.{sh,cmd}` for an
>   actual deployment.
>
> See also: the `[tool.uv]` section of `pyproject.toml`, `uv.lock`, and the
> "Currently Supported Data Sources" table in `data/external/README.md`.

---

## What this guide is for 🎯

1. **Step-by-step deployment for both environments**
2. **Per-package compatibility analysis and fallbacks**
3. **A complete verification procedure**
4. **A troubleshooting handbook**

---

# Part One: Environment Comparison

## Hardware comparison

| Item | Windows development | DGX Spark edge | Impact |
|------|---------------------|----------------|--------|
| **Processor** |
| Architecture | x86-64 (Intel/AMD) | ARM v9.2 (Cortex-X925 + A725) | ⚠️ Different ISA; some packages need rebuilding |
| Cores | Varies (assume 8-16) | 20 (10+10 hybrid) | ✅ ARM advantage: more cores |
| Clock | ~3-5 GHz | ~3.0 GHz (X925) | ≈ Comparable |
| **GPU** |
| Model | NVIDIA Blackwell (discrete) | Blackwell (GB10 SoC integrated) | ⚠️ Integrated vs discrete changes the API surface |
| CUDA cores | Varies | 6144 | ✅ Sufficient |
| Tensor cores | 5th Gen | 5th Gen | ✅ Same |
| VRAM | 16GB GDDR6 | — | ⚠️ **Windows constraint** |
| **Memory** |
| System RAM | 32GB+ | 128GB LPDDR5X-9400 | ✅ **Large ARM advantage** |
| Architecture | Split (CPU/GPU) | **Unified memory** | ✅ ARM advantage: zero-copy |
| Bandwidth | RAM ~50 GB/s, VRAM ~448 GB/s | **Unified ~301 GB/s** | ≈ Trade-off |
| **Interconnect** |
| CPU-GPU | PCIe Gen4/5 (~32 GB/s) | **NVLink-C2C (~600 GB/s)** | ✅ **Substantial ARM advantage** |
| **Power** |
| TDP | 200-400W (CPU+GPU) | **140W (whole system)** | ✅ ARM efficiency advantage |

### Key observations

#### ✅ **Where ARM (DGX Spark) wins**
1. **128GB unified memory**: larger knowledge graphs load without subgraph sampling
2. **Very high CPU-GPU bandwidth**: NVLink-C2C is ~18× PCIe, zero-copy transfers
3. **More CPU cores**: 20 vs the usual 8-16
4. **Low power**: 140W vs 200-400W

#### ⚠️ **Where ARM (DGX Spark) is harder**
1. **Package ecosystem**: some Python packages ship no ARM wheel
2. **No VRAM isolation**: no discrete VRAM; GPU and CPU share the 128GB
3. **Driver maturity**: ARM + Blackwell is a relatively new combination

#### ⚠️ **Windows x86 limits**
1. **VRAM capacity**: consumer GPU VRAM (typically 8-24 GB) is smaller than DGX
   Spark's 96 GB unified memory. A larger KG (full PrimeKG at 130K nodes / 4M
   edges, say) needs subgraph sampling or batched training; GNN inference still
   runs.
2. **Copy overhead**: CPU ↔ GPU transfers go over PCIe, slower than DGX Spark's
   NVLink-C2C.

---

# Part Two: Package Compatibility

## 2.1 Core dependency matrix

| Package | Version | Win x86 | ARM | Difficulty | Notes |
|---------|---------|---------|-----|------------|-------|
| **Python** | 3.12 | ✅ | ✅ | Easy | Both supported |
| **PyTorch** | 2.10.0 | ✅ | ✅ | Easy | Official ARM+CUDA wheel |
| **CUDA Toolkit** | 13.0 | ✅ | ✅ | Easy | Required by Blackwell |
| **cuDNN** | 9.x | ✅ | ✅ | Easy | Bundled with PyTorch |
| **PyTorch Geometric** | 2.6+ | ✅ | ⚠️ | Moderate | ARM may need a source build |
| **pyg-lib** | latest | ✅ | ⚠️ | Moderate | As above |
| **torch-scatter** | latest | ✅ | ⚠️ | Moderate | As above |
| **torch-sparse** | latest | ✅ | ⚠️ | Moderate | As above |
| **FlashAttention-2** | 2.8+ | ✅ | ❌ | Hard | **No ARM support** |
| **xformers** | 0.0.27+ | ✅ | ⚠️ | Hard | ARM needs a source build |
| **FAISS (GPU)** | 1.8+ | ✅ | ❌ | Moderate | **Deprecated — use cuVS** |
| **hnswlib** | 0.8+ | ✅ | ✅ | Easy | **Deprecated — use Voyager** |
| **Voyager** | 2.0+ | ✅ | ✅ | Easy | Spotify HNSW, cross-platform |
| **cuVS** | 24.12+ | ✅ | ✅ | Moderate | NVIDIA RAPIDS, Linux GPU |
| **transformers** | 4.40+ | ✅ | ✅ | Easy | Fully supported |
| **owlready2** | 0.46+ | ✅ | ✅ | Easy | Pure Python |
| **neo4j** | 5.0+ | ✅ | ✅ | Easy | Pure Python driver |
| **fastapi** | 0.110+ | ✅ | ✅ | Easy | Pure Python |
| **pandas/numpy** | latest | ✅ | ✅ | Easy | Fully supported |

### Legend
- ✅ **Green**: prebuilt wheel, plain `pip install`
- ⚠️ **Yellow**: may need a source build, but workable
- ❌ **Red**: unsupported; needs a fallback

---

## 2.2 Key packages in detail

### 🔴 **PyTorch Geometric on ARM**

**Where things stand**:
- PyG 2.6+ supports ARM in principle, but prebuilt wheels may be missing
- Its extension libraries (pyg-lib, torch-scatter, …) have uneven ARM support

**Installation strategies, in order of preference**:

```bash
# Strategy 1: official wheel (success rate ~60%)
pip install torch-geometric==2.6.0

# Strategy 2: the PyG wheel index (success rate ~80%)
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.10.0+cu130.html

# Strategy 3: build from source (success rate ~95%, takes 1-2h)
git clone https://github.com/pyg-team/pytorch_geometric.git
cd pytorch_geometric
pip install -e .

# Strategy 4: conda (success rate ~70%, not recommended on ARM)
conda install pyg -c pyg

# Strategy 5: install the extensions by hand (most reliable, 2-3h)
pip install torch-geometric

git clone https://github.com/pyg-team/pyg-lib.git
cd pyg-lib && pip install -e . && cd ..

git clone https://github.com/rusty1s/pytorch_scatter.git
cd pytorch_scatter && pip install -e . && cd ..
# and so on
```

**Verification script**:
```python
# test_pyg_arm.py
import torch
import torch_geometric as pyg
from torch_geometric.data import Data

print(f"PyG version: {pyg.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
x = torch.randn(3, 16)
data = Data(x=x, edge_index=edge_index)

if torch.cuda.is_available():
    data = data.cuda()
    print("✅ PyG on ARM + CUDA: OK")
else:
    print("❌ CUDA not available")
```

**Fallback: DGL**
```bash
# If PyG cannot be installed at all
pip install dgl -f https://data.dgl.ai/wheels/torch-2.10/cu130/repo.html
# DGL's ARM support is better
```

### 🔴 **FlashAttention-2 on ARM**

**Where things stand**:
- ❌ **ARM is not supported**
- The upstream repository states x86-64 only
- Building from source on ARM fails or hangs

**Solution: a three-tier downgrade**

The implementation lives in `src/models/attention/adaptive_backend.py`. It
detects the machine and picks, in order: FlashAttention-2 (x86 only), xformers
memory-efficient attention (the ARM alternative), PyTorch SDPA (the final
fallback), and a manual implementation that should never be reached. All four
present one `compute(query, key, value, attn_mask, dropout_p)` interface, so
callers do not branch on the platform; the backends differ only in the tensor
layout each wants, which the wrapper handles.

**Expected performance (relative to FlashAttention-2 on x86)**:
- FlashAttention-2 (x86): **1.00×** (baseline)
- xformers (ARM): **0.65-0.75×**
- PyTorch SDPA (ARM): **0.45-0.55×**
- Manual (ARM): **0.25-0.35×** (not recommended)

### 🟢 **Vector index: Voyager + cuVS (v3.2)** — detached from diagnosis

> **Note**: FAISS and hnswlib are deprecated. The current architecture uses
> Voyager (cross-platform) + cuVS (Linux GPU).
>
> **Status**: this subsystem is **not on the diagnosis path**. It was detached
> from the inference pipeline by decision; the implementation and its tests are
> retained for the planned natural-language input / vector mapping work. No index
> needs to be built at deployment time, and building one does not change any
> diagnosis result. Background: `docs/RETRIEVAL_AND_CANDIDATE_DISCOVERY_FINDINGS.md`.

**Backend selection**:
- Linux (x86/ARM): cuVS (GPU) → Voyager (CPU fallback)
- Windows: Voyager only (cuVS does not support Windows)

```python
# src/retrieval/vector_index.py (v3.2)
from src.retrieval import create_index, resolve_backend

index = create_index(backend="auto", dim=768, metric="ip")

voyager_index = create_index(backend="voyager", dim=768)
cuvs_index = create_index(backend="cuvs", dim=768)  # Linux only

embeddings = {"entity_1": vec1, "entity_2": vec2}
index.build_index(embeddings)
results = index.search(query_vector, top_k=10)
# Returns: [("entity_id", score), ...]
```

**Installation**:
```bash
# Voyager (required, cross-platform)
pip install voyager>=2.0

# cuVS (optional, Linux GPU; use the cu13 build to match the project's torch cu130)
pip install --extra-index-url https://pypi.nvidia.com cuvs-cu13
```

**Comparison**:
| Operation | cuVS GPU (Linux) | Voyager CPU | Notes |
|-----------|------------------|-------------|-------|
| Build index (1M vectors) | 3s | 30s | cuVS uses IVF-PQ |
| Query (batch=100, k=10) | 1ms | 8ms | Voyager uses HNSW |
| Memory | Low (GPU VRAM) | Moderate (CPU RAM) | — |
| Platforms | Linux only | Cross-platform | Windows is Voyager only |

---

# Part Three: Deployment Scripts

> Both scripts below are original v2.0 content, kept to show the mechanics. For
> an actual deployment run `deploy.sh` / `deploy.cmd`, which drive uv and the
> lock file.

## 3.1 Windows x86 + Blackwell

The `setup_windows.ps1` flow, in order: check CUDA (expects 12.8 for Blackwell),
check Python 3.12, create and activate `.venv`, upgrade pip/setuptools/wheel,
install PyTorch 2.10.0 + CUDA 13.0 from the PyTorch index, verify CUDA is
actually available, install PyTorch Geometric with its extensions from the PyG
wheel index, install FlashAttention-2 with `--no-build-isolation`, then the
remaining pure-Python packages. It finishes by printing the resolved versions and
the CUDA device.

```powershell
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 `
    --index-url https://download.pytorch.org/whl/cu130

pip install torch-geometric pyg-lib torch-scatter torch-sparse torch-cluster `
    -f https://data.pyg.org/whl/torch-2.10.0+cu130.html

pip install flash-attn --no-build-isolation
```

### Manual steps if the script fails

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu130

# PyG option A: the wheel index
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.10.0+cu130.html
# PyG option B: from source
git clone https://github.com/pyg-team/pytorch_geometric.git
cd pytorch_geometric && pip install -e .

# FlashAttention-2 — if this fails, check for Visual Studio Build Tools
pip install flash-attn --no-build-isolation

pip install -r requirements.txt
```

---

## 3.2 DGX Spark (ARM + Blackwell)

The `setup_dgx_spark.sh` flow checks the architecture is `aarch64`, checks CUDA
and Python, creates the venv, installs the ARM PyTorch build, installs PyG
(wheel first, source build as fallback), skips FlashAttention-2 deliberately and
falls through to xformers or SDPA, installs the remaining packages, and attempts
cuVS with a graceful skip:

```bash
pip install --extra-index-url https://pypi.nvidia.com cuvs-cu13 || \
    echo "cuVS install failed; Voyager (CPU) will be used"
```

Platform configuration is centralised in `configs/deployment.yaml`. Hardware
detection (GPU / arch / CUDA) happens automatically at startup in `deploy.sh`
and `scripts/launch/shep_launch.py` — there is no platform config file to write
by hand.

**After setup**:
1. Build the knowledge graph: `python scripts/build_knowledge_graph.py`
2. Precompute shortest paths: `python scripts/compute_shortest_paths.py`
3. Train: `python scripts/train_model.py` (optionally `--config <hyperparameters.yaml>`)
4. Launch: `./launch_shepherd.sh`

**Things to know**:
- FlashAttention-2 is unavailable on ARM; the downgrade is automatic
- The vector index is detached from the diagnosis pipeline and does not affect
  results; it is retained for future vector mapping work
- The configuration is tuned for 128GB of unified memory

### Manual troubleshooting steps

```bash
uname -m           # expect aarch64
nvidia-smi         # expect a GPU
python3 --version  # expect 3.12.x

python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"

python3 -c "
from src.models.attention.adaptive_backend import get_attention_backend
print('Using:', get_attention_backend().backend)"
```

---

# Part Four: Troubleshooting

## 4.1 Common Windows problems

### Problem 1: FlashAttention-2 fails to install

```
error: Microsoft Visual C++ 14.0 or greater is required
```

1. Install Visual Studio 2022 Build Tools — <https://visualstudio.microsoft.com/downloads/>,
   selecting "Desktop development with C++"
2. Or use a prebuilt wheel if one exists
3. Last resort: skip FlashAttention and let the backend fall through to PyTorch SDPA

### Problem 2: CUDA out of memory (the 16GB VRAM limit)

```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

```python
# 1. Smaller batches
batch_size = 8  # down from 32

# 2. Gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 3. Subgraph sampling
subgraph = sample_subgraph(full_graph, num_nodes=50000)

# 4. Gradient checkpointing
from torch.utils.checkpoint import checkpoint
output = checkpoint(model_layer, input)
```

### Problem 3: PyG version mismatch

```
undefined symbol: _ZN...
```

```bash
pip uninstall torch torch-geometric pyg-lib torch-scatter torch-sparse -y
pip cache purge

pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu130
pip install torch-geometric pyg-lib torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.10.0+cu130.html
```

---

## 4.2 Common ARM (DGX Spark) problems

### Problem 1: PyG fails to build on ARM

```
error: command 'gcc' failed with exit status 1
```

```bash
sudo apt-get update
sudo apt-get install -y build-essential python3-dev

export MAX_JOBS=10                  # use 10 cores
export TORCH_CUDA_ARCH_LIST="9.0"   # Blackwell

git clone https://github.com/rusty1s/pytorch_scatter.git
cd pytorch_scatter && python setup.py install
```

### Problem 2: unified memory not used correctly

Check that the reported device memory matches the machine's unified pool rather
than a discrete VRAM figure, and that transfers are not being staged through a
host copy.

### Problem 3: attention is unexpectedly slow

Confirm which backend was selected. On ARM, SDPA is roughly half the speed of
FlashAttention-2 on x86, and the manual fallback is far slower again — if
`get_attention_backend().backend` reports `manual_fallback`, neither xformers nor
SDPA was found and that is the problem to fix.

---

## 4.3 The shortest-path table and its hop bound

A workspace's `shortest_paths.pt` is a precomputed distance table, and
`shortest_paths.meta.json` is its **sidecar**, recording the parameters the table
was generated with — `max_hops` above all. **The two belong to one build and must
be copied or moved together.**

### Why the sidecar cannot be skipped

`max_hops` sets the cost assigned to "unreachable" (`max_hops + 1`). Every
shortest-path score is measured against that value, and it **cannot be recovered
from the table itself**: the BFS stops expanding at `max_hops`, so the largest
distance recorded is only a **lower bound** — a genuine 3-hop table and a table
built to 5 in which no pair happens to reach 5 are identical at the tensor level.

Scoring against the wrong ceiling does not merely shift numbers. **It changes the
ranking.** One 3-hop table, two candidates:

| max_hops used | A | B | Order |
|---|---:|---:|---|
| the correct 3 | 0.4357142857 | 0.4250000000 | A > B |
| 5 by mistake | 0.4166666667 | 0.4250000000 | **B > A** |

So the system **does not guess for you**.

### Order of recovery

1. **Recover the original sidecar first**, then keep and move the two files
   together. When a sidecar is present its value wins and
   `SHEPHERD_SP_HOP_BOUND` is ignored.
2. **No sidecar, but the original setting is known** (from a build log, say) —
   only then set `SHEPHERD_SP_HOP_BOUND=5`. **Do not set 5 everywhere by default,
   and do not infer it from the largest distance in the table** — that number
   cannot establish the ceiling.
3. **Original setting unknown: regenerate** (`scripts/compute_shortest_paths.py`).
   If you run on pure GNN in the meantime, know that the scoring mode has changed.
4. Set the variable, **restart the backend**, then check the status below.

### The environment variable

```bash
# Only when the original build's max_hops is known
export SHEPHERD_SP_HOP_BOUND=5
```

- Valid values are **integers in 1..127** (the same range the generator validates)
  and must **not be below the largest distance observed in the table** — below it
  the refusal fires, because the unreachable cost would land beneath real distances.
- A value that cannot be parsed as an integer is **refused, not ignored**, so
  "I thought I set it" cannot happen silently.
- This is a **process-level** setting: it must be present in the environment that
  starts the backend, and the backend must be restarted to pick it up.
  **Exporting it in another shell does not change a running process.**
- When loading a different workspace (via `/api/v1/pipeline/reload`), confirm the
  bound still applies to the new table.

### What happens with no sidecar and no setting

Under the default configuration (GNN loaded, `sp_optional=True`) the **service
does not stop**. It serves pure GNN scoring: `sp_ready=False`,
`scoring_mode=gnn_only`, effective eta 1.0. **Scores and rankings differ from a
run with shortest paths enabled** — a deliberate trade: better one signal fewer
than scoring against a ceiling nobody chose.

> **Exception**: a deployment that has deliberately set `sp_optional=False` will,
> with SP unavailable, have **GNN scoring disabled** by the current code. This is
> not "always falls back to pure GNN". The API build path uses the default
> `sp_optional=True`.

### A corrupt or invalid sidecar

When the sidecar is **present but invalid** (not a JSON object, `max_hops` not an
integer or outside 1..127, or below the distances in the table), the candidate
pipeline is **refused** rather than falling back to a guess:

- The service has **no** usable pipeline yet (cold start) → `/api/v1/diagnose`
  returns **503** and no candidate diseases.
- The service **has** a pipeline running → that reload is refused and the **old
  pipeline keeps serving**.

### Checking the result

```bash
curl -s http://localhost:8000/api/v1/pipeline/status
```

| Field | Expected |
|---|---|
| `sp_ready` | `true` when enabled; `false` when the bound is missing |
| `sp_max_hops` | the ceiling actually in effect |
| `sp_hop_bound_source` | `sidecar` (read from the file) or `configured` (from the variable) |
| `scoring_mode` | `gnn_plus_shortest_path` when enabled; `gnn_only` when SP is off and the GNN is fine |

That is what `sp_hop_bound_source` exists for: **"read" and "stated" are
different strengths of evidence**, and the status endpoint should be able to tell
them apart rather than sending an operator to the logs.

---


### When the table and its graph do not match

A shortest-path table is published as **a pair** — `shortest_paths.pt` and
`shortest_paths.meta.json` — and both files record the same `build_id`, plus the
SHA-256 of the graph the distances were computed from. Three refusals follow
from that, and all three are deployment problems rather than data problems:

| What the log says | What happened | What to do |
|---|---|---|
| *"published by different runs"* | The two files are from different builds. A publication was interrupted between them, or one was copied over the other | Re-run `scripts/compute_shortest_paths.py` for this workspace. Do **not** copy one file from elsewhere to make them match |
| *"computed from a different graph"* | The table belongs to another workspace's `kg.json`. Its node indices point at different nodes | Use the table built from *this* workspace, or rebuild it here |
| *"One side of a pair is not a pair"* | One file declares the protocol and the other does not — a table published without its sidecar, or the reverse | Re-run the generator; the pair is published together or not at all |

**A table built before this existed still loads.** It carries no binding, so the
service reports its provenance as `unrecorded` rather than verified, logs one
warning, and serves. Rebuilding records one.

`GET /api/v1/pipeline/status` reports `sp_kg_binding`: `verified` when the table
names the graph being served, `unrecorded` for a table older than the protocol.
**It is never absent while shortest paths are on** — if it says `unrecorded`, no
check confirmed where that table came from.

**Building from a graph held in memory rather than from a workspace** cannot
check a binding at all, and a table that declares one is refused rather than
assumed. Build from a workspace, or use a table published without a binding.

## 4.4 The build provenance record (`kg.provenance.json`)

Every build writes `kg.provenance.json` beside `kg.json`, recording **which
files** the graph was built from.

### Why it exists

`kg.json` has always carried a digest, but a digest only says *which bytes*. Two
machines whose ontologies differ by a deployment date produce different
`kg_digest` values — **the divergence is detected and cannot be explained**.

### What it holds

| Field | Meaning |
|---|---|
| `kg_digest` | SHA-256 of the `kg.json` written in the same build. **This is what binds the record to its graph** |
| `origin` | `files` for a real build, `synthetic` for an in-memory demo or test graph |
| `sources` | Per input: role, filename, content digest, declared version |
| `missing_roles` | Inputs that could not be identified |
| `counters` | Parsing statistics, e.g. `rows_skipped_unresolved_disease_id` |
| `incomplete_by_design` | **What this record deliberately does not cover** |

The four roles are `mondo`, `hpo`, `phenotype_hpoa`, `genes_to_phenotype`.
**The filename is a hint; the digest is the identity** — a file under the same
directory name can be replaced in place.

`declared_version` is the raw `data-version` from an ontology header, and
**absent means `null`**. It is never filled in from the OBO format version.

Read `counters` literally: `rows_skipped_unresolved_disease_id` counts **rows of
`phenotype.hpoa` at one parsing stage**, including deliberately unsupported
identifier types such as DECIPHER. It is **not a count of lost diseases** and not
a total of version mismatches. The builder drops further edges elsewhere when a
node is absent, and those are not in this number.

### What it does not claim

`incomplete_by_design` is written into the file itself: ontology imports resolved
at parse time, parser and builder versions, and graph construction parameters are
all outside it. It is a **list of source files, not a rebuild recipe**.

### Three states

| State | Meaning | Treatment |
|---|---|---|
| No such file | Built before this existed | **unknown**. Never back-filled from whatever the cache holds now |
| Present and matching `kg.json` | This graph's sources | Trust it |
| **Present but missing, corrupt or mismatched** | A claim that does not hold | **Reported as that**, never folded into unknown |

The third is the one that matters: two graph-only workspaces whose provenance
files were swapped during a copy have **every file well-formed** and only the
pairing wrong. `workspace_provenance_status()` compares the record's `kg_digest`
against the `kg.json` present, which is what tells them apart.

**Verifying provenance does not gate service.** A source state describes where a
graph came from; whether the graph is usable is a different question, answered by
the digest bindings on `kg.json` and the three tensors — and those still refuse.
`workspace_provenance_status()` **does not raise**; it returns a state, and
deciding to stop anything because of one is a policy choice no code here makes.
That promise covers the disk as well as the content, and the existence checks
as well as the reads: a `kg.json`, a record or a `manifest.json` that this
process cannot open or decode — or a **workspace directory it cannot enter**,
which makes even `Path.is_file()` raise, since `pathlib` re-raises `EACCES`
rather than returning `False` — comes back as `unreadable` with the cause in the
detail, rather than as an exception out of a function every caller was written
not to wrap. Not being able to find out whether a record is there is reported as
that, never as the record not being there. Note the manifest case in
particular — an unreadable manifest means *whether a record was ever declared
cannot be established*, which is not the same as *nothing declared one*, and
only the second of those licenses the "this workspace predates provenance"
reading. Refusing a workspace over a broken manifest remains
`verify_graph_artifacts()`'s job.

> **Correction.** An earlier version of this section had
> `verify_graph_artifacts()` raise on a broken record. That function is called by
> `verify_graph_source()`, which `DiagnosisPipeline` runs before loading a
> tensor — so a lost note about where a graph came from stopped a model from
> initialising, and a cold start answered `/diagnose` with 503 while the tensors
> were sound. It reports now rather than refusing.

### Older workspaces

A schema-3 workspace built before Phase 1 **does not become invalid** for lacking
the file. When the manifest declares no provenance, the reader does not look for
it.

### Relationship to ontology versions

`latest` is a **cache default, not a freshness guarantee** — it means "use the
cached file; download only if absent", and does not check the remote for a newer
release on each call.

## 4.5 Choosing which ontology a build uses

**Name the file.** `--mondo-path` and `--hpo-path` take a path and win over
everything else. A path that does not exist **refuses** rather than quietly
opening something in the cache, so a build cannot succeed with a file you did
not ask for.

With no path, the build searches `paths.ontology_roots` in
`configs/deployment.yaml` and then `--ontology-cache-dir`:

| What is found | What happens |
|---|---|
| Exactly one candidate | It is used, and logged with its `data-version` |
| **More than one** | **Refused**, listing each one's path, `data-version` and digest |
| None, a source configured | Downloaded |
| None, no source | Refused, naming the roots searched |

**Nothing is ranked.** Not the order of the roots, not the file's modification
time, and not "the newest `data-version`". Keeping several releases side by side
is supported — that is what naming a path is for — but a build will not pick one
for you.

**This changes an existing behaviour.** A cache directory holding both
`mondo.obo` and `mondo.owl` used to take the `.obo` silently; it now refuses and
asks for `--mondo-path`. A cache ends up holding both when a download fell back
from OBO to OWL, so this is not a rare state. The fix is to add the path
argument; nothing has to be deleted.

**`--force-download` and a path together are refused** — naming a file and
demanding a fresh copy are two different instructions.

### Two refusals a build can now hit

| Refusal | What it means |
|---|---|
| "declares N import(s) and this project requires a self-contained ontology file" | The file names other ontologies to pull in. They are **not** fetched: resolving them would add content this file's digest does not cover, and ignoring them would build a graph missing what they carry while looking complete. Supply a file that declares no imports |
| "declares itself to be X, and it is being used as the Y input" / "carries no `HP:` terms" | The file is not the ontology that slot asked for. Left alone it builds **zero** nodes of that type, with a provenance record that looks complete |

### Where the download URLs live

`configs/deployment.yaml`, under `ontology.sources`. They used to be a constant
in `src/ontology/loader.py`, so a PURL that stopped resolving meant editing
code; now it is a configuration change. Leave the key unset to use the four OBO
Foundry PURLs the project ships with.

Two rules apply to **every request**, including each redirect — and every source
here is a PURL, which *is* a redirect:

- **Scheme:** `http` and `https` only.
- **Destination:** every address the host resolves to must be globally
  routable. A host inside your own network is refused unless it is listed under
  `ontology.allowed_hosts` — which is how you enable an in-house mirror, on
  purpose rather than by accident.

**Not yet delivered:** editing these URLs from the interface. The requirement is
recorded and the backend is built for it — one configuration model, one
download-and-verify entry point that a UI and the CLI would share — but the
interface itself is outside this programme's scope. Until it exists, editing the
file is the way.

**Changing a source does not change what is being served.** Editing a URL,
downloading a file and building a workspace are three separate things, and only
a new build produces a new workspace; putting one into service is the existing
load-and-publish step, unchanged.

---

## Summary: deployment checklist ✅

### Windows x86 + Blackwell

- [ ] ✅ Python 3.12 installed
- [ ] ✅ CUDA 13.0 installed
- [ ] ✅ PyTorch + CUDA working
- [ ] ✅ PyTorch Geometric working
- [ ] ✅ FlashAttention-2 installed
- [ ] ✅ Voyager available (CPU vector search)
- [ ] ✅ All unit tests pass
- [ ] ✅ Model trains (watch VRAM)

### ARM + Blackwell (DGX Spark)

- [ ] ✅ Architecture confirmed as aarch64
- [ ] ✅ CUDA 13.0 available
- [ ] ✅ PyTorch ARM build working
- [ ] ✅ PyTorch Geometric available (wheel or build)
- [ ] ⚠️ Attention backend downgraded (xformers or SDPA)
- [ ] ✅ Vector index: cuVS (GPU) or Voyager (CPU fallback)
- [ ] ✅ Unified memory correctly identified
- [ ] ✅ Model trains (check the 128GB advantage)

---

**Version**: v3.2
**Maintenance**: kept in step with `deployment-guide.md`; update both together.
