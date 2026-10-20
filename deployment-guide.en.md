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

**A table is served only with the sidecar it was published with.** Since contract
M2.5 (2026-10-13) the sidecar must carry the table's source binding
(`schema_version`, `build_id`, `kg_digest`), and its `max_hops` is the only source
of the hop bound. A table without a sidecar, or with one written before the
pairing protocol, is refused at load.

1. **Recover the sidecar published with the table**, then keep and move the two
   files together.
2. **Otherwise rebuild the pair for this workspace** with
   `scripts/compute_shortest_paths.py`, which publishes the table and its sidecar
   together, bound to this workspace's graph. **Do not write a sidecar by hand,
   and do not infer `max_hops` from the largest distance in the table** — that
   number cannot establish the ceiling.
3. Until then, either keep the deployment that is serving, or serve without
   shortest paths: move `shortest_paths.pt` **and** `shortest_paths.meta.json` out
   of the workspace together, keep them, then reload. A sidecar left behind alone
   is refused as one side of a pair. The scoring mode changes.

`SHEPHERD_SP_HOP_BOUND`, which used to supply the bound for a table without a
sidecar, is **retired** with M2.5: such a table is refused before any bound is
read, so the variable is no longer read. Remove it from the service environment.

### A missing, unbound or invalid sidecar

When a table is present and its sidecar is **missing, unbound, or invalid** (not a
JSON object, `max_hops` not an integer or outside 1..127, or below the distances
in the table), the candidate pipeline is **refused** rather than falling back to a
guess:

- The service has **no** usable pipeline yet (cold start) → `/api/v1/diagnose`
  returns **503** and no candidate diseases.
- The service **has** a pipeline running → that reload is refused and the **old
  pipeline keeps serving**.

### No table at all

With no `shortest_paths.pt` (and no sidecar) under the default configuration (GNN
loaded, `sp_optional=True`) the **service does not stop**. It serves pure GNN
scoring: `sp_ready=False`, `scoring_mode=gnn_only`, effective eta 1.0. **Scores and
rankings differ from a run with shortest paths enabled.** Every result says so in its
served identity (§4.7): shortest paths `absent`, scoring mode `gnn_only`, eta `1.0`, and
the Diagnosis tab and its report show a `NO SHORTEST-PATH TABLE: GNN score alone.` banner.

> **Exception**: a deployment that has deliberately set `sp_optional=False` gets
> **no pipeline** when SP is unavailable: the build is refused
> (`PipelineBuildError`), a startup publishes nothing and `/diagnose` answers 503,
> and a reload is refused with the previous pipeline still serving. This is not
> "always falls back to pure GNN". The API build path uses the default
> `sp_optional=True`.

### Checking the result

```bash
curl -s http://localhost:8000/api/v1/pipeline/status
```

| Field | Expected |
|---|---|
| `sp_ready` | `true` when enabled; `false` when no table is present |
| `sp_max_hops` | the ceiling in effect, read from the sidecar |
| `sp_hop_bound_source` | `sidecar` (the only source since the configured bound was retired) |
| `sp_kg_binding` | `verified` |
| `scoring_mode` | `gnn_plus_shortest_path` when enabled; `gnn_only` when SP is off and the GNN is fine |
| `served_identity` | What the loaded pipeline serves (§4.7). When enabled: `shortest_paths.state` `verified` with the sidecar's `build_id` and `max_hops`, and `scoring_mode` `gnn_plus_shortest_path`. With no table: `shortest_paths.state` `absent`, `gnn_only`, `settings.eta` `1.0` |

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

**A table built before this existed is refused** (since contract M2.5,
2026-10-13; it used to load with a warning and be reported `unrecorded`). It
carries no binding, so nothing says which graph its node indices belong to.
Rebuild it with `scripts/compute_shortest_paths.py`, or remove it to serve without
shortest paths.

`GET /api/v1/pipeline/status` reports `sp_kg_binding`: `verified` when the table
names the graph being served. **It is never absent while shortest paths are on.**

**Building from a graph held in memory rather than from a workspace** cannot
check a binding at all, and a table that declares one is refused rather than
assumed; a table without one is refused as above. Build from a workspace.

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
| `missing_roles` | For `files`, the roles its sources leave out — always none in a record the pipeline accepts; for `synthetic`, every role, by construction |
| `counters` | Parsing statistics, e.g. `rows_skipped_unresolved_disease_id` |
| `incomplete_by_design` | **What this record deliberately does not cover** |

The four roles are `mondo`, `hpo`, `phenotype_hpoa`, `genes_to_phenotype`, and
**a real build requires all four**: the builder stops without either annotation
file, always loads both ontologies, and stops rather than record an input it
parsed and cannot identify. **The filename is a hint; the digest is the
identity** — a file under the same directory name can be replaced in place.

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

### Required: training, serving and measurement refuse without it

**Since contract M3b the record is required.** The manifest binds its digest
(`artifacts.provenance` in `split_manifest.json`), and training, serving (with
a model over a workspace) and measurement each check it after their graph
check, reading the file once: its bytes must be the ones the manifest binds,
and its `kg_digest` must name the graph that check verified. Anything else
refuses, before a run directory exists, a checkpoint is read or a model is
built. A refusal names the **binding state**:

| Binding state | Meaning |
|---|---|
| `verified` | The record the manifest binds, naming this workspace's graph. The only state accepted |
| `mismatch` | The file is not the bytes the manifest binds, or it describes another graph — for example two workspaces whose records were swapped in a copy, every file well-formed and only the pairing wrong |
| `unrecorded` | The manifest binds no record |
| `unverifiable` | The manifest binds one and it is not here, cannot be read, or is not a record this revision reads |

A bound record from `files` that leaves one of the four roles unidentified is
refused too, with its binding `verified`: a required input went unidentified.

**The origin is a separate fact, shown beside the binding state, never in
place of it.** `files` means the graph was built from real inputs; `synthetic`
means a demo, a test or the deployment probe built it in memory. A synthetic
workspace is fully bound and passes every check, and is shown as synthetic
wherever its record is shown:

- the training log (`Dataset record kg.provenance.json: binding verified,
  origin synthetic`);
- serving's pipeline configuration and the API's pipeline status
  (`dataset_record`: binding, origin, the record's digest, its sources and
  `incomplete_by_design`);
- the measurement report's manifest (`dataset_record`) and its console summary;
- the inventory (`scripts/audit_checkpoint_family.py --inventory`), which lists
  the record's verdict with its binding state and origin as separate fields;
- since contract M3c, the served identity (§4.7): `dataset_record` `verified`
  with origin `synthetic`, in every diagnosis response and in the API's
  pipeline status;
- the Diagnosis tab: a `SYNTHETIC WORKSPACE: not a clinical result.` banner
  above the results and in the model status, and `dataset synthetic ✓` on the
  result's "Produced by" line;
- the downloadable report: the same banner in its "What produced this result"
  section, and its file name, `diagnosis_SYNTHETIC_<patient>_<session>_report.md`;
- the CSV export: `served_dataset_origin` `synthetic` on every row, and its file
  name, `diagnosis_SYNTHETIC_<patient>_<session>.csv`.

What the record says it does not establish (`incomplete_by_design`) is
displayed, never refused. Serving without a model — path reasoning over
`kg.json` alone — verifies no workspace and is unaffected; its served identity
says so (checkpoint `absent`, graph `unverified`, dataset record `absent`;
§4.7, "Serving without a model"). The pipeline reload lists
`kg.provenance.json` among the required files.

### Older workspaces

**Not supported.** A workspace whose manifest binds no record is refused as
`unrecorded`, and one whose record is gone as `unverifiable`; there is no
compatibility mode. Rebuild it with
`scripts/build_knowledge_graph.py --generate-samples`. The read-only inventory
lists which of your workspaces are affected before you deploy.

> **History.** Until contract M3b the record was reported and never enforced:
> a status reader (`workspace_provenance_status()`, removed) returned a state
> and nothing refused on it. An earlier version still had the graph verifier
> raise on a broken record before a tensor was loaded, which was withdrawn
> then. The owner's decision reverses that policy for the formal pipeline; the
> graph check and the record check stay separate checks with separate states.

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
| Exactly one candidate | It is used, and logged with its `data-version` and digest |
| **More than one** | **Refused**, listing each one's path, `data-version`, digest and any declared imports |
| None, a source configured | Downloaded, checked, then placed in the cache (below) |
| None, no source | Refused, naming the roots searched and saying nothing was attempted |

A **release product** is the ontology it is a product of: `mondo-simple.obo`,
`mondo-base.obo`, `hp-base.obo`, `hp-international.obo` and
`hp-simple-non-classified.obo` are MONDO and HPO candidates like `mondo.obo` and
`hp.obo`. So a root holding `mondo.obo` and `mondo-simple.obo` holds two MONDO
candidates and refuses until one is named.

Files whose names start with `.` are never candidates. A file named like one
ontology that declares another is passed over **with a warning** naming both.

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

### How a download lands

- **Nothing on disk is overwritten that is not this ontology.** If the cache
  path the download would write to (`<cache>/hpo.obo`, say) already holds a file
  that is not an HPO candidate, the build refuses before fetching anything and
  names the file. Move it, or name it with the path argument of the slot it
  belongs to.
- **Checked before it is kept.** The download lands in a hidden file of its
  own next to its destination (`.hpo.obo.<random>.staged`) and is opened with
  the same import and slot checks as any other input. Only a file that passes replaces
  the cached one; one that fails is deleted and the previous file is untouched.
  This holds for `--force-download` too, and for two builds fetching into
  the same cache at once: each checks and publishes only its own download,
  and each records the digest of exactly the file it read.
- **A failed download is reported, never substituted.** If no source delivers,
  the build stops with the manual download route; it does not fall back to
  whatever file happens to be on disk.

Every refusal in this section is printed as a message, not a traceback, and
the build exits with status **2**.

### Two refusals a build can now hit

| Refusal | What it means |
|---|---|
| "declares N import(s) and this project requires a self-contained ontology file" | The file names other ontologies to pull in. They are **not** fetched: resolving them would add content this file's digest does not cover, and ignoring them would build a graph missing what they carry while looking complete. Supply a file that declares no imports |
| "declares itself to be X, and it is being used as the Y input" / "carries no `HP:` terms" | The file is not the ontology that slot asked for. Left alone it builds **zero** nodes of that type, with a provenance record that looks complete |

### Where the download URLs live

`configs/deployment.yaml`, under `ontology.sources`. They used to be a constant
in `src/ontology/loader.py`, so a PURL that stopped resolving meant editing
code; now it is a configuration change. Leave the key unset to use the project's
defaults: the OBO Foundry PURL of each of the four ontologies, with the OWL PURL
as a second attempt. A source listed as `hp` is the HPO source.

Two rules apply to **every request**, including each redirect — and every source
here is a PURL, which *is* a redirect:

- **Scheme:** `http` and `https` only.
- **Destination:** every address the host resolves to must be globally
  routable. A host inside your own network is refused unless it is listed under
  `ontology.allowed_hosts` — which is how you enable an in-house mirror, on
  purpose rather than by accident. List the **bare host name**
  (`mirror.hospital.example`); an entry with a scheme, path, port or `@` can
  never match and is refused when the file is read.
- **Behind a proxy:** if this machine cannot resolve a name itself and the
  request goes to the proxy set in `http_proxy` / `https_proxy` (and the host
  is not in `no_proxy` — an entry may carry a port, `mirror.example:8080`, and
  is matched exactly as Python's own `urllib` matches it), the name is left to
  the proxy, because on a
  proxy-only network external names have no local answer. What the proxy then
  reaches is the proxy's policy. A name that *does* resolve here to an internal
  address is still refused, proxy or not. Without a proxy, an unresolvable name
  is reported as a failed download, not as a policy refusal.

**Not yet delivered:** editing these URLs from the interface. The requirement is
recorded and the backend is built for it — one configuration model, one
download-and-verify entry point that a UI and the CLI would share — but the
interface itself is outside this programme's scope. Until it exists, editing the
file is the way.

**Changing a source does not change what is being served.** Editing a URL,
downloading a file and building a workspace are three separate things, and only
a new build produces a new workspace; putting one into service is the existing
load-and-publish step, unchanged.

### Checking the download on this machine

The tests cover the download rules with controlled responses; only the machine
itself can show that its network, proxy and certificates let a real download
through. The deployment probe has an opt-in phase for that:

```bash
python scripts/probe_deployment.py --work-dir /tmp/shepherd_probe \
    --report probe_report.json --download-only
```

It downloads MONDO and HPO **in full** (tens of megabytes) from the sources in
`configs/deployment.yaml`, through the same path a build uses, into the work
directory — never into your cache or configured roots, and it checks that
afterwards. `--download` adds the phase to a full probe run instead. No GPU is
needed, and a `--download-only` run launches nothing on one: it describes the
software stack and checks nothing about the device, so a GPU problem cannot fail
a download check.

In the report, probes `G1` (MONDO) and `G2` (HPO) each carry an `outcome`:

| `outcome` | Meaning |
|---|---|
| `published` | Downloaded, passed the import and slot checks, and published as the verified bytes. `source_digest` is the exact identity of what arrived; `data_version` is its declared release when that has the standard form (`releases/2026-09-01`, `hp/releases/…`); anything else the file declares is withheld (`data_version_withheld: true`) and only its date kept |
| `refused_by_policy` | Every source was refused by the scheme or destination rule — a configuration decision (`ontology.sources` / `ontology.allowed_hosts`) |
| `transfer_failed` | No source delivered. `attempts` says how each one failed: `host_unresolved` (no DNS answer and no proxy to hand it to), `truncated`, or `transfer_failed` |
| `wrong_ontology` / `imports_declared` | A file arrived and was refused by the slot or imports check |
| `no_source` / `configuration_invalid` | Nothing is configured to fetch, or the configuration could not be read |

`network_at_probe_time` records, per source, whether its name resolves on this
machine, whether the request goes through a proxy, and whether its host is
allow-listed — for the configured URL itself, not for the redirects it leads to
(an `http://` PURL that redirects to `https://` is judged, and proxied, again at
each hop). Sources are identified by their position in the list; the report
names no host, no proxy and no directory. If `G9` finds a change in your cache
or roots it reports which location and how many files, not their names.

## 4.6 File permissions of the files a build and the service write

The shortest-path table and its sidecar, the split manifest, `kg.provenance.json`,
the evaluation ledger, the UI config and the ontology cache are all written to a
temporary file and renamed into place, so a reader never sees a half-written one.
They are published with the mode a plain `open()` would give them:

- **a new file** gets the directory's policy — the umask (commonly `022`, which
  gives `0644`), or the directory's default ACL where it has one;
- **a rewritten file keeps its mode** — an operator's `chmod 600` stays `600`,
  and a group-shared `640` stays `640`.

From September 2026 until this was fixed, these files were published `0600`,
readable only by the account that wrote them. Files written in that window keep
that mode; if the service reads them as another account, `chmod 644` (or `640`
with a shared group) them once.

**What renaming cannot reproduce.** The rewritten file is a new file: it is
owned by the account that wrote it, in that account's group — or the
directory's group, if the directory is setgid — and extended ACL entries on the
old file are not copied. For a service that runs as a different account from
the one that builds, give both a shared group and make the workspace directory
setgid, or set a default ACL on it. The reading account also needs search
(`x`) permission on every directory above the file.

**While it is being written**, the temporary file already has the mode the
published file will have, so anyone who may read the file may read the partial
copy too. And **the directories must not be writable by accounts you do not
trust**: an account that can rename files in them can substitute a file
between its write and its rename, as it could before this change.

## 4.7 What produced a diagnosis: the served identity (since contract M3c)

Since contract M3c (2026-10-16) every diagnosis carries a **served identity**:
the checkpoint, the graph, the workspace manifest, the shortest-path table and
the dataset record that produced it, by digest or identifier, the scoring mode
and the serving settings in effect — **each part stating its state in words**.
The pipeline builds it from the reads it took when it was built; nothing is
hashed again when a result is produced, and nothing is filled in with
"unknown".

### The diagnose response changed

`POST /api/v1/diagnose`:

- **`model_version` is removed**, with no replacement string. It always read
  `1.0.0`, whatever was served. A client that read it must stop.
- **`served_identity` is added** and is always present. It is copied from the
  result the response answers with, never from the service's shared state, so
  a reload that lands while a request runs cannot relabel its answer.
- **It is `null` only for the demo answer.** With no pipeline configured at all,
  the service still answers with illustrative candidates and the warning
  `Pipeline not fully initialized - using mock data`; nothing was served, so
  nothing is identified. A deployment that configured a pipeline and could not
  build it answers 503, never the demo answer.

The service's own version (`1.0.0` in `/health`, `/` and the Swagger title) is
the API's version, not the identity of what is served.

### Its parts and states

```json
"served_identity": {
  "scoring_mode": "gnn_plus_shortest_path",
  "checkpoint": {"state": "verified", "sha256": "<SHA-256 of the checkpoint read>"},
  "graph": {"state": "verified", "kg_sha256": "<SHA-256 of kg.json>",
            "split_manifest_sha256": "<SHA-256 of split_manifest.json>"},
  "shortest_paths": {"state": "verified", "build_id": "<the sidecar's build_id>", "max_hops": 5},
  "dataset_record": {"state": "verified", "origin": "files",
                     "sha256": "<SHA-256 of kg.provenance.json>"},
  "settings": {"device": "cuda:0", "device_requested": "auto", "eta": 0.7,
               "max_path_length": 4, "path_length_penalty": 0.9,
               "aggregation_method": "weighted_sum", "max_paths_per_candidate": 10,
               "use_direct_path_optimization": true,
               "phenotype_normalisation_version": 1}
}
```

| Part | State | Meaning | Values |
|---|---|---|---|
| `checkpoint` | `verified` | The model was built from one read of a checkpoint whose record matched the served graph (contract M2.4) | `sha256`: that read's digest |
| | `supplied_in_memory` | A program handed the pipeline a model object; no checkpoint bytes were read | `sha256` null |
| | `absent` | No model: path reasoning ranks the candidates | `sha256` null |
| `graph` | `verified` | `kg.json` and `split_manifest.json` were each read once, and the graph's digest matched the manifest | `kg_sha256`, `split_manifest_sha256` |
| | `unverified` | The graph was read from a file and checked against no workspace manifest (serving without a model, below) | `kg_sha256`; `split_manifest_sha256` null |
| | `in_memory` | A program supplied the graph in memory; no file was read | both null |
| `shortest_paths` | `verified` | The table and its sidecar share one `build_id` and name the served graph (§4.3) | `build_id`; `max_hops`, the sidecar's |
| | `absent` | No table contributes to the score | both null |
| `dataset_record` | `verified` | The record the manifest binds, naming the served graph (§4.4); verified exactly when the graph is | `origin` (`files` or `synthetic`); `sha256` of `kg.provenance.json` |
| | `absent` | No workspace was verified, so no record was read | all null |
| `scoring_mode` | — | `gnn_plus_shortest_path`, `gnn_only` or `path_reasoning_fallback` | |
| `settings` | — | The serving settings that change scores, as in effect | below |

`settings`:

- `device`: where the model ran — torch's name for the device its parameters
  are on (`cuda:0`, `cpu`). Null with no model.
- `device_requested`: what was asked for — `auto` when nothing was named,
  `cuda`, `cuda:N` or `cpu` (§4.8). Null with no model.
- `eta`: the mixing weight in effect — the configured value with a table,
  `1.0` without one. Null with no model.
- `max_path_length`, `path_length_penalty`, `aggregation_method`,
  `max_paths_per_candidate`, `use_direct_path_optimization`: path discovery
  and scoring, in effect in every mode, because the candidates are the
  diseases these paths reach.
- `phenotype_normalisation_version`: the rule applied to the phenotype list
  before scoring (currently `1`).

The parts cannot contradict each other: a model is served exactly when the
mode is not `path_reasoning_fallback`; the table is `verified` exactly in
`gnn_plus_shortest_path`; a verified checkpoint or table is served only over a
verified graph. A value is null only where its part's state says there is
nothing to identify.

### The same field in the pipeline status

`GET /api/v1/pipeline/status` carries `served_identity` for the pipeline loaded
now, beside its existing fields, which are unchanged (`scoring_mode`,
`sp_kg_binding`, `dataset_record`, `checkpoint_meta` with its `device`, …). It
is null when nothing is loaded. **The status describes what is loaded now, and
a reload replaces it.** To know what produced a given result, read that
result's own `served_identity`, or its report or CSV, not the status.

### How to read it

- **A normal deployment** reads: checkpoint `verified`, graph `verified`,
  shortest paths `verified`, dataset record `verified` with origin `files`,
  `gnn_plus_shortest_path`, device `cuda:0`, requested `auto` or `cuda`.
- **Compare the digests with the files you deployed**: the SHA-256 of the
  checkpoint, `kg.json`, `split_manifest.json` and `kg.provenance.json` equal
  the identity's digests, and the table's `build_id` is the one in
  `shortest_paths.meta.json`.
- **Anything else is stated, not hidden**: origin `synthetic` is a generated
  workspace, not clinical data (§4.4); shortest paths `absent` with `gnn_only`
  and eta `1.0` means no table was served (§4.3); checkpoint `absent` is path
  reasoning without a model (below); device `cpu` was asked for explicitly
  (§4.8).
- **What it does not establish** is displayed under the identity, as the
  contract's claim boundary: "Checked: these files are the ones their records
  name, and the records agree on the relations marked ✓. Not checked: who
  changed them, or anything a person able to edit both files and records could
  alter."

### Serving without a model

Path reasoning over `kg.json` alone, with no checkpoint configured, verifies no
workspace (§4.4), and its identity says so: checkpoint `absent`; graph
`unverified`, with the digest of the `kg.json` it read and no manifest digest;
shortest paths and dataset record `absent`; `path_reasoning_fallback`; device,
requested device and eta null. Nothing in it is claimed verified. Whether this
mode stays available by default is an open policy question (B-2), not decided
by M3c.

### In the Diagnosis tab and its exports

The WebUI shows **the result's own identity**, never the status's:

- **Banners above the results**, one per state that must not be missed:
  `SYNTHETIC WORKSPACE: not a clinical result.`, `NO TRAINED MODEL:
  path-reasoning fallback.`, `NO SHORTEST-PATH TABLE: GNN score alone.`,
  `DEMO ANSWER: not computed.`, and `WHAT PRODUCED THIS RESULT WAS NOT
  REPORTED.` for a response without the field (a server from before M3c).
- **A "Produced by" line** under the results, with shortened digests, and a
  closed block, "What produced this result — full identity", holding the full
  table, the claim boundary and "Not a certified medical device; decision
  support for a qualified clinician."
- **The model status** (Model Configuration) shows the loaded pipeline's
  identity on its "Serving" line, and names the served workspace and
  checkpoint file on one line. Once a pipeline is loaded, the Data Directory
  and Checkpoint fields show those two paths, a file chosen automatically
  included, so Load / Reload reloads the checkpoint file that is served;
  clear the Checkpoint field to let the server choose again. Two cases
  differ: with no checkpoint served the Checkpoint field is blank, and a
  reload then chooses one automatically (or is refused when the workspace
  has none); after a start from `SHEPHERD_KG_PATH` alone, Data Directory is
  blank and a reload needs a workspace typed. With nothing loaded the fields
  show the paths Save Config had stored when the service started; a Save
  takes effect from the next start.
- **The Markdown report** has a "What produced this result" section before the
  candidates: the same banners, a Part | State | Value table with digests in
  full (`sha256:<64 hex>`), the claim boundary and the fixed statement. An
  absent part is written as its state and the reason, never left blank.
- **The CSV** carries the identity in columns (below).
- **File names** carry the two states a copied file must not lose:
  `diagnosis_SYNTHETIC_<patient>_<session>.csv` / `_report.md` for a synthetic
  workspace, `diagnosis_DEMO_…` for the demo answer; otherwise unchanged.

### The CSV's identity columns

The CSV keeps **one header row** and no preamble. The 15 candidate columns come
first, unchanged in order; 21 columns follow, each holding one part of the
identity, **the same on every row**:

| Column | Holds | When the identity holds null |
|---|---|---|
| `served_scoring_mode` | `gnn_plus_shortest_path`, `gnn_only` or `path_reasoning_fallback` | — |
| `served_checkpoint_state` | `verified`, `supplied_in_memory` or `absent` | — |
| `served_checkpoint_sha256` | `sha256:<hex>` of the checkpoint read | `absent` (no model); `none` (a model supplied in memory) |
| `served_kg_state` | `verified`, `unverified` or `in_memory` | — |
| `served_kg_sha256` | `sha256:<hex>` of the `kg.json` read | `none` (a graph supplied in memory) |
| `served_split_manifest_sha256` | `sha256:<hex>` of the `split_manifest.json` read | `absent` (graph `unverified` or `in_memory`) |
| `served_sp_state` | `verified` or `absent` | — |
| `served_sp_build_id` | the `build_id` the table and its sidecar share | `absent` |
| `served_sp_max_hops` | the sidecar's `max_hops` | `absent` |
| `served_dataset_record_state` | `verified` or `absent` | — |
| `served_dataset_origin` | `files` or `synthetic` | `absent` |
| `served_dataset_record_sha256` | `sha256:<hex>` of the `kg.provenance.json` read | `absent` |
| `served_device` | the device in effect (`cuda:0`, `cpu`) | `none` (no model); `not_stated` (a model supplied in memory that does not say) |
| `served_device_requested` | `auto`, `cuda`, `cuda:N` or `cpu`, as asked | `none` (no model, or a model supplied in memory with no device named) |
| `served_eta` | the mixing weight in effect | `none` (no model) |
| `served_max_path_length` | integer | — |
| `served_path_length_penalty` | number | — |
| `served_aggregation_method` | text | — |
| `served_max_paths_per_candidate` | integer | — |
| `served_direct_path_optimization` | `true` or `false` (`use_direct_path_optimization`) | — |
| `served_phenotype_normalisation_version` | integer (currently `1`) | — |

- **Digests are written in full with a `sha256:` prefix** (`sha256:<64 hex>`).
  The prefix names the algorithm and stops a spreadsheet reading an all-digit
  digest as a number.
- Booleans are `true` / `false`; numbers are written as the API sends them
  (`0.7`, `1.0`).
- **No identity cell is empty.** A value the identity holds as null is written
  as a word (the candidate columns keep their earlier convention, where an empty
  cell means no value):

| Word | Written when |
|---|---|
| `absent` | the part is absent: no checkpoint, table, manifest or dataset record |
| `none` | nothing applies: no device, requested device or eta without a model; no digest for a model or graph supplied in memory; no requested device for a model supplied in memory |
| `not_stated` | a model supplied in memory does not say its device (`served_device` only) |
| `demo` | every `served_` column of the demo answer |
| `not_reported` | every `served_` column of a response that carried no `served_identity`, or a value one left out |

- A text cell that a spreadsheet would run as a formula — one starting with
  `=`, `+`, `-`, `@`, a tab or a carriage return — is written with a leading
  `'`. This covers the free-text candidate columns too; numeric columns are
  unaffected.

## 4.8 The serving device (`SHEPHERD_DEVICE`)

**CUDA is a requirement of this project, and serving does not substitute the
CPU for it** (since contract M3c, 2026-10-16; training refuses the same way
since M3a).

| Value | Meaning |
|---|---|
| unset, empty or `auto` | CUDA. **Refused** on a host without CUDA |
| `cuda`, `cuda:N` | That CUDA device. Refused without CUDA |
| `cpu` | The CPU, asked for explicitly: **for development only** |
| anything else | Handed to torch as given, as training does |

Where it is set:

- **At startup**: `SHEPHERD_DEVICE` in the service's environment.
- **On a reload**: the request's `device` (`POST /api/v1/pipeline/reload`). Null
  or blank falls back to `SHEPHERD_DEVICE`, and then to `auto`; a blank variable
  counts as unset too.

**When it is checked**: whenever a checkpoint is to be served, before anything
is read — the service settles the device before it parses `kg.json` (a reload,
right after its file check, before automatic selection reads any candidate
checkpoint), and the pipeline again before it reads the workspace or the
checkpoint. A refused
startup publishes nothing and `/api/v1/diagnose` answers 503; a refused reload
leaves the served pipeline serving. The refusal reads:

> Serving device 'auto' requires CUDA, which is not available here. Hardware
> acceleration is a hard requirement for deployment and CUDA is currently the
> only supported backend, so serving does not substitute the CPU for it. Set
> SHEPHERD_DEVICE=cpu (or send device 'cpu' with a reload) explicitly for
> development; the served identity records the device asked for and the one in
> effect.

Serving with no checkpoint — path reasoning alone — resolves no device.

**Hardware acceleration is required for deployment, and CUDA (NVIDIA) is currently the only
supported backend.** That is the present scope, not a permanent limit. torch's ROCm builds also
name their device `cuda`, but they have not been verified here. Another accelerator would be
adapted and accepted on its actual hardware, with its own selection rule; until then no other
backend is offered, and none is substituted silently (`docs/working/BACKLOG.md`, item 24).

**The served identity records both** (§4.7): `settings.device_requested` as
asked (`auto` when nothing was named) and `settings.device` as in effect, the
device the model's parameters are on (`cuda:0`, not just `cuda`). The status's
`checkpoint_meta.device` stays.

> **Behaviour change (M3c).** A host without CUDA that served with no device
> named used to fall back to the CPU without refusing or warning, after
> reading the checkpoint and building the model; only the log and the status's
> `checkpoint_meta.device` said so. **It is now refused.** A CPU development host must set `SHEPHERD_DEVICE=cpu` (and send
> `"device": "cpu"` with a reload, or leave it null so that variable applies).
> A literal `auto`, which used to reach torch and fail there, is the same
> request as unset. The WebUI has no device selector.
> `scripts/setup_demo.py` prints its serving command with the device its demo
> was trained on.

---

## 4.9 Test cohorts: import, measure, record (since contract M4)

Since contract M4 (2026-10-16) a checkpoint is tested from the command line on a **registered
test cohort version** — MyGene2, the institution's offline cohort, or a synthetic cohort for an
acceptance — and its result is recorded beside the checkpoint. The flow is:

1. `scripts/import_cohort.py` previews, then commits, a cohort version mapped to one workspace's
   graph;
2. `scripts/measure_scorer.py --cohort LABEL --registry DIR` measures the checkpoint on it (Mode
   C) and keeps the full report;
3. `--record`, or later `scripts/record_evaluation.py --report`, registers the report in
   `evaluations.json` beside the checkpoint;
4. the Diagnosis tab's model status shows the loaded checkpoint's recorded results.

All of it runs on the machine, by the operator. No UI control and no API request imports,
measures or records anything.

### The registry

The registry is a directory **you create**, given to every command as an **absolute path outside
the repository**. A relative path, a path that does not exist, and a path that resolves inside
the repository (through a symlink too) are refused. Nothing creates it for you, and no request
names it. The commands take it as `--registry`; the service reads it from `SHEPHERD_REGISTRY`
(§4.12), checked by the same rule.

The directories below it are checked too. Each one (`sources/`, `versions/`, a version's
directory, its `reports/` and each run directory) is resolved again just before anything is
written into it or a version is read from it. It is refused when a link takes it into the
repository, or outside the directory it belongs under: the registry, or for `reports/` and a
run directory, the version's own directory. A link that stays inside the registry is
followed. The refusal says what had already been written, for example only the source copy.
These checks run just before each write, not atomically with it, so a link swapped in at that
moment is not caught. The registry is yours, on this machine, with one writer at a time.

It holds patient-level data — the source file kept verbatim, the mapped samples and the per-case
ranks — so create it where only the accounts that run these commands can read it (for example
`mkdir -m 700 /srv/shepherd/cohorts`). Its files are published as §4.6 describes: staged, then
renamed into place, with the directory's permission policy.

```
<registry>/
  sources/<sha256>.json                  the source file's bytes, verbatim, named by their digest
  versions/<label>/cohort.json           the version manifest, written last: it makes the version exist
  versions/<label>/samples.json          the mapped samples, bound by cohort.json's digest
  versions/<label>/reports/<run id>/     one directory per measurement: ranks.json, then report.json
```

**A version never changes.** A label is used once. A `versions/<label>/` without `cohort.json`
is an interrupted import: every command refuses it, and the importer names it so you can remove
it before reusing the label. No command deletes anything in the registry.

### The source file: `shepherd-cases/1`

The importer reads one format, the project's shared import format. Its name says how a file is
laid out, not where its cases come from: a synthetic cohort and a converted real one are written
the same way (a real source gets an adapter that writes this format):

```json
{
  "format": "shepherd-cases/1",
  "cases": [
    {"case_id": "c001", "phenotypes": ["HP:0001250", "HP:0001263"], "disease": "MONDO:0011073"}
  ]
}
```

- Exactly these keys, at both levels. `case_id` is a non-empty string of printable text (no
  control or format character, no separator other than the space, no lone surrogate), unique
  in the file; `phenotypes` a non-empty list of `HP:` + seven digits; `disease` `MONDO:` +
  seven digits.
- **Anything that does not fit refuses the import as a whole**, with every problem listed by case
  index and case id; nothing is written. A problem says where a value is wrong (the case, the
  field, the position) and what type it was or which rule it broke, **never the value**, so a
  refusal never prints a phenotype or a diagnosis. A case id that is not printable text is
  shown escaped.
- **A disease given as `OMIM:`, `ORPHA:` or `Orphanet:`** also refuses the import as a whole,
  naming the namespace and the number of cases: the graph's diseases are MONDO terms, and this
  revision does not translate. Real formats (MyGene2's, the institution's) are converted to this
  format by an adapter, written once samples exist (`docs/working/PLAN_TEST_RESULTS.md` §4.9).

### Import: preview, then commit

```bash
python scripts/import_cohort.py --registry /srv/shepherd/cohorts \
    --data-dir data/workspaces/<ws> --source /srv/intake/cases.json \
    --label mygene2-2026-10 --role mygene2 [--partial-phenotypes keep|exclude]
```

Labels and roles are lowercase identifiers (letters, digits, `_`, `-`, `.`; at most 64
characters); a role may not be `train` or `val`.

**The preview writes nothing.** It maps every case against the graph the workspace manifest
binds, through the same readers and checks as training and serving, and prints:

- the source cases, the cases eligible for ranking, and the excluded cases **by category, each
  with its case id and reasons** (on this machine only: case ids appear, phenotypes and diseases
  do not; a case id that is not printable ASCII is printed escaped):
  - `ground_truth_absent` — the case's disease is not a disease node of this graph;
  - `phenotype_unmappable` — no term maps, or, with `--partial-phenotypes exclude`, some term does
    not;
- the HPO terms that did not map (an obsolete or alternative id is not a node, so it is one of
  them), the cases ranked on fewer terms than they listed, and the repeats removed (a term listed
  twice counts once, phenotype normalisation rule 1);
- the graph's four digests, the dataset record's origin and digest, and its HPO and MONDO
  entries;
- a **preview id** and the exact commit command.

`--partial-phenotypes keep` (the default) ranks a case on the terms that mapped;
`exclude` excludes it. A case with no term that maps is always excluded. The policy is recorded in
the version.

**Commit** by running the printed command: the same arguments plus `--commit <preview id>`. The
importer reads every input again and publishes only if they still give that id; otherwise it
refuses ("inputs changed since the preview"). It prints the version's directory and the manifest's
SHA-256. A source in which no case can be ranked is refused at commit, since there would be nothing
to measure.

### Measure

```bash
python scripts/measure_scorer.py \
    --checkpoint data/workspaces/<ws>/runs/<arch>/<run>/checkpoints/<file>.pt \
    --data-dir data/workspaces/<ws> \
    --cohort mygene2-2026-10 --registry /srv/shepherd/cohorts \
    [--training-evidence DIR ...] [--record] [--device auto|cuda|cpu] [--batch-size 32]
```

Before anything is written the run checks, on the bytes it read: the version's `cohort.json` and
its `samples.json` against the digest it binds; **the version's counts**, which become the
report's and the ledger's denominators — they must be present and agree with each other and with
the version (`ranked_eligible` = `samples.n`; `source_cases` = `ranked_eligible` + `n_excluded`
= the source's case count; `n_excluded` = the sum of the categories = the exclusions listed),
and the samples read must number `samples.n`; the workspace's graph files against its manifest
and the dataset record (§4.4); **the version against this workspace** — its four graph digests and
its dataset record must be this workspace's, or the run is refused before the checkpoint is read;
then the checkpoint (read with `weights_only=True`) against the graph it was trained on. The
device follows §4.8: `auto` and `cuda` need CUDA and are refused without it before anything is
read, and `cpu` is a development run, recorded as such and a different measurement from a CUDA
one.

**R10 — overlap with what the model was trained and validated on.** The run looks for the
training and validation sample files whose digests the checkpoint records: first the workspace's
own `train_samples.json` and `val_samples.json`, then every `*_samples.json` directly inside each
`--training-evidence` directory (give the directory where the training run's workspace is, if it
was trained elsewhere or the workspace has been rebuilt since). Only a file whose bytes have the
recorded digest counts. Four labels: training and validation × disease overlap and phenotype-set
overlap, each `overlap` (with its count and denominator), `none`, `unverifiable` (**never shown
as zero**), or, for validation, `no validation inputs recorded` (the checkpoint records its
training inputs and has no validation entry at all: training ran no validation pass).
`unverifiable` gives its scope's reason:

- **not found**: no file looked at holds the recorded bytes. Measure again with
  `--training-evidence` naming the directory that holds them;
- **found, not readable**: the file that holds them was found and does not parse (for example
  a non-integer id, or a case that lists a phenotype twice);
- **not recorded by the checkpoint**: it names no file to look for in this scope, and its
  record is not one the pipeline wrote: every training run records its training inputs as a
  digest, so a record without one, or with an entry that is there and is not a digest, cannot
  come from it. `--training-evidence` cannot help.

`none` is not proof of no leakage: it speaks only about the inputs the checkpoint records.

**What refuses the run instead of giving a label.** A `--training-evidence` directory that does
not exist, is not a directory, or cannot be listed (permission, an I/O error on a network mount)
is refused before any candidate is read: a directory that is not looked in would report the
inputs as not found. A candidate that exists but cannot be read (permission, an I/O error)
refuses the run, and nothing is written, because it may be the evidence; make it readable, or
leave its directory out of `--training-evidence`. Candidates are read only until every recorded
digest has been found, so an unreadable file after the match is never reached.

**Where the report is kept.** In the registry, under the version:
`<registry>/versions/<label>/reports/<UTC time>-<first 12 hex of the checkpoint digest>-<8 random
hex>/`, a new directory for every run: `ranks.json` (per case, patient-level) and then
`report.json`, which names the ranks file by its digest. Reports are never overwritten, and every
report is kept whether or not it is registered. `--output` is refused with `--cohort`.

The console prints Mode C's metrics, the coverage (`N ranked of M source cases; K excluded`, with
each category), the four R10 labels in the words the Diagnosis tab uses (an `unverifiable` label
with its reason), the label definitions once, the report's path and the run id. The report itself
keeps the states and holds no prose.

### Record

- **`--record`** registers the report right after the run. **Without it**, and when `--record`
  is refused (exit 3), the run prints the command that registers it later, on one line, naming
  the file it measured:

  ```bash
  python scripts/record_evaluation.py --report "<registry>/versions/<label>/reports/<run id>/report.json" --checkpoint "data/workspaces/<ws>/runs/<arch>/<run>/checkpoints/<file>.pt"
  ```

  The checkpoint is the file the run read, at its resolved path, so the command works whatever
  the file is named. Each path is double-quoted, so a path with a space stays one argument in a
  POSIX shell, cmd or PowerShell. A path holding `"`, `$`, a backtick, `%` or `!` is expanded by
  one shell or another and has to be quoted by hand.
- **`record_evaluation.py` names the checkpoint with exactly one of two arguments** (neither or
  both is a rejected command line):
  - `--checkpoint FILE`: that one file, under any name, must hold the bytes the report measured;
  - `--checkpoint-dir DIR`: a `*.pt` file directly in DIR must hold them.

  Otherwise nothing is written. `--record` and `record_evaluation.py` go through the same
  function, so the record is the same.
- **The ledger** is `evaluations.json` in the directory of the checkpoint's **resolved** path,
  whichever way the report arrives: `--record`, `--checkpoint FILE`, or `--checkpoint-dir DIR`
  (beside the matching file's resolved path, which is not DIR when that file is a link). Under the
  run layout it is `data/workspaces/<ws>/runs/<arch>/<run>/checkpoints/evaluations.json`. A
  symlinked checkpoint files its ledger beside the link's target, which is where the service
  looks, so one set of weights never gets a second ledger beside a link. The measurement and the
  service both read the checkpoint at its resolved path, so the directory is settled when the file
  is read. A link retargeted during a run, or after a load without a reload, does not move where
  that run's record is filed or where the loaded model's records are looked up. No directory is
  created for a ledger.
- **Outcomes:** registered (with its sequence number); already held (the same report again:
  nothing written); agrees with an existing record (the report's digest is added to that record's
  corroborating reports). R10 and the environment are not compared, because each is true of its
  own report. When a corroborating report's R10 or environment differ from the record's (say, a
  re-run with `--training-evidence` found evidence the first run did not), **the record keeps
  its own values and is not updated**. The registration says so in one line, naming the report,
  which keeps the later run's R10. A report that contradicts a record with the same key — same
  checkpoint, cohort version and measurement settings, different results — is refused, naming
  the fields that differ. There is no override flag; a wrong record is corrected by editing the
  readable JSON file, which leaves a diff.
- **What is shown later:** `python scripts/record_evaluation.py --checkpoint <file> --show
  <checkpoint sha256>` reads the ledger the service reads for that file, beside its resolved
  path. With `--checkpoint-dir <dir>` instead, `--show` reads the `evaluations.json` inside that
  directory, which for a directory of links is not the one the service reads. An empty answer
  means this ledger holds no result for those weights, not that they were never evaluated.
- **One writer at a time.** A registration that finds the ledger changed since it read it is
  refused rather than overwriting; this is not a lock, so do not register into one directory from
  two places at once.
- Only a **registered cohort version's** report is recorded; a report from `--split` is not.

### Exit codes

| Command | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| `import_cohort.py` | preview printed, or version committed | refused, with one message ending "Nothing was written." (also a preview in which no case can be ranked) | the command line was rejected | — |
| `measure_scorer.py` | measured (with `--record`: registered, already held or agreeing) | a check refused the run: one message, nothing written (so does `--device auto` or `cuda` on a host without CUDA, before anything is read); or **measured, but the report was not written** (a full or read-only disk, for example): one message saying what may be left, which is never a report that can be registered -- this run's ranks beside `--output`, or a run directory without `report.json` in the report store (a run that fails after writing its report, its console closed for example, says where the report is) | the command line was rejected, including an argument of the other kind of cohort | **measured, but the registration refused**: the message names the report and the `record_evaluation.py` command to register it later; or, when the report file no longer holds the bytes the run published, says so and asks for a re-measurement instead (that report may be gone); or, when runs under the report's key disagree, says "runs disagree — investigate" and keeps the report, with no register-later command (none would be accepted while they disagree) |
| `record_evaluation.py` | registered, already held, agreeing, or shown | refused, with one message; nothing written | the command line was rejected, including neither or both of `--checkpoint` and `--checkpoint-dir` | — |
| `audit_split_overlap.py` | report written | refused, with one message; nothing written | the command line was rejected (for example the removed `--evaluation-cohort-kind`, `--registry` without `--cohort` or the reverse, or `--splits` beside `--cohort`), with the usage line | — |

A measurement whose checkpoint directory is read-only to you ends with exit 3: the report is kept,
and whoever may write that directory registers it with the `record_evaluation.py` command the run
printed.

**`--record` registers only the report this run published, of the checkpoint this run read.** The
run keeps the digest of the report bytes it wrote and the identity of its checkpoint read, and the
registration reads the report back and compares the digest of that read before anything in it is
decoded, then the checkpoint it names. If the file was replaced, changed, damaged, removed or
cannot be read in between -- even when it now holds another valid report, of another checkpoint or
of another cohort version, or is now a link to a file outside the report store -- nothing is
registered, the run exits 3, and it says the report at that path is not the one it published (or
is gone), so the run's own report may not exist any more: measure again to register this
measurement. The command for a later registration is printed only when the report read back is
the one published and something else refused it, such as the ledger, and never while runs under
its key disagree. A later registration with `record_evaluation.py` checks the checkpoint file it
is given against the report, as before.

### What the Diagnosis tab shows

The model status (Model Configuration), under its "Serving" line, adds **"Recorded test results
(Mode C)"** for the loaded checkpoint (`sha256:<first 12>…`). It is read through
`GET /api/v1/evaluations/loaded-model` each time the status is read, and found by the checkpoint's
digest, not its file name: after another model is loaded the previous model's results are never
shown, and a checkpoint moved with its ledger and unchanged bytes still finds them.

- The records are introduced once, above the first, by **"Disease ranking by the model,
  approved scorer policy (Mode C)."** and the note **"This is the model's own Mode C disease
  ranking. Until B-1 ships, the Diagnosis tab ranks by the legacy GNN+SP mixture behind the path
  gate, so this number does not describe the tab's current ranking."** The R10 label definitions
  and the line on what was and was not checked follow the last record, once.
- Per record: role · label with the version's manifest digest (`version sha256:<first 12>…`),
  so two imports of one label, in a recreated or a second registry, read as two versions; Mode
  C's metrics over the cases ranked (its MRR is labelled apart from the trainer's "Val MRR" on
  the Checkpoint line); coverage (`ranked/source` with each exclusion count); the relations
  checked (✓ or not applicable); the four R10 labels — `unverifiable` (with its scope's reason,
  as the console gives it) and `no validation inputs recorded` are shown as words, **never as
  zero or a number**; the environment class (CUDA with deterministic algorithms on or off, or
  CPU) and the GPU; the dataset origin (a synthetic workspace is called out); the registration's
  sequence and time and the number of corroborating reports.
- Instead of records, a sentence: no ledger beside the loaded checkpoint; the ledger was not read,
  with the reason, which names the file and never its directory (for example a schema 2 file,
  below); this ledger holds no result for these weights; some records not shown because they do
  not have the shape registration writes; or "The loaded model changed while its recorded results
  were read; refresh the page."
- **The route** takes no parameters, reads the ledger beside the loaded checkpoint on every
  request, writes nothing, and returns no path, case id or rank. "Beside" is the directory of the
  checkpoint's resolved path as the pipeline read it at load, so a link retargeted since, without
  a reload, does not move the lookup away from the served bytes' ledger. It is mounted on its
  own router, not the pipeline router, so a deployment that unmounts or protects the pipeline routes
  (`docs/working/results-review/SPEC_4_DEPLOYMENT_SECURITY.md`, B2) keeps it.

### Behaviour changes (M4)

- **`measure_scorer.py --cohort-kind` is removed**, and so is the loose `<split>_samples.json` a
  workspace used to carry as a supplied cohort. `--split` takes the generator's `train` or `val`
  only and needs `--output`; `--registry`, `--training-evidence` and `--record` are refused with
  it. An institutional or external cohort is imported and measured with `--cohort LABEL
  --registry DIR`.
- **A generated split's report is not registrable**: `record_evaluation.py` refuses it ("a
  generated split's report is not registered"). `--split val` is the checkpoint-selection split,
  not held-out data.
- **A generated split's report and ranks are staged and renamed into place** (ranks first); an
  existing `--output` is still replaced.
- **Ledger schema 3.** An existing `evaluations.json` of schema 1 or 2 is refused — by
  `--record`, by `record_evaluation.py` (`--report` and `--show`) and by the model status, which
  reports the ledger as not read. Nothing migrates it and its results are not carried over:
  **archive it by hand**, for example `mv evaluations.json evaluations.v2.json`, read it with the
  revision that wrote it, and re-measure what you need with this revision, then register the new
  reports (the refusal says the same). A report written before M4 is refused too, with an
  instruction to re-measure.
- **`record_evaluation.py`** takes `--checkpoint FILE` or `--checkpoint-dir DIR`, exactly one.
  With either, the ledger is written beside the checkpoint's resolved path. For a directory whose
  checkpoint is a link, that is beside the link's target, no longer in the directory named.
- **`scripts/audit_split_overlap.py`**: `--evaluation-cohort-kind` is removed; a test cohort is
  audited with `--registry DIR --cohort LABEL`. `--registry` without `--cohort` (or the reverse),
  and `--splits` beside `--cohort`, are rejected command lines (exit 2, with the usage line). The
  version's graph and dataset record digests are compared with those this workspace's manifest
  records, without loading tensors; only `train` is read. The report's `schema_version` is 6.

### Since M5-1: registration from the report store, disagreeing runs, history, determinism

These apply to `measure_scorer.py --record` and `record_evaluation.py --report` alike, since
both go through the one registration function. The exit codes above are unchanged.

**A report is registered only from the registry's report store.** `--report` must name
`<registry>/versions/<label>/reports/<run id>/report.json`: the path the measurement printed as
`Report:`. Before anything is written, the registration checks four things:

- it finds the registry and the version from that path, and reads the version;
- the report must name that version exactly (label, role, manifest digest, sources, graph,
  mapping and counts);
- its run directory must carry the first 12 hex digits of the checkpoint it measured;
- the checkpoint and ledger checks above still apply.

A copy anywhere else is refused before it is read: "… is not a report the registry's report
store holds … Register the store's original instead". You can still read and share a copy, but
you cannot register it. A link to a stored report counts as the stored original. A `report.json`
in the store that is a link to a file elsewhere does not count. A registry moved as a whole
still works, because no absolute path is stored as an identity. A generated split's report,
written to `--output`, is never in a store, so this is now why it is refused.

**Runs that disagree block their key ("runs disagree — investigate").** The store's reports
under one key (same checkpoint, cohort version and measurement settings) are compared with the
same comparison the ledger uses. If two of them differ in results, no report under that key is
registered. This applies to a run with `--record` or without it, to a later
`record_evaluation.py`, and to registering again a report that was already registered. The
message names what differs and which runs. A record registered before the disagreement appeared
stays in the ledger. Under `--record` the run exits 3 and keeps its report, but prints no
register-later command, since none would be accepted while the runs disagree. Without `--record`
the run checks the store the same way after publishing: it exits 0, as a run that registers
nothing does, prints "Not registered (no --record), and runs disagree — investigate" with what
differs instead of the command, and its `--outcome` file's `registration.why` says the same.
When whether the runs disagree cannot be checked (below), the run without `--record` prints the
command and says a registration is refused until that can be checked. To move forward:

- compare the reports' environment facts (each `report.json`'s `manifest.environment`);
- re-run with `--deterministic`, which is a different key.

An investigation note records who looked, when and what they found. Notes go to
`<registry>/versions/<label>/reports/notes.jsonl`, kept with the key's reports and never in the
ledger, and they do not lift the block. Adding a note from the Test tab comes with M5's user
interface. The block rests on plain files: deleting a disagreeing report lifts it. The tools
never delete anything.

**"Cannot be checked" is refused too, and never read as uncontested.** The whole version cannot
be checked when the store holds:

- a report that cannot be read or parsed;
- a run directory or `report.json` that is a link out of the store;
- a report whose key cannot be told.

A report from an earlier revision, whose manifest this one does not read, blocks only the key
it names. The refusal names each file. Move the file out of `reports/`, or repair it, and
register again.

**The usage history** is `<registry>/versions/<label>/reports/usage.jsonl`, one JSON object
per line, appended and never rewritten. It records:

- every measurement of the version (`test_run`: when, run id, checkpoint and report digests,
  Mode C's metrics, `n_ranked`, whether it was deterministic, the device, whether `--record`
  was asked);
- every registration that wrote a ledger (`registered`: outcome and sequence);
- every model loaded from a list ordered by the version's score (`model_load`; written by the
  Load control that comes with M5's Model Management page).

It holds aggregates only, no case id. The measurement appends its entry right after the report
is published, before it writes its `--outcome` file. If that append fails, the run stops with
exit 1 ("measured, but the run did not finish"), says the report was written and that the run is
not in the history, and does not register; its outcome file is still written, with
`"complete": false`. If the outcome file cannot be written, the run stops the same way and says
it is in the history. If a registration's own entry cannot be appended, the registration stands
and says so on a `NOTE:` line. Every entry starts a line of its own: an append that was cut short
(a crash, a power loss) leaves a broken last line, which the next entry closes rather than
continues, and which a reader names as a line that is not an entry. An append whose write fails
is taken back, so it leaves no part of its entry; if even that fails, the error says the end of
the file may hold part of it.

**New `measure_scorer.py` options.** They are there for M5's job runner, and you can use them
by hand too. `--expect-version-sha256` and `--outcome` apply to `--cohort` runs only (exit 2
with `--split`); the other two apply to both:

```bash
python scripts/measure_scorer.py --checkpoint <file>.pt --data-dir data/workspaces/<ws> \
    --cohort <label> --registry /srv/shepherd/cohorts [--record] \
    [--deterministic] [--expect-checkpoint-sha256 HEX] [--expect-version-sha256 HEX] \
    [--outcome FILE]
```

- `--deterministic` turns on several settings before CUDA starts: PyTorch's deterministic
  algorithms (never warn-only), deterministic cuDNN with its benchmark off, and
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` when that variable is unset. It never falls back to
  non-deterministic execution or to another device. An operation with no deterministic
  implementation refuses the run with nothing written, and so does a `CUBLAS_WORKSPACE_CONFIG`
  that PyTorch does not document as deterministic. The manifest records the settings, so a
  deterministic run is a different key. PyTorch promises reproducibility only on the same
  platform, device and versions; deterministic mode guards against known non-deterministic
  operations and does not certify a number.
- `--expect-checkpoint-sha256` and `--expect-version-sha256` refuse the run before anything is
  measured or written when the checkpoint read, or the version's `cohort.json`, has another
  SHA-256: the file changed after it was chosen.
- `--outcome FILE` writes a small JSON file. FILE must be new and its directory must exist; the
  run refuses before reading anything otherwise, so a file an earlier run left at that path is
  never read as this run's outcome, and a file named by mistake is never overwritten. Name a new
  file for each run (a job's own directory, say). The file is written once the report is published
  (`"complete": false`) and again at the end (`"complete": true`). It gives the report's path,
  run id and digest, the checkpoint and version digests, whether the run was deterministic, and
  under `registration` whether the report was registered and why. A run refused before its
  report writes no outcome file. The file says what the run did. The report store and the ledger
  remain the record, and "kept" and "registered" are read from them. The job runner gives every
  test run it starts a new outcome file in a directory of its own (§4.11).

---

## 4.10 The dashboard on Gradio 6.30.0 (since M5, 2026-10-20)

The dashboard at `/ui` runs on **Gradio 6.30.0** (it was 5.29.1). `pyproject.toml` pins it
exactly, and `uv.lock` changed only in what that forces: gradio-client 2.7.2, starlette 1.9.0
and huggingface-hub 1.16.1, with brotli 1.2.0 and hf-gradio 0.4.1 added and aiofiles and ffmpy
no longer needed. No other package changed version.

**Run `uv sync --inexact` after pulling** (or rerun `deploy.sh` / `deploy.cmd`, which do the
same; `--inexact` keeps the PyG extensions the deploy scripts install outside the lock, see
`requirements.txt`). There is no fallback to Gradio 5: with 5.29.1 still installed the service
starts and the REST API works, but the log says
`Failed to mount Gradio dashboard: mount_gradio_app() got an unexpected keyword argument 'theme'`
and there is no `/ui`.

### What is switched off, and why

The dashboard is mounted with these settings, whatever the `GRADIO_*` environment variables say
(`src/webui/app.py`, `mount_options`):

| Setting | Why |
|---|---|
| `run_history=False` | Gradio 6 otherwise records each call's inputs and outputs in the browser and offers to save them to a Hugging Face bucket. Diagnosis inputs are a patient's phenotypes. The run-history page answers 404 |
| `enable_monitoring=False` | Left unset, `/ui/monitoring/summary` serves the queue's event analytics. It answers 403 |
| `mcp_server=False` | It would publish every dashboard event as an MCP tool. Its routes are absent |
| `ssr_mode=False` | Server-side rendering starts a Node.js process |
| `analytics_enabled=False` | On the dashboard itself, since 2026-10-12: no usage analytics and no version check |

**CORS** is tightened since M5-1, together with D8 (§4.13): **by default no cross-origin
browser access is granted; administrative operations are still checked by D8 in the backend.**
The parent app keeps its `CORSMiddleware`, now with an empty allow list (`src/api/main.py`), and
since Gradio 6.25 a mounted app adds no CORS headers of its own when the parent app has exactly
that class, so `/ui` follows the same policy. Until M5-1 the configuration was
`allow_origins=["*"]` with `allow_credentials=True`, which under Starlette 1.9 echoed every
request's `Origin` with `Access-Control-Allow-Credentials: true`: a page on any site, opened in a
browser holding HTTP credentials or a client certificate for the service, could read the
responses of `/api/*` and `/ui/gradio_api/*`. It no longer can.

### The route guard

Gradio 6.30 adds two routes to every mounted app that cannot be switched off upstream, and keeps
an upload route the dashboard does not use. The service refuses them before Gradio sees the
request (`src/api/middleware/gradio_routes.py`):

- **`/ui/gradio_api/file=<URL>`** (and the older `file/<URL>`). Since Gradio 6.20 (issue
  #13593, PR #13596) the server fetches an `http(s)://` target itself, ignoring `HTTPS_PROXY`,
  and streams it back on the dashboard's own address; for some URLs it serves a remote HTML page
  inline, as if it were the dashboard's own page. Every spelling of a URL scheme is refused:
  upper or mixed case, percent-encoded, `//host`.
- **`/ui/gradio_api/process_recording`**, which turns an uploaded screen recording into
  temporary files and may run ffmpeg. The screen recorder in Gradio's Settings panel therefore
  says "Failed to process recording. Saving original version." and saves the unprocessed
  recording in the browser.
- **`/ui/gradio_api/upload`** and its progress stream `upload_progress` (since M5-1). The
  dashboard has no upload control, and another site's page can post files there with no
  preflight: Gradio writes up to 1000 files a request into its temporary directory and never
  removes them.

A refusal is `403` with a one-line reason and `connection: close`. The request body is not
read, and the request log shows the refusal like any other response. Local downloads are not
affected: the Training Console's metrics CSV and the Diagnosis tab's CSV and Markdown exports
download as before, `HEAD` and `Range` requests included.

### What else changed in the dashboard

- **Training Console charts** look as before (the y axis starts at 0, each series keeps its
  colour). The Epoch Window now only filters rows; the x axis follows them.
- **Runtime Settings** refreshes the restart lock when its tab is opened: Gradio 6 does not run
  a timer in a tab that has not been opened yet. The backend refuses a restart during training
  either way.

### Platforms

Tested backends: Linux x86_64 (the development environment) and Linux ARM64 (the homelab,
checked by the reviewer). **A Windows backend has not been tested with Gradio 6.30.0** and stays
marked untested until one is available; Windows browsers are part of the browser checks.

## 4.11 One GPU job at a time: the GPU lock and the job runner (since M5, 2026-10-20)

Managed training and evaluation jobs exclude each other: the jobs the WebUI and the API start,
and `scripts/train_model.py` and `scripts/measure_scorer.py` run from a shell. A job that cannot
have the GPU is refused, naming who has it; nothing is queued.

### The GPU lock

One lock file per deployment, whatever the workspace:

| Settings | Lock file |
|---|---|
| `SHEPHERD_GPU_LOCK_FILE=/absolute/path` | that file; its directory must exist (nothing creates it) |
| otherwise `SHEPHERD_WORKSPACES_ROOT=/absolute/root` | `<root>/.shepherd-gpu.lock` |
| neither | **not configured**: the CLIs run without the lock and say so in their log; the service keeps only its own one-job rule |

A setting that is set but unusable (a relative path, a directory that does not exist, a path
under a directory the service account may not search) refuses every job rather than running it
unlocked, and the status shows why (`gpu.lock.problem`).

`train_model.py` and `measure_scorer.py` take the lock right after reading their command line,
before the device or any input is read, and hold it until they end. When it is busy they keep
trying for about a second (so a job that is just ending does not refuse the next one), then
refuse with exit 1 and write nothing:

```
measure_scorer: refused: the GPU lock /srv/shepherd/workspaces/.shepherd-gpu.lock is held by a training job (train_model.py, pid 41234 on gb10, since 2026-10-20T09:12:03+00:00, as it described itself). One managed GPU job runs at a time: wait for it to end, or stop it where it was started. Nothing was written.
```

`train_model.py` logs the same sentence after `train_model: refused:`. The WebUI and the API
check the lock before starting a job and refuse the start with the same holder
("Nothing was started."); the job's own process then takes the lock itself.

**Set the variable for the service and in the shell where you run the CLIs** (for example, load
the same environment file the systemd unit uses). A CLI that does not see it runs without the
lock and logs `GPU lock not configured (...), so this run does not exclude other managed GPU
jobs.`

The holder describes itself in `<lock file>.json` (job type, command, pid, host, start time)
while it holds the lock, for the message above. That file is descriptive only: the lock is the
`flock` on the lock file. One left behind by a killed job is harmless: the next holder removes
it as soon as it holds the lock, and a description whose process has exited is never named as the
holder (the refusal says instead that the lock is held by a process that inherited its descriptor,
or by a new holder that has not described itself). **Never delete the lock file while a job
runs**: a new file would be a second lock.

What the lock does and does not guarantee:

- **It is released however the holder ends**: a normal exit or an error releases it, and after
  SIGTERM or SIGKILL the kernel does, so there is no stale lock to clean up after a crash.
- **It is cooperative.** A GPU program that does not take it is not stopped by it. The inference
  service does not take it and may hold GPU memory; a test run that cannot get memory fails as a
  run, is reported, and registers nothing.
- **Inherited descriptors.** A program a job starts does not keep the lock (the descriptor is
  close-on-exec). A process the job forks without starting a new program — DataLoader workers
  under the `fork` start method, for example — shares it: a job that ends normally releases it
  for that process too, but when the job is killed (or exits without releasing), that process
  keeps the lock until it exits; the refusal then says the recorded holder has exited.
- **Keep the lock file on a local filesystem.**
- **Windows** uses `msvcrt.locking` on the file's first byte; a Windows backend is untested.

### The job runner

The service behind the Training Console and the training API now runs two kinds of job, each its
own script: training (`train_model.py`) and test runs (`measure_scorer.py` on a registered
cohort version, the same CLI an operator runs). One job at a time:

- **A job that is running or still stopping holds the GPU**, whatever its type. A training start
  while a stop is still waiting for its process is now refused (it used to be accepted, and could
  start a second process beside the one being stopped).
- **A stop reaches only its own job type**: a training stop never stops a test run, nor the
  reverse. It sends SIGINT, then SIGTERM after 30 s, then SIGKILL after 10 s more, and answers
  once the process has ended; `POST /api/v1/training/stop` no longer holds up the rest of the API
  and the dashboard while it waits. **A stop that arrives after the process has ended on its own**
  (while its result is still being read) is refused (`409`, "had already ended on its own ...;
  nothing was stopped") and changes nothing: a failure stays `failed`, with its message.
- **How a job ended:** training `completed`, `failed` or `stopped` (as before); a test run
  `completed`, `failed` or `cancelled`, `cancelled` only when the cancel is what ended it (a run
  that ended on its own as the cancel arrived keeps that ending). A failure caused by a signal
  says which (`Process ended by SIGKILL`, for the out-of-memory killer).
- **A test run's report is read from the disk once it ends.** Each test run writes its outcome
  file (`measure_scorer.py --outcome`, §4.9) into a directory made for that job alone, naming the
  report it published and the digest of those bytes; whether that report is kept and registered
  is then read from the report store and the ledger by the registration's own helper, with that
  digest, never inferred from the exit status (the status's `outcome` gives `kept`, `registered`,
  a `summary` with the reasons, and the run's own `run_outcome`). **No outcome file means no
  report was published**, unless a run directory for its checkpoint appeared in the store during
  the run (a run stopped between publishing its report and writing its outcome), which is then
  read the same way and said so. The job's directory is removed once the run has been read.
  After a cancel the status says what was left; after SIGTERM or SIGKILL it also names where
  staged files (`*.tmp`) may be left: the run directory and, for a run that was registering its
  report, the checkpoint's directory, where its ledger is.
- **Runtime Settings' Restart Backend** is refused while any job runs, training or a test run
  (its note now says "A GPU job (training or a test run) is in progress").
- `GET /api/v1/training/status` adds `job_id`, `end_time`, `exit_code` and `gpu`: the job holding
  the GPU, of either type, and the lock (`configured`, `path`, `source`, `held_by`, `problem`).
  A test run holding the GPU shows there, never as training.

Starting test runs from the WebUI, and their API routes, come with the Test tab later in M5. The
exit codes of both CLIs are unchanged: a busy lock is a refusal before anything was written.

---

## 4.12 Server settings and the server's lists (since M5-1, 2026-10-20)

M5's controls act on what the server lists, never on a path a request names (D7, D8). What the
server lists, and the deployment mode its operator controls depend on, come from **environment
variables of the service**. Set them where the service starts (the shell that runs
`launch_shepherd.sh`, or `Environment=` lines in the systemd unit) and restart it. Each is read
when it is used. No request, form or saved UI file can set one. A missing or refused setting
turns off what needs it and says why. **Nothing falls back to a default directory**, the
repository's `data/workspaces/` included.

| Variable | Value | Unset or refused |
|---|---|---|
| `SHEPHERD_DEPLOYMENT_MODE` | `M1`, exactly (`m1`, ` M1` or `M2` are other values) | read-only: every operator control is refused (D8, §4.13) |
| `SHEPHERD_WORKSPACES_ROOT` | absolute path of an existing directory; its subdirectories are the workspaces | no workspace is listed |
| `SHEPHERD_REGISTRY` | the cohort registry (§4.9), the directory the commands take as `--registry`: absolute, existing, outside the repository | no cohort version is listed |
| `SHEPHERD_INTAKE_DIR` | an existing directory outside the repository where you place `shepherd-cases/1` files to import; absolute | no import source is offered |
| `SHEPHERD_TRAINING_EVIDENCE_DIRS` | optional: R10 evidence directories, as `--training-evidence` takes them, separated by `:` (`;` on Windows); each absolute and existing | none; one bad entry refuses the whole setting |
| `SHEPHERD_ORDERING_ROLES` | optional: cohort roles that offer ordering by test score (D4), comma-separated | none; `institutional_acceptance` is ignored even when listed (§4.9 question 1) |

```bash
export SHEPHERD_DEPLOYMENT_MODE=M1
export SHEPHERD_WORKSPACES_ROOT=/home/you/SHEPHERD-Advanced/data/workspaces
export SHEPHERD_REGISTRY=/srv/shepherd/cohorts
export SHEPHERD_INTAKE_DIR=/srv/shepherd/intake
```

- **`M1`** is SPEC_4's single-workspace deployment, reached on the machine or through SSH
  forwarding, operated by the owner or a designated technician (§4.9 question 3). It is one of the
  conditions an operator control checks, not an authorisation. D8 checks each request it handles
  for the others. Do not set it behind a reverse proxy or on a machine other people can log in to.
- **The deployment mode is applied by D8** (§4.13), on every operator control at the moment it
  handles a request: unset or any other value, each one is refused and the deployment is
  read-only for them. Diagnosis, the status pages and the read-only lists work either way.
- **The workspaces root may be inside the repository**, as the usual `data/workspaces/` is, but
  only when you name it. A relative path, a missing directory and a file are refused.
- **The registry and the intake directory hold patient-level data.** They follow the
  registry's rule (§4.9): a relative path, a missing directory, and a path that resolves inside
  the repository (through a symlink too) are refused. `--registry` on the command line and
  `SHEPHERD_REGISTRY` are checked by the same function. Create both where only the accounts that
  run the service and the commands can read them, for example `mkdir -m 700`.
- **A directory the service account cannot reach** (a parent it may not enter) is refused as
  "cannot be checked", with the error, and its feature is off. Inside the registry, a `versions/`
  the service cannot read refuses the cohort list with that reason; a version, a `reports/` or a
  run directory it cannot read is listed with the reason, and the rest of the list stands.
- **Evidence directories**: an empty entry (a doubled or trailing separator) is skipped, never
  read as the working directory. A repeated one is kept once.

### What the server lists

Each item is named by a name and a digest, never by a path: a request names a workspace, a run
and a file, or a label, and the server finds that name in its own list again.

- **Workspaces** (`GET /api/v1/workspaces`): each subdirectory of the root, by name. A name
  starting with `.` is skipped. A subdirectory that is a symlink out of the root is listed but
  not offered. Each workspace shows:
  - its **dataset fingerprint**, the full SHA-256 of its `split_manifest.json`. The manifest
    binds the graph files, both sample files and the dataset record, so two different dataset
    builds always differ. A display may shorten it to 12 hex digits; comparisons use all 64;
  - its dataset record's binding state and origin, kept apart (§4.4);
  - its **build time** from `workspace_build.json` (below), or "not recorded" with the reason;
  - its sources' declared releases, from the dataset record (none for a synthetic workspace);
  - its train and val sample counts;
  - whether it is **trainable**, and why not: no readable `split_manifest.json`, a dataset record
    that is not verified, or a file the manifest binds is missing or cannot be checked. The list
    hashes no graph file. A training run checks every file on the bytes it reads, and may still
    refuse.
- **A workspace's checkpoints**: the run layout's checkpoints of one listed workspace, every
  architecture's or one's, latest run first. Each file is read once, weights only, and shows:
  - the SHA-256 of the bytes read;
  - the Trainer's validation figure: the trainer's top-20 selection metric, never a test result,
    and none when the file records none;
  - the dataset fingerprint the checkpoint records, and whether it is this workspace's now;
  - for a file that cannot be read, why (a link whose target is gone included). No file is
    dropped;
  - for a file that resolves outside the workspace (a run or a checkpoint that is a symlink out of
    it), that reason only: it is never opened, so it shows no digest, size or record.

  Each item names the workspace as the list does: for `current -> builds/2026-10-20`, it says
  `current`. Listing opens every checkpoint, which SPEC_4 classes as checkpoint inspection (C3),
  and D8 offers that only where the operator controls are: it is served as
  `GET /api/v1/workspaces/{name}/checkpoints` (`?architecture=gat`, say, for one architecture)
  behind D8's check (§4.13), and a refused request opens nothing.
- **The existing checkpoint list** (`GET /api/v1/training/checkpoints` and the Training
  Console's resume list) reads each file the same way. Each entry gains `sha256`, `validation`,
  `dataset_fingerprint` and `metadata_problem`, all from the one read. Before, it loaded each
  file without keeping its digest, and dropped a load error without a word. A file that is gone
  by the time it is read (a link whose target was removed) has no size or time to show and is
  left out of this list; it no longer fails the whole list. Since M5-1 D8 checks it like the
  controls (§4.13), the route and the Training Console's run list alike.
- **Cohort versions** (`GET /api/v1/cohorts`): each version in the registry, by label and the
  SHA-256 of its `cohort.json`, with its role and whether that role offers ordering. It also shows
  its counts, the graph and dataset record it is bound to, and when it was imported. The list
  never shows a case identifier or the exclusion list: D7 keeps those in the registry, on the
  machine. An interrupted import is listed with its reason. Each version's report store is
  listed by run id, and each run says whether its report is kept and whether it is registered
  (`kept`, `registered`, each with its reason, the report's digest and the record's sequence),
  from the registration's own reading of the store and of the ledger beside the checkpoint the
  report records it measured. `registered` is `registered` when that ledger names the report,
  and otherwise `cannot_be_checked`, never `not_registered`: a registration filed beside
  another copy of the same weights (`record_evaluation --checkpoint` takes any file with the
  measured bytes) is not looked for, and the reason names the ledger that was. A report that
  names its checkpoint by a relative path, or that cannot be read, has no ledger to look in:
  `registered` is then `cannot_be_checked` too, saying why. A checkpoint moved since the run is
  looked for where the run read it.

### The build time: `workspace_build.json`

Since M5-1, every build through the workspace writer (`scripts/build_knowledge_graph.py
--generate-samples`, `scripts/setup_demo.py`, the probe) writes `workspace_build.json` beside
`split_manifest.json`, after it:

```json
{
  "built_at": "2026-10-20T08:15:00Z",
  "code_revision": {"state": "recorded", "revision": "84371dd...", "dirty": false, "modified_tracked": 0, "untracked": 0},
  "schema_version": 1,
  "split_manifest_sha256": "...",
  "what_this_is": "when this workspace was built and from which code revision; descriptive, not bound by split_manifest.json"
}
```

- `built_at` is UTC. `code_revision` is the record training stores as a checkpoint's producer: the
  commit and whether the checkout was dirty, with no path.
- **Descriptive, not bound by the manifest.** The manifest does not name it, so two
  byte-identical builds keep one dataset fingerprint, whatever their build times.
- **It names the manifest it describes.** A record beside another manifest reads "not recorded",
  never another build's time. That happens after a rebuild that stopped after writing its
  manifest, or a rebuild by a revision older than M5-1, which does not write this file.
- **A workspace built before M5-1 has none.** Its build time reads "not recorded", shown with its
  sources' declared releases; never a file modification time. It does not need rebuilding for
  that. The reason says "built by a revision before M5-1, or its build stopped before writing
  one": a first build stopped between the manifest and this file looks the same.
- **A graph-only build** (no `--generate-samples`) writes neither a manifest nor this file. Its
  build time reads "not recorded" because there is no `split_manifest.json` to describe.

---

## 4.13 Operator controls, the bind and CORS (since M5-1, 2026-10-20)

M5 runs under SPEC_4's deployment mode **M1**: the owner or a designated technician operates the
service on the machine, or from elsewhere through SSH forwarding (§4.9 question 3). Three things
changed together, and a deployment may need to act on each:

- the operator controls are refused unless D8 allows the request;
- the service binds `127.0.0.1` by default, not `0.0.0.0`;
- no other website's page can read the service's responses (CORS).

### What an operator sets

Set `SHEPHERD_DEPLOYMENT_MODE=M1` where the service starts (§4.12; in the systemd unit, the
commented `Environment=` line) and restart it. Unset or any other value, every operator control
is refused and the deployment is read-only for them. Diagnosis, the lookups, the status pages and
the read-only lists keep working.

### Which controls are checked

| Where | Controls |
|---|---|
| API | `POST /api/v1/pipeline/reload`, `POST /api/v1/pipeline/config`, `POST /api/v1/training/start`, `POST /api/v1/training/stop`, `GET /api/v1/training/checkpoints`, `GET /api/v1/workspaces/{name}/checkpoints` |
| Dashboard | Diagnosis: Reload Pipeline and Save Config. ⚙ Settings (Runtime Settings): Restart Backend and Apply Settings. Research › Training (the Training Console): Start, Stop (the confirmed stop), Continue Training, Check checkpoint, the run list (Refresh, and changing the architecture) and Save Path |

The two checkpoint lists, and Check checkpoint, open checkpoint files, which SPEC_4 classes as
checkpoint inspection (C3), so they are checked like the controls. The Test tab and Model
Management that M5's UI step adds are to use the same check. Everything else stays open:
diagnosis, the HPO and disease lookups, the pipeline status and the saved UI paths, the training
status and metrics, the workspace and cohort lists, the recorded results and the system pages.

### When a request is allowed

Each control checks the request it is handling, at the moment it handles it, and allows it only
when all of these hold:

1. `SHEPHERD_DEPLOYMENT_MODE` is exactly `M1`;
2. the request arrived on a loopback address of this machine: the server's own socket decides,
   never an address the client states;
3. it carries no header a proxy adds (`Forwarded`, `X-Forwarded-For`, `X-Forwarded-Host`,
   `X-Forwarded-Proto`, `X-Real-IP`, `X-Original-Forwarded-For`, `CF-Connecting-IP`,
   `True-Client-IP`): M1 has no reverse proxy;
4. it was addressed to `localhost`, `127.0.0.1` or `[::1]`, with any port (its `Host` header),
   which guards against DNS rebinding;
5. if it carries an `Origin`, that is the dashboard's own address (`http://` or `https://`, the
   same host and port). A browser sends the page's origin with every `POST` and with every
   cross-origin request a script makes;
6. if it carries `Sec-Fetch-Site`, that says `same-origin` (the dashboard's own page) or `none`
   (a request you started outside any web page: the address bar, a bookmark, a link opened from
   another program). A browser sends **no** `Origin` on a `GET` that
   another site's page makes with an image, a script or a link, and current browsers (Chrome and
   Edge 76, Firefox 90, Safari 16.4 and later) say here where it came from: `cross-site`, or
   `same-site` for another port of the same host. So a page on another website cannot trigger a
   control from a current browser, the two checkpoint lists included.

A request whose arrival cannot be read is refused. The checks are in the backend: a greyed-out
button is only a display, and a direct call, to the API or to a dashboard event through Gradio's
HTTP API (`/ui/gradio_api/...`, as `gradio_client` makes one), meets the same check. The
dashboard's controls call the services in process; they never call the API over HTTP. The
dashboard's Reload still runs where the API's reload runs, so it waits behind a running
diagnosis as before; and it no longer gives up after 180 s, so a long reload is no longer shown
as failed while it is still being published.

**A refusal** is `403` from the API, with an `error` that names the condition that failed in
words to act on and ends "Nothing was done."; it never repeats a header's value. The dashboard
shows the same sentence where the control shows its result (the run list, which cannot hold a
sentence, is emptied and the sentence appears as a notice). For example:

```
Refused: This request did not arrive on a loopback address of this machine. The operator controls work only on the machine itself, or through an SSH forward to localhost (for example ssh -L 8264:localhost:8264 <server>, then http://localhost:8264/ui). Nothing was done.
```

**Is it enabled for this browser?** `GET /api/v1/system/operator-access` answers for the request
that asks: `enabled`, the `condition` that decided it and its `reason`. It is read-only and open;
showing it is not the control.

**A reload that D8 refuses changes nothing.** Before, any caller's reload request, refused or
not, turned every later diagnosis from the demo answer into "configured but not served" (503).

**Not claimed.** Loopback is not authorisation: whoever reaches the machine's loopback, through
an SSH forward of their own too, passes. M1 therefore means one verified workspace deployment
operated by its owner or a designated technician. A reverse proxy on the same host that adds no
forwarding header is excluded by this guide, not detected: do not put one in front of an M1
service. An old browser that sends neither `Origin` nor `Sec-Fetch-Site` on such a `GET` is not
caught by condition 6: operate the dashboard from a current browser.

### Operating from another machine: SSH forwarding

Forward a local port to the service's **localhost**, not to its network address:

```bash
ssh -N -L 8264:localhost:8264 you@gb10
# then open http://localhost:8264/ui on your own machine
```

The forwarded connection arrives on the server's loopback addressed to `localhost:8264`, so the
controls work. `ssh -L 8264:<server LAN address>:8264` would arrive on the LAN address and be
refused, and with the new bind it finds nothing listening. Another local port works too
(`-L 9000:localhost:8264`, then `http://localhost:9000/ui`).

### The bind: `127.0.0.1` by default

The launcher (`launch_shepherd.sh`, `shep_launch.py`), the systemd unit and `python -m
src.api.main` bind `127.0.0.1` unless told otherwise, and the launcher prints what it binds:

```
[SHEPHERD] Listening on          : 127.0.0.1:8264 (loopback: this machine only)
```

**A bind to any other address is refused and nothing is started** (the launcher exits 2, before
it installs anything) unless `SHEPHERD_ACCEPT_UNAUTHENTICATED_NETWORK_BIND=1` is set, because the
service has no authentication. That flag is **risk acceptance, not a control**: it authenticates
nobody, and it never lifts D8, so on a network bind the operator controls stay refused for every
request that does not arrive on loopback, while diagnosis and the read-only pages are open to
whoever reaches the address. The launcher prints a `WARNING:` line that says so. The launcher
always starts uvicorn with `--no-proxy-headers`, last on its command line, so the client address
uvicorn reports is the connection's.

The launcher reads what uvicorn will listen on as uvicorn does: a socket it is handed (`--fd`,
as a systemd socket unit passes one) before a unix socket (`--uds`) before the last `--host`,
each from the command line or else from `UVICORN_FD` / `UVICORN_UDS`. An inherited socket's
address is not read, so it counts as a network bind and needs the flag. A unix socket reaches
this machine only and needs none; the operator controls are refused through it (D8 sees no
loopback address there). For either, the launcher prints the socket instead of a URL and opens
no browser.

The launcher and `python -m src.api.main` pass the host they bind to the application as
`SHEPHERD_BIND_HOST`, and application startup applies the same rule to it before serving (a
refused one ends startup, and uvicorn exits). They set it every time; it is not yours to set.

Not covered: a uvicorn started by hand (`uvicorn src.api.main:app --host 0.0.0.0`), which the
application cannot see and which passes no host to startup. Use the launcher.

**Migrating a deployment that relied on `0.0.0.0`.** Until M5-1 the launcher and the unit bound
every interface. After this update:

- whoever opened `http://<server LAN address>:8264/ui` gets no connection. Give them an SSH
  forward (above): that is the M1 way;
- if the LAN must keep reaching diagnosis, set `--host 0.0.0.0` (in the unit's `ExecStart`, or
  after `--` on the launcher's command line) **and** `SHEPHERD_ACCEPT_UNAUTHENTICATED_NETWORK_BIND=1`
  in the same environment. Without the flag the launcher refuses to start (exit status 2), and under
  systemd the unit stops without restarting (`RestartPreventExitStatus=2`;
  `journalctl --user -u shepherd` shows the reason). The
  operator controls then work only on the machine itself or through an SSH forward;
- a script that reloads the pipeline over HTTP needs the service in mode M1 and must call
  `http://127.0.0.1:<port>`. `scripts/measure_served_pipeline.py` starts its own service with
  `SHEPHERD_DEPLOYMENT_MODE=M1` for that reason.

### CORS: no cross-origin browser access

**By default no cross-origin browser access is granted; administrative operations are still
checked by D8 in the backend.** No response carries `Access-Control-Allow-Origin` for another
origin, on `/api` or on the dashboard, whatever address the browser used, and every cross-origin
preflight is answered 400. Same-origin use (the dashboard itself, `curl`, scripts,
`gradio_client`) is unaffected.

CORS only decides what a page may read. A cross-origin request that needs no preflight still
reaches the server: that is why the operator controls are checked by D8, not by CORS. For the
same reason Gradio's upload route is refused in front of the dashboard, before the request body
is read (§4.10, the route guard): the dashboard has no upload control (import sources come from
`SHEPHERD_INTAKE_DIR`), and another site's page could otherwise write up to 1000 files a request
into Gradio's temporary directory. Gradio's limit of 1 MB per file stays as a second line of
defence; it bounds each file, not their number.

## 4.14 Finding your way in the dashboard (since M5-2, 2026-10-20)

The dashboard at `/ui` is laid out in two levels, with two status chips in its header and one
help control for every field. Nothing an operator sets changed; what moved is where things are.

### Navigation

| Where | What is there |
|---|---|
| **Diagnosis (clinical)** | The Diagnosis tab, as before. It is the first screen |
| **Research (pre-clinical)** › Prepare › **Datasets** | The server's workspaces, read-only (below) |
| **Research (pre-clinical)** › Prepare › **Training** | The Training Console, as before. It is the research page shown first |
| **Research (pre-clinical)** › Evaluate › **Test**, **Models** | "being built in M5" for now |
| **⚙ Settings**, at the right of the header | Runtime Settings, as before. For a read-only browser, Apply Settings and Restart Backend are greyed out, with the reason written below them |

- The selected stage (Diagnosis or Research) is filled and bold; the selected research page is
  underlined and bold; Settings, when selected, has a light fill and an underline. Prepare and
  Evaluate are group labels, not buttons.
- No tab is ever hidden in a menu. In a narrow window, or at a large zoom, the chips and
  Settings move onto a second header line, and the research pages wrap.
- The browser tab's title says where you are, for example `Training · Research —
  SHEPHERD-Advanced`. Reloading the page opens Diagnosis again: the address does not keep the
  location.
- In a window at least 1400 × 760 CSS pixels the page itself does not scroll; a region too long
  for its space scrolls inside its own panel. Narrower, or in portrait, the regions stack and
  the page scrolls. The theme's smallest text is 14 px.
- The dashboard is always drawn in its light colours, whatever dark mode the system or the
  browser asks for (its colours have no dark version, and Gradio's dark mode left its text
  unreadable on the page); the address gains `?__theme=light`. Gradio's own footer (its API
  page, "Built with Gradio" and its own "Settings") is not shown.
- A research page is read again each time it is shown: when you select it, and when you return
  to Research while it is the page shown. The other research pages are not read then.

### The two chips in the header

Both are read again for your own browser when the page loads, every 5 seconds, and after each
operator control you use. They are displays: every control still checks its own request in the
backend (§4.13).

**Operator controls** says what D8 answers for this browser's requests:

| Chip | Meaning |
|---|---|
| `🔒 Operator controls: on (this machine)` | D8 allows this browser's requests |
| `Read-only: deployment mode not set to M1` | `SHEPHERD_DEPLOYMENT_MODE` is not exactly `M1` |
| `Read-only: request not local` | The request did not arrive on this machine's loopback: use an SSH forward to `localhost` |
| `Read-only: forwarded by a proxy` | The request carries a header a proxy adds |
| `Read-only: not addressed to localhost` | The browser used another host name than `localhost`, `127.0.0.1` or `[::1]` |
| `Read-only: sent from another origin` / `sent from another site` | The request came from another page |
| `Read-only: request not readable` | The server could not read how the request arrived |

Its ⓘ lists every operator control, how to reach them (the SSH forward of §4.13) and, when
read-only, the full reason D8 gives. It never repeats a header's value.

**GPU jobs** says which managed job holds the one GPU slot (§4.11):

| Chip | Meaning |
|---|---|
| `GPU jobs: none` | No managed training or evaluation job is running. It never says "free": the lock is cooperative, and the resident inference service is outside it and may still hold GPU memory |
| `GPU job: training (job runner)`, `GPU job: test run (job runner)` | A job this service started; `· stopping` while it stops |
| `GPU job: train_model.py (CLI, pid 41822, since 09:58 UTC)` | A CLI holds the GPU lock, as its holder record describes it |
| `GPU job: lock held, holder not described` | The lock is held, but no running process is described as its holder |
| `GPU jobs: none in the job runner` | No GPU lock is configured (neither `SHEPHERD_GPU_LOCK_FILE` nor `SHEPHERD_WORKSPACES_ROOT`), so a CLI's job would not be seen |
| `GPU jobs: not read` (grey) | The state could not be read. When the lock setting cannot be used, the ⓘ says so, naming the path only to a browser D8 allows |

### Help: the ⓘ buttons

The explanations that were printed under each field of Diagnosis, the Training Console and
Runtime Settings are now behind a ⓘ after the field's name. Point at it, reach it with Tab, or
click it: the help appears in a bubble you can move the pointer into. A click keeps it open; Esc
or a click elsewhere closes it. A click on the field's name still puts the cursor in the field.
The bubble opens below or above the ⓘ, wherever there is room, and is never taller than the
window: a long text scrolls inside it. What must stay visible is still printed: a refusal's
reason, a job's state, the synthetic marker, an unverified source, and the Training Console's
AMP refusal for HGT ("Disabled: HGT requires full float32 …").

### Research › Prepare › Datasets

A read-only list of the workspaces the server lists under `SHEPHERD_WORKSPACES_ROOT` (§4.12):
the same list Training, Test and Models choose from. Each row shows the name; when it was built
(UTC, from `workspace_build.json`), or "build time not recorded" with its sources' declared
releases; the **dataset fingerprint**, the first 12 hex digits of the SHA-256 of its
`split_manifest.json` (Copy gives all 64, and more digits are shown when two workspaces share
the first 12); the origin (`synthetic` in amber); whether the dataset record is verified; the
train and validation sample counts. The markers:

- **serving reads this workspace**: the pipeline is serving from it;
- **same dataset as served**: another workspace with the same full fingerprint;
- **in your Training run bar**: this browser tab's choice for Training;
- **not trainable: <why>**: listed, but training would refuse it, with the reason: `no split
  manifest` when the workspace has no `split_manifest.json`, `no readable split manifest` when
  it has one that cannot be read, or the list's own reason (for example a workspace that is a
  link to a directory outside the root: "… resolves outside the workspaces root").

A long name or source list wraps in its cell, so the "In use" column stays in view. Selecting a
row shows its details: the full fingerprint with Copy, what the manifest binds (short digests),
the seeds and split counts, the sources and the build record, the workspace's path relative to
the server's workspace root, and how it compares with the other listed workspaces: "Same dataset
as …" (the same full fingerprint) or "Same graph build as … (kg, tensors and record identical),
another cut: the allocation seed differs …". No absolute server path is shown, to any browser:
a path under the root is shown relative to it, any other one reads "a path outside the
workspaces root", and a refused `SHEPHERD_WORKSPACES_ROOT` is said without its path. The page
reads the manifests when it is opened, not on each selection.

**Use for training ›** opens Research › Training with that workspace in this browser tab's Data
Directory (Data Paths is opened to show it). Nothing changes on the server; Start is still the
control D8 checks. A workspace that is not trainable cannot be used. A workspace inside the
repository is shown relative to it; one outside it would be an absolute server path, which only a
browser D8 allows is given (the one that can Start): a read-only browser sees a note instead.
Until the Training run bar is built, the Data Directory is where the choice lives: editing it by
hand, or Reset Default, removes the "in your Training run bar" marker, and choosing the
workspace again fills the field again.

Nothing on this page is an operator control: it reads the workspaces' small JSON files, never a
checkpoint, so a read-only browser sees the same page. With `SHEPHERD_WORKSPACES_ROOT` unset,
the page says so. Datasets are still built on the server with the CLI
(`scripts/build_knowledge_graph.py`, `scripts/preprocess_data.py`).

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
- [ ] ⚠️ Dashboard on Gradio 6.30.0 (`uv sync --inexact` after pulling): a Windows backend is not tested yet (§4.10)

### ARM + Blackwell (DGX Spark)

- [ ] ✅ Architecture confirmed as aarch64
- [ ] ✅ CUDA 13.0 available
- [ ] ✅ PyTorch ARM build working
- [ ] ✅ PyTorch Geometric available (wheel or build)
- [ ] ⚠️ Attention backend downgraded (xformers or SDPA)
- [ ] ✅ Vector index: cuVS (GPU) or Voyager (CPU fallback)
- [ ] ✅ Unified memory correctly identified
- [ ] ✅ Model trains (check the 128GB advantage)
- [ ] ✅ Dashboard on Gradio 6.30.0 (`uv sync --inexact` after pulling; `/ui` loads, §4.10)

---

**Version**: v3.2
**Maintenance**: kept in step with `deployment-guide.md`; update both together.
