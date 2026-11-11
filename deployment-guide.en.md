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

> **The formal deployment** installs `torch_geometric` from `uv.lock` at the verified version and
> its native extensions per platform (§4.15). The strategies below are manual fallbacks from the
> planning phase: keep `torch_geometric` at the version `pyproject.toml` pins (`uv sync --inexact`
> restores it) and take the extensions from the index for the installed torch.

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
2. Train a base: `python scripts/train_model.py --data-dir <workspace> --stage base` (optionally
   `--config <settings.yaml>`), in a shell where the GPU lock is configured: it refuses without
   one (§4.11). Then its task models on the dashboard's Research › Prepare › Fine-tune, or
   `--stage finetune --task <task> --finetune-from <base run>/checkpoints/last.pt`
3. Launch: `./launch_shepherd.sh`

There is no shortest-path step: a task model never reads `shortest_paths.pt`, and neither does
the GB10 acceptance probe.

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
# torch and torch_geometric back to the lock's versions (§4.15)
uv sync --inexact --reinstall-package torch --reinstall-package torch-geometric
# the native extensions from the index for the installed torch
TVER="$(.venv/bin/python -c 'import torch;print(torch.__version__)')"
uv pip install --reinstall pyg-lib torch-scatter torch-sparse \
    -f "https://data.pyg.org/whl/torch-${TVER}.html"
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

A workspace built by an earlier version may still hold `shortest_paths.pt`, a precomputed
distance table, and `shortest_paths.meta.json`, its sidecar. **This build neither writes,
reads, refuses nor deletes them.** Serving, training, task measurement and the inventory
(`scripts/audit_checkpoint_family.py --inventory`) are the same with or without them; the
inventory does not list the table as a checkpoint. The GB10 acceptance
(`scripts/probe_deployment.py --m2-acceptance`) reads neither file either: a pair in the
workspace is covered by its check that the workspace is left unchanged (R8). Keep the two
files or archive them, as you like.

No task model needs the table. A disease model ranks every disease of its graph view, and a
gene model computes the exact hop distances it uses from the view in process
(`src/kg/graph_view.py`, `exact_distances`), so building a workspace has no shortest-path
step.

The table's producer, `scripts/compute_shortest_paths.py`, and the reachability audit behind
evidence M5 (BACKLOG item 10), `scripts/audit_sp_reachability.py`, were removed in K8-5. The
last commit that has them is `e02c0f7` (`e02c0f7562b166bf6e69677e51b8bc8ee2f0cb1a`), where
their tests passed before the removal; to rebuild or audit an old table, run them from a
checkout of that commit. `SHEPHERD_SP_HOP_BOUND` is not read; if the service environment
still sets it, it has no effect.

### Checking the result

`GET /api/v1/pipeline/status` reports nothing about the table since K8-4: its
`sp_ready`, `sp_max_hops`, `sp_hop_bound_source`, `sp_kg_binding` and
`eta_configured` fields are removed with the mixture. What it reports is the served
model's own:

| Field | Expected |
|---|---|
| `scoring_mode` | the served model's task: `disease_task`, `gene_task` or `patients_like_me`; a table never changes it |
| `served_identity` | What the loaded pipeline serves (§4.7): `shortest_paths.state` `computed` for a gene model (exact hop distances from its graph view, no table), `absent` for the other two; `settings.eta` the gene model's η, null for the other two |

---

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
- the Diagnosis page: a `SYNTHETIC WORKSPACE: not a clinical result.` banner
  above the results, the amber `synthetic data` chip on the "Model in use" card
  (the banner again in its Details), and `verified · synthetic` on the dataset
  record's row of the result's identity ("This result"; since M5-2, §4.14);
- the downloadable report: the same banner in its "What produced this result"
  section, and its file name, `diagnosis_SYNTHETIC_<patient>_<session>_report.md`;
- the CSV export: `served_dataset_origin` `synthetic` on every row, and its file
  name, `diagnosis_SYNTHETIC_<patient>_<session>.csv`.

What the record says it does not establish (`incomplete_by_design`) is
displayed, never refused. Nothing is served without a model (since K7): every
served pipeline reads its workspace, and with it this record (§4.7, "No model,
no answer"). The pipeline reload lists `kg.provenance.json` among the required
files.

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

A relative path given to the probe (`probe_report.json` here, or an acceptance's
`--real-workspace data/workspaces/<workspace>`) is taken from the directory you start it in and
made absolute once, before anything runs; one that does not exist from there is refused at once,
not after the training (since Batch 1, BACKLOG 49).

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

The split manifest, `kg.provenance.json`, the evaluation ledger, the UI config and
the ontology cache are all written to a temporary file and renamed into place, so
a reader never sees a half-written one.
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
the checkpoint, the graph, the workspace manifest, the shortest-path term and
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
- **It is never `null`** (since K7): every answer comes from a model. With no
  pipeline served — nothing configured, a graph named without a model, or a
  build that failed — `/api/v1/diagnose` answers 503 with the reason, and so
  does `/ready`; `/health` stays 200 so the service starts and a model can be
  loaded. The illustrative "demo answer" the service used to give when nothing
  was configured (`demo/v1`, `Pipeline not fully initialized - using mock
  data`) is removed.

The service's own version (`1.0.0` in `/health`, `/` and the Swagger title) is
the API's version, not the identity of what is served.

### The diagnose response is one kind of the result union (Batch 4; tasks since K3–K4)

`POST /api/v1/diagnose` answers one kind of result, named by `result_kind`, with
`result_schema_version` 3 (`docs/working/results-review/SPEC_1_RESULTS_REVIEW.md`
§2.1). The kind follows the model served; each kind carries its own values and
refuses another kind's. The format version changes with the response's shape: 3
since K8-4r marks K8-4's change, in which the served identity's `shortest_paths`
kept only its `state` (`build_id` and `max_hops` are gone). A strict client of
version 2 refuses a version-3 body at that field, beside the keys that differ;
the Diagnosis page's Markdown report names the version of the response it keeps.

| `result_kind` | Answered when | Each candidate's values | On the result |
|---|---|---|---|
| `disease_task/v1` | a model finetuned for the disease task is served | `score`, the disease head's score (in [−4, 0], uncalibrated) over every disease of the model's graph view; the evidence fields explain a row and change neither the score nor the order | `tie_policy` |
| `gene_task/v1` | a gene task model is served; it ranks the request's `candidate_genes` | `score` = η × `embsim` + (1 − η) × `sp_similarity`; `embsim`; `sp_similarity`; `mean_distance`, the exact mean hop distance computed from the model's graph view | `sp_bounds`, `tie_policy` (η is the served identity's `settings.eta`) |
| `patients_like_me/v1` | a patients-like-me model is served, with a reference cohort version | `similarity` (in [−4, 0], uncalibrated); `case_id`, the recorded `phenotypes`, diagnosis and `causal_genes` | `reference_cohort`, `tie_policy` |

The path-reasoning answer (`path_reasoning/v1`, a graph served with no model) and the demo
answer (`demo/v1`) are removed since K7: with no model there is no answer (503, above). The
mixture's answer (`served_mixture/v1`: `served_score`, `model_term`, `sp_term`, `eta`,
`candidates_reached`), which a checkpoint without a model spec was served with, is removed since
K8-4: such a file is refused at load (below, "No model, no answer").

- **Removed, with no alias:** `confidence_score`, `gnn_score`, `sp_score`,
  `reasoning_score` and `confidence_label` (now `evidence_label`, inside
  `evidence_package` too). A client that read them reads the kind's own fields
  instead.
- **Refusals.** A model term the service cannot compute (a phenotype the graph
  knows with no row in its node mapping) or a score that is not finite refuses
  the result: HTTP 500,
  its `error` starting with the code (`MODEL_TERM_UNAVAILABLE: …`,
  `NON_FINITE_SCORE: …`). On a verified workspace these states are not
  expected. An input with no phenotype in the graph is refused as before
  (`No valid phenotypes after validation`). An unknown id is always listed in
  `phenotype_input.unknown`, and stated as a warning (`Unknown phenotype: …`)
  when the request is answered; when no id is known, the warning is the
  refusal.
- **`phenotype_input`** (received, unknown, repeats removed, used) is in the
  response; null when a request was refused before its phenotypes were mapped.

### Its parts and states

```json
"served_identity": {
  "scoring_mode": "disease_task",
  "checkpoint": {"state": "verified", "sha256": "<SHA-256 of the checkpoint read>"},
  "graph": {"state": "verified", "kg_sha256": "<SHA-256 of kg.json>",
            "split_manifest_sha256": "<SHA-256 of split_manifest.json>"},
  "shortest_paths": {"state": "absent"},
  "dataset_record": {"state": "verified", "origin": "files",
                     "sha256": "<SHA-256 of kg.provenance.json>"},
  "settings": {"device": "cuda:0", "device_requested": "auto", "eta": null,
               "max_path_length": 4, "path_length_penalty": 0.9,
               "max_paths_per_candidate": 10,
               "use_direct_path_optimization": true,
               "phenotype_normalisation_version": 1}
}
```

**A served model is always read from a checkpoint over a workspace verified with its dataset
record** (since K7), so those three parts have one state each; only the shortest-path part
varies, and it follows the scoring mode.

| Part | State | Meaning | Values |
|---|---|---|---|
| `checkpoint` | `verified` | The model was built from one read of a checkpoint whose record matched the served graph (contract M2.4) | `sha256`: that read's digest |
| `graph` | `verified` | `kg.json` and `split_manifest.json` were each read once, and the graph's digest matched the manifest | `kg_sha256`, `split_manifest_sha256` |
| `shortest_paths` | `computed` | The gene task's exact hop distances, computed in process from the model's graph view (no table) | none |
| | `absent` | No shortest-path term contributes to the score (the disease and patients-like-me tasks) | none |
| `dataset_record` | `verified` | The record the manifest binds, naming the served graph (§4.4) | `origin` (`files` or `synthetic`); `sha256` of `kg.provenance.json` |
| `scoring_mode` | — | `disease_task`, `gene_task` or `patients_like_me`: the served model's task | |
| `settings` | — | The serving settings that change scores, as in effect | below |

The checkpoint states `supplied_in_memory` and `absent`, the graph states `unverified` and
`in_memory`, and an `absent` dataset record are removed since K7, with the in-memory seams and
the path-reasoning answer that produced them. The shortest-path state `verified` with its
`build_id` and `max_hops`, and the scoring modes `gnn_plus_shortest_path` and `gnn_only`, are
removed since K8-4 with the mixture that produced them.

`settings`:

- `device`: where the model ran — torch's name for the device its parameters
  are on (`cuda:0`, `cpu`).
- `device_requested`: what was asked for — `auto` when nothing was named,
  `cuda`, `cuda:N` or `cpu` (§4.8).
- `eta`: the mixing weight in effect — the gene task's η (from its model spec);
  null for the disease and patients-like-me tasks, which mix nothing.
- `max_path_length`, `path_length_penalty`, `max_paths_per_candidate`,
  `use_direct_path_optimization`: path discovery and path scoring — the
  evidence paths shown for a disease row. The path aggregation setting (`aggregation_method`) is gone since
  K7: nothing ranks by an aggregate of path scores.
- `phenotype_normalisation_version`: the rule applied to the phenotype list
  before scoring (currently `1`).

The parts cannot contradict each other: each scoring mode has exactly one
shortest-path state (`computed` for `gene_task`, `absent` otherwise), and an
identity that says otherwise is refused when it is built.

### The same field in the pipeline status

`GET /api/v1/pipeline/status` carries `served_identity` for the pipeline loaded
now, beside its other fields (`scoring_mode` and `eta_effective`, equal to the
identity's scoring mode and `settings.eta`; `dataset_record`; `checkpoint_meta`
with its `device`, …). It is null when nothing is loaded. The table and mixture
fields (`sp_ready`, `eta_configured`, `sp_max_hops`, `sp_hop_bound_source`,
`sp_kg_binding`) and `fingerprint_warnings` are removed since K8-4. **The status
describes what is loaded now, and a reload replaces it.** To know what produced a
given result, read that result's own `served_identity`, or its report or CSV, not
the status.

### How to read it

- **A normal deployment** reads: checkpoint `verified`, graph `verified`,
  dataset record `verified` with origin `files`, the task's scoring mode
  (`disease_task`, `gene_task` or `patients_like_me`) with its shortest-path
  state, device `cuda:0`, requested `auto` or `cuda`.
- **Compare the digests with the files you deployed**: the SHA-256 of the
  checkpoint, `kg.json`, `split_manifest.json` and `kg.provenance.json` equal
  the identity's digests.
- **Anything else is stated, not hidden**: origin `synthetic` is a generated
  workspace, not clinical data (§4.4); device `cpu` was asked for explicitly
  (§4.8).
- **What it does not establish** is displayed under the identity, as the
  contract's claim boundary: "Checked: these files are the ones their records
  name, and the records agree on the relations marked ✓. Not checked: who
  changed them, or anything a person able to edit both files and records could
  alter."

### No model, no answer

Since K7 a graph is never served without a model (the path-reasoning fallback
is removed, closing the open policy question B-2), and nothing is served from
a model or graph handed over in memory. A service started with
`SHEPHERD_KG_PATH` and no `SHEPHERD_CHECKPOINT_PATH`, or with nothing
configured, publishes no pipeline: `/api/v1/diagnose` and `/ready` answer 503
with the reason, `/health` answers 200, and the Diagnosis page offers "Load a
model…". A reload that fails keeps the model in use and its ready state.

**A checkpoint without a model spec (a ShepherdGNN history file) is refused at
load** (since K8-4), at startup and on a reload with or without a task, before
any model is built: "written before the model spec: not readable by this build;
start a new run" (since K8-7b, 2026-11-08, the one phrase every refusal of such a file gives). A
file from before the provenance contract, which records no training inputs either, is refused
first for that ("records no training_input_digests"). A refused startup publishes nothing (503,
above); a refused reload leaves the model in use serving. Such files stay on disk and listed,
with their SHA-256, records and usage history, as a record this build cannot read ("Record not
readable", with that phrase; no longer "History · never run" since K8-7b); the mixture that used
to serve them is removed.

**A checkpoint whose record names one input two ways is refused the same way**
(since K8-4r), for example a `data_roles.kg` that differs from its
`training_input_digests.kg`: "the record contradicts itself: data_roles.kg …
differs from training_input_digests.kg …". Serving, measurement and training
(finetune, resume, extend, branch) read the record through one reader, so none
of them uses such a file, whichever of its two claims is wrong. Two models of
one graph trained on different cases are not affected: each agrees with itself.

### In the Diagnosis tab and its exports

The WebUI shows **the result's own identity**, never the status's:

- **Banners above the results**, one per state that must not be missed:
  `SYNTHETIC WORKSPACE: not a clinical result.` and `WHAT PRODUCED THIS RESULT
  WAS NOT REPORTED.` for a response without the field (a server from before
  M3c). The no-trained-model and demo banners are gone since K7, and the
  no-shortest-path-table banner since K8-4, with the answers they described.
- **A badge above the results** (since M5-2) compares the result's whole
  identity, part by part, with the model in use as the "Model in use" card last
  read it: "Produced by the model in use", "Produced by an earlier model or
  pipeline — N parts differ ›", or "Cannot check against the model in use yet"
  with its one reason. Its › opens the identity drawer on "This result": each
  part beside what is in use now, the full digests with Copy, the claim
  boundary and "Not a certified medical device; decision support for a
  qualified clinician." (§4.14, "Diagnosis (clinical)").
- **What is served now** is on the page's "Model in use" card (§4.14). Since
  M5-2 the Model Configuration accordion (its path fields, Load / Reload, Save
  Config and Reset) and the header's "Serving now" line are gone: an operator
  changes the model with "🔒 Change model…" on the card, which lists the
  server's workspaces and their checkpoints and reloads only what the server
  lists. `POST /api/v1/pipeline/config` stays for API clients, checked by D8
  (§4.13); the dashboard no longer writes it.
- **The Markdown report** has a "What produced this result" section before the
  candidates: the same banners, a Part | State | Value table with digests in
  full (`sha256:<64 hex>`), the claim boundary and the fixed statement. An
  absent part is written as its state and the reason, never left blank.
- **The kind's label above the table** says what the Score column holds: "Disease
  task model: every disease of the model's graph ranked by its Score
  (uncalibrated); paths only explain the rows shown", "Gene task model: your
  candidate genes ranked by their Score (uncalibrated), the model's similarity
  mixed with shortest-path proximity", "Patients-like-me model: the reference
  cohort's patients ranked by Similarity (uncalibrated); a shared causal gene is
  not implied". The columns follow the kind: Score for the disease task; Score,
  model similarity and SP similarity for the gene task; Similarity for
  patients-like-me.
- **A refused result** (`MODEL_TERM_UNAVAILABLE`, `NON_FINITE_SCORE`) is shown as
  one sentence naming the code and the node; nothing is ranked.
- **The CSV** carries the identity in columns (below).
- **File names** carry the state a copied file must not lose:
  `diagnosis_SYNTHETIC_<patient>_<session>.csv` / `_report.md` for a synthetic
  workspace; otherwise unchanged (the `diagnosis_DEMO_…` name went with the demo
  answer, K7).

### The CSV's candidate columns (Batch 4)

The candidate columns are the result kind's, and every row names the kind:

| Kind | Candidate columns, in order |
|---|---|
| `disease_task/v1` | `result_kind`, `rank`, `disease_id`, `disease_name`, `score`, then the evidence columns |
| `gene_task/v1` | `result_kind`, `rank`, `gene_id`, `gene_name`, `score`, `embsim`, `sp_similarity`, `mean_distance`, `sp_bounds_lower`, `sp_bounds_upper` |
| `patients_like_me/v1` | `result_kind`, `rank`, `case_id`, `similarity`, `disease_id`, `disease_name`, `causal_genes`, `phenotypes`, `reference_cohort_label`, `reference_cohort_version_sha256`, `reference_cohort_samples_sha256`, `reference_cohort_patients` |

The `path_reasoning/v1` (`path_score`) and `demo/v1` rows are gone since K7, and the
`served_mixture/v1` rows (`served_score`, `model_term`, the `sp_term_` columns) since K8-4.

The evidence columns are `evidence_label`, `evidence_mode`, `evidence_summary`,
`min_path_length`, `num_direct_paths`, `supporting_genes` and `matching_phenotypes`
(`num_analogies` is gone since K8-4, with the analogy evidence). A path count is
empty when no evidence package was built (explanations off). `score` and
`similarity` are candidate columns, not part of the identity block below. A gene result's SP
bounds and a patients-like-me result's reference cohort version are the same on
every row.

### The CSV's identity columns

The CSV keeps **one header row** and no preamble. The result kind's candidate
columns come first (above); 18 columns follow, each holding one part of the
identity, **the same on every row**:

| Column | Holds | When the identity holds null |
|---|---|---|
| `served_scoring_mode` | `disease_task`, `gene_task` or `patients_like_me` | — |
| `served_checkpoint_state` | `verified` | — |
| `served_checkpoint_sha256` | `sha256:<hex>` of the checkpoint read | — |
| `served_kg_state` | `verified` | — |
| `served_kg_sha256` | `sha256:<hex>` of the `kg.json` read | — |
| `served_split_manifest_sha256` | `sha256:<hex>` of the `split_manifest.json` read | — |
| `served_sp_state` | `computed` or `absent` | — |
| `served_dataset_record_state` | `verified` | — |
| `served_dataset_origin` | `files` or `synthetic` | — |
| `served_dataset_record_sha256` | `sha256:<hex>` of the `kg.provenance.json` read | — |
| `served_device` | the device in effect (`cuda:0`, `cpu`) | — |
| `served_device_requested` | `auto`, `cuda`, `cuda:N` or `cpu`, as asked | — |
| `served_eta` | the mixing weight in effect | `none` (a disease or patients-like-me model, which mixes nothing) |
| `served_max_path_length` | integer | — |
| `served_path_length_penalty` | number | — |
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
| `none` | nothing applies: no η for a model that mixes nothing |
| `not_reported` | every `served_` column of a response that carried no `served_identity`, or a value one left out |

The words for a model or graph supplied in memory (`none` digests, `not_stated`), for no model
and for the demo answer (`demo`), and the `served_aggregation_method` column, are gone since K7.
The `served_sp_build_id` and `served_sp_max_hops` columns, and the word `absent` they were
written with, are gone since K8-4 (the table's build id and hop bound left the identity), so the
columns after `served_sp_state` sit two places earlier than in a file written before.

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
  counts as unset too. The request's five fields, and the 422 for any other, are in
  §4.14 ("The reload request").

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
> `scripts/setup_demo.py --train-model` prints its serving command, naming the
> disease child it trained (a base, then one disease child, since K7), with the
> device they were trained on.

### What the acceptance reads of serving: S1 (since research comparison Stage 10b)

The GB10 acceptance (`scripts/probe_deployment.py --m2-acceptance`) ends its demo part with
**S1**, a minimal reading of what serving a task model takes (BACKLOG 64). It starts the
service as the launcher does, with the demo's disease child configured at startup, checks that
the service is ready and serves that child, and then reads:

- the **cold start**: from the launch to the first `/health` answer, which includes building
  the pipeline;
- the **warm-up** (the first 5 diagnose requests) and the **steady state** (the next 50), sent
  one at a time: each request's time to its whole answer, and the service's own
  `inference_time_ms`, with their median, smallest and largest;
- the **service process's memory**: its resident size (VmRSS) at readiness and after each phase,
  and its peak since it started (VmHWM), from `/proc/<pid>/status`; and the allocator variables
  it was started with.

On CUDA a request's time already includes the device's work: the service copies the scores to
the host before it answers. torch's own allocator counters (`max_memory_allocated`) live inside
the service's process, and no route exposes them, so S1 records them as not obtainable, with
why. Any reading S1 cannot take is recorded as a state with its reason, never as a number.

**S1 is informational.** It is in neither of the acceptance's required lists, has no threshold
and judges nothing, so a slow or large reading never leaves the acceptance incomplete. It fails
only when it could not read the child at all: the service did not start, was not ready, or did
not serve that child. It then sends no request, and its report entry keeps what it read before
(the cold start, `/ready`'s answer) and says of every later reading that it was not taken. Even
then it does not fail the run: the probe's exit status never counts S1, and the console prints
its failure as an INFO line. Its readings are the demo child's, for one query and one cold
start. They are not the real workspace's model, not concurrent load and not a capacity
decision. What they do not cover is listed beside them in the report (`not_covered`), and their
fields are in `docs/working/research-comparison/GB10_ACCEPTANCE.md` §2.1.

---

## 4.9 Test cohorts: import, measure, record (since contract M4)

Since contract M4 (2026-10-16) a checkpoint is tested from the command line on a **registered
test cohort version** — MyGene2, the institution's offline cohort, or a synthetic cohort for an
acceptance — and its result is recorded beside the checkpoint. The flow is:

1. `scripts/import_cohort.py` previews, then commits, a cohort version mapped to one workspace's
   graph;
2. `scripts/measure_scorer.py --mode PROTOCOL --cohort LABEL --registry DIR` measures the
   checkpoint on it by a protocol this build implements that the model's capabilities admit,
   and keeps the full report;
3. `--record`, or later `scripts/record_evaluation.py --report`, registers the report in
   `evaluations.json` beside the checkpoint;
4. the Diagnosis page's "Model in use" card shows the loaded checkpoint's recorded tests as a
   state (since M5-2; the records themselves are listed under Research › Evaluate › Models).

The commands run on the machine, by the operator. Since M5-2 the dashboard's Research ›
Evaluate › Test page runs the same import, measurement and registration from a browser, each an
operator control D8 checks (§4.13, §4.14); no API request imports, measures or records
anything.

### The registry

The registry is a directory **you create**, given to every command as an **absolute path outside
the repository**. A relative path, a path that does not exist, and a path that resolves inside
the repository (through a symlink too) are refused. Nothing creates it for you, and no request
names it. The commands take it as `--registry`; the service reads it from `SHEPHERD_REGISTRY`
(§4.12), checked by the same rule.

**The commands need `SHEPHERD_REGISTRY` too** (since the M5 review, BACKLOG 37): a report is
registered (`--record`, `record_evaluation.py --report`, the Test page), and `measure_scorer.py
--cohort` measures, only in the deployment's registry, the one `SHEPHERD_REGISTRY` names. The
two are compared as resolved paths, so `--registry` through a link to that directory is
accepted, while a copy of the registry, or the same directory reached through a bind mount or
another mount, is another registry and is refused before anything in it is read. A
`measure_scorer.py --cohort` run resolves `--registry` once, when it is allowed to start, and
reads and writes through that directory: a link retargeted while the run takes the GPU lock or
starts the device does not move it. With the
variable unset, every such command is refused, saying to set it as the service's environment
sets it. Set it in the shell where you run the commands, from the same file the service reads
(§4.11):

```
measure_scorer: refused: --cohort measures a version of the deployment's registry only, and SHEPHERD_REGISTRY is not set here, so which registry is the deployment's cannot be told. Set SHEPHERD_REGISTRY in this environment to the value the service's environment gives it (deployment guide §4.11: one file that both the service and this shell read). [...] Nothing was read or written.
```

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
  versions/<label>/samples.json          the mapped samples (a generated split's: the workspace's file), bound by cohort.json's digest
  versions/<label>/pairs.bin             a KG edge set's pairs instead of samples (below), bound the same way
  versions/<label>/split_manifest.json   a generated split's copy of the workspace's split manifest (below), bound the same way
  versions/<label>/task_fixture.json     a generated split's copy of the workspace's gene fixture record, when it keeps one (below), bound the same way
  versions/<label>/task_fixture_samples.json   that fixture's rows of the split, bound the same way
  versions/<label>/reports/<run id>/     one directory per measurement: ranks.json (a KG edge set's run: scores.bin), then report.json
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
characters); a supplied cohort's role may not be `train` or `val` (a KG edge set's role is its
part of the split, and a generated split's the split it is, below).

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

### A KG edge set: `--kind kg_edges` (since research comparison Stage 4)

A **KG edge set** is a version of another kind: one part of the edge split a model was trained
on, kept as fixed link-prediction pairs, for the KG link-prediction measurement (`--mode
kg_link_prediction`, research comparison Stage 5; "The KG link-prediction measurement" below).
The same preview and commit import it:

```bash
python scripts/import_cohort.py --kind kg_edges --registry /srv/shepherd/cohorts \
    --data-dir data/workspaces/<ws> \
    --from-checkpoint data/workspaces/<ws>/runs/gatv2/<run>/checkpoints/best.pt \
    --label kg-test-2026-11 --role test
```

- **The role is the part, recorded as given:** `train`, `val` or `test`. None is refused for not
  being test. Measuring the train part is a fitting diagnostic and the val part a development
  measurement; neither is ever relabelled as test.
- **Where the split comes from.** The checkpoint's model record names its graph view rule and
  edge split (`data_roles`); a finetune's are its base's. The importer reads the workspace's
  graph through the same readers as measurement and checks that the checkpoint was trained on
  it. It rebuilds the view by the recorded rule, refusing a digest that differs, and rebuilds the
  split, checking it field by field (rule, seed, view digest, digest, edges per part) and naming
  the first that differs. A checkpoint without a model record, of another graph, or whose record
  contradicts itself is refused. Every check runs before anything is written, and nothing of the
  checkpoint is copied. The version names the checkpoint by its SHA-256 only, never by its file
  name or path, so a copy or a link of the same checkpoint makes the same version.
- **The pairs** follow base validation's rule on the role's part (`fixed_lp_pairs`, `BASE_FIXED`):
  every edge once, permuted and given one negative tail by a generator seeded from the split's
  seed, in batches of 512. An edge whose relation has no other tail in its batch has no negative:
  it is left out and counted, never given an invented one. The version records the pairs each
  batch kept (`pairs.batch_pairs`, counts only), so the batch boundaries can be found in the stored
  rows, and each batch's left-out edges are its edges less that count. So a `val` set's pairs are
  exactly the pairs base validation scores. Another rule, seed or batch size is another version.
- **The message-passing graph** is the split's train part (scorer-retraining README A4). For role
  `train` the measured edges are in it, and the version says so; nothing is removed.
- **The preview** prints the view, the split's parts, the positive edges, the pairs and the edges
  without a negative, each with its denominator. Counts are in directed graph-view edges: an edge
  and its reverse twin are one fact and two edges, always in the same part. It also prints the
  pair rule, the pairs file's digest, the message-passing graph and the graph and dataset record
  it binds, but never an edge. A part with no pair cannot be committed.

The version is `cohort.json` (schema 2, kind `kg_edges`, written last) and `pairs.bin`: the pairs
as little-endian int64 rows (head, relation, tail, negative tail) in the order made. The manifest
binds `pairs.bin` by the digest of its bytes; a reader compares that digest before it parses the
bytes, so a changed, truncated or missing file is refused. The pairs are kept, never rebuilt,
because torch's random stream is not promised across releases. No source file is kept: the graph
is the workspace's, bound by its four digests and the dataset record, which a measurement
compares as it does a supplied version's.

The server's lists (`GET /cohorts`) give a KG edge set with its kind, role and counts, never an
edge; the Test page lists it by label, kind and role ("kg-test · KG edge set (test)", after the
supplied cohorts), with its counts in their own units, never with a supplied cohort's counts
(since research comparison Stage 7). With no KG edge set in the registry, the Test page names the
command that imports one (this section). The KG link-prediction protocol measures it (below); a task protocol
naming one, from `measure_scorer.py --cohort` or the Test page's Start, is refused with that
protocol's reason ("the disease measurement does not measure a KG edge set; ..."), and nothing is
written.
Each reader of a supplied cohort's cases refuses one with one shared reason, "a KG edge set
(test), not a supplied cohort version": the Diagnosis route, as a reference cohort, and
`audit_split_overlap.py --cohort` (exit 1, one message, nothing written). The Diagnosis page's
Reference cohort select lists it marked "KG edge set" and never chooses it by default; when the
registry holds no readable supplied cohort, Run Diagnosis is off with that reason.

**Schema 1 and 2.** A supplied import still writes schema 1, byte for byte as before, and schema 1
is read as a supplied cohort. A KG edge set and a generated split (below) are schema 2, which
states its kind. Schema 2 is read only for the kinds written in it: a schema-2 manifest of kind
`supplied`, which no revision writes, is refused ("which no revision writes: a supplied version is
schema 1"), never read by the schema-1 rules. A revision from before schema 2 lists a KG edge set
or a generated split as unreadable with its reason ("is schema 2; this revision reads schema 1
only"), so an older revision never reads one as a supplied cohort; one that reads KG edge sets
but not generated splits (research comparison Stage 4a) lists a generated split as "of kind
'generated'; this revision reads kinds supplied and kg_edges only".

### A generated split: `--kind generated` (since research comparison Stage 4b)

A **generated split** is one of the workspace's own generated patient splits, the
`<role>_samples.json` the generator wrote and `measure_scorer.py --split` measures in place, kept
as a registry version with a limited data identity of its own:

```bash
python scripts/import_cohort.py --kind generated --registry /srv/shepherd/cohorts \
    --data-dir data/workspaces/<ws> --label gen-val-2026-11 --role val
```

- **The role is the split, recorded as given:** `train`, `val` or `test`; none is refused for
  what it is, and none is relabelled. The generator writes `train` and `val` only, so `--role
  test` is refused where the workspace's split manifest binds no generated test split, naming
  the splits it binds, with nothing written.
- **What is checked.** The importer reads the workspace as `--split` measurement does: the graph
  against the split manifest, the dataset record, and both generated splits against that one
  manifest reading (their digests, disease sets and disjointness). A split that is not the file
  the manifest binds is refused, and so is a workspace re-sampled between the preview and the
  commit (its preview id changes). Every check runs before anything is written. `--source`,
  `--partial-phenotypes` and `--from-checkpoint` do not apply (exit 2).
- **Its data identity** is the samples' digest, the split manifest's digest and the digest of each
  samples file it binds, the generation definition (the generator's algorithm, version and
  recorded settings, and the disease allocation it cut, copied from the split manifest as they
  are), and the role. The counts are the split's generated patients and its distinct diseases,
  counted from the samples verified, each with its unit; a split manifest whose own count of
  them differs is refused. Each split file is named by its split's name, never a path. The
  version records no path.
- **The samples and the split manifest are copied, not bound in place.** The version keeps the
  workspace's file byte for byte as `samples.json`, bound by the digest the split manifest
  records, and the split manifest itself as `split_manifest.json` (digests and counts only, no
  sample), bound by its own digest, so it stays readable and verifiable after the workspace is
  rebuilt or re-sampled, its generation definition included; whether the workspace still holds
  that split is then a reference's state, not a condition of reading the version.
- **The functional gene fixture travels with it** (since research comparison Stage 5b). When the
  workspace holds the gene task's functional fixture of its build (`derived/gene_fixture/`,
  `scripts/derive_task_fixture.py`) and its record describes the split kept (the same split
  manifest, graph and samples), the version keeps the record as `task_fixture.json` and the
  role's rows as `task_fixture_samples.json`, each byte for byte and bound by digest, under a
  `task_fixture` block (its rule, seed, distractors and what the derivation did to the split)
  and a counted `task_fixture_samples`. A fixture that is absent, of another build or unreadable
  is not kept and refuses nothing: the preview says why, and the version holds the split for the
  disease measurement only. To keep the fixture, derive it, then import the split again under a
  new label.
- **The preview** prints the counts, the generation definition and allocation, the split manifest
  and the split's files by digest, the samples kept, what was verified, the functional gene
  fixture kept (or why none is), and the graph and dataset record it binds, never a sample or a
  patient id. A split with no sample cannot be committed.

The server's lists (`GET /cohorts`) give a generated split with its kind, role and counts, never a
sample; the Test page lists it by label, kind and role ("gen-val · Generated split (val)"), with
its counts in their own units, never with a supplied cohort's counts. It is never presented as a supplied cohort, or
as test unless its role is test: the Diagnosis route and `audit_split_overlap.py --cohort` refuse
it with the registry's shared reason ("a generated split (val), not a supplied cohort version"),
and the Diagnosis page's Reference cohort select lists it marked "generated split" and never
chooses it by default.

**Measuring it** (since research comparison Stage 5b): `measure_scorer.py --mode <protocol>
--cohort <label> --registry <registry>` measures a registered generated split by each task
protocol the model's capabilities admit (disease, gene, patients_like_me), through the same
evaluator `--split` uses, so the same model on the same split gives the same figures either way.
Disease reads the version's `samples.json`; gene and patients-like-me read the fixture the version
keeps; each file is read once and refused before it is parsed unless it is the bytes the version
binds. A version that keeps no fixture, or whose fixture holds no row of its split, is refused
for gene and patients-like-me with that stated data gap before anything is read, never for its
role; the job runner and the Test page's Start say the same. The run is refused, with nothing
written, where it cannot be valid: a version file that is not the bytes the version binds, or a
`cohort.json` that contradicts them; a version of another graph than the workspace and the
model; a protocol the model's capabilities or the data kind do not take; and
`--training-evidence` (R10 is a supplied cohort's; a generated split's relation to training is
its disease allocation's). Its report is kept in the version's report store like any version's,
each run in a new run directory (`ranks.json`, then `report.json`), never overwriting one. The
manifest records `cohort_kind` `generated`, the version's block (its role as recorded, train or
val today, its generation identity and the fixture it keeps) and the role as `split`; there is no
R10, and the role is never presented as test. Its report is registered like any kept report
(`--record`, or the register-later command the run prints; research comparison Stage 6): its
ledger record keeps the role and the generation identity. A generated test role cannot be
imported today, so none is measured. The job
runner chooses R10's evidence once, by the version's kind (since research comparison Stage 7): a
supplied cohort's run is given the `SHEPHERD_TRAINING_EVIDENCE_DIRS` directories, and a refused
setting refuses its start; a generated split's or a KG edge set's run is given none, whether the
setting is set or refused, so a Test-page run on one starts and is measured. Reports measured with
`--split` are untouched: they stay where they were written, no identity is made up for them, and
`record_evaluation.py` still refuses to register them.

**Moving a registry.** No version records a path: each names what it depends on by content (the
graph's four digests, the dataset record and each source it names by role, and by kind the view,
split, pairs and checkpoint of a KG edge set, or the split manifest, split files, samples and
any functional gene fixture of a generated split). A registry copied or moved to another directory that `--registry` or
`SHEPHERD_REGISTRY` then names reads the same versions under the same digests; which directory is
allowed is that setting's rule, apart from the content check. No command or page shows whether
a version's dependencies are present yet: the registry's library check (`check_references`)
gives each as a state (verified, differs, missing, unreadable or not checked), never a reason a
version cannot be read, and the results service is to show those states (research comparison
Stage 8). The source roles a version names are the dataset record's, with no list of allowed
roles; today's KG builder records four (MONDO, HPO, the phenotype annotations and
genes-to-phenotype), and a new source needs the builder's own extension first.

### Measure

```bash
python scripts/measure_scorer.py --mode disease \
    --checkpoint data/workspaces/<ws>/runs/<arch>/<run>/checkpoints/<file>.pt \
    --data-dir data/workspaces/<ws> \
    --cohort mygene2-2026-10 --registry /srv/shepherd/cohorts \
    [--training-evidence DIR ...] [--record] [--device auto|cuda|cpu] [--batch-size 32]
```

`--mode` is required and names the **protocol**: a measurement this build implements, judged by
what the model offers (its capability description, read from its model spec;
`src/evaluation/protocols.py`), today `disease`, `gene` and `patients_like_me`, each taking a
finetuned model that carries its task head, and `kg_link_prediction`, taking a base or any
finetune by its LP head and measuring a KG edge set (below). The report's mode is the
protocol's name. The run judges the protocol on the bytes it reads, and a refusal says why and
ends "Nothing was written.": a registered version must list the genes the protocol needs (`gene`
causal and candidate genes, `patients_like_me` causal genes), checked right after its
`cohort.json` is read, before the workspace or the checkpoint; the checkpoint's model must offer
what the protocol needs (for the three task protocols, its task head), checked right after the
checkpoint is read. A base asked for a task protocol is refused with that protocol's reason, which
names the one that takes it ("the disease measurement takes a finetune model with the disease
head; this checkpoint is a base model with no task head. It can be measured by
kg_link_prediction."), and a
checkpoint without a model spec (a ShepherdGNN history file) with the record reader's phrase,
"written before the model spec: not readable by this build; start a new run" (since K8-7b,
2026-11-08; it used to be refused "as history"). The Test page passes the
protocol the operator chose (§4.14). Since K8-6a no command runs Mode C, the model's own disease
ranking from before the tasks: a run without `--mode` is refused by the argument parser (exit 2),
and so is `--mode C`, which is not a protocol. Kept Mode C reports and records are read by the
shared readers (policy rev 15).

Run it in a shell where the GPU lock is configured, the same lock as the service's (§4.11);
without one it refuses, exit 1, with nothing written.

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
  come from it. `--training-evidence` cannot help;
- **search incomplete**: no file that was read holds the recorded bytes, and a place that may
  hold them could not be read: a candidate (permission, an I/O error), or a `--training-evidence`
  directory that cannot be listed or cannot be checked. The console names each place and its
  error. Make it readable to the process that measures, and measure again.

`none` is not proof of no leakage: it speaks only about the inputs the checkpoint records.

**A place that cannot be read is recorded, not refused** (since K8-6r.3, R10 definition version
2). A candidate that cannot be read, or a `--training-evidence` directory that cannot be listed
or cannot itself be checked (under a directory this account cannot enter, for one; since the
K8-6 completion review, same version), leaves its scope `unverifiable` with that reason, never
"not found" and never a count. The search goes on
through the remaining candidates, so bytes held by another file are still found, and the
measurement still runs and its report can be registered. **What still refuses the run:** a
`--training-evidence` path known not to be a directory (it does not exist, is a file, is a link
loop, or has a name too long to exist), an argument error before anything is read; and the
required inputs: a checkpoint, graph file or cohort samples file that is missing, cannot be read,
or is not the bytes recorded for it. Running out of memory while reading a candidate ends the run
as a resource failure that names the file: it says nothing about overlap, and no kept report or
ledger record is changed. Candidates are read only until every recorded
digest has been found, so an unreadable file after the match is never reached. A report written
before K8-6r.3 (R10 definition version 1) is read as written: there, "not found" means every
candidate was read and compared.

**Where the report is kept.** In the registry, under the version:
`<registry>/versions/<label>/reports/<UTC time>-<first 12 hex of the checkpoint digest>-<8 random
hex>/`, a new directory for every run: `ranks.json` (per case, patient-level) and then
`report.json`, which names the ranks file by its digest. Reports are never overwritten, and every
report is kept whether or not it is registered. `--output` is refused with `--cohort`.

The console prints the run's metrics (its protocol's), the coverage (`N ranked of M source cases; K excluded`, with
each category), the four R10 labels in the shared words of `src/evaluation/caveats.py` (an
`unverifiable` label with its reason), the label definitions once, the report's path and the run
id. The report itself keeps the states and holds no prose.

### The KG link-prediction measurement: `--mode kg_link_prediction` (since research comparison Stage 5)

One protocol measures a KG edge set, for a base and each of its finetunes alike (the owner's O3):
the formal KG test and a family's common read-out use this one evaluator, and differ in the data
role, the exposure and the research question, not in how they are scored.

```bash
python scripts/measure_scorer.py --mode kg_link_prediction \
    --checkpoint data/workspaces/<ws>/runs/<arch>/<run>/checkpoints/<file>.pt \
    --data-dir data/workspaces/<ws> \
    --cohort kg-test-2026-11 --registry /srv/shepherd/cohorts [--batch-size 512] [--device cpu]
```

- **What is scored.** Each stored pair, a positive edge against its one rule-generated negative,
  by the model's LP head (the DistMult raw score), over the full-view vectors of the version's
  message-passing graph (its split's train part, with the view's node types) for every model. No
  patient samples are read. `--batch-size` is the pairs scored at once.
- **Metric schema 3.** LP accuracy (a positive's raw score above 0, a negative's at or below 0,
  over both), ROC-AUC (average ranks), AP and lp_loss, over the whole set, each with its unit and
  denominator; a metric that cannot be computed is a status with its reason, never a number. The
  counts carry their units: `n_samples` is the version's positive edges, `n_ranked` the pairs.
  There is no R10: `--training-evidence` is refused for a KG edge set.
- **Checked as for a supplied version,** before anything is written: the version's `cohort.json`
  and `pairs.bin` against the digest it binds (a changed file is refused before it is parsed, and
  pairs that parse to other rows or another pairs digest than the version records are refused),
  the workspace's graph and dataset record, the version against the workspace, the checkpoint
  against the graph, and `--expect-*` digests. Also refused: a version of another graph view than
  the model's (there is no node mapping between views in this build: a capability gap, not a
  research rule), a file without a model spec, a version with no pair, a recorded split that does
  not rebuild from the view, pairs that are not the role's edges, and `--split` with this
  protocol. **Not refused:** a model of another split of the same view, another family, another
  environment, or another result.
- **The exposure** is a top-level block of the report (definition `kg-lp-exposure/v1`, with its
  sources: the checkpoint and the version). It says how the measured edges stand to what the
  model's record says it used of the graph: "held out relative to the recorded training and
  validation graph use" (a base on its own test set), "k of n" (its val or train set, or a model
  of another split), "after full-view exposure" (a finetune, whose validation passed messages over
  the full view), or "exposure not recorded" (a record this build cannot read). It is never a
  gate, and the negatives are never claimed as proven non-relations.
- **What is kept.** In the version's report store, a new run directory holds `scores.bin` (each
  pair's actual raw scores and state, keyed to the pairs file's rows, streamed in chunks and
  placed whole) and then `report.json`, which names it by digest and row count. A run cancelled
  or failed before the report leaves no report, and what it left is listed as an interrupted
  write.
- **Registering it** (research comparison Stage 6) works as for any kept report: `--record`, or
  the register-later command the run prints. Its ledger record is a KG edge set's: the version's
  view, split, message-passing graph, pairs and counts with their units, no R10, the run's counts
  and each metric's status, and its exposure block as the report wrote it, so the ledger alone
  still says how the measured edges stand to the model's recorded graph use. Models and
  `record_evaluation.py --show` show it by its kind, with that exposure.

### Mode D (removed in K7)

Mode D (`--mode D`, Batch 3's measurement of what the Diagnosis page served in October 2026, the
input to the scorer policy's gate 1 comparison) was removed in K7 with its executor and its probe:
`--mode D` is refused at the command line (exit 2), and no page or job starts it. Its GB10 run of
2026-10-27 is kept as evidence in `docs/working/batch3-acceptance-2026-10-27/`, and its design
in `docs/working/scorer-measurement/PLAN_B05.md`.

**Kept Mode D reports and records stay readable**, shown as recorded by the one display every
mode this version does not describe gets (since K8-6e.2, 2026-11-07; before it, under a Mode D
label of their own): on the Test page's Result and the Models page they read "Mode D (a
measurement mode this version does not describe)", every metric by its recorded name and value,
unrounded (`post_import_mrr 0.35`), and the score semantics exactly as recorded
(`served:gnn_only:eta=1.0:sp=none`), never split into a scoring mode and η. The Models page lists
each mode's records in a table of its own, in the order first shown, and ranks no record by its
test score. What Batch 3's names meant: `post_import_*` and `coverage` were over the version's
import-eligible cases, a case not reached counting 0; `reached_*` over the reached cases only; the
cases excluded at import were counted beside them, never in them. Each report also set Mode C
beside it on the same embeddings. A Mode D number is not the model's approved ranking and does not
describe the current diagnosis model.

**A kept Mode D report registers by the general rules, under its own mode** (since K8-6c, policy
rev 15): `scripts/record_evaluation.py` and the Test page's Register check it as they check every
report (the report store, its version block, its run directory, the checkpoint's bytes and the
ledger), and its record keeps mode D. The same run's report again answers `already_registered`,
with nothing written. The records a ledger already holds stay, and do not stop the next
registration of another report.

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
  one shell or another and has to be quoted by hand. A run on a version of any kind prints it (a
  KG edge set's and a generated split's included, since research comparison Stage 6).
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
- **Registered by the run** (policy rev 16, since K8-6r.2). A run is its run directory in the
  version's report store. Each record names it (`run_id`), and the ledger looks for this run
  first. The outcomes:
  - **registered** (with its sequence number): every verified run gets a record of its own,
    with its own values, environment and R10, whatever its bytes or times. Two runs with
    byte-identical reports are two records, and so are two runs whose values differ.
  - **already held** (`already_registered`): the same run with the same report bytes again;
    nothing is written.
  - **refused** when the same run comes back with other bytes: a run's report is sealed when it
    is published, so the file was replaced after it was registered. The message names the
    record and both digests.
  - **refused, nothing written,** when the write itself cannot be made safely, naming the
    record: a record that is not an object or has lost its registration sequence (the next
    sequence cannot be allocated), or this run's own identity record damaged (a record from the
    run-identity boundary on that should name its run and does not, a record naming this run
    whose cohort, checkpoint or report digest cannot be read, or two records naming this run).
    Restore the record from a trusted backup, then register again. Nothing is guessed, skipped
    or renumbered. A run whose intact record is found still answers as above.

  After the outcome, two notes may follow. They are information only and never refuse
  anything. **The group note** names each other record of the same conditions (same checkpoint,
  cohort version and measurement settings) whose results differ, with its run and what differs:
  "Runs under these conditions differ: … Each run is kept as recorded." When both runs had
  deterministic algorithms on, it adds that they were expected to agree: investigate. **The
  historical note** names a record written before each run had its own record that holds this
  report's bytes: "possible overlap with record N, … not recorded whether this run is one of
  them". Such an older record never answers `already_registered` for another run.
  There is no override flag; a wrong record is corrected by editing the readable JSON file,
  which leaves a diff.
- **What is shown later:** `python scripts/record_evaluation.py --checkpoint <file> --show
  <checkpoint sha256>` reads the ledger the service reads for that file, beside its resolved
  path. It prints the records as written (JSON), then one line each with its run ("not
  recorded" for a record written before each run had its own record) and any possible
  overlap between such a record and a named run. A record the shared check cannot read as a
  result is said instead as "record N (the ledger's record I) cannot be read as a result:
  <reason>" (since K8-6e.1), with the reason the API and Models give; the JSON still prints it as
  written. With `--checkpoint-dir <dir>` instead, `--show` reads the `evaluations.json` inside that
  directory, which for a directory of links is not the one the service reads. An empty answer
  means this ledger holds no result for those weights, not that they were never evaluated.
- **Registrations into one ledger take turns.** A registration holds the ledger's lock
  (`evaluations.json.lock` beside it, kept after use) from reading the ledger to writing it and
  its usage-history entry, across processes. One that waits more than 10 s is refused as busy:
  nothing was written this time, the report stays kept, and you register it again (exit 3 for
  `--record`, 1 for `record_evaluation.py`). If the ledger was written but the usage history
  was not, it says "the usage history was not written" with the reason, and the record stands.
  A ledger changed by something that takes no lock, such as a hand edit, is still refused
  rather than overwritten.
- A report of a **registered version** of any kind is recorded: a supplied cohort's, a KG edge
  set's, or a generated split's kept with `import_cohort.py --kind generated` (research
  comparison Stage 6). A report from `--split` is not, because it has no registered version, so
  no data identity the ledger records results by, not because such a result counts for less
  (policy rev 16). Its report and ranks stay at `--output`; the same split kept as a version is
  measured with `--cohort` and registered.

### Export every result: `scripts/export_results.py` (since research comparison Stage 8)

One read-only results service (`src/api/services/results.py`) reads every kept report and every
readable ledger record, and this command exports them as CSV or JSON. **It runs as the operator on
this machine**: it reads the directories you name with this account's rights, never through the
server, so nothing is offered over the network, and it writes nothing but the export.

```bash
python scripts/export_results.py --workspaces-root /srv/shepherd/workspaces \
    --registry /srv/shepherd/cohorts --format csv --output results.csv

# several sources, each named; filters keep the rows that match
python scripts/export_results.py --workspaces-root lab=/data/lab/workspaces \
    --workspaces-root archive=/mnt/archive/workspaces --registry /srv/shepherd/cohorts \
    --protocol kg_link_prediction --role test --format json --output kg.json
```

- **Sources are named explicitly**, as many as needed: `--workspaces-root [NAME=]DIR` (a directory
  whose subdirectories are workspaces, as `SHEPHERD_WORKSPACES_ROOT` names one) and
  `--registry [NAME=]DIR`. Each is read with the rule the service reads its own setting by
  (absolute, existing; a registry outside the repository), as a source of its own; this shell's
  `SHEPHERD_*` settings are neither read nor changed for them. `NAME` is what rows show for the
  source, never its path or the directory's own name (which may say more than an export should); by
  default `workspaces-root-1`, `workspaces-root-2`, ... and `registry-1`, ... in the order given,
  and two sources may not share one.
- **One row per result.** A kept report and a ledger record are one row only when the record names
  that report's run and holds its bytes; copies of one run in several sources are one row with each
  place. A ledger record whose report no named store keeps reads "report not available"; a kept
  report no named ledger registered reads "not registered in a listed ledger"; a record merged
  before each run had its own record is its own row with no run, never counted as a run, and the
  reports it holds say they may overlap it. Two runs of one condition group, with equal or
  different numbers, are two rows of one `condition_group`, with the fields they differ on.
- **What a row lists**, where its records hold it: the checkpoint's SHA-256 and where the
  server's lists find it (source, workspace, run and file), lineage, spec summary and structure
  (`spec`: every value of the model's spec as its record holds it), the training settings the
  checkpoint records (rules, learning settings, what took effect; a weights-only file records
  none), its weight transfer and loss identity as its record holds them (`weight_transfer`: how
  its stage's weights began, each part trainable or frozen and the SHA-256 of the downstream
  weights training started from, where the record names them; `loss_identity`: the loss's terms
  and coefficients and the terms without a trainable parameter), the data's kind, label, role
  and digests, the measurement's protocol, settings and
  environment (class, regime, GPU, native build identity, and `components`: every fact each named
  component's record holds, its requested, installed, available, selected and in-effect states,
  fallback reason, version, native identity and source facts, a fact the record does not hold left
  out and any server path in a text replaced), a KG edge set's run counts
  (`run_counts`: pairs scored, pairs whose score was not finite), the data's relation to the
  model's training (a supplied cohort's R10 states and counts in `r10`; a KG edge set's exposure;
  a generated split's is its allocation's and is not compared), the kept / registered / group
  states and the version's content references. Every ledger of every listed workspace is read,
  whatever became of the weights it names: a record whose checkpoint file was removed or is
  listed with a problem is still a row, its model unknown with why. `unknown` names each field
  without a source, with why; nothing is filled from today's recipe.
- **The long form**: one row per result and metric, with `protocol`, `metric_name`, `value` (the
  number as recorded, an integer as an integer, or a status such as `not_computable`), `unit`, `role` and `semantics`, so two
  protocols never share a metric column. CSV cells follow the Diagnosis CSV's rules (a digest as
  `sha256:<hex>`, a text a spreadsheet would run defused, numbers unrounded); JSON keeps the same
  rows typed. No case id, rank or server path; a report's `checkpoint_path` and `data_dir` are never
  opened. Nothing is capped or ordered by a metric (newest first).
- **An item that cannot be read** (a refused root, a workspace entry outside its root, a version,
  a store, a damaged report, a checkpoint file, a ledger, a hand-edited record, one naming no
  weights among them) is a row with its reason, and hides nothing; a kept report whose run a
  listed ledger names in a record that cannot be read says so rather than "not registered";
  stderr counts them. Filters (`--family`, `--workspace` (where a row was found: a listed
  checkpoint of its model or a ledger holding its record; since Stage 8b; it matches the
  workspace's name in every named `--workspaces-root`, and each row's locations name the root),
  `--protocol`, `--data-kind`, `--role`, `--environment-class`, `--device`, `--torch-version`,
  `--cuda-version`, `--software-revision`) always keep such rows; a row whose field has no value
  matches no filter on it.
- `--output` must name a new file; an existing one is never overwritten. Without it the export goes
  to standard output.

### Exit codes

| Command | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| `import_cohort.py` | preview printed, or version committed | refused, with one message ending "Nothing was written." (also a preview with nothing to commit: no case can be ranked, no edge of the part has a negative tail, or the generated split holds no sample) | the command line was rejected | — |
| `measure_scorer.py` | measured (with `--record`: registered or already held) | a check refused the run: one message, nothing written (so does `--device auto` or `cuda` on a host without CUDA, before anything is read); or **measured, but the report was not written** (a full or read-only disk, for example): one message saying what may be left, which is never a report that can be registered -- this run's ranks beside `--output` with no report, or an empty file it claimed at `--output` or at the ranks (since K8-6r.4 each refuses the next run on that path), or a run directory without `report.json` in the report store (a run that fails after writing its report, its console closed for example, says where the report is) | the command line was rejected, including an argument of the other kind of cohort | **measured, but the registration refused**: the message names the report and the `record_evaluation.py` command to register it later; or, when the report file no longer holds the bytes the run published, says so and asks for a re-measurement instead (that report may be gone) |
| `record_evaluation.py` | registered, already held, or shown | refused, with one message; nothing written | the command line was rejected, including neither or both of `--checkpoint` and `--checkpoint-dir` | — |
| `export_results.py` | export written (an item that could not be read is a row of it, counted on stderr) | — | the command line was rejected (no source, a source name given twice), or `--output` already exists: nothing written | — |
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
the one published and something else refused it, such as the ledger. Without `--record` the
command is always printed: no other run's report or result can refuse a registration, whatever
the kind of data the run measured. A later registration with `record_evaluation.py` checks the
checkpoint file it is given against the report, as before.

### What the Diagnosis page shows

**Since M5-2 (2026-10-21) the Diagnosis page shows the loaded checkpoint's recorded tests as a
state only**, on its "Model in use" card: "Recorded results: 3 · runs differ for 1", "3 ·
comparison incomplete" (no registry is set, or no record's cohort version or report store can be
read; never shown as agreeing), "3 · runs differ for 1; comparison incomplete for 1" (counted
apart), "3 · 1 not readable" (since K8-6e.1: 3 results and 1 record the shared check cannot read
as a result, counted apart and never as a result; "0 · 1 not readable" when it is the only one),
"none in ledger" (neither), "no ledger", "ledger not
readable" or "not read", the same words as the checkpoint's row in
"Change model…", found by the checkpoint's digest, not its file name (after another model is
loaded the previous model's state is never shown). Its › opens Details at Evidence, where the same
state stands beside the Trainer's validation figure ("Validation pooled MRR (Trainer) … —
selection metric on val, not an independent test" for a task model); no test number is shown on
the page. The
records themselves are listed under Research › Evaluate › Models.

Until M5-2 the model status (Model Configuration) listed them in full under its "Serving" line,
as **"Recorded test results (Mode C)"** for the loaded checkpoint (`sha256:<first 12>…`), read
through `GET /api/v1/evaluations/loaded-model`, in the words below. The route still answers API
clients with those records, and since the M5 review (BACKLOG 36) each record carries how its
condition group compares in the store (`contest`), from the dashboard's own implementation over
the service's `SHEPHERD_REGISTRY`, without the reasons' own words, which may name a server path.
**Since K8-6r.2 these values only describe** (policy rev 16): `uncontested` (the runs agree),
`contested` (runs under these conditions differ, with the fields and every run that carries a
differing report) and `cannot_be_checked` (the comparison is incomplete, never to be read as
agreeing) stay as the same strings for API clients, but none of them hides, orders out or
refuses a record any more. Each record also carries `run_id` (the run it was registered from, or
`null` for a record written before each run had its own record, shown as "not recorded") and
`historical_overlap` (the sequences of the records it may be one execution with, never
confirmed). **Since K8-6e.1 a record for these weights that the shared check cannot read as a
result** (its run identity cannot be read, or it does not have the shape registration writes) is
listed in `unreadable_records`, in ledger order: its `index` in the ledger's records list
(counted from 0, as the writer's refusals name it), its registration `sequence`, its
`source_artifact_digest` and its `run_id` where it still carries them readably (`null`
otherwise), and its `reason`, one line of at most 300 characters that names each refused field
by its path in the ledger record, the key a hand edit corrects, and never its value (for example
"n_ranked: missing", "environment.facts.gpu: missing" or "the regime's device is neither CUDA nor
CPU"); a key the record itself carries inside an object, such as a metric's name or an exclusion
reason, is named as written, as a readable record's metric names are shown. It is never a
result, and the ledger keeps it as written. `records_not_shown` keeps
its meaning and always equals the list's length. A record that is not an object, or names no
checkpoint digest, belongs to no model and is not listed; a ledger unreadable as a whole is
still `ledger_not_read`, with none of its records listed.
Investigation notes are not shown (BACKLOG 40):

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
  sequence and time, its run, and the number of reports merged into it before each run had its
  own record.
- Instead of records, a sentence: no ledger beside the loaded checkpoint; the ledger was not read,
  with the reason, which names the file and never its directory (for example a schema 2 file,
  below); this ledger holds no result for these weights; or "The loaded model changed while its
  recorded results were read; refresh the page."
- **Records the shared check cannot read as results** (they do not have the shape registration
  writes, or their run identity cannot be read; since K8-6e.1) are listed under the results, or
  alone when there are none: one amber sentence with ▲ ("1 record for these weights cannot be
  read as a result. The ledger keeps it as written.") and a table, Index · Registration · Report
  · Reason, each with the route's `unreadable_records` values (a sequence or report digest the
  record does not carry readably reads "not readable"). They are never counted as results.
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
- **A generated split's report is not registered**: `record_evaluation.py` refuses it ("a
  generated split's report is not registered"). *Reworded in K8-6r.4 (2026-11-07), behaviour
  unchanged:* the reason is that the ledger does not yet support a generated split's data
  identity (it has no registered cohort version), not that such a result counts for less; its
  report and ranks stay at `--output`, and extending the ledger to them is later work. Since M5
  a report at `--output` is refused before that, because no registry's report store holds it
  ("… is not a report the registry's report store holds", below); the message quoted here is
  for one placed in a store. Behind both is the same reason: a generated split has no registered
  cohort version, and so no report store. `--split val` is the checkpoint-selection split, not
  held-out data.
- **A generated split's report and ranks are staged and renamed into place** (ranks first); an
  existing `--output` is still replaced. *Superseded by K8-6r.4 (2026-11-07):* neither file ever
  replaces one. A run whose `--output` or `<output stem>_ranks.json` already exists (a link
  included) is refused before anything is read, saying what it found, and a file that appears
  after that check is not overwritten either: the run then fails, saying what it left. An empty
  file at `--output` is a claimed place whose publication failed, not a report, and ranks with no
  report beside them are an interrupted publication, not a result: each refuses the next run on
  that path and stays where it is, since no earlier file is deleted or renamed to make room.
  The refusal looks at those two names only, so it also says what else such a file may be:
  another run's publication not yet finished, or, at the ranks name, a file another `--output`
  wrote. Measure to a new `--output`. Each file is placed by a hard link, which fails when
  anything is already there; where the filesystem has no hard links (FAT or exFAT, say), the
  target is first claimed as a new empty file and the written file then renamed onto it. That
  rename is itself an overwriting call: it overwrites no other run's file only while the target is
  still the empty file this run has just claimed, so it relies on nothing else replacing or
  removing that claim in between, which this program never does and cannot stop another writer or
  a person from doing. Neither Windows nor a real FAT or exFAT mount has been tested.
- **Ledger schema 3 (M4), schema 4 since K8-6r.2.** An existing `evaluations.json` of schema
  1 or 2 is refused — by `--record`, by `record_evaluation.py` (`--report` and `--show`) and by
  the recorded-results readers, which report the ledger as not read (on the Diagnosis card:
  "Recorded results: ledger not readable"). Nothing migrates it and its results are not carried
  over: **archive it by hand**, for example `mv evaluations.json evaluations.v2.json`, read it
  with the revision that wrote it, and re-measure what you need with this revision, then
  register the new reports (the refusal says the same). A report written before M4 is refused
  too, with an instruction to re-measure. **Schema 3 and 4 are both read** as written, and
  reading never rewrites a file. The first registration into a schema 3 file writes schema 4
  and records from which sequence every record names its run (`run_identity_from_sequence`);
  earlier records stay as written, their run "not recorded", and no run is invented for them. A
  new ledger starts at schema 4. A build that knows only schema 3 refuses a schema 4 file
  ("ledger not readable") until it is upgraded again; nothing in the file is lost. A schema 4
  file whose `run_identity_from_sequence` was lost or made unreadable (missing while the file
  holds records, not an integer, below 1, or above the next sequence) is still read, but no run
  that a record does not name can be registered into it: restore that field from a trusted
  backup. A hand edit that leaves it readable is taken as written, as any other hand edit is.
- **`record_evaluation.py`** takes `--checkpoint FILE` or `--checkpoint-dir DIR`, exactly one.
  With either, the ledger is written beside the checkpoint's resolved path. For a directory whose
  checkpoint is a link, that is beside the link's target, no longer in the directory named.
- **`scripts/audit_split_overlap.py`**: `--evaluation-cohort-kind` is removed; a test cohort is
  audited with `--registry DIR --cohort LABEL`. `--registry` without `--cohort` (or the reverse),
  and `--splits` beside `--cohort`, are rejected command lines (exit 2, with the usage line). The
  version's graph and dataset record digests are compared with those this workspace's manifest
  records, without loading tensors; only `train` is read. The report's `schema_version` is 6.

### Since M5-1: registration from the report store, runs that differ, history, determinism

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
in the store that is a link to a file elsewhere does not count. **The registry must be the
deployment's** (since the M5 review, BACKLOG 37): a copy of the whole registry holds the same
bytes in the same structure, but it would keep its own report store and usage history, which
Research › Models and the job runner never read, so its reports are refused ("…
is not registered: the registry … is not the deployment's: SHEPHERD_REGISTRY (…) resolves to …").
A registry moved as a whole works again once `SHEPHERD_REGISTRY` names its new place, for the
service and in your shell; no absolute path is stored as an identity. If `--record` finds its
registry is no longer the deployment's (a link retargeted during the run), the run exits 3,
keeps its report and offers no register-later command, since that command would be refused the
same way. The check catches a registry reached by mistake; a command run with the variable set to
a copy on purpose uses that copy. A generated split's report,
written to `--output`, is never in a store, so this is now why it is refused.

**Runs that differ are kept and described, never refused** (policy rev 16, since K8-6r.2;
until then they blocked their key). Every valid run is its own record, so runs under one set of
conditions (same checkpoint, cohort version and measurement settings) whose results differ are
all kept as recorded. The registration reads only the run it registers, and prints the group
note after its outcome (above). The store, the Test page, Models and the API describe such a
group as "runs under these conditions differ", naming what differs and every run that carries a
differing report. Without deterministic algorithms, runs can differ: the description is
neutral, and every result is valid as recorded. Only when the group records deterministic
algorithms on were its runs expected to agree, so the pages mark the difference in amber with
"— investigate". The API keeps the value `contested` for it, with this new meaning. Without
`--record` the run always prints the register-later command. To investigate:

- compare the reports' environment facts (each `report.json`'s `manifest.environment`);
- re-run with `--deterministic`, which is its own set of conditions. Deterministic runs are
  expected to agree, so the note and the pages flag a difference between them for
  investigation.

An investigation note records who looked, when and what they found. Notes go to
`<registry>/versions/<label>/reports/notes.jsonl`, kept with the group's reports and never in
the ledger. They change nothing. Research › Test adds one from the dashboard, as an operator
control (§4.14). The tools never delete anything.

**Another run's report never blocks a registration.** A report in the store that cannot be read
or parsed, a run directory or `report.json` that is a link out of the store, and a report whose
conditions cannot be told are each listed by their own run with their own reason. They mark no
other run and no group. A report of the same conditions that this revision cannot compare makes
that group's comparison incomplete (`cannot_be_checked`, "comparison incomplete"), which is a
description only. The report you register is still checked in full: its place in the store, its
bytes, its version block, its run directory, its checkpoint and the ledger. A store that cannot
be listed at all leaves the Test page unable to confirm a report, so its Register stays off and
says why; register the run with `record_evaluation.py --report …`, which reads only that
report.

**The usage history** is `<registry>/versions/<label>/reports/usage.jsonl`, one JSON object
per line, appended and never rewritten. It records:

- every measurement of the version (`test_run`: when, run id, checkpoint and report digests,
  the run's mode and metrics, `n_ranked`, whether it was deterministic, the device, whether `--record`
  was asked);
- every registration that wrote a ledger (`registered`: its run id, outcome and the run's own
  sequence). Entries written before K8-6r.2 with the retired outcome `corroborated` stay in the
  file and are shown as recorded;
- until K8-6d, every model loaded from a list ordered by the version's score (`model_load`: the
  metric and scoring group that ordered the list, and whether the load took effect, with why).
  Ordering by test score has retired and nothing writes these entries now; entries already
  written stay in the file, and Research › Models shows them as recorded (§4.14).

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
the file may hold part of it. Each append to `usage.jsonl` or `notes.jsonl` holds a lock file beside
it (`usage.jsonl.lock`, `notes.jsonl.lock`), kept after use: do not remove it. Another writer waits
up to 10 s for it. Past that, nothing is written to that file: a note asks you to add it again, a
registration stands and says the usage history was not written, and a measurement stops as above
(exit 1) with the command that registers it later.

**New `measure_scorer.py` options.** They are there for M5's job runner, and you can use them
by hand too, in a shell where the GPU lock is configured (§4.11). `--expect-version-sha256` and `--outcome` apply to `--cohort` runs only (exit 2
with `--split`); the other two apply to both:

```bash
python scripts/measure_scorer.py --mode <protocol> \
    --checkpoint <file>.pt --data-dir data/workspaces/<ws> \
    --cohort <label> --registry /srv/shepherd/cohorts [--record] \
    [--deterministic] [--expect-checkpoint-sha256 HEX] [--expect-version-sha256 HEX] \
    [--outcome FILE]
```

- `--deterministic` turns on several settings before CUDA starts: PyTorch's deterministic
  algorithms (never warn-only), deterministic cuDNN with its benchmark off, and
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` when that variable is unset. It never falls back to
  non-deterministic execution or to another device. An operation with no deterministic
  implementation refuses the run with nothing written, and so does a `CUBLAS_WORKSPACE_CONFIG`
  that PyTorch does not document as deterministic. The manifest records the settings, so
  deterministic runs are their own set of conditions. PyTorch promises reproducibility only on the same
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
  run id and digest, the checkpoint and version digests, the measurement mode (`mode`: `C` or
  `D`, since Batch 3), whether the run was deterministic, and under `registration` whether the
  report was registered and why. A run refused before its
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
| `max_file_size="1mb"` | Gradio's per-file upload limit, a second line of defence behind the guard's refusal of the upload route (below; §4.13) |
| `footer_links=[]` | Since M5-2: Gradio's footer (its API page, "Built with Gradio" and its own Settings, with a dark mode the dashboard's colours lack and the screen recorder) is not shown |

**CORS** is tightened since M5-1, together with D8 (§4.13): **by default no cross-origin
browser access is granted; administrative operations are still checked by D8 in the backend.**
The parent app keeps its `CORSMiddleware`, now with an empty allow list (`src/api/main.py`), and
since Gradio 6.25 a mounted app adds no CORS headers of its own when the parent app has exactly
that class, so `/ui` follows the same policy. Until M5-1 the configuration was
`allow_origins=["*"]` with `allow_credentials=True`, which under Starlette 1.9 echoed every
request's `Origin` with `Access-Control-Allow-Credentials: true`: a page on any site, opened in a
browser holding HTTP credentials or a client certificate for the service, could read the
responses of `/api/*` and `/ui/gradio_api/*`. It no longer can.

**Framing** is refused since M5-3: every response, the dashboard's and the API's, carries
`X-Frame-Options: DENY` and `Content-Security-Policy: frame-ancestors 'none'`
(`src/api/middleware/frame_guard.py`). D8 (§4.13) checks where a request arrives from, not the
page it was made from: a dashboard loaded inside another site's frame makes same-origin requests
that D8 allows, so without this an invisible frame laid over a hostile page could turn the
operator's click into Restart Backend or a Load. No page frames the dashboard, its own included.

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
  temporary files and may run ffmpeg. Gradio's Settings panel, whose
  screen recorder posts there, is hidden since M5-2 with the rest of Gradio's footer; a recording
  posted by any other means is refused like the rest.
- **`/ui/gradio_api/upload`** and its progress stream `upload_progress` (since M5-1). The
  dashboard has no upload control, and another site's page can post files there with no
  preflight: Gradio writes up to 1000 files a request into its temporary directory and never
  removes them.

A refusal is `403` with a one-line reason and `connection: close`. The request body is not
read, and the request log shows the refusal like any other response. Local downloads are not
affected: the Diagnosis tab's CSV and Markdown exports download as before, `HEAD` and `Range`
requests included. (Since M5-2 the Training page's Export CSV comes as text through the page's
own server function and the browser saves it; no file is served for it.)

### What else changed in the dashboard

- **Training Console charts** look as before (the y axis starts at 0, each series keeps its
  colour). The Epoch Window now only filters rows; the x axis follows them. (Replaced in M5-2 by
  one zoomable figure without an Epoch Window: §4.14, Research › Prepare › Training.)
- **Runtime Settings** refreshes the restart lock when its tab is opened: Gradio 6 does not run
  a timer in a tab that has not been opened yet. Since M5-3 it then reads it again on the
  dashboard's shared slow timer while the page is shown, and has no timer of its own. The backend
  refuses a restart while any GPU job runs, training or a test run (§4.11), either way.

### Platforms

Tested backends: Linux x86_64 (the development environment) and Linux ARM64 (the homelab,
checked by the reviewer). **A Windows backend has not been tested with Gradio 6.30.0** and stays
marked untested until one is available; Windows browsers are part of the browser checks.

## 4.11 One GPU job at a time: the GPU lock and the job runner (since M5, 2026-10-20)

Managed training and evaluation jobs exclude each other: the jobs the WebUI and the API start,
and `scripts/train_model.py` and `scripts/measure_scorer.py` run from a shell. A job that cannot
have the GPU is refused, naming who has it; nothing is queued.

### The GPU lock

One lock file per deployment, whatever the workspace, named by the workspaces root alone:

| Settings | Lock file |
|---|---|
| `SHEPHERD_WORKSPACES_ROOT=/absolute/root` | `<root>/.shepherd-gpu.lock`: **the root is the lock's one name** (since the root-cause round of 2026-10-22); no other setting names a lock |
| not set | **not configured**: every managed job is refused (since the review of 8242b47): the CLIs exit 1 before anything is read, the service refuses every start, each saying how to configure the lock |

**`SHEPHERD_GPU_LOCK_FILE` is no longer read** (since the root-cause round of 2026-10-22). It
named a lock file of its own, beside the root or instead of it, so a service configured with it
and a shell given only the root each counted as configured, locked two files and ran side by
side (a service with no root lists no workspace, but `POST /api/v1/training/start` takes a
workspace path). Remove it from the service's environment and from your shell; where it was the
only setting, every job is now refused as not configured until the root is set.

**Upgrade only while no training or test run is running** (since round 3 of the review fixes,
2026-10-22), from a shell or from the service: a job started by an earlier build is not seen by
this build's lock. A build without M5's GPU lock mechanism takes no lock at all, and an M5 build
from before the root-cause round may hold the file `SHEPHERD_GPU_LOCK_FILE` named, the old lock
name, which this build does not check. Stopping the service for the upgrade stops the jobs it started; for a CLI started from a
shell, check first with `pgrep -af 'train_model.py|measure_scorer.py|setup_demo.py'` (or
`nvidia-smi`). After the upgrade, remove `SHEPHERD_GPU_LOCK_FILE` from the service's environment
and from your shells.

A root that is set but unusable (a relative path, a directory that does not exist, a path under a
directory the service account may not search) refuses every job rather than running it unlocked,
and the status shows why (`gpu.lock.problem`).

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

**Every refusal of `train_model.py` is one line** (since K8-7b, 2026-11-08): whatever it refuses
before it claims its run directory is said as `train_model: refused: <why> Nothing was
written.`, exit 1, with no traceback (it is logged at `--debug` only). Among them: a run without
`--stage` (every run names its stage, a continued one included: `--resume <checkpoint>` with the
`--stage` of the run it continues), a run directory that already holds files, and a
`--config` file naming a setting this version does not have ("The configuration … names
settings this version does not have: num_workers."; since K8-7, `num_workers` and the old model
settings such as `conv_type` and `hidden_dim` are such settings, and `--conv-type`,
`--hidden-dim` and `--num-layers` are gone). An error after the claim is a failed run, logged as
"Training failed:" with its traceback, and so is an unexpected error before it, one that is no
refusal.

**Set the root for the service and in the shell where you run the CLIs** (for example, the
`Environment=` lines you set in the systemd unit (§4.12; its template has a commented line for
the root and one for the registry), which reads no environment file of its own, or one file that
both the unit's `EnvironmentFile=` and your shell read). A CLI that sees no root
is refused, exit 1, with nothing written (`train_model: refused: SHEPHERD_WORKSPACES_ROOT is not
set, so the GPU lock is not configured ...`); so is `setup_demo.py --train-model`, which trains
through the same trainer. **Give the shell the service's workspaces root**: every process that
names the root takes `<root>/.shepherd-gpu.lock`, the service, your shell and its CLIs alike,
and nothing else names a lock. What one environment variable cannot settle, since each process
reads its own: a process that names another root takes another lock.

**`SHEPHERD_REGISTRY` is required in that shell as well** (since the M5 review, BACKLOG 37):
`measure_scorer.py --cohort`, `--record` and `record_evaluation.py --report` work only in the
registry it names and refuse while it is unset (§4.9). An `EnvironmentFile=` is not read the way
a shell reads it: systemd keeps interior whitespace and keeps quotes that do not open the value
(systemd.exec(5)), so a line such as `SHEPHERD_REGISTRY=/srv/my cohorts` gives the service and a
shell that sources the file two different values. Use a path without spaces or quotes. A
difference is refused, never guessed: the refusal shows the value as set and the path it
resolves to.

The holder describes itself in `<lock file>.json` (job type, command, pid, host, start time)
while it holds the lock, for the message above. That file is descriptive only: the lock is the
`flock` on the lock file. One left behind by a killed job is harmless: the next holder removes
it as soon as it holds the lock, and a description whose process has exited is never named as the
holder (the refusal says instead that the lock is held by a process that inherited its descriptor,
or by a new holder that has not described itself). **Never delete the lock file while a job
runs**: a new file would be a second lock. Git ignores it and its description wherever a root in
the checkout puts them (since round 3 of the review fixes), so `git clean -fd` and `git stash -u`
leave them alone; `git clean -x` and `git stash -a`, which remove ignored files too, still delete
it, so do not run them in the deploy checkout while a job runs.

What the lock does and does not guarantee:

- **It is released however the holder ends**: a normal exit or an error releases it, and after
  SIGTERM or SIGKILL the kernel does, so there is no stale lock to clean up after a crash.
- **It is cooperative.** A GPU program that does not take it is not stopped by it. The inference
  service does not take it and may hold GPU memory; a test run that cannot get memory fails as a
  run, is reported, and registers nothing.
- **Inherited descriptors.** A program a job starts does not keep the lock (the descriptor is
  close-on-exec). A process the job forks without starting a new program — a worker pool under
  the `fork` start method, for example — shares it: a job that ends normally releases it
  for that process too, but when the job is killed (or exits without releasing), that process
  keeps the lock until it exits; the refusal then says the recorded holder has exited. **A job
  the service starts leads its own process group** (since the review of 8242b47), and a stop and
  the runner end the whole group, those processes included (below). A training CLI
  (`train_model.py`, `setup_demo.py --train-model`) forks no workers (since K8-7, 2026-11-08: its
  stage samplers draw batches in the main process, and the DataLoader workers it used to fork
  went with the ShepherdGNN loader), so ending its main process, by any signal, releases the
  lock; `kill -TERM -- -<pid>` ends the whole group when the CLI leads one (a CLI started in an
  interactive shell does, through job control). `measure_scorer.py` forks no workers either.
- **Keep the workspaces root, where the lock file is, on a local filesystem.**
- **Windows** uses `msvcrt.locking` on the file's first byte; a Windows backend is untested.
  There a child process is spawned, not forked, and does not inherit the lock's handle; the
  system's release of a dead process's file lock can be delayed; a stop sends CTRL_BREAK_EVENT
  to the job's console process group, and a backend with no console, which cannot send it, goes
  on to TerminateProcess, which ends the main process only. A Job Object, the true equivalent of
  a process group, is not added while the Windows backend is untested.

### The job runner

The service behind the Training Console and the training API now runs two kinds of job, each its
own script: training (`train_model.py`) and test runs (`measure_scorer.py` on a registered
cohort version, the same CLI an operator runs). One job at a time:

- **A job that is running or still stopping holds the GPU**, whatever its type. A training start
  while a stop is still waiting for its process is now refused (it used to be accepted, and could
  start a second process beside the one being stopped).
- **A stop reaches only its own job type**: a training stop never stops a test run, nor the
  reverse. It sends SIGINT to the job's main process (its graceful stop), then SIGTERM to the
  job's whole process group after 30 s, then SIGKILL to the group after 10 s more, and answers
  once no process of the job is left: each job leads its own process group, so any process it
  forked without exec, which shares the GPU lock, ends with it, and once the main process has ended
  the runner ends what is left of the group before it says how the job ended and frees the slot
  (since the review of 8242b47: a worker used to outlive a stop that said "stopped" and keep the
  lock). A worker left as a zombie (a container whose PID 1 never reaps) holds nothing and does
  not keep the slot. `POST /api/v1/training/stop` no longer holds up the rest of the API
  and the dashboard while it waits. **A stop that arrives after the process has ended on its own**
  (while its result is still being read) is refused (`409`, "had already ended on its own ...;
  nothing was stopped") and changes nothing: a failure stays `failed`, with its message.
- **How a job ended:** training `completed`, `failed` or `stopped` (as before); a test run
  `completed`, `failed` or `cancelled`, `cancelled` only when the cancel is what ended it (a run
  that ended on its own as the cancel arrived keeps that ending). A failure caused by a signal
  says which (`Process ended by SIGKILL`, for the out-of-memory killer). **A cancel that lands
  while a test run's interpreter is still starting is a cancel too** (since K8-13, 2026-11-08): on
  POSIX the runner starts a test run with SIGINT held, and `measure_scorer.py` takes it as soon as
  its own handler is in place, so the run says "measure_scorer: cancelled" and ends by SIGINT. It
  used to end the start with exit 1 or 2 ("Fatal Python error" while Python imported `site`) and
  be recorded `failed`. Training runs are not started this way.
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
  (its note now says "A GPU job (training or a test run) is in progress"). After its short delay
  it closes the job runner's GPU slot, only if no job holds it and in the same step as that
  check, and keeps it closed until the process is replaced: a job started during the delay
  aborts the restart, and a start asked for after that is refused, never orphaned by the restart
  (since the root-cause round of 2026-10-22). A stop of the backend begun before the process is
  replaced (Ctrl-C, `systemctl stop`) aborts the restart too: its signal is recorded before
  anything else, and the restart's last check and its exec are one step (since round 3 of the
  review fixes). One case this self-exec restart cannot close: a stop signal that arrives as the
  exec is made, before its handler has recorded it, is lost with the old process, and the new one
  serves on; under systemd the unit then SIGKILLs the whole service, a job started meanwhile
  included, after `TimeoutStopSec`, and in a terminal a second Ctrl-C stops the new backend. It is
  a limit of the restart replacing its own process image: a restart made by a supervisor could
  close it, and none is built (BACKLOG 47).
- **Stopping the backend stops its job** (the owner's decision of 2026-10-22). Because a job
  leads its own process group, a Ctrl-C, a Ctrl-\ or a hangup of the terminal the backend runs
  in, or a SIGTERM sent to the backend alone, no longer reaches the job itself; the backend stops
  it instead, as a stop above does, starting when the signal arrives (a Ctrl-\ is routed there
  too since the root-cause round of 2026-10-22: it used to leave the job training with the lock;
  the backend still ends by it, after the job's stop). **The process that was started is the
  server on every POSIX start path** (since round 3 of the review fixes): `launch_shepherd.sh`
  execs the launcher, and the launcher execs into uvicorn, in its browser mode too (the browser is
  opened by a helper of its own), so a SIGTERM sent to the PID you or a supervisor started is the
  server's; a launcher waiting beside the server used to end alone, and the server served on with
  its job. On Windows the launcher waits beside the server, and the stop goes through the
  console's Ctrl-C or Ctrl-Break. Under systemd the unit's
  `KillMode=mixed` sends the stop's SIGTERM to the server alone (the launcher execs into it),
  which stops its job the same way, and systemd SIGKILLs what is left only once the server has
  exited, within `TimeoutStopSec=150`, which covers the backend's longest stop of its job; the
  unit used to SIGTERM the job's processes directly, ahead of the backend's graceful stop. When
  the server crashes, systemd SIGKILLs the job's processes. **From the signal on, the job
  runner's GPU slot is closed**: a start still under way (a request in flight, a dashboard event
  already queued) is refused ("The backend is stopping; nothing was started."), so no job can be
  running while the backend waits. Open dashboard tabs do not hold this up: uvicorn waits for
  every open connection before it finishes shutting down, and each tab keeps one open (its
  session heartbeat), so the backend ends those streams at the same signal. Another connection
  that does not close (a client that never finishes its request) keeps the backend waiting
  ("Waiting for connections to close") with the job already stopped; a second Ctrl-C ends that
  wait, and the backend still waits for the job's stop before it exits. A backend that is killed (SIGKILL, the
  out-of-memory killer) leaves the job running and holding the lock; the GPU chip names it; stop
  it with `kill -INT <pid>` or `kill -TERM -- -<pid>`. **Do not run the backend with
  `--reload`**: uvicorn's reloader restarts the server on every saved source file, and each
  restart's shutdown would stop a running training. The commands `setup_demo.py` prints have no
  `--reload`, and `python -m src.api.main` no longer reloads.
- `GET /api/v1/training/status` adds `job_id`, `end_time`, `exit_code` and `gpu`: the job holding
  the GPU, of either type, and the lock (`configured`, `path`, `held_by` with the holder's own
  record `holder` and `holder_alive`, `problem`, and `unset` when no workspaces root is set), the
  same reading the GPU chip shows (the `source` field went with `SHEPHERD_GPU_LOCK_FILE`: the
  root is the only one).
  A test run holding the GPU shows there, never as training.

Since M5-2, Research › Test starts and cancels test runs from the dashboard through this runner
(§4.14); no API route starts one. The exit codes of both CLIs are unchanged: a busy lock is a refusal before anything was written.

---

## 4.12 Server settings and the server's lists (since M5-1, 2026-10-20)

M5's controls act on what the server lists, never on a path a request names (`docs/working/PLAN_TEST_RESULTS.md` D7 and D8; D8 decides who may trigger an operator control, §4.13). What the
server lists, and the deployment mode its operator controls depend on, come from **environment
variables of the service**. Set them where the service starts (the shell that runs
`launch_shepherd.sh`, or `Environment=` lines in the systemd unit) and restart it. Each is read
when it is used. No request, form or saved UI file can set one. A missing or refused setting
turns off what needs it and says why. **Nothing falls back to a default directory**, the
repository's `data/workspaces/` included.

| Variable | Value | Unset or refused |
|---|---|---|
| `SHEPHERD_DEPLOYMENT_MODE` | `M1`, exactly (`m1`, ` M1` or `M2` are other values) | read-only: every operator control is refused (D8, §4.13) |
| `SHEPHERD_WORKSPACES_ROOT` | absolute path of an existing directory; its subdirectories are the workspaces, and `.shepherd-gpu.lock` directly under it is the GPU lock (§4.11) | no workspace is listed, and no GPU lock: every training and test run is refused |
| `SHEPHERD_REGISTRY` | the cohort registry (§4.9), the directory the commands take as `--registry`: absolute, existing, outside the repository; the only registry reports are registered from and `--cohort` runs measure in, for the service and the commands alike | no cohort version is listed; every registration and `--cohort` measurement is refused |
| `SHEPHERD_INTAKE_DIR` | an existing directory outside the repository where you place `shepherd-cases/1` files to import; absolute | no import source is offered |
| `SHEPHERD_TRAINING_EVIDENCE_DIRS` | optional: R10 evidence directories, as `--training-evidence` takes them, separated by `:` (`;` on Windows); each absolute and existing | none; an entry known not to be a directory refuses the whole setting; one that cannot be checked is kept and noted |

```bash
export SHEPHERD_DEPLOYMENT_MODE=M1
export SHEPHERD_WORKSPACES_ROOT=/home/you/SHEPHERD-Advanced/data/workspaces
export SHEPHERD_REGISTRY=/srv/shepherd/cohorts
export SHEPHERD_INTAKE_DIR=/srv/shepherd/intake
```

- **`M1`** is SPEC_4's single-workspace deployment, reached on the machine or through SSH
  forwarding, operated by the owner or a designated technician (`docs/working/PLAN_TEST_RESULTS.md` §4.9 question 3). It is one of the
  conditions an operator control checks, not an authorisation. D8 checks each request it handles
  for the others. Do not set it behind a reverse proxy or on a machine other people can log in to.
- **The deployment mode is applied by D8** (§4.13), on every operator control at the moment it
  handles a request: unset or any other value, each one is refused and the deployment is
  read-only for them. Diagnosis, the status pages and the read-only lists work either way.
- **The workspaces root may be inside the repository**, as the usual `data/workspaces/` is, but
  only when you name it. A relative path, a missing directory and a file are refused.
- **The GPU lock follows the workspaces root** (§4.11): the lock is `<root>/.shepherd-gpu.lock`,
  the lock's one name; `SHEPHERD_GPU_LOCK_FILE` is no longer read. A shell where you run the
  formal CLIs needs the same root, and with no root set every training and test run is refused.
- **The registry and the intake directory hold patient-level data.** They follow the
  registry's rule (§4.9): a relative path, a missing directory, and a path that resolves inside
  the repository (through a symlink too) are refused. `--registry` on the command line and
  `SHEPHERD_REGISTRY` are checked by the same function. Create both where only the accounts that
  run the service and the commands can read them, for example `mkdir -m 700`.
- **A directory the service account cannot reach** (a parent it may not enter) is refused as
  "cannot be checked", with the error, and its feature is off; an evidence directory is the
  exception (next item). Inside the registry, a `versions/` the service cannot read refuses the
  cohort list with that reason; a version, a `reports/` or a run directory it cannot read is
  listed with the reason, and the rest of the list stands.
- **Evidence directories**: an empty entry (a doubled or trailing separator) is skipped, never
  read as the working directory. A repeated one is kept once. One the service account cannot
  check (a parent it may not enter, an I/O error) is optional evidence, so it is kept: the Test
  page names it in amber, and Start stays enabled. R10 cannot look in it: a scope found
  elsewhere stays found, and one found nowhere else reads `unverifiable`, naming it. A relative
  path, a missing directory and a file still refuse the setting.

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
- **A workspace's checkpoints**: the run checkpoints of one listed workspace, latest run first,
  from every directory under its `runs/` whatever the directory's name (since K8-7b, 2026-11-08;
  before, only the layout's architecture directories were read, so a model under another name
  was listed nowhere), or from one directory's. What a file is comes from its own record, never
  from its directory. Each file is read once, weights only, and shows:
  - the SHA-256 of the bytes read;
  - the Trainer's validation figure: the stage's selection metric the Trainer logged (a base's
    LP accuracy, a task model's pooled MRR), read from the file's logs whatever its record says,
    never a test result; none when the file was not read or its logs hold neither (a file
    without a model spec logs `val_mrr`, which is not read);
  - its model record's summary: `record` is `model`, or `refused` with the reason (a file
    without a model spec: "written before the model spec: not readable by this build; start a
    new run"), or null when the file's record was not read (its `problem` says why); since
    K8-7b there is no `history` record value, History group or `history` filter (`?task=` and
    `?learning_stage=` still keep the files whose record names them);
  - the dataset fingerprint the checkpoint records, and whether it is this workspace's now;
  - for a file that cannot be read, why (a link whose target is gone included). No file is
    dropped;
  - for a file that resolves outside the workspace (a run or a checkpoint that is a symlink out of
    it), that reason only: it is never opened, so it shows no digest, size or record.

  A directory under `runs/` that cannot be listed is one item with its reason, and every other
  run is still listed; only `runs/` itself failing fails the list. A directory whose name is not
  valid text is such an item, its name written with backslash escapes (`runs/bad\xff`; since
  the fix after K8-7b). So is a run directory or a checkpoint file whose own name is not valid
  text (since research comparison Stage 7, BACKLOG 66): one item with its reason ("… is not
  listed (its name is not valid text, …), so it is not read"), never opened, chosen or found by
  name; every other file is listed, and this list and the one below answer. Each item carries
  `key`, its identity apart from the names shown, made of its own names under `runs/`:
  `architecture/run/file` for a file whose names are text, two names for a run directory and one
  for a directory directly under `runs/`, and for a name that is not text a form no text name
  gives (its bytes in hexadecimal), so a name shown with backslash escapes never takes the place
  of a file whose name really reads so, and no two items share a key. `run_key` is its run's
  identity the same way (`architecture/run`), by which a list groups its items, so two runs
  shown alike are two groups. Rename such a file to use it.
  Each item names the workspace as the list does: for
  `current -> builds/2026-10-20`, it says `current`. Listing opens every checkpoint, which SPEC_4
  classes as checkpoint inspection (C3), and D8 offers that only where the operator controls
  are: it is served as `GET /api/v1/workspaces/{name}/checkpoints` (`?architecture=<directory>`
  for the runs of one directory under `runs/`, matched by its exact name; a name no directory
  has lists nothing) behind D8's check (§4.13), and a refused request opens nothing.
- **The existing checkpoint list** (`GET /api/v1/training/checkpoints`; since M5-2 the
  Training page's parent list is the per-workspace list above) reads each file the same way,
  inside the workspace: since K8-7b a file reached through a link out of it is listed with "resolves
  outside the workspace, so it is not read", as the server list has said since K8-1r (it used to
  be read and hashed). Each entry gains `sha256`, `validation`, `dataset_fingerprint` and
  `metadata_problem`, all from the one read. Before, it loaded each
  file without keeping its digest, and dropped a load error without a word. A file that is gone
  by the time it is read (a link whose target was removed) has no size or time to show and is
  left out of this list; it no longer fails the whole list. A directory under `runs/` that
  cannot be listed is kept, with its reason; its size and time are the directory's own, whatever
  its name. Since M5-1 D8 checks it like the controls (§4.13).
- **Cohort versions** (`GET /api/v1/cohorts`): each version in the registry, by label and the
  SHA-256 of its `cohort.json`, with its role. It also shows
  its counts (a KG edge set's and a generated split's with the units its manifest records,
  `count_units`, since research comparison Stage 7), the graph and dataset record it is bound
  to, and when it was imported. The list
  never shows a case identifier or the exclusion list: D7 keeps those in the registry, on the
  machine. An interrupted import is listed with its reason. Each version's report store is
  listed by run id, and each run says whether its report is kept and whether it is registered
  (`kept`, `registered`, each with its reason, the report's digest and the record's sequence),
  from the registration's own reading of the store and of the ledger beside the checkpoint the
  report records it measured. `registered` is `registered` when that ledger names the report;
  `not_registered` only when it names this run with other bytes (`replaced_after_registration`
  true: the report was replaced after it was registered, and the reason names the record and
  both digests); and otherwise `cannot_be_checked`: a registration filed beside
  another copy of the same weights (`record_evaluation --checkpoint` takes any file with the
  measured bytes) is not looked for, and the reason names the ledger that was. A report that
  names its checkpoint by a relative path, or that cannot be read, has no ledger to look in:
  `registered` is then `cannot_be_checked` too, saying why. A checkpoint moved since the run is
  looked for where the run read it. `registered` means a record names **this run** with these
  bytes (since K8-6r.2): another run's byte-identical report is not this run's registration. A
  run whose bytes only a record written before each run had its own record holds is not
  registered, and its reason says so ("its bytes equal a report merged into record N …"). Each
  run with a `report.json` also carries how its report's conditions compare in the store
  (`contest`), the same description Research › Test shows for that run, without the reasons'
  own words, which may name a server path. Since K8-6r.2 these values only describe, and the
  strings stayed for API clients: `uncontested`; `contested` (runs under these conditions
  differ, with the fields and every run that carries a differing report, byte-identical runs
  each named); `cannot_be_checked` (the comparison is incomplete, never to be read as agreeing).
  `contest` is `null` for a run directory with no `report.json` (an interrupted, cancelled or
  refused run) and for a run whose state was not read (its `problem` says why). A run whose
  report cannot be read, or whose conditions cannot be told, is `cannot_be_checked` with its own
  reason only, and marks no other run. A report that can be read but not compared makes its
  group's comparison incomplete unless the group's runs already differ (then `contested`). So
  is every run of a store that cannot be scanned, and a run whose report was replaced between
  the store's scan and the read of its state: a run's state and its `contest` always describe
  the same bytes. Each request scans every listed version's report store once.

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
service on the machine, or from elsewhere through SSH forwarding (`docs/working/PLAN_TEST_RESULTS.md` §4.9 question 3). Three things
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
| API | `POST /api/v1/pipeline/reload`, `POST /api/v1/pipeline/config`, `POST /api/v1/training/start`, `POST /api/v1/training/finetune`, `POST /api/v1/training/stop`, `GET /api/v1/training/checkpoints`, `GET /api/v1/workspaces/{name}/checkpoints`, `GET /api/v1/workspaces/{name}/family` |
| Dashboard | Diagnosis: Change model… (its list of checkpoints and its Load; since M5-2). ⚙ Settings (Runtime Settings): Restart Backend and Apply Settings. Research › Training: Start, Stop (its confirmation reads the run's last checkpoint), Continue… (it lists the parents), the parent list's Refresh, Check checkpoint and Continue Training |

The three checkpoint lists (the family view among them), and Check checkpoint, open checkpoint files, which SPEC_4 classes as
checkpoint inspection (C3), so they are checked like the controls. So are the reads that open a
checkpoint for display: the Diagnosis card's "trained on" line (the served checkpoint) and the
Training page's run header (the run's checkpoint). A read-only browser is told "not shown ·
reading the checkpoint is an operator control" on the card, and "dataset not shown · operator
control" in the run header. Research › Test (§4.14)
uses the same check for its import (the intake list, Preview and Commit), Start test, Cancel,
Register this report, Add note and its model list, and Research › Models for its checkpoint
list, its Load and its Results segment (reading, showing, comparing and exporting the results,
since research comparison Stage 8b; §4.14). Everything else stays open:
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
dashboard's Load (Change model… and Research › Models) runs the API's reload function in
process, where the API's reload runs, so it waits behind a running diagnosis as before. A
reload still running after 180 s is no longer shown as failed: the page says "The page stopped
waiting after 180 s; the server may still finish the reload." with Read again, and "The reload
finished after the page stopped waiting." if its answer arrives later.

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
not, turned every later diagnosis from the demo answer (removed since K7) into "configured but
not served" (503).

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
  `http://127.0.0.1:<port>`; a script that starts its own service for that sets
  `SHEPHERD_DEPLOYMENT_MODE=M1` in its environment.

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
| **Diagnosis (clinical)** | Patient phenotypes, the results and the selected candidate's details, with the model in use on a card (below). It is the first screen |
| **Research (pre-clinical)** › Prepare › **Datasets** | The server's workspaces, read-only (below) |
| **Research (pre-clinical)** › Prepare › **Training** | The run bar, the parameters for the next run, Continue… and the run view with its charts (below). It is the research page shown first |
| **Research (pre-clinical)** › Prepare › **Fine-tune** | A base's task models (since K6): one listed base, one to three tasks (Disease, Gene, Patients-like-me), each trained as its own run in order, and a job card with each task's status, Stop and products |
| **Research (pre-clinical)** › Evaluate › **Test** | Import a cohort version from the server's intake directory, run a test, keep and register its report (below) |
| **Research (pre-clinical)** › Evaluate › **Models** | The server's checkpoints of a workspace, their recorded test results, newest first or by family, and Load (below); the results of a model's family or the workspace, compared and exported (Results, since research comparison Stage 8b) |
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
| `GPU job: train_model.py (pid 41822, since 09:58 UTC)` | A process this service did not start holds the GPU lock (a formal CLI, or another server's job runner on the same lock); the chip says what its holder record says: the command, the pid and the start |
| `GPU job: lock held, holder not described` | The lock is held, but no running process is described as its holder |
| `GPU jobs: refused, lock not configured` | No GPU lock is configured (`SHEPHERD_WORKSPACES_ROOT`, which names it, is not set), so every training and test run is refused until it is set (§4.11) |
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

### Diagnosis (clinical)

The first screen, in three panels: the patient's phenotypes, the ranked candidates, and the
selected candidate's details (Evidence · Explanation · Matching phenotypes · Paths). Selecting a
row (a click, or Enter on a focused row) shows that candidate in the details panel. In a window
at least 1400 × 760 CSS pixels each panel scrolls inside itself; narrower, or in portrait, the
three panels stack and the page scrolls.

**Patient ID** left blank is sent as no id: the server gives the result its own (`patient_…`), so
results from different browsers never share one (before M5-2 a blank field was sent as
`webui_patient`).

**The "Model in use" card** is the last item of the input panel, with Run Diagnosis right below
it: both follow Options, and stay pinned at the panel's bottom when the inputs need to scroll (at
1536 × 760, for example). It shows:

- the title row "MODEL IN USE ⓘ · checked HH:MM:SS ↻": what is served is read for this browser
  when the page opens, every 5 seconds while Diagnosis is shown, after each diagnosis, after a
  model change and with ↻; a read that takes more than 5 seconds is "timed out (5 s)";
- the model, "GAT · run 2026-10-17 09:15 · epoch 5", and "model fingerprint <12 hex> ✓ verified"
  (the SHA-256 of the checkpoint file served);
- "trained on <workspace> · dataset fingerprint <12 hex>": the dataset the checkpoint records it
  was trained on (`training_input_digests.split_manifest`), named by the listed workspaces with
  that full fingerprint, or "matches no listed workspace"; never the served workspace's own
  manifest. Reading it opens the checkpoint file, an operator control (§4.13): a read-only
  browser sees "not shown · reading the checkpoint is an operator control". When another browser
  has changed the model, this card says "not read yet · read it" until the page is opened again
  or "read it" is clicked (the 5-second read never opens a checkpoint);
- a chip, only when true: `synthetic data` (amber). The `no trained model` and `unverified`
  chips went with the path-reasoning state (K7), and `no SP table` and `GNN not loaded` with the
  mixture (K8-4);
- "Recorded results: <state>" (§4.9), whose › opens Details at Evidence; "Details ›"; "🔒 Change
  model…". The recorded tests, and the workspace names, are read again each time Diagnosis is
  shown, after each diagnosis, after a model change and with ↻ (the 5-second read reads only what
  is served): a test recorded in Research › Test shows when you come back to Diagnosis.

| The card shows | Meaning |
|---|---|
| "Model changed at HH:MM — was <old>, now <new>" with × | A read found another model, or another part of what is served, than this browser's previous read (another browser's Change model, for example). × dismisses it |
| grey: "cannot check · <reason> · Retry", "last known HH:MM:SS · not current" | What is served could not be read: "status not read" (the full reason is in the badge's line and in Details) or "timed out (5 s)". The lines shown are the last known state, without ✓, with grey dashed chips |
| grey: "read HH:MM:SS · older than this result" | The last good read is older than the result shown |
| "No model loaded" and "🔒 Load a model…" | No model is served: none configured, a graph named without a model, or a build that failed (since K7 a graph is never served on its own). Run Diagnosis is greyed out and cannot be clicked: pointing at it, or reaching it with Tab, shows why, and "Disabled: no model loaded. …" is written under it. An API client asking now gets 503 with the reason (§4.7) |
| "🔒 Change model…" greyed, "Operator controls only: <reason>" | A read-only browser (§4.13). It never sees the list of checkpoints |

**The badge** above the results (§4.7) compares the result's whole served identity with the card's
last read, and the card is read again after each diagnosis; a read that failed, or one older than
the result, never shows "Produced by the model in use". The CSV and the report keep the result's
own identity whatever the badge says. "Not a certified medical device; decision support for a
qualified clinician." is at the foot of the results.

**The identity drawer** opens over the details panel, with two views. "Model in use" is the
served model's facts in five sections (Identity, with the full digests and Copy; Uses; Data;
Evidence; Warnings, always open) and a link to the full records in Research › Models (the
structural-fingerprint line is gone since K8-4: the checkpoint's record binds its graph by
digest, checked at load). "This result" shows only what the result carries, each part beside
what is in use now. "Details ›" and "Recorded results ›" open the first, the badge the second; while
a model change is open (its confirmation, or a reload running) they leave the Change model drawer
in place.

**🔒 Change model…** (an operator control) opens a drawer over the results and the details: the
workspace select (the server's list; the serving workspace is chosen first), "In use now" on top,
the latest four runs ("Show older runs" for the rest), the selected checkpoint's facts, "Compare
in Research › Models ›", and "🔒 Load this model" / Cancel. Load asks once ("Reload the pipeline
with <12 hex> for every session?", from → to, with the changed rows marked) and reloads the
pipeline with the workspace and checkpoint the server lists. **The reload serves only the model
the question named**: it compares its own read of the checkpoint with that fingerprint before
anything is published, so a file rewritten in the meantime (a running training job rewrites its
`last.pt` every epoch, and the reload reads the graph first) changes nothing, and the outcome is
"Not reloaded; the model in use is unchanged." with the reason. The API takes the same check as
an optional field of the reload request, `expected_checkpoint_sha256` (64 lowercase hex digits,
else 422): other bytes are refused and the running pipeline goes on serving; without it nothing
is checked, as before. It replaces what every browser is served; results already produced keep
their own identity, and diagnoses wait while the pipeline reloads. While the confirmation is
open the list's own footer is hidden, so the from → to rows and that sentence stay in view;
"Keep <old>" drops the confirmation. While the confirmation is open, Run Diagnosis is disabled:
"Disabled: finish or cancel the model change first." Once "🔒 Reload with <new>" is clicked, the
confirmation shows only the reload's status ("Loading <12 hex> for every session… 00:00:42") and
Close (closing does not stop it; after 180 s, Read again and Close), so a second reload cannot
be asked for; the card says the same "Loading …" line and Run Diagnosis says "Disabled: a model
load is in progress; diagnoses wait until it finishes." Cancel or × closes the drawer without
reloading. While the reload runs, Close or × closes the drawer at once and the reload goes on
(it cannot be stopped): the card keeps its "Loading …" line, and when the reload answers, the
card shows the model then in use and its outcome is not shown again; "Change model…" meanwhile
opens the drawer again on the reload's status. In Research › Models, Close closes the
confirmation the same way. A Load from Research › Models refreshes this card too. Only one
drawer is open at a time.

**The reload request** (`POST /api/v1/pipeline/reload`, the API under the dashboard's Load)
declares five fields: `data_dir`, `checkpoint_path`, `task`, `device` and
`expected_checkpoint_sha256`. It names a task or a `checkpoint_path`, or both: a task alone
serves the latest run holding a model finetuned for it, a path serves that file (with a task,
only if the model is that task's), and a request with neither is refused, naming the missing
task or path, before any checkpoint is read (since K8-4). **Any other field is refused with
422** (`extra_forbidden`, naming the field) before anything is read or selected, so a field the
service does not declare is never dropped and the rest served. *Old scripts* (since K8-4):
`conv_type` is gone, because the architecture is read from the checkpoint's model spec. Drop
it, whatever its value, from a request that names a path, a task or both. A task with
`conv_type` `"hgt"` asked for an architecture no field names now: once the multi-architecture
port makes HGT servable, name a spec-confirmed HGT checkpoint by `checkpoint_path` (and
`expected_checkpoint_sha256`); until then this build serves no HGT model. And spell
`expected_checkpoint_sha256` exactly: a misspelt `expected_checkpoint_sha` used to be dropped,
leaving the reload unbound, and is now refused. **Automatic selection reads only files inside
the workspace**, judged on the file each of its reads actually opens (the server asks the open
file where it lies, not its name), so a selected file replaced by one outside the workspace
before it is read is refused, nothing is read from it or published and the model in use stays;
where the server cannot tell where an opened file lies, automatic selection refuses it; an
explicit `checkpoint_path` is read wherever it is (since K8-4r). Since K8-7b (2026-11-08) it
reads the runs of every directory under `runs/`, whatever the directory's name, so a model filed
under another name is found and served; a directory under `runs/` it cannot list is named first
in its reason and among what it did not read, never passed over in silence. The response's
`architecture` is the loaded model spec's encoder kind, for a task or a path alike; a run
directory's name only locates the file, and the reason names the run as `<directory>/<run>`, the
directory under `runs/` that holds it (since K8-4r; any directory's name since K8-7b).

### Research › Prepare › Datasets

A read-only list of the workspaces the server lists under `SHEPHERD_WORKSPACES_ROOT` (§4.12):
the same list Training, Fine-tune, Test and Models choose from. Each row shows the name; when it
was built (UTC, from `workspace_build.json`), or "build time not recorded" with its sources'
declared releases; the **dataset fingerprint**, the first 12 hex digits of the SHA-256 of its
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

**Use for training ›** opens Research › Training with that workspace chosen in its run bar, for
this browser tab only. Nothing changes on the server; Start is still the control D8 checks. A
workspace that is not trainable cannot be used, and the line beside the button says why ("Not
available: scratch_ws is not trainable — no split manifest."). Choosing another dataset in the run
bar moves the "in your Training run bar" marker to it.

Nothing on this page is an operator control: it reads the workspaces' small JSON files, never a
checkpoint, so a read-only browser sees the same page. With `SHEPHERD_WORKSPACES_ROOT` unset,
the page says so. Datasets are still built on the server with the CLI
(`scripts/build_knowledge_graph.py --generate-samples`; the demo with `scripts/setup_demo.py`).

### Research › Prepare › Training

The Training Console is now one page in three parts: the run bar across the top, the parameters
for the next run on the left (or the Continue… drawer in their place), and the run view on the
right. How training runs did not change; what changed is how the page picks the dataset and shows
a run.

**The run bar** never scrolls:

- **Dataset**: the server's workspace list, the same as Datasets'; each option is the workspace's
  name (one that cannot be trained also says why), its identity said once, by the chip beside the
  select. Choosing one reads its manifest and record, and the chip shows when it was built, the
  dataset fingerprint
  (Copy gives all 64), `synthetic`, whether the record is verified and **Details ›** (what the
  manifest binds, the seeds and counts, the sources, the path relative to the workspaces root,
  **All datasets ›**). The page opens on the workspace Datasets' **Use for training ›** chose,
  else this tab's earlier choice, else the running job's, else the one serving reads (when it is
  trainable), else the first trainable one. While a training job runs, the select is disabled and
  shows that job's dataset: "in use by the running job".
- **The status pill**: `Idle`, `Starting`, `RUNNING · epoch 12/100 · batch 340/812 · 00:14:32`,
  `Stopping`, `Completed`, `Stopped` (grey) or `FAILED` (red). A failed or stopped run is never
  green.
- **Run name**, then **one reason line** saying why an action is disabled: "Stop disabled: no
  training is running.", D8's reason for a read-only browser ("Operator controls only: request
  not local"), or the GPU job holding the slot ("GPU job: test run (…) · Open Test ›",
  "GPU job: train_model.py (pid …, since …)").
- **🔒 Start, 🔒 Continue…, 🔒 Stop.** Stop asks first: "Stop training now? The job stops. The
  checkpoints it has saved are kept and progress not yet saved is lost; if it has saved none yet,
  this run cannot be continued. Once it has stopped, the page says where it can be continued
  from." with **■ Stop training** and **Keep training**. The question names no epoch: the job
  trains on while it is open, so an epoch read when it was asked would be out of date by the
  answer; once stopped, the notice names the checkpoint the stop left. The question belongs to the job it was
  asked for: if that job ends before you answer, it closes, and a job started later never shows
  it; **■ Stop training** answered for a job that is no longer the one running stops nothing
  ("Not stopped: …; click Stop again").
- **Start sends the dataset fingerprint the chip showed**, and the job runner reads the manifest
  again before anything else: if the workspace was rebuilt in the meantime, nothing is written or
  started, and the page says "Not started: The dataset changed since it was shown; select it
  again. Its split_manifest.json holds … now (shown: …). Nothing was started." The API takes the
  same check as an optional query parameter,
  `POST /api/v1/training/start?expected_dataset_fingerprint=<64 hex digits>`: another manifest is
  refused with 409, a value that is not 64 lowercase hex digits with 422 before the service is
  reached; without it nothing is checked, as before. **The training process checks it again on
  its own manifest read** (`train_model.py --expect-dataset-sha256`, since the review of
  8242b47), so a workspace republished after the runner's check is refused there too, before the
  graph and the samples are read: the run fails with `train_model: refused: The dataset changed
  since it was shown; select it again. …` in its output, exit 1, and no run directory. Both
  checks are the manifest reader's own, made before the file is decoded (implemented in the
  root-cause round), and the process checks nothing about the workspace before it, so a
  manifest that no longer parses, or is gone (its whole workspace included), is refused in the
  same words; a manifest that is the one shown but cannot be read now is refused as such.
- **Over HTTP, `POST /api/v1/training/start` is this form's start** (since K7): a strict base
  start. `learning_stage` must be `base`; it takes the run (`data_dir`, `run_name`, `device`,
  `resume_from`, `extend_epochs`) and the reference recipe's base settings, each unset by default
  so the recipe fills it. A finetune stage, a ShepherdGNN setting (`conv_type`, `hidden_dim`, …)
  or any field it does not define is refused with 422, and continuing a checkpoint that is not a
  base with 409, before a GPU slot or a run directory is taken. Task models are started with
  `POST /api/v1/training/finetune`, as the Fine-tune page starts them; its `settings` take the
  finetune recipe's settings, `frozen_parts` among them since research comparison Stage 9b
  (`graph_side`: the frozen-backbone control, below; the Fine-tune page does not offer it this
  round). The command line's `--frozen-parts graph_side` is the same setting.
- A failed run's **error banner** spans the page. Under its headline ("Training failed: Process
  exited with code 1") is the last error line the run printed ("Last error printed:
  TrainingDeviceUnavailable: --device auto requires CUDA, …"), redacted for a read-only browser
  as the output is. Beside the run it says where the run's log is only when there is one (its
  `logs/train_….json`), else that there is none ("no run directory, so no log" for a run refused
  before it made its run directory). The last lines of its output follow
  under "Last output (N lines)": open in a window at least 1000 px tall, closed to that one line
  in a shorter one so the charts keep their room (opening or closing it holds for this page);
  four lines show at a time, the rest scroll in the box (in a narrow window all of them show).
  Dismiss hides the banner for this browser tab. Every notice in the run bar shows names and
  paths as text, whatever characters a workspace or run name holds.

**Parameters, "for the next run"** (K6: a base's settings): exactly the base stage's reference
settings (`src/config/recipe.py` BASE), each starting at its reference value. Epochs, Learning
rate, Batch size and Seed show; the other seven sit under "More reference settings", closed.
Early stopping patience reads "off" when it is off (the reference); a whole number turns it on.
A line under the form says "Uses the reference settings." or "Differs from the reference
settings: Epochs 5 (200)", and a value its field refuses is named there ("Not valid: …") and
refused by Start, which sends only the settings that differ. Task models are trained from a base
on Research › Prepare › Fine-tune. **Data Paths and its Save Path are gone**: the workspace
is the run bar's choice, and the dashboard no longer reads or writes
`.shepherd_training_config.json` in the server's working directory (a file left there is
ignored; whether to offer "Save as the default dataset" is an owner question). When the three
groups do not fit the window (they fit at 1920 × 1080), the parameters scroll inside their
column; the run bar never scrolls. The run view is three rows: its header and state line, the
charts, and the resources line. The charts take what the other two leave, and each of the four
plots keeps a plot area of at least 140 px: when the window is too short for that (1536 × 760,
for example), only the charts' row scrolls, its toolbar staying at its top; the resources line is
a row of its own under it, so nothing is ever drawn over an axis (at 1920 × 1080 and at about
1850 × 960 the four charts, their axes and labels and the resources line all show at once). A
plot's box is sized by the layout alone, never by the canvas drawn in it, so the same window size
gives the same plots whether the page was opened there, reloaded, or came back from a narrow
window.

**Continue…** opens a drawer in place of the parameters: the model records of the run bar's
workspace, bases and task models (since K6 a file without a model spec is never continued from the
dashboard; the server's list: an operator control, since listing opens every checkpoint), chosen
in two steps, or the path of one of those checkpoints typed by hand. **Run**: the runs, latest
first, from every directory under `runs/` (since K8-7b, 2026-11-08), each as its name, directory
and start time ("base-a · GATv2 · 2026-10-28 13:57": the directory's name, upper-cased but
`gatv2`, which reads "GATv2"; a location, not the model's architecture, which its record names;
an unnamed run is its directory and time). **Checkpoint**: that run's files, newest first, each in
at most 40 characters saying what it is and how far it trained: "Latest save · 195 epochs
completed", or "Top-k save · 21 epochs completed" (a file the trainer kept for its validation
figure); the epochs are the file's progress record (a file without one says the epoch it records,
never a count made from it), and a file that holds weights only or cannot be read says so
("weights only", "cannot be read"). Choosing another run offers its files with its latest save
chosen. Under the two, a table says what the option does not: the file, its stage, what it holds,
what continuing does when its stage ended ("not possible: it stopped early" for an early-stopped
one), the Trainer's validation figure ("LP accuracy (Trainer) 0.5000" for a base; the Trainer's,
not a test result), its model fingerprint (Copy gives all 64) and its run's directory. Since M5-3
a typed path only finds a checkpoint that list holds, and any other file is refused without being
opened (`PLAN_TEST_RESULTS.md` D8). **Check checkpoint** shows what continuing would do, and the
dataset fingerprint the parent recorded against the workspace's current one: ✓ the same, or amber
when it differs (with the listed workspace it matches, or "matches no listed workspace"). Since
K8-7 (2026-11-08) a continued run inherits only the settings this version has: a setting the
parent recorded that this version no longer has (`num_workers`, the old model settings such as
`conv_type`) is neither inherited nor a reason to refuse, a parent missing a current setting is
refused, and Check checkpoint and the training process decide alike, in the same words. Since
research comparison Stage 9a (2026-11-10) new checkpoints record training rules 2, which add one
setting, `frozen_parts` (which parts of a finetune do not train: `none`, the recipe's, or
`graph_side`). A parent of rules 1 continues: one settings decode, shared by Check checkpoint and
the training process, reads it with `frozen_parts` none from what its record shows (rules 1 had
no freeze; a finetune's weight transfer records every part trainable), lists it once under
"Settings taken from this checkpoint" with that source ("none, decoded from rules 1: rules 1 had
no freeze and trained every part"), as the training log says it on a line of its own, and
records that source on the continued run's checkpoints; the parent's file is not changed. A
parent that records no rules (K4's) or rules this version does not know is refused, and so is
any other missing, contradictory or unknown setting. A continued run never changes the freeze; a
branch keeps it too. Since Stage 9b (2026-11-10) `graph_side` trains: a finetune with it keeps its
base's input module, encoder and LP head as they are (no gradient, no optimizer group, evaluation
mode for the whole run, so BatchNorm's statistics are used and never updated and dropout is off)
and trains only its patient encoder and task head, at the recipe's loss coefficients; its record
says the LP term has no trainable parameter, each part trainable or frozen, the frozen parts'
forward mode and the SHA-256 of the downstream weights it started from (two children of one base
and seed start alike). Resumed or extended it stays frozen, with the same optimizer groups
restored, and its new checkpoints record their stage's weight transfer, their parent's
unchanged, so an arm paused and resumed still names the downstream weights it started from (Models
› Results shows it in Compare and the export). While
the drawer is open the run bar shows Start (disabled: "close Continue… to start a new run"),
**🔒 Continue Training** (disabled while the datasets differ, with the reason) and **Cancel**.
A listed checkpoint is found again in the server's list. **The model fingerprint shown is the one
continued from** (implemented in the root-cause round). For a listed checkpoint, Check checkpoint
and Continue Training send the list's fingerprint; for a typed path, Check checkpoint reads the
file and shows its fingerprint, and Continue Training sends the one shown. The job runner's
preview compares it on its read of the file, before loading it: another file is refused with
"The checkpoint changed since it was chosen. It holds … now (chosen: …)." and what to do for
where it was shown: "Refresh the list, then choose it again." for a listed checkpoint (the same
option sends the list's fingerprint again until the list is read again), "Click Check checkpoint
to read the file there now." for a typed path. The job runner then gives the
training process the digest of the parent it decided on (`train_model.py
--expect-resume-sha256`, always, for every continued run): the run's placement under the
parent's architecture and its settings come from that read, and a file replaced after the start
is refused by the process before it is loaded (`train_model: refused: The checkpoint changed
since it was chosen; choose it again. …`, exit 1, no run directory; Continue… lists the
checkpoints again when it opens). The API takes the parent's digest as an
optional query parameter too, `expected_parent_sha256=<64 hex digits>` beside `resume_from`
(409 when it is not the file's, 422 when malformed); without it the process is still bound to
the parent the runner read. The job runner and the training process still check everything
else at start. Check checkpoint names the listed file it read ("Checked gatv2/<run>/<file> from
the run list"; a typed path's check gives its path and fingerprint), and a matching dataset is said
once, as "✓ Trained on the dataset selected in the run bar" (the fingerprint is the run bar's).
While the drawer is open the run bar's line reads "Run: a new one, continued from the checkpoint
chosen in Continue a run": a continued run trains in a run directory of its own. The drawer has
its own **Device** (auto, cuda or cpu), which starts from the
parameters' Device each time it opens and is what Check checkpoint and Continue Training use.
When the run bar's dataset changes while the drawer is open (by hand, with Datasets'
"Use for training ›", or to follow a training job started on another dataset), the list
becomes that workspace's runs, read again the same way.

**The run view** shows the run the job runner started last or, before one since the server
started, the latest run of the run bar's workspace (listed again every 10 seconds while the page
is open, so a run started from the CLI appears once it writes its first checkpoint). Its header
is one line: "Run shown" and the run as a person names it ("ui-live-stop-02 · GAT · 2026-10-24
17:39"), "latest" for the workspace's latest run, and how the dataset the run used, read from its
checkpoint, stands to the run bar's: "✓ same dataset as the run bar", amber "≠ not the selected
dataset", "dataset recorded at the first checkpoint" while it runs without one yet, or "dataset
not recorded (no checkpoint)" once it has ended without one. While Continue… is open with a
listed checkpoint chosen, it also says whether this is "the run continued from" or "not the run
continued from": the curves are the run shown's, whatever the drawer chose. **Details ›** holds
what is read less often: the run's directory, its checkpoint (relative to the workspaces root;
the newest one while it runs), its logs and the dataset fingerprint it used (Copy gives all 64);
its workspace when it is not the run bar's. Under the header, one state line: "195 of 1000
epochs completed · can be continued", "Finished · 5000 of 5000 epochs completed · to train
further, use Continue…", "Stopped early (…) · … · cannot be continued", "Running · epoch 12 of
100", or "No checkpoint saved, so nothing can be continued"; a run that has no run directory
(refused before it made its run directory) says so, with no checkpoint, no log and nothing to
chart. Below them, one figure of four charts sharing the epoch
axis: Loss (train, and validation dashed: a logged `val_loss` or `val_lp_loss`, whatever the run),
the run's validation figure and its companions, and the learning rate on a log axis (a 0 or a
missing value is a gap). The middle two follow what the run logged: a base charts LP accuracy,
then ROC-AUC and AP; a task model pooled MRR, then pooled top-k. A log whose validation entries
hold neither figure (a file without a model spec, say) draws no series there, under the titles
"No other figure charted: Export CSV has every logged figure" and "No further figure charted"
(since K8-7b, 2026-11-08; its MRR and Hits@K charts are gone), and its Loss and learning-rate
charts are drawn as for any run; before the first validation the two wait, as before.
Under it, the resources line (GPU and memory).

- **Zoom and pan**: the wheel over a chart zooms around the pointer; over the epoch axis it zooms
  only the epochs (the four charts follow), over a value axis only that chart's values. Drag pans;
  a double-click or **Reset view** fits the data again; + and − zoom. With a chart focused: + − x
  X y Y, the arrows, 0 and End. The view stays while new epochs arrive, across page switches and a
  reload in the same browser tab (for the last 20 runs shown); a run not shown before fits its
  data, its epochs and every chart's values. The epoch axis counts whole epochs. While zoomed,
  "N new epochs →" jumps to the latest at the same width. The epoch axis never starts below
  epoch 0: a view that would (a zoom out, a pan, a view kept from before) is moved right with its
  width kept; a value axis has no such bound. The wheel zooms in a narrow window too; over
  anything but a chart's plot or axes it scrolls the page, and in a narrow window a touch on a
  chart scrolls the page. The gestures are listed in the ⓘ beside Reset view, not kept on the
  toolbar.
- **Export CSV** saves the run's metrics log (epoch, split and each logged value) from the browser
  as `training_metrics_<architecture>_<run>.csv`.
- The charts read the run's metrics log through the page's own server function, for a run the
  server lists (never a path), and only when it has new rows. The page is read again every 1.5 s
  while it is shown, and not at all on another page. The chart library is uPlot 1.6.32, served by
  the dashboard itself (`src/webui/static/vendor/uplot-1.6.32/`, MIT); nothing comes from a CDN.
- While another run's charts load, the figure says "Charts loading…" instead of showing the
  previous run's charts under the new run's name, and an answer that arrives after the selection
  changed is never drawn. If the charts of the run asked for cannot be read (the server
  unreachable, or an error), the figure says "The charts could not be loaded. Trying again…" and
  asks again every 3 s, up to 20 times; a run already drawn keeps its charts while its newer
  epochs are asked for again.

**A read-only browser** sees the same page with every action disabled and D8's reason, and no
checkpoint is opened for it: the run header's tag says "dataset not shown · operator control",
the state line gives only the run's state, and the parent list is not shown. The metrics, the charts and the dataset records are read for it as for an operator.

### Research › Evaluate › Test

The page for the flow of §4.9: import a cohort version, measure a model on a data version of
any kind (a supplied cohort, a KG edge set or a generated split), keep its report and register
it. Three panels:

- **Data** (since research comparison Stage 7; "Cohort" before). The workspace (the server's
  list, shown by its dataset fingerprint as on Datasets) and the **Data version** (the registry's
  list, `SHEPHERD_REGISTRY`), by kind and role: supplied cohorts first, as before ("label · role R
  · n of N ranked-eligible"), then KG edge sets ("kg-test · KG edge set (test)"), then generated
  splits ("gen-val · Generated split (val)"). The chosen version shows its counts (a supplied
  cohort's role and cases; a KG edge set's pairs and edges; a generated split's patients,
  diseases and fixture rows, each with the unit its manifest records under **Counts and units**;
  a KG edge set's or a generated split's kind and role are its option's, not repeated), its
  version digest and whether it was mapped against this workspace's graph ("bound to this graph",
  or "mapped against another graph", which a run would refuse; never in the option). With no KG
  edge set in the registry, a line under it names the command that imports one on the machine (the
  importer's `--kind kg_edges` with `--from-checkpoint`, §4.9); the page's Import makes supplied
  cohorts only. Below them, every run
  directory of the version's report store, newest first: when it ran, on which model, whether a
  report was kept, whether it is registered, and "runs under these conditions differ" when the
  runs of its group differ (in amber with "— investigate" only when the group records
  deterministic algorithms on), or "comparison incomplete". A run that kept no
  report says so; a test running now is the top
  row, and the last test run on this server that ended with nothing in the store (cancelled
  before its run directory, or failed) stays a row until the next start or a restart. The
  newest 8 reports are listed, and the selected one; **Show all N reports** lists the rest
  (this browser tab keeps the choice). For an operator, each row's registration is checked in
  the ledger beside the listed checkpoint, in the workspace shown, that holds the weights it
  measured, so a report left to register reads "not registered" whichever row is selected; one
  whose checkpoint is listed only in another workspace says "registration cannot be checked
  from the list" until it is selected.
- **Run.** The model: the workspace's checkpoints from the server's list, by run, epoch, file
  and model fingerprint. Listing them opens every checkpoint, so it is an operator control: a
  read-only browser is not shown it, and the page lists them once per workspace (↻ reads the
  list again). Then **Measurement** (since K8-6a), the protocol the run passes as
  `--mode <protocol>` (§4.9): the options are the protocols this build implements whose needs
  the chosen model's listed capabilities meet and that measure the chosen version's kind (since
  research comparison Stage 7): a task head's measurement of a finetuned model (Disease, Gene or
  Patients-like-me) on a supplied cohort or a generated split, and KG link prediction for a base
  and every finetune on a KG edge set. So a finetune on a supplied cohort is offered its task's
  measurement alone, chosen already, and on a KG edge set KG link prediction alone; a base on a
  supplied cohort is offered none, and Start says which measurements take it and what they
  measure. With no version chosen, every measurement the model's capabilities meet is listed. A
  single option is chosen already; of several, none is until you choose one; with none, Start
  says why. One Start runs one measurement: measure the model by another protocol with another
  Start, one GPU job at a time. A file whose record cannot be read offers none either, and Start is off
  with the record's own reason ("Record not readable (written before the model spec: not
  readable by this build; start a new run): not measured here. Choose a model with a readable
  record" for a file without a model spec; since K8-7b, 2026-11-08). Like the model, it is an
  operator control. Mode C and Mode D are not
  protocols, so no page or command runs them. Then the **R10 evidence**, chosen by the runner once
  by the chosen version's kind (since research comparison Stage 7): for a supplied cohort, the
  directories the run will look in (the measured workspace, and
  `SHEPHERD_TRAINING_EVIDENCE_DIRS`), and a refused setting in red with ✕ and "Refused:" and
  its reason, which turns Start off; for a KG edge set or a generated split, "Not applicable"
  in grey with ○ and why (R10 is a supplied cohort's), and the run is given no evidence
  directory whatever the setting says. Then **Register the result**,
  **Deterministic execution**, and the batch size and device the server fixes (the measurement's
  default; `SHEPHERD_DEVICE`).
  **🔒 Start test** starts the real `measure_scorer.py` through the job runner (§4.11), with the
  workspace, version and checkpoint the server finds again in its lists, and **the full model
  fingerprint and version digest the page showed as the expected ones**: the run refuses before
  measuring when the file or the version it reads holds other bytes. Start says why it is off:
  a GPU job first ("Disabled: GPU job: … One managed GPU job at a time (training or
  evaluation)."), since no choice enables it while one runs, then a choice missing or another
  graph. The job card shows the state (Running, Completed, Failed in red, Cancelled in grey:
  never green for a failed or cancelled run), the elapsed time, what the run was given, its
  start and end times in UTC (as the report rows name runs) and the last 12 lines of its output
  (folded after a cancel, whose last line says the run was cancelled, with no traceback; a
  patient id a measurement refusal names is left out, D7); **🔒 Cancel**, beside the state,
  stops the test run only, never a training job. On a wide screen, Start, its reason and the job card stay at the foot of the panel (the
  choices above scroll) and the output scrolls inside the card under the line with Cancel, so
  both stay in view. Once the run has ended, the card shows what the runner read
  from disk as the run ended (headed "When the run ended:"): whether a report was kept and whether
  it was registered then, with the runner's sentence ("No report was published …" when the run
  wrote no outcome file; after a cancel, what was left). A report registered afterwards with
  Register this report keeps the card's reading of the run's end, and the card points to Result,
  which shows the report's registration now. While the page is shown it reads the job every 5 seconds; the report of a run that has
  just ended is selected, once for that run. A choice made while the page reads the version's
  store again (after the job changed, an act or an import) applies once that read has ended;
  the read never undoes it, and of several choices made meanwhile the last one stays.
- **Result.** The selected report as the record the ledger would hold, by one display for every
  mode (since K8-6e.2): a task's tiles and figures in words, with every metric as recorded one
  click away ("Recorded metrics (N)"); a report of a mode this version does not describe, a kept
  Mode C or Mode D report included, under "Mode <mode> (a measurement mode this version does not
  describe)", with one "Recorded metrics" row listing every metric by its recorded name and
  value; the score semantics exactly as recorded; coverage with each exclusion, the relations,
  R10 in words (*unverifiable* is never a 0), the environment, the dataset origin, and whether it is registered (for an
  operator, checked in the ledger beside the listed checkpoint holding the weights it
  measured). Since research comparison Stage 8b the report is read by the one reader the results
  service and Models read with (its data kind's result view, registered or not): a KG edge set's
  exposure and a generated split's generation are said in the same lines as on Models, and a
  record the reader cannot read is shown with its reason (the field named, never its value),
  not as a result. When a record names this run with other bytes (the report was replaced after it
  was registered), or that ledger cannot tell, the Result gives the registration's reason.
  **🔒 Register this report** registers a kept report of any kind (a supplied cohort's, a KG
  edge set's or a generated split's) through the one registration function (§4.9): the server finds the report in the version's store, checks that its bytes
  are the ones shown, finds where its ledger goes from its own checkpoint list (the listed file
  holding the weights, by full digest, never the path the report records: the file the run read
  when it is listed with those weights, which is where the list looks, else another listed copy
  in the workspace the run measured, else one in the workspace shown) and registers it. A
  checkpoint moved since the run is not found that way; the page says so, and
  `scripts/record_evaluation.py` on the machine remains. A kept Mode D report registers the
  same way, under its own mode; the same run again says it is already registered, with nothing
  written (§4.9). **Register is on for any kept report of this run the page can confirm**
  (policy rev 16, since K8-6r.2). It is off only for a read-only request, a run that kept no
  report, this run already registered, or this run's own report not built (its own reason) or
  not read by the listing. When the store cannot be listed, it says so and points to
  `record_evaluation.py --report …`. Another run, a group whose runs differ and an unreadable
  report elsewhere never turn it off. When the runs of the report's group differ, or their
  comparison is incomplete, the page describes it and lists **investigation notes** (who, when,
  what; 🔒 Add note), kept with the group's reports. A note records what was found and changes
  nothing. The registration's message carries the group note and the historical note.
  On a wide screen the note form (Who, What was found, 🔒 Add note) stays at the panel's foot,
  above Register, so it is in view however long the report or the note (What was found grows
  to four lines, then scrolls inside itself).
  When the notes file cannot be read, the page says so (a note may be missing), never "no
  note is kept".

**🔒 Import a new version…** opens a drawer over Run and Result: the file in the intake directory
(`SHEPHERD_INTAKE_DIR`, §4.12: its `*.json` files by name, size and modification time; one that
is a link leaving the directory is listed and not offered; nothing is uploaded from the
browser), the version label, the role (the version's declared purpose, for example mygene2,
research_selection or institutional_acceptance) and the lost-terms
policy. **🔒 Preview** runs the importer in the service, on the CPU (it reads the workspace's
graph), and writes nothing. The preview shows the counts, the exclusion reasons and their
counts, the unmappable terms with how many cases list each, and what the version binds; **case
ids are not sent to the browser** (the command-line preview on the machine lists them, and the
committed version records them). A file that does not fit `shepherd-cases/1` is refused with its
problems listed by case index (at most 10), never by case id; the command-line preview
(`scripts/import_cohort.py`) lists each one with its case id. **🔒 Commit as version …**
prepares again from the same inputs and commits under the preview id shown: a form changed
since the preview, or a source or graph changed since, is refused. One preview or commit runs
at a time; after a commit the drawer closes and the new version is chosen. Below 1400 × 760 the
drawer is an overlay at the right that scrolls inside, and the keyboard focus moves to its
title.

A **read-only browser** sees the versions, their reports and states, the job and the results:
every control is disabled with "Operator controls only: <reason>" (the note's Who and What
fields too), and the checkpoint list is not shown. Every control checks its own request in the
backend (§4.13). No absolute server path is shown: a path under the registry, the workspaces
root, the intake directory, an evidence directory or the repository is written under its name
(`<registry>/versions/…`), any other as "a server path".

### Research › Evaluate › Models

The research view of the same model list the Diagnosis tab's "Change model…" uses, over the
checkpoints the server lists for one workspace (choose the **Workspace**; since K6 **Task**
narrows the list: All, Base, Disease, Gene or Patients-like-me; History, and the "history" record
it chose, went in K8-7b, 2026-11-08, so a file whose record cannot be read is listed under All
only). Under the workspace
select, the chosen workspace's dataset identity is said again (build time, dataset fingerprint,
`synthetic`, the record's state), so a long workspace name that the select cuts never hides it.
Every run is shown, newest first, from every directory under the workspace's `runs/` whatever
its name (since K8-7b). Columns:

- **Model**: what the file is first (Base, Disease, Gene, Patients-like-me, "Record not readable"
  or "Record not read"; a file without a model spec is "Record not readable", with "written
  before the model spec: not readable by this build; start a new run", and no longer "History ·
  never run" since K8-7b), then the run's directory (upper-cased, "GATv2" for `gatv2`: a location,
  not the architecture, which the record names), the run's start time, the epoch; **In use
  (Diagnosis)** on the checkpoint being served (matched by its full model fingerprint);
- **Fingerprints: model · dataset trained on**: the model fingerprint (12 hex digits of the
  checkpoint's SHA-256) over the listed workspace whose full dataset fingerprint the checkpoint
  records, or "matches no listed workspace"; `synthetic` in amber;
- **Validation (Trainer)**: the Trainer's own figure on the val samples, the metric training
  selects by, named under its value (a base's LP accuracy, a task model's pooled MRR), read from
  the file's logs whatever its record says; none when the logs hold neither, as for a file
  without a model spec, whose `val_mrr` is not read (since K8-7b a history file's MRR over its
  top 20 is no longer shown; its raw values stay in the file, and the detail says "Validation
  (Trainer): not read (record not readable)"). It is never a test result and never shares a
  column with one;
- **Recorded results**: what the ledger beside the checkpoint holds, as a state ("none in
  ledger", "no ledger", "3 · runs differ for 1", …), never a number of a result.

**Listing opens every checkpoint of the workspace**, so the list is an operator control (§4.13).
Since M5-3 the server reads each checkpoint file once and again only when it changes (its
device, inode, size, modification and change times), so opening the list again, or the re-read
after a Load, opens no unchanged file; a load, a start or a measurement still checks the bytes it
uses against the digest shown.
A read-only browser gets no list: the page says why, naming the three conditions (deployment
mode M1, a request on this machine's loopback, here or through an SSH forward, and no forwarding
header), and shows instead **Recorded results of the model in use**, read from the ledger
beside the served checkpoint by its model fingerprint, opening no checkpoint. Each record's
regime (device, deterministic on or off) is the one the record holds, as an operator sees it
(`GET /api/v1/evaluations/loaded-model` carries each record's `deterministic_algorithms` for
this; a CPU record's environment class alone does not say it). There are no Trainer validation
figures and no Load there.

**Order: Newest first | By family.** By family (since K6) draws one group per family, the parent
first, with a Lineage column. No order ranks a model by a test score: ordering by test score
retired in K8-6d, with its role setting, its Mode C rule and its one record per checkpoint. Every
record stays readable in the detail below, and no model is ever chosen automatically by its test
score.

**Selecting a row** shows its full model fingerprint with Copy, what it was trained on (and
whether that is this workspace's current dataset), the Trainer's validation figure in its own
card, any warning, and three segments:

- **Recorded results** ("Recorded test results" until research comparison Stage 6): every
  record the ledger holds for these weights, of any data kind and role, each named by its
  version's kind and role (a KG edge set's with its exposure, "exposure not recorded" with the
  reason when its block does not read), one table per measurement mode, in the order first shown, each under its mode's label (and a task's note),
  by one display for every mode (since K8-6e.2; before it, Mode C's records came first, then
  Mode D's, under labels of their own). Each row gives the cohort version and its role; a task's
  pooled MRR and its second figure named in full in the header, or, for a mode this version does
  not describe (a kept Mode C or Mode D record included), one "Recorded metrics" column listing
  every metric by its recorded name and value, unrounded; coverage with every exclusion; and
  `registered #N`, or `runs differ #N` with "runs under these conditions differ" and what
  differs (neutral; amber with ▲ and "— investigate" only when the group records deterministic
  algorithms on, so its runs were expected to agree), or "comparison incomplete" (never shown
  as agreeing),
  deterministic on or off, the reports merged into it before each run had its own record when
  there are any, and "possible overlap with record N (not recorded)" between such a record and a
  named run; then the relations, a task record's other metrics in words, R10 as states
  (*unverifiable* is a reason, never 0), "Score semantics: …" exactly as recorded, a task
  record's every metric as recorded behind "Recorded metrics (N)", the environment, the dataset
  and the registration with its run ("not recorded" for a record written before each run had its
  own record). Every record is shown;
  none is hidden or ordered out by these descriptions. A record for these weights that cannot be
  read as a result (since K8-6e.1) is never shown or counted as one: under the tables, "▲ 1
  record for these weights cannot be read as a result. The ledger keeps it as written." and a
  small table, Index · Registration · Report · Reason (a sequence or report digest it does not
  carry readably reads "not readable"); the segment then reads "Recorded results · 3 (1 not
  readable)". The read-only view lists the same items, from the route;
- **Usage history**: the entries that name this checkpoint in every listed cohort version's
  usage history, newest first: test runs and registrations, and any model load kept from before
  K8-6d, shown as recorded ("ordered by" its metric's name, and its scoring semantics). A test
  run reads a task's first figure over its ranked cases, or another mode's label and case count,
  then how it ran, then every metric it records behind "Recorded metrics (N)" ("no metric
  recorded" when it records none). An entry written before the usage history recorded a mode
  reads "Measurement mode not recorded" (since K8-6e.2; before it, Mode C);
- **Results** (since research comparison Stage 8b): every result of **This family** (the
  selected model's family, in this workspace) or **This workspace** (each result found in it: a
  listed checkpoint of its model, or a ledger holding its record), read by the one results
  service (the rows `scripts/export_results.py` exports) when the segment is opened, and again
  with **↻ Read again**. A result found in no listed workspace (a kept report no listed ledger
  registered, whose weights are not listed now: moved, or a `last.pt` saved over) is listed
  under This workspace too, its model "Unknown" with why, and the line over the tables counts
  such results in either scope, so none is hidden. **Protocol**, **Data kind**, **Role** and
  **Environment** filter them, each offering the values the scope holds (a field without a
  value offers none). One table per protocol, newest first by run time
  (the service's listing order, so nothing is ranked; nothing is ordered by a metric, and a
  default order for task records stays the owner's open item, BACKLOG 63). Each row: the
  result's name (its run's time and its report's 12 hex digits; where two would read alike, as
  two runs of one model in a minute with byte-equal reports do, the time to the second and then
  the run id's own last part; its Compare option and column head say the same name), the
  model (fingerprint and kind; its family under This workspace), the data
  (label, kind and role), the metrics as a record of that protocol shows them (a task's two
  figures, unrounded where its condition group's runs differ on them; any other protocol's every
  metric as recorded), the state (report kept, or "Report not available (ledger record
  only)"; registered, not registered or merged; ▲ report bytes that differ elsewhere; how its
  condition group compares), the exposure where its record holds one (a KG edge set's reading,
  a supplied cohort's R10 one click away; a generated split's is its allocation's, not
  compared) and the environment. Every item that could not be read is listed under the tables
  with its reason, whatever the scope and filters. **Compare** puts the first five of the
  **Chosen results** side by side, one column each headed by its name and protocol,
  **Differences only** on by default: the result, the model (stage, task, operation, parent,
  family, encoder, relation to the reference recipe, capabilities and every spec value; the
  stage, the encoder's kind and the task once, in the model's own rows, not again under the spec,
  the capabilities or the training settings), the recorded and effective training settings with
  the weight transfer and the loss identity (so two arms of one base are checked to start from
  the same downstream weights here), the data's identity and role, the measurement, the
  environment, the exposure and every metric, as recorded, each group once. Each row is named in
  words (a metric as the tables name it, a learning setting as the Training page does; its
  record key in the row's title), and a nested setting is one row per entry, so a difference
  inside it is a row of its own. No difference is computed, and each metric row is one
  protocol's (a result of another protocol reads "another protocol" there). Each missing value
  keeps its words: a field without a source reads "Unknown" with why (the why once per field:
  its other rows read "Unknown", the why in their tooltip), a value recorded as null "none", a
  field the result's records do not hold "not recorded", and a field that does not apply to the
  result's data kind "not applicable" (a KG edge set's exposure definition and counts, R10, and
  an entry of the data's identity, counts or content references that only another data kind
  records). Five is this view's
  cap only: the choice is not capped. **Export** writes every chosen result or every shown row
  (the unreadable items among them), uncapped, as CSV or JSON through the export's one
  serializer (the columns and bytes of `scripts/export_results.py`), as a new file to
  download. Reading the results opens every
  workspace's checkpoints, so each of these is an operator control (§4.13): a refused request
  reads and lists nothing and says why. Comparing results across workspaces is not built in
  this page yet; on the machine, `scripts/export_results.py` joins workspaces.

**🔒 Load this model** is the shared Load (the same confirmation as Diagnosis' "Change model…";
it covers the detail while it is open, its buttons stay in view under its table however short
the window, and Close returns to the detail with the same checkpoint still selected). It is off,
with the reason beside it, for a base and for a file whose record cannot be read, before anything
is read ("Record not readable: written before the model spec: not readable by this build; start
a new run." for a file without a model spec). A Load
writes nothing to a usage history. The page is read when it is shown, when the workspace changes
and after every Load, including one that changed nothing served (a refused reload, the model in
use loaded again), so what changed meanwhile is shown; nothing on it runs on a timer.

**Narrow** (below 1400 × 760, where the regions stack under one another), choosing a row scrolls
to its detail and moves the keyboard focus there; **↑ Back to the list** in the detail's header
returns to the row.

## 4.15 PyTorch Geometric's version (since Batch 1, 2026-10-26)

**`torch_geometric`, PyG's pure-Python core, is in the lock** (BACKLOG 50). `pyproject.toml`
declares `torch-geometric==2.8.1`, and `uv.lock` holds it with what it requires (aiohttp 3.14.5
and its family, pyparsing 3.3.3, xxhash 4.0.2: the versions M5's GB10 acceptance ran). `uv sync
--inexact`, and so `deploy.sh` and `deploy.cmd`, install it; the deploy scripts no longer install
it separately, and only check that it imports. A plain `uv sync` no longer removes it (it still
removes the native extensions and cuVS, so keep `--inexact`).

**Its native extensions stay per platform, as before**: pyg-lib, torch-scatter, torch-sparse and
torch-cluster come from the data.pyg.org index for the installed torch (the scripts now derive
`https://data.pyg.org/whl/torch-<torch.__version__>.html` from the installed torch instead of
writing the version a second time), or on ARM from the prebuilt release or a local build
(`PYG_ARM_STRATEGY`). Their versions differ by platform (the GB10's prebuilt pyg-lib is 0.6.0, the
x86 index's 0.8.0) and are recorded by the probe's A1 and A5, not pinned.

**The ARM + CUDA route** (`deploy.sh` Stage 3; since research-comparison Stage 2a, BACKLOG 70 and
71). data.pyg.org has no aarch64 wheels, so the menu, or `PYG_ARM_STRATEGY`, chooses the prebuilt
release (`pull`), a source build (`compile`), `skip` or `abort`; with no terminal and no setting it
is `skip`, as before.

- **A failed pull stops by default.** The pull downloads the release asset pinned in `deploy.sh`
  (`PYG_RELEASE_TAG`, one SHA-256 per CPython tag), verifies its SHA-256, installs exactly its four
  wheels by file path (`--reinstall --no-deps`) and checks what was installed. When a step fails,
  the deployment stops and names it: the release is not usable on this host (exit 11: the host's
  torch or CUDA is not the release's, "Case B", or the release has no asset for its Python), the
  download failed (12), the checksum failed or could not be taken (13; such a download is never
  reported as verified), the install failed (14), or the check failed: the installed extensions are
  not shown to be the release's wheels (15). When the wheels are installed and checked but the source
  receipt (below) cannot be written, it stops with 17 and builds nothing in their place: a build
  would keep its receipt in the same place.
- **`PYG_ARM_ON_PULL_FAILURE=build`** lets a failed pull build from source instead, non-interactively
  too; `stop` is the default. On a terminal, choosing the pull in the menu asks this once, right
  then; nothing waits for a person after the pull has started. An unknown value is refused before
  anything runs (exit 4). Before Stage 2a a failed pull, and `PYG_ARM_STRATEGY=pull` on a Case B
  host, built from source without being asked; both now need this setting.
- **The source build** (`compile`, or a failed pull where allowed: `scripts/build_pyg_arm.sh`)
  builds all four packages with `--no-cache-dir` into a fresh directory per run under
  `PYG_WHEEL_OUT` (default `~/pyg-wheels-arm`; earlier runs' directories are left as they are). Only
  when all four built does it install them, exactly those four wheels by file path (`--reinstall
  --no-deps`); it then checks that each imports and that what is installed is exactly those wheels,
  and only then writes the receipt. It exits non-zero when a build (2, nothing installed), the
  install (3), an import or the check (4) fails, or when the receipt cannot be written (5).
  `SCATTER_SPEC`, `SPARSE_SPEC`, `CLUSTER_SPEC` and `PYGLIB_SPEC` choose each package's source: a
  registry requirement (`torch-scatter==2.1.2`), a VCS URL
  (`git+https://github.com/pyg-team/pyg-lib.git@0.6.0`) or a local sdist or tree; `deploy.sh` passes
  them through. Without them the builder takes the newest pyg-lib tag the data.pyg.org index lists
  for the installed torch and the unversioned names of the other three, so a build can resolve other
  versions than the release's; the receipt says which. Under `compile`, a build that stopped before
  its install (nothing installed) is said and the deployment continues, as before. Once the build
  has installed its wheels, a failed install, import or check stops the deployment (exit 16: what is
  installed is not verified), and so does a receipt that cannot be written (17). The build a failed
  pull is allowed to run behaves the same; and when that pull had already installed the release's
  wheels (its install or its check failed), a build that does not complete stops the deployment too
  (16), since those wheels stay installed, unverified.
- **The check** (`scripts/validate_pyg_ext.py --check-wheels`, over `src/utils/pyg_native_check.py`):
  each installed distribution's version, and each installed native library's SHA-256, equal its
  wheel's. Only the two routes above run it: the Linux index route (x86, or no CUDA) and Windows'
  `deploy.cmd` install from the data.pyg.org index by name, as before, and are **not covered** by it.
- **The source receipt.** After a passing check (for a build, once each extension has imported),
  either route writes `.venv/share/shepherd-advanced/pyg-native-receipt.json`, outside every
  distribution, so `uv sync --inexact` leaves it (a test runs uv to show it). It is kept in a
  virtual environment only: the builder run for an interpreter that is not one keeps none, and says
  so. Per wheel: the requested source (a registry name and specifier, a VCS URL with the commit it
  resolved to, or a local source's SHA-256; for the pull, the release tag, the asset and its
  SHA-256), the resolved and installed version, the wheel's SHA-256 and the installed native
  libraries' count, digests and aggregate digest. Both routes record the torch and CUDA build
  compiled against, in one notation (torch `2.10.0+cu130`, CUDA `13.0`): a build its target
  environment's, the pull the release's declared target. A build adds the version numbers of nvcc,
  gcc and g++ (never their banners, which can carry a path or a name) and `TORCH_CUDA_ARCH_LIST`,
  `FORCE_CUDA` and `MAX_JOBS`; the release carries no compiler facts, and they read unknown. No
  path, user name or URL credential is recorded. The receipt is keyed by the installed native
  digests: `scripts/validate_pyg_ext.py` (or `--receipt`) shows per distribution whether what is
  installed now is what it records, and a later install from elsewhere reads as differing, never as
  the receipt's. A record, never a gate: no version is judged anywhere.
- **What every run records** (since research-comparison Stage 2b, BACKLOG 71). The run environment
  that training writes into each checkpoint and a measurement into each manifest (and so each ledger
  record), and the probe's A1, name each extension as a component: whether it is installed and
  imports, its distribution version, its native libraries' count and aggregate SHA-256 (so two builds
  of one version read apart), and, only where the receipt above matches what is installed, the
  receipt's route, requested and resolved source, wheel SHA-256, torch and CUDA build, compiler
  versions and flags; otherwise "not recorded", with why. Whether PyG used an extension for a given
  operation is not observed and reads unknown. No path on the machine and no wheel, library or
  release asset file name is recorded (their SHA-256 names them); a requested source's URL is kept
  without user name, password, query or fragment. The Models and Test pages show the identity as
  "Native build", on one line on Models and a component per line on Test; a record written before
  reads "not recorded". Every recorded fact of each component (its states, fallback reason and
  source facts as well) reaches the API's record views (`components`), the results query, Models ›
  Results' Compare (one row per fact) and the export (since review fix RC-P2-1). This holds for every
  route, the Linux index route and `deploy.cmd` included; only the receipt is the two ARM routes'.
- **What is verified**: these routes run in tests with stand-ins for the ARM host, GitHub, pip and
  the compilers (`tests/unit/test_deploy_pyg_route.py`), which is software simulation. Neither route
  has run on a GB10 since Stage 2a; each counts as verified there only by its own GB10 run.

**The verified set**: `torch_geometric` 2.8.1 with torch 2.10.0+cu130 and the prebuilt ARM
release's extensions (pyg-lib 0.6.0, torch-scatter 2.1.2, torch-sparse 0.6.18, torch-cluster
1.6.3), in M5's GB10 acceptance at `b14c4de`. Since Batch 1, `deploy.sh` installs that set from the
lock, as checked on the GB10 at `0bc5178` (2026-10-26: A1–A5 passed, A5 equal to the committed lock,
the PyG native check 5 of 5; `docs/working/batch1-acceptance-2026-10-26/`). Windows' `deploy.cmd`
has not been run. M4's acceptance ran 2.8.0.post1; its evidence stays with that environment. After pulling, `uv sync --inexact` moves an installed
2.8.0.post1 to 2.8.1: that is a change of environment, and an acceptance made before it does not
carry over.

**What A5 says**: an installed `torch_geometric` at another version is named with both versions;
one the project declares but that is not installed is named with the lock's version; in either
case A5 no longer reads "the installed versions are the lock's". Since K8-7 (2026-11-08) A5 no
longer records whether a DataLoader worker's socket fits under the temporary directory, nor
prints a TMPDIR hint: nothing in this build starts DataLoader workers. Its other facts (the
comparison with the lock, the environment) are unchanged, and earlier reports keep the fact as
they recorded it.

**This pins today's formal deployment set, not a permanent limit** on PyG, torch or CUDA. The
versions live in the dependency and deployment layer only: no training, inference or UI code
checks a package version (the runtime check asks only whether torch and PyG import), and no
model is refused because a version differs: loading and resume follow the model contract.

**Upgrading** PyTorch, CUDA or PyG, for instance when today's combination is no longer
maintained, goes through the same entry points:

1. In `pyproject.toml`: the versions of torch, torchvision, torchaudio and torch-geometric, and,
   for another CUDA, the PyTorch wheel index (`pytorch-cu130`, its name and URL).
2. `uv lock`; then redeploy with `deploy.sh` / `deploy.cmd`. The PyG extension index follows the
   installed torch by itself.
3. On ARM, the prebuilt extension release (`PYG_PREBUILT_*` in `deploy.sh`) matches one torch and
   CUDA only. Another torch takes the source build (`scripts/build_pyg_arm.sh`, which builds
   against whatever torch is installed; `compile`, or `pull` with `PYG_ARM_ON_PULL_FAILURE=build`)
   or the fallback, so the release never blocks an upgrade; publishing a new release and updating
   those four values is optional.
4. Run the deployment probe and keep its record; checks follow what the upgrade changed, not a
   rerun of every past acceptance. The new set becomes the baseline once verified, and an older
   acceptance stays with the environment it ran in.

There is no second pipeline for old environments and no matrix of historical combinations.

**The launcher's optional accelerators** (`--flash-attn`, `--xformers`, `--sage-attn`, and on x86
FlashAttention by default) are optional third-party plugins, outside the project's native support,
and stay out of `uv.lock`. They install with `python -m pip`, which a deploy-created `.venv` lacks
until the ARM source build (`scripts/build_pyg_arm.sh`) puts it there. Every such install now goes
through constraints that `uv export` makes from `uv.lock` (`pip_install` in
`scripts/launch/shep_launch.py`): an accelerator that would need another torch, or another version
of any locked package, is refused by pip and not installed, and the launcher goes on without it;
when the constraints cannot be made (no `uv` on `PATH`, no `uv.lock`), nothing is installed. What
an accelerator brings that the lock does not hold installs as before. No model path uses these
packages (ARCHITECTURE §9).

---

## Summary: deployment checklist ✅

### Windows x86 + Blackwell

- [ ] ✅ Python 3.12 installed
- [ ] ✅ CUDA 13.0 installed
- [ ] ✅ PyTorch + CUDA working
- [ ] ✅ PyTorch Geometric working (`torch_geometric` from `uv.lock`, §4.15)
- [ ] ✅ FlashAttention-2 installed
- [ ] ✅ Voyager available (CPU vector search)
- [ ] ✅ All unit tests pass
- [ ] ✅ Model trains (watch VRAM)
- [ ] ⚠️ Dashboard on Gradio 6.30.0 (`uv sync --inexact` after pulling): a Windows backend is not tested yet (§4.10)

### ARM + Blackwell (DGX Spark)

- [ ] ✅ Architecture confirmed as aarch64
- [ ] ✅ CUDA 13.0 available
- [ ] ✅ PyTorch ARM build working
- [ ] ✅ PyTorch Geometric available (`torch_geometric` from `uv.lock`; extensions prebuilt or built, §4.15)
- [ ] ⚠️ Attention backend downgraded (xformers or SDPA)
- [ ] ✅ Vector index: cuVS (GPU) or Voyager (CPU fallback)
- [ ] ✅ Unified memory correctly identified
- [ ] ✅ Model trains (check the 128GB advantage)
- [ ] ✅ Dashboard on Gradio 6.30.0 (`uv sync --inexact` after pulling; `/ui` loads, §4.10)

---

**Version**: v3.2
**Maintenance**: kept in step with `deployment-guide.md`; update both together.
