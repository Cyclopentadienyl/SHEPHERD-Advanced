# Training Pipeline Playbook — SHEPHERD-Advanced

> **Purpose**: an end-to-end, copy-pasteable runbook for taking raw HPO data to a
> trained, queryable diagnosis model. Written for fresh machines (incl. DGX Spark
> GB10) where nothing is pre-built. For the *design* of each component see
> `docs/ARCHITECTURE.md`; this doc is operational only.
>
> **Golden rule**: one **workspace** = one dataset version. Everything for a given
> dataset (KG, shortest paths, checkpoints, index) lives under a single
> `data/workspaces/<name>/` folder. Never mix a checkpoint from one workspace with
> the data of another — the fingerprint check (see end) is the safety net, but the
> discipline is the real fix.

---

## The pipeline at a glance

```
 (manual)            Step 1                 Step 2                Step 3              Step 4
HPO download  →  build_knowledge_graph  →  compute_shortest_  →  train_model    →  build_index
                                            paths                 (WebUI)
   │                    │                      │                    │                  │
phenotype.hpoa     kg.json                shortest_paths.pt    checkpoints/*.pt   vector_index.*
genes_to_           node_features.pt                           (fingerprint        (ANN for
 phenotype.txt      edge_indices.pt                             embedded)           inference)
                    num_nodes.json
                    train/val_samples.json
                              ↓
                         Step 5: launch_shepherd.sh → /ui diagnosis
```

Steps 1, 2, 4 are **CLI prerequisites** run once per dataset version. Step 3
(training) is driven from the **WebUI Training Console**. Step 5 serves inference.

---

## Prerequisites

- `./deploy.sh` already run successfully (creates `.venv/` from `uv.lock`). All
  `python` calls below use that interpreter — prefix with `.venv/bin/python` if it
  is not on your `PATH`.
- Confirm PyG native extensions are actually active (matters for HGT speed):
  ```bash
  .venv/bin/python scripts/validate_pyg_ext.py      # expect 5/5 on GB10
  ```
  At training start the console/log also prints one line, e.g.
  `PyG native extensions: pyg_lib 0.6.0 OK | torch_scatter ... OK | ...`.
  If any read `MISSING`, you are on slow fallback kernels — re-run the deploy
  PyG-extension step before training (especially for HGT).

Pick a workspace name up front (used in every step):

```bash
export WS=data/workspaces/hpo_2026_v1      # name it per dataset version
```

---

## Step 0 — Download HPO annotation files (manual)

Download these two files from the HPO annotation page
(<https://hpo.jax.org/data/annotations>) into `data/external/`:

| File | Description |
|------|-------------|
| `phenotype.hpoa` | phenotype–disease annotations (~250K rows) |
| `genes_to_phenotype.txt` | gene–phenotype / gene–disease links (~200K rows) |

```bash
ls data/external/phenotype.hpoa data/external/genes_to_phenotype.txt   # verify
```

Ontology files (HPO, MONDO `.obo`) are **auto-downloaded** by `OntologyLoader`
(cached in `~/.shepherd/ontologies/`) — you do not fetch these manually.

---

## Step 1 — Build the knowledge graph

```bash
.venv/bin/python scripts/build_knowledge_graph.py \
    --workspace "$WS" \
    --external-dir data/external \
    --feature-dim 128 \
    --generate-samples \
    --num-train 80000 \
    --num-val 15000
```

| Flag | Notes |
|------|-------|
| `--feature-dim 128` | **Input** node-feature dimension. Keep it identical across rebuilds of the same dataset — it is part of the fingerprint, and is *separate* from the model's `hidden_dim` (set in the UI, e.g. 256). |
| `--num-train / --num-val` | Simulated patient samples. Recommended 50K–100K train, val ≈ 10–20% of train. More = less overfitting, slower build. |

**Produces in `$WS/`**: `kg.json`, `node_features.pt`, `edge_indices.pt`,
`num_nodes.json`, `train_samples.json`, `val_samples.json`.

**Reference scale** (real HPO data, Milestone 4): ~52,308 nodes
(27,990 diseases / 19,389 phenotypes / 4,929 genes) and ~494,223 edges.

---

## Step 2 — Pre-compute shortest paths

This is the heaviest prerequisite (it powers the SP-similarity term in scoring).

```bash
.venv/bin/python scripts/compute_shortest_paths.py \
    --kg-path "$WS/kg.json" \
    --output-dir "$WS" \
    --max-hops 5
```

**Produces**: `$WS/shortest_paths.pt` (must live in the same dir as
`node_features.pt`).

**Cost reference** (Milestone 4 dataset): ~374 million (phenotype → target)
pairs, file ≈ **8.9 GB**; loading it peaks ~12 GB RAM, steady ~8–9 GB. This grows
with KG size — budget disk + RAM accordingly if you expand the ontology later.
On Spark's 128 GB unified memory this is comfortable.

---

## Step 3 — Train (WebUI Training Console)

1. Launch the app (see Step 5) and open `/ui` → **Training Console**.
2. Set **Data Dir** to your workspace (`$WS`).
3. Recommended first run — validate the whole pipeline cheaply:
   - **Conv Type = `gat`** (AMP works, fastest, lowest memory). Get a clean
     baseline end-to-end before touching HGT.
   - A handful of epochs first to confirm loss curves move and checkpoints save.
4. Then switch **Conv Type = `hgt`** for the real run. HGT forces float32
   (pyg_lib limitation) and uses more memory — this is exactly the case that OOMs
   a 16 GB 5070 Ti but fits easily on Spark's unified memory. On Spark you can
   raise `batch_size` and `max_subgraph_nodes` (defaults are tuned for 16 GB).
5. Watch for `hits@10` to climb. Milestone reference: 3 epochs → 0.581; the ≥ 80%
   target needs 30+ epochs.

Checkpoints (with the data fingerprint embedded) are saved under
`$WS/checkpoints/` (`last.pt` + top-K `model-<epoch>-<metric>.pt`).

> CLI fallback (no UI): `scripts/train_model.py --config <yaml>` exists, but the
> Training Console is the intended path — it generates the config and monitors
> progress, PyG status, resources, and errors (the dedicated error banner).

---

## Step 4 — Build the vector index (for inference ANN)

Enables ANN candidate discovery so high-GNN-similarity diseases surface even
without an explicit KG path.

```bash
.venv/bin/python scripts/build_index.py \
    --checkpoint "$WS/checkpoints/last.pt" \
    --data-dir "$WS" \
    --output "$WS/vector_index" \
    --node-types disease \
    --backend auto
```

`--backend auto` picks cuVS (Linux GPU) or Voyager (cross-platform CPU). Point the
inference pipeline's `vector_index_path` at `$WS/vector_index`.

---

## Step 5 — Launch & run inference

```bash
./launch_shepherd.sh          # uvicorn src.api.main:app on :8000
```

- Diagnosis UI: <http://localhost:8000/ui> (Diagnosis Panel tab)
- API docs: <http://localhost:8000/docs>

---

## The fingerprint safety net (read this)

Every checkpoint embeds a **data fingerprint** — the structural identity of the
graph it was trained on: node types, per-type node counts, feature dimensions,
and edge-type count (`src/utils/fingerprint.py:compute_fingerprint`, attached in
`scripts/train_model.py` and the `ModelCheckpoint` callback).

At inference load (`src/inference/pipeline.py` →
`verify_fingerprint`), the checkpoint's fingerprint is compared against the
currently loaded data. Mismatches produce **warnings, not hard errors**, and
surface up through the pipeline API (`fingerprint_warnings`) into the Diagnosis
Panel. This is what catches "a checkpoint was loaded against the wrong dataset"
before it silently produces garbage rankings.

**What it means for you:**
- Keep each dataset version in its own workspace; load a checkpoint against the
  same workspace it was trained on.
- If you rebuild the KG (e.g. expand the ontology, more samples, different
  `--feature-dim`), the fingerprint changes — **old checkpoints will warn**. That
  warning is correct: retrain (or rebuild the index) for the new data.
- A "legacy checkpoint has no fingerprint" warning means an old checkpoint that
  predates this mechanism — verify compatibility manually.

---

## Spark (GB10) specific notes

- **Unified memory**: nvidia-smi/NVML report GPU memory as `[N/A]` (it is one
  shared 128 GB pool). The Training Console already handles this (shows a single
  unified-memory gauge; GPU util + temperature still read normally). Not a bug.
- **`sm_121` capability warning** from torch on GB10 is harmless/expected.
- **Memory headroom**: the 16 GB-oriented defaults (`max_subgraph_nodes=5000`,
  conservative `batch_size`, HGT batch ≤ 16) can be raised on Spark.
- **PyG**: ARM wheels were self-compiled during deploy; the startup PyG status
  line and `scripts/validate_pyg_ext.py` (5/5) confirm they are active.

---

## Troubleshooting quick reference

| Symptom | Where to look |
|---------|---------------|
| Training crashed | Red error banner in the Training Console (filtered traceback tail); full run output in the console where uvicorn runs |
| "Running slow" / HGT sluggish | PyG status line at training start; if any extension `MISSING`, re-run deploy PyG step |
| Inference rankings look wrong after a rebuild | Fingerprint warnings in the Diagnosis Panel — likely a checkpoint/data version mismatch |
| `No paths found` / empty results at inference | Confirm `shortest_paths.pt` and the vector index exist in the workspace, and the pipeline points at the right `$WS` |
| Build/SP killed (OOM) | Expected only on small-RAM machines; Spark's 128 GB is fine. See `docs/MILESTONE_REPORT.md` SP memory section |
