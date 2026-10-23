# Training Pipeline Playbook — SHEPHERD-Advanced

> **Purpose**: an end-to-end, copy-pasteable runbook for taking raw HPO data to a
> trained, queryable diagnosis model. Written for fresh machines (incl. DGX Spark
> GB10) where nothing is pre-built. For the *design* of each component see
> `docs/ARCHITECTURE.md`; this doc is operational only.
>
> **Golden rule**: one **workspace** = one dataset version. Everything for a given
> dataset (KG, shortest paths, checkpoints, index) lives under a single
> `data/workspaces/<name>/` folder. Never mix a checkpoint from one workspace with
> the data of another — serving refuses such a checkpoint (see "The safety net"
> near the end), but the discipline is the real fix.

---

## The pipeline at a glance

```
 (manual)            Step 1                 Step 2                Step 3
HPO download  →  build_knowledge_graph  →  compute_shortest_  →  train_model
                                            paths                 (WebUI)
   │                    │                      │                    │
phenotype.hpoa     kg.json                shortest_paths.pt    runs/<arch>/<run>/
genes_to_           node_features.pt                           (fingerprint
 phenotype.txt      edge_indices.pt                             embedded)
                    num_nodes.json
                    train/val_samples.json
                              ↓
                         Step 4: launch_shepherd.sh → /ui diagnosis
```

Steps 1 and 2 are **CLI prerequisites** run once per dataset version. Step 3
(training) is driven from the dashboard's **Research › Prepare › Training** page.
Step 4 serves inference.

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
- **The service's settings**, as the [deployment guide](../deployment-guide.en.md)
  §4.12 states them: the dashboard lists the subdirectories of
  `SHEPHERD_WORKSPACES_ROOT` as datasets, and its Start is an operator control,
  refused unless `SHEPHERD_DEPLOYMENT_MODE=M1`. Every training run, from the
  dashboard or from a shell, takes the GPU lock `SHEPHERD_WORKSPACES_ROOT` names
  and is refused while it is unset, so give the shell where you run
  `train_model.py` the service's root too (§4.11).

Pick a workspace name up front (used in every step), directly under the
service's workspaces root, so the Training page lists it:

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

## Step 3 — Train (Research › Prepare › Training)

1. Launch the app with the service's settings (see Step 4) and open `/ui` →
   **Research (pre-clinical)** › Prepare › **Training**.
2. Choose your workspace, `$WS`'s last part (`hpo_2026_v1`), in the run bar's
   **Dataset** list.
3. Recommended first run — validate the whole pipeline cheaply:
   - **GNN Conv Type = `gat`** (AMP works, fastest, lowest memory). Get a clean
     baseline end-to-end before touching HGT.
   - A handful of epochs first to confirm loss curves move and checkpoints save.
4. Then switch **GNN Conv Type = `hgt`** for the real run. HGT forces float32
   (pyg_lib limitation) and uses more memory — this is exactly the case that OOMs
   a 16 GB 5070 Ti but fits easily on Spark's unified memory. On Spark you can
   raise **Batch Size** and **Max Subgraph Nodes** (defaults are tuned for 16 GB).
5. Watch for `hits@10` to climb. Milestone reference: 3 epochs → 0.581; the ≥ 80%
   target needs 30+ epochs.

**Every run writes into a new folder of its own** (contract M3a-2):
`$WS/runs/<arch>/<date-time>_<run name>_<id>/`, holding `config.yaml`,
`runtime.json`, `final_metrics.json`, `checkpoints/` (`last.pt`, written every
epoch, and the top-K `model-<epoch>-<metric>.pt`) and `logs/`. The optional
**Run name** field names it; the run view's header names the run once training
starts. No run writes into or prunes another run's folder.

**Continue Training** (contract M3a-3, M3a-4): open **Continue…**, pick a
checkpoint of the run bar's dataset from the run list or type its path, click
**Check checkpoint**, then **Continue Training**.
What happens is decided by the checkpoint's own record:

- an unfinished run (stopped between epochs, or paused) is resumed with its own
  settings, shown read-only; only its remaining epochs run;
- a finished run can be extended by a number of extra epochs at its final
  learning rate, held fixed;
- a run that stopped early, a weights-only file or one trained before the M3a
  record is refused, with the reason.

The child is a new run in its own folder and records its parent by SHA-256. To
train from a checkpoint with other settings, start a new run: changing settings
from a parent (a branch) is not available yet.

**Stop** asks first and says how much progress is lost; a stopped run shows
**Stopped** and the latest checkpoint it can be continued from.

> CLI (no UI), in a shell given the service's `SHEPHERD_WORKSPACES_ROOT` (the
> GPU lock's one name; [deployment guide](../deployment-guide.en.md) §4.11):
> `scripts/train_model.py --data-dir "$WS" [--run-name NAME]`, or
> `--config <yaml>`. `--resume <checkpoint>` continues a parent as above,
> `--extend-epochs N` extends a finished one, `--extend-lr LR` replaces a rate
> that ended at 0 (and only such a rate), and `--pause-after-epoch N` stops with
> a resumable checkpoint after N epochs of total progress. `--run-dir` writes into
> an explicit folder, which must be absent or empty. The old `--output-dir` and
> `--checkpoint-dir` are gone, and a configuration that still sets `output_dir`,
> `checkpoint_dir` or `log_dir` is refused.
>
> Checkpoints from before this layout (`$WS/checkpoints/`) are not listed or
> selected automatically; they are still served or measured by explicit path.
> Move them once with `scripts/migrate_checkpoints.py "$WS"` (a plan) and
> `--apply`; see the script's description.

---

## There is no step 4 — building a vector index is not part of this pipeline

This step used to build an ANN index and point the inference pipeline at it. The
vector-index subsystem has been **detached from diagnosis**: nothing on the
inference path reads an index, and `vector_index_path` no longer exists. Building
one has no effect on diagnosis.

The subsystem itself is kept — implemented, tested, and retained for planned
natural-language and vector-mapping work. Build one with `make vector-index` when
that work needs it, and read
[`RETRIEVAL_AND_CANDIDATE_DISCOVERY_FINDINGS.md`](RETRIEVAL_AND_CANDIDATE_DISCOVERY_FINDINGS.md)
first: the similarity contract must be defined before the interface is reused.

---

## Step 4 — Launch & run inference

The launcher reads no environment file: set the service's settings first, in
this shell or in the systemd unit ([deployment guide](../deployment-guide.en.md)
§4.12; for Step 3, at least `SHEPHERD_DEPLOYMENT_MODE` and
`SHEPHERD_WORKSPACES_ROOT`).

```bash
./launch_shepherd.sh          # uvicorn src.api.main:app on :8000
```

- Diagnosis UI: <http://localhost:8000/ui> (**Diagnosis (clinical)**, the first screen)
- API docs: <http://localhost:8000/docs>

---

## The safety net: a checkpoint is served only over its own graph (read this)

Every checkpoint records **what it was trained on** (`training_input_digests`):
the SHA-256 of each input the run read (the samples, the graph tensors, the
workspace's `split_manifest.json`) and of `kg.json` as that manifest binds it.
Beside it, a structural **data fingerprint**: node types, per-type node counts,
feature dimensions and edge-type count
(`src/utils/fingerprint.py:compute_fingerprint`). `scripts/train_model.py`
attaches both and the `ModelCheckpoint` callback writes them into every
checkpoint.

At inference load (`src/inference/pipeline.py`), the record is compared first
(`verify_training_record`, contract M2.4). Unless the checkpoint records this
workspace's `kg.json`, `node_features.pt`, `edge_indices.pt` and
`num_nodes.json`, each by its digest, the load is **refused before any model is
built**; a checkpoint that records none (one from before the provenance
contract) is refused the same way. This is what stops "a checkpoint loaded
against the wrong dataset", a same-shaped graph the fingerprint cannot tell
apart included. The refusal says "Checkpoint <path> does not match this
workspace's graph" and names each graph file that differs (for example
"node_features differs"). A service started with such a checkpoint serves no
model (its log's "Startup failed" line says why), and **🔒 Change model…** ›
**🔒 Load this model** on the Diagnosis page answers "Not reloaded; the model
in use is unchanged." with that reason.

The structural fingerprint (`verify_fingerprint`) is then a **diagnostic, not
the graph's identity**, and never stops a load. Its warnings surface through the
pipeline API (`fingerprint_warnings`) in the Warnings section of the Diagnosis
page's **Details ›** drawer: "Graph structure differs from the checkpoint's
structural fingerprint", or "Graph structure not compared" when the checkpoint
records no fingerprint.

**What it means for you:**
- Keep each dataset version in its own workspace; load a checkpoint against the
  same workspace it was trained on.
- If a rebuild changes the graph's files (e.g. an expanded ontology or a
  different `--feature-dim`), **the old checkpoints are refused** on the rebuilt
  workspace. That refusal is correct: retrain for the new data.
- A checkpoint refused because it "records no training_input_digests" predates
  the provenance contract: retrain it. `scripts/migrate_checkpoints.py` writes
  nothing into a checkpoint; it moves one with no record to the workspace's
  `archive/`.

---

## Spark (GB10) specific notes

- **Unified memory**: nvidia-smi/NVML report GPU memory as `[N/A]` (it is one
  shared 128 GB pool). The Training page's Resources line already handles this
  (one shared unified-memory figure; GPU util + temperature still read normally).
  Not a bug.
- **`sm_121` capability warning** from torch on GB10 is harmless/expected.
- **Memory headroom**: the 16 GB-oriented defaults (`max_subgraph_nodes=5000`,
  conservative `batch_size`, HGT batch ≤ 16) can be raised on Spark.
- **PyG**: ARM wheels were self-compiled during deploy; the startup PyG status
  line and `scripts/validate_pyg_ext.py` (5/5) confirm they are active.

---

## Troubleshooting quick reference

| Symptom | Where to look |
|---------|---------------|
| Training crashed | Red error banner on Research › Prepare › Training (the last error line and the output's tail); full run output in the console where uvicorn runs |
| "Running slow" / HGT sluggish | PyG status line at training start; if any extension `MISSING`, re-run deploy PyG step |
| A model will not load after a rebuild ("does not match this workspace's graph") | Expected: the checkpoint was trained on the old build (see the safety net above); retrain on the rebuilt workspace |
| Inference rankings look wrong | The Diagnosis page's **Details ›** drawer: which workspace is served and whether the model was trained on it ("the same dataset ✓" or "another dataset"), and the structural fingerprint's warnings |
| `No paths found` / empty results at inference | Check, in order: the pipeline is pointed at the intended `$WS`; `kg.json` and `shortest_paths.pt` are present there; the submitted HPO ids exist in the KG (unknown ids are dropped before scoring); at least one phenotype has a KG path to a disease — candidate discovery is BFS-gated, so a phenotype with no qualifying path yields nothing |
| Build/SP killed (OOM) | Expected only on small-RAM machines; Spark's 128 GB is fine. See `docs/MILESTONE_REPORT.md` SP memory section |
