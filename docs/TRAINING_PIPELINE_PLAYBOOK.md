# Training Pipeline Playbook — SHEPHERD-Advanced

> **Purpose**: an end-to-end, copy-pasteable runbook for taking raw HPO data to a
> trained, queryable diagnosis model. Written for fresh machines (incl. DGX Spark
> GB10) where nothing is pre-built. For the *design* of each component see
> `docs/ARCHITECTURE.md`; this doc is operational only.
>
> **Golden rule**: one **workspace** = one dataset version. Everything for a given
> dataset (KG, checkpoints, index) lives under a single
> `data/workspaces/<name>/` folder. Never mix a checkpoint from one workspace with
> the data of another — serving refuses such a checkpoint (see "The safety net"
> near the end), but the discipline is the real fix.

---

## The pipeline at a glance

```
 (manual)            Step 1                 Step 2
HPO download  →  build_knowledge_graph  →  train: a base, then its task models
                                            (WebUI)
   │                    │                      │
phenotype.hpoa     kg.json                runs/<arch>/<run>/
genes_to_           node_features.pt      (fingerprint embedded)
 phenotype.txt      edge_indices.pt
                    num_nodes.json
                    train/val_samples.json
                              ↓
                         Step 3: launch_shepherd.sh → /ui diagnosis
```

Step 1 is the **CLI prerequisite**, run once per dataset version. Step 2
(training) is driven from the dashboard: a base on **Research › Prepare ›
Training**, then its task models on **Research › Prepare › Fine-tune**. Step 3
serves inference.

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
| `--feature-dim 128` | **Input** node-feature dimension. Keep it identical across rebuilds of the same dataset — it is part of the fingerprint, and is *separate* from the model's encoder width (the model spec's, not a form field). |
| `--num-train / --num-val` | Simulated patient samples. Recommended 50K–100K train, val ≈ 10–20% of train. More = less overfitting, slower build. |

**Produces in `$WS/`**: `kg.json`, `node_features.pt`, `edge_indices.pt`,
`num_nodes.json`, `train_samples.json`, `val_samples.json`.

**Reference scale** (real HPO data, Milestone 4): ~52,308 nodes
(27,990 diseases / 19,389 phenotypes / 4,929 genes) and ~494,223 edges.

### No shortest-path step

No task model reads a shortest-path table: a disease model ranks every disease of its graph view,
and a gene model computes the exact hop distances it uses from that view in process. So nothing
here builds `shortest_paths.pt`. **Until K8** the GB10 acceptance probe still needs the table
with its bound sidecar in a workspace it accepts (its R1 and R6 checks; M4 builds its own pair on
a copy of the demo workspace); how to build the
pair for a new workspace, and why R1 is never skipped, is in the
[deployment guide](../deployment-guide.en.md) §4.3, "Until K8: the acceptance probe still needs
the table".

---

## Step 2 — Train (Research › Prepare › Training, then Fine-tune)

1. Launch the app with the service's settings (see Step 3) and open `/ui` →
   **Research (pre-clinical)** › Prepare › **Training**.
2. Choose your workspace, `$WS`'s last part (`hpo_2026_v1`), in the run bar's
   **Dataset** list.
3. The form trains a **base** (link prediction on the knowledge graph) with the
   reference recipe's base settings, each starting at the recipe's value:
   **Epochs**, **Learning rate**, **Batch size** and **Seed**, the rest under
   "More reference settings". The line under the form says which you changed
   ("Differs from the reference settings: Epochs 5 (200)"); Start sends only
   those, and the run records the rest as the recipe's.
4. Recommended first run — validate the whole pipeline cheaply: a handful of
   **Epochs** first, to confirm the loss curves move and checkpoints save. On
   Spark you can raise **Batch size** (the reference is the source's 512).
5. Watch the Trainer's validation LP accuracy (with ROC-AUC and AP beside it)
   climb in the run view. It selects the top-k saves on the val pairs; it is not
   a test result.
6. Then open Research › Prepare › **Fine-tune**: choose the dataset, a base
   checkpoint of it, and one to three **Tasks** (Disease, Gene,
   Patients-like-me; each option says whether it is ready on the dataset).
   Start trains one child run per task, in order. The job card shows each
   child's status and, once it has ended, its best save by the Trainer's pooled
   MRR; **Stop** there names the queued children it cancels.

**Every run writes into a new folder of its own** (contract M3a-2):
`$WS/runs/<arch>/<date-time>_<run name>_<id>/`, holding `config.yaml`,
`runtime.json`, `final_metrics.json`, `checkpoints/` (`last.pt`, written every
epoch, and the top-K `model-<epoch>-<metric>.pt`) and `logs/`. The optional
**Run name** field names it; the run view's header names the run once training
starts. No run writes into or prunes another run's folder.

**Continue Training** (contract M3a-3, M3a-4): open **Continue…**, pick a
model checkpoint (a base or a task model) of the run bar's dataset from the run
list or type its path, click **Check checkpoint**, then **Continue Training**.
A file without a model spec is listed on Research › Models but never continued
from the dashboard. What happens is decided by the checkpoint's own record:

- an unfinished run (stopped between epochs, or paused) is resumed with its own
  settings, shown read-only; only its remaining epochs run;
- a finished run can be extended by a number of extra epochs at its final
  learning rate, held fixed;
- a run that stopped early, a weights-only file or one trained before the M3a
  record is refused, with the reason.

The child is a new run in its own folder and records its parent by SHA-256. To
train from a checkpoint with other settings, start a new run: changing settings
from a parent (a branch) is not available from the dashboard yet.

**Stop** asks first and says how much progress is lost; a stopped run shows
**Stopped** and the latest checkpoint it can be continued from.

> CLI (no UI), in a shell given the service's `SHEPHERD_WORKSPACES_ROOT` (the
> GPU lock's one name; [deployment guide](../deployment-guide.en.md) §4.11):
> `scripts/train_model.py --data-dir "$WS" --stage base [--run-name NAME]`, or
> `--config <yaml>`; then each task model with
> `--stage finetune --task <task> --finetune-from <the base's checkpoint>`.
> `--resume <checkpoint>` continues a parent as above,
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

## There is no vector-index step — building a vector index is not part of this pipeline

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

## Step 3 — Launch & run inference

The launcher reads no environment file: set the service's settings first, in
this shell or in the systemd unit ([deployment guide](../deployment-guide.en.md)
§4.12; for Step 2, at least `SHEPHERD_DEPLOYMENT_MODE` and
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
- **Memory headroom**: on Spark the base's **Batch size** can be raised above
  the reference 512; the line under the form records it as changed.
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
| `No paths found` beside a disease, or no result at inference | Check, in order: a model is loaded (with none, nothing is served and a diagnosis is refused with the reason); the pipeline is pointed at the intended `$WS`; the submitted HPO ids exist in the KG (unknown ids are dropped before scoring and listed in `phenotype_input.unknown`; with none known the request is refused). Ranking is not path-gated: a disease task model ranks every disease of its graph view, and paths are looked for only to explain the rows shown, so a disease with no path is still ranked |
| Build, or the acceptance probe's shortest-path table (until K8), killed (OOM) | Expected only on small-RAM machines; Spark's 128 GB is fine. See `docs/MILESTONE_REPORT.md` SP memory section |
