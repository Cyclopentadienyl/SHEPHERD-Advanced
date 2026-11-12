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
- Confirm PyG native extensions are actually active:
  ```bash
  .venv/bin/python scripts/validate_pyg_ext.py      # expect 5/5 on GB10
  ```
  At training start the console/log also prints one line, e.g.
  `PyG native extensions: pyg_lib 0.6.0 OK | torch_scatter ... OK | ...`.
  If any read `MISSING`, you are on slow fallback kernels — re-run the deploy
  PyG-extension step before training.
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
here builds `shortest_paths.pt`, and the GB10 acceptance probe accepts a workspace without one.

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
7. **The frozen-backbone control** (since research comparison Stage 9b,
   2026-11-10): a child with `frozen_parts` `graph_side` keeps its base's input
   module, encoder and LP head exactly as they are (no gradient, no optimizer
   group, evaluation mode for the whole run: BatchNorm's statistics are used and
   never updated, dropout is off) and trains only its patient encoder and task
   head. Its loss keeps the recipe's coefficients; its record says the LP term
   has no trainable parameter, which parts were frozen and how they ran, and the
   SHA-256 of the downstream weights it started from, so two children of one
   base can be checked to start alike (with the recipe's seed they do). Both are
   ordinary task models, measured, registered, listed and exported like any
   other, the freeze among their learning settings; Models › Results' Compare
   and the export show each one's weight transfer and loss identity as recorded,
   so the two downstream starts are compared there (an arm paused and resumed
   keeps its stage's transfer in its record). It is set on the command
   line (`--frozen-parts graph_side`, below) or in the job runner's Fine-tune
   settings (`frozen_parts`, `POST /api/v1/training/finetune`); the Fine-tune
   page does not offer it in this round.

**Every run writes into a new folder of its own** (contract M3a-2):
`$WS/runs/<encoder kind>/<date-time>_<run name>_<id>/` (`gatv2` for the
reference recipe), holding `config.yaml`, `runtime.json`, `final_metrics.json`,
`checkpoints/` (`last.pt`, written every epoch, and the top-K
`model-<epoch>-<metric>.pt`) and `logs/`. The optional
**Run name** field names it; the run view's header names the run once training
starts. No run writes into or prunes another run's folder. The lists read
every folder under `$WS/runs/`, whatever its name (since K8-7b, 2026-11-08): a
model moved into a folder of another name is still listed, chosen and served,
judged by its own record.

**Continue Training** (contract M3a-3, M3a-4): open **Continue…**, pick a
model checkpoint (a base or a task model) of the run bar's dataset from the run
list or type its path, click **Check checkpoint**, then **Continue Training**.
A file without a model spec is listed on Research › Models ("Record not readable")
but never continued. What happens is decided by the checkpoint's own record:

- an unfinished run (stopped between epochs, or paused) is resumed with its own
  settings, shown read-only; only its remaining epochs run;
- a finished run can be extended by a number of extra epochs at its final
  learning rate, held fixed;
- a run that stopped early, a weights-only file or one trained before the M3a
  record is refused, with the reason, and so is a file without a model spec
  ("written before the model spec: not readable by this build; start a new
  run").

Only the current settings are inherited (since K8-7, 2026-11-08): a setting the
parent recorded that this version no longer has (`num_workers`, the old model
settings such as `conv_type`) is neither inherited nor a reason to refuse, and a
parent missing a current setting is refused. Check checkpoint and the run decide
alike. Since research comparison Stage 9a (2026-11-10) checkpoints record training
rules 2, which add `frozen_parts` (which parts of a finetune do not train). A
rules-1 parent still continues: one settings decode reads it with `frozen_parts`
none, from what its record shows (rules 1 had no freeze), lists it with that
source beside the inherited settings (the drawer's "Settings taken from this
checkpoint", a line of its own in the training log) and records that source on
the continued run; its file is not changed. A parent
with no rules recorded (K4's) or rules this version does not know is refused, and
a continued or branched run never changes the freeze: a frozen run resumed or
extended stays frozen, with the same optimizer groups restored, and a branch of
it keeps its freeze (since Stage 9b). A resumed or extended run records its
stage's weight transfer, its parent's unchanged, so every checkpoint of a stage
names the downstream weights the stage started from (since the Stage 9b fixes).

The child is a new run in its own folder and records its parent by SHA-256. To
train from a checkpoint with other settings, or again with the same settings,
branch it: a new run from its weights with a new optimizer, from the command line
(`--branch-from`) or `POST /api/v1/training/branch`; the dashboard's Branch mode
comes with the Training page's next stage. Models says what each branch changed,
or "Same settings as its parent".

**Stop** asks first and says how much progress is lost; a stopped run shows
**Stopped** and the latest checkpoint it can be continued from.

> CLI (no UI), in a shell given the service's `SHEPHERD_WORKSPACES_ROOT` (the
> GPU lock's one name; [deployment guide](../deployment-guide.en.md) §4.11):
> `scripts/train_model.py --data-dir "$WS" --stage base [--run-name NAME]`, or
> `--config <yaml>`; then each task model with
> `--stage finetune --task <task> --finetune-from <the base's checkpoint>`,
> adding `--frozen-parts graph_side` (or `frozen_parts: graph_side` in the
> configuration) for the frozen-backbone control (item 7 above); a base given a
> freeze is refused, since nothing would train.
> Every run names its `--stage`, a continued one included (a run without it is
> refused, naming `--stage`): `--resume <checkpoint>`, with that run's
> `--stage`, continues a parent as above,
> `--extend-epochs N` extends a finished one, `--extend-lr LR` replaces a rate
> that ended at 0 (and only such a rate), and `--pause-after-epoch N` stops with
> a resumable checkpoint after N epochs of total progress. `--run-dir` writes into
> an explicit folder, which must be absent or empty. The old `--output-dir` and
> `--checkpoint-dir` are gone, and a configuration that still sets `output_dir`,
> `checkpoint_dir` or `log_dir` is refused. So is one naming a setting this
> version does not have ("names settings this version does not have: …"); since
> K8-7 `num_workers` and the old model settings (`conv_type`, `hidden_dim`, …) are
> among them, and `--conv-type`, `--hidden-dim` and `--num-layers` are gone.
> A refusal is one line, `train_model: refused: <why> Nothing was written.`,
> exit 1 (since K8-7b); an error after the run's folder is made is a failed run,
> logged as "Training failed:" with its traceback, as is an unexpected error (one
> that is no refusal) before it.
>
> Checkpoints from before this layout (`$WS/checkpoints/`) are not listed or
> selected automatically; they stay on disk, reached by explicit path only. A
> checkpoint without a model spec (every file written before the model bundles)
> is refused at load, before any model is built.

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

A checkpoint whose record matches but which carries no model spec (a
ShepherdGNN history file) is refused next, still before any model is built:
"written before the model spec: not readable by this build; start a new run".

The structural fingerprint is **recorded, and compared by nothing at load**
(since K8-4): the record's digests above are the graph's identity. The pipeline
API reports no fingerprint warnings, and the Diagnosis page's **Details ›**
drawer shows no structural line.

**What it means for you:**
- Keep each dataset version in its own workspace; load a checkpoint against the
  same workspace it was trained on.
- If a rebuild changes the graph's files (e.g. an expanded ontology or a
  different `--feature-dim`), **the old checkpoints are refused** on the rebuilt
  workspace. That refusal is correct: retrain for the new data.
- A checkpoint refused because it "records no training_input_digests" predates
  the provenance contract: retrain it.

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
| "Running slow" | PyG status line at training start; if any extension `MISSING`, re-run deploy PyG step |
| A model will not load after a rebuild ("does not match this workspace's graph") | Expected: the checkpoint was trained on the old build (see the safety net above); retrain on the rebuilt workspace |
| Inference rankings look wrong | The Diagnosis page's **Details ›** drawer: which workspace is served and whether the model was trained on it ("the same dataset ✓" or "another dataset"), the graph's state ("graph ✓"), whether the task uses SP, and the dataset record ("dataset record ✓") |
| `No paths found` beside a disease, or no result at inference | Check, in order: a model is loaded (with none, nothing is served and a diagnosis is refused with the reason); the pipeline is pointed at the intended `$WS`; the submitted HPO ids exist in the KG (unknown ids are dropped before scoring and listed in `phenotype_input.unknown`; with none known the request is refused). Ranking is not path-gated: a disease task model ranks every disease of its graph view, and paths are looked for only to explain the rows shown, so a disease with no path is still ranked |
| Build killed (OOM) | Expected only on small-RAM machines; Spark's 128 GB is fine. The build has no shortest-path step (Step 1), so `docs/MILESTONE_REPORT.md`'s SP memory appendix, about loading the retired table, does not apply |
