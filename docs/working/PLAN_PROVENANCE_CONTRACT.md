# PLAN — the provenance contract: every link in the pipeline checked where it is used

**Status: draft for review, revision 1.** Nothing here is implemented. Facts about the code are
cited at `627ed08`. §1 records decisions already made; §4 is the order of work.

**What this is.** The project's goal for this stage, in the owner's words: the model must always
match the exact setup that trained it, dataset version and every other stage's settings included,
with nothing misaligned between stages. This plan turns that into checkable relations between
artifacts, says which exist today and which do not, and orders the work to close the gaps.

**What this is not.** It defends against misalignment, not against a malicious account holder.
It adds no login, no keys, no signing service and no hash chain. SLSA provenance
(https://slsa.dev/spec/v1.0/provenance), the in-toto attestation statement
(https://github.com/in-toto/attestation) and W3C PROV (https://www.w3.org/TR/prov-dm/) inform the
shape of the records — inputs, parameters, producer, outputs. The project does not adopt those
platforms and claims no conformance to them.

---

## 1. Decisions this plan rests on

| # | Decision | By |
|---|---|---|
| 1 | **A model whose recorded training graph does not match the graph it is used with is invalid.** Its SP scores, and every other score derived through the training data, cannot match it. Such a pairing is refused | Owner, 2026-10-06 |
| 2 | **Artifacts from the old pipeline are not supported.** Its training data and models were not built rigorously and will not meet these checks. There is no compatibility mode and no long-lived warning-only mode. They are archived, not deleted: the pipeline stops using them | Owner, 2026-10-06 |
| 3 | **The threat is misalignment between stages, not an account holder.** No accounts, keys, signing or chain | Owner and reviewer, 2026-10-06 |
| 4 | **B-2's fail-closed core lands first, as its own change, and takes effect together with the first enforced check.** Otherwise a check that finds a mismatch makes the model fail to load, the pipeline falls back to another scorer, and the user still gets a result that looks successful | Reviewer, 2026-10-06 |
| 5 | **Backlog item 14 is split into two phases.** Phase 1 is a complete CLI flow: run a test, verify the inputs, keep the report, record it automatically or by hand, and show the result on the existing model status. Phase 2 is the Test tab, job management and advanced ordering | Owner and reviewer, 2026-10-06 |
| 6 | **Milestones follow real dependencies.** A full environment refactor, a full provenance platform and every historical defect are not prerequisites of item 14. Each milestone delivers a complete flow that can be run and accepted | Reviewer, 2026-10-06 |

## 2. The rule

> The formal pipeline accepts an input only when it matches what the consuming step's governing
> record names. A mismatch, a missing required record, or a required check that cannot be
> completed is a refusal.

It is not necessary to show that a particular mismatch would make a particular score wrong.
Being unable to verify the pairing is reason enough.

1. **Identity is the SHA-256 of bytes.** A path or file name locates an artifact and never
   identifies it. Location checks are still used where location is the point — an overwrite
   guard, a directory containment check — but never as identity.
2. **Each producer records, in the record it already writes:**
   - the inputs it consumed, by role and digest;
   - the parameters actually in effect: resolved, not requested. A request that did not take
     effect is recorded as such — compile requested but fallen back, AMP requested on CPU;
   - the code revision that produced it;
   - the environment facts that bound how its numbers can be reproduced (§5.3).
3. **Each consumer checks, at its entry point, the relations it depends on**, before it publishes
   or writes anything. Some relations can only be checked after a load — a checkpoint's recorded
   inputs live inside the checkpoint — so the order is "before anything is published or
   written", not "before any load".
   - States: *verified*, *mismatch*, *unrecorded*, *unverifiable*.
   - For a required relation, anything but *verified* refuses.
4. **Claims are limited to the relations checked** (§6).

## 3. The relations

Each row is one "X must match Y". *Today* is the state at `627ed08`.

| # | Relation | Recorded today | Checked today | Required from |
|---|---|---|---|---|
| R1 | Workspace graph files (`kg.json`, `node_features.pt`, `edge_indices.pt`, `num_nodes.json`) ↔ `split_manifest.json` | Manifest `artifacts` | `verify_graph_artifacts` (`src/kg/artifacts.py:58`) at `scripts/train_model.py:716` and `scripts/measure_scorer.py:128`; `verify_graph_source` (`artifacts.py:357`) at serving (`src/inference/pipeline.py:501`). **Refuses** | Exists |
| R2 | Generated samples ↔ manifest | Manifest | `verify_generated_cohorts` (`src/evaluation/cohort.py:216`) at `train_model.py:717`, `measure_scorer.py:134`. **Refuses** | Exists |
| R3 | **Model ↔ the graph it consumed**: `kg.json` (node index), `node_features`, `edge_indices`, `num_nodes` | `training_input_digests` (`train_model.py:551-574`; written by `src/training/callbacks.py:325-327`) holds the three tensors and `split_manifest`, **not `kg.json`** | **Structure only, and only a warning** (`src/utils/fingerprint.py:156-176`; `pipeline.py:909-918`). The digest comparison was "deliberately deferred" (`fingerprint.py:168-176`); only the probe performs it (`scripts/probe_deployment.py:812-836`). Measurement fits the weights by shape (`measure_scorer.py:593-594`) | **M2** |
| R4 | SP table ↔ the graph | Sidecar `kg_digest`, `build_id`, `max_hops` (`scripts/compute_shortest_paths.py:400-405, 481-489`) | At serving (`pipeline.py:724-747`): a `build_id` or `kg_digest` mismatch refuses; an **unrecorded** pair is served with a log warning (`docs/working/PLAN_SP_ARTIFACT_INTEGRITY.md:272`, "Unknown … serves") | **M2** (unrecorded refuses) |
| R5 | Workspace ↔ its dataset record (`kg.provenance.json`) | Written at build (`src/kg/provenance.py`); bound from the manifest | **Read by nothing in production**: `workspace_provenance_status` (`artifacts.py:208`) has no caller outside the module. Policy so far: "reported, not enforced" (`artifacts.py:123-137`; `docs/working/PLAN_ONTOLOGY_PROVENANCE.md:249-254`) | **M3b** (decision 2 reverses that policy for the formal pipeline) |
| R6 | Model ↔ the training settings, code and environment in effect | `config` = `TrainerConfig` plus `model_config` (`src/training/trainer.py:937-981`). Not recorded: the loader and sampling settings (`DataLoaderConfig`, `train_model.py:585-592`), compile (`train_model.py:760-777`), the code revision, the environment | — | **M3a** |
| R7 | Model ↔ its resume parent | The parent's digest (`train_model.py:535-545`) | **Nothing compared**: `load_checkpoint` restores state without looking at the parent's inputs or config (`trainer.py:1009-1044`). A missing parent only warns, and the run trains from scratch (`train_model.py:505-507`). The WebUI's default resume target is `last.pt` (`src/webui/components/training_console.py:398-404`), which `ModelCheckpoint` overwrites (`callbacks.py:271-274`) | **M3a** |
| R8 | A diagnosis result ↔ the model, graph and SP table that produced it | `InferenceResult.model_version` / `kg_version` exist (`src/core/types.py:448-449`) | Filled with a class version and `"unknown"` (`pipeline.py:1062, 1129-1130, 1703`). The API answers with the literal `"1.0.0"` (`src/api/routes/diagnose.py:351`) | **M3c** |
| R9 | Test result ↔ model, cohort and measurement settings | Measurement manifest (`measure_scorer.py:135-155, 342-390`); ledger (`src/evaluation/sidecar.py:291-319`) | R1 and R2 are checked; R3 is not (above) | M2 (R3), **M4** |
| R10 | Test cohort ↔ the model's own training and selection splits | `cohort_kind=generated` claims disjointness by construction **relative to the workspace's cut** (`src/evaluation/measurement.py:527-530`), not relative to the model | — | **M4** |
| R11 | Environment facts that bound reproduction | Measurement: `torch_version`, `cuda_version`, the device string and the determinism flags (`measure_scorer.py:376-389`). No GPU model, compute capability, driver or cuDNN version. Training and serving record none tied to their artifacts | — | M3a (training), M4 (measurement) |

**What the table shows.** The links inside a workspace refuse today (R1, R2), and so does a
broken SP pair. Every link with the model on one side is checked by structure only, or recorded
and unread, or absent: R3, R6, R7, R8, R10. So is the dataset record (R5).

**R3's consequence, concretely.** A checkpoint trained on workspace A can be served or measured
on a workspace B whose tensors have the same shapes — for example, the same graph exported with
another feature seed or another torch version. The structural check passes, the strict weight
load passes, and the cached embeddings are computed from B. Under decision 1 that pairing is
invalid, and nothing stops it.

## 4. Order of work

Each milestone is one reviewable change, or a few, and ends in a flow that can be run and
accepted.

### M1 — B-2's fail-closed core

**Today.** `_load_model_from_checkpoint` returns `None` when the file is missing, when graph data
is absent, or when the model cannot be built (`pipeline.py:899-901, 920-926, 934-938`).
`_init_gnn_inference` then returns without the GNN (`pipeline.py:513-517`). The pipeline is
published as success, and diagnosis serves path-reasoning scores in place of the model.

**Change:**
- **The error reaches the caller.** A requested checkpoint that does not load raises out of
  `build_pipeline`, and nothing above turns it back into `None`.
- **Startup** leaves no pipeline. `/diagnose` already answers 503 when a configured pipeline
  failed (`diagnose.py:269-277`).
- **A reload is rejected, and the pipeline already serving stays.** This uses the existing
  build-the-candidate-then-publish order (`src/api/routes/pipeline.py:391-466`). Rejecting a
  failed candidate is not shutting down a good service.
- **A diagnosis that needs the GNN never returns path-reasoning scores in its place.**

**Scope.** This is B-2's fail-closed half only. B-2's discriminated result semantics stay with
the result-union work it shares with B-1 (`docs/DISEASE_SCORER_POLICY.md` §7). Serving with path
reasoning only, when no checkpoint is configured at all, is unchanged here; whether it needs an
explicit opt-in is B-2's policy question (§8).

**Acceptance:**
- a startup with a checkpoint that cannot load leaves `/diagnose` at 503;
- a reload to such a checkpoint is refused, and the previous pipeline keeps serving;
- tests for both, and a mutant that restores `return None` is caught.

### M2 — the model↔graph relation, producer and consumer together

This is the first enforced check. It lands with M1 or after it, never before.

- **Producer.** Training adds `kg.json`'s digest to its input roles (role `kg`), beside the three
  tensors and the manifest (`train_model.py:551-574`).
- **Consumer: serving and measurement.**
  - They compare the checkpoint's recorded `kg`, `node_features`, `edge_indices` and `num_nodes`
    digests with the workspace's verified ones: R1's return value, kept at serving
    (`pipeline.py:501`) and today discarded at measurement (`measure_scorer.py:128`).
  - A missing `training_input_digests`, or a missing role, refuses.
  - The comparison lives in the slot `verify_fingerprint` deferred (`fingerprint.py:168-176`),
    so there is one model↔graph comparator. The structural fingerprint stays as a diagnostic,
    never as identity. The probe asserts through the same function instead of its own
    comparison (`probe_deployment.py:812-836`).
  - A refusal raises (M1), so it cannot become a fallback.
- **SP.** An unrecorded SP binding refuses. This ends the legacy "serves with a warning" state
  (R4), under decision 2.
- **Inventory before deployment.** A read-only listing of every checkpoint, SP table and
  workspace that M2 would refuse, and why. It is an option of an existing audit
  (`scripts/audit_checkpoint_family.py` or the deployment probe), not a new tool. It is
  deployment preparation, not a second execution path. Old artifacts are archived, not deleted.
- **Consequence.** No existing checkpoint records the `kg` role, so every one of them is refused
  after M2 and the model is retrained. This follows decision 2.

**Acceptance:**
- a checkpoint from workspace A is refused against a same-shaped workspace B, at serving and at
  measurement;
- a fresh build → train → serve passes;
- the inventory runs on the homelab.

### M3 — producers record what consumers need, and consumers check it

Three changes, each complete. M3a can proceed in parallel with M3b and M3c.

**M3a — training records the setup that took effect (R6, R7, R11).**
- **`checkpoint['config']` gains the values in effect:**
  - the loader and sampling settings;
  - compile, both requested and whether it enabled;
  - AMP and warmup, both requested and effective.
- **`producer`:** the code revision, plus a dirty flag that counts untracked files.
- **`environment`:** §5.3.
- **The seed is applied before the data loaders and the model are built.** Today it is applied
  in `Trainer.__init__` (`trainer.py:255-257`), after the loaders (`train_model.py:748`) and the
  model (`train_model.py:751`), so the recorded seed does not govern initialisation.
- **Resume:**
  - a missing parent refuses instead of training from scratch;
  - a parent whose recorded graph roles differ from the current workspace's refuses;
  - a write-time guard in `ModelCheckpoint` refuses to overwrite a file whose digest is the run's
    own resume input.
- **Training start's explicit `checkpoint_dir`**, honoured verbatim today, is restricted to
  directories the server resolves. This is a visible behaviour change, listed in the
  milestone's acceptance and in the documentation.
- **One constraint on every new field:** measurement loads checkpoints with `weights_only=True`
  (`measure_scorer.py:593`), so everything new is a primitive container or a scalar.

**M3b — the dataset record is required (R5).**
- Training, serving and measurement refuse a workspace whose `kg.provenance.json` is missing,
  unreadable or does not match. The manifest's binding to the record becomes required at write
  and read.
- A record that is bound but declares a synthetic origin or missing roles is shown with that
  origin, never as plain *verified*.

**M3c — what is served is identified (R8).**
- The checkpoint's digest is taken from the bytes that are loaded.
- `InferenceResult.model_version` and `kg_version` carry real identities: the checkpoint and
  `kg.json` digests, the SP `build_id`, the scoring mode and the serving settings in effect,
  through `get_pipeline_config()`.
- The diagnose response copies them from the result it is answering with, not from shared
  state a reload could swap.
- The CSV and report exports carry them, and the literal versions go.

### M4 — item 14, phase 1

`docs/working/PLAN_TEST_RESULTS.md` revision 4 holds the detail.
- **Import:** a test cohort version is imported from the CLI, bound to the graph.
- **Run:** a Mode C test runs from the CLI with R1, R2, R3, R5 and the cohort↔graph binding
  checked. It records R10 as two labels — overlap with the model's training split, and with its
  selection split — and records the R11 facts.
- **Keep and record:** every report is kept, and is recorded automatically (`--record`) or later
  by hand, from the CLI.
- **Display:** the Diagnosis tab's existing model status shows the loaded checkpoint's records,
  found by its digest (M3c) and read through the API.
- **No new protection mechanism is needed.** Phase 1 adds no operation that the network can
  trigger to change state; a UI "register" button would be such an operation, and it belongs to
  phase 2 with its protection.

### M5 — item 14, phase 2

The Test tab, the job runner with progress and cancellation, the model list, the network-side
restrictions those operations need, and the advanced ordering and arbitration. All of it is in
`PLAN_TEST_RESULTS.md`.

### Not prerequisites

Each of these is done when its entry point is next touched, or on its own merit:
- **one environment collector**, merging today's scattered ones as each entry point is updated;
- **the KG build record v2:**
  - builder parameters;
  - annotation-file digests taken from the bytes parsed, not re-read afterwards
    (`scripts/build_knowledge_graph.py:342-343`);
  - the annotation release labels;
  - edge-drop counters;
- **an SP producer check** against the workspace it publishes into
  (`compute_shortest_paths.py:419-425, 458`);
- **thresholds for source compatibility.**

## 5. What changes in existing records

No new record type is introduced, except item 14's cohort version and report store, which its
own plan defines.

### 5.1 By record

| Record | Change | Milestone |
|---|---|---|
| Checkpoint (`callbacks.py:296-327`) | `training_input_digests` gains `kg`; `config` gains the values in effect; new `producer` and `environment` keys | M2, M3a |
| `split_manifest.json` | Binding to `kg.provenance.json` required | M3b |
| `kg.provenance.json` | Unchanged until the KG record v2 | — |
| SP sidecar | Unchanged; it already records `kg_digest`, `build_id` and `max_hops` | — |
| Measurement manifest and report | Environment facts; the workspace's graph digests for both cohort kinds | M4 |
| Ledger | v2, per item 14 | M4 |
| Serving state | `get_pipeline_config()` and `InferenceResult` carry identities and the settings in effect | M3c |

### 5.2 Required versus displayed

| Kind of relation | Treatment |
|---|---|
| R1–R5, R7 | **Required**: anything but *verified* refuses |
| R6, R11 | **Recorded**, and displayed beside results. An environment difference never refuses |
| R10 | **Recorded as two labels.** It never refuses, because writing the true label is enough |

### 5.3 Environment facts

**What is recorded:**
- the Python, torch and CUDA versions, and torch's build identifier;
- the cuDNN version;
- the GPU model, compute capability and driver;
- whether deterministic algorithms are on, and in which mode;
- the TF32 setting;
- the PyG version, and which of its native extensions are importable.

**What they are for.** They say under which conditions a number was produced. PyTorch promises
reproducibility only on the same platform, device and versions
(https://docs.pytorch.org/docs/stable/notes/randomness.html). So a difference between two results
under different recorded conditions calls for investigation under those conditions, not a
verdict of defect. Deterministic mode guards against known non-deterministic operations; it does
not certify correctness, and it does not promise equal numbers across environments.

## 6. The claim boundary

**For the documentation:**

> When every required check passes, the pipeline establishes this: for each relation listed as
> *verified*, the files compared were byte-identical to what their records name at the time of
> the check, and the records agree with each other at those relations.
>
> It does not establish:
> - who produced or changed a file;
> - that a record is truthful;
> - integrity against anyone able to modify the data and its records together. The records are
>   unsigned, and a digest is not a signature (`docs/working/PLAN_ONTOLOGY_PHASE2.md:812`);
> - that a result reproduces under other recorded conditions;
> - that a test cohort stayed independent, which is an institutional operating rule;
> - clinical validity.

**For the UI**, one line beside the identities:

> Checked: these files are the ones their records name, and the records agree on the relations
> marked ✓. Not checked: who changed them, or anything a person able to edit both files and
> records could alter.

## 7. Out of scope

- Accounts, keys, signing, an append-only or chained log.
- Conformance claims to SLSA, in-toto or PROV.
- Reading or migrating old-pipeline artifacts (decision 2).
- `scripts/build_index.py`, whose output is detached from diagnosis.

## 8. Questions left open

1. **Revision source.** Does any deployment run without `.git`? If so, the revision has to be
   stamped at deploy time, because the measurement's revision is otherwise `None`
   (`measure_scorer.py:192-200`).
2. **B-2 policy.** When no checkpoint is configured at all, should path-reasoning-only serving
   remain available, or require an explicit opt-in?
