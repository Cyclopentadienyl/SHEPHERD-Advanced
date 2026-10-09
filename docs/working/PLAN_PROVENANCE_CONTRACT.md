# PLAN — the provenance contract: every link in the pipeline checked where it is used

**Status: revision 2, amended twice after review. M1 is implemented, code-reviewed with no P1
or P2 at `ed4872a`, and accepted on the homelab GPU at `7714459`; the reviewer re-checked that
evidence with no P1 or P2 (2026-10-08; §4, "What M1 did" and "M1's acceptance on the homelab").
M2.1's work plan was reviewed and is revised with the owner's decisions of 2026-10-09; it awaits
re-review (§4, "M2.1 — the work, at `90a668f`"). M2–M5 are not implemented.** Facts about the code are cited at
`627ed08`, before M1; the code was unchanged at `463a0df`. §1 records decisions already made; §4
is the order of work.

**Revision 2 (2026-10-06)** follows the review of `463a0df`.
- **The resume parent is checked in M2**, before any state is restored (§4, M2.3). Left to M3a,
  a parent from the old pipeline, or from a same-shaped other graph, could be resumed; the child
  would record the current workspace and pass M2's check.
- **The digest recorded is the digest of the bytes parsed** (§2 rule 1; §4, M2.1), with two
  stated exceptions. Today each input is hashed by path at one moment and loaded by path at
  another, so an ordinary workspace rebuild in between makes a run that consumed A record B.
- **R10 is defined** (§5.4): its units, which evidence is used and how it is found, how resume
  history is covered, and an *unverifiable* state that is never shown as zero overlap.
- **Smaller corrections:**
  - training start's `checkpoint_dir` restriction moves back to item 14's phase 2, with the
    listing it belongs to (M5);
  - M3c names the fields of the served identity instead of filling two version strings;
  - M3b separates a dataset record's origin from its binding state;
  - capability checks stay, and `auto` stops falling back to CPU silently in training and
    serving (§5.2);
  - §5.3's reproducibility wording is narrowed;
  - §8 carries the reviewer's recommendations on the two open questions, and adds one.
- **Amended after the review of `b58cea3`:**
  - R10's case label compares phenotype *sets*, and no longer claims the model saw the same
    scoring input (§5.4). A repeated phenotype id changes Mode C's input, and the set ignores
    it;
  - M2.3's data-role rule is written as a product restriction awaiting the owner's decision,
    with what an operator will meet (§8, question 3). Superseded: the owner has since decided it
    (decision 7);
  - M2.1's two exceptions state what they establish and how they are displayed, and its buffer
    cost becomes a capacity measurement;
  - M3c moves every existing consumer of the old version fields in the same change.
- **Amended 2026-10-07, after the review of `15dfcb5`:**
  - the owner has decided §8 question 3: in the first version, resume continues the same data.
    It is decision 7 in §1, and M2.3 and §5.4 state it as decided;
  - whether a case may list a phenotype twice is taken up by its own plan,
    `PLAN_PHENOTYPE_NORMALISATION.md` (§5.4).

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
| 7 | **In the first version, resume continues the same data.** A resumed run's graph and data roles must equal its parent's (M2.3). Fine-tuning on other data is not supported yet. This is a product restriction, not a validity verdict, and it gates data, not settings | Owner, 2026-10-06 |

## 2. The rule

> The formal pipeline accepts an input only when it matches what the consuming step's governing
> record names. A mismatch, a missing required record, or a required check that cannot be
> completed is a refusal.

It is not necessary to show that a particular mismatch would make a particular score wrong.
Being unable to verify the pairing is reason enough.

1. **Identity is the SHA-256 of the bytes consumed.**
   - A path or file name locates an artifact and never identifies it. Location checks are still
     used where location is the point — an overwrite guard, a directory containment check — but
     never as identity.
   - A digest that a record states as an input, or that a check compares, is taken from the
     same buffer the step parsed. Hashing the path again at another moment identifies whatever
     the path holds by then, not what was read (§4, M2.1).
   - This applies to the inputs a step parses. A file a step does not parse may be identified,
     without being recorded, by one refusing check against the step's manifest reading (§4,
     M2.1 decision 1; the owner, 2026-10-09).
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
| R1 | Workspace graph files (`kg.json`, `node_features.pt`, `edge_indices.pt`, `num_nodes.json`) ↔ `split_manifest.json` | Manifest `artifacts` | `verify_graph_artifacts` (`src/kg/artifacts.py:58`) at `scripts/train_model.py:716` and `scripts/measure_scorer.py:128`; `verify_graph_source` (`artifacts.py:357`) at serving (`src/inference/pipeline.py:501`). **Refuses**, on digests taken by path at another moment than the load (`artifacts.py:82-86`) | Exists; on the bytes read from **M2** |
| R2 | Generated samples ↔ manifest | Manifest | `verify_generated_cohorts` (`src/evaluation/cohort.py:216`) at `train_model.py:717`, `measure_scorer.py:134`. **Refuses**; it hashes the file (`cohort.py:286`) and reads it again (`:294`) | Exists; on the bytes read from **M2** |
| R3 | **Model ↔ the graph it consumed**: `kg.json` (node index), `node_features`, `edge_indices`, `num_nodes` | `training_input_digests` (`train_model.py:551-574`; written by `src/training/callbacks.py:325-327`) holds the three tensors and `split_manifest`, **not `kg.json`** | **Structure only, and only a warning** (`src/utils/fingerprint.py:156-176`; `pipeline.py:909-918`). The digest comparison was "deliberately deferred" (`fingerprint.py:168-176`); only the probe performs it (`scripts/probe_deployment.py:812-836`). Measurement fits the weights by shape (`measure_scorer.py:593-594`) | **M2** |
| R4 | SP table ↔ the graph | Sidecar `kg_digest`, `build_id`, `max_hops` (`scripts/compute_shortest_paths.py:400-405, 481-489`) | At serving (`pipeline.py:724-747`): a `build_id` or `kg_digest` mismatch refuses; an **unrecorded** pair is served with a log warning (`docs/working/PLAN_SP_ARTIFACT_INTEGRITY.md:273`, "Unknown … serves") | **M2** (unrecorded refuses) |
| R5 | Workspace ↔ its dataset record (`kg.provenance.json`) | Written at build (`src/kg/provenance.py`); bound from the manifest | **Read by nothing in production**: `workspace_provenance_status` (`artifacts.py:208`) has no caller outside the module. Policy so far: "reported, not enforced" (`artifacts.py:123-137`; `docs/working/PLAN_ONTOLOGY_PROVENANCE.md:249-254`) | **M3b** (decision 2 reverses that policy for the formal pipeline) |
| R6 | Model ↔ the training settings, code and environment in effect | `config` = `TrainerConfig` plus `model_config` (`src/training/trainer.py:937-981`). Not recorded: the loader and sampling settings (`DataLoaderConfig`, `train_model.py:585-592`), compile (`train_model.py:760-777`), the code revision, the environment | — | **M3a** |
| R7 | Model ↔ its resume parent | The parent's digest (`train_model.py:558-559`), hashed by path after the parent was loaded (`:835`, `:854-860`) | **Nothing compared**: `load_checkpoint` restores state without looking at the parent's inputs or config (`trainer.py:1009-1044`). A missing parent only warns, and the run trains from scratch (`train_model.py:505-507`). The WebUI's default resume target is `last.pt` (`src/webui/components/training_console.py:398-404`), which `ModelCheckpoint` overwrites (`callbacks.py:271-274`) | **M2** (the parent's records, graph and data); M3a (overwrite guard) |
| R8 | A diagnosis result ↔ the model, graph and SP table that produced it | `InferenceResult.model_version` / `kg_version` exist (`src/core/types.py:448-449`) | Filled with a class version and `"unknown"` (`pipeline.py:1062, 1129-1130, 1703`). The API answers with the literal `"1.0.0"` (`src/api/routes/diagnose.py:351`) | **M3c** |
| R9 | Test result ↔ model, cohort and measurement settings | Measurement manifest (`measure_scorer.py:135-155, 342-390`); ledger (`src/evaluation/sidecar.py:291-319`) | R1 and R2 are checked; R3 is not (above) | M2 (R3), **M4** |
| R10 | Test cohort ↔ the data the model was trained and validated on | `cohort_kind=generated` claims disjointness by construction **relative to the workspace's cut** (`src/evaluation/measurement.py:527-530`), not relative to the model. `training_input_digests` names the model's `train_samples` and `val_samples` files by digest, and a digest cannot be intersected | — | **M4**, as defined in §5.4 |
| R11 | Environment facts that bound reproduction | Measurement: `torch_version`, `cuda_version`, the device string and the determinism flags (`measure_scorer.py:376-389`). No GPU model, compute capability, driver or cuDNN version. Training and serving record none tied to their artifacts | — | M3a (training), M4 (measurement) |

**What the table shows.** The links inside a workspace refuse today (R1, R2), and so does a
broken SP pair. Every link with the model on one side is checked by structure only, or recorded
and unread, or absent: R3, R6, R7, R8, R10. So is the dataset record (R5).

**R3's consequence, concretely.** A checkpoint trained on workspace A can be served or measured
on a workspace B whose tensors have the same shapes — for example, the same graph exported with
another feature seed or another torch version. The structural check passes, the strict weight
load passes, and the cached embeddings are computed from B. Under decision 1 that pairing is
invalid, and nothing stops it.

**One gap under every row: the digest and the load are separate reads.**
- **Training** verifies at `train_model.py:716-717`, loads at `:747-748`, and hashes the same
  paths again for its record at `:854-860`.
- **Serving** verifies at `pipeline.py:501` and loads at `:503`. The graph object was loaded
  earlier still, by the API (`src/api/main.py:493`).
- **Measurement** loads at `measure_scorer.py:571-572` and the checkpoint at `:593`. Only
  afterwards, when it builds each manifest (`:370`, reached from `:619-638`), does it verify and
  hash the same paths (`artifact_digests`, `:102-155`).

The code states this as an accepted limitation (`fingerprint.py:74-76`, `artifacts.py:82-86`).
It is reachable without any bad actor: a run reads `node_features` version A; the workspace is
rebuilt and B is published; the path is hashed and B is recorded. The model was trained on A,
its record names B, and a later comparison with B passes. The reviewer reproduced the record
half with the real `compute_input_digests` and an atomic replacement. The graph export and the
checkpoints are written in place by `torch.save` to the final path (`src/kg/graph.py:645-646`,
`callbacks.py:329`), so not even an open handle keeps the bytes it was opened on.

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

#### What M1 did (2026-10-07; code-reviewed 2026-10-08)

**Review.** Three rounds, all at no P1:
- the independent review of `fad6be7` found three P2s, closed at `41dc927`;
- the reviewer's review of `41dc927` found one P2: a successful reload did not restore
  readiness. It was closed at `52f0805`;
- the reviewer's incremental review of `ed4872a` found no P1 or P2. The reviewer recommends
  keeping the shared `requested_pipeline_missing` definition rather than the one-line fix, and
  asks that a pull request state the readiness change below.

`/ready`'s observable behaviour changes in two states: a configured graph file missing at
startup, and a refused reload on a service that had no pipeline. `/ready` now answers 503 in
both, where it answered 200. `/diagnose` already refused in both, so the old 200 was a false
"ready". The way back is a reload that succeeds, or a restart. `/health` is unchanged, and the
launcher and the WebUI read only `/health`.

The reviewer ran the synthetic CPU cases on a pinned snapshot. Afterwards the service, run from
`7714459` on the homelab GPU, was accepted ("M1's acceptance on the homelab", below).


**A requested model is built, or the build raises `PipelineBuildError`**
(`src/inference/pipeline.py`). A model is requested when a checkpoint path or a pre-loaded model
is given. These conditions used to log and return without the GNN, and the pipeline was
published as a success:
- PyTorch is not available;
- there is no graph to compute embeddings from, whether no `data_dir` or `graph_data` was given
  or the directory lacks its tensors;
- the checkpoint is missing, or does not build a model over this graph
  (`_load_model_from_checkpoint`; the builder's error is kept as the cause);
- shortest paths are required (`sp_optional=False`) and not available. This used to switch the
  GNN off, so a configuration requiring both signals was served with neither.

Three more cases now raise it, naming what is wrong:
- an unreadable checkpoint. It already raised, as torch's own error;
- a file that is not a training checkpoint;
- a checkpoint with no weights.

One invariant is added: `_gnn_ready` is set only with a model and with phenotype and disease
embeddings, the two `_calculate_gnn_score` reads. It cannot be reached today, because every
loader raises. It stops a loader that one day returns `None` again from leaving a pipeline that
reports a GNN and scores every GNN term as zero. It does not show that the embedding rows agree
with the graph's node identifiers; that is M2, and M1 passing is no claim about it.

**The API's build-then-publish order put the refusal in the right place.** `build_pipeline`
already re-raised, startup already published nothing on failure, and the reload route already
built the candidate before publishing it. Four gaps beside it are closed (`src/api/main.py`,
`src/api/routes/diagnose.py`). The first three were found by the independent review, the fourth
by the reviewer's review of `41dc927`:
- **a checkpoint configured with no knowledge graph is refused.** Before, `build_pipeline`
  returned `None` before recording the request, and `/diagnose` gave the demo answer: invented
  candidates over HTTP 200. Startup now also attempts the build when only
  `SHEPHERD_CHECKPOINT_PATH` is set, so the refusal is seen there;
- **`/diagnose` no longer rebuilds a failed pipeline on every request.** Before, the lazy
  initialisation reran the whole build each time: graph load, workspace digests and tensor
  reads. It ran synchronously in an async route, so `/health` and the WebUI waited, and each
  request got the same 503. M1 would have sent the commonest misconfigurations there. Trying
  again is now an explicit act, a reload or a restart, and the 503 says so. It is a choice, not a
  necessity: the files or mount behind a failed build can be repaired while the service runs,
  and a reload is how to try again without expensive I/O on every diagnosis;
- **a blank environment value is unset.** An exported-but-empty `SHEPHERD_CHECKPOINT_PATH` would
  otherwise be read as a request for a model;
- **a reload that succeeds restores readiness.** A startup whose build fails sets `is_ready` to
  false, and nothing set it back. After the reload the 503 points to, the pipeline served and
  `/pipeline/status` agreed, while `/ready` stayed 503 until a restart, so anything routing on
  the probe kept the service out. The cause predates M1; M1 made it the outcome of the commonest
  misconfigurations and named the reload as the way back.
  - `publish_pipeline`, the one writer of the served-pipeline fields, now also sets the
    readiness flag. It is the one path that restores the flag after startup.
  - So after a failed startup, readiness returns only once a candidate is published, never when
    a reload starts. A refused reload leaves the flag as it was.
  - `/ready` and `/diagnose` also read one definition of the state `/diagnose` refuses: a real
    pipeline was asked for and none is being served (`AppState.requested_pipeline_missing`).
    Two states left the flag set while every diagnosis was refused, and `/ready` answered 200
    in both: a configured graph file missing at startup (`build_pipeline` returns `None` there
    rather than raising), and a refused reload on a service that had no pipeline.

A reload refusal no longer ends in a doubled full stop.

**Unchanged, deliberately:**
- with no model source at all, the pipeline still serves path reasoning. Whether that needs an
  opt-in is B-2's policy question (§8, question 2);
- absent shortest paths with `sp_optional=True`, the default, still give a GNN-only pipeline
  (`DISEASE_SCORER_POLICY.md` §2).

**Tests:**
- `tests/unit/test_pipeline_fails_closed.py`, 31 tests, covers:
  - each condition above, plus the invariant;
  - controls: a sound checkpoint builds, a GNN with optional SP absent builds, and no model
    source still builds path reasoning;
  - startup with a missing, another graph's, and an unreadable checkpoint: nothing published,
    `/diagnose` 503, with the same startup and a sound checkpoint as control. The first two were
    the old fallback;
  - one build for a failed startup followed by three diagnoses;
  - readiness, through the real lifespan and routes, with `/ready`, `/pipeline/status` and
    `/diagnose` compared each time:
    - a failed startup: all three say not serving;
    - a refused reload after it: still not serving;
    - a successful reload after it: all three say serving, and readiness was false while the
      candidate was built;
    - a refused reload on a healthy service: the pipeline, its paths and readiness are the same
      objects as before, and it still serves;
    - the whole chain in one lifetime: failed startup, refused reload, successful reload, and a
      refusal after it that changes nothing;
    - a configured graph file missing at startup: not serving;
    - nothing configured: the demo answers and is ready, until a reload naming a workspace is
      refused;
  - a checkpoint with no knowledge graph, and a blank checkpoint setting;
  - reloads to an unreadable checkpoint and to one trained over another graph: refused, with
    every served field the same object as before.
- The graph-binding tests' stand-in now finishes the model build. Before, it ended in the
  fallback this change removes.
- `tests/unit/conftest.py` puts `app_state.real_pipeline_requested` and `is_ready` back after
  each test.
  - Three files left `real_pipeline_requested` set, so `test_diagnose_reserved_fields.py`
    failed whenever it ran after them. This was already true at `4298aac`; the alphabetical
    full run hid it.
  - `is_ready` could not leak before this change, because only the lifespan wrote it and its
    shutdown clears it. `publish_pipeline` now sets it, so a reload driven without a lifespan
    would leak it. No current test does; the restore is defensive.

**Mutation check.** 19 mutants were run in a fresh copy of the tree, each restoring one old
behaviour or removing one new check. Four are on readiness: publication not restoring it, a
reload setting it when it starts, a refused reload clearing it, and `/ready` not reading the
shared definition. All 19 were caught.

**Found, not changed:**
- **One for M2.** `_calculate_gnn_score` clamps a disease index past the end of the embedding
  table to the last row (`disease_idx = min(...)`). A graph whose node mapping disagrees with its
  tensors is therefore scored against another disease's embedding, not refused.
  - On the API path, the workspace binding prevents that mismatch.
  - A direct caller's graph object is not checked against its `kg_path`, and a caller supplying
    `graph_data` in memory has no binding at all.
  - M2's model↔graph check is where the refusal belongs.
- **A configured knowledge graph file that is missing makes `build_pipeline` return `None`,**
  rather than raise. Both callers still fail closed: startup leaves `/diagnose` and `/ready` at
  503, and a reload is refused.

#### M1's acceptance on the homelab (2026-10-08)

Run by the owner at `7714459`, a detached checkout. The evidence was collected by the author
from the owner's uploads and pasted output.

**Environment.**
- NVIDIA GB10, as for N1. PyTorch warns that the GPU's CUDA capability (12.1) is past the 12.0
  it supports.
- **The owner's command, as pasted:** the usual launcher (`./launch_shepherd.sh`) on port 8000,
  with `SHEPHERD_KG_PATH`, `SHEPHERD_DATA_DIR`, `SHEPHERD_DEVICE=cuda` and a
  `SHEPHERD_CHECKPOINT_PATH` naming a file that does not exist. Its output was captured with
  `tee` to `server_m1.log`.
- **Workspace:** the same path as N1's, `data/workspaces/hpo_2026_0929_5a`. The startup log read
  57,239 nodes and 617,773 edges.
  - The owner hashed two files after the run, and both SHA-256s match N1's record:
    - `checkpoints/hgt/model-02-0.1813.pt` `33a7b39a58519ff4974ed61266a5a35204edbb253d23c845340d0dcdf5eb8b79`;
    - `kg.json` `6cae2d1a58690eec9aa9c5e3ca9182c2fc942db0f468a5123983251bd4c51e43`.
  - The tensors, the manifest, the provenance record and the SP files were not hashed again.
  - As with N1, a file hashed afterwards is not proof of the bytes consumed during the run.
    That proof is the contract's M2.1.
- **A first attempt ran the wrong code and was discarded.** The owner's `git switch` landed on a
  stale local branch at `60b3c89`, from August, so the service ran old code. The log the owner
  pasted shows the pre-M1 behaviour: `Checkpoint not found`, then `Pipeline initialized:
  scoring_mode=path_reasoning_fallback`. That attempt's full log is not in the archive.
  - Before the run below, the homelab's 16 local branches other than `main` were archived to a
    bundle outside the repository. `git bundle verify` reported all 17 refs and a complete
    history, and the bundle's SHA-256 is
    `2106db34355b6811f76e441ee970e2489019b9c0e43b5ad65e57268cc0acc179`. The 16 branches were then
    removed.
  - Two of them held commits found nowhere on GitHub: `60b3c89`, a lockfile refresh, and
    `61978b6`, an August backup. In the homelab clone, they were the only branches with commits
    outside every remote branch and tag. In a clone with every GitHub branch and tag fetched,
    neither commit exists.

**Startup.** The graph loaded in about 10 s, from 16:09:43 to 16:09:53. The log then shows
`Failed to build pipeline: Checkpoint … does-not-exist.pt does not exist` and `Startup failed`.
No pipeline was published, and the fallback did not appear.

**Through the API.** The acceptance client, `m1_acceptance.py`, uses the standard library and
calls only the running API. Its SHA-256 as delivered to the owner is
`a82d42853d15bde0e3bb8d6ae16e817d0a33115142242ae74bb3d759d5d2f4e1`; it is not in the repository.
It ran 12 checks, and all passed:
1. **the failed startup:** `/ready` answered 503 with the reload guidance, `/diagnose` answered
   503, and the status said not initialized;
2. **a reload to an unreadable checkpoint** was refused in 10.2 s. A 16-byte file, not a pickle,
   stood in for it. All three surfaces still said not serving;
3. **a reload to the good checkpoint** succeeded in 79.5 s:
   - **status:** `gnn_plus_shortest_path`, with GNN and SP ready. Eta was 0.7, and SP `max_hops`
     was 5, from the sidecar. The SP table's graph binding was *verified*, and there were no
     fingerprint warnings;
   - **checkpoint metadata in the status:** epoch 2, val MRR 0.1813, 4,228,429 parameters
     counted from the built model, and device cuda;
   - `/ready` answered 200 with a pipeline loaded;
   - `/diagnose` answered 200 with 10 candidates for `HP:0001250`, `HP:0001263`. Every GNN score
     was non-zero, and every total was 0.7 × GNN + 0.3 × SP. The top candidate was
     `MONDO:0014942`, at 0.83057 = 0.7 × 0.97224 + 0.3 × 0.5;
4. **a reload to the unreadable checkpoint while serving** was refused in 10.0 s, with "still
   being served". The status, `/ready` and all 10 candidate records were identical to step 3,
   field for field; only the session identifier, timestamp and timing differed.

**The server log agrees.** The owner ran `grep -nE "Startup failed|Failed to build
pipeline|Pipeline reload failed|Pipeline published|Lazy pipeline init" server_m1.log`, and the
extract is `m1_log_grep.txt`. It shows:
- one failed build at startup, ending at 16:09:54;
- refused reloads at 16:12:38 and 16:14:08;
- one publication, at 16:13:58, `gnn_plus_shortest_path`.

The extract has no other `Failed to build pipeline` line and no `Lazy pipeline init` line. A
rebuild triggered by a diagnosis would have logged one of them with this missing checkpoint, so
the diagnoses in steps 1 and 2 did not rebuild the failed pipeline.

**Evidence.**
- **Uploaded,** with SHA-256 as uploaded:
  - `summary.json` `1992dac75e6e0000c0d096d5e579b27bf88376a6087096a4afe751102b777bdf`;
  - `3_after_good_reload.json` `9472ef15447fe27356c6d7288c906eca770ca083df4d99dc6977698cc909603b`;
  - `4_after_refused_reload_on_healthy.json`
    `7a09701e7eccf783551cdd58d5c285dd3b8bc95c967f6a26c13363d4d70981b7`;
  - `m1_sha256.txt` `8f76b872a8b92033977b4d15ff8cee2b22cbf1bc2bc7a910afcb01015e40bc1b`;
  - `m1_log_grep.txt` `e153059ee6f38e498c1995b94cf742b5f87d7fc483be048a3c117765f6b58d4f`.
- **Written by the client and not uploaded:** the snapshots of steps 1 and 2, the three reload
  responses, and the full `server_m1.log`. Steps 1 and 2 rest on `summary.json` and the
  terminal output the owner pasted.
- **Where they are:** after the run, the owner moved all of them to
  `~/Desktop/SHEPHERD-archive/m1-acceptance-2026-10-08/` on the homelab, by the pasted command.
  That covers the client's output directory, the full `server_m1.log`, both text files and the
  client script.
  - The branch bundle is in `~/Desktop/SHEPHERD-archive/`.
  - The homelab checkout is back on `main`, level with `origin/main`.

**Before the homelab run,** the author ran the client in the development container, on CPU, with
a synthetic workspace of 9 nodes and 14 edges. The services were built from two commits:
- **`7714459`:** all 12 checks passed;
- **`41dc927`:** exactly three failed. 3c and 4c are the reviewer's P2: a successful reload did
  not restore readiness. 1a failed because `/ready`'s 503 had no recovery guidance until
  `ed4872a`;
- **a service already serving:** the client stopped at its precondition.

**What the reviewer checked independently** (2026-10-08, at document head `57c8684`; no P1 or
P2). The reviewer read the archive on the homelab directly, not only the record.
- **Hashes:** every file the record hashes was re-hashed and matched, the client script
  included. The full `server_m1.log`, which the record did not hash, is
  `c59567f8a3d27f084f480360585a88db306bf1087db53d52b5d13f133201ebb7`.
- **Responses:** steps 1–4 were compared as whole JSON documents, beyond the fields the client
  checks. Before and after the refused reload on the healthy service, the status, the `/ready`
  body and all 10 candidates were identical. Only the diagnosis's session identifier, timestamp
  and timing changed. Every total was exactly 0.7 × GNN + 0.3 × SP.
- **The full log:**
  - `Building diagnosis pipeline` appears 4 times: once at startup and once per reload;
  - `Startup failed` once, `Pipeline reload failed` twice, `Pipeline published` once;
  - `Lazy pipeline init` and `path_reasoning_fallback` never appear;
  - regenerated with the recorded pattern, the extract is byte-identical to `m1_log_grep.txt`.
- **Version:**
  - `ed4872a..57c8684` changes no file under `src`, `scripts` or `tests`, so the code under test
    is the code reviewed;
  - the homelab's reflog puts HEAD at `7714459` from 16:09:39 to 16:42:15, which covers the run.
    This is the operator's checkout, not a version reported by the service process;
  - the bundle verifies, with 17 refs and a complete history, and the homelab has only `main`,
    at `4298aac`.
- **Not re-checked:**
  - the discarded attempt's behaviour, which rests on the owner's paste;
  - an audit of every GitHub ref for the two local-only commits.
- **Read only:** nothing was re-run against the service, and nothing was changed.

**What this holds for, and what it does not.**
- **Failures exercised:** two — a missing checkpoint at startup, and an unreadable file on
  reload. The other refusals in "What M1 did" were tested only on CPU, in the unit tests:
  - a checkpoint trained over another graph;
  - one with no weights;
  - a file that is not a training checkpoint;
  - a missing graph file;
  - a checkpoint configured with no graph;
  - a blank setting.
- **Setup:** it holds for this GB10, this model, this workspace path and this SP table, with
  these settings.
- **WebUI:** the reloads went through the API route the WebUI's Load / Reload button calls. The
  WebUI itself was not exercised in this run.
- **Not covered:**
  - other GPUs or operating systems;
  - any of the contract's checks. Those begin at M2, and M1 passing is no claim that the model
    matches its graph.

### M2 — the model↔graph relation, on the bytes actually read, with the resume parent

This is the first enforced check. It lands with M1 or after it, never before. Its parts land in
the order below. M2.1 comes first because a comparison of digests that were not taken from the
bytes read says nothing about what was read.

#### M2.1 — one read per input

**Change:**
- **One primitive**, beside `file_sha256` in `src/utils/fingerprint.py`: read a file's bytes
  once and return them with their SHA-256.
- **Every reader whose input a record names parses that buffer, and returns what it parsed
  together with the digest.** These are:
  - `read_graph_artifacts` and `read_samples` (`src/kg/storage/file_storage.py`);
  - the manifest read;
  - `KnowledgeGraph.load_json` (`src/kg/graph.py:815`), at serving and in the SP producer.
    The producer hashes `kg.json` by path (`scripts/compute_shortest_paths.py:458`) and loads
    it again (`:461`), so a replacement in between computes the table from one graph and
    records another. Its end-of-run comparison (`:472-479`) stays as a warning: the record
    then names the bytes that were traversed;
  - every checkpoint load: training's resume parent, serving, measurement.
- **The verifiers compare those digests and stop hashing paths:** `verify_graph_artifacts`,
  `verify_generated_cohorts` and `verify_graph_source`.
  - The manifest is read once per run, and every file's digest is compared with that one
    reading.
  - A file replaced between two reads therefore shows up as a mismatch, and the run refuses
    instead of mixing.
  - At serving, the API passes the digest of the bytes the graph object was built from, where
    it now passes a path to be hashed again.
- **Records are assembled from the same digests:** `training_input_digests` and the measurement
  manifest's inputs.
  - `compute_input_digests` stops being used to record a consumed input.
  - `file_sha256` stays only for identifying a file nobody is about to parse: the search in
    `find_checkpoint` (`src/evaluation/sidecar.py:366-383`) and, in the M2.6 inventory, the
    files it does not parse.
  - **Amended 2026-10-09** (§4, "M2.1 — the work"):
    - an inventory that reads a checkpoint's `training_input_digests` parses that checkpoint, so
      its digest comes from that read;
    - by the owner's decision, the allowance also covers one refusing check: `kg.json`, before a
      run that does not parse it, compared with the run's one manifest reading.
- **The copies collapse onto the shared reader.** This is not a separate project: the binding
  needs one reader per kind of file.
  - The callers moved: training's `load_graph_data` and `load_samples`
    (`train_model.py:398-468`), and serving's `_load_graph_data` (`pipeline.py:549`). This is
    `file_storage`'s own migration (`file_storage.py:9-16`), for these callers.
  - `read_samples` gains the two optional fields training reads, `candidate_disease_ids` and
    `gene_ids`.
- **The buffer, not the handle.** Hashing an open handle and then parsing from it survives an
  atomic rename, but not an in-place rewrite, and `torch.save` rewrites in place.
- **Memory.** Files are parsed one at a time. After each parse only the parsed result and the
  digest are kept, and the raw buffer is released. Wrappers such as `BytesIO` may add copies of
  their own, so "one transient copy per file" is an expectation to measure, not a bound (see
  the acceptance below).
- **Three stated exceptions** (the third decided by the owner, 2026-10-09).
  - **The SP table at serving.** Its identity is the `build_id` inside the tensor it loads,
    checked against the sidecar (`pipeline.py:727`). So the identity checked already comes from
    the bytes loaded, and the table gets no file digest. Copying a multi-gigabyte table into
    memory to hash it would guard against a threat nobody observed.
    - **What it establishes:** the tensor and the sidecar are one publication, and the sidecar
      names this graph. It is not a checksum of the tensor's bytes, it does not detect bit
      corruption, and it does not show the distances are correct (`sp_artifact.py:18-24`).
    - **How it is displayed:** as the `build_id` and the binding state, never as "SP bytes
      verified".
    - **What the implementation keeps:** the staged publication (`sp_artifact.py:392-402`), the
      single sidecar read (`pipeline.py:664`), and the comparison with the token inside the
      loaded data (`pipeline.py:724-729`).
  - **Training's `kg` role** (M2.2). Training does not parse `kg.json`. The role is the
    manifest's binding for the tensors it did parse, and that manifest is itself read once. It
    is displayed as the manifest-bound source graph, never as a file training loaded. Serving,
    which does parse `kg.json`, checks its own read.
  - **Measurement's `kg` role**, decided by the owner on 2026-10-09. Measurement does not parse
    `kg.json` either; every mode computes from the tensors. Its role is the manifest-bound source
    graph, as training's, never a file measurement read.

**Precedent.** The KG build already does this for the ontologies: it records the digest the
loader took from the handle it parsed, because hashing the path afterwards recorded a
replacement (`scripts/build_knowledge_graph.py:330-336`).

**Acceptance:**
- **A test double replaces an input between its read and everything after it** (the check, the
  record), once by atomic rename and once by in-place rewrite. It runs at each entry point, on
  the inputs that entry point reads:
  - training: a graph tensor, a sample file and the resume parent;
  - measurement: a graph tensor, the cohort's sample file and the checkpoint;
  - serving: `kg.json` as the API loads it (`src/api/main.py:493`), a graph tensor and the
    checkpoint;
  - the SP producer: `kg.json`.

  The only outcomes allowed are that the run uses *and* records the bytes it read, or that it
  refuses. It never reads A and records B.
- **A replacement between the manifest read and a file read is refused**, and the message names
  the file.
- **The record maps are built only from the readers' results.** A test pins this, so that
  bringing back a path-hashing call at a producer fails it.
- **Capacity.** Peak memory is measured at each entry point on the homelab's largest workspace
  and checkpoint (measurement: Mode C, §4 S10). A shortfall is fixed in the reader. A CPU
  fallback is not a remedy.

#### M2.1 — the work, at `90a668f` (2026-10-08; revised 2026-10-09 after review)

This is how M2.1 would be built. It was mapped against the code at `90a668f`, the merge of M1,
by independent readers of each entry point and of the shared readers. A completeness critic then
checked the map and this breakdown, and a second check verified this text against the code.
**Citations in this subsection are current at `90a668f`**; those above, taken at `627ed08`, are
left as written. Nothing here is implemented.

**Revision of 2026-10-09.** It follows the reviewer's plan review of `4ecd2c2` (one P2, no P1)
and the owner's decisions of the same day, recorded under "Decisions" below.
- **The P2:** S10 asked for capacity readings of measurement Modes A and B, which cannot run on
  the designated checkpoint. S10 is now an executable matrix.
- **The calibration launcher:** checked as the owner asked, and found to have no necessary user.
  It is removed in a new step, S0.
- **For the owner:** open question 6 (`scripts/evaluate_model.py` after S0), and whether S10's
  subject is the homelab's largest workspace and checkpoint.
- **Approval of this plan is not approval of any implementation.**

**What the code does today.** None of M2.1's four entry points hashes and parses a recorded input
from one read. The pattern does exist elsewhere:
- the KG build records the digest the ontology loader took from the handle it parsed
  (`src/ontology/loader.py:242-264`), the precedent above;
- the SP-reachability audit hashes the sidecar bytes it parsed (`audit_sp_reachability.py:544`).

At the entry points:
- **Training reads each input it consumes three or four times.** `kg.json`, which it only
  verifies, is read once, at `artifacts.py:113`.
  - **The checks:** `train_model.py:716-717`. They go to:
    - `artifacts.py:100`, the manifest, and `:113`, each graph file, hashed;
    - `cohort.py:275`, the manifest again, `:286`, each sample file, hashed, and `:294`, each
      sample file, parsed through `file_storage.py:100`.
  - **The writes:** the run directories, `config.yaml` and `runtime.json` (`:731-744`).
  - **The parse training uses:** `train_model.py:418`, `:424`, `:430` and `:455`.
  - **The record hash:** taken last, after the model is built and any parent is restored
    (`:854-860`, through `fingerprint.py:103`).

  The manifest is parsed separately by each verifier and hashed a third time for the record, so
  the record describes neither parse.
- **Serving** parses `kg.json` at `src/api/main.py:545`, then hashes it by path (`artifacts.py:113`,
  `:386`). It reads the tensors again (`src/inference/pipeline.py:609-655`) and loads the
  checkpoint at `:973`.
- **Measurement** verifies and hashes once per mode, after it has parsed (`measure_scorer.py:370`,
  through `:102-155`).
- **The SP producer** hashes `kg.json` at `compute_shortest_paths.py:458` and reads it a second
  time, to parse it, at `:461`.

**The primitive.**
- `read_once(path) -> FileRead(path, data, sha256)` goes in `src/utils/fingerprint.py`, beside
  `file_sha256` (`:55-82`). `ReadIdentity(path, sha256)` carries no bytes.
- **One open and one full read,** with the SHA-256 taken over that same bytes object.
- **It never returns `None`,** as `file_sha256` does (`:79-80`). A missing path raises, so a file
  deleted between two reads is refused instead of recorded as `None`.
- **No retry, no fallback, no device handling.**
- **Layers** (`.import-linter.ini`): it lives in `src.utils` and imports only the standard
  library. `src.kg`, `src.evaluation`, `src.inference`, `src.training` and the scripts may all
  import it.

**The readers.** Each returns what it parsed and a `ReadIdentity`, never the bytes:
- `read_graph_artifacts(data_dir, *, map_location="cpu") -> GraphArtifactsRead(graph_data,
  reads)` and `read_samples(data_dir, split, *, training_fields=False) -> SamplesRead(samples,
  identity)`, in `src/kg/storage/file_storage.py`;
- `read_split_manifest(data_dir) -> ManifestRead(manifest, identity)`, in `src/kg/artifacts.py`.
  It keeps today's refusals of an absent or old-schema manifest;
- `KnowledgeGraph.read_json(path) -> GraphRead(kg, identity)`, in `src/kg/graph.py`.
  `load_json` delegates to it, for callers that record nothing;
- `read_checkpoint(path, *, map_location, weights_only) -> CheckpointRead(checkpoint,
  identity)`, in a new `src/utils/checkpoint_io.py` with a lazy torch import. It does not go in
  `checkpoint_paths.py`, which promises no torch (`checkpoint_paths.py:14`).

How they parse:
- **JSON:** decoded as UTF-8 and parsed with `json.loads`.
  - `kg.json`, the sample files and the manifest are written as UTF-8 (`graph.py:806`,
    `sample_generator.py:334`, `:343`).
  - `num_nodes.json` is written in the locale default (`graph.py:647-648`), but `json.dump`'s
    default `ensure_ascii` makes it ASCII, which decodes the same way.
  - Today's `read_text()` uses the locale default (`file_storage.py:56`, `:100`).
- **Torch files:** `torch.load(BytesIO(data), ...)`.
  - **Graph tensors** are read with `weights_only=True` and `map_location="cpu"` everywhere.
    Serving passes `"cpu"` today (`src/inference/pipeline.py:621-623`, `:634-636`); training and
    measurement pass none (`train_model.py:418`, `:424`; `file_storage.py:50`, `:53`), which is
    equivalent for CPU-saved exports.
  - **Checkpoints** keep each site's current `map_location` and `weights_only`.
- **Memory:** one file at a time, with the buffer released before the next read. Each file costs
  about one transient copy of its size while it is parsed:
  - about 29 MB of tensor payload for `node_features.pt`, computed from the recorded shapes; the
    file's size is not recorded;
  - 48 MB for the designated checkpoint;
  - a JSON file's bytes and its decoded text coexist, so `kg.json` (size unrecorded) is likely
    the largest new transient, at serving and in the SP producer.

  These are expectations to explain the readings with (S10), not bounds.

**Two constraints the plan above does not mention.**
- **`read_samples`' new fields are opt-in** (`training_fields=True`). The frozen evaluator builds
  samples with three fields (`scripts/evaluate_model.py:210-216`), and `gene_ids` become subgraph
  seeds (`src/kg/data_loader.py:675-676`, `:947-950`). Adding them by default would change
  measurement's Modes A and B.
- **The test fixtures must become parseable.** `tests/fixtures/generated_workspace.py:102` writes
  placeholder bytes for the graph files. Today's path-hashing verifiers accept them; a parsing
  reader would reject them.

**The graph object and its identity travel together.** At serving, `DiagnosisPipeline` takes a
`GraphRead` (the graph and the identity of the bytes it was parsed from) whenever it uses a
workspace. It does not take the graph and an identity as two separate arguments.
- With two arguments, a caller could pair an in-memory graph with another file's identity, and
  the SP binding would then be reported against bytes the graph was never parsed from
  (`src/inference/pipeline.py:532`, `:788-790`).
- A caller with an in-memory graph uses the `graph_data` seam, which makes no workspace claim.

**Boundaries the review set.** These hold for every step:
- **One buffer, one parse.** Hashing and parsing the same buffer makes the digest and the parse
  agree. It is not an operating-system snapshot of several files. Consistency across files comes
  from comparing each role with the one manifest reading, and no snapshot is claimed.
- **Replacing a path already read is not a refusal.** If every input the run uses has been read
  as A, replacing a path afterwards leaves the run using and recording A. Nothing is read again
  to force a refusal.
- **Training's extra fields are passed explicitly.** Training calls `read_samples(...,
  training_fields=True)`, and its tests use cases that carry both `candidate_disease_ids` and
  `gene_ids`, not only the flag. This is the existing difference between what training and
  measurement consume, not a fallback.
- **No retired evaluator is restored, no checkpoint field is added, and no fallback is added**
  to complete M2.1.

**The steps.** Each is one reviewable change. They are ordered smallest and safest first, and
each moves its callers before anything is removed.

0. **S0 — remove the retired calibration launcher.** The owner's rule: a module nothing
   necessary uses is removed, not migrated, so no fully retired module stays behind as a dead
   island (with question 6 (a); under (b), `scripts/evaluate_model.py` would be one). The check,
   at `4ecd2c2`:
   - **Who depends on `scripts/calibrate_mode_a.py`.**
     - There is no CI. No Makefile target, launcher, deploy script or systemd unit names it.
     - No module under `src/` and no other script imports or runs it.
     - No runbook or deployment guide instructs running it. The documents that name it are the
       draft `PLAN_TEST_RESULTS.md:745` and `:766` (a planned flow, dropped below) and
       `scorer-measurement/README.md:33`, `:54` and `:103-105` (marked SUPERSEDED, or the
       decision amended below).
     - Only its own tests import it:
       - `tests/integration/test_seeding_bootstrap.py`;
       - in `tests/integration/test_legacy_equivalence.py`: the launcher fixture (`:41-73`); the
         tests that read it, including the verdict-schema and `compare()` digest tests, which
         import it at `:223`, `:243` and `:263` (`:79-272`); and the split-default test
         (`:431-449`, import at `:440`);
       - a seed-check class in `tests/unit/test_measurement_mode_a.py:890-939`;
       - one entry in `tests/unit/test_split_caveat.py:39`.
   - **It cannot run on any real checkpoint.**
     - Its oracle, `scripts/evaluate_model.py`, builds a structurally wrong model, and its harness
       call reaches `build_legacy_mode_a_model`, which indexes keys no current checkpoint writer
       produces (`measure_scorer.py:251-252`; BACKLOG §3.1).
     - Its tests pass only because the fixture writes those keys
       (`tests/fixtures/synthetic_workspace.py:159-160`).
   - **What it does is either covered or unwanted:**
     - its comparison is replaced by `compare_trainer_against_mode_a`
       (`src/evaluation/differential.py:373-546`);
     - its subprocess seeding exists only for the frozen evaluator;
     - its `auto` device falls back to CPU with only a warning, recording `cuda_executed=false`
       (`calibrate_mode_a.py:164-172`), against the no-fallback rule; `measure_scorer.py`'s
       `auto` refuses without CUDA;
     - its before/after digest bracket exists because two processes read the files separately,
       which one read per input removes.
   - **Removing it takes away only the fixture-level parity with the frozen evaluator,** which
     BACKLOG §3.1.2 retired as an acceptance. Mode A stays pinned by its unit tests, and the
     adopted acceptance, `tests/unit/test_differential_calibration.py`, is unchanged.

   **What S0 changes:**
   - **Deleted:** `scripts/calibrate_mode_a.py` and `tests/integration/test_seeding_bootstrap.py`.
   - **Tests:**
     - in `test_legacy_equivalence.py`, the launcher fixture (`:41-73`) and the tests at
       `:79-272` go. Their measurement-only assertions move onto a direct `measure_scorer` run
       with `--num-workers 4` and `--predictions-output`, which keeps those two paths covered end
       to end. The split-default test (`:431-449`) keeps only its `measure_scorer` half;
     - `test_measurement_mode_a.py:890-939` goes;
     - `test_split_caveat.py` keeps only `measure_scorer` as an entry point.
   - **Comments and docs that name the launcher in the present tense:**
     - `measure_scorer.py:29-33`, `:96-98`, `:659-660`; `src/evaluation/measurement.py:255-258`;
     - `test_legacy_equivalence.py`'s module docstring (`:1-25`) and `:284-285`;
       `tests/unit/test_split_caveat.py:4-5`, `:17`; `tests/unit/test_frozen_evaluator.py:13`, if
       question 6 keeps that file;
     - `scorer-measurement/README.md:33` and `:53-55`;
     - `PLAN_TEST_RESULTS.md:745` drops it, and the staging design at `:766-775` goes: it existed
       so the launcher and `test_legacy_equivalence.py` could drive the oracle on supplied
       cohorts. The line citations into `test_legacy_equivalence.py` at `:749-751` are updated.
       The oracle's named exception (`:763-765`) follows question 6.

     Left as history: `PLAN_B03.md:23` and `:103`, `PLAN_CONFIGURABILITY_AND_PROVENANCE.md:235`,
     and the past-tense notes in `src/evaluation/caveats.py:5-6` and
     `src/utils/fingerprint.py:64-68`.
   - **Decisions amended when S0 lands** (this revision amends none of them), each with the
     owner's rule and date:
     - `scorer-measurement/README.md:103-105` ("rewritten, not deleted") is reversed;
     - **the deletion gate gains one exception,** in BACKLOG §5 ("Item 9 waits for everything"),
       in `scorer-measurement/README.md:78-79` and its removal order's step 5 (`:122`), and in
       item 9's row. The exception: the oracle-parity tests in `test_legacy_equivalence.py`,
       which are oracle-only (`README.md:95-97`), go with the launcher before 1d's institutional
       run. They check the frozen-evaluator parity that §3.1.2 retired as an acceptance, the
       launcher could not run them on any real checkpoint, and the adopted acceptance, the
       differential calibration, is untouched;
     - BACKLOG §5.0's file table drops the launcher.
   - **Item 7a's entry point is recorded as a backlog item.** Its runner is new code built on
     `compare_trainer_against_mode_a`. It records, beside its result, the digests of the bytes it
     read (through M2.1's readers), the software revision and the deployment host. History stays
     in git.
   - **`scripts/evaluate_model.py`** follows question 6 below.
1. **S1 — the primitive.** `read_once`, `FileRead` and `ReadIdentity`; no caller changes.
   - **Tests:** the digest is the digest of the bytes returned; the file is opened once; a
     missing file raises; replacing the file afterwards, by atomic rename or in-place rewrite,
     changes neither the bytes nor the digest returned.
2. **S2 — test infrastructure.** No production change.
   - **Parseable fixtures:** each workspace's `kg.json`, `num_nodes.json` and tensors are real
     files whose content differs per workspace. Tests that verify only digests gain
     `importorskip("torch")`.
   - **Training samples** that carry both `candidate_disease_ids` and `gene_ids`.
   - **The replacement double** (`tests/fixtures/replacement.py`): it wraps `read_once` and
     replaces a target file once, right after a named read returns. It does so by atomic rename
     (a new inode) or by in-place rewrite (the same inode). It can also republish a manifest that
     binds the replacement.
   - **An open counter per path,** for the "read once" assertions.
3. **S3 — `read_json` and the SP producer.** `compute_shortest_paths.py` records the digest of
   the `kg.json` bytes it traversed, from its one read. The end-of-run comparison (`:472-479`)
   stays a warning, as the plan above says.
4. **S4 — the shared readers.** The other four readers above (`read_json` is S3's). Callers only
   unpack the new return values; no recorded digest changes yet.
5. **S5 — the verifiers compare reader results with one manifest reading.**
   - `verify_graph_artifacts` and `verify_graph_source` take a `ManifestRead` and the readers'
     identities.
   - `verify_generated_cohorts` takes the `ManifestRead` and the `SamplesRead`s. It compares each
     sample file's digest with the manifest, recomputes the disease sets against `realised`,
     compares `realised` with `allocation.allocated`, and, when both cohorts are in scope, checks
     both the manifest's `disjoint` claim and the measured disjointness (`cohort.py:216-331`),
     all from the samples already parsed.
   - None of them does I/O, and each names the file and the manifest in a refusal.
   - `verify_graph_artifacts` keeps returning the manifest-bound map for all four graph roles, as
     it does today (`artifacts.py:138`). That is the opening for M2.2's `kg` and M2.4's
     comparison.
   - **The `kg.json` identification check** (decision 1) is one narrow helper. It hashes the
     file and compares it with the run's one `ManifestRead`; it never reads the manifest itself.
     It is used only before runs that do not parse `kg.json`, and it is not offered as a second
     verification API to consumers that parse their inputs.
   - **The old path forms are a migration aid only.** They remain until S9, for callers not yet
     moved, and S9 deletes them.
6. **S6 — serving.**
   - `build_pipeline` reads `kg.json` with `read_json` and passes the `GraphRead`.
   - The pipeline reads the manifest, the graph tensors and the checkpoint once each, and
     verifies from those reads. After verification nothing is hashed again by path.
   - It keeps all four bound digests, the opening for M2.4. Today only `bound["kg"]` is kept
     (`src/inference/pipeline.py:532`).
   - **The model is built from the final read.** `read_checkpoint`'s parsed object and digest are
     what the build uses and what M2.4 will check, never the data from a selection scan
     (decision 4).
   - `_load_graph_data` (`src/inference/pipeline.py:609-655`) is deleted, collapsing onto
     `file_storage`, and so is the reader copy in `scripts/test_gnn_inference.py:170-182`.
   - **Callers that build the graph apart from its identity move to `read_json`:**
     - `scripts/test_gnn_inference.py:360-366`;
     - the `bind_workspace` sites in `tests/integration/test_pipeline.py`;
     - `tests/unit/test_pipeline_fails_closed.py:248`, `:275` and `:426`.

     The archived `loadcheck.py` is left as evidence of its own commit.
   - The SP table stays the stated exception (`src/inference/pipeline.py:745`, its `build_id`
     check at `:787`).
   - **Behaviour change, accepted by the reviewer:** serving checks the `kg.json` it parsed. So
     `data_dir/kg.json` is no longer read when `SHEPHERD_KG_PATH` names another file holding the
     bound bytes, as `artifacts.py:369-373` already allows. A reload still derives `kg_path` from
     `data_dir` (`src/api/routes/pipeline.py:389`). That is a choice of which file to load, not
     a second verification rule.
   - **Acceptance for this step:**
     - an external `kg.json` that matches the manifest, with `data_dir/kg.json` absent or
       different: startup is decided by the external file;
     - an external `kg.json` that does not match, or any tensor that does not match: refused;
     - a reload of the same deployment still uses `data_dir/kg.json`. A refused reload keeps the
       pipeline already serving, and readiness does not change;
     - the `GraphRead` and its identity arrive together, and nothing is hashed by path after
       verification;
     - graph tensors keep `map_location="cpu"` and `weights_only=True` at serving.
   - **Tests updated:**
     - the graph-binding and reload-availability tests change with the new read order;
     - junk-byte cases become another workspace's valid file;
     - the `load_json` stubs (`tests/unit/test_pipeline_fails_closed.py:309`,
       `tests/unit/test_hop_bound_reaches_the_service.py:75`) are replaced by the real reader.
7. **S7 — measurement.**
   - **Before any model is built or output written:**
     - `resolve_cohort` first;
     - then one read each of: the manifest; the graph tensors; the cohort's samples (default
       fields); for a generated cohort, the other generated split's samples, for its binding and
       the disjointness check; and the checkpoint;
     - then verification from those reads, with the `kg.json` identification check (decision 1).
   - **One digest map** feeds every mode's manifest, replacing the per-mode `artifact_digests`
     (`measure_scorer.py:102-155`). The verifier's bound map is kept instead of discarded
     (`:128`). The manifest's `kg` role is the manifest-bound source graph (decision 2).
   - **Mode A uses the run's single read.**
     - `load_legacy_mode_a_inputs` is deleted, and its docstring (`:205-208`) with it. It
       already delegates to `file_storage` (`:225-227`), so Mode A reads the same files.
     - `build_legacy_mode_a_model` takes the checkpoint dict from that read instead of loading
       the path (`:248`). Its behaviour is otherwise unchanged. It remains item 9's oracle-only
       surface (BACKLOG item 19), and it still cannot run on a pipeline checkpoint.
     - Two statements that tie these functions' lifecycle to the frozen evaluator are corrected
       to name item 9: `src/kg/storage/__init__.py:35-37` ("until both are deleted together"),
       and `build_legacy_mode_a_model`'s "It retires with scripts/evaluate_model.py" (`:240`).
     - The test that proves a C-only run never reaches the loader is rewritten to prove it never
       builds Mode A's model.
   - **What runs where:** on the designated checkpoint only Mode C runs, and only Mode C is bound,
     measured and accepted on it. Modes A and B keep their shared-read regression tests on the
     executable fixtures. Fixture numbers are never offered as capacity readings.
8. **S8 — training.**
   - **Before the run directories, `config.yaml` and `runtime.json`:** one read of the manifest,
     the graph tensors and both sample files (`training_fields=True`), then verification from
     those reads, with the `kg.json` identification check (decision 1). `with_validation` is
     computed there from the parsed validation samples, which is the opening for M2.3.
   - `load_graph_data` and `load_samples` (`train_model.py:398-470`) are deleted, collapsing onto
     `file_storage`.
   - `Trainer.load_checkpoint(path)` becomes `restore_checkpoint(checkpoint)` over a dict already
     read. The resume parent is read once with `read_checkpoint`, and its digest is its role.
   - `training_input_digests` is built only from the reads. The role set is unchanged until M2.2
     adds `kg`, and a `None` digest can no longer be recorded.
   - `TrainerProtocol.load_checkpoint` (`src/core/protocols.py`) follows.
9. **S9 — remove the path forms, pin the contract, move the remaining callers.**
   - **Deleted:** the verifiers' path forms. `compute_input_digests` goes once no caller records
     through it.
   - **Tests that import what goes:**
     - `tests/unit/test_training_provenance.py` imports `compute_input_digests` at module level;
     - it and `tests/unit/test_measurement_mode_a.py:649-660` pin the `file_sha256` re-export in
       `scripts/measure_scorer.py:99`.

     Those tests move with the removal. The re-export is either kept, with its reason, or removed
     with its pins.
   - **Callers moved:** `probe_deployment.py`; `audit_generator_fidelity.py`, including its
     `kg.json` parse and its separate manifest read; and `audit_split_overlap.py`.
   - **A static pin** covers `src/**`, the entry-point scripts (`train_model.py`,
     `measure_scorer.py`, `compute_shortest_paths.py`) and the scripts moved here. It fails on
     any path hash, `torch.load` or JSON load of a known input outside the readers and an
     allowlist that states each entry's reason:
     - the stated exceptions;
     - the SP producer's end-of-run comparison (`compute_shortest_paths.py:472-479`), a warning
       that records nothing;
     - `find_checkpoint` and the evaluation ledger (BACKLOG items 18 and 19);
     - the selection-time checkpoint reads (decision 4; BACKLOG item 19);
     - the `kg.json` identification helper (decision 1);
     - the provenance status readers, which M3b moves onto M2.1's reads: `artifacts.py:278` and
       `:291`, and `src/kg/provenance.py:381`, which hashes the record before `:398` parses it
       through `:275`;
     - the provenance and workspace writers, which are M3b's (`src/kg/provenance.py:233`,
       `src/kg/workspace.py:297-300`).

     Outside the pin: the frozen evaluator, if question 6 keeps it; `build_index.py`; and item
     19's scripts outside the pin's scope (`record_evaluation.py`, `measure_served_pipeline.py`,
     `audit_split_feasibility.py`, `audit_checkpoint_family.py`, `benchmark_sp_lookup.py`,
     `audit_sp_reachability.py`, `build_knowledge_graph.py`, `migrate_checkpoints.py`). Item
     19's sites under `src/**` are allowlisted above.
   - **A dynamic pin** runs each entry point with `compute_input_digests` and every path hash at a
     recording site returning a poison digest.
     - The `kg.json` identification helper gets the true digest, so each run completes and writes
       its record.
     - No poison may reach a record: `training_input_digests`, the measurement manifest, the SP
       sidecar or the pipeline's bound digests.
   - **The last batch is checked at the entry points and the remaining call sites,** not only
     through the shared functions' own tests.
   - Docs and docstrings that name removed functions are updated.
10. **S10 — capacity on the homelab, an executable matrix.**
    - **The subject:** workspace `data/workspaces/hpo_2026_0929_5a` (`$WS` below), checkpoint
      `$WS/checkpoints/hgt/model-02-0.1813.pt` (47,931,987 bytes), SP table `shortest_paths.pt`
      (10,797,575,893 bytes), on the GB10 with CUDA. It is M1's and N1's subject. **The owner is
      asked to confirm that it is the homelab's largest workspace and checkpoint,** as M2.1's
      capacity acceptance requires (§4, M2.1, "Capacity").
    - **Before and after:** the same matrix runs at `90a668f` and at the M2.1 head, in the same
      checkout, venv and boot. The 2026-09-30 readings (23.77 GB VmHWM at ready, 28.45 GB over a
      reload; `ecf17cd`, `scorer-measurement/PLAN_B04_PRODUCTIONISATION.md:1199`, `:1202`) predate
      M1 and are context only.
    - **Pre-flight, once and read-only:** load the designated checkpoint with
      `weights_only=True`, the way Mode C does (`measure_scorer.py:593`), and list its keys.
      - Expected: it loads, without `metadata` or `in_channels_dict`. No Mode C run on this file
        is recorded yet; serving loaded it with `weights_only=False`
        (`src/inference/pipeline.py:973`).
      - If it does not load, the reading records that Mode C cannot run on this file. No
        `weights_only=False` or other fallback is added.

    | Entry point | Command (key arguments) | What runs on the subject | Readings | Compared |
    |---|---|---|---|---|
    | Serving | `measure_served_pipeline.py --workspace $WS --checkpoint $WS/checkpoints/hgt/model-02-0.1813.pt --output <scratch>/served_readings.json --repeats 3 --seed 20260930`. It applies its own environment; its SHA-256 must be equal at both commits | launch to ready, 200 serial `/diagnose` requests, then one reload, each asserting `gnn_plus_shortest_path` | R1: VmHWM and VmRSS at ready, system peak less R0. R4: VmHWM over the reload. R2, R3: VmRSS after | all of these, with equal subject digests, allocator reading and `readings_complete` |
    | Measurement | `measure_scorer.py --checkpoint $WS/checkpoints/hgt/model-02-0.1813.pt --data-dir $WS --split val --cohort-kind generated --modes C --device cuda --output <scratch>/measurement.json` | **Mode C only** | final VmHWM; VmRSS at the entry and exit of `run_mode_c`; CUDA max allocated and reserved; system peak | the totals, since the phases move between commits; VmRSS at `run_mode_c` entry as the retention check |
    | Training | `train_model.py --data-dir $WS --device cuda --conv-type hgt --hidden-dim 256 --num-layers 4 --epochs 3 --seed 42 --output-dir <scratch> --checkpoint-dir <scratch>` (the subject's flags, `scorer-measurement/subject-hgt-2026-09-29/commands.txt`), with `Trainer.train` replaced by a reading; once without and once with `--resume $WS/checkpoints/hgt/model-02-0.1813.pt` | the checks, reads, model, `Trainer` and resume read; no epoch and no checkpoint written | VmHWM and VmRSS at `Trainer.__init__` and at the replaced `train`; CUDA max allocated; system peak | all of these; VmRSS at the replaced `train` as the retention check |
    | SP producer | `compute_shortest_paths.py --kg-path …/kg.json --output-dir <scratch> --max-hops 5 --workers 15` | a full run, about 15 minutes. The published pair in the workspace stays untouched, checked by size and time | parent VmHWM after the graph read; VmRSS when the workers fork (retention); at publication; tree PSS; system peak | all of these; the new sidecar names graph `6cae2d1a…`, 431,902,937 pairs, `max_hops` 5 |

    - **Not applicable, recorded as such:**
      - Mode A: `KeyError` at `measure_scorer.py:251`;
      - Mode B: allowed only beside A (`:443`);
      - the calibration launcher: removed in S0;
      - serving's CUDA allocator counters, which `PLAN_B04_PRODUCTIONISATION.md` §7.3 excludes.

      **M2.1's capacity acceptance is complete with Mode C as measurement's reading.** M2.1 owes
      no Mode A or B reading on a real checkpoint, and none is wanted until a runnable subject is
      named (BACKLOG 19.18).
    - **Why training stops before its first epoch:** an epoch with the default checkpoint
      directory writes `last.pt` beside the designated checkpoint (`checkpoint_paths.py:98-100`,
      `callbacks.py:271-274`), and M2.1 does not change the training loop. No step limit exists,
      only `--epochs` (`train_model.py:283-287`).
    - **Pinned for comparability:**
      - the allocator: measurement, training and the SP producer run with the saved preset's
        value set explicitly, since `measure_scorer.py` and `compute_shortest_paths.py` apply none
        (`src/config/runtime_presets.py:29-35`); serving applies its own;
      - the kernel, driver, Python and torch versions, and the venv;
      - a quiet machine, with the service stopped;
      - the page-cache state;
      - the byte size and SHA-256 of every input: `kg.json`, the tensors, `num_nodes.json`, the
        sample files, the manifest, the SP pair and the checkpoint;
      - the sampler and the probe, with SHA-256 equal at both commits, and the sampling interval:
        0.2 s, as `measure_served_pipeline.py` (`:86`).
    - **The tools:** a `/proc` sampler outside the measured process and a probe inside it, kept
      outside the repository like M1's acceptance client. Their digests are recorded with the
      readings.
    - **Each reading lists the objects alive in each phase** beside its numbers: for measurement,
      for example, the graph tensors, the parsed samples, the checkpoint dict, the model and the
      encoded embeddings. Acceptance 3 explains peaks against this list.
    - **Acceptance, for the owner to confirm with the reviewer:**
      1. **Every run completes at the head** on the real subject, by the same procedure as at
         `90a668f`, with no swap increase and no OOM kill. That means:
         - serving's `readings_complete`;
         - Mode C exits 0 with `cuda_executed`;
         - the training readings, without and with `--resume`;
         - the SP producer publishes into scratch with the expected sidecar, and the workspace
           pair is unchanged.
      2. **Resident memory after the reads is unchanged** within the spread between repeats. A
         rise of about an input's size means a buffer was kept after its parse.
      3. **Each change in a peak is explained** against the recorded file sizes and the objects
         alive in that phase. "About one transient copy" guides the explanation; it is not a
         bound.
      4. **A shortfall is fixed in the reader,** never with a CPU fallback.
    - **The evidence's limits:** `measure_served_pipeline.py` hashes the subject by path before
      the service starts (`:1219-1223`), so its readings name the files designated, not the bytes
      the service read (BACKLOG item 19).
    - **Time:** serving takes about 1 h 37 min per commit; the SP producer about 15 minutes a
      run; Mode C and training's setup take minutes.

**How the acceptance above is shown.**
- **Read and record agree (item 1).** The replacement double fires after each read, by rename and
  by in-place rewrite, at each entry point on the inputs that entry point reads. Each run uses
  and records the bytes it read, or refuses; it never reads A and records B. The inputs:
  - training: a graph tensor, a sample file, the resume parent (S8);
  - measurement: a graph tensor, the cohort's sample file, the checkpoint (S7);
  - serving: `kg.json` as the API loads it, a graph tensor, the checkpoint (S6);
  - the SP producer: `kg.json` (S3).

  **Which outcome is right depends on what was read:**
  - replacing a path already read leaves the run using and recording A;
  - replacing a file not yet read, with bytes that disagree with the manifest already read, is
    refused (item 2);
  - **a republish** (a file replaced together with a manifest that binds the replacement):
    - fired after both the manifest and the replaced file have been read (for `kg.json` in
      training and measurement, after its identification check), the run uses and records A;
    - fired after one of the two has been read and before the other, it is refused;
    - fired before either is read, the run reads and records B. At serving, where the workspace
      holds an SP pair, a `kg.json` republished this way is also refused by the sidecar's
      `kg_digest` (`src/inference/pipeline.py:787-790`), unless the pair is republished with it;
    - at serving, the API reads `kg.json` before the pipeline reads the manifest, so a `kg.json`
      republish between those two reads is refused ("is not the graph").
- **A replacement after the manifest read is refused (item 2).** The double fires on the
  manifest's read and replaces a sample file or a tensor. The refusal names the file, and nothing
  is written or published (S5–S8).
- **Records come only from readers (item 3):** the static and dynamic pins (S9).
- **Capacity (item 4):** S10, on Mode C for measurement.

**Review and delivery.** Three review batches: S0–S5, the removal and the shared base; S6–S8,
the entry points; S9–S10, removal of the path forms, pins and capacity. M2.1 ends as one pull
request, as M1 did.

**What M2.1 does not change.**
- **M2.1 adds no checkpoint schema or provenance gate,** so no checkpoint is refused for what it
  records. Refusals of old checkpoints come with M2.3 and M2.4. This does not promise that every
  mixed or replaced workspace that happened to pass before still passes.
- **M2.1 claims its four entry points and their records,** not consumed-byte provenance across
  the repository. The rest is BACKLOG item 19.
- Not in M2.1: M2.2's `kg` role, M2.3's parent checks and their placement, M2.4's comparator,
  M2.5 and M2.6.
- **Also unchanged:**
  - path-reasoning-only serving, which parses `kg.json` without verifying it (B-2's question);
  - the KG build record and the ontology loader;
  - the SP producer still does not check its `kg.json` against the workspace it publishes into
    (§4, "Not prerequisites");
  - resume from `last.pt` overwriting its parent, which is M3a's;
  - the served identity, which is M3c's.

**Decisions — the owner, 2026-10-09, after the reviewer's advice.**
1. **`kg.json` where it is not parsed (training, measurement): an identification check is kept,
   not recorded.**
   - It compares the file with the run's one manifest reading (S5).
   - The role recorded stays the manifest-bound source graph.
   - The check shows only that the file on disk matched at that moment. It does not mean the run
     consumed `kg.json`, and it does not promise the file will load at serving later.
   - It extends the plan's allowance for "a file nobody is about to parse" to this one refusing
     check. §2 rule 1 now says that its "or that a check compares" applies to the inputs a step
     parses. It is not generalised into a verification API for consumers.
2. **Measurement's `kg` role is the manifest-bound source graph,** as training's. Every mode
   computes from the tensors, never from a graph object. The stated exceptions above, M2.3,
   M2.4 and §5.1 are updated to say so, and the role is never presented as a graph that
   measurement read.
3. **The calibration launcher is removed (S0).** The owner asked first whether it was worth
   keeping, and the check found no necessary user. The question of its digest bracket lapses with
   it.
4. **Checkpoint reads that only select or list stay outside M2.1** (BACKLOG item 19):
   - the reload route's candidate scoring (`src/api/routes/pipeline.py:328`);
   - its second load of the chosen file for the reported metric (`:363`);
   - the training console's list (`src/api/services/training_manager.py:464`);
   - the conv-type classification in `migrate_checkpoints.py:50`.

   The build, and later M2.4, use the final `read_checkpoint`'s object and digest. M2.1 does not
   claim that the score selected, the metric shown and the model served are the same; that is
   M3c's. No candidate is cached to close it.
5. **Audit and evidence writers are follow-ups, each recorded** (BACKLOG items 18 and 19, with
   the relation each does not guarantee and its home).
   - The ledger fix (item 18) binds the expected digest to the read that is appended to, so a
     stale read is detected.
   - It does not close the check-then-replace window, and no support for several writers is
     claimed.
   - No lock, job runner or new ledger is added.

**Open question 6, for the owner after the reviewer's view: `scripts/evaluate_model.py` once S0
removes its only runner.**
- **After S0, nothing runs it.** `tests/unit/test_frozen_evaluator.py` pins its bytes and fails if
  any module under `src` imports it. Without a driver and a comparator, it is no longer an
  acceptance path; it is a file nothing executes.
- **What holds it today:**
  - item 9's reviewed removal order, at its last step, and BACKLOG's deletion gate;
  - the exceptions named in `EVALUATION_COHORTS.md:974-979`, BACKLOG item 11i and
    `PLAN_PHENOTYPE_NORMALISATION.md:312` (which says it "retires with the frozen evaluator");
  - comments that cite its lines as the source of semantics Mode A keeps
    (`src/evaluation/measurement.py:157`, `:166`, `:249-251`, `:702`, `:1112-1113`;
    `tests/fixtures/synthetic_workspace.py:109`; `tests/unit/test_measurement_mode_a.py:546`;
    `measure_scorer.py:203`, `:234`);
  - docstrings that tie other code's lifecycle to it: `measure_scorer.py:5-6` and `:240`;
    `src/evaluation/measurement.py:179-181`; `src/kg/storage/__init__.py:35-37`, which S7
    corrects either way.
- **(a) Remove it and its pin test with S0.**
  - The gate exists so the harness is never left without an acceptance. Without the launcher this
    file accepts nothing, so the gate's reason no longer applies to it.
  - The comments cite its last commit instead of a live file.
  - The lifecycle statements are re-anchored to item 9 (`build_legacy_mode_a_model`), or dropped
    where the code survives: `legacy_ranking`'s sort is shared with the trainer
    (`scorer-measurement/README.md:90`), so its "deletion date" goes.
  - The exceptions in `EVALUATION_COHORTS.md:974-979`, BACKLOG 11i and
    `PLAN_PHENOTYPE_NORMALISATION.md:312` are closed, and `PLAN_TEST_RESULTS.md:763-765` with
    them.
  - Item 9 keeps the rename and the rest of the oracle-only surface, which still runs on
    fixtures: `build_legacy_mode_a_model` and the parity assertions in
    `test_measurement_mode_a.py`.
- **(b) Keep it until item 9's last step,** as a pinned historical file.
  - Under the owner's rule this needs a necessary user, or a stated purpose, and a removal
    condition, written beside it; "it might be used later" is not one.
  - After S0 it is no longer the artefact Mode A is calibrated against. So
    `EVALUATION_COHORTS.md:975`, BACKLOG 11i and `tests/unit/test_frozen_evaluator.py:1-15` and
    `:38-40` are reworded to "a pinned historical file nothing runs".

**Recommended: (a).** Its only remaining purpose is to be a reference for semantics git history
already holds. By the owner's rule, and the reviewer's "not because it might be used later", that
does not justify keeping a file nothing runs.

#### M2.2 — the producer records the graph it consumed

**Change:**
- **`training_input_digests` gains `kg`**, beside the three tensors and the manifest
  (`train_model.py:551-574`).
- **Training does not parse `kg.json`.** Its `kg` role is the digest bound by the manifest the
  run read, against which the tensors it parsed were verified. This is the transitive claim
  `verify_graph_artifacts` already describes (`artifacts.py:73-75`).
- **Every other role's digest comes from M2.1's reads.**

#### M2.3 — the resume parent, checked before anything is restored or written

**Today.** The parent is loaded by `trainer.load_checkpoint` (`train_model.py:510`), which
restores weights, optimizer, scheduler, scaler and state without comparing anything
(`trainer.py:1026-1044`). The child then records the current workspace (`train_model.py:854-860`).
A parent from the old pipeline, or from a same-shaped other graph, would produce a child that
passes M2.4.

**Change.** All of the following happen together with the workspace checks at the top of the run
(`train_model.py:716-717`). That is before the run directories, `config.yaml`, `runtime.json`
(`:720-744`), the model and any checkpoint.
- **A requested parent that does not exist refuses.** The run does not train from scratch
  instead (today `train_model.py:505-507`).
- **The parent is read once (M2.1).** The digest of those bytes is its `resume_checkpoint` role.
- **The parent must carry `training_input_digests` with every workspace role.** A missing record
  or role refuses. So no checkpoint from before the contract can be resumed (decision 2).
- **The parent's graph roles must equal this run's:** `kg`, the manifest-bound source graph,
  and `node_features`, `edge_indices` and `num_nodes` as this run read them. This one is not
  optional: it is decision 1 applied to the parent.
- **Decided by the owner, 2026-10-06 (§8, question 3): its data roles must equal too.**
  - These are `split_manifest`, `train_samples` and `val_samples`.
  - The set of roles must be the same as well, so a parent that ran validation, resumed by a run
    that does not, refuses.
  - **This is a product restriction, not a validity verdict.** Resume continues training on the
    same data. Fine-tuning on other data is not supported yet. A model trained on other cases
    of the same graph is not thereby invalid: that is a different question from a graph
    mismatch.
  - **What an operator will meet:**
    - adding cases to the workspace and resuming: refused;
    - changing the train/val allocation, or going from validation to none: refused;
    - re-serialising the JSON, or rewriting the manifest, with the same cases: refused, since
      identity is bytes. That is the cost of strict identity;
    - the same bytes at another path: accepted.
  - **It gates data, not settings.** M3a records the settings in effect; it does not require a
    resumed run's settings to equal its parent's.
- **Whether this run validates is decided here, from the samples already parsed.** It validates
  exactly when the parsed `val_samples` are not empty, which is the condition
  `create_dataloaders` applies (`train_model.py:606-624`). Today that is only known after the
  loaders are built (`:748`, `:857`), by which time the run has written its directories.
- **The comparison is M2.4's comparator** over a wider set of roles. It is one function, and the
  set of roles is its argument.
- **`Trainer.load_checkpoint` restores from the dict already read and checked.** It no longer
  reads the path a second time (`trainer.py:1009-1016`).

**Why the same data rather than a lineage walk.**
- By induction, every checkpoint in a resume chain then read the same sample files. Each link
  was refused unless it matched, and the chain can only start from a checkpoint that passed
  M2.2.
- So a checkpoint's own record covers its whole history. R10 (§5.4) needs no walk over ancestor
  files, which may since have been overwritten, and no registry.
- This rests on the data-role rule. If fine-tuning on other data is added later, the history
  set §8 question 3 describes lands in the same change, so R10 never reads a scope narrower than
  the training history.

The rest of resume — the overwrite guard — is M3a's.

#### M2.4 — the consumers: serving and measurement

- **They compare the checkpoint's recorded `kg`, `node_features`, `edge_indices` and `num_nodes`
  digests with the workspace's.** For serving these are taken from M2.1's reads. For
  measurement, `kg` is the manifest-bound source graph (M2.1's stated exceptions, amended
  2026-10-09), and the other three roles come from M2.1's reads.
  - Serving keeps only `kg` today (`src/inference/pipeline.py:532` at `90a668f`), and all four
    after M2.1's S6. Measurement discards them today (`measure_scorer.py:128`).
  - A missing `training_input_digests`, or a missing role, refuses.
- **The comparison lives in the slot `verify_fingerprint` deferred** (`fingerprint.py:168-176`),
  so there is one model↔graph comparator.
  - The structural fingerprint stays as a diagnostic, never as identity.
  - The probe asserts through the same function instead of its own comparison
    (`probe_deployment.py:812-836`).
- **A refusal raises (M1)**, so it cannot become a fallback.

#### M2.5 — SP

An unrecorded SP binding refuses. This ends the legacy "serves with a warning" state (R4), under
decision 2.

#### M2.6 — inventory before deployment

A read-only listing of every checkpoint, SP table and workspace that M2 would refuse, and why.
- It is an option of an existing audit (`scripts/audit_checkpoint_family.py` or the deployment
  probe), not a new tool.
- It is deployment preparation, not a second execution path.
- Old artifacts are archived, not deleted.

**Consequence.** No existing checkpoint records the `kg` role, so every one of them is refused
after M2, as a model to serve or measure and as a parent to resume, and the model is retrained.
This follows decision 2.

**Acceptance for M2**, besides M2.1's:
- a checkpoint from workspace A is refused against a same-shaped workspace B, at serving and at
  measurement;
- **resume — each of these is refused before any state is restored, and leaves no run
  directory, `config.yaml`, `runtime.json` or checkpoint behind:**
  - a parent from before the contract;
  - a parent from a same-shaped other graph;
  - a parent that read other sample files;
  - a parent that ran validation, where this run's parsed `val_samples` are empty;
  - a requested parent that does not exist, and no run trains from scratch instead;
- **resume that succeeds:**
  - a resume from a qualifying parent succeeds, and the child's record names the bytes of that
    parent;
- a fresh build → train → resume → serve passes;
- the inventory runs on the homelab.

### M3 — producers record what consumers need, and consumers check it

Three changes, each complete. M3a can proceed in parallel with M3b and M3c.

**M3a — training records the setup that took effect (R6, R11).**
- **`checkpoint['config']` gains the values in effect:**
  - the loader and sampling settings;
  - compile, both requested and whether it enabled;
  - AMP and warmup, both requested and effective.
- **`producer`:** the code revision, plus a dirty flag that counts untracked files.
- **`environment`:** §5.3.
- **The seed is applied before the data loaders and the model are built.** Today it is applied
  in `Trainer.__init__` (`trainer.py:255-257`), after the loaders (`train_model.py:748`) and the
  model (`train_model.py:751`), so the recorded seed does not govern initialisation.
- **Device.** `auto` without CUDA refuses, as measurement's does (§5.2). Today training turns it
  into CPU silently (`train_model.py:688-691`).
- **Resume, what M2 leaves.** A write-time guard in `ModelCheckpoint` refuses to overwrite the
  file the run resumed from.
  - After M2.1 the record no longer depends on that file: the parent's digest comes from the
    bytes read.
  - The file the record names should still exist, for the inventory and for audit. The WebUI's
    default resume target, `last.pt`, is exactly that file (`training_console.py:398-404`,
    `callbacks.py:271-274`).
- **One constraint on every new field:** measurement loads checkpoints with `weights_only=True`
  (`measure_scorer.py:593`), so everything new is a primitive container or a scalar.

**M3b — the dataset record is required (R5).**
- **Refused:** training, serving and measurement refuse a workspace whose `kg.provenance.json`
  is missing, unreadable or does not match. The manifest's binding to the record becomes
  required at write and read.
- **Read through M2.1, like every other input.**
  - The record is hashed and parsed from one buffer. Today `provenance.py:275` reads it and
    `:381` hashes it again.
  - It is compared with the run's single manifest reading and `kg` digest. Today
    `artifacts.py:278` reads the manifest again and `:291` hashes `kg.json` again.
- **Two facts about a record, kept apart and shown apart:**
  - **Its binding state:** *verified*, *mismatch*, *unrecorded* or *unverifiable*. Is this
    record's file the one the manifest binds, and does its `kg_digest` name this workspace's
    graph? This is the required relation.
  - **Its origin:** `files` or `synthetic` (`provenance.py`). This is a description, never a
    binding state.
- **A synthetic workspace is fully bound and passes every check.** That covers the test fixtures
  and a synthetic acceptance run. Wherever its results are shown, it is shown as synthetic,
  never as plain *verified*. No acceptance relies on skipping a check.
- **Missing roles.**
  - **All four `SOURCE_ROLES` are inputs a real build requires.** The builder stops without the
    two annotation files (`scripts/build_knowledge_graph.py:219-232`) and always loads both
    ontologies (`:260-263`).
  - **A `files` record with a missing role therefore means a required input went unidentified**
    (`:322-328`). The formal pipeline refuses it. The builder is fixed to identify the input;
    the check is not relaxed.
  - **A synthetic record lists every role as missing by construction** (`provenance.py`,
    `missing_roles`). That is its origin, not an omission.
  - **What the record says it does not establish is displayed, never refused**:
    `INCOMPLETE_BY_DESIGN` lists parser versions, builder parameters and imports. Closing that
    gap is the KG record v2 (below).

**M3c — what is served is identified (R8).**
- **The checkpoint's digest comes from M2.1's read** of the bytes loaded.
- **One structured field replaces the two version strings.** It replaces
  `InferenceResult.model_version` and `kg_version` (`src/core/types.py:448-449`) and
  `DiagnoseResponse.model_version` (`diagnose.py:180`). Its named parts:
  - the checkpoint's SHA-256, or a stated absence where B-2's policy allows serving without one;
  - the SHA-256 of the verified workspace's `kg.json` and `split_manifest.json`;
  - the SP table's `build_id` and binding state, or "absent";
  - the scoring mode;
  - the serving settings that change scores, as in effect — the device among them;
  - the dataset record's origin (M3b).

  This upgrades the existing result type. It adds no parallel record, and no version string
  that would have to be parsed.
- **Filled from `get_pipeline_config()` when the result is produced.**
  - The API response copies it from the result it is answering with, not from shared state a
    reload could swap. `app_state.model_version` goes (`src/api/main.py:84, 554-558`).
  - The literal versions go: `pipeline.py:1062, 1129-1130, 1703` and `diagnose.py:201, 351`.
  - The Diagnosis tab's downloadable report has a "Model version" line
    (`src/webui/components/diagnosis_panel.py:484-493`, written at `:559`). It carries the
    identity's parts instead, and so does the CSV export.
  - The response schema changes, and the milestone's documentation says so.
  - Every existing consumer — the API, the UI, the exports — moves in the same change. No
    consumer is left choosing between an old field and a new one.
- **Device.** Serving's `auto` without CUDA refuses (§5.2). Today it turns into CPU silently
  (`pipeline.py:965-966`).

### M4 — item 14, phase 1

`docs/working/PLAN_TEST_RESULTS.md` revision 4 holds the detail.
- **Import:** a test cohort version is imported from the CLI, bound to the graph.
- **Run:** a Mode C test runs from the CLI with R1, R2, R3, R5 and the cohort↔graph binding
  checked, on M2.1's reads. It records R10 as §5.4 defines it, and the R11 facts.
- **Keep and record:** every report is kept, and is recorded automatically (`--record`) or later
  by hand, from the CLI.
- **Display:** the Diagnosis tab's existing model status shows the loaded checkpoint's records,
  found by its digest (M3c) and read through the API.
- **No new protection mechanism is needed.** Phase 1 adds no operation that the network can
  trigger to change state; a UI "register" button would be such an operation, and it belongs to
  phase 2 with its protection.
- **Acceptance for R10** is §5.4's list.

### M5 — item 14, phase 2

All of it is in `PLAN_TEST_RESULTS.md`:
- the Test tab;
- the job runner, with progress and cancellation;
- the model list;
- the network-side restrictions those operations need;
- the advanced ordering and arbitration.

**Training start's explicit `checkpoint_dir`** is honoured verbatim today. It is restricted to
directories the server resolves in the same change that upgrades the checkpoint listing, because
the two share the write path. It is a visible behaviour change, documented there. No input
identity depends on it, so it does not hold up M2–M4.

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
| Checkpoint (`callbacks.py:296-327`) | `training_input_digests` gains `kg`, the manifest-bound source graph (M2.1's stated exceptions); every other digest comes from the bytes read; `config` gains the values in effect; new `producer` and `environment` keys | M2, M3a |
| `split_manifest.json` | Binding to `kg.provenance.json` required | M3b |
| `kg.provenance.json` | Unchanged until the KG record v2 | — |
| SP sidecar | Unchanged; it already records `kg_digest`, `build_id` and `max_hops` | — |
| Measurement manifest and report | Input digests from the bytes read, except `kg`, the manifest-bound source graph (M2; M2.1's stated exceptions); environment facts; the workspace's graph digests for both cohort kinds; R10's evidence and labels (§5.4) | M2, M4 |
| Ledger | v2, per item 14 | M4 |
| `InferenceResult`, `DiagnoseResponse`, serving state | One structured served-identity field replaces `model_version` and `kg_version`; `get_pipeline_config()` carries the settings in effect | M3c |

### 5.2 Required versus displayed

| Kind of relation | Treatment |
|---|---|
| R1–R5, R7 | **Required**: anything but *verified* refuses |
| R6, R11 | **Recorded**, and displayed beside results. An environment difference never refuses |
| R10 | **Recorded as four labels** (§5.4). It never refuses, because the true label is written, *unverifiable* included |

**Capability checks are a separate matter, and they stay.**
- That an environment difference never refuses is a provenance policy. It does not waive a
  check that the hardware can do what was asked.
- Measurement already refuses `auto` without CUDA instead of substituting CPU
  (`measure_scorer.py:157-188`), because CUDA is a stated project requirement.
- Training and serving turn `auto` into CPU silently today (`train_model.py:688-691`,
  `pipeline.py:965-966`). M3a and M3c align them with measurement.
- An explicit `cpu` stays available for development, and is recorded as such.

### 5.3 Environment facts

**What is recorded:**
- the Python, torch and CUDA versions, and torch's build identifier;
- the cuDNN version;
- the GPU model, compute capability and driver;
- whether deterministic algorithms are on, and in which mode;
- the TF32 setting;
- the PyG version, and which of its native extensions are importable.

**What they are for.** They say under which conditions a number was produced.
- **Reproducibility is limited by conditions these fields do not fully capture.** PyTorch states
  that completely reproducible results are not guaranteed across releases, individual commits or
  platforms, nor between CPU and GPU executions even with identical seeds. Its deterministic
  settings cover known operations only (https://docs.pytorch.org/docs/stable/notes/randomness.html).
- **Matching fields are a precondition for comparing two results, not a guarantee that they
  agree.** A difference between two results calls for investigation under the recorded
  conditions, not a verdict of defect.
- **Deterministic mode guards against known non-deterministic operations.** It does not certify
  correctness.

### 5.4 R10, defined

R10 says how a test cohort relates to the data the model was trained and validated on. It is
recorded, never enforced (§5.2). Its rules exist to keep a label from saying more than its
evidence.

**Scope: what the training run recorded.**
- **Two scopes, each named for what it is.**
  - *Recorded training inputs*: the `train_samples` the checkpoint's `training_input_digests`
    names.
  - *Recorded validation inputs*: its `val_samples`. Early stopping and checkpoint ranking read
    these (`trainer.py:124, 400, 412`), and so does auto-selection, through `RANKING_SCORE_KEYS`
    (`src/utils/checkpoint_paths.py:45`).
- **The second scope is not called "selection data".** A person choosing among checkpoints by
  other means is not recorded, and the label does not claim to know it.
- **The scopes cover the whole resume history.** M2.3 refuses a resume over other sample files,
  so every ancestor of a checkpoint read the same files. This rests on M2.3's data-role rule,
  which the owner has decided (§8, question 3).

**Units: two labels per scope, four in all.**

| Label | Counts | Reported as |
|---|---|---|
| **Disease overlap** | The test cohort's distinct `disease_id` values that occur as a `disease_id` in the scope's samples | *k* of *n* distinct test diseases, and the number of test cases whose disease is among them |
| **Phenotype-set overlap** (same disease, same phenotype set) | Test cases whose disease equals that of a sample in scope, and whose set of distinct phenotype ids equals that sample's. Order and repetition are ignored | *k* of *N* test cases |

- **The two labels answer different questions.** Disease overlap says whether the model was
  trained toward this answer. Phenotype-set overlap says whether it was trained on the same
  clinical presentation of the same disease. A cohort can be disjoint by phenotype set and
  overlapping by disease, and both labels are shown.
- **Phenotype-set overlap does not say the model saw the same scoring input.**
  - Repetition is not neutral. Mode C averages over every listed position
    (`src/evaluation/measurement.py:1285-1310`), and nothing removes a repeated id on the way
    (`src/kg/storage/file_storage.py:94-100`, `src/kg/data_loader.py:646-652`,
    `measurement.py:1193-1223`).
  - So two cases with the same set can be scored differently. A training case `[0, 1]` and a
    test case `[0, 0, 1]` average to different patient vectors, and can rank the true disease
    differently.
- **The set is the unit this label's purpose needs.** R10 shows a reader where a test result
  may rest on what the model was trained on. A comparison that kept repetition would report
  `[0, 1]` and `[0, 0, 1]` as unrelated, and hide exactly that.
- **Whether a case may list a term twice is a question about the data, not about this label.**
  Mapping can produce one, when two source terms map to the same node. A rule would have to
  apply alike at import, in training, in measurement and at serving.
  `PLAN_PHENOTYPE_NORMALISATION.md` proposes one, for the owner to decide. R10 removes nothing
  either way.
- **The patient id is not part of a case.** Generated patient ids are labels, not identities.
- **Near-duplicates are not counted** — subsets, supersets, a term or two apart — and the
  label's name states the rule.
- **Ids compare within one graph.** The test cohort's ids are the graph's indices under the
  cohort↔graph binding. The samples in scope were verified against the graph the checkpoint
  records, and R3 has verified that this is the measured graph. Equal ids are therefore equal
  concepts. Without R3 verified, the run is refused before R10 is computed.

**Evidence: only the bytes the checkpoint names.**
- **Only a file with the recorded digest is evidence.** Each scope's evidence is the sample file
  whose SHA-256 equals the digest the checkpoint records for that role.
- **Candidates are places to look; the digest decides:**
  - the measurement workspace's own `train_samples.json` and `val_samples.json`;
  - every `*_samples.json` in directories the operator names (`--training-evidence DIR`).

  Each candidate is read once (M2.1) and counts only if its digest matches. This is
  `find_checkpoint`'s rule applied to samples (`sidecar.py:366-383`).
- **The current workspace is a candidate like any other, never a default.** A file with another
  digest — a rebuilt workspace, another seed, another split — says nothing about what the model
  saw, in either direction.
- **The report records the evidence:** for each scope, the digest the checkpoint names, whether
  matching bytes were found, where, and the counts.

**States, per label:**
- ***overlap*** — the evidence is verified and the intersection is not empty. Shown with its
  counts.
- ***none*** — the evidence is verified and the intersection is empty.
- ***unverifiable*** — one of these:
  - the checkpoint names a file, and no candidate holds those bytes;
  - the candidate that holds them cannot be read;
  - for the training scope, the checkpoint names no `train_samples` at all. Every training run
    records it, so its absence means the record is not one the pipeline wrote.

  Shown as *unverifiable*, never as zero and never as *none*.
- ***no validation inputs recorded*** — validation scope only. The checkpoint records no
  `val_samples`, because training ran no validation pass (`train_model.py:525-529`). Shown as
  such, not as *none*.

**Acceptance (M4):**
- **The current split is disjoint from the test cohort, but the model was trained on another
  one.** The current file's digest differs, so it is not used. With the real evidence absent,
  the label is *unverifiable*. With it supplied, the overlap is reported.
- **A resume over other sample files is refused at training** (M2.3), so a chain cannot widen
  the scope behind the record.
- **The evidence file has moved, with its bytes intact.** It is found through
  `--training-evidence`, and the label is computed.
- **The evidence file is missing, or altered.** *unverifiable*, never zero.
- **The phenotype sets are disjoint, and the diseases overlap.** Phenotype-set *none*; disease
  *overlap*, with counts.
- **A training case `[0, 1]` and a test case `[0, 0, 1]` of the same disease** count as
  phenotype-set *overlap*. The report's definition of the label says the scoring inputs may
  differ.

## 6. The claim boundary

**For the documentation:**

> When every required check passes, the pipeline establishes this: for each relation listed as
> *verified*, the bytes each step parsed were the ones its records name, and the records agree
> with each other at those relations.
>
> It does not establish:
> - who produced or changed a file;
> - that a record is truthful;
> - integrity against anyone able to modify the data and its records together. The records are
>   unsigned, and a digest is not a signature (`docs/working/PLAN_ONTOLOGY_PHASE2.md:812`);
> - that a result reproduces under other recorded conditions;
> - what data a model saw beyond its recorded training and validation inputs. A person's choice
>   among checkpoints is not recorded (§5.4);
> - that a test cohort stayed independent, which is an institutional operating rule;
> - clinical validity.

**For the UI**, one line beside the identities:

> Checked: these files are the ones their records name, and the records agree on the relations
> marked ✓. Not checked: who changed them, or anything a person able to edit both files and
> records could alter.

## 7. Out of scope

- Accounts, keys, signing, an append-only or chained log.
- A lineage registry, or walking a resume chain through ancestor files (M2.3 makes it
  unnecessary).
- File locks or a locking service. M2.1 reads each input once and compares it with one manifest
  reading, so a mixed read is refused rather than locked out. This is not a snapshot (§4, M2.1,
  "Boundaries"). The evaluation ledger's check-then-replace window stays open, and the ledger
  stays single-writer by operational rule (BACKLOG item 18; M2.1 decision 5).
- A general verification framework. Each check lives at its consumer's entry point.
- Conformance claims to SLSA, in-toto or PROV.
- Reading or migrating old-pipeline artifacts (decision 2).
- `scripts/build_index.py`, whose output is detached from diagnosis.

## 8. Questions, open and decided

1. **Revision source.** Does any deployment run without `.git`? The measurement's revision is
   otherwise `None` (`measure_scorer.py:192-200`).
   - **The reviewer's recommendation:**
     - the revision is stamped at packaging or deployment, from the same source that versions
       the deployment;
     - when it cannot be obtained, it is recorded as *unrecorded*, never guessed from `main` or
       the latest commit;
     - no version-management system is added for it.
   - Whether such a deployment exists is for the deployment inventory to answer.
2. **B-2 policy.** When no checkpoint is configured at all, should path-reasoning-only serving
   remain available, or require an explicit opt-in?
   - **The reviewer's recommendation, given the formal GNN diagnosis requirement:**
     - by default, no alternative score is served;
     - if research needs path-only results, they come later, by explicit opt-in, with result
       semantics that tell them apart;
     - no new formal path is built to keep the old fallback.
   - This is a recommendation. The owner has not decided.
3. **Fine-tuning on other data — decided by the owner, 2026-10-06.** In the first version,
   resume continues the same data as its parent (M2.3). Fine-tuning on other data is not
   supported yet. This is a product restriction, not a validity verdict, and it gates data, not
   settings.
   - **If fine-tuning is needed later,** it uses the same training entry point and checkpoint
     schema. The parent's recorded train and validation digests are carried into the child,
     and R10 takes the union of every set it can verify. If any one of them cannot be
     verified, R10 says the scope is incomplete. This is neither a separate project nor a
     second training pipeline, and nothing is built for it now.
