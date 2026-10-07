# PLAN — the provenance contract: every link in the pipeline checked where it is used

**Status: revision 2, amended twice after review. M1 is implemented, awaiting review
(2026-10-07; §4, "What M1 did"). M2–M5 are not implemented.** Facts about the code are cited at
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

#### What M1 did (2026-10-07, awaiting review)

**A requested model is built, or the build raises `PipelineBuildError`**
(`src/inference/pipeline.py`). A model is requested when a checkpoint path or a pre-loaded model
is given. Each condition below used to log and return without the GNN:
- PyTorch is not available;
- there is no graph to compute embeddings from, whether no `data_dir` or `graph_data` was given
  or the directory lacks its tensors;
- the checkpoint is missing, cannot be read, or does not build a model over this graph
  (`_load_model_from_checkpoint` raises, and the reader's or builder's error is kept as the
  cause);
- shortest paths are required (`sp_optional=False`) and not available. This used to switch the
  GNN off, so a configuration requiring both signals was served with neither.

One invariant is added: `_gnn_ready` is set only with a model and its embeddings in hand. It
cannot be reached today, because every loader raises. It stops a loader that one day returns
`None` again from leaving a pipeline that reports a GNN and scores every GNN term as zero.

**The API is unchanged.** `build_pipeline` already re-raised, startup already published nothing
on failure, and the reload route already built the candidate before publishing it. So the
refusal lands as the acceptance requires, once the pipeline raises.

**Unchanged, deliberately:**
- with no model source at all, the pipeline still serves path reasoning. Whether that needs an
  opt-in is B-2's policy question (§8, question 2);
- absent shortest paths with `sp_optional=True`, the default, still give a GNN-only pipeline
  (`DISEASE_SCORER_POLICY.md` §2).

**Tests:**
- `tests/unit/test_pipeline_fails_closed.py` covers:
  - each condition above, plus the invariant;
  - controls: a sound checkpoint builds, a GNN with optional SP absent builds, and no model
    source still builds path reasoning;
  - startup with an unreadable checkpoint: nothing published, `/diagnose` 503, with the same
    startup and a sound checkpoint as control;
  - reloads to an unreadable checkpoint and to one trained over another graph: refused, with
    every served field the same object as before.
- The graph-binding tests' stand-in now finishes the model build. Before, it ended in the
  fallback this change removes.
- `tests/unit/conftest.py` puts `app_state.real_pipeline_requested` back after each test.
  - Three files left it set, so `test_diagnose_reserved_fields.py` failed whenever it ran after
    them.
  - This was already true at `4298aac`; the alphabetical full run hid it.

**Mutation check.** 8 mutants were run in a fresh copy of the tree, each restoring one old
swallow, the invariant's removal, or both together. All 8 were caught.

**Found, not changed: one for M2.** `_calculate_gnn_score` clamps a disease index past the end
of the embedding table to the last row (`disease_idx = min(...)`). A graph whose node mapping
disagrees with its tensors is therefore scored against another disease's embedding, not
refused. The workspace binding prevents that mismatch for file-backed pipelines. A caller
supplying `graph_data` in memory has no binding. M2's model↔graph check is where the refusal
belongs.

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
    `find_checkpoint` (`src/evaluation/sidecar.py:366-383`) and the M2.6 inventory.
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
- **Two stated exceptions.**
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
  and checkpoint. A shortfall is fixed in the reader. A CPU fallback is not a remedy.

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
- **The parent's graph roles must equal what this run read:** `kg`, `node_features`,
  `edge_indices` and `num_nodes`. This one is not optional: it is decision 1 applied to the
  parent.
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
  digests with the workspace's**, taken from M2.1's reads.
  - Serving keeps them already (`pipeline.py:501`); measurement discards them today
    (`measure_scorer.py:128`).
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
| Checkpoint (`callbacks.py:296-327`) | `training_input_digests` gains `kg`, and every digest comes from the bytes read; `config` gains the values in effect; new `producer` and `environment` keys | M2, M3a |
| `split_manifest.json` | Binding to `kg.provenance.json` required | M3b |
| `kg.provenance.json` | Unchanged until the KG record v2 | — |
| SP sidecar | Unchanged; it already records `kg_digest`, `build_id` and `max_hops` | — |
| Measurement manifest and report | Input digests from the bytes read (M2); environment facts; the workspace's graph digests for both cohort kinds; R10's evidence and labels (§5.4) | M2, M4 |
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
- File locks or a locking service. One read per input (M2.1) makes them unnecessary.
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
