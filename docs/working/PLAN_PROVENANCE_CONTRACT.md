# PLAN — the provenance contract: every link in the pipeline checked where it is used

**Status: revision 2, amended twice after review. M1 is implemented, code-reviewed with no P1 or P2
at `ed4872a`, and accepted on the homelab GB10 at `7714459` (evidence recorded by SHA-256 below; the
files are archived on the homelab, not in the repository); the reviewer re-checked that evidence
with no P1 or P2 (2026-10-08; §4, "What M1 did" and "M1's acceptance on the homelab"). M2.1's work
plan was reviewed and revised with the owner's decisions of 2026-10-09 and 2026-10-10. The re-review
of `0708789` found one P2; the incremental review of `7e1af96` found it closed and no new P1 or P2;
the review of the decision record at `45d9dea` found no P1 or P2 (§4, "M2.1 — the work, at
`90a668f`"). M2.1's S0 is implemented (2026-10-10) and code-reviewed with no P1 or P2
(`45d9dea..3433bf8`); S1–S5 are implemented (2026-10-10) and code-reviewed with no P1 or P2 (`e3eefee..6f16373`);
S6 and S8 are implemented (2026-10-11); their batch (`8421b8c..cddad9d`) is code-reviewed with no P1 or P2
(2026-10-12), and the
later increment that narrowed their claims (`cddad9d..3e8e559`) was reviewed with no P1 or P2
(2026-10-12), which is not a review of that batch. S7 is implemented for Mode C only (2026-10-11)
and code-reviewed with no P1 or P2 (`3d53c9b..7327798`, 2026-10-12), as that declared scope, not as every
mode migrated. M3a's seed ordering is implemented ahead of the rest of M3a (2026-10-11) and
code-reviewed with no P1 or P2 (`7327798..290c03a`, 2026-10-12). Measurement runs that include Mode A
were not migrated; since the owner's clarification of 2026-10-12 they were deleted by the Mode A/B
retirement's R2 (2026-10-12; code-reviewed with no P1 or P2 in
`1d0b5a6..06fd0f1`, 2026-10-12), so since R2 every measurement run takes S7's reads;
that measurement-wide state rests on R2 and is not covered by S7's C-only review. **M2.1's technical
work is complete (2026-10-13)**: the owner accepted S10's capacity on its designated subject, with the
remaining uncertainty explicitly accepted, on the reviewer's recommendation ([the capacity README §6.2](scorer-measurement/capacity-m2.1-2026-10-12/README.md#62-acceptance-2026-10-13));
this is not a merge, and M2 is not complete. The record that led there: S9 is implemented (2026-10-12) and code-reviewed with no P1 or P2 (`02a6954..f481882`, 2026-10-12),
and S10's tools, kept outside the repository, passed their re-review with no P1 or P2 (draft 3, 2026-10-12),
and S10 ran on 2026-10-12 at both commits ([evidence](scorer-measurement/capacity-m2.1-2026-10-12/README.md)): every acceptance 1 check passes at
the after commit; comparability has 1 UNKNOWN (the service state, supplemented by the owner's statement; the
comparison's exit status 2 kept); acceptance 2 and 3 are explained except the serving reload's resident rise (R3
+0.023 to +0.031 GB), whose source is not located; the reviewer's evidence review (2026-10-13) recommends a
limited diagnostic of consecutive reloads before acceptance 2 is closed; the owner chose it (2026-10-13), and it
ran on 2026-10-13 ([`reload-diag-2026-10-13/`](scorer-measurement/reload-diag-2026-10-13/README.md)): each process grew by about 0.4–0.5 GB
at every reload at both commits, before M2.1 as after it, and the difference between the commits is ambiguous, so
the owner ended the reload-memory investigation (2026-10-13) with the wheel's bundled mimalloc purge defects as
a strong candidate cause, attribution not established, and accepted S10's capacity with that uncertainty explicit. R2 (`e210b12`) and item 8 (`06fd0f1`), the latter a
change to the shared reader `read_samples` made after S1–S5's review, are code-reviewed with no P1 or P2
(`1d0b5a6..06fd0f1`, 2026-10-12), as is the S6/S8 batch (above); none of these reviews accepts anything on the
homelab. M2.2–M2.6 are implemented (2026-10-13; their sections in §4, "As implemented"); the
reviewer's code review of `810d344..0aabfc3` (2026-10-13) found three P2 and no P1, and its
re-review of the fixes (`0aabfc3..26fe84c`, 2026-10-13) closed all three with no new P1 or P2.
**M2's acceptance ran on the GB10 at `82d7573` (2026-10-14) and passed**, demo and real data, as the
reviewer's review of the evidence found ([evidence](m2-acceptance-2026-10-14/README.md)): a
functional acceptance of the contract, not a capacity reading (the probe ran under the native
allocator, not the saved `cuda_async`) and not a measure of model quality. **M2's technical work is
complete and its acceptance, within the scope agreed, is complete** (the owner, 2026-10-14, on the
reviewer's recommendation); the owner merged it into `main` on 2026-10-14 (#121, `9fe5113`). The
probe's allocator start-up was fixed since (`b8a43c8`), reviewed with no P1 or P2 and confirmed on
the GB10: its demo phases ran under `cudaMallocAsync`, as torch reported (2026-10-14). M2.1's
acceptance stands as recorded. M3a's design was revised on 2026-10-15 after the reviewer's design
review and the owner's decisions 8–10 (§1; §4, M3a). **M3a is implemented (2026-10-15):** M3a-1
with its four review fixes (`26a7268..fbd850d`), and M3a-2, M3a-3 and M3a-4 with the stop status,
the console's continue block, the migration tool and the probe (`6143e17` onward; §4, "M3a as
implemented"). The reviewer's code review of `8876d2b..094b90d` (2026-10-15) found three P2 and
no P1; its re-review of the fixes (`094b90d..815d763`, 2026-10-15) closed all three and found no
new P1 or P2. **M3a's functional acceptance, within the scope agreed, is passed (2026-10-15)**, in
two rounds on the GB10 that the reviewer reviewed: round 01 at `4b1571e` (real data R1–R8) and
round 02 at `c4a21b5` (demo A–E and M), the product code the same in both (§4, "How M3a's
acceptance is run"). It is a functional acceptance, not a capacity, model-quality or clinical
benchmark reading; neither round's report is a complete single run; two P3 are tracked. The owner
merged M3a into `main` afterwards (#122, `2049b64`, 2026-10-15).** **M3b is implemented
(2026-10-15) and code-reviewed with no P1 or P2 at `04ecf42` (2026-10-16)** (§4, "M3b as
implemented"); **its GB10 functional acceptance, within the scope agreed, is passed
(2026-10-16)**: one run at `c80c1e4`, the demo and the real workspace `hpo_2026_0929_5a`, 75 of 75
probes, reviewed by the reviewer (§4, "M3b's acceptance on the GB10"). It is a functional
acceptance, not a capacity, AMP, model-quality or clinical reading. The owner merged M3b into
`main` afterwards (#123, `cbba015`, 2026-10-16). **M3c is implemented on the branch (2026-10-16)
and awaits review** (§4, "M3c as implemented"); it has no GB10 acceptance yet. M4 and M5 are not
implemented. Facts about the code are cited at `627ed08`, before M1;
the code was unchanged at `463a0df`. §1 records decisions already made; §4 is the order of work.

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
| 8 | **Resume continues an unfinished schedule, with every learning setting inherited** (M3a-3). The schedule is rebuilt from the parent's record and only the remaining epochs run; a finished schedule is continued only by `extend`, at a fixed rate (M3a-4). A mode that starts from a parent and changes learning settings is not built in M3; it is a prerequisite of the GNN experiment phase (BACKLOG item 23). This adds to decision 7, which gated data and said nothing about settings | Reviewer's recommendation (2026-10-15), confirmed by the owner (2026-10-15) |
| 9 | **One directory per training run** (M3a-2). No run overwrites another's files. Outputs in the old layout are moved once, explicitly, by the owner, with no runtime fallback. An explicitly entered model path stays available as a locator; whether a file is accepted is decided by its bytes | Owner, 2026-10-15 |
| 10 | **Serving loads checkpoints with `weights_only=True`** once M3a's shared checkpoint payload is in place | Owner, 2026-10-15 |

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
   - This applies to the inputs a step parses. `kg.json`, where a step does not parse it, may
     be identified, without being recorded, by one refusing check against the step's manifest
     reading (§4, M2.1 decision 1; the owner, 2026-10-09). This is not a verification API for
     consumers that parse their inputs.
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
| R8 | A diagnosis result ↔ the model, graph and SP table that produced it | `InferenceResult.model_version` / `kg_version` exist (`src/core/types.py:448-449`) | Filled with a class version and `"unknown"` (`pipeline.py:1062, 1129-1130, 1703`). The API answers with the literal `"1.0.0"` (`src/api/routes/diagnose.py:351`) | **M3c** (implemented 2026-10-16, awaiting review; §4, "M3c as implemented") |
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
  - **Amended 2026-10-09 after review:** that the buffer is released is shown by release checks
    on the readers (§4, M2.1's S2–S4), never inferred from resident memory. The allocator may
    keep a freed buffer's pages resident, and live parsed objects may pin them.
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
  - **Amended 2026-10-10 by the owner** (§4, M2.1 decision 7): it is measured on S10's
    designated subject, and the acceptance is limited to that subject. It does not claim to
    cover all of the largest inputs (another workspace has larger sample files) or every
    deployment.

#### M2.1 — the work, at `90a668f` (2026-10-08; revised 2026-10-09; decisions 2026-10-10)

This is how M2.1 would be built. It was mapped against the code at `90a668f`, the merge of M1,
by independent readers of each entry point and of the shared readers. A completeness critic then
checked the map and this breakdown, and a second check verified this text against the code.
**Citations in this subsection are current at `90a668f`**; those above, taken at `627ed08`, are
left as written. S0 is implemented (2026-10-10) and code-reviewed with no P1 or P2 at `3433bf8`;
S1–S5 are implemented (2026-10-10) and code-reviewed with no P1 or P2 at `6f16373`; S6 and S8
are implemented (2026-10-11) and their batch is code-reviewed with no P1 or P2 at
`cddad9d` (2026-10-12; the later increment that narrowed
their claims was reviewed with no P1 or P2, 2026-10-12); S7 is implemented for Mode C only
(2026-10-11) and code-reviewed with no P1 or P2 at `7327798` (2026-10-12) as that declared scope,
and measurement runs that include Mode A were not migrated: since the owner's clarification of
2026-10-12 they were deleted by the Mode A/B retirement's R2 (2026-10-12; code-reviewed with no P1
or P2 in `1d0b5a6..06fd0f1`, 2026-10-12); S9 is implemented (2026-10-12) and code-reviewed with no P1
or P2 at `f481882` (2026-10-12; its "As implemented" note below); S10 ran on 2026-10-12 with its re-reviewed
tools, its evidence awaiting review, and nothing else here
is implemented.

**Revisions of 2026-10-09 and 2026-10-10.** They follow the reviewer's plan review of `4ecd2c2`
(one P2, no P1) and its two later reviews, and the owner's decisions of 2026-10-09 and
2026-10-10, recorded under "Decisions" below.
- **The P2:** S10 asked for capacity readings of measurement Modes A and B, which cannot run on
  the designated checkpoint: its writer writes neither key Mode A indexes
  (`src/training/callbacks.py:296-327`), and S10's pre-flight lists its keys. S10 is now an
  executable matrix.
- **The calibration launcher:** checked as the owner asked, and found to have no necessary user.
  It is removed in a new step, S0.
- **The re-review of `0708789`** closed that P2 and found one more: S10 inferred a kept buffer
  from resident memory. Release is now shown by checks on the readers (S2–S4), and S10 compares
  and explains resident memory instead. Its non-blocking notes are applied: S0's claim is limited
  to the checkpoints checked, S0 ends with a final search for present-tense references, the
  measuring tools are bounded, and question 6 listed what each option changed. The re-review
  accepted S0, M2.1 decisions 1–5 as recorded, the three exceptions, the M2.4 and BACKLOG
  relations, and the republish rule with its SP binding supplement. On question 6 it recommended
  (a), removing `scripts/evaluate_model.py` with its pin test, which the owner chose
  (decision 6).
- **The incremental review of `7e1af96`** (2026-10-10) closed the re-review's P2 and found no new P1
  or P2. It accepted the plan as revised by then, at plan level, which is not acceptance of the
  readers, the release check, the measuring tools or any reading, none of which exists yet. Its
  notes for the release check's implementation are in S2. The text changed since, for decisions 6
  and 7 below, was reviewed next: S0's additions (decision 3 lists them), the amended "Capacity"
  rule, the memory sizes, and S10's subject, pre-flight and acceptance heading.
- **The review of the decision record** (`7e1af96..45d9dea`, 2026-10-10) found no P1 or P2. It
  accepted decisions 6 and 7 and the S0 and S10 text they bring, at plan level. That is not
  acceptance of S0's change, the readers, the release check, the measuring tools or any reading.
  Its notes for S0's review: the old launcher, the evaluator and the pin test are deleted; their
  surviving callers are moved; history no longer serves as a current deletion gate; and item 7a
  reuses the differential core rather than a second evaluation engine. "Frozen" binds only the
  file S0 deletes; it is no reason for a new reader, a fallback, or a new entry point shaped
  around the retired evaluator.
- **The owner's decisions of 2026-10-10** settle the last two choices: `scripts/evaluate_model.py`
  is removed with its pin test in S0, and S10's capacity acceptance is limited to its designated
  subject (decisions 6 and 7 below). Neither is asked again unless a new dependency or
  measurement fact appears.
- **The code review of S0** (`45d9dea..3433bf8`, 2026-10-10) found no P1 or P2 and accepted S0 at
  code-review level. Its two non-blocking notes are applied: `build_legacy_mode_a_model`'s
  docstring now says only that S0 did not change how it builds the model and that replacing or
  deleting it follows BACKLOG item 9's verification; and the worker-stream test's loader has a
  finite `timeout`, so a hung worker fails rather than stalls. Its focus for S1–S5: bytes and
  digest from the same source; the real reader's buffer lifecycle; no identity made up when a
  read fails; and no new branch, metadata, schema or fallback for `build_legacy_mode_a_model`.
- **S1–S5 as implemented (2026-10-10),** in `eecd803`, `2faaaa4`, `350cfee`, `e4cf737` and
  `f581f9e`, then corrected after an internal adversarial review: five lenses, each finding
  checked by a skeptic, and a completeness critic over the code map. It found no defect in what
  the readers parse or record. The corrections:
  - `verify_graph_reads` refused a missing tensor role but accepted a role given as `None`; it
    now refuses both;
  - the read forms raised a bare `AttributeError` on a manifest whose `artifacts`, `realised` or
    `allocation` is not an object; they now refuse naming the manifest, and so do the path forms,
    which share those checks;
  - some refusals did not name the file they were about; they do now;
  - `verify_cohort_reads` reported its scope in the generator's order where the path form keeps
    the caller's; both keep the caller's;
  - the test double's wrapper could stay installed in a module first imported while it was
    active; it now goes inert when the test's patches are undone;
  - tests were added where an option, a refusal or a check was claimed but not shown;
  - text that claimed more than the code or the check does was narrowed (the release check's
    limits, the open counter's, `GraphRead`'s pairing), and text S4 made false was corrected.

  The decisions S1–S5 took within the plan are recorded under their steps: the open counter in
  S1, the read forms' names and the `kg` pairing in S5. Neither changes a requirement.
- **The code review of S1–S5** (`e3eefee..6f16373`, 2026-10-10) found no P1 or P2 and accepted
  S1–S5 as the shared base for the entry-point migrations, which are reviewed in batches. That is
  not acceptance of S6–S10, not a claim that any existing checkpoint meets M2's provenance
  contract, and not the homelab capacity acceptance. It confirmed four limits the text already
  states: `GraphRead` keeps the graph and its identity together by convention, not by
  enforcement, so an entry point passes it whole; several files read in turn are not an atomic
  snapshot, so what is guaranteed is that the bytes parsed are the bytes recorded, with the
  manifest refusing a mismatch; the release check shows the raw read buffer is dropped, not that
  decoded text, tensor storage, the allocator or resident memory fall back, which S10 measures
  at the real entry points; and `read_graph_artifacts` still omits a missing file from its result,
  so a reader's result is not a verified workspace until `verify_graph_reads` has accepted it.
  Its focus for the migrations:
  - S6–S8 show at the real entry points that the manifest is read once; that the checkpoint
    record takes only the reader's result; that the graph and its identity are never re-paired;
    that a failure publishes nothing and leaves no success artifact; and that no path-form digest
    flows back into a record;
  - a reader's result is never handed back to a path verifier to be read again, and the path and
    read forms are not kept as two permanent options;
  - S9 removes the migration interfaces, and only S10 concludes on capacity, for its subject.
- **The code review at `1d0b5a6`** (2026-10-12) found no P1 or P2 in the increment that narrowed
  S6/S8's claims (`cddad9d..3e8e559`), in S7 for Mode C only (`3d53c9b..7327798`), reviewed as the
  declared C-only scope (code-review level) and not as every mode migrated, and in M3a's seed ordering
  (`7327798..290c03a`). The first is a text increment, not a review of the S6/S8 batch
  (`8421b8c..cddad9d`), which then still awaited code review (since code-reviewed with no P1 or P2,
  2026-10-12; below). The same review found no P1 or P2 in the
  retirement's R1 (`3e8e559..3d53c9b`), which is not part of this plan. M2.1 is not complete.
- **The owner's clarification of 2026-10-12.** The owner reported on 2026-10-12 that the
  institution had already authorised retiring the old pipeline's Mode A/B comparison tools, and
  that the AMP question the owner and the institution want answered is a training study, not the
  scoring comparison the differential recorded. The retirement assessment's §8 items 4 and 7
  therefore no longer hold the retirement's R2 back: S7's Mode A/B items will never be built, runs that include
  Mode A are to be deleted by R2 rather than migrated, and S9's measurement part follows R2. R2 was
  implemented the same day (then awaiting code review; since code-reviewed with no P1 or P2 in
  `1d0b5a6..06fd0f1`, 2026-10-12): runs that include Mode A are deleted, and
  `measure_scorer.py` no longer calls a path form.
- **The reviewer's convergence instruction (2026-10-12), adopted by the owner.** Finish S9 and S10
  to close M2.1, then M2.2–M2.6, so the contract is enforced: producer and consumer delivered
  together, with a formal way to produce acceptable new models (`train_model.py` →
  `ModelCheckpoint`) and the deployment inventory. HGT (BACKLOG item 20), AMP (item 21), item 14 and
  M3 stay independent. No new framework, registry, signing or second reader; a capacity shortfall
  is fixed in the real reader, never by a CPU fallback, a second reader, `malloc_trim`, forced
  collection or `empty_cache`. Each status here says which of planned, implemented, code-reviewed
  (with its range) and accepted in the designated environment it is, and one never stands for
  another. Code review version for that round: `06fd0f1`; awaiting code review: the S6/S8 batch
  (`8421b8c..cddad9d`) and `1d0b5a6..06fd0f1` (HGT P2 `418b37e`, the record `84d69fb`, HGT Edit 1
  `8eae614`, telemetry `a535ae2`, R2 `e210b12`, item 8 `06fd0f1`). Both have since been
  code-reviewed with no P1 or P2 (2026-10-12; the next bullet).
- **The code review at `c1b8a12`** (2026-10-12), read-only, judged five ranges apart, with
  targeted CPU tests in an isolated copy. Each is a code review of its range, not an acceptance:
  - **the S6/S8 batch** (`8421b8c..cddad9d`; serving and training read each recorded input once):
    no new P1 or P2; 212 targeted CPU tests passed. The reviewer notes that M2.3's parent checks
    are not in this range: the resume parent is still read where it is restored, and a missing
    requested parent still warns and the run trains from scratch; both stay with M2.3;
  - **HGT, telemetry, R2 and item 8** (`1d0b5a6..06fd0f1`, six commits): no new P1 or P2; 179
    targeted CPU tests passed, 1 skipped. The commits: the HGT proposal states the kernel as
    observed, not inferred (`418b37e`), so the earlier P2 on that proposal can be closed; the
    record of the owner's clarification of 2026-10-12 (`84d69fb`); HGT Edit 1 (`8eae614`), at CPU
    code-review level only, not a GB10 performance or CUDA numeric acceptance; the dashboard's
    telemetry off (`a535ae2`, Gradio's `analytics_enabled=False`), which does not mean the project
    never makes other network requests; R2 (`e210b12`), Modes A and B deleted with the retirement
    assessment's §8 items 5 and 9 (ledger schema 2); and item 8 (`06fd0f1`), `read_samples`
    refusing non-integer and boolean ids;
  - **the status wording** (`06fd0f1..02a6954`; planned, implemented, reviewed, accepted): no new
    P1 or P2;
  - **S9** (`02a6954..f481882`): no new P1 or P2 in the product code; 554 targeted CPU tests
    passed, and import-linter reported 4 kept. S10's tools, delivered with it and kept outside the
    repository as planned, had three P2: the comparison prints "met" for evidence that fails the
    declared acceptance conditions; the sampler stops only its direct child, so SP workers can be
    left running; and the driver can reuse an old output directory, so a failed run mixes with an
    earlier run's evidence. The tools were fixed afterwards, outside the repository (S10, "The
    tools"), and await re-review (since passed, draft 3; below);
  - **S10's plan amendments** (`f481882..c1b8a12`): no new P1 or P2. The plan is acceptable; the
    tools do not yet meet all its requirements.

  The whole suite at `c1b8a12`, in an isolated copy: 2615 passed, 9 skipped and 1 failed, only
  because a `git archive` copy has no git metadata; with temporary metadata that test and the docs
  index passed (61 passed). **What this round does not establish:** S10's tools are not passed;
  S10 has not run and capacity is not accepted; M2.1 is not complete (S10 is outstanding), and
  neither is M2; nothing is accepted on the homelab. HGT Edit 2, the AMP study and item 14 are not
  part of it.
- **The re-review at `3c9563f`** (2026-10-12), read-only, of `c1b8a12..3c9563f` (docs only) and of
  S10's fixed tools: no new P1 or P2 in the documents; the three P2 on the tools are closed. It
  found one more P2 in the tools: a warm pre-read that failed did not stop its row, which then ran
  and was recorded as a warm reading. Fixed the same day (S10, "The tools"); the fixed driver and
  its tests await a narrow re-review. S10 has not run, and M2.1 and M2 are not complete.
- **The incremental review of draft 3 at `343e90e`** (2026-10-12), read-only, of `3c9563f..343e90e`
  (docs only) and of the tools' draft 3 (the driver and its tests changed; the probe, the sampler
  and the comparison unchanged): no P1 or P2; the pre-read P2 is closed. The tools may be used for
  S10's measurement. This is not acceptance of S10's capacity, of M2.1 or of M2. The "after"
  commit is then fixed at `343e90e` (S10).
- **S10 ran on the homelab (2026-10-12)**, at `90a668f` and `343e90e` in one boot, with draft 3 of
  the tools: the evidence and the explanation of every reading are in
  [`capacity-m2.1-2026-10-12/`](scorer-measurement/capacity-m2.1-2026-10-12/README.md), awaiting review. Not acceptance of S10's capacity, of M2.1
  or of M2.
- **The evidence review of `e0cfb68..968d5e4`** (2026-10-13), read-only: no P1; one P2, the two
  `driver.log` files missing from the commit (the repository ignores `*.log`), added at
  `40fcf0b` and checked from a clean tree; the reload's causal wording narrowed. It recommends a
  limited diagnostic of consecutive reloads before acceptance 2 is closed, which the owner chose
  the same day (S10, "As run"). Not acceptance of S10's capacity, of M2.1 or of M2.
- **The review of `40fcf0b..7b026c9` and of the diagnostic's draft 5** (2026-10-13), read-only:
  no P1; no P1 or P2 in the tool or the documents; two P2 in the runbook's reading rules, fixed in
  the runbook only (tool and tests unchanged). Not acceptance of anything.
- **The diagnostic ran on the homelab (2026-10-13)**, at both commits in one boot, and the
  reviewer reviewed its results the same day: [`reload-diag-2026-10-13/`](scorer-measurement/reload-diag-2026-10-13/README.md). Not acceptance
  of S10's capacity, of M2.1 or of M2.
- **The reviewer's closing review (2026-10-13)**, read-only: recommends accepting S10's capacity on
  its designated subject with the remaining uncertainty explicit, and closing M2.1's technical
  work; `src/`, `scripts/` and `tests/` at `6e17360` identical to `343e90e`; the evidence's
  `SHA256SUMS` pass from a clean copy. **Adopted by the owner on 2026-10-13** ([the capacity README §6.2](scorer-measurement/capacity-m2.1-2026-10-12/README.md#62-acceptance-2026-10-13)).
  Not a merge; M2 is not complete.
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
- **One open and one full read,** with the SHA-256 taken over that same bytes object. The read
  is alone on its line (S1), because S2's release check filters on it.
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
- **Memory:** one file at a time, with each buffer released before the next read (shown by S2's
  release check, which runs at each `read_once` call a reader makes and after the reader returns;
  not by RSS). Each file costs about one transient copy of its size while it is parsed:
  - about 29 MB of tensor payload for `node_features.pt`, computed from the recorded shapes (the
    subject's file is 29,308,567 bytes, decision 7);
  - 48 MB for the designated checkpoint;
  - a JSON file's bytes and its decoded text coexist, so `kg.json` (209,928,843 bytes on the
    subject, decision 7) is likely the largest new transient, at serving and in the SP producer.
    **S10 showed this premise wrong (2026-10-12):** at `90a668f` `load_json` already held the
    bytes and the decoded text together (a text-mode `json.load` reads the whole file and decodes
    it while the bytes are alive), so `kg.json`'s transient is not new. The new transients are
    the whole-file buffers of the two tensor files and the checkpoint ([evidence](scorer-measurement/capacity-m2.1-2026-10-12/README.md), §4.3).

  These are expectations to explain the readings with (S10), not bounds.

**Two constraints the plan above does not mention.**
- **`read_samples`' new fields are opt-in** (`training_fields=True`). Measurement's samples carry
  three fields (`src/kg/storage/file_storage.py:94-100`), as the frozen evaluator's did
  (`scripts/evaluate_model.py:185-191` at `7dab728`), and `gene_ids` become subgraph seeds
  (`src/kg/data_loader.py:675-676`, `:947-950`). Adding them by default would have changed
  measurement's Modes A and B. Since the retirement's R2 (2026-10-12) the reason is that training
  passes its two extra fields explicitly and measurement (Mode C) reads neither; the interface is
  kept (retirement assessment §8 item 6).
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

0. **S0 — remove the retired calibration launcher and the frozen evaluator.** The owner's rule
   for the launcher (decision 3, 2026-10-09): it is removed if nothing necessary uses it, and
   migrated minimally only if it must stay, so that it does not stay behind as a dead island.
   `scripts/evaluate_model.py`, which the launcher's removal leaves with no runner, goes with it
   by the owner's separate decision (decision 6, 2026-10-10). The check, at `4ecd2c2`:
   - **Who depends on `scripts/calibrate_mode_a.py`.**
     - There is no CI. No Makefile target, launcher, deploy script or systemd unit names it.
     - No module under `src/` and no other script imports or runs it.
     - No runbook or deployment guide instructs running it. The documents that name it:
       - as a planned flow, the draft `PLAN_TEST_RESULTS.md:745` and `:766` (dropped below);
       - `scorer-measurement/README.md:33` (marked SUPERSEDED), `:53-55` (present tense, updated
         below) and `:103-105` (the decision amended below);
       - as history, `PLAN_B03.md:23`, `:103-105` and `:253-255`, and
         `PLAN_CONFIGURABILITY_AND_PROVENANCE.md:235`;
       - BACKLOG 7a, 19.19 and §5.0's file table.
     - Only its own tests import it:
       - `tests/integration/test_seeding_bootstrap.py`;
       - in `tests/integration/test_legacy_equivalence.py`: the launcher fixture (`:41-73`); the
         tests that read it, including the verdict-schema and `compare()` digest tests, which
         import it at `:223`, `:243` and `:263` (`:79-272`); and the split-default test
         (`:431-449`, import at `:440`);
       - a seed-check class in `tests/unit/test_measurement_mode_a.py:890-939`;
       - one entry in `tests/unit/test_split_caveat.py:39`.
   - **It cannot run on the checkpoints checked, or on any the current writer produces.**
     - Its harness call reaches `build_legacy_mode_a_model`, which indexes `metadata` and
       `in_channels_dict` and raises `KeyError` without them (`measure_scorer.py:251-252`). Its
       oracle, `scripts/evaluate_model.py`, falls back to three node types, two edge types and
       width 256, and so builds a structurally wrong model (BACKLOG §3.1).
     - The current writer, `ModelCheckpoint._save_checkpoint` (`src/training/callbacks.py:296-327`),
       writes neither key. The scanned family, 15 checkpoints (BACKLOG M1;
       `EVIDENCE_M1_M3_hgt.json`, `EVIDENCE_M1_M3_gat.json`), carries neither. The designated
       checkpoint came from the same writer, and the reviewer reports it carries neither (plan
       review of `4ecd2c2`); S10's pre-flight records its keys.
     - Its tests pass only because the fixture writes those keys
       (`tests/fixtures/synthetic_workspace.py:159-160`).
     - Not checked: no historical audit was run (BACKLOG M1, §6), and the homelab's other
       checkpoint files were compared by digest only (`PLAN_B04_PRODUCTIONISATION.md` §7.1.1).
       S0 does not rest on them: it removes the launcher because §3.1.2 retired the parity it
       runs, not because no checkpoint could carry the keys.
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
   - **Deleted:** `scripts/calibrate_mode_a.py` and `tests/integration/test_seeding_bootstrap.py`;
     and, by decision 6, `scripts/evaluate_model.py` and `tests/unit/test_frozen_evaluator.py`.
   - **Tests:**
     - in `test_legacy_equivalence.py`, the launcher fixture (`:41-73`) and the tests at
       `:79-272` go. Their measurement-only assertions move onto a direct `measure_scorer` run
       with `--num-workers 4` and `--predictions-output`, which keeps those two paths covered end
       to end. The split-default test (`:431-449`) keeps only its `measure_scorer` half;
     - `test_measurement_mode_a.py:890-939` goes;
     - `test_split_caveat.py` keeps only `measure_scorer` as an entry point.
     - **Kept from the deleted tests, found in S0's review:** three assertions guarded live code,
       not the launcher, and stay on the surviving entry point. `test_seeding_bootstrap.py`'s
       worker half (negatives drawn in worker processes follow the parent seed, and the worker
       count is part of the stream) moves to `tests/unit/test_measurement_mode_a.py`, without
       the launcher's bootstrap; so does the seed validator's acceptance of both ends of its
       domain. `test_legacy_equivalence.py` checks that the manifest's artifact digests are the
       digests of the files the command was given, hashing each file itself; the launcher's
       verdict had compared them only with its own call to `artifact_digests`.
   - **Text the launcher's removal makes false.** The launcher
     was the only thing that compared Mode A with the frozen evaluator; after S0 nothing does.
     Text that names the launcher in the present tense, says that an artifact, a stream or a
     default exists for that comparison, or lists "calibration" as an entry point, is reworded to
     the rule it keeps and the reason that remains. Nothing it describes changes for that reason:
     the predictions shape, the legacy order, `--num-workers 4` and the default mode stay.
     - **Code:** `measure_scorer.py:14-16`, `:24-33`, `:96-98`, `:405-410`, `:415`, `:434-436`,
       `:658-661`, `:682` (a log line) and `:705-706` (printed text);
       `src/evaluation/measurement.py:8-9`, `:159-161`, `:167-169`, `:176-177`, `:255-259`,
       `:678-692`, `:706-708` and `:948-950`; `src/evaluation/cohort.py:232-234`;
       `scripts/audit_split_overlap.py:112-114`;
     - **Tests:** `test_legacy_equivalence.py`'s module docstring (`:1-25`), `:278-285` (that
       test's name and docstring) and `:316`; `tests/unit/test_measurement_mode_a.py:543-549`;
       `tests/unit/test_measurement_modes_bc.py:131`, `:154-155`;
       `tests/unit/test_measurement_ranking.py:125-128`, `:138`;
       `tests/fixtures/synthetic_workspace.py:110`; `tests/unit/test_split_caveat.py:4-5`;
     - **Docs:** `scorer-measurement/README.md:33`, `:53-55`; `EVALUATION_COHORTS.md:473`,
       `:918-919`, `:930-933`; BACKLOG M9's "on both entry points" (`:565-566`).
       `PLAN_TEST_RESULTS.md:745` drops the launcher, and `:763-775` goes: the oracle's named
       exception closes with the evaluator (decision 6), and the staging design after it existed
       so the launcher and `test_legacy_equivalence.py` could drive the oracle on supplied
       cohorts. The line citations into `test_legacy_equivalence.py` at `:749-751` are
       updated.

     Left as history: `PLAN_B03.md:23`, `:103-105` and `:253-255`,
     `PLAN_CONFIGURABILITY_AND_PROVENANCE.md:235`,
     the past-tense notes in `src/evaluation/caveats.py:5-6` and `src/utils/fingerprint.py:64-68`,
     and `tests/unit/test_split_caveat.py:17`, which is past tense.
   - **Lifecycle statements that tie surviving code to the frozen evaluator.** §3.1.2 retired
     the parity they wait for, so each is dropped or tied to item 9:
     - `legacy_ranking`'s "deletion date" (`src/evaluation/measurement.py:179-182`) is dropped:
       the trainer makes the same sort (`scorer-measurement/README.md:90`);
     - `src/evaluation/measurement.py:1044-1045` is dropped: the clamp and the A/B traversal are
       retained (`scorer-measurement/README.md:71-74`, `:89`);
     - `measure_scorer.py:5-6` loses its condition, and `:235-237` is tied to item 9.

     S0 deletes the evaluator (decision 6), so it also rewords the statements in code S7 later
     rewrites that tie a function's lifetime to the evaluator: in `load_legacy_mode_a_inputs`'
     docstring, `measure_scorer.py:205-217`, with its count at `:222` ("those five");
     `build_legacy_mode_a_model`'s `:240`; the dispatch comment `:561-565`;
     `test_legacy_equivalence.py:337-343`; and the paragraph in `src/kg/storage/__init__.py:35-37`.
     What these say of `build_legacy_mode_a_model` is tied to item 9; what they say of
     `load_legacy_mode_a_inputs` is tied to its deletion in S7, not to item 9. S7 then deletes
     `load_legacy_mode_a_inputs` (`:202-227`), with its docstring and that paragraph, and rewrites
     the test and the comment for the shared read. As implemented (2026-10-11), S7 covers Mode C
     only: the function, its docstring and that paragraph stay, reworded to say that runs which
     include Mode A are not migrated (S7, "As implemented").
   - **A final search before S0 lands.** The repository is searched again, case-insensitively, for
     `calibrate_mode_a`, "launcher", "calibration", "frozen evaluator", "frozen oracle" and
     "oracle", and for `evaluate_model` and "evaluator", which find text that names the
     evaluator without those words. A hit is reworded, or recorded as history, in S0's change
     only if S0 makes it false, that is, if it:
     - names the calibration launcher in the present tense;
     - says that something exists for the comparison of Mode A with the frozen evaluator, or
       that the comparison still runs;
     - ties the lifetime of surviving code to that parity;
     - or cites `scripts/evaluate_model.py` without a fixed revision, or names it as a live file.

     The other hits stay: the application launcher (`scripts/launch/`), the differential
     calibration, Mode A's retained semantics, and item 9's surface, such as
     `src/evaluation/measurement.py:557-559`, which `scorer-measurement/README.md:106-109` leaves to
     item 9's rename. Hits in code S7 rewrites belong to S7, except those that S0's lists or
     decision 6's name: S0 changes those first, even inside a function S7 later rewrites, such as
     `measure_scorer.py:203`, `:205-217`, `:222`, `:234-237`, `:240`, `:561-565`, `:658-661` and
     `:682`. The lists above and decision 6's, with its two further corrections, are the hits S0
     changes as found at `0708789`. The search is what makes them complete.
   - **Decisions amended when S0 lands** (amended in S0's change, 2026-10-10), each with the
     owner's rule and date:
     - `scorer-measurement/README.md:103-105` ("rewritten, not deleted") is reversed;
     - **the deletion gate gains one exception,** in BACKLOG §5 ("Item 9 waits for everything")
       and its restatement in §5.0 (`:835-837`), in `scorer-measurement/README.md:78-79` and its
       removal order's step 5 (`:122`), and in item 9's row. The exception: the oracle-parity
       tests in `test_legacy_equivalence.py`, which the README counts as oracle-only (`:95-97`),
       go with the launcher before 1d's institutional run, and so, by decision 6, do
       `scripts/evaluate_model.py` and its pin test. The oracle-parity tests check the
       frozen-evaluator parity that §3.1.2 retired as an acceptance; `scripts/evaluate_model.py`
       is that parity's reference, and its pin test keeps the reference's bytes unchanged. The
       launcher could not run that parity on the
       scanned family, or on a checkpoint the current writer produces (S0's check above). The
       adopted acceptance, the differential calibration, is untouched;
     - `scorer-measurement/README.md:95-97` drops `scripts/evaluate_model.py`, and stops listing
       `test_legacy_equivalence.py` as a whole: after S0 its remaining tests are the
       `measure_scorer` ladder, Mode C and `--split` tests, not oracle parity, so step 8 does not
       delete them. Step 8 (`:125`) notes what went with S0;
     - BACKLOG §5.0's file table drops the launcher.

     S0's change updates every document listed here in the same change. Until it lands, the gate
     stands as written.
   - **Item 7a's entry point is recorded as a backlog item.** Its runner is new code built on
     `compare_trainer_against_mode_a`. It records, beside its result, the digests of the bytes it
     read (through M2.1's readers), the software revision and the deployment host. History stays
     in git. Superseded by the owner's clarification of 2026-10-12: 7a is cancelled, so this runner
     will not be written (BACKLOG item 7a).
   - **`scripts/evaluate_model.py` and its pin test** go by decision 6, which lists the text
     that changes with them.
1. **S1 — the primitive.** `read_once`, `FileRead` and `ReadIdentity`; no caller changes.
   `read_once` opens the file on one line and reads it with `handle.read()` on a line that does
   nothing else. The digest, and anything else its `ReadIdentity` keeps, is made on other lines,
   because S2's release check filters on the read's line.
   - **Tests:** the digest is the digest of the bytes returned; the file is opened once; a
     missing file raises; replacing the file afterwards, by atomic rename or in-place rewrite,
     changes neither the bytes nor the digest returned; with `tracemalloc` filtered on the read's
     line (`all_frames=True`), a kept `ReadIdentity` holds no live allocation from it.
2. **S2 — test infrastructure.** No production change.
   - **Parseable fixtures:** each workspace's `kg.json`, `num_nodes.json` and tensors are real
     files whose content differs per workspace. Tests that verify only digests gain
     `importorskip("torch")`.
   - **Training samples** that carry both `candidate_disease_ids` and `gene_ids`.
   - **The replacement double** (`tests/fixtures/replacement.py`): it wraps `read_once` and
     replaces a target file once, right after a named read returns. It does so by atomic rename
     (a new inode) or by in-place rewrite (the same inode). It can also republish a manifest that
     binds the replacement.
   - **An open counter per path,** for the "read once" assertions. *As implemented:* it landed
     with S1, whose own test needs it (`tests/fixtures/opens.py`). It sees opens through Python's
     io and os layers, so not an open made inside a C extension; torch readers are held to the
     hashed buffer by the release check's tracked-`BytesIO` rule instead.
   - **A release check** (`tests/fixtures/release.py`), on fixture-sized files on CPU, through the
     same `read_once` wrapper as the replacement double.
     - **When it runs:** on entry to each `read_once` call a reader makes, covering the reads
       before it, and again after the reader returns. Each time:
       - a `tracemalloc` snapshot holds no live allocation whose traceback runs through an
         earlier `read_once` read. This catches the raw buffer whoever holds it: a local still
         bound inside a reader of several files (`read_graph_artifacts` reads three,
         `file_storage.py:48-56`, and a loop that rebinds one name still holds the previous
         buffer during the next read), the return value, a closure or long-lived state. `bytes`
         cannot be weakly referenced, so the buffer is checked through its allocation;
       - a weak reference to each earlier `BytesIO` wrapper is dead.
     - **Its settings let it see the buffer.**
       - `tracemalloc` keeps enough frames to reach `read_once`. The filter names `read_once`'s
         file and the line of its read, and matches it in any frame (`all_frames=True`); the
         default matches only the most recent frame.
       - `read_once` opens the file on an earlier line and reads it with `handle.read()` on a line
         that does nothing else. The digest, and anything else a `ReadIdentity` keeps, is made on
         other lines, so a correct reader's identities never match the filter.
       - The `BytesIO` weak references come from replacing the reader module's `BytesIO` with a
         tracking subclass. For each torch file a reader reads, the check requires at least one
         tracked `BytesIO`, so a reader that builds its wrapper some other way fails instead of
         passing unseen.
     - **It is shown to fail.** A self-test, on the real `read_once` and a non-empty file,
       requires the check to fail on a reader that keeps the bytes on a module global, in a
       closure, on its result or across its next read. It must also fail on a reader that keeps
       a `BytesIO` holding its own copy of the bytes (as after `getbuffer()`), which only the weak
       reference catches; a `BytesIO` that still shares the bytes is caught through them.
     - **What it does not see:** a copy the reader makes into another object, such as a
       `bytearray` or the decoded text of a JSON file, and torch's tensor storages. Those are
       parsed results or copies, not the raw buffer; the readers' results are listed in S10's
       readings, and S10 explains peaks against them. Nor does it see a buffer kept between its
       parse and the reader's return, since it checks at the next read and after the return
       (found in S1–S5's review); that the readers drop the bytes before parsing is their code,
       not something the check shows.
     - **What its implementation's review checks** (the reviewer, 2026-10-10): that it is wired
       to each real reader, that tracing starts before the first read, that the self-test's
       negative controls do fail, and that the wrapper itself keeps no strong reference to a
       buffer.

     It is a test, not a capacity reading, and it never runs inside S10's measured runs.
3. **S3 — `read_json` and the SP producer.** `compute_shortest_paths.py` records the digest of
   the `kg.json` bytes it traversed, from its one read. The end-of-run comparison (`:472-479`)
   stays a warning, as the plan above says. `read_json` passes S2's release check.
4. **S4 — the shared readers.** The other four readers above (`read_json` is S3's). Callers only
   unpack the new return values; no recorded digest changes yet. Each reader passes S2's release
   check, and its result carries a `ReadIdentity`, never a `FileRead`.
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
   - **The `kg.json` identification check** (M2.1 decision 1) is one narrow helper. It hashes the
     file and compares it with the run's one `ManifestRead`; it never reads the manifest itself.
     It is used only before runs that do not parse `kg.json`, and it is not offered as a second
     verification API to consumers that parse their inputs.
   - **The old path forms are a migration aid only.** They remain until S9, for callers not yet
     moved, and S9 deletes them.
   - **As implemented (2026-10-10).**
     - The read forms are named apart from the path forms, rather than overloading them by
       argument type, so that S9 can delete the path forms and pin their absence by name:
       `verify_graph_reads` and `verify_graph_source_read` (`src/kg/artifacts.py`),
       `verify_cohort_reads` (`src/evaluation/cohort.py`), and the identification helper
       `check_unparsed_kg_json`.
     - `verify_graph_reads` returns the manifest's `kg` digest and compares it only when `kg` is
       among the reads. Each entry point therefore pairs it with `verify_graph_source_read`
       (serving) or `check_unparsed_kg_json` (training, measurement), and S6–S8's tests replace
       `kg.json` alone and expect a refusal.
     - A refusal about a file names the file and the manifest. One about the manifest itself, or
       about the caller's scope, names the manifest or the scope. `check_unparsed_kg_json` is the
       one read form that opens a file: `kg.json`, to hash it.
6. **S6 — serving.**
   - `build_pipeline` reads `kg.json` with `read_json` and passes the `GraphRead`.
   - The pipeline reads the manifest, the graph tensors and the checkpoint once each, and
     verifies from those reads. After verification nothing is hashed again by path.
   - It keeps all four bound digests, the opening for M2.4. Today only `bound["kg"]` is kept
     (`src/inference/pipeline.py:532`).
   - **The model is built from the final read.** `read_checkpoint`'s parsed object and digest are
     what the build uses and what M2.4 will check, never the data from a selection scan
     (M2.1 decision 4).
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
   - **As implemented (2026-10-11).**
     - `DiagnosisPipeline` and `create_diagnosis_pipeline` take `graph=` (the `GraphRead`) in
       place of `kg_path`, which is removed with no alias. `graph=` and the in-memory `kg=` are
       exclusive, so a graph and an identity cannot be given apart. The pipeline keeps all four
       bound digests (`_bound_digests`), the graph's identity (`_graph_identity`) and the
       checkpoint's (`_checkpoint_identity`), for M2.4 and S9's dynamic pin; none is published,
       since the served identity is M3c's.
     - **The checkpoint is read after the graph check**, inside `_load_model_from_checkpoint`,
       with this site's own options (`map_location="cpu"`, `weights_only=False`). M2.1 checks
       nothing about the checkpoint, so the pipeline reads none for a refused workspace. When a
       reload selects a checkpoint itself (no `checkpoint_path`), the route's selection-time reads
       (M2.1 decision 4; BACKLOG 19.1 and 19.2) still come before that refusal.
     - **The shared readers name a file that was read but does not parse** (`f3136d3`). Serving
       and training are the entry points M2.1 moves to parse before they compare (measurement
       already parsed first, so its refusals of a junk file named nothing either and now name the
       file too); without this, a junk tensor or JSON file failed with the parser's message,
       which names nothing, where the path check used to name it. Two refinements came with the
       internal review: a memory shortfall while loading tensors is passed through unchanged,
       not reported as an unparseable file (in the three forms the reader recognises:
       `MemoryError`, the CPU allocator's `RuntimeError` and `torch.OutOfMemoryError`), and a
       JSON file that parses but holds entries that are not sample rows (an object with keys,
       `null`, a number) is refused under its name. An empty object or string still parses as
       zero samples; the manifest's digest comparison refuses that for a generated cohort, and
       training refuses an empty training split.
     - Text S6 made false is corrected with it: the `file_storage` copy lists, `graph.py`'s
       export docstring, `artifacts.py`'s provenance note, `EVALUATION_COHORTS.md`'s binding
       paragraph and BACKLOG 19.20's citation. `test_kg_provenance.py`'s serving pin, missing
       from the list above, now runs the pipeline's own sequence.
     - Acceptance tests (CPU, development container; not a homelab acceptance, which is S10's):
       `tests/unit/test_serving_reads_once.py`.
7. **S7 — measurement.**
   - **Mode A/B-specific items deferred (owner, 2026-10-10)** pending the Mode A/B retirement
     assessment (`scorer-measurement/ASSESSMENT_MODE_AB_RETIREMENT.md`, §1.3): "Mode A uses the
     run's single read", the C-only test's rewrite around Mode A, and "Modes A and B keep their
     shared-read regression tests". No A/B-specific compatibility design is added meanwhile. The
     mode-agnostic items stand. If S7 runs for Mode C before the retirement is decided, it states
     that Modes A and B are not migrated, M2.1 is not reported complete while they are neither
     migrated nor retired, and no new read path is built for them (the S1–S5 code review).
     **Superseded by the owner's clarification of 2026-10-12:** the owner reported that the
     institution had already authorised retiring the old pipeline's Mode A/B tools, so these
     Mode A/B items will never be built; Modes A and B are to be deleted by the retirement
     assessment's R2 instead of migrated. R2 deleted them (2026-10-12).
   - **Before any model is built or output written:**
     - `resolve_cohort` first;
     - then one read each of: the manifest; the graph tensors; the cohort's samples (default
       fields); for a generated cohort, the other generated split's samples, for its binding and
       the disjointness check; and the checkpoint;
     - then verification from those reads, with the `kg.json` identification check (M2.1
       decision 1).
   - **One digest map** feeds every mode's manifest, replacing the per-mode `artifact_digests`
     (`measure_scorer.py:102-155`). The verifier's bound map is kept instead of discarded
     (`:128`). The manifest's `kg` role is the manifest-bound source graph (M2.1 decision 2).
   - **Mode A uses the run's single read.** *Never built; retired by the retirement's R2 (2026-10-12).*
     - `load_legacy_mode_a_inputs` is deleted, and its docstring (`:203-224`, whose lifecycle
       paragraph is `:205-208`) with it. It
       already delegates to `file_storage` (`:225-227`), so Mode A reads the same files.
     - `build_legacy_mode_a_model` takes the checkpoint dict from that read instead of loading
       the path (`:248`). Its behaviour is otherwise unchanged. It remains item 9's oracle-only
       surface (BACKLOG item 19), and it still cannot run on a checkpoint the current writer
       produces.
     - `build_legacy_mode_a_model`'s `:240` already names item 9 (S0). The paragraph in
       `src/kg/storage/__init__.py:35-37`, reworded in S0, is removed with
       `load_legacy_mode_a_inputs`.
     - The test that proves a C-only run never reaches the loader is rewritten to prove it never
       builds Mode A's model. Its docstring (`test_legacy_equivalence.py:337-343`) and the dispatch
       comment (`measure_scorer.py:561-565`), which S0 stopped tying to the frozen evaluator, are
       rewritten to match.
   - **What runs where:** on the designated checkpoint only Mode C runs, and only Mode C is to be bound,
     measured and accepted on it (S10). Modes A and B keep their shared-read regression tests on the
     executable fixtures (never written; Modes A and B were deleted by the retirement's
     R2, 2026-10-12). Fixture numbers are never offered as capacity readings.
   - **As implemented (2026-10-11), for Mode C only.**
     - **Scope, under the owner's boundary of 2026-10-11** (the retirement assessment's §8
       item 11: run S7 for Mode C now). A run without Mode A (`--modes C`, the formal Test's
       path) is migrated. Runs that include Mode A (`A`, `A,B`, `A,B,C`) are **not migrated**,
       pending the Mode A/B retirement: they keep `load_legacy_mode_a_inputs`, the checkpoint
       loaded by path and `artifact_digests`, and the Mode C manifest of an `A,B,C` run records
       no `kg`. M2.1 is not complete while they are neither migrated nor retired. Since the owner's
       clarification of 2026-10-12 they will not be migrated: they are to be deleted by the
       retirement assessment's R2, which deleted them on 2026-10-12. The Mode A
       bullets above are not built, no new read path is built for Modes A or B, and there is no
       Mode C-specific entry point: the change is in `scripts/measure_scorer.py`.
     - `read_measurement_inputs` runs right after the device is resolved and the modes parsed:
       `resolve_cohort` first (stat only; for a generated cohort the sibling split too); one read
       each of the manifest (both cohort kinds, for the graph binding), the graph tensors
       (`map_location="cpu"`, as before), the cohort's samples (default fields) and, for a
       generated cohort, the sibling split (checked, not consumed, and released before the
       checkpoint read); then `verify_graph_reads`, `check_unparsed_kg_json` and, for a generated
       cohort, `verify_cohort_reads` over both splits; then the checkpoint (`read_checkpoint`,
       `map_location` the run's device, `weights_only=True`, as before). **The checkpoint is read
       after the workspace checks**, as S6 does and unlike the order listed above: M2.1 checks
       nothing about it, so a refused workspace reads no checkpoint. All of it precedes any
       model build and any output.
     - `measurement_input_digests` builds the record only from those reads: `checkpoint`,
       `samples`, the three tensors, `split_manifest` for a generated cohort only, and `kg`, the
       verifier's bound digest (decision 2). `kg` is **a new role in the measurement record**;
       the semantics digest and the ledger record are unchanged, because `kg` is not in
       `SEMANTIC_ARTIFACT_ROLES`. `build_manifest` takes the record as `digests=`; omitting it
       keeps the path form, which only runs that include Mode A still use, and it goes with
       their migration or retirement (since the owner's clarification of 2026-10-12, their
       retirement in R2) and with S9. The retirement's R2 removed the fallback: `digests` is required. The bound
       map is kept
       (`MeasurementInputs.bound`), the opening for M2.4.
     - **Behaviour changes for C-only runs:** cohort refusals (an absolute or traversing split, a
       wrong kind, a missing file, sibling or manifest) come before any file is opened and before
       the model is built and encoded; an absolute split used to open a file outside the
       workspace first. A missing split's message is `resolve_cohort`'s ("does not exist")
       rather than the reader's listing of what exists, as in S8. The record gains `kg`. Scoring,
       ranking, the seed and the manifest's other fields are unchanged.
     - The `file_sha256` re-export and its `_file_sha256` alias stay, uncalled by a C-only run
       (S9's). S9's deletion of the path forms and its static pin over `measure_scorer.py` wait
       for the Mode A/B decision, because runs that include Mode A still use them. Since the
       owner's clarification of 2026-10-12 that decision is no longer open: the deletion and the
       pin follow R2, which deleted those runs on 2026-10-12.
     - Acceptance tests (CPU, development container; not a homelab acceptance, which is S10's):
       `tests/unit/test_measurement_reads_once.py`. The Mode C unit fixture
       (`tests/unit/test_measurement_mode_c.py`) now reads through `read_measurement_inputs`, and
       `tests/integration/test_measure_scorer_run.py` pins the `kg` role.
     - **The retirement's R2 (2026-10-12), code-reviewed with no P1 or P2 (`1d0b5a6..06fd0f1`, 2026-10-12).** Modes A and B are deleted, with the runs that
       include Mode A, `load_legacy_mode_a_inputs`, the checkpoint loaded by path,
       `artifact_digests` and `--modes`, so every measurement run is the path above and records
       `kg`. With them went the manifest's loader, sampler and seed fields, `--seed` and
       `--num-workers` (the retirement assessment's §8 items 5 and 9). `measure_scorer.py` binds
       no path form at import; it keeps the `file_sha256` re-export, which is S9's.
     - **Found, not changed:** a checkpoint that does not parse is refused with torch's own
       error, which names no file, as before S7 (`read_checkpoint` passes torch's errors through
       by design). Naming it would change the exception a C-only run raises; for the owner to
       schedule.
8. **S8 — training.**
   - **Before the run directories, `config.yaml` and `runtime.json`:** one read of the manifest,
     the graph tensors and both sample files (`training_fields=True`), then verification from
     those reads, with the `kg.json` identification check (M2.1 decision 1). `with_validation` is
     computed there from the parsed validation samples, which is the opening for M2.3.
   - `load_graph_data` and `load_samples` (`train_model.py:398-470`) are deleted, collapsing onto
     `file_storage`.
   - `Trainer.load_checkpoint(path)` becomes `restore_checkpoint(checkpoint)` over a dict already
     read. The resume parent is read once with `read_checkpoint`, and its digest is its role.
   - `training_input_digests` is built only from the reads. The role set is unchanged until M2.2
     adds `kg`, and a `None` digest can no longer be recorded.
   - `TrainerProtocol.load_checkpoint` (`src/core/protocols.py`) follows.
   - **As implemented (2026-10-11).**
     - `read_training_inputs(data_dir)` reads the manifest, the graph tensors and both sample
       files once each and verifies from those reads, with the identification check; it returns
       one `TrainingInputs` value that is carried whole to the record, so the data trained on
       and the digests recorded cannot be paired from different reads. It first calls
       `resolve_cohort` for both splits, which only stats, so a missing split keeps today's
       message. The empty-train refusal moved there, after verification and before anything is
       written.
     - `training_input_digests(inputs, parent=...)` builds the record from the reads only and
       refuses a role with no read, so a `None` digest cannot be recorded. The role set is
       unchanged.
     - *Since M2.3 (2026-10-13) the parent is read and checked at the top of the run
       (`read_resume_parent`), and a missing parent refuses; what follows is S8 as implemented.*
     - **The parent is read where it is restored** (`resolve_resume_checkpoint`), with the site's
       own options (`map_location` the trainer's device, `weights_only=False`). That is after the
       run directories, `config.yaml` and `runtime.json` exist and after the model and the
       Trainer are built, so only the workspace checks above precede the first run artifact.
       Moving that read to the top, with its checks, is M2.3's. A requested parent that is missing
       still warns and the run trains from scratch (M2.3 refuses it); one that vanishes after that
       check raises. **Not done in S8, and not to be read into it:** M2.3's parent checks and
       their placement, and M2.4's comparison of the model with the data it is served or resumed
       on. S8 makes the digests recorded those of the bytes parsed; it does not make the parent
       match the workspace.
     - `Trainer.restore_checkpoint(checkpoint)` replaces `load_checkpoint(path)` with no alias,
       and `TrainerProtocol` follows; `trainer.py` loads nothing itself any more. The seed is not
       moved (M3a).
     - The reader imports sit inside the functions that use them: `train_model.py` applies the
       allocator before importing torch, so module-level imports after it each add an E402.
     - Acceptance tests (CPU, development container; not a homelab acceptance, which is S10's):
       `tests/unit/test_training_reads.py`; the record checked at the end of a real
       run in `tests/integration/test_train_model_run.py`. The path-form poisoning is a fixture,
       `tests/fixtures/path_forms.py`, for S9 to reuse.
9. **S9 — remove the path forms, pin the contract, move the remaining callers.**
   - **Follows the retirement's R2 where measurement is concerned** (S7, as implemented; until the owner's
     clarification of 2026-10-12 this waited for the Mode A/B decision):
     measurement runs that include Mode A still call `artifact_digests`, `verify_graph_artifacts`,
     `verify_generated_cohorts` and `compute_input_digests`, so the deletion below and the static
     pin over `measure_scorer.py` follow their deletion in the retirement assessment's R2. R2 deleted
     them on 2026-10-12: `measure_scorer.py` calls none of the four and binds none at import, so
     for measurement only the `file_sha256` re-export is left for S9. The path forms' remaining
     callers are outside measurement.
   - **Deleted:** the verifiers' path forms. `compute_input_digests` goes once no caller records
     through it.
   - **Tests that import what goes:**
     - `tests/unit/test_training_provenance.py` imports `compute_input_digests` at module level;
     - it and `tests/unit/test_measure_scorer_cli.py`'s
       `test_artifact_digests_identify_content_not_paths` (moved there from
       `test_measurement_mode_a.py` by the retirement's R1; renamed
       `test_the_file_sha256_reexport_identifies_content_not_paths` by the retirement's R2, when
       `artifact_digests` went) pin the `file_sha256` re-export in `scripts/measure_scorer.py`.

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
     - the selection-time checkpoint reads (M2.1 decision 4; BACKLOG item 19);
     - the `kg.json` identification helper, `check_unparsed_kg_json` (M2.1 decision 1);
     - the provenance status readers, which M3b moves onto M2.1's reads: `artifacts.py:278` and
       `:291`, and `src/kg/provenance.py:381`, which hashes the record before `:398` parses it
       through `:275`;
     - the provenance and workspace writers, which are M3b's (`src/kg/provenance.py:233`,
       `src/kg/workspace.py:297-300`).

     Outside the pin: `build_index.py`, and item 19's scripts outside the pin's scope
     (`record_evaluation.py`, `measure_served_pipeline.py`, `audit_split_feasibility.py`,
     `audit_checkpoint_family.py`, `benchmark_sp_lookup.py`, `audit_sp_reachability.py`,
     `build_knowledge_graph.py`, `migrate_checkpoints.py`). Item 19's sites under `src/**` are
     allowlisted above.
   - **A dynamic pin** runs each entry point with `compute_input_digests` and every path hash at a
     recording site returning a poison digest.
     - The `kg.json` identification helper gets the true digest, so each run completes and writes
       its record.
     - No poison may reach a record: `training_input_digests`, the measurement manifest, the SP
       sidecar or the pipeline's bound digests.
   - **The last batch is checked at the entry points and the remaining call sites,** not only
     through the shared functions' own tests.
   - Docs and docstrings that name removed functions are updated, including the copy lists that
     S6 and S8 shorten: `src/kg/storage/__init__.py:10-30` and
     `src/kg/storage/file_storage.py:9-16`.
   - **As implemented (2026-10-12); code-reviewed with no P1 or P2 (`02a6954..f481882`, 2026-10-12).**
     That review covered the product code (554 targeted CPU tests passed; import-linter 4 kept).
     S10's tools, delivered with it, had three P2; their fixes passed re-review on 2026-10-12 (S10,
     "The tools").
     - **Deleted:** `verify_graph_artifacts`, `verify_graph_source`, `verify_generated_cohorts`,
       `compute_input_digests`, and the `file_sha256` re-export in `scripts/measure_scorer.py`
       (the plan allowed either; its pins became pins of its absence). Their reasoning moved into
       the read forms' docstrings (`verify_graph_reads`, `verify_graph_source_read`,
       `verify_cohort_reads`).
     - **Callers moved onto the readers,** each input read once and its record taken from that
       read: `audit_split_overlap.py` (`resolve_cohort`, one manifest read, one `read_samples` per
       split, `verify_cohort_reads`); `audit_generator_fidelity.py` (the manifest, `kg.json`, the
       tensors and both cohorts read once and checked with `verify_graph_source_read` and
       `verify_cohort_reads`; `phenotype.hpoa`, which the plan did not name, is read once with
       `read_once` and its rows counted from those bytes as they stream, so its recorded digest is
       the read's); `probe_deployment.py` (a script-local `verify_workspace` composing the shared
       readers as serving and training compose them, returning the read identities that C1's
       listing now reports; D2 reads the checkpoint with `read_checkpoint`). No composition helper
       was added to `src/`; the tests' own composition is `tests/fixtures/workspace_reads.py`.
     - **The fidelity audit parses the tensors** to check the workspace binding it reports, as
       every graph consumer does; it measures nothing from them, and so now imports torch.
     - **Static pin:** `tests/unit/test_provenance_pins.py` scans `src/**`, the three entry points
       and the three moved scripts. It counts every reference to `file_sha256`, `read_once`,
       `hashlib.*`, `torch.load`, `json.load(s)` and `load_json`, whether called, passed as a
       value, re-exported or imported under another name, and fails on any site outside its
       `ALLOWED` list and on any stale entry. `ALLOWED` stays in the test and names more sites than
       the allowlist above (the readers themselves, the ontology loader and resolver, the SP
       sidecar read, in-memory digests, configuration and UI loads, and the probe's identification
       sites), each with its use and owner. The removed names may appear nowhere in scope, and
       nothing in scope may bind `hashlib.file_digest`, which the poison replaces.
     - **Dynamic pin:** the poison now sits on `hashlib.file_digest`, so it also reaches a
       `file_sha256` bound at import; a self-test checks that it reaches `file_sha256` and spares
       the files named in `true_digest_for`. Serving uses the shared fixture, the SP producer has a
       poison pin (the sidecar records the bytes traversed; the end-of-run comparison only warns),
       and the moved scripts each have one.
     - **`KnowledgeGraph.read_json`** now refuses a `kg.json` that is not UTF-8 JSON describing a
       graph with a `ValueError` naming the file, as the workspace readers refuse theirs; the
       fidelity audit and the probe refuse a missing `kg.json` naming it and the manifest.
     - **Behaviour changes:** a non-JSON sample file is refused by the reader ("could not be
       parsed", naming the file) rather than by a digest mismatch, as S6 and S8 already did; a
       non-generated training split under a supplied evaluation cohort gets `resolve_cohort`'s
       refusal; the probe's C2 and C4 expect "is not the graph" for `kg`, since it checks
       `kg.json` as serving does.
     - Acceptance tests (CPU, development container; not a homelab acceptance):
       `test_provenance_pins.py`, `test_probe_workspace_phase.py`, the poison pins in
       `test_evidence_scripts.py`, `test_audit_generator_fidelity.py` and
       `test_sp_artifact_binding.py`, and the moved read-form tests. The static pin run against the
       code before S9 fails, and so do the new audit poison pins.
10. **S10 — capacity on the homelab, an executable matrix.**
    - **The subject:** workspace `data/workspaces/hpo_2026_0929_5a` (`$WS` below), checkpoint
      `$WS/checkpoints/hgt/model-02-0.1813.pt` (47,931,987 bytes), SP table `shortest_paths.pt`
      (10,797,575,893 bytes), on the GB10 with CUDA. It is M1's and N1's subject. **By the owner's
      decision (2026-10-10, decision 7), S10's capacity acceptance is limited to this subject.** It
      does not claim to cover all of the largest inputs, other workspaces or checkpoints, or any
      other deployment. It has the largest graph files, SP table and checkpoint in the homelab's
      `data/workspaces`, but not the largest sample files (decision 7).
    - **Before and after:** the same matrix runs at `90a668f` and at the M2.1 head, in the same
      checkout, venv and boot. The 2026-09-30 readings (23.77 GB VmHWM at ready, 28.45 GB over a
      reload; `ecf17cd`, `scorer-measurement/PLAN_B04_PRODUCTIONISATION.md:1199`, `:1202`) predate
      M1 and are context only.
      - **The "after" commit, fixed (2026-10-12).** It is the commit that completes S9 once the
        code reviews of the S6/S8 batch, of `1d0b5a6..06fd0f1` (R2 and item 8 among them) and of
        S9 with S10's tools have closed, with any review fixes on it, so that no review can change
        measured code after the run. Its full SHA is written here and in the evidence before the
        run, and `git diff <after> <the M2.1 pull request's head> -- src scripts` must be empty.
        On 2026-10-12 the reviews of the S6/S8 batch, of `1d0b5a6..06fd0f1` and of S9's code
        (`02a6954..f481882`) closed with no P1 or P2; S10's tools had three P2 in that last review,
        and their fixes (draft 3) passed re-review with no P1 or P2 the same day. **The "after"
        commit is `343e90e993d0a5129cd547b95276b7e2b8b12fa9`:** its `src` and `scripts` are those of S9's commit `f481882`, since
        every commit after it is docs only and no review asked for a code fix, and it is the head
        the reviewer cleared for the measurement. Dependencies are equal at `90a668f` and at it. The
        M2.1 pull request's head must have the same `src` and `scripts`.
      - **It runs before M2.3 and M2.4 refuse.** The subject records no `kg` role (its training
        roles at `43d47c5` are the samples, the tensors and the manifest), so once M2.3 refuses a
        resume parent missing a role and M2.4 refuses serving and measurement on one, the
        training `--resume`, serving and measurement rows would refuse (M2.3, M2.4, and "every
        existing checkpoint is refused" under M2's consequences). If those land first, S10 still
        runs at the fixed commit, from a detached checkout, and the evidence says so. Nothing is
        added to the subject's checkpoint to make it pass.
      - **Not only M2.1 changed between the two commits** on the measured paths: M3a's seed
        ordering (`290c03a`, training), HGT Edit 1 (`8eae614`, every HGT forward), the dashboard's
        telemetry off (`a535ae2`, the served process mounts it), R2 (`e210b12`, the measurement
        CLI) and item 8 (`06fd0f1`, the sample reader). Acceptance 2 and 3 account for them.
        Dependencies are equal at both commits.
    - **Pre-flight, once and read-only:** load the designated checkpoint with
      `weights_only=True`, the way Mode C does (`read_checkpoint` in `read_measurement_inputs`),
      and list its keys.
      - Expected: it loads. No Mode C run on this file is recorded yet; serving loaded it with
        `weights_only=False` (`src/inference/pipeline.py:973`). The earlier expectation that it
        carries neither `metadata` nor `in_channels_dict` supported only Mode A's "not
        applicable" row, which went with Mode A in the retirement's R2 (2026-10-12).
      - If it does not load, the reading records that Mode C cannot run on this file. No
        `weights_only=False` or other fallback is added.

    | Entry point | Command (key arguments) | What runs on the subject | Readings | Compared |
    |---|---|---|---|---|
    | Serving | `measure_served_pipeline.py --workspace $WS --checkpoint $WS/checkpoints/hgt/model-02-0.1813.pt --output <evidence>/served_readings.json --repeats 3 --seed 20260930` (an aggregate with no paths). It applies its own environment; its SHA-256 must be equal at both commits | launch to ready, 200 serial `/diagnose` requests, then one reload, each asserting `gnn_plus_shortest_path` | R1: VmHWM and VmRSS at ready, system peak less R0. R4: VmHWM over the reload. R2, R3: VmRSS after | all of these, with equal subject digests, allocator reading and `readings_complete` |
    | Measurement | `measure_scorer.py --checkpoint $WS/checkpoints/hgt/model-02-0.1813.pt --data-dir $WS --split val --cohort-kind generated --device cuda --output <scratch>/measurement.json`, plus `--modes C` at `90a668f` only, where `--modes` defaults to `A` and Mode A cannot run on this checkpoint; the retirement's R2 removed the option (2026-10-12), so the "after" commit takes none | **Mode C only** | final VmHWM; VmRSS at the entry and exit of `run_mode_c`; CUDA max allocated and reserved; system peak | the totals, since the phases move between commits; VmRSS at `run_mode_c` entry, compared and explained (acceptance 2) |
    | Training | `train_model.py --data-dir $WS --device cuda --conv-type hgt --hidden-dim 256 --num-layers 4 --epochs 3 --seed 42 --output-dir <scratch> --checkpoint-dir <scratch>` (the subject's flags, `scorer-measurement/subject-hgt-2026-09-29/commands.txt`), with `Trainer.train` replaced by a reading; once without and once with `--resume $WS/checkpoints/hgt/model-02-0.1813.pt` | the checks, reads, model, `Trainer` and resume read; no epoch and no checkpoint written | VmHWM and VmRSS at `Trainer.__init__` and at the replaced `train`; CUDA max allocated; system peak | all of these; VmRSS at the replaced `train`, compared and explained (acceptance 2) |
    | SP producer | `compute_shortest_paths.py --kg-path …/kg.json --output-dir <scratch> --max-hops 5 --workers 15` | a full run, about 15 minutes. The published pair in the workspace stays untouched, checked by size and time | parent VmHWM after the graph read; VmRSS when the workers fork, after the graph read; at publication; tree PSS; system peak | all of these; the new sidecar names graph `6cae2d1a…`, 431,902,937 pairs, `max_hops` 5 |

    - **Not applicable, recorded as such:**
      - Modes A and B: deleted by the Mode A/B retirement's R2 (2026-10-12); the CLI has no `--modes`;
      - the calibration launcher: removed in S0;
      - serving's CUDA allocator counters, which `PLAN_B04_PRODUCTIONISATION.md` §7.3 excludes.

      **M2.1's capacity acceptance needs no measurement reading other than Mode C's** (S10 ran on
      2026-10-12; nothing here is accepted until its evidence is reviewed). M2.1 owes
      no Mode A or B reading on a real checkpoint, and none is wanted until a runnable subject is
      named (BACKLOG 19.18). Since the owner's clarification of 2026-10-12 none will be: Modes A
      and B were deleted by the retirement assessment's R2 (2026-10-12).
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
      - the page-cache state: `PAGE_CACHE=warm`, by the owner's decision through the reviewer
        (2026-10-12), which replaces "warm by default, or dropped before each row if the owner
        chooses it":
        - both commits use the same pre-read file set, order and tools: every `.pt` and `.json`
          file in the workspace, in byte order of the name (a symlinked file followed), then the
          checkpoint, listed in the evidence's `preread.txt`;
        - the same pre-read runs before each measured row, and R0 is taken after it;
        - the environment record and the report say this is a capacity comparison after a
          pre-read, not a cold start;
        - there is no drop control and no extra matrix;
      - the byte size and SHA-256 of every input: `kg.json`, the tensors, `num_nodes.json`, the
        sample files, the manifest, the SP pair and the checkpoint;
      - the sampler and the probe, with SHA-256 equal at both commits, and the sampling interval:
        0.2 s, as `measure_served_pipeline.py` (`:86`).
    - **The tools:** a `/proc` sampler outside the measured process and a probe inside it, kept
      outside the repository like M1's acceptance client. Their digests are recorded with the
      readings. They are acceptance instruments, not a second pipeline:
      - the probe runs inside the real entry point, only records, and stops at the stated
        boundary. It never rebuilds the reads, the verification or the model;
      - the same instrumentation runs at both commits. Each hook runs once, holds no large object
        and reads none of the run's inputs again: no hook opens a subject file, and a hook's only
        file read is the process's own `/proc/self/status`. This binds the hooks inside the
        process; the outside sampler reads the `/proc` entries it samples;
      - CUDA readings synchronise, and reset the peak counters, at the same points at both
        commits; whether torch's peak counters work under the saved `cudaMallocAsync` preset is
        confirmed on the GB10, and if they do not, the probe records the error and the row goes
        on;
      - **the boundaries,** which exist by name at both commits: `run_mode_c` entry and exit;
        `Trainer.__init__`'s return, and `Trainer.train`, which the probe replaces by a reading
        and a stop (the only function it replaces); in the SP producer, after the graph read, the
        first worker fork (`os.register_at_fork`), and the start and end of `publish_sp_artifact`;
        and the end of the process. The probe observes them with `sys.monitoring` on those named
        code objects, running the entry point in its own process as `__main__`. Besides
        `/proc/self/status`, its only reads are its own source, for its digest, and the entry
        script it runs;
      - **a boundary's peak is the maximum over the probe's readings,** because VmHWM as read was
        not monotonic in the dry runs (a later reading was lower), and the interpreter's shutdown
        after the last boundary is reported apart: it moved the root's high-water mark by up to
        0.12 GB, in either direction, between identical runs;
      - outputs, logs and the SP publication go to named scratch paths. Nothing is written to the
        designated checkpoint or the workspace;
      - a system or process-tree peak from 0.2 s sampling is labelled a sampled peak. It does not
        guarantee that every transient was caught.

      The tools' full source and SHA-256 are sent with the measurement instructions, for review
      before any run. They are a sampler, a probe, a matrix driver that runs the rows the same way
      at both commits (it adds `--modes C` only where the CLI has the option), a comparison, and
      the tools' own tests. Drafts were dry-run on CPU, without serving, at `90a668f`, `06fd0f1`
      and `f481882`; a dry run is not a reading.
      - **The review of 2026-10-12** (with S9's code, `02a6954..f481882`) found three P2 in those
        drafts: the comparison printed "met" for evidence that fails the declared conditions; the
        sampler stopped only its direct child, so SP workers could be left running; the driver
        could reuse an old output directory, so a failed run could mix with an earlier run's
        evidence. The plan as amended (`f481882..c1b8a12`) was reviewed with no P1 or P2.
      - **The fixes (2026-10-12, outside the repository).** The reviewer's re-review at `3c9563f`
        closed the three P2 and found one more, a failed pre-read that did not stop its row; that
        fix and its tests (draft 3) were re-reviewed at `343e90e` with no P1 or P2 (2026-10-12).
        The reviewed draft 3, by SHA-256: `s10_probe.py` `04458fd77a96b6728eda66cfbae6a3f42700953de08cf556e660258c61d02a0b`,
        `s10_sampler.py` `1ae557ad304dbb184b54e4c4f305d75936ed319b0f2f75849dcd377167dfa1cc`, `s10_run_matrix.sh`
        `6eaeffcf52a1c2192c1339deaa62d0f24eb2191ebe6a32ec3f170976ee1e2afe`, `s10_compare.py` `7fbb86d59151d60cfe481ae97cbc287d98bb70607e53b1d1f64d4d1f39b3578f`,
        `test_s10_tools.py` `63916d48c68c6bcd3a4e2cbc5801db59025ff145792c4bb76a5d54662f35f897`. An
        internal adversarial review and a re-check of its findings followed; every P2 they
        confirmed is fixed in the tools delivered, and what they left is listed in the
        instructions:
        - the comparison judges each condition as an explicit check over the evidence files,
          PASS, FAIL or UNKNOWN (a missing file or field, named, never read as zero); a definite
          failure is not hidden by missing data in the same check; its one verdict says "met"
          only when every check passes, with exit status 0, 1 or 2. It checks acceptance 1 at the
          after commit (this run's rows each once with exit 0, every boundary, training's stop
          with `--resume` only on the resume row and CUDA initialised, scratch with no epoch
          output, Mode C with `cuda_executed`, the SP sidecar, serving complete with no OOM kill
          or swap increase in any phase of any repeat, no sampled row stopped, killed or leaving a
          process, serving's R0 swap precondition and increase rule for every sampled row, the
          workspace unchanged, the subject by digest and by the sizes above, readings written by
          the recorded tools) and comparability (the same row checks on the before run; inputs,
          tools, pre-read list, environment, boot and the matrix's own settings equal and the
          plan's; the rows' explicit allocator the same as serving's, and serving's allocator
          reported and equal; the service stopped and no GPU process; each commit's `--modes`
          form). Its output names the reference values and one evidence digest per directory.
          Acceptance 2 and 3 stay a person's explanation;
        - the sampler starts the measured command in a session and process group of its own and
          ends that group however the row ends: a stop signal (SIGINT, SIGTERM, SIGHUP, SIGQUIT)
          is forwarded to the group, a second one or one outliving 30 s becomes SIGKILL, workers
          left after the root exits are recorded and stopped, every wait is bounded, and the root
          is reaped last, after which nothing is sent to its group id. It signals nothing outside
          the group it created;
        - the driver refuses, before it creates or measures anything, an evidence or scratch
          directory that exists and is not empty, a scratch directory inside the evidence one (other
          than its default) or the reverse, a wrong HEAD and modified tracked files; it claims both
          directories, deletes nothing, records the matrix's settings, and stops before the next row
          when a stop arrives between rows; one pre-read step serves every row, and a pre-read that
          fails stops the run before that row's command starts, recorded in `rows.txt` as not run
          (exit status 4); a rerun uses a new directory;
        - tests: a complete passing control and single negative controls for the comparison; the
          stop signals, a second signal during the cleanup, a root exiting before its workers, a
          root outliving SIGKILL, an error in the sampler's own readings, a signal before and
          while the command starts, nohup, and serving's swap rules, with real forked workers; the
          directory refusals; a good pre-read and a failed one before serving, before a sampled row
          and without serving. Dry-run again, without serving, at `90a668f` and `c1b8a12`.
    - **The evidence** goes under `docs/working/scorer-measurement/` beside the 2026-09-30 readings,
      as aggregates with no paths or host names: the environment, the tools' and inputs'
      SHA-256, the pre-flight, each row's sampler and probe readings, and a README with the
      before, after, difference and explanation of each reading. The run's scratch (server logs,
      the measurement report with paths, ranks, configurations and the SP table) is not
      evidence and stays on the homelab.
      - **As run (2026-10-12):** [`capacity-m2.1-2026-10-12/`](scorer-measurement/capacity-m2.1-2026-10-12/README.md). Every acceptance 1 check
        passes at the after commit (51 PASS). Comparability: 60 PASS and 1 UNKNOWN,
        the service state, because the homelab has no `shepherd` user unit; the owner's statement
        of 2026-10-13 that the backends were closed supplements it, and the exit status 2 is
        kept. Acceptance 2 and 3 are explained there, except the serving reload's resident rise
        (R3 +0.023 to +0.031 GB): no raw-buffer retention in the readers measured, source not
        located, allocator retention a candidate, one reload per process. Acceptance 4 is not
        triggered. The reviewer's evidence review (2026-10-13) found no P1, one P2 (two
        `driver.log` files missing from the commit, since added) and recommends a limited
        diagnostic of consecutive reloads before acceptance 2 is closed. Not accepted.
      - **The reload diagnostic (the owner's choice, 2026-10-13; the evidence README §6.1).**
        It answers one question: does the reload's resident difference keep growing over
        three successful reloads in one server process, and does the service stay healthy.
        One diagnostic per commit (`90a668f`, `343e90e`), back to back in one boot, after
        S10's pre-read, with the same workspace, checkpoint and settings: the same 200
        requests first, then three reloads in a row through the HTTP route, each followed by a
        readiness check and diagnosis requests; RSS, anonymous memory and Pss at fixed points,
        with the OOM-kill counter and swap. No `strace`, `mallinfo2`, allocator change,
        `malloc_trim`, `gc.collect`, `empty_cache` or second loader. Its driver,
        `s10_reload_diag.py` (SHA-256 `2d3cfc397b4867fe2cc876b6a58254967e64bec1f5771b695609ddfc963a41ff`),
        and its tests `test_s10_reload_diag.py` (`16f523b5b433719f4d0ac3b97717dcabdcaa66f856da9bf7cd528c277fc5931b`), are kept outside the
        repository. The reviewer's review of draft 5 (2026-10-13) found no P1 or P2 in them and
        two P2 in the runbook's reading rules: a bounded cost needs each process to level off on
        its own, not only the difference to stay flat; and a difference already present before
        any reload does not confirm the reproduction. Both were fixed in the runbook only, and
        the revised rules were checked before the run. The bands (6 and 11 MB;
        the first reload's difference within +16 to +38 MB, the baseline within -11 to +5 MB)
        are observation bands declared from S10's readings, not confidence intervals. Its
        results are kept apart from S10's evidence. A difference that does not
        grow, with each process levelling off on its own, supports
        accepting a bounded cost for this subject and three reloads, the source still not
        located; growth with the reloads, or a failed reload, readiness or diagnosis, or an
        OOM kill or swap rise, leads to a second phase that locates it.
        - **As run (2026-10-13):** [`reload-diag-2026-10-13/`](scorer-measurement/reload-diag-2026-10-13/README.md). Healthy at both commits
          (6 reloads and 430 diagnoses accepted, no OOM kill, no swap). Each process grew at
          every reload, by about +410 to +470 MB per cycle and +1.2 GB after three reloads, at
          `90a668f` as at `343e90e` (rule 4): present before M2.1, not attributable to it, and
          no bounded cost is supported. The difference between the commits reproduced the
          first reload's (+32.6 MB), then moved by −5.0 and +7.6 MB: ambiguous (rule 6).
          Acceptance 2 stays open. Next, the second phase: the growth common to both commits
          first, M2.1's difference separately; a read-only reading of the reload path finds no
          cache that keeps the old pipeline, and any in-process diagnostic is proposed with its
          hypothesis, observations, runs and stopping rule and reviewed before it runs.
        - **Second phase, in process (2026-10-13):** [`reload-inproc-2026-10-13/`](scorer-measurement/reload-inproc-2026-10-13/README.md). The growth
          recurs without HTTP (+1.2 GB after three reloads at `343e90e`); every watched object
          of the earlier pipelines was already freed; a full collection and a trim return
          about 70 MB. Cause not located. Next, by the reviewer's instruction: a read-only
          check of the embedding and SP build steps, then at most one run of the same tool with
          readings around those two methods.
        - **The supplementary run and the conclusion (2026-10-13):** [`reload-inproc-segments-2026-10-13/`](scorer-measurement/reload-inproc-segments-2026-10-13/README.md).
          The run with segments did not repeat the steady growth; the reviewer found that the
          wheel's `libc10.so` allocates CPU memory through mimalloc 2.2.4 with two purge
          defects since fixed upstream. The owner ended the investigation: the PyTorch wheel in use is confirmed to contain mimalloc purge defects that upstream has since fixed; they are a strong candidate cause of the observed memory retention; the attribution of all of the increase is not established.
          No causal control, no pipeline change, no `MIMALLOC_*` setting, no new tool; the
          dependency limitation is BACKLOG item 22. The common growth and M2.1's ~30 MB
          difference are recorded apart; acceptance 2 is the owner's separate decision, and S10's
          capacity is judged on its own readings and limits.
        - **Accepted (the owner, 2026-10-13)**, on the reviewer's recommendation: S10's capacity on
          its designated subject, with the remaining uncertainty explicitly accepted; acceptance 2
          closed with the reviewer's wording; the scope (the GB10, the recorded environment, the
          warm policy, this subject, the five-row matrix; training read only to its preparation
          boundary) and the known limitations (the unattributed ~30 MB, the common retention as
          BACKLOG item 22, the comparability UNKNOWN with exit status 2) are in [the capacity README §6.2](scorer-measurement/capacity-m2.1-2026-10-12/README.md#62-acceptance-2026-10-13).
          M2.1's technical work is complete; not merged; M2 is not complete.
    - **Each reading lists the objects alive in each phase** beside its numbers: for measurement,
      for example, the graph tensors, the parsed samples, the checkpoint dict, the model and the
      encoded embeddings. Acceptance 3 explains peaks against this list.
    - **Acceptance, on this subject only (decision 7):**
      1. **Every run completes at the head** on the real subject, by the same procedure as at
         `90a668f`, with no swap increase and no OOM kill. That means:
         - serving's `readings_complete`;
         - Mode C exits 0 with `cuda_executed`;
         - the training readings, without and with `--resume`;
         - the SP producer publishes into scratch with the expected sidecar, and the workspace
           pair is unchanged.
      2. **Resident memory after the reads is compared between the two commits and explained.**
         - A rise is an observation to investigate, not by itself proof that a buffer was kept.
           The allocator may keep freed pages resident, live parsed objects may pin them, and the
           new reads change the size and order of transient allocations.
         - A buffer kept after its parse is ruled out by the readers' release checks (S2–S4), not
           by RSS.
         - An investigation uses a separate diagnostic run, never a run whose readings are
           reported, so the diagnosis does not disturb the capacity readings. `tracemalloc` in
           that run shows live Python allocations, such as a kept buffer or decoded text. It does
           not see torch's tensor storages, which are accounted for from their shapes and dtypes
           (as above, "Memory") or with torch's profiler, nor pages the allocator keeps after a
           free, which only `/proc` shows.
         - An added resident cost is explained and assessed even when no buffer is kept. It does
           not by itself require the reader to be rewritten.
      3. **Each change in a peak is explained** against the recorded file sizes and the objects
         alive in that phase. "About one transient copy" guides the explanation; it is not a
         bound.
      4. **A shortfall** (a run that does not complete, an OOM kill or a swap increase) **is fixed
         in the reader,** never with a CPU fallback. Nothing is added only to make RSS fall: no
         `malloc_trim`, forced collection, `empty_cache` or second read path.
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
      holds an SP pair whose sidecar records its graph, a `kg.json` republished this way is also
      refused by the sidecar's `kg_digest` (`src/inference/pipeline.py:788-790`), unless the
      pair is republished with it;
    - at serving, the API reads `kg.json` before the pipeline reads the manifest, so a `kg.json`
      republish between those two reads is refused ("is not the graph").

    This is a rule for the roles the manifest binds. It is not an atomic transaction over a
    multi-file publication, and it is not the model↔graph check, which is M2.4's.
- **A replacement after the manifest read is refused (item 2).** The double fires on the
  manifest's read and replaces a sample file or a tensor. The refusal names the file, and nothing
  is written or published (S5–S8).
- **Records come only from readers (item 3):** the static and dynamic pins (S9).
- **Capacity (item 4):** S10, on Mode C for measurement. That each buffer is released is shown
  by the readers' release checks (S2–S4); S10's readings are explained, not used to infer it.

**Review and delivery.** Three review batches: S0–S5, the removal and the shared base; S6–S8,
the entry points; S9–S10, removal of the path forms, pins and capacity. M2.1 ends as one pull
request, as M1 did. As delivered: S0 (`45d9dea..3433bf8`) and S1–S5 (`e3eefee..6f16373`) were
reviewed apart, S7 (Mode C only) on its own (`3d53c9b..7327798`), and the S6/S8 batch
(`8421b8c..cddad9d`) on its own (no P1 or P2, 2026-10-12). S9's code (`02a6954..f481882`) and S10's
tools were reviewed together and judged apart (2026-10-12): S9 with no P1 or P2; S10's tools with
three P2, whose fixes passed re-review (draft 3, 2026-10-12). S10's
evidence follows its run.

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
3. **The calibration launcher: by the owner's rule, it is removed if nothing necessary uses it**
   (migrated minimally only if it must stay; no dead islands). S0's check found no necessary user,
   and the reviewer accepted S0 in the re-review of `0708789` (2026-10-09). The incremental review
   of `7e1af96` (2026-10-10, no P1 or P2) accepted S0 as revised by then, at plan level: its opening
   (the owner's rule stated for the launcher only, with the evaluator left to question 6), its claim
   limited to the checkpoints checked, the wider list of text it rewords, the lifecycle statements
   it drops or ties to item 9, the hand-off to S7, the final search, and the gate's documents
   (§5.0's restatement added). That is not acceptance of S0's change. S0's additions for decision 6,
   made since (the evaluator in its opening and deletions, the evaluator and its pin test in the
   gate exception, the statements it now rewords before S7, and the corrections that go with
   them), were accepted at plan level in the review of `45d9dea` (2026-10-10). The rule is the
   launcher's; the frozen evaluator is decision 6. The question of the launcher's digest
   bracket (BACKLOG 19.19) lapses with the launcher's removal.
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

**Decisions — the owner, 2026-10-10, after the reviewer's view.** Neither is asked again unless a
new dependency or measurement fact appears.

6. **`scripts/evaluate_model.py` is removed with its pin test in S0.** The reviewer recommended
   this in the re-review of `0708789` and the incremental review of `7e1af96`.
   - **Why it can go.** After S0 nothing runs it: the launcher was its only runner, and
     `tests/unit/test_frozen_evaluator.py` only pins its bytes and fails if a module under `src`
     imports it. Its own banner is already false and pinned: it says the file is "kept only so
     the new harness can be calibrated against it", and to delete it "together with
     `legacy_ranking` and the `LEGACY_TRUNCATION_K` path" (`evaluate_model.py:3-6`, `:19-24`).
     §3.1.2 retired that calibration, and the trainer shares `legacy_ranking`'s sort
     (`scorer-measurement/README.md:90`).
   - **The gate.** It exists so the harness is never left without an acceptance, and this file
     accepts nothing once the launcher goes. S0's gate exception names it and its pin test: in
     BACKLOG §5, §5.0 and item 9's row, and in `scorer-measurement/README.md:78-79`, `:95-97`,
     `:122` and `:125`.
   - **Historical citations name a fixed revision, `7dab728`.** `62b8103` only added the 25-line
     banner, so today's file is 25 lines further on.
     - Already in `7dab728`'s numbering: `src/evaluation/measurement.py:157` (`:295`), `:166`
       (`:505-519`), `:249` (`:513`) and `:702` (`:508-519`);
       `tests/fixtures/synthetic_workspace.py:109` (`:164-223`);
       `tests/unit/test_measurement_mode_a.py:546` (`:508-519`);
       `results-review/SPEC_4_DEPLOYMENT_SECURITY.md:92` (`:111, 201, 205`); and
       `measure_scorer.py:203` (`:164-223`), which was to go with `load_legacy_mode_a_inputs` in S7 and went with it in the retirement's R2 (2026-10-12; S7 was implemented for Mode C only).
     - In today's numbering, re-anchored to `7dab728`: `src/evaluation/measurement.py:250-251`
       (`:324`, `:116`, which are `:299`, `:91` there); and BACKLOG §3.1 (`BACKLOG.md:346`,
       `:163-173`, which is `:138-148` there).
     - `scorer-measurement/PLAN_B02_shipped.md`, a "History, not authority" plan, cites the file
       in `7dab728`'s numbering throughout (for example `:22`, `:95`, `:214`, `:248`, `:261`). Its
       history note gains one sentence: those citations are at `7dab728`.
     - This plan's constraint on `read_samples`' fields is already re-anchored, to `:185-191` at
       `7dab728`, and rested on Modes A and B (above, "Two constraints"; since the retirement's R2, on training passing its two fields explicitly).
     - By name only, named at `7dab728`: `measure_scorer.py:234` and `:420-422`, and
       `scorer-measurement/README.md:93`.
   - **Text that names it as a live file** becomes history or is removed:
     `src/kg/storage/__init__.py:13-14`, with its counts at `:22-23` and `:28`;
     `src/kg/storage/file_storage.py:9-16`; `src/utils/logger.py:11-17`;
     `scripts/audit_checkpoint_family.py:9-10` and `:26-27`, and
     `tests/unit/test_evidence_scripts.py:108-109`, which repeats `:26-27`;
     `tests/unit/test_measurement_mode_a.py:434-435`; `tests/fixtures/synthetic_workspace.py:5-8`,
     whose first sentence becomes "The dataloader seeds nothing";
     `results-review/SPEC_4_DEPLOYMENT_SECURITY.md:91-93`; `PLAN_TEST_RESULTS.md:162-164`, `:291`
     and `:293`.
   - **The exceptions it held are closed:** `EVALUATION_COHORTS.md:974-979`, BACKLOG item 11i,
     `PLAN_PHENOTYPE_NORMALISATION.md:312` (which says it "retires with the frozen evaluator";
     the oracle clause goes with its citation, and that row, "Two readers ... neither checks", and
     `:523`, "the two named exceptions (§5)", then name one exception,
     `measure_served_pipeline.py`) and `PLAN_TEST_RESULTS.md:763-766`.
   - **Two further corrections in S0's change:**
     - `src/evaluation/measurement.py:1112-1113` cites `evaluate_model.py:285, 366`, which match
       no version of the file. The `compute_all` call is at `:318` at `7dab728`;
     - `src/inference/scoring.py:4-7`, `tests/unit/test_scoring_primitives.py:10-11` and
       `results-review/SPEC_1_RESULTS_REVIEW.md:332-336` stop saying the frozen evaluator is still
       to be migrated onto the shared primitives ("is migrated later in B-0", "does not yet", and
       "evaluator migration" as remaining B-0 scope). It was never migrated, and S0 deletes it.
   - **Nothing else is removed with it.** Item 9 keeps the rename and the rest of the
     oracle-only surface, which still runs on fixtures: `build_legacy_mode_a_model` and the
     parity assertions in `test_measurement_mode_a.py`. Code that the measurement CLI or a valid
     test still calls is not deleted because its name says "legacy". The frozen file is not
     edited before it goes, and item 7a's runner need not exist first.
     *Superseded by the Mode A/B retirement (the owner, 2026-10-12):* item 9 was replaced by the
     retirement's R2, which deleted `build_legacy_mode_a_model`, the parity assertions and the rest
     of Modes A and B (2026-10-12) because nothing current used them, not because a name says
     "legacy"; the rule in the sentence before stands.
   - Not chosen: keeping it as a pinned historical file until item 9's last step.
7. **S10's capacity acceptance is limited to its designated subject:** workspace
   `hpo_2026_0929_5a` with checkpoint `model-02-0.1813.pt` (S10). It does not claim to cover all
   of the largest inputs, other workspaces or checkpoints, or any other deployment. M2.1's
   "Capacity" rule is amended to say so.
   - **The reviewer added** that any future guarantee for larger cohorts would need its own
     capacity scenario. M2.1 plans none.
   - **The facts behind it,** from the reviewer's read-only inventory of the homelab's
     `data/workspaces` on 2026-10-10 (file sizes only; nothing loaded, hashed or run):
     - the subject has the largest `kg.json` (209,928,843 bytes), `node_features.pt` (29,308,567),
       `edge_indices.pt` (19,772,821) and SP table (10,797,575,893), and the largest of the 21
       `.pt` files under the workspaces' `checkpoints` and `train_outputs` directories
       (47,931,987);
     - its sample files are not the largest: `hpo_2026_0623_v2_orpha` has the largest
       `train_samples.json` (16,306,773 against 12,867,342) and `val_samples.json` (2,444,868
       against 2,364,974);
     - so no workspace is largest in every kind of input.
   - **The inventory's limits:** it covered that directory only, not the whole machine, downloads,
     external disks or archives; and the largest file is not necessarily the largest peak at
     every entry point, since parsed shapes and lifetimes shape the readings.

#### M2.2 — the producer records the graph it consumed

**Change:**
- **`training_input_digests` gains `kg`**, beside the three tensors and the manifest
  (`train_model.py:551-574` at `90a668f`; since S8, the builder `training_input_digests` over
  the run's reads).
- **Training does not parse `kg.json`.** Its `kg` role is the digest bound by the manifest the
  run read, against which the tensors it parsed were verified. This is the transitive claim
  `verify_graph_artifacts` already describes (`artifacts.py:73-75`).
- **Every other role's digest comes from M2.1's reads.**

*As implemented (2026-10-13):* `TrainingInputs.bound` keeps what `verify_graph_reads` returned, and
`training_input_digests` records `bound["kg"]`, refusing a manifest that binds none.

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
  exactly when the parsed `val_samples` are not empty. S8 already decides it there
  (`read_training_inputs`, `with_validation`), before the run writes its directories.
- **The comparison is M2.4's comparator** over a wider set of roles. It is one function, and the
  set of roles is its argument.
- **The trainer restores from the dict already read and checked.** S8 made the restore read
  nothing (`Trainer.restore_checkpoint`, over `read_checkpoint`'s result); M2.3 moves that read
  to the top and puts its checks before the restore.

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

*As implemented (2026-10-13):* `read_resume_parent` runs right after `read_training_inputs`, before
any run artifact: a missing parent raises `FileNotFoundError`; the parent is read once
(`read_checkpoint`, the site's options) and `check_resume_parent` compares it with
`training_input_digests(inputs)` through `verify_training_record`, over the union of this run's
roles and the parent's recorded workspace roles, so a role on one side only refuses. The parent's
own `resume_checkpoint` is its lineage and is not compared. `train` restores from that read's dict
and records its identity; only that identity outlives the restore, so the parent's source tensors
are not held through training (the review's P2-2). `Trainer.save_checkpoint` embeds
`training_input_digests` as `ModelCheckpoint` does, so a checkpoint from either writer can be
resumed.

#### M2.4 — the consumers: serving and measurement

- **They compare the checkpoint's recorded `kg`, `node_features`, `edge_indices` and `num_nodes`
  digests with the workspace's.** For serving these are taken from M2.1's reads. For
  measurement, `kg` is the manifest-bound source graph (M2.1's stated exceptions, amended
  2026-10-09), and the other three roles come from M2.1's reads.
  - Serving keeps only `kg` today (`src/inference/pipeline.py:532` at `90a668f`), and all four
    after M2.1's S6. Measurement's runs without Mode A keep them since S7
    (`MeasurementInputs.bound`); runs that include Mode A discarded them
    (`measure_scorer.py:128` at `90a668f`) until the retirement's R2 deleted those runs (2026-10-12).
  - A missing `training_input_digests`, or a missing role, refuses.
- **The comparison lives in the slot `verify_fingerprint` deferred** (`fingerprint.py:168-176`),
  so there is one model↔graph comparator.
  - The structural fingerprint stays as a diagnostic, never as identity.
  - The probe asserts through the same function instead of its own comparison
    (`probe_deployment.py:812-836`).
- **A refusal raises (M1)**, so it cannot become a fallback.

*As implemented (2026-10-13):* `verify_training_record` (`src/utils/fingerprint.py`) refuses with
`TrainingRecordMismatch`, a `ValueError`, unless every role in its argument is recorded and held,
with the same digest; a missing record or an empty role set refuses. Serving calls it in
`_load_model_from_checkpoint` before the model is built (raised as `PipelineBuildError`),
measurement in `read_measurement_inputs` right after the checkpoint read, and the probe's D2
through it. **A checkpoint is loaded only over a graph verified from a workspace:** a pipeline whose
graph is supplied in memory (`graph_data=`) refuses one before opening it, whatever it records (the
review's P2-1 and the reviewer's recommendation B, 2026-10-13); a model handed in built is the
caller's. The callers that paired a checkpoint with an in-memory graph moved to the workspace path.

#### M2.5 — SP

An unrecorded SP binding refuses. This ends the legacy "serves with a warning" state (R4), under
decision 2.

*As implemented (2026-10-13):* `_load_shortest_paths` raises `SPArtifactError` for an unrecorded
binding, before the empty-table decision. A table without a sidecar records no binding, so the
configured hop bound (`PipelineConfig.sp_hop_bound`, `SHEPHERD_SP_HOP_BOUND`), which existed for
such tables, could no longer reach a served table; it is **retired** (the reviewer's
recommendation A, 2026-10-13): the variable is not read, the field and the configured fallback are
gone, and the bound comes from the sidecar alone. The producer's `--max-hops`, the sidecar's
`max_hops` and `validate_hop_bound` stay. The deployment guides now rebuild a bound pair instead.

#### M2.6 — inventory before deployment

A read-only listing of every checkpoint, SP table and workspace that M2 would refuse, and why.
- It is an option of an existing audit (`scripts/audit_checkpoint_family.py` or the deployment
  probe), not a new tool.
- It is deployment preparation, not a second execution path.
- Old artifacts are archived, not deleted.

*As implemented (2026-10-13):* `scripts/audit_checkpoint_family.py --inventory <workspaces>` lists,
per workspace, the graph check, the training inputs, the shortest-path table and every checkpoint
in it, each with the refusal's reason, calling the consumers' own functions. It writes only its
report, with names relative to the inventoried directory. After the review's P2-3: the table's
**source binding** is judged from its sidecar, and whether the table would be **served** is
`refused` where the binding refuses it and otherwise `not judged`, never accepted, since the tensor
is not loaded. A checkpoint is read once with `read_checkpoint` (CPU, the safe loader), its digest
recorded from that read, and its two verdicts are M2's source-contract checks only (`m2_contract`:
`serve_and_measure`, M2.4; `resume`, M2.3), not whether its weights build or a device suffices.

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

*How it is run (the owner's decision, 2026-10-13):* through the deployment probe, extended and
nothing else (`scripts/probe_deployment.py --m2-acceptance --device cuda`), every check from the
production entry point that makes it -- the trainer, the measurement CLI, the serving build, the
shortest-path producer and the inventory -- never from the comparator alone.
- **Demo part (phase M, after B–E):** the successes first (a resume that trains on and names its
  parent; measurement; serving the resumed model with a shortest-path table the loader verifies),
  then the five resume refusals, each leaving no run directory and nothing changed, and serving
  and measurement refusing a checkpoint over a workspace that is valid on its own and the same
  shape; then the inventory. Faults are injected only into copies under the work directory.
- **Real-data part (phase R):** on `hpo_2026_0929_5a`, read and never written: the workspace and
  its table's binding check out; the settings (a `config.yaml`) carry the architecture recorded by
  `model-02-0.1813.pt`; one epoch from scratch and one resumed, every output under the work
  directory; measurement and serving of the new checkpoint, SP verified, a diagnosis; the
  inventory refuses `model-02` for its missing record and passes the new model; the workspace is
  unchanged by file listing, size and modification time (not by content digest). The settings are
  the workspace's own `train_outputs/config.yaml`, whose six architecture fields, with
  TrainConfig's defaults, match `model-02`'s record (the reviewer's check, 2026-10-13). The
  short-trained model shows the pipeline and its checks work; it is not a serving model, which the
  established training and evaluation choose.
- **Accounting:** CUDA is required and the run stops after A2 without it; the report lists, per
  part, every required probe and any not run, skipped or failed; the acceptance is complete only
  when both parts are. The work directory is kept, with the tested commit, the device, the
  artifacts' digests and each probe's result. The report must be inside the work directory, can be
  no input and no existing file, and is created, never replaced.
- **Review (2026-10-13):** the reviewer's review of `26fe84c..a0fcc4a` found two P2 in the probe and
  none in the product code: the report path could overwrite an input or an earlier result, and M3
  measured the parent instead of the resumed model. Both were fixed in `82d7573`, the commit run.

*As run (2026-10-14, at `82d7573`):* 62 of 62 probes passed and the acceptance was complete in both
parts ([evidence](m2-acceptance-2026-10-14/README.md)); the reviewer recomputed the digests of the
real checkpoints and reports and found it passed. Limits recorded there: R8 compares listing, size
and modification time; the probe ran the trainer as a module and so skipped `train_model.py`'s
start-up allocator rule, the runs recording no allocator variable while the saved preset was
`cuda_async`, so the run used the native backend and its memory (about 104 GB attributed to the
probe by `nvidia-smi` during R3; about 3.84 GB of swap in use afterwards; no continuous trace) is no
capacity reading; the short-trained model is not a serving model.
- **The allocator follow-up (2026-10-14):** the probe, when it is the program, applies the shared
  rule (`allocator_env(..., inherit=True)`, as `train_model._apply_allocator`) before anything
  imports torch, keeping an explicit override under either variable name, and the report records
  the variables, their source and the backend torch reports (`allocator`). Importing the module
  changes nothing. *Done (2026-10-14):* reviewed with no P1 or P2 (`82d7573..b8a43c8`), and at
  `b8a43c8` the probe's existing demo phases on the GB10 passed (42 of 42) with the allocator set
  from the saved preset before torch's first import and the backend torch reports `cudaMallocAsync`
  ([evidence](m2-acceptance-2026-10-14/README.md#6-the-allocator-follow-up-on-the-gb10-at-b8a43c8-2026-10-14)).
  The two epochs were not rerun; nothing here is a reading of a full HGT training's memory.

### M3 — producers record what consumers need, and consumers check it

Three changes, each complete. M3a can proceed in parallel with M3b and M3c.

**M3a — training records the setup that took effect, and resume continues exactly (R6, R11).**

**Revised on 2026-10-15** after the reviewer's design review of that day (five P2 against the first
proposal, each answered below) and its incremental review of `0142436..76ca541` (two P2: an
extension's own stage, and the migration's labels, answered below), with decisions 8–10 (§1). The seed ordering is implemented (below); nothing else
in M3a is. Line numbers are at `0142436`, whose code is `main`'s `9fe5113`.

**What the code does today.**
- **Every run of a workspace and architecture writes the same directories:**
  `{workspace}/checkpoints/{conv_type}/` and one fixed output directory (`src/utils/checkpoint_paths.py`;
  `scripts/train_model.py:782-806`). Each run replaces `last.pt` and `config.yaml`. A best-k name,
  `model-{epoch:02d}-{val_mrr:.4f}` (`src/training/trainer.py:411`), can repeat across runs, and the
  later run's pruning then deletes the file (`src/training/callbacks.py:333-341`).
- **Resume compares data and graph roles only** (`train_model.py:542-567`; decision 7).
- **Resume restores the parent's optimizer and scheduler state into a scheduler built from the
  child's settings** (`trainer.py:1015-1051`). `LambdaLR` saves its progress, not its function.
  So the child's `learning_rate` is replaced by the parent's. The cosine schedule's progress has no
  ceiling (`trainer.py:327-341`), so resuming a finished schedule climbs back to the full rate.
  `OneCycleLR` raises past its total. Reproduced with PyTorch 2.10 outside the trainer.
- **The schedule's step count and the updates taken disagree.** The total is
  `num_epochs × len(train_loader) // gradient_accumulation_steps` (`trainer.py:365-376`), while
  `_train_epoch` steps only when the batch index divides by the accumulation (`:550`). With two
  epochs of five batches and an accumulation of two, the formula says 5 and the loop updates 4 times.
  The tail batch's gradients stay in `.grad` at the epoch's end, and no state dict holds them.
- **Resume restarts the epoch loop at 0** and runs the full `num_epochs` (`trainer.py:467`).
  - The checkpoints ModelCheckpoint writes (`last.pt` and best-k, `callbacks.py:289-330`) hold no
    global step and no trainer state, unlike `Trainer.save_checkpoint` (`trainer.py:983-1013`).
  - `on_train_begin` clears early stopping's and ModelCheckpoint's state (`callbacks.py:150-156`,
    `:233-235`).
  - ModelCheckpoint writes nothing in an epoch without its monitored metric (`:240-246`).
- **Training turns `auto` into CPU silently** (`train_model.py:742-745`).

**M3a-1 — the setup in effect, and one checkpoint payload.**
- **`config` gains the values in effect,** requested beside effective where they differ:
  - the loaders, per split, with the effective fan-out (the third hop of `num_neighbors` is cut,
    `src/kg/data_loader.py:206`), validation's negative count (the default 5, not the configured
    value) and `batch_size`;
  - compile: requested, and whether it installed without raising, not that it ran compiled;
  - AMP: requested and effective (on only on CUDA; bf16 without a scaler);
  - the schedule as built: its type, base rate, total steps, effective warmup and minimum ratio.

  The values recorded are today's behaviour, apart from the fixes this section commits to (the
  accumulation tail, the device rule, resume and `extend`). Changing any other is a separate
  behaviour change.
- **`producer`: the code revision and its dirty state.**
  - No `.git`, no git, or a git that refuses gives *unrecorded*: never guessed, never recorded as
    clean (§8 question 1).
  - Dirty counts modified tracked files and untracked files, under one explicit shared exclusion
    for generated data, caches and outputs.
  - Counts only, never file names.
- **`environment`:** §5.3, from one collector in `src/utils`, collected once at the start of the run.
- **One payload builder,** which ModelCheckpoint and `Trainer.save_checkpoint` both delegate to.
  Every metadata field is a primitive container or a scalar; model and optimizer tensors stay
  tensors. The whole payload loads with `weights_only=True`, as serving will (decision 10), tested
  on both writers' output, with no fallback to `weights_only=False`.
- **The resume state, taken at one consistent point:**
  - the model, the optimizer, the scheduler's definition and state, and the AMP scaler where used;
  - the completed epochs and the next epoch;
  - the counts of optimizer attempts, successful updates and scheduler steps, kept apart: they
    differ when AMP skips an update (`trainer.py:560-581`);
  - early stopping's and best-k's state, with what restoring the best weights needs;
  - the active stage, the original schedule or an extension (M3a-4): where it started, its target
    end, its progress within the stage and cumulatively, and its schedule's definition;
  - how the run ended: finished its stage, stopped early (with the reason), or paused.

  A stage is finished when its agreed budget of epochs is done, or by its recorded stop reason;
  never by the scheduler reaching an estimated step count. When AMP skips updates, a finished
  budget may leave the scheduler short of that estimate, and nothing runs extra steps or changes
  the rate to close the gap.
- **The accumulation tail:** one stated policy, under which the schedule's step count equals the
  updates actually taken. No gradient crosses an epoch boundary unrecorded. If the tail is applied
  at the epoch boundary, its loss scaling and the step count agree with that; clearing gradients
  without saying so is not a policy. Accepted on a case that does not divide.
- **Device:** `auto` without CUDA refuses before anything is written (§5.2). An explicit `cpu`
  stays available and is recorded.
- **The seed is applied before the data loaders and the model are built.** Today it is applied
  in `Trainer.__init__` (`trainer.py:255-257`), after the loaders (`train_model.py:748`) and the
  model (`train_model.py:751`), so the recorded seed does not govern initialisation.
  - **Implemented ahead of the rest of M3a (2026-10-11); code-reviewed with no P1 or P2 at
    `290c03a` (2026-10-12).** `train()`
    applies the seed with the Trainer's own rule (`apply_seed` in `src/training/trainer.py`:
    Python's `random`, NumPy and torch, every CUDA device included) before it builds the loaders
    and the model, so the same seed gives the same initial weights. `Trainer.__init__` still
    applies it, so the stream training draws from (negatives, shuffling, dropout) starts from
    the seed as before, whatever construction consumed. In `train_model`'s own entry, not a
    wrapper. It does not promise bit-identical GPU training or a restored random state on
    resume. Tested on CPU by `tests/unit/test_training_seed.py`.

**M3a-2 — one run directory per training run** (decision 9).
- **Every run, new or resumed, writes into a directory no run has used:**
  `{workspace}/runs/{conv_type}/{date}_{optional name}_{run_id}/`.
  - Its checkpoint, output and log directories are all inside it.
  - Directories are flat, never nested by lineage.
  - A colliding run id is drawn again, never reused.
  - The name is for people. A run is identified by its run id; a checkpoint by its SHA-256.
- **What is protected:** no run overwrites or deletes another run's files or its inputs.
  - A run updates its own `last.pt`, staged and then replaced.
  - It prunes only the best-k files it wrote. A restored best-k list never makes a parent's files
    deletable.
  - An explicitly given output path obeys the same rule, and a collision is refused before
    anything is created.
- **Every existing consumer moves in the same change,** through `checkpoint_paths`: the listing,
  the resume source, the reload's automatic selection, the training manager, the inventory, the
  probe and the deployment guides.
  - One resolver. No recursive scan written separately in the UI, the CLI and the API.
  - The selection's existing ordering is stated for runs. No new ranking across experiments.
  - The training console's free output fields give way to an optional run name.
- **The source and the destination are separate.**
  - An explicitly entered model path stays available, for the resume source and for serving.
  - It locates a file. What the file is, and whether it is accepted, come from its bytes (SHA-256,
    M2, M3), through the same checks as a listed file.
  - A resumed run still writes into its own new directory, and records the parent by its SHA-256.
- **The run listing (`run.json`) is a projection of the checkpoints' records, never an authority.**
  - Run-level fields: the run id, the parent checkpoint's SHA-256, the settings in effect.
  - One entry per checkpoint: its SHA-256, epoch, metrics, and whether it can be resumed.
  - A checkpoint never contains its own listing's digest.
  - A checkpoint published without its listing entry (an interrupted update) is shown as incomplete
    until an explicit refresh.
  - The listing never overrides a checkpoint's own record, and listing a file does not verify its
    weights.
  - Until these rules are implemented, the listing reads checkpoint metadata as today, with no
    second path beside it.
- **Legacy outputs are moved once, by the owner** (decision 9).
  - `scripts/migrate_checkpoints.py`, updated for this layout, lists a plan first and moves only
    with `--apply`.
  - Checkpoints M2 refuses go to `{workspace}/archive/`.
  - Ones whose M2 record checks pass go into an imported run, labelled "M2 record checks pass; no
    M3 resume state; servability not judged". M2's comparison reads the recorded roles only, not
    the weights (`src/utils/fingerprint.py:192-223`; the inventory says the same,
    `scripts/audit_checkpoint_family.py:424-429`). Whether a file serves is decided by serving's own
    entry point, never by the move.
  - A check the tool cannot complete is shown as *unverifiable*, apart from a real mismatch, and is
    never reported as proof that a model is invalid.
  - An imported run's id is assigned by the move and is not in the checkpoint's bytes; settings the
    checkpoint does not record stay unknown. Nothing is backfilled into a checkpoint.
  - A destination that already exists stops the move and keeps the source; the old tool's
    `--overwrite` does not carry over.
  - A move keeps the bytes, so every digest stands. Settings that name a path are listed for the
    owner to change.
  - There is no runtime fallback to the old layout. Whether a checkpoint is servable is decided by
    its records, never by where it sits.

**M3a-3 — resume continues an unfinished schedule** (decision 8).
- **The learning settings are inherited and locked:**
  - the learning rate, the scheduler, its warmup and minimum, and the planned number of epochs;
  - `batch_size` and gradient accumulation, which set the steps per epoch;
  - the optimizer and its weight decay;
  - the objective's settings: the loss weights, temperature, margin, label smoothing and negative
    counts.

  Operational fields may differ: the name, the output location, the pause limit, logging and the
  device. The UI shows the inherited values, locked. The CLI refuses a conflicting value instead of
  ignoring it.
- **The schedule is rebuilt from the parent's record,** then the optimizer's and the scheduler's
  progress is restored. Never a curve built from the child's settings with the parent's progress.
  A parent whose record cannot rebuild it is not resumed as a full restore.
- **Resume continues the active stage,** an unfinished extension included: only that stage's
  remaining epochs run, and `on_train_begin` does not clear the restored state.
- **A pause flag:** stop after epoch N of the run's total progress.
  - It stops at a safe boundary: validation and state updated, a resume snapshot written, the
    accumulation tail handled.
  - The snapshot is the latest training state, never the best weights paired with the latest
    optimizer. It is written in an epoch without validation too.
  - The run is recorded as paused, not finished.
  - It is separate from the planned horizon, and a child does not inherit it.
- **A parent that finished its active stage is refused for resume,** with the reason: that is
  `extend`'s case. A run that stopped early keeps its stop reason, and resume does not reset its
  patience to continue.

**M3a-4 — extend a finished schedule at a fixed rate.**
- **`extend` opens a new stage after a finished one,** from 100 epochs to 150, say, at a fixed
  rate. The stage records its start, its target end and its schedule; paused, it is resumed like any
  stage (M3a-3).
  - The model, the optimizer state and the cumulative progress continue.
  - No warmup, no recomputed curve, no cosine past its end, no OneCycle stepped past its total.
- **The fixed rate is each param group's effective rate after the restore,** recorded as the
  extension's starting value. An ordinary `extend` gives no rate.
  - A zero rate is never replaced by a guess.
  - Replacing a zero rate with a non-zero one takes an explicit rate, with the groups it applies
    to.
- **One code path:** the same entry point, trainer, loaders and writer.
- **An early-stopped run is not extended as if it had paused.**

**Not in M3a.**
- **A mode that changes learning settings from a parent (a branch) is not built in M3.** Until it
  is, resume refuses changed settings. It is a prerequisite of the GNN experiment phase, which
  starts only once the platform is stable, and is recorded as BACKLOG item 23.
- **Room left for a branch, without building it** (the reviewer's recommendation, 2026-10-15):
  - **Provenance checks and mode rules stay apart.** The checkpoint, graph and data checks are the
    shared contract. Whether settings must equal the parent's is `resume`'s and `extend`'s rule,
    never written into the checkpoint reader or the model builder.
  - **One decision before the Trainer.** The settings in effect and how the run restores are
    resolved in one place, before any output exists: which settings come from the parent, which
    operational fields from the request, and what happens to the optimizer, the scheduler, the
    progress and the stop state. The UI, the CLI and the probe all go through it.
  - **Restoring the weights, the optimizer and the scheduler are separate steps,** combined only as
    each mode allows: `resume` restores the active stage; `extend` keeps the model and the optimizer
    and opens a fixed-rate stage. No `strict=False` that bypasses both the settings rule and the
    provenance checks.
  - **The checkpoint's record is versioned,** and says enough of a run's history: the parent's
    digest, the settings in effect, the active stage, the schedule, the progress and the stop
    reason. No empty branch fields, stub or plugin system.
  - **One loader, Trainer, checkpoint writer and run-directory manager for every mode.** Only
    `new`, `resume` and `extend` exist; a branch request is refused, with no hidden entry point.
- **Tags, filters and a lineage tree** come with M4/M5's model list.

**Acceptance for M3a.**
- **Run directories:**
  - two new runs with the same settings get different directories;
  - a child leaves its parent's checkpoints, settings, metrics and logs unchanged;
  - a run updates its own `last.pt`, and best-k never deletes a parent's file;
  - a colliding output is refused before any run artifact exists.
- **Consumers:** a new run's checkpoints are listed by the existing UI, selectable as a resume
  source, and found by the reload's selection.
- **Listing:** a run with several checkpoints shows each one's epoch and metrics; a stale listing or
  a replaced file is not shown as verified.
- **Resume, on a small controlled input:**
  - an uninterrupted run and a paused-then-resumed run give the same learning rate at every step,
    through warmup and into decay;
  - the optimizer, the scaler, early stopping and the progress are restored;
  - an accumulation that does not divide, and an update AMP skipped, keep the schedule's position.
- **Ends:** a finished schedule refuses resume with the reason; `extend` continues at a fixed rate
  with no rise and no warmup.
- **An extension paused and resumed:** an original stage of two epochs finishes; `extend` adds two;
  the extension pauses after one; resume runs only the remaining one, with the rate, the optimizer
  state and the cumulative progress correct and the original curve not replayed.
- **The probe** keeps M2's data and graph refusals. Its resume case becomes a real pause and resume:
  plan two epochs, pause after one, resume the remaining one. `extend` is tested apart.
- **Not claimed:** bit-identical training with and without a pause. Random state and sampling are a
  separate layer, described apart.

*How M3a's acceptance is run (prepared 2026-10-15, after the re-review):* through the same
deployment probe and production entry points as M2's, which now carry M3a's checks
(`scripts/probe_deployment.py --m2-acceptance --device cuda`); no other training pipeline.
- **Locked:** the commit the probe records (the branch head at the run); the inputs, the same as
  M2's acceptance and checked by SHA-256 against its evidence (`hpo_2026_0929_5a`, its
  `train_outputs/config.yaml`, `checkpoints/hgt/model-02-0.1813.pt`, given by path); the allocator,
  the saved preset `cuda_async`, applied by the probe before torch's first import and recorded in
  the report. A new work directory and report outside the repository, kept as evidence.
- **Evidence layout (the reviewer's arrangement, 2026-10-15):** a new evidence root per attempt,
  `/home/user/m3a-acceptance-20261015-01/` (then `-02`, `-03`, every attempt kept, failed ones
  included), holding `checkout-before.txt`, `input-sha256.txt`, `console.log`, `exit.txt` and
  `work/`. `work/` is `--work-dir` and must be absent or empty when the probe starts, so nothing
  else is put in it first; the report is `work/m3a_report.json`. The real workspace and its table
  are read in place, not copied; the older checkpoints are not moved first; earlier M2, S10 and
  diagnostic evidence is not cleared. The checkout is restored afterwards to what
  `checkout-before.txt` records (`main` at `9fe5113` when the review looked), not to an older
  example.
- **What M3a's part shows:** demo M2, a run with mixed precision on CUDA paused after one of two
  epochs and resumed for the remaining one, its gradient scaler restored (`scaler_restore`: the
  scaler's state right after the production `Trainer.continue_from`, before the first update,
  equal to the parent's `scaler_state_dict` from the same read; the scale and growth tracker at
  the ends are recorded and not judged, since an overflow and a full growth interval can bring
  the scale back to where it was); M9, the finished child extended by one epoch at a fixed rate,
  its scaler checked the same way; M10, resuming it refused before anything exists;
  real-data R3 and R4, one epoch of two paused and its remaining epoch resumed; every run in its
  own run directory under the work directory; M2's refusals and checks as before. The older
  `config.yaml` names `output_dir`, `checkpoint_dir` and `log_dir`; the probe drops them as
  locations (R2 records the fields it sets).
- **The probe's scaler check, reviewed (2026-10-15):** the first version (`7e1e373`) judged the
  scaler from its end values and would have refused a correct restore whose scale fell and rose
  again (the reviewer's P2, reproduced with real `GradScaler`s). It now observes the restore
  through the production `Trainer.continue_from`, wrapped by the probe for the run's duration;
  no other loader, Trainer or training mode, AMP unchanged. Tested with the reviewer's
  counterexample (not refused) and with a restore that leaves the scaler fresh (caught), the
  latter end to end through the Trainer on the CPU with a real CPU `GradScaler`.
- **Round 01 (2026-10-15, at `4b1571e`; evidence `/home/user/m3a-acceptance-4b1571e-01/`,
  reviewed by the reviewer):** the probe ran to the end and wrote its report; 59 passed, 5 failed,
  0 errors, 0 skipped; exit 1, acceptance not passed. **Real data R1–R8 all passed** and stand as
  that version's real-data functional evidence: R4 recorded `scaler_restore` verified at the
  restore (the parent 2500 attempts and 2498 updates, the child 5000 and 4997 cumulative: one
  remaining epoch); R5 measurement, R6 serving with the table verified and a diagnosis, R7 the
  inventory, R8 the inputs unchanged (listing, size and modification time; the reviewer also
  re-hashed the config and `model-02`, both unchanged); allocator from the saved preset, set
  before torch's first import, `cudaMallocAsync`. **The five failures are one wrong assertion in
  the probe and its four dependants:** demo M2 required the child's successful updates to be twice
  the parent's, and AMP's scaler had skipped one update in the parent's epoch (8 attempts, 7
  updates) and none in the remaining one (8 and 8), so a correct resume (16 attempts, 15 updates,
  15 scheduler steps, completed 2/2, finished) was refused; M3, M4, M9 and M10 then failed for
  want of the resumed checkpoint, which exists. Not four product defects. The product's counts
  separate attempts from successful updates, as designed.
- **The fix (probe and tests only):** "only the remaining epoch" is counted by optimizer attempts:
  the child's grow by `ceil(batches / accumulation)` for each epoch its parent had left, from the
  parent's recorded schedule and train loader; its successful updates grow, by no more than its
  attempts (AMP may skip); its scheduler steps grow by its successful updates. The check passes on
  round 01's demo checkpoints (8 attempts, 8 updates, 8 scheduler steps in the child's run) and
  is tested with a skipped update in the parent's epoch and in the child's, end to end through the
  Trainer with a real CPU `GradScaler`, and with a restore that replays the curve from epoch 0
  (refused). Not a warning, not dropped; AMP, the scale, the seed and the Trainer's counts
  untouched. The R phase's code is unchanged.
- **Round 02, the demo part only (A–E and M), at the fix's commit:** the product code is the same as
  round 01's (`git diff 4b1571e..<the fix> -- src scripts/train_model.py` is empty; the probe
  changes only M2's check), so the real-data part's 43 minutes are not rerun. The same probe
  command without `--real-*`, in a new evidence root (`/home/user/m3a-acceptance-<commit>-02/`,
  the layout above). **It ends `M2 acceptance, real data: not requested in this run`, `M2
  acceptance: NOT COMPLETE` and exit 1 by design**, and is never accepted on its exit alone nor
  presented as a complete single run. It is read by the report: no probe failed or errored;
  `m2_acceptance.demo.complete` is true and `m2_acceptance.real_data.requested` false; every
  required demo probe passed (M2, M3, M4, M9 and M10 among them); and the exit's only reason is
  the real data not requested. Round 01's evidence root is kept as it is, neither reused nor
  cleared. M3a's functional acceptance is the two rounds together, recorded with their commits,
  their scopes and the statement that the product code is the same; it is complete when the
  reviewer has reviewed round 02's evidence.
- **The fix, reviewed (2026-10-15):** the reviewer's review of `4b1571e..c4a21b5` closed the P2
  with no new P1 or P2, and applied the new check offline to round 01's checkpoints (CPU,
  `weights_only=True`, no training): demo, one epoch, 8 attempts, 8 updates, 0 skipped, 8
  scheduler steps; real data, one epoch, 2500 attempts, 2499 updates, 1 skipped, 2499 scheduler
  steps; both pass. That is a check of the rule on recorded counts, not a rerun of the demo's
  measurement, serving or extension, which round 02 supplies.
- **Round 02 (2026-10-15, at `c4a21b5`; evidence `/home/user/m3a-acceptance-c4a21b5-02/`,
  reviewed by the reviewer):** 56 passed, 0 failed, 0 errors, 0 skipped; probe exit 1, log exit 0;
  `m2_acceptance.demo.complete` true, `m2_acceptance.real_data.requested` false, overall
  `complete` false, its only reason the real data not requested, as agreed. The tested commit
  recorded, `tracked_files_modified` false; CUDA, allocator `cudaMallocAsync` from the saved
  preset, set before torch's first import. What the reviewer checked from the report and the kept
  checkpoints, without training again:
  - **M2:** the parent completed one epoch (8 attempts, 7 updates), the child two cumulatively (16
    attempts, 15 updates): exactly one more epoch; the scaler verified at the restore; the AMP
    skip no longer misread.
  - **M3:** `measurement.json`'s digest matches the report, and the checkpoint it records is M2's
    child.
  - **M4:** on CUDA, GNN with shortest paths, `sp_kg_binding` verified, `max_hops` 5, diagnosis
    candidates returned.
  - **M9:** the parent is M2's child; the extension completed three epochs; both groups' rate
    `2.0816937636766182e-06`, the parent's last; `scheduler_steps` stayed 15 while
    `optimizer_updates` went from 15 to 23.
  - **M10:** resuming the finished run refused, no run directory created, the parent and the
    workspace unchanged.
  - Every other probe of the round passed; no required probe hidden behind a failure.
  - Checkpoints (SHA-256): paused `407538365c9c019a14e1075690ad864d9dccdf3e2cd6baa10e86a75a8045813e`,
    resume_ok `42a4ee41f75c233195dbe6b086b48f96b5d06debda0d0f469be570aad9d54442`, extended
    `ed56d77626477377d09b81640dcf1d3ea58687507158664af74a49c06dac8dc9`; every parent digest a
    child records matches its file.
- **The two rounds together (the reviewer, 2026-10-15):** `src/` and `scripts/train_model.py` are
  the same at both commits; what changed between them is the probe's M2 check, its tests and the
  documents. Round 01 passed real data R1–R8 (its report SHA-256
  `4410f6b7bcd79ebdddb863ce1cf9ad4fcfc9151d1a473b764e2d1a56202a3397`; its demo checkpoints
  `15e16fd332660eb90c40cd41f5bf393f8bd425e019f9efa3d75637ce571ea5a7` and
  `3b032a435ef2a9f6dd0d5f29c279c76a78fb42ed4f2143cd7ea24b808a94b2ad`); round 02 supplied A–E and
  M, M2, M3, M4, M9 and M10 among them, which round 01's wrong assertion had blocked. Together they
  meet M3a's functional acceptance as agreed, with no new P1 or P2; no rerun of the real data or
  another identical round is needed. Both rounds' raw reports are kept as written; neither is
  rewritten as a complete single run. Not included: capacity, model quality, clinical benchmarks,
  M3b and M3c. The two non-blocking P3 stay tracked as recorded above.
- **The homelab checkout restored (2026-10-15, done):** by the round 02 runbook's last step, to what
  `checkout-before.txt` records; the reviewer checked it directly, `main` at `9fe5113` with the
  tracked files clean, and round 02's `checkout-restored.txt` is kept. The review's note that the
  checkout was still detached at `c4a21b5` is the state before the restore; nothing is rerun.
- **Not part of it:** moving the workspace's older checkpoints into the run layout. That is the
  owner's, with `scripts/migrate_checkpoints.py` (a plan first, `--apply` to move), and the
  acceptance does not need it: the reference is read by path.
- **Not claimed:** capacity, model quality, identical weights with and without a pause.
- **Complete** when the evidence (beside M2's, with digests) is reviewed and every required probe
  passed; until then M3a's functional acceptance is not passed.

**M3a as implemented (2026-10-15).** Code-reviewed at `094b90d` with three P2 and no P1 (below,
"The review of `094b90d`"); the re-review of `094b90d..815d763` closed them with no new P1 or P2.
The GB10 acceptance follows (above). What differs from the text above, or was decided while
building, is stated here.

- **M3a-2, run directories.** `src/utils/checkpoint_paths.py` names (`propose_run_dir`) and claims
  (`claim_run_dir`, by creating `checkpoints/`) a run's directory; the training entry point claims
  it after its last refusal, the parent's schedule check included, so a refused run leaves nothing.
  `output_dir`, `checkpoint_dir` and `log_dir` are retired: a configuration file, CLI flag or API
  request that still sets one is refused, not ignored; `--run-dir` names an explicit directory,
  which must be absent or empty. The training service names the directory before launching and
  reads the run's logs from it; the console's two output fields became an optional run name, and
  Status shows the directory. The listing, the resume source, the reload's automatic selection
  (latest run, then the existing in-run order), the inventory, `setup_demo` and the probe use the
  same functions; the old resolvers are removed. The sample generator's guard covers `runs/`.
  - **`run.json` is not built.** As the text above allows, the listing reads each checkpoint's own
    record with the safe loader when it is listed, which deserialises each file whole (weights
    included); nothing is cached, and a listed file is not shown as verified. Listing without
    loading the model files is not claimed.
  - **The one-time move** (`scripts/migrate_checkpoints.py`) follows decision 9: plan first,
    `--apply` to move; M2's serving comparison passes -> an imported run labelled "M2 record checks
    pass; no M3 resume state; servability not judged" (in its `imported.json`); refused ->
    `archive/`; not completable -> left in place as unverifiable. It renames within the workspace,
    checks each digest after, stops before anything moves when a destination exists, deletes
    nothing, and lists saved settings that name a moved file.
- **M3a-3, resume.** One decision before the Trainer, `src/utils/run_plan.py`, which the entry
  point, the training service and the console's preview call.
  - **One parent field for resume and extend** (`resume_from`, `--resume`): what happens follows
    the parent's record (unfinished -> resume; finished with `extend_epochs` -> extend; stopped
    early, weights only, no M3a record -> refused with the reason). This replaces the earlier
    draft's separate `--extend-from`, after the reviewer's interface recommendation forwarded by the
    owner (2026-10-15): one continue entry whose action is read from the checkpoint.
  - **Learning settings** are every `TrainConfig` field outside `OPERATIONAL_FIELDS` (`data_dir`,
    `config_file`, `run_name`, `run_dir`, `resume_from`, `device`, `num_workers`, `compile`,
    `pause_after_epoch`, `extend_epochs`, `extend_lr`). They are inherited; one **set explicitly**
    to another value is refused. Explicit means a configuration file's key or a CLI flag given; the
    API passes on only the fields a caller set (`exclude_unset`); a library caller that builds
    `TrainConfig` itself counts a field that differs from its default. Mixed precision that would
    take effect differently on the requested device is refused as a settings change.
  - `Trainer.continue_from` replaces `restore_checkpoint`; the rebuilt schedule must equal the
    recorded one before the scheduler's progress is loaded. Early stopping's best-weights file is
    read once and accepted only with its recorded SHA-256. A parent's best-k files are ranked but
    never deleted, and every best-k file is named relative to the run's checkpoint directory.
  - `last.pt` is written every epoch, with or without validation. `--pause-after-epoch N` records
    `end: "paused"`. **Random state is not restored:** a continued run seeds from its inherited
    `seed`, so its remaining epochs draw the stream the first epochs drew. Not bit-identical, as
    the text above says.
- **M3a-4, extend.** The fixed rates are decided in the plan from the parent's optimizer record, so
  the preview shows them; no scheduler is stepped. A rate that ended at 0 is refused until
  `--extend-lr` is given, which replaces only zero rates and is refused otherwise.
- **Stop.** An operator's stop is reported as `stopped`, with the latest resumable snapshot from its
  own record or that none was saved, never as `completed`; a stop no longer runs the final
  validation, so a stopped run writes no `final_metrics.json`. The console asks before stopping and
  says what is lost.
- **Console.** One "Continue Training" block: a listed or a typed checkpoint through one check;
  the action, the check result and the inherited settings shown read-only before start (Markdown,
  not disabled inputs); extra epochs offered only for a finished parent; only operational fields
  sent. "Pause after epoch N" stays CLI and probe only. UX sources consulted: Gradio's docs, W3C
  WAI, GOV.UK Design System, Carbon, Nielsen Norman Group.
- **Probe.** Demo M2 is a real pause and resume (plan two, pause after one, resume one); M9 extends
  the finished child at a fixed rate; M10 refuses to resume it. Real-data R3/R4 pause and resume
  the same way.
- **Where the acceptance is tested:** run directories and consumers in
  `tests/unit/test_checkpoint_paths.py`, `test_training_api.py`, `test_training_console_resume.py`;
  resume, ends and the paused extension in `tests/unit/test_continue_training.py` (the learning rate
  at every update, the optimizer step count, the scheduler position, early stopping, best-k, the
  refusals); the move in `test_migrate_checkpoints.py`; the probe in `test_probe_m2_phase.py`. An
  update AMP skipped keeps the schedule's position (`test_accumulation_tail.py`) and resume restores
  the scaler; a pause and resume under AMP runs only on CUDA, in the GB10 acceptance.
- **The review of `094b90d` (2026-10-15): three P2, no P1,** each reproduced by the reviewer on a
  synthetic workspace through the production entry points; the fixes reuse the shared run plan,
  readers and writer.
  - **P2-1, a best file's weights lost after one generation.** A best-k file is written before it
    is listed, so its own record says its weights are the best (`{"kind": "self"}`) with a best-k
    list without itself; a child continued from it could not name those weights and a grandchild
    lost them. The run plan now carries such a parent as a best-k entry of its own run, named by
    the digest of the read already made (`parent_sha256`), so every later checkpoint names it and
    the next generation checks it like any best-k file. A record whose best weights must be
    restored and cannot be located is refused before anything exists.
  - **P2-2, a hidden "Extra epochs" value blocking a resume** in the console. A change of the
    chosen checkpoint or its source clears the field and the last check; the console sends the
    value only for a finished checkpoint (the service's `extend_if_finished`, the console's alone).
    An explicit `extend_epochs` for an unfinished parent is still refused everywhere else.
  - **P2-3, the API checking a continuation's cross-field rules on new-run defaults.** The rules
    are one function (`routes/training.cross_field_errors`); a new run's request is checked with
    its defaults, a continued run's on the settings the run plan resolves from its parent, inside
    the training service (422). Explicit conflicts are refused as before.
  - **One verification round after the fixes** (four read-only verifiers, each finding put to a
    skeptic) confirmed no P1 or P2. Of its P3 findings, these were fixed: a Device change now
    clears the last check; a Refresh or conv_type change no longer clears a typed path's check
    (the run list clears only when it is the source); Continue on a finished checkpoint shows
    Extra epochs and the hint; the API refuses `extend_epochs` without `resume_from` (422); the
    preview's settings for the cross-field rules are its learning settings only. Two P3 left as
    they are; the re-review keeps both as P3, not blocking:
    - **Continued from a best-k file that is not early stopping's best epoch**, the parent's own
      file is not ranked in the child's best-k list, which is the one the parent recorded before
      it was written (it can name a file the parent's run pruned right after). As before
      `094b90d`; no weights are lost, since foreign files are never deleted and the best weights
      are named and checked. Continued from the run's last.pt, the ranking is a run never
      interrupted's. `ModelCheckpoint.restore_foreign` now says so instead of claiming the
      latter in every case. **This is a real limitation, tracked (BACKLOG item 15), not fixed by
      the docstring:** the re-review reproduced it (scores 0.9, 0.1, 0.8, 0.5, top-2, continued
      from the 0.8 file: the child ranks 0.9 and 0.5 and leaves out the 0.8 file, which still
      exists; the best weights are still named and restored exactly, and the parent is
      unchanged). It is not claimed that every best-k checkpoint restores the full top-k
      candidate list. To complete it later: the run plan carries the chosen parent as a
      candidate, and the best-k restore rebuilds the list by the same pruning rule; no index,
      scanner or management system.
    - **A continuation that repeats the parent's value for a field outside the API's range for a
      new run** (a CLI-trained `hidden_dim=18`, say) is refused by that field's own range check;
      omitting the field inherits it. Each field's own type and range apply to every request, as
      the reviewer asked to keep. These are two existing boundaries, kept apart: the API's range
      for an explicitly given value, and the inheritance of an existing model's settings; a legal
      model pair such as 18/3 does not make `hidden_dim=18` an API input for a new run. Nothing here
      claims that the API's and the CLI's value ranges are unified.

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

**M3b as implemented (2026-10-15).** Code-reviewed with no P1 or P2 (below, "The review of
`04ecf42`"); the GB10 functional acceptance follows ("How M3b's acceptance is run"). What
differs from the text above, or was decided while building, is stated here.

- **One check, after the graph check.** `src.kg.artifacts.verify_dataset_record(manifest,
  kg_digest)` takes the run's one manifest reading and the `kg` digest its graph check returned
  (`verify_graph_reads` or `verify_graph_source_read`), and reads `kg.provenance.json` beside that
  manifest once (`read_once`): the digest of the bytes is compared with the manifest's binding
  (`artifacts.provenance`), the same buffer is decoded and parsed, the record's own rules are
  applied, and its `kg_digest` is compared with the run's. It returns a `DatasetRecord` (the
  record, the identity of its bytes, binding `verified`, its origin) or raises
  `DatasetRecordRefused` carrying the binding state: *unrecorded* (the manifest binds none),
  *unverifiable* (bound and not here, not readable, or not a record this revision reads),
  *mismatch* (not the bytes bound, or a record of another graph). A bound `files` record that
  leaves a role unidentified is refused with its binding *verified* and its origin `files`. A
  `kg_digest` that is not the one this manifest reading binds is a caller's pairing error
  (`ValueError`), not a state. The status readers that reported and never refused
  (`workspace_provenance_status`, `provenance_status`, `read_provenance`, `verify_provenance`) are
  removed, with their static-pin entries (BACKLOG 19.14, 19.15); no production code called them.
- **The record's rules in one place** (`src.kg.provenance.validate_record`, `unidentified_roles`,
  `require_identified`): an object of schema 1 (the integer) with the required fields,
  `incomplete_by_design` among them (every writer has written it); a SHA-256 `kg_digest`; an
  origin of `files` or `synthetic`; a list of sources, each exactly what `source_entry` builds (its
  role, its file's basename, a lower-case SHA-256 hexdigest, its declared version, nothing else),
  no role twice; none for a synthetic origin; `missing_roles`, where present, equal to the roles the
  sources leave out (every role for a synthetic record); `incomplete_by_design` a list of
  statements. The writer applies the same rules before it writes (`build_provenance`, which stores
  each source as `source_entry` builds it, and `write_provenance`), so a `files` record with an
  unidentified input, or one that contradicts itself, is never written; the workspace writer runs
  them before its first byte. A parse that fails in any way, a nesting past the parser's limit
  included, is *unverifiable*; a refusal names one of `BINDING_STATES` or is not made.
- **Consumers.** Training (`read_training_inputs`, at the top of the run, before the run directory),
  measurement (`read_measurement_inputs`, both cohort kinds, before the checkpoint is read) and
  serving (`DiagnosisPipeline`, before the tensors are adopted and the checkpoint read) call it
  after their graph check. The inventory (`audit_checkpoint_family.py --inventory`) lists the
  record's verdict with its binding state and origin as separate fields, and judges no checkpoint
  servable over a refused record; the one-time move (`migrate_checkpoints.py`) leaves a refused
  workspace's files in place as unverifiable. The reload route lists `kg.provenance.json` among the
  required files, as it lists the manifest. Path-only serving, with no checkpoint, verifies no
  workspace and is unchanged (BACKLOG 19.20).
- **Shown apart, never as plain verified.** Training and measurement log the record's binding and
  origin. Serving publishes `dataset_record` in `get_pipeline_config()` and the API's pipeline
  status (an additive field; the Diagnosis tab is unchanged until M3c's served identity): binding,
  origin, the digest of its bytes, the sources it names and `incomplete_by_design`. The measurement
  manifest gains the same summary as `dataset_record` (§5.1), outside the measurement's semantics
  digest, since no mode computes from it (`NON_SEMANTIC_FIELDS`), and its console summary prints
  the binding and origin, with "SYNTHETIC WORKSPACE" for a synthetic one. The checkpoint's record is
  unchanged (§5.1): the manifest it records binds the record's digest.
  - **Stated exception: the evaluation ledger.** `record_evaluation.py`'s ledger record and
    `--show` do not carry the origin; the ledger is item 14's ledger v2 (§5.1, M4), and the
    measurement report it is filed from carries `dataset_record`. Until then a ledger row does not
    say whether its workspace was synthetic.
- **The manifest binding is required at write.** `generate_training_samples` refuses a persisting
  call whose `graph_digests` lack `provenance` (or carry a malformed one) before anything is
  written, as it refuses an unbound graph role. The schema stays 3: the field was already written
  by every workspace this writer produced since the record existed; one without it is refused as
  unrecorded at read, with the instruction to rebuild (decision 2).
- **The writers' digests are of the bytes written** (BACKLOG 19.16, 19.17). `DigestingFile` and
  `write_digested` (`src/utils/fingerprint.py`) hash what a serialiser writes as it passes, so
  nothing is held to be hashed later. `KnowledgeGraph.save_json` returns the digest of `kg.json`'s
  bytes (encoded in pieces of about a megabyte; the same bytes `json.dump` wrote, measured by a
  test); `write_graph_data` writes the three tensor files and returns theirs, and
  `export_graph_data` calls it; `write_provenance` returns the digest of the record's bytes. The
  workspace writer binds these and hashes no path. One consequence: `torch.save` into a stream
  names its archive `archive/` rather than after the file, so a tensor file written now differs
  in bytes from one written before by the same content; every digest is still of the bytes on disk,
  and two builds by this revision still agree (the probe's F2).
- **The builder identifies every input, or stops.** An ontology the build parsed with neither the
  loader's digest of its bytes nor a path stops the build before anything is written; it was
  recorded as a missing role. The loader always gives both; the fallback that hashes the path for
  an ontology the loader did not produce, and the annotation files' path digests, are unchanged
  (BACKLOG 19.13, the KG record v2).
- **Fixtures.** The generated-workspace fixture writes a synthetic record through the production
  writer and binds it, so every workspace built on it is fully bound and synthetic; no test skips
  the check. Tests that need a `files` record build one from files: real test inputs in the build
  tests, and written stand-ins with their own digests in the probe's real-data test, which says it
  is a stand-in (its graph is the in-memory demo).
- **Internal review before delivery** (2026-10-15, four lenses and an independent check of each
  serious finding): no P1. One P2, the deployment guides (§4.4, both languages) still describing
  the record as reported and not enforced, rewritten. P3s fixed: the writer-refusal tests for
  malformed digests had become refusals for an unbound record (now each reaches the shape check
  and names its reason, with malformed record digests added); measurement's console did not name
  the origin; source digests were checked for length only, a non-path filename escaped as
  `TypeError`, `schema_version` `true` or `1.0` read as 1, a record without
  `incomplete_by_design` was shown as establishing everything, and a caller's absolute path or
  extra keys could reach the published summary (all now refused or normalised); one record failure
  no rule names could stop the inventory; R5 and R6 compared only the binding and origin (now the
  whole record the workspace check verified). The ledger is the stated exception above.

**The review of `04ecf42` (the reviewer, 2026-10-16; read only, range `2049b64..04ecf42`):** no
P1 or P2; the GB10 functional acceptance may proceed. No unneeded parallel pipeline, old-format
admission or CPU fallback was found. What the review established, in short:
- the order and the reads: `verify_dataset_record` uses the caller's one manifest reading, checks
  the given `kg` digest is that reading's, reads the record once (identity and parse from one
  buffer), compares the binding before parsing, and refuses every other state; a `files` record
  missing a role is refused with its binding kept as *verified*, and a synthetic record passes the
  same binding checks, not a skipped one. Training, measurement (both cohort kinds) and serving
  call it after their graph check and before a run directory, a checkpoint or the tensors; the
  inventory and the move call the same verifier; a refused reload leaves the served pipeline;
- the writers: one rule set for writer and reader (`validate_record`), digests taken while
  serialising, one graph-export writer, the binding required before the manifest is written, and
  no record invented for an older workspace;
- the boundaries kept: path reasoning without a model and an in-memory graph and model handed in
  by a program are existing interfaces that claim no persisted workspace, and a named checkpoint
  that fails the checks still refuses; M3b is not an atomic workspace transaction and does not
  show the sources are the releases an institution chose; the builder's annotation path digests
  stay BACKLOG 19.13 (the KG record v2); the ledger is M4's. The tensors' new archive name means a
  rebuild may differ in bytes for the same content, and no consumer admits a digest mismatch
  because content looks the same;
- tests: 823 relevant tests passed in a separate checkout on the homelab with CUDA disabled (not a
  CUDA acceptance), import-linter 4 kept, `git diff --check` clean. One environment note: the
  serving test file stalled inside the reviewer's sandbox at FastAPI's `TestClient.__enter__` and
  passed outside it (22 tests, 2.92 s); the 823 is the union of completed files, not a claim for
  the aborted batch.
- **One P3, synced:** `PLAN_PHENOTYPE_NORMALISATION.md` and BACKLOG item 16 still made N2 ride on
  an M3b manifest change and rebuild that M3b did not make. They now record N2 as not implemented
  by M3b and not yet scheduled, its own change deciding how its manifest field is introduced; no
  code, rebuild or compatibility path was added for it.
- **The real workspace, checked read-only before the run** (the reviewer, at `04ecf42`, the
  production `inventory_workspace`, CUDA disabled, on
  `/home/user/Desktop/SHEPHERD-Advanced/data/workspaces/hpo_2026_0929_5a`): graph accepted;
  training inputs accepted; `dataset_record` accepted, binding *verified*, origin `files`;
  `kg.provenance.json` SHA-256 `2c7abdb6660cbbbbcb7579e74797a9bfbcd775114b7b9b93c8de640175e412ef`;
  `split_manifest.json` SHA-256
  `131cdf81cfa20ab8b44609be78473d6dce3f3a18d4436fafd589c6221ddcf664`; the record's `kg_digest`
  `6cae2d1a58690eec9aa9c5e3ca9182c2fc942db0f468a5123983251bd4c51e43`; all four roles, both
  ontologies declaring `2026-09-01` (read from the record, not re-checked at the source); the
  table's source binding accepted and its serving not judged (the tensor is not loaded); its four
  older checkpoints refused by M2 for want of a `kg` role, as before. So the workspace needs no
  rebuild for M3b. What the record says it does not establish (imports, parser and builder
  versions, construction parameters) stays unestablished and is not read as verified.

*How M3b's acceptance is run (prepared 2026-10-15; confirmed by the reviewer, 2026-10-16):* through
the same deployment probe and production entry points (`scripts/probe_deployment.py
--m2-acceptance --device cuda`), with the inputs, allocator and evidence layout of M3a's
acceptance; no other runner. One run, demo and real data together.
- **Locked:** the commit, the branch head that records this review, documents only over `04ecf42`
  (`git diff 04ecf42..<head> -- src scripts tests` empty); the inputs, as M2's and M3a's:
  `hpo_2026_0929_5a`, its `train_outputs/config.yaml` (SHA-256 `14bf8eea…970e`), its
  `checkpoints/hgt/model-02-0.1813.pt` (SHA-256 `33a7b39a…8b79`), given by path, with the
  record and manifest digests above; the allocator, the saved preset `cuda_async`, applied by the
  probe before torch's first import.
- **Evidence:** a new root per attempt, `/home/user/m3b-acceptance-<commit>-01/` (then `-02`, …,
  every attempt kept), holding `checkout-before.txt`, `input-sha256.txt`, `console.log`,
  `exit.txt`, `checkout-restored.txt` and `work/` (`--work-dir`, absent or empty at the start;
  the report `work/m3b_report.json`). The real workspace and its table are read in place; earlier
  M2, M3a, S10 and diagnostic evidence is not touched. The checkout is restored afterwards to
  what `checkout-before.txt` records.
- **Read by the report, not by the exit alone:** no probe failed or errored;
  `m2_acceptance.demo.complete` and `m2_acceptance.real_data.complete` true; the tested commit
  recorded with `tracked_files_modified` false; CUDA, `cudaMallocAsync` from the saved preset.
  What M3b's part shows:
- **Demo:** C1, the demo workspace's record verified and synthetic; C5, each way a record can fail
  to bind refused with the state it establishes (missing → unverifiable; another workspace's
  record → mismatch; another graph's record rebound → mismatch; the binding removed →
  unrecorded; a bound file that is not a record → unverifiable; a bound `files` record leaving an
  input unidentified → refused, binding verified); B22 and B23, the writer refusing a manifest
  without the binding and a source list leaving an input unidentified; M3 and M4, the measurement
  manifest and the served configuration showing the record as verified and synthetic; M11,
  training, serving and measurement each refusing a workspace whose record is missing, with
  nothing written and the served pipeline unchanged. All are in `M2_DEMO_REQUIRED`.
- **Real data:** R1, the real workspace's record verified with origin `files` (its SHA-256 the
  pre-check's `2c7abdb6…12ef`), every source named with its digest and declared version; R5 and
  R6, the measurement manifest and the served configuration carrying that same record whole
  (equal to what R1 verified, digest and sources included), R6 with the table verified by the
  production loader; R2–R4 and R7 as in M2's and M3a's acceptance; R8, the inputs unchanged by
  listing, size and modification time, which is not a content hash of the workspace. F1 is not
  part of this run (no fresh real build is requested).
- **Not shown:** that a record names the right releases; the record identifies the bytes a build
  consumed, and whether those are the intended releases is the operator's to read from it. Not
  part of it: capacity, S10, the memory attribution, AMP comparisons, HGT Edit 2, model quality or
  a clinical benchmark.
- **Complete** when the evidence is reviewed and every required probe passed; until then M3b's
  functional acceptance is not passed.

**M3b's acceptance on the GB10 (2026-10-16, at `c80c1e4`; evidence
`/home/user/m3b-acceptance-c80c1e4-01/` on the homelab, its report, exit status and the
reviewer's cross-checks in [`m3b-acceptance-2026-10-16/`](m3b-acceptance-2026-10-16/README.md);
reviewed by the reviewer).** **Passed, within the scope agreed.** The report (SHA-256
`9038a235907ff03b18f4a153bffc64f79b36657b9e8264dfecce58c6246ef489`, generated
2026-10-16 13:05 +08:00) and what the reviewer checked, read only, without training again:
- **Run:** the tested commit recorded and equal to the checked-out head, `tracked_files_modified`
  false; probe exit 0, `tee` exit 0; 75 distinct probes, 75 passed, 0 failed, errored or skipped,
  the console's PASS lines the same 75; every required probe run and passed (demo 33, real data
  8, as read from the tested code's lists); `m2_acceptance` demo, real data and overall complete.
- **Device and allocator:** NVIDIA GB10, resolved device `cuda`, the real serving model's
  parameters on CUDA; backend `cudaMallocAsync` from the saved preset
  (`PYTORCH_ALLOC_CONF=backend:cudaMallocAsync`, set before torch's first import); the
  measurement reports record `cuda_executed` true.
- **M3b's behaviour:** C1, M3 and M4, the demo record verified and synthetic, the measurement and
  serving summaries the same record; C5, all six tamperings refused with the states expected
  (missing and unreadable → *unverifiable*; replaced and another graph's → *mismatch*; unbound →
  *unrecorded*; unidentified → binding *verified* and refused for the missing source, which is
  the probe's pass condition, not an admission); B22 and B23, the writer refusing; M11, training,
  serving and measurement each refusing a workspace without its record. R1, R5 and R6, the real
  record verified with origin `files`, all four roles, its SHA-256
  `2c7abdb6660cbbbbcb7579e74797a9bfbcd775114b7b9b93c8de640175e412ef` in all three; the reviewer
  also read the raw `measurement.json` and found its sources and `incomplete_by_design` equal to
  the real `kg.provenance.json`, not merely the same count.
- **The real short training, cross-checked:** parent `real/initial/checkpoints/last.pt`, one epoch,
  ended paused (SHA-256 `27d1fd4c6f69459d86d9fd93f2c1f8b95740d753181548bc2f87390ef7c06b9e`);
  child `real/resumed/checkpoints/last.pt`, two epochs, finished (SHA-256
  `4f8032198e5cf872476dd04548ea1b4ba8298dd7f4db476c7c73172a18d80c4f`). Read with the shared
  reader on the CPU (`weights_only=True`), these and the demo's parent and child have the digests
  the report gives; the child's `resume_checkpoint` names its parent and its `kg` role agrees. R5's
  measurement (15,000 samples) and the new-model inventory name the child, not the parent. R6:
  `gnn_plus_shortest_path`, the loader's `sp_kg_binding` verified, `max_hops` 5, five candidates.
  R7: `model-02` refused for its missing `kg` record, the child accepted by the source contract;
  the inventory still claims no SP tensor loaded and no model quality. The measurement and
  inventory artifacts' digests match the report. 33 cross-checks in all, kept in the reviewer's
  `cross-checks.json`.
- **Logs and inputs:** the four ERROR lines are the injected refusals (E5's damaged tensor, two
  lines; M6's crossed checkpoint; M11.serving's missing record). The GB10's capability 12.1 above
  the wheel's listed 12.0 is reported as a runtime notice; A4 and this run's training, measurement
  and serving completed on CUDA, which does not establish that every operation has optimal native
  support, and no compatibility or performance study is opened here. R8 compared 19 entries by
  listing, size and modification time, not a content hash of the workspace; the reviewer
  re-hashed the four named inputs (the config, `model-02`, `kg.provenance.json`,
  `split_manifest.json`), all equal to `input-sha256.txt`.
- **Not claimed:** capacity, AMP's effect, model quality or a hospital test; the short-training
  checkpoints are not a chosen serving model. The measurement is `cohort_kind` generated, split
  `val`: its numbers are not a MyGene2 or institutional held-out result. N2, M3c and M4 are not
  completed by this. The evidence and `work/` stay where they are; no rerun or rebuild is needed.
- **The homelab checkout restored (2026-10-16, the operator):** back on `main` at `9fe5113`, the
  state `checkout-before.txt` records; the reviewer's evidence package found `checkout-before.txt`
  and `checkout-restored.txt` byte-identical, the tracked files clean and the four named inputs
  rechecked equal. At the review it was still detached at `c80c1e4`, the state before the
  restore.

**M3c — what is served is identified (R8).** *Line citations in this section are at `cbba015`,
the base M3c was built on; the code that replaced them is cited in "M3c as implemented" below.*
- **The checkpoint's digest comes from M2.1's read** of the bytes loaded.
- **One structured field replaces the two version strings.** It replaces
  `InferenceResult.model_version` and `kg_version` (`src/core/types.py:468-469`) and
  `DiagnoseResponse.model_version` (`diagnose.py:200`). Its named parts:
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
    reload could swap. `app_state.model_version` goes (`src/api/main.py:89, 592-596`).
  - The literal versions go: `pipeline.py:1184, 1262-1263, 1901` (with the constant at `:327`
    and the configuration's `"version"` key at `:1920`) and `diagnose.py:221, 408`.
  - The Diagnosis tab's downloadable report has a "Model version" line
    (`src/webui/components/diagnosis_panel.py:568`, in the report builder at `:559`; the CSV's
    columns at `:489-493`). It carries the identity's parts instead, and so does the CSV export.
  - The response schema changes, and the milestone's documentation says so.
  - Every existing consumer — the API, the UI, the exports — moves in the same change. No
    consumer is left choosing between an old field and a new one.
- **Device.** Serving's `auto` without CUDA refuses (§5.2). Until M3c it turned into CPU silently
  (`pipeline.py:1085-1086`).

**M3c as implemented (2026-10-16).** Implemented on the branch over `cbba015`, internally
reviewed (below) and awaiting the reviewer's review; no GB10 acceptance yet. Line citations here
are at the head delivered for that review. What differs from the text above, or was decided while building, is
stated here.

- **One structured value, `ServedIdentity`** (`src/core/types.py:603`), frozen, with five parts
  and the scoring mode; `to_dict()` gives JSON primitives. Each part states its state in words,
  and a value is null only where that state says there is nothing to identify:
  - `checkpoint`: *verified* (the model built from one read of a checkpoint whose record matched
    the workspace graph, M2.4; `sha256` is that read's digest), *supplied_in_memory* (a model
    object handed in; no bytes, no digest) or *absent* (no model; path reasoning ranks);
  - `graph`: *verified* (`kg.json` and `split_manifest.json` each read once and the graph's
    digest matching the manifest; both digests from those reads), *unverified* (read from a
    file, its digest stated, checked against no manifest: path-only serving, BACKLOG 19.20) or
    *in_memory* (no file, no digest);
  - `shortest_paths`: *verified* (the `build_id` the tensor and its sidecar share, `max_hops`
    from the sidecar, M2.5) or *absent*;
  - `dataset_record`: *verified* (origin `files` or `synthetic` and the record's digest, M3b) or
    *absent*; verified exactly when the graph is;
  - `scoring_mode`: `gnn_plus_shortest_path`, `gnn_only` or `path_reasoning_fallback`;
  - `settings`, "the serving settings that change scores", enumerated: `device` (torch's name
    for where the model's parameters are, e.g. `cuda:0`), `device_requested` (as asked; `auto`
    when a checkpoint was loaded with none named), `eta` in effect (the configured value with a
    table, 1.0 without, null with no model), the path settings (`max_path_length`,
    `path_length_penalty`, `aggregation_method`, `max_paths_per_candidate`,
    `use_direct_path_optimization`; in effect in every mode, since candidates are
    path-discovered) and `phenotype_normalisation_version` (the rule's constant, 1).
  - Each part refuses a value its state does not allow (`__post_init__`), and the whole refuses
    parts that contradict each other: a model exactly when the mode is not path reasoning; a
    verified table exactly in `gnn_plus_shortest_path`; a verified record exactly over a verified
    graph; a verified checkpoint or table only over a verified graph.
  - **"A stated absence where B-2's policy allows serving without one"** has no policy to follow
    yet (§8 question 2), so the identity states path-only serving as it is: checkpoint *absent*,
    graph *unverified*, manifest, table and record *absent*, no device. Nothing in it is claimed
    verified, and M3c decides nothing about whether that mode stays.
- **Built from the pipeline's own reads, never hashed again.** `DiagnosisPipeline.served_identity()`
  (`src/inference/pipeline.py:2014`) reads the identity of the checkpoint read the model was built
  from, the bound graph digest and the manifest's identity from the one workspace reading (kept at
  `:686`), the accepted table's `build_id` (`:1020`), the verified record, and the device the
  model's parameters landed on (`:756`). Every `InferenceResult` carries it as a required field,
  built when the result is produced: a refused request's, a scored result and an empty one
  (`:1280`, `:1358`, `:1996`). `get_pipeline_config()["served_identity"]` reports it from the same
  method (`:2093`). `InferenceResult.model_version` and `kg_version`, the class constant `VERSION`
  and the configuration's `"version"` key are gone.
- **Copied, never taken from shared state.** `DiagnoseResponse.served_identity`
  (`src/api/routes/diagnose.py:302`) is required and copied from the result the response answers
  with (`:516`). It is null only for the demo answer given when no pipeline was ever configured
  (which keeps its warning): nothing was served, and a configured pipeline that failed still
  answers 503. `GET /api/v1/pipeline/status` carries the live pipeline's identity
  (`src/api/routes/pipeline.py:67`; null with nothing loaded) beside its existing fields, which
  stay, `checkpoint_meta.device` among them; this is the API read M4's display needs.
  `AppState.model_version` is gone; `app_state.version`, the service's own version, stays and is
  not the identity. The response models are mirrors defined in `diagnose.py` (`:282`), strict (an
  unknown key refused, every key required), because `measure_served_pipeline.py` loads that
  module by file without the application; `tests/unit/test_served_identity.py` pins the two
  definitions to one shape and one vocabulary of states.
- **The response schema change is documented** in both deployment guides (§4.7, with the CSV's
  column dictionary) and in the response model's descriptions and example, which Swagger shows:
  `model_version` removed with no replacement string, `served_identity` added.
- **Device** (§5.2). `resolve_serving_device` (`src/inference/pipeline.py:253`): unset, blank or
  `auto` means CUDA and refuses without it (`ServingDeviceUnavailable`, a `PipelineBuildError`,
  whose message names `SHEPHERD_DEVICE=cpu`); `cuda` and `cuda:N` refuse without CUDA; `cpu` is
  permitted and recorded as asked; another string goes to torch as given, as training's does.
  Without PyTorch it names PyTorch, not the CPU remedy. Applied whenever a checkpoint is to be
  loaded, after the PyTorch check and before any workspace input is read (`:619`); in
  `build_pipeline` before `kg.json` is parsed (`src/api/main.py:540`); in the reload route right
  after its file check, before automatic selection reads any candidate
  (`src/api/routes/pipeline.py:328`); and again before the model is moved (`pipeline.py:1173`). A
  model supplied in memory with no device named stays where it is; one with a device named gets
  the same rule (`:622`); path-only serving resolves no device. The device asked for is the
  request's, else `SHEPHERD_DEVICE`, blank counting as unset at both steps
  (`requested_serving_device`, `src/api/main.py:417`), passed on as given so the identity records
  it; a refused reload leaves the served pipeline. **The consequence for operators:** a CPU-only host that served with no device
  named, silently on the CPU, is refused now; development hosts set `SHEPHERD_DEVICE=cpu`. No
  device selector was added to the WebUI. `setup_demo.py` prints its serving command with
  `SHEPHERD_DEVICE` set to the device its demo trained on.
- **BACKLOG 19.1 and 19.2, the reload's selection.** The scan reads each candidate with
  `read_checkpoint` (no bare `torch.load`) and keeps each read's digest; the second read for the
  metric is gone. After the build, the metric in `selection_reason` comes from the built
  pipeline's `checkpoint_meta`, the served read (`src/api/routes/pipeline.py:475`): "; best
  val_mrr=…" for a file the scan ranked by its metric, and for a fallback pick (no candidate
  carries one) "; not ranked by metric …", with the loaded file's metric if it has one.
  Under M2.1 decision 4 the build's read is what is served: when the selected file's scored digest
  differs from the served identity's checkpoint digest, the reload still succeeds and
  `selection_reason` states that the file was replaced after it was scored, with both digests
  (`:491`). It is not a refusal. The two allowed pin entries for those `torch.load` sites are removed.
- **BACKLOG 19.8, the measurement's subject.** `measure_served_pipeline.py` hashes no input by path
  for its subject: the subject is the served identity each repeat's service reports at R1, which
  must pass the service's own response model and name a verified checkpoint, graph and manifest,
  table and record. Readiness reads the table's state, the scoring mode and the device in effect
  from it. A counted response must carry that same identity (otherwise `served_identity_mismatch`,
  never accepted), R3's reloaded service must report it, and the readings are complete only when
  every repeat reported one identity. The workload read stays 19.8's separate fix.
  `scorer-measurement/PLAN_B04_PRODUCTIONISATION.md` §7.3, which that script implements and which
  says a change to the procedure is made there first, is amended in the same change (readiness,
  the counted response, the subject and completeness, each marked "since contract M3c"); the
  readings already taken there and S10's evidence keep their earlier subject form.
- **Consumers moved in the same change.**
  - **The WebUI's Diagnosis tab** reads the result's own identity: banners above the results for
    a synthetic workspace, no trained model, no shortest-path table and the demo answer (and for a
    response without the field, never labelled demo); a "Produced by" line and a closed
    full-identity block under them. The model status shows the loaded pipeline's identity with the
    §6 line, and its "KG version mismatch" heading now names what the fingerprint compares
    (structure, not identity).
  - **The Markdown report** replaces its "Model version" line with a "What produced this result"
    section before the candidates: the banners, a Part | State | Value table with full `sha256:`
    digests and no blank cell, the §6 "For the UI" line verbatim (with one more line when no row is
    marked ✓, so it is not read as a claim that something was checked) and "Not a certified
    medical device; decision support for a qualified clinician." The table rows have one builder,
    shared with the tab.
  - **The CSV** keeps one header row and appends 21 `served_*` columns after the 15 candidate
    columns, the same on every row; a null is always a word (`absent`, `none`, `not_stated`,
    `demo`, `not_reported`), digests are `sha256:`-prefixed. A text cell a spreadsheet would run
    as a formula now gets a leading apostrophe; there was no such protection before. Export file
    names carry `SYNTHETIC_` or `DEMO_`.
  - **The probe** adds checks, not probes: E1 compares the identity part by part with the probe's
    own reads (the checkpoint D1 trained, the workspace's graph, manifest and record, the
    sidecar's `build_id` or the table's absence, the device asked for and in effect) and records
    it; E2 requires the status to report it unchanged; E5, a refused reload to leave it; E7, a
    diagnosis to carry it; M4 and R6 the same of a diagnosis served with the table verified.
  - **`measure_served_pipeline.py`** and **`setup_demo.py`**, as above.
  - **Tests:** `test_served_identity.py`, `test_serving_device.py`, `test_probe_serving_phase.py`
    (phase E on the CPU) and `test_setup_demo.py` are new; existing tests moved off
    `model_version`, and those that load a checkpoint name `device="cpu"` or set
    `SHEPHERD_DEVICE=cpu` themselves, never through a shared fixture.
- **Not done.**
  - No GB10 acceptance yet: the identity has not been read from a model served on CUDA.
  - N2 is not part of M3c: `phenotype_normalisation_version` is the current rule constant, and no
    stored input records the rule yet.
  - The probe's own A2 handling of `--device auto` is unchanged.
  - B-2's policy for path-only serving is undecided; the identity states such serving as
    checkpoint *absent* and graph *unverified* (above) and adds no policy of its own.
  - Not built: a flag when a displayed result was produced by a pipeline no longer loaded.
    `DiagnoseResponse` ignores an unknown top-level key, as it did before, so a stale
    `model_version` in a body passes unnoticed; the nested identity is strict.
- **Found while building, and fixed in the same change.** The reload route settles the device (the
  request, then `SHEPHERD_DEVICE`, then `auto`) right after its file check, so a reload refused
  for its device has not first read every candidate in the run layout. The structural-fingerprint
  warning (`src/utils/fingerprint.py`) no longer calls itself a "KG/data version mismatch": it
  compares structure, and says it is not the graph's identity.
- **Internal review (2026-10-16) of `39aec16`.** Eight dimensions (the identity, the device rule,
  the API contract, the WebUI, the scripts, test strength with mutations run in a scratch copy,
  the documents, and architecture against this section), each finding put to independent skeptics
  (three for P2, one for P3). None P1; 19 confirmed, all P3 once verified (the one raised as P2,
  a startup refused for its device answering 503 with no test pinning it, was agreed P3 by all
  three: the code was right); 5 refuted. Fixed in `fe36a72` and the commit after it:
  - the device rule for a model supplied in memory with a device named, for a host without
    PyTorch, and for a blank request (above);
  - "best" said only of a file ranked by its metric, and the replaced-file note mentioning a
    metric only when one is shown (above);
  - the dataset origins as one named constant (`SERVED_DATASET_ORIGINS`), pinned equal to the
    record reader's and the response model's;
  - the model status adding "nothing checked" under the §6 line when no part is marked ✓, worded
    for the loaded pipeline, by the same rule as the report and the tab; a response without an
    identity saying it did not say what was checked, instead of "nothing was checked";
  - tests that each kill the mutation they target: the 503 after a device-refused startup, the
    device in effect read from the model, the request passed on as given, the empty result over
    a full pipeline, every digest pattern, the probe's E2 and M4/R6 checks firing, and assertions
    that had compared a constant with itself made real;
  - documents: the CSV's "no empty cell" scoped to the identity columns, the old fallback's
    traces, and these citations.

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

**Training start's explicit `checkpoint_dir`** is honoured verbatim today. **Superseded by M3a-2
(2026-10-15):** every run writes into its own run directory, the listing moves with it, and an
explicitly given output path obeys the same protection. What phase 2 adds here is the network-side
restriction of the operations that start training.

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
own plan defines. M3a-2's run listing is derived from checkpoint records and is not a record of its
own.

### 5.1 By record

| Record | Change | Milestone |
|---|---|---|
| Checkpoint (`callbacks.py:296-327`) | `training_input_digests` gains `kg`, the manifest-bound source graph (M2.1's stated exceptions); every other digest comes from the bytes read; `config` gains the values in effect; new `producer` and `environment` keys; one payload builder for both writers, with the resume state (M3a-1); written into the run's own directory (M3a-2) | M2, M3a |
| Run listing (`run.json`), new | A projection of its run's checkpoint records, never an authority (M3a-2). Not built in M3a: the listing reads each checkpoint's record when listed | Later |
| `split_manifest.json` | Binding to `kg.provenance.json` required | M3b |
| `kg.provenance.json` | Unchanged until the KG record v2 | — |
| SP sidecar | Unchanged; it already records `kg_digest`, `build_id` and `max_hops` | — |
| Measurement manifest and report | Input digests from the bytes read, except `kg`, the manifest-bound source graph (M2; M2.1's stated exceptions); environment facts; the workspace's graph digests for both cohort kinds; the dataset record's binding state and origin, apart (`dataset_record`, M3b, outside the semantics digest); R10's evidence and labels (§5.4) | M2, M3b, M4 |
| Ledger | item 14's ledger v2 (`PLAN_TEST_RESULTS.md` §5), which takes the schema number after the `schema_version` 2 that the retirement's R2 set (`src/evaluation/sidecar.py`) | M4 |
| `InferenceResult`, `DiagnoseResponse`, serving state | `get_pipeline_config()` and the API's pipeline status carry the served workspace's dataset record, binding state and origin apart (M3b); one structured served-identity field replaces `model_version` and `kg_version`; `get_pipeline_config()` carries the settings in effect. M3c's part is implemented (2026-10-16, awaiting review): every result and the diagnose response carry the served identity, with the settings in effect, and `get_pipeline_config()` and the pipeline status report it for the loaded pipeline (§4, "M3c as implemented") | M3b, M3c |

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
- Training and serving turned `auto` into CPU silently (`train_model.py:742-745` and
  `pipeline.py:1065-1066` at `0142436`; serving's at `pipeline.py:1085-1086` at `cbba015`). M3a
  and M3c align them with measurement: training refuses `auto` without CUDA since M3a
  (`resolve_training_device`, `scripts/train_model.py:897`), and serving refuses it since M3c, as
  training does (`resolve_serving_device`, `src/inference/pipeline.py:253`; applied before any
  workspace input is read, `:609`, and in `build_pipeline` before `kg.json` is parsed,
  `src/api/main.py:525`).
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
   - **M3a-1 (2026-10-15)** follows the second and third points for training's `producer`:
     *unrecorded* when the revision cannot be obtained, never recorded as clean. Stamping at
     packaging or deployment is not built in M3.
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
