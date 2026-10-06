# PLAN — backlog item 14: test results recorded by the pipeline, shown where a model is chosen

**Status: draft for discussion, revision 4.** Nothing in item 14 is implemented. §4 lists
decisions that the owner, and where marked the institution, make before any code is written.
§6's order applies only once they are made.

**Revision 4 (2026-10-06)** follows the review of `627ed08` and the provenance contract
(`docs/working/PLAN_PROVENANCE_CONTRACT.md`, "the contract" below), which now carries the
relations that a test result depends on.
- **Item 14 is two phases** (§6).
  - **Phase 1** is the contract's M4: a complete CLI flow from import to display.
  - **Phase 2** is M5: the Test tab, job runner, model list, D8, and D4's ordering and D5's
    arbitration.
  - Phase 1 depends on the contract's M1–M3, not on phase-2 machinery.
- **Deterministic runs are no longer described as a cross-environment guarantee** (D4, D5).
  - The measurement key does not record every hardware condition.
  - A disagreement calls for investigation under the recorded conditions, not a verdict of
    defect.
  - Preferring a deterministic record is a declared protocol choice, not a claim that it is
    more accurate.
- **D2 is settled by the owner:** digest binding and its stated limit only, with options (b)
  and (c) removed.
- **The owner's decisions of 2026-10-06 apply** (contract §1):
  - a model whose recorded training graph does not match is refused, so measurement checks the
    model↔graph relation;
  - old-pipeline artifacts are not supported, so §4.9 question 4 is answered and ledger v2
    starts clean.
- **Amended 2026-10-06, after the review of `463a0df`**, to match the contract's revision 2:
  - the read-only display of a v1 ledger, kept as a condition until question 4 was answered,
    is removed (§5);
  - R10 is recorded as the contract's §5.4 defines it (§6, step 2). Its case label compares
    phenotype sets and does not claim the same scoring input; repeated terms are a new data
    question (§4.9, question 2);
  - every digest the measurement records comes from the same read that parses the file (§5,
    §6 step 2);
  - training start's `checkpoint_dir` restriction moves back to step 5, with the listing.

Facts about the code are cited from `main` at `1cab39f`, except where §3 notes the F1–F3 fix on
this branch. **A bare § number refers to this plan.** Sections of `EVALUATION_COHORTS.md` are
written "EC §x", and other documents are named.

**Revision 3 (2026-10-02; amended 2026-10-05)** follows the review of revision 2 (`6854265`),
and the owner's caution that between two pipelines a compatibility layer easily becomes a
parallel one:
- **One path per concern** is now a stated rule (§4.0), and the plan is checked against it. As a
  result, ledger v2 no longer reads v1 files (§5).
- **D8:** a recorded M3 risk acceptance does not unlock the controls. The refusal happens where
  the event is handled, and a direct call to the endpoint is part of acceptance.
- **D6:** Mode C's metrics keep their own denominator, the ranked cases. Coverage and the
  all-source figure are shown separately.
- **D5:** a key whose runs disagree stays blocked. The way forward is a deterministic re-run,
  which is a different key.
- **Existing paths are upgraded to fit, not worked around** (§4.0, the owner's rule; amended
  2026-10-05, `cf23970`). The job runner is made by reshaping `training_manager`, with the
  Training Console moved onto it. The existing checkpoint listing route is upgraded instead of
  adding a second one.
- **F1–F3 are fixed** on this branch (`d95965e`, `88569e4`, `fe44b8f`, `20f43d1`, `a57ed2d`).
- **Corrected after an independent verification (2026-10-05).** Five reviewers checked every
  claim in revision 3. The corrections:
  - **the pool is not shared** (§2.2, D3, §4.0). Only the cosine is, so unifying the pool is
    part of the constraint on B-1;
  - **D4 groups by scoring semantics**, not the whole semantics digest, which splits at every
    commit; and it orders by the first registered record;
  - **D5's block holds within one key.** Across keys the choice is made visible: test runs fix
    the seed and batch formation, and D4 uses the first registered record. The block rests on
    plain files;
  - **the supplied-cohort path has other consumers,** which move with it (§5; the second
    round, below, found where the file is really read);
  - **the listing route's `weights_only=False` load is converted** when the route is upgraded
    (§5, §6 step 5);
  - **D8** excludes a same-host reverse proxy from M1 and names training start as a known
    ungated exception.

  The pool claim and D4's grouping were carried over from revision 2. The same pass stated
  D2(b)'s single-account limit and moved reports onto the shared staging helper.
- **A second verification round (2026-10-05)** corrected the first round's corrections:
  - **the cohort file is located twice today,** by `resolve_cohort` and by `read_samples`. One
    resolver now locates it, so the digest recorded is that of the samples scored. The frozen
    oracle is the named exception (§5);
  - **the checkpoint list is C3 checkpoint inspection,** so it follows D8 rather than being
    display. The Training Console's resume dropdown is a second listing, and it moves onto
    the one listing function;
  - **D4 names its fields** (seven, after a third check moved two Mode-A-only constants out).
    Contested means not ordered, and only a deterministic record can supersede. Ledger v2
    carries what D4 needs from it, and contested status comes from the report store;
  - **D5:** batch size and code changes do move Mode C's numbers at floating-point level, so
    the job runner fixes a test run's inputs;
  - **D8:** M1 is an explicit setting, checked together with the request's arrival socket and
    the absence of forwarding headers. Stop is per job type;
  - **D3:** B-1 must also settle the served disease clamp and the silent phenotype drop.
- **A third, focused check (2026-10-05)** corrected the frozen oracle's staging, which now
  stages a directory the oracle reads as its `--data-dir`. It also corrected the reader list,
  now limited to measurement and audit readers with the generated-split readers named; the
  Training Console's resume listing, now a named C3 exception; D4's field count; and the D3
  citations, now at `1cab39f`.

Revision 2 (`6854265`) settled D3 by the scorer policy, made D5 follow the owner's checkbox, and
reworked D4, D6 and D8.

---

## 1. What the owner asked for

Restated from the owner's description of 2026-10-02 (recorded in `EVALUATION_COHORTS.md` §5,
question one, and BACKLOG item 14):

- **When it happens.** A checkpoint chosen on validation is tested on MyGene2 and on the
  hospital's own offline cohort (`EVALUATION_COHORTS.md` §6.0, steps 4–5).
- **The problem today.** Its measured performance would be written into the checkpoint's file
  name. That is easy to forget, easy to get wrong, and easy to alter.
- **What the pipeline should do instead.** It writes the result itself, as part of the
  checkpoint's identity, so that a user's mistake cannot drop or corrupt it.
- **Where the result is shown:**
  - in a field when the model is loaded for inference;
  - beside each entry in the list of selectable models;
  - as an advanced option, ordering that list by test score.
- **The shape the owner proposed:**
  - a separate **Test** tab;
  - **test-cohort management** with version verification modelled on the training datasets.
    MyGene2's release cadence is unknown, so versions are verified either way;
  - a **checkbox** on the test output. Checked, the result is written automatically. Unchecked
    covers a special case: several rounds of testing with a result chosen by hand. That case
    arises while a new test cohort is itself being evaluated, before it becomes a baseline.
- **What a test measures** (the owner, on revision 1). A test examines the neural network
  *itself*. Shortest-path (SP) analysis is not part of it. In the reference paper SP enters only
  the candidate-gene score, and the clinician-facing results view was redesigned so that
  auxiliary scores are optional views the clinician turns on.

## 2. What exists today

### 2.1 The evaluation ledger

`src/evaluation/sidecar.py` and `scripts/record_evaluation.py` implement
`EVALUATION_COHORTS.md` §6.5.

- **The file.** One `evaluations.json` sits beside the checkpoints it describes.
- **The key.** Each record is keyed by `checkpoint_digest`, `cohort_role`, `cohort_digest` and
  a `measurement_semantics_digest` (`sidecar.py:157-162`). That digest covers every manifest
  field that can move a number, seeds and the numerical regime included (`sidecar.py:72-115`).
- **A result stays tied to the weights.** It is bound to the checkpoint's SHA-256, never to its
  file name, so a rename does not lose it.
- **A contradiction is refused.** A second record with the same key is accepted only if it is
  *equal as a whole* (`sidecar.py:342`); anything else is refused, with no override flag. The
  record carries `source_artifact_digest`, the digest of the report file. So re-recording the
  same measurement from a report whose bytes differ is refused even when its metrics are
  identical, for example after the checkpoint has moved and its path in the manifest changed.
- **A seed is not bit-determinism.** The ledger's own documentation says so: a seed controls
  the harness's random streams, not CUDA's arithmetic, so a refusal is "an instruction to look,
  not a proof of a defect" (`sidecar.py:174-181`, `:333-336`).
- **Writing.** It is single-writer: an atomic replace with best-effort stale-write detection,
  and no locking.
- **Schema.** A file whose `schema_version` is not 1 is refused, with the instruction to read it
  with the revision that wrote it "rather than reinterpreting its fields under new rules"
  (`sidecar.py:225-230`).
- **Location.** `find_checkpoint` looks only directly inside the directory it is given. Under
  the per-architecture layout, the ledger therefore sits in `checkpoints/<arch>/`.
- **Readers.** Nothing reads it except `record_evaluation.py --show`. No API route, UI or
  serving code does.

### 2.2 The measurement engine, and what each mode measures

`scripts/measure_scorer.py` is the only producer whose output the ledger accepts. The frozen
`scripts/evaluate_model.py` is not one: its report has no manifest, and its `--split` choices
are `train`/`val`/`test`.

- **How a cohort is given.** It is `<data_dir>/<split>_samples.json` with `--cohort-kind
  supplied`. Even then the workspace's graph artifacts are verified (`measure_scorer.py:122-128`),
  so a test cohort has to sit in a valid generated workspace.
- **Mode A** cannot run on any checkpoint in the scanned family (`BACKLOG.md`, the M1/M2
  finding), and Mode B depends on it.
- **Mode C is the approved disease scorer, measured.** `run_mode_c` (`measurement.py:1227`):
  - pools each patient's phenotype embeddings (masked mean);
  - scores the result by raw cosine against **every disease node in the graph**;
  - ranks by score, breaking ties by ascending global disease id (`canonical_ranking`,
    `measurement.py:105`).

  That is `DISEASE_SCORER_POLICY.md` statements 1–3, which the institution accepted (§3.4
  there).
- **Mode C and the served pipeline share the cosine, not the pool** (`src/inference/scoring.py`).
  - The cosine is one primitive: the served `cosine_scores` delegates to
    `cosine_score_matrix` (`scoring.py:212-243`).
  - The pool is two implementations, bound by an equivalence test on valid input
    (`scoring.py:158-164`). The served `pool_patient_embeddings` clamps an out-of-range
    phenotype index into range (`scoring.py:135-155`, the clamp at `:154`). Mode C validates
    every id first and refuses an out-of-range one (`_assert_ids_in_range`,
    `measurement.py:1193`, called at `:1285`), because a clamp scores a different patient
    with a plausible rank.
- **What the Diagnosis tab shows today is the legacy behaviour.** It is
  `0.7 × ((cos+1)/2) + 0.3 × SP` behind a BFS discovery gate. The policy replaces it in work
  item B-1, which is gated and not implemented (`DISEASE_SCORER_POLICY.md` §2, §6).
- **Mode D**, the production path-reachable candidates with the η/SP mixture, measures that
  legacy behaviour. It is for B-0.5's attribution work. It is not built, and its design has an
  unresolved problem (`docs/working/scorer-measurement/PLAN_B03.md`).
- **Patient-level output.** The rank and prediction side files carry patient `sample_id`s and
  are written without staging.

### 2.3 Test cohorts

- **Format.** A supplied cohort uses the generated format: a JSON list of
  `{patient_id, phenotype_ids, disease_id}`, whose values are **integer graph node indices**.
- **No importer.** Nothing converts HPO or MONDO/OMIM/ORPHA identifiers from an external file
  into those indices. MyGene2 has no loader, no format and no download in this repository. The
  reference repository's preprocessing writes its diseases as MONDO ids, which matches this
  graph's disease nodes (`EVALUATION_COHORTS.md` §2).
- **No binding to a graph.** A supplied cohort records no graph digest. An index that is in
  range but belongs to a *different* graph or index order is detected nowhere.
- **No version label.** No such field exists for cohorts, or for workspaces.

### 2.4 WebUI and API

- **Tabs.** The WebUI has four: Training Console, Diagnosis, Runtime Settings, and **Model
  Management, a placeholder** that reads "Coming soon — will provide checkpoint listing and
  metrics comparison" (`src/webui/app.py:191-206`).
- **How the tabs reach the backend.** It is mixed:
  - the Training Console imports `training_manager` directly;
  - the Diagnosis tab calls the API over HTTP, at a hard-coded `http://127.0.0.1:8000` on
    `main` (`diagnosis_panel.py:44`; F1, fixed on this branch).
- **One job runner.** `training_manager` tracks one subprocess, with `train_model.py` hard-coded.
  It is not a general job runner.
- **Checkpoint listing.** The only listing route is `GET /api/v1/training/checkpoints`. No route
  lists checkpoints by workspace and architecture.
  - **Which directory.** It reads the training singleton's `checkpoint_dir`: the relative
    `checkpoints` until something sets it. A training start sets it, and so do two Training
    Console handlers, from values the request carries (`training_console.py:368, 510`). It
    looks at the top level only (`*.pt`), so under the per-architecture layout it finds
    nothing (`src/api/services/training_manager.py:88, 102, 448`).
  - **A second listing already exists.** The Training Console's resume dropdown
    (`_refresh_checkpoints`, `training_console.py:499-524`) lists by workspace and
    architecture through the same `get_checkpoints`.
  - **How it reads.** It loads every file with `torch.load(weights_only=False)` (`:464`). That is
    one of SPEC_4 §2.1's client-reachable deserialisation sites, reached through a
    client-influenced directory.
- **Diagnosis tab.** It has no checkpoint picker, only a free-text path and an architecture
  dropdown.
- **Network exposure.** The launcher's default bind is `0.0.0.0`
  (`scripts/launch/shep_launch.py:17`), and the systemd unit keeps it while SPEC_4 item C1 is
  open. Gradio is mounted at `/ui` in the same app, and every UI callback is an HTTP endpoint.
  Anyone who can reach the port can therefore trigger any UI action, the Training Console's
  included.
- **Authentication.** The API has none. `docs/working/results-review/SPEC_4_DEPLOYMENT_SECURITY.md`
  (normative, draft) classes reload, config and checkpoint inspection (C3), and training start
  and stop (C4), as **disable or protect**. The Gradio handlers call the managers directly, so
  protecting the HTTP routes alone would not cover `/ui`.

### 2.5 Protocol constraints already decided

From `docs/working/EVALUATION_COHORTS.md`:

- **EC §4.1.** A valid test cohort:
  - is frozen, versioned and digested;
  - declares its metrics and acceptance rules before the result;
  - is separated from training and from selection;
  - is access-controlled, *so that repeated inspection does not silently convert it into a
    selection set*. Condition 6 is "an operating rule for the institution, not a piece of
    software".
- **EC §4.2.** MyGene2 and the institutional cohort are **reported separately, never pooled**.
- **EC §5 item 5.** A cohort is identified by **digest and version label, never by filename or
  path**. Fetching a dataset by URL into a clinical system "would need its own review".
- **EC §6.7.**
  - "A control that changes a cohort's role is a protocol change, not a display setting."
  - A burned cohort "cannot be returned to untouched by a checkbox, a rename, or a new run
    label".
  - "A UI that permits the relabelling silently is worse than no UI."

## 3. Found while planning — outside item 14

These were not looked for, and each is from reading the code. They touch what item 14 builds
on. The owner authorised F1–F3 as one small change separate from this item, and it is on this
branch. F4 is a documentation fix, left to §6 step 4.

| | Finding | Evidence |
|---|---|---|
| F1 | **Under the systemd unit, the Diagnosis tab could not reach the API** (verified). The unit passes `--port 8264` after the launcher's default `--port 8000`, and uvicorn takes the last value. The tab calls port 8000. **The fix takes the address from one shared source**: an address the server accepted a request on. It adds no second hard-coded port and does not probe several | `scripts/service/systemd/shepherd.service:25`; `scripts/launch/shep_launch.py:17, 416`; `diagnosis_panel.py:44`. **Verified by running**: the launcher with the unit's arguments served 8264 and refused 8000. **Fixed in `d95965e`**: the address is recorded from each request's ASGI `server` field. Follow-up `fe44b8f`: self-calls bypass proxy settings, and a failed status read gives its real reason |
| F2 | **For checkpoints written by the current trainer, the loaded model's ranking metrics are never shown.** The pipeline looks for `mrr`, `hits_at_1` and `hits_at_10` in the checkpoint's `logs`. The current trainer stores its `val_`-prefixed validation dict there (`val_mrr`, `val_hits@k`), so only the losses appear. No trainer in this repository has written the names looked for | `src/inference/pipeline.py:934`; `src/training/trainer.py:723`; `src/training/callbacks.py:303`. **Fixed in `d95965e`, `88569e4`**: the ranking keys copied are `RANKING_SCORE_KEYS`, the list auto-selection uses, and the old names are not kept as a fallback |
| F3 | **The Diagnosis tab's model status is not refreshed when the page loads.** Its initial value is computed once, when the app is built, before the server is serving. A page opened later shows that build-time status until Reload is pressed; Reload does update it. `_on_load_status` is defined and never wired | `diagnosis_panel.py:719-722, 754-756, 909-913`. **Fixed in `d95965e`**: read on every page load |
| F4 | **EC §6.5 says the ledger key includes mode and tie policy.** The code keys on the semantics digest, which hashes both. A documentation fix | `EVALUATION_COHORTS.md` §6.5; `sidecar.py:157-162` |

## 4. Decisions

Each lists the options and a recommendation. **D3 is settled by the scorer policy**; the rest
are recommendations until the owner confirms them. §4.9 gathers what is actually asked.

### 4.0 One path per concern

The project is moving from older paths to newer, verifiable ones. In that transition, a
compatibility layer easily becomes a second path that has to be kept working and kept agreeing
with the first, although only the newer one will be used. This item adds none:

| Concern | The one path | Not used for it here, and not added alongside |
|---|---|---|
| Producing a test number | `measure_scorer.py`, Mode C | `evaluate_model.py`, the frozen oracle; Mode D, which measures the legacy behaviour and is a separate item |
| Scoring | one implementation of the pool, the cosine, the candidate universe and the tie rule, shared by Mode C and B-1 (D3). Only the cosine is shared today | a test-only scorer; two pools |
| Resolving a test cohort | the cohort registry (D6): one resolver locates a cohort's file, and both the digest recorded and the samples scored come from that file (§5) | the loose `<split>_samples.json` supplied path, which today is located twice, by `resolve_cohort` and by `read_samples`. The frozen oracle is the one named exception (§5) |
| Recording | ledger v2 | a v1 reader inside v2 code (§5) |
| Running a job | one job runner, made by reshaping `training_manager`; the Training Console moves onto it | a second process manager, or a runner extracted around the old one |
| Listing checkpoints | one listing function behind the existing route and the Training Console's resume dropdown, upgraded: by workspace and architecture chosen among those the server lists, with a safe metadata read (§5) | a second listing path |
| A checkpoint's validation ranking score | `RANKING_SCORE_KEYS` and `ranking_score_detail` | a second key list (F2 removed one) |

**Existing paths are changed to fit, not worked around.** Where an existing path has a defect or
a shape that does not fit, it is upgraded, and the new module is built for the upgraded path. A
new module is not bent around an old one's limits.

A compatibility reader is added only where a file that must stay readable actually exists. It
then states the condition for removing it.

### D1 — Where a test result lives

| Option | Consequence |
|---|---|
| Inside the `.pt` | Changes the checkpoint's SHA-256. Every record, evidence file and citation of that model then points at bytes that no longer exist (EC §5 item 4) |
| In the file name | The current practice, and what the owner wants to replace |
| **Beside it, in the existing ledger** | Bound by checkpoint and cohort digest; survives a rename; contradictions refused; already built |

**Recommendation:** the ledger. The file name stays as it is; a display name can show the score
without becoming the record.

### D2 — What tamper resistance is claimed

- **What digest binding gives:**
  - a result cannot be moved onto another model or cohort without the mismatch showing;
  - a rename cannot detach it;
  - a second, different number for the same measurement is refused.
- **What it does not give.** Anyone who can write the directory can edit `evaluations.json`.
  `PLAN_ONTOLOGY_PHASE2.md`'s rule applies: *a digest is not a signature*.

**Settled (owner, 2026-10-06):** state the limit and claim only binding. The threat this stage
addresses is misalignment between stages, not a malicious account holder. Keyed signing (HMAC
or asymmetric), an append-only log and a hash chain are out of scope. The claim boundary is the
contract's §6.

### D3 — What a test result measures: settled by the scorer policy

Revision 1 offered Mode C as the model-only fallback and asked whether acceptance needed the
served ranking. That framing was wrong. A test examines the network itself (§1), and the
institution has already said what the network's disease ranking is.

**The task, stated exactly:**

- **Ranked:** every disease node in the graph, for each patient (statement 3).
- **Score:** raw cosine between the masked-mean pool of the patient's phenotype embeddings and
  the disease embedding (statements 1–2). Embeddings come from one full-graph forward pass, as
  in serving.
- **Order:** by score, descending; ties go to the lower global disease id.
- **Ground truth:** the patient's disease node. A truth that is not a disease node in the graph
  cannot be ranked, so it is handled at import (D6), never inside the measurement. Mode C treats
  absence as fatal.
- **Metrics:** the authoritative family that `measurement.py:_authoritative` computes —
  `untruncated_mrr`, mean rank and hits@1/5/10/20/50/100. Which of them, if any, carries an
  acceptance rule is declared before the first result (EC §4.1, and §4.9 question 1).

**This is Mode C**, and it is the primary score this item records, not a substitute for
something else.

**What is not in it:**
- **SP.** Statement 4a forbids SP from admitting, excluding, reordering or rescoring a disease
  candidate. Statement 5 allows it only as an optional analysis after ranking.
- **The BFS discovery gate**, which statement 3 removes.
- **Patients-like-me** (the paper's Eq 16) and **causal gene discovery** (Eq 14, the paper's
  only SP fusion). These are other tasks with their own measures, outside this item.

**How it is labelled.** Everywhere a result appears it reads "disease ranking by the model,
approved scorer policy (Mode C)", with the cohort's role and version. Until B-1 lands, the
Diagnosis tab still ranks by the legacy mixture behind the BFS gate. While that holds, the
label adds that the tab's current ranking differs from the measured one. The note goes when
B-1 ships.

**One scorer, not two.** The number is meant to describe what the Diagnosis tab shows once B-1
ships. That holds only if B-1's production scorer and Mode C share one implementation of the
pool, the cosine, the candidate universe and the tie rule. Today only the cosine is shared
(§2.2):
- **the pool is two implementations.** The served one clamps an out-of-range index, and Mode C
  validates ids and refuses one. Under §4.0's rule the served path adopts Mode C's id
  validation: the clamp is the defect, and Mode C's rule is the one to keep;
- **the universe and the tie rule** live in `measurement.py`;
- **two more served-path behaviours** that Mode C does not share must also be settled:
  - the disease-index clamp (`src/inference/pipeline.py:1536` at `1cab39f`), which an
    all-disease universe removes;
  - the silent drop of phenotypes the graph does not map (`pipeline.py:1509-1513` at
    `1cab39f`). The clinician should be told about it, rather than having the ranking scored
    around it.

This is recorded as a **constraint on B-1**. It is not built here.

**What it does not claim:**
- **The paper's number.** Statement 2 is interim. The paper's disease scorer is a squared
  distance (Eq 18), not cosine. If retraining changes the scorer, `score_semantics`, which is in
  the semantics digest, keeps old and new results apart.
- **The paper's cohort.** A MyGene2 snapshot taken from the public pages need not be the cohort
  the paper evaluated (N = 146 in its deposit). The version label records which one was used.

**Mode D** measures the legacy served behaviour for B-0.5's attribution work. It is a separate
item and **not a prerequisite** for this one.

### D4 — Ordering by test score

A list ordered by test score invites choosing by it. A cohort used to choose stops being an
independent test (EC §4.1 condition 6, EC §6.7). The owner asked for ordering as an advanced option,
so the aim is to make that use visible and comparable, not to forbid it.

**Proposed rules:**

- **Only comparable numbers are ordered together.** Ordering is by one metric over records of
  one cohort version and one **scoring semantics**.
  - **Scoring semantics is narrower than `measurement_semantics_digest`.** That digest also
    covers the seeds, the batch formation and the numerical regime, `software_revision`
    included (`sidecar.py:87-92`, `:104-114`). Grouping by it would split the list at every
    commit, so checkpoints tested at different commits could never be compared.
  - **Scoring semantics is a named list of seven fields,** the ones that say *what* a Mode C
    test computed: `mode`, `cohort_kind`, `candidate_construction`, `score_semantics`,
    `model_construction`, `canonical_tie_policy_version` and `metric_schema_version`.
  - **Four groups are left out:**
    - the seeds and batch formation (`sidecar.py:87-92`);
    - the numerical regime (`:104-114`);
    - the sampler and loader fields. Mode C records these from loader defaults but does not
      use them (`measure_scorer.py:352-358`), so a changed default must not split the group;
    - `legacy_truncation_k` and `legacy_tie_policy`. They are constants on every manifest
      (`measure_scorer.py:364-365`) that only Mode A's legacy ranking reads, so for the same
      reason: they would split every group when the oracle's surface retires.

    The graph needs no field of its own, because a cohort version is bound to one graph
    (D6). The regime is shown on each row, and a row whose regime differs from the rest is
    marked. A CPU run is such a row: a different key (§5), in the same group, marked.
  - **Labels say why a checkpoint is not ordered.** One with no record on this cohort version
    is listed after the rest as "not tested on this cohort version". One whose records are
    all under other scoring semantics is listed as "tested under different scoring
    semantics". Neither is interleaved by a number that is not comparable.
- **One number per checkpoint, chosen by a fixed rule.** A checkpoint can have several records
  in one group, from re-runs after commits or under a deterministic regime. Ordering uses:
  - the first registered record from a deterministic regime, if there is one;
  - otherwise, the first registered record.

  The others are shown beside it, with any difference marked.
  - **Contested means not ordered.** If the record the rule picks is contested (D5), the
    checkpoint is listed as "contested — not ordered". The next record is not promoted, so
    contesting a score can take a checkpoint out of the order but cannot put a better number
    in.
  - **A deterministic record takes precedence by a declared protocol choice.** A later
    deterministic record takes precedence over an earlier non-deterministic one. This is a
    display rule declared in advance, not a claim that the deterministic number is more
    accurate. Another environment can give another deterministic number: the key does not
    record every hardware condition (contract §5.3). Otherwise a later re-run cannot replace an
    earlier score.
  - **The same limit as D5.** Contested status comes from the report store, so ordering reads
    the ledger and the report store's contested status, and carries D5's limit: it rests on
    plain files.
- **Auto-selection never reads test results.** `select_auto_checkpoint`
  (`src/utils/checkpoint_paths.py:151`) stays on validation under every option.
- **The cohort's usage history is kept.** It holds:
  - every run and every registration against a cohort version;
  - every model load made from a list ordered by that version's score.

  Nothing removes an entry: not a rename, re-map, relabel or "baseline" label. The history
  makes use for selection visible; it does not pretend to prevent it.
- **Per role.** Ordering is offered for research roles, MyGene2 among them. For the
  institutional acceptance role, ordering by its score *is* selecting by it.
  **Recommendation:** off by default there, unless the institution decides otherwise (§4.9
  question 1).

### D5 — The checkbox, reports and records

The owner's semantics, followed as stated:

- **Every run keeps its report**, checked or not. The report is the measurement's full output:
  manifest, metrics and counts. It is stored with the cohort version it ran on, and the UI
  never deletes it.
- **Checked.** When the run completes, its report is registered in the ledger automatically.
- **Unchecked.** The report is kept and not registered. The operator may register it later from
  the Test tab, as a **whole report**.
- **"By hand" means choosing which report, never what it says.** No metric is typed, edited or
  copied in by hand.
- **No status gate.** Revision 1 required a candidate→baseline promotion before recording; that
  is dropped. A cohort version is usable once it is imported and verified. "Baseline" may be
  shown as a label. Setting it writes no data and clears no usage history.

**Repeat runs that disagree.** Under a non-deterministic CUDA regime, two runs under one key can
differ (§2.1). So can two runs under one key on different hardware: the key records the device
string, torch and CUDA versions and the determinism flags, but not the GPU model, driver or
cuDNN version. The contract's M4 adds those facts to the report (contract §5.3).

- **Evidence is never lost.** Every report is kept, so refusing a registration loses nothing.
- **A disagreement is flagged.** When reports under one key disagree, the Test tab marks the
  key "runs disagree — investigate" and shows which fields differ.
- **Within one key, picking the best run is prevented.** While a disagreeing report exists
  under a key, registering any report under it is refused, checked or unchecked.
- **A disagreeing key stays blocked, by design.** Two numbers were obtained under one
  identity, and registering either would be choosing between results. An investigation adds a
  recorded note: who, when and what was found. The note is kept in the report store with that
  key's reports, not in the ledger. It deletes no report, and it does not lift the block.
- **Across keys, the choice is made visible rather than prevented.** Mode C consumes no
  randomness, so its seeds, shuffle and worker count do not change what it computes.
  - **Batch size and code changes can.** They move numbers at floating-point level: with the
    repository's own pool and cosine, batch size 1 against 32 differs in the seventh decimal.
    A code change can move them further while `model_construction` stays the same.
  - **All of these are key fields** (`sidecar.py:87-92`, `:104`), so a re-run under another
    seed or batch size, or after any commit, is a new key. Two rules keep that from becoming
    a way round the block:
    - **the job runner fixes the inputs a test run has no reason to vary:** the seed, batch
      size and worker count, and the device, which is the deployment's own with no fallback
      (§5). Only the deterministic setting is selectable, and it is recorded;
    - **D4's fixed rule** picks one record per checkpoint, and later records in the same
      group are shown beside it with any difference marked.
- **The way forward is a deterministic re-run.** `deterministic_algorithms`,
  `cudnn_deterministic` and `cudnn_benchmark` are in the semantics digest (`sidecar.py:72-115`).
  A run under deterministic settings is therefore a different key. A single deterministic run
  is registrable.
  - **A second one that disagrees with it calls for investigation under the recorded
    conditions.** Compare the environment facts first. It is not presumed to be a code
    defect.
  - **What deterministic mode does, and does not do.** It guards against known
    non-deterministic operations; it does not certify a number, and PyTorch promises
    reproducibility only on the same platform, device and versions
    (https://docs.pytorch.org/docs/stable/notes/randomness.html).
  - **There is no automatic fallback.** If deterministic execution is unsupported on the
    device, the run reports that it could not complete as requested. It does not change device
    or mode to finish.
  - **The switch.** `measure_scorer` records these settings but has no switch to request them;
    phase 2 adds one (§6).
- **If deterministic execution is not possible** for an operation the run needs, the key stays
  contested. The Test tab shows every report's values for it, and no single number is
  registered.
- **A record registered before the disagreement appeared** stays, marked contested wherever it
  is shown, with the note.
- **The block rests on plain files.** Reports, notes and the usage history are files in the
  workspace, and deleting a disagreeing report lifts the block. The tools refuse; they do not
  make the refusal tamper-proof (D2).

**The duplicate rule, a ledger change.** Today, re-registering an identical measurement from a
report whose bytes differ is refused (§2.1). The proposed rule:

- **Two records under one key are duplicates when every field agrees except those shown to be
  non-semantic.** Today that is exactly one field, `source_artifact_digest`. The key fixes
  everything else that describes the run:
  - the mode, the tie and metric versions and the cohort block are in the semantics digest;
  - so are the runtime block's fields, or `device` implies them.

  What is left to compare is the results: `metrics`, `n_ranked` and `n_ground_truth_absent`.
- **A duplicate keeps the first record** and adds the new report's digest to that record's list
  of corroborating reports.
- **Anything else is refused**, as now.
- **The split is checkable**, as the semantics list is (`sidecar.py:66-71`). A test lists the
  record's fields, and fails if a new one is in neither the compared set nor the excluded set.
  v2's new fields (§5) are classified the same way when they are added. The list of
  corroborating reports is excluded by its nature.
- **v1 and v2 records never meet.** A v2 ledger holds only v2 records (§5). The rule always
  compares two records of one schema, so no field is ever missing on one side.

This changes the ledger's semantics and gets its own review.

### D6 — How a test cohort is imported, identified and versioned

**Two levels:**

- **A source case set** is the operator's file, identified by its digest. It never changes, and
  re-mapping does not change its identity.
- **A mapped cohort version** is that set mapped to node indices for one graph. It carries:
  - a version label, for example `mygene2-2026-10`;
  - a role: `mygene2`, `institutional_acceptance`, or another named research role;
  - a manifest recording the source digest, the graph's artifact digests (`kg.json` and the
    PyG export whose indices the samples use, `src/kg/artifacts.py:32-37`), the HPO version
    mapped against, the mapping rule's version and the counts below.

  It is immutable. A new graph needs a re-map, which is a new mapped version from the same
  source.

**Import has two steps: preview, then commit.** The preview writes nothing. It shows the source
case count and every case that will not map, in categories:

| Category | Example | Effect |
|---|---|---|
| **Malformed** | The file or a record cannot be parsed, or lacks a required field | The import is refused as a whole |
| **Phenotype term unmappable** | An obsolete HPO term, or one absent from the graph's HPO version | Per case, the terms lost are counted. A case with no mappable term cannot be scored |
| **Ground truth absent from the graph** | The case's disease is not a disease node in this graph | The case cannot be ranked |

- **A case is never dropped silently.** Every excluded case is in the manifest with its
  category.
- **Each figure keeps its own denominator, and they are shown apart:**

  | Figure | Denominator | Shown |
  |---|---|---|
  | Mode C's metrics: MRR, hits@k, mean rank | the ranked cases, `n_ranked`, as Mode C computes them | always, with `n` |
  | Coverage | ranked over source cases, for example `80/100`, with each exclusion count | always, beside the metrics |
  | All-source figure: MRR and hits@k, each unrankable case counted as a miss | the source cases | only if the owner wants it (§4.9), labelled with its own denominator. Mean rank has no value for a miss and is omitted |

  For example: 100 source cases, 20 excluded, and the other 80 all ranked first. Mode C's MRR
  is 1.0 over 80, coverage is 80/100, and the all-source MRR is 0.8 over 100. "MRR 1.0" is never
  shown beside a denominator of 100. The all-source figure exists so that a cohort cannot look
  better by losing its hard cases.
- **Binding.** A run against a workspace whose graph artifact digests differ from the mapped
  version's is refused. This closes §2.3's undetected-index hazard.
- **No URL download** in this item (EC §5 item 5). If wanted later, it goes through the existing
  guarded downloader (`src/ontology/download.py`) and its own review.

**Left to the owner** (§4.9 question 2): the real file formats; whether a case that loses some
phenotype terms is kept with the rest or excluded; whether the all-source figure is reported;
and MyGene2's terms of use, which is not an engineering question.

### D7 — Patient data in the institutional cohort

**Proposed rules:**

- **Patient-level files stay on the machine and out of the repository and all evidence.** These
  are the source file, the mapped samples, and the per-patient rank and prediction files.
- **The UI shows aggregate metrics only.** The import preview lists excluded cases by their
  case identifier only, on the machine.
- **They are stored outside the repository tree**, at a path set by server configuration and
  never by a request (SPEC_4: a client-supplied path is not trusted).
- **Every run on the acceptance cohort is in its usage history** (D4): when it ran, on which
  checkpoint, and its result. Condition 6 is the institution's rule. The history makes its
  observance visible; it does not pretend to enforce it.

### D8 — Who can trigger a test

**Applies to phase 2.** Phase 1's import, run and record are CLI operations on the machine, and
it adds no operation that the network can trigger to change state (§6). D8 governs the UI
controls that phase 2 adds.

Revision 1 said an in-process UI adds "no unauthenticated network surface". That was wrong: a
Test-tab button is a Gradio callback, an HTTP endpoint reachable by anyone who reaches the port
(§2.4).

- **What a test run does:**
  - it loads a checkpoint, which is a pickle;
  - it uses the GPU;
  - on the institutional cohort, it reads patient data;
  - it writes the ledger.

  This is C3/C4's exposure.

**Proposed rules:**

- **The controls that change state** (import, run and register) work only where an actual
  protection limits who can trigger them:
  - SPEC_4's **M1**: a verified single-workspace deployment, reached locally or through
    controlled SSH forwarding. The server checks three things:
    - its deployment setting says M1, which is an explicit operator setting, never inferred;
    - the request arrived on a loopback socket;
    - the request carries no forwarding header (`Forwarded`, `X-Forwarded-For`,
      `X-Forwarded-Host`, `X-Real-IP`).

    This plan reads M1 as excluding a reverse proxy on the same host; SPEC_4's M1 row
    (SPEC_4:128) does not say so. Behind such a proxy every request arrives on loopback, and
    the socket check alone would pass for any remote user (SPEC_4 §3: loopback is not
    authorisation). The header check catches a proxy that announces itself. A silent proxy
    is excluded by the deployment guide, not detected;
  - **M2**: an authenticated, authorised actor, once SPEC_4's protection exists.
- **A recorded M3 risk acceptance does not unlock them.**
  - Treating these as C3/C4-class operations is this plan's classification; SPEC_4 does not
    list the Test tab.
  - SPEC_4 §2 marks both classes "Disable or protect". C3 adds "Not covered by
    clinical-exposure risk acceptance" (SPEC_4:61). For both, the acceptance flag "does not
    unlock them" (SPEC_4:64-68), and M3 itself requires C3/C4 "disabled or protected"
    (SPEC_4:130).
  - Under M3 without an actual protection, the Test tab is read-only.
- **The refusal is where the event is handled, not in the button.** Each state-changing
  handler checks the condition itself before doing anything. Disabled buttons only show the
  reason; they are not the control. A direct call to the event endpoint, through Gradio's HTTP
  API as `gradio_client` makes one, meets the same refusal.
- **The check reads the triggering request's own arrival address**: its ASGI `server` field,
  never a client-supplied header.
  - F1's recorded address cannot serve here. It is the latest across all requests, not this
    request's.
  - How Gradio exposes the triggering request is verified in §6 step 6. If it cannot be read
    reliably, the controls stay refused: the check fails closed.
- **Checkpoints and cohorts are chosen from server-side lists**, never from a path in a request.
- **Acceptance includes direct calls.** Each state-changing endpoint, called directly from a
  non-loopback address, is refused; called from loopback under M1, it works.
- **The checkpoint list follows SPEC_4's C3.** Listing opens checkpoint files, which SPEC_4
  classes as checkpoint inspection: "Disable or protect" (SPEC_4:61). So the list is available
  only where the state-changing controls are.
  - The read-only view shows recorded results from the ledger, which opens no checkpoint.
  - Taking a safe metadata read out of C3 would be an amendment to SPEC_4. This plan does not
    propose one.
- **Known exceptions, outside this item,** named so the gap is not read as closed:
  - **training start and stop stay ungated,** through the Training Console and
    `POST /api/v1/training/start` and `/stop` alike. Until SPEC_4's C4 is addressed, the one
    job runner has two entry policies: test start gated, training start not. Stop is per job
    type, so a training stop cannot stop a test job;
  - **the Training Console's resume dropdown** calls the same listing function (§6 step 5) and
    stays ungated with the rest of the Console. It is a C3 listing outside D8 until SPEC_4's
    C3 and C4 are addressed.

**Left to the owner** (§4.9 question 3): who operates, and which deployment mode applies on the
machine where the controls are enabled.

### 4.9 What is asked of the owner

D1, D5 and D7 stand as recommended unless the owner objects; D5's ledger rule also gets its own
review. D2 is settled by the owner and D3 by the scorer policy. Questions 1 and 2 gate phase 1;
question 3 gates phase 2; question 4 is answered:

1. **The test's use.**
   - Confirm D3: a test result is the model's disease ranking under the approved policy.
   - For the institutional cohort: which metrics, and which acceptance rule, are declared
     before the first result (EC §4.1)?
   - Is ordering by its score offered (D4)?
2. **The data.**
   - The actual file formats.
   - D6's exclusion rules: a case with some lost phenotype terms, and whether the all-source
     figure is reported.
   - A case that lists a phenotype term twice. Mapping can produce one, when two source terms
     map to the same node, and a repeat changes Mode C's input (contract §5.4). Is it kept as
     listed, or does a rule apply? A rule would apply alike at import, in training and in
     measurement, through the shared reader, not in one place only. Until this is decided,
     cases are scored as listed.
   - MyGene2's terms of use.
3. **Operators and access.**
   - Who may import, run and register, above all on the institutional cohort?
   - Which deployment mode (D8) does the machine with enabled controls run under? Under this
     revision only deployment mode M1 enables them today: mode M3 does not unlock them, and
     mode M2 does not exist yet. These are SPEC_4's deployment modes (D8), not the contract's
     milestones.
4. **Existing ledgers — answered by the owner's decision (contract §1, decision 2).**
   Old-pipeline artifacts are not supported. Any v1 `evaluations.json` describes old-pipeline
   models, and is archived with them. v2 starts clean (§5).

## 5. Proposed shape, if §4 is taken

This is one pipeline extended, not a second one.

- **Cohort registry** (new, in `src/evaluation/`):
  - **import:** source case sets; preview; mapping; mapped versions with their manifest, staged
    and published as the workspace manifest is;
  - **verification:** the source and samples digests, the graph binding and the manifest schema;
  - **listing:** versions by role;
  - **usage history:** append-only, per mapped version.
- **Measurement.** `measure_scorer`'s Mode C path resolves a registered mapped version.
  - **Today a cohort's file is located twice.** `resolve_cohort`
    (`src/evaluation/cohort.py:161-173`) locates `<split>_samples.json` for the provenance and
    the recorded digest (`measure_scorer.py:122`, via `artifact_digests` at `:370`).
    `read_samples` (`src/kg/storage/file_storage.py:60-77`) builds the same path again for the
    samples actually scored (`measure_scorer.py:227`, `:572`) and audited
    (`scripts/audit_split_overlap.py:98`; the audit also hashes the path directly, `:227`).
    Changing only one would record a registered version's digest while measuring the loose
    file.
  - **The upgrade makes one resolver the only place a cohort's file is located.** For a
    supplied cohort, that is a registered mapped version. `read_samples` reads the file the
    resolver returns, and the digest is taken from the same read that parses it (the
    contract's M2.1), so the digest recorded is the digest of the samples scored.
  - **Every measurement and audit reader moves onto it:**
    - `measure_scorer` in all its modes;
    - `scripts/audit_split_overlap.py`, including its direct hash;
    - `scripts/audit_generator_fidelity.py`, including its direct hashes
      (`audit_generator_fidelity.py:538-539`);
    - `src/evaluation/cohort.py`'s own overlap reader;
    - `scripts/calibrate_mode_a.py`, which passes `--cohort-kind` through.

    A supplied cohort's argument changes from a split name to a registered cohort version.
    The tests that pass `--split test --cohort-kind supplied`, or test `read_samples`'
    split-name contract, change with it: `tests/integration/test_legacy_equivalence.py`
    (`:53`, `:290-466`) and `tests/unit/test_measurement_mode_a.py` (its supplied-cohort
    cases, from `:54`).
  - **Readers of generated splits only are not moved onto the resolver,** and are listed so the
    claim is not read as wider.
    - `scripts/train_model.py` locates `train`/`val` twice, as `measure_scorer` does today
      (`:449`, `:551-557`, and the check at `:717`). The contract's M2.1 moves it onto the
      shared reader, so it reads each file once and records that read's digest. It still needs
      no cohort resolver.
    - `scripts/measure_served_pipeline.py` and `scripts/probe_deployment.py` read `val` by
      name.

    None of them reads a test cohort. Moving them onto the resolver is a follow-up outside
    this item.
  - **The frozen oracle is the one named exception.** `scripts/evaluate_model.py` is
    byte-pinned (`tests/unit/test_frozen_evaluator.py`). It reads `<split>_samples.json` and
    the graph tensors from its `--data-dir` (`evaluate_model.py:203, 224-234`), and its
    `--split` accepts only train, val or test (`:443`). `calibrate_mode_a.py` and
    `tests/integration/test_legacy_equivalence.py` drive it on supplied cohorts. For them:
    - **a staging directory per run** holds the registered version written as
      `test_samples.json`, checked against the version's digest;
    - **beside it** are links to the workspace's `node_features.pt`, `edge_indices.pt` and
      `num_nodes.json`, checked against the workspace's graph digests;
    - **the driver** passes that directory as the oracle's `--data-dir`, with `--split test`.

    Nothing else reads the staging directory, and no loose file stays in the workspace. This
    exception goes when the frozen oracle retires.
  - Generated cohorts, the workspace's own splits, stay where they are. The measurement and
    audit readers locate them through the same resolver.
  - The report carries the cohort's label and role, the source case count and the exclusion
    counts.
- **Report store.** Every run's report is kept with its cohort version, and disagreements are
  found by key. D5's investigation notes live here, beside the reports they concern. Reports
  publish through the shared staging helper (`src/utils/file_modes.py`), replacing today's
  plain `write_text` (`scripts/measure_scorer.py:669-679`), so a report is never half-written.
- **Ledger v2:**
  - **New fields:**
    - the cohort's version label and role;
    - the source case count, with the exclusion counts;
    - corroborating report digests;
    - and what D4 needs from the ledger: a digest over D4's seven scoring-semantics fields,
      the numerical-regime fields, and a registration sequence number. D4 also reads
      contested status, which comes from the report store (D5);
  - **Rules:** D5's duplicate rule, and its refusal while reports disagree.
- **Ledger compatibility, decided before the format changes: v2 does not read v1.**
  - v2 code refuses a v1 file, as today's code refuses any other version: "read it with the
    revision that wrote it" (§2.1). There is no dual reader, and so no question of how a v1
    record compares with a v2 one.
  - Registering into a directory that holds a v1 ledger is refused, naming the file. The
    operator archives it deliberately, for example as `evaluations.v1.json`.
  - The archived file stays readable with the revision that wrote it, under its original name:
    v1's `record_evaluation.py` reads `evaluations.json` (`record_evaluation.py:83`). Its
    results are re-measured through the new path if they are wanted.
  - Nothing is backfilled, converted or inferred, and no v1 record is displayed. A v1 ledger
    describes old-pipeline models, which are archived with it (§4.9 question 4).
- **Job runner.** `training_manager`'s subprocess core becomes the one job runner, reshaped for
  both jobs rather than extracted around the old one (§4.0).
  - **A test run's inputs are fixed** (D5): the seed, batch size and worker count, and the
    deployment's device. Only the deterministic setting is selectable.
  - **Stop is per job type.** A training stop cannot stop a test job (D8).
  - Training becomes one job type, and the Training Console moves onto the runner. The runner
    no longer hard-codes `train_model.py`; each job type names its own script.
  - **The test job** runs `measure_scorer.py` as a subprocess, so the CLI remains the single
    producer.
  - **One GPU job at a time.** A test start while training runs is refused, with the reason.
  - **The served model may hold GPU memory.** A test that cannot get it fails as a run: it is
    reported, and no record is written.
  - **There is no automatic CPU fallback.** `device` is a semantic field, so a CPU run is a
    different measurement, not a degraded one.
- **UI:**
  - **Test tab** (new):
    - cohorts: import with preview, source sets and mapped versions, usage history;
    - runs: choose a checkpoint from the server's list by workspace and architecture. That list
      is the one listing function, upgraded in §6 step 5 (§2.4). The request names a
      workspace and architecture from those the server lists, and the server resolves the
      directory, so a path never comes from a request or the last training configuration.
      Metadata is read with `weights_only=True` (SPEC_4 A1). Choose a
      cohort version; the checkbox "record the result automatically"; start; follow progress;
    - reports: read them, see disagreement flags, register one.
  - **Model Management** (replacing the placeholder): checkpoints by workspace and
    architecture, with the validation ranking score and each recorded test result, labelled as
    D3 says. The validation score comes from `ranking_score_detail`, the number auto-selection
    used, and is labelled as the trainer's validation metric.
    Ordering follows D4.
  - **Diagnosis:** the loaded checkpoint's recorded results, read-only, with D3's label.
  - **Gating and components:** D8 gates the controls; Gradio's own components are used
    (SPEC_1 §5.1).
- **No change** to auto-selection, to the `.pt` format, or to the served scorer.

## 6. Order of work

**What comes first.** Phase 1 depends on the provenance contract's M1–M3:
- B-2's fail-closed core;
- the model↔graph relation, enforced;
- training, serving and dataset records with their checks.

It does not depend on any phase-2 machinery. Each step below is one reviewable change. F1–F3
are done on this branch (`d95965e`, `88569e4`, `fe44b8f`, `20f43d1`, `a57ed2d`).

### Phase 1 — a complete CLI flow (contract M4)

The aim is that an operator can do all of this from the CLI:
- choose a model and a test cohort that pass the checks;
- run the test;
- keep the full report and record it, or not;
- see the result, correctly bound, when that model is loaded.

1. **The cohort registry, from the CLI.**
   - Import with a printed preview (D6), producing mapped versions bound to the graph.
   - One resolver for the measurement and audit readers (§5).
   - Tests use fixtures only; no real patient data enters the repository.
2. **Measurement and recording.**
   - **Checks:** Mode C on a registered version, with the contract's relations checked: R1,
     R2, R3 (model↔graph), R5, and the cohort↔graph binding. Each is checked on the bytes the
     run read (contract M2.1).
   - **Recorded:**
     - R10 as the contract defines it (§5.4):
       - disease overlap and phenotype-set overlap (same disease, same set of phenotype
         terms) with the model's *recorded training inputs* and its *recorded validation
         inputs*, four labels in all. Phenotype-set overlap does not claim the same scoring
         input: a repeated term changes Mode C's input, and the set ignores it;
       - each label is *overlap* with counts, *none*, *unverifiable*, or, for validation, *no
         validation inputs recorded*. *Unverifiable* is never shown as zero;
       - the evidence is only the sample files whose bytes have the digests the checkpoint
         records. They are found in the measurement workspace or in directories named with
         `--training-evidence`, never assumed from the current workspace;
     - the environment facts (contract §5.3);
     - D6's denominators.
   - **Reports:** kept in the report store, published through the staging helper.
   - **Ledger v2:** keeps today's conservative contradiction refusal (`sidecar.py:342`). It
     refuses rather than arbitrates.
   - **Recording:** automatic with `--record`, or later by hand with `record_evaluation`, on a
     whole report. One recording path serves both.
3. **Display on the existing model status.**
   - The Diagnosis tab's model status shows the records of the checkpoint that is actually
     loaded. They are found by its digest (contract M3c) and read through the API, not from the
     UI process.
   - Each record carries D3's label, its relation states, its environment class and the
     contract's §6 line.
   - **No network-triggered write operation.** Phase 1 adds none, so no new protection
     mechanism is needed. A UI "register" button would be such an operation, and it belongs to
     phase 2.
4. **Documentation and acceptance on the homelab.**
   - The deployment guide.
   - `EVALUATION_COHORTS.md` §6.5, F4 included.
   - Acceptance with a synthetic supplied cohort, and with MyGene2 once its data and terms
     allow.

### Phase 2 — the operating interface (contract M5)

5. **The job runner.** `training_manager` is reshaped into it, with the Training Console moved
   onto it, and test runs go through it. It enforces the GPU-busy rule, test runs have fixed
   inputs, and stop is per job type. The same change upgrades the checkpoint listing:
   - **one listing function** serves both the route and the Training Console's resume
     dropdown;
   - **it lists by a workspace and architecture** chosen among those the server lists, and the
     server resolves the directory. The two Training Console handlers stop setting
     `checkpoint_dir` from request values;
   - **training start's explicit `checkpoint_dir`**, honoured verbatim today, is restricted to
     directories the server resolves in this same change, since it shares the write path
     (contract M5). It is a visible behaviour change, documented with it;
   - **its load becomes `weights_only=True`** (SPEC_4 A1). A checkpoint whose format that
     refuses is listed without metadata.
6. **The UI.** The Test tab, with progress and cancellation, and Model Management. It also
   carries D8: the refusal in each state-changing handler, and the direct-call acceptance
   test.
7. **Advanced ordering and arbitration.**
   - D4's ordering;
   - D5's disagreement block and contested states;
   - the usage history in the UI;
   - a `measure_scorer` switch that requests deterministic execution.

D3's shared-scorer constraint goes to B-1's plan. Mode D and B-1 are separate items, and
neither is a prerequisite here.

## 7. Out of scope

- Downloading cohorts by URL.
- Signing records, an append-only log or a hash chain (D2).
- Refit, and a synthetic test partition (`EVALUATION_COHORTS.md` §5).
- Auto-selection by test score.
- Pooling MyGene2 with the institutional cohort.
- Patient-level display.
- Evaluating Patients-like-me and causal gene discovery.
- Mode D, B-1, and the Training Console's exposure.
