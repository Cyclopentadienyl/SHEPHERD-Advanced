# PLAN — backlog item 14: test results recorded by the pipeline, shown where a model is chosen

**Status: revision 4, corrected 2026-10-16.** **Phase 1 (the contract's M4) is implemented on
this branch (2026-10-16)** (§6, "Phase 1 as implemented"; the contract's §4, "M4 as implemented").
Its internal review's 29 findings (none P1, two P2) are fixed in `f3108c4`; the reviewer's code
review of `59f519d..f3108c4` (2026-10-17) found one P2, fixed in `29c3d1f`, and its incremental
review closed it and found one P3, fixed in `2c94a4e`. **Its GB10 functional acceptance, within
the scope agreed, passed at `2c94a4e`** (2026-10-17; 84 of 84 probes, demo and real data; the
contract's §4, "M4's acceptance on the GB10"). Functional only: not model quality and not an
institutional test. Merged into `main` by the owner (#126, `d2d31dd`, 2026-10-17). Phase 2 (M5)
is not implemented; its boundaries are recorded (§6, "M5's boundaries", 2026-10-19), and it runs
under deployment mode M1 (§4.9 question 3). §4 lists the decisions; phase 1
runs on the engineering defaults §4.9 records, each asked of the owner once, and an answer
changes a default without adding a second path.

**Corrected 2026-10-16, with M4** (the reviewer's instructions, relayed by the owner):
- **Repeated phenotype terms are decided, not open** (§4.9, question 2). The owner adopted the
  rule on 2026-10-07 (`PLAN_PHENOTYPE_NORMALISATION.md` §8): map, then remove repeats by graph
  node, keeping the first occurrence. Three states are kept apart: the principle is decided; N1
  is delivered (serving, and since M4 the importer); N2 (the generator, the reader's check and
  the workspace manifest) is not delivered: it is its own change, not yet scheduled. M4 has no N2
  dependency.
- **"Ledger v2" was a feature name.** The ledger file's `schema_version` was already 2 (R2), so
  the format this plan designs is **ledger schema 3** (§5); no two formats are labelled 2.
- **What M2 and M3 already completed is stated as done**: single reads (M2.1), the checkpoint
  listing over the run layout (M3a-2) and its `weights_only=True` read (M3a) (§2.4, §5, §6
  step 5).
- §2.1's ledger location follows the run layout; citations in the sections touched name
  functions rather than lines.

**Amended 2026-10-19, later:** the reviewer's six decisions, relayed by the owner: D8 gates every
administrative operation, training start and stop included, with no exceptions left; ordering by
test score is answered (§4.9 question 1); import sources come from a server intake directory
(D7); registration takes reports from the registry's report store only (D5); and §6, "M5's
boundaries", adds the loopback default bind and one GPU lock for managed training and
evaluation jobs.

**Amended 2026-10-19:** §4.9 question 3 is answered (deployment mode M1), and §6 records M5's
boundaries ("M5's boundaries"), both from the reviewer's M5 boundaries as the owner relayed
them; N2's bullet in §4.9 records the reviewer's review of batch B (no P1 or P2).

**Amended 2026-10-18, with N2:** §4.9, question 2, records N2 as implemented on the development
branch without its manifest field (batch B; awaiting review, not merged): the shared reader
refuses a stored case that lists a phenotype twice, the generator applies the rule, and the
split manifest's field is deferred (`PLAN_PHENOTYPE_NORMALISATION.md` §6). The 2026-10-16
bullet above is the record as it stood.

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
this branch, and where a note says "since R2" (the Mode A/B retirement's step R2, not the
contract's relation R2): R2 (2026-10-12) left the measurement CLI with Mode C only, no `--modes`, no
`--seed` and no `--num-workers`, and removed the seeds, the loader and sampler fields and the two
`legacy_*` fields from the manifest. R2 also set the ledger file's `schema_version` to 2, so a file
written before it is refused; the "ledger v2" this plan designs (§5) is a later, different change,
and takes the next schema number: **ledger schema 3**, implemented in M4. "Ledger v2" and "v1" in
the revision notes are the old feature names (v1 the ledger before this item, schema 1 and since
R2 schema 2; v2 this item's format, schema 3); everywhere else this plan says schema 3.
**A bare § number refers to this plan.** Sections of `EVALUATION_COHORTS.md` are
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
    oracle was then the named exception (since closed with the oracle, §5);
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
- **A third, focused check (2026-10-05)** corrected the frozen oracle's staging design, a
  directory the oracle would read as its `--data-dir` (since closed with the oracle, §5). It
  also corrected the reader list, now limited to measurement and audit readers with the
  generated-split readers named; the Training Console's resume listing, now a named C3
  exception; D4's field count; and the D3 citations, now at `1cab39f`.

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
`EVALUATION_COHORTS.md` §6.5. The bullets below describe the ledger at `1cab39f`, before M4; the
last one says what M4 changed.

- **The file.** One `evaluations.json` sits beside the checkpoints it describes.
- **The key.** Each record is keyed by `checkpoint_digest`, `cohort_role`, `cohort_digest` and
  a `measurement_semantics_digest` (`record_key`). That digest covers every manifest
  field that can move a number, seeds and the numerical regime included
  (`SEMANTIC_MANIFEST_FIELDS`; since R2 the manifest records no seed).
- **A result stays tied to the weights.** It is bound to the checkpoint's SHA-256, never to its
  file name, so a rename does not lose it.
- **A contradiction is refused.** A second record with the same key is accepted only if it is
  *equal as a whole* (`append_record` at `1cab39f`); anything else is refused, with no override
  flag. The record carries `source_artifact_digest`, the digest of the report file. So
  re-recording the same measurement from a report whose bytes differ is refused even when its
  metrics are identical, for example after the checkpoint has moved and its path in the manifest
  changed.
- **A seed is not bit-determinism.** The ledger's own documentation says so: a seed controls
  the harness's random streams, not CUDA's arithmetic, so a refusal is "an instruction to look,
  not a proof of a defect" (the module docstring at `1cab39f`). Since R2 a measurement records no
  seed, and the ledger's documentation rests the point on the regime fields alone.
- **Writing.** It is single-writer: an atomic replace with best-effort stale-write detection,
  and no locking.
- **Schema.** A file whose `schema_version` is not 1 is refused, with the instruction to read it
  with the revision that wrote it "rather than reinterpreting its fields under new rules"
  (`read_ledger`). Since R2 the version is 2, so a file written before R2 is refused the same way.
- **Location.** `find_checkpoint` looks only directly inside the directory it is given. Under
  the run layout (contract M3a-2, which replaced `1cab39f`'s per-architecture
  `checkpoints/<arch>/`), the ledger is `<workspace>/runs/<arch>/<run>/checkpoints/evaluations.json`,
  beside the checkpoint's resolved path.
- **Readers.** Nothing reads it except `record_evaluation.py --show`. No API route, UI or
  serving code does.
- **Since M4 (merged as #126, 2026-10-17):** ledger schema 3 (§5). The cohort digest
  in the key is the registered cohort version's manifest digest; D5's duplicate rule replaces
  whole-record equality (a second agreeing report is added to the first record's corroborating
  reports); a schema 1 or 2 file is refused and archived by hand, never read or migrated; only a
  registered cohort version's report is recorded. `register_report`
  (`src/evaluation/registration.py`) is the one registration function, used by
  `measure_scorer.py --record` and `record_evaluation.py --report`, and parses and identifies the
  report and the ledger each from one read (BACKLOG item 18). The ledger stays single-writer, and
  sits beside the checkpoint's resolved path on every write path (`--record`, and
  `record_evaluation.py` with `--checkpoint FILE` or `--checkpoint-dir DIR`) and on the read
  side, so a symlinked checkpoint's ledger is beside the link's target. The read-only route
  `GET /api/v1/evaluations/loaded-model` reads it for the Diagnosis tab's model status. The
  contract's §4, "M4 as implemented", has the detail.

### 2.2 The measurement engine, and what each mode measures

`scripts/measure_scorer.py` is the only producer whose output the ledger accepts.

- **How a cohort is given.** It is `<data_dir>/<split>_samples.json` with `--cohort-kind
  supplied`. Even then the workspace's graph artifacts are verified (`read_measurement_inputs`),
  so a test cohort has to sit in a valid generated workspace. **Since M4** that path and
  `--cohort-kind` are gone: a test cohort is a registered version, `--cohort LABEL --registry
  DIR`, and `--split` names the generator's `train` or `val` only.
- **Mode A** could not run on any checkpoint in the scanned family (`BACKLOG.md`, the M1/M2
  finding), and Mode B depended on it. Both were retired by R2 (2026-10-12); the CLI measures Mode C.
- **Mode C is the approved disease scorer, measured.** `run_mode_c`
  (`src/evaluation/measurement.py`):
  - pools each patient's phenotype embeddings (masked mean);
  - scores the result by raw cosine against **every disease node in the graph**;
  - ranks by score, breaking ties by ascending global disease id (`canonical_ranking`).

  That is `DISEASE_SCORER_POLICY.md` statements 1–3, which the institution accepted (§3.4
  there).
- **Mode C and the served pipeline share the cosine, not the pool** (`src/inference/scoring.py`).
  - The cosine is one primitive: the served `cosine_scores` delegates to
    `cosine_score_matrix`.
  - The pool is two implementations, bound by an equivalence test on valid input: the served
    `pool_patient_embeddings` and Mode C's `masked_mean_pool`. The served one clamps an
    out-of-range phenotype index into range. Mode C validates every id first and refuses an
    out-of-range one (`_assert_ids_in_range`, called in `run_mode_c`), because a clamp scores a
    different patient with a plausible rank.
- **What the Diagnosis tab shows today is the legacy behaviour.** It is
  `0.7 × ((cos+1)/2) + 0.3 × SP` behind a BFS discovery gate. The policy replaces it in work
  item B-1, which is gated and not implemented (`DISEASE_SCORER_POLICY.md` §2, §6).
- **Mode D**, the production path-reachable candidates with the η/SP mixture, measures that
  legacy behaviour. It is for B-0.5's attribution work. It is not built, and its design has an
  unresolved problem (`docs/working/scorer-measurement/PLAN_B03.md`).
- **Patient-level output.** The rank side file carries patient `sample_id`s and is written
  without staging (the prediction side file went with Mode A in R2). **Since M4** both files are
  staged and renamed into place (`publish_bytes`); a registered version's ranks stay in the
  registry's report store, outside the repository.

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
- **Since M4 (merged as #126, 2026-10-17)** each of these is closed for test cohorts:
  `scripts/import_cohort.py` maps a `shepherd-cases/1` source (HPO phenotypes, MONDO diseases)
  against the graph a workspace manifest binds and publishes an immutable version with a label
  and a role, bound to the four graph digests and the dataset record, in a registry outside the
  repository (D6, as implemented in the contract's §4, "M4 as implemented"). OMIM and ORPHA
  ground truth, and real file formats, are §4.9's M4 questions 3 and 5. Workspaces still carry
  no version label.

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
  - **Since M3a (2026-10-15)** two of these are done: the route and the resume dropdown list
    through one function over the run layout, `list_run_checkpoints`
    (`src/utils/checkpoint_paths.py`, M3a-2), and the Training Console's handlers no longer set
    `checkpoint_dir`; each file's record is read with `weights_only=True` (decision 10). The
    workspace is still the one the last training request or the console names, not one chosen
    among those the server lists, and the metadata read keeps no digest (BACKLOG 19.3); both
    stay with §6 step 5.
- **Diagnosis tab.** It has no checkpoint picker, only a free-text path and an architecture
  dropdown. **Since M4** its model status shows the loaded checkpoint's recorded test results,
  read-only, through `GET /api/v1/evaluations/loaded-model`.
- **Network exposure.** The launcher's default bind is `0.0.0.0`
  (`scripts/launch/shep_launch.py:17`), and the systemd unit keeps it while SPEC_4 item C1 is
  open. Gradio is mounted at `/ui` in the same app, and every UI callback is an HTTP endpoint.
  Anyone who can reach the port can therefore trigger any UI action, the Training Console's
  included. *Since M5-1 (2026-10-20):* the bind defaults to `127.0.0.1` and every operator
  control, in the API and in the dashboard, checks D8 (D8, "As implemented").
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
| F4 | **EC §6.5 says the ledger key includes mode and tie policy.** The code keys on the semantics digest, which hashes both. A documentation fix | `EVALUATION_COHORTS.md` §6.5; `record_key` in `src/evaluation/sidecar.py`. **Fixed with M4 (2026-10-16)**: EC §6.5's "As built" states the key as the code has it |

## 4. Decisions

Each lists the options and a recommendation. **D3 is settled by the scorer policy**; the rest
are recommendations until the owner confirms them. §4.9 gathers what is actually asked.

### 4.0 One path per concern

The project is moving from older paths to newer, verifiable ones. In that transition, a
compatibility layer easily becomes a second path that has to be kept working and kept agreeing
with the first, although only the newer one will be used. This item adds none:

| Concern | The one path | Not used for it here, and not added alongside |
|---|---|---|
| Producing a test number | `measure_scorer.py`, Mode C | Mode D, which measures the legacy behaviour and is a separate item |
| Scoring | one implementation of the pool, the cosine, the candidate universe and the tie rule, shared by Mode C and B-1 (D3). Only the cosine is shared today | a test-only scorer; two pools |
| Resolving a test cohort | the cohort registry (D6): one resolver locates a cohort's file, and both the digest recorded and the samples scored come from that file (§5) | the loose `<split>_samples.json` supplied path, which was located twice, by `resolve_cohort` and by `read_samples` (removed in M4) |
| Recording | ledger schema 3 | a reader of schema 1 or 2 inside schema 3 code (§5) |
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
    covers the batch formation (and, before R2, the seeds) and the numerical regime,
    `software_revision` included (`REGIME_FIELDS` in `sidecar.py`). Grouping by it would split
    the list at every
    commit, so checkpoints tested at different commits could never be compared.
  - **Scoring semantics is a named list of seven fields,** the ones that say *what* a Mode C
    test computed: `mode`, `cohort_kind`, `candidate_construction`, `score_semantics`,
    `model_construction`, `canonical_tie_policy_version` and `metric_schema_version`.
  - **Four groups are left out:**
    - the seeds and batch formation (since R2 only `batch_size` remains);
    - the numerical regime (the rest of `REGIME_FIELDS`);
    - the sampler and loader fields. Mode C records these from loader defaults but does not
      use them (before R2), so a changed default must not split the group;
    - `legacy_truncation_k` and `legacy_tie_policy`. They are constants on every manifest
      (before R2) that only Mode A's legacy ranking reads, so for the same
      reason: they would split every group when the oracle's surface retires.

    Since R2 the manifest carries neither the sampler and loader fields nor the two `legacy_*`
    fields, so those two exclusions are moot; they stay here as the reason for the list.

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
  (`src/utils/checkpoint_paths.py`) stays on validation under every option.
- **The cohort's usage history is kept.** It holds:
  - every run and every registration against a cohort version;
  - every model load made from a list ordered by that version's score.

  Nothing removes an entry: not a rename, re-map, relabel or "baseline" label. The history
  makes use for selection visible; it does not pretend to prevent it.
- **Per role.** Ordering is offered for research roles, MyGene2 among them. *Since 2026-10-19
  (§4.9 question 1):* by the cohort version's declared purpose, its role, never by the source's
  name; a MyGene2 cohort offers ordering only when its role declares that purpose. For the
  institutional acceptance role, ordering by its score *is* selecting by it.
  **Recommendation:** off by default there, unless the institution decides otherwise (§4.9
  question 1).

**As implemented in M5-1 (2026-10-20), backend only:** `order_by_test_score`
(`src/evaluation/ordering.py`) is one pure function over ledger records of one cohort version
(its manifest digest) and one scoring-semantics digest. It is offered only for a role the caller
passes from `SHEPHERD_ORDERING_ROLES`, and never for `institutional_acceptance`, even when listed.
The fixed rule (`chosen_record`) takes the lowest registration sequence among records with
`regime.deterministic_algorithms` true, else the lowest overall. A record whose sequence or flag
cannot be read is not picked around. A pick whose key is contested, or whose state cannot be
checked (`StoreScan.state`, D5), is listed as not ordered, and the next record is never promoted.
Mean rank is lower-better and ties share a position. One digest with records in two ledgers is
not ordered. A row whose regime is not the most common one is marked. A record on the version
naming another role than the caller's stops the ordering, and each candidate names its ledger
(review, 2026-10-20). The usage history is `<version>/reports/usage.jsonl`
(`src/evaluation/report_store.py`), append-only. The measurement appends each run after its
report is published, before its outcome file, `register_report` each registration that wrote a
ledger, and `record_model_load` serves M5-2's Load from an ordered list. Automatic selection is
kept away from the ledger by the import-linter contract `selection-ignores-test-results` (direct
imports of the reload route and `checkpoint_paths`) and a behavioural test. Model Management and
the Load control are M5-2's.

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
- **Registration takes a report from the registry's report store only** (the reviewer's decisions, relayed by the owner, 2026-10-19).
  Checking for disagreeing reports under the key needs the store, which an outside copy cannot
  give, so `--record`, the Test tab and `record_evaluation.py --report` all go through the one
  registration function on a report the store holds. Its place in the store is not taken as
  proof: the source is located through the registry's own structure and refused when that
  cannot be established, never guessed and never falling back to a copy; the digest,
  checkpoint and cohort checks all stay; and a registry moved as a whole still works, so no
  original absolute path becomes a permanent identity. A copy can still be shared and read; it
  is not an entry point for registration. No database, signature or second writer is added.
  This narrows M4's `record_evaluation.py --report`, which accepted a report anywhere.
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
  randomness, so its seeds, shuffle and worker count did not change what it computes; since R2
  the manifest records none of them.
  - **Batch size and code changes can.** They move numbers at floating-point level: with the
    repository's own pool and cosine, batch size 1 against 32 differs in the seventh decimal.
    A code change can move them further while `model_construction` stays the same.
  - **All of these are key fields** (`SEMANTIC_MANIFEST_FIELDS`), so a re-run under another
    batch size (before R2, also another seed), or after any commit, is a new key. Two rules
    keep that from becoming a way round the block:
    - **the job runner fixes the inputs a test run has no reason to vary:** the batch size
      (since R2 the CLI takes no seed and no worker count), and the device, which is the
      deployment's own with no fallback (§5). Only the deterministic setting is selectable,
      and it is recorded;
    - **D4's fixed rule** picks one record per checkpoint, and later records in the same
      group are shown beside it with any difference marked.
- **The way forward is a deterministic re-run.** `deterministic_algorithms`,
  `cudnn_deterministic` and `cudnn_benchmark` are in the semantics digest
  (`SEMANTIC_MANIFEST_FIELDS`).
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
- **The split is checkable**, as the semantics list is (`SEMANTIC_MANIFEST_FIELDS`). A test
  lists the record's fields, and fails if a new one is in neither the compared set nor the
  excluded set. Schema 3's new fields (§5) are classified the same way when they are added. The
  list of corroborating reports is excluded by its nature.
- **Records of different schemas never meet.** A schema 3 ledger holds only schema 3 records
  (§5). The rule always compares two records of one schema, so no field is ever missing on one
  side.

This changes the ledger's semantics and gets its own review.

**As implemented in M4 (ledger schema 3):** the key stays the four fields, with `cohort_digest`
now the registered version's manifest digest. The compared fields are
`RECORD_COMPARED_FIELDS` (the scoring-semantics digest, the seven scoring fields, the twelve
regime fields, the cohort block, `metrics`, `n_ranked`, `relations`, `dataset_record`); the
excluded ones are `RECORD_EXCLUDED_FIELDS`, each with its reason (`r10`, `environment`,
`source_artifact_digest`, `corroborating_reports`, `registration`). `n_ground_truth_absent` is not
carried: it is 0 in every written report. The same report again is a no-op; another report
agreeing on every compared field is added to the first record's corroborating reports; anything
else is refused, naming the fields that differ. A corroborating report does not update the
record: the record keeps its first report's R10 and environment, and when the later report's
differ, the registration says so in one sentence naming the report that keeps them. The
disagreement block, contested states and investigation notes are phase 2.

**As implemented in M5-1 (2026-10-20):**
- **Registration from the store only.** `register_report` (both entry points) finds the version
  from the report's resolved path, `<registry>/versions/<label>/reports/<run id>/report.json`,
  and resolves it (`locate_store_report`). It then requires the report's cohort block to equal
  `cohort_version_block` of that version, and the run directory to be named for the measured
  checkpoint. A copy elsewhere, or a path off that structure, is refused before it is read,
  saying to register the store's original. A registry moved as a whole still registers. The
  digest, checkpoint and ledger checks are unchanged.
- **The block.** `scan_report_store` builds every stored report with `build_record`, groups the
  records by `record_key` and compares them with `differing_fields`, the comparison
  `append_record` now uses too. A contested key refuses every registration under it
  (`RunsDisagree`), the same report again included. A key that cannot be checked is refused too
  (`DisagreementNotChecked`): an unreadable report, a link out of the store, or a report whose
  key cannot be told affects every key; a report `build_record` refuses affects only the key its
  manifest names. Under `--record` the refusal exits 3 and offers no register-later command;
  without it, the run checks the same way after publishing (`key_refusal`) and prints "runs
  disagree — investigate" instead of the command (review, 2026-10-20). A run's own report that
  no longer resolves into the store is refused as not the one it published.
- **Notes** (`add_investigation_note`) go to `<version>/reports/notes.jsonl`: the key, who (as
  typed, unverified under M1), when and what. They are accepted only for a key that a stored
  report holds, and nothing reads them to decide anything.
- **The switch.** `measure_scorer --deterministic` sets deterministic algorithms (never
  warn-only), cuDNN deterministic, benchmark off and `CUBLAS_WORKSPACE_CONFIG` before CUDA
  starts, refuses rather than falls back, and restores the process's settings. The job runner's
  hooks are `--expect-checkpoint-sha256`, `--expect-version-sha256`, `--outcome FILE` (a new
  file: an existing one is refused, so no earlier run's outcome reads as this one's) and
  `registration.report_state` (kept and registered, read from disk).
- **Not wired yet:** the contested state on the read-only route and the Diagnosis block, which
  need the registry setting (M5-2, through `StoreScan.status`). The scan is not atomic with a
  run publishing into the version (one writer).

**As implemented in M5-2 (2026-10-20), the Test page** (`src/webui/components/test_page.py`):
each run directory of a version's store is listed with kept and registered from
`stored_report_state` and its key's state from `scan_report_store`; a contested key reads "runs
disagree — investigate" with the fields that differ, its reports' Register is off, and the
investigation notes kept with the key (who, when, what) are listed and added, gated, lifting
nothing. Register calls `register_report` (store only) with the checkpoint the server lists
holding the weights the report measured, found by full digest in the workspace's checkpoint list
(never the report's recorded path), after the report's bytes are compared with the digest shown;
the one function checks the store, the version, the block and the checkpoint's bytes again. For
the selected report, an operator's page asks `report_state` again with that listed file, so "not
registered" is said for the ledger a registration from the page would write. *Review fixes
(2026-10-21):* Register and that answer find the listed file in one order: the file the run read
when it is listed with those weights (the ledger `stored_report_state` reads), then a listed copy
in the workspace the run measured, then one in the workspace shown, so a report measured on one
workspace is not filed beside a copy in another (a digest with records in two ledgers is left
out of D4's ordering). Every row is asked again from the session's checkpoint listing; another
workspace is listed only for the selected report, D8 asked on that request. Notes that cannot be
read are said, never shown as "no note".

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
and MyGene2's terms of use, which is not an engineering question. **Since M4** the first three
are asked once with an engineering default each (§4.9, M4 questions 1, 2 and 5), and M4 adds
two about the data (questions 3 and 4).

### D7 — Patient data in the institutional cohort

**Proposed rules:**

- **Patient-level files stay on the machine and out of the repository and all evidence.** These
  are the source file, the mapped samples, and the per-patient rank and prediction files.
- **The UI shows aggregate metrics only.** The import preview lists excluded cases by their
  case identifier only, on the machine.
- **Sources come from a server intake directory** (the reviewer's decisions, relayed by the owner, 2026-10-19). The operator places a
  `shepherd-cases/1` file in it and chooses it from the Test tab's list; the same importer
  previews and commits it. The intake directory and the registry are both outside the
  repository and set by server configuration; the backend checks the chosen file again and
  refuses any other path or a link that leaves the directory. The browser preview shows only
  the exclusion reasons and their counts; the per-case exclusion detail stays in a controlled
  record on the machine, for troubleshooting. **No browser upload in M5:** that is this
  round's scope, not a judgement that an upload is unsafe; it would later feed the same
  importer once a controlled staging area and its cleanup exist (BACKLOG item 25). *As
  implemented in M5-2 (2026-10-20):* the Test page lists the intake directory's `*.json` files
  by name (`server_lists.list_intake_files`; nothing read to list them) and resolves the chosen
  name in that listing again (`resolve_intake_file`), refusing any other name, a path or a link
  that leaves the directory; the preview and the commit run `prepare_import` and
  `commit_import` in the service (on the CPU; the graph is read), one at a time, and the page
  keeps the manifest's aggregates only, so no case id reaches the browser; Commit prepares again
  and commits under the preview id shown. There is no upload component. *Review fixes
  (2026-10-21):* a source refused as malformed (`MalformedSource`, whose problems name case ids)
  is shown by case index only, at most ten problems, with the command-line preview named for
  the ids; the job card's output tail and error line leave out the patient id a measurement
  refusal names.
- **They are stored outside the repository tree**, at a path set by server configuration and
  never by a request (SPEC_4: a client-supplied path is not trusted). *As implemented in M4:*
  phase 1 is CLI only, so the registry is the operator's `--registry DIR` on the machine: an
  existing absolute directory, refused when it resolves inside the repository. Each directory
  below it is resolved again right before it is written into or a version is read from it, and
  refused when a link takes it into the repository or outside the registry; those checks are not
  atomic with the write (one writer, the operator's machine). A refusal of a malformed source
  names where a value is wrong, never a phenotype or a diagnosis. No request can name the
  registry, and no configuration key is read for it yet; phase 2's controls need the server
  configuration this rule names. *As implemented in M5-1 (2026-10-20):* the service reads the
  registry from `SHEPHERD_REGISTRY` and the intake directory from `SHEPHERD_INTAKE_DIR`
  (`src/config/server_settings.py`), at each use and never from a request. Both are checked by
  the registry's one rule (`check_registry_root`, which `validate_registry` applies to
  `--registry` too: absolute, existing, outside the repository). Unset or refused, their
  features are off and say why; no default path is assumed.
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
- **Since 2026-10-19 there are no such exceptions** (the reviewer's decisions, relayed by the owner, 2026-10-19). Administrative
  operations are restricted in the backend (§6, "M5's boundaries"), so one check covers, in M5:
  - import, test start, registration and the checkpoint list, as above;
  - **training start and stop**, through the Training Console, the API routes and the Gradio
    events alike; the stop stays per job type;
  - **the operations the multi-frontend audit found ungated:** pipeline reload, the UI
    configuration save, Runtime Settings' restart and apply, the training-path save, and
    loading a model from Model Management;
  - the Training Console's workspace comes from the server's list; a parent checkpoint for
    resume or extend, if still typed, only locates an item the server lists and is checked
    like the rest. The CLI keeps its typed paths: it runs on the machine.
  The check reads the triggering request's own arrival: a WebUI handler that reaches the API
  through a loopback self-call is checked in the handler, never by the API's view of that
  self-call.
- **As implemented (M5-1, 2026-10-20).** One check, `check_operator_access`
  (`src/utils/operator_access.py`), a pure function of the triggering request's own ASGI scope
  and the deployment setting; one FastAPI dependency (`src/api/operator_access.py`) and one
  Gradio helper (`src/webui/operator_access.py`, reading `gr.Request.request.scope`, the HTTP
  call that started the event under the queue and with `queue=False` alike) call it, and
  nothing else decides. An operator control is allowed only when all hold, and the first that
  fails is named:
  - `SHEPHERD_DEPLOYMENT_MODE` is exactly `M1` (read at each check);
  - the request can be read (an HTTP or WebSocket scope with a host and port `server`, and
    headers as byte pairs): anything unreadable, or an exception while reading, is refused;
  - `scope["server"]` is a loopback address (127.0.0.0/8, `::1` or an IPv4-mapped form);
    `scope["client"]`, which uvicorn rewrites from `X-Forwarded-For`, is never read;
  - no forwarding header: the four above, and also `X-Forwarded-Proto`,
    `X-Original-Forwarded-For`, `CF-Connecting-IP` and `True-Client-IP`;
  - exactly one `Host`, naming `localhost`, `127.0.0.1` or `[::1]` with any port (the DNS
    rebinding guard, added to D8's three checks);
  - an `Origin`, when present, equal to `http://<Host>` or `https://<Host>` (the cross-site
    request guard, added likewise);
  - a `Sec-Fetch-Site`, when present, equal to `same-origin` or `none` (added after review,
    2026-10-20): a GET another site's page makes with an image, a script or a link carries no
    Origin, so the Origin check alone let such a page make the server open every checkpoint of
    a workspace through the two gated lists. Current browsers (Chrome and Edge 76, Firefox 90,
    Safari 16.4) send the header; an older one that sends neither is not caught on such a GET.
  A refusal names the condition in words an operator can act on, ends "Nothing was done." and
  never repeats a header's value. Gated: the API's reload, configuration save, training start
  and stop, and both checkpoint lists (`GET /training/checkpoints`, and
  `GET /workspaces/{name}/checkpoints`, mounted now); in the dashboard, Diagnosis Reload and
  Save Config, Runtime Settings Restart and Apply, and the Training Console's Start, Stop,
  Continue, Check checkpoint, run list and path save. The Console's run list and Check
  checkpoint open checkpoint files, so they follow C3 like the route; the earlier exception for
  the resume dropdown is closed. Gated handlers call the shared services in process: the
  Diagnosis Reload runs the route's own function (`run_reload`), on the event loop as the route
  does, and Save Config the API's one atomic writer (`write_ui_config`); none calls the API over
  HTTP. Read-only routes stay open. A read-only status, `GET /api/v1/system/operator-access`,
  answers for the request that asks (for M5-2's chip). A reload D8 refuses no longer marks a
  real pipeline as requested. Tests call every gated route and every gated event through
  Gradio's HTTP API, refused and allowed. The import, test start and cancel, registration and
  load controls M5-2 adds are to take the same dependency or helper. *M5-2 (2026-10-20), the
  Test page:* listing the intake directory, Preview, Commit, Start, Cancel, Register, Add note
  and the checkpoint list each call the Gradio helper on their own request first and then the
  shared service in process (the job runner, the importer, the registration, the report store);
  a read-only request is shown every control disabled with the reason and no checkpoint list,
  and the lists a start uses are the session's state, never values a client posts. No API route
  was added.

**Left to the owner** (§4.9 question 3): who operates, and which deployment mode applies on the
machine where the controls are enabled. *Answered 2026-10-19:* deployment mode M1, operated by
the owner or a designated technician (§4.9).

### 4.9 What is asked of the owner

D1, D5 and D7 stand as recommended unless the owner objects; D5's ledger rule also gets its own
review. D2 is settled by the owner and D3 by the scorer policy. Questions 1 and 2 gated phase 1;
since M4, phase 1 is built on the engineering defaults of "Asked once with M4" below, and what is
still open in questions 1 and 2 is either one of those questions or not engineering. Question 3
gates phase 2; question 4 is answered:

1. **The test's use.**
   - Confirm D3: a test result is the model's disease ranking under the approved policy.
   - For the institutional cohort: which metrics, and which acceptance rule, are declared
     before the first result (EC §4.1)? **Open, and not an engineering question:** it is the
     institution's, and it is declared before the first institutional result.
   - Is ordering by its score offered (D4)? Phase 2. **Answered 2026-10-19** (the reviewer's decisions, relayed by the owner, 2026-10-19):
     - a cohort version whose declared purpose is exploration, comparison and model selection
       offers ordering; the purpose is its role, never inferred from the source's name;
     - `institutional_acceptance` does not, until the institution's formal evaluation rules
       exist; *as implemented (M5-1, 2026-10-20):* the roles that offer ordering are
       `SHEPHERD_ORDERING_ROLES` (none by default), and `institutional_acceptance` is ignored
       even when listed (`role_offers_ordering`, the one predicate D4 asks);
     - only records of one cohort version with comparable scoring semantics are ordered
       together, and a contested record (D5) is left out;
     - "Load this model" from that list is offered, gated by D8 and recorded in the usage
       history; no model is ever chosen automatically by its test score;
     - **a cohort whose scores were used to choose a model has taken part in model selection**,
       so it is no longer an independent final test for that model; a manual choice is no
       different, and its record in the usage history does not undo that. The UI says so where
       ordering is offered.
2. **The data.**
   - The actual file formats: M4 question 5 below.
   - D6's exclusion rules: a case with some lost phenotype terms (M4 question 1), and whether
     the all-source figure is reported (M4 question 2).
   - A case that lists a phenotype term twice. **Decided by the owner on 2026-10-07**
     (`PLAN_PHENOTYPE_NORMALISATION.md` §8): map, then remove repeats by graph node, keeping the
     first occurrence; keep the source bytes; record the rule version and counts. Three states,
     kept apart:
     - **the principle is decided**;
     - **N1 is delivered**: serving since `af49bd3` (2026-10-07), and since M4 the importer,
       which calls `normalise_phenotypes` (`src/kg/phenotype_normalisation.py`) after mapping,
       keeps the source bytes verbatim, and records the rule's version
       (`mapping.phenotype_normalisation_version`) and its counts (`repeats_removed`,
       `cases_with_repeats`) in the version manifest;
     - **N2 is implemented on the development branch without its manifest field** (2026-10-18,
       BACKLOG item 16, batch B; reviewed by the reviewer with no P1 or P2 at `2e099b6` (2026-10-19), not merged): the
       generator applies the rule, and the shared reader refuses a stored samples file with a
       repeated phenotype, naming its producer. The workspace manifest's field is deferred to
       the next split-manifest schema change or the first change of the rule's version
       (`PLAN_PHENOTYPE_NORMALISATION.md` §6). M4 rebuilt no workspace and has no N2
       dependency: the importer normalises through N1's function, Mode C reads the mapped
       samples, and R10 compares sets; since N2, R10 evidence with a repeat is *unverifiable*.
   - MyGene2's terms of use. **Open, and not an engineering question.**
3. **Operators and access.**
   - Who may import, run and register, above all on the institutional cohort?
   - Which deployment mode (D8) does the machine with enabled controls run under? Under this
     revision only deployment mode M1 enables them today: mode M3 does not unlock them, and
     mode M2 does not exist yet. These are SPEC_4's deployment modes (D8), not the contract's
     milestones.
   - **Answered (2026-10-19; the reviewer's M5 boundaries, relayed by the owner):** M5's first
     version runs under SPEC_4/D8's deployment mode **M1**: the owner or a designated
     technician operates it, on the machine or through controlled SSH forwarding. The
     institution's multi-user remote operation is planned separately, and this is not asked
     again unless the deployment changes. D8's rules hold as written (the check in each
     handler, on the triggering request's own arrival, never a loopback self-call made by the
     WebUI; server-side lists; the named exceptions recorded as exceptions), and no claim is
     made that the system's access control as a whole is complete.
4. **Existing ledgers — answered by the owner's decision (contract §1, decision 2).**
   Old-pipeline artifacts are not supported. Any earlier `evaluations.json` (schema 1, or 2
   since R2) describes old-pipeline models, and is archived with them. Schema 3 starts clean
   (§5): since M4 a schema 1 or 2 file is refused, naming the file, and the operator archives it
   by hand (for example as `evaluations.v2.json`).

**Asked once with M4 (2026-10-16).** Each changes behaviour, so each is asked once, with the
default M4 runs on and a concrete example. An answer changes a default; it adds no second path.

1. **A case that loses some, but not all, of its phenotype terms in mapping.** *Default:
   keep* (`import_cohort.py --partial-phenotypes keep`): the case is ranked on the terms that
   mapped and counted in `cases_with_lost_terms`; `--partial-phenotypes exclude` is supported
   and excludes it as `phenotype_unmappable`. Either policy is recorded in the version
   (`mapping.partial_phenotypes`), and a case with no mappable term is always excluded. *Example:*
   a case lists three HPO terms and one is absent from the graph's HPO release; under `keep` it
   is ranked on two, under `exclude` it leaves the denominator. *Recommendation:* keep, so a
   cohort does not look better by dropping its harder cases, the effect D6's all-source figure
   guards against.
2. **The all-source figure** (D6). *Default:* not computed or displayed. Coverage
   (`n_ranked` of `source_cases`) and every exclusion count are recorded and shown beside Mode
   C's metrics, so the figure stays derivable without re-measuring. *Example:* D6's 100 source
   cases with 20 excluded and the other 80 ranked first show MRR 1.0 over 80 with coverage
   80/100; the all-source MRR would be 0.8 over 100. Should the display show it, labelled with
   its own denominator? *Recommendation:* not by default; add it, so labelled, if the protocol
   the institution declares names it.
3. **Ground truth given as OMIM or ORPHA.** *Default:* refused. A source naming an `OMIM:`,
   `ORPHA:` or `Orphanet:` disease is refused as a whole, naming the namespace and the count;
   the graph's diseases are MONDO terms. *Later:* translation through the MONDO file the dataset
   record verifies, with an ambiguous cross-reference excluded under its own category, never
   resolved by the build's current last-match rule. *Example:* a MyGene2 export whose diagnosis
   column holds `OMIM:` numbers is refused today. Is MONDO-only acceptable for the first real
   cohort, or is the translation needed first? *Recommendation:* MONDO-only until a real source
   that needs the translation is in hand; then build it on the verified MONDO file.
4. **Obsolete and alternative HPO ids.** *Default:* they are not nodes of the graph, so they
   count as unmappable and are listed by term (`unmappable_terms`, with the number of cases
   listing each). *Later, with real data:* following `replaced_by` needs the HPO file verified
   against the dataset record. *Example:* a case recorded under an older HPO release lists a
   term that release later made obsolete; today that term is lost, and under question 1's
   default the case is ranked on its other terms. *Recommendation:* keep counting them until real
   data shows how many there are; the preview lists them by term before anything is committed.
5. **Real file formats** (MyGene2, the institution's). *Default:* only `shepherd-cases/1` is
   read, the project's shared import format (its name says how a file is laid out, not whether
   its cases are synthetic or real); a real format gets an adapter that writes it,
   once de-identified samples exist. *Example:* a phenopacket or a CSV of case id, HPO list and
   diagnosis is converted to `{"format": "shepherd-cases/1", "cases": [...]}` before import.
   Which formats, and who supplies a sample to build the adapter against? *Recommendation:* one
   adapter per real format, writing `shepherd-cases/1`, so there is still one importer and one
   mapping rule.
6. **A generated split's report is not registrable.** *Default:* the ledger records results on
   registered test cohort versions only. `measure_scorer.py --split val --output …` still
   measures and writes its report, but `--record` is refused with `--split` and
   `record_evaluation.py` refuses that report ("a generated split's report is not registered"):
   `val` is the selection split, and D3 labels a result by cohort role and version. Confirm.
   *Recommendation:* confirm; a `val` number stays in its own report as checkpoint-selection
   data.

**Checked against the literature (the reviewer, 2026-10-17).** The reviewer checked the six
questions against the SHEPHERD paper and its code [S1–S3], PhEval [S4], STARD 2015 [S5], Mondo
[S6], the OBO format guide [S7] and the Phenopacket schema [S8]. **The owner found the
recommendations reasonable (2026-10-17).** They keep M4's defaults above for its functional
acceptance and add no gate to it. They are not the institution's protocol: its target
population, primary metric, denominators, mapping and exclusion policy, model choice and pass
threshold are still declared before its first result, and a paper does not set a threshold or
license data. MyGene2's terms were not checked. Questions 2–4 describe today's behaviour as a
support limit, not a permanent best practice. Per question:

1. **Keep.** The SHEPHERD code scores only the positive phenotypes its mapping holds
   (`dataset.py:118`) [S2], and PhEval tests robustness with perturbed, less precise
   phenotypes [S4]. Normalise ids where a rule is certain (question 4), score on what maps, record
   the terms and cases lost, and treat a case with nothing mappable as unrankable. Keep or
   exclude is fixed before results are seen, never chosen by score; keep reduces selective
   exclusion and is not shown to be best on every dataset.
2. **Two denominators in a formal evaluation.** Keep Mode C's metrics over `n_ranked` with
   coverage and each exclusion reason, as M4 does. A formal end-to-end evaluation adds a second
   denominator: the cases eligible beforehand with a determinable truth, in which a case the
   system could not rank contributes zero. STARD asks that missing results and exclusions be
   accounted for, without prescribing zero [S5]; a case with no reliable truth, a duplicate or an
   ineligible case is not a model error. *Example:* 100 eligible cases with truth, 80 ranked
   first, 20 unrankable by the system: conditional MRR 1.0, coverage 80%, end-to-end MRR 0.8,
   shown side by side. MRR and hits@k follow from the recorded aggregates times coverage only when
   that denominator and zero rule are exactly the ones used; mean rank does not, and no formal
   figure is computed from rounded display values. A display, if added, reads the same report and
   scorer, not a second evaluator.
3. **MONDO inside is right; refusing OMIM and ORPHA is a support limit.** The SHEPHERD paper maps
   UDN's OMIM diagnoses to MONDO [S1], and Mondo distinguishes exact equivalence from related
   mappings [S6]. With a real cohort that needs it: the ontology version the workspace's dataset
   record verifies; only an exact, unique mapping to a disease in the graph accepted
   automatically; ambiguous and unmapped cases excluded under their own reasons. Never the last
   match, a fuzzy name or a hand edit. No format is guessed before a real sample exists.
4. **Deterministic HPO normalisation belongs to preparing real-data import.** The SHEPHERD
   preprocessing maps old HPO ids to current nodes (`preprocess_patients_and_kg.py:113-115,
   146-147`) [S3], and the OBO guide separates `alt_id`, `replaced_by` and `consider` [S7]. With
   the verified HPO release: an `alt_id` to its unique primary id; an obsolete id only when its
   `replaced_by` chain is unique, acyclic and ends at a graph node; `consider`, several targets or
   none stay unresolved; no ancestor or semantic expansion. The original id, the result and the
   rule and source versions are kept, then the existing dedup by node runs. It goes in the same
   import and mapping layer. Until then the exact lookup is not "full support for historical HPO
   ids".
5. **One shared import format, thin adapters.** PhEval standardises cases as Phenopackets [S4],
   which keep positive and excluded phenotypes apart, beside fields such as onset and severity
   [S8]. `shepherd-cases/1` stays the project's exchange format; a CSV or Phenopacket adapter is
   written once a de-identified sample exists, into the same importer, mapping and Mode C.
   Positive and excluded phenotypes are never mixed, and information the format cannot express
   gets an explicit rule. A format's name says nothing of whether data is real, independent or
   licensed.
6. **Keep train and val out of the test record.** The SHEPHERD paper selects hyperparameters on
   simulated validation, apart from its UDN evaluation [S1]. Train and val stay measured and
   tested, not registered as an independent test; an independent synthetic test cohort can be
   registered. Renaming or importing `val` does not make it independent, R10 keeps disclosing the
   overlap, and the probe's `val`-derived cohort shows only that the flow works.

Sources, consulted by the reviewer on 2026-10-17:
- [S1] SHEPHERD, *npj Digital Medicine* (2025), Methods (cohort preprocessing, hyperparameter
  selection): https://www.nature.com/articles/s41746-025-01749-1 (full text:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12181314/). Its tasks differ; the causal-gene task's
  numbers are not an acceptance threshold for Mode C's disease ranking.
- [S2] SHEPHERD code at `e95433a`, `shepherd/dataset.py:118`:
  https://github.com/mims-harvard/SHEPHERD/blob/e95433a/shepherd/dataset.py#L118
- [S3] SHEPHERD preprocessing at `e95433a`, `data_prep/preprocess_patients_and_kg.py:113, 146`:
  https://github.com/mims-harvard/SHEPHERD/blob/e95433a/data_prep/preprocess_patients_and_kg.py#L113
- [S4] Bridges et al., PhEval, *BMC Bioinformatics* 26:87 (2025):
  https://doi.org/10.1186/s12859-025-06105-4
- [S5] Cohen et al., STARD 2015 explanation and elaboration, *BMJ Open* (2016), items 15, 16, 19:
  https://doi.org/10.1136/bmjopen-2016-012799 (a reporting guideline for diagnostic accuracy
  studies, borrowed for its transparency requirements; it does not define MRR).
- [S6] Mondo, "About" and "How was Mondo created": https://mondo.monarchinitiative.org/
- [S7] OBO Flat File Format Guide 1.4 (`alt_id`, `replaced_by`, `consider`; a companion draft, not
  a normative standard): https://owlcollab.github.io/oboformat/doc/GO.format.obo-1_4.html
- [S8] GA4GH Phenopacket schema 2.0, PhenotypicFeature:
  https://phenopacket-schema.readthedocs.io/en/2.0.0/phenotype.html

**Not asked again — decided:** the repeat rule (above), no legacy compatibility path (contract
decision 2), no silent CPU fallback (contract §5.2). **Still open, and not engineering:** the
institution's declared metrics and acceptance rule before its first result (question 1 of the
list above), and MyGene2's terms of use (question 2 of it). Real institutional testing waits for data, terms and a
pre-declared protocol; M4's functional acceptance on a synthetic cohort does not replace it.

## 5. Proposed shape, if §4 is taken

This is one pipeline extended, not a second one.

- **Cohort registry** (new, in `src/evaluation/`):
  - **import:** source case sets; preview; mapping; mapped versions with their manifest, staged
    and published as the workspace manifest is;
  - **verification:** the source and samples digests, the graph binding and the manifest schema;
  - **listing:** versions by role;
  - **usage history:** append-only, per mapped version.

  *As implemented in M4:* `src/evaluation/cohort_registry.py` with the CLI
  `scripts/import_cohort.py` does the import (preview, then commit by preview id) and the
  verification (`resolve_version`, `verify_cohort_binding`). Listing and the usage history are
  not built; they serve phase 2's Test tab and D4.
- **Measurement.** `measure_scorer`'s Mode C path resolves a registered mapped version.
  - **A cohort's file was located twice.** `resolve_cohort` (`src/evaluation/cohort.py`)
    located `<split>_samples.json` for the provenance and the recorded digest, and
    `read_samples` (`src/kg/storage/file_storage.py`) built the same path again for the samples
    scored and audited. Changing only one would have recorded a registered version's digest
    while measuring the loose file. Since M2.1 (S7, S9) each file was read once, with its digest
    taken from that read, by the measurement and both audits; what remained was the two path
    constructions, and M4 removes the supplied one.
  - **The upgrade makes one resolver the only place a cohort's file is located.** For a
    supplied cohort, that is a registered mapped version. The file is read once and its digest
    taken from the same read that parses it (the contract's M2.1), so the digest recorded is the
    digest of the samples scored. *As implemented in M4:* `resolve_version` reads the version's
    `cohort.json` once, and `read_samples_file` (`src/kg/storage/file_storage.py`) reads its
    `samples.json` once and refuses it, before decoding, unless its digest is the one that
    manifest binds (`SamplesDigestMismatch`). `resolve_cohort(data_dir, split)` resolves the
    generator's `train` and `val` only.
  - **Every measurement and audit reader moves onto it.** As implemented in M4:
    - `measure_scorer` (Mode C alone since the Mode A/B retirement's R2, 2026-10-12):
      `--cohort LABEL --registry DIR`, or `--split train|val` for a generated split;
    - `scripts/audit_split_overlap.py`: `--registry DIR --cohort LABEL` (report schema 6). It
      had no direct hash since S9;
    - `scripts/audit_generator_fidelity.py`: generated splits only, through the two-argument
      `resolve_cohort`; its digests come from its reads since S9;
    - `src/evaluation/cohort.py`'s overlap reader (`verify_generated_cohorts`, which at `1cab39f`
      hashed each generated split by path and read it again through `read_samples` to measure
      train/val disjointness) has compared only, with no read of its own, since M2.1
      (`verify_cohort_reads`). It covers generated splits only, so M4 had nothing to move.

    A supplied cohort's argument changed from a split name to a registered cohort version, and
    `--cohort-kind` is gone. The tests that passed `--split test --cohort-kind supplied`, or
    tested `read_samples`' split-name contract, moved with it in M4
    (`tests/integration/test_measure_scorer_run.py`, `tests/unit/test_measurement_reads_once.py`,
    `tests/unit/test_measurement_mode_c.py`, `tests/unit/test_measure_scorer_cli.py` among them).
  - **Readers of generated splits only are not moved onto the resolver,** and are listed so the
    claim is not read as wider.
    - `scripts/train_model.py` reads `train` and `val` through the shared reader once each and
      records those reads' digests since M2.1 (S8). It needs no cohort resolver.
    - `scripts/measure_served_pipeline.py` and `scripts/probe_deployment.py` read `val` by
      name (the former's workload read is BACKLOG 19.8's separate fix).

    None of them reads a test cohort. Moving them onto the resolver is a follow-up outside
    this item.
  - **The frozen oracle's named exception is closed.** `scripts/evaluate_model.py` was
    byte-pinned, so this plan exempted it from the resolver and designed a staging directory for
    the calibration launcher and `tests/integration/test_legacy_equivalence.py` to drive it on
    supplied cohorts. The launcher and the oracle were deleted on 2026-10-10 (the contract's
    M2.1, decisions 3 and 6), so neither the exception nor the staging design is needed.
  - Generated cohorts, the workspace's own splits, stay where they are. The measurement and
    audit readers locate them through `resolve_cohort`.
  - The report carries the cohort's label and role, the source case count and the exclusion
    counts. *As implemented:* the manifest's `cohort_version` block (aggregates only), with
    `relations`, `r10` and `environment` beside it.
- **Report store.** Every run's report is kept with its cohort version, and disagreements are
  found by key. D5's investigation notes live here, beside the reports they concern. Reports
  publish through the shared staging helper (`src/utils/file_modes.py`), replacing the plain
  `write_text` measurement used before M4, so a report is never half-written. *As implemented
  in M4:* `<registry>/versions/<label>/reports/<run id>/`, a new directory per run, with
  `ranks.json` and then `report.json`, which names the ranks by digest, both through
  `publish_bytes`. Finding disagreements by key and the investigation notes are phase 2.
- **Ledger schema 3:**
  - **New fields:**
    - the cohort's version label and role;
    - the source case count, with the exclusion counts;
    - corroborating report digests;
    - and what D4 needs from the ledger: a digest over D4's seven scoring-semantics fields,
      the numerical-regime fields, and a registration sequence number. D4 also reads
      contested status, which comes from the report store (D5);
  - **Rules:** D5's duplicate rule, and its refusal while reports disagree.

  *As implemented in M4:* all four new fields, plus the relation states, R10, the environment
  and the dataset record's origin and digest (D5, "As implemented in M4"). D5's duplicate rule
  is built; the refusal while reports disagree needs the report store's disagreement finding
  and is phase 2.
- **Ledger compatibility, decided before the format changes: schema 3 does not read schema 1 or
  2.**
  - Schema 3 code refuses an earlier file, as the code before it refused any other version:
    "read it with the revision that wrote it" (§2.1). There is no dual reader, and so no
    question of how an earlier record compares with a schema 3 one.
  - Registering into a directory that holds an earlier ledger is refused, naming the file. The
    operator archives it deliberately, for example as `evaluations.v2.json` (`read_ledger`'s
    message names the schema it found, and says its results are not carried over: re-measure
    with this revision and register the new reports, since no earlier report can be registered
    under schema 3).
  - The archived file stays readable with the revision that wrote it, under its original name:
    that revision's `record_evaluation.py` reads `evaluations.json`. Its results are re-measured
    through the new path if they are wanted.
  - Nothing is backfilled, converted or inferred, and no earlier record is displayed. An
    earlier ledger describes old-pipeline models, which are archived with it (§4.9 question 4).
- **Job runner.** `training_manager`'s subprocess core becomes the one job runner, reshaped for
  both jobs rather than extracted around the old one (§4.0).
  - **A test run's inputs are fixed** (D5): the batch size (since R2 there is no seed or worker
    count to fix), and the
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
  - *As implemented (M5-1, 2026-10-20):* `src/api/services/training_manager.py`, reshaped in
    place, with job types `training` and `test`. Running and stopping both hold the one slot
    for either type (a start during a stop was accepted before, with a regression test now);
    `stop(job_type)` refuses across types; each monitor holds its own process; each job keeps
    its own output tail; a test run ends `completed`, `failed` or `cancelled` (training keeps
    `stopped`). The test command line is built from `TestRunInputs` (server lists and
    settings), with expected digests and `SHEPHERD_DEVICE` as the device, no batch size, and
    `--outcome` naming a new file in a directory made for the job. Once a test run ends, what
    it published is read from that outcome file, and kept and registered from the report store
    and the ledger by the registration's one helper (`registration.report_state`, given the
    published digest), never from the exit status; no outcome file means no report was
    published, unless a run directory for its checkpoint appeared meanwhile (a run stopped
    between publishing and writing its outcome), which is read the same way and said so. A
    stop that arrives after the process ended on its own is refused and changes nothing; a
    test run is `cancelled` only when the stop ended it. The restart lock reads any job; the
    stop route is a plain `def`.
    Routes for test runs, and the Test tab, are not in this step.
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
    Built in M4, inside the existing model status.
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
   - **Ledger schema 3:** keeps the conservative contradiction refusal (`append_record`), with
     D5's duplicate rule. It refuses rather than arbitrates.
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

### Phase 1 as implemented (2026-10-16)

**Implemented 2026-10-16; code-reviewed, its functional acceptance passed on the GB10 at
`2c94a4e` (84 of 84), merged as #126 (`d2d31dd`, 2026-10-17).** Built over `main` at `59f519d`
in `6e37208` (the registry and importer, R10, ledger schema 3, the registration function, the
read-only view) and `d3d6ce7` (the measurement's
registered path, the report store, the audits, the tests moved), with the probe's M4 probes and
these documents after them. The full suite passed at `d3d6ce7` (3751 passed, 14 skipped). The
contract's §4, "M4 as implemented", is the record: what was built and where, the decisions taken,
the behaviour changes, what is not claimed, the reviews, and the acceptance ("M4's acceptance on
the GB10").

- **Step 1:** `scripts/import_cohort.py` previews, then commits by preview id, an immutable
  version in a registry outside the repository, mapped through N1's rule; tests use synthetic
  fixtures only (`tests/fixtures/registered_cohort.py`).
- **Step 2:** `scripts/measure_scorer.py --cohort LABEL --registry DIR` checks R1, R3, R5 and
  the cohort↔graph binding on the bytes read (R2 is a generated split's, recorded as not
  applicable), records R10, the environment and every denominator, keeps every report in the
  report store, and registers it with `--record` or later with `record_evaluation.py --report`
  (naming the checkpoint with `--checkpoint FILE` or `--checkpoint-dir DIR`) through one
  function, into the ledger beside the checkpoint's resolved path. A refused `--record` keeps the
  report, prints the `--checkpoint` command that registers it later, and exits 3.
- **Step 3:** the Diagnosis tab's model status shows the loaded checkpoint's records, found by
  its digest through `GET /api/v1/evaluations/loaded-model`. D3's label and the note that until
  B-1 the tab ranks by the legacy mixture introduce the records once, and the R10 definitions and
  the contract's §6 line follow them once: once per block, not per record.
- **Step 4:** the deployment guides' §4.9 and EC §6.5 (F4 included) are written. **Done since
  (2026-10-17):** the reviewer's code review and incremental review (one P2 and one P3, fixed in
  `29c3d1f` and `2c94a4e`) and the bounded GPU acceptance through the deployment probe's formal
  run, passed at `2c94a4e`.
  MyGene2 waits for its data and terms, and the institutional test for data, terms and a
  declared protocol (§4.9).
- **Differs from the text above, by decision:** the registry is a CLI argument, not server
  configuration (D7, as implemented); a generated split's report is not registrable (§4.9, M4
  question 6); the new format is ledger schema 3; step 3's label, note and §6 line are shown once
  per block rather than with each record. **Not built in phase 1:** version listing, the
  usage history, finding disagreements by key, investigation notes (all phase 2).
- **Internal review:** ten dimensions, each finding put to independent skeptics; 29 kept (none
  P1; two P2, the ledger split for a symlinked checkpoint and links below the registry root; the
  rest P3), one refuted, all 29 fixed in `f3108c4`. The contract's §4, "M4 as implemented",
  "Internal review", lists them. It was not the external review, and at the time the GPU
  acceptance was still pending; both have since been done (the reviewer's reviews, and the
  acceptance at `2c94a4e`).

### Phase 2 — the operating interface (contract M5)

5. **The job runner.** `training_manager` is reshaped into it, with the Training Console moved
   onto it, and test runs go through it. It enforces the GPU-busy rule, test runs have fixed
   inputs, and stop is per job type. The same change upgrades the checkpoint listing:
   - **one listing function** serves both the route and the Training Console's resume
     dropdown. *Done in M3a-2:* `list_run_checkpoints`;
   - **it lists by a workspace and architecture** chosen among those the server lists, and the
     server resolves the directory. The two Training Console handlers stop setting
     `checkpoint_dir` from request values. *The handlers no longer set it since M3a-2; the
     choice among server-listed workspaces is still to do;* *M5-1 (2026-10-20): the server
     lists the workspaces under `SHEPHERD_WORKSPACES_ROOT` and resolves one only by finding its
     name in that list (`src/api/services/server_lists.py`); moving the console onto it is
     M5-2's;*
   - **training start's explicit `checkpoint_dir`**, honoured verbatim at the time, is restricted
     to directories the server resolves in this same change, since it shares the write path
     (contract M5). *Superseded by M3a-2:* every run writes into its own run directory, and what
     phase 2 adds is the network-side restriction (contract M5);
   - **its load becomes `weights_only=True`** (SPEC_4 A1). A checkpoint whose format that
     refuses is listed without metadata. *Done in M3a* (decision 10); the read keeps no digest
     yet (BACKLOG 19.3). *M5-1 (2026-10-20):* `read_run_checkpoints` reads each listed file once
     through `read_checkpoint` and keeps the SHA-256 of those bytes, the Trainer's validation
     figure, the dataset fingerprint it records and a load problem as a value;
     `TrainingManager.get_checkpoints`, and so the route and the resume dropdown, use it.
     `list_run_checkpoints` stays the read-free form for the reload's file check.
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

### M5's boundaries (2026-10-19)

The reviewer's boundaries for M5, relayed by the owner. They narrow phase 2 above; this plan, the
contract and BACKLOG stay the record, and no new planning document is added.

- **Deployment:** mode M1 (§4.9 question 3; D8 above). *Its setting (M5-1, 2026-10-20):*
  `SHEPHERD_DEPLOYMENT_MODE`, read at each use. D8's check applies it (D8, "As implemented"):
  only the exact value `M1` enables operator controls, and unset or anything else is read-only
  for them.
- **The default bind is `127.0.0.1`** (the reviewer's decisions, relayed by the owner, 2026-10-19; SPEC_4 B1, §3.1). Another address
  stays possible by explicit configuration, under SPEC_4's rules, and never lifts D8's checks on
  operator controls; a deployment that relied on the old `0.0.0.0` default gets migration notes.
  No account system or institutional SSO is added. *As implemented (M5-1, 2026-10-20):* the
  launcher, `src.api.main:main()` and the systemd unit bind `127.0.0.1`; a bind to any other
  address is refused, nothing started, unless `SHEPHERD_ACCEPT_UNAUTHENTICATED_NETWORK_BIND=1`
  (`server_settings.network_bind`), whose warning says it is risk acceptance, not a control,
  at SPEC_4 §3.1's three enforcement points (the launcher, before it installs anything, reading
  `--fd`, `--uds` and `--host` as uvicorn does; `main()`; and application startup, on the host
  those two pass in as `SHEPHERD_BIND_HOST`); the launcher prints what it binds and starts
  uvicorn with `--no-proxy-headers`; migration notes in the deployment guide (§4.13).
- **One GPU lock for managed training and evaluation jobs** (the reviewer's decisions, relayed by the owner, 2026-10-19). Jobs the
  WebUI or the API start and the formal training and evaluation CLIs (`train_model.py`,
  `measure_scorer.py`) take the same lock, one per deployment whatever the workspace; a busy
  lock is refused with who holds it, never queued. The lock covers the working process's whole
  life, with tests for a stop, an abnormal exit and a parent and child process. What is claimed
  is exactly that: managed training and evaluation jobs exclude each other. `flock` is
  released only when every descriptor of its file is closed, which a child can inherit, and it
  is cooperative, so a GPU process that does not take it is not stopped; the resident
  inference service may still hold memory. No cross-process inference scheduling, multi-GPU
  allocation or automatic model unloading is added.
  *As implemented (M5-1, 2026-10-20):* `src/utils/gpu_lock.py`. The lock file is
  `SHEPHERD_GPU_LOCK_FILE`, or `.shepherd-gpu.lock` under `SHEPHERD_WORKSPACES_ROOT`; with
  neither set the lock is not configured and a CLI says so in its log and runs; a setting that
  is set but unusable refuses. `train_model.py` and `measure_scorer.py` take it right after
  argument parsing, before the device or any input is read, and refuse naming the holder from
  its descriptive `<lock file>.json` (exit 1, "Nothing was written."); the runner checks it
  before a start and its child takes it itself. The descriptor is close-on-exec; tests cover a
  normal exit, SIGTERM, SIGKILL, a program the holder starts (does not keep it), a child
  forked without exec (a holder that ends normally releases it for the child too; one that is
  killed or exits without releasing leaves it with the child until it exits, and the message
  says the recorded holder exited) and two workspaces sharing one lock. Windows uses
  `msvcrt.locking`, untested.
- **Gradio 6.30.0 is M5's first commit** (the owner, 2026-10-20, with the reviewer's
  boundaries). The smallest dependency change: `gradio==6.30.0`, and in `uv.lock` only what it
  forces (gradio-client, starlette, huggingface-hub; brotli and hf-gradio new). The mount
  switches off run history, monitoring, MCP and server-side rendering, and a guard in front of
  the mount refuses the two routes Gradio gives no switch for, `file=<URL>` (fetched
  server-side since 6.20) and `process_recording` (deployment guide §4.10; BACKLOG item 34).
  CORS is tightened in the D8 batch: by default no cross-origin browser access is granted;
  administrative operations are still checked by D8 in the backend; the parent app keeps its
  `CORSMiddleware`, with an empty allow list, so the mounted Gradio follows it. Until then the
  CORS configuration was unchanged (`allow_origins=["*"]`, `allow_credentials=True`), and under
  Starlette 1.9 it echoed every `Origin` with credentials allowed on every response (0.52.1 did
  so only when a cookie was sent, and sent `*` otherwise). *As implemented (M5-1, 2026-10-20):*
  `fastapi.middleware.cors.CORSMiddleware` itself (Gradio defers only to that class, never a
  subclass) with `allow_origins=[]`: no `Access-Control-Allow-Origin` for another origin on
  `/api`, `/ui/config` or `file=`, on a LAN Host too, and every cross-origin preflight is 400;
  same-origin works. Gradio's upload route, which the dashboard does not use and which a
  cross-origin multipart POST still reaches (up to 1000 files a request), is refused in front of
  the mount before its body is read; its 1 MB per-file limit (`max_file_size`) stays as a second
  line of defence (deployment guide §4.10, §4.13). Platforms are
  accepted on the backends actually run: Linux x86_64 (the development environment) and Linux
  ARM64 (the homelab, checked by the reviewer); Windows browsers are in the browser checks, and
  a Windows backend is marked untested until one exists.
- **Several frontends stay possible** (the reviewer's advice on the owner's question, relayed by
  the owner, 2026-10-19). M5 is built for mode M1 and opens nothing on the LAN, but keeps the
  architecture able to serve several frontends through the shared API from one inference
  backend:
  - every frontend reaches the one real pipeline through the shared API: no local-only or
    LAN-only inference, and nothing that assumes a single browser;
  - a case, its result and its exports belong to their own request or session, never to a
    global "current patient" or "latest result". A random session id separates requests;
    it is not access isolation and is never claimed as such;
  - administrative operations are restricted in the backend, never only by hiding controls;
  - each diagnosis uses one consistent pipeline and identity: a reload never mixes two models
    within a running request, and never rewrites a produced result's served identity.
  LAN inference itself is BACKLOG item 28.
- **Scope (steps 5–7):**
  - **one job runner**, reshaped from `training_manager`, for training and tests alike; a test
    runs the real `measure_scorer`, never a second evaluator;
  - **the Test tab** on M4's flow: the import preview and commit, the version choice, start,
    progress, cancel, the report and registration; *as implemented (M5-2, 2026-10-20):* Research
    › Test (`src/webui/components/test_page.py`; deployment guide §4.14): the workspace and the
    cohort version from the server's lists, with whether the version was mapped against the
    workspace's graph; the model from the workspace's checkpoint list (operator only, read once
    per workspace); Start through the job runner with the inputs the server resolves again and
    the digests shown as the expected ones, the batch size and device fixed by the server and
    only the deterministic switch and "Register the result" chosen; the job's state (never green
    when failed or cancelled), elapsed time and output tail, read every 5 s while the page is
    shown; Cancel for test runs only; the outcome as the runner reads it from disk; the report as
    its record (Mode C apart from the Trainer's validation, *unverifiable* never a number);
    Register, the D5 block and notes (D5 above); the import (D7 above). No absolute server path
    is shown;
  - **Model Management** on the shared model list, keeping Trainer validation figures and each
    cohort's Mode C results apart; *as implemented (M5-2, 2026-10-20):* Research › Models
    (`src/webui/components/models_page.py`) places the research preset of the one shared list
    and Load and shapes it through the list's view and ordering, drawing no row and loading
    nothing itself. Ordering by test score is offered only for a version whose role WP1's list
    says offers it (`SHEPHERD_ORDERING_ROLES`; `institutional_acceptance` listed, off, with its
    reason) and comes from WP4's one function over the records each listed checkpoint's ledger
    holds (`read_ledger`), one metric, one version and one scoring semantics (a second semantics
    is a choice, never interleaved), each record key's state from the version's report store
    (`scan_report_store`): groups "Ordered by …", "Contested on this version · not ordered",
    "Not compared on this version · not ordered" and "Not tested on this cohort version · newest
    first", the regime on each ordered row and "regime differs" where it is not the most common;
    the independence sentence of §4.9 question 1 wherever ordering is offered; a refusal by the
    function itself (records naming another role) said, the list left newest first. The detail
    keeps the Trainer's figure in its own card and shows every record as the API's record view
    (`_record_view`) with its contested state ("cannot be checked" is never uncontested), D4's
    pick and the others' difference, and the checkpoint's entries in every listed version's
    usage history. A Load from the ordered list passes the ordering to the shared Load, which
    writes the `model_load` entry; newest first writes none. A request D8 refuses gets no list
    (D8's three conditions named) and "Recorded test results of the model in use" from the API's
    read-only route function, by the served digest, opening no checkpoint, with each record's
    contested state from the report store (reasons naming no server path);
  - **D4 and D5 as planned:** ordering, disagreement and contested states, usage history, the
    deterministic-execution switch, with no change to scoring semantics;
  - **the Diagnosis tab's integration:** the shared layout, the identity display and the
    recorded results;
  - **the Training Console:** only its move onto the job runner and what shared components
    need, not a redesign of its forms. *Amended at the owner's request (2026-10-19, second
    mockup round):* its move onto the job runner; D8 on start and stop; the workspace chosen
    from the server list; its controls in a top run bar; the dataset identity shown at all
    times; a start refused when that dataset changed after it was shown; and the metric charts
    replaced by zoomable ones, with the Epoch Window removed. Its parameter fields are not
    redesigned;
  - **added at the owner's request (2026-10-19):**
    - **navigation in two levels**: "Diagnosis (clinical)" and "Research (pre-clinical)";
      under Research one row of tabs in two labelled groups, Prepare (Datasets, Training) and
      Evaluate (Test, Models); Settings at the header's right; Diagnosis is the first screen;
      *as implemented (M5-2, 2026-10-20):* `src/webui/shell.py`, two `gr.Tabs` (so a cross-link
      switches both levels from Python, `PageContext.go`) that wrap and never use an overflow
      menu; Training is the default research page; the group labels are generated content, not
      clickable, with the grouping in the tablist's accessible name; the browser title follows
      the location; the header row holds the brand, two chips read per request (D8's answer for
      that browser, never a header value; the GPU job from the job runner and the lock's holder
      record, never "free") and Settings. The shell's docstring is the page contract the other
      pages build on (one builder per page, one context: the shared ⓘ, the timers outside the
      tabs and the page shown for their listeners to skip work, the chips, the shared model
      list, the served-model-changed trigger, each page's own style sheet registered beside its
      builder). A research page is read again only when it is the page shown (returning to
      Research refreshes that one page, never the other three, since Models' list opens every
      checkpoint). The page is always drawn light (Gradio's dark mode left its text unreadable on
      the dashboard's colours) and Gradio's own footer is off. Test holds a one-line body until
      it is built; Models is built (Model Management, above; deployment guide §4.14);
    - **the Diagnosis tab's "Model in use" card**, directly above Run Diagnosis, so the model
      is known before a diagnosis, with "Change model…": a drawer on the one shared model list
      and the one gated Load that Models also uses (one component, two views, no second
      implementation); it replaces the Model Configuration accordion and the header's
      "Serving now" strip, so the identity is shown once; *the shared component as implemented
      (M5-2, 2026-10-20):* `src/webui/components/model_list.py`, with a clinical preset (the
      latest four runs, "In use" pinned) and a research preset (Models' table, every run); a
      page shapes its view (one architecture, groups such as "Ordered by …", one extra column,
      a note beside Load) without drawing rows itself. Listing opens checkpoints, so it is
      checked by D8 like the Load; the Load is D8 on its own request, the item found again in
      the server's list and its bytes hashed again, one confirmation from → to with the changed
      rows marked, D8 again on the confirm, **both columns read again** (what is served, as the
      whole served identity and data directory, and what would be loaded: the file's bytes, the
      workspace's graph and SP table; a change in either is said with the new values and
      nothing is reloaded until that is confirmed), the API's one reload (`run_reload`) with
      what the server resolves, the served fingerprint checked against the one shown, the
      served-model-changed trigger set when any part of what is served changed, and a usage
      history entry only for a list ordered by a cohort version's score; the page stops waiting
      after 180 s, counting in that list's own confirmation, and offers "Read again". Placing it
      (the card's drawer, Models) is the Diagnosis and Models pages' part;
    - **Run Diagnosis greyed out and not clickable while no model is loaded**, with a bubble on
      hover saying no model is loaded; the API's demo answer for API clients is unchanged;
    - **the dataset identity**: the workspace name, its build time and one fingerprint, the
      first 12 hex digits of the SHA-256 of its `split_manifest.json` (compared in full,
      never by the short form), which binds the graph files, both sample files and the dataset
      record, so two different dataset builds always differ; *as implemented (M5-2,
      2026-10-20):* one renderer for every page (`src/webui/components/renderers.py`): 12 hex
      shown, all 64 in the tooltip and on Copy, more digits when two listed workspaces share the
      first 12, every comparison on the full digest; the same module draws a model row and the
      Model Facts, with a checkpoint's recorded tests as a state from its ledger, never a
      result's number;
    - **the build time is recorded from now on** (the owner, 2026-10-19): the workspace builder
      writes a `workspace_build.json` beside the manifest (`built_at` in UTC and the code
      revision), descriptive and not bound by the manifest, so two byte-identical builds keep
      one fingerprint; a workspace built before it shows "build time not recorded" with its
      sources' declared releases, never a file modification time; *as implemented (M5-1,
      2026-10-20):* `write_workspace` writes it after the manifest, with schema version 1 and
      `split_manifest_sha256`, the digest of the manifest it describes, so a record left
      beside another manifest reads "not recorded" (`read_workspace_build`); the server's
      workspace list shows it with the full fingerprint, the dataset record's origin and
      binding state, the declared releases, the sample counts and whether it is trainable;
    - **a read-only Datasets page** over the server's workspace list; *as implemented (M5-2,
      2026-10-20):* `src/webui/components/datasets_page.py`, on the server's list
      (`list_workspaces`, read through the dashboard's one workspace reader,
      `src/webui/components/workspaces.py`, which the model list's select uses too), with the
      markers "serving reads this workspace", "same dataset as served" (the full fingerprint),
      "in your Training run bar" and "not trainable: <why>" ("no split manifest" when the file
      is absent, "no readable split manifest" when it cannot be read); the details show the path
      relative to the workspace root and how the workspace compares with the others ("Same
      graph build as …, another cut: …"), and no reason shows an absolute path to any request (a
      link's target outside the root, a refused root setting); the manifests are read once per
      refresh; "Use for training ›" sets this browser session's workspace for Training and opens
      it, changing nothing on the server. Nothing on it opens a checkpoint, so it is not gated;
    - **the charts**: axis-aware wheel zoom (over the X axis only X, over the Y axis only Y,
      inside the plot both, centred on the cursor), drag to pan, one-click and double-click
      reset, keyboard equivalents, and the view kept across the periodic refresh. Neither
      Gradio 5.29.1 nor 6.30.0 does this with its own components (tested in a browser: the
      native plots have no wheel or Y zoom, and `gr.Plot` rebuilds the chart on every data
      change), so a small vendored chart library (uPlot, MIT) with the project's own interaction
      code is used, loading nothing from the network;
    - **the Epoch Window's two defects** fixed with its removal: returning to "All" never
      cleared the locked range (Gradio drops `x_lim=None` from an update), and the refresh
      cache shared by every browser session left a reloaded page or a second browser showing
      "IDLE" and empty charts.
    Recorded for later, not in M5: dataset building and management in the WebUI, a release
    designation for diagnosis, model comparison and housekeeping, navigation follow-ups, and
    exporting and importing a model with its dataset (BACKLOG items 29 to 33).
- **Layout and help** (BACKLOG item 27): in an ordinary window at normal text size, with no
  details expanded, the main operations fit on one screen as far as possible, clear and
  uncrowded. Text is never shrunk, content never clipped and warnings never hidden to avoid
  scrolling. The layout follows the browser's usable width and height, not the monitor's
  resolution.
  - **Wide:** the Diagnosis tab is input, candidate list and the selected candidate's details;
    selecting a candidate updates one fixed details area instead of adding content below.
  - **Narrow or portrait:** candidates and details can be switched, and the necessary vertical
    scrolling is allowed, without nested scroll containers.
  - **Help:** an ordinary field's help sits behind one shared "i" control that works with the
    mouse, the keyboard and a click, can be dismissed, and stays readable when the pointer
    moves into the bubble. Long explanations go into the details area. A refusal's reason, a
    job's state, a synthetic workspace and an unverified source stay visible as short text.
    *As implemented (M5-2, 2026-10-20):* `src/webui/components/help.py`, a registry of help
    texts by component `elem_id` and one head script (the alternative, restyling Gradio's
    `info=` in place, was not taken: Gradio 6.30 puts `info=` inside the field's label, and a
    state-bearing text such as the HGT/AMP refusal travels in it). Every ordinary help text of
    Diagnosis, the Training Console and Runtime Settings moved behind it; `info=` now carries
    only texts that must stay visible. The texts are escaped once in Python and shown with
    `textContent`; the bubble opens on hover, focus and click, stays open while the pointer is
    in it, pins on click and closes on Esc or a click outside; the field's input is labelled by
    its title and described by the help, and a click on the title still focuses the field; the
    bubble is drawn in the dashboard's font, on the side of its ⓘ with room, and never taller
    than the window.
- **The served identity is shown together, never merged.** The identity serving now and each
  result's own identity are kept apart. One shared block shows a readable summary by default;
  expanded, it shows either one in full, with its digests to copy. Beside a result, one of
  three states:

  | State | Shown as |
  |---|---|
  | The full identities are equal | The result's source is what is served now |
  | They differ | The result came from an earlier pipeline; the differing parts can be viewed |
  | What is served cannot be read | What is served cannot be checked yet |

  The whole structure is compared, never a checkpoint or a short digest alone. A reload never
  rewrites a result's source, and the CSV and Markdown exports always use the identity the
  result kept. A status that could not be read, or is stale, never leaves "equal" showing; the
  status is refreshed through the existing flow, with no second provenance system.
- **Acceptance:**
  - the UI and the CLI share the import, measurement, registration and provenance checks: no
    second reader, scorer or ledger, and no fallback to the old pipeline;
  - GPU jobs exclude each other; a stop affects only its job type; a failed or cancelled job is
    never shown as a success;
  - whether a report was kept and whether it was registered is said as the publication contract
    defines, never inferred from the job's exit alone;
  - a direct call to a gated API route or Gradio event meets the same D8 check;
  - the chosen model and cohort are checked again by the provenance contract when the job
    starts, never trusted from an earlier preview;
  - after a model switch no other model's records show; validation, Mode C, coverage, R10 and
    provenance states stay apart, and *unverifiable* is never shown as zero;
  - "A produces a result, B is reloaded, A's result is viewed and exported" keeps A's source;
  - the layout is checked landscape and portrait, at high DPI and zoom, with long disease names,
    many results and warnings, recording the viewport and zoom used; neither content nor the
    main operations are clipped.
  The GPU acceptance uses the existing tools and a qualified model, centred on starting and
  stopping jobs, their exclusion, registration and UI operation; long training, S10 and the
  memory investigation are not rerun unless a related regression appears.
- **Two checkpoints:** the layout direction, confirmed once on a thin, operable prototype in the
  existing Gradio, showing representative wide and narrow layouts and the identity switch (a
  static image may help, but cannot stand in for zoom, scrolling and keyboard checks); then M5's
  delivery for review, once. The owner asks for a mockup before UI work starts and screenshots
  at its completion for acceptance (2026-10-19). Component choices, spacing and wording are
  engineering's; a question is raised only for new product behaviour, a data-format or
  compatibility change, going beyond the deployment boundary, or a conflict with a decision.
- **Not in M5:** a new frontend framework, a site rewrite, a generic UI platform or job
  scheduler; the institution's multi-user remote deployment, a reverse proxy, accounts and
  permissions; real institutional data, MyGene2's terms, clinical metrics and acceptance
  thresholds; a new scorer, Mode D or B-1, choosing a model by test score; branch training,
  the AMP study, HGT Edit 2, other hardware backends and the memory attribution.

## 7. Out of scope

- Downloading cohorts by URL.
- Signing records, an append-only log or a hash chain (D2).
- Refit, and a synthetic test partition (`EVALUATION_COHORTS.md` §5).
- Auto-selection by test score.
- Pooling MyGene2 with the institutional cohort.
- Patient-level display.
- Evaluating Patients-like-me and causal gene discovery.
- Mode D, B-1, and the Training Console's exposure.
