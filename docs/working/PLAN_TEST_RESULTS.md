# PLAN — backlog item 14: test results recorded by the pipeline, shown where a model is chosen

**Status: draft for discussion, revision 2.** Nothing here is implemented. §4 lists decisions
that the owner, and where marked the institution, make before any code is written. §6's order
applies only once they are made.

Facts about the code are cited from `main` at `1cab39f`.

**Revision 2 (2026-10-02)** follows the first review of revision 1 (`a412fc2`) and the owner's
correction on what a test measures. The main changes:
- **D3 is no longer open.** A test result is the model's own disease ranking under the approved
  scorer policy (Mode C), and it is the primary score, not a fallback.
- **D5 now follows the owner's checkbox as stated.** Every run keeps its report, and the checkbox
  only decides whether that report is registered automatically. Non-deterministic repeats are
  handled without losing evidence.
- **D4, D6 and D8 are reworked:**
  - D4: ordering by test score, within comparable results, with the usage history kept;
  - D6: an import preview with exclusion categories;
  - D8: the real access limits, since a Gradio callback can be triggered over the network.
- **F2 and F3 are corrected.** The ledger's v1 compatibility, the job runner's scope and its
  GPU-busy rule are now stated.
- **The owner's questions are narrowed to three** (§4.9).

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
  there). It uses the same pooling and cosine primitives as the served pipeline
  (`src/inference/scoring.py`).
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
  - the Diagnosis tab calls the API over HTTP at a hard-coded `http://127.0.0.1:8000`
    (`diagnosis_panel.py:44`).
- **One job runner.** `training_manager` tracks one subprocess, with `train_model.py` hard-coded.
  It is not a general job runner.
- **Checkpoint listing.** The only listing route is `GET /api/v1/training/checkpoints`. It reads
  whatever directory the training singleton last used. No route lists checkpoints by workspace
  and architecture.
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

- **§4.1.** A valid test cohort:
  - is frozen, versioned and digested;
  - declares its metrics and acceptance rules before the result;
  - is separated from training and from selection;
  - is access-controlled, *so that repeated inspection does not silently convert it into a
    selection set*. Condition 6 is "an operating rule for the institution, not a piece of
    software".
- **§4.2.** MyGene2 and the institutional cohort are **reported separately, never pooled**.
- **§5 item 5.** A cohort is identified by **digest and version label, never by filename or
  path**. Fetching a dataset by URL into a clinical system "would need its own review".
- **§6.7.**
  - "A control that changes a cohort's role is a protocol change, not a display setting."
  - A burned cohort "cannot be returned to untouched by a checkbox, a rename, or a new run
    label".
  - "A UI that permits the relabelling silently is worse than no UI."

## 3. Found while planning — outside item 14

These were not looked for, and each is from reading the code. They touch what item 14 builds
on. **Recording them here does not authorise fixing them.** If the owner wants them fixed, the
fix is one small change of its own, separate from this item.

| | Finding | Evidence |
|---|---|---|
| F1 | **Under the systemd unit, the Diagnosis tab probably cannot reach the API.** The unit passes `--port 8264` after the launcher's default `--port 8000`, and uvicorn takes the last value. The tab calls port 8000. **The fix takes the address from one shared source**, the address the server actually bound. It adds no second hard-coded port and does not probe several | `scripts/service/systemd/shepherd.service:25`; `scripts/launch/shep_launch.py:17, 416`; `diagnosis_panel.py:44`. **Not yet verified by running** |
| F2 | **For checkpoints written by the current trainer, the loaded model's ranking metrics are never shown.** The pipeline looks for `mrr`, `hits_at_1` and `hits_at_10` in the checkpoint's `logs`. The current trainer stores its `val_`-prefixed validation dict there (`val_mrr`, `val_hits@k`), so only the losses appear. Checkpoints from older trainers were not examined and may carry other names, so a fix reads the current names and keeps the old ones | `src/inference/pipeline.py:934`; `src/training/trainer.py:723`; `src/training/callbacks.py:303` |
| F3 | **The Diagnosis tab's model status is not refreshed when the page loads.** Its initial value is computed once, when the app is built, before the server is serving. A page opened later shows that build-time status until Reload is pressed; Reload does update it. `_on_load_status` is defined and never wired | `diagnosis_panel.py:719-722, 754-756, 909-913` |
| F4 | **§6.5 says the ledger key includes mode and tie policy.** The code keys on the semantics digest, which hashes both. A documentation fix | `EVALUATION_COHORTS.md` §6.5; `sidecar.py:157-162` |

## 4. Decisions

Each lists the options and a recommendation. **D3 is settled by the scorer policy**; the rest
are recommendations until the owner confirms them. §4.9 gathers what is actually asked.

### D1 — Where a test result lives

| Option | Consequence |
|---|---|
| Inside the `.pt` | Changes the checkpoint's SHA-256. Every record, evidence file and citation of that model then points at bytes that no longer exist (§5 item 4) |
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

| Option | Cost |
|---|---|
| **(a) State the limit, and claim only binding** | None |
| (b) A server-held key signs each record (HMAC), so an edit by anyone without the key is detected | Key generation, storage, rotation and backup; a lost key leaves every record unverifiable |
| (c) Records also go to an append-only log outside the workspace | A second store to keep consistent |

**Recommendation:** (a) now. (b) only if the institution requires detection of deliberate
edits.

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
  acceptance rule is declared before the first result (§4.1, and §4.9 question 1).

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
pool, the cosine, the candidate universe and the tie rule:
- the pool and the cosine are shared already (`src/inference/scoring.py`, used by `run_mode_c`
  and by the pipeline);
- the universe and the tie rule live in `measurement.py`.

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
independent test (§4.1 condition 6, §6.7). The owner asked for ordering as an advanced option,
so the aim is to make that use visible and comparable, not to forbid it.

**Proposed rules:**

- **Only comparable numbers are ordered together.** Ordering is by one metric over records of
  one cohort version and one `measurement_semantics_digest`. Checkpoints without such a record
  are listed after the rest, marked "not tested on this cohort version". They are never
  interleaved by a number from another version or semantics.
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
differ (§2.1).

- **Evidence is never lost.** Every report is kept, so refusing a registration loses nothing.
- **A disagreement is flagged.** When reports under one key disagree, the Test tab marks the
  key "runs disagree — investigate" and shows which fields differ.
- **Picking the best run is prevented.** While a disagreeing report exists under a key,
  registering any report under it is refused, checked or unchecked.
- **An existing record is flagged too.** If a key already has a record and a later report
  disagrees, the record stays. It is shown flagged everywhere it appears.
- **Resolution adds a note and deletes nothing.** A person investigates, for example by
  re-running with deterministic algorithms, and records the finding: who, when and what was
  found. Both reports stay.

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

This changes the ledger's semantics and gets its own review.

### D6 — How a test cohort is imported, identified and versioned

**Two levels:**

- **A source case set** is the operator's file, identified by its digest. It never changes, and
  re-mapping does not change its identity.
- **A mapped cohort version** is that set mapped to node indices for one graph. It carries:
  - a version label, for example `mygene2-2026-10`;
  - a role: `mygene2`, `institutional_acceptance`, or another named research role;
  - a manifest recording the source digest, the graph's artifact digests (`kg.json` and the
    PyG export whose indices the samples use, `src/kg/artifacts.py:31-36`), the HPO version
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
  category. The denominator shown beside every result is the source case count, with the
  ranked count and each exclusion count next to it.
- **A fixed-denominator figure is proposed alongside.** It counts an unrankable case as a miss:
  reciprocal rank 0, no hit. That keeps a cohort from looking better by losing its hard cases.
  Mean rank has no value for a miss, so this figure omits it.
- **Binding.** A run against a workspace whose graph artifact digests differ from the mapped
  version's is refused. This closes §2.3's undetected-index hazard.
- **No URL download** in this item (§5 item 5). If wanted later, it goes through the existing
  guarded downloader (`src/ontology/download.py`) and its own review.

**Left to the owner** (§4.9 question 2): the real file formats; whether a case that loses some
phenotype terms is kept with the rest or excluded; whether the fixed-denominator figure is
reported; and MyGene2's terms of use, which is not an engineering question.

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

- **Who can reach the controls:**
  - **The controls that change state** (import, run and register) are enabled only under a
    deployment that limits who reaches the UI:
    - SPEC_4's **M1**: a loopback bind on a verified single-workspace deployment, reached
      through controlled SSH forwarding;
    - **M3**, with the institution's recorded acceptance naming these controls;
    - **M2**, once SPEC_4's protection exists.
  - **Elsewhere the Test tab is read-only.** Results stay visible; the controls are disabled,
    with the reason shown.
- **How the server knows.** It reads the address it actually bound (the same shared source as
  F1) and an explicit deployment setting. A non-loopback bind without the recorded acceptance
  disables the controls.
- **Checkpoints and cohorts are chosen from server-side lists**, never from a path in a request.
- **The Training Console** has the same exposure today. It is outside this item, and named here
  so the gap is not read as closed.

**Left to the owner** (§4.9 question 3): who operates, and which deployment mode applies on the
machine where the controls are enabled.

### 4.9 What is asked of the owner

D1, D2 and D7 stand as recommended unless the owner objects. D3 follows from the scorer policy.
Three questions remain:

1. **The test's use.**
   - Confirm D3: a test result is the model's disease ranking under the approved policy.
   - For the institutional cohort: which metrics, and which acceptance rule, are declared
     before the first result (§4.1)?
   - Is ordering by its score offered (D4)?
2. **The data.**
   - The actual file formats.
   - D6's exclusion rules: a case with some lost phenotype terms, and whether the
     fixed-denominator figure is reported.
   - MyGene2's terms of use.
3. **Operators and access.**
   - Who may import, run and register, above all on the institutional cohort?
   - Which deployment mode (D8) does the machine with enabled controls run under?

## 5. Proposed shape, if §4 is taken

This is one pipeline extended, not a second one.

- **Cohort registry** (new, in `src/evaluation/`):
  - **import:** source case sets; preview; mapping; mapped versions with their manifest, staged
    and published as the workspace manifest is;
  - **verification:** the source and samples digests, the graph binding and the manifest schema;
  - **listing:** versions by role;
  - **usage history:** append-only, per mapped version.
- **Measurement.** `measure_scorer`'s Mode C path resolves a registered mapped version. This
  replaces today's loose `<split>_samples.json` for supplied cohorts, so there is one resolution
  path, not two. The report carries the cohort's label, role and denominator counts.
- **Report store.** Every run's report is kept with its cohort version; disagreements are found
  by key.
- **Ledger v2:**
  - **New fields:** the cohort's version label and role, the source case count with the
    exclusion counts, and corroborating report digests;
  - **Rules:** D5's duplicate rule, and its refusal while reports disagree.
- **Ledger compatibility, decided before the format changes:**
  - v2 code reads a v1 file. A field a v1 record lacks is shown as "not recorded".
  - Nothing is backfilled or inferred. A version label is never guessed from a path.
  - Appending to a v1 file raises its header to v2 and leaves the v1 records unchanged.
  - This keeps the current loader's intent (§2.1): a v1 field keeps its v1 meaning, and nothing
    is reinterpreted. A v2 record that duplicates a v1 record under D5's rule corroborates it.
    It does not fill the v1 record's missing fields.
  - v1 code keeps refusing a v2 file, its existing behaviour (§2.1).
- **Job runner.** The smallest extraction from `training_manager` that works: subprocess
  launch, output capture and status. Training-specific code stays where it is.
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
    - runs: choose a checkpoint from the server's list by workspace and architecture; choose a
      cohort version; the checkbox "record the result automatically"; start; follow progress;
    - reports: read them, see disagreement flags, register one.
  - **Model Management** (replacing the placeholder): checkpoints by workspace and
    architecture, with the validation score and each recorded test result, labelled as D3 says.
    Ordering follows D4.
  - **Diagnosis:** the loaded checkpoint's recorded results, read-only, with D3's label.
  - **Gating and components:** D8 gates the controls; Gradio's own components are used
    (SPEC_1 §5.1).
- **No change** to auto-selection, to the `.pt` format, or to the served scorer.

## 6. Order of work

Each step is one reviewable change.

0. **F1–F3**, as a separate small fix, only if the owner authorises it.
1. **The cohort registry**: source sets, preview, mapping, mapped versions, graph binding and
   usage history. Tests use fixtures only; no real patient data enters the repository.
2. **Measurement integration**, the report store, **ledger v2** with the compatibility rules
   and D5's duplicate and disagreement rules, and the `record_evaluation` CLI on reports.
3. **The shared job runner**, and test runs through it, with the GPU-busy rule.
4. **The UI**: the Test tab, Model Management, the Diagnosis display, and D8's gating.
5. **Documentation and acceptance on the homelab**:
   - the deployment guide;
   - `EVALUATION_COHORTS.md` §6.5, F4 included;
   - acceptance with a synthetic supplied cohort, and with MyGene2 once its data and terms
     allow.

D3's shared-scorer constraint goes to B-1's plan. Mode D and B-1 are separate items, and
neither is a prerequisite here.

## 7. Out of scope

- Downloading cohorts by URL.
- Signing records, unless D2 (b) is chosen.
- Refit, and a synthetic test partition (`EVALUATION_COHORTS.md` §5).
- Auto-selection by test score.
- Pooling MyGene2 with the institutional cohort.
- Patient-level display.
- Evaluating Patients-like-me and causal gene discovery.
- Mode D, B-1, and the Training Console's exposure.
