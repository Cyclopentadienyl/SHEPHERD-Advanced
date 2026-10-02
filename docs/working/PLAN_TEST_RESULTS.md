# PLAN — backlog item 14: test results recorded by the pipeline, shown where a model is chosen

**Status: draft for discussion.** Nothing here is implemented. §4 lists decisions that the owner,
and where marked the institution, make before any code is written. §6's order applies only once
they are made.

Facts about the code are cited from `main` at `1cab39f`.

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

## 2. What exists today

### 2.1 The evaluation ledger

`src/evaluation/sidecar.py` and `scripts/record_evaluation.py` implement
`EVALUATION_COHORTS.md` §6.5.

- **The file.** One `evaluations.json` sits beside the checkpoints it describes.
- **The key.** Each record is keyed by `checkpoint_digest`, `cohort_role`, `cohort_digest` and
  a `measurement_semantics_digest` (`sidecar.py:157-162`).
- **A result stays tied to the weights.** It is bound to the checkpoint's SHA-256, never to its
  file name, so a rename does not lose it.
- **A contradiction is refused.** A second record with the same key is accepted only if it is
  *equal as a whole* (`sidecar.py:342`); anything else is refused, with no override flag. The
  record carries `source_artifact_digest`, the digest of the report file. So re-recording the
  same measurement from a report whose bytes differ is refused even when its metrics are
  identical, for example after the checkpoint has moved and its path in the manifest changed.
- **Writing.** It is single-writer: an atomic replace with best-effort stale-write detection,
  and no locking.
- **Location.** `find_checkpoint` looks only directly inside the directory it is given. Under
  the per-architecture layout, the ledger therefore sits in `checkpoints/<arch>/`.
- **Readers.** Nothing reads it except `record_evaluation.py --show`. No API route, UI or
  serving code does.

### 2.2 The measurement engine

`scripts/measure_scorer.py` is the only producer whose output the ledger accepts. The frozen
`scripts/evaluate_model.py` is not one: its report has no manifest, and its `--split` choices
are `train`/`val`/`test`.

- **How a cohort is given.** It is `<data_dir>/<split>_samples.json` with `--cohort-kind
  supplied`. Even then the workspace's graph artifacts are verified (`measure_scorer.py:122-128`),
  so a test cohort has to sit in a valid generated workspace.
- **Mode A** cannot run on any checkpoint in the scanned family (`BACKLOG.md`, the M1/M2
  finding), and Mode B depends on it.
- **Mode C** runs. It scores full-graph embeddings by cosine similarity against every disease in
  the graph, which is the reference method's candidate universe. It is a measurement of the
  *model alone*.
- **The served ranking** is GNN 0.7 + shortest-path 0.3, when both are loaded. That is **Mode
  D, which is not built**, and its design has an unresolved problem
  (`docs/working/scorer-measurement/PLAN_B03.md`). **No existing mode measures what a
  clinician is shown.**
- **Patient-level output.** The rank and prediction side files carry patient `sample_id`s and
  are written without staging.

### 2.3 Test cohorts

- **Format.** A supplied cohort uses the generated format: a JSON list of
  `{patient_id, phenotype_ids, disease_id}`, whose values are **integer graph node indices**.
- **No importer.** Nothing converts HPO or MONDO/OMIM/ORPHA identifiers from an external file
  into those indices. MyGene2 has no loader, no format and no download in this repository.
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
on. The recommendation is to fix them first, in one small separate change.

| | Finding | Evidence |
|---|---|---|
| F1 | **Under the systemd unit, the Diagnosis tab probably cannot reach the API.** The unit passes `--port 8264` after the launcher's default `--port 8000`, and uvicorn takes the last value. The tab calls port 8000 | `scripts/service/systemd/shepherd.service:25`; `shep_launch.py:17, 416`; `diagnosis_panel.py:44`. **Not yet verified by running** |
| F2 | **The loaded model's ranking metrics are never shown.** The pipeline looks for `mrr`, `hits_at_1` and `hits_at_10` in the checkpoint's logs, while the trainer writes `val_mrr` and `val_hits@k`, so only the losses appear | `src/inference/pipeline.py:934`; `src/training/trainer.py:723` |
| F3 | **The Diagnosis tab's model status is computed once, when the app is built**, which is before the server is serving. `_on_load_status` is defined and never wired | `diagnosis_panel.py:719-722, 754-756` |
| F4 | **§6.5 says the ledger key includes mode and tie policy.** The code keys on the semantics digest, which hashes both | `EVALUATION_COHORTS.md` §6.5; `sidecar.py:157-162` |

## 4. Decisions this plan asks for

Each decision lists the options and a recommendation; none is decided here.

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

### D3 — Which score a "test result" is

This is the largest decision.

| Option | What a clinician is told |
|---|---|
| **(a) Mode C: the model alone, every disease in the graph** | A measurement of the GNN's ranking, not of the served ranking, which adds the shortest-path term. Must be labelled so everywhere it appears |
| (b) Build Mode D first | The served ranking itself. It is a separate item with an unresolved design problem (`PLAN_B03.md`) |

**Recommendation:** (a) for MyGene2 as research comparison, labelled "model only (Mode C)".
Whether the **institutional acceptance gate** (§6.0 step 5) may rest on Mode C, or needs the
served ranking and so Mode D, is **the institution's call**. If it needs Mode D, the acceptance
use of this feature waits for it, and the plan says so rather than presenting Mode C as
acceptance.

### D4 — May models be ordered or chosen by test score?

A model list ordered by test score invites choosing by it. A cohort used to choose stops being
an independent test (§4.1 condition 6, §6.7).

| Option | Effect |
|---|---|
| **(a) Show test scores; order and auto-select by validation only** | Test results inform, and do not select |
| (b) Allow ordering by test score, behind a warning; each use is logged against the cohort | The cohort's use for selection becomes visible, not prevented |
| (c) Allow it freely | The acceptance cohort silently becomes a selection set |

**Recommendation:** (a). (b) is a coherent alternative for MyGene2 as research comparison, and
should never apply to the institutional acceptance cohort. Auto-selection
(`select_auto_checkpoint`) never reads test scores under any option.

### D5 — What the checkbox means, and a cohort's status

The owner's unchecked case is a cohort that is **not yet a baseline**. Status is therefore made a
property of the cohort version, not of the run:

- **candidate.** Under evaluation. Its runs are kept as **trial reports**, outside the ledger.
  Results are recorded from trial reports one by one, by hand.
- **baseline.** Frozen and official. **Every run is recorded automatically.** This is the
  checked case, and here it is the only case, so nothing can be skipped.
- **Promotion** from candidate to baseline is a one-way, recorded event, shown in the UI. It is
  not a toggle (§6.7).

**A caution about "choosing a result by hand".** The measurement is seeded. The same
checkpoint, cohort version and settings give the same numbers, and the ledger refuses a second,
different number. So "choose which result to record" can mean choosing *which runs*
(checkpoint × cohort version) to record. It cannot mean choosing the best of several numbers
for one measurement. Two runs that differ are a reproducibility problem to investigate.

**A ledger change this needs.** As §2.1 notes, re-running an identical measurement from a moved
checkpoint is refused today because the report's bytes differ. The proposed rule:
- a record with the same key and the same metrics is a duplicate: the first stays, and the new
  report's digest is noted beside it;
- a record with the same key and different metrics stays refused.

This changes the ledger's semantics and needs its own review.

**Recommendation:** the status model above. The owner confirms whether it matches the intended
workflow.

### D6 — How a test cohort is identified, imported and versioned

**Proposed rules:**

- **A cohort version is immutable.**
- **Its identity** is the digest of its samples file, plus a human **version label** (for
  example `mygene2-2026-10`) and a **role** (`mygene2`, `institutional_acceptance`, or another
  named research role).
- **Import is from a file the operator places on the machine**:
  - HPO CURIEs for phenotypes, MONDO/OMIM/ORPHA identifiers for the disease;
  - mapped to node indices through the graph's own mapping;
  - the cohort manifest records the source file's digest, the graph's `kg` digest and the HPO
    version it was mapped against, and counts of records in, mapped and excluded, each exclusion
    with its reason.
- **A mapped cohort is bound to its graph.** A run against a workspace whose `kg` digest
  differs is refused. This closes §2.3's undetected-index hazard. A new graph needs a re-map,
  which is a new cohort version derived from the same source digest.
- **No URL download** in this item (§5 item 5). If wanted later, it goes through the existing
  guarded downloader (`src/ontology/download.py`) and its own review.

**Questions for the owner:**

1. A record that does not map, such as an obsolete HPO term or a disease absent from the graph:
   - **refuse the whole import**; or
   - **exclude the record**, counted and listed in the manifest.

   Excluding records changes what the cohort measures, so the manifest has to make it visible.
2. **Which file format** the hospital and MyGene2 data will actually arrive in. The importer is
   written against a real example, not a guess.
3. **MyGene2's terms of use** for this purpose. This is not an engineering question.

### D7 — Patient data in the institutional cohort

**Proposed rules:**

- **Patient-level files stay on the machine and out of the repository and all evidence.** These
  are the source file, the mapped samples, and the per-patient rank and prediction files.
- **The UI shows aggregate metrics only.**
- **They are stored outside the repository tree**, at a path set by server configuration and
  never by a request (SPEC_4: a client-supplied path is not trusted).
- **Every run on the acceptance cohort is logged:** when it ran, on which checkpoint, and its
  result. Condition 6 is the institution's rule. The log makes its observance visible; it does
  not pretend to enforce it.

**Question for the owner:** who may run tests on the institutional cohort.

### D8 — The security boundary for running and recording tests

A test run loads checkpoints, which are pickles, and uses the GPU, the same exposure as C3/C4.

| Option | Note |
|---|---|
| **(a) No new HTTP route; the UI calls an in-process service, like the Training Console** | Adds no unauthenticated network surface. The `/ui` path itself is still unprotected, so this assumes SPEC_4's M1, a local single-workspace deployment |
| (b) New API routes, behind SPEC_4's protection | Waits for authentication, which does not exist |

**Recommendation:** (a), with the deployment assumption stated in the deployment guide.

## 5. Proposed shape, if §4's recommendations are taken

This is one pipeline extended, not a second one.

- **Cohort registry** (new, in `src/evaluation/`):
  - import, map and write a cohort version with its manifest, staged and published as the
    workspace manifest is;
  - verify it: the samples digest, the graph binding and the manifest schema;
  - list versions by role and status;
  - promote a version, one-way and recorded.
- **Measurement.** `measure_scorer`'s existing Mode C path resolves a registered cohort version.
  This replaces today's loose `<split>_samples.json` for supplied cohorts, so there is one
  resolution path, not two. The report carries the cohort's label, role and status.
- **Ledger.** A record schema v2 adds the cohort's version label, role and status. The D5
  duplicate rule applies. A v1 ledger is read as it is; whether to migrate or refuse it follows
  the project's rule for older schemas, and is decided in review.
- **Job runner.** The subprocess and monitor core of `training_manager` is extracted and shared,
  so a test run and a training run use one mechanism. The shared runner refuses to start one
  GPU job while another is running.
- **UI.**
  - **Model Management** (replacing the placeholder):
    - checkpoints by workspace and architecture, with the validation score and each recorded
      test result, labelled with the cohort role, version and engine;
    - a test section: import or list cohorts, see their status, run a test, follow its
      progress, read the results, and record a trial result.
  - **Diagnosis.** The loaded checkpoint's recorded results, read-only.
  - Built with Gradio's own components (SPEC_1 §5.1).
- **No change** to auto-selection, to the `.pt` format, or to the served scorer.

## 6. Order of work

Each step is one reviewable change.

0. **F1–F3**, as a separate small fix, before anything here.
1. **The cohort registry**: import, mapping, manifest, verification and graph binding. Tests use
   fixtures only; no real patient data enters the repository.
2. **Measurement integration and ledger v2**, including the duplicate rule, and the
   `record_evaluation` CLI on the new records.
3. **The shared job runner**, and test runs through it.
4. **The UI**: Model Management and the Diagnosis display.
5. **Documentation and acceptance on the homelab**:
   - the deployment guide;
   - `EVALUATION_COHORTS.md` §6.5 (and F4);
   - acceptance with a synthetic supplied cohort, and with MyGene2 once its data and terms
     allow.

Mode D (D3 b) is a separate item. It is a prerequisite only if the institution requires the
served ranking for acceptance.

## 7. Out of scope

- Downloading cohorts by URL.
- Signing records, unless D2 (b) is chosen.
- Refit, and a synthetic test partition (`EVALUATION_COHORTS.md` §5).
- Selecting models by test score, under the recommended D4.
- Pooling MyGene2 with the institutional cohort.
- Patient-level display.
