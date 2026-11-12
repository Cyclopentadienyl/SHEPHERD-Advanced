# Operations wiring and the daily UI/UX flow

**Status: in progress on `claude/ops-wiring-and-ui-ux`, from `main` `30e8ae1` (2026-11-12).** This is the round's
plan and record. The owner authorised it on 2026-11-12 by adopting the reviewer's next-phase plan, rev 2 of
2026-11-12, and its one-time authorisation (both outside this repository, in
`SHEPHERD_next_phase_plan_30e8ae1_20261112_rev2_evidence.zip`). The owner added one condition: the UI/UX work follows
the design rules agreed earlier with the owner and the reviewer, and waits if they cannot be found. They are
[`.claude/skills/webui-design/SKILL.md`](../../../.claude/skills/webui-design/SKILL.md) and
[`../WEBUI_UX_REVIEW_2026-10-25.md`](../WEBUI_UX_REVIEW_2026-10-25.md), so stage B goes ahead.

The owner's terms:

- one authorisation for A and B, built continuously;
- internal checkpoints that pass let the work continue; only the stop conditions (§9) pause the affected part;
- the whole batch goes to one external review at the end;
- the GB10 acceptance on a candidate slot comes after that review; a PR, a merge and any deployment wait for the
  owner's explicit instruction;
- C0 is an assessment only; no second architecture is built until the owner chooses one.

The stages below are a delivery index, not approval gates.

## 1. Goal

**The main goal.** From an existing workspace, an operator can go from settings to a base or a variant, to a full
fine-tune and its frozen-backbone control, to measurement and registration, to comparison and export, to loading a
model and running an inference, and back to research. At each step a page says plainly whether it shows the current
selection, a running job or a past result.

**What this round is.** A: the backend wiring that the flow lacks, through the roots that already exist. B: the daily
UI/UX flow over the pages that exist. Not a new research-comparison platform, not a new training engine.

**Not pass conditions.** An accuracy gain, equal experimental conditions and equal results are not engineering pass
conditions.

## 2. Binding rules

The research-comparison round's rules hold as written
([`../research-comparison/README.md`](../research-comparison/README.md) §2): natural retirement, one parser of known
shapes, one operation retires at a time, dispatch without parallel pipelines, nothing kept or rewritten for
compatibility. This round adds:

- **Reuse the root.** Each capability starts from its existing root (§5). A new interface is wired into the product's
  normal entries (start-up, a route or a manager dispatch, the CLI, a page) in the same change. A class, an enum or a
  test-only path is not a capability.
- **No deliberate incompatibility, no compatibility shim.** What the current implementation can run or read
  faithfully is not refused by version, directory, mode or age. No old path is kept beside the new one, and no extra
  piece is added to the new one to mimic the old.
- **Lock the resource that conflicts.** Exclusion is decided by the resource actually used (the GPU, by the resolved
  device), never by a page, a mode or a request string. No blanket exclusion is added where nothing conflicts.
- **Integrity is not sameness.** Different models, data, seeds, environments or scores are kept and compared side by
  side. Fingerprint and capability checks stay where an operation cannot be valid.
- **Files.** No whole-file deletion. A superseded function or branch retires inside its file in the same change; a
  file that could go is listed (§11). The owner's formal repository, deployment, models, data, reports and ledgers are
  never changed; tests use their own copies.

## 3. The owner's authorisation (2026-11-12)

The eight points, in summary:

1. **Inference is loaded and stopped by the operator.** Nothing unloads silently. CUDA inference loads and stays
   under the deployment's one GPU lock. A stop lets accepted requests finish, and new requests are told that
   inference is unavailable. Pure CPU inference adds no blanket exclusion. The managed job slot is not reworked.
2. **No explicit model config means not loaded after a restart.** No other model is chosen. An existing explicit
   start-up config goes through the shared builder, checks, device and lock, and a failure is reported. No
   "restore the last model" is added.
3. **A branch may rerun with unchanged settings.** It is a new run with a new optimizer, its parent's weights as
   source and an empty `changes`, all recorded as they are. The shared planner and the record reader change
   together. Resume and extend keep their integrity checks. No separate pipeline, and no seed change to get past a
   check.
4. **The UI uses the existing design rules and shared components for the whole flow.** Layout, advanced fields and
   ordinary wording are not asked page by page. Settings, fingerprints, raw values and important reasons stay
   viewable. Every necessary new interface reaches a formal entry and the normal start-up.
5. **Work on the round's own branch.** Product code, tests and documents may be changed or added, committed and
   pushed by reason. Passing A/B checkpoints, regressions and browser checks let the work continue; the batch goes to
   one external review.
6. **Whole-file deletion is not required by default.** Superseded functions and branches retire in the same change.
   Deferred whole-file cleanup is listed at delivery. A truly unavoidable whole-file deletion is asked with a
   concrete list.
7. **Nothing formal is changed.** Not the owner's formal repository, deployment, models, data, reports or ledgers.
   Tests use independent copies or controlled test data. The GB10 candidate-slot acceptance follows the external
   review. A PR, a merge or a deployment waits for explicit instruction.
8. **Not in this round:** new acceleration engineering, an unchosen new GNN, full packaging, a data builder, real
   institutional data, joint training.

## 4. The decisions of this round

The author's, within §3. The plan's wording is the source where this section is silent.

### Settings (A1)

- **S1. One definition of what a new run takes.** `run_plan.new_run_settings(stage, task)` is the learning settings
  minus the operational ones, minus `recipe.unread_settings(stage, task)`: the rule `train()` already applies. The job
  runner's start check, the HTTP request types and the pages read it, and `train()` calls it.
- **S2. Explicit is what the operator sent.** A YAML key or a flag on the CLI, a field present in the body (null
  included) on the API, a field the operator touched on a page. A value equal to the reference is never read as "not
  set". The record names each learning setting's source: reference, operator, build default, inherited or decoded.
- **S3. A setting the run would not read is refused when it is explicit**, as `unread_settings` already is, never
  accepted and ignored: `warmup_steps` unless the scheduler is cosine, `min_lr_ratio` with no scheduler, `amp_dtype`
  with AMP off. Inherited values are never refused for this: a continuation or a branch meets these rules only in the
  settings it changes, and a setting stated at its parent's value is kept as inherited. A request whose effect the
  record already states apart
  (AMP asked on a CPU device: requested and in effect recorded) stays accepted. The trainer's warmup semantics are
  not changed (with cosine, `warmup_steps` 0 means a ratio of the steps); only the words that misdescribe it are.
- **S4. What the child process checks, the runner checks before it starts**, through the same functions: a
  fine-tune's hops against its base's encoder, its split seed against its base's, its model spec's graph side
  against its base's.
- **S5. Architecture options are the implemented declarations.** `ENCODER_DECLARATIONS` (`gatv2` today), pinned
  equal to the builder's implementations. Names that are only in `ENCODER_KINDS` are never offered as trainable.
- **S6. The HTTP request types are thin projections.** Bounds come from the field spec (a `-1` hop is legal where the
  shared check says so). `learning_stage` is optional for a continuation and compared when given, so a fine-tune child
  can be resumed or extended over HTTP. `extend_lr` is added. An explicit null reaches the service.

### Branch (A1)

- **B1. A same-settings branch.** A branch with no setting changed is a new run from the parent's weights with a new
  optimizer and schedule. `operation` stays `branch`; `parent.changes` is `{}`; the parent's digest is recorded.
  `plan_branch` and `ModelRecord.validate` change together. An absent `changes` is still refused: not recorded is not
  "none changed". Resume and extend are untouched.
- **B2. One branch path.** The job runner, `POST /api/v1/training/branch` and the Training page branch through the
  one preview (`plan_branch`) and the one start (slot, run directory, spawn). The parent digest the preview read is
  bound to the process (`--expect-parent-sha256`), which checks it again on its own read.
- **B3. Early stopping patience is branchable.** Its "not this round" reason is not a dependency: a branch builds a
  fresh Trainer. The freeze stays a kept setting (O4a).
- **B4. Lists show what a branch changed**, `{}` as "same settings".

### Freeze (A2)

- **F1. Fine-tune offers the freeze** as a common field (Graph backbone: Trained or Frozen). The model it makes is
  named a task model with a frozen graph backbone wherever a model or a result is described, never the base's score.
  The full fine-tune and the control are two ordinary starts from one base.
- **F2. A short fp32 functional check** of the freeze joins the existing required probe flow (`L7.disease_frozen`):
  on the CPU in CI, on CUDA at the GB10 candidate run. AMP and compile with a freeze stay recorded as not verified;
  nothing is removed for it.
- **F3. Invalid fine-tune settings answer 422**, as a base start does.

### Inference (A3)

- **I1. Explicit load and stop.** Load and Change model use the existing reload. Stop inference is
  `POST /api/v1/pipeline/stop` (operator only). Starting a job never unloads a model.
- **I2. CUDA serving holds the one GPU lock** (`take_gpu_lock`), from before its candidate is read until it is
  released, decided by the resolved device. CPU serving takes no lock. With no lock configured, CUDA serving is
  refused with the jobs' guidance (nothing runs on the GPU without excluding the others); CPU serving is unaffected.
- **I3. A load reads the job runner's slot after it holds the lock**, so a job spawned just before is seen.
- **I4. Stop.** New requests are refused at once, with the reason. Accepted ones finish: no new timeout, no kill.
  Then the model, graph and cache references are cleared and the lock is released, and only then is "released"
  reported. `empty_cache` is not claimed to return memory; the measured reading is shown.
- **I5. Switch and reload.** CPU to CUDA takes the lock first; CUDA to CPU releases it after the old requests end. A
  candidate is checked before it is published; a refused candidate leaves the serving model and its holding as they
  were. Out of memory is refused with "stop inference first", never by unloading first.
- **I6. Start-up.** Without an explicit model config the service starts not loaded, and the page says how to load.
  An explicit environment config loads through the same lifecycle. Busy or failing, it reports not ready with its
  reason and never picks another model.
- **I7. Restart and shutdown stop serving.** Restart Backend drains and releases before it execs. A page load is not
  restored; an environment config loads again through the lifecycle. A killed process's lock is released by the
  kernel; no state file pretends a holding.
- **I8. Busy is shown, not hidden.** While CUDA inference holds the lock, GPU research jobs are refused, naming
  inference and how to stop it. Browsing runs, reports and results, and export, stay available.
- **I9. A diagnosis never loads a model.** `/diagnose` holds a lease on the served pipeline, and its lazy load
  retires.
- **I10. The unconsumed config route retires.** `/pipeline/config` and `write_ui_config` write a file nothing reads.
  They are removed from their file in the same change and not wired as a start-up source. An operator's existing
  `.shepherd_ui_config.json` is left untouched.

### UI (B)

- **U1.** The design rules are the skill's. Its §8 sentence that "best" means the Trainer's val MRR is corrected to
  the run's own recorded selection metric and direction (A0). No other rule text changes.
- **U2.** Shared pieces first (state words and marks, gated buttons, options, details and value words, the help
  bubble's placement, selection words), then the pages in the flow's order.
- **U3.** Not in this round: rewriting all help texts, a draggable dashboard, a framework change, archive or delete
  tools, a new uploader. The export link's cache path is a D7 judgement for the owner (BACKLOG 72).

## 5. Existing roots, this round's wiring, and what is not built

| Capability | Existing root | This round | Not built |
|---|---|---|---|
| Settings and model spec | `training_fields`, `recipe`, `spec`, `train_model.TrainConfig` | S1–S6: the settings the CLI runs reach the API and pages; explicit, inherited and reference kept apart; sources recorded | A page-only recipe; a second list of architectures |
| Resume, extend, branch | `run_plan.plan_continuation`, `plan_branch`, `decode_learning_settings`; the runner's preview and start; `train_model.train` | B1–B4: branch through the runner, API and page; an empty `changes`; the parent digest checked again at the real start | A branch Trainer, launcher or lock; skipping the process's final check |
| Branch record | `record.py` `ModelRecord.validate`; `train_model` `parent_ref` | `{}` is a new run with no change, apart from "not recorded"; planned, written, read and listed alike | A page-only allowance; a seed change or fake entry; a lenient reader; rewritten files |
| Freeze | `recipe.frozen_parts_problem`, `ModelBundle.freeze`, the runner's settings check, `FinetuneStartRequest.settings` | F1–F3: the Fine-tune field, words and the probe | A frozen-experiment family, Trainer or evaluator |
| Model construction and capabilities | `builder`, `spec.EncoderDeclaration`, `capabilities`, `protocols` | Options and capabilities shown as they are; C0 assesses one second architecture | A `conv_type` guesser, `ShepherdGNN`, a second builder, a plugin registry |
| Inference lifecycle | `api.main.AppState`, `build_pipeline`, `publish_pipeline`, `run_reload`, `gpu_lock`, `file_lock`, the runner, `backend_control` | I1–I10 in the one service state | A page busy flag as authority; a second GPU lock; a scheduler; persisting "the last model" |
| Data identity | `cohort_registry`, the KG view, split, sample and fixture checks | Read as they are on Datasets and Test | A test-only registry; val relabelled as test |
| Measure, register, read, compare, export | `protocols`, `task_measurement`, the ledger, the one reader, the results service | Pages select and present | A page that re-reads checkpoints, re-scores or keeps a second database |
| Past curves and identity | `training_manager.get_metrics_history`, the run resolver, `renderers`, `model_list`, Results | A past run's curves; short options carrying the full key; the selection metric named | A second log parser; a display name used as identity |
| Future interfaces E1–E6 | research comparison §6 | Reused as they are | A second receipt, loader, message-passing path or manifest |

## 6. The stages (a delivery index)

Each stage is implemented, read by two reviewers and fixed before the next. Its related tests run in the stage; the
full suite runs at the checkpoints.

- **A0. The blueprint's current state.** The statements made false by #133's merge, and the stale "best" and
  architecture sentences, are corrected; dated history is kept (BACKLOG §0, the research-comparison README,
  ARCHITECTURE, scorer-retraining §3.3, GNN_ARCHITECTURE_NOTES, the skill, the indexes).
- **A1a. Settings projection** (S1–S6).
- **A1b. Branch** (B1–B4), with a branch step in the probe's demo phase.
- **A2. Freeze wiring** (F1–F3), with `L7.disease_frozen`.
- **A3a. The inference lifecycle in the service** (I1–I10).
- **A3b. Its probe, cross-process test and minimal page wiring**: `W12` on CUDA; Stop inference on the Diagnosis
  card; the GPU chip and Restart words.
- **Checkpoint A.** The CLI and the service give the same judgement of the same request and parent. A parent changed
  after a preview is refused at the real start. A branch runs through the one execution root. The inference/research
  hand-off holds in the backend.
- **B0. Shared pieces** (U2).
- **B1. Training**: New, Continue and Branch kept apart; common settings first, the rest one click away; the current
  job apart from a past run being viewed; a past run's curves.
- **B2. Fine-tune**: the touched settings; a single task's own settings.
- **B3. Test, Models and Results**: Run → Checkpoint; measurements by capability; the current selection apart from
  completed results; one fact once.
- **B4. Diagnosis**: load, stop and back to research; each task's inputs; a result keeps the model and data it used.
  The card's not-serving reason, shown whole since A3b (about 50 words with no model, naming environment settings
  and a route), becomes the state word and the reason's lead (who stopped it and when, or why a load was refused),
  the full reason one click away (Details); the page's own "Loading <12 hex> for every session…" line folds into
  the server's Loading state, said once (the skill's §1.2, §4, §5; review of A3b).
- **B5. Datasets, Settings and the footer**: versions, roles and admissible measurements; environment facts; no
  acceleration toggles ("Not implemented yet").
- **Checkpoint B.** From existing data and without editing YAML: a GATv2 base, a variant, a full fine-tune and its
  frozen control, common and task measurement, registration, comparison, export and one inference.
- **C0.** The fit assessment (§10). It blocks nothing.
- **Close.** The full suite, import contracts, a diff check, the browser checks (§8), the GB10 field contract of this
  round's new probes, then the external review.

## 7. Evidence for each new interface

Filled at delivery: which entry calls it; that the normal launcher reaches it; its inputs, outputs and errors through
the shared checks; one legal and one illegal example; the paths it replaced, retired; its GPU, start-up or
concurrency items in the candidate-slot acceptance.

## 8. Acceptance

**Integrity and freedom** (the plan's §7). Two runs under the same conditions keep their own identities, whatever
their scores. Legal results across seeds, data, families and environments are queried, compared and exported, each
protocol with its own meaning and denominator. A fingerprint that contradicts the actual input is refused before the
operation. A missing component refuses only the operation it makes impossible. Automatic checkpoint selection never
reads a test score. One unreadable item never brings the others down. A page option never decides a model's identity.

**Browser** (the skill's §12 for a broad change). The normal launcher; 1850 × 960, 1536 × 760 and 960 × 1080, and
narrow to wide. Operator and read-only. Page refresh, a model switch, a job's start, end, failure and cancel, no model
loaded, and a bad candidate. The full key or digest chosen is never replaced by a short label; a result stays with
its run; no error shows a patient's or a server's path.

**GB10, on a candidate slot after the review.** A small base, a branch, a full and a frozen disease fine-tune, saved
and continued once. LP and disease measurement of the two children, kept apart. Load a disease model, infer, stop,
run a research job, load again; a concurrent attempt is refused with the shared reason; a bad candidate leaves the
serving model as it was. Gene and patients-like-me through the UI on the small functional data, marked as not ground
truth. No accuracy gain is required.

## 9. Stop conditions

Only these pause the affected part; the rest continues:

- the work would need a change to the training algorithm, continuation semantics, scoring or metrics, data roles,
  privacy use or a clinical rule, beyond what §3 lists;
- a whole-file deletion is needed and has no list the owner approved; or the owner's data, models, reports, ledgers or
  formal deployment would be touched;
- a new architecture needs a different research question, head or data than the chosen fit;
- delivery would need weaker fingerprints, provenance, natural retirement or a promised capability.

## 10. C0: the second architecture, a fit assessment

An assessment, not a support list and not a ranking. Only `gatv2` is declared and built (`spec.py`
`ENCODER_DECLARATIONS`, `builder.py` `_ENCODERS`). `hgt`, `hetero_gat` and `sage` are names in `ENCODER_KINDS` with no
declaration; a spec naming one validates and is refused at a run's start and by the builder. `HeteroGNNLayer`
(`src/models/gnn/layers.py`) is kept and reached only by tests. Which candidate C1 adapts, if any, is the owner's
choice (§3 point 8).

**What is already in place.** Each checkpoint's view record holds the node types and the `src|rel|dst` relation keys
(`ViewShape`). Both computations receive each node's type and each edge's relation, with the sampler's edge
permutation applied, and the protocols supply them.

**What a typed architecture would need at the shared boundary.** The builder passes an encoder only a relation count;
it would pass the `ViewShape`, and the relation keys would be parsed into triples there. The shared `EncoderSpec`
check holds GATv2's norm rule, which would move into each declaration. A declaration has no dtype member; the kernel
and the conv's observed dtype would be recorded apart from the autocast request.

Scale for the estimates: about 52,300 nodes, 494,200 edges and 3 node types; the GB10 has 130.7 GB of unified
memory. The memory figures are estimates, not measurements.

| | HGT (PyG `HGTConv`) | Heterogeneous GAT | SAGE | GraphGPS-style | ULTRA-style |
|---|---|---|---|---|---|
| Research fit | Type-aware attention over node and relation types: a direct contrast to GATv2 with and without relation embedding. An established architecture, not the newest | Relation-specific attention; overlaps relation-aware GATv2, so a narrow question | No attention: an ablation or control | A recent global-attention family | Query-conditioned KG reasoning; no query form for the disease task |
| Heterogeneous needs | Node types and triples from the recorded `ViewShape`; no new data | As HGT | None when homogeneous | The existing type embedding; no new metadata | Relation ids and its own relation graph |
| GB10 compute and memory | About 1 GB per layer in fp32 at hidden 256 over the full view; the kernel choice must be pinned and observed | About R times a GATv2 layer | The cheapest | Dense attention over the graph exceeds the memory without an observed memory-efficient backend | One full conditioned pass per query |
| Output to the existing heads | One vector per node; types with no incoming edge carried through | Yes | Yes | Yes | No: scores per query, not node vectors |
| Sampled and full view | Both equal to the reference on the CPU (sampled seeds within 1.5e-8, chunked full view exact), given a per-node norm | Possible over relation masks | Same loop as GATv2 | Not equivalent: global attention over a batch is another function | Neither applies |
| Licence and local dependencies | MIT; in PyG, installed on both deploy routes | MIT; installed | MIT; installed | MIT; in PyG | MIT; adds packages and a C++/CUDA kernel compiled at import |
| Shared-boundary changes | `ViewShape` to the builder, triples, a declaration, the norm rule per kind, kernel and dtype recorded, `HeteroGNNLayer`'s order protections reused | As HGT | A declaration and an implementation | A new computation declaration and training strategy; positional encodings as a new derived artefact | Representation, head and protocol extensions |

Observed on the CPU (PyG 2.8.1, `pyg-lib` 0.8.0): `HGTConv` (with bfloat16 parameters and inputs), `segment_matmul` and
`grouped_matmul` ran in bfloat16 under both kernel policies, so the old "fp32 only" note's `RuntimeError` does not hold
for this build; under CPU bfloat16 autocast with float32 parameters `HGTConv` ran and returned float32, so its dtype
under autocast is not observed. The GB10 is unobserved, and its two routes carry `pyg-lib` 0.6.0 and 0.8.0. `HGTConv` leaves a type with no incoming edge out of its output.

**Recommendation.** HGT first, if the owner opens C1: it exercises exactly the missing typed-metadata boundary, needs no
new dependency, fits every existing head, and its remaining risks (kernel policy, dtype on the GB10) are bounded and
documented (BACKLOG 20). GraphGPS second, if a recent family is preferred: it tests the computation-strategy boundary
instead. ULTRA is a separate research track at the head and protocol boundary, outside A and B.

## 11. Whole-file cleanup candidates

Listed, not deleted. None yet.

## 12. Record

| Date | Step | Base or head | Note |
|---|---|---|---|
| 2026-11-12 | plan | `30e8ae1` | The owner adopted the reviewer's rev 2 plan and its one-time authorisation; the round's branch created from `main` `30e8ae1`; a read-only survey of A0–A3, B and C0 |
| 2026-11-12 | A0 | `21b247f`..`20c050e` | The blueprint's current state after #133's merge; the skill's "best" is the run's recorded selection metric; other architectures a fit assessment of one (C0) |
| 2026-11-12 | A1a | `17bf1f1`..`6e0d22f` | S1–S6: what a new run takes stated once (`run_plan.new_run_settings`); the rules across settings as data; the fine-tune start's base checks before any child; `POST /training/start` a thin projection; each setting's source recorded |
| 2026-11-12 | A1b | `9293161`..`4c395f5` | B1–B4: a same-settings branch (`parent.changes` `{}`); the runner's one branch preview and start; `POST /api/v1/training/branch`; the lists say what a branch changed; patience branchable; probe `L7.branch`. Full suite at `4c395f5`: 6939 passed, 9 skipped; import contracts kept |
| 2026-11-12 | A2 | `ca19d16`..`b3a3433` | F1–F3: Fine-tune offers the freeze; the job card and the model words name the frozen-backbone task model; invalid fine-tune settings answer 422; probe `L7.disease_frozen` (an fp32 check) |
| 2026-11-12 | A3a | `a012959`..`7656d8c` | I1–I10: CUDA serving under the one GPU lock in one service lifecycle; `POST /api/v1/pipeline/stop`; restart and shutdown stop serving; `/pipeline/config` retired |
| 2026-11-12 | A3b | `77c47ac`..`7ed04e9` | The probe serves under its own lock and stops at each phase's end; probe `W12`; Stop inference on the Diagnosis card; the GPU chip and Restart words. Full suite at `7ed04e9`: 7050 passed, 9 skipped |
| 2026-11-12 | checkpoint A, verification | `7ed04e9` | Four independent verifiers, by real executions on a copy: a parent replaced after its preview is refused at the real start, and a branch runs through the one root (both held); two claims refuted: continuations did not get one judgement on the CLI, the runner and HTTP, and a stop on a real server did not refuse new diagnoses at once (the diagnosis blocked the event loop) |
| 2026-11-12 | AF1 | `dc08792`..`e079c05` | Both fixed at their roots: continuations judged by the shared functions (unread, values, the parent's identity, the device); no inference-lifecycle request runs its work on the event loop (BACKLOG 28), diagnoses serialised by one run lock, the stop's cut-off on arrival. The verifiers' cases re-run on a copy: continuations 22 of 22 and devices 6 of 6 agree; no diagnosis arriving after a stop is admitted. **Checkpoint A** at `e079c05`: 7070 passed, 9 skipped; import contracts kept |
