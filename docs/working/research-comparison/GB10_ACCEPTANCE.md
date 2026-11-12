# The round's GB10 acceptance: the new probes' field contract

**Status: run on the GB10 at `e60f23e`, and passed on both deploy routes (2026-11-11 the prebuilt pull, 2026-11-12 a
local build): 116 of 116 probes and the closing check 182 of 182 on each. The record is
[`../research-comparison-acceptance-2026-11-12/`](../research-comparison-acceptance-2026-11-12/README.md).** Stage 10a
of this round (README §5 Stage 10, the owner's O9) added required probes to `scripts/probe_deployment.py
--m2-acceptance`. This page is the contract the external closing check reads them by: the IDs, the report keys and what
each must hold. It also records what the check's revision had to do (TA-10, §3).
Stage 10b added one informational probe, S1, M1's minimal served readings (§2.1): it is in neither required list, the
probe's exit status never counts it, and the check must never require it or judge its numbers.

- **What was verified, and where.** On the CPU in the round's container: `tests/unit/test_probe_m2_phase.py` runs the
  demo phase end to end and checks the evidence against the disk, and `tests/unit/test_probe_research_comparison.py`
  shows that their own checks fire. On the GB10 at `e60f23e`, with `--device cuda`, once per deploy route: every
  required probe passed, and the values below that say "on the GB10" were observed (the record §1, §2).
- **What was not verified.** What §4 lists as not covered, and the deploy layer's failure paths on the GB10 (a failed
  pull's stop, the build fallback, a wrong wheel, a mismatched receipt), which remain software simulation (the record
  §3).
- **When it ran.** As O9 says: after integration, the external review and RC-R1's recheck (`e17ffe7`..`e60f23e`), with
  the new probes required. Each deploy route was recorded as it was actually verified. Both routes ran, one after the
  other, in the reused candidate slot, each with a new environment, data copy and evidence directory: the owner's A/B
  slot decision, made before the run, which replaces the new clone of O9 and §4 items 1–2 (the record §1). The
  reviewer's review of the record passed (the record §5).

## 1. Where the closing check reads them

The report keeps its shape (`REPORT_SCHEMA_VERSION` 1). Each probe is an entry of `probes`, with the keys `id`, `phase`,
`claim`, `status`, `detail`, `facts` and `seconds`. A probe counts only with `status` `"passed"`. `detail` is the
probe's own words and is never parsed.

**The required lists**

- `m2_acceptance.demo.required`: the demo part's list (`M2_DEMO_REQUIRED`). It now has 66 entries: the 52 it had at
  checkpoint C, unchanged and in the same order, followed by the 14 new ones:
  `A6`, `L1`, `L2`, `L3`, `L4.other_graph`, `L4.tampered_pairs`, `L4.other_view`, `L4.split_record`, `L5`, `L6`,
  `L7.disease`, `L7.gene`, `L7.patients_like_me` and `L8`.
- `m2_acceptance.real_data.required`: the real-data part's list (`M2_REAL_REQUIRED`). Unchanged: 15 entries.
- `m2_acceptance.demo.complete` and `m2_acceptance.complete` are false until every new probe has passed.
- S1 (§2.1) is in neither list. `m2_acceptance` never reads it, so no reading of it, however slow or large, and not even
  a failed S1, leaves `m2_acceptance` incomplete.
- **The probe's exit status never counts S1 either** (`INFORMATIONAL_PROBES`, `run_exit_status`). The exit is 1 when any
  other probe failed or raised, or the acceptance is not complete, and 0 otherwise. A failed or erroring S1 is printed as
  an INFO line and leaves the exit at 0, so hardware variance in its start (a start slower than the probe's 300 s, say)
  never fails the run the check reads. The report's `summary` still counts every probe, S1 included, by its status.

**Unchanged.** No earlier ID was renamed, dropped or moved. Every key the current check reads keeps its name and
meaning:

- `tested_commit`, `allocator`, `environment` and `settings`;
- A5's facts, and `m2_acceptance.environment`;
- the facts of R1–R4, R8, M2 and M9, `R8`'s `history_file` among them.

**One value changed.** R3's `training_rules` keeps its name and meaning: the training rules the base records, which R3
requires to be the tested build's `TRAINING_RULES_VERSION` (`src/utils/checkpoint_io.py`). That constant is now 2
(Stage 9a), so a run of this round records 2. K8's tested commit `f2c2012` had 1, and its report records 1. The check
must not pin either number (§3, item 3).

`m2_acceptance.scope` gained a sentence about the new probes, and one about S1. It is prose, and nothing here needs the
check to read it.

## 2. The probes and their facts

Every value is plain JSON. Digests are SHA-256 in 64 lowercase hex digits. No fact carries a path, a host name, a case
id or a rank. A run id is a run directory's name, such as `20261110T093522Z-f5acd61944ab-d14ed320`. The demo flow
runs on the probe's own deployment under `<work>/rc`:

- a copy of the demo workspace, `probe-rc`;
- its runs: `probe-rc-base`, a copy of D1's base; `probe-rc-d1-disease`, a copy of D1's disease child; and L7's
  `probe-rc-disease`, `probe-rc-gene` and `probe-rc-patients-like-me`;
- one registry, with the versions `probe-kg-test-1`, `probe-kg-other-graph-1`, `probe-kg-tampered-pairs-1`,
  `probe-kg-split-record-1`, `probe-rc-supplied-1` and `probe-gen-val-1`.

**What the check must not require.** No metric value, no threshold and no comparison between two models' figures: they
are functional checks, not results (README §1, "Not pass conditions"). The probes never compare a version or a route
with an expected one, or one run's with another's. L8 checks only that the export shows each run's native identity and
every component fact as that run's own report recorded them. On CUDA, two runs of one model need not give equal bytes or
numbers; that they are both kept is what counts.

**Where each run ran.** Every kept run's report is checked when the probe reads it (`_ran_on`, in L2, L3, L5, L6 and
L7). Its `cuda_executed` must be true exactly when the probe's device is CUDA. The measurement command records true
only for a run that executed there, and it refuses `--device cuda` without CUDA.

### A6 — each component's native identity and route (phase `environment`)

A6 requires A1's components. They are the facts every run's environment records
(`src/utils/run_environment.collect_components`): it stops only when A1 did not pass, recorded no component, or
recorded one outside that shape. It is never a gate on versions.

| Key | Holds |
|---|---|
| `components` | the number of named components, at least 1 (today PyG's four native extensions) |
| `native_identity` | by component name: `kind`, `version` (text or null), `native` (`state`, `count`, `aggregate_sha256`) and `source` (`state`, and `route` only when `state` is `recorded`); the projection the record views show |
| `routes` | by component name: the deploy route, where the source receipt matched what is installed, else `"not recorded"` |
| `line` | the same identity in the one wording the console and Models use |

What a GB10 run must show depends on the route actually deployed, and the check states it rather than assuming one.
`routes[name]` is the route the source receipt names, and only three values occur. They are the receipt's routes
(`src/utils/pyg_native_check.py`, `ROUTE_PREBUILT_RELEASE` and `ROUTE_LOCAL_BUILD`), never `deploy.sh`'s strategy
names `pull` and `compile`:

- `"prebuilt release"`: a verified pull. The release's wheels were installed, checked and their receipt written.
- `"local build"`: a compile. It is also what a pull records when it failed and fell back to a source build
  (`PYG_ARM_ON_PULL_FAILURE=build`). The check must not count that as a verified pull.
- `"not recorded"`: anything else. There is no receipt, or it does not match what is installed.

Either way the probe passes. That is why each route is recorded as verified only by its own run (O9).

### L1 — a KG edge set imported from D1's base

| Key | Holds |
|---|---|
| `preview_wrote_nothing` | `true` |
| `preview_id` | 16 hex digits |
| `kind`, `role` | `"kg_edges"`, `"test"` |
| `manifest_sha256` | the version's `cohort.json` digest (L2's `version.manifest_sha256`) |
| `graph_is_the_workspaces` | `true` |
| `graph_view_digest`, `split_digest` | the base's recorded view and split |
| `taken_from_checkpoint_sha256` | the listed base's digest (L2's `checkpoint_sha256`, L7's `child.parent_sha256`) |
| `pairs` | `sha256`, `lp_pairs_digest`, `rows` (more than 0) |
| `counts` | `positive_edges`, `pairs` (equal to `pairs.rows`), `edges_without_negative`, `message_passing_edges` |

### L2 and L3 — the KG link-prediction protocol, kept and registered

L2 runs `measure_scorer --record`. L3 runs the job runner from the server's lists, asked to record. Both measure the
listed base on L1's set, so they have the same keys:

| Key | Holds |
|---|---|
| `measured` | `run` `"probe-rc-base"`, `protocol` `"kg_link_prediction"` |
| `run_id`, `report_sha256` | the kept run and its report's bytes |
| `checkpoint_sha256` | the base's digest |
| `version` | `label` `"probe-kg-test-1"`, `kind` `"kg_edges"`, `role` `"test"`, `manifest_sha256` (L1's) |
| `exposure` | `definition_version` `"kg-lp-exposure/v1"`, `state` `"held_out"`, `reading` `"held out relative to the recorded training and validation graph use"`, `measured_edges` (L1's `pairs.rows`) |
| `n_ranked` | the pairs scored (L1's `pairs.rows`) |
| `metrics` | `lp_accuracy`, `lp_roc_auc`, `lp_average_precision`, `lp_loss`, raw as recorded |
| `scores` | `rows`, `scores_sha256`, `pairs_sha256` (L1's `pairs.sha256`), `lp_pairs_digest`, `pairs_not_finite`, `metrics_recomputed_from_the_scores`, `metrics_defined`, `raw_scores_recomputed`, `largest_raw_score_difference` |
| `cuda_executed` | `true` on the GB10, `false` on the CPU: required to match the device |
| `record` | `sequence` (L2 `1`, L3 `2`, in the one ledger beside the base), `run_id` (equal to `run_id`), `mode` `"kg_link_prediction"`, `view` `"KgEdgesRecordView"`, `ledger_sha256` |

L3 adds:

- `inputs`, prose;
- `status` `"completed"`, `exit_code` `0`;
- `outcome_file_names_the_kept_report`, `kept` and `registered`, each `true`;
- `report_state` (`kept` `"kept"`, `registered` `"registered"`).

**The scores check** (`_lp_scores_verified`). The scores file, the version's pairs and the report's summary verify each
other: the bytes, the rows and their order, each row's state, the raw scores recomputed through the LP core over the
version's message-passing graph, and the metrics recomputed from the scores file alone by the LP core's one summary
rule (`src/models/bundle/lp_head.lp_set_metrics`, which the measurement itself uses). On the CPU the recomputed raw
scores are exact, so `largest_raw_score_difference` is `0.0`. On CUDA, message passing sums in an order CUDA does not
promise. The probe therefore requires the recomputation within `RC_SCORE_TOLERANCE` (rtol 1e-4, atol 1e-5) and
records the largest difference. The check reads that difference and does not require 0.

### L4 — the refusals before anything is written

Each case measures with `--record` and must be refused. The registry, the flow's workspace and the checkpoint's own
directory are unchanged, and no run directory is made. Every case has `nothing_written` `true` and a fixed-words
`refused_as`.

| ID | What is refused | `checkpoint_reads` | `models_built` | More |
|---|---|---|---|---|
| `L4.other_graph` | a KG edge set of another graph: taken from a base trained on M1's same-shaped graph | 0 | 0 | `differing_roles` contains `"kg"` |
| `L4.tampered_pairs` | a KG edge set whose `pairs.bin` changed after its import | 0 | 0 | |
| `L4.other_view` | a model of another graph view (a base trained without the ortholog relation), refused as a stated capability gap | 1 | 0 | |
| `L4.split_record` | a KG edge set whose split record's seed and digest contradict each other on its graph's view | 1 | 1 | |

### L5 — a disease child measured by both of its protocols

The subject is `probe-rc-d1-disease`, a copy of D1's disease child.

| Key | Holds |
|---|---|
| `measured` | `run` `"probe-rc-d1-disease"`, `checkpoint_sha256` |
| `disease` | `run_id`, `report_sha256`, `protocol` `"disease"`, `n_ranked`, `metrics`, `cuda_executed`, `version` (`label` `"probe-rc-supplied-1"`, `kind` `"supplied"`, `manifest_sha256`), `record` (`sequence` `1`, `view` `"RecordView"`) |
| `kg_link_prediction` | L2's keys except `measured`, for the child: `checkpoint_sha256` the child's, `version` L1's, `exposure.state` `"after_full_view_exposure"` (the reading `"after full-view exposure"`) and `record` (`sequence` `2`, `view` `"KgEdgesRecordView"`) |
| `records` | the two records the one reader (`read_records`) reads beside the child, `[{mode, view, sequence}]`, sorted by mode: `disease` then `kg_link_prediction` |

### L6 — a generated split imported, measured and registered

The workspace's functional gene fixture is derived first (`scripts/derive_task_fixture.py --distractors 4`, the
demo's `DEMO_FIXTURE_DISTRACTORS`). The version keeps it.

| Key | Holds |
|---|---|
| `preview_wrote_nothing`, `preview_id` | as L1 |
| `version` | `label` `"probe-gen-val-1"`, `kind` `"generated"`, `role` `"val"`, `manifest_sha256`, `counts` (`samples`, `diseases`, `task_fixture_samples`) |
| `measured` | `run` `"probe-rc-d1-disease"`, `checkpoint_sha256` (L5's) |
| `run_id`, `report_sha256`, `protocol` `"disease"`, `n_ranked`, `metrics`, `cuda_executed` | the kept run |
| `record` | `sequence` `3` (after L5's two), `view` `"GeneratedRecordView"` |

### L7.disease, L7.gene, L7.patients_like_me — the GPU regression cases

Each is a short child of the listed base: one epoch at the demo's sizes (`scripts/setup_demo.demo_task_training`),
trained on the probe's device. Each is measured by its own protocol and registered: the disease child on
`probe-rc-supplied-1`, and the gene and patients-like-me children on `probe-gen-val-1` through its fixture. These are
regression cases, not studies.

| Key | Holds |
|---|---|
| `task`, `protocol` | the task, and the same protocol |
| `device` | the probe's device: `"cuda"` on the GB10 |
| `trained_on_cuda` | `true` on the GB10, `false` on the CPU: required to match the device (the child's environment names its GPU) |
| `cuda_executed` | `true` on the GB10, `false` on the CPU: required to match the device |
| `training_seconds` | a number: this child's training, in seconds |
| `epochs`, `sizes` | `1`, and the words naming the demo's sizes |
| `child` | `run`, `sha256`, `parent_sha256` (L1's `taken_from_checkpoint_sha256`), `training_rules` as the child records it (the tested build's `TRAINING_RULES_VERSION`: `2` in this round) |
| `version` | `label`, `kind` (`"supplied"` for disease, `"generated"` for the others), `manifest_sha256` |
| `run_id`, `report_sha256`, `n_ranked`, `metrics` | the kept run |
| `record` | `sequence` `1`, `mode` the task, `view` `"RecordView"` (disease) or `"GeneratedRecordView"` |

### L8 — the results export, uncapped

`scripts/export_results.py --format json` runs over the flow's workspaces root and registry, as the sources
`probe-rc-workspaces` and `probe-rc-registry`.

| Key | Holds |
|---|---|
| `format` | `"shepherd-results-export/1"` |
| `export_sha256` | the digest of the export written (`<work>/rc/export.json`) |
| `results` | `8`: every run L2–L7 kept, and no other |
| `rows`, `unreadable` | the long form's row count; `0` |
| `ui_compare_cap`, `capped` | `5` (Models › Results' side-by-side cap, `COMPARE_CAP`, which an export never applies), `false` |
| `runs` | one entry per run, in this order: L2, L3, L5 (disease), L5 (KG link prediction), L6, L7.disease, L7.gene, L7.patients_like_me. Each entry has `probe`, `run_id` (that probe's), `report_sha256`, `kind`, `role` and `metric_rows` |
| `carries_no_server_path` | `true` |

Before passing, the probe checks every exported row:

- its report, checkpoint and version digests, data kind and role;
- kept and registered;
- the recorded training settings, rules, learning and effective settings, and loss identity, with a finetune's weight
  transfer listing `parts`, `frozen_forward` and `downstream_initial_sha256`;
- the environment class, and the native identity and every component fact (`components`: each state, the fallback
  reason, the version, the native identity and the source's facts) that run's own report recorded, as the record views
  show them (never the probe's own process: L3's run was measured in the job runner's);
- units for a KG edge set's metrics, and the absence said for the others;
- the exposure: a KG edge set's state and reading, a supplied cohort's R10, a generated split's allocation note.

## 2.1 S1 — M1's minimal served readings, informational (phase `m2`, last)

Stage 10b (the owner's item D; BACKLOG 64). S1 runs after L8. It serves D1's disease child on the demo workspace from a
service of its own, started as the launcher starts it, with the child configured at startup, and reads what that
service does. It takes a minimal part of PLAN_B04 §7.3, the reviewed procedure of the retired
`scripts/measure_served_pipeline.py`, which stays that script's record. Its readings are recorded and never judged:
there is no threshold, and no accuracy figure stands in for them.

- **When it fails.** Only when no reading of the child could be taken, and never on a reading's size: the service did
  not answer `/health` (it exited first, or did not answer within 300 s), `/ready` did not answer 200, or the status's
  served identity is not the one the probe read (`_assert_served_identity`). Then no request is sent. Its `facts` keep
  what was read before (`readiness`, and `cold_start` once `/health` answered) and give every later reading as
  `state` `"not_taken"` with its `reason`. A failed S1 never sets the exit status (§1). A request that fails afterwards
  is counted by class and leaves its phase incomplete; it does not fail S1.
- **Whose readings they are.** Every reading is the server process's: its answers, or its PID's `/proc` entries. None
  is taken in the probe's own process, which runs no part of a request.

| Key | Holds |
|---|---|
| `required` | `false` |
| `version` | `"served-readings/1"`, the definitions' version |
| `subject` | `task` `"disease"`, `checkpoint_sha256` (D1's child), and `served_identity` as the status reported it at readiness; `state` `"not_taken"` with its `reason` when the service was not shown ready |
| `device` | the probe's device: `"cuda"` on the GB10 |
| `query` | where the one query came from: `split` `"val"`, `case_position`, `phenotypes` (a count), `cases`, `cases_bound`. No identifier |
| `request` | `fields_sent` `["phenotypes"]`, and `api_defaults`, the request model's own defaults (`top_k`, `include_explanations`, `include_paths`) |
| `readiness` | `state` `"ready"` with `ready_status` `200`; or `"not_ready"` with `failed_at` (`"health"`, `"ready"` or `"served_identity"`), its `reason` (the probe's words, as in `detail`), and `ready_status` once `/ready` answered |
| `cold_start` | `state` `"read"` with `seconds`: launch to the first `/health` answer, the startup's pipeline build included, polled every 0.5 s and rounded to 0.1 s (`wait_ready`; request times keep microseconds). `memory_at_ready`: a memory reading (below), `"not_taken"` when the service was not shown ready. When `/health` never answered, `state` `"not_obtainable"` with its `reason` |
| `warm_up`, `steady_state` | one phase each: 5 and then 50 serial requests (`SERVED_WARM_UP_REQUESTS`, `SERVED_STEADY_REQUESTS`). Each has `state` `"read"`, `planned`, `sent`, `counted`, `failures` (by class), `ended` (`null`, `"no_response"` or `"server_exited"`), `complete`, `seconds` and `server_handler_seconds` (one per counted request, in order), `summary` and `server_handler_summary` (`median_seconds`, `min_seconds`, `max_seconds`), and `memory_after`. A steady state the warm-up ended is `state` `"not_taken"` with its `reason`. When the service was not shown ready, each is `state` `"not_taken"` with its `reason`, `planned` and `complete` `false` |
| `allocator` | `state` `"read"` with `PYTORCH_ALLOC_CONF`, `PYTORCH_CUDA_ALLOC_CONF` and `SHEPHERD_ALLOC_SOURCE` as the server process was started with them, raw, `null` when absent; or `state` `"not_obtainable"` with its `reason` when that environment cannot be read or the process has exited. Read while the server runs, whether it was ready or not |
| `cuda_allocator_peak` | on the GB10, `state` `"not_obtainable"` with its `reason`: torch keeps it inside the server's process and no route exposes it. On the CPU, `"not_applicable"` |
| `complete` | `true` when both phases sent and counted every planned request; `false` otherwise, and when the service was not shown ready |
| `definitions`, `not_covered` | the readings' definitions and what they do not cover, as the probe states them |

**A memory reading** is `state` `"read"` with `rss_bytes` (VmRSS) and `peak_rss_since_start_bytes` (VmHWM). VmHWM is
never reset here, so it is the peak since the process started, never one phase's. Otherwise it is
`state` `"not_obtainable"` with its `reason`. Every reading that has no value is a `state` and a `reason`, never a
number.

**A request's time** is the probe's clock from sending the request to reading its whole answer. The service answers
after the disease head's scores are copied to the host, so on CUDA the time includes the device's work, synchronised in
the server's process. No synchronise call is made in the probe's process. `server_handler_seconds` is each answer's
own `inference_time_ms`, in seconds.

**Not covered** (`not_covered`):

- a peak per phase (no VmHWM reset);
- system-wide memory, swap, a baseline and OOM kills;
- a reload, with the old and new pipelines held together;
- concurrency, a varied or largest workload, and repeats over several cold starts;
- the real workspace's children, and gene and patients-like-me children;
- the server's CUDA allocator counters.

## 3. TA-10: the external closing check's next revision

The check lives outside this repository, in the owner's session. Its next revision must do three things, and be
dry-run first (item 4).

*Done before the run (2026-11-11):* the reviewer revised the check, and its dry run on K8's report passed 114 of 114,
with R3 expecting `f2c2012`'s rules (1), RW2 reading 1,191.9 s, no W8 line and the new probes reported absent (the
record §4). As run, it reads more strictly than items 1 and 3 say. It takes the required lists and R3's rules from
digest-checked copies of the tested commit, and requires the report's lists and tested commit to equal them. It also
requires A6's routes to equal the route under test, §4 item 2's rule for a verified pull; item 1's "do not gate on a
route" is read as: do not pin which route a run must use.

1. **Read the new probes.**
   - Take the required lists from the report (§1), not from a copy in the script. Require the 14 new IDs to have
     passed.
   - Read their facts by the keys above. These cross-checks hold by construction:
     - L1's `manifest_sha256` equals L2's and L3's `version.manifest_sha256`, and L5's
       `kg_link_prediction.version.manifest_sha256`;
     - L1's `taken_from_checkpoint_sha256` equals L2's and L3's `checkpoint_sha256` and every L7 `child.parent_sha256`;
     - L8's `runs` run ids equal those of L2, L3, L5 (both), L6 and L7.
   - Do not gate on a version, a route, a metric value or a raw-score difference (§2).
   - Do not require S1, and do not judge its numbers (§2.1). The probe's exit status never counts it (§1); where the
     check counts failed probes itself, from `summary` or from `probes`, leave S1 out too. The check may print its
     readings, or its failure, as INFO lines.
2. **Fix the two INFO lines that read nothing.** K8's record names them (`../k8-acceptance-2026-11-09/README.md` §3).
   - `W8 training_seconds None`: W8 trains nothing, since it measures W1's listed copy of M2's child. Drop the line.
     The demo's own training time is D1's `facts.training_seconds`, a mapping by stage (K8's report gives
     `{"base": 1.0, "disease": 0.6}`). Each new child's is L7's `facts.training_seconds`, a number.
   - `RW2 training_seconds None`: RW2 records its child's time under `facts.measured.training_seconds`, a mapping by
     stage. K8's report gives `{"disease": 1191.9}`.
3. **Read R3's expected training rules from the tested commit.** K8's check requires R3 to be a base under training
   rules 1 (`R3: a GATv2 base under training rules 1 (Q19 (a))`). The expected value is the tested commit's
   `TRAINING_RULES_VERSION`, read from `src/utils/checkpoint_io.py` at the report's `tested_commit.commit`, as the
   required lists are the tested commit's. It is never a pinned number: 1 at `f2c2012`, and 2 from Stage 9a on. A
   pinned 1 would fail R3 on this round's GB10 run, and a dry run on K8's report could not show it.
4. **Dry-run first on the kept `../k8-acceptance-2026-11-09/k8_report.json`, before the GB10 run.** That report
   predates the new probes: its demo list has 52 entries and holds no A6 or L probe. The dry run passes when:
   - every check that passed at K8 passes again;
   - the R3 line expects `f2c2012`'s rules, 1, read from that commit, and passes;
   - the new checks report the new probes as absent, and nothing else;
   - the RW2 line reads 1191.9 s, and no W8 training line is printed.

## 4. The GB10 run

**The code to test** is the head delivered for the round's external review, named in that delivery; as run,
`e60f23e`, the head of RC-R1's recheck (`e17ffe7`..`e60f23e`). Only documents may follow it before the run, as at
K8; `src`, `scripts`, `tests`, `configs`, `pyproject.toml` and `uv.lock` are the ones tested. The run starts only after
the external review, as O9 says.

1. **A fresh clone at that head**, checked out detached in a directory of its own. The deployment checkout is not
   touched, and neither is the real workspace: the probe gets a new copy of it. *As run, by the owner's decision
   before the run:* the candidate slot is reused for each route in turn, recycled and checked out detached at the
   head, in place of a fresh clone here and another clone in item 2 (the record §1).
2. **Each deploy route in its own environment, with its own run and record** (O5 (ii), O9). One route's pass never
   stands for the other's.
   - **The prebuilt release:** `PYG_ARM_STRATEGY=pull ./deploy.sh`. A failed pull stops with its exit (11–15) and its
     words, and nothing is built. That record says the route did not complete, and why. The run is a verified pull
     only when A6's `routes` read `"prebuilt release"` for each PyG component (A6 passes either way).
   - **The local build:** `PYG_ARM_STRATEGY=compile ./deploy.sh`, in another clone and environment. Its A6 `routes` read
     `"local build"`.
   - A pull that fails and is then built (`PYG_ARM_ON_PULL_FAILURE=build`) is a local build, recorded as one, never
     as a verified pull.
   - After either deploy, `scripts/validate_pyg_ext.py --receipt` shows whether the receipt matches what is installed.
3. **The probe**, as at K8 (§1 of its record), while no managed GPU job runs:
   `scripts/probe_deployment.py --m2-acceptance --device cuda --work-dir <a new directory outside the clone>
   --real-workspace <the new copy> --real-reference-checkpoint <its model-02 file> --workspaces-root <the copy's root>`.
4. **The closing check** in its TA-10 revision (§3), dry-run first on K8's kept report.
5. **A run passes** when:
   - the probe exits 0;
   - `m2_acceptance.complete` is true, with the tested commit's lists: demo 66 and real data 15, every probe passed;
   - the closing check passes.
   S1's readings and any failure of S1 are printed as INFO and never judged (§2.1).
6. **The record**, per route: a new evidence directory (`docs/working/<round>-acceptance-<date>/README.md`). It states
   the environment by the product's own facts (A1, A5, A6), the route as A6 read it, the inputs' digests before and
   after, the duration and the disk, and what the run does not cover. As run: one directory,
   [`../research-comparison-acceptance-2026-11-12/`](../research-comparison-acceptance-2026-11-12/README.md), with a
   column per route in its §1 and each route's evidence in `pull/` and `compile/`.

**What the run does not cover, already known:** the frozen-backbone control on CUDA, with AMP, its scaler or
`torch.compile` (its tests run on the CPU only, and no probe trains one); the WebUI in a browser on the GB10 (the
round's browser check ran in a CPU container); the server's CUDA allocator counters (S1 says so); HGT, SAGE or hetero
GAT; bfloat16; model quality.

**Time and disk.** The new probes add only demo-sized work: five short trainings (the two refusal bases and L7's
three children), about a dozen measurements, one job-runner test run and one export. On the CPU here the whole flow
(L1–L8) took about 7 s. S1 adds one service start and 55 demo requests, about 8 s on the CPU here. On the GB10, A6
and L1–L8 recorded 9.7 s (pull) and 8.9 s (compile) of probe time, and S1 8.9 s and 8.7 s; their share of the work
directory's disk is not measured apart (the whole probe and its disk are in the record §1).
