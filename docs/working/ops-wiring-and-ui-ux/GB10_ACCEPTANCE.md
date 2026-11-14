# The round's GB10 acceptance: the new probes' field contract

**Status: written against the product code at `500a8d6` (2026-11-14); not run.** It runs on a candidate slot after the
external review, by the owner's instruction (the reviewer's plan, §八 item 6; [`README.md`](README.md) §3 point 7, §8).
Nothing here has run on the GB10, and nothing here says it did. This page is the contract the external closing check
reads this round's new required probes by: where they sit in the report, their IDs, the keys each records and what each
must hold, what the check's next revision has to change, and how the README's GB10 acceptance (§8) maps onto the probes
and onto the UI steps the owner runs by hand. It follows the research-comparison round's contract
([`../research-comparison/GB10_ACCEPTANCE.md`](../research-comparison/GB10_ACCEPTANCE.md)), whose run at `e60f23e` is
the base the check was last revised on.

- **What was verified, and where.** On the CPU in the round's container only: `tests/unit/test_probe_m2_phase.py`
  runs the demo phase end to end and checks every new probe's facts against the disk
  (`test_the_demo_phase_passes_and_every_required_probe_runs`, with `_check_the_research_comparison_flow` and
  `_check_the_inference_lifecycle`), and pins the required lists (§1).
  `tests/unit/test_probe_research_comparison.py` shows that the frozen arm's own checks fire, and
  `tests/unit/test_probe_m5_acceptance.py` that W12's judges fire (§2). At `500a8d6` the demo-phase test and the
  four tests of the required lists passed here (the demo phase about 90 s).
- **What was not verified.** Everything on CUDA: each value below marked "on the GB10" is what the code requires
  there, never an observation. §4 lists what no probe covers.
- **When it runs.** After the external review, on the head delivered for it, by the owner's instruction (README §3
  point 7; §4 here).

## 1. Where the closing check reads them

The report keeps its shape (`REPORT_SCHEMA_VERSION` 1). Each probe is an entry of `probes`, with the keys `id`,
`phase`, `claim`, `status`, `detail`, `facts` and `seconds`. A probe counts only with `status` `"passed"`. `detail` is
the probe's own words and is never parsed. The probe's command line is unchanged: no option was added, renamed or
dropped.

**The required lists**

- `m2_acceptance.demo.required`: the demo part's list (`M2_DEMO_REQUIRED`). At `main` `30e8ae1` it had 66 entries;
  it now has 69. The 66 keep their names and their order. The three new IDs are placed where they run:
  `L7.disease_frozen` and `L7.branch` after `L7.patients_like_me` and before `L8`, and `W12` last:

  `… L7.disease, L7.gene, L7.patients_like_me, L7.disease_frozen, L7.branch, L8, W12`

  So `L8` is now the 68th entry rather than the 66th. A check that reads the list from the report, or from the tested
  commit, by name is unaffected; one that compared positions would not be.
- `m2_acceptance.real_data.required`: the real-data part's list (`M2_REAL_REQUIRED`). Unchanged: 15 entries.
- `m2_acceptance.demo.complete` and `m2_acceptance.complete` are false until each of the three has passed.
- S1 stays informational (`INFORMATIONAL_PROBES`, `run_exit_status`, both unchanged): in neither list, and never
  counted by the exit status.

**Unchanged.** No earlier ID was renamed, dropped or reordered. Every key the current check reads keeps its name and
meaning: `tested_commit`, `allocator`, `environment` and `settings`; A5's facts and `m2_acceptance.environment`; the
facts of R1–R4, R8, M2 and M9; A6's and L1–L7's. R3's `training_rules` is still the tested build's
`TRAINING_RULES_VERSION`, 2 at `30e8ae1` and at `500a8d6`; the check reads it from the tested commit, as before.

**Values that changed in earlier probes.** None of these is a key the check is known to pin, except L8's, which its
`L8:` lines read (the record of `e60f23e`, `pull/acceptance-check.txt`):

| Probe | What changed | Why |
|---|---|---|
| L8 | `results` is 9, not 8; `runs` gains a ninth entry, `L7.disease_frozen`, after `L7.patients_like_me`. The rule is the same: every run the flow kept and registered, and no other. L7.branch measures nothing, so it adds no run | The frozen child is measured and registered (decision F2) |
| W2 | `gated.api_routes` lists `POST /api/v1/pipeline/stop` and `POST /api/v1/training/branch`, and no longer `POST /api/v1/pipeline/config`; `gated.gradio_events` has 4 entries, `diagnosis_panel._on_stop_inference` the new one; the claim says "four gated dashboard events". W5's `loopback_request.routes` reads the same route list | The stop and the branch routes are operator controls (I1, B2); the config route retired (I10) |
| E4 | One new key, `live_holds_gpu_lock`: `true` on the GB10, `false` on the CPU. E4 now serves D1's child through the lifecycle (the reload route's handler, bound to its digest) before it reloads by task | A CUDA model is served only under the GPU lock (I2) |
| S1 | No key changed. Its service now has a workspaces root of its own, so its CUDA start-up load holds a GPU lock of its own | I2: CUDA serving is refused without a lock |

Phase E now runs under the probe's own GPU lock, as phases M and R already did, and each of the three stops what it
served in process at its end (E4, T5, RT3), so the next phase's in-process measurements find the lock free.
`m2_acceptance.scope` gained sentences about the three new probes; it is prose, and nothing here needs the check to read
it.

## 2. The probes and their facts

Every value is plain JSON. Digests are SHA-256 in 64 lowercase hex digits. No fact carries a path, a host name, a case
id or a rank. All three have phase `"m2"`.

**What the check must not require.** No metric value, no threshold, no training or start-up time, and no comparison
between the full and the frozen arm's figures: they are functional checks (README §1, "Not pass conditions"). W12's
counts of diagnoses vary with scheduling; only the invariants its judge enforces (below) hold. The memory reading after
W12's stop is recorded and never judged (decision I4).

### L7.disease_frozen — the frozen-backbone control, fp32, paused and resumed

Claim: "ops wiring: an fp32 frozen-backbone disease child of D1's base on the probe's device, paused and resumed: its
graph side the base's, its downstream parts trained and alone in the optimizer, its record kept across the resume, then
measured and registered".

In the research comparison's flow under `<work>/rc`, after L7.patients_like_me: a demo-sized disease child of the
listed base (`probe-rc-base`) with the graph side frozen, mixed precision off and no `torch.compile`, trained on the
probe's device by the production trainer for 2 epochs, paused after the first, and resumed in a run of its own
(decision F2). The resumed child is measured by the disease protocol on `probe-rc-supplied-1` and registered.

| Key | Holds |
|---|---|
| `task`, `device` | `"disease"`; the probe's device: `"cuda"` on the GB10 |
| `training_seconds` | a number: both runs' training, in seconds |
| `epochs`, `paused_after` | `2`, `1` |
| `sizes` | the words naming the demo's sizes |
| `settings` | `{"frozen_parts": "graph_side", "use_amp": false, "compile": false}` |
| `runs` | `{"paused": "probe-rc-disease-frozen", "resumed": "probe-rc-disease-frozen-resumed"}` |
| `parent_sha256` | the listed base's digest |
| `training_rules` | as the resumed child records it: the tested build's `TRAINING_RULES_VERSION` (2 at `500a8d6`) |
| `version` | `label` `"probe-rc-supplied-1"`, `kind` `"supplied"`, `manifest_sha256` |
| `precision` | text beginning "fp32 only": mixed precision off and `torch.compile` not requested in both files, and AMP and compile with a freeze not verified |
| `frozen_parts` | `"graph_side"` |
| `parts` | `{"input": "frozen", "encoder": "frozen", "lp_head": "frozen", "patient_encoder": "trainable", "task_head": "trainable"}` |
| `frozen_forward` | `"eval"` |
| `terms_without_trainable_parameters` | `["lp"]` |
| `graph_side` | `base_sha256` (the base's graph-side parts digest), `paused_equal` and `resumed_equal` (`true`: every graph-side parameter and buffer equal to the base's, key by key, and by digest), `entries`, `batch_norm_buffers` (at least 1) |
| `downstream` | `initial_sha256` (the start the record names), `paused_sha256` and `resumed_sha256`: the three differ (each run trained) |
| `optimizer_holds_only_downstream_parameters` | `true`: the optimizer holds exactly the downstream parameters, each once, each with a saved state, their shapes counted |
| `paused` | `sha256`, `completed_epochs` `1`, `parent_sha256` (the base's) |
| `resumed` | `sha256`, `operation` `"resume"`, `completed_epochs` `2`, `parent_sha256` (the paused file's), `weight_transfer_kept` `true` |
| `same_downstream_start_as_the_full_arm` | `true` when L7.disease passed (the full arm of the same base starts from the same downstream initialisation); `"not compared"` otherwise |
| `trained_on_cuda` | `true` on the GB10, `false` on the CPU: required to match the device (both files' environments) |
| `run_id`, `report_sha256`, `protocol` (`"disease"`), `n_ranked`, `metrics` | the kept run of the resumed child |
| `cuda_executed` | `true` on the GB10, `false` on the CPU: required to match the device |
| `record` | `sequence` `1`, `run_id`, `mode` `"disease"`, `view` `"RecordView"`, `ledger_sha256`; the paused file has no ledger |

The graph-side equality is exact (`torch.equal`) on every device: the frozen parts are never updated.

### L7.branch — a same-settings branch through the job runner

Claim: "ops wiring: a same-settings branch of D1's base through the job runner, on the probe's device, listed under its
parent with no change, its parent unchanged, and loaded".

In the same flow, after L7.disease_frozen: the job runner's one start (`start_training` with `branch_from`, run name
`probe-rc-branch`), bound to the listed base's digest and the dataset's, runs a branch of the listed base that changes
no setting on the probe's device to its end (decisions B1, B2).

| Key | Holds |
|---|---|
| `device` | the probe's device: `"cuda"` on the GB10 |
| `trained_on_cuda` | `true` on the GB10, `false` on the CPU: required to match the device |
| `training_seconds` | a number: start to the job's end, in seconds |
| `started_through` | prose: the runner's start, bound to the parent's digest and the dataset's |
| `branch` | `run` (a new run directory's name, holding `probe-rc-branch`), `sha256`, `operation` `"branch"`, `changes` `{}`, `optimizer` `"new"`, `completed_epochs` (the base's epochs, inherited), `training_rules` (the tested build's) |
| `parent` | `run` `"probe-rc-base"`, `sha256` (the listed base's), `unchanged` `true` |
| `listed_under_its_parent`, `listed_changes` | `true`, `{}`: the server's family view lists the branch under its parent's exact file |
| `loaded` | `true`: the one loader read it as a base of its parent's family, its stage run from epoch 0 |

The job ended `completed` with exit 0. The run is not measured, so it is not in L8.

### W12 — the inference lifecycle across processes on the device

Claim: "Ops round: the inference lifecycle across processes on the probe's device: no start-up model is not loaded;
CUDA serving holds the GPU lock and research jobs are refused naming it, CPU serving holds nothing and research jobs
run beside it; a bad candidate keeps the model; a stop finishes accepted diagnoses, refuses new ones and only then
reports released; research runs after it; device switches move the lock".

After L8 and before S1, on a service of its own under `<work>/w12`, started as the launcher starts it: mode M1, a
workspaces root of its own (so a GPU lock of its own), the probe's device and no start-up model, over a copy of the demo
workspace holding a copy of D1's disease child. Every step goes through the service's routes or `train_model.py` with
the service's root (decisions I1–I9). The expectations follow the device:

| Key | On the GB10 (CUDA) | On the CPU |
|---|---|---|
| `device` | `"cuda"` | `"cpu"` |
| `checkpoint_sha256` | D1's disease child's digest | the same |
| `service` | `started_as` (prose), `seconds_to_health` (a number, 0.1 s) | the same |
| `started_without_a_model` | `{"serving": "not_loaded", "ready_status": 503, "lock_free": true}` | the same |
| `loaded` | `{"serving": "serving", "holds_gpu_lock": true, "lock_names_the_service": true}` | the last two `false` |
| `while_serving` | `runner_start` "refused (409), naming this service's inference"; `training_cli` "refused (exit 1), naming the inference holder; nothing written" | `runner_start` "started (200) and ran to its end (completed, exit 0)"; `training_cli` "ran to its end (exit 0)" |
| `corrupt_reload` | `{"refused": true, "model_kept": true, "holds_gpu_lock": true}` | the same with `false` |
| `stop` | the stop that waited for an accepted diagnosis (below), `gpu_lock_given_back` `true` | the same, `gpu_lock_given_back` `false` |
| `memory_after_the_stop` | `state` `"read"`, `judged` `false`, `unified_memory`, `gpu_memory_used_mb` (a list), `ram_used_gb`, `ram_available_gb`; or `state` `"not_read"` with `reason` | the same |
| `after_the_stop` | `{"training_cli": "ran to its end (exit 0)"}` | the same |
| `loaded_again` | `{"serving": "serving", "holds_gpu_lock": true}` | `holds_gpu_lock` `false` |
| `device_switch` | `cuda_to_cpu` "the GPU lock given back", `cpu_to_cuda` "the GPU lock taken again, naming the service" | `"not applicable on the CPU"` |

**The stop** (`stop`). 8 clients send diagnoses while `POST /api/v1/pipeline/stop`, bound to the child's digest, runs;
each sends one more after the stop has answered. A stop that found no accepted diagnosis still running is followed by
the child loaded again and another stop, at most 12 in all; none in 12 fails W12. The facts are the stop that waited:

- `released` `true`, and `status_after` `{"serving": "not_loaded", "in_flight": 0, "holds_gpu_lock": false}`;
- `diagnoses` = `accepted` + `refused_stopping` + `refused_stopped`; no other answer occurred;
- `accepted` at least 1; `finished_first_as_the_stop_counts` at least 1 and at most `accepted`;
- `sent_after_the_stop_answered` 8, every one refused as stopped (`refused_stopped` at least 8);
- `stops`, from 1 to 12, and `drain_observed` `true`;
- `gpu_lock_given_back`: the stop's words say the lock was given back exactly on CUDA.

After the stop, the lock is free and the status says inference was stopped; when the service has stopped, the lock is
free again.

### Cross-checks that hold by construction

- L7.disease_frozen's `parent_sha256` and `paused.parent_sha256`, and L7.branch's `parent.sha256`, equal L1's
  `taken_from_checkpoint_sha256` and every other L7 `child.parent_sha256`.
- L7.disease_frozen's `resumed.parent_sha256` equals its `paused.sha256`.
- L7.disease_frozen's `version.manifest_sha256` equals L7.disease's and L5's `disease.version.manifest_sha256`.
- L8's `runs` holds one entry with `probe` `"L7.disease_frozen"` and its `run_id`, last.
- W12's `checkpoint_sha256` equals L5's `measured.checkpoint_sha256` and S1's `subject.checkpoint_sha256`: each is a
  copy of D1's disease child.

## 3. The external closing check's next revision

The check lives outside this repository, in the owner's session (TA-10 at `e60f23e`). Its next revision must:

1. **Require the three new IDs** by the tested commit's lists (69 and 15), and read their facts by the keys above, with
   the cross-checks of §2. It must not gate on a metric, a time, a count of diagnoses or stops beyond the stop's
   invariants, or the memory reading.
2. **Read L8 as it now is**: 9 runs, in flow order L2, L3, L5 (disease), L5 (KG link prediction), L6, L7.disease,
   L7.gene, L7.patients_like_me, L7.disease_frozen. A check that expects the eight of `e60f23e` fails a correct run.
3. **If it reads W2 or W5**, expect the route list and the four events of §1.
4. **Keep S1 informational**, as before; it may print W12's `stop` counts and the L7 training times as INFO.
5. **Be dry-run first** on `e60f23e`'s kept reports
   ([`../research-comparison-acceptance-2026-11-12/`](../research-comparison-acceptance-2026-11-12/README.md),
   `pull/probe-report.json` and `compile/probe-report.json`): every check that passed there passes again, reading
   that commit's lists (66 and 15) and its L8 (8 runs), and the three new probes are reported absent and nothing
   else.

## 4. The GB10 run, and what it covers

**The code to test** is the head delivered for the external review. Only documents may follow it before the run.
`deploy.sh`, `pyproject.toml`, `uv.lock` and the launcher are unchanged since `30e8ae1`, so, by the plan's §八 item 6,
one existing accepted environment in a reused candidate slot is enough: no second PyG environment is built for this
round. A6 records the route that environment has, and its version and native identity are recorded before the run.
The owner's formal checkout, deployment, models and ledgers are not touched; the probe works on new copies.

1. **The probe**, while no managed GPU job runs:
   `scripts/probe_deployment.py --m2-acceptance --device cuda --work-dir <a new directory outside the clone>
   --real-workspace <a new copy> --real-reference-checkpoint <its model-02 file> --workspaces-root <the copy's root>`.
2. **The closing check** in its next revision (§3), dry-run first.
3. **The UI steps** of the mapping below, in a browser on the candidate slot's service, recorded with what each
   showed.
4. **A run passes** when the probe exits 0, `m2_acceptance.complete` is true with the tested commit's lists (demo 69,
   real data 15), the closing check passes, and each UI step shows what it is listed for. S1 stays INFO.
5. **The record**: a new evidence directory, `docs/working/ops-wiring-and-ui-ux-acceptance-<date>/README.md`, as the
   research-comparison round's.

**The README's GB10 acceptance (§8), item by item** (the plan's §八 "GB10 最小驗收" in the same order):

| Item | Covered by a probe | Not covered by a probe |
|---|---|---|
| A small base, a branch, a full and a frozen disease fine-tune, saved and continued once; the rules and records checkable | D1 (the demo base, trained on the device) and D2 (its child's provenance); L7.branch (a same-settings branch through the job runner, its record and listing); L7.disease (a full child) and M2 (a full disease fine-tune paused and resumed, float16 with its scaler); L7.disease_frozen (the frozen child paused and resumed in fp32, its record, parts and optimizer checked); R3 (the tested training rules) | A branch that changes a setting, on CUDA (CPU tests only); a start from the Training or Fine-tune page on CUDA: **UI step 1** |
| LP and disease measurement of the two children, kept apart; a repeat never merges a real result away | L5 (a full disease child of the same base by both protocols, two records); L7.disease and L7.disease_frozen (each child by the disease protocol, its own record); L2 and L3 (the same base on the same set twice: two runs, sequences 1 and 2); L8 (every run its own row) | The frozen child's KG link-prediction measurement: **UI step 2** |
| Load a disease model, infer, stop, a research job, load again; a concurrent attempt refused with the shared reason; a bad candidate leaves the served model as it was | W12 (on CUDA: the load holds the lock; a runner start 409 and `train_model.py` exit 1, each naming this service's inference; a corrupt checkpoint refused, the model and the lock kept; diagnoses across a stop; a short training after it; the load again; the device switches); E5 (a refused reload keeps serving) | The same hand-off from the pages: **UI step 3**. A start-up with an explicit model config on CUDA is exercised by S1, which is informational, so its failure would show only as INFO |
| Gene and patients-like-me through the UI on the small functional data, marked as not ground truth | L7.gene and L7.patients_like_me (each child trained on the device, measured on the generated split's fixture and registered); not through the UI | **UI step 4** |
| C1's GPU, dtype and vertical acceptance, only once C1 is chosen | Not applicable: no C1 in this round. A6 and the scope say only `gatv2` is declared and built | HGT, SAGE, hetero GAT; bfloat16 |
| No accuracy gain required | No probe and no check judges a metric | — |

**The UI steps** (the owner's, by hand; not probes). The service is the candidate slot's, started by the normal
launcher, with a copy of the probe's flow deployment (`<work>/rc/workspaces`, its registry `<work>/rc/registry`) as its
workspaces root and registry, so the probe's own files stay as they were:

1. **Training and Fine-tune**: from `probe-rc-base`, a branch with one setting changed from the Continue drawer, and a
   frozen disease fine-tune from Fine-tune (Graph backbone: Frozen), each on CUDA to its end; each listed under its
   parent with what it changed or froze.
2. **Test**: the frozen child (`probe-rc-disease-frozen-resumed`) measured by KG link prediction on
   `probe-kg-test-1`, registered; Results lists it apart from its disease record and from L5's.
3. **Diagnosis**: load a disease child, run one query, Stop inference (the GPU chip names this service's inference
   before and is free after); start a short Training job and let it end; load again; while it serves on CUDA, a
   Training start is refused naming inference and how to stop it. A bad candidate is W12's and E5's (HTTP and in
   process), not a UI step.
4. **Diagnosis, the other tasks**: load the gene child (`probe-rc-gene`) and run one query with candidate genes, then
   the patients-like-me child (`probe-rc-patients-like-me`) and one query; each result names its model and carries the
   synthetic banner ("SYNTHETIC WORKSPACE: not a clinical result.", `renderers.IDENTITY_BANNERS`).

**What the run does not cover, already known**: AMP, its scaler or `torch.compile` with a freeze (L7.disease_frozen
is fp32 only, and says so); Restart Backend with a CUDA model loaded (CPU tests only,
`tests/unit/test_backend_control.py`); concurrent loads or stops on one service (CPU tests only,
`tests/unit/test_serving_lifecycle.py`); the memory a stop returns (recorded, never judged); HGT, SAGE or hetero GAT;
bfloat16; model quality.

**Time.** The new probes add demo-sized work only: L7.disease_frozen two short trainings (2 epochs in all) and one
measurement; L7.branch one branch at the base's epochs through the job runner; W12 one service start, up to 12 stops
under 8 diagnosis clients, each after a load, and on the CPU three short trainings (one on CUDA, the other two being
refused there). On the CPU here, the whole demo phase test, every demo probe included, took about 90 s.
