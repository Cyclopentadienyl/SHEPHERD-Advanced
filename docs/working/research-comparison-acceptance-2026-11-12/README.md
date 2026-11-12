# The research-comparison round's GB10 acceptance at `e60f23e`, 2026-11-11/12

**Passed on both deploy routes: the prebuilt release (pull) and a local build (compile), each run in its own new
environment.**

- On each route, 116 of 116 probes passed (0 failed, 0 errors, 0 skipped), on the demo and on the real workspace
  `hpo_2026_0929_5a`. The tested commit's required lists are complete: demo 66, real data 15.
- The closing check, in its TA-10 revision, passed 182 of 182 on each route (`pull/acceptance-check.txt`,
  `compile/acceptance-check.txt`).
- A6 read the route each run was meant to verify: "prebuilt release" for all four PyG extensions on pull, "local build"
  for all four on compile. The pull did not fail, and nothing fell back to a build.
- Both runs used one existing clone, the K8 candidate clone, one after the other. That is not O9's isolated new clone,
  nor the fresh clones of `../research-comparison/GB10_ACCEPTANCE.md` §4; the reviewer's frozen operation sheet
  planned this reuse (§1, §3).

This is a functional acceptance, not a measure of model quality, a clinical test or a merge.

**The code tested is `e60f23ee0314c33d7785d13656a146e0ca0022d8`.** Its product code equals `ee36cb9`'s; `e60f23e`
adds one line to the round's record. The reviewer's narrow recheck of `e17ffe7`..`e60f23e` passed with no P1 or P2.
Only this record and the status lines it updates follow `e60f23e`. They touch documentation alone, so `src`,
`scripts`, `tests`, `configs`, `pyproject.toml` and `uv.lock` are the ones tested.

## 1. The run

Common to both runs:

- **Machine:** NVIDIA GB10 (capability 12.1; driver 580.178.04, from `nvidia-smi`, not a product fact), Linux
  aarch64, Python 3.12.3, torch `2.10.0+cu130` (CUDA build 13.0), `torch_geometric` 2.8.1 from the lock.
  `cudaMallocAsync` from the saved preset, set before torch's first import; `torch_compile` false.
- **Probe:** `scripts/probe_deployment.py --m2-acceptance --device cuda`, with `--real-workspace` a new copy of
  `hpo_2026_0929_5a` per route, `--real-reference-checkpoint` its `model-02-0.1813.pt` (a file without a model spec),
  and `--workspaces-root` the copy's root. The probe writes its report as `work/k8_report.json`; it is handed over as
  `probe-report.json`.
- **Lock (A5):** `uv.lock` is the one committed at `e60f23e` (`0f92a5ec…c356`). No installed package it lists differs
  from it, and no dependency the project declares is missing. 66 of its 172 packages are not installed.
- **Inputs:** before each probe, the copy equalled its source byte for byte. The five inputs equal M2–M4's digests
  before and after each run: `config.yaml` `14bf8eea…970e`, `model-02-0.1813.pt` `33a7b39a…8b79`,
  `kg.provenance.json` `2c7abdb6…12ef`, `split_manifest.json` `131cdf81…f664`, `kg.json` `6cae2d1a…1e43`
  (`input-sha256.txt`, `input-sha256-after.txt`; the after-check reads the probe's copy). R8 finds the workspace's 19
  entries unchanged.
- **Run by:** the owner with the reviewer's session, on the homelab GB10, through a staged runner outside the
  repository (`rc-acceptance.sh`, SHA-256 `58844064…897b`) and the reviewer's frozen control bundle (§4). The notes,
  in Chinese and as written, are `acceptance-conclusion.zh.txt` and `version-review.zh.txt`. Where the conclusion says
  the whole real-data path ran in fp32, the report records precision for R4 only (§2). Where it says each original
  report equals its copy byte for byte, see §3. The reviewer's review of this record follows.

| | Pull 01 | Compile 01 |
|---|---|---|
| Probe time | 2026-11-11 15:12:32–15:55:54 UTC (23:12–23:55 Taipei) | 2026-11-12 02:45:25–03:28:42 UTC (10:45–11:28 Taipei) |
| Checkout | The K8 candidate clone, at `f2c2012` before. Its `.venv`, caches and `egg-info` were removed; `.git`, `data`, `models`, `outputs`, `reports` and the runtime preset were kept. `e60f23e` was fetched and checked out detached | The same clone after pull 01. Only pull's `.venv` was removed; `e60f23e` fetched and checked out detached |
| Checkout state | Clean before and after; no tracked file modified (the report's `tested_commit`) | The same |
| Install | `PYG_ARM_STRATEGY=pull`, a failed pull set to stop. The release `pyg-arm-cu130-torch2.10.0`, its asset's SHA-256 verified (`7cba46d2…8967`). 4 of 4 wheels installed as they hold them; nothing built | `PYG_ARM_STRATEGY=compile`; no pull attempted. `pyg-lib` 0.8.0 built from `pyg-team/pyg-lib` tag 0.8.0 (commit `26a2d1a`), and `torch-scatter`, `torch-sparse` and `torch-cluster` built by name from the package index (`pip wheel --no-cache-dir`, 308, 350 and 530 s), in a fresh directory. The four wheels it built were installed exactly |
| Receipt | `scripts/validate_pyg_ext.py --receipt`: 4 of 4 as the receipt records, route "prebuilt release" | 4 of 4, route "local build" |
| PyG extensions | `pyg-lib` 0.6.0, `torch-scatter` 2.1.2, `torch-sparse` 0.6.18, `torch-cluster` 1.6.3 | `pyg-lib` 0.8.0, the other three at the same versions. Every native aggregate digest differs from the release's |
| Lock (A5) | 119 installed; 13 outside the lock, from the deploy scripts: `cuda-python`, `cuda-toolkit`, the seven packages of the cuVS chain and the four PyG extensions | 121 installed; 15 outside the lock: the same 13, and `pip` 26.3 and `wheel` 0.48.0, which the build added |
| Outcome | **116 passed.** `m2_acceptance` complete. Probe exit 0, `tee` exit 0. Every stage, `init` to `after`, passed. The environment facts were byte-identical at `envcheck`, before the probe and after it (`dd263f03…47b4`) | **116 passed**, the same. Environment facts unchanged (`502ae3f9…464f`) |
| Closing check | TA-10: **182 passed, 0 failed** | **182 passed, 0 failed** |
| Duration and disk | The probe ran 2,602 s. R4.finetune's disease child trained 1,142.8 s, and RW2's 1,082.0 s. Largest 15 s disk sample 14,867,140,775 bytes; final size 14,867,280,429 bytes; least free 3.10 TB | 2,597 s. 1,125.1 s and 1,100.6 s. Largest sample 14,857,318,251 bytes; final size 14,867,382,590 bytes; least free 3.07 TB |

As at K8, each run's largest sample is below its final size, so the peak was at least the final size. These figures
are recorded, not judged (`../BACKLOG.md` 64). The probe times show nothing about the routes or the PyG versions.

## 2. What it shows

- **The round's new flow on each route, with `--device cuda`** (A6, L1–L8, the field contract of
  `../research-comparison/GB10_ACCEPTANCE.md`). L2, L3, L5, L6 and L7 record that they ran on CUDA; L1 imports, L4
  refuses before any scoring and L8 exports, so they compute nothing on the GPU:
  - L1: a KG edge set of 6 test pairs imported from the demo base's recorded split;
  - L2 and L3: KG link prediction measured through `measure_scorer` and through the job runner, kept with its scores
    and registered with its exposure. The probe recomputed the scores within its tolerance (largest differences 1.2e-7
    and 2.4e-7 on pull, 2.4e-7 and 6.0e-8 on compile);
  - L4: four refusals before anything is written: another graph, changed pairs, another graph view, a contradicting
    split record;
  - L5: one disease child measured by both the disease and the KG link-prediction protocols;
  - L6: a kept generated split (`val`) measured;
  - L7: one-epoch disease, gene and patients-like-me children trained on CUDA under training rules 2, at demo sizes,
    and measured;
  - L8: the export: 8 results, 142 metric rows, 0 unreadable, not capped. Compare's limit of 5 in the UI is not an
    export limit.
  - The cross-checks of `GB10_ACCEPTANCE.md` §3 hold: one manifest across L1, L2, L3 and L5; one base across L1, L2,
    L3 and L7's parents; L8's runs are those of L2, L3, L5, L6 and L7. Each route trained its own base, so their
    digests differ between the routes.
- **The K8 flow again, at this code, on the real data:**
  - R3: a GATv2 base under training rules 2, the tested commit's rules;
  - R4: the base resumed, fp32 with no gradient scaler. R4 is the only real-data step whose facts record its
    precision;
  - R4.finetune: a disease child of it; R5 measured it (15,000 simulated validation cases) and R6 served it on CUDA;
  - RT1–RT3 (M4's flow) and RW1–RW3 (the server's lists, a test run kept and registered, a training job holding the GPU
    lock and stopped);
  - R2: the deployment's file without a model spec is refused by finetune and resume, and a stripped copy of the child
    also by serving, before anything is written.
- **The demo flow:** A–E, M1–M11, T1–T6 and W1–W11, as the tested commit's required list states. M2 and M9 ran float16
  with a gradient scaler, restored from the parent before the first update.
- **Each route as itself.** A6 and the receipt name the route, and each component's native identity is recorded. The
  two environments differ (`pyg-lib` 0.6.0 and 0.8.0, and every native digest), and both pass. No result is required
  to equal the other route's.
- **S1's served readings now exist on the GB10** (informational: in neither required list, never judged). On pull and
  compile: cold start 7.6 s each; steady-state median 2.08 and 1.99 ms per request (the server's handler 1.65 and 1.57
  ms); the server's RSS 1.61 GB at readiness and 1.91 GB after the requests. The CUDA allocator peak is recorded as not
  obtainable.

## 3. What it does not show

- **The new clones O9 and the run sheet asked for.** O9 (`../research-comparison/README.md` §3) asked for an isolated
  new clone. `GB10_ACCEPTANCE.md` §4 asked for a fresh clone, and for the local build another clone. The reviewer's
  frozen operation sheet (`本機驗收操作單.txt`, in each archive's `00-init/control_bundle/`) planned instead to reuse
  the K8 candidate clone: its `clone` stage fetches `e60f23e` there and checks it out detached. Both runs used it, one
  after the other (§1). Each run had its own new `.venv`, data copy, work directory and evidence directory, and the
  checkout was detached and clean. The environments were new; the clone was not. The package's conclusion
  (`acceptance-conclusion.zh.txt` §六) says no new clone is needed for this round's pass.
- **The deploy layer's failure paths on the GB10:** a failed pull's stop, the `PYG_ARM_ON_PULL_FAILURE=build`
  fallback, a wrong wheel and a mismatched receipt. These four remain software simulation
  (`tests/unit/test_deploy_pyg_route.py`). The Linux index route and `deploy.cmd` are outside both the GB10 run and
  that simulation.
- **The source of the compile route's three registry builds.** The receipt does not name the source distribution
  `torch-scatter`, `torch-sparse` and `torch-cluster` were built from. It records the request (by name, with no
  version), the resolved version, the wheel and native digests, and the build's toolchain and flags. The build logs
  stay on the homelab.
- **Whether an extension took effect.** "requested", "selected" and "in effect" read unknown for every component (12
  values per route). A matching receipt and passing CUDA probes do not stand for them.
- **From `GB10_ACCEPTANCE.md` §4's list:** the frozen-backbone control on CUDA, with AMP, its scaler or `torch.compile`; the
  WebUI in a browser on the GB10; the server's CUDA allocator counters; HGT, SAGE or hetero GAT; bfloat16; model
  quality.
- **On the real data:** gene and patients-like-me children (L7 trains them on the demo only).
- **A controlled PyG-version comparison.** The routes differ in more than `pyg-lib`'s version, so nothing follows
  about one version's quality or speed.
- **cuVS:** installed by the deploy scripts but without `cupy`, so not part of the pass. The required Voyager backend
  is checked.
- **Model quality, capacity, other hardware, an institutional test.**
- **What the handoff cannot show here.** The package states that each handed-over `probe-report.json` equals its run's
  `work/k8_report.json` byte for byte, and that each archive's 123 files equal the evidence on the GB10; `work/` stays
  there. TA-10 read the original (`rc-acceptance.sh`, line 511). The reviewer's replay gave byte-identical outputs
  (§4), but the package does not say which of the two files it read. The deployment checkout was clean at `773dc4a`
  when the package was made; the runs' evidence holds no before-and-after record of it.

The probe's ERROR lines belong to its injected faults (E5, M6, M11.serving, R2), each refused as required. PyTorch's
build lists `sm_80`–`sm_120` and `compute_120` PTX, not 12.1 natively (A3). That is kept in the record; every CUDA step
here passed.

## 4. The closing check (TA-10)

- **Who and what.** The reviewer wrote the TA-10 revision and the runner. Both ran from a frozen control bundle whose
  `SHA256SUMS` (`28733205…0f64`, 22 files) the runner checks at the start of every stage. The bundle is the same on
  both routes. The checker is `ta10.py` (`06d2dca3…d7aa`). It reads the tested commit's contract through `contract.py`
  (`9875e21e…84a0`): digest-checked copies of `scripts/probe_deployment.py`, `src/utils/checkpoint_io.py` and
  `uv.lock` at `f2c2012` and `e60f23e`. Each copy equals the repository's file at that commit.
- **What it does, per `GB10_ACCEPTANCE.md` §3:** it requires the 14 new probes and their cross-checks; it drops W8's
  empty line and reads RW2's and R4.finetune's training time where they are recorded; R3 expects the tested commit's
  rules, 2. S1 is not required and is printed as INFO. Each route printed 11 INFO lines, among them the LP score
  differences, the L7 training times and the disk samples.
- **Where it reads more strictly than §3's wording:**
  - it takes the required lists and R3's rules from the pinned copies of the tested commit, and requires the report's
    lists and tested commit to equal them. §3 says to take the lists from the report. The outcome is the same here, and
    the stricter form also refuses a report that drops a probe from its own list;
  - it requires A6's route and each component's recorded route to equal the route under test (four checks per route).
    That is §4 item 2's rule for a verified pull. §3 item 1's "do not gate on a route" is read as: do not pin which
    route a run must use.
- **Dry runs before the GB10 runs** (not GPU evidence):
  - on K8's kept report, TA-10 passed 114 of 114: K8's 110 checks, with R3 reading `f2c2012`'s rules (1), and 4 report
    structure checks. RW2 read 1,191.9 s, no W8 line was printed, and the new probes were reported absent;
  - K8's unmodified checker passed 110 of 110 on the same report. Its output equals K8's `acceptance-check.txt`;
  - 26 cases of the checker (6 expected to pass, 20 to fail) and 6 cases of the clone's recycle guard all came out as
    expected.
- **The reviewer's replay** of TA-10 on both runs' reports gave `ta10.json` and logs byte-identical to the runs' own.
  The package does not say whether it read the originals or the handed-over copies (§3).

## 5. Files here

| File | What it is |
|---|---|
| `pull/probe-report.json`, `compile/probe-report.json` | The handed-over copy of each run's report (`work/k8_report.json`), byte for byte as handed over (§3). Neither names a local path |
| `pull/ta10.json`, `compile/ta10.json` | TA-10's result: 182 checks, each with its value, and the INFO lines |
| `pull/acceptance-check.txt`, `compile/acceptance-check.txt` | TA-10's console output and the count |
| `pull/exit.txt`, `pull/exit-both.txt` and the same in `compile/` | The probe's exit status, and the probe's and its `tee`'s |
| `pull/environment.json`, `compile/environment.json` | The product's environment facts at `envcheck`, with each component's facts and the receipt; byte-identical before and after the probe |
| `pull/input-sha256.txt`, `pull/input-sha256-after.txt` and the same in `compile/` | The five inputs' digests before the run, and the check after it |
| `acceptance-conclusion.zh.txt`, `version-review.zh.txt` | The run's notes, in Chinese and as written: the conclusion, and the version and environment review. The file names they cite are the handoff package's: `pull/environment.json` and `compile/environment.json` here are each run's `04-envcheck/environment.json`, and the per-file native digests are in those files, not in the package's `summary.json` |
| `SHA256SUMS` | The digests of the files above |
| `.gitattributes` | Turns off line-ending normalisation here, and exempts TA-10's output, which prints a space after an empty value, from the whitespace check |

The rest stays on the homelab, in `/home/user/rc-acceptance-e60f23e-pull-01/` and
`/home/user/rc-acceptance-e60f23e-compile-01/`: each run's work directory (about 14.9 GB), build files and wheels. The
candidate clone keeps compile 01's environment. The run's handoff package, `SHEPHERD_RC_GB10_acceptance_e60f23e_20261112`,
came in two parts:

- `…_part1of2.zip` (18,982,086 bytes, SHA-256 `b1ab6d243c1237ce34ba79f6acb814e37c61b366a0d066053d0b70fb75d77897`);
- `…_part2of2.zip` (16,790,210 bytes, SHA-256 `195b16175e88a9cc5a3525e68415bec9937c3d5db33858ac345d83faa0d60dc2`).

Together they hold:

- the notes and a machine-readable summary;
- the reviewer's TA-10 replays;
- the frozen control bundle and the earlier preparation package;
- each run's archive, with every stage's record, the logs and the control bundle as used:
  - `rc-acceptance-e60f23e-pull-01-handoff.tar.gz` (SHA-256 `429a647a…1e97`, 123 files);
  - `rc-acceptance-e60f23e-compile-01-handoff.tar.gz` (SHA-256 `69241e5e…10a7`, 123 files).

The package's `SHA256SUMS` covers its 20 files.
