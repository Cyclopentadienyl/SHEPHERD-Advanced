# K8's GB10 acceptance at `f2c2012`, 2026-11-09

**Passed: the K8 reference flow, in an explicitly selected source-built PyG environment.**

- 101 of 101 probes passed (0 failed, 0 errors, 0 skipped), on the demo and the real workspace.
- The script's closing check passed 110 of 110 (`acceptance-check.txt`).
- The deployment probe ran on a fresh clone at the reviewed K8 head and on a new copy of the real workspace
  `hpo_2026_0929_5a`.

**The environment is not the one `deploy.sh` pins.**

- GitHub answered the prebuilt PyG release download with HTTP 504, and `deploy.sh` fell back to a source build. That
  build resolved `pyg-lib` 0.8.0, where the pinned prebuilt release carries 0.6.0.
- The first attempt therefore failed at its install gate, and its record stays failed.
- The owner then chose to accept the K8 flow in that source-built environment, to check that the product does not
  depend on one PyG build. A second attempt adopted the environment without reinstalling it (§4).
- **The prebuilt deployment path is not exercised at this code.** Its earlier GB10 passes (`0bc5178`, `b14c4de`) are
  other code and do not carry over (`../BACKLOG.md` 50, 70).

This is a functional acceptance, not a measure of model quality, a clinical test or a merge.

**The code tested is `f2c201213fd2da59588af7bef2cb030951bdb5c3`.** Only this record follows it. It touches
documentation alone, so `src`, `scripts`, `tests`, `configs`, `pyproject.toml` and `uv.lock` are the ones tested.

## 1. The run

| | |
|---|---|
| Commit | `f2c2012`, checked out detached in a fresh clone; no tracked file modified before or after (the report's `tested_commit`). The deployment checkout (`773dc4a`) was not touched |
| How it was run | A staged acceptance script outside the repository, run on the homelab GB10 by the owner's session, one stage per call. It is K8's reviewed script (SHA-256 `7b63e05a…4402`, F1 fixed) with the environment stages changed (§4). The used script's SHA-256 is `ea21db70…f569`; `source-build-acceptance.diff` is the change, byte for byte. Attempt 02's stages `init adopt preset envcheck data probe after` all passed, and its evidence was handed over as a review package (§5) |
| Environment | NVIDIA GB10 (capability 12.1, driver 580.178.04), Linux aarch64, Python 3.12.3, torch `2.10.0+cu130` (CUDA build 13.0), Gradio 6.30.0, `torch_geometric` 2.8.1 from the lock. PyG extensions: `pyg-lib` 0.8.0, `torch-scatter` 2.1.2, `torch-sparse` 0.6.18, `torch-cluster` 1.6.3, all as one recorded source build (§4). `cudaMallocAsync` from the saved preset, set before torch's first import |
| Lock (A5) | `uv.lock` is the one committed at `f2c2012`, and every package it lists is installed at its version. 15 installed packages are outside the lock: 13 from the deploy scripts, plus `pip` and `wheel` |
| Probe | `scripts/probe_deployment.py --m2-acceptance --device cuda`, with `--real-workspace` a new copy of `hpo_2026_0929_5a`, `--real-reference-checkpoint` its `model-02-0.1813.pt` (a file without a model spec), and `--workspaces-root` the copy's root. There is no `--real-config` (Q9 (A)) |
| Inputs | Before the probe, the copy was compared byte for byte with the read-only backup. The five inputs equal M2–M4's recorded digests before and after the run: `config.yaml` `14bf8eea…970e`, `model-02-0.1813.pt` `33a7b39a…8b79`, `kg.provenance.json` `2c7abdb6…12ef`, `split_manifest.json` `131cdf81…f664`, `kg.json` `6cae2d1a…1e43`. R8 finds the workspace's 19 entries unchanged |
| Outcome | **101 probes passed.** The tested commit's required lists, demo 52 and real data 15, are both complete. Probe exit 0, `tee` exit 0. The closing check, **110 of 110**, passed: the run's record (stages, exits, inputs, checkout, environment) and the facts R1–R4, R8, M2 and M9 |
| Duration and disk | The probe ran 2,781 s (06:30:22–07:16:43 UTC). R4.finetune's disease child trained 1,242.9 s, and RW2's 1,191.9 s. The work directory peaked at 14,853,412,907 bytes (sampled every 15 s), ended at 14,853,498,462 bytes, and least free space was 3.1 TB. These are recorded, not judged (`../BACKLOG.md` 64) |
| Run by | The owner's session on the homelab, 2026-11-09. Its notes, in Chinese and as written, are `acceptance-conclusion.zh.txt` and `version-review.zh.txt`. The reviewer's review of this record follows |

## 2. What it shows

- **The two-stage training flow on CUDA, on the real data:**
  - R3: a GATv2 base under training rules 1, from the operator's one stated setting and the reference recipe;
  - R4: the base resumed, fp32 with no gradient scaler;
  - R4.finetune: a disease child of it;
  - R5 and R6: the child measured and served on CUDA;
  - RT1–RT3: M4's flow;
  - RW1–RW3: the server's lists, a test run kept and registered, and a training job holding the GPU lock and stopped as
    stopped.
- **AMP on CUDA across a continuation.** M2 and M9 ran float16 with a gradient scaler, and the scaler was restored from
  the parent before the first update.
- **Natural retirement holds on the deployment's own file.** `model-02-0.1813.pt`, which has no model spec, is refused
  by finetune and resume before anything is written, and stays unchanged. A copy of R4.finetune's child stripped of its
  spec is refused by finetune, resume and serving (R2, M5.no_model_spec).
- **The demo flow:** A–E, M1–M11, T1–T6 and W1–W11, the WebUI's server foundations among them, as the tested commit's
  required list states.
- **The PyG build does not change what the product verifies.** The data and model fingerprint checks, the probe and the
  closing check are the reviewed ones. Only the identity of the environment changed (§4).

## 3. What it does not show

- **The prebuilt deployment path.** Attempt 01's prebuilt install did not complete (HTTP 504). As `../BACKLOG.md` 50
  decided, the PyG extensions stay per platform (the index for the installed torch, the prebuilt ARM release or a local
  build), and an environment holds only its own acceptance. This one is a local build.
- **`deploy.sh`'s fallback as a route:** the fallback and its versions are easy to miss, and the builder installs by
  name (`../BACKLOG.md` 70). The record of a local build does not identify it (`../BACKLOG.md` 71).
- **On GPU, other heads and architectures:** gene and patients-like-me (CPU tests only); HGT, SAGE or hetero GAT;
  bfloat16; the WebUI in a browser.
- **Served resource and latency:** M1 is retired and pending re-attachment (`../BACKLOG.md` 64). Disease accuracy is
  not its equivalent.
- **cuVS:** the optional backend lacks `cupy` here and is not part of the pass. The required Voyager backend is checked.
- **Model quality, capacity, other hardware, an institutional test.**

The probe's ERROR lines belong to its injected faults, each refused as required. PyTorch's notice that capability 12.1
is above its listed 12.0 is kept in the logs; every CUDA step here passed.

The closing check prints two INFO lines that read nothing: `W8 training_seconds None` and `RW2 training_seconds None`.
They are the acceptance script's own fault, not the product's:

- W8 trains nothing;
- RW2 records its time under `measured.training_seconds`, which is 1,191.9 s above.

These lines are informational and change no verdict. They are for the script's next revision.

## 4. The environment, and how attempt 02 adopted it

- **Attempt 01** (`/home/user/k8-acceptance-f2c2012-01`, kept as it is) failed at install, after "CUDA smoke test
  passed" and "Host matches prebuilt target":
  - the release download returned HTTP 504;
  - `deploy.sh` switched to a source build;
  - the script's gate requires "SHA-256 verified" for the prebuilt wheels, so the stage stopped there.
- **The source build:**
  - `pyg-lib` was built from `git+https://github.com/pyg-team/pyg-lib.git@0.8.0`, commit
    `26a2d1a06a4714da9bb804c19049afb4796971d7`, for torch 2.10.0+cu130 with `CUDA_HOME` `/usr/local/cuda-13.0` and arch
    list 12.1;
  - the other three were rebuilt at the versions the prebuilt release carries;
  - for those three, the one native library the acceptance checks has the same digest as the prebuilt release records,
    while `pyg-lib`'s differs;
  - the four wheels and the build logs are kept with the run (not here). The repository's `scripts/validate_pyg_ext.py`
    passed 5 of 5 on CUDA before the acceptance, `segment_matmul` among them, and again in `envcheck`.
- **Attempt 02** replaced `clone` and `install` with `adopt`. It installs nothing and marks nothing passed that failed.
  - It requires attempt 01's clone to have passed and its install to have failed.
  - It checks the chosen environment against a frozen profile: the package list, each distribution's version, and the
    SHA-256 of all 51 native libraries.
  - The same check runs at `adopt`, `envcheck`, before the probe and after it, and every time found nothing changed.
  - `envcheck`'s identity table names the selected build where it named the prebuilt release.
  - `source-profile-checks.txt` shows the profile check refusing `pyg-lib` 0.6.0 and a changed `libpyg.so`.
- **Unchanged from the reviewed script:** the probe, its arguments, the data copy and its comparisons, the five input
  digests, and the closing check (the `after` stage's Python is byte-identical).

## 5. Files here

| File | What it is |
|---|---|
| `k8_report.json` | The probe's report, byte for byte. It names no local path and no case id |
| `exit.txt`, `exit-both.txt` | The probe's exit status, and the probe's and its `tee`'s |
| `acceptance-check.txt` | The script's closing check: 110 results and the count |
| `source-profile-checks.txt` | The environment profile check: the selected build accepted, a wrong version and a changed native library refused |
| `source-build-acceptance.diff` | The used acceptance script against K8's reviewed one |
| `acceptance-conclusion.zh.txt`, `version-review.zh.txt` | The run's notes, in Chinese and as written: the conclusion, and the PyG version review written before the full run (its "not yet run" describes that time) |
| `SHA256SUMS` | The digests of the files above |
| `.gitattributes` | Turns off line-ending normalisation here, and exempts the diff's single-space context lines from the whitespace check |

The rest stays on the homelab in `/home/user/k8-acceptance-f2c2012-02/` (about 14.9 GB) and
`/home/user/k8-acceptance-f2c2012-01/`. The run's review package,
`SHEPHERD_K8_GB10_f2c2012_pyg080_20261109_review.zip` (SHA-256
`28cdce18b3ad08d3a06a143169d21c81d8a1d8cc27b0c55a6fa236dac546166f`), holds:

- the stage records;
- attempt 01's failure;
- the scripts and their diff;
- the build logs and the four wheels;
- the console log, reports, ledgers and usage history.

Its `HANDOFF-SHA256SUMS` covers 129 files. Its `原始產物清單.json` lists every file left out (the model tensors and the
large input copies) by path, size and SHA-256.
