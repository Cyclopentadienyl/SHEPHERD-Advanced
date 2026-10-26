# Batch 1's minimal deployment and environment check on the GB10, 2026-10-26

**Passed, within the scope agreed, as the reviewer's review of the evidence found (2026-10-26).**
The check covered BACKLOG items 49 and 50 (Batch 1). It ran a fresh clone and a fresh environment
from the formal installer `deploy.sh`, then the deployment probe's environment phase (A1–A5), a
path check of the probe's argument handling, and the PyG native-extension check. It is not M5's
functional acceptance and not a full deployment acceptance. It is not a measure of model quality, a
capacity reading, or a test on any other platform. It authorises no merge.

**The code tested is `0bc5178743c6ae1243910702ce37d000b314413c`.** The commits after it touch no
product code: `60feb6d` (docs), `7b0162f` (Batch 2's design, docs only) and this record were not
tested again. Between `0bc5178` and this record, `src`, `scripts`, `tests`, `deploy.sh`,
`deploy.cmd`, `pyproject.toml`, `uv.lock` and `configs` are unchanged.

## 1. The run

| | |
|---|---|
| Commit | `0bc5178`, checked out detached in a fresh clone at `/home/user/batch1-acceptance-0bc5178-01/repo` ([`checkout.log`](checkout.log), [`preparation.txt`](preparation.txt)); tracked files clean at the end ([`final-state.txt`](final-state.txt)) |
| Inputs | `pyproject.toml` `f458369c…cd80` and `uv.lock` `0f92a5ec…c356`, the committed files; equal before and after the run ([`source-inputs.sha256`](source-inputs.sha256), [`source-recheck.txt`](source-recheck.txt)) |
| Environment | a new `.venv` from `/usr/bin/python3.12` (Python 3.12.3, [`venv.log`](venv.log)); NVIDIA GB10, driver 580.178.04 ([`gpu-before.txt`](gpu-before.txt)); uv 0.11.16 |
| Install | `deploy.sh` with `PYTHON_EXE=/usr/bin/python3.12 PYG_ARM_STRATEGY=pull UV_NO_PROGRESS=1`, exit 0, `tee` exit 0 ([`deploy-exit.txt`](deploy-exit.txt)). In the log ([`deploy.log`](deploy.log)): `uv sync --inexact` installs the lock's 106 packages, `torch_geometric` 2.8.1 among them ("torch_geometric 2.8.1 (from uv.lock)"; no separate install of it); the four ARM extensions come from the prebuilt release after its SHA-256 check (pyg-lib 0.6.0, torch-scatter 2.1.2, torch-sparse 0.6.18, torch-cluster 1.6.3), with no source build and no fallback. The sync took 95 packages from uv's cache, so this was a new environment, not a cold download |
| Environment phase | the probe's `phase_environment(Report(), 'cuda')` called directly, A1–A5 only: **5 passed**, 0 failed, 0 skipped, 0 errors ([`environment-check.json`](environment-check.json), [`environment-check.log`](environment-check.log)). torch 2.10.0+cu130, `torch_geometric` 2.8.1, all four extensions present (A1); the serving device `cuda`, capability 12.1 (A2); kernels execute and agree with the host, matmul error 4.6e-05 (A4) |
| Lock (A5) | the lock read is the one committed at the tested commit; versions match the lock, none differs, no declared dependency missing; 13 installed packages are outside the lock, all from the deploy scripts (the four PyG extensions, cuVS and what it requires). `torch_geometric` is no longer among them: at M5's run (`b14c4de`) it was, with nine others, which Batch 1 added to the lock at the same versions (23 outside the lock then, 13 now) |
| Probe paths (item 49) | the same module's `parse_args` called with relative and absolute forms of the path arguments: the same targets, the relative ones made absolute, and no work directory created (`environment-check.json`, `checks`: `relative_absolute_same_target`, `relative_paths_made_absolute`, `path_check_created_no_work_directory`, all true). The refusal case and the regression with a symbolic link followed by `..` were not run here; they are the unit tests reviewed at `0bc5178` |
| PyG kernels | `scripts/validate_pyg_ext.py` on CUDA: **5 of 5** (scatter-sum, sparse matmul, knn graph, pyg-lib's load with its `segment_matmul` kernel, a SAGEConv forward), exit 0 ([`pyg-kernels.log`](pyg-kernels.log), [`pyg-kernels-exit.txt`](pyg-kernels-exit.txt)) |
| Existing checkout | the deployment checkout at `/home/user/Desktop/SHEPHERD-Advanced` read `773dc4a`, clean, before and after ([`existing-checkout-before.txt`](existing-checkout-before.txt), [`existing-checkout-after.txt`](existing-checkout-after.txt)) |
| Run by | the owner and the reviewer, on the homelab; reviewed by the reviewer the same day ([`CODEX_REVIEW.txt`](CODEX_REVIEW.txt), in Chinese, as written; [`review-verdict.json`](review-verdict.json)) |

## 2. What it shows

- **Item 50 on the GB10.** A fresh install through the formal entry gets `torch_geometric` from
  `uv.lock` through the sync. A5 compares the installed set with the committed lock and finds it
  equal. The native extensions still come per platform: on this ARM host, from the prebuilt release,
  checked by SHA-256, and they run on the device.
- **Item 49 on the GB10, as far as recorded.** The probe's path arguments are made absolute once,
  from the starting directory, and reach the same files as the absolute forms.
- **Nothing outside the run changed:** the inputs, the tested checkout and the existing deployment
  checkout read the same before and after.

## 3. What it does not show

- **Other platforms.** `deploy.cmd` on Windows was not run. The x86 path, which derives the PyG
  index from the installed torch, was not exercised either: this host took the ARM prebuilt path.
- **The launcher's accelerators.** No `--xformers`, FlashAttention or SageAttention install was tried
  on the GB10. The constrained install (`shep_launch.pip_install` with `uv export` constraints)
  rests on its unit tests at `0bc5178`.
- **A5's other branches.** Only the equal case ran here. The "version differs" and "declared but not
  installed" cases rest on the unit tests.
- **Every kernel.** torch warns that the GB10's capability 12.1 is above the 12.0 its build declares
  (A3: no native kernels for 12.1, PTX for `compute_120`). The evidence is the small kernels that
  ran, A4 and the PyG check, not every kernel.
- **The GPU vector backend.** cuVS 26.6.0 is installed without cupy, so the GPU backend cannot be
  built. The installer reports this ("cuVS installed but its prerequisites are not importable"), and
  Voyager imports and is chosen. This is known and recorded: BACKLOG §2.5 (item 13), and
  `../../RETRIEVAL_AND_CANDIDATE_DISCOVERY_FINDINGS.md` §2.5. Diagnosis does not use retrieval (the
  `diagnosis-retrieval-detachment` import contract). The installer runs cuVS's install with its
  output discarded, which is why the nine cuVS packages appear in A5 and `installed-packages.json`
  but in no install line of the log.
- **The same environment as M5's.** The packages outside the lock are recorded here by name and
  version; this record does not claim they equal M5's.
- **M5, capacity, model quality.** No training, service, measurement or WebUI ran.

## 4. Read with these notes

- **The wrapper's exit 1 is not a failed probe.** [`environment-check-exit.txt`](environment-check-exit.txt)
  reads `environment_check=1`, and `environment-check.json` has `"passed": false`. Both come only from
  the reviewer's own extra assertion, `no_duplicate_distributions`, which the probe does not make.
  A1–A5 all passed. The one name it caught is this project's: `shepherd-advanced` 0.2.0 appears
  twice, as the editable install's `.dist-info` in the environment (its `direct_url` this checkout,
  editable) and as `src/shepherd_advanced.egg-info` in the checkout ([`editable-metadata.json`](editable-metadata.json)).
  - **The cause.** The editable install puts `src/` on the path, and the build leaves its
    `.egg-info` there, so `importlib.metadata` lists the project twice. It is one installation.
  - **It is not new.** M5's report at `b14c4de` shows the same line under A5
    (`../m5-acceptance-2026-10-24/m5_report.json`, `installed_more_than_once`); A5 records it as
    information and passed then too. The probe will stop counting the project's own build output as
    a second installation (BACKLOG 59).
- **The package lists do not name their environment.** `environment-before.json`,
  `environment-after.json` and `installed-packages.json` are byte-identical (one SHA-256). By their
  times `installed-packages.json` is the new `.venv`, written with the environment phase. The
  other two do not say whose list they are, and the existing checkout's lock gives the same names
  and versions. This record therefore states the existing checkout's state only from its commit and
  status files. It makes no claim about its environment or data beyond what the reviewer wrote.
- **The GPU after the run.** [`gpu-after.txt`](gpu-after.txt) is empty. The reviewer reads it as
  an empty list of GPU compute processes. The query that wrote it is not in the package.
- **The commands.** The reviewer's review describes the commands that ran. No script is kept, unlike
  M5's staged script. The deploy log, the probe's own JSON and the exit files are the raw evidence.

## 5. Files here

The whole evidence package, byte for byte. The reviewer packed it as
`batch1-acceptance-0bc5178-01-review.tar.gz` (13,182 bytes, SHA-256
`2e679c471a734efd134130ada65ef07edcecf04865d28af70027be5b710adae7`). It holds no repository,
environment, model or patient data, only homelab paths. Check it with `sha256sum -c SHA256SUMS`.
`SHA256SUMS` lists the package's 25 other files: not this README, nor `.gitattributes`.

| File | What it is |
|---|---|
| `CODEX_REVIEW.txt`, `review-verdict.json` | the reviewer's review (Chinese) and its machine-readable verdict |
| `preparation.txt`, `clone.log`, `checkout.log`, `final-state.txt` | the tested commit, uv and Python versions, the two input digests; the clone; the checkout's end state |
| `source-inputs.sha256`, `source-recheck.txt` | `pyproject.toml` and `uv.lock` digests, and their re-check after the install |
| `venv.log`, `venv-exit.txt` | the new environment |
| `deploy.log`, `deploy-exit.txt` | `deploy.sh`'s output and exit codes |
| `environment-check.json`, `environment-check.log`, `environment-check-exit.txt` | A1–A5's facts and console lines, the path checks and the wrapper's own checks; its exit codes |
| `installed-packages.json`, `environment-before.json`, `environment-after.json` | name and version lists (identical; §4) |
| `editable-metadata.json` | the two metadata entries of `shepherd-advanced` behind the duplicate (§4) |
| `pyg-kernels.log`, `pyg-kernels-exit.txt` | `validate_pyg_ext.py`'s output and exit codes |
| `gpu-before.txt`, `gpu-after.txt` | the GPU and driver before; the empty list after (§4) |
| `existing-checkout-before.txt`, `existing-checkout-after.txt` | the deployment checkout's commit and status |
| `SHA256SUMS` | the package's own digests |
| `.gitattributes` | keeps the files byte for byte in git |
