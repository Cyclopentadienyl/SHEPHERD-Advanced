# M5's functional acceptance of a clean deployment on the GB10, 2026-10-24

**Passed: the automated functional acceptance of a clean deployment.** A fresh clone, a fresh
environment from `deploy.sh`, the identified real workspace `hpo_2026_0929_5a` and an explicit
allocator preset, through the deployment probe's acceptance run, which carries M5's acceptance where it runs
headless together with what M2, M3a–M3c and M4's acceptances require. The reviewer reviewed the evidence
the same day. **The actual service configuration and the WebUI's manual acceptance are still to
be done**, so M5 is not closed. This is a functional acceptance: not a measure of model quality, a
capacity reading or an institutional test, and not a merge.

**The code tested is `b14c4def5082edae3ac948ce55ffbd770897c2df`.** The documentation commits after
it, this record included, were not tested again; the product code at `b14c4de` equals `87d189c`
apart from comments in `src/api/routes/evaluations.py`.

## 1. The run

| | |
|---|---|
| Commit | `b14c4def5082edae3ac948ce55ffbd770897c2df`, checked out detached in a fresh clone at `/home/user/Desktop/SHEPHERD-Advanced`; no tracked file modified (the report's `tested_commit`) |
| How it was run | A staged acceptance script outside the repository (SHA-256 `354bfac214f95679f1e37c56a1b3d1c4c1ecf93d3b351b0aca92bb8785913ca8`, reviewed by the reviewer before the run and kept in the evidence): init, clone, install, preset, envcheck, data, probe, after, package. Every stage passed, none failed |
| Install | `PYG_ARM_STRATEGY=pull ./deploy.sh` into a `.venv` made from the system Python 3.12; the deploy log read for the prebuilt PyG install and for no fallback |
| Environment | NVIDIA GB10 (capability 12.1), Linux aarch64, Python 3.12.3, torch `2.10.0+cu130`, Gradio 6.30.0, `torch_geometric` 2.8.1. The four PyG native extensions are the prebuilt release's: one native library of each matched the release wheel's recorded digest. The allocator was `cudaMallocAsync` from the saved preset, set before torch's first import |
| Lock (A5) | `uv.lock` is the one committed at the tested commit; no installed package that the lock lists differs from it; 23 packages come from the deploy scripts outside the lock. This does not mean every dependency is locked (BACKLOG 50) |
| Probe | `scripts/probe_deployment.py --m2-acceptance --device cuda`, every path absolute (BACKLOG 49): `--real-workspace` a copy of `hpo_2026_0929_5a` under the new data root `/home/user/shepherd-data/workspaces`, `--real-config` its `train_outputs/config.yaml`, `--real-reference-checkpoint` its `checkpoints/hgt/model-02-0.1813.pt`, `--workspaces-root` that data root; the work directory and the report outside the repository |
| Inputs | The copy compared byte for byte with the backup before the run. The five inputs (`config.yaml` `14bf8eea…970e`, `model-02-0.1813.pt` `33a7b39a…8b79`, `kg.provenance.json` `2c7abdb6…12ef`, `split_manifest.json` `131cdf81…f664`, `kg.json` `6cae2d1a…1e43`) equal M2 to M4's records before the run and again after it |
| Outcome | **99 probes passed**, 0 failed, 0 errors, 0 skipped. The required lists of the tested commit: demo 51 and real data 14, both complete, W1–W11 and RW1–RW3 among them. Probe exit 0, `tee` exit 0. The script's closing check, **102 of 102**, passed: checks of the run's record (the stages, exit codes, inputs, checkout, report fields), not product tests |
| Duration | The probes' own time is about 62 minutes, R3 and R4 (the real training and its resume) about 60 of them |
| Run by | The owner, with the reviewer's guidance; the reviewer reviewed the evidence on 2026-10-24 |

## 2. What it shows

- **A clean deployment works through the formal entry points.** The documented installer on a fresh
  clone gives an environment in which every probe M2, M3a–M3c, M4 and M5 require passes, on the
  demo workspace and on the real one.
- **M5's headless part.** The server's lists by name and digest (W1, RW1). D8 on the running
  service started as the launcher starts it, in M1 and read-only, with CORS and the route guard
  (W2–W6). The GPU lock refusing the runner and both CLIs naming its holder (W7). Test runs through
  the job runner, kept and then registered from the report store (W8, W9, RW2). A start for another
  model or cohort failing before measuring (W10). A training job holding the lock while the other
  jobs are refused, and stopped as stopped (W11, RW3). Both test services logged a normal shutdown.
- **On the real data**: a short training from scratch paused (R3) and resumed with its scaler (R4),
  then measured and served with the shortest-path table verified (R5, R6); M4's flow (RT1–RT3); the
  workspace unchanged at the end (R8: 19 entries by listing, size and modification time, not by
  content digest).
- **The result does not depend on where the data sits**: the workspace was copied to a new location
  and every digest the probe checks equals M4's.

## 3. What it does not show

- The real service configuration (`SHEPHERD_WORKSPACES_ROOT`, `SHEPHERD_REGISTRY`,
  `SHEPHERD_INTAKE_DIR`, `SHEPHERD_DEPLOYMENT_MODE=M1`) and the WebUI's manual acceptance: next, by
  the deployment guide's §4.11–§4.14. The layout and browser sessions are outside the probe.
- Any other dependency set: the evidence holds for the packages installed here. `torch_geometric`
  was 2.8.1, where M4's run had 2.8.0.post1, so M4's evidence does not carry over to this
  environment, and the PyG version is to be pinned before a formal release (BACKLOG 50).
- `deploy.sh`'s interactive PyG menu: the run used the non-interactive `pull` path.
- Model quality, capacity, an institutional test.

The main log's ERROR lines belong to the probes' injected faults (damaged files, another graph's
checkpoint, a missing dataset record), each refused as the probe requires. PyTorch's notice about the
GB10's capability and Starlette's TestClient deprecation warning are kept; no kernel, training,
resume, measurement or serving step failed. Three example commands at the end of the deploy log
print a literal `\033[0m`, a display issue.

## 4. Files here

| File | What it is |
|---|---|
| `m5_report.json` | The probe's report, byte for byte (it names no local path and no case id) |
| `exit.txt` | The probe's exit status |
| `exit-both.txt` | The probe's and its `tee`'s exit statuses |
| `acceptance-check.txt` | The script's closing check, 102 lines of results and its count |
| `SHA256SUMS` | The digests of the four files above |
| `.gitattributes` | Turns off the repository's line-ending normalisation here, so the files stay byte for byte |

The rest stays on the homelab in `/home/user/m5-acceptance-b14c4de-01/` and in two packages:

- the full evidence package, `m5-acceptance-b14c4de-01-handoff.tar.gz` (181,000,981 bytes,
  SHA-256 `18ea2dba6f286dc6fe8ce3400fb9ca5c09ab750c1b0b6216d3f53374cb894155`), on the homelab;
- the reduced review package, `m5-acceptance-b14c4de-01-review.tar.gz` (SHA-256
  `3a26b49f1853f2c74a362444959bf97716748ec77b5da8c94c9690a18b49735c`): 280 of the 284 regular files
  byte for byte, with the four real-training checkpoints (about 48 MB each) and the 12 symbolic links
  listed by path, size and SHA-256 in its `ARCHIVE_CONTENTS.json`, not included.

They hold the stage records (the script itself, the clone and install logs with their exit codes,
the deploy log, the package lists, the environment check, the data copy's comparison), the console
log (SHA-256 `bd94a81578bfcb2ca3b947e1b31180c5022406da6df2007b6543db9b8dc37e2e`), the service logs,
the cohort registries, reports, ranks and ledgers, and the run directories. Those name local paths or
carry per-case rows, which is why they are not copied here. The checkpoints' bytes stay only on the
homelab; their digests are in the packages.
