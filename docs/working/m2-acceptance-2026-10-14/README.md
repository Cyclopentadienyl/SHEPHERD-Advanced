# Contract M2's acceptance on the GB10, 2026-10-14

**Passed**, as the reviewer's review of the evidence found (2026-10-14): M2's source contract, on
the demo workspace and on real data, through the production entry points, at `82d7573`. It is a
functional acceptance of the contract. It is not a capacity reading under the deployment's
`cuda_async` allocator (§3), not a measure of model quality, and not a merge. The probe's allocator
start-up was fixed afterwards and confirmed on the GB10 at `b8a43c8` (§6).

## 1. The run

| | |
|---|---|
| Commit | `82d7573e1a11e8325456a340da719c7e11356bf0`, no tracked file modified (the report's `tested_commit`) |
| Command | `scripts/probe_deployment.py --m2-acceptance --device cuda` with `--real-workspace data/workspaces/hpo_2026_0929_5a`, `--real-config` that workspace's `train_outputs/config.yaml` (SHA-256 `14bf8eea14bc4fcdcbcbea1e2fb457346ba53d29bb1367bbb246e2d55607970e`) and `--real-reference-checkpoint` its `checkpoints/hgt/model-02-0.1813.pt` (SHA-256 `33a7b39a58519ff4974ed61266a5a35204edbb253d23c845340d0dcdf5eb8b79`); work directory and report outside the repository |
| Device | NVIDIA GB10, capability 12.1, 130.7 GB; torch 2.10.0+cu130 |
| Outcome | 62 passed, 0 failed, 0 errors, 0 skipped; exit 0. `m2_acceptance.complete` true, the demo and real-data parts each complete with nothing not passed |
| Run by | the owner; the reviewer reviewed the results the same day |

## 2. What it shows

**Demo part (phase M, after the writer, workspace, training and serving phases).**
- M1: a workspace valid on its own and of the same shape as the demo workspace, over another graph
  (`kg` and `node_features` differ).
- M2: a resume from the trained checkpoint trains: optimizer step 23 → 30, weights moved, the child
  (`e0e7e3e7…`) records its parent's digest and the same workspace roles.
- M3: the measurement entry point measures that child; its manifest names the child's digest.
- M4: the child serves with a shortest-path table the producer published and the loader verified
  (`gnn_plus_shortest_path`, `sp_kg_binding` verified), and a diagnosis runs.
- M5: resume refuses a parent from before the contract, from a same-shaped other graph, that read
  other samples, that validated when the run does not, and one that does not exist; each before
  anything is restored, with no run directory created and nothing changed.
- M6, M7: serving and measurement refuse the trained checkpoint over the same-shaped workspace.
- M8: the inventory lists those verdicts.

**Real-data part (phase R, on `hpo_2026_0929_5a`, read and never written).**
- R1: the workspace's graph and training inputs verify (graph `6cae2d1a…`; 8,990 training and
  1,587 validation diseases), and its table's sidecar names that graph.
- R2: the settings in effect carry `model-02`'s recorded architecture in all six fields compared
  (`hgt`, hidden 256, 4 layers, 8 heads, dropout 0.1, ortholog gate on).
- R3: one epoch from scratch: `330b8339…`, optimizer step 2,498, every workspace role recorded,
  `kg` included.
- R4: one epoch resumed from it: `d0d15dad…`, optimizer step 4,996, 140 of 162 weight tensors
  changed, the child records `resume_checkpoint` as its parent's digest.
- R5: the measurement entry point measures that child: 15,000 ranked, 0 absent, the manifest naming
  its digest. These are the workspace's own `val` cohort, not MyGene2 or the in-house test.
- R6: the serving build loads it on CUDA, `gnn_plus_shortest_path`, the table verified by the
  loader (`max_hops` 5), and a diagnosis returns 5 candidates.
- R7: the inventory refuses `model-02` for `kg is not recorded` and passes the new model for serving
  and measurement and for a resume. The other workspaces' refusals and unjudged entries in the
  deployment inventory are listed, not repaired; old artifacts are archived, not deleted.
- R8: the workspace and `model-02` are unchanged **by file listing, size and modification time**
  (19 entries); not by content digest.

**Checked beyond the PASS lines.** The reviewer recomputed the SHA-256 of both real checkpoints,
the real measurement report and both inventory reports, all equal to the report. The author
re-read, from the package, both measurement manifests (each names the resumed child), the digests
of the five reports the probe cites, and the two inventory verdicts above.

## 3. What it does not show

- **Not the deployment's allocator.** The probe ran the trainer as a module, which skipped
  `train_model.py`'s start-up allocator rule (applied only as a program); the runs' `runtime.json`
  record no allocator variable and `not applied (imported as a module)`, while the saved preset is
  `cuda_async`. By the code and PyTorch's default the run used the native backend; the backend was
  not queried inside the process. Its memory is not comparable with the earlier `cudaMallocAsync`
  readings, and no capacity conclusion is drawn from it.
- **Memory, as observed** (the reviewer's read-only snapshots; no continuous trace): during R3,
  `nvidia-smi` attributed 99,181 MiB (about 104 GB) to the probe, with no swap in use at that
  moment and batches advancing; after the run, MemAvailable 121,339,136 KiB (about 124.3 GB) and
  about 3.84 GB of swap in use. So it cannot be said that no swap was used throughout; the peak,
  its timing and its attribution are unknown; the difference is not attributed to fragmentation or
  shown to be a leak. The S10 reload-memory investigation is not reopened.
- **Not model quality.** The short-trained model shows the pipeline and its checks work; it does
  not replace the serving model, which the established training and evaluation choose.

## 4. Follow-up

- **The allocator start-up, fixed and confirmed (§6).** The probe applies the shared allocator rule
  before torch is imported and records the backend torch reports (`b8a43c8`, reviewed with no P1 or
  P2); a run of its existing demo phases at that commit on the GB10 used `cudaMallocAsync`. The two
  epochs were not rerun for that start-up change.
- **The homelab checkout is restored** to `main`, `90a668f`, as before the run (the owner's report,
  checked by the reviewer; the follow-up's own before and after records are in §6).

## 5. Files

As the owner's package holds them (archive `m2-acceptance-82d7573-evidence-2026-10-14.tar.gz`,
SHA-256 `3e7d4a60731c4d9de2defab3a2820b92f7fd19a0e44fd27087dfabf3379f30cf`, whose own `SHA256SUMS`
passed, 40 of 40); `SHA256SUMS` here lists them.

| File | What |
|---|---|
| `m2_report.json` | the probe's report: no paths, host or operator names by design |
| `exit.txt` | the exit statuses, both 0 |

The rest stays in the package and on the homelab, because its configs, logs and measurement
manifests carry local paths. Cited by digest: the real initial and resumed checkpoints
(`330b8339…`, 47,907,279 bytes; `d0d15dad…`, 47,908,943 bytes), the real measurement report
`da2e1caae1ab649b8afdab136fddab9360cca3dd6ba88527bcb961877cf62f1d`, the deployment inventory
`b7009d1d93087121563416f0a9c0985cf22e5df634d0dc3f95f9b163dc780521`, the new-model inventory
`cd5638f95551560876a1f5789ebc184370b8d91b3659b0acbd6123c9d8c09485`, the demo child `e0e7e3e7…` and
the demo measurement `3621b0f134485702b256297196cb4ebdfd55195adee9663139ac36abaf1ff64b`. No model
weights, tensors, sample lists or per-case ranks are in the repository.

## 6. The allocator follow-up on the GB10, at `b8a43c8` (2026-10-14)

A separate run, at a separate commit; §1–§5 and their files describe `82d7573` and are unchanged.

| | |
|---|---|
| Commit | `b8a43c80e3dbb683d349e24987474872e219aea9`, no tracked file modified |
| Command | `scripts/probe_deployment.py --device cuda --epochs 1 --keep` with a new work directory outside the repository: the existing phases A–E on the demo workspace only, no `--m2-acceptance`, nothing real read |
| Saved settings | `allocator_preset` `cuda_async`, `torch_compile` false; no allocator variable in the shell |
| Outcome | 42 passed, 0 failed, 0 errors, 0 skipped; exit 0. `resolved_device` cuda (NVIDIA GB10, torch 2.10.0+cu130) |
| Allocator | `PYTORCH_ALLOC_CONF` `backend:cudaMallocAsync`, `SHEPHERD_ALLOC_SOURCE` `preset`, source `saved preset`, set before torch was first imported; the backend torch reports, in the report's `allocator` and in A2, `cudaMallocAsync` |
| Checkout | `main`, `90a668f` before and after |

The demo training's `runtime.json` still reads `allocator_source: not applied (imported as a module)`:
that is `train_model.py`'s own start-up, which does not run when the trainer is imported. The same
file records the allocator variables the probe set, and the backend query shows they took effect.

What this shows and does not: the probe's entry point now runs CUDA under the saved allocator, and
a small demo trains, serves and reloads under it. It says nothing about a full HGT training's
memory under `cudaMallocAsync`, capacity, model quality or a clinical test; S10 is not redone.

Files, under `allocator-b8a43c8/`, as the owner's package holds them (archive
`m2-backend-b8a43c8-evidence-2026-10-14.tar.gz`, SHA-256
`de66d254fc7f46bd3402d72e6912304f568dc8d7f83f661ab26239fd0ee83bc8`, whose own `SHA256SUMS` passed,
14 of 14), listed in that directory's `SHA256SUMS`: `backend_report.json` (the probe's report, no
paths), `exit.txt`, `checkout-before.txt`, `checkout-restored.txt` and
`runtime-settings-before.json`. The training logs and outputs stay in the package.
