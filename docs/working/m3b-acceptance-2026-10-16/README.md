# Contract M3b's acceptance on the GB10, 2026-10-16

**Passed, within the scope agreed**, as the reviewer's review of the evidence found
(2026-10-16): the dataset record required by training, serving and measurement, on the demo
workspace and on the real workspace `hpo_2026_0929_5a`, through the production entry points, at
`c80c1e4`. It is a functional acceptance. It is not a capacity reading, an AMP study, a measure of
model quality or a hospital test, and not a merge. The plan's record is
[`PLAN_PROVENANCE_CONTRACT.md`](../PLAN_PROVENANCE_CONTRACT.md) §4, "M3b's acceptance on the
GB10".

## 1. The run

| | |
|---|---|
| Commit | `c80c1e421b3814716b1789a3b237c7dd0f0bb432`, no tracked file modified (the report's `tested_commit`). Its product code is the reviewed `04ecf42`: `git diff 04ecf42..c80c1e4 -- src scripts tests` is empty |
| Command | `scripts/probe_deployment.py --m2-acceptance --device cuda` with `--real-workspace data/workspaces/hpo_2026_0929_5a`, `--real-config` that workspace's `train_outputs/config.yaml` and `--real-reference-checkpoint` its `checkpoints/hgt/model-02-0.1813.pt`; work directory and report outside the repository |
| Inputs (SHA-256, before and after) | `config.yaml` `14bf8eea14bc4fcdcbcbea1e2fb457346ba53d29bb1367bbb246e2d55607970e`; `model-02-0.1813.pt` `33a7b39a58519ff4974ed61266a5a35204edbb253d23c845340d0dcdf5eb8b79`; `kg.provenance.json` `2c7abdb6660cbbbbcb7579e74797a9bfbcd775114b7b9b93c8de640175e412ef`; `split_manifest.json` `131cdf81cfa20ab8b44609be78473d6dce3f3a18d4436fafd589c6221ddcf664`; all four rechecked equal after the run |
| Device and allocator | NVIDIA GB10, resolved device `cuda`; backend `cudaMallocAsync` from the saved preset (`allocator_preset` `cuda_async`, `torch_compile` false), set before torch's first import |
| Outcome | 75 passed, 0 failed, 0 errors, 0 skipped; probe exit 0, `tee` exit 0. `m2_acceptance` demo, real data and overall complete |
| Checkout | before: `main` at `9fe5113`; restored afterwards to the same, the two records byte-identical, tracked files clean |
| Run by | the owner; the reviewer reviewed the evidence the same day |

## 2. What it shows

**M3b's own probes.**
- C1, M3, M4: the demo workspace's record verified with origin `synthetic`, and the measurement
  manifest and the served configuration carrying that same record.
- C5: six tamperings refused with the binding state each establishes: missing and unreadable →
  *unverifiable*; another workspace's record and another graph's record rebound → *mismatch*; the
  binding removed → *unrecorded*; a bound `files` record leaving an input unidentified → refused
  with its binding *verified* (the probe passes only if it is refused).
- B22, B23: the writer refusing a manifest without the record's binding, and a source list
  leaving a required input unidentified.
- M11: training, serving and measurement each refusing a workspace whose record is missing, with
  nothing written and the served pipeline unchanged.
- R1, R5, R6: the real record verified with origin `files`, all four roles, its SHA-256
  `2c7abdb6…12ef` in the workspace check, the measurement manifest and the served
  configuration; the reviewer read the raw measurement report and found its sources and
  `incomplete_by_design` equal to the real `kg.provenance.json`.

**The rest of the contract, on the same run:** the writer, workspace, training and serving phases;
M2's resume, measurement, serving and refusals (M1–M8) and M3a's pause, resume and extension
(M2, M9, M10); real data R2–R8, with a short training paused after one epoch and resumed for the
second, measured over the generated `val` cohort (15,000 samples) on CUDA, served as
`gnn_plus_shortest_path` with the table verified by the production loader (`max_hops` 5), the
older `model-02` refused by the inventory for its missing `kg` record and the new child accepted.

**Checkpoints (SHA-256), read again by the reviewer with the shared reader on the CPU:**
real parent `27d1fd4c6f69459d86d9fd93f2c1f8b95740d753181548bc2f87390ef7c06b9e` (one epoch,
paused), real child `4f8032198e5cf872476dd04548ea1b4ba8298dd7f4db476c7c73172a18d80c4f` (two
epochs, finished, naming its parent); the demo's parent and child as the report gives them.

## 3. Files here

| File | What it is |
|---|---|
| `m3b_report.json` | The probe's report, byte for byte (SHA-256 `9038a235907ff03b18f4a153bffc64f79b36657b9e8264dfecce58c6246ef489`) |
| `exit.txt` | The probe's exit status |
| `cross-checks.json` | The reviewer's 33 offline cross-checks and the checkpoints' recorded metadata; not a checkpoint |
| `SHA256SUMS` | The digests of the three files above |

The rest — the console log (SHA-256 `8e73f7143ee8d77a572cb4568eee0b785e1828d2e31823aeded1eca8a60dfc8e`),
the run directories, measurements and inventories, the 40 checkpoint and tensor files — stays in
`/home/user/m3b-acceptance-c80c1e4-01/` on the homelab, with an index of every file's size and
SHA-256 in the reviewer's evidence package. Some of those files name local paths, which is why
they are not copied here.

## 4. How to read it

- The four ERROR lines in the console are the injected refusals: E5's damaged tensor (two lines),
  M6's crossed checkpoint and M11.serving's missing record.
- The GB10's capability 12.1, above the 12.0 the wheel lists, is reported as a runtime notice. The
  run's training, measurement and serving completed on CUDA; that does not establish optimal
  native support for every operation, and no compatibility or performance study is opened here.
- R8 compares 19 entries of the real workspace by listing, size and modification time. It is not a
  content hash of the workspace.
- The measurement is over the generated `val` cohort. Its numbers are not a MyGene2 or
  institutional held-out result, and the short-training checkpoints are not a chosen serving
  model.
- N2, M3c and M4 are not completed by this run.
