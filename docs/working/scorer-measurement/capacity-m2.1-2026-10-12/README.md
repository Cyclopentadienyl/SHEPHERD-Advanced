# S10 capacity readings: M2.1 on the GB10, 2026-10-12

Contract M2.1 step S10 ([`PLAN_PROVENANCE_CONTRACT.md`](../../PLAN_PROVENANCE_CONTRACT.md), step
10). This directory holds the evidence of the one reported run at each commit, byte for byte as
the owner sent it, and this explanation.

**Status.** The run is complete. Acceptance 1's mechanical part is met at the after commit (51
checks PASS, none FAIL). Comparability: 60 checks PASS and 1 UNKNOWN, the service state, which the
owner's statement below supplements; the comparison's exit status 2 is kept as recorded.
Acceptance 2 and 3 are explained below; acceptance 4 is not triggered (no shortfall). **Awaiting
the reviewer's review: S10's capacity is not accepted until that review closes, and this is not
acceptance of M2.1 or M2.** One question is left to the owner (§6.1): whether the serving reload's
small resident rise needs a separate diagnostic run before acceptance.

## 1. What was run

| | |
|---|---|
| Subject | workspace `hpo_2026_0929_5a`, checkpoint `model-02-0.1813.pt` (S10, decision 7) |
| Before | `90a668f1334dfe6f73637dd8f11659ea906b9e1d` (M1's merge) |
| After | `343e90e993d0a5129cd547b95276b7e2b8b12fa9` (`src` and `scripts` those of S9's `f481882`) |
| Machine | NVIDIA GB10, aarch64, kernel `7.0.0-1019-nvidia`, driver `580.178.04`; one boot for both runs (`boot_id 88899c2c…`) |
| Software | Python 3.12.3; torch 2.10.0+cu130 (CUDA 13.0); one venv (python `c86f21d0…`, distributions `a117e53e…`), equal at both commits |
| Allocator | `PYTORCH_ALLOC_CONF=backend:cudaMallocAsync`, the saved preset, set explicitly for the sampled rows; serving applied the same value itself |
| Page cache | warm, by the owner's decision: before each row the same 10 workspace files (`preread.txt`) and the checkpoint were read once, and R0 was taken after that. A capacity comparison after a pre-read, not a cold start |
| Tools | draft 3, re-reviewed with no P1 or P2: `s10_probe.py` `04458fd7…`, `s10_sampler.py` `1ae557ad…`, `s10_run_matrix.sh` `6eaeffcf…`, `s10_compare.py` `7fbb86d5…`; serving script `measure_served_pipeline.py` `d097cf97…`, equal at both commits (`tools_sha256.txt`, full digests in the plan) |
| Times (homelab, local) | pre-flight 13:44; before 13:44–15:42; after 15:53–17:37 |
| Matrix | serving 3 repeats (seed 20260930); Mode C on the generated val cohort, `--device cuda` (`--modes C` at `90a668f` only); training `--conv-type hgt --hidden-dim 256 --num-layers 4 --epochs 3 --seed 42`, without and with `--resume`, stopped at `Trainer.train`; SP producer `--max-hops 5 --workers 15` into scratch |

**Inputs, equal at both commits and before and after each run** (`inputs_sha256_*_rows.txt`):
`kg.json` 209,928,843 B (`6cae2d1a…`, the subject's graph); `node_features.pt` 29,308,567;
`edge_indices.pt` 19,772,821; `train_samples.json` 12,867,342; `val_samples.json` 2,364,974;
`split_manifest.json` 1,970; `num_nodes.json` 52; `shortest_paths.pt` 10,797,575,893
(`dd506e73…`) with its sidecar; checkpoint 47,931,987 B (`33a7b39a…`).

**Pre-flight** (`preflight.json`): the checkpoint loads with `weights_only=True`; 8 keys; its
training roles are the two sample files, the two tensors, `num_nodes` and the split manifest (no
`kg`), as S10 expected.

**The owner's statement (2026-10-13).** Before both runs the SHEPHERD backends were closed. The
environment records 0 GPU compute processes at both commits. The comparison reports the service
state as UNKNOWN because the homelab has no `shepherd` systemd user unit (`LoadState not-found`):
the `systemctl --user stop shepherd` step of the runbook did nothing there, and the backends were
closed by hand. The UNKNOWN and the exit status 2 stay as recorded; the statement supplements
them and does not turn them into a pass.

## 2. Acceptance 1 (mechanical), from `compare.txt`

At the after commit every check passes: all five rows of this run exited 0; every probe boundary
fired; training stopped at its boundary, the resume row with `--resume` and the fresh row
without, both on CUDA, with no epoch output and no checkpoint written; Mode C ran with
`cuda_executed` at this commit (15,000 cases ranked); the SP producer published into scratch a
sidecar naming graph `6cae2d1a…`, 431,902,937 pairs and `max_hops` 5, and a table of
10,797,575,893 bytes; serving completed every phase of its three repeats with `readings_complete`;
no row saw an OOM kill, a swap increase or a swap rise during R0; no row was stopped or left a
process behind; the workspace was unchanged; every reading was written by the recorded tools. The
before run passes the same checks. The two runs are comparable: inputs, tools, pre-read list,
environment, boot, the matrix's settings and serving's allocator are equal, and the rows'
explicit allocator is serving's.

## 3. How the readings are judged

- **Noise.** One sampled row per commit; serving three repeats per commit. Serving's repeats
  spread by up to 1–2 MB (R1, R2) and 5–11 MB (R3, R4) within a commit, so a process reading of a
  few MB is within noise. Anonymous memory moves in 2 MiB steps (huge pages). VmHWM is the
  kernel's high-water mark, read at named boundaries.
- **System figures** (`system in use, sampled peak less R0`) are whole-machine and sampled: every
  process, the CUDA pool (GB10 memory is unified), kernel memory and page cache that is not
  reclaimable all count. They are reported, not attributed. In the SP row the sampler took about
  one sample a second (856 and 793 samples over about 870–890 s), not every 0.2 s, because each
  sample reads the 16 processes' `smaps_rollup`.
- **Release.** Whether a reader kept a buffer is shown by the readers' release checks (S2–S4,
  `tests/unit/test_shared_readers.py`, `tests/unit/test_graph_read_json.py`), not by RSS
  (acceptance 2). Those checks run the readers with their own options (for the checkpoint,
  `weights_only=True` on the CPU); the measured entry points call the same reader code with other
  `torch.load` options (CUDA, or `weights_only=False` in training).
- **Wall times** are not acceptance quantities. Every row ran 1–3 s faster at the after commit,
  much of it before the first boundary; the code explains a few tenths of a second (fewer page
  reads of `kg.json`, less hashing). Not investigated.

## 4. The readings, explained

Figures are GB (10^9 B) as printed by the comparison; after − before.

### 4.1 Measurement (Mode C)

| Reading | Before | After | Δ |
|---|---|---|---|
| VmHWM at `run_mode_c` entry | 1.465 | 1.370 | −0.095 |
| VmRSS at `run_mode_c` entry (acceptance 2) | 1.389 | 1.368 | −0.022 |
| VmHWM, max over the readings (the row's peak) | 1.608 | 1.611 | +0.003 |
| CUDA max allocated / reserved, each phase | 6.043 / 7.449; 0.196 / 7.449; 0.143 / 7.449 | equal | 0 |
| VmRSS at exit / end of process | 1.600 / 1.587 | 1.602 / 1.602 | +0.002 / +0.015 |
| System sampled peak less R0 | 9.233 | 9.101 | −0.132 |

- **The entry peak, −0.095 GB, is a phase move.** The probe reads at `run_mode_c`'s start, after
  its arguments are evaluated. At `90a668f` the `manifest` argument ran `build_manifest` →
  `artifact_digests` → `verify_generated_cohorts`, which re-parsed `train_samples.json` and
  `val_samples.json` after the model was built and the graph encoded, just before entry: VmHWM −
  VmRSS = 0.076 GB at entry. At `343e90e` those parses happen in `read_measurement_inputs`, before
  the model exists, below the later RSS: VmHWM − VmRSS = 0.002 GB. The row's peak is unchanged
  (+0.003 GB). That 0.095 GB is close to the checkpoint and the two tensors' sizes together
  (0.097 GB) is a coincidence: `90a668f` loaded those by path, and `343e90e`'s whole-file buffers
  for them come before CUDA's first use or during the checkpoint's own load, and set no peak.
- **Resident after the reads, −0.022 GB.** The objects alive at entry are the same at both commits
  (the graph tensors, 29.3 + 19.8 MB on the CPU; the val samples; the checkpoint, model and
  embeddings on CUDA); `343e90e` adds identities of a few bytes. The difference is anonymous
  memory, most plausibly pages left resident after `90a668f`'s late parse; it closes by
  `run_mode_c`'s exit (+0.002 GB), inside code identical at both commits. A decrease: no added
  cost to assess.
- **End of process, +0.015 GB:** about 1% of RSS, below any single transient copy, after the
  peak; which freed allocations stayed resident at each commit is not established. Not a peak
  change.
- **CUDA** is identical to the byte at every boundary: Mode C's device work is unchanged.

### 4.2 Training (fresh and resume)

| Reading | Before | After | Δ |
|---|---|---|---|
| fresh: VmHWM = VmRSS at `Trainer.__init__` return and at the replaced `train` (acceptance 2) | 1.966 | 1.966 | < 1 MB |
| resume: VmRSS at the replaced `train` (acceptance 2) | 1.968 | 1.968 | +0.1 MB |
| resume: VmHWM at the replaced `train` (the row's peak) | 1.968 | 2.015 | **+0.047** |
| CUDA max allocated, each phase | 0.017; resume 0.065 | equal | 0 |

- **Fresh: unchanged.** VmHWM equals VmRSS at `Trainer.__init__`'s return at both commits, so the
  new reads' transients (whole-file buffers of up to 29.3 MB at `343e90e`) come before the
  loaders and the model and set no peak.
- **Resume: +0.047 GB at the peak, resident equal. This is one transient copy of the checkpoint**
  (47,931,987 B; the window's rise is 47,325,184 B more at the after commit). At `343e90e`
  `read_checkpoint` holds the file's bytes while `torch.load(BytesIO)` parses them; at `90a668f`
  `torch.load` read the records from the file by path. The buffer is released on return: VmRSS
  is equal, and the release check covers the reader. This is the plan's "about 48 MB for the
  designated checkpoint, one transient copy": +2.4% of the row's peak, no resident cost.

### 4.3 SP producer

| Reading | Before | After | Δ |
|---|---|---|---|
| VmHWM / VmRSS after the graph read | 1.601 / 1.596 | 1.601 / 1.596 | +0.2 MB / +0.05 MB |
| VmRSS at the first worker fork (acceptance 2) | 1.612 | 1.612 | +0.05 MB |
| VmHWM, the row's peak (BFS and tensor assembly) | 53.737 | 53.629 | −0.107 |
| Tree Pss, sampled peak | 53.727 | 53.621 | −0.106 |
| VmRSS at publication / end of process | 12.805 / 1.314 | 12.513 / 1.023 | −0.292 / −0.291 |
| System sampled peak less R0 | 53.026 | 53.506 | +0.480 |

- **Where M2.1 acts, nothing changed.** S3 moved the graph read into `KnowledgeGraph.read_json`
  (one read; bytes, then text, then the parsed tree). The resident and peak figures after the read
  and at the fork are equal within a page or two.
- **The plan's kg.json expectation did not hold, because its premise was wrong.** The plan named
  `kg.json`'s bytes and decoded text, coexisting, as likely the largest new transient at serving
  and in the SP producer (about 0.21 GB). They are not new: at `90a668f` `load_json` called
  `json.load` on a text-mode file, and CPython's `TextIOWrapper.read()` reads the whole file into
  one bytes object and decodes it while that object is alive. `343e90e`'s `read_json` has the same
  shape (read, decode, release the bytes, parse, release the text). The transient is the same at
  both commits (at least the file's size; more if the text holds characters beyond Latin-1, since
  `save_json` writes `ensure_ascii=False`). The plan's Memory section is corrected accordingly.
- **The later decreases come from code identical at both commits** (the BFS, the conversion to
  tensors, publication). The peak (−0.107 GB, matched by the tree's Pss) and the resident figure
  at publication (−0.29 GB, all anonymous) fall; which order-dependent allocations set them is not
  recorded. Decreases: no added cost to assess, and no diagnostic needed.
- **The system figure, +0.48 GB,** is whole-machine and sampled about once a second: 0.20 GB of it
  is a lower R0 at the after commit, and the rest lies outside the process tree, whose own peak
  fell.

### 4.4 Serving: launch to ready (R1) and 200 requests (R2)

| Reading (repeats 1 / 2 / 3) | Before | After | Δ |
|---|---|---|---|
| R1 VmHWM at ready (the startup peak) | 23.773 / 23.773 / 23.772 | 23.771 / 23.771 / 23.772 | −0.002 / −0.001 / −0.0006 |
| R1 VmRSS at ready (acceptance 2) | 6.505 / 6.506 / 6.505 | 6.504 / 6.504 / 6.503 | −0.001 / −0.002 / −0.002 |
| R2 VmRSS after 200 requests | 6.447 / 6.447 / 6.451 | 6.446 / 6.445 / 6.446 | −0.001 / −0.002 / −0.005 |
| R1 system sampled peak less R0 | 30.974 / 30.790 / 30.820 | 30.726 / 31.106 / 30.574 | −0.25 / +0.32 / −0.25 |

- **Unchanged within noise.** The startup peak is set in the SP-table phase (the 10.8 GB table
  and the index build), whose code is identical at both commits. S6's new transients — whole-file
  buffers for `node_features.pt` (29.3 MB), `edge_indices.pt` (19.8 MB) and the checkpoint
  (47.9 MB), each released before the next read — come earlier, about 21 GB below that peak, and
  leave nothing at ready: S6 keeps only identities. The `kg.json` transient is not new (§4.3).
- The 1–2 MB lower resident baseline at the after commit is plausibly the dashboard's analytics
  being off (`a535ae2`); not established and not needed. The R1 system figure moves both ways by
  up to 0.3 GB against a within-commit range of up to 0.53 GB: noise.
- These figures reproduce the 2026-09-30 context readings (R1 VmHWM 23.77 GB, R2 after 6.45 GB;
  [`PLAN_B04_PRODUCTIONISATION.md`](../PLAN_B04_PRODUCTIONISATION.md)).

### 4.5 Serving: one reload (R3, R4) — the one resident rise

| Reading (repeats 1 / 2 / 3) | Before | After | Δ |
|---|---|---|---|
| R3 VmRSS after the reload (acceptance 2) | 6.793 / 6.789 / 6.788 | 6.816 / 6.820 / 6.815 | **+0.023 / +0.031 / +0.028** |
| R4 VmHWM over the reload | 28.455 / 28.449 / 28.448 | 28.477 / 28.481 / 28.475 | +0.022 / +0.032 / +0.027 |
| R4 VmHWM increase over the reload | 22.008 / 22.002 / 21.997 | 22.031 / 22.035 / 22.029 | +0.024 / +0.033 / +0.032 |
| R4 system sampled peak increase | 22.118 / 21.960 / 22.292 | 21.969 / 22.179 / 22.218 | −0.15 / +0.22 / −0.07 |

- **The rise is real and small:** the same sign in every repeat, outside the 5–11 MB spread within
  a commit, about 0.4% of RSS after a reload and 0.1% of the reload's 28.5 GB peak. The before
  values reproduce the 2026-09-30 readings (R4 over the reload 28.45 GB).
- **What the reload reads.** The reload route and the pipeline swap are identical at both commits.
  At `90a668f` it parsed `kg.json` by a text-mode `json.load`, hashed it and the tensors by path,
  and loaded the tensors and the checkpoint by path. At `343e90e` `kg.json`, the manifest,
  `num_nodes`, the two tensors and the checkpoint are each read once and parsed from that read;
  the SP table is loaded by path at both. The new transients are the same three whole-file
  buffers as at startup.
- **Not a kept buffer:** the release checks cover every reader on the reload path. **Not a larger
  live set:** the same build code leaves R1 and R2 unchanged at startup, and within each repeat the
  offset is the same at the reload's peak and after the swap — the drop from peak to after is
  21.660–21.662 GB at both commits — so it is taken before the peak and stays.
- **What it is, inferred and not investigated:** memory the C allocator keeps after the reload's
  transient buffers are freed (for example a freed buffer's region kept in the heap, or a raised
  trim threshold), rather than a Python object. Which buffer, and why the same reads leave no such
  offset at startup, the code does not settle. **No diagnostic run was made.**
- **Not measured:** each server process does one reload, so whether the offset recurs or
  accumulates over several reloads in one process is unknown. The 0.4% holds for one reload.
- **Assessment (acceptance 2):** an added resident cost of about 0.03 GB after a reload, with no
  kept buffer and no larger live set. It does not by itself call for a reader rewrite, and
  acceptance 4 rules out `malloc_trim` and the like. Whether "an observation to investigate"
  needs the diagnostic run before acceptance is the owner's decision (§6.1).
- The R4 system figure is whole-machine noise (mean −0.001 GB). R3's wall time was 1.8–14 s shorter
  at the after commit; the code explains at most about 0.5 s of it; not investigated.

## 5. Objects alive at each boundary (acceptance 3's list)

- **Measurement entry:** CPU — the graph tensors (`x_dict` 29.3 MB, `edge_index_dict` 19.8 MB),
  the val samples, the manifest record and the digests; CUDA — the checkpoint's tensors, the
  model, the encoded embeddings and the allocator's pool. At `90a668f` the cohort re-parse was
  freed moments before entry; at `343e90e` every file's raw bytes were freed as its reader
  returned, before the model existed.
- **Training at the replaced `train`:** the graph tensors, the parsed train and val samples, the
  two loaders (not iterated), the model, optimizer, scheduler and scaler; with `--resume`, the
  restored state (CUDA tensors). The checkpoint's raw bytes (resume, `343e90e`) were alive only
  during its parse.
- **SP producer:** after the read, the `KnowledgeGraph` (the parsed tree freed); at the fork, the
  adjacency and the BFS inputs; at the peak, the BFS results as Python lists and their tensor
  conversion (about 10.8 GB of tensors); at publication, the four tensors. How the 53.6–53.7 GB peak
  splits between them is not recorded.
- **Serving at ready and after the reload:** the pipeline (graph, model, node embeddings, the SP
  index, about 3.9 GB). At the startup and reload peaks, the SP table's columns and the index
  build's key, sort output and order; the code drops the columns as they are folded into the key,
  so which of them coexist at the 23.8 GB and 28.5 GB marks (the sort's scratch, or columns still
  referenced) is not settled. That code is identical at both commits. At `343e90e` the parsed
  split manifest (a few KB) stays alive through the SP load.

## 6. Acceptance 2, 3 and 4

| Row | Resident after the reads (2) | Peaks (3) |
|---|---|---|
| Measurement | −0.022 GB at entry; converges by exit | +0.003 GB; CUDA equal; the entry peak's −0.095 GB is a phase move |
| Training | fresh < 1 MB; resume +0.1 MB | fresh unchanged; resume +0.047 GB = one transient copy of the checkpoint, as planned |
| SP producer | +0.05 MB at the fork | −0.107 GB, identical code; no kg.json transient added (premise corrected) |
| Serving R1, R2 | −1 to −5 MB | −0.6 to −2 MB |
| Serving reload R3, R4 | **+0.023 to +0.031 GB** (inferred allocator-level; one reload) | +0.022 to +0.033 GB, the same offset |

**Acceptance 4:** no row failed to complete, no OOM kill, no swap increase. Nothing was added to
make RSS fall.

### 6.1 Open for the owner

The serving reload's +0.03 GB is explained in kind (allocator-level, not a kept buffer, not a
larger live set) but by inference. Two choices:

1. **Record it as explained and assessed** (this README), with the attribution marked as inferred
   and the accumulation over repeated reloads as unmeasured; or
2. **Run a separate diagnostic** before acceptance, never a reported run: at both commits, startup
   and three reloads in one server process, with `strace -e trace=mmap,munmap,brk` filtered to the
   29,308,567 / 19,772,821 / 47,931,987-byte requests, glibc `mallinfo2()` at the reader
   boundaries, and a control with `GLIBC_TUNABLES=glibc.malloc.mmap_threshold=4194304`. It would
   name the allocation and show whether the offset accumulates. It needs the homelab for about an
   hour and a small instrument outside the repository.

Other observations, optional and not needed for acceptance: the measurement row's −0.022 GB and
+0.015 GB; the SP row's −0.29 GB and −0.107 GB (decreases in identical code); the serving
baseline's 1–2 MB and R2's 54–59 MB fall during requests (equal at both commits); the wall times.

## 7. The evidence's limits

- `measure_served_pipeline.py` hashes the subject by path before the service starts, so the
  serving readings name the files designated, not the bytes the service read (BACKLOG 19.8). Its
  fix lands after S10 so that the script stays equal at both commits.
- Sampled figures can miss a transient shorter than the interval (0.2 s; about 1 s in the SP row).
- Serving records no CUDA counters (PLAN_B04 §7.3); the sampled rows do.
- One reported run per commit for the sampled rows; three repeats for serving.
- This covers the designated subject only (decision 7): not other workspaces, checkpoints or
  deployments.

## 8. Files

`before/` and `after/` (25 files each), `preflight.json` and `compare.txt` are as the owner sent
them (archive `s10-evidence-2.tar.gz`, SHA-256
`e58ef65993a0cbd89729f152d0a8d72fe1339a161b441f5507cfa178aa540279`); `SHA256SUMS` lists each
file. The run's scratch (server logs, the measurement report with paths, ranks, configurations
and the SP table) stayed on the homelab. Running `s10_compare.py` on these two directories with
`--after-commit 343e90e993d0a5129cd547b95276b7e2b8b12fa9` reproduces `compare.txt` byte for byte
(exit status 2).
