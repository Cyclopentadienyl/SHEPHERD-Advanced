# S10 capacity readings: M2.1 on the GB10, 2026-10-12

Contract M2.1 step S10 ([`PLAN_PROVENANCE_CONTRACT.md`](../../PLAN_PROVENANCE_CONTRACT.md), step
10). This directory holds the evidence of the one reported run at each commit, byte for byte as
the owner sent it, and this explanation.

**Status.** The run is complete. Every acceptance 1 check passes at the after commit (51 PASS, 0
FAIL, 0 UNKNOWN). Comparability: 60 checks PASS and 1 is UNKNOWN, the service state, so the
comparison's single verdict, which covers both, is "NOT judged: data missing" (exit status 2),
kept as recorded; the owner's statement below supplements the UNKNOWN and does not turn it into a
pass. Acceptance 2 and 3 are explained below, including the changes other than M2.1 on the
measured paths (§4.6), except one: the serving reload's small resident rise (R3 +0.023 / +0.031 /
+0.028 GB) is not located. Allocator retention is a candidate explanation; no diagnostic run was
made, and one reload per process cannot show whether it accumulates. Acceptance 4 is not
triggered (no shortfall). **S10's capacity is not accepted, and this is not acceptance of M2.1 or
M2.** The reviewer's evidence review (2026-10-13) found the readings consistent and recommends a
limited diagnostic of consecutive reloads before acceptance 2 is closed; the owner chose it
(2026-10-13, §6.1). Its driver was reviewed with no P1 or P2; its revised reading rules await the
reviewer's check; it has not run.

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
| Tools | draft 3, re-reviewed with no P1 or P2: `s10_probe.py` `04458fd7…`, `s10_sampler.py` `1ae557ad…`, `s10_run_matrix.sh` `6eaeffcf…`, `s10_compare.py` `7fbb86d5…`; serving script `measure_served_pipeline.py` `d097cf97…`, equal at both commits (`tools_sha256.txt`; `s10_compare.py`'s digest in `compare.txt`; full digests in the plan) |
| Times (UTC, from the files and `driver.log`) | pre-flight 13:44; before 13:44–15:42; after 15:53–17:37, on 2026-10-12 |
| Matrix | serving 3 repeats (seed 20260930); Mode C on the generated val cohort, `--device cuda` (`--modes C` at `90a668f` only); training `--conv-type hgt --hidden-dim 256 --num-layers 4 --epochs 3 --seed 42`, without and with `--resume`, stopped at `Trainer.train`; SP producer `--max-hops 5 --workers 15` into scratch |

**Inputs, equal at both commits and before and after each run** (`inputs_sha256_*_rows.txt`):
`kg.json` 209,928,843 B (`6cae2d1a…`, the subject's graph); `node_features.pt` 29,308,567;
`edge_indices.pt` 19,772,821; `train_samples.json` 12,867,342; `val_samples.json` 2,364,974;
`split_manifest.json` 1,970; `num_nodes.json` 52; `shortest_paths.pt` 10,797,575,893
(`dd506e73…`) with its sidecar; checkpoint 47,931,987 B (`33a7b39a…`).

**Pre-flight** (`preflight.json`): the checkpoint loads with `weights_only=True`; 8 keys; its
training roles are the two sample files, the two tensors, `num_nodes` and the split manifest (no
`kg`), as S10 expected.

**The page cache at R0 was not the same.** In every sampled row R0's page cache was larger at the
after commit, by 10.843, 10.830, 10.815 and 10.815 GB (measurement, training fresh, training
resume, SP; medians about 18.9 GB at `90a668f` and 29.7 GB at `343e90e`): about one SP table more,
plausibly the before run's own scratch publication still cached. The pre-read procedure and list
were the same; the system figures exclude reclaimable cache. Recorded as a limit on
comparability, not attributed: no speed-up is credited to the code for it, and it is not a reason
to rerun or to drop the cache (the warm decision stands).
The whole-machine R0 baselines differ by up to 0.5 GB between the runs (serving 10.70–10.99 GB
before, 11.08–11.51 GB after); the system figures are taken less their own R0.

**The owner's statement (2026-10-13, the owner's date).** The owner states that the SHEPHERD
backends were closed by hand before both runs. The evidence: both environment records show no
`shepherd` systemd user unit (`service_load_state not-found`) and 0 GPU compute processes. So
the runbook's `systemctl --user stop shepherd` had no unit to stop (inferred), and the comparison
reports the service state as UNKNOWN, as the runbook foresaw for a service run another way. The
UNKNOWN and the exit status 2 stay as recorded; the statement supplements them and does not turn
them into a pass.

## 2. Acceptance 1 (mechanical), from `compare.txt`

At the after commit every acceptance 1 check passes: all five rows of this run exited 0; every
probe boundary fired; training stopped at its boundary, the resume row with `--resume` and the fresh row
without, both on CUDA, with no epoch output and no checkpoint written; Mode C ran with
`cuda_executed` at this commit (15,000 cases ranked); the SP producer published into scratch a
sidecar naming graph `6cae2d1a…`, 431,902,937 pairs and `max_hops` 5, and a non-empty table
(10,797,575,893 bytes per `sp_table_bytes.txt`, a size the tool does not compare); serving
completed every phase of its three repeats with `readings_complete`; no row saw an OOM kill, a
swap increase or a swap rise during R0; no sampled row was stopped or left a process behind
(serving's server is ended by its own script, and the comparison does not check it for
leftovers); the workspace was unchanged; every reading was written by the recorded tools. The
before run passes the same checks.

Comparability: 60 checks pass (inputs, tools, pre-read list, environment, boot, the matrix's
settings, serving's allocator, the rows' explicit allocator being serving's, and each commit's
`--modes` form). The quiet-machine check is UNKNOWN (§1), so the comparison's verdict is "NOT
judged" (exit status 2).

## 3. How the readings are judged

- **Noise.** One sampled row per commit; serving three repeats per commit. Within a commit,
  serving's repeats spread by under 1 MB in R1 and in R2's VmHWM, by up to 3.8 MB in R2's VmRSS
  after the requests, and by 4.7–10.4 MB in R3 and R4, so a process reading of a few MB is within
  noise. Anonymous memory moves in 4 KiB pages and sometimes in 2 MiB huge-page steps (the
  measurement exit readings differ by exactly 2,097,152 B). VmHWM is the kernel's high-water mark,
  read at named boundaries.
- **System figures** (`system in use, sampled peak less R0`) are whole-machine and sampled: every
  process, the CUDA pool (GB10 memory is unified), kernel memory and page cache that is not
  reclaimable all count. They are reported, not attributed. In the SP row the sampler took about
  one sample a second (856 and 793 samples over about 870–890 s), not every 0.2 s, because each
  sample reads the 16 processes' `smaps_rollup`.
- **Release.** Whether a reader kept a buffer is shown by the readers' release checks (S2–S4,
  `tests/unit/test_shared_readers.py`, `tests/unit/test_graph_read_json.py`), not by RSS
  (acceptance 2). Those checks run the readers with their own options (for the checkpoint,
  `weights_only=True` on the CPU); the measured entry points call the same reader code with other
  `torch.load` options: measurement on CUDA with `weights_only=True`, training on CUDA with
  `weights_only=False`, serving on the CPU with `weights_only=False`. For those, release rests on
  the reader's code (the buffer is a local of `read_checkpoint`), not on a check — except that the
  reviewer ran the existing release check on the real checkpoint with serving's options
  (`map_location="cpu"`, `weights_only=False`) on 2026-10-13: it passes, one read, digest
  `33a7b39a…`. Measurement's and training's CUDA loads are not covered by a check. A release
  check answers only whether a reader's raw bytes outlive it; it says nothing about other objects
  a caller keeps, or about pages the allocator keeps after a free.
- **Wall times** are not acceptance quantities. Every row ran faster at the after commit: the
  measurement and training rows by 1.2–2.9 s, mostly before the first boundary; the SP producer by
  20.1 s, 3.3 s before the graph-read boundary and about 18.5 s in the BFS, whose code is
  identical; serving's R1, R2 and R3 by 1.9–13.0 s, 137–494 s and 1.8–14.1 s. The code explains a
  few tenths of a second (fewer page reads of `kg.json`, less hashing). Not investigated; the R0
  page-cache difference (§1) is one candidate.

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
  `artifact_digests` → `verify_generated_cohorts`, which parsed `train_samples.json` and parsed
  `val_samples.json` a second time after the model was built and the graph encoded, just before
  entry: VmHWM −
  VmRSS = 0.076 GB at entry. At `343e90e` those parses happen in `read_measurement_inputs`, before
  the model exists, below the later RSS: VmHWM − VmRSS = 0.002 GB. The row's peak is unchanged
  (+0.003 GB). That 0.095 GB is close to the checkpoint and the two tensors' sizes together
  (0.097 GB) is a coincidence: `90a668f` loaded those by path, and `343e90e`'s whole-file buffers
  for them come before CUDA's first use or during the checkpoint's own load, and set no peak.
- **Resident after the reads, −0.022 GB.** The objects alive at entry are the same at both commits
  (the graph tensors, 29.3 + 19.8 MB on the CPU; the val samples; the checkpoint, model and
  embeddings on CUDA); `343e90e` adds a few KB (the parsed split manifest, a 1,970-byte file,
  and the read identities). The difference is anonymous memory, most plausibly pages left
  resident after `90a668f`'s late parse; it closes by `run_mode_c`'s exit (+0.002 GB), inside code
  that differs between the commits only by two evidence fields R2 removed. A decrease: no added
  cost to assess.
- **End of process, +0.015 GB:** about 1% of RSS, below each of the three torch-file buffers
  (19.8–47.9 MB) though above the `train_samples.json` buffer (12.9 MB), after the peak; which freed allocations stayed resident at each commit is not established. Not a peak
  change.
- **CUDA** is identical to the byte at every boundary: Mode C's device work is unchanged.

### 4.2 Training (fresh and resume)

| Reading | Before | After | Δ |
|---|---|---|---|
| fresh: VmHWM = VmRSS at `Trainer.__init__` return and at the replaced `train` (acceptance 2) | 1.966 | 1.966 | < 1 MB |
| resume: VmRSS at the replaced `train` (acceptance 2) | 1.968 | 1.968 | +0.1 MB |
| resume: VmHWM at the replaced `train` (the row's peak) | 1.968 | 2.015 | **+0.047** |
| CUDA max allocated, each phase | 0.017; resume 0.065 | equal | 0 |
| System sampled peak less R0 | fresh 1.828; resume 1.938 | 1.877; 2.001 | +0.049; +0.063 |

- **Fresh: unchanged.** VmHWM equals VmRSS at `Trainer.__init__`'s return at both commits, so the
  new reads' transients (whole-file buffers of up to 29.3 MB at `343e90e`) come before the
  loaders and the model and set no peak.
- **Resume: +0.047 GB at the peak, resident equal: the size of one transient copy of the
  checkpoint** (47,931,987 B; VmHWM at the replaced `train` is 47,325,184 B higher at the after commit, and its
  rise from `Trainer.__init__`'s return is 48,939,008 B against 1,998,848 B, 46,940,160 B more). At `343e90e`
  `read_checkpoint` holds the file's bytes while `torch.load(BytesIO)` parses them; at `90a668f`
  `torch.load` read the records from the file by path. The buffer is released on return: VmRSS
  is equal, and the release rests on the reader's code (§3). The rise matches that buffer's size
  and the reader change; it is consistent with, not a proof of, exactly one copy. It is the
  plan's "about one transient copy" per file, "48 MB for the designated checkpoint" (Memory
  section): +2.4% of the row's peak, no resident cost.
- **The system figures** are whole-machine and sampled (26–42 samples a row): resume's +0.063 GB
  includes the checkpoint's transient; fresh's +0.049 GB, with the process figures equal, is
  reported and not attributed.

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
  and at the fork are equal within 0.05–0.23 MB (12–57 pages of 4 KiB).
- **The plan's kg.json expectation did not hold, because its premise was wrong.** The plan named
  `kg.json`'s bytes and decoded text, coexisting, as likely the largest new transient at serving
  and in the SP producer (about 0.21 GB). They are not new: at `90a668f` `load_json` called
  `json.load` on a text-mode file, and CPython's `TextIOWrapper.read()` reads the whole file into
  one bytes object and decodes it while that object is alive. `343e90e`'s `read_json` has the same
  shape (read, decode, release the bytes, parse, release the text). The transient is the same at
  both commits: at least twice the file's size (the bytes and the text, about 0.42 GB) for ASCII
  text, and 3, 4 or 6 times it if the text holds any non-ASCII, BMP or astral character, since
  `save_json` writes `ensure_ascii=False`. This is read from the code (and a CPU check of the
  same pattern), not a measurement of each decoder allocation. The plan's Memory section is
  corrected accordingly.
- **The later decreases come from code identical at both commits** (the BFS, the conversion to
  tensors, publication), apart from three more modules imported before publication at `343e90e`
  (`src/inference/pipeline.py`'s imports): a sub-MB rise in the opposite direction. The peak (−0.107 GB, matched by the tree's Pss) and the resident figure
  at publication (−0.29 GB, all anonymous) fall; which order-dependent allocations set them is not
  recorded. Decreases: no added cost to assess, and no diagnostic needed.
- **The system figure, +0.48 GB,** is whole-machine and sampled about once a second: 0.20 GB of it
  is a lower R0 at the after commit. The rest is not attributed: the system and tree peaks need not
  fall on the same sample, and the tree's own sampled peak fell.

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
  being off (`a535ae2`), if `GRADIO_ANALYTICS_ENABLED` was unset in the before run's server
  environment, which the evidence does not record; not established and not needed. The R1 system figure moves both ways by
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
- **Three questions, kept apart** (the reviewer's framing): (a) do a reader's raw bytes or its
  `BytesIO` outlive the reader; (b) does the service keep other Python objects, graph data,
  tensor storage or the old pipeline; (c) does the allocator keep resident pages after objects
  are freed.
- **(a): no raw-buffer retention found in the readers measured.** The release checks cover
  `kg.json`, the manifest, `num_nodes` and the two tensors with serving's options, and the
  reviewer's check covers the real checkpoint with serving's options (§3).
- **(b) and (c) are not answered.** The offset is about the same at the reload's peak and after
  the swap (the drop from peak to after is 21.660–21.661 GB at both commits) and absent at startup
  (R1, R2). Pages kept by the allocator would give that shape, but so would an object alive across
  both points; neither is excluded.
- **The source is not located.** Allocator retention is a candidate explanation, not a finding.
  **No diagnostic run was made.**
- **Not measured:** each server process does one reload, so whether the offset recurs or
  accumulates over several reloads in one process is unknown. The 0.4% holds for one reload.
- **Assessment (acceptance 2), not closed:** an added resident cost of about 0.03 GB after one
  reload, source not located. It does not by itself call for a reader rewrite, and acceptance 4
  rules out `malloc_trim` and the like. Whether it is investigated before acceptance is the
  owner's decision (§6.1).
- The R4 system figure is whole-machine noise (mean −0.001 GB). R3's wall time was 1.8–14 s shorter
  at the after commit; the code explains at most about 0.5 s of it; not investigated.

### 4.6 Changes other than M2.1 on the measured paths

S10 requires acceptance 2 and 3 to account for them. None is needed to explain a reading beyond
what is said here:

- **`290c03a`** (M3a: training's seed applied before the loaders and the model are built): moves
  a seed call; no allocation. The training rows are unchanged where the model is built.
- **`8eae614`** (HGT Edit 1: each HGT convolution receives its inputs in its own node- and
  edge-type order): no parameter is added; its work is on CUDA. Mode C's and training's CUDA
  counters are equal to the byte at every boundary.
- **`a535ae2`** (the dashboard's Gradio analytics off): only at serving's startup; the plausible
  cause of the 1–2 MB lower baseline (§4.4).
- **`e210b12`** (R2: Modes A and B deleted): the measurement CLI takes `--modes C` at `90a668f`
  only; it removes measurement's seeding and its Mode A/B code (well under 1 MB) and two of
  `run_mode_c`'s evidence fields; elsewhere on the measured paths it changed docstrings only.
- **`06fd0f1`** (item 8: `read_samples` refuses non-integer ids): an id check during the sample
  parse; CPU time only, no retained object.

## 5. Objects alive at each boundary (acceptance 3's list)

- **Measurement entry:** CPU — the graph tensors (`x_dict` 29.3 MB, `edge_index_dict` 19.8 MB),
  the val samples, the manifest record and the digests; CUDA — the checkpoint's tensors, the
  model, the encoded embeddings and the allocator's pool. At `90a668f` the cohort re-parse was
  freed moments before entry; at `343e90e` every file's raw bytes were freed as its reader
  returned, before the model existed.
- **Measurement after `run_mode_c` returns, where the row's peak is set** (VmHWM 1.600 → 1.608 GB
  at `90a668f`, 1.602 → 1.611 GB at `343e90e`): the entry objects plus the result (ranks and
  sample ids) and the JSON being written; the same code at both commits.
- **Training at the replaced `train`:** the graph tensors, the parsed train and val samples, the
  two loaders (not iterated), the model, optimizer, scheduler and scaler; with `--resume`, the
  restored state (CUDA tensors). The checkpoint's raw bytes (resume, `343e90e`) were alive only
  during its parse.
- **SP producer:** after the read, the `KnowledgeGraph` (the parsed tree freed), which stays alive
  in `main` until the process ends; at the fork, also the adjacency, the gene and disease maps and
  their sets, which `_init_worker_state` keeps in module globals in the parent for the rest of the
  run; at the peak, the graph, the adjacency, the BFS results as Python lists (including the last
  chunk's four lists, still bound to the loop variables) and their tensor conversion (about
  10.8 GB of tensors); at publication, the graph, the adjacency and the four tensors. How the
  53.6–53.7 GB peak splits between them is not recorded.
- **Serving at ready and after the reload:** the pipeline (graph, model, node embeddings, and the
  SP index of about 3.9 GB). At the startup peak, the SP table's columns and the index build's
  key, sort output and order; at the reload peak, the same plus the pipeline still being served,
  released only at the swap. The code drops the columns as they are folded into the key, so which
  of them coexist at the 23.8 GB and 28.5 GB marks (the sort's scratch, or columns still
  referenced) is not settled. That code is identical at both commits. At `343e90e` the parsed
  split manifest (a few KB) stays alive through the SP load.

## 6. Acceptance 2, 3 and 4

| Row | Resident after the reads (2) | Peaks (3) |
|---|---|---|
| Measurement | −0.022 GB at entry; converges by exit | +0.003 GB; CUDA equal; the entry peak's −0.095 GB is a phase move |
| Training | fresh < 1 MB; resume +0.1 MB | fresh unchanged; resume +0.047 GB, the size of one transient copy of the checkpoint, as planned |
| SP producer | +0.05 MB at the fork | −0.107 GB, identical code; no kg.json transient added (premise corrected) |
| Serving R1, R2 | −1 to −5 MB | −0.6 to −2 MB |
| Serving reload R3, R4 | **+0.023 / +0.031 / +0.028 GB** (R3; source not located; one reload per process) | +0.022 to +0.033 GB at the reload's peak (R4), about the same offset |

**Acceptance 4:** no row failed to complete, no OOM kill, no swap increase. Nothing was added to
make RSS fall.

### 6.1 Open for the owner

The serving reload's resident rise (R3 +0.023 / +0.031 / +0.028 GB) has no raw-buffer retention
behind it in the readers measured, but its source is not located (§4.5). Three choices:

**Chosen (the owner, 2026-10-13): choice 2.** Its driver, `s10_reload_diag.py` (SHA-256 `2d3cfc397b4867fe2cc876b6a58254967e64bec1f5771b695609ddfc963a41ff`) and its tests `test_s10_reload_diag.py` (`16f523b5b433719f4d0ac3b97717dcabdcaa66f856da9bf7cd528c277fc5931b`),
are kept outside the repository like S10's tools. The driver only combines
`measure_served_pipeline.py`'s own launcher, child environment, readiness rule, seeded workload,
HTTP client, VmHWM reset and stop, S10's pre-read, and reads of `/proc`. Its runbook fixes the
reading rules, with bands taken from S10's readings. The reviewer's review of draft 5
(2026-10-13) found no P1 or P2 in the driver and two P2 in those rules, fixed in the runbook
only; the revised rules await the reviewer's check, and the diagnostic has not run. Its results
will go in a directory of their own; this directory's files, `SHA256SUMS` and `compare.txt` do
not change. Choices 1 and 3 were not taken.

1. **Accept it uninvestigated.** Acceptance 2 calls a rise "an observation to investigate" and
   says "an investigation uses a separate diagnostic run"; choosing this is the owner's decision to
   accept the rise without one, recorded as such in the plan, with the source not located and the
   accumulation unmeasured.
2. **A limited diagnostic of consecutive reloads (the reviewer's recommendation, 2026-10-13).**
   It answers one question: does the 23–31 MB difference keep growing over several successful
   reloads in one process, and does the service stay healthy. Never a reported run, and not a
   rerun of the capacity matrix (no training, Mode C or SP producer):
   - one diagnostic at each commit, with the same workspace, checkpoint, service settings and
     normal allocator;
   - the same request history before the reloads as the reported readings (the 200 requests), or
     first evidence that a shorter history still reproduces the difference;
   - one server process per commit, reloaded three times in a row through the existing HTTP
     reload route (not three restarts with one reload each);
   - at fixed waits, RSS and, where available, anonymous memory and Pss after startup, after the
     requests and before and after each reload; each reload's success, readiness and diagnosis
     checked; both the step increments and the process's accumulated total reported;
   - its results kept apart as diagnostic evidence; S10's files, `SHA256SUMS` and `compare.txt`
     unchanged;
   - no `strace`, `mallinfo2`, `GLIBC_TUNABLES`, allocator change, `malloc_trim`, `gc.collect`,
     `empty_cache` or second model loader in this phase; a small driver, if needed, only combines
     the existing launcher, HTTP routes and readings, with its scope, source and SHA-256 fixed and
     reviewed before the run.

   Reading it: a small first difference that then levels off, with no growth in step with the
   reloads, supports accepting a bounded cost for this subject and three reloads, the source
   still not located; three reloads do not show that a long-running service never accumulates.
   Growth at every reload, or a health or availability problem, leads to a second phase that
   locates it (Python allocations and tensor lifetimes, then the native allocator). An ambiguous
   result is not written up as accumulation excluded; a bounded extension is decided then.
3. **The allocator tracing first** (`strace` of `mmap`/`munmap`/`mremap`/`brk`, `mallinfo2()`, a
   `GLIBC_TUNABLES` control). Deeper, and not guaranteed to name a buffer; the reviewer advises it
   only as a second phase after choice 2.

Other observations, optional and not needed for acceptance: the measurement row's −0.022 GB and
+0.015 GB; the SP row's −0.29 GB and −0.107 GB (decreases in identical code); the serving
baseline's 1–2 MB and R2's 54–59 MB fall during requests (equal at both commits); the wall times.

## 7. The evidence's limits

- `measure_served_pipeline.py` hashes the subject (`kg.json`, the checkpoint, the SP table) by
  path before the service starts, so the serving readings name the files designated, not the bytes
  the service read. It also parses `val_samples.json` and hashes it again by path, so the
  workload's recorded digest need not be that of the samples drawn (BACKLOG 19.8). The subject's
  fix is M3c's; the workload read's fix lands after S10, so that the script stays byte-identical
  at both commits.
- Not applicable, as the plan records: Modes A and B (deleted by R2, `e210b12`; `--modes C` at
  `90a668f` only), the calibration launcher (removed in S0), and serving's CUDA allocator counters
  (PLAN_B04 §7.3).
- Sampled figures can miss a transient shorter than the interval (0.2 s; about 1 s in the SP row).
- Serving records no CUDA counters (PLAN_B04 §7.3); the sampled rows do.
- The R0 page cache differed by about one SP table between the runs (§1).
- One reported run per commit for the sampled rows; three repeats for serving.
- This covers the designated subject only (decision 7): not other workspaces, checkpoints or
  deployments.

## 8. Files

`before/` and `after/` (25 files each), `preflight.json` and `compare.txt` are as the owner sent
them. The two `driver.log` files were missing from the first commit (`cfaa28a`) because the
repository ignores `*.log`; the reviewer found it from a clean tree (2026-10-13), and they were
added explicitly, unchanged, so that a clean checkout passes `SHA256SUMS` and reproduces
`compare.txt` (archive `s10-evidence-2.tar.gz`, SHA-256
`e58ef65993a0cbd89729f152d0a8d72fe1339a161b441f5507cfa178aa540279`); `SHA256SUMS` lists each
file. The run's scratch (server logs, the measurement report with paths, ranks, configurations
and the SP table) stayed on the homelab. Running `s10_compare.py` on these two directories with
`--after-commit 343e90e993d0a5129cd547b95276b7e2b8b12fa9` reproduces `compare.txt` byte for byte
(exit status 2).
