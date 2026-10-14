# S10 reload diagnostic, phase 2: inside one process, GB10, 2026-10-13

A diagnostic, not a reading. Phase 1 ([`reload-diag-2026-10-13/`](../reload-diag-2026-10-13/README.md))
saw each serving process grow by about 0.4–0.5 GB at every reload, at `90a668f` and `343e90e`
alike. This run looked inside one process at `343e90e` for what that growth is made of.

**Result.** The growth is reproduced in the simpler setting: +460.3 and +436.9 MB at reloads 2
and 3 (at the reload's response), +1204.9 MB after three reloads. **Every watched object of
every earlier pipeline was already gone** before any intervention, and a full `gc.collect()`
collected nothing and lowered RSS by 1.0 MB; `malloc_trim(0)` then lowered it by 68.9 MB, leaving
it 1134.9 MB above where it started. The root cause is **not located**: the earlier pipeline,
its graph, model and the representative tensors watched are not being held, and are not waiting
for a cycle collection, but memory the process kept is not explained by them, and the glibc
heap's own free space accounts for little of it.

Nothing here is accepted: not S10's capacity, not M2.1, not M2.

## 1. What was run

| | |
|---|---|
| Commit | `343e90e993d0a5129cd547b95276b7e2b8b12fa9`, no tracked file modified |
| Subject | checkpoint, graph and SP table digests the same as phase 1's and S10's |
| Tool | `s10_reload_inproc.py`, SHA-256 `9b79576809c04395b0bbf67180ff1cb93aa1da20db8d5ab5ab7152a8b61f000d` (kept outside the repository; tests `e47aec49…`; it imports phase 1's `s10_reload_diag.py`, `2d3cfc39…`) |
| Runbook | `RUNBOOK_S10_RELOAD_INPROC.md`, SHA-256 `1a962db56c3922aad2fb691afb69973bd2566984307b8e483e4dd26f44ef01ca` |
| Procedure | one process started through the service's own launcher (`--entry`, so its allocator and attention settings applied before torch was imported; recorded: `PYTORCH_ALLOC_CONF=backend:cudaMallocAsync`, source preset; `ATTENTION_ORDER=cudnn_sdpa,torch_sdpa,naive`), with the service's environment; startup through `initialize_pipeline()`; three reloads through the `reload_pipeline` route function; 5 diagnoses through the `diagnose` route function after startup and after each reload; no HTTP and no 200 requests. Weak references, after each publication, to the pipeline, its graph, model, largest embedding tensor, SP lookup and its two tensors, and largest graph tensor. At the end only: readings before any intervention, after `gc.collect()`, after `malloc_trim(0)` |
| Outcome | completed, exit 0; startup 81.0 s, reloads 81.1 / 82.7 / 82.5 s, all successful; 20 diagnoses accepted; 514 s in all |

Run by the owner; the reviewer reviewed the results the same day.

## 2. The readings (VmRSS, MB = 10^6 bytes)

| Point | VmRSS | step |
|---|---|---|
| after startup's 5 diagnoses | 6439.7 | |
| reload 1's response | 6793.1 | |
| after reload 1's diagnoses | 6731.1 | |
| reload 2's response | 7253.4 | +460.3 over reload 1's |
| after reload 2's diagnoses | 7196.6 | +465.5 |
| reload 3's response | 7690.3 | +436.9 |
| after reload 3's diagnoses | 7644.6 | +448.0 |
| before any intervention | 7644.6 | +1204.9 over startup |
| after `gc.collect()` (collected 0) | 7643.6 | −1.0 |
| after `malloc_trim(0)` (returned 1) | 7574.7 | −68.9; still +1134.9 over startup |

RssAnon moves with VmRSS; RssFile (412.7 MB) and RssShmem (50.8 MB) do not move. Each reload's
peak is about 28.5 GB (VmHWM, not reset in this run).

`mallinfo2` (glibc's own accounting) before and after the trim: `arena` 764.1 → 728.9 MB,
`fordblks` (free in the heap) 76.6 → 41.4 MB, `uordblks` (in use) 687.5 MB both, `hblkhd`
(blocks glibc mapped directly) 6.9 MB both. These are glibc's figures, not the process's: most of
the resident memory, including the 3.46 GB SP key tensor that the served pipeline holds, is not
in them.

## 3. What can and cannot be said

Can: in this setting, the growth phase 1 saw recurs; the watched objects of every earlier
pipeline (24 objects in generations 0–2) were gone at every point checked, before any
intervention, so they are neither held nor waiting for a cycle collection; a full collection and
a trim together return about 70 MB of about 1.2 GB.

Cannot: that all the memory of the earlier pipelines was freed (a tensor's storage can be shared
with another tensor, and native allocations are not covered), that no reference elsewhere keeps
something alive, that the remainder is or is not an allocator's keeping, or where in the build
it arises. `collected = 0` says nothing about cycles at other moments.

## 4. Next

**Superseded (2026-10-13):** the supplementary run and the investigation's conclusion are in
[`reload-inproc-segments-2026-10-13/`](../reload-inproc-segments-2026-10-13/README.md). What follows is the instruction as it stood.

The reviewer's convergence instruction (2026-10-13): first a read-only check of the two build
steps of the pipeline that are the likeliest places, `_precompute_node_embeddings` and
`_load_shortest_paths` with `build_sp_index` (the SP distance tensor is 431.9 MB, the same order
as the step; that is a lead, not a cause); then, if that finds no evidenced fix, at most one
supplementary run of the same tool at `343e90e` with readings on entry to and exit from those
two methods. A fix, when found, comes with a regression test that catches its cause and is
checked in this setting and then with the real HTTP reload. No forced collection, trim or
`empty_cache` in the product.

## 5. Files

As the owner's archive (`s10-reload-inproc-2026-10-13.tar.gz`, SHA-256
`4ca55b8c3e05341cb1fd11b48ddf96aa586c558da076d545ea466624cbe97f6e`, whose own `SHA256SUMS`
passed) holds them; `SHA256SUMS` here lists them. The tools, runbook and the reviewer's report
from that archive are cited by SHA-256 above rather than copied; the child's log stays on the
homelab.

| File | What |
|---|---|
| `inproc.json` | the diagnostic's document |
| `exit.txt` | its exit status, 0 |
