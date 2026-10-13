# S10 reload diagnostic, phase 2 with segments, and the reload-memory conclusion, GB10, 2026-10-13

The last run of the reload-memory investigation, and its closing record. Earlier:
[`reload-diag-2026-10-13/`](../reload-diag-2026-10-13/README.md) (phase 1, HTTP, both commits) and
[`reload-inproc-2026-10-13/`](../reload-inproc-2026-10-13/README.md) (phase 2, in process,
`343e90e`).

**Conclusion (the owner's decision, 2026-10-13).** The PyTorch wheel in use is confirmed to
contain mimalloc purge defects that upstream has since fixed; they are a strong candidate cause
of the observed memory retention; the attribution of all of the increase is not established.
This is **not** a confirmed root cause, nothing is fixed, and other factors are not excluded. The
reload-memory investigation ends here: no causal control run, no change to the product pipeline,
no `MIMALLOC_*` setting, no profiler or new diagnostic tool. The evidence and the dependency
limitation are kept (BACKLOG item 22). The growth common to both commits and M2.1's ~30 MB
difference are recorded apart; S10's capacity is judged on its own readings and limits, and
finding the upstream defect accepts nothing by itself.

## 1. This run

| | |
|---|---|
| Commit | `343e90e993d0a5129cd547b95276b7e2b8b12fa9`, no tracked file modified |
| Tool | `s10_reload_inproc.py` draft 3, SHA-256 `406a7b527f8d54ff24583b164e1fa0a758f7da2aef8df00763446f25a24a3741` (kept outside the repository; tests `85ac1f4f…`; reviewed before the run) |
| Runbook | SHA-256 `87b4e97a4759751aa1a3076a192dbc151e478d4de8bb87d6eb54619059c10eaf` |
| Procedure | phase 2's (the service's launcher, `initialize_pipeline`, three `reload_pipeline` calls, 5 `diagnose` calls after each), plus readings on entry to and exit from `_precompute_node_embeddings` and `_load_shortest_paths`, with glibc's `mallinfo2`, the mapping size distribution from `/proc/self/smaps` and the device's free memory at every reading |
| Outcome | completed, exit 0; startup and three reloads successful; 20 diagnoses accepted; 16 segment readings; about 524 s |

Run by the owner; the reviewer reviewed the results the same day.

## 2. It did not repeat phase 2's steady growth

VmRSS, MB = 10^6 bytes, after each reload's five diagnoses, over the level before reload 1:

| Run | reload 1 | reload 2 | reload 3 |
|---|---|---|---|
| phase 1, HTTP, `90a668f` | +258.8 | +728.1 | +1194.9 |
| phase 1, HTTP, `343e90e` | +293.2 | +756.8 | +1231.4 |
| phase 2, in process, `343e90e` | +291.4 | +756.9 | +1204.9 |
| **this run**, in process, `343e90e` | +287.6 | +321.6 | +266.8 |

During reload 3's five diagnoses, before any collection or trim, RSS fell by 476.8 MB, all of it
in mappings larger than 1 GiB. `gc.collect()` then lowered nothing; `malloc_trim(0)` lowered RSS
by 72.0 MB (in the `[heap]` class only). Every watched object of the earlier pipelines was gone at
every check, as in phase 2. RssAnon and Pss moved by the same amounts as RSS in all four runs.

The segments (RSS change, MB):

| | reload 1 | reload 2 | reload 3 |
|---|---|---|---|
| `_precompute_node_embeddings`, entry → exit | −3.1 | −3.5 | −26.1 |
| `_load_shortest_paths`, entry → exit | +3889.2 | +4319.1 | +4319.2 |
| `_load_shortest_paths` exit → the reload's response | −4409.6 | −4835.7 | −4476.0 |
| the response → after the five diagnoses | −60.9 | −56.7 | −476.8 |

The served SP lookup's two tensors are 3887.1 MB (keys 3455.2, distance 431.9). The SP step's
rise exceeds that by about 432 MB at reloads 2 and 3, while the old and the new pipeline are both
resident; `build_sp_index` makes row-sized temporaries (the reordered distance, a one-byte-per-row
comparison) that the lookup does not keep. That matches the size of the earlier steps but is a
lead, not an attribution: the readings hold no addresses, and mapping sizes say nothing about
which allocator made a mapping.

By mapping size class, the run's whole residual (+266.8 MB) is: `[heap]` +88.1, up to 256 MiB
+358.4, up to 64 MiB −189.4, up to 8 MiB −6.1, up to 1 MiB +16.3, larger than 1 GiB −0.7; the large
mappings ended within 1 MB of where they started.

Draft 3 added readings and changed when things were read; it and draft 1 are not a controlled
comparison, and three reloads show neither a steady rise nor a plateau over a longer service.

## 3. The dependency defect

Found by the reviewer, read-only, on the GB10 (`libc10-sha256.txt`,
`libc10-disassembly-interpretation.txt`; the full disassembly is in the owner's archive below):

- The wheel's `torch/lib/libc10.so` (SHA-256
  `1789acef0ce56aece3aafb940f9b74233354937cc5910b3081702899fae2de65`) allocates and frees CPU
  memory through mimalloc: `c10::alloc_cpu` calls `mi_malloc_aligned`, `c10::free_cpu` calls
  `mi_free`; `mi_version` returns 224 (mimalloc 2.2.4). glibc's `mallinfo2` and `malloc_trim`
  therefore neither measure nor reclaim this path, which is why they showed almost none of the
  SP tensors.
- Its arena purge has the logic of two defects mimalloc has since fixed upstream:
  [`acd6f6c`](https://github.com/microsoft/mimalloc/commit/acd6f6ced8e15a0958a2c6117b4cb0242646bdce)
  (an inverted expiry comparison: an ordinary purge returned early once its deadline had
  *passed*) and
  [`4e50cec`](https://github.com/microsoft/mimalloc/commit/4e50cec3eeeeeab9de049066749d108a9d640765)
  (the global purge deadline cleared even when arenas were only not yet due). Both were checked
  against the machine code at the addresses the interpretation lists, not inferred from a
  version string.
- The ELF was inspected after the runs; the runs did not record its hash at launch.

What this does and does not show is the conclusion above: freed CPU memory of the SP build being
returned late or not at all by this purge path fits the sizes and the timing, and is not proven to
account for all of the retention.

## 4. What is recorded apart

- **The per-reload retention common to both commits:** a pre-existing serving behaviour of this
  wheel, with the strong candidate cause above; BACKLOG item 22 keeps it as a dependency
  limitation, to be revisited only when a PyTorch wheel is shown, in its `libc10.so`, to carry both
  upstream fixes. No change to SHEPHERD's pipeline is made for it.
- **M2.1's ~30 MB difference at the reload** (phase 1: reproduced at reload 1, then ambiguous):
  S10's acceptance 2, decided separately by the owner and closed on 2026-10-13 with the remaining
  uncertainty explicit ([`capacity-m2.1-2026-10-12/` §6.2](../capacity-m2.1-2026-10-12/README.md#62-acceptance-2026-10-13)).
- **S10's capacity:** judged on its own readings and limits
  ([`capacity-m2.1-2026-10-12/`](../capacity-m2.1-2026-10-12/README.md)); accepted by the owner on
  2026-10-13 on its designated subject, not by this run.

## 5. Files

| File | What |
|---|---|
| `inproc.json`, `exit.txt` | this run's document and exit status, as the owner sent them (archive `s10-reload-inproc-draft3-2026-10-13.tar.gz`, SHA-256 `23bc8ddc56161d1deba9a1786190a1455ea406971c61c675352f229210ba7e9f`, whose `SHA256SUMS` passed) |
| `libc10-sha256.txt`, `libc10-disassembly-interpretation.txt` | the reviewer's read-only binary analysis, as sent (archive `s10-reload-cross-analysis-2026-10-13.tar.gz`, SHA-256 `c9315d75564cc0a762b1b806da424b1f23c843659723c195bc7b40ef6a183b98`, which also holds the full disassembly and the two upstream patches) |
| `SHA256SUMS` | the four files above |

The tools, runbook, child log and request list stay outside the repository.
