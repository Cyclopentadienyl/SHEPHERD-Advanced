# PLAN — backlog item 5a: productionising approach A

**Status: draft for review. No file under `src/inference/` is edited until this
is approved** — `PLAN_B04.md` §13 requires exactly that, and this document
exists to satisfy it.

**Authority above everything here:** `docs/DISEASE_SCORER_POLICY.md`.
`PLAN_B04.md` governs what was measured and what was selected; this plan governs
only how the selection reaches production.

---

## 1. What is already settled, and may not be re-opened here

Re-stating these so the review is about the wiring and not about the choice:

| | Settled in | Result |
|---|---|---|
| Which implementation ships | §12.5 | **A — `global`**, for the primary GB10 platform |
| Correctness of A against the scanning primitive | §11.2 | **exact equality**, mutation-checked. §11.2 says 25 tests; the file collects **28** today, so the plan cites what is there rather than the figure written when it was |
| Latency at the shipped caller shape | §12.2-12.3 | 8-33× faster; 0/60 over the provisional budget vs 22/60 today |
| Memory cost of A | §12.4 | **3.44 GB permanent resident**, process peak unchanged |
| Duplicate-freedom of real tables | §5.3.2, §10.1 | two artifacts, two HPO vintages, zero duplicates |
| That the caller keeps its per-candidate shape | §4.1 | deferred to B-1, not this plan |

**No second prototype, no runtime selector, no flag.** §13 is explicit: B is not
kept alive behind a switch. This plan ships one implementation.

---

## 2. Scope

### In

1. Move approach A from `scripts/sp_index_prototypes.py` into `src/inference/`.
2. Build A's index inside `_load_shortest_paths`, once, after the table is
   loaded and sorted.
3. Dispatch `sp_mean_distances` to the indexed implementation when the index is
   present, with **an unchanged signature and unchanged results**.
4. Carry the §5.3.2 uniqueness assertion into the load path.
5. Whatever of `PLAN_B04.md` §13's five readings this machine can produce
   (§7 below), reported at the strength the run supports.

### Out

- **The production caller's shape.** §4.1 — B-1 restructures it anyway, and
  changing it now means changing it twice.
- The SP transform `1/(1+d)`, η, and what SP is used for. Policy's, and B-1's.
- Any runtime A/B selector, backend registry, memory-budget framework, cache,
  or second prototype (§13).
- `scripts/compute_shortest_paths.py`. Three defects are recorded in §8 and
  deliberately **not** bundled into this change.
- Deciding what to do if the §13 gate fails. §13: *"the finding is a memory
  result about A on a specific target, and what to do about it is decided
  then — not pre-built now against a failure that has not happened."*

---

## 3. Where the code lives — one copy, not two

`scripts/sp_index_prototypes.py` holds both prototypes and is imported by
`scripts/benchmark_sp_lookup.py` and pinned by
`tests/unit/test_sp_index_prototypes.py`. Copying A into `src/` would leave two
implementations of the shipped primitive, which is the defect class this
project has repaired three times (`setup_demo`'s second workspace producer, and
twice before).

**Proposal.** A moves to `src/inference/sp_index.py`. The prototypes module
imports it and re-exports it under its existing name, keeping B where it is.

- one implementation of A, in the layer that ships it;
- `benchmark_sp_lookup.py` can still re-run the §12 matrix, which is what makes
  a future measurement on another target possible at all;
- B stays a **measurement artifact**, not a production alternative behind a
  flag. §13 forbids the latter and says nothing about the former;
- `test_sp_index_prototypes.py`'s 28 tests keep testing the shipped code rather
  than a copy of it that can drift.

---

## 4. The changes, file by file

| File | Change |
|---|---|
| `src/inference/sp_index.py` | **new.** Approach A: `GlobalKeyIndex`, `build_global_key_index`, `sp_mean_distances_global`, and the private helpers A uses. Moved, not rewritten |
| `scripts/sp_index_prototypes.py` | A's definitions replaced by an import from `src.inference.sp_index`; B unchanged |
| `src/inference/pipeline.py` | `_load_shortest_paths` sorts by the composite key (D3) and builds the index from the sorted tensors, so the index shares `distance` rather than copying it; `SPLookup` construction otherwise unchanged |
| `src/inference/scoring.py` | `sp_mean_distances` uses the index when the lookup carries one, and the existing scan when it does not. Signature, return types and the float64 contract unchanged |

**The scan is not deleted.** `sp_mean_distances` keeps it for the case where no
index was built — which is what the equivalence tests compare against, and what
`SPLookup`s constructed outside the loader (tests, the offline harness) still
get. Keeping it is not a runtime selector between two shipped implementations;
it is the primitive's existing behaviour for a lookup with no index.

---

## 5. Decisions this plan asks the reviewer to make

### D1 — what a duplicate table does at load time

§5.3.2 says the index build **"asserts uniqueness and fails loudly"**, and notes
this changes startup behaviour. What "loudly" means operationally is not settled,
and the two readings differ in what an operator loses.

`_load_shortest_paths` already has three early-return paths — missing file,
unreadable file, missing required keys — each leaving `_sp_ready = False` while
the pipeline serves without SP. SP is optional by design: the probe records
`sp_ready: false` on a passing deployment, and `scoring_mode=gnn_only` is a
supported state.

- **(a) Recommended — refuse SP, not the pipeline.** Log at ERROR naming the
  artifact and the offending key, leave `_sp_ready = False`. Loud, surfaced
  through the existing `sp_ready` field, and consistent with the three paths
  already there. An optional subsystem's bad artifact does not take down a
  clinical service that can still answer.
- **(b) Raise.** `initialize()` fails and the pipeline does not build. Stronger
  reading of "fails loudly", but it converts an optional subsystem into a
  mandatory one, which no policy states.

I recommend (a) and will implement (b) if the reviewer reads §5.3.2 as requiring
it. **Either way the condition is never silent** — the current `except
Exception: pass` shape is not on the table.

### D2 — where the index is held

- **(a) Recommended — on `SPLookup`.** It already bundles `target`,
  `target_type`, `distance`, `offsets` and `max_hops` as the view the scoring
  primitives read. An optional `index` field keeps the primitive's inputs in one
  object, and `sp_mean_distances` needs no new parameter.
- **(b) A separate `_sp_index` attribute on the pipeline**, passed alongside.
  Keeps `SPLookup` untouched, at the cost of two things travelling together
  that can be separated by a caller.

### D3 — the loader must adopt the composite sort, and here is the arithmetic

§5.6 names a consequence: adopting the single composite sort reorders `_sp_tg`,
`_sp_ty` and `_sp_di`, which a comment calls *"part of this class's observable
surface"*. Results are invariant, because uniqueness means exactly one row
answers any `(phenotype, target, target_type)` query whatever the storage order.

**My first draft of this plan proposed skipping the reorder**, on the correct
observation that `build_global_key_index` performs its own `argsort` and does
not require its input in composite order. That was wrong, and checking the
memory arithmetic is what showed it.

The prototype sorts independently of the loader, so it keeps **its own reordered
copy of `distance`** beside the key column. §12.4 reports both numbers and they
reconcile exactly at 430M rows:

| | bytes/row | 430M rows | §12.4 column |
|---|---|---|---|
| int64 key column | 8 | 3.44 GB | **production steady-state 3.44 GB** |
| int8 `distance` copy | 1 | 0.43 GB | — |
| both | 9 | **3.87 GB** | **resident (measured) 3.88 GB** |

So §12.4's headline 3.44 GB — the figure §12.5 calls *"the whole of the case
against A"* — is the **key column alone**, and it is only what production pays
**if the loader's tensors are already in composite order**, letting the index
share `distance` instead of copying it. §11.1's *"retained beyond the loader's
own tensors: a full-length int64 key column"* is a statement about that
arrangement, not about the prototype as measured.

**Proposal: adopt the composite sort in `_load_shortest_paths`, as §5.6 says.**
Not doing so would ship an index costing 0.43 GB more than the figure the
selection was argued on, on the unified memory §12.5 says is the constraint —
and §13's reading 2 would then measure a number the plan never predicted.

The observable-order change is accepted rather than avoided, exactly as §5.6
frames it. `_sp_ph` is already reordered by the existing `argsort`, so these
attributes have never been in input order.

**Every reader enumerated, so this is not left to implementation.** Across
`src/`, `scripts/` and `tests/`, the three attributes are read in exactly one
place: `_load_shortest_paths` assigns them and passes them into `SPLookup`.
Nothing reads them positionally, and the one test-file mention is a comment.

The real constraint is `_sp_ph`, not those three: `_sp_offsets` is built from
`torch.where(self._sp_ph[1:] != self._sp_ph[:-1])`, which requires each
phenotype's rows to be **contiguous**. A composite key with phenotype as its most
significant component preserves that by construction — it is the same property
`build_global_key_index` relies on — so `offsets` remains valid and the scan
path keeps working for lookups built without an index.

---

## 6. Invariants inherited, and how each is checked

| Invariant | Source | Check |
|---|---|---|
| float64 for the whole computation | §5.3.1 | existing `test_scoring_primitives.py`, unchanged |
| Exact equality with the scanning primitive | §5.3, §6 | the 28 existing prototype tests, now against the shipped module |
| Uniqueness asserted, not assumed | §5.3.2 | existing test that constructs a duplicate table; plus D1's behaviour |
| int64 key domain bounds-checked in Python integers before any int64 tensor exists | §5.2, §6 | existing overflow test — mutation-checked in §11.2 and kept |
| No query path touches more than `O(log L)` rows | §11.1 | the query is cast to the stored dtype, never the reverse. Preserved by moving rather than rewriting |
| `tests/unit/test_scoring_primitives.py` passes unchanged | §6 | run unchanged |

**Nothing new is invented here.** Every row is an existing check that must keep
holding after the move.

---

## 7. The §13 gate — what this machine can answer, and what it cannot

| # | Reading | Status |
|---|---|---|
| 1 | complete pipeline cold start with A wired in | **available here** |
| 2 | steady and peak RSS/UMA once serving | **available here** |
| 3 | one real reload, *if live reload is supported* — establish that first | **supported**: probe E4 exercises it and returns a built candidate |
| 4 | peak while old and new pipeline state coexist | **available here, and currently open**: probe E4 reports `double_residency_conclusive: false` because the demo model is 43,553 parameters / 18.5 MB. A deployment-sized workspace is what makes it conclusive, and this plan produces one |
| 5 | the same on the **smallest supported deployment target** | **blocked** — that machine is not available. The reading is deferred, not waived, and the gate is not claimed complete without it |

Reading 4 is the open item this project has carried since the reload work: it is
not a separate task that happens to be nearby, it is one of the five readings.

**Acceptance beyond the gate needs a designated loadable checkpoint** (BACKLOG
§3.5, item 6 — an institutional decision). This plan therefore ends at
*implemented, gate readings 1-4 recorded, 5 deferred, acceptance pending item 6*.
It does not claim clearance to ship.

---

## 8. Three adjacent defects, recorded and deliberately not bundled

Found while reading the SP path. All three are in
`scripts/compute_shortest_paths.py` or its sidecar contract — a different entry
point from the one this plan edits. **They are listed so they are not lost, and
excluded so this change stays reviewable.**

1. **`max_hops` falls back to 5 in silence.** `_load_shortest_paths` reads
   `shortest_paths.meta.json` inside `try: ... except Exception: pass`, so a
   missing or malformed sidecar leaves `_sp_max_hops = 5`. `max_hops` sets the
   unreachable sentinel (`max_hops + 1`), so a table built with a different
   ceiling is scored against the wrong one **and the scores look ordinary**.
   This is the most serious of the three and the only one inside the function
   this plan edits — I still propose keeping it separate, because it changes
   scoring behaviour and deserves its own review rather than riding along with a
   performance change.
2. **The tensor is written before its sidecar.** Same ordering defect class as
   the one just repaired in the workspace writer: a failure between the two
   leaves a table whose ceiling is unrecorded, which is defect 1's input.
3. **`--kg-path` and `--output-dir` are independent**, so SP artifacts can be
   written into a workspace they were not computed from, with nothing binding
   them.

Proposal: raise these as one small backlog item after 5a lands.

---

## 9. What this plan does not build

No runtime A/B selector. No backend registry. No memory-budget framework. No
cache or memoisation. No second prototype. No change to the production caller's
shape. No pre-built response to a gate failure that has not happened.

---

## 10. Sequence

1. This plan reviewed and approved.
2. Move A to `src/inference/sp_index.py`; prototypes module re-exports it. A
   references no part of B — every `slices` symbol in that file is below A's
   section — so the move is a clean lift. The 28 existing tests pass unchanged,
   against the moved code.
3. Wire the index into `_load_shortest_paths` per D1-D3.
4. Dispatch in `sp_mean_distances`; equivalence tests indexed-vs-scan.
5. Mutation-check the load-time uniqueness assertion and the dispatch.
6. `make check`.
7. Build a deployment-sized workspace and record §13 readings 1-4.
8. Report, with reading 5 and item 6 named as outstanding.

Steps 2-6 need no checkpoint, no calibration and no institutional input. Step 7
needs a real build, which this machine has done in 39.9 s (probe F1).
