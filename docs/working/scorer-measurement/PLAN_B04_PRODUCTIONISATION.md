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
| `src/inference/sp_index.py` | **new.** Approach A: `GlobalKeyIndex`, `sp_mean_distances_global` and A's private helpers, moved unchanged; plus `build_global_key_index_presorted` — the seam D4 requires, which is new code and not a move |
| `scripts/sp_index_prototypes.py` | A's definitions replaced by an import from `src.inference.sp_index`; `build_global_key_index` stays here as the **sorting wrapper** the benchmark needs for unsorted input; B unchanged |
| `src/inference/pipeline.py` | `_load_shortest_paths` sorts by the composite key (D3) and builds the index from the sorted tensors, so the index shares `distance` rather than copying it; `SPLookup` construction otherwise unchanged |
| `src/inference/scoring.py` | `sp_mean_distances` uses the index when the lookup carries one, and the existing scan when it does not. Signature, return types and the float64 contract unchanged |

**The scan is not deleted.** `sp_mean_distances` keeps it for the case where no
index was built — which is what the equivalence tests compare against, and what
`SPLookup`s constructed outside the loader (tests, the offline harness) still
get. Keeping it is not a runtime selector between two shipped implementations;
it is the primitive's existing behaviour for a lookup with no index.

---

## 5. Decisions this plan asks the reviewer to make

### D4 — the presorted seam, without which D3 buys nothing

**This was a blocker in review, and it was right.** D3 establishes that the
loader must composite-sort so the index can share `distance` instead of copying
it. The plan then said A would be *"moved, not rewritten"* — and
`build_global_key_index` unconditionally does:

    order = keys.argsort()
    keys = keys[order]
    distance = lookup.distance[order]

so it re-sorts an already-sorted table and advanced-indexes `distance` into a new
tensor regardless. Measured on a table already in composite order:

    existing builder  : distance shared = False
    presorted seam    : distance shared = True
    keys identical    : True

The plan as drafted would therefore have shipped the **3.87 GB** design while
citing the 3.44 GB one, and paid a second full-table sort at every cold start.
The arithmetic in D3 was right and the implementation strategy did not deliver
it.

**A separate builder for presorted input.** It derives the same domain and the
same keys, then:

- verifies the keys are **non-decreasing** — a table not in composite order is
  refused rather than silently mis-indexed;
- verifies **no two adjacent keys are equal** — §5.3.2's uniqueness assertion,
  now falling out of the same single pass;
- keeps `lookup.distance` **as the same tensor**, with no advanced index;
- performs **no argsort**.

Measured on the three cases that matter:

| input | outcome |
|---|---|
| composite-sorted | accepted, `distance` shared |
| target descends inside a run | refused — not in composite order |
| duplicate row | refused — duplicate rows |

*(The first version of that check accepted an out-of-order table, because the
case I chose changed `target` across rows whose `target_type` already differed —
the type component broke the tie first, so the table was still ordered. The case
above changes `target` **within** one `(phenotype, target_type)` run, which is
the only place ordering can actually break.)*

**`build_global_key_index` is not deleted, and does not keep its own build.**
It stays in the prototypes module as the sorting wrapper, because
`benchmark_sp_lookup.py` feeds it a table straight off disk in whatever order the
artifact has. It **sorts, then delegates to the presorted builder** rather than
constructing an index itself — otherwise the two entry points would each carry a
copy of the domain derivation, the key construction and the uniqueness check, and
the wrapper's copy would be the one nothing in production exercises. One build,
two ways in.

**Tests must prove storage sharing, not only equal numbers.**
`index.distance.data_ptr() == lookup.distance.data_ptr()` on the production
path, so a future edit that reintroduces the copy fails rather than merely
costing 0.43 GB in silence. Mutation-checked by putting the argsort back.

### D5 — validate before narrowing, not after

**Also a blocker in review, and also right.** `_load_shortest_paths` narrows
before anything inspects the values:

    ph_t = sp_data["phenotype_idx"].to(torch.int32)
    tg_t = sp_data["target_idx"].to(torch.int32)
    ty_t = sp_data["target_type"].to(torch.int8)

`_derive_domain` runs later, on the narrowed tensors, so it can only see what
survived. Measured:

    int64 2147483648  -> int32 -2147483648
    int64 2147483655  -> int32 -2147483641
    target_type [2, 7] -> int8 [2, 7]   (no wrap, but outside {0, 1})

A wrapped id becomes a negative one; the domain derivation then reads a
different table from the one on disk, and the index is built over it. This is
latent today — the scan path has the same exposure — and productionising A is
when it stops being acceptable, because the index's whole correctness rests on
the domain.

**A small load validator, run on the columns as loaded, before any `.to()`:**

- the mapping root and the four required columns are present;
- each is a tensor, one-dimensional, and all four the same length;
- index columns are integral and not `bool`;
- `phenotype_idx` and `target_idx` are non-negative and within int32;
- `target_type` is exactly within `{0, 1}`;
- `distance` is integral and not `bool`, and on a non-empty table lies within
  `1 .. max_hops` — the producer's domain, stated as a rule a test can execute
  rather than as "its producer's domain", which the draft left to the reader.
  `max_hops` here is the validated sidecar value, which is why §8.1's fix is a
  precondition and not a nicety: without it the domain is checked against a
  guess.

**The empty table is a hole today, and the validator is where it closes.**
Measured on the current loader's offset construction:

    changes: []   starts: [0]
    ph[starts] -> IndexError: index is out of bounds for dimension with size 0

A `shortest_paths.pt` that is well-formed but carries no rows therefore aborts
`initialize()` with an `IndexError` from a tensor index — not a named refusal,
and from a call site with no handler. The prototypes' empty-table test does not
reach this, because it constructs a lookup directly rather than going through the
loader.

**Proposed: an empty table is the absent case.** It binds nothing, so there is
nothing for SP to be ready for; the loader takes its existing early-return path,
`_sp_ready` stays False, and the operator sees a named reason. Refusing it as
invalid is the other defensible reading, and I take the weaker one because an
empty table makes no false claim — it is indistinguishable in effect from having
no file, which is a state policy blesses.

**Not a schema framework**, and no new module: one private function beside the
loader, refusing with the column named. It is the same shape as
`validate_feature_dim` and `validate_allocation_seed` — a domain check where the
value enters, not a validation layer.

### D1 — a rejected artifact is not an absent one — resolved against policy

My draft recommended fail-open: log at ERROR, leave `_sp_ready = False`, serve
without SP. Review objected that this does not distinguish *absent by
configuration* from *present but rejected*. That distinction is right, and the
resolution is to refuse — but my argument for it over-reached and is narrowed
here.

**The direct authority is `PLAN_B04.md` §5.3.2**, which requires the index build
to assert uniqueness and fail rather than preserve first-match semantics for a
case the data model says cannot occur. That covers duplicate rows on its own
terms and needs no help.

**What §2 of the policy does and does not say.** It records that an **absent**
`shortest_paths.pt` degrades to pure GNN — so *absent* is an accepted state, and
that much is load-bearing here. My draft also leaned on the adjacent row's
*"fail-closed by default; fallback only by explicit request or approved
deployment policy"*. **That row's subject is GNN unavailability, not SP**, and
citing it as settled policy for SP rejection was a misattribution. Withdrawn.

**So this is a proposal, not an existing mandate**, for the invalid-artifact
cases §5.3.2 does not name — non-monotonic keys, an id outside its domain, a
malformed column: treat them as §5.3.2 treats duplicates. The argument stands on
its own: *absent* is a state an operator chose, *rejected* is one nobody chose,
and serving as though they were the same reports a degraded mode that was never
configured. If the reviewer wants that narrowed to duplicates alone until the
institution rules, say so — the validator's shape does not change, only which
failures abort.

**What refusing costs, stated for both situations rather than one.**

- **On reload, nothing.** `build_pipeline` constructs a candidate without
  touching application state and `publish_pipeline` swaps only after it
  succeeds, so a rejected SP artifact fails the candidate and the running
  pipeline keeps serving. Probe E5 already exercises this for a different
  refusal.
- **On cold start, the service does not come up.** There is no previous pipeline
  to keep serving, so refusal is a full outage until the artifact is fixed or
  removed. My draft said refusing "costs no availability" without this
  qualification, which was true of the case I had in mind and false of the other
  one.

That second bullet is not an argument for fail-open — an invalid artifact at cold
start is a deployment that was never valid, and the operator's remedy is to
remove the file, which reaches the *absent* path policy blesses. It is stated
because a reviewer weighing this deserves both halves.

**`_sp_ready` is published too early today, and this plan must fix it.**
`pipeline.py:619` sets it before the `max_hops` sidecar is read (620-628) and
before `SPLookup` is constructed (636-643) — so there is a window in which SP
reports ready while its ceiling is unknown and its lookup does not exist. Under
5a the index build joins that tail, widening the window. The flag moves to
**after** the complete, validated index exists. This is the same ordering rule
the workspace writer was just corrected for: publish after the fallible work,
not before it.

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

## 6.1 What the implementation review will be shown

Agreed with the reviewer, and recorded so the implementation is built to produce
it. **Ordinary unit and integration tests — no new deployment probe, no
validation framework.** The last round's process finding stands: probes are for
claims CI cannot establish, and every item here is deterministic and CPU-only.

1. **Entry through the real loader, not the validator.** Illegal raw ids, dtypes,
   `target_type` values and distances are refused *via a load call* — and
   deleting the validator's call site makes those tests fail. The rule and its
   application are separate claims and both get mutated.
2. **Every non-happy load path matches D1.** Unreadable artifact, missing
   columns, duplicates. The current `torch.load` handler warns and returns —
   fail-open — and that path does not survive this change.
3. **Storage sharing is asserted, not inferred.** On a non-empty production
   lookup, `index.distance.data_ptr() == lookup.distance.data_ptr()`; the
   ordering check, the uniqueness check and the dispatch each have a mutation
   that fails exactly the test claiming them.
4. **The equivalence comparison uses an index-free lookup for its expected
   values.** Otherwise the new dispatch sends both sides to the same indexed
   implementation and the test compares it with itself — which would pass for the
   wrong reason, and is the failure mode this project keeps finding.
5. **Refusal is tested in both situations D1 distinguishes.** A rejected
   candidate does not publish and the old pipeline keeps serving; a cold start
   with no previous pipeline refuses and is verified separately.

---

## 7. The §13 gate — what this machine can answer, and what it cannot

| # | Reading | Status |
|---|---|---|
| 1 | complete pipeline cold start with A wired in | **pending a designated measurement subject** — see below |
| 2 | steady and peak RSS/UMA once serving | **pending a designated measurement subject.** This is an *integrated* reading — §13 exists precisely because an isolated benchmark does not cover model, graph, embeddings and API resident together — so an SP-only figure cannot complete it, however useful it is |
| 3 | one real reload, *if live reload is supported* — establish that first | **half answered.** Live reload *is* supported: probe E4 builds a candidate beside the live pipeline and E5 shows a refused one leaves it serving. A reload **with the index wired in** has not been measured, and that is the half §13 asks for |
| 4 | peak while old and new pipeline state coexist | **pending a designated measurement subject** — probe E4 reports `double_residency_conclusive: false` because the demo model is 43,553 parameters / 18.5 MB. A deployment-sized *graph* does not fix this; the resident model is the other half |
| 5 | the same on the **smallest supported deployment target** | **blocked** — that machine is not available. The reading is deferred, not waived, and the gate is not claimed complete without it |

### 7.1 The measurement subject — designated, not authoritative

**Two revisions in a row got this wrong in opposite directions.** The first
draft claimed readings 1 and 4 were available here because this machine builds a
deployment-sized workspace in 39.9 s (probe F1) — which measures the graph half
and leaves a 43,553-parameter demo model resident, so it measures neither. The
correction then marked them *pending BACKLOG item 6*, the clinical decision about
which checkpoint is authoritative.

That second move invented a dependency the programme explicitly disclaims.
BACKLOG §3.5:

> **"Designated loadable" is not "authoritative".** 5a does not wait on item 6's
> clinical decision; it needs *a* checkpoint that loads against the artifact set,
> which is a far weaker requirement. If the finally deployed model has a
> materially different architecture or memory footprint, compatibility is
> confirmed **for that model** — that is a re-run of the gate, not a blocker on
> it now.
>
> No checkpoint registry. **One designated file, named in the 5a plan.**

So naming the subject is this plan's job, and blocking on item 6 both invents a
dependency and skips the task.

**What designation requires**, and all of it is engineering:

1. the file's **SHA-256**, recorded as the measurement subject;
2. that it **loads against this artifact set** — the graph export and SP table
   the readings are taken over;
3. a statement of **which deployment shape it stands for**, so a materially
   different final model is recognised as needing a re-run rather than silently
   covered.

**Candidates already evidenced.** `EVIDENCE_M1_M3_hgt.json` and
`EVIDENCE_M1_M3_gat.json` record ten checkpoints each by digest, from a machine
in an `identical-sibling` deployment relationship, with load results and an
`in_channels` of 128 established across the family. Those digests are a
designation list, not a decision: one is chosen, its availability on the
measuring machine confirmed, and it is named here before step 7 runs.

I have **not** named one in this revision, because the evidence records digests
and loadability but no parameter count or resident size, and I will not assert a
deployment shape I have not measured. The blocking condition is therefore
*"awaiting a designated measurement subject"* — satisfiable by this project, not
by the institution.

### 7.2 What this plan completes unaided

The implementation and its tests. **No integrated reading.**

Readings 1, 2 and 4 are all integrated — they measure what is resident while the
service runs — so all three wait on the subject above. An SP-only memory figure
is worth recording and this plan will record it, but as a **supplementary
measurement**, named as such, never as reading 2. Treating it as the reading
would leave unverified serving memory shelved as done.

The plan ends at: *implemented; a supplementary SP-only memory figure recorded;
readings 1, 2, 3b and 4 pending a designated measurement subject; reading 5
deferred for want of the smallest supported target*. Not a clearance to ship, and
the gate is not claimed complete.

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
   this plan edits. It stays a separate commit and a separate review — it changes
   scoring behaviour and should not ride along with a performance change — but
   review added a sequencing requirement this plan now carries: **it must land
   before the indexed path is activated.** Exact agreement between the indexed
   and scanning implementations under the same wrong `max_hops` is agreement, not
   correctness; shipping a faster primitive over an unverified ceiling would make
   the wrong answer arrive sooner. Defect 2 is its input and defect 3 is
   unrelated to it, and neither is bundled without separate approval.
2. **The tensor is written before its sidecar.** Same ordering defect class as
   the one just repaired in the workspace writer: a failure between the two
   leaves a table whose ceiling is unrecorded, which is defect 1's input.
3. **`--kg-path` and `--output-dir` are independent**, so SP artifacts can be
   written into a workspace they were not computed from, with nothing binding
   them.

Proposal: defects 2 and 3 become one small backlog item after 5a lands. **Defect
1 is not among them** — it is listed in §10 as step 0, a precondition of
activating the indexed path, and the draft's closing line contradicted that by
sweeping all three into the same post-5a bucket.

---

## 9. What this plan does not build

No runtime A/B selector. No backend registry. No memory-budget framework. No
cache or memoisation. No second prototype. No change to the production caller's
shape. No pre-built response to a gate failure that has not happened.

---

## 10. Sequence

0. **`max_hops` sidecar defect fixed, reviewed and landed** (§8.1). A
   precondition of step 3, not of steps 1-2.
1. This plan reviewed and approved.
2. Move A to `src/inference/sp_index.py`; prototypes module re-exports it. A
   references no part of B — every `slices` symbol in that file is below A's
   section — so the move is a clean lift. The 28 existing tests pass unchanged,
   against the moved code.
3. Wire the index into `_load_shortest_paths` per D1-D3.
4. Dispatch in `sp_mean_distances`; equivalence tests indexed-vs-scan.
5. Mutation-check the load-time uniqueness assertion and the dispatch.
6. `make check`.
7. Record the SP-only memory figure as a **supplementary measurement**, labelled
   as such. It completes no §13 reading.
8. Report: implemented; supplementary figure recorded; readings 1, 2, 3b and 4
   awaiting a designated measurement subject (§7.1); reading 5 deferred for want
   of the smallest supported target.

Steps 0-7 need no calibration and **no institutional decision** — §7.1's
designation is engineering, and item 6's clinical choice is not a prerequisite
for any of it. What steps 1-7 cannot do is complete an integrated reading, which
is why the plan stops at step 8 rather than at a gate verdict.
