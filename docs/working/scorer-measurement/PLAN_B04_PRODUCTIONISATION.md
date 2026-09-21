# PLAN — backlog item 5a: productionising approach A

**Status: revised and cleared to implement.** The blanket prohibition this
document carried — *"No file under `src/inference/` is edited until this is
approved"* — is **removed, not satisfied**. It was written to stop production
code being edited ahead of a plan. What it produced was a gate whose key nobody
could find: the approval it names left no record in this repository, and step 2
sat behind it while three documents repeated that the plan was not approved.

**What replaces it.** The project owner directed the revision: where the
existing pipeline needs changing, change it, and **do not grow a second pipeline
to avoid touching the first** — parallel pipelines get miswired during
maintenance and double the cost of it, for no benefit. The reviewer concurred,
ruled that this direction is the authority to implement rather than a further
search for an earlier approval whose subject cannot be identified, and set the
acceptance conditions in §6.1. Both are recorded here rather than left in a
review thread, because this document is what a later reader will have.

**Cleared to implement is not accepted for deployment**, and the two are kept
apart deliberately. Nothing here asserts that `PLAN_B04.md` §13's gate has been
passed, that `DISEASE_SCORER_POLICY.md` has changed, that B-1 has started, or
that any figure in §7 has been measured on a designated subject. A reading this
hardware cannot produce is marked pending and stays pending.

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

1. **One served implementation of the shortest-path primitive, which is A.**
   The scan is not kept behind a condition, a flag or an optional field. A
   `SPLookup` that a caller can query is one whose index was built; there is no
   second state in which the same call takes a different algorithm.
2. A single official builder, taking the **raw columns** —
   `phenotype`, `target`, `target_type`, `distance` — and an explicit
   `max_hops`. It validates, derives the domain, builds the composite key,
   **sorts once**, asserts §5.3.2's uniqueness, and returns the queryable
   object. Nothing constructs an index any other way.
3. `_load_shortest_paths` reads the artifact and the already-validated hop
   bound and hands both to that builder. Validation happens **before** any
   narrowing, so a value the narrow type cannot hold is refused rather than
   wrapped.
4. Every construction site in the repository moves with it — the offline
   harness and the test fixtures included. A fixture that builds the old shape
   is a second pipeline with a smaller blast radius, not an exception to this.
5. `_sp_ready` is published only after all of that has succeeded. An unreadable
   artifact, an absent table or an unknown hop bound each reach a named state;
   **none of them falls back to a scan**, because there is none to fall back
   to.
6. Whatever of `PLAN_B04.md` §13's five readings this machine can produce
   (§7 below), reported at the strength the run supports — and the memory
   conclusions re-measured, because the shape being measured has changed.

### Out

- **The production caller's shape.** §4.1 — B-1 restructures it anyway, and
  changing it now means changing it twice.
- The SP transform `1/(1+d)`, η, and what SP is used for. Policy's, and B-1's.
- Any runtime A/B selector, backend registry, memory-budget framework, cache,
  or second prototype (§13). **This is now stronger than §13 requires.** §13
  forbids shipping a selector; this revision also declines to ship the *shape*
  of one — an optional index field with a scan behind it is a selector whose
  condition happens to be `is None`.
- `scripts/compute_shortest_paths.py`. Three defects are recorded in §8 and
  deliberately **not** bundled into this change.
- Deciding what to do if the §13 gate fails. §13: *"the finding is a memory
  result about A on a specific target, and what to do about it is decided
  then — not pre-built now against a failure that has not happened."*

---

## 3. Where the code lives — one copy, not two

**Settled, and further than this section proposed.** `scripts/sp_index_prototypes.py`
held both candidate implementations and was imported by `benchmark_sp_lookup.py`
and pinned by `tests/unit/test_sp_index_prototypes.py`. The proposal below was
that A move to `src/inference/sp_index.py` while the prototypes module
re-exported it under the old name and kept B.

What shipped keeps the goal and drops the re-export. The primitive lives in
`src/inference/sp_index.py`; `scripts/sp_index_prototypes.py` is **gone**, and
so is B. The reason is not tidiness: B's query walks `SPLookup.offsets`, and the
served lookup no longer has offsets — the composite key replaced them. Keeping B
would have meant keeping the CSR layout alive in `scripts/` for it, which is the
second construction path this revision exists to remove. Git history holds it;
`PLAN_B04.md` §12's matrix is the record of what it measured.

The benchmark now times two things: `indexed`, which is what ships, and
`reference`, the independent Python-dictionary scan in
`scripts/sp_scan_reference.py` that the equivalence tests use. That reference
is not a candidate — it is O(rows) in Python — and production does not import
it.

**The original proposal, for the record:**

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
| `src/inference/sp_index.py` | **new, and the only home of the primitive.** `SPLookup` — now the index itself — `build_sp_index(phenotype, target, target_type, distance, max_hops)`, `sp_mean_distances`, the domain derivation and the uniqueness refusal. One builder; no presorted variant, because nothing hands it a sorted table |
| `src/inference/scoring.py` | `SPLookup` and `sp_mean_distances` **move out**. What stays is what is not about the index: `sp_scores_from_distances`, `validate_hop_bound`, the hop-bound constants, and the cosine/pooling primitives. It does not re-export the moved names — one name, one home, and five import sites updated |
| `src/inference/pipeline.py` | `_load_shortest_paths` validates the raw tensors, resolves `max_hops` **before** the expensive work, and hands raw columns to the builder. `_sp_ph` / `_sp_tg` / `_sp_ty` / `_sp_di` / `_sp_offsets` are **retired**: the composite key encodes all three id columns, the query needs no offsets, and keeping them is the copy this design exists to avoid |
| `scripts/benchmark_sp_lookup.py` | Builds through the one builder, ending its duplicate reader of the loader's layout — a duplicate its own docstring asked to have reconciled the next time the production loader changed. Implementations become `indexed` and `reference` |
| `scripts/sp_index_prototypes.py` | **Deleted.** A moved; B's query walks `SPLookup.offsets`, which the served lookup no longer has, so keeping B would keep the retired layout alive. Git history holds it |
| `scripts/sp_scan_reference.py` | **New.** The independent baseline: rows in a Python dictionary, no shared structure or arithmetic with the served index, so the equivalence tests compare two programs |
| `tests/` | Fixtures and imports move with the refactor. What is preserved is the **behaviour asserted on valid inputs, the independent expected values, and the coverage** — not the text of the files |

**The scan is deleted from the served path**, and that is the point of the
revision. It survives only as a small, independent reference used by tests and
the benchmark to check A's float64 results on valid inputs. Two rules keep that
from becoming a second pipeline: **production never imports the reference**, and
**the reference never calls A** — otherwise the comparison is a program against
itself. Where the scan was wrong (§8.4's dtype aliasing) the reference asserts
the *correct* expected value rather than reproducing the old answer.

---

## 5. Decisions this plan asks the reviewer to make

### D4 — the presorted seam — **resolved: not built, because its premise is gone**

**Read the argument below on its own terms first; it was right about the design
it was arguing against.** Given a loader that composite-sorts and then calls a
builder that sorts again, the second sort is real waste and the presorted seam
removes it. What the revision changes is the premise: there is no longer a
caller that hands the builder an already-sorted table. The single official
builder takes raw columns and performs the one and only sort itself, so the
"sorted input" case it existed to serve does not arise, and a `_presorted`
public entry point would be a second way in that nothing uses.

**What survives is the obligation, not the mechanism.** Sort exactly once, and
let the index hold the tensor that sort produced rather than a copy of it.
§6.1's acceptance keeps both, tested directly — the test is no longer
`index.distance.data_ptr() == lookup.distance.data_ptr()`, because after D2
there is only one object and the comparison is vacuously true; it is that the
builder performs one sort and gathers `distance` once.

The original argument follows unchanged.

### D4 (original) — the presorted seam, without which D3 buys nothing

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

*(Superseded. The concern — two entry points each carrying their own domain
derivation, key construction and uniqueness check — is exactly the concern the
revision answers, and it answers it with one entry point instead of one build
behind two. The benchmark feeds raw columns to the same builder every other
caller uses.)*

**Tests must prove storage sharing, not only equal numbers.**
`index.distance.data_ptr() == lookup.distance.data_ptr()` on the production
path, so a future edit that reintroduces the copy fails rather than merely
costing 0.43 GB in silence. Mutation-checked by putting the argsort back.

*(Superseded in form, kept in substance. With one object there is nothing to
compare pointers between; the equivalent assertion is that the builder sorts
once and gathers `distance` once, mutation-checked by adding a second sort.)*

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
  *(the int32 half is **superseded**: the loader no longer narrows. The columns
  are consumed into an int64 key and released, so the range that matters is
  int64's, which `_derive_domain`'s overflow check covers. Recorded rather than
  dropped, so a reader does not go looking for a check that was deliberately
  not written.)*
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

**`_sp_ready` was published too early; it no longer is, and this plan now owes
the invariant rather than the fix.** When this was written, `_sp_ready` was set
before the `max_hops` sidecar was read and before `SPLookup` was constructed, so
SP reported ready while its ceiling was unknown and its lookup did not exist.
§10 step 0 closed that: the loader now sets `_sp_max_hops`,
`_sp_hop_bound_source`, `_sp_lookup` and then `_sp_ready` last
(`src/inference/pipeline.py:753-757`), after the floor check and the lookup both
succeed.

**What 5a owes is not to reopen it.** The index build joins that same fallible
tail — it derives a domain, allocates a key column, sorts, and can refuse for
duplicates or an out-of-range domain. Every one of those must complete before
the flag, not after. Stated as an invariant with a test rather than as a change,
because the change is already made and a plan that still asks for it would send
a reader looking for a defect that is not there.

### D2 — where the index is held — **resolved: `SPLookup` *is* the index**

The two options offered were an **optional** `index` field on `SPLookup`, or a
separate attribute travelling alongside it. Both were asked under the
assumption that the scan stays; with one served implementation neither is
available, because both leave a lookup that is queryable without an index and
therefore a second algorithm behind a condition.

The resolution is narrower than either: `SPLookup` carries the composite key,
the sorted distances, the domain and `max_hops`, and carries nothing else.
Constructing one *is* building the index, so "the index is absent" is not a
state the type can be in, and `sp_mean_distances` needs no new parameter
because there is nothing to choose between.

**This retires `target`, `target_type` and `offsets` from the served object,
which is a memory consequence and not only a tidiness one.** The key encodes
all three id columns and the query reaches rows by `searchsorted`, so nothing
reads them. §4's table retires the loader's parallel copies for the same
reason. The arithmetic is in D3; it is arithmetic, and §6.1 requires it be
measured before it is quoted as a result.

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
3. **One sort, and the sorted tensor is the one kept.** Asserted directly on
   the builder rather than inferred from timings, and mutation-checked by
   adding a second sort. (This replaces a `data_ptr()` equality between index
   and lookup, which after D2 compares an object with itself.)
4. **The equivalence comparison's expected values do not come from A.** The
   reference implementation is independent of the served one and is not
   importable from production; if it ever delegates to A the test compares a
   program with itself, which is the failure mode this project keeps finding.
   Where the reference's original behaviour was wrong — §8.4 — the expected
   value is the correct one, not the old one.
5. **Refusal is tested in both situations D1 distinguishes.** A rejected
   candidate does not publish and the old pipeline keeps serving; a cold start
   with no previous pipeline refuses and is verified separately.

**Added by the revision, and binding on it:**

6. **Out-of-domain queries cannot read another node's distance.** Covering at
   least `target = 2**32`, `target_type = 256`, negative ids, and an ordinary
   id that is simply absent — through the **official entry point**, not the
   reference. A miss inside a non-empty phenotype/candidate set keeps the
   existing semantics: it contributes `unreachable` and `available` stays true.
   This is §8.4's defect, and it is an acceptance condition of the replacement
   rather than a deferred repair, because the implementation that had it is not
   being kept.
7. **Validation precedes narrowing.** Raw dtype, sign, range and column-length
   checks run on the tensors as loaded. A value the narrow type cannot hold is
   refused, never wrapped — the same mechanism as §8.4, one step earlier.
8. **`_sp_ready` is published only after the index exists**, with the index
   build inside the fallible tail. Tested; see D1.
9. **No second construction path anywhere in the repository.** Every site that
   builds a queryable lookup goes through the one builder, test fixtures
   included, and a fixture that assembles the object by hand is a finding.
10. **Memory is re-measured, not carried over.** The shape changed, so
    `PLAN_B04.md`'s resident and peak figures for A describe a design that is
    no longer the one shipping. Any number quoted for the new path is measured
    or marked pending; the 3.44 GB projection is not evidence about it.
11. **F821 covers the new production module** and whatever SP tooling is kept,
    not only the pre-move prototype.
12. **The benchmark measures against the same bound the service does.** It
    resolves `max_hops` from the sidecar, exactly as the loader does, and the
    builder takes it as an argument. Re-deriving it from `max(distance)` is
    §8.1's defect in the consumer that exists to measure §8.1's code: a
    legitimate 5-hop table need contain no 5-hop row, and the two then disagree
    on the unreachable sentinel. Two implementations agreeing with each other
    cannot detect it, because both receive the same wrong bound.

---

## 6.2 The one memory figure this change can honestly report

**Measured on this machine, at 4,996,922 rows** — synthetic, de-duplicated,
built through `build_sp_index`:

| Quantity | Value |
|---|---|
| Index resident | 44.97 MB, **9.00 bytes/row** |
| The four-column layout it replaces (int32, int32, int8, int8) | 49.97 MB, 10.00 bytes/row |
| Steady-state change | **−10.0%**, before counting the retired offsets dict |

**What this establishes and what it does not.** Bytes per row is a property of
the *shape* — one int64 key plus one int8 distance — so it extrapolates, and it
confirms the arithmetic in `sp_index.py`'s docstring rather than restating it.
It says nothing about deployment scale: 5M rows is about 1/86th of the row count
`PLAN_B04.md` §10 records, and **peak** does not extrapolate the way resident
does, because the sort's permutation and its output are transient terms with
their own scaling. The `ru_maxrss` reading taken alongside this is not quoted:
it is a process high-water mark already polluted by the columns the test itself
allocated.

So: resident is measured and extrapolable; peak at deployment scale is
**pending a run on the real artifact**, and `PLAN_B04.md`'s figures remain
evidence about the design that kept the id columns, not about this one.

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

**Candidates already evidenced.** `EVIDENCE_M1_M3_hgt.json` records **ten**
checkpoints by digest and `EVIDENCE_M1_M3_gat.json` records **five** — this
section said "ten each", which is wrong for the GAT file — from a machine
in an `identical-sibling` deployment relationship, with load results and an
`in_channels` of 128 established across the family. Those digests are a
designation list, not a decision: one is chosen, its availability on the
measuring machine confirmed, and it is named here **before the integrated
readings are taken**. Not before step 7 — that step is the SP-only supplementary
figure, which needs no checkpoint, and tying it to one would reintroduce the
dependency §7.1 exists to remove.

I have **not** named one in this revision, because the evidence records digests
and loadability but no parameter count or resident size, and I will not assert a
deployment shape I have not measured. The blocking condition is therefore
*"awaiting a designated measurement subject"* — satisfiable by this project, not
by the institution.

**What is actually missing, checked rather than assumed.** Both evidence files
were re-read for this: `checkpoint_digests`, `in_channels`, `key_presence_counts`
and `load_error_categories` are all there; `parameters` and anything resident is
not. And no checkpoint file is present on the machine this is being written on —
`find . -name "*.pt"` outside `.venv` returns nothing. So designation is not
blocked on a decision, a policy or another work item. It is blocked on **one
file being where the measurement runs**, and then three numbers taken from it:

| Needed | How | Who |
|---|---|---|
| One evidenced checkpoint present on the measuring machine | copy it — the digests in `EVIDENCE_M1_M3_*.json` say which files were seen, which is not the same as which are compatible with *this* artifact set | **the operator**; the file is not in this repository and cannot be |
| A compatible load against the graph export and SP table the readings are taken over | attempt it; appearing in the evidence list is not proof of it | engineering, once the file is there |
| Its parameter count and resident size | load it and read them. **Parameter bytes are not the pipeline's RSS/UMA** — they are one term in it | engineering |
| The deployment shape it stands for, stated | write it into this section beside the digest | engineering |

**And the rest of the measurement environment has to be ready**, not only the
checkpoint: a compatible graph export, an SP artifact matching it, and the
runtime. "Only one file is missing" is true of *this* list and is not a claim
that everything else is in place — it is a claim that nothing on this list waits
on a decision.

*The search behind this: `find` over the repository and over `/home`, `/root`,
`/opt` and `/data` to depth 6, excluding `.venv`, returns no `.pt`. That covers
the paths a checkpoint would plausibly be on and does not prove the machine has
none anywhere.*

**Until the file arrives, the SP-only supplementary figure is the only memory
reading this plan can produce**, and §6.2 has it. That is not a reason to delay
the producer work, which needs no checkpoint at all — which is why the two run
in parallel rather than in sequence.

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

## 8. Four adjacent defects: two closed here, two still open

Found while reading the SP path. **Two are closed**: defect 1 landed as §10
step 0, and defect 4 was absorbed by the replacement in this plan's own change.
**Two remain open**, both in `scripts/compute_shortest_paths.py` — a different
entry point from the one this plan edits, listed so they are not lost and
excluded so this change stayed reviewable. They are the subject of
[`PLAN_SP_ARTIFACT_INTEGRITY.md`](../PLAN_SP_ARTIFACT_INTEGRITY.md).

*(This heading said "three not bundled" and was written when defect 1 was still
open. Counting it among the deferred after it shipped is the kind of arithmetic
a status line gets wrong by standing still.)*

1. **`max_hops` falls back to 5 in silence — FIXED, ahead of the indexed path
   as §10 step 0 requires.** The investigation changed the fix materially from
   what this plan first proposed, and the changes are recorded in the commit
   rather than here. **Two rounds, each refuting the other's easy answer.**
   Refusing a *missing* sidecar outright was wrong: it would have replaced a
   correct score with a different one for the 5-hop tables every operator-facing
   build path in this repository produces. But assuming 5 and merely recording
   the assumption was also wrong: the producer's CLI accepts `1..127`, so a
   legitimate 3-hop table is in support, and read against 5 it reorders
   candidates while the floor check sees nothing — the observed maximum is 3
   under either reading. What ships: a present sidecar is binding and validated
   against the producer's own `[1, 127]`; a missing one takes the bound from
   `config.sp_hop_bound`, held to the same domain and recorded as `configured`;
   with neither, shortest-path scoring stays off rather than guessing, and the
   workspace itself is not refused. A one-sided floor check refuses any bound
   the table contradicts. Separately, a configured deployment whose pipeline
   fails now refuses at `/diagnose` rather than reaching the mock generator —
   the new refusals had made loadable artifacts reachable from a path that
   answers with fabricated candidates over HTTP 200.
   Original description, kept because it is what the defect was: `_load_shortest_paths` reads
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

4. **The scan accepts an out-of-domain target and answers about a different
   one.** `sp_mean_distances` compares an int32 column against a Python int
   (`src/inference/scoring.py:372`), and torch does not refuse a scalar the
   dtype cannot hold — it wraps it. Measured on a two-row lookup with
   torch 2.10.0+cu130:

   ```
   int32 tensor([0,1,2]) == 2**32  ->  [True, False, False]
   int8  tensor([0,1,2]) == 256    ->  [True, False, False]

   target=2**32      scan=[1.0]  global=[6.0]
   target_type=256   scan=[1.0]  global=[6.0]
   ```

   So a caller asking about target `2**32` receives **target 0's real
   distance**, marked available. Approach A masks the out-of-domain id and
   returns unreachable, which is why the two disagree.

   **Why it is inside this change rather than deferred.** The reviewer's
   ruling, and the reasoning is the revision's: the implementation that has
   this defect is **not being kept**, so there is nothing to repair. Patching
   the scan to preserve a fallback would be maintaining a path this change
   exists to remove. A already masks out-of-domain ids by construction — that
   is why it answers 6.0 above — so what the replacement owes is not new logic
   but **proof through the official entry point**, which is §6.1 item 6. The
   existing equivalence test uses `[-1, 0, 10_000]`
   (`tests/unit/test_sp_index_prototypes.py:176` as it then was), none of which
   aliases, which is why the suite had never seen it. Its successor,
   `tests/unit/test_sp_index.py`, covers `2**32`, `2**31`, a negative and an
   ordinary absent id, through the official entry point.

   The same mechanism one step earlier is §6.1 item 7: the loader narrows
   int64 columns to int32/int8, and a value the narrow type cannot hold wraps
   there too. Validation moves ahead of the narrowing.

Proposal: defects 2 and 3 become one small backlog item after 5a lands.
**Defect 1 is not among them** — it is listed in §10 as step 0, a precondition
of activating the indexed path, and the draft's closing line contradicted that
by sweeping all three into the same post-5a bucket. **Defect 4 is not among them
either**, for the opposite reason: it is absorbed here.

---

## 9. What this plan does not build

No runtime A/B selector. No backend registry. No memory-budget framework. No
cache or memoisation. No second prototype. No change to the production caller's
shape. No pre-built response to a gate failure that has not happened.

---

## 10. Sequence

0. **`max_hops` sidecar defect fixed, reviewed and landed** (§8.1). Now a
   precondition of the whole refactor, not only of the wiring: the builder takes
   `max_hops` as an argument, so it has to be resolved before the index exists.
1. **This plan revised and cleared to implement.** Done — see the status block.
   The condition is recorded there rather than inferred from an approval nobody
   can locate, and "cleared to implement" is not "accepted for deployment".
2. **Preparation outside `src/`.** Done. Stale and inverted citations in
   `scripts/sp_index_prototypes.py` corrected before anything was moved, the
   file added to `_F821_CLEAN` so a split that drops a helper is visible, and
   three false statements in `BACKLOG.md` and `SP_SCORE_GUIDE.md` fixed.

**Steps 3-5 are one refactor, split for review rather than into stages that can
ship apart.** The repository is not left with two ways to build a lookup at the
end of any of them.

3. **The primitive gets one home and one shape.** `src/inference/sp_index.py`:
   `SPLookup` as the index (D2), `build_sp_index` over raw columns with a single
   sort (D4), the domain derivation and the uniqueness refusal, and
   `sp_mean_distances` fixed on A. `scoring.py` gives up the two names; the five
   import sites move. The old scan leaves production and becomes the independent
   reference §4 describes.
4. **The loader hands it raw columns.** Validation before narrowing (§6.1 item
   7), `max_hops` resolved before the expensive work rather than after the
   offsets are built, the parallel id columns and the offsets table retired, and
   `_sp_ready` still published last with the index build inside the fallible
   tail (D1).
5. **Every other construction site follows**, including the offline harness and
   the fixtures. A fixture still assembling the old shape is the second pipeline
   in miniature, so this is not optional cleanup.
6. **Mutation-check what §6.1 claims**, at minimum: the single sort, the
   uniqueness refusal, each pre-narrowing validation, the out-of-domain masks,
   and the publish-last ordering.
7. **`make check`.**
8. **Measure, and mark what cannot be measured.** The resident and peak figures
   are re-taken against the shipped shape — `PLAN_B04.md`'s numbers describe a
   design that kept the id columns and this one does not, so they are not
   evidence about it. Report: implemented; the supplementary figure recorded as
   supplementary; readings 1, 2, 3b and 4 awaiting a designated measurement
   subject (§7.1); reading 5 deferred for want of the smallest supported target.

Steps 0-8 need no calibration and **no institutional decision** — §7.1's
designation is engineering, and item 6's clinical choice is not a prerequisite
for any of it. What steps 1-7 cannot do is complete an integrated reading, which
is why the plan stops at step 8 rather than at a gate verdict.
