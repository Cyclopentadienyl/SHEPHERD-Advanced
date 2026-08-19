# B-0.4 — vectorised shortest-path lookup

**Status:** proposed, not implemented. This plan **corrects** the description of
B-0.4 in [`PLAN_B03.md`](PLAN_B03.md) §5, which was wrong in a way that made the
work look smaller and pointed at the wrong file.

---

## 1. What PLAN_B03 §5 got wrong

It said:

> The vectorisation is therefore a **caller change in production, not a new
> primitive**.

That conflated two different things. `sp_mean_distances` does take a sequence of
targets and return a `(C,)` tensor — the **signature** is final. But the
**body** is a Python double loop, and the function's own docstring says so
(`src/inference/scoring.py:289`):

> **"The interface is batched; this implementation is not yet vectorised."**

So the primitive does need replacing. §5 also cited `pipeline.py:1399` as the
call site; that line is now the pipeline config dict. The real path is
`pipeline.py:1068` → `_calculate_sp_score` (`:1090`) → `sp_mean_distances`
(`:1147`) → `[0]` (`:1152`).

Both corrections land in this plan rather than by editing §5's claim away: the
plan history is what stops the same mistake being made a third time.

---

## 2. What the cost actually is

Verified in the tree, not estimated.

```python
# src/inference/scoring.py:311-326
for target_idx in target_indices:            # C candidates
    for ph_idx in phenotype_indices:         # P phenotypes
        target_slice   = lookup.target[start:end]        # length L
        type_slice     = lookup.target_type[start:end]
        distance_slice = lookup.distance[start:end]
        match = (target_slice == target_idx) & (type_slice == target_type_idx)
        hits  = distance_slice[match]
        total += float(hits[0]) if len(hits) > 0 else unreachable
```

Every (candidate, phenotype) pair **linearly scans that phenotype's whole
slice**. Cost is `O(C × P × L)`, with roughly six tensor operations, one boolean
mask allocation and one Python scalar extraction per iteration.

The docstring's own counts: **~4,000 slice scans at 200 candidates × 20
phenotypes**, and **over 550,000 at full-universe scale**.

**The table is on CPU.** `_load_shortest_paths` concatenates the offset tensors
with explicitly-CPU tensors (`pipeline.py:510-511`), which would fail otherwise,
and nothing moves the three parallel tensors to a device. So `float(hits[0])` is
a CPU scalar extraction, **not** a device→host synchronisation. That was an open
question when this work was described in conversation; it is closed, and it
lowers the expected win from removing it. The `O(L)` scan remains the cost.

**The current caller makes `C = 1`.** `_calculate_sp_score` passes one target and
takes `[0]`, so the outer loop runs once and the function is called once per
candidate. Total arithmetic is identical either way; what the per-candidate shape
adds is `C` function calls and `C` two-element tensor allocations. **That is the
small part.** The large part is `L → log L`, and it is entirely inside the
primitive.

---

## 3. Why this is a latency prerequisite, not polish

`docs/DISEASE_SCORER_POLICY.md` §1.2 already decided the interaction model:

> **"SP is computed eagerly for the entire selected set, once, when the canonical
> result is produced."** … "The clinician waits once, at inference, and then
> browses, sorts and filters with no further computation."

§1.2 also records what eager computation buys structurally: it makes conditions
C7 and C8 true by construction, and removes any need for request identity,
stale-response rejection or cache invalidation on the SP path.

The institution then required the candidate list to reach **200+** with
pagination, SP sorting and filtering (policy record, rev 3 note). At 200
candidates and 20 phenotypes that one-time wait contains ~4,000 slice scans.
**B-0.4 is what makes the §1.2 decision affordable**; without it the single wait
may be long enough that the design stops delivering what it was chosen for.

---

## 4. Scope

### In

1. **Vectorise the body of `sp_mean_distances`.** Signature unchanged, results
   unchanged, float64 contract unchanged.
2. **Equivalence test**: the vectorised implementation and the current one agree
   exactly on the same inputs, including the unreachable, empty-phenotype and
   empty-candidate paths.
3. **Benchmark** reporting cost as a **curve over C**, not a single number.
4. **Correct `PLAN_B03.md` §5** and its stale citation.

### Out

- **The production caller keeps its per-candidate shape.** See §4.1.
- The SP transform `1/(1+d)`, the value of η, and what SP is used for. Those are
  `DISEASE_SCORER_POLICY.md`'s, and B-1's.
- Full-universe SP. Policy §4 rejected it as an alternative; nothing here revives
  it.
- Any cache, memoisation or precomputed result store. The index in §5 is built
  once at load, which is not a cache: it has no invalidation, no keys and no
  lifetime.

### 4.1 Why the caller change is deferred to B-1

Two reasons, and the second is the decisive one.

**The benefit does not need it.** Even keeping `C = 1` per call, each call's `P`
linear scans become `P` binary searches. The `L → log L` win lands immediately.
Batching the caller only removes call overhead on top of that.

**B-1 restructures that caller anyway.** Policy §2 records SP's ranking role as
"Not implemented — B-1", and §1.2 moves SP to eager enrichment over the selected
set — **which is the batched call**. Changing the caller now means changing it
twice, and the intermediate shape would be neither what exists nor what B-1
needs.

The offline harness is not bound by this: it can pass a full candidate list from
its first line, because it has no legacy caller to preserve.

---

## 5. Implementation

### 5.1 What the data structure already gives

`SPLookup` (`scoring.py:84-101`) is CSR-style: three parallel tensors plus
`offsets: Dict[int, (start, end)]` per phenotype. Two properties matter:

- **The table is already grouped by phenotype.** `_load_shortest_paths` sorts by
  phenotype index and derives contiguous offsets (`pipeline.py:496-519`).
- **Exactly one row exists per `(phenotype, target, target_type)`** — the offline
  BFS records a node the first time it is reached, which is its minimum distance
  (`scoring.py:91-94`, citing `scripts/compute_shortest_paths.py:79-89`). This is
  what makes "take the first match" unambiguous today, and it is what makes a
  binary search correct tomorrow.

**Within a phenotype's slice, `target` is not sorted.** The load-time sort keys
on phenotype alone, so slice-internal order is BFS discovery order permuted by an
unstable `argsort`. A binary search therefore needs an ordering the table does
not currently have.

### 5.2 Approach

Build the ordering **once at load**, beside the existing offsets — which is the
same principle as §1.2 applied one level down: pay at load, not per request.

Two candidate representations, both already named in the primitive's docstring:

| Candidate | Shape |
|---|---|
| **A — global composite key** | Sort the whole table by `(phenotype, target_type, target)` into one monotone int64 key; a query batch becomes one `torch.searchsorted` over `P × C` keys, then one gather and one masked reduction |
| **B — per-phenotype sorted slices** | Keep the existing offsets, sort each slice by `(target_type, target)`, and `searchsorted` within slices |

**A is expected to win** — it collapses the whole computation to a constant
number of kernel launches rather than `P` of them — but the choice is made by the
benchmark in §4.3, not here. A's one risk is key construction: the composite must
not overflow int64, which needs checking against the real table's node count
rather than assumed.

### 5.3 Invariants the replacement must hold

Named here because each is a way the change could silently alter results:

1. **float64, for the whole computation.** The docstring (`scoring.py:295-300`)
   already forbids reducing in float32 and widening afterwards. Inherited as-is.
2. **Uniqueness must be verified, not assumed.** Sorting by a composite key makes
   duplicate `(phenotype, target, target_type)` rows adjacent, and a binary
   search would return the first *in sorted order* — which need not be the first
   in original order. If the table ever violated the uniqueness the docstring
   asserts, first-match and searchsorted-match could disagree on the distance.
   **The index build asserts uniqueness and fails loudly**, rather than
   preserving original-order semantics for a case the data model says cannot
   occur.
3. **Unreachable handling.** A phenotype with no row for a candidate contributes
   `unreachable_distance`, and a phenotype absent from `offsets` entirely does
   too. Both are misses in the vectorised form and must produce the same value,
   not be dropped from the mean.
4. **`available` semantics stay as narrow as they are.** False only for no
   phenotypes or no candidates. A far-away candidate is available with a large
   distance. The primitive does not acquire the typed status contract, which
   belongs to B-1.

### 5.4 Benchmark

Matching the repository's existing convention — `scripts/benchmark_attention.py`
is a small standalone script that prints a JSON-shaped result — and not a
framework.

- Sweeps `C` across the range the institution's decision actually spans, at a
  representative `P`, reporting old and new.
- **Runs on CPU**, because the table is on CPU (§2). Unlike the measurement
  harness, it is not gated on institutional CUDA hardware and can be run
  anywhere, including in a development container.
- **The output is an input to an open institutional decision.** `selection_limit`
  is marked `[OPEN]` in policy §1.3 and awaits an institutional value. A cost
  curve over `C` is what that decision needs; a single speedup number is not.

---

## 6. Acceptance

- The vectorised primitive agrees exactly with the current one on the same
  inputs, including every path in §5.3;
- the uniqueness assertion fires on a table that violates it, proven by a test
  that constructs one;
- the benchmark reports a curve over `C` for both implementations;
- `tests/unit/test_scoring_primitives.py` continues to pass unchanged, since the
  contract it tests is unchanged;
- `PLAN_B03.md` §5 no longer describes B-0.4 as a caller change;
- **no production caller, scorer, transform or policy behaviour changes.**

---

## 7. Known unknowns

- **Whether approach A or B wins**, and by how much. Not asserted before
  measurement. No speedup figure appears anywhere in this plan for that reason.
- **The real table's node count**, needed for A's int64 key bound. Read at
  implementation time from the deployed artifact's metadata, not assumed.
- **Whether `L` varies enough across phenotypes** for the linear scan's cost to
  be dominated by a few dense phenotypes. If it does, the mean-case benchmark
  understates the tail, and the curve should carry a worst-case phenotype too.

**Authority above everything here:** `docs/DISEASE_SCORER_POLICY.md`.
