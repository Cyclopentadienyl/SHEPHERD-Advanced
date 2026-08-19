# B-0.4 — vectorised shortest-path lookup

**Status:** rev 2, proposed, not implemented. This plan **corrects** the
description of B-0.4 in [`PLAN_B03.md`](PLAN_B03.md) §5, which was wrong in a way
that made the work look smaller and pointed at the wrong file.

Rev 2 after review: the stage is **benchmark-gated and baseline-first** (§3), the
benchmark covers **both caller shapes** and sweeps `P` and the slice tail rather
than `C` alone (§5.4), the load-time index is built by **replacing** the loader's
existing sort rather than adding a second one (§5.2), and index-build time and
memory are part of the selection (§5.4). No representation is preferred in
advance any more.

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

## 3. What this stage decides — and what it may not assume

**B-0.4 measures whether the current lookup threatens the eager-SP latency
budget and, if so, replaces the linear scan with a measured indexed
implementation.**

An earlier revision of this plan headed this section "Why this is a latency
prerequisite, not polish" and asserted that B-0.4 is what makes the eager-SP
design affordable. **That claim was not available to make.** Operation counts are
not wall-clock evidence — least of all after §2 established that the table is on
CPU and no device-to-host synchronisation occurs, which removed the one cost that
would have been large regardless of `L`. Nothing has been timed. The plan cannot
both refuse to quote a speedup and assert that the speedup is necessary.

The interaction model it would have been a prerequisite *for* is real, and
already decided in `docs/DISEASE_SCORER_POLICY.md` §1.2:

> **"SP is computed eagerly for the entire selected set, once, when the canonical
> result is produced."** … "The clinician waits once, at inference, and then
> browses, sorts and filters with no further computation."

§1.2 also records what eager computation buys structurally: it makes conditions
C7 and C8 true by construction, and removes any need for request identity,
stale-response rejection or cache invalidation on the SP path. The institution
then required the candidate list to reach **200+** with pagination, SP sorting
and filtering (policy record, rev 3 note), so that single wait contains ~4,000
slice scans at 20 phenotypes. **Whether that is 20 ms or 2 s is exactly what is
unknown**, and it is what the baseline run in §5.4 answers first.

**Two open institutional values bound the claim.** `selection_limit` and the
interaction latency target are both `[OPEN]` (policy §1.3). A provisional
engineering budget may be used to interpret the curves **provided it is labelled
non-institutional**; final acceptance uses the institutional target when it is
supplied. B-0.4 is not cancelled or postponed because that target is still open —
the baseline measurement is useful to the institution deciding it.

### 3.1 Shipping no replacement is a legitimate outcome

Stated so the gate is not a formality. If the baseline shows the current
implementation comfortably inside any plausible budget across the §5.4 matrix,
**B-0.4 ships the benchmark, the recorded curves and nothing else.** The
correction to `PLAN_B03.md`, the measurement and the recorded distribution are
the deliverable in that case; the index is not built. A stage that measures and
declines to act on the measurement has done its job, and it is a cheaper mistake
than optimising a cost nobody has seen.

---

## 4. Scope

### In

1. **Baseline first.** Run the current implementation across the §5.4 matrix
   before any index is prototyped. This is the deliverable that exists whatever
   else does (§3.1).
2. **Prototype both representations** (§5.2) and measure them on **both caller
   shapes** — the singleton loop production actually ships, and the batched call
   B-1 and the offline harness will use.
3. **Vectorise the body of `sp_mean_distances`** *if the baseline warrants it*.
   Signature unchanged, results unchanged, float64 contract unchanged.
4. **Equivalence test**: the vectorised implementation and the current one agree
   exactly on the same inputs, including the unreachable, empty-phenotype and
   empty-candidate paths.
5. **Correct `PLAN_B03.md` §5** and its stale citation.

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

**Whatever improvement exists is available without it.** Even at `C = 1` per
call, each call's `P` linear scans become `P` binary searches, so the `L → log L`
change in asymptotic work is reachable in the shape production already has.
Batching the caller adds only the removal of call overhead. *Asymptotic* is the
operative word: whether that translates into a benefit worth shipping is what
§5.4 measures, not something this section may assert (§3).

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

Build the ordering **once at load** — the same principle as §1.2 applied one
level down: pay at load, not per request.

**One sort, not two.** An earlier revision said "beside the existing offsets",
which implies adding a second full-table sort to the one the loader already does.
Review's alternative is better and is adopted: **replace** the loader's
phenotype-only sort key with a single composite/lexicographic ordering that does
all four jobs at once —

| Job | How the single sort delivers it |
|---|---|
| Group by phenotype | Phenotype is the most significant component, so runs stay contiguous |
| Order targets within a phenotype | `(target_type, target)` are the less significant components |
| Derive offsets | Unchanged: run boundaries in the phenotype column (`pipeline.py:509-519`) |
| Detect duplicates | Adjacent equal composite keys — one comparison over the sorted key vector |

That keeps the loader at today's sort count and today's peak-memory shape, which
matters because the existing loader compacts tensors incrementally (`del ph_t`,
`del tg_t`, … at `pipeline.py:497-505`) precisely to control peak RAM.

Two candidate representations, both already named in the primitive's docstring:

| Candidate | Shape |
|---|---|
| **A — global composite key** | One monotone int64 key over the whole table; a query becomes one `torch.searchsorted` over `P × C` keys, then one gather and one masked reduction |
| **B — per-phenotype sorted slices** | Keep the existing offsets and `searchsorted` within each phenotype's slice — `P` searches per call over `log L` each |

**No representation is preferred here, and the earlier revision was wrong to
prefer A.** A's advertised advantage — one search over `P × C` keys, a constant
number of launches — **only exists when the caller supplies all `C` candidates at
once.** B-0.4 deliberately retains the singleton caller (§4.1), where the real
workload is `C` separate searches over `P` keys each. Selecting a representation
from a batched benchmark would pick for a workload this stage does not ship.
Hence both shapes in §5.4.

If the two shapes prefer different representations: take the measured compromise,
or leave the batched-specific optimisation to B-1, which is where the batched
caller arrives. **Prototype both; productionise one.** No strategy hierarchy, no
backend registry, no runtime selector.

A's one construction risk: the composite key must not overflow int64, which is
checked against the real table's node counts — available in the artifact's
`.meta.json` sidecar (`compute_shortest_paths.py:409-417`) — rather than assumed.

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

A small standalone script matching the repository's existing convention —
`scripts/benchmark_attention.py` prints a JSON-shaped result and is nothing more.
**Not a framework.** Nested loops over a fixed matrix, printing rows.

**The matrix.** An earlier revision proposed a curve over `C` alone at one
representative `P`. That does not cover the cost, which is `C × P × L`:

| Axis | Values |
|---|---|
| `C` candidates | 10, 50, 100, 200, and a **stated provisional ceiling** |
| `P` phenotypes | 1, 20, 100 — the API's contractual range (`api/routes/diagnose.py:56`) |
| Slice distribution | representative, and dense-tail |
| Caller shape | singleton loop (`C` calls of one target), and one batched call of `C` |
| Implementation | current, prototype A, prototype B |

The ceiling is **stated, not "the configured upper bound"** — `selection_limit`
has no configured value yet (policy §1.3, `[OPEN]`), and writing as though it did
would manufacture an institutional number. It is labelled non-institutional
(§3).

Report the **median over repeated runs and a high percentile or the worst
repeated sample**, not a mean. Report the two caller shapes as separate curves;
**do not average or collapse them into one speedup**.

**Slice distribution — measured, not assumed.** The `L` tail is the axis most
likely to be misrepresented by a uniform synthetic shape, and the metadata
sidecar does not record it: `compute_shortest_paths.py:409-417` writes
`num_pairs`, `num_phenotypes`, `num_genes`, `num_diseases`, `kg_total_nodes`,
`kg_total_edges` and `max_hops`, from which only the **mean** `L` is derivable.

Rather than add distribution statistics to the builder — which would only help
artifacts built afterwards, not the one already deployed — **the benchmark
derives the distribution from an artifact directly when one is present.** The
slice sizes *are* the run lengths of the phenotype column, which is what the
loader already computes to build offsets (`pipeline.py:509-519`), so this needs
no new recording anywhere and works on the deployed table as it stands. When no
artifact is available the benchmark falls back to a declared synthetic
distribution, and **its output states which of the two it used**.

No SP artifact exists in this development container, so the first runs here are
synthetic by necessity.

**Index-build cost is part of the decision, not a footnote.** For each prototype
the benchmark reports: table row count; index construction time; transient peak
memory; steady-state added memory; the query curves for both caller shapes; and
the duplicate-check cost. A representation that wins at query time and doubles
startup memory has not obviously won.

**Runs on CPU**, because the table is on CPU (§2). Unlike the measurement
harness, it is **not gated on institutional CUDA hardware**.

### 5.5 Provenance of any curve used for an institutional decision

A development CPU run is valid for **correctness and relative algorithm
comparison**. It is not valid for choosing `selection_limit`. A curve used for
that must be re-run with:

- deployment-equivalent CPU;
- the real SP artifact, or its verified distribution;
- recorded CPU model, torch version and thread count;
- the artifact's fingerprint and statistics;
- the warmup and repeat configuration.

CUDA is not required for any of it.

### 5.6 What the loader change touches

Adopting the single composite sort (§5.2) edits `_load_shortest_paths`
(`pipeline.py:496-545`) — **production code, though not a production caller.**
The distinction §4.1 draws is between the primitive and its *call sites*; the
loader is neither, and it is in scope.

One consequence to name rather than discover: reordering `_sp_tg`, `_sp_ty` and
`_sp_di` changes attributes the pipeline's own comment calls *"part of this
class's observable surface"* (`pipeline.py:533-536`). **Results are invariant**,
because uniqueness (§5.3.2) means exactly one row answers any
`(phenotype, target, target_type)` query regardless of storage order — which is
another reason that assertion is load-bearing rather than defensive. What changes
is the order a direct reader of those attributes would see.

---

## 6. Acceptance

**Unconditional:**

- the baseline of the current implementation is recorded across the whole §5.4
  matrix, on both caller shapes, with the slice distribution's provenance stated;
- `PLAN_B03.md` §5 no longer describes B-0.4 as a caller change;
- **no production caller, scorer, transform or policy behaviour changes.**

**If, and only if, the baseline warrants a replacement** (§3.1):

- the vectorised primitive agrees exactly with the current one on the same
  inputs, including every path in §5.3;
- the uniqueness assertion fires on a table that violates it, proven by a test
  that constructs one;
- both prototypes are measured on both caller shapes, with index-build time and
  memory reported, and **one** is productionised;
- `tests/unit/test_scoring_primitives.py` continues to pass unchanged, since the
  contract it tests is unchanged.

---

## 7. Known unknowns

- **Whether the current implementation is already fast enough.** That is the
  question §3 now says this stage asks rather than assumes. Nothing is timed.
- **Whether approach A or B wins, and on which caller shape.** A's batched
  advantage does not transfer to the singleton caller this stage retains, so the
  two shapes may disagree. No speedup figure appears anywhere in this plan.
- **The interaction latency target and `selection_limit`**, both `[OPEN]`
  institutional values (policy §1.3). A provisional engineering budget reads the
  curves in the meantime and is labelled non-institutional.
- **The `L` distribution**, which the metadata sidecar does not record. §5.4
  measures it from an artifact when one is present; the first development runs
  have none and are synthetic.
- **The real table's node counts**, needed for A's int64 key bound. Read from the
  artifact's `.meta.json` at implementation time, not assumed.

**Authority above everything here:** `docs/DISEASE_SCORER_POLICY.md`.
