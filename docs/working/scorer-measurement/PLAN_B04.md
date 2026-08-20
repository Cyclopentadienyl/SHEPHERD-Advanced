# B-0.4 — vectorised shortest-path lookup

**Status:** rev 7. **Both gates are cleared and the stage is ready to
prototype.**

- **§3.1, the latency gate — answered.** On the real artifact and a GB10 SPARK,
  the hardware class the institution names as its primary edge deployment
  platform, the current lookup **exceeds the provisional budget by 1.7-2.5x**
  (§9). A replacement is warranted.
- **§5.3.2, the compatibility gate — passed.** Zero duplicate rows on two
  independently built artifacts (§10), so the uniqueness assertion will not
  refuse to start.

**No index prototype is built and no production code has changed.** The plan
also **corrects** the description of B-0.4 in [`PLAN_B03.md`](PLAN_B03.md) §5,
which was wrong in a way that made the work look smaller and pointed at the
wrong file.

Rev 2 after review: the stage is **benchmark-gated and baseline-first** (§3), the
benchmark covers **both caller shapes** and sweeps `P` and the slice tail rather
than `C` alone (§5.4), the load-time index is built by **replacing** the loader's
existing sort rather than adding a second one (§5.2), and index-build time and
memory are part of the selection (§5.4). No representation is preferred in
advance any more.

Rev 3 after review: a synthetic run **may not** close the stage with "no
replacement" (§3.1); the int64 key domain is derived from the tensors rather than
from the optional metadata sidecar (§5.2); the claim that one composite sort
preserves peak memory is **withdrawn** — sort count only, memory measured
(§5.2, §5.4); the uniqueness assertion gains a **real-artifact compatibility
gate** because it changes startup behaviour (§5.3.2); prototyping is conditional
on the gate rather than unconditional (§4); and the index is described by what it
is for rather than by a false claim that it has no keys (§4).

Rev 4 after review of the baseline run: `--artifact` now times the artifact's
**own slices** instead of a synthetic table parameterised by its mean, and the
two modes are separate code paths (§8, `scripts/benchmark_sp_lookup.py`); the
raw evidence is committed and referenced by SHA-256, and the false
"reproduces the raw JSON exactly" claim is replaced (§8); the non-worst-case
phenotype selection is a seeded `randperm` subset rather than a fixed prefix
called random (§8.2); and the caller conclusion is narrowed to the current
primitive rather than ruling the question out for B-1 (§8.1).

Rev 5 after review: the measurement-order alternation was **claimed but never
executed** — it branched on `len(rows) % 2`, which is even at every cell
boundary — so it is now counted per timed cell, pinned by a regression test, and
the whole 120-cell / 240-row matrix re-run with every §8 aggregate recomputed. The artifact
verdict also no longer implies acceptance: a real artifact is one of §3.1's two
requirements and the deployment-equivalent CPU is the other, which this script
cannot self-attest and must not gain a flag to fake.

Rev 6: the artifact run landed (§9). The synthetic sweep understated the real
mean slice length by 2.2x and its `dense_tail` shape does not resemble the
artifact, whose p100 is only 1.17x its p50; the caller-shape conclusion holds at
a median ratio of 1.001; the baseline misses the provisional budget by 1.7-2.5x;
and 430 million rows makes index-build memory a first-order criterion that may
invert §5.2's expected ordering.

Rev 7: the §5.3.2 compatibility gate **passed** on a second, independently built
artifact — zero duplicates, non-negative ids, and a composite-key maximum seven
orders of magnitude inside int64 (§10). The two tables differ because their HPO
vintages do, which reframes the gate as evidence about the generator's invariant
rather than about one file, and makes the load-time assertion the right ongoing
mechanism. Measured index-build memory came in at roughly three times §9.4's
estimate, on unified memory shared with the model.

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

### 3.1 Shipping no replacement is a legitimate outcome — under a stated gate

Stated so the gate is not a formality: a stage that measures and declines to act
on the measurement has done its job, and that is a cheaper mistake than
optimising a cost nobody has seen.

**But an earlier revision let a synthetic run close the stage**, which
contradicts what the same plan says about the slice tail being the axis a
synthetic distribution is most likely to misrepresent (§5.4). The decision rule,
not a governance framework:

| Run | May establish | May **not** establish |
|---|---|---|
| Synthetic, development CPU | That the benchmark works; relative comparison between implementations; gross regressions | That the **deployed** baseline is fast enough |
| Deployment-equivalent CPU **and** the real artifact — or a distribution verified from it | Everything above, plus the baseline verdict | — |

- **Closing B-0.4 with "no replacement" requires the second row.**
- **If no real artifact is available, the valid outcome is "benchmark complete;
  production replacement decision pending institutional run"** — not "baseline
  accepted".
- **If the institutional latency target is still open, a provisional
  non-institutional decision threshold is stated *before* the results are
  examined.** Declaring the threshold afterwards is choosing the verdict.

No SP artifact exists in this development container (§5.4), so on today's
evidence the expected outcome of a run here is the pending one.

---

## 4. Scope

### In

1. **Baseline first.** Run the current implementation across the §5.4 matrix
   before any index is prototyped. This is the deliverable that exists whatever
   else does (§3.1).
2. **Only if the §3.1 gate warrants a replacement:** prototype both
   representations (§5.2) and measure them on **both caller shapes** — the
   singleton loop production actually ships, and the batched call B-1 and the
   offline harness will use.
3. **Then vectorise the body of `sp_mean_distances`**, again only under that
   gate. Signature unchanged, results unchanged, float64 contract unchanged.
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
- **No request-result cache, no persistent memoisation, no cache-invalidation
  subsystem.** A derived lookup index built with the artifact at load time **is**
  in scope. An earlier revision drew this line by claiming the index "has no
  keys and no lifetime", which is simply false — a lookup index is keys, and it
  lives as long as the loaded artifact. The boundary is what it is for, not a
  taxonomy.

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

**That keeps the loader at one full-table sort. It does not establish unchanged
peak memory**, and an earlier revision claimed it did. Constructing an int64
composite key allocates another full-length tensor, and the arithmetic
intermediates and duplicate-check vectors are further allocations; peak transient
memory can rise even with the sort count unchanged. It matters that it is
measured rather than argued, because the existing loader compacts tensors
incrementally (`del ph_t`, `del tg_t`, … at `pipeline.py:497-505`) precisely to
control peak RAM. **Peak memory is a measured selection criterion (§5.4), not a
claim of this section.**

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

**A's key domain comes from the tensors, never from the sidecar.** An earlier
revision proposed bounding the int64 key using the node counts in
`shortest_paths.meta.json`. **That would make correctness depend on an optional
file.** The loader treats the sidecar as optional and swallows any failure
reading it (`pipeline.py:523-531`), and a sidecar can additionally be stale or
mismatched with the `.pt` it sits beside. So:

- validate that phenotype, target and target-type ids are **non-negative**;
- derive the strides from the **actual tensor maxima**;
- compute the largest possible key in **Python integers**, which are arbitrary
  precision, and check it against the int64 bound **before** any int64 key tensor
  is constructed — an overflow check performed in int64 has already overflowed;
- use the sidecar's counts only as provenance, or as an optional consistency
  check that does not gate correctness.

The sidecar's existing role — supplying `max_hops`, and through it
`unreachable_distance` — is unchanged and out of scope here.

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

   **This changes startup behaviour, so it needs a compatibility gate against
   real artifacts, not only a synthetic test.** A table that today's first-match
   implementation loads happily would make the new loader refuse to start.

   **The gate is evidence about the generator, not clearance for one file
   (§10.1).** The knowledge base is updated on purpose — that is a project
   feature, not drift — so there is no single "deployed artifact" to clear:
   there is a sequence of them, and the institution's will be a vintage nobody
   has built yet. What the gate establishes is that real tables satisfy the
   invariant, which is a property of `compute_shortest_paths.py`'s BFS rather
   than of any one build. **Satisfied by two independently built artifacts from
   different HPO vintages, both with zero duplicates.**

   The ongoing guarantee is the load-time assertion itself, which runs on
   whatever table is present. If a future rebuild ever violates uniqueness it
   fails at startup rather than scoring wrongly — and **B-0.4 does not invent a
   deduplication or migration policy** for that case.
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

**Memory must be measured at the process level.** `tracemalloc` sees Python
allocations and **not** PyTorch's CPU tensor storage, which is allocated by the
C++ allocator — it would report a fraction of the real footprint. Peak RSS
(`resource.getrusage(...).ru_maxrss`, or `VmHWM` from `/proc/self/status`) sees
it. Still not a profiling framework: two reads and a subtraction.

*One trap that would silently invalidate those numbers:* **peak RSS is a
process high-water mark that never decreases.** Measuring the baseline and both
prototypes in one process would report the highest of the three for all three,
and the first one measured would look best purely from ordering. **Each
implementation's memory is therefore measured in its own subprocess.** Timing
curves have no such constraint and may share a process.

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

- **the verdict is stated at the honest strength the run supports** (§3.1): with
  no real artifact, "benchmark complete; production replacement decision pending
  institutional run", never "baseline accepted";
- any provisional decision threshold was **recorded before** the results were
  examined.

**If, and only if, the §3.1 gate warrants a replacement:**

- the vectorised primitive agrees exactly with the current one on the same
  inputs, including every path in §5.3;
- the uniqueness assertion fires on a table that violates it, proven by a test
  that constructs one;
- **the deployed artifact's duplicate count is recorded and is zero**, or
  deployment compatibility is declared pending (§5.3.2);
- the int64 key domain is derived and bounds-checked from the tensors, with the
  check performed in Python integers before any int64 key tensor exists (§5.2);
- both prototypes are measured on both caller shapes, with index-build time and
  per-subprocess memory reported, and **one** is productionised;
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
- **The key domain**, needed for A's int64 bound. Derived and bounds-checked
  from the **loaded tensors** (§5.2) — non-negative id validation, strides from
  the actual maxima, and the largest possible key checked in Python integers
  before any int64 key tensor exists. An earlier revision said the counts come
  from the `.meta.json` sidecar; that contradicted §5.2 and would have made
  correctness depend on an optional file.

---

## 8. Baseline results — synthetic, development CPU

`scripts/benchmark_sp_lookup.py`, seed 0. **120 timed cells, 240 result rows, 0 skipped** — the matrix is 2 mean lengths x 2 distributions x 2 phenotype selections x 3 phenotype counts x 5 candidate counts, and each cell emits one row per caller shape.

**Evidence:** [`EVIDENCE_B04_baseline_synthetic.json`](EVIDENCE_B04_baseline_synthetic.json)
— the full 240 rows with repeat counts, queried-slice totals, measurement order
and provenance, so the ratios below can be audited rather than taken.
SHA-256 `b0ce086171a84705055c1380029fd24599549df0ff6126450aa3ee8f491f22c3`.

**Reproducibility, stated accurately:** the seed reproduces the same workload;
**timing observations are expected to vary.** An earlier revision claimed the
script "reproduces the raw JSON exactly", which is false — `perf_counter`
readings, OS scheduling, CPU state and the adaptive repeat counts all move. That
is why the evidence file is committed rather than regenerated on demand.

**Verdict, at the strength §3.1 permits: *synthetic sensitivity sweep complete;
production replacement decision pending institutional run.* No SP artifact
exists here, so this run may not accept the deployed baseline and does not.**

### 8.1 No caller-only optimisation is justified for the current primitive

Median `singleton / batched` ratio across 120 configurations: **1.029** (range
0.70–1.30). Measurement order alternates per timed cell and is recorded in
`measured_first` — 60 cells each way, represented by 120 rows carrying each
value — so the difference is not confounded with
a fixed order.

*It did not, in the first run.* The alternation was written as
`if len(rows) % 2`, and each cell appends two rows, so `len(rows)` is even at
every cell boundary and the branch never fired: all 240 rows recorded
`measured_first="singleton"` while this section claimed the order was alternated.
Fixed with an explicit per-cell counter, pinned by a regression test verified to
fail against the old condition, and the whole matrix re-run — the figures here
are from that re-run. The visible effect is the range widening from 0.87–1.30 to
0.70–1.30, in the low-work cells where one call's overhead is a large share of
the total.

**This is close to a structural identity, and must not be read as a finding
about caller design.** `sp_mean_distances` loops over candidates *inside* the
function, so the singleton caller runs `C` calls of a one-iteration loop and the
batched caller runs one call of a `C`-iteration loop. **The same Python loop
executes either way; only its position moves.** The measurement therefore says
one thing:

> **No caller-only optimisation is justified in B-0.4 for the current
> primitive.**

It does **not** say caller restructuring is ruled out for B-1, and an earlier
revision of this section claimed exactly that. It predicts nothing about
prototype A, prototype B, or B-1's eager selected-set enrichment — under a
vectorised primitive the two shapes stop being the same work. B-1 need not
revisit batching as a 3% micro-optimisation, but it may well use a batched
primitive because eager enrichment naturally produces one SP vector over the
selected set. That is a design consequence, not an optimisation.

### 8.2 The selection axis needed correcting twice

**First defect.** The benchmark reported `dense_tail` as **0.93×** the cost of
`representative` — cheaper, the opposite of that axis's purpose. Cause: the query
used phenotypes `0..P-1`, drawing from the lognormal near its median, which sits
below its mean. The *table* had a heavy tail; the *query* never touched it.

**Second defect.** The fix added a selection axis but drew the non-worst case as
that same fixed prefix while calling it `random`. It is now `sampled`: a genuine
seeded `randperm` subset, and named for what it is — **one sensitivity example,
not an estimate of typical selection.** Estimating typical selection needs a
bounded set of seeds and belongs to the artifact run.

Corrected sensitivity:

| Distribution | `longest / sampled`, median | max |
|---|---|---|
| representative | 1.03× | 1.31× |
| **dense_tail** | **2.19×** | **3.42×** |

### 8.3 The verdict hinges on two unknowns — and they are not the same kind

At the pre-declared provisional budget point — **C = 200, P = 20, 250 ms,
non-institutional**, recorded in the script before the run — 14 of 16
configurations pass and 2 breach:

| Configuration | median |
|---|---|
| L = 1,000, any distribution or selection | 102 – 171 ms |
| L = 10,000, `sampled` selection, either distribution | 161 – 195 ms |
| L = 10,000, `representative` × `longest` | 200 – 203 ms |
| **L = 10,000, `dense_tail` × `longest`** | **453 – 473 ms** ← over |

The two unknowns that decide it are of **different kinds**, and an earlier
revision wrongly called both artifact properties:

| Unknown | Kind | Measured from |
|---|---|---|
| Mean and tail slice length | **artifact property** | the deployed artifact alone (§5.4) |
| Whether a patient's phenotypes correlate with slice length | **artifact–cohort relationship** | the artifact *and* a real cohort together |

The distinction matters operationally: the first is answered by pointing the
benchmark at `shortest_paths.pt`, the second is not answerable without patient
phenotype sets, and §8.2 shows it carries a 2.19× median swing.

**This is the useful outcome of a baseline stage.** It converts "we should
probably optimise this" into two specific measurements that decide it, and it
settles the caller question for *this* primitive without overreaching into B-1's.


---

## 9. Artifact results — the real table, on the primary deployment platform

Run on a **GB10 SPARK (aarch64)** — the hardware *class* the institution states
is its **primary edge deployment platform**. §3.1's second condition, a
deployment-equivalent CPU, is satisfied **for that class** and is asserted by a
person, as the script refuses to self-attest it.

*Class, not machine.* The run was made on the author's own SPARK rather than on
an institutional host. For a CPU-bound lookup on identical silicon that is what
"deployment-equivalent" means, but a different SPARK will differ in thread count,
thermal behaviour and load, so a second machine's numbers are a second data point
rather than a contradiction. An earlier revision of this section said "the
deploying institution's SPARK", which claimed the machine and not just the class.

| Provenance | |
|---|---|
| Artifact | `shortest_paths.pt`, SHA-256 `9ada0c1aa16510f7c55c71d5e3eab01b48fd9ce165a63ad07f352bd29994d4df` |
| Host | aarch64, `Linux-6.17.0-1029-nvidia`, torch `2.10.0+cu130`, 20 CPU threads |
| Cells | 60 timed cells, 120 rows, 0 skipped |
| Evidence | [`EVIDENCE_B04_artifact_spark.json`](EVIDENCE_B04_artifact_spark.json), SHA-256 `53a63df1be7d49661a40a3826f94f951eeeadefec4a680ebcf2a542603b4102f` — transferred from the deployment host and **verified against the hash recorded there** before committing |

**Note the platform scope.** The SP lookup is CPU-bound, and the institution names
three deployment targets with different CPUs. **This measurement governs the
primary one.** Whether `selection_limit` is set per platform or by the slowest is
an institutional decision, not this plan's.

### 9.1 The table is far larger and far flatter than the synthetic sweep assumed

| | Value |
|---|---|
| Rows | **429,971,678** |
| Phenotypes | 19,540 |
| Disease targets | 23,640 |
| Mean slice length | **22,005** |
| p50 / p90 / p99 / p100 | 24,518 / 26,380 / 27,412 / **28,580** |

**Two of §8's assumptions are refuted by this.**

*The sweep's ceiling was too low.* §5.4 declared synthetic mean slice lengths of
1,000 and 10,000. The real mean is **22,005 — 2.2× beyond the top of the sweep.**

*The tail hypothesis does not hold here.* **p100 is only 1.17× p50**, and the mean
(22,005) sits *below* the median (24,518): the distribution is flat and
left-skewed, not heavy-tailed. §8.2's `dense_tail` shape does not resemble this
artifact; `representative` is much closer. Slice totals for `longest` exceed
`sampled` by only **1.24×**, against the 2.19× cost ratio the synthetic sweep
produced.

**The selection axis is therefore not separable in timing at these repeat
counts**, and no claim is made from it: the `longest` cells at the budget point
show a max/median spread of 1.55 on three repeats, which is wider than the
difference being compared. What the artifact does settle is the *shape*, and the
shape says the axis matters far less here than the synthetic sweep implied.

### 9.2 The caller-shape conclusion holds on real hardware

**Median `singleton / batched` = 1.001** over 30 configurations, against 1.029 on
synthetic. The range is wider (0.62–1.91), and the noise metric explains it:
**median `max/median` per cell is 1.01, worst 1.82.** Most cells are very stable;
the extremes sit in the low-repeat cells. §8.1's conclusion — no caller-only
optimisation is justified for the current primitive — survives contact with the
real table.

### 9.3 The gate: the baseline does not meet the provisional budget

At the pre-declared point — **C = 200, P = 20, 250 ms, non-institutional**:

| Selection | singleton | batched |
|---|---|---|
| `sampled` | **629.5 ms** | **629.6 ms** |
| `longest` | **428.4 ms** | **478.5 ms** |

**All four exceed the threshold, by 1.7× to 2.5×.** At C = 500, P = 20 the cost is
1,024–1,554 ms; at C = 500, P = 100, 3,920–5,966 ms.

**Under §3.1's rule this warrants a replacement**, and the stage moves to the
prototype phase. Two limits on that statement, both stated rather than buried:

- **250 ms is this plan's placeholder, not the institution's target**, which
  remains `[OPEN]` (policy §1.3). The margin is a factor of two, not a few per
  cent, but the threshold is still not theirs.
- The result governs the **primary** deployment platform. The other two targets
  have different CPUs and are unmeasured.

### 9.4 What 430 million rows does to §5.2's two candidates

The row count is an input §5.2 did not have, and it may **invert** the expected
ordering:

| | Index-build memory |
|---|---|
| **A — global composite key** | An int64 key over 429,971,678 rows is **3.44 GB**, and sorting it needs a comparable index buffer — an estimate of order **+7 GB transient**, on top of a table already in the multi-GB range. **§10.2 measures roughly three times that** for a comparable operation, on unified memory shared with the model |
| **B — per-phenotype sorted slices** | No global key; the offsets already exist; sorting happens within slices |

§5.2 expected A to win on kernel-launch count and explicitly refused to prefer it
in advance. **That refusal is now earning itself**: at this scale A's memory cost
is a first-order selection criterion rather than a footnote, which is exactly what
MAJOR 2 required be measured. Both are still prototyped; neither is preferred here.


---

---

## 10. The §5.3.2 compatibility gate — passed, on a different artifact

Run on a second GB10 SPARK. **Result: PASS.**

| Check | Result |
|---|---|
| Duplicate `(phenotype, target, target_type)` rows | **0** |
| Non-negative ids | `ph=0`, `tg=0`, `ty=0` minima — all pass |
| Composite key maximum | **1,184,843,951** against the int64 limit of 9,223,372,036,854,775,807 — **seven orders of magnitude of headroom**, and computed in Python integers before any int64 tensor existed, as §5.2 requires |

So the uniqueness assertion §5.3.2 introduces will not refuse to start on this
table, and approach A's key domain is nowhere near overflow.

### 10.1 A different artifact — and that is the normal case, not an anomaly

| | §9's artifact | This one |
|---|---|---|
| Rows | 429,971,678 | **430,585,772** |
| Difference | — | **+614,094 (+0.14%)** |

The two machines built their knowledge graphs from **HPO releases roughly two
weeks apart**, and HPO updates about monthly. So the tables differ because the
source ontology moved, not because either build is wrong — and **the artifact the
institution eventually deploys will be a third version again.**

**This changes what the gate can establish, and makes it stronger rather than
weaker.** A one-off scan of "the deployed artifact" is not a thing that can be
completed, because there is no single deployed artifact — there is a sequence of
them. What two *independently built* tables, from different ontology vintages,
both passing with **zero duplicates** does establish is evidence about the
**generator's invariant** rather than about one file: uniqueness comes from
`compute_shortest_paths.py`'s BFS recording each node at first arrival
(`scoring.py:91-94`), and that property survived an ontology update.

Two mechanisms follow, and they are not the same one:

| | What it is for |
|---|---|
| **The load-time assertion** (§5.3.2) | The ongoing guarantee. It runs on whatever table is present, so a future rebuild that violated uniqueness fails at startup instead of scoring wrongly. **This is the right mechanism precisely because the artifact keeps changing.** |
| **The manual scan** | Evidence for the decision to productionise the assertion at all — that it will not refuse to start on real tables. Two vintages passing is the evidence; a third would not add much. |

### 10.1.1 Version matching already exists, and is deliberately a warning

**Nothing here needs a new version mechanism.** The project was scoped with an
updatable knowledge base as a *feature* — rare-disease data improves over time
and the model is meant to be upgraded with it — and
`src/utils/fingerprint.py` is the mechanism that already serves it. It captures
node types and counts, edge types including reverse, per-type feature dimensions
and total KG node/edge counts; the trainer embeds it as `data_fingerprint`, and
`verify_fingerprint` compares it at load (`pipeline.py:574-587`).

**It warns rather than refuses, on purpose** — the module's own docstring says
"so operators can decide". Since the two SPARKs' graphs differ, their
fingerprints differ, and loading a checkpoint built against one KG onto the other
would raise exactly that warning. That is the designed behaviour, not a defect.

**Why §5.3.2's uniqueness assertion fails instead of warning, and why that is
not an inconsistency.** The two checks differ in whether an operator can act on
them:

| Check | Behaviour | Why |
|---|---|---|
| Fingerprint mismatch | **Warn** | An operator can assess it. A KG that gained nodes since training may be perfectly fine to serve, and only they know whether it is |
| Duplicate `(phenotype, target, target_type)` rows | **Fail** | Nobody can assess it. A duplicate makes the indexed lookup return a different distance from the linear scan it replaces, so the failure mode is a *silently different score* — there is no "accepted with duplicates" state that behaves correctly |

Recorded because a project whose philosophy is "warn and let the operator judge"
should not acquire a hard failure by accident, and this one is not an accident.

**One consequence for §9's timings.** They were measured on one vintage. The
table grew 0.14% in two weeks, which is slow but monotone, and the cost is linear
in slice length — so the budget overshoot in §9.3 drifts in the wrong direction
as HPO grows. Not alarming at this rate; recorded so nobody reads §9 as a fixed
property of the system rather than of one snapshot of the ontology.

### 10.2 Measured memory, and why UMA makes it a first-order constraint

| | Reading |
|---|---|
| Peak during the scan | **28.7 GB** |
| Settled afterwards | ~7 GB |
| OS baseline at boot | ~5 GB |
| **Attributable to the scan** | **~24 GB** |

§9.4 estimated "order +7 GB transient" for approach A's key plus sort buffer.
**The measured figure for a comparable operation is roughly three times that**,
and the reason is visible in the scan itself: it widened three narrow columns to
int64 before building the key, which is 3 × 430M × 8 B ≈ 10.3 GB before the key
or the sort. A careful index build would not do that, so **24 GB is an upper
bound for a naive implementation rather than a floor for a careful one** — but it
places the operation in the tens of gigabytes, not the single digits.

**On SPARK this is unified memory.** The index build does not draw from a
separate host pool; it competes directly with the model, the graph tensors and
the embeddings. §9.4 said 430 million rows makes index-build memory a
first-order selection criterion. On UMA it is not merely first-order, it is
shared with the thing the system exists to run — which is the strongest argument
yet for prototyping approach B rather than assuming A.

---

## 11. Prototype phase — both built, correctness closed, timing pending

`scripts/sp_index_prototypes.py`. **Measurement subjects, not production code**,
and not importable from `src/` — when one wins it moves into the loader and the
primitive, and this file goes with the loser. No strategy base class, no
registry, no runtime selector.

### 11.1 What each one is

| | Query | Retained beyond the loader's own tensors |
|---|---|---|
| **A — `global`** | one `searchsorted` over all `P x C` keys, one gather, one masked reduction — a constant number of launches | a full-length int64 key column |
| **B — `slices`** | per phenotype: two `searchsorted` narrow the slice to the `target_type` run, one more resolves all `C` candidates inside it | **nothing** |

Both turn the `O(L)` scan into `O(log L)`. **The difference between them is
memory, not asymptotics** — which is why §9.4 and §10.2 are the sections that
matter for the choice.

Two implementation constraints that are easy to violate silently, so they are
recorded rather than left to review:

- **No query path may touch more than `O(log L)` rows.** Casting a slice to a
  wider dtype to satisfy `torch.searchsorted` is `O(L)` and would quietly
  reinstate the cost being removed. The *query* is cast to the stored dtype
  instead, never the reverse. The first draft of B did the opposite.
- **B's index is built one slice at a time.** A single global lexicographic sort
  would produce the same ordering faster, but would allocate the same
  full-length int64 key A does — erasing the difference being measured. The cost
  moves into build *time*, which is a reported figure.

### 11.2 Correctness — closed, in `tests/unit/test_sp_index_prototypes.py`

25 tests, both prototypes, **exact equality with the scanning primitive** rather
than a tolerance. Exactness is available because distances are BFS hop counts
stored as int8 (`pipeline.py:493`) and the unreachable value is `max_hops + 1`,
so every partial sum is a small integer well inside float64's exact range and
summation order cannot change the bits.

Covered: every target type; a phenotype absent from `offsets`; a candidate no
phenotype reaches; an out-of-range `target_type`; negative and out-of-domain ids;
the narrow `available` semantics both ways; the `C = 1` shape production ships;
duplicate rows refused by both builders while the scanning path tolerates them;
negative ids refused at build; the int64 domain overflow; and invariance to
within-slice row order, which the loader's unstable `argsort` leaves arbitrary.

**The tests were mutation-checked, not merely passed.** Three deliberate defects
— dropping A's target-domain mask, disabling B's duplicate check, and performing
the overflow check in int64 instead of Python integers — each failed exactly the
test claiming that coverage. The overflow mutation is the one worth noting: a
check performed in int64 has already wrapped and would pass, and only the
Python-integer form catches it.

### 11.3 What remains, and why it is not in this container

Timing and memory need the real artifact and deployment-class hardware. Neither
is here.

```
# one prototype per process — see below
.venv/bin/python scripts/benchmark_sp_lookup.py --artifact <shortest_paths.pt> \
    --implementations current,global \
    --output docs/working/scorer-measurement/EVIDENCE_B04_proto_global.json
.venv/bin/python scripts/benchmark_sp_lookup.py --artifact <shortest_paths.pt> \
    --implementations current,slices \
    --output docs/working/scorer-measurement/EVIDENCE_B04_proto_slices.json
```

**One prototype per process is a measurement requirement, not tidiness.**
`ru_maxrss` is a process high-water mark, so a second index built in the same
process inherits the first's peak and its own cost becomes unattributable. The
report carries `memory_attribution_isolated`, which is false whenever more than
one prototype was built there; the numbers are still recorded, and the flag says
how to read them. Including `current` in both runs is free — it builds no index —
and gives each file its own baseline to compare against.

Every implementation is timed over **the same cells** with the same phenotypes
and candidates, so a difference between two rows is the implementation and not
the workload; a test asserts that rather than trusting the loop. Shape order
alternates per *(cell, implementation)* rather than per cell, so measurement
order cannot become a fixed property of which implementation is being timed.

**No timing figure appears in this section**, and the development-CPU smoke run
that exercised the plumbing is not evidence: its synthetic slices are ~200 rows,
where `log L` has almost nothing to save. §9's conclusion — that the baseline
misses the provisional budget — stands on the artifact run, and the prototypes'
verdict must stand on one too.

**Authority above everything here:** `docs/DISEASE_SCORER_POLICY.md`.
