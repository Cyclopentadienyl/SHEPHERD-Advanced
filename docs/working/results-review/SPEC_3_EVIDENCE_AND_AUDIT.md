# Spec 3 — Evidence and action provenance

**Status:** draft, under review. **Normative for the constraints below.** The evidence design itself
is Gate 3: §7 holds it, written 2026-10-26 (Batch 2) for review and approval; until approved the gate
is open.

**Scope discipline.** Evidence traversal is not implemented; its design is §7, awaiting approval. An
earlier draft specified a version-lineage schema, `supersedes` semantics and a durable audit state
machine for it — designing a subsystem before its requirements exist. §1–§6 state the
**constraints any Gate 3 design must satisfy**; §7 is the design that meets them, and still adds no
lineage schema or audit store.

---

## 1. Evidence is lazy and per page

Evidence paths are fetched on demand for the candidates on the current page, never eagerly for the
whole selected set (policy §1.2). Before implementation the Gate 3 design must define: request
identity, rejection of stale responses, cache keys, target ordering, cancellation, and
loading / partial / error states.

---

## 2. Constraint: a target's outcome must not depend on its page

**The page-peer problem.** If one traversal budget is shared across the targets on a page, a
candidate's outcome depends on which other candidates share its page: change the page size or the
sort order and the same candidate flips between `UNKNOWN` and `NO_PATH_WITHIN_HOPS`. To the clinician
that is a result changing for no visible reason.

A per-target minimum reservation with a shared surplus does **not** fix it: the reservation would
have to resolve reachability exhaustively (likely infeasible at four hops over this graph), or the
surplus would have to be forbidden from changing a verdict, throwing away real discoveries.

> **Each target's outcome is a function of that target, the result and a declared policy — and of
> nothing else.**

Which implies: a fixed per-target budget independent of page composition; internal sharing of
computation is an implementation freedom, but each target's outcome *accounting* stays independent of
what else was in flight; and results are cached by `(result_id, target_id, policy, budget)`.

---

## 3. Constraint: never overwrite an evidence record that exists

> **Never overwrite an evidence record that has been computed and attached to a result, or captured
> in an artifact.**

A recomputation at a higher budget produces a **new record alongside** the old one, never a silent
replacement. Stated this way because the server cannot know what a person saw (§4); "attached" and
"captured" are things it can observe.

**Snapshots and evidence have different lifecycles**, so they are kept apart:

| Snapshot kind | Evidence in it, schema v1 |
|---|---|
| **Automatic** (written at result production, before any page renders) | **none** — nothing has been computed yet |
| **Manual** | **none in v1** — see below |

**Schema v1 captures no evidence in either kind**, and records
`evidence_capture_status = NOT_SUPPORTED_IN_SCHEMA_V1` (Spec 2 §4). An earlier draft had manual
snapshots pin "the evidence records existing at save time" while the normative payload schema had no
evidence field at all — a storage format depending on a record Gate 3 has not defined. Capture is
added under a new `storage_schema_version` once that record exists.

A reopened snapshot shows what it captured as its own content, and anything computed later
**separately and labelled as such**. The UI must never imply later evidence existed when the snapshot
was created. **A snapshot is self-contained with respect to what it captured** — that is the whole
claim; the merged view is not frozen and is not described as frozen.

**No silent re-fetch under mismatched fingerprints.** Re-fetching against a changed graph would show
the clinician different evidence for the same result without saying so.

**What identifies an evidence record is Gate 3's to define.** It will need enough to answer "is this
the same computation?" — at minimum the policy and budget it ran under and the artifact fingerprints
— but the field list, lineage semantics and upgrade rules belong to that design, not to this one.

---

## 4. Constraint: name timestamps for what the server can observe

Record `computed_at`. Do not record when a client received a response, when a panel rendered, or when
a person looked at it — the server observes none of those, and a field that can never be populated is
worse than an absent one. No surface says evidence was "viewed"; the available word is **computed**.

---

## 5. Action provenance

The requirement is C5: **the view state at the time of a clinician action is recorded with it.** That
is provenance, not a compliance audit trail — the institution has not asked for one.

| Action | Where its provenance goes |
|---|---|
| **Manual snapshot** | the snapshot payload (Spec 2 §4), which is being written anyway |
| **Export (Markdown)** | an ordinary metadata section at the top of the document |
| **Export (CSV)** | **[OPEN]** — see the decision order below |
| **Delete, and ordinary action telemetry** | the existing **stdlib logger** — `logging.getLogger(__name__)`, the live idiom in 36 modules |

**There is no bespoke action-log file.** An earlier draft of this section said "every action appends
one line to a log file". That is a storage subsystem in one sentence — unbounded growth, concurrent
append semantics, permissions, location, rotation, retention, backup policy, write-failure behaviour,
partial-line recovery, and patient data in a file nobody classified. It reintroduced, two sections
later, the durable audit store this document had just declined to build.

A logger call uses what already exists and creates no new storage module.

**Two corrections to how that was stated.** An earlier draft called this "the project's existing
structured logging facility" and cited `src/utils/logger.py`. Neither is accurate:

- **`src/utils/logger.py` is not a facility.** Its own docstring says *"RESERVED home (not yet
  implemented) … Status: intentionally empty … Nothing imports this module."* It documents that the
  live idiom is stdlib `getLogger`; it does not provide one. Citing it as a dependency read a
  description of a gap as evidence the gap was filled.
- **It is not structured.** `src/api/main.py:64` configures
  `format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"`, which renders **the message
  only**. Anything passed as `extra={...}` is silently dropped.

> **Delete and ordinary action telemetry use the existing stdlib logger. This is operational logging,
> not a durable provenance guarantee. Fields that must appear under the current formatter are
> rendered into the log message, unless and until structured logging is separately configured.**

**After a deletion an ordinary log line may be the only trace**, and its retention is deployment
policy and currently unspecified. That is acceptable **only because no compliance audit requirement
exists**. If the institution requires durable, queryable, retained or tamper-resistant provenance,
that is a separately triggered work item — and **this design must not claim stdout or service logs
already provide it.**

### 5.1 CSV provenance — decision order

Not decided here, and deliberately not worked around:

1. **Confirm the actual C5 requirement with the institution.** It may need only the active sort and
   filters, in which case page and scroll are recorded nowhere.
2. If ordinary operational logging satisfies it, use that and stop — knowing exactly what it is and
   is not (above).
3. Only if the CSV must be **self-contained**: standard metadata **columns**.
4. **No custom preamble** — CSV has no standard place for one, and a custom header breaks Excel,
   pandas and `csv.DictReader`.
5. **No JSON sidecar** — a second file a clinician must keep with the first, and will not.
6. **No new audit-log file.**

Rejecting 4 and 5 was right. It did not license inventing 6.

**The result's identity took option 3** (provenance contract M3c, implemented 2026-10-16, merged
as #124). The Diagnosis CSV carries what produced the result — the checkpoint, graph, manifest,
shortest-path table and dataset record by digest or identifier, the scoring mode and the settings
in effect — as 21 `served_*` metadata columns appended after the candidate columns, the same on
every row, under the one header row: no preamble, no sidecar, no log file, and no blank identity
cell (a null is written as a word such as `absent`; the candidate columns keep their convention,
where an empty cell means no value). The Markdown export puts the same identity in an
ordinary section, "What produced this result", before the candidates. That is the identity of the
result, not the view state at the time of export: C5's CSV provenance above stays **[OPEN]**, and
whatever it adds should follow the same convention (one atomic value per column, a common prefix,
appended, never blank) so the file is not reshaped twice. The column dictionary is in the
deployment guides, §4.7.

Automatic snapshots are system events and carry no view state — no clinician was acting.

**An earlier draft specified a durable audit-action state machine** — `PREPARED` / `COMPLETED` /
`FAILED` events, idempotent operation IDs, an outbox, and reconciliation — plus a rule that a failed
audit write must fail the action. That rule is unimplementable once a side effect is irreversible: an
export may already have been delivered when a final update fails, and it cannot be recalled. More to
the point, it is distributed-systems machinery for a single process writing a handful of files, in
service of a requirement nobody stated.

> **Deferred, with its trigger:** a durable audit trail with its own retention, tamper-resistance and
> reconciliation becomes a work item **when the institution states a compliance requirement for one**.
> It is not a prerequisite for R7.

---

## 6. One asymmetry worth keeping

| Failure | Consequence |
|---|---|
| **Automatic safety-snapshot write fails** | The inference still succeeds. It is a backup; losing it must not cost a valid clinical result. Surfaced loudly (Spec 2 §7.4) |
| **A manual snapshot or export fails** | The action fails and says so. The clinician asked for an artifact and must not be told they have one |

These point in opposite directions on purpose.

---

## 7. Gate 3: the target-restricted evidence traversal (Batch 2 design, for approval)

**Status:** design, written 2026-10-26 for review and approval; not implemented. Once approved it
clears policy gate 3. The owner names the approval route (BACKLOG §0). It is built with B-1, as B-1c,
in batch 5.

**Authority:**
- policy §6, gate 3's seven requirements;
- statements 4a, 5 and 6;
- the schedule in §1.2: SP is computed eagerly, evidence lazily;
- §1–§4 of this spec.

| Requirement | Answered in |
|---|---|
| Legal reachability × completeness combinations | §7.2, §7.3 |
| Visited-state and cycle semantics | §7.4 |
| Multi-source provenance | §7.5 |
| Deterministic ordering | §7.6 |
| Global-budget units and enforcement; target fairness | §7.7 |
| Cancellation | §7.9 |
| Request identity, stale responses, cache keys, target ordering, loading/partial/error states (§1); page independence (§2); no overwrite (§3); `computed_at` (§4) | §7.6, §7.7, §7.10–§7.12 |

### 7.1 What it answers, and what it never does

**The question.** Which approved kinds of knowledge-graph path connect this patient's phenotypes to
this candidate?
- **Inputs:** an immutable result (`result_id`), a target in that result's selected set, and a
  declared policy.
- **Output:** one evidence record per target.

**What it never does:**
- admit, remove, reorder or rescore a candidate (statement 4a);
- feed SP, or take input from SP (statement 6);
- run when the result is produced.

**How that is enforced:**
- **By the interface.** The traversal receives the result read-only and returns records.
- **By a contract.** A new import-linter contract, `ranking-without-evidence`, forbids
  `src.inference.scoring` from importing `src.reasoning`.
- **By a test.** Evidence is computed for every target of a result, and the result's digest is the
  same afterwards.

### 7.2 Legal paths: the metapath set `target-evidence/v1`

Paths are typed and directed as the graph builder stores edges:
- phenotype → disease (`phenotype_of_disease`);
- gene → disease (`gene_associated_with_disease`);
- gene → phenotype (`gene_has_phenotype`);
- child → parent (`is_a`).

Here `p` is a patient phenotype and `t` is the target disease.

| Id | Path | Hops | Reads as |
|---|---|---|---|
| M1 | `p →phenotype_of_disease→ t` | 1 | The disease is annotated with this phenotype |
| M2 | `p ←gene_has_phenotype← g →gene_associated_with_disease→ t` | 2 | A gene with this phenotype is associated with the disease |
| M3 | `p →is_a→ a →phenotype_of_disease→ t` | 2 | The disease is annotated with a parent term of this phenotype |
| M4 | `p ←gene_has_phenotype← g →gene_has_phenotype→ a →phenotype_of_disease→ t`, with `a ≠ p` | 3 | A gene with this phenotype also has a phenotype the disease is annotated with |

**Not in v1, and why:**

| Excluded | Why |
|---|---|
| `is_a` over more than one step | a grandparent's annotation reads as support it is not |
| The descendant direction (the disease is annotated with a term more specific than the patient's) | a v2 candidate if clinicians ask for it |
| Ortholog and publication edges | the real graph has none. The demo graph's ortholog edges give no v1 evidence, and the record says so |
| Disease–disease `is_a`, and any path through another disease | Mode B (§7.8) covers a related disease explicitly, under its own label |
| Edge types the builder never emits (`gene_causes_disease`, the pathway types) | there are no such edges to traverse |

A new metapath is a new policy version.

### 7.3 Reachability and completeness

| A path found? | Every scan finished within the budget? | Outcome |
|---|---|---|
| yes | yes | `FOUND_COMPLETE`: the counts are exact |
| yes | no | `FOUND_PARTIAL`: the counts are lower bounds |
| no | yes | `NO_PATH_IN_POLICY` |
| no | no | `UNKNOWN` |

**What "in policy" means.** It is reachability through the v1 metapaths, not reachability in the
graph. SP measures any-path distance within 5 hops, so a candidate can have an SP distance of 3 and
still have no evidence path. The two surfaces say which is which (statement 6).

**A renamed outcome.** `NO_PATH_IN_POLICY` replaces §2's provisional name `NO_PATH_WITHIN_HOPS`,
because the restriction is by kind of path, not only by length.

### 7.4 The traversal, visited state and cycles

**Once per result.** This work is a function of the result alone:
- **`P`:** the used phenotypes, with their request positions.
- **`G_P`:** a map from each gene to the patient phenotypes it has (the `gene_has_phenotype` in-edges
  of each `p`).
- **`U_P`:** a map from each parent term to the patient phenotypes under it (the `is_a` out-edges of
  each `p`).

**Per target `t`, expanded from the target's side:**
1. **Scan the `phenotype_of_disease` in-edges of `t`.** For each source term `a`:
   - `a ∈ P` gives M1;
   - `a ∈ U_P` gives M3, once for each child `p`.
2. **Scan the `gene_associated_with_disease` in-edges of `t`.** Each gene `g ∈ G_P` gives M2, once
   for each of its patient phenotypes.
3. **Scan the `gene_has_phenotype` in-edges of each `a` from step 1.** Each gene `g ∈ G_P` gives M4,
   once for each `p ∈ G_P[g]` with `p ≠ a`.

**Why from the target's side.** The work is then bounded by the target's own edges, and the
patient-side sets are built once per result and shared.

**Visited state: none is needed.** Every legal path has a fixed type sequence of at most three
edges, so the enumeration is exhaustive within the budget. No path is dropped because another path
reached a node first. Today's BFS keeps one `(node, depth)` set per phenotype, which drops
same-length alternatives.

**Cycles and duplicates:**
- A path with a repeated node is not emitted: an `is_a` self-loop gives no M3, and `a = p` gives no
  M4.
- Parallel edges of one type between the same pair give one path, which carries the first such edge
  in graph order.

### 7.5 Multi-source provenance

**Every path names:**
- its source phenotype, by request position and HPO term;
- its intermediate nodes.

A path's identity is `(metapath, source, intermediates)`.

**Per-phenotype coverage.** The record lists every used phenotype with one of:
- its path count for each metapath;
- "none in policy";
- under `FOUND_PARTIAL` or `UNKNOWN`, "lower bound".

Phenotypes the result did not use (unknown terms) are listed as such, from the result's account.

### 7.6 Deterministic order

**Scan order.** The steps run in the order 1, 2, 3. Within a scan, in-edges are read in graph order:
the order of `kg.json`, which the record's graph digest fixes.

**Path order.** Emitted paths are sorted by:
1. metapath (M1 < M2 < M3 < M4);
2. the source's request position;
3. the first intermediate's global index;
4. the second intermediate's global index;
5. the edge index.

**What is shown.** The first five paths per metapath (`paths_shown_per_metapath = 5`, a policy
constant) follow that order. Counts cover every path found.

**Reproducibility.** The same graph, result, target and policy give the same record, byte for byte,
apart from `computed_at`.

### 7.7 Budget, fairness and the request ceiling

**Unit.** The unit is typed adjacency entries examined in steps 1–3:
- the in-edges of the one edge type a scan names;
- read from a per-type in-edge index, built once per graph, in graph order.

Membership tests against `P`, `G_P` and `U_P` are not counted.

**One fixed budget per target, set by the policy version.** Enumeration does not stop at the first
path, because the counts are wanted. Two consequences:
- A target's cost depends on the target and the graph only, never on the patient.
- Whether a target completes is therefore known for every disease in advance: an offline pass over
  every disease lists those whose cost exceeds the budget.

**Value.** `entry_budget` is set by measurement: the per-disease cost distribution on the deployed
graph, from that same offline pass.
- The provisional value is the 99th percentile, rounded up.
- It stands until the latency target, an open institutional value (Spec 1 §6), gives it a time bound.
- Every record names the value it ran under.
- The measurement is batch 5's first step.

**Fairness:**
- Every target gets the same budget, and no target's budget is shared.
- A target's unused budget never passes to another target (§2 rejected a shared surplus).
- Page size, sort order and page peers change no outcome.

**The request ceiling: the policy's "global-budget units and enforcement".**
- A request names at most `max_targets_per_request` targets. This equals `page_size`'s maximum, 100.
- Its worst-case work is the number of targets × `entry_budget`, plus the per-result sets, which
  are bounded by 100 phenotypes' degrees.
- The ceiling is enforced at admission. A request over it is refused whole (422) before any
  traversal starts. It is never partly served from a shared pool.

That is how "global" in the policy and §2 agree: the global bound limits what is admitted, never
any target's outcome.

### 7.8 Mode B and INSUFFICIENT, made reachable

**When analogies run.** Only for a target whose own outcome is `NO_PATH_IN_POLICY`.

**How:**
- **Which diseases.** The `K = 3` diseases nearest the target by raw cosine of the checkpoint's
  disease embeddings. The target itself is excluded, ties go to the lower global id, and only a
  cosine of at least 0 qualifies.
- **Same rule as today, stated on the raw scale.** Today the panel requires (s+1)/2 ≥ 0.5, which is
  the same as a raw cosine ≥ 0.
- **Who computes the list.** B-1's pipeline computes it with `src.inference.scoring` and passes it
  in. `src.reasoning` sits below `src.inference`, so it never needs embeddings.
- **The traversal.** Each analogy runs steps 1–3 against the same patient. The cost counts against
  the same target's budget, the part its own traversal left.
- **Outcomes.** The analogy part has the same four outcomes as §7.3.
- **Naming.** These are named "similar diseases". Today's fields say `similar_gene_*` while holding
  diseases.

**INSUFFICIENT.** The label applies when the target's outcome is `NO_PATH_IN_POLICY` and the analogy
part's is also `NO_PATH_IN_POLICY`. Wherever `UNKNOWN` appears, the label is "not determined within
the budget", never INSUFFICIENT.

**Cost.** The analogy list depends on the target alone, so its cost does not depend on the patient
either.

### 7.9 Cancellation

**Cooperative.** The traversal checks a cancel flag every 4,096 entries and between targets.

**A request is cancelled when:**
- the same view sends a newer request;
- its result is replaced;
- the server shuts down.

**What a cancel leaves:**
- a cancelled target is `NOT_COMPUTED`: no record is written and nothing is cached;
- targets finished before the cancel keep their records.

**Off the event loop.** The route runs the traversal in a worker thread, so a long traversal never
blocks other requests. Today's diagnose route runs synchronously inside `async def` (BACKLOG 56).

### 7.10 Request identity, stale responses and the cache

**The request.** `EvidenceRequest` carries:
- `result_id`;
- the target ids, in page order, at most the ceiling;
- `policy_version`;
- `view_request_seq`, which increases with every request from the view.

**The response.** `EvidenceResponse` echoes `result_id` and `view_request_seq`. For each target it
holds a record, a `NOT_COMPUTED` entry with its reason, or an error.

**Stale responses.** The client discards a response whose `result_id` is not the current result's or
whose `view_request_seq` is not the latest it sent.

**Server refusals:**

| Situation | Answer |
|---|---|
| An unknown or no longer held `result_id` | 410, "run the diagnosis again" |
| A target outside the result's selected set | 422 |
| The result's fingerprints differ from the loaded artifacts | 409. There is no silent re-fetch against a changed graph (§3) |

**The cache key:**
- `result_id`;
- `target_id`;
- `policy_version`;
- `entry_budget`;
- the graph digest;
- the checkpoint digest, when analogies ran.

**Eviction.** The cache is in-process and bounded by count, least recently used out first. A record
is a deterministic function of its key, so a record recomputed after eviction has the same content
and its own `computed_at`: a new record, not a rewrite.

**A higher budget** is a different key, so its record sits beside the old one (§3).

**Where results live.** B-1 keeps produced results in memory by `result_id`, with a bound, for
evidence requests. Spec 2's repository adds persistence when it is built, and snapshot v1 stores no
evidence (§3).

### 7.11 The evidence record, `evidence-record/1`

| Group | Fields |
|---|---|
| Identity | `record_schema`; `result_id`; the target's global id, NodeID and name |
| Policy | `policy_version` (`target-evidence/v1`); `metapath_set`; `entry_budget` |
| Outcome | `entries_used`; `outcome`; the analogy part's outcome (when it ran) |
| Paths | path counts per metapath; the shown paths (each with metapath, source position and term, node ids, edge types and edge indices); per-phenotype coverage (§7.5); the analogies (each with similarity, outcome, counts and shown paths) |
| Provenance | `fingerprints` (the graph digest; the checkpoint digest when analogies ran); `software_revision`; `computed_at` (§4) |

Nothing records when a response was received, rendered or seen (§4).

### 7.12 Loading, partial and error states

| State | The panel says |
|---|---|
| Computing | "Computing evidence…" |
| `FOUND_COMPLETE` | the paths, grouped by metapath, with the counts |
| `FOUND_PARTIAL` | the paths and "The search stopped at its budget, so more may exist" |
| `NO_PATH_IN_POLICY` | "No supporting path of the approved kinds", then the analogy part |
| `UNKNOWN` | "Not determined within the search budget" |
| INSUFFICIENT | "No supporting path, and no similar disease with one; validate independently" |
| `NOT_COMPUTED` | "Not computed (the page changed)"; computed again when the target is visible |
| An error | "Evidence unavailable: \<reason\>. The ranking is unaffected" |

A failed or stale fetch degrades only the panel (Spec 1 §2).

### 7.13 Where the code goes, and what goes away

| Piece | Where |
|---|---|
| The traversal, the record and the per-type in-edge index | `src/reasoning/target_evidence.py`. It reuses `KnowledgeGraph`'s adjacency for the index and `ReasoningPath` for nodes and edges; that type's unused `evidence` field stays unused |
| The analogy list, the result store and the admission check | B-1's pipeline methods in `src/inference/pipeline.py` |
| The route | `POST /api/v1/results/{result_id}/evidence`, a new `src/api/routes/evidence.py` |
| The contract | `ranking-without-evidence` in `.import-linter.ini` |

**What goes away, and when:**
- **With `served_mixture/v1` and `path_reasoning/v1`:** `EvidencePanel` and the explanation step's
  evidence (`_add_explanations`) retire. `EvidencePanel` includes `_find_direct_paths`, which
  searches to a node type and filters afterwards, and `_find_analogies`, which has no budget.
- **Discovery** (`_find_all_paths`, the BFS and `DirectPathFinder`) stays as long as the fallback
  ranks with it. B-2's decision on no-model serving decides how long that is.

### 7.14 Tests that make it acceptable

| Test | Checks |
|---|---|
| Determinism | the same inputs, twice, give identical records |
| Page independence | a target's record is the same across page sizes, sort orders and peers |
| Exhaustiveness | on a fixture graph, the records equal a brute-force enumeration of M1–M4 |
| Budget boundary | a fixture target costing exactly `entry_budget` completes; one costing one entry more is `FOUND_PARTIAL` or `UNKNOWN` |
| Cancellation | a cancel mid-target gives `NOT_COMPUTED`, and the earlier targets are kept |
| Ranking untouched | the result's digest is the same before and after evidence for every target |
| Refusals | a stale response is discarded; a fingerprint mismatch gives 409; a request over the ceiling gives 422 |
| Analogies | ties go to the lower id; INSUFFICIENT is not shown under `UNKNOWN` |

### 7.15 Open, and whose

| Open item | Whose |
|---|---|
| The value of `entry_budget` | engineering, measured as batch 5's first step; provisional until the latency target exists |
| The latency target and `page_size`'s bounds | the institution (Spec 0 §3) |
| The approval route for this design | the owner |
