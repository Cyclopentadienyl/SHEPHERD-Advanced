# Archive — results-review proposal, rev 1–6

**Status: historical record. Not an implementation authority.** Superseded by Specs 0–4.

Kept for one purpose: so a claim that was investigated and withdrawn is not re-derived, and so the
reasoning behind a non-obvious rule can be found. Nothing here is normative. Where this document and
a spec disagree, the spec is right.

The full rev 6 text is retained separately as `RESULTS_REVIEW_DESIGN_PROPOSAL_rev6.md`.

---

## 1. Withdrawn claims

Each of these was stated as fact in some revision and is false. They are listed because each was
plausible enough to be written down once, and would be plausible enough to be written down again.

| Claim | Why it is false | Withdrawn in |
|---|---|---|
| "The API is protected by being bound to loopback" | Configured deployment defaults bind `0.0.0.0`. The `127.0.0.1` that suggested otherwise is the **WebUI's destination**, not the server's bind. The launcher also *prints* `127.0.0.1` while binding `0.0.0.0` (`shep_launch.py:323-333`) | rev 6 |
| "Every launch path binds `0.0.0.0`" | `scripts/setup_demo.py:29` omits `--host` and inherits Uvicorn's loopback default | rev 6 |
| "Today the exposure is transient in-memory results" | `_write_exports` writes patient-identifying files on **every** diagnosis, and they are served through Gradio `DownloadButton`s | rev 6 |
| "The unauthenticated surface is diagnosis and future history" | It also includes `/pipeline/reload`, `/pipeline/config`, `/training/start`, `/training/stop`, and `torch.load(weights_only=False)` | Spec 4 |
| "FastAPI's default `JSONResponse` emits non-finite JSON tokens" | Starlette renders with `allow_nan=False` | rev 6 |
| "Concurrent diagnoses cannot corrupt each other" | Unverified. `initialize_pipeline` reassigns the shared pipeline (`main.py:424`) with no lock | rev 4 |
| "`session_id` identifies a session" | Generated fresh inside every request (`diagnose.py:180`) — a run identifier, 48 bits, truncated | rev 4 |
| "Sorting by mean distance or by SP score gives the same order" | Same information, **opposite directions** | rev 6 |
| "An all-unreachable candidate has the minimum value" | It has the **maximum** mean distance and the minimum SP score | rev 6 |
| "`sp_status != COMPUTED` means unavailable" | Wrongly captures the numeric `COMPUTED_PARTIAL` | Spec 1 |
| "`TARGET_UNMAPPED` = absent from the SP artifact" | The artifact is sparse; absence means no path within the hop limit. Unmapped is defined against the **node mapping** | Spec 1 |
| "The snapshot folder must not be reachable through any HTTP route" | As written it forbids the repository API the same section requires | rev 6 |
| "A depth-one ring keeps the previous round" | Under write-on-production it keeps only the current result | rev 6 |
| "Decomposition explains why a candidate ranks here" | It explains the **absolute score**. A phenotype contributing strongly to every candidate explains none of the differences | rev 2 |
| "Zero norm makes the score undefined" | Epsilon-clamped `F.normalize` leaves the cosine defined at `0.0`; only the decomposition is unavailable — and the boundary is `≤ eps`, not `= 0` | rev 6, Spec 1 |
| "B-0 vectorises the SP lookup, so eager SP is viable" | It does not. B-0's own source comment says the interface is batched and the implementation is not | rev 6 |
| "Vectorised SP lookup completes B-0" | It completes the **B-1 eager-SP dependency**; B-0 has substantial remaining scope | rev 6 |
| "A failed audit write fails the action" | Unimplementable once the side effect is irreversible | Spec 3 |
| "SQLite is too heavy for a rotating buffer" | Correct for the requirement as it then stood; obsolete once the requirement grew to transactions, locks, holds, tombstones and reconciliation | Spec 2 |
| "Key the snapshot buffer by `session_id`" | It is per-request, so it groups nothing | rev 4 |
| "Persistent storage is a line not yet crossed" | Already crossed by `_write_exports`, and less carefully | rev 3 |
| "A per-target minimum reservation fixes the page-peer problem" | Only if the reservation exhausts reachability or the surplus cannot change a verdict; neither holds | rev 4 |
| "History access requires loopback or authentication" | It substituted an architecture requirement for a deployment decision that belongs to the institution — which had already accepted unauthenticated clinical access on the same segment | scope audit |
| "The requirement set now needs SQLite" | The multi-writer, transactional and high-cardinality conditions were never present: one process writes, and querying means reading ten file headers | scope audit |
| "A canonical serialisation format is needed for the checksum" | It defends truncation, which atomic rename prevents and `json.load` detects — 3,000 random truncations of a 51 KB payload produced **zero** valid parses. Canonical bytes matter when two independent implementations must agree on a hash; here one program reads its own files | scope audit |
| "A failed audit write must fail the action" | Unimplementable once the side effect is irreversible; an export already delivered cannot be recalled | scope audit |
| "`weights_only=False` is a remote-code-execution vulnerability" | An unauthenticated endpoint does reach a code-executing deserialiser, which is worth fixing — but a working chain also needs a file-write primitive not obviously available through these routes. The fix is one keyword argument, verified to read the `logs` dict correctly | scope audit |

---

## 1a. What the scope audit removed, and what would revive it

Two independent audits — reviewer and author — converged: successive rounds promoted individually
valid findings into blocking requirements without weighting them against the deployment, and a
ten-entry recovery feature for a single-workspace tool acquired a records-management platform.

| Removed | Would be revived by |
|---|---|
| Database-backed repository metadata, transactions, tombstones | multi-process or multi-user access |
| Retention-hold modelling, deletion-role governance | the institution confirming manual snapshots are clinical records |
| Durable audit state machine, outbox, idempotent operation IDs, reconciliation | a stated compliance requirement for an audit trail |
| Evidence-version lineage, `supersedes` semantics, delivered/displayed tracking | Gate 3 defining the evidence contract |
| Authenticated integrity (HMAC/signature) and key management | an explicit requirement; it presumes a key store this deployment lacks |
| Canonical cross-implementation serialisation | a second implementation needing to verify these files |
| Parent-directory `fsync`, multi-process transaction semantics | a hardening pass |
| UUIDv7-versus-ULID selection | never — a full UUID with a uniqueness check is sufficient |
| The floating-point ULP derivation in the product spec | never — it lives in `src/inference/scoring.py`, next to the code it constrains |

**The process finding, recorded because it will recur.** Across two rounds the author accepted 17/17
and 14/14 review findings. Factual corrections were correctly accepted; the failure was applying no
proportionality filter to findings that *added machinery*. A reviewer asked to find problems will
find problems — weighting them against the deployment is the author's job, not the reviewer's.

---

## 2. Reasoning worth keeping

Non-obvious rules in the specs, with the argument that produced them.

**Why the SP filter operates on hop distance rather than the score.** "Within an average of 3 steps"
is checkable by a clinician; "SP ≥ 0.25" is not. And `1/(1+d)` compresses the far end so heavily that
the 1→2 step gap is seven times the 5→unreachable gap, so the transform obscures exactly the
distinctions a filter is built on.

**Why `0.0` is a dangerous unavailable sentinel.** The computed range at `max_hops = 5` is
`[1/7, 1/2]` — never 0. So `0.0` sits *below* every genuine value, and a numeric sort places a
mapping failure beneath a real negative result. The legacy `_calculate_sp_score` returns `0.0` on
four distinct failure paths.

**Why the decomposition must not decide about duplicates.** It consumes exactly the phenotype
sequence, multiplicity, mask and denominator the canonical scorer used. Deciding independently — even
"correctly" — makes the contributions stop summing to the score they claim to decompose.

**Why snapshots are written at production rather than at the start of the next run.** A browser
refresh between two runs cannot then lose a result that was never snapshotted; "the previous round"
is simply the second-newest entry; and there is one rule instead of a rule plus a special case. The
number of writes is unchanged.

**Why JSON and not pickle.** Loading a `.pt` executes code, and these files re-enter a clinical tool.
The project has already had to reason about `weights_only` for exactly this hazard — and Spec 4 §1.1
shows that reasoning was not applied everywhere.

**Why checksum verification precedes full schema validation.** A truncated file reported as a schema
violation sends the reader looking for the wrong problem.

**Why float64 in the scoring wrappers.** The pre-extraction code computed in Python doubles. Rounding
the mixture to float32 lets two candidates whose true scores differ below its resolution become an
exact tie, resolved thereafter by sort stability — that is, by input order — which can move a
candidate across the top-k boundary a clinician sees.

**Why the "too small to matter" argument was rejected for the SP distance.** The drift
(`83/24` → eighth significant digit) provably cannot reorder candidates. It was corrected anyway:
B-0's contract is that the extraction preserves behaviour, the tensor holds one value per candidate
so the wider type costs nothing measurable, and arguing for float64 in the mixture while accepting
float32 six lines away is not a defensible position.

---

## 3. Checked and found consistent

Recorded so a later reader knows these were examined rather than overlooked: eager contribution
computation against `selection_limit`; snapshot contents against the four result types; view-state
reset on reopening against C2; automatic snapshots against C5, which governs clinician actions rather
than system writes; and the safety net against the institution's "the clinician's own problem"
stance, which applies only *beyond* the rotation depth.

---

## 4. Revision summary

| Rev | What it settled |
|---|---|
| 1 | First proposal: two surfaces, SP sorting/filtering, decomposition |
| 2 | Four separate types instead of one envelope; decomposition claim narrowed |
| 3 | Institutional storage mechanism; multi-user investigated; backlog |
| 4 | `session_id` and concurrency claims withdrawn; storage contract; page-peer requirement |
| 5 | Network bind corrected; `COMPUTED_PARTIAL` semantics; identity reduced to one |
| 6 | Ring depth 2; evidence lifecycle split; deployment-access gate; TTL sweeper; durability; audit shape |
| — | **Split into Specs 0–4.** The monolith had become a correctness risk and exceeded one review pass |
