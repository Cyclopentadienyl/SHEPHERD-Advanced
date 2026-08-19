# Spec 3 — Evidence and action provenance

**Status:** draft, under review. **Normative for the constraints below; the evidence design itself is
Gate 3 and is open.**

**Scope discipline.** Evidence traversal is not implemented and its contract is undesigned. An
earlier draft specified a version-lineage schema, `supersedes` semantics and a durable audit state
machine for it — designing a subsystem before its requirements exist. This document now states the
**constraints any Gate 3 design must satisfy**, and nothing more.

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
