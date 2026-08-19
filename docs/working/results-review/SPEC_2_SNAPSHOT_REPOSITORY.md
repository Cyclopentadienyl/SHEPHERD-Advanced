# Spec 2 — Snapshot repository

**Status:** draft, under review. **Normative.** Supersedes rev 6 §9.
**Scope discipline:** this is a bounded recovery feature for a single-workspace clinical tool — at
most ten automatic entries plus manual saves, one writing process. Earlier drafts grew it into a
records-management platform. Machinery that only pays for itself under multi-process, multi-tenant or
compliance-audit conditions is **deferred and named** in §10, not designed here.

---

## 1. What the institution asked for (R7)

Jump to the viewer on completion; manual snapshots with clinician-supplied names; a fully automatic
rotating safety net capped at 10 rounds. Losing an un-snapshotted result is accepted **beyond the
rotation depth** — inside the buffer, nothing is lost.

---

## 2. Storage

**JSON files in a configured protected directory. No index file, no database.**

| Concern | Mechanism |
|---|---|
| Snapshot payload | one immutable JSON file per snapshot, opaque UUID filename |
| Listing and rotation order | **`list_snapshots` (§5.3)** — validate, compare fingerprints, read the listing fields |
| Retention state | derived: expiry is `snapshot_created_at` plus the *current* retention policy |
| Concurrent writes | **one repository lock** (§7.3) |

Everything a listing needs is in the payload (§4), so every listing question is answerable from the
files themselves.

**Listing fully validates at most `automatic_depth + manual_count_limit` payloads** (§7.2). That is
the bound the design rests on — not an informal "ten to a few dozen", which stops being true if the
manual limit is configured upward. **A cache for that many files is machinery this deployment does
not need.**

**Listing reuses the §5 loader — there is no second parser path.** JSON has no standard random-access
header, so a "read just the header" shortcut would mean a partial parser of our own. At this scale
the full loader is fast enough and is already the thing that validates.

**Why JSON and not pickle.** Loading a `.pt` executes code, and these files re-enter a clinical
tool. At 200 candidates with 20 contributions each a snapshot is a few hundred kilobytes.

**Why no database.** An earlier draft specified SQLite for transactional indexes, multi-writer locks,
tombstones and TTL queries. Those requirements do not exist here: one process writes, and "querying"
means reading ten file headers. SQLite becomes the right answer **if** multi-process or multi-user
access lands (§10).

---

## 3. Namespace and identity

```
result_id         # a full UUID, checked for uniqueness on write
```

**The configured directory is the repository boundary.** With one directory and one workspace there
is nothing for a workspace identifier to distinguish, so schema v1 has none.

**No `workspace_id`, `actor_id`, `tenant_id` or `case_id`.** Each was proposed to "prevent a
migration" and none does: a field whose semantics do not exist still needs a migration to define and
backfill it, and `workspace_id` would additionally require a marker file, mismatch handling and clone
semantics — none of which changes a single current operation. Workspace identity arrives when
multiple repositories, import/export, authenticated ownership, merge or replication becomes real.

The existing API `session_id` is `f"sess_{uuid.uuid4().hex[:12]}"` — 48 bits, truncated, never
checked for uniqueness (`src/api/routes/diagnose.py:180`). Existing values are accepted on read
through an explicit legacy field and never minted again. **The reason is schema hygiene and enforced
uniqueness, not security:** 48 bits is not guessable over HTTP, and the exposure question is §9's,
not the identifier's.

---

## 4. Payload schema

Self-contained: a snapshot stores its derived analysis rather than recomputing on load, because
recomputation would require the model and knowledge graph still to be loaded *and identical* — and
the whole point of reopening an old result is that they may not be.

```
storage_schema_version                       # the file format
analysis_version, tie_breaking_version       # the computation
result_id, legacy_session_id?, produced_at, deployment_mode
snapshot_kind = AUTOMATIC | MANUAL           # the discriminator
snapshot_created_at                          # when THIS SNAPSHOT was written
manual_label                                 # required on MANUAL; forbidden on AUTOMATIC
patient_id
inputs:       phenotype IDs, confidences, mapping outcomes, unmapped IDs
config:       selection_limit, eta, max_hops, scorer identity, score_semantics,
              denominator_policy                       (Spec 1 §3.4)
runtime:      software revision, warnings, degraded-mode flags
fingerprints: scorer, knowledge graph, shortest-path artifact
CanonicalDiagnosisResult                     # candidates, canonical scores, canonical ranks
CandidateAnalysisRecord[]                    # SP quantities, mapping metadata, contributions
evidence_capture_status: Literal["NOT_SUPPORTED_IN_SCHEMA_V1"]
view_state                                   # required on MANUAL; forbidden on AUTOMATIC (C5)
```

**Schema v1 contains no evidence, for either snapshot kind.** Gate 3 has not defined what an evidence
record is, and a storage schema must not depend on a record that does not exist. When Gate 3 lands, a
validated evidence section is added under a new `storage_schema_version`.

`evidence_capture_status` is partly redundant with `storage_schema_version`, and is kept anyway
because it distinguishes *"this schema cannot carry evidence"* from *"it can, and this snapshot has
none"* — a distinction the next version will need. It is a `Literal`, so a future value is a schema
change rather than a free-text drift.

`storage_schema_version` is separate from `analysis_version`: the file format and the computation
change for different reasons.

### 4.1 Two variants, as a Pydantic discriminated union

`snapshot_created_at` is **not** `produced_at`: a manual snapshot may be written long after the result
was produced, and age cleanup, rotation, manual exclusion from ring eviction and post-restart
reconstruction all key on when the *snapshot* was written.

| | `AutomaticSnapshot` | `ManualSnapshot` |
|---|---|---|
| `snapshot_kind` | `Literal["AUTOMATIC"]` | `Literal["MANUAL"]` |
| `manual_label` | **forbidden** | **required**, non-blank, **max 256 characters** |
| `view_state` | **forbidden** — no clinician was acting | **required**, per the C5 interpretation (Spec 3 §5) |

**`manual_label` is not an open question.** R7 asks for "manual snapshots with clinician-supplied
names", so the requirement already settles it. Non-blankness and the 256-character ceiling use
**Pydantic's existing stripped-string and length constraints** — an ordinary deployment bound, not a
sanitisation framework and not another institutional decision. An earlier draft marked it `[OPEN]`, which invented a
decision out of an answered requirement. The institution may still clarify *which view-state fields*
matter; it does not get to decide whether a manual snapshot has its name.

**Omitting a field is not forbidding it.** Pydantic's default is `extra="ignore"`, so an
`AutomaticSnapshot` that simply lacks `manual_label` would silently **discard** one rather than
reject it. The base model sets it explicitly:

```python
class SnapshotBase(BaseModel):
    model_config = ConfigDict(extra="forbid")
```

with `Literal` discriminator values and Pydantic's built-in discriminated union. No custom variant
dispatch — the library does all of this.

**Retention state is derived, not stored.** Expiry is `snapshot_created_at` plus the *current*
retention policy, so changing the policy takes effect without rewriting immutable files.

---

## 5. Loading

Two steps, because they answer different questions and are needed at different times. **One parser,
two entry points** — the second calls the first.

### 5.1 `validate_snapshot_file(path)` — is this file a well-formed snapshot?

**Used by publication (§7.3), the startup scan, listing, and load.**

1. Enforce the **raw-size bound**.
2. Parse JSON with `parse_constant` **rejecting `NaN` and `Infinity`**.
3. Read `storage_schema_version`.
4. Apply an **explicitly implemented migration**, or refuse an unsupported or unknown version.
5. Validate through the **Pydantic discriminated union** (§4.1).
6. Validate **required internal invariants** not expressible structurally (§5.1.1).
7. Return the validated snapshot.

> **This entry point does not compare against currently loaded artifacts.**

#### 5.1.1 The minimum internal invariants

Named, so "internal invariants" is a checklist rather than a gesture. Expressed as **Pydantic model
validators or small domain checks** — **not a generic invariant engine**:

- result IDs agree across the envelope;
- candidate ranks are exactly the permutation `1..N`: `set(ranks) == set(range(1, candidate_count + 1))`. *Positive and unique* is too weak — it admits `{1, 2, 5}` for three candidates;
- the selected count matches the candidate count;
- the automatic / manual variant invariants hold (§4.1);
- `phenotypes_mapped <= phenotypes_submitted`;
- numeric SP fields occur **only** on numeric statuses — `COMPUTED` and `COMPUTED_PARTIAL`
  (Spec 1 §3.3);
- every numeric value is finite.

### 5.2 `load_for_current_view` — and is it commensurable with what is loaded now?

**Used when reopening a snapshot in the UI.**

1. Call `validate_snapshot_file`.
2. Call `compare_artifact_fingerprints` (§5.3).
3. Return the validated snapshot **plus its compatibility status**.
4. The caller surfaces any mismatch (§5.4).

> **A fingerprint mismatch is not corruption.** It is a compatibility state, and it must not run at
> write time: a pipeline reload between result production and snapshot publication would otherwise
> make a perfectly correct snapshot **fail to be written**.

### 5.3 `compare_artifact_fingerprints` — one pure helper, two callers

**Listing must report compatibility too** (§5.4), and an earlier draft broke that by routing listing
through validation alone. The comparison is extracted rather than duplicated:

```
compare_artifact_fingerprints(stored_fingerprints, current_artifacts) -> CompatibilityStatus
```

| Entry point | Steps |
|---|---|
| `validate_snapshot_file` | parse, migrate, validate. **Nothing else** |
| `list_snapshots` | validate, compare, return listing metadata **plus compatibility status** |
| `load_for_current_view` | validate, compare, return the snapshot **plus compatibility status** |

It is a pure function over two fingerprint sets: no parser, no loader framework, no second index.

**`CompatibilityStatus` is a small value object carrying two sets.** A single status value cannot
express the real states: the scorer and the knowledge graph can differ *at the same time*, SP can be
unavailable while the other two are loaded, and mismatch and unavailability can co-occur.

```
CompatibilityStatus:
    mismatched_artifacts:           set[ArtifactKind]   # differ from what is loaded
    unavailable_current_artifacts:  set[ArtifactKind]   # nothing loaded to compare against

ArtifactKind = SCORER | KG | SP
MATCH is both sets empty.
```

Two sets rather than an enum **combination matrix**: with three artifacts and three per-artifact
states, an enum would need to enumerate what two set memberships express directly.

**No checksum in v1, and no canonical serialisation format.** Three mechanisms already cover
corruption: atomic publication means a partially written file never receives its final name; a
truncated JSON file does not parse; and schema validation rejects a structurally wrong one. An
earlier draft specified RFC 8785 canonical bytes, separator and Unicode rules, float rendering and
checksum-field self-exclusion — an interoperability format for a case where two independent
implementations must agree on a hash. Here one program writes and reads its own files.

**Non-finite numbers:** the writer passes `allow_nan=False`; the loader rejects them. Defence in
depth — no current path produces one — and it is one keyword argument. (Starlette's `JSONResponse`
already renders with `allow_nan=False`; bare `json.dumps` does not.)

**Schema migration policy, selected rather than listed:**

- read `storage_schema_version`;
- **migrate** only versions with an explicit migration function;
- **refuse** an unknown newer version;
- **refuse** an unsupported older version, with an error naming the version and what to do;
- **no generic read-only mode** until a real legacy format requires one.

**Validate with the project's existing Pydantic v2 models.** Reject `NaN` and `Infinity` at parse
time using `json.load`'s built-in `parse_constant` hook, before Pydantic runs. Do not write a
recursive validation framework.

### 5.4 Fingerprints are shown, never silently ignored

An older snapshot may have been produced by an older model and graph. Any snapshot listing and any
reopened snapshot **displays its scorer and knowledge-graph fingerprints and marks those not produced
by the currently loaded artifacts.** Without that, two results can be compared as though
commensurable when they are not.

---

## 6. Snapshot kinds

| | Automatic (safety net) | Manual |
|---|---|---|
| Written | at result production | on a clinician action |
| Naming | automatic label only | clinician-supplied name or tag, plus the automatic label |
| Evidence | none — `NOT_SUPPORTED_IN_SCHEMA_V1` (§4) | none in v1; see Spec 3 §3 |
| View state | none — no clinician was acting | recorded in the payload (C5) |
| Ring eviction | yes | **no** — only explicit deletion removes it |

**Writing at production, not at the start of the next run:** a browser refresh between two runs
cannot lose a result that was never snapshotted, "the previous round" is simply the second-newest
entry, and there is one rule rather than a rule plus a special case. The number of writes is
unchanged.

**View state is recorded, never reapplied.** A reopened snapshot opens in the canonical view (C2).
The view state on a manual save is provenance for that action, not a setting to restore.

**The automatic label** — timestamp, patient identifier, `selection_limit`, fingerprints — is display
metadata **inside the file, never a filename**. Filenames stay opaque UUIDs, so a directory listing
is not a patient list.

---

## 7. Rotation, retention and durability

### 7.1 Depth counts the current result

```
produce A → ring [A]
produce B → write B, evict A → ring [B]
```

A depth-one ring holds only the current result.

| Guarantee | Minimum depth |
|---|---|
| Recover the newest result after a browser refresh | 1 |
| **Recover the preceding result after a new run** — the institutional minimum | **2** |
| Configured history | 2–10 |

A transient over-capacity is expected between a successful write and the eviction that follows it,
and it is not limited to one entry: a large payload may require a **victim set** (§7.2). The startup
scan reconciles whatever it finds, oldest-first, until both ceilings hold (§7.3).

### 7.2 Retention

- **Count bound:** the rotation depth, which already bounds storage.
- **Age-based cleanup on startup and on write**, deleting over-age entries. No sweeper daemon, no
  clock-skew protocol, no quarantine tier — this deletes at most a handful of files in a directory
  the process already scans.
  **Expiry is therefore opportunistic, not a hard maximum:** an idle long-running process performs no
  cleanup, so a file can outlive its nominal age. If the institution requires a hard bound, add
  periodic enforcement then — and only then.
- **A file that fails to load is reported and left in place**, not deleted, and is surfaced in health
  reporting.
  **Every regular file in the repository directory counts toward the total raw-byte ceiling — final
  files and orphaned temporaries alike. An unparseable file counts toward that ceiling but toward
  neither the automatic nor the manual logical quota.** This keeps the disk bound honest without a
  quarantine tier.

  **The payload being written is counted once, by `len(payload_bytes)`.** The bound check runs under
  the repository lock *before* the temporary file exists, so the in-flight write is never counted
  twice.

#### Logical and physical bounds are separate, and conflating them deadlocks the ring

An earlier draft checked `existing usage + new payload <= bounds` before publication and eviction.
**That makes a full automatic ring unable to rotate:** at the count or byte ceiling, every
replacement fails before the victim it would replace can be evicted — a rotating buffer that stops
rotating exactly when rotation is the whole point.

Under the repository lock, for an **automatic** snapshot, select a **victim set** — not one victim.
One large payload can require several evictions before it fits:

> Take the **smallest oldest-first prefix** of the automatic snapshots for which both
> `projected_count <= automatic_depth` and `projected_bytes <= automatic_byte_ceiling` hold.

**Only `len(payload_bytes) > automatic_byte_ceiling` proves the payload cannot fit**, because that is
the case where removing *every* eligible automatic victim still leaves it too large. That is a
configuration error — the single-payload bound and the ring ceiling disagree — reported as such, not
retried. Anything short of it is a rotation, however many entries it takes.

| | Check |
|---|---|
| **Logical, post-rotation** | count and bytes **after removing the whole victim set** |
| **Physical, transient** | `current raw bytes + len(payload_bytes)` against the repository's physical byte ceiling |
| **Physical file count** | the whole victim set is still present, plus the new payload and one temporary |

Publish, and **evict the victim set only after publication succeeds** (§7.3).

**Manual snapshots have no victim set.** They keep the simple form: `current manual usage + new
payload` against their independent quota, and a manual save that would exceed it **fails visibly**
rather than evicting anything.

#### The transient-capacity invariant, validated at startup

Because victims are deleted only *after* publication, a healthy full ring transiently holds
everything plus the newcomer. If the physical ceilings are configured smaller than that, a
**corruption-free full ring silently stops accepting writes** — the failure mode the separation of
logical from physical bounds was meant to remove.

So the configuration is validated once, at startup:

```
physical_byte_ceiling  >=  automatic_byte_ceiling
                         + manual_byte_quota
                         + max_single_snapshot_bytes

physical_file_count    >=  automatic_depth
                         + manual_count_limit
                         + 1          # one active temporary
```

Corrupt and orphaned files may consume headroom beyond this and **visibly block writes** — that is
correct, and it is what surfaces them. **A normal, corruption-free, full ring must never be blocked.**
- **A total regular-file count ceiling** covers valid, corrupt and orphaned files together.
  `automatic_depth + manual_count_limit` bounds only *valid* payloads; corrupt files and orphaned
  temporaries sit outside both logical quotas and would otherwise be unbounded in number. One
  ceiling, checked in the same scan — **no index and no quarantine subsystem.**

**Manual snapshots are never ring-evicted, so they need their own bound.** The automatic ring is
count-bounded; without a separate limit, manual saves are unbounded durable growth.

| Tier | Bound |
|---|---|
| Automatic ring | rotation depth (2–10), and its own byte ceiling |
| Manual | **an independent count and byte quota** |

**Manual and automatic capacity are separate pools.** When the manual quota is full, a manual save
**fails visibly** — it never evicts a manual snapshot, and it never consumes capacity reserved for
the automatic safety net.

Whether an age bound applies to manual snapshots, and who may delete what, are
**[OPEN — institutional]** (§10). Retention holds as a modelled concept are **not designed here**: at
this scale the manual snapshot *is* the hold.

### 7.3 Atomic publication

Ordered, and the ordering is the contract:

0. **serialise once**, with `allow_nan=False`, encoded to UTF-8 → `payload_bytes`;
1. **under the lock**, select the eviction **victim set** (automatic only) and run the **separate
   logical and physical bound checks** of §7.2. **The actual serialised length — never an estimate
   from candidate count, and never a second serialisation**;
2. create the temporary file **in the destination directory**, same filesystem — elsewhere makes the
   rename a non-atomic cross-device copy. Use **`tempfile.mkstemp(dir=snapshot_dir, ...)`** or an
   equivalent secure standard-library primitive; **do not generate temporary filenames by hand**;
3. write **the same `payload_bytes`**, then `fsync` the file;
4. **run `validate_snapshot_file` (§5.1) on the temporary file** — read back what was actually
   written rather than trusting what was intended. **Do not call `compare_artifact_fingerprints`
   during publication** (§5.3): a pipeline reload between production and publication would otherwise
   fail a correct snapshot, and a fingerprint mismatch is a compatibility state, not corruption;
5. `os.replace` to the final opaque name;
6. **`fsync` the snapshot directory** — `os.open(snapshot_dir, os.O_RDONLY)`, `os.fsync(fd)`, close.
   The rename is a directory-metadata change, and `os.replace` only orders it against other renames,
   not against power loss. Until this returns, the replacement is *atomic but not durable*: a crash
   can leave the directory entry unwritten. Step 7 deletes files, so it may not run on the strength of
   a replacement that might not survive a reboot;
7. only then, evict the victim set selected in step 1.

**Step 6 is normative, not hardening.** Steps 5 and 7 together are a delete predicated on a write; the
`fsync` is what makes the write true before the delete happens. Deferring it does not make the design
weaker in some abstract way — it removes the precondition that the rest of the ordering exists to
establish.

**The ordering exists for one guarantee (I4): no repository-controlled publication failure removes
planned victims before a validated replacement has been atomically published and made durable.** Note
what this does *not* say. Successful rotation deliberately deletes previously valid snapshots — that
is what rotation is — so the guarantee cannot be that no valid snapshot is ever destroyed. It is an
ordering guarantee: eviction is unreachable until steps 5 and 6 have both succeeded. A crash or error
before step 5 leaves the old entries untouched and the new one absent; a failure at or after step 6
leaves every entry valid, and over capacity **only if an eviction was planned and did not complete**
— a `fsync` failure on a ring that was not full removes nothing and exceeds nothing. Neither outcome
is short of a good result, which is the only outcome that would matter to a clinician.

It is also scoped to what the repository controls, and that scope is stated exactly rather than as
"hardware failure":

> **Underlying-media failure, filesystem corruption and external deletion are outside the guarantee.
> Process termination, system crash and power loss are covered only to the extent established by
> successful file and directory `fsync`.**

The second sentence is the important half. `fsync` returning successfully is what the crash-safety
claim rests on; where it has not returned, there is no claim, which is precisely why step 6 gates
step 7 and why a failed step 6 is reported as a distinct outcome below rather than folded into
"published".

**One repository lock covers the whole operation** — storage-bound check, publication, rotation
selection and eviction — not just the eviction step. A single API process still serves concurrent
requests, so two diagnoses completing together would otherwise both pass the bound check and both
choose the same victim.

**Execution model, decided:** the repository operation is ordinary synchronous code guarded by a
`threading.Lock`, and the FastAPI handler runs it through **`starlette.concurrency.run_in_threadpool`
— one helper, used consistently** — and awaits completion. Blocking file I/O and `fsync` never run on
the event loop, and `snapshot_status` is still final in the response.

**No background queue and no custom worker subsystem** — the payload is a few hundred kilobytes.
Benchmark the synchronous path before considering anything else. Do not write a transaction manager.

#### Failure and recovery

**Before `os.replace` — any failure:** close the temporary file and **best-effort unlink** it. An
unlink that itself fails is **reported through health status**, because the orphan now counts against
the physical ceilings (§7.2) and will otherwise be invisible until writes start failing.

**After `os.replace` there are three outcomes, and only two of them have a caller to report to.**

*Control returns and the process is still running.* The replacement is visible under its final name
and validates. If the directory `fsync` failed, **do not evict** and return
`VISIBLE_DURABILITY_UNCONFIRMED` — visible now, survival across a crash **not claimed**. If eviction
then fails, return `PUBLISHED_RETENTION_CLEANUP_FAILED`. Neither is a failed result: the snapshot
exists in both.

*The process or the system terminates before the directory `fsync` has returned.* **There is no
caller result to classify at all.** After restart the repository may hold the previous ring alone or
the visible replacement; which of the two is filesystem behaviour, and **this document does not claim
the replacement survived** — that claim would require the `fsync` that did not return. What is
claimed is the part that is ordered rather than probabilistic: **the planned victims are untouched**,
because eviction is unreachable until the directory `fsync` succeeds.

*The process terminates after durability is established but before eviction completes.* The
replacement is durable; an oldest-prefix subset of the victims may be gone. The startup scan finishes
the retention that was interrupted.

These are not one outcome under three labels. Collapsing them loses the fact an operator needs —
whether the replacement is known to survive a reboot — and it would assert a survival the storage
layer never confirmed.

**Over capacity is a separate question from all three.** It is a property of the repository's actual
state, not of which step failed: publication onto a ring that was not full plans no victims and
leaves nothing over any bound. Health reports it when a bound is genuinely exceeded, and reports
unconfirmed durability whenever `fsync` did not return — the two are independent signals.

Three caller-visible statuses, because there are three distinct states of the world:

| `snapshot_status` | What is true | What is not yet true |
|---|---|---|
| `PUBLISHED` | Replacement is validated, atomically in place and **durable**; retention completed | — |
| `VISIBLE_DURABILITY_UNCONFIRMED` | Replacement is validated and visible under its final name; **no eviction has been attempted** and every planned victim is still present | Durability across power loss. `fsync` did not return, so nothing may be claimed about a crash |
| `PUBLISHED_RETENTION_CLEANUP_FAILED` | Replacement is validated and **durable**; eviction started | Retention is incomplete: an **oldest-prefix subset** of the planned automatic victims may already be gone |

These are values of the existing `snapshot_status` field (§7.4), not a new object and not an operation
state machine. They may — and should — share one health signal and one startup repair; what may not
happen is the durability distinction being dropped on the way to the caller, because the two states
call for different operator action. `VISIBLE_DURABILITY_UNCONFIRMED` says the storage layer failed to
confirm a write and the next power loss may lose the snapshot; `PUBLISHED_RETENTION_CLEANUP_FAILED`
says the snapshot is safe and the directory is untidy. Only the first is urgent.

Recovery is shared:

| When | Behaviour |
|---|---|
| Directory `fsync` fails | **Do not evict.** Status `VISIBLE_DURABILITY_UNCONFIRMED`. **Always** surface unconfirmed durability. Surface over-capacity **only if the repository actually exceeds a bound** — the ring may not have been full, in which case no victim was planned and nothing is over anything |
| Runtime eviction fails | Status `PUBLISHED_RETENTION_CLEANUP_FAILED`; surface over-capacity, since planned retention did not complete. Eviction is **not** rolled back — a deleted victim was a legitimate deletion the moment the replacement became durable |
| Startup scan | Deterministically repair an over-capacity automatic ring with the **same oldest-first policy** (§7.2), until both the count and byte ceilings hold |

**Manual snapshots are never victims of that repair**, at startup or at runtime.

Plus orphan temporary-file cleanup in the same startup scan.

**No rollback, deliberately.** Partial eviction is not a corrupted state to undo; it is a legal
prefix of the work I1 prescribes, and the startup scan finishes it. Restoring a deleted victim would
require having kept it, which is the transaction journal this design has already refused (§10).

### 7.4 Write failure must not be silent

A failed safety snapshot never turns a valid inference into an inference failure — it is a backup,
and losing it must not cost a valid clinical result. But silence is its own defect: the clinician
then believes a safety net exists at the moment it stopped existing.

1. a non-blocking but prominent UI warning;
2. `snapshot_status` in the result response;
3. a server log entry with the failure reason;
4. **no eviction of any planned victim** — guaranteed, for every failure up to and including the
   directory `fsync`, by the step ordering in §7.3. After that point eviction is permitted and a
   partial oldest-prefix deletion is a legal outcome, reported as its own status rather than as a
   failure;
5. degraded storage status in health reporting.

`snapshot_status` is one enumerated field carrying the whole outcome — the three success-shaped values
of §7.3 plus the failure values for a snapshot that was never published. A caller reads one field and
learns both whether the snapshot exists and whether it is durable; splitting that across a boolean and
a message is how the durability half gets dropped.

---

## 8. Directory protection

| Requirement | Note |
|---|---|
| Configured path, restrictive permissions set at creation | not inherited from an umask |
| Opaque UUID filenames | a directory listing must not be a patient list |
| **Never statically served, never URL-addressable, never reached by a client-supplied path** | a controlled server-side API may return **validated, parsed content** by `result_id`; nothing streams bytes from that directory to a client |
| Encryption at rest, or a stated encrypted-volume dependency | **[OPEN — institutional]** |
| Maximum payload size, entry count and total storage | enforced, so a durable writer cannot exhaust the disk |

---

## 9. Access decision before history is exposed

**A documented access decision is required before snapshot list, load, delete and manual creation are
enabled over HTTP.** Three paths are acceptable:

| Path | Condition |
|---|---|
| **A — Local single workspace** | Loopback or SSH-forwarded access to a verified single-workspace deployment |
| **B — Authenticated** | An approved authenticated and authorised layer (Spec 4) |
| **C — Accepted protected segment** | Explicit institutional acceptance of the protected-segment risk for the current single-workspace deployment |

**Path C is not equivalent to authentication and is never described as though it were.** It records
that the institution, knowing the audience its network segment admits, accepts it. That is the
institution's decision to make; an earlier draft effectively overrode it by permitting only A and B,
which substituted an enterprise architecture requirement for a deployment decision that is not the
designer's.

What the decision must be able to see, stated plainly so it can be made knowingly:

- the automatic writer is **ungated** — writing to a protected directory exposes nothing over HTTP —
  provided §3's namespace, §8's protections and bounds, §7's retention and §7.4's failure reporting
  are in place;
- **history access is what changes the exposure**: today the application serves only the *current*
  result's exports (`_write_exports` → Gradio `DownloadButton`, `diagnosis_panel.py:487-496, 580-588`);
  after B-1 it can enumerate and reload *past* results;
- so the new property is **historical discoverability and reloadability of durable patient data**,
  not first-time file access;
- the existing eager-export accumulation is a **defect scheduled for removal**, not a precedent that
  authorises a second unbounded writer.

---

## 10. Open and deferred

**Open institutional values**

| # | Value |
|---|---|
| 1 | Snapshot folder path; payload, count and total storage bounds |
| 2 | Default rotation depth, 2–10 |
| 3 | Maximum age, and whether it applies to manual snapshots |
| 4 | Who may delete automatic and manual snapshots |
| 5 | Encryption at rest, or an encrypted-volume dependency |
| 6 | Backup inclusion for the snapshot directory |
| 7 | The §9 access decision |

**Deferred by scope, with the trigger that would revive each**

| Deferred | Revived when |
|---|---|
| Database-backed metadata, transactions, tombstones | multi-process or multi-user access lands |
| Retention-hold modelling, deletion-role governance | the institution confirms manual snapshots are clinical records |
| Authenticated integrity (HMAC or signature) and key management | authenticated integrity is explicitly required; it presumes a key store this deployment does not have |
| Canonical cross-implementation serialisation | a second independent implementation must verify these files |
| Formal multi-process transaction semantics | hardening pass |

---

## 11. Frozen invariants, and when this document reopens

**§5 and §7 are FROZEN.** Not "will be" — the closure round completed, the
bounded provenance audit was accepted, and the freeze took effect at the commit
that carries this sentence. Changes to those sections from here require one of
T1–T6 in §11.2, cited by name.

They describe a rotation and
publication algorithm, and prose has no type checker — each round of English
refinement kept exposing the next undefined transition, which is the compiler's
job rather than a reviewer's. Precision beyond this point moves into the
implementation's tests, which fail on an unhandled branch where a paragraph
cannot.

**What is frozen is the mechanics, not the guarantees.** The loop that builds a
victim set, the dataclass, the `try`/`finally`, the startup scan and the Pydantic
validators are implementation. The externally observable semantics below are
**normative** and survive any reimplementation:

| # | Invariant |
|---|---|
| **I1** | Automatic rotation evicts the **smallest oldest-first victim set** that satisfies both ceilings, and **never evicts a manual snapshot** — not during rotation, not during startup repair |
| **I2** | A **normal, corruption-free, full ring always rotates**. The physical ceilings are configured with transient headroom for it, validated at startup |
| **I3** | Compatibility reports **mismatched** and **unavailable** artifacts **independently**; either set may be non-empty alone or together |
| **I4** | **No repository-controlled publication failure removes planned victims before a validated replacement has been atomically published and made durable.** Rotation deliberately deletes previously valid snapshots, so this is an ordering guarantee, not a permanence one: eviction is unreachable until `os.replace` **and** the snapshot-directory `fsync` have both succeeded. An interruption before replacement publication leaves the previous ring intact. After replacement visibility it may leave the repository over capacity when planned eviction has not completed, but it never removes planned victims before replacement durability is established |
| **I5** | A post-replacement failure is **surfaced as the state it actually is**, not swallowed and not flattened. `VISIBLE_DURABILITY_UNCONFIRMED` (directory `fsync` failed; no victim removed; durability unclaimed) and `PUBLISHED_RETENTION_CLEANUP_FAILED` (replacement durable; an oldest-prefix subset of victims already gone) are **distinguishable to the caller**. Unconfirmed durability is always surfaced; over-capacity is surfaced only where the repository genuinely exceeds a bound. The next startup repairs whatever is left over |
| **I6** | Canonical candidate ranks are exactly the permutation `1..N` |

### 11.1 What the implementation tests must cover

A **bounded, table-driven transition suite** — ordinary pytest parameterisation,
`tmp_path` and `monkeypatch`:

- rotation with zero, one and multiple victims;
- count ceiling and byte ceiling, independently;
- an oversized payload — the one case that genuinely cannot fit;
- startup repair after an interruption between publication and eviction;
- simultaneous artifact mismatch and partial unavailability.

**Injected failures assert two different contracts, and the boundary between them
is the directory `fsync`.** One blanket assertion across all injection points
would contradict I4: after durability, deleting a victim is the design working,
not a violation.

*Failure injected at write, file `fsync`, validation, `os.replace`, or the
directory `fsync`* — before replacement durability is established:

- **no planned victim is removed**; the previous ring is byte-for-byte intact;
- the temporary file is gone, or its survival is reported through health;
- on a directory-`fsync` failure specifically, the status is
  `VISIBLE_DURABILITY_UNCONFIRMED`, **eviction was never attempted**, and
  unconfirmed durability is reported **whether or not the ring was full** —
  over-capacity is asserted only in the case where a bound is genuinely
  exceeded, because a `fsync` failure on a half-full ring plans no victims and
  exceeds nothing.

*Failure injected during eviction* — after replacement durability is established:

- an **oldest-prefix subset** of the planned automatic victim set may be gone,
  and that is legal: assert the survivors are a suffix of the oldest-first order,
  never an arbitrary subset;
- **no manual snapshot and no non-victim automatic snapshot is gone**;
- the replacement is still present and valid;
- the status is `PUBLISHED_RETENTION_CLEANUP_FAILED` and health shows
  over-capacity;
- a startup scan from that state converges to I1 — the same victim set the
  uninterrupted run would have produced.

**No rollback machinery is tested because none exists.** A partially evicted ring
is repaired forward by the startup scan, not undone.

**No state-machine framework, model checker, Hypothesis dependency, transaction
journal, database or worker subsystem.** If the suite starts needing one, that is
a signal the design grew, not that the tooling is missing.

### 11.2 Reopen conditions

This document reopens only for one of these six triggers:

| # | Trigger |
|---|---|
| **T1** | Implementation exposes **externally observable behaviour that I1–I6 do not determine** — an ambiguity in the semantics, not in the prose |
| **T2** | A path is found that can leave **no old snapshot and no durable replacement** |
| **T3** | A **security, access-control or unsafe-deserialisation** finding against this design |
| **T4** | **Incompatibility with the supported deployment model** (§9, Spec 4 §3) |
| **T5** | A **contradiction between accepted invariants** |
| **T6** | **Measured deployment evidence** that an accepted bound (§7.2) or the synchronous-latency requirement (§7.3) cannot be met on institutional hardware |

T6 is the one trigger that cannot fire from reading. It fires from the numbers in
Spec 0 §3 value 3, and it is the reason the freeze does not depend on a
performance claim this container is unable to make.

Helper structure, branch mechanics and naming are **code-review** matters. They
do not open a specification round.
