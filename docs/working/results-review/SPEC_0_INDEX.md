# Spec 0 — Index, traceability and open decisions

**Status:** draft, under review. Entry point for the results-review design.

The single rev 1–6 proposal is **frozen as a historical record** (`ARCHIVE_rev1_6.md`) and is no
longer an implementation authority. It had grown to mix normative policy, UX, SP mathematics,
snapshot schema, repository transactions, evidence semantics, threat model, source audit,
institutional decisions, backlog and six revision histories — at which point cross-section drift
became a correctness risk rather than a readability one, and it exceeded what a reviewer could ingest
in one pass.

**Scope correction.** Two independent audits — one by the reviewer, one by the author — converged on
the same conclusion: successive review rounds had promoted individually valid findings into blocking
requirements without weighting them against the actual deployment, and a bounded ten-entry recovery
feature for a single-workspace clinical tool had acquired a records-management platform. The specs
below keep the correctness and operational findings and defer the rest, each with a **named trigger**
that would revive it. What was removed, and why, is in the archive.

| Spec | Scope |
|---|---|
| **1 — Results review** | Scorer/view authority, SP sort/filter policy, decomposition, two-surface UX, limits |
| **2 — Snapshot repository** | Payload schema, rotation depth, age bound, quotas, atomic publication, and the access decision. **§5 and §7 are frozen; §11 carries the six normative invariants and the reopen conditions** |
| **3 — Evidence and action provenance** | Constraints any Gate 3 evidence design must satisfy, and where a clinician action records its view state |
| **4 — Deployment security** | Route inventory and classes, bind modes, authentication, admin policy, CORS, risk acceptance |
| **Archive** | Rev 1–6 logs, withdrawn claims, and what the scope audit removed with each revival trigger |

**Authority above all of these:** `docs/DISEASE_SCORER_POLICY.md` and `docs/SP_SCORE_GUIDE.md`.

---

## 1. Requirements traceability

| # | Requirement | Spec | Status |
|---|---|---|---|
| R1 | 200+ candidates | 1 §6 | design complete; `selection_limit` values [OPEN] |
| R2 | Pagination, selectable page size | 1 §2, §6 | design complete; `page_size` default [OPEN] |
| R3 | Sort by SP | 1 §3.1 | design complete |
| R4 | Range filter by SP | 1 §3.5, §3.6 | design complete |
| R5 | Filter defaults off | 1 §3.6 | decided |
| R6 | Score decomposition | 1 §4 | design complete |
| R7 | Result storage | 2 | write path design complete; history path needs the access decision (Spec 2 §9) |
| R8 | Low-friction navigation | 1 §5.1 | design complete; one Gradio behaviour to confirm on the deployed version |
| R9 | `selection_limit` bounds the clinician's own wait | 1 §6.1 | decided; **blocked** on the vectorised SP lookup (1 §6.2) |
| R10 | Multi-user | 4 §6 | correctly deferred |

---

## 2. Gates

| Gate | Blocks | Cleared by |
|---|---|---|
| **Access decision** | Snapshot list/load/delete and manual snapshot creation | **Any of three documented paths** — local single workspace, authenticated, or accepted protected segment (Spec 2 §9) |
| **Eager SP performance** | R9 at 200 candidates | Vectorised SP lookup, benchmarked on institutional hardware (Spec 1 §6.2) |
| **Gate 3 — per-target evidence** | Evidence implementation | A written design meeting Spec 3 §2 |

Admin and training route exposure is **not a gate on this design** — it is a defect in the running
system, and it sits in the P0 list below.

**One operational answer shapes the rest:** *does anyone reach the WebUI by the server's LAN address,
rather than through SSH?* It decides whether the bind returns to loopback, and which of the three
access paths applies. It does **not** block R7: path C exists precisely so the institution can decide
its own posture rather than have one imposed.

---

## 3. Open institutional values

| # | Value | Spec |
|---|---|---|
| 1 | `selection_limit` default, minimum, maximum | 1 §6 |
| 2 | `page_size` default | 1 §6 |
| 3 | Interaction latency target — **inference + eager analysis + snapshot publication** | 1 §6 |
| 4 | Snapshot folder path; **separate automatic and manual** count and byte quotas | 2 §7.2, §10 |
| 5 | Default rotation depth (2–10) | 2 §7.1 |
| 6 | Maximum snapshot age (**opportunistic, not hard**); whether it applies to manual snapshots | 2 §7.2 |
| 7 | Who may delete automatic and manual snapshots | 2 §7.2 |
| 8 | Encryption at rest, or an encrypted-volume dependency | 2 §8 |
| 9 | Backup inclusion for the snapshot directory | 2 §10 |
| 10 | **The access decision before history is exposed** — path A, B or C | 2 §9 |
| 10a | **What C5 actually requires** — active sort and filters only, or page and scroll too. Decides CSV provenance | 3 §5.1 |
| 11 | Whether multi-user is a real scenario | 4 §6 |
| 12 | Whether triage state is wanted | 1 §7 |
| 13 | **Whether anyone uses the server's LAN address** | 4 §4 |

---

## 4. Backlog

| Priority | Item | Spec |
|---|---|---|
| **P0** | Audit and convert **every client-reachable `torch.load`**, including the final model load | 4 §2.1 |
| **P0** | Fix the launcher's display: report bind and destinations separately | 4 §4 |
| **P0** | Fix the unbounded eager export accumulation | 2 §9 |
| **P0** | Document the actual deployment contract; withdraw the concurrency-safety claim | 4 §4, §5 |
| **P0** | Bind defaults to loopback, with the risk-acceptance opt-in — *needs value 13* | 4 §3.1 |
| **P0** | Fix the systemd unit's **invalid module path** — broken under every mode, does not wait for the bind decision | 4 §4 |
| **P0** | Correct the **host values** in launch instructions and the service unit — *needs value 13* | 4 §4 |
| **P0 — M3 only** | **Unmount or protect** admin and training routes (hiding from OpenAPI is not protection) | 4 §2, §4 |
| **P1** | Payload schema (discriminated union), load sequence, protected directory, permissions, bounds | 2 §4, §5, §8 |
| **P1** | Atomic publication under one repository lock; **no index**; failure surfacing | 2 §2, §7.3, §7.4 |
| **P1** | Rotation (depth ≥ 2), the age bound, and the **separate manual quota** | 2 §7.1, §7.2 |
| **P1** | Automatic write-on-production — *after the §8 preconditions* | 2 §6, §9 |
| **P1** | Vectorised SP lookup, benchmarked | 1 §6.2 |
| **P1 — needs the access decision** | Snapshot list / load / delete, and manual snapshot creation | 2 §9 |
| **P2** | Per-target deterministic evidence contract (Gate 3) — then the captured/current separation | 3 §2, §3 |
| **P4** | Multi-user and authentication | 4 §6 |

**Deferred with a named trigger**, so nothing is silently dropped: database-backed metadata,
retention-hold modelling, authenticated integrity, canonical serialisation, raw-byte checksum sidecar,
and **workspace / actor / tenant identity** (Spec 2 §10); durable audit state machine and
evidence-version lineage (Spec 3 §3, §5).

The snapshot-directory `fsync` is **no longer on that list**: it was deferred as hardening, and the
freeze review established that it is a precondition of the publication ordering rather than an
improvement on it. It is normative in Spec 2 §7.3 step 6.

---

## 5. Hardware and what a local test run does not prove

**CUDA is a hard project requirement.** Every deployment environment has it; the container these
specs were written in does not. That is an environment limitation, not a property of the design, and
it sets what a green local run is allowed to claim:

| | Status |
|---|---|
| CPU tests | fast unit coverage, run everywhere |
| CUDA and supported AMP tests | **required acceptance checks**, written here, run on deployment hardware |
| GPU latency and memory | measured on institutional hardware only |

> **A skipped test is not a passed test.** Local skips do not count as completed hardware validation,
> and no acceptance claim may rest on them.

---

## 6. Evidence base

All specs cite `5c5c4b2` (main, after PR #99) plus branch commits on
`claude/dev-context-review-3wuh05` — **committed and pushed, under review, not merged**:

| Commit | Contribution |
|---|---|
| `42e5d1e` | Scorer-policy amendment |
| `337266f` | `src/inference/scoring.py` — the B-0 extraction |
| `dc9663b` | Mixture float32 fix; regression tests at the production wrapper; `sp_mean_distances` float64 |
| `7dab728` | Fractional multi-phenotype precision regression; corrected SP precision explanation; removed the fabricated `mean_rank` report |
| `582ad9a` | B-0.2 step 1 — batched scoring primitives (`masked_mean_pool`, `cosine_score_matrix`) |
| `8ce8805` | B-0.2 step 2 — `RankingMetrics.compute_from_ranks`, making mean rank computable |
