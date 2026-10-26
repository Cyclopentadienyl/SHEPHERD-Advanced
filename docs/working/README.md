# Working documents

Design proposals, specifications and review material for work **in progress**.
One subfolder per work phase. That is the whole convention.

**These are not for clinicians or the engineering team.** The documents in
`docs/` above this folder are written to be read by people who use or operate
SHEPHERD-Advanced. These are written for whoever is building and reviewing a
particular change, and they say things a finished document should not — open
questions, rejected alternatives, revision history, disagreements between author
and reviewer. The separation is by folder because that is enough; nothing here is
access-controlled and nothing here needs to be.

**A phase folder is not permanent.** When a phase lands, its normative decisions
move into the committed documents that outlive it — `docs/DISEASE_SCORER_POLICY.md`,
`docs/ARCHITECTURE.md`, an ADR — and the working folder is either deleted or left
as a record with a header saying it is one. A phase folder that is still here
after its work shipped is stale, not authoritative.

[`BACKLOG.md`](BACKLOG.md) sits beside the phase folders rather than inside one,
because it is the only thing here that spans them: **ordering, dependencies and
blockers across every live phase, and nothing else.** Each phase's own decisions
stay in its folder. Read the backlog to find what is next; read the phase
document to find what it means.

| Phase | Status |
|---|---|
| [`results-review/`](results-review/) | under review — snapshot repository, results-review UX, deployment security. Of SPEC_4, the loopback bind and its guard, the launcher's display, cross-origin refusal and the operator controls refused off loopback (D8) were implemented in M5-1 (merged as #129); authentication, the multi-user state and a real systemd run on a deployment host are not |
| [`scorer-measurement/`](scorer-measurement/) | in progress — work item B-0: Mode C implemented (the formal Test's measurement), Mode D (B-0.5) not started, its design problem open (`PLAN_B03.md` §4); Modes A and B retired (R2). Policy Gate 1 stands as written; Mode D's minimal design is BACKLOG §0's batch 2 |
| [`task-scope/`](task-scope/) | what the institution's supplied-short-list use case changes. Scope decisions settled; the reserved-interface item is **implemented**, the rest unscheduled |
| [`scorer-retraining/`](scorer-retraining/) | scoping only — selecting a patient-encoder / score-family / objective bundle, and the checkpoint scorer schema. No gate cleared |

**Authority above everything here:** `docs/DISEASE_SCORER_POLICY.md` and
`docs/SP_SCORE_GUIDE.md`.
