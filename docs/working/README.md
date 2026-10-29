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
| [`scorer-measurement/`](scorer-measurement/) | winding down since the owner archived the Mode C/D comparison (2026-10-27) — work item B-0: Mode C implemented (the formal Test's measurement until K6; a command-line run on history files until K8); Mode D (B-0.5) designed in batch 2, built in batch 3, reviewed, run on a synthetic cohort on the GB10 (`PLAN_B05.md` §11–§12.3), and **removed in K7** (2026-10-29), its kept reports readable; the statistical protocol and the institutional run (BACKLOG 8a, 8b) not started; Modes A and B retired (R2). Policy Gate 1 stands as written |
| [`task-scope/`](task-scope/) | what the institution's supplied-short-list use case changes. Scope decisions settled; the reserved-interface item is **implemented**, the rest unscheduled |
| [`scorer-retraining/`](scorer-retraining/) | in progress — the paper-aligned two-stage design (revision 2, 2026-10-27), built on `claude/paper-aligned-two-stage` in K0–K8: K0–K6 built, K7's retirement check done (2026-10-29, README §8), K8's integration and GB10 acceptance next. No clinical gate cleared |

**Authority above everything here:** `docs/DISEASE_SCORER_POLICY.md` and
`docs/SP_SCORE_GUIDE.md`.
