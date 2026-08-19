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

| Phase | Status |
|---|---|
| [`results-review/`](results-review/) | under review — snapshot repository, results-review UX, deployment security |
| [`scorer-measurement/`](scorer-measurement/) | in progress — work item B-0, the A/B/C/D measurement ladder |
| [`task-scope/`](task-scope/) | what the institution's supplied-short-list use case changes. Scope decisions settled; the reserved-interface item is **implemented**, the rest unscheduled |
| [`scorer-retraining/`](scorer-retraining/) | scoping only — selecting a patient-encoder / score-family / objective bundle, and the checkpoint scorer schema. No gate cleared |

**Authority above everything here:** `docs/DISEASE_SCORER_POLICY.md` and
`docs/SP_SCORE_GUIDE.md`.
