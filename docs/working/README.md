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
| [`ops-wiring-and-ui-ux/`](ops-wiring-and-ui-ux/) | **built, the external review's two P2 findings fixed, awaiting its narrow re-review** — built on `claude/ops-wiring-and-ui-ux` from `main` `30e8ae1` (2026-11-12): operations wiring (A) and the daily UI/UX flow (B), authorised by the owner on 2026-11-12 on the reviewer's rev 2 plan and its one-time authorisation; A and B done, checkpoints A and B passed internally; its README is the plan and the record; [`GB10_ACCEPTANCE.md`](ops-wiring-and-ui-ux/GB10_ACCEPTANCE.md) is the field contract of the round's new required probes for the GB10 acceptance's closing check, not run: the GB10 acceptance follows the external review, by the owner's instruction |
| [`research-comparison/`](research-comparison/) | **built, reviewed, accepted and merged** — built on `claude/research-comparison` from `main` `2da28cf` (2026-11-10): the research-comparison round, authorised by the owner on 2026-11-10 on the reviewer's rev 2 plan review; its README is the plan and the record; `GB10_ACCEPTANCE.md` is the field contract of the round's new required probes for the GB10 acceptance's closing check. Its GB10 acceptance passed on both deploy routes at `e60f23e` ([`research-comparison-acceptance-2026-11-12/`](research-comparison-acceptance-2026-11-12/README.md)), and it merged into `main` as #133 (`30e8ae1`, 2026-11-12; product code `e60f23e`); what is next is [`BACKLOG.md`](BACKLOG.md) §0 |
| [`results-review/`](results-review/) | under review — snapshot repository, results-review UX, deployment security. Of SPEC_4, the loopback bind and its guard, the launcher's display, cross-origin refusal and the operator controls refused off loopback (D8) were implemented in M5-1 (merged as #129); authentication, the multi-user state and a real systemd run on a deployment host are not |
| [`scorer-measurement/`](scorer-measurement/) | **its measurement Modes A–D all retired** — work item B-0: Modes A and B in R2 (2026-10-12); Mode D (B-0.5) designed in batch 2, built in batch 3, reviewed, run on a synthetic cohort on the GB10 (`PLAN_B05.md` §11–§12.3), and **removed in K7** (2026-10-29); Mode C, the formal Test's measurement until K6, left the command line in K8-6a and its executor in K8-6b (`f35bb93`); kept Mode C and D reports and records stay readable (policy rev 15). The owner archived the Mode C/D comparison (2026-10-27); BACKLOG 7b, 8a and 8b (the institutional measurement, its statistical protocol and run, not started) still name Mode D and depend on that archived B-0 machinery. Policy Gate 1 stands as written (B-0's Mode C against Mode D report, reproducible only from a revision before K7) and is not cleared. It gates B-1, which is not built, and does not block the research implementation (rev 12); until a revision is approved, it binds the clinical switch (rev 11). What replaces its B-0 prerequisite for task models is open: rev 12's proposal, not approved, the owner and then the institution (policy §6; `scorer-retraining/README.md`, "Not decided here, and open") |
| [`task-scope/`](task-scope/) | what the institution's supplied-short-list use case changes. Scope decisions settled; Q1's reserved `candidate_genes` interface is the gene task's input (K4, K6) and Q5, causal-gene scoring, was built as the gene task (work package B, merged as #131); Q2–Q4 unscheduled |
| [`scorer-retraining/`](scorer-retraining/) | **built and merged** — the paper-aligned two-stage design (revision 2, 2026-10-27), built in K0–K8 and merged into `main` as #131 (`db8ee5a`, 2026-11-09; product code `f2c2012`). *How it got there:* on `claude/paper-aligned-two-stage`, K0–K6 built, K7's retirement check done (2026-10-29, README §8), K8 completed per the converged scope (2026-11-08, README §8's K8 record; TM27 not changed); checked since (2026-11-09): the full suite on `f2c2012`, one external review of `af3a8a9..f2c2012`, and the GB10 acceptance at `f2c2012`, passed in an explicitly selected environment with a locally built `pyg-lib` ([`k8-acceptance-2026-11-09/`](k8-acceptance-2026-11-09/README.md); the prebuilt path at this code not exercised), whose review (`f2c2012..be898de`) accepted it and closed K8 with no P1 or P2 (its one P3 applied in `2b188c4`). What was deferred is README §8's K8 record; the open decisions, with who decides each, are README's "Not decided here, and open (2026-11-09)" (the policy's own list is its §2.1, "Who decides what"); what is next is [`BACKLOG.md`](BACKLOG.md) §0. No clinical gate cleared |

**Authority above everything here:** `docs/DISEASE_SCORER_POLICY.md` and
`docs/SP_SCORE_GUIDE.md`.
