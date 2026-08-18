# Scorer measurement — work item B-0

Measure what the disease scorer actually does, before changing it. The offline
evaluator and the deployed pipeline score different candidate sets with different
formulas, so neither of their numbers describes the other. B-0 builds a harness
that measures both, and a ladder of modes that isolates **one difference per
step** so a score change can be attributed rather than guessed at.

| Mode | Encoder | Candidates | Scorer |
|---|---|---|---|
| **A** | per-batch subgraph | the batch's subgraph, seeded from the answers | cosine |
| **B** | full graph | same as A | cosine |
| **C** | full graph | every disease in the KG | cosine |
| **D** | full graph | production's path-reachable construction | η mixture with the SP term |

A→B isolates encoder scope. B→C isolates the candidate universe. **A is the
control and preserves the legacy behaviour deliberately, including what is wrong
with it** — a control that has been improved is not a control.

**C→D does not isolate one difference, and this table should not be read as
though it does.** It changes the candidate construction *and* the scorer at once,
so a metric difference between them cannot be attributed to either, nor to their
interaction. B-0.5 therefore needs an intermediate comparison — full-graph
embeddings, production's path-reachable candidates, **cosine** scoring — followed
by those same candidates scored with the η/SP mixture. That splits C→D into one
candidate-construction step and one scorer step. It is recorded here so the
roadmap stops promising an attribution it cannot make; it is not named,
designed or implemented in B-0.3.

| Stage | Scope | Status |
|---|---|---|
| B-0.1 | scoring primitives extracted from the pipeline | shipped |
| B-0.2 | harness, Mode A, both metric families, manifest, calibration launcher | **implementation complete**; institutional CUDA run pending |
| B-0.3 | Modes B and C | [`PLAN_B03.md`](PLAN_B03.md) — three decisions, then implement |
| B-0.4 | vectorised SP lookup in the pipeline | not started; smaller than it looks — a caller change, not a new primitive ([`PLAN_B03.md`](PLAN_B03.md) §5) |
| B-0.5 | Mode D, the intermediate candidate-construction step above, statistical protocol, institutional run | not started; Mode D has an unresolved design problem |

[`PLAN_B02_shipped.md`](PLAN_B02_shipped.md) is the plan the shipped B-0.2 code
was built from, kept as the reasoning behind it. History, not authority.

**Authority above everything here:** `docs/DISEASE_SCORER_POLICY.md`.
