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

A→B isolates encoder scope. B→C isolates the candidate universe. C→D isolates
candidate construction and the scorer together. **A is the control and preserves
the legacy behaviour deliberately, including what is wrong with it** — a control
that has been improved is not a control.

| Stage | Scope | Status |
|---|---|---|
| B-0.1 | scoring primitives extracted from the pipeline | shipped |
| B-0.2 | harness, Mode A, both metric families, manifest, calibration launcher | **implementation complete**; institutional CUDA run pending |
| B-0.3 | Modes B and C | [`PLAN_B03.md`](PLAN_B03.md) — three decisions, then implement |
| B-0.4 | vectorised SP lookup in the pipeline | not started; smaller than it looks (see the plan's §4) |
| B-0.5 | Mode D, statistical protocol, institutional run | not started; Mode D has an unresolved design problem |

[`PLAN_B02_shipped.md`](PLAN_B02_shipped.md) is the plan the shipped B-0.2 code
was built from, kept as the reasoning behind it. History, not authority.

**Authority above everything here:** `docs/DISEASE_SCORER_POLICY.md`.
