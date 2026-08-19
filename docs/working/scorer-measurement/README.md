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
| B-0.3 | Modes B and C | **implementation complete**; institutional CUDA run pending. Plan: [`PLAN_B03.md`](PLAN_B03.md) |
| B-0.4 | vectorised SP lookup in the pipeline | not started; smaller than it looks — a caller change, not a new primitive ([`PLAN_B03.md`](PLAN_B03.md) §5) |
| B-0.5 | Mode D, the intermediate candidate-construction step above, statistical protocol, institutional run | not started; Mode D has an unresolved design problem |

[`PLAN_B02_shipped.md`](PLAN_B02_shipped.md) is the plan the shipped B-0.2 code
was built from, kept as the reasoning behind it. History, not authority.

## Running the ladder

```
python scripts/measure_scorer.py --checkpoint <ckpt> --data-dir <data> \
    --split test --output reports/measurement.json --modes A,B,C
```

Mode A keeps `--output`'s name and its predictions artifact, so
`scripts/calibrate_mode_a.py` reads the same file it always did; B and C are
written beside it, one file per mode plus per-sample ranks.

**No cross-mode conclusion may rest on the synthetic fixture.** It is built so
the 2-hop subgraph *is* the whole graph, which is what makes the shared-cohort
claims checkable — and which also means A, B and C agree on it by construction.
A test asserts that agreement so it is not mistaken for a result.

## Removing the legacy path

The frozen evaluator does not match the paper's design and does not answer the
clinical question. It exists to calibrate the harness against the historical
number and **nothing else**, so it should come out cleanly when that calibration
succeeds — no pipeline refactor, no archaeology. This is the checklist, kept
current so removal is a deletion rather than an investigation.

| Delete | What it is |
|---|---|
| `scripts/evaluate_model.py` | The frozen oracle itself |
| `scripts/calibrate_mode_a.py` | Runs both scorers and compares them; has no purpose after |
| `scripts/measure_scorer.py`: `load_legacy_mode_a_inputs`, `build_legacy_mode_a_model` | The two entry points that mirror the oracle. **Nothing but Mode A reaches them** — Mode C reads through `src.kg.storage.file_storage` for exactly this reason |
| `src/evaluation/measurement.py`: `run_mode_a`, `run_modes_ab`, `legacy_ranking`, `LEGACY_TRUNCATION_K`, `ModeAResult` | **The whole A/B traversal**, the legacy ranking stream, and the result type that carries it. B is defined as A's candidates and is produced by A's loop, so the loop leaves with A. `ModeResult`, `canonical_ranking` and `run_mode_c` stay |
| Mode A's phenotype-id **clamp** in `run_modes_ab` | Oracle index parity on `-1` padding. It leaves inside the traversal above rather than separately; it is listed because it must not be carried into Mode C, whose ids are validated instead. It is correct only as parity; without an oracle it is a defect |
| `MeasurementManifest`: `legacy_truncation_k`, `legacy_tie_policy`, and the two lines populating them in `scripts/measure_scorer.py: build_manifest` | Fields describing the frozen oracle's truncation depth and tie behaviour. `build_manifest` sets them **unconditionally**, so a Mode C manifest carries them today although Mode C has no oracle and no legacy ranking stream. Delete the fields and their assignment; every authoritative field, the artifact digests and the CUDA metadata stay |
| `--modes A` and `A,B`, and the `A` branch of the CLI | Mode B is defined as *A's candidates*, so it goes with A. **C survives alone** |
| `tests/integration/test_legacy_equivalence.py`, the legacy tests in `tests/unit/test_measurement_mode_a.py` | Everything that asserts oracle parity |

**What must not need touching:** `ModeResult`, `canonical_ranking`,
`to_global_ids`, `ranks_of_truth`, `encode_full_graph`, `run_mode_c`,
`build_shepherd_model`, the manifest's authoritative fields, the digests, the
CUDA gate, and `src.kg.storage.file_storage`. If a removal round finds itself
editing those, the boundary has drifted and that is the finding.

The manifest itself is **not** on that list, and an earlier revision of this
checklist wrongly said it was: it carries two legacy fields that must go with the
oracle, as the row above records. **Do not pre-empt that edit with machinery** —
no manifest subclass hierarchy, no schema framework, no discriminated union over
mode. Deleting two fields and two assignments once is cheaper than any structure
built to avoid deleting them, and the structure would outlive the problem.

**Authority above everything here:** `docs/DISEASE_SCORER_POLICY.md`.
