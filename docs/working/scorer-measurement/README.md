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
| B-0.2 | harness, Mode A, both metric families, manifest, calibration launcher | implementation complete, **acceptance redefined**: bit-parity with the frozen evaluator is unexecutable, and the replacement is a same-batch differential test against the trainer's own validation pass. See `../BACKLOG.md` §3.1 |
| B-0.3 | Modes B and C | implementation complete; institutional run inherits B-0.2's acceptance. Plan: [`PLAN_B03.md`](PLAN_B03.md) |
| B-0.4 | vectorised SP lookup | **both prototypes built and measured on the real artifact** (`PLAN_B04.md` §12). Approach A recommended, awaiting review: 8-34x faster on the caller production ships, 0/60 cells over the provisional budget, at a cost of 3.44 GB steady-state. **Independent of the calibration decision** — it consumes `shortest_paths.pt` and no checkpoint, split or model. Plan: [`PLAN_B04.md`](PLAN_B04.md) |
| B-0.5 | Mode D, the intermediate candidate-construction step above, statistical protocol, institutional run | not started; Mode D has an unresolved design problem. Split: protocol and output-contract design come **before** any institutional run, so required evidence is not discovered after it |

[`PLAN_B02_shipped.md`](PLAN_B02_shipped.md) is the plan the shipped B-0.2 code
was built from, kept as the reasoning behind it. History, not authority.

## Running the ladder

```
python scripts/measure_scorer.py --checkpoint <ckpt> --data-dir <data> \
    --split val --output reports/measurement.json --modes A,B,C
```

**`--split` has no default, and `val` is not held-out data.** It is the
split the current training configuration uses for early stopping and
checkpoint selection, and ordinary generated workspaces contain no test
split at all — `src/kg/sample_generator.py` writes train and val only.

Mode A keeps `--output`'s name and its predictions artifact, so
`scripts/calibrate_mode_a.py` reads the same file it always did; B and C are
written beside it, one file per mode plus per-sample ranks.

**No cross-mode conclusion may rest on the synthetic fixture.** It is built so
the 2-hop subgraph *is* the whole graph, which is what makes the shared-cohort
claims checkable — and which also means A, B and C agree on it by construction.
A test asserts that agreement so it is not mistaken for a result.

## Removing the legacy path — SUSPENDED, DO NOT EXECUTE

> **This checklist is superseded and must not be run.** It was written when Mode A's
> acceptance was bit-parity with `scripts/evaluate_model.py`. That target is
> unexecutable — no checkpoint in the scanned family carries the
> `metadata`/`in_channels_dict` keys either loader needs — and the replacement
> acceptance is a **same-batch differential test against the trainer's own
> validation calculation**. See `../BACKLOG.md` §3.1.
>
> **The hazard is concrete.** Most rows below name machinery that the replacement
> calibration *keeps*: the padding clamp, the local ranking, the truncation at 20,
> the per-sample local top-20 rows and the A/B traversal are all **also** what
> `Trainer._validate` does. An engineer executing this checklist in good faith
> would delete the calibration's own subject matter while believing they were
> following documented procedure.
>
> Nothing may be deleted until the differential calibration has passed review
> **including its institutional CUDA run**. The corrected boundary is below; the
> corrected checklist is written at step 7 of that order, not now.

### What is actually oracle-only

Verified against `trainer.py`, not recalled. Four of the five things "bit parity"
was carried by are trainer-validation shapes and survive:

| Carrier | Oracle-only? | Where the trainer does the same thing |
|---|---|---|
| phenotype-id `clamp` on `-1` padding | **no** | `trainer.py:739` |
| `legacy_ranking` — `Tensor.sort(descending=True)`, and its tie behaviour | **no** | `trainer.py:651`, the same call |
| truncation at 20 (`LEGACY_TRUNCATION_K`) | **no** | `trainer.py:656`, `pred_indices[:20]` |
| the per-sample local top-20 rows (`ModeAResult.legacy_top_k_local`) | **no** | `trainer.py:654-656` builds the same rows |
| `build_legacy_mode_a_model` — mirrors `create_model_from_checkpoint` **including its hardcoded fallbacks** | **yes** | the trainer builds from real feature dims |

So the genuinely oracle-only surface is: `scripts/evaluate_model.py`,
`build_legacy_mode_a_model`, `tests/integration/test_legacy_equivalence.py`, and
the oracle-parity assertions inside `tests/unit/test_measurement_mode_a.py`.
Everything else the old table lists is retained.

Two rows of the old table were wrong in a second way as well, and both are
recorded here so the correction is not lost when the checklist is rewritten:

- **`scripts/calibrate_mode_a.py` is not purposeless after the oracle goes.**
  Calibration still happens; only its reference changes. It is rewritten, not
  deleted.
- **`MeasurementManifest.legacy_truncation_k` / `legacy_tie_policy` describe
  surviving semantics.** They are renamed, not deleted. `model_construction`'s
  docstring ("Mode A mirrors the oracle deliberately") becomes false at the same
  moment and is corrected in the same commit.

### The order removal must follow

Reviewed and approved. Each step gates the next; step 6 is behaviour-neutral by
construction and step 8 is last.

| # | Step | Note |
|---|---|---|
| 1 | Correct and suspend this checklist | **done — this section** |
| 2 | Characterization tests freezing `Trainer._validate` and `Trainer.evaluate` observable behaviour | metric keys, loss aggregation, callback order and count, best-metric updates, forward count, local top-20 rows, truth ids, AMP placement, empty-result behaviour |
| 3 | Extract the pass those two already duplicate | private and narrow; no evaluation framework, protocol hierarchy, callback extension point or generic result subsystem |
| 4 | Same-batch differential calibration | non-tautological **only if `trainer.py` never imports or calls the harness's traversal or ranking**. Review permitted sharing `masked_mean_pool` / `cosine_score_matrix`; `.import-linter.ini` places `src.training` **below** `src.inference` and forbids it outright, so the trainer keeps its own inline `F.normalize` + `torch.mm` and independence is mechanically enforced. The harness sits in `src.evaluation`, above both, which is the one direction it needs |
| 5 | Bounded synthetic tests, then the institutional CUDA acceptance run | this is the deletion gate |
| 6 | One mechanical rename commit | ~70 references across 9 files. **No** scoring, ranking, tie, schema, CLI, builder or manifest behaviour change |
| 7 | Rewrite this checklist against the final boundary | |
| 8 | Delete the oracle-only surface | |

**Why the rename waits until step 6:** the trainer helper's shape, the per-sample
result contract, manifest ownership and the calibration CLI are not settled until
step 4 lands. Renaming first would buy a second rename.

**What must not need touching, at any step:** `ModeResult`, `canonical_ranking`,
`to_global_ids`, `ranks_of_truth`, `encode_full_graph`, `run_mode_c`,
`build_shepherd_model`, the manifest's authoritative fields, the digests, the
CUDA gate, and `src.kg.storage.file_storage`. If a step finds itself editing
those, the boundary has drifted and that is the finding.

**Do not pre-empt any of this with machinery** — no manifest subclass hierarchy,
no schema framework, no discriminated union over mode, no artifact registry or
compatibility database. Renaming fields once is cheaper than any structure built
to avoid renaming them, and the structure would outlive the problem.

**Authority above everything here:** `docs/DISEASE_SCORER_POLICY.md`.
