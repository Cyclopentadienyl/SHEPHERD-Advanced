# Scorer measurement — work item B-0

Measure what the disease scorer actually does, before changing it. The offline
evaluator and the deployed pipeline scored different candidate sets with different
formulas, so neither of their numbers described the other. B-0 builds a harness
that measures the scorer the institution's policy approves, and B-0.5 is to measure
the served behaviour beside it.

| Mode | Encoder | Candidates | Scorer | Status |
|---|---|---|---|---|
| **C** | full graph | every disease in the KG | raw cosine | implemented: `scripts/measure_scorer.py`, `run_mode_c`. The formal Test's measurement (`../PLAN_TEST_RESULTS.md` D3) |
| **D** | full graph | production's path-reachable construction | η mixture with the SP term | B-0.5, not built; an unresolved design problem ([`PLAN_B03.md`](PLAN_B03.md)) |

**Modes A and B are retired** (the retirement's R2, 2026-10-12;
[`ASSESSMENT_MODE_AB_RETIREMENT.md`](ASSESSMENT_MODE_AB_RETIREMENT.md) §9). Mode A
scored each batch's 2-hop subgraph, seeded from the answers, with the subgraph
encoder, as the control the ladder was read against; Mode B scored the same
candidates with the full-graph encoder; A→B isolated encoder scope and B→C the
candidate universe. The formal Test's mode was decided (Mode C), Mode A could not
build a model from any checkpoint the current writer produces (`../BACKLOG.md`
19.18), and the institution authorised retiring the old pipeline (the owner's
clarification of 2026-10-12, below). The live guarantees Modes A and B protected
in passing were moved to direct tests first (R1, below). Their code is in git
history; nothing keeps it runnable.

**Owner's clarification (2026-10-12).** The owner reported on 2026-10-12 that the
institution had already authorised retiring the old pipeline, Modes A and B and the
ladder's comparison tools included; the new pipeline governs. The institution's AMP
question is whether a model *trained* with AMP is worse, a training study recorded
separately (`../BACKLOG.md` item 21), not the scoring comparison that the
differential's AMP report and item 7a's AMP leg made. Item 7a is cancelled, and 7b
dropped A/B and its 7a dependency. **Policy Gate 1 stays, and so does its committed
comparison:** the B-0 measurement report is to compare the currently served ranking
(Mode D, B-0.5) with the planned new GNN ranking (Mode C).

**C→D does not isolate one difference, and the table should not be read as though
it does.** It changes the candidate construction *and* the scorer at once, so a
metric difference between them cannot be attributed to either, nor to their
interaction. B-0.5 therefore needs an intermediate comparison — full-graph
embeddings, production's path-reachable candidates, **cosine** scoring — followed
by those same candidates scored with the η/SP mixture. That splits C→D into one
candidate-construction step and one scorer step. It is recorded here so the
roadmap does not promise an attribution it cannot make; it is not named, designed
or implemented.

| Stage | Scope | Status |
|---|---|---|
| B-0.1 | scoring primitives extracted from the pipeline | shipped |
| B-0.2 | harness, Mode A, both metric families, manifest, calibration launcher | shipped. **What remains is Mode C's:** the manifest, the authoritative metric family, the canonical ranking and the CUDA gate. The acceptance was redefined from bit-parity with the frozen evaluator, which was unexecutable, to a same-batch differential test against the trainer's own validation pass (`../BACKLOG.md` §3.1). The launcher `scripts/calibrate_mode_a.py` was removed with the frozen evaluator in contract M2.1's S0 ([`../PLAN_PROVENANCE_CONTRACT.md`](../PLAN_PROVENANCE_CONTRACT.md) M2.1, decisions 3 and 6). Mode A, the legacy metric family, the predictions artifact and the differential calibration were retired in R2 (2026-10-12); its institutional CUDA run, item 7a, was cancelled |
| B-0.3 | Modes B and C | shipped. Mode C is the formal Test's measurement; Mode B was retired with Mode A in R2. Its institutional run is 7b's Mode C side, under Policy Gate 1. Plan: [`PLAN_B03.md`](PLAN_B03.md), kept as the design record |
| B-0.4 | vectorised SP lookup | **measurement complete and reviewed.** Approach A selected for the primary GB10 platform (`PLAN_B04.md` §12.5): 8-33x faster on the caller production ships, 0 of 60 measurements over the provisional budget, at a cost of 3.44 GB permanent residence. **Production adoption is not cleared** — it waits on §13's integrated memory and reload gate, which needs A wired into production code and therefore its own plan and review. **Independent of the calibration decision** — it consumes `shortest_paths.pt` and no checkpoint, split or model. Plan: [`PLAN_B04.md`](PLAN_B04.md) |
| B-0.5 | Mode D, the intermediate candidate-construction step above, statistical protocol, institutional run | not started; Mode D has an unresolved design problem. Split: protocol and output-contract design come **before** any institutional run, so required evidence is not discovered after it |

[`PLAN_B02_shipped.md`](PLAN_B02_shipped.md) is the plan the shipped B-0.2 code
was built from, kept as the reasoning behind it. History, not authority.

## Running a measurement

```
python scripts/measure_scorer.py --checkpoint <ckpt> --data-dir <data> \
    --split val --output reports/measurement.json
```

The CLI measures Mode C; it has no mode option. It reads each input once,
verifies the workspace from those reads and records them (contract M2.1, S7),
then writes two files: the report at `--output` (manifest, authoritative metrics,
counts) and the per-sample ground-truth ranks beside it, `<output stem>_ranks.json`.

**`--split` has no default, and `val` is not held-out data.** It is the
split the current training configuration uses for early stopping and
checkpoint selection, and ordinary generated workspaces contain no test
split at all — `src/kg/sample_generator.py` writes train and val only.

**A run on a synthetic workspace is not a result.** The shared fixture is
degenerate on purpose (`tests/fixtures/synthetic_workspace.py`), so the tests take
Mode C's expected ranks from hand-built geometry
(`tests/unit/test_measurement_mode_c.py`), and the subprocess test
(`tests/integration/test_measure_scorer_run.py`) shows only that the command runs
and writes complete artifacts.

## The retirement of Modes A and B — implemented 2026-10-12, awaiting code review

> **Supersedes "Removing the legacy path — SUSPENDED, DO NOT EXECUTE".** That
> checklist was written when Mode A's acceptance was bit-parity with
> `scripts/evaluate_model.py`, and was corrected and suspended when that parity
> proved unexecutable. Its replacement, the differential calibration's
> institutional CUDA run (item 7a), was cancelled by the owner's clarification of
> 2026-10-12, and item 9 (rename, rewrite this checklist, delete the oracle-only
> surface) was replaced by the retirement's R1 and R2
> ([`ASSESSMENT_MODE_AB_RETIREMENT.md`](ASSESSMENT_MODE_AB_RETIREMENT.md) §4, §9).
> The suspended checklist and its removal order are in git history at `a535ae2`, the
> last revision before R2.

**The order it followed.** Steps 1–4 of the old removal order were done:
characterization tests freezing `Trainer._validate` and `Trainer.evaluate` (backlog
1b), the extraction of their shared pass (1c), the same-batch differential
calibration (1d) and the manifest additions (1e). Step 5's institutional run (7a)
was cancelled, and steps 6–8 were replaced:

| Step | Change | Done |
|---|---|---|
| R1 | the live guarantees Modes A and B protected in passing moved to their use points, with mutation checks (assessment §9.1) | 2026-10-11 |
| S7 (contract M2.1) | Mode C reads each input once and records those reads | 2026-10-11 |
| R2 | Modes A and B, their builder and loader, `run_mode_a`/`run_modes_ab` and their helpers, the differential calibration and the class-1 tests deleted; `--modes`, the measurement seed and the manifest fields Mode C does not use removed (assessment §8 items 5 and 9) | 2026-10-12 |

### What the retirement removed, and what stayed in the Trainer

Verified against `trainer.py`. Four of the five things "bit parity" was carried by
were trainer-validation shapes, and they are **live Trainer code**, not Mode A: only
measurement's copies went.

| Carrier | Measurement's copy | Where the Trainer does it | Its value test |
|---|---|---|---|
| phenotype-id `clamp` on `-1` padding | retired with Mode A | `trainer.py:795` | shape only (`test_trainer_validation_characterization.py`); a known gap (assessment §5.3) |
| `Tensor.sort(descending=True)` and its tie behaviour | `legacy_ranking`, retired | `trainer.py:684` | `tests/unit/test_trainer_validation_values.py` (a reversed sort fails it) |
| truncation at 20 | `LEGACY_TRUNCATION_K`, retired | `trainer.py:689`, `pred_indices[:20]` | the same file's wide-cohort test (a wrong twenty columns fails it) |
| the per-sample local top-20 rows | `ModeAResult.legacy_top_k_local`, retired | `trainer.py:686-689`, carried out as `_EvaluationPass.predictions` | the same file, row by row |
| `build_legacy_mode_a_model` — mirrored `create_model_from_checkpoint` (`scripts/evaluate_model.py` at `7dab728`) in its hardcoded `hidden_dim`/`num_layers`/`num_heads` fallbacks, and indexed `metadata` and `in_channels_dict` directly | retired | none: the trainer builds from real feature dims | — |

Mode C refuses where the Trainer clamps: a real id out of range is fatal and names
the patient (`_assert_ids_in_range`), because in Mode C every id is a real value
under a `True` mask. The malformed-truth contract (`../BACKLOG.md` D3, §2.3) keeps
three boundaries: the loader, the loss, and measurement's, which is now Mode C's
`_assert_ids_in_range` instead of Mode A's `to_global_ids`.

`MeasurementManifest.legacy_truncation_k` and `legacy_tie_policy` were **deleted,
not renamed**, as were the loader and sampler fields and the three seeds: Mode C uses
none of them. The Trainer's `[:20]` is a literal in Trainer code. An earlier
revision of this section planned to rename the two `legacy_*` fields; that is
superseded by the owner's decision of 2026-10-12 (assessment §8 item 5). Reports
and ledger records written before R2 keep the fields they were written with; R2
converts none and adds no branch for them: the ledger refuses a report whose
manifest carries a field the current one does not, and a ledger file written
before R2 (`LEDGER_SCHEMA_VERSION` is 2 since R2).

**What a later change must not need to touch:** `ModeResult`, `canonical_ranking`,
`ranks_of_truth`, `encode_full_graph`, `run_mode_c`, `build_shepherd_model`, the
manifest's authoritative fields, the digests, the CUDA gate, and
`src.kg.storage.file_storage`. If a change finds itself editing those, the boundary
has drifted and that is the finding. The list guarded the retirement and now guards
what follows it (Mode D). R2 itself changed two of them by decision: `run_mode_c`'s
`sampler_evidence` lost its `max_subgraph_nodes` and `negative_sampling` entries
(assessment §8 item 5), and their docstrings no longer cite Modes A and B; scores,
ranks, metrics and digests did not change. (`to_global_ids` was on this list while
Mode A used it; it went with Mode A.)

**The scorer stacks stay independent.** The Trainer's validation ranking is checked
against the served primitives `masked_mean_pool` and `cosine_score_matrix`
(`tests/unit/test_trainer_validation_values.py`). That comparison means something
only while the two are separately written, so `.import-linter.ini` keeps
`src.training` below `src.inference` (layers) and forbids
`src.inference.scoring` from importing `src.training` (`scorer-independence`).

**Do not pre-empt any of this with machinery** — no manifest subclass hierarchy,
no schema framework, no discriminated union over mode, no artifact registry or
compatibility database. When Mode D lands it adds its own fields and its own
values; it does not need a structure built in advance to receive them.

**Authority above everything here:** `docs/DISEASE_SCORER_POLICY.md`.
