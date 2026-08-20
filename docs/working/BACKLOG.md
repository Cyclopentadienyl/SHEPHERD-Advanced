# Programme backlog — what is open, in what order, and why

**Purpose.** One ordered list across every live phase, so the next action is
readable without reconstructing six review threads. Phase documents keep their
own detail; this file holds **ordering, dependencies and blockers only** and must
not restate their decisions.

**Status:** second revision. The calibration decision at item 1 is **made and
reviewed** (§3.1.2); the legacy-removal checklist it invalidated is corrected and
suspended. Two directions this file proposed were withdrawn along the way — §6
records them rather than hiding them.

---

## 1. The goal, unchanged

**Measure what the disease scorer actually does before changing it.** The offline
evaluator and the deployed pipeline score different candidate sets with different
formulas, so neither number describes the other. Work item B-0 builds a harness
and an A/B/C/D ladder that isolates **one difference per step**.

Everything below serves that, or is a consequence of discovering that part of it
does not currently work.

---

## 2. Established facts

M1-M7 were **measured on deployment hardware**; M8 was **read from source**. Both
are checkable; neither is recalled or inferred. These are what changed the
ordering.

| # | Fact | Where |
|---|---|---|
| M1 | **No checkpoint carries `metadata`.** All carry `config`, `data_fingerprint`, `epoch`, `logs` | checkpoint scan, both workspaces |
| M2 | **`in_channels` is 128** in every checkpoint; the frozen evaluator's hardcoded fallback is 256, which is the size mismatch it dies on | same |
| M3 | **The filename number is `val_mrr`** — `model-45-0.6975.pt` carries `logs["val_mrr"] = 0.69754…` | same |
| M4 | **100% of validation diseases appear in training** — 7,970 of 7,970. Train: 100,000 samples over 10,576 diseases; val: 15,000 over 7,970 | overlap audit |
| M5 | **SP reachability is dense**: a typical phenotype reaches 71.3% of diseases within 5 hops (16,845 of 23,640) | artifact scan |
| M6 | **The SP lookup misses the provisional latency budget by 1.7-2.5x** on the real artifact and a GB10 SPARK | `scorer-measurement/PLAN_B04.md` §9 |
| M7 | **Zero duplicate rows** on two independently built artifacts of different HPO vintages | same, §10 |
| M8 | **The trainer's own validation loop is Mode-A-shaped, not Mode-C-shaped** — per-batch subgraph forward, cosine against *the subgraph's* disease rows, top-20 truncation, MRR from the same function Mode A calls. See §2.1 | read from `trainer.py`, `metrics.py`, `measurement.py` |

### 2.1 M8 in detail — where `val_mrr` comes from

Read from source rather than recalled, because it overturns a direction this file
previously proposed.

| Step | `trainer._validate` | Mode A's legacy stream |
|---|---|---|
| Encoder input | `batch_data["subgraph_x_dict"]`, `subgraph_edge_index_dict` — the **per-batch subgraph** (`trainer.py:628-634`) | the same two keys off the same dataloader (`measurement.py:706-710`) |
| Padded phenotype ids | `phenotype_ids.clamp(min=0, max=…)` (`trainer.py:739`) | the same clamp, kept as oracle parity (`measurement.py:738`) |
| Patient vector | masked mean (`:744-751`) | `masked_mean_pool` (`:742`) |
| Candidates | `disease_emb = node_embeddings["disease"]` — **the subgraph's diseases**, not the KG's (`:762-763`) | `cosine_score_matrix(patients, disease_emb)`, same rows (`:743`) |
| Score | cosine, both sides `F.normalize`d (`:761-763`) | `cosine_score_matrix` |
| Ranking | `scores.sort(dim=-1, descending=True)` (`:651`) | `legacy_ranking`, which is that same call (`:151-157`) |
| Truncation | `pred_indices[:20]` (`:656`) | `LEGACY_TRUNCATION_K = 20` (`:243, :749`) |
| Metric | `RankingMetrics().compute_all(...)["mrr"]` (`:663-666`; `metrics.py:285`) | `RankingMetrics().mean_reciprocal_rank(...)` (`:809-810`) |

Every row is the same operation. **`val_mrr` is a Mode-A-shaped number, and M3
says every checkpoint carries it.**

**Every row above holds only when identical model weights, batch/subgraph
objects, ID mappings and numerical execution context are supplied.** It is a
statement about algorithmic shape. It is **not** a claim of historical execution
parity — see §3.1.1.

### 2.2 The D-list: what still has to be recorded or closed

**Most of this is already recorded.** `MeasurementManifest`
(`measurement.py:259-340`) already carries `batch_size`, `shuffle`,
`num_workers`, `negative_sampling_strategy`, `num_negative_samples`,
`subgraph_strategy`, `subgraph_hops`, `num_neighbors`, `max_subgraph_nodes`, the
three seeds, `device`, `torch_version`, `cuda_version`, `dtype`, `amp_enabled`,
`deterministic_algorithms`, `cudnn_deterministic`, `cudnn_benchmark`, `split`,
`n_samples` and `artifact_digests`. Re-specifying those would duplicate fields
that exist. Only the rows below are open.

| # | Open item |
|---|---|
| D1 | **Record the limitation, not a field.** Historical epoch RNG state was never saved, so the historical stochastic validation traversal is not exactly reproducible. Sample order in a *new* run is pinned by `shuffle=False` plus the samples digest |
| D2 | Add `amp_dtype` (only the boolean is recorded today) and the **observed** `torch.compile` execution state — an execution fact, not a requested config value. No compile-metadata machinery |
| D3 | **Closed from source.** `_remap_indices` (`data_loader.py:956-964`) maps `batch["disease_ids"]` through `node_mapping["disease"]` into subgraph-local space, the same space as the score-matrix columns; trainer and Mode A read the same remapped tensor. The invariant is **already enforced loudly**: `_assert_cohort_is_intact` raises on any absent truth and names id translation as a cause (`measurement.py:606-613`). What is missing is only the regression tests — see §2.3 |
| D4 | No excluded-sample list is required **while the run is fail-fast**. Record that the run does not silently skip samples and that ranked plus ground-truth-absent account for the intact cohort. `n_ground_truth_absent` is a count, **not** an excluded- or failed-sample list; if skip-and-continue is ever introduced, skipped ids and reasons become required then |
| D5 | Exact artifact identity. A/B/C already digest checkpoint, samples, node features, edge indices and num_nodes (`measure_scorer.py:79-92`). Open: the **shortest-path artifact digest, added when Mode D consumes it**, and comparing recorded digests against the institution-approved artifact set at acceptance. No registry, no compatibility database |
| D6 | Aggregate `val_mrr` is insufficient for parity and is a historical sanity reference only (§3.1.1) |

### 2.3 The D3 regression tests

The invariant to assert is stronger and simpler than a guard-boundary check:

> every valid sample's seeded ground-truth disease must remap to
> `0 <= local_truth < number_of_subgraph_disease_rows`.

The mapping tensor has length `max(sampled_global_id) + 1`, is initialised to
`-1`, and holds local indices only at sampled global-id positions
(`data_loader.py:358-366`). So a global id inside the tensor range but absent
from the subgraph maps to `-1`; an id beyond the tensor length is left unchanged
and will not normally be a valid local column, since the local candidate count is
no larger than the sampled-node count.

Cover: a successful seeded-truth mapping; a `-1` hole; an id beyond the mapping
tensor; local↔global round-trip identity; and equality of the truth id consumed
by the trainer reference and by Mode A.

---

## 3. What these facts broke

### 3.1 The calibration target did not exist — the root blocker, now decided

Two loaders fail on the same missing key, in two different ways:

- `build_legacy_mode_a_model` indexes `checkpoint["metadata"]` and
  `checkpoint["in_channels_dict"]` directly (`measure_scorer.py:189-190`).
  Neither exists (M1) — **`KeyError`**.
- `create_model_from_checkpoint` uses `.get()` with defaults instead
  (`evaluate_model.py:163-173`): three node types, two edge types, 256 channels.
  It therefore builds a **structurally wrong model** and dies later in
  `load_state_dict` against a real 128 (M2).

**Mode A cannot execute on any checkpoint this project has produced**, and neither
can the frozen evaluator.

**Everything downstream inherits this.** B-0.2 and B-0.3 are implementation-
complete but their acceptance was always "the harness reproduces the oracle".
Without an oracle run, the harness is unvalidated — and B-0.5 would build a
statistical protocol on top of it.

#### What Mode A is actually carrying

Mode A has been doing **two** jobs under one name, and only one of them is broken:

- **(a) the bottom rung of the ladder** — per-batch subgraph encoder, subgraph
  candidates, cosine. A→B isolates encoder scope only if A holds this shape.
- **(b) bit-parity with the frozen evaluator** — the clamp, the truncation at 20,
  the local-index prediction artifact, and the oracle-mirroring model builder.

Job (b) is what requires `checkpoint["metadata"]`, and job (b) is unexecutable.
Job (a) needs no oracle at all.

#### Directions, corrected by M8

| Direction | Verdict |
|---|---|
| Repair the frozen evaluator to derive `in_channels` from features | **Rejected.** It is frozen on purpose, and a repaired evaluator is a *new* evaluator, not a reproduction of a historical executable artifact |
| Accept that no oracle number exists and drop Mode A | **Rejected.** Discards the control the whole ladder is built around |
| ~~Calibrate **Mode C** against the trainer's `val_mrr`~~ | **Withdrawn — premise false.** The trainer's validation loop scores the batch's subgraph, not the full disease matrix (M8, §2.1) |
| ~~Calibrate **Mode A** against the stored `val_mrr`~~ | **Withdrawn — not reproducible, and insufficient even if it were.** See §3.1.1 |
| **Same-batch differential calibration against the trainer's own validation pass** | **Adopted.** §3.1.2 |

#### 3.1.1 Why the stored `val_mrr` is not an oracle

Two independent reasons, the first confirmed from source:

- **The historical batching cannot be reconstructed.** Mode A's candidate set
  *is* the batch's subgraph, and validation subgraphs and negatives are drawn
  stochastically. `grep -n "rng\|get_rng_state" src/training/callbacks.py
  src/training/trainer.py` returns **no matches** — the checkpoint saves epoch,
  `state_dict`, `optimizer_state_dict`, `logs`, `config`, optionally
  `scheduler_state_dict` and optionally `data_fingerprint`, and no RNG state. The
  candidate universe at epoch 45 depended on the RNG stream position after 45
  epochs of training, which is unrecoverable.
- **One aggregate scalar cannot establish parity anyway.** Per-sample ranking,
  candidate-set, tie-order and ID-mapping disagreements can cancel in aggregate.

Stored `val_mrr` is retained as a **historical aggregate sanity reference** and
nothing more.

#### 3.1.2 The adopted direction

Retire frozen-evaluator bit parity; keep Mode A as the trainer-validation-shaped
bottom rung; calibrate it with a **same-batch differential test**: build the real
model through `build_shepherd_model`, hand the *same* batch and subgraph objects
to a trainer-path reference calculation and to the Mode A harness, and compare
per-sample local top-20 ids, truth ranks, reciprocal ranks, then aggregate MRR.

**The reference is the trainer's own code, not a third implementation.**
`Trainer._validate` (`trainer.py:615-681`) and `Trainer.evaluate`
(`:773-849`) already duplicate this calculation; the shared pass is extracted
once, privately, and all three callers use it. Writing a fourth expression of a
calculation that must agree would leave the newest copy the only untested one.

**Non-tautology is enforced by the layer contract, not by discipline.**
`.import-linter.ini` orders `src.evaluation` above `src.inference` above
`src.training`, and a lower layer may not import a higher one — so
`src/training/` **cannot** import `masked_mean_pool` or `cosine_score_matrix`
from `src/inference/scoring.py` even if someone wanted to. `make lint-imports`
fails if it is attempted. The trainer keeps its own inline `F.normalize` +
`torch.mm` (`trainer.py:761-763`), which is what makes it a genuinely
independent expression. Review permitted primitive sharing; the architecture
forbids it in this direction, and the stricter rule is the one that holds.

The differential harness itself sits in `src.evaluation`, which is above both,
so it may import the trainer's extracted pass and the Mode A traversal. That
direction is legal and is the only one that needs to be.

**Its cost, stated plainly:** job (b) is retired. The legacy-removal checklist in
`scorer-measurement/README.md` was written around a parity that will now never be
demonstrated, and most of what it lists for deletion is machinery the replacement
keeps. That checklist is **corrected and suspended**, not executed.

### 3.2 Validation measures no unseen-disease generalisation at all

M4 is the maximal case: **every** disease in val appears in train. The escalation
already sustained by review said the generator "permits" overlap; the measurement
says it is total.

This does not invalidate any engineering result. It **bounds what every number
may claim**, and the bound is tighter than the caveat currently drafted.

### 3.3 A policy inference is contradicted

`DISEASE_SCORER_POLICY.md` §3.5 records, explicitly as unmeasured, that most
candidates fall outside the 5-hop table so the SP term degenerates to a
reachability indicator. M5 says the opposite. The correction is factual and
touches no normative statement.

---

## 4. Ordered backlog

Ordered by **dependency**, not by size. An item may only start when everything it
depends on is resolved.

| # | Item | Depends on | Owner | Size |
|---|---|---|---|---|
| **1** | **The calibration decision** — §3.1.2, adopted and reviewed | — | **decided** | — |
| **1a** | Correct and **suspend** the legacy-removal checklist | 1 | author | **done** |
| **1b** | Characterization tests freezing `Trainer._validate` / `Trainer.evaluate` observable behaviour | 1a | author | small |
| **1c** | Extract the pass those two already duplicate — private, narrow | 1b | author | small |
| **1d** | Same-batch differential calibration | 1c | author | the calibration itself |
| **1e** | D2 manifest additions (`amp_dtype`, observed compile state); D3 regression tests (§2.3) | 1c | author | small |
| **2** | Update the contamination caveat to the measured 100% (§3.2), with both split file hashes | — | author | small |
| **3** | `DISEASE_SCORER_POLICY.md` §3.5 correction (§3.3) | — | author | ~5 lines |
| **4** | Reply to the sustained-with-narrowing contamination review | 2 | author | text only |
| **5** | **B-0.4 prototype phase** — prototype A and B, both caller shapes, per-subprocess memory | — **independent of 1** | author | the next real engineering |
| **6** | Which checkpoint is authoritative. Engineering supplies hashes, logs, artifact-compatibility evidence and load results; the **institution decides**. The question must separate the *deployed* checkpoint from the one `select_checkpoint_in_dir` picks by the highest **contaminated** `val_mrr` — `model-22` winning that metric makes it neither clinically authoritative nor a held-out-generalisation winner | 2 | institution | question |
| **7a** | Engineering differential calibration run | 1d, D5 artifact set, a designated loadable checkpoint | author | blocked |
| **7b** | Institutional measurement (B-0.2 / B-0.3) | 7a, 2, 3, 6, deployment CUDA verification | both | blocked |
| **8a** | B-0.5 protocol and output-contract **design** | 1 | author | **before** any expensive run |
| **8b** | B-0.5 institutional execution | 8a, 7b, 6, exact artifacts, production-path prerequisites | both | blocked |
| **9** | Mechanical rename (~70 refs, 9 files), then rewrite the checklist, then delete the oracle-only surface | **1d passed review incl. its institutional CUDA run** | author | behaviour-neutral |
| **10** | **Commit evidence files for M1-M5.** M6 and M7 have `EVIDENCE_B04_*.json` beside their plan; M1-M5 have nothing — they exist only as text pasted into a review thread. Needs the raw scan and audit outputs, and the scripts that produced them | — | institution + author | small, see §5.1 |

**Parked deliberately, not forgotten:** `task-scope/` Q2–Q5 (settled, unscheduled)
and `scorer-retraining/` (scoping only, four gates uncleared). Neither blocks nor
is blocked by anything above.

---

## 5. Two orderings that are easy to get wrong

**Item 5 does not wait for item 1.** B-0.4 measures the shortest-path **lookup
cost**. It consumes no checkpoint, no sample split and no model — only
`shortest_paths.pt`. Both its gates are cleared (M6, M7), and its findings do not
depend on whether the harness is calibrated. The root blocker blocks the
**numbers**, not the **work**.

**Item 9 waits for everything.** The rename is behaviour-neutral and looks safe,
which is exactly why it must come last: the trainer helper's shape, the
per-sample result contract, manifest ownership and the calibration CLI are not
settled until 1d lands, so renaming earlier buys a second rename. And **nothing
oracle-only may be deleted until 1d has passed review including its institutional
CUDA run** — deleting the old acceptance path before the new one is accepted
would leave the harness with no acceptance at all.

**8a comes before 7b for a reason:** designing B-0.5's output contract after an
expensive institutional run is how required evidence gets discovered too late to
collect.

### 5.0 Is any of this order forced? Checked at file level

Asked because reordering was being considered, and "these are independent" is a
structural claim that decays.

| Line | Files it touches |
|---|---|
| **B-0.4** (item 5) | `scripts/benchmark_sp_lookup.py`; `src/inference/scoring.py` — `sp_mean_distances`, `SPLookup`; `tests/unit/test_scoring_primitives.py` |
| **Calibration** (1b-1e) | `src/training/trainer.py`; `src/evaluation/measurement.py`; `scripts/measure_scorer.py`; `scripts/calibrate_mode_a.py`; new trainer and D3 tests |

**Disjoint, and the layer contract keeps them that way.** The one file that could
have been shared is `src/inference/scoring.py` — it holds both the SP primitives
B-0.4 rewrites *and* the cosine primitives a naive extraction might have reached
for. `src.training` sits **below** `src.inference`, so the trainer cannot import
them; `trainer.py` today imports only `src.training.*` and `src.utils.*`, and
`make lint-imports` reports 3 contracts kept. A B-0.4 regression therefore has no
path to the calibration reference.

**Only two orderings are genuinely forced**, both already in the table:

- Item 9 (rename, then checklist rewrite, then deletion) after 1d's institutional
  acceptance — deleting the old acceptance path before the new one is accepted
  would leave the harness with none.
- 8a (protocol design) before 7b/8b — evidence requirements have to be known
  before an expensive run, not after.

**One soft coupling worth naming:** item 2's caveat quotes split file hashes, and
item 10's evidence file would carry hashes for the same claim. If they are
produced on different machines they will not match, and both must then say
**which workspace** they measured. That is the M7 situation and it is a strength,
not a conflict — two independent replications of a generator-level property —
but it must be written that way rather than discovered later.

### 5.1 An asymmetry in the evidence, recorded because it is load-bearing

M6 and M7 have committed artifacts — `EVIDENCE_B04_baseline_synthetic.json` and
`EVIDENCE_B04_artifact_spark.json` sit beside `PLAN_B04.md`, and anyone may
recompute from them.

**M1-M5 have no artifact.** They exist as text pasted into a review thread and
summarised here. That is the wrong way round: M1 and M2 are what established that
the calibration target does not exist — the largest decision this phase has made
— and M4 is what bounds every number the project will report. Those are precisely
the facts that most need to be independently checkable, and they are the ones a
reviewer has to take on trust.

Nothing here is disputed. The point is that "reviewed and approved" currently
means *approved on a summary* for five of the eight established facts, and the
existing `EVIDENCE_*.json` convention already shows what fixes it.

---

## 6. Contradictions found and resolved in this revision

Recorded because the request that produced this file was to stop them recurring.

| Contradiction | Resolution |
|---|---|
| `PLAN_B04` §5.3.2 required scanning "the deployed artifact" while §10.1 established there is no single one | §5.3.2 restated as evidence about the generator's invariant |
| `PLAN_B04` §9.4's estimated "+7 GB" stood beside §10.2's measured ~24 GB with no link | §9.4 marked as an estimate and pointed at the measurement |
| The `PLAN_B04` status header was spliced into a broken sentence by a scripted edit | Rebuilt as two named gates with their verdicts |
| `--split` defaulted to `test`, which the generator never writes | Made required on both entry points; the error names the splits that exist |
| This file's own first draft proposed calibrating **Mode C** against the trainer's `val_mrr`, on an unverified guess about the trainer's candidate universe | Verified before submission and **withdrawn**: the trainer scores the batch subgraph, so the number is Mode-A-shaped (M8). The direction table now records the withdrawal rather than deleting it |
| Its second draft then proposed calibrating **Mode A** against the *stored* `val_mrr` — right shape, but unreproducible and insufficient | Withdrawn on review. §3.1.1 gives both reasons; the stored value is demoted to a historical sanity reference |
| The legacy-removal checklist instructed deletion of the clamp, local ranking, top-20 semantics, the A/B traversal and two manifest fields — all of which the adopted calibration **keeps** | Checklist **corrected and suspended**, with the oracle-only surface separated out and the removal order fixed at eight gated steps |
| An earlier reply claimed an out-of-range disease id "would be read as if it were already local" | Imprecise. The mapping tensor is `-1`-initialised, so an in-range unsampled id maps to `-1`; an out-of-range id is left unchanged and is not normally a valid local column. §2.3 states the invariant instead of the guard |

**Authority above everything here:** `docs/DISEASE_SCORER_POLICY.md`.
