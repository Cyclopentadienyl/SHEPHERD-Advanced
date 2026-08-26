# Programme backlog — what is open, in what order, and why

**Purpose.** One ordered list across every live phase, so the next action is
readable without reconstructing six review threads. Phase documents keep their
own detail; this file holds **ordering, dependencies and blockers only** and must
not restate their decisions.

**Status:** sixth revision. The calibration decision at item 1 is **made and
reviewed** (§3.1.2); the legacy-removal checklist it invalidated is corrected and
suspended. B-0.4's measurement is **complete and reviewed**, and its
productionisation is a separate item (5a) behind its own gate. This revision adds
**M9 and item 11**: the standard generated workspace has no held-out split and no
protocol defines one — known in two passing mentions and owned by nothing. Item
11's scope is the **unit** of holdout, not a filename (§3.4).

**Next action: item 11** — the evaluation-holdout protocol decision, which is the
only remaining item that is neither blocked on institutional evidence nor on a
designated checkpoint. Items 1 through 1e are complete. Two directions this file proposed were withdrawn along
the way, and several of its factual claims have been narrowed or re-cited under
review — §6 records all of them rather than hiding them.

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

M1-M7 were **measured on deployment hardware**; M8 and M9 were **read from
source**. Both kinds are checkable; neither is recalled or inferred. These are
what changed the ordering.

| # | Fact | Where |
|---|---|---|
| M1 | **No scanned checkpoint carries `metadata` or `in_channels_dict`.** All carry `config`, `data_fingerprint`, `epoch`, `logs`, `state_dict`. The claim is about the **current producers and the scanned family**, not about every checkpoint the project has ever written — no historical audit was run, and none is needed to reject the frozen evaluator as the acceptance oracle. **Confirmed by artifact** over 15 checkpoints in two families, `hgt` (10) and `gat` (5): both watched keys absent in all 15, the four expected keys present in all 15, nothing unreadable | [`EVIDENCE_M1_M3_hgt.json`](EVIDENCE_M1_M3_hgt.json) `39176ea3…`, [`EVIDENCE_M1_M3_gat.json`](EVIDENCE_M1_M3_gat.json) `5473fedb…` |
| M2 | **`in_channels` is 128** in every checkpoint; the frozen evaluator's hardcoded fallback is 256, which is the size mismatch it dies on. **Confirmed by artifact**: read from `feature_encoder.projections.<node_type>.weight` rather than from a config field, 45 projections across 15 checkpoints, every one 128, none without a readable width | same |
| M3 | **The filename number is `val_mrr`** — `model-45-0.6975.pt` carries `logs["val_mrr"] = 0.69754…`. **Confirmed by artifact**: 13 agreements, **0 disagreements**, each compared at the precision its filename was written to; the 2 uncomparable are both `last.pt`, which carries no score. `val_mrr` is the ranking metric in all 15 | same |
| M4 | **100% of validation diseases appear in training** — 7,970 of 7,970. Train: 100,000 samples over 10,576 diseases; val: 15,000 over 7,970. **Confirmed by artifact, every digit**, with both split digests recorded | [`EVIDENCE_M4.json`](EVIDENCE_M4.json) `b3aed32a…` |
| M5 | **SP reachability is dense** — but the recorded figure was wrong and is superseded. Measured on `shortest_paths.pt` `7268900c…`: the **median** phenotype reaches **19,216.5 of 29,866** diseases (**64.3%**) within the configured 5 hops; the **mean** is 16,845.5 (**56.4%**); q1 51.2%, q3 68.9%, max 78.2%; **270 of 19,836 phenotypes reach none**. Dense is a property of the graph, not of one node — q1 is already above half. The recorded "71.3% (16,845 of 23,640)" divided a mean by a denominator from a **different artifact**; see §2.4 | [`EVIDENCE_M5.json`](EVIDENCE_M5.json) `58d79584…` |
| M6 | **The SP lookup breaches the provisional latency budget in 22 of 60 measurements**, worst 3,722 ms, on the real artifact and a GB10 SPARK. Approach A brings that to **0 of 60** at a cost of 3.44 GB permanent residence; the earlier "1.7-2.5x" figure came from a different artifact vintage and host and is superseded | `scorer-measurement/PLAN_B04.md` §12.3 |
| M7 | **Zero duplicate rows** on two independently built artifacts of different HPO vintages — evidence about the generator's invariant, not clearance for one file; every future artifact is protected by the load-time assertion | same, §10.1 |
| M8 | **The trainer's own validation loop is Mode-A-shaped, not Mode-C-shaped** — per-batch subgraph forward, cosine against *the subgraph's* disease rows, top-20 truncation, MRR from the same function Mode A calls. See §2.1 | read from `trainer.py`, `metrics.py`, `measurement.py` |
| M9 | **The standard generated workspace has no held-out test split.** `sample_generator.py:97-111` writes train and checkpoint-selection val only, and `val` is what `early_stopping_monitor=val_mrr` selects on. The measurement tools **can** consume a separately supplied `test` cohort — `--split` accepts it and `read_samples` would load it — but **no accepted protocol currently defines, creates, freezes or proves the independence of one** | read from `sample_generator.py`, `trainer.py:124`, `measure_scorer.py:275-277` |

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
says every scanned checkpoint carries it.**

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
| D2 | **Closed by 1e.** `amp_dtype` and `torch_compile_wrapped` are on `MeasurementManifest`, and the latter is on `DifferentialResult` too — that is the artifact whose regime actually varies. **The field is named for what its probe can see**: `isinstance` against the dynamo wrapper proves the object is wrapped, not that a compiled graph ran, since graph breaks and eager fallback leave a wrapped model running eagerly. Renamed from `torch_compiled` in review round 1 for the reason `cuda_executed` was renamed from `calibration_eligible`. Two corrections to this row's own wording: `amp_enabled` was not "recorded", it was **hardcoded `False`** in `build_manifest`; and a structural claim is not an observation, so 1e made `assert_no_autocast` **refuse** to run either traversal inside an autocast block rather than let the manifest describe a run that did not happen. **Superseded by Proposal B**, which removed that ground instead of living with it: the fields now record the regime observed at the computation that produced each mode's numbers, and `assert_manifest_describes_regime` refuses only the case a single field cannot describe — numbers produced under one regime and recorded under another. `observe_torch_compile_wrapper` is one tri-state function, not compile-metadata machinery. **`isinstance` against the imported wrapper class is the only evidence**, aggregated over two import paths (`torch._dynamo`, `torch._dynamo.eval_frame`) so a release that moves one still answers exactly. No class resolves -> `None`. **There is no attribute fallback, and three review rounds are why.** R1: the first version turned "not observable" into `False`, never reached its own documented fallback, and claimed "never raises" while `hasattr` propagates any `non-AttributeError`. R2: the repair over-corrected — `_orig_mod` alone became `True`, letting **any** object opt in by naming a field, and the test written for it froze that false positive as a specification. R3: pairing the attribute with a `torch._dynamo` module prefix did not rescue it either — `startswith` also matches `torch._dynamoevil`, and `__module__` is **assignable class metadata**, so it records what a class says about itself rather than how it was built. `_orig_mod` is a necessary marker, never sufficient evidence, and no arrangement of weak signals makes it so. Every mechanism confirmed by execution, and each superseded version fails cases the next one added |
| D3 | **Closed.** `_remap_indices` (`data_loader.py:956-964`) maps `batch["disease_ids"]` into subgraph-local space, the same space as the score-matrix columns, and trainer and Mode A read the same remapped tensor. The range invariant is enforced at **three independent boundaries** — loader, loss, harness — none of them in the CUDA hot path. See §2.3. **The legal-truth equality test is done (1e)**, and it is not the one 1d already had: both 1d cohorts are identity-mapped (`original_indices["disease"] == [0..n-1]`), under which a side reading the wrong id space produces the right numbers anyway. 1e adds a hand-built batch with a non-identity, non-sorted map so local != global. A trainer emitting global ids is refused there by **`DiagnosisLoss`**, not by the harness — the boundary §2.3 measured, confirmed by execution after the test's first draft predicted the wrong one |
| D4 | No excluded-sample list is required **while the run is fail-fast**. Record that the run does not silently skip samples and that ranked plus ground-truth-absent account for the intact cohort. `n_ground_truth_absent` is a count, **not** an excluded- or failed-sample list; if skip-and-continue is ever introduced, skipped ids and reasons become required then |
| D5 | Exact artifact identity. A/B/C already digest checkpoint, samples, node features, edge indices and num_nodes (`measure_scorer.py:79-92`). Open: the **shortest-path artifact digest, added when Mode D consumes it**, and comparing recorded digests against the institution-approved artifact set at acceptance. No registry, no compatibility database |
| D6 | Aggregate `val_mrr` is insufficient for parity and is a historical sanity reference only (§3.1.1) |

### 2.3 D3 — where the malformed-truth invariant is enforced

The invariant at issue:

> every valid sample's seeded ground-truth disease must remap to
> `0 <= local_truth < number_of_subgraph_disease_rows`.

**How a violation could arise.** The mapping tensor has length
`max(sampled_global_id) + 1`, is initialised to `-1`, and holds local indices
only at sampled global-id positions (`data_loader.py:358-366`). A global id
inside the tensor range but absent from the subgraph therefore maps to `-1`; an
id beyond the tensor length is left unchanged by `_remap_indices`' own guard. In
Mode A the truth is one of the subgraph's seed nodes, so neither should happen —
which is exactly why it needs a guard rather than an assumption.

**Mode A already enforces it, and the check is already tested.** `to_global_ids`
raises on both failures *before* any indexing (`measurement.py:91-100`):

```
local id 4 is outside the subgraph's 4 disease nodes   # >= n_rows
local ids must be non-negative                          # the -1 hole
```

`tests/unit/test_measurement_ranking.py:52-56` parametrises exactly `[3]`, `[99]`
and `[-1]`. A `-1` never reaches PyTorch's negative indexing on this path, so it
cannot silently select the last candidate.

An earlier revision of this file credited `_assert_cohort_is_intact` with this.
That was the wrong citation: that function checks absent canonical truths and
cohort counts, not local-truth range. The conclusion held; the reference did not.

**The trainer refuses too — through the loss, not the metric.** An earlier
revision of this file claimed the trainer silently scored such a truth `0.0`
into `val_mrr`. That was wrong: it traced `_validate`'s metric fragment
(`trainer.py:648-657`) without checking what runs before it. `self.loss_fn(...)`
is called first (`:640`), `MultiTaskLoss` invokes `DiagnosisLoss` whenever
`diagnosis_scores` and `diagnosis_targets` are present
(`loss_functions.py:513-516`), and that reaches the malformed target through
`cross_entropy`, `gather` and `scatter_`. Measured on both branches:

```
label_smoothing=0.0  -1 hole  -> IndexError:   Target -1 is out of bounds.
label_smoothing=0.0  >= n     -> IndexError:   Target 5 is out of bounds.
label_smoothing=0.1  -1 hole  -> RuntimeError: index -1 is out of bounds ...
label_smoothing=0.1  >= n     -> RuntimeError: index 5 is out of bounds ...
```

`mean_reciprocal_rank` *is* permissive in isolation — it scores `"-1"` as `0.0` —
but the path never reaches it. **The primitive is permissive; the path is not.**

So there is no raise-versus-plausible-number asymmetry, only a difference in
where the error surfaces and how it reads.

**The contract is REFUSE, and it is not an open decision.** A malformed local
truth means the subgraph seeding, the id remap, the candidate alignment or the
batch wiring is wrong. Scoring it as a rank miss would convert a data-pipeline
failure into apparent model error and contaminate the loss, `val_mrr`, early
stopping and checkpoint selection.

**What was still worth fixing** — and is now done: the disease gather was
guarded by `disease_ids.clamp(min=0, max=...)`, a *silent correction* sitting
where the refusal belongs. It protected nothing, since the loss rejects the same
ids two statements later, and it made the refusal an accident of which tensor got
passed where — a later edit passing the clamped tensor as `diagnosis_targets`
would have removed the refusal without touching anything that looked
load-bearing. The clamp is gone and the gather uses the original ids.

**The explicit check lives at the loader, not the trainer, and the reason is
performance.** A first version put it in `_compute_model_outputs`, where
`disease_ids` has already been through `_move_to_device`; `bool(t.any())` on a
CUDA tensor forces a **host-device synchronisation on every valid training and
validation batch**. That is a permanent throughput cost to catch a condition that
cannot occur. It now sits in
`DiagnosisDataLoader._assert_disease_truth_in_range`, on the host, immediately
after the remap that creates the failure.

**Three boundaries, each small and local, none shared:**

| Boundary | Where | Covers |
|---|---|---|
| `_assert_disease_truth_in_range` | `data_loader.py`, CPU | the point the `-1` hole is created |
| `DiagnosisLoss` | `loss_functions.py` | any caller that bypasses that loader |
| `to_global_ids` | `measurement.py:91-100` | the measurement harness's own boundary |

No cross-module validator: three checks of a few lines each keep the coupling
lower than one shared one would. A regression test asserts that **no `bool()` is
taken of the disease-id tensor in the hot path**, so the CUDA sync cannot be
reinstated by a later well-meaning edit.

The **phenotype** clamp at `trainer.py:739` is a different thing and stays:
`diagnosis_collate_fn` pads phenotype ids with `-1` by design and the mask
discards those positions. Disease truths are never padded.

Still to cover, and only reachable once 1c exists: **equality of the legal truth
id consumed by the trainer reference and by Mode A** on the same batch.

---


### 2.4 What M5's recorded figure got wrong

Recorded as *"a typical phenotype reaches 71.3% of diseases within 5 hops (16,845
of 23,640)"*. The measurement that replaced it disagrees in two independent ways,
and the sentence above is retired rather than adjusted.

**The denominator came from a different artifact.** 23,640 appears in
`scorer-measurement/PLAN_B04.md` §9.1 as **"Disease targets"** for
`shortest_paths.pt` SHA-256 `9ada0c1a…` — 19,540 phenotypes, 429,971,678 rows.
The artifact audited here is `7268900c…` — 19,836 phenotypes, 430,585,772 rows —
and its graph holds **29,866** diseases, agreed between `num_nodes.json` and the
producer's own sidecar. Whatever "Disease targets" counted, it is not this
graph's disease count, so 71.3% is a ratio across two artifacts.

**What "Disease targets" counted is not established here, and nothing needs it.**
It may be an earlier vintage's disease count or the number of diseases actually
present in that table; this run does not say, and M5's schema deliberately does
not record a second denominator. Item 3 — the `DISEASE_SCORER_POLICY.md` §3.5
correction — needs only that reachability is *dense*, and that holds under either
reading: the median phenotype reaches 64.3% of the graph's diseases and q1
reaches 51.2%.

**"Typical" was a mean.** Mean 16,845.5 and median 19,216.5 differ by 2,371
diseases, 7.9 percentage points. The word chosen implied the second and the
number reported was the first.

**A prediction recorded in §5.2 of the previous revision was wrong, and is
corrected there.** It said the recorded figure was probably an overestimate
*because* phenotypes reaching no disease have no rows and would be dropped by a
count taken over the table. The mechanism is real and the direction was right,
but the magnitude is not: **270 of 19,836 phenotypes (1.36%)** reach nothing, far
too few to move a median by fifteen points. The denominator is the cause.

**Artifact vintages differ across the facts.** M6 and M7 were measured on
`9ada0c1a…`; M5 here is on `7268900c…`. Both are recorded by digest so the two
are never silently compared.

**One property re-confirmed in passing.** M7 recorded zero duplicate
`(phenotype, target)` pairs on two independently built artifacts. This is a
third, of a newer vintage: **0 duplicate pairs collapsed** out of 334,147,192
disease rows.

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

**Mode A cannot execute on any checkpoint in the scanned family**, and neither
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

#### 3.1.3 What 1d found: bit-exactness is a contract only when AMP is off

§3.1.2 specified the comparison but not the precision it runs at, and the two
paths do not agree about that.

- `Trainer._run_evaluation_pass` forwards inside
  `autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp)`.
- The Mode A traversal in `run_modes_ab` has **no autocast at all**.
- `trainer.py:380` resolves `use_amp = config.use_amp and device.type == "cuda"`.

So the answer depends on the device, and not by a little:

| Where | `use_amp` | What a comparison establishes |
|---|---|---|
| **CPU** — the bounded tests | always **False**, whatever the config says | **bit-exact agreement.** Both paths are fp32 and every disagreement is a real one |
| **CUDA** — item 7a's institutional run | **True** by default, `float16` | the trainer's scores differ from Mode A's in the last bits by construction. Anything near a tie may reorder |

**A CUDA disagreement is therefore a measurement, not a fault**, and the harness
must not be built to refuse it — refusing would make AMP's effect on the ranking
the one thing the calibration cannot observe. `DifferentialResult` records the
**resolved** `amp_enabled` / `amp_dtype` beside the verdict and exposes
`bit_exact_contract`, which says which of the two questions the run answered.

**This promotes item 1e.** `amp_dtype` was listed as a manifest tidiness item; it
is the field that says which of two comparisons a recorded number came from — an
exact comparison at **equal** precision, or an exact comparison of discrete ranking
artifacts computed at **unequal** precision. Both are exact; they are not the same
question. Neither is a tolerance comparison, and an earlier revision of this
sentence called the second one that, which was wrong twice over: no tolerance
exists anywhere in the harness, and the phrase invited a reader to expect a
threshold that is deliberately absent. A calibration artifact without `amp_dtype`
cannot say which of the two it is.

**What the bounded tests consequently cannot do**, stated so item 7a does not
inherit it silently: they cannot ask the CUDA question at all. The container this
work is done in has no CUDA device. The CPU result is a genuine and necessary
precondition — the two implementations agree when precision is held equal — and
it is not a substitute for the institutional run.

**And a gap item 7a must close before it runs, found on audit rather than in
review.** On CUDA the trainer's defaults are `use_amp=True, amp_dtype=float16`, so
a differential run left at its defaults compares an autocast trainer pass against
an fp32 Mode A pass. **Two legs, and 7a must say which one its acceptance turns
on rather than inheriting a default:**

| `TrainerConfig` | What the run answers | Who decides the criterion |
|---|---|---|
| `use_amp=False` | do the two implementations agree exactly | **engineering** — `agreed is True`, the same hard parity gate as on CPU |
| `use_amp=True` (the CUDA default) | does autocast reorder anything, on this model, cohort and device | **the institution's experimenters** — see below |

**A first draft of this section said an AMP-on run reports `agreed=False` "by
construction". That was wrong and is corrected here.** Nothing in the harness
compares score bits. The comparison is over discrete ranking artifacts — the
top-`K` rows, the truths, the reciprocal ranks and the aggregate — so a precision
difference registers only where it actually *reorders* something. A run in which
fp16 reorders nothing agrees exactly. There is likewise **no numerical tolerance
anywhere in the harness**, so an AMP-on result must not be described as a
"tolerance observation", which an earlier revision also called it. It is an exact
comparison of discrete artifacts computed under unequal precision regimes.

**The AMP-on criterion is not engineering's to set, and this is a decision, not a
deferral.** Whether autocast changes a diagnosis ranking enough to matter is an
empirical question about a particular model, cohort and device, and the acceptable
answer is a clinical judgement. Neither can be predicted from this side. The
institution's position is that **the switch stays a switch**: the hospital's
experimenters configure it, run it both ways if they wish, and record what they
observe.

So engineering's deliverable is the **evidence, not the verdict**, and it is
already in place — `DifferentialResult` carries the resolved `amp_enabled` and
`amp_dtype`, `bit_exact_contract` for which question was asked,
`n_samples_disagreeing` and its rate, the affected rows with their sample ids and
both rankings, and `mrr_absolute_difference`. Those are the quantities a
disagreement criterion would be written in terms of. **No threshold is hardcoded
and none should be.**

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

### 3.4 No accepted holdout protocol, and nothing owned that

M9. The mechanical guards landed and work: `--split` is required with no default
on both entry points and `read_samples` lists what exists rather than
substituting. Those stop a number from being produced *silently* on the wrong
split.

**What is missing is a protocol, not a capability.** `--split` already accepts
`test` and `read_samples` would load a supplied `test_samples.json`. What no
document defines is which cohort that should be, who creates it, how it is
frozen, or what makes it independent. So the honest statement is not "the project
cannot measure held-out data" — it is that **nothing has decided what held-out
would mean here.**

**And the CLI help names only one of the two contamination kinds.** It states
checkpoint-selection contamination — `val` is what `early_stopping_monitor`
selects on — and says a test split exists only where an evaluation protocol
created one. It does **not** mention the measured train/val disease overlap; that
caveat is **item 2** and is blocked on item 10's M4 evidence. The figure must not
reach user-facing help ahead of its evidence.

Until M9 was written down this appeared only as a parenthetical in item 6 and one
line in `scorer-measurement/README.md`. **No item owned it**, which is how a
known problem becomes a forgotten one.

#### The unit of holdout has to be decided before any split is created

Item 11 is not "add `test_samples.json`". Three different things get called
held-out and they support different claims:

| Unit | What it can support | What it cannot |
|---|---|---|
| **Held-out sample views** over diseases already in training — a fresh post-training synthetic cohort | independence from *checkpoint selection*; robustness to phenotype dropout | **not** unseen-disease generalisation: it shares every disease with training |
| **Disease-disjoint** evaluation | unseen-disease generalisation | requires **retraining** — it cannot be produced for an existing checkpoint |
| **External clinical cohort** | independence of origin, and **patient-level external evaluation** on its own terms | **unseen-disease** claims, until its disease overlap with training is measured and reported. Overlap decides whether it *also* supports that claim — it does not decide whether the cohort supports any result at all |

**Item 11's first job is deciding which claim each phase needs**, then choosing
the unit that supports it. Choosing a split first and asking what it proves
afterwards is how a contaminated number acquires a clean-sounding name.

#### One decision, three consumers

| Waiting on it | How it appears there |
|---|---|
| item **8a** | B-0.5's protocol and output-contract design |
| `scorer-retraining/README.md` §5 | gate "a fixed evaluation protocol, cohort and split" — **not defined** |
| any future model comparison offered as generalisation evidence | — |

Decide it once. Deciding it three times is how they end up disagreeing. The
decision lives **here**; the other two carry a pointer, not a copy.

#### What item 11 does *not* block

It does **not** block trainer characterization (1b), the extraction (1c) or
differential-calibration *correctness* (1d). A contaminated `val` cohort remains
usable for **describing the deployed scorer**, which is what work item B-0 exists
to do — provided no held-out or generalisation claim is attached to the result.

### 3.5 What item 5a does and does not depend on

Recorded because "independent of the calibration line" was true of the
*implementation* and got read as true of the *acceptance*.

| | Depends on |
|---|---|
| **5a implementation** — wiring A into the loader and the primitive | nothing in B-0. Not calibration, not the split protocol, not the authoritative-checkpoint decision |
| **5a acceptance** — §13's integrated memory and reload gate | a **designated loadable checkpoint**, plus compatible graph and SP artifacts. A complete pipeline cold start with a resident model cannot be measured without a model |

**"Designated loadable" is not "authoritative".** 5a does not wait on item 6's
clinical decision; it needs *a* checkpoint that loads against the artifact set,
which is a far weaker requirement. If the finally deployed model has a materially
different architecture or memory footprint, compatibility is confirmed **for that
model** — that is a re-run of the gate, not a blocker on it now.

No checkpoint registry. One designated file, named in the 5a plan.

---

## 4. Ordered backlog

Ordered by **dependency**, not by size. An item may only start when everything it
depends on is resolved.

| # | Item | Depends on | Owner | Size |
|---|---|---|---|---|
| **1** | **The calibration decision** — §3.1.2, adopted and reviewed | — | **decided** | — |
| **1a** | Correct and **suspend** the legacy-removal checklist | 1 | author | **done** |
| **1a2** | **Malformed-truth invariant** (§2.3) — remove the silent disease clamp; enforce the range at the **loader** boundary on CPU, not in the trainer hot path where it would sync CUDA every batch. **Not a decision gate**: the contract is REFUSE and all three boundaries implement it | 1a | author | **done** |
| **1b** | Characterization tests freezing `Trainer._validate` / `Trainer.evaluate` observable behaviour — 32 tests. Shared-pass behaviour is driven through **both** entry points; caller-specific contracts stay separate. Each contract group mutation-checked against a representative defect (9 mutations). Found that the malformed-truth refusal has **two independent sources**, and which fires depends on the sign of the bad id — so the tests run both signs | 1a2 | author | **done** |
| **1c** | Extract the pass those two already duplicate — private, narrow. Behaviour-neutral: the 32 characterization tests pass **unchanged**, and each shared operation went 2 occurrences to 1 | 1b | author | **done** |
| **1d** | Same-batch differential calibration — `src/evaluation/differential.py`, 20 bounded tests. One materialised batch list to both paths; per-sample top-20, truth, reciprocal rank, then aggregate MRR. Five legs mutation-checked (scoring, pooling, truth, truncation, aggregate-only). Review round 1 upheld three findings — the aggregate did not gate the verdict, the supplied-`mode_a_result` seam let an acceptance gate accept evidence about itself, and the import contract was directional; two further sub-requests were declined with citations (§6). Found that **bit-exactness is a contract only when AMP is off** (§3.1.3), and that the shared synthetic cohort is too narrow to exercise the truncation at all — so the fixture gained size parameters and a second, wider cohort | 1c | author | **done** |
| **1e** | D2 manifest additions and the legal-truth equality test (§2.3) — **done**. `amp_dtype` + `torch_compile_wrapped` on the manifest and on `DifferentialResult`; `assert_no_autocast` turned the manifest's `amp_enabled=False` from a structural claim into an enforced one, in both traversals — **later superseded by Proposal B**, which records the observed regime rather than forbidding a non-default one. Legal-truth equality tested under a **non-identity** id map, which is the case 1d's identity-mapped cohorts could not distinguish. Found that `build_manifest` had `amp_enabled` **hardcoded**, and that the wrong-space mutation is refused by the loss rather than the harness | 1c, 1d | author | **done** |
| **P** | **Configurability and provenance** — `PLAN_CONFIGURABILITY_AND_PROVENANCE.md`, both proposals **done and approved**. **A**: `training_input_digests` on the checkpoint, a sibling of `data_fingerprint`, covering the semantic roles a run consumed including a resume parent that was actually loaded; `file_sha256` moved to `src/utils/fingerprint.py`. **B**: the AMP regime is **recorded at the computation that produced each mode's numbers** rather than forbidden — `EncodedGraph` carries the embeddings with their regime into the B/C traversals, and the invariant enforced is `encoded.regime == manifest.regime == scoring regime`, checked per batch. Capture only: no current-workspace comparison, no registry, no AMP CLI, no threshold | — | author | **done** |
| **2** | Update the contamination caveat to the measured 100% (§3.2), with both split file hashes | 10 — **satisfied**, `EVIDENCE_M4.json` | author | small, **unblocked** |
| **3** | `DISEASE_SCORER_POLICY.md` §3.5 correction (§3.3). The §3.5 inference — that most candidates fall outside the 5-hop table — is refuted under either reading of the old denominator: median 64.3%, q1 51.2%. Use the measured distribution, **not** the retired 71.3% | 10 — **satisfied**, `EVIDENCE_M5.json` | author | ~5 lines, **unblocked** |
| **4** | Reply to the sustained-with-narrowing contamination review | 2 | author | text only |
| **5** | **B-0.4 prototype phase** — both prototypes measured on the real artifact, twice; approach A selected for the primary GB10 platform | — **independent of 1 and of 10** | author | **measurement complete and reviewed** |
| **12** | **`evaluate([])` silently evaluates the val set** — `test_dataloader or self.val_dataloader` (`trainer.py:813`), so an explicit empty list is falsy and becomes a full validation pass. Frozen by a 1b test marked *observed, not endorsed*. Kept out of 1c deliberately, since an extraction commit must stay behaviour-neutral; raised here so the frozen defect is **not stranded** as a permanent monument to a known bug | 1c | author | small, unscheduled |
| **5a** | **B-0.4 productionisation** — wire A into `_load_shortest_paths` and `sp_mean_distances`, then `PLAN_B04.md` §13's gate. **Production code: needs its own plan and review before any edit.** Its *implementation* depends on no calibration, split or checkpoint decision; its **acceptance does need a designated loadable checkpoint** plus compatible graph and SP artifacts — see §3.5 | 5; acceptance also needs a loadable checkpoint | author + institution | not started |
| **11** | **Decide the evaluation-holdout protocol** (M9, §3.4) — **first** which claim each phase needs, then the unit that supports it: held-out sample views, disease-disjoint, or an external cohort. A **protocol decision**, not a code fix; the mechanical guards are already in and the tools already accept a supplied `test` split. Blocks 8a, `scorer-retraining` acceptance, and any held-out or generalisation claim. Does **not** block 1b/1c/1d | 2, 10 | needs review | design question |
| **6** | Which checkpoint is authoritative. Engineering supplies hashes, logs, artifact-compatibility evidence and load results; the **institution decides**. The question must separate the *deployed* checkpoint from the one `select_checkpoint_in_dir` picks by the highest **contaminated** `val_mrr` — `model-22` winning that metric makes it neither clinically authoritative nor a held-out-generalisation winner | 2, 10 — **satisfied**, `EVIDENCE_M1_M3_hgt.json` and `EVIDENCE_M1_M3_gat.json` | institution | question, **unblocked** |
| **7a** | Engineering differential calibration run | 1d, 10 — **satisfied**, and every scanned checkpoint carries a `data_fingerprint`; D5 artifact set; a designated loadable checkpoint | author | blocked on 1d and the checkpoint designation only |
| **7b** | Institutional measurement (B-0.2 / B-0.3) | 7a, 2, 3, 6, deployment CUDA verification | both | blocked |
| **8a** | B-0.5 protocol and output-contract **design**. **Consumes item 11's holdout decision and may not redefine it** | 1, **11** | author | **before** any expensive run |
| **8b** | B-0.5 institutional execution | 8a, 7b, 6, exact artifacts, production-path prerequisites | both | blocked |
| **9** | Mechanical rename (~70 refs, 9 files), then rewrite the checklist, then delete the oracle-only surface | **1d passed review incl. its institutional CUDA run** | author | behaviour-neutral |
| **10** | **Commit bounded evidence for M1-M5** — four JSON files and the three scripts that emit them. **Not raw console output** (§5.2). Unblocks 2, 3, 6 and 7a | — | institution + author | **done** — run on the deployment-sibling machine, evidence cited by digest in §2; M1-M4 confirmed, M5 corrected (§2.4) |

**Parked deliberately, not forgotten:** `task-scope/` Q2–Q5 (settled, unscheduled)
and `scorer-retraining/` (scoping only, four gates uncleared). Neither blocks nor
is blocked by anything above.

---

## 5. Two orderings that are easy to get wrong

**Item 5 did not wait for item 1, and that held.** B-0.4 measures the
shortest-path **lookup cost**: it consumes no checkpoint, no sample split and no
model — only `shortest_paths.pt`. It ran to completion and was reviewed while
item 1 was still open, which is the independence claim discharged rather than
merely asserted. Item **5a** inherits it: productionising A still needs no
checkpoint and no calibration.

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

**M1-M5 had no artifact.** They existed as text pasted into a review thread and
summarised here. That was the wrong way round: M1 and M2 are what established that
the calibration target does not exist — the largest decision this phase has made
— and M4 is what bounds every number the project will report. Those are precisely
the facts that most needed to be independently checkable, and they were the ones a
reviewer had to take on trust.

**Closed.** Four evidence files now sit beside this one, each emitted by a
committed script and cited by digest in §2. M1-M4 were confirmed against the
artifacts, every digit. M5 was not: the recorded figure combined a mean with a
denominator from a different artifact, which is exactly the class of error that
survives indefinitely in prose and does not survive one reproducible run. §2.4
records it.

### 5.2 What that evidence may and may not contain

**Not raw institutional console output.** These runs touch a clinical
deployment, and a console transcript carries filesystem paths, sample and patient
identifiers, and whatever else was on screen. Each item below is an **aggregate
JSON plus the script that emits it** — the script is what makes the number
reproducible, and the bounded schema is what keeps the artifact publishable.

| Fact | The file records | It must not record |
|---|---|---|
| M1-M3 | input digests, checkpoint count, key-presence summary, `in_channels` summary, and the filename-vs-`logs` metric comparison | checkpoint tensors, absolute paths, operator or host names |
| M4 | both split hashes, the two disease counts, and the size of their intersection | any patient id, sample id, or per-disease list |
| M5 | the SP artifact digest **and its sidecar's**, the disease denominator, the **configured** hop bound beside the observed one, the reachable count and percentage, **and the phenotype-selection rule** | per-phenotype rows |

**"A typical phenotype reaches 71.3%" was not reproducible as written**, and
putting it in JSON would not have made it so. The selection rule had to be
operational — which phenotype or phenotypes, chosen how, and whether 71.3% is one
phenotype's value, a median or a mean.

`scripts/audit_sp_reachability.py` states it by **removing the choice**: the
distribution is computed over every phenotype in the graph, and the report's
`selection_rule` says so. Two design points follow.

  - **Phenotypes reaching no disease are counted.** They have no rows in the
    artifact, so a count taken over the table drops exactly the zeroes and reports
    a distribution shifted upward.
  - **The hop bound is the configured one, not the largest distance present.** An
    artifact built to 5 hops whose longest path happens to be 4 would otherwise be
    reported as a 4-hop artifact, and every percentage in it read against a bound
    nobody chose. The configured value is read from the producer's
    `<artifact>.meta.json`; the observed maximum is recorded beside it.

**The run has happened, and it corrected this file's own prediction.** A previous
revision said here that the recorded 71.3% was probably an overestimate *because*
zero-reach phenotypes were being dropped. Measured: **270 of 19,836 phenotypes
(1.36%)** reach nothing — the mechanism is real, the direction was right, and the
magnitude is nowhere near enough to explain a fifteen-point gap. The cause is the
denominator, which came from a different artifact. §2.4 records what was measured
and what the recorded figure actually was.

**No evidence database, registry or index.** Three files beside the plans they
support, exactly as `EVIDENCE_B04_*.json` already sit beside `PLAN_B04.md`.

---

## 6. Contradictions found and resolved in this revision

Recorded because the request that produced this file was to stop them recurring.

| Contradiction | Resolution |
|---|---|
| `PLAN_B04` §5.3.2 required scanning "the deployed artifact" while §10.1 established there is no single one | §5.3.2 restated as evidence about the generator's invariant |
| **This file's own §3.1.2 was silent on precision** while specifying an exact per-sample comparison, and the two paths run at different precisions on CUDA | §3.1.3 added; the verdict now carries the resolved AMP state and `bit_exact_contract` rather than implying one answer |
| **1d's first draft claimed the truth comparison "exercises the `original_indices` translation"** | **False, and self-corrected before review.** Both sides translate through the *same* gather, so it cancels. What it actually tests is that the trainer's `diagnosis_targets` and Mode A's `disease_ids_local` are the same ids — which holds exactly because the gather is injective. The docstring now says that, and cites where `to_global_ids` *is* tested |
| **1d's first draft froze the legacy metric key at import** while `run_modes_ab` rebuilds it per call — and the comment on it claimed the opposite | Resolved per call via `legacy_mrr_key()`. The stale constant would have raised `KeyError` the moment anything moved the truncation, while the comment promised it could not |
| **The shared synthetic cohort is 4 candidates wide**, so a top-20 truncation and no truncation are the same list, and every existing test on it certifies a truncation it never performed | `build_workspace` gained size parameters (defaults unchanged, so no existing caller moved) and 1d added a second cohort wider than `LEGACY_TRUNCATION_K`. `assert_candidate_universe_is_stable` now takes the loader config, because stability is a property of the graph **and** the sampling limits together — and it immediately rejected the first wide config, whose hop-1 limit was below the disease count |
| `PLAN_B04` §9.4's estimated "+7 GB" stood beside §10.2's measured ~24 GB with no link | §9.4 marked as an estimate and pointed at the measurement |
| The `PLAN_B04` status header was spliced into a broken sentence by a scripted edit | Rebuilt as two named gates with their verdicts |
| `--split` defaulted to `test`, which the generator never writes | Made required on both entry points; the error names the splits that exist |
| This file's own first draft proposed calibrating **Mode C** against the trainer's `val_mrr`, on an unverified guess about the trainer's candidate universe | Verified before submission and **withdrawn**: the trainer scores the batch subgraph, so the number is Mode-A-shaped (M8). The direction table now records the withdrawal rather than deleting it |
| Its second draft then proposed calibrating **Mode A** against the *stored* `val_mrr` — right shape, but unreproducible and insufficient | Withdrawn on review. §3.1.1 gives both reasons; the stored value is demoted to a historical sanity reference |
| The legacy-removal checklist instructed deletion of the clamp, local ranking, top-20 semantics, the A/B traversal and two manifest fields — all of which the adopted calibration **keeps** | Checklist **corrected and suspended**, with the oracle-only surface separated out and the removal order fixed at eight gated steps |
| An earlier reply claimed an out-of-range disease id "would be read as if it were already local" | Imprecise. The mapping tensor is `-1`-initialised, so an in-range unsampled id maps to `-1`; an out-of-range id is left unchanged and is not normally a valid local column. §2.3 states the invariant instead of the guard |
| §2.2 credited `_assert_cohort_is_intact` with enforcing the local-truth range invariant | **Wrong citation, right conclusion.** `to_global_ids` (`measurement.py:91-100`) is what enforces it, and `tests/unit/test_measurement_ranking.py:52-56` already covers `[3]`, `[99]` and `[-1]`. Corrected in §2.3 |
| Review held that a `-1` local truth reaches PyTorch negative indexing and silently selects the last candidate, leaving `n_absent` at zero | **Refuted empirically.** `to_global_ids` raises `"local ids must be non-negative"` before any indexing, and `tests/unit/test_measurement_ranking.py:52-56` already covers it. The trainer-side asymmetry proposed in the same reply was itself wrong — see the row below |
| This file claimed the trainer silently scored a malformed truth `0.0` into `val_mrr`, making it an open contract decision (item 1f) | **Wrong, and wrong the same way I had criticised: a fragment traced without its path.** `self.loss_fn` runs before prediction collection (`trainer.py:640`) and `DiagnosisLoss` raises on both label-smoothing branches. Verified by execution. Item 1f is dissolved; the contract is REFUSE and both paths already implement it |
| The first fix for the malformed-truth invariant put the check in `Trainer._compute_model_outputs`, after `_move_to_device` | `bool(cuda_tensor.any())` synchronises host and device **every valid batch**. Moved to `DiagnosisDataLoader._assert_disease_truth_in_range` — CPU, at the boundary that creates the hole. A test now asserts no `bool()` is taken of that tensor in the hot path |
| The trainer test file claimed to characterize "the complete trainer path" | It called `DiagnosisLoss` directly and rested the rest on source ordering. Claim narrowed to loss-level refusal; full orchestration coverage stays with item 1b |
| `benchmark_sp_lookup` chose shape order with `(cell_index + position) % 2`, intending to decorrelate it from the rotated implementation order | With two implementations the rotation moves `position` with `cell_index`, so the sum was **constant per implementation identity** — `current` singleton-first 60/60, prototypes batched-first 60/60. Keyed on the cell alone; regression test now runs the two-implementation configs. Impact on the collected data bounded in `PLAN_B04.md` §12.6 |
| M9 said the project "cannot produce a held-out number at all" | Overclaimed. `--split` accepts `test` and `read_samples` would load it, so the **tools** are not the blocker. Narrowed to: the standard generated workspace has none, and no accepted protocol defines, creates, freezes or proves the independence of one |
| §3.4 said the `--split` help "names both contamination kinds" | **False, and checked.** The help names checkpoint-selection contamination only; `grep -c "overlap\|100%"` over both entry points returns 0. The disease-overlap caveat is item 2 and is blocked on item 10's evidence — the figure must not reach user-facing help ahead of it |
| Item 11 read as "add `test_samples.json`" | Reduced a protocol question to a file. It must first decide **which claim each phase needs**, then the unit: held-out sample views over seen diseases, disease-disjoint (needs retraining), or an external cohort (overlap must be measured). §3.4 tabulates what each can and cannot support |
| §3.4's prose said three places share one decision, but item 8a still depended only on item 1 | Prose without a dependency edge is a wish. 8a now depends on 11 and may not redefine it; `scorer-retraining`'s gate carries a **pointer**, not a copy of the text |
| "No transient memory threat" from A's index build | Broader than the evidence. Narrowed to what was measured: **in the isolated cold benchmark** A added no peak above artifact loading. Integrated steady-state and reload peaks stay open under `PLAN_B04.md` §13 |
| Item 5a read as fully independent of the calibration line | True of its **implementation**, false of its **acceptance**: §13 measures a cold start with a resident model, which needs a designated loadable checkpoint plus compatible artifacts. §3.5 separates the two. Not the authoritative-checkpoint decision, and no registry |
| The absence of a held-out test split appeared only as a parenthetical in item 6 and one line in a phase README | Promoted to **M9** and **item 11**. The `--split` guards stop silent misuse but do not create the split; three places were already waiting on the same undecided protocol (§3.4) |
| M1 was written as "no checkpoint this project has produced" | Overclaimed: no historical audit was run. Narrowed to the current producers and the scanned family, which is sufficient to reject the frozen evaluator as the acceptance oracle |
| Item 10 asked for "the raw scan and audit outputs" | Raw institutional console output carries paths and sample identifiers. Replaced by three bounded aggregate JSONs plus their scripts (§5.2) |
| **1d shipped claiming aggregate MRR as the fourth agreement check while `agreed` ignored it** — review finding, upheld | The aggregate now gates the verdict via `aggregate_mrr_agreed`. On the path the function takes, per-sample agreement implies aggregate agreement, so this is a derived property — but a contract enforced only by a docstring is the defect this file keeps recording, and it survives a change to how either aggregate is obtained |
| **1d shipped an optional `mode_a_result` seam** — an acceptance gate accepting caller-supplied evidence about the thing it gates. Review finding, upheld | **Seam removed rather than field-checked.** Checking patient ids and counts does not repair it: a result from a different negative draw over the *same* patients matches on both. The one legitimate use — one pass, two artifacts — is served better by `DifferentialResult.mode_a_result` carrying the run the function performed, which gives the same saving with nothing to trust |
| **The commit message claimed non-tautology is "enforced by the build"; it was enforced in one direction only** | Verified by experiment, not argument: with a probe `from src.training.trainer import Trainer` in `src/inference/scoring.py`, all three contracts still reported KEPT. Layers are directional — `src.inference` may import `src.training`. A fourth contract, `scorer-independence`, closes that direction and was mutation-checked against the same probe. **Review round 2 narrowed the claim further, and correctly**: no contract can stop both sides delegating to a helper *below* both, and they already share `F.normalize` and `torch.mm` — `masked_mean_pool`'s own docstring says it mirrors the trainer "operation for operation". What the build forbids is a direct cross-stack import in either direction; what the calibration detects is divergence between two maintained copies, not correctness of either. Mode A is a control that preserves the legacy behaviour including its defects, so both being wrong the same way is the design |

**Authority above everything here:** `docs/DISEASE_SCORER_POLICY.md`.
