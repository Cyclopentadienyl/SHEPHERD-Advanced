# Programme backlog — what is open, in what order, and why

**Purpose.** One ordered list across every live phase, so the next action is
readable without reconstructing six review threads. Phase documents keep their
own detail; this file holds **ordering, dependencies and blockers only** and must
not restate their decisions.

**Status:** sixth revision. The calibration decision at item 1 is **made and
reviewed** (§3.1.2); the legacy-removal checklist it invalidated is corrected and
suspended. (Both superseded on 2026-10-12: R2, implemented and code-reviewed with no P1 or P2 (`1d0b5a6..06fd0f1`, 2026-10-12), retired the
differential calibration with Modes A and B, and the scorer README carries the retirement's
record in place of the checklist.) B-0.4's measurement is **complete and reviewed**, and its
productionisation is a separate item (5a) behind its own gate. This revision adds
**M9 and item 11**: the standard generated workspace has no held-out split and no
protocol defines one — known in two passing mentions and owned by nothing. Item
11's scope is the **unit** of holdout, not a filename (§3.4).

**Next action (2026-10-14): item 15 — the probe's allocator follow-up.** M2's acceptance ran on the GB10 at `82d7573` and passed, demo and real data, 62 of 62 probes, as the reviewer found ([evidence](m2-acceptance-2026-10-14/README.md)): functional only, since the probe ran the trainer as a module and so under the native allocator, not the saved `cuda_async`; no capacity reading and no model-quality claim. The probe now applies the shared allocator rule before torch is imported and records the backend: its review, then a minimal backend check in a new process on the GB10 (no two-epoch rerun), and the homelab checkout restored to `main`. The acceptance was run through the deployment probe extended for it (`--m2-acceptance`, the owner's decision of 2026-10-13; the plan's "Acceptance for M2", "How it is run"), after its review's two P2 (the report could overwrite an input or earlier result; M3 measured the parent) were fixed. M2.2–M2.6 are implemented; the code review of `810d344..0aabfc3` found three P2 and no P1 (a checkpoint over an in-memory graph escaped M2.4, the resume parent's read was held through training, and the inventory reported a sidecar check as a servable table), fixed with the reviewer's recommendations to retire `SHEPHERD_SP_HOP_BOUND` and to refuse a checkpoint over an unverified graph, and the re-review of `0aabfc3..26fe84c` closed all three with no new P1 or P2 (the plan's §4, M2.2–M2.6, "As implemented"). M2 is not complete until the acceptance: fresh build → train → resume → serve with a retrained model, since no existing checkpoint records `kg`, the refusals of a wrong graph and wrong data leaving no run artifact, and the inventory run (M2.6). M2.1's technical work is complete: the owner accepted S10's capacity on its designated subject on 2026-10-13, with the remaining uncertainty explicitly accepted (the capacity README §6.2); not merged. **The record before that:** close M2.1 (S9, implemented and code-reviewed with no P1 or P2, `02a6954..f481882`, 2026-10-12; then S10 on its designated subject, run on 2026-10-12 with its re-reviewed tools at `90a668f` and the "after" commit `343e90e`, its evidence and explanation awaiting review, and the reviewer's recommended limited diagnostic of consecutive reloads chosen by the owner and run on 2026-10-13: each process grew by about 0.4–0.5 GB at every reload at both commits, before M2.1 as after it, the difference between the commits ambiguous, so acceptance 2 stays open; its second phase found every watched object of the earlier pipelines freed, and the owner ended the reload-memory investigation on 2026-10-13: the PyTorch wheel in use is confirmed to contain mimalloc purge defects that upstream has since fixed; they are a strong candidate cause of the observed memory retention; the attribution of all of the increase is not established (item 22); acceptance 2, with M2.1's ~30 MB difference recorded apart, is the owner's separate decision, closed on 2026-10-13 as above), then
M2.2–M2.6**, so that the contract is enforced with producer and consumer together, with a formal
way to produce acceptable new models and the deployment inventory (the reviewer's convergence
instruction of 2026-10-12, adopted by the owner and recorded in the plan's §4). Items 20, 21, 14
and M3 stay independent of it. Item 11, the evaluation-holdout protocol decision, remains open
and blocked on neither institutional evidence nor a designated checkpoint, but it is no longer
the only such item. Items 1 through 1e are complete. Since the owner's clarification of
2026-10-12, R2, the Mode A/B retirement change that replaces item 9, is not blocked on
institutional evidence either (item 9's row); it was implemented on 2026-10-12 and code-reviewed with no P1 or P2 the same day
(`1d0b5a6..06fd0f1`). Two directions this file
proposed were withdrawn along
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

**The A and B rungs are retired; the goal is not.** The owner reported on 2026-10-12
that the institution had already authorised retiring the old pipeline's Mode A/B comparison
tools (§3.1; item 9's row), and R2 deleted them the same day. Policy Gate 1 stays, and its B-0 report is kept: it compares the
currently served ranking (Mode D, B-0.5's served behaviour) with the planned new GNN ranking
(Mode C; item 7b).

---

## 2. Established facts

M1-M7 were **measured on deployment hardware**; M8 and M9 were **read from
source**. Both kinds are checkable; neither is recalled or inferred. These are
what changed the ordering.

| # | Fact | Where |
|---|---|---|
| M1 | **No scanned checkpoint carries `metadata` or `in_channels_dict`.** All carry `config`, `data_fingerprint`, `epoch`, `logs`, `state_dict`. The claim is about the **current producers and the scanned family**, not about every checkpoint the project has ever written — no historical audit was run, and none is needed to reject the frozen evaluator as the acceptance oracle. **Confirmed by artifact** over 15 checkpoints in two families, `hgt` (10) and `gat` (5): both watched keys absent in all 15, the four expected keys present in all 15, nothing unreadable | [`EVIDENCE_M1_M3_hgt.json`](EVIDENCE_M1_M3_hgt.json) `39176ea3…`, [`EVIDENCE_M1_M3_gat.json`](EVIDENCE_M1_M3_gat.json) `5473fedb…` |
| M2 | **`in_channels` is 128** in every checkpoint; the frozen evaluator's hardcoded fallback was 256 (`scripts/evaluate_model.py` at `7dab728`, deleted 2026-10-10), which was the size mismatch it died on. **Confirmed by artifact**: read from `feature_encoder.projections.<node_type>.weight` rather than from a config field, 45 projections across 15 checkpoints, every one 128, none without a readable width | same |
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
| Padded phenotype ids | `phenotype_ids.clamp(min=0, max=…)` (`trainer.py:739`) | the same clamp, kept as trainer parity (`measurement.py:738`) |
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
that exist. Only the rows below are open. (Since R2, 2026-10-12, the manifest no longer
carries `shuffle`, `num_workers`, the sampler fields or the three seeds, which Mode C does
not use; `batch_size` stays.)

| # | Open item |
|---|---|
| D1 | **Record the limitation, not a field.** Historical epoch RNG state was never saved, so the historical stochastic validation traversal is not exactly reproducible. Sample order in a *new* run is pinned by `shuffle=False` plus the samples digest. Since R2 (2026-10-12) measurement is Mode C alone, which takes patients in the samples file's order with no loader, so the samples digest pins it and no `shuffle` is recorded |
| D2 | **Closed by 1e.** `amp_dtype` and `torch_compile_wrapped` are on `MeasurementManifest`, and the latter was on `DifferentialResult` too — that was the artifact whose regime actually varied, until R2 retired the differential (2026-10-12). **The field is named for what its probe can see**: `isinstance` against the dynamo wrapper proves the object is wrapped, not that a compiled graph ran, since graph breaks and eager fallback leave a wrapped model running eagerly. Renamed from `torch_compiled` in review round 1 for the reason `cuda_executed` was renamed from `calibration_eligible`. Two corrections to this row's own wording: `amp_enabled` was not "recorded", it was **hardcoded `False`** in `build_manifest`; and a structural claim is not an observation, so 1e made `assert_no_autocast` **refuse** to run either traversal inside an autocast block rather than let the manifest describe a run that did not happen. **Superseded by Proposal B**, which removed that ground instead of living with it: the fields now record the regime observed at the computation that produced each mode's numbers, and `assert_manifest_describes_regime` refuses only the case a single field cannot describe — numbers produced under one regime and recorded under another. `observe_torch_compile_wrapper` is one tri-state function, not compile-metadata machinery. **`isinstance` against the imported wrapper class is the only evidence**, aggregated over two import paths (`torch._dynamo`, `torch._dynamo.eval_frame`) so a release that moves one still answers exactly. No class resolves -> `None`. **There is no attribute fallback, and three review rounds are why.** R1: the first version turned "not observable" into `False`, never reached its own documented fallback, and claimed "never raises" while `hasattr` propagates any `non-AttributeError`. R2: the repair over-corrected — `_orig_mod` alone became `True`, letting **any** object opt in by naming a field, and the test written for it froze that false positive as a specification. R3: pairing the attribute with a `torch._dynamo` module prefix did not rescue it either — `startswith` also matches `torch._dynamoevil`, and `__module__` is **assignable class metadata**, so it records what a class says about itself rather than how it was built. `_orig_mod` is a necessary marker, never sufficient evidence, and no arrangement of weak signals makes it so. Every mechanism confirmed by execution, and each superseded version fails cases the next one added |
| D3 | **Closed.** `_remap_indices` (`data_loader.py:956-964`) maps `batch["disease_ids"]` into subgraph-local space, the same space as the score-matrix columns, and trainer and Mode A read the same remapped tensor. The range invariant is enforced at **three independent boundaries** — loader, loss, harness — none of them in the CUDA hot path. See §2.3. **The legal-truth equality test is done (1e)**, and it is not the one 1d already had: both 1d cohorts are identity-mapped (`original_indices["disease"] == [0..n-1]`), under which a side reading the wrong id space produces the right numbers anyway. 1e adds a hand-built batch with a non-identity, non-sorted map so local != global. A trainer emitting global ids is refused there by **`DiagnosisLoss`**, not by the harness — the boundary §2.3 measured, confirmed by execution after the test's first draft predicted the wrong one. **The harness boundary moved when Mode A went:** R2 (item 9's row, 2026-10-12) re-pointed the third boundary to Mode C's `_assert_ids_in_range` (retirement assessment §4; `src/kg/data_loader.py`'s docstring and the two tests that name the boundaries). D3 stays closed and the contract is unchanged |
| D4 | No excluded-sample list is required **while the run is fail-fast**. Record that the run does not silently skip samples and that ranked plus ground-truth-absent account for the intact cohort. `n_ground_truth_absent` is a count, **not** an excluded- or failed-sample list; if skip-and-continue is ever introduced, skipped ids and reasons become required then |
| D5 | Exact artifact identity. Measurement already records the digests of the checkpoint, samples, node features, edge indices and num_nodes, since contract M2.1's S7 from the bytes it parsed (`measurement_input_digests`; A/B/C hashed them by path, `measure_scorer.py:79-92` at the time). Open: the **shortest-path artifact digest, added when Mode D consumes it**, and comparing recorded digests against the institution-approved artifact set at acceptance. No registry, no compatibility database |
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

**Mode A enforced it, and the check was tested** (superseded: R2 retired Mode A and
`to_global_ids` on 2026-10-12; measurement's boundary is now Mode C's `_assert_ids_in_range`, in
global ids, in the table below). `to_global_ids`
raises on both failures *before* any indexing (`measurement.py:91-100`):

```
local id 4 is outside the subgraph's 4 disease nodes   # >= n_rows
local ids must be non-negative                          # the -1 hole
```

`tests/unit/test_measurement_ranking.py:52-56` parametrised exactly `[3]`, `[99]`
and `[-1]` (deleted with `to_global_ids` in R2). A `-1` never reached PyTorch's negative
indexing on this path, so it
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
| `_assert_ids_in_range` | `src/evaluation/measurement.py`, Mode C | measurement's own boundary: a ground truth outside the graph's disease nodes is refused before ranking, in global ids, since Mode C has no subgraph (`tests/unit/test_measurement_mode_c.py`, `test_an_out_of_range_ground_truth_is_refused`). Until R2 (2026-10-12) this boundary was Mode A's `to_global_ids` (`measurement.py:91-100`), which checked local ids |

No cross-module validator: three checks of a few lines each keep the coupling
lower than one shared one would. A regression test asserts that **no `bool()` is
taken of the disease-id tensor in the hot path**, so the CUDA sync cannot be
reinstated by a later well-meaning edit.

The **phenotype** clamp at `trainer.py:739` is a different thing and stays:
`diagnosis_collate_fn` pads phenotype ids with `-1` by design and the mask
discards those positions. Disease truths are never padded.

Still to cover, and only reachable once 1c exists: **equality of the legal truth
id consumed by the trainer reference and by Mode A** on the same batch. (Done in 1e, D3's row; moot
since R2 retired Mode A, 2026-10-12, code-reviewed with no P1 or P2 in `1d0b5a6..06fd0f1`.)

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

**The run's resource cost is recorded separately and more weakly.**
[`EVIDENCE_ITEM10_run_metrics.json`](EVIDENCE_ITEM10_run_metrics.json)
`305fb1a2…` holds the wall time, peak RSS, major page faults and bytes read
for the four runs. It is **transcribed by hand** from `/usr/bin/time -v` rather
than emitted by a committed script, so it is a weaker class of artifact than the
four beside it and says so in its own text. The console transcripts it was taken
from are not committed: §5.2 forbids absolute paths in evidence, and both the
`.out` and `.time` files carry them.

Two things in it are worth more than operational curiosity. The shortest-path scan
read **10.78 GB** from the filesystem against an artifact of **10.76 GB** — that
the whole table was read is established by disk traffic, independently of the
sidecar's `num_pairs` check, which concludes the same thing from tensor length.
And it took **21,846 major page faults** against 0-8 on the three runs that read
no large artifact, which is what a mapped read produces and a resident read does
not.


### 2.5 A defect that only the deployment hardware could show

Found while running the full suite on the institutional machine after item 10's
evidence run — not by looking for it.

`tests/unit/test_retrieval.py` reported **4 skips** there against **3** on a CPU
container. The extra one is `TestFactoryFunctions::test_create_index_auto`, and
the direction is backwards from the usual: a test that *passes* without CUDA
*skips* with it.

**The mechanism — and the seam this file first named wrongly.** The visible
symptom is in `resolve_backend`: it resolves to `cuvs` when `is_gpu_available()`
is true, and the block beneath it, commented *"Verify it can actually be
instantiated"*, validates only whether the platform is Windows. That description
is accurate and it is not where the defect was.

`_register_backends` guarded on whether a backend's **module** imports. Neither
backend imports its dependency at module level — `CuVSIndex` imports
`cuvs.neighbors` and `cupy` inside `_validate_cuvs`, `VoyagerIndex` imports
`voyager` inside a method — so that guard never fired for **either** backend and
every backend was registered on every host. `resolve_backend` then behaved
correctly given a registry that was wrong: `"cuvs"` was in it, so it was returned,
and the `["cuvs", "voyager"]` fallback beneath never ran because nothing raised.

Registration was the seam because selection, fallback *and* the operator report
all read the same registry: `scripts/build_index.py` printed
`Available backends: ['voyager', 'cuvs']` on a host with neither cuvs nor cupy.

**The test converted that into green.** It caught `ImportError` and skipped with
*"No vector backend available"* — false on that machine: voyager was installed,
and `test_create_index_voyager` passed in the same run.

**What this is and is not.** Not on the clinical path: the vector index was
detached from diagnosis by institutional decision and the
`diagnosis-retrieval-detachment` contract still enforces that. It is on the path
of the planned natural-language / vector-mapping work, and it was invisible on
every machine except one with CUDA and without cuVS — which describes the
deployment target and no CPU development box.

**It also contradicted a completed stage.** Stage 5 was *honest cuVS reporting and
orphan removal*. Its result lives in `scripts/validate_installation.py`, whose
comment forbids claiming availability on the strength of `import cuvs` alone —
*"a combination observed on a real deployment machine"*. That honesty was applied
there and not here, so two files answered the same question differently on the
same host. `_is_cuvs_available` was left behind by the same stage with zero
callers.

#### What was fixed, and what was deliberately left

Registration now asks whether each backend's **immediate required packages are
discoverable** — `voyager` for Voyager, `cuvs` **and** `cupy` for cuVS, since
`_validate_cuvs` imports both and cuvs-without-cupy is the state already observed
on a deployment machine. cuVS stays unregistered on Windows whatever is
discoverable; Voyager is gated on every platform. The registry is rebuilt rather
than accumulated, and cleared in place rather than rebound so a caller holding it
is not left reading a detached copy. Discovery fails closed: `find_spec` raises
for ordinary conditions, and an optional-dependency probe must never be the
reason `import src.retrieval` fails. `_is_cuvs_available` is deleted.
`resolve_backend` is unchanged.

**Discovery is the ceiling of the claim, on purpose.** It says the package is
findable. It does not say the package imports, that a CUDA build matches the
driver, that an index can be constructed, or that a search returns anything.
`validate_installation.py` remains the place where the deeper states are
distinguished, and this does not duplicate it into the import path.

Three limitations, recorded rather than discovered later:

  - **Discoverable is weaker than importable.** A package that is findable but
    raises on import still registers, and still fails at construction.
  - **Availability is inferred by the registry, not answered by the backend.** The
    structural fix is a `VectorIndexBase.is_available()` classmethod so each
    backend states its own contract. That is a design change to a detached
    subsystem and belongs with the natural-language retrieval integration, when
    its real support contract is known — not here.
  - **The probe logic still lives in two places**, here and in
    `validate_installation.py`. Consolidating them is worth doing and is not this
    fix.

## 3. What these facts broke

### 3.1 The calibration target did not exist — the root blocker, now decided

Two loaders fail on the same missing key, in two different ways:

- `build_legacy_mode_a_model` indexes `checkpoint["metadata"]` and
  `checkpoint["in_channels_dict"]` directly (`measure_scorer.py:189-190`).
  Neither exists (M1) — **`KeyError`**.
- `create_model_from_checkpoint` uses `.get()` with defaults instead
  (`evaluate_model.py:138-148` at `7dab728`): three node types, two edge types,
  256 channels. It therefore builds a **structurally wrong model** and dies later
  in `load_state_dict` against a real 128 (M2).

**Mode A cannot execute on any checkpoint in the scanned family**, and neither
could the frozen evaluator.

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
| Accept that no oracle number exists and drop Mode A | **Rejected** at the time: it discards the control the whole ladder is built around. **Superseded by the owner's clarification of 2026-10-12:** the owner reported that the institution had already authorised retiring the old pipeline's Mode A/B tools, so Mode A was retired with Mode B by R2 (item 9's row; 2026-10-12). The ground is current use, policy and the live tests R1 already moved, not a claim that Mode A never had scientific use. Policy Gate 1 stays (item 7b) |
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

**Superseded in part by the owner's clarification of 2026-10-12.** The owner reported on
2026-10-12 that the institution had already authorised retiring the old pipeline's Mode A/B
comparison tools, and that the old pipeline may be modified or rewritten entirely. Mode A as
the bottom rung, and the differential calibration that checks it, were therefore retired
by R2 (item 9's row; 2026-10-12) rather than kept, and 7a is cancelled. The basis is
current use, policy and the live tests R1 already moved to their use points, not a claim that
Mode A never had scientific use. The direction above stays as the record of what was decided
and why.

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

**Superseded by the owner's clarification of 2026-10-12.** The owner reported on 2026-10-12
that what the owner and the institution want to know about AMP is whether an AMP-*trained*
model is worse — the accuracy or quality lost against the large training-time saving some GNN
architectures get — which is a training study. It is not the same-checkpoint scoring comparison
with autocast switched that this section and 7a's AMP leg describe. That scoring evidence is
therefore no longer a precondition for deleting the differential or Modes A/B (retirement
assessment §8 item 7, closed), and 7a, its AMP leg included, is cancelled. The AMP training
study is recorded separately as bounded research (item 21), not as an M2 or R2 prerequisite.
This section stays as the record of what the differential established.

### 3.2 Validation measures no unseen-disease generalisation at all

M4 is the maximal case: **every** disease in val appears in train. The escalation
already sustained by review said the generator "permits" overlap; the measurement
says it is total.

This does not invalidate any engineering result. It **bounds what every number
may claim**, and the bound is tighter than the caveat currently drafted.

**Fixed under 11i, and `EVIDENCE_M4.json` is now a baseline rather than a
description of the present.** Generation consumes a disease allocation, so the
partitions are disjoint by construction. The M4 figure describes the regime that
produced it and is kept unchanged for that reason; the audit that produced it
gained a `--require-disjoint` gate and a cross-check of the sample files against
`split_manifest.json`, which is the one claim the generating process cannot make
about itself. Two consequences are **not** closed by the fix and are recorded
rather than assumed away: every checkpoint in the repository was trained under
the old regime, and a disease-level cut does not close the channel where one
phenotype set carries two different labels across it — `scripts/audit_generator_fidelity.py`
measures that.

### 3.3 A policy inference is contradicted — in its premise, not its conclusion

`DISEASE_SCORER_POLICY.md` §3.5 recorded, explicitly as unmeasured, that most
candidates fall outside the 5-hop table so the SP term degenerates to a
reachability indicator.

**The premise is refuted, per phenotype.** The median phenotype reaches 64.3% of
diseases within the configured 5 hops, q1 51.2%, max 78.16%; 270 of 19,836
phenotypes (1.36%) reach none at all.

**1.36% sizes phenotypes that reach nothing — not unreachable candidates.** Using
it for the second would understate them by orders of magnitude: the median
phenotype still leaves ~10,650 diseases unreachable and the best-connected leaves
~6,522. Unreachable pairs are common at every phenotype, and M5 does not size them
as a fraction of pairs.

**M5 does not reach the patient level at all.** The deployed scorer takes no union
over a patient's phenotypes and does not use the nearest reachable one: for each
candidate it iterates over every phenotype, gives each unreachable pair
`unreachable_distance`, and averages (`src/inference/scoring.py::sp_mean_distances`).
A per-phenotype distribution therefore cannot be carried across to what a patient
sees, in either direction.

**The conclusion is not refuted with the premise, and saying so would be the same
error in the other direction.** A candidate whose *mean* distance is 5 scores `1/6`
against a floor of `1/7` — one phenotype's five-hop path does not produce that
score by itself — so near-degeneracy could still arise from the distribution of
mean distances. Neither the reachable-distance distribution nor the patient-level
mean-distance distribution is measured, and no current artifact carries either.

The correction is factual and touches no normative statement. It also lands in
`SP_SCORE_GUIDE.md`, which restated the same claim for operators in two places —
one of them saying in as many words that the size of the group "has not been
measured on this deployment".

### 3.4 No accepted holdout protocol, and nothing owned that

M9. The mechanical guards landed and work: `--split` is required with no default
on the measurement entry point and `read_samples` lists what exists rather than
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
| **1a2** | **Malformed-truth invariant** (§2.3) — remove the silent disease clamp; enforce the range at the **loader** boundary on CPU, not in the trainer hot path where it would sync CUDA every batch. **Not a decision gate**: the contract is REFUSE and all three boundaries implement it (the third, measurement's, is Mode C's `_assert_ids_in_range` since R2; §2.3) | 1a | author | **done** |
| **1b** | Characterization tests freezing `Trainer._validate` / `Trainer.evaluate` observable behaviour — 32 tests. Shared-pass behaviour is driven through **both** entry points; caller-specific contracts stay separate. Each contract group mutation-checked against a representative defect (9 mutations). Found that the malformed-truth refusal has **two independent sources**, and which fires depends on the sign of the bad id — so the tests run both signs | 1a2 | author | **done** |
| **1c** | Extract the pass those two already duplicate — private, narrow. Behaviour-neutral: the 32 characterization tests pass **unchanged**, and each shared operation went 2 occurrences to 1 | 1b | author | **done** |
| **1d** | Same-batch differential calibration — `src/evaluation/differential.py`, 20 bounded tests. One materialised batch list to both paths; per-sample top-20, truth, reciprocal rank, then aggregate MRR. Five legs mutation-checked (scoring, pooling, truth, truncation, aggregate-only). Review round 1 upheld three findings — the aggregate did not gate the verdict, the supplied-`mode_a_result` seam let an acceptance gate accept evidence about itself, and the import contract was directional; two further sub-requests were declined with citations (§6). Found that **bit-exactness is a contract only when AMP is off** (§3.1.3), and that the shared synthetic cohort is too narrow to exercise the truncation at all — so the fixture gained size parameters and a second, wider cohort | 1c | author | **done**; superseded by the owner's clarification of 2026-10-12 — the differential it built was retired with Modes A/B by R2 (item 9's row; 2026-10-12), its live guarantees having moved to direct tests in R1. Kept as history |
| **1e** | D2 manifest additions and the legal-truth equality test (§2.3) — **done**. `amp_dtype` + `torch_compile_wrapped` on the manifest and on `DifferentialResult`; `assert_no_autocast` turned the manifest's `amp_enabled=False` from a structural claim into an enforced one, in both traversals — **later superseded by Proposal B**, which records the observed regime rather than forbidding a non-default one. Legal-truth equality tested under a **non-identity** id map, which is the case 1d's identity-mapped cohorts could not distinguish. Found that `build_manifest` had `amp_enabled` **hardcoded**, and that the wrong-space mutation is refused by the loss rather than the harness | 1c, 1d | author | **done** |
| **P** | **Configurability and provenance** — `PLAN_CONFIGURABILITY_AND_PROVENANCE.md`, both proposals **done and approved**. **A**: `training_input_digests` on the checkpoint, a sibling of `data_fingerprint`, covering the semantic roles a run consumed including a resume parent that was actually loaded; `file_sha256` moved to `src/utils/fingerprint.py`. **B**: the AMP regime is **recorded at the computation that produced each mode's numbers** rather than forbidden — `EncodedGraph` carries the embeddings with their regime into the B/C traversals, and the invariant enforced is `encoded.regime == manifest.regime == scoring regime`, checked per batch. Capture only: no current-workspace comparison, no registry, no AMP CLI, no threshold | — | author | **done** |
| **2** | Contamination caveat updated to the measured 100% (§3.2). Both measurement entry points' `--split` help now names **two** independent reasons `val` is not held-out — checkpoint selection, and that `sample_generator.py` slices one shuffled pool and never partitions by disease, so the overlap is structural rather than incidental to one workspace. The figure cites `EVIDENCE_M4.json`, which carries both split digests, rather than inlining hashes that are workspace-specific into help shown for any workspace | 10 — satisfied | author | **done** |
| **3** | `DISEASE_SCORER_POLICY.md` §3.5 corrected (§3.3), and the same claim where it was restated in `SP_SCORE_GUIDE.md`. **The premise is refuted per phenotype; the conclusion is not, and neither is anything at the patient level** — see §3.3 | 10 — satisfied | author | **done** |
| **4** | Reply to the sustained-with-narrowing contamination review | 2 | author | text only |
| **5** | **B-0.4 prototype phase** — both prototypes measured on the real artifact, twice; approach A selected for the primary GB10 platform | — **independent of 1 and of 10** | author | **measurement complete and reviewed** |
| **12** | **`evaluate([])` silently evaluated the val set** — `test_dataloader or self.val_dataloader` (`trainer.py:869`), so an explicit empty list was falsy and became a full validation pass: a caller asking for one cohort received numbers about a different set of patients. Selection is now `is None`. The other half was equally silent — `mean_loss` divides by `max(num_batches, 1)`, so an empty traversal returns `{"loss": 0.0}`, a statement about a model made without reading a patient — so an empty cohort is **refused**, matching `RankingMetrics`, `measurement.py` and `differential.py`, which already decline to invent a value for one. Checked after the traversal, so an empty generator is caught as surely as an empty list. The 1b test that froze this as *observed, not endorsed* said the fix had to be made by editing it rather than by an extraction nobody re-read; it was | 1c | author | **done** |
| **13** | **Backend registration did not check dependencies**, so `auto` could select an unusable backend, the fallback chain beneath it could never run, and `build_index.py` printed backends the host did not have. Affected **both** backends: neither imports its dependency at module level, so the module-import guard never fired. Not on the diagnosis path — retrieval is detached by decision and the contract holds. See §2.5 | — | author | **done** — registration gated on discoverable dependencies; three limitations recorded, not closed |
| **5a** | **B-0.4 productionisation** — A becomes **the** shortest-path primitive rather than one of two: one builder over raw columns, one sort, the scan retired from the served path and kept only as an independent test reference. The plan it needs exists and is **revised and cleared to implement**; the blanket "no `src/inference/` edit" gate it carried is removed and the reason is recorded in its status block. Its *implementation* depends on no calibration, split or checkpoint decision; its **acceptance does need a designated loadable checkpoint** plus compatible graph and SP artifacts — see §3.5 | 5; acceptance also needs a loadable checkpoint | author + institution | **implemented, not accepted** — §10 steps 0-7 done and reviewed; step 8: the supplementary SP-only figure is recorded at deployment scale (plan §6.2.1 — 9.00 bytes/row, 3.87 GB, on the real 429,971,678-row artifact, confirming §6.2's extrapolation; it yields **no serving peak**, because the benchmark does work before and around the build that the service does not), the measurement subject is **designated** (plan §7.1.1: an HGT checkpoint trained on the homelab against the artifact set, checked to load with the SP table bound, evidence in `scorer-measurement/subject-hgt-2026-09-29/`), and the procedure for readings 1, 2, 3b and 4 is reviewed (§7.3). **Readings 1, 2, 3b and 4 are taken and complete by §7.3's rule** (plan §7.4; three repeats on the designated subject by `scripts/measure_served_pipeline.py`, evidence `scorer-measurement/readings-hgt-2026-09-30/served_readings.json`, `5d509ba4…`); taking them is not passing a capacity gate, and §13 reading 5 is still blocked for want of the smallest supported target |
| **5b** | **SP artifact publication and binding** — the two defects `PLAN_B04_PRODUCTIONISATION.md` §8 left open: a tensor and a sidecar written with nothing recording that they belong together, and `--kg-path` / `--output-dir` independent so an artifact can land in a workspace it was not computed from. The consumer can now refuse a malformed table; neither check asks whether this is the *right* table. Plan: [`PLAN_SP_ARTIFACT_INTEGRITY.md`](PLAN_SP_ARTIFACT_INTEGRITY.md). **Needs no checkpoint and no institutional input**, which is why it runs ahead of 5a's remaining readings rather than behind them | — | author | **implemented** — producer, shared reader and all three consumers in one change; acceptance conditions tested and mutation-checked |
| **11** | **Decide the evaluation-holdout protocol** (M9, §3.4). **Institutional input received and recorded** in [`EVALUATION_COHORTS.md`](EVALUATION_COHORTS.md): three cohorts with three different jobs, MyGene2 for research comparison and the institutional offline cohort as the acceptance benchmark. §5's two gating answers — no direct synthetic unseen-disease result for the deployed weights, and option B with no refit — are **confirmed by the institution** (reported 2026-10-02). What remains open is the rest of that document's §5 — how many diseases to reserve and from which strata, whether to apply for the larger in-hospital database, and whether the generator is faithful enough. A **protocol decision**, not a code fix. Blocks 8a, `scorer-retraining` acceptance, and any held-out or generalisation claim. Does **not** block 1b/1c/1d | 2, 10 | needs review | design question, **partly answered** |
| **6** | Which checkpoint is authoritative. Engineering supplies hashes, logs, artifact-compatibility evidence and load results; the **institution decides**. The question must separate the *deployed* checkpoint from the one `select_checkpoint_in_dir` picks by the highest **contaminated** `val_mrr` — `model-22` winning that metric makes it neither clinically authoritative nor a held-out-generalisation winner | 2, 10 — **satisfied**, `EVIDENCE_M1_M3_hgt.json` and `EVIDENCE_M1_M3_gat.json` | institution | question, **unblocked** |
| **7a** | Engineering differential calibration run. **Its entry point will be new code.** By the owner's rule of 2026-10-09, contract M2.1's S0 (accepted at plan level in the re-review of `0708789` and, as revised, in the incremental review of `7e1af96`; extended since by the owner's decision of 2026-10-10 to remove `scripts/evaluate_model.py` and its pin test, contract M2.1 decision 6, accepted at plan level in the review of `45d9dea`; implemented 2026-10-10 and code-reviewed with no P1 or P2 at `3433bf8`) removed the superseded launcher `scripts/calibrate_mode_a.py` instead of rewriting it. 7a's runner is built on `compare_trainer_against_mode_a` (`src/evaluation/differential.py`) and records, beside its result, the digests of the bytes it read (through contract M2.1's readers), the software revision and the deployment host | 1d, 10 — **satisfied**, and every scanned checkpoint carries a `data_fingerprint`; D5 artifact set; a designated loadable checkpoint | author | **cancelled — superseded by the owner's clarification of 2026-10-12.** The owner reported that the institution had already authorised retiring the old pipeline's Mode A/B comparison tools, so the differential was retired with them (R2, item 9's row; 2026-10-12) and no run certifies it; and that the institution's AMP question is a training study (item 21), not this run's AMP leg (§3.1.3). Its runner was never written. Kept as history; was: blocked on 1d and the checkpoint designation only |
| **7b** | **Institutional measurement for Policy Gate 1's B-0 report** — rewritten by the owner's clarification of 2026-10-12, without Modes A/B and without 7a. Gate 1 stays and its committed comparison is kept: the currently served ranking (Mode D, B-0.5's served behaviour) against the planned new GNN ranking (Mode C). The served side is B-0.5's Mode D measurement (8b; retirement assessment §8 item 4). Retiring the A/B tools does not cancel this evaluation duty. Its other preconditions carry over: item 6, deployment CUDA verification, and Mode C's peak memory on institutional CUDA. Was: Institutional measurement (B-0.2 / B-0.3), depending on 7a, 2, 3, 6 and deployment CUDA verification | 8b, 6, deployment CUDA verification; 2, 3 — satisfied | both | blocked |
| **11i** | **Item 11's implementation arm** — the ordered work in [`EVALUATION_COHORTS.md`](EVALUATION_COHORTS.md) §6.6, tracked here so "where are we" is answerable from one table. **1** correct the document — done, revision 20. **2** split feasibility audit (`scripts/audit_split_feasibility.py`) — **done**, evidence committed as `EVIDENCE_split_feasibility_homelab.json` (schema 2, identical-sibling, KG `6889ed11…`). Worst validation-representation risk across four stratifications is a 69-disease band at 2.9% (f = 0.05) and 0.0013% (f = 0.15), so **stratified allocation is not built** and `disease_allocation.py` no longer calls that reading provisional. The figures belong to one MONDO vintage; a run on the institute's graph is a second vintage, not a second machine. **3** partition before generation (`src/kg/disease_allocation.py`, and `scripts/audit_split_overlap.py` reframed from a measurement into a gate) — done. **4** characterise the generator (`scripts/audit_generator_fidelity.py`) — done for the half needing no external tool; the upstream-simulator comparison is **deferred with its reason recorded** (§6.6). **5** evaluation records beside the checkpoint (`src/evaluation/sidecar.py`) — done. **6** UI — **out of this project's scope**, so the arm is complete at step 5. **Old paths removed, not kept beside the new ones**: a workspace cut before the allocation step is refused at every entry point, and the cohort kind (generated / supplied) is stated by the caller rather than inferred from whether a manifest exists — an inference that could not tell a legitimately manifest-free supplied cohort from a pre-allocation defect. Refusal means **verified**, not present: `verify_cohort_reads` (the read form that replaced `verify_generated_cohorts` in M2.1's S9) binds the manifest to the digests of the sample files a run read, to the disease sets recomputed from those records, to its own allocated digests and to disjointness. The one exception it named rather than glossed — `scripts/evaluate_model.py`, behaviourally frozen as Mode A's calibration reference with no preflight, its bytes pinned by `tests/unit/test_frozen_evaluator.py` — is closed: both were deleted by the owner's decision of 2026-10-10 (contract M2.1 decision 6). Note that 11i does **not** answer item 11: §5's (i), (ii) and (iv) are institutional questions, and this arm builds what makes them answerable | 11 | author | **done in engineering**; §5's (i), (ii) and (iv) remain institutional decisions |
| **14** | **Test results recorded by the pipeline, and shown where a model is chosen** — proposed by the project owner (2026-10-02). When a checkpoint selected on validation is tested on MyGene2 or the institutional cohort, the pipeline writes the result itself, bound to that checkpoint, so it cannot be omitted or mistyped and does not live in an editable file name; the result is then shown when the model is loaded and in the list of selectable models. Proposed shape: a separate Test tab; test-cohort management with version verification modelled on the training datasets'; and a choice between writing results automatically and, while a new test cohort is itself being evaluated before it becomes a baseline, running several rounds and writing one by hand. **What exists:** EVALUATION_COHORTS §6.5's ledger (`src/evaluation/sidecar.py`, `scripts/record_evaluation.py`) already binds a result to the checkpoint's SHA-256 and the cohort's digest beside the weights, refuses contradictions and survives a rename; it is written by a separate command and read by no API or UI. **To settle before design:** where the result lives (inside the `.pt` changes the digest every record cites — EVALUATION_COHORTS §5 item 4); whether ordering or choosing models by test score is allowed, since a cohort used to choose stops being an independent acceptance test (EVALUATION_COHORTS §4.1(6)); what tamper resistance is claimed beyond digest binding (since settled: binding only, no signing; D2, revision 4); and how a cohort is identified (digest and version label, EVALUATION_COHORTS §5 item 5). Taking it up reverses 11i's "UI out of scope" for this surface. **Plan drafted:** [`PLAN_TEST_RESULTS.md`](PLAN_TEST_RESULTS.md) sets out what exists, the decisions (D1–D8) and an order of work, and records adjacent defects found while planning (F1–F4). Revision 2 settles D3 by the scorer policy — a test result is the model's own disease ranking (Mode C, policy statements 1–3), with no SP term. Revision 3 states one path per concern (plan §4.0): ledger v2 does not read v1, one resolver locates each cohort's file so that the digest recorded is that of the samples scored (today `resolve_cohort` and `read_samples` locate it separately), and the copied checkpoint metrics share auto-selection's key list. Its 2026-10-05 amendment upgrades existing paths instead of working around them — the job runner is `training_manager` reshaped, with the Training Console moved onto it, and the existing checkpoint listing route is upgraded, its unsafe load converted — and records corrections from an independent verification: only the cosine is shared with the served scorer, not the pool, so unifying the pool is part of the constraint on B-1; D4 groups by seven named scoring-semantics fields and never promotes past a contested record; D5's block holds within one key, with a test run's inputs fixed; the checkpoint list follows SPEC_4's C3. F1–F3 are fixed (`d95965e`, `88569e4`, `fe44b8f`, `20f43d1`, `a57ed2d`, all on `main`). Revision 4 (2026-10-06) splits it into two phases: phase 1 is a complete CLI flow (import, run with the contract's checks, keep and record the report, show it on the existing model status), phase 2 the Test tab, job runner, D8 and ordering/arbitration; deterministic runs are no longer described as a cross-environment guarantee; D2 is settled (binding only, no signing). Its amendment after the review of `463a0df` removes the v1-ledger display condition, records R10 as the contract's §5.4 defines it, and moves training start's `checkpoint_dir` restriction back to phase 2 with the listing | 11, 11i, **15** (M1–M3 before phase 1) | author + institution | **plan revision 4; questions 1–2 gate phase 1, question 3 phase 2** — not implemented |
| **15** | **The provenance contract: every link in the pipeline checked where it is used** — the owner's goal (2026-10-06) is that a model always matches the exact setup that trained it, dataset version and every stage's settings included, with nothing misaligned between stages; the threat is misalignment, not an account holder, so no accounts, keys, signing or chain. **Decided:** a model whose recorded training graph does not match is refused (its SP and every graph-derived score cannot match it); old-pipeline artifacts are not supported (archived, not deleted, no compatibility mode); in the first version, resume continues the same data (fine-tuning on other data not yet supported). **Found:** links inside a workspace refuse today, but every link with the model on one side is checked by structure only, recorded and unread, or absent — model↔graph (structural warning only; the digest comparison was deferred, `fingerprint.py:226-234`), resume parent, training settings in effect (seed applied after the model is built; that ordering fixed ahead of the rest of M3a, `290c03a`), served identity (literal `model_version`), dataset record (no production reader); a failed model load fell back to path reasoning and was published as success (fixed by M1). **Plan:** [`PLAN_PROVENANCE_CONTRACT.md`](PLAN_PROVENANCE_CONTRACT.md) — M1 B-2's fail-closed core; M2 model↔graph enforced, producer and consumer together, on digests taken from the bytes actually parsed (before M2.1 each input was hashed by path at one moment and loaded at another; serving, training and measurement now record from one read, measurement's runs that included Mode A having been deleted by the Mode A/B retirement's R2, while item 19's sites do not), with the resume parent checked before anything is restored and a read-only rebuild inventory; M3 training/dataset/serving records and checks; M4–M5 are item 14's two phases. Existing records are upgraded; nothing new beside them except item 14's cohort version and report store | 14 depends on it; B-2 (`DISEASE_SCORER_POLICY.md`) | author | **plan revision 2; M1 implemented (2026-10-07), code-reviewed with no P1/P2 at `ed4872a`, and accepted on the homelab GPU at `7714459` (2026-10-08); M2.1's work plan revised after review with the owner's decisions of 2026-10-09; the plan-level re-review of `0708789` closed the earlier P2 and found one more (S10's resident-memory rule), since fixed, and the plan-level incremental review of `7e1af96` found no P1 or P2; the owner decided on 2026-10-10 that S0 also removes `scripts/evaluate_model.py` and its pin test, and that S10's capacity acceptance is limited to its designated subject (contract M2.1 decisions 6 and 7), and the plan-level review of that record (`7e1af96..45d9dea`) found no P1 or P2; its deferrals are items 18–19; M2.1's S0 implemented (2026-10-10) and code-reviewed with no P1 or P2 (`45d9dea..3433bf8`); S1–S5 implemented (2026-10-10) and code-reviewed with no P1/P2 (`e3eefee..6f16373`; the shared `read_samples` has changed since, in item 8's `06fd0f1`, code-reviewed with no P1 or P2 in `1d0b5a6..06fd0f1` on 2026-10-12); S7 implemented for Mode C only (2026-10-11) and code-reviewed with no P1/P2 at `7327798` (2026-10-12), as that declared scope and not as every mode migrated: a `--modes C` run reads each input once, checks the manifest, the graph tensors and (for a generated cohort) both sample files from those reads, and records what it read (M2.1 checks nothing about the checkpoint), while measurement runs that include Mode A (`A`, `A,B`, `A,B,C`) were not migrated — since the owner's clarification of 2026-10-12 they were deleted by the Mode A/B retirement's R2 (item 9's row; 2026-10-12, code-reviewed with no P1 or P2 in `1d0b5a6..06fd0f1`), so since R2 every measurement run takes S7's reads (implemented; not covered by S7's C-only review); M2.1 is still not complete: the S6/S8 batch (`8421b8c..cddad9d`) and, in `1d0b5a6..06fd0f1`, R2 (`e210b12`) and item 8 (`06fd0f1`) were code-reviewed with no P1 or P2 on 2026-10-12, and S9, implemented on 2026-10-12, was code-reviewed with no P1 or P2 the same day (`02a6954..f481882`), but S10 is outstanding: its plan amendments (`f481882..c1b8a12`) were reviewed with no P1 or P2, its tools (kept outside the repository) had three P2 in S9's review and, fixed, passed re-review with no P1 or P2 (draft 3, 2026-10-12), and it ran on 2026-10-12 (every acceptance 1 check passes at the after commit; comparability has 1 UNKNOWN, the service state, supplemented by the owner's statement, exit status 2 kept; acceptance 2 and 3 explained except the serving reload's resident rise, R3 +0.023 to +0.031 GB, source not located; the reviewer recommends a limited diagnostic of consecutive reloads before acceptance 2 is closed, chosen by the owner and run on 2026-10-13: each process grew by about 0.4–0.5 GB at every reload at both commits, before M2.1 as after it, and the difference between the commits is ambiguous; the owner ended the reload-memory investigation on 2026-10-13 with the wheel's bundled mimalloc purge defects as a strong candidate cause, attribution not established (item 22), and decides acceptance 2 separately), and the owner accepted its capacity on 2026-10-13 on its designated subject with the remaining uncertainty explicit, so **M2.1's technical work is complete** (not merged); M3a's seed ordering implemented ahead of the rest of M3a (2026-10-11) and code-reviewed with no P1/P2 at `290c03a` (2026-10-12): the seed now fixes the initial weights; S6 and S8 implemented (2026-10-11), their batch (`8421b8c..cddad9d`) code-reviewed as its own range with no P1 or P2 on 2026-10-12 (the later increment narrowing their claims, `cddad9d..3e8e559`, was reviewed with no P1/P2 on 2026-10-12, which is not a review of that batch) — they covered the workspace reads only; the resume parent's read and its checks moved to the top of the run, and a missing parent refuses, in M2.3; M2.2–M2.6 implemented (2026-10-13); their code review found three P2, closed by the re-review of `0aabfc3..26fe84c` with no new P1 or P2; M2's acceptance passed on the GB10 at `82d7573` (2026-10-14), functional only, the probe's allocator follow-up open; not merged; the rest of M3–M5 not implemented** |
| **16** | **One rule for a case's phenotype list, from import to scoring** — raised by R10's case label (contract §5.4) and the owner's question whether removing repeated phenotypes is what the original SHEPHERD or current practice does. **Found:** a repeated phenotype changes the GNN mean, the SP mean distance and path search; the WebUI removes repeats by string, the API keeps them, the direct pipeline keeps them and truncates to 100 before dropping unknown ids, so "100 × A, then B" loses B there; `InputValidator` removes repeats after truncating and has no production caller; the current generator writes no repeats; the original SHEPHERD keeps repeats and gives them no stated meaning; Phenopacket does not require unique terms. **Plan:** [`PLAN_PHENOTYPE_NORMALISATION.md`](PLAN_PHENOTYPE_NORMALISATION.md) — remove repeats by graph node after mapping, first-occurrence order, never merge patients, ancestors or present/excluded pairs; refuse instead of truncating; producers normalise and record the rule version and counts, readers check; N1 (serving) stands alone, N2 (stored data) rides on the contract's M2.1 and M3b, and the D6 importer uses it from its first version | 15 (M2.1, M3b, M3c); 14 (D6) | author | **plan revision 2, amended; the owner adopted the rule, `InputValidator`'s removal and decision W (the WebUI stops removing repeats; a clear message for a list over the limit), 2026-10-07; N1 implemented, code-reviewed with no P1/P2, and accepted on the homelab GPU at `af49bd3` (2026-10-07); N2 not started** |
| **17** | **Three defects found during N1's homelab acceptance** (2026-10-07; not caused by N1, present on `main` at `cffad34`). **F5 — the Diagnosis tab's model-configuration fields show the saved UI preferences, not the loaded pipeline.** `data_dir_input` takes its value from `.shepherd_ui_config.json` (`src/webui/components/diagnosis_panel.py:903-905`), and the status line names no path, so after a start with `SHEPHERD_*` set the tab showed `hpo_2026_0623_v2_orpha` while `hpo_2026_0929_5a` was served; pressing Load / Reload would have switched to the shown one. Likely fix: when a pipeline is loaded, show its data directory and checkpoint (the status route already returns `current_data_dir` and `current_checkpoint_path`). **F6 — `matching_phenotypes` is always empty on the real path.** The response model declares it (`src/api/routes/diagnose.py:183`), only the mock path fills it (`:453`), and the CSV and report exports carry the empty column. **F7 — the launcher gives up opening a browser after 60 s** (`scripts/launch/shep_launch.py:37`), and a startup that loads a large SP table eagerly took about 84 s. Separately recorded, not a defect: PyTorch warns that GB10's CUDA capability 12.1 is above the 12.0 it lists | — | author | **recorded** — not started |
| **18** | **The evaluation ledger's stale-write guard compares a digest that may not describe the ledger it read** — found while planning M2.1 (contract §4, M2.1 decision 5). `record_evaluation.py` parses the ledger (`:84` → `sidecar.py:224`) and then hashes the path again (`:85` → `sidecar.py:403`); `write_ledger` compares that second digest with the file (`sidecar.py:433`) and replaces it (`:448`). A write landing between the two reads makes the expected digest describe the other writer's file while the append is built on the older parse, so the comparison passes and the other writer's records are erased with no trace — reproduced against the real functions on temporary files. **Fix:** parse and hash the ledger from one read, so the expected digest is that of the records appended to; the same interleaving is then refused. The report the record is derived from has the same shape (`:100` parse, `:102` hash, recorded as `source_artifact_digest`, `sidecar.py:318`) and is fixed in the same change. **Not closed, and not claimed:** the check-then-replace window inside `write_ledger` (`:433` to `:448`), which `sidecar.py:418-426` already states; the ledger stays single-writer by operational rule. No lock, job runner or new ledger is added | — | author | **recorded** — not started |
| **19** | **Reads and writes deferred out of M2.1, each with its home** — M2.1 makes its four entry points parse and hash each recorded input from one read (contract §4, M2.1). The sites in the registry below the table read an input apart from its digest, or record a digest apart from their parse, and stay outside M2.1 by its decisions 4 and 5 (the owner, 2026-10-09), its "also unchanged" list, or, for 19.18, the reviewer's finding that Mode A cannot run on the designated checkpoint (its read stays its own: S7 was implemented for Mode C only, and runs that include Mode A were not migrated: since the owner's clarification of 2026-10-12 they were deleted by the Mode A/B retirement's R2 (item 9's row; 2026-10-12), and no Mode A builder that can run on a checkpoint the current writer produces was built, 19.18); 19.19 is instead removed by S0 (accepted at plan level in the re-review of `0708789` and, as revised by then, in the incremental review of `7e1af96`; its additions for contract M2.1 decision 6 accepted at plan level in the review of `45d9dea`; implemented 2026-10-10 and code-reviewed with no P1 or P2 at `3433bf8`). Each entry names the relation it does not guarantee and where it belongs, so none is lost when M2.1 closes. M2.1 restores nothing of the frozen evaluator, adds no checkpoint field and adds no fallback | 15 (M2.1) | author | **recorded** — each entry moves with its home |
| **20** | **HGT: two PyG problems on the formal build path, described separately in [`PLAN_HGT_LINEAR_PATHS.md`](PLAN_HGT_LINEAR_PATHS.md)** — found while researching an AMP-only comparison ([`FINDINGS_PAPER_PROTOCOL_AND_AMP.md`](FINDINGS_PAPER_PROTOCOL_AND_AMP.md) §4.3); torch_geometric 2.8.0.post1. **(a) A correctness defect.** `HeteroDictLinear`'s grouped branch (HGT's `kqv_lin`, `out_lin`) builds its inputs in its layers' order and stores the outputs under `x_dict`'s keys (`torch_geometric/nn/dense/linear.py:439-449`), so each node type can receive another's projection. On the formal path the two orders differ (training's loader, layers after the first, `build_index`). When the branch runs it is more often silently wrong than an `IndexError`. It runs with `torch_geometric.backend.use_segment_matmul = True`, or by PyG's default from ten node types up; production has three and the builder can name at most eight, so it is not reached today. Fix (Edit 1, approved by the owner on 2026-10-12 to land on its own; implemented 2026-10-12): `HeteroGNNLayer` passes the conv its node features in its layers' order, with a construction check, and its edges in the conv's edge-type order (the related hazard of the plan's §1.7); bitwise neutral today (CPU), pinned by `tests/unit/test_hgt_node_type_order.py`. **(b) A computation-path and reproducibility problem.** `HeteroLinear` (HGT's `k_rel`, `v_rel`) times `segment_matmul` against a loop per instance, row-count magnitude and process, and keeps the faster (`linear.py:258-285`, `:301-313`). The choice flips between processes, is recorded nowhere, differs between training and serving, and hides a loop backward about 35× slower. Proposed fix: a bounded GB10 comparison first, then the simplest sufficient policy applied in `HeteroGNNLayer`, explicit in the model configuration, with the requested policy, the compile setting and the observed kernel recorded separately (a compile wrapper does not mean the loop ran). Both fixes sit at the one place every entry point builds and runs HGT. GAT is unaffected | — | author | **(a) implemented (2026-10-12)**, code-reviewed with no P1 or P2 on 2026-10-12 (`1d0b5a6..06fd0f1`) at CPU code-review level only, not a GB10 performance or CUDA numeric acceptance; **(b) proposed (2026-10-11)**, waiting for its GB10 comparison. The plan was reviewed 2026-10-12 (`290c03a..1d0b5a6`) with one P2 (the kernel inferred from the compile setting), applied in `418b37e` (the kernel observed, not inferred), whose review (`1d0b5a6..06fd0f1`, 2026-10-12) found no new P1 or P2 and lets that P2 be closed; Edit 1 is `8eae614` |
| **21** | **AMP training study: is a model trained with AMP worse, and what does AMP save in training time and memory?** — bounded research. The owner reported on 2026-10-12 that what the owner and the institution want to know is whether an AMP-**trained** model loses accuracy or quality, weighed against the large training-time saving AMP gives some GNN architectures: a **training** study, **not** the same-checkpoint scoring comparison with autocast switched (the old differential's AMP report, 7a's AMP leg, the retirement assessment's §8 item 7 options (a)/(b)), whose wait that clarification closed. Background: [`FINDINGS_PAPER_PROTOCOL_AND_AMP.md`](FINDINGS_PAPER_PROTOCOL_AND_AMP.md) §4 — what is fixed and recorded today (§4.1), GPU nondeterminism and HGT's kernel path (§4.3, item 20), the two experiments told apart (§4.6). **Only the existing path:** `train_model` → `Trainer` → checkpoint → Mode C; no AMP-specific trainer, launcher or test engine. **Pilot first:** one pair on one CUDA GPU (on CPU `use_amp` has no effect, `trainer.py:380`), same verified workspace bytes (equal `training_input_digests` in both checkpoints), same architecture, hyper-parameters, training length, seed and worker count, so the two start from identical initial weights (possible since M3a's seed ordering, `290c03a`). The two runs' `config.yaml` (`scripts/train_model.py:739-741`) differ only in the AMP setting, with the AMP arm's dtype declared in advance; the dtype and the worker count are set only through YAML, where a missing file or an unknown key is ignored silently (findings §4.2), so the written `config.yaml` is what is compared. Separate output, checkpoint and log directories, since the defaults collide. It first confirms which ops AMP actually reaches for the architecture: HGT runs its convolution in fp32 with autocast off (`src/models/gnn/layers.py:200-218`), so for HGT AMP changes only the other layers. It records wall time, peak memory, GradScaler overflow (skipped) steps (float16 only; bfloat16 uses no scaler, `trainer.py:382-388`), and the training and validation curves. The curves and per-epoch time are already in the metrics log; skipped steps are detected (`trainer.py:561-568`) but not counted, and training reads no peak memory, so those two are recorded through the existing training path. **Formal comparison, only if the owner widens after the pilot:** several paired seeds; each arm picks its checkpoint by the same validation rule, declared before the run, ties included (with AMP on, the selecting validation pass runs under autocast, `trainer.py:666`, which is part of what the AMP arm is); both arms scored by the same Mode C, on the same test cohort, at the same scoring precision (Mode C runs without autocast); a seed, an epoch or the AMP setting is never chosen by test score. Report per-pair differences and their spread across seeds, beside the time and memory saved. Fixed seeds do not remove CUDA nondeterminism, so not every observed difference is attributable to AMP; whether runs use deterministic mode stays open (findings §5 point 5). **HGT:** an HGT pair needs item 20 (b)'s kernel path pinned identically in both arms through the formal build path, not a study script (item 20 (b)'s fix, Edit 2, waits for its own GB10 comparison and is not decided); otherwise the pair uses GAT, so the timing-based kernel choice cannot confound the comparison. The WebUI forces AMP off for HGT (`src/webui/components/training_console.py:239`), so an HGT AMP arm runs from the CLI, YAML or API. **Not decided:** the acceptable quality loss is a product/research tolerance not yet given, so no margin is assumed; the pilot is reported first, and whether to widen is decided on it. **Not a prerequisite of M2 or of R2**, and nothing else waits on it | Pilot: M3a's seed ordering (`290c03a`) — **satisfied** (implemented and code-reviewed, `7327798..290c03a`; shown on CPU by `tests/unit/test_training_seed.py`, not yet observed on the GB10). An HGT pair: item 20 (b)'s pinned path, else GAT. Formal comparison: the pilot's report and a decision to widen; a test cohort Mode C scores (which cohort is findings §5 point 7; the formal Test arrives through item 14's import) | author; the owner sets the tolerance and decides on widening | **recorded (2026-10-12)** — not started; pilot first |
| **22** | **Serving reload keeps resident memory: a dependency limitation, not a SHEPHERD change** — each reload left about 0.4–0.5 GB more resident memory in the serving process, at `90a668f` and `343e90e` alike (S10's reload diagnostic, [`reload-diag-2026-10-13/`](scorer-measurement/reload-diag-2026-10-13/README.md)); the earlier pipelines' watched objects were freed, and a later run did not repeat the steady growth. The PyTorch wheel in use is confirmed to contain mimalloc purge defects that upstream has since fixed (`libc10.so` allocates CPU memory through mimalloc 2.2.4; upstream `acd6f6c` and `4e50cec`); they are a strong candidate cause of the observed memory retention; the attribution of all of the increase is not established ([`reload-inproc-segments-2026-10-13/`](scorer-measurement/reload-inproc-segments-2026-10-13/README.md)). **Decided (the owner, 2026-10-13):** the investigation ends; no change to the product pipeline, no `MIMALLOC_*` setting, no profiler or new diagnostic tool, and no targeted patch for a dependency-level defect | a PyTorch wheel whose `libc10.so` is shown, in its machine code rather than a version string, to carry both upstream fixes | owner | **recorded (2026-10-13)** — no action until such a wheel exists; then a reload check with phase 1's procedure |
| **8a** | B-0.5 protocol and output-contract **design**. **Consumes item 11's holdout decision and may not redefine it** | 1, **11** | author | **before** any expensive run |
| **8b** | B-0.5 institutional execution | 8a, 6, exact artifacts, production-path prerequisites. 7b dropped by the owner's clarification of 2026-10-12: it carried 7a and the A/B measurement, which B-0.5 does not need (retirement assessment §4); the rewritten 7b takes its served ranking from 8b instead. Was: 8a, 7b, 6, exact artifacts, production-path prerequisites | both | blocked |
| **9** | **Replaced by the Mode A/B retirement change, R2** ([`ASSESSMENT_MODE_AB_RETIREMENT.md`](scorer-measurement/ASSESSMENT_MODE_AB_RETIREMENT.md) §9) — superseded by the owner's clarification of 2026-10-12, which reported that the institution had already authorised retiring the old pipeline's Mode A/B tools. R2 deletes Modes A and B per the assessment's existing lists and rewrites the texts those lists name; it no longer waits for the institution (assessment §8 items 4 and 7 closed), and its deletion gate is that the migrated tests are in place (R1), not a test *of* Mode A. The owner decided on 2026-10-12 to start R2 now, together with the assessment's §8 item 5 (the measurement-only manifest fields and the measurement seed removed) and item 9 (`--modes` removed), with no compatibility flag or read branch for older records; item 8 (refuse non-integer and boolean ids) follows R2 as its own change, implemented the same day (`06fd0f1`; code-reviewed with no P1 or P2 on 2026-10-12, in `1d0b5a6..06fd0f1`). Was: Mechanical rename (~70 refs, 9 files), then rewrite the checklist, then delete the oracle-only surface. `scripts/evaluate_model.py`, its pin test and the oracle-parity tests in `tests/integration/test_legacy_equivalence.py` already went in contract M2.1's S0, the gate's one exception (§5; the owner's rule of 2026-10-09 and decision of 2026-10-10, contract M2.1 decisions 3 and 6) | R1 — implemented and code-reviewed with no P1/P2 at `3d53c9b` (2026-10-12). Was: **1d passed review incl. its institutional CUDA run** | author | **R2 implemented (2026-10-12) and code-reviewed with no P1 or P2 (`e210b12`, in `1d0b5a6..06fd0f1`, 2026-10-12)** — deletions and wording, with the assessment's §8 items 5 and 9 (retirement assessment §9.2). Was: behaviour-neutral |
| **10** | **Commit bounded evidence for M1-M5** — four JSON files and the three scripts that emit them. **Not raw console output** (§5.2). Unblocks 2, 3, 6 and 7a | — | institution + author | **done** — run on the deployment-sibling machine, evidence cited by digest in §2; M1-M4 confirmed, M5 corrected (§2.4) |


**Item 19's registry — reads and writes deferred out of M2.1.** Sites at `90a668f`. M2.1 refuses
none of them, and the relation each row names is left as it is. M2.1 touches two: S0 removes
19.19, and S3 fixes 19.21's recorded digest but not its placement. S7 was to pass 19.18's Mode A
builder (`build_legacy_mode_a_model`) the run's checkpoint read; it was implemented for Mode C only
(2026-10-11), so that builder kept loading its own path until it left with Modes A and B in R2
(2026-10-12), which replaces item 9 (superseded by the owner's clarification of 2026-10-12: it was
to leave with item 9's oracle-only surface, after 7a, now cancelled). No replacement that can run
on a checkpoint the current writer produces was built, and 19.18 is closed by R2 (implemented
2026-10-12; code-reviewed with no P1 or P2 in `1d0b5a6..06fd0f1`).
The others that M2.1's static pin meets are allowlisted with the
reason given. (As implemented in S9, 2026-10-12: the list is `ALLOWED` in
`tests/unit/test_provenance_pins.py`, each entry with its use and owner.)

| # | Site | Reads → records or compares | Does not guarantee | Belongs to |
|---|---|---|---|---|
| **19.1** | Reload route candidate scoring — `_checkpoint_score`, `src/api/routes/pipeline.py:325-333` (`torch.load(weights_only=False)`, `:328`), per candidate via `select_checkpoint_in_dir` / `select_auto_checkpoint` (`src/utils/checkpoint_paths.py:103-182`) | every `*.pt` in the architecture directory → compares the `logs` ranking score; records nothing | that the file served holds the bytes that were scored | **M3c** — contract M2.1 decision 4: "the build, and later M2.4, use the final `read_checkpoint`"; M3c takes the reported metric from the served build's own read |
| **19.2** | Reload route second read — `src/api/routes/pipeline.py:361-368` (`:363`) | the chosen path again → `best {metric}={value}` in `selection_reason` | that the reported metric is of the bytes scored or served: the build reads a third time (`src/inference/pipeline.py:973`) and `checkpoint_meta` comes from that read — chose A, reported B, served C | **M3c** — same text |
| **19.3** | `TrainingManager.get_checkpoints` — `src/api/services/training_manager.py:436-473` (`:464`, `weights_only=False`); also feeds the resume dropdown (`src/webui/components/training_console.py:499-524`) | each top-level `*.pt` → name, path, size, mtime, `epoch`, `logs`; no digest | that the epoch and metrics shown describe the file later resumed or loaded | **Item 14 phase 2 (contract M5)** — `PLAN_TEST_RESULTS.md` §6 step 5, one listing function with `weights_only=True`; added here: what the list shows, and any digest, comes from one read |
| **19.4** | `scripts/migrate_checkpoints.py` — `_conv_type_of` `:42-58` (`:50`, `weights_only=False`), then `shutil.move` by path (`:107`) | a legacy flat checkpoint's `config` or weight names → moves it into `checkpoints/<conv_type>/` | that the file moved is the file classified | **None** — old-pipeline artifacts (contract decision 2; §7). No existing checkpoint records `kg`, so M2 refuses every one (M2.6, "Consequence"). The reload refusal still recommends this script (`src/api/routes/pipeline.py:375-379`); that text changes with M2.4 |
| **19.5** | Ledger stale-write guard — `scripts/record_evaluation.py:84-85`, `:133`; `src/evaluation/sidecar.py:214-234`, `:399-451` | parses the ledger, hashes it again, compares that hash before replacing | that the expected digest describes the ledger appended to | **Item 18** |
| **19.6** | `source_artifact_digest` — `scripts/record_evaluation.py:100` parse, `:102` path hash; `sidecar.py:318` | the report → the record's source digest | that it is the digest of the report the record was derived from | **Item 18**; otherwise item 14's ledger v2 (M4; not the ledger file's `schema_version` 2 that R2 set, `src/evaluation/sidecar.py`), which carries report digests |
| **19.7** | `find_checkpoint` — `src/evaluation/sidecar.py:366-383` | hashes each `*.pt` → refuses unless one matches the report's checkpoint digest | that the matching file keeps those bytes afterwards; identification, not a recorded input | **None** — kept by M2.1 ("`file_sha256` stays only for identifying a file nobody is about to parse") |
| **19.8** | `scripts/measure_served_pipeline.py` — subject `:1219-1223`; workload `:487-488`, `:523` | `kg.json`, checkpoint and SP table hashed by path before the service starts; the service reads them itself; `val_samples.json` parsed `:487`, hashed `:523`; `kg.json` parsed `:488`, unrecorded | that `served_readings.json` names the bytes the service loaded, or the samples the workload was drawn from | **M3c** for the subject — cite the served identity from the service's own read. The workload read: **a separate fix** on M2.1's `read_samples`, landing **after** S10, since S10 requires this script byte-identical at both commits. M2.1's S10 runs this script, and its evidence states the limitation |
| **19.9** | `scripts/audit_split_feasibility.py:456`, `:545` | parses `kg.json` → `inputs.kg = file_sha256(kg_path)` | that `inputs.kg` is the graph analysed | **A separate fix**, on M2.1's `KnowledgeGraph.read_json` (S3) |
| **19.10** | `scripts/audit_checkpoint_family.py` — `inspect_checkpoint` `:117-165` (`:138`), `build_report` `:290` | loads each `*.pt` → keys, widths, score; then hashes every file for `checkpoint_digests` | that each digest names the bytes inspected | **M2.6** — the inventory is "an option of an existing audit". It parses each checkpoint, so its digest comes from that read (M2.1, amended 2026-10-09) |
| **19.11** | `scripts/benchmark_sp_lookup.py` — `build_artifact_lookup` `:245-369` | loads the table (`:298`), checks the `build_id` pairing (`:305`) → `sha256 = file_sha256(path)` (`:352`) | that `sha256` names the table measured | **A separate fix, optional** — record the loaded tensor's `build_id`, as M2.1's SP exception does at serving, and label the file digest as identification of the path |
| **19.12** | `scripts/audit_sp_reachability.py` — `build_report` `:436-569` | memory-maps the table (`:427`) → `artifact_digest = file_sha256` (`:540`); `num_nodes.json` parsed `:482`, unrecorded; the sidecar digest from its parsed bytes (`:544`) | that `artifact_digest` names the table scanned; `num_nodes.json` is unidentified | **A separate fix, optional** — as 19.11, plus `num_nodes.json`'s digest from its read |
| **19.13** | `scripts/build_knowledge_graph.py` — annotation files parsed `:291`, `:298`, hashed `:342-343`; ontology digest `:336` | each source by role and digest → `kg.provenance.json` | that the annotation digests are of the bytes parsed; the ontology digest is the loader's handle digest (`src/ontology/loader.py:243-262`), which survives an atomic rename and not an in-place rewrite | **KG build record v2** — contract §4 "Not prerequisites", named by M3b |
| **19.14** | `_workspace_provenance_status` — `src/kg/artifacts.py:252-300` | parses the manifest again (`:278`), hashes `kg.json` (`:291`) → `provenance_status`; no production caller | that the declaration and graph digest it checks are the run's own reads | **M3b** — "read through M2.1, like every other input" |
| **19.15** | `_provenance_status` — `src/kg/provenance.py:357-418` | hashes the record (`:381`), compares with the declaration (`:389`), then parses it (`:398` → `:275`, `:283`) and compares its `kg_digest` (`:404`) | that the record matched against the declaration is the record parsed and matched against the graph | **M3b** — same text |
| **19.16** | `write_provenance` — `src/kg/provenance.py:211-233` | staged write → returns `file_sha256(target)` (`:233`), bound into the manifest (`src/kg/workspace.py:315`, `:342`) | that the bound digest is of the payload written | **M3b** ("required at write and read") |
| **19.17** | `write_workspace` graph digests — `src/kg/workspace.py:297-300` | hashes `kg.json` and the tensors by path after in-place writes → manifest and record | that the bound digests are of the bytes this writer serialised | **M3b** — as 19.16 |
| **19.18** | Measurement Modes A and B — `build_legacy_mode_a_model`, `scripts/measure_scorer.py:230-262`; `SUPPORTED_MODE_SETS` `:443` | indexes `checkpoint["metadata"]` and `["in_channels_dict"]` (`:251-252`); B ran only beside A. It loaded its own checkpoint path until R2: M2.1's S7 was implemented for Mode C only, and runs that included Mode A were never migrated | that it runs on a checkpoint the current writer produces: that writer (`src/training/callbacks.py:296-327`) writes neither key and no scanned checkpoint carries them (M1), so it raises `KeyError` on those; only fixtures reach it | **Removed by R2** (implemented 2026-10-12; code-reviewed with no P1 or P2 in `1d0b5a6..06fd0f1`), which replaces item 9: deleted with Modes A and B, as 19.19 was by S0 (retirement assessment §4). Superseded by the owner's clarification of 2026-10-12: it was "Item 9, after 7a", the oracle-only surface (`scorer-measurement/README.md`, "What is actually oracle-only"), and 7a is cancelled. M2.1 binds, measures and accepts Mode C only. Nothing replaces this builder: the CLI's Mode A was retired, not rebuilt |
| **19.19** | `scripts/calibrate_mode_a.py` — the digest bracket `:396`, `:403`; `calibration.json` `:433` | path hashes before the oracle and after the harness → `calibration.json` | that the recorded digests are of the bytes either subprocess parsed | **Removed in M2.1's S0** (2026-10-10; code-reviewed with no P1 or P2 at `3433bf8`). The owner's rule of 2026-10-09 is to remove it if nothing necessary uses it; S0's check found no necessary user, and the reviewer accepted S0 at plan level in the re-review of `0708789` and, as revised, in the incremental review of `7e1af96`; the owner has since extended S0 to remove `scripts/evaluate_model.py` and its pin test (contract M2.1 decision 6, 2026-10-10, accepted at plan level in the review of `45d9dea`). Item 7a was to write its own entry point; 7a is cancelled by the owner's clarification of 2026-10-12, so none is written (item 7a's row) |
| **19.20** | Path-reasoning-only serving — `src/inference/pipeline.py:531-532` (cited at M2.1 S6) | no model and no checkpoint: returns before the workspace is read and verified (`:574-578`); the graph was parsed by the API (`src/api/main.py:552`) | that the graph served matches any manifest | **B-2** — contract §8 question 2, not decided |
| **19.21** | The SP producer's workspace check — `scripts/compute_shortest_paths.py:418-425` | `--kg-path` and `--output-dir` independent; M2.1's S3 fixes the recorded digest, not the placement | that the table lands in the workspace whose graph it came from (serving refuses a mismatch, `src/inference/pipeline.py:787-790`) | **Contract §4 "Not prerequisites"** |

**Parked deliberately, not forgotten:** `task-scope/` Q2–Q5 (settled, unscheduled)
and `scorer-retraining/` (scoping only, four gates uncleared). Neither blocks nor
is blocked by anything above.

**Ontology provenance is unrecorded and its version is not pinnable — raised by
the maintainer, and the sharper half of the packaging problem below.** Verified
against the code rather than recalled:

| | Established |
|---|---|
| Cache location | `Path.home()/'.shepherd'/'ontologies'` — **outside the project**, in the operator's home |
| On a missing file | **Was** `urlretrieve('http://purl.obolibrary.org/obo/mondo.obo')` from a constant in `loader.py`. **Phase 2 changed how, not what:** the sources are configuration, every request and redirect passes the scheme and destination rules, and a fetched file is verified before it replaces anything. It is still **whatever is latest** at the source — Phase 2 lets a site name a file instead; it does not pin a download |
| The `version` argument | Used only as an in-memory cache key; the filename is `mondo.obo` regardless. **It reads as version selection and is not** — **closed by Phase 0**, which refuses any value but `"latest"` rather than ignoring it |
| Is the version knowable? | **Yes** — the OBO header carries `data-version`, and `hierarchy.py` already parses it |
| Is it recorded? | **Was no; closed by Phase 1.** `kg.provenance.json` carries all four source files by role, digest and raw `data-version`, bound to `kg.json`'s digest |
| Can a site select one? | **Was no; closed by Phase 2.** `--mondo-path` / `--hpo-path` name the file; without them, exactly one candidate across the configured roots is taken, and more than one refuses rather than ranking |

This is not hypothetical: the two machines used this session differ in MONDO
vintage (29,866 vs 32,109 disease nodes, different `kg_digest`) purely because
they were deployed on different dates. The differing digest *detects* it, which
is the manifest working — but it cannot say what differed, and a site cannot
deliberately reproduce another site's graph.

**The inert `version` argument is the part that should not wait**, whatever is
decided about pinning: an API that looks like it pins and does not is worse than
one that does not offer it. Recording `data-version` beside the build is the
cheap half and needs no new subsystem. Pinning — where the files live, whether a
mismatch refuses or warns — is a design question with real alternatives and
deserves its own plan. Both were out of scope for 5a.

**Where this stands now.** Phase 0 and Phase 1 of
[`PLAN_ONTOLOGY_PROVENANCE.md`](PLAN_ONTOLOGY_PROVENANCE.md) are implemented and
merged, which closes the two rows marked above. **Phase 2 is implemented** to
[`PLAN_ONTOLOGY_PHASE2.md`](PLAN_ONTOLOGY_PHASE2.md), whose design passed review:
`--mondo-path` / `--hpo-path` on the build CLI, a resolver over configured roots
that refuses rather than ranking, an imports policy that refuses every declared
import, a role check that stops a file being built into the wrong slot, and the
source list moved from `loader.py` into `configs/deployment.yaml` with the
scheme and destination rules enforced on every request. **One item is owed and
undelivered by design**: editing those URLs from the interface (§3.6.1) — the
backend is built for it, the front end is outside this programme.

A whole-branch adversarial review before the PR found the fetch path still
placing fallible steps after the state they should gate — a misfiled cache file
overwritten, a fresh copy published before it was checked — and those are
fixed (§3.8 there). **It also amended one approved default, flagged for
review:** a name this machine cannot resolve is left to a configured proxy
instead of being refused (§3.5.2), because proxy-only hospital networks could
otherwise never download.

**Measured on the homelab GB10 since**
([`EVIDENCE_probe_phase2_homelab.json`](EVIDENCE_probe_phase2_homelab.json)
`d01621f3…`, plan §4.4): the full deployment probe passed 48 of 48 — selection on
the June cache, two byte-identical real builds, and a real download of the
2026-09-01 releases through the production path, byte-identical to the
development sandbox's download-only run
([`EVIDENCE_probe_download_sandbox.json`](EVIDENCE_probe_download_sandbox.json)
`3b1a9beb…`). What routes the redirect hops took, and whether the real build
touched the cache, are not recorded (plan §4.4). That run's
first predecessor, in the sandbox, is what found the HPO regression fixed in
#107. **A hospital network has not been tried**: the files come from
`release-assets.githubusercontent.com`, not from a biomedical host, and only a
run there shows whether it is allowed.

Phase 3 (packaging) is unstarted and converges with the export/import idea
below.

**A workspace export/import pair — the maintainer's idea, recorded so it is not
lost.** Not a review finding and not scheduled. The hop-bound work exposed the
underlying problem: a workspace is seven files plus a shortest-path pair, the
`.pt` and its `.meta.json` belong to one build, and nothing stops a hand copy
from separating them. A packaging step that moves a workspace whole would remove
that class of accident at the source. **Whatever it would do, it replaces no
verification** — the manifest digests, the schema-3 recipe and the hop-bound
checks stay exactly where they are; an importer that trusted its own bundle
would reintroduce what they exist to catch.

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
would leave the harness with no acceptance at all. One exception, by the owner's
rule of 2026-10-09 and decision of 2026-10-10 (contract M2.1, decisions 3 and 6):
the oracle-parity tests in `tests/integration/test_legacy_equivalence.py`,
`scripts/evaluate_model.py` and its pin test went with the launcher
`scripts/calibrate_mode_a.py` before 1d's institutional run. The tests checked the
parity §3.1.2 retired as an acceptance, the evaluator was that parity's reference,
and its pin test kept the reference's bytes unchanged; the launcher could not run
that parity on the scanned family or on a checkpoint the current writer produces
(§3.1, M1, 19.18); and the differential calibration is untouched.

**Superseded by the owner's clarification of 2026-10-12.** Item 9 is replaced by the retirement
change R2, whose gate is that the migrated tests are in place (R1, implemented and reviewed),
not 1d's institutional CUDA run: 7a is cancelled, and Mode A is deleted rather than
accepted, so no harness acceptance is left for the gate to protect; Mode C's acceptance is direct
(retirement assessment §4, §7). The rule above held until then and is kept as history. R2 was
implemented on 2026-10-12 and code-reviewed with no P1 or P2 the same day (`1d0b5a6..06fd0f1`).

**8a comes before 7b for a reason:** designing B-0.5's output contract after an
expensive institutional run is how required evidence gets discovered too late to
collect.

### 5.0 Is any of this order forced? Checked at file level

Asked because reordering was being considered, and "these are independent" is a
structural claim that decays.

| Line | Files it touches |
|---|---|
| **B-0.4** (item 5) and **5a** | `src/inference/sp_index.py` — `SPLookup`, `build_sp_index`, `sp_mean_distances`; `src/inference/pipeline.py` — `_load_shortest_paths`, `_calculate_sp_score`; `src/inference/scoring.py` — `sp_scores_from_distances`, `validate_hop_bound`; `scripts/benchmark_sp_lookup.py`; `scripts/sp_scan_reference.py`; `tests/unit/test_sp_index.py`; `tests/unit/test_scoring_primitives.py`; `tests/integration/test_benchmark_sp_lookup.py`; `tests/integration/test_pipeline.py` |
| **Calibration** (1b-1e) | `src/training/trainer.py`; `src/evaluation/measurement.py`; `scripts/measure_scorer.py`; new trainer and D3 tests |

**Disjoint, and the layer contract keeps them that way.** The one file that could
have been shared is `src/inference/scoring.py` — it holds both the SP primitives
B-0.4 rewrites *and* the cosine primitives a naive extraction might have reached
for. `src.training` sits **below** `src.inference`, so the trainer cannot import
them; `trainer.py` today imports only `src.training.*` and `src.utils.*`, and
`make lint-imports` reports 4 contracts kept — 3 when this paragraph was
written, and `scorer-independence` was added afterwards to close the direction
the layers contract cannot express. A B-0.4 regression therefore has no
path to the calibration reference.

**Only two orderings were genuinely forced**, both in the table; the first is superseded by the
owner's clarification of 2026-10-12 (item 9 is replaced by R2, which follows R1's moved tests
rather than 1d's institutional acceptance):

- Item 9 (rename, then checklist rewrite, then deletion) after 1d's institutional
  acceptance — deleting the old acceptance path before the new one is accepted
  would leave the harness with none. §5's one exception went first: the
  oracle-parity tests in `test_legacy_equivalence.py`, `scripts/evaluate_model.py`
  and its pin test.
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
and what the recorded figure actually was. That 1.36% counts *phenotypes reaching
nothing*; it is not the prevalence of unreachable phenotype–candidate pairs, and
§3.3 records why the two must not be substituted for each other.

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
