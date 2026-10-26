# B-0.5 — minimal Mode D: the served ranking, measured

**Status:** design (Batch 2, 2026-10-26), for review; not implemented. It resolves the design
problem `PLAN_B03.md` §4 recorded. Mode D is the input to Policy Gate 1's comparison, not B-1:
building it changes no ranking anyone is served. The statistical protocol for the institutional
run stays BACKLOG 8a's, and the run 8b's.

**Authority:** `docs/DISEASE_SCORER_POLICY.md` (statements 1–7, §2.1, §6 gate 1). **Constraints
carried from this folder's README:** Mode D adds its own values, not a manifest structure; it does
not revive Modes A or B and is not a second evaluator; and `ModeResult`, `canonical_ranking`,
`ranks_of_truth`, `encode_full_graph`, `run_mode_c`, `build_shepherd_model`, the manifest's
authoritative fields, the digests, the CUDA gate and `src.kg.storage.file_storage` are not edited.

---

## 1. What Mode D answers, and what it does not

| | Mode C (implemented) | Mode D (this design) |
|---|---|---|
| Question | how well does the model rank the truth among **every** disease, by the approved scorer (raw cosine)? | how well does **what the Diagnosis page serves today** rank the truth? |
| Candidates | every disease row | the diseases path discovery reaches (`_find_all_paths`, BFS and the direct finder, with their order-dependent caps) |
| Score | raw cosine of the masked-mean pool | the served score of the deployment's scoring mode: `gnn_plus_shortest_path` (0.7·((cos+1)/2) + 0.3·SP) or `gnn_only` ((cos+1)/2) |
| Ties | lower global disease id | the served order: a stable sort, so discovery order |
| Code | `run_mode_c` | the served `DiagnosisPipeline`, through one new method that serving itself uses (§2) |

**Mode D does not measure** the path-reasoning fallback (it ranks without a model, so there is no
checkpoint to compare with Mode C; refused, §3), the live service process (it measures a pipeline
built from the same files by the same builder, §3), or anything B-1 will serve.

---

## 2. One served ranking, untruncated

`PLAN_B03.md` §4's problem: `run` truncates to `top_k` and builds paths and explanations, so an
untruncated served rank could not be had without duplicating candidate construction.

**Resolution: split `DiagnosisPipeline.run` at its truncation point, keep one implementation.**

- `DiagnosisPipeline.rank(patient_input) -> ServedRanking`: input validation and normalisation
  (`validate_input`), the node mapping, `_find_all_paths`, `_score_and_rank_candidates`' scoring and
  its stable sort — **everything `run` does up to and including the sort, nothing after it**. It
  returns the full ordered candidate list with each candidate's served fields, the phenotype
  account (`PhenotypeInputAccount`), and whether the input was refused and why.
- `run(patient_input, top_k, include_explanations)` becomes `rank()` followed by what it does today
  after the sort: the `top_k` slice and rank numbering, matching phenotypes, explanations, evidence
  and the summary. **Serving's output does not change**; a golden test over the demo graph in each
  scoring mode, ties included, pins that (§8).
- `top_k` never changes the order, only where the list is cut. So Mode D needs no `top_k`: a served
  rank `r` is what a clinician sees whenever the page's Top-K is at least `r`.

Mode D calls `rank()` only. It runs no explanation, evidence or summary step. Of each
`ServedRanking` entry it reads the order, the disease and the served score, nothing else, so the
result union that batch 4 builds (`../results-review/SPEC_1_RESULTS_REVIEW.md` §2.1), which
renames and re-types the other fields, does not reach Mode D.

---

## 3. Inputs, alignment and refusals

**The pipeline is built as the service builds it.** Mode D calls the service's own builder,
`src/api/main.py:build_pipeline(kg_path, checkpoint_path, data_dir, device,
expected_checkpoint_sha256)` (refactored into an importable home if `src.evaluation` cannot reach it
under the layers contract), so `PipelineConfig` is the default the service uses and the served
identity reports the states the service would (checkpoint `verified`, graph `verified`, the SP table
`verified` or `absent`). Each file is read once, by the component that parses it, as the service
reads it.

**The cohort is read as Mode C reads it**: `resolve_version(expected_sha256=)`,
`_read_version_samples` (the samples' digest, and `len == samples.n == counts.ranked_eligible`).

**Alignment, checked before any case is ranked:**

| What | Check |
|---|---|
| Checkpoint | the served identity's checkpoint digest equals `--expect-checkpoint-sha256` when given, and is recorded |
| Graph and dataset record | `verify_cohort_binding(version, bound)` with the pipeline's bound digests (the four graph roles and the dataset-record digest), through a new public read-only `DiagnosisPipeline.bound_inputs()`; today they are private (`_bound_digests`) |
| SP table | recorded by its `build_id` and digest; whichever the workspace holds is what the deployment serves, so Mode D measures `gnn_plus_shortest_path` with a table and `gnn_only` without |
| Settings in effect | the served identity's `settings` (η, discovery settings, normalisation version) recorded whole; folded into the manifest's semantic values (§6) so a change of settings is a different measurement |
| Numerical regime | the pipeline's embedding forward observes and exposes its regime (`embedding_regime`: dtype, autocast); the manifest's `amp_*` and `dtype` fields must equal it, as Mode C's `assert_manifest_describes_regime` requires of `encode_full_graph` |

**Refusals, before ranking:** a scoring mode of `path_reasoning_fallback` (no model); a generated
split (`--split`: Mode D measures a registered cohort version only, the form Gate 1's report needs);
any alignment check above failing.

**Each case's input.** `samples.json` holds graph indices. Mode D turns them into the served input
form through the graph's own reverse mapping (`kg.get_reverse_node_mapping`), the mapping the
importer used in the other direction, so the pipeline's validation sees exactly the terms the
importer mapped. No source strings are needed, and none are read.

---

## 4. Per-case outcomes and denominators

The cohort version already excluded cases at import (`ground_truth_absent`: the disease is not a
graph node; `phenotype_unmappable`). Those never reach either mode; both modes rank the same
`counts.ranked_eligible` cases, and the report repeats the import counts beside the result.

Each ranked-eligible case gets exactly one Mode D outcome:

| Outcome | Meaning | Served rank |
|---|---|---|
| `ranked` | the truth is among the served candidates | its 1-based position in the full served order |
| `not_reached` | discovery did not reach the truth (including a case with no candidates at all, counted separately as `no_candidates`) | none |
| `refused` | the served validation refused the input; today only "more than `max_phenotypes` (100) distinct terms", which the importer does not cap | none |

`n_cases = n_ranked + n_not_reached + n_refused`, checked. The word "absent" is not used: Mode C's
`n_ground_truth_absent` means the truth was missing from every disease row and is fatal there; a
truth Mode D's discovery does not reach is a result, not a defect.

---

## 5. Metrics, the comparison and its labels

**Two denominators, both from the one report** (the reviewer's literature note, PLAN_TEST_RESULTS
§4.9: an end-to-end figure counts what the system could not rank as zero):

| Family | Denominator | Metrics |
|---|---|---|
| End to end | `n_cases` | `end_to_end_mrr` (1/r for `ranked`, 0 otherwise), `end_to_end_hits@K` for K = 1, 5, 10, 20, 50, 100, `coverage` = `n_ranked / n_cases` |
| Among reached | `n_ranked` | `reached_mrr`, `reached_mean_rank`, `reached_hits@K`; omitted, with the reason, when `n_ranked` is 0 |

`end_to_end_hits@K` for K up to the page's Top-K is exactly the fraction of cases in which a
clinician would see the truth on the page. Metric code sits beside `_authoritative` in
`src/evaluation/measurement.py` and uses `RankingMetrics.compute_from_ranks` for the reached set.

**The comparison, in the same run, on the same embeddings and cases.** The pipeline's cached
embeddings are already in memory, so Mode D computes, with Mode C's own primitives
(`masked_mean_pool`, `cosine_score_matrix`, `canonical_ranking`), for every case:

- **Mode C's rank** of the truth over every disease (in process);
- **the path-gated cosine rank**: the served candidates only, ordered by raw cosine with Mode C's
  tie rule. This is the README's intermediate step, as a derived reading with no ledger record of
  its own. C → path-gated isolates the candidate gate; path-gated → D isolates the served score
  (the (cos+1)/2 rescale in float32 and the SP mixture).

The report's `comparison` block holds the three families side by side on `n_cases`, the paired
counts (truth ranked better, equal, worse; D not reached while C ranked it in the top 10, top 100),
and the rank-difference distribution's quantiles. **When a kept Mode C report exists for the same
checkpoint digest and cohort-version manifest digest**, Mode D reads its `ranks.json` by the
digest the report names and states how many in-process Mode C ranks equal the kept ones (a
cross-check that both runs measured the same model and data; a difference is stated, not fatal).

**Labels, fixed text.** Mode C: "Disease ranking by the model, approved scorer policy (Mode C)"
(unchanged). Mode D: "Served ranking today (Mode D): diseases reached by path discovery, scored as
the Diagnosis page scores them (`<scoring mode>`); not the approved scorer policy." The comparison
block opens with: "Mode C and Mode D are different ranking policies: Mode C ranks every disease by
raw cosine; Mode D measures what the Diagnosis page serves today." No surface shows one figure
without its mode.

---

## 6. Report, ledger record, usage history

**No manifest field is added or removed** (`build_record` refuses a manifest whose field set
differs, which would turn every kept Mode C report into "cannot be checked"). Mode D gives the
existing fields its own values:

| Field | Mode D value |
|---|---|
| `mode` | `"D"` |
| `candidate_construction` | `"served-path-discovery/v1:settings=<sha256 of the served settings, 12 hex>"` |
| `score_semantics` | `"served:<scoring mode>:eta=<η>:sp=<build_id or none>"` |
| `canonical_tie_policy_version` | `"served-stable-discovery-order/v1"` |
| `batch_size` | `1` (one case per call) |
| `metric_schema_version` | `"D1"` |
| the regime fields, `model_construction` | as observed, as Mode C records them |

Other top-level report keys are ignored by `build_record`, so Mode D's extra blocks change no
schema: `served` (the served identity whole), `outcomes` (the §4 counts), `import_counts` and
`comparison` (§5). `authoritative_metrics` holds the end-to-end family plus `coverage` and the
reached family; `n_ranked` is the number of `ranked` cases. `ranks.json` rows are
`{sample_id, ground_truth, outcome, served_rank, n_served_candidates, mode_c_rank,
path_gated_rank}`, in cohort order, and the report names the file by digest as Mode C's does.

**Ledger and ordering need no change of rule.** `mode` and the other semantic values are in the
D5 key and in D4's scoring group, so a Mode D record is never compared with a Mode C record for
disagreement and is listed as a different kind of scoring, never interleaved.
`ORDERABLE_METRICS` stays Mode C's: Mode D records are shown, not ordered.

**Usage history.** `record_test_run` gains a `mode` field (an added key; `_read_entries` reads any
JSON object). Entries written before it read as Mode C, the only mode that wrote any.

**Labels by record.** The Test page's Result, the Models page's records and the API's record view
take the label from each record's `scoring.mode` instead of printing Mode C's for all, and the
pre-B-1 note appears beside Mode C records only.

---

## 7. Where the code goes, and what goes away

| Change | Where |
|---|---|
| `rank()` split; `bound_inputs()`; `embedding_regime` | `src/inference/pipeline.py` |
| `run_mode_d`, `ServedModeResult`, the §4 outcomes, the §5 metrics and comparison | `src/evaluation/measurement.py`, beside `run_mode_c` (it imports the pipeline: evaluation sits above inference) |
| `--mode {C,D}` (default C); Mode D's manifest values; its refusals | `scripts/measure_scorer.py`, sharing every existing check, the GPU lock, publication and registration |
| Mode labels | `src/evaluation/caveats.py`; Test page, Models page, `routes/evaluations.py` descriptions |
| `mode` in the usage history | `src/evaluation/report_store.py` |
| A Mode D probe on the demo cohort | `scripts/probe_deployment.py` (phase M, beside T1–T6) |
| Job runner and a Test-page start for Mode D | not in this package: the institutional run is an operator's CLI run; the page gains it with the Test/Models UI batch if wanted |

**Not touched:** the README's list above. `canonical_ranking` stays where it is; its move into
`src/inference` is B-1's (`../results-review/SPEC_1_RESULTS_REVIEW.md` §2.2), not Mode D's.

**Old entries:** none removed by Mode D. `scripts/test_gnn_inference.py`, which pins
`confidence_score == gnn_score` and is wrong in `gnn_plus_shortest_path`, goes with the result
union (BACKLOG §0, Batch 4).

---

## 8. Tests and acceptance

- **Serving unchanged:** before/after golden output of `run` over the demo graph in
  `gnn_plus_shortest_path`, `gnn_only` and the fallback, a tie case included, byte-equal.
- **Mode C unchanged:** a fixture cohort's Mode C report, metrics and ranks equal before and after;
  every kept report fixture still builds a ledger record.
- **Mode D:** outcome counts add up; a truth not reached, a refused case (more than 100 distinct
  terms) and a case with no candidates each counted as designed; metrics checked by hand on a small
  fixture; the in-process Mode C ranks equal `run_mode_c`'s on CPU for the same checkpoint and
  cohort; a Mode D record keyed and grouped apart from Mode C's and never ordered; refusals of the
  fallback, a generated split and a binding mismatch, before ranking.
- **Labels:** each surface shows the record's own mode; the comparison block's fixed text present.
- **GB10:** one short targeted run, Mode C then Mode D on a registered synthetic cohort (the M5
  close-out's `m5-closeout-demo-01` or a new one), minutes, no training. Earlier acceptances are not
  rerun.

**Cost.** Mode D ranks one case per call through the served path (per-candidate pooling, BFS per
phenotype). Its duration on the institutional cohort is measured on a pilot before the run is
scheduled; the served path is not optimised for it, since an optimisation could move a served
value.

---

## 9. Open, and whose

- **Institution, with the author (BACKLOG 8a, 8b):** the statistical protocol (pairing, strata,
  intervals) and the institutional run, after item 11's holdout protocol and item 6's authoritative
  checkpoint. Mode D's report is 8a's output contract on the engineering side.
- **Institution:** whether a bounded per-candidate component artifact may be persisted (policy
  rev 4), for a later cascade or tie-break analysis; Mode D persists none (its `ranks.json` holds
  the truth's ranks only).
- **Owner:** whether a Gate 1 revision is ever proposed; without one the gate stands (policy §2.1).
