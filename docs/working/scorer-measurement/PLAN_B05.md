# B-0.5 — minimal Mode D: the served ranking, measured

**Status:** designed in Batch 2 (2026-10-26; the reviewer passed `a5652c1..e61253e`) and
**implemented in Batch 3** (2026-10-26, for one review; §11 says what was built where), and
**revised after an internal review** of that implementation (§12: one P2 and nineteen P3, each
fixed; every code fix pinned by a test, and the fixes themselves verified once more, §12.1). The reviewer passed it (`e61253e..1f51fb2`, one P3 fixed at `be0ee1a`, §12.2), and the short GB10 run of §8 passed at `be0ee1a` (2026-10-27, §12.3). It resolves the design
problem `PLAN_B03.md` §4 recorded. Mode D is the input to Policy Gate 1's comparison, not B-1:
building it changes no ranking anyone is served. The statistical protocol for the institutional
run stays BACKLOG 8a's, and the run 8b's.

**Authority:** `docs/DISEASE_SCORER_POLICY.md` (statements 1–7, §2.1, §6 gate 1). **Constraints
carried from this folder's README:** Mode D adds its own values, not a manifest structure; it does
not revive Modes A or B and is not a second evaluator; and `ModeResult`, `canonical_ranking`,
`ranks_of_truth`, `encode_full_graph`, `build_shepherd_model`, the manifest's authoritative fields,
the digests, the CUDA gate and `src.kg.storage.file_storage` are not edited. `run_mode_c` gains one
optional read-only observer and nothing else (§5, with its reason).

**Revision:** the reviewer's review of `7b0162f` (2026-10-26) found four P2, two of them here
(P2-1 the case sets and their names, §4; P2-2 a second Mode C inside Mode D and P2-3 the
attribution of the intermediate step, §5), plus two clarifications (the golden comparison's fields,
§8; `metric_schema_version`'s type, §6). This version answers them; the first version is in git
history at `7b0162f`. The reviewer's second review (`7b0162f..a5652c1`) found that G2 took its
ties from the served order, which the served score had already decided. `rank()` now keeps each
candidate's `discovery_ordinal` (§2, §5, §8), and §10 records the owner's boundary for retiring
Mode D later.

---

## 1. What Mode D answers, and what it does not

| | Mode C (implemented) | Mode D (this design) |
|---|---|---|
| Question | how well does the model rank the truth among **every** disease, by the approved scorer (raw cosine)? | how well does **what the Diagnosis page serves today** rank the truth? |
| Candidates | every disease row | the diseases path discovery reaches (`_find_all_paths`, BFS and the direct finder, with their order-dependent caps) |
| Score | raw cosine of the masked-mean pool | the served score of the deployment's scoring mode: `gnn_plus_shortest_path` (0.7·((cos+1)/2) + 0.3·SP) or `gnn_only` ((cos+1)/2) |
| Ties | lower global disease id | the served order: a stable sort, so ties keep discovery order (`discovery_ordinal`, §2) |
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
- **Each entry also carries its `discovery_ordinal`**: its 0-based position in the candidate list
  `_find_all_paths` built, before the served-score sort. It is recorded where the sort happens,
  in the same candidate construction. It is internal metadata of `ServedRanking`, not a field of
  the API or the result union. Nothing re-discovers candidates, and nothing infers the ordinal from
  the served order: once the served score has sorted the list, the original order cannot be
  recovered from it.
- `run(patient_input, top_k, include_explanations)` becomes `rank()` followed by what it does today
  after the sort: the `top_k` slice and rank numbering, matching phenotypes, explanations, evidence
  and the summary. **Serving's output does not change**; a golden test over the demo graph in each
  scoring mode, ties included, pins that (§8).
- `top_k` never changes the order, only where the list is cut. So Mode D needs no `top_k`: a served
  rank `r` is what a clinician sees whenever the page's Top-K is at least `r`.

Mode D calls `rank()` only. It runs no explanation, evidence or summary step. Of each
`ServedRanking` entry it reads the order, the disease, the served score and the
`discovery_ordinal`, nothing else, so the
result union that batch 4 builds (`../results-review/SPEC_1_RESULTS_REVIEW.md` §2.1), which
renames and re-types the other fields, does not reach Mode D. *(As built in Batch 4, 2026-10-27:
the served score's attribute is `served_score`, Mode D's one change; a result the served path
refuses (`MODEL_TERM_UNAVAILABLE`, `NON_FINITE_SCORE`) stops the run, with no report kept.)*

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
| SP table | recorded by its `build_id`, which the loader pairs with the table's sidecar, and the sidecar binds the table to the graph whose `kg` digest the manifest records; the tensor's bytes are not hashed (`src/inference/sp_artifact.py`, by decision), so a table built again has a new `build_id` and a new `score_semantics`. Whichever the workspace holds is what the deployment serves, so Mode D measures `gnn_plus_shortest_path` with a table and `gnn_only` without |
| Settings in effect | the served identity's `settings` (η, discovery settings, normalisation version) recorded whole; folded into the manifest's semantic values (§6) so a change of settings is a different measurement |
| Numerical regime | the pipeline's embedding forward observes and exposes its autocast regime (`embedding_regime`: enabled and the autocast dtype); the manifest's `amp_*` fields must equal it, as Mode C's `assert_manifest_describes_regime` requires of `encode_full_graph`. The manifest's `dtype` is the process's default dtype, recorded as Mode C records it: the weights are float32, so the embeddings are float32 unless autocast is on, which the check covers |

**Refusals, before ranking:** a scoring mode of `path_reasoning_fallback` (no model); a generated
split (`--split`: Mode D measures a registered cohort version only, the form Gate 1's report needs);
any alignment check above failing; a cohort version mapped under another phenotype normalisation
version than the served one (the two sides would not score the same terms); a `--batch-size` that is
not a positive integer, refused at the command line. During the pass, a case whose served validation
uses other terms than the version mapped refuses the run (the premise below would not hold).

**Each case's input.** `samples.json` holds graph indices. Mode D turns them into the served input
form through the graph's own reverse mapping (`kg.get_reverse_node_mapping`), the mapping the
importer used in the other direction, so the pipeline's validation sees exactly the terms the
importer mapped. No source strings are needed, and none are read.

---

## 4. Per-case outcomes and the three case sets

*(Revised 2026-10-26 after the reviewer's review of `7b0162f`, P2-1: the first version called a
figure over the import-eligible cases "end to end".)*

**Three sets, never merged under one name:**

| Set | What it holds | Where it is recorded |
|---|---|---|
| **E — eligible beforehand, with a reliable truth** | the cases a protocol admits before any result is seen, each with a truth that can be determined | **not recorded today.** A cohort version records `source_cases` (what its source file held) and its import exclusions by category; whether every source case was eligible beforehand with a reliable truth is a property of the source and the protocol (BACKLOG 8a), which no version states |
| **I — import-eligible** | `counts.ranked_eligible`: the source cases the importer mapped to the graph | the version (`counts`), with `ground_truth_absent` and `phenotype_unmappable` counted separately |
| **R — reached by Mode D** | the cases of I whose truth is among the served candidates | Mode D's report (`outcomes`) |

A case excluded at import may be a case the system cannot handle (the graph lacks a known disease;
no term maps), or a case that was never a fair test (a duplicate, an unreliable truth, an ineligible
case). The version cannot tell which, so Mode D does not count import exclusions as failures, and
does not count them as absent from the population either. **Neither mode reports an end-to-end
figure over E** until 8a's protocol defines E and a version records it (a registry change, not in
batch 3). The report states `source_cases` and each import exclusion count beside its figures,
with the reason the end-to-end figure is absent.

**Each case of I gets exactly one Mode D outcome:**

| Outcome | Meaning | Served rank |
|---|---|---|
| `ranked` | the truth is among the served candidates | its 1-based position in the full served order |
| `not_reached` | discovery did not reach the truth (including a case with no candidates at all, counted separately as `no_candidates`) | none |
| `refused` | the served validation refused the input; today only "more than `max_phenotypes` (100) distinct terms", which the importer does not cap | none |

`n_import_eligible = n_ranked + n_not_reached + n_refused`, checked. The word "absent" is not used:
Mode C's `n_ground_truth_absent` means the truth was missing from every disease row and is fatal
there; a truth Mode D's discovery does not reach is a result, not a defect.

**The C/D comparison is paired on I**: both modes rank exactly the cases of I, and every comparison
block names I as its case set.

---

## 5. Metrics, the comparison and its labels

*(Revised 2026-10-26, the reviewer's P2-2 and P2-3: the C side is `run_mode_c` itself, and the
intermediate readings change one thing at a time.)*

**Two figure families, both from the one report:**

| Family | Denominator | Metrics |
|---|---|---|
| Post-import | `n_import_eligible` (set I) | `post_import_mrr` (1/r for `ranked`, 0 for `not_reached` and `refused`), `post_import_hits@K` for K = 1, 5, 10, 20, 50, 100, and `coverage` = `n_ranked / n_import_eligible` |
| Among reached | `n_ranked` (set R) | `reached_mrr`, `reached_mean_rank`, `reached_hits@K`; omitted, with the reason, when `n_ranked` is 0 |

`post_import_hits@K` for K up to the page's Top-K is the fraction of import-eligible cases in which
a clinician would see the truth on the page. Metric code sits in `src/evaluation/mode_d.py` (§7)
and uses `RankingMetrics.compute_from_ranks` for the reached set; Mode C's `_authoritative` is not
changed.

**The C side is the formal Mode C, called once, on the same embeddings.** Mode D does not assemble a
second Mode C from its primitives. It wraps the pipeline's cached embeddings, with the regime the
pipeline observed (`embedding_regime`, §3), as the `EncodedGraph` that `run_mode_c` already takes.
`scripts/measure_scorer.py --mode D` builds the C side's manifest for that regime, device and
`batch_size` with its existing `build_manifest`, as for any Mode C run, and hands it to
`run_mode_d`. `run_mode_d` calls `run_mode_c` with the cohort's samples. The result is a
`ModeResult` with per-case canonical ranks, under Mode C's own validation, padding, chunking and
integrity checks. Nothing is reloaded: the model forward ran once, in the pipeline. The device and
`batch_size` the C side ran with are recorded as they were; they are execution conditions of this
run, and Mode C's numbers can move with them at floating-point level.

**One small change to `run_mode_c`, with its reason.** The intermediate readings below need the
scores the formal C side computed, not a second computation of them. `run_mode_c` gains one
optional, read-only observer, called with each chunk's case ids and score matrix. It is `None` in
every Mode C run, and changes no value, order or report: Mode C's report is compared before and
after, excluding the volatile fields (§8). Mode D's observer keeps, per case, only the scores of
that case's served candidates, so memory stays bounded by the served list. This edits a function on
the README's "not touched" list by decision, and the README says so.

**The intermediate readings: derived, one change at a time.** Each is computed from the scores the C
side produced and the served candidates `rank()` returned. None is a mode, a report entry point or a
ledger record.

| Step | Candidates | Score | Ties | What changes from the step before |
|---|---|---|---|---|
| C (formal) | every disease | raw cosine | lower global id | — |
| G1 | the served candidates | raw cosine | lower global id | the candidate set only |
| G2 | the served candidates | raw cosine | discovery order: order by (raw cosine descending, `discovery_ordinal` ascending). Never a re-sort of the order `rank()` returned, which the served score already decided | the tie policy only |
| D (formal) | the served candidates | the served score | discovery order (`discovery_ordinal`) | the score only: the (cos+1)/2 rescale rounded in float32, and the SP mixture where a table is served |

C→D is the joint difference of the three, and is labelled so. The report gives each step's rank of
the truth per case and the paired counts per step: truth ranked better, equal or worse; D not
reached while C ranked it in the top 10, or the top 100, and D refused while C ranked it, counted
apart (a refusal is not a case discovery failed to reach). The rank differences are after minus
before (negative: the truth moved up), their quantiles by nearest rank. **G1 and G2 see a tie only
where the C side's cosines are exactly equal**, as computed at its batch size (recorded in the
comparison's `mode_c.run_conditions`); the report says so beside the steps. **A worked tie case, also a pass condition
(§8):** two served candidates in discovery order [9, 2], both with cosine 0.5. G1 orders them [2, 9]
(lower id first), G2 [9, 2] (discovery order), and `gnn_only` serves [9, 2]: both score 0.75. The
change from G1 to D is the tie policy, all of it at G1→G2; G2→D changes nothing. The score
rescale and SP are not blamed for it.

**A second case, where SP breaks a cosine tie** (also a pass condition; the reviewer's
counterexample to the first version of this table). Discovery order [9, 2], cosine 0.5 for both,
SP score 0.1 for disease 9 and 0.9 for disease 2, η = 0.7. The served scores are 0.555 for 9 and
0.795 for 2. G1 orders them [2, 9], G2 [9, 2] (the cosine tie kept in discovery order), and D
serves [2, 9]. With the truth 9, the ranks are 2 at G1, 1 at G2 and 2 at D: the tie step improves
it by one, and the score step (the SP mixture) worsens it by one. Had G2 re-sorted the served
order, it would read [2, 9], and the score's effect would be blamed on the tie step.

**A kept Mode C report** for the same checkpoint digest and cohort-version manifest digest is read by
the `ranks.json` digest the report names. The report states how many of its ranks equal the C
side's. This is a cross-check under possibly different execution conditions (device, `batch_size`,
regime), each named. A difference is stated, never fatal, and never read as an error in either run.

**Labels, fixed text.**
- **Mode C:** "Disease ranking by the model, approved scorer policy (Mode C)" (unchanged).
- **Mode D:** "Served ranking today (Mode D): diseases reached by path discovery, scored as the
  Diagnosis page scores them (`<scoring mode>`, η = `<the identity's η>`); not the approved scorer
  policy."
- **The comparison block** opens with: "Mode C and Mode D are different ranking policies: Mode C
  ranks every disease by raw cosine; Mode D measures what the Diagnosis page serves today. Both are
  measured on the same import-eligible cases; the cases excluded at import are counted beside them."

No surface shows one figure without its mode and its case set.

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
| `metric_schema_version` | `2`: the next value of the one integer sequence (`METRIC_SCHEMA_VERSION` is 1 for Mode C's family). The field's type stays `int`; Mode C keeps 1 |
| the regime fields, `model_construction` | as observed, as Mode C records them |

Other top-level report keys are ignored by `build_record`, so Mode D's extra blocks change no
schema: `served` (the served identity whole), `outcomes` (the §4 counts), `import_counts`
(`source_cases` and each exclusion category, with the reason no end-to-end figure is given) and
`comparison` (§5: the C side's run conditions, the steps and their paired counts).
`authoritative_metrics` holds the post-import family plus `coverage` and the reached family;
`n_ranked` is the number of `ranked` cases. `ranks.json` rows are `{sample_id, ground_truth,
outcome, served_rank, n_served_candidates, c_rank, g1_rank, g2_rank}`, in cohort order, and the
report names the file by digest as Mode C's does. `MeasurementManifest.__post_init__` accepts Mode
D's values; no field is added.

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
| `run_mode_d`, `ServedModeResult`, the §4 outcomes, the §5 metrics, steps and comparison | one new module, `src/evaluation/mode_d.py`, which imports `measurement` and the pipeline (evaluation sits above inference). Everything specific to Mode D sits there, so retiring it is one module, one CLI option and one test file (§10) |
| The optional read-only score observer | `run_mode_c`, `src/evaluation/measurement.py` (§5). It must not modify what it is given, and keeps only the served candidates' values, never whole score blocks |
| `--mode {C,D}` (default C); Mode D's manifest values; its refusals | `scripts/measure_scorer.py`, sharing every existing check, the GPU lock, publication and registration |
| Mode labels | `src/evaluation/caveats.py`; Test page, Models page, `routes/evaluations.py` descriptions |
| `mode` in the usage history | `src/evaluation/report_store.py` |
| A Mode D probe on the demo cohort | `scripts/probe_deployment.py` (phase M, beside T1–T6) |
| Job runner and a Test-page start for Mode D | not in this package: the institutional run is an operator's CLI run; the page gains it with the Test/Models UI batch if wanted |

**Not touched:** the README's list above, except `run_mode_c`'s observer. `canonical_ranking` stays
where it is; its move into `src/inference` is B-1's (`../results-review/SPEC_1_RESULTS_REVIEW.md`
§2.2), not Mode D's.

**Old entries:** none removed by Mode D. `scripts/test_gnn_inference.py`, which pins
`confidence_score == gnn_score` and is wrong in `gnn_plus_shortest_path`, goes with the result
union (BACKLOG §0, Batch 4; removed 2026-10-27).

---

## 8. Tests and acceptance

**What "unchanged" compares.** A golden test compares what is promised not to change. For serving
that is every candidate's disease, rank and score fields, the candidate order, the warnings and the
phenotype account, and the served identity's non-volatile parts. For Mode C it is the metrics,
`n_ranked`, the per-case ranks, the digests and the manifest's semantic fields. Timestamps
(`timestamp`, `inference_time_ms`, a report's creation time), the software revision and other
run metadata are left out of the comparison, never forged to make whole files byte-equal.

- **Serving unchanged:** golden output of `run` over the demo graph in `gnn_plus_shortest_path`,
  `gnn_only` and the fallback, a tie case included, equal before and after the split.
- **Mode C unchanged:** a fixture cohort's Mode C report, metrics and ranks equal before and after
  the observer is added; every kept report fixture still builds a ledger record.
- **Mode D:** outcome counts add up; a truth not reached, a refused case (more than 100 distinct
  terms) and a case with no candidates each counted as designed; metrics checked by hand on a small
  fixture; a Mode D record keyed and grouped apart from Mode C's and never ordered; refusals of the
  fallback, a generated split and a binding mismatch, before ranking.
- **The review's counterexamples, as pass conditions:**
  - *Case sets (P2-1).* A source of 100 cases, 20 excluded at import as `phenotype_unmappable`, the
    other 80 ranked first: `post_import_mrr` is 1.0 and labelled post-import; `import_counts` shows
    100 source cases and 20 excluded; no figure is named or labelled end to end, and the report says
    why.
  - *One Mode C (P2-2).* The C side is `run_mode_c`'s own `ModeResult`: its ranks equal a direct
    `run_mode_c` call on the same `EncodedGraph`, samples, device and `batch_size`, and Mode D's code
    calls no scoring primitive itself (a test asserts `run_mode_c` is the one call that scores).
  - *Ties (P2-3).* The two [9, 2] cases of §5:
    - in `gnn_only`, G1 [2, 9], G2 [9, 2], D [9, 2]: the change is the tie step's, none the score
      step's;
    - with SP breaking the cosine tie, G1 [2, 9], G2 [9, 2], D [2, 9], so the truth 9 ranks 2, 1
      and 2: G2 is computed from `discovery_ordinal`, never from the served order.
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

---

## 10. Retiring Mode D, when its comparison is done (the owner, 2026-10-26)

Mode D exists to compare the served ranking with the approved one (gate 1). It is built so it can
be removed cleanly afterwards. This is a boundary, not a retirement plan.

1. **It stays while it is needed.** Mode D is not removed before the comparison and the gate 1
   work that rests on it are done.
2. **Dependencies run one way.** Mode D calls `DiagnosisPipeline.rank()` and `run_mode_c`.
   Serving, the trainer, Mode C and B-1 never depend on Mode D's orchestration or its reports.
3. **What is Mode D's is kept together:**
   - its orchestration, metrics and derived readings: `src/evaluation/mode_d.py`;
   - its CLI option: `--mode D`;
   - its tests: one file.

   It reuses the existing loaders, registration, GPU lock and report store, and adds no copies of
   them. `rank()` is a serving capability, not Mode D's, and stays when Mode D goes.
4. **Removing it keeps the history.** When Mode D's execution is removed, its kept reports,
   summaries and conclusions stay. So does the read-only identification its records need (the
   label by `scoring.mode`, the manifest values of §6). No old executor, fallback or command alias
   is kept for those files. The `run_mode_c` observer goes in the same change if nothing else calls
   it.
5. **Two retirements, counted apart:**
   - **Mode D's measurement** goes when its comparison is done.
   - **The legacy mixture score** (`served_mixture/v1` and its helpers) goes only when B-1 has
     taken over and no live caller needs it.

   Neither removes the SP capability that the paper's other tasks need: the candidate-gene task's
   SP term, and SP as an analysis view.
6. **The shared correctness tests stay.** The removal is one explicit change, made with three
   things in hand: the finished comparison, the switch's state, and an inventory of callers. No
   plugin or retirement framework is built for it.

---

## 11. As implemented (Batch 3, 2026-10-26)

| Design | Code | Pinned by |
|---|---|---|
| `rank()` split from `run()`, `discovery_ordinal`, `bound_inputs()`, `embedding_regime` (§2, §3) | `src/inference/pipeline.py` (`ServedRanking`, `BoundInputs`); the regime's one home is `src/utils/run_environment.py` (`AutocastRegime`, `observe_autocast_regime`), which Mode C's encoder uses too | `tests/unit/test_served_rank.py`, with the golden output captured before the split (`tests/fixtures/served_rank_golden.json`: the three scoring modes, the fallback's exact ties) |
| The service's builder importable by measurement (§3) | `build_served_pipeline` in `src/inference/pipeline.py`; the API's `build_pipeline` keeps its settings and state handling and calls it | the same file: the API's build and this builder give one served identity |
| The observer (§5) | `run_mode_c(..., on_scores=None)`, `src/evaluation/measurement.py` | `tests/unit/test_mode_d.py`: a run with an observer reading every score equals one without |
| Outcomes, case sets, metrics, the steps C → G1 → G2 → D, the comparison, the kept-report cross-check (§4, §5) | `src/evaluation/mode_d.py` | `tests/unit/test_mode_d.py` (the three counterexamples, each kind of case, the C side equal to a direct `run_mode_c`) |
| `--mode {C,D}`, Mode D's manifest values, the refusals (§3, §6) | `scripts/measure_scorer.py`, through the existing publication, R10, registration and GPU lock | `tests/unit/test_mode_d.py` (through `main`: the report's blocks, the field set, the record keyed apart, the refusals before ranking) |
| Labels by record, `mode` in the usage history (§5, §6) | `src/evaluation/caveats.py` (`mode_label`, `served_scoring`, `mode_note`, `metric_text`); the Test page's Result, the Models page's records and usage history; `RecordView` gains `mode` and `score_semantics`; `record_test_run(..., mode)` and `entry_mode` in `src/evaluation/report_store.py` | `tests/unit/test_mode_d.py` (each surface over a real C and D record); `tests/unit/test_loaded_model_evaluations_api.py` (the view's fields) |
| The probe (§7) | `scripts/probe_deployment.py`, phase M's **T7**, after T6: Mode D with `--record` on T1's version | `tests/unit/test_probe_m2_phase.py` |

**Two places where the code enforces §6's rule rather than leaving it to the caller.** §6 says
Mode D records are shown, never ordered. The Models page offers only Mode C records' scoring groups
for ordering (`scoring_groups`), and `order_by_test_score` itself does not order a scoring group
whose records are not Mode C (it answers "not offered" with the reason), so no caller can order a
Mode D group by asking for it. No ordering rule changed for Mode C.

**Not built, as §7 says:** a job-runner or Test-page start for Mode D (the institutional run is an
operator's CLI run). The Test and Models pages already show a Mode D record by its own label, so
that start, if wanted, is the UI batch's.

**What removing Mode D touches beyond §10.3's three** (found by the internal review, §12): probe
T7 and the run list W1 is given (`scripts/probe_deployment.py`) with T7's assertions in
`tests/unit/test_probe_m2_phase.py`; the provenance pin's entry for `mode_d.served_semantics`
(`tests/unit/test_provenance_pins.py`, which fails once an entry is stale); and the import-linter
contract `mode-d-one-way`, which enforces §10.2. The labels by record (`caveats`, the pages) stay,
as §10.4 says.

---

## 12. The internal review of `71bdad6` and its fixes (2026-10-26)

Before the reviewer's review, an internal read-only review (four reviewers, one per dimension, each
finding retried by an independent verifier, then a gap critic) read `e61253e..71bdad6`. It found no
P1, one P2 and nineteen P3 confirmed (one refuted: the candidate-construction digest, since a code
change is already another `software_revision`). Each is fixed here. **Every code fix is pinned by a
test that fails without it**, checked by reverting the fix in a copy (§12.1); the two documentation
rows (marked —) and cli-3's Windows URL part, which this Linux host cannot run, have none. **What it
confirmed sound** is not repeated: serving unchanged across the split (28,512
runs compared field by field with the pre-split code), the served build's verification at least
Mode C's, the D5 comparison covering Mode D's metrics, `rank()` holding no shared state.

| Finding | Fix | Where | Pinned by |
|---|---|---|---|
| **ui_api-1 (P2)**: the Models page's mixed C/D table named its columns "MRR" and "Hits@10", dropping each figure's case set and Mode C's "(untruncated)" | one table per mode, each under its mode's label and note, its columns named in full (`metric_text`), each named for its mode for assistive technology, the next mode's label set apart by 16 px; the row mode lines removed | `models_page.py` (`records_table_html`, `_mode_table`, `_by_mode`, `_column_head`, `records_block_html`) | `test_mode_d.py` (each surface): the order label → columns per mode |
| ui_api-2: the Models detail's segment read "Recorded test results (Mode C)" over every record | "Recorded test results · N"; the module docstring and the Validation help text | `models_page.py`, `renderers.py` | `test_webui_models_page.py` (the detail) |
| ui_api-3: the Models page's Mode D label lacked §5's "(scoring mode, η)" (the guides said it was there) | in the label when every Mode D record shown shares it, else on each record's line | `models_page.py` (`_uniform_semantics`); both guides | `test_mode_d.py` (one and two served settings) |
| ui_api-4: an unknown or unreadable mode got Mode C's coverage words; a usage entry with a malformed `mode` read as Mode C | neutral wording (the record's count, "what they are is not described for this mode"); `mode_label(None)` "Measurement mode not recorded", a blank mode included; `entry_mode` returns None for a key that is there and unreadable (still "C" when absent) | `caveats.py`, `models_page.py`, `test_page.py`, `report_store.py` | `test_webui_models_page.py`, `test_webui_test_page.py` (another or no mode) |
| ui_api-5: the Test page's report rows did not name their mode, and nothing said Start runs Mode C | each row's state line starts with its mode ("served ranking (Mode D): report kept · …", one separator); the fixed settings show "Mode: model ranking (Mode C)" and its help says Mode D is a CLI run | `test_page.py` (`_row_state`, `fixed_html`, `test-fixed`) | `test_mode_d.py`, `test_webui_test_page.py` |
| ui_api-6: Mode D's note read as if a Mode C row shown were its same-embeddings comparison | reworded, in shorter sentences: its report sets Mode C beside it; a Mode C record shown here is a separate run | `caveats.py` (`MODE_D_NOTE`) | `test_mode_d.py` (the note's wording) |
| gap: a version mapped under another phenotype normalisation, or a served input dropping a mapped term, would compare different inputs silently | refused before the served pass (versions differ) and during it (a case's used terms differ from the mapped ones) | `mode_d.py` (`run_mode_d`, `rank_served_cases`) | `test_mode_d.py` (two refusal tests) |
| cli-5, mode_d-4: `--batch-size` 0 or below failed only after the served pass, once with a message pointing at the cohort | positive integers only at the command line (both modes); `run_mode_d` checks the C side's manifest, mode and batch size before the served pass | `measure_scorer.py` (`_positive_int`), `mode_d.py` | `test_mode_d.py` (the command line; `run_mode_d` called directly), `test_measure_scorer_cli.py` |
| mode_d-1: refused cases were counted as "not reached while C ranked" | counted only for `not_reached`; `d_refused_while_c_ranked` beside it; in the paired steps a refused case is `ranked_then_refused`, never `ranked_then_not` (the candidate set did not lose it) | `mode_d.py` (`paired`, the comparison) | `test_mode_d.py` (a refused case; `paired`) |
| mode_d-2: a kept Mode C report whose `ranks` block is not an object ended every Mode D run with a traceback | listed as unreadable with the reason | `mode_d.py` (`kept_mode_c_reports`) | `test_mode_d.py` |
| mode_d-5: C→D was not labelled the joint difference | `what_changes["C->D"]` says so | `mode_d.py` | `test_mode_d.py` (the report's blocks) |
| mode_d-6: the quantiles were not nearest rank as named; the sign was unstated | nearest rank; `rank_difference_is` states after minus before, negative = moved up | `mode_d.py` (`_quantiles`, `paired`) | `test_mode_d.py` |
| mode_d-7: G1/G2's ties depend on the C side's float scores at its batch size, unstated | `steps.ties_are` says so; §5 too | `mode_d.py`; §5 | `test_mode_d.py` (the report's blocks) |
| mode_d-8, cli-4: §3 said the SP table is recorded "by build_id and digest" | §3 says what is recorded and why (build_id paired with the sidecar, bound to the graph; the tensor not hashed by decision) | this document | — |
| serving-2: the golden lacked §8's phenotype account and served identity | both added, captured from the pre-split code (`e61253e`, the same fixture; old and new identities byte-identical); in the golden the identity's digests are reduced to whether they are there, and the test checks each against the fixture's own files (the SP build id is random per build) | `tests/fixtures/served_rank_golden.json`, `test_served_rank.py` | the golden test |
| serving-3: §3 promised a dtype in `embedding_regime` and a manifest dtype check | §3 says what is built (the autocast regime; the manifest's dtype the process default, as Mode C) | this document | — |
| serving-1: `bound_inputs` said "serves no model" for a model supplied in memory | its own message for that state | `pipeline.py` | `test_served_rank.py` |
| cli-1: the ordering's Mode C-only guard refused a whole group when one record had no readable `scoring` ("mode None") | only a readable mode other than C refuses the group; a record whose mode cannot be read makes its own checkpoint unreadable, never ordered on it | `ordering.py` | `test_test_score_ordering.py` |
| cli-2: probe W1 depended on T7, so one Mode D failure failed W1-W11 | W1 lists exactly the runs M4's flow published (T2's, T7's when it published one), checks each kept and each whose probe passed listed as registered, and keeps what later probes need before its own checks | `probe_deployment.py` (`_m5_demo`, T7) | `test_probe_m2_phase.py` (a Mode D failure is T7's alone) |
| cli-3: BACKLOG 59's fix merged any source-tree copy in the checkout, and missed Windows file URLs | only exactly one copy, where the editable install puts the project (its `.pth`, else `src/`); `url2pathname` for the URL (not testable on this Linux host) | `probe_deployment.py` (`_editable_paths`, `_in_source_tree`) | `test_probe_m5_acceptance.py` (nested copy, copy elsewhere, `.pth`) |
| cli-6: the CLI's docstring was Mode C only, a doc comment had moved to the wrong constant, the outcome file did not name its mode | docstring and usage for Mode D; the comment back; `--outcome` gains `mode` (an added key) | `measure_scorer.py` | `test_mode_d.py` (the outcome file) |
| gaps: no progress during the long served pass; §10.2's one-way rule unenforced | a progress line about 20 times a pass, each after its case; the import-linter contract `mode-d-one-way`, a protected module only itself may import (the package's own __init__ and any later module included) | `mode_d.py`; `.import-linter.ini` | `test_mode_d.py`; `lint-imports` |

### 12.1 The fixes, verified (2026-10-26)

The fixes above were then verified read-only by four reviewers, one per group, and a cross-check of
the whole fix diff. Each reproduced the original scenario against the fixed tree and reverted each
fix in a copy to see its test fail. They found no regression and no change no finding asked for,
and thirteen P3 in the fixes themselves, each addressed here:

- the guides still described the Models segment as "Recorded test results (Mode C)", the Run
  panel's fixed settings without the mode, the `--outcome` file without `mode`, and Mode D's
  section without its new refusals and progress lines: all four updated, in both guides;
- texts past the WebUI skill's word rules (the Mode D note's and the fixed-settings help's long
  sentences, two separators in a report row, the bare field name `n_ranked`): reworded;
- the two per-mode tables shared one accessible name and the second label sat on the first table:
  each table named for its mode, the label set apart;
- refused cases were still counted as lost at the candidate-set step: kept apart in `paired`;
- `run_mode_d`'s checks before the served pass and the Mode D note's wording were pinned by no test:
  both pinned now;
- `run_mode_d`'s docstring claimed no refusal after the pass: narrowed;
- the last progress line left out the last case: logged after each case;
- the fix record claimed a failing test for every row: reworded (§12's lead);
- the golden pinned the identity's digests only by presence on a reason that was not true: each is
  checked against the fixture's files;
- the Mode D import contract listed today's modules only: a protected-module contract instead;
- the ordering took a record whose mode cannot be read for Mode C's, even as the chosen one: its
  checkpoint is unreadable;
- W1 no longer checked T7's run as registered: it does, when T7 passed;
- a blank mode string read as a mode: read as not recorded.

### 12.2 The reviewer's review of `1f51fb2` (2026-10-27)

The reviewer read `e61253e..1f51fb2` read-only: no P1 or P2, one P3. **The ordering built its set of
modes before leaving out the unreadable ones**, so a damaged `scoring.mode` of `[]` or `{}` (which
cannot be hashed) raised `TypeError` out of `order_by_test_score`, and the Models page, which catches
only `ValueError`, ordered no checkpoint of the group. A normally written ledger never holds such a
value. **Fixed at its root**: the mode is checked before it enters the set, so the damaged record
makes only its own checkpoint unreadable, as cli-1 decided. The set of the records' roles two steps
earlier had the same defect (from M5, before Batch 3): a damaged `cohort_role` raised the same
error. It is gathered by each role's text now, so a damaged role is named as another role and the
version is not ordered, as for any role but the one asked for. Nothing else changes: no new reader,
and no error caught more broadly. Pinned by `test_test_score_ordering.py` (`[]` and `{}` for the
mode and the role); without the fix exactly those four cases fail.

The reviewer then read `1f51fb2..be0ee1a` and passed it: the P3 closed, the role fix accepted as the
same defect's local guard (a damaged or other role still refuses the whole version, a damaged mode
only its own checkpoint), no new finding.

### 12.3 The short GB10 run, Mode C then Mode D (2026-10-27)

§8's one short run passed at `be0ee1a`, within the scope agreed, as the reviewer's review of the
evidence found. **With it, Batch 3's technical and synthetic functional acceptance, within the
agreed scope, is complete.** The record and the whole evidence package are in
[`../batch3-acceptance-2026-10-27/README.md`](../batch3-acceptance-2026-10-27/README.md). On the
synthetic cohort `m5-closeout-demo-01` (12 source cases, 2 excluded at import, 10 eligible) and one
checkpoint: both runs exited 0 on CUDA; Mode D's outcomes 7 ranked, 3 not reached, 0 refused, the
console's last progress line equal to them; its C side equal case by case to this run's Mode C report
and to the kept one of an earlier revision (10 of 10 each); the ledger keeps its record and adds the
C and the D record, under different scoring semantics. The workspace has no shortest-path table, so
Mode D ran in `gnn_only`, as the service serves it there; on these cases the difference from Mode C
comes from the candidate set alone (C→G1), the tie and score steps changing no rank. It is a
functional run: it shows nothing about ranking quality, the SP mixture on the GB10 or the Models
page, and gate 1 is not cleared.
