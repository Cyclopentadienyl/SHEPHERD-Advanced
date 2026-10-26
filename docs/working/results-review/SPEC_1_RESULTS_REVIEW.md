# Spec 1 — Results review: authority, SP policy, decomposition, surfaces

**Status:** draft, under review. **Normative.** Supersedes rev 6 §1–§8. §2.1 (the result union) and
§2.2 (the shared scoring boundary) were added as Batch 2's design on 2026-10-26, for review.
**Authority above this document:** `docs/DISEASE_SCORER_POLICY.md` — statements 4a/4b, conditions
C1–C10, the eager-SP / lazy-evidence schedule, and the three limits that replace `top_k`. Where a
choice here is constrained by a condition, the condition is cited; this document does not restate or
override them.

---

## 1. Requirements

| # | Requirement |
|---|---|
| R1 | Candidate list raised from 10 to **200+** |
| R2 | Paginated, clinician-selectable page size (10 / 20 / 50 / 100) |
| R3 | SP analysis surface supports **sorting** by SP |
| R4 | SP analysis surface supports **range filtering** by SP |
| R5 | SP filter **defaults off** (§3.6) |
| R6 | Per-candidate score decomposition, LIRICAL-style, is acceptable to these clinicians (§4) |
| R7 | Result storage — Spec 2 |
| R8 | Navigating to results and back must not feel like a chore (§5.1) |
| R9 | `selection_limit` stays an inference-time setting, so the clinician bounds their own wait |
| R10 | Multi-user is a possible future scenario, **not a current requirement** — Spec 4 §6 |

R6 borrows a presentation *idea* — each row explains itself. No LIRICAL code is used and the
mathematics differs entirely: LIRICAL composes likelihood ratios, §4 decomposes an inner product.

---

## 2. Data model — four separate types

A single envelope carrying canonical data, page state and lazily-populated fields would make C10 —
binding a view to an immutable result ID — **unenforceable**, not merely untidy.

| Type | Contents | Lifetime |
|---|---|---|
| `CanonicalDiagnosisResult` | candidates, canonical scores, canonical ranks, selected count, scorer provenance, KG and SP fingerprints, `result_id` | **immutable** once produced |
| `AnalysisViewState` | sort key and direction, active filters, page index, page size, hidden count, facet counts, actor and session, view-state version | transient, per view |
| `CandidateAnalysisRecord` | SP quantities (§3) and per-phenotype contributions (§4), with provenance | produced eagerly with the result; versioned separately |
| `EvidenceRequest` / `EvidenceResponse` | Spec 3 | lazy subresource |

- **Page size and offset never appear in the canonical result.** They describe a view, not an inference.
- **Evidence never mutates the canonical result.** A stale or failed evidence fetch degrades the
  evidence panel, never the ranking.
- **View state is never reapplied on load.** A reopened snapshot opens in the canonical view (C2).
  The view state accompanying a *clinician action* is retained as audit metadata on that action (C5,
  Spec 3 §5).

### 2.1 The result union (Batch 2 design, 2026-10-26)

**Status:** design for review. Today's kinds are built in BACKLOG §0 batch 4, `gnn_ranking/v1` with
B-1 in batch 5. **Authority:** policy §7: one versioned discriminated union with required and
forbidden fields per variant, never a flat envelope; statement 7; C9.

**The discriminator is `result_kind`, on the result, never on a candidate.** Every candidate of a
result has the same score semantics. The kind is derived from what produced the result, never
chosen by a caller, by one mapping function in `src/inference`:

| `result_kind` | Produced when | Served identity's `scoring_mode` |
|---|---|---|
| `gnn_ranking/v1` | B-1's switch is on and a model is served | `gnn_raw_cosine`, added by B-1. The SP table may be `verified` or `absent`, since SP no longer ranks; `ServedIdentity`'s coupling "SP verified ⇔ `gnn_plus_shortest_path`" is relaxed for this mode only |
| `served_mixture/v1` | a model is served and the switch is off: today | `gnn_plus_shortest_path` or `gnn_only` |
| `path_reasoning/v1` | no model source is configured | `path_reasoning_fallback` |
| `demo/v1` | no pipeline was ever configured (the API's demo answer) | none: the identity is null, as today |

`scoring_mode` is not the discriminator itself: it describes the configuration, while the kind says
what the numbers in the candidates mean. Under B-1, a deployment with a table serves a GNN ranking
with SP beside it, not in it.

**Every kind carries:**
- **On the result:** `result_kind`, `result_schema_version = 2`, `served_identity`,
  `phenotype_input` (the account), `warnings`, and `candidates` in rank order.
- **On each candidate:** `rank` (exactly 1..N), `disease_id`, `disease_name` and `matching_phenotypes`.

| Candidate field | `gnn_ranking/v1` | `served_mixture/v1` | `path_reasoning/v1` | `demo/v1` |
|---|---|---|---|---|
| `ranking_score`: raw cosine, finite, in [−1, 1] | required | forbidden | forbidden | forbidden |
| `served_score`: the value ranked by, η·model term + (1−η)·SP, or the model term alone in `gnn_only` | forbidden | required | forbidden | forbidden |
| `model_term`: (cos+1)/2 | forbidden | required | forbidden | forbidden |
| `sp_term`: an SP status record (§3.3), with numbers only on a numeric status | forbidden: SP is in `candidate_analysis` | required | forbidden: SP is never loaded without a model | forbidden |
| `path_score`: the weighted path aggregate, Σ sᵢ·0.8ⁱ capped at 1 | forbidden | forbidden | required | forbidden |
| `evidence_package`, `evidence_label`, `explanation`, `reasoning_paths` (built eagerly from discovery paths) | forbidden: evidence is the lazy subresource (Spec 3 §7) | as today | as today | forbidden |

**Result-level fields by kind:**
- **`gnn_ranking/v1`:**
  - `result_id` (C10);
  - `universe_size`, the number of diseases scored;
  - `selected_count`;
  - `tie_policy = "lower-global-id/v1"`;
  - `candidate_analysis`, which is either one `CandidateAnalysisRecord` per selected candidate (`sp-analysis/1`, §3) or `NO_TABLE`, stated once.
- **`served_mixture/v1`:** `eta`, `candidates_reached` (how many diseases discovery reached before the
  cut) and `tie_policy = "served-stable-discovery-order/v1"`.
- **`path_reasoning/v1`:** `path_score_aggregation = "weighted_sum/v1"` and the same tie policy as the mixture.
- **`demo/v1`:** no score of any kind. The demo answer keeps its purpose, a service tried with nothing
  configured, and loses its invented numbers.

**Removed from the response, with no compatibility alias** (an alias would keep the defect):

| Removed field | Why |
|---|---|
| `confidence_score` | two quantities in one field |
| `gnn_score` | replaced by `model_term`, present only where a model ranked |
| a bare `sp_score` | replaced by `sp_term` or `candidate_analysis` |
| `reasoning_score` outside `path_reasoning/v1` | today it is computed in every mode and ranks nothing there |
| `confidence_label` | renamed `evidence_label`: it grades evidence, not confidence |

**Unavailable is never zero.**
- **SP.** `sp_term` and `candidate_analysis` carry §3.3's statuses.
  - In `served_mixture/v1`, a non-numeric status is still mixed as 0, so the legacy ranking does not
    change, and the candidate says so (`sp_term.mixed_as = 0.0`, shown as "mixed as 0 (legacy)").
  - Of the non-numeric statuses, only `NO_TABLE` (in `gnn_only`, where nothing is mixed) is expected
    on a verified workspace: validation refuses an empty phenotype set, and the graph and tensors
    are bound by digest (M2.4). That binding identifies the files; it does not prove every runtime
    lookup succeeds, so a test forces each other status and checks its record.
- **The model term fails closed.** Today's five paths that return GNN `0.0` become a refusal of the
  result (`MODEL_TERM_UNAVAILABLE`, naming the node), never a number:
  - no embeddings;
  - no phenotype index;
  - no phenotype table;
  - a disease with no row;
  - no disease table.

  The two index clamps (disease row, phenotype row) also become refusals. Two claims are kept
  apart:
  - **On valid input, nothing changes.** The golden fixtures, valid inputs on a verified workspace,
    give the same values and order before and after (the golden test, which compares the fields
    Mode D's plan names: `../scorer-measurement/PLAN_B05.md` §8).
  - **Invalid states are now refused.** Each refusal is new behaviour for a state that today
    yields a number. Each is tested by forcing that state. M2.4's binding makes these states
    unexpected on a verified workspace; it is not offered as proof that they cannot occur.
- **Non-finite.** A non-finite score refuses the result. Today it goes unchecked, and at the API it
  becomes a 500.

**Invariants, checked when a result is built** (`InferenceResult.__post_init__`):
- one kind for every candidate;
- ranks exactly 1..N;
- every score finite;
- order non-increasing by the kind's score, ties by its tie policy;
- required fields present and forbidden fields absent.

Spec 2's snapshot stores `CanonicalDiagnosisResult`, which is this union, so its frozen sections are
not reopened.

**Labels, fixed text, one per kind.**

| Kind | Label |
|---|---|
| `gnn_ranking/v1` | "Model ranking: raw cosine over every disease (approved scorer policy)" |
| `served_mixture/v1` | "Served ranking today: diseases reached by path discovery, scored as η × model term + (1 − η) × SP, η = `<the result's η>`", with η read from the result's served identity, never written as a constant (in `gnn_only`: "…scored by the model term alone") |
| `path_reasoning/v1` | "No model loaded: ranked by path reasoning, not by a model" |
| `demo/v1` | "Demo answer: no pipeline is configured; not a diagnosis" |

**Who migrates.** Each batch moves all of these together, in one change:

| Consumer | Change |
|---|---|
| `src/core/types.py` | `DiagnosisCandidate` is replaced by one frozen candidate type per kind; `InferenceResult` gains `result_kind` and the invariants |
| `src/inference/pipeline.py` | `_score_and_rank_candidates` writes the kind's fields; the SP term comes from §2.2's one SP caller as a status record; `_calculate_gnn_score` refuses instead of returning 0.0; the kind mapping |
| `src/api/routes/diagnose.py` | the response becomes a Pydantic discriminated union on `result_kind`; `confidence_score` and the field descriptions that are wrong in some modes are removed; `_generate_mock_candidates` answers `demo/v1`. The path stays `/api/v1/diagnose`: the prefix names the API, the body names its schema |
| WebUI Diagnosis page | the table, details header, help texts and score segment follow the kind; a non-numeric SP status shows its words; the fallback no longer shows GNN "0.000" |
| CSV and report | columns follow the kind, plus a `result_kind` column; a status column sits beside each value column, with an empty value cell where the status is non-numeric (the candidate-column convention, Spec 3 §5.1); the report's `or 0` coercion is removed; the deployment guides' column dictionary (§4.7) changes with the code |
| `src/reasoning/explanation_generator.py` | the score line prints the kind's score and label; `generate_evidence_summary` and `_get_confidence_level` are removed (they have no production caller) |
| Probes E7, R6, M4 (`scripts/probe_deployment.py`); `scripts/measure_served_pipeline.py` | read the kind's score and validate against the union |
| `scripts/test_gnn_inference.py` | removed: it pins `confidence_score == gnn_score`, which is wrong in `gnn_plus_shortest_path`. Its build-and-rank checks move to `tests/integration/test_pipeline.py`, against the union |
| `docs/SP_SCORE_GUIDE.md`'s status box | changes with the code (policy §7) |
| Mode D (`run_mode_d`, `docs/working/scorer-measurement/PLAN_B05.md`) | nothing: it reads batch 3's `ServedRanking`, and only its order, disease, served score and `discovery_ordinal` (internal metadata, not a response field) |

**Not decided here:**
- the B-1 switch's default (the owner's call, policy §7);
- whether serving with no model needs an opt-in (the owner's call, B-2). Until that is decided,
  `path_reasoning/v1` is served when no model source is configured, as today.

**Assumed:** no client outside this repository reads `/api/v1/diagnose`. If one does, the owner names it
before batch 4.

### 2.2 The shared scoring boundary (Batch 2 design, 2026-10-26)

**Status:** design for review. It is built with B-1 (batch 5), except the parts marked *batch 4*.
It settles `PLAN_TEST_RESULTS.md` D3's two open served behaviours, the disease clamp and the
unknown-phenotype handling, and policy §2.1's "one scorer, not two".

**The home is `src/inference/scoring.py`.**
- Serving and Mode C can both import it.
- The trainer cannot, under the `layers` contract. That keeps the trainer's validation values written
  independently, as the `scorer-independence` contract in `.import-linter.ini` intends.
- `src.reasoning` is not a home: statements 4a and 6 keep ranking and evidence apart.
- A layer lower down is not a home either: the trainer could adopt it there and undo that independence.

| Step | Rule | The one implementation |
|---|---|---|
| Phenotype input | One mapping step: an exact HPO term, then its graph node, then its tensor row.<br>- Repeats are removed by node, keeping the first occurrence (`normalise_phenotypes`, version 1).<br>- An unknown term is dropped and always stated (`phenotype_input.unknown` and a warning), as the importer's `keep` policy does.<br>- More than `max_phenotypes` (100) distinct terms is refused, never truncated.<br>- A known node with no tensor row is refused as a model–graph mismatch, never skipped | `DiagnosisPipeline.validate_input` returns the rows.<br>`_phenotypes_to_node_ids` and the two per-score lookups are deleted.<br>`PipelineConfig.validate_phenotypes` is removed: turning it off only silences the warning and sends the empty case to a second behaviour (`_create_empty_result`).<br>*Batch 4* |
| Patient representation | The unweighted masked mean of the phenotype rows. Out-of-range or non-integer rows are refused, and so is an empty row | `masked_mean_pool`. `pool_patient_embeddings`, which clamps, is deleted, and its clamp test is inverted into a refusal test |
| Disease set | Every disease row of the bound tensors: global ids `0..D−1` in the node mapping's disease order. NodeIDs appear only at the edge, through `kg.get_reverse_node_mapping` | `score_patients` (below). There is no lookup to miss and no clamp |
| Score | Raw cosine in the embedding dtype: float32 at serving, with no autocast. Confidence is never a weight (§4.5) | `cosine_score_matrix` |
| Order and ties | Score descending, ties to the lower global id; non-finite refused | `canonical_ranking` and `_require_integer_ids`, moved unchanged from `src/evaluation/measurement.py` |
| Selection | `selection_limit` cuts the ordered list after scoring, never before (§6.1) | the caller, B-1's pipeline method |

**One composed function.** `score_patients(phenotype_table, disease_table, rows, mask, disease_ids)`
lives in `scoring.py` and returns the scores and the order.
- **Argument shapes:** `rows` is `(B, N)` int64, `mask` is `(B, N)` bool, and `disease_ids` is `(D,)`.
- **Return shapes:** the scores and the order are both `(B, D)`.
- **What it refuses:** out-of-range rows, an all-false mask row and non-finite scores.
- **What it computes:** the pool, the cosine and the order, in that order.

**Who calls it:**
- Serving's B-1 path calls it with B = 1.
- `run_mode_c` calls it in place of its own three calls to the same primitives.
- Mode C keeps its cohort checks (`_assert_ids_in_range`, `_assert_cohort_is_intact`): they run first
  and name the case.

**Missing data and invalid values, in one place:**

| Case | Result |
|---|---|
| An unknown phenotype | dropped and stated |
| Every phenotype unknown | refused (today's "No valid phenotypes") |
| A node or a disease with no tensor row | refused |
| A non-finite score | refused |
| No model | no `gnn_ranking/v1` result (fail-closed); the fallback is served only under B-2's rule |

**Precision.** The ranking uses raw float32 cosine. The legacy `(cos+1)/2` is rounded in float32, which merges
adjacent cosines into ties. It exists only in `served_mixture/v1` and goes with it.

**The SP caller (one, batch 4).** `sp_mean_distances` reports availability per batch. `sp_analysis_records` takes
§3.3's per-candidate statuses from the node mapping and the mapped phenotypes. It sits in
`src/inference/sp_index.py`, beside the primitive. The primitive gains one output, the
per-candidate count of unreachable pairs, which it already computes internally.
- **Batch 4** calls `sp_analysis_records` for the legacy SP term, over every reached candidate in one
  call. The distances are sums of small integers, so they are exact and equal to today's
  one-at-a-time values.
- **B-1** calls it for eager SP over the selected set.
- **Two clean-ups:** `scripts/audit_sp_reachability.py` imports `PRODUCER_HOP_RANGE` instead of declaring its own copy, and
  the unused `ASSUMED_HOP_BOUND` is deleted.

**Callers to migrate:**

| Caller | Moves to |
|---|---|
| `run_mode_c` | `score_patients`, importing the ranking from `scoring`. Its metrics, ranks, digests and the manifest's semantic fields stay equal before and after (timestamps, revision and run metadata excluded, as `../scorer-measurement/PLAN_B05.md` §8 defines). The README's "not edited" list binds Mode D's package; B-1 moves the code and changes no rule |
| `ranks_of_truth` | imports `_require_integer_ids` from `scoring` |
| `tests/unit/test_measurement_ranking.py` | imports from `scoring` |
| `tests/unit/test_scoring_primitives.py` | the clamp test (`test_pooling_clamps_out_of_range_indices`) is inverted into a refusal; its other uses of `pool_patient_embeddings` move to `masked_mean_pool`. `test_exact_ties_break_by_candidate_order` stays while `served_mixture/v1` is served, and goes when it retires |
| B-1's serving path | `score_patients`, then the selection |
| `_calculate_gnn_score`, `cosine_scores`, `normalise_cosine_to_unit_interval`, `mix_embedding_and_sp_scores` | used by `served_mixture/v1` only; deleted when it retires. The owner's switch default decides when |
| The trainer | unchanged: its own cosine and its own index clamp (`trainer.py`, `_compute_model_outputs`). The clamp is a known difference, outside B-1 |

### 2.3 What the scorer is, against the reference paper (2026-10-26)

The interim disease scorer here is the approved project policy (statement 2): raw cosine of an
unweighted masked mean. It is not a reproduction of the paper's disease task, which uses
`−‖z_d − z_P‖²` (Eq 18) on its own patient encoder. The paper mixes SP only into its candidate-gene
task (Eq 12–14). For disease ranking, SP here is the optional, default-off analysis view of §3,
whose sorting and filtering change the view only and never the canonical rank. The one table of the
paper's three tasks, their inputs, formulas, status, this project's differences and where each
would land is `docs/ARCHITECTURE.md` §3, kept in one place so two copies cannot drift.

---

## 3. SP analysis contract

### 3.1 Which quantity

| Quantity | Definition | Role |
|---|---|---|
| **Mean hop distance** | mean over the patient's mapped phenotypes of `d(p, candidate)`, unreachable pairs contributing `max_hops + 1` | **the filter and sort key**, and the primary display |
| **SP score** | `1 / (1 + mean hop distance)` | shown alongside, for continuity with the existing score |

"Show me candidates within an average of 3 steps" is a statement a clinician can form and check;
"show me SP ≥ 0.25" is not. The `1/(1+d)` transform also compresses the far end so heavily that the
gap between 1 and 2 steps is seven times the gap between 5 and unreachable, so it actively obscures
the distinctions a filter would be built on.

> **The two quantities carry equivalent ordering information in opposite directions: ascending mean
> hop distance equals descending SP score. The UI labels the active direction explicitly.**

*Numerical footing:* the ordering equivalence holds under `max_hops = 5`, `N ≤ 100` (contractual —
`DiagnoseRequest.phenotypes` in `src/api/routes/diagnose.py` caps the phenotype list) and float64
accumulation. **The derivation belongs in `sp_mean_distances`' docstring, next to the code it
constrains, and is not restated here** — a product specification is the wrong home for a
floating-point proof, and two copies would drift. *(Factual note, 2026-10-26: the derivation stood in
that docstring while the primitive lived in `src/inference/scoring.py`; the move to
`src/inference/sp_index.py` (`87f8d13`) did not carry it. Batch 4 restores it there, BACKLOG §0.)*
Regression tests pin it (`tests/unit/test_scoring_primitives.py`).

### 3.2 Reference set

**No candidate-relative normalisation at this stage.** Both quantities are absolute, so C7 holds
trivially: nothing depends on which candidates are present, and values are stable under paging,
sorting and filtering.

Values are stable **within** a result. Comparing them **across** results is meaningful only when the
KG and SP fingerprints, `max_hops`, traversal semantics and mapping coverage all match; any surface
placing two results side by side checks compatibility and says so when it fails.

Candidate-relative normalisation — the form the reference paper uses — is deferred by institutional
decision. If adopted, C7 requires its reference set fixed to the canonical selected set at result
production, never to a page or a filtered subset.

### 3.3 Status

```
sp_status = COMPUTED            # every submitted phenotype mapped;   numeric
          | COMPUTED_PARTIAL    # some phenotypes unmapped;           numeric
          | NO_TABLE            # no shortest-path artifact loaded;   no number
          | TARGET_UNMAPPED     # target not in the node mapping;     no number
          | NO_PHENOTYPE_MAPPED # no phenotype mapped at all;         no number
```

> **Numeric set: `{COMPUTED, COMPUTED_PARTIAL}`. Non-numeric set: `{NO_TABLE, TARGET_UNMAPPED,
> NO_PHENOTYPE_MAPPED}`.** Predicates use these explicit sets. `sp_status != COMPUTED` is **not** a
> test for unavailability — it wrongly captures the numeric `COMPUTED_PARTIAL`.

The schema is a discriminated union: numeric fields **required** on the first two variants,
**forbidden** on the other three. An absent number and a zero must not be representable as the same
thing. (The legacy `_calculate_sp_score` returns `0.0` on five distinct failure paths (policy C9),
and `0.0` is *below* the value a genuine "no path found" produces — so a numeric sort would place a
mapping failure beneath a real negative.)

**`TARGET_UNMAPPED` is defined against the canonical node mapping, not against the SP artifact.** The
artifact is sparse: absence of a `(phenotype, target, type)` row means *no path was found within the
hop limit*, which is a normal computed outcome. A target that resolves in the node index but appears
in no SP row is a valid **computed, all-unreachable** candidate.

### 3.4 Two "unreachable" notions, and the denominator

| | What it is | Representation |
|---|---|---|
| A **pair** with no path | normal and common | a number: contributes `max_hops + 1` to the mean, inside a `COMPUTED` or `COMPUTED_PARTIAL` result |
| A **candidate** that could not be evaluated | no table, target unmapped, no phenotype mapped | a non-numeric status |

An all-unreachable candidate is still computed, and **both extrema are named**: it carries the
**maximum** mean hop distance (`max_hops + 1`) and therefore the **minimum** SP score
(`1/(max_hops + 2)`). Unreachable pairs and all-unreachable candidates can occur under
`COMPUTED_PARTIAL` too.

Every analysis record carries what makes a value interpretable:

```
phenotypes_submitted, phenotypes_mapped, unmapped_ids[]
unreachable_pair_count, max_hops
denominator_policy = MAPPED_ONLY_DENOMINATOR | ALL_SUBMITTED_DENOMINATOR
```

**Current behaviour is `MAPPED_ONLY_DENOMINATOR`**, structurally rather than by choice: unmapped
phenotypes never enter `phenotype_indices`, so the mean divides by the mapped count. Mapped
phenotypes with no table entry are the *other* case and **are** imputed as `max_hops + 1`.

Consequence: a `COMPUTED_PARTIAL` value is on the same scale as a `COMPUTED` value but not on the
same footing — at equal true reachability it reads closer. Sorting them together is permitted;
hiding coverage is not. **Every partial row shows `phenotypes_mapped / phenotypes_submitted`.**

### 3.5 Presentation of unavailable candidates

They sort into a single group at one end, never interleaved as low scores; they occupy their **own
facet**, never a numeric bucket; and a numeric range filter never removes them — it selects among
computed values, and the unavailable group is included or excluded by its own explicit control.

### 3.6 Default state and facets

**The SP filter defaults off and stays off.** A default-on filter would grant SP candidate-admission
authority through the view layer, which statement 4a denies it in the pipeline, and a first-time
clinician would receive a list already narrowed by a signal they did not choose.

Instead, show the **distribution** of mean hop distance across the selected set — a compact histogram
or quantile summary — so the affordance is discoverable without being applied.

Per C8: facet counts are computed over the **whole selected set**, never the current page;
pagination never changes them; facets show the count a filter *would* leave before it is applied
(C6). The unavailable group is a facet in its own right.

---

## 4. Per-candidate score decomposition

### 4.1 The exact decomposition

For phenotype embeddings `p₁…p_N`, their mean `m`, and candidate embedding `d`:

```
cos(m, d) = Σᵢ (pᵢ · d) / (N ‖m‖ ‖d‖)
contributionᵢ(d) = (pᵢ · d) / (N ‖m‖ ‖d‖)
```

Contributions **sum exactly to the raw cosine**, which is what makes them displayable next to the
score. Cost: one `(N, H) × (H, C)` matmul, about 1M multiply-accumulates at 20 phenotypes, `H = 256`
and 200 candidates.

### 4.2 Required handling

| Case | Rule |
|---|---|
| Score semantics | Contributions decompose the **raw cosine**. Any transformed score displayed elsewhere has its relationship stated |
| Phenotype not mapped | Contributes nothing and is listed as **not analysable**, distinct from contributing zero |
| Duplicate entries | **The analysis does not decide.** It consumes exactly the sequence, multiplicity, mask and denominator `N` the canonical scorer used. Deciding independently — even "correctly" — makes contributions stop summing to the score they claim to decompose |
| Padding masks | Excluded from `N` and from the sum |
| Negative contributions | Real and clinically meaningful; displayed, never clipped |
| Tolerance | Displayed contributions sum to the displayed score within a declared tolerance, asserted in test |

### 4.3 The near-zero-norm boundary

The canonical scorer normalises with epsilon clamping (`F.normalize`, `eps = 1e-12`), so
`denominator = max(norm, eps)`. Exempting only *exact* zero is insufficient: a nonzero norm below
`eps` still diverges, because the scorer would use `eps` while the contribution formula uses the
smaller raw norm — the contributions then stop summing to the canonical score and can display as
enormous values.

> **When `‖m‖ ≤ eps` or `‖d‖ ≤ eps` — using exactly the scorer's epsilon, dtype and device semantics
> — the canonical score remains defined and is displayed. The decomposition is marked unavailable and
> is explicitly exempt from the exact-sum invariant.**

The invariant is asserted in test over the non-degenerate case only, and the exemption is named in
that test rather than left as an untested branch.

*Why not mirror the clamp instead.* Mirroring is mathematically exact but produces contributions of
order `10¹²` cancelling to zero — unreadable and actively misleading on a clinical surface. That
applies to a **near-zero patient mean produced by cancellation**; if the *disease* vector is exactly
zero, every dot product is zero and no large contributions arise. Both cases are covered by the rule
above.

### 4.4 What this explains — and what it does not

The decomposition explains a candidate's **absolute score**. It does **not** explain why that
candidate ranks above another: a phenotype contributing strongly to *every* candidate explains none
of the differences between them, yet would dominate the display by being the largest number on every
row.

Required wording, verbatim:

> **"Per-phenotype contribution to this candidate's raw cosine score."**

Forbidden: ~~"Why this candidate is ranked here."~~

The comparator analysis that *would* address ranking differences — each phenotype's contribution
relative to its mean contribution across the selected set — is nearly free to compute but is a
different analysis with different failure modes. **Deferred, and recorded so the gap is visible
rather than forgotten.**

### 4.5 Phenotype confidence is not a weight

Snapshots retain phenotype confidences, but pooling is an unweighted mean and the scorer does not use
confidence as a weight. Displaying a confidence next to a contribution invites the inference that one
influenced the other. The contribution surface records:

```
used_by_scorer = false
effective_aggregation = UNWEIGHTED_MEAN
```

Contribution provenance binds each output to the original ordered phenotype occurrence, duplicate
multiplicity, confidence, mapping outcome, and effective scorer index.

---

## 5. Two surfaces

| | Inference page | Analysis workspace |
|---|---|---|
| Question | "Did the run succeed, and is the answer obvious?" | "Work through the whole list systematically" |
| Shows | `summary_limit` rows (default 10), essential columns | full selected set, paginated, facets, sort, decomposition, SP panel |

The split is forced: 200+ rows cannot be usefully rendered on the inference page, and the only choice
is whether that page shows 0 rows or ~10 — and 0 turns it into a form with no feedback. The two tasks
are also different cognitive modes.

**Share the candidate view-model and the formatting authority; do not force both surfaces through one
heavily conditional component.** Two thin presentational components over one shared view-model. The
detailed reasoning display currently on the inference page moves to the workspace.

### 5.1 Navigation (R8)

The UI is already tabbed (`src/webui/app.py:191-201`) and per-session result state already exists
(`results_state = gr.State(None)`, `src/webui/components/diagnosis_panel.py:747`), so the mechanism
is present rather than new. Gradio renders tab content and hides it rather than unmounting, so
component state is expected to survive the round trip — **to be confirmed against the deployed
version rather than assumed**.

**The direction needing design attention is coming back, not going.**

- Within one result, the round trip preserves **page, sort and filters** — these live in
  `gr.State` and are normative.
- **Scroll position is best-effort, not normative.** It is browser DOM state, not component state,
  and whether it survives depends on the deployed Gradio version. Use Gradio's built-in `Tabs`,
  `Tab`, `State`, event outputs and component updates first; add custom JavaScript only if that is
  demonstrably insufficient, and do not build a separate client router or state store.
- Re-running inference produces a **new `result_id`**, so the previous view state does not apply.
  Required by C10 and correct, but it must be **explained on screen** or it reads as settings
  vanishing for no reason.

*Worth measuring after deployment:* how often clinicians open the workspace. If it is every time, the
signal is that `summary_limit` or the summary's columns are wrong — not that navigation needs work.

---

## 6. Limits and cost

| Parameter | Meaning | Authority | Default | Min | Max |
|---|---|---|---|---|---|
| `selection_limit` | candidates selected and eagerly enriched | clinician, within deployment bounds | **[OPEN]** | **[OPEN]** | **[OPEN]** |
| `summary_limit` | rows on the inference page | deployment | 10 | — | — |
| `page_size` | rows per workspace page | clinician | **[OPEN]** | 10 | 100 |

Also **[OPEN]**: an **interaction latency target** covering **inference + eager enrichment +
automatic snapshot publication**. The snapshot write is inside the target, not outside it: if
`snapshot_status` is final in the diagnosis response, then the write, the file `fsync`, read-back
validation, the rename, the **snapshot-directory `fsync`** and rotation all complete before the
response does (Spec 2 §7.3, §7.4). Two `fsync` calls, not one — the second is what the eviction is
allowed to depend on, and it is inside the target for the same reason the first is.

**Do not add a background-job subsystem to move it out of the target.** The payload is a few hundred
kilobytes; benchmark the synchronous path first and only then decide whether it needs hiding.

Without this number, Spec 3's budget design has nothing to aim at and §6.2's acceptance criterion has
nothing to pass or fail against.

### 6.1 What `selection_limit` bounds

| Stage | Bounded? |
|---|---|
| Scoring every disease in the KG universe | **No.** Statement 3 requires the full universe; the cost is a single matmul, independent of this parameter |
| Selecting candidates from those scores | Yes — this is its definition |
| Eager SP enrichment | **Yes** — where R9's cost control acts |
| Per-phenotype contributions | Yes |
| Evidence traversal | No — bounded per target instead (Spec 3 §2) |

> **`selection_limit` is a presentation and enrichment bound, never a scoring bound.** Using it to
> limit how many diseases are scored would be candidate gating under a new name.

### 6.2 The eager-SP performance dependency: delivered up to 500 candidates, unmeasured beyond

*(Rewritten 2026-10-26, Batch 2: the earlier text described a scanning lookup that no longer
exists.)* `sp_mean_distances` (`src/inference/sp_index.py`) is the one served SP primitive: a sorted
composite key queried with `torch.searchsorted`, vectorised over a (phenotypes × candidates) query
(B-0.4, BACKLOG 5a: implemented, acceptance pending). Measured on GB10 against the real table with
the shipped primitive (`docs/working/scorer-measurement/EVIDENCE_B04_served_primitive_spark.json`): a
batched call takes at most 4.6 ms (median) up to 100 phenotypes × 200 candidates, and 13.2 ms
(median) at 100 × 500.

> **Eager SP over a selected set of up to 500 candidates is within those measurements. Above 500
> it is unmeasured**, and so is the full-universe query (about 32,000 diseases), which eager SP does
> not need: SP runs over the selected set only (§6.1).

**Still a B-1 acceptance criterion**, measured on institutional hardware against the declared
latency target (§6, open), for a representative request at the deployment's `selection_limit`. No
serving code calls the primitive batched yet; §2.2 names its one caller.

**This completes no part of B-0.** B-0's remaining scope: Mode D (designed in
`docs/working/scorer-measurement/PLAN_B05.md` and built in Batch 3, 2026-10-26, for review; Modes A
and B are retired, not remaining), the
paired-cohort, strata and bootstrap protocol (BACKLOG 8a), and the institutional measurement run
(BACKLOG 8b). Full-universe candidates, untruncated MRR and Hits@{1,5,10,20,50,100}, the manifests
and fingerprints and the output schema exist in Mode C; negative-set instrumentation went with
Modes A and B. Evaluator migration is no longer part of it:
the frozen evaluator was never migrated onto the shared primitives, and was deleted on 2026-10-10
(`docs/working/PLAN_PROVENANCE_CONTRACT.md` M2.1, decision 6).

---

## 7. Deferred, with reasons

| Item | Why |
|---|---|
| **Triage state** (reviewed / dismissed / flagged) | Persistent clinical state: actors, reasons, history, permissions, concurrency, retention, export. "Dismissed" must never silently remove a candidate — the principle behind C6 |
| **Multi-user and authentication** | Spec 4 §6 |
| **Candidate-relative SP normalisation** | §3.2 — institutional decision |
| **Comparator contribution analysis** | §4.4 — needs its own design and validation |
