# Disease Scorer Policy — Decision Record

**Status:** Accepted (2026-08) by the deploying institution. **Not yet implemented** — see the
status table in §2. Implementation is work item B-1, which remains gated (§6).

**Revision: rev 4. Factual correction.** The implemented B-0 artifacts record aggregate metrics
and per-sample ground-truth ranks, not per-candidate score components. The deferred cascade/tie-break
analysis can be performed without a second institutional inference run only if B-0.5 persists an
approved bounded component artifact, or computes the predeclared analysis during that run. No
normative scorer-policy decision changed. **The deploying institution should be informed that its
authority document was corrected**; reapproval is not required for a factual amendment.

**Revision:** rev 3. Amended to permit clinician-controlled view operations over an immutable
canonical result (statement 4a/4b, §1.1), to record the eager-SP / lazy-evidence computation
schedule (§1.2), and to separate the three limits previously conflated as `top_k` (§1.3). The
amendment follows an institutional requirement to raise the candidate list to 200+ with pagination
and SP sorting and filtering.

**Revision:** rev 2. Corrections applied after review: the top-k analysis set no longer claims to
reconstruct the paper's short-list condition (§5); the rollout uses a versioned discriminated union
rather than a flat envelope (§7); the deployed-checkpoint claim is split into repository fact and
reported observation (§3.2, §3.3); the full-universe objection is restated as unvalidated meaning
and outlier sensitivity rather than an absent reference set (§4); and the C-vs-E′ revisit condition
is conditional on that optional audit being run (§8).

**Scope.** How disease candidates are discovered, scored and ranked, and what role the
shortest-path (SP) signal and path reasoning may play. This record is the **authority** for that
policy. [`SP_SCORE_GUIDE.md`](SP_SCORE_GUIDE.md) explains what the SP score *means* to a clinician
and does not define policy; [`ARCHITECTURE.md`](ARCHITECTURE.md) describes the system as a whole and
will be reconciled with this record when B-1 lands.

---

## 1. Decision

Seven normative statements. Each is binding on the target design; none describes current behaviour.

1. **Disease candidate ranking is GNN-primary.**
2. **The current checkpoint's interim disease score is raw cosine** between the pooled patient
   phenotype embedding and the disease embedding.
3. **Every disease in the configured KG universe is scored before selection.**
4. **4a — The scoring pipeline may not use SP to admit, exclude, filter, reorder or rescore disease
   candidates.** The canonical candidate set and its canonical rank order are produced by the
   disease scorer alone, and are immutable once produced.
   **4b — A clinician may sort or filter *the view* by SP within a dedicated SP analysis surface**,
   subject to every condition in §1.1. A view operation is a **projection over an immutable
   canonical result**: it never alters that result, and never affects any other surface.
5. **SP may be exposed only as an optional post-ranking contextual analysis** over candidates
   already selected by the disease scorer.
6. **Numeric SP analysis and typed evidence-path traversal are separate concerns** and must not be
   merged into a single mechanism or a single UI affordance.
7. **Path-reasoning fallback uses separately discriminated score semantics** and cannot be mixed
   into a GNN-ranked result.

Statement 2 is explicitly **interim**. It reflects what the deployed checkpoint's training objective
optimised, not the reference paper's disease scorer (§3.1). It is expected to change if and when the
paper-parity retraining track is undertaken, which is outside work item B.

### 1.1 Conditions on clinician-controlled view operations

Statement 4b is permitted **only** under all of the following. C1–C5 apply to both sorting and
filtering; C6 applies to filtering alone; C7–C10 govern the data the view is built on.

| # | Condition | Applies to |
|---|---|---|
| **C1** | **Canonical rank stays visible.** Every row shows its canonical rank (e.g. `#7`) regardless of the current view order | sort, filter |
| **C2** | **Reversible.** Canonical order with no filter is the state on entry; one action restores it | sort, filter |
| **C3** | **View state is labelled on screen.** The view states plainly that it is not the canonical presentation | sort, filter |
| **C4** | **Export defaults to the truth.** The default export is the **full canonical result in canonical order**. "Export current view" is a **separate, explicit** action whose artifact carries a prominent filtered-view label, the hidden count, the full view state, the canonical rank of every row, the result ID, actor and time, and the scorer / KG / SP artifact fingerprints | sort, filter |
| **C5** | **Audit provenance.** The view state at the time of any action taken from the view is recorded with that action | sort, filter |
| **C6** | **The exclusion is always countable and visible.** A filtered view permanently displays how many candidates it is hiding (`showing 47 of 200`), and facet counts show the effect of a filter **before** it is applied | **filter only** |
| **C7** | **Fixed SP semantics and reference set.** Which SP quantity is displayed and filtered — and, if any candidate-relative normalisation is used, its reference set — are fixed when the canonical result is produced. They are **never** recomputed from a page or from a filtered subset | all |
| **C8** | **Whole-set before pagination.** Filtering and sorting apply to the whole selected set, and facet counts are computed over the whole selected set. **Pagination never affects candidate values, counts, or which candidates a filter matches** | all |
| **C9** | **Unavailable SP is a distinct state, not a number.** A value that could not be computed (no table loaded, unmapped node, lookup failure) is represented as unavailable, never as numeric zero. It carries its own facet and is never ordered as though it were a low score | all |
| **C10** | **Every view is bound to provenance.** A view is addressed by an immutable result ID and carries the scorer, KG and SP artifact fingerprints of the result it projects | all |

**Why C9 is a condition and not a detail.** `_calculate_sp_score` currently returns `0.0` on four
distinct failure paths (`src/inference/pipeline.py:1333, 1337, 1347, 1358`), and `0.0` is **below**
the value a genuine "no path found" produces (`1/7`). Under a numeric sort, a mapping failure would
therefore rank below a real negative and be indistinguishable from a very poor score.

### 1.2 Computation schedule — decided

**SP is computed eagerly for the entire selected set, once, when the canonical result is produced.**
Path evidence is **not**: it is computed lazily for the visible page, as a separately versioned
subresource (§7).

The two differ because their costs differ by orders of magnitude — SP is one lookup per
(phenotype, candidate) pair, while evidence is a bounded graph traversal — and because statement 6
requires them to remain separate concerns.

Eager SP has a consequence worth stating: **it makes C7 and C8 true by construction** rather than by
careful implementation, and it removes any need for request identity, stale-response rejection or
cache invalidation on the SP path. The clinician waits once, at inference, and then browses, sorts
and filters with no further computation.

### 1.3 Three limits, previously conflated as `top_k`

`top_k` (`src/inference/pipeline.py:913`, `src/webui/components/diagnosis_panel.py:105`, default 10)
currently means "how many to produce" and "how many to show" at once. The target design separates
them:

| Parameter | Meaning | Authority | Where it is set |
|---|---|---|---|
| `selection_limit` | how many candidates the scorer selects, and therefore how many receive eager SP enrichment | clinician, within deployment bounds | inference settings |
| `summary_limit` | how many appear in the inference page's concise summary | deployment | configuration |
| `page_size` | how many rows the analysis workspace shows at once | clinician | view control |

Only `selection_limit` is an inference-time parameter. Its default, minimum and maximum, the
storage and export bounds, and the interaction latency target are **[OPEN]** and require
institutional values.

---

## 2. Status — current behaviour vs approved target

**The gap between these columns is the point of this table.** Nothing in the "approved target"
column is implemented.

| Concern | Current behaviour | Approved target | Implementation status |
|---|---|---|---|
| **Candidate universe** | BFS path discovery gates scoring. Two order-dependent caps truncate it: `max_paths_per_source = 100` (`src/reasoning/path_reasoning.py:105`) and `max_genes = 100` per phenotype (`:507`), both applied to unordered neighbour lists | Every disease in the configured KG universe is scored; no discovery gate | **Not implemented** — work item B-1, gated |
| **Disease score** | `0.7 × ((cos+1)/2) + 0.3 × SP` when a checkpoint and `shortest_paths.pt` are both loaded (`src/inference/pipeline.py:1310`) | Raw cosine (interim, statement 2) | **Not implemented** — B-1 |
| **SP role** | A ranking term with nominal coefficient 0.3, and therefore able to reorder candidates. The coefficient is not the term's effective contribution — see §3.5 | Optional post-ranking contextual analysis only; separate field or panel; no effect on identity, order, rank or ranking score | **Not implemented** — B-1 |
| **Evidence-path role** | Path reasoning performs candidate discovery, supplies evidence, **and** becomes the ranking score when no GNN is loaded | Evidence only, via a target-restricted traversal that cannot alter the candidate set | **Not implemented** — design not yet approved (gate 3, §6) |
| **GNN-unavailable fallback** | `confidence_score = reasoning_score` (`src/inference/pipeline.py:1224`) — a different quantity written into the same field with no per-result discriminator | Discriminated score semantics; fail-closed by default; fallback only by explicit request or approved deployment policy | **Not implemented** — work item B-2 |

Two further current behaviours, for completeness:

- When a checkpoint is loaded but `shortest_paths.pt` is absent, scoring degrades to pure GNN
  (`_calculate_combined_score` returns the embedding score, effective η = 1.0).
- `get_pipeline_config()` reports a `scoring_mode` of `gnn_plus_shortest_path`, `gnn_only` or
  `path_reasoning_fallback`. This is a **pipeline-level** report; it is not attached to individual
  results or candidates.

---

## 3. Evidence

Classified by kind, because the five kinds carry different weight and are verifiable in different
ways: what the paper states, what the source tree shows, what the deployment reports, what the
institution decided, and what the author inferred.

### 3.1 Primary-source evidence — the reference paper

Alsentzer E, Li MM, Kobren SN, Noori A, Kohane IS, Zitnik M. *Few shot learning for
phenotype-driven diagnosis of patients with rare genetic diseases.* **npj Digital Medicine** 8:380
(2025). DOI `10.1038/s41746-025-01749-1`. PMC `PMC12181314`.

**The decisive item is task attribution.** The paper defines the SP fusion for **candidate-gene**
scoring while using a **separate formulation** for disease scoring:

| Task | Scoring function | SP term |
|---|---|---|
| Causal gene discovery | `SIM(P,g) = η · EMBSIM(P,g) + (1 − η) · SPLSIM(P,g)` (Eq 14) | **present** |
| Patients-like-me | `SIM(Pᵢ,Pⱼ) = −‖z_Pᵢ − z_Pⱼ‖²₂` (Eq 16) | absent |
| Novel disease characterisation | `SIM(P,d) = −‖z_d − z_P‖²₂` (Eq 18) | **absent** |

Also from the Methods: *"we calculate a patient's similarity to all disease nodes in the KG at
inference time"* — supporting statement 3.

**Supporting context, explicitly not independent proof.** The following describe how firmly the
paper establishes SP's value *in the gene task*. They are recorded because they bear on how much
weight to give the fusion design in general, but **none of them is evidence about the disease task**,
and the decision does not rest on them:

- The stated rationale for SPLSIM is that local methods rank true candidates higher *"when provided
  a short list of candidate genes"*, cited to a survey of gene-prioritisation tools (Zolotareva &
  Kleine, *J. Integr. Bioinform.* 16, 20180069, 2019) rather than to an experiment in the paper.
  The gene task supplies such a list — **13.3 genes** on average (EXPERT-CURATED, SD 8.0) or
  **244.3** (VARIANT-FILTERED, SD 244.0). The disease task does not.
- No ablation of SPLSIM appears in the main article as reviewed. Supplementary materials were not
  examined.
- The selected value of η is published in the authors' repository rather than in the article.

### 3.2 Repository evidence

| Fact | Source |
|---|---|
| Training optimises cosine over mean-pooled phenotype embeddings; no learned task head participates | `src/training/trainer.py:744-766`; `src/training/loss_functions.py:513-517` |
| `ShepherdGNN` constructs no task head, and checkpoint loading uses strict `load_state_dict` semantics | `src/models/gnn/shepherd_gnn.py:96-173`; `src/inference/pipeline.py:831` |
| η defaults to 0.7 and is a project choice, not a paper value | `src/inference/pipeline.py:296-309` |
| The SP transform is absolute, `1/(1 + mean(d))`, with range `[1/7, 1/2]` under `max_hops = 5` | `src/inference/pipeline.py:1361, 1380` |
| Candidate discovery is BFS-gated and truncated in traversal order | `src/reasoning/path_reasoning.py:105, 275-276, 507` |

### 3.3 Reported deployment observation, and what it supports

Kept separate from §3.2 because it is not readable from source. **Reported by the institution:** the
deployed checkpoint `model-39-0.7004.pt` loads successfully and the pipeline reports the GNN as
ready.

**[INFERENCE]** Combined with the repository facts above — `ShepherdGNN` builds no task head, and
`load_state_dict` is strict — a successful load implies that this checkpoint's key set matches the
model's exactly, and therefore that it carries no task-head parameters. The inference is sound but
depends on the report; it is not a source-code fact and is not a claim about any other checkpoint
file. Its practical bound is that a checkpoint which cannot pass a strict load is not deployable at
all.

### 3.4 Institutional decision

The deploying institution decided (2026-08) that the pipeline should reproduce the original
SHEPHERD design as the primary direction, deviating only where a design is demonstrably better and
supported by citable evidence; that disease ranking is GNN-primary; and that SP is removed from
ranking and re-exposed as an optional, clearly separated analysis.

**Provenance matters here and is recorded deliberately.** This decision was taken **before** the B-0
measurements were run, and it is consistent with the pre-declared decision rule that raw cosine is
the default and that η must *earn* adoption with evidence. It was not derived from reading results.
A future reader should not mistake it for an empirical finding, nor reopen it as though it were an
unexamined default.

### 3.5 Engineering inference — weigh accordingly; one item is now measured, the rest are not

- ~~Over the full disease universe most candidates fall outside the 5-hop table and receive the same
  floor value, so the SP term degenerates towards a binary reachability indicator.~~ **The premise is
  measured and false at the per-phenotype level.** On the deployment artifact the median phenotype
  reaches **19,216.5 of 29,866** diseases within the configured 5 hops (**64.3%**), first quartile
  51.2%, maximum 78.16%; **270 of 19,836** phenotypes (1.36%) reach none at all
  ([`EVIDENCE_M5.json`](working/EVIDENCE_M5.json), BACKLOG §2.4).

  **What that 1.36% does and does not size.** It counts phenotypes that reach *no* disease. It is
  **not** the prevalence of unreachable phenotype–candidate pairs, and using it that way would
  understate them by orders of magnitude: the median phenotype still leaves ~10,650 diseases
  unreachable, and even the best-connected phenotype in the graph leaves ~6,522. Unreachable pairs
  are common at every phenotype; M5 does not size them as a fraction of pairs.

  **What M5 does not establish at all: the deployed, patient-level score.** M5 counts per phenotype.
  The deployed scorer does not take a union over a patient's phenotypes and does not use the nearest
  reachable one — for each candidate it iterates over **every** phenotype, gives each unreachable
  pair `unreachable_distance`, and averages
  (`src/inference/scoring.py::sp_mean_distances`). Per-phenotype reachability therefore cannot be
  carried across to patient-level coverage in either direction.

  **What does not follow, in either direction.** The conclusion no longer follows from the refuted
  premise, but is not thereby refuted. A candidate whose **mean** distance is 5 scores `1/6` against
  a floor of `1/7` — a five-hop path from one phenotype does not by itself produce that score — so
  near-degeneracy could still arise from the distribution of mean distances rather than from
  unreachability. Neither the reachable-distance distribution nor the patient-level mean-distance
  distribution is measured, and no current artifact carries either.
- Because the SP range is `[1/7, 1/2]` while the embedding term spans `[0, 1]`, η is not the
  effective weight. The **maximum theoretical spans** stand at 0.7 versus 0.107; the actual
  contribution of each term depends on their observed spread, which has not been measured.
- Candidate discovery may be biased towards diseases with shorter or denser KG connections, because
  the path search truncates in traversal order. Whether that translates into a bias towards
  commoner diseases is **not established** by the code and would require measurement.

---

## 4. Alternatives considered

| Alternative | Why not chosen |
|---|---|
| **Keep η in ranking, applied to the full universe** | Requires SP over the whole disease universe — ~27,990 when this table was written, 29,866 on the audited deployment workspace. The degeneracy this row relied on is **no longer an established reason**: §3.5's premise was measured and refuted, and what remains of the concern is unsettled rather than supported. Min–max normalisation over that set is *mathematically* well defined; the objections are that its clinical and task meaning is unvalidated, and that min–max is set by the two extreme candidates, so extreme candidates may compress much of the remaining distribution into a narrow band. No paper support for the disease task. |
| **Keep η in ranking, applied to a GNN top-N cut** | Reintroduces a candidate gate — softer than BFS, but still able to hide a candidate the GNN ranked highly. Adds an N-selection problem (recall, top-k set and rank preservation, latency) that pure cosine does not have. |
| **Cascade: use SP only to break ties among candidates the GNN cannot separate** | The most defensible of the alternatives, because it applies SP to a bounded set of already-plausible candidates. Deferred rather than rejected: it needs an operational definition of "cannot separate", which is an uncalibrated threshold; and it is a deviation from the paper requiring its own evidence — the resemblance to the paper's short-list setting is limited to list size and does not carry the paper's validation across (see §5). It does not require a new scoring mode. It can be evaluated without a second institutional inference run only if B-0.5 records an approved bounded per-candidate component artifact, or computes the predeclared analysis during the B-0.5 run. **The currently implemented B-0 artifacts do not contain those components.** |
| **Remove the SP subsystem entirely** | Rejected. B-0's comparison modes need it; the paper places KG-distance fusion in candidate-gene scoring, which is unbuilt future work; and clinicians may legitimately want SP context on a short list. Demoted, not deleted. |

---

## 5. Consequences

- **`eta`, `_calculate_combined_score`, `_calculate_sp_score`, `shortest_paths.pt` and
  `scripts/compute_shortest_paths.py` are demoted, not removed.** They remain required for B-0's
  comparison modes and for the optional contextual analysis.
- **This is a behaviour change on a live clinical system.** η = 0.7 is what runs today. The change
  must be visible to the institution, not silent, and is gated accordingly (§6).
- **The candidate list a clinician sees will change**: composition, ordering, and which candidates
  carry supporting paths. Some candidates will appear with no path at all, labelled.
- **A separate SP analysis over the displayed top-k creates a bounded, candidate-relative analysis
  set.** This resembles the paper's short-list setting in one limited respect — bounded size — but
  **does not reproduce the paper's candidate-gene task and does not validate SP fusion for disease
  scoring**. The paper's list contains candidate genes supplied by variant filtering or expert
  curation; this list contains model-ranked diseases. What the bounded set does provide is a
  well-defined reference set for candidate-relative normalisation, which the current absolute
  transform does not perform. How to implement that normalisation is deferred to the implementation
  stage.
- **Two implementation defects are recorded and deferred**, not fixed by this decision: the range
  mismatch between the two score terms, and the absence of candidate-relative normalisation
  (§3.5, and `SP_SCORE_GUIDE.md` §3).

---

## 6. Gates

B-1 implementation may not begin until all three clear:

| Gate | Status |
|---|---|
| **1 — B-0 measurement report** | Not cleared. B-0 is approved for implementation; the measurement has not been run. |
| **2 — Institutional scorer-policy decision** | **Cleared by this record.** |
| **3 — Approved target-restricted evidence-traversal design** | Not cleared. Requires legal reachability × completeness combinations, visited-state and cycle semantics, multi-source provenance, deterministic ordering, global-budget units and enforcement, target fairness, and cancellation behaviour. |

---

## 7. Rollout

- The change ships behind a configuration switch. **Its default value is an explicit
  pre-implementation decision**, not an implementation detail, because it determines whether the
  live system changes behaviour on deploy or on a later deliberate switch. Recorded as open.
- B-1 and B-2 both modify the candidate and result types. **One versioned discriminated result
  union is agreed before either lands.** Each result variant defines its own **required** and
  **forbidden** fields, selected by the mode discriminator. This explicitly rules out a flat
  envelope carrying every field with nulls where irrelevant — that shape is how `confidence_score`
  came to hold two incompatible quantities, and repeating it with more fields would scale the defect
  rather than fix it. Whichever work item lands first defines the union and its first variants; the
  second adds variants rather than widening a shared record.
- The clinician-facing guide's status box and the SP presentation change together with the code, not
  before it.

---

## 8. Conditions for revisiting

This record should be reopened if any of the following occurs:

1. **If the optional C-vs-E′ audit is run and shows the η mixture beating raw cosine** by a
   clinically meaningful margin that holds across disease-area strata. E′ is optional and
   non-gating under the agreed B-0 scope, so this condition may simply never be evaluated. Note also
   that the comparison tests *this repository's* SP term, `1/(1+mean(d))`, which is not the paper's
   Eq 13 — so a result in either direction is a statement about this implementation, not about the
   paper's formula.
2. **The paper-parity retraining track is undertaken**, changing the trained scorer. Statement 2 is
   interim precisely because of this.
3. **The knowledge graph changes materially** in coverage or connectivity, altering what SP measures.
4. **A cascade or tie-break design is evaluated and shown to help** (§4). Evaluating it without a
   further measurement run is conditional on B-0.5 either recording an approved bounded
   per-candidate component artifact or computing the predeclared analysis during that run; the
   implemented B-0 artifacts do not carry those components (rev 4).

---

## Related documents

- [`SP_SCORE_GUIDE.md`](SP_SCORE_GUIDE.md) — what the SP score means clinically, and safe use
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — system architecture; to be reconciled with this record when
  B-1 lands
- [`RETRIEVAL_AND_CANDIDATE_DISCOVERY_FINDINGS.md`](RETRIEVAL_AND_CANDIDATE_DISCOVERY_FINDINGS.md) —
  the findings that opened the candidate-discovery question
