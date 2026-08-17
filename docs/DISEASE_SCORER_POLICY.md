# Disease Scorer Policy — Decision Record

**Status:** Accepted (2026-08) by the deploying institution. **Not yet implemented** — see the
status table in §2. Implementation is work item B-1, which remains gated (§6).

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
4. **SP cannot admit, exclude, filter, reorder, or rescore disease candidates.**
5. **SP may be exposed only as an optional post-ranking contextual analysis** over candidates
   already selected by the disease scorer.
6. **Numeric SP analysis and typed evidence-path traversal are separate concerns** and must not be
   merged into a single mechanism or a single UI affordance.
7. **Path-reasoning fallback uses separately discriminated score semantics** and cannot be mixed
   into a GNN-ranked result.

Statement 2 is explicitly **interim**. It reflects what the deployed checkpoint's training objective
optimised, not the reference paper's disease scorer (§3.1). It is expected to change if and when the
paper-parity retraining track is undertaken, which is outside work item B.

---

## 2. Status — current behaviour vs approved target

**The gap between these columns is the point of this table.** Nothing in the "approved target"
column is implemented.

| Concern | Current behaviour | Approved target | Implementation status |
|---|---|---|---|
| **Candidate universe** | BFS path discovery gates scoring. Two order-dependent caps truncate it: `max_paths_per_source = 100` (`src/reasoning/path_reasoning.py:105`) and `max_genes = 100` per phenotype (`:507`), both applied to unordered neighbour lists | Every disease in the configured KG universe is scored; no discovery gate | **Not implemented** — work item B-1, gated |
| **Disease score** | `0.7 × ((cos+1)/2) + 0.3 × SP` when a checkpoint and `shortest_paths.pt` are both loaded (`src/inference/pipeline.py:1310`) | Raw cosine (interim, statement 2) | **Not implemented** — B-1 |
| **SP role** | 30% of the ranking score, and therefore able to reorder candidates | Optional post-ranking contextual analysis only; separate field or panel; no effect on identity, order, rank or ranking score | **Not implemented** — B-1 |
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

Classified by kind, because the four kinds carry different weight and are verifiable in different
ways.

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
| The deployed checkpoint contains no task-head parameters — `load_state_dict` is strict and the load succeeds | `src/inference/pipeline.py:831`; `src/models/gnn/shepherd_gnn.py:96-173` |
| η defaults to 0.7 and is a project choice, not a paper value | `src/inference/pipeline.py:296-309` |
| The SP transform is absolute, `1/(1 + mean(d))`, with range `[1/7, 1/2]` under `max_hops = 5` | `src/inference/pipeline.py:1361, 1380` |
| Candidate discovery is BFS-gated and truncated in traversal order | `src/reasoning/path_reasoning.py:105, 275-276, 507` |

### 3.3 Institutional decision

The deploying institution decided (2026-08) that the pipeline should reproduce the original
SHEPHERD design as the primary direction, deviating only where a design is demonstrably better and
supported by citable evidence; that disease ranking is GNN-primary; and that SP is removed from
ranking and re-exposed as an optional, clearly separated analysis.

**Provenance matters here and is recorded deliberately.** This decision was taken **before** the B-0
measurements were run, and it is consistent with the pre-declared decision rule that raw cosine is
the default and that η must *earn* adoption with evidence. It was not derived from reading results.
A future reader should not mistake it for an empirical finding, nor reopen it as though it were an
unexamined default.

### 3.4 Engineering inference — weigh accordingly, none is measured

- Over the full disease universe most candidates fall outside the 5-hop table and receive the same
  floor value, so the SP term degenerates towards a binary reachability indicator. That
  systematically disadvantages candidates with no KG path — the cases GNN generalisation exists to
  surface.
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
| **Keep η in ranking, applied to the full universe** | Requires SP over ~27,990 candidates, where the term degenerates (§3.4) and the paper's candidate-relative normalisation has no meaningful reference set. No paper support for the disease task. |
| **Keep η in ranking, applied to a GNN top-N cut** | Reintroduces a candidate gate — softer than BFS, but still able to hide a candidate the GNN ranked highly. Adds an N-selection problem (recall, top-k set and rank preservation, latency) that pure cosine does not have. |
| **Cascade: use SP only to break ties among candidates the GNN cannot separate** | The most defensible of the alternatives — it reconstructs the paper's short-list condition. Deferred rather than rejected: it needs an operational definition of "cannot separate", which is an uncalibrated threshold, and it is a deviation from the paper requiring its own evidence. It can be evaluated offline from B-0's recorded per-candidate score components without a further run. |
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
- **A separate SP analysis over the displayed top-k reconstructs the paper's short-list condition**,
  and makes candidate-relative normalisation well defined on that set — which the current absolute
  transform is not. How to implement that normalisation is deferred to the implementation stage.
- **Two implementation defects are recorded and deferred**, not fixed by this decision: the range
  mismatch between the two score terms, and the absence of candidate-relative normalisation
  (§3.4, and `SP_SCORE_GUIDE.md` §3).

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
- B-1 and B-2 both modify the candidate and result types. **One discriminated result envelope is
  agreed before either lands**; whichever lands first carries the whole envelope with the other's
  fields present and unpopulated.
- The clinician-facing guide's status box and the SP presentation change together with the code, not
  before it.

---

## 8. Conditions for revisiting

This record should be reopened if any of the following occurs:

1. **B-0's C-vs-E′ comparison shows the η mixture beating raw cosine** by a clinically meaningful
   margin that holds across disease-area strata. Note that comparison tests *this repository's* SP
   term, `1/(1+mean(d))`, which is not the paper's Eq 13 — so a result in either direction is a
   statement about this implementation, not about the paper's formula.
2. **The paper-parity retraining track is undertaken**, changing the trained scorer. Statement 2 is
   interim precisely because of this.
3. **The knowledge graph changes materially** in coverage or connectivity, altering what SP measures.
4. **A cascade or tie-break design is evaluated and shown to help** (§4), which B-0's recorded
   per-candidate score components make possible without a further measurement run.

---

## Related documents

- [`SP_SCORE_GUIDE.md`](SP_SCORE_GUIDE.md) — what the SP score means clinically, and safe use
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — system architecture; to be reconciled with this record when
  B-1 lands
- [`RETRIEVAL_AND_CANDIDATE_DISCOVERY_FINDINGS.md`](RETRIEVAL_AND_CANDIDATE_DISCOVERY_FINDINGS.md) —
  the findings that opened the candidate-discovery question
