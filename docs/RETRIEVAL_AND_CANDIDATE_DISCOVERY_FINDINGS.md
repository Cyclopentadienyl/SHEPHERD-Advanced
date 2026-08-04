# Findings — candidate discovery, retrieval, and the vector-index subsystem

**Date:** 2026-08 · **Type:** findings report (descriptive) · **Status:** open — no decisions taken

## 0. What this document is, and is not

**Is:** a record of what an architecture review found while cleaning up the configuration layer. It
documents the *current* state of candidate discovery, the retrieval subsystem, and the Mode B
evidence path, with a source citation for every claim, so the knowledge is not lost.

**Is not:** a decision record, a removal plan, or an approved design. Nothing here has been decided,
and no product code has been changed on the basis of these findings. Where a course of action is
implied, it is written as an open question, not a recommendation to execute.

**Why it exists:** the findings below are larger than the cleanup task that surfaced them, and they
touch the core diagnosis flow. Recording them separately keeps a small cleanup from silently
becoming an architectural change, and lets the sequencing be decided deliberately.

### Status vocabulary

A single label cannot express "the code exists and is correct, but nothing ever ran it", which is
the actual state of several items here. Status is therefore recorded along four independent
dimensions:

| Dimension | Values |
|---|---|
| **Implementation** | `IMPLEMENTED` · `PARTIAL` (narrower than its documented design) · `ABSENT` |
| **Wiring** | `WIRED` (on the runtime path) · `OPTIONAL` (behind config/flag) · `NOT WIRED` (nothing constructs it) |
| **Deployment** | `OBSERVED` · `NOT OBSERVED` (in the audited environment) · `UNKNOWN` |
| **Test coverage** | `EXERCISED` (build/load/plumbing covered by tests or observed runs) · `UNEXERCISED` |
| **Semantic verification** | `VERIFIED` (tests assert that the behaviour is *correct*) · `INCOMPLETE` |

Test coverage and semantic verification are kept apart deliberately: the central failure in this
report is code that was **exercised but never semantically verified** — tests ran and passed while
asserting the wrong properties (§2.4). A single "verification" axis would hide exactly that.

`PLANNED` marks a capability documented as intended but not built. `DEFECT` marks behaviour that is
incorrect on its own terms, independent of roadmap intent.

### Claim labels

| Label | Meaning |
|---|---|
| **[FACT]** | Read directly from the cited source |
| **[MEASURED]** | Reproduced by running code, method stated |
| **[INFERENCE]** | The author's interpretation of the evidence — weigh accordingly |
| **[OPEN]** | A design question with no answer taken here |

### Project context (governs how gaps below should be read)

This is a staged build. **Phase 1 deliberately targets reproducing the original SHEPHERD design**;
the broader GNN + PathReasoner collaborative framework — including cross-species and
convergent-pathway reasoning — is the intended end state, not the current one. Several in-code
descriptions therefore document the *planned* architecture rather than present behaviour.

This context explains most gaps below, but it is **not** a blanket excuse: a runtime docstring, an
API field name, or clinician-facing text that misdescribes current behaviour is a defect regardless
of roadmap intent, because it misleads a reader about what the system does today. Findings are
labelled accordingly.

---

## 1. Candidate discovery diverges from the documented architecture

**This is the primary finding. The others are downstream of it.**

### 1.1 What the architecture requires

`docs/ARCHITECTURE.md:26-37` states this as the fundamental design principle, attributed to the
original SHEPHERD paper:

> **GNN generates embeddings and drives all scoring/ranking.**
> **PathReasoner provides explainable evidence paths for clinician review.**
> **PathReasoner NEVER gates candidate discovery** or drives scoring when GNN is available.
>
> If PathReasoner becomes the gatekeeper for candidate discovery, GNN's generalization ability is
> blocked.

The stated purpose is explicit: the GNN's advantage over path-based methods is **inductive
generalization** — inferring relationships for diseases never seen in training, tolerating missing
KG edges, and generalizing to novel phenotype combinations. Gating on graph reachability removes
exactly that advantage. **[FACT]**

> **Source-attribution error in `ARCHITECTURE.md` — `DEFECT`.** The architecture document supports
> this principle with the claim that "83% of test diseases were unseen" in the paper
> (`docs/ARCHITECTURE.md:32`, repeated as "Disease split: 83% of test diseases never seen in
> training" at `:132`). **Primary-source verification performed during review of this report
> confirms that the paper's 83% statistic refers to diseases represented by only one UDN patient,
> not to the percentage of test diseases unseen during training.** The paper does separately discuss
> disease-stratified generalization and novel or sparsely represented conditions, but that is a
> different claim. Both lines in `docs/ARCHITECTURE.md` should be corrected separately; the design
> principle itself does not depend on the number.

### 1.2 What the implementation does — `PARTIAL`

`src/inference/pipeline.py:1097-1105`:

```python
# Step 3: Find reasoning paths (BFS-based)
all_paths = self._find_all_paths(source_ids, include_ortholog_evidence)

# Step 3b: ANN candidate discovery (vector index)
ann_only_candidates: Dict[str, float] = {}
if self._vector_index_ready and self._gnn_ready:
    ...
```

`_score_and_rank_candidates` then scores the union of `all_paths` and `ann_only_candidates`
(`:1343-1400`). Candidates absent from both are never scored.

Because the vector index is optional and off by default (§2), **BFS reachability defines the disease
universe in the default configuration**. A disease with no KG path from the patient's phenotypes is
never scored, regardless of what the GNN embedding would say — which is the situation
`ARCHITECTURE.md:37` warns against.

### 1.3 Interpretation

**[INFERENCE]** Read against the staged-build context, this is most plausibly a Phase-1
implementation that started from the tractable path-based flow and has not yet been inverted to the
GNN-primary flow the architecture specifies. On that reading it is a **roadmap gap**, not a coding
error — but it is a gap in a *documented core principle*, not a peripheral feature, and the doc
currently describes behaviour the runtime does not have.

**Scope note (to prevent over-reading in either direction).** The immediate control-flow divergence
is at candidate discovery. The existing scorer can technically accept non-BFS candidates —
`_find_ann_candidates` already feeds it one — so candidate-source work does **not** automatically
require a full scorer rewrite, and this finding should not be read as "the pipeline must be
rebuilt".

Equally, it should not be read as "only one function changes". Changing the candidate source can
affect SP lookups, the per-candidate fallback path search (§3.2), Mode B workload, explanation cost,
top-k displacement, latency, and clinician-facing output. A paper-parity review may additionally
require changes to patient aggregation and scoring semantics, since the authoritative patient
encoder/scorer is itself unresolved (§4).

The η fusion (`final = η·emb + (1-η)·sp`, η=0.7, `:297,309`) is derived from the paper and is **not
implicated by the candidate-admission defect described here**. Its task-specific calibration and
paper parity are outside this report's scope — noting only that the paper presents the formula for
patient–candidate-gene scoring while the pipeline applies it to disease ranking, score ranges and
calibration differ, and η=0.7 is the project default rather than a demonstrated universal constant.

### 1.4 Open questions (not decided here)

- What is the canonical Phase-1 candidate universe for disease ranking — all valid diseases, or an
  explicitly defined subset?
- Would scoring a larger universe be acceptable in latency and memory terms at ~27,990 diseases
  (`docs/TRAINING_PIPELINE_PLAYBOOK.md:99`)?
- Should PathReasoner run *after* ranking, on the top-K, purely to generate evidence?

---

## 2. The vector-index subsystem

### 2.1 Provenance and current status

`docs/archive/ARCHITECTURE_REVIEW_2026-02-25.md:141` records the component as an addition over the
original design — *original: none; this project: Voyager/cuVS ANN index; rationale: "加速大規模推理"
(accelerate large-scale inference)*.

**Deployment audit result [FACT, one machine]:** on the maintainer's development machine,
`*.voyager` / `*.cuvs` artifacts, `vector_index_path` in `configs/`, and
`SHEPHERD_VECTOR_INDEX_PATH` are **all absent**. Combined with the default
`vector_index_path: Optional[str] = None` (`src/inference/pipeline.py:336`) and the early return
when it is unset (`:937-938`), the subsystem was **not active in that environment**.

Status: Implementation `IMPLEMENTED` · Wiring `OPTIONAL` · Deployment `NOT OBSERVED` (audited
machine) / `UNKNOWN` (elsewhere) · Test coverage `EXERCISED` · Semantic verification `INCOMPLETE`.

The last two matter together: the subsystem *does* have unit build/search tests, an integration load
test, end-to-end ANN plumbing tests, a measured Voyager run (§2.7) and observed cuVS initialisation
failures on DGX. It was exercised. What no test asserted is whether the results *mean* what the
pipeline assumes (§2.2, §2.4).

Note that "off by default" alone would not establish this: `docs/TRAINING_PIPELINE_PLAYBOOK.md:20,154,163`
instructs operators to run `scripts/build_index.py` and point `vector_index_path` at
`$WS/vector_index`, and `src/api/main.py:394` reads `SHEPHERD_VECTOR_INDEX_PATH`. The audit, not the
default, is what establishes non-use — and it covers one machine, not all environments.

### 2.2 Score-conversion defect — `DEFECT`

`VectorIndexBase.search()` documents its return as **distance** (`src/retrieval/backends/base.py:193`),
and `VoyagerIndex._search_impl` returns Voyager's raw distances unmodified
(`src/retrieval/backends/voyager_backend.py:171-181`). The pipeline treats that value as a cosine
similarity (`src/inference/pipeline.py:1026`):

```python
score = (distance + 1.0) / 2.0
```

Measured with 2-D vectors of known dot product against `voyager.Space.InnerProduct`:

| vector | true dot | returned | pipeline's `(v+1)/2` |
|---|---|---|---|
| same direction, ‖v‖=2 | 2 | −1.0 | 0.00 |
| identical | 1 | 0.0 | 0.50 |
| orthogonal | 0 | 1.0 | 1.00 |
| opposite | −1 | 2.0 | 1.50 |

Voyager's InnerProduct space returns `distance = 1 − dot`. Consequences, stated precisely:

- **Score direction is inverted** relative to similarity.
- **Range is unbounded**, not `[0,1]` as the mapping assumes: disease vectors are inserted
  **unnormalised** (`voyager_backend.py:147-161`; `scripts/build_index.py:338` passes no `metric`, so
  the default `"ip"` applies, matching `configs/deployment.yaml:117`), so dot is not bounded to
  `[-1,1]`.
- **The threshold rejects the wrong members.** Voyager returns its top-k in correct native order;
  the faulty conversion then discards the *higher*-dot members of that retrieved set
  (`score >= ann_score_threshold`, default `0.3` at `:338` → rejection when `dot > 1.4`).
- **Normalisation is asymmetric:** the ANN path computes `normalised_patient · raw_disease`
  (patient normalised at `:1016`), whereas `_calculate_gnn_score` normalises both sides
  (`:1648-1650`).

**Blast radius is bounded:** the faulty score governs **candidate admission only**.
`_score_and_rank_candidates` discards `ann_score` and recomputes confidence via
`_calculate_combined_score` (`:1414-1420`). The paper-derived η fusion and final ranking are **not**
affected by this defect.

Because the subsystem was not active on the audited machine (§2.1), **diagnosis runs from that
audited environment were not affected by this admission defect. Impact on other or historical
deployments remains unverified.**

Follow-on observation: since the mapping's values are discarded after admission,
`ann_only_candidates` is arguably better modelled as a set of IDs than a dict.

### 2.3 Underlying contract defect — `DEFECT`

`VectorIndexBase` exposes a raw backend "distance" with no defined direction, range, or
normalisation, and Voyager (HNSW, `voyager_backend.py:5,55`) and cuVS (IVF-Flat/IVF-PQ,
`cuvs_backend.py:21,85-88`) need not share native output semantics. **No single pipeline-level
conversion can therefore be correct for both backends.** Any future retrieval interface should define
the similarity contract explicitly rather than reinterpreting backend-specific values.

### 2.4 Why this went undetected

Five factors compound, and the last is the root cause:

1. **Off by default** (`:336`, `:937-938`) — the code path does not run unless explicitly configured.
2. **Load failures are non-fatal and weakly surfaced** — `except Exception: logger.warning(...)`
   (`:967-968`). Observable via `vector_index_ready` in `get_pipeline_config()` (`:1825`) and the
   status API (`src/api/routes/pipeline.py:51`), but not raised as an operational error.
3. **Absence is invisible in output** — without the source, all BFS+GNN+SP candidates are still
   present and correctly ranked, so results look normal.
4. **Wrong scores are camouflaged by the feature's own framing** — these candidates are described as
   "potential novel associations" (`:1102-1103`); a badly-ranked novel association is
   indistinguishable from a correct one without ground truth.
5. **No test asserts score semantics.** Unit tests exercise `VoyagerIndex` build/search directly. The
   integration test asserts only that the index *loads*
   (`tests/integration/test_pipeline.py:354-359`). The end-to-end ANN test sets
   `ann_score_threshold=0.0  # Accept all ANN results for testing` (`:370,400`), disabling the one
   check that could have exposed the inversion. A repo-wide search finds **no assertion on ANN score
   values, ordering, or threshold behaviour**.

This is the same pattern found elsewhere in this review cycle (the silent `/diagnose` mock fallback,
the silent `num_neighbors` default, a `cuvs_available` test gate that did not check `cupy`, and
`resolve_backend`'s comment claiming a verification it never performs): **coverage verified that
something ran, not that it ran correctly.**

### 2.5 Deployment reporting is inaccurate — `DEFECT`

`deploy.sh` prints "cuVS installed (GPU vector backend available)" after installing `cuvs-cu13`, and
`scripts/validate_installation.py` checks only `import cuvs`. But `CuVSIndex._validate_cuvs`
(`src/retrieval/backends/cuvs_backend.py:127`) also imports `cupy`, which `cuvs-cu13` does not
declare as a dependency and `deploy.sh` does not install. On the DGX Spark machines cuVS therefore
imports but cannot construct. Reporting a backend as available should require a construct/build/search
probe, not a module import.

### 2.6 Global indexing config is never merged into the backend call — `DEFECT`

`scripts/build_index.py:305-311` builds the backend arguments from the **backend-specific
subsection only**:

```python
indexing_cfg = deploy_config.get("indexing", {})
backend_config = indexing_cfg.get(resolved, {})     # e.g. indexing.voyager
...
index = create_index(backend=backend, dim=hidden_dim, **backend_config)   # :338
```

The **parent-level** `indexing.metric` and `indexing.dim` in `configs/deployment.yaml:113-117` are
never merged into that call. Consequences:

- `metric` is currently `"ip"` at runtime **because `create_index()` defaults to `"ip"`**, not
  because the YAML value was consumed. Changing the YAML to `cosine` or `l2` would have **no
  effect** — a silently ignored configuration knob.
- `dim` likewise comes from the checkpoint's `hidden_dim`, not from `indexing.dim: 768`.

This is the same failure class as §2.4: a setting that appears to work only because two independent
defaults happen to agree.

### 2.7 Measured performance (context only — not decision-grade)

27,990 × 256, top-50, CPU, mean of 20 runs; NumPy exact search with the normalised matrix cached vs
Voyager HNSW (`M=32, ef_construction=200, ef_search=64`): exact **0.958 ms/query**, Voyager
**0.514 ms/query**, Voyager build 10.36 s.

**[MEASURED]** In this specific uncontrolled synthetic run, Voyager measured **~1.9× faster**. This
**falsifies the earlier universal assumption** that exact search must always win below ~10⁶ vectors;
it does **not** establish production superiority.

This measurement is **not sufficient to decide anything**. Missing: recall@k against exact ground
truth (an ANN latency without recall is uninterpretable), a latency-vs-recall curve over `ef_search`,
controlled thread counts (Voyager's backend default is `num_threads=-1`; NumPy BLAS threading was
also uncontrolled), p50/p95 rather than a mean, Torch-CPU and Torch-CUDA baselines (the deployed path
holds embeddings on CPU as Torch tensors, `:918`), real GNN vectors rather than Gaussian noise, and
batch/concurrent behaviour.

---

## 3. Mode B analogy evidence

### 3.1 Purpose

`EvidencePanel` (`src/reasoning/evidence_panel.py:14-27`) implements two evidence modes. **Mode A**
surfaces direct KG paths. **Mode B** exists for the zero-shot case: when the GNN ranks a candidate
highly but the KG has no direct path, present the paths of a *similar* node as analogy evidence, so
that a clinician has something inspectable rather than a bare score. Every package carries a
confidence label, with `INSUFFICIENT` when neither mode succeeds.

The purpose is sound and directly serves the project's explainability requirement: it extends
inspectability to precisely the cases where path-based evidence is unavailable — which are the cases
the GNN's generalization is meant to surface.

### 3.2 Implementation is narrower than its description — `PARTIAL`

Three differences from the in-code description:

1. **Same-type, not cross-type.** `_find_analogies()` derives the search pool from the candidate's own
   node type (`:350-352`): `target_type_str = target_node.node_type.value;
   target_emb_pool = node_embeddings.get(target_type_str)`. A disease candidate is therefore compared
   against **other diseases**, not genes — while the results are stored in fields named
   `similar_gene_id` / `similar_gene_name`. The description of a gene bridge is `PLANNED`; the
   implementation is disease↔disease analogy.
   **The mismatched field names are a `DEFECT` regardless of roadmap intent**, because any consumer
   or clinician-facing text derived from them is misinformed.
2. **K nearest, not K path-qualified.** `k = min(analogy_top_k + 1, sims.size(0))` (`:369`), then the
   remainder are checked for paths, stopping at `analogy_top_k` (`:411`). With `analogy_top_k=3`
   (`src/inference/pipeline.py:506`), if the three nearest have no path but the fourth does, the
   fourth is never examined and the result is `INSUFFICIENT`. The description ("find K nearest known
   nodes that DO have paths") implies progressive search to a scan cap; the code inspects a fixed
   window.
3. **Trigger chain has an extra step.** `build_evidence()` (`:193-224`) tries Mode A first, and when
   `existing_paths` is `None` it runs a **fresh targeted path search** (`:220`). Mode B runs only if
   that *also* fails. Since the initial global enumeration is bounded by path-count/traversal limits,
   some candidates absent from `all_paths` are merely omitted from the first enumeration rather than
   genuinely disconnected. Three states should be reported separately: absent from the initial bounded
   enumeration; no path from the targeted search; genuinely disconnected.

### 3.3 Reachability in the current pipeline

Mode B fires only for candidates reaching `_add_explanations` with empty paths
(`existing_paths=paths if paths else None`, `:1730`; `paths = all_paths.get(disease_key, [])`,
`:1711`). BFS candidates always have paths (`if not paths: continue`, `:1344-1345`), so within the
current `DiagnosisPipeline.run()` flow the only such candidates are the ANN-discovered ones, which
require `_vector_index_ready` (`:1105`). It also requires `include_explanations` (`:1137`).

**Therefore, in an environment that never built an index, Mode B has never executed.**

This is a statement about the *current pipeline trigger*, not an architectural dependency:
`EvidencePanel` is a public component that accepts a no-path candidate plus embeddings directly, so
Mode B can be driven by any future candidate source. Removing the ANN source would remove today's
trigger; it would not require deleting the capability.

### 3.4 Cross-type analogy would need a trained contract

Equal hidden dimensions do not make disease, gene, pathway, human, and model-organism embeddings
directly comparable by cosine. A gene bridge or cross-species analogy would require an explicit
training objective, a projection head or typed decoder, calibration, and relation coverage — not
merely widening the search pool. This is worth recording so the `PLANNED` gene-bridge semantics are
not mistaken for a small change.

---

## 4. Components defined but not wired

| Component | Status | Evidence |
|---|---|---|
| `PhenotypeDiseaseMatcher` (`src/models/gnn/shepherd_gnn.py:314`) | `NOT WIRED` | No construction call anywhere in the repo (searched for `PhenotypeDiseaseMatcher(`); referenced only in imports and `__all__` |

This matters because the class carries a **learned phenotype aggregator** and supports
cosine / bilinear / MLP scoring (`:354`, `:442`), whereas the live inference path uses unweighted
mean pooling (`src/inference/pipeline.py:1630-1631`). If paper parity requires learned or
attention-based patient aggregation, this class may be the intended home for it, currently unused.

**Consequence for any future work:** treating the current mean-pooled cosine as a "correctness
oracle" would freeze the simplified behaviour rather than reproduce the paper. The authoritative
patient encoder and scorer must be established first.

---

## 5. Retrieval as a concern is not redundant

The vector-index *implementation* is separable from the retrieval *concern*. The architecture
documents retrieval-shaped capabilities that are planned and unbuilt:

| Capability | Status | Evidence |
|---|---|---|
| Similar Patient Retrieval (Patients-Like-Me) | `PLANNED` | `docs/ARCHITECTURE.md:92-96` (§3.2, cosine between patient embedding and all known patient embeddings); `docs/Repair/REPAIR_CHECKLIST.md:161` unchecked; `docs/MILESTONE_REPORT.md:157` 規劃中 |
| Novel Disease Characterization | `PLANNED` | `docs/ARCHITECTURE.md:97+` (§3.3) |
| Free-text → HPO mapping | `PLANNED` | `docs/ARCHITECTURE.md:19`; `src/nlp/` reserved |

Patients-Like-Me is a core task of the original work, not a speculative extension. Its eventual design
needs a patient embedding store, representation/checkpoint versioning, cohort metadata, privacy
boundaries, and access control — none of which the current disease `.voyager`/`.cuvs` artifact
provides. **The conclusion this supports is that the concrete backends and the retrieval boundary are
separate questions**, and that removing the former should not be conflated with removing the latter.

Note also that free-text → HPO mapping would operate in a *different* vector space (text-encoder
embeddings) over a *different* entity set (~19,389 phenotypes), so it could not reuse the current
disease-GNN artifacts.

---

## 6. Summary: defects vs roadmap gaps

Separated deliberately, because the two require different responses.

**A. Concrete defects** (incorrect on their own terms, independent of roadmap):
1. Voyager distance interpreted as similarity — inverted score, unbounded range, wrong threshold
   rejection (§2.2).
2. Backend score contract underspecified — no defined direction/range/normalisation (§2.3).
3. Global `indexing.metric` / `indexing.dim` never merged into `create_index()`; the metric appears
   correct only because two defaults coincide (§2.6).
4. Disease IDs stored in `similar_gene_*` fields (§3.2).
5. Deployment reports cuVS "available" after an import check the backend does not actually satisfy
   (§2.5).
6. Runtime documentation describing planned rather than current behaviour, without status markers
   (§1.1 vs §1.2; §3.1 vs §3.2).
7. `docs/ARCHITECTURE.md:32` and `:132` misattribute the paper's 83% statistic — it refers to
   diseases with a single UDN patient, not to unseen test diseases (§1.1).
8. No tests covering candidate-admission semantics (§2.4).

**B. Phase-1 paper-parity gaps** (in scope for reproducing the original design; not yet reached):
1. GNN-primary candidate discovery (§1).
2. Authoritative patient encoder/scorer — learned aggregation via `PhenotypeDiseaseMatcher` (§4).
3. Patients-Like-Me retrieval (§5).
4. Novel Disease Characterization (§5).

**C. Future hospital-driven extensions** (explicitly beyond Phase 1; must not be treated as
immediate work):
1. Gene / cross-species analogy bridge (§3.2, §3.4).
2. Convergent-pathway reasoning across species.
3. Free-text → HPO mapping (§5).

**D. Experimental / unratified** (exists or proposed, but no approved product requirement):
1. Current Mode B same-node-type disease analogy (§3.2) — useful, but its provenance and product
   requirement are undocumented; it should not drive Phase-1 architecture.
2. Progressive path-qualified analogy search (§3.2).

The B/C/D split matters operationally: a later session reading only the defect list could otherwise
treat cross-species reasoning as immediate Phase-1 work. It is not.

## 7. Open questions requiring a decision (none taken here)

1. **Candidate universe** — what is the canonical Phase-1 disease set for ranking, and does
   PathReasoner move to a post-ranking evidence role (§1.4)?
2. **Authoritative patient encoder/scorer** — mean-pooled cosine, or `PhenotypeDiseaseMatcher`'s
   learned aggregation (§4)?
3. **Analogy semantics** — disease↔disease, disease↔gene, or gene↔gene; and align field names,
   clinician-facing text, and tests to whichever is chosen (§3.2).
4. **BFS-unreached candidates in clinician-facing output** — should a candidate with no direct
   reasoning path (but possibly SP connectivity and analogy evidence) be ranked and shown? Note this
   is *not* the same as the backend question.
5. **Concrete backends** — retain, replace, or remove Voyager/cuVS and the persisted disease index,
   given §2 and the separation established in §5.

Any change to §7.4's answer is a **product behaviour change**, not a refactor: the current source is
opt-in via `vector_index_path`, so making an equivalent source unconditional would turn a default-off
feature into an always-on one, altering default candidate sets, top-k composition, evidence modes,
latency, and API output. Such a change should be gated (default off) and preferably evaluated in a
shadow mode that logs without affecting clinician-facing ranking.

## 8. Method

**Source-derived facts are cited; interpretations, hypotheses and design implications are labelled
or phrased explicitly as such.** This report does contain judgements — for example that the candidate
divergence is "most plausibly" a staged-build artefact (§1.3), that Mode B's purpose "is sound"
(§3.1), and that `PhenotypeDiseaseMatcher` "may be the intended home" for learned aggregation (§4).
These are marked **[INFERENCE]** or worded to signal uncertainty; they should be weighed as opinion,
not treated as established fact.

Two claims are **[MEASURED]** and reproducible as described: the Voyager distance semantics (§2.2,
2-D vectors of known dot product) and the latency comparison (§2.7, synthetic Gaussian vectors at the
production shape, threads uncontrolled, recall unmeasured).

The deployment audit (§2.1) covers **one development machine** and does not generalise to other or
historical environments.

The paper-attribution finding in §1.1 rests on primary-source verification performed during review of
this report, not on a reading by this report's author.
