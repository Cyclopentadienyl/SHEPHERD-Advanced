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

Applied to capabilities throughout:

| Label | Meaning |
|---|---|
| `IMPLEMENTED` | Built, wired into the runtime path, exercised |
| `PARTIAL` | Built, wired, but narrower than the documented design |
| `NOT WIRED` | Code exists; nothing constructs or calls it |
| `PLANNED` | Documented as intended; not built |
| `DEFECT` | Implemented behaviour is incorrect on its own terms |

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
generalization** — inferring relationships for diseases never seen in training (the doc notes 83% of
the paper's test diseases were unseen), tolerating missing KG edges, and generalizing to novel
phenotype combinations. Gating on graph reachability removes exactly that advantage.

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

Read against the staged-build context, this is most plausibly a Phase-1 implementation that started
from the tractable path-based flow and has not yet been inverted to the GNN-primary flow the
architecture specifies. It is a **roadmap gap**, not a coding error — but it is a gap in a
*documented core principle*, not a peripheral feature, and the doc currently describes behaviour the
runtime does not have.

**Scope note (to prevent over-reading):** the divergence is confined to *which candidates get
scored*. Everything downstream — GNN scoring, SP scoring, the η fusion
(`final = η·emb + (1-η)·sp`, η=0.7, `:297,309`), explanation generation, and the evidence panel — is
independent of candidate provenance and unaffected. Changing the candidate source changes the input
set to the scoring stage; it does not require reworking the scoring stage. `_find_ann_candidates`
already demonstrates that a non-BFS candidate source can feed the same scorer.

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

**Deployment audit result:** on the maintainer's development machine, `*.voyager` / `*.cuvs`
artifacts, `vector_index_path` in `configs/`, and `SHEPHERD_VECTOR_INDEX_PATH` are **all absent**.
Combined with the default `vector_index_path: Optional[str] = None`
(`src/inference/pipeline.py:336`) and the early return when it is unset (`:937-938`), the subsystem
has **never been active** in that environment. `IMPLEMENTED` as code; not exercised.

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
affected by this defect. Because the subsystem was never active (§2.1), no diagnosis result has been
affected either.

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

### 2.6 Measured performance (context only — not decision-grade)

27,990 × 256, top-50, CPU, mean of 20 runs; NumPy exact search with the normalised matrix cached vs
Voyager HNSW (`M=32, ef_construction=200, ef_search=64`): exact **0.958 ms/query**, Voyager
**0.514 ms/query**, Voyager build 10.36 s.

**Voyager is ~1.9× faster at this scale.** Recorded to correct an earlier assumption that exact
search would win here — it does not.

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

**Concrete defects** (incorrect on their own terms, independent of roadmap):
1. Voyager distance interpreted as similarity — inverted score, unbounded range, wrong threshold
   rejection (§2.2).
2. Backend score contract underspecified — no defined direction/range/normalisation (§2.3).
3. Disease IDs stored in `similar_gene_*` fields (§3.2).
4. Deployment reports cuVS "available" after an import check the backend does not actually satisfy
   (§2.5).
5. Runtime documentation describing planned rather than current behaviour, without status markers
   (§1.1 vs §1.2; §3.1 vs §3.2).
6. No tests covering candidate-admission semantics (§2.4).

**Roadmap gaps** (Phase-1 scope not yet reached; not errors):
1. GNN-primary candidate discovery (§1).
2. Gene / cross-species analogy bridge (§3.2, §3.4).
3. Progressive path-qualified analogy search (§3.2).
4. Patients-Like-Me and Novel Disease Characterization (§5).
5. Learned patient aggregation via `PhenotypeDiseaseMatcher` (§4).

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

Claims were verified by reading the cited source. Two claims are measurements, reproducible as
described: the Voyager distance semantics (§2.2, 2-D vectors with known dot products) and the
latency comparison (§2.6, synthetic Gaussian vectors at the production shape). The deployment audit
(§2.1) covers one development machine. Everything else in this document is a source citation, not an
inference.
