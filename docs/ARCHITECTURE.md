# SHEPHERD-Advanced System Architecture

> **Last Updated**: 2026-03-24
> **Status**: Authoritative reference — all development must align with this document.
> **Anchor**: SHEPHERD (npj Digital Medicine 2025) by Harvard MIMS Lab.

---

## 1. System Identity

SHEPHERD-Advanced is a rare disease diagnosis system that:
- Uses a heterogeneous knowledge graph (PrimeKG-based) and GNN to generate node embeddings
- Ranks candidate diseases/genes by embedding distance in a unified vector space
- Provides explainable evidence paths via PathReasoner (auxiliary)
- Supports three diagnostic tasks in a single unified framework

**Hardware targets**: Blackwell GPU / PyTorch 2.8+ / CUDA 13.0
**Architecture upgrade**: HGT (Heterogeneous Graph Transformer) replacing GAT
**Extension**: Fuzzy input support (free-text to HPO/MONDO mapping)

---

## 2. Core Architectural Principle: GNN-Primary

> **Status note (2026-08).** The disease scorer policy — what ranks disease candidates, the role of
> the shortest-path signal, and the role of path reasoning — is owned by
> [`DISEASE_SCORER_POLICY.md`](DISEASE_SCORER_POLICY.md), which also records **which parts of this
> section the runtime does not yet implement**. In particular, candidate discovery is currently
> gated by path search and the ranking score currently includes a shortest-path term; §2.1 below
> describes the approved target, not present behaviour. This section will be reconciled with the
> implementation when work item B-1 lands.

### 2.1 The Non-Negotiable Rule

> **GNN generates embeddings and drives all scoring/ranking.**
> **PathReasoner provides explainable evidence paths for clinician review.**
> **PathReasoner NEVER gates candidate discovery or drives scoring when GNN is available.**

This is the fundamental design principle derived from the original SHEPHERD paper. The GNN's key advantage over traditional path-based methods is **inductive generalization** — the ability to:
- Infer relationships for diseases never seen in training
- Handle missing edges in the knowledge graph via learned neighborhood structure
- Automatically learn which edge types and multi-hop signals are diagnostically relevant
- Generalize to novel phenotype combinations

If PathReasoner becomes the gatekeeper for candidate discovery, GNN's generalization ability is blocked.

### 2.2 Correct Flow

```
Patient Phenotypes (HPO terms)
    │
    ▼
┌─────────────────────────────────────────────┐
│  GNN Forward Pass (precomputed at init)      │
│  All node embeddings cached                  │
└──────────────────┬──────────────────────────┘
                   │
    ┌──────────────▼──────────────────┐
    │  Patient Embedding Generation    │
    │  Aggregate phenotype embeddings  │
    │  via attention/mean pooling      │
    └──────────────┬──────────────────┘
                   │
    ┌──────────────▼──────────────────┐
    │  Distance Computation            │
    │  Patient embedding vs ALL        │
    │  disease/gene embeddings         │
    │  → Cosine similarity ranking     │
    └──────────────┬──────────────────┘
                   │
    ┌──────────────▼──────────────────┐
    │  Top-K Results                   │
    │  + PathReasoner evidence paths   │
    │  + Attention weight explanation  │
    │  + Patients-Like-Me retrieval    │
    └─────────────────────────────────┘
```

### 2.3 What PathReasoner Does and Does NOT Do

| PathReasoner DOES | PathReasoner Does NOT |
|---|---|
| Find BFS paths between phenotypes and top-K diseases for explanation | Gate which diseases can be scored |
| Provide human-readable evidence chains for clinicians | Contribute to confidence score when GNN is active |
| Supply shortest path length as optional supplementary feature | Act as first-pass candidate filter |
| Serve as scoring fallback when no trained GNN model exists | Override GNN ranking |

---

## 3. Three Diagnostic Tasks

All three tasks share the same GNN-generated embedding space:

### 3.1 Causal Gene Discovery
- **Input**: Patient phenotypes (Channel 1) + Candidate gene list (Channel 2)
- **Method**: Cosine similarity between patient embedding and each candidate gene embedding
- **Output**: Ranked gene list with confidence scores
- **Channel 2 source**: Upstream variant filtering pipeline (e.g., Exomiser) — SHEPHERD is the consumer, not producer

### 3.2 Similar Patient Retrieval (Patients-Like-Me)
- **Input**: Patient phenotypes only (Channel 1)
- **Method**: Cosine similarity between patient embedding and all known patient embeddings
- **Output**: K most similar confirmed patients with shared phenotypes
- **Clinical value**: Diagnosis validation, especially before ordering genetic testing

### 3.3 Novel Disease Characterization
- **Input**: Patient phenotypes only (Channel 1)
- **Method**: Cosine similarity between patient embedding and all disease embeddings
- **Output**: "Most similar known diseases" ranking
- **Clinical value**: When the patient may have an undescribed condition

---

## 4. Two-Channel Input Design

| Channel | Content | Format | Required For |
|---------|---------|--------|-------------|
| **Channel 1** (Phenotypes) | Observed symptoms/features | HPO term IDs | ALL tasks (always required) |
| **Channel 2** (Candidate Genes) | Sequencing-derived suspects | Ensembl Gene IDs | Causal gene discovery only |

The two channels are processed independently inside the model and merged only at scoring time:
- Channel 1 determines the patient embedding direction
- Channel 2 determines comparison targets
- Channel 2 does NOT alter the patient embedding

---

## 5. Two-Step Training

### Step 1: Knowledge Graph Pretraining (Self-Supervised)
- **Task**: Link prediction on KG (DistMult scoring)
- **Purpose**: Learn meaningful embeddings for all 129K+ nodes
- **Data**: KG edges only (no patient data needed)
- **Run once**, then freeze or fine-tune

### Step 2: Diagnosis Fine-Tuning (Metric Learning)
- **Task**: Train patient embeddings to be close to causal genes/diseases, far from others
- **Data**: ~40K simulated patients (zero real patient data required)
- **Loss**: DiagnosisLoss (weight 1.0) + ContrastiveLoss (0.3) + LinkPrediction (0.5) + OrthologConsistency (0.2)
- **Disease split**: held-out diseases are unseen during training (disease-stratified evaluation)

### Training → Inference Bridge
1. Training produces model checkpoint
2. Inference loads checkpoint → runs GNN forward pass → caches all node embeddings
3. `_gnn_ready = True` → GNN scoring activates
4. Without checkpoint: system falls back to PathReasoner-only (degraded mode)

---

## 6. Module Architecture

```
Layer 0 (Foundation):  src/core/          types, protocols, schema
Layer 1 (Config):      src/config/        model-type constants, reserved deployment validation
Layer 2 (Data):        src/ontology/      HPO/MONDO/GO via pronto
                       src/data_sources/  ortholog, pubmed (Phase 2)
Layer 3 (Knowledge):   src/kg/            graph, builder, data_loader
                       src/nlp/           fuzzy input (NOT YET IMPLEMENTED)
Layer 4 (Models):      src/models/        GNN, encoders, decoders, attention
                       src/retrieval/     vector index (cuVS/Voyager)
Layer 5 (Reasoning):   src/reasoning/     path_reasoning (auxiliary), explanation
Layer 6 (Training):    src/training/      trainer, loss_functions
Layer 7 (Inference):   src/inference/      pipeline (GNN-primary scoring)
Layer 7b (Evaluation): src/evaluation/    Mode C measurement, test cohort registry, R10, evaluation ledger
Layer 8 (Interface):   src/api/           FastAPI REST endpoints
                       src/webui/         Gradio dashboard
```

**Import rules** (enforced by import-linter):
- Lower layers cannot import higher layers; `src/evaluation` sits above `src/inference` and below
  the interface (`.import-linter.ini`)
- `src/webui` cannot directly import `src/training` (use subprocess/API)

---

## 7. Key Implementation Files

| Component | File | Key Responsibility |
|-----------|------|-------------------|
| GNN Model | `src/models/gnn/shepherd_gnn.py` | Heterogeneous message passing, node embedding generation |
| GNN Layers | `src/models/gnn/layers.py` | HeteroGNNLayer + OrthologGate |
| Diagnosis Head | `src/models/decoders/heads.py` | Similarity-based disease ranking |
| Loss Functions | `src/training/loss_functions.py` | Multi-task loss (diagnosis, link pred, contrastive, ortholog) |
| Trainer | `src/training/trainer.py` | Training loop with mixed precision |
| Inference Pipeline | `src/inference/pipeline.py` | GNN-primary scoring, PathReasoner explanation |
| PathReasoner | `src/reasoning/path_reasoning.py` | BFS evidence path enumeration |
| Knowledge Graph | `src/kg/graph.py` | KG data structure with PyG export |
| Vector Index | `src/retrieval/vector_index.py` | Standalone ANN index. **Detached from diagnosis** — retained for planned vector-mapping work, not on any inference path |
| Measurement (Mode C) | `src/evaluation/measurement.py`; CLI `scripts/measure_scorer.py` | The formal Test's measurement: every disease ranked by the model's raw cosine, on a registered test cohort version or a generated split |
| Test cohort registry | `src/evaluation/cohort_registry.py`; CLI `scripts/import_cohort.py` | Immutable cohort versions mapped to one graph and bound to its digests, kept outside the repository (provenance contract M4) |
| R10 | `src/evaluation/r10.py` | A test cohort's disease and phenotype-set overlap with the training and validation inputs a checkpoint records, found by digest |
| Evaluation ledger | `src/evaluation/sidecar.py`, `src/evaluation/registration.py`; CLI `scripts/record_evaluation.py` | Test results beside the checkpoint, keyed by its digest (ledger schema 3); one registration function for `--record` and `record_evaluation` |
| Recorded results view | `src/api/routes/evaluations.py` | Read-only `GET /api/v1/evaluations/loaded-model`: the loaded checkpoint's records, shown on the Diagnosis tab's model status |

---

## 8. Scoring Formula

Per the original SHEPHERD paper, the final score combines two signals:

```
final_score = η × embedding_similarity + (1 - η) × shortest_path_similarity
```

### Signal 1: GNN Embedding Similarity (implemented)
```
patient_embedding = mean_pool(GNN_embeddings[phenotype_nodes])
patient_norm = L2_normalize(patient_embedding)
disease_norm  = L2_normalize(GNN_embedding[disease_node])

embedding_similarity = (dot(patient_norm, disease_norm) + 1.0) / 2.0
```
This formula is identical in training (`trainer.py:704-707`) and inference (`pipeline.py:1134-1140`).

### Signal 2: Shortest Path Similarity (Step B — implemented)
```
sp_lengths = [SP_lookup[(phenotype_idx, target_idx)] for phenotype in patient]
# Phenotypes with no path within max_hops contribute (max_hops + 1)
avg_sp = mean(sp_lengths)
shortest_path_similarity = 1.0 / (1.0 + avg_sp)
```
- Pre-computed offline by `scripts/compute_shortest_paths.py`
- Stored as parallel int64/int8 tensors (`shortest_paths.pt`) alongside `node_features.pt`
- Loaded automatically by `pipeline._load_shortest_paths()` if present
- Default `max_hops = 5`; pairs beyond this are treated as unreachable
- Acts as a deterministic fallback when GNN embeddings are unreliable (e.g., sparse phenotype input)

### Mixing parameter η (`PipelineConfig.eta`)
- **Default**: `eta = 0.7` (70% GNN, 30% shortest path)
- `eta = 1.0` → pure GNN
- `eta = 0.0` → pure shortest path
- Tunable via validation set

### Graceful degradation
Every result states which of these applies in its **served identity** (provenance contract M3c): the checkpoint, graph, workspace manifest, SP table and dataset record that produced it, by digest or identifier, the scoring mode and the settings in effect, each part with its state (`verified`, `absent`, …). `get_pipeline_config()` and `GET /api/v1/pipeline/status` report the same value for the loaded pipeline.
- **`shortest_paths.pt` missing + `sp_optional=True`** (default): pipeline reports `scoring_mode="gnn_only"`, effective η = 1.0; the served identity states the SP table `absent`, `gnn_only`, eta 1.0
- **`shortest_paths.pt` missing + `sp_optional=False`**: the build is refused (`PipelineBuildError`); no pipeline is published
- **A model requested (checkpoint or pre-loaded model) and not built** — missing or unreadable checkpoint, one that does not build over the graph, no graph data, no PyTorch, or a device the host cannot provide (provenance contract M3c: for a checkpoint, `auto` or unset without CUDA; for a pre-loaded model, `auto` or `cuda` named without CUDA, while a pre-loaded model with no device named stays on its own): the build is refused (`PipelineBuildError`). Startup publishes nothing and `/diagnose` answers 503; a reload is refused and the served pipeline stays. Never replaced by path reasoning (provenance contract M1), and never moved to the CPU unless `cpu` was asked for
- **No model source configured at all**: pipeline uses `confidence_score = path_reasoning_aggregate_score` (PathReasoner only); the served identity states the checkpoint `absent`, the graph `unverified` (read from `kg.json`, checked against no workspace manifest), the SP table and dataset record `absent`, `path_reasoning_fallback`, no device. Whether this needs an explicit opt-in is B-2's open policy question

### PathReasoner Role (explanation only)
PathReasoner does NOT contribute to scoring. After ranking is determined, the **EvidencePanel** module (`src/reasoning/evidence_panel.py`) generates clinician-facing evidence for each top-K candidate. EvidencePanel uses PathReasoner internally as a building block.

#### Mode A — Direct Path Evidence
When the KG contains explicit paths from patient phenotypes to the candidate (≤ `weak_path_max_hops`, default 4), surface those paths verbatim:
- **STRONG_PATH** label if shortest path ≤ 2 hops
- **WEAK_PATH** label if shortest path is 3-4 hops

#### Mode B — Analogy-Based Evidence (the GNN's killer feature)
When no direct path exists in the KG but the GNN ranks the candidate highly:
1. In the GNN embedding space, find the K nearest known nodes (default K=3)
2. For each similar known node, run PathReasoner — they have full KG connections
3. Present those known paths as **analogy evidence** with shared-feature summary
4. Label as **ANALOGY_BASED**

This handles the zero-shot case the original SHEPHERD paper highlights — GNN can rank a candidate via neighborhood structure similarity even with missing KG edges, and Mode B gives the clinician a trustworthy "why" without inventing fake direct paths.

#### Mode INSUFFICIENT
When neither mode produces usable evidence, the candidate is labeled **INSUFFICIENT** so clinicians know to validate via independent clinical judgment.

#### Critical invariant
EvidencePanel is run **after** scoring/ranking is complete. Toggling `include_explanations` does not change `confidence_score` or candidate ordering — verified by `TestEvidencePanel.test_evidence_panel_does_not_affect_ranking`.

### Two families called "Mode": they share letters, nothing else

The evidence modes above and the scorer-measurement modes are unrelated. Evidence Mode A has nothing to do with measurement Mode A, and there is no evidence Mode C or D.

| Family | Mode | What it is | Status |
|---|---|---|---|
| **Evidence**: the "why" in Diagnosis' candidate details, built after ranking (`src/reasoning/evidence_panel.py`) | A | Direct paths in the KG from the patient's phenotypes to the candidate | in use |
| | B | Analogy: the nearest known nodes in the GNN's embedding space, and their paths | in use |
| | INSUFFICIENT | Neither found usable evidence | in use |
| **Scorer measurement**: how a disease ranker is measured, work item B-0 (`src/evaluation/measurement.py`, `scripts/measure_scorer.py`) | A | Subgraph encoder; each batch's 2-hop subgraph seeded from the answers (the trainer's validation shape); the ladder's control | retired (R2, 2026-10-12) |
| | B | Full-graph encoder; the same candidates as A, so A→B isolates the encoder's scope | retired (R2, 2026-10-12) |
| | C | Full-graph encoder; every disease in the KG; raw cosine; ties go to the lower disease id. B→C isolates the candidate universe. The approved scorer, measured | implemented: the formal Test's measurement, labelled "approved scorer policy (Mode C)" on the Test page |
| | D | Full-graph encoder; production's path-reachable candidates; the η/SP mixture. The ranking the Diagnosis tab serves today (the formula above), measured | not built (B-0.5); its design has an open problem |

**How the measurement modes were decided.**
- The deploying institution decided (2026-08), before any measurement, that disease ranking is GNN-primary (every disease scored, raw cosine as the interim score) and that SP leaves the ranking for an optional, separate analysis (`docs/DISEASE_SCORER_POLICY.md` §1, §3.4). In the reference paper the disease task uses embedding distance alone; the SP fusion belongs to the candidate-gene task, which has a short candidate list. Mode C measures that scorer. Work item B-1 makes the Diagnosis tab rank by it; until then a Mode C result says the tab's ranking differs (`src/evaluation/caveats.py`).
- Modes A and B were retired on 2026-10-12: the formal Test's mode was settled as C (`docs/working/PLAN_TEST_RESULTS.md` D3), Mode A could not build a model from any checkpoint the current writer produces, and the institution had authorised retiring the old pipeline (`docs/working/scorer-measurement/ASSESSMENT_MODE_AB_RETIREMENT.md` §9).
- B-1 may not start before Policy Gate 1, a B-0 report comparing the served ranking (Mode D) with the planned GNN ranking (Mode C) (`docs/DISEASE_SCORER_POLICY.md` §6). C→D changes the candidates and the scorer at once, so B-0.5 adds an intermediate step to tell the two apart (`docs/working/scorer-measurement/README.md`).

---

## 9. Platform Support

| Platform | GPU Attention | Vector Backend | Deploy Script |
|----------|--------------|----------------|--------------|
| Linux x86_64 | FlashAttention-2 / cuDNN SDPA | cuVS (GPU) → Voyager (CPU) | `deploy.sh` |
| Linux aarch64 (DGX Spark) | cuDNN SDPA (no FlashAttn) | cuVS (GPU) → Voyager (CPU) | `deploy.sh` |
| Windows x86_64 | FlashAttention-2 / torch SDPA | Voyager (CPU only) | `deploy.cmd` |

PyTorch 2.9.0 + CUDA 13.0 unified across all platforms.

**Hardware acceleration is required; CUDA is currently the only supported backend** — the current deployment and acceptance scope, not a permanent architectural limit. Another accelerator is adapted, and accepted, on the actual target hardware once one is chosen, with its own explicit device-selection rule (never a silent fallback); hardware differences stay in the existing entry points (device resolution, allocator, AMP, capability checks, environment facts) rather than vendor branches across training, measurement, serving and the UI; and every backend shares the provenance contract and the scoring semantics, so a converted model is recorded as derived from its source checkpoint, never presented as it. No multi-backend framework is built in advance (`docs/working/BACKLOG.md`, item 24).

---

## 10. Design Constraints

1. **Zero real patient data for training**: All training uses simulated patients
2. **16GB VRAM support**: Subgraph sampling keeps memory bounded
3. **Offline-capable**: No internet required after initial data/model download
4. **Clinician-facing output**: Every prediction must include explainable evidence paths
5. **No false completeness**: Empty stub modules must not be presented as functional
