# SHEPHERD-Advanced System Architecture

> **Last updated**: 2026-10-25, at `main` `416a242` (M5 merged, #129).
> **How to read it**: each section says whether it describes **what runs** at that commit or an
> **approved target that is not implemented yet**. The disease scorer policy's authority is
> [`DISEASE_SCORER_POLICY.md`](DISEASE_SCORER_POLICY.md). §11 maps each backend capability to its formal
> entry point and WebUI page; §12 holds the rules that keep one implementation per function.
> **Anchor**: SHEPHERD (npj Digital Medicine 2025) by Harvard MIMS Lab.

---

## 1. System Identity

SHEPHERD-Advanced is a rare disease diagnosis system that:
- Uses a heterogeneous knowledge graph (PrimeKG-based) and GNN to generate node embeddings
- Ranks candidate diseases/genes by embedding distance in a unified vector space
- Provides explainable evidence paths via PathReasoner (auxiliary)
- Supports three diagnostic tasks in a single unified framework

**Running at `416a242`:** disease ranking from a patient's HPO terms (the Diagnosis page and
`POST /api/v1/diagnose`), by the interim served score of §8, with evidence paths, the served
identity on every result, and the research tools around it (workspaces, training, test cohorts,
Mode C measurement, registration). Gene ranking and patients-like-me are not implemented (§3).

**Hardware**: CUDA GPUs (the GB10 is the accepted target; BACKLOG 24 for others). Library versions
are the verified set in `pyproject.toml`, `uv.lock` and `deploy.sh` (§9).
**Graph layers**: GAT, HGT and SAGE, chosen per run (`--conv-type`).
**Extension (not implemented)**: fuzzy input, free text to HPO/MONDO (`src/nlp/`).

---

## 2. Core Architectural Principle: GNN-Primary (approved target, B-1)

> **Status note (2026-08; still true at `416a242`).** The disease scorer policy — what ranks disease candidates, the role of
> the shortest-path signal, and the role of path reasoning — is owned by
> [`DISEASE_SCORER_POLICY.md`](DISEASE_SCORER_POLICY.md), which also records **which parts of this
> section the runtime does not yet implement**. In particular, candidate discovery is currently
> gated by path search and the ranking score currently includes a shortest-path term; §2.1 below
> describes the approved target, not present behaviour; §8 describes what runs. This section will be
> reconciled with the implementation when work item B-1 lands.

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

### 2.3 What PathReasoner Does and Does NOT Do (approved target)

**What runs instead, until B-1:** path search decides which diseases are scored at all (a disease
with no path from the patient's phenotypes is never a candidate; the search is capped in traversal
order), and with no model configured the path-reasoning score becomes the ranking score
(`path_reasoning_fallback`, §8). The table is the target.

| PathReasoner DOES | PathReasoner Does NOT |
|---|---|
| Find BFS paths between phenotypes and top-K diseases for explanation | Gate which diseases can be scored |
| Provide human-readable evidence chains for clinicians | Contribute to confidence score when GNN is active |
| Supply shortest path length as optional supplementary feature | Act as first-pass candidate filter |
| Serve as scoring fallback when no trained GNN model exists | Override GNN ranking |

---

## 3. Three Diagnostic Tasks (one implemented)

All three tasks share the same GNN-generated embedding space. **Only 3.3's disease ranking runs**,
as the Diagnosis page, by the interim score of §8, not the target formulation below.

### 3.1 Causal Gene Discovery (not implemented)

`candidate_genes` is a reserved request field: accepted, ignored, and answered with a warning
(`src/api/routes/diagnose.py`).
- **Input**: Patient phenotypes (Channel 1) + Candidate gene list (Channel 2)
- **Method**: Cosine similarity between patient embedding and each candidate gene embedding
- **Output**: Ranked gene list with confidence scores
- **Channel 2 source**: Upstream variant filtering pipeline (e.g., Exomiser) — SHEPHERD is the consumer, not producer

### 3.2 Similar Patient Retrieval (Patients-Like-Me) (not implemented)
- **Input**: Patient phenotypes only (Channel 1)
- **Method**: Cosine similarity between patient embedding and all known patient embeddings
- **Output**: K most similar confirmed patients with shared phenotypes
- **Clinical value**: Diagnosis validation, especially before ordering genetic testing

### 3.3 Novel Disease Characterization (running as Diagnosis' disease ranking)
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

Channel 2 is reserved, not used (§3.1). As designed, the two channels are processed independently
inside the model and merged only at scoring time:
- Channel 1 determines the patient embedding direction
- Channel 2 determines comparison targets
- Channel 2 does NOT alter the patient embedding

---

## 5. Training

**What runs: one joint multi-task run.** `scripts/train_model.py` (also started by the job runner)
trains on a generated workspace's simulated patients with one loss: diagnosis (weight 1.0),
contrastive (0.3), link prediction (0.5) and ortholog consistency (0.2) by default
(`src/training/trainer.py`, `src/training/loss_functions.py`). Each run writes its own run
directory; a run can be paused, resumed with every learning setting inherited, or extended at a
fixed rate (M3a). The checkpoint payload records the producer, the environment, the settings
requested and in effect, the progress, and the digests of the graph, split manifest and dataset
record it trained on (M2, M3a, M3b).

- **Validation** is disease-disjoint by construction: the workspace builder allocates diseases to
  train or val before it generates samples. It is the Trainer's own measure (val split, top-20),
  used for early stopping and checkpoint selection; it is not a test.
- **No held-out test split** is generated, and no institutional holdout protocol is defined
  (BACKLOG item 11). External cohorts are measured as registered versions (Mode C, §11).
- **Not implemented**: a separate KG pretraining stage, and *branch* training that starts from a
  parent with changed learning settings (BACKLOG 23).

### Training → Inference Bridge
1. Training writes checkpoints into its run directory.
2. Serving loads a listed checkpoint, checks it against the graph and dataset record it names, runs
   the GNN forward pass once and caches every node embedding.
3. With no model source configured, the pipeline serves `path_reasoning_fallback` (§8); a requested
   model that cannot be built is refused, never replaced by that fallback.

---

## 6. Module Architecture

```
Layer 0 (Foundation):  src/core/          types, protocols, schema
Layer 1 (Config):      src/config/        server settings, allocator presets, serving device,
                                          training field spec, model-type constants
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
Layer 8 (Interface):   src/api/           FastAPI REST endpoints; src/api/services: job runner,
                                          server lists, record states, backend control
                       src/webui/         Gradio dashboard, mounted at /ui in the same process
```

**Import rules** (enforced by import-linter):
- Lower layers cannot import higher layers; `src/evaluation` sits above `src/inference` and below
  the interface (`.import-linter.ini`)
- `src/webui` cannot directly import `src/training`. It calls the services in `src/api/services` in
  process (no HTTP hop), and the job runner there starts the formal CLIs (`train_model.py`,
  `measure_scorer.py`) as subprocesses under the one GPU lock

---

## 7. Key Implementation Files

| Component | File | Key Responsibility |
|-----------|------|-------------------|
| GNN Model | `src/models/gnn/shepherd_gnn.py` | Heterogeneous message passing, node embedding generation |
| GNN Layers | `src/models/gnn/layers.py` | HeteroGNNLayer + OrthologGate |
| Diagnosis Head | `src/models/decoders/heads.py` | Not on any serving or measurement path: `ShepherdGNN` builds no task head (policy §3.2); constructed by tests only |
| Loss Functions | `src/training/loss_functions.py` | Multi-task loss (diagnosis, link pred, contrastive, ortholog) |
| Trainer | `src/training/trainer.py` | Training loop with mixed precision |
| Inference Pipeline | `src/inference/pipeline.py` | The served Diagnosis: path-gated candidates, the score of §8, evidence and explanations |
| PathReasoner | `src/reasoning/path_reasoning.py` | BFS path enumeration: today's candidate discovery and evidence (§2.3) |
| Knowledge Graph | `src/kg/graph.py` | KG data structure with PyG export |
| Vector Index | `src/retrieval/vector_index.py` | Standalone ANN index. **Detached from diagnosis** — retained for planned vector-mapping work, not on any inference path |
| Measurement (Mode C) | `src/evaluation/measurement.py`; CLI `scripts/measure_scorer.py` | The formal Test's measurement: every disease ranked by the model's raw cosine, on a registered test cohort version or a generated split |
| Test cohort registry | `src/evaluation/cohort_registry.py`; CLI `scripts/import_cohort.py` | Immutable cohort versions mapped to one graph and bound to its digests, kept outside the repository (provenance contract M4) |
| R10 | `src/evaluation/r10.py` | A test cohort's disease and phenotype-set overlap with the training and validation inputs a checkpoint records, found by digest |
| Evaluation ledger | `src/evaluation/sidecar.py`, `src/evaluation/registration.py`; CLI `scripts/record_evaluation.py` | Test results beside the checkpoint, keyed by its digest (ledger schema 3); one registration function for `--record` and `record_evaluation` |
| Recorded results view | `src/api/routes/evaluations.py` | Read-only `GET /api/v1/evaluations/loaded-model`: the loaded checkpoint's records; shown by Models' read-only view (the Diagnosis card shows only a recorded-tests state, since M5-2) |
| Report store | `src/evaluation/report_store.py` | Every test run's kept report per cohort version, D5's disagreement scan, investigation notes, the usage history |
| Workspace producer and readers | `src/kg/workspace.py`; `src/kg/artifacts.py` | One writer for every workspace (build, demo, probe); the manifest, dataset record and build record readers |
| Server lists | `src/api/services/server_lists.py`; `src/api/routes/lists.py` | Workspaces, checkpoints, cohort versions and intake files by name and digest, resolving only what they list |
| Job runner and GPU lock | `src/api/services/training_manager.py`; `src/utils/gpu_lock.py` | One managed GPU job at a time (training, test), its process group, cancel and stop; one lock file per deployment, also taken by the formal CLIs |
| Run layout and checkpoint reads | `src/utils/checkpoint_paths.py`; `src/utils/checkpoint_io.py`; `src/training/checkpoint_payload.py` | Run directories, one read per checkpoint with its digest, one payload builder for both writers |
| Operator access (D8) | `src/utils/operator_access.py`; `src/api/operator_access.py`; `src/webui/operator_access.py` | One check behind every operator control, in the API and the dashboard |
| Server settings | `src/config/server_settings.py` | `SHEPHERD_DEPLOYMENT_MODE`, `_WORKSPACES_ROOT`, `_REGISTRY`, `_INTAKE_DIR`, `_ORDERING_ROLES`, `_DEVICE`, read where used |
| Record states, ordering | `src/api/services/record_states.py`; `src/evaluation/ordering.py` | Kept, registered and contested states for display; D4's ordering by test score |
| Backend control | `src/api/services/backend_control.py`; `src/api/main.py` | Restart Backend (re-exec in place); shutdown that stops the active job |
| WebUI pages | `src/webui/components/` (`diagnosis_panel.py`, `diagnosis_card.py`, `model_list.py`, `datasets_page.py`, `training_console.py`, `test_page.py`, `models_page.py`, `runtime_settings.py`) | The pages; the shared help, renderers and model list |

---

## 8. The served Diagnosis score (running, until B-1)

**What runs.** The Diagnosis page and `/api/v1/diagnose` rank the diseases that path search reaches
(§2.3) by this mixture whenever a checkpoint and a bound `shortest_paths.pt` are both served:

```
final_score = η × embedding_similarity + (1 - η) × shortest_path_similarity      (η = 0.7)
```

It is **not** the approved target and **not** what a test measures. The target is raw cosine over
every disease with SP taken out of the ranking (policy §1, work item B-1), and Mode C measures that
target (the table at the end of this section). The paper fuses SP only in its candidate-gene task;
its disease task uses embedding distance alone (policy §3.1).

### Signal 1: GNN Embedding Similarity (implemented)
```
patient_embedding = mean_pool(GNN_embeddings[phenotype_nodes])
patient_norm = L2_normalize(patient_embedding)
disease_norm  = L2_normalize(GNN_embedding[disease_node])

embedding_similarity = (dot(patient_norm, disease_norm) + 1.0) / 2.0
```
Serving maps the cosine to `(cos + 1) / 2` (`src/inference/scoring.py`); training and Mode C score raw
cosine (`src/training/trainer.py`, `src/evaluation/measurement.py`). The two rank alike; they are not the
same number.

### Signal 2: Shortest Path Similarity (implemented; leaves the ranking under B-1)
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
- Mixed into every candidate's score whenever the table is served; it is not a conditional fallback.
  A pair that cannot be looked up collapses to `0.0`, below the value of a real "no path" (policy
  §1.1, C9: a B-1 gap)

### Mixing parameter η (`PipelineConfig.eta`)
- `eta = 0.7` (nominally 70% GNN, 30% shortest path; by span about 87/13, policy §3.5), a project
  choice, not a paper value
- Serving builds `PipelineConfig()` with its defaults: nothing sets or tunes η in a deployment
- With no table served, scoring is the GNN term alone (`gnn_only`, effective η = 1.0)

### Graceful degradation
Every result states which of these applies in its **served identity** (provenance contract M3c): the checkpoint, graph, workspace manifest, SP table and dataset record that produced it, by digest or identifier, the scoring mode and the settings in effect, each part with its state (`verified`, `absent`, …). `get_pipeline_config()` and `GET /api/v1/pipeline/status` report the same value for the loaded pipeline.
- **`shortest_paths.pt` missing + `sp_optional=True`** (default): pipeline reports `scoring_mode="gnn_only"`, effective η = 1.0; the served identity states the SP table `absent`, `gnn_only`, eta 1.0
- **`shortest_paths.pt` missing + `sp_optional=False`**: the build is refused (`PipelineBuildError`); no pipeline is published
- **A model requested (checkpoint or pre-loaded model) and not built** — missing or unreadable checkpoint, one that does not build over the graph, no graph data, no PyTorch, or a device the host cannot provide (provenance contract M3c: for a checkpoint, `auto` or unset without CUDA; for a pre-loaded model, `auto` or `cuda` named without CUDA, while a pre-loaded model with no device named stays on its own): the build is refused (`PipelineBuildError`). Startup publishes nothing and `/diagnose` answers 503; a reload is refused and the served pipeline stays. Never replaced by path reasoning (provenance contract M1), and never moved to the CPU unless `cpu` was asked for
- **No model source configured at all**: pipeline uses `confidence_score = path_reasoning_aggregate_score` (PathReasoner only); the served identity states the checkpoint `absent`, the graph `unverified` (read from `kg.json`, checked against no workspace manifest), the SP table and dataset record `absent`, `path_reasoning_fallback`, no device. Whether this needs an explicit opt-in is B-2's open policy question

### Evidence after ranking (running)
After ranking, the **EvidencePanel** module (`src/reasoning/evidence_panel.py`) builds clinician-facing
evidence for each candidate, using PathReasoner as a building block. Today PathReasoner also decides
which diseases are candidates (§2.3), so every served candidate already has discovery paths and the
evidence is always Mode A. **Mode B and INSUFFICIENT are implemented but unreachable on the served
path**; they become reachable when B-1 removes the discovery gate, and Mode B's per-neighbour search
then needs the global budget that gate 3's design must define (policy §6).

#### Mode A — Direct Path Evidence
When the KG contains explicit paths from patient phenotypes to the candidate (≤ `weak_path_max_hops`, default 4), surface those paths verbatim:
- **STRONG_PATH** label if shortest path ≤ 2 hops
- **WEAK_PATH** label if shortest path is 3-4 hops

#### Mode B — Analogy-Based Evidence (implemented; unreachable until B-1)
When no direct path exists in the KG but the GNN ranks the candidate highly:
1. In the GNN embedding space, find the K nearest known nodes (default K=3)
2. For each similar known node, run PathReasoner — they have full KG connections
3. Present those known paths as **analogy evidence** with shared-feature summary
4. Label as **ANALOGY_BASED**

This handles the zero-shot case the original SHEPHERD paper highlights — GNN can rank a candidate via neighborhood structure similarity even with missing KG edges, and Mode B gives the clinician a trustworthy "why" without inventing fake direct paths.

#### Mode INSUFFICIENT (implemented; unreachable until B-1)
When neither mode produces usable evidence, the candidate is labeled **INSUFFICIENT** so clinicians know to validate via independent clinical judgment.

#### Critical invariant
EvidencePanel is run **after** scoring/ranking is complete. Toggling `include_explanations` does not change `confidence_score` or candidate ordering — verified by `TestEvidencePanel.test_evidence_panel_does_not_affect_ranking`.

### Two families called "Mode": they share letters, nothing else

The evidence modes above and the scorer-measurement modes are unrelated. Evidence Mode A has nothing to do with measurement Mode A, and there is no evidence Mode C or D.

| Family | Mode | What it is | Status |
|---|---|---|---|
| **Evidence**: the "why" in Diagnosis' candidate details, built after ranking (`src/reasoning/evidence_panel.py`) | A | Direct paths in the KG from the patient's phenotypes to the candidate | in use (the only mode reached today) |
| | B | Analogy: the nearest known nodes in the GNN's embedding space, and their paths | implemented; unreachable on the served path until B-1 (every candidate has discovery paths) |
| | INSUFFICIENT | Neither found usable evidence | implemented; unreachable until B-1, as B |
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

**Versions** live in one place each: PyTorch and the locked dependencies in `pyproject.toml` and
`uv.lock`, PyG and its extensions in `deploy.sh` (to be pinned with the verified version, BACKLOG 50).
The GB10 acceptance ran torch 2.10.0+cu130 with `torch_geometric` 2.8.1. Upgrades (PyTorch, CUDA, PyG)
are allowed: they go through the same deployment, loading and training flow, are verified, and become
the new baseline; an older environment's evidence stays with that environment (BACKLOG 50).

The "GPU Attention" column names what the deploy scripts can install. No running model path selects
an attention backend: the graph layers are PyG convolutions, and `src/models/attention/adaptive_backend.py`
is not wired in (§11, "reserved").

**Hardware acceleration is required; CUDA is currently the only supported backend** — the current deployment and acceptance scope, not a permanent architectural limit. Another accelerator is adapted, and accepted, on the actual target hardware once one is chosen, with its own explicit device-selection rule (never a silent fallback); hardware differences stay in the existing entry points (device resolution, allocator, AMP, capability checks, environment facts) rather than vendor branches across training, measurement, serving and the UI; and every backend shares the provenance contract and the scoring semantics, so a converted model is recorded as derived from its source checkpoint, never presented as it. No multi-backend framework is built in advance (`docs/working/BACKLOG.md`, item 24).

---

## 10. Design Constraints

1. **Zero real patient data for training**: All training uses simulated patients
2. **16GB VRAM support**: Subgraph sampling keeps memory bounded
3. **Offline-capable**: No internet required after initial data/model download
4. **Clinician-facing output**: Every prediction must include explainable evidence paths
5. **No false completeness**: Empty stub modules must not be presented as functional

---

## 11. Backend capabilities and their front-end entries (at `416a242`)

Each row is checked against the code: the formal entry really calls the implementation named, and
its output is what a front end would show. **Status**: *Connected* (a formal entry and a WebUI page
use it end to end); *Awaiting UI* (the formal entry works and its output is usable, but no page offers
it; not backend work); *CLI only* (a UI is not needed or was decided against); *Partial* (the named
part is missing); *Not implemented* (plans, reserved fields or unwired classes only). Evidence names
the GB10 acceptance and its tested commit where one exists (`docs/working/m*-acceptance-*/`); unit
and integration tests sit beside each module.

**Boundaries that are easy to confuse:**
- **A Mode C test score is not the Diagnosis page's ranking.** Mode C ranks every disease by raw
  cosine; Diagnosis serves the path-gated GNN/SP mixture of §8 until B-1.
- **Resume and extend are done; branch training is not.** A continued run inherits every learning
  setting (BACKLOG 23).
- **The GPU lock covers managed training and test jobs and the formal CLIs, not the resident
  inference service.** Training and serving do not exclude each other (BACKLOG 48).
- **Trustworthy provenance and test records are not model quality or institutional acceptance.**
  Every GB10 run so far used synthetic cohorts or short trainings; no acceptance rule is declared.
- **M5's functional acceptance is not an acceptance of the WebUI's usability** (BACKLOG 51, 52).
- **A reserved field, an unwired class or an unreachable branch is not a feature** (the last rows).

| Capability and boundary | Formal entry | Shared implementation | WebUI | Status | Evidence | Remaining |
|---|---|---|---|---|---|---|
| **Build a workspace** from MONDO, HPO and the annotation files: graph, tensors, train/val samples (disease-disjoint), manifest, dataset record, build record. No test split | `scripts/build_knowledge_graph.py` | `src/kg/workspace.py` (the one writer), `src/kg/sample_generator.py`, `src/ontology/resolver.py` | Datasets shows the command only | Awaiting UI | demo build in the probe at `b14c4de`; the real build last ran 2026-09-29, before the dataset- and build-record requirements | BACKLOG 29: a build job type in the job runner that runs this CLI behind D8 (the CLI stays the entry; budgets, ontology precedence and refusals stay in it); re-run a real build at current code |
| **Shortest-path table** (`shortest_paths.pt` and sidecar, bound to the graph); not part of the dataset fingerprint | `scripts/compute_shortest_paths.py` | `src/inference/sp_artifact.py`, `sp_index.py` | none; lists do not show its state | Awaiting UI (optional) | probe M4, R6, R7 at `b14c4de` | optional job type and an SP state on Datasets, read through `sp_artifact`, not a new check |
| **Demo workspace**, optionally with a small trained model | `scripts/setup_demo.py` | the same writer; the GPU lock | none | CLI only | probe B1 at `b14c4de` | none |
| **Workspace list, dataset and build records, Datasets page** (read-only, compare, "Use for training") | `GET /api/v1/workspaces`; `server_lists` | `src/api/services/server_lists.py`, `src/kg/artifacts.py`, `src/webui/components/workspaces.py` | Datasets; the Training run bar; Test's workspace | Connected | W1, RW1 at `b14c4de` | the open route returns problem texts with absolute paths (BACKLOG 53) |
| **Inventory before deployment** (M2.6) and dataset audits | `scripts/audit_checkpoint_family.py --inventory`, `scripts/audit_*.py` | the readers they audit | none | CLI only | M2's acceptance | none |
| **Start a training run** | `scripts/train_model.py`; `POST /api/v1/training/start` (D8); the job runner | `src/training/trainer.py`, `src/utils/run_plan.py`, `src/api/services/training_manager.py` | Training run bar, Start | Connected | W11, RW3 at `b14c4de`; manual Start at `b14c4de` | the API takes raw paths and drops unknown keys silently (BACKLOG 54) |
| **Resume and extend** (inherited settings; extend at a fixed rate). Not branch | CLI flags; the API's `resume_from`; the job runner | `src/utils/run_plan.py`, the trainer, `checkpoint_paths.py` | Continue… drawer: Run → Checkpoint, Check, preview | Connected | M2, M9, M10, R3, R4 at `b14c4de` (the CLI path); Continue preview by hand at `f9202b6`; no GB10 record of a resume started from the drawer (unit tests cover it) | none |
| **Pause after epoch N** | `train_model.py --pause-after-epoch` | the trainer | none, by decision | CLI only | probe M2 | the API drops the field silently (BACKLOG 54) |
| **Stop training** | `POST /api/v1/training/stop` (D8); the runner | `training_manager.py` | Stop, with its question | Connected | W11, RW3; by hand at `b14c4de` | none |
| **Status, charts, CSV** of the run shown | `GET /api/v1/training/status`, `/training/metrics` | `training_manager.get_status`, `get_metrics_history` | run view: four charts, Export CSV | Partial | by hand at `b14c4de`, `f9202b6` | no page charts or exports an earlier run, though the backend reads any listed run (BACKLOG 52 H); BACKLOG 41 |
| **Run directories and checkpoint lists** (one read per file, with its digest) | `server_lists.list_checkpoints`; `GET /api/v1/workspaces/{name}/checkpoints` | `checkpoint_paths.read_run_checkpoints` | Continue drawer; Models; Change model; Test's model select | Connected | probes at `b14c4de` | three pickers over one list: move Test onto the shared model list (BACKLOG 52); `GET /training/checkpoints` is a legacy duplicate no page uses |
| **Move old checkpoints into the run layout** | `scripts/migrate_checkpoints.py` | the safe reader; M2's record check | none | CLI only | its tests | none |
| **Training settings validation** | the API's cross-field check; per-surface limits; `src/config/training_fields.py` (read by tests only) | `routes/training.py` `cross_field_errors` | the parameters form | Partial | `test_training_fields.py` | UI-only rules (HGT turns AMP off; compile from Settings) are not in the shared service (BACKLOG 54) |
| **Run housekeeping** (delete, archive, rename, tags, compare, lineage) | — | — | — | Not implemented | — | BACKLOG 31, 52 (D) |
| **Branch training** (a parent with changed learning settings) | — | — | — | Not implemented | — | BACKLOG 23 |
| **One managed GPU job at a time**: the lock, its holder, refusals naming it | the job runner; `train_model.py`, `measure_scorer.py`, `setup_demo.py --train-model` | `src/utils/gpu_lock.py`, `training_manager.py` | header chip; Training and Test GPU lines | Connected | W7, W11, RW3 at `b14c4de` | a job this backend did not start is shown and refused, and stopped by pid only (by design) |
| **Test run start and cancel**, process-group reaping | the runner's test job | `training_manager.start_test`, `cancel_test` | Test: Start test, Cancel | Connected | W8, W10 at `b14c4de`; by hand at `773dc4a` | none |
| **Backend stop stops its job; Restart Backend** | process signals, the systemd unit; Settings | `src/api/main.py` shutdown, `backend_control.py` | Settings: Restart Backend; stopping stays an operator action | Connected (Restart); CLI only (stop) | normal stops at `b14c4de`, `f9202b6`, `773dc4a` (no job running) | no GB10 record of a stop while a job runs; a SIGKILLed backend leaves its job holding the lock |
| **Training and the resident inference service exclude each other** | — | — | — | Not implemented | — | BACKLOG 48 |
| **Model list, preview and Load** (confirmation bound to the fingerprint shown) | `server_lists`; `POST /api/v1/pipeline/reload` (D8) | `src/webui/components/model_list.py` (one list), `pipeline.run_reload`, `publish_pipeline` | Diagnosis: Change model; Models: Load | Connected | T5, RT3 at `b14c4de`; by hand at `f9202b6` | the SP table is not bound to the confirmation (BACKLOG 42); the API reload still takes a path |
| **Served identity and status**; Diagnosis card, badge and drawer | `GET /api/v1/pipeline/status` | `served_identity`, `src/webui/components/served_status.py` | Diagnosis card, badge, identity drawer | Connected | M3c's acceptance at `52152e4`; probes at `b14c4de` | none |
| **Which model serves after a start or a restart** | environment variables at startup; `/api/v1/pipeline/config` writes a file nothing reads | `src/api/main.py` lifespan | none | Partial | — | a Load is lost on Restart Backend; decide (BACKLOG 55, with 30 and 48) |
| **Import a test cohort** (preview, then commit a version) | `scripts/import_cohort.py`; Test's import drawer | `src/evaluation/cohort_registry.py` | Test: Import a new version… | Connected | T1–T6 at `2c94a4e`; by hand at `773dc4a` | MyGene2 and institutional adapters, OMIM/ORPHA (BACKLOG 25) |
| **Mode C measurement** of a registered version; every report kept | `scripts/measure_scorer.py --cohort`; the runner's test job | `measurement.run_mode_c`, `report_store.py` | Test: Start test, Result | Connected | W8 at `b14c4de`; by hand at `773dc4a` | none. Not the Diagnosis ranking |
| **Measure a generated split** (`--split train|val`) | `measure_scorer.py --split` | the same | none | CLI only | its tests | none |
| **Register a kept report** (with `--record`, or later by hand); report store and deployment registry only; D5's block | `registration.register_report`; `scripts/record_evaluation.py`; Test | `src/evaluation/registration.py`, the ledger (schema 3) | Test: Register this report | Connected | W9, RW2 at `b14c4de`; by hand at `773dc4a` | none |
| **D5 investigation notes** | Test page only (in process) | `report_store.add_investigation_note` | Test | Partial | — | not shown on Models or the loaded-model route (BACKLOG 40) |
| **Records read back, contested state, D4 ordering, usage history** | `GET /api/v1/evaluations/loaded-model`; `ordering.py`; the usage history | `record_states.py`, `src/evaluation/ordering.py`, `report_store.py` | Models (records, ordering, usage, read-only view); Diagnosis card state | Connected | T5, RT3 at `b14c4de`; by hand at `773dc4a` (one tested model) | before a non-Gradio client: move the page-resident glue (ledger placement, ranking, usage entries) into `src/api/services` |
| **Compare the served ranking with Mode C** (Mode D, B-0.5) | — | — | — | Not implemented | — | an open design problem (`scorer-measurement/PLAN_B03.md`); policy Gate 1 |
| **Model quality, institutional acceptance** | — | — | — | Not implemented (external) | — | data, terms and a declared protocol |
| **Diagnose** (served score of §8, `gnn_only`, path fallback), result identity, CSV and report | `POST /api/v1/diagnose` | `src/inference/pipeline.py`, `scoring.py`, `served_identity` | Diagnosis page and exports | Connected | HTTP and the tab at `52152e4` (M3c); by hand at `f9202b6`; probes E7, R6 at `b14c4de` call the pipeline without explanations | the path fallback's semantics (B-2); exports write SP 0.0 when no table is served (policy C9, B-1) |
| **Evidence and explanations** | the pipeline's explanations step | `evidence_panel.py`, `explanation_generator.py` | Evidence, Explanation, Paths | Connected (Mode A only) | by hand at `f9202b6` | evidence Mode B and INSUFFICIENT unreachable until B-1; a computed summary the route drops; inert flags (BACKLOG 56) |
| **B-1**: GNN-primary ranking over every disease; SP as an optional analysis surface; evidence for the visible page only | — | to reuse: `scoring.cosine_score_matrix`, Mode C's canonical ranking, `sp_index.sp_mean_distances` | — | Not implemented | — | gated (policy §2.1, §6) |
| **HPO search, disease details, keeping a diagnosis, `candidate_genes`** | mock routes; a reserved field | — | — | Not implemented | — | BACKLOG 56 |
| **Operator access (D8) and the read-only view** | `src/utils/operator_access.py`; `OPERATOR_ONLY` | `src/webui/operator_access.py` | chips; greyed controls with the reason | Connected | W2–W6 at `b14c4de`; a read-only tab by hand at `773dc4a` | open read routes name server paths (BACKLOG 53) |
| **Server settings** | `SHEPHERD_*` environment, read where used | `src/config/server_settings.py` | shown, never edited | Connected | probes at `b14c4de` | none |
| **Launcher, bind guard, service unit** | `launch_shepherd.sh`, `scripts/launch/shep_launch.py`, the systemd unit | — | none | CLI only | W2 at `b14c4de`; starts by hand | the launcher's accelerator auto-install is a second, unpinned dependency source (BACKLOG 50) |
| **Allocator preset, torch.compile, Restart Backend** | Settings | `src/config/runtime_presets.py` | Settings | Connected; compile Partial | its tests | compile reaches WebUI-started training only (BACKLOG 54) |
| **Device resolution** (`auto` is CUDA and refuses without it; `cpu` only when asked) | the training and serving resolvers | `src/config/serving_device.py`, the trainer's resolver | device choices | Connected | probes at `b14c4de` | the API and CLI pass `mps` to torch unverified (BACKLOG 24) |
| **Reserved, not features**: the attention backend selector and the launcher's accelerator flags; fuzzy input; the platform and config homes; several front ends on the LAN; non-CUDA hardware | — | — | — | Not implemented | — | BACKLOG 28, 24 |

---

## 12. One implementation per function

Different uses may have different interfaces; one function never keeps two sets of rules. Every next
piece of work says first **which existing implementation it reuses, which gap it fills, and whether
it replaces an old entry**.

- **The front end never decides** model compatibility, continuation rules, registration
  eligibility, operator access or GPU use. It asks the shared implementation (`server_lists`,
  `run_plan`, `register_report` and `report_state`, `operator_access`, the job runner) and shows the
  answer. A kept, registered or contested state comes from the report store's readers, never from a
  job's exit status.
- **New pages use the same lists, readers, checks and job runner** as the existing ones: one
  checkpoint reader, one model list, one reload, one served-status reader, one device resolver per
  role, allocator settings through `runtime_presets` only. No editor for `SHEPHERD_*` settings.
- **The CLIs stay formal operating and test entries.** CLI and API may parse their own arguments, but
  they share the business logic; a CLI does not go through HTTP for the sake of uniformity.
- **A new need changes the shared implementation** and moves its users with it; no second "new" copy.
  Logic that lives in a page today (the Test page's ledger placement, the Models page's ranking and
  usage entries) moves into `src/api/services` before a second client needs it.
- **No fallback for an old format nobody asked for.** The old pipeline's artifacts are unsupported and
  archived (contract decision 2); an old plan's compatibility passage is checked against it first.
- **No framework ahead of a user**: no plugin system, general job framework or free drag-and-drop
  dashboard as a prerequisite.
- **Dependency versions**: the formal deployment installs the currently verified set, kept in one place
  per tool; an upgrade of PyTorch, CUDA or PyG goes through the same deployment, loading and training
  flow and becomes the new baseline once verified, with no second pipeline for old environments; a new
  environment records its own evidence and does not inherit an older acceptance; a model is never
  refused only because a package version string differs: loading follows the model contract and the
  compatibility checks it needs (BACKLOG 50).
