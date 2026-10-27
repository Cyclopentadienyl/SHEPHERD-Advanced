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

**Hardware**: CUDA GPUs (the GB10 is the accepted target; BACKLOG 24 for others). Library versions:
PyTorch, `torch_geometric` (PyG's core) and the rest of the locked set in `pyproject.toml` and
`uv.lock`; PyG's native extensions per platform (§9, BACKLOG 50).
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

**The reference paper's three tasks and this project** (checked against the paper's Methods,
npj Digital Medicine 8:380, 2025, and `docs/working/FINDINGS_PAPER_PROTOCOL_AND_AMP.md` §3.4 and
§3.6; added 2026-10-26). The paper's auxiliary scores belong to their tasks. They are not a set of
scores to add to any ranking.

| Task | Inputs it needs | The paper's formula and model head | Here today | This project's difference | Where it would land |
|---|---|---|---|---|---|
| Causal gene discovery | patient phenotypes and a candidate gene list from variant filtering or curation (13.3 genes on average in one setting, 244.3 in the other; policy §3.1) | Eq 12–14: `η·EMBSIM + (1−η)·SPLSIM`, its own task head, and SP also in the training loss | not implemented: `candidate_genes` is a reserved field, ignored with a warning | — | its own work item (`docs/working/task-scope/README.md`); the only task in which the paper mixes SP into a ranking |
| Patients-like-me | patient phenotypes and a reference cohort of patients with embeddings | Eq 16: `−‖z_Pi − z_Pj‖²` | not implemented | — | its own work item; it needs a reference cohort |
| Novel disease characterisation (disease ranking) | patient phenotypes | Eq 18: `−‖z_d − z_P‖²` over every KG disease, on the patient encoder's output (a transformer encoder and an attention-weighted average) | served as Diagnosis by the legacy mixture of §8 until B-1; Mode C measures the approved interim policy | the approved interim policy (statement 2) is raw cosine of an **unweighted** masked mean: not the paper's encoder, projection or objective. Cosine and negative squared L2 order alike only on the same unit vectors; that does not make the encoders equal, and Mode C is not a reproduction of the paper | B-1 serves the interim policy; paper parity is the scorer-retraining track (`docs/working/scorer-retraining/README.md`) |

**SP for disease ranking** is this project's policy, not the paper's formula. It is an optional
analysis view, off by default, whose sorting and filtering change the view only and never the
canonical rank (policy statement 4b, C1–C10). **The evidence metapaths** of gate 3's design are
likewise this project's evidence policy, not a method the paper specifies
(`docs/working/results-review/SPEC_3_EVIDENCE_AND_AUDIT.md` §7.2).

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
trains on a generated workspace's simulated patients with one loss, `MultiTaskLoss`
(`src/training/loss_functions.py`, wired by `src/training/trainer.py`). Its default weights are
diagnosis 1.0, contrastive 0.3, link prediction 0.5 and ortholog consistency 0.2, but only the
diagnosis and contrastive terms are computed: the other two run only when a batch carries
`positive_triples`/`negative_triples` or `ortholog_pairs`, and no loader produces them, so their
weights (sliders on the Training page among them) change nothing (BACKLOG 57). Each run writes its own run
directory; a run can be paused, resumed with every learning setting inherited, or extended at a
fixed rate (M3a). The checkpoint payload records the producer, the environment, the settings
requested and in effect, the progress, and the digests of the graph, the sample files and the split
manifest it trained on; the manifest binds the dataset record's digest, which the checkpoint does not
record itself (M2, M3a, M3b).

- **Validation** is disease-disjoint by construction: the workspace builder allocates diseases to
  train or val before it generates samples. It is the Trainer's own measure (val split, top-20),
  used for early stopping and checkpoint selection; it is not a test.
- **No held-out test split** is generated, and no institutional holdout protocol is defined
  (BACKLOG item 11). External cohorts are measured as registered versions (Mode C, §11).
- **Not implemented**: a separate KG pretraining stage, and *branch* training that starts from a
  parent with changed learning settings (BACKLOG 23).

### Training → Inference Bridge
1. Training writes checkpoints into its run directory.
2. Serving loads a checkpoint (named by `SHEPHERD_CHECKPOINT_PATH` at startup; chosen from the
   server's list by a WebUI Load; or, through the reload API, given by path or selected from the
   workspace's latest run). It refuses the checkpoint unless its recorded graph digests (`kg` and the
   three tensors) match the workspace it is served over, and that workspace's dataset record is the
   one its manifest binds; then it runs the GNN forward pass once and caches every node embedding.
3. With a graph configured (`SHEPHERD_KG_PATH`) and no model source, the pipeline serves
   `path_reasoning_fallback` (§8). With no graph configured, no pipeline is built and `/diagnose`
   returns its demo answer (invented candidates, a "mock data" warning, `served_identity` null), which
   the Diagnosis page labels as such (BACKLOG 55). A requested model that cannot be built is refused,
   never replaced by that fallback.

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
| Measurement (Mode D) | `src/evaluation/mode_d.py`; CLI `scripts/measure_scorer.py --mode D` | What the Diagnosis page serves today, measured through `DiagnosisPipeline.rank()` on a registered cohort version, with Mode C beside it on the same embeddings (policy gate 1's comparison; kept apart for a clean retirement, PLAN_B05 §10) |
| Test cohort registry | `src/evaluation/cohort_registry.py`; CLI `scripts/import_cohort.py` | Immutable cohort versions mapped to one graph and bound to its digests, kept outside the repository (provenance contract M4) |
| R10 | `src/evaluation/r10.py` | A test cohort's disease and phenotype-set overlap with the training and validation inputs a checkpoint records, found by digest |
| Evaluation ledger | `src/evaluation/sidecar.py`, `src/evaluation/registration.py`; CLI `scripts/record_evaluation.py` | Test results beside the checkpoint, keyed by its digest (ledger schema 3); one registration function for `--record` and `record_evaluation` |
| Recorded results view | `src/api/routes/evaluations.py` | Read-only `GET /api/v1/evaluations/loaded-model`: the loaded checkpoint's records; shown by Models' read-only view (the Diagnosis card shows only a recorded-tests state, since M5-2) |
| Report store | `src/evaluation/report_store.py` | Every test run's kept report per cohort version, D5's disagreement scan, investigation notes, the usage history |
| Workspace producer and readers | `src/kg/workspace.py`; `src/kg/artifacts.py` | One writer for every workspace (build, demo, probe); the manifest, dataset record and build record readers |
| Server lists | `src/api/services/server_lists.py`; `src/api/routes/lists.py` | Workspaces, checkpoints, cohort versions and intake files by name and digest, resolving only what they list |
| Job runner and GPU lock | `src/api/services/training_manager.py`; `src/utils/gpu_lock.py` | One managed GPU job at a time (training, test), its process group, cancel and stop; one lock file per deployment, also taken by the GPU CLIs (`train_model.py`, `measure_scorer.py`, `setup_demo.py --train-model`) |
| Run layout and checkpoint reads | `src/utils/checkpoint_paths.py`; `src/utils/checkpoint_io.py`; `src/training/checkpoint_payload.py` | Run directories, one read per checkpoint with its digest, one payload builder for both writers |
| Operator access (D8) | `src/utils/operator_access.py`; `src/api/operator_access.py`; `src/webui/operator_access.py` | One check behind every operator control, in the API and the dashboard |
| Server settings | `src/config/server_settings.py`; `src/config/serving_device.py` | `SHEPHERD_DEPLOYMENT_MODE`, `_WORKSPACES_ROOT`, `_REGISTRY`, `_INTAKE_DIR`, `_TRAINING_EVIDENCE_DIRS`, `_ORDERING_ROLES` (server settings) and `_DEVICE` (serving device), read where used |
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
  A pair with no path contributes `max_hops + 1`; a candidate whose SP cannot be computed at all (no
  node mapping, an unmapped candidate, no mappable phenotype, the primitive's unavailable state)
  collapses to `0.0`, below the value of a real "no path" (policy §1.1, C9). The result union
  replaces that `0.0` with a status (`docs/working/results-review/SPEC_1_RESULTS_REVIEW.md` §2.1,
  BACKLOG §0 batch 4)

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
then needs a budget. Gate 3's design (policy §6) is written: a target-restricted traversal over four
declared path kinds, with one fixed budget per target and a request ceiling at admission
(`docs/working/results-review/SPEC_3_EVIDENCE_AND_AUDIT.md` §7, awaiting approval).

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
| | D | Full-graph encoder; production's path-reachable candidates; the served score (the η/SP mixture, or the model term alone without a table). The ranking the Diagnosis tab serves today (the formula above), measured | implemented (B-0.5, Batch 3, 2026-10-26; `docs/working/scorer-measurement/PLAN_B05.md`), labelled "Served ranking today (Mode D) … not the approved scorer policy" |

**How the measurement modes were decided.**
- The deploying institution decided (2026-08), before any measurement, that disease ranking is GNN-primary (every disease scored, raw cosine as the interim score) and that SP leaves the ranking for an optional, separate analysis (`docs/DISEASE_SCORER_POLICY.md` §1, §3.4). In the reference paper the disease task uses embedding distance alone; the SP fusion belongs to the candidate-gene task, which has a short candidate list. Mode C measures that scorer. Work item B-1 makes the Diagnosis tab rank by it; until then a Mode C result says the tab's ranking differs (`src/evaluation/caveats.py`).
- Modes A and B were retired on 2026-10-12: the formal Test's mode was settled as C (`docs/working/PLAN_TEST_RESULTS.md` D3), Mode A could not build a model from any checkpoint the current writer produces, and the institution had authorised retiring the old pipeline (`docs/working/scorer-measurement/ASSESSMENT_MODE_AB_RETIREMENT.md` §9).
- B-1 may not start before Policy Gate 1, a B-0 report comparing the served ranking (Mode D) with the planned GNN ranking (Mode C) (`docs/DISEASE_SCORER_POLICY.md` §6). C→D changes the candidates, the score and the tie rule at once, so Mode D also reports derived readings on the formal Mode C scores that change one of them at a time (`docs/working/scorer-measurement/PLAN_B05.md` §5).

---

## 9. Platform Support

| Platform | GPU Attention | Vector Backend | Deploy Script |
|----------|--------------|----------------|--------------|
| Linux x86_64 | FlashAttention-2 / cuDNN SDPA | cuVS (GPU) → Voyager (CPU) | `deploy.sh` |
| Linux aarch64 (DGX Spark) | cuDNN SDPA (no FlashAttn) | cuVS (GPU) → Voyager (CPU) | `deploy.sh` |
| Windows x86_64 | FlashAttention-2 / torch SDPA | Voyager (CPU only) | `deploy.cmd` |

**Versions** live in `pyproject.toml` and `uv.lock` (since Batch 1, BACKLOG 50): PyTorch,
`torch_geometric` (PyG's pure-Python core, one wheel for every platform) and everything they
require. PyG's native extensions stay out of the lock because they come per platform: the deploy
scripts take them from the data.pyg.org index for the installed torch (derived, not written a
second time), or on ARM from the prebuilt release or a local build; A1 and A5 record their
versions. The launcher's optional accelerators stay out of the lock, and every install of one is
constrained to it (deployment guide §4.15). M5's GB10 acceptances (`b14c4de`, and the WebUI rounds of 2026-10-25) ran torch 2.10.0+cu130 with
`torch_geometric` 2.8.1, the current baseline, which `deploy.sh` installs from the lock since Batch 1
(its GB10 deploy and environment check at `0bc5178` passed, 2026-10-26; Windows not run); M4's ran
2.8.0.post1. Upgrades (PyTorch, CUDA, PyG)
are allowed: they go through the same deployment, loading and training flow, are verified, and become
the new baseline; an older environment's evidence stays with that environment (BACKLOG 50).

The "GPU Attention" column names what can be enabled on each platform: SDPA comes with PyTorch, and
FlashAttention-2 (like xFormers and SageAttention) is installed only by the launcher's optional flags
(`launch_shepherd.sh --flash-attn`, `scripts/launch/shep_launch.py`), not by the deploy scripts. No running model path selects
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
it; what remains is the page and, where the row names one, a job type in the job runner that runs the
formal entry, whose rules stay in that entry); *CLI only* (a UI is not needed or was decided against); *Partial* (the named
part is missing); *Not implemented* (plans, reserved fields or unwired classes only). Evidence names
the GB10 acceptance and its tested commit where one exists (`docs/working/m*-acceptance-*/`); unit
and integration tests sit beside each module.

**Boundaries that are easy to confuse:**
- **A Mode C test score is not the Diagnosis page's ranking.** Mode C ranks every disease by raw
  cosine; Diagnosis serves the path-gated GNN/SP mixture of §8 until B-1.
- **Resume and extend are done; branch training is not.** A continued run inherits every learning
  setting (BACKLOG 23).
- **The GPU lock covers managed training and test jobs and the GPU CLIs (`train_model.py`,
  `measure_scorer.py`, `setup_demo.py --train-model`), not the resident inference service.** Training and serving do not exclude each other (BACKLOG 48).
- **Trustworthy provenance and test records are not model quality or institutional acceptance.**
  Every GB10 run so far used synthetic cohorts or short trainings; no acceptance rule is declared.
- **M5's functional acceptance is not an acceptance of the WebUI's usability** (BACKLOG 51, 52).
- **A reserved field, an unwired class or an unreachable branch is not a feature** (the last rows).

| Capability and boundary | Formal entry | Shared implementation | WebUI | Status | Evidence | Remaining |
|---|---|---|---|---|---|---|
| **Build a workspace** from MONDO, HPO and the annotation files: graph, tensors, train/val samples (disease-disjoint), manifest, dataset record, build record. No test split | `scripts/build_knowledge_graph.py` | `src/kg/workspace.py` (the one writer), `src/kg/sample_generator.py`, `src/ontology/resolver.py` | Datasets shows the command only | Awaiting UI | demo build in the probe at `b14c4de`; the real build last ran 2026-09-29: its dataset record verifies (origin `files`, RW1 at `b14c4de`), but it has no build record, which M5 added | BACKLOG 29: a build job type in the job runner that runs this CLI behind D8 (the CLI stays the entry; budgets, ontology precedence and refusals stay in it); re-run a real build at current code |
| **Shortest-path table** (`shortest_paths.pt` and sidecar, bound to the graph); not part of the dataset fingerprint | `scripts/compute_shortest_paths.py` | `src/inference/sp_artifact.py`, `sp_index.py` | none; lists do not show its state | Awaiting UI (optional) | probe M4, R6, R7 at `b14c4de` | optional job type and an SP state on Datasets, read through `sp_artifact`, not a new check |
| **Demo workspace**, optionally with a small trained model | `scripts/setup_demo.py` | the same writer; the GPU lock | none | CLI only | probe B1 at `b14c4de` writes the demo graph (`build_demo_kg`) through the shared writer; `setup_demo.py` itself and its `--train-model` lock have no GB10 record | none |
| **Workspace list, dataset and build records, Datasets page** (read-only, compare, "Use for training") | `GET /api/v1/workspaces`; `server_lists` | `src/api/services/server_lists.py`, `src/kg/artifacts.py`, `src/webui/components/workspaces.py` | Datasets; the Training run bar; Test's workspace | Connected | W1, RW1 at `b14c4de` | the open route returns problem texts with absolute paths (BACKLOG 53) |
| **Inventory before deployment** (M2.6) and dataset audits | `scripts/audit_checkpoint_family.py --inventory`, `scripts/audit_*.py` | the readers they audit | none | CLI only | M2's acceptance | two audits record a digest from a second read of what they parsed (BACKLOG 19.9; 19.12, optional) |
| **Start a training run** | `scripts/train_model.py`; `POST /api/v1/training/start` (D8); the job runner | `src/training/trainer.py`, `src/utils/checkpoint_paths.py` (the run directory), `src/api/services/training_manager.py` | Training run bar, Start | Connected | W11, RW3 at `b14c4de`; manual Start at `b14c4de` | the API takes raw paths and drops unknown keys silently (BACKLOG 54); two of the four loss terms never run, so their weight sliders change nothing (BACKLOG 57) |
| **Resume and extend** (inherited settings; extend at a fixed rate). Not branch | CLI flags; the API's `resume_from`; the job runner | `src/utils/run_plan.py`, the trainer, `checkpoint_paths.py` | Continue… drawer: Run → Checkpoint, Check, preview | Connected | M2, M9, M10, R3, R4 at `b14c4de` (the CLI path); Continue preview by hand at `f9202b6`; no GB10 record of a resume started from the drawer (unit tests cover it) | the API takes `resume_from` as a raw path and drops `extend_lr` silently (BACKLOG 54); the drawer sends no `extend_lr` either, so a parent whose rate ended at 0 is extended from the CLI only |
| **Pause after epoch N** | `train_model.py --pause-after-epoch` | the trainer | none, by decision | CLI only | probe M2 | the API drops the field silently (BACKLOG 54) |
| **Stop training** | `POST /api/v1/training/stop` (D8); the runner | `training_manager.py` | Stop, with its question | Connected | W11, RW3; by hand at `b14c4de` | none |
| **Status, charts, CSV** of the run shown | `GET /api/v1/training/status`, `/training/metrics` | `training_manager.get_status`, `get_metrics_history` | run view: four charts, Export CSV | Partial | by hand at `b14c4de`, `f9202b6` | no page charts or exports an earlier run, though the backend reads any listed run (BACKLOG 52 H); BACKLOG 41 |
| **Run directories and checkpoint lists** (one read per file, with its digest) | `server_lists.list_checkpoints`; `GET /api/v1/workspaces/{name}/checkpoints` | `checkpoint_paths.read_run_checkpoints` | Continue drawer; Models; Change model; Test's model select | Connected | probes at `b14c4de` | three pickers over one list: move Test onto the shared model list (BACKLOG 52); `GET /training/checkpoints` is a legacy duplicate no page uses |
| **Move old checkpoints into the run layout** | `scripts/migrate_checkpoints.py` | the safe reader; M2's record check | none | CLI only | its tests | none |
| **Training settings validation** | the API's cross-field check; per-surface limits; `src/config/training_fields.py` (read by tests only) | `routes/training.py` `cross_field_errors` | the parameters form | Partial | `test_training_fields.py` | `cross_field_errors` is called by the API route only: the page's Start relies on the form's choices, and its Continue starts from the parent's recorded settings without the check; the CLI checks neither rule and the trainer repeats the onecycle rule; two rules exist only in the page (HGT turns AMP off; compile from Settings): each is checked against its dependency and the evidence before any becomes shared, and the HGT one is contradicted by M5's acceptance, which trained HGT with AMP (R2, R4 at `b14c4de`) (BACKLOG 54) |
| **Run housekeeping** (delete, archive, rename, tags, compare, lineage) | — | — | — | Not implemented | — | BACKLOG 31 (compare, archive), 52 (D) (lineage); tags are named in BACKLOG 23 as a prerequisite of the experiment phase; delete and rename have no item yet |
| **Branch training** (a parent with changed learning settings) | — | — | — | Not implemented | — | BACKLOG 23 |
| **One managed GPU job at a time**: the lock, its holder, refusals naming it | the job runner; `train_model.py`, `measure_scorer.py`, `setup_demo.py --train-model` | `src/utils/gpu_lock.py`, `training_manager.py` | header chip; Training and Test GPU lines | Connected | W7, W11, RW3 at `b14c4de` | a job this backend did not start is shown and refused, and stopped by pid only (by design) |
| **Test run start and cancel**, process-group reaping | the runner's test job | `training_manager.start_test`, `cancel_test` | Test: Start test, Cancel | Connected | W8, W10 at `b14c4de` (a start run to the end, a training stop refused meanwhile; a start refused before measuring); Start test by hand at `773dc4a`; no GB10 record of a running test cancelled or of leftover group processes ended (unit and integration tests cover them) | none |
| **Backend stop stops its job; Restart Backend** | process signals, the systemd unit; Settings | `src/api/main.py` shutdown, `backend_control.py` | Settings: Restart Backend; stopping stays an operator action | Connected (Restart); CLI only (stop) | normal stops at `b14c4de`, `f9202b6`, `773dc4a` (no job running); no GB10 record of a Restart Backend (`tests/unit/test_backend_control.py` covers it) | no GB10 record of a stop while a job runs; a SIGKILLed backend leaves its job holding the lock |
| **Training and the resident inference service exclude each other** | — | — | — | Not implemented | — | BACKLOG 48 |
| **Model list, preview and Load** (confirmation bound to the fingerprint shown) | `server_lists`; `POST /api/v1/pipeline/reload` (D8) | `src/webui/components/model_list.py` (one list), `routes/pipeline.py` `run_reload`, `src/api/main.py` `publish_pipeline` | Diagnosis: Change model; Models: Load | Connected | T5, RT3 at `b14c4de`; by hand at `f9202b6` | the SP table is not bound to the confirmation (BACKLOG 42); the API reload still takes a path |
| **Served identity and status**; Diagnosis card, badge and drawer | `GET /api/v1/pipeline/status` | `served_identity`, `src/webui/components/served_status.py` | Diagnosis card, badge, identity drawer | Connected | M3c's acceptance at `52152e4`; probes at `b14c4de` | none |
| **Which model serves after a start or a restart** | environment variables at startup; `/api/v1/pipeline/config` writes and reads back a file nothing else consumes | `src/api/main.py` lifespan | none | Partial | — | a Load is lost on Restart Backend; decide (BACKLOG 55, with 30 and 48) |
| **Import a test cohort** (preview, then commit a version) | `scripts/import_cohort.py`; Test's import drawer | `src/evaluation/cohort_registry.py` | Test: Import a new version… | Connected | T1 and T4 at `2c94a4e` (T2, T3, T5, T6 measure, register, view and audit the imported version); the import drawer by hand at `773dc4a` | MyGene2 and institutional adapters, OMIM/ORPHA (BACKLOG 25) |
| **Mode C measurement** of a registered version; every report kept | `scripts/measure_scorer.py --cohort`; the runner's test job | `measurement.run_mode_c`, `report_store.py` | Test: Start test, Result | Connected | W8 at `b14c4de`; by hand at `773dc4a` | none. Not the Diagnosis ranking |
| **Measure a generated split** (`--split {train,val}`) | `measure_scorer.py --split` | the same | none | CLI only | M3, R5 at `b14c4de` (`--split val`); its tests | none |
| **Register a kept report** (with `--record`, or later by hand); report store and deployment registry only; D5's block | `registration.register_report`; `scripts/record_evaluation.py`; Test | `src/evaluation/registration.py`, the ledger (schema 3) | Test: Register this report | Connected | W9, RW2 at `b14c4de`; by hand at `773dc4a` | none |
| **D5 investigation notes** | Test page only (in process) | `report_store.add_investigation_note` | Test | Partial | — | not shown on Models or the loaded-model route (BACKLOG 40) |
| **Records read back, contested state, D4 ordering, usage history** | `GET /api/v1/evaluations/loaded-model` (records, contested state); D4 ordering and the usage history: the Models page only (in process) | `record_states.py`, `src/evaluation/ordering.py`, `report_store.py` | Models (records, ordering, usage, read-only view); Diagnosis card state | Connected (records, contested state); Partial (ordering, usage: no formal entry) | T5, RT3 at `b14c4de`; by hand at `773dc4a` (one tested model) | before a non-Gradio client: move the page-resident glue (ledger placement, ranking, usage entries) into `src/api/services` |
| **Compare the served ranking with Mode C** (Mode D, B-0.5) | `scripts/measure_scorer.py --mode D --cohort` | `src/evaluation/mode_d.py` over `DiagnosisPipeline.rank()` and `run_mode_c` | Test: Result and Models: records show a Mode D record by its own label; no page starts one (PLAN_B05 §7) | CLI only (results shown) | implemented in BACKLOG §0 batch 3; its tests and probe T7; the batch's short GB10 run passed at `be0ee1a` (2026-10-27, `working/batch3-acceptance-2026-10-27/`) | the institutional run and its protocol (8a, 8b); its report is policy Gate 1's evidence |
| **Model quality, institutional acceptance** | — | — | — | Not implemented (external) | — | data, terms and a declared protocol |
| **Diagnose** (served score of §8, `gnn_only`, path fallback), result identity, CSV and report | `POST /api/v1/diagnose` | `src/inference/pipeline.py`, `scoring.py`, `served_identity` | Diagnosis page and exports | Connected | HTTP and the tab at `52152e4` (M3c); by hand at `f9202b6`; probes E7, R6 at `b14c4de` call the pipeline without explanations | the path fallback's semantics (B-2); exports write SP 0.0 when no table is served (policy C9). Both designed as the result union (§11.1 step 7), built in BACKLOG §0 batch 4 |
| **Evidence and explanations** | the pipeline's explanations step | `evidence_panel.py`, `explanation_generator.py` | Evidence, Explanation, Paths | Connected (Mode A only) | by hand at `f9202b6` | evidence Mode B and INSUFFICIENT unreachable until B-1, whose target-restricted traversal is gate 3's design (§11.1 step 6); a computed summary the route drops; inert flags (BACKLOG 56) |
| **B-1**: GNN-primary ranking over every disease; SP as an optional analysis surface; evidence for the visible page only | — | to reuse: `scoring.cosine_score_matrix` and `masked_mean_pool`, Mode C's canonical ranking (moved into `scoring.py`), `sp_index.sp_mean_distances` | — | Not implemented | — | designed (§11.1); built in BACKLOG §0 batch 5, gated by policy gates 1 and 3 (§6) |
| **HPO search, disease details, keeping a diagnosis, `candidate_genes`** | mock routes (HPO search, disease details); `GET /api/v1/diagnose/{session_id}` answers 501; a reserved field | — | — | Not implemented | — | BACKLOG 56 |
| **Operator access (D8) and the read-only view** | `OPERATOR_ONLY` (`src/api/operator_access.py`) on every gated route; the dashboard's helper `src/webui/operator_access.py` in each operator handler | `src/utils/operator_access.py` (`check_operator_access`, the one check) | chips; greyed controls with the reason | Connected | W2–W6 at `b14c4de`; a read-only tab by hand at `773dc4a` | open read routes name server paths (BACKLOG 53) |
| **Server settings** | `SHEPHERD_*` environment, read where used | `src/config/server_settings.py` | shown, never edited | Connected | probes at `b14c4de` | none |
| **Launcher, bind guard, service unit** | `launch_shepherd.sh`, `scripts/launch/shep_launch.py`, the systemd unit | the bind rule in `src/config/server_settings.py` (`network_bind`, `startup_bind`), applied by the launcher, `src.api.main` and startup | none | CLI only | W2 at `b14c4de` starts the service with the launcher's uvicorn entry and arguments (not the bind guard's refusal or the unit, which their unit tests cover); `launch_shepherd.sh` by hand at `773dc4a` | the launcher's optional accelerators stay out of the lock; every pip install of one is constrained to `uv.lock` (exported by `uv`) and refused when that cannot be had, so none replaces a locked package (BACKLOG 50; deployment guide §4.15) |
| **Allocator preset, torch.compile, Restart Backend** | Settings | `src/config/runtime_presets.py` | Settings | Connected; compile Partial | its tests | compile reaches WebUI-started training only (BACKLOG 54) |
| **Device resolution** (`auto` is CUDA and refuses without it; `cpu` only when asked) | the training, measurement and serving resolvers | `resolve_serving_device` (`src/inference/pipeline.py`; the requested value read by `src/config/serving_device.py`, `SHEPHERD_DEVICE`), `resolve_training_device` (`scripts/train_model.py`), `_resolve_device` (`scripts/measure_scorer.py`) | device choices | Connected | probes at `b14c4de` | the API and CLI hand `mps` to torch unchecked and check `cuda:N` only for CUDA, not its ordinal (BACKLOG 24) |
| **WebUI usability** (density, layout, help texts) | — | the pages in `src/webui/components/` | every page | Partial: functionally accepted only | M5's manual rounds to `773dc4a` | BACKLOG 51, 52, in §0's UI batches; not a condition for reopening M5 |
| **Reserved, not features**: the attention backend selector and the launcher's accelerator flags (they install packages no model path uses); fuzzy input; the platform and config homes; several front ends on the LAN; non-CUDA hardware | — | — | — | Not implemented | — | BACKLOG 28, 24 |

### 11.1 The diagnosis and measurement chain (Batch 2 design, at `60feb6d`)

The same chain read end to end, from input to registration. For each step the table gives four
things:
- today's formal entry and its shared function;
- the gap;
- what changes, and in which batch of BACKLOG §0;
- which old entry goes.

The designs are:
- `docs/working/scorer-measurement/PLAN_B05.md` (Mode D);
- `docs/working/results-review/SPEC_1_RESULTS_REVIEW.md` §2.1 (the result union) and §2.2 (the
  shared scoring boundary);
- `docs/working/results-review/SPEC_3_EVIDENCE_AND_AUDIT.md` §7 (gate 3's traversal).

**What each piece does:**
- **Mode C** measures the approved policy: raw cosine over every disease.
- **Mode D** measures what the Diagnosis page serves today, through serving's own method, and sets it
  beside Mode C in the same run.
- **B-1** makes the Diagnosis page serve the approved policy, behind a switch.

Mode C and Mode D change no served value. B-1 does.

| Step | Today: formal entry and shared function | Gap | Change (batch) | Old entry removed |
|---|---|---|---|---|
| 1. **Input** (phenotype terms) | `DiagnosisPipeline.validate_input` with `normalise_phenotypes` (`src/kg/phenotype_normalisation.py`); the API's `DiagnoseRequest`; for cohorts, the importer's `map_cases` | Serving maps a term three times, each handling a miss differently. A graph node with no tensor row is skipped silently. `validate_phenotypes=False` is a second behaviour | One mapping step that returns tensor rows. An unknown term is dropped and stated; a node without a row is refused (SPEC_1 §2.2). **Batch 4** | `_phenotypes_to_node_ids`; the per-score lookups; `PipelineConfig.validate_phenotypes` |
| 2. **Model forward** | Serving: `_precompute_node_embeddings` (float32, cached). Mode C: `encode_full_graph`. Both build through `build_shepherd_model` | Serving does not expose its numerical regime | `embedding_regime` exposed and checked against Mode D's manifest (PLAN_B05 §3). **Batch 3, implemented** | none |
| 3. **Candidate set** | Serving: `_find_all_paths` (BFS and the direct finder, both with caps that depend on order). Mode C: every disease row | Statement 3 (every disease is scored) is not served; a truth discovery does not reach has no served rank | Batch 3 (implemented): Mode D measures today's gate and counts what it does not reach (`not_reached`). Batch 5: B-1 scores every disease row | Batch 5: discovery stops being the candidate gate; it stays only for the path-reasoning fallback |
| 4. **Scoring** | Serving: `_calculate_combined_score`, `_calculate_gnn_score` and `_calculate_sp_score`, each called once per candidate, over the primitives in `src/inference/scoring.py`. Mode C: `masked_mean_pool`, `cosine_score_matrix` | The pool clamps indices. An unavailable model term or SP value becomes `0.0`. `(cos+1)/2` is rounded in float32 | Batch 4: the model term fails closed; SP gets statuses from one SP caller. Batch 5: `score_patients` (raw cosine) | Batch 5: `pool_patient_embeddings`. The mixture helpers go when `served_mixture/v1` retires |
| 5. **Ranking** (order, ties, cut) | Serving: a stable sort, then the `top_k` slice, inside `_score_and_rank_candidates`. Mode C: `canonical_ranking` (`src/evaluation/measurement.py`) | No untruncated served rank exists. There are two tie rules, and serving cannot import Mode C's | Batch 3 (implemented): `DiagnosisPipeline.rank()`, which `run()` calls before its cut. Batch 5: `canonical_ranking` moves into `scoring.py`; `selection_limit` cuts after scoring | Batch 5: serving's own sort, for the GNN kind |
| 6. **Evidence** | `_add_explanations` calls `EvidencePanel.build_evidence`, always Mode A from discovery paths, eagerly, for the top K only | Not restricted to the target; no budget; Mode B and INSUFFICIENT unreachable; no record identity | `src/reasoning/target_evidence.py`: four declared path kinds, one fixed budget per target, computed lazily per page (SPEC_3 §7). **Batch 5, gate 3** | `EvidencePanel` and the eager step, together with the legacy result kinds |
| 7. **Result and report** | `InferenceResult` and `DiagnosisCandidate` (`src/core/types.py`); `POST /api/v1/diagnose`; the Diagnosis page, its CSV and its report | One flat record. `confidence_score` holds two quantities. An unavailable value is written as `0.0`. The demo answer carries invented scores | The result union, discriminated by `result_kind` (SPEC_1 §2.1). Batch 4: `served_mixture/v1`, `path_reasoning/v1`, `demo/v1`. Batch 5: `gnn_ranking/v1` | `confidence_score`, `gnn_score` and the bare `sp_score`; `scripts/test_gnn_inference.py`; `generate_evidence_summary` |
| 8. **Measurement** | Mode C: `scripts/measure_scorer.py --cohort` calls `run_mode_c`. Mode D: none | Gate 1's comparison does not exist | `run_mode_d` and `--mode D`: post-import and among-reached metrics over named case sets; the C side is `run_mode_c` itself on the same embeddings; derived readings one change at a time (PLAN_B05). **Batch 3, implemented** | none |
| 9. **Registration and comparison** | `registration.register_report`; the ledger (D4 groups and the D5 key, both including `mode`); `report_store`'s usage history; the Test and Models pages | Labels assume Mode C; the usage history has no mode | Mode D's values go in the existing manifest fields, with no new field; labels come from each record; the usage history records a mode (PLAN_B05 §6). **Batch 3, implemented**; the ordering refuses a non-Mode-C scoring group itself (PLAN_B05 §11) | none: no ledger rule changes |

**Proving correctness without rerunning old acceptances:**
- **Golden tests** pin what must not move: serving's output across batches 3 and 4, and Mode C's
  reports when batch 5 moves its ranking. They compare the promised fields (scores, order, ranks,
  cases, digests, semantic fields), never timestamps, revisions or run metadata
  (`docs/working/scorer-measurement/PLAN_B05.md` §8).
- **New tests** cover only what is new.
- **One short GB10 run per batch** where hardware matters: batch 3 runs Mode C, then Mode D, on a
  registered synthetic cohort (passed at `be0ee1a`, 2026-10-27).
- **Kept evidence still counts**, unchanged: M4's and M5's acceptances, B-0.4's SP benchmarks and the
  Mode C reports.

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
- **One disease scorer** (Batch 2 design, §11.1). Every ranking that claims the approved policy, B-1's
  serving and Mode C alike, goes through `score_patients` and `canonical_ranking` in
  `src/inference/scoring.py`. There is no copy and no test-only scorer. The trainer's validation
  scorer stays separately written, on purpose (the `scorer-independence` contract).
- **The served ranking is measured through serving's own method.** Mode D calls
  `DiagnosisPipeline.rank()`, which `run()` itself calls. There is no second candidate construction,
  and Modes A and B are not revived.
- **Evidence never touches the ranking.** It takes an immutable result and returns records.
  `src.inference.scoring` never imports `src.reasoning` (the `ranking-without-evidence` contract,
  batch 5).
- **A value that could not be computed is a status, never a number**, in every result kind, export
  and measurement.
- **Dependency versions**: the formal deployment installs the currently verified set, its versions kept
  in one central place, not scattered across entry points or tests; an upgrade of PyTorch, CUDA or PyG
  goes through the same deployment, loading and training flow and becomes the new baseline once verified, with no second pipeline for old environments; a new
  environment records its own evidence and does not inherit an older acceptance; a model is never
  refused only because a package version string differs: loading follows the model contract and the
  compatibility checks it needs (BACKLOG 50).
  The pinned set is today's formal deployment set, not a permanent limit on PyG, torch or CUDA: the
  versions live in `pyproject.toml` and `uv.lock` only, and no training, inference or UI code checks
  a package version (deployment guide §4.15 lists what an upgrade changes). No general package
  manager and no matrix of historical combinations.
