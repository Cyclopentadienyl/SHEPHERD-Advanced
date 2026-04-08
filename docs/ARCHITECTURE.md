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

### 2.1 The Non-Negotiable Rule

> **GNN generates embeddings and drives all scoring/ranking.**
> **PathReasoner provides explainable evidence paths for clinician review.**
> **PathReasoner NEVER gates candidate discovery or drives scoring when GNN is available.**

This is the fundamental design principle derived from the original SHEPHERD paper. The GNN's key advantage over traditional path-based methods is **inductive generalization** — the ability to:
- Infer relationships for diseases never seen in training (83% of test diseases were unseen)
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
- **Disease split**: 83% of test diseases never seen in training

### Training → Inference Bridge
1. Training produces model checkpoint
2. Inference loads checkpoint → runs GNN forward pass → caches all node embeddings
3. `_gnn_ready = True` → GNN scoring activates
4. Without checkpoint: system falls back to PathReasoner-only (degraded mode)

---

## 6. Module Architecture

```
Layer 0 (Foundation):  src/core/          types, protocols, schema
Layer 1 (Config):      src/config/        hyperparameters, validation
Layer 2 (Data):        src/ontology/      HPO/MONDO/GO via pronto
                       src/data_sources/  ortholog, pubmed (Phase 2)
Layer 3 (Knowledge):   src/kg/            graph, builder, data_loader
                       src/nlp/           fuzzy input (NOT YET IMPLEMENTED)
Layer 4 (Models):      src/models/        GNN, encoders, decoders, attention
                       src/retrieval/     vector index (cuVS/Voyager)
Layer 5 (Reasoning):   src/reasoning/     path_reasoning (auxiliary), explanation
Layer 6 (Training):    src/training/      trainer, loss_functions
Layer 7 (Inference):   src/inference/      pipeline (GNN-primary scoring)
Layer 8 (Interface):   src/api/           FastAPI REST endpoints
                       src/webui/         Gradio dashboard
```

**Import rules** (enforced by import-linter):
- Lower layers cannot import higher layers
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
| Vector Index | `src/retrieval/vector_index.py` | ANN search for novel candidate discovery |

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
- **`shortest_paths.pt` missing + `sp_optional=True`** (default): pipeline reports `scoring_mode="gnn_only"`, effective η = 1.0
- **`shortest_paths.pt` missing + `sp_optional=False`**: pipeline init refuses to set `_gnn_ready=True`
- **No GNN model loaded**: pipeline falls back to `confidence_score = path_reasoning_aggregate_score` (PathReasoner only)

### PathReasoner Role (explanation only)
PathReasoner does NOT contribute to scoring. After ranking is determined, it generates evidence for clinician review:
- **Mode A** (direct path): when KG paths exist, show them
- **Mode B** (analogy-based): when no KG path exists, find K nearest known genes in embedding space and show their paths as analogy evidence

---

## 9. Platform Support

| Platform | GPU Attention | Vector Backend | Deploy Script |
|----------|--------------|----------------|--------------|
| Linux x86_64 | FlashAttention-2 / cuDNN SDPA | cuVS (GPU) → Voyager (CPU) | `deploy.sh` |
| Linux aarch64 (DGX Spark) | cuDNN SDPA (no FlashAttn) | cuVS (GPU) → Voyager (CPU) | `deploy.sh` |
| Windows x86_64 | FlashAttention-2 / torch SDPA | Voyager (CPU only) | `deploy.cmd` |

PyTorch 2.9.0 + CUDA 13.0 unified across all platforms.

---

## 10. Design Constraints

1. **Zero real patient data for training**: All training uses simulated patients
2. **16GB VRAM support**: Subgraph sampling keeps memory bounded
3. **Offline-capable**: No internet required after initial data/model download
4. **Clinician-facing output**: Every prediction must include explainable evidence paths
5. **No false completeness**: Empty stub modules must not be presented as functional
