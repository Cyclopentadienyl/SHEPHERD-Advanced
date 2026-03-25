# SHEPHERD-Advanced Full Repository Scan Report

> **Date**: 2026-03-24 (updated 2026-03-25)
> **Purpose**: Comprehensive audit of all modules, dependencies, and documentation to inform architectural repair.
> **Anchor**: Original SHEPHERD paper (npj Digital Medicine 2025) — GNN-primary architecture with PathReasoner as auxiliary explainability layer.
>
> **2026-03-25 update**: Feature scope decisions finalized. Drug suggestions = Phase 3+ reserved (schema-only, not in model/training/inference). NLP = separate project (RAG+LLM with doctor review UI). Literature/FHIR = frozen. Ortholog = Phase 2 (after core verified). PyG confirmed officially supporting torch 2.9.0+cu130.

---

## 1. Executive Summary

### 1.1 Architecture Verification Result

The codebase was suspected of having deviated toward a "PathReasoner-primary, GNN-auxiliary" architecture. After thorough scanning, the actual situation is **more nuanced**:

| Aspect | Status | Detail |
|--------|--------|--------|
| **Scoring logic** | ✅ Correct | `_calculate_gnn_score()` implements cosine similarity; GNN score is primary when available |
| **Fallback design** | ✅ Correct | When `_gnn_ready=False`, PathReasoner scoring takes over as documented fallback |
| **Training loop** | ✅ Correct | MultiTaskLoss with DiagnosisLoss(weight=1.0) trains GNN embeddings end-to-end |
| **Training-inference consistency** | ✅ Correct | Both use identical cosine similarity on L2-normalized embeddings |
| **Candidate discovery** | ⚠️ Concern | BFS is called first; only BFS-discovered diseases are scored unless ANN vector index is enabled |
| **GNN activation** | ⚠️ Conditional | `_gnn_ready` requires trained model checkpoint + graph data; without these, system is PathReasoner-only |
| **End-to-end pipeline** | ❓ Unverified | train → save checkpoint → load for inference chain not confirmed working |

### 1.2 Historical Context

- **Feb 2026 docs** (HANDOFF_2026-02-21, ARCHITECTURE_REVIEW_2026-02-25) documented `_calculate_gnn_score()` returning hardcoded 0.0.
- **Current code** (Mar 2026) shows this function is now fully implemented (pipeline.py:1070-1142).
- The docs are outdated; the code has been partially fixed since then.

---

## 2. Module Scan Results

### 2.1 Models (`src/models/`)

| Module | File | Lines | Status | Role |
|--------|------|-------|--------|------|
| GNN Core | `gnn/shepherd_gnn.py` | ~500 | **KEEP** | PRIMARY — 4-layer heterogeneous GNN, generates node embeddings |
| GNN Layers | `gnn/layers.py` | ~354 | **KEEP** | HeteroGNNLayer + OrthologGate (post-message-passing modulation) |
| Feature Encoder | `encoders/feature_encoder.py` | ~87 | **KEEP** | Input projection to unified hidden space |
| Type Encoder | `encoders/type_encoder.py` | ~100 | **KEEP** | Learnable node type embeddings |
| Position Encoder | `encoders/position_encoder.py` | ~100 | **KEEP** | Laplacian PE + RWSE |
| Flex Attention | `attention/flex_attention.py` | ~100 | **KEEP** | PyTorch 2.5+ FlexAttention for heterogeneous graphs |
| Decoder Heads | `decoders/heads.py` | ~150 | **KEEP** | DiagnosisHead (primary), LinkPredictionHead, ExplanationHead |

**Assessment**: Model layer is well-designed and correctly implements GNN-primary architecture. No changes needed.

### 2.2 Training (`src/training/`)

| Module | File | Lines | Status | Notes |
|--------|------|-------|--------|-------|
| Trainer | `trainer.py` | ~911 | **KEEP** | Complete training loop with mixed precision, gradient accumulation |
| Loss Functions | `loss_functions.py` | ~602 | **KEEP** | DiagnosisLoss(1.0) + LinkPrediction(0.5) + Contrastive(0.3) + Ortholog(0.2) |

**Key findings**:
- Training is GNN-centric: all losses operate on GNN-generated embeddings
- PathReasoner has zero involvement in training (no gradient flow to path scoring)
- Scoring formula in trainer matches inference exactly: `cosine_sim(normalize(patient), normalize(disease))`

### 2.3 Inference (`src/inference/`)

| Module | File | Lines | Status | Notes |
|--------|------|-------|--------|-------|
| Pipeline | `pipeline.py` | ~1318 | **ADJUST** | Scoring correct; candidate discovery needs attention |

**Detailed findings**:

1. **Scoring** (L964-967): GNN-primary when `_gnn_ready`, PathReasoner fallback otherwise. ✅
2. **GNN scoring** (L1070-1142): Cosine similarity with proper L2 normalization. ✅
3. **Candidate discovery** (L709-746): BFS paths found first, then optional ANN candidates. ⚠️
4. **`_gnn_ready` activation** (L290-344): Requires model + graph_data. If neither provided, GNN is disabled. ⚠️
5. **Unused config fields** (L115-121): `reasoning_weight` and `gnn_weight` are never used in scoring. Should clean up.

**The candidate discovery concern**: When ANN vector index is disabled (default), only diseases reachable via BFS paths are scored. This means GNN's ability to find latent associations (its key advantage per the paper) is limited by KG path completeness. The ANN vector index partially addresses this but is optional.

### 2.4 Reasoning (`src/reasoning/`)

| Module | File | Lines | Status | Notes |
|--------|------|-------|--------|-------|
| PathReasoner | `path_reasoning.py` | ~549 | **KEEP** | BFS path enumeration, correctly auxiliary |
| Explanation Generator | `explanation_generator.py` | ~500 | **KEEP** | Human-readable evidence paths |
| Constraint Checker | `constraint_checker.py` | 0 | **EMPTY** | Phase 2 stub |

**Assessment**: PathReasoner is correctly positioned as explanation-only when GNN is available. Its BFS scoring is only used as fallback.

### 2.5 Core Infrastructure (`src/core/`)

| Module | File | Lines | Status | Notes |
|--------|------|-------|--------|-------|
| Types | `types.py` | ~537 | **KEEP** | Comprehensive type system (NodeType, EdgeType, DataSource enums) |
| Schema | `schema.py` | ~682 | **KEEP** | KG schema with 11 metapaths |
| Protocols | `protocols.py` | ~1354 | **KEEP** | Interface contracts for all components |

### 2.6 Knowledge Graph (`src/kg/`)

| Module | File | Lines | Status | Notes |
|--------|------|-------|--------|-------|
| Graph | `graph.py` | ~700 | **KEEP** | Dict-based KG with PyG export |
| Data Loader | `data_loader.py` | ~890 | **KEEP** | SubgraphSampler (neighbor/random_walk/k-hop), NegativeSampler |
| Builder | `builder.py` | ~712 | **KEEP** | Incremental KG construction from multiple sources |
| Entity Linker | `entity_linker.py` | 0 | **EMPTY** | Phase 2 stub |

### 2.7 Ontology (`src/ontology/`)

| Module | File | Lines | Status | Notes |
|--------|------|-------|--------|-------|
| Hierarchy | `hierarchy.py` | ~640 | **KEEP** | Pronto backend, semantic similarity (Resnik/Lin/JC) |
| Loader | `loader.py` | ~373 | **KEEP** | OBO/OWL loading with caching |
| Constraints | `constraints.py` | ~309 | **KEEP** | Phenotype consistency validation |

### 2.8 Retrieval (`src/retrieval/`)

| Module | File | Lines | Status | Notes |
|--------|------|-------|--------|-------|
| Vector Index | `vector_index.py` | ~305 | **KEEP** | Factory with platform detection |
| cuVS Backend | `backends/cuvs_backend.py` | ~321 | **KEEP** | GPU-accelerated (Linux only) |
| Voyager Backend | `backends/voyager_backend.py` | ~239 | **KEEP** | Cross-platform CPU fallback |

### 2.9 Data Sources (`src/data_sources/`)

| Module | File | Lines | Status | Notes |
|--------|------|-------|--------|-------|
| Ortholog | `ortholog.py` | ~465 | **KEEP structure** | Skeleton — Phase 2 |
| PubMed | `pubmed.py` | ~451 | **KEEP structure** | Skeleton — Phase 2 |

### 2.10 NLP (`src/nlp/`) — CRITICAL GAP

| Module | File | Lines | Status | Notes |
|--------|------|-------|--------|-------|
| HPO Matcher | `hpo_matcher.py` | 0 | **EMPTY** | Required for fuzzy input |
| Entity Recognizer | `entity_recognizer.py` | 0 | **EMPTY** | Required for free-text parsing |
| Symptom Extractor | `symptom_extractor.py` | 0 | **EMPTY** | Required for clinical notes |
| Clinical BERT | `clinical_bert.py` | 0 | **EMPTY** | Biomedical NLP encoding |

**This is the most critical functional gap.** The project's stated goal includes fuzzy input support, but all NLP modules are empty.

### 2.11 Medical Standards (`src/medical_standards/`)

All files empty (fhir_adapter, icd_mapper, snomed_mapper, hiss_adapter). Phase 2.

### 2.12 API & Web UI

| Module | File | Lines | Status | Notes |
|--------|------|-------|--------|-------|
| FastAPI App | `api/main.py` | ~467 | **KEEP** | Well-structured, graceful degradation |
| Diagnose Route | `api/routes/diagnose.py` | ~321 | **KEEP** | Mock responses when pipeline unavailable |
| Gradio Dashboard | `webui/app.py` | ~100 | **KEEP** | Training console + placeholders |

### 2.13 Config & Utils

| Module | File | Lines | Status | Notes |
|--------|------|-------|--------|-------|
| Hyperparameters | `config/hyperparameters.py` | ~1072 | **KEEP** | Comprehensive parameter management |
| Metrics | `utils/metrics.py` | ~832 | **KEEP** | Ranking, ontology, evidence, training metrics |
| Config Validator | `config/config_validator.py` | 0 | **EMPTY** | Phase 2 stub |

---

## 3. Dependency Scan Results

### 3.1 Core Dependencies (pyproject.toml)

| Package | Version | Category | Assessment |
|---------|---------|----------|------------|
| `pronto` | >=2.7 | ESSENTIAL | Ontology processing |
| `networkx` | >=3.2 | ESSENTIAL | Graph operations |
| `pandas` | >=2.2 | ESSENTIAL | Data processing |
| `numpy` | >=2.0 | ESSENTIAL | Numerical core |
| `scipy` | >=1.14 | ESSENTIAL | Scientific computing |
| `pydantic` | >=2.7 | ESSENTIAL | Data validation + FastAPI |
| `pydantic-settings` | >=2.2 | ESSENTIAL | Environment config |
| `fastapi` | >=0.115 | ESSENTIAL | REST API |
| `uvicorn[standard]` | >=0.29 | ESSENTIAL | ASGI server |
| `voyager` | >=2.0 | ESSENTIAL | Vector retrieval (CPU) |
| `pyyaml` | >=6.0 | USEFUL | Config parsing |
| `jsonschema` | >=4.21 | USEFUL | Schema validation |
| `gradio` | >=5.20,<5.30 | USEFUL | Web UI (restrictive pin) |
| `psutil` | >=5.9 | USEFUL | System monitoring |
| `tqdm` | >=4.66 | USEFUL | Progress bars |
| `requests` | >=2.31 | USEFUL | HTTP client |
| `python-dotenv` | >=1.0 | USEFUL | Env loading |
| `packaging` | >=23.0 | USEFUL | Version parsing |
| `toml` | >=0.10 | **QUESTIONABLE** | Likely unused (YAML is used everywhere) |

**PyTorch stack** (installed separately via deploy scripts, not in pyproject.toml):
- `torch==2.9.0` (pinned), `torch_geometric`, optional: `pyg-lib`, `torch-sparse`, `torch-scatter`, `torch-cluster`
- GPU vector: `cuvs-cu12`/`cuvs-cu13` (Linux only)
- Attention: `flash-attn` (x86 only), `xformers`, `sageattention`

### 3.2 Deploy Scripts

| Script | Platform | Status | Notes |
|--------|----------|--------|-------|
| `deploy.sh` | Linux x86/ARM | ✅ Active | 4-stage: venv → PyTorch → deps → validate |
| `deploy.cmd` | Windows x86 | ✅ Active | Same 4-stage process |
| `launch_shepherd.sh` | Linux | ✅ Active | Activates venv, calls shep_launch.py |
| `launch_shepherd.cmd` | Windows | ✅ Active | Same flow |
| `scripts/launch/shep_launch.py` | All | ✅ Active | Dynamic accelerator selection + auto-install |

**No deploy scripts install packages not declared in pyproject.toml or the PyTorch stack.** The separation is clean.

### 3.3 Configuration Issues

| File | Status | Issue |
|------|--------|-------|
| `configs/deployment.yaml` | ✅ Active | Unified platform config (v3.2) |
| `configs/accelerators.json` | ✅ Active | Accelerator specs |
| `configs/base_config.yaml` | ❌ Empty | Placeholder — remove or populate |
| `configs/data_config.yaml` | ❌ Empty | Placeholder — remove or populate |
| `configs/model_config.yaml` | ❌ Empty | Placeholder — remove or populate |
| `configs/medical_standards.yaml` | ❌ Empty | Placeholder — remove or populate |
| `configs/deployment_config.yaml` | ❌ Empty | Deprecated — remove |
| `Makefile` | ⚠️ Outdated | References non-existent scripts |

### 3.4 CI/CD

All GitHub workflow files are empty stubs (deploy.yml, test-x86.yml, test-arm.yml). No automated testing.

---

## 4. Documentation Scan Results

### 4.1 Files to REMOVE (empty or outdated)

| File | Reason |
|------|--------|
| `architecture_v3.md` | Empty file (0 bytes) |
| `api_reference.md` | Empty file (0 bytes) |
| `developer_guide.md` | Empty file (0 bytes) |
| `medical_integration.md` | Empty file (0 bytes) |
| `implementation_plan_v1.md` | Outdated (Nov 2025), assumes fresh start; project is 70%+ complete |
| `TODO_Advanced_Features.md` | Premature Phase 2-3 features; core pipeline needs verification first |

### 4.2 Files to KEEP (accurate/useful)

| File | Reason |
|------|--------|
| `DIRECTORY_STRUCTURE.md` | Infrastructure reference, won't become outdated |
| `module_dependencies.md` | Layered architecture reference with import-linter rules |
| `TRAINING_MODULE_AUDIT_2026-01-25.md` | Accurate training audit, correctly identifies coupling issues |
| `MODULE_SCAN_REPORT_2026-01-26.md` | Accurate code inventory (86 files, 49 implemented, 37 empty) |
| `HANDOFF_SESSION_2026-02-23.md` | Dashboard/Gradio specs, no architectural conflicts |

### 4.3 Files to ARCHIVE (historically valuable but superseded)

| File | Reason |
|------|--------|
| `HANDOFF_SESSION_2026-02-21.md` | Documented fatal GNN issue — now partially fixed, keep for history |
| `ARCHITECTURE_REVIEW_2026-02-25.md` | Honest audit — some findings still relevant, some outdated |
| `PROGRESS_2026-01-20.md` | Status snapshot, superseded by this report |
| `SESSION_HANDOFF.md` | Session prompt, needs rewrite |
| `ENGINEERING_PROGRESS_REPORT_2026-02.md` | Hospital-facing report — needs update before showing externally |
| `data_structure_and_validation_v3.md` | Mix of implemented and aspirational — needs split |

---

## 5. Identified Issues (Priority Ordered)

### P0 — Critical

1. **End-to-end pipeline unverified**: train → checkpoint → inference chain not confirmed working
2. **Candidate discovery gatekeeper**: Without ANN vector index, only BFS-discovered diseases are scored, limiting GNN's latent association discovery
3. **NLP modules entirely empty**: Fuzzy input (free-text → HPO) is a stated project goal but has zero implementation

### P1 — High

4. **Unused config fields**: `reasoning_weight`/`gnn_weight` in PipelineConfig are never used — misleading
5. **Empty placeholder configs**: 5 empty YAML files in `configs/` create confusion
6. **Makefile outdated**: References non-existent deployment scripts
7. **Outdated documentation**: Multiple docs describe Feb 2026 state that has been partially fixed

### P2 — Medium

8. **Ortholog/PubMed data sources**: Phase 2 skeletons, not wired up
9. **CI/CD empty**: No automated testing workflows
10. **Gradio version pin restrictive**: `>=5.20,<5.30` is very narrow
11. **CORS allows all origins**: Should restrict for production
12. **`toml` dependency questionable**: Project uses YAML everywhere

### P3 — Low

13. **Medical standards modules empty**: Phase 2-3 (FHIR, ICD, SNOMED)
14. **Config validator empty**: Phase 2
15. **Pre-commit config empty**: Not configured

---

## 6. Module Disposition Summary

```
KEEP (no changes):     31 modules
ADJUST (minor fixes):   1 module  (inference/pipeline.py)
KEEP STRUCTURE:         4 modules (Phase 2 skeletons)
EMPTY (Phase 2 stubs): 13 modules
REMOVE (docs only):     6 markdown files
ARCHIVE (docs):         6 markdown files
```
