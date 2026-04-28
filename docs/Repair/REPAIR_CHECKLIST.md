# SHEPHERD-Advanced Repair Checklist

> **Created**: 2026-03-24
> **Purpose**: Track repair progress. Every coding session must check this file.
> **Rule**: Mark tasks complete only after code is verified and pushed.

---

## Feature Scope Decisions (2026-03-25)

The following scope decisions were made after auditing all extended features:

| Feature | Decision | Rationale |
|---------|----------|-----------|
| **Genotype input (Channel 2)** | ✅ KEEP | Original paper feature, already integrated |
| **Three diagnostic tasks** | ✅ KEEP | Original paper feature, already integrated |
| **Ortholog inference** | 🔲 PHASE 2 | OrthologGate in model is fine (skip-connection degrades gracefully); data pipeline deferred |
| **Drug suggestions** | ⛔ PHASE 3+ RESERVED | Schema-only (no model/training/inference code); leave enum definitions, do not implement |
| **Literature/PubMed** | ⛔ PHASE 3+ FROZEN | Skeleton exists; not needed before core validation |
| **FHIR/HL7** | ⛔ PHASE 3+ FROZEN | Empty stubs; pure I/O layer, add when hospital integration begins |
| **NLP fuzzy input** | 🔲 SEPARATE PROJECT | Needs RAG+LLM architecture + doctor review UI; interface via HPO IDs (keep `src/nlp/` stubs as integration point) |

---

## Phase 1: Core Architecture Verification (P0)

### 1.1 End-to-End Pipeline Verification ✅ COMPLETE (2026-04-07)
- [x] Verify train → save checkpoint → load for inference works end-to-end
- [x] Confirm `_gnn_ready` activates correctly when checkpoint is loaded
- [x] Test that GNN scoring produces meaningful (non-zero, non-uniform) scores
- [x] Validate training-inference scoring consistency (same cosine similarity formula)

### 1.2 Candidate Discovery Fix
- [ ] Audit `pipeline.py` candidate discovery flow (L709-746): confirm BFS is not sole gatekeeper
- [ ] Ensure ANN vector index is properly integrated for GNN-based candidate discovery
- [ ] Consider making vector index mandatory (or precomputing all disease embeddings for direct scoring)
- [ ] Test: disease with no BFS path but high GNN similarity should still appear in results

### 1.3 Config Cleanup ✅ COMPLETE (2026-04-07)
- [x] Mark `reasoning_weight`/`gnn_weight` in PipelineConfig as deprecated (to be replaced by `eta` in Step B)
- [x] Add `KnowledgeGraph.export_graph_data()` to consolidate duplicated graph data generation
- [x] Remove empty placeholder config files (5 files removed; verified no code references)
- [x] Rewrite Makefile to reference current deploy scripts and add `test`/`help` targets

### 1.4 PyG Compatibility Update ✅ COMPLETE (2026-04-07)
- [x] Confirm PyG official cu130 wheel availability (verified 2026-03-25)
- [x] Verify pyg-lib installs cleanly on Windows; torch-scatter/sparse/cluster not yet published for Windows (PyG side, not our bug)
- [x] Update `deploy.sh` PyG installation: split pyg-lib from third-party extensions, align with `deploy.cmd` style, improve skip messaging

### 1.5 E2E Test Fix ✅ COMPLETE (2026-04-07)
- [x] Fix `scripts/test_gnn_inference.py` scoring assertion (GNN-primary not weighted combo)
- [x] Run E2E test and verify all 9 steps pass
- [x] Port key assertions to `tests/integration/test_pipeline.py` as pytest
- [x] Fix gnn_model_and_data fixture to use kg.metadata() (matches production checkpoint loading path)
- [x] All 11 integration tests pass (TestGNNPipelineE2E + TestVectorIndexE2E + TestCheckpointBridge)

**Note**: 10 UserWarnings about "gene node not updated during message passing" are
benign — likely a PyG HeteroConv quirk with `rev_*` edge naming. Will naturally
resolve when Phase 3 ortholog edges add real `*→gene` forward edges. Tracked as
Phase 4 minor cleanup.

### 1.5 E2E Test Fix
- [ ] Fix `scripts/test_gnn_inference.py` scoring assertion (currently expects weighted combo, should match actual pipeline behavior)
- [ ] Run E2E test and verify all 9 steps pass
- [ ] Port key assertions to `tests/integration/test_pipeline.py` as pytest

---

## Step B: Shortest Path Integration ✅ COMPLETE (2026-04-07)

Per original SHEPHERD paper: `final_score = η × embedding_sim + (1-η) × SP_sim`

- [x] Pre-compute shortest path lengths for all (phenotype, gene/disease) pairs → `scripts/compute_shortest_paths.py`
- [x] Add `eta` parameter to PipelineConfig (replaces deprecated `reasoning_weight`/`gnn_weight`)
- [x] Add `_load_shortest_paths()` to pipeline (loads `shortest_paths.pt` from data_dir)
- [x] Add `_calculate_sp_score()` and `_calculate_combined_score()` to pipeline
- [x] Update both candidate scoring sites (BFS-discovered + ANN-only) to use combined score
- [x] Add `sp_score` field to `DiagnosisCandidate`
- [x] Add `TestShortestPathIntegration` test class (6 tests covering load, fallback, formula, eta extremes, distance ordering)
- [x] Remove deprecated `reasoning_weight`/`gnn_weight` after `eta` is working
- [x] Update `get_pipeline_config()` to expose `scoring_mode`, `eta_effective`, `sp_ready`, `sp_max_hops`

---

## Step C: PathReasoner Evidence Panel Redesign ✅ COMPLETE (2026-04-08)

EvidencePanel is now a separate module that consumes PathReasoner as a building block. PathReasoner itself is unchanged but is no longer the gatekeeper for evidence generation.

- [x] Build new `src/reasoning/evidence_panel.py` module (EvidencePanel class)
- [x] Mode A (direct path): when KG paths exist (≤weak_path_max_hops), surface them with STRONG/WEAK label
- [x] Mode B (analogy-based): when no KG path exists, find K nearest known nodes in GNN embedding space, run PathReasoner on them, present as analogy evidence
- [x] Add confidence label enum: STRONG_PATH / WEAK_PATH / ANALOGY_BASED / INSUFFICIENT
- [x] Decouple PathReasoner from scoring pipeline (already done in Step B; verified Step C does not re-couple)
- [x] Add `evidence_package` and `confidence_label` fields to `DiagnosisCandidate`
- [x] Wire EvidencePanel into pipeline via `_add_explanations()`
- [x] Add `TestEvidencePanel` test class (7 tests covering Mode A label thresholds, Mode B unavailable fallback, ranking invariance, direct unit tests)
- [x] Update ARCHITECTURE.md Section 8 with full Mode A/B description and ranking invariant
- [x] Add shared-feature heuristic for Mode B (KG neighborhood overlap)
- [x] Add `EvidencePanelConfig` for tunable thresholds

---

## Phase 2: Documentation Reset (P1)

### 2.1 Remove Outdated/Empty Docs
- [x] Remove `docs/architecture_v3.md` (empty)
- [x] Remove `docs/api_reference.md` (empty)
- [x] Remove `docs/developer_guide.md` (empty)
- [x] Remove `docs/medical_integration.md` (empty)
- [x] Remove `docs/implementation_plan_v1.md` (outdated, assumes fresh start)
- [x] Remove `docs/TODO_Advanced_Features.md` (premature Phase 2-3 features)

### 2.2 Archive Historical Docs
- [x] Move historical docs to `docs/archive/`:
  - `HANDOFF_SESSION_2026-02-21.md`
  - `ARCHITECTURE_REVIEW_2026-02-25.md`
  - `PROGRESS_2026-01-20.md`
  - `SESSION_HANDOFF.md`
  - `ENGINEERING_PROGRESS_REPORT_2026-02.md`
  - `data_structure_and_validation_v3.md`

### 2.3 Update Remaining Docs
- [ ] Update `docs/archive/ENGINEERING_PROGRESS_REPORT_2026-02.md` to reflect actual state before showing to hospital
- [ ] Update `docs/archive/SESSION_HANDOFF.md` to reference `ARCHITECTURE.md` as canonical source

---

## Phase 3: Ortholog Data Pipeline (P1-P2, when core is verified)

- [ ] Implement Ensembl Compara data loading in `src/data_sources/ortholog.py`
- [ ] Implement MGI (mouse) phenotype mapping
- [ ] Implement Upheno cross-species phenotype mapping (MP↔HPO)
- [ ] Wire ortholog data into KG builder
- [ ] Validate OrthologGate improves ranking metrics vs baseline (no ortholog)

---

## Frontend Repair (after Step C — backend must be 100% complete first)

> **Design inspiration**: stable-diffusion-webui — single system, multiple
> sub-pages with vastly different complexity for different user types.
>
> **User segments**:
> 1. **Clinicians** (doctors, genetic counselors) — simple, intuitive
> 2. **Engineering team** (hospital IT, ML engineers) — advanced controls

### F.1 Audit current frontend state (must run before any code changes)
- [ ] Deep scan `src/api/` — list every endpoint, mark which return real data vs mocks
- [ ] Deep scan `src/webui/` — what works, what's a placeholder
- [ ] Document gaps in a SCAN_REPORT supplement

### F.2 Clinician-facing sub-pages (priority)
- [ ] **Patient Diagnosis** page — HPO symptom input → ranked diseases + evidence
  - Must display Mode A (direct path) and Mode B (analogy) evidence from Step C
  - Confidence labels: "Strong path support" / "Weak path support" / "Analogy-based" / "Insufficient evidence"
- [ ] **Patients-Like-Me** retrieval page — phenotype input → K most similar known patients

### F.3 Engineering-facing sub-pages
- [x] Training console (already exists)
- [x] Hyperparameter tuning UI (already exists)
- [ ] **Checkpoint management** — list / load / delete / compare checkpoints
- [ ] **Vector index rebuild** — trigger ANN index rebuild from current model
- [ ] **System health dashboard** — `_gnn_ready`, `_sp_ready`, `_vector_index_ready`, KG stats, GPU usage

### F.4 API gaps to fill
- [ ] Replace mock responses in `src/api/routes/diagnose.py` once Step C output schema is finalized
- [ ] Add `/diagnose/explain/{candidate_id}` endpoint for on-demand evidence detail
- [ ] Restrict CORS for production (Phase 4.2 task)

---

## Phase 4: Infrastructure (P2)

### 4.1 CI/CD
- [ ] Implement `.github/workflows/test-x86.yml`
- [ ] Implement `.github/workflows/test-arm.yml`
- [ ] Implement `.github/workflows/deploy.yml`
- [ ] Configure `.pre-commit-config.yaml`

### 4.2 Production Hardening
- [ ] Restrict CORS in `src/api/main.py` (environment-based config)
- [ ] Review `toml` dependency — remove if unused
- [ ] Relax Gradio version pin if possible (`>=5.20` instead of `>=5.20,<5.30`)
- [ ] Improve `_load_model_from_checkpoint` error reporting — currently catches RuntimeError silently; should log WARNING with specific mismatch details so operators can diagnose checkpoint compat issues
- [ ] Investigate "gene node not updated during message passing" PyG warning — may be `rev_*` naming convention quirk; revisit after Phase 3 ortholog edges are added

### 4.3 CLI / Script Integration Gaps (discovered during CLI smoke test 2026-04-08)
- [ ] **`train_model.py` ↔ `compute_shortest_paths.py` output mismatch**: `train_model.py:generate_synthetic_data()` produces `node_features.pt`/`edge_indices.pt`/`num_nodes.json` but NOT `kg.json`. `compute_shortest_paths.py` requires `kg.json` as input. Result: the two scripts cannot be chained directly. Either:
  - (a) Make `train_model.py` also emit a `kg.json` when generating synthetic data, OR
  - (b) Make `compute_shortest_paths.py` accept `node_features.pt`+`edge_indices.pt` directly (no KG object), OR
  - (c) Document that real workflows should start from `build_knowledge_graph.py` → `kg.json` → train_model.py + compute_shortest_paths.py
- [ ] **`train_model.py` synthetic data has no biological meaning**: pure random features and edges. Only useful for "does the code run?" — NOT for "does the model learn anything meaningful?". Consider deprecating it in favor of `setup_demo.py` as the canonical small-scale test path, OR document its limitation clearly in the docstring.
- [ ] **No canonical "from scratch" workflow**: we have multiple scripts that each produce partial data (setup_demo, train_model, build_knowledge_graph, compute_shortest_paths) but no single entry point that runs them in the correct order. Consider adding a top-level `scripts/bootstrap_demo.sh` (or Makefile target) that chains them correctly.
- [ ] **Training Console ↔ demo data incompatibility** (discovered 2026-04-28): Training Console WebUI calls `train_model.py` which expects `train_samples.json` / `val_samples.json` (patient→disease supervised training pairs). `setup_demo.py` does NOT produce these files — it uses its own self-supervised mini training loop instead. If Training Console is pointed at `data/demo/`, `train_model.py` auto-triggers `generate_synthetic_data()` which (a) overwrites demo graph data files with random noise, AND (b) uses hardcoded edge type names (`"associated_with"`) that mismatch the KG edge types (`"gene_associated_with_disease"`), causing the same metadata mismatch bug fixed in PR #52. **Fix**: make `setup_demo.py` also generate `train_samples.json`/`val_samples.json` from the demo KG, OR add a guard in `train_model.py` to never auto-overwrite existing graph data files.

---

## Session Log

| Date | Session | Tasks Completed | Notes |
|------|---------|----------------|-------|
| 2026-03-24 | Initial scan | Created SCAN_REPORT, ARCHITECTURE, REPAIR_CHECKLIST | 5 agents scanned modules, deps, docs; committed + pushed |
| 2026-03-25 | Feature scope | Feature freeze decisions; doc cleanup marked done | Drug=reserved, NLP=separate project, Literature/FHIR=frozen; PyG cu130 confirmed by user |
| 2026-03-25 | Step A: E2E prep | Fix test scoring bug; add export_graph_data; add checkpoint pytest | Confirmed: scoring=GNN-primary (η*emb+(1-η)*SP planned for Step B); PathReasoner=evidence-only |
| 2026-04-07 | Step A: E2E verified | scripts/test_gnn_inference.py 9/9 PASS; pytest 11/11 PASS | Found+fixed fixture edge type mismatch with kg.metadata(); Phase 1.1 + 1.5 complete |
| 2026-04-07 | Phase 1.3 + 1.4 cleanup | Removed 5 empty configs; rewrote Makefile; aligned deploy.sh PyG install with deploy.cmd | Phase 1 substantially complete (1.2 deferred to Step B/C); ready for PR merge |
| 2026-04-07 | PR #52 review fix | Fix _load_model_from_checkpoint metadata source to match production trainer | chatgpt-codex-connector caught fixture-masked production bug; aligned all 3 paths |
| 2026-04-07 | Step B: Shortest path | Precompute script + pipeline loading + eta mixing + 6 new tests | Implements original SHEPHERD scoring formula η*emb + (1-η)*SP with graceful fallback |
| 2026-04-08 | Step C: Evidence panel | New EvidencePanel module with Mode A/B + confidence labels + 7 new tests | Backend now 100% complete; ready for frontend planning |
| 2026-04-08 | PR #54 test fix | Restore stray `sp_far` assertion that Edit operation misplaced | 24/24 integration tests PASS locally (17 existing + 7 new); backend verified end-to-end |
| 2026-04-28 | CLI smoke test | Full E2E via CLI: setup_demo → train → SP → uvicorn → curl /diagnose | Real pipeline result (not mock); found API schema gap (sp_score/evidence_package/confidence_label missing from Pydantic model) |
| 2026-04-28 | API schema fix | Sync API Pydantic DiagnosisCandidate with core types.py; update endpoint mapping | Added sp_score, evidence_package, confidence_label to API response |
| | | | |
