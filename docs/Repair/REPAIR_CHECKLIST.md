# SHEPHERD-Advanced Repair Checklist

> **Created**: 2026-03-24
> **Purpose**: Track repair progress. Every coding session must check this file.
> **Rule**: Mark tasks complete only after code is verified and pushed.

---

## Phase 1: Core Architecture Verification (P0)

### 1.1 End-to-End Pipeline Verification
- [ ] Verify train → save checkpoint → load for inference works end-to-end
- [ ] Confirm `_gnn_ready` activates correctly when checkpoint is loaded
- [ ] Test that GNN scoring produces meaningful (non-zero, non-uniform) scores
- [ ] Validate training-inference scoring consistency (same cosine similarity formula)

### 1.2 Candidate Discovery Fix
- [ ] Audit `pipeline.py` candidate discovery flow (L709-746): confirm BFS is not sole gatekeeper
- [ ] Ensure ANN vector index is properly integrated for GNN-based candidate discovery
- [ ] Consider making vector index mandatory (or precomputing all disease embeddings for direct scoring)
- [ ] Test: disease with no BFS path but high GNN similarity should still appear in results

### 1.3 Config Cleanup
- [ ] Remove or properly document unused `reasoning_weight`/`gnn_weight` fields in PipelineConfig
- [ ] Remove empty placeholder config files (`configs/base_config.yaml`, `data_config.yaml`, `model_config.yaml`, `medical_standards.yaml`, `deployment_config.yaml`)
- [ ] Update Makefile to reference current deploy scripts (`deploy.sh`/`deploy.cmd`)

---

## Phase 2: Documentation Reset (P1)

### 2.1 Remove Outdated/Empty Docs
- [ ] Remove `docs/architecture_v3.md` (empty)
- [ ] Remove `docs/api_reference.md` (empty)
- [ ] Remove `docs/developer_guide.md` (empty)
- [ ] Remove `docs/medical_integration.md` (empty)
- [ ] Remove `docs/implementation_plan_v1.md` (outdated, assumes fresh start)
- [ ] Remove `docs/TODO_Advanced_Features.md` (premature Phase 2-3 features)

### 2.2 Archive Historical Docs
- [ ] Move historical docs to `docs/archive/` (or clearly mark as superseded):
  - `HANDOFF_SESSION_2026-02-21.md`
  - `ARCHITECTURE_REVIEW_2026-02-25.md`
  - `PROGRESS_2026-01-20.md`
  - `SESSION_HANDOFF.md`
  - `ENGINEERING_PROGRESS_REPORT_2026-02.md`
  - `data_structure_and_validation_v3.md`

### 2.3 Update Remaining Docs
- [ ] Update `ENGINEERING_PROGRESS_REPORT_2026-02.md` to reflect actual state before showing to hospital
- [ ] Update `SESSION_HANDOFF.md` to reference `ARCHITECTURE.md` as canonical source

---

## Phase 3: Missing Functionality (P1-P2)

### 3.1 NLP / Fuzzy Input (Critical Gap)
- [ ] Implement `src/nlp/hpo_matcher.py` — fuzzy string matching (free-text → HPO terms)
- [ ] Implement `src/nlp/symptom_extractor.py` — clinical note parsing
- [ ] Implement `src/nlp/entity_recognizer.py` — biomedical NER
- [ ] Implement `src/nlp/clinical_bert.py` — biomedical text encoding
- [ ] Implement `src/kg/entity_linker.py` — link recognized entities to KG nodes
- [ ] Wire NLP modules into API diagnose endpoint (optional `phenotype_text` field)

### 3.2 Data Source Integration (Phase 2)
- [ ] Implement ortholog data loading in `src/data_sources/ortholog.py`
- [ ] Implement PubMed/Pubtator integration in `src/data_sources/pubmed.py`
- [ ] Wire data sources into KG builder

### 3.3 Constraint Validation
- [ ] Implement `src/reasoning/constraint_checker.py`
- [ ] Add inference result validation against ontology constraints

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
- [ ] Add performance warnings in validation script when using PyG fallback kernels

---

## Session Log

| Date | Session | Tasks Completed | Notes |
|------|---------|----------------|-------|
| 2026-03-24 | Initial scan | Phase A complete | Created SCAN_REPORT, ARCHITECTURE, this checklist |
| | | | |
