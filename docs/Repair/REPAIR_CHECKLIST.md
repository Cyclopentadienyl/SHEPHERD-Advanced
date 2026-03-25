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
- [ ] Mark `reasoning_weight`/`gnn_weight` in PipelineConfig as deprecated (to be replaced by `eta` in Step B)
- [ ] Remove empty placeholder config files (`configs/base_config.yaml`, `data_config.yaml`, `model_config.yaml`, `medical_standards.yaml`, `deployment_config.yaml`)
- [ ] Update Makefile to reference current deploy scripts (`deploy.sh`/`deploy.cmd`)
- [ ] Add `KnowledgeGraph.export_graph_data()` to consolidate duplicated graph data generation

### 1.4 PyG Compatibility Update
- [x] Confirm PyG official cu130 wheel availability (verified 2026-03-25)
- [ ] Update `deploy.sh` line 108 comment — remove outdated "torch 2.9.1 breaks pyg-lib" warning
- [ ] Consider aligning `deploy.sh` PyG install with `deploy.cmd` style (`--only-binary :all:` for cleaner failure)

### 1.5 E2E Test Fix
- [ ] Fix `scripts/test_gnn_inference.py` scoring assertion (currently expects weighted combo, should match actual pipeline behavior)
- [ ] Run E2E test and verify all 9 steps pass
- [ ] Port key assertions to `tests/integration/test_pipeline.py` as pytest

---

## Step B: Shortest Path Integration (after Phase 1 verified)

Per original SHEPHERD paper, scoring should be: `final_score = η × embedding_sim + (1-η) × SP_sim`

- [ ] Pre-compute shortest path lengths for all (phenotype, gene/disease) pairs → store as lookup table
- [ ] Add `eta` parameter to PipelineConfig (replaces deprecated `reasoning_weight`/`gnn_weight`)
- [ ] Modify `_calculate_gnn_score()` to incorporate SP similarity signal
- [ ] Update E2E tests for new scoring formula
- [ ] Remove deprecated `reasoning_weight`/`gnn_weight` after `eta` is working

---

## Step C: PathReasoner Evidence Panel Redesign (after Step B)

PathReasoner becomes pure post-hoc explanation layer (no scoring involvement):

- [ ] Mode A (direct path): when KG paths exist (≤3 hops), show them as direct evidence
- [ ] Mode B (analogy-based): when no KG path exists (zero-shot):
  - Find K nearest known genes in GNN embedding space
  - Run PathReasoner on those known genes
  - Present as analogy evidence with confidence labels
- [ ] Add confidence labels: "Strong path support" / "Weak path support" / "Analogy-based" / "Insufficient evidence"
- [ ] Decouple PathReasoner from scoring pipeline completely
- [ ] Update ExplanationGenerator to support both modes

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

---

## Session Log

| Date | Session | Tasks Completed | Notes |
|------|---------|----------------|-------|
| 2026-03-24 | Initial scan | Created SCAN_REPORT, ARCHITECTURE, REPAIR_CHECKLIST | 5 agents scanned modules, deps, docs; committed + pushed |
| 2026-03-25 | Feature scope | Feature freeze decisions; doc cleanup marked done | Drug=reserved, NLP=separate project, Literature/FHIR=frozen; PyG cu130 confirmed by user |
| 2026-03-25 | Step A: E2E prep | Fix test scoring bug; add export_graph_data; add checkpoint pytest | Confirmed: scoring=GNN-primary (η*emb+(1-η)*SP planned for Step B); PathReasoner=evidence-only |
| | | | |
