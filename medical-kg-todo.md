# 醫療知識圖譜診斷引擎 - TODO 清單 v2.1

## 專案狀態總覽

**當前階段**: 🚀 Phase 1 核心模組開發中
**開始日期**: 2025-10-07
**最後更新**: 2026-01-20
**預計完成**: 2026-02 (4-5 個月)

**進度指標**:
- [x] Phase 1.1-1.4: 基礎設施與KG構建 (完成)
- [ ] Phase 1.5-1.6: 模型訓練與實驗 (進行中)
- [ ] Phase 1.7-1.10: 推理管線與測試 (部分完成)
- [ ] Phase 2: 進階功能 (0/32 任務完成)
- [ ] Phase 3: ARM部署與優化 (0/24 任務完成)

---

## 已完成項目 ✅

### 核心類型系統
- [x] `src/core/types.py` - 全部核心數據類型
- [x] `src/core/protocols.py` - 30+個協議定義
- [x] `src/core/schema.py` - Schema驗證

### 知識圖譜模組
- [x] `src/kg/graph.py` - KnowledgeGraph (含metadata(), node mappings)
- [x] `src/kg/builder.py` - 知識圖譜構建器
- [x] `src/kg/preprocessing.py` - GNN預處理 (Laplacian PE, RWSE)

### 推理模組
- [x] `src/reasoning/path_reasoning.py` - PathReasoner, DirectPathFinder
- [x] `src/reasoning/explanation_generator.py` - ExplanationGenerator

### 推理管線
- [x] `src/inference/pipeline.py` - DiagnosisPipeline, PipelineConfig
- [x] `src/inference/input_validator.py` - InputValidator, ExtensibleInputValidator

### 模型框架
- [x] `src/models/gnn/shepherd_gnn.py` - ShepherdGNN框架
- [x] `src/models/gnn/layers.py` - HeteroGNNLayer, OrthologGate
- [x] `src/models/encoders/` - 位置/類型/特徵編碼器
- [x] `src/models/decoders/heads.py` - DiagnosisHead
- [x] `src/models/attention/` - AdaptiveAttentionBackend

### 本體模組
- [x] `src/ontology/hierarchy.py` - OntologyHierarchy
- [x] `src/ontology/loader.py` - OntologyLoader
- [x] `src/ontology/constraints.py` - OntologyConstraints

### 測試
- [x] 130 單元測試通過
- [x] ~52% 測試覆蓋率

---

## 當前進行中 🔄

### 🔴 P0 - 訓練流程（下一優先）
- [ ] 實現 `scripts/train_model.py`
  - [ ] 資料載入器（子圖採樣，處理16GB VRAM限制）
  - [ ] 多任務損失函數
  - [ ] 優化器配置（AdamW + 學習率調度）
  - [ ] FP16混合精度訓練
  - [ ] 模型檢查點儲存

### 🔴 P0 - 評估指標
- [ ] 實現 `src/utils/metrics.py`
  - [ ] Hits@k (k=1,5,10,20)
  - [ ] Mean Reciprocal Rank (MRR)
  - [ ] NDCG
  - [ ] 本體約束違反率

### 🟠 P1 - 資料整合
- [ ] 完善資料下載腳本
- [ ] 本體載入整合到pipeline
- [ ] 測試資料集準備

---

## 待完成項目 📋

### Phase 1 剩餘任務

#### 1.5 模型訓練 (Week 4-6)
- [ ] GNN前向傳播完整實現
- [ ] 訓練迴圈
- [ ] 驗證與早停
- [ ] 超參數調優

#### 1.6 API服務 (Week 8-9)
- [ ] `src/api/main.py` - FastAPI服務
- [ ] `/api/v2/diagnose` 端點
- [ ] `/api/v2/explain` 端點

#### 1.7 跨平台兼容 (Week 9-10)
- [ ] Windows環境腳本
- [ ] ARM環境腳本
- [ ] 容器化 (Docker)

#### 1.8 文檔 (Week 10)
- [ ] API文檔
- [ ] 部署指南
- [ ] 架構說明更新

### Phase 2 (進階功能)
- [ ] Neural ODE 時序建模
- [ ] GraphRAG 深度整合
- [ ] 模型壓縮與量化
- [ ] LLM證據解釋整合

### Phase 3 (ARM部署)
- [ ] DGX Spark環境驗證
- [ ] 模型遷移與優化
- [ ] 生產部署
- [ ] CI/CD

---

## P1 Ortholog 功能（接口已預留）

### 已預留接口
- [x] `PipelineConfig.ortholog_weight`
- [x] `PipelineConfig.ortholog_species`
- [x] `PipelineConfig.min_ortholog_confidence`
- [x] `OrthologGate` in models
- [x] `OrthologMapping` in types

### 待實現
- [ ] `src/reasoning/ortholog_reasoning.py`
- [ ] `src/data_sources/ortholog.py` 整合
- [ ] OrthologGate 實際邏輯

---

## 關鍵約束

1. **精度要求**: 醫療系統需高精度，不接受大幅犧牲精度的做法
2. **可解釋性**: 必須提供完整推理路徑與證據鏈
3. **VRAM限制**: Windows 16GB，需子圖採樣
4. **跨平台**: x86 + ARM (DGX Spark)
5. **協議合規**: 符合 `src/core/protocols.py`

---

## 風險項目

### 高風險 🔴
1. **16GB VRAM限制** - 需要子圖採樣策略
2. **ARM環境依賴** - PyG等套件可能需源碼編譯

### 中風險 🟡
1. **本體對齊品質** - 跨本體映射準確度
2. **訓練時間** - 完整PrimeKG訓練預計48h+

---

**版本**: v2.1
**最後更新**: 2026-01-20
**下次審查**: 每週一更新進度
