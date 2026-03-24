# SHEPHERD-Advanced 專案交接文件

**生成日期**: 2026-02-21
**生成者**: Claude 4.6 Opus (Session: claude/reload-repo-report-HZipv)
**用途**: 跨 Claude Code Session 的專案狀態交接

---

## 一、專案背景

### 1.1 原始論文
**SHEPHERD: Few-shot learning for phenotype-driven diagnosis of patients with rare genetic diseases**
- 來源: [npj Digital Medicine](https://www.nature.com/articles/s41746-024-01332-0) | [Harvard Zitnik Lab](https://zitniklab.hms.harvard.edu/projects/SHEPHERD/) | [GitHub](https://github.com/mims-harvard/SHEPHERD)
- 核心思想: 使用 **圖神經網路 (GNN)** 在醫療知識圖譜上進行 few-shot learning，實現罕見疾病的可解釋性診斷

### 1.2 原論文核心架構
```
患者表型 (HPO terms)
       ↓
┌─────────────────────────────────┐
│  Knowledge Graph Embedding      │
│  - Gene/Disease/Phenotype nodes │
│  - Heterogeneous GNN            │
└─────────────────────────────────┘
       ↓
┌─────────────────────────────────┐
│  Patient Encoder                │
│  - Phenotype subgraph           │
│  - Attention aggregation        │
└─────────────────────────────────┘
       ↓
┌─────────────────────────────────┐
│  Metric Learning                │
│  - Patient ↔ Disease similarity │
│  - Few-shot classification      │
└─────────────────────────────────┘
       ↓
診斷結果 + 解釋性路徑
```

### 1.3 醫院方需求擴展 (工程藍圖 v2.0)
參考: `medical-kg-blueprint.md`

額外需求:
- 分層本體感知 GNN (HPO/MONDO/GO 層次結構)
- 知識超圖 (高階關係)
- DR.KNOWS 式路徑推理
- Neural ODE 時序建模
- 本體約束解碼
- 跨平台部署 (Windows x86 + DGX Spark ARM)
- 向量檢索加速 (cuVS/Voyager)

---

## 二、架構偏離分析 (CRITICAL)

### 2.1 核心問題: GNN 未接入推理管線

| 組件 | 原論文設計 | 實際實現 | 偏離程度 |
|------|-----------|---------|---------|
| **GNN 模型** | 核心推理引擎 | ✅ 完整實現 (shepherd_gnn.py) | - |
| **訓練流程** | 兩階段訓練 | ✅ 完整實現 (trainer.py) | - |
| **推理管線** | GNN 產生診斷分數 | ⚠️ **PathReasoner 替代** | **嚴重** |
| **GNN 推理** | 直接使用 | ❌ `_calculate_gnn_score()` 返回 0.0 | **致命** |

### 2.2 問題根源

```python
# src/inference/pipeline.py:562-579
def _calculate_gnn_score(self, ...):
    """
    Calculate GNN-based score for a disease candidate.
    This is a placeholder for future GNN integration.
    Currently returns 0.0 if model is not available.
    """
    # TODO: Implement actual GNN scoring when model is integrated
    return 0.0  # ← GNN 完全被繞過!
```

**後果**:
1. 訓練模組和推理模組完全脫鉤
2. 訓練好的 GNN 模型無法用於實際診斷
3. PathReasoner (符號化 BFS 路徑搜索) 變成唯一的推理方法
4. 喪失 GNN 的**向量空間泛化能力**和**隱藏關聯發現能力**

### 2.3 當前推理流程 (偏離後)

```
患者表型 (HPO terms)
       ↓
┌─────────────────────────────────┐
│  PathReasoner (BFS)             │  ← 符號化方法，無 GNN
│  - 多跳路徑搜索                  │
│  - 邊權重評分                    │
└─────────────────────────────────┘
       ↓
┌─────────────────────────────────┐
│  候選疾病排序                    │
│  - path_weight * path_score     │
│  - gnn_weight * 0.0 ← 無效!     │
└─────────────────────────────────┘
       ↓
診斷結果
```

### 2.4 藍圖 vs 實際對比

| 藍圖功能 | 實現狀態 | 備註 |
|---------|---------|------|
| 分層本體感知 GNN | ❌ 未實現 | shepherd_gnn.py 是標準 HeteroGNN |
| 知識超圖 | ❌ 空檔案 | src/kg/hypergraph.py = 0 lines |
| DR.KNOWS 路徑推理 | ⚠️ 簡化版 | PathReasoner 是 BFS，非 DR.KNOWS |
| Neural ODE | ❌ 未實現 | 無時序建模 |
| 本體約束解碼 | ✅ 完成 | src/ontology/constraints.py |
| 向量檢索 | ✅ 完成 | cuVS + Voyager |
| GraphRAG | ❌ 未實現 | - |
| LLM 整合 | ⚠️ 框架存在 | src/llm/* 有基礎結構 |

---

## 三、模組完成度總覽

### 3.1 完成模組 (✅ 100%)

| 模組 | 檔案 | 行數 | 說明 |
|------|-----|------|------|
| **GNN 模型** | src/models/gnn/shepherd_gnn.py | 500 | HeteroGNN + Ortholog Gating |
| **GNN 層** | src/models/gnn/layers.py | ~400 | GAT/HGT/SAGE convolutions |
| **訓練器** | src/training/trainer.py | ~850 | PyTorch 2.9 兼容 |
| **損失函數** | src/training/loss_functions.py | ~600 | 多任務損失 |
| **回調** | src/training/callbacks.py | ~550 | EarlyStopping, Checkpoint |
| **路徑推理** | src/reasoning/path_reasoning.py | 548 | BFS + 評分 |
| **解釋生成** | src/reasoning/explanation_generator.py | 500 | 人類可讀解釋 |
| **本體載入** | src/ontology/loader.py | 373 | Pronto-based |
| **本體層次** | src/ontology/hierarchy.py | 640 | 相似度計算 |
| **本體約束** | src/ontology/constraints.py | 309 | 一致性檢查 |
| **知識圖譜** | src/kg/builder.py | 711 | 多源整合 |
| **圖結構** | src/kg/graph.py | 571 | 異質圖 |
| **資料載入** | src/kg/data_loader.py | 889 | 子圖採樣 |
| **向量索引** | src/retrieval/vector_index.py | 305 | cuVS/Voyager |
| **注意力後端** | src/models/attention/adaptive_backend.py | ~200 | Flash/SDPA |

### 3.2 部分完成 (⚠️ 70-90%)

| 模組 | 檔案 | 問題 |
|------|-----|------|
| **推理管線** | src/inference/pipeline.py | `_calculate_gnn_score()` 返回 0.0 |
| **API 主程式** | src/api/main.py | 需要實際 KG/模型初始化 |
| **診斷路由** | src/api/routes/diagnose.py | Session 管理未實現 |

### 3.3 未完成/空檔 (❌ 0-30%)

| 模組 | 檔案 | TODO 數量 | 優先級 |
|------|-----|----------|--------|
| **同源數據** | src/data_sources/ortholog.py | 12 | P1 |
| **PubMed** | src/data_sources/pubmed.py | 9 | P1 |
| **超圖** | src/kg/hypergraph.py | - (空檔) | Low |
| **實體連結** | src/kg/entity_linker.py | - (空檔) | Low |
| **約束檢查** | src/reasoning/constraint_checker.py | - (空檔) | Low |

---

## 四、本次 Session 修復紀錄

### 4.1 PyTorch 2.9 + CUDA 13.0 兼容性修復

| Commit | 修復內容 |
|--------|---------|
| `ece773a` | AMP API: `torch.cuda.amp` → `torch.amp` |
| `db194b2` | loss_functions.py 嵌套字典設備檢測 |
| `0694573` | loss tensors 啟用 `requires_grad=True` |
| `9818eb8` | loss_functions.py 字典查找邏輯修正 (batch → model_outputs) |
| `2790e08` | Scheduler 除以零保護 (warmup_steps >= total_steps) |

### 4.2 訓練驗證結果
```
python scripts/train_model.py --epochs 2 --batch-size 4

Training Complete!
  val_mrr: 0.0556
  val_hits@1: 0.0100
  val_hits@10: 0.1700
  val_loss: 7.0751
```
(2 epochs 測試數據，指標正常)

---

## 五、下一步優先任務 (P0)

### 5.1 實現 `_calculate_gnn_score()` (CRITICAL)

**位置**: `src/inference/pipeline.py:562-579`

**需要實現**:
```python
def _calculate_gnn_score(
    self,
    source_ids: List[NodeID],
    disease_id: NodeID,
    patient_input: PatientPhenotypes,
) -> float:
    """
    使用訓練好的 GNN 計算患者-疾病相似度
    """
    if self.model is None:
        return 0.0

    # 1. 將患者表型轉換為節點嵌入
    phenotype_embeddings = self._get_phenotype_embeddings(source_ids)

    # 2. 運行 GNN 前向傳播 (已訓練的模型)
    with torch.no_grad():
        node_embeddings = self.model(x_dict, edge_index_dict)

    # 3. 獲取疾病嵌入
    disease_embedding = node_embeddings['disease'][disease_idx]

    # 4. 計算患者-疾病相似度
    patient_embedding = self._aggregate_phenotypes(phenotype_embeddings)
    similarity = F.cosine_similarity(patient_embedding, disease_embedding, dim=-1)

    return similarity.item()
```

### 5.2 模型載入與初始化

需要在 `DiagnosisPipeline.__init__()` 中加入:
1. 載入訓練好的 checkpoint
2. 初始化 ShepherdGNN 模型
3. 將模型設為 eval 模式

### 5.3 端到端測試

1. 訓練模型並保存 checkpoint
2. 推理管線載入 checkpoint
3. 驗證 GNN 分數非零
4. 比較 PathReasoner-only vs GNN+PathReasoner 結果

---

## 六、長期路線圖

### Phase 1: GNN 推理整合 (本週)
- [ ] 實現 `_calculate_gnn_score()`
- [ ] 模型 checkpoint 載入
- [ ] 推理效能基準測試

### Phase 2: 訓練優化 (下週)
- [ ] 完整數據集訓練
- [ ] 超參數調優
- [ ] Hits@10 目標: ≥70%

### Phase 3: 藍圖功能補齊 (下月)
- [ ] 同源數據載入 (ortholog.py)
- [ ] PubMed 文獻整合 (pubmed.py)
- [ ] API 完整初始化
- [ ] WebUI 完善

### Phase 4: 進階功能 (遠期)
- [ ] 分層本體感知 GNN
- [ ] 知識超圖
- [ ] Neural ODE 時序建模
- [ ] GraphRAG + LLM 整合

---

## 七、技術環境

### 7.1 當前測試環境
```yaml
OS: Windows 11
GPU: NVIDIA GeForce RTX 5070 Ti (17.1 GB)
CUDA: 13.0
Python: 3.12
PyTorch: 2.9
PyTorch Geometric: 2.7+
```

### 7.2 關鍵依賴
```
torch>=2.9.0
torch-geometric>=2.7.0
voyager>=2.0  # 向量檢索
pronto>=2.5   # 本體解析
```

---

## 八、關鍵文件參考

| 文件 | 用途 |
|-----|------|
| `medical-kg-blueprint.md` | 工程藍圖 v2.0 |
| `docs/ENGINEERING_PROGRESS_REPORT_2026-02.md` | 工程進度報告 |
| `configs/default_config.yaml` | 預設配置 |
| `scripts/train_model.py` | 訓練腳本 |
| `src/inference/pipeline.py` | 推理管線 (重點修復對象) |
| `src/models/gnn/shepherd_gnn.py` | GNN 模型 (完整) |

---

## 九、總結

**專案整體狀態**: 70% 完成，核心訓練和基礎設施就緒

**致命問題**: GNN 訓練完成但未接入推理管線，導致:
1. 訓練成果無法使用
2. 推理僅依賴符號化路徑搜索
3. 喪失 GNN 的泛化和關聯發現能力

**立即行動**: 實現 `_calculate_gnn_score()` 並完成端到端測試

---

*本文件由 Claude 4.6 Opus 自動生成，用於跨 Session 專案交接*
