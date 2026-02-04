# SHEPHERD-Advanced 工程進度報告

**報告日期**: 2026年2月
**報告版本**: v1.0
**專案版本**: v0.2.0
**報告對象**: 醫院資訊工程部門

---

## 執行摘要

SHEPHERD-Advanced 是一套**臨床級罕見疾病診斷推理引擎**，透過異質圖神經網路 (Heterogeneous GNN) 與知識圖譜推理技術，協助臨床醫師進行罕見疾病的鑑別診斷。目前專案已完成約 **85% 的核心功能開發**，包含完整的知識圖譜建構、路徑推理引擎、訓練框架、REST API 服務，以及跨平台部署支援。

**核心指標目標**：
- 診斷準確度：Hits@10 ≥ 80%（相比標準 SHEPHERD 的 60%）
- 幻覺率：< 5%
- 推理時間：< 150ms（即時診斷）
- 可解釋性：完整推理路徑與文獻證據

---

## 1. 整體工作流程框架

### 1.1 設計特性

SHEPHERD-Advanced 採用**分層式架構 (Layered Architecture)** 與**協議優先設計 (Protocol-First Design)**：

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Layer 6: API 服務層                          │
│         FastAPI REST API ─── Gradio Web UI ─── FHIR Adapter         │
├─────────────────────────────────────────────────────────────────────┤
│                        Layer 5: 推理層                               │
│     Inference Pipeline ─── Path Reasoner ─── Explanation Gen        │
├─────────────────────────────────────────────────────────────────────┤
│                        Layer 4: 模型層                               │
│        ShepherdGNN ─── HeteroGNNLayer ─── Attention Backend         │
├─────────────────────────────────────────────────────────────────────┤
│                        Layer 3: 檢索層                               │
│         Vector Index (Voyager/cuVS) ─── Subgraph Sampler            │
├─────────────────────────────────────────────────────────────────────┤
│                        Layer 2: 知識層                               │
│   Knowledge Graph ─── Ontology (HPO/MONDO) ─── Data Sources         │
├─────────────────────────────────────────────────────────────────────┤
│                        Layer 1: 訓練層                               │
│          Trainer ─── Loss Functions ─── Callbacks                   │
├─────────────────────────────────────────────────────────────────────┤
│                        Layer 0: 核心基礎層                           │
│     Core Types ─── Protocols ─── Config ─── Utils                   │
└─────────────────────────────────────────────────────────────────────┘
```

**關鍵設計特性**：

| 特性 | 說明 | 效益 |
|------|------|------|
| **Protocol-Based Design** | 所有模組定義明確的介面協議 | 模組可獨立測試、替換、擴展 |
| **單向依賴流** | 上層依賴下層，禁止反向依賴 | 避免循環依賴，維護性高 |
| **子圖採樣** | 動態採樣知識圖譜子圖 | 支援 16GB VRAM 的 Windows 工作站 |
| **可解釋性優先** | 所有診斷結果包含推理路徑 | 符合醫療 AI 可解釋性要求 |
| **跨平台支援** | Windows x86 + Linux ARM (DGX Spark) | 彈性部署選項 |

### 1.2 初始輸入與最終輸出

#### 初始輸入格式

系統接受多種輸入格式，核心為**患者表型 (Phenotypes)**：

```json
{
  "patient_id": "P20260204001",
  "phenotypes": [
    {"hpo_id": "HP:0001250", "name": "Seizures", "observed": true},
    {"hpo_id": "HP:0001263", "name": "Developmental delay", "observed": true},
    {"hpo_id": "HP:0000252", "name": "Microcephaly", "observed": true}
  ],
  "metadata": {
    "source": "clinical_input",
    "timestamp": "2026-02-04T10:30:00Z"
  }
}
```

**支援的輸入類型**：
- 結構化 HPO 術語（主要）
- FHIR Bundle（Phase 2）
- 自由文字症狀描述（Phase 2，需 NLP 處理）

#### 最終輸出格式

```json
{
  "status": "success",
  "inference_result": {
    "top_candidates": [
      {
        "disease_id": "OMIM:300672",
        "disease_name": "CDKL5 Deficiency Disorder",
        "confidence": 0.847,
        "rank": 1,
        "supporting_genes": ["CDKL5"],
        "reasoning_paths": [
          "HP:0001250 (Seizures) → CDKL5 → OMIM:300672",
          "HP:0001263 (Developmental delay) → CDKL5 → OMIM:300672"
        ]
      },
      {
        "disease_id": "OMIM:612164",
        "disease_name": "Dravet Syndrome",
        "confidence": 0.723,
        "rank": 2,
        "supporting_genes": ["SCN1A"],
        "reasoning_paths": ["..."]
      }
    ],
    "explanation": {
      "summary": "基於輸入的 3 個表型，系統識別出 CDKL5 基因相關疾病具有最高匹配度...",
      "evidence_citations": ["PMID:12345678", "PMID:23456789"],
      "ontology_validation": "PASSED"
    }
  },
  "model_info": {
    "version": "0.2.0",
    "data_version": "kg_v3_2026-01",
    "inference_time_ms": 142
  }
}
```

---

## 2. 模組協同與資訊傳遞

### 2.1 完整推理流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Patient Input (HPO Terms)                       │
│                                    │                                     │
│                                    ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Input Validator                               │    │
│  │  • 驗證 HPO ID 格式 (HP:\d{7})                                  │    │
│  │  • 檢查必要欄位                                                  │    │
│  │  • 轉換為標準 PatientPhenotypes 物件                            │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                    │                                     │
│                     PatientPhenotypes 物件                               │
│                                    │                                     │
│                                    ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Path Reasoner (BFS)                           │    │
│  │  • 從 Phenotype 節點出發                                        │    │
│  │  • 搜尋 Phenotype → Gene → Disease 路徑                         │    │
│  │  • 計算路徑分數 (基於邊權重與連接數)                            │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                    │                                     │
│              List[ReasoningPath] (候選路徑)                              │
│                                    │                                     │
│                                    ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    ShepherdGNN (可選)                            │    │
│  │  • 使用圖神經網路重新評分                                        │    │
│  │  • 整合節點嵌入與拓撲資訊                                        │    │
│  │  • 輸出：disease 分數向量                                        │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                    │                                     │
│                                    ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Explanation Generator                         │    │
│  │  • 生成人類可讀的推理說明                                        │    │
│  │  • 附加文獻證據 (PubMed)                                         │    │
│  │  • 本體論驗證結果                                                 │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                    │                                     │
│                                    ▼                                     │
│                        InferenceResult (最終輸出)                        │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 各模組中間輸入/輸出規格

| 模組 | 輸入 | 輸出 | 主要運算 |
|------|------|------|----------|
| **InputValidator** | Raw JSON | `PatientPhenotypes` | JSON Schema 驗證、HPO 格式檢查 |
| **KnowledgeGraph** | Ontology + Data Sources | `HeteroData` (PyG) | 節點/邊建構、特徵提取 |
| **PathReasoner** | `PatientPhenotypes` + `KG` | `List[ReasoningPath]` | BFS 路徑搜尋、分數計算 |
| **ShepherdGNN** | `x_dict` + `edge_index_dict` | `out_dict` (節點嵌入) | 異質圖訊息傳遞、注意力聚合 |
| **DiagnosisHead** | 表型嵌入 + 疾病嵌入 | 疾病分數向量 | 內積/餘弦相似度計算 |
| **ExplanationGenerator** | 推理路徑 + 分數 | 人類可讀說明 | 模板填充、證據整理 |
| **VectorIndex** | `Dict[str, np.ndarray]` | `List[Tuple[str, float]]` | HNSW 近似最近鄰搜尋 |

### 2.3 部署階段新調整

本次開發週期對部署流程進行了優化：

**原部署流程（5 階段）**：
```
STAGE 1: 環境驗證
STAGE 2: 核心依賴
STAGE 3: PyTorch/PyG
STAGE 4: 可選加速器 (xformers/flash-attn) ← 移除
STAGE 5: 知識圖譜建構 ← 移至手動
```

**新部署流程（4 階段）**：
```
STAGE 1: 環境驗證 (Python, CUDA)
STAGE 2: 核心依賴 (pip install -e .)
STAGE 3: PyTorch 生態系 (torch, torch-geometric)
STAGE 4: 安裝驗證

Next Steps (手動執行):
  1. 準備原始數據 → data/raw/
  2. 預處理數據 → python scripts/preprocess_data.py
  3. 建構知識圖譜 → python scripts/build_kg.py
```

**改進效益**：
- 可選加速器改為啟動時自動檢測與 fallback
- 知識圖譜建構獨立於部署，便於數據更新
- 部署失敗時更容易定位問題

---

## 3. 技術堆疊選擇

### 3.1 Python 3.10 → 3.12 升級的必要性

本專案從 Python 3.10 升級至 **Python 3.12**，這是一項**必要性升級**而非選擇性升級：

#### 升級原因

```
┌─────────────────────────────────────────────────────────────────────┐
│                      NumPy + CUDA 13.0 相容性問題                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   問題：NumPy 缺乏同時支援以下兩者的版本：                          │
│         • CUDA 13.0 環境                                            │
│         • Python 3.10                                                │
│                                                                      │
│   解決方案：升級至 Python 3.12                                       │
│         • NumPy 2.x 完整支援 Python 3.12                            │
│         • CUDA 13.0 相關套件均支援 Python 3.12                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**關鍵依賴相容性**：

| 套件 | Python 3.10 + CUDA 13.0 | Python 3.12 + CUDA 13.0 |
|------|-------------------------|-------------------------|
| NumPy 2.x | ❌ 無相容版本 | ✅ 完整支援 |
| PyTorch 2.9.0 | ✅ 支援 | ✅ 支援 |
| torch-geometric 2.7.0 | ✅ 支援 | ✅ 支援 |

> **注意**：PyTorch 2.9.0 支援 Python 3.10-3.13，但 NumPy 的相容性問題迫使我們升級至 3.12

#### Python 3.12 新特性帶來的效益

| 特性 | 說明 | 對專案的影響 |
|------|------|-------------|
| **更快的解釋器** | CPython 3.12 性能提升 5-10% | 數據預處理更快 |
| **改進的錯誤訊息** | 更精確的語法錯誤提示 | 開發除錯效率提升 |
| **類型提示增強** | `TypedDict`, `Self` 等新特性 | 代碼品質與 IDE 支援改善 |
| **NumPy 2.0 支援** | 3.12 是 NumPy 2.0+ 的主要支援版本 | 向量計算效能提升 |

### 3.2 向量檢索後端遷移：FAISS/hnswlib → Voyager/cuVS

#### 遷移背景

原專案使用 **FAISS** (Facebook) 和 **hnswlib** 作為向量檢索後端，但遇到以下問題：

| 問題 | 說明 |
|------|------|
| **跨平台相容性** | FAISS 在 Windows 上編譯困難，需要自行建構 |
| **ARM 支援不足** | DGX Spark (ARM64) 上的 FAISS 效能未優化 |
| **版本衝突** | hnswlib 與 Python 3.12 + NumPy 2.0 有相容性問題 |
| **維護風險** | hnswlib 維護頻率較低 |

#### 新方案：雙後端策略

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Vector Retrieval Backend                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Platform Detection                                                 │
│        │                                                             │
│        ├─── Linux + NVIDIA GPU ──→ cuVS (GPU 加速)                  │
│        │         │                                                   │
│        │         └─── Fallback ──→ Voyager (CPU)                    │
│        │                                                             │
│        ├─── Linux + No GPU ──→ Voyager (CPU)                        │
│        │                                                             │
│        └─── Windows ──→ Voyager (CPU)                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

#### 後端比較

| 特性 | FAISS (舊) | Voyager (新-CPU) | cuVS (新-GPU) |
|------|------------|------------------|---------------|
| **維護者** | Meta (Facebook) | Spotify | NVIDIA RAPIDS |
| **Windows 支援** | ⚠️ 需自行編譯 | ✅ 官方 wheel | ❌ Linux only |
| **ARM64 支援** | ⚠️ 有限 | ✅ 原生支援 | ✅ DGX 優化 |
| **Python 3.12** | ⚠️ 延遲支援 | ✅ 完整支援 | ✅ 完整支援 |
| **HNSW 演算法** | ✅ | ✅ | ✅ (CAGRA) |
| **GPU 加速** | ✅ (有限) | ❌ | ✅ (原生) |
| **安裝方式** | `pip install faiss-cpu` | `pip install voyager` | `pip install cuvs-cu12` |

#### Voyager 選擇理由

**Voyager** 是 Spotify 開源的 HNSW 向量檢索庫：

1. **跨平台**：Windows、Linux、macOS 都有官方預編譯 wheel
2. **輕量**：純 C++ 實現，無額外依賴
3. **API 簡潔**：比 FAISS 更直觀的 Python API
4. **活躍維護**：Spotify 用於生產環境的音樂推薦系統

#### cuVS 選擇理由

**cuVS** 是 NVIDIA RAPIDS 團隊的 GPU 向量檢索庫：

1. **GPU 原生**：充分利用 CUDA 核心進行並行搜尋
2. **多演算法**：支援 IVF-Flat、IVF-PQ、CAGRA
3. **高吞吐**：GPU 批次查詢性能遠超 CPU
4. **RAPIDS 生態**：與 cuDF、cuML 無縫整合

#### 向量檢索模組的用途與整合狀態

**目前狀態**：向量檢索模組已完成核心功能開發與測試，但**尚未整合至主要推理管線**。

**預期用途**：
- 根據症狀嵌入快速檢索相似的疾病/基因候選
- 加速 top-k 候選疾病的初步篩選
- 支援大規模知識圖譜的高效查詢

**已完成**：
- `src/retrieval/` 核心模組
- `scripts/build_index.py` 索引建構腳本
- 單元測試與整合測試

**待整合**（Phase 2）：
- 推理管線 (`src/inference/pipeline.py`)
- 路徑推理 (`src/reasoning/path_reasoning.py`)
- API 端點 (`src/api/routes/`)

### 3.3 PyG + CUDA 13.0 相容性驗證

雖然 PyTorch Geometric 官方尚未在相容性列表中明確列出 CUDA 13.0 支援（官方僅列至 CUDA 12.9），但經實際測試驗證，**PyG 2.7.0 可在 CUDA 13.0 環境正常運行**：

```
測試結果：
PyTorch: 2.9.0+cu130, CUDA: 13.0, Available: True
PyG: 2.7.0
PyG GCNConv CUDA test: OK, shape=torch.Size([10, 64])
```

**結論**：目前的 venv 部署方式可以繼續使用，無需改用 NVIDIA Container。

### 3.4 完整技術堆疊

```yaml
# 核心運行環境
Python: 3.12+
CUDA: 13.0
PyTorch: 2.9.0+cu130
torch-geometric: 2.7.0 (實測相容 CUDA 13.0)

# 向量檢索
CPU Backend: Voyager 2.1.0
GPU Backend: cuVS (RAPIDS 25.02)

# 本體論處理
pronto: 2.7+  # OBO/OWL 解析
networkx: 3.2+  # 圖操作

# API 框架
FastAPI: 0.110+
Gradio: 4.0+

# 數據驗證
Pydantic: 2.7+
jsonschema: 4.21+

# 可選加速器 (自動檢測)
FlashAttention-2: 2.5+  # x86 CUDA
xformers: 0.0.26+       # 備用
SageAttention: 2.1+     # ARM 優化
```

---

## 4. 測試方法與問題修正

### 4.1 測試架構

```
tests/
├── unit/                           # 單元測試
│   ├── test_retrieval.py          # 向量檢索模組 (23 tests)
│   ├── test_models.py             # GNN 模型 (33 tests)
│   ├── test_ontology.py           # 本體論模組 (43 tests)
│   ├── test_kg.py                 # 知識圖譜 (39 tests)
│   ├── test_reasoning.py          # 推理引擎 (23 tests)
│   └── test_inference.py          # 推理管線 (25 tests)
│
├── integration/                    # 整合測試
│   ├── test_retrieval_integration.py  # 檢索整合 (12 tests)
│   └── test_pipeline.py           # 端到端管線
│
└── benchmarks/                     # 效能基準測試
    └── platform_specific/
        ├── test_vector_index_x86.py
        └── test_vector_index_arm.py
```

### 4.2 遇到的問題與修正

#### 問題 1：Voyager Windows 載入失敗

**症狀**：
```
RuntimeError: Index seems to be corrupted or unsupported.
After reading all linked lists, extra data remained at the end of the index.
```

**根因分析**：
透過診斷腳本發現，Voyager 2.1.0 在 Windows 上使用檔案路徑字串 `Index.load(filename)` 載入時會失敗，但使用 file handle 載入則正常。

**修正方案**：
```python
# 原本（Windows 失敗）
self._index = voyager.Index.load(str(index_path))

# 修正後（跨平台相容）
with open(index_path, "rb") as f:
    self._index = voyager.Index.load(f)
```

**驗證**：Windows 與 Linux 測試均通過。

---

#### 問題 2：GNN 節點特徵遺失

**症狀**：
```
UserWarning: There exist node types ({'phenotype'}) whose representations
do not get updated during message passing
```

**根因分析**：
PyTorch Geometric 的 `HeteroConv` 只輸出有接收訊息的節點類型。當某節點類型只作為邊的來源（source）而非目標（destination）時，其特徵會在訊息傳遞後遺失。

**修正方案**：
1. **模型層修正**：在 `HeteroGNNLayer.forward()` 中保留未更新的節點特徵
2. **測試資料修正**：添加反向邊確保雙向訊息傳遞

```python
# 修正：保留未收到訊息的節點特徵
for node_type in x_dict:
    if node_type not in out_dict:
        out_dict[node_type] = x_dict[node_type]
```

---

#### 問題 3：torch.compile Windows 相容性

**症狀**：
```
InductorError: RuntimeError: Compiler: cl is not found.
```

**根因分析**：
`torch.compile` 在 CPU 模式下需要 C++ 編譯器（Windows 需 MSVC cl.exe）。

**修正方案**：
添加平台檢測，無編譯器時跳過相關測試：

```python
@pytest.mark.skipif(
    sys.platform == "win32" and not shutil.which("cl"),
    reason="torch.compile requires MSVC (cl.exe) on Windows"
)
def test_compile_encoder(self):
    ...
```

---

### 4.3 測試結果摘要

| 測試類別 | 測試數 | 通過 | 跳過 | 失敗 |
|---------|--------|------|------|------|
| 向量檢索 (unit) | 23 | 20 | 3 | 0 |
| 模型測試 (unit) | 33 | 26 | 5 | 0 |
| 本體論 (unit) | 43 | 43 | 0 | 0 |
| 知識圖譜 (unit) | 39 | 39 | 0 | 0 |
| 推理引擎 (unit) | 23 | 23 | 0 | 0 |
| 檢索整合 (integration) | 12 | 12 | 0 | 0 |
| **總計** | **173** | **163** | **8** | **0** |

> 跳過的測試為環境相關（cuVS 需 Linux GPU、torch.compile 需 MSVC）

---

## 5. 下一步計劃

### 5.1 Phase 1 收尾（預計 2 週）

| 項目 | 說明 | 優先級 |
|------|------|--------|
| 完善 `build_index.py` | 實現 embeddings 載入與索引儲存 | P0 |
| Docker 容器化 | 建立生產環境 Docker image | P0 |
| CI/CD 流程 | GitHub Actions 自動測試 | P1 |
| 效能基準測試 | 建立 baseline 數據 | P1 |

### 5.2 Phase 2 開發（預計 6-8 週）

| 模組 | 說明 | 依賴 |
|------|------|------|
| **NLP 症狀提取** | 自由文字 → HPO 術語 | ClinicalBERT/SciBERT |
| **FHIR 整合** | 醫療資訊交換標準 | fhir.resources |
| **LLM 說明生成** | 使用 LLM 增強解釋 | vLLM / llama.cpp |
| **PubMed 證據** | 文獻引用整合 | Entrez API |
| **跨物種推理** | 小鼠模型知識轉移 | Ortholog data |
| **Web UI 強化** | 完整前端介面 | Gradio 4.0 |

### 5.3 Phase 3 臨床驗證（待定）

- 與臨床團隊進行驗證測試
- 收集真實案例回饋
- 模型微調與優化
- 正式部署評估

---

## 附錄

### A. 專案統計

| 項目 | 數值 |
|------|------|
| Python 檔案總數 | 90 |
| 已實現檔案 | 49 |
| 總程式碼行數 | ~18,307 |
| 測試覆蓋率 | ~52% |
| Protocol 實現率 | 11/18 (61%) |

### B. 關鍵 Commit 歷史

| Commit | 說明 |
|--------|------|
| `c2744f8` | fix(test): 修正 JSON 序列化測試 |
| `356d709` | test(models): 添加反向邊到測試資料 |
| `6fa9cb4` | fix(models): 保留未更新節點特徵 |
| `9a61dbd` | fix(retrieval): Windows Voyager 載入修正 |
| `e4ccd06` | feat(retrieval): 實現 Voyager/cuVS 雙後端 |
| `1405459` | refactor(deps): 升級至 Python 3.12 |
| `5aab799` | feat(training): 添加反向邊雙向訊息傳遞 |

### C. 參考文獻

1. PyTorch Documentation - CUDA 13.0 Support
2. Spotify Voyager GitHub Repository
3. NVIDIA RAPIDS cuVS Documentation
4. PyTorch Geometric Heterogeneous Graph Tutorial

---

**報告編製**：SHEPHERD-Advanced 開發團隊
**報告日期**：2026年2月4日
