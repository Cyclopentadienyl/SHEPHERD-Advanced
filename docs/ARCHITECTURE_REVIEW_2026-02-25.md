# SHEPHERD-Advanced 專案架構系統性審查報告

**審查日期**: 2026-02-25
**審查範圍**: 全 Repo 程式碼、commit 歷史、工程藍圖、技術文檔
**審查起因**: 醫院開會後提出的架構合理性質疑

---

## Context（背景）

此次審查的起因：開發者在與醫院開會後，院方提出了兩個根本性問題：
1. **此框架是否過度複雜化？** RAG Graph 或基於 ontology map 的傳統 path reasoning 是否就能達成需求？使用 GNN 的意義在哪？
2. **相比原版 SHEPHERD 論文，到底改善了什麼？** 是否在做無用功？

開發者自己也感到隨著開發推進，項目方向越來越模糊，模塊間功能和 IO 參數無法協調，出現反覆調整和重複設計的問題。

以下是基於 Repo 中的程式碼、146 條 commit 記錄、工程藍圖（`medical-kg-blueprint.md`）、技術審核報告（`tech-audit-report.md`）、工程進度報告（`docs/ENGINEERING_PROGRESS_REPORT_2026-02.md`）、模組掃描報告（`docs/MODULE_SCAN_REPORT_2026-01-26.md`）、以及所有核心 Python 模組的系統性分析。

---

## 一、專案現狀總覽

### 1.1 代碼規模
- **Python 檔案**: 86 個（49 個已實現，37 個空/佔位符）
- **代碼行數**: ~18,307 行
- **測試**: 173 個（163 通過，8 跳過，0 失敗）
- **Commit 數**: 146 個（Claude 63%, 人類 33%, RACSIS 4%）

### 1.2 已完成的核心模組
| 模組 | 狀態 | 功能 |
|------|------|------|
| `src/core/` | ✅ 完整 | 類型定義、Protocol 接口 |
| `src/kg/` | ✅ 完整 | 知識圖譜建構、異質圖存儲 |
| `src/ontology/` | ✅ 完整 | HPO/MONDO 本體載入、層次結構、約束 |
| `src/models/gnn/` | ✅ 完整 | ShepherdGNN 異質圖神經網路 |
| `src/reasoning/` | ✅ 完整 | PathReasoner (BFS 路徑推理) |
| `src/inference/` | ✅ 完整 | DiagnosisPipeline (端到端推理) |
| `src/training/` | ✅ 完整 | Trainer、Loss、Callbacks |
| `src/retrieval/` | ✅ 完整 | Voyager/cuVS 向量索引 |
| `src/api/` | ✅ 完整 | FastAPI REST API |
| Gradio 訓練儀表板 | ✅ 完整 | 訓練控制台 + 資源監控 |

### 1.3 未完成的模組（全空佔位符）
| 模組 | 檔案數 | 說明 |
|------|--------|------|
| `src/llm/` | 5 | LLM 整合（GPT-4/Claude 解釋生成）|
| `src/nlp/` | 5 | 症狀 NLP 提取（ClinicalBERT）|
| `src/medical_standards/` | 5 | FHIR/ICD/SNOMED 映射 |
| `src/kg/storage/` | 2 | Neo4j 圖資料庫持久化 |
| `src/kg/hypergraph.py` | 1 | 知識超圖（高階關係）|

---

## 二、回答院方問題一：GNN vs 傳統方法

### 2.1 系統目前實際在做什麼？

根據 `src/inference/pipeline.py` 的實際代碼，推理流程是：

```
患者 HPO 症狀 → InputValidator 驗證
    → PathReasoner (BFS 搜尋 Phenotype→Gene→Disease 路徑)
    → [可選] ShepherdGNN (圖神經網路計算 cosine similarity)
    → [可選] VectorIndex (ANN 向量檢索候選疾病)
    → ExplanationGenerator (生成解釋)
    → 排序輸出 Top-K 候選疾病
```

**關鍵發現**：系統實際上已經是一個「PathReasoner 為核心 + GNN 為可選增強」的混合架構。PathReasoner 能獨立工作（做 BFS 路徑搜索 + 評分），GNN 是在 PathReasoner 之上的一層增強。

### 2.2 院方的問題本質：為什麼不只用 PathReasoner？

**PathReasoner（已實現的）本質上就是基於 ontology map 的 path reasoning 算法。** 它做的事情是：
- 在知識圖譜上做 BFS，找 `Phenotype → Gene → Disease` 的路徑
- 根據邊權重和路徑長度計算分數
- 聚合多條路徑的分數得到最終排名

這其實就是院方說的「基於 ontology map 的傳統 path reasoning」。**已經實現了它，而且它是目前唯一實際在用的推理引擎。**

### 2.3 GNN 的真正價值

GNN 相比純 path reasoning 的核心價值在於：

| 面向 | Path Reasoning (BFS) | GNN |
|------|---------------------|-----|
| **搜索空間** | 只看顯式路徑（知識圖譜中有邊的） | 學習隱式關係（無直接邊也能推斷） |
| **泛化能力** | 無法泛化到未見過的組合 | 能對新的症狀組合做推斷 |
| **罕見疾病** | 如果 KG 中缺少邊就找不到 | 能從相似疾病的 pattern 遷移學習 |
| **多症狀整合** | 路徑獨立計算再聚合（naive） | 一次性編碼整個症狀集合的結構特徵 |
| **訓練數據要求** | 不需要 | 需要有標注的患者-疾病對 |

**GNN 的意義：** 當知識圖譜不完整（罕見疾病的數據缺口很常見）時，BFS 路徑推理會完全失效。GNN 能通過圖結構學習，在沒有直接路徑的情況下仍然推斷出可能的疾病關聯。這是原版 SHEPHERD 論文的核心貢獻。

### 2.4 但 GNN 的使用前提

**GNN 只有在具備以下條件時才有意義：**
1. **有訓練數據**：需要真實的「患者症狀→確診疾病」的標注數據集來訓練
2. **知識圖譜足夠大**：PrimeKG 有 ~400 萬條邊，GNN 在大圖上的優勢更明顯
3. **評估基準明確**：需要測試集來量化 GNN vs BFS 的差異

**如果目前沒有訓練數據和測試基準，GNN 確實是過度設計。** 在沒有數據的情況下，PathReasoner + Ontology 約束已經是最務實的方案。

### 2.5 關於 RAG Graph 的對比

院方提到的 RAG Graph（GraphRAG）本質上是：
- 用 LLM 對知識圖譜做問答
- 檢索相關子圖 → 送入 LLM 生成答案

**GraphRAG 的問題**：
- LLM 的幻覺問題在醫療場景中不可接受（這是藍圖中特別強調要把幻覺率降到 <5% 的原因）
- 無法提供可追溯的推理路徑（PathReasoner 可以）
- 推理成本高（每次都要調 LLM API）
- 不可控（LLM 輸出不可預測）

**結論**：在罕見疾病診斷這個具體場景中，PathReasoner + GNN 的組合比 GraphRAG 更合適。但如果只是想要一個「症狀查疾病」的工具，PathReasoner 單獨就夠了。

---

## 三、回答院方問題二：相比原版 SHEPHERD 改善了什麼？

### 3.1 原版 SHEPHERD 論文做了什麼

原版 SHEPHERD (Nat Med, 2023, mims-harvard) 的架構是：
1. **KG**: 使用 PrimeKG（~4M 邊的異質知識圖譜）
2. **GNN**: 在 PrimeKG 上做 node embedding（用 GAT/RGCN）
3. **Few-shot**: 給定一個患者的症狀集合，通過 GNN embeddings 做 few-shot matching
4. **任務**: 基因候選排序、疾病候選排序、因果基因預測

原版的核心流程是：
```
PrimeKG → GNN pre-training → 患者症狀 embedding → cosine similarity → 排序
```

### 3.2 相比原版的實際改動

| 改動 | 原版 SHEPHERD | 本系統 | 改善/質疑 |
|------|-------------|---------|-----------|
| **推理方式** | 純 GNN embedding matching | PathReasoner + GNN 雙軌 | ✅ 增加了可解釋性 |
| **GNN 架構** | GAT/RGCN | GAT + Ortholog Gate + 位置編碼 | ⚠️ 更複雜但未有基準驗證 |
| **本體整合** | HPO 作為節點特徵 | HPO/MONDO 本體載入 + 約束 | ✅ 更深度的本體利用 |
| **向量檢索** | 無 | Voyager/cuVS ANN 索引 | ✅ 加速大規模推理 |
| **跨物種** | 支持 mouse ortholog | 支持 + OrthologGate | ⚠️ Gate 機制是新的 |
| **API 服務** | 無（純研究代碼） | FastAPI REST + Gradio UI | ✅ 工程化 |
| **跨平台** | Linux only | Windows + ARM (DGX Spark) | ✅ 部署靈活性 |
| **訓練框架** | 簡單 training loop | Trainer + Callbacks + 監控 | ✅ 工程化 |
| **知識超圖** | 無 | 藍圖規劃中（未實現）| ❌ 尚未實現 |
| **Neural ODE** | 無 | 藍圖規劃中（未實現）| ❌ 尚未實現 |
| **DR.KNOWS 路徑推理** | 無 | PathReasoner（簡化實現）| ⚠️ 部分實現 |
| **LLM 整合** | 無 | 空佔位符（未實現）| ❌ 尚未實現 |
| **NLP 症狀提取** | 無 | 空佔位符（未實現）| ❌ 尚未實現 |
| **FHIR 醫療標準** | 無 | 空佔位符（未實現）| ❌ 尚未實現 |

### 3.3 坦率的評估

**已實現且有價值的改善：**
1. ✅ PathReasoner 帶來的可解釋性（原版沒有明確的推理路徑輸出）
2. ✅ 工程化（API、UI、跨平台部署、訓練監控）
3. ✅ 向量檢索加速
4. ✅ 本體約束推理框架

**藍圖中規劃但完全未實現的「改善」：**
1. ❌ 知識超圖（`src/kg/hypergraph.py` 是空的）
2. ❌ Neural ODE 時序建模（不存在任何代碼）
3. ❌ LLM 整合（5 個空文件）
4. ❌ NLP 症狀提取（5 個空文件）
5. ❌ FHIR 醫療標準整合（5 個空文件）
6. ❌ 藥物建議引擎（僅存在於 TODO 文檔）
7. ❌ 文獻檢索引擎（僅存在於 TODO 文檔）

**存在疑問的改動：**
1. ⚠️ ShepherdGNN 加了 OrthologGate、位置編碼等，但沒有基準測試證明這些加了比原版好
2. ⚠️ 藍圖中聲稱 Hits@10 能從 60% 提升到 82%，但沒有任何實驗數據支撐
3. ⚠️ 大量時間花在部署腳本和 UI，而不是核心算法驗證

---

## 四、核心問題診斷

### 4.1 藍圖與實現之間的鴻溝

藍圖 (`medical-kg-blueprint.md`) 描繪了一個極其宏大的系統（5+1 層架構、超圖、Neural ODE、GraphGPS、DR.KNOWS、LLM 整合...），但實際實現的是一個相對簡單的系統：

```
實際實現 = PathReasoner(BFS) + ShepherdGNN(GAT+Ortholog) + VectorIndex + REST API
```

**問題不是實現得不好，而是藍圖畫得太大。** 這導致：
- 37 個空佔位符文件散落在代碼中，製造了「系統很複雜」的錯覺
- 開發者自己搞不清楚哪些是核心、哪些是未來計劃
- 院方看到藍圖和文檔會覺得過度設計

### 4.2 缺少基準驗證

最根本的問題是：**沒有任何實驗證明本系統比原版 SHEPHERD 更好。**

- 沒有在標準數據集上跑過 evaluation
- 藍圖中的數字（Hits@10: 82%、幻覺率 <5%）完全是推測
- 不知道 GNN 加了 OrthologGate 之後效果是提升了還是降低了
- 不知道 PathReasoner 的 BFS 推理質量如何

### 4.3 工程投入分配失衡

根據 commit 分析：
- **部署腳本**: 18 commits（最多修改的文件類型）
- **UI 訓練控制台**: 16 commits
- **依賴管理**: 13 commits
- **核心推理算法**: 相對較少

大量時間花在了讓系統「能跑起來」，而不是驗證系統「跑得對」。

---

## 五、建議：下一步該做什麼

### 方案 A：務實精簡路線（推薦）

**目標：先證明核心價值，再擴展功能**

1. **清理專案**
   - 刪除所有空佔位符文件（37 個）
   - 精簡藍圖文檔，只保留已實現的部分
   - 明確標注什麼是 Phase 1（已完成）、什麼是 Future Work

2. **建立評估基準**（最高優先級）
   - 用原版 SHEPHERD 的測試數據集跑一次 PathReasoner
   - 跑一次 GNN（如果有訓練數據的話）
   - 得到 Hits@1/5/10/20 和 MRR 的實際數字
   - 與原版 SHEPHERD 論文的數字做對比

3. **向院方展示的核心價值**
   - PathReasoner 的可解釋推理路徑（這是原版沒有的）
   - 完整的工程化部署方案（API + UI）
   - 本體約束推理（防止不合理預測）

4. **只有在基準結果好的情況下**才繼續以下開發：
   - NLP 症狀提取
   - FHIR 整合
   - LLM 解釋增強

### 方案 B：如果決定不用 GNN

如果院方或團隊決定 GNN 過於複雜，可以只保留：
- `src/kg/` (知識圖譜)
- `src/ontology/` (本體)
- `src/reasoning/` (PathReasoner)
- `src/inference/` (pipeline，去掉 GNN 部分)
- `src/api/` (API)

這實際上就是「基於 ontology map 的 path reasoning」方案，技術上可行，且代碼已經支持在無 GNN 的情況下運行（`DiagnosisPipeline` 在 `_gnn_ready=False` 時會 fallback 到 path reasoning scoring）。

---

## 六、關鍵文件參考

| 文件 | 用途 |
|------|------|
| `src/inference/pipeline.py` | 核心推理管線（GNN + PathReasoner 雙軌） |
| `src/reasoning/path_reasoning.py` | BFS 路徑推理引擎 |
| `src/models/gnn/shepherd_gnn.py` | 異質圖神經網路模型 |
| `src/kg/graph.py` | 知識圖譜數據結構 |
| `src/ontology/hierarchy.py` | 本體層次結構 |
| `src/training/trainer.py` | 訓練器 |
| `medical-kg-blueprint.md` | 工程藍圖（vs 實際實現差距大） |
| `tech-audit-report.md` | 技術審核報告 |
| `docs/ENGINEERING_PROGRESS_REPORT_2026-02.md` | 工程進度報告 |
| `docs/MODULE_SCAN_REPORT_2026-01-26.md` | 模組掃描報告 |

---

## 七、後續行動計劃

### Step 1: 架構分析文檔（本次 PR）
- 產出一份完整的架構分析文檔（即本報告）
- 提交到 `claude/review-project-architecture-fvnuF` 分支

### Step 2: 建立評估基準（後續工作）
- 下載 SHEPHERD 原版測試數據（mims-harvard/SHEPHERD GitHub repo）
- 在本系統上跑 evaluation
- 對比結果

### Step 3: 根據結果決策
- 如果 GNN 比 BFS 好 → 繼續 GNN 路線
- 如果差不多 → 簡化為 BFS-only
- 如果更差 → 調查原因

---

**報告編製**: Claude (Architecture Reviewer)
**審查日期**: 2026-02-25
