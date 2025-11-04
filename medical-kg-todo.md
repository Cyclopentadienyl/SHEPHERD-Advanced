# 醫療知識圖譜診斷引擎 - TODO 清單 v2.0（升級版）

## 專案狀態總覽

**當前階段**: 🚀 啟動階段（v2.0 - 前沿技術整合版）  
**開始日期**: 2025-10-07  
**預計完成**: 2026-02 (4-5 個月)

**進度指標**:
- [ ] Phase 1: Windows MVP+ (0/58 任務完成)
- [ ] Phase 2: 進階功能 (0/32 任務完成)
- [ ] Phase 3: ARM部署與優化 (0/24 任務完成)

**重大升級項目**:
- 🆕 分層本體感知架構
- 🆕 知識超圖整合
- 🆕 DR.KNOWS路徑推理
- 🆕 本體約束解碼
- 🆕 跨平台自適應部署

---

## 優先級說明

- 🔴 **P0 - 關鍵阻塞**: 必須立即完成，影響核心功能
- 🟠 **P1 - 高優先級**: 核心創新功能，直接影響效能
- 🟡 **P2 - 中優先級**: 重要但非關鍵
- 🟢 **P3 - 低優先級**: 優化項目，可延後

**時間估算**: 🕐 (小時), 📅 (天), 📆 (週)

---

# Phase 1: Windows 環境 MVP+ 開發

**目標**: 實現完整的本體增強系統，Hits@10 ≥ 70%  
**預計時間**: 8-10 週  
**驗收標準**: 本體約束生效、路徑推理可用、兩平台兼容

---

## 1.1 環境設置與驗證 (Week 1)

### 🔴 P0 - 基礎環境
- [ ] 安裝 Python 3.12+ 🕐 0.5h
- [ ] 創建專案目錄結構（v2.0擴展版） 🕐 1.5h
- [ ] 初始化 Git 倉庫 + .gitignore 🕐 0.5h
- [ ] 建立 `README.md` 與 `ARCHITECTURE.md` 🕐 1.5h

### 🔴 P0 - GPU 環境配置與測試
- [ ] 驗證 Blackwell GPU 驅動 (R570+) 🕐 0.5h
- [ ] 安裝 CUDA 12.8 Toolkit 🕐 1h
- [ ] 測試 `nvidia-smi` 與 VRAM 檢測 🕐 0.5h
- [ ] 記錄硬體規格（VRAM: 16GB, TDP等） 🕐 0.5h

### 🔴 P0 - Python 虛擬環境
- [ ] 創建 venv: `python -m venv .venv` 🕐 0.5h
- [ ] 安裝 PyTorch 2.8.0 + CUDA 12.8 🕐 1h
  ```bash
  pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
  ```
- [ ] 測試 PyTorch GPU 可用性與 FP16 支持 🕐 0.5h
- [ ] 測試 CUDA 記憶體分配（驗證16GB限制） 🕐 0.5h

### 🟠 P1 - 圖神經網路套件
- [ ] 安裝 PyTorch Geometric 2.6+ 🕐 1h
  ```bash
  pip install torch-geometric pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
  ```
- [ ] 測試異質圖 (HeteroData) 基本操作 🕐 1h
- [ ] 測試大規模圖載入（模擬500萬節點） 🕐 1h

### 🟡 P2 - 加速與優化庫
- [ ] 安裝 FlashAttention-2 🕐 2-3h
  ```bash
  pip install flash-attn --no-build-isolation
  ```
- [ ] 測試 FlashAttention 啟用狀態與加速效果 🕐 1h
- [ ] 安裝 xformers (備用) 🕐 1h
- [ ] 實現 `AdaptiveAttentionBackend` 🕐 2h
- [ ] 安裝其他依賴（transformers, owlready2, faiss-gpu, hnswlib） 🕐 1h
- [ ] 建立完整的 `requirements.txt` 與 `requirements_arm.txt` 🕐 1h

**小計**: 📅 3-4 天

---

## 1.2 本體知識庫構建 (Week 1-2) **【新增/擴展】**

### 🔴 P0 - 本體下載與解析
- [ ] 下載 HPO 本體 (OBO/OWL格式) 📅 0.5天
- [ ] 下載 MONDO 本體 📅 0.5天
- [ ] 下載 GO 本體 📅 0.5天
- [ ] 實現 `owlready2` 本體載入器 🕐 3h
- [ ] 解析本體層次結構（is-a關係） 🕐 4h
- [ ] 建立本體節點索引（快速查找） 🕐 2h

### 🔴 P0 - 本體推理引擎
- [ ] 實現 `OntologyKnowledgeBase` 類 📅 1天
  - [ ] 載入多個本體 🕐 2h
  - [ ] 構建統一的層次索引 🕐 3h
  - [ ] 實現祖先/後代查詢 🕐 2h
  - [ ] 實現最短路徑計算 🕐 2h

### 🟠 P1 - 本體約束規則提取
- [ ] 實現 `OntologyConstraints` 類 📅 2天
  - [ ] 從本體提取互斥規則（disjoint with） 🕐 3h
  - [ ] 從本體提取蘊含規則（必然伴隨） 🕐 3h
  - [ ] 從 EHR 統計提取共現規則 🕐 4h
  - [ ] 實現約束驗證函數 🕐 3h

### 🟠 P1 - 疾病共現圖構建
- [ ] 實現 `DiseaseCooccurrenceGraph` 📅 2天
  - [ ] 下載 MIMIC-III 數據（或使用公開統計） 🕐 4h
  - [ ] 計算疾病共現頻率 🕐 3h
  - [ ] 構建加權共現圖 🕐 2h
  - [ ] 計算統計顯著性（卡方檢驗） 🕐 2h

### 🟡 P2 - 本體可視化與驗證
- [ ] 視覺化本體樹（使用 networkx） 🕐 2h
- [ ] 驗證約束規則的正確性 🕐 2h
- [ ] 生成本體統計報告 🕐 1h

**小計**: 📆 1.5 週

---

## 1.3 資料下載與整合 (Week 2-3)

### 🔴 P0 - 資料源腳本開發
- [ ] 實現 `scripts/download_data.py` 📅 2-3 天
  - [ ] GO (Gene Ontology) 下載 🕐 2h
  - [ ] Reactome 下載 🕐 2h
  - [ ] DisGeNET 下載與解析 🕐 3h
  - [ ] NCBI Gene 下載 🕐 2h
  - [ ] ClinVar 下載（變異數據） 🕐 3h
  - [ ] Orphanet 下載 🕐 2h
  - [ ] Pubtator 3.0 API 整合 🕐 3h
  - [ ] OMIM 資料獲取（需授權） 🕐 2h
  - [ ] 實現斷點續傳與錯誤處理 🕐 2h

### 🟠 P1 - 資料清理與標準化
- [ ] 實現 `scripts/preprocess_data.py` 📅 2-3 天
  - [ ] 統一 ID 格式（Entrez Gene, UniProt等） 🕐 4h
  - [ ] 處理同義詞與別名映射 🕐 3h
  - [ ] 刪除自迴邊與重複邊 🕐 2h
  - [ ] 處理缺失值與異常值 🕐 3h
  - [ ] 資料版本標記與元數據管理 🕐 2h

### 🟠 P1 - 本體對齊與映射
- [ ] 實現 `src/kg/ontology_mapping.py` 📅 2 天
  - [ ] HPO → MONDO 映射 🕐 3h
  - [ ] 疾病 ID 統一（OMIM, Orphanet, MONDO） 🕐 4h
  - [ ] 基因 ID 映射（Entrez, Ensembl） 🕐 3h
  - [ ] 跨本體相似度計算 🕐 3h

### 🟡 P2 - 測試資料集準備
- [ ] 下載 SHEPHERD 論文測試集 🕐 2h
- [ ] 格式轉換與驗證 🕐 2h
- [ ] 準備 10-20 個真實罕見疾病案例 🕐 3h
- [ ] 建立黃金標準（ground truth） 🕐 2h

**小計**: 📆 1.5 週

---

## 1.4 知識圖譜構建（異質+超圖） (Week 3-4)

### 🔴 P0 - 異質圖構建
- [ ] 實現 `src/kg/build_hetero_kg.py` 📅 3 天
  - [ ] 定義節點類型（gene, disease, phenotype, pathway） 🕐 2h
  - [ ] 定義邊類型（15+種關係） 🕐 3h
  - [ ] 構建 PyG HeteroData 對象 🕐 4h
  - [ ] 添加反向邊（雙向可達） 🕐 2h
  - [ ] 計算閉三角關係增強 🕐 3h
  - [ ] 整合本體層次信息到圖中 🕐 3h
  - [ ] 儲存為 `.pt` 檔案 🕐 1h

### 🟠 P1 - 知識超圖構建 **【新增】**
- [ ] 實現 `src/kg/build_hypergraph.py` 📅 2 天
  - [ ] 從 EHR 挖掘頻繁症狀組合（Apriori） 🕐 4h
  - [ ] 構建超邊（3-5個實體） 🕐 3h
  - [ ] 計算超邊置信度 🕐 2h
  - [ ] 添加文獻證據（PMID） 🕐 2h
  - [ ] 超圖數據結構實現 🕐 3h

### 🟠 P1 - 圖統計與驗證
- [ ] 計算基本統計 📅 1 天
  - [ ] 節點數量（按類型分組） 🕐 1h
  - [ ] 邊數量（按關係分組） 🕐 1h
  - [ ] 度分佈分析（識別孤立節點） 🕐 2h
  - [ ] 連通性檢查（強/弱連通分量） 🕐 2h
  - [ ] 與 PrimeKG 論文數據對比 🕐 2h
- [ ] 視覺化圖結構（子圖採樣） 🕐 3h
- [ ] 生成圖質量報告 🕐 2h

### 🟡 P2 - 圖資料庫（可選）
- [ ] 安裝並配置 Neo4j 🕐 2h
- [ ] 實現 `src/kg/neo4j_interface.py` 🕐 4h
- [ ] 將圖匯入 Neo4j 🕐 3h
- [ ] 實現複雜查詢範例（Cypher） 🕐 3h

**小計**: 📆 1 週

---

## 1.5 分層本體感知模型實現 (Week 4-6) **【核心創新】**

### 🔴 P0 - 本體層次編碼器
- [ ] 實現 `src/models/ontology/hierarchy_encoder.py` 📅 2 天
  - [ ] 祖先節點嵌入聚合 🕐 3h
  - [ ] 兄弟節點嵌入聚合 🕐 3h
  - [ ] 層次位置編碼（類似Transformer） 🕐 3h
  - [ ] 多本體融合機制 🕐 3h
  - [ ] 單元測試與驗證 🕐 2h

### 🔴 P0 - 本體引導注意力
- [ ] 實現 `src/models/attention/ontology_guided_attention.py` 📅 2 天
  - [ ] 本體相似度計算（基於樹距離） 🕐 3h
  - [ ] 注意力分數混合（結構 + 語義） 🕐 3h
  - [ ] 整合 AdaptiveAttentionBackend 🕐 3h
  - [ ] 效能測試（vs標準注意力） 🕐 2h

### 🔴 P0 - 雙重本體聚合模組
- [ ] 實現 `src/models/ontology/dual_aggregation.py` 📅 2 天
  - [ ] 層次結構聚合器 🕐 3h
  - [ ] 共現關係聚合器 🕐 3h
  - [ ] 自適應權重學習 🕐 3h
  - [ ] 單元測試 🕐 2h

### 🟠 P1 - GraphGPS 實現
- [ ] 實現 `src/models/gnn/graph_gps.py` 📅 2 天
  - [ ] 本地 MPNN 層（GAT） 🕐 3h
  - [ ] 全局 Transformer 層 🕐 4h
  - [ ] Laplacian 位置編碼 🕐 3h
  - [ ] 整合本體引導注意力 🕐 3h
  - [ ] 單元測試 🕐 2h

### 🟠 P1 - 超圖卷積
- [ ] 實現 `src/models/gnn/hypergraph_conv.py` 📅 1 天
  - [ ] 超邊消息傳遞 🕐 3h
  - [ ] 超邊注意力機制 🕐 3h
  - [ ] 整合到 GraphGPS 🕐 2h

### 🟠 P1 - 本體約束解碼器
- [ ] 實現 `src/models/decoders/ontology_constrained_decoder.py` 📅 2 天
  - [ ] 基礎 DistMult 解碼器 🕐 2h
  - [ ] 本體約束檢查器整合 🕐 3h
  - [ ] 懲罰不合理預測 🕐 3h
  - [ ] 本體相似度增強 🕐 3h
  - [ ] 單元測試 🕐 2h

### 🟡 P2 - 替代解碼器（可選）
- [ ] 實現 RotatE 解碼器 🕐 4h
- [ ] 實現 QuatE 解碼器 🕐 4h
- [ ] 效能對比實驗 🕐 3h

**小計**: 📆 2 週

---

## 1.6 訓練流程與實驗 (Week 6-7)

### 🔴 P0 - 訓練管道
- [ ] 實現 `scripts/train_model.py` 📅 3 天
  - [ ] 資料載入器（大圖採樣，處理16GB限制） 🕐 4h
  - [ ] 多任務損失函數（連結預測 + 約束損失） 🕐 3h
  - [ ] 優化器配置（AdamW + 學習率調度） 🕐 2h
  - [ ] 訓練迴圈（支援FP16混合精度） 🕐 4h
  - [ ] 驗證與早停 (Early Stopping) 🕐 2h
  - [ ] 模型檢查點儲存 🕐 2h
  - [ ] TensorBoard 日誌整合 🕐 2h

### 🟠 P1 - 評估指標
- [ ] 實現 `src/utils/metrics.py` 📅 1 天
  - [ ] Hits@k (k=1,5,10,20) 🕐 2h
  - [ ] Mean Reciprocal Rank (MRR) 🕐 2h
  - [ ] Mean Rank 🕐 1h
  - [ ] NDCG 🕐 2h
  - [ ] 本體約束違反率（新指標） 🕐 2h

### 🔴 P0 - 基線訓練與調優
- [ ] 執行第一次完整訓練 📅 2 天
  - [ ] 配置訓練參數（batch size=16, lr=1e-4） 🕐 1h
  - [ ] 啟動訓練（預計24-48小時，後台運行） 🕐 48h
  - [ ] 監控訓練過程（GPU利用率、記憶體） 🕐 定期檢查
  - [ ] 驗證模型收斂 🕐 2h
  - [ ] 在測試集上評估 🕐 2h

### 🟠 P1 - 超參數調優
- [ ] 學習率掃描（1e-5 到 1e-3） 📅 0.5 天
- [ ] 隱藏層維度實驗（256, 512, 768） 📅 0.5 天
- [ ] 層數實驗（4, 6, 8層） 📅 0.5 天
- [ ] 注意力頭數實驗（4, 8, 16） 📅 0.5 天
- [ ] 本體權重調整（hierarchy vs cooccurrence） 📅 0.5 天

### 🟡 P2 - 消融實驗
- [ ] 測試不使用本體增強的效果 📅 0.5 天
- [ ] 測試不使用超圖的效果 📅 0.5 天
- [ ] 測試不使用約束解碼的效果 📅 0.5 天
- [ ] 整理實驗報告 🕐 4h

**小計**: 📆 1.5 週

---

## 1.7 DR.KNOWS路徑推理模組 (Week 7-8) **【新增】**

### 🔴 P0 - 多跳路徑檢索器
- [ ] 實現 `src/retrieval/path_retriever.py` 📅 2 天
  - [ ] BFS/DFS 多跳搜索（max_hops=3） 🕐 4h
  - [ ] 路徑過濾（長度、類型限制） 🕐 3h
  - [ ] 路徑緩存機制（加速重複查詢） 🕐 2h
  - [ ] 批次路徑檢索 🕐 3h

### 🟠 P1 - 路徑評分與排序
- [ ] 實現 `src/retrieval/path_scorer.py` 📅 2 天
  - [ ] 結構評分（路徑長度、節點度） 🕐 3h
  - [ ] 語義評分（本體相似度） 🕐 3h
  - [ ] 證據評分（文獻支持數量） 🕐 3h
  - [ ] 融合評分機制 🕐 2h
  - [ ] 路徑多樣化（MMR演算法） 🕐 3h

### 🟠 P1 - 路徑到文本轉換
- [ ] 實現 `src/retrieval/path_to_text.py` 📅 1 天
  - [ ] 結構化路徑表示 🕐 2h
  - [ ] 自然語言模板生成 🕐 3h
  - [ ] 證據鏈格式化 🕐 2h

### 🟡 P2 - LLM 整合（可選）
- [ ] 整合 OpenAI API / Claude API 🕐 2h
- [ ] 設計診斷解釋 Prompt 🕐 3h
- [ ] 實現證據增強生成 🕐 3h
- [ ] 測試解釋質量 🕐 2h

**小計**: 📅 5-6 天

---

## 1.8 患者推理管道與API (Week 8-9)

### 🔴 P0 - 患者資料處理
- [ ] 實現 `src/models/patient/patient_encoder.py` 📅 2 天
  - [ ] 患者 JSON/CSV 解析 🕐 2h
  - [ ] Phenotype 映射到 HPO 節點 🕐 3h
  - [ ] 生成患者子圖（K-hop=2） 🕐 4h
  - [ ] 子圖與本體知識融合 🕐 3h

### 🟠 P1 - 時序建模（可選，Phase 2實現）
- [ ] 實現 `src/models/patient/temporal_encoder.py` 📅 2 天
  - [ ] Visit-level編碼器 🕐 3h
  - [ ] Neural ODE 實現 🕐 4h
  - [ ] Time-aware Transformer 🕐 4h

### 🔴 P0 - 基因與疾病評分
- [ ] 實現 `src/models/tasks/gene_scoring.py` 📅 1 天
  - [ ] 候選基因提取 🕐 2h
  - [ ] GNN 評分機制 🕐 3h
  - [ ] 本體約束驗證 🕐 2h
  - [ ] 排序與過濾 🕐 2h

### 🔴 P0 - 完整推理流程
- [ ] 實現 `src/inference/diagnostic_pipeline.py` 📅 2 天
  - [ ] 患者數據 → 子圖生成 🕐 2h
  - [ ] GNN編碼 → 候選生成 🕐 2h
  - [ ] 路徑檢索 → 證據收集 🕐 3h
  - [ ] 約束檢查 → 最終排序 🕐 2h
  - [ ] 結果格式化 🕐 2h

### 🔴 P0 - FastAPI 服務
- [ ] 實現 `src/api/main.py` 📅 2 天
  - [ ] 初始化 FastAPI 應用 🕐 1h
  - [ ] 載入預訓練模型 🕐 2h
  - [ ] `/api/v2/diagnose` 端點 🕐 3h
  - [ ] `/api/v2/explain` 端點（路徑解釋） 🕐 2h
  - [ ] 請求驗證（Pydantic） 🕐 2h
  - [ ] 錯誤處理與日誌 🕐 2h

### 🟠 P1 - 測試與驗證
- [ ] 使用示例患者測試 📅 1 天
  - [ ] 載入10個測試案例 🕐 1h
  - [ ] 執行推理並記錄結果 🕐 2h
  - [ ] 驗證本體約束是否生效 🕐 2h
  - [ ] 計算 Hits@10 指標 🕐 1h
  - [ ] 檢查幻覺率（錯誤預測） 🕐 2h

**小計**: 📆 1-1.5 週

---

## 1.9 跨平台兼容性實現 (Week 9-10) **【關鍵】**

### 🔴 P0 - 自適應注意力後端
- [ ] 完整實現 `src/models/attention/adaptive_backend.py` 📅 1 天
  - [ ] 平台檢測（x86 vs ARM） 🕐 1h
  - [ ] FlashAttention-2 包裝 🕐 2h
  - [ ] xformers 包裝 🕐 2h
  - [ ] PyTorch SDPA 包裝 🕐 1h
  - [ ] 自動降級邏輯 🕐 2h
  - [ ] 效能基準測試 🕐 2h

### 🔴 P0 - 跨平台向量索引
- [ ] 實現 `src/retrieval/cross_platform_index.py` 📅 1 天
  - [ ] FAISS (x86) 實現 🕐 2h
  - [ ] hnswlib (ARM) 實現 🕐 2h
  - [ ] 統一接口封裝 🕐 2h
  - [ ] 效能對比測試 🕐 2h

### 🔴 P0 - 平台自適應配置
- [ ] 實現 `src/utils/platform_config.py` 📅 1 天
  - [ ] 硬體檢測（CPU, GPU, RAM） 🕐 2h
  - [ ] 根據平台調整模型配置 🕐 3h
  - [ ] Batch size自動調整 🕐 2h
  - [ ] 記憶體預算管理 🕐 2h

### 🟠 P1 - 環境設置腳本
- [ ] 撰寫 `env/windows_x86_blackwell/install.ps1` 📅 0.5 天
  - [ ] CUDA環境檢查 🕐 1h
  - [ ] PyTorch + PyG 安裝 🕐 1h
  - [ ] FlashAttention-2 安裝 🕐 1h
  - [ ] 驗證腳本 🕐 1h

- [ ] 撰寫 `env/linux_arm_blackwell/install.sh` 📅 0.5 天
  - [ ] ARM + CUDA 檢查 🕐 1h
  - [ ] PyTorch ARM build 安裝 🕐 1h
  - [ ] 依賴編譯（如需要） 🕐 2h
  - [ ] 驗證腳本 🕐 1h

### 🟠 P1 - 容器化
- [ ] 撰寫 `docker/Dockerfile.windows` 📅 0.5 天
- [ ] 撰寫 `docker/Dockerfile.arm` 📅 0.5 天
- [ ] 撰寫 `docker-compose.yml` 🕐 2h
- [ ] 在 Windows 上測試容器 🕐 2h

**小計**: 📅 4-5 天

---

## 1.10 測試與文檔 (Week 10)

### 🔴 P0 - 單元測試
- [ ] 建立 `tests/` 目錄結構 🕐 1h
- [ ] 本體模組測試 📅 1 天
  - [ ] 測試本體載入 🕐 2h
  - [ ] 測試約束驗證 🕐 2h
  - [ ] 測試層次查詢 🕐 2h
- [ ] 模型測試 📅 1 天
  - [ ] 測試本體編碼器 🕐 2h
  - [ ] 測試 GraphGPS 前向傳播 🕐 2h
  - [ ] 測試約束解碼器 🕐 2h
- [ ] 路徑推理測試 📅 0.5 天
  - [ ] 測試路徑檢索 🕐 2h
  - [ ] 測試路徑評分 🕐 2h

### 🟠 P1 - 整合測試
- [ ] 端到端測試 📅 1 天
  - [ ] 完整推理流程測試 🕐 3h
  - [ ] API 端點測試 🕐 2h
  - [ ] 錯誤處理測試 🕐 2h

### 🟠 P1 - 效能測試
- [ ] 推理延遲基準（單患者） 🕐 2h
- [ ] 記憶體使用分析（16GB限制） 🕐 2h
- [ ] 吞吐量測試 (QPS) 🕐 2h
- [ ] 與 SHEPHERD 基準對比 🕐 3h

### 🟡 P2 - 文檔完善
- [ ] 撰寫 `docs/architecture.md`（架構說明） 🕐 4h
- [ ] 撰寫 `docs/ontology_guide.md`（本體使用指南） 🕐 3h
- [ ] 撰寫 `docs/deployment.md`（部署指南） 🕐 3h
- [ ] 更新 `README.md` 🕐 2h
- [ ] 生成 API 文檔（Swagger） 🕐 1h

**小計**: 📅 4-5 天

---

## Phase 1 檢查清單 ✅

完成 Phase 1 需滿足以下所有標準:

### 功能性
- [ ] ✅ Windows 環境可正常運行
- [ ] ✅ 完整知識圖譜構建完成（含本體+超圖）
- [ ] ✅ 分層本體感知GNN訓練收斂
- [ ] ✅ 本體約束有效（驗證率>95%）
- [ ] ✅ 路徑推理可用（能返回證據鏈）

### 效能指標
- [ ] ✅ **Hits@10 ≥ 70%** (vs SHEPHERD 60%)
- [ ] ✅ **幻覺率 ≤ 10%** (vs 傳統20%)
- [ ] ✅ 能為測試患者返回候選基因
- [ ] ✅ API 正常回應

### 技術質量
- [ ] ✅ 推理延遲 < 2 秒（單患者，Windows）
- [ ] ✅ 單元測試覆蓋率 > 70%
- [ ] ✅ 跨平台兼容層實現（未必在ARM上測試）
- [ ] ✅ 完整文檔

**Phase 1 預計完成時間**: 8-10 週

---

# Phase 2: 進階功能與優化

**目標**: 提升至臨床級精準度，Hits@10 ≥ 80%  
**預計時間**: 4-5 週  
**前置條件**: Phase 1 完成

---

## 2.1 Neural ODE 時序建模 (Week 11-12)

### 🟠 P1 - ODE 函數實現
- [ ] 實現 `src/models/patient/ode_func.py` 📅 2 天
  - [ ] 定義狀態演化函數 🕐 3h
  - [ ] 整合患者特徵 🕐 3h
  - [ ] 可訓練參數設計 🕐 2h

### 🟠 P1 - 時序編碼器整合
- [ ] 完整實現 `TemporalPatientEncoder` 📅 2 天
  - [ ] ODE 求解器整合（torchdiffeq） 🕐 3h
  - [ ] Time-aware Transformer 實現 🕐 4h
  - [ ] 與圖編碼器融合 🕐 3h

### 🟡 P2 - 時序實驗
- [ ] 準備時序數據集（多次就診） 🕐 4h
- [ ] 訓練時序模型 📅 1 天
- [ ] 評估時序預測能力 🕐 3h
- [ ] 與靜態模型對比 🕐 2h

**小計**: 📅 5-6 天

---

## 2.2 GraphRAG 深度整合 (Week 12-13)

### 🟠 P1 - 向量索引優化
- [ ] 實現增量索引更新 🕐 3h
- [ ] 實現索引壓縮（PQ/OPQ） 🕐 4h
- [ ] GPU加速搜索（FAISS GPU） 🕐 2h

### 🟠 P1 - 子圖檢索優化
- [ ] 實現個性化 PageRank 🕐 3h
- [ ] 實現圖采樣策略（PPR, RWR） 🕐 4h
- [ ] 子圖質量評估 🕐 2h

### 🟠 P1 - LLM 證據解釋
- [ ] 設計多種 Prompt 模板 🕐 3h
- [ ] 實現 Chain-of-Thought 推理 🕐 3h
- [ ] 實現證據驗證機制 🕐 3h
- [ ] 人工評估解釋質量（10個案例） 🕐 4h

**小計**: 📅 4-5 天

---

## 2.3 模型壓縮與加速 (Week 13-14)

### 🟠 P1 - 量化
- [ ] FP16 自動混合精度 🕐 2h
- [ ] INT8 量化（PyTorch量化工具） 📅 1 天
- [ ] 動態量化 vs 靜態量化對比 🕐 3h

### 🟡 P2 - 知識蒸餾（可選）
- [ ] 訓練教師模型（大模型） 📅 2 天
- [ ] 蒸餾到學生模型（小模型） 📅 2 天
- [ ] 效能評估 🕐 3h

### 🟠 P1 - 推理優化
- [ ] 批次推理優化 🕐 3h
- [ ] KV Cache 優化（Transformer） 🕐 2h
- [ ] 預計算常用嵌入 🕐 2h
- [ ] TorchScript 導出（加速推理） 🕐 3h

**小計**: 📅 6-8 天

---

## 2.4 進階實驗與調優 (Week 14-15)

### 🟠 P1 - 消融研究
- [ ] 完整消融實驗設計 🕐 2h
- [ ] 運行所有消融變體 📅 2 天
- [ ] 統計分析與可視化 🕐 4h

### 🟡 P2 - 與其他方法對比
- [ ] 實現 SHEPHERD 基線 📅 2 天
- [ ] 實現簡單 GNN 基線 📅 1 天
- [ ] 公平對比實驗 🕐 4h
- [ ] 撰寫對比報告 🕐 4h

### 🟡 P2 - 可視化與分析
- [ ] 注意力權重可視化 🕐 3h
- [ ] 路徑可視化（交互式） 🕐 4h
- [ ] 錯誤分析（失敗案例） 🕐 4h
- [ ] 生成分析報告 🕐 3h

**小計**: 📅 5-6 天

---

## Phase 2 檢查清單 ✅

完成 Phase 2 需滿足以下標準:

### 效能提升
- [ ] ✅ **Hits@10 ≥ 80%** (vs Phase 1的70%)
- [ ] ✅ **幻覺率 ≤ 5%** (vs Phase 1的10%)
- [ ] ✅ MRR ≥ 0.65

### 功能完整性
- [ ] ✅ 時序建模可用（如有時序數據）
- [ ] ✅ GraphRAG 提供高質量解釋
- [ ] ✅ 推理延遲 < 1.5 秒
- [ ] ✅ 模型量化成功（INT8可用）

### 可用性
- [ ] ✅ 完整的可視化界面（可選）
- [ ] ✅ 消融研究完成
- [ ] ✅ 與人類專家對比（至少10個案例）

**Phase 2 預計完成時間**: 4-5 週

---

# Phase 3: ARM 部署與生產化

**目標**: DGX Spark部署，完整CI/CD  
**預計時間**: 3-4 週  
**前置條件**: Phase 1 完成（Phase 2 可選）

---

## 3.1 DGX Spark 環境準備 (Week 15)

### 🔴 P0 - 環境探測與驗證
- [ ] SSH 連線至 DGX Spark 🕐 0.5h
- [ ] 驗證硬體規格 📅 0.5 天
  - [ ] CPU: 20核 ARM v9.2 🕐 0.5h
  - [ ] GPU: Blackwell (nvidia-smi) 🕐 0.5h
  - [ ] 記憶體: 128GB 統一記憶體 🕐 0.5h
  - [ ] 儲存: 檢查可用空間 🕐 0.5h
  - [ ] 網絡: 測試 200GbE 🕐 1h

### 🔴 P0 - 軟體環境測試
- [ ] 測試 DGX OS 版本 🕐 0.5h
- [ ] 測試 CUDA 12.8 可用性 🕐 1h
  ```bash
  nvcc --version
  python -c "import torch; print(torch.cuda.is_available())"
  ```
- [ ] 測試 PyTorch ARM build 📅 0.5 天
  ```bash
  pip install torch==2.8.0 --index-url .../cu128
  ```

### 🔴 P0 - 依賴安裝測試
- [ ] PyTorch Geometric 安裝 📅 0.5 天
  - [ ] 嘗試 pip 安裝 🕐 1h
  - [ ] 如失敗，從源碼編譯 🕐 3h
  - [ ] 驗證基本功能 🕐 1h

- [ ] FlashAttention-2 嘗試 📅 0.5 天
  - [ ] 嘗試編譯（預期失敗） 🕐 2h
  - [ ] 確認降級到 xformers 或 SDPA 🕐 1h
  - [ ] 驗證注意力後端 🕐 1h

- [ ] 其他依賴安裝 🕐 2h
  - [ ] hnswlib (替代FAISS) 🕐 0.5h
  - [ ] transformers, owlready2 等 🕐 1h
  - [ ] 驗證所有import 🕐 0.5h

**小計**: 📅 3-4 天

---

## 3.2 模型遷移與優化 (Week 16)

### 🔴 P0 - 模型適配
- [ ] 載入 Windows 訓練的模型 🕐 1h
- [ ] 驗證模型在 ARM 上可運行 🕐 2h
- [ ] 測試推理正確性（vs Windows結果） 🕐 3h

### 🟠 P1 - ARM 特定優化
- [ ] 根據128GB記憶體調整配置 📅 1 天
  - [ ] 增大 batch size (64 → 128) 🕐 2h
  - [ ] 調整圖採樣策略 🕐 2h
  - [ ] 利用統一記憶體優勢 🕐 2h

- [ ] 注意力後端驗證 🕐 2h
  - [ ] 確認使用 PyTorch SDPA 🕐 1h
  - [ ] 測量效能差異 🕐 1h

- [ ] 效能基準測試 📅 0.5 天
  - [ ] 推理延遲（單患者） 🕐 1h
  - [ ] 吞吐量（批次推理） 🕐 1h
  - [ ] 記憶體使用分析 🕐 1h
  - [ ] 與 Windows 對比 🕐 1h

**小計**: 📅 3-4 天

---

## 3.3 部署與服務 (Week 17)

### 🔴 P0 - 生產部署
- [ ] 使用 Docker 部署 API 服務 🕐 3h
  ```bash
  docker build -f docker/Dockerfile.arm -t kg-engine:arm .
  docker run --gpus all -p 8000:8000 kg-engine:arm
  ```
- [ ] 配置 Nginx 反向代理 🕐 2h
- [ ] 設置 HTTPS (Let's Encrypt) 🕐 2h
- [ ] 測試外部訪問 🕐 1h

### 🟠 P1 - 服務優化
- [ ] 實現健康檢查端點 🕐 1h
- [ ] 實現優雅關閉 🕐 1h
- [ ] 請求限流 (Rate Limiting) 🕐 2h
- [ ] 結果緩存 (Redis) 🕐 3h

### 🟠 P1 - 監控與日誌
- [ ] 設置 Prometheus + Grafana 📅 1 天
  - [ ] 安裝與配置 🕐 3h
  - [ ] 自定義儀表板 🕐 3h
  - [ ] 告警規則設置 🕐 2h

- [ ] 設置日誌收集 🕐 3h
  - [ ] 結構化日誌（JSON） 🕐 1h
  - [ ] 日誌輪轉 🕐 1h
  - [ ] 錯誤追蹤 🕐 1h

**小計**: 📅 3-4 天

---

## 3.4 CI/CD 與自動化 (Week 18)

### 🟠 P1 - GitHub Actions
- [ ] 設置多平台測試 📅 1 天
  - [ ] Windows x86 workflow 🕐 2h
  - [ ] ARM 模擬測試 (QEMU) 🕐 3h
  - [ ] 自動化單元測試 🕐 2h

- [ ] 設置自動部署 🕐 3h
  - [ ] 推送到 Docker Hub 🕐 1h
  - [ ] 自動部署至 ARM 伺服器 🕐 2h

### 🟡 P2 - 資料更新自動化
- [ ] 實現 `scripts/update_kg.py` 📅 1 天
  - [ ] 檢查資料源更新 🕐 2h
  - [ ] 增量更新知識圖譜 🕐 3h
  - [ ] 重新訓練嵌入 🕐 2h
  - [ ] 更新向量索引 🕐 2h

- [ ] 設置定時任務 🕐 2h
  - [ ] cron job 或 Airflow 🕐 1h
  - [ ] 通知機制 🕐 1h

### 🟡 P2 - 版本管理
- [ ] 資料版本標記系統 🕐 2h
- [ ] 模型版本控制 (DVC) 🕐 3h
- [ ] 回滾機制 🕐 2h

**小計**: 📅 3-4 天

---

## 3.5 最終驗證與文檔 (Week 18-19)

### 🔴 P0 - 完整測試
- [ ] 端到端測試（兩個平台） 📅 1 天
- [ ] 壓力測試（高並發） 🕐 3h
- [ ] 容錯測試（故障恢復） 🕐 3h

### 🟠 P1 - 效能報告
- [ ] 生成完整效能報告 🕐 4h
  - [ ] 兩平台對比 🕐 2h
  - [ ] 與 SHEPHERD 對比 🕐 2h

### 🟡 P2 - 使用者文檔
- [ ] API 使用指南 🕐 3h
- [ ] 部署運維手冊 🕐 4h
- [ ] 故障排查指南 🕐 3h
- [ ] 視頻演示（可選） 🕐 4h

**小計**: 📅 3-4 天

---

## Phase 3 檢查清單 ✅

完成 Phase 3 需滿足以下標準:

### 部署完整性
- [ ] ✅ ARM 環境成功部署
- [ ] ✅ API 正常運行且可外部訪問
- [ ] ✅ Docker 容器穩定運行

### 效能指標
- [ ] ✅ 推理延遲 < 2 秒 (ARM, 可接受降級)
- [ ] ✅ 吞吐量 > 30 QPS
- [ ] ✅ 效能與 Windows 差距 < 30%

### 運維準備
- [ ] ✅ 完整監控與日誌系統
- [ ] ✅ CI/CD 流程運作正常
- [ ] ✅ 資料自動更新機制
- [ ] ✅ 完整部署文檔

### 質量標準
- [ ] ✅ 效能報告與對比
- [ ] ✅ 通過壓力測試
- [ ] ✅ 故障恢復機制驗證

**Phase 3 預計完成時間**: 3-4 週

---

# 長期維護與迭代

## 定期任務

### 每週
- [ ] 檢查系統健康狀態（Grafana）
- [ ] 審查監控告警
- [ ] 備份模型與配置

### 每月
- [ ] 更新 OMIM, DisGeNET, ClinVar
- [ ] 重新訓練嵌入向量（增量）
- [ ] 效能基準測試
- [ ] 安全性漏洞掃描

### 每季
- [ ] 更新 GO, Reactome, HPO, MONDO
- [ ] 模型完整重新訓練
- [ ] 系統效能優化審查
- [ ] 文檔更新

### 每年
- [ ] 更新 Orphanet 等緩慢變化資料
- [ ] 技術棧升級評估
- [ ] 架構評估與重構
- [ ] 外部安全審計

---

# 風險管理矩陣

## 當前高風險項目 🔴

### 1. ARM環境本體依賴安裝
**風險**: PyG 或其他套件在 ARM 上安裝失敗  
**影響**: 阻塞 Phase 3  
**緩解**: 準備從源碼編譯腳本，或使用替代庫（DGL）  
**責任人**: DevOps  
**截止時間**: Week 15

### 2. 16GB VRAM 限制
**風險**: Windows環境無法載入完整圖  
**影響**: 訓練效能受限  
**緩解**: 子圖採樣、梯度累積  
**責任人**: ML Engineer  
**截止時間**: Week 4

## 中風險項目 🟡

### 3. 本體對齊品質
**風險**: 跨本體映射錯誤率高  
**影響**: 約束驗證失效  
**緩解**: 人工審核關鍵映射  
**責任人**: Data Engineer  
**截止時間**: Week 3

### 4. LLM API 成本
**風險**: GraphRAG 使用 LLM 成本過高  
**影響**: 運營成本  
**緩解**: 使用本地小模型或模板生成  
**責任人**: Product  
**截止時間**: Week 13

---

# 附錄

## 時間估算總結 v2.0

| 階段 | 最短 | 最長 | 平均 | 關鍵路徑 |
|------|------|------|------|----------|
| Phase 1: Windows MVP+ | 8 週 | 10 週 | 9 週 | 本體+模型+路徑 |
| Phase 2: 進階功能 | 4 週 | 5 週 | 4.5 週 | ODE+GraphRAG |
| Phase 3: ARM 部署 | 3 週 | 4 週 | 3.5 週 | 環境適配 |
| **總計** | **15 週** | **19 週** | **17 週** | **約4-5月** |

## 資源需求

### 人力（建議）
- **ML 工程師**: 1人（全職）- 模型與演算法
- **資料工程師**: 0.5人 - 本體與知識圖譜
- **DevOps**: 0.3人 - 部署與CI/CD
- **領域專家**: 0.2人 - 醫療知識驗證

### 計算資源
- **Windows 工作站**: 1 台（Blackwell GPU @ 16GB）
- **DGX Spark**: 1 台（128GB統一記憶體）
- **儲存**: 1TB (Windows), 2TB (DGX Spark推薦)
- **網絡**: 1Gbps+ 互聯網（下載資料）

### 外部服務（可選）
- **LLM API**: OpenAI / Claude (~$100-500/月)
- **監控**: Grafana Cloud (免費層足夠)
- **CI/CD**: GitHub Actions (免費額度)
- **資料庫**: Neo4j (社群版免費)

---

**版本**: v2.0 (升級版)  
**最後更新**: 2025-10-07  
**升級重點**: 本體深度整合、知識超圖、跨平台兼容  
**下次審查**: 每週一更新進度