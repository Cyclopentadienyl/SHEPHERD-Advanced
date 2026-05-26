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

#### ✅ 已驗證修復：cuVS 在 cu13 環境的正確整合（2026-05-26，Phase E，DGX GB10 實測）
**定位釐清**：cuVS 是 Linux 上的**主力 GPU 向量加速後端**（加速效益強、NVIDIA 官方積極維護），**不是** faiss/Voyager 等級的備選。Voyager 只是因為 cuVS **在 Windows 沒有官方 wheel**、為了跨平台才採用的 CPU 替代品；在 Linux/DGX 上 cuVS 應為首選。

**根因（完整）**：deploy 的 Stage 3 用 `uv pip install` 裝 cuVS，而該指令**只解析 cuVS 自己的子樹**，看不到 torch 已安裝的版本約束。torch 2.10.0+cu130 對一整組共用套件有精確 pin（`cuda-bindings==13.0.3` 及 `nvidia-cublas/nvrtc/curand/cusolver/cusparse/nvjitlink` 等），但 cuVS 的 RAPIDS 鏈只要求 `>=`，於是 uv 在無約束下把它們全升到最新 → 下次 `uv sync` 依 lock 還原 → **每次 deploy 版本翻來覆去**。（早期還疊加一個 `cuvs-cu12`(CUDA 12) 的錯誤選擇，先以 `cuvs-cu13` 修正。）

**最終修復**：`deploy.sh` 改用 `cuvs-cu13`，且安裝時用 `uv export` 產生 lock 的版本約束、以 `-c` 餵給 `uv pip install`——cuVS 仍可裝自己的新套件，但**不准動任何 torch 鎖定的共用套件**。已線上驗證 `cuvs-cu13 26.4.0` 在 aarch64/py312 下對 torch 全套 pin 乾淨解析；DGX 實測連跑兩次 deploy，`cuda-bindings` 穩定 `13.0.3`、整組 nvidia 庫不再翻轉，`cuvs` 26.04.000 正常 import。同步更新 deployment-guide / blueprint 文件。

**已知殘留（cosmetic，已接受）**：`nvidia-cusparselt-cu13==0.8.0` 在**每次** `uv sync` 都會被「同版本重裝」（uv 輸出的 `~`）。經隔離測試確認此現象**與 cuVS / 本次修改無關**（兩次乾淨 `uv sync`、甚至清快取重新下載後仍重現），是 uv（含最新版）對該特定 wheel 的內在 quirk；版本不變、暖快取下僅 ~2ms hardlink 重連，無功能影響。待 uv 上游修復。

#### 📌 未來：放寬 torch/cuda 版本支援（維護筆記，2026-05-26）
目前整個依賴鏈**硬鎖在 torch 2.10.0 + cu130** 單一組合：
- `pyproject.toml`：`torch/vision/audio==2.10.0` 精確 pin + 強制 `pytorch-cu130` 索引
- `uv.lock`：鎖到帶 hash 的 `2.10.0+cu130` wheel（含 aarch64）
- `deploy.sh`：`PYG_WHEEL_URL` 寫死 `torch-2.10.0+cu130.html`

計畫適度放寬（例如新增 cu131/132 + torch 2.11/2.12 適配）。**升版時三處必須同步處理**：
1. 放寬 `pyproject.toml` 的 pin/索引並重新 `uv lock`
2. 在 DGX 上用 `scripts/build_pyg_arm.sh` 對新 torch **重編 PyG ARM wheel 並重新上傳到 GitHub Release**
   - ⚠️ **pyg-lib 的 git tag 與 torch 版本耦合**：須 pin 到對應 torch 的 release（在 `data.pyg.org/whl/torch-<新版>.html` 查官方建的 pyg_lib 版本），用 `PYGLIB_SPEC` 覆寫。HEAD/錯版會 `undefined symbol` 載入失敗。torch 2.10 → pyg-lib 0.6.0。
3. 更新 `deploy.sh` 的版本判斷與 Release 下載 URL

註：ARM 自編譯腳本設計為**版本無關**（直接對 deploy `.venv` 的 torch 編譯、GPU 算力自動偵測），故升版時**腳本本身不需改**，只需重編+重傳 wheel。版本不符時 deploy 會走「自編譯救援」分支。

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
