# SHEPHERD-Advanced 專案交接文件

**生成日期**: 2026-02-23
**生成者**: Claude 4.6 Opus (Session: claude/fix-gnn-inference-bspgX)
**用途**: 跨 Claude Code Session 的專案狀態交接 — 前端 Dashboard 開發

---

## 一、上一 Session 完成事項

### 1.1 Phase 1: GNN 推理整合 (已完成)

| Commit | 內容 |
|--------|------|
| `b90deb3` | 實現 `_calculate_gnn_score()`，替換 hardcoded 0.0 |
| `9650f78` | API 啟動流程 + PathReasoner 重構為僅做解釋 |
| `7ff4e8c` | 向量索引整合 (ANN 候選發現) |
| `f95b80f` | GNN 推理 + 向量索引整合測試 |

### 1.2 測試狀態
```
219 passed, 7 skipped (Windows 平台限制: cuVS x3, torch.compile x2, cuVS benchmark x2)
0 failed
```

### 1.3 之前 Session 修復的 PyTorch 2.9 兼容性問題
- AMP API 遷移 (`torch.cuda.amp` → `torch.amp`)
- loss_functions.py 嵌套字典/設備檢測/requires_grad
- Scheduler 除以零保護

---

## 二、本次 Session 任務: Gradio Training Dashboard

### 2.1 決策背景

**為什麼先做前端再做 Phase 2 訓練優化:**
1. 40+ 個可調參數用 CLI 調整效率低，GUI slider/dropdown 更快
2. 收斂曲線 (loss/MRR/Hits@K) 即時可視化幫助判斷訓練策略
3. GPU/RAM 資源監視幫助決定 batch size 上限
4. 訓練完直接在同介面跑推理驗證，形成閉環

### 2.2 技術選型: Gradio

- `pyproject.toml` 已宣告 `gradio>=4.0` 依賴
- Python-native，不需要寫 JS/React
- 可 mount 到現有 FastAPI app
- 內建圖表支援 (LinePlot, Plot with Matplotlib/Plotly)

---

## 三、前端現狀 (起始點)

### 3.1 目錄結構 (全空)
```
src/webui/
├── app.py                    # 空檔案 (0 lines)
└── components/
    └── __init__.py           # 空檔案 (0 lines)
```

### 3.2 規劃的元件 (來自 docs/data_structure_and_validation_v3.md)
```
src/webui/components/
├── input_form.py       # 智能患者資料輸入表單
├── hpo_search.py       # HPO 術語搜尋
└── result_viewer.py    # 結果可視化
```

### 3.3 架構約束 (來自 import-linter 規則)
```
# 允許:
src.webui → src.inference (可以)
src.webui → src.api (可以)

# 禁止:
src.webui → src.training (直接調用訓練模組)
src.inference → src.webui (推理不應依賴前端)
```

**注意**: `src.webui.* -> src.training.*` 被列為禁止依賴。
Dashboard 的訓練控制需要**透過 API 層**或**subprocess** 間接調用，不能直接 import training 模組。

---

## 四、要實作的 Dashboard 規格

### 4.1 Tab 結構
```
┌─────────────────────────────────────────────────────┐
│  SHEPHERD-Advanced Dashboard                         │
├──────────┬──────────────────────────────────────────┤
│          │                                          │
│  Tab 1:  │  訓練控制台 (Training Console)             │
│  Train   │  - 參數面板 (分層摺疊: 基本/進階/專家)       │
│          │  - Start / Stop / Resume 按鈕             │
│          │  - 即時 loss + metrics 曲線               │
│          │  - GPU/RAM 資源監視                        │
│          │                                          │
│  Tab 2:  │  推理測試 (Inference)                      │
│  Infer   │  - 輸入 HPO terms                         │
│  (次要)   │  - 顯示診斷結果 + 解釋路徑                  │
│          │                                          │
│  Tab 3:  │  模型管理 (Models)                         │
│  Models  │  - Checkpoint 列表                        │
│  (次要)   │  - 指標比較表                              │
│          │                                          │
└──────────┴──────────────────────────────────────────┘
```

### 4.2 優先順序
1. **Tab 1 (Train)** — 最先做，直接幫到 Phase 2 調參
2. **Tab 2 (Infer)** — 接著做，驗證訓練成果
3. **Tab 3 (Models)** — 最後做，checkpoint 管理

### 4.3 訓練參數分層 (供 GUI 面板設計)

**Tier 1 — 基本 (必露出)**
| 參數 | 預設值 | 控件類型 |
|------|--------|---------|
| num_epochs | 100 | Number |
| learning_rate | 1e-4 | Slider (log scale) |
| batch_size | 32 | Dropdown [8,16,32,64] |
| conv_type | "gat" | Radio [gat, hgt, sage] |
| device | "auto" | Radio [auto, cuda, cpu] |
| resume_from | None | File picker |
| seed | 42 | Number |

**Tier 2 — 進階 (摺疊區)**
| 參數 | 預設值 | 控件類型 |
|------|--------|---------|
| hidden_dim | 256 | Dropdown [128,256,512] |
| num_layers | 4 | Slider [2-8] |
| dropout | 0.1 | Slider [0-0.5] |
| weight_decay | 0.01 | Slider (log) |
| scheduler_type | "cosine" | Dropdown |
| warmup_steps | 500 | Number |
| early_stopping_patience | 10 | Number |
| diagnosis_weight | 1.0 | Slider [0-2] |
| link_prediction_weight | 0.5 | Slider [0-2] |
| contrastive_weight | 0.3 | Slider [0-2] |
| ortholog_weight | 0.2 | Slider [0-2] |

**Tier 3 — 專家 (預設隱藏)**
| 參數 | 預設值 |
|------|--------|
| gradient_accumulation_steps | 1 |
| max_grad_norm | 1.0 |
| num_heads | 8 |
| use_ortholog_gate | True |
| use_amp / amp_dtype | True / float16 |
| temperature | 0.07 |
| label_smoothing | 0.1 |
| margin | 1.0 |
| num_neighbors | [15, 10, 5] |
| max_subgraph_nodes | 5000 |
| sampling_strategy | "neighbor" |

### 4.4 收斂曲線需顯示的指標
- `train_loss` (每 step)
- `val_loss` (每 epoch)
- `val_mrr` — Mean Reciprocal Rank
- `val_hits@1`, `val_hits@10`
- `learning_rate` (scheduler 曲線)

### 4.5 資源監視
- GPU utilization % (nvidia-smi / pynvml)
- GPU memory used / total
- RAM usage
- 訓練速度 (samples/sec, steps/sec)

---

## 五、現有 API 端點 (可直接用)

| 端點 | 方法 | 用途 | 狀態 |
|------|------|------|------|
| `/api/v1/diagnose` | POST | 診斷推理 | ✅ 可用 (有 mock fallback) |
| `/api/v1/hpo/search` | GET | HPO 搜尋 | ✅ 可用 (mock 資料) |
| `/api/v1/hpo/{id}` | GET | HPO 詳情 | ✅ 可用 |
| `/api/v1/disease/{id}` | GET | 疾病資訊 | ✅ 可用 (mock 資料) |
| `/health` | GET | 健康檢查 | ✅ 可用 |
| `/ready` | GET | 就緒探針 | ✅ 可用 |

**需要新增的 API (為 Dashboard 服務):**
- `POST /api/v1/training/start` — 啟動訓練
- `POST /api/v1/training/stop` — 停止訓練
- `GET /api/v1/training/status` — 訓練狀態 + 即時指標
- `GET /api/v1/training/metrics` — 歷史指標 (圖表用)
- `GET /api/v1/training/checkpoints` — Checkpoint 列表
- `GET /api/v1/system/resources` — GPU/RAM 監視

---

## 六、技術環境

```yaml
OS: Windows 11
GPU: NVIDIA GeForce RTX 5070 Ti (17.1 GB)
CUDA: 13.0
Python: 3.12
PyTorch: 2.9
PyTorch Geometric: 2.7+
Gradio: >=4.0 (已在依賴中)
FastAPI: >=0.110
```

---

## 七、關鍵檔案參考

| 檔案 | 用途 |
|-----|------|
| `src/webui/app.py` | **主要開發對象** (空) |
| `src/webui/components/` | 元件目錄 (空) |
| `src/api/main.py` | FastAPI 主程式 (Gradio mount 點) |
| `src/api/routes/diagnose.py` | 診斷 API |
| `src/training/trainer.py` | Trainer 類 (~850 行) |
| `scripts/train_model.py` | 訓練腳本 + TrainConfig dataclass |
| `src/training/loss_functions.py` | LossConfig dataclass |
| `src/kg/data_loader.py` | DataLoaderConfig dataclass |
| `docs/data_structure_and_validation_v3.md` | 含 input_form.py 範例程式碼 |
| `docs/HANDOFF_SESSION_2026-02-21.md` | 前一次交接文件 |
| `configs/deployment.yaml` | 部署配置 |

---

## 八、注意事項

1. **Import 規則**: webui 不能直接 import training，需透過 API 或 subprocess
2. **Gradio + FastAPI 整合**: 使用 `gr.mount_gradio_app(fastapi_app, gradio_app, path="/ui")` 模式
3. **訓練是長時間任務**: 需要用 subprocess 或 background task，不能阻塞 API event loop
4. **即時指標推送**: 考慮 WebSocket 或 Gradio 的 `gr.Timer` 輪詢
5. **測試**: 目前 219 passed，新增前端不應破壞現有測試

---

---

## 九、套件相容性分析 (2026-02-23)

### 9.1 背景

`pyproject.toml` 中宣告 `gradio>=5.20,<5.30`，但本環境實際安裝的是 5.50.0。
執行降版安裝時，pip 會連帶變動多個依賴套件。本節記錄完整的影響分析。

### 9.2 Gradio 5.29.1 降版影響 (pip dry-run 驗證)

| 套件 | 當前版本 | 降版至 | 影響程度 |
|------|---------|--------|---------|
| gradio | 5.50.0 | **5.29.1** | 目標版本 |
| gradio-client | 1.14.0 | **1.10.1** | Gradio 内部依賴，低風險 |
| pydantic | 2.12.3 | **2.11.10** | 見 9.3 分析 |
| pydantic-core | 2.41.4 | **2.33.2** | 跟隨 pydantic |
| websockets | 15.0.1 | 不變 | `<16,>=10` 約束，15.0.1 在範圍内 |
| pandas | 2.3.3 | 不變 | `<3,>=1` 約束 |
| pillow | 11.3.0 | 不變 | `<12,>=8` 約束 |

**不受影響的套件**: websockets, pandas, pillow, numpy, fastapi, starlette, uvicorn, httpx, anyio 等全部保持不變。

### 9.3 pydantic 降版分析

**原因**: Gradio 5.23.2 ~ 5.49.x 全部要求 `pydantic<2.12,>=2.0`。
僅 5.50.0 放寬至 `<=2.12.3`。

**pydantic `<2.12` 上限時間線** (經 PyPI JSON 原始數據驗證):

| Gradio 版本 | pydantic 約束 |
|-------------|-------------|
| 5.20.0 ~ 5.23.1 | `>=2.0` (無上限) |
| **5.23.2** ~ 5.49.x | `<2.12,>=2.0` |
| 5.50.0 | `<=2.12.3,>=2.0` |

**程式碼掃描結果**: 本專案使用的 pydantic 功能全部在 2.0~2.7 範圍内：
- `BaseModel` 子類 + `Field()` (4 個 API route 檔案)
- `field_validator` 裝飾器 (標準用法，無 2.12 新參數)
- `model_config = {"json_schema_extra": ...}`
- `model_dump(exclude_none=True)`

**未使用的 2.12+ 功能**: `defer_build`, `JsonValue`, `FailFast`, `with_config`, `experimental_allow_partial` — 全部未使用。

**結論**: pydantic 2.12.3 → 2.11.10 **安全**，不會造成功能破壞。

### 9.4 pandas 版本分析

**現狀**: pandas 3.0.0/3.0.1 已發布於 PyPI，但**所有 Gradio 5.x 均要求 `pandas<3.0`**。
安裝任何 Gradio 5.x 都會阻止使用 pandas 3.0。

**程式碼掃描結果**: 本專案僅在 1 個檔案中使用 pandas：
- `src/webui/components/training_console.py` — 僅用 `pd.DataFrame()` 建構
- 無 `append()`, `swaplevel()`, `inplace=True` 等 3.0 移除的 API
- 無 Copy-on-Write 敏感操作

**結論**: pandas 2.3.3 完全滿足需求。未來若需 pandas 3.0，須等 Gradio 放寬約束。

### 9.5 pyproject.toml 依賴完整性

| 套件 | 版本約束 | 已安裝版本 | 狀態 |
|------|---------|-----------|------|
| pronto | >=2.7 | — | 未安裝 (本地部署需安裝) |
| networkx | >=3.2 | — | 未安裝 |
| pandas | >=2.2 | 2.3.3 | ✅ |
| numpy | >=2.0 | 2.4.2 | ✅ |
| scipy | >=1.14 | — | 未安裝 |
| pyyaml | >=6.0 | 6.0.1 | ✅ |
| jsonschema | >=4.21 | — | 未安裝 |
| toml | >=0.10 | 0.10.2 | ✅ |
| pydantic | >=2.7 | 2.12.3 (→2.11.10) | ✅ |
| pydantic-settings | >=2.2 | — | 未安裝 |
| fastapi | >=0.110 | 0.131.0 | ✅ |
| uvicorn | >=0.29 | 0.41.0 | ✅ |
| gradio | >=5.20,<5.30 | 5.50.0 (→5.29.1) | ✅ |
| voyager | >=2.0 | — | 未安裝 |
| tqdm | >=4.66 | 4.67.3 | ✅ |
| requests | >=2.31 | 2.32.5 | ✅ |
| python-dotenv | >=1.0 | — | 未安裝 |

**Dry-run 驗證**: 所有 7 個未安裝套件 + `pydantic<2.12` 約束可同時安裝，零衝突。

### 9.6 本地環境驗證 (Windows, pipdeptree -w fail)

**驗證日期**: 2026-02-23
**結果**: `pipdeptree -w fail` 零失敗，所有依賴約束一致。

**核心依賴鏈交叉驗證**:
```
gradio 5.29.1     → pydantic >=2.0,<2.12  → 2.11.10 ✅
fastapi 0.128.0   → pydantic >=2.7.0      → 2.11.10 ✅
pydantic-settings  → pydantic >=2.7.0      → 2.11.10 ✅
gradio 5.29.1     → pandas >=1.0,<3.0     → 2.3.3   ✅
```
三條鏈的 pydantic 約束交集 = `>=2.7,<2.12`，2.11.10 完美落在範圍内。

**本地實際版本快照** (Windows 11, Python 3.12, CUDA 13.0):

| 套件 | 版本 | 備註 |
|------|------|------|
| torch | 2.9.0+cu130 | |
| torch-geometric | 2.7.0 | |
| gradio | 5.29.1 | 目標版本 |
| gradio-client | 1.10.1 | |
| fastapi | 0.128.0 | starlette `<0.51` |
| starlette | 0.50.0 | |
| pydantic | 2.11.10 | |
| pydantic-core | 2.33.2 | |
| pydantic-settings | 2.12.0 | |
| pandas | 2.3.3 | |
| numpy | 2.3.5 | |
| scipy | 1.17.0 | |
| pillow | 11.3.0 | |
| websockets | 15.0.1 | |

**注意事項**:
- 本地 `pyproject.toml` 中 pipdeptree 顯示 `gradio [required: >=4.0]`，
  與最新 pyproject.toml 的 `>=5.20,<5.30` 不同步，建議重新 `pip install -e .` 同步。
- 本地 FastAPI 0.128.0 鎖定 `starlette<0.51.0`；sandbox 的 0.131.0 允許到 `<0.53.0`，
  兩者均相容但更新需留意上限。

### 9.7 檢查指令備忘

```bash
# 快速檢查依賴衝突
pip check

# 完整依賴樹 (含版本約束比對，有衝突時回傳非零 exit code)
pipdeptree -w fail

# Dry-run 模擬安裝 (不實際改動)
pip install "gradio>=5.20,<5.30" --dry-run

# 比對 pyproject.toml vs 實際安裝
pip install -e ".[dev]" --dry-run
```

---

*本文件由 Claude 4.6 Opus 自動生成，用於跨 Session 專案交接*
