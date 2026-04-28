# SHEPHERD-Advanced 開發進度匯報

> **用途**：向醫療團隊與醫院管理層匯報系統開發進度。每個重要里程碑都會記錄在此文件中，包含可展示的具體成果。
>
> **更新原則**：只記錄值得匯報的節點（milestone），不記錄內部 debug 細節。

---

## 里程碑 1：後端核心架構修復完成（2026-04-08）

### 系統狀態

SHEPHERD-Advanced 的後端（診斷推理引擎）已完整對齊原始 SHEPHERD 論文（Alsentzer et al., npj Digital Medicine 2025）的設計，並通過全套自動化測試驗證（24 個整合測試全部通過）。

### 三個關鍵能力已就緒

1. **診斷排名**：GNN + 最短路徑融合評分，按照原論文公式 `final = η × GNN_sim + (1-η) × SP_sim`
2. **可解釋證據**：每個候選都附帶 Mode A（直接路徑）或 Mode B（類比推理）的證據說明
3. **信心標籤**：四種層級的信心標籤，讓醫生快速判斷結果可信度

### 設計契約

- **評分與證據完全分離**：證據只是「為什麼這樣排」的說明，絕不反向影響分數
- **這個契約已經被自動化測試鎖定**，防止未來的代碼修改不小心破壞這個邊界

---

## 里程碑 2：後端 CLI 端到端驗證通過（2026-04-28）

### 驗證流程

首次完整的 CLI 端到端測試，從資料生成到 HTTP API 回傳，全部在真實環境中運行：

| 步驟 | 操作 | 結果 |
|------|------|------|
| 1 | 建立示範知識圖譜 | ✅ 24 節點（10 症狀 + 8 基因 + 5 疾病 + 1 小鼠基因） |
| 2 | 訓練 GNN 模型 | ✅ Loss 收斂，checkpoint 儲存成功 |
| 3 | 預計算最短路徑 | ✅ 98 對 phenotype-target 路徑 |
| 4 | 啟動 API 伺服器 | ✅ GNN 就緒、SP 就緒、GPU 自動偵測 |
| 5 | 呼叫診斷端點 | ✅ 真實 pipeline 結果（見下方 demo） |

### API 回傳結果展示

**輸入**：病患表型 = Seizure (HP:0001250) + Global developmental delay (HP:0001263)

**輸出**（前 3 名候選診斷）：

| Rank | 疾病名稱 | 綜合信心分數 | GNN 嵌入相似度 | 最短路徑相似度 | 信心標籤 | 最短路徑長度 |
|------|---------|------------|--------------|-------------|---------|------------|
| #1 | Achondroplasia（軟骨發育不全） | 0.515 | 0.665 | 0.167 | ⚠️ Weak path support | 4 hops |
| #2 | Dravet syndrome（Dravet 綜合症） | 0.488 | 0.526 | 0.400 | ✅ Strong path support | 1 hop |
| #3 | Tuberous sclerosis complex（結節性硬化症） | 0.462 | 0.517 | 0.333 | ✅ Strong path support | 2 hops |

### 各欄位解讀（供醫療團隊理解）

- **綜合信心分數**：最終排名依據，由 70% GNN 嵌入相似度 + 30% 最短路徑相似度混合而成
- **GNN 嵌入相似度**：AI 從知識圖譜結構中學到的「這個症狀組合有多像某個疾病」的判斷（0-1）
- **最短路徑相似度**：知識圖譜上，從病患症狀到候選疾病的最短距離轉換成的分數（越近越高）
- **信心標籤**：系統自動為每個候選附上的可信度標記：
  - 🟢 **Strong path support**：知識圖譜上有 ≤2 步的直接路徑，證據強度高
  - 🟡 **Weak path support**：知識圖譜上有 3-4 步的間接路徑，需臨床驗證
  - 🔵 **Analogy-based**：沒有直接路徑，但 AI 發現它與某個有路徑的已知基因高度相似
  - ⚪ **Insufficient evidence**：無法組裝有效證據，建議以獨立臨床判斷為準
- **最短路徑長度**：從症狀到疾病在知識圖譜上需要走幾步

### 證據路徑範例（Dravet syndrome，Rank #2）

系統為 Dravet syndrome 提供了以下直接證據路徑：

```
路徑 1（1 hop）：Seizure → Dravet syndrome
路徑 2（3 hops）：Global developmental delay → SCN1A → Seizure → Dravet syndrome  
路徑 3（4 hops）：Global developmental delay → CDKL5 deficiency → CDKL5 → Seizure → Dravet syndrome
```

醫生可以直觀看到 AI 的「推理邏輯」——系統認為 Dravet syndrome 是候選，是因為病患的 Seizure 症狀在知識圖譜上直接連接到 Dravet syndrome，而且透過 SCN1A 基因也有間接連接。

### 關於排名準確性的說明

> **注意**：上述示範使用的是 24 節點的迷你知識圖譜和簡易訓練模型（20 個 epoch 的自監督學習），**不是真正的臨床級模型**。因此排名不完全反映臨床合理性（例如 Achondroplasia 排在 Dravet 前面）。
>
> 在使用完整 PrimeKG（13 萬+ 節點）和正式的 MultiTaskLoss 訓練後，GNN 會學到真正的「症狀-疾病」關聯，排名會顯著改善。
>
> 但本次 demo 的目的是**驗證系統管線能完整運作**——從資料載入到模型推理到 HTTP API 回傳，全部正常。這個管線的正確性已經被 24 個自動化測試鎖定。

---

## 里程碑 3：前端診斷介面上線（2026-04-28）

### 完成的工作

醫生用的「病患診斷介面」已實作完成並通過端到端測試。這是整個系統中**醫療團隊直接使用**的核心功能頁面。

### 功能介紹

#### 診斷介面（Diagnosis Tab）

醫生在左側輸入病患的 HPO 症狀代碼（例如 `HP:0001250 — Seizure`），點擊「Run Diagnosis」後，右側即時顯示：

1. **候選疾病排名表**：每個候選顯示綜合信心分數、GNN 嵌入相似度、最短路徑相似度、以及信心標籤
2. **證據面板**：選擇任一候選後，展示 AI 的推理路徑（Mode A 直接路徑或 Mode B 類比證據）
3. **完整解釋**：可展開查看詳細的推理說明，包含關聯基因和推理路徑分數

<!-- 如果有截圖，放在 docs/images/ 並取消下面的註解 -->
[診斷結果展示](images/diagnosis_results.png) 
[證據面板展示](images/evidence_panel.png) 

#### 模型配置面板（Model Configuration）

診斷介面頂部有一個可摺疊的「Model Configuration」面板，提供：

- **Workspace 路徑設定**：指定包含 KG 和模型檔案的資料夾（預設使用標準路徑，醫生不需要更改）
- **Pipeline 狀態顯示**：即時展示 GNN、SP、KG 的載入狀態和統計資料
- **檔案完整性檢查**：自動掃描資料夾中的必要檔案，用 ✅/❌ 標記
- **KG 版本校驗**：當模型和 KG 版本不匹配時，自動顯示黃色警告
- **三個操作按鈕**：
  - 🔄 Reload Pipeline — 載入或重新載入模型
  - 💾 Save Config — 儲存路徑設定（下次啟動自動記住）
  - ↩️ Reset Defaults — 恢復預設路徑

### 展示結果（Seizure + Global developmental delay）

| Rank | 疾病 | 信心分數 | GNN | SP | 信心標籤 |
|------|------|---------|-----|-----|---------|
| #1 | Dravet syndrome | 0.512 | 0.560 | 0.400 | 🟢 Strong path support |
| #2 | Rett syndrome | 0.489 | 0.556 | 0.333 | 🟢 Strong path support |
| #3 | CDKL5 deficiency disorder | 0.476 | 0.509 | 0.400 | 🟢 Strong path support |
| #4 | Achondroplasia | 0.416 | 0.522 | 0.167 | 🟡 Weak path support |
| #5 | Tuberous sclerosis complex | 0.371 | 0.388 | 0.333 | 🟢 Strong path support |

**臨床解讀**：Dravet syndrome 排名第一，符合 Seizure + Developmental delay 的臨床表現。Supporting genes 包含 SCN1A（Dravet 的主要致病基因），Evidence Panel 顯示從 HP:0001250（Seizure）到 MONDO:0011073（Dravet syndrome）只需 1 hop 的直接路徑。

> **注意**：此結果使用 24 節點迷你知識圖譜和簡易訓練模型。正式 PrimeKG 訓練後排名精確度會進一步提升。

### 資料結構重整

同步完成了資料目錄結構的重大重構——從舊的「data/ + models/ 分離」改為「workspace 統一管理」：

```
data/workspaces/{KG版本名稱}/      ← 一個資料夾 = 一個完整的工作環境
├── kg.json                         ← 知識圖譜
├── node_features.pt, edge_indices.pt, num_nodes.json  ← 圖數據
├── shortest_paths.pt               ← 最短路徑 lookup
└── checkpoints/                    ← 基於此 KG 訓練的模型（可有多個）
```

**好處**：
- 不可能配錯 KG 版本和 model 版本（同資料夾 = 同版本）
- 切換版本只需要改一個路徑
- 內建 fingerprint 校驗機制防止誤配

### 目前完成的子頁面

| 使用者 | 子頁面 | 狀態 |
|--------|-------|------|
| 臨床醫生 | **病患診斷頁面** | ✅ **已完成** |
| 臨床醫生 | 相似病患檢索（Patients-Like-Me） | ⏸ 規劃中 |
| 工程團隊 | 訓練監控 console | ✅ 已完成 |
| 工程團隊 | 超參數調整 | ✅ 已完成 |
| 工程團隊 | Checkpoint 管理 | ⏸ 規劃中 |
| 工程團隊 | 系統健康儀表板 | ⏸ 規劃中 |

### 下一步

- 完善 UI 細節（HPO 自動完成搜尋、結果匯出等）
- Patients-Like-Me 檢索頁面
- 工程團隊進階頁面

---

<!-- 
未來里程碑模板：

## 里程碑 N：標題（日期）

### 完成的工作
- ...

### 可展示的成果
- ...

### 下一步
- ...
-->
