# GNN Architecture Notes

> **Purpose**: Technical reference for the engineering team. Evaluates GNN architecture options for SHEPHERD-Advanced, including compatibility with PyTorch 2.9+ and PyG 2.7+.
>
> **Last updated**: 2026-05-14

---

## Current Architecture

SHEPHERD-Advanced supports three GNN conv types via PyG's `HeteroConv` wrapper:

| Conv Type | PyG Class | AMP | Heterogeneous | VRAM (batch=128, 52K KG) | Status |
|-----------|-----------|-----|---------------|--------------------------|--------|
| **GAT** | `GATConv` via `HeteroConv` | ✅ float16 | ✅ via wrapper | ~6 GB | **Production default** |
| **HGT** | `HGTConv` (native) | ❌ float32 only | ✅ native | >16 GB (batch=16 minimum) | DGX only |
| **SAGE** | `SAGEConv` via `HeteroConv` | ✅ float16 | ✅ via wrapper | ~5 GB | Available |

### HGT 的 AMP 不相容問題

HGTConv 內部使用 `pyg_lib.ops.segment_matmul`，該操作只支援 float32。float16 和 bfloat16 都會報 `RuntimeError`。因此 HGT 必須禁用 AMP，導致 VRAM 消耗約為 GAT 的 2-3 倍（float32 + 統一 attention 矩陣）。

---

## PyG 內建的異質圖 Conv 層（可直接使用）

以下全部已在 PyG 2.7 中，可用於 `HeteroConv` 包裝或直接使用：

| 層 | 論文 | 特點 | AMP | 適用場景 |
|----|------|------|-----|---------|
| **GATConv** | Veličković 2018 | 靜態 attention | ✅ | 目前使用中 |
| **GATv2Conv** | Brody 2022 (ICLR) | **動態 attention**，修復 GAT 的排名問題 | ✅ | **建議升級** |
| **HGTConv** | Hu 2020 (WWW) | 原生 type-aware attention | ❌ | DGX / 大 VRAM |
| **RGCNConv** | Schlichtkrull 2018 | 每種邊類型獨立權重 | ✅ | 簡單 baseline |
| **RGATConv** | Busbridge 2019 | RGCN + attention | ✅ | 邊類型較少時 |
| **HANConv** | Wang 2019 (WWW) | 階層式 meta-path attention | ✅ | 有明確 meta-path 時 |
| **HEATConv** | Mo 2021 | 異質邊增強 attention | ✅ | 軌跡預測 |

### GATv2Conv：最值得嘗試的升級

GATv2 修復了原始 GAT 的「靜態 attention」問題 — 原始 GAT 的 attention ranking 對所有 query 節點相同（無法區分），GATv2 使用動態 attention 讓每個節點能根據自身特徵調整關注度。

**實作難度**：極低。在 `layers.py` 中把 `GATConv` 換成 `GATv2Conv`，參數完全相同。AMP 相容，VRAM 消耗和 GAT 幾乎一樣。

```python
# 現在
from torch_geometric.nn import GATConv
conv = GATConv(256, 32, heads=8, ...)

# 升級
from torch_geometric.nn import GATv2Conv
conv = GATv2Conv(256, 32, heads=8, ...)
```

---

## 外部架構評估

### Tier 1：直接和罕見病相關

#### PhenoKG (2025, arXiv 2506.13119)

在 MyGene2 數據集上超越 SHEPHERD：MRR 24.64% vs 19.02%。

**架構細節**（從論文提取）：
- GNN 層：**GATv2** × 3 層，2 attention heads
- Hidden dims：1024 → 256 → 512
- 加上 Transformer-based gene encoder（4 層，8 heads）
- 知識圖譜：PrimeKG（105K nodes, 1.1M edges）
- 關鍵創新：patient-specific subgraph 自動建構，不需要外部候選基因列表

**對我們的啟示**：
- PhenoKG 核心用的就是 GATv2 — 和我們的 GAT 同家族，升級成本極低
- 它的優勢來自 subgraph 建構策略和 Transformer encoder，不是 GNN 層本身
- **目前沒有公開程式碼**（2025 年 6 月論文）
- 我們可以先升級到 GATv2，再逐步實驗 patient-specific subgraph 策略

#### RareNet (2025, arXiv 2510.08655)

基於子圖的 GNN，只需病患表型即可識別致病基因。可作為獨立方法或其他方法的前/後處理篩選器。

### Tier 2：通用 Graph Transformer（不直接支援異質圖）

| 架構 | 會議 | GitHub | 異質圖 | 備註 |
|------|------|--------|--------|------|
| DUALFormer | ICLR 2025 (accepted, not awarded) | [JiamingZhuo/DUALFormer](https://github.com/JiamingZhuo/DUALFormer) | ❌ 同質圖 | Graph Transformer，不是 GNN；論文明確說異質圖支援是 open question |
| SGFormer | NeurIPS 2023 | [qitianwu/SGFormer](https://github.com/qitianwu/SGFormer) | ❌ 需適配 | O(N) 複雜度，已整合進 PyG；學術專案，2024 年中停止更新 |
| SeHGNN | AAAI 2023 | [ICT-GIMLab/SeHGNN](https://github.com/ICT-GIMLab/SeHGNN) | ✅ | 預計算鄰居聚合，訓練極快；學術專案 |
| NAGphormer | ICLR 2023 | [JHL-HUST/NAGphormer](https://github.com/JHL-HUST/NAGphormer) | ❌ | Hop2Token 設計，大圖友好 |

**重要說明**：以上均為學術論文的 reference implementation，不是持續維護的開源框架。真正活躍且持續維護的只有 **PyG 本身**（21K+ stars）。

### 頂會得獎情況

ICLR 2025 和 NeurIPS 2025 的 Best Paper 均未頒給 GNN 相關論文。GNN 目前不在 AI 研究的聚光燈下（LLM/RL/Diffusion 才是），但在生物醫學等垂直領域仍是核心工具。

---

## HGT 資源消耗優化方案

### 方案 1：Gradient Checkpointing（PyTorch 內建）

用時間換空間 — 前向傳播時不保存中間值，反向傳播時重新計算。

- **VRAM 節省**：理論 ~50%（O(n) → O(√n)）
- **速度代價**：~10-20% 慢（重新計算一次前向）
- **實作難度**：中等。用 `torch.utils.checkpoint.checkpoint()` 包裝每層 GNN

```python
from torch.utils.checkpoint import checkpoint

for layer in self.gnn_layers:
    x_dict = checkpoint(layer, x_dict, edge_index_dict, use_reentrant=False)
```

### 方案 2：降低 max_subgraph_nodes

當前預設 5000。降到 2000-3000 可以減少 subgraph 大小，但會限制 GNN 的感受野。

### 方案 3：混合精度的局部應用

HGTConv 內部 `segment_matmul` 不支援 float16/bf16，但 conv 前後的 LayerNorm、Dropout、殘差連接可以保持 float16。目前的實作已經這樣做（只有 conv 層用 float32，其餘 AMP 正常）。

### 方案 4：禁用 segment_matmul（HGT Lite）

PyG 的 `HeteroLinear`（HGTConv 內部使用）有一個 naive fallback — 當 `segment_matmul` 不可用時退回到標準 PyTorch 矩陣乘法（`x @ self.weight`），而標準矩陣乘法支援 float16。

```python
import torch_geometric
torch_geometric.typing.WITH_SEGMM = False
torch_geometric.typing.WITH_GMM = False
```

理論上可以讓 HGT 和 AMP float16 相容。速度會慢一些（沒有 CUTLASS kernel），但 VRAM 應該大幅降低。**尚未驗證**。

---

## PhenoKG 架構深度分析

### Patient-Specific Subgraph 建構策略

PhenoKG 的核心創新不在 GNN 層，而在子圖建構：

```
步驟 1：拿到病患的 HPO 表型列表（3-10 個）
步驟 2：在 KG 中從每個表型出發，計算到候選基因的最短路徑
步驟 3：收集路徑經過的所有節點 + 邊，形成 patient-specific 子圖
步驟 4：從 KG 補充相關的疾病、通路等節點（augmentation）

有候選基因列表 → 最短路徑直連到指定 ~20 個基因
無候選列表 → 取表型的 k-hop 鄰居（k=2），區域內所有基因為候選
```

**和我們的差異**：我們對所有病患用同一個通用 subgraph sampling（`max_subgraph_nodes=5000`）。PhenoKG 為每個病患建不同的子圖，讓 GNN 只看到相關的局部結構。

**實作難度**：中等（~1 週）。需要改 `DiagnosisDataLoader` 的 subgraph sampling 邏輯。核心算法是 BFS，已有現成（`compute_shortest_paths.py`）。

### Transformer Gene Encoder

```
GATv2 輸出的基因 embedding（512 維）× L 個候選基因
  ↓ Transformer Encoder（4 層，8 heads，中間維度 2048）
精煉後的基因 embedding（512 維）× L 個
  ↓ cosine similarity with patient representation
基因排名
```

GATv2 給基因「初步身份向量」，Transformer 讓候選基因互相對比（「我和其他候選有什麼不同？」），產出更精確的表示。

**實作難度**：低（~2-3 天）。`nn.TransformerEncoder` 是 PyTorch 原生組件。

### 雙目標損失函數

```
L_total = L_gene（三元組損失：拉近正確基因、推遠錯誤基因，margin=0.3）
        + L_sim（病患相似度：相同致病基因的病患在 embedding 空間靠近）
```

**實作難度**：中等（~3-5 天）。需要在 `MultiTaskLoss` 中加入 triplet loss 和 patient similarity。

### 整體評估

PhenoKG 沒有公開程式碼（2025 年 6 月論文），但其架構完全可以在 PyG + PyTorch 中實作：
- GATv2Conv：PyG 內建 ✅
- TransformerEncoder：PyTorch 內建 ✅
- Patient-specific subgraph：需要自建，但不複雜 ⚠️
- Triplet loss：PyTorch `nn.TripletMarginLoss` ✅

---

## 混合架構方案：GATv2 + Type Embedding

在不使用 HGTConv 的情況下獲得 type-awareness：

```
目前：
  HeteroConv 包 10 個獨立 GATv2 → 互不知道對方存在

加入 type embedding：
  把邊類型和節點類型編碼成 embedding，注入 GATv2 的 attention
  → 每個 GATv2 知道「我在處理什麼類型的邊和節點」
  → 保持 HeteroConv 的逐個處理（AMP 相容、低 VRAM）
  → 但有了 HGT 的核心優勢：type-awareness
```

GATv2Conv 支持 `edge_attr` 參數，可以把 type embedding 作為邊特徵傳入。

**實作難度**：低-中（~2 天）。不需要改框架。

---

## SeHGNN 家族

### SeHGNN（AAAI 2023）

核心思路：預計算 + 語義融合

```
離線預處理（一次性）：
  對每種 meta-path，用無參數的 mean aggregator 聚合鄰居特徵
  → 每個節點得到多組聚合特徵（每種 meta-path 一組）

訓練時（每 epoch）：
  Transformer-based semantic fusion → 融合不同 meta-path 的語義
  → 只訓練融合模組，不重複計算鄰居聚合
```

**優勢**：訓練速度極快（鄰居聚合只算一次）、VRAM 低
**劣勢**：只支持 2-hop meta-path，長距離依賴不足
**GitHub**：[ICT-GIMLab/SeHGNN](https://github.com/ICT-GIMLab/SeHGNN)（學術專案）

### LMSPS — Long-range Meta-path Search（NeurIPS 2024）

針對 SeHGNN 的 2-hop 限制，擴展到 6-hop：

- 自動搜索最佳 meta-path 組合（不需要人工指定）
- 長距離依賴可以捕捉更複雜的關係（例如 表型 → 基因 → 通路 → 基因 → 疾病）
- 在大規模異質圖上表現優於 SeHGNN

### Seq-HGNN

加入序列建模的 SeHGNN 變體，處理有時間順序的異質圖（和我們的靜態 KG 場景關係不大）。

---

## 計劃中的 Conv Type 選項

| 選項 | 架構 | AMP | 預估 VRAM | 狀態 |
|------|------|-----|-----------|------|
| `gat` | GATConv via HeteroConv | ✅ float16 | ~6 GB | ✅ 已實作 |
| `gatv2` | GATv2Conv via HeteroConv | ✅ float16 | ~6 GB | 🔲 待實作 |
| `hgt` | HGTConv (原生，float32) | ❌ | >16 GB | ✅ 已實作 |
| `hgt-lite` | HGTConv (禁用 segment_matmul) | ✅ 待驗證 | 待測 | 🔲 待驗證 |
| `sage` | SAGEConv via HeteroConv | ✅ float16 | ~5 GB | ✅ 已實作 |

未來可能加入：
- `gatv2-typed`：GATv2 + type embedding（混合架構）
- `sehgnn`：SeHGNN 預計算架構（需要較大改動）

### UI 設計方案

當架構選項超過 5 個時，Radio 按鈕 + 單行 info 不再適用。改用 Dropdown + 動態說明區：

```
┌──────────────────────────────────────────┐
│ GNN Conv Type                       ▼   │
│  GAT (default)                          │
└──────────────────────────────────────────┘
┌──────────────────────────────────────────┐
│ Standard graph attention network.        │
│ AMP compatible. ~6GB VRAM.               │
│ Recommended for GPUs ≤16GB.              │
└──────────────────────────────────────────┘
```

選擇不同架構時，說明區自動更新為對應的描述。每個架構 2-3 行說明，不用擠在一行裡。

---

## 建議路線圖

| 優先級 | 動作 | 預估工時 | 影響 |
|--------|------|---------|------|
| **P0** | GAT → GATv2 升級 | 0.5 天 | 動態 attention，可能改善排名精度；零 VRAM 增加 |
| **P0.5** | HGT Lite 驗證（禁用 segment_matmul） | 0.5 天 | 若成功，HGT 可在 16GB GPU 上用 AMP |
| **P1** | UI 改版（Dropdown + 動態說明） | 1 天 | 支援更多架構選項 |
| **P1** | HGT + gradient checkpointing | 1 天 | 原版 HGT 在 16GB GPU 上可用 |
| **P2** | GATv2 + type embedding | 2 天 | 「窮人版 HGT」— type-aware 但 AMP 相容 |
| **P2** | 研究 PhenoKG 的 subgraph 策略 | 1-2 週 | patient-specific subgraph 可能是精度提升的關鍵 |
| **P3** | 評估 SeHGNN / LMSPS | 1 週 | 預計算策略大幅加速訓練 |
| **P0** | GAT → GATv2 升級 | 0.5 天 | 動態 attention，可能改善排名精度；零 VRAM 增加 |
| **P1** | HGT + gradient checkpointing | 1 天 | 在 16GB GPU 上也能用 HGT |
| **P2** | 研究 PhenoKG 的 subgraph 策略 | 1-2 週 | patient-specific subgraph 可能是精度提升的關鍵 |
| **P3** | 評估 SeHGNN | 1 週 | 預計算策略可能大幅加速訓練 |

---

## References

- [PyG Heterogeneous Graph Tutorial](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/heterogeneous.html)
- [PyG Conv Layers List](https://pytorch-geometric.readthedocs.io/en/2.7.0/modules/nn.html)
- [GATv2 Paper (ICLR 2022)](https://arxiv.org/abs/2105.14491)
- [PhenoKG (arXiv 2025)](https://arxiv.org/abs/2506.13119)
- [RareNet (arXiv 2025)](https://arxiv.org/html/2510.08655)
- [SeHGNN (AAAI 2023)](https://github.com/ICT-GIMLab/SeHGNN)
- [Graph Transformer Survey (2025)](https://arxiv.org/html/2502.16533v2)
- [PyTorch Gradient Checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html)
- [LMSPS — Long-range Meta-path Search (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/4e392aa9bc70ed731d3c9c32810f92fb-Paper-Conference.pdf)
- [PyG HeteroLinear Source (segment_matmul fallback)](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/dense/linear.html)
- [GATv2 Paper (ICLR 2022)](https://arxiv.org/abs/2105.14491)
- [GATv2Conv PyG Docs](https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.nn.conv.GATv2Conv.html)
