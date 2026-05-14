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

---

## 建議路線圖

| 優先級 | 動作 | 預估工時 | 影響 |
|--------|------|---------|------|
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
