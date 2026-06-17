# torch.compile 實驗 — 結論記錄(已封存)

> **狀態**:**已封存(SHELVED)**。選項保留(預設關閉的開關),未合入 `main`。
> **目的**:記錄這次評估的事實結論,**避免未來重複調查同一條死路**。
> **日期**:2026-06-17
> **硬體**:NVIDIA DGX Spark,GB10(Grace-Blackwell,**sm_121**),128 GB 統一記憶體
> **軟體**:PyTorch 2.10.0+cu130,PyTorch Geometric 2.7.0
> **資料紀律**:以下數字均來自實機執行;估計值/未測項明確標示。

---

## 這是什麼

一個**可選的 `torch.compile` 開關**(CLI `--compile` / WebUI 的「🧪 Experimental Features」勾選框),**預設關閉**。實作前做過聚焦調查;設計細節見 commit `85f9a1a` / `90f70eb` / `19cd97f` 的訊息(重點:用就地 `nn.Module.compile()` 保 checkpoint 相容、`dynamic=None`、`mode=default`、`suppress_errors=True` 規避 sm_121 Triton 崩潰)。

---

## 實測結果(本次,HGT)

設定:`conv_type=hgt`、`batch_size=64`、GB10。

| | 吞吐量 | 來源 |
|---|---|---|
| **開啟 compile** | ~1.3 batch/秒(289 batch / 3m39s) | 使用者實測 |
| **eager 基線(關閉)** | ~3.5 batch/秒(1250 batch / ~6 分鐘) | 先前 HGT 訓練 |

→ **compile 反而慢約 2.6 倍**,且到第 289 batch 仍未回升(**不是一次性暖機**)。

以 `TORCH_LOGS="graph_breaks,recompiles"` 短跑數十個 batch 的診斷:
- **~43 個 graph break**
- **~45 次 recompile**(約**每個 batch 重編譯一次** → 持續重編譯 thrashing)

---

## 根因

本模型是 `torch.compile` 文獻記載的**困難案例**,三重因素疊加:
1. **異質 `HeteroConv`**(per-edge-type dispatch)→ graph break;
2. **`pyg_lib` / `torch_scatter` / `torch_sparse` 等 C++ 自訂 op**(含 `segment_matmul`)→ Dynamo 無法 trace → graph break;
3. **每 batch 子圖形狀都不同** → 動態形狀導致**持續重編譯**。

重編譯成本 + 因 graph break 喪失的融合效益,**遠超過任何加速** → 淨變慢。
(註:即使用 `mark_dynamic` 也只能減少 recompile,**救不了那 43 個 graph break**。)

---

## 範圍與保留

- **僅測過 HGT**,本次未測 GAT。GAT 是「launch-bound」案例(理論上 compile 最可能幫上忙),但同樣有 hetero + 自訂 op + 動態形狀的結構,預期結果相近或仍邊際——**此為推測,未驗證**。
- **sm_121 特有**:Triton 對 sm_121 有已知未修復的 ptxas 問題(外部 issue triton#9181);本次並非直接死因(已用 `suppress_errors` 退回 eager 繞過),但屬於整體工具鏈不利的一環。

---

## 決策

- **封存**。開關保留(預設關),以便日後 PyTorch / PyG / Triton 對「sm_121 + 異質 + 動態形狀」的支援成熟後可再測。
- **不合入 `main`**(主線不含此功能,零風險)。

## 未來若要重測,照三步

1. `TORCH_LOGS="graph_breaks,recompiles"` — 看 break / recompile 數是否大幅下降。
2. `nsys` — 看 `cudaLaunchKernel` 次數是否較 eager 下降(對照本專案 GAT eager 的 724,500 次)。
3. 比對 **MRR / Hits** 與 eager,確認精度未受影響。
