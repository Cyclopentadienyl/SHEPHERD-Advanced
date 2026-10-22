# 醫療知識圖譜診斷引擎 - 跨平台部署完整指南

**版本**: v2.0  
**最後更新**: 2025-10-07  
**適用環境**: Windows x86 + Blackwell | ARM + Blackwell (DGX Spark)

---

> **⚠️ Phase C 更新通知 (2026-05)**
>
> 部署流程已遷移至 [uv](https://docs.astral.sh/uv/) 作為核心套件管理工具。
> **使用者應以 `deploy.sh` (Linux) / `deploy.cmd` (Windows) 為主要部署入口**。
>
> - PyTorch (torch/torchvision/torchaudio) 現由 uv 透過 `pyproject.toml`
>   中的 `[tool.uv.sources]` 管理，並鎖定於 `uv.lock`，跨平台版本完全一致。
> - PyG native ext (pyg-lib, torch-scatter/sparse/cluster) 仍由 deploy
>   腳本 post-sync 安裝，缺 wheel 時 graceful skip，fallback 到
>   `torch.scatter_reduce`。
> - 本文檔中內嵌的 `pip install` / `python -m venv` 範例（例如
>   下方「DGX Spark 設置腳本」段落）為原 v2.0 內容，**僅供原理理解參考**，
>   實際部署請執行 `deploy.{sh,cmd}`。
>
> 詳見：`pyproject.toml` 的 `[tool.uv]` 段、`uv.lock`、以及
> `data/external/README.md` 的「Currently Supported Data Sources」表格。

---

## 文檔目的 🎯

本指南提供：
1. **兩個環境的逐步部署指令**
2. **每個套件的兼容性分析與備用方案**
3. **完整的測試驗證流程**
4. **常見問題排查手冊**

---

# 第一部分：環境對比分析

## 硬體規格對比表

| 項目 | Windows 開發環境 | DGX Spark 邊緣環境 | 影響分析 |
|------|------------------|-------------------|----------|
| **處理器** |
| 架構 | x86-64 (Intel/AMD) | ARM v9.2 (Cortex-X925 + A725) | ⚠️ 指令集不同，部分套件需重編譯 |
| 核心數 | 變動（假設8-16核） | 20核 (10+10混合) | ✅ ARM優勢：更多核心 |
| 頻率 | ~3-5 GHz | ~3.0 GHz (X925) | ≈ 相當 |
| **GPU** |
| 型號 | NVIDIA Blackwell (獨立) | Blackwell (GB10 SoC集成) | ⚠️ 集成vs獨立影響API |
| CUDA核心 | 變動 | 6144 | ✅ 充足 |
| Tensor核心 | 5th Gen | 5th Gen | ✅ 相同 |
| VRAM | 16GB GDDR6 | - | ⚠️ **Windows限制** |
| **記憶體** |
| 系統RAM | 32GB+ | 128GB LPDDR5X-9400 | ✅ **ARM巨大優勢** |
| 架構 | 分離（CPU/GPU） | **統一記憶體** | ✅ ARM優勢：零拷貝 |
| 頻寬 | RAM: ~50 GB/s, VRAM: ~448 GB/s | **統一: ~301 GB/s** | ≈ 權衡 |
| **互聯** |
| CPU-GPU | PCIe Gen4/5 (~32 GB/s) | **NVLink-C2C (~600 GB/s)** | ✅ **ARM顯著優勢** |
| **功耗** |
| TDP | 200-400W (CPU+GPU) | **140W (整體)** | ✅ ARM效能優勢 |

### 關鍵洞察

#### ✅ **ARM (DGX Spark) 的優勢**
1. **128GB統一記憶體**: 可載入更大的知識圖譜，無需子圖採樣
2. **超高CPU-GPU頻寬**: NVLink-C2C比PCIe快18倍，零拷貝傳輸
3. **更多CPU核心**: 20核 vs 常見的8-16核
4. **低功耗**: 140W vs 200-400W

#### ⚠️ **ARM (DGX Spark) 的挑戰**
1. **套件生態**: 部分Python套件沒有ARM wheel
2. **VRAM隔離**: 沒有獨立VRAM，GPU與CPU共享128GB
3. **驅動成熟度**: ARM + Blackwell組合較新

#### ⚠️ **Windows x86 的限制**
1. **VRAM 容量限制**: 消費級 GPU 的 VRAM（通常 8-24 GB）小於 DGX Spark 的 96 GB 統一記憶體，較大的 KG（如完整 PrimeKG 130K 節點 / 4M 邊）需要子圖採樣或分批訓練；GNN 推理仍可運行。
2. **記憶體拷貝開銷**: CPU ↔ GPU 數據傳輸經 PCIe，比 DGX Spark 的 NVLink-C2C 慢。

---

# 第二部分：套件兼容性詳細分析

## 2.1 核心依賴矩陣

| 套件 | 版本 | Win x86 | ARM | 安裝難度 | 備註 |
|------|------|---------|-----|----------|------|
| **Python** | 3.12 | ✅ | ✅ | 簡單 | 兩者都支持 |
| **PyTorch** | 2.10.0 | ✅ | ✅ | 簡單 | 官方ARM+CUDA wheel |
| **CUDA Toolkit** | 13.0 | ✅ | ✅ | 簡單 | Blackwell要求 |
| **cuDNN** | 9.x | ✅ | ✅ | 簡單 | PyTorch包含 |
| **PyTorch Geometric** | 2.6+ | ✅ | ⚠️ | 中等 | ARM可能需從源碼編譯 |
| **pyg-lib** | 最新 | ✅ | ⚠️ | 中等 | 同上 |
| **torch-scatter** | 最新 | ✅ | ⚠️ | 中等 | 同上 |
| **torch-sparse** | 最新 | ✅ | ⚠️ | 中等 | 同上 |
| **FlashAttention-2** | 2.8+ | ✅ | ❌ | 困難 | **ARM不支持** |
| **xformers** | 0.0.27+ | ✅ | ⚠️ | 困難 | ARM需從源碼編譯 |
| **FAISS (GPU)** | 1.8+ | ✅ | ❌ | 中等 | **已棄用，改用cuVS** |
| **hnswlib** | 0.8+ | ✅ | ✅ | 簡單 | **已棄用，改用Voyager** |
| **Voyager** | 2.0+ | ✅ | ✅ | 簡單 | Spotify HNSW，跨平台 |
| **cuVS** | 24.12+ | ✅ | ✅ | 中等 | NVIDIA RAPIDS，Linux GPU |
| **transformers** | 4.40+ | ✅ | ✅ | 簡單 | 完全支持 |
| **owlready2** | 0.46+ | ✅ | ✅ | 簡單 | 純Python |
| **neo4j** | 5.0+ | ✅ | ✅ | 簡單 | 純Python驅動 |
| **fastapi** | 0.110+ | ✅ | ✅ | 簡單 | 純Python |
| **pandas/numpy** | 最新 | ✅ | ✅ | 簡單 | 完全支持 |

### 圖例
- ✅ **綠色**: 有預編譯wheel，直接pip安裝
- ⚠️ **黃色**: 可能需要從源碼編譯，但可行
- ❌ **紅色**: 不支持，需要備用方案

---

## 2.2 關鍵套件詳細分析

### 🔴 **PyTorch Geometric on ARM**

**現狀**:
- PyG 2.6+ 理論上支持 ARM，但預編譯wheel可能缺失
- 依賴的擴展庫（pyg-lib, torch-scatter等）在ARM上支持參差不齊

**安裝策略（優先級排序）**:

```bash
# 策略 1: 嘗試官方wheel (成功率: 60%)
pip install torch-geometric==2.6.0

# 策略 2: 嘗試從PyG倉庫安裝 (成功率: 80%)
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.10.0+cu130.html

# 策略 3: 從源碼編譯 (成功率: 95%, 耗時: 1-2h)
git clone https://github.com/pyg-team/pytorch_geometric.git
cd pytorch_geometric
pip install -e .

# 策略 4: 使用conda (成功率: 70%, 但不推薦ARM)
conda install pyg -c pyg

# 策略 5: 手動安裝依賴 (最可靠，耗時: 2-3h)
# 先安裝核心PyG
pip install torch-geometric

# 逐個編譯擴展
git clone https://github.com/pyg-team/pyg-lib.git
cd pyg-lib && pip install -e . && cd ..

git clone https://github.com/rusty1s/pytorch_scatter.git
cd pytorch_scatter && pip install -e . && cd ..
# 依此類推...
```

**驗證腳本**:
```python
# test_pyg_arm.py
import torch
import torch_geometric as pyg
from torch_geometric.data import Data

print(f"PyG version: {pyg.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# 創建測試圖
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
x = torch.randn(3, 16)
data = Data(x=x, edge_index=edge_index)

if torch.cuda.is_available():
    data = data.cuda()
    print("✅ PyG on ARM + CUDA: OK")
else:
    print("❌ CUDA not available")
```

**備用方案: 使用DGL**
```bash
# 如果PyG完全無法安裝
pip install dgl -f https://data.dgl.ai/wheels/torch-2.10/cu130/repo.html

# DGL on ARM 支持較好
```

### 🔴 **FlashAttention-2 on ARM**

**現狀**:
- ❌ **不支持 ARM 架構**
- 官方倉庫明確說明：僅支持 x86-64
- 從源碼編譯在 ARM 上會失敗或hang住

**解決方案: 三層降級**

```python
# src/models/attention/adaptive_backend.py

import platform
import torch
import torch.nn.functional as F
from typing import Optional

class AdaptiveAttentionBackend:
    """
    三層降級策略:
    1. FlashAttention-2 (僅x86)
    2. xformers Memory-Efficient Attention (ARM備選)
    3. PyTorch SDPA (最終備案)
    """
    
    def __init__(self):
        self.backend = self._detect_backend()
        print(f"[AttentionBackend] Using: {self.backend}")
    
    def _detect_backend(self) -> str:
        arch = platform.machine()
        is_arm = arch in ['aarch64', 'arm64', 'armv8l']
        
        # 第1層: FlashAttention-2 (僅x86)
        if not is_arm:
            try:
                import flash_attn
                if hasattr(torch.backends.cuda, 'flash_sdp_enabled'):
                    if torch.backends.cuda.flash_sdp_enabled():
                        return 'flash_attention_2'
            except ImportError:
                pass
        
        # 第2層: xformers (ARM備選)
        try:
            import xformers.ops
            return 'xformers_memory_efficient'
        except ImportError:
            pass
        
        # 第3層: PyTorch原生SDPA
        if hasattr(F, 'scaled_dot_product_attention'):
            return 'pytorch_sdpa'
        
        # 第4層: 手動實現（不應該到這）
        return 'manual_fallback'
    
    def compute(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0
    ) -> torch.Tensor:
        """
        統一的注意力計算接口
        
        Args:
            query: (B, H, L, D)
            key: (B, H, S, D)
            value: (B, H, S, D)
        """
        
        if self.backend == 'flash_attention_2':
            from flash_attn import flash_attn_func
            # FlashAttention-2需要 (B, L, H, D) 格式
            q = query.transpose(1, 2)  # (B, L, H, D)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)
            out = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=False)
            return out.transpose(1, 2)  # 轉回 (B, H, L, D)
        
        elif self.backend == 'xformers_memory_efficient':
            from xformers.ops import memory_efficient_attention
            # xformers需要 (B, L, H, D) 格式
            q = query.transpose(1, 2)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)
            out = memory_efficient_attention(
                q, k, v,
                attn_bias=attn_mask,
                p=dropout_p
            )
            return out.transpose(1, 2)
        
        elif self.backend == 'pytorch_sdpa':
            # PyTorch SDPA 接受 (B, H, L, D)
            return F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=False,
                enable_gqa=True  # Grouped Query Attention
            )
        
        else:  # manual_fallback
            return self._manual_attention(query, key, value, attn_mask, dropout_p)
    
    def _manual_attention(self, q, k, v, mask, dropout_p):
        """手動實現（最慢，但最穩定）"""
        scale = q.size(-1) ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if mask is not None:
            attn = attn + mask
        
        attn = F.softmax(attn, dim=-1)
        
        if dropout_p > 0 and self.training:
            attn = F.dropout(attn, p=dropout_p)
        
        return torch.matmul(attn, v)

# 全局單例
_attention_backend = None

def get_attention_backend():
    global _attention_backend
    if _attention_backend is None:
        _attention_backend = AdaptiveAttentionBackend()
    return _attention_backend
```

**效能基準測試**:
```python
# benchmark_attention.py
import torch
import time
from src.models.attention.adaptive_backend import get_attention_backend

backend = get_attention_backend()

B, H, L, D = 32, 8, 512, 64  # Batch, Heads, Length, Dim
device = 'cuda' if torch.cuda.is_available() else 'cpu'

q = torch.randn(B, H, L, D, device=device)
k = torch.randn(B, H, L, D, device=device)
v = torch.randn(B, H, L, D, device=device)

# 預熱
for _ in range(10):
    _ = backend.compute(q, k, v)

# 測試
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    out = backend.compute(q, k, v)
    torch.cuda.synchronize()
end = time.time()

print(f"Backend: {backend.backend}")
print(f"Avg time: {(end - start) / 100 * 1000:.2f} ms")
```

**預期效能（相對FlashAttention-2 @ x86）**:
- FlashAttention-2 (x86): **1.00x** (基準)
- xformers (ARM): **0.65-0.75x**
- PyTorch SDPA (ARM): **0.45-0.55x**
- Manual (ARM): **0.25-0.35x** (不推薦)

### 🟢 **Vector Index: Voyager + cuVS (v3.2)** — 已自診斷剝離

> **Note**: FAISS 和 hnswlib 已棄用。新架構使用 Voyager (跨平台) + cuVS (Linux GPU)。
>
> **狀態**: 此子系統**不在診斷路徑上**。它已依決策自推理管線剝離，實作與測試保留，
> 供規劃中的自然語言輸入／向量映射使用。部署時無需建立索引，建立了也不影響診斷結果。
> 背景見 `docs/RETRIEVAL_AND_CANDIDATE_DISCOVERY_FINDINGS.md`。

**後端選擇策略**:
- Linux (x86/ARM): cuVS (GPU) → Voyager (CPU fallback)
- Windows: Voyager only (cuVS 不支持 Windows)

**解決方案: Voyager + cuVS**

```python
# src/retrieval/vector_index.py (v3.2)
from src.retrieval import create_index, resolve_backend

# 自動選擇最佳後端
index = create_index(backend="auto", dim=768, metric="ip")

# 或明確指定後端
voyager_index = create_index(backend="voyager", dim=768)
cuvs_index = create_index(backend="cuvs", dim=768)  # Linux only

# 使用範例
embeddings = {"entity_1": vec1, "entity_2": vec2, ...}
index.build_index(embeddings)

results = index.search(query_vector, top_k=10)
# Returns: [("entity_id", score), ...]
```

**安裝方式**:
```bash
# Voyager (必裝，跨平台)
pip install voyager>=2.0

# cuVS (選裝，Linux GPU；須用 cu13 build 與專案 torch cu130 對齊)
pip install --extra-index-url https://pypi.nvidia.com cuvs-cu13
```

**效能對比**:
| 操作 | cuVS GPU (Linux) | Voyager CPU | 備註 |
|------|------------------|-------------|------|
| 構建索引 (100萬向量) | 3s | 30s | cuVS 使用 IVF-PQ |
| 查詢 (batch=100, k=10) | 1ms | 8ms | Voyager 使用 HNSW |
| 記憶體使用 | 低（GPU VRAM） | 中（CPU RAM） | - |
| 平台支持 | Linux only | 跨平台 | Windows 僅 Voyager |

---

# 第三部分：完整部署腳本

## 3.1 Windows x86 + Blackwell 環境

### 自動化安裝腳本

```powershell
# setup_windows.ps1
# Windows x86 + Blackwell GPU 環境設置腳本

param(
    [switch]$SkipCudaCheck,
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  醫療知識圖譜引擎 - Windows 環境設置  " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 1. 檢查CUDA
if (-not $SkipCudaCheck) {
    Write-Host "[1/8] 檢查 CUDA 環境..." -ForegroundColor Yellow
    try {
        $nvidiaSmi = nvidia-smi
        Write-Host "✅ NVIDIA GPU 檢測成功" -ForegroundColor Green
        
        # 檢查CUDA版本
        $cudaVersion = nvcc --version 2>&1 | Select-String -Pattern "release (\d+\.\d+)"
        if ($cudaVersion -match "12\.8") {
            Write-Host "✅ CUDA 12.8 已安裝" -ForegroundColor Green
        } else {
            Write-Warning "CUDA版本不是12.8，可能影響Blackwell支持"
        }
    }
    catch {
        Write-Error "❌ CUDA未檢測到，請先安裝CUDA 12.8"
        exit 1
    }
}

# 2. 檢查Python
Write-Host "[2/8] 檢查 Python 版本..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($pythonVersion -match "3\.12") {
    Write-Host "✅ Python 3.12 已安裝" -ForegroundColor Green
} else {
    Write-Error "❌ 需要 Python 3.12，當前版本: $pythonVersion"
    exit 1
}

# 3. 創建虛擬環境
Write-Host "[3/8] 創建虛擬環境..." -ForegroundColor Yellow
if (Test-Path ".venv") {
    Write-Host "⚠️  虛擬環境已存在，跳過創建" -ForegroundColor Yellow
} else {
    python -m venv .venv
    Write-Host "✅ 虛擬環境創建成功" -ForegroundColor Green
}

# 4. 啟動虛擬環境
Write-Host "[4/8] 啟動虛擬環境..." -ForegroundColor Yellow
.\.venv\Scripts\Activate.ps1

# 5. 升級pip
Write-Host "[5/8] 升級 pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel

# 6. 安裝PyTorch
Write-Host "[6/8] 安裝 PyTorch 2.10.0 + CUDA 13.0..." -ForegroundColor Yellow
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 `
    --index-url https://download.pytorch.org/whl/cu130

# 驗證PyTorch
python -c "import torch; assert torch.cuda.is_available(), 'CUDA不可用'; print(f'✅ PyTorch {torch.__version__} + CUDA {torch.version.cuda}')"

# 7. 安裝PyTorch Geometric
Write-Host "[7/8] 安裝 PyTorch Geometric..." -ForegroundColor Yellow
pip install torch-geometric pyg-lib torch-scatter torch-sparse torch-cluster `
    -f https://data.pyg.org/whl/torch-2.10.0+cu130.html

# 8. 安裝其他依賴
Write-Host "[8/8] 安裝其他依賴..." -ForegroundColor Yellow

# FlashAttention-2
Write-Host "  - 安裝 FlashAttention-2 (可能需要幾分鐘)..." -ForegroundColor Cyan
pip install flash-attn --no-build-isolation

# 其他套件
pip install `
    transformers>=4.40.0 `
    owlready2>=0.46 `
    voyager>=2.0 `
    fastapi>=0.110.0 `
    uvicorn>=0.29.0 `
    pandas>=2.2.0 `
    numpy>=2.0 `
    tqdm>=4.66.0 `
    pyyaml>=6.0 `
    tensorboard>=2.16.0

# 9. 最終驗證
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  環境驗證" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

python -c @"
import torch
import torch_geometric as pyg
import flash_attn
import voyager

print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ PyG: {pyg.__version__}')
print(f'✅ CUDA可用: {torch.cuda.is_available()}')
print(f'✅ CUDA設備: {torch.cuda.get_device_name(0)}')
print(f'✅ CUDA記憶體: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print(f'✅ FlashAttention: 已安裝')
print(f'✅ Voyager: 已安裝 (v{voyager.__version__})')
"@

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  ✅ Windows 環境設置完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "下一步："
Write-Host "  1. 下載資料: python scripts/download_data.py"
Write-Host "  2. 構建知識圖譜: python scripts/build_kg.py"
Write-Host "  3. 訓練模型: python scripts/train_model.py（須在已設定 GPU 鎖的視窗中執行，§4.11）"
```

### 手動安裝步驟（如果腳本失敗）

```powershell
# 步驟1: 創建環境
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 步驟2: PyTorch
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu130

# 步驟3: PyG (如果上面的安裝失敗)
# 方案A: 使用PyG倉庫
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.10.0+cu130.html

# 方案B: 從源碼
git clone https://github.com/pyg-team/pytorch_geometric.git
cd pytorch_geometric
pip install -e .

# 步驟4: FlashAttention-2
# 如果失敗，檢查是否安裝了Visual Studio Build Tools
pip install flash-attn --no-build-isolation

# 步驟5: 其他依賴
pip install -r requirements.txt
```

---

## 3.2 DGX Spark (ARM + Blackwell) 環境

### 自動化安裝腳本

```bash
#!/bin/bash
# setup_dgx_spark.sh
# ARM + Blackwell (DGX Spark) 環境設置腳本

set -e  # 遇到錯誤立即退出

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}========================================"
echo -e "  醫療知識圖譜引擎 - DGX Spark 環境設置"
echo -e "========================================${NC}"
echo ""

# 1. 檢查架構
echo -e "${YELLOW}[1/10] 檢查系統架構...${NC}"
ARCH=$(uname -m)
if [ "$ARCH" == "aarch64" ]; then
    echo -e "${GREEN}✅ 確認 ARM64 架構${NC}"
else
    echo -e "${RED}❌ 錯誤: 不是ARM64架構 (檢測到: $ARCH)${NC}"
    exit 1
fi

# 2. 檢查GPU
echo -e "${YELLOW}[2/10] 檢查 GPU 環境...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo -e "${GREEN}✅ NVIDIA GPU 檢測成功${NC}"
else
    echo -e "${RED}❌ 錯誤: nvidia-smi 未找到${NC}"
    exit 1
fi

# 3. 檢查CUDA
echo -e "${YELLOW}[3/10] 檢查 CUDA 版本...${NC}"
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo -e "${GREEN}✅ CUDA 版本: $CUDA_VERSION${NC}"
    
    if [[ ! "$CUDA_VERSION" == 12.8* ]]; then
        echo -e "${YELLOW}⚠️  警告: CUDA版本不是12.8，可能影響Blackwell支持${NC}"
    fi
else
    echo -e "${RED}❌ 錯誤: CUDA未安裝${NC}"
    exit 1
fi

# 4. 檢查Python
echo -e "${YELLOW}[4/10] 檢查 Python 版本...${NC}"
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
if [[ "$PYTHON_VERSION" == 3.12* ]]; then
    echo -e "${GREEN}✅ Python 版本: $PYTHON_VERSION${NC}"
else
    echo -e "${RED}❌ 錯誤: 需要 Python 3.12，當前: $PYTHON_VERSION${NC}"
    exit 1
fi

# 5. 創建虛擬環境
echo -e "${YELLOW}[5/10] 創建虛擬環境...${NC}"
if [ -d ".venv" ]; then
    echo -e "${YELLOW}⚠️  虛擬環境已存在，跳過創建${NC}"
else
    python3 -m venv .venv
    echo -e "${GREEN}✅ 虛擬環境創建成功${NC}"
fi

# 6. 啟動虛擬環境
echo -e "${YELLOW}[6/10] 啟動虛擬環境...${NC}"
source .venv/bin/activate

# 7. 升級pip
echo -e "${YELLOW}[7/10] 升級 pip...${NC}"
pip install --upgrade pip setuptools wheel

# 8. 安裝PyTorch (ARM + CUDA)
echo -e "${YELLOW}[8/10] 安裝 PyTorch 2.10.0 (ARM + CUDA 13.0)...${NC}"
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu130

# 驗證PyTorch
python -c "import torch; assert torch.cuda.is_available(), 'CUDA不可用'; print(f'✅ PyTorch {torch.__version__} + CUDA OK')"

# 9. 安裝PyTorch Geometric (ARM)
echo -e "${YELLOW}[9/10] 安裝 PyTorch Geometric (可能需要從源碼編譯)...${NC}"

# 嘗試方案A: 直接安裝
echo -e "${CYAN}  嘗試方案A: pip安裝...${NC}"
if pip install torch-geometric pyg-lib torch-scatter torch-sparse 2>/dev/null; then
    echo -e "${GREEN}✅ PyG安裝成功 (使用wheel)${NC}"
else
    echo -e "${YELLOW}⚠️  方案A失敗，嘗試方案B: 從源碼編譯${NC}"
    
    # 方案B: 從源碼
    git clone https://github.com/pyg-team/pytorch_geometric.git /tmp/pyg
    cd /tmp/pyg
    pip install -e .
    cd -
    
    echo -e "${GREEN}✅ PyG安裝成功 (從源碼)${NC}"
fi

# 10. 安裝其他依賴
echo -e "${YELLOW}[10/10] 安裝其他依賴...${NC}"

# 嘗試FlashAttention-2 (預期失敗)
echo -e "${CYAN}  嘗試安裝 FlashAttention-2 (可能失敗，這是正常的)...${NC}"
if pip install flash-attn --no-build-isolation 2>/dev/null; then
    echo -e "${GREEN}✅ FlashAttention-2 安裝成功 (意外！)${NC}"
else
    echo -e "${YELLOW}⚠️  FlashAttention-2 不可用 (預期中)，將使用降級方案${NC}"
fi

# 嘗試xformers (備用)
echo -e "${CYAN}  嘗試安裝 xformers (備用注意力)...${NC}"
if pip install xformers 2>/dev/null; then
    echo -e "${GREEN}✅ xformers 安裝成功${NC}"
else
    echo -e "${YELLOW}⚠️  xformers 也不可用，將使用 PyTorch SDPA${NC}"
fi

# 其他純Python套件
echo -e "${CYAN}  安裝其他依賴...${NC}"
pip install \
    transformers>=4.40.0 \
    owlready2>=0.46 \
    voyager>=2.0 \
    fastapi>=0.110.0 \
    uvicorn>=0.29.0 \
    pandas>=2.2.0 \
    numpy>=2.0 \
    tqdm>=4.66.0 \
    pyyaml>=6.0 \
    tensorboard>=2.16.0

# cuVS for GPU acceleration (optional)
echo -e "${CYAN}  嘗試安裝 cuVS (GPU 加速)...${NC}"
pip install --extra-index-url https://pypi.nvidia.com cuvs-cu13 || \
    echo -e "${YELLOW}  cuVS 安裝失敗，將使用 Voyager (CPU)${NC}"

# 11. 環境驗證
echo ""
echo -e "${CYAN}========================================"
echo -e "  環境驗證"
echo -e "========================================${NC}"

python -c """
import platform
import torch
import torch_geometric as pyg

print(f'✅ 架構: {platform.machine()}')
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ PyG: {pyg.__version__}')
print(f'✅ CUDA可用: {torch.cuda.is_available()}')
print(f'✅ CUDA設備: {torch.cuda.get_device_name(0)}')
print(f'✅ 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

# 檢測注意力後端
try:
    import flash_attn
    print('✅ FlashAttention-2: 可用')
except ImportError:
    try:
        import xformers
        print('⚠️  FlashAttention-2: 不可用，使用 xformers')
    except ImportError:
        print('⚠️  FlashAttention-2: 不可用，使用 PyTorch SDPA')
"""

# 12. 平台配置
echo ""
echo -e "${CYAN}平台配置由 configs/deployment.yaml 集中管理；${NC}"
echo -e "${CYAN}硬體偵測（GPU / arch / CUDA）由 deploy.sh 與 scripts/launch/shep_launch.py 啟動時自動完成，無需手動寫入 platform 配置檔。${NC}"

echo ""
echo -e "${GREEN}========================================"
echo -e "  ✅ DGX Spark 環境設置完成！"
echo -e "========================================${NC}"
echo ""
echo -e "下一步："
echo -e "  1. 構建知識圖譜: python scripts/build_knowledge_graph.py"
echo -e "  2. 計算最短路徑: python scripts/compute_shortest_paths.py"
echo -e "  3. 訓練模型:     python scripts/train_model.py  (可選 --config <hyperparameters.yaml>；須在已設定 GPU 鎖的 shell 中執行，§4.11)"
echo -e "  4. 啟動系統:     ./launch_shepherd.sh"
echo ""
echo -e "${YELLOW}注意事項：${NC}"
echo -e "  - FlashAttention-2 在 ARM 上不可用，已自動降級"
echo -e "  - Vector Index 已自診斷管線剝離，不影響診斷；保留供未來向量映射使用"
echo -e "  - 已針對128GB記憶體優化配置"
```

### 手動排查步驟（如果腳本失敗）

```bash
# 步驟1: 基礎檢查
uname -m  # 應該是 aarch64
nvidia-smi  # 應該顯示GPU
python3 --version  # 應該是 3.12.x

# 步驟2: 測試PyTorch
python3 << 'EOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    # 測試張量運算
    x = torch.randn(100, 100).cuda()
    y = torch.matmul(x, x)
    print("✅ GPU計算正常")
EOF

# 步驟3: 測試PyG
python3 << 'EOF'
import torch
from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
x = torch.randn(2, 16)
data = Data(x=x, edge_index=edge_index)

if torch.cuda.is_available():
    data = data.cuda()
    print("✅ PyG on ARM + CUDA: OK")
EOF

# 步驟4: 測試注意力後端
python3 << 'EOF'
from src.models.attention.adaptive_backend import get_attention_backend
backend = get_attention_backend()
print(f"Using: {backend.backend}")
EOF
```

---

# 第四部分：故障排查手冊

## 4.1 Windows 常見問題

### 問題 1: FlashAttention-2 安裝失敗

**錯誤信息**:
```
error: Microsoft Visual C++ 14.0 or greater is required
```

**解決方案**:
1. 安裝 Visual Studio 2022 Build Tools
   - 下載: https://visualstudio.microsoft.com/downloads/
   - 選擇「使用C++的桌面開發」

2. 或者使用預編譯wheel（如果可用）

3. 最終方案：跳過FlashAttention，使用PyTorch SDPA

### 問題 2: CUDA Out of Memory (16GB VRAM限制)

**錯誤信息**:
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**解決方案**:
```python
# 1. 減小batch size
batch_size = 8  # 從32降到8

# 2. 使用梯度累積
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 3. 使用子圖採樣
subgraph = sample_subgraph(full_graph, num_nodes=50000)

# 4. 啟用梯度檢查點
from torch.utils.checkpoint import checkpoint
output = checkpoint(model_layer, input)
```

### 問題 3: PyG 安裝版本不匹配

**錯誤信息**:
```
undefined symbol: _ZN...
```

**解決方案**:
```bash
# 完全卸載並重裝
pip uninstall torch torch-geometric pyg-lib torch-scatter torch-sparse -y
pip cache purge

# 按順序重新安裝
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu130
pip install torch-geometric pyg-lib torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.10.0+cu130.html
```

---

## 4.2 ARM (DGX Spark) 常見問題

### 問題 1: PyG 在 ARM 上編譯失敗

**錯誤信息**:
```
error: command 'gcc' failed with exit status 1
```

**解決方案**:
```bash
# 1. 安裝編譯工具
sudo apt-get update
sudo apt-get install -y build-essential python3-dev

# 2. 設置編譯選項
export MAX_JOBS=10  # 使用10個核心
export TORCH_CUDA_ARCH_LIST="9.0"  # Blackwell架構

# 3. 逐個編譯
git clone https://github.com/rusty1s/pytorch_scatter.git
cd pytorch_scatter
python setup.py install
cd ..

# 重複其他擴展...
```

### 問題 2: 統一記憶體未正確使用

**症狀**: GPU和CPU似乎在複製數據

**驗證**:
```python
import torch

# 檢查是否真的是統一記憶體
x = torch.randn(1000, 1000)
x_gpu = x.cuda()

# 在統一記憶體架構下，這應該幾乎是瞬間的
import time
start = time.time()
y = x_gpu.cpu()
print(f"Transfer time: {time.time() - start:.4f}s")
# 應該 < 0.001s

# 如果很慢，可能CUDA沒有正確識別統一記憶體
```

**解決方案**:
```python
# 在代碼中明確使用統一記憶體
torch.cuda.set_device(0)

# 檢查設備屬性
props = torch.cuda.get_device_properties(0)
print(f"Unified memory: {props.unified_addressing}")  # 應該是True
```

### 問題 3: 注意力計算異常慢

**排查**:
```bash
# 運行benchmark
python benchmark_attention.py

# 預期輸出（相對時間）:
# FlashAttention-2: 不可用
# xformers: ~15ms (如果安裝成功)
# PyTorch SDPA: ~25ms
# Manual: ~80ms (不應該用到這個)

# 如果實際時間遠高於此，檢查：
nvidia-smi  # GPU利用率應該>80%
```

---

## 4.3 最短路徑表與 hop bound 設定

工作區裡的 `shortest_paths.pt` 是預先計算好的距離表，`shortest_paths.meta.json`
是它的 **sidecar**，記錄這張表是用什麼參數生成的——其中 `max_hops` 是最關鍵的一
項。**兩個檔案屬於同一次 build，複製或搬移工作區時必須一起帶走。**

### 為什麼 sidecar 不能省

`max_hops` 決定「不可達」的代價值（`max_hops + 1`）。所有最短路徑分數都是相對這
個值計算的，而它**無法從距離表本身還原**：BFS 在 `max_hops` 處停止展開，所以表內
記錄的最大距離只是 `max_hops` 的**下界**——一張 3-hop 的表與一張「上限 5 但剛好
沒有任何配對走到 5」的表，在張量層面完全相同。

用錯的天花板評分不只是數字偏移，**它會改變候選排序**。同一張 3-hop 表、兩個候選：

| 使用的 max_hops | A 分數 | B 分數 | 排序 |
|---|---:|---:|---|
| 正確的 3 | 0.4357142857 | 0.4250000000 | A > B |
| 誤用 5 | 0.4166666667 | 0.4250000000 | **B > A** |

因此系統**不會替你猜**。

### 優先順序

**距離表只會連同發佈時的 sidecar 一起上線。** 自 contract M2.5 起（2026-10-13），
sidecar 必須帶有這張表的來源綁定（`schema_version`、`build_id`、`kg_digest`），而
它的 `max_hops` 是 hop bound 唯一的來源。沒有 sidecar、或 sidecar 寫於配對協定之前
的距離表，會在載入時被拒絕。

1. **優先找回與距離表一起發佈的 sidecar**，之後兩個檔案一起保存、搬移。
2. **否則為這個工作區重建這一對**：執行 `scripts/compute_shortest_paths.py`，它會把
   距離表與 sidecar 一起發佈，並綁定到這個工作區的圖。**不要手寫 sidecar，也不要用
   表內最大距離回推 `max_hops`**——那個數字證明不了天花板。
3. 在那之前，要嘛保留正在服務的部署，要嘛改以不含最短路徑的方式服務：把
   `shortest_paths.pt` **與** `shortest_paths.meta.json` 成對移出工作區並保留，再重新
   載入。只留下 sidecar 會被當成「一對中的一邊」而拒絕。評分模式會因此改變。

原本用來替缺 sidecar 的表補上 bound 的 `SHEPHERD_SP_HOP_BOUND`，已隨 M2.5 **退役**：
這類表在讀取任何 bound 之前就會被拒絕，因此這個變數不再被讀取。請從服務環境中移除它。

### 表與圖對不上的時候

最短路徑表是**成對發布**的——`shortest_paths.pt` 與 `shortest_paths.meta.json`
兩個檔案記錄同一個 `build_id`，另外還記錄了距離是從哪個圖算出來的 SHA-256。由此
產生三種拒絕，而且三種都是部署問題，不是資料問題：

| 日誌訊息 | 發生了什麼 | 怎麼處理 |
|---|---|---|
| *"published by different runs"* | 兩個檔案來自不同次建置。發布過程中斷，或其中一個被別處的檔案蓋過 | 對這個工作區重跑 `scripts/compute_shortest_paths.py`。**不要**從別處複製其中一個檔案去湊成對 |
| *"computed from a different graph"* | 這張表屬於另一個工作區的 `kg.json`，它的節點索引指向的是別的節點 | 改用**這個**工作區建出來的表，或在這裡重新建 |
| *"One side of a pair is not a pair"* | 一個檔案宣告了協定、另一個沒有——表發布了但 sidecar 沒跟上，或反過來 | 重跑生成腳本;這一對要嘛一起發布,要嘛都不發布 |

**在這個機制之前建的表會被拒絕**（自 contract M2.5 起，2026-10-13；以前會帶著警告
載入並回報為 `unrecorded`）。它沒有綁定資訊，沒有任何紀錄說明它的節點索引屬於哪個
圖。請用 `scripts/compute_shortest_paths.py` 重新建置，或移除它、改以不含最短路徑的
方式服務。

`GET /api/v1/pipeline/status` 會回報 `sp_kg_binding`：表確實指向正在服務的那個圖
時為 `verified`。**只要最短路徑是開啟的，它就不會是空的。**

**用記憶體中的圖（而非工作區）建立管線時**根本無法檢查綁定，此時宣告了綁定的表會
被**拒絕**而不是被假定成立；未帶綁定的表則如上所述同樣被拒絕。請改用工作區建立。


### 缺少、未綁定或不合法的 sidecar

距離表存在，而它的 sidecar **缺少、未帶綁定、或不合法**（不是 JSON 物件、`max_hops`
不是整數或超出 1..127、或低於表內距離）時，候選 pipeline 會被**拒絕**，而不是退回
猜測：

- 服務**尚無**可用 pipeline（冷啟動）→ `/api/v1/diagnose` 回 **503**，不會回傳任何
  候選疾病。
- 服務**已有**正在運行的 pipeline → 該次 reload 被拒絕，**舊 pipeline 繼續服務**。

### 完全沒有距離表時

沒有 `shortest_paths.pt`（也沒有 sidecar）時，在預設配置（GNN 已載入、
`sp_optional=True`）下**服務不會停止**，而是改為純 GNN 評分：`sp_ready=False`、
`scoring_mode=gnn_only`、eta 實際為 1.0。**分數與排序會與啟用最短路徑時不同。** 每個
結果的 served identity（§4.7）也會如實標明：最短路徑 `absent`、評分模式 `gnn_only`、eta
`1.0`；Diagnosis 分頁與其報告會顯示 `NO SHORTEST-PATH TABLE: GNN score alone.` 橫幅。

> **例外：** 若部署刻意設定了 `sp_optional=False`，SP 不可用時**不會建出 pipeline**：
> 建置被拒絕（`PipelineBuildError`），啟動時不發布任何 pipeline、`/diagnose` 回 503；
> reload 則被拒絕，原本的 pipeline 繼續服務。這不是「一律退回純 GNN」。目前 API 的
> 建置路徑使用預設值 `sp_optional=True`。

### 狀態檢查

```bash
curl -s http://localhost:8000/api/v1/pipeline/status
```

| 欄位 | 預期 |
|---|---|
| `sp_ready` | 啟用時 `true`；沒有距離表時 `false` |
| `sp_max_hops` | 實際生效的天花板，讀自 sidecar |
| `sp_hop_bound_source` | `sidecar`（設定值退役後唯一的來源） |
| `sp_kg_binding` | `verified` |
| `scoring_mode` | 啟用時 `gnn_plus_shortest_path`；SP 關閉且 GNN 正常時 `gnn_only` |
| `served_identity` | 目前載入的 pipeline 服務的是什麼（§4.7）。啟用時：`shortest_paths.state` 為 `verified`，帶 sidecar 的 `build_id` 與 `max_hops`，`scoring_mode` 為 `gnn_plus_shortest_path`。沒有距離表時：`shortest_paths.state` 為 `absent`、`gnn_only`、`settings.eta` 為 `1.0` |

---

## 4.4 建圖來源紀錄（kg.provenance.json）

每次建圖都會在 `kg.json` 旁產生 `kg.provenance.json`，記錄這個圖是**從哪些檔案**
建出來的。

### 為什麼需要它

`kg.json` 一直有 digest，但 digest 只說「是哪些位元組」。兩台機器的本體版本若因部署
日期而不同，它們的 `kg_digest` 會不一樣——**偵測得到，卻說不出差在哪**。

### 紀錄內容

| 欄位 | 意義 |
|---|---|
| `kg_digest` | 同一次建圖寫出的 `kg.json` 的 SHA-256。**這是紀錄與圖的綁定** |
| `origin` | `files`（真實建圖）或 `synthetic`（demo／測試用的記憶體圖） |
| `sources` | 每個輸入的角色、檔名、內容 digest、宣告版本 |
| `missing_roles` | `files` 時為來源清單缺少的角色——在管線接受的紀錄裡永遠為空；`synthetic` 時依定義列出全部角色 |
| `counters` | 解析統計，例如 `rows_skipped_unresolved_disease_id` |
| `incomplete_by_design` | **這份紀錄刻意不涵蓋的東西** |

`sources` 的四個角色是 `mondo`、`hpo`、`phenotype_hpoa`、`genes_to_phenotype`，
**真實建圖四者缺一不可**：缺任一註解檔 builder 就停止，兩個本體一定會載入，而對已解析
卻無法識別的輸入，builder 會停止而不是把它記成缺少。**檔名只是線索，digest 才是身分**
——同一個目錄名下的檔案可以被原地替換。

`declared_version` 是本體檔頭的原始 `data-version`，**沒有就是 `null`**，不會用
OBO 的 format version 冒充 release。

`counters` 的命名要照字面讀：`rows_skipped_unresolved_disease_id` 是
**`phenotype.hpoa` 在那一個解析階段跳過的列數**，其中包含刻意不支援的 ID 類型
（如 DECIPHER），**不是「遺失的疾病數」，也不是版本不符的總量**。builder 另外還會
因節點不存在而丟邊，那些不在這個數字裡。

### 這份紀錄不宣稱什麼

`incomplete_by_design` 直接寫在檔案裡：本體解析時解析的 imports、parser 與 builder
的版本、建圖參數，**都不在紀錄範圍內**。它是**來源檔清單，不是重建配方**。

### 必要條件：沒有它，訓練、serving 與量測一律拒絕

**自 contract M3b 起，這份紀錄是必要的。** manifest 綁定它的 digest（`split_manifest.json`
裡的 `artifacts.provenance`），訓練、serving（以 workspace 上的模型服務時）與量測都在各自
的圖檢查之後檢查它，且只讀一次：檔案位元組必須正是 manifest 綁定的那份，`kg_digest` 必須
指向剛驗證過的那個圖。其他情況一律拒絕，而且發生在建立 run 目錄、讀取 checkpoint 或建
模型之前。拒絕訊息會指明**綁定狀態**：

| 綁定狀態 | 意義 |
|---|---|
| `verified` | 正是 manifest 綁定、且指向本 workspace 圖的紀錄。唯一被接受的狀態 |
| `mismatch` | 檔案不是 manifest 綁定的位元組，或描述的是另一個圖——例如兩個 workspace 在搬移時互換了紀錄，每個檔案都合法，只有配對錯了 |
| `unrecorded` | manifest 沒有綁定任何紀錄 |
| `unverifiable` | manifest 綁定了紀錄，但檔案不在、讀不到，或不是本版本能讀的紀錄 |

已綁定、來源為 `files`、卻缺少四個角色之一的紀錄同樣拒絕，此時綁定狀態為 `verified`：
這表示有必要輸入未被識別。

**來源（origin）是另一項事實，與綁定狀態並列顯示，絕不取代它。** `files` 表示圖由真實
輸入建成；`synthetic` 表示由 demo、測試或部署探針在記憶體中建成。合成 workspace 同樣
完整綁定、通過所有檢查，且在顯示紀錄的每個地方都標明為合成：

- 訓練日誌（`Dataset record kg.provenance.json: binding verified, origin synthetic`）；
- serving 的 pipeline 設定與 API 的 pipeline 狀態（`dataset_record`：綁定狀態、來源、紀錄
  的 digest、來源檔清單與 `incomplete_by_design`）；
- 量測報告的 manifest（`dataset_record`）與主控台摘要；
- 盤點工具（`scripts/audit_checkpoint_family.py --inventory`），以兩個欄位分別列出紀錄的
  判定、綁定狀態與來源；
- 自 contract M3c 起，served identity（§4.7）：`dataset_record` 為 `verified`、來源
  `synthetic`，出現在每一個診斷回應與 API 的 pipeline 狀態裡；
- Diagnosis 頁面：結果上方的 `SYNTHETIC WORKSPACE: not a clinical result.` 橫幅、
  「Model in use」卡片上琥珀色的 `synthetic data` 標籤（其 Details 裡也有同一個橫幅），以及
  結果身分（「This result」）中 dataset record 那一列的 `verified · synthetic`（自 M5-2 起，
  §4.14）；
- 可下載的報告：「What produced this result」段落裡的同一個橫幅，以及檔名
  `diagnosis_SYNTHETIC_<patient>_<session>_report.md`；
- CSV 匯出：每一列的 `served_dataset_origin` 都是 `synthetic`，檔名為
  `diagnosis_SYNTHETIC_<patient>_<session>.csv`。

紀錄自述不涵蓋的部分（`incomplete_by_design`）只顯示、不拒絕。沒有模型的 serving（只用
`kg.json` 做路徑推理）不驗證 workspace，不受影響；它的 served identity 也如實這樣標明
（checkpoint `absent`、圖 `unverified`、dataset record `absent`；見 §4.7「沒有模型的
serving」）。pipeline reload 把 `kg.provenance.json` 列為必要檔案。

### 舊 workspace

**不支援。** manifest 未綁定紀錄的 workspace 會以 `unrecorded` 拒絕，紀錄遺失的會以
`unverifiable` 拒絕，沒有相容模式。請用 `scripts/build_knowledge_graph.py --generate-samples`
重建。部署前可先跑唯讀的盤點工具，列出哪些 workspace 受影響。

> **沿革**：contract M3b 之前，這份紀錄只回報、不強制：狀態讀取函數
> （`workspace_provenance_status()`，已移除）只回傳狀態，沒有任何程式據此拒絕。更早的版本
> 曾讓圖驗證在載入張量前因紀錄損壞而拋例外，當時已撤回。擁有者的決定為正式管線推翻了
> 這個政策；圖檢查與紀錄檢查仍是兩個獨立的檢查，各有各的狀態。

### 與本體版本的關係

目前 `latest` 是**快取預設值，不是新鮮度保證**——它的意思是「用快取裡那份；沒有才
下載」，不會每次去確認遠端是否有更新版本。

## 4.5 指定建圖要用哪一份本體

**直接指名檔案。** `--mondo-path` 與 `--hpo-path` 接受路徑，優先於其他所有來源。
路徑不存在時**拒絕**，不會安靜地改開快取裡的檔案——所以建置不可能在你沒指定的檔案
上成功。

沒有給路徑時，會依序搜尋 `configs/deployment.yaml` 的 `paths.ontology_roots`，
以及 `--ontology-cache-dir`：

| 找到的情況 | 行為 |
|---|---|
| 剛好一個候選 | 使用它，並記錄其 `data-version` 與 digest |
| **超過一個** | **拒絕**，列出每一個的路徑、`data-version`、digest 與宣告的 imports |
| 沒有候選、有設定來源 | 下載、檢查，再放進快取（見下方） |
| 沒有候選、也沒有來源 | 拒絕，列出搜尋過的目錄，並說明沒有嘗試任何下載 |

**發行產品就是它所屬的本體**：`mondo-simple.obo`、`mondo-base.obo`、`hp-base.obo`、
`hp-international.obo` 與 `hp-simple-non-classified.obo`，和 `mondo.obo`、`hp.obo`
一樣是 MONDO／HPO 的候選。所以同一個 root 裡有 `mondo.obo` 與 `mondo-simple.obo`
時就是兩個 MONDO 候選，會拒絕直到指名其一。

檔名以 `.` 開頭的檔案永遠不是候選。檔名像某個本體、內容卻宣告另一個本體的檔案會被
略過，並**發出警告**指出兩者。

**不做任何排序。** 不依 root 順序、不依檔案修改時間，也不依「`data-version` 最新
者」。同時保留多個版本是支援的——指名路徑正是為此而設——但建置不會替你挑一個。

**這改變了一項既有行為。** 快取目錄同時有 `mondo.obo` 與 `mondo.owl` 時，以往會
安靜地採用 `.obo`；現在會拒絕並要求 `--mondo-path`。下載從 OBO 退回 OWL 時就會
造成兩個檔案並存，所以這不是罕見狀態。解法是加上路徑參數，不需要刪除任何檔案。

**`--force-download` 與路徑參數同時給出會被拒絕**——指名檔案和要求重新下載是兩個
不同的指令。

### 下載如何落地

- **不是這個本體的檔案，絕不會被覆寫。** 下載要寫入的快取路徑（例如
  `<cache>/hpo.obo`）若已有一個不是 HPO 候選的檔案，建置會在抓取任何東西之前拒絕，
  並指出該檔案。請把它移走，或用它所屬插槽的路徑參數指名它。
- **先檢查，再保留。** 下載先落在目的地旁、這次下載專屬的隱藏檔
  （`.hpo.obo.<隨機>.staged`），用和其他輸入相同的 imports 與插槽檢查開啟。只有通過
  的檔案會取代快取中的舊檔；未通過的會被刪除，舊檔不受影響。`--force-download` 也
  一樣；兩個建置同時下載到同一個快取時，各自只檢查並發布自己的下載，也各自記錄自己
  實際讀到的那個檔案的 digest。
- **下載失敗會回報，絕不替換。** 沒有任何來源成功時，建置會停下並給出手動下載的
  方式，不會退回去使用剛好在磁碟上的檔案。

本節所有拒絕都以訊息（而非 traceback）輸出，建置以狀態碼 **2** 結束。

### 兩種新的拒絕

| 拒絕訊息 | 意義 |
|---|---|
| declares N import(s) and this project requires a self-contained ontology file | 檔案宣告要引入其他本體。這些**不會**被抓取：解析它們會帶進這個檔案 digest 涵蓋不到的內容，而忽略它們會建出缺少那些內容、外表卻完整的圖。請改用未宣告 imports 的檔案 |
| declares itself to be X, and it is being used as the Y input／carries no `HP:` terms | 檔案不是該插槽要的本體。放著不管會建出**零個**該類型節點，而 provenance 記錄看起來完整 |

### 下載 URL 放在哪裡

`configs/deployment.yaml` 的 `ontology.sources`。它們以往是 `src/ontology/
loader.py` 裡的常數，PURL 失效就得改程式碼；現在改設定即可。不設這個鍵就使用專案
內建的預設：四個本體各自的 OBO Foundry PURL，並以 OWL PURL 作為第二次嘗試。列在
`hp` 下的來源就是 HPO 的來源。

有兩條規則套用在**每一次請求**上，包含每一次轉址——而這裡的每個來源都是 PURL，
PURL **就是**轉址：

- **協定：** 只允許 `http` 與 `https`。
- **目的地：** 主機解析出的每一個位址都必須是全球可路由的。院內網路的主機會被拒絕，
  除非列在 `ontology.allowed_hosts` 裡——這就是啟用院內鏡像的方式，出於明確設定而
  非意外。請填**純主機名稱**（`mirror.hospital.example`）；帶有協定、路徑、連接埠或
  `@` 的項目永遠不會匹配，讀取設定時就會被拒絕。
- **透過 proxy 時：** 若本機無法自行解析某個名稱，且該請求會送往
  `http_proxy`／`https_proxy` 設定的 proxy（主機不在 `no_proxy` 內；項目可以帶
  連接埠，例如 `mirror.example:8080`，比對方式與 Python 的 `urllib` 完全相同），名稱就交給
  proxy 解析——在只能經由 proxy 對外的網路裡，外部名稱在本機本來就沒有答案。proxy
  之後連到哪裡，屬於 proxy 的政策。在本機**能**解析且落在院內位址的名稱，不論有無
  proxy 都會被拒絕。沒有 proxy 時，無法解析的名稱會回報為下載失敗，而不是政策拒絕。

**尚未交付：** 從介面上編輯這些 URL。需求已記錄，後端也是為它而建——單一設定模型、
單一下載與驗證入口，供 UI 與 CLI 共用——但介面本身不在本專案範圍內。在它出現之前，
編輯設定檔就是做法。

**更改來源不等於切換正在服務的圖。** 編輯 URL、下載檔案、建置 workspace 是三件不同
的事，只有新的建置會產出新的 workspace；把它投入服務是既有的載入／發布步驟，未變。

### 在這台機器上確認下載

測試以受控的回應涵蓋下載規則；這台機器的網路、proxy 與憑證能否讓真實下載通過，只有
機器本身能證明。部署探測腳本為此提供一個需明確開啟的階段：

```bash
python scripts/probe_deployment.py --work-dir /tmp/shepherd_probe \
    --report probe_report.json --download-only
```

它會從 `configs/deployment.yaml` 的來源**完整**下載 MONDO 與 HPO（數十 MB），走的是
和建置相同的路徑，下載到工作目錄——絕不寫入你的快取或設定的 root，並在結束後檢查這
一點。改用 `--download` 則是把這個階段加進完整的探測。不需要 GPU；`--download-only`
也完全不在 GPU 上執行任何東西：它只描述軟體環境、不檢查裝置，所以 GPU 的問題不會讓
下載檢查失敗。

報告中，`G1`（MONDO）與 `G2`（HPO）各有一個 `outcome`：

| `outcome` | 意義 |
|---|---|
| `published` | 已下載、通過 imports 與插槽檢查，並以驗證過的 bytes 發布。`source_digest` 是收到檔案的精確身分；`data_version` 在符合標準格式（`releases/2026-09-01`、`hp/releases/…`）時才列出宣告的版本，其他任何宣告內容都會被保留不列（`data_version_withheld: true`），只留下其中的日期 |
| `refused_by_policy` | 每個來源都被協定或目的地規則拒絕——屬於設定決定（`ontology.sources`／`ontology.allowed_hosts`） |
| `transfer_failed` | 沒有任何來源成功。`attempts` 說明每個來源如何失敗：`host_unresolved`（本機無 DNS 答案、也沒有可交付的 proxy）、`truncated` 或 `transfer_failed` |
| `wrong_ontology`／`imports_declared` | 檔案已送達，但被插槽或 imports 檢查拒絕 |
| `no_source`／`configuration_invalid` | 沒有設定可下載的來源，或設定無法讀取 |

`network_at_probe_time` 逐一記錄每個來源：名稱能否在本機解析、請求是否經由 proxy、
主機是否在 allow list 中——針對設定的 URL 本身，不含它導向的轉址（`http://` 的 PURL
轉址到 `https://` 時，每一跳都會重新判斷、也各自決定是否經由 proxy）。來源以其在清單中的位置表示；報告不含任何主機、proxy 或目錄
名稱。`G9` 若發現你的快取或 root 有變動，只回報是哪個位置、幾個檔案，不列出檔名。

## 4.6 建圖與服務寫出的檔案權限

最短路徑表與 sidecar、split manifest、`kg.provenance.json`、evaluation ledger、UI
設定檔以及本體快取，都是先寫進暫存檔再換名到位，讀取的一方不會看到寫到一半的檔案。
它們發布時的權限，與一般 `open()` 寫出時相同：

- **新建的檔案**：依目錄的規則，也就是 umask（常見是 `022`，產生 `0644`）；目錄
  設有 default ACL 時則依 ACL。
- **覆寫的檔案保留原本的權限**：操作者設成 `600` 的仍是 `600`，群組共享的 `640`
  仍是 `640`。

2026 年 9 月起到這次修正之前，這些檔案發布時是 `0600`，只有寫入的帳號能讀。那段期間
寫出的檔案會維持原本的權限；如果服務用另一個帳號讀取，請對它們執行一次 `chmod 644`
（或在共享群組下用 `640`）。

**換名無法重現的部分。** 覆寫後的檔案其實是一個新檔：擁有者是寫入的帳號，群組是該帳號
的群組（目錄設有 setgid 時則是目錄的群組），舊檔上的延伸 ACL 項目不會被複製。如果服務
和建圖用的是不同帳號，請讓兩者共用一個群組並把 workspace 目錄設為 setgid，或在目錄上
設定 default ACL。讀取的帳號也需要檔案上層每一層目錄的搜尋（`x`）權限。

**寫入期間**，暫存檔就已經具有發布後的權限，所以能讀這個檔案的人，也能讀到寫到一半的
內容。另外，**這些目錄不可讓不信任的帳號寫入**：能在目錄裡改名的帳號，可以在檔案寫完
到換名之間把它換掉，這一點在這次修改之前就是如此。

## 4.7 診斷結果是由什麼產生的：served identity（自 contract M3c 起）

自 contract M3c 起（2026-10-16），每一個診斷結果都帶有 **served identity**：產生它的
checkpoint、圖、workspace manifest、最短路徑表與 dataset record，以 digest 或識別碼標明，
再加上評分模式與實際生效的 serving 設定——**每個部分都用文字寫出自己的狀態**。pipeline
是從建置時實際讀取的內容組出這份身分，產生結果時不會重新雜湊任何檔案，也不會用
「unknown」填空。

### 診斷回應的變更

`POST /api/v1/diagnose`：

- **移除 `model_version`**，沒有任何替代字串。不論服務的是什麼，它永遠是 `1.0.0`。
  有讀取它的用戶端必須停止讀取。
- **新增 `served_identity`**，而且一定存在。它複製自這個回應所依據的那個結果，絕不取自
  服務的共享狀態，因此請求進行中剛好發生的 reload 不會改掉它的標示。
- **只有 demo 回答時它才是 `null`。** 完全沒有設定任何 pipeline 時，服務仍會回傳示意用的
  候選與警告 `Pipeline not fully initialized - using mock data`；沒有任何東西被服務，所以
  也沒有可標明的身分。已設定 pipeline 卻建置失敗的部署回 503，絕不會給 demo 回答。

服務本身的版本（`/health`、`/` 與 Swagger 標題中的 `1.0.0`）是 API 的版本，不是所服務
內容的身分。

### 組成部分與狀態

```json
"served_identity": {
  "scoring_mode": "gnn_plus_shortest_path",
  "checkpoint": {"state": "verified", "sha256": "<所讀 checkpoint 的 SHA-256>"},
  "graph": {"state": "verified", "kg_sha256": "<kg.json 的 SHA-256>",
            "split_manifest_sha256": "<split_manifest.json 的 SHA-256>"},
  "shortest_paths": {"state": "verified", "build_id": "<sidecar 的 build_id>", "max_hops": 5},
  "dataset_record": {"state": "verified", "origin": "files",
                     "sha256": "<kg.provenance.json 的 SHA-256>"},
  "settings": {"device": "cuda:0", "device_requested": "auto", "eta": 0.7,
               "max_path_length": 4, "path_length_penalty": 0.9,
               "aggregation_method": "weighted_sum", "max_paths_per_candidate": 10,
               "use_direct_path_optimization": true,
               "phenotype_normalisation_version": 1}
}
```

| 部分 | 狀態 | 意義 | 值 |
|---|---|---|---|
| `checkpoint` | `verified` | 模型由一次讀取的 checkpoint 建成，且其紀錄與正在服務的圖相符（contract M2.4） | `sha256`：那次讀取的 digest |
| | `supplied_in_memory` | 程式直接交給 pipeline 一個模型物件，沒有讀任何 checkpoint 位元組 | `sha256` 為 null |
| | `absent` | 沒有模型：由路徑推理排序候選 | `sha256` 為 null |
| `graph` | `verified` | `kg.json` 與 `split_manifest.json` 各讀一次，且圖的 digest 與 manifest 相符 | `kg_sha256`、`split_manifest_sha256` |
| | `unverified` | 圖從檔案讀入，但沒有與任何 workspace manifest 比對（沒有模型的 serving，見下文） | `kg_sha256`；`split_manifest_sha256` 為 null |
| | `in_memory` | 程式在記憶體中提供圖，沒有讀任何檔案 | 兩者皆為 null |
| `shortest_paths` | `verified` | 距離表與其 sidecar 共用同一個 `build_id`，並指向正在服務的圖（§4.3） | `build_id`；`max_hops`（取自 sidecar） |
| | `absent` | 沒有距離表參與評分 | 兩者皆為 null |
| `dataset_record` | `verified` | manifest 綁定、且指向正在服務之圖的紀錄（§4.4）；恰在圖為 verified 時才是 verified | `origin`（`files` 或 `synthetic`）；`kg.provenance.json` 的 `sha256` |
| | `absent` | 沒有驗證任何 workspace，因此沒有讀取紀錄 | 全部為 null |
| `scoring_mode` | — | `gnn_plus_shortest_path`、`gnn_only` 或 `path_reasoning_fallback` | |
| `settings` | — | 會改變分數的 serving 設定，以實際生效者為準 | 見下 |

`settings`：

- `device`：模型實際在哪裡執行——torch 對其參數所在裝置的名稱（`cuda:0`、`cpu`）。
  沒有模型時為 null。
- `device_requested`：要求的是什麼——沒有指定時為 `auto`，或 `cuda`、`cuda:N`、`cpu`
  （§4.8）。沒有模型時為 null。
- `eta`：實際生效的混合權重——有距離表時為設定值，沒有時為 `1.0`。沒有模型時為 null。
- `max_path_length`、`path_length_penalty`、`aggregation_method`、
  `max_paths_per_candidate`、`use_direct_path_optimization`：路徑探索與路徑評分，
  在每一種模式下都生效，因為候選就是這些路徑能到達的疾病。
- `phenotype_normalisation_version`：評分前套用於表型清單的規則版本（目前為 `1`）。

各部分不能互相矛盾：恰在模式不是 `path_reasoning_fallback` 時才有模型在服務；距離表
恰在 `gnn_plus_shortest_path` 時才是 `verified`；verified 的 checkpoint 或距離表只會建立
在 verified 的圖之上。只有在某部分的狀態表示沒有東西可標明時，它的值才會是 null。

### pipeline 狀態裡的同一個欄位

`GET /api/v1/pipeline/status` 會針對目前載入的 pipeline 帶出 `served_identity`，與既有
欄位並列，既有欄位不變（`scoring_mode`、`sp_kg_binding`、`dataset_record`、含 `device` 的
`checkpoint_meta` 等）。沒有載入任何東西時為 null。**狀態描述的是現在載入的東西，reload
會換掉它。** 要知道某個結果是由什麼產生的，請讀那個結果自己的 `served_identity`，或它的
報告與 CSV，而不是狀態。

### 怎麼讀

- **正常的部署**應該是：checkpoint `verified`、圖 `verified`、最短路徑 `verified`、
  dataset record `verified` 且來源為 `files`、`gnn_plus_shortest_path`、裝置 `cuda:0`，
  要求的是 `auto` 或 `cuda`。
- **把 digest 與你部署的檔案比對**：checkpoint、`kg.json`、`split_manifest.json` 與
  `kg.provenance.json` 的 SHA-256 應與身分中的 digest 相同，距離表的 `build_id` 應與
  `shortest_paths.meta.json` 裡的一致。
- **其他情況都會明寫，不會隱藏**：來源 `synthetic` 表示這是產生出來的 workspace，不是臨床
  資料（§4.4）；最短路徑 `absent` 加上 `gnn_only` 與 eta `1.0` 表示沒有服務距離表（§4.3）；
  checkpoint `absent` 是沒有模型的路徑推理（見下文）；裝置 `cpu` 是明確要求的（§4.8）。
- **它不證明什麼**，以 contract 的宣稱界線顯示在身分下方（原文照錄）："Checked: these
  files are the ones their records name, and the records agree on the relations marked ✓.
  Not checked: who changed them, or anything a person able to edit both files and records
  could alter."

### 沒有模型的 serving

沒有設定 checkpoint、只用 `kg.json` 做路徑推理時，不會驗證任何 workspace（§4.4），它的
身分也如實這樣寫：checkpoint `absent`；圖 `unverified`，帶它所讀 `kg.json` 的 digest，
沒有 manifest digest；最短路徑與 dataset record `absent`；`path_reasoning_fallback`；
裝置、要求的裝置與 eta 皆為 null。其中沒有任何部分宣稱已驗證。這個模式是否預設保留，
是尚未決定的政策問題（B-2），M3c 沒有決定它。

### Diagnosis 分頁與其匯出

WebUI 顯示的是**結果自己的身分**，絕不是狀態的：

- **結果上方的橫幅**，每個不可漏看的狀態一條：`SYNTHETIC WORKSPACE: not a clinical
  result.`、`NO TRAINED MODEL: path-reasoning fallback.`、`NO SHORTEST-PATH TABLE: GNN
  score alone.`、`DEMO ANSWER: not computed.`，以及回應沒有這個欄位時（M3c 之前的伺服器）
  的 `WHAT PRODUCED THIS RESULT WAS NOT REPORTED.`。
- **結果上方的標記（badge）**（自 M5-2 起）把結果的整個身分逐項與「Model in use」卡片最近一次
  讀到的使用中模型比對："Produced by the model in use"、"Produced by an earlier model or
  pipeline — N parts differ ›"，或附一個原因的 "Cannot check against the model in use yet"。
  它的 › 會在「This result」上打開身分抽屜：每一項與目前使用中的對照、可 Copy 的完整 digest、
  宣稱界線與 "Not a certified medical device; decision support for a qualified clinician."
  （§4.14「Diagnosis (clinical)」）。
- **目前服務的是什麼**顯示在頁面的「Model in use」卡片上（§4.14）。自 M5-2 起，Model
  Configuration 折疊區（其路徑欄位、Load / Reload、Save Config 與 Reset）以及頁首的
  「Serving now」一行都已移除：操作者以卡片上的「🔒 Change model…」更換模型，它列出伺服器的
  workspace 與其 checkpoint，只重新載入伺服器列出的項目。`POST /api/v1/pipeline/config`
  仍保留給 API 用戶端，由 D8 檢查（§4.13）；儀表板不再寫入它。
- **Markdown 報告**在候選之前有「What produced this result」段落：同樣的橫幅、一個
  Part | State | Value 表格（digest 完整寫出，`sha256:<64 hex>`）、宣稱界線與固定聲明。
  缺少的部分寫成它的狀態與原因，絕不留白。
- **CSV** 以欄位帶出身分（見下）。
- **檔名**帶有複製出去後也不能遺失的兩種狀態：合成 workspace 為
  `diagnosis_SYNTHETIC_<patient>_<session>.csv` / `_report.md`，demo 回答為
  `diagnosis_DEMO_…`；其他情況不變。

### CSV 的身分欄位

CSV 仍然**只有一列標題**、沒有前言。原本的 15 個候選欄位在前，順序不變；其後接 21 個
欄位，每欄存放身分的一個部分，**每一列都相同**：

| 欄位 | 內容 | 身分中為 null 時 |
|---|---|---|
| `served_scoring_mode` | `gnn_plus_shortest_path`、`gnn_only` 或 `path_reasoning_fallback` | — |
| `served_checkpoint_state` | `verified`、`supplied_in_memory` 或 `absent` | — |
| `served_checkpoint_sha256` | 所讀 checkpoint 的 `sha256:<hex>` | `absent`（沒有模型）；`none`（記憶體中提供的模型） |
| `served_kg_state` | `verified`、`unverified` 或 `in_memory` | — |
| `served_kg_sha256` | 所讀 `kg.json` 的 `sha256:<hex>` | `none`（記憶體中提供的圖） |
| `served_split_manifest_sha256` | 所讀 `split_manifest.json` 的 `sha256:<hex>` | `absent`（圖為 `unverified` 或 `in_memory`） |
| `served_sp_state` | `verified` 或 `absent` | — |
| `served_sp_build_id` | 距離表與其 sidecar 共用的 `build_id` | `absent` |
| `served_sp_max_hops` | sidecar 的 `max_hops` | `absent` |
| `served_dataset_record_state` | `verified` 或 `absent` | — |
| `served_dataset_origin` | `files` 或 `synthetic` | `absent` |
| `served_dataset_record_sha256` | 所讀 `kg.provenance.json` 的 `sha256:<hex>` | `absent` |
| `served_device` | 實際生效的裝置（`cuda:0`、`cpu`） | `none`（沒有模型）；`not_stated`（記憶體中提供、未表明裝置的模型） |
| `served_device_requested` | 所要求的 `auto`、`cuda`、`cuda:N` 或 `cpu` | `none`（沒有模型，或記憶體中提供且未指定裝置的模型） |
| `served_eta` | 實際生效的混合權重 | `none`（沒有模型） |
| `served_max_path_length` | 整數 | — |
| `served_path_length_penalty` | 數字 | — |
| `served_aggregation_method` | 文字 | — |
| `served_max_paths_per_candidate` | 整數 | — |
| `served_direct_path_optimization` | `true` 或 `false`（`use_direct_path_optimization`） | — |
| `served_phenotype_normalisation_version` | 整數（目前為 `1`） | — |

- **digest 完整寫出並加上 `sha256:` 前綴**（`sha256:<64 hex>`）。前綴標明演算法，也避免
  試算表把全為數字的 digest 當成數值。
- 布林值寫成 `true` / `false`；數字照 API 傳來的樣子寫（`0.7`、`1.0`）。
- **身分欄位沒有空白儲存格。** 身分中為 null 的值一律寫成一個字（候選欄位沿用原本的慣例，空白
  表示沒有值）：

| 字 | 何時寫出 |
|---|---|
| `absent` | 該部分不存在：沒有 checkpoint、距離表、manifest 或 dataset record |
| `none` | 不適用：沒有模型時的裝置、要求的裝置或 eta；記憶體中提供的模型或圖沒有 digest；記憶體中提供的模型沒有要求的裝置 |
| `not_stated` | 記憶體中提供的模型沒有表明它的裝置（僅 `served_device`） |
| `demo` | demo 回答的每一個 `served_` 欄位 |
| `not_reported` | 回應沒有帶 `served_identity` 時的每一個 `served_` 欄位，或回應漏掉的某個值 |

- 試算表會當成公式執行的文字儲存格——以 `=`、`+`、`-`、`@`、tab 或 carriage return
  開頭者——會在前面加上 `'`。這也適用於自由文字的候選欄位；數值欄位不受影響。

## 4.8 serving 的裝置（`SHEPHERD_DEVICE`）

**CUDA 是本專案的必要條件，serving 不會用 CPU 代替它**（自 contract M3c 起，2026-10-16；
訓練自 M3a 起已用同樣方式拒絕）。

| 值 | 意義 |
|---|---|
| 未設定、空白或 `auto` | CUDA。沒有 CUDA 的主機上**拒絕** |
| `cuda`、`cuda:N` | 該 CUDA 裝置。沒有 CUDA 時拒絕 |
| `cpu` | 明確要求 CPU：**僅供開發使用** |
| 其他任何值 | 原樣交給 torch，與訓練的做法相同 |

在哪裡設定：

- **啟動時**：服務環境中的 `SHEPHERD_DEVICE`。
- **reload 時**：請求的 `device`（`POST /api/v1/pipeline/reload`）。null 或空白時先退回
  `SHEPHERD_DEVICE`，再退回 `auto`；空白的環境變數也視為未設定。

**何時檢查**：只要要服務 checkpoint，就在讀取任何東西之前檢查——服務在解析 `kg.json`
之前先確定裝置（reload 則在檔案檢查之後、自動選擇讀取任何候選 checkpoint 之前），pipeline
在讀 workspace 或 checkpoint 之前再確定一次。啟動被拒絕時不發布
任何 pipeline，`/api/v1/diagnose` 回 503；reload 被拒絕時，原本的 pipeline 繼續服務。
拒絕訊息為：

> Serving device 'auto' requires CUDA, which is not available here. Hardware
> acceleration is a hard requirement for deployment and CUDA is currently the
> only supported backend, so serving does not substitute the CPU for it. Set
> SHEPHERD_DEVICE=cpu (or send device 'cpu' with a reload) explicitly for
> development; the served identity records the device asked for and the one in
> effect.

沒有 checkpoint 的 serving（只做路徑推理）不會決定任何裝置。

**部署必須有硬體加速，目前唯一支援的後端是 CUDA（NVIDIA）。** 這是目前的範圍，不是永久的
架構限制。torch 的 ROCm 版本也把裝置稱為 `cuda`，但這裡沒有驗證過。其他加速硬體會在實際目標
硬體到位時再適配並驗收，並有自己明確的選擇規則；在此之前不提供其他後端，也不會悄悄改用其他
裝置（`docs/working/BACKLOG.md` 第 24 項）。

**served identity 兩者都記錄**（§4.7）：`settings.device_requested` 是所要求的（沒有指定
時為 `auto`），`settings.device` 是實際生效的、模型參數所在的裝置（`cuda:0`，而不只是
`cuda`）。狀態裡的 `checkpoint_meta.device` 保留。

> **行為變更（M3c）**：沒有 CUDA 的主機若未指定裝置就服務，以前會在讀完 checkpoint、
> 建好模型之後改用 CPU，既不拒絕也不警告，只有日誌與狀態裡的 `checkpoint_meta.device`
> 看得出來。**現在會被拒絕。** CPU 開發機必須設定
> `SHEPHERD_DEVICE=cpu`（reload 時送 `"device": "cpu"`，或留 null 讓這個變數生效）。
> 字面上的 `auto` 以前會傳到 torch 才失敗，現在與未設定是同一個要求。WebUI 沒有裝置
> 選擇器。`scripts/setup_demo.py` 印出的 serving 指令會帶上 demo 訓練時用的裝置。

---

## 4.9 測試 cohort：匯入、量測、登錄（自 contract M4 起）

自 contract M4 起（2026-10-16），checkpoint 的測試是在命令列上、以一個**已登錄的測試 cohort
版本**進行——MyGene2、院內離線 cohort，或驗收用的合成 cohort——結果登錄在 checkpoint 旁邊。
流程如下：

1. `scripts/import_cohort.py` 先預覽、再提交一個對應到某個 workspace 圖的 cohort 版本；
2. `scripts/measure_scorer.py --cohort LABEL --registry DIR` 在這個版本上量測 checkpoint
   （Mode C），並保留完整報告；
3. `--record`，或之後用 `scripts/record_evaluation.py --report`，把報告登錄到 checkpoint 旁的
   `evaluations.json`；
4. Diagnosis 頁面的「Model in use」卡片以狀態顯示目前載入之 checkpoint 已登錄的測試（自 M5-2
   起；紀錄本身列在 Research › Evaluate › Models）。

這些指令由操作者在機器上執行。自 M5-2 起，儀表板的 Research › Evaluate › Test 頁面也能從瀏覽器執行同樣的匯入、量測與登錄，每一項都是由 D8 檢查的操作者控制項（§4.13、§4.14）；沒有任何 API 請求會匯入、量測或登錄任何東西。

### registry

registry 是**由你建立**的目錄，每個指令都以**位於 repository 之外的絕對路徑**傳入。相對路徑、
不存在的路徑，以及解析後落在 repository 內的路徑（包括經由 symlink）一律拒絕。不會有任何東西
替你建立它，也沒有任何請求指定它。各指令以 `--registry` 接收它；服務則從 `SHEPHERD_REGISTRY`
讀取（§4.12），以同一個規則檢查。

**各指令也需要 `SHEPHERD_REGISTRY`**（自 M5 審查起，BACKLOG 37）：報告只從部署的 registry 登錄
（`--record`、`record_evaluation.py --report`、Test 頁面），`measure_scorer.py --cohort` 也只在
部署的 registry 中量測，也就是 `SHEPHERD_REGISTRY` 所指名的那一個。兩者以解析後的路徑比較，所以
經由指向該目錄的連結給出的 `--registry` 會被接受；registry 的副本，或經由 bind mount 或另一個
掛載點到達的同一個目錄，則是另一個 registry，在讀取其中任何東西之前就會被拒絕。
`measure_scorer.py --cohort` 只在獲准開始時解析一次 `--registry`，之後都經由那個目錄讀寫：在它取得
GPU 鎖或啟動裝置期間被改指向別處的連結，不會讓這次執行搬到別的目錄。這個變數未設定時，
每個這類指令都會被拒絕，並說明要像服務的環境那樣設定它。請在執行指令的 shell 中，從服務讀取的
同一個檔案設定它（§4.11）：

```
measure_scorer: refused: --cohort measures a version of the deployment's registry only, and SHEPHERD_REGISTRY is not set here, so which registry is the deployment's cannot be told. Set SHEPHERD_REGISTRY in this environment to the value the service's environment gives it (deployment guide §4.11: one file that both the service and this shell read). [...] Nothing was read or written.
```

registry 之下的目錄也會檢查。每一個目錄（`sources/`、`versions/`、版本目錄、其 `reports/` 與每個
run 目錄）在寫入任何東西、或從中讀取版本之前，都會再解析一次；若某個連結把它帶進 repository，
或帶到它所屬目錄之外——registry，或對 `reports/` 與 run 目錄而言是該版本自己的目錄——就拒絕。
停留在 registry 內的連結會被跟隨。拒絕訊息會說明已經寫入了什麼，例如只有來源檔的副本。這些
檢查在每次寫入之前進行，而不是與寫入同為一個不可分割的操作，因此恰在那一刻被換掉的連結不會被
抓到。registry 是你的，在這台機器上，一次只有一個寫入者。

它存放病人層級的資料——原樣保存的來源檔、對應後的樣本與逐案的排名——所以請建在只有執行這些
指令的帳號讀得到的地方（例如 `mkdir -m 700 /srv/shepherd/cohorts`）。其中的檔案依 §4.6 的方式
發布：先寫入暫存檔，再改名就位，權限依目錄的政策。

```
<registry>/
  sources/<sha256>.json                  來源檔的位元組，原樣保存，以其 digest 命名
  versions/<label>/cohort.json           版本 manifest；最後寫入，它存在才代表版本存在
  versions/<label>/samples.json          對應後的樣本，由 cohort.json 的 digest 綁定
  versions/<label>/reports/<run id>/     每次量測一個目錄：先 ranks.json，再 report.json
```

**版本永不改變。** 一個 label 只能用一次。沒有 `cohort.json` 的 `versions/<label>/` 是中斷的
匯入：每個指令都拒絕它，匯入工具會指出它，讓你在重用該 label 之前移除。沒有任何指令會刪除
registry 中的東西。

### 來源檔：`shepherd-cases/1`

匯入工具只讀一種格式：本專案共用的匯入格式。格式名稱只說明檔案的排列方式，不說明病例的來源：
合成 cohort 與轉換後的真實資料都以同一種格式寫成（真實來源會有一個轉接器，輸出這種格式）：

```json
{
  "format": "shepherd-cases/1",
  "cases": [
    {"case_id": "c001", "phenotypes": ["HP:0001250", "HP:0001263"], "disease": "MONDO:0011073"}
  ]
}
```

- 兩層都必須恰好是這些鍵。`case_id` 為可列印文字的非空字串（不含控制或格式字元、空白以外的
  分隔字元，或孤立的 surrogate），在檔案中唯一；`phenotypes` 為非空串列，每項是 `HP:` 加七位
  數字；`disease` 是 `MONDO:` 加七位數字。
- **任何不符的地方都會讓整個匯入被拒絕**，並依案例索引與 case id 列出每一個問題；不寫入任何
  東西。每個問題指出值錯在哪裡（案例、欄位、位置），以及該值是什麼型別或違反了哪條規則，
  **絕不印出值本身**，所以拒絕訊息絕不會印出表型或診斷。不是可列印文字的 case id 會以跳脫形式
  顯示。
- **以 `OMIM:`、`ORPHA:` 或 `Orphanet:` 給出的疾病**同樣讓整個匯入被拒絕，並指出命名空間與
  案例數：圖中的疾病是 MONDO 詞彙，這個版本不做轉換。真實格式（MyGene2 的、院方的）在有樣本
  之後以轉接器轉成這個格式（`docs/working/PLAN_TEST_RESULTS.md` §4.9）。

### 匯入：先預覽，再提交

```bash
python scripts/import_cohort.py --registry /srv/shepherd/cohorts \
    --data-dir data/workspaces/<ws> --source /srv/intake/cases.json \
    --label mygene2-2026-10 --role mygene2 [--partial-phenotypes keep|exclude]
```

label 與 role 是小寫識別字（字母、數字、`_`、`-`、`.`；最多 64 字元）；role 不可為 `train`
或 `val`。

**預覽不寫入任何東西。** 它以與訓練、serving 相同的讀取器與檢查，把每個案例對應到 workspace
manifest 所綁定的圖，並印出：

- 來源案例數、可排名的案例數，以及**依類別列出的排除案例，各附 case id 與原因**（只在本機
  顯示：會出現 case id，不會出現表型與疾病；不是可列印 ASCII 的 case id 以跳脫形式印出）：
  - `ground_truth_absent`——案例的疾病不是這張圖的疾病節點；
  - `phenotype_unmappable`——沒有任何詞彙對應得上，或在 `--partial-phenotypes exclude` 下有
    詞彙對應不上；
- 對應不上的 HPO 詞彙（過時或替代 id 不是節點，因此屬於此類）、以少於所列詞彙排名的案例數，
  以及移除的重複（列兩次的詞彙只算一次，表型正規化規則 1）；
- 圖的四個 digest、dataset record 的來源與 digest，以及其 HPO 與 MONDO 條目；
- 一個 **preview id** 與完整的提交指令。

`--partial-phenotypes keep`（預設）以對應得上的詞彙為案例排名；`exclude` 則排除它。沒有任何
詞彙對應得上的案例一律排除。所採政策記錄在版本中。

**提交**：執行印出的指令，也就是相同參數加上 `--commit <preview id>`。匯入工具會重新讀取所有
輸入，只有結果仍是同一個 id 時才發布；否則拒絕（"inputs changed since the preview"）。它會印出
版本目錄與 manifest 的 SHA-256。沒有任何案例可排名的來源在提交時會被拒絕，因為沒有東西可量測。

### 量測

```bash
python scripts/measure_scorer.py \
    --checkpoint data/workspaces/<ws>/runs/<arch>/<run>/checkpoints/<file>.pt \
    --data-dir data/workspaces/<ws> \
    --cohort mygene2-2026-10 --registry /srv/shepherd/cohorts \
    [--training-evidence DIR ...] [--record] [--device auto|cuda|cpu] [--batch-size 32]
```

請在已設定 GPU 鎖、且與服務使用同一把鎖的 shell 中執行（§4.11）；沒有鎖時它會拒絕，結束碼 1，
不寫出任何東西。

在寫入任何東西之前，這次執行會以它實際讀到的位元組檢查：版本的 `cohort.json`，以及其
`samples.json` 是否符合它所綁定的 digest；**版本的計數**——它們會成為報告與 ledger 的分母——必須
存在，且彼此一致、也與版本一致（`ranked_eligible` = `samples.n`；`source_cases` =
`ranked_eligible` + `n_excluded` = 來源的案例數；`n_excluded` = 各類別之和 = 所列的排除案例數），
而且讀到的樣本數必須等於 `samples.n`；workspace 的圖檔是否符合其 manifest 與 dataset
record（§4.4）；**版本是否屬於這個 workspace**——它的四個圖 digest 與 dataset record 必須是這個
workspace 的，否則在讀取 checkpoint 之前就拒絕；接著是 checkpoint（以 `weights_only=True` 讀取）
是否與它訓練所用的圖相符。裝置依 §4.8：`auto` 與 `cuda` 需要 CUDA，沒有時在讀取任何東西之前
就會被拒絕；`cpu` 是開發用的執行，會如實記錄，而且與 CUDA 的量測是不同的量測。

**R10——與模型訓練及驗證資料的重疊。** 這次執行會尋找 checkpoint 所記錄 digest 的訓練與驗證
樣本檔：先找 workspace 自己的 `train_samples.json` 與 `val_samples.json`，再找每個
`--training-evidence` 目錄中（不遞迴）的每一個 `*_samples.json`（若模型在別處訓練，或
workspace 之後重建過，請給出訓練當時 workspace 所在的目錄）。只有位元組具有所記錄 digest 的
檔案才算數。共四個標籤：訓練與驗證 × 疾病重疊與表型集合重疊，每個標籤為 `overlap`（附計數與
分母）、`none`、`unverifiable`（**絕不顯示為零**），或驗證範圍的 `no validation inputs
recorded`（checkpoint 記錄了訓練輸入，而且完全沒有驗證的條目：訓練沒有跑驗證）。`unverifiable`
會附上該範圍的原因：

- **找不到**：看過的檔案中沒有一個具有所記錄的位元組。請以 `--training-evidence` 指出存放它們的
  目錄，再量測一次；
- **找到、但讀不了**：存放它們的檔案找到了，但無法解析（例如 id 不是整數，或某個病例重複列出
  同一個表型）；
- **checkpoint 沒有記錄**：它在這個範圍沒有指出任何可找的檔案，而且它的紀錄不是 pipeline 寫出
  的：每次訓練都會以 digest 記錄其訓練輸入，所以沒有這個 digest 的紀錄，或條目在、卻不是 digest
  的紀錄，都不可能出自 pipeline。`--training-evidence` 也幫不上忙。

`none` 不代表證明沒有洩漏：它只就 checkpoint 所記錄的輸入而言。

**哪些情況會拒絕這次執行，而不是給出標籤。** 不存在、不是目錄，或無法列出內容（權限、網路掛載
的 I/O 錯誤）的 `--training-evidence` 目錄，會在讀取任何候選檔之前被拒絕：沒有被查看的目錄會把
輸入回報成找不到。存在但讀不了的候選檔（權限、I/O 錯誤）會讓這次執行被拒絕，不寫入任何東西，
因為它可能正是證據；請讓它可讀，或把它所在的目錄從 `--training-evidence` 中拿掉。候選檔只讀到
每個所記錄的 digest 都找到為止，所以排在符合之後的讀不了的檔案根本不會被讀到。

**報告保存在哪裡。** 在 registry 中、該版本之下：
`<registry>/versions/<label>/reports/<UTC 時間>-<checkpoint digest 前 12 個十六進位字元>-<8 個
隨機十六進位字元>/`，每次執行一個新目錄：先 `ranks.json`（逐案、病人層級），再 `report.json`，
後者以 digest 指明排名檔。報告絕不覆寫，不論是否登錄，每份報告都會保留。`--cohort` 搭配
`--output` 會被拒絕。

主控台會印出 Mode C 的指標、覆蓋率（`N ranked of M source cases; K excluded`，附各類別）、
以 `src/evaluation/caveats.py` 共用字句表示的四個 R10 標籤（`unverifiable` 附其原因）、一次標籤定義、報告路徑
與 run id。報告本身只保存狀態，不含任何說明文字。

### 登錄

- **`--record`** 在執行結束後立即登錄報告。**沒有加它時**，以及 `--record` 被拒絕時（結束碼
  3），執行會在一行中印出之後登錄用的指令，指名它量測的檔案：

  ```bash
  python scripts/record_evaluation.py --report "<registry>/versions/<label>/reports/<run id>/report.json" --checkpoint "data/workspaces/<ws>/runs/<arch>/<run>/checkpoints/<file>.pt"
  ```

  checkpoint 是這次執行讀取的檔案，取其解析後路徑，所以不論檔名為何，這個指令都能用。每個路徑
  都加上雙引號，因此含空白的路徑在 POSIX shell、cmd 或 PowerShell 中都維持為一個參數。含有
  `"`、`$`、反引號、`%` 或 `!` 的路徑會被某個 shell 展開，必須自行加引號。
- **`record_evaluation.py` 以兩個參數之一、且恰好一個，指明 checkpoint**（兩個都沒給或都給，
  命令列會被拒絕）：
  - `--checkpoint FILE`：這一個檔案，不論名稱為何，必須具有報告所量測的位元組；
  - `--checkpoint-dir DIR`：DIR 中直接存放的某個 `*.pt` 檔必須具有這些位元組。

  否則不寫入任何東西。`--record` 與 `record_evaluation.py` 走同一個函式，所以登錄的紀錄相同。
- **ledger** 是 checkpoint **解析後路徑**所在目錄中的 `evaluations.json`，不論報告經由哪條路徑
  而來：`--record`、`--checkpoint FILE`，或 `--checkpoint-dir DIR`（放在符合之檔案的解析後路徑
  旁；該檔案是連結時，那就不是 DIR）。在 run layout 下為
  `data/workspaces/<ws>/runs/<arch>/<run>/checkpoints/evaluations.json`。以 symlink 指向的
  checkpoint，其 ledger 放在連結目標旁邊，服務也是在那裡找，所以同一組權重絕不會在連結旁多出
  第二個 ledger。量測與服務都以解析後路徑讀取 checkpoint，所以目錄在讀取檔案時就已確定。執行
  期間、或載入後未重新載入時被改指向的連結，不會改變該次執行的紀錄登錄在哪裡，也不會改變到哪裡
  找已載入模型的紀錄。不會為 ledger 建立任何目錄。
- **結果**：已登錄（附序號）；已存在（同一份報告再次登錄：不寫入任何東西）；與既有紀錄一致
  （報告的 digest 加入該紀錄的佐證報告清單）。R10 與環境不參與比對，因為各自只對自己的報告
  成立。當佐證報告的 R10 或環境與紀錄不同時（例如重跑時加上 `--training-evidence`，找到了第一次
  沒找到的證據），**紀錄保留自己的值，不會更新**。登錄會以一行說明這件事並指出該報告，較晚那次
  執行的 R10 就保存在其中。與同一鍵——同一個 checkpoint、cohort 版本與量測設定，結果卻不同——
  的紀錄互相矛盾的報告會被拒絕，並指出不同的欄位。沒有覆寫旗標；錯誤的紀錄以編輯這個可讀的
  JSON 檔來更正，會留下 diff。
- **之後查看**：`python scripts/record_evaluation.py --checkpoint <file> --show
  <checkpoint sha256>` 讀取服務為該檔案所讀的 ledger，也就是其解析後路徑旁的那一個。若改用
  `--checkpoint-dir <dir>`，`--show` 讀的是該目錄內的 `evaluations.json`；對一個放連結的目錄
  而言，那不是服務所讀的 ledger。空的回答代表這個 ledger 沒有這組權重的結果，不代表它們從未被
  評估過。
- **一次只能有一個寫入者。** 登錄時若發現 ledger 在讀取之後被改過，會拒絕而不是覆蓋；這不是
  鎖，所以不要從兩個地方同時登錄到同一個目錄。
- 只有**已登錄 cohort 版本**的報告會被登錄；`--split` 的報告不會。

### 結束碼

| 指令 | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| `import_cohort.py` | 已印出預覽，或已提交版本 | 拒絕，一則以 "Nothing was written." 結尾的訊息（沒有任何案例可排名的預覽也是） | 命令列被拒絕 | — |
| `measure_scorer.py` | 已量測（加 `--record` 時：已登錄、已存在或一致） | 某項檢查拒絕了這次執行：一則訊息，不寫入任何東西（在沒有 CUDA 的主機上給 `--device auto` 或 `cuda` 也是，在讀取任何東西之前）；或**已量測，但報告沒有寫出**（例如磁碟已滿或唯讀）：一則訊息說明可能留下了什麼，而那絕不會是可登錄的報告——`--output` 旁這次執行的排名檔，或報告存放處中沒有 `report.json` 的執行目錄（寫出報告之後才失敗的執行，例如主控台被關閉，會說明報告在哪裡） | 命令列被拒絕，包括屬於另一種 cohort 的參數 | **已量測，但登錄被拒絕**：訊息會指出報告與之後登錄用的 `record_evaluation.py` 指令；若報告檔已不是這次執行發佈的位元組，則如實說明，並要求重新量測（那份報告可能已不存在）；若這份報告的鍵下有結果不一致的執行，則說明 "runs disagree — investigate" 並保留報告，不印出之後登錄用的指令（只要這些執行仍不一致，任何登錄都不會被接受） |
| `record_evaluation.py` | 已登錄、已存在、一致，或已顯示 | 拒絕，一則訊息；不寫入任何東西 | 命令列被拒絕，包括 `--checkpoint` 與 `--checkpoint-dir` 都沒給或都給 | — |
| `audit_split_overlap.py` | 已寫出報告 | 拒絕，一則訊息；不寫入任何東西 | 命令列被拒絕（例如已移除的 `--evaluation-cohort-kind`、只給 `--registry` 不給 `--cohort` 或反之，或與 `--cohort` 一起給出 `--splits`），並印出 usage 行 | — |

若 checkpoint 目錄對你而言是唯讀的，量測會以結束碼 3 結束：報告會保留，由可以寫入該目錄的人
用這次執行印出的 `record_evaluation.py` 指令登錄。

**`--record` 只登錄這次執行發佈的報告，且必須是這次執行讀取的 checkpoint。** 執行會保留它寫入的
報告位元組的 digest，以及這次 checkpoint 讀取的 identity；登錄時重新讀取報告，在解碼任何內容之前先
核對這次讀取的 digest，再核對它記錄的 checkpoint。若報告檔在這之間被替換、修改、損壞、刪除或無法讀取
——即使換成另一份合法報告（另一個 checkpoint 的，或另一個 cohort 版本的），或已變成指向報告存放處之外
檔案的連結——就不登錄任何東西，以結束碼 3 結束，並說明該路徑上的檔案不是它發佈的報告（或已不存在），
因此這次執行自己的報告可能已不存在：請重新量測以登錄這次量測。只有在讀回的報告確實是發佈的那份、而是
其他環節（例如 ledger）拒絕時，才會印出之後登錄用的指令；這個鍵下的執行結果不一致時則絕不印出。之後用
`record_evaluation.py` 登錄時，仍照舊以指定的 checkpoint 檔核對報告。

### Diagnosis 頁面顯示什麼

**自 M5-2 起（2026-10-21），Diagnosis 頁面只以狀態顯示目前載入之 checkpoint 已登錄的測試**，
位於「Model in use」卡片上："Recorded tests: 3 · 1 contested"、"3 · contested state not checked"（未設定 registry，
或沒有任何一筆紀錄的 cohort 版本或其報告存放處能讀取；絕不顯示為沒有爭議）、"3 · 1 contested ·
contested state of 1 not checked"（部分紀錄的狀態無法檢查；分開計數）、"none in ledger"、
"no ledger"、"ledger not readable" 或 "not read"，與該 checkpoint 在「Change model…」中那一列的用語相同，並以
checkpoint 的 digest 而不是檔名來找（換成另一個模型之後，絕不會顯示前一個模型的狀態）。它的 ›
會打開 Details 的 Evidence，同一個狀態在那裡與訓練器的驗證數值並列（"Validation MRR (Trainer,
top-20) … — selection metric on val, not an independent test"）；頁面上不顯示任何 Mode C 數字。
紀錄本身列在 Research › Evaluate › Models。

在 M5-2 之前，模型狀態（Model Configuration）在「Serving」那一行下方，為目前載入的 checkpoint
（`sha256:<前 12 字元>…`）完整列出 **「Recorded test results (Mode C)」**，透過
`GET /api/v1/evaluations/loaded-model` 讀取，用語如下。這個 route 仍以這些紀錄回應 API 用戶端，
且自 M5 審查起（BACKLOG 36）每筆紀錄都附上它的 D5 狀態（`contest`：`uncontested`；`contested`，
附上不一致的欄位與各次執行；`cannot_be_checked`，絕不視為沒有爭議），由儀表板本身的實作依服務的
`SHEPHERD_REGISTRY` 算出，但不附原因本身的文字，因為那些文字可能指名伺服器上的路徑。D5 的調查
備註（investigation notes）不會顯示（BACKLOG 40）：

- 紀錄區塊開頭（第一筆紀錄之前）出現一次 **"Disease ranking by the model, approved scorer
  policy (Mode C)."** 與說明 **"This is the model's own Mode C disease ranking. Until B-1 ships,
  the Diagnosis tab ranks by the legacy GNN+SP mixture behind the path gate, so this number does
  not describe the tab's current ranking."**（原文照錄）；R10 標籤定義與已檢查／未檢查的說明在
  最後一筆紀錄之後出現一次。
- 每筆紀錄：role · label，附版本 manifest 的 digest（`version sha256:<前 12 字元>…`），因此同一個
  label 的兩次匯入（在重建的或另一個 registry 中）會顯示為兩個版本；以已排名案例計算的 Mode C
  指標（其 MRR 與 Checkpoint 那一行訓練器的 "Val MRR" 分開標示）；覆蓋率（`ranked/source`，附各
  排除數）；已檢查的關係（✓ 或不適用）；四個 R10 標籤——`unverifiable`（附該範圍的原因，與主控台
  相同）與 `no validation inputs recorded` 以文字顯示，**絕不顯示為零或數字**；環境類別（CUDA
  且開或關 deterministic algorithms，或 CPU）與 GPU；dataset 來源（合成 workspace 會特別標出）；
  登錄序號與時間，以及佐證報告數。
- 沒有紀錄時改為一句話：載入的 checkpoint 旁沒有 ledger；ledger 未讀取，附原因，原因會指出
  檔案、絕不指出其目錄（例如下述的 schema 2 檔案）；這個 ledger 沒有這組權重的結果；有些紀錄
  因不符登錄寫出的形狀而未顯示；或 "The loaded model changed while its recorded results were
  read; refresh the page."。
- **這個 route** 不接受任何參數，每次請求都讀取載入之 checkpoint 旁的 ledger，不寫入任何東西，
  也不回傳任何路徑、case id 或排名。「旁」指的是 pipeline 載入時所讀 checkpoint 之解析後路徑的
  目錄，所以之後未重新載入而被改指向的連結，不會讓查找離開服務中位元組的 ledger。它掛在自己的
  router 上，不在 pipeline router 中，因此卸載或保護 pipeline routes 的部署
  （`docs/working/results-review/SPEC_4_DEPLOYMENT_SECURITY.md`，B2）仍保有它。

### 行為變更（M4）

- **`measure_scorer.py --cohort-kind` 已移除**，以前 workspace 中當作 supplied cohort 的鬆散
  `<split>_samples.json` 也一併移除。`--split` 只接受產生器的 `train` 或 `val`，且需要
  `--output`；與它一起給出 `--registry`、`--training-evidence` 或 `--record` 會被拒絕。院方或外部
  cohort 以 `--cohort LABEL --registry DIR` 匯入並量測。
- **產生的 split 的報告不可登錄**：`record_evaluation.py` 拒絕它（"a generated split's report
  is not registered"）。`--split val` 是挑選 checkpoint 用的 split，不是保留的測試資料。
- **產生的 split 的報告與排名檔改為先寫入暫存檔再改名就位**（排名檔先）；已存在的 `--output`
  仍會被取代。
- **ledger schema 3。** 既有的 schema 1 或 2 的 `evaluations.json` 會被拒絕——`--record`、
  `record_evaluation.py`（`--report` 與 `--show`）以及讀取已登錄結果的程式都會拒絕，後者會回報
  ledger 未讀取（Diagnosis 卡片上為 "Recorded tests: ledger not readable"）。沒有任何東西會遷移它，其結果也不會被帶過來：**請手動封存**，例如
  `mv evaluations.json evaluations.v2.json`，以寫出它的版本讀取，並以這個版本重新量測需要的
  結果，再登錄新的報告（拒絕訊息也這麼說）。M4 之前寫出的報告同樣會被拒絕，並指示重新量測。
- **`record_evaluation.py`** 接受 `--checkpoint FILE` 或 `--checkpoint-dir DIR`，恰好一個。不論
  哪一個，ledger 都寫在 checkpoint 解析後路徑的旁邊。對於 checkpoint 是連結的目錄，那是在連結
  目標旁邊，不再是所指名的目錄中。
- **`scripts/audit_split_overlap.py`**：`--evaluation-cohort-kind` 已移除；測試 cohort 以
  `--registry DIR --cohort LABEL` 稽核。只給 `--registry` 不給 `--cohort`（或反之），以及與
  `--cohort` 一起給出 `--splits`，都是被拒絕的命令列（結束碼 2，並印出 usage 行）。版本的圖與
  dataset record digest 會與這個 workspace 的 manifest 所記錄的比對，不載入 tensor；只讀
  `train`。報告的 `schema_version` 為 6。

### 自 M5-1 起：只從報告存放處登錄、結果不一致的執行、使用紀錄、確定性執行

以下同時適用於 `measure_scorer.py --record` 與 `record_evaluation.py --report`，因為兩者走同一個
登錄函式。上面的結束碼不變。

**只有 registry 報告存放處中的報告才能登錄。** `--report` 必須指向
`<registry>/versions/<label>/reports/<run id>/report.json`，也就是量測印出的 `Report:` 路徑。
寫入任何東西之前，登錄會檢查四件事：

- 從這個路徑找出 registry 與版本，並讀取該版本；
- 報告必須完全指名這個版本（label、role、manifest digest、來源、圖、mapping 與計數）；
- 它的執行目錄名稱必須含有它量測的 checkpoint 的前 12 個十六進位字元；
- 上述 checkpoint 與 ledger 的檢查照舊適用。

放在其他任何地方的副本，在讀取之前就會被拒絕：「… is not a report the registry's report store
holds … Register the store's original instead」。副本仍可閱讀與分享，但不能登錄。指向存放處中
報告的連結，視為存放處中的原件；存放處中那個本身是連結、指向別處檔案的 `report.json` 則不算。
**registry 必須是部署的 registry**（自 M5 審查起，BACKLOG 37）：整個 registry 的副本以相同的結構
持有相同的位元組，但它會有自己的使用紀錄與自己的不一致封鎖，而 Research › Models 與 job runner
從不讀取它們，所以副本中的報告會被拒絕（「… is not registered: the registry … is not the
deployment's: SHEPHERD_REGISTRY (…) resolves to …」）。整個搬移的 registry，在服務與你的 shell
中的 `SHEPHERD_REGISTRY` 都指名它的新位置之後，便可再次使用；沒有任何絕對路徑被存成身分。如果
`--record` 發現它的 registry 已不再是部署的 registry（執行期間連結被改指向別處），執行會以 exit 3
結束、保留報告，且不提供稍後登錄的命令，因為那個命令也會以同樣的原因被拒絕。這項檢查攔下的是
誤用的 registry；刻意把這個變數設成副本後執行的指令，就會使用那個副本。產生的 split 的報告寫在
`--output`，從不在存放處中，所以現在它是因為這個原因被拒絕。

**結果不一致的執行會封鎖其鍵（"runs disagree — investigate"）。** 存放處中同一個鍵（同一個
checkpoint、cohort 版本與量測設定）下的報告，會用 ledger 所用的同一種比對互相比較。只要其中兩份
結果不同，這個鍵下的任何報告都不會被登錄。這適用於加或不加 `--record` 的執行、之後的
`record_evaluation.py`，以及再次登錄一份已登錄的報告。訊息會指出哪些不同、是哪些執行。在不一致
出現之前已登錄的紀錄仍留在 ledger 中。在 `--record` 之下，執行以結束碼 3 結束並保留報告，但不會
印出之後登錄用的指令，因為只要這些執行仍不一致，任何登錄都不會被接受。不加 `--record` 時，執行在
發佈報告後以同樣方式檢查存放處：它以結束碼 0 結束（如同任何不登錄的執行），印出
"Not registered (no --record), and runs disagree — investigate" 與不同之處，取代那行指令；它的
`--outcome` 檔中 `registration.why` 也這樣說明。若無法檢查這些執行是否不一致（見下文），不加
`--record` 的執行會印出指令，並說明在能夠檢查之前登錄都會被拒絕。往下走的方式：

- 比較各報告的環境資訊（每份 `report.json` 的 `manifest.environment`）；
- 以 `--deterministic` 重跑，這是不同的鍵。

調查備註記錄誰看過、何時看過、發現了什麼。備註存在
`<registry>/versions/<label>/reports/notes.jsonl`，與該鍵的報告放在一起，絕不寫進 ledger，也不會
解除封鎖。Research › Test 可從儀表板新增備註，這是操作者控制項（§4.14）。封鎖建立在一般檔案上：刪除不一致的
報告就會解除它。這些工具從不刪除任何東西。

**「無法檢查」同樣會被拒絕，絕不視為沒有爭議。** 存放處中有下列任一情形時，整個版本都無法檢查：

- 一份無法讀取或解析的報告；
- 一個是連結、指向存放處之外的執行目錄或 `report.json`；
- 一份無法判斷其鍵的報告。

較早版本寫出、本版本不讀取其 manifest 的報告，只會封鎖它所指名的那個鍵。拒絕訊息會指出每個檔案。
把檔案移出 `reports/` 或加以修復，再登錄一次。

**使用紀錄**是 `<registry>/versions/<label>/reports/usage.jsonl`，每行一個 JSON 物件，只追加、
從不改寫。它記錄：

- 這個版本的每一次量測（`test_run`：時間、run id、checkpoint 與報告的 digest、Mode C 的指標、
  `n_ranked`、是否為確定性執行、裝置、是否要求 `--record`）；
- 每一次寫入 ledger 的登錄（`registered`：結果與序號）；
- 每一次從依該版本分數排序的清單載入的模型（`model_load`：排序清單所用的指標與 scoring 分組，以及
  載入是否生效並附原因；由儀表板唯一的 Load 在 Research › Models 依測試分數排序時使用而寫入，
  §4.14）。

它只保存彙總資料，不含任何 case id。量測會在報告發佈後、寫出 `--outcome` 檔之前，立即追加它的
條目。若追加失敗，執行會以結束碼 1 停止（"measured, but the run did not finish"），說明報告已寫出、
這次執行不在紀錄中，並且不會登錄；它的 outcome 檔仍會寫出，內容為 `"complete": false`。若 outcome 檔
無法寫出，執行也同樣停止，並說明它已在紀錄中。若登錄自己的條目無法追加，登錄仍然成立，並以一行
`NOTE:` 說明。每個條目都從新的一行開始：被中斷的追加（當機、斷電）會留下一行不完整的最後一行，下一個
條目會先把它結束，而不是接在它後面，讀取時則把它指出為不是條目的一行。寫入失敗的追加會被收回，不留下
條目的任何部分；若連收回也失敗，錯誤訊息會說明檔案末端可能留有它的一部分。

**`measure_scorer.py` 的新選項。** 它們是為 M5 的工作執行器準備的，也可以在已設定 GPU 鎖的
shell 中手動使用（§4.11）。
`--expect-version-sha256` 與 `--outcome` 只適用於 `--cohort` 的執行（與 `--split` 一起給出時
結束碼為 2）；另外兩個兩者皆適用：

```bash
python scripts/measure_scorer.py --checkpoint <file>.pt --data-dir data/workspaces/<ws> \
    --cohort <label> --registry /srv/shepherd/cohorts [--record] \
    [--deterministic] [--expect-checkpoint-sha256 HEX] [--expect-version-sha256 HEX] \
    [--outcome FILE]
```

- `--deterministic` 會在 CUDA 啟動之前開啟幾項設定：PyTorch 的 deterministic algorithms（絕不是
  warn-only）、確定性的 cuDNN 並關閉其 benchmark，以及在該變數未設定時設定
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`。它絕不退回非確定性執行或改用其他裝置。沒有確定性實作的運算會
  讓這次執行被拒絕、不寫入任何東西；PyTorch 未記載為確定性的 `CUBLAS_WORKSPACE_CONFIG` 也是。
  manifest 會記錄這些設定，所以確定性執行是不同的鍵。PyTorch 只在相同平台、裝置與版本上承諾可重現；
  確定性模式防範已知的非確定性運算，並不為某個數字背書。
- `--expect-checkpoint-sha256` 與 `--expect-version-sha256`：當讀到的 checkpoint，或版本的
  `cohort.json`，具有其他 SHA-256 時（檔案在被選定之後改變了），在量測或寫入任何東西之前就拒絕這次
  執行。
- `--outcome FILE` 寫出一個小 JSON 檔。FILE 必須是新檔案，且所在目錄必須存在，否則在讀取任何東西
  之前就會被拒絕；因此先前執行留在該路徑的檔案絕不會被當成這次執行的 outcome，誤指的檔案也絕不會被
  覆寫。每次執行請指定一個新檔案（例如該工作自己的目錄中）。
  它在報告發佈後寫一次（`"complete": false`），結束時再寫一次（`"complete": true`）。內容包括報告的
  路徑、run id 與 digest、checkpoint 與版本的 digest、是否為確定性執行，以及在 `registration` 之下
  報告是否已登錄與原因。在報告之前就被拒絕的執行不會寫出 outcome 檔。這個檔說明的是這次執行做了什麼；
  紀錄仍是報告存放處與 ledger，「已保留」與「已登錄」從它們讀取。job runner 為它啟動的每次測試執行，
  在一個只屬於該執行的目錄中指定一個新的 outcome 檔（§4.11）。

---

## 4.10 Gradio 6.30.0 上的儀表板（自 M5 起，2026-10-20）

`/ui` 的儀表板現在執行於 **Gradio 6.30.0**（原為 5.29.1）。`pyproject.toml` 精確鎖定此版本，
`uv.lock` 只改了它強制帶來的部分：gradio-client 2.7.2、starlette 1.9.0 與 huggingface-hub
1.16.1，新增 brotli 1.2.0 與 hf-gradio 0.4.1，不再需要 aiofiles 與 ffmpy。其他套件的版本都沒有
改變。

**拉取這個版本後請執行 `uv sync --inexact`**（或重新執行 `deploy.sh` / `deploy.cmd`，兩者做的
是同一件事；`--inexact` 會保留 deploy 腳本在 lock 之外安裝的 PyG 擴充，見 `requirements.txt`）。
沒有退回 Gradio 5 的路徑：若仍裝著 5.29.1，服務會啟動、REST API 可用，但日誌會出現
`Failed to mount Gradio dashboard: mount_gradio_app() got an unexpected keyword argument 'theme'`，
而且沒有 `/ui`。

### 關閉了什麼，以及原因

儀表板以下列設定掛載，不論 `GRADIO_*` 環境變數怎麼設（`src/webui/app.py` 的 `mount_options`）：

| 設定 | 原因 |
|---|---|
| `run_history=False` | 否則 Gradio 6 會在瀏覽器中記錄每次呼叫的輸入與輸出，並提供存到 Hugging Face bucket 的選項。診斷的輸入是病人的表型。run history 頁面回 404 |
| `enable_monitoring=False` | 未設定時，`/ui/monitoring/summary` 會提供佇列的事件分析資料。現在回 403 |
| `mcp_server=False` | 否則會把儀表板的每個事件公開為 MCP 工具。相關 routes 不存在 |
| `ssr_mode=False` | 伺服器端渲染會啟動一個 Node.js 行程 |
| `analytics_enabled=False` | 設在儀表板本身，自 2026-10-12 起：不送使用分析，也不檢查版本 |
| `max_file_size="1mb"` | Gradio 每個上傳檔案的上限，作為 route guard 拒絕上傳路由之外的第二道防線（見下文；§4.13） |
| `footer_links=[]` | 自 M5-2 起：不顯示 Gradio 的頁尾（它的 API 頁面、「Built with Gradio」與它自己的 Settings——含儀表板配色沒有的深色模式與螢幕錄影） |

**CORS** 自 M5-1 起與 D8 一起收緊（§4.13）：**預設不允許任何跨來源的瀏覽器存取；管理操作仍由
後端的 D8 檢查。** 上層 app 保留它的 `CORSMiddleware`，但允許清單現在是空的（`src/api/main.py`）；
自 Gradio 6.25 起，上層 app 恰好使用這個類別時，掛載的 app 不再加上自己的 CORS 標頭，因此 `/ui`
也遵循同一政策。在 M5-1 之前，設定是 `allow_origins=["*"]` 與 `allow_credentials=True`，在
Starlette 1.9 下會回送每個請求的 `Origin` 並附上 `Access-Control-Allow-Credentials: true`：任何
網站的頁面，只要在持有服務的 HTTP 憑證或用戶端憑證的瀏覽器中打開，都能讀取 `/api/*` 與
`/ui/gradio_api/*` 的回應。現在不能了。

**被嵌入框架（framing）** 自 M5-3 起一律拒絕：每個回應——儀表板的與 API 的——都附上
`X-Frame-Options: DENY` 與 `Content-Security-Policy: frame-ancestors 'none'`
（`src/api/middleware/frame_guard.py`）。D8（§4.13）檢查的是請求從哪裡抵達，而不是它是從哪個頁面
發出的：在其他網站的 frame 中載入的儀表板，發出的是 D8 會允許的同源請求；沒有這項設定時，蓋在
惡意頁面上的隱形 frame 可能把操作者的點擊變成 Restart Backend 或 Load。沒有任何頁面嵌入儀表板，
包括它自己的頁面。

### route guard

Gradio 6.30 在每個掛載的 app 上加入兩個無法從上游關閉的 routes，並保留一個儀表板用不到的上傳
route。服務會在 Gradio 看到請求之前拒絕它們（`src/api/middleware/gradio_routes.py`）：

- **`/ui/gradio_api/file=<URL>`**（以及舊的 `file/<URL>`）。自 Gradio 6.20 起（issue #13593，
  PR #13596），伺服器會自己抓取 `http(s)://` 目標——不經過 `HTTPS_PROXY`——再以儀表板自己的位址
  回傳；對某些 URL，它會把遠端的 HTML 頁面以 inline 方式提供，就像儀表板自己的頁面一樣。任何
  URL scheme 的寫法都會被拒絕：大寫或大小寫混用、percent-encoded、`//host`。
- **`/ui/gradio_api/process_recording`**：把上傳的螢幕錄影轉成暫存檔，並可能執行 ffmpeg。Gradio 的
  設定面板（其螢幕錄影功能會送到這裡）自 M5-2 起已隨 Gradio 的頁尾一起隱藏；以其他方式送來的錄影
  同樣會被拒絕。
- **`/ui/gradio_api/upload`** 及其進度串流 `upload_progress`（自 M5-1 起）。儀表板沒有上傳
  控制項，而其他網站的頁面不需預檢就能把檔案 POST 到這裡：Gradio 每個請求最多會把 1000 個檔案
  寫進它的暫存目錄，而且從不刪除。

拒絕的回應是 `403`，附一行原因與 `connection: close`。請求本體不會被讀取，請求日誌會像其他
回應一樣記下這次拒絕。本機下載不受影響：Diagnosis 分頁的 CSV 與 Markdown 匯出照常下載，
`HEAD` 與 `Range` 請求也一樣。（自 M5-2 起，Training 頁面的 Export CSV 以文字經由頁面自己的
server function 傳來，由瀏覽器存檔；伺服器不為它提供檔案。）

### 儀表板的其他變更

- **Training Console 的圖表**外觀與以前相同（y 軸從 0 開始，每條數列維持自己的顏色）。Epoch
  Window 現在只篩選資料列，x 軸隨之調整。（M5-2 以一張可縮放、沒有 Epoch Window 的圖取代：
  §4.14，Research › Prepare › Training。）
- **Runtime Settings** 在打開分頁時會重新整理重啟鎖定狀態：Gradio 6 不會在尚未打開過的分頁中
  執行 timer。自 M5-3 起，頁面顯示期間它改由儀表板共用的慢速 timer 再次讀取，自己不再有 timer。
  不論如何，只要有 GPU 工作（訓練或測試執行）正在執行，後端都會拒絕重啟（§4.11）。

### 平台

已測試的後端：Linux x86_64（開發環境）與 Linux ARM64（homelab，由 reviewer 檢查）。
**Windows 後端尚未以 Gradio 6.30.0 測試**，在有 Windows 後端可測之前一律標為未測試；Windows
瀏覽器則包含在瀏覽器檢查中。

## 4.11 一次只跑一個 GPU 工作：GPU 鎖與 job runner（自 M5 起，2026-10-20）

受管理的訓練與評估工作彼此互斥：WebUI 與 API 啟動的工作，以及從 shell 執行的
`scripts/train_model.py` 與 `scripts/measure_scorer.py`。拿不到 GPU 的工作會被拒絕，並指出是誰
占用；不會排隊等候。

### GPU 鎖

每個部署一個鎖檔，不論使用哪個 workspace：

| 設定 | 鎖檔 |
|---|---|
| `SHEPHERD_GPU_LOCK_FILE=/絕對路徑` | 該檔案；所在目錄必須已存在（不會自動建立） |
| 否則 `SHEPHERD_WORKSPACES_ROOT=/絕對路徑的根目錄` | `<根目錄>/.shepherd-gpu.lock` |
| 兩者皆未設定 | **未設定鎖**：所有受管理的工作都會被拒絕（自 8242b47 的審查起）：CLI 在讀取任何東西之前以結束碼 1 結束，服務拒絕每一次啟動，並各自說明如何設定這把鎖 |

設定了卻無法使用的值（相對路徑、不存在的目錄、服務帳號無權進入的目錄底下的路徑）會拒絕所有
工作，而不是在沒有鎖的情況下執行，狀態也會說明原因（`gpu.lock.problem`）。

`train_model.py` 與 `measure_scorer.py` 在讀完命令列後立刻取鎖——在解析裝置或讀取任何輸入
之前——並持有到結束。鎖被占用時，它們會再嘗試約一秒（讓剛要結束的工作不會擋下下一個），
然後以結束碼 1 拒絕，且不寫出任何東西：

```
measure_scorer: refused: the GPU lock /srv/shepherd/workspaces/.shepherd-gpu.lock is held by a training job (train_model.py, pid 41234 on gb10, since 2026-10-20T09:12:03+00:00, as it described itself). One managed GPU job runs at a time: wait for it to end, or stop it where it was started. Nothing was written.
```

`train_model.py` 會在 `train_model: refused:` 之後記錄同樣的句子。WebUI 與 API 在啟動工作前會
先檢查鎖，並以同樣的占用者拒絕這次啟動（"Nothing was started."）；工作自己的行程隨後再自行
取鎖。

**請在服務與執行 CLI 的 shell 中設定同一個變數**（例如與 systemd unit 相同的
`Environment=` 行——unit 本身不讀取任何環境檔——或一個同時由 unit 的 `EnvironmentFile=` 與你的
shell 讀取的檔案）。兩個變數都看不到的 CLI 會被拒絕，結束碼 1，不寫出任何東西（`train_model:
refused: neither SHEPHERD_GPU_LOCK_FILE nor SHEPHERD_WORKSPACES_ROOT is set, so the GPU lock is not
configured ...`）；`setup_demo.py --train-model` 透過同一個訓練器訓練，同樣如此。**若服務指定了鎖檔
（`SHEPHERD_GPU_LOCK_FILE`），shell 必須指定同一個檔案**：只設定 `SHEPHERD_WORKSPACES_ROOT` 指的是
`<根目錄>/.shepherd-gpu.lock`，是另一個檔案，兩邊的工作就不會互斥。只有在沒有執行服務的機器上，
`SHEPHERD_GPU_LOCK_FILE` 才可以指定任何所在目錄已存在的絕對路徑。

**那個 shell 中也必須設定 `SHEPHERD_REGISTRY`**（自 M5 審查起，BACKLOG 37）：
`measure_scorer.py --cohort`、`--record` 與 `record_evaluation.py --report` 只在它所指名的
registry 中運作，它未設定時一律拒絕（§4.9）。`EnvironmentFile=` 的讀法和 shell 不同：systemd
會保留值中間的空白，也會保留不是出現在值開頭的引號（systemd.exec(5)），所以像
`SHEPHERD_REGISTRY=/srv/my cohorts` 這樣的一行，服務與 source 這個檔案的 shell 會得到兩個不同的
值。請使用不含空白與引號的路徑。兩者不同時會被拒絕，絕不會猜測：拒絕訊息會列出設定的值，以及它
解析成的路徑。

持有者在持有鎖期間會把自己的描述寫在 `<鎖檔>.json`（工作類型、命令、pid、主機、開始時間），
供上面的訊息使用。這個檔案只是描述：鎖是鎖檔上的 `flock`。被強制結束的工作留下的描述檔無害：
下一個持有者一取得鎖就會移除它；而所描述的行程已經結束的描述檔，絕不會被當成持有者說出來
（拒絕訊息會改為說明：鎖由繼承了其描述子的行程持有，或由尚未描述自己的新持有者持有）。**工作執行中絕不要刪除鎖檔**：新建的檔案會成為第二把鎖。

這把鎖保證什麼、不保證什麼：

- **不論持有者如何結束都會釋放**：正常結束或發生錯誤時由持有者自己釋放；遭 SIGTERM 或
  SIGKILL 結束時由核心釋放，因此當機後沒有殘留的鎖需要清理。
- **它是合作式的。** 不取鎖的 GPU 程式不會被它擋住。推論服務不取這把鎖，可能占用 GPU 記憶體；
  拿不到記憶體的測試執行會以執行失敗回報，不登錄任何東西。
- **繼承的檔案描述子。** 工作啟動的程式不會保有這把鎖（描述子設為 close-on-exec）。工作以 fork
  產生、卻沒有啟動新程式的行程——例如 `fork` 啟動方式下的 DataLoader worker——與工作共用這把鎖：
  工作正常結束時，鎖也會一併對該行程釋放；但工作被強制結束（或未釋放就結束）時，該行程會保有
  鎖直到它自己結束；這時拒絕訊息會說明所記錄的持有者已經結束。**服務啟動的工作各自是自己行程群組
  的 leader**（自 8242b47 的審查起），停止與 job runner 會結束整個群組，包括這些行程（見下）。你在
  互動式 shell 中啟動的 CLI 在那裡也是自己群組的 leader（job control）：`kill -TERM -- -<pid>`
  會連同它的 worker 一起結束它，而只用 `kill -9 <pid>` 則會讓 worker 自行結束。
- **鎖檔請放在本機檔案系統上。**
- **Windows** 使用 `msvcrt.locking` 鎖住檔案的第一個位元組；Windows 後端尚未測試。在那裡
  DataLoader worker 是以 spawn 而非 fork 產生，不會繼承鎖的 handle；系統釋放已結束行程的檔案鎖可能
  有延遲；停止時對工作的主控台行程群組送出 CTRL_BREAK_EVENT，沒有主控台、無法送出的後端會改用
  TerminateProcess，它只結束主行程。在 Windows 後端未經測試之前，不加入行程群組真正的對應物 Job
  Object。

### job runner

Training Console 與訓練 API 背後的服務現在執行兩種工作，各自使用自己的腳本：訓練
（`train_model.py`）與測試執行（在已登錄的 cohort 版本上執行 `measure_scorer.py`，與操作者
執行的是同一個 CLI）。一次一個工作：

- **正在執行或仍在停止中的工作占用 GPU**，不論類型。停止仍在等待其行程結束時的訓練啟動，現在
  會被拒絕（以前會被接受，可能在正被停止的行程旁再啟動第二個行程）。
- **停止只作用於自己的工作類型**：訓練的停止永遠不會停止測試執行，反之亦然。它先對工作的主行程送
  SIGINT（它的正常停止），30 秒後對工作的整個行程群組送 SIGTERM，再過 10 秒對群組送 SIGKILL，並在
  工作沒有任何行程留下之後才回應：每個工作是自己行程群組的 leader，所以它 fork 出、與它共用 GPU 鎖的
  DataLoader worker 會跟著結束；主行程結束後，runner 先結束群組中剩下的行程，才說明工作如何結束並
  釋出 GPU（自 8242b47 的審查起：以前 worker 可能在停止回答 "stopped" 之後仍存活並保有鎖）。成為
  zombie 的 worker（PID 1 從不回收的容器中）不持有任何東西，不會占住 GPU。
  `POST /api/v1/training/stop` 等待期間不再卡住 API 的其他部分與儀表板。**行程已經自行結束後
  才送達的停止**（其結果仍在讀取中）會被拒絕（`409`，"had already ended on its own ...;
  nothing was stopped"），且不改變任何東西：失敗仍是 `failed`，並保留其訊息。
- **工作如何結束：** 訓練為 `completed`、`failed` 或 `stopped`（與以前相同）；測試執行為
  `completed`、`failed` 或 `cancelled`，只有在取消確實結束了它時才是 `cancelled`（取消送達時
  正好自行結束的執行，保留它自己的結束狀態）。因訊號而失敗時會說明是哪個訊號（例如記憶體不足被系統
  結束時為 `Process ended by SIGKILL`）。
- **測試執行結束後，它的報告從磁碟讀取。** 每次測試執行都把它的 outcome 檔
  （`measure_scorer.py --outcome`，§4.9）寫進一個只屬於這個工作的目錄，其中寫明它發布的報告與那些
  位元組的 digest；這份報告是否保留、是否登錄，接著由登錄流程本身的 helper 以這個 digest 從
  report store 與 ledger 讀取，絕不從結束碼推斷（狀態的 `outcome` 提供 `kept`、`registered`、
  附原因的 `summary`，以及執行本身的 `run_outcome`）。**沒有 outcome 檔就表示沒有發布任何報告**，
  除非執行期間 store 中出現了它的 checkpoint 的執行目錄（在發布報告與寫出 outcome 之間被停止的
  執行），這時會以同樣的方式讀取，並說明這一點。執行讀取完畢後，工作目錄會被移除。取消後，狀態會
  說明留下了什麼；遭 SIGTERM 或 SIGKILL 結束時，還會指出暫存檔（`*.tmp`）可能留在哪裡：執行目錄，
  以及正在登錄報告的執行所屬 checkpoint 的目錄（其 ledger 所在處）。
- **Runtime Settings 的 Restart Backend** 在任何工作執行中時都會被拒絕，不論是訓練或測試執行
  （提示文字現在是 "A GPU job (training or a test run) is in progress"）。
- **停止後端就會停止它的工作**（owner 的決定，2026-10-22）。因為工作是自己行程群組的 leader，在
  執行後端的終端機按 Ctrl-C、終端機掛斷，或只對後端送 SIGTERM，都不再直接到達工作本身；改由後端在
  訊號到達時就開始像上面的停止一樣停止它——適用於從終端機啟動的後端（launcher 會在 Ctrl-C 時等待伺服器
  結束，或手動執行的 uvicorn），以及在 systemd 之外收到 SIGTERM 的後端；在 systemd 下，停止 unit
  本來就會對服務的每個行程送出訊號。開著的儀表板分頁不會拖延它：uvicorn 結束前會等待每個開著的連線，
  而每個分頁都保有一個（它的 session heartbeat），所以後端在同一個訊號就結束這些串流。其他不會關閉的
  連線（例如一個永遠沒送完請求的客戶端）會讓後端繼續等待（"Waiting for connections to close"），
  但工作已經停止；第二次 Ctrl-C 會結束這個等待，而後端仍會等工作的停止完成才退出。後端被直接結束
  （SIGKILL、記憶體不足被系統結束）會讓工作繼續執行並保有鎖；GPU chip 會指出它；請用
  `kill -INT <pid>` 或 `kill -TERM -- -<pid>` 停止它。**不要以 `--reload` 執行後端**：uvicorn 的
  reloader 每次儲存原始碼都會重啟伺服器，每次重啟的關閉程序都會停止正在進行的訓練。
  `setup_demo.py` 印出的命令不再含 `--reload`，`python -m src.api.main` 也不再自動重新載入。
- `GET /api/v1/training/status` 新增 `job_id`、`end_time`、`exit_code` 與 `gpu`：占用 GPU 的
  工作（任一類型），以及鎖的狀態（`configured`、`path`、`source`、`held_by`、`problem`）。占用
  GPU 的測試執行會顯示在這裡，絕不會顯示成訓練。

自 M5-2 起，Research › Test 透過這個 job runner 從儀表板啟動與取消測試執行（§4.14）；沒有任何 API
route 會啟動測試執行。兩個 CLI 的
結束碼都沒有改變：鎖被占用屬於「寫出任何東西之前的拒絕」。

---

## 4.12 伺服器設定與伺服器的清單（自 M5-1 起，2026-10-20）

M5 的控制項只作用於伺服器列出的項目，從不作用於請求指定的路徑（`docs/working/PLAN_TEST_RESULTS.md` 的 D7 與 D8；D8 決定誰可以觸發操作者控制項，§4.13）。伺服器列出什麼，以及
操作者控制項所依賴的部署模式，都來自**服務的環境變數**。請在啟動服務的地方設定它們（執行
`launch_shepherd.sh` 的 shell，或 systemd unit 中的 `Environment=` 行），然後重啟服務。每個
設定都在使用時讀取。沒有任何請求、表單或儲存的 UI 檔案能設定它們。缺少或被拒絕的設定會關閉
需要它的功能，並說明原因。**沒有任何東西會退回預設目錄**，repository 的 `data/workspaces/`
也一樣。

| 變數 | 值 | 未設定或被拒絕時 |
|---|---|---|
| `SHEPHERD_DEPLOYMENT_MODE` | 恰好是 `M1`（`m1`、` M1` 或 `M2` 都算其他值） | 唯讀：每個操作者控制項都被拒絕（D8，§4.13） |
| `SHEPHERD_WORKSPACES_ROOT` | 一個已存在目錄的絕對路徑；其子目錄就是各個 workspace | 不列出任何 workspace |
| `SHEPHERD_REGISTRY` | cohort registry（§4.9），即各指令以 `--registry` 接收的目錄：絕對路徑、已存在、位於 repository 之外；服務與各指令唯一從中登錄報告、在其中進行 `--cohort` 量測的 registry | 不列出任何 cohort 版本；每次登錄與 `--cohort` 量測都被拒絕 |
| `SHEPHERD_INTAKE_DIR` | 位於 repository 之外的已存在目錄，你把要匯入的 `shepherd-cases/1` 檔案放在這裡；絕對路徑 | 不提供任何匯入來源 |
| `SHEPHERD_TRAINING_EVIDENCE_DIRS` | 選填：R10 證據目錄，與 `--training-evidence` 接收的相同，以 `:` 分隔（Windows 為 `;`）；每一個都必須是已存在的絕對路徑 | 無；只要有一個項目不合格，整個設定就被拒絕 |
| `SHEPHERD_ORDERING_ROLES` | 選填：提供依測試分數排序（D4）的 cohort 角色，以逗號分隔 | 無；`institutional_acceptance` 即使列出也會被忽略（`docs/working/PLAN_TEST_RESULTS.md` §4.9 問題 1） |

```bash
export SHEPHERD_DEPLOYMENT_MODE=M1
export SHEPHERD_WORKSPACES_ROOT=/home/you/SHEPHERD-Advanced/data/workspaces
export SHEPHERD_REGISTRY=/srv/shepherd/cohorts
export SHEPHERD_INTAKE_DIR=/srv/shepherd/intake
```

- **`M1`** 是 SPEC_4 的單一 workspace 部署：在本機上，或經由 SSH forwarding 存取，由 owner 或
  指定的技術人員操作（`docs/working/PLAN_TEST_RESULTS.md` §4.9 問題 3）。它是操作者控制項檢查的條件之一，而不是授權。其餘條件由
  D8 在處理每個請求時檢查。不要在 reverse proxy 後面、或在其他人也能登入的機器上設定它。
- **部署模式由 D8 套用**（§4.13）：每個操作者控制項在處理請求的當下都會讀取它；未設定或任何
  其他值時，每個控制項都被拒絕，部署對它們而言是唯讀的。診斷、狀態頁面與唯讀清單不受影響。
- **workspace 根目錄可以位於 repository 內**，一般使用的 `data/workspaces/` 就是如此，但必須
  由你明確指定。相對路徑、不存在的目錄與檔案都會被拒絕。
- **GPU 鎖跟著這些設定走**（§4.11）：設定了根目錄而沒有 `SHEPHERD_GPU_LOCK_FILE` 時，鎖是
  `<根目錄>/.shepherd-gpu.lock`。執行正式 CLI 的 shell 需要相同的設定（若服務指定了鎖檔，就是同一個
  檔案）；兩者都沒有設定時，每個訓練與測試執行都會被拒絕。
- **registry 與 intake 目錄存放病人層級的資料。** 兩者都遵循 registry 的規則（§4.9）：相對
  路徑、不存在的目錄，以及解析後落在 repository 內的路徑（包括經由 symlink）一律拒絕。命令列的
  `--registry` 與 `SHEPHERD_REGISTRY` 由同一個函式檢查。請把兩者建在只有執行服務與指令的帳號
  讀得到的地方，例如 `mkdir -m 700`。
- **服務帳號無法到達的目錄**（它無權進入的上層目錄）會以「無法檢查」（cannot be checked）
  拒絕並附上錯誤，其功能關閉。在 registry 內，服務讀不到的 `versions/` 會讓 cohort 清單附上
  這個原因、不列出任何項目；讀不到的版本、`reports/` 或 run 目錄會附原因列出，清單的其餘部分
  照常列出。
- **證據目錄**：空的項目（重複或結尾的分隔符）會被略過，絕不會被當成工作目錄讀取。重複的項目
  只保留一次。

### 伺服器列出什麼

每個項目都以名稱與 digest 識別，從不以路徑識別：請求指定的是 workspace、run 與檔名，或 label，
伺服器會在自己的清單中重新找到這個名稱。

- **Workspace**（`GET /api/v1/workspaces`）：根目錄下的每個子目錄，依名稱列出。以 `.` 開頭的
  名稱會被略過。指向根目錄之外的 symlink 子目錄會列出，但不提供使用。每個 workspace 顯示：
  - 它的**資料集指紋**：其 `split_manifest.json` 完整的 SHA-256。manifest 綁定了圖檔案、兩個
    樣本檔與資料集紀錄，因此兩次不同的資料集建置一定不同。顯示時可以縮短為前 12 個十六進位
    數字；比較時一律使用完整的 64 個；
  - 它的資料集紀錄的綁定狀態與來源，兩者分開呈現（§4.4）；
  - 它的**建置時間**，來自 `workspace_build.json`（見下文），或附原因的「未記錄」；
  - 它的各來源宣告的版本，取自資料集紀錄（合成 workspace 沒有）；
  - 它的 train 與 val 樣本數；
  - 它是否**可訓練**，以及不可訓練的原因：沒有可讀取的 `split_manifest.json`、資料集紀錄未
    通過驗證，或 manifest 綁定的某個檔案不存在或無法檢查。清單不計算任何圖檔案的雜湊。訓練
    執行時會以實際讀取的位元組檢查每個檔案，仍可能拒絕。
- **某個 workspace 的 checkpoint**：某個已列出 workspace 的 run 版面中的 checkpoint，所有
  架構或單一架構，最新的 run 在前。每個檔案只讀取一次，且只載入權重，並顯示：
  - 所讀位元組的 SHA-256；
  - Trainer 的驗證數字：trainer 的 top-20 選擇指標，絕不是測試結果；檔案沒有記錄時為無；
  - checkpoint 記錄的資料集指紋，以及它是否就是這個 workspace 目前的指紋；
  - 無法讀取的檔案會附上原因（包括目標已不存在的連結）。沒有任何檔案被略去；
  - 解析後落在 workspace 之外的檔案（以 symlink 指向 workspace 之外的 run 或 checkpoint）只
    附上這個原因：它絕不會被開啟，因此不顯示 digest、大小或紀錄。

  每個項目都以清單中的名稱指稱 workspace：對 `current -> builds/2026-10-20` 而言，它寫的是
  `current`。列出時會開啟每個 checkpoint，SPEC_4 將此歸類為 checkpoint 檢視（C3），而 D8 只在
  操作者控制項所在之處提供它：它以 `GET /api/v1/workspaces/{name}/checkpoints` 提供（例如
  `?architecture=gat` 只列一種架構），位於 D8 的檢查之後（§4.13），被拒絕的請求不會開啟任何檔案。
- **既有的 checkpoint 清單**（`GET /api/v1/training/checkpoints`；自 M5-2 起，Training
  頁面的 parent 清單是上面那份依 workspace 的清單）以同樣的方式讀取每個檔案。每個項目新增
  `sha256`、`validation`、`dataset_fingerprint` 與 `metadata_problem`，全部來自同一次讀取。以前它載入每個檔案卻不保留
  digest，而且載入錯誤時一聲不響地略過。讀取時已經不存在的檔案（目標已被移除的連結）沒有
  大小或時間可顯示，會從這份清單中略去；它不再讓整份清單失敗。自 M5-1 起，D8 像檢查控制項
  一樣檢查它（§4.13）。
- **Cohort 版本**（`GET /api/v1/cohorts`）：registry 中的每個版本，以 label 及其 `cohort.json`
  的 SHA-256 識別，附上它的角色，以及這個角色是否提供排序。它也顯示版本的計數、所綁定的圖與
  資料集紀錄，以及匯入時間。清單絕不顯示病例識別碼或排除清單：D7 把它們留在這台機器上的
  registry 中。中斷的匯入會附原因列出。每個版本的報告庫以 run id 列出，每個 run 都會說明它的
  報告是否保留、是否已登錄（`kept`、`registered`，各附原因，以及報告的 digest 與紀錄的序號），
  來自登錄流程本身對報告庫的讀取，以及對報告所記錄、它量測的 checkpoint 旁邊那份 ledger 的讀取。
  那份 ledger 載有這份報告時，`registered` 為 `registered`，否則為 `cannot_be_checked`，絕不是
  `not_registered`：登錄在同一份權重的其他副本旁邊的紀錄（`record_evaluation --checkpoint` 接受
  任何位元組相同的檔案）不會被查找，原因會指出查過的是哪一份 ledger。以相對路徑記錄 checkpoint 的
  報告，或無法讀取的報告，沒有可查的 ledger：這時 `registered` 同樣為 `cannot_be_checked`，並說明
  原因。執行之後被移走的 checkpoint，會在執行讀取它的位置查找。有 `report.json` 的每個 run 也帶有
  它的報告所屬 key 的 D5 狀態（`contest`：`uncontested`；`contested`，附上不同的欄位與 run；
  `cannot_be_checked`，絕不能當成 uncontested 解讀），與 Research › Test 對那個 run 顯示的狀態相同，
  但不含原因本身的文字，因為它可能指出伺服器路徑（自審查修正的根因回合起）。沒有 `report.json` 的
  run 目錄（中斷、取消或被拒絕的執行），以及狀態沒有被讀取的 run（它的 `problem` 說明原因），
  `contest` 為 `null`。報告無法讀取、或無法得知其 key 的 run 是 `cannot_be_checked`；可以讀取但無法
  比較的 run 是 `cannot_be_checked`，除非它的 key 底下的報告已經不一致（這時是 `contested`）。無法
  掃描的報告庫中的每個 run 也是 `cannot_be_checked`，報告在報告庫的掃描與其狀態的讀取之間被替換的
  run 也是：一個 run 的狀態與它的 `contest` 永遠描述同一份位元組。每個請求會把每個列出版本的報告庫
  掃描一次。

### 建置時間：`workspace_build.json`

自 M5-1 起，每次經由 workspace writer 的建置（`scripts/build_knowledge_graph.py
--generate-samples`、`scripts/setup_demo.py`、probe）都會在 `split_manifest.json` 旁、於其後
寫入 `workspace_build.json`：

```json
{
  "built_at": "2026-10-20T08:15:00Z",
  "code_revision": {"state": "recorded", "revision": "84371dd...", "dirty": false, "modified_tracked": 0, "untracked": 0},
  "schema_version": 1,
  "split_manifest_sha256": "...",
  "what_this_is": "when this workspace was built and from which code revision; descriptive, not bound by split_manifest.json"
}
```

- `built_at` 為 UTC。`code_revision` 與訓練存為 checkpoint producer 的紀錄相同：commit，以及
  checkout 是否有未提交的變更，不含任何路徑。
- **描述性的，不受 manifest 綁定。** manifest 不會提到它，因此兩次位元組完全相同的建置保有
  同一個資料集指紋，不論建置時間為何。
- **它指明自己描述的是哪一份 manifest。** 放在另一份 manifest 旁的紀錄會顯示「未記錄」，絕不
  顯示另一次建置的時間。這發生在寫完 manifest 後中斷的重建之後，或是由早於 M5-1、不寫這個檔案
  的版本所做的重建之後。
- **M5-1 之前建置的 workspace 沒有這個檔案。** 它的建置時間顯示為「未記錄」，並附上各來源
  宣告的版本；絕不使用檔案修改時間。不需要為此重建。原因寫的是「由 M5-1 之前的版本建置，或
  其建置在寫入這個檔案之前就停止」：在 manifest 與這個檔案之間停止的首次建置看起來也一樣。
- **只建圖的建置**（沒有 `--generate-samples`）既不寫 manifest，也不寫這個檔案。它的建置時間
  顯示為「未記錄」，原因是沒有可描述的 `split_manifest.json`。

---

## 4.13 操作者控制項、綁定位址與 CORS（自 M5-1 起，2026-10-20）

M5 在 SPEC_4 的部署模式 **M1** 下執行：由 owner 或指定的技術人員在本機上操作服務，或從其他地方
經由 SSH forwarding 操作（`docs/working/PLAN_TEST_RESULTS.md` §4.9 問題 3）。三件事一起改變，部署可能需要對每一件採取行動：

- 除非 D8 允許該請求，否則操作者控制項會被拒絕；
- 服務預設綁定 `127.0.0.1`，不再是 `0.0.0.0`；
- 其他網站的頁面無法讀取服務的回應（CORS）。

### 操作者要設定什麼

在啟動服務的地方設定 `SHEPHERD_DEPLOYMENT_MODE=M1`（§4.12；systemd unit 中是被註解掉的那行
`Environment=`），然後重啟服務。未設定或任何其他值時，每個操作者控制項都被拒絕，部署對它們而言是
唯讀的。診斷、查詢、狀態頁面與唯讀清單照常運作。

### 哪些控制項會被檢查

| 位置 | 控制項 |
|---|---|
| API | `POST /api/v1/pipeline/reload`、`POST /api/v1/pipeline/config`、`POST /api/v1/training/start`、`POST /api/v1/training/stop`、`GET /api/v1/training/checkpoints`、`GET /api/v1/workspaces/{name}/checkpoints` |
| 儀表板 | Diagnosis：Change model…（其 checkpoint 清單與它的 Load；自 M5-2 起）。⚙ Settings（Runtime Settings）：Restart Backend 與 Apply Settings。Research › Training：Start、Stop（它的確認會讀取該 run 最後的 checkpoint）、Continue…（它列出 parent）、parent 清單的 Refresh、Check checkpoint 與 Continue Training |

兩份 checkpoint 清單與 Check checkpoint 會開啟 checkpoint 檔案，SPEC_4 將此歸類為 checkpoint 檢視
（C3），因此它們與控制項一樣受檢查。為了顯示而開啟 checkpoint 的讀取也一樣：Diagnosis 卡片的
「trained on」一行（所服務的 checkpoint）與 Training 頁面的 run 標題（該 run 的 checkpoint）；
唯讀的瀏覽器會看到 "not shown · reading the checkpoint is an operator control"。Research › Test（§4.14）對它的匯入（intake 清單、Preview 與
Commit）、Start test、Cancel、Register this report、Add note 與模型清單使用同一個檢查，Research ›
Models 的 checkpoint 清單與 Load（§4.14）也一樣。其餘一切保持開放：診斷、HPO 與疾病查詢、pipeline 狀態與儲存的 UI 路徑、訓練狀態與指標、
workspace 與 cohort 清單、已記錄的結果，以及系統頁面。

### 什麼時候允許一個請求

每個控制項都檢查它正在處理的請求，在處理的當下檢查，只有在下列條件全部成立時才允許：

1. `SHEPHERD_DEPLOYMENT_MODE` 恰好是 `M1`；
2. 請求抵達這台機器的 loopback 位址：由伺服器自己的 socket 決定，絕不採用用戶端自稱的位址；
3. 請求不帶 proxy 會加上的標頭（`Forwarded`、`X-Forwarded-For`、`X-Forwarded-Host`、
   `X-Forwarded-Proto`、`X-Real-IP`、`X-Original-Forwarded-For`、`CF-Connecting-IP`、
   `True-Client-IP`）：M1 沒有 reverse proxy；
4. 請求的目標是 `localhost`、`127.0.0.1` 或 `[::1]`，任何埠號皆可（它的 `Host` 標頭），以防範
   DNS rebinding；
5. 若帶有 `Origin`，它就是儀表板自己的位址（`http://` 或 `https://`，相同的主機與埠號）。
   瀏覽器在每個 `POST`，以及每個由腳本發出的跨來源請求中，都會送出頁面的來源；
6. 若帶有 `Sec-Fetch-Site`，它的值是 `same-origin`（儀表板自己的頁面）或 `none`（你在任何網頁
   之外發起的請求：網址列、書籤、從其他程式打開的連結）。其他網站的頁面以圖片、腳本或連結發出的 `GET`，瀏覽器**不會**附上 `Origin`，而目前的
   瀏覽器（Chrome 與 Edge 76、Firefox 90、Safari 16.4 及更新版本）會在這個標頭說明請求從哪裡來：
   `cross-site`，或對同一主機的其他埠號是 `same-site`。因此其他網站的頁面無法從目前的瀏覽器觸發
   控制項，兩份 checkpoint 清單也包括在內。

無法讀取抵達資訊的請求一律拒絕。檢查在後端：反灰的按鈕只是顯示，直接呼叫 API，或透過 Gradio 的
HTTP API（`/ui/gradio_api/...`，像 `gradio_client` 那樣）直接呼叫儀表板事件，都會遇到同一個檢查。
儀表板的控制項在行程內呼叫服務，從不透過 HTTP 呼叫 API。儀表板的 Load（Change model… 與
Research › Models）在行程內執行 API 的 reload 函式，與 API 的 reload 在同一處執行，因此和以前
一樣會排在正在執行的診斷之後。180 秒後仍在執行的重新載入不再被顯示為失敗：頁面顯示 "The page
stopped waiting after 180 s; the server may still finish the reload." 並提供 Read again；若回應
之後才抵達，則顯示 "The reload finished after the page stopped waiting."。

**拒絕時**，API 回應 `403`，其 `error` 以可據以行動的文字說明哪個條件未通過，並以 "Nothing was
done." 結尾；絕不重複任何標頭的值。儀表板在控制項顯示結果的地方顯示同一句話（run 清單無法容納
句子，因此會被清空，這句話以通知的形式出現）。例如：

```
Refused: This request did not arrive on a loopback address of this machine. The operator controls work only on the machine itself, or through an SSH forward to localhost (for example ssh -L 8264:localhost:8264 <server>, then http://localhost:8264/ui). Nothing was done.
```

**這個瀏覽器是否啟用了？** `GET /api/v1/system/operator-access` 針對發出詢問的請求回答：
`enabled`、決定結果的 `condition`，以及它的 `reason`。它是唯讀且開放的；顯示它並不是控制本身。

**被 D8 拒絕的重新載入不改變任何東西。** 以前，任何人的重新載入請求，不論是否被拒絕，都會讓之後的
每次診斷從 demo 回答變成「已設定但未提供服務」（503）。

**不宣稱的事。** loopback 不是授權：任何能到達這台機器 loopback 的人，包括經由他自己的 SSH
forward，都會通過。因此 M1 指的是一個經過驗證、由 owner 或指定技術人員操作的單一 workspace 部署。
同一台主機上、不加任何轉送標頭的 reverse proxy 由本指南排除，而不是被偵測：不要在 M1 服務前面
放置這樣的 proxy。在這類 `GET` 上既不送 `Origin` 也不送 `Sec-Fetch-Site` 的舊瀏覽器，條件 6
攔不住：請用目前的瀏覽器操作儀表板。

### 從其他機器操作：SSH forwarding

把本機的一個埠轉送到服務的 **localhost**，而不是它的網路位址：

```bash
ssh -N -L 8264:localhost:8264 you@gb10
# 然後在你自己的機器上打開 http://localhost:8264/ui
```

轉送的連線會以 `localhost:8264` 為目標抵達伺服器的 loopback，因此控制項可以使用。
`ssh -L 8264:<伺服器的區域網路位址>:8264` 會抵達區域網路位址而被拒絕，而且在新的綁定設定下那裡
根本沒有服務在監聽。使用其他本機埠也可以（`-L 9000:localhost:8264`，然後打開
`http://localhost:9000/ui`）。

### 綁定位址：預設 `127.0.0.1`

launcher（`launch_shepherd.sh`、`shep_launch.py`）、systemd unit 與 `python -m src.api.main`
在沒有另外指定時都綁定 `127.0.0.1`，launcher 會印出它綁定的位址：

```
[SHEPHERD] Listening on          : 127.0.0.1:8264 (loopback: this machine only)
```

**綁定任何其他位址都會被拒絕，什麼都不會啟動**（launcher 在安裝任何東西之前就以 2 結束），除非
設定了 `SHEPHERD_ACCEPT_UNAUTHENTICATED_NETWORK_BIND=1`，因為服務沒有任何認證。這個旗標是**風險承擔，
而不是控制**：它不認證任何人，也從不解除 D8，因此在網路綁定下，操作者控制項對每個不是從 loopback
抵達的請求仍然拒絕，而診斷與唯讀頁面對任何能到達該位址的人開放。launcher 會印出一行說明這點的
`WARNING:`。launcher 一律以 `--no-proxy-headers` 啟動 uvicorn，並放在其命令列的最後，因此 uvicorn
回報的用戶端位址就是連線本身的位址。

launcher 以 uvicorn 自己的方式判讀 uvicorn 會在哪裡監聽：交給它的 socket（`--fd`，systemd 的
socket unit 會這樣傳入）優先於 unix socket（`--uds`），再優先於最後一個 `--host`；每一項都取自
命令列，沒有時取自 `UVICORN_FD` / `UVICORN_UDS`。繼承來的 socket 的位址不會被讀取，因此算作網路
綁定，需要這個旗標。unix socket 只有本機到得了，不需要旗標；經由它的操作者控制項一律被拒絕（D8
在那裡看不到 loopback 位址）。這兩種情況下，launcher 都印出 socket 而不是 URL，也不開啟瀏覽器。

launcher 與 `python -m src.api.main` 會把它們綁定的主機以 `SHEPHERD_BIND_HOST` 傳給應用程式，
應用程式啟動時在提供服務之前對它套用同一個規則（被拒絕時啟動中止，uvicorn 結束）。它們每次都會
設定它；這不是由你設定的。

不涵蓋：手動啟動的 uvicorn（`uvicorn src.api.main:app --host 0.0.0.0`），應用程式看不到它的綁定，
它也不會把任何主機傳給啟動程序。請使用 launcher。

**依賴 `0.0.0.0` 的部署如何遷移。** 在 M5-1 之前，launcher 與 unit 綁定所有介面。這次更新之後：

- 原本打開 `http://<伺服器的區域網路位址>:8264/ui` 的人會連不上。請給他們一個 SSH forward
  （見上文）：這就是 M1 的方式；
- 如果區域網路必須繼續使用診斷，請設定 `--host 0.0.0.0`（在 unit 的 `ExecStart` 中，或放在
  launcher 命令列的 `--` 之後），**並且**在同一個環境中設定
  `SHEPHERD_ACCEPT_UNAUTHENTICATED_NETWORK_BIND=1`。沒有這個旗標時 launcher 會拒絕啟動（結束狀態
  2），在 systemd 下 unit 會停止而不重啟（`RestartPreventExitStatus=2`；`journalctl --user -u
  shepherd` 會顯示原因）。這時操作者
  控制項只能在本機上或經由 SSH forward 使用；
- 透過 HTTP 重新載入 pipeline 的腳本需要服務處於 M1 模式，並且必須呼叫
  `http://127.0.0.1:<port>`。`scripts/measure_served_pipeline.py` 因此在它自己啟動的服務中設定
  `SHEPHERD_DEPLOYMENT_MODE=M1`。

### CORS：沒有跨來源的瀏覽器存取

**預設不允許任何跨來源的瀏覽器存取；管理操作仍由後端的 D8 檢查。** 不論瀏覽器使用哪個位址，
`/api` 與儀表板的回應都不會對其他來源帶上 `Access-Control-Allow-Origin`，每個跨來源的預檢請求
（preflight）都回應 400。同源的使用（儀表板本身、`curl`、腳本、`gradio_client`）不受影響。

CORS 只決定頁面能讀取什麼。不需要預檢的跨來源請求仍會抵達伺服器：這正是操作者控制項由 D8 而不是
由 CORS 檢查的原因。基於同樣的理由，Gradio 的上傳路由在儀表板前面就被拒絕，請求本體不會被讀取
（§4.10 的 route guard）：儀表板沒有上傳控制項（匯入來源來自 `SHEPHERD_INTAKE_DIR`），否則其他網站
的頁面每個請求最多能把 1000 個檔案寫進 Gradio 的暫存目錄。Gradio 每個檔案 1 MB 的上限保留作為
第二道防線；它限制的是每個檔案，而不是檔案的數量。

## 4.14 在儀表板中找到方向（自 M5-2 起，2026-10-20）

`/ui` 的儀表板改為兩層導覽，頁首有兩個狀態標籤（chip），每個欄位共用同一個說明控制項。操作者
要設定的東西都沒有改變；改變的是各項功能所在的位置。

### 導覽

| 位置 | 內容 |
|---|---|
| **Diagnosis (clinical)** | 病患表型、結果與選中候選的細節，使用中的模型顯示在一張卡片上（見下文）。它是第一個畫面 |
| **Research (pre-clinical)** › Prepare › **Datasets** | 伺服器的 workspace，唯讀（見下文） |
| **Research (pre-clinical)** › Prepare › **Training** | run bar、下一次 run 的參數、Continue… 與附圖表的 run 檢視（見下文）。它是 Research 下最先顯示的頁面 |
| **Research (pre-clinical)** › Evaluate › **Test** | 從伺服器的 intake 目錄匯入 cohort 版本、執行測試、保留並登錄其報告（見下文） |
| **Research (pre-clinical)** › Evaluate › **Models** | 伺服器為某個 workspace 列出的 checkpoint、它們已記錄的測試結果、依測試分數排序與 Load（見下文） |
| 頁首右側的 **⚙ Settings** | Runtime Settings，與以前相同。對唯讀的瀏覽器，Apply Settings 與 Restart Backend 會呈灰色不可用，原因寫在它們下方 |

- 選中的階段（Diagnosis 或 Research）以填色加粗體顯示；選中的研究頁面以底線加粗體顯示；
  Settings 被選中時有淺色填色與底線。Prepare 與 Evaluate 是分組標籤，不是按鈕。
- 不會有任何分頁被收進選單。視窗較窄或放大倍率較高時，兩個 chip 與 Settings 會移到頁首的第二
  行，研究頁面的分頁也會換行。
- 瀏覽器分頁的標題會說明目前位置，例如 `Training · Research — SHEPHERD-Advanced`。重新載入
  頁面會再次打開 Diagnosis：網址不保留位置。
- 視窗至少 1400 × 760 CSS 像素時，頁面本身不捲動；某個區域的內容超出空間時，在它自己的面板
  內捲動。視窗較窄或直向時，各區域上下堆疊，由頁面捲動。佈景主題
  最小的文字為 14 px。
- 無論系統或瀏覽器要求深色模式，儀表板一律以它的淺色配色顯示（它的配色沒有深色版本，而
  Gradio 的深色模式會讓頁面上的文字無法閱讀）；網址會加上 `?__theme=light`。Gradio 自己的頁尾
  （它的 API 頁面、「Built with Gradio」與它自己的「Settings」）不再顯示。
- 每次顯示某個研究頁面時都會重新讀取它：選取它時，以及在它是目前顯示的頁面時回到 Research
  時。這時其他研究頁面不會被讀取。

### 頁首的兩個 chip

兩者都會針對你自己的瀏覽器重新讀取：頁面載入時、每 5 秒，以及你每次使用操作者控制項之後。
它們只是顯示：每個控制項仍在後端檢查它自己的請求（§4.13）。

**Operator controls** 顯示 D8 對這個瀏覽器的請求的回答：

| Chip | 意義 |
|---|---|
| `🔒 Operator controls: on (this machine)` | D8 允許這個瀏覽器的請求 |
| `Read-only: deployment mode not set to M1` | `SHEPHERD_DEPLOYMENT_MODE` 不完全等於 `M1` |
| `Read-only: request not local` | 請求不是經由本機的 loopback 抵達：請使用轉發到 `localhost` 的 SSH forward |
| `Read-only: forwarded by a proxy` | 請求帶有 proxy 會加上的標頭 |
| `Read-only: not addressed to localhost` | 瀏覽器使用的主機名稱不是 `localhost`、`127.0.0.1` 或 `[::1]` |
| `Read-only: sent from another origin` / `sent from another site` | 請求來自另一個頁面 |
| `Read-only: request not readable` | 伺服器無法讀取請求是如何抵達的 |

它的 ⓘ 列出每一個操作者控制項、如何使用它們（§4.13 的 SSH forward），在唯讀時並附上 D8 給出
的完整原因。它從不重複任何標頭的值。

**GPU jobs** 顯示哪個受管理的工作佔用唯一的 GPU 名額（§4.11）：

| Chip | 意義 |
|---|---|
| `GPU jobs: none` | 沒有受管理的訓練或評估工作在執行。它從不說「free」：這個鎖是協作式的，常駐的推論服務不在鎖的範圍內，仍可能佔用 GPU 記憶體 |
| `GPU job: training (job runner)`、`GPU job: test run (job runner)` | 本服務啟動的工作；停止中時加上 `· stopping` |
| `GPU job: train_model.py (pid 41822, since 09:58 UTC)` | 某個不是本服務啟動的行程持有 GPU 鎖（正式的 CLI，或同一把鎖上另一台伺服器的 job runner）；chip 只顯示其持有者紀錄所寫的內容：指令、pid 與開始時間 |
| `GPU job: lock held, holder not described` | 鎖被持有，但沒有任何仍在執行的行程被描述為持有者 |
| `GPU jobs: refused, lock not configured` | 沒有設定 GPU 鎖（`SHEPHERD_GPU_LOCK_FILE` 與 `SHEPHERD_WORKSPACES_ROOT` 都沒有設定），因此在設定之前每個訓練與測試執行都會被拒絕（§4.11） |
| `GPU jobs: not read`（灰色） | 無法讀取狀態。鎖的設定無法使用時，ⓘ 會說明，但只對 D8 允許的瀏覽器顯示路徑 |

### 說明：ⓘ 按鈕

以前印在 Diagnosis、Training Console 與 Runtime Settings 各欄位下方的說明，現在放在欄位名稱
後面的 ⓘ 裡。把滑鼠指向它、用 Tab 移到它，或點擊它：說明會出現在一個泡泡中，滑鼠可以移進泡泡
閱讀。點擊會讓它保持打開；按 Esc 或點擊其他地方會關閉它。點擊欄位名稱仍會把游標放進欄位。
泡泡會在 ⓘ 的下方或上方有空間的一側打開，而且永遠不會比視窗高：較長的文字在泡泡內捲動。必須
一直可見的內容仍然直接印出：拒絕的原因、工作的狀態、合成資料的標記、未經驗證的來源，以及
Training Console 對 HGT 的 AMP 拒絕（「Disabled: HGT requires full float32 …」）。

### Diagnosis (clinical)

第一個畫面分為三個面板：病患的表型、排名的候選，以及選中候選的細節（Evidence · Explanation ·
Matching phenotypes · Paths）。選取一列（點擊，或在聚焦的列上按 Enter）就會在細節面板顯示該
候選。視窗至少 1400 × 760 CSS 像素時，每個面板在自己內部捲動；視窗較窄或直向時，三個面板上下
堆疊，由頁面捲動。

**Patient ID** 留空時以「沒有 id」送出：伺服器為結果給它自己的 id（`patient_…`），因此不同
瀏覽器的結果絕不會共用同一個（M5-2 之前，空白欄位會以 `webui_patient` 送出）。

**「Model in use」卡片**是輸入面板的最後一項，Run Diagnosis 緊接在它下方：兩者跟在 Options
後面；當輸入欄位需要捲動時（例如 1536 × 760），兩者固定在面板底部。卡片顯示：

- 標題列 "MODEL IN USE ⓘ · checked HH:MM:SS ↻"：目前服務的內容會為這個瀏覽器讀取——頁面打開
  時、Diagnosis 顯示期間每 5 秒、每次診斷之後、更換模型之後，以及按 ↻ 時；超過 5 秒的讀取顯示
  為 "timed out (5 s)"；
- 模型 "GAT · run 2026-10-17 09:15 · epoch 5"，以及 "model fingerprint <12 hex> ✓ verified"
  （所服務之 checkpoint 檔的 SHA-256）；
- "trained on <workspace> · dataset fingerprint <12 hex>"：checkpoint 記錄的訓練所用 dataset
  （`training_input_digests.split_manifest`），以具有相同完整 fingerprint 的已列出 workspace
  命名，或 "matches no listed workspace"；絕不是所服務 workspace 自己的 manifest。讀取它需要打開
  checkpoint 檔，屬於操作者控制項（§4.13）：唯讀的瀏覽器看到 "not shown · reading the checkpoint
  is an operator control"。若另一個瀏覽器更換了模型，這張卡片會顯示 "not read yet · read it"，
  直到頁面再次打開或按下 "read it"（每 5 秒的讀取絕不打開 checkpoint）；
- 只在成立時出現的標籤：`synthetic data`（琥珀色）、`no SP table`、`no trained model` 或
  `GNN not loaded`（紅色）、`unverified`（琥珀色：圖未對照任何 workspace manifest 檢查）；
- "Recorded tests: <狀態>"（§4.9），它的 › 會打開 Details 的 Evidence；"Details ›"；
  "🔒 Change model…"。紀錄的測試與 workspace 名稱會在每次顯示 Diagnosis、每次診斷後、更換模型後
  以及按下 ↻ 時重新讀取（每 5 秒的讀取只讀目前服務的內容）：在 Research › Test 記錄的測試，回到
  Diagnosis 時就會出現。

| 卡片顯示 | 意義 |
|---|---|
| "Model changed at HH:MM — was <舊>, now <新>"，附 × | 這次讀取發現的模型（或所服務內容的其他部分）與這個瀏覽器上一次讀取的不同（例如另一個瀏覽器做了 Change model）。× 可關閉它 |
| 灰色："cannot check · <原因> · Retry"、"last known HH:MM:SS · not current" | 無法讀取目前服務的內容："status not read"（完整原因寫在標記下方那一行與 Details 中）或 "timed out (5 s)"。顯示的是最後已知的狀態，沒有 ✓，標籤為灰色虛線 |
| 灰色："read HH:MM:SS · older than this result" | 最後一次成功的讀取早於畫面上的結果 |
| "No model loaded" 與 "🔒 Load a model…" | 沒有設定任何 pipeline。Run Diagnosis 呈灰色且無法點擊：滑鼠指向它，或以 Tab 移到它，會說明原因，下方也寫著 "Disabled: no model loaded. …"。API 用戶端此時提出請求，仍會得到 demo 回答 |
| "No trained model · path reasoning only" 與紅色標籤 | 服務的是沒有訓練模型的圖。Run Diagnosis 仍可使用，結果帶有 `NO TRAINED MODEL: path-reasoning fallback.` |
| "🔒 Change model…" 呈灰色，附 "Operator controls only: <原因>" | 唯讀的瀏覽器（§4.13）。它永遠看不到 checkpoint 清單 |

結果上方的**標記（badge）**（§4.7）把結果的整個 served identity 與卡片最近一次的讀取比對，而
卡片在每次診斷後都會再讀一次；讀取失敗或早於結果的讀取，絕不會顯示 "Produced by the model in
use"。不論標記怎麼說，CSV 與報告都保留結果自己的身分。"Not a certified medical device; decision
support for a qualified clinician." 位於結果的底部。

**身分抽屜**蓋在細節面板上，有兩個檢視。"Model in use" 是所服務模型的資料，分五節（Identity，
附完整 digest 與 Copy；Uses；Data；Evidence；Warnings，永遠展開）、結構 fingerprint 那一行（列出
並計數不同之處；checkpoint 沒有記錄結構 fingerprint 時為 "not compared"），以及通往 Research ›
Models 完整紀錄的連結。"This result" 只顯示結果本身帶有的內容，每一項與目前使用中的對照。
"Details ›" 與 "Recorded tests ›" 打開前者，標記打開後者；更換模型進行中（確認畫面打開，或正在
重新載入）時，它們不會蓋掉 Change model 抽屜。

**🔒 Change model…**（操作者控制項）會打開一個蓋住結果與細節的抽屜：workspace 選單（伺服器的
清單；預設選中正在服務的 workspace）、置頂的 "In use now"、最近四個 run（其餘在 "Show older
runs"）、選中 checkpoint 的資料、"Compare in Research › Models ›"，以及 "🔒 Load this model" /
Cancel。Load 只詢問一次（"Reload the pipeline with <12 hex> for every session?"，from → to，
變更的列會標出），並以伺服器列出的 workspace 與 checkpoint 重新載入 pipeline。**重新載入只服務
問句中的那個模型**：它在發布任何東西之前，把自己讀到的 checkpoint 與那個指紋比對，因此期間被改寫
的檔案（執行中的訓練工作每個 epoch 都會改寫它的 `last.pt`，而重新載入會先讀圖譜）不會改變任何
東西，結果是 "Not reloaded; the model in use is unchanged." 加上原因。API 以重新載入請求的選用
欄位 `expected_checkpoint_sha256` 提供同樣的檢查（64 個小寫十六進位數字，否則 422）：其他位元組
會被拒絕，正在服務的 pipeline 照常服務；沒有這個欄位時與以前一樣不做檢查。它會取代每個瀏覽器
所服務的內容；已產生的結果保留自己的身分，重新載入期間診斷會等待。確認畫面打開期間，清單本身的
底部列會隱藏，讓 from → to 各列與這句話保持在畫面內；"Keep <舊>" 會放棄這次確認。確認畫面打開
期間，Run Diagnosis 停用："Disabled: finish or cancel the model change first."。按下
"🔒 Reload with <新>" 之後，確認畫面只顯示重新載入的狀態（"Loading <12 hex> for every
session… 00:00:42"）與 Close（關閉不會中止重新載入；180 秒後為 Read again 與 Close），因此無法
再要求第二次重新載入；卡片顯示同一行 "Loading …"，Run Diagnosis 顯示 "Disabled: a model load is
in progress; diagnoses wait until it finishes."。Cancel 或 × 關閉抽屜而不重新載入。重新載入期間，
Close 或 × 會立即關閉抽屜，重新載入照常進行（無法中止）：卡片保留那一行 "Loading …"，重新載入
回應後，卡片顯示當時使用中的模型，結果不會再顯示；期間按 "Change model…" 會重新打開抽屜，顯示
重新載入的狀態。在 Research › Models 中，Close 也以同樣方式關閉確認畫面。
從 Research › Models 做的 Load 也會更新這張卡片。同一時間只會打開一個抽屜。

### Research › Prepare › Datasets

伺服器在 `SHEPHERD_WORKSPACES_ROOT` 下列出的 workspace 的唯讀清單（§4.12）：也就是 Training、
Test 與 Models 選擇時使用的同一份清單。每一列顯示名稱；建置時間（UTC，來自
`workspace_build.json`），或「build time not recorded」加上各來源宣告的版本；**資料集指紋**，
即其 `split_manifest.json` 的 SHA-256 的前 12 個十六進位數字（Copy 會複製完整的 64 個；兩個
workspace 前 12 個相同時會顯示更多位數）；來源（`synthetic` 以琥珀色顯示）；資料集紀錄是否通過
驗證；train 與 validation 的樣本數。標記：

- **serving reads this workspace**：pipeline 正從它提供服務；
- **same dataset as served**：另一個完整指紋相同的 workspace；
- **in your Training run bar**：這個瀏覽器分頁為 Training 選擇的 workspace；
- **not trainable: <原因>**：有列出，但訓練會拒絕它，並附上原因：workspace 沒有
  `split_manifest.json` 時為 `no split manifest`，有但無法讀取時為 `no readable split
  manifest`，否則為清單本身給出的原因（例如一個連結到根目錄之外目錄的 workspace：「… resolves
  outside the workspaces root」）。

較長的名稱或來源清單會在儲存格內換行，因此「In use」欄一直保持在視野內。選取一列會顯示它的細節：
完整指紋與 Copy、manifest 綁定的內容（短 digest）、種子與切分數量、來源與建置紀錄、workspace
相對於伺服器 workspace 根目錄的路徑，以及它與其他列出的 workspace 的比較：「Same dataset as …」
（完整指紋相同）或「Same graph build as … (kg, tensors and record identical), another cut: the
allocation seed differs …」。對任何瀏覽器都不會顯示伺服器的絕對路徑：根目錄之下的路徑以相對於
根目錄的方式顯示，其他路徑寫作「a path outside the workspaces root」，被拒絕的
`SHEPHERD_WORKSPACES_ROOT` 會說明被拒絕但不顯示它的路徑。頁面在打開時讀取 manifest，而不是每次
選取時讀取。

**Use for training ›** 會打開 Research › Training，並在它的 run bar 中選好該 workspace，只對
這個瀏覽器分頁有效。伺服器上什麼都不會改變；Start 仍是 D8 檢查的控制項。不可訓練的 workspace
無法使用，按鈕旁的說明會寫出原因（「Not available: scratch_ws is not trainable — no split
manifest.」）。在 run bar 中改選另一個資料集，「in your Training run bar」標記會隨之移到那一個。

這個頁面上沒有任何操作者控制項：它只讀取 workspace 的小型 JSON 檔案，從不開啟 checkpoint，因此
唯讀的瀏覽器看到的是同樣的頁面。沒有設定 `SHEPHERD_WORKSPACES_ROOT` 時，頁面會說明這一點。資料集
仍在伺服器上以 CLI 建置（`scripts/build_knowledge_graph.py --generate-samples`；示範資料用 `scripts/setup_demo.py`）。

### Research › Prepare › Training

Training Console 現在是分成三部分的一個頁面：上方橫跨整頁的 run bar、左側下一次 run 的參數
（或取而代之的 Continue… 抽屜），以及右側的 run 檢視。訓練的執行方式沒有改變；改變的是頁面
選擇資料集與顯示 run 的方式。

**run bar** 永遠不捲動：

- **Dataset**：伺服器的 workspace 清單，與 Datasets 的相同；每個選項列出 workspace 名稱、
  資料集指紋的前 12 個十六進位數字與來源。選擇一個時會讀取它的 manifest 與紀錄，旁邊的 chip
  顯示建置時間、資料集指紋（Copy 會複製完整的 64 個）、`synthetic`、紀錄是否通過驗證，以及
  **Details ›**（manifest 綁定的內容、種子與數量、來源、相對於 workspace 根目錄的路徑、
  **All datasets ›**）。頁面打開時選的是 Datasets 的 **Use for training ›** 所選的 workspace，
  否則是這個分頁先前的選擇，否則是執行中工作的，否則是 serving 讀取的那一個（它可訓練時），
  否則是第一個可訓練的。訓練工作執行期間，選單會停用並顯示該工作的資料集：「in use by the
  running job」。
- **狀態標籤**：`Idle`、`Starting`、`RUNNING · epoch 12/100 · batch 340/812 · 00:14:32`、
  `Stopping`、`Completed`、`Stopped`（灰色）或 `FAILED`（紅色）。失敗或停止的 run 永遠不會是
  綠色。
- **Run name**，接著是**一行原因**，說明某個動作為何停用：「Stop disabled: no training is
  running.」、對唯讀瀏覽器顯示 D8 的原因（「Operator controls only: request not local」），或
  占用 GPU 的工作（「GPU job: test run (…) · Open Test ›」、「GPU job: train_model.py (pid …,
  since …)」）。
- **🔒 Start、🔒 Continue…、🔒 Stop。** Stop 會先詢問：「Stop training now? The last saved
  checkpoint is from epoch N (last.pt). Progress since then is lost; you can continue from
  epoch N.」，按鈕為 **■ Stop training** 與 **Keep training**。這個詢問屬於提出它時正在執行的
  工作：若該工作在你回答前結束，詢問就會關閉，之後啟動的工作不會顯示它；對已不是目前執行中的
  工作按下 **■ Stop training** 不會停止任何東西（「Not stopped: …; click Stop again」）。
- **Start 會送出 chip 顯示的資料集指紋**，job runner 在做任何事之前會再讀一次 manifest：若
  workspace 在這段時間內重建過，什麼都不會寫入或啟動，頁面會顯示「Not started: The dataset
  changed since it was shown; select it again. Nothing was started.」。API 以選用的查詢參數
  提供同樣的檢查，`POST /api/v1/training/start?expected_dataset_fingerprint=<64 個十六進位數字>`：
  manifest 不同時以 409 拒絕，不是 64 個小寫十六進位數字的值在到達服務之前就以 422 拒絕；沒有
  這個參數時與以前一樣不做檢查。**訓練行程會在它自己讀取 manifest 時再檢查一次**
  （`train_model.py --expect-dataset-sha256`，自 8242b47 的審查起），所以在 runner 檢查之後才重新
  發布的 workspace 也會在那裡被拒絕，在讀取圖譜與樣本之前：這次執行會失敗，輸出中有
  `train_model: refused: The dataset changed since it was shown; select it again. …`，結束碼 1，
  不建立 run 目錄。
- 失敗的 run 的**錯誤橫幅**橫跨整頁。標題（"Training failed: Process exited with code 1"）
  之下是 run 印出的最後一行錯誤（"Last error printed: TrainingDeviceUnavailable: --device auto
  requires CUDA, …"），對唯讀瀏覽器與輸出一樣經過遮蔽。run 旁邊只在有日誌時說明日誌在哪裡（它的
  `logs/train_….json`），否則說明沒有日誌（在建立 run 目錄之前就被拒絕的 run 顯示 "no run
  directory, so no log"）。接著是輸出的最後幾行，放在「Last output (N lines)」之下：視窗高度至少 1000 px 時展開，較矮的視窗則收合成這一行，讓圖表保有空間（你展開或收合
  後，在這個頁面中會保持）；一次顯示四行，其餘在框內捲動（窄視窗中全部顯示）。Dismiss 會在這個
  瀏覽器分頁中隱藏橫幅。run bar 中的每則通知都把名稱與路徑當作文字顯示，不論 workspace 或 run
  名稱含有什麼字元。

**參數，「for the next run」**：欄位、分組、預設值與驗證都與以前相同，Basic 展開，Advanced 與
Expert 收合。**Data Paths 與它的 Save Path 已移除**：workspace 由 run bar 選擇，儀表板不再
讀取或寫入伺服器工作目錄中的 `.shepherd_training_config.json`（留在那裡的檔案會被忽略；是否
提供「Save as the default dataset」是擁有者要決定的問題）。三個分組放不進視窗時（1920 × 1080
放得下），參數在自己的欄內捲動；run bar 永遠不捲動。四張圖各自至少保留 140 px 的繪圖區：
視窗太矮放不下時（例如 1536 × 760），run 檢視在自己的面板內捲動，其標題、run 的說明句與圖表
工具列固定在頂端，資源列則在圖表下方。

**Continue…** 會打開一個取代參數欄的抽屜：run bar 所選 workspace 的 checkpoint，最新的在前
（伺服器的清單：列出時會開啟每個 checkpoint，所以是操作者控制項），或手動輸入其中某個
checkpoint 的路徑：自 M5-3 起，手動輸入的路徑只會找到該清單中的 checkpoint，其他任何檔案都會在
開啟之前被拒絕（`PLAN_TEST_RESULTS.md` D8）。
**Check checkpoint** 顯示繼續訓練會做什麼，並比較 parent 記錄的資料集指紋與 workspace 目前的
指紋：相同時為 ✓，不同時為琥珀色（附上相符的已列出 workspace，或「matches no listed
workspace」）。抽屜打開期間，run bar 顯示 Start（停用：「close Continue… to start a new
run」）、**🔒 Continue Training**（資料集不同時停用，並附原因）與 **Cancel**。從清單選的
checkpoint 會在伺服器的清單中重新找到，並以位元組比對清單所列的 digest（「The checkpoint
changed since it was listed; refresh the list.」）；job runner 與訓練行程在啟動時仍會檢查
一切。每個項目列出 run、epoch、檔案以及它保存的內容（`last.pt (training state)`，或模型檔
`(weights only)`，後者無法繼續訓練）。抽屜有自己的 **Device**（auto、cuda 或 cpu），每次打開時
從參數的 Device 開始，Check checkpoint 與 Continue Training 使用的就是它。抽屜打開期間若 run
bar 的資料集改變（手動選擇，或從 Datasets 的「Use for training ›」），清單會以同樣的方式重新
讀取，改為該 workspace 的 run。

**run 檢視**顯示 job runner 最後啟動的 run；伺服器啟動後還沒有啟動過時，顯示 run bar 所選
workspace 最新的 run（頁面開著時每 10 秒重新列出一次，所以從 CLI 啟動的 run 在寫下第一個
checkpoint 後就會出現）。標題列出 run 名稱與這個 run 使用的資料集指紋，從它的 checkpoint 讀取：
「✓ same as the run bar」、琥珀色的「≠ not the selected dataset」、執行中還沒有 checkpoint 時的
「recorded at the first checkpoint」，或已結束而沒有 checkpoint 時的「not recorded (the run saved
no checkpoint)」。快照句說明 run 如何結束，以及可從哪個 checkpoint 繼續，
路徑相對於 workspace 根目錄；沒有 run 目錄的 run（在建立 run 目錄之前就被拒絕）會如實說明：沒有
checkpoint、沒有日誌，也沒有可畫的圖。下方是一張共用 epoch 軸的四格圖：Loss（train，以及以虛線表示的
validation）、MRR（validation，Trainer top-20）、Hits@K（validation，Trainer），以及對數軸的
learning rate（0 或缺少的值為空隙）。圖下是資源列（GPU 與記憶體）。

- **縮放與平移**：在圖上滾動滾輪會以指標為中心縮放；在 epoch 軸上只縮放 epoch（四張圖一起），
  在數值軸上只縮放該圖的數值。拖曳可平移；雙擊或 **Reset view** 會重新容納全部資料；+ 與 −
  可縮放。圖表取得焦點時：+ − x X y Y、方向鍵、0 與 End。新的 epoch 到來時、切換頁面時，以及在
  同一個瀏覽器分頁中重新載入時，檢視範圍都會保留（最近顯示的 20 個 run）；之前沒顯示過的 run
  會容納它的資料，包括 epoch 與每張圖的數值。epoch 軸以整數 epoch 計。縮放中時，
  「N new epochs →」會以相同寬度跳到最新的 epoch。在窄視窗中滾輪捲動的是頁面，圖表上方的說明
  會告訴你如何縮放。
- **Export CSV** 由瀏覽器把 run 的 metrics log（epoch、split 與每個記錄的值）存成
  `training_metrics_<architecture>_<run>.csv`。
- 圖表經由頁面自己的 server function 讀取 run 的 metrics log，只針對伺服器列出的 run（從不
  使用路徑），而且只在有新資料列時讀取。頁面顯示期間每 1.5 秒重新讀取一次，在其他頁面時完全
  不讀。圖表函式庫是 uPlot 1.6.32，由儀表板自己提供（`src/webui/static/vendor/uplot-1.6.32/`，
  MIT）；沒有任何東西來自 CDN。
- 載入另一個 run 的圖表時，圖區顯示「Charts loading…」，不會把前一個 run 的圖表放在新 run 的名稱
  之下；選擇改變之後才到的回應一律不畫。所要求 run 的圖表讀取失敗時（伺服器無法連線或發生錯誤），
  圖區顯示「The charts could not be loaded. Trying again…」，每 3 秒重試一次，最多 20 次；已畫出的
  run 在重新要求較新 epoch 的期間保留它的圖表。

**唯讀的瀏覽器**看到同一個頁面，每個動作都停用並附上 D8 的原因，而且不會為它開啟任何
checkpoint：run 標題顯示「not shown · reading the checkpoint is an operator control」，快照句
只給出 run 的狀態，也不顯示 parent 清單。metrics、圖表與資料集紀錄對它的讀取方式與對操作者
相同。

### Research › Evaluate › Test

§4.9 流程的頁面：匯入 cohort 版本、在其上量測模型、保留報告並登錄它。共有三個面板：

- **Cohort。** workspace（伺服器的清單，與 Datasets 相同以資料集指紋顯示）與 cohort 版本
  （registry 的清單，`SHEPHERD_REGISTRY`），附上它的 role、各項數量、版本 digest，以及它是否
  對照這個 workspace 的圖譜映射（「bound to this graph」，或「mapped against another graph」，
  執行時會被拒絕）。其下是該版本 report store 的每一個執行目錄，最新的在前：何時執行、用哪個
  模型、是否保留了報告、是否已登錄；當同一個 key 下的報告不一致時顯示「runs disagree —
  investigate」（D5）。沒有保留報告的執行會說明這一點；目前正在執行的測試位於最上方一列，而這台
  伺服器上最後一次結束時 store 中什麼也沒有的測試執行（在建立執行目錄之前就被取消，或失敗）會
  保留為一列，直到下一次啟動或重新啟動伺服器。清單列出最新的 8 份報告以及選取的那一份；
  **Show all N reports** 會列出其餘的（這個瀏覽器分頁會記住這個選擇）。對操作者而言，每一列的
  登錄狀態都是在所顯示的 workspace 中、持有它所量測權重的已列出 checkpoint 旁的帳本中檢查，因此
  尚待登錄的報告無論選取哪一列都顯示「not registered」；其 checkpoint 只列在另一個 workspace 的
  報告，在被選取之前顯示「registration cannot be checked from the list」。
- **Run。** 模型：伺服器清單中該 workspace 的 checkpoint，依執行、epoch、檔案與模型指紋列出。
  列出它們會開啟每一個 checkpoint，因此這是操作者控制項：唯讀的瀏覽器不會看到它，而且頁面對
  每個 workspace 只列出一次（↻ 會重新讀取清單）。接著是執行時會尋找的 R10 證據（被量測的
  workspace，以及 `SHEPHERD_TRAINING_EVIDENCE_DIRS`）、**Register the result**、
  **Deterministic execution**，以及由伺服器固定的 batch size 與裝置（量測程式的預設值；
  `SHEPHERD_DEVICE`）。**🔒 Start test** 透過 job runner（§4.11）啟動真正的
  `measure_scorer.py`，使用伺服器在自己的清單中重新找到的 workspace、版本與 checkpoint，並以
  **頁面顯示過的完整模型指紋與版本 digest 作為預期值**：執行時讀到的檔案或版本若是其他位元組，
  會在量測之前拒絕。Start 會說明它為何不可用：先說 GPU 工作（「Disabled: GPU job: … One
  managed GPU job at a time (training or evaluation).」），因為工作執行期間任何選擇都無法讓它
  可用，然後才是缺少某個選擇或另一個圖譜。工作卡片顯示狀態（Running、Completed、紅色的
  Failed、灰色的 Cancelled：失敗或取消的執行從不顯示為綠色）、已經過的時間、執行收到的輸入、
  以 UTC 表示的開始與結束時間（與報告列命名執行的方式相同），以及其輸出的最後 12 行（取消後
  會收合，其最後一行說明執行已被取消，沒有 traceback；量測拒絕訊息中提到的病患 id 會被略去，
  D7）；狀態旁的 **🔒 Cancel** 只停止測試執行，從不停止訓練工作。在寬螢幕上，Start、其原因與工作卡片固定在
  面板底部（上方的選擇會捲動），輸出則在卡片內、Cancel 所在那一行之下捲動，所以兩者都保持
  可見。執行結束後，卡片顯示 job runner 從磁碟讀到的結果：是否保留了報告、是否已
  登錄，並附上 job runner 的說明（執行沒有寫出 outcome 檔案時為「No report was published …」；
  取消後說明留下了什麼）。頁面顯示時每 5 秒讀取一次工作狀態；剛結束的執行的報告會被選取。
- **Result。** 選取的報告，以帳本將會保存的紀錄形式顯示：對已排名案例的 Mode C 指標、含每一類
  排除的涵蓋率、各項關係、以文字表示的 R10（*unverifiable* 從不顯示為 0）、環境、資料集來源，
  以及是否已登錄（對操作者而言，是在清單中持有它所量測權重的 checkpoint 旁的帳本中檢查）。
  **🔒 Register this report** 透過唯一的登錄函式（§4.9）登錄一份已保留的報告：伺服器在版本的
  store 中找到報告、確認其位元組就是顯示過的那些、從它自己的 checkpoint 清單找出帳本應在的
  位置（以完整 digest 找到持有該權重的已列出檔案，從不使用報告記錄的路徑：若執行讀取的那個
  檔案以這些權重列出，就用它，這也是清單查看的地方；否則用執行所量測的 workspace 中另一個已
  列出的副本；再否則用所顯示 workspace 中的副本），然後登錄。執行之後
  被移動過的 checkpoint 無法以這種方式找到；頁面會說明，機器上的
  `scripts/record_evaluation.py` 仍然可用。當報告所屬 key 下的多次執行不一致時，頁面顯示這個
  封鎖、Register 不可用，並列出與新增 **investigation notes**（誰、何時、發現了什麼；🔒 Add
  note），與該 key 的報告保存在一起。筆記從不解除封鎖。在寬螢幕上，筆記表單（Who、What was
  found、🔒 Add note）固定在面板底部、Register 之上，所以無論報告或筆記多長都保持可見（What
  was found 最多長到四行，之後在框內捲動）。筆記檔案無法讀取時，頁面會說明這一點（可能缺少某
  則筆記），從不顯示「沒有保存筆記」。

**🔒 Import a new version…** 會在 Run 與 Result 上方打開一個抽屜：intake 目錄中的檔案
（`SHEPHERD_INTAKE_DIR`，§4.12：其中的 `*.json` 檔案，依名稱、大小與修改時間列出；連結到目錄
之外的檔案會列出但不提供選擇；不會從瀏覽器上傳任何東西）、版本標籤、role（並說明它是否在
Models 中提供依測試分數排序：`SHEPHERD_ORDERING_ROLES`，`institutional_acceptance` 永遠不提供）
以及遺失 term 的政策。**🔒 Preview** 在服務內以 CPU 執行匯入程式（它會讀取 workspace 的圖譜），
不寫入任何東西。預覽顯示各項數量、排除原因及其數量、無法映射的 term 及列出每個 term 的案例數，
以及版本綁定的內容；**案例 id 不會傳送到瀏覽器**（機器上的命令列預覽會列出它們，提交後的版本
會記錄它們）。不符合 `shepherd-cases/1` 的檔案會被拒絕，其問題依案例索引列出（最多 10 項），
從不依案例 id；機器上的命令列預覽（`scripts/import_cohort.py`）會連同案例 id 列出每一項。
**🔒 Commit as version …** 以相同的輸入重新準備，並在顯示過的 preview id 下提交：
預覽之後表單有變更，或之後來源或圖譜有變更，都會被拒絕。一次只執行一個預覽或提交；提交之後
抽屜會關閉，並選取新的版本。寬度或高度低於 1400 × 760 時，抽屜是右側的覆蓋層，在其內部捲動，
鍵盤焦點會移到它的標題。

**唯讀的瀏覽器**可以看到各版本、它們的報告與狀態、工作與結果：每個控制項都不可用，並寫明
「Operator controls only: <原因>」（筆記的 Who 與 What 欄位也一樣），checkpoint 清單也不顯示。每個控制項都在後端檢查它自己的
請求（§4.13）。不會顯示伺服器的絕對路徑：registry、workspace 根目錄、intake 目錄、證據目錄或
repository 之下的路徑以其名稱書寫（`<registry>/versions/…`），其他路徑寫作「a server path」。

### Research › Evaluate › Models

這是 Diagnosis 分頁「Change model…」所用的同一份模型清單的研究檢視，列出伺服器為某一個 workspace
列出的 checkpoint（選擇 **Workspace**；**Architecture** 可縮小清單）。Workspace 選單下方會再次列出所選
workspace 的資料集識別（建置時間、資料集指紋、`synthetic`、紀錄的狀態），因此即使選單截斷了很長的
workspace 名稱，這些資訊也不會被遮住。所有 run 都會顯示，最新的在前。各欄：

- **Model**：架構、run 的開始時間、epoch；正在提供服務的 checkpoint 標示 **In use (Diagnosis)**
  （以完整的模型指紋比對）；
- **Fingerprints: model · dataset trained on**：模型指紋（checkpoint 的 SHA-256 的前 12 個十六進位
  數字），下方是 checkpoint 所記錄的完整資料集指紋所對應的已列出 workspace，或「matches no listed
  workspace」；`synthetic` 以琥珀色顯示；
- **Validation MRR (Trainer, top-20)**：Trainer 自己在 val 樣本上的數值，也就是訓練用來挑選
  checkpoint 的指標。它從來不是測試結果，也從不與測試結果放在同一欄；
- **Recorded tests**：checkpoint 旁的 ledger 所記錄的內容，以狀態顯示（「none in ledger」、「no
  ledger」、「3 · 1 contested」……），從不顯示某個結果的數值。

**列出清單會開啟該 workspace 的每一個 checkpoint**，因此清單是操作者控制項（§4.13）。
自 M5-3 起，伺服器對每個 checkpoint 檔案只讀一次，檔案改變時（依其裝置、inode、大小、修改與
變更時間判斷）才再讀，因此再次打開清單或 Load 之後的重新讀取不會開啟未改變的檔案；載入、開始
訓練或量測時，仍會以所顯示的摘要檢查它所使用的位元組。唯讀的瀏覽器
看不到清單：頁面會說明原因，並列出三個條件（部署模式 M1、請求經由本機的 loopback 抵達——在本機上或
經由 SSH forward——且沒有轉發標頭），改為顯示 **Recorded test results of the model in use**：依模型指紋
從正在服務的 checkpoint 旁的 ledger 讀取，不開啟任何 checkpoint。每筆紀錄的 regime（裝置，deterministic
開或關）就是紀錄本身所寫的，與操作者看到的相同（為此 `GET /api/v1/evaluations/loaded-model` 會為每筆
紀錄附上 `deterministic_algorithms`；CPU 紀錄單憑 environment class 看不出來）。那裡沒有 Trainer 的
驗證數值，也沒有 Load。

**Order: Newest first | By test score。** 只有角色列在 `SHEPHERD_ORDERING_ROLES`（§4.12）中的 cohort
version 才提供依測試分數排序；其他 version 會列出並說明為何關閉，而 `institutional_acceptance` 一律
關閉（「awaits the institution's formal evaluation rules」），即使設定中列出它也一樣。選擇指標（Mode C
MRR (untruncated)、Hits@k，或 mean rank——它以最低者在前排序）與 cohort version。其他 scoring
semantics 下的紀錄從不與這些一起排序；當該 version 的紀錄分屬不止一種時，會出現 **Scoring
semantics** 選項。凡是提供排序的地方，頁面都會說明：「A cohort version whose scores were used to
choose a model is no longer an independent final test for that model.」系統從不依測試分數自動選擇
模型。各分組：

| 分組 | 內容 |
|---|---|
| Ordered by … on … · highest first | 每個 checkpoint 一個數值，依固定規則：它第一筆以 deterministic 環境登錄的紀錄，否則為它第一筆登錄的紀錄（有多筆時顯示 `#3 · D4 rule`）。每一列顯示該紀錄的數值環境（`CUDA · deterministic`），若不是多數已排序列共有的環境則標示 **regime differs** 並附 ⓘ |
| Contested on this version · not ordered | 規則選中的紀錄有爭議：該 version 的報告庫中同一鍵下的報告彼此不一致。它不參與排序，也從不以下一筆紀錄遞補 |
| Contested status cannot be checked · not ordered | 無法讀出該鍵下的報告是否彼此一致（該 version 的報告庫或其中某份報告無法讀取；該 version 不在登錄庫的清單中）。它同樣不參與排序，也從不顯示為無爭議 |
| Not compared on this version · not ordered | 其他 scoring semantics 下的紀錄、分在兩個 ledger 中的紀錄、無法讀取的紀錄或 ledger，或無法讀取的 checkpoint；該列會說明是哪一種 |
| Not tested on this cohort version · newest first | 在此 version 上沒有紀錄 |

剛選擇另一個 workspace 時，它的各列可能短暫顯示在「Records not read yet」之下，直到頁面讀完它們的
ledger。

**選取一列**會顯示完整的模型指紋與 Copy、它在什麼資料上訓練（以及那是否為此 workspace 目前的資料
集）、自成一張卡片的 Trainer 驗證數值、任何警告，以及兩個分段：

- **Recorded test results (Mode C)**：ledger 為這組權重記錄的每一筆紀錄：cohort version 與其角色、
  Mode C MRR 與 Hits@10、列出每一項排除的涵蓋率，以及 `registered #N` 或琥珀色的 `contested #N` 並
  說明各次執行在哪裡不一致（或「contested status cannot be checked」，從不顯示為無爭議）、
  deterministic 開或關；清單已排序時，標出固定規則選中哪一筆，以及其他紀錄與它的差異；接著是各項
  關係、其他 Mode C 指標、以狀態顯示的 R10（*unverifiable* 是一個原因，從不是 0）、執行環境、資料集
  與登錄資訊；
- **Usage history**：每一個已列出的 cohort version 的使用紀錄中提及此 checkpoint 的項目：測試執行、
  登錄，以及從已排序清單進行的載入，最新的在前。

**🔒 Load this model** 是共用的 Load（與 Diagnosis 的「Change model…」相同的確認；確認開啟時會蓋住
細節區，不論視窗多矮，它的按鈕都留在表格下方的可見範圍內；按 Close 回到細節區，且仍選取同一個
checkpoint）。清單依測試分數排序時，按鈕旁的說明會指出這一點，而這次 Load 會以「依該指標排序的模型
載入」寫入該 version 的使用紀錄，不論是否生效，並附上原因。從「Newest first」清單進行的 Load 不會在
那裡寫入任何內容。頁面在顯示時、更換 workspace 時以及每一次 Load 之後讀取，包括沒有改變任何服務內容
的 Load（重新載入被拒絕、再次載入正在使用的模型），因此使用紀錄會顯示該次 Load 寫入的項目；頁面上
沒有任何東西依計時器執行。

**窄版**（寬度低於 1400 或高度低於 760，各區域上下堆疊時），選取一列會捲動到它的細節區並把鍵盤焦點
移到那裡；細節區標題列的 **↑ Back to the list** 會回到該列。

---

## 總結：部署檢查清單 ✅

### Windows x86 + Blackwell

- [ ] ✅ Python 3.12 安裝
- [ ] ✅ CUDA 13.0 安裝
- [ ] ✅ PyTorch 2.9.0 + CUDA 正常
- [ ] ✅ PyTorch Geometric 正常
- [ ] ✅ FlashAttention-2 安裝成功
- [ ] ✅ Voyager 可用 (CPU vector search)
- [ ] ✅ 所有單元測試通過
- [ ] ✅ 模型可訓練（檢查VRAM使用）
- [ ] ⚠️ 儀表板使用 Gradio 6.30.0（拉取後執行 `uv sync --inexact`）：Windows 後端尚未測試（§4.10）

### ARM + Blackwell (DGX Spark)

- [ ] ✅ 架構確認為 aarch64
- [ ] ✅ CUDA 13.0 可用
- [ ] ✅ PyTorch 2.9.0 ARM build 正常
- [ ] ✅ PyTorch Geometric 可用（wheel或編譯）
- [ ] ⚠️ 注意力後端已降級（xformers或SDPA）
- [ ] ✅ Vector Index: cuVS (GPU) 或 Voyager (CPU fallback)
- [ ] ✅ 統一記憶體正確識別
- [ ] ✅ 模型可訓練（檢查128GB優勢）
- [ ] ✅ 儀表板使用 Gradio 6.30.0（拉取後執行 `uv sync --inexact`；`/ui` 可載入，§4.10）

---

**下一步**: 參考 TODO 清單開始開發！

**版本**: v3.2
**維護**: 隨著套件更新持續更新本指南