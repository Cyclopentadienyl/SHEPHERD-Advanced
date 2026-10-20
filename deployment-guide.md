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
Write-Host "  3. 訓練模型: python scripts/train_model.py"
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
echo -e "  3. 訓練模型:     python scripts/train_model.py  (可選 --config <hyperparameters.yaml>)"
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
- Diagnosis 分頁：結果上方與模型狀態裡的 `SYNTHETIC WORKSPACE: not a clinical result.`
  橫幅，以及結果「Produced by」那一行的 `dataset synthetic ✓`；
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
- 結果下方的 **「Produced by」一行**（digest 縮短顯示），以及一個預設收合的區塊
  「What produced this result — full identity」，內含完整表格、宣稱界線與
  "Not a certified medical device; decision support for a qualified clinician."。
- **模型狀態**（Model Configuration）在「Serving」那一行顯示目前載入之 pipeline 的身分，
  並另以一行列出正在服務的 workspace 與 checkpoint 檔。載入 pipeline 後，Data Directory
  與 Checkpoint 欄位顯示這兩個路徑（自動選出的檔案也照實顯示），因此 Load / Reload 重新載入的
  就是正在服務的 checkpoint 檔；清空 Checkpoint 欄位即讓伺服器重新自動選擇。有兩種情況不同：
  沒有服務任何 checkpoint 時，Checkpoint 欄位是空白的，重新載入會自動選擇一個（workspace 裡
  沒有時則被拒絕）；只以 `SHEPHERD_KG_PATH` 啟動時，Data Directory 是空白的，重新載入需要先
  填入 workspace。未載入任何 pipeline 時，欄位顯示服務啟動時 Save Config 已儲存的路徑；新的
  儲存自下次啟動起生效。
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
4. Diagnosis 分頁的模型狀態顯示目前載入之 checkpoint 已登錄的結果。

以上都由操作者在機器上執行。沒有任何 UI 控制項或 API 請求會匯入、量測或登錄任何東西。

### registry

registry 是**由你建立**的目錄，每個指令都以**位於 repository 之外的絕對路徑**傳入。相對路徑、
不存在的路徑，以及解析後落在 repository 內的路徑（包括經由 symlink）一律拒絕。不會有任何東西
替你建立它，也沒有任何請求指定它。各指令以 `--registry` 接收它；服務則從 `SHEPHERD_REGISTRY`
讀取（§4.11），以同一個規則檢查。

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
以 Diagnosis 分頁所用字句表示的四個 R10 標籤（`unverifiable` 附其原因）、一次標籤定義、報告路徑
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
| `measure_scorer.py` | 已量測（加 `--record` 時：已登錄、已存在或一致） | 某項檢查拒絕了這次執行：一則訊息，不寫入任何東西（在沒有 CUDA 的主機上給 `--device auto` 或 `cuda` 也是，在讀取任何東西之前）；或**已量測，但報告沒有寫出**（例如磁碟已滿或唯讀）：一則訊息說明可能留下了什麼，而那絕不會是可登錄的報告——`--output` 旁這次執行的排名檔，或報告存放處中沒有 `report.json` 的執行目錄（寫出報告之後才失敗的執行，例如主控台被關閉，會說明報告在哪裡） | 命令列被拒絕，包括屬於另一種 cohort 的參數 | **已量測，但登錄被拒絕**：訊息會指出報告與之後登錄用的 `record_evaluation.py` 指令；若報告檔已不是這次執行發佈的位元組，則如實說明，並要求重新量測（那份報告可能已不存在） |
| `record_evaluation.py` | 已登錄、已存在、一致，或已顯示 | 拒絕，一則訊息；不寫入任何東西 | 命令列被拒絕，包括 `--checkpoint` 與 `--checkpoint-dir` 都沒給或都給 | — |
| `audit_split_overlap.py` | 已寫出報告 | 拒絕，一則訊息；不寫入任何東西 | 命令列被拒絕（例如已移除的 `--evaluation-cohort-kind`、只給 `--registry` 不給 `--cohort` 或反之，或與 `--cohort` 一起給出 `--splits`），並印出 usage 行 | — |

若 checkpoint 目錄對你而言是唯讀的，量測會以結束碼 3 結束：報告會保留，由可以寫入該目錄的人
用這次執行印出的 `record_evaluation.py` 指令登錄。

**`--record` 只登錄這次執行發佈的報告，且必須是這次執行讀取的 checkpoint。** 執行會保留它寫入的
報告位元組的 digest，以及這次 checkpoint 讀取的 identity；登錄時重新讀取報告，在解碼任何內容之前先
核對這次讀取的 digest，再核對它記錄的 checkpoint。若報告檔在這之間被替換、修改、損壞、刪除或無法讀取
——即使換成另一份合法報告（另一個 checkpoint 的，或另一個 cohort 版本的）——就不登錄任何東西，以結束碼
3 結束，並說明該路徑上的檔案不是它發佈的報告（或已不存在），因此這次執行自己的報告可能已不存在：請重新
量測以登錄這次量測。只有在讀回的報告確實是發佈的那份、而是其他環節（例如 ledger）拒絕時，才會印出之後
登錄用的指令。之後用 `record_evaluation.py` 登錄時，仍照舊以指定的
checkpoint 檔核對報告。

### Diagnosis 分頁顯示什麼

模型狀態（Model Configuration）在「Serving」那一行下方，為目前載入的 checkpoint
（`sha256:<前 12 字元>…`）加上 **「Recorded test results (Mode C)」**。每次讀取狀態時，都透過
`GET /api/v1/evaluations/loaded-model` 讀取，並以 checkpoint 的 digest 而不是檔名來找：換成另一個
模型之後，絕不會顯示前一個模型的結果；連同 ledger 一起搬移且位元組不變的 checkpoint，仍找得到
自己的結果。

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
  `record_evaluation.py`（`--report` 與 `--show`）以及模型狀態都會拒絕，模型狀態會回報 ledger
  未讀取。沒有任何東西會遷移它，其結果也不會被帶過來：**請手動封存**，例如
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

**CORS** 會與 D8 一起收緊。這一步不改它的設定（`src/api/main.py` 中的 `allow_origins=["*"]` 與
`allow_credentials=True`），但 Starlette 1.9 改變了這個設定送出的內容：現在每個跨來源的回應都會
回送請求的 `Origin`，並附上 `Access-Control-Allow-Credentials: true`；Starlette 0.52.1 只在請求
帶有 cookie 時才這樣做，否則送 `*`。自 Gradio 6.25 起，上層 app 有 `CORSMiddleware` 時，掛載的
app 不再加上自己的 CORS 標頭，因此 `/ui` 也遵循同一政策。在 D8 批次之前，服務前面的 HTTP 認證
（Basic、Negotiate/Kerberos）或 TLS 用戶端憑證因此擋不住其他網站：任何網站的頁面，只要在持有
這些憑證的瀏覽器中打開，都能讀取 `/api/*` 與 `/ui/gradio_api/*` 的回應。

### route guard

Gradio 6.30 在每個掛載的 app 上加入兩個無法從上游關閉的 routes。服務會在 Gradio 看到請求之前
拒絕兩者（`src/api/middleware/gradio_routes.py`）：

- **`/ui/gradio_api/file=<URL>`**（以及舊的 `file/<URL>`）。自 Gradio 6.20 起（issue #13593，
  PR #13596），伺服器會自己抓取 `http(s)://` 目標——不經過 `HTTPS_PROXY`——再以儀表板自己的位址
  回傳；對某些 URL，它會把遠端的 HTML 頁面以 inline 方式提供，就像儀表板自己的頁面一樣。任何
  URL scheme 的寫法都會被拒絕：大寫或大小寫混用、percent-encoded、`//host`。
- **`/ui/gradio_api/process_recording`**：把上傳的螢幕錄影轉成暫存檔，並可能執行 ffmpeg。因此
  Gradio 設定面板中的螢幕錄影功能會顯示 "Failed to process recording. Saving original
  version."，並在瀏覽器中存下未處理的錄影。

拒絕的回應是 `403`，附一行原因與 `connection: close`。請求本體不會被讀取，請求日誌會像其他
回應一樣記下這次拒絕。本機下載不受影響：Training Console 的 metrics CSV，以及 Diagnosis 分頁的
CSV 與 Markdown 匯出照常下載，`HEAD` 與 `Range` 請求也一樣。

### 儀表板的其他變更

- **Training Console 的圖表**外觀與以前相同（y 軸從 0 開始，每條數列維持自己的顏色）。Epoch
  Window 現在只篩選資料列，x 軸隨之調整。
- **Runtime Settings** 在打開分頁時會重新整理重啟鎖定狀態：Gradio 6 不會在尚未打開過的分頁中
  執行 timer。不論如何，訓練進行中時後端都會拒絕重啟。

### 平台

已測試的後端：Linux x86_64（開發環境）與 Linux ARM64（homelab，由 reviewer 檢查）。
**Windows 後端尚未以 Gradio 6.30.0 測試**，在有 Windows 後端可測之前一律標為未測試；Windows
瀏覽器則包含在瀏覽器檢查中。

---

## 4.11 伺服器設定與伺服器的清單（自 M5-1 起，2026-10-20）

M5 的控制項只作用於伺服器列出的項目，從不作用於請求指定的路徑（D7、D8）。伺服器列出什麼，以及
操作者控制項所依賴的部署模式，都來自**服務的環境變數**。請在啟動服務的地方設定它們（執行
`launch_shepherd.sh` 的 shell，或 systemd unit 中的 `Environment=` 行），然後重啟服務。每個
設定都在使用時讀取。沒有任何請求、表單或儲存的 UI 檔案能設定它們。缺少或被拒絕的設定會關閉
需要它的功能，並說明原因。**沒有任何東西會退回預設目錄**，repository 的 `data/workspaces/`
也一樣。

| 變數 | 值 | 未設定或被拒絕時 |
|---|---|---|
| `SHEPHERD_DEPLOYMENT_MODE` | 恰好是 `M1`（`m1`、` M1` 或 `M2` 都算其他值） | 唯讀：操作者控制項關閉，待 D8 的檢查套用它之後（目前尚未，見下文） |
| `SHEPHERD_WORKSPACES_ROOT` | 一個已存在目錄的絕對路徑；其子目錄就是各個 workspace | 不列出任何 workspace |
| `SHEPHERD_REGISTRY` | cohort registry（§4.9），即各指令以 `--registry` 接收的目錄：絕對路徑、已存在、位於 repository 之外 | 不列出任何 cohort 版本 |
| `SHEPHERD_INTAKE_DIR` | 位於 repository 之外的已存在目錄，你把要匯入的 `shepherd-cases/1` 檔案放在這裡；絕對路徑 | 不提供任何匯入來源 |
| `SHEPHERD_TRAINING_EVIDENCE_DIRS` | 選填：R10 證據目錄，與 `--training-evidence` 接收的相同，以 `:` 分隔（Windows 為 `;`）；每一個都必須是已存在的絕對路徑 | 無；只要有一個項目不合格，整個設定就被拒絕 |
| `SHEPHERD_ORDERING_ROLES` | 選填：提供依測試分數排序（D4）的 cohort 角色，以逗號分隔 | 無；`institutional_acceptance` 即使列出也會被忽略（§4.9 問題 1） |

```bash
export SHEPHERD_DEPLOYMENT_MODE=M1
export SHEPHERD_WORKSPACES_ROOT=/home/you/SHEPHERD-Advanced/data/workspaces
export SHEPHERD_REGISTRY=/srv/shepherd/cohorts
export SHEPHERD_INTAKE_DIR=/srv/shepherd/intake
```

- **`M1`** 是 SPEC_4 的單一 workspace 部署：在本機上，或經由 SSH forwarding 存取，由 owner 或
  指定的技術人員操作（§4.9 問題 3）。它是操作者控制項檢查的條件之一，而不是授權。其餘條件由
  D8 在處理每個請求時檢查。不要在 reverse proxy 後面、或在其他人也能登入的機器上設定它。
- **部署模式目前尚未套用。** M5 的下一步，D8 的檢查，會讀取它，並以它把關每個操作者控制項。
  在那一步完成之前，沒有任何東西讀取它：訓練的啟動與停止、pipeline 重新載入與設定儲存都與
  M5-1 之前一樣開放，不論這個變數的值為何。不設定它，並不會讓這個版本變成唯讀。
- **workspace 根目錄可以位於 repository 內**，一般使用的 `data/workspaces/` 就是如此，但必須
  由你明確指定。相對路徑、不存在的目錄與檔案都會被拒絕。
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
  操作者控制項所在之處提供它。**目前沒有任何路由提供這份清單。** M5 的下一步會在 D8 的檢查
  之後把它掛載為 `GET /api/v1/workspaces/{name}/checkpoints`；在那之前，它只存在於服務內部，
  供 M5 的 UI 步驟的儀表板使用。
- **既有的 checkpoint 清單**（`GET /api/v1/training/checkpoints` 與 Training Console 的
  resume 清單）以同樣的方式讀取每個檔案。每個項目新增 `sha256`、`validation`、
  `dataset_fingerprint` 與 `metadata_problem`，全部來自同一次讀取。以前它載入每個檔案卻不保留
  digest，而且載入錯誤時一聲不響地略過。讀取時已經不存在的檔案（目標已被移除的連結）沒有
  大小或時間可顯示，會從這份清單中略去；它不再讓整份清單失敗。這份清單與 M5-1 之前一樣保持
  開放，直到 D8 的檢查完成。
- **Cohort 版本**（`GET /api/v1/cohorts`）：registry 中的每個版本，以 label 及其 `cohort.json`
  的 SHA-256 識別，附上它的角色，以及這個角色是否提供排序。它也顯示版本的計數、所綁定的圖與
  資料集紀錄，以及匯入時間。清單絕不顯示病例識別碼或排除清單：D7 把它們留在這台機器上的
  registry 中。中斷的匯入會附原因列出。每個版本的報告庫以 run id 列出。一次 run 的報告是否
  保留、是否已登錄，來自登錄流程本身對報告庫與 ledger 的讀取（M5 的評估步驟）。在接上之前，
  每個 run 都會說明它的狀態未被讀取。

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