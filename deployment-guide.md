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

1. **優先找回與距離表匹配的原始 sidecar**，之後兩個檔案一起保存、搬移。有 sidecar
   時它的值優先，`SHEPHERD_SP_HOP_BOUND` 會被忽略。
2. **找不到 sidecar，但能確認原始設定**（例如從 build log 確認確實是 5-hop）才設定
   `SHEPHERD_SP_HOP_BOUND=5`。**不要所有部署一律填 5，也不要用表內最大距離回推**
   ——那個數字證明不了天花板。
3. **完全不知道原始設定就重新生成**（`scripts/compute_shortest_paths.py`）。若暫時
   以純 GNN 運行，必須明確知道評分模式已經改變。
4. 設定環境變數後**重啟後端**，再用下面的狀態檢查確認。

### 環境變數

```bash
# 只在確知原始 build 的 max_hops 時才設定
export SHEPHERD_SP_HOP_BOUND=5
```

- 有效值為 **1..127 的整數**（生成腳本自己驗證的同一個範圍），且**不得低於表內觀察
  到的最大距離**——低於就會被拒絕，因為不可達代價會落在真實距離之下。
- 無法解析成整數的值會**被拒絕而非忽略**，避免「以為設好了、其實沒有」。
- 這是**行程層級**的設定：必須設在啟動後端的那個環境裡，並重啟後端才會生效。
  **在另一個終端機 export 不會改變已在執行的行程。**
- 換載不同工作區時（透過 `/api/v1/pipeline/reload`），必須確認這個 bound 對新的表
  仍然適用。

### 缺 sidecar 又沒設定時會發生什麼

在預設配置（GNN 已載入、`sp_optional=True`）下，**服務不會停止**，而是改為純 GNN
評分：`sp_ready=False`、`scoring_mode=gnn_only`、eta 實際為 1.0。**分數與排序會與
啟用最短路徑時不同**，這是刻意的取捨——寧可少一個訊號，也不拿沒人選過的天花板評分。

> **例外：** 若部署刻意設定了 `sp_optional=False`，現行程式在 SP 不可用時會**關閉
> GNN 評分**。這不是「一律退回純 GNN」。目前 API 的建置路徑使用預設值
> `sp_optional=True`。

### 損壞或不合法的 sidecar

sidecar **存在但不合法**（不是 JSON 物件、`max_hops` 不是整數或超出 1..127、或低於
表內距離）時，候選 pipeline 會被**拒絕**，而不是退回猜測：

- 服務**尚無**可用 pipeline（冷啟動）→ `/api/v1/diagnose` 回 **503**，不會回傳任何
  候選疾病。
- 服務**已有**正在運行的 pipeline → 該次 reload 被拒絕，**舊 pipeline 繼續服務**。

### 設定後的狀態檢查

```bash
curl -s http://localhost:8000/api/v1/pipeline/status
```

| 欄位 | 預期 |
|---|---|
| `sp_ready` | 啟用時 `true`；缺 bound 時 `false` |
| `sp_max_hops` | 實際生效的天花板 |
| `sp_hop_bound_source` | `sidecar`（讀自檔案）或 `configured`（來自環境變數） |
| `scoring_mode` | 啟用時 `gnn_plus_shortest_path`；SP 關閉且 GNN 正常時 `gnn_only` |

`sp_hop_bound_source` 存在的理由就是這個：**「讀來的」和「設定的」是不同強度的證
據**，狀態端點要能分辨，而不是只能去翻日誌。

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
| `missing_roles` | 未能識別的輸入角色 |
| `counters` | 解析統計，例如 `rows_skipped_unresolved_disease_id` |
| `incomplete_by_design` | **這份紀錄刻意不涵蓋的東西** |

`sources` 的四個角色是 `mondo`、`hpo`、`phenotype_hpoa`、`genes_to_phenotype`。
**檔名只是線索，digest 才是身分**——同一個目錄名下的檔案可以被原地替換。

`declared_version` 是本體檔頭的原始 `data-version`，**沒有就是 `null`**，不會用
OBO 的 format version 冒充 release。

`counters` 的命名要照字面讀：`rows_skipped_unresolved_disease_id` 是
**`phenotype.hpoa` 在那一個解析階段跳過的列數**，其中包含刻意不支援的 ID 類型
（如 DECIPHER），**不是「遺失的疾病數」，也不是版本不符的總量**。builder 另外還會
因節點不存在而丟邊，那些不在這個數字裡。

### 這份紀錄不宣稱什麼

`incomplete_by_design` 直接寫在檔案裡：本體解析時解析的 imports、parser 與 builder
的版本、建圖參數，**都不在紀錄範圍內**。它是**來源檔清單，不是重建配方**。

### 三種狀態

| 狀態 | 意義 | 該怎麼看 |
|---|---|---|
| 沒有這個檔案 | 這個 workspace 建於此功能之前 | **unknown**。不會、也不該用現在快取裡的檔案回填 |
| 有，且與 `kg.json` 相符 | 這個圖的來源 | 可信 |
| **有，但缺檔／損壞／不相符** | 一個不成立的宣稱 | **回報為此狀態**，不等同 unknown |

第三種最重要：兩個 graph-only workspace 在搬移時互換了 provenance 檔案，**每個檔案都
合法**，只有配對錯了。`workspace_provenance_status()` 比對紀錄裡的 `kg_digest` 與現場的
`kg.json` 就能分辨。

**來源驗證不會擋下服務。** 來源狀態是「這個圖從哪裡來」的敘述，不是「這個圖能不能
用」的判定——後者由 `kg.json` 與三個張量的 digest 綁定負責，那些仍然會拒絕。
`workspace_provenance_status()` **不拋例外**，只回傳狀態；要不要因為來源有問題而停止
什麼，是呼叫端的政策決定，目前沒有任何程式這樣做。這個承諾同時涵蓋磁碟層面，不只是
內容層面：`kg.json`、來源紀錄或 `manifest.json` 若本行程無法開啟或解碼，會以
`unreadable` 狀態回傳並在 detail 裡帶上原因，而不是從一個所有呼叫端都沒有包
try 的函數裡丟出例外。manifest 這一項尤其要分清楚——manifest 讀不到的意思是
**「無從得知是否曾宣告過來源紀錄」**，不是**「從未宣告過」**，而只有後者才能推得
「這個 workspace 建於 provenance 之前」。要不要因為 manifest 壞掉而拒絕整個
workspace，仍然是 `verify_graph_artifacts()` 的職責。

> **修正紀錄**：本節的早期版本讓 `verify_graph_artifacts()` 在來源紀錄損壞時直接拋
> 例外。那個函數被 `verify_graph_source()` 呼叫，而後者在 `DiagnosisPipeline`
> 載入張量前執行——結果是「一張便條不見了」會讓模型無法初始化，冷啟動時 `/diagnose`
> 回 503，而張量本身完好無損。已改為回報而非拒絕。

### 舊 workspace

Phase 1 之前建的 schema-3 workspace **不會因為沒有這個檔案而失效**。manifest 沒有
宣告 provenance 時，reader 不會去找它。

### 與本體版本的關係

目前 `latest` 是**快取預設值，不是新鮮度保證**——它的意思是「用快取裡那份；沒有才
下載」，不會每次去確認遠端是否有更新版本。要更換本體版本，換掉快取目錄裡的檔案，
下次建圖的 provenance 就會記下新的 digest 與 `data-version`。

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

### ARM + Blackwell (DGX Spark)

- [ ] ✅ 架構確認為 aarch64
- [ ] ✅ CUDA 13.0 可用
- [ ] ✅ PyTorch 2.9.0 ARM build 正常
- [ ] ✅ PyTorch Geometric 可用（wheel或編譯）
- [ ] ⚠️ 注意力後端已降級（xformers或SDPA）
- [ ] ✅ Vector Index: cuVS (GPU) 或 Voyager (CPU fallback)
- [ ] ✅ 統一記憶體正確識別
- [ ] ✅ 模型可訓練（檢查128GB優勢）

---

**下一步**: 參考 TODO 清單開始開發！

**版本**: v3.2
**維護**: 隨著套件更新持續更新本指南