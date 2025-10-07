# 醫療知識圖譜診斷引擎 - 跨平台部署完整指南

**版本**: v2.0  
**最後更新**: 2025-10-07  
**適用環境**: Windows x86 + Blackwell | ARM + Blackwell (DGX Spark)

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
1. **16GB VRAM瓶頸**: 無法載入完整PrimeKG（~500萬節點）
2. **記憶體拷貝開銷**: CPU ↔ GPU數據傳輸慢

---

# 第二部分：套件兼容性詳細分析

## 2.1 核心依賴矩陣

| 套件 | 版本 | Win x86 | ARM | 安裝難度 | 備註 |
|------|------|---------|-----|----------|------|
| **Python** | 3.12 | ✅ | ✅ | 簡單 | 兩者都支持 |
| **PyTorch** | 2.8.0 | ✅ | ✅ | 簡單 | 官方ARM+CUDA wheel |
| **CUDA Toolkit** | 12.8 | ✅ | ✅ | 簡單 | Blackwell要求 |
| **cuDNN** | 9.x | ✅ | ✅ | 簡單 | PyTorch包含 |
| **PyTorch Geometric** | 2.6+ | ✅ | ⚠️ | 中等 | ARM可能需從源碼編譯 |
| **pyg-lib** | 最新 | ✅ | ⚠️ | 中等 | 同上 |
| **torch-scatter** | 最新 | ✅ | ⚠️ | 中等 | 同上 |
| **torch-sparse** | 最新 | ✅ | ⚠️ | 中等 | 同上 |
| **FlashAttention-2** | 2.8+ | ✅ | ❌ | 困難 | **ARM不支持** |
| **xformers** | 0.0.27+ | ✅ | ⚠️ | 困難 | ARM需從源碼編譯 |
| **FAISS (GPU)** | 1.8+ | ✅ | ❌ | 中等 | **ARM不支持GPU版** |
| **hnswlib** | 0.8+ | ✅ | ✅ | 簡單 | 跨平台 |
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
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.8.0+cu128.html

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
pip install dgl -f https://data.dgl.ai/wheels/torch-2.8/cu128/repo.html

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

### 🔴 **FAISS vs hnswlib**

**FAISS GPU on ARM**:
- ❌ FAISS的GPU版本在ARM上支持極差
- CPU版本可用，但慢很多

**解決方案: hnswlib**

```python
# src/retrieval/vector_index.py

import platform
import numpy as np

class CrossPlatformVectorIndex:
    """
    跨平台向量索引
    x86: FAISS GPU
    ARM: hnswlib (CPU優化)
    """
    
    def __init__(self, dimension: int, max_elements: int = 1000000):
        self.dimension = dimension
        self.max_elements = max_elements
        
        arch = platform.machine()
        is_arm = arch in ['aarch64', 'arm64']
        
        if is_arm or not self._check_faiss_gpu():
            # ARM 或 FAISS不可用: 使用hnswlib
            self.backend = 'hnswlib'
            self._init_hnswlib()
        else:
            # x86 + CUDA: 使用FAISS GPU
            self.backend = 'faiss_gpu'
            self._init_faiss_gpu()
        
        print(f"[VectorIndex] Using backend: {self.backend}")
    
    def _check_faiss_gpu(self) -> bool:
        try:
            import faiss
            return faiss.get_num_gpus() > 0
        except:
            return False
    
    def _init_hnswlib(self):
        """hnswlib: HNSW算法，快速近似搜索"""
        import hnswlib
        
        self.index = hnswlib.Index(space='l2', dim=self.dimension)
        self.index.init_index(
            max_elements=self.max_elements,
            ef_construction=200,  # 構建時的搜索深度
            M=16  # 每個節點的連接數
        )
        self.index.set_ef(50)  # 查詢時的搜索深度
        self.current_count = 0
    
    def _init_faiss_gpu(self):
        """FAISS GPU: 精確搜索，極快"""
        import faiss
        
        # CPU索引
        cpu_index = faiss.IndexFlatL2(self.dimension)
        
        # 轉移到GPU
        res = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        self.current_count = 0
    
    def add(self, vectors: np.ndarray):
        """添加向量"""
        if self.backend == 'hnswlib':
            ids = np.arange(self.current_count, self.current_count + len(vectors))
            self.index.add_items(vectors, ids)
        else:  # faiss_gpu
            self.index.add(vectors)
        
        self.current_count += len(vectors)
    
    def search(self, queries: np.ndarray, k: int = 10):
        """搜索最近鄰"""
        if self.backend == 'hnswlib':
            labels, distances = self.index.knn_query(queries, k=k)
            return labels, distances
        else:  # faiss_gpu
            distances, labels = self.index.search(queries, k)
            return labels, distances

# 使用範例
index = CrossPlatformVectorIndex(dimension=512)
vectors = np.random.randn(10000, 512).astype('float32')
index.add(vectors)

queries = np.random.randn(10, 512).astype('float32')
labels, distances = index.search(queries, k=10)
```

**效能對比**:
| 操作 | FAISS GPU (x86) | hnswlib (ARM) | 比率 |
|------|-----------------|---------------|------|
| 構建索引 (100萬向量) | 5s | 45s | 9x |
| 查詢 (batch=100, k=10) | 2ms | 15ms | 7.5x |
| 記憶體使用 | 低（GPU） | 中（CPU） | - |

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
Write-Host "[6/8] 安裝 PyTorch 2.8.0 + CUDA 12.8..." -ForegroundColor Yellow
pip install torch==2.8.0 torchvision==0.19.0 torchaudio==2.8.0 `
    --index-url https://download.pytorch.org/whl/cu128

# 驗證PyTorch
python -c "import torch; assert torch.cuda.is_available(), 'CUDA不可用'; print(f'✅ PyTorch {torch.__version__} + CUDA {torch.version.cuda}')"

# 7. 安裝PyTorch Geometric
Write-Host "[7/8] 安裝 PyTorch Geometric..." -ForegroundColor Yellow
pip install torch-geometric pyg-lib torch-scatter torch-sparse torch-cluster `
    -f https://data.pyg.org/whl/torch-2.8.0+cu128.html

# 8. 安裝其他依賴
Write-Host "[8/8] 安裝其他依賴..." -ForegroundColor Yellow

# FlashAttention-2
Write-Host "  - 安裝 FlashAttention-2 (可能需要幾分鐘)..." -ForegroundColor Cyan
pip install flash-attn --no-build-isolation

# 其他套件
pip install `
    transformers>=4.40.0 `
    owlready2>=0.46 `
    faiss-gpu>=1.8.0 `
    hnswlib>=0.8.0 `
    fastapi>=0.110.0 `
    uvicorn>=0.29.0 `
    pandas>=2.2.0 `
    numpy>=1.26.0 `
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
import faiss

print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ PyG: {pyg.__version__}')
print(f'✅ CUDA可用: {torch.cuda.is_available()}')
print(f'✅ CUDA設備: {torch.cuda.get_device_name(0)}')
print(f'✅ CUDA記憶體: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print(f'✅ FlashAttention: 已安裝')
print(f'✅ FAISS GPU數量: {faiss.get_num_gpus()}')
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
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# 步驟3: PyG (如果上面的安裝失敗)
# 方案A: 使用PyG倉庫
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.8.0+cu128.html

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
echo -e "${YELLOW}[8/10] 安裝 PyTorch 2.8.0 (ARM + CUDA 12.8)...${NC}"
pip install torch==2.8.0 torchvision==0.19.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

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
    hnswlib>=0.8.0 \
    fastapi>=0.110.0 \
    uvicorn>=0.29.0 \
    pandas>=2.2.0 \
    numpy>=1.26.0 \
    tqdm>=4.66.0 \
    pyyaml>=6.0 \
    tensorboard>=2.16.0

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

# 12. 寫入配置文件
echo ""
echo -e "${CYAN}寫入平台配置...${NC}"
cat > config/platform.yaml << EOF
platform: dgx_spark_arm_blackwell
architecture: aarch64
cuda_version: 12.8
memory_gb: 128
unified_memory: true

model_config:
  hidden_dim: 256  # 比x86小（節省記憶體）
  num_layers: 4
  attention_backend: auto  # 自動檢測
  batch_size: 64  # 利用大記憶體優勢

vector_index:
  backend: hnswlib  # ARM上不用FAISS
EOF

echo -e "${GREEN}✅ 配置文件已生成: config/platform.yaml${NC}"

echo ""
echo -e "${GREEN}========================================"
echo -e "  ✅ DGX Spark 環境設置完成！"
echo -e "========================================${NC}"
echo ""
echo -e "下一步："
echo -e "  1. 下載資料: python scripts/download_data.py"
echo -e "  2. 構建知識圖譜: python scripts/build_kg.py"
echo -e "  3. 訓練模型: python scripts/train_model.py --config config/platform.yaml"
echo ""
echo -e "${YELLOW}注意事項：${NC}"
echo -e "  - FlashAttention-2 在 ARM 上不可用，已自動降級"
echo -e "  - 使用 hnswlib 替代 FAISS"
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
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install torch-geometric pyg-lib torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
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

## 總結：部署檢查清單 ✅

### Windows x86 + Blackwell

- [ ] ✅ Python 3.12 安裝
- [ ] ✅ CUDA 12.8 安裝
- [ ] ✅ PyTorch 2.8.0 + CUDA 正常
- [ ] ✅ PyTorch Geometric 正常
- [ ] ✅ FlashAttention-2 安裝成功
- [ ] ✅ FAISS GPU 可用
- [ ] ✅ 所有單元測試通過
- [ ] ✅ 模型可訓練（檢查VRAM使用）

### ARM + Blackwell (DGX Spark)

- [ ] ✅ 架構確認為 aarch64
- [ ] ✅ CUDA 12.8 可用
- [ ] ✅ PyTorch 2.8.0 ARM build 正常
- [ ] ✅ PyTorch Geometric 可用（wheel或編譯）
- [ ] ⚠️ 注意力後端已降級（xformers或SDPA）
- [ ] ✅ hnswlib 替代 FAISS
- [ ] ✅ 統一記憶體正確識別
- [ ] ✅ 模型可訓練（檢查128GB優勢）

---

**下一步**: 參考 TODO 清單開始開發！

**版本**: v2.0  
**維護**: 隨著套件更新持續更新本指南