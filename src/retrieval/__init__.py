"""
SHEPHERD-Advanced Retrieval Module
==================================
功能:
  - 向量索引工廠和後端管理
  - 自動偵測平台並選擇最佳後端
  - 支援 cuVS (Linux GPU) 和 Voyager (跨平台 CPU)

路徑:
  - 相對路徑: src/retrieval/__init__.py
  - 絕對路徑: SHEPHERD-Advanced/src/retrieval/__init__.py

主要介面:
  - create_index(): 工廠函數，創建向量索引實例
  - resolve_backend(): 解析最佳可用後端
  - list_available_backends(): 列出所有可用後端

後端選擇策略:
  - Linux (x86/ARM): cuVS (GPU) → Voyager (CPU fallback)
  - Windows: Voyager (CPU only)
"""
from src.retrieval.vector_index import (
    create_index,
    create_index_from_config,
    get_index,
    list_available_backends,
    resolve_backend,
)
from src.retrieval.backends import VectorIndexBase, VoyagerIndex

# Re-export CuVSIndex only if available
try:
    from src.retrieval.backends import CuVSIndex
except ImportError:
    CuVSIndex = None  # type: ignore

__all__ = [
    # Factory functions
    "create_index",
    "create_index_from_config",
    "get_index",
    "resolve_backend",
    "list_available_backends",
    # Backend classes
    "VectorIndexBase",
    "VoyagerIndex",
    "CuVSIndex",
]
