"""
SHEPHERD-Advanced Retrieval Backends
====================================
功能:
  - 向量索引後端實作集合
  - 提供統一的 VectorIndexProtocol 實作

路徑:
  - 相對路徑: src/retrieval/backends/__init__.py
  - 絕對路徑: SHEPHERD-Advanced/src/retrieval/backends/__init__.py

輸出:
  - VectorIndexBase: 抽象基類
  - VoyagerIndex: Voyager 後端 (跨平台 CPU)
  - CuVSIndex: cuVS 後端 (Linux GPU)
"""
from src.retrieval.backends.base import VectorIndexBase
from src.retrieval.backends.voyager_backend import VoyagerIndex

# cuVS is Linux-only; import conditionally
try:
    from src.retrieval.backends.cuvs_backend import CuVSIndex
except ImportError:
    CuVSIndex = None  # type: ignore

__all__ = [
    "VectorIndexBase",
    "VoyagerIndex",
    "CuVSIndex",
]
