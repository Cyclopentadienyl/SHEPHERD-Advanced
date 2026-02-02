"""
SHEPHERD-Advanced Vector Index Factory
======================================
功能:
  - 向量索引後端的工廠函數
  - 自動偵測平台並選擇最佳後端
  - 支援從配置檔載入參數

路徑:
  - 相對路徑: src/retrieval/vector_index.py
  - 絕對路徑: SHEPHERD-Advanced/src/retrieval/vector_index.py

輸入:
  - backend: str — 後端名稱 ("auto", "cuvs", "voyager")
  - config: Dict — 後端配置參數

輸出:
  - VectorIndexBase — 向量索引實例

後端選擇策略:
  - Linux (x86/ARM): cuVS (GPU) → Voyager (CPU fallback)
  - Windows: Voyager (CPU only)

參考:
  - configs/deployment.yaml: retrieval_backend 配置
  - 工程藍圖 第3層: 檢索層 (Retrieval Layer)
"""
from __future__ import annotations

import importlib.util
import logging
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from src.retrieval.backends.base import VectorIndexBase

logger = logging.getLogger(__name__)

# Backend registry
_BACKEND_REGISTRY: Dict[str, Type[VectorIndexBase]] = {}


def _register_backends() -> None:
    """註冊可用的後端"""
    global _BACKEND_REGISTRY

    # Always register Voyager (cross-platform)
    try:
        from src.retrieval.backends.voyager_backend import VoyagerIndex
        _BACKEND_REGISTRY["voyager"] = VoyagerIndex
        logger.debug("Registered backend: voyager")
    except ImportError as e:
        logger.warning(f"Failed to register voyager backend: {e}")

    # Register cuVS (Linux only)
    if sys.platform != "win32":
        try:
            from src.retrieval.backends.cuvs_backend import CuVSIndex
            _BACKEND_REGISTRY["cuvs"] = CuVSIndex
            logger.debug("Registered backend: cuvs")
        except ImportError as e:
            logger.debug(f"cuVS not available: {e}")


# Initialize registry on module load
_register_backends()


def get_platform_key() -> str:
    """
    獲取平台標識符。

    Returns:
        平台標識符 (e.g., "linux_x86_64", "linux_aarch64", "windows_x86_64")
    """
    os_name = "windows" if sys.platform.startswith("win") else "linux"
    arch = platform.machine().lower()

    # Normalize architecture names
    if arch in {"amd64", "x86_64"}:
        arch = "x86_64"
    elif arch in {"aarch64", "arm64"}:
        arch = "aarch64"

    return f"{os_name}_{arch}"


def is_gpu_available() -> bool:
    """
    檢查 GPU 是否可用。

    Returns:
        True if CUDA GPU is available
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        pass

    # Fallback: check for cupy
    try:
        import cupy as cp
        cp.cuda.Device(0).compute_capability
        return True
    except Exception:
        pass

    return False


def resolve_backend(
    requested: str = "auto",
    fallback_chain: Optional[List[str]] = None,
) -> str:
    """
    解析最佳可用後端。

    Args:
        requested: 請求的後端 ("auto", "cuvs", "voyager")
        fallback_chain: 備選後端列表

    Returns:
        可用的後端名稱

    Raises:
        RuntimeError: 如果沒有可用的後端
    """
    platform_key = get_platform_key()

    # Default fallback chains per platform
    if fallback_chain is None:
        if platform_key.startswith("windows"):
            fallback_chain = ["voyager"]
        else:
            fallback_chain = ["cuvs", "voyager"]

    # Handle "auto" by using platform default
    if requested == "auto":
        if platform_key.startswith("windows"):
            requested = "voyager"
        elif is_gpu_available():
            requested = "cuvs"
        else:
            requested = "voyager"

    # Check if requested backend is available
    if requested in _BACKEND_REGISTRY:
        # Verify it can actually be instantiated
        try:
            # Quick validation check
            backend_cls = _BACKEND_REGISTRY[requested]
            if requested == "cuvs" and sys.platform == "win32":
                raise RuntimeError("cuVS not supported on Windows")
            return requested
        except Exception as e:
            logger.warning(f"Backend {requested} check failed: {e}")

    # Try fallback chain
    for backend in fallback_chain:
        if backend in _BACKEND_REGISTRY:
            logger.info(f"Using fallback backend: {backend}")
            return backend

    raise RuntimeError(
        f"No available vector index backend. "
        f"Requested: {requested}, Tried: {fallback_chain}. "
        f"Install voyager: pip install voyager>=2.0"
    )


def create_index(
    backend: str = "auto",
    dim: int = 768,
    metric: str = "ip",
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> VectorIndexBase:
    """
    創建向量索引實例。

    Args:
        backend: 後端名稱 ("auto", "cuvs", "voyager")
        dim: 向量維度
        metric: 相似度度量 ("ip", "cosine", "l2")
        config: 後端特定配置
        **kwargs: 額外參數

    Returns:
        向量索引實例

    使用範例:
        >>> # 自動選擇最佳後端
        >>> index = create_index(backend="auto", dim=768)

        >>> # 指定 Voyager 後端
        >>> index = create_index(backend="voyager", dim=768, M=48)

        >>> # 從配置檔載入
        >>> config = {"ef_construction": 400, "M": 48}
        >>> index = create_index(backend="voyager", config=config)
    """
    # Resolve backend
    resolved_backend = resolve_backend(backend)

    # Get backend class
    if resolved_backend not in _BACKEND_REGISTRY:
        raise ValueError(f"Unknown backend: {resolved_backend}")

    backend_cls = _BACKEND_REGISTRY[resolved_backend]

    # Merge config
    merged_config = config.copy() if config else {}
    merged_config.update(kwargs)

    # Create instance
    logger.info(f"Creating {resolved_backend} index: dim={dim}, metric={metric}")
    return backend_cls(dim=dim, metric=metric, **merged_config)


def create_index_from_config(
    deployment_config: Dict[str, Any],
) -> VectorIndexBase:
    """
    從部署配置創建向量索引。

    Args:
        deployment_config: 從 configs/deployment.yaml 載入的配置

    Returns:
        向量索引實例

    使用範例:
        >>> import yaml
        >>> with open("configs/deployment.yaml") as f:
        ...     config = yaml.safe_load(f)
        >>> index = create_index_from_config(config)
    """
    # Extract retrieval config
    retrieval_cfg = deployment_config.get("retrieval_backend", {})
    indexing_cfg = deployment_config.get("_indexing", {})

    # Get backend preference
    backend = retrieval_cfg.get("default", "auto")
    fallback_chain = retrieval_cfg.get("fallback_chain", ["voyager"])

    # Resolve backend
    resolved = resolve_backend(backend, fallback_chain)

    # Get backend-specific config
    backend_config = indexing_cfg.get(resolved, {})

    # Get global indexing params
    dim = indexing_cfg.get("dim", 768)
    metric = indexing_cfg.get("metric", "ip")

    return create_index(
        backend=resolved,
        dim=dim,
        metric=metric,
        config=backend_config,
    )


def list_available_backends() -> List[str]:
    """
    列出所有可用的後端。

    Returns:
        後端名稱列表
    """
    return list(_BACKEND_REGISTRY.keys())


# Convenience aliases
def get_index(backend: str = "auto", **kwargs: Any) -> VectorIndexBase:
    """create_index 的別名"""
    return create_index(backend=backend, **kwargs)


# For backwards compatibility with older code
def resolve_backend_legacy() -> str:
    """
    舊版後端解析函數 (deprecated)。

    請改用 resolve_backend()。
    """
    import warnings
    warnings.warn(
        "resolve_backend_legacy() is deprecated. Use resolve_backend() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return resolve_backend()
