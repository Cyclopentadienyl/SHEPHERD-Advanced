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

# Backend registry. Rebuilt by `_register_backends`, never replaced — callers that
# hold a reference to this object keep seeing the current answer.
_BACKEND_REGISTRY: Dict[str, Type[VectorIndexBase]] = {}

#: What each backend needs before it can be constructed **at all**, plus where its
#: class lives. The package names are the ones the backend imports on the way to a
#: working index, not the ones it imports somewhere:
#: `CuVSIndex._validate_cuvs` imports `cuvs.neighbors` **and** `cupy`, so a host
#: with cuvs and without cupy cannot construct one — a state this repository has
#: already observed on a real deployment machine and records in
#: `scripts/validate_installation.py`.
#:
#: Top-level package names on purpose. `find_spec("cuvs.neighbors")` imports the
#: parent `cuvs` as a side effect and raises `ModuleNotFoundError` when it is
#: absent; `find_spec("cuvs")` does neither, which is what keeps this cheap enough
#: to run at import time.
_BACKEND_SPECS: Dict[str, "tuple"] = {
    "voyager": (("voyager",), "src.retrieval.backends.voyager_backend", "VoyagerIndex"),
    "cuvs": (("cuvs", "cupy"), "src.retrieval.backends.cuvs_backend", "CuVSIndex"),
}

#: Backends that cannot run on this platform whatever is installed. Separate from
#: `_BACKEND_SPECS` because it is a support statement, not a dependency.
_WINDOWS_UNSUPPORTED = frozenset({"cuvs"})


def _discoverable(package: str) -> bool:
    """Whether `package` can be found on this interpreter's path.

    **Discovery, and nothing beyond it.** A true answer means the immediate
    required package is findable. It does not mean the package imports, that its
    CUDA build matches the driver, that an index can be constructed, or that a
    search returns anything. `scripts/validate_installation.py` is where the
    deeper states are distinguished, and this deliberately does not duplicate it.

    **Fails closed.** `find_spec` raises rather than returning for several ordinary
    conditions — `ModuleNotFoundError` when a parent package is absent,
    `ValueError` when a module is already imported but carries no `__spec__`. An
    optional-dependency probe must never be the reason `import src.retrieval`
    fails, so anything it raises is read as "not discoverable".
    """
    try:
        return importlib.util.find_spec(package) is not None
    except (ImportError, ValueError):
        # ModuleNotFoundError is an ImportError; both are discovery outcomes here.
        return False


def _register_backends() -> None:
    """Register the backends whose immediate required packages are discoverable.

    **Registration is what selection, fallback and reporting all read.**
    `resolve_backend`, `create_index` and `list_available_backends` consume this
    registry, so a backend registered without its dependencies is not one wrong
    answer but three: `auto` selects it, the fallback chain beneath never runs
    because nothing raised, and `build_index.py` prints it to the operator as
    available. That was the state before this function checked anything — both
    backends import their dependency lazily, inside a method, so the module import
    this used to guard on never failed and every backend was always registered.

    **Rebuilt, not accumulated.** Discovery is redone from scratch and the registry
    is cleared before the new answers land, so a backend registered by an earlier
    call cannot survive as falsely available once its dependency is gone. Cleared
    in place rather than rebound, so a caller holding this dict is not left
    reading a detached copy.
    """
    discovered: Dict[str, Type[VectorIndexBase]] = {}

    for name, (requirements, module_path, class_name) in _BACKEND_SPECS.items():
        if sys.platform == "win32" and name in _WINDOWS_UNSUPPORTED:
            logger.debug("Backend %s: not registered, unsupported on Windows", name)
            continue

        missing = [pkg for pkg in requirements if not _discoverable(pkg)]
        if missing:
            logger.debug("Backend %s: not registered, %s not discoverable", name, missing)
            continue

        try:
            module = importlib.import_module(module_path)
            discovered[name] = getattr(module, class_name)
            logger.debug("Registered backend: %s", name)
        except (ImportError, AttributeError) as exc:
            # The backend module itself is broken or has been renamed. Distinct
            # from a missing dependency, and worth more than debug.
            logger.warning("Backend %s: %s could not be loaded (%s)", name, module_path, exc)

    _BACKEND_REGISTRY.clear()
    _BACKEND_REGISTRY.update(discovered)


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
    else:
        # Explicit backend requested - validate it exists
        valid_backends = set(_BACKEND_REGISTRY.keys()) | {"auto"}
        if requested not in valid_backends:
            raise ValueError(
                f"Backend '{requested}' not available. "
                f"Valid options: {sorted(valid_backends)}"
            )

    # Availability is decided by `_register_backends`, not here: a name is in the
    # registry only if its immediate required packages were discoverable and the
    # platform supports it. This block used to be commented "Verify it can actually
    # be instantiated" / "Quick validation check", and it never verified anything —
    # the only condition it tested was Windows, so an absent package sailed through
    # and the fallback chain below could not run.
    #
    # The Windows branch below is now unreachable for the same reason: cuVS is not
    # registered on Windows at all. Left as it stands rather than simplified,
    # because `resolve_backend` is not what this change is about.
    if requested in _BACKEND_REGISTRY:
        try:
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
