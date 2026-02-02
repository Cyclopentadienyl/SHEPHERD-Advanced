"""
SHEPHERD-Advanced cuVS Vector Index Backend
===========================================
功能:
  - 基於 NVIDIA RAPIDS cuVS 的 GPU 加速向量索引
  - 支援 Linux x86_64 和 Linux ARM64 (DGX Spark)
  - CUDA 13.0 + Blackwell GPU 原生支援

路徑:
  - 相對路徑: src/retrieval/backends/cuvs_backend.py
  - 絕對路徑: SHEPHERD-Advanced/src/retrieval/backends/cuvs_backend.py

輸入:
  - embeddings: Dict[str, NDArray[np.float32]] — 實體ID到嵌入向量的映射
  - query: NDArray[np.float32] — 查詢向量 (dim,)

輸出:
  - search(): List[Tuple[str, float]] — (實體ID, 距離) 列表

配置參數 (from configs/deployment.yaml):
  - build_algo: 建構演算法 (ivf_flat, ivf_pq, cagra)
  - search_algo: 搜尋演算法 (auto)
  - n_lists: IVF 分區數量 (預設 4096)
  - n_probes: 搜尋時探測的分區數 (預設 32)

平台限制:
  - 僅支援 Linux (x86_64, aarch64)
  - Windows 不支援 (使用 Voyager 作為替代)

參考:
  - cuVS 文檔: https://docs.rapids.ai/api/cuvs/stable/
  - 工程藍圖 第3層: 檢索層 (Retrieval Layer)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray

from src.retrieval.backends.base import VectorIndexBase

logger = logging.getLogger(__name__)


def _is_cuvs_available() -> bool:
    """檢查 cuVS 是否可用"""
    if sys.platform == "win32":
        return False
    try:
        import cuvs  # noqa: F401
        return True
    except ImportError:
        return False


class CuVSIndex(VectorIndexBase):
    """
    cuVS (NVIDIA RAPIDS) GPU 加速向量索引。

    特點:
    - GPU 原生 ANN 演算法 (IVF-Flat, IVF-PQ, CAGRA)
    - CUDA 13.0 + Blackwell (CC 10.0) 支援
    - 比 CPU 方案快 10-100 倍
    - 支援 Linux x86_64 和 ARM64 (DGX Spark)

    平台限制:
    - 僅支援 Linux
    - 需要 NVIDIA GPU 和 CUDA 13.0+

    使用範例:
        >>> index = CuVSIndex(dim=768, metric="ip")
        >>> index.build_index({"gene_1": emb1, "gene_2": emb2, ...})
        >>> results = index.search(query_embedding, top_k=10)
        >>> print(results)  # [("gene_1", 0.95), ("gene_2", 0.87), ...]
    """

    def __init__(
        self,
        dim: int = 768,
        metric: str = "ip",
        build_algo: str = "ivf_pq",
        search_algo: str = "auto",
        n_lists: int = 4096,
        n_probes: int = 32,
        **kwargs: Any,
    ) -> None:
        """
        初始化 cuVS 索引。

        Args:
            dim: 向量維度 (預設 768)
            metric: 相似度度量 ("ip", "l2")
            build_algo: 建構演算法 ("ivf_flat", "ivf_pq", "cagra")
            search_algo: 搜尋演算法 ("auto")
            n_lists: IVF 分區數量 (越大越精確，建構越慢)
            n_probes: 搜尋時探測的分區數 (越大越精確，越慢)
        """
        super().__init__(dim=dim, metric=metric, **kwargs)

        self.build_algo = build_algo
        self.search_algo = search_algo
        self.n_lists = n_lists
        self.n_probes = n_probes

        self._index = None
        self._index_params = None
        self._search_params = None

        # Validate platform and import cuVS
        self._validate_cuvs()

    def _validate_cuvs(self) -> None:
        """檢查 cuVS 是否可用"""
        if sys.platform == "win32":
            raise RuntimeError(
                "cuVS is not supported on Windows. "
                "Use VoyagerIndex instead, or run on Linux with WSL2."
            )

        try:
            # Import cuVS components
            from cuvs.neighbors import ivf_flat, ivf_pq  # noqa: F401
            import cupy as cp  # noqa: F401

            self._cuvs_ivf_flat = ivf_flat
            self._cuvs_ivf_pq = ivf_pq
            self._cp = cp

            logger.debug("cuVS loaded successfully")

        except ImportError as e:
            raise ImportError(
                "cuVS is required for GPU-accelerated vector search. "
                "Install with: pip install cuvs-cu13 --extra-index-url=https://pypi.nvidia.com"
            ) from e

    @property
    def backend_name(self) -> str:
        return "cuvs"

    def _get_metric_type(self) -> str:
        """轉換度量類型為 cuVS 格式"""
        metric_map = {
            "ip": "inner_product",
            "inner_product": "inner_product",
            "l2": "sqeuclidean",
            "euclidean": "sqeuclidean",
            "cosine": "inner_product",  # Normalize vectors for cosine
        }
        return metric_map.get(self.metric.lower(), "inner_product")

    def _build_index_impl(self, vectors: NDArray[np.float32]) -> None:
        """建立 cuVS 索引"""
        # Transfer to GPU
        dataset_gpu = self._cp.asarray(vectors, dtype=self._cp.float32)

        # Determine n_lists based on dataset size
        n_vectors = len(vectors)
        n_lists = min(self.n_lists, max(1, n_vectors // 40))

        if self.build_algo == "ivf_flat":
            # Build IVF-Flat index
            self._index_params = self._cuvs_ivf_flat.IndexParams(
                n_lists=n_lists,
                metric=self._get_metric_type(),
            )
            self._index = self._cuvs_ivf_flat.build(
                self._index_params,
                dataset_gpu,
            )

        elif self.build_algo == "ivf_pq":
            # Build IVF-PQ index (more memory efficient)
            # PQ subquantizers: typically dim/4 or dim/8
            pq_dim = max(1, self.dim // 8)

            self._index_params = self._cuvs_ivf_pq.IndexParams(
                n_lists=n_lists,
                metric=self._get_metric_type(),
                pq_dim=pq_dim,
            )
            self._index = self._cuvs_ivf_pq.build(
                self._index_params,
                dataset_gpu,
            )

        else:
            raise ValueError(
                f"Unsupported build algorithm: {self.build_algo}. "
                "Supported: ivf_flat, ivf_pq"
            )

        # Configure search parameters
        if self.build_algo == "ivf_flat":
            self._search_params = self._cuvs_ivf_flat.SearchParams(
                n_probes=min(self.n_probes, n_lists),
            )
        else:
            self._search_params = self._cuvs_ivf_pq.SearchParams(
                n_probes=min(self.n_probes, n_lists),
            )

        logger.debug(
            f"cuVS {self.build_algo} index built: "
            f"{n_vectors} vectors, n_lists={n_lists}"
        )

    def _search_impl(
        self,
        query: NDArray[np.float32],
        top_k: int,
    ) -> Tuple[NDArray[np.int64], NDArray[np.float32]]:
        """執行單一查詢搜尋"""
        # Transfer query to GPU
        query_gpu = self._cp.asarray(
            query.reshape(1, -1),
            dtype=self._cp.float32,
        )

        # Search
        if self.build_algo == "ivf_flat":
            distances, indices = self._cuvs_ivf_flat.search(
                self._search_params,
                self._index,
                query_gpu,
                top_k,
            )
        else:
            distances, indices = self._cuvs_ivf_pq.search(
                self._search_params,
                self._index,
                query_gpu,
                top_k,
            )

        # Transfer results back to CPU
        return (
            self._cp.asnumpy(indices[0]).astype(np.int64),
            self._cp.asnumpy(distances[0]).astype(np.float32),
        )

    def _batch_search_impl(
        self,
        queries: NDArray[np.float32],
        top_k: int,
    ) -> Tuple[NDArray[np.int64], NDArray[np.float32]]:
        """執行批量查詢搜尋"""
        # Transfer queries to GPU
        queries_gpu = self._cp.asarray(queries, dtype=self._cp.float32)

        # Search
        if self.build_algo == "ivf_flat":
            distances, indices = self._cuvs_ivf_flat.search(
                self._search_params,
                self._index,
                queries_gpu,
                top_k,
            )
        else:
            distances, indices = self._cuvs_ivf_pq.search(
                self._search_params,
                self._index,
                queries_gpu,
                top_k,
            )

        # Transfer results back to CPU
        return (
            self._cp.asnumpy(indices).astype(np.int64),
            self._cp.asnumpy(distances).astype(np.float32),
        )

    def _save_impl(self, path: Path) -> None:
        """保存索引到檔案"""
        # cuVS indices can be serialized
        index_path = path.with_suffix(".cuvs")

        # Save index using cuVS serialization
        # Note: cuVS 24.12+ supports save/load
        try:
            if self.build_algo == "ivf_flat":
                self._cuvs_ivf_flat.save(str(index_path), self._index)
            else:
                self._cuvs_ivf_pq.save(str(index_path), self._index)
            logger.debug(f"cuVS index saved to {index_path}")
        except AttributeError:
            # Fallback: save as numpy arrays
            logger.warning(
                "cuVS save not available; index will need to be rebuilt on load"
            )
            # Save placeholder to indicate algo type
            np.save(str(index_path), np.array([self.build_algo], dtype=object))

    def _load_impl(self, path: Path) -> None:
        """從檔案載入索引"""
        index_path = path.with_suffix(".cuvs")

        if not index_path.exists():
            raise FileNotFoundError(f"cuVS index file not found: {index_path}")

        try:
            if self.build_algo == "ivf_flat":
                self._index = self._cuvs_ivf_flat.load(str(index_path))
                self._search_params = self._cuvs_ivf_flat.SearchParams(
                    n_probes=self.n_probes
                )
            else:
                self._index = self._cuvs_ivf_pq.load(str(index_path))
                self._search_params = self._cuvs_ivf_pq.SearchParams(
                    n_probes=self.n_probes
                )
            logger.debug(f"cuVS index loaded from {index_path}")
        except AttributeError:
            raise RuntimeError(
                "cuVS load not available. "
                "Please rebuild the index using build_index()."
            )
