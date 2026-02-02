"""
SHEPHERD-Advanced Voyager Vector Index Backend
==============================================
功能:
  - 基於 Spotify Voyager 的 HNSW 向量索引
  - 跨平台支援 (Windows, Linux x86, Linux ARM64)
  - CPU 向量檢索，無 GPU 依賴

路徑:
  - 相對路徑: src/retrieval/backends/voyager_backend.py
  - 絕對路徑: SHEPHERD-Advanced/src/retrieval/backends/voyager_backend.py

輸入:
  - embeddings: Dict[str, NDArray[np.float32]] — 實體ID到嵌入向量的映射
  - query: NDArray[np.float32] — 查詢向量 (dim,)

輸出:
  - search(): List[Tuple[str, float]] — (實體ID, 距離) 列表

配置參數 (from configs/deployment.yaml):
  - ef_construction: 建構時的搜尋深度 (預設 200)
  - M: 每個節點的最大連接數 (預設 32)
  - ef_search: 搜尋時的深度 (預設 64)
  - storage_type: 儲存類型 (float32, float8, e4m3)

參考:
  - Voyager 文檔: https://spotify.github.io/voyager/python/
  - 工程藍圖 第3層: 檢索層 (Retrieval Layer)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray

from src.retrieval.backends.base import VectorIndexBase

logger = logging.getLogger(__name__)

# Metric mapping: SHEPHERD metric -> Voyager Space
_METRIC_TO_SPACE = {
    "ip": "InnerProduct",
    "inner_product": "InnerProduct",
    "cosine": "Cosine",
    "l2": "Euclidean",
    "euclidean": "Euclidean",
}


class VoyagerIndex(VectorIndexBase):
    """
    Voyager (Spotify) HNSW 向量索引。

    特點:
    - 比 hnswlib 快 10 倍，記憶體使用減少 4 倍
    - 生產級穩定性 (Spotify 每日數億次查詢)
    - 支援 Windows, Linux x86_64, Linux ARM64
    - Python 3.7-3.12 支援

    使用範例:
        >>> index = VoyagerIndex(dim=768, metric="ip")
        >>> index.build_index({"gene_1": emb1, "gene_2": emb2, ...})
        >>> results = index.search(query_embedding, top_k=10)
        >>> print(results)  # [("gene_1", 0.95), ("gene_2", 0.87), ...]
    """

    def __init__(
        self,
        dim: int = 768,
        metric: str = "ip",
        ef_construction: int = 200,
        M: int = 32,
        ef_search: int = 64,
        storage_type: str = "float32",
        num_threads: int = -1,
        **kwargs: Any,
    ) -> None:
        """
        初始化 Voyager 索引。

        Args:
            dim: 向量維度 (預設 768)
            metric: 相似度度量 ("ip", "cosine", "l2")
            ef_construction: HNSW 建構參數 (越大越精確，越慢)
            M: 每個節點的最大連接數 (影響記憶體和精度)
            ef_search: 搜尋時的深度 (越大越精確，越慢)
            storage_type: 儲存精度 ("float32", "float8", "e4m3")
            num_threads: 並行執行緒數 (-1 = 自動)
        """
        super().__init__(dim=dim, metric=metric, **kwargs)

        self.ef_construction = ef_construction
        self.M = M
        self.ef_search = ef_search
        self.storage_type = storage_type
        # Voyager requires int for num_threads, -1 means auto
        self.num_threads = num_threads if num_threads > 0 else -1

        self._index = None
        self._space = None

        # Validate and import Voyager
        self._validate_voyager()

    def _validate_voyager(self) -> None:
        """檢查 Voyager 是否可用"""
        try:
            import voyager
            self._voyager = voyager

            # Get Space enum
            space_name = _METRIC_TO_SPACE.get(self.metric.lower())
            if space_name is None:
                raise ValueError(
                    f"Unsupported metric: {self.metric}. "
                    f"Supported: {list(_METRIC_TO_SPACE.keys())}"
                )
            self._space = getattr(voyager.Space, space_name)

            # Get version safely (voyager may not have __version__)
            version = getattr(voyager, "__version__", "unknown")
            logger.debug(f"Voyager {version} loaded, space={space_name}")

        except ImportError as e:
            raise ImportError(
                "Voyager is required for VoyagerIndex. "
                "Install with: pip install voyager>=2.0"
            ) from e

    @property
    def backend_name(self) -> str:
        return "voyager"

    def _build_index_impl(self, vectors: NDArray[np.float32]) -> None:
        """建立 Voyager HNSW 索引"""
        # Create index with specified parameters
        self._index = self._voyager.Index(
            space=self._space,
            num_dimensions=self.dim,
            M=self.M,
            ef_construction=self.ef_construction,
        )

        # Add vectors in batch
        logger.debug(f"Adding {len(vectors)} vectors to Voyager index...")
        self._index.add_items(vectors)

        logger.debug(f"Voyager index built: {len(vectors)} vectors, M={self.M}")

    def _search_impl(
        self,
        query: NDArray[np.float32],
        top_k: int,
    ) -> Tuple[NDArray[np.int64], NDArray[np.float32]]:
        """執行單一查詢搜尋"""
        # Voyager query returns (neighbor_ids, distances)
        neighbors, distances = self._index.query(
            query.reshape(1, -1),
            k=top_k,
            num_threads=self.num_threads,
            query_ef=self.ef_search,
        )

        return (
            np.asarray(neighbors[0], dtype=np.int64),
            np.asarray(distances[0], dtype=np.float32),
        )

    def _batch_search_impl(
        self,
        queries: NDArray[np.float32],
        top_k: int,
    ) -> Tuple[NDArray[np.int64], NDArray[np.float32]]:
        """執行批量查詢搜尋"""
        neighbors, distances = self._index.query(
            queries,
            k=top_k,
            num_threads=self.num_threads,
            query_ef=self.ef_search,
        )

        return (
            np.asarray(neighbors, dtype=np.int64),
            np.asarray(distances, dtype=np.float32),
        )

    def _save_impl(self, path: Path) -> None:
        """保存索引到檔案"""
        index_path = path.with_suffix(".voyager")
        self._index.save(str(index_path))
        logger.debug(f"Voyager index saved to {index_path}")

    def _load_impl(self, path: Path) -> None:
        """從檔案載入索引"""
        index_path = path.with_suffix(".voyager")
        if not index_path.exists():
            raise FileNotFoundError(f"Voyager index file not found: {index_path}")

        # Voyager Index.load() requires space and num_dimensions
        self._index = self._voyager.Index.load(
            str(index_path),
            space=self._space,
            num_dimensions=self.dim,
        )
        logger.debug(f"Voyager index loaded from {index_path}")

    def get_vector(self, entity_id: str) -> NDArray[np.float32] | None:
        """
        獲取實體的嵌入向量。

        注意：Voyager 支援透過 index.get_vectors() 獲取向量，
        但需要額外的記憶體開銷。
        """
        if entity_id not in self._id_to_idx:
            return None

        idx = self._id_to_idx[entity_id]
        try:
            # Voyager 2.0+ supports get_vectors
            vectors = self._index.get_vectors([idx])
            return np.asarray(vectors[0], dtype=np.float32)
        except (AttributeError, NotImplementedError):
            # Older versions may not support this
            return None
