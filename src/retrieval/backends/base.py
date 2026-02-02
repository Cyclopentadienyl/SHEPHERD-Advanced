"""
SHEPHERD-Advanced Vector Index Base Class
==========================================
功能:
  - 定義向量索引的抽象基類
  - 實作 VectorIndexProtocol 介面
  - 提供共用的輔助方法

路徑:
  - 相對路徑: src/retrieval/backends/base.py
  - 絕對路徑: SHEPHERD-Advanced/src/retrieval/backends/base.py

輸入:
  - embeddings: Dict[str, NDArray[np.float32]] — 實體ID到嵌入向量的映射
  - query: NDArray[np.float32] — 查詢向量 (dim,)
  - queries: List[NDArray[np.float32]] — 批量查詢向量

輸出:
  - search(): List[Tuple[str, float]] — (實體ID, 距離) 列表
  - batch_search(): List[List[Tuple[str, float]]] — 批量搜尋結果

參考:
  - 工程藍圖 第3層: 檢索層 (Retrieval Layer)
  - src/core/protocols.py: VectorIndexProtocol
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from src.core.types import EmbeddingType

logger = logging.getLogger(__name__)


class VectorIndexBase(ABC):
    """
    向量索引抽象基類

    所有向量檢索後端必須繼承此類並實作抽象方法。
    實作 src/core/protocols.py 中定義的 VectorIndexProtocol。

    Attributes:
        dim: 向量維度 (預設 768，對應 sentence-transformers)
        metric: 相似度度量 ("ip" = inner product, "cosine", "l2")
        _id_to_idx: 實體ID到內部索引的映射
        _idx_to_id: 內部索引到實體ID的映射
    """

    def __init__(
        self,
        dim: int = 768,
        metric: str = "ip",
        **kwargs: Any,
    ) -> None:
        """
        初始化向量索引。

        Args:
            dim: 向量維度
            metric: 相似度度量 ("ip", "cosine", "l2")
            **kwargs: 後端特定參數
        """
        self.dim = dim
        self.metric = metric
        self._id_to_idx: Dict[str, int] = {}
        self._idx_to_id: List[str] = []
        self._is_built = False

        # Store backend-specific config
        self._config = kwargs

        logger.info(
            f"Initialized {self.backend_name} index: dim={dim}, metric={metric}"
        )

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """返回後端名稱 (e.g., 'voyager', 'cuvs')"""
        ...

    @abstractmethod
    def _build_index_impl(self, vectors: NDArray[np.float32]) -> None:
        """
        內部實作：建立索引。

        Args:
            vectors: 形狀 (N, dim) 的向量矩陣
        """
        ...

    @abstractmethod
    def _search_impl(
        self,
        query: NDArray[np.float32],
        top_k: int,
    ) -> Tuple[NDArray[np.int64], NDArray[np.float32]]:
        """
        內部實作：單一查詢搜尋。

        Args:
            query: 形狀 (dim,) 的查詢向量
            top_k: 返回的最近鄰數量

        Returns:
            (indices, distances): 形狀均為 (top_k,)
        """
        ...

    @abstractmethod
    def _batch_search_impl(
        self,
        queries: NDArray[np.float32],
        top_k: int,
    ) -> Tuple[NDArray[np.int64], NDArray[np.float32]]:
        """
        內部實作：批量查詢搜尋。

        Args:
            queries: 形狀 (B, dim) 的查詢向量矩陣
            top_k: 每個查詢返回的最近鄰數量

        Returns:
            (indices, distances): 形狀均為 (B, top_k)
        """
        ...

    @abstractmethod
    def _save_impl(self, path: Path) -> None:
        """內部實作：保存索引到檔案"""
        ...

    @abstractmethod
    def _load_impl(self, path: Path) -> None:
        """內部實作：從檔案載入索引"""
        ...

    # =========================================================================
    # Public API (VectorIndexProtocol implementation)
    # =========================================================================

    def build_index(self, embeddings: Dict[str, EmbeddingType]) -> None:
        """
        建立向量索引。

        Args:
            embeddings: 實體ID到嵌入向量的映射
                        {entity_id: np.ndarray of shape (dim,)}
        """
        if not embeddings:
            raise ValueError("Cannot build index from empty embeddings")

        # Build ID mappings
        self._idx_to_id = list(embeddings.keys())
        self._id_to_idx = {eid: idx for idx, eid in enumerate(self._idx_to_id)}

        # Stack vectors into matrix
        vectors = np.stack(
            [embeddings[eid] for eid in self._idx_to_id],
            axis=0,
        ).astype(np.float32)

        # Validate dimensions
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dim}, got {vectors.shape[1]}"
            )

        logger.info(f"Building {self.backend_name} index with {len(self._idx_to_id)} vectors")
        self._build_index_impl(vectors)
        self._is_built = True
        logger.info(f"Index built successfully: {len(self._idx_to_id)} vectors indexed")

    def search(
        self,
        query: EmbeddingType,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        向量搜尋：找出最相似的 top_k 個實體。

        Args:
            query: 查詢向量，形狀 (dim,)
            top_k: 返回的最近鄰數量

        Returns:
            List of (entity_id, distance) tuples, sorted by distance
        """
        if not self._is_built:
            raise RuntimeError("Index not built. Call build_index() first.")

        query = np.asarray(query, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # Clamp top_k to available vectors
        top_k = min(top_k, len(self._idx_to_id))

        indices, distances = self._search_impl(query.squeeze(), top_k)

        # Convert indices to entity IDs
        results = [
            (self._idx_to_id[int(idx)], float(dist))
            for idx, dist in zip(indices, distances)
            if 0 <= idx < len(self._idx_to_id)
        ]

        return results

    def batch_search(
        self,
        queries: List[EmbeddingType],
        top_k: int = 10,
    ) -> List[List[Tuple[str, float]]]:
        """
        批量向量搜尋。

        Args:
            queries: 查詢向量列表
            top_k: 每個查詢返回的最近鄰數量

        Returns:
            List of search results for each query
        """
        if not self._is_built:
            raise RuntimeError("Index not built. Call build_index() first.")

        if not queries:
            return []

        # Stack queries into matrix
        query_matrix = np.stack(
            [np.asarray(q, dtype=np.float32) for q in queries],
            axis=0,
        )

        # Clamp top_k
        top_k = min(top_k, len(self._idx_to_id))

        all_indices, all_distances = self._batch_search_impl(query_matrix, top_k)

        # Convert to list of results
        results = []
        for indices, distances in zip(all_indices, all_distances):
            query_results = [
                (self._idx_to_id[int(idx)], float(dist))
                for idx, dist in zip(indices, distances)
                if 0 <= idx < len(self._idx_to_id)
            ]
            results.append(query_results)

        return results

    def save(self, path: Path) -> None:
        """
        保存索引到檔案。

        Args:
            path: 保存路徑 (不含副檔名，後端會自動添加)
        """
        if not self._is_built:
            raise RuntimeError("Cannot save: index not built")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save ID mappings
        import json
        mapping_path = path.with_suffix(".ids.json")
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(self._idx_to_id, f)

        # Save index data
        self._save_impl(path)
        logger.info(f"Index saved to {path}")

    def load(self, path: Path) -> None:
        """
        從檔案載入索引。

        Args:
            path: 索引路徑 (不含副檔名)
        """
        path = Path(path)

        # Load ID mappings
        import json
        mapping_path = path.with_suffix(".ids.json")
        if not mapping_path.exists():
            raise FileNotFoundError(f"ID mapping file not found: {mapping_path}")

        with open(mapping_path, "r", encoding="utf-8") as f:
            self._idx_to_id = json.load(f)
        self._id_to_idx = {eid: idx for idx, eid in enumerate(self._idx_to_id)}

        # Load index data
        self._load_impl(path)
        self._is_built = True
        logger.info(f"Index loaded from {path}: {len(self._idx_to_id)} vectors")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def __len__(self) -> int:
        """返回索引中的向量數量"""
        return len(self._idx_to_id)

    def __contains__(self, entity_id: str) -> bool:
        """檢查實體ID是否在索引中"""
        return entity_id in self._id_to_idx

    def get_vector(self, entity_id: str) -> Optional[NDArray[np.float32]]:
        """
        獲取實體的嵌入向量 (如果後端支援)。

        注意：不是所有後端都支援此操作。
        """
        return None  # Override in subclass if supported
