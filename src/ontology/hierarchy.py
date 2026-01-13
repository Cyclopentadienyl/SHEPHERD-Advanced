"""
SHEPHERD-Advanced Ontology Hierarchy Operations
================================================
本體層次結構處理，包括祖先/後代查詢、語義相似度計算等

版本: 1.0.0
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from src.core.types import DataSource, NodeType

# 避免循環導入
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.ontology.loader import OBOHeader, OBOTerm

logger = logging.getLogger(__name__)


# =============================================================================
# Ontology Class
# =============================================================================
class Ontology:
    """
    本體操作類

    提供:
    - 層次結構遍歷 (祖先、後代、父節點、子節點)
    - 語義相似度計算 (Resnik, Lin, Jiang-Conrath)
    - Information Content (IC) 計算
    - 術語搜尋
    """

    def __init__(
        self,
        header: 'OBOHeader',
        terms: Dict[str, 'OBOTerm'],
    ):
        """
        Args:
            header: OBO header 資訊
            terms: 術語字典 {term_id: OBOTerm}
        """
        self._header = header
        self._terms = terms

        # Build indexes
        self._children: Dict[str, Set[str]] = defaultdict(set)
        self._parents: Dict[str, Set[str]] = defaultdict(set)

        # IC values (computed lazily)
        self._ic_values: Optional[Dict[str, float]] = None
        self._term_descendants_count: Optional[Dict[str, int]] = None

        # Source tracking
        self._source: Optional[DataSource] = None

        # Build parent-child index
        self._build_hierarchy_index()

        logger.info(
            f"Ontology initialized: {self.name} v{self.version} "
            f"({len(self._terms)} terms)"
        )

    def _build_hierarchy_index(self) -> None:
        """建立層次結構索引"""
        for term_id, term in self._terms.items():
            if term.is_obsolete:
                continue

            # IS_A relationships
            for parent_id in term.is_a:
                self._parents[term_id].add(parent_id)
                self._children[parent_id].add(term_id)

            # Part_of relationships (optional, based on use case)
            # for parent_id in term.part_of:
            #     self._parents[term_id].add(parent_id)
            #     self._children[parent_id].add(term_id)

    # =========================================================================
    # Properties
    # =========================================================================
    @property
    def name(self) -> str:
        """本體名稱"""
        return self._header.ontology or "Unknown"

    @property
    def version(self) -> str:
        """本體版本"""
        return self._header.data_version or self._header.format_version or "Unknown"

    @property
    def num_terms(self) -> int:
        """非廢棄術語數量"""
        return sum(1 for t in self._terms.values() if not t.is_obsolete)

    @property
    def root_terms(self) -> Set[str]:
        """獲取根節點 (沒有父節點的術語)"""
        roots = set()
        for term_id, term in self._terms.items():
            if not term.is_obsolete and not term.is_a:
                roots.add(term_id)
        return roots

    # =========================================================================
    # Term Access
    # =========================================================================
    def get_term(self, term_id: str) -> Optional[Dict[str, Any]]:
        """
        獲取單個術語

        Args:
            term_id: 術語 ID (e.g., "HP:0001250")

        Returns:
            術語資訊字典，如果不存在則返回 None
        """
        term = self._terms.get(term_id)
        if term is None:
            return None

        return {
            "id": term.id,
            "name": term.name,
            "namespace": term.namespace,
            "definition": term.definition,
            "synonyms": [s[0] for s in term.synonyms],
            "xrefs": term.xrefs,
            "is_obsolete": term.is_obsolete,
            "parents": list(term.is_a),
        }

    def get_term_name(self, term_id: str) -> Optional[str]:
        """獲取術語名稱"""
        term = self._terms.get(term_id)
        return term.name if term else None

    def has_term(self, term_id: str) -> bool:
        """檢查術語是否存在"""
        return term_id in self._terms

    def is_obsolete(self, term_id: str) -> bool:
        """檢查術語是否已廢棄"""
        term = self._terms.get(term_id)
        return term.is_obsolete if term else True

    def get_replacement(self, term_id: str) -> Optional[str]:
        """獲取廢棄術語的替代術語"""
        term = self._terms.get(term_id)
        if term and term.is_obsolete:
            return term.replaced_by
        return None

    # =========================================================================
    # Hierarchy Traversal
    # =========================================================================
    def get_parents(self, term_id: str) -> Set[str]:
        """
        獲取直接父節點

        Args:
            term_id: 術語 ID

        Returns:
            父節點 ID 集合
        """
        return self._parents.get(term_id, set()).copy()

    def get_children(self, term_id: str) -> Set[str]:
        """
        獲取直接子節點

        Args:
            term_id: 術語 ID

        Returns:
            子節點 ID 集合
        """
        return self._children.get(term_id, set()).copy()

    def get_ancestors(
        self,
        term_id: str,
        include_self: bool = False,
    ) -> Set[str]:
        """
        獲取所有祖先節點 (IS_A 關係的傳遞閉包)

        使用 BFS 遍歷

        Args:
            term_id: 術語 ID
            include_self: 是否包含自身

        Returns:
            祖先節點 ID 集合
        """
        if term_id not in self._terms:
            return set()

        ancestors = set()
        if include_self:
            ancestors.add(term_id)

        queue = deque(self._parents.get(term_id, set()))
        visited = set()

        while queue:
            current = queue.popleft()
            if current in visited:
                continue

            visited.add(current)
            ancestors.add(current)

            for parent in self._parents.get(current, set()):
                if parent not in visited:
                    queue.append(parent)

        return ancestors

    def get_descendants(
        self,
        term_id: str,
        include_self: bool = False,
    ) -> Set[str]:
        """
        獲取所有後代節點

        使用 BFS 遍歷

        Args:
            term_id: 術語 ID
            include_self: 是否包含自身

        Returns:
            後代節點 ID 集合
        """
        if term_id not in self._terms:
            return set()

        descendants = set()
        if include_self:
            descendants.add(term_id)

        queue = deque(self._children.get(term_id, set()))
        visited = set()

        while queue:
            current = queue.popleft()
            if current in visited:
                continue

            visited.add(current)
            descendants.add(current)

            for child in self._children.get(current, set()):
                if child not in visited:
                    queue.append(child)

        return descendants

    def get_depth(self, term_id: str) -> int:
        """
        獲取術語的深度 (到根節點的最短路徑長度)

        Args:
            term_id: 術語 ID

        Returns:
            深度值，根節點為 0
        """
        if term_id not in self._terms:
            return -1

        if not self._parents.get(term_id):
            return 0

        # BFS to find shortest path to root
        visited = {term_id}
        queue = deque([(term_id, 0)])

        while queue:
            current, depth = queue.popleft()

            parents = self._parents.get(current, set())
            if not parents:
                return depth

            for parent in parents:
                if parent not in visited:
                    visited.add(parent)
                    queue.append((parent, depth + 1))

        return -1

    # =========================================================================
    # Lowest Common Ancestor (LCA)
    # =========================================================================
    def get_common_ancestors(
        self,
        term1: str,
        term2: str,
    ) -> Set[str]:
        """
        獲取兩個術語的共同祖先

        Args:
            term1: 第一個術語 ID
            term2: 第二個術語 ID

        Returns:
            共同祖先 ID 集合
        """
        ancestors1 = self.get_ancestors(term1, include_self=True)
        ancestors2 = self.get_ancestors(term2, include_self=True)
        return ancestors1 & ancestors2

    def get_lowest_common_ancestors(
        self,
        term1: str,
        term2: str,
    ) -> Set[str]:
        """
        獲取最低共同祖先 (LCA)

        可能有多個 LCA (在 DAG 結構中)

        Args:
            term1: 第一個術語 ID
            term2: 第二個術語 ID

        Returns:
            LCA ID 集合
        """
        common = self.get_common_ancestors(term1, term2)
        if not common:
            return set()

        # Find terms in common that have no descendants also in common
        lcas = set()
        for term in common:
            descendants = self.get_descendants(term)
            if not (descendants & common):
                lcas.add(term)

        return lcas

    # =========================================================================
    # Information Content (IC)
    # =========================================================================
    def _compute_ic_values(self) -> None:
        """
        計算所有術語的 Information Content

        IC(t) = -log(p(t))
        其中 p(t) = (descendants(t) + 1) / total_terms

        這是基於本體結構的 IC，不需要語料庫
        """
        if self._ic_values is not None:
            return

        logger.info("Computing Information Content values...")

        self._ic_values = {}
        total_terms = self.num_terms

        # Compute descendants count for each term
        self._term_descendants_count = {}
        for term_id in self._terms:
            if not self._terms[term_id].is_obsolete:
                desc_count = len(self.get_descendants(term_id))
                self._term_descendants_count[term_id] = desc_count

        # Compute IC
        for term_id, desc_count in self._term_descendants_count.items():
            # p(t) = probability of randomly selecting a term that is t or its descendant
            p_term = (desc_count + 1) / total_terms
            self._ic_values[term_id] = -math.log(p_term) if p_term > 0 else 0.0

        logger.info(f"Computed IC for {len(self._ic_values)} terms")

    def get_information_content(self, term_id: str) -> float:
        """
        獲取術語的 Information Content

        Args:
            term_id: 術語 ID

        Returns:
            IC 值 (越大表示越具體)
        """
        self._compute_ic_values()
        return self._ic_values.get(term_id, 0.0)

    # =========================================================================
    # Semantic Similarity
    # =========================================================================
    def compute_similarity(
        self,
        term1: str,
        term2: str,
        method: str = "resnik",
    ) -> float:
        """
        計算兩個術語的語義相似度

        Args:
            term1: 第一個術語 ID
            term2: 第二個術語 ID
            method: 相似度方法
                - "resnik": IC of LCA
                - "lin": 2 * IC(LCA) / (IC(t1) + IC(t2))
                - "jiang": 1 - (IC(t1) + IC(t2) - 2 * IC(LCA))
                - "jaccard": |ancestors(t1) ∩ ancestors(t2)| / |ancestors(t1) ∪ ancestors(t2)|

        Returns:
            相似度分數 [0, 1] 或 [0, max_IC]
        """
        if term1 == term2:
            if method == "resnik":
                return self.get_information_content(term1)
            return 1.0

        if term1 not in self._terms or term2 not in self._terms:
            return 0.0

        if method == "resnik":
            return self._similarity_resnik(term1, term2)
        elif method == "lin":
            return self._similarity_lin(term1, term2)
        elif method == "jiang":
            return self._similarity_jiang(term1, term2)
        elif method == "jaccard":
            return self._similarity_jaccard(term1, term2)
        else:
            raise ValueError(f"Unknown similarity method: {method}")

    def _similarity_resnik(self, term1: str, term2: str) -> float:
        """Resnik similarity: IC of LCA"""
        lcas = self.get_lowest_common_ancestors(term1, term2)
        if not lcas:
            return 0.0

        # Return max IC among LCAs
        return max(self.get_information_content(lca) for lca in lcas)

    def _similarity_lin(self, term1: str, term2: str) -> float:
        """Lin similarity: 2 * IC(LCA) / (IC(t1) + IC(t2))"""
        ic1 = self.get_information_content(term1)
        ic2 = self.get_information_content(term2)

        if ic1 + ic2 == 0:
            return 0.0

        ic_lca = self._similarity_resnik(term1, term2)
        return (2 * ic_lca) / (ic1 + ic2)

    def _similarity_jiang(self, term1: str, term2: str) -> float:
        """
        Jiang-Conrath similarity (normalized)

        Original: distance = IC(t1) + IC(t2) - 2 * IC(LCA)
        Normalized to [0, 1]: similarity = 1 / (1 + distance)
        """
        ic1 = self.get_information_content(term1)
        ic2 = self.get_information_content(term2)
        ic_lca = self._similarity_resnik(term1, term2)

        distance = ic1 + ic2 - 2 * ic_lca
        return 1.0 / (1.0 + distance)

    def _similarity_jaccard(self, term1: str, term2: str) -> float:
        """Jaccard similarity based on ancestors"""
        ancestors1 = self.get_ancestors(term1, include_self=True)
        ancestors2 = self.get_ancestors(term2, include_self=True)

        if not ancestors1 and not ancestors2:
            return 0.0

        intersection = len(ancestors1 & ancestors2)
        union = len(ancestors1 | ancestors2)

        return intersection / union if union > 0 else 0.0

    # =========================================================================
    # Set Similarity (for phenotype profiles)
    # =========================================================================
    def compute_set_similarity(
        self,
        terms1: List[str],
        terms2: List[str],
        method: str = "bma",
        term_method: str = "resnik",
    ) -> float:
        """
        計算兩個術語集合的相似度 (用於患者表型比較)

        Args:
            terms1: 第一個術語集合
            terms2: 第二個術語集合
            method: 集合相似度方法
                - "bma": Best Match Average
                - "max": Maximum similarity
                - "avg": Average of all pairs
            term_method: 術語間相似度方法

        Returns:
            集合相似度分數
        """
        if not terms1 or not terms2:
            return 0.0

        # Filter valid terms
        terms1 = [t for t in terms1 if t in self._terms]
        terms2 = [t for t in terms2 if t in self._terms]

        if not terms1 or not terms2:
            return 0.0

        if method == "bma":
            return self._set_similarity_bma(terms1, terms2, term_method)
        elif method == "max":
            return self._set_similarity_max(terms1, terms2, term_method)
        elif method == "avg":
            return self._set_similarity_avg(terms1, terms2, term_method)
        else:
            raise ValueError(f"Unknown set similarity method: {method}")

    def _set_similarity_bma(
        self,
        terms1: List[str],
        terms2: List[str],
        term_method: str,
    ) -> float:
        """Best Match Average"""
        # For each term in set1, find best match in set2
        bm1 = []
        for t1 in terms1:
            best = max(
                self.compute_similarity(t1, t2, term_method)
                for t2 in terms2
            )
            bm1.append(best)

        # For each term in set2, find best match in set1
        bm2 = []
        for t2 in terms2:
            best = max(
                self.compute_similarity(t1, t2, term_method)
                for t1 in terms1
            )
            bm2.append(best)

        # Average of both directions
        return (sum(bm1) / len(bm1) + sum(bm2) / len(bm2)) / 2

    def _set_similarity_max(
        self,
        terms1: List[str],
        terms2: List[str],
        term_method: str,
    ) -> float:
        """Maximum pairwise similarity"""
        max_sim = 0.0
        for t1 in terms1:
            for t2 in terms2:
                sim = self.compute_similarity(t1, t2, term_method)
                max_sim = max(max_sim, sim)
        return max_sim

    def _set_similarity_avg(
        self,
        terms1: List[str],
        terms2: List[str],
        term_method: str,
    ) -> float:
        """Average of all pairwise similarities"""
        total = 0.0
        count = 0
        for t1 in terms1:
            for t2 in terms2:
                total += self.compute_similarity(t1, t2, term_method)
                count += 1
        return total / count if count > 0 else 0.0

    # =========================================================================
    # Search
    # =========================================================================
    def search(
        self,
        query: str,
        max_results: int = 10,
        include_synonyms: bool = True,
        include_obsolete: bool = False,
    ) -> List[Tuple[str, str, float]]:
        """
        搜尋術語

        Args:
            query: 搜尋字串
            max_results: 最大結果數
            include_synonyms: 是否搜尋同義詞
            include_obsolete: 是否包含廢棄術語

        Returns:
            List of (term_id, term_name, score)
        """
        query_lower = query.lower().strip()
        results = []

        for term_id, term in self._terms.items():
            if term.is_obsolete and not include_obsolete:
                continue

            # Check exact match
            if term.name.lower() == query_lower:
                results.append((term_id, term.name, 1.0))
                continue

            # Check prefix match
            if term.name.lower().startswith(query_lower):
                score = len(query_lower) / len(term.name)
                results.append((term_id, term.name, score * 0.9))
                continue

            # Check contains match
            if query_lower in term.name.lower():
                score = len(query_lower) / len(term.name)
                results.append((term_id, term.name, score * 0.7))
                continue

            # Check synonyms
            if include_synonyms:
                for syn_text, syn_type in term.synonyms:
                    if query_lower in syn_text.lower():
                        score = len(query_lower) / len(syn_text)
                        results.append((term_id, term.name, score * 0.6))
                        break

        # Sort by score descending
        results.sort(key=lambda x: x[2], reverse=True)

        return results[:max_results]

    # =========================================================================
    # Export
    # =========================================================================
    def get_all_terms(self, include_obsolete: bool = False) -> List[str]:
        """獲取所有術語 ID"""
        return [
            term_id for term_id, term in self._terms.items()
            if include_obsolete or not term.is_obsolete
        ]

    def to_edges(self) -> List[Tuple[str, str, str]]:
        """
        導出為邊列表 (用於構建知識圖譜)

        Returns:
            List of (child_id, parent_id, edge_type)
        """
        edges = []
        for child_id, parents in self._parents.items():
            for parent_id in parents:
                edges.append((child_id, parent_id, "is_a"))
        return edges

    def __repr__(self) -> str:
        return f"Ontology({self.name}, version={self.version}, terms={self.num_terms})"
