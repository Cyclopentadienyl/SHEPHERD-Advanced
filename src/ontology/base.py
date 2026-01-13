"""
Ontology Base Classes
=====================
本體基礎類別實現

使用 pronto 庫處理 OBO/OWL 格式的本體檔案

支援的本體:
- HPO (Human Phenotype Ontology) - 人類表型
- MONDO (Mondo Disease Ontology) - 疾病
- GO (Gene Ontology) - 基因功能
- MP (Mammalian Phenotype Ontology) - 小鼠表型 (用於同源基因)

版本: 1.0.0
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union
from functools import lru_cache
import math

from src.core import (
    OntologyProtocol,
    DataSource,
    NodeType,
    NodeID,
    Node,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Term Data Class
# =============================================================================
@dataclass
class OntologyTerm:
    """
    本體術語
    """
    id: str                          # e.g., "HP:0001250"
    name: str                        # e.g., "Seizure"
    definition: Optional[str] = None
    synonyms: List[str] = field(default_factory=list)
    xrefs: List[str] = field(default_factory=list)  # Cross-references
    is_obsolete: bool = False

    # Relationships (populated after loading)
    parents: Set[str] = field(default_factory=set)    # IS_A parents
    children: Set[str] = field(default_factory=set)   # IS_A children

    # Additional relationships
    part_of: Set[str] = field(default_factory=set)
    has_part: Set[str] = field(default_factory=set)

    # Metadata
    namespace: Optional[str] = None  # e.g., "HP", "MONDO"
    alt_ids: List[str] = field(default_factory=list)  # Alternative IDs

    def to_node(self, node_type: NodeType, source: DataSource) -> Node:
        """轉換為知識圖譜節點"""
        return Node(
            id=NodeID(source=source, local_id=self.id),
            node_type=node_type,
            name=self.name,
            aliases=self.synonyms,
            description=self.definition,
            data_sources={source},
            attributes={
                "xrefs": self.xrefs,
                "namespace": self.namespace,
                "is_obsolete": self.is_obsolete,
            },
        )


# =============================================================================
# Base Ontology Class
# =============================================================================
class Ontology(OntologyProtocol):
    """
    本體基礎類別

    實現 OntologyProtocol，提供本體操作功能

    使用方式:
        ontology = Ontology.from_obo("/path/to/hp.obo", name="HPO")
        term = ontology.get_term("HP:0001250")
        ancestors = ontology.get_ancestors("HP:0001250")
    """

    def __init__(
        self,
        name: str,
        version: str = "unknown",
        source: DataSource = DataSource.HPO,
    ):
        self._name = name
        self._version = version
        self._source = source

        # Term storage
        self._terms: Dict[str, OntologyTerm] = {}
        self._name_to_id: Dict[str, str] = {}  # For name lookup

        # Precomputed data (lazy initialization)
        self._ancestors_cache: Dict[str, Set[str]] = {}
        self._descendants_cache: Dict[str, Set[str]] = {}
        self._information_content: Dict[str, float] = {}

        # Root terms
        self._root_terms: Set[str] = set()

        logger.info(f"Ontology '{name}' initialized (version: {version})")

    # =========================================================================
    # Properties
    # =========================================================================
    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @property
    def source(self) -> DataSource:
        return self._source

    @property
    def num_terms(self) -> int:
        return len(self._terms)

    @property
    def root_terms(self) -> Set[str]:
        return self._root_terms.copy()

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
            "definition": term.definition,
            "synonyms": term.synonyms,
            "xrefs": term.xrefs,
            "is_obsolete": term.is_obsolete,
            "parents": list(term.parents),
            "children": list(term.children),
            "namespace": term.namespace,
        }

    def get_term_object(self, term_id: str) -> Optional[OntologyTerm]:
        """獲取術語物件"""
        return self._terms.get(term_id)

    def get_term_name(self, term_id: str) -> Optional[str]:
        """獲取術語名稱"""
        term = self._terms.get(term_id)
        return term.name if term else None

    def has_term(self, term_id: str) -> bool:
        """檢查術語是否存在"""
        return term_id in self._terms

    def all_term_ids(self) -> Set[str]:
        """獲取所有術語 ID"""
        return set(self._terms.keys())

    def iter_terms(self) -> Iterator[OntologyTerm]:
        """迭代所有術語"""
        return iter(self._terms.values())

    # =========================================================================
    # Hierarchy Navigation
    # =========================================================================
    def get_parents(self, term_id: str) -> Set[str]:
        """
        獲取直接父節點 (IS_A 關係)

        Args:
            term_id: 術語 ID

        Returns:
            父節點 ID 集合
        """
        term = self._terms.get(term_id)
        if term is None:
            return set()
        return term.parents.copy()

    def get_children(self, term_id: str) -> Set[str]:
        """
        獲取直接子節點

        Args:
            term_id: 術語 ID

        Returns:
            子節點 ID 集合
        """
        term = self._terms.get(term_id)
        if term is None:
            return set()
        return term.children.copy()

    def get_ancestors(
        self,
        term_id: str,
        include_self: bool = False,
    ) -> Set[str]:
        """
        獲取所有祖先節點 (遞歸 IS_A)

        Args:
            term_id: 術語 ID
            include_self: 是否包含自身

        Returns:
            祖先節點 ID 集合
        """
        if term_id not in self._terms:
            return set()

        # Check cache
        if term_id in self._ancestors_cache:
            ancestors = self._ancestors_cache[term_id].copy()
        else:
            # Compute ancestors recursively
            ancestors = self._compute_ancestors(term_id)
            self._ancestors_cache[term_id] = ancestors
            ancestors = ancestors.copy()

        if include_self:
            ancestors.add(term_id)

        return ancestors

    def _compute_ancestors(self, term_id: str) -> Set[str]:
        """遞歸計算祖先節點"""
        ancestors = set()
        term = self._terms.get(term_id)

        if term is None:
            return ancestors

        for parent_id in term.parents:
            ancestors.add(parent_id)
            ancestors.update(self._compute_ancestors(parent_id))

        return ancestors

    def get_descendants(
        self,
        term_id: str,
        include_self: bool = False,
    ) -> Set[str]:
        """
        獲取所有後代節點

        Args:
            term_id: 術語 ID
            include_self: 是否包含自身

        Returns:
            後代節點 ID 集合
        """
        if term_id not in self._terms:
            return set()

        # Check cache
        if term_id in self._descendants_cache:
            descendants = self._descendants_cache[term_id].copy()
        else:
            # Compute descendants recursively
            descendants = self._compute_descendants(term_id)
            self._descendants_cache[term_id] = descendants
            descendants = descendants.copy()

        if include_self:
            descendants.add(term_id)

        return descendants

    def _compute_descendants(self, term_id: str) -> Set[str]:
        """遞歸計算後代節點"""
        descendants = set()
        term = self._terms.get(term_id)

        if term is None:
            return descendants

        for child_id in term.children:
            descendants.add(child_id)
            descendants.update(self._compute_descendants(child_id))

        return descendants

    def get_depth(self, term_id: str) -> int:
        """
        獲取術語深度 (到根節點的最短路徑)

        Returns:
            深度 (根節點深度為 0)，如果不存在則返回 -1
        """
        if term_id not in self._terms:
            return -1

        if term_id in self._root_terms:
            return 0

        parents = self.get_parents(term_id)
        if not parents:
            return 0

        return 1 + min(self.get_depth(p) for p in parents)

    def get_path_to_root(self, term_id: str) -> List[List[str]]:
        """
        獲取到根節點的所有路徑

        Returns:
            路徑列表，每條路徑是術語 ID 列表
        """
        if term_id not in self._terms:
            return []

        if term_id in self._root_terms or not self.get_parents(term_id):
            return [[term_id]]

        paths = []
        for parent_id in self.get_parents(term_id):
            for path in self.get_path_to_root(parent_id):
                paths.append([term_id] + path)

        return paths

    # =========================================================================
    # Lowest Common Ancestor
    # =========================================================================
    def get_common_ancestors(
        self,
        term1: str,
        term2: str,
    ) -> Set[str]:
        """獲取兩個術語的共同祖先"""
        ancestors1 = self.get_ancestors(term1, include_self=True)
        ancestors2 = self.get_ancestors(term2, include_self=True)
        return ancestors1 & ancestors2

    def get_lowest_common_ancestor(
        self,
        term1: str,
        term2: str,
    ) -> Optional[str]:
        """
        獲取最低共同祖先 (LCA)

        Returns:
            LCA 術語 ID，如果不存在則返回 None
        """
        common = self.get_common_ancestors(term1, term2)
        if not common:
            return None

        # Find the LCA with maximum depth (most specific)
        return max(common, key=lambda t: self.get_depth(t))

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
                - "resnik": Resnik similarity (IC of LCA)
                - "lin": Lin similarity
                - "jiang": Jiang-Conrath distance (converted to similarity)
                - "jaccard": Jaccard similarity of ancestors

        Returns:
            相似度分數 [0, 1] (某些方法可能超過 1)
        """
        if term1 not in self._terms or term2 not in self._terms:
            return 0.0

        if term1 == term2:
            return 1.0

        if method == "resnik":
            return self._resnik_similarity(term1, term2)
        elif method == "lin":
            return self._lin_similarity(term1, term2)
        elif method == "jiang":
            return self._jiang_similarity(term1, term2)
        elif method == "jaccard":
            return self._jaccard_similarity(term1, term2)
        else:
            raise ValueError(f"Unknown similarity method: {method}")

    def _resnik_similarity(self, term1: str, term2: str) -> float:
        """Resnik similarity: IC(LCA)"""
        lca = self.get_lowest_common_ancestor(term1, term2)
        if lca is None:
            return 0.0
        return self.get_information_content(lca)

    def _lin_similarity(self, term1: str, term2: str) -> float:
        """Lin similarity: 2*IC(LCA) / (IC(t1) + IC(t2))"""
        lca = self.get_lowest_common_ancestor(term1, term2)
        if lca is None:
            return 0.0

        ic_lca = self.get_information_content(lca)
        ic1 = self.get_information_content(term1)
        ic2 = self.get_information_content(term2)

        denominator = ic1 + ic2
        if denominator == 0:
            return 0.0

        return 2 * ic_lca / denominator

    def _jiang_similarity(self, term1: str, term2: str) -> float:
        """Jiang-Conrath similarity: 1 - distance (normalized)"""
        lca = self.get_lowest_common_ancestor(term1, term2)
        if lca is None:
            return 0.0

        ic_lca = self.get_information_content(lca)
        ic1 = self.get_information_content(term1)
        ic2 = self.get_information_content(term2)

        # Jiang-Conrath distance
        distance = ic1 + ic2 - 2 * ic_lca

        # Convert to similarity (bounded to [0, 1])
        # Using exponential decay
        return math.exp(-distance)

    def _jaccard_similarity(self, term1: str, term2: str) -> float:
        """Jaccard similarity of ancestor sets"""
        ancestors1 = self.get_ancestors(term1, include_self=True)
        ancestors2 = self.get_ancestors(term2, include_self=True)

        if not ancestors1 and not ancestors2:
            return 0.0

        intersection = len(ancestors1 & ancestors2)
        union = len(ancestors1 | ancestors2)

        return intersection / union if union > 0 else 0.0

    # =========================================================================
    # Information Content
    # =========================================================================
    def get_information_content(self, term_id: str) -> float:
        """
        獲取術語的 Information Content (IC)

        IC = -log(p(term))
        其中 p(term) = |descendants(term)| / |all_terms|

        Returns:
            IC 值 (越高表示越具體)
        """
        if term_id not in self._terms:
            return 0.0

        # Check cache
        if term_id in self._information_content:
            return self._information_content[term_id]

        # Compute IC
        descendants = self.get_descendants(term_id, include_self=True)
        total_terms = len(self._terms)

        if total_terms == 0:
            return 0.0

        probability = len(descendants) / total_terms
        ic = -math.log(probability) if probability > 0 else 0.0

        self._information_content[term_id] = ic
        return ic

    def precompute_information_content(self) -> None:
        """預計算所有術語的 IC (加速後續查詢)"""
        logger.info(f"Precomputing IC for {len(self._terms)} terms...")

        for term_id in self._terms:
            self.get_information_content(term_id)

        logger.info("IC precomputation complete")

    # =========================================================================
    # Search
    # =========================================================================
    def search(
        self,
        query: str,
        max_results: int = 10,
        include_synonyms: bool = True,
    ) -> List[Tuple[str, str, float]]:
        """
        搜尋術語

        Args:
            query: 搜尋字串
            max_results: 最大結果數
            include_synonyms: 是否搜尋同義詞

        Returns:
            List of (term_id, term_name, score)
        """
        query_lower = query.lower().strip()
        results = []

        for term in self._terms.values():
            if term.is_obsolete:
                continue

            # Exact match on ID
            if term.id.lower() == query_lower:
                results.append((term.id, term.name, 1.0))
                continue

            # Match on name
            name_lower = term.name.lower()
            if query_lower == name_lower:
                results.append((term.id, term.name, 0.95))
            elif query_lower in name_lower:
                # Partial match - score based on position
                score = 0.7 * (1 - name_lower.index(query_lower) / len(name_lower))
                results.append((term.id, term.name, score))

            # Match on synonyms
            if include_synonyms:
                for syn in term.synonyms:
                    syn_lower = syn.lower()
                    if query_lower == syn_lower:
                        results.append((term.id, term.name, 0.85))
                        break
                    elif query_lower in syn_lower:
                        score = 0.5 * (1 - syn_lower.index(query_lower) / len(syn_lower))
                        results.append((term.id, term.name, score))
                        break

        # Sort by score and deduplicate
        seen = set()
        unique_results = []
        for term_id, name, score in sorted(results, key=lambda x: -x[2]):
            if term_id not in seen:
                seen.add(term_id)
                unique_results.append((term_id, name, score))

        return unique_results[:max_results]

    # =========================================================================
    # Conversion
    # =========================================================================
    def to_nodes(self, node_type: NodeType) -> List[Node]:
        """轉換為知識圖譜節點列表"""
        return [
            term.to_node(node_type, self._source)
            for term in self._terms.values()
            if not term.is_obsolete
        ]

    # =========================================================================
    # Internal Methods
    # =========================================================================
    def _add_term(self, term: OntologyTerm) -> None:
        """添加術語 (內部使用)"""
        self._terms[term.id] = term
        self._name_to_id[term.name.lower()] = term.id

        # Track root terms
        if not term.parents:
            self._root_terms.add(term.id)

    def _build_children_index(self) -> None:
        """建立子節點索引 (從父節點關係推導)"""
        for term_id, term in self._terms.items():
            for parent_id in term.parents:
                if parent_id in self._terms:
                    self._terms[parent_id].children.add(term_id)

    def _clear_caches(self) -> None:
        """清除快取"""
        self._ancestors_cache.clear()
        self._descendants_cache.clear()
        self._information_content.clear()

    def __len__(self) -> int:
        return len(self._terms)

    def __contains__(self, term_id: str) -> bool:
        return term_id in self._terms

    def __repr__(self) -> str:
        return f"Ontology(name='{self._name}', terms={len(self._terms)})"
