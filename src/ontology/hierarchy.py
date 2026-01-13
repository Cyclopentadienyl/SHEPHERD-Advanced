"""
SHEPHERD-Advanced Ontology Hierarchy Operations
================================================
本體層次結構處理，支援 pronto 後端和 legacy 模式

版本: 1.1.0
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict, deque
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, TYPE_CHECKING

from src.core.types import DataSource

if TYPE_CHECKING:
    import pronto
    from src.ontology.loader import OBOHeader, OBOTerm

logger = logging.getLogger(__name__)


# =============================================================================
# Ontology Class
# =============================================================================
class Ontology:
    """
    本體操作類

    支援兩種模式:
    1. pronto 後端模式 (生產環境，推薦)
    2. legacy 模式 (測試 fixtures)

    提供:
    - 層次結構遍歷 (祖先、後代、父節點、子節點)
    - 語義相似度計算 (Resnik, Lin, Jiang-Conrath)
    - Information Content (IC) 計算
    - 術語搜尋
    """

    def __init__(
        self,
        backend: Union['pronto.Ontology', 'OBOHeader'],
        terms: Optional[Dict[str, 'OBOTerm']] = None,
        source_path: Optional[Path] = None,
    ):
        """
        Args:
            backend: pronto.Ontology 實例或 OBOHeader (legacy 模式)
            terms: legacy 模式的術語字典
            source_path: 來源檔案路徑
        """
        self._source_path = source_path
        self._source: Optional[DataSource] = None
        self._ontology_name: Optional[str] = None

        # IC values (computed lazily)
        self._ic_values: Optional[Dict[str, float]] = None
        self._term_descendants_count: Optional[Dict[str, int]] = None

        # Detect mode
        try:
            import pronto as pronto_module
            if isinstance(backend, pronto_module.Ontology):
                self._mode = "pronto"
                self._pronto_ont: pronto.Ontology = backend
                self._header = None
                self._terms = None
                self._init_pronto_indexes()
            else:
                self._mode = "legacy"
                self._pronto_ont = None
                self._header: OBOHeader = backend
                self._terms: Dict[str, OBOTerm] = terms or {}
                self._init_legacy_indexes()
        except ImportError:
            # pronto not installed, use legacy mode
            self._mode = "legacy"
            self._pronto_ont = None
            self._header = backend
            self._terms = terms or {}
            self._init_legacy_indexes()

        logger.info(
            f"Ontology initialized ({self._mode} mode): {self.name} v{self.version} "
            f"({self.num_terms} terms)"
        )

    def _init_pronto_indexes(self) -> None:
        """初始化 pronto 模式的索引"""
        # pronto 已經內建索引，不需要額外建立
        pass

    def _init_legacy_indexes(self) -> None:
        """初始化 legacy 模式的索引"""
        self._children: Dict[str, Set[str]] = defaultdict(set)
        self._parents: Dict[str, Set[str]] = defaultdict(set)

        for term_id, term in self._terms.items():
            if term.is_obsolete:
                continue

            for parent_id in term.is_a:
                self._parents[term_id].add(parent_id)
                self._children[parent_id].add(term_id)

    # =========================================================================
    # Properties
    # =========================================================================
    @property
    def name(self) -> str:
        """本體名稱"""
        if self._mode == "pronto":
            meta = self._pronto_ont.metadata
            return meta.ontology or self._ontology_name or "Unknown"
        else:
            return self._header.ontology or "Unknown"

    @property
    def version(self) -> str:
        """本體版本"""
        if self._mode == "pronto":
            meta = self._pronto_ont.metadata
            return meta.data_version or meta.format_version or "Unknown"
        else:
            return self._header.data_version or self._header.format_version or "Unknown"

    @property
    def num_terms(self) -> int:
        """非廢棄術語數量"""
        if self._mode == "pronto":
            return sum(1 for t in self._pronto_ont.terms() if not t.obsolete)
        else:
            return sum(1 for t in self._terms.values() if not t.is_obsolete)

    @property
    def root_terms(self) -> Set[str]:
        """獲取根節點 (沒有父節點的術語)"""
        if self._mode == "pronto":
            roots = set()
            for term in self._pronto_ont.terms():
                if not term.obsolete and not list(term.superclasses(distance=1, with_self=False)):
                    roots.add(term.id)
            return roots
        else:
            roots = set()
            for term_id, term in self._terms.items():
                if not term.is_obsolete and not term.is_a:
                    roots.add(term_id)
            return roots

    # =========================================================================
    # Term Access
    # =========================================================================
    def get_term(self, term_id: str) -> Optional[Dict[str, Any]]:
        """獲取單個術語"""
        if self._mode == "pronto":
            try:
                term = self._pronto_ont.get(term_id)
                if term is None:
                    return None
                return {
                    "id": term.id,
                    "name": term.name,
                    "namespace": str(term.namespace) if term.namespace else None,
                    "definition": str(term.definition) if term.definition else None,
                    "synonyms": [str(s.description) for s in term.synonyms],
                    "xrefs": [str(x) for x in term.xrefs],
                    "is_obsolete": term.obsolete,
                    "parents": [str(p.id) for p in term.superclasses(distance=1, with_self=False)],
                }
            except KeyError:
                return None
        else:
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
        if self._mode == "pronto":
            try:
                term = self._pronto_ont.get(term_id)
                return term.name if term else None
            except KeyError:
                return None
        else:
            term = self._terms.get(term_id)
            return term.name if term else None

    def has_term(self, term_id: str) -> bool:
        """檢查術語是否存在"""
        if self._mode == "pronto":
            try:
                return self._pronto_ont.get(term_id) is not None
            except KeyError:
                return False
        else:
            return term_id in self._terms

    def is_obsolete(self, term_id: str) -> bool:
        """檢查術語是否已廢棄"""
        if self._mode == "pronto":
            try:
                term = self._pronto_ont.get(term_id)
                return term.obsolete if term else True
            except KeyError:
                return True
        else:
            term = self._terms.get(term_id)
            return term.is_obsolete if term else True

    def get_replacement(self, term_id: str) -> Optional[str]:
        """獲取廢棄術語的替代術語"""
        if self._mode == "pronto":
            try:
                term = self._pronto_ont.get(term_id)
                if term and term.obsolete and term.replaced_by:
                    # replaced_by 是一個 frozenset
                    replacements = list(term.replaced_by)
                    return str(replacements[0].id) if replacements else None
            except (KeyError, AttributeError):
                pass
            return None
        else:
            term = self._terms.get(term_id)
            if term and term.is_obsolete:
                return term.replaced_by
            return None

    # =========================================================================
    # Hierarchy Traversal
    # =========================================================================
    def get_parents(self, term_id: str) -> Set[str]:
        """獲取直接父節點"""
        if self._mode == "pronto":
            try:
                term = self._pronto_ont.get(term_id)
                if term is None:
                    return set()
                return {str(p.id) for p in term.superclasses(distance=1, with_self=False)}
            except KeyError:
                return set()
        else:
            return self._parents.get(term_id, set()).copy()

    def get_children(self, term_id: str) -> Set[str]:
        """獲取直接子節點"""
        if self._mode == "pronto":
            try:
                term = self._pronto_ont.get(term_id)
                if term is None:
                    return set()
                return {str(c.id) for c in term.subclasses(distance=1, with_self=False)}
            except KeyError:
                return set()
        else:
            return self._children.get(term_id, set()).copy()

    def get_ancestors(self, term_id: str, include_self: bool = False) -> Set[str]:
        """獲取所有祖先節點"""
        if self._mode == "pronto":
            try:
                term = self._pronto_ont.get(term_id)
                if term is None:
                    return set()
                ancestors = {str(a.id) for a in term.superclasses(with_self=include_self)}
                if not include_self:
                    ancestors.discard(term_id)
                return ancestors
            except KeyError:
                return set()
        else:
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

    def get_descendants(self, term_id: str, include_self: bool = False) -> Set[str]:
        """獲取所有後代節點"""
        if self._mode == "pronto":
            try:
                term = self._pronto_ont.get(term_id)
                if term is None:
                    return set()
                descendants = {str(d.id) for d in term.subclasses(with_self=include_self)}
                if not include_self:
                    descendants.discard(term_id)
                return descendants
            except KeyError:
                return set()
        else:
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
        """獲取術語的深度"""
        if not self.has_term(term_id):
            return -1

        parents = self.get_parents(term_id)
        if not parents:
            return 0

        # BFS to find shortest path to root
        visited = {term_id}
        queue = deque([(term_id, 0)])

        while queue:
            current, depth = queue.popleft()
            current_parents = self.get_parents(current)

            if not current_parents:
                return depth

            for parent in current_parents:
                if parent not in visited:
                    visited.add(parent)
                    queue.append((parent, depth + 1))

        return -1

    # =========================================================================
    # LCA
    # =========================================================================
    def get_common_ancestors(self, term1: str, term2: str) -> Set[str]:
        """獲取兩個術語的共同祖先"""
        ancestors1 = self.get_ancestors(term1, include_self=True)
        ancestors2 = self.get_ancestors(term2, include_self=True)
        return ancestors1 & ancestors2

    def get_lowest_common_ancestors(self, term1: str, term2: str) -> Set[str]:
        """獲取最低共同祖先"""
        common = self.get_common_ancestors(term1, term2)
        if not common:
            return set()

        lcas = set()
        for term in common:
            descendants = self.get_descendants(term)
            if not (descendants & common):
                lcas.add(term)

        return lcas

    # =========================================================================
    # Information Content
    # =========================================================================
    def _compute_ic_values(self) -> None:
        """計算所有術語的 Information Content"""
        if self._ic_values is not None:
            return

        logger.info("Computing Information Content values...")

        self._ic_values = {}
        total_terms = self.num_terms

        # Get all term IDs
        if self._mode == "pronto":
            all_terms = [t.id for t in self._pronto_ont.terms() if not t.obsolete]
        else:
            all_terms = [t_id for t_id, t in self._terms.items() if not t.is_obsolete]

        # Compute descendants count and IC for each term
        for term_id in all_terms:
            desc_count = len(self.get_descendants(term_id))
            p_term = (desc_count + 1) / total_terms
            self._ic_values[term_id] = -math.log(p_term) if p_term > 0 else 0.0

        logger.info(f"Computed IC for {len(self._ic_values)} terms")

    def get_information_content(self, term_id: str) -> float:
        """獲取術語的 Information Content"""
        self._compute_ic_values()
        return self._ic_values.get(term_id, 0.0)

    # =========================================================================
    # Semantic Similarity
    # =========================================================================
    def compute_similarity(self, term1: str, term2: str, method: str = "resnik") -> float:
        """計算兩個術語的語義相似度"""
        if term1 == term2:
            if method == "resnik":
                return self.get_information_content(term1)
            return 1.0

        if not self.has_term(term1) or not self.has_term(term2):
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
        lcas = self.get_lowest_common_ancestors(term1, term2)
        if not lcas:
            return 0.0
        return max(self.get_information_content(lca) for lca in lcas)

    def _similarity_lin(self, term1: str, term2: str) -> float:
        ic1 = self.get_information_content(term1)
        ic2 = self.get_information_content(term2)
        if ic1 + ic2 == 0:
            return 0.0
        ic_lca = self._similarity_resnik(term1, term2)
        return (2 * ic_lca) / (ic1 + ic2)

    def _similarity_jiang(self, term1: str, term2: str) -> float:
        ic1 = self.get_information_content(term1)
        ic2 = self.get_information_content(term2)
        ic_lca = self._similarity_resnik(term1, term2)
        distance = ic1 + ic2 - 2 * ic_lca
        return 1.0 / (1.0 + distance)

    def _similarity_jaccard(self, term1: str, term2: str) -> float:
        ancestors1 = self.get_ancestors(term1, include_self=True)
        ancestors2 = self.get_ancestors(term2, include_self=True)
        if not ancestors1 and not ancestors2:
            return 0.0
        intersection = len(ancestors1 & ancestors2)
        union = len(ancestors1 | ancestors2)
        return intersection / union if union > 0 else 0.0

    # =========================================================================
    # Set Similarity
    # =========================================================================
    def compute_set_similarity(
        self,
        terms1: List[str],
        terms2: List[str],
        method: str = "bma",
        term_method: str = "resnik",
    ) -> float:
        """計算兩個術語集合的相似度"""
        if not terms1 or not terms2:
            return 0.0

        terms1 = [t for t in terms1 if self.has_term(t)]
        terms2 = [t for t in terms2 if self.has_term(t)]

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

    def _set_similarity_bma(self, terms1: List[str], terms2: List[str], term_method: str) -> float:
        bm1 = [max(self.compute_similarity(t1, t2, term_method) for t2 in terms2) for t1 in terms1]
        bm2 = [max(self.compute_similarity(t1, t2, term_method) for t1 in terms1) for t2 in terms2]
        return (sum(bm1) / len(bm1) + sum(bm2) / len(bm2)) / 2

    def _set_similarity_max(self, terms1: List[str], terms2: List[str], term_method: str) -> float:
        return max(
            self.compute_similarity(t1, t2, term_method)
            for t1 in terms1 for t2 in terms2
        )

    def _set_similarity_avg(self, terms1: List[str], terms2: List[str], term_method: str) -> float:
        total = sum(
            self.compute_similarity(t1, t2, term_method)
            for t1 in terms1 for t2 in terms2
        )
        return total / (len(terms1) * len(terms2))

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
        """搜尋術語"""
        query_lower = query.lower().strip()
        results = []

        if self._mode == "pronto":
            for term in self._pronto_ont.terms():
                if term.obsolete and not include_obsolete:
                    continue

                name = term.name or ""
                name_lower = name.lower()

                # Exact match
                if name_lower == query_lower:
                    results.append((term.id, name, 1.0))
                    continue

                # Prefix match
                if name_lower.startswith(query_lower):
                    score = len(query_lower) / len(name) if name else 0
                    results.append((term.id, name, score * 0.9))
                    continue

                # Contains match
                if query_lower in name_lower:
                    score = len(query_lower) / len(name) if name else 0
                    results.append((term.id, name, score * 0.7))
                    continue

                # Synonym match
                if include_synonyms:
                    for syn in term.synonyms:
                        syn_text = str(syn.description).lower()
                        if query_lower in syn_text:
                            score = len(query_lower) / len(syn_text) if syn_text else 0
                            results.append((term.id, name, score * 0.6))
                            break
        else:
            for term_id, term in self._terms.items():
                if term.is_obsolete and not include_obsolete:
                    continue

                name_lower = term.name.lower()

                if name_lower == query_lower:
                    results.append((term_id, term.name, 1.0))
                    continue

                if name_lower.startswith(query_lower):
                    score = len(query_lower) / len(term.name)
                    results.append((term_id, term.name, score * 0.9))
                    continue

                if query_lower in name_lower:
                    score = len(query_lower) / len(term.name)
                    results.append((term_id, term.name, score * 0.7))
                    continue

                if include_synonyms:
                    for syn_text, _ in term.synonyms:
                        if query_lower in syn_text.lower():
                            score = len(query_lower) / len(syn_text)
                            results.append((term_id, term.name, score * 0.6))
                            break

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:max_results]

    # =========================================================================
    # Export
    # =========================================================================
    def get_all_terms(self, include_obsolete: bool = False) -> List[str]:
        """獲取所有術語 ID"""
        if self._mode == "pronto":
            return [
                t.id for t in self._pronto_ont.terms()
                if include_obsolete or not t.obsolete
            ]
        else:
            return [
                term_id for term_id, term in self._terms.items()
                if include_obsolete or not term.is_obsolete
            ]

    def to_edges(self) -> List[Tuple[str, str, str]]:
        """導出為邊列表"""
        edges = []

        if self._mode == "pronto":
            for term in self._pronto_ont.terms():
                if term.obsolete:
                    continue
                for parent in term.superclasses(distance=1, with_self=False):
                    edges.append((term.id, parent.id, "is_a"))
        else:
            for child_id, parents in self._parents.items():
                for parent_id in parents:
                    edges.append((child_id, parent_id, "is_a"))

        return edges

    def __repr__(self) -> str:
        return f"Ontology({self.name}, version={self.version}, terms={self.num_terms}, mode={self._mode})"
