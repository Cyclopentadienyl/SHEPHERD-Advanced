"""
PubMed Data Source
==================
PubMed 文獻資料整合模組

支援兩種模式:
1. Offline (推薦): 使用預下載的 Pubtator 資料庫
2. Online: 使用 PubMed API (需要網路許可)

主要功能:
- 文獻檢索和獲取
- Pubtator NER 標註 (基因、疾病、化合物)
- 文獻可信度評分計算
- 知識圖譜邊的文獻證據

版本: 1.0.0
狀態: Phase 2 (Skeleton Implementation)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
from enum import Enum

from src.core.types import (
    DataSource,
    Publication,
    LiteratureEvidence,
    EvidenceLevel,
    EvidenceSource,
)
from src.core.protocols import (
    PubMedDataSourceProtocol,
    PubtatorLocalDBProtocol,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class PubMedConfig:
    """PubMed 配置"""
    mode: str = "offline"  # offline, online

    # Credibility scoring weights
    impact_factor_weight: float = 0.4
    evidence_level_weight: float = 0.3
    citation_weight: float = 0.2
    institution_weight: float = 0.1

    # Filtering thresholds
    min_impact_factor: float = 2.0
    min_citations: int = 10
    min_credibility_score: float = 0.5

    # API settings (online mode only)
    api_key: Optional[str] = None
    rate_limit_per_second: float = 3.0


# =============================================================================
# Evidence Level Mapping
# =============================================================================
EVIDENCE_LEVEL_SCORES: Dict[EvidenceLevel, float] = {
    EvidenceLevel.META_ANALYSIS: 1.0,
    EvidenceLevel.SYSTEMATIC_REVIEW: 0.95,
    EvidenceLevel.RCT: 0.85,
    EvidenceLevel.COHORT_STUDY: 0.70,
    EvidenceLevel.CASE_CONTROL: 0.60,
    EvidenceLevel.CASE_SERIES: 0.45,
    EvidenceLevel.CASE_REPORT: 0.30,
    EvidenceLevel.EXPERT_OPINION: 0.20,
    EvidenceLevel.IN_VITRO: 0.40,
    EvidenceLevel.IN_SILICO: 0.25,
    EvidenceLevel.UNKNOWN: 0.10,
}


# =============================================================================
# Pubtator Local Database
# =============================================================================
class PubtatorLocalDB(PubtatorLocalDBProtocol):
    """
    Pubtator 本地資料庫

    用於離線模式的文獻資料存取
    支援預下載的 Pubtator 3.0 資料
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path
        self._db = None
        self._loaded = False

    def load_database(self, db_path: Path) -> None:
        """
        載入預下載的 Pubtator 資料庫

        Expected format: SQLite database with tables:
        - publications(pmid, title, abstract, journal, year, ...)
        - gene_mentions(pmid, gene_id, gene_name, ...)
        - disease_mentions(pmid, disease_id, disease_name, ...)
        - chemical_mentions(pmid, chemical_id, chemical_name, ...)
        """
        self.db_path = db_path

        if not db_path.exists():
            raise FileNotFoundError(f"Pubtator database not found: {db_path}")

        # TODO: Implement actual database loading
        # import sqlite3
        # self._db = sqlite3.connect(db_path)

        logger.info(f"Loading Pubtator database from: {db_path}")
        self._loaded = True
        logger.warning("PubtatorLocalDB.load_database is a skeleton implementation")

    def query_by_gene(self, gene_id: str) -> List[Publication]:
        """按基因查詢相關文獻"""
        if not self._loaded:
            raise RuntimeError("Database not loaded. Call load_database() first.")

        # TODO: Implement actual query
        logger.warning(f"PubtatorLocalDB.query_by_gene({gene_id}) is a skeleton")
        return []

    def query_by_disease(self, disease_id: str) -> List[Publication]:
        """按疾病查詢相關文獻"""
        if not self._loaded:
            raise RuntimeError("Database not loaded. Call load_database() first.")

        # TODO: Implement actual query
        logger.warning(f"PubtatorLocalDB.query_by_disease({disease_id}) is a skeleton")
        return []

    def query_gene_disease_pair(
        self,
        gene_id: str,
        disease_id: str,
    ) -> List[Publication]:
        """查詢同時提及基因和疾病的文獻"""
        if not self._loaded:
            raise RuntimeError("Database not loaded. Call load_database() first.")

        # TODO: Implement actual query
        logger.warning(
            f"PubtatorLocalDB.query_gene_disease_pair({gene_id}, {disease_id}) is a skeleton"
        )
        return []


# =============================================================================
# PubMed Data Source
# =============================================================================
class PubMedDataSource(PubMedDataSourceProtocol):
    """
    PubMed 資料來源

    混合式文獻檢索系統，支援離線和線上模式
    """

    def __init__(
        self,
        config: Optional[PubMedConfig] = None,
    ):
        self.config = config or PubMedConfig()

        # Initialize backends based on mode
        self._offline_db: Optional[PubtatorLocalDB] = None
        self._online_api = None  # PubMed API client

        if self.config.mode == "offline":
            self._offline_db = PubtatorLocalDB()

        logger.info(f"PubMedDataSource initialized in {self.config.mode} mode")

    @property
    def source_name(self) -> DataSource:
        return DataSource.PUBMED

    @property
    def version(self) -> str:
        return "2025.01"  # Pubtator version

    def load_pubtator_db(self, db_path: Path) -> None:
        """載入 Pubtator 離線資料庫"""
        if self._offline_db is None:
            self._offline_db = PubtatorLocalDB()
        self._offline_db.load_database(db_path)

    # =========================================================================
    # Search and Fetch
    # =========================================================================
    def search(
        self,
        query: str,
        max_results: int = 1000,
    ) -> List[str]:
        """
        搜尋 PubMed (返回 PMID 列表)

        僅在 online 模式可用
        """
        if self.config.mode != "online":
            raise RuntimeError("search() is only available in online mode")

        # TODO: Implement PubMed API search
        # from Bio import Entrez
        # Entrez.email = "your@email.com"
        # handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)

        logger.warning(f"PubMedDataSource.search({query}) is a skeleton")
        return []

    def fetch_article(self, pmid: str) -> Optional[Publication]:
        """獲取文章詳情"""
        if self.config.mode == "offline" and self._offline_db:
            # Query from local database
            results = self._offline_db.query_by_gene(pmid)  # This is a workaround
            return results[0] if results else None

        # TODO: Implement online fetch
        logger.warning(f"PubMedDataSource.fetch_article({pmid}) is a skeleton")
        return None

    def batch_fetch_articles(self, pmids: List[str]) -> List[Publication]:
        """批量獲取文章"""
        # TODO: Implement batch fetch
        logger.warning(f"PubMedDataSource.batch_fetch_articles is a skeleton")
        return []

    def get_pubtator_annotations(
        self,
        pmid: str,
    ) -> Dict[str, List[str]]:
        """
        獲取 Pubtator NER 標註

        Returns:
            {"genes": [...], "diseases": [...], "chemicals": [...]}
        """
        # TODO: Implement Pubtator annotation extraction
        logger.warning(f"get_pubtator_annotations({pmid}) is a skeleton")
        return {
            "genes": [],
            "diseases": [],
            "chemicals": [],
            "species": [],
            "mutations": [],
        }

    # =========================================================================
    # Credibility Scoring
    # =========================================================================
    def compute_credibility_score(self, publication: Publication) -> float:
        """
        計算文獻可信度評分

        評分公式:
            score = (IF_score * 0.4) + (evidence_score * 0.3) +
                    (citation_score * 0.2) + (institution_score * 0.1)

        各項評分範圍: [0, 1]
        """
        config = self.config

        # 1. Impact Factor Score (0-40%)
        if publication.impact_factor is not None:
            # Normalize: IF 0-5 -> 0-0.5, IF 5-20 -> 0.5-0.9, IF 20+ -> 0.9-1.0
            if publication.impact_factor < 5:
                if_score = publication.impact_factor / 10.0
            elif publication.impact_factor < 20:
                if_score = 0.5 + (publication.impact_factor - 5) / 30.0
            else:
                if_score = min(0.9 + (publication.impact_factor - 20) / 100.0, 1.0)
        else:
            if_score = 0.0

        # 2. Evidence Level Score (0-30%)
        evidence_score = EVIDENCE_LEVEL_SCORES.get(
            publication.evidence_level,
            EVIDENCE_LEVEL_SCORES[EvidenceLevel.UNKNOWN],
        )

        # 3. Citation Score (0-20%)
        if publication.citation_count > 0:
            # Log scale: 1 citation -> ~0.0, 10 -> ~0.33, 100 -> ~0.67, 1000 -> ~1.0
            import math
            citation_score = min(math.log10(publication.citation_count + 1) / 3.0, 1.0)
        else:
            citation_score = 0.0

        # 4. Institution Score (0-10%)
        # TODO: Implement institution scoring based on author affiliations
        institution_score = 0.5  # Default middle score

        # Combine scores
        total_score = (
            if_score * config.impact_factor_weight +
            evidence_score * config.evidence_level_weight +
            citation_score * config.citation_weight +
            institution_score * config.institution_weight
        )

        return round(total_score, 4)

    def _classify_evidence_level(self, publication: Publication) -> EvidenceLevel:
        """
        根據文章標題/摘要自動分類證據等級

        使用關鍵詞匹配 + 期刊類型
        """
        if publication.title is None:
            return EvidenceLevel.UNKNOWN

        title_lower = publication.title.lower()
        abstract_lower = (publication.abstract or "").lower()
        text = title_lower + " " + abstract_lower

        # Rule-based classification
        if "meta-analysis" in text or "metaanalysis" in text:
            return EvidenceLevel.META_ANALYSIS

        if "systematic review" in text:
            return EvidenceLevel.SYSTEMATIC_REVIEW

        if "randomized" in text and ("controlled" in text or "trial" in text):
            return EvidenceLevel.RCT

        if "cohort" in text:
            return EvidenceLevel.COHORT_STUDY

        if "case-control" in text or "case control" in text:
            return EvidenceLevel.CASE_CONTROL

        if "case series" in text:
            return EvidenceLevel.CASE_SERIES

        if "case report" in text:
            return EvidenceLevel.CASE_REPORT

        if "in vitro" in text or "cell line" in text:
            return EvidenceLevel.IN_VITRO

        if "in silico" in text or "computational" in text or "bioinformatics" in text:
            return EvidenceLevel.IN_SILICO

        return EvidenceLevel.UNKNOWN

    # =========================================================================
    # Literature Evidence for KG
    # =========================================================================
    def get_gene_disease_literature(
        self,
        gene_id: str,
        disease_id: str,
        min_credibility: float = 0.5,
    ) -> LiteratureEvidence:
        """
        獲取支持特定基因-疾病關聯的文獻證據

        用於為知識圖譜邊添加文獻支持
        """
        # Query publications mentioning both gene and disease
        if self._offline_db:
            publications = self._offline_db.query_gene_disease_pair(gene_id, disease_id)
        else:
            publications = []

        # Filter by credibility
        filtered_pubs = []
        for pub in publications:
            # Compute and assign credibility score
            if pub.credibility_score is None:
                pub.credibility_score = self.compute_credibility_score(pub)

            if pub.credibility_score >= min_credibility:
                filtered_pubs.append(pub)

        # Compute aggregated metrics
        if filtered_pubs:
            avg_credibility = sum(p.credibility_score for p in filtered_pubs) / len(filtered_pubs)
            max_evidence = max(p.evidence_level for p in filtered_pubs)

            # Literature confidence: combination of count and quality
            import math
            count_factor = min(math.log10(len(filtered_pubs) + 1) / 2.0, 1.0)
            quality_factor = avg_credibility
            literature_confidence = 0.6 * count_factor + 0.4 * quality_factor
        else:
            avg_credibility = 0.0
            max_evidence = EvidenceLevel.UNKNOWN
            literature_confidence = 0.0

        return LiteratureEvidence(
            publications=filtered_pubs,
            total_publications=len(filtered_pubs),
            avg_credibility=avg_credibility,
            max_evidence_level=max_evidence,
            literature_confidence=literature_confidence,
        )

    # =========================================================================
    # KG Integration
    # =========================================================================
    def get_gene_disease_associations(
        self,
    ) -> Iterator[Tuple[str, str, float, Dict]]:
        """
        從文獻提取基因-疾病關聯

        Yields:
            (gene_id, disease_id, confidence, metadata)

        用於建構知識圖譜時添加文獻支持的邊
        """
        # TODO: Implement full extraction from Pubtator database
        logger.warning("get_gene_disease_associations is a skeleton")
        return iter([])


# =============================================================================
# Factory Function
# =============================================================================
def create_pubmed_source(
    mode: str = "offline",
    pubtator_db_path: Optional[Path] = None,
    **kwargs,
) -> PubMedDataSource:
    """
    工廠函數: 創建 PubMed 資料來源

    Args:
        mode: "offline" or "online"
        pubtator_db_path: Path to Pubtator SQLite database (offline mode)

    Returns:
        Configured PubMedDataSource instance
    """
    config = PubMedConfig(mode=mode, **kwargs)
    source = PubMedDataSource(config)

    if mode == "offline" and pubtator_db_path:
        source.load_pubtator_db(pubtator_db_path)

    return source
