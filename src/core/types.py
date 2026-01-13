"""
SHEPHERD-Advanced Core Types
============================
統一的資料類型定義，所有模組共享

版本: 1.0.0
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    TypeVar,
    Generic,
    Sequence,
)
from datetime import datetime
import numpy as np
from numpy.typing import NDArray


# =============================================================================
# Type Variables
# =============================================================================
T = TypeVar("T")
EmbeddingType = NDArray[np.float32]  # (dim,) or (batch, dim)
TensorType = TypeVar("TensorType")  # For torch.Tensor compatibility


# =============================================================================
# Enums - Node Types
# =============================================================================
class NodeType(str, Enum):
    """
    知識圖譜節點類型

    設計原則:
    - 使用 str 繼承以支援 JSON 序列化
    - 預留擴展空間
    """
    # === Core Medical Entities ===
    GENE = "gene"
    DISEASE = "disease"
    PHENOTYPE = "phenotype"  # HPO terms
    PATHWAY = "pathway"      # GO/Reactome pathways
    DRUG = "drug"
    PROTEIN = "protein"
    VARIANT = "variant"      # Genetic variants

    # === Ontology Nodes ===
    HPO_TERM = "hpo_term"
    MONDO_TERM = "mondo_term"
    GO_TERM = "go_term"

    # === Cross-Species (同源基因比對) ===
    ORTHOLOG_GROUP = "ortholog_group"  # 同源基因群組
    MOUSE_GENE = "mouse_gene"          # 小鼠基因 (MGI)
    MOUSE_PHENOTYPE = "mouse_phenotype"  # 小鼠表型 (MP)
    ZEBRAFISH_GENE = "zebrafish_gene"  # 斑馬魚基因 (ZFIN)

    # === Literature (PubMed) ===
    PUBLICATION = "publication"  # PubMed articles
    AUTHOR = "author"            # Publication authors (optional)

    # === Clinical ===
    PATIENT = "patient"          # Anonymized patient node
    CLINICAL_FINDING = "clinical_finding"


class EdgeType(str, Enum):
    """
    知識圖譜邊類型

    命名慣例: SOURCE_RELATION_TARGET
    """
    # === Gene-Disease Associations ===
    GENE_ASSOCIATED_WITH_DISEASE = "gene_associated_with_disease"
    GENE_CAUSES_DISEASE = "gene_causes_disease"

    # === Phenotype Relations ===
    PHENOTYPE_OF_DISEASE = "phenotype_of_disease"
    GENE_HAS_PHENOTYPE = "gene_has_phenotype"

    # === Ontology Hierarchy ===
    IS_A = "is_a"                    # Ontology subsumption
    PART_OF = "part_of"              # Parthood relation
    HAS_SUBCLASS = "has_subclass"    # Inverse of IS_A

    # === Cross-Species (同源基因比對) ===
    ORTHOLOG_OF = "ortholog_of"              # Human gene <-> Mouse gene
    HUMAN_MOUSE_ORTHOLOG = "human_mouse_ortholog"
    HUMAN_ZEBRAFISH_ORTHOLOG = "human_zebrafish_ortholog"
    ORTHOLOG_IN_GROUP = "ortholog_in_group"  # Gene -> OrthologGroup
    MOUSE_GENE_HAS_PHENOTYPE = "mouse_gene_has_phenotype"

    # === Pathway Relations ===
    GENE_IN_PATHWAY = "gene_in_pathway"
    PATHWAY_INVOLVES_GENE = "pathway_involves_gene"
    PROTEIN_INTERACTS_WITH = "protein_interacts_with"

    # === Drug Relations ===
    DRUG_TARGETS_GENE = "drug_targets_gene"
    DRUG_TREATS_DISEASE = "drug_treats_disease"

    # === Literature (PubMed) ===
    PUBLICATION_MENTIONS_GENE = "publication_mentions_gene"
    PUBLICATION_MENTIONS_DISEASE = "publication_mentions_disease"
    PUBLICATION_SUPPORTS_ASSOCIATION = "publication_supports_association"

    # === Clinical ===
    PATIENT_HAS_PHENOTYPE = "patient_has_phenotype"
    PATIENT_DIAGNOSED_WITH = "patient_diagnosed_with"


class Species(str, Enum):
    """支援的物種 (用於同源基因比對)"""
    HUMAN = "human"           # Homo sapiens (NCBI: 9606)
    MOUSE = "mouse"           # Mus musculus (NCBI: 10090)
    ZEBRAFISH = "zebrafish"   # Danio rerio (NCBI: 7955)
    RAT = "rat"               # Rattus norvegicus (NCBI: 10116)


class EvidenceLevel(str, Enum):
    """證據等級 (用於文獻可信度評估)"""
    META_ANALYSIS = "meta_analysis"      # 最高: 統合分析
    SYSTEMATIC_REVIEW = "systematic_review"
    RCT = "rct"                          # 隨機對照試驗
    COHORT_STUDY = "cohort_study"
    CASE_CONTROL = "case_control"
    CASE_SERIES = "case_series"
    CASE_REPORT = "case_report"          # 最低: 病例報告
    EXPERT_OPINION = "expert_opinion"
    IN_VITRO = "in_vitro"
    IN_SILICO = "in_silico"
    UNKNOWN = "unknown"


class DataSource(str, Enum):
    """資料來源"""
    # === Ontologies ===
    HPO = "hpo"
    MONDO = "mondo"
    GO = "go"
    REACTOME = "reactome"

    # === Gene-Disease Databases ===
    DISGENET = "disgenet"
    CLINVAR = "clinvar"
    OMIM = "omim"
    ORPHANET = "orphanet"

    # === Literature ===
    PUBMED = "pubmed"
    PUBTATOR = "pubtator"

    # === Cross-Species ===
    MGI = "mgi"              # Mouse Genome Informatics
    ZFIN = "zfin"            # Zebrafish Information Network
    ENSEMBL = "ensembl"
    ORTHODB = "orthodb"
    PANTHER = "panther"

    # === Drug ===
    DRUGBANK = "drugbank"
    CHEMBL = "chembl"

    # === Clinical ===
    FHIR = "fhir"
    HISS = "hiss"


# =============================================================================
# Core Data Classes
# =============================================================================
@dataclass(frozen=True)
class NodeID:
    """
    節點唯一識別碼

    格式: {source}:{id}
    例如: hpo:HP:0001250, gene:BRCA1, pubmed:12345678
    """
    source: DataSource
    local_id: str

    def __str__(self) -> str:
        return f"{self.source.value}:{self.local_id}"

    @classmethod
    def from_string(cls, s: str) -> "NodeID":
        """從字串解析 NodeID"""
        parts = s.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid NodeID format: {s}")
        return cls(DataSource(parts[0]), parts[1])

    def __hash__(self) -> int:
        return hash((self.source, self.local_id))


@dataclass
class Node:
    """
    知識圖譜節點
    """
    id: NodeID
    node_type: NodeType
    name: str

    # Optional metadata
    aliases: List[str] = field(default_factory=list)
    description: Optional[str] = None
    species: Optional[Species] = None  # For cross-species nodes

    # Embedding (populated after encoding)
    embedding: Optional[EmbeddingType] = None

    # Provenance
    data_sources: Set[DataSource] = field(default_factory=set)
    last_updated: Optional[datetime] = None

    # Extra attributes (flexible)
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    """
    知識圖譜邊
    """
    source_id: NodeID
    target_id: NodeID
    edge_type: EdgeType

    # Edge attributes
    weight: float = 1.0
    confidence: float = 1.0

    # Evidence
    evidence_sources: List["EvidenceSource"] = field(default_factory=list)
    publication_count: int = 0  # For literature-backed edges

    # Provenance
    data_source: Optional[DataSource] = None
    last_updated: Optional[datetime] = None

    # Extra attributes
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceSource:
    """
    證據來源 (用於邊的可解釋性)
    """
    source_type: DataSource
    source_id: str  # e.g., PMID, ClinVar ID
    evidence_level: EvidenceLevel = EvidenceLevel.UNKNOWN
    confidence_score: float = 1.0

    # For PubMed sources
    publication_year: Optional[int] = None
    journal_impact_factor: Optional[float] = None
    citation_count: Optional[int] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Ortholog (同源基因) Types
# =============================================================================
@dataclass
class OrthologMapping:
    """
    同源基因映射
    用於跨物種基因比對
    """
    human_gene_id: NodeID
    ortholog_gene_id: NodeID
    ortholog_species: Species

    # Orthology confidence
    ortholog_type: str = "one2one"  # one2one, one2many, many2many
    confidence_score: float = 1.0

    # Source database
    source: DataSource = DataSource.ENSEMBL

    # Associated phenotypes in model organism
    model_phenotypes: List[NodeID] = field(default_factory=list)

    # Evidence
    evidence_sources: List[EvidenceSource] = field(default_factory=list)


@dataclass
class OrthologGroup:
    """
    同源基因群組
    包含跨多物種的同源基因
    """
    group_id: str  # e.g., OrthoGroup:12345
    group_name: Optional[str] = None

    # Member genes by species
    human_genes: List[NodeID] = field(default_factory=list)
    mouse_genes: List[NodeID] = field(default_factory=list)
    zebrafish_genes: List[NodeID] = field(default_factory=list)

    # Group-level annotations
    go_terms: List[NodeID] = field(default_factory=list)

    # Source
    source: DataSource = DataSource.ORTHODB


# =============================================================================
# Literature (PubMed) Types
# =============================================================================
@dataclass
class Publication:
    """
    PubMed 文獻記錄
    """
    pmid: str
    title: str
    abstract: Optional[str] = None

    # Journal info
    journal: Optional[str] = None
    publication_year: Optional[int] = None
    impact_factor: Optional[float] = None

    # Citation metrics
    citation_count: int = 0

    # Evidence level
    evidence_level: EvidenceLevel = EvidenceLevel.UNKNOWN

    # Extracted entities (from Pubtator)
    mentioned_genes: List[str] = field(default_factory=list)
    mentioned_diseases: List[str] = field(default_factory=list)
    mentioned_phenotypes: List[str] = field(default_factory=list)

    # Computed credibility score
    credibility_score: Optional[float] = None


@dataclass
class LiteratureEvidence:
    """
    文獻證據 (用於知識圖譜邊的文獻支持)
    """
    publications: List[Publication]

    # Aggregated metrics
    total_publications: int = 0
    avg_credibility: float = 0.0
    max_evidence_level: EvidenceLevel = EvidenceLevel.UNKNOWN

    # Confidence based on literature
    literature_confidence: float = 0.0


# =============================================================================
# Inference Types
# =============================================================================
@dataclass
class PatientPhenotypes:
    """
    患者表型輸入
    """
    patient_id: str

    # HPO phenotypes
    phenotypes: List[str]  # HPO IDs: HP:XXXXXXX
    phenotype_confidences: Optional[List[float]] = None

    # Optional: genetic data
    candidate_genes: Optional[List[str]] = None
    variants: Optional[List[str]] = None

    # Optional: demographics
    age: Optional[int] = None
    sex: Optional[str] = None


@dataclass
class DiagnosisCandidate:
    """
    診斷候選結果
    """
    rank: int

    # Disease info
    disease_id: NodeID
    disease_name: str

    # Scores
    confidence_score: float
    gnn_score: float
    reasoning_score: float

    # Evidence
    supporting_genes: List[str] = field(default_factory=list)
    reasoning_paths: List[List[NodeID]] = field(default_factory=list)
    evidence_sources: List[EvidenceSource] = field(default_factory=list)

    # Ortholog evidence (同源基因證據)
    ortholog_evidence: Optional[List[OrthologMapping]] = None

    # Explanation
    explanation: Optional[str] = None


@dataclass
class InferenceResult:
    """
    完整推理結果
    """
    patient_id: str
    timestamp: datetime

    # Top candidates
    candidates: List[DiagnosisCandidate]

    # Overall explanation
    summary_explanation: Optional[str] = None

    # Metadata
    model_version: str = ""
    kg_version: str = ""
    inference_time_ms: float = 0.0

    # Warnings
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# Configuration Types
# =============================================================================
@dataclass
class ModelConfig:
    """模型配置"""
    hidden_dim: int = 512
    num_layers: int = 4
    num_attention_heads: int = 8
    dropout: float = 0.1

    # GNN specific
    gnn_type: str = "hetero_gat"  # hetero_gat, hetero_sage, hetero_rgcn
    aggregation: str = "mean"

    # Attention backend
    attention_backend: str = "auto"  # auto, flash_attn, xformers, cudnn_sdpa, torch_sdpa


@dataclass
class TrainingConfig:
    """訓練配置"""
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100

    # Optimizer
    optimizer: str = "adamw"
    weight_decay: float = 0.01

    # Scheduler
    scheduler: str = "cosine"
    warmup_steps: int = 1000

    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4


@dataclass
class DataConfig:
    """資料配置"""
    # Ontology versions
    hpo_version: str = "latest"
    mondo_version: str = "latest"
    go_version: str = "latest"

    # Data sources to include
    enabled_sources: List[DataSource] = field(default_factory=lambda: [
        DataSource.HPO,
        DataSource.MONDO,
        DataSource.DISGENET,
        DataSource.CLINVAR,
    ])

    # Cross-species settings (同源基因)
    enable_orthologs: bool = True
    ortholog_species: List[Species] = field(default_factory=lambda: [
        Species.MOUSE,
    ])

    # Literature settings (PubMed)
    enable_literature: bool = True
    min_publication_confidence: float = 0.5
    min_impact_factor: float = 2.0
    min_citations: int = 10


# =============================================================================
# Result Wrapper
# =============================================================================
@dataclass
class Result(Generic[T]):
    """
    通用結果包裝器
    用於錯誤處理和狀態傳遞
    """
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, data: T, **metadata) -> "Result[T]":
        return cls(success=True, data=data, metadata=metadata)

    @classmethod
    def fail(cls, error: str, **metadata) -> "Result[T]":
        return cls(success=False, error=error, metadata=metadata)
