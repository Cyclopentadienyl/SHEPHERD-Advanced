"""
SHEPHERD-Advanced Core Module
==============================
核心模組，包含所有 Protocol 定義和共享類型

所有其他模組都應依賴此模組以獲取接口契約

使用方式:
    from src.core import NodeType, EdgeType, Node, Edge
    from src.core import KnowledgeGraphProtocol, InferencePipelineProtocol
    from src.core import get_kg_schema
"""

# Core Types
from src.core.types import (
    # Enums
    NodeType,
    EdgeType,
    Species,
    EvidenceLevel,
    DataSource,
    # Data Classes
    NodeID,
    Node,
    Edge,
    EvidenceSource,
    OrthologMapping,
    OrthologGroup,
    Publication,
    LiteratureEvidence,
    PatientPhenotypes,
    DiagnosisCandidate,
    InferenceResult,
    # Config Classes
    ModelConfig,
    TrainingConfig,
    DataConfig,
    # Utilities
    Result,
    EmbeddingType,
)

# KG Schema
from src.core.schema import (
    KnowledgeGraphSchema,
    NodeTypeSchema,
    EdgeTypeSchema,
    MetapathSchema,
    get_kg_schema,
)

# Protocols
from src.core.protocols import (
    # Ontology
    OntologyLoaderProtocol,
    OntologyProtocol,
    OntologyConstraintProtocol,
    # Knowledge Graph
    KnowledgeGraphProtocol,
    KnowledgeGraphBuilderProtocol,
    # Data Sources
    DataSourceProtocol,
    PubMedDataSourceProtocol,
    PubtatorLocalDBProtocol,
    OrthologDataSourceProtocol,
    # Models
    NodeEncoderProtocol,
    GNNProtocol,
    AttentionBackendProtocol,
    DecoderProtocol,
    DiagnosisModelProtocol,
    # Reasoning
    PathReasonerProtocol,
    OrthologReasonerProtocol,
    ConstraintCheckerProtocol,
    ExplanationGeneratorProtocol,
    # Retrieval
    VectorIndexProtocol,
    SubgraphSamplerProtocol,
    # LLM
    LLMProtocol,
    MedicalLLMProtocol,
    # NLP
    SymptomExtractorProtocol,
    HPOMatcherProtocol,
    # Inference
    InferencePipelineProtocol,
    InputValidatorProtocol,
    OutputFormatterProtocol,
    # Training
    TrainerProtocol,
    # API
    APIServiceProtocol,
    # Medical Standards
    FHIRAdapterProtocol,
    MedicalCodeMapperProtocol,
    # Config
    ConfigLoaderProtocol,
)

__all__ = [
    # === Enums ===
    "NodeType",
    "EdgeType",
    "Species",
    "EvidenceLevel",
    "DataSource",
    # === Data Classes ===
    "NodeID",
    "Node",
    "Edge",
    "EvidenceSource",
    "OrthologMapping",
    "OrthologGroup",
    "Publication",
    "LiteratureEvidence",
    "PatientPhenotypes",
    "DiagnosisCandidate",
    "InferenceResult",
    # === Config Classes ===
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    # === Utilities ===
    "Result",
    "EmbeddingType",
    # === Schema ===
    "KnowledgeGraphSchema",
    "NodeTypeSchema",
    "EdgeTypeSchema",
    "MetapathSchema",
    "get_kg_schema",
    # === Protocols ===
    "OntologyLoaderProtocol",
    "OntologyProtocol",
    "OntologyConstraintProtocol",
    "KnowledgeGraphProtocol",
    "KnowledgeGraphBuilderProtocol",
    "DataSourceProtocol",
    "PubMedDataSourceProtocol",
    "PubtatorLocalDBProtocol",
    "OrthologDataSourceProtocol",
    "NodeEncoderProtocol",
    "GNNProtocol",
    "AttentionBackendProtocol",
    "DecoderProtocol",
    "DiagnosisModelProtocol",
    "PathReasonerProtocol",
    "OrthologReasonerProtocol",
    "ConstraintCheckerProtocol",
    "ExplanationGeneratorProtocol",
    "VectorIndexProtocol",
    "SubgraphSamplerProtocol",
    "LLMProtocol",
    "MedicalLLMProtocol",
    "SymptomExtractorProtocol",
    "HPOMatcherProtocol",
    "InferencePipelineProtocol",
    "InputValidatorProtocol",
    "OutputFormatterProtocol",
    "TrainerProtocol",
    "APIServiceProtocol",
    "FHIRAdapterProtocol",
    "MedicalCodeMapperProtocol",
    "ConfigLoaderProtocol",
]
