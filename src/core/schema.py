"""
SHEPHERD-Advanced Knowledge Graph Schema
=========================================
定義知識圖譜的完整 Schema，包括節點類型、邊類型和約束

版本: 1.0.0

重要設計決策:
1. 預留同源基因 (Ortholog) 節點和邊類型 - 支援跨物種推理
2. 預留文獻 (Publication) 節點 - 支援 PubMed 資料整合
3. 使用 Metapath 定義推理路徑模式
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, FrozenSet
from enum import Enum

from src.core.types import (
    NodeType,
    EdgeType,
    Species,
    DataSource,
    EvidenceLevel,
)


# =============================================================================
# Schema Definition
# =============================================================================
@dataclass
class NodeTypeSchema:
    """節點類型 Schema 定義"""
    node_type: NodeType
    description: str

    # Required attributes
    required_attributes: List[str] = field(default_factory=list)

    # Optional attributes
    optional_attributes: List[str] = field(default_factory=list)

    # Valid data sources for this node type
    valid_sources: Set[DataSource] = field(default_factory=set)

    # For cross-species nodes
    species_specific: bool = False
    valid_species: Set[Species] = field(default_factory=set)

    # Index settings
    indexed: bool = True
    embedding_dim: Optional[int] = None


@dataclass
class EdgeTypeSchema:
    """邊類型 Schema 定義"""
    edge_type: EdgeType
    description: str

    # Source and target node types
    source_types: Set[NodeType]
    target_types: Set[NodeType]

    # Edge properties
    directed: bool = True
    allow_self_loops: bool = False
    allow_multi_edges: bool = False

    # Weight constraints
    min_weight: float = 0.0
    max_weight: float = 1.0

    # Evidence requirements
    requires_evidence: bool = False
    min_evidence_level: Optional[EvidenceLevel] = None


@dataclass
class MetapathSchema:
    """
    Metapath 定義
    用於定義知識圖譜上的推理路徑模式

    Example:
        Gene -> Disease -> Phenotype
        Gene -> Ortholog -> MouseGene -> MousePhenotype -> Phenotype
    """
    name: str
    description: str

    # Path pattern: [(NodeType, EdgeType, NodeType), ...]
    path: List[Tuple[NodeType, EdgeType, NodeType]]

    # Usage
    for_inference: bool = True
    for_explanation: bool = True

    # Priority (higher = more important)
    priority: int = 1


# =============================================================================
# Complete Knowledge Graph Schema
# =============================================================================
class KnowledgeGraphSchema:
    """
    完整的知識圖譜 Schema

    包含:
    1. 所有節點類型定義
    2. 所有邊類型定義
    3. Metapath 定義
    4. 約束和驗證規則
    """

    def __init__(self):
        self.node_schemas: Dict[NodeType, NodeTypeSchema] = {}
        self.edge_schemas: Dict[EdgeType, EdgeTypeSchema] = {}
        self.metapaths: Dict[str, MetapathSchema] = {}

        self._define_node_schemas()
        self._define_edge_schemas()
        self._define_metapaths()

    def _define_node_schemas(self):
        """定義所有節點類型"""

        # =====================================================================
        # Core Medical Entities
        # =====================================================================
        self.node_schemas[NodeType.GENE] = NodeTypeSchema(
            node_type=NodeType.GENE,
            description="Human gene (HGNC symbol)",
            required_attributes=["symbol", "entrez_id"],
            optional_attributes=["ensembl_id", "uniprot_id", "description"],
            valid_sources={DataSource.DISGENET, DataSource.CLINVAR, DataSource.OMIM},
            species_specific=True,
            valid_species={Species.HUMAN},
            embedding_dim=512,
        )

        self.node_schemas[NodeType.DISEASE] = NodeTypeSchema(
            node_type=NodeType.DISEASE,
            description="Human disease (MONDO/OMIM)",
            required_attributes=["mondo_id", "name"],
            optional_attributes=["omim_id", "orphanet_id", "description", "prevalence"],
            valid_sources={DataSource.MONDO, DataSource.OMIM, DataSource.ORPHANET},
            embedding_dim=512,
        )

        self.node_schemas[NodeType.PHENOTYPE] = NodeTypeSchema(
            node_type=NodeType.PHENOTYPE,
            description="Human phenotype (HPO term)",
            required_attributes=["hpo_id", "name"],
            optional_attributes=["definition", "synonyms", "xrefs"],
            valid_sources={DataSource.HPO},
            embedding_dim=512,
        )

        self.node_schemas[NodeType.PATHWAY] = NodeTypeSchema(
            node_type=NodeType.PATHWAY,
            description="Biological pathway (GO/Reactome)",
            required_attributes=["pathway_id", "name"],
            optional_attributes=["go_id", "reactome_id", "description"],
            valid_sources={DataSource.GO, DataSource.REACTOME},
            embedding_dim=256,
        )

        self.node_schemas[NodeType.DRUG] = NodeTypeSchema(
            node_type=NodeType.DRUG,
            description="Drug/compound",
            required_attributes=["drugbank_id", "name"],
            optional_attributes=["chembl_id", "smiles", "mechanism"],
            valid_sources={DataSource.DRUGBANK, DataSource.CHEMBL},
            embedding_dim=256,
        )

        self.node_schemas[NodeType.VARIANT] = NodeTypeSchema(
            node_type=NodeType.VARIANT,
            description="Genetic variant",
            required_attributes=["variant_id", "gene"],
            optional_attributes=["clinvar_id", "hgvs", "pathogenicity", "frequency"],
            valid_sources={DataSource.CLINVAR},
            embedding_dim=256,
        )

        # =====================================================================
        # Cross-Species Nodes (同源基因比對)
        # =====================================================================
        self.node_schemas[NodeType.ORTHOLOG_GROUP] = NodeTypeSchema(
            node_type=NodeType.ORTHOLOG_GROUP,
            description="Ortholog gene group across species",
            required_attributes=["group_id"],
            optional_attributes=["group_name", "functional_annotation"],
            valid_sources={DataSource.ORTHODB, DataSource.PANTHER, DataSource.ENSEMBL},
            embedding_dim=512,
        )

        self.node_schemas[NodeType.MOUSE_GENE] = NodeTypeSchema(
            node_type=NodeType.MOUSE_GENE,
            description="Mouse gene (MGI)",
            required_attributes=["mgi_id", "symbol"],
            optional_attributes=["ensembl_id", "description"],
            valid_sources={DataSource.MGI, DataSource.ENSEMBL},
            species_specific=True,
            valid_species={Species.MOUSE},
            embedding_dim=512,
        )

        self.node_schemas[NodeType.MOUSE_PHENOTYPE] = NodeTypeSchema(
            node_type=NodeType.MOUSE_PHENOTYPE,
            description="Mouse phenotype (MP term)",
            required_attributes=["mp_id", "name"],
            optional_attributes=["definition", "synonyms"],
            valid_sources={DataSource.MGI},
            species_specific=True,
            valid_species={Species.MOUSE},
            embedding_dim=512,
        )

        self.node_schemas[NodeType.ZEBRAFISH_GENE] = NodeTypeSchema(
            node_type=NodeType.ZEBRAFISH_GENE,
            description="Zebrafish gene (ZFIN)",
            required_attributes=["zfin_id", "symbol"],
            optional_attributes=["ensembl_id", "description"],
            valid_sources={DataSource.ZFIN, DataSource.ENSEMBL},
            species_specific=True,
            valid_species={Species.ZEBRAFISH},
            embedding_dim=512,
        )

        # =====================================================================
        # Literature Nodes (PubMed)
        # =====================================================================
        self.node_schemas[NodeType.PUBLICATION] = NodeTypeSchema(
            node_type=NodeType.PUBLICATION,
            description="PubMed publication",
            required_attributes=["pmid", "title"],
            optional_attributes=[
                "abstract",
                "journal",
                "publication_year",
                "impact_factor",
                "citation_count",
                "evidence_level",
                "credibility_score",
            ],
            valid_sources={DataSource.PUBMED, DataSource.PUBTATOR},
            indexed=False,  # Not indexed in GNN by default
            embedding_dim=768,  # SciBERT embedding
        )

    def _define_edge_schemas(self):
        """定義所有邊類型"""

        # =====================================================================
        # Gene-Disease Associations
        # =====================================================================
        self.edge_schemas[EdgeType.GENE_ASSOCIATED_WITH_DISEASE] = EdgeTypeSchema(
            edge_type=EdgeType.GENE_ASSOCIATED_WITH_DISEASE,
            description="Gene is associated with disease (general association)",
            source_types={NodeType.GENE},
            target_types={NodeType.DISEASE},
            requires_evidence=True,
        )

        self.edge_schemas[EdgeType.GENE_CAUSES_DISEASE] = EdgeTypeSchema(
            edge_type=EdgeType.GENE_CAUSES_DISEASE,
            description="Gene mutation causes disease (causal relationship)",
            source_types={NodeType.GENE},
            target_types={NodeType.DISEASE},
            requires_evidence=True,
            min_evidence_level=EvidenceLevel.CASE_CONTROL,
        )

        # =====================================================================
        # Phenotype Relations
        # =====================================================================
        self.edge_schemas[EdgeType.PHENOTYPE_OF_DISEASE] = EdgeTypeSchema(
            edge_type=EdgeType.PHENOTYPE_OF_DISEASE,
            description="Phenotype is a manifestation of disease",
            source_types={NodeType.PHENOTYPE},
            target_types={NodeType.DISEASE},
        )

        self.edge_schemas[EdgeType.GENE_HAS_PHENOTYPE] = EdgeTypeSchema(
            edge_type=EdgeType.GENE_HAS_PHENOTYPE,
            description="Gene mutation leads to phenotype",
            source_types={NodeType.GENE},
            target_types={NodeType.PHENOTYPE},
        )

        # =====================================================================
        # Ontology Hierarchy
        # =====================================================================
        self.edge_schemas[EdgeType.IS_A] = EdgeTypeSchema(
            edge_type=EdgeType.IS_A,
            description="Ontology subsumption (subclass relation)",
            source_types={
                NodeType.PHENOTYPE,
                NodeType.DISEASE,
                NodeType.PATHWAY,
                NodeType.MOUSE_PHENOTYPE,
            },
            target_types={
                NodeType.PHENOTYPE,
                NodeType.DISEASE,
                NodeType.PATHWAY,
                NodeType.MOUSE_PHENOTYPE,
            },
            allow_self_loops=False,
        )

        self.edge_schemas[EdgeType.PART_OF] = EdgeTypeSchema(
            edge_type=EdgeType.PART_OF,
            description="Parthood relation in ontology",
            source_types={NodeType.PATHWAY, NodeType.PHENOTYPE},
            target_types={NodeType.PATHWAY, NodeType.PHENOTYPE},
        )

        # =====================================================================
        # Cross-Species Edges (同源基因比對)
        # =====================================================================
        self.edge_schemas[EdgeType.ORTHOLOG_OF] = EdgeTypeSchema(
            edge_type=EdgeType.ORTHOLOG_OF,
            description="Ortholog relationship between genes of different species",
            source_types={NodeType.GENE},
            target_types={NodeType.MOUSE_GENE, NodeType.ZEBRAFISH_GENE},
            directed=False,  # Orthology is symmetric
        )

        self.edge_schemas[EdgeType.HUMAN_MOUSE_ORTHOLOG] = EdgeTypeSchema(
            edge_type=EdgeType.HUMAN_MOUSE_ORTHOLOG,
            description="Human-Mouse ortholog pair",
            source_types={NodeType.GENE},
            target_types={NodeType.MOUSE_GENE},
            directed=False,
        )

        self.edge_schemas[EdgeType.HUMAN_ZEBRAFISH_ORTHOLOG] = EdgeTypeSchema(
            edge_type=EdgeType.HUMAN_ZEBRAFISH_ORTHOLOG,
            description="Human-Zebrafish ortholog pair",
            source_types={NodeType.GENE},
            target_types={NodeType.ZEBRAFISH_GENE},
            directed=False,
        )

        self.edge_schemas[EdgeType.ORTHOLOG_IN_GROUP] = EdgeTypeSchema(
            edge_type=EdgeType.ORTHOLOG_IN_GROUP,
            description="Gene belongs to ortholog group",
            source_types={NodeType.GENE, NodeType.MOUSE_GENE, NodeType.ZEBRAFISH_GENE},
            target_types={NodeType.ORTHOLOG_GROUP},
        )

        self.edge_schemas[EdgeType.MOUSE_GENE_HAS_PHENOTYPE] = EdgeTypeSchema(
            edge_type=EdgeType.MOUSE_GENE_HAS_PHENOTYPE,
            description="Mouse gene knockout causes phenotype",
            source_types={NodeType.MOUSE_GENE},
            target_types={NodeType.MOUSE_PHENOTYPE},
        )

        # =====================================================================
        # Pathway Relations
        # =====================================================================
        self.edge_schemas[EdgeType.GENE_IN_PATHWAY] = EdgeTypeSchema(
            edge_type=EdgeType.GENE_IN_PATHWAY,
            description="Gene participates in pathway",
            source_types={NodeType.GENE},
            target_types={NodeType.PATHWAY},
        )

        self.edge_schemas[EdgeType.PROTEIN_INTERACTS_WITH] = EdgeTypeSchema(
            edge_type=EdgeType.PROTEIN_INTERACTS_WITH,
            description="Protein-protein interaction",
            source_types={NodeType.GENE},  # Using gene as proxy for protein
            target_types={NodeType.GENE},
            directed=False,
            allow_self_loops=False,
        )

        # =====================================================================
        # Drug Relations
        # =====================================================================
        self.edge_schemas[EdgeType.DRUG_TARGETS_GENE] = EdgeTypeSchema(
            edge_type=EdgeType.DRUG_TARGETS_GENE,
            description="Drug targets gene product",
            source_types={NodeType.DRUG},
            target_types={NodeType.GENE},
        )

        self.edge_schemas[EdgeType.DRUG_TREATS_DISEASE] = EdgeTypeSchema(
            edge_type=EdgeType.DRUG_TREATS_DISEASE,
            description="Drug is indicated for disease treatment",
            source_types={NodeType.DRUG},
            target_types={NodeType.DISEASE},
        )

        # =====================================================================
        # Literature Edges (PubMed)
        # =====================================================================
        self.edge_schemas[EdgeType.PUBLICATION_MENTIONS_GENE] = EdgeTypeSchema(
            edge_type=EdgeType.PUBLICATION_MENTIONS_GENE,
            description="Publication mentions gene (from Pubtator NER)",
            source_types={NodeType.PUBLICATION},
            target_types={NodeType.GENE},
            allow_multi_edges=True,  # Multiple mentions
        )

        self.edge_schemas[EdgeType.PUBLICATION_MENTIONS_DISEASE] = EdgeTypeSchema(
            edge_type=EdgeType.PUBLICATION_MENTIONS_DISEASE,
            description="Publication mentions disease (from Pubtator NER)",
            source_types={NodeType.PUBLICATION},
            target_types={NodeType.DISEASE},
            allow_multi_edges=True,
        )

        self.edge_schemas[EdgeType.PUBLICATION_SUPPORTS_ASSOCIATION] = EdgeTypeSchema(
            edge_type=EdgeType.PUBLICATION_SUPPORTS_ASSOCIATION,
            description="Publication provides evidence for gene-disease association",
            source_types={NodeType.PUBLICATION},
            target_types={NodeType.GENE, NodeType.DISEASE},
            requires_evidence=True,
        )

    def _define_metapaths(self):
        """
        定義 Metapaths (推理路徑模式)

        這些路徑模式用於:
        1. GNN message passing
        2. Path-based reasoning
        3. Explanation generation
        """

        # =====================================================================
        # Basic Diagnostic Paths
        # =====================================================================
        self.metapaths["phenotype_to_disease"] = MetapathSchema(
            name="phenotype_to_disease",
            description="Direct phenotype to disease association",
            path=[
                (NodeType.PHENOTYPE, EdgeType.PHENOTYPE_OF_DISEASE, NodeType.DISEASE),
            ],
            priority=10,
        )

        self.metapaths["gene_to_disease_direct"] = MetapathSchema(
            name="gene_to_disease_direct",
            description="Direct gene-disease association",
            path=[
                (NodeType.GENE, EdgeType.GENE_ASSOCIATED_WITH_DISEASE, NodeType.DISEASE),
            ],
            priority=10,
        )

        self.metapaths["phenotype_gene_disease"] = MetapathSchema(
            name="phenotype_gene_disease",
            description="Phenotype -> Gene -> Disease path",
            path=[
                (NodeType.PHENOTYPE, EdgeType.GENE_HAS_PHENOTYPE, NodeType.GENE),
                (NodeType.GENE, EdgeType.GENE_ASSOCIATED_WITH_DISEASE, NodeType.DISEASE),
            ],
            priority=8,
        )

        # =====================================================================
        # Ortholog-based Paths (同源基因推理路徑)
        # =====================================================================
        self.metapaths["ortholog_phenotype_inference"] = MetapathSchema(
            name="ortholog_phenotype_inference",
            description="Infer human disease via mouse ortholog phenotype",
            path=[
                (NodeType.GENE, EdgeType.HUMAN_MOUSE_ORTHOLOG, NodeType.MOUSE_GENE),
                (NodeType.MOUSE_GENE, EdgeType.MOUSE_GENE_HAS_PHENOTYPE, NodeType.MOUSE_PHENOTYPE),
            ],
            priority=6,
            for_explanation=True,
        )

        self.metapaths["ortholog_group_reasoning"] = MetapathSchema(
            name="ortholog_group_reasoning",
            description="Reasoning via ortholog group",
            path=[
                (NodeType.GENE, EdgeType.ORTHOLOG_IN_GROUP, NodeType.ORTHOLOG_GROUP),
                (NodeType.ORTHOLOG_GROUP, EdgeType.ORTHOLOG_IN_GROUP, NodeType.MOUSE_GENE),
                (NodeType.MOUSE_GENE, EdgeType.MOUSE_GENE_HAS_PHENOTYPE, NodeType.MOUSE_PHENOTYPE),
            ],
            priority=5,
        )

        # =====================================================================
        # Pathway-based Paths
        # =====================================================================
        self.metapaths["gene_pathway_gene"] = MetapathSchema(
            name="gene_pathway_gene",
            description="Genes connected via shared pathway",
            path=[
                (NodeType.GENE, EdgeType.GENE_IN_PATHWAY, NodeType.PATHWAY),
                (NodeType.PATHWAY, EdgeType.GENE_IN_PATHWAY, NodeType.GENE),
            ],
            priority=4,
        )

        self.metapaths["pathway_disease_inference"] = MetapathSchema(
            name="pathway_disease_inference",
            description="Disease inference via pathway",
            path=[
                (NodeType.GENE, EdgeType.GENE_IN_PATHWAY, NodeType.PATHWAY),
                (NodeType.PATHWAY, EdgeType.GENE_IN_PATHWAY, NodeType.GENE),
                (NodeType.GENE, EdgeType.GENE_ASSOCIATED_WITH_DISEASE, NodeType.DISEASE),
            ],
            priority=3,
        )

        # =====================================================================
        # Literature-backed Paths (PubMed)
        # =====================================================================
        self.metapaths["publication_supported_association"] = MetapathSchema(
            name="publication_supported_association",
            description="Gene-disease association supported by publication",
            path=[
                (NodeType.GENE, EdgeType.PUBLICATION_MENTIONS_GENE, NodeType.PUBLICATION),
                (NodeType.PUBLICATION, EdgeType.PUBLICATION_MENTIONS_DISEASE, NodeType.DISEASE),
            ],
            priority=7,
            for_explanation=True,
        )

        # =====================================================================
        # Ontology Hierarchy Paths
        # =====================================================================
        self.metapaths["phenotype_hierarchy"] = MetapathSchema(
            name="phenotype_hierarchy",
            description="Phenotype subsumption via IS_A",
            path=[
                (NodeType.PHENOTYPE, EdgeType.IS_A, NodeType.PHENOTYPE),
            ],
            priority=9,
            for_inference=True,
        )

        self.metapaths["disease_hierarchy"] = MetapathSchema(
            name="disease_hierarchy",
            description="Disease subsumption via IS_A",
            path=[
                (NodeType.DISEASE, EdgeType.IS_A, NodeType.DISEASE),
            ],
            priority=9,
            for_inference=True,
        )

    # =========================================================================
    # Validation Methods
    # =========================================================================
    def validate_node(self, node_type: NodeType, attributes: Dict) -> List[str]:
        """
        驗證節點是否符合 Schema

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if node_type not in self.node_schemas:
            errors.append(f"Unknown node type: {node_type}")
            return errors

        schema = self.node_schemas[node_type]

        # Check required attributes
        for attr in schema.required_attributes:
            if attr not in attributes:
                errors.append(f"Missing required attribute '{attr}' for {node_type}")

        return errors

    def validate_edge(
        self,
        edge_type: EdgeType,
        source_type: NodeType,
        target_type: NodeType,
    ) -> List[str]:
        """
        驗證邊是否符合 Schema

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if edge_type not in self.edge_schemas:
            errors.append(f"Unknown edge type: {edge_type}")
            return errors

        schema = self.edge_schemas[edge_type]

        if source_type not in schema.source_types:
            errors.append(
                f"Invalid source type {source_type} for edge {edge_type}. "
                f"Expected one of: {schema.source_types}"
            )

        if target_type not in schema.target_types:
            errors.append(
                f"Invalid target type {target_type} for edge {edge_type}. "
                f"Expected one of: {schema.target_types}"
            )

        return errors

    def get_valid_edge_types(
        self,
        source_type: NodeType,
        target_type: Optional[NodeType] = None,
    ) -> List[EdgeType]:
        """獲取給定節點類型之間的有效邊類型"""
        valid_edges = []

        for edge_type, schema in self.edge_schemas.items():
            if source_type in schema.source_types:
                if target_type is None or target_type in schema.target_types:
                    valid_edges.append(edge_type)

        return valid_edges

    def get_metapaths_for_inference(self) -> List[MetapathSchema]:
        """獲取用於推理的 Metapaths (按優先級排序)"""
        return sorted(
            [mp for mp in self.metapaths.values() if mp.for_inference],
            key=lambda x: x.priority,
            reverse=True,
        )

    def get_metapaths_for_explanation(self) -> List[MetapathSchema]:
        """獲取用於解釋生成的 Metapaths"""
        return [mp for mp in self.metapaths.values() if mp.for_explanation]

    # =========================================================================
    # Export Methods
    # =========================================================================
    def to_dict(self) -> Dict:
        """Export schema to dictionary (for serialization)"""
        return {
            "version": "1.0.0",
            "node_types": [nt.value for nt in self.node_schemas.keys()],
            "edge_types": [et.value for et in self.edge_schemas.keys()],
            "metapaths": list(self.metapaths.keys()),
        }

    def get_hetero_metadata(self) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """
        獲取 PyTorch Geometric HeteroData 所需的 metadata

        Returns:
            (node_types, edge_types) for HeteroData
        """
        node_types = [nt.value for nt in self.node_schemas.keys()]

        edge_types = []
        for edge_type, schema in self.edge_schemas.items():
            for src in schema.source_types:
                for tgt in schema.target_types:
                    edge_types.append((src.value, edge_type.value, tgt.value))

        return node_types, edge_types


# =============================================================================
# Global Schema Instance
# =============================================================================
# Singleton instance for global access
_schema_instance: Optional[KnowledgeGraphSchema] = None


def get_kg_schema() -> KnowledgeGraphSchema:
    """獲取全局 KG Schema 實例"""
    global _schema_instance
    if _schema_instance is None:
        _schema_instance = KnowledgeGraphSchema()
    return _schema_instance
