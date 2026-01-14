"""
# ==============================================================================
# Module: src/kg/builder.py
# ==============================================================================
# Purpose: Knowledge Graph construction from multiple data sources
#
# Dependencies:
#   - External: networkx
#   - Internal: src.kg.graph (KnowledgeGraph)
#              src.ontology (Ontology)
#              src.core.types (Node, Edge, NodeID, NodeType, EdgeType, DataSource)
#              src.core.schema (KnowledgeGraphSchema)
#
# Input:
#   - Ontologies: HPO, MONDO, GO (via src.ontology.Ontology)
#   - Gene-Disease associations: DisGeNET, ClinVar format
#   - Ortholog mappings: OrthologMapping objects
#   - Literature: Publication objects
#
# Output:
#   - Populated KnowledgeGraph instance
#   - Graph statistics
# ==============================================================================
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from src.core.types import (
    DataSource,
    Edge,
    EdgeType,
    EvidenceLevel,
    EvidenceSource,
    Node,
    NodeID,
    NodeType,
    OrthologMapping,
    Publication,
)
from src.core.schema import KnowledgeGraphSchema, get_kg_schema
from src.kg.graph import KnowledgeGraph

logger = logging.getLogger(__name__)


# ==============================================================================
# Builder Configuration
# ==============================================================================
@dataclass
class KGBuilderConfig:
    """Knowledge Graph Builder Configuration"""

    # Ontology settings
    include_ontology_hierarchy: bool = True
    ontology_edge_type: EdgeType = EdgeType.IS_A

    # Gene-Disease association settings
    min_association_score: float = 0.0
    min_evidence_level: Optional[EvidenceLevel] = None

    # Ortholog settings
    include_orthologs: bool = True
    ortholog_species: List[str] = field(default_factory=lambda: ["mouse", "zebrafish"])

    # Literature settings
    include_literature: bool = False

    # Validation
    validate_schema: bool = True


# ==============================================================================
# Knowledge Graph Builder
# ==============================================================================
class KnowledgeGraphBuilder:
    """
    Knowledge Graph Builder

    Constructs a heterogeneous knowledge graph from multiple data sources:
    1. Ontologies (HPO, MONDO, GO) -> Nodes + IS_A edges
    2. Gene-Disease associations -> GENE_ASSOCIATED_WITH_DISEASE edges
    3. Phenotype-Disease annotations -> PHENOTYPE_OF_DISEASE edges
    4. Ortholog mappings -> Cross-species edges
    5. Literature -> Publication nodes + edges
    """

    def __init__(
        self,
        config: Optional[KGBuilderConfig] = None,
        schema: Optional[KnowledgeGraphSchema] = None,
    ):
        """
        Args:
            config: Builder configuration
            schema: KG schema for validation
        """
        self.config = config or KGBuilderConfig()
        self._schema = schema or get_kg_schema()
        self._graph = KnowledgeGraph(schema=self._schema)

        # Track data sources added
        self._sources_added: Set[str] = set()

        logger.info("KnowledgeGraphBuilder initialized")

    @property
    def graph(self) -> KnowledgeGraph:
        """Get the constructed graph"""
        return self._graph

    # ==========================================================================
    # Ontology Integration
    # ==========================================================================
    def add_ontology(
        self,
        ontology,  # Ontology type from src.ontology
        node_type: NodeType,
        include_hierarchy: Optional[bool] = None,
    ) -> int:
        """
        Add ontology terms as nodes and IS_A edges

        Args:
            ontology: Ontology instance (from src.ontology)
            node_type: NodeType for the ontology terms
            include_hierarchy: Whether to add IS_A edges (default: config setting)

        Returns:
            Number of nodes added
        """
        include_hierarchy = (
            include_hierarchy
            if include_hierarchy is not None
            else self.config.include_ontology_hierarchy
        )

        logger.info(f"Adding ontology {ontology.name} as {node_type.value} nodes")

        nodes_added = 0

        # Add term nodes
        for term_id in ontology.get_all_terms(include_obsolete=False):
            term_info = ontology.get_term(term_id)
            if term_info is None:
                continue

            # Determine data source from term ID prefix
            if term_id.startswith("HP:"):
                data_source = DataSource.HPO
            elif term_id.startswith("MONDO:"):
                data_source = DataSource.MONDO
            elif term_id.startswith("GO:"):
                data_source = DataSource.GO
            else:
                # Default based on node type
                data_source = DataSource.HPO

            node = Node(
                id=NodeID(source=data_source, local_id=term_id),
                node_type=node_type,
                name=term_info.get("name", ""),
                data_sources={data_source},
                attributes={
                    "definition": term_info.get("definition", ""),
                    "synonyms": term_info.get("synonyms", []),
                },
            )
            self._graph.add_node(node)
            nodes_added += 1

        # Add IS_A edges (hierarchy)
        if include_hierarchy:
            edges = ontology.to_edges()
            edges_added = 0

            for child_id, parent_id, rel_type in edges:
                if rel_type == "is_a":
                    # Determine data source from term ID prefix
                    if child_id.startswith("HP:"):
                        data_source = DataSource.HPO
                    elif child_id.startswith("MONDO:"):
                        data_source = DataSource.MONDO
                    elif child_id.startswith("GO:"):
                        data_source = DataSource.GO
                    else:
                        data_source = DataSource.HPO

                    edge = Edge(
                        source_id=NodeID(source=data_source, local_id=child_id),
                        target_id=NodeID(source=data_source, local_id=parent_id),
                        edge_type=EdgeType.IS_A,
                        weight=1.0,
                    )
                    self._graph.add_edge(edge)
                    edges_added += 1

            logger.info(f"Added {edges_added} IS_A edges from {ontology.name}")

        self._sources_added.add(f"ontology:{ontology.name}")
        logger.info(f"Added {nodes_added} {node_type.value} nodes from {ontology.name}")

        return nodes_added

    # ==========================================================================
    # Gene-Disease Associations
    # ==========================================================================
    def add_gene_disease_associations(
        self,
        associations: List[Dict[str, Any]],
        source: DataSource = DataSource.DISGENET,
    ) -> Tuple[int, int]:
        """
        Add gene-disease associations

        Args:
            associations: List of dicts with keys:
                - gene_id: Gene identifier (HGNC symbol or Entrez ID)
                - gene_symbol: HGNC symbol
                - disease_id: Disease identifier (MONDO/OMIM)
                - score: Association score [0, 1]
                - evidence_level: Optional EvidenceLevel
                - pmids: Optional list of PubMed IDs
            source: Data source

        Returns:
            (nodes_added, edges_added)
        """
        logger.info(f"Adding gene-disease associations from {source.value}")

        genes_added = set()
        edges_added = 0

        for assoc in associations:
            # Filter by score
            score = assoc.get("score", 0.0)
            if score < self.config.min_association_score:
                continue

            # Filter by evidence level
            evidence = assoc.get("evidence_level")
            if self.config.min_evidence_level and evidence:
                if evidence.value < self.config.min_evidence_level.value:
                    continue

            gene_id = assoc.get("gene_id", "")
            gene_symbol = assoc.get("gene_symbol", gene_id)
            disease_id = assoc.get("disease_id", "")

            if not gene_id or not disease_id:
                continue

            # Add gene node if not exists
            gene_node_id = NodeID(source=source, local_id=gene_id)
            if not self._graph.has_node(gene_node_id):
                gene_node = Node(
                    id=gene_node_id,
                    node_type=NodeType.GENE,
                    name=gene_symbol,
                    data_sources={source},
                    attributes={
                        "symbol": gene_symbol,
                        "entrez_id": gene_id if gene_id.isdigit() else "",
                    },
                )
                self._graph.add_node(gene_node)
                genes_added.add(gene_id)

            # Create edge - determine disease source from ID prefix
            if disease_id.startswith("MONDO:"):
                disease_source = DataSource.MONDO
            elif disease_id.startswith("OMIM:"):
                disease_source = DataSource.OMIM
            else:
                disease_source = DataSource.MONDO

            disease_node_id = NodeID(source=disease_source, local_id=disease_id)

            # Only add edge if disease node exists
            if self._graph.has_node(disease_node_id):
                evidence_sources = []
                if evidence:
                    evidence_sources.append(EvidenceSource(
                        source_type=source,
                        source_id=gene_id,
                        evidence_level=evidence,
                        metadata={"pmids": assoc.get("pmids", [])},
                    ))

                edge = Edge(
                    source_id=gene_node_id,
                    target_id=disease_node_id,
                    edge_type=EdgeType.GENE_ASSOCIATED_WITH_DISEASE,
                    weight=score,
                    evidence_sources=evidence_sources,
                    data_source=source,
                )
                self._graph.add_edge(edge)
                edges_added += 1

        self._sources_added.add(f"associations:{source.value}")
        logger.info(
            f"Added {len(genes_added)} gene nodes, {edges_added} association edges"
        )

        return len(genes_added), edges_added

    # ==========================================================================
    # Phenotype-Disease Annotations
    # ==========================================================================
    def add_phenotype_disease_annotations(
        self,
        annotations: List[Dict[str, Any]],
        source: DataSource = DataSource.HPO,
    ) -> int:
        """
        Add phenotype-disease annotations

        Args:
            annotations: List of dicts with keys:
                - phenotype_id: HPO ID (e.g., "HP:0001250")
                - disease_id: Disease ID (MONDO/OMIM)
                - frequency: Optional frequency of phenotype in disease

        Returns:
            Number of edges added
        """
        logger.info(f"Adding phenotype-disease annotations from {source.value}")

        edges_added = 0

        for annot in annotations:
            pheno_id = annot.get("phenotype_id", "")
            disease_id = annot.get("disease_id", "")
            frequency = annot.get("frequency", 1.0)

            if not pheno_id or not disease_id:
                continue

            # Determine sources from ID prefixes
            pheno_source = DataSource.HPO if pheno_id.startswith("HP:") else source
            if disease_id.startswith("MONDO:"):
                disease_source = DataSource.MONDO
            elif disease_id.startswith("OMIM:"):
                disease_source = DataSource.OMIM
            else:
                disease_source = DataSource.MONDO

            pheno_node_id = NodeID(source=pheno_source, local_id=pheno_id)
            disease_node_id = NodeID(source=disease_source, local_id=disease_id)

            # Only add edge if both nodes exist
            if self._graph.has_node(pheno_node_id) and self._graph.has_node(disease_node_id):
                edge = Edge(
                    source_id=pheno_node_id,
                    target_id=disease_node_id,
                    edge_type=EdgeType.PHENOTYPE_OF_DISEASE,
                    weight=frequency,
                )
                self._graph.add_edge(edge)
                edges_added += 1

        self._sources_added.add(f"annotations:{source.value}")
        logger.info(f"Added {edges_added} phenotype-disease edges")

        return edges_added

    # ==========================================================================
    # Gene-Phenotype Associations
    # ==========================================================================
    def add_gene_phenotype_associations(
        self,
        associations: List[Dict[str, Any]],
        source: DataSource = DataSource.HPO,
    ) -> int:
        """
        Add gene-phenotype associations

        Args:
            associations: List of dicts with keys:
                - gene_id: Gene identifier
                - phenotype_id: HPO ID

        Returns:
            Number of edges added
        """
        logger.info(f"Adding gene-phenotype associations from {source.value}")

        edges_added = 0

        for assoc in associations:
            gene_id = assoc.get("gene_id", "")
            pheno_id = assoc.get("phenotype_id", "")

            if not gene_id or not pheno_id:
                continue

            # Determine sources from ID prefixes
            gene_source = source
            pheno_source = DataSource.HPO if pheno_id.startswith("HP:") else source

            gene_node_id = NodeID(source=gene_source, local_id=gene_id)
            pheno_node_id = NodeID(source=pheno_source, local_id=pheno_id)

            if self._graph.has_node(gene_node_id) and self._graph.has_node(pheno_node_id):
                edge = Edge(
                    source_id=gene_node_id,
                    target_id=pheno_node_id,
                    edge_type=EdgeType.GENE_HAS_PHENOTYPE,
                    weight=1.0,
                )
                self._graph.add_edge(edge)
                edges_added += 1

        logger.info(f"Added {edges_added} gene-phenotype edges")
        return edges_added

    # ==========================================================================
    # Ortholog Integration (Deep Integration - Plan B)
    # ==========================================================================
    def add_orthologs(
        self,
        ortholog_mappings: List[OrthologMapping],
    ) -> Tuple[int, int]:
        """
        Add ortholog gene mappings (Deep Integration into GNN)

        This implements Plan B: orthologs are integrated into the main KG
        for GNN-based cross-species reasoning.

        Args:
            ortholog_mappings: List of OrthologMapping objects

        Returns:
            (nodes_added, edges_added)
        """
        if not self.config.include_orthologs:
            logger.info("Ortholog integration disabled, skipping")
            return 0, 0

        logger.info(f"Adding {len(ortholog_mappings)} ortholog mappings")

        nodes_added = 0
        edges_added = 0

        for mapping in ortholog_mappings:
            # Add ortholog gene node
            ortholog_node_id = mapping.ortholog_gene_id

            if not self._graph.has_node(ortholog_node_id):
                # Determine node type based on species
                if mapping.ortholog_species.value == "mouse":
                    node_type = NodeType.MOUSE_GENE
                elif mapping.ortholog_species.value == "zebrafish":
                    node_type = NodeType.ZEBRAFISH_GENE
                else:
                    continue

                ortholog_node = Node(
                    id=ortholog_node_id,
                    node_type=node_type,
                    name=str(ortholog_node_id.local_id),
                    species=mapping.ortholog_species,
                    data_sources={mapping.source},
                    attributes={
                        "symbol": str(ortholog_node_id.local_id),
                    },
                )
                self._graph.add_node(ortholog_node)
                nodes_added += 1

            # Add ortholog edge (human gene -> ortholog gene)
            if mapping.ortholog_species.value == "mouse":
                edge_type = EdgeType.HUMAN_MOUSE_ORTHOLOG
            elif mapping.ortholog_species.value == "zebrafish":
                edge_type = EdgeType.HUMAN_ZEBRAFISH_ORTHOLOG
            else:
                edge_type = EdgeType.ORTHOLOG_OF

            edge = Edge(
                source_id=mapping.human_gene_id,
                target_id=ortholog_node_id,
                edge_type=edge_type,
                weight=mapping.confidence_score,
                data_source=mapping.source,
                attributes={
                    "ortholog_type": mapping.ortholog_type,
                },
            )
            self._graph.add_edge(edge)
            edges_added += 1

        self._sources_added.add("orthologs")
        logger.info(f"Added {nodes_added} ortholog nodes, {edges_added} ortholog edges")

        return nodes_added, edges_added

    # ==========================================================================
    # Literature Integration
    # ==========================================================================
    def add_literature_edges(
        self,
        publications: List[Publication],
    ) -> Tuple[int, int]:
        """
        Add publication nodes and literature evidence edges

        Args:
            publications: List of Publication objects

        Returns:
            (nodes_added, edges_added)
        """
        if not self.config.include_literature:
            logger.info("Literature integration disabled, skipping")
            return 0, 0

        logger.info(f"Adding {len(publications)} publications")

        nodes_added = 0
        edges_added = 0

        for pub in publications:
            # Add publication node
            pub_node_id = NodeID(source=DataSource.PUBMED, local_id=pub.pmid)

            if not self._graph.has_node(pub_node_id):
                pub_node = Node(
                    id=pub_node_id,
                    node_type=NodeType.PUBLICATION,
                    name=pub.title,
                    data_sources={DataSource.PUBMED},
                    attributes={
                        "abstract": pub.abstract or "",
                        "journal": pub.journal or "",
                        "year": pub.publication_year or 0,
                    },
                )
                self._graph.add_node(pub_node)
                nodes_added += 1

            # Add edges for mentioned genes - need to find the actual gene node
            for gene_id in pub.mentioned_genes:
                # Try multiple sources for gene lookup
                gene_node_id = self._find_gene_node(gene_id)
                if gene_node_id and self._graph.has_node(gene_node_id):
                    edge = Edge(
                        source_id=pub_node_id,
                        target_id=gene_node_id,
                        edge_type=EdgeType.PUBLICATION_MENTIONS_GENE,
                        weight=1.0,
                        data_source=DataSource.PUBMED,
                    )
                    self._graph.add_edge(edge)
                    edges_added += 1

            # Add edges for mentioned diseases
            for disease_id in pub.mentioned_diseases:
                # Determine disease source
                if disease_id.startswith("MONDO:"):
                    disease_source = DataSource.MONDO
                elif disease_id.startswith("OMIM:"):
                    disease_source = DataSource.OMIM
                else:
                    disease_source = DataSource.MONDO

                disease_node_id = NodeID(source=disease_source, local_id=disease_id)
                if self._graph.has_node(disease_node_id):
                    edge = Edge(
                        source_id=pub_node_id,
                        target_id=disease_node_id,
                        edge_type=EdgeType.PUBLICATION_MENTIONS_DISEASE,
                        weight=1.0,
                        data_source=DataSource.PUBMED,
                    )
                    self._graph.add_edge(edge)
                    edges_added += 1

        self._sources_added.add("literature")
        logger.info(f"Added {nodes_added} publication nodes, {edges_added} literature edges")

        return nodes_added, edges_added

    # ==========================================================================
    # Generic Association Loader
    # ==========================================================================
    def add_associations(
        self,
        associations: List[Tuple[str, str, EdgeType, Dict]],
        source: DataSource,
    ) -> int:
        """
        Add generic associations

        Args:
            associations: List of (source_id, target_id, edge_type, attributes)
            source: Data source

        Returns:
            Number of edges added
        """
        logger.info(f"Adding {len(associations)} associations from {source.value}")

        edges_added = 0

        for src_id, tgt_id, edge_type, attrs in associations:
            # Try to find nodes
            source_node_id = self._find_node_id(src_id)
            target_node_id = self._find_node_id(tgt_id)

            if source_node_id and target_node_id:
                if self._graph.has_node(source_node_id) and self._graph.has_node(target_node_id):
                    edge = Edge(
                        source_id=source_node_id,
                        target_id=target_node_id,
                        edge_type=edge_type,
                        weight=attrs.get("weight", 1.0),
                        attributes=attrs,
                    )
                    self._graph.add_edge(edge)
                    edges_added += 1

        logger.info(f"Added {edges_added} edges from {source.value}")
        return edges_added

    def _find_node_id(self, id_str: str) -> Optional[NodeID]:
        """Try to find a node ID by string"""
        # Try common prefixes mapped to DataSource
        prefixes = [
            ("HP:", DataSource.HPO),
            ("MONDO:", DataSource.MONDO),
            ("OMIM:", DataSource.OMIM),
            ("GO:", DataSource.GO),
        ]

        for prefix, data_source in prefixes:
            if id_str.startswith(prefix):
                node_id = NodeID(source=data_source, local_id=id_str)
                if self._graph.has_node(node_id):
                    return node_id

        # Check if it exists in graph with any common source
        for data_source in [DataSource.DISGENET, DataSource.HPO, DataSource.MONDO, DataSource.GO]:
            node_id = NodeID(source=data_source, local_id=id_str)
            if self._graph.has_node(node_id):
                return node_id

        return None

    def _find_gene_node(self, gene_id: str) -> Optional[NodeID]:
        """Try to find a gene node by ID, checking multiple sources"""
        gene_sources = [
            DataSource.DISGENET,
            DataSource.CLINVAR,
            DataSource.ENSEMBL,
            DataSource.MGI,
            DataSource.ZFIN,
        ]

        for source in gene_sources:
            node_id = NodeID(source=source, local_id=gene_id)
            if self._graph.has_node(node_id):
                return node_id

        return None

    # ==========================================================================
    # Build and Statistics
    # ==========================================================================
    def build(self) -> KnowledgeGraph:
        """
        Finalize and return the built graph

        Returns:
            The constructed KnowledgeGraph
        """
        stats = self._graph.get_statistics()
        logger.info(f"KnowledgeGraph built: {stats}")
        logger.info(f"Data sources: {self._sources_added}")

        return self._graph

    def get_build_summary(self) -> Dict[str, Any]:
        """Get build summary"""
        return {
            "graph_stats": self._graph.get_statistics(),
            "sources_added": list(self._sources_added),
        }


# ==============================================================================
# Factory Function
# ==============================================================================
def create_kg_builder(
    config: Optional[KGBuilderConfig] = None,
    schema: Optional[KnowledgeGraphSchema] = None,
) -> KnowledgeGraphBuilder:
    """
    Factory function: Create a KnowledgeGraph builder

    Args:
        config: Builder configuration
        schema: KG schema

    Returns:
        KnowledgeGraphBuilder instance
    """
    return KnowledgeGraphBuilder(config=config, schema=schema)
