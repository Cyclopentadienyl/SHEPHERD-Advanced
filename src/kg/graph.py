"""
# ==============================================================================
# Module: src/kg/graph.py
# ==============================================================================
# Purpose: Core Knowledge Graph data structure with heterogeneous node/edge support
#
# Dependencies:
#   - External: networkx, numpy, torch (optional), torch_geometric (optional)
#   - Internal: src.core.types (Node, Edge, NodeID, NodeType, EdgeType)
#              src.core.schema (KnowledgeGraphSchema)
#
# Input:
#   - Nodes: Node objects with NodeID, NodeType, and attributes
#   - Edges: Edge objects with source/target NodeID, EdgeType, and weight
#
# Output:
#   - Graph queries: neighbors, subgraphs, paths
#   - Export: PyG HeteroData, NetworkX graph, edge lists
# ==============================================================================
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, TYPE_CHECKING

import networkx as nx
import numpy as np

from src.core.types import (
    DataSource,
    Edge,
    EdgeType,
    Node,
    NodeID,
    NodeType,
)
from src.core.schema import KnowledgeGraphSchema, get_kg_schema

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ==============================================================================
# Knowledge Graph
# ==============================================================================
class KnowledgeGraph:
    """
    Heterogeneous Knowledge Graph

    Supports:
    - Multiple node types (Gene, Disease, Phenotype, etc.)
    - Multiple edge types (GENE_ASSOCIATED_WITH_DISEASE, IS_A, etc.)
    - Efficient neighbor queries
    - Subgraph extraction
    - Export to PyTorch Geometric HeteroData
    """

    def __init__(self, schema: Optional[KnowledgeGraphSchema] = None):
        """
        Args:
            schema: KG schema for validation (optional)
        """
        self._schema = schema or get_kg_schema()

        # Node storage: {node_id_str: Node}
        self._nodes: Dict[str, Node] = {}

        # Node type index: {NodeType: Set[node_id_str]}
        self._nodes_by_type: Dict[NodeType, Set[str]] = defaultdict(set)

        # Edge storage: list of Edge objects
        self._edges: List[Edge] = []

        # Edge indices for fast lookup
        # {source_id_str: [(target_id_str, edge_idx)]}
        self._outgoing_edges: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        # {target_id_str: [(source_id_str, edge_idx)]}
        self._incoming_edges: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        # {EdgeType: [edge_idx]}
        self._edges_by_type: Dict[EdgeType, List[int]] = defaultdict(list)

        # Node ID mapping for PyG export: {node_type: {node_id_str: int_idx}}
        self._node_id_to_idx: Dict[NodeType, Dict[str, int]] = defaultdict(dict)

        logger.info("KnowledgeGraph initialized")

    # ==========================================================================
    # Properties
    # ==========================================================================
    @property
    def num_nodes(self) -> Dict[NodeType, int]:
        """Number of nodes per type"""
        return {nt: len(nodes) for nt, nodes in self._nodes_by_type.items()}

    @property
    def num_edges(self) -> Dict[EdgeType, int]:
        """Number of edges per type"""
        return {et: len(edges) for et, edges in self._edges_by_type.items()}

    @property
    def total_nodes(self) -> int:
        """Total number of nodes"""
        return len(self._nodes)

    @property
    def total_edges(self) -> int:
        """Total number of edges"""
        return len(self._edges)

    # ==========================================================================
    # Node Operations
    # ==========================================================================
    def add_node(self, node: Node) -> None:
        """
        Add a node to the graph

        Args:
            node: Node object to add
        """
        node_id_str = str(node.id)

        # Skip if already exists
        if node_id_str in self._nodes:
            return

        # Validate against schema (optional)
        if self._schema:
            errors = self._schema.validate_node(node.node_type, node.attributes)
            if errors:
                logger.warning(f"Node validation warnings for {node_id_str}: {errors}")

        # Store node
        self._nodes[node_id_str] = node
        self._nodes_by_type[node.node_type].add(node_id_str)

        # Update ID mapping
        idx = len(self._node_id_to_idx[node.node_type])
        self._node_id_to_idx[node.node_type][node_id_str] = idx

    def add_nodes(self, nodes: List[Node]) -> None:
        """Add multiple nodes"""
        for node in nodes:
            self.add_node(node)

    def get_node(self, node_id: NodeID) -> Optional[Node]:
        """Get a node by ID"""
        return self._nodes.get(str(node_id))

    def has_node(self, node_id: NodeID) -> bool:
        """Check if node exists"""
        return str(node_id) in self._nodes

    def get_nodes_by_type(self, node_type: NodeType) -> List[Node]:
        """Get all nodes of a specific type"""
        return [
            self._nodes[nid]
            for nid in self._nodes_by_type.get(node_type, set())
        ]

    def iter_nodes(self, node_type: Optional[NodeType] = None) -> Iterator[Node]:
        """Iterate over nodes"""
        if node_type:
            for nid in self._nodes_by_type.get(node_type, set()):
                yield self._nodes[nid]
        else:
            yield from self._nodes.values()

    # ==========================================================================
    # Edge Operations
    # ==========================================================================
    def add_edge(self, edge: Edge) -> None:
        """
        Add an edge to the graph

        Args:
            edge: Edge object to add
        """
        source_str = str(edge.source_id)
        target_str = str(edge.target_id)

        # Validate nodes exist
        if source_str not in self._nodes:
            logger.warning(f"Source node {source_str} not found, skipping edge")
            return
        if target_str not in self._nodes:
            logger.warning(f"Target node {target_str} not found, skipping edge")
            return

        # Validate against schema (optional)
        if self._schema:
            source_type = self._nodes[source_str].node_type
            target_type = self._nodes[target_str].node_type
            errors = self._schema.validate_edge(edge.edge_type, source_type, target_type)
            if errors:
                logger.warning(f"Edge validation warnings: {errors}")

        # Store edge
        edge_idx = len(self._edges)
        self._edges.append(edge)

        # Update indices
        self._outgoing_edges[source_str].append((target_str, edge_idx))
        self._incoming_edges[target_str].append((source_str, edge_idx))
        self._edges_by_type[edge.edge_type].append(edge_idx)

    def add_edges(self, edges: List[Edge]) -> None:
        """Add multiple edges"""
        for edge in edges:
            self.add_edge(edge)

    def get_edge(self, source_id: NodeID, target_id: NodeID, edge_type: Optional[EdgeType] = None) -> Optional[Edge]:
        """Get a specific edge"""
        source_str = str(source_id)
        target_str = str(target_id)

        for _, edge_idx in self._outgoing_edges.get(source_str, []):
            edge = self._edges[edge_idx]
            if str(edge.target_id) == target_str:
                if edge_type is None or edge.edge_type == edge_type:
                    return edge
        return None

    def get_edges(
        self,
        source_id: Optional[NodeID] = None,
        target_id: Optional[NodeID] = None,
        edge_type: Optional[EdgeType] = None,
    ) -> List[Edge]:
        """
        Query edges with optional filters

        Args:
            source_id: Filter by source node
            target_id: Filter by target node
            edge_type: Filter by edge type
        """
        results = []

        if source_id is not None:
            source_str = str(source_id)
            for _, edge_idx in self._outgoing_edges.get(source_str, []):
                edge = self._edges[edge_idx]
                if target_id is not None and str(edge.target_id) != str(target_id):
                    continue
                if edge_type is not None and edge.edge_type != edge_type:
                    continue
                results.append(edge)

        elif target_id is not None:
            target_str = str(target_id)
            for _, edge_idx in self._incoming_edges.get(target_str, []):
                edge = self._edges[edge_idx]
                if edge_type is not None and edge.edge_type != edge_type:
                    continue
                results.append(edge)

        elif edge_type is not None:
            for edge_idx in self._edges_by_type.get(edge_type, []):
                results.append(self._edges[edge_idx])

        else:
            results = list(self._edges)

        return results

    def iter_edges(self, edge_type: Optional[EdgeType] = None) -> Iterator[Edge]:
        """Iterate over edges"""
        if edge_type:
            for edge_idx in self._edges_by_type.get(edge_type, []):
                yield self._edges[edge_idx]
        else:
            yield from self._edges

    # ==========================================================================
    # Neighbor Queries
    # ==========================================================================
    def get_neighbors(
        self,
        node_id: NodeID,
        edge_types: Optional[List[EdgeType]] = None,
        direction: str = "both",
    ) -> List[Tuple[NodeID, EdgeType]]:
        """
        Get neighbor nodes

        Args:
            node_id: Query node
            edge_types: Filter by edge types (None = all)
            direction: "in", "out", or "both"

        Returns:
            List of (neighbor_id, edge_type) tuples
        """
        node_str = str(node_id)
        neighbors = []

        # Outgoing edges
        if direction in ("out", "both"):
            for target_str, edge_idx in self._outgoing_edges.get(node_str, []):
                edge = self._edges[edge_idx]
                if edge_types is None or edge.edge_type in edge_types:
                    neighbors.append((edge.target_id, edge.edge_type))

        # Incoming edges
        if direction in ("in", "both"):
            for source_str, edge_idx in self._incoming_edges.get(node_str, []):
                edge = self._edges[edge_idx]
                if edge_types is None or edge.edge_type in edge_types:
                    neighbors.append((edge.source_id, edge.edge_type))

        return neighbors

    def get_neighbor_nodes(
        self,
        node_id: NodeID,
        edge_types: Optional[List[EdgeType]] = None,
        direction: str = "both",
    ) -> List[Node]:
        """Get neighbor Node objects"""
        neighbors = self.get_neighbors(node_id, edge_types, direction)
        return [
            self._nodes[str(nid)]
            for nid, _ in neighbors
            if str(nid) in self._nodes
        ]

    # ==========================================================================
    # Subgraph Extraction
    # ==========================================================================
    def get_subgraph(
        self,
        node_ids: List[NodeID],
        num_hops: int = 1,
    ) -> "KnowledgeGraph":
        """
        Extract a subgraph around given nodes

        Args:
            node_ids: Seed nodes
            num_hops: Number of hops to expand

        Returns:
            New KnowledgeGraph containing the subgraph
        """
        # Collect nodes via BFS
        visited = set(str(nid) for nid in node_ids)
        frontier = list(visited)

        for _ in range(num_hops):
            next_frontier = []
            for node_str in frontier:
                node_id = self._nodes[node_str].id if node_str in self._nodes else None
                if node_id is None:
                    continue

                for neighbor_id, _ in self.get_neighbors(node_id):
                    neighbor_str = str(neighbor_id)
                    if neighbor_str not in visited:
                        visited.add(neighbor_str)
                        next_frontier.append(neighbor_str)

            frontier = next_frontier

        # Create subgraph
        subgraph = KnowledgeGraph(schema=self._schema)

        # Add nodes
        for node_str in visited:
            if node_str in self._nodes:
                subgraph.add_node(self._nodes[node_str])

        # Add edges between visited nodes
        for edge in self._edges:
            source_str = str(edge.source_id)
            target_str = str(edge.target_id)
            if source_str in visited and target_str in visited:
                subgraph.add_edge(edge)

        return subgraph

    # ==========================================================================
    # Export Methods
    # ==========================================================================
    def to_networkx(self) -> nx.MultiDiGraph:
        """
        Export to NetworkX MultiDiGraph

        Returns:
            NetworkX graph with node/edge attributes
        """
        G = nx.MultiDiGraph()

        # Add nodes
        for node in self._nodes.values():
            G.add_node(
                str(node.id),
                node_type=node.node_type.value,
                **node.attributes,
            )

        # Add edges
        for edge in self._edges:
            G.add_edge(
                str(edge.source_id),
                str(edge.target_id),
                edge_type=edge.edge_type.value,
                weight=edge.weight,
            )

        return G

    def to_pyg_hetero_data(self):
        """
        Export to PyTorch Geometric HeteroData

        Returns:
            HeteroData object ready for GNN training

        Raises:
            ImportError if torch_geometric not installed
        """
        try:
            import torch
            from torch_geometric.data import HeteroData
        except ImportError:
            raise ImportError(
                "torch and torch_geometric required for PyG export. "
                "Install via platform-specific script."
            )

        data = HeteroData()

        # Build node features (placeholder: just indices)
        for node_type, node_ids in self._nodes_by_type.items():
            num_nodes = len(node_ids)
            if num_nodes > 0:
                # Placeholder features (replace with actual embeddings later)
                data[node_type.value].x = torch.zeros(num_nodes, 1)
                data[node_type.value].num_nodes = num_nodes

        # Build edge indices
        edge_index_dict: Dict[Tuple[str, str, str], List[List[int]]] = defaultdict(
            lambda: [[], []]
        )

        for edge in self._edges:
            source_str = str(edge.source_id)
            target_str = str(edge.target_id)

            source_node = self._nodes.get(source_str)
            target_node = self._nodes.get(target_str)

            if source_node is None or target_node is None:
                continue

            src_type = source_node.node_type.value
            tgt_type = target_node.node_type.value
            edge_type = edge.edge_type.value

            key = (src_type, edge_type, tgt_type)

            src_idx = self._node_id_to_idx[source_node.node_type].get(source_str)
            tgt_idx = self._node_id_to_idx[target_node.node_type].get(target_str)

            if src_idx is not None and tgt_idx is not None:
                edge_index_dict[key][0].append(src_idx)
                edge_index_dict[key][1].append(tgt_idx)

        # Convert to tensors
        for key, (src_list, tgt_list) in edge_index_dict.items():
            if src_list:
                data[key].edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)

        return data

    def to_edge_list(self) -> List[Tuple[str, str, str, float]]:
        """
        Export to edge list format

        Returns:
            List of (source_id, target_id, edge_type, weight) tuples
        """
        return [
            (str(e.source_id), str(e.target_id), e.edge_type.value, e.weight)
            for e in self._edges
        ]

    def metadata(self) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """
        Get graph metadata for PyG models.

        Returns:
            (node_types, edge_types) where:
            - node_types: List of node type strings
            - edge_types: List of (src_type, rel_type, dst_type) tuples

        Usage:
            metadata = kg.metadata()
            model = ShepherdGNN(metadata=metadata, ...)
        """
        # Collect node types that have nodes
        node_types = [nt.value for nt in self._nodes_by_type.keys() if self._nodes_by_type[nt]]

        # Collect edge types that have edges
        edge_types = []
        seen_edge_types = set()

        for edge in self._edges:
            source_node = self._nodes.get(str(edge.source_id))
            target_node = self._nodes.get(str(edge.target_id))

            if source_node is None or target_node is None:
                continue

            key = (
                source_node.node_type.value,
                edge.edge_type.value,
                target_node.node_type.value,
            )

            if key not in seen_edge_types:
                seen_edge_types.add(key)
                edge_types.append(key)

        return node_types, edge_types

    def get_node_id_mapping(self) -> Dict[str, Dict[str, int]]:
        """
        Get node ID to index mapping for each node type.

        Returns:
            {node_type: {node_id_str: int_idx}}

        Useful for:
        - Looking up node indices in PyG HeteroData
        - Mapping inference results back to original IDs
        """
        return {
            nt.value: dict(mapping)
            for nt, mapping in self._node_id_to_idx.items()
        }

    def get_reverse_node_mapping(self) -> Dict[str, Dict[int, str]]:
        """
        Get index to node ID mapping for each node type.

        Returns:
            {node_type: {int_idx: node_id_str}}
        """
        return {
            nt.value: {idx: nid for nid, idx in mapping.items()}
            for nt, mapping in self._node_id_to_idx.items()
        }

    # ==========================================================================
    # Statistics
    # ==========================================================================
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        return {
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "nodes_by_type": {nt.value: cnt for nt, cnt in self.num_nodes.items()},
            "edges_by_type": {et.value: cnt for et, cnt in self.num_edges.items()},
        }

    # ==========================================================================
    # Serialization
    # ==========================================================================
    def save_json(self, filepath: str) -> None:
        """
        Save knowledge graph to JSON file.

        Format:
            {
                "nodes": [{"id": "hpo:HP:0001250", "node_type": "phenotype", "name": "...", ...}],
                "edges": [{"source": "...", "target": "...", "edge_type": "...", "weight": 1.0}]
            }

        Args:
            filepath: Output file path (.json)
        """
        nodes_data = []
        for node in self._nodes.values():
            node_dict = {
                "id": str(node.id),
                "source": node.id.source.value,
                "local_id": node.id.local_id,
                "node_type": node.node_type.value,
                "name": node.name,
            }
            if node.description:
                node_dict["description"] = node.description
            if node.aliases:
                node_dict["aliases"] = node.aliases
            if node.species:
                node_dict["species"] = node.species.value
            nodes_data.append(node_dict)

        edges_data = []
        for edge in self._edges:
            edge_dict = {
                "source": str(edge.source_id),
                "target": str(edge.target_id),
                "source_ds": edge.source_id.source.value,
                "source_lid": edge.source_id.local_id,
                "target_ds": edge.target_id.source.value,
                "target_lid": edge.target_id.local_id,
                "edge_type": edge.edge_type.value,
                "weight": edge.weight,
                "confidence": edge.confidence,
            }
            edges_data.append(edge_dict)

        data = {
            "format_version": "1.0",
            "nodes": nodes_data,
            "edges": edges_data,
        }

        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(
            f"KG saved to {filepath}: "
            f"{len(nodes_data)} nodes, {len(edges_data)} edges"
        )

    @classmethod
    def load_json(cls, filepath: str) -> "KnowledgeGraph":
        """
        Load knowledge graph from JSON file.

        Args:
            filepath: Input file path (.json)

        Returns:
            Reconstructed KnowledgeGraph instance
        """
        from src.core.types import Species

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        kg = cls()

        # Load nodes
        for node_dict in data.get("nodes", []):
            node_id = NodeID(
                source=DataSource(node_dict["source"]),
                local_id=node_dict["local_id"],
            )
            species = None
            if "species" in node_dict:
                species = Species(node_dict["species"])
            node = Node(
                id=node_id,
                node_type=NodeType(node_dict["node_type"]),
                name=node_dict["name"],
                description=node_dict.get("description"),
                aliases=node_dict.get("aliases", []),
                species=species,
            )
            kg.add_node(node)

        # Load edges
        for edge_dict in data.get("edges", []):
            source_id = NodeID(
                source=DataSource(edge_dict["source_ds"]),
                local_id=edge_dict["source_lid"],
            )
            target_id = NodeID(
                source=DataSource(edge_dict["target_ds"]),
                local_id=edge_dict["target_lid"],
            )
            edge = Edge(
                source_id=source_id,
                target_id=target_id,
                edge_type=EdgeType(edge_dict["edge_type"]),
                weight=edge_dict.get("weight", 1.0),
                confidence=edge_dict.get("confidence", 1.0),
            )
            kg.add_edge(edge)

        logger.info(
            f"KG loaded from {filepath}: "
            f"{kg.total_nodes} nodes, {kg.total_edges} edges"
        )
        return kg

    def __repr__(self) -> str:
        return f"KnowledgeGraph(nodes={self.total_nodes}, edges={self.total_edges})"
