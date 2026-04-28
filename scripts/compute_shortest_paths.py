#!/usr/bin/env python3
"""
SHEPHERD-Advanced Shortest Path Precomputation Script
======================================================
Pre-computes shortest path lengths from every phenotype node to every
gene/disease node in the knowledge graph.

Per the original SHEPHERD paper (Alsentzer et al., npj Digital Medicine 2025),
the final score is:
    final_score = eta * embedding_similarity + (1 - eta) * SP_similarity

Where SP_similarity is derived from these pre-computed shortest path lengths.

Script: scripts/compute_shortest_paths.py

Output format (parallel tensors, saved as shortest_paths.pt):
    {
        "phenotype_idx":  Tensor(N,) int64    # source phenotype global index
        "target_idx":     Tensor(N,) int64    # target gene/disease global index
        "target_type":    Tensor(N,) int64    # 0 = gene, 1 = disease
        "distance":       Tensor(N,) int8     # hop count (1-5; >5 dropped)
    }

Pairs not in the table are implicitly "no connection within max_hops" and
should be assigned SP_similarity = 0.0 by the inference pipeline.

Usage:
    python scripts/compute_shortest_paths.py \\
        --kg-path data/workspaces/demo/kg.json \\
        --output-dir data/workspaces/demo/ \\
        --max-hops 5
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.core.types import NodeType
from src.kg.graph import KnowledgeGraph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Encoding for target_type tensor
TARGET_TYPE_GENE = 0
TARGET_TYPE_DISEASE = 1


def bfs_from_source(
    adjacency: Dict[str, List[str]],
    source: str,
    max_hops: int,
) -> Dict[str, int]:
    """
    Single-source BFS up to max_hops.

    Returns dict {node_str: hop_count} for all reachable nodes within max_hops.
    The source itself is included with hop_count=0.
    """
    distances = {source: 0}
    queue = deque([(source, 0)])

    while queue:
        current, depth = queue.popleft()
        if depth >= max_hops:
            continue
        for neighbor in adjacency.get(current, ()):
            if neighbor not in distances:
                distances[neighbor] = depth + 1
                queue.append((neighbor, depth + 1))

    return distances


def build_undirected_adjacency(kg: KnowledgeGraph) -> Dict[str, List[str]]:
    """
    Build undirected adjacency list from KG edges.

    Shortest path for clinical reasoning is direction-agnostic — a path from
    phenotype to gene through disease is meaningful regardless of edge
    direction in the underlying KG.
    """
    adj: Dict[str, List[str]] = {}
    for edge in kg._edges:
        src = str(edge.source_id)
        tgt = str(edge.target_id)
        adj.setdefault(src, []).append(tgt)
        adj.setdefault(tgt, []).append(src)
    return adj


def compute_shortest_paths(
    kg: KnowledgeGraph,
    max_hops: int = 5,
) -> Dict[str, torch.Tensor]:
    """
    Compute shortest path lengths from all phenotype nodes to all gene
    and disease nodes (up to max_hops).

    Uses kg.get_node_id_mapping() so that the indices in the output tensors
    correspond directly to the indices in node_features.pt — the inference
    pipeline can use them as direct lookups.
    """
    node_mapping = kg.get_node_id_mapping()

    pheno_map = node_mapping.get(NodeType.PHENOTYPE.value, {})
    gene_map = node_mapping.get(NodeType.GENE.value, {})
    disease_map = node_mapping.get(NodeType.DISEASE.value, {})

    if not pheno_map:
        raise ValueError("KG has no phenotype nodes")
    if not gene_map and not disease_map:
        raise ValueError("KG has no gene or disease nodes")

    logger.info(
        f"Computing shortest paths: "
        f"{len(pheno_map)} phenotypes -> "
        f"{len(gene_map)} genes + {len(disease_map)} diseases "
        f"(max_hops={max_hops})"
    )

    # Build adjacency once
    logger.info("Building undirected adjacency list...")
    adjacency = build_undirected_adjacency(kg)
    logger.info(f"  {len(adjacency)} nodes have at least one edge")

    # Pre-build set lookups for fast target classification
    gene_set = set(gene_map.keys())
    disease_set = set(disease_map.keys())

    # Output buffers
    src_indices: List[int] = []
    tgt_indices: List[int] = []
    tgt_types: List[int] = []
    distances: List[int] = []

    # BFS from each phenotype
    start_time = time.time()
    n_phenotypes = len(pheno_map)
    for i, (pheno_id, pheno_idx) in enumerate(pheno_map.items()):
        if (i + 1) % 100 == 0 or i == n_phenotypes - 1:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            logger.info(
                f"  Processed {i + 1}/{n_phenotypes} phenotypes "
                f"({rate:.1f}/s, {len(distances):,} pairs collected)"
            )

        reachable = bfs_from_source(adjacency, pheno_id, max_hops)

        for tgt_id, dist in reachable.items():
            if dist == 0:
                continue  # skip self
            if tgt_id in gene_set:
                src_indices.append(pheno_idx)
                tgt_indices.append(gene_map[tgt_id])
                tgt_types.append(TARGET_TYPE_GENE)
                distances.append(dist)
            elif tgt_id in disease_set:
                src_indices.append(pheno_idx)
                tgt_indices.append(disease_map[tgt_id])
                tgt_types.append(TARGET_TYPE_DISEASE)
                distances.append(dist)

    elapsed = time.time() - start_time
    logger.info(
        f"Done. Computed {len(distances):,} (phenotype, target) pairs "
        f"in {elapsed:.1f}s"
    )

    # Convert to tensors
    return {
        "phenotype_idx": torch.tensor(src_indices, dtype=torch.int64),
        "target_idx": torch.tensor(tgt_indices, dtype=torch.int64),
        "target_type": torch.tensor(tgt_types, dtype=torch.int64),
        "distance": torch.tensor(distances, dtype=torch.int8),
    }


def save_shortest_paths(
    sp_data: Dict[str, torch.Tensor],
    output_path: Path,
    metadata: Dict[str, int],
) -> None:
    """Save shortest path tensors and a small JSON metadata sidecar."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sp_data, output_path)

    metadata_path = output_path.with_suffix(".meta.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved {output_path} ({size_mb:.2f} MB)")
    logger.info(f"Saved {metadata_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-compute shortest path lengths for SHEPHERD scoring"
    )
    parser.add_argument(
        "--kg-path", type=Path, required=True,
        help="Path to KG JSON file (produced by KnowledgeGraph.save_json)",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to write shortest_paths.pt (same dir as node_features.pt)",
    )
    parser.add_argument(
        "--max-hops", type=int, default=5,
        help="Maximum BFS depth (default: 5; pairs beyond this are dropped)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.kg_path.exists():
        logger.error(f"KG file not found: {args.kg_path}")
        return 1
    if args.max_hops < 1 or args.max_hops > 127:
        logger.error(f"max_hops must be in [1, 127], got {args.max_hops}")
        return 1

    logger.info(f"Loading KG from {args.kg_path}...")
    kg = KnowledgeGraph.load_json(str(args.kg_path))
    logger.info(f"Loaded: {kg.total_nodes} nodes, {kg.total_edges} edges")

    sp_data = compute_shortest_paths(kg, max_hops=args.max_hops)

    metadata = {
        "max_hops": args.max_hops,
        "num_pairs": int(sp_data["distance"].numel()),
        "num_phenotypes": len(kg.get_node_id_mapping().get(NodeType.PHENOTYPE.value, {})),
        "num_genes": len(kg.get_node_id_mapping().get(NodeType.GENE.value, {})),
        "num_diseases": len(kg.get_node_id_mapping().get(NodeType.DISEASE.value, {})),
        "kg_total_nodes": kg.total_nodes,
        "kg_total_edges": kg.total_edges,
    }

    output_path = args.output_dir / "shortest_paths.pt"
    save_shortest_paths(sp_data, output_path, metadata)

    return 0


if __name__ == "__main__":
    sys.exit(main())
