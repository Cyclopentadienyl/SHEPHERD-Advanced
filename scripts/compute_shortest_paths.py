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
import math
import multiprocessing as mp
import os
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


# ---------------------------------------------------------------------------
# Parallel BFS workers
#
# Each phenotype's BFS is independent, so the work is embarrassingly parallel.
# The GIL makes threads useless for this pure-Python loop, so we use processes.
# On Linux (the Spark target) the default `fork` start method lets workers
# inherit the big read-only adjacency via copy-on-write — no pickling. On
# spawn platforms (Windows/macOS) the data is passed through an initializer.
# Workers do pure-Python BFS only (no torch/CUDA), so forking is safe here.
# ---------------------------------------------------------------------------
_W_ADJ: Dict[str, List[str]] = {}
_W_GENE_MAP: Dict[str, int] = {}
_W_DISEASE_MAP: Dict[str, int] = {}
_W_GENE_SET: set = set()
_W_DISEASE_SET: set = set()
_W_MAX_HOPS: int = 5


def _init_worker_state(adjacency, gene_map, disease_map, max_hops) -> None:
    """Populate module-global read-only state used by _process_chunk."""
    global _W_ADJ, _W_GENE_MAP, _W_DISEASE_MAP, _W_GENE_SET, _W_DISEASE_SET, _W_MAX_HOPS
    _W_ADJ = adjacency
    _W_GENE_MAP = gene_map
    _W_DISEASE_MAP = disease_map
    _W_GENE_SET = set(gene_map.keys())
    _W_DISEASE_SET = set(disease_map.keys())
    _W_MAX_HOPS = max_hops


def _process_chunk(
    chunk: List[Tuple[str, int]],
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """Run BFS for a chunk of (phenotype_id, phenotype_idx) and collect the
    (src, tgt, type, dist) pairs. Reads the module-global worker state."""
    src: List[int] = []
    tgt: List[int] = []
    typ: List[int] = []
    dist: List[int] = []
    adj = _W_ADJ
    gset, dset = _W_GENE_SET, _W_DISEASE_SET
    gmap, dmap = _W_GENE_MAP, _W_DISEASE_MAP
    max_hops = _W_MAX_HOPS
    for pheno_id, pheno_idx in chunk:
        reachable = bfs_from_source(adj, pheno_id, max_hops)
        for tgt_id, d in reachable.items():
            if d == 0:
                continue
            if tgt_id in gset:
                src.append(pheno_idx)
                tgt.append(gmap[tgt_id])
                typ.append(TARGET_TYPE_GENE)
                dist.append(d)
            elif tgt_id in dset:
                src.append(pheno_idx)
                tgt.append(dmap[tgt_id])
                typ.append(TARGET_TYPE_DISEASE)
                dist.append(d)
    return src, tgt, typ, dist


def _resolve_workers(requested: int, n_items: int) -> int:
    """Resolve worker count. requested<=0 => auto (~75% of cores). Small jobs
    stay serial to avoid process-spawn overhead."""
    cpu = os.cpu_count() or 1
    if requested and requested > 0:
        workers = requested
    else:
        workers = max(1, round(cpu * 0.75))
    workers = min(workers, cpu, max(1, n_items))
    if n_items < 1000:
        return 1  # not worth parallelizing
    return workers


def compute_shortest_paths(
    kg: KnowledgeGraph,
    max_hops: int = 5,
    workers: int = 1,
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

    # BFS from each phenotype (serial or parallel across processes)
    pheno_items = list(pheno_map.items())
    n_phenotypes = len(pheno_items)
    n_workers = _resolve_workers(workers, n_phenotypes)
    start_time = time.time()

    if n_workers <= 1:
        logger.info("Running BFS serially (1 worker)")
        for i, (pheno_id, pheno_idx) in enumerate(pheno_items):
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
    else:
        # Over-partition into chunks for load balancing + progress granularity.
        n_chunks = n_workers * 8
        chunk_size = max(1, math.ceil(n_phenotypes / n_chunks))
        chunks = [
            pheno_items[i:i + chunk_size]
            for i in range(0, n_phenotypes, chunk_size)
        ]
        logger.info(
            f"Running BFS on {n_workers} processes "
            f"({len(chunks)} chunks of ~{chunk_size} phenotypes)"
        )

        try:
            ctx = mp.get_context("fork")
            use_fork = True
        except ValueError:
            ctx = mp.get_context()  # spawn (Windows/macOS): pass data explicitly
            use_fork = False

        if use_fork:
            # Children inherit the big adjacency via copy-on-write (no pickling).
            _init_worker_state(adjacency, gene_map, disease_map, max_hops)
            pool = ctx.Pool(n_workers)
        else:
            pool = ctx.Pool(
                n_workers,
                initializer=_init_worker_state,
                initargs=(adjacency, gene_map, disease_map, max_hops),
            )

        try:
            done = 0
            log_every = max(1, len(chunks) // 20)
            for s, t, ty, d in pool.imap_unordered(_process_chunk, chunks):
                src_indices.extend(s)
                tgt_indices.extend(t)
                tgt_types.extend(ty)
                distances.extend(d)
                done += 1
                if done % log_every == 0 or done == len(chunks):
                    elapsed = time.time() - start_time
                    logger.info(
                        f"  {done}/{len(chunks)} chunks done "
                        f"({elapsed:.1f}s, {len(distances):,} pairs collected)"
                    )
        finally:
            pool.close()
            pool.join()

    elapsed = time.time() - start_time
    logger.info(
        f"Done. Computed {len(distances):,} (phenotype, target) pairs "
        f"in {elapsed:.1f}s ({n_workers} worker(s))"
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
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Parallel BFS worker processes. 0 = auto (~75%% of CPU cores); "
             "1 = serial. The BFS is embarrassingly parallel, so on a "
             "many-core box this is a large speedup (default: auto)",
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

    sp_data = compute_shortest_paths(kg, max_hops=args.max_hops, workers=args.workers)

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
