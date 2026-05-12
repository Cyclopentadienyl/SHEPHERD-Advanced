#!/usr/bin/env python3
"""
Build Knowledge Graph for SHEPHERD-Advanced
=============================================
Constructs a production-scale knowledge graph from HPO annotation files
and ontologies (HPO, MONDO).

Prerequisites:
    python scripts/download_data.py --output-dir data/external/

Usage:
    # Build KG with training samples
    python scripts/build_knowledge_graph.py \\
        --workspace data/workspaces/hpo_2026/ \\
        --external-dir data/external/ \\
        --generate-samples --num-train 5000 --num-val 1000

    # Build KG only (no training samples)
    python scripts/build_knowledge_graph.py \\
        --workspace data/workspaces/hpo_2026/ \\
        --external-dir data/external/

Output files:
    <workspace>/kg.json             - Knowledge graph (JSON)
    <workspace>/node_features.pt    - Node feature tensors
    <workspace>/edge_indices.pt     - Edge index tensors
    <workspace>/num_nodes.json      - Node counts per type
    <workspace>/train_samples.json  - Training samples (if --generate-samples)
    <workspace>/val_samples.json    - Validation samples (if --generate-samples)
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.core.types import DataSource, NodeType
from src.kg.builder import KnowledgeGraphBuilder, KGBuilderConfig
from src.data_sources.hpo_annotations import HPOAnnotationParser
from src.ontology.loader import OntologyLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_knowledge_graph(
    external_dir: Path,
    workspace: Path,
    feature_dim: int = 128,
    generate_samples: bool = False,
    num_train: int = 5000,
    num_val: int = 1000,
    ontology_cache_dir: Path | None = None,
) -> None:
    """
    Build a production knowledge graph from HPO annotation files.

    Order matters:
      1. MONDO ontology (disease nodes)
      2. HPO ontology (phenotype nodes)
      3. phenotype.hpoa (phenotype-disease edges)
      4. genes_to_phenotype.txt (gene nodes + gene-phenotype/gene-disease edges)
    """
    workspace.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # --- Load ontologies ---
    logger.info("Loading ontologies...")
    ont_loader = OntologyLoader(cache_dir=ontology_cache_dir)
    mondo = ont_loader.load_mondo()
    hpo = ont_loader.load_hpo()

    # --- Build KG ---
    config = KGBuilderConfig(
        include_ontology_hierarchy=True,
        include_orthologs=False,
        include_literature=False,
    )
    builder = KnowledgeGraphBuilder(config=config)

    # Step 1: Add MONDO disease nodes (must be first so disease nodes exist for edges)
    logger.info("Adding MONDO disease nodes...")
    n_diseases = builder.add_ontology(mondo, NodeType.DISEASE)
    logger.info(f"  -> {n_diseases} disease nodes")

    # Step 2: Add HPO phenotype nodes
    logger.info("Adding HPO phenotype nodes...")
    n_phenos = builder.add_ontology(hpo, NodeType.PHENOTYPE)
    logger.info(f"  -> {n_phenos} phenotype nodes")

    # --- Parse annotation files ---
    parser = HPOAnnotationParser(mondo_ontology=mondo)

    # Step 3: Phenotype-disease annotations
    hpoa_path = external_dir / "phenotype.hpoa"
    if hpoa_path.exists():
        logger.info("Adding phenotype-disease annotations...")
        pheno_disease = parser.parse_phenotype_hpoa(hpoa_path)
        n_pd_edges = builder.add_phenotype_disease_annotations(pheno_disease)
        logger.info(f"  -> {n_pd_edges} phenotype-disease edges")
    else:
        logger.warning(f"phenotype.hpoa not found at {hpoa_path}, skipping")

    # Step 4: Gene-phenotype and gene-disease associations
    g2p_path = external_dir / "genes_to_phenotype.txt"
    if g2p_path.exists():
        logger.info("Adding gene associations...")
        gene_pheno, gene_disease = parser.parse_genes_to_phenotype(g2p_path)

        n_genes, n_gd_edges = builder.add_gene_disease_associations(
            gene_disease, source=DataSource.HPO
        )
        logger.info(f"  -> {n_genes} gene nodes, {n_gd_edges} gene-disease edges")

        n_gp_edges = builder.add_gene_phenotype_associations(
            gene_pheno, source=DataSource.HPO
        )
        logger.info(f"  -> {n_gp_edges} gene-phenotype edges")
    else:
        logger.warning(f"genes_to_phenotype.txt not found at {g2p_path}, skipping")

    # --- Finalize KG ---
    kg = builder.build()
    stats = kg.get_statistics()

    # Save KG
    kg_path = workspace / "kg.json"
    kg.save_json(str(kg_path))
    logger.info(f"KG saved to {kg_path}")

    # Export PyG graph data
    logger.info(f"Exporting graph data (feature_dim={feature_dim})...")
    kg.export_graph_data(output_dir=workspace, feature_dim=feature_dim)

    # --- Generate training samples ---
    if generate_samples:
        logger.info("Generating training samples...")
        from src.kg.sample_generator import generate_training_samples

        train_samples, val_samples = generate_training_samples(
            kg=kg,
            num_train=num_train,
            num_val=num_val,
            output_dir=workspace,
        )
        logger.info(
            f"Generated {len(train_samples)} train, {len(val_samples)} val samples"
        )

    elapsed = time.time() - t0

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Knowledge Graph Construction Complete")
    print("=" * 60)
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Total nodes: {stats['total_nodes']:,}")
    print(f"  Total edges: {stats['total_edges']:,}")
    print(f"  Nodes by type:")
    for nt, count in sorted(stats["nodes_by_type"].items()):
        print(f"    {nt}: {count:,}")
    print(f"  Edges by type:")
    for et, count in sorted(stats["edges_by_type"].items()):
        print(f"    {et}: {count:,}")
    print(f"  Workspace: {workspace}")

    if generate_samples:
        print(f"\n  Training samples: {len(train_samples)}")
        print(f"  Validation samples: {len(val_samples)}")

    print(f"\nNext steps:")
    print(f"  # Precompute shortest paths (may take 30-60 min for large KGs)")
    print(f"  python scripts/compute_shortest_paths.py \\")
    print(f"      --kg-path {kg_path} --output-dir {workspace}")
    print(f"\n  # Train model")
    print(f"  python scripts/train_model.py --data-dir {workspace} --epochs 50")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Build a production knowledge graph for SHEPHERD-Advanced"
    )
    parser.add_argument(
        "--workspace",
        type=str,
        required=True,
        help="Workspace directory for all output files",
    )
    parser.add_argument(
        "--external-dir",
        type=str,
        default="data/external",
        help="Directory containing downloaded annotation files (default: data/external/)",
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=128,
        help="Dimensionality of node feature vectors (default: 128)",
    )
    parser.add_argument(
        "--ontology-cache-dir",
        type=str,
        default=None,
        help="Directory for cached ontology files (default: ~/.shepherd/ontologies/)",
    )
    parser.add_argument(
        "--generate-samples",
        action="store_true",
        help="Generate training/validation samples from the KG",
    )
    parser.add_argument(
        "--num-train",
        type=int,
        default=5000,
        help="Number of training samples (default: 5000)",
    )
    parser.add_argument(
        "--num-val",
        type=int,
        default=1000,
        help="Number of validation samples (default: 1000)",
    )
    args = parser.parse_args()

    build_knowledge_graph(
        external_dir=Path(args.external_dir),
        workspace=Path(args.workspace),
        feature_dim=args.feature_dim,
        generate_samples=args.generate_samples,
        num_train=args.num_train,
        num_val=args.num_val,
        ontology_cache_dir=Path(args.ontology_cache_dir) if args.ontology_cache_dir else None,
    )


if __name__ == "__main__":
    main()
