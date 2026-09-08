#!/usr/bin/env python3
"""
Build Knowledge Graph for SHEPHERD-Advanced
=============================================
Constructs a production-scale knowledge graph from locally provided
annotation files and ontologies (HPO, MONDO).

Before running, manually download the annotation files into data/external/:
    See data/external/README.md for download links and instructions.

Usage:
    # Build KG with training samples
    python scripts/build_knowledge_graph.py \\
        --workspace data/workspaces/hpo_2026/ \\
        --external-dir data/external/ \\
        --generate-samples --num-train 100000 --num-val 15000

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
from typing import Any, Optional

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.core.types import DataSource, NodeType
from src.kg.builder import KnowledgeGraphBuilder, KGBuilderConfig
from src.data_sources.hpo_annotations import HPOAnnotationParser
from src.ontology.loader import OntologyLoader
from src.utils.fingerprint import file_sha256

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def require_explicit_budgets(
    num_train: Optional[int],
    num_val: Optional[int],
    allocation: Any,
    val_disease_fraction: float,
) -> None:
    """No default sample budgets, and the message carries the real minimums.

    **A default that cannot succeed is worse than no default.** The full-coverage
    contract requires every allocated disease to receive at least one sample, and
    the audited universe allocates roughly 8,990 training and 1,586 validation
    diseases at f = 0.15 — so the former 5,000 / 1,000 could never work on a real
    workspace, and would have failed only after the graph was already built.

    Named rather than inlined so it can be tested. Inline it was unfalsifiable:
    removing the check broke nothing.
    """
    if num_train is not None and num_val is not None:
        return
    raise SystemExit(
        "--num-train and --num-val are required with --generate-samples. This "
        f"workspace allocates {len(allocation.train)} training and "
        f"{len(allocation.val)} validation diseases at --val-disease-fraction "
        f"{val_disease_fraction}, and every allocated disease must receive at "
        "least one sample, so those are the minimums."
    )


def require_sufficient_budgets(
    num_train: int, num_val: int, allocation: Any
) -> None:
    """Budgets must cover their partitions, checked before anything is written.

    ``_generate_partition`` refuses an under-sized budget too, but by then the
    graph artifacts have been saved. Full coverage needs one sample per allocated
    disease, and that requirement is knowable from the allocation alone — so it
    is knowable before the workspace is touched.
    """
    for budget, partition, flag in (
        (num_train, allocation.train, "--num-train"),
        (num_val, allocation.val, "--num-val"),
    ):
        if budget < len(partition):
            raise SystemExit(
                f"{flag}={budget} cannot cover {len(partition)} allocated "
                "diseases; every allocated disease must receive at least one "
                "sample. Nothing was written."
            )


def build_knowledge_graph(
    external_dir: Path,
    workspace: Path,
    feature_dim: int = 128,
    generate_samples: bool = False,
    num_train: Optional[int] = None,
    num_val: Optional[int] = None,
    val_disease_fraction: float = 0.15,
    sample_seed: int = 42,
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

    # **Refuse before the first byte, not before the last.** The generator
    # carries the same check, but by the time it runs kg.json and the graph
    # tensors have already been overwritten — leaving old checkpoints paired with
    # a new graph and only then refusing to regenerate samples. A workspace half
    # rebuilt under trained checkpoints is worse than one not rebuilt at all.
    from src.kg.sample_generator import refuse_if_checkpoints_exist

    refuse_if_checkpoints_exist(Path(workspace))

    # --- Validate annotation files exist (fail fast before expensive ontology loading) ---
    required_files = {
        "phenotype.hpoa": "https://hpo.jax.org/data/annotations -> phenotype.hpoa",
        "genes_to_phenotype.txt": "https://hpo.jax.org/data/annotations -> genes_to_phenotype.txt",
    }
    missing = [
        (name, hint)
        for name, hint in required_files.items()
        if not (external_dir / name).exists()
    ]
    if missing:
        print("\nERROR: Required annotation files not found.", file=sys.stderr)
        print(f"Expected location: {external_dir}/\n", file=sys.stderr)
        for name, hint in missing:
            print(f"  Missing: {name}", file=sys.stderr)
            print(f"    Download from: {hint}", file=sys.stderr)
        print(f"\nSee data/external/README.md for detailed instructions.", file=sys.stderr)
        sys.exit(1)

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
    logger.info("Adding phenotype-disease annotations...")
    pheno_disease = parser.parse_phenotype_hpoa(hpoa_path)
    n_pd_edges = builder.add_phenotype_disease_annotations(pheno_disease)
    logger.info(f"  -> {n_pd_edges} phenotype-disease edges")

    # Step 4: Gene-phenotype and gene-disease associations
    g2p_path = external_dir / "genes_to_phenotype.txt"
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

    # --- Finalize KG, in memory ---
    kg = builder.build()
    stats = kg.get_statistics()

    # **Everything that can refuse, refuses before the first workspace byte.**
    # The allocation and both budget checks need only the in-memory graph, so
    # they run here rather than after `kg.save_json`. The earlier ordering wrote
    # the graph, then discovered the budgets were missing or too small — leaving
    # a rebuilt graph beside stale samples, which is the same class of half-
    # written workspace the checkpoint preflight exists to prevent.
    allocation = None
    if generate_samples:
        from src.kg import allocate_diseases, build_eligible_disease_profiles

        # The split is decided here, before generation, and at the disease
        # level. The generator consumes the allocation; it does not own split
        # policy (EVALUATION_COHORTS §6.2). A sample-level slice, which is what
        # this used to be, leaves every multi-sample disease on both sides.
        eligible = build_eligible_disease_profiles(kg, min_phenotypes=2)
        allocation = allocate_diseases(eligible, val_disease_fraction, seed=sample_seed)
        require_explicit_budgets(
            num_train, num_val, allocation, val_disease_fraction
        )
        require_sufficient_budgets(num_train, num_val, allocation)

    # Save KG
    kg_path = workspace / "kg.json"
    kg.save_json(str(kg_path))
    logger.info(f"KG saved to {kg_path}")

    # **The digest of the artifact this function just wrote, from this graph.**
    # Only the writer can vouch for that binding; generation hashing kg.json on
    # its own would happily digest a file some other graph produced and record it
    # beside this allocation's universe digest as one provenance chain.
    kg_digest = file_sha256(kg_path)

    # Export PyG graph data
    logger.info(f"Exporting graph data (feature_dim={feature_dim})...")
    kg.export_graph_data(output_dir=workspace, feature_dim=feature_dim)

    # --- Generate training samples ---
    if generate_samples:
        logger.info("Generating training samples...")
        from src.kg.sample_generator import generate_training_samples

        train_samples, val_samples, manifest = generate_training_samples(
            kg=kg,
            allocation=allocation,
            num_train=num_train,
            num_val=num_val,
            output_dir=workspace,
            kg_digest=kg_digest,
        )
        logger.info(
            "Generated %d train samples over %d diseases, %d val over %d — "
            "disjoint: %s",
            len(train_samples), manifest["realised"]["train_diseases"],
            len(val_samples), manifest["realised"]["val_diseases"],
            manifest["disjoint"],
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
        "--num-train", type=int, default=None,
        help="Training sample budget. Required with --generate-samples: it must "
             "be at least the number of diseases allocated to training, and that "
             "count depends on the workspace. There is no default, because any "
             "fixed one would fail on a real disease universe.",
    )
    parser.add_argument(
        "--val-disease-fraction",
        type=float, default=0.15,
        help="Fraction of eligible diseases withheld from training and used for "
             "validation. 0.15 is the upstream value (EVALUATION_COHORTS 1.6). "
             "The split is at the DISEASE level: a withheld disease keeps its "
             "knowledge-graph node and edges and loses only its labelled patient "
             "examples.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int, default=42,
        help="Root seed for allocation and generation. Allocation, training and "
             "validation draw from independent derived streams, so changing a "
             "sample budget cannot move the disease cut.",
    )
    parser.add_argument(
        "--num-val", type=int, default=None,
        help="Validation sample budget. Required with --generate-samples, and at "
             "least the number of diseases allocated to validation. No default, "
             "for the same reason as --num-train.",
    )
    args = parser.parse_args()

    build_knowledge_graph(
        external_dir=Path(args.external_dir),
        workspace=Path(args.workspace),
        feature_dim=args.feature_dim,
        generate_samples=args.generate_samples,
        num_train=args.num_train,
        num_val=args.num_val,
        val_disease_fraction=args.val_disease_fraction,
        sample_seed=args.sample_seed,
        ontology_cache_dir=Path(args.ontology_cache_dir) if args.ontology_cache_dir else None,
    )


if __name__ == "__main__":
    main()
