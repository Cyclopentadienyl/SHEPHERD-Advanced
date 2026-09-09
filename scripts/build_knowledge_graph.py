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
from src.kg.workspace import (
    SampleBudget,
    require_budget_coverage,
    write_workspace,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def require_usable_budgets(num_train: Optional[int], num_val: Optional[int]) -> None:
    """Preflight phase one: are these budgets numbers at all?

    Runs before an ontology is opened, because it needs nothing but the two
    arguments. **A default that cannot succeed is worse than no default** — full
    coverage requires one sample per allocated disease, and the audited universe
    allocates roughly 8,990 training and 1,586 validation diseases at f = 0.15,
    so the former 5,000 / 1,000 could never work on a real workspace.

    The domain rules live in ``validate_sample_budgets``, shared with the
    generator, so the entry point and the library cannot disagree about what a
    budget is. This adds only what the library must not know: the flag names, and
    the exit convention of a command-line tool.
    """
    from src.kg.sample_generator import validate_sample_budgets

    try:
        validate_sample_budgets(num_train, num_val)
    except ValueError as exc:
        raise SystemExit(
            f"{exc}. --num-train and --num-val must both be supplied with "
            "--generate-samples, as non-negative integers; their minimums are "
            "the allocated disease counts, which this build reports once the "
            "graph exists. Nothing was read or written."
        ) from exc


def require_sufficient_budgets(
    num_train: int, num_val: int, allocation: Any
) -> None:
    """Preflight phase two: are these budgets large enough?

    Split from phase one because this is the half that *cannot* run early: the
    partition sizes exist only once the graph has been built and cut. It still
    runs before the first workspace write.

    **The comparison belongs to the writer; this adds the flag names and the
    exit convention of a command-line tool.** The writer enforces it too, and
    would refuse this run on its own — but in library terms, naming an argument
    an operator never typed.
    """
    try:
        require_budget_coverage(
            num_train, num_val, allocation,
            train_label="--num-train", val_label="--num-val",
        )
    except ValueError as exc:
        raise SystemExit(f"{exc} Nothing was written.") from exc


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

    # **Phase one of the budget preflight, before anything is read.** A budget's
    # own domain — supplied, integral, non-negative — needs no ontology, no
    # parser and no graph, so refusing here costs the operator seconds instead of
    # the minutes an ontology load and KG build take. Phase two, which needs the
    # allocation, runs below.
    if generate_samples:
        require_usable_budgets(num_train, num_val)

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

    # **One writer produces a workspace, and this is a call to it.** Allocate,
    # refuse, write, digest, generate — that ordering is what binds the six
    # artifacts into one production event, and it lives in `src.kg.workspace`
    # so this script is not the only place that knows it. `preflight` is where
    # a refusal decidable from the allocation alone lands: before the first
    # workspace byte, in this tool's own vocabulary. The superseded ordering
    # wrote the graph and only then discovered the budgets were too small,
    # leaving a rebuilt graph beside stale samples.
    written = write_workspace(
        kg,
        workspace,
        feature_dim=feature_dim,
        samples=(
            SampleBudget(
                num_train=num_train,
                num_val=num_val,
                val_disease_fraction=val_disease_fraction,
                seed=sample_seed,
            )
            if generate_samples
            else None
        ),
        preflight=(
            (lambda alloc: require_sufficient_budgets(num_train, num_val, alloc))
            if generate_samples
            else None
        ),
    )
    kg_path = workspace / "kg.json"
    train_samples, val_samples = written.train_samples, written.val_samples

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

    # **Name what was produced, and print only the steps that work on it.**
    # Without cohorts this is a graph export, not a trainable workspace: nothing
    # records which export the tensors are, so `train_model.py` and GNN serving
    # both refuse it. Printing the training command anyway is how an operator
    # learns that at the end of a 30-minute build instead of at the start.
    if generate_samples:
        print(f"\n  Training samples: {len(train_samples)}")
        print(f"  Validation samples: {len(val_samples)}")
    else:
        print("\n  Cohorts: none (graph export only, --generate-samples not given)")
        print("  This workspace CANNOT be trained on or served with GNN scoring:")
        print("  no split manifest binds these tensors to this export.")

    print("\nNext steps:")
    print("  # Precompute shortest paths (may take 30-60 min for large KGs)")
    print("  python scripts/compute_shortest_paths.py \\")
    print(f"      --kg-path {kg_path} --output-dir {workspace}")
    if generate_samples:
        print("\n  # Train model")
        print(f"  python scripts/train_model.py --data-dir {workspace} --epochs 50")
    else:
        print("\n  # Complete the workspace before training. Cohorts cannot be")
        print("  # added to an export afterwards -- only the writer that exported")
        print("  # the tensors can vouch for their digests -- so this rebuilds:")
        print(f"  python scripts/build_knowledge_graph.py --workspace {workspace} \\")
        print(f"      --external-dir {external_dir} --generate-samples \\")
        print("      --num-train <n> --num-val <n>")
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
