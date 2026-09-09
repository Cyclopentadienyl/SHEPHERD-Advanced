#!/usr/bin/env python3
"""
Setup Demo Data for SHEPHERD-Advanced
======================================
Writes a small demonstration workspace through the production workspace writer,
and optionally trains a small model on it through the production trainer.

**Everything here is the real path at a small size.** This script owns the demo
knowledge graph and nothing else: the allocation, the cohorts, the manifest and
the checkpoint all come from the same code a real build and a real training run
use. It previously carried its own copies — a sample generator that split
patients rather than diseases, and a hand-rolled training loop — and produced a
directory that looked complete, trained, and was then refused by the clinical
verifier because no manifest bound it.

Usage:
    # Workspace only (path-reasoning fallback mode):
    python scripts/setup_demo.py

    # Workspace + a small trained model (full GNN-primary mode):
    python scripts/setup_demo.py --train-model

    # Custom output directory:
    python scripts/setup_demo.py --output-dir data/demo

Output files:
    <output_dir>/kg.json              - Knowledge graph
    <output_dir>/node_features.pt     - Node feature tensors
    <output_dir>/edge_indices.pt      - Edge index tensors
    <output_dir>/num_nodes.json       - Node counts per type
    <output_dir>/train_samples.json   - Training cohort
    <output_dir>/val_samples.json     - Validation cohort (disease-disjoint)
    <output_dir>/split_manifest.json  - Binds all six to one production event
    <output_dir>/checkpoints/gat/     - Checkpoints (if --train-model)

The command to start the API is printed on completion, with the checkpoint the
run actually selected.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.core.types import (
    DataSource,
    Edge,
    EdgeType,
    Node,
    NodeID,
    NodeType,
)
from src.kg.graph import KnowledgeGraph
from src.kg.workspace import SampleBudget, write_workspace

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# The demo graph is 10 phenotypes / 8 genes / 5 diseases, so the budgets only
# have to reach every allocated disease; they are generous rather than tuned.
DEMO_FEATURE_DIM = 64
DEMO_NUM_TRAIN = 200
DEMO_NUM_VAL = 50
DEMO_VAL_DISEASE_FRACTION = 0.2
DEMO_SEED = 42


def _refuse_undersized_budgets(allocation) -> None:
    """Refuse before the first workspace byte, in this script's vocabulary.

    The demo's budgets are constants, so this cannot fire unless the demo graph
    is edited to hold more diseases than samples. It is wired anyway because the
    alternative — relying on the generator's own refusal — happens after the
    graph has been written, and a half-written demo workspace teaches the same
    wrong lesson as a half-written real one.
    """
    for budget, partition, name in (
        (DEMO_NUM_TRAIN, allocation.train, "DEMO_NUM_TRAIN"),
        (DEMO_NUM_VAL, allocation.val, "DEMO_NUM_VAL"),
    ):
        if budget < len(partition):
            raise SystemExit(
                f"{name}={budget} cannot cover {len(partition)} allocated "
                "diseases; every allocated disease must receive at least one "
                "sample. Nothing was written."
            )


# =============================================================================
# Demo KG construction
# =============================================================================
def build_demo_kg() -> KnowledgeGraph:
    """
    Build a small but realistic knowledge graph for demonstration.

    Contains:
    - 10 phenotypes (HPO terms)
    - 8 genes
    - 5 diseases (MONDO terms)
    - Gene-Phenotype, Gene-Disease, Phenotype-Disease edges
    - 1 ortholog example (mouse gene, optional)

    The graph covers several real rare disease patterns so the
    diagnosis pipeline can produce meaningful output.
    """
    kg = KnowledgeGraph()

    # --- Phenotypes (HPO) ---
    phenotypes = [
        ("HP:0001250", "Seizure"),
        ("HP:0001263", "Global developmental delay"),
        ("HP:0002311", "Incoordination"),
        ("HP:0001252", "Hypotonia"),
        ("HP:0000256", "Macrocephaly"),
        ("HP:0001290", "Generalized hypotonia"),
        ("HP:0002069", "Bilateral tonic-clonic seizure"),
        ("HP:0000729", "Autistic behavior"),
        ("HP:0000486", "Strabismus"),
        ("HP:0002360", "Sleep abnormality"),
    ]
    for hpo_id, name in phenotypes:
        kg.add_node(Node(
            id=NodeID(source=DataSource.HPO, local_id=hpo_id),
            node_type=NodeType.PHENOTYPE,
            name=name,
            attributes={"hpo_id": hpo_id, "name": name},
        ))

    # --- Genes ---
    genes = [
        ("SCN1A", "SCN1A"),
        ("SCN2A", "SCN2A"),
        ("CDKL5", "CDKL5"),
        ("MECP2", "MECP2"),
        ("PTEN", "PTEN"),
        ("TSC1", "TSC1"),
        ("TSC2", "TSC2"),
        ("FGFR3", "FGFR3"),
    ]
    for gene_id, name in genes:
        kg.add_node(Node(
            id=NodeID(source=DataSource.DISGENET, local_id=gene_id),
            node_type=NodeType.GENE,
            name=name,
            attributes={"symbol": gene_id, "entrez_id": ""},
        ))

    # --- Diseases (MONDO) ---
    diseases = [
        ("MONDO:0011073", "Dravet syndrome"),
        ("MONDO:0010582", "Rett syndrome"),
        ("MONDO:0010730", "Tuberous sclerosis complex"),
        ("MONDO:0007037", "Achondroplasia"),
        ("MONDO:0012091", "CDKL5 deficiency disorder"),
    ]
    for mondo_id, name in diseases:
        kg.add_node(Node(
            id=NodeID(source=DataSource.MONDO, local_id=mondo_id),
            node_type=NodeType.DISEASE,
            name=name,
            attributes={"mondo_id": mondo_id, "name": name},
        ))

    # --- Gene -> Phenotype edges ---
    gene_pheno_edges = [
        # SCN1A: Dravet syndrome phenotypes
        ("SCN1A", "HP:0001250", 0.95),  # Seizure
        ("SCN1A", "HP:0001263", 0.80),  # Developmental delay
        ("SCN1A", "HP:0002069", 0.90),  # Tonic-clonic seizure
        ("SCN1A", "HP:0001252", 0.70),  # Hypotonia
        # SCN2A: Overlapping phenotypes
        ("SCN2A", "HP:0001250", 0.85),
        ("SCN2A", "HP:0001263", 0.75),
        # CDKL5: CDKL5 deficiency
        ("CDKL5", "HP:0001250", 0.90),
        ("CDKL5", "HP:0001263", 0.85),
        ("CDKL5", "HP:0000486", 0.60),  # Strabismus
        # MECP2: Rett syndrome
        ("MECP2", "HP:0001263", 0.90),
        ("MECP2", "HP:0001252", 0.85),
        ("MECP2", "HP:0000729", 0.70),  # Autistic behavior
        ("MECP2", "HP:0002360", 0.65),  # Sleep abnormality
        # PTEN: Macrocephaly
        ("PTEN", "HP:0000256", 0.90),
        ("PTEN", "HP:0001263", 0.60),
        # TSC1/TSC2: Tuberous sclerosis
        ("TSC1", "HP:0001250", 0.85),
        ("TSC1", "HP:0001263", 0.70),
        ("TSC2", "HP:0001250", 0.88),
        ("TSC2", "HP:0001263", 0.72),
        # FGFR3: Achondroplasia
        ("FGFR3", "HP:0000256", 0.75),
    ]
    for gene_id, pheno_id, weight in gene_pheno_edges:
        kg.add_edge(Edge(
            source_id=NodeID(source=DataSource.DISGENET, local_id=gene_id),
            target_id=NodeID(source=DataSource.HPO, local_id=pheno_id),
            edge_type=EdgeType.GENE_HAS_PHENOTYPE,
            weight=weight,
        ))

    # --- Gene -> Disease edges ---
    gene_disease_edges = [
        ("SCN1A", "MONDO:0011073", 0.95),  # Dravet
        ("SCN2A", "MONDO:0011073", 0.50),  # SCN2A also linked weakly to Dravet
        ("CDKL5", "MONDO:0012091", 0.95),  # CDKL5 deficiency
        ("MECP2", "MONDO:0010582", 0.95),  # Rett
        ("TSC1", "MONDO:0010730", 0.90),   # TSC
        ("TSC2", "MONDO:0010730", 0.90),   # TSC
        ("FGFR3", "MONDO:0007037", 0.95),  # Achondroplasia
        ("PTEN", "MONDO:0010730", 0.40),   # PTEN weakly linked to TSC phenotypes
    ]
    for gene_id, mondo_id, weight in gene_disease_edges:
        kg.add_edge(Edge(
            source_id=NodeID(source=DataSource.DISGENET, local_id=gene_id),
            target_id=NodeID(source=DataSource.MONDO, local_id=mondo_id),
            edge_type=EdgeType.GENE_ASSOCIATED_WITH_DISEASE,
            weight=weight,
        ))

    # --- Phenotype -> Disease (direct) edges ---
    pheno_disease_edges = [
        ("HP:0001250", "MONDO:0011073", 0.80),  # Seizure -> Dravet
        ("HP:0001263", "MONDO:0010582", 0.70),  # Dev delay -> Rett
        ("HP:0001263", "MONDO:0012091", 0.70),  # Dev delay -> CDKL5 deficiency
    ]
    for pheno_id, mondo_id, weight in pheno_disease_edges:
        kg.add_edge(Edge(
            source_id=NodeID(source=DataSource.HPO, local_id=pheno_id),
            target_id=NodeID(source=DataSource.MONDO, local_id=mondo_id),
            edge_type=EdgeType.PHENOTYPE_OF_DISEASE,
            weight=weight,
        ))

    # --- Optional: Mouse ortholog example (demonstrates P1 feature) ---
    kg.add_node(Node(
        id=NodeID(source=DataSource.MGI, local_id="Scn1a"),
        node_type=NodeType.MOUSE_GENE,
        name="Scn1a (mouse)",
        attributes={"mgi_id": "Scn1a", "symbol": "Scn1a"},
    ))
    kg.add_edge(Edge(
        source_id=NodeID(source=DataSource.DISGENET, local_id="SCN1A"),
        target_id=NodeID(source=DataSource.MGI, local_id="Scn1a"),
        edge_type=EdgeType.HUMAN_MOUSE_ORTHOLOG,
        weight=0.95,
    ))

    stats = kg.get_statistics()
    logger.info(f"Demo KG built: {stats}")
    return kg


# =============================================================================
# Model training
# =============================================================================
def train_demo_model(workspace: Path, num_epochs: int) -> Path:
    """Train a small model on the demo workspace **through the real trainer**.

    This script used to carry its own: it read the exported tensors, built a
    GAT by hand, ran a loop and saved a checkpoint of its own shape. That
    checkpoint never passed the training preflight, so nothing ever checked that
    the workspace it came from was internally consistent — and the workspace it
    came from was not, because the samples beside it were split by patient.

    Calling `train` costs a few more seconds and buys the thing the demo is for:
    if this returns a checkpoint, the production path works end to end on this
    machine. The configuration is small because the demo graph is small; it is
    not a different pipeline.

    Returns:
        The checkpoint the API should serve.
    """
    from scripts.train_model import TrainConfig, train
    from src.utils.checkpoint_paths import resolve_checkpoint_dir, select_checkpoint_in_dir

    config = TrainConfig(
        data_dir=str(workspace),
        output_dir=str(workspace / "outputs"),
        log_dir=str(workspace / "logs"),
        conv_type="gat",
        hidden_dim=DEMO_FEATURE_DIM,
        num_layers=2,
        num_heads=4,
        num_epochs=num_epochs,
        batch_size=8,
        warmup_steps=0,
    )
    if not train(config):
        raise SystemExit(
            "Demo training produced no metrics; see the log above. The workspace "
            "itself was written and is valid — re-run training with "
            f"scripts/train_model.py --data-dir {workspace}"
        )

    ckpt_dir = resolve_checkpoint_dir(str(workspace), config.conv_type, None)
    selected = select_checkpoint_in_dir(ckpt_dir)
    if selected is None:
        raise SystemExit(f"Training wrote no checkpoint under {ckpt_dir}.")
    logger.info("Demo checkpoint: %s", selected)
    return selected


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Setup demo data for SHEPHERD-Advanced"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/workspaces/demo",
        help="Output directory (default: data/workspaces/demo)",
    )
    parser.add_argument(
        "--train-model",
        action="store_true",
        help="Also train a small model, so the API can start in GNN mode",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Training epochs when --train-model is given (default: 20)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    logger.info("Building demo knowledge graph...")
    kg = build_demo_kg()

    # **The same writer the production build uses.** Not a demo-shaped
    # imitation of one: the allocation is cut at the disease level, the cohorts
    # are generated from it, and `split_manifest.json` binds all six artifacts
    # to this one event. A workspace this script writes is a workspace every
    # consumer accepts — which is the only useful thing a demo of this system
    # can demonstrate.
    written = write_workspace(
        kg,
        output_dir,
        feature_dim=DEMO_FEATURE_DIM,
        samples=SampleBudget(
            num_train=DEMO_NUM_TRAIN,
            num_val=DEMO_NUM_VAL,
            val_disease_fraction=DEMO_VAL_DISEASE_FRACTION,
            seed=DEMO_SEED,
        ),
        preflight=_refuse_undersized_budgets,
    )
    kg_path = output_dir / "kg.json"

    checkpoint_path = None
    if args.train_model:
        checkpoint_path = train_demo_model(output_dir, args.epochs)

    print("\n" + "=" * 60)
    print("Demo setup complete!")
    print("=" * 60)
    print(f"  Workspace: {output_dir}")
    print(f"  Train: {len(written.train_samples)} samples over "
          f"{written.manifest['realised']['train_diseases']} diseases")
    print(f"  Val:   {len(written.val_samples)} samples over "
          f"{written.manifest['realised']['val_diseases']} diseases "
          f"(disjoint: {written.manifest['disjoint']})")

    if checkpoint_path is not None:
        print(f"""
Start the API with GNN scoring (full mode):

  SHEPHERD_KG_PATH={kg_path} \\
  SHEPHERD_CHECKPOINT_PATH={checkpoint_path} \\
  SHEPHERD_DATA_DIR={output_dir} \\
  python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
""")
    else:
        print(f"""
Start the API with path-reasoning only (fallback mode):

  SHEPHERD_KG_PATH={kg_path} \\
  python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload

To enable GNN scoring, train on this workspace:
  python scripts/setup_demo.py --train-model
  # or, equivalently, the production entry point:
  python scripts/train_model.py --data-dir {output_dir} --epochs 20
""")


if __name__ == "__main__":
    main()
