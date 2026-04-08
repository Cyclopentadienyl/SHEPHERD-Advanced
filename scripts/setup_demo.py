#!/usr/bin/env python3
"""
Setup Demo Data for SHEPHERD-Advanced
======================================
Generates a small demonstration knowledge graph and optionally trains
a minimal GNN model so the API can start end-to-end.

Usage:
    # KG only (path-reasoning fallback mode):
    python scripts/setup_demo.py

    # KG + minimal GNN model (full GNN-primary mode):
    python scripts/setup_demo.py --train-model

    # Custom output directory:
    python scripts/setup_demo.py --output-dir data/demo

Output files:
    <output_dir>/kg.json              - Knowledge graph
    <output_dir>/node_features.pt     - Node feature tensors (if --train-model)
    <output_dir>/edge_indices.pt      - Edge index tensors   (if --train-model)
    <output_dir>/num_nodes.json       - Node counts           (if --train-model)
    <output_dir>/model_checkpoint.pt  - Trained GNN checkpoint (if --train-model)

After running, start the API with:
    SHEPHERD_KG_PATH=data/demo/kg.json \\
    SHEPHERD_CHECKPOINT_PATH=data/demo/model_checkpoint.pt \\
    SHEPHERD_DATA_DIR=data/demo \\
    python -m uvicorn src.api.main:app --reload
"""
from __future__ import annotations

import argparse
import json
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
# Graph data generation (for GNN)
# =============================================================================
def generate_graph_data(kg: KnowledgeGraph, output_dir: Path) -> None:
    """
    Generate PyG-compatible graph data files from the KG.

    Delegates to KnowledgeGraph.export_graph_data() which is the single
    source of truth for graph-data generation. It automatically adds
    reverse `rev_*` edges so every node type participates as a destination
    in message passing — without this, node types that only appear as
    source (e.g. `gene` in the demo KG) would never have their embeddings
    updated during training.

    Creates node_features.pt, edge_indices.pt, num_nodes.json in output_dir.
    """
    kg.export_graph_data(output_dir=output_dir, feature_dim=64)
    logger.info(f"Graph data saved to {output_dir}")


# =============================================================================
# Minimal model training
# =============================================================================
def train_minimal_model(
    kg: KnowledgeGraph,
    output_dir: Path,
    hidden_dim: int = 64,
    num_epochs: int = 20,
) -> None:
    """
    Train a minimal GNN model on the demo KG for demonstration purposes.

    This produces a valid checkpoint that the pipeline can load.
    """
    import torch
    from src.models.gnn.shepherd_gnn import ShepherdGNN, ShepherdGNNConfig

    node_features = torch.load(output_dir / "node_features.pt", weights_only=True)
    edge_indices = torch.load(output_dir / "edge_indices.pt", weights_only=True)

    # CRITICAL: metadata MUST be derived from the graph_data's edge keys
    # (which include rev_* reverse edges added by export_graph_data), not
    # from kg.metadata() (which only has forward edges). Using kg.metadata()
    # here would build a model with only forward-edge conv layers, and PyG's
    # HeteroConv would silently ignore the reverse edges during forward
    # pass — leaving any "source-only" node type (e.g. `gene` in this demo)
    # with its initial random embeddings forever.
    #
    # This matches the pattern in:
    #   - scripts/train_model.py:create_model_from_config
    #   - src/inference/pipeline.py:_load_model_from_checkpoint
    node_types = list(node_features.keys())
    edge_types = list(edge_indices.keys())
    metadata = (node_types, edge_types)

    in_channels_dict = {nt: feat.size(-1) for nt, feat in node_features.items()}

    config = ShepherdGNNConfig(
        hidden_dim=hidden_dim,
        num_layers=2,
        num_heads=4,
        conv_type="gat",
        dropout=0.1,
        use_positional_encoding=False,
        use_ortholog_gate=False,
    )

    model = ShepherdGNN(
        metadata=metadata,
        in_channels_dict=in_channels_dict,
        config=config,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    logger.info(f"Training minimal GNN model for {num_epochs} epochs...")
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(node_features, edge_indices)

        # Simple self-supervised loss: pull similar node embeddings together
        loss = torch.tensor(0.0)
        for nt, emb in out.items():
            if emb.size(0) > 1:
                # Encourage embeddings to be diverse (not all the same)
                norms = torch.nn.functional.normalize(emb, dim=-1)
                sim = torch.mm(norms, norms.t())
                # Target: identity-like (each node distinct)
                target = torch.eye(sim.size(0))
                loss = loss + torch.nn.functional.mse_loss(sim, target)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            logger.info(f"  Epoch {epoch + 1}/{num_epochs}, loss={loss.item():.4f}")

    model.eval()

    # Save checkpoint (Trainer format)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "hidden_dim": config.hidden_dim,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "conv_type": config.conv_type,
            "use_positional_encoding": config.use_positional_encoding,
            "use_ortholog_gate": config.use_ortholog_gate,
        },
    }
    ckpt_path = output_dir / "model_checkpoint.pt"
    torch.save(checkpoint, ckpt_path)
    logger.info(f"Model checkpoint saved to {ckpt_path}")


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
        default="data/demo",
        help="Output directory (default: data/demo)",
    )
    parser.add_argument(
        "--train-model",
        action="store_true",
        help="Also train a minimal GNN model",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Build and save KG
    logger.info("Building demo knowledge graph...")
    kg = build_demo_kg()
    kg_path = output_dir / "kg.json"
    kg.save_json(str(kg_path))

    if args.train_model:
        # Step 2: Generate graph data
        logger.info("Generating graph data for GNN...")
        generate_graph_data(kg, output_dir)

        # Step 3: Train minimal model
        train_minimal_model(kg, output_dir)

    # Print startup instructions
    print("\n" + "=" * 60)
    print("Demo setup complete!")
    print("=" * 60)

    if args.train_model:
        print(f"""
Start the API with GNN scoring (full mode):

  SHEPHERD_KG_PATH={kg_path} \\
  SHEPHERD_CHECKPOINT_PATH={output_dir / 'model_checkpoint.pt'} \\
  SHEPHERD_DATA_DIR={output_dir} \\
  python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
""")
    else:
        print(f"""
Start the API with path-reasoning only (fallback mode):

  SHEPHERD_KG_PATH={kg_path} \\
  python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

To enable GNN scoring, re-run with --train-model:
  python scripts/setup_demo.py --train-model
""")


if __name__ == "__main__":
    main()
