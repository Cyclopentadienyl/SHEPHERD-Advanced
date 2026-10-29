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
    # Workspace only (nothing to serve until a model is trained on it):
    python scripts/setup_demo.py

    # Workspace + a small base and its disease model, which the printed
    # command serves:
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
    <output_dir>/runs/gatv2/<..._demo-base_...>/     - The base run (if --train-model)
    <output_dir>/runs/gatv2/<..._demo-disease_...>/  - Its disease child (if --train-model)

**`--train-model` trains the two stages a task model is made of** (K7, TM3):
a base (KG link prediction) and one disease child fine-tuned from it, each by
the production trainer (`scripts/train_model.py`'s `train`, the model bundle
path), at the stage's reference recipe with the demo graph's sizes
(`demo_task_training`, at which the deployment probe trains its demo disease
model too). It used to train a ShepherdGNN of its own configuration, which
nothing serves any more.

With `--train-model`, the command to start the API is printed on completion,
with the disease child's checkpoint (its run's `last.pt`) and the device it was
trained on: serving refuses `auto` without CUDA (contract M3c), so the command
names the device. Without a model nothing is served, so no command is printed.
It has no `--reload`: uvicorn's reloader restarts the server on every saved
source file, and the server's shutdown stops the GPU job it runs.

**`--train-model` takes the deployment's GPU lock** (M5; review of 8242b47,
P2-4), as `scripts/train_model.py` does: the demo trains through the real
trainer on the GPU, so it is a managed GPU job. The lock is taken before
anything is written and held until training ends; a busy lock, or one that is
not configured (SHEPHERD_WORKSPACES_ROOT, the lock's one name, not set), is
refused with exit 1 and nothing written.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

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
#: Epochs of each stage `--train-model` trains, the base and its disease child:
#: the demo graph is learnt in that, and the point is the path, not the score.
DEMO_EPOCHS = 2


def demo_task_training() -> Dict[str, Dict[str, Any]]:
    """The demo graph's training sizes, by stage ("base", "disease"): the
    reference base spec shrunk (input 8, two layers of width 4 with 2 heads)
    and small batches, hops and samples, so the base and its disease child each
    train in seconds on CPU. Every other setting is the stage's reference
    recipe (`src.config.recipe`), which the trainer fills and records.

    **The one statement of the demo's sizes**: `train_demo_model` trains at
    them, and the deployment probe trains its demo disease model at them
    (`scripts/probe_deployment.py`, `_train_disease_child`)."""
    import dataclasses

    from src.models.bundle import reference_spec
    from src.models.bundle.spec import InputSpec

    spec = reference_spec("base")
    small = dataclasses.replace(
        spec, input=InputSpec(dim=8),
        encoder=dataclasses.replace(spec.encoder, layers=2, hidden=4, output=4, heads=2),
    ).validate()
    return {
        "base": {"model_spec": small.to_dict(), "batch_size": 32, "num_neighbors": [4, 4]},
        "disease": {"batch_size": 8, "num_neighbors": [3, 3], "candidate_sample_size": 8,
                    "sparse_sample_size": 8},
    }


def train_demo_model(workspace: Path, num_epochs: int, device: str = "auto") -> Path:
    """Train a base and one disease child of it on the demo workspace
    **through the real trainer**, and return the child's checkpoint.

    Both runs are `scripts/train_model.py`'s `train` on the model bundle path,
    as `tests/fixtures/trained_children.py` and the probe's
    `_train_disease_child` train them: `learning_stage` "base", then
    "finetune" with task "disease" and `finetune_from` the base's `last.pt`,
    each filed under the workspace's runs (contract M3a-2) and given every
    setting here as the operator's (the stage's recipe fills the rest). If
    this returns, the production path works end to end on this machine; the
    sizes are small because the demo graph is small, not a different pipeline.

    Returns:
        The disease child's ``last.pt``, the checkpoint the API should serve.
    """
    from scripts.train_model import TrainConfig, train
    from src.models.bundle.spec import ModelSpec
    from src.utils.checkpoint_paths import propose_run_dir, run_checkpoint_dir

    sizes = demo_task_training()
    # A bundle run is filed under its spec's encoder kind (K2).
    encoder = ModelSpec.from_dict(sizes["base"]["model_spec"]).encoder.kind

    def run(stage: str, name: str, **fields: Any) -> Path:
        run_dir = propose_run_dir(workspace, encoder, name)
        fields = {**sizes[stage], **fields, "num_epochs": num_epochs,
                  # `auto` refuses without CUDA (contract M3a); a CPU-only
                  # development host passes --device cpu.
                  "device": device}
        train(TrainConfig(data_dir=str(workspace), run_dir=str(run_dir), **fields), set(fields))
        last = run_checkpoint_dir(run_dir) / "last.pt"
        if not last.is_file():
            raise SystemExit(
                f"Demo training wrote no checkpoint under {last.parent}; see the log above. "
                "The workspace itself was written and is valid: train on it with "
                f"scripts/train_model.py --data-dir {workspace} --stage base, in a shell where "
                "the GPU lock is configured (SHEPHERD_WORKSPACES_ROOT, deployment guide §4.11)")
        return last

    base = run("base", "demo-base", learning_stage="base")
    child = run("disease", "demo-disease", learning_stage="finetune", task="disease",
                finetune_from=str(base))
    logger.info("Demo disease model: %s (base %s)", child, base)
    return child


def serving_device(training_device: str) -> str:
    """The device the printed serving command names: the one the demo trained on.

    **Serving refuses `auto` without CUDA** (contract M3c), as training does, so
    a command naming no device would be refused for a demo trained with
    `--device cpu` on a host without CUDA. Training's `auto` ran on CUDA -- it
    refuses otherwise -- so the command names `cuda` for it; any other device is
    named as it was given.
    """
    requested = (training_device or "").strip() or "auto"
    return "cuda" if requested == "auto" else requested


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
        help="Also train a small base and its disease model, which the printed command "
             "serves",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEMO_EPOCHS,
        help="With --train-model: epochs of each stage, the base and its disease model "
             f"(default: {DEMO_EPOCHS})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device when --train-model is given: auto (CUDA, refused "
             "without it), cuda, cuda:N or cpu (default: auto)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # **A managed GPU job takes the deployment's GPU lock** before anything is
    # written, and keeps it until training ends (src/utils/gpu_lock.py).
    gpu_lock = None
    if args.train_model:
        from src.utils.gpu_lock import GpuLockMisconfigured, take_gpu_lock

        try:
            gpu_lock = take_gpu_lock("training", "setup_demo.py --train-model", log=logger)
        except (OSError, GpuLockMisconfigured) as exc:  # GpuLockBusy is an OSError
            logger.error(f"setup_demo: refused: {exc} Nothing was written.")
            return 1
    try:
        return _setup(args, output_dir)
    finally:
        if gpu_lock is not None:
            gpu_lock.release()


def _setup(args: argparse.Namespace, output_dir: Path) -> int:
    """`main`'s work, under the GPU lock when it trains."""
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
    )
    kg_path = output_dir / "kg.json"

    checkpoint_path = None
    if args.train_model:
        checkpoint_path = train_demo_model(output_dir, args.epochs, args.device)

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
Start the API serving the demo's disease model:

  SHEPHERD_KG_PATH={kg_path} \\
  SHEPHERD_CHECKPOINT_PATH={checkpoint_path} \\
  SHEPHERD_DATA_DIR={output_dir} \\
  SHEPHERD_DEVICE={serving_device(args.device)} \\
  python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
""")
    else:
        print(f"""
No model was trained, so this workspace has nothing to serve yet: the API
answers diagnoses only with a trained model, and refuses them (503) until one
is loaded.

Train on this workspace, in a shell where the GPU lock is configured (training
refuses without it; deployment guide §4.11), for example with the workspaces
root the service is given:
  export SHEPHERD_WORKSPACES_ROOT={output_dir.resolve().parent}
  python scripts/setup_demo.py --train-model
  # or the production entry point at the reference recipe's full size: a base,
  # then its disease model fine-tuned from the base run's checkpoint
  python scripts/train_model.py --data-dir {output_dir} --stage base
  python scripts/train_model.py --data-dir {output_dir} --stage finetune \\
      --task disease --finetune-from <base run>/checkpoints/last.pt
  # on a host without CUDA, add --device cpu to each: auto refuses there
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
