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

**Where it writes** (`default_output_dir`, T08): `demo` in the workspaces
folder in effect, resolved by the one resolver every process uses
(`server_settings.workspaces_root`: `SHEPHERD_WORKSPACES_ROOT` set for this
command, else the folder saved in Settings › Data locations). That is the
folder the dashboard lists and the one whose GPU lock `--train-model` takes,
so the data and the lock go to one place. With no folder in effect (none
chosen, or the one chosen is refused) it is refused with exit 1 and nothing
written, with or without `--train-model`, naming why and the suggested
folder; nothing falls back to the repository's `data/workspaces/`. An
explicit `--output-dir` always wins. *History: until T08 the default was the
repository's `data/workspaces/demo` whatever folder was chosen, so a demo
trained under the saved folder's lock was written where the dashboard did not
look.*

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

With `--train-model`, what to do next is printed on completion, **led by the
dashboard** (`dashboard_lines`, T08): start it (its desktop entry or
`launch_shepherd.sh`), which lists the demo when it is in the workspaces folder
saved in Settings (or says which folder to choose there), and load the disease
child in Diagnosis › Load a model…; a demo trained on the CPU says the
dashboard must then be started with ``SHEPHERD_DEVICE=cpu``. A command
starting the API alone follows only as a "Ref:", with the disease child's
checkpoint (its run's `last.pt`) and the device it was trained on: serving
refuses `auto` without CUDA (contract M3c), so the command names the device.
Without a model nothing is served, so no command is printed. It has no
`--reload`: uvicorn's reloader restarts the server on every saved source
file, and the server's shutdown stops the GPU job it runs. *History: until
T08 the uvicorn command was all it printed, so a first run was told to start
a bare API instead of the dashboard it had just configured.*

**`--train-model` takes the deployment's GPU lock** (M5; review of 8242b47,
P2-4), as `scripts/train_model.py` does: the demo trains through the real
trainer on the GPU, so it is a managed GPU job. The lock is taken before
anything is written and held until training ends; a busy lock, or one that is
not configured (no workspaces folder set, the lock's one name: Settings › Data
locations; Ref: SHEPHERD_WORKSPACES_ROOT), is refused with exit 1 and nothing
written. Without a model, the training step it prints says the lock
this shell would take, or how to choose the folder
(`gpu_lock.training_step_lines`, the hint `build_knowledge_graph.py` prints).
"""
from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import server_settings
from src.config.server_settings import SOURCE_SAVED
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
#: The demo workspace's folder in the workspaces folder in effect
#: (`default_output_dir`).
DEMO_WORKSPACE = "demo"


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
#: The names of the two runs `--train-model` trains, as the dashboard's lists
#: show them: the base, and the disease model fine-tuned from it.
DEMO_BASE_RUN = "demo-base"
DEMO_DISEASE_RUN = "demo-disease"
#: Distractor candidates per case of the demo workspace's functional gene
#: fixture (`scripts/derive_task_fixture.py --distractors`), which a gene or
#: patients-like-me child of the demo trains and is measured on: the demo
#: graph holds eight genes, too few for the script's default of 19.
DEMO_FIXTURE_DISTRACTORS = 4


def demo_task_training() -> Dict[str, Dict[str, Any]]:
    """The demo graph's training sizes, by stage and task ("base", "disease",
    "gene", "patients_like_me"): the reference base spec shrunk (input 8, two
    layers of width 4 with 2 heads) and small batches, hops and samples, so
    the base and each child of it train in seconds on CPU. Every other setting
    is the stage's reference recipe (`src.config.recipe`), which the trainer
    fills and records. A gene or patients-like-me child trains on the gene
    task's functional fixture of the demo workspace
    (`scripts/derive_task_fixture.py --distractors`, `DEMO_FIXTURE_DISTRACTORS`),
    which `--train-model` does not derive.

    **The one statement of the demo's sizes**: `train_demo_model` trains its
    base and disease child at them, the deployment probe trains and starts
    every demo model at them (`scripts/probe_deployment.py`: D1's base and
    child, M2's finetune, the base starts of M11, W7 and W11, L4's two bases
    and L7's disease, gene and patients-like-me children), and so do the
    tests' trained children (`tests/fixtures/trained_children.py`)."""
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
        "gene": {"batch_size": 4, "num_neighbors": [3, 3], "sparse_sample_size": 4},
        "patients_like_me": {"batch_size": 4, "num_neighbors": [3, 3], "sparse_sample_size": 4},
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
                f"scripts/train_model.py --data-dir {workspace} --stage base, once the GPU lock "
                "is configured (a workspaces folder chosen in Settings › Data locations; Ref: "
                "SHEPHERD_WORKSPACES_ROOT; deployment guide §4.11)")
        return last

    base = run("base", DEMO_BASE_RUN, learning_stage="base")
    child = run("disease", DEMO_DISEASE_RUN, learning_stage="finetune", task="disease",
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


def default_output_dir(environ: Optional[Mapping[str, str]] = None
                       ) -> Tuple[Optional[Path], Optional[str]]:
    """**Where the demo goes when `--output-dir` is not given**: `demo` in
    the workspaces folder in effect, by the one resolver
    (`server_settings.workspaces_root`: the variable set for this command,
    else the folder saved in Settings › Data locations), the folder the
    dashboard lists and whose GPU lock `--train-model` takes. With none in
    effect (not chosen, refused, or a saved file that cannot be read), None
    and that setting's own words for why; no other folder is assumed."""
    root = server_settings.workspaces_root(environ)
    if root.value is None:
        return None, root.problem
    return root.value / DEMO_WORKSPACE, None


def dashboard_lines(output_dir: Path, training_device: str,
                    environ: Optional[Mapping[str, str]] = None) -> List[str]:
    """**What to do with a trained demo, led by the dashboard** (T08): start
    it, find the workspace on Datasets and load the disease model in
    Diagnosis › Load a model…, which work with the folder saved in Settings
    and nothing exported. Said truthfully for where the demo went: in the
    workspaces folder saved in Settings, the dashboard lists it; in one set
    for this command only, it does once the same folder is chosen there; in
    another folder, once its parent is. A demo trained on the CPU is served
    only by a dashboard started with ``SHEPHERD_DEVICE=cpu`` (serving refuses
    `auto` without CUDA, contract M3c). Two-space indent, no trailing blank."""
    root = server_settings.workspaces_root(environ)
    parent = Path(output_dir).resolve().parent
    name = Path(output_dir).name
    if root.value is not None and parent == root.value and root.source == SOURCE_SAVED:
        where = (f"It lists the workspace {name}: it is in the workspaces folder saved in "
                 f"{server_settings.DATA_LOCATIONS}.")
    elif root.value is not None and parent == root.value:
        where = (f"It lists the workspace {name} once its workspaces folder is {parent}: this "
                 f"command was given that folder by {server_settings.ENV_WORKSPACES_ROOT}, so "
                 f"choose the same folder in {server_settings.DATA_LOCATIONS} if it is not "
                 "saved there.")
    else:
        where = (f"It lists the workspace {name} once {parent} is its workspaces folder: "
                 f"choose that folder in {server_settings.DATA_LOCATIONS}.")
    device = serving_device(training_device)
    start = "Start the dashboard: its desktop entry, or bash launch_shepherd.sh"
    lines = [f"  1. {start} (launch_shepherd.cmd on Windows). {where}"]
    if device == "cpu":
        lines.append("     It was trained on the CPU: on a host without CUDA, start it as "
                     "SHEPHERD_DEVICE=cpu bash launch_shepherd.sh, or the load is refused "
                     "(development hosts only; deployment guide §4.8).")
    lines.append(f"  2. Diagnosis › Load a model…: the run {DEMO_DISEASE_RUN} of the workspace "
                 f"{name} (the disease model; its base is the run {DEMO_BASE_RUN}).")
    return [wrapped for line in lines
            for wrapped in textwrap.wrap(line, 92, subsequent_indent="     ",
                                         break_long_words=False, break_on_hyphens=False)]


#: The refusal with no `--output-dir` and no workspaces folder in effect.
NO_OUTPUT_DIR = ("setup_demo: refused: the demo is written to the workspaces folder in effect "
                 "(the one the dashboard lists), and there is none. {why} Or name the folder "
                 "for this demo with --output-dir. Nothing was written.")


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
        default=None,
        help=f"Output directory (default: {DEMO_WORKSPACE} in the workspaces folder in effect, "
             "the one the dashboard lists: chosen in Settings › Data locations, or "
             "SHEPHERD_WORKSPACES_ROOT set for this command; with neither, the demo is "
             "refused unless this is given)",
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

    # **Where the demo goes** (`default_output_dir`), decided before the lock
    # and before anything is written: the lock and the data go to one folder.
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir, why = default_output_dir()
        if output_dir is None:
            logger.error(NO_OUTPUT_DIR.format(why=why))
            return 1

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
        steps = "\n".join(dashboard_lines(output_dir, args.device))
        print(f"""
Open the demo in the dashboard and load its disease model:
{steps}

Ref: the API alone, with no dashboard, serving this disease model:
  SHEPHERD_KG_PATH={kg_path} \\
  SHEPHERD_CHECKPOINT_PATH={checkpoint_path} \\
  SHEPHERD_DATA_DIR={output_dir} \\
  SHEPHERD_DEVICE={serving_device(args.device)} \\
  python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
""")
    else:
        from src.utils.gpu_lock import training_step_lines

        step = "\n".join(training_step_lines(output_dir))
        print(f"""
No model was trained, so this workspace has nothing to serve yet: the API
answers diagnoses only with a trained model, and refuses them (503) until one
is loaded.

Train on this workspace; training takes the GPU lock and refuses without it
(deployment guide §4.11):
{step}
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
