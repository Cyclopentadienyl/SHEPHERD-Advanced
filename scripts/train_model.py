#!/usr/bin/env python3
"""
SHEPHERD-Advanced Model Training Script
========================================
Main entry point for training the ShepherdGNN model.

Script: scripts/train_model.py
Absolute Path: /home/user/SHEPHERD-Advanced/scripts/train_model.py

Purpose:
    Command-line interface for training ShepherdGNN on rare disease diagnosis.
    Handles configuration loading, data preparation, model initialization,
    and training loop execution.

Usage:
    python scripts/train_model.py --config configs/train_config.yaml
    python scripts/train_model.py --data-dir data/workspaces/demo --epochs 100
    python scripts/train_model.py --data-dir data/workspaces/demo --run-name pilot
    python scripts/train_model.py --resume data/workspaces/<ws>/runs/hgt/<run>/checkpoints/last.pt

    Every run writes into a new directory of its own,
    {data_dir}/runs/{conv_type}/{date}_{run_name}_{run_id}/ (contract M3a-2).

    One managed GPU job at a time (M5): `main` takes the deployment's GPU lock
    (src/utils/gpu_lock.py) before anything is read; a busy lock is refused,
    naming its holder, and so is a lock that is not configured (neither
    SHEPHERD_GPU_LOCK_FILE nor SHEPHERD_WORKSPACES_ROOT set), each with exit 1
    and nothing written.

    A caller that chose the dataset beforehand passes the fingerprint it showed
    (`--expect-dataset-sha256`, M5's job runner does): a workspace whose
    split_manifest.json, as this run reads it, has other bytes is refused with
    exit 1 and nothing written, before the graph and the samples are read.

Dependencies:
    - argparse: CLI argument parsing
    - yaml: Configuration file loading
    - torch: Model training
    - src.training.trainer: Trainer, TrainerConfig
    - src.kg.data_loader: DiagnosisDataLoader, DataLoaderConfig
    - src.kg.graph: KnowledgeGraph
    - src.models.gnn.shepherd_gnn: ShepherdGNN, ShepherdGNNConfig

Input:
    - Configuration file (YAML) or CLI arguments
    - Processed knowledge graph data
    - Training samples (patient phenotypes -> disease)

Output:
    - Trained model checkpoints
    - Training logs and metrics
    - Final evaluation report

Called by:
    - User via command line
    - CI/CD pipelines
    - Automated training scripts

Version: 1.0.0
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Collection, Dict, List, NamedTuple, Optional, Set, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.runtime_presets import (
    ALLOC_SOURCE_ENV,
    allocator_env,
    load_runtime_settings,
)


def _apply_allocator() -> str:
    """Put the CUDA allocator into the environment; say where it came from.

    **Before torch is imported, because torch fixes its allocator backend when
    its CUDA library loads**, so this cannot wait for ``main()``. A run started
    from a shell used to get the native allocator, which fragments under HGT
    training, while the same run started from the WebUI inherited the backend's
    preset. The shared rule, with ``inherit=True``, keeps whatever allocator is
    already in the environment -- the backend's, or an explicit override -- and
    applies the saved preset only where there is none.
    """
    inherited = "PYTORCH_ALLOC_CONF" in os.environ or "PYTORCH_CUDA_ALLOC_CONF" in os.environ
    resolved = allocator_env(os.environ, load_runtime_settings(), inherit=True)
    for key in ("PYTORCH_ALLOC_CONF", ALLOC_SOURCE_ENV):
        if key in resolved:
            os.environ[key] = resolved[key]
    if not inherited:
        return "saved preset"
    if os.environ.get(ALLOC_SOURCE_ENV) == "preset":
        return "inherited from the backend (preset)"
    return "environment (explicit)"


# Only when this file is the program. Imported as a module -- by a test, say --
# torch may already be loaded, which makes it too late to matter, and the
# importing process's environment is not this module's to rewrite.
ALLOCATOR_SOURCE = _apply_allocator() if __name__ == "__main__" else None

import torch
import yaml

from src.training.trainer import Trainer, TrainerConfig, apply_seed
from src.training.loss_functions import LossConfig
from src.kg.data_loader import (
    DIAGNOSIS_SUBGRAPH_HOPS,
    DataLoaderConfig,
    DiagnosisDataLoader,
    DiagnosisDataset,
    DiagnosisSample,
    create_diagnosis_dataloader,
)
from src.models.gnn.shepherd_gnn import ShepherdGNN, ShepherdGNNConfig, create_model
from src.evaluation.cohort import resolve_cohort, verify_cohort_reads
from src.utils.gpu_lock import GpuLockMisconfigured, take_gpu_lock

if TYPE_CHECKING:
    from src.kg.artifacts import DatasetRecord, ManifestRead
    from src.kg.storage.file_storage import GraphArtifactsRead, SamplesRead
    from src.utils.checkpoint_io import CheckpointRead
    from src.utils.fingerprint import ReadIdentity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class TrainConfig:
    """Complete training configuration"""
    # Paths
    data_dir: str = "data/workspaces/default"
    config_file: Optional[str] = None
    # Where the run writes (contract M3a-2): a directory of its own, allocated
    # as {data_dir}/runs/{conv_type}/{date}_{run_name}_{run_id}/ unless
    # `run_dir` names one, which must be absent or empty. Its checkpoints/ and
    # logs/ are inside it. The name is for people and optional.
    run_name: str = ""
    run_dir: str = ""

    # Model configuration
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    conv_type: str = "gat"
    dropout: float = 0.1
    use_ortholog_gate: bool = True

    # Training configuration
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Learning rate scheduling
    scheduler_type: str = "cosine"
    warmup_steps: int = 500
    min_lr_ratio: float = 0.01

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "float16"

    # Data loading
    num_workers: int = 4
    num_neighbors: List[int] = field(default_factory=lambda: [15, 10, 5])
    max_subgraph_nodes: int = 5000
    num_negative_samples: int = 5

    # Validation
    eval_every_n_epochs: int = 1
    early_stopping_patience: int = 10
    save_top_k: int = 3

    # Loss weights
    diagnosis_weight: float = 1.0
    link_prediction_weight: float = 0.5
    contrastive_weight: float = 0.3
    ortholog_weight: float = 0.2

    # Loss shape knobs. These are accepted by the API and the WebUI, so they must
    # exist here: load_config() only copies a YAML key when TrainConfig has the
    # attribute, and until they were added the user's values were silently dropped
    # and LossConfig always used its own defaults. Defaults below match LossConfig
    # exactly, so a run that does not set them behaves identically.
    margin: float = 1.0
    label_smoothing: float = 0.1
    temperature: float = 0.07

    # Device
    device: str = "auto"  # "auto", "cuda", "cpu"

    # Reproducibility
    seed: int = 42

    # torch.compile (experimental). Off by default. When True, the model's
    # forward is compiled in-place via nn.Module.compile() (no _orig_mod prefix,
    # so checkpoints stay compatible). dynamic=None (auto) handles variable
    # subgraph shapes; mode="default". A graph break / codegen failure (e.g. the
    # sm_121 Triton/ptxas issue on GB10) falls back to eager via suppress_errors.
    compile: bool = False

    # Continue from a parent checkpoint (contract M3a-3, M3a-4). What happens
    # is decided by the parent's own record (`src.utils.run_plan`): an
    # unfinished stage is resumed with every learning setting inherited; a
    # finished one is extended at a fixed rate only with `extend_epochs`;
    # `extend_lr` replaces a zero rate, and only a zero rate. None of these, nor
    # `pause_after_epoch`, is inherited by a child.
    resume_from: Optional[str] = None
    extend_epochs: Optional[int] = None
    extend_lr: Optional[float] = None
    # Stop at a safe boundary once this many epochs of the run's total progress
    # are complete, recorded as paused. CLI and probe only.
    pause_after_epoch: Optional[int] = None


# PyTorch device grammar, kept in step with the API (src/api/routes/training.py::_DEVICE_RE) and
# the field spec (src/config/training_fields.py, field "device", valid_pattern). The CLI used to
# restrict --device to {auto,cuda,cpu}, which made it impossible to target a specific GPU on a
# multi-GPU host even though torch.device() and the rest of the pipeline accept "cuda:N".
_DEVICE_RE = re.compile(r"^(auto|cpu|mps|cuda(:\d+)?)$")


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _sha256_arg(value: str) -> str:
    """argparse type for --expect-dataset-sha256: 64 lowercase hexadecimal
    digits, the form the server's lists show a dataset fingerprint in."""
    if not _SHA256_RE.fullmatch(value):
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a SHA-256 (64 lowercase hexadecimal digits)")
    return value


class DatasetChanged(ValueError):
    """The workspace's split_manifest.json, as this run read it, is not the one
    the run was started for (`--expect-dataset-sha256`): refused before the
    graph and the samples are read and before anything is written, as
    measure_scorer refuses a checkpoint or cohort version that is not the one
    chosen (review of 8242b47, P2-2)."""


def _device_arg(value: str) -> str:
    """argparse type for --device: validate against the PyTorch device grammar."""
    # fullmatch, not match: Python's ``$`` also matches just before a trailing newline, so
    # ``match()`` would accept "cuda\n" (e.g. from a file-fed argument).
    if not _DEVICE_RE.fullmatch(value):
        raise argparse.ArgumentTypeError(
            f"device must be 'auto', 'cpu', 'mps', 'cuda' or 'cuda:N' (got {value!r})."
        )
    return value


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Train ShepherdGNN for rare disease diagnosis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Configuration file
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to YAML configuration file",
    )

    # Paths
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing processed data",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional name for this run, part of its allocated directory's name",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Write this run into this directory instead of an allocated one; it must "
             "be absent or empty",
    )

    # Model
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Hidden dimension size",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of GNN layers",
    )
    parser.add_argument(
        "--conv-type",
        type=str,
        choices=["gat", "hgt", "sage"],
        default=None,
        help="GNN convolution type",
    )

    # Training
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr", "--learning-rate",
        type=float,
        default=None,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=None,
        help="Gradient accumulation steps",
    )

    # Device
    parser.add_argument(
        "--device",
        type=_device_arg,
        default=None,
        help="Device: auto, cpu, mps, cuda, or cuda:N (e.g. cuda:1 to pick a specific GPU)",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile (experimental; falls back to eager on failure)",
    )

    # Continue from a parent
    parser.add_argument(
        "--resume",
        type=str,
        help="Parent checkpoint to continue from. An unfinished stage is resumed with "
             "the parent's settings; a finished one is extended with --extend-epochs",
    )
    parser.add_argument(
        "--extend-epochs",
        type=int,
        default=None,
        help="Extend a finished parent by this many epochs at a fixed rate",
    )
    parser.add_argument(
        "--extend-lr",
        type=float,
        default=None,
        help="With --extend-epochs: the rate for parameter groups whose rate ended at 0 "
             "(only those; refused when none did)",
    )
    parser.add_argument(
        "--pause-after-epoch",
        type=int,
        default=None,
        help="Stop once this many epochs of the run's total progress are complete, "
             "with a resumable checkpoint, recorded as paused",
    )

    # A precondition of a run whose dataset was chosen beforehand (M5's job
    # runner): neither a setting nor recorded, so not a TrainConfig field.
    parser.add_argument(
        "--expect-dataset-sha256",
        type=_sha256_arg,
        default=None,
        metavar="HEX",
        help="Refuse before anything is written unless the workspace's split_manifest.json, "
             "as this run reads it, has this SHA-256 (the dataset fingerprint shown when the "
             "run was chosen). Without it the run trains on the manifest it reads and records "
             "that one",
    )

    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (verbose logging)",
    )

    return parser.parse_args()


def load_config(args: argparse.Namespace) -> TrainConfig:
    """Load configuration from file and command-line arguments"""
    return load_config_with_explicit(args)[0]


#: CLI flags whose value lands in a `TrainConfig` field of another name.
_FLAG_FIELDS = {"epochs": "num_epochs", "lr": "learning_rate",
                "grad_accum": "gradient_accumulation_steps", "resume": "resume_from",
                "no_amp": "use_amp"}


def load_config_with_explicit(args: argparse.Namespace) -> Tuple[TrainConfig, Set[str]]:
    """The configuration, and the fields the operator set: the configuration
    file's keys and the CLI flags given. A continued run inherits its parent's
    learning settings and refuses only those set explicitly to another value
    (contract M3a-3), so a default is never mistaken for a choice."""
    config = TrainConfig()
    explicit: Set[str] = set()

    # Load from YAML if provided
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            yaml_config = yaml.safe_load(f)
        from src.utils.checkpoint_paths import refuse_retired_locations

        refuse_retired_locations(yaml_config, source=f"The configuration {args.config}")
        for key, value in yaml_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
                explicit.add(key)

    # Override with command-line arguments (only if explicitly provided)
    if args.data_dir is not None:
        config.data_dir = args.data_dir
    if args.run_name is not None:
        config.run_name = args.run_name
    if args.run_dir is not None:
        config.run_dir = args.run_dir
    if args.hidden_dim is not None:
        config.hidden_dim = args.hidden_dim
    if args.num_layers is not None:
        config.num_layers = args.num_layers
    if args.conv_type is not None:
        config.conv_type = args.conv_type
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.grad_accum is not None:
        config.gradient_accumulation_steps = args.grad_accum
    if args.device is not None:
        config.device = args.device
    if args.no_amp:
        config.use_amp = False
    if args.compile:
        config.compile = True
    if args.resume:
        config.resume_from = args.resume
    if args.extend_epochs is not None:
        config.extend_epochs = args.extend_epochs
    if args.extend_lr is not None:
        config.extend_lr = args.extend_lr
    if args.pause_after_epoch is not None:
        config.pause_after_epoch = args.pause_after_epoch
    if args.seed is not None:
        config.seed = args.seed

    for name, value in vars(args).items():
        if name in ("config", "debug", "expect_dataset_sha256") or value is None \
                or value is False:
            continue
        explicit.add(_FLAG_FIELDS.get(name, name))
    return config, explicit


# =============================================================================
# Data Loading
# =============================================================================
#: The generated splits a training run consumes.
TRAINING_SPLITS = ("train", "val")


class TrainingInputs(NamedTuple):
    """What a training run read from its workspace, once each, and checked.

    The resume parent is not among them: it is read later, where it is restored,
    and nothing compares it with this workspace yet (M2.3).

    Carried whole from `read_training_inputs` to the record, so the tensors and
    samples a run trains on and the digests it records come from the same reads
    and cannot be paired with another read of a file that changed in between.
    """

    manifest: ManifestRead
    graph: GraphArtifactsRead
    samples: Dict[str, SamplesRead]
    #: Whether the validation split holds samples, decided once from the
    #: samples parsed. A run without them trains without validation and must
    #: not record the file as an input its results rest on.
    with_validation: bool
    #: The digest the manifest this run read binds for each of the four graph
    #: roles, as `verify_graph_reads` returned it after checking the tensors
    #: against it. `kg` is the source graph the tensors were exported from;
    #: training records it without parsing `kg.json` (contract M2.2).
    bound: Dict[str, str]
    #: The workspace's dataset record, read once and verified against this
    #: manifest reading and `bound["kg"]` (contract M3b). Not a recorded role:
    #: the manifest this run records binds its digest.
    dataset_record: DatasetRecord


def read_training_inputs(
    data_dir: Path, *, expected_manifest_sha256: Optional[str] = None,
) -> TrainingInputs:
    """Read the manifest, the graph tensors and both sample files once each, and
    verify from those reads (contract M2.1, S8).

    **The dataset the run was started for** (`expected_manifest_sha256`, the
    fingerprint a caller showed; review of 8242b47, P2-2) is compared on this
    one manifest read, right after it and before the graph and the samples are
    read: a workspace republished since it was shown is refused as changed
    (`DatasetChanged`), never trained on, and never reported as a manifest
    that contradicts its files.

    **Before anything is written.** `train` calls this ahead of the run
    directories, `config.yaml` and `runtime.json`: whether this workspace's graph
    is the export its manifest names, and whether its cohorts are the ones that
    manifest describes, is knowable from the files alone.

    **The digests compared are those of the bytes parsed.** A file replaced
    after its read is either the one trained on and recorded, or refused as not
    what the manifest records; nothing is hashed again by path.

    `kg.json` is not parsed by training, so it gets the identification check
    (M2.1 decision 1): the file on disk is compared with this run's one manifest
    reading, and nothing about it is recorded.

    **The dataset record is required** (contract M3b): `kg.provenance.json` is
    read once and refused unless it is the record this manifest reading binds
    and names the graph the tensors were verified as. Its origin is logged
    beside its binding state.

    `training_fields=True`: training carries the two optional sample fields
    measurement does not (`candidate_disease_ids`, `gene_ids`).
    """
    from src.kg.artifacts import (
        check_unparsed_kg_json,
        read_split_manifest,
        verify_dataset_record,
        verify_graph_reads,
    )
    from src.kg.storage.file_storage import read_graph_artifacts, read_samples

    # Stat only: a missing split or a workspace generated before the allocation
    # step is named before anything is read.
    for split in TRAINING_SPLITS:
        resolve_cohort(data_dir, split)

    manifest = read_split_manifest(data_dir)
    read_sha256 = manifest.identity.sha256
    if expected_manifest_sha256 is not None and read_sha256 != expected_manifest_sha256:
        raise DatasetChanged(
            f"The dataset changed since it was shown; select it again. "
            f"{manifest.identity.path} holds bytes {read_sha256[:12]}..., not the "
            f"{expected_manifest_sha256[:12]}... this run was started for. Nothing was written.")
    logger.info(f"Dataset fingerprint (split_manifest.json sha256): {read_sha256}")
    graph = read_graph_artifacts(data_dir, map_location="cpu")
    samples = {
        split: read_samples(data_dir, split, training_fields=True)
        for split in TRAINING_SPLITS
    }

    bound = verify_graph_reads(manifest, graph.reads)
    check_unparsed_kg_json(data_dir, manifest)
    dataset_record = verify_dataset_record(manifest, bound["kg"])
    verify_cohort_reads(manifest, samples)

    # After verification: an emptied cohort file is refused as not the file the
    # manifest records, before it is refused as empty.
    if not samples["train"].samples:
        raise ValueError(
            f"No training samples found: {samples['train'].identity.path} is empty"
        )
    with_validation = bool(samples["val"].samples)
    logger.info(
        f"Read {len(samples['train'].samples)} train and "
        f"{len(samples['val'].samples)} val samples; "
        f"node counts: {graph.graph_data['num_nodes_dict']}"
    )
    logger.info(
        f"Dataset record {dataset_record.identity.path.name}: binding "
        f"{dataset_record.binding}, origin {dataset_record.origin} "
        f"(sha256 {dataset_record.identity.sha256[:12]})"
    )
    return TrainingInputs(manifest, graph, samples, with_validation, bound, dataset_record)


def read_resume_parent(
    requested: Optional[str], inputs: TrainingInputs, device: str,
) -> Optional["CheckpointRead"]:
    """Read the resume parent once and refuse it unless it was trained on this
    run's inputs (contract M2.3). Returns `None` when nothing was requested.

    **At the top of the run**, beside the workspace checks: `train()` calls this
    right after `read_training_inputs`, before the run directories,
    `config.yaml`, `runtime.json`, the model or any checkpoint exist, so a refused
    parent leaves none of them behind and nothing is restored.

    - **A requested parent that does not exist refuses.** The run does not train
      from scratch instead.
    - **Read once** (M2.1), with this site's options (`map_location` the run's
      device, `weights_only=True`: tensors and plain data, which every
      checkpoint contract M3a writes is). The trainer restores from this read's dict
      and the record's `resume_checkpoint` is this read's identity, so the state
      restored and the digest recorded cannot come from two reads.
    - **The parent must record every one of this run's workspace roles with the
      same digest, and no other** (`verify_training_record`, the comparator
      serving and measurement use; the role set is its argument). The graph
      roles (`kg`, `node_features`, `edge_indices`, `num_nodes`) are decision 1
      applied to the parent. The data roles (`split_manifest`, `train_samples`,
      and `val_samples` when this run validates) are the owner's decision of
      2026-10-06: resume continues training on the same data. A parent with no
      record (from before the contract), a role missing on either side (a
      parent that validated, resumed by a run whose parsed `val_samples` are
      empty, among them) or a different digest refuses. That is a product
      restriction, not a verdict on the parent.

    The parent's own `resume_checkpoint` is its lineage, not a workspace input,
    and is not compared: every link in a resume chain read the same files, so a
    checkpoint's record covers its history.
    """
    from src.utils.checkpoint_io import read_checkpoint

    if not requested:
        return None

    resume_path = Path(requested)
    if not resume_path.exists():
        raise FileNotFoundError(
            f"The resume parent {resume_path} does not exist. It was requested, so "
            "the run does not train from scratch instead; name an existing "
            "checkpoint, or start a new run without --resume."
        )

    # Named before the read, so a parent that does not load fails after a line
    # that says which file it was.
    logger.info(f"Reading the resume parent: {resume_path}")
    parent = read_checkpoint(resume_path, map_location=device, weights_only=True)
    check_resume_parent(
        parent.checkpoint, inputs, recorded_by=f"The resume parent {resume_path}"
    )
    logger.info(
        f"Resume parent accepted: {resume_path} (sha256 {parent.identity.sha256[:12]})"
    )
    return parent


def plan_from_parent(
    parent: "CheckpointRead", config: TrainConfig, explicit: Optional[Collection[str]],
    device: str,
) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """How this run continues from `parent` (`src.utils.run_plan`), and
    early stopping's best weights when the parent names a file for them.

    That file is read once, with the safe loader, and accepted only with the
    SHA-256 the parent recorded for it: the weights a run restores at its end
    are the ones its record names, or the run is refused.
    """
    from src.utils.run_plan import RunPlanRefused, plan_continuation
    from src.utils.checkpoint_io import read_checkpoint

    request = asdict(config)
    if explicit is None:
        defaults = asdict(TrainConfig())
        explicit = {name for name, value in request.items() if value != defaults[name]}
    plan = plan_continuation(parent.checkpoint, Path(parent.identity.path), request,
                             set(explicit), device=device,
                             parent_sha256=parent.identity.sha256)
    logger.info(f"{plan.action}.")
    if plan.inherited:
        logger.info("Inherited from the parent: " + ", ".join(
            f"{name}={value!r}" for name, value in sorted(plan.inherited.items())))

    best_weights = None
    if (plan.best_weights or {}).get("kind") == "file":
        source = plan.best_weights
        read = read_checkpoint(source["path"], map_location="cpu", weights_only=True)
        if read.identity.sha256 != source["sha256"]:
            raise RunPlanRefused(
                f"The parent names {source['path'].name} as early stopping's best weights "
                f"with SHA-256 {source['sha256']}; the file there now has "
                f"{read.identity.sha256}. Put the recorded file back or choose another "
                "checkpoint.")
        best_weights = read.checkpoint["state_dict"]
    return plan, best_weights


def check_resume_parent(
    checkpoint: Any, inputs: TrainingInputs, *, recorded_by: str,
    compared_with: str = "this run",
) -> None:
    """M2.3's rule, on a checkpoint already read: it must record every one of
    the workspace roles a run over `inputs` records, with the same digests, and
    no other workspace role. Raises `TrainingRecordMismatch`.

    One function, so the inventory (`audit_checkpoint_family.py --inventory`,
    M2.6) applies the rule `train` applies rather than a copy of it. The
    parent's own `resume_checkpoint` is its lineage and is not compared.
    """
    from src.utils.fingerprint import verify_training_record

    current = training_input_digests(inputs)
    record = (
        checkpoint.get("training_input_digests") if isinstance(checkpoint, dict) else None
    )
    recorded = set(record) - {"resume_checkpoint"} if isinstance(record, dict) else set()
    verify_training_record(
        checkpoint,
        current,
        sorted(set(current) | recorded),
        recorded_by=recorded_by,
        compared_with=compared_with,
    )


def training_input_digests(
    inputs: TrainingInputs, *, parent: Optional[ReadIdentity] = None,
) -> Dict[str, str]:
    """The record of what a training run consumed, by role, built only from the
    reads (contract M2.1, S8).

    **Only from the readers' results.** Every value is the digest of the bytes a
    reader parsed: the samples, the three graph tensors, the manifest, and the
    parent restored. Nothing here hashes a path, so a file replaced after its
    read cannot change what is recorded, and a digest that was never taken
    cannot be recorded as `None`.

    `inputs.with_validation` is the observation that validation ran: a run whose
    validation split is empty trains without it, and recording the role anyway
    would claim an input the run's results do not rest on.

    `parent` is the identity of the checkpoint restored (`read_resume_parent`'s
    read). A resumed run restores model weights, optimizer, scheduler, scaler and
    training state from it, so two runs over identical workspace files but
    different parents produce materially different results. **One role, not a
    lineage**: the parent's own provenance lives in the parent.

    **The manifest says how the workspace was cut, which the sample digests
    cannot.** Two byte-different sample files could have been split at the
    disease level or at the sample level, and nothing in their digests
    distinguishes those regimes.

    **`kg` is the manifest-bound source graph (contract M2.2).** Training does
    not parse `kg.json`; its `kg` role is the digest the manifest it read binds,
    against which the tensors it parsed were verified (`verify_graph_reads`).
    That is the transitive claim the manifest makes: these tensors are that
    graph's export. It is the one value here that is not a read's identity, and
    it comes from the same manifest reading as `split_manifest`.
    Nothing globs the directory, so an unrelated split appearing beside these
    files does not change the record of a run that never read it.
    """
    reads: Dict[str, Optional[ReadIdentity]] = {
        "train_samples": inputs.samples["train"].identity,
        "node_features": inputs.graph.reads.get("node_features"),
        "edge_indices": inputs.graph.reads.get("edge_indices"),
        "num_nodes": inputs.graph.reads.get("num_nodes"),
    }
    if inputs.with_validation:
        reads["val_samples"] = inputs.samples["val"].identity
    if parent is not None:
        reads["resume_checkpoint"] = parent
    reads["split_manifest"] = inputs.manifest.identity

    missing = sorted(role for role, read in reads.items() if read is None)
    if missing:
        raise ValueError(
            f"no read of {', '.join(missing)}, so there is no digest to record; a "
            "training record names only bytes a reader parsed"
        )
    digests = {role: read.sha256 for role, read in reads.items()}
    if not inputs.bound.get("kg"):
        raise ValueError(
            "the manifest this run read binds no kg digest, so the source graph "
            "of these tensors cannot be recorded"
        )
    digests["kg"] = inputs.bound["kg"]
    return digests


def create_dataloaders(
    config: TrainConfig,
    graph_data: Dict[str, Any],
    train_samples: List[DiagnosisSample],
    val_samples: Optional[List[DiagnosisSample]],
) -> Tuple[DiagnosisDataLoader, Optional[DiagnosisDataLoader]]:
    """Create training and validation data loaders over samples already read.

    `val_samples` is `None` when the run trains without validation; that is
    decided once, in `read_training_inputs`, and not again here.
    """
    # Data loader config
    dl_config = DataLoaderConfig(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        num_neighbors=config.num_neighbors,
        max_subgraph_nodes=config.max_subgraph_nodes,
        num_negative_samples=config.num_negative_samples,
    )

    train_loader = create_diagnosis_dataloader(
        samples=train_samples,
        graph_data=graph_data,
        config=dl_config,
    )

    if val_samples is not None:
        val_config = DataLoaderConfig(
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
            num_neighbors=config.num_neighbors,
            max_subgraph_nodes=config.max_subgraph_nodes,
        )
        val_loader = create_diagnosis_dataloader(
            samples=val_samples,
            graph_data=graph_data,
            config=val_config,
        )
    else:
        val_loader = None
        logger.warning("No validation samples found, skipping validation")

    return train_loader, val_loader


def loader_record(loader: DiagnosisDataLoader) -> Dict[str, Any]:
    """A loader's settings as they take effect (contract M3a-1).

    The subgraph is sampled over `DIAGNOSIS_SUBGRAPH_HOPS` hops, so only that
    many entries of `num_neighbors` are used; both lists are recorded. The
    validation loader is built without `num_negative_samples`, so it uses the
    `DataLoaderConfig` default, and that is what is recorded for it. Values are
    recorded as they are today, not corrected.
    """
    cfg = loader.config
    return {
        "batch_size": int(cfg.batch_size),
        "num_batches": int(len(loader)),
        "num_samples": int(len(loader.dataset)),
        "shuffle": bool(cfg.shuffle),
        "num_workers": int(cfg.num_workers),
        "pin_memory_requested": bool(cfg.pin_memory),
        "prefetch_factor": int(cfg.prefetch_factor) if cfg.num_workers > 0 else None,
        "num_hops": int(DIAGNOSIS_SUBGRAPH_HOPS),
        "num_neighbors_configured": [int(n) for n in cfg.num_neighbors],
        "num_neighbors_effective": [int(n) for n in cfg.num_neighbors[:DIAGNOSIS_SUBGRAPH_HOPS]],
        "max_subgraph_nodes": int(cfg.max_subgraph_nodes),
        "sampling_strategy": str(cfg.sampling_strategy),
        "num_negative_samples": int(cfg.num_negative_samples),
        "negative_sampling_strategy": str(cfg.negative_sampling_strategy),
        "pipeline_prefetch": int(cfg.pipeline_prefetch),
        "fast_subgraph_build": bool(cfg.fast_subgraph_build),
    }


# =============================================================================
# Model Creation
# =============================================================================
def create_model_from_config(
    config: TrainConfig,
    graph_data: Dict[str, Any],
) -> ShepherdGNN:
    """Create ShepherdGNN model from configuration"""
    # Extract metadata
    node_types = list(graph_data["num_nodes_dict"].keys())
    edge_types = list(graph_data["edge_index_dict"].keys())
    metadata = (node_types, edge_types)

    # Infer input channels from features
    in_channels_dict = {}
    for node_type, features in graph_data["x_dict"].items():
        if features.dim() >= 2:
            in_channels_dict[node_type] = features.size(-1)
        else:
            in_channels_dict[node_type] = config.hidden_dim

    # Model config
    model_config = ShepherdGNNConfig(
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        conv_type=config.conv_type,
        dropout=config.dropout,
        use_ortholog_gate=config.use_ortholog_gate,
    )

    model = ShepherdGNN(
        metadata=metadata,
        in_channels_dict=in_channels_dict,
        config=model_config,
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,} total, {num_trainable:,} trainable")

    return model


#: Fields of `TrainConfig` that say where files are, not how training runs.
#: Left out of the checkpoint's record of the configuration requested: the inputs
#: are identified by digest (`training_input_digests`), the parent by its digest,
#: and a checkpoint travels without the layout of the machine that wrote it.
LOCATION_FIELDS = frozenset({"data_dir", "run_dir", "config_file", "resume_from"})


class TrainingDeviceUnavailable(RuntimeError):
    """The device a run asked for cannot be had on this host."""


def resolve_training_device(requested: str) -> str:
    """The device training runs on, or a refusal (contract M3a, §5.2).

    **`auto` means CUDA, and without CUDA it refuses**, as measurement's does
    (`scripts/measure_scorer.py`, `_resolve_device`). It used to become CPU
    silently, so a run asked to train on the deployment's hardware trained
    somewhere else and said nothing; the checkpoint recorded only where it ran.
    An explicit `cpu` stays available for development and is recorded as asked.

    A request for CUDA on a host without it refuses here too, before any input
    is read or any output exists, instead of failing later when the model is
    moved. Other device strings are handed to torch as given.
    """
    wants_cuda = requested == "auto" or requested == "cuda" or requested.startswith("cuda:")
    if wants_cuda and not torch.cuda.is_available():
        raise TrainingDeviceUnavailable(
            f"--device {requested} requires CUDA, which is not available here. "
            "Hardware acceleration is a hard requirement for deployment and CUDA "
            "is currently the only supported backend, so training does not "
            "substitute the CPU for it. Pass --device cpu explicitly for "
            "development; the run records the device it was given."
        )
    return "cuda" if requested == "auto" else requested


# =============================================================================
# Main Training Function
# =============================================================================
def train(config: TrainConfig, explicit: Optional[Collection[str]] = None, *,
          expected_dataset_sha256: Optional[str] = None) -> Dict[str, float]:
    """
    Main training function

    Args:
        config: the requested configuration.
        explicit: the fields the operator set (`load_config_with_explicit`).
            `None`, for a caller that built `config` itself, counts every
            field that differs from its default as set.
        expected_dataset_sha256: the dataset fingerprint the run was started
            for (`--expect-dataset-sha256`), compared on the run's one manifest
            read (`read_training_inputs`); `DatasetChanged` otherwise. A
            precondition, not a setting: neither recorded nor configured.

    Returns:
        Final evaluation metrics
    """
    logger.info("=" * 60)
    logger.info("SHEPHERD-Advanced Training")
    logger.info("=" * 60)
    logger.info(f"Configuration: {asdict(config)}")

    # Set device. Refused here, before any read or write, when it cannot be had.
    device = resolve_training_device(config.device)

    logger.info(f"Using device: {device} (requested: {config.device})")

    if device == "cuda":
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # **Refuse before the first run artifact exists.** Everything below this line
    # creates or writes something: run directories, `config.yaml`, a checkpoint
    # directory, then a model built from the graph. Whether this workspace's graph
    # is the export its manifest names, and whether its cohorts are the ones that
    # manifest describes, is knowable from the files alone — so it is knowable
    # before any of that. Verifying later left a run that had already written its
    # configuration and loaded a mixed graph.
    #
    # **One read per workspace input, and the check compares those reads**
    # (contract M2.1, S8). The graph tensors and samples verified here are the
    # ones trained on and recorded; `kg.json` is only identified (decision 1).
    # The resume parent is read and checked here too (contract M2.3): a missing
    # parent, one from before the contract, or one trained on other graph or
    # sample files is refused before anything below exists, and the trainer
    # later restores from this one read.
    #
    # No rollback and no transactional writing: the refusal simply moves ahead of
    # the side effects.
    data_dir = Path(config.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.info("Build a workspace first:")
        logger.info("  python scripts/build_knowledge_graph.py --generate-samples")
        return {}

    # An explicit run directory that holds files is refused here, before the
    # first artifact exists (contract M3a-2). It is claimed below, after every
    # refusal, which decides again.
    from src.utils.checkpoint_paths import (
        check_run_dir_free, claim_run_dir, propose_run_dir, run_checkpoint_dir, run_log_dir,
    )

    if config.run_dir:
        check_run_dir_free(config.run_dir)

    inputs = read_training_inputs(data_dir, expected_manifest_sha256=expected_dataset_sha256)
    parent = read_resume_parent(config.resume_from, inputs, device)

    # **How the run continues from its parent, decided once, before any output**
    # (contract M3a-3, M3a-4): from the parent's own record, its learning
    # settings inherited and an explicit conflict refused. The run uses the
    # plan's settings from here on.
    plan = None
    best_weights = None
    if parent is not None:
        plan, best_weights = plan_from_parent(parent, config, explicit, device)
        config = TrainConfig(**plan.settings)
    elif config.extend_epochs or config.extend_lr is not None:
        raise ValueError("extend_epochs and extend_lr continue a parent; give resume_from.")

    # **The code revision and the environment, collected once** (contract
    # M3a-1), after every refusal above and before anything is written. Both
    # go into every checkpoint; neither refuses anything.
    from src.utils.run_environment import code_revision, collect_environment

    producer = code_revision(PROJECT_ROOT)
    environment = collect_environment(device)

    # **The run's own directory** (contract M3a-2): named now, claimed below
    # after the last refusal (the parent's schedule is checked once the
    # trainer is built), so a refused run leaves nothing. Nothing writes
    # outside it, so no run overwrites or prunes another's files, a continued
    # run's parent included.
    run_dir = Path(config.run_dir) if config.run_dir else propose_run_dir(
        config.data_dir, config.conv_type, config.run_name)
    checkpoint_dir = run_checkpoint_dir(run_dir)
    log_dir = run_log_dir(run_dir)

    # **The seed governs the initial weights too** (contract M3a). The model
    # draws its weights when it is built, and the seed used to be applied first in
    # `Trainer.__init__`, after the loaders and the model, so `--seed` fixed the
    # training stream but not the weights that stream started from. The Trainer
    # applies the same seed again, so the stream training draws from is unchanged.
    if config.seed is not None:
        apply_seed(config.seed)

    # The data read and verified above, before anything was written.
    graph_data = inputs.graph.graph_data
    train_loader, val_loader = create_dataloaders(
        config,
        graph_data,
        inputs.samples["train"].samples,
        inputs.samples["val"].samples if inputs.with_validation else None,
    )

    # Create model
    model = create_model_from_config(config, graph_data)

    # torch.compile (experimental, opt-in). Compile in-place so the model
    # object/state_dict are unchanged (no _orig_mod prefix -> checkpoints stay
    # compatible with the non-compiled inference path). dynamic=None (auto)
    # tolerates the variable per-batch subgraph shapes; mode stays "default"
    # (reduce-overhead/CUDA-graphs need static shapes). suppress_errors makes a
    # graph break or codegen failure (e.g. the known sm_121 Triton/ptxas crash
    # on GB10) fall back to eager per-graph instead of killing training.
    compile_installed = False
    if config.compile:
        try:
            # NOTE: `import torch._dynamo as _dynamo` binds `_dynamo`, NOT `torch`.
            # A bare `import torch._dynamo` here would bind `torch` as a function-local
            # name, shadowing the module-level torch for the whole function and
            # breaking earlier `torch.*` uses with UnboundLocalError.
            import torch._dynamo as _dynamo
            _dynamo.config.suppress_errors = True
            model.compile()
            # Installed without raising; whether graphs then ran compiled is not
            # observed (suppress_errors falls back to eager per graph).
            compile_installed = True
            logger.info(
                "torch.compile ENABLED (in-place, dynamic=auto, mode=default). "
                "Tip: run once with TORCH_LOGS=graph_breaks to count breaks; "
                "verify MRR/Hits vs eager before trusting the speedup."
            )
        except Exception as exc:  # never let compile break training
            logger.warning(f"torch.compile failed to enable; using eager: {exc}")
    else:
        logger.info("torch.compile disabled (eager mode)")

    # Training configuration
    trainer_config = TrainerConfig(
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_grad_norm=config.max_grad_norm,
        scheduler_type=config.scheduler_type,
        warmup_steps=config.warmup_steps,
        min_lr_ratio=config.min_lr_ratio,
        use_amp=config.use_amp,
        amp_dtype=config.amp_dtype,
        eval_every_n_epochs=config.eval_every_n_epochs,
        early_stopping_patience=config.early_stopping_patience,
        checkpoint_dir=str(checkpoint_dir),
        save_top_k=config.save_top_k,
        log_dir=str(log_dir),
        device=device,
        seed=config.seed,
        pause_after_epoch=config.pause_after_epoch,
        loss_config=LossConfig(
            diagnosis_weight=config.diagnosis_weight,
            link_prediction_weight=config.link_prediction_weight,
            contrastive_weight=config.contrastive_weight,
            ortholog_weight=config.ortholog_weight,
            margin=config.margin,
            label_smoothing=config.label_smoothing,
            temperature=config.temperature,
        ),
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=trainer_config,
    )

    # Two identities, attached side by side, answering different questions.
    #
    #   data_fingerprint        is this checkpoint STRUCTURALLY COMPATIBLE with
    #                           the graph in front of me?
    #   training_input_digests  WHICH EXACT INPUTS produced it?
    #
    # The first cannot answer the second: it records node types, counts, feature
    # dims and edge types, and sample files do not enter it at all. Two runs over
    # different sample files therefore share a fingerprint, and a comparison
    # between their checkpoints cannot attribute a difference to data rather than
    # to configuration, randomness or training behaviour.
    #
    # Both are read by name in `ModelCheckpoint._save_checkpoint`; setting an
    # attribute here is not enough on its own.
    from src.utils.fingerprint import compute_fingerprint

    # Continue from the parent read and checked at the top (contract M2.3,
    # M3a-3); the trainer reads nothing. A schedule the parent's record cannot
    # rebuild is refused here, still before the run directory exists.
    if parent is not None:
        trainer.continue_from(parent.checkpoint, plan, best_weights)
        logger.info(
            f"Continuing from checkpoint: {parent.identity.path} "
            f"(sha256 {parent.identity.sha256[:12]})"
        )
    del best_weights

    # Claimed after the last refusal: the run's first artifact.
    claim_run_dir(run_dir)
    logger.info(f"Run directory: {run_dir}")

    # Save configuration: the settings in effect, with the directory the run
    # was given.
    config_path = run_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump({**asdict(config), "run_dir": str(run_dir)}, f)
    logger.info(f"Configuration saved to {config_path}")

    # The allocator is not a training parameter, so it is not in config.yaml,
    # but it changes memory by a multiple on this project; runs are compared
    # with it beside them.
    with open(run_dir / "runtime.json", "w") as f:
        json.dump(_allocator_record(), f, indent=2)
    # **Only the identity outlives the restore.** The model, optimizer,
    # scheduler and scaler hold what they took from the parent; the read itself
    # would otherwise keep every source tensor it parsed, on the run's device,
    # alive until training returns. Nothing is cleared: what the optimizer
    # adopted stays with the optimizer.
    parent_identity = parent.identity if parent is not None else None
    del parent

    # **The parent is in the record.** A first version attached the digests
    # before the resume decision; a resumed run then recorded every current file
    # and stayed silent about the parent checkpoint whose weights, optimizer,
    # scheduler, scaler and training state it restored. Two runs over identical
    # workspace files with different parents would have carried identical
    # digests.
    #
    # Both attributes are read by name in `ModelCheckpoint._save_checkpoint`, and
    # nothing reads them before `trainer.train()` below, so attaching them here
    # changes nothing except what they can see.
    trainer.data_fingerprint = compute_fingerprint(graph_data)

    # What only this entry point knows, for every checkpoint the trainer writes
    # (contract M3a-1; read by `build_checkpoint_payload`): the configuration
    # as requested, the device asked for, compile, the loaders as built, and
    # the revision and environment collected above.
    trainer.run_record = {
        "producer": producer,
        "environment": environment,
        "requested": {k: v for k, v in asdict(config).items() if k not in LOCATION_FIELDS},
        "device_requested": config.device,
        "compile": {"requested": bool(config.compile), "installed": compile_installed},
        "loaders": {
            "train": loader_record(train_loader),
            "val": loader_record(val_loader) if val_loader is not None else None,
        },
    }

    # Only the roles this run consumed, and only from the reads: the inputs read
    # and verified at the top, and the parent restored above.
    trainer.training_input_digests = training_input_digests(inputs, parent=parent_identity)

    # Train
    logger.info("Starting training...")
    final_metrics = trainer.train()
    if trainer.interrupted:
        last = checkpoint_dir / "last.pt"
        logger.info(
            "Training stopped. "
            + (f"The latest resumable checkpoint is {last} (epoch "
               f"{trainer.state.completed_epochs}); progress after it is lost."
               if last.is_file() else "No checkpoint was saved, so nothing can be resumed."))
        return {}

    # Log final results
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)

    for name, value in final_metrics.items():
        logger.info(f"  {name}: {value:.4f}")

    # Save final metrics
    metrics_path = run_dir / "final_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    logger.info(f"Final metrics saved to {metrics_path}")

    return final_metrics


# =============================================================================
# Entry Point
# =============================================================================
def _allocator_record() -> Dict[str, Optional[str]]:
    return {
        "PYTORCH_ALLOC_CONF": os.environ.get("PYTORCH_ALLOC_CONF"),
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        ALLOC_SOURCE_ENV: os.environ.get(ALLOC_SOURCE_ENV),
        "allocator_source": ALLOCATOR_SOURCE or "not applied (imported as a module)",
    }


def main():
    """Main entry point"""
    args = parse_args()
    # **One managed GPU job at a time** (M5): the deployment's GPU lock, taken
    # before anything is read and held until this returns; the kernel releases
    # it however the process ends (src/utils/gpu_lock.py).
    try:
        gpu_lock = take_gpu_lock("training", "train_model.py", log=logger)
    except (OSError, GpuLockMisconfigured) as exc:  # GpuLockBusy is an OSError; not
        # configured is a GpuLockMisconfigured (review of 8242b47, P2-4)
        logger.error(f"train_model: refused: {exc} Nothing was written.")
        return 1
    with gpu_lock:
        return _train_main(args)


def _train_main(args: argparse.Namespace) -> int:
    """`main`'s run, under the GPU lock."""
    record = _allocator_record()
    logger.info(
        "CUDA allocator: %s (%s)",
        record["PYTORCH_ALLOC_CONF"] or record["PYTORCH_CUDA_ALLOC_CONF"],
        record["allocator_source"],
    )

    # Set debug logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    config, explicit = load_config_with_explicit(args)

    # Validate training data exists
    data_dir = Path(config.data_dir)
    has_graph_data = (data_dir / "node_features.pt").exists()
    has_train_samples = (data_dir / "train_samples.json").exists()

    if has_graph_data and not has_train_samples:
        logger.error(
            f"Workspace {data_dir} has graph data but no train_samples.json. "
            f"Re-run build_knowledge_graph.py with --generate-samples."
        )
        return 1
    elif not data_dir.exists() or not has_graph_data:
        logger.error(f"Training data not found in {data_dir}")
        logger.error("Required files: train_samples.json, node_features.pt, edge_indices.pt")
        logger.error("Build training data first:")
        logger.error("  python scripts/build_knowledge_graph.py --workspace <dir> --external-dir data/external --generate-samples")
        return 1

    # Run training
    try:
        metrics = train(config, explicit, expected_dataset_sha256=args.expect_dataset_sha256)
        return 0 if metrics else 1
    except DatasetChanged as exc:  # a refusal, said once, not a failure with a traceback
        logger.error(f"train_model: refused: {exc}")
        return 1
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
