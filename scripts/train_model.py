#!/usr/bin/env python3
"""
SHEPHERD-Advanced Model Training Script
========================================
The one training entry point: a model bundle's KG base, or a task finetune
of one (docs/working/scorer-retraining/README.md §3-§4).

Script: scripts/train_model.py
Absolute Path: /home/user/SHEPHERD-Advanced/scripts/train_model.py

Purpose:
    Command-line interface for training a model bundle for rare disease
    diagnosis. Every run names its learning stage (`--stage base` or
    `--stage finetune`), a continued one included; a run without it is
    refused before anything is read. Handles configuration loading, data
    preparation, the model spec and the bundle, and the training loop.

Usage:
    python scripts/train_model.py --stage base --data-dir data/workspaces/demo --epochs 100
    python scripts/train_model.py --stage base --config configs/base.yaml --run-name pilot
    python scripts/train_model.py --stage finetune --task disease --data-dir <ws> \
        --finetune-from <ws>/runs/gatv2/<run>/checkpoints/last.pt
    python scripts/train_model.py --stage base --data-dir <ws> \
        --resume <ws>/runs/gatv2/<run>/checkpoints/last.pt

    Every run writes into a new directory of its own,
    {data_dir}/runs/{encoder kind}/{date}_{run_name}_{run_id}/, under its
    model spec's encoder (contract M3a-2).

    One managed GPU job at a time (M5): `main` takes the deployment's GPU lock
    (src/utils/gpu_lock.py) before anything is read; a busy lock is refused,
    naming its holder, and so is a lock that is not configured
    (SHEPHERD_WORKSPACES_ROOT, the lock's one name, not set), each with exit 1
    and nothing written.

    A caller that chose the dataset beforehand passes the fingerprint it showed
    (`--expect-dataset-sha256`, M5's job runner does): a workspace whose
    split_manifest.json, as this run reads it, has other bytes is refused with
    exit 1 and nothing written, before the graph and the samples are read.
    Likewise a resume parent chosen beforehand (`--expect-resume-sha256`, which
    the job runner always passes for a continued run): other bytes at --resume
    are refused with exit 1 and nothing written, before they are loaded.

    Every refusal comes before the run directory is claimed, and the command
    line says it in one line, ``train_model: refused: <why> Nothing was
    written.``, exit 1, with no traceback (a missing --stage among them); an
    error after the claim is a failed run, logged with its traceback.

Dependencies:
    - argparse: CLI argument parsing
    - yaml: Configuration file loading
    - torch: Model training
    - src.training.trainer: Trainer, TrainerConfig
    - src.training.lp_task, src.training.finetune_task: the stages' task modules
    - src.models.bundle: ModelSpec, build_bundle, the model record

Input:
    - Configuration file (YAML) or CLI arguments
    - The workspace's graph export and, for a finetune, its samples
    - A parent checkpoint for a finetune, a resume, an extension or a branch

Output:
    - Trained model checkpoints, each with its model record
    - Training logs and metrics
    - Final validation metrics

Called by:
    - User via command line
    - The job runner (src/api/services/training_manager.py)
    - scripts/setup_demo.py and the deployment probe

Version: 1.0.0
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING, Any, Callable, Collection, Dict, List, Mapping, NamedTuple, Optional, Set,
    Tuple,
)

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
from src.evaluation.cohort import verify_cohort_reads
from src.utils.gpu_lock import GpuLockMisconfigured, take_gpu_lock
# The refusals of a dataset or a resume parent that is not the one chosen,
# raised by their readers themselves (`expected_sha256=`); `main` says each once.
from src.kg.artifacts import DatasetChanged
from src.utils.checkpoint_io import CheckpointChanged
from src.utils.checkpoint_paths import RunDirectoryTaken

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
    # as {data_dir}/runs/{encoder kind}/{date}_{run_name}_{run_id}/ (its model
    # spec's encoder) unless `run_dir` names one, which must be absent or empty.
    # Its checkpoints/ and logs/ are inside it. The name is for people and
    # optional.
    run_name: str = ""
    run_dir: str = ""

    # The learning stage (docs/working/scorer-retraining/README.md §3–§4),
    # required: "base" trains KG link prediction into a model bundle
    # checkpoint with a model spec, "finetune" a task head on a base. A run
    # without one is refused before anything is read. In a new run (a base, a
    # finetune) the settings the operator leaves unset take the stage's
    # reference values (src/config/recipe.py), recorded as such; a resumed,
    # extended or branched run takes its parent's, recorded as inherited.
    learning_stage: Optional[str] = None
    # The model spec as plain data (ModelSpec.to_dict), read strictly; None is
    # the reference spec of the stage.
    model_spec: Optional[Dict[str, Any]] = None
    # The seed of the edge split (README F2), apart from the training seed so
    # runs over one graph view share one split unless asked otherwise.
    split_seed: int = 0
    # A finetune: its task, and the base checkpoint it starts from (a new
    # optimizer, the graph side loaded, a new task head; contract §9.2).
    task: Optional[str] = None
    finetune_from: Optional[str] = None
    # A branch: a run of the parent's stage and task from its weights, its
    # settings kept but those this request changes on the stage's closed list
    # (src/config/recipe.py `branch_settings`, contract §9.2, BACKLOG 23); a
    # new optimizer, each change recorded.
    branch_from: Optional[str] = None
    # Finetune settings; unset, the task's reference values
    # (src/config/recipe.py, README C3, U3, U9).
    task_loss_weight: Optional[float] = None
    lp_loss_weight: Optional[float] = None
    candidate_sample_size: Optional[int] = None
    sparse_sample_size: Optional[int] = None
    candidate_upsample_size: Optional[int] = None
    # The parts of a finetune that do not train (research comparison Stage 9,
    # owner's O4a; `src.config.recipe.FROZEN_PARTS`): "none", or "graph_side"
    # (the input module, the encoder and the LP head, frozen before the
    # optimizer is built and run in eval: `ModelBundle.freeze`). A base with
    # any freeze is refused (`frozen_parts_problem`); a continued or branched
    # run keeps its parent's.
    frozen_parts: str = "none"

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

    # Sampling: the neighbour fan-out per hop, one per encoder layer
    num_neighbors: List[int] = field(default_factory=lambda: [15, 10, 5])

    # Validation
    eval_every_n_epochs: int = 1
    early_stopping_patience: int = 10
    save_top_k: int = 3

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
    """argparse type for --expect-dataset-sha256 and --expect-resume-sha256:
    64 lowercase hexadecimal digits, the form the server's lists show a
    dataset or model fingerprint in."""
    if not _SHA256_RE.fullmatch(value):
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a SHA-256 (64 lowercase hexadecimal digits)")
    return value


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
        description="Train a model bundle (KG base, or a task finetune) for rare disease "
                    "diagnosis",
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

    # The model bundle path
    parser.add_argument(
        "--stage",
        type=str,
        choices=["base", "finetune"],
        default=None,
        help="The learning stage, required: 'base' is KG link prediction (the paper's "
             "pretraining), 'finetune' a task head on a base. A run without it, a continued "
             "one included, is refused",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["disease", "gene", "patients_like_me"],
        default=None,
        help="With --stage finetune: the task",
    )
    parser.add_argument(
        "--finetune-from",
        type=str,
        default=None,
        help="With --stage finetune: the base checkpoint to start from",
    )
    parser.add_argument(
        "--branch-from",
        type=str,
        default=None,
        help="With --stage: branch this checkpoint of the same stage and task, changing the "
             "settings given (each on the stage's list), with a new optimizer",
    )
    parser.add_argument(
        "--expect-parent-sha256",
        type=_sha256_arg,
        default=None,
        metavar="HEX",
        help="With --finetune-from or --branch-from: refuse unless the parent checkpoint, as "
             "this run reads it, has this SHA-256",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="With --stage: the edge split's seed",
    )
    parser.add_argument(
        "--frozen-parts",
        type=str,
        choices=["none", "graph_side"],
        default=None,
        help="With --stage finetune: the parts that do not train. 'graph_side' keeps the "
             "base's input module, encoder and LP head as they are (in eval mode), so only "
             "the patient encoder and task head train: the frozen-backbone control. A "
             "continued or branched run keeps its parent's",
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
    parser.add_argument(
        "--expect-resume-sha256",
        type=_sha256_arg,
        default=None,
        metavar="HEX",
        help="With --resume (or resume_from): refuse before anything is written unless the "
             "parent checkpoint, as this run reads it, has this SHA-256 (the model fingerprint "
             "of the parent chosen). Compared before the file is loaded",
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
                "no_amp": "use_amp", "stage": "learning_stage"}


def load_config_with_explicit(args: argparse.Namespace) -> Tuple[TrainConfig, Set[str]]:
    """The configuration, and the fields the operator set: the configuration
    file's keys and the CLI flags given. A continued run inherits its parent's
    learning settings and refuses only those set explicitly to another value
    (contract M3a-3), so a default is never mistaken for a choice."""
    config = TrainConfig()
    explicit: Set[str] = set()

    # The configuration file, read strictly (BACKLOG 54): a file that is not
    # there, is not a mapping, or names a key this version does not have is
    # refused, never skipped, so a mistyped setting is not trained without.
    if args.config:
        if not Path(args.config).exists():
            raise ValueError(f"The configuration {args.config} does not exist.")
        try:
            with open(args.config) as f:
                yaml_config = yaml.safe_load(f)
        except OSError as exc:
            raise ValueError(f"The configuration {args.config} cannot be read "
                             f"({type(exc).__name__}: {exc.strerror or exc}).") from None
        if yaml_config is None:
            yaml_config = {}
        if not isinstance(yaml_config, dict):
            raise ValueError(f"The configuration {args.config} is not a mapping of settings.")
        from src.utils.checkpoint_paths import refuse_retired_locations

        refuse_retired_locations(yaml_config, source=f"The configuration {args.config}")
        unknown = sorted(str(key) for key in yaml_config if not hasattr(config, str(key)))
        if unknown:
            raise ValueError(f"The configuration {args.config} names settings this version "
                             f"does not have: {', '.join(unknown)}.")
        for key, value in yaml_config.items():
            setattr(config, key, value)
            explicit.add(key)

    # Override with command-line arguments (only if explicitly provided)
    if args.data_dir is not None:
        config.data_dir = args.data_dir
    if args.run_name is not None:
        config.run_name = args.run_name
    if args.run_dir is not None:
        config.run_dir = args.run_dir
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
    if getattr(args, "stage", None) is not None:
        config.learning_stage = args.stage
    if getattr(args, "split_seed", None) is not None:
        config.split_seed = args.split_seed
    if getattr(args, "task", None) is not None:
        config.task = args.task
    if getattr(args, "finetune_from", None) is not None:
        config.finetune_from = args.finetune_from
    if getattr(args, "branch_from", None) is not None:
        config.branch_from = args.branch_from
    if getattr(args, "frozen_parts", None) is not None:
        config.frozen_parts = args.frozen_parts

    for name, value in vars(args).items():
        if name in ("config", "debug", "expect_dataset_sha256", "expect_resume_sha256",
                    "expect_parent_sha256") \
                or value is None or value is False:
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
    #: The gene task's functional fixture (`src.kg.task_fixture`), read once
    #: against this manifest reading, for the tasks that train on it.
    task_data: Optional[Any] = None


def read_training_inputs(
    data_dir: Path, *, expected_manifest_sha256: Optional[str] = None,
    with_samples: bool = True, task: Optional[str] = None,
) -> TrainingInputs:
    """Read the manifest, the graph tensors and both sample files once each, and
    verify from those reads (contract M2.1, S8).

    **The dataset the run was started for** (`expected_manifest_sha256`, the
    fingerprint a caller showed; review of 8242b47, P2-2) is a parameter of
    the one manifest read, the first read here: the reader compares it before
    the bytes are decoded (`read_split_manifest(expected_sha256=)`), so a
    workspace republished since it was shown, torn, of another schema or
    without a manifest, is refused as changed (`DatasetChanged`), never
    trained on, and never reported as a manifest that contradicts its files.

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

    The samples are read with the three fields every reader reads
    (`patient_id`, `phenotype_ids`, `disease_id`): a finetune binds a case's
    phenotypes and disease, and the gene and patients-like-me tasks read their
    own fixture with its gene fields (`with_task_data`).

    `with_samples=False` is the base stage's read (README §3): KG link
    prediction trains on the graph alone, so no sample file is read, verified
    or recorded, and an empty training cohort is not a refusal.

    `task`: a task that trains on the gene fixture (`FIXTURE_TASKS`) reads it
    here too, its record against this manifest reading and each file against
    its recorded digest (`src.kg.task_fixture.read_gene_fixture`).
    """
    from src.kg.artifacts import (
        check_unparsed_kg_json,
        read_split_manifest,
        verify_dataset_record,
        verify_graph_reads,
    )
    from src.kg.storage.file_storage import read_graph_artifacts, read_samples

    # The manifest first, and nothing about the workspace before it (`train`
    # and `main` stat nothing), so a dataset that is not the one shown, its
    # directory gone included, is said as changed before any other refusal
    # about its files (P22-S2); its absence is the reader's own refusal.
    manifest = read_split_manifest(data_dir, expected_sha256=expected_manifest_sha256)
    logger.info(f"Dataset fingerprint (split_manifest.json sha256): {manifest.identity.sha256}")
    # The samples before the graph, so a missing split is refused, by the one
    # reader that locates a workspace split and names the splits there are,
    # before the graph tensors are loaded.
    samples = {
        split: read_samples(data_dir, split)
        for split in (TRAINING_SPLITS if with_samples else ())
    }
    graph = read_graph_artifacts(data_dir, map_location="cpu")

    bound = verify_graph_reads(manifest, graph.reads)
    check_unparsed_kg_json(data_dir, manifest)
    dataset_record = verify_dataset_record(manifest, bound["kg"])
    if not with_samples:
        logger.info(f"Graph-only read; node counts: {graph.graph_data['num_nodes_dict']}")
        return TrainingInputs(manifest, graph, samples, False, bound, dataset_record)
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
    return with_task_data(
        TrainingInputs(manifest, graph, samples, with_validation, bound, dataset_record),
        data_dir, task)


def with_task_data(inputs: TrainingInputs, data_dir: Path, task: Optional[str]
                   ) -> TrainingInputs:
    """`inputs` with the gene task's functional fixture, for a task that trains
    on it (`FIXTURE_TASKS`): its record read against `inputs`' manifest reading
    and each file against its recorded digest (`read_gene_fixture`). A model
    bundle run calls it after its parent's stage and task are checked, so a
    parent of another task is refused for that, not for a missing fixture
    (README §8 regression 7: the task filter runs first)."""
    from src.kg.task_fixture import FIXTURE_TASKS, read_gene_fixture

    if task not in FIXTURE_TASKS:
        return inputs

    task_data = read_gene_fixture(data_dir, manifest_sha256=inputs.manifest.identity.sha256)
    logger.info(f"Gene task fixture ({task_data.record.get('label')}): "
                f"{task_data.record.get('counts')}")
    return inputs._replace(task_data=task_data)


def read_resume_checkpoint(requested: Optional[str], device: str, *,
                           expected_sha256: Optional[str] = None
                           ) -> Optional["CheckpointRead"]:
    """The one read of a requested resume parent, not yet compared with this
    run's inputs (`accept_resume_parent`); `None` when nothing was requested.

    **The parent chosen** (`expected_sha256`, `--expect-resume-sha256`: the
    digest of the read the job runner decided on, the one the operator was
    shown; root-cause round, P22-S1) is a parameter of the one read: other
    bytes at the path, or none, are refused as `CheckpointChanged` before they
    are loaded, so the run never continues from, places or records a parent
    that was not chosen.

    **At the top of the run**, beside the workspace checks: `train()` takes two
    steps, this read and then `accept_resume_parent` once the parent's stage
    and task are checked and the task's data is read, before the run
    directories, `config.yaml`, `runtime.json`, the model or any checkpoint
    exist, so a refused parent leaves none of them behind and nothing is
    restored.

    - **A requested parent that does not exist refuses.** The run does not train
      from scratch instead.
    - **Read once** (M2.1), with this site's options (`map_location` the run's
      device, `weights_only=True`: tensors and plain data, which every
      checkpoint `build_checkpoint_payload` writes is). The trainer restores
      from this read's dict and the record's `resume_checkpoint` is this read's
      identity, so the state restored and the digest recorded cannot come from
      two reads.
    """
    from src.utils.checkpoint_io import read_checkpoint

    if not requested:
        if expected_sha256 is not None:
            raise ValueError("--expect-resume-sha256 names a parent, but this run continues "
                             "none (no --resume or resume_from). Nothing was written.")
        return None

    resume_path = Path(requested)
    # Named before the read, so a parent that does not load fails after a line
    # that says which file it was.
    logger.info(f"Reading the resume parent: {resume_path}")
    try:
        parent = read_checkpoint(resume_path, map_location=device, weights_only=True,
                                 expected_sha256=expected_sha256)
    except FileNotFoundError:  # nothing expected; with an expectation, CheckpointChanged
        raise FileNotFoundError(
            f"The resume parent {resume_path} does not exist. It was requested, so "
            "the run does not train from scratch instead; name an existing "
            "checkpoint, or start a new run without --resume."
        ) from None
    return parent


def accept_resume_parent(parent: "CheckpointRead", inputs: TrainingInputs, *,
                         role: str = "resume") -> None:
    """Refuse a resume (or branch) parent unless it was trained on this run's
    inputs (contract M2.3); the parent is `read_resume_checkpoint`'s read.

    **The parent must record every one of this run's workspace roles with the
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
    resume_path = Path(parent.identity.path)
    check_resume_parent(
        parent.checkpoint, inputs, recorded_by=f"The {role} parent {resume_path}"
    )
    logger.info(
        f"The {role} parent accepted: {resume_path} (sha256 {parent.identity.sha256[:12]})"
    )


def plan_from_parent(
    parent: "CheckpointRead", config: TrainConfig, explicit: Collection[str],
    device: str, *, record_version: int,
) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """How this run continues from `parent` (`src.utils.run_plan`), and
    early stopping's best weights when the parent names a file for them.

    `explicit` is the fields the operator set, required: a setting stated at
    its default value is the operator's as much as any other, so `None` is
    refused, never read as "whatever differs from the defaults" (P2-1).

    That file is read once, with the safe loader, and accepted only with the
    SHA-256 the parent recorded for it, which the reader compares before it
    loads the file (`read_checkpoint(expected_sha256=)`): the weights a run
    restores at its end are the ones its record names, or the run is refused.
    """
    from src.utils.run_plan import RunPlanRefused, decoded_sentence, plan_continuation
    from src.utils.checkpoint_io import read_checkpoint

    if explicit is None:
        raise TypeError("plan_from_parent needs the fields the operator set (explicit); "
                        "they are never inferred from the defaults")
    plan = plan_continuation(parent.checkpoint, Path(parent.identity.path), asdict(config),
                             set(explicit), device=device,
                             parent_sha256=parent.identity.sha256,
                             record_version=record_version)
    logger.info(f"{plan.action}.")
    if plan.inherited:
        logger.info("Inherited from the parent: " + ", ".join(
            f"{name}={value!r}" for name, value in sorted(plan.inherited.items())))
    if plan.decoded:
        logger.info(decoded_sentence(plan.decoded))

    best_weights = None
    if (plan.best_weights or {}).get("kind") == "file":
        source = plan.best_weights
        try:
            read = read_checkpoint(source["path"], map_location="cpu", weights_only=True,
                                   expected_sha256=source["sha256"])
        except CheckpointChanged as exc:
            now = (f"the file there now has {exc.found}" if exc.found is not None
                   else "there is no file there now")
            raise RunPlanRefused(
                f"The parent names {source['path'].name} as early stopping's best weights "
                f"with SHA-256 {source['sha256']}; {now}. Put the recorded file back or "
                "choose another checkpoint.") from None
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

    `parent` is the identity of the checkpoint restored (`read_resume_checkpoint`'s
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
        "node_features": inputs.graph.reads.get("node_features"),
        "edge_indices": inputs.graph.reads.get("edge_indices"),
        "num_nodes": inputs.graph.reads.get("num_nodes"),
    }
    # A graph-only read (the base stage) records no sample role.
    if "train" in inputs.samples:
        reads["train_samples"] = inputs.samples["train"].identity
    if inputs.with_validation:
        reads["val_samples"] = inputs.samples["val"].identity
    if inputs.task_data is not None:
        reads["task_fixture"] = inputs.task_data.identity
        reads["task_train_samples"] = inputs.task_data.samples["train"].identity
        reads["task_val_samples"] = inputs.task_data.samples["val"].identity
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


#: Fields of `TrainConfig` that say where files are, not how training runs.
#: Left out of the checkpoint's record of the configuration requested: the inputs
#: are identified by digest (`training_input_digests`), the parent by its digest,
#: and a checkpoint travels without the layout of the machine that wrote it.
LOCATION_FIELDS = frozenset({"data_dir", "run_dir", "config_file", "resume_from",
                             "finetune_from", "branch_from"})


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
LEARNING_STAGES_HERE = ("base", "finetune")
FINETUNE_TASKS_HERE = ("disease", "gene", "patients_like_me")


def bind_task_data(task: str, spec: Any, view: Any, inputs: TrainingInputs
                   ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """The task's cases bound to the view (F7), per split, and the derived
    tables the gene task scores with (README §6, F10), before any output."""
    derived: Dict[str, Any] = {}
    if task == "disease":
        from src.models.tasks.scoring import bind_disease_cases

        binding = {split: bind_disease_cases(view, inputs.samples[split].samples)
                   for split in ("train", "val")}
    elif task == "patients_like_me":
        from src.models.tasks.patients_like_me import bind_patients

        binding = {split: bind_patients(view, inputs.task_data.samples[split].samples)
                   for split in ("train", "val")}
    else:
        from src.models.tasks.gene import bind_gene_cases, distance_table, neighbour_table

        binding = {split: bind_gene_cases(view, inputs.task_data.samples[split].samples)
                   for split in ("train", "val")}
        phenotypes = [c.phenotypes for split in ("train", "val") for c in binding[split].cases]
        if phenotypes:
            derived["distances"] = distance_table(view, torch.cat(phenotypes),
                                                  view.type_nodes("gene"))
        gene_spec = spec.task_head.gene
        derived["neighbours"] = (neighbour_table(view, gene_spec.augmentation_k)
                                 if gene_spec.augmentation else None)
    if not binding["train"].cases:
        raise ValueError(f"no training case binds to the graph view: {binding['train'].counts}")
    return binding, derived


def _stage_rule_sources(stage: str, task: Optional[str]) -> Dict[str, str]:
    """The settings of a new run that its stage's rule fills when unset, not a
    recipe value or a build default, each with that rule as its source: the
    model spec, the stage's reference spec (a finetune's: its base's graph
    side with its task's reference head), recorded whole in the model record;
    and a finetune's split seed, its base's (README G4), recorded in its data
    roles. The training process fills them (`train`)."""
    if stage != "finetune":
        return {"model_spec": "the base stage's reference spec (spec.reference_spec), "
                              "recorded whole in the model record"}
    return {"model_spec": f"its base's graph view, input, encoder and LP head with the {task} "
                          "task's reference head (spec.finetune_spec_from_parent), recorded "
                          "whole in the model record",
            "split_seed": "its base's split seed: a finetune trains on its base's split "
                          "(README G4), recorded in data_roles.split"}


def apply_stage_recipe(config: TrainConfig, explicit: Set[str]
                       ) -> Tuple[TrainConfig, Dict[str, Any]]:
    """Fill every setting a new run reads that the operator left unset, by the
    one fill of a new run (`run_plan.new_run_values`): the stage's recipe
    (`src.config.recipe`), else this build's default; and record each
    setting's source (decision S2), every setting the run reads
    (`run_plan.new_run_settings`) in exactly one of: `operator_set`, every
    one the operator set, whatever its value; `reference_filled`, each one
    the stage's reference filled, with its source (the recipe's, and the
    model spec and a finetune's split seed by their stage's rule,
    `_stage_rule_sources`); `build_defaults`, every other one, with this
    build's default. Informational: no reader decides by it. A run whose
    settings come from its parent is not filled (`inherited_recipe_record`)."""
    from dataclasses import replace

    from src.config.recipe import reference_settings
    from src.utils.run_plan import new_run_settings, new_run_values

    stage, task = config.learning_stage, config.task
    settings = new_run_settings(stage, task)
    references, sources = reference_settings(stage, task)
    sources = {**sources, **_stage_rule_sources(stage, task)}
    given = {name: getattr(config, name) for name in settings if name in explicit}
    values = new_run_values(stage, task, given)
    record = {
        "learning_stage": stage,
        "reference_filled": {name: sources[name] for name in sorted(sources)
                             if name in settings and name not in given},
        "operator_set": sorted(given),
        "build_defaults": {name: values[name] for name in sorted(
            settings - set(sources) - set(given))},
    }
    return replace(config, **values), record


def inherited_recipe_record(stage: str, task: Optional[str], explicit: Set[str],
                            inherited: Mapping[str, Any],
                            decoded: Mapping[str, str]) -> Dict[str, Any]:
    """The recipe record of a run whose settings come from its parent (a
    resume, an extension, a branch): nothing is filled from the recipe or a
    build default; each setting the run reads (`run_plan.new_run_settings`)
    is the parent's or the operator's. `decoded` is each
    setting the one settings decode gave that the parent's record does not
    hold (`run_plan.decode_learning_settings`), with its source in words
    ("none, decoded from rules 1: ..."), listed there and not as inherited
    (one source per setting); the parent's file is not changed."""
    from src.utils.run_plan import new_run_settings

    settings = new_run_settings(stage, task)
    return {"learning_stage": stage, "reference_filled": {}, "build_defaults": {},
            "inherited_from_parent": sorted(name for name in settings if name in inherited),
            "operator_set": sorted(name for name in settings if name in explicit),
            "decoded_from_parent": dict(sorted(decoded.items()))}


def _read_bundle_parent(requested: str, role: str, device: str, *,
                        expected_sha256: Optional[str] = None):
    """One read of a parent checkpoint and its model record (contract §9.2),
    compared with `expected_sha256` before it is loaded."""
    from src.models.bundle.record import ModelRecordError, read_model_record
    from src.utils.checkpoint_io import read_checkpoint
    from src.utils.run_plan import RunPlanRefused

    path = Path(requested)
    logger.info(f"Reading the {role} parent: {path}")
    try:
        parent = read_checkpoint(path, map_location=device, weights_only=True,
                                 expected_sha256=expected_sha256)
    except FileNotFoundError:
        raise FileNotFoundError(f"The {role} parent {path} does not exist.") from None
    try:
        record = read_model_record(parent.checkpoint)
    except ModelRecordError as exc:
        raise RunPlanRefused(f"The {role} parent {path.name}: {exc}") from None
    if parent.checkpoint.get("state_dict") is None:
        raise RunPlanRefused(f"The {role} parent {path.name} holds no model weights.")
    return parent, record


def read_finetune_parent(requested: str, inputs: TrainingInputs, device: str, *,
                         expected_sha256: Optional[str] = None):
    """Read a finetune's parent once and check it (provenance contract §9.2):
    a base checkpoint with a model spec, trained on this workspace's graph. A
    training state, early stopping's best weights and a weights-only file are
    all accepted; a finetune checkpoint is refused (children never chain), and
    so is a checkpoint without a model spec (nothing is guessed). Patient data
    may differ from the parent's: only the graph is compared."""
    from src.models.bundle.record import finetune_parent_problem
    from src.utils.run_plan import RunPlanRefused

    parent, record = _read_bundle_parent(requested, "finetune", device,
                                         expected_sha256=expected_sha256)
    path = Path(requested)
    problem = finetune_parent_problem(record, parent.checkpoint)
    if problem:
        raise RunPlanRefused(f"The finetune parent {path.name} {problem}")
    if record.data_roles.get("kg") != inputs.bound.get("kg"):
        raise RunPlanRefused(
            f"The finetune parent {path.name} was trained on another knowledge graph; its "
            "learned node table belongs to that graph (decision 1).")
    return parent, record


def read_branch_parent(requested: str, stage: str, task: Optional[str], device: str, *,
                       expected_sha256: Optional[str] = None):
    """Read a branch's parent once and check its stage and task, before any
    task data is read (contract §9.2): a checkpoint of this run's stage and
    task, with a model spec and weights. Its data, settings and spec are
    compared once the inputs are read (`accept_resume_parent`, `plan_branch`)."""
    from src.utils.run_plan import RunPlanRefused

    parent, record = _read_bundle_parent(requested, "branch", device,
                                         expected_sha256=expected_sha256)
    if record.learning_stage != stage or record.task != task:
        raise RunPlanRefused(
            f"The branch parent is a {record.learning_stage} checkpoint"
            + (f" for {record.task}" if record.task else "")
            + "; a branch keeps its parent's stage and task.")
    return parent, record


def train(config: TrainConfig, explicit: Collection[str], *,
          expected_dataset_sha256: Optional[str] = None,
          expected_resume_sha256: Optional[str] = None,
          expected_parent_sha256: Optional[str] = None,
          on_claim: Optional[Callable[[Path], None]] = None) -> Dict[str, float]:
    """Train a model bundle: `learning_stage` "base" or "finetune" (README §3,
    §4; provenance contract §9). The one training entry point.

    Every refusal comes before the run directory is claimed, one read per
    workspace input; the graph view and its fact split are formed from the
    graph read (G1, F2), the model is the one builder's bundle from a model
    spec, the stage's task module drives the steps, the Trainer owns the
    optimizer, AMP, accumulation, callbacks and saving, and every checkpoint
    carries the model record.

    Args:
        config: the requested configuration. Its `learning_stage` is required:
            a run without one is refused before anything is read or written.
        explicit: the fields the operator set (`load_config_with_explicit`),
            required. A setting stated at its default value is the operator's
            as much as any other (P2-1); `None` is refused (a TypeError), never
            read as "whatever differs from the defaults".
        expected_dataset_sha256: the dataset fingerprint the run was started
            for (`--expect-dataset-sha256`), compared on the run's one manifest
            read (`read_training_inputs`); `DatasetChanged` otherwise. A
            precondition, not a setting: neither recorded nor configured.
        expected_resume_sha256: the resume parent's SHA-256 the run was
            started for (`--expect-resume-sha256`), compared on the parent's
            one read before it is loaded (`read_resume_checkpoint`);
            `CheckpointChanged` otherwise. A precondition too.
        expected_parent_sha256: the finetune or branch parent's SHA-256
            (`--expect-parent-sha256`), compared on its one read likewise.
        on_claim: called with the run directory once it is claimed, the run's
            first artifact. Every refusal comes before it, so a caller tells a
            refusal (nothing written) from a failure of a run that started
            (the command line's `_train_main`).

    Returns:
        The final validation metrics; empty for a run that was stopped.
    """
    from dataclasses import asdict as _asdict

    from src.kg.graph_view import SplitMismatchError, build_graph_view, rebuild_split, split_facts
    from src.models.bundle import ModelSpec, build_bundle, finetune_spec_from_parent, reference_spec
    from src.models.bundle.spec import hops_problem, run_spec_problem
    from src.models.bundle.record import (
        EXPOSURE_FULL_VIEW, EXPOSURE_FULL_VIEW_SP, EXPOSURE_TRAIN_CASES, EXPOSURE_TRAIN_EDGES,
        EXPOSURE_TRAIN_EDGES_DRAWN_PAIRS_REMOVED, EXPOSURE_VAL_CASES, EXPOSURE_VAL_EDGES,
        MODEL_RECORD_VERSION, ModelRecord, ModelRecordError, read_model_record,
    )
    from src.config.recipe import BASE_FIXED, FROZEN_PARTS, frozen_parts_problem
    from src.models.bundle.builder import DOWNSTREAM_PARTS, GRAPH_SIDE_PARTS
    from src.training.lp_task import BASE_SELECTION_METRIC, BaseLPSampler, BaseLPTask
    from src.utils.checkpoint_paths import (
        check_run_dir_free, claim_run_dir, propose_run_dir, run_checkpoint_dir, run_log_dir,
    )
    from src.utils.run_plan import (
        RUN_SETTINGS, RunPlanRefused, decoded_sentence, finetune_split_seed_problem,
        new_run_problem, plan_branch, unread_problem,
    )

    if explicit is None:
        raise TypeError("train() needs the fields the operator set (explicit); they are "
                        "never inferred from the defaults")
    stage = config.learning_stage
    if stage not in LEARNING_STAGES_HERE:
        raise ValueError(f"A run names its learning stage (learning_stage, --stage): one of "
                         f"{list(LEARNING_STAGES_HERE)}, not {stage!r}.")
    explicit_set = set(explicit)
    if stage == "base" and (config.task is not None or config.finetune_from):
        raise ValueError("a base run has no task and no finetune parent")
    parents = [name for name in ("resume_from", "finetune_from", "branch_from")
               if getattr(config, name)]
    if len(parents) > 1:
        raise ValueError(f"{' and '.join(parents)} name two parents; a run continues one "
                         "(resume_from), finetunes a base (finetune_from) or branches one "
                         "(branch_from)")
    if stage == "finetune":
        if config.task not in FINETUNE_TASKS_HERE:
            raise ValueError(f"a finetune names its task, one of {list(FINETUNE_TASKS_HERE)} "
                             f"in this build (got {config.task!r})")
        if not parents:
            raise ValueError("a finetune run starts from a base (finetune_from), resumes a "
                             "finetune run (resume_from) or branches one (branch_from)")
    # The settings the operator stated, with their values: the explicit
    # settings a run may state (`run_plan.RUN_SETTINGS`; its stage and task
    # name it, and the operational fields are not settings).
    stated = {name: getattr(config, name) for name in sorted(explicit_set & RUN_SETTINGS)}
    if not (config.resume_from or config.branch_from):
        # A new run (a base, a finetune), by the one check of the settings a
        # new run is started with (`run_plan.new_run_problem`), which the job
        # runner's starts make too, in the same sentences: the freeze, the
        # settings its stage and task do not read, each value by its field
        # spec, and the rules across settings (decision S3) for what the
        # operator set. The freeze it trains with is judged first, set or not,
        # so its value is final here (a finetune's recipe value is "none").
        problem = (frozen_parts_problem(stage, config.frozen_parts)
                   or new_run_problem(stage, (config.task,), stated))
    else:
        # A continued or branched run trains with its parent's settings, which
        # its plan checks (`run_plan`: a different one stated is refused, a
        # branch's changes judged by the rules across settings, the freeze by
        # the settings decode), as the job runner's preview does; it reads the
        # settings of its stage and task alone (`unread_problem`).
        problem = unread_problem(stage, (config.task,), stated)
    if problem:
        raise ValueError(problem)
    if expected_parent_sha256 is not None and not (config.finetune_from or config.branch_from):
        raise ValueError("--expect-parent-sha256 names a finetune or branch parent; give "
                         "finetune_from or branch_from.")
    # A run with a parent to inherit from (resume, extend, branch) takes the
    # parent's settings, recorded as inherited once the plan is known; the
    # recipe fills only a run that starts anew (a base, a finetune).
    recipe = None
    if not (config.resume_from or config.branch_from):
        config, recipe = apply_stage_recipe(config, explicit_set)

    device = resolve_training_device(config.device)
    logger.info(f"Model bundle run, learning stage {stage}"
                + (f", task {config.task}" if config.task else "")
                + f"; device {device} (requested: {config.device})")
    data_dir = Path(config.data_dir)
    if config.run_dir:
        check_run_dir_free(config.run_dir, data_dir=config.data_dir)

    inputs = read_training_inputs(data_dir, expected_manifest_sha256=expected_dataset_sha256,
                                  with_samples=(stage == "finetune"))

    # The parent: a run of the same stage to resume or extend, or a base to
    # finetune from. Both are read once and checked before any output: the
    # stage and task first, then the task's data is read, then a resume parent
    # is compared with every input (README §8 regression 7).
    parent = read_resume_checkpoint(config.resume_from, device,
                                    expected_sha256=expected_resume_sha256)
    parent_record = None
    plan = None
    best_weights = None
    if parent is not None:
        try:
            parent_record = read_model_record(parent.checkpoint)
        except ModelRecordError as exc:
            raise RunPlanRefused(f"The parent {Path(parent.identity.path).name}: {exc}") from None
        if parent_record.learning_stage != stage or parent_record.task != config.task:
            raise RunPlanRefused(
                f"The parent is a {parent_record.learning_stage} checkpoint"
                + (f" for {parent_record.task}" if parent_record.task else "")
                + "; a run continues only its own stage and task.")
    elif (config.extend_epochs or config.extend_lr is not None) and not config.branch_from:
        # A branch with them is refused by its plan, in a branch's words.
        raise ValueError("extend_epochs and extend_lr continue a parent; give resume_from.")
    base_parent = base_record = None
    if config.finetune_from:
        base_parent, base_record = read_finetune_parent(
            config.finetune_from, inputs, device, expected_sha256=expected_parent_sha256)
    branch_parent = branch_record = branch_plan = None
    if config.branch_from:
        branch_parent, branch_record = read_branch_parent(
            config.branch_from, stage, config.task, device,
            expected_sha256=expected_parent_sha256)
    if stage == "finetune":
        inputs = with_task_data(inputs, data_dir, config.task)
    if parent is not None:
        accept_resume_parent(parent, inputs)
        plan, best_weights = plan_from_parent(parent, config, explicit_set, device,
                                              record_version=MODEL_RECORD_VERSION)
        config = TrainConfig(**plan.settings)
        recipe = inherited_recipe_record(stage, config.task, explicit_set, plan.inherited,
                                         plan.decoded)
    if branch_parent is not None:
        # The branch's data is its parent's (BACKLOG 23, M2.3's rule), its
        # settings the parent's but the changes on the stage's list.
        accept_resume_parent(branch_parent, inputs, role="branch")
        branch_plan = plan_branch(branch_parent.checkpoint, Path(branch_parent.identity.path),
                                  _asdict(config), explicit_set, device=device,
                                  record_version=MODEL_RECORD_VERSION)
        logger.info(f"{branch_plan.action}.")
        if branch_plan.decoded:
            logger.info(decoded_sentence(branch_plan.decoded))
        config = TrainConfig(**branch_plan.settings)
        recipe = inherited_recipe_record(stage, config.task, explicit_set,
                                         branch_plan.inherited, branch_plan.decoded)

    # The spec: the operator's (read strictly), the parent's (inherited with
    # the settings), the base's graph side with the task's reference head, or
    # the stage's reference.
    if config.model_spec is not None:
        spec = ModelSpec.from_dict(config.model_spec)
    elif base_record is not None:
        spec = finetune_spec_from_parent(base_record.model_spec, config.task)
    else:
        spec = reference_spec(stage, config.task)
    # Its stage and task, and a finetune's graph side its base's, by the one
    # check the job runner's starts make too (`run_spec_problem`).
    problem = run_spec_problem(spec, stage, config.task,
                               base=base_record.model_spec if base_record is not None else None)
    if problem:
        raise ValueError(problem)
    kept_from = parent_record or branch_record
    if kept_from is not None and spec != kept_from.model_spec:
        raise ValueError("model_spec differs from the parent's; a continued or branched run "
                         "keeps its parent's architecture")
    hops = hops_problem(config.num_neighbors, spec.encoder)
    if hops:
        raise ValueError(hops)
    config.model_spec = spec.to_dict()

    # The view and its split, from the graph read (G1, F2), before any output.
    # A finetune trains on its base's split (G4): the seed is the base's.
    lineage = parent_record or branch_record or base_record
    if base_record is not None:
        base_split = base_record.data_roles.get("split") or {}
        if "split_seed" in explicit_set:
            problem = finetune_split_seed_problem(config.split_seed, base_split)
            if problem:
                raise ValueError(problem)
        config.split_seed = int(base_split.get("seed"))
    view_rule = _asdict(spec.graph_view)
    view_rule["exclude_relations"] = list(view_rule["exclude_relations"])
    view = build_graph_view(inputs.graph.graph_data, rule=view_rule)
    if lineage is None:
        split = split_facts(view, config.split_seed)
    else:
        # The parent's recorded split, rebuilt from this view and checked field
        # by field, this run's seed included (`rebuild_split`); the field that
        # differs is the refusal's cause, the sentence stays the same.
        recorded = lineage.data_roles
        mismatch: Optional[SplitMismatchError] = None
        try:
            split = rebuild_split(view, recorded.get("split") or {}, seed=config.split_seed)
        except SplitMismatchError as exc:
            mismatch = exc
        if mismatch is not None or (recorded.get("graph_view") or {}).get("digest") != view.digest:
            raise ValueError("the graph view or edge split differs from the parent's record; "
                             "a child trains on its parent's view and split") from mismatch
    train_mask, val_mask = split.mask("train"), split.mask("val")

    # The task data, bound to the view (F7), and its derived tables, before
    # any output.
    binding: Dict[str, Any] = {}
    derived: Dict[str, Any] = {}
    if stage == "finetune":
        binding, derived = bind_task_data(config.task, spec, view, inputs)

    from src.utils.run_environment import code_revision, collect_environment

    producer = code_revision(PROJECT_ROOT)
    environment = collect_environment(device)
    run_dir = Path(config.run_dir) if config.run_dir else propose_run_dir(
        config.data_dir, spec.encoder.kind, config.run_name)
    checkpoint_dir = run_checkpoint_dir(run_dir)
    log_dir = run_log_dir(run_dir)

    if config.seed is not None:
        apply_seed(config.seed)
    bundle = build_bundle(spec, view)
    # The freeze (`frozen_parts`, its value final here: a new run's checked
    # above, a continued or branched run's its parent's, through the settings
    # decode), before any weight is loaded and before the Trainer builds its
    # optimizer, which then holds no frozen parameter; a frozen part runs in
    # eval for the whole run, whatever calls `train()` (`ModelBundle.freeze`).
    # A resume restores the optimizer state of the same groups.
    bundle.freeze(FROZEN_PARTS[config.frozen_parts])
    # The stage's weight transfer: how the weights it trains began. A new
    # base has none; a finetune and a branch make theirs here; a resume and an
    # extension continue their parent's stage, whose freeze they keep (the
    # settings decode checked that transfer against it), and record its
    # transfer unchanged, so every checkpoint of a stage names the downstream
    # weights it started from in its own record (research comparison Stage 9b
    # fixes: two arms compared after a pause are checked where they are
    # listed). A parent that records none passes none on.
    weight_transfer = (copy.deepcopy(parent_record.weight_transfer)
                       if parent_record is not None else None)
    if base_parent is not None:
        weight_transfer = _transfer_parent_weights(
            bundle, base_parent.checkpoint["state_dict"], new_parts=DOWNSTREAM_PARTS)
    if branch_parent is not None:
        weight_transfer = _transfer_parent_weights(
            bundle, branch_parent.checkpoint["state_dict"], new_parts=())
    seed = int(config.seed or 0)
    # Both graph inputs carry the view's own node types (research-comparison
    # README §3, F6): each step's sampled nodes' and, in validation, every
    # view node's; a spec whose input adds a type embedding reads them.
    lp_sampler = BaseLPSampler(
        view.num_nodes, view.edge_index(train_mask), view.rel[train_mask],
        view.edge_index(train_mask), view.rel[train_mask], sizes=config.num_neighbors,
        batch_size=config.batch_size, seed=seed, node_type=view.node_type)
    if stage == "base":
        sampler = lp_sampler
        # The validation pairs are the split's (BASE_FIXED): seeded by the
        # split's seed, in fixed batches, whatever this run's seed and batch
        # size, so runs over one split, branches among them, compare on one set.
        task = BaseLPTask(
            num_nodes=view.num_nodes, train_edges=view.edge_index(train_mask),
            train_relations=view.rel[train_mask], val_edges=view.edge_index(val_mask),
            val_relations=view.rel[val_mask], seed=int(config.split_seed),
            val_batch_size=BASE_FIXED["validation_batch_size"], node_type=view.node_type)
        monitor = BASE_SELECTION_METRIC
    elif config.task == "disease":
        from src.training.finetune_task import (
            FINETUNE_SELECTION_METRIC, DiseaseTask, FinetuneSampler,
        )

        sampler = FinetuneSampler(
            view, binding["train"].cases, lp_sampler, batch_size=config.batch_size,
            candidate_sample_size=config.candidate_sample_size,
            sparse_sample_size=config.sparse_sample_size, seed=seed)
        task = DiseaseTask(
            view=view, val_cases=binding["val"].cases, val_excluded=binding["val"].excluded_outcomes(),
            full_edges=view.edge_index(), full_relations=view.rel,
            task_loss_weight=config.task_loss_weight, lp_loss_weight=config.lp_loss_weight)
        monitor = FINETUNE_SELECTION_METRIC
    elif config.task == "patients_like_me":
        from src.training.finetune_task import (
            FINETUNE_SELECTION_METRIC, PatientsSampler, PatientsTask,
        )

        sampler = PatientsSampler(
            view, binding["train"].cases, lp_sampler, batch_size=config.batch_size,
            upsample_size=config.candidate_upsample_size,
            sparse_sample_size=config.sparse_sample_size, seed=seed)
        task = PatientsTask(
            view=view, val_cases=binding["val"].cases, val_excluded=binding["val"].excluded_outcomes(),
            val_unranked=binding["val"].unranked_positives,
            full_edges=view.edge_index(), full_relations=view.rel,
            task_loss_weight=config.task_loss_weight, lp_loss_weight=config.lp_loss_weight)
        monitor = FINETUNE_SELECTION_METRIC
    else:
        from src.training.finetune_task import FINETUNE_SELECTION_METRIC, GeneSampler, GeneTask

        sampler = GeneSampler(
            view, binding["train"].cases, lp_sampler, derived["neighbours"],
            batch_size=config.batch_size, upsample_size=config.candidate_upsample_size,
            sparse_sample_size=config.sparse_sample_size, seed=seed,
            augmentation_k=spec.task_head.gene.augmentation_k)
        task = GeneTask(
            view=view, val_cases=binding["val"].cases, val_excluded=binding["val"].excluded_outcomes(),
            full_edges=view.edge_index(), full_relations=view.rel,
            distances=derived["distances"], neighbours=derived["neighbours"],
            task_loss_weight=config.task_loss_weight, lp_loss_weight=config.lp_loss_weight)
        monitor = FINETUNE_SELECTION_METRIC

    compile_installed = False
    if config.compile:
        try:
            import torch._dynamo as _dynamo
            _dynamo.config.suppress_errors = True
            bundle.compile()
            compile_installed = True
        except Exception as exc:  # never let compile break training
            logger.warning(f"torch.compile failed to enable; using eager: {exc}")

    # Early stopping is disabled in the reference (C6: no patience); a patience
    # in effect turns it on, the operator's or the parent's, so a continued
    # run keeps early stopping on or off as its parent had it.
    early_stopping = config.early_stopping_patience is not None
    trainer_config = TrainerConfig(
        num_epochs=config.num_epochs, learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_grad_norm=config.max_grad_norm, scheduler_type=config.scheduler_type,
        warmup_steps=config.warmup_steps, min_lr_ratio=config.min_lr_ratio,
        use_amp=config.use_amp, amp_dtype=config.amp_dtype,
        eval_every_n_epochs=config.eval_every_n_epochs, early_stopping=early_stopping,
        early_stopping_patience=(int(config.early_stopping_patience) if early_stopping else 0),
        early_stopping_monitor=monitor, early_stopping_mode="max",
        checkpoint_dir=str(checkpoint_dir), save_top_k=config.save_top_k,
        log_dir=str(log_dir), device=device, seed=config.seed,
        pause_after_epoch=config.pause_after_epoch,
    )
    trainer = Trainer(model=bundle, train_dataloader=sampler, config=trainer_config, task=task)
    if parent is not None:
        trainer.continue_from(parent.checkpoint, plan, best_weights)
        logger.info(f"Continuing from checkpoint: {parent.identity.path} "
                    f"(sha256 {parent.identity.sha256[:12]})")
    del best_weights

    claim_run_dir(run_dir, data_dir=config.data_dir)
    if on_claim is not None:
        on_claim(run_dir)
    logger.info(f"Run directory: {run_dir}")
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump({**asdict(config), "run_dir": str(run_dir)}, f)
    with open(run_dir / "runtime.json", "w") as f:
        json.dump(_allocator_record(), f, indent=2)
    # The view's record and the split artifact (F2), beside the run's outputs.
    with open(run_dir / "graph_view.json", "w") as f:
        json.dump({"source": {"kg": inputs.bound["kg"]}, **view.record()}, f, indent=2,
                  sort_keys=True)
    torch.save({"record": split.record(), "part": split.part}, run_dir / "graph_split.pt")

    parent_identity = parent.identity if parent is not None else None
    base_identity = base_parent.identity if base_parent is not None else None
    branch_identity = branch_parent.identity if branch_parent is not None else None
    base_logs = ({k: v for k, v in (base_parent.checkpoint.get("logs") or {}).items()}
                 if base_parent is not None else {})
    del parent, base_parent, branch_parent

    def _ref(identity):
        return {"sha256": identity.sha256, "run": Path(identity.path).parent.parent.name}

    if parent_record is not None:
        operation, family_id, parent_ref = plan.mode, parent_record.family_id, _ref(parent_identity)
    elif base_record is not None:
        operation, family_id, parent_ref = "finetune", base_record.family_id, _ref(base_identity)
    elif branch_record is not None:
        # Every change, from the parent's value to the new one (BACKLOG 23),
        # beside the exact parent file.
        operation, family_id = "branch", branch_record.family_id
        parent_ref = {**_ref(branch_identity), "changes": branch_plan.changes}
    else:
        operation, family_id, parent_ref = "new", run_dir.name, None
    data_roles: Dict[str, Any] = {
        "kg": inputs.bound["kg"],
        "graph_view": {"digest": view.digest, "rule": dict(view.rule)},
        "split": split.record(),
    }
    # The record's graph exposure, in the words `src.models.bundle.record`
    # names once for this writer and its readers (`recorded_graph_use`).
    if stage == "base":
        exposure = {"base": {
            "training": {"message_passing": EXPOSURE_TRAIN_EDGES_DRAWN_PAIRS_REMOVED,
                         "supervision": EXPOSURE_TRAIN_EDGES},
            "validation": {"message_passing": EXPOSURE_TRAIN_EDGES,
                           "supervision": EXPOSURE_VAL_EDGES}}}
        loss_identity = {"terms": ["lp"], "lp_loss_weight": 1.0, "lp_head": spec.lp_head.kind,
                         "margin": spec.lp_head.margin}
        scoring = None
        selection = None
    else:
        task_rows = inputs.task_data.samples if inputs.task_data is not None else inputs.samples
        data_roles["task_data"] = {
            split_name: {"samples": task_rows[split_name].identity.sha256,
                         "binding": dict(binding[split_name].counts)}
            for split_name in ("train", "val")}
        exposure = {"finetune": {
            "training": {"message_passing": EXPOSURE_TRAIN_EDGES_DRAWN_PAIRS_REMOVED,
                         "lp_supervision": EXPOSURE_TRAIN_EDGES, "task": EXPOSURE_TRAIN_CASES},
            "validation": {"message_passing": EXPOSURE_FULL_VIEW, "task": EXPOSURE_VAL_CASES}}}
        # Both terms at the recipe's coefficients, frozen or not (owner's O4a);
        # a term every part of whose parameters is frozen trains nothing, said
        # so here, never dropped: the LP term reads the graph side alone, the
        # task term every part.
        untrained = [term for term, reads in (("task", bundle.parts), ("lp", GRAPH_SIDE_PARTS))
                     if set(reads) <= set(bundle.frozen)]
        loss_identity = {"terms": ["task", "lp"], "task_loss": "nca",
                         "task_loss_weight": float(config.task_loss_weight),
                         "lp_loss_weight": float(config.lp_loss_weight),
                         "lp_head": spec.lp_head.kind, "margin": spec.lp_head.margin,
                         "terms_without_trainable_parameters": untrained}
        scoring = {"task": config.task, "score": "-||z_d - z_P||^2 on L2-normalised vectors",
                   "candidates": "every disease of the graph view", "ties": "average rank",
                   "version": 1}
        if inputs.task_data is not None:
            record = inputs.task_data.record
            data_roles["task_fixture"] = {"record": inputs.task_data.identity.sha256,
                                          "label": record.get("label"),
                                          "rule": record.get("rule"), "seed": record.get("seed")}
        if config.task == "patients_like_me":
            scoring = {"task": "patients_like_me",
                       "score": "-||z_a - z_b||^2 of the patient tower's L2-normalised vectors",
                       "positives": "the two patients' causal-gene sets intersect (F9)",
                       "candidates": "every other patient of the reference set; self excluded "
                                     "by case id",
                       "ties": "average rank", "version": 1}
        if config.task == "gene":
            from src.models.bundle.spec import GeneScoringSpec
            from src.models.tasks.gene import SP_RULE

            gene_spec = spec.task_head.gene
            distances = derived["distances"]
            data_roles["derived_tables"] = {
                "sp_distances": {"digest": distances.digest, "rule": SP_RULE,
                                 "phenotypes": int(distances.table.shape[0]),
                                 "genes": int(distances.table.shape[1])},
                "augmentation_neighbours": (derived["neighbours"].record
                                            if derived["neighbours"] is not None else None)}
            exposure["finetune"]["sp"] = EXPOSURE_FULL_VIEW_SP
            alpha, beta, base, ms_margin = spec.task_head.multi_similarity
            loss_identity.update({"task_loss": "multi_similarity", "multi_similarity": {
                "alpha": alpha, "beta": beta, "base": base, "mining_margin": ms_margin}})
            scoring = {
                "task": "gene", "score": "eta*EMBSIM + (1 - eta)*SPLSIM",
                "embsim": "tanh(p^T W g + b) on L2-normalised projected vectors",
                "sp": SP_RULE, "sp_lower": gene_spec.sp_lower, "sp_upper": gene_spec.sp_upper,
                "sp_bounds_source": (
                    "upstream fixed reference, not calibrated on this data"
                    if (gene_spec.sp_lower, gene_spec.sp_upper) == (
                        GeneScoringSpec().sp_lower, GeneScoringSpec().sp_upper)
                    else "operator-set (model_spec), not the upstream reference"),
                "eta": gene_spec.eta,
                "augmentation": ({"k": gene_spec.augmentation_k,
                                  "omega": gene_spec.augmentation_omega}
                                 if gene_spec.augmentation else "off"),
                "candidates": ("training: the batch's union of its cases' lists, upsampled "
                               "(U14); validation, measurement and serving: each case's own "
                               "list"),
                "ties": "average rank", "version": 1}
        selection = ({"rule": "the operator's choice of base checkpoint",
                      "parent_logs": base_logs}
                     if base_record is not None else None)
    trainer.model_record = ModelRecord(
        model_spec=spec, graph_view=bundle.view, operation=operation, family_id=family_id,
        parent=parent_ref, weight_transfer=weight_transfer, data_roles=data_roles,
        graph_exposure=exposure, loss_identity=loss_identity, scoring_identity=scoring,
        selection=selection)
    from src.utils.fingerprint import compute_fingerprint

    trainer.data_fingerprint = compute_fingerprint(inputs.graph.graph_data)
    trainer.run_record = {
        "producer": producer,
        "environment": environment,
        "requested": {k: v for k, v in asdict(config).items() if k not in LOCATION_FIELDS},
        "device_requested": config.device,
        "compile": {"requested": bool(config.compile), "installed": compile_installed},
        "loaders": {
            "train": {"kind": type(sampler).__name__, "batch_size": int(config.batch_size),
                      "sizes": list(config.num_neighbors), "batches": len(sampler),
                      "recipe": recipe},
            "val": ({**task.val_counts, "seed": int(config.split_seed),
                     "batch_size": BASE_FIXED["validation_batch_size"]} if stage == "base"
                    else dict(binding["val"].counts)),
        },
    }
    # The workspace roles; a finetune's base is named in the model record's
    # parent, once, not as a workspace role a resume would compare.
    trainer.training_input_digests = training_input_digests(inputs, parent=parent_identity)

    logger.info("Starting training...")
    final_metrics = trainer.train()
    if trainer.interrupted:
        logger.info("Training stopped; the latest resumable checkpoint is "
                    f"{checkpoint_dir / 'last.pt'}")
        return {}
    with open(run_dir / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    logger.info(f"Training complete: {final_metrics}")
    return final_metrics


def _transfer_parent_weights(bundle: Any, parent_state: Mapping[str, Any], *,
                             new_parts: Collection[str]) -> Dict[str, Any]:
    """Load a parent's weights into a new bundle: every parent parameter must
    exist with the same shape, and only `new_parts` may be absent from the
    parent: a finetune's patient encoder and task head over its base's graph
    side (README A10), nothing for a branch, whose load is strict and complete.

    Returns the weight-transfer record (the model record's `weight_transfer`)
    of the bundle as it will train, its freeze already applied
    (`ModelBundle.freeze`): what was loaded and what initialised new; `trainable`, "all" when no part is frozen
    (as training rules 1 recorded every transfer) or else the parts that
    train; the optimizer, always new; each part, trainable or frozen
    (`parts`), and the frozen parts' forward mode (`frozen_forward`, None when
    none is frozen); and the SHA-256 of the downstream parts' weights as
    training starts (`downstream_initial_sha256`, by `downstream_digest_rule`;
    None for a base's branch, which has none), so two runs from one base can
    be checked to start from the same downstream weights, not inferred from a
    seed (research comparison Stage 9, owner's O4a)."""
    from src.models.bundle.builder import DOWNSTREAM_PARTS, FROZEN_FORWARD, PARTS_DIGEST_RULE

    own = bundle.state_dict()
    foreign = sorted(k for k in parent_state if k not in own)
    mismatched = sorted(k for k in parent_state
                        if k in own and tuple(own[k].shape) != tuple(parent_state[k].shape))
    if foreign or mismatched:
        raise ValueError(f"the parent's weights do not fit this model "
                         f"(unknown {foreign[:5]}, shape {mismatched[:5]})")
    missing = [k for k in own if k not in parent_state]
    absent = sorted({k.split(".")[0] for k in missing})
    if any(part not in new_parts for part in absent):
        raise ValueError(f"the parent lacks weights of {absent}")
    bundle.load_state_dict(dict(parent_state), strict=not new_parts)
    loaded = sorted({k.split(".")[0] for k in parent_state})
    frozen = bundle.frozen
    downstream = [part for part in DOWNSTREAM_PARTS if part in bundle.parts]
    return {"loaded_from_parent": loaded, "initialised_new": absent,
            "trainable": [part for part in bundle.parts if part not in frozen] if frozen else "all",
            "optimizer": "new",
            "parts": {part: "frozen" if part in frozen else "trainable" for part in bundle.parts},
            "frozen_forward": FROZEN_FORWARD if frozen else None,
            "downstream_initial_sha256": bundle.parts_sha256(downstream) if downstream else None,
            "downstream_digest_rule": PARTS_DIGEST_RULE if downstream else None}


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


#: What `train()` refuses with before it claims its run directory: every
#: refusal it states is a ValueError (its own, `RunPlanRefused`,
#: `TaskFixtureError`, `ModelRecordError`, `SpecError` among them), a requested
#: parent that does not exist, a device this host cannot give, or a run
#: directory that is not free (`RunDirectoryTaken`: it holds files, or another
#: run claimed it first). Raised after the claim, any of them is a failure of a
#: run that started.
TRAINING_REFUSALS = (ValueError, FileNotFoundError, TrainingDeviceUnavailable,
                     RunDirectoryTaken)


def _sentence(exc: BaseException) -> str:
    """An exception's message as one line and sentence: its lines joined by a
    space, ended with a full stop when it has none."""
    text = " ".join(line.strip() for line in str(exc).splitlines() if line.strip())
    text = text or type(exc).__name__
    return text if text.endswith((".", "!", "?")) else f"{text}."


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
    try:
        config, explicit = load_config_with_explicit(args)
    except ValueError as exc:  # a refusal of the configuration, before anything runs
        logger.error(f"train_model: refused: {exc} Nothing was written.")
        return 1

    # Run training. The workspace is checked by its readers, the manifest's
    # first (`read_training_inputs`), each refusal naming the missing file.
    claimed: List[Path] = []
    try:
        metrics = train(config, explicit, expected_dataset_sha256=args.expect_dataset_sha256,
                        expected_resume_sha256=args.expect_resume_sha256,
                        expected_parent_sha256=getattr(args, "expect_parent_sha256", None),
                        on_claim=claimed.append)
        return 0 if metrics else 1
    except DatasetChanged as exc:  # a refusal, said once, not a failure with a traceback
        logger.error(f"train_model: refused: {exc} Nothing was written.")
        return 1
    except CheckpointChanged as exc:  # the resume parent's read, the one `train` makes
        # with an expectation (`--expect-resume-sha256`); the reader states the facts
        logger.error("train_model: refused: The checkpoint changed since it was chosen; "
                     f"choose it again. {exc} Nothing was written.")
        return 1
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except TRAINING_REFUSALS as exc:
        if claimed:  # after the run directory: a run that started and failed
            logger.exception(f"Training failed: {exc}")
            return 1
        # **A refusal of `train()`**, before its run directory, its first
        # artifact (`on_claim`): one line, never a traceback, as the
        # configuration's refusal above.
        logger.error(f"train_model: refused: {_sentence(exc)} Nothing was written.")
        logger.debug("The refusal, as raised:", exc_info=True)
        return 1
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
