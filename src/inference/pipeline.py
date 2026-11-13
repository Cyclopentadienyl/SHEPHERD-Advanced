"""
# ==============================================================================
# Module: src/inference/pipeline.py
# ==============================================================================
# Purpose: End-to-end inference pipeline for rare disease diagnosis
#
# Dependencies:
#   - External: torch (optional, for GNN inference)
#   - Internal: src.core.types, src.kg, src.reasoning, src.models.bundle
#
# Input:
#   - PatientPhenotypes: Patient's phenotype data (HPO IDs)
#   - A workspace: its knowledge graph (read once, with its identity) and the
#     graph tensors and dataset record its manifest binds
#   - A trained model, read from a checkpoint trained on the workspace's graph
#
# Output:
#   - InferenceResult: Complete diagnosis result with candidates and explanations
#
# Design Notes:
#   - Scoring: always by a trained model read from a checkpoint. No pipeline
#     is built without one: a model that is missing or cannot be built raises
#     PipelineBuildError, and a graph is never served on its own
#   - PathReasoner: Explanation-only role (evidence paths, not scoring)
#   - Ortholog support (P1): Cross-species evidence (interfaces preserved)
#   - Interpretable: Full evidence paths and human-readable explanations
#   - Production-ready: Input validation, error handling, logging
#
# Usage:
#   from src.inference import DiagnosisPipeline
#
#   pipeline = DiagnosisPipeline(
#       graph=KnowledgeGraph.read_json("data/workspaces/demo/kg.json"),
#       checkpoint_path="data/workspaces/demo/runs/gatv2/<run>/checkpoints/last.pt",
#       data_dir="data/workspaces/demo",
#   )
#   result = pipeline.run(patient_phenotypes, top_k=10)
# ==============================================================================
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING

from src.kg.artifacts import (
    GRAPH_ARTIFACTS,
    read_split_manifest,
    verify_dataset_record,
    verify_graph_source_read,
)
from src.kg.storage.file_storage import read_graph_artifacts
from src.utils.checkpoint_io import CheckpointChanged, read_checkpoint
from src.utils.fingerprint import ReadOutside
from src.kg.phenotype_normalisation import (
    PHENOTYPE_NORMALISATION_VERSION,
    normalise_phenotypes,
)
from src.core.types import (
    SCORING_MODE_SP_STATES,
    TASK_SCORING_MODES,
    Candidate,
    DiseaseTaskCandidate,
    GeneTaskCandidate,
    PatientCandidate,
    DataSource,
    EvidenceSource,
    InferenceResult,
    NodeID,
    NodeType,
    PatientPhenotypes,
    PhenotypeInputAccount,
    ResultRefused,
    ServedCheckpoint,
    ServedDatasetRecord,
    ServedGraph,
    ServedIdentity,
    ServedSettings,
    ServedShortestPaths,
    candidate_score,
    result_kind_for,
    result_label,
)
from src.reasoning import (
    PathReasoner,
    PathReasoningConfig,
    ReasoningPath,
    DirectPathFinder,
    ExplanationGenerator,
    EvidencePanel,
    EvidencePanelConfig,
    EvidencePackage,
    SCORE_MEANING,
    create_path_reasoner,
    create_explanation_generator,
    create_evidence_panel,
    find_matching_phenotypes,
)
from src.utils.checkpoint_paths import RANKING_SCORE_KEYS

# Optional torch dependency for GNN inference
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if TYPE_CHECKING:
    from src.kg.graph import GraphRead
    from src.utils.fingerprint import ReadIdentity

logger = logging.getLogger(__name__)

#: Training-log metrics copied from a checkpoint into `checkpoint_meta`, in the
#: order the Diagnosis tab shows them -- the ranking metrics in auto-selection's
#: priority order.
#:
#: **The ranking metrics are the ones auto-selection ranks checkpoints by**,
#: `RANKING_SCORE_KEYS`, taken from where that is defined rather than listed
#: again here, so the status line shows the numbers a model was chosen on and a
#: rename cannot leave the two disagreeing. The current trainer writes them: it
#: saves its epoch metrics as the checkpoint's `logs`
#: (`src/training/callbacks.py:_save_checkpoint`), validation metrics prefixed
#: `val_` (`Trainer._validate`).
#:
#: This lookup used to ask for `mrr`, `hits_at_1` and `hits_at_10`, which no
#: trainer in this repository has written: `trainer.py` was empty in the initial
#: scaffold, its first implementation already prefixed `val_`, and `hits_at_1`
#: appears only in the display code that asked for it. So a checkpoint showed its
#: losses and none of its ranking metrics. Those names are not kept as a fallback.
#:
#: The ranking metrics are a trainer's own validation numbers as the checkpoint
#: records them (a stage's task reports its selection metric, `Trainer._validate`
#: prefixes it); `train_loss` is its training loss. `val_loss` is copied when an
#: existing checkpoint's logs hold it; this build's Trainer writes none. Each is
#: copied only when a checkpoint holds it. None of them is a test result.
CHECKPOINT_LOG_METRICS: Tuple[str, ...] = ("val_loss", "train_loss", *RANKING_SCORE_KEYS)


# ==============================================================================
# Configuration
# ==============================================================================
@dataclass
class PipelineConfig:
    """Configuration for the diagnosis pipeline."""

    # Path discovery and path scoring (the evidence paths). PathReasoner is
    # explanation-only and does NOT contribute to scoring.
    max_path_length: int = 4
    path_length_penalty: float = 0.9

    # Output control
    include_explanations: bool = True
    include_ortholog_evidence: bool = True
    include_literature_evidence: bool = True
    max_paths_per_candidate: int = 10

    # Validation. Unknown ids are always dropped and stated, and an input left
    # with none is refused: there is no switch that only silences the warning.
    min_phenotypes: int = 1
    max_phenotypes: int = 100

    # Performance
    use_direct_path_optimization: bool = True

    # P1 Ortholog Configuration
    ortholog_species: List[str] = field(
        default_factory=lambda: ["mouse", "zebrafish", "rat"]
    )
    min_ortholog_confidence: float = 0.5


# ==============================================================================
# Result Types
# ==============================================================================
@dataclass
class ValidationResult:
    """Result of input validation."""

    is_valid: bool
    #: The phenotypes to score: known, distinct, in first-occurrence order.
    validated_phenotypes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    #: Where each validated phenotype sat in the request, so a per-position
    #: field such as `phenotype_confidences` follows the entries that survived.
    kept_positions: List[int] = field(default_factory=list)
    #: None when the request was refused before its phenotypes were mapped.
    phenotype_input: Optional[PhenotypeInputAccount] = None
    #: **The one mapping step's output** (Spec 1 §2.2): each validated phenotype's
    #: graph node and its row in the graph's phenotype node index, in the same
    #: order. `rows` is None when the request was refused.
    node_ids: List[NodeID] = field(default_factory=list)
    rows: Optional[List[int]] = None


@dataclass
class ServedRanking:
    """The served ranking of one input, whole and untruncated (`DiagnosisPipeline.rank`).

    **Everything `run` does up to and including its sort, nothing after it**
    (`docs/working/scorer-measurement/PLAN_B05.md` §2). `run` cuts it at
    `top_k` and builds matching phenotypes, explanations and the summary on what
    it keeps. `top_k` never changes the order, only where the list is cut, so a
    served rank `r` is what a clinician sees whenever the page's Top-K is at
    least `r`.

    `outcome` says which of `run`'s three returns this is: ``"refused"`` (the
    input validation refused it), ``"no_candidates"`` (nothing mapped to a graph
    node, or nothing to rank) or ``"ranked"``. `warnings` is the
    list `run` returns with it, as `run` would return it.
    """

    outcome: str
    warnings: List[str]
    patient_input: PatientPhenotypes
    phenotype_input: Optional[PhenotypeInputAccount] = None
    #: Every served candidate in the served order, ranked 1..N, of the kind the
    #: pipeline serves.
    candidates: List[Candidate] = field(default_factory=list)
    source_ids: List[NodeID] = field(default_factory=list)


@dataclass(frozen=True)
class ReferenceCohort:
    """A registered reference cohort version for patients-like-me (README §8,
    PLAN_TEST_RESULTS §8.4), resolved, read against its digest and checked
    against the served graph by the caller (the API, which may read the
    registry; this layer may not).

    `identity` names it on every result (`label`, `version_sha256`,
    `samples_sha256`); `samples` are its rows with their causal genes. The
    pipeline embeds them with the served model once per samples digest."""

    identity: Dict[str, Any]
    samples: Sequence[Any]


@dataclass(frozen=True)
class BoundInputs:
    """What a pipeline built over a verified workspace bound, for a caller that
    must not read those files again: the patients-like-me reference cohort's
    binding (`src/api/routes/diagnose.py`).

    Each value is the pipeline's own: the four graph digests its workspace check
    verified (`kg`, `node_features`, `edge_indices`, `num_nodes`) and the
    dataset record it verified with them. Read-only: a copy of the digests, and
    a reference the caller must not change.
    """

    graph: Dict[str, str]
    dataset_record: Any


class PipelineBuildError(RuntimeError):
    """The model this pipeline was asked to serve could not be built.

    **Raised, never logged and stepped past.** A missing checkpoint, one that
    did not build over the graph, a model with no graph and a missing PyTorch
    each used to log and return, leaving a pipeline without its GNN. The
    service then published that pipeline as a success, and every diagnosis was
    answered with path-reasoning scores in the model's place
    (docs/working/PLAN_PROVENANCE_CONTRACT.md, M1). A caller
    that asks for a model gets one, or gets this. An unreadable checkpoint
    already raised, as torch's own error; it raises this now, naming the file.

    **No model source at all is the same refusal** (`require_model_source`): a
    graph is never served without a trained model, so a deployment that names a
    graph and no checkpoint is refused, never answered by path reasoning.
    """


class ServingDeviceUnavailable(PipelineBuildError):
    """The device serving was asked to run the model on cannot be had here."""


def require_model_source(checkpoint_path: Optional[Union[str, Path]],
                         data_dir: Optional[Union[str, Path]]) -> None:
    """**A pipeline serves a trained model, or it is not built.** Refuses
    (`PipelineBuildError`) before anything is read when no checkpoint is named,
    or no workspace whose graph tensors the model is computed over. A graph is
    never served on its own: every answer comes from a model."""
    if not checkpoint_path:
        raise PipelineBuildError(
            "No model is configured: a diagnosis is answered only by a trained model, "
            "and no checkpoint was named (SHEPHERD_CHECKPOINT_PATH, on the service). "
            "An operator loads one with “Load a model…” on the Diagnosis page or POST "
            "/api/v1/pipeline/reload; a graph is never served without a model."
        )
    if not data_dir:
        raise PipelineBuildError(
            "A model was requested and there is no workspace to compute its "
            "embeddings over: data_dir (SHEPHERD_DATA_DIR, on the service) was not "
            "given. No pipeline is built without it."
        )


def resolve_serving_device(requested: Optional[str]) -> str:
    """The device a checkpoint is served on, or a refusal (contract M3c, §5.2).

    **`auto` means CUDA, and without CUDA it refuses**, as training's
    (`scripts/train_model.py`, `resolve_training_device`) and measurement's do.
    Unset (None or blank) is `auto`: that is what a deployment that names no
    device asks for. It used to become CPU silently, after the checkpoint had
    been read and the model built, so a service asked to run on the deployment's
    hardware ran somewhere else and said so only in a log line. A literal
    `"auto"` used to reach torch and fail there; it is the same request.

    An explicit `cpu` stays available for development and is recorded as asked.
    A request for CUDA on a host without it refuses here too, before any input
    is read. Other device strings are handed to torch as given, as training's
    are.
    """
    if not HAS_TORCH:
        # **The cause before the remedy.** Without PyTorch no device is
        # available, and telling the operator to set the CPU would send them to
        # a second refusal for the real reason.
        raise PipelineBuildError(
            "A GNN was requested, and PyTorch is not available to load it. No "
            "pipeline is built without the model it was asked for; install torch."
        )
    if requested is None or not str(requested).strip():
        requested = "auto"
    requested = str(requested).strip()
    wants_cuda = requested in ("auto", "cuda") or requested.startswith("cuda:")
    if wants_cuda and not torch.cuda.is_available():
        raise ServingDeviceUnavailable(
            f"Serving device {requested!r} requires CUDA, which is not available "
            "here. Hardware acceleration is a hard requirement for deployment and "
            "CUDA is currently the only supported backend, so serving does not "
            "substitute the CPU for it. Set SHEPHERD_DEVICE=cpu (or send "
            "device 'cpu' with a reload) explicitly for development; the served "
            "identity records the device asked for and the one in effect."
        )
    return "cuda" if requested == "auto" else requested


# ==============================================================================
# Diagnosis Pipeline
# ==============================================================================
def _as_int(value: Any) -> Optional[int]:
    """One integer, or ``None``.

    **This is display metadata, so it never raises.** It runs after the model
    has been built and returned; an exception here would take a working
    checkpoint out of service over a number shown in a status panel. Everything
    it cannot interpret becomes ``None``, and the key simply carries no value.

    Three refusals are deliberate rather than incidental:

    * ``True`` is an ``int`` in Python. Reported as epoch 1, it would be
      metadata the checkpoint never contained.
    * ``1.9`` is not "epoch 1". Truncating it invents a value; ``3.0`` is
      genuinely the integer 3 and is kept.
    * ``inf`` raises ``OverflowError`` from ``int()`` — the conversion failure
      that motivated writing this down.
    * ``"7"`` parses, and parsing it reports a number the checkpoint did not
      contain. A string is already JSON-safe; converting it is invention, not
      protection.
    """
    if isinstance(value, (bool, str, bytes)):
        return None
    try:
        as_float = float(value)
        if not math.isfinite(as_float) or as_float != int(as_float):
            return None
        return int(value)
    except Exception:  # noqa: BLE001 — see the docstring: display never raises
        return None


def _as_metric(value: Any) -> Optional[Union[float, str]]:
    """One metric value, as a float where that is what it is.

    Non-raising for the same reason as ``_as_int``. Scalar conversion fails in
    more ways than a type error: a multi-element tensor raises ``ValueError``, a
    tensor that cannot be read on this device raises ``RuntimeError``, and an
    out-of-range integer raises ``OverflowError``. None of them is a reason to
    stop serving a model.

    A non-finite metric is kept as its name rather than dropped: a ``nan``
    validation loss is a fact about the run worth showing, and ``NaN`` is not
    valid JSON. ``True`` is refused, as it is for epochs — reporting a loss of
    1.0 because the trainer logged a flag is worse than reporting nothing. So
    is a string: these read the numbers a trainer recorded, and a string is not
    one of them.
    """
    if isinstance(value, (bool, str, bytes)):
        return None
    try:
        number = float(value)
    except Exception:  # noqa: BLE001 — see the docstring: display never raises
        return None
    return number if math.isfinite(number) else str(number)


def _device_of(model: Any) -> str:
    """torch's name for the device a model's parameters are on.

    The device in effect, read from the model rather than from the request: a
    `cuda` request serves on `cuda:0`, and that is what is recorded.
    """
    return str(next(iter(model.parameters())).device)


class DiagnosisPipeline:
    """
    End-to-end diagnosis inference pipeline.

    **Always a model finetuned for a task, read from a checkpoint over a
    verified workspace**; one result kind per task (Spec 1 §2.1):
    `disease_task/v1`, `gene_task/v1` or `patients_like_me/v1`. A checkpoint
    without a model spec is refused at load before anything is built
    (`_load_task_bundle`). No checkpoint, or one that cannot be built, raises
    `PipelineBuildError`: nothing is served without a model.

    PathReasoner role:
    - Finds the evidence paths (Phenotype → Gene → Disease) that explain a
      disease row
    - Provides human-readable reasoning chains for clinician review
    - Ranks nothing

    Usage:
        pipeline = DiagnosisPipeline(graph=graph_read, checkpoint_path=checkpoint,
                                     data_dir=workspace)
        result = pipeline.run(patient_phenotypes, top_k=10)
    """

    # A finetuned model bundle's state (README §3, K3, K4): none until a
    # checkpoint with a model spec is loaded (`_load_task_bundle`).
    _bundle_view: Optional[Any] = None
    _bundle_record: Optional[Any] = None
    _bundle_z: Optional[Any] = None
    _bundle_diseases: Optional[Any] = None
    _bundle_candidates: Optional[Any] = None
    _bundle_neighbours: Optional[Any] = None
    _bundle_reference: Optional[Tuple[str, Any, Any]] = None

    def __init__(
        self,
        *,
        graph: "GraphRead",
        checkpoint_path: str,
        data_dir: str,
        config: Optional[PipelineConfig] = None,
        device: Optional[str] = None,
        expected_checkpoint_sha256: Optional[str] = None,
        checkpoint_within: Optional[Path] = None,
    ):
        """
        Initialize the diagnosis pipeline: the workspace verified, the model
        read from its checkpoint and built over it. Anything missing or
        refused raises; no pipeline exists without its model.

        Args:
            graph: The `GraphRead` `KnowledgeGraph.read_json` returned: the
                     graph and the identity of the bytes it was parsed from, in
                     one value, checked against the workspace manifest:
                     `data_dir` proves a workspace is internally consistent,
                     and two consistent workspaces can still be crossed, one's
                     `kg.json` paired with the other's tensors, so embedding
                     rows are read through another graph's node identifiers. It
                     is passed whole, so when it is the value `read_json`
                     returned, the graph served and the digest checked come
                     from one read (contract M2.1, S6); the type does not
                     enforce that pairing.
            checkpoint_path: The trained model's checkpoint (.pt file).
            data_dir: The workspace: its manifest, graph tensors and dataset
                     record (`node_features.pt`, `edge_indices.pt`,
                     `num_nodes.json`, `split_manifest.json`,
                     `kg.provenance.json`).
            config: Pipeline configuration
            device: Device for the model: "cuda", "cuda:N", "cpu", or None /
                "auto" for CUDA, refused without it (`resolve_serving_device`).
            expected_checkpoint_sha256: The SHA-256 the checkpoint at
                `checkpoint_path` must have (the model a caller was shown, a
                bound reload). Compared on the one checkpoint read before the
                bytes are parsed (`read_checkpoint(expected_sha256=)`): other
                bytes, a checkpoint or not, or no file raise `CheckpointChanged`
                unchanged, and nothing about them is built. None reads unbound.
            checkpoint_within: The resolved directory the checkpoint file must
                lie inside (an automatically selected checkpoint: the
                workspace). The one checkpoint read refuses a file its open
                finds outside it, before any byte is read
                (`read_checkpoint(within=)`), and `ReadOutside` is raised
                unchanged. None reads wherever the path leads.
        """
        from src.kg.graph import GraphRead

        require_model_source(checkpoint_path, data_dir)
        if not isinstance(graph, GraphRead):
            raise TypeError(
                "graph= takes the GraphRead KnowledgeGraph.read_json returned, "
                f"not a {type(graph).__name__}"
            )
        self.kg = graph.kg
        #: The identity of the bytes `self.kg` was parsed from.
        self._graph_identity: "ReadIdentity" = graph.identity
        self.config = config or PipelineConfig()
        self.model: Any = None

        # GNN inference state (populated by _init_gnn_inference)
        self._node_id_to_idx: Optional[Dict[str, Dict[str, int]]] = None
        self._graph_data: Optional[Dict[str, Any]] = None
        #: The four digests the manifest binds (`kg`, `node_features`,
        #: `edge_indices`, `num_nodes`), each compared with the read it names,
        #: once the workspace has been verified. Kept whole for M2.4's
        #: comparison with the checkpoint.
        self._bound_digests: Optional[Dict[str, str]] = None
        #: The workspace's dataset record, read once and verified with the graph
        #: (contract M3b). Published by `get_pipeline_config` with its binding
        #: state and origin apart.
        self._dataset_record: Optional[Any] = None
        #: The identity of the checkpoint bytes the model was built from.
        self._checkpoint_identity: Optional["ReadIdentity"] = None
        #: SHA-256 of the one `split_manifest.json` read the workspace was
        #: verified against (contract M3c).
        self._manifest_sha256: Optional[str] = None
        #: The device asked for, as given ("auto" when none was named), and the
        #: one the model's parameters are on once it is built. Both are part of
        #: the served identity (contract M3c).
        self._device_requested: Optional[str] = (
            str(device).strip() if device is not None and str(device).strip() else None
        )
        self._device_in_effect: Optional[str] = None
        self._gnn_ready = False
        #: SHA-256 of the bytes this pipeline's graph was parsed from, as the
        #: manifest binds it, once the workspace has been verified: the served
        #: identity's graph digest (contract M3c).
        self._graph_kg_digest: Optional[str] = None
        self._checkpoint_meta: Dict[str, Any] = {}

        # Initialize reasoning components
        self._init_reasoning_components()
        self._init_explanation_generator()

        # The workspace verified, the model read and built over it
        self._init_gnn_inference(
            checkpoint_path=checkpoint_path,
            data_dir=data_dir,
            device=device,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            checkpoint_within=checkpoint_within,
        )

        logger.info(
            f"DiagnosisPipeline initialized: "
            f"kg_nodes={len(self.kg._nodes)}, "
            f"gnn_ready={self._gnn_ready}, "
            f"device={self._device_in_effect}"
        )

    def _init_reasoning_components(self) -> None:
        """Initialize path reasoning components."""
        reasoning_config = PathReasoningConfig(
            max_path_length=self.config.max_path_length,
            length_penalty=self.config.path_length_penalty,
        )
        self.path_reasoner = create_path_reasoner(config=reasoning_config)

        if self.config.use_direct_path_optimization:
            self.direct_finder = DirectPathFinder(self.kg)
        else:
            self.direct_finder = None

    def _init_explanation_generator(self) -> None:
        """Initialize explanation generator and Step C evidence panel."""
        self.explanation_generator = create_explanation_generator(
            include_ortholog_evidence=self.config.include_ortholog_evidence,
            include_literature_evidence=self.config.include_literature_evidence,
        )
        # Step C: clinician-facing evidence builder. Mode A (direct path)
        # uses self.path_reasoner; without a path the evidence is insufficient.
        self.evidence_panel = create_evidence_panel(
            kg=self.kg,
            path_reasoner=self.path_reasoner,
            config=EvidencePanelConfig(
                strong_path_max_hops=2,
                weak_path_max_hops=self.config.max_path_length,
            ),
        )

    # ==========================================================================
    # GNN Inference Initialization
    # ==========================================================================
    def _init_gnn_inference(
        self,
        checkpoint_path: str,
        data_dir: str,
        device: Optional[str],
        expected_checkpoint_sha256: Optional[str] = None,
        checkpoint_within: Optional[Path] = None,
    ) -> None:
        """
        Verify the workspace, build the model and precompute what it scores with.

        This method handles the full setup chain:
        1. Read and verify the workspace's graph tensors and dataset record
        2. Load the model bundle from its checkpoint
        3. Compute the bundle's full-view vectors once
        4. Build NodeID -> index mapping for score lookup
        """
        if not HAS_TORCH:
            raise PipelineBuildError(
                "A GNN was requested, and PyTorch is not available to load "
                "it. No pipeline is built without the model it was asked "
                "for; install torch."
            )

        # **The device is settled before any input is read** (contract M3c,
        # §5.2), as training's is: `auto` without CUDA refuses here, instead of
        # after the workspace, the checkpoint and the model build, and instead of
        # turning into the CPU.
        device = resolve_serving_device(device)
        self._device_requested = self._device_requested or "auto"

        # Step 1: the workspace's graph tensors, verified against the graph.
        # **The clinical path is a graph consumer, and the costliest one.** A
        # same-shaped `node_features.pt` from another workspace is refused by
        # training and by measurement, and without this it would still be
        # adopted here, paired with a checkpoint, precomputed into embeddings and
        # served. The tensors are parsed first (one read per input); the check
        # runs before they are adopted, before the checkpoint is read and before
        # anything is served. **The graph's identity is part of the check**
        # (`graph=`): verifying only that `data_dir` is internally consistent
        # would leave the `kg` object bound to nothing, and a caller could pair
        # workspace B's graph with workspace A's tensors.
        #
        # **One read per input, and the check compares those reads**
        # (contract M2.1, S6). The manifest, the graph tensors and `kg.json`
        # (read by the caller) are each read once; the digests compared
        # are those of the bytes parsed, and nothing is hashed again by path.
        # A file replaced after its read is either the one used and checked,
        # or refused as not the artifact the manifest records.
        workspace = Path(data_dir)
        manifest = read_split_manifest(workspace)
        graph_read = read_graph_artifacts(workspace, map_location="cpu")
        bound = verify_graph_source_read(
            manifest, self._graph_identity, graph_read.reads
        )
        # **The dataset record is required** (contract M3b): read once and
        # refused unless it is the record this manifest reading binds and
        # names the graph just verified. Before the tensors are adopted and
        # before the checkpoint is read.
        dataset_record = verify_dataset_record(manifest, bound["kg"])
        # **All four are kept**, the opening for M2.4's comparison with the
        # checkpoint.
        self._bound_digests = dict(bound)
        self._graph_kg_digest = bound["kg"]
        self._dataset_record = dataset_record
        # The manifest's identity from the same one read, for the served
        # identity (contract M3c); nothing hashes it again.
        self._manifest_sha256 = manifest.identity.sha256
        # **Only after the check.** The reader omits a file that is absent,
        # and only the check turns that into a refusal.
        self._graph_data = graph_read.graph_data

        # Step 2: Load the model from its checkpoint. A checkpoint that does
        # not load raises out of here (`_load_model_from_checkpoint`).
        self.model = self._load_model_from_checkpoint(
            checkpoint_path, device, expected_sha256=expected_checkpoint_sha256,
            within=checkpoint_within,
        )

        # Step 3: the model bundle's full-view vectors
        self._precompute_node_embeddings(device)

        # Step 4: Build NodeID -> index mapping from KG
        self._node_id_to_idx = self.kg.get_node_id_mapping()

        # **`_gnn_ready` means built, not requested.** Every loader above raises
        # rather than returning empty, so this is unreachable today; it is here so
        # a loader that one day returns None again cannot leave a pipeline that
        # reports a model it cannot score with: every task ranks from the
        # bundle's full-view vectors.
        missing = [] if self._bundle_z is not None else ["the view"]
        if self.model is None or missing:
            raise PipelineBuildError(
                "The GNN was requested and produced no node embeddings for "
                f"{missing or 'any node type'}. No pipeline is built."
            )

        self._device_in_effect = _device_of(self.model)
        self._gnn_ready = True
        logger.info(
            f"GNN inference initialized successfully "
            f"(device={self._device_in_effect})"
        )

    def _load_model_from_checkpoint(
        self,
        checkpoint_path: str,
        device: Optional[str],
        *,
        expected_sha256: Optional[str] = None,
        within: Optional[Path] = None,
    ) -> Any:
        """
        Read the training checkpoint once and build its model bundle
        (`_load_task_bundle`). A checkpoint without a model spec is refused
        before anything is built.

        With `expected_sha256` (a bound reload) the one read is compared with it
        before torch parses the bytes, and the reader decides when the file is
        gone: there is no existence check first.

        With `within` (an automatically selected checkpoint's workspace) the
        one read refuses a file its open finds outside that directory, judged
        on the open descriptor, before any byte is read and so before the
        expected digest.

        Raises:
            ReadOutside: `within` is given and the file opened does not lie
                inside it, or cannot be placed; raised unchanged, before any
                byte is read, so the caller can say nothing was read from it.
            CheckpointChanged: `expected_sha256` is given and the read has other
                bytes, or there is no file; raised unchanged, before anything is
                parsed, so the caller can say the file is not the model shown.
            PipelineBuildError: if the checkpoint is missing, cannot be read, is
                not a training checkpoint, was not trained on this graph (M2.4),
                has no model spec, or does not build a model over it. A missing
                checkpoint and one that did not build used to return None, and
                the pipeline went on without its model.

        Called once the workspace is verified (`_init_gnn_inference`): the
        checkpoint's record is compared with the digests that check took from
        this build's reads (contract M2.4).
        """
        ckpt_path = Path(checkpoint_path)
        if expected_sha256 is None and not ckpt_path.exists():
            raise PipelineBuildError(
                f"Checkpoint {ckpt_path} does not exist. It was requested, so no "
                "pipeline is built without it."
            )

        logger.info(f"Loading GNN model from checkpoint: {ckpt_path}")
        # **The model is built from this read** (contract M2.1, S6; decision 4).
        # Its parsed object is what the build uses and its identity is what is
        # kept, never anything a selection scan read earlier.
        #
        # **`weights_only=True`** (contract decision 10, M3a-1): a checkpoint is
        # parsed as tensors and plain data, never as arbitrary pickled objects,
        # so a path a client names cannot run code here. The trainer's payload
        # loads this way; a file that does not is refused like any unreadable
        # one, with no second attempt.
        #
        # **Read at its resolved path** (M4 decision 15), so the identity kept is
        # the file whose bytes were loaded and its directory is settled now.
        # The recorded results view looks for the ledger beside that read
        # (`served_checkpoint_read`); a symlink given here and retargeted later,
        # without a reload, would otherwise move the lookup to another run's
        # directory while the digest stayed this model's. Messages keep the path
        # as it was given, which is what the operator named.
        #
        # **The model a caller was shown is compared on this read** (a bound
        # reload, `expected_sha256`), before torch parses the bytes: a file
        # rewritten, torn, of another graph or removed since it was shown is
        # `CheckpointChanged`, never parsed, never built and never called
        # unreadable, so it is not reworded below.
        #
        # **An automatically selected file is read only inside the workspace**
        # (`within`). The `resolve()` above is a lookup of the name and decides
        # nothing: the read itself asks the open descriptor where the file
        # lies, and refuses one outside before any byte (`ReadOutside`), so a
        # name swapped out of the workspace after the selection is never read,
        # built or published. Before the expected digest, on the same read.
        try:
            checkpoint_read = read_checkpoint(
                ckpt_path.resolve(), map_location="cpu", weights_only=True,
                expected_sha256=expected_sha256, within=within,
            )
        except (CheckpointChanged, ReadOutside):
            raise
        except Exception as exc:  # noqa: BLE001 — any unreadable file is the same refusal
            raise PipelineBuildError(
                f"Checkpoint {ckpt_path} could not be read "
                f"({type(exc).__name__}: {exc})."
            ) from exc
        checkpoint = checkpoint_read.checkpoint
        if not isinstance(checkpoint, dict):
            raise PipelineBuildError(
                f"{ckpt_path} is not a training checkpoint: it holds a "
                f"{type(checkpoint).__name__}, not the dictionary the trainer "
                "writes."
            )

        # **Was this model trained on this workspace's graph?** (contract M2.4.)
        # The checkpoint's recorded `kg`, `node_features`, `edge_indices` and
        # `num_nodes` against the four digests the workspace check verified from
        # this build's reads. A checkpoint from before the contract, one missing
        # a role, or one from a same-shaped other graph is refused before any
        # model is built.
        from src.utils.fingerprint import (
            TrainingRecordMismatch,
            verify_training_record,
        )
        try:
            verify_training_record(
                checkpoint,
                self._bound_digests,
                tuple(GRAPH_ARTIFACTS),
                recorded_by=f"Checkpoint {ckpt_path}",
                compared_with="this workspace's graph",
            )
        except TrainingRecordMismatch as exc:
            raise PipelineBuildError(str(exc)) from exc

        # **The model bundle, or a refusal before anything is built**
        # (`_load_task_bundle`): its model record is read first, so a
        # checkpoint without a model spec is refused before the graph view,
        # the bundle or any vector is computed.
        model = self._load_task_bundle(checkpoint, ckpt_path)

        # Store checkpoint training metadata for UI display.
        #
        # **Normalised to JSON primitives here, at the only place that knows what
        # these values are.** A checkpoint's `logs` are whatever the trainer put
        # there, and a metric recorded as `loss.detach()` rather than
        # `loss.item()` is a zero-dimensional tensor. It reaches the API through
        # `checkpoint_meta`, which is `Dict[str, Any]` and so accepts it, and
        # fails where the response is encoded -- a serving concern created by a
        # training detail, discovered at the HTTP boundary. Converting at the
        # source is what keeps the boundary a formality.
        # The device the caller settled (`resolve_serving_device`); the same
        # rule again here, so a caller of this method cannot reach the move
        # below with an unresolved `auto`.
        device = resolve_serving_device(device)
        # The epochs its progress record says it completed, beside the
        # record's epoch index (an older file records only the index), so a
        # page names the served file as the lists name it
        # (`renderers.epoch_words`).
        progress = checkpoint.get("progress")
        self._checkpoint_meta = {
            "epoch": _as_int(checkpoint.get("epoch")),
            "completed_epochs": _as_int(progress.get("completed_epochs")
                                        if isinstance(progress, dict) else None),
            "params": int(sum(p.numel() for p in model.parameters())),
            "device": str(device),
        }
        # Extract best metrics from logs if available
        logs = checkpoint.get("logs", {})
        if isinstance(logs, dict):
            for key in CHECKPOINT_LOG_METRICS:
                if key in logs:
                    metric = _as_metric(logs[key])
                    if metric is not None:
                        self._checkpoint_meta[key] = metric

        # Move to device
        model = model.to(device)

        # Kept for M2.4 and M3c; set only once the model it describes is built.
        self._checkpoint_identity = checkpoint_read.identity
        num_params = self._checkpoint_meta["params"]
        logger.info(
            f"GNN model loaded: {num_params:,} parameters, device={device}"
        )
        return model

    def _load_task_bundle(self, checkpoint: Dict[str, Any], ckpt_path: Path) -> Any:
        """A checkpoint with a model spec: the model record read and checked,
        stage and task first (provenance §9.3 hazard 3: a base serves no
        diagnosis), the graph view rebuilt from the verified graph by the spec's
        rule and compared with the record, and the bundle built and strictly
        loaded by the one builder. Nothing is inferred from weight names.

        **The model record is read first** (`read_model_record`): a checkpoint
        without a model spec is refused here, before the graph view, the bundle
        or any vector is built."""
        from dataclasses import asdict as _asdict

        from src.kg.graph_view import build_graph_view
        from src.models.bundle.record import (
            ModelRecordError,
            check_model_record,
            load_bundle,
            read_model_record,
        )

        try:
            record = read_model_record(checkpoint)
            check_model_record(record, learning_stage="finetune")
            check_model_record(record, learning_stage="finetune", task=record.task)
            rule = _asdict(record.model_spec.graph_view)
            rule["exclude_relations"] = list(rule["exclude_relations"])
            view = build_graph_view(self._graph_data, rule=rule)
            bundle, record = load_bundle(checkpoint, learning_stage="finetune",
                                         task=record.task, view_digest=view.digest)
        except (ModelRecordError, ValueError) as exc:
            raise PipelineBuildError(
                f"Checkpoint {ckpt_path}: {exc}. The Diagnosis page serves a model "
                "finetuned for a task (disease, gene or patients-like-me).") from exc
        self._bundle_view = view
        self._bundle_record = record
        return bundle

    def _precompute_node_embeddings(
        self,
        device: Optional[str] = None,
    ) -> None:
        """
        Compute the model bundle's full-view vectors once, and what its task
        reads from them on every request.

        A loader that returned no model leaves nothing to compute; the build
        refuses that (`_init_gnn_inference`).
        """
        if self.model is None:
            return

        if device is None:
            device = next(self.model.parameters()).device
        else:
            device = torch.device(device)

        # The full-view rule over the whole view (README §2.8, G4), once;
        # the diseases' projections are fixed, so each request projects
        # only its patient.
        from src.models.tasks.scoring import full_view_vectors

        view = self._bundle_view
        self._bundle_z = full_view_vectors(self.model, view.edge_index(), view.rel, device,
                                           node_type=view.node_type)
        task = self.model.task
        if task == "disease":
            self._bundle_diseases = view.type_nodes("disease")
            with torch.no_grad():
                self._bundle_candidates = self.model.task_head.project_candidates(
                    self._bundle_z[self._bundle_diseases.to(self._bundle_z.device)])
        elif task == "gene" and self.model.task_head.spec.gene.augmentation:
            from src.models.tasks.gene import neighbour_table

            self._bundle_neighbours = neighbour_table(
                view, self.model.task_head.spec.gene.augmentation_k)
        logger.info(f"Model bundle vectors computed for the {task} task: "
                    f"{tuple(self._bundle_z.shape)} over view {view.digest[:12]}")

    def rank(
        self,
        patient_input: PatientPhenotypes,
        include_ortholog_evidence: Optional[bool] = None,
        reference_cohort: Optional[ReferenceCohort] = None,
    ) -> ServedRanking:
        """The served ranking of `patient_input`, whole: everything `run` does
        up to and including its sort, and nothing after it.

        Validation and normalisation, the node mapping and the served task's
        ranking, exactly as `run` does them; `run` is this followed by its cut at
        `top_k` and what it builds on the rows it keeps. **One implementation**: `run` calls this, and nothing constructs
        candidates a second way.
        """
        if include_ortholog_evidence is None:
            include_ortholog_evidence = self.config.include_ortholog_evidence

        # Step 1: Validate input
        validation = self.validate_input(patient_input)
        if not validation.is_valid:
            logger.error(f"Input validation failed: {validation.errors}")
            return ServedRanking(
                outcome="refused",
                warnings=list(validation.errors),
                patient_input=patient_input,
                phenotype_input=validation.phenotype_input,
            )
        warnings: List[str] = list(validation.warnings)

        # **From here on the list as used is the input, not the list as
        # received.** Scoring, explanations and the summary all read
        # `patient_input`, and each of them read the raw request before: the
        # summary counted repeats and unknown ids as "input phenotypes", and the
        # explanations listed them. Rebinding once, here, leaves nothing below
        # able to reach the raw list.
        patient_input = self._as_used(patient_input, validation)

        # Step 2: the graph nodes, from the one mapping step. Validation refused
        # an input with none, so the list is never empty here.
        source_ids = list(validation.node_ids)

        task = self.model.task
        if (task == "patients_like_me") != (reference_cohort is not None):
            raise ValueError(
                "a patients-like-me model ranks a registered reference cohort version, "
                "which must be chosen" if task == "patients_like_me" else
                "a reference cohort is ranked by a patients-like-me model only")
        if task == "gene":
            return self._rank_by_gene_task(patient_input, validation, warnings, source_ids)
        if task == "patients_like_me":
            return self._rank_patients_like_me(patient_input, validation, warnings,
                                               source_ids, reference_cohort)
        return self._rank_by_disease_task(patient_input, validation, warnings, source_ids)

    def _rank_by_disease_task(self, patient_input: PatientPhenotypes,
                              validation: "ValidationResult", warnings: List[str],
                              source_ids: List[NodeID]) -> ServedRanking:
        """`disease_task/v1`: every disease of the model's graph view, ranked by
        the disease head (README A7, U6), ties by view id. The mapping step
        admitted only phenotypes in the view (`_knows_phenotype`)."""
        from src.models.tasks.scoring import patient_vectors

        view = self._bundle_view
        kept, warnings = self._bundle_phenotypes(validation, warnings)
        z = self._bundle_z
        kept = kept.to(z.device)
        with torch.no_grad():
            patient = patient_vectors(self.model, z, kept.unsqueeze(0),
                                      torch.ones(1, kept.numel(), dtype=torch.bool,
                                                 device=z.device))
            scores = self.model.task_head.scores(patient, self._bundle_candidates)[0].cpu()
        if not bool(torch.isfinite(scores).all()):
            raise ResultRefused("NON_FINITE_SCORE",
                                "the disease head produced a non-finite score; the result "
                                "was refused, not ranked")
        diseases = self._bundle_diseases
        order = sorted(range(diseases.numel()),
                       key=lambda i: (-float(scores[i]), int(diseases[i])))
        locals_ = view.local_index(diseases).tolist()
        reverse = self._disease_node_ids()
        candidates: List[Candidate] = []
        for rank, i in enumerate(order, start=1):
            node_id = reverse[locals_[i]]
            node = self.kg.get_node(node_id)
            candidates.append(DiseaseTaskCandidate(
                rank=rank, disease_id=node_id,
                disease_name=node.name if node else str(node_id), score=float(scores[i])))
        return ServedRanking(
            outcome="ranked", warnings=warnings, patient_input=patient_input,
            phenotype_input=validation.phenotype_input, candidates=candidates,
            source_ids=source_ids)

    def _bundle_phenotypes(self, validation: "ValidationResult", warnings: List[str]
                           ) -> Tuple[Any, List[str]]:
        """The request's phenotypes as view ids of the model's graph; the
        mapping step (`_knows_phenotype`) admitted only phenotypes in the view."""
        rows = torch.tensor(list(validation.rows or []), dtype=torch.long)
        view_ids = self._bundle_view.view_ids("phenotype", rows) if rows.numel() else rows
        return torch.unique(view_ids), warnings

    def _rank_by_gene_task(self, patient_input: PatientPhenotypes,
                           validation: "ValidationResult", warnings: List[str],
                           source_ids: List[NodeID]) -> ServedRanking:
        """`gene_task/v1`: the request's candidate genes (PLAN_TEST_RESULTS
        §8.3), ranked by `η·EMBSIM + (1 − η)·SPLSIM` with exact hops on the
        whole view (README §6), ties by view id. A gene that maps to no gene of
        the view is left out and counted ("N mapped of M submitted"); one with
        an undefined distance is not scorable and said so."""
        from src.models.tasks.gene import distance_table, score_gene_lists

        submitted = list(patient_input.candidate_genes or [])
        if not submitted:
            return ServedRanking(
                outcome="no_candidates",
                warnings=warnings + ["A gene task model ranks the patient's candidate genes; "
                                     "none were given (candidate_genes)"],
                patient_input=patient_input, phenotype_input=validation.phenotype_input,
                source_ids=source_ids)
        view = self._bundle_view
        lookup = self._gene_lookup()
        local = [lookup.get(g, lookup.get(g.strip().upper())) for g in submitted]
        known = sorted({i for i in local if i is not None})
        view_ids = view.view_ids("gene", torch.tensor(known, dtype=torch.long)) if known \
            else torch.empty(0, dtype=torch.long)
        genes = torch.unique(view_ids[view_ids >= 0])
        if genes.numel() < len(submitted):
            unmapped = sum(i is None for i in local)
            repeats = len(submitted) - unmapped - len(known)
            outside = len(known) - int(genes.numel())
            parts = [f"{n} {what}" for n, what in (
                (unmapped, "map to no gene of the graph"),
                (repeats, "repeat an earlier entry"),
                (outside, "are outside the model's graph view")) if n]
            warnings = warnings + [f"{genes.numel()} of {len(submitted)} candidate genes "
                                   f"submitted are scored: {', '.join(parts)}"]
        phenotypes, warnings = self._bundle_phenotypes(validation, warnings)
        if phenotypes.numel() == 0 or genes.numel() == 0:
            return ServedRanking(
                outcome="no_candidates",
                warnings=warnings + ["Nothing to rank: no phenotype or no candidate gene is in "
                                     "the model's graph view"],
                patient_input=patient_input, phenotype_input=validation.phenotype_input,
                source_ids=source_ids)
        # The one gene scorer validation and measurement use (`score_gene_lists`).
        scored = score_gene_lists(
            self.model, self._bundle_z, phenotypes.unsqueeze(0),
            torch.ones(1, phenotypes.numel(), dtype=torch.bool), genes.unsqueeze(0),
            torch.ones(1, genes.numel(), dtype=torch.bool),
            distance_table(view, phenotypes, genes), self._bundle_neighbours)
        scores, embsim, sp = scored.scores[0], scored.embsim[0], scored.sp_similarity[0]
        mean = scored.mean_distance
        scorable = scored.defined[0]
        if not bool(scorable.all()):
            warnings = warnings + [f"{int((~scorable).sum())} candidate gene(s) have an "
                                   "unreachable phenotype and are not scorable"]
        if not bool(torch.isfinite(scores[scorable]).all()):
            raise ResultRefused("NON_FINITE_SCORE",
                                "the gene head produced a non-finite score; the result was "
                                "refused, not ranked")
        order = sorted((i for i in range(genes.numel()) if bool(scorable[i])),
                       key=lambda i: (-float(scores[i]), int(genes[i])))
        reverse = self._reverse_node_ids(NodeType.GENE.value)
        locals_ = view.local_index(genes).tolist()
        candidates: List[Candidate] = []
        for rank, i in enumerate(order, start=1):
            node_id = NodeID.from_string(reverse[locals_[i]])
            node = self.kg.get_node(node_id)
            candidates.append(GeneTaskCandidate(
                rank=rank, gene_id=node_id, gene_name=node.name if node else str(node_id),
                score=float(scores[i]), embsim=float(embsim[i]), sp_similarity=float(sp[i]),
                mean_distance=float(mean[0, i])))
        if not candidates:
            return ServedRanking(
                outcome="no_candidates", warnings=warnings + ["No candidate gene is scorable"],
                patient_input=patient_input, phenotype_input=validation.phenotype_input,
                source_ids=source_ids)
        return ServedRanking(
            outcome="ranked", warnings=warnings, patient_input=patient_input,
            phenotype_input=validation.phenotype_input, candidates=candidates,
            source_ids=source_ids)

    def _rank_patients_like_me(self, patient_input: PatientPhenotypes,
                               validation: "ValidationResult", warnings: List[str],
                               source_ids: List[NodeID],
                               reference: Optional[ReferenceCohort]) -> ServedRanking:
        """`patients_like_me/v1`: the reference cohort version's patients
        (PLAN_TEST_RESULTS §8.4), ranked by the patient tower's similarity,
        ties by case id. A reference patient whose case id is the query's is
        left out (self-exclusion by stable identity, E5). Known causal genes
        are reference context only; the query's genes are never an input."""
        from src.models.tasks.patients_like_me import bind_patients, patient_tower_vectors

        phenotypes, warnings = self._bundle_phenotypes(validation, warnings)
        if phenotypes.numel() == 0:
            return ServedRanking(
                outcome="no_candidates",
                warnings=warnings + ["None of the phenotypes is in the model's graph view"],
                patient_input=patient_input, phenotype_input=validation.phenotype_input,
                source_ids=source_ids)
        view, z = self._bundle_view, self._bundle_z
        key = str(reference.identity.get("samples_sha256"))
        if self._bundle_reference is None or self._bundle_reference[0] != key:
            binding = bind_patients(view, reference.samples)
            vectors = patient_tower_vectors(self.model, z, binding.cases)
            self._bundle_reference = (key, binding, vectors)
        _, binding, vectors = self._bundle_reference
        if binding.excluded:
            warnings = warnings + [f"{len(binding.excluded)} reference patient(s) have no "
                                   "phenotype in the model's graph view and are not ranked"]
        keep = [i for i, c in enumerate(binding.cases) if c.case_id != patient_input.patient_id]
        if len(keep) < len(binding.cases):
            warnings = warnings + [f"Reference patient {patient_input.patient_id} is left out: "
                                   "its case id is the query's (self-exclusion)"]
        if not keep:
            return ServedRanking(
                outcome="no_candidates",
                warnings=warnings + ["The reference cohort has no other patient to rank"],
                patient_input=patient_input, phenotype_input=validation.phenotype_input,
                source_ids=source_ids)
        with torch.no_grad():
            query = patient_tower_vectors(
                self.model, z, [type(binding.cases[0])(patient_input.patient_id, phenotypes,
                                                       frozenset())])
            sims = self.model.task_head.scores(query, vectors[keep])[0]
        if not bool(torch.isfinite(sims).all()):
            raise ResultRefused("NON_FINITE_SCORE",
                                "the patient tower produced a non-finite similarity; the "
                                "result was refused, not ranked")
        order = sorted(range(len(keep)),
                       key=lambda j: (-float(sims[j]), binding.cases[keep[j]].case_id))
        rows = {s.patient_id: s for s in reference.samples}
        candidates: List[Candidate] = []
        for rank, j in enumerate(order, start=1):
            case = binding.cases[keep[j]]
            candidates.append(self._patient_candidate(rank, case.case_id, float(sims[j]),
                                                      rows[case.case_id]))
        return ServedRanking(
            outcome="ranked", warnings=warnings, patient_input=patient_input,
            phenotype_input=validation.phenotype_input, candidates=candidates,
            source_ids=source_ids)

    def _patient_candidate(self, rank: int, case_id: str, similarity: float,
                           row: Any) -> PatientCandidate:
        """A reference patient with what its cohort recorded: HPO terms, the
        diagnosis and the causal genes, in the graph's ids and names."""
        def ids(node_type: str, locals_: Any) -> List[str]:
            reverse = self._reverse_node_ids(node_type)
            return [reverse[i] for i in (locals_ or []) if i in reverse]

        phenotypes = tuple(NodeID.from_string(k).local_id for k in ids("phenotype",
                                                                       row.phenotype_ids))
        disease = ids("disease", [row.disease_id])
        disease_id = NodeID.from_string(disease[0]) if disease else None
        node = self.kg.get_node(disease_id) if disease_id else None
        genes = []
        for key in ids("gene", getattr(row, "causal_gene_ids", None)):
            gene = self.kg.get_node(NodeID.from_string(key))
            genes.append(gene.name if gene else key)
        # Declared causal genes outside the graph, as the cohort recorded them.
        genes.extend(getattr(row, "causal_genes_unmapped", None) or ())
        return PatientCandidate(rank=rank, case_id=case_id, similarity=similarity,
                                phenotypes=phenotypes, disease_id=disease_id,
                                disease_name=node.name if node else None,
                                causal_genes=tuple(genes))

    def _reverse_node_ids(self, node_type: str) -> Dict[int, str]:
        """Index -> node id string for one node type of the served graph,
        built once per graph (the mapping is fixed while it is served)."""
        cache = getattr(self, "_reverse_node_id_cache", None)
        if cache is None or cache[0] is not self._node_id_to_idx:
            cache = (self._node_id_to_idx, {})
            self._reverse_node_id_cache = cache
        reverse = cache[1].get(node_type)
        if reverse is None:
            reverse = {i: key for key, i in (self._node_id_to_idx or {}).get(node_type,
                                                                             {}).items()}
            cache[1][node_type] = reverse
        return reverse

    def _gene_lookup(self) -> Dict[str, int]:
        """Gene identifiers -> gene index, the importer's rule
        (`KnowledgeGraph.gene_lookup`), cached."""
        if getattr(self, "_gene_lookup_cache", None) is None:
            self._gene_lookup_cache = self.kg.gene_lookup()
        return self._gene_lookup_cache

    def _disease_node_ids(self) -> Dict[int, NodeID]:
        """Each disease's NodeID by its row in the graph's disease table."""
        if getattr(self, "_disease_node_id_cache", None) is None:
            mapping = (self._node_id_to_idx or {}).get(NodeType.DISEASE.value, {})
            self._disease_node_id_cache = {row: NodeID.from_string(node)
                                           for node, row in mapping.items()}
        return self._disease_node_id_cache

    def run(
        self,
        patient_input: PatientPhenotypes,
        top_k: int = 10,
        include_explanations: Optional[bool] = None,
        include_ortholog_evidence: Optional[bool] = None,
        include_literature_evidence: Optional[bool] = None,
        reference_cohort: Optional[ReferenceCohort] = None,
    ) -> InferenceResult:
        """
        Execute the full diagnosis pipeline: `rank`, then the cut at `top_k`
        and what is built on the rows kept.

        Args:
            patient_input: Patient phenotype data
            top_k: Number of top candidates to return
            include_explanations: Override config for explanation generation
            include_ortholog_evidence: Override config for ortholog evidence
            include_literature_evidence: Override config for literature evidence

        Returns:
            InferenceResult with diagnosis candidates and explanations
        """
        start_time = time.time()

        # Use config defaults if not overridden
        if include_explanations is None:
            include_explanations = self.config.include_explanations
        if include_ortholog_evidence is None:
            include_ortholog_evidence = self.config.include_ortholog_evidence
        if include_literature_evidence is None:
            include_literature_evidence = self.config.include_literature_evidence

        logger.info(
            f"Starting diagnosis for patient {patient_input.patient_id} "
            f"with {len(patient_input.phenotypes)} phenotypes"
        )

        ranking = self.rank(patient_input, include_ortholog_evidence, reference_cohort)
        # A patients-like-me result names its reference cohort version, a
        # refused or empty one included (rank refuses one without it).
        reference = dict(reference_cohort.identity) if reference_cohort is not None else None
        sp_bounds = self._sp_bounds()
        if ranking.outcome == "refused":
            return InferenceResult(
                patient_id=patient_input.patient_id,
                timestamp=datetime.now(),
                candidates=[],
                served_identity=self.served_identity(),
                warnings=ranking.warnings,
                inference_time_ms=(time.time() - start_time) * 1000,
                phenotype_input=ranking.phenotype_input,
                reference_cohort=reference, sp_bounds=sp_bounds,
            )
        patient_input = ranking.patient_input
        if ranking.outcome == "no_candidates":
            return self._create_empty_result(
                patient_input.patient_id,
                start_time,
                ranking.warnings,
                ranking.phenotype_input,
                reference_cohort=reference, sp_bounds=sp_bounds,
            )
        if self.model.task != "disease":
            # Gene and reference-patient rows carry their own values; path
            # evidence explains disease rows (README §1) and is not built here.
            return InferenceResult(
                patient_id=patient_input.patient_id, timestamp=datetime.now(),
                candidates=list(ranking.candidates[:top_k]),
                served_identity=self.served_identity(),
                inference_time_ms=(time.time() - start_time) * 1000,
                warnings=ranking.warnings, phenotype_input=ranking.phenotype_input,
                reference_cohort=reference, sp_bounds=sp_bounds)

        # **The cut is the only thing `top_k` does**: the order is `rank`'s.
        # Step 4b: The phenotypes the graph annotates directly to each candidate
        # returned, from the list as used. **Not gated on explanations**: a
        # caller that turns them off still reads the field, and it costs one scan
        # of each returned disease's incoming edges. The explanation's "Matching
        # Phenotypes" section comes from the same helper and the same list: it
        # lists the first ten of this field, and says how many more there are.
        # Candidates are frozen, so each kept one is a copy with the field set,
        # and the `ServedRanking` keeps what `rank` built.
        candidates = [
            replace(candidate, matching_phenotypes=find_matching_phenotypes(
                candidate.disease_id, patient_input.phenotypes, self.kg))
            for candidate in ranking.candidates[:top_k]
        ]
        all_paths: Dict[str, List[ReasoningPath]] = {}
        if candidates:
            # **Paths explain, they never rank** (README §1, the PathReasoner
            # extension): discovered for the rows shown, after the model's order.
            all_paths = self._find_all_paths(ranking.source_ids, include_ortholog_evidence)
            candidates = [
                replace(candidate,
                        supporting_genes=self._extract_supporting_genes(paths),
                        reasoning_paths=[list(p.nodes) for p in paths[:5]],
                        evidence_sources=self._determine_evidence_sources(paths))
                for candidate in candidates
                for paths in [all_paths.get(str(candidate.disease_id), [])]
            ]

        # Step 5: Generate explanations + Step C evidence packages
        if include_explanations:
            logger.debug("Generating explanations and evidence packages...")
            candidates = self._add_explanations(
                candidates=candidates,
                all_paths=all_paths,
                patient_input=patient_input,
                source_ids=ranking.source_ids,
            )

        # Step 6: Create summary explanation
        summary = None
        if include_explanations and candidates:
            summary = self._generate_summary_explanation(
                candidates[:min(3, len(candidates))],
                patient_input,
            )

        # Build result
        inference_time = (time.time() - start_time) * 1000
        logger.info(
            f"Diagnosis complete: {len(candidates)} candidates "
            f"in {inference_time:.1f}ms"
        )

        return InferenceResult(
            patient_id=patient_input.patient_id,
            timestamp=datetime.now(),
            candidates=candidates,
            served_identity=self.served_identity(),
            summary_explanation=summary,
            inference_time_ms=inference_time,
            warnings=ranking.warnings,
            phenotype_input=ranking.phenotype_input,
        )

    def validate_input(
        self,
        patient_input: PatientPhenotypes,
    ) -> ValidationResult:
        """
        Validate patient input, and apply the phenotype-list rule.

        In this order (docs/working/PLAN_PHENOTYPE_NORMALISATION.md §3-§4):

        1. **Per-position fields match the list as received.** A supplied
           `phenotype_confidences` has one entry per phenotype. This comes first
           because step 3 reduces it by position, and a mismatch is refused, never
           padded or truncated.
        2. **Ids are mapped to graph nodes.** An id with no node is dropped and
           always stated (a warning, and `phenotype_input.unknown`).
        3. **Repeats are removed by node**, keeping the first occurrence
           (`src/kg/phenotype_normalisation.py`).
        4. **The count is checked on what remains.** More than `max_phenotypes`
           is refused. Nothing is truncated: the previous version kept the first
           100 entries, repeats and unknown ids included, so "100 × A, then B"
           silently scored A alone. An input left with no phenotype is refused.
        5. **Each node is mapped to its row** in the graph's phenotype index,
           which the model's graph view maps (Spec 1 §2.2: one mapping step,
           exact term, graph node, row). A known node with no row is a
           model–graph mismatch: the result is refused (`ResultRefused`,
           `MODEL_TERM_UNAVAILABLE`), never scored without it.

        Args:
            patient_input: Patient phenotype data

        Returns:
            ValidationResult with the phenotypes to score, their request positions,
            the account of what was received and used, and any warnings/errors
        """
        errors: List[str] = []
        warnings: List[str] = []
        phenotypes = list(patient_input.phenotypes)
        confidences = patient_input.phenotype_confidences

        if confidences is not None and len(confidences) != len(phenotypes):
            return ValidationResult(
                is_valid=False,
                errors=[
                    f"phenotype_confidences has {len(confidences)} entries for "
                    f"{len(phenotypes)} phenotypes. It needs one per phenotype, "
                    "or none at all; nothing was scored."
                ],
            )

        # Check patient_id
        if not patient_input.patient_id:
            errors.append("patient_id is required")

        # Check phenotype count
        if len(phenotypes) < self.config.min_phenotypes:
            errors.append(
                f"At least {self.config.min_phenotypes} phenotype(s) required, "
                f"got {len(phenotypes)}"
            )

        # Map each id to its node, keeping the request position of every one
        # that has a node.
        known_nodes: List[str] = []
        known_positions: List[int] = []
        unknown: List[str] = []
        for position, pheno_id in enumerate(phenotypes):
            node_id = NodeID(source=DataSource.HPO, local_id=pheno_id)
            if self._knows_phenotype(node_id):
                known_nodes.append(str(node_id))
                known_positions.append(position)
            else:
                unknown.append(pheno_id)
                warnings.append(f"Unknown phenotype: {pheno_id}" if not self.kg.has_node(node_id)
                                else f"Phenotype outside the model's graph view: {pheno_id}")

        normalised = normalise_phenotypes(known_nodes, positions=known_positions)
        used = [phenotypes[position] for position in normalised.kept_positions]
        if normalised.repeats_removed:
            warnings.append(
                f"Removed {normalised.repeats_removed} repeated phenotype "
                f"{'entry' if normalised.repeats_removed == 1 else 'entries'}; "
                "each phenotype is scored once."
            )

        if len(used) > self.config.max_phenotypes:
            errors.append(
                f"{len(used)} distinct phenotypes found in the graph; at most "
                f"{self.config.max_phenotypes} are accepted. Nothing was scored, "
                "and the list was not shortened."
            )

        if not used and not errors:
            errors.append("No valid phenotypes after validation")

        node_ids = [NodeID(source=DataSource.HPO, local_id=pheno_id) for pheno_id in used]
        rows = None if errors else self._phenotype_rows(node_ids)

        return ValidationResult(
            is_valid=len(errors) == 0,
            validated_phenotypes=used,
            warnings=warnings,
            errors=errors,
            kept_positions=list(normalised.kept_positions),
            phenotype_input=PhenotypeInputAccount(
                received=len(phenotypes),
                unknown=tuple(unknown),
                repeats_removed=normalised.repeats_removed,
                # A refused request scored nothing, so it used nothing — even
                # when its phenotypes mapped, as with too many of them.
                used=() if errors else tuple(used),
            ),
            node_ids=[] if errors else node_ids,
            rows=rows,
        )

    def _knows_phenotype(self, node_id: NodeID) -> bool:
        """Whether the served model can score this phenotype: a node of the
        graph and of the model's graph view."""
        if getattr(self, "_view_phenotypes", None) is None:
            index = (self._node_id_to_idx or {}).get(NodeType.PHENOTYPE.value, {})
            keys = list(index)
            rows = torch.tensor([index[k] for k in keys], dtype=torch.long)
            in_view = (self._bundle_view.view_ids("phenotype", rows) >= 0).tolist() if keys \
                else []
            self._view_phenotypes = {k for k, ok in zip(keys, in_view, strict=True) if ok}
        return str(node_id) in self._view_phenotypes

    def _phenotype_rows(self, node_ids: List[NodeID]) -> List[int]:
        """Each node's row in the graph's phenotype index, in order; the
        model's graph view maps the rows (`_rank_by_disease_task`).

        Raises `ResultRefused` (`MODEL_TERM_UNAVAILABLE`, naming the node) for a
        node with no row: a model–graph mismatch is refused, never skipped."""
        index = (self._node_id_to_idx or {}).get(NodeType.PHENOTYPE.value, {})
        missing = [str(node) for node in node_ids if index.get(str(node)) is None]
        if missing:
            raise ResultRefused(
                "MODEL_TERM_UNAVAILABLE",
                f"phenotype {missing[0]} is in the graph but has no row in its node "
                "mapping")
        return [int(index[str(node)]) for node in node_ids]

    @staticmethod
    def _as_used(
        patient_input: PatientPhenotypes, validation: ValidationResult
    ) -> PatientPhenotypes:
        """The input as scored: its validated phenotypes, and every per-position
        field reduced to the same request positions."""
        confidences = patient_input.phenotype_confidences
        return replace(
            patient_input,
            phenotypes=list(validation.validated_phenotypes),
            phenotype_confidences=(
                None if confidences is None
                else [confidences[position] for position in validation.kept_positions]
            ),
        )

    def _find_all_paths(
        self,
        source_ids: List[NodeID],
        include_ortholog: bool = True,
    ) -> Dict[str, List[ReasoningPath]]:
        """
        Find all reasoning paths from phenotypes to diseases.

        Args:
            source_ids: Phenotype NodeIDs
            include_ortholog: Include ortholog paths (P1 feature)

        Returns:
            Dictionary mapping disease ID strings to their supporting paths
        """
        all_paths: Dict[str, List[ReasoningPath]] = {}

        # Use direct path finder for optimization if available
        if self.direct_finder and self.config.use_direct_path_optimization:
            direct_paths = self.direct_finder.find_phenotype_gene_disease_paths(
                source_ids
            )
            # Score the direct paths (DirectPathFinder returns unscored paths)
            direct_paths = self.path_reasoner.score_paths(direct_paths, self.kg)
            for path in direct_paths:
                disease_key = str(path.target)
                if disease_key not in all_paths:
                    all_paths[disease_key] = []
                all_paths[disease_key].append(path)

        # Find general paths via BFS
        general_paths = self.path_reasoner.find_paths(
            source_ids=source_ids,
            target_type=NodeType.DISEASE,
            kg=self.kg,
            max_length=self.config.max_path_length,
        )

        # Score paths
        scored_paths = self.path_reasoner.score_paths(general_paths, self.kg)

        # Diagnostic: log path scores by length
        by_length: Dict[int, list] = {}
        for p in scored_paths:
            by_length.setdefault(p.length, []).append(p.score)
        for length, scores in sorted(by_length.items()):
            logger.info(
                f"[PathScoring] {length}-hop paths: {len(scores)} found, "
                f"score range [{min(scores):.4f}, {max(scores):.4f}]"
            )

        # Group by disease
        for path in scored_paths:
            disease_key = str(path.target)
            if disease_key not in all_paths:
                all_paths[disease_key] = []

            # Avoid duplicates
            existing_path_strs = {
                str([str(n) for n in p.nodes]) for p in all_paths[disease_key]
            }
            path_str = str([str(n) for n in path.nodes])
            if path_str not in existing_path_strs:
                all_paths[disease_key].append(path)

        # Sort paths within each disease by score
        for disease_key in all_paths:
            all_paths[disease_key].sort(key=lambda p: p.score, reverse=True)
            # Keep only top paths per candidate
            all_paths[disease_key] = all_paths[disease_key][
                :self.config.max_paths_per_candidate
            ]

        return all_paths

    def _extract_supporting_genes(
        self, paths: List[ReasoningPath]
    ) -> List[str]:
        """Extract unique supporting genes from paths."""
        genes = set()
        for path in paths:
            for node_id in path.nodes:
                node = self.kg.get_node(node_id)
                if node and node.node_type == NodeType.GENE:
                    genes.add(node.name)
        return sorted(list(genes))

    def _determine_evidence_sources(
        self, paths: List[ReasoningPath]
    ) -> List[EvidenceSource]:
        """Determine evidence sources from paths."""
        sources: List[EvidenceSource] = []
        seen_sources: set = set()

        for path in paths:
            for node_id in path.nodes:
                source_key = f"{node_id.source.value}:{node_id.local_id}"
                if source_key not in seen_sources:
                    seen_sources.add(source_key)
                    sources.append(
                        EvidenceSource(
                            source_type=node_id.source,
                            source_id=node_id.local_id,
                        )
                    )

        # Limit to top sources to avoid overwhelming results
        return sources[:10]

    def _add_explanations(
        self,
        candidates: List[Candidate],
        all_paths: Dict[str, List[ReasoningPath]],
        patient_input: PatientPhenotypes,
        source_ids: Optional[List[NodeID]] = None,
    ) -> List[Candidate]:
        """
        Copies of the candidates with explanations and Step C evidence packages.

        For each candidate, the EvidencePanel decides between:
          - Mode A (direct path): use existing paths from BFS, label as
            STRONG/WEAK depending on shortest hop count
          - INSUFFICIENT: when no direct path is available

        The explanation names the candidate's score with the kind's label, so
        the text says what the number is (Spec 1 §2.1).
        """
        label = self._result_label()
        explained: List[Candidate] = []
        for candidate in candidates:
            paths = all_paths.get(str(candidate.disease_id), [])

            # Legacy narrative explanation (kept for backwards compat)
            changes: Dict[str, Any] = {
                "explanation": self.explanation_generator.generate_explanation(
                    candidate=candidate,
                    phenotypes=patient_input,
                    kg=self.kg,
                    paths=paths,
                    score_label=label,
                ),
            }

            # Step C: structured evidence package with its evidence label,
            # from the discovered paths (Mode A) or insufficient without one.
            if source_ids is not None:
                package: EvidencePackage = self.evidence_panel.build_evidence(
                    candidate=candidate,
                    patient_input=patient_input,
                    source_ids=source_ids,
                    existing_paths=paths if paths else None,
                )
                changes["evidence_package"] = package.to_dict()
                changes["evidence_label"] = package.evidence_label.value
            explained.append(replace(candidate, **changes))

        return explained

    def _generate_summary_explanation(
        self,
        top_candidates: List[Candidate],
        patient_input: PatientPhenotypes,
    ) -> str:
        """Generate overall summary explanation."""
        if not top_candidates:
            return "No diagnosis candidates found."

        lines = [
            f"## Diagnosis Summary for Patient {patient_input.patient_id}",
            "",
            f"Based on {len(patient_input.phenotypes)} input phenotypes, "
            f"the following top candidates were identified:",
            "",
        ]

        for candidate in top_candidates:
            lines.append(
                f"**{candidate.rank}. {candidate.disease_name}** "
                f"(Score: {candidate_score(candidate):.3f})"
            )
            if candidate.supporting_genes:
                lines.append(
                    f"   Key genes: {', '.join(candidate.supporting_genes[:3])}"
                )
            lines.append("")

        lines.extend([
            "---",
            f"*{self._result_label()}.*",
            f"*Score: {SCORE_MEANING}.*",
            "*Note: These results are computational predictions for clinical review. "
            "Final diagnosis should be made by qualified healthcare professionals.*",
        ])

        return "\n".join(lines)

    def _create_empty_result(
        self,
        patient_id: str,
        start_time: float,
        warnings: List[str],
        phenotype_input: Optional[PhenotypeInputAccount] = None,
        reference_cohort: Optional[Dict[str, Any]] = None,
        sp_bounds: Optional[Tuple[float, float]] = None,
    ) -> InferenceResult:
        """A result with no candidates (nothing to rank), with its warnings."""
        return InferenceResult(
            patient_id=patient_id,
            timestamp=datetime.now(),
            candidates=[],
            served_identity=self.served_identity(),
            warnings=warnings,
            inference_time_ms=(time.time() - start_time) * 1000,
            phenotype_input=phenotype_input,
            reference_cohort=reference_cohort, sp_bounds=sp_bounds,
        )

    def _sp_bounds(self) -> Optional[Tuple[float, float]]:
        """A gene task model's SP bounds (its spec's), which its results name;
        None for every other model."""
        if self.model.task != "gene":
            return None
        gene = self.model.task_head.spec.gene
        return (float(gene.sp_lower), float(gene.sp_upper))

    def _result_label(self) -> str:
        """The served kind's label (`result_label`)."""
        return result_label(result_kind_for(self._scoring_mode()[0]))

    def bound_inputs(self) -> BoundInputs:
        """What this pipeline bound from its verified workspace (`BoundInputs`),
        for a caller that must not read those files again: the patients-like-me
        reference cohort's binding (the graph and the dataset record). Every
        pipeline has them: none is built without its workspace verified and its
        model read from a checkpoint.
        """
        return BoundInputs(
            graph=dict(self._bound_digests),
            dataset_record=self._dataset_record,
        )

    def _scoring_mode(self) -> Tuple[str, Optional[float]]:
        """The scoring mode in effect and the mixing weight it reports (the
        gene task's η; None for the disease and patients-like-me tasks, which
        mix nothing)."""
        task = self.model.task
        eta = self.model.task_head.spec.gene.eta if task == "gene" else None
        return TASK_SCORING_MODES[task], eta

    def served_identity(self) -> ServedIdentity:
        """What this pipeline serves, part by part (contract M3c, R8).

        **Built from the pipeline's own state, from the reads it took**: the
        checkpoint's digest from the read the model was built from, the graph's
        and the manifest's from the reads the workspace was verified with, the
        dataset record's origin from the record verified. Nothing is hashed again here.
        Every result the pipeline returns carries this, built when the result
        is produced, and `get_pipeline_config()` reports it from this same
        method. A part with nothing to identify says so in its `state`.
        """
        scoring_mode, eta = self._scoring_mode()

        checkpoint = ServedCheckpoint(state="verified", sha256=self._checkpoint_identity.sha256)
        graph = ServedGraph(
            state="verified",
            kg_sha256=self._graph_kg_digest,
            split_manifest_sha256=self._manifest_sha256,
        )

        shortest_paths = ServedShortestPaths(state=SCORING_MODE_SP_STATES[scoring_mode])

        record = self._dataset_record
        dataset_record = ServedDatasetRecord(
            state="verified", origin=record.origin, sha256=record.identity.sha256
        )

        settings = ServedSettings(
            device=self._device_in_effect,
            device_requested=self._device_requested,
            eta=float(eta) if eta is not None else None,
            max_path_length=self.config.max_path_length,
            path_length_penalty=self.config.path_length_penalty,
            max_paths_per_candidate=self.config.max_paths_per_candidate,
            use_direct_path_optimization=self.config.use_direct_path_optimization,
            phenotype_normalisation_version=PHENOTYPE_NORMALISATION_VERSION,
        )
        return ServedIdentity(
            scoring_mode=scoring_mode,
            checkpoint=checkpoint,
            graph=graph,
            shortest_paths=shortest_paths,
            dataset_record=dataset_record,
            settings=settings,
        )

    def served_checkpoint_read(self) -> "ReadIdentity":
        """The checkpoint read the served model was built from.

        For a reader that needs the read's path as well as its digest: the
        recorded results view finds the ledger beside the checkpoint and filters
        it by digest (`src/api/routes/evaluations.py`). Both come from this one
        identity, so the directory looked in and the digest looked up cannot
        belong to two different loads, which a path taken from application state
        beside a digest taken from here could during a reload. The path is the
        checkpoint's resolved path, as it was read (`_load_model_from_checkpoint`),
        so its directory is the one the loaded bytes were in, whatever a link
        named at load time points to now.

        The same read `served_identity()` names the checkpoint by. Nothing is
        hashed or read here.
        """
        return self._checkpoint_identity

    def get_pipeline_config(self) -> Dict[str, Any]:
        """Get pipeline configuration as dictionary."""
        # The served task's scoring mode and the η it reports (the gene task's;
        # None for the other tasks), as its served identity states them.
        scoring_mode, effective_eta = self._scoring_mode()

        return {
            # What is served, part by part (contract M3c): the same value every
            # result carries, from the same builder.
            "served_identity": self.served_identity().to_dict(),
            "max_path_length": self.config.max_path_length,
            "path_length_penalty": self.config.path_length_penalty,
            "scoring_mode": scoring_mode,
            "eta_effective": effective_eta,
            # The workspace's dataset record (contract M3b): its binding state
            # and its origin apart, so a synthetic workspace is never shown as
            # plainly verified. None when no workspace was verified.
            "dataset_record": self._dataset_record.summary(),
            "include_explanations": self.config.include_explanations,
            "include_ortholog_evidence": self.config.include_ortholog_evidence,
            "include_literature_evidence": self.config.include_literature_evidence,
            "has_model": True,
            "gnn_ready": self._gnn_ready,
            "checkpoint_meta": getattr(self, "_checkpoint_meta", {}),
            "kg_nodes": len(self.kg._nodes) if self.kg else 0,
            "kg_edges": len(self.kg._edges) if self.kg else 0,
        }


# ==============================================================================
# Factory Function
# ==============================================================================
def build_served_pipeline(
    kg_path: Union[str, Path],
    checkpoint_path: Optional[str],
    data_dir: Optional[str],
    device: Optional[str] = None,
    expected_checkpoint_sha256: Optional[str] = None,
    checkpoint_within: Optional[Path] = None,
) -> DiagnosisPipeline:
    """Build a pipeline as the service builds it: settle the device, read the
    graph once, and create the pipeline over it with the default configuration.

    **The service's own build, in one place** (Batch 3). `src/api/main.py`'s
    `build_pipeline` resolves its paths from the request or the environment,
    records what was requested and then calls this, so the service's pipeline is
    built in one place, never by a second construction.

    **No checkpoint, no pipeline** (`require_model_source`), refused before the
    graph is read: a graph is never served without a trained model. The device
    is settled before the graph is read too (contract M3c): `auto` without CUDA
    refuses before a multi-gigabyte `kg.json` is parsed for a model that cannot
    be served here. The pipeline applies the same rules itself; the device
    request is passed on as given, so the served identity records what was
    asked for beside what took effect.
    """
    from src.kg.graph import KnowledgeGraph

    require_model_source(checkpoint_path, data_dir)
    resolve_serving_device(device)
    kg_file = Path(kg_path)
    logger.info(f"Loading knowledge graph from {kg_file}...")
    # **One read, kept whole** (contract M2.1, S6). The graph and the identity of
    # the bytes it was parsed from travel together into the pipeline, which
    # checks that identity against the workspace manifest. Nothing hashes the
    # file again by path.
    graph = KnowledgeGraph.read_json(kg_file)
    kg = graph.kg
    logger.info(f"KG loaded: {kg.total_nodes} nodes, {kg.total_edges} edges")
    # **The graph read is passed whole, and the composition is checked there.**
    # The graph and the workspace can be named independently, so a deployment
    # can point them at two different workspaces -- each internally consistent,
    # and together producing embeddings from one graph's tensors read through
    # the other graph's node identifiers. The pipeline refuses that composition.
    return create_diagnosis_pipeline(
        graph=graph,
        checkpoint_path=checkpoint_path,
        data_dir=data_dir,
        device=device,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        checkpoint_within=checkpoint_within,
    )


def create_diagnosis_pipeline(
    *,
    graph: "GraphRead",
    checkpoint_path: str,
    data_dir: str,
    config: Optional[PipelineConfig] = None,
    device: Optional[str] = None,
    expected_checkpoint_sha256: Optional[str] = None,
    checkpoint_within: Optional[Path] = None,
) -> DiagnosisPipeline:
    """
    Factory function to create a diagnosis pipeline: a trained model read from
    `checkpoint_path`, over the workspace `data_dir` verified against `graph`
    (`DiagnosisPipeline`, which refuses anything less).

    Args:
        graph: The `GraphRead` `KnowledgeGraph.read_json` returned, passed
            whole, so the composition can be checked
        checkpoint_path: Path to the trained model's checkpoint
        data_dir: The workspace directory
        config: Pipeline configuration
        device: Device for the model
        expected_checkpoint_sha256: The SHA-256 the checkpoint read must have
            (`DiagnosisPipeline`)
        checkpoint_within: The resolved directory the checkpoint file must lie
            inside (`DiagnosisPipeline`); None for no bound

    Returns:
        Configured DiagnosisPipeline instance
    """
    return DiagnosisPipeline(
        graph=graph,
        checkpoint_path=checkpoint_path,
        data_dir=data_dir,
        config=config,
        device=device,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        checkpoint_within=checkpoint_within,
    )
