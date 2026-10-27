"""
# ==============================================================================
# Module: src/inference/pipeline.py
# ==============================================================================
# Purpose: End-to-end inference pipeline for rare disease diagnosis
#
# Dependencies:
#   - External: torch (optional, for GNN inference)
#   - Internal: src.core.types, src.kg, src.reasoning, src.models.gnn
#
# Input:
#   - PatientPhenotypes: Patient's phenotype data (HPO IDs)
#   - KnowledgeGraph: Pre-built knowledge graph
#   - Model (optional): Pre-trained GNN model for scoring
#
# Output:
#   - InferenceResult: Complete diagnosis result with candidates and explanations
#
# Design Notes:
#   - Core functionality (P0): Phenotype -> Gene -> Disease reasoning
#   - GNN scoring (P0): Neural phenotype-disease similarity via trained GNN
#   - Ortholog support (P1): Cross-species evidence (interfaces preserved)
#   - Scoring: GNN-primary when a model is configured; path reasoning only when
#     none is. A configured model that cannot be built raises PipelineBuildError
#   - PathReasoner: Explanation-only role (evidence paths, not scoring)
#   - Interpretable: Full evidence paths and human-readable explanations
#   - Production-ready: Input validation, error handling, logging
#
# Usage:
#   from src.inference import DiagnosisPipeline
#
#   # Path-only reasoning (no GNN)
#   pipeline = DiagnosisPipeline(kg=knowledge_graph)
#
#   # With GNN scoring from checkpoint, over a workspace
#   pipeline = DiagnosisPipeline(
#       graph=KnowledgeGraph.read_json("data/workspaces/demo/kg.json"),
#       checkpoint_path="checkpoints/last.pt",
#       data_dir="data/workspaces/demo",
#   )
#   result = pipeline.run(patient_phenotypes, top_k=10)
# ==============================================================================
"""
from __future__ import annotations

import copy
import logging
import math
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING

from src.kg.artifacts import (
    GRAPH_ARTIFACTS,
    read_split_manifest,
    verify_dataset_record,
    verify_graph_source_read,
)
from src.kg.storage.file_storage import read_graph_artifacts
from src.utils.checkpoint_io import CheckpointChanged, read_checkpoint
from src.kg.phenotype_normalisation import (
    PHENOTYPE_NORMALISATION_VERSION,
    normalise_phenotypes,
)
from src.core.types import (
    SERVED_MIXTURE,
    Candidate,
    DiseaseTaskCandidate,
    GeneTaskCandidate,
    PatientCandidate,
    DataSource,
    EvidenceSource,
    InferenceResult,
    NodeID,
    NodeType,
    PathReasoningCandidate,
    PatientPhenotypes,
    PhenotypeInputAccount,
    ResultRefused,
    SPAnalysisRecord,
    ServedMixtureCandidate,
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
from src.utils.run_environment import observe_autocast_regime

# Optional torch dependency for GNN inference
try:
    import torch
    from torch import Tensor
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if TYPE_CHECKING:
    from src.kg import KnowledgeGraph
    from src.kg.graph import GraphRead
    from src.models.gnn.shepherd_gnn import ShepherdGNN, ShepherdGNNConfig
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
#: `val_loss` and the ranking metrics are the trainer's own validation numbers,
#: computed as `Trainer._run_evaluation_pass` computes them; `train_loss` is its
#: training loss. None of them is a test result.
CHECKPOINT_LOG_METRICS: Tuple[str, ...] = ("val_loss", "train_loss", *RANKING_SCORE_KEYS)


# ==============================================================================
# Type Aliases for Callbacks
# ==============================================================================
# Callback for custom scoring: (candidate, paths, kg) -> modified_score
ScoringCallback = Callable[
    ["Candidate", List[ReasoningPath], "KnowledgeGraph"],
    float
]
# Callback for post-processing candidates
PostProcessCallback = Callable[
    [List["Candidate"], PatientPhenotypes],
    List["Candidate"]
]


# ==============================================================================
# Configuration
# ==============================================================================
@dataclass
class PipelineConfig:
    """Configuration for the diagnosis pipeline."""

    # Path reasoning
    max_path_length: int = 4
    path_length_penalty: float = 0.9
    aggregation_method: str = "weighted_sum"

    # Scoring architecture (per original SHEPHERD paper):
    #   final_score = eta * embedding_similarity + (1 - eta) * SP_similarity
    #
    # - embedding_similarity: GNN cosine sim between patient and target
    # - SP_similarity:        derived from pre-computed shortest path lengths
    #                         (loaded from shortest_paths.pt; computed offline
    #                         by scripts/compute_shortest_paths.py)
    # - eta:                  mixing weight (1.0 = pure GNN, 0.0 = pure SP).
    #                         The paper chooses η by a search per task and does
    #                         not print the value; 0.7 is this project's default,
    #                         favouring the GNN's learned signal while letting SP
    #                         catch sparse-input cases.
    #
    # PathReasoner is explanation-only and does NOT contribute to scoring.
    eta: float = 0.7

    # Fallback when shortest path lookup table is not available.
    # If True, missing SP table is treated as eta=1.0 (pure GNN).
    # If False, pipeline raises at init when both signals are not available.
    sp_optional: bool = True

    ortholog_weight: float = 0.3  # P1: Weight for ortholog evidence

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

    # Extensibility: Custom scoring callbacks (P1-ready)
    # These will be called during scoring phase
    custom_scorers: List[ScoringCallback] = field(default_factory=list)
    post_process_callbacks: List[PostProcessCallback] = field(default_factory=list)


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
    #: graph node, and with a model served its row in the phenotype embedding
    #: table, in the same order. `rows` is None when no model is served.
    node_ids: List[NodeID] = field(default_factory=list)
    rows: Optional[List[int]] = None


@dataclass
class ServedRanking:
    """The served ranking of one input, whole and untruncated (`DiagnosisPipeline.rank`).

    **Everything `run` does up to and including its sort, nothing after it**
    (`docs/working/scorer-measurement/PLAN_B05.md` §2). `run` cuts it at
    `top_k` and builds matching phenotypes, explanations and the summary on what
    it keeps; Mode D reads it whole. `top_k` never changes the order, only where
    the list is cut, so a served rank `r` is what a clinician sees whenever the
    page's Top-K is at least `r`.

    `outcome` says which of `run`'s three returns this is: ``"refused"`` (the
    input validation refused it), ``"no_candidates"`` (nothing mapped to a graph
    node, or discovery reached no disease) or ``"ranked"``. `warnings` is the
    list `run` returns with it, as `run` would return it.

    **`discovery_ordinals[i]` is `candidates[i]`'s 0-based position in the list
    `_find_all_paths` built, before the served-score sort.** It is recorded where
    the sort happens, in the same candidate construction, because once the served
    score has sorted the list the discovery order cannot be recovered from it.
    Internal metadata: not a field of the API or of the result.
    """

    outcome: str
    warnings: List[str]
    patient_input: PatientPhenotypes
    phenotype_input: Optional[PhenotypeInputAccount] = None
    #: Every served candidate in the served order, ranked 1..N, of the kind the
    #: pipeline serves (`served_mixture/v1` or `path_reasoning/v1`).
    candidates: List[Candidate] = field(default_factory=list)
    discovery_ordinals: List[int] = field(default_factory=list)
    source_ids: List[NodeID] = field(default_factory=list)
    all_paths: Dict[str, List[ReasoningPath]] = field(default_factory=dict)


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
    """What a pipeline built over a verified workspace read and bound, for a caller
    that measures the served pipeline without reading those files again
    (Mode D, `docs/working/scorer-measurement/PLAN_B05.md` §3).

    Each value is the pipeline's own: the four graph digests its workspace check
    verified (`kg`, `node_features`, `edge_indices`, `num_nodes`), the dataset
    record it verified with them, the identity of the checkpoint bytes its model
    was built from and that checkpoint's recorded training inputs (for R10), the
    workspace manifest's digest, and the graph data it encoded. Read-only: a copy
    of the digests, and references the caller must not change.
    """

    graph: Dict[str, str]
    dataset_record: Any
    checkpoint: Optional["ReadIdentity"]
    checkpoint_training_inputs: Any
    manifest_sha256: Optional[str]
    graph_data: Optional[Dict[str, Any]]


class PipelineBuildError(RuntimeError):
    """The model this pipeline was asked to serve could not be built.

    **Raised, never logged and stepped past.** A missing checkpoint, one that
    did not build over the graph, a model with no graph, a missing PyTorch and
    required shortest paths that were not there each used to log and return,
    leaving a pipeline without its GNN. The service then published that pipeline
    as a success, and every diagnosis was answered with path-reasoning scores in
    the model's place (docs/working/PLAN_PROVENANCE_CONTRACT.md, M1). A caller
    that asks for a model gets one, or gets this. An unreadable checkpoint
    already raised, as torch's own error; it raises this now, naming the file.

    A pipeline built with no model source at all is a different case and is
    unchanged: it serves path reasoning, which is the state B-2's open policy
    question is about.
    """


class ServingDeviceUnavailable(PipelineBuildError):
    """The device serving was asked to run the model on cannot be had here."""


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


def _device_of(model: Any) -> Optional[str]:
    """torch's name for the device a model's parameters are on, or None.

    The device in effect, read from the model rather than from the request: a
    `cuda` request serves on `cuda:0`, and that is what is recorded. None for a
    model with no parameters to ask, which only an in-memory double is.
    """
    try:
        return str(next(iter(model.parameters())).device)
    except Exception:  # noqa: BLE001 — a model that cannot say states no device
        return None


class DiagnosisPipeline:
    """
    End-to-end diagnosis inference pipeline.

    Scoring architecture, one result kind per configuration (Spec 1 §2.1):
    - `served_mixture/v1`, when a trained model is served: each disease path
      discovery reaches is ranked by `served_score`, η × model term + (1 − η) ×
      SP with a shortest-path table, the model term alone without one.
    - `path_reasoning/v1`, when no model source is given at all: diseases are
      ranked by their paths' `path_score`. A model that was requested and cannot
      be built raises `PipelineBuildError` instead; it is never replaced by this
      fallback.

    PathReasoner role:
    - Always runs to find the diseases ranked and their evidence paths
      (Phenotype → Gene → Disease)
    - Provides human-readable reasoning chains for clinician review
    - Ranks nothing when a model is served

    Combines:
    - Knowledge graph path reasoning for explainability (P0 core)
    - GNN scoring for primary confidence (P0 core)
    - Ortholog evidence integration (P1 feature, optional)

    Usage:
        pipeline = DiagnosisPipeline(kg=knowledge_graph)
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
        kg: Optional["KnowledgeGraph"] = None,
        config: Optional[PipelineConfig] = None,
        model: Optional[Any] = None,  # Optional pre-loaded GNN model
        checkpoint_path: Optional[str] = None,
        graph_data: Optional[Dict[str, Any]] = None,
        data_dir: Optional[str] = None,
        graph: Optional["GraphRead"] = None,
        device: Optional[str] = None,
        expected_checkpoint_sha256: Optional[str] = None,
    ):
        """
        Initialize the diagnosis pipeline.

        Args:
            kg: An in-memory knowledge graph for reasoning, which makes no claim
                about a persisted workspace. Give this or `graph`, not both.
            config: Pipeline configuration
            model: Optional pre-loaded GNN model instance
            checkpoint_path: Path to trained model checkpoint (.pt file).
                            If provided and model is None, the model will be
                            loaded from this checkpoint automatically.
            graph_data: Pre-loaded graph data dict with keys:
                       "x_dict", "edge_index_dict", "num_nodes_dict".
                       Required for GNN inference if data_dir is not set.
            data_dir: Path to processed data directory containing
                     node_features.pt, edge_indices.pt, num_nodes.json.
                     Alternative to graph_data for GNN inference.
            graph: The `GraphRead` `KnowledgeGraph.read_json` returned: the
                     graph and the identity of the bytes it was parsed from, in
                     one value. **Required whenever `data_dir` is used for GNN
                     inference**, and checked: `data_dir` proves a workspace is
                     internally consistent, and two consistent workspaces can
                     still be crossed, one's `kg.json` paired with the other's
                     tensors, so embedding rows are read through another graph's
                     node identifiers. It is passed whole, so when it is the
                     value `read_json` returned, the graph served and the digest
                     checked come from one read (contract M2.1, S6); the type
                     does not enforce that pairing. An in-memory graph
                     (`kg`) has no source digest and is not pretended to have
                     one.
            device: Device for a checkpoint's model: "cuda", "cuda:N", "cpu",
                or None / "auto" for CUDA, refused without it
                (`resolve_serving_device`). A model supplied in memory stays
                on its own device when this is None.
            expected_checkpoint_sha256: The SHA-256 the checkpoint at
                `checkpoint_path` must have (the model a caller was shown, a
                bound reload). Compared on the one checkpoint read before the
                bytes are parsed (`read_checkpoint(expected_sha256=)`): other
                bytes, a checkpoint or not, or no file raise `CheckpointChanged`
                unchanged, and nothing about them is built. Only with
                `checkpoint_path` and no `model`; None reads unbound.
        """
        from src.kg.graph import GraphRead

        if expected_checkpoint_sha256 is not None and (checkpoint_path is None
                                                       or model is not None):
            raise ValueError(
                "expected_checkpoint_sha256 binds the checkpoint read from "
                "checkpoint_path; give checkpoint_path= and no model="
            )

        if graph is not None:
            if kg is not None:
                raise ValueError(
                    "pass the graph read from a file (graph=) or an in-memory "
                    "graph (kg=), not both: a graph and an identity given apart "
                    "could come from two different reads"
                )
            if not isinstance(graph, GraphRead):
                raise TypeError(
                    "graph= takes the GraphRead KnowledgeGraph.read_json returned, "
                    f"not a {type(graph).__name__}"
                )
            kg = graph.kg
        elif kg is None:
            raise ValueError("a pipeline needs a graph: pass kg= or graph=")
        self.kg = kg
        #: The identity of the bytes `self.kg` was parsed from, when it was read
        #: from a file. None for an in-memory graph.
        self._graph_identity: Optional["ReadIdentity"] = (
            graph.identity if graph is not None else None
        )
        self.config = config or PipelineConfig()
        self.model = model

        # GNN inference state (populated by _init_gnn_inference)
        self._node_embeddings: Optional[Dict[str, Any]] = None
        self._node_id_to_idx: Optional[Dict[str, Dict[str, int]]] = None
        self._graph_data: Optional[Dict[str, Any]] = graph_data
        #: The four digests the manifest binds (`kg`, `node_features`,
        #: `edge_indices`, `num_nodes`), each compared with the read it names,
        #: once the workspace has been verified. Kept whole for M2.4's
        #: comparison with the checkpoint; None for in-memory graph data, over
        #: which no checkpoint is loaded.
        self._bound_digests: Optional[Dict[str, str]] = None
        #: The workspace's dataset record, read once and verified with the graph
        #: (contract M3b); None when no workspace was verified. Published by
        #: `get_pipeline_config` with its binding state and origin apart.
        self._dataset_record: Optional[Any] = None
        #: The identity of the checkpoint bytes the model was built from.
        self._checkpoint_identity: Optional["ReadIdentity"] = None
        #: That checkpoint's recorded training inputs (`training_input_digests`),
        #: from the same one read, for a measurement's R10 (`bound_inputs`).
        self._checkpoint_training_inputs: Any = None
        #: The autocast regime the embedding forward ran under, observed at that
        #: computation (`observe_autocast_regime`); None until embeddings exist.
        self._embedding_regime: Optional[Any] = None
        #: SHA-256 of the one `split_manifest.json` read the workspace was
        #: verified against (contract M3c); None when no workspace was verified.
        self._manifest_sha256: Optional[str] = None
        #: The device asked for, as given ("auto" when a checkpoint was loaded
        #: with none named), and the one the model's parameters are on once it
        #: is built. Both are part of the served identity (contract M3c).
        self._device_requested: Optional[str] = (
            str(device).strip() if device is not None and str(device).strip() else None
        )
        self._device_in_effect: Optional[str] = None
        self._gnn_ready = False

        # Shortest path lookup state (populated by _load_shortest_paths).
        # `_sp_lookup` is the whole of it: a sorted composite key over
        # (phenotype, target_type, target) beside the distances in that order.
        # The parallel id columns and the CSR offsets table that used to live
        # here are gone — the key encodes them and the query searches it, so
        # keeping them would keep a second shape for a second implementation to
        # be written against. When `_sp_ready` is False the pipeline degrades to
        # pure GNN scoring (eta=1.0).
        self._sp_max_hops: int = 5
        #: Where `_sp_max_hops` came from: "sidecar", the only source since the
        #: configured bound was retired (contract M2.5). Published, so a status
        #: reader can see where the ceiling every SP score uses was read.
        self._sp_hop_bound_source: Optional[str] = None
        #: SHA-256 of the bytes this pipeline's graph was parsed from, as the
        #: manifest binds it, once the workspace has been verified. **None for a
        #: graph supplied in memory**, which is not an oversight: that caller made
        #: no claim about a persisted workspace, so there is nothing to bind an
        #: artifact to.
        #: The shortest-path artifact is checked against *this*, never against
        #: whatever `kg.json` happens to sit beside it — otherwise the binding
        #: could be bypassed by deleting a file.
        self._graph_kg_digest: Optional[str] = None
        #: "verified" once a table has been accepted; an unrecorded binding is
        #: refused (contract M2.5), so nothing else is published.
        self._sp_kg_binding: Optional[str] = None
        #: The `build_id` the accepted table and its sidecar share, published
        #: with `_sp_ready`; part of the served identity (contract M3c).
        self._sp_build_id: Optional[str] = None
        self._sp_lookup: Optional[Any] = None  # scoring.SPLookup, built on load
        self._sp_ready = False

        # Fingerprint verification warnings (populated by _load_model_from_checkpoint)
        self._fingerprint_warnings: List[str] = []
        self._checkpoint_meta: Dict[str, Any] = {}

        # Initialize reasoning components
        self._init_reasoning_components()
        self._init_explanation_generator()

        # Initialize GNN inference if resources are available
        self._init_gnn_inference(
            checkpoint_path=checkpoint_path,
            graph_data=graph_data,
            data_dir=data_dir,
            device=device,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
        )

        logger.info(
            f"DiagnosisPipeline initialized: "
            f"kg_nodes={len(self.kg._nodes)}, "
            f"has_model={self.model is not None}, "
            f"gnn_ready={self._gnn_ready}, "
            f"device={self._device_in_effect}"
        )

    def _init_reasoning_components(self) -> None:
        """Initialize path reasoning components."""
        reasoning_config = PathReasoningConfig(
            max_path_length=self.config.max_path_length,
            length_penalty=self.config.path_length_penalty,
            aggregation_method=self.config.aggregation_method,
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
        # uses self.path_reasoner; Mode B (analogy) reuses GNN embeddings
        # which are passed in at build_evidence() call time.
        self.evidence_panel = create_evidence_panel(
            kg=self.kg,
            path_reasoner=self.path_reasoner,
            config=EvidencePanelConfig(
                strong_path_max_hops=2,
                weak_path_max_hops=self.config.max_path_length,
                analogy_top_k=3,
            ),
        )

    # ==========================================================================
    # GNN Inference Initialization
    # ==========================================================================
    def _init_gnn_inference(
        self,
        checkpoint_path: Optional[str],
        graph_data: Optional[Dict[str, Any]],
        data_dir: Optional[str],
        device: Optional[str],
        expected_checkpoint_sha256: Optional[str] = None,
    ) -> None:
        """
        Initialize GNN model and precompute node embeddings for inference.

        This method handles the full setup chain:
        1. Load graph data (from graph_data dict or data_dir files)
        2. Load model from checkpoint (if no pre-loaded model)
        3. Run GNN forward pass to cache all node embeddings
        4. Build NodeID -> index mapping for score lookup
        """
        if not HAS_TORCH:
            if checkpoint_path is not None or self.model is not None:
                raise PipelineBuildError(
                    "A GNN was requested, and PyTorch is not available to load "
                    "it. No pipeline is built without the model it was asked "
                    "for; install torch."
                )
            return

        # Nothing to do if no model source is available
        if self.model is None and checkpoint_path is None:
            return

        # **The device is settled before any input is read** (contract M3c,
        # §5.2), as training's is: `auto` without CUDA refuses here, instead of
        # after the workspace, the checkpoint and the model build, and instead of
        # turning into the CPU. A model supplied in memory with no device named
        # stays where it is; there is nothing to resolve for it. One with a
        # device named gets the same rule, so `auto` or CUDA on a host without
        # it is this refusal, not torch's error after the workspace was read.
        if self.model is None:
            device = resolve_serving_device(device)
            self._device_requested = self._device_requested or "auto"
        elif self._device_requested is not None:
            device = resolve_serving_device(self._device_requested)
        else:
            device = None

        # Step 1: Load graph data
        if self._graph_data is None and data_dir is not None:
            # **The clinical path is a graph consumer, and the costliest one.**
            # A same-shaped `node_features.pt` from another workspace is refused
            # by training and by measurement, and without this it would still be
            # adopted here, paired with a checkpoint, precomputed into embeddings
            # and served. The tensors are parsed first (one read per input); the
            # check runs before they are adopted, before the checkpoint is read
            # and before anything is served.
            #
            # Only for the file-backed path. A caller supplying `graph_data`
            # directly makes no claim about a persisted workspace, so there is no
            # manifest for it to match and nothing to verify -- and, for that
            # reason, no checkpoint is loaded over it (M2.4,
            # `_load_model_from_checkpoint`); a model handed in is the caller's.
            # **The graph's identity is required here, not optional.** Verifying
            # only that `data_dir` is internally consistent leaves the `kg`
            # object bound to nothing: a caller can still pair workspace B's
            # graph with workspace A's tensors and get embeddings read through
            # the wrong node identifiers. An optional check is no check for the
            # caller who omits it, and that caller is the one this exists for.
            #
            # Nothing legitimate is lost. A caller who built its graph in memory
            # never had a source file and belongs on the `graph_data` seam above,
            # which makes no persisted-workspace claim; a caller who loaded one
            # from disk passes what `KnowledgeGraph.read_json` returned.
            if self._graph_identity is None:
                raise ValueError(
                    "loading graph data from a workspace requires graph= — the "
                    "GraphRead KnowledgeGraph.read_json returned, so the graph "
                    "served is the one whose bytes are checked. Without it the "
                    "graph object is bound to nothing, and this workspace's "
                    "tensors could belong to a different graph entirely. Pass "
                    "graph_data directly if the graph was not loaded from a file."
                )
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
            # checkpoint. The shortest-path binding reads `kg` from them, so it
            # binds to the graph that was verified, not to a file hashed again.
            self._bound_digests = dict(bound)
            self._graph_kg_digest = bound["kg"]
            self._dataset_record = dataset_record
            # The manifest's identity from the same one read, for the served
            # identity (contract M3c); nothing hashes it again.
            self._manifest_sha256 = manifest.identity.sha256
            # **Only after the check.** The reader omits a file that is absent,
            # and only the check turns that into a refusal.
            self._graph_data = graph_read.graph_data

        if self._graph_data is None:
            # A workspace without its tensors is refused by the check above, so
            # only an absent source is left to report.
            raise PipelineBuildError(
                "A GNN was requested and there is no graph to compute its "
                "embeddings from: neither data_dir (SHEPHERD_DATA_DIR, on the "
                "service) nor graph_data was given."
            )

        # Step 2: Load model from checkpoint if needed. A checkpoint that does
        # not load raises out of here (`_load_model_from_checkpoint`).
        if self.model is None and checkpoint_path is not None:
            self.model = self._load_model_from_checkpoint(
                checkpoint_path, device, expected_sha256=expected_checkpoint_sha256
            )

        # Step 3: Precompute node embeddings
        self._precompute_node_embeddings(device)

        # Step 4: Build NodeID -> index mapping from KG
        self._node_id_to_idx = self.kg.get_node_id_mapping()

        # Step 5: Load shortest path lookup table (optional). A disease task
        # model ranks without it (README §6: SP is the gene task's feature).
        if data_dir is not None and self._bundle_view is None:
            self._load_shortest_paths(Path(data_dir))

        if not self._sp_ready and not self.config.sp_optional and self._bundle_view is None:
            # **Names the state, not a guess at its cause.** This used to say
            # "no shortest_paths.pt found", which is false whenever the table is
            # present and something else about it is unusable — an unknown hop
            # bound, for one. The loader logs the specific reason at the point it
            # establishes it; this says what the configuration then does about it.
            #
            # **A refusal, which is what `sp_optional=False` documents.** It used
            # to log and switch the GNN off, so a configuration that required both
            # signals was served with neither: path reasoning stood in for both.
            why = (
                "see the shortest-path load messages above for which condition "
                "stopped it"
                if data_dir is not None
                else "no data_dir was given, so there was no table to load"
            )
            raise PipelineBuildError(
                "Shortest path lookup required (sp_optional=False) but it is "
                f"not available; {why}. No pipeline is built."
            )

        # **`_gnn_ready` means built, not requested.** Every loader above raises
        # rather than returning empty, so this is unreachable today; it is here so
        # a loader that one day returns None again cannot leave a pipeline that
        # reports a GNN it cannot score with. The two node types are the two
        # `_calculate_gnn_score` and `_patient_embedding` read; without either, a
        # request would be refused at scoring time (`MODEL_TERM_UNAVAILABLE`).
        embeddings = self._node_embeddings or {}
        missing = [
            node_type.value
            for node_type in (NodeType.PHENOTYPE, NodeType.DISEASE)
            if embeddings.get(node_type.value) is None
        ] if self._bundle_view is None else ([] if self._bundle_z is not None else ["the view"])
        if self.model is None or missing:
            raise PipelineBuildError(
                "The GNN was requested and produced no node embeddings for "
                f"{missing or 'any node type'}. No pipeline is built."
            )

        self._device_in_effect = _device_of(self.model)
        self._gnn_ready = True
        logger.info(
            f"GNN inference initialized successfully "
            f"(sp_ready={self._sp_ready}, eta={self.config.eta}, "
            f"device={self._device_in_effect})"
        )

    def _load_shortest_paths(self, data_dir: Path) -> None:
        """Load the shortest-path table and build the index the scorer reads.

        Expected format (produced by `scripts/compute_shortest_paths.py`):
            shortest_paths.pt: dict with keys
                phenotype_idx: int64 tensor (N,)
                target_idx:    int64 tensor (N,)
                target_type:   int64 tensor (N,) — 0=gene, 1=disease
                distance:      int8  tensor (N,)
            shortest_paths.meta.json: {"max_hops": int, ...}

        **The four columns do not survive this method, and that is the design.**
        They are validated, handed to `build_sp_index`, and folded into one
        sorted int64 composite key beside the distances. Nothing reads them
        back — the query searches the key — so the CSR offsets table and the
        parallel `_sp_ph` / `_sp_tg` / `_sp_ty` / `_sp_di` attributes this class
        used to publish are gone. Keeping them would be keeping the copy this
        design exists to avoid, and would keep alive the shape a second
        implementation could be written against.

        **Order matters here more than anywhere else in this class.** The hop
        bound is resolved *before* the table is validated, because the bound is
        what the distances are checked against; validation runs *before* any
        narrowing, because a value a narrow type cannot hold wraps rather than
        raising; and `_sp_ready` is published *after* the index exists, because
        every step between can refuse.

        A missing file is not a refusal: the pipeline serves without shortest
        paths, which is the state `DISEASE_SCORER_POLICY.md` §2 blesses. A file
        that is present and unusable is refused, because absent-by-configuration
        and present-but-broken are different deployments and only one of them
        was chosen.
        """
        # **Cleared on entry, not assumed clean.** This is re-callable, and a
        # second call that refuses must not leave `_sp_ready` True from the
        # first, with a lookup over the previous table's bytes and the previous
        # ceiling while the observable attributes show the new ones.
        self._sp_ready = False
        self._sp_lookup = None
        self._sp_hop_bound_source = None
        self._sp_kg_binding = None
        self._sp_build_id = None

        from src.inference.scoring import validate_hop_bound
        from src.inference.sp_artifact import (
            declares_binding,
            kg_binding_state,
            read_binding,
            read_sidecar,
            require_paired,
            sidecar_path,
        )
        from src.inference.sp_index import (
            SPArtifactError,
            build_sp_index,
            validate_sp_columns,
            validate_sp_values,
        )

        sp_path = data_dir / "shortest_paths.pt"

        # **Read once, and before the absent decision.** A sidecar that declares
        # the pairing protocol beside a table that is not here is one side of a
        # pair, not an absent artifact — a publication that lost its tensor, or
        # a tensor deleted from a workspace that still advertises one. Treating
        # it as absent takes the "serve without shortest paths" path for a
        # deployment that is broken, and on a reload replaces a healthy service
        # with one scoring on the GNN alone.
        sidecar = read_sidecar(sp_path)
        if not sp_path.exists():
            if sidecar is not None and declares_binding(sidecar.meta, ()):
                raise SPArtifactError(
                    f"{sidecar_path(sp_path)} declares a published pair and "
                    f"{sp_path} is not here. One side of a pair is not an "
                    "absent artifact: the table this sidecar describes was "
                    "either never written or has been removed."
                )
            logger.info(
                f"No shortest_paths.pt found in {data_dir}; "
                f"scoring will use pure GNN (eta=1.0 effective)"
            )
            return

        # **Present and unreadable is a refusal, not a shrug.** This warned and
        # returned, which made a corrupt table indistinguishable from an absent
        # one: the operator saw the same "scoring will use pure GNN" they would
        # have seen with no file at all, and the remedy for each is the
        # opposite. Removing the file reaches the absent path deliberately.
        try:
            sp_data = torch.load(sp_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise SPArtifactError(
                f"{sp_path} is present and could not be read "
                f"({type(exc).__name__}: {exc}). An unreadable table is not an "
                "absent one; remove the file to serve without shortest paths."
            ) from exc

        required = ("phenotype_idx", "target_idx", "target_type", "distance")
        missing = [key for key in required if key not in sp_data]
        if missing:
            raise SPArtifactError(
                f"{sp_path} is missing {missing}. The columns are parallel and "
                "the index cannot be built from a subset of them."
            )
        # **Structure first, and "how many rows" is its answer rather than its
        # input.** Reading `distance.numel()` directly and branching on it looked
        # equivalent and was not: it asks one column how long it is and then
        # skips every check on all four. Four malformed tables reached the
        # absent path that way — `distance` emptied while the id columns kept a
        # row, a `(0, 2)` distance column, an empty float column, and an id
        # column that was a Python list — and on a reload each of them replaced
        # a healthy SP-ready pipeline with one serving on the GNN alone, while
        # reporting success.
        n_pairs = validate_sp_columns(
            sp_data["phenotype_idx"],
            sp_data["target_idx"],
            sp_data["target_type"],
            sp_data["distance"],
            source=str(sp_path),
        )

        # **Before the empty decision, not after it.** These checks used to sit
        # below the early return, so a mixed publication whose new tensor
        # happened to be empty reached "no rows, treat as absent" and switched
        # shortest paths off — and on a reload that replaced a healthy SP-ready
        # service with one scoring on the GNN alone, reporting success. A pair
        # that does not claim itself is refused whether it carries a million
        # rows or none.
        binding = read_binding(
            sidecar.meta if sidecar else None, sp_data.keys(), source=str(sp_path)
        )
        require_paired(binding, sp_data.get("build_id"), source=str(sp_path))
        binding_state = kg_binding_state(
            binding, self._graph_kg_digest, source=str(sp_path)
        )
        if binding_state == "unverifiable":
            raise SPArtifactError(
                f"{sp_path} records the graph it was computed from, and this "
                "pipeline cannot check it: its graph was supplied in memory, so "
                "no verified workspace digest exists to compare against. The "
                "shortest-path term is part of the ranking, so an unverifiable "
                "binding is refused rather than assumed. Build the pipeline from "
                "a workspace; a table published without a binding is refused "
                "too (contract M2.5)."
            )
        if binding_state == "unrecorded":
            # **Refused, no longer served with a warning** (contract M2.5,
            # decision 2). A table that predates the pairing protocol never
            # recorded which graph its node indices are positions in, and the
            # shortest-path term is part of the ranking.
            raise SPArtifactError(
                f"{sp_path} carries no source binding: it predates the pairing "
                "protocol, so which graph it was computed from is unrecorded. "
                "An unrecorded table is not served. Rebuild it with "
                "scripts/compute_shortest_paths.py to record one, or remove it "
                "to serve without shortest paths."
            )

        # **A table with no rows is the absent case, not a ready one.** D5 took
        # the weaker of the two defensible readings deliberately: an empty table
        # makes no false claim, and is indistinguishable in effect from having
        # no file — which is a state `DISEASE_SCORER_POLICY.md` §2 blesses.
        #
        # Publishing it instead is not merely a different object shape. A query
        # against an empty index returns `unreachable` with `available` True, so
        # the mixture would take SP in at `1 / (1 + max_hops + 1)`
        # for every candidate — measured at 1/7 with the default bound, against
        # the pure-GNN score an absent file produces. Zero rows would change
        # both what the service reports and the numbers it returns.
        #
        # **Taken after the pair has been checked and before the hop bound is
        # resolved**, and the comment this replaces described the older order.
        # An empty table now reaches here having had its sidecar read, parsed,
        # and its schema, pairing and graph binding validated — a corrupt
        # sidecar beside an empty table is refused rather than ignored, which
        # is what moving those checks above this return bought.
        #
        # What an empty table still skips is exactly `validate_hop_bound`:
        # `max_hops` missing, zero or non-numeric is not refused here, because
        # the bound is the sentinel distances are scored against and nothing
        # will be scored. Measured, so the gap is stated rather than assumed —
        # each of those three publishes an empty table with SP off, while a
        # sidecar that is not readable JSON refuses.
        if n_pairs == 0:
            logger.warning(
                "%s carries no rows. An empty table binds nothing, so it is "
                "treated as absent: shortest-path scoring stays off and the "
                "pipeline serves on the GNN alone (eta=1.0 effective). Rebuild "
                "it with scripts/compute_shortest_paths.py if this is not "
                "expected.",
                sp_path,
            )
            return

        # **The ceiling, before the table is checked against it and before any
        # of the expensive work.** `max_hops` sets the unreachable sentinel every
        # shortest-path score is measured against, so a wrong one does not merely
        # mis-score — it reorders candidates. It is resolved first because
        # `validate_sp_artifact` checks the distances against it, and because a
        # deployment that cannot establish it should not pay for a sort it is
        # about to discard.
        meta_path = sidecar_path(sp_path)
        # **From the sidecar alone** (contract M2.5). A table that reaches here
        # has one: without it the table records no binding and was refused
        # above. The configured bound that once stood in for a missing sidecar
        # (`sp_hop_bound`, `SHEPHERD_SP_HOP_BOUND`) is retired with it, since it
        # could no longer reach a served table; nothing measurable on the
        # tensors can stand in either, as they bound the ceiling from below only.
        #
        # **The same parsed object the binding was checked against.** Reading
        # the file a second time for the bound would let a publication that
        # completes in between hand this one version and the pairing check
        # another.
        max_hops = validate_hop_bound(sidecar.meta.get("max_hops"), str(meta_path))
        hop_bound_source = "sidecar"

        # **The value rules, which are the half that needed the bound.** The
        # producer's `target_type` encoding, and distances inside `1 .. max_hops`.
        # The ceiling is a floor and honest about being one: `max(distance)`
        # proves the declared bound is not too LOW and can never prove it is not
        # too high, because a 3-hop table is consistent with a declared 5. That
        # one direction is the direction a stale sidecar from a smaller run fails
        # in, and there the sentinel lands *below* distances really in the table,
        # so an unreachable phenotype outranks a connected one. Both run on the
        # tensors as loaded, before anything is cast.
        validate_sp_values(
            sp_data["target_type"],
            sp_data["distance"],
            max_hops,
            source=str(sp_path),
        )

        # **`pop`, not `[]`.** The builder reads the columns and never writes
        # them, but it releases its own reference to each as it folds it into the
        # key — and that only frees anything if this method is not still holding
        # one. At hundreds of millions of rows the difference is the whole of the
        # saving.
        lookup = build_sp_index(
            sp_data.pop("phenotype_idx"),
            sp_data.pop("target_idx"),
            sp_data.pop("target_type"),
            sp_data.pop("distance"),
            max_hops,
        )
        del sp_data

        # **Published last.** `_sp_ready` used to be set before the sidecar was
        # read and before the lookup existed, so a refusal landed on a pipeline
        # already advertising a usable lookup — and on a second call it was
        # already True, making "publish only after validation" a no-op. The index
        # build joins that same fallible tail: it can refuse for duplicate rows
        # or a domain int64 cannot hold, and those refusals must land before
        # anything here is set.
        self._sp_max_hops = max_hops
        self._sp_hop_bound_source = hop_bound_source
        self._sp_kg_binding = binding_state
        self._sp_build_id = binding.build_id
        self._sp_lookup = lookup
        self._sp_ready = True

        logger.info(
            f"Loaded shortest paths: {n_pairs:,} pairs, "
            f"max_hops={max_hops} ({hop_bound_source}), "
            f"index {lookup.resident_bytes() / 1e9:.2f} GB resident"
        )

    def _load_model_from_checkpoint(
        self,
        checkpoint_path: str,
        device: Optional[str],
        *,
        expected_sha256: Optional[str] = None,
    ) -> Any:
        """
        Load ShepherdGNN model from a training checkpoint.

        Handles both checkpoint formats:
        - Trainer format: key "model_state_dict"
        - ModelCheckpoint callback format: key "state_dict"

        With `expected_sha256` (a bound reload) the one read is compared with it
        before torch parses the bytes, and the reader decides when the file is
        gone: there is no existence check first.

        Raises:
            CheckpointChanged: `expected_sha256` is given and the read has other
                bytes, or there is no file; raised unchanged, before anything is
                parsed, so the caller can say the file is not the model shown.
            PipelineBuildError: if the graph was not verified from a workspace,
                or the checkpoint is missing, cannot be read, is not a training
                checkpoint, was not trained on this graph (M2.4), or does not
                build a model over it. A missing checkpoint and one that did not
                build used to return None, and the pipeline went on without its
                model.
        """
        from src.models.gnn.shepherd_gnn import build_shepherd_model

        ckpt_path = Path(checkpoint_path)
        # **A checkpoint is used only over a graph verified from a workspace**
        # (contract M2.4). Its record can be compared only with the digests the
        # workspace check took from this build's reads; a graph supplied in
        # memory (`graph_data=`) has none, so nothing shows the model was trained
        # on it, and that is a refusal, not an exemption. Before the checkpoint
        # is opened.
        if self._bound_digests is None:
            raise PipelineBuildError(
                f"Checkpoint {ckpt_path} is loaded only over a graph verified "
                "from a workspace, and this pipeline's graph was supplied in "
                "memory: nothing shows the checkpoint was trained on it. Build "
                "the pipeline from the workspace (graph= and data_dir=)."
            )
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
        try:
            checkpoint_read = read_checkpoint(
                ckpt_path.resolve(), map_location="cpu", weights_only=True,
                expected_sha256=expected_sha256,
            )
        except CheckpointChanged:
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
        # model is built; the structural fingerprint below cannot tell them apart.
        from src.utils.fingerprint import (
            TrainingRecordMismatch,
            verify_fingerprint,
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

        # Verify data fingerprint compatibility (KG version check). Pipeline
        # business, not construction: it decides what an operator is warned
        # about. A diagnostic beside the comparison above, never identity.
        fp_warnings = verify_fingerprint(
            checkpoint,
            self._graph_data,
            kg_total_nodes=len(self.kg._nodes) if self.kg else None,
            kg_total_edges=len(self.kg._edges) if self.kg else None,
        )
        if fp_warnings:
            for w in fp_warnings:
                logger.warning(f"[Fingerprint] {w}")
        self._fingerprint_warnings = fp_warnings

        if self._graph_data is None:
            raise PipelineBuildError(
                "Cannot reconstruct model: graph_data not loaded. "
                "_init_gnn_inference must load graph_data before model."
            )

        # Construction is shared with the measurement harness
        # (`src/models/gnn/shepherd_gnn.py:build_shepherd_model`), so measurement's
        # Mode C cannot encode with semantics that have drifted from the served model.
        # Metadata derivation from graph_data keys, architecture recovery and
        # weight loading all live there; what stays here is what only a running
        # pipeline needs.
        from src.models.bundle.record import has_model_spec

        if has_model_spec(checkpoint):
            model = self._load_task_bundle(checkpoint, ckpt_path)
        else:
            model = None
        try:
            if model is None:
                model = build_shepherd_model(checkpoint, self._graph_data)
        except KeyError as exc:
            # A format question, not a graph one: no weights to build from.
            raise PipelineBuildError(
                f"Checkpoint {ckpt_path} carries no model weights: {exc}"
            ) from exc
        except (ValueError, RuntimeError) as exc:
            raise PipelineBuildError(
                f"Checkpoint {ckpt_path} does not build a model over this graph: "
                f"{exc}"
            ) from exc

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
        self._checkpoint_meta = {
            "epoch": _as_int(checkpoint.get("epoch")),
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
        # Kept from the same read for a measurement's R10 (Mode D): the model
        # served and the inputs it records cannot come from two reads.
        self._checkpoint_training_inputs = copy.deepcopy(
            checkpoint.get("training_input_digests")
        )
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
        loaded by the one builder. Nothing is inferred from weight names."""
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
        Run GNN forward pass on the full graph and cache all node embeddings.

        Embeddings are moved to CPU after computation for memory efficiency.
        """
        if self.model is None or self._graph_data is None:
            return

        if device is None:
            device = next(self.model.parameters()).device
        else:
            device = torch.device(device)

        if self._bundle_view is not None:
            # The full-view rule over the whole view (README §2.8, G4), once;
            # the diseases' projections are fixed, so each request projects
            # only its patient.
            from src.models.tasks.scoring import full_view_vectors

            regime = observe_autocast_regime(device)
            view = self._bundle_view
            self._bundle_z = full_view_vectors(self.model, view.edge_index(), view.rel, device)
            self._embedding_regime = regime
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
            return

        logger.info("Precomputing node embeddings via GNN forward pass...")

        # Move graph data to model device
        x_dict = {
            k: v.to(device) for k, v in self._graph_data["x_dict"].items()
        }
        edge_index_dict = {
            k: v.to(device)
            for k, v in self._graph_data["edge_index_dict"].items()
        }

        # Forward pass (no gradient needed for inference). The regime is observed
        # here, at the computation, as Mode C observes its own: a caller's
        # autocast context is what these embeddings were computed under.
        regime = observe_autocast_regime(device)
        with torch.no_grad():
            embeddings = self.model(x_dict, edge_index_dict)
        self._embedding_regime = regime

        # Move embeddings to CPU for memory efficiency
        self._node_embeddings = {
            k: v.cpu() for k, v in embeddings.items()
        }

        emb_info = {k: v.shape for k, v in self._node_embeddings.items()}
        logger.info(f"Node embeddings cached: {emb_info}")

    def rank(
        self,
        patient_input: PatientPhenotypes,
        include_ortholog_evidence: Optional[bool] = None,
        reference_cohort: Optional[ReferenceCohort] = None,
    ) -> ServedRanking:
        """The served ranking of `patient_input`, whole: everything `run` does
        up to and including its sort, and nothing after it.

        Validation and normalisation, the node mapping, path discovery, scoring
        and the stable sort by the served score, exactly as `run` does them; `run`
        is this followed by its cut at `top_k` and what it builds on the rows it
        keeps. **One implementation**: a caller that needs the untruncated served
        order (Mode D, `docs/working/scorer-measurement/PLAN_B05.md` §2) calls
        this, and nothing constructs candidates a second way.
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

        if self._bundle_view is not None:
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
        if reference_cohort is not None:
            raise ValueError("a reference cohort is ranked by a patients-like-me model only")

        # Step 3: Find reasoning paths (BFS-based)
        logger.debug("Finding reasoning paths...")
        all_paths = self._find_all_paths(source_ids, include_ortholog_evidence)

        if not all_paths:
            logger.warning("No paths found from phenotypes to diseases")
            return ServedRanking(
                outcome="no_candidates",
                warnings=warnings + ["No paths found from phenotypes to diseases"],
                patient_input=patient_input,
                phenotype_input=validation.phenotype_input,
                source_ids=source_ids,
            )

        # Step 4: Score and rank every candidate
        logger.debug("Scoring candidates...")
        candidates, ordinals = self._score_and_rank_candidates(
            all_paths=all_paths,
            source_ids=source_ids,
            patient_input=patient_input,
            include_ortholog_evidence=include_ortholog_evidence,
            rows=validation.rows,
            unmapped_ids=(validation.phenotype_input.unknown
                          if validation.phenotype_input else ()),
        )
        return ServedRanking(
            outcome="ranked",
            warnings=warnings,
            patient_input=patient_input,
            phenotype_input=validation.phenotype_input,
            candidates=candidates,
            discovery_ordinals=ordinals,
            source_ids=source_ids,
            all_paths=all_paths,
        )

    def _rank_by_disease_task(self, patient_input: PatientPhenotypes,
                              validation: "ValidationResult", warnings: List[str],
                              source_ids: List[NodeID]) -> ServedRanking:
        """`disease_task/v1`: every disease of the model's graph view, ranked by
        the disease head (README A7, U6), ties by view id. A phenotype outside
        the view is dropped and said so; with none left, nothing is ranked."""
        from src.models.tasks.scoring import patient_vectors

        view = self._bundle_view
        rows = torch.tensor(list(validation.rows or []), dtype=torch.long)
        view_ids = view.view_ids("phenotype", rows) if rows.numel() else rows
        kept = view_ids[view_ids >= 0]
        dropped = int((view_ids < 0).sum())
        if dropped:
            warnings = warnings + [
                f"{dropped} phenotype(s) are outside the model's graph view and were not "
                "scored"]
        if kept.numel() == 0:
            return ServedRanking(
                outcome="no_candidates",
                warnings=warnings + ["None of the phenotypes is in the model's graph view"],
                patient_input=patient_input, phenotype_input=validation.phenotype_input,
                source_ids=source_ids)
        z = self._bundle_z
        kept = torch.unique(kept).to(z.device)
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
            discovery_ordinals=order, source_ids=source_ids, all_paths={})

    def _bundle_phenotypes(self, validation: "ValidationResult", warnings: List[str]
                           ) -> Tuple[Any, List[str]]:
        """The request's phenotypes as view ids of the model's graph, each
        outside it dropped and said so."""
        view = self._bundle_view
        rows = torch.tensor(list(validation.rows or []), dtype=torch.long)
        view_ids = view.view_ids("phenotype", rows) if rows.numel() else rows
        dropped = int((view_ids < 0).sum())
        if dropped:
            warnings = warnings + [
                f"{dropped} phenotype(s) are outside the model's graph view and were not "
                "scored"]
        return torch.unique(view_ids[view_ids >= 0]), warnings

    def _rank_by_gene_task(self, patient_input: PatientPhenotypes,
                           validation: "ValidationResult", warnings: List[str],
                           source_ids: List[NodeID]) -> ServedRanking:
        """`gene_task/v1`: the request's candidate genes (PLAN_TEST_RESULTS
        §8.3), ranked by `η·EMBSIM + (1 − η)·SPLSIM` with exact hops on the
        whole view (README §6), ties by view id. A gene that maps to no gene of
        the view is left out and counted ("N mapped of M submitted"); one with
        an undefined distance is not scorable and said so."""
        from src.models.tasks.gene import (
            distance_table,
            gene_embsim,
            mean_distances,
            neighbour_inputs,
            splsim,
        )

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
            warnings = warnings + [f"{genes.numel()} of {len(submitted)} candidate genes "
                                   "submitted map to a gene of the model's graph view"]
        phenotypes, warnings = self._bundle_phenotypes(validation, warnings)
        if phenotypes.numel() == 0 or genes.numel() == 0:
            return ServedRanking(
                outcome="no_candidates",
                warnings=warnings + ["Nothing to rank: no phenotype or no candidate gene is in "
                                     "the model's graph view"],
                patient_input=patient_input, phenotype_input=validation.phenotype_input,
                source_ids=source_ids)
        spec = self.model.task_head.spec.gene
        table = distance_table(view, phenotypes, genes)
        rows = table.row_of[phenotypes].unsqueeze(0)
        cols = table.col_of[genes].unsqueeze(0)
        mask = torch.ones_like(rows, dtype=torch.bool)
        mean, defined = mean_distances(table.table, rows, mask, cols)
        z = self._bundle_z
        if self._bundle_neighbours is not None:
            near, weight, degree = neighbour_inputs(self._bundle_neighbours, genes.unsqueeze(0))
        else:
            near = torch.zeros(1, genes.numel(), spec.augmentation_k, dtype=torch.long)
            weight = torch.zeros(1, genes.numel(), spec.augmentation_k)
            degree = torch.zeros(1, genes.numel())
        with torch.no_grad():
            embsim = gene_embsim(self.model, z, phenotypes.unsqueeze(0).to(z.device),
                                 mask.to(z.device), genes.unsqueeze(0).to(z.device),
                                 near.to(z.device), weight.to(z.device),
                                 degree.to(z.device))[0].cpu()
        sp = splsim(mean[0], spec.sp_lower, spec.sp_upper)
        scores = spec.eta * embsim + (1.0 - spec.eta) * sp
        scorable = defined[0]
        if not bool(scorable.all()):
            warnings = warnings + [f"{int((~scorable).sum())} candidate gene(s) have an "
                                   "unreachable phenotype and are not scorable"]
        if not bool(torch.isfinite(scores[scorable]).all()):
            raise ResultRefused("NON_FINITE_SCORE",
                                "the gene head produced a non-finite score; the result was "
                                "refused, not ranked")
        order = sorted((i for i in range(genes.numel()) if bool(scorable[i])),
                       key=lambda i: (-float(scores[i]), int(genes[i])))
        reverse = {index: key for key, index in
                   (self._node_id_to_idx or {}).get(NodeType.GENE.value, {}).items()}
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
            discovery_ordinals=order, source_ids=source_ids, all_paths={})

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
            discovery_ordinals=order, source_ids=source_ids, all_paths={})

    def _patient_candidate(self, rank: int, case_id: str, similarity: float,
                           row: Any) -> PatientCandidate:
        """A reference patient with what its cohort recorded: HPO terms, the
        diagnosis and the causal genes, in the graph's ids and names."""
        mapping = self._node_id_to_idx or {}

        def ids(node_type: str, locals_: Any) -> List[str]:
            reverse = {i: key for key, i in mapping.get(node_type, {}).items()}
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
        return PatientCandidate(rank=rank, case_id=case_id, similarity=similarity,
                                phenotypes=phenotypes, disease_id=disease_id,
                                disease_name=node.name if node else None,
                                causal_genes=tuple(genes))

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
        if ranking.outcome == "refused":
            return InferenceResult(
                patient_id=patient_input.patient_id,
                timestamp=datetime.now(),
                candidates=[],
                served_identity=self.served_identity(),
                warnings=ranking.warnings,
                inference_time_ms=(time.time() - start_time) * 1000,
                phenotype_input=ranking.phenotype_input,
                candidates_reached=self._candidates_reached(0),
                reference_cohort=reference,
            )
        patient_input = ranking.patient_input
        if ranking.outcome == "no_candidates":
            return self._create_empty_result(
                patient_input.patient_id,
                start_time,
                ranking.warnings,
                ranking.phenotype_input,
                reference_cohort=reference,
            )
        if self._bundle_view is not None and self.model.task != "disease":
            # Gene and reference-patient rows carry their own values; path
            # evidence explains disease rows (README §1) and is not built here.
            return InferenceResult(
                patient_id=patient_input.patient_id, timestamp=datetime.now(),
                candidates=list(ranking.candidates[:top_k]),
                served_identity=self.served_identity(),
                inference_time_ms=(time.time() - start_time) * 1000,
                warnings=ranking.warnings, phenotype_input=ranking.phenotype_input,
                reference_cohort=reference)

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
        all_paths = ranking.all_paths
        if self._bundle_view is not None and candidates:
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
            candidates_reached=self._candidates_reached(len(ranking.candidates)),
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
        5. **With a model served, each node is mapped to its row** in the
           phenotype embedding table (Spec 1 §2.2: one mapping step, exact term,
           graph node, tensor row). A known node with no row, or a row outside
           the table, is a model–graph mismatch: the result is refused
           (`ResultRefused`, `MODEL_TERM_UNAVAILABLE`), never scored without it.

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
            if self.kg.has_node(node_id):
                known_nodes.append(str(node_id))
                known_positions.append(position)
            else:
                unknown.append(pheno_id)
                warnings.append(f"Unknown phenotype: {pheno_id}")

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

    def _phenotype_rows(self, node_ids: List[NodeID]) -> Optional[List[int]]:
        """Each node's row in the phenotype embedding table, in order; None when
        no model is served (the path-reasoning kind reads no tensor).

        Raises `ResultRefused` (`MODEL_TERM_UNAVAILABLE`, naming the node) for a
        node with no row or a row outside the table: a model–graph mismatch is
        refused, never skipped and never clamped onto a neighbouring row."""
        if not self._gnn_ready:
            return None
        if self._bundle_view is not None:
            # The graph's phenotype rows; the view maps them (`_rank_by_disease_task`).
            index = (self._node_id_to_idx or {}).get(NodeType.PHENOTYPE.value, {})
            missing = [str(node) for node in node_ids if index.get(str(node)) is None]
            if missing:
                raise ResultRefused(
                    "MODEL_TERM_UNAVAILABLE",
                    f"phenotype {missing[0]} is in the graph but has no row in its node "
                    "mapping")
            return [int(index[str(node)]) for node in node_ids]
        table = (self._node_embeddings or {}).get(NodeType.PHENOTYPE.value)
        if table is None:
            raise ResultRefused(
                "MODEL_TERM_UNAVAILABLE",
                "the served model has no phenotype embedding table, so no patient "
                "representation can be computed")
        index = (self._node_id_to_idx or {}).get(NodeType.PHENOTYPE.value, {})
        rows: List[int] = []
        for node in node_ids:
            row = index.get(str(node))
            if row is None:
                raise ResultRefused(
                    "MODEL_TERM_UNAVAILABLE",
                    f"phenotype {node} is in the graph but has no row in the model's "
                    "node mapping; the model and the graph disagree")
            if isinstance(row, bool) or not isinstance(row, int) or not 0 <= row < table.size(0):
                raise ResultRefused(
                    "MODEL_TERM_UNAVAILABLE",
                    f"phenotype {node} maps to row {row!r}, outside the phenotype "
                    f"embedding table's {table.size(0)} rows")
            rows.append(row)
        return rows

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

    def _score_and_rank_candidates(
        self,
        all_paths: Dict[str, List[ReasoningPath]],
        source_ids: List[NodeID],
        patient_input: PatientPhenotypes,
        include_ortholog_evidence: bool,
        rows: Optional[List[int]] = None,
        unmapped_ids: Sequence[str] = (),
    ) -> Tuple[List[Candidate], List[int]]:
        """
        Score every disease candidate and rank them all, as the served kind does.

        - `served_mixture/v1` (a model served): `model_term` per candidate from
          the pooled patient (`rows`, validation's), one `sp_analysis_records`
          call over every reached candidate for `sp_term`, and `served_score`
          mixed from the two (in `gnn_only` the model term alone). A non-numeric
          SP status is still mixed as 0, and says so (`mixed_as`), so the legacy
          ranking does not change.
        - `path_reasoning/v1` (no model): `path_score`, the paths' aggregate.

        A score that is not finite refuses the result (`ResultRefused`,
        `NON_FINITE_SCORE`), as does a model term that cannot be computed.

        Args:
            all_paths: Paths grouped by disease
            source_ids: Patient phenotype NodeIDs, as used
            patient_input: Patient phenotype data, as used
            include_ortholog_evidence: Include ortholog evidence
            rows: The phenotypes' embedding rows (None without a model)
            unmapped_ids: The request's ids no node was found for

        Returns:
            Every candidate sorted by its kind's score, ranked 1..N, and beside
            each its discovery ordinal: its position in the list built in
            `all_paths` order, before the sort (`ServedRanking`). Nothing is cut
            here; `run` cuts at `top_k`.
        """
        from src.inference.sp_index import sp_analysis_records

        mixture = result_kind_for(self._scoring_mode()[0]) == SERVED_MIXTURE
        entries: List[Tuple[NodeID, str, List[ReasoningPath]]] = []
        for paths in all_paths.values():
            if not paths:
                continue
            disease_id = paths[0].target
            disease_node = self.kg.get_node(disease_id)
            entries.append(
                (disease_id, disease_node.name if disease_node else str(disease_id), paths))

        model_terms: List[float] = []
        records: List[SPAnalysisRecord] = []
        if mixture:
            # The patient is pooled once: the same operation on the same rows
            # for every candidate. Each candidate's cosine stays a batch of one,
            # since a taller float32 matrix product can move a value's last bit.
            patient = self._patient_embedding(rows)
            model_terms = [self._calculate_gnn_score(patient, disease_id)
                           for disease_id, _name, _paths in entries]
            records = sp_analysis_records(
                self._sp_lookup if self._sp_ready else None,
                self._node_id_to_idx,
                [str(node) for node in source_ids],
                [str(disease_id) for disease_id, _name, _paths in entries],
                1,  # disease
                unmapped_ids=unmapped_ids,
            )
            if self._sp_ready:
                records = [record if record.numeric else replace(record, mixed_as=0.0)
                           for record in records]
                scores = [self._served_score(term, record)
                          for term, record in zip(model_terms, records)]
            else:
                # Nothing is mixed without a table: the model term ranks alone.
                scores = list(model_terms)
        else:
            scores = [self._calculate_reasoning_score(paths)
                      for _id, _name, paths in entries]
        for (disease_id, _name, _paths), score in zip(entries, scores):
            if not math.isfinite(score):
                raise ResultRefused(
                    "NON_FINITE_SCORE",
                    f"disease {disease_id} scored {score!r}; the result was refused, "
                    "not ranked")

        # Sorted by the kind's score. **Stable**, so equal scores keep discovery
        # order; the order is sorted by position so each candidate's discovery
        # ordinal travels with it, which the sorted list alone cannot give back.
        order = sorted(range(len(entries)), key=lambda i: scores[i], reverse=True)
        ranked: List[Candidate] = []
        for rank, i in enumerate(order, start=1):
            disease_id, disease_name, paths = entries[i]
            common = dict(
                rank=rank,
                disease_id=disease_id,
                disease_name=disease_name,
                supporting_genes=self._extract_supporting_genes(paths),
                reasoning_paths=[list(p.nodes) for p in paths[:5]],
                evidence_sources=self._determine_evidence_sources(paths),
            )
            if mixture:
                ranked.append(ServedMixtureCandidate(
                    served_score=scores[i], model_term=model_terms[i],
                    sp_term=records[i], **common))
            else:
                ranked.append(PathReasoningCandidate(path_score=scores[i], **common))
        return ranked, order

    def _calculate_reasoning_score(self, paths: List[ReasoningPath]) -> float:
        """Calculate aggregate reasoning score from paths."""
        if not paths:
            return 0.0

        # Use configured aggregation method
        method = self.config.aggregation_method
        scores = [p.score for p in paths]

        if method == "max":
            return max(scores)
        elif method == "mean":
            return sum(scores) / len(scores)
        elif method == "sum":
            return min(sum(scores), 1.0)  # Cap at 1.0
        elif method == "weighted_sum":
            # Weighted by path count with diminishing returns
            weighted = sum(s * (0.8 ** i) for i, s in enumerate(scores))
            return min(weighted, 1.0)
        else:
            return max(scores)

    def _served_score(self, model_term: float, record: SPAnalysisRecord) -> float:
        """`served_mixture/v1`'s value with a table: η × model term + (1 − η) × SP,
        where a non-numeric SP status enters as its `mixed_as`, 0.0 (legacy).

        float64 is required, not incidental. The scores arriving here are Python
        doubles; constructing float32 tensors would round them to ~7 significant
        digits, and two candidates whose true scores differ below that threshold
        would collapse to an exact tie. A tie is then resolved by sort stability
        — that is, by input order — which can move a candidate across the top-k
        boundary a clinician sees. The extraction is only behaviour-preserving in
        double precision.
        """
        from src.inference.scoring import mix_embedding_and_sp_scores

        sp_value = record.score if record.numeric else record.mixed_as
        return float(
            mix_embedding_and_sp_scores(
                torch.tensor([model_term], dtype=torch.float64),
                torch.tensor([sp_value], dtype=torch.float64),
                self.config.eta,
            )[0]
        )

    def _patient_embedding(self, rows: Optional[List[int]]) -> "Tensor":
        """The unweighted mean of the patient's phenotype rows, in the order used.

        Raises `ResultRefused` (`MODEL_TERM_UNAVAILABLE`) when there are no
        embeddings, no phenotype table or no rows; validation has refused a
        row outside the table, so the pool's clamp cannot move one."""
        from src.inference.scoring import pool_patient_embeddings

        if not self._gnn_ready or self._node_embeddings is None:
            raise ResultRefused(
                "MODEL_TERM_UNAVAILABLE",
                "the pipeline holds no node embeddings, so no model term can be computed")
        table = self._node_embeddings.get(NodeType.PHENOTYPE.value)
        if table is None:
            raise ResultRefused(
                "MODEL_TERM_UNAVAILABLE",
                "the served model has no phenotype embedding table, so no patient "
                "representation can be computed")
        if not rows:
            raise ResultRefused(
                "MODEL_TERM_UNAVAILABLE",
                "no phenotype row to pool: the phenotypes as used have no rows in "
                "the model's node mapping")
        return pool_patient_embeddings(table, rows)

    def _calculate_gnn_score(self, patient_embedding: "Tensor", disease_id: NodeID) -> float:
        """
        The model term of one candidate: (cos + 1) / 2 of the pooled patient and
        the disease's embedding, from the node embeddings the GNN forward pass
        cached. This matches the scoring approach used during training (see
        Trainer._compute_model_outputs).

        **It fails closed** (Spec 1 §2.1). Each state that used to return 0.0
        refuses the result instead (`ResultRefused`, `MODEL_TERM_UNAVAILABLE`,
        naming the node): no embeddings, no disease table, a disease with no
        row, and a row outside the table, which used to be clamped onto the
        last one. A value that is not finite refuses it too (`NON_FINITE_SCORE`).
        """
        from src.inference.scoring import (
            cosine_scores,
            normalise_cosine_to_unit_interval,
        )

        if not self._gnn_ready or self._node_embeddings is None:
            raise ResultRefused(
                "MODEL_TERM_UNAVAILABLE",
                "the pipeline holds no node embeddings, so no model term can be computed")
        disease_emb = self._node_embeddings.get(NodeType.DISEASE.value)
        if disease_emb is None:
            raise ResultRefused(
                "MODEL_TERM_UNAVAILABLE",
                "the served model has no disease embedding table, so no disease can "
                "be scored")
        disease_idx = (self._node_id_to_idx or {}).get(
            NodeType.DISEASE.value, {}).get(str(disease_id))
        if disease_idx is None:
            raise ResultRefused(
                "MODEL_TERM_UNAVAILABLE",
                f"disease {disease_id} was reached in the graph but has no row in the "
                "model's node mapping; the model and the graph disagree")
        if isinstance(disease_idx, bool) or not isinstance(disease_idx, int) or \
                not 0 <= disease_idx < disease_emb.size(0):
            raise ResultRefused(
                "MODEL_TERM_UNAVAILABLE",
                f"disease {disease_id} maps to row {disease_idx!r}, outside the disease "
                f"embedding table's {disease_emb.size(0)} rows")

        # Cosine similarity, then [-1,1] -> [0,1]. One candidate is a batch of one.
        cosine = cosine_scores(patient_embedding, disease_emb[disease_idx].unsqueeze(0))
        value = float(normalise_cosine_to_unit_interval(cosine)[0])
        if not math.isfinite(value):
            raise ResultRefused(
                "NON_FINITE_SCORE",
                f"disease {disease_id}'s model term is {value!r}; the result was "
                "refused, not ranked")
        return value

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
          - Mode B (analogy-based): when no direct path is available, find
            the K nearest known candidates in GNN embedding space whose
            paths to the patient phenotypes DO exist, and surface them
          - INSUFFICIENT: if neither mode succeeds

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

            # Step C: structured evidence package with its evidence label.
            # Mode B requires GNN embeddings and the index map; both are
            # passed when available so the panel can attempt analogy search.
            if source_ids is not None:
                package: EvidencePackage = self.evidence_panel.build_evidence(
                    candidate=candidate,
                    patient_input=patient_input,
                    source_ids=source_ids,
                    existing_paths=paths if paths else None,
                    node_embeddings=self._node_embeddings,
                    node_id_to_idx=self._node_id_to_idx,
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
    ) -> InferenceResult:
        """A result with no candidates (discovery reached no disease), with
        its warnings."""
        return InferenceResult(
            patient_id=patient_id,
            timestamp=datetime.now(),
            candidates=[],
            served_identity=self.served_identity(),
            warnings=warnings,
            inference_time_ms=(time.time() - start_time) * 1000,
            phenotype_input=phenotype_input,
            candidates_reached=self._candidates_reached(0),
            reference_cohort=reference_cohort,
        )

    def _candidates_reached(self, reached: int) -> Optional[int]:
        """`candidates_reached` as the served kind carries it: the count for
        `served_mixture/v1`, None for the kinds that do not carry it."""
        return reached if result_kind_for(self._scoring_mode()[0]) == SERVED_MIXTURE else None

    def _result_label(self) -> str:
        """The served kind's label with this pipeline's own η (`result_label`)."""
        scoring_mode, eta = self._scoring_mode()
        return result_label(result_kind_for(scoring_mode), scoring_mode, eta)

    @property
    def embedding_regime(self) -> Optional[Any]:
        """The autocast regime (`AutocastRegime`) the cached embeddings were
        computed under, observed at that forward; None before any embeddings.

        A measurement that scores these embeddings records it, and Mode C's own
        check (`assert_manifest_describes_regime`) compares it with the manifest
        (Mode D, `docs/working/scorer-measurement/PLAN_B05.md` §3)."""
        return self._embedding_regime

    @property
    def node_embeddings(self) -> Optional[Dict[str, Any]]:
        """The cached embeddings by node type, as the forward produced them and
        moved them to the CPU; None without a model. Read-only: scoring reads
        the same tensors, so a caller must not change them."""
        return self._node_embeddings

    def bound_inputs(self) -> BoundInputs:
        """What this pipeline read and bound from its verified workspace
        (`BoundInputs`), for a measurement of the served pipeline that must not
        read those files again (Mode D).

        Raises `ValueError` for a pipeline whose graph was not verified from a
        workspace or that serves no model: nothing binds what it would score.
        """
        if self._bound_digests is None or self._dataset_record is None:
            raise ValueError(
                "this pipeline was not built over a verified workspace, so nothing "
                "binds its graph, dataset record or model to files a measurement "
                "can name"
            )
        if self._checkpoint_identity is None:
            if self.model is None or not self._gnn_ready:
                raise ValueError(
                    "this pipeline serves no model (path_reasoning_fallback), so there "
                    "is no checkpoint to name and no embeddings to measure"
                )
            # A model supplied in memory over a verified workspace serves a GNN,
            # but no checkpoint read names the weights a measurement would score.
            raise ValueError(
                "this pipeline's model was supplied in memory, not read from a "
                "checkpoint, so nothing binds the weights a measurement would score"
            )
        return BoundInputs(
            graph=dict(self._bound_digests),
            dataset_record=self._dataset_record,
            checkpoint=self._checkpoint_identity,
            checkpoint_training_inputs=copy.deepcopy(self._checkpoint_training_inputs),
            manifest_sha256=self._manifest_sha256,
            graph_data=self._graph_data,
        )

    def _scoring_mode(self) -> Tuple[str, Optional[float]]:
        """The scoring mode in effect and the mixing weight it reports (None
        for the disease task, which mixes nothing)."""
        if self._gnn_ready and self._bundle_view is not None:
            return {"disease": "disease_task", "gene": "gene_task",
                    "patients_like_me": "patients_like_me"}[self.model.task], None
        if self._gnn_ready and self._sp_ready:
            return "gnn_plus_shortest_path", self.config.eta
        if self._gnn_ready:
            return "gnn_only", 1.0
        return "path_reasoning_fallback", 0.0

    def served_identity(self) -> ServedIdentity:
        """What this pipeline serves, part by part (contract M3c, R8).

        **Built from the pipeline's own state, from the reads it took**: the
        checkpoint's digest from the read the model was built from, the graph's
        and the manifest's from the reads the workspace was verified with, the
        shortest-path table's `build_id` from the tensor accepted, the dataset
        record's origin from the record verified. Nothing is hashed again here.
        Every result the pipeline returns carries this, built when the result
        is produced, and `get_pipeline_config()` reports it from this same
        method. A part with nothing to identify says so in its `state`.
        """
        scoring_mode, eta = self._scoring_mode()

        if self.model is None or not self._gnn_ready:
            checkpoint = ServedCheckpoint(state="absent")
        elif self._checkpoint_identity is not None:
            checkpoint = ServedCheckpoint(
                state="verified", sha256=self._checkpoint_identity.sha256
            )
        else:
            checkpoint = ServedCheckpoint(state="supplied_in_memory")

        if self._bound_digests is not None:
            graph = ServedGraph(
                state="verified",
                kg_sha256=self._graph_kg_digest,
                split_manifest_sha256=self._manifest_sha256,
            )
        elif self._graph_identity is not None:
            graph = ServedGraph(state="unverified", kg_sha256=self._graph_identity.sha256)
        else:
            graph = ServedGraph(state="in_memory")

        if self._sp_ready:
            shortest_paths = ServedShortestPaths(
                state="verified", build_id=self._sp_build_id, max_hops=self._sp_max_hops
            )
        else:
            shortest_paths = ServedShortestPaths(state="absent")

        record = self._dataset_record
        dataset_record = (
            ServedDatasetRecord(
                state="verified", origin=record.origin, sha256=record.identity.sha256
            )
            if record is not None
            else ServedDatasetRecord(state="absent")
        )

        served_model = checkpoint.state != "absent"
        settings = ServedSettings(
            device=self._device_in_effect if served_model else None,
            device_requested=self._device_requested if served_model else None,
            eta=float(eta) if served_model and eta is not None else None,
            max_path_length=self.config.max_path_length,
            path_length_penalty=self.config.path_length_penalty,
            aggregation_method=self.config.aggregation_method,
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

    def served_checkpoint_read(self) -> Optional["ReadIdentity"]:
        """The checkpoint read the served model was built from, or ``None``.

        For a reader that needs the read's path as well as its digest: the
        recorded results view finds the ledger beside the checkpoint and filters
        it by digest (`src/api/routes/evaluations.py`). Both come from this one
        identity, so the directory looked in and the digest looked up cannot
        belong to two different loads, which a path taken from application state
        beside a digest taken from here could during a reload. The path is the
        checkpoint's resolved path, as it was read (`_load_model_from_checkpoint`),
        so its directory is the one the loaded bytes were in, whatever a link
        named at load time points to now.

        **Returned exactly when `served_identity()` reports the checkpoint
        verified**, by asking it rather than restating its condition, so the two
        cannot drift apart: a model supplied in memory, or no model at all, has
        no read to return. Nothing is hashed or read here.
        """
        if self.served_identity().checkpoint.state != "verified":
            return None
        return self._checkpoint_identity

    def get_pipeline_config(self) -> Dict[str, Any]:
        """Get pipeline configuration as dictionary."""
        # Effective scoring mode reflects whether SP signal is contributing
        scoring_mode, effective_eta = self._scoring_mode()

        return {
            # What is served, part by part (contract M3c): the same value every
            # result carries, from the same builder.
            "served_identity": self.served_identity().to_dict(),
            "max_path_length": self.config.max_path_length,
            "path_length_penalty": self.config.path_length_penalty,
            "aggregation_method": self.config.aggregation_method,
            "scoring_mode": scoring_mode,
            "eta_configured": self.config.eta,
            "eta_effective": effective_eta,
            "sp_max_hops": self._sp_max_hops if self._sp_ready else None,
            "sp_hop_bound_source": self._sp_hop_bound_source if self._sp_ready else None,
            # Reported so a surface can show that the table was checked; an
            # unrecorded binding is refused at load (contract M2.5).
            "sp_kg_binding": self._sp_kg_binding if self._sp_ready else None,
            # The workspace's dataset record (contract M3b): its binding state
            # and its origin apart, so a synthetic workspace is never shown as
            # plainly verified. None when no workspace was verified.
            "dataset_record": (
                self._dataset_record.summary() if self._dataset_record is not None else None
            ),
            "path_reasoner_role": "explanation_only" if self._gnn_ready else "scoring_and_explanation",
            "include_explanations": self.config.include_explanations,
            "include_ortholog_evidence": self.config.include_ortholog_evidence,
            "include_literature_evidence": self.config.include_literature_evidence,
            "has_model": self.model is not None,
            "gnn_ready": self._gnn_ready,
            "sp_ready": self._sp_ready,
            "fingerprint_warnings": getattr(self, "_fingerprint_warnings", []),
            "checkpoint_meta": getattr(self, "_checkpoint_meta", {}),
            "kg_nodes": len(self.kg._nodes) if self.kg else 0,
            "kg_edges": len(self.kg._edges) if self.kg else 0,
        }


# ==============================================================================
# Factory Function
# ==============================================================================
def build_served_pipeline(
    kg_path: Union[str, Path],
    checkpoint_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    device: Optional[str] = None,
    expected_checkpoint_sha256: Optional[str] = None,
) -> DiagnosisPipeline:
    """Build a pipeline as the service builds it: settle the device, read the
    graph once, and create the pipeline over it with the default configuration.

    **The service's own build, in one place** (Batch 3). `src/api/main.py`'s
    `build_pipeline` resolves its paths from the request or the environment,
    records what was requested and then calls this; a measurement of the served
    ranking (Mode D) calls it too, so it measures a pipeline built from the same
    files by the same code, never a second construction. It sits here because
    `src.evaluation` may import `src.inference` and not `src.api`.

    The device is settled before the graph is read when a checkpoint is named
    (contract M3c): `auto` without CUDA refuses before a multi-gigabyte
    `kg.json` is parsed for a model that cannot be served here. The pipeline
    applies the same rule itself; the request is passed on as given, so the
    served identity records what was asked for beside what took effect.
    """
    from src.kg.graph import KnowledgeGraph

    if checkpoint_path:
        resolve_serving_device(device)
    kg_file = Path(kg_path)
    logger.info(f"Loading knowledge graph from {kg_file}...")
    # **One read, kept whole** (contract M2.1, S6). The graph and the identity of
    # the bytes it was parsed from travel together into the pipeline, which
    # checks that identity against the workspace manifest when it builds a GNN
    # over `data_dir`; a pipeline with no checkpoint checks nothing (BACKLOG
    # 19.20). Nothing hashes the file again by path.
    graph = KnowledgeGraph.read_json(kg_file)
    kg = graph.kg
    logger.info(f"KG loaded: {kg.total_nodes} nodes, {kg.total_edges} edges")
    # **The graph read is passed whole, and the composition is checked there.**
    # The graph and the workspace can be named independently, so a deployment
    # can point them at two different workspaces -- each internally consistent,
    # and together producing embeddings from one graph's tensors read through
    # the other graph's node identifiers. The pipeline refuses that composition.
    # The hop bound comes from the shortest-path table's sidecar alone (contract
    # M2.5).
    return create_diagnosis_pipeline(
        graph=graph,
        checkpoint_path=checkpoint_path,
        data_dir=data_dir,
        device=device,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )


def create_diagnosis_pipeline(
    kg: Optional["KnowledgeGraph"] = None,
    config: Optional[PipelineConfig] = None,
    model: Optional[Any] = None,
    checkpoint_path: Optional[str] = None,
    graph_data: Optional[Dict[str, Any]] = None,
    data_dir: Optional[str] = None,
    graph: Optional["GraphRead"] = None,
    device: Optional[str] = None,
    expected_checkpoint_sha256: Optional[str] = None,
) -> DiagnosisPipeline:
    """
    Factory function to create a diagnosis pipeline.

    Args:
        kg: An in-memory knowledge graph (give this or `graph`)
        config: Pipeline configuration
        model: Optional pre-loaded GNN model
        checkpoint_path: Path to trained model checkpoint
        graph_data: Pre-loaded graph data dict
        data_dir: Path to processed data directory
        graph: The `GraphRead` `KnowledgeGraph.read_json` returned, passed
            whole, so the composition can be checked
        device: Device for GNN inference
        expected_checkpoint_sha256: The SHA-256 the checkpoint read must have
            (`DiagnosisPipeline`)

    Returns:
        Configured DiagnosisPipeline instance
    """
    return DiagnosisPipeline(
        kg=kg,
        config=config,
        model=model,
        checkpoint_path=checkpoint_path,
        graph_data=graph_data,
        data_dir=data_dir,
        graph=graph,
        device=device,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
