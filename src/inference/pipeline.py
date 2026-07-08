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
#   - Scoring: GNN-primary when available, path reasoning as fallback
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
#   # With GNN scoring from checkpoint
#   pipeline = DiagnosisPipeline(
#       kg=knowledge_graph,
#       checkpoint_path="checkpoints/last.pt",
#       data_dir="data/workspaces/demo",
#   )
#   result = pipeline.run(patient_phenotypes, top_k=10)
# ==============================================================================
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

from src.core.types import (
    DataSource,
    DiagnosisCandidate,
    EvidenceSource,
    InferenceResult,
    NodeID,
    NodeType,
    PatientPhenotypes,
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
    create_path_reasoner,
    create_explanation_generator,
    create_evidence_panel,
)

# Optional torch dependency for GNN inference
try:
    import torch
    import torch.nn.functional as F
    from torch import Tensor
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if TYPE_CHECKING:
    from src.kg import KnowledgeGraph
    from src.models.gnn.shepherd_gnn import ShepherdGNN, ShepherdGNNConfig

logger = logging.getLogger(__name__)


# ==============================================================================
# Type Aliases for Callbacks
# ==============================================================================
# Callback for custom scoring: (candidate, paths, kg) -> modified_score
ScoringCallback = Callable[
    ["DiagnosisCandidate", List[ReasoningPath], "KnowledgeGraph"],
    float
]
# Callback for post-processing candidates
PostProcessCallback = Callable[
    [List["DiagnosisCandidate"], PatientPhenotypes],
    List["DiagnosisCandidate"]
]


# ==============================================================================
# Checkpoint architecture inference
# ==============================================================================
def _infer_conv_type_from_keys(state_keys) -> Optional[str]:
    """Infer the GNN conv type from a checkpoint's parameter names.

    Older checkpoints don't record ``conv_type`` in their config (the trainer
    serialized only training hyperparameters), so we recover it from the weight
    names — otherwise an HGT/SAGE checkpoint is silently rebuilt as the GAT
    default and the state_dict load fails.

    Distinguishing markers (params are under ``gnn_layers.<n>.conv...``):
      - HGT (``HGTConv``): ``kqv_lin`` / ``k_rel`` / ``v_rel`` / ``p_rel`` / ``skip``
      - GAT (``GATConv`` in ``HeteroConv``): ``att_src`` / ``att_dst``
      - SAGE (``SAGEConv`` in ``HeteroConv``): ``lin_l`` / ``lin_r`` (no attention)
    """
    keys = state_keys if isinstance(state_keys, (set, frozenset)) else set(state_keys)
    if any(
        (".conv.kqv_lin." in k or ".conv.k_rel" in k or ".conv.v_rel" in k
         or ".conv.p_rel." in k or ".conv.skip." in k)
        for k in keys
    ):
        return "hgt"
    if any(k.endswith(".att_src") or k.endswith(".att_dst") for k in keys):
        return "gat"
    if any((".conv.convs." in k) and (".lin_l." in k or ".lin_r." in k) for k in keys):
        return "sage"
    return None


def _infer_num_layers_from_keys(state_keys) -> Optional[int]:
    """Infer the number of GNN layers from the highest ``gnn_layers.<n>`` index."""
    import re

    indices = set()
    for k in state_keys:
        m = re.match(r"gnn_layers\.(\d+)\.", k)
        if m:
            indices.add(int(m.group(1)))
    return (max(indices) + 1) if indices else None


def _resolve_arch_params(
    ckpt_config: dict,
    state_keys,
    *,
    valid_fields,
    supported_conv,
    has_pos_encoder: bool = False,
    has_ortholog_gate: bool = False,
) -> dict:
    """Resolve model-architecture kwargs from a checkpoint, by precedence.

    This is the schema resolution the diagnosis pipeline uses to reconstruct a
    model. It is deliberately free of torch/model imports so it can be unit
    tested in isolation: the caller passes ``valid_fields`` (the current
    ``ShepherdGNNConfig`` field names) and ``supported_conv`` (the conv types the
    factory can build).

    Precedence (highest first):
      1. ``ckpt_config["model_config"]`` — the full, self-describing sub-dict
         written by current trainers.
      2. Legacy flat arch fields at the top level of ``ckpt_config`` (written by
         the interim fix before ``model_config`` existed).
      3. Inference from the parameter names (``conv_type`` / ``num_layers``) for
         checkpoints that carry no architecture metadata at all.
      4. ``ShepherdGNNConfig`` defaults (applied by the caller when a key is
         simply absent from the returned dict).

    Unknown keys are filtered against ``valid_fields`` to tolerate version drift
    (and are logged). ``conv_type`` handling depends on where it came from, which
    is tracked explicitly:

      - ``model_config`` (tier 1) is the trainer's authoritative self-description
        and is TRUSTED over the weight-key heuristic (only warned on conflict).
        The heuristic is a legacy fallback that is not future-proof — a new
        architecture may reuse PyG key patterns (e.g. ``att_src`` / ``lin_l``),
        so it must not override an explicit model_config value.
      - ``legacy_flat`` (tier 2) predates ``model_config``; here the weights are
        treated as structural ground truth and override a conflicting value.
      - ``inferred`` (tier 3) is derived from the weights, so it cannot conflict.

    In all cases an explicit-but-unsupported ``conv_type`` raises rather than
    silently degrading to GAT; only a truly absent/undetectable one defaults.
    """
    params: dict = {}
    conv_source = None  # "model_config" | "legacy_flat" | "inferred"

    # Tier 1: full self-describing model_config sub-dict.
    model_config = ckpt_config.get("model_config")
    if isinstance(model_config, dict):
        ignored = []
        for k, v in model_config.items():
            if k in valid_fields:
                params[k] = v
            else:
                ignored.append(k)
        if ignored:
            logger.warning(
                "Ignoring unknown model_config field(s) not in the current "
                "ShepherdGNNConfig schema: %s",
                sorted(ignored),
            )
        if "conv_type" in params:
            conv_source = "model_config"

    # Tier 2: legacy flat arch fields (fill only what tier 1 didn't provide).
    for key in (
        "conv_type",
        "hidden_dim",
        "num_layers",
        "num_heads",
        "use_positional_encoding",
        "use_ortholog_gate",
    ):
        if key not in params and key in valid_fields and ckpt_config.get(key) is not None:
            params[key] = ckpt_config[key]
            if key == "conv_type":
                conv_source = "legacy_flat"

    # Tier 3: infer structural fields from the parameter names when still absent.
    detected_conv = _infer_conv_type_from_keys(state_keys)
    if "conv_type" not in params and detected_conv:
        params["conv_type"] = detected_conv
        conv_source = "inferred"
        logger.info(
            "conv_type not in checkpoint config; detected %r from weights.",
            detected_conv,
        )
    if "num_layers" not in params:
        detected_layers = _infer_num_layers_from_keys(state_keys)
        if detected_layers:
            params["num_layers"] = detected_layers
    if "use_positional_encoding" in valid_fields:
        params.setdefault("use_positional_encoding", has_pos_encoder)
    if "use_ortholog_gate" in valid_fields:
        params.setdefault("use_ortholog_gate", has_ortholog_gate)

    # Conflict handling depends on the SOURCE of conv_type:
    #   - legacy_flat: weights are ground truth -> override on conflict.
    #   - model_config: authoritative -> keep it, only warn (the key heuristic is
    #     a legacy fallback that may misclassify future architectures).
    conv_type = params.get("conv_type")
    if conv_type is not None and detected_conv and conv_type != detected_conv:
        if conv_source == "legacy_flat":
            logger.warning(
                "Legacy flat conv_type=%r disagrees with parameter names "
                "(detected %r); trusting the weights.",
                conv_type, detected_conv,
            )
            params["conv_type"] = detected_conv
        else:  # model_config
            logger.warning(
                "model_config conv_type=%r disagrees with the weight-key "
                "heuristic (detected %r); trusting model_config, as the heuristic "
                "is a legacy fallback that may misclassify newer architectures.",
                conv_type, detected_conv,
            )

    # Validate: an explicit-but-unsupported conv_type must fail loudly. Only a
    # truly absent/undetectable conv_type falls back to the GAT default.
    conv_type = params.get("conv_type")
    if conv_type is None:
        params["conv_type"] = "gat"
    elif conv_type not in supported_conv:
        raise ValueError(
            f"Checkpoint specifies unsupported conv_type={conv_type!r}; "
            f"supported types are {tuple(supported_conv)}."
        )

    # Inference-time override (never carry training dropout into eval).
    if "dropout" in valid_fields:
        params["dropout"] = 0.0

    return params


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
    #                         The paper does not fix a value; 0.7 is a
    #                         reasonable default that favors GNN's learned
    #                         signal but lets SP catch sparse-input cases.
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

    # Validation
    validate_phenotypes: bool = True
    min_phenotypes: int = 1
    max_phenotypes: int = 100

    # Performance
    use_direct_path_optimization: bool = True

    # Vector index for ANN-based candidate pre-filtering
    # When set, the pipeline uses the vector index to discover additional
    # disease candidates that may not be reachable via BFS paths, enabling
    # discovery of latent associations captured by GNN embeddings.
    vector_index_path: Optional[str] = None
    ann_top_k: int = 50  # Number of ANN candidates to retrieve
    ann_score_threshold: float = 0.3  # Min normalized score to include ANN candidate

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
    validated_phenotypes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# ==============================================================================
# Diagnosis Pipeline
# ==============================================================================
class DiagnosisPipeline:
    """
    End-to-end diagnosis inference pipeline.

    Scoring architecture:
    - GNN-primary: When a trained GNN model is available, the confidence
      score equals the GNN cosine similarity score (normalized to [0,1]).
    - Path-reasoning fallback: When no GNN is available, path reasoning
      scores are used as the confidence score.

    PathReasoner role (explanation-only):
    - Always runs to find evidence paths (Phenotype → Gene → Disease)
    - Provides human-readable reasoning chains for clinician review
    - Does NOT contribute to the confidence score when GNN is active

    Combines:
    - Knowledge graph path reasoning for explainability (P0 core)
    - GNN scoring for primary confidence (P0 core)
    - Ortholog evidence integration (P1 feature, optional)

    Usage:
        pipeline = DiagnosisPipeline(kg=knowledge_graph)
        result = pipeline.run(patient_phenotypes, top_k=10)
    """

    # Version info
    VERSION = "1.0.0"

    def __init__(
        self,
        kg: "KnowledgeGraph",
        config: Optional[PipelineConfig] = None,
        model: Optional[Any] = None,  # Optional pre-loaded GNN model
        checkpoint_path: Optional[str] = None,
        graph_data: Optional[Dict[str, Any]] = None,
        data_dir: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the diagnosis pipeline.

        Args:
            kg: Knowledge graph for reasoning
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
            device: Device for GNN inference ("cpu", "cuda", or None for auto).
        """
        self.kg = kg
        self.config = config or PipelineConfig()
        self.model = model

        # GNN inference state (populated by _init_gnn_inference)
        self._node_embeddings: Optional[Dict[str, Any]] = None
        self._node_id_to_idx: Optional[Dict[str, Dict[str, int]]] = None
        self._graph_data: Optional[Dict[str, Any]] = graph_data
        self._gnn_ready = False

        # Shortest path lookup state (populated by _load_shortest_paths)
        # Sparse storage: parallel tensors of (phenotype_idx, target_idx,
        # target_type, distance). See scripts/compute_shortest_paths.py.
        # When None, the pipeline degrades to pure GNN scoring (eta=1.0).
        self._sp_ph: Optional["torch.Tensor"] = None
        self._sp_tg: Optional["torch.Tensor"] = None
        self._sp_ty: Optional["torch.Tensor"] = None
        self._sp_di: Optional["torch.Tensor"] = None
        self._sp_offsets: Optional[Dict[int, Tuple[int, int]]] = None
        self._sp_max_hops: int = 5
        self._sp_ready = False

        # Fingerprint verification warnings (populated by _load_model_from_checkpoint)
        self._fingerprint_warnings: List[str] = []
        self._checkpoint_meta: Dict[str, Any] = {}

        # Vector index state (populated by _init_vector_index)
        self._vector_index = None
        self._vector_index_ready = False

        # Initialize reasoning components
        self._init_reasoning_components()
        self._init_explanation_generator()

        # Initialize GNN inference if resources are available
        self._init_gnn_inference(
            checkpoint_path=checkpoint_path,
            graph_data=graph_data,
            data_dir=data_dir,
            device=device,
        )

        # Initialize vector index if configured
        self._init_vector_index()

        logger.info(
            f"DiagnosisPipeline initialized: "
            f"version={self.VERSION}, "
            f"kg_nodes={len(kg._nodes)}, "
            f"has_model={self.model is not None}, "
            f"gnn_ready={self._gnn_ready}"
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
                logger.warning(
                    "PyTorch not available. GNN inference disabled. "
                    "Install torch to enable GNN scoring."
                )
            return

        # Nothing to do if no model source is available
        if self.model is None and checkpoint_path is None:
            return

        # Step 1: Load graph data
        if self._graph_data is None and data_dir is not None:
            self._graph_data = self._load_graph_data(data_dir)

        if self._graph_data is None:
            logger.warning(
                "No graph data available for GNN inference. "
                "Provide graph_data or data_dir to enable GNN scoring."
            )
            return

        # Step 2: Load model from checkpoint if needed
        if self.model is None and checkpoint_path is not None:
            self.model = self._load_model_from_checkpoint(
                checkpoint_path, device
            )
            if self.model is None:
                return

        # Step 3: Precompute node embeddings
        self._precompute_node_embeddings(device)

        # Step 4: Build NodeID -> index mapping from KG
        self._node_id_to_idx = self.kg.get_node_id_mapping()

        # Step 5: Load shortest path lookup table (optional)
        if data_dir is not None:
            self._load_shortest_paths(Path(data_dir))

        self._gnn_ready = True
        logger.info(
            f"GNN inference initialized successfully "
            f"(sp_ready={self._sp_ready}, eta={self.config.eta})"
        )

        if not self._sp_ready and not self.config.sp_optional:
            logger.error(
                "Shortest path lookup required (sp_optional=False) but "
                "no shortest_paths.pt found in data_dir. Run "
                "scripts/compute_shortest_paths.py to generate it."
            )
            self._gnn_ready = False

    def _load_graph_data(self, data_dir: str) -> Optional[Dict[str, Any]]:
        """Load graph data files from a processed data directory."""
        data_path = Path(data_dir)
        graph_data: Dict[str, Any] = {
            "x_dict": {},
            "edge_index_dict": {},
            "num_nodes_dict": {},
        }

        # Load node features
        node_features_path = data_path / "node_features.pt"
        if node_features_path.exists():
            graph_data["x_dict"] = torch.load(
                node_features_path, map_location="cpu", weights_only=True
            )
            logger.info(
                f"Loaded node features: {list(graph_data['x_dict'].keys())}"
            )
        else:
            logger.warning(f"Node features not found: {node_features_path}")
            return None

        # Load edge indices
        edge_indices_path = data_path / "edge_indices.pt"
        if edge_indices_path.exists():
            graph_data["edge_index_dict"] = torch.load(
                edge_indices_path, map_location="cpu", weights_only=True
            )
            logger.info(
                f"Loaded edge indices: "
                f"{len(graph_data['edge_index_dict'])} edge types"
            )
        else:
            logger.warning(f"Edge indices not found: {edge_indices_path}")
            return None

        # Load node counts
        num_nodes_path = data_path / "num_nodes.json"
        if num_nodes_path.exists():
            with open(num_nodes_path) as f:
                graph_data["num_nodes_dict"] = json.load(f)
        else:
            # Infer from feature tensors
            for node_type, features in graph_data["x_dict"].items():
                graph_data["num_nodes_dict"][node_type] = features.size(0)

        return graph_data

    def _load_shortest_paths(self, data_dir: Path) -> None:
        """
        Load pre-computed shortest path lookup table from data_dir.

        Expected format (produced by scripts/compute_shortest_paths.py):
            shortest_paths.pt: dict with keys
                phenotype_idx: int64 tensor (N,)
                target_idx:    int64 tensor (N,)
                target_type:   int64 tensor (N,) — 0=gene, 1=disease
                distance:      int8  tensor (N,)
            shortest_paths.meta.json: {"max_hops": int, ...}

        Builds an in-memory dict for O(1) lookup keyed by
        (phenotype_idx, target_idx, target_type) → distance. This trades
        a one-time init cost for fast per-query access during inference.

        If the file is missing and sp_optional=True, the pipeline silently
        falls back to pure GNN scoring (eta is ignored, treated as 1.0).
        """
        sp_path = data_dir / "shortest_paths.pt"
        if not sp_path.exists():
            logger.info(
                f"No shortest_paths.pt found in {data_dir}; "
                f"scoring will use pure GNN (eta=1.0 effective)"
            )
            return

        try:
            sp_data = torch.load(sp_path, map_location="cpu", weights_only=True)
        except Exception as e:
            logger.warning(f"Failed to load {sp_path}: {e}")
            return

        # Validate expected keys
        required = {"phenotype_idx", "target_idx", "target_type", "distance"}
        if not required.issubset(sp_data.keys()):
            logger.warning(
                f"shortest_paths.pt missing required keys. "
                f"Expected {required}, got {set(sp_data.keys())}"
            )
            return

        n_pairs = sp_data["distance"].numel()

        # Compact to int32 one at a time, freeing originals to limit peak RAM.
        ph_t = sp_data["phenotype_idx"].to(torch.int32)
        del sp_data["phenotype_idx"]
        tg_t = sp_data["target_idx"].to(torch.int32)
        del sp_data["target_idx"]
        ty_t = sp_data["target_type"].to(torch.int8)
        del sp_data["target_type"]
        di_t = sp_data["distance"]  # already int8
        del sp_data

        # Sort by phenotype_idx so each phenotype's entries are contiguous.
        sort_idx = ph_t.argsort()
        self._sp_ph = ph_t[sort_idx]
        del ph_t
        self._sp_tg = tg_t[sort_idx]
        del tg_t
        self._sp_ty = ty_t[sort_idx]
        del ty_t
        self._sp_di = di_t[sort_idx]
        del di_t, sort_idx

        # Build offset table using tensor ops (no .tolist()).
        # Find indices where phenotype_idx changes value.
        changes = torch.where(self._sp_ph[1:] != self._sp_ph[:-1])[0] + 1
        starts = torch.cat([torch.zeros(1, dtype=torch.int64), changes])
        ends = torch.cat([changes, torch.tensor([len(self._sp_ph)], dtype=torch.int64)])
        unique_phs = self._sp_ph[starts].tolist()  # only ~19K ints, trivial
        starts_list = starts.tolist()
        ends_list = ends.tolist()
        del changes, starts, ends

        self._sp_offsets: Dict[int, Tuple[int, int]] = {}
        for i, ph in enumerate(unique_phs):
            self._sp_offsets[ph] = (starts_list[i], ends_list[i])

        self._sp_ready = True

        # Load metadata sidecar if present
        meta_path = sp_path.with_suffix(".meta.json")
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                self._sp_max_hops = int(meta.get("max_hops", 5))
            except Exception:
                pass

        logger.info(
            f"Loaded shortest paths: {n_pairs:,} pairs, "
            f"max_hops={self._sp_max_hops}"
        )

    def _load_model_from_checkpoint(
        self,
        checkpoint_path: str,
        device: Optional[str],
    ) -> Optional[Any]:
        """
        Load ShepherdGNN model from a training checkpoint.

        Handles both checkpoint formats:
        - Trainer format: key "model_state_dict"
        - ModelCheckpoint callback format: key "state_dict"
        """
        from src.models.gnn.shepherd_gnn import ShepherdGNN, ShepherdGNNConfig

        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            logger.error(f"Checkpoint not found: {ckpt_path}")
            return None

        logger.info(f"Loading GNN model from checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Extract state dict (handle both formats)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            logger.error(
                f"Checkpoint has no recognized state dict key. "
                f"Available keys: {list(checkpoint.keys())}"
            )
            return None

        # Extract model config from checkpoint
        ckpt_config = checkpoint.get("config", {})

        # Verify data fingerprint compatibility (KG version check)
        from src.utils.fingerprint import verify_fingerprint
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

        # Reconstruct model architecture.
        # CRITICAL: metadata MUST be derived from graph_data["edge_index_dict"]
        # to match how scripts/train_model.py builds the model. The trainer
        # uses graph_data keys (which include rev_* reverse edges for
        # bidirectional message passing), NOT kg.metadata() (which only has
        # forward edges from the KG). Using kg.metadata() here would create a
        # model with fewer conv layers than the checkpoint expects, causing
        # state_dict load to fail.
        if self._graph_data is None:
            logger.error(
                "Cannot reconstruct model: graph_data not loaded. "
                "_init_gnn_inference must load graph_data before model."
            )
            return None

        node_types = list(self._graph_data["x_dict"].keys())
        edge_types = list(self._graph_data["edge_index_dict"].keys())
        metadata = (node_types, edge_types)

        # Get in_channels_dict from graph data features
        in_channels_dict = {}
        for node_type, features in self._graph_data["x_dict"].items():
            if features.dim() >= 2:
                in_channels_dict[node_type] = features.size(-1)

        # Reconstruct the model architecture from the checkpoint, by precedence:
        #   1) model_config sub-dict (self-describing, current trainers)
        #   2) legacy flat arch fields  3) inference from weight names  4) defaults
        # (see _resolve_arch_params). This makes the loader deterministic for
        # self-describing checkpoints while still recovering legacy ones, so an
        # HGT/SAGE checkpoint is never silently rebuilt as GAT.
        import dataclasses as _dc

        from src.config.model_types import SUPPORTED_CONV_TYPES

        state_keys = set(state_dict.keys())
        has_pos_encoder = any(k.startswith("pos_encoder.") for k in state_keys)
        has_ortholog_gate = any(k.startswith("ortholog_gate.") for k in state_keys)

        valid_fields = {f.name for f in _dc.fields(ShepherdGNNConfig)}
        arch_params = _resolve_arch_params(
            ckpt_config,
            state_keys,
            valid_fields=valid_fields,
            supported_conv=SUPPORTED_CONV_TYPES,
            has_pos_encoder=has_pos_encoder,
            has_ortholog_gate=has_ortholog_gate,
        )
        model_config = ShepherdGNNConfig(**arch_params)
        hidden_dim = model_config.hidden_dim

        # Provide default in_channels if not inferred from data
        if not in_channels_dict:
            for node_type in node_types:
                in_channels_dict[node_type] = hidden_dim

        model = ShepherdGNN(
            metadata=metadata,
            in_channels_dict=in_channels_dict,
            config=model_config,
        )

        # Load trained weights. If this fails, log the SPECIFIC mismatch so
        # operators can diagnose checkpoint compat issues (e.g., metadata
        # source mismatch between trainer and inference paths).
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            logger.error(
                f"Failed to load state dict from {ckpt_path}: {e}\n"
                f"Model expects {len(metadata[1])} edge types: "
                f"{[f'{s}--{r}--{t}' for s, r, t in metadata[1]][:5]}...\n"
                f"Checkpoint has {len(state_dict)} parameter tensors. "
                f"Common cause: trainer and inference use different metadata "
                f"sources (trainer uses graph_data keys, inference must too)."
            )
            return None

        model.eval()

        # Store checkpoint training metadata for UI display
        self._checkpoint_meta = {
            "epoch": checkpoint.get("epoch"),
            "params": sum(p.numel() for p in model.parameters()),
            "device": str(device) if device else "cpu",
        }
        # Extract best metrics from logs if available
        logs = checkpoint.get("logs", {})
        if isinstance(logs, dict):
            for key in ("val_loss", "train_loss", "mrr", "hits_at_1", "hits_at_10"):
                if key in logs:
                    self._checkpoint_meta[key] = logs[key]

        # Move to device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        self._checkpoint_meta["device"] = str(device)

        num_params = self._checkpoint_meta["params"]
        logger.info(
            f"GNN model loaded: {num_params:,} parameters, device={device}"
        )
        return model

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

        logger.info("Precomputing node embeddings via GNN forward pass...")

        # Move graph data to model device
        x_dict = {
            k: v.to(device) for k, v in self._graph_data["x_dict"].items()
        }
        edge_index_dict = {
            k: v.to(device)
            for k, v in self._graph_data["edge_index_dict"].items()
        }

        # Forward pass (no gradient needed for inference)
        with torch.no_grad():
            embeddings = self.model(x_dict, edge_index_dict)

        # Move embeddings to CPU for memory efficiency
        self._node_embeddings = {
            k: v.cpu() for k, v in embeddings.items()
        }

        emb_info = {k: v.shape for k, v in self._node_embeddings.items()}
        logger.info(f"Node embeddings cached: {emb_info}")

    def _init_vector_index(self) -> None:
        """
        Load pre-built vector index for ANN candidate retrieval.

        The vector index enables discovery of disease candidates that may not
        be reachable via BFS paths in the knowledge graph, allowing the GNN's
        latent knowledge to surface novel associations.

        Requires:
        - config.vector_index_path to be set
        - GNN to be ready (for query embedding computation)
        """
        if not self.config.vector_index_path:
            return

        try:
            from src.retrieval import create_index

            index_path = Path(self.config.vector_index_path)

            # Look for disease index file specifically
            disease_index_path = index_path.parent / f"{index_path.name}_disease"

            # Determine dimension from GNN embeddings if available
            dim = 256  # default
            if self._node_embeddings is not None:
                disease_emb = self._node_embeddings.get(NodeType.DISEASE.value)
                if disease_emb is not None:
                    dim = disease_emb.shape[1]

            index = create_index(backend="auto", dim=dim)
            index.load(disease_index_path)

            self._vector_index = index
            self._vector_index_ready = True
            logger.info(
                f"Vector index loaded: {len(index)} disease entities, "
                f"dim={index.dim}, backend={index.backend_name}"
            )

        except FileNotFoundError as e:
            logger.warning(f"Vector index files not found: {e}")
        except Exception as e:
            logger.warning(f"Failed to load vector index: {e}")

    def _find_ann_candidates(
        self,
        source_ids: List[NodeID],
    ) -> Dict[str, float]:
        """
        Use vector index to find disease candidates by ANN search.

        Aggregates patient phenotype embeddings (mean pooling) and queries
        the disease vector index for nearest neighbors.

        Args:
            source_ids: Patient phenotype NodeIDs

        Returns:
            {disease_id_str: ann_score} for candidates above threshold
        """
        if not self._vector_index_ready or self._node_embeddings is None:
            return {}

        import numpy as np

        node_mapping = self._node_id_to_idx
        phenotype_type = NodeType.PHENOTYPE.value

        # Aggregate patient phenotype embeddings
        phenotype_emb = self._node_embeddings.get(phenotype_type)
        if phenotype_emb is None:
            return {}

        phenotype_mapping = node_mapping.get(phenotype_type, {})
        indices = []
        for nid in source_ids:
            idx = phenotype_mapping.get(str(nid))
            if idx is not None:
                indices.append(idx)

        if not indices:
            return {}

        # Mean pooling of phenotype embeddings
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        indices_tensor = indices_tensor.clamp(min=0, max=phenotype_emb.size(0) - 1)
        selected = phenotype_emb[indices_tensor]  # (N, H)
        patient_emb = selected.mean(dim=0)  # (H,)

        # Normalize for cosine similarity (inner product on normalized = cosine)
        patient_emb = F.normalize(patient_emb, dim=0)
        query = patient_emb.numpy().astype(np.float32)

        # ANN search
        results = self._vector_index.search(query, top_k=self.config.ann_top_k)

        # Normalize scores to [0, 1] and filter by threshold
        candidates = {}
        for entity_id, distance in results:
            # Inner product similarity: higher is better, normalize to [0, 1]
            score = (distance + 1.0) / 2.0
            if score >= self.config.ann_score_threshold:
                candidates[entity_id] = score

        logger.info(
            f"ANN search returned {len(results)} results, "
            f"{len(candidates)} above threshold {self.config.ann_score_threshold}"
        )
        return candidates

    def run(
        self,
        patient_input: PatientPhenotypes,
        top_k: int = 10,
        include_explanations: Optional[bool] = None,
        include_ortholog_evidence: Optional[bool] = None,
        include_literature_evidence: Optional[bool] = None,
    ) -> InferenceResult:
        """
        Execute the full diagnosis pipeline.

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
        warnings: List[str] = []

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

        # Step 1: Validate input
        validation = self.validate_input(patient_input)
        if not validation.is_valid:
            logger.error(f"Input validation failed: {validation.errors}")
            return InferenceResult(
                patient_id=patient_input.patient_id,
                timestamp=datetime.now(),
                candidates=[],
                warnings=validation.errors,
                model_version=self.VERSION,
                inference_time_ms=(time.time() - start_time) * 1000,
            )
        warnings.extend(validation.warnings)

        # Step 2: Convert phenotypes to NodeIDs
        source_ids = self._phenotypes_to_node_ids(validation.validated_phenotypes)
        if not source_ids:
            logger.warning("No valid phenotype NodeIDs found")
            return self._create_empty_result(
                patient_input.patient_id,
                start_time,
                warnings + ["No valid phenotypes found in knowledge graph"],
            )

        # Step 3: Find reasoning paths (BFS-based)
        logger.debug("Finding reasoning paths...")
        all_paths = self._find_all_paths(source_ids, include_ortholog_evidence)

        # Step 3b: ANN candidate discovery (vector index)
        # Finds disease candidates via embedding similarity that may lack
        # explicit KG paths. These represent potential novel associations.
        ann_only_candidates: Dict[str, float] = {}
        if self._vector_index_ready and self._gnn_ready:
            ann_candidates = self._find_ann_candidates(source_ids)
            for disease_id_str, ann_score in ann_candidates.items():
                if disease_id_str not in all_paths:
                    # This candidate has no BFS path — novel association
                    ann_only_candidates[disease_id_str] = ann_score
            if ann_only_candidates:
                logger.info(
                    f"ANN discovered {len(ann_only_candidates)} candidates "
                    f"without explicit KG paths (potential novel associations)"
                )

        if not all_paths and not ann_only_candidates:
            logger.warning("No paths found from phenotypes to diseases")
            return self._create_empty_result(
                patient_input.patient_id,
                start_time,
                warnings + ["No paths found from phenotypes to diseases"],
            )

        # Step 4: Score and rank candidates
        logger.debug("Scoring candidates...")
        candidates = self._score_and_rank_candidates(
            all_paths=all_paths,
            source_ids=source_ids,
            patient_input=patient_input,
            top_k=top_k,
            include_ortholog_evidence=include_ortholog_evidence,
            ann_only_candidates=ann_only_candidates,
        )

        # Step 5: Generate explanations + Step C evidence packages
        if include_explanations:
            logger.debug("Generating explanations and evidence packages...")
            candidates = self._add_explanations(
                candidates=candidates,
                all_paths=all_paths,
                patient_input=patient_input,
                source_ids=source_ids,
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
            summary_explanation=summary,
            model_version=self.VERSION,
            kg_version=getattr(self.kg, "version", "unknown"),
            inference_time_ms=inference_time,
            warnings=warnings,
        )

    def validate_input(
        self,
        patient_input: PatientPhenotypes,
    ) -> ValidationResult:
        """
        Validate patient input.

        Args:
            patient_input: Patient phenotype data

        Returns:
            ValidationResult with validated phenotypes and any warnings/errors
        """
        errors: List[str] = []
        warnings: List[str] = []
        validated_phenotypes: List[str] = []

        # Check patient_id
        if not patient_input.patient_id:
            errors.append("patient_id is required")

        # Check phenotype count
        if len(patient_input.phenotypes) < self.config.min_phenotypes:
            errors.append(
                f"At least {self.config.min_phenotypes} phenotype(s) required, "
                f"got {len(patient_input.phenotypes)}"
            )

        if len(patient_input.phenotypes) > self.config.max_phenotypes:
            warnings.append(
                f"More than {self.config.max_phenotypes} phenotypes provided, "
                f"using first {self.config.max_phenotypes}"
            )

        # Validate each phenotype
        for pheno_id in patient_input.phenotypes[:self.config.max_phenotypes]:
            if self._validate_phenotype(pheno_id):
                validated_phenotypes.append(pheno_id)
            else:
                warnings.append(f"Unknown phenotype: {pheno_id}")

        if not validated_phenotypes and not errors:
            errors.append("No valid phenotypes after validation")

        return ValidationResult(
            is_valid=len(errors) == 0,
            validated_phenotypes=validated_phenotypes,
            warnings=warnings,
            errors=errors,
        )

    def _validate_phenotype(self, pheno_id: str) -> bool:
        """Check if a phenotype ID is valid in the KG."""
        if not self.config.validate_phenotypes:
            return True

        # Check if phenotype exists in KG
        node_id = NodeID(source=DataSource.HPO, local_id=pheno_id)
        return self.kg.has_node(node_id)

    def _phenotypes_to_node_ids(self, phenotypes: List[str]) -> List[NodeID]:
        """Convert HPO ID strings to NodeID objects."""
        node_ids = []
        for pheno_id in phenotypes:
            node_id = NodeID(source=DataSource.HPO, local_id=pheno_id)
            if self.kg.has_node(node_id):
                node_ids.append(node_id)
        return node_ids

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
        top_k: int,
        include_ortholog_evidence: bool,
        ann_only_candidates: Optional[Dict[str, float]] = None,
    ) -> List[DiagnosisCandidate]:
        """
        Score and rank disease candidates.

        Args:
            all_paths: Paths grouped by disease
            source_ids: Patient phenotype NodeIDs
            patient_input: Patient phenotype data
            top_k: Number of top candidates to return
            include_ortholog_evidence: Include ortholog evidence
            ann_only_candidates: Disease candidates from ANN that have no
                                BFS paths (potential novel associations)

        Returns:
            List of DiagnosisCandidate sorted by score
        """
        candidates: List[DiagnosisCandidate] = []

        # --- Score path-based candidates ---
        for disease_key, paths in all_paths.items():
            if not paths:
                continue

            # Get disease info
            disease_id = paths[0].target
            disease_node = self.kg.get_node(disease_id)
            disease_name = disease_node.name if disease_node else str(disease_id)

            # Calculate reasoning score from paths (used for explanation
            # quality ranking and as fallback when GNN is unavailable)
            reasoning_score = self._calculate_reasoning_score(paths)

            # Combined score per original SHEPHERD paper:
            #   confidence = eta * embedding_sim + (1-eta) * SP_sim
            # PathReasoner's `reasoning_score` is NOT part of confidence;
            # it's only used as a fallback when GNN is unavailable.
            gnn_score = 0.0
            sp_score = 0.0
            if self._gnn_ready:
                confidence_score, gnn_score, sp_score = (
                    self._calculate_combined_score(
                        source_ids, disease_id, patient_input,
                        target_type_idx=1,  # disease
                    )
                )
            else:
                confidence_score = reasoning_score

            # Extract supporting genes
            supporting_genes = self._extract_supporting_genes(paths)

            # Extract reasoning path nodes
            reasoning_paths = [list(p.nodes) for p in paths[:5]]

            # Determine evidence sources
            evidence_sources = self._determine_evidence_sources(paths)

            # Create candidate
            candidate = DiagnosisCandidate(
                rank=0,  # Will be set after sorting
                disease_id=disease_id,
                disease_name=disease_name,
                confidence_score=confidence_score,
                gnn_score=gnn_score,
                reasoning_score=reasoning_score,
                sp_score=sp_score,
                supporting_genes=supporting_genes,
                reasoning_paths=reasoning_paths,
                evidence_sources=evidence_sources,
            )
            candidates.append(candidate)

        # --- Score ANN-only candidates (no BFS paths — potential novel associations) ---
        if ann_only_candidates and self._gnn_ready:
            for disease_id_str, ann_score in ann_only_candidates.items():
                # Try to resolve the NodeID from KG
                disease_node = None
                disease_id = None
                for node_id_str, node in self.kg._nodes.items():
                    if node_id_str == disease_id_str:
                        disease_id = node.id
                        disease_node = node
                        break

                if disease_id is None:
                    continue

                disease_name = disease_node.name if disease_node else disease_id_str

                # Combined score for ANN-only candidate. SP signal still
                # applies if the lookup table is loaded — even ANN-discovered
                # candidates without explicit BFS paths can have shortest
                # path connectivity in the underlying KG.
                confidence_score, gnn_score, sp_score = (
                    self._calculate_combined_score(
                        source_ids, disease_id, patient_input,
                        target_type_idx=1,  # disease
                    )
                )

                candidate = DiagnosisCandidate(
                    rank=0,
                    disease_id=disease_id,
                    disease_name=disease_name,
                    confidence_score=confidence_score,
                    gnn_score=gnn_score,
                    reasoning_score=0.0,  # No paths found
                    sp_score=sp_score,
                    supporting_genes=[],
                    reasoning_paths=[],
                    evidence_sources=[],
                )

                # Flag as potential novel association in explanation
                candidate.explanation = (
                    "Note: GNN confidence is significantly higher than "
                    "path-based evidence suggests, indicating potential "
                    "novel associations. No direct reasoning paths were "
                    "found in the knowledge graph for this candidate."
                )

                candidates.append(candidate)

        # Sort by confidence score
        candidates.sort(key=lambda c: c.confidence_score, reverse=True)

        # Assign ranks and limit to top_k
        for i, candidate in enumerate(candidates[:top_k]):
            candidate.rank = i + 1

        return candidates[:top_k]

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

    def _calculate_combined_score(
        self,
        source_ids: List[NodeID],
        target_id: NodeID,
        patient_input: PatientPhenotypes,
        target_type_idx: int = 1,  # 1 = disease, 0 = gene
    ) -> Tuple[float, float, float]:
        """
        Compute the final score using the original SHEPHERD formula:

            final = eta * embedding_similarity + (1 - eta) * SP_similarity

        Returns (combined_score, embedding_score, sp_score). When the
        shortest path lookup table is not loaded, the formula degrades
        gracefully to pure GNN scoring (effectively eta=1.0).
        """
        emb_score = self._calculate_gnn_score(
            source_ids, target_id, patient_input
        )

        if not self._sp_ready:
            # No SP table → pure GNN
            return emb_score, emb_score, 0.0

        sp_score = self._calculate_sp_score(
            source_ids, target_id, target_type_idx
        )

        eta = self.config.eta
        combined = eta * emb_score + (1.0 - eta) * sp_score
        return combined, emb_score, sp_score

    def _calculate_sp_score(
        self,
        source_ids: List[NodeID],
        target_id: NodeID,
        target_type_idx: int,
    ) -> float:
        """
        Compute shortest path similarity between patient phenotypes and a
        target node (gene or disease).

        For each phenotype, looks up the pre-computed shortest path length
        to the target. Averages those distances and converts to a similarity
        score in [0, 1] via 1 / (1 + avg_distance). Phenotypes with no
        connection within max_hops contribute the maximum possible distance
        (max_hops + 1) to penalize but not zero out the average.

        Returns 0.0 if the SP table is not loaded or no phenotypes can be
        looked up.
        """
        if not self._sp_ready:
            return 0.0

        node_mapping = self._node_id_to_idx
        if node_mapping is None:
            return 0.0

        phenotype_type = NodeType.PHENOTYPE.value
        target_type_str = (
            NodeType.GENE.value if target_type_idx == 0 else NodeType.DISEASE.value
        )

        # Resolve target index
        target_idx = node_mapping.get(target_type_str, {}).get(str(target_id))
        if target_idx is None:
            return 0.0

        # Resolve phenotype indices
        phenotype_mapping = node_mapping.get(phenotype_type, {})
        phenotype_indices = []
        for nid in source_ids:
            idx = phenotype_mapping.get(str(nid))
            if idx is not None:
                phenotype_indices.append(idx)

        if not phenotype_indices:
            return 0.0

        # Look up distances from per-phenotype sorted index.
        unreachable_distance = float(self._sp_max_hops + 1)
        total = 0.0
        for ph_idx in phenotype_indices:
            offsets = self._sp_offsets.get(ph_idx)
            if offsets is None:
                total += unreachable_distance
                continue
            s, e = offsets
            tg_slice = self._sp_tg[s:e]
            ty_slice = self._sp_ty[s:e]
            di_slice = self._sp_di[s:e]
            mask = (tg_slice == target_idx) & (ty_slice == target_type_idx)
            matches = di_slice[mask]
            if len(matches) > 0:
                total += float(matches[0])
            else:
                total += unreachable_distance

        avg_distance = total / len(phenotype_indices)
        return 1.0 / (1.0 + avg_distance)

    def _calculate_gnn_score(
        self,
        source_ids: List[NodeID],
        disease_id: NodeID,
        patient_input: PatientPhenotypes,
    ) -> float:
        """
        Calculate GNN-based embedding similarity score for a candidate.

        Uses precomputed node embeddings from the GNN forward pass to compute
        cosine similarity between the aggregated patient phenotype profile
        and the candidate disease embedding. This matches the scoring approach
        used during training (see Trainer._compute_model_outputs).

        The cosine similarity [-1, 1] is normalized to [0, 1] via
        (sim + 1) / 2 to match the reasoning_score scale.

        This is the embedding-only signal — the final score is a mixture
        of this and the shortest path similarity, computed in
        _calculate_combined_score().

        Returns 0.0 if GNN inference is not available.
        """
        if not self._gnn_ready or self._node_embeddings is None:
            return 0.0

        node_mapping = self._node_id_to_idx
        phenotype_type = NodeType.PHENOTYPE.value  # "phenotype"

        # 1. Map phenotype NodeIDs to integer indices
        phenotype_indices = []
        phenotype_mapping = node_mapping.get(phenotype_type, {})
        for nid in source_ids:
            nid_str = str(nid)
            idx = phenotype_mapping.get(nid_str)
            if idx is not None:
                phenotype_indices.append(idx)

        if not phenotype_indices:
            return 0.0

        # 2. Get phenotype embeddings and aggregate (mean pooling)
        phenotype_emb = self._node_embeddings.get(phenotype_type)
        if phenotype_emb is None:
            return 0.0

        indices_tensor = torch.tensor(phenotype_indices, dtype=torch.long)
        # Clamp indices to valid range
        indices_tensor = indices_tensor.clamp(
            min=0, max=phenotype_emb.size(0) - 1
        )
        selected_emb = phenotype_emb[indices_tensor]  # (N, hidden_dim)
        patient_embedding = selected_emb.mean(dim=0, keepdim=True)  # (1, H)

        # 3. Map disease NodeID to integer index
        disease_type = NodeType.DISEASE.value  # "disease"
        disease_str = str(disease_id)
        disease_idx = node_mapping.get(disease_type, {}).get(disease_str)
        if disease_idx is None:
            return 0.0

        disease_emb = self._node_embeddings.get(disease_type)
        if disease_emb is None:
            return 0.0

        disease_idx = min(disease_idx, disease_emb.size(0) - 1)
        disease_embedding = disease_emb[disease_idx].unsqueeze(0)  # (1, H)

        # 4. Cosine similarity (matches training: Trainer._compute_model_outputs)
        patient_norm = F.normalize(patient_embedding, dim=-1)
        disease_norm = F.normalize(disease_embedding, dim=-1)
        cosine_sim = torch.mm(patient_norm, disease_norm.t()).item()

        # 5. Normalize from [-1, 1] to [0, 1]
        score = (cosine_sim + 1.0) / 2.0

        return score

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
        candidates: List[DiagnosisCandidate],
        all_paths: Dict[str, List[ReasoningPath]],
        patient_input: PatientPhenotypes,
        source_ids: Optional[List[NodeID]] = None,
    ) -> List[DiagnosisCandidate]:
        """
        Add explanations and Step C evidence packages to each candidate.

        For each candidate, the EvidencePanel decides between:
          - Mode A (direct path): use existing paths from BFS, label as
            STRONG/WEAK depending on shortest hop count
          - Mode B (analogy-based): when no direct path is available, find
            the K nearest known candidates in GNN embedding space whose
            paths to the patient phenotypes DO exist, and surface them
          - INSUFFICIENT: if neither mode succeeds
        """
        for candidate in candidates:
            disease_key = str(candidate.disease_id)
            paths = all_paths.get(disease_key, [])

            # Legacy narrative explanation (kept for backwards compat)
            explanation = self.explanation_generator.generate_explanation(
                candidate=candidate,
                phenotypes=patient_input,
                kg=self.kg,
                paths=paths,
            )
            candidate.explanation = explanation

            # Step C: structured evidence package with confidence label.
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
                candidate.evidence_package = package.to_dict()
                candidate.confidence_label = package.confidence_label.value

        return candidates

    def _generate_summary_explanation(
        self,
        top_candidates: List[DiagnosisCandidate],
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
            confidence_level = (
                "HIGH" if candidate.confidence_score >= 0.7
                else "MEDIUM" if candidate.confidence_score >= 0.4
                else "LOW"
            )
            lines.append(
                f"**{candidate.rank}. {candidate.disease_name}** "
                f"(Confidence: {confidence_level}, Score: {candidate.confidence_score:.3f})"
            )
            if candidate.supporting_genes:
                lines.append(
                    f"   Key genes: {', '.join(candidate.supporting_genes[:3])}"
                )
            lines.append("")

        lines.extend([
            "---",
            "*Note: These results are computational predictions for clinical review. "
            "Final diagnosis should be made by qualified healthcare professionals.*",
        ])

        return "\n".join(lines)

    def _create_empty_result(
        self,
        patient_id: str,
        start_time: float,
        warnings: List[str],
    ) -> InferenceResult:
        """Create an empty result with warnings."""
        return InferenceResult(
            patient_id=patient_id,
            timestamp=datetime.now(),
            candidates=[],
            warnings=warnings,
            model_version=self.VERSION,
            inference_time_ms=(time.time() - start_time) * 1000,
        )

    def get_pipeline_config(self) -> Dict[str, Any]:
        """Get pipeline configuration as dictionary."""
        # Effective scoring mode reflects whether SP signal is contributing
        if self._gnn_ready and self._sp_ready:
            scoring_mode = "gnn_plus_shortest_path"
            effective_eta = self.config.eta
        elif self._gnn_ready:
            scoring_mode = "gnn_only"
            effective_eta = 1.0
        else:
            scoring_mode = "path_reasoning_fallback"
            effective_eta = 0.0

        return {
            "version": self.VERSION,
            "max_path_length": self.config.max_path_length,
            "path_length_penalty": self.config.path_length_penalty,
            "aggregation_method": self.config.aggregation_method,
            "scoring_mode": scoring_mode,
            "eta_configured": self.config.eta,
            "eta_effective": effective_eta,
            "sp_max_hops": self._sp_max_hops if self._sp_ready else None,
            "path_reasoner_role": "explanation_only" if self._gnn_ready else "scoring_and_explanation",
            "include_explanations": self.config.include_explanations,
            "include_ortholog_evidence": self.config.include_ortholog_evidence,
            "include_literature_evidence": self.config.include_literature_evidence,
            "has_model": self.model is not None,
            "gnn_ready": self._gnn_ready,
            "sp_ready": self._sp_ready,
            "vector_index_ready": self._vector_index_ready,
            "vector_index_size": len(self._vector_index) if self._vector_index else 0,
            "fingerprint_warnings": getattr(self, "_fingerprint_warnings", []),
            "checkpoint_meta": getattr(self, "_checkpoint_meta", {}),
            "kg_nodes": len(self.kg._nodes) if self.kg else 0,
            "kg_edges": len(self.kg._edges) if self.kg else 0,
        }


# ==============================================================================
# Factory Function
# ==============================================================================
def create_diagnosis_pipeline(
    kg: "KnowledgeGraph",
    config: Optional[PipelineConfig] = None,
    model: Optional[Any] = None,
    checkpoint_path: Optional[str] = None,
    graph_data: Optional[Dict[str, Any]] = None,
    data_dir: Optional[str] = None,
    device: Optional[str] = None,
    vector_index_path: Optional[str] = None,
) -> DiagnosisPipeline:
    """
    Factory function to create a diagnosis pipeline.

    Args:
        kg: Knowledge graph
        config: Pipeline configuration
        model: Optional pre-loaded GNN model
        checkpoint_path: Path to trained model checkpoint
        graph_data: Pre-loaded graph data dict
        data_dir: Path to processed data directory
        device: Device for GNN inference
        vector_index_path: Path to pre-built vector index for ANN search

    Returns:
        Configured DiagnosisPipeline instance
    """
    if vector_index_path and config is None:
        config = PipelineConfig(vector_index_path=vector_index_path)
    elif vector_index_path and config is not None:
        config.vector_index_path = vector_index_path

    return DiagnosisPipeline(
        kg=kg,
        config=config,
        model=model,
        checkpoint_path=checkpoint_path,
        graph_data=graph_data,
        data_dir=data_dir,
        device=device,
    )
