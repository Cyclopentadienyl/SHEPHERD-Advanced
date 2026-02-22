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
#       data_dir="data/processed",
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
    create_path_reasoner,
    create_explanation_generator,
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
# Configuration
# ==============================================================================
@dataclass
class PipelineConfig:
    """Configuration for the diagnosis pipeline."""

    # Path reasoning
    max_path_length: int = 4
    path_length_penalty: float = 0.9
    aggregation_method: str = "weighted_sum"

    # Scoring: GNN is the primary scoring source when available.
    # PathReasoner provides explanation paths only (not blended into score).
    # When GNN is unavailable, reasoning_score is used as fallback.
    # Legacy weight fields kept for backward compatibility / future tuning.
    reasoning_weight: float = 0.5
    gnn_weight: float = 0.5
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
        """Initialize explanation generator."""
        self.explanation_generator = create_explanation_generator(
            include_ortholog_evidence=self.config.include_ortholog_evidence,
            include_literature_evidence=self.config.include_literature_evidence,
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

        self._gnn_ready = True
        logger.info("GNN inference initialized successfully")

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

        # Reconstruct model architecture
        # Get metadata from KG
        kg_metadata = self.kg.metadata()

        # Get in_channels_dict from graph data features
        in_channels_dict = {}
        for node_type, features in self._graph_data["x_dict"].items():
            if features.dim() >= 2:
                in_channels_dict[node_type] = features.size(-1)

        # Build model config from checkpoint config or defaults.
        # Infer boolean flags from state dict keys when not in config,
        # so we don't create layers that weren't in the trained model.
        hidden_dim = ckpt_config.get("hidden_dim", 256)
        state_keys = set(state_dict.keys())

        has_pos_encoder = any(k.startswith("pos_encoder.") for k in state_keys)
        has_ortholog_gate = any(
            k.startswith("ortholog_gate.") for k in state_keys
        )

        model_config = ShepherdGNNConfig(
            hidden_dim=hidden_dim,
            num_layers=ckpt_config.get("num_layers", 4),
            num_heads=ckpt_config.get("num_heads", 8),
            conv_type=ckpt_config.get("conv_type", "gat"),
            dropout=0.0,  # No dropout at inference time
            use_positional_encoding=ckpt_config.get(
                "use_positional_encoding", has_pos_encoder
            ),
            use_ortholog_gate=ckpt_config.get(
                "use_ortholog_gate", has_ortholog_gate
            ),
        )

        # Provide default in_channels if not inferred from data
        if not in_channels_dict:
            for node_type in kg_metadata[0]:
                in_channels_dict[node_type] = hidden_dim

        model = ShepherdGNN(
            metadata=kg_metadata,
            in_channels_dict=in_channels_dict,
            config=model_config,
        )

        # Load trained weights
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            logger.error(f"Failed to load state dict: {e}")
            return None

        model.eval()

        # Move to device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        num_params = sum(p.numel() for p in model.parameters())
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

        # Step 5: Generate explanations
        if include_explanations:
            logger.debug("Generating explanations...")
            candidates = self._add_explanations(
                candidates=candidates,
                all_paths=all_paths,
                patient_input=patient_input,
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

            # Calculate GNN score if model available
            gnn_score = 0.0
            if self._gnn_ready:
                gnn_score = self._calculate_gnn_score(
                    source_ids, disease_id, patient_input
                )

            # Confidence score: GNN-primary when available, path-reasoning
            # as fallback. PathReasoner's role is explanation, not scoring.
            if self._gnn_ready:
                confidence_score = gnn_score
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
                        disease_id = node.node_id
                        disease_node = node
                        break

                if disease_id is None:
                    continue

                disease_name = disease_node.name if disease_node else disease_id_str

                # Compute GNN score
                gnn_score = self._calculate_gnn_score(
                    source_ids, disease_id, patient_input
                )

                # For ANN-only candidates, confidence comes from GNN
                confidence_score = gnn_score

                candidate = DiagnosisCandidate(
                    rank=0,
                    disease_id=disease_id,
                    disease_name=disease_name,
                    confidence_score=confidence_score,
                    gnn_score=gnn_score,
                    reasoning_score=0.0,  # No paths found
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

    def _calculate_gnn_score(
        self,
        source_ids: List[NodeID],
        disease_id: NodeID,
        patient_input: PatientPhenotypes,
    ) -> float:
        """
        Calculate GNN-based score for a disease candidate.

        Uses precomputed node embeddings from the GNN forward pass to compute
        cosine similarity between the aggregated patient phenotype profile
        and the candidate disease embedding. This matches the scoring approach
        used during training (see Trainer._compute_model_outputs).

        The cosine similarity [-1, 1] is normalized to [0, 1] via
        (sim + 1) / 2 to match the reasoning_score scale.

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
    ) -> List[DiagnosisCandidate]:
        """Add explanations to candidates."""
        for candidate in candidates:
            disease_key = str(candidate.disease_id)
            paths = all_paths.get(disease_key, [])

            explanation = self.explanation_generator.generate_explanation(
                candidate=candidate,
                phenotypes=patient_input,
                kg=self.kg,
                paths=paths,
            )
            candidate.explanation = explanation

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
        return {
            "version": self.VERSION,
            "max_path_length": self.config.max_path_length,
            "path_length_penalty": self.config.path_length_penalty,
            "aggregation_method": self.config.aggregation_method,
            "scoring_mode": "gnn_primary" if self._gnn_ready else "path_reasoning_fallback",
            "path_reasoner_role": "explanation_only" if self._gnn_ready else "scoring_and_explanation",
            "include_explanations": self.config.include_explanations,
            "include_ortholog_evidence": self.config.include_ortholog_evidence,
            "include_literature_evidence": self.config.include_literature_evidence,
            "has_model": self.model is not None,
            "gnn_ready": self._gnn_ready,
            "vector_index_ready": self._vector_index_ready,
            "vector_index_size": len(self._vector_index) if self._vector_index else 0,
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
