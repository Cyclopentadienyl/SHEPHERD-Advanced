"""
# ==============================================================================
# Module: src/reasoning/evidence_panel.py
# ==============================================================================
# Purpose: Generate clinician-facing evidence for diagnosis candidates.
#
# Evidence is computed AFTER scoring and never influences the ranking. The
# scoring is done by the GNN + shortest path mixture in pipeline.py
# (see _calculate_combined_score). This module's job is to answer the
# question "WHY is this candidate ranked here?" in a form a doctor can
# read and trust.
#
# Two evidence modes per the original SHEPHERD paper's design intent:
#
#   Mode A — Direct Path Evidence
#     When the knowledge graph contains explicit short paths (≤3 hops by
#     default) from patient phenotypes to the candidate, surface those
#     paths verbatim as the evidence. Strong, traditional, easy to trust.
#
#   Mode B — Analogy-Based Evidence (the GNN's killer feature)
#     When NO direct path exists in the KG (zero-shot situation) but the
#     GNN ranks the candidate highly, we use the GNN's own embedding space
#     to find K nearest known genes that DO have paths to the patient's
#     phenotypes. We then present those known genes' paths as analogy
#     evidence: "this candidate has no direct evidence, but is highly
#     similar to gene X, which does — and they share these properties."
#
# Confidence labels are attached to every evidence package so clinicians
# can quickly triage which results to trust at a glance.
#
# Dependencies:
#   - Internal: src.core.types, src.kg, src.reasoning.path_reasoning
#   - Optional: torch (only for Mode B; gracefully degrades without it)
# ==============================================================================
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from src.core.types import (
    DiagnosisCandidate,
    EdgeType,
    NodeID,
    NodeType,
    PatientPhenotypes,
)
from src.reasoning.path_reasoning import PathReasoner, ReasoningPath

if TYPE_CHECKING:
    from src.kg.graph import KnowledgeGraph

logger = logging.getLogger(__name__)


# ==============================================================================
# Evidence types and confidence labels
# ==============================================================================
class EvidenceMode(str, Enum):
    """How the evidence was generated."""
    DIRECT_PATH = "direct_path"
    ANALOGY_BASED = "analogy_based"
    INSUFFICIENT = "insufficient"


class ConfidenceLabel(str, Enum):
    """
    Plain-language label for clinicians, derived from evidence quality.

    Triage hierarchy:
        STRONG_PATH    → direct path of length ≤2 hops
        WEAK_PATH      → direct path of length 3-4 hops
        ANALOGY_BASED  → no direct path, but high embedding similarity to a
                         known gene that does have a direct path
        INSUFFICIENT   → no usable evidence found
    """
    STRONG_PATH = "Strong path support"
    WEAK_PATH = "Weak path support"
    ANALOGY_BASED = "Analogy-based (no direct KG path)"
    INSUFFICIENT = "Insufficient evidence"


@dataclass
class AnalogyMatch:
    """
    A single 'similar known gene' used as analogy evidence.

    When the candidate has no direct KG path but its GNN embedding is close
    to this known gene's embedding, we surface the known gene's paths as
    indirect supporting evidence.
    """
    similar_gene_id: NodeID
    similar_gene_name: str
    embedding_similarity: float  # cosine sim in [0, 1]
    known_paths: List[ReasoningPath] = field(default_factory=list)
    shared_features: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "similar_gene_id": str(self.similar_gene_id),
            "similar_gene_name": self.similar_gene_name,
            "embedding_similarity": round(self.embedding_similarity, 4),
            "num_known_paths": len(self.known_paths),
            "shared_features": self.shared_features,
            "known_paths": [
                [str(n) for n in p.nodes] for p in self.known_paths[:3]
            ],
        }


@dataclass
class EvidencePackage:
    """
    Full evidence bundle attached to a single DiagnosisCandidate.

    Either direct_paths (Mode A) or analogies (Mode B) is populated,
    never both. The mode field encodes which one was used. The confidence
    label summarises the quality at a glance for clinicians.
    """
    mode: EvidenceMode
    confidence_label: ConfidenceLabel
    summary: str  # one-line plain-language description for the UI

    # Mode A payload
    direct_paths: List[ReasoningPath] = field(default_factory=list)
    min_path_length: Optional[int] = None

    # Mode B payload
    analogies: List[AnalogyMatch] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "confidence_label": self.confidence_label.value,
            "summary": self.summary,
            "min_path_length": self.min_path_length,
            "direct_paths": [
                [str(n) for n in p.nodes] for p in self.direct_paths[:5]
            ],
            "analogies": [a.to_dict() for a in self.analogies],
        }


# ==============================================================================
# Evidence Panel
# ==============================================================================
@dataclass
class EvidencePanelConfig:
    """Configuration for EvidencePanel."""

    # Mode A: direct path thresholds
    strong_path_max_hops: int = 2  # ≤2 hops → STRONG_PATH
    weak_path_max_hops: int = 4    # 3-4 hops → WEAK_PATH

    # Mode B: analogy search
    analogy_top_k: int = 3                  # how many similar genes to surface
    analogy_min_similarity: float = 0.5     # cosine sim threshold (post-norm)
    analogy_max_path_length: int = 3        # path search depth on similar genes
    analogy_max_paths_per_gene: int = 5     # cap output volume

    # Cross-cutting
    max_direct_paths_displayed: int = 5


class EvidencePanel:
    """
    Generates clinician-facing evidence packages for diagnosis candidates.

    DOES NOT score. Scoring is the pipeline's job (GNN + shortest path).
    This module only answers "why does this candidate make sense?"
    """

    def __init__(
        self,
        kg: "KnowledgeGraph",
        path_reasoner: Optional[PathReasoner] = None,
        config: Optional[EvidencePanelConfig] = None,
    ):
        self.kg = kg
        self.path_reasoner = path_reasoner or PathReasoner()
        self.config = config or EvidencePanelConfig()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def build_evidence(
        self,
        candidate: DiagnosisCandidate,
        patient_input: PatientPhenotypes,
        source_ids: List[NodeID],
        existing_paths: Optional[List[ReasoningPath]] = None,
        # Mode B inputs (optional — Mode B silently disabled if absent)
        node_embeddings: Optional[Dict[str, Any]] = None,
        node_id_to_idx: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> EvidencePackage:
        """
        Compute the evidence package for a single candidate.

        Decision flow:
          1. If existing_paths (or freshly computed paths) are reasonable → Mode A
          2. Else if Mode B inputs provided → run analogy search → Mode B
          3. Else → INSUFFICIENT

        Args:
            candidate: The ranked candidate (already scored)
            patient_input: Patient phenotype bundle
            source_ids: Phenotype NodeIDs
            existing_paths: Paths already computed by the pipeline (if any)
            node_embeddings: GNN node embeddings keyed by node type
            node_id_to_idx: NodeID-string → index mapping per node type

        Returns:
            EvidencePackage with mode + confidence label + payload
        """
        # Try Mode A first
        direct_paths = existing_paths
        if direct_paths is None:
            direct_paths = self._find_direct_paths(source_ids, candidate.disease_id)

        if direct_paths:
            return self._build_mode_a(direct_paths)

        # Mode A failed. Try Mode B if we have what we need.
        if node_embeddings is not None and node_id_to_idx is not None:
            analogies = self._find_analogies(
                candidate=candidate,
                source_ids=source_ids,
                node_embeddings=node_embeddings,
                node_id_to_idx=node_id_to_idx,
            )
            if analogies:
                return self._build_mode_b(candidate, analogies)

        # No path, no analogy
        return EvidencePackage(
            mode=EvidenceMode.INSUFFICIENT,
            confidence_label=ConfidenceLabel.INSUFFICIENT,
            summary=(
                "No knowledge-graph evidence could be assembled for this "
                "candidate. Consider this result with caution and validate "
                "via clinical judgment."
            ),
        )

    # ------------------------------------------------------------------
    # Mode A: Direct path
    # ------------------------------------------------------------------
    def _find_direct_paths(
        self,
        source_ids: List[NodeID],
        target_id: NodeID,
    ) -> List[ReasoningPath]:
        """
        Search the KG for paths from any source phenotype to the target.

        Used as a fallback when the pipeline didn't already compute paths
        for this candidate (e.g., ANN-discovered candidates).
        """
        all_paths: List[ReasoningPath] = []
        target_node = self.kg.get_node(target_id)
        if target_node is None:
            return []

        target_type = target_node.node_type
        for src in source_ids:
            paths = self.path_reasoner.find_paths(
                source_ids=[src],
                target_type=target_type,
                kg=self.kg,
                max_length=self.config.weak_path_max_hops,
            )
            for p in paths:
                if p.target == target_id:
                    all_paths.append(p)

        return all_paths

    def _build_mode_a(self, paths: List[ReasoningPath]) -> EvidencePackage:
        """Construct a Mode A evidence package from a list of direct paths."""
        # Sort by length, then by score
        sorted_paths = sorted(paths, key=lambda p: (p.length, -p.score))
        min_len = sorted_paths[0].length if sorted_paths else None

        if min_len is not None and min_len <= self.config.strong_path_max_hops:
            label = ConfidenceLabel.STRONG_PATH
            summary = (
                f"Direct knowledge-graph evidence found "
                f"(shortest path = {min_len} hop{'s' if min_len != 1 else ''})."
            )
        elif min_len is not None and min_len <= self.config.weak_path_max_hops:
            label = ConfidenceLabel.WEAK_PATH
            summary = (
                f"Indirect knowledge-graph path exists but is somewhat distant "
                f"(shortest path = {min_len} hops). Verify with clinical context."
            )
        else:
            # Should not happen because _find_direct_paths bounds by weak_max,
            # but defend against caller-supplied paths.
            label = ConfidenceLabel.WEAK_PATH
            summary = (
                f"Distant path support (≥{min_len} hops). Treat as weak signal."
            )

        return EvidencePackage(
            mode=EvidenceMode.DIRECT_PATH,
            confidence_label=label,
            summary=summary,
            direct_paths=sorted_paths[: self.config.max_direct_paths_displayed],
            min_path_length=min_len,
        )

    # ------------------------------------------------------------------
    # Mode B: Analogy via embedding nearest neighbors
    # ------------------------------------------------------------------
    def _find_analogies(
        self,
        candidate: DiagnosisCandidate,
        source_ids: List[NodeID],
        node_embeddings: Dict[str, Any],
        node_id_to_idx: Dict[str, Dict[str, int]],
    ) -> List[AnalogyMatch]:
        """
        Find K most similar known genes/diseases (in GNN embedding space)
        that DO have direct paths from the patient phenotypes, and surface
        those paths as analogy evidence.
        """
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            logger.debug("torch unavailable; Mode B analogy search disabled")
            return []

        target_node = self.kg.get_node(candidate.disease_id)
        if target_node is None:
            return []

        target_type_str = target_node.node_type.value
        target_emb_pool = node_embeddings.get(target_type_str)
        if target_emb_pool is None:
            return []

        target_idx = node_id_to_idx.get(target_type_str, {}).get(
            str(candidate.disease_id)
        )
        if target_idx is None or target_idx >= target_emb_pool.size(0):
            return []

        # Cosine similarity from candidate to all nodes of the same type
        candidate_emb = F.normalize(
            target_emb_pool[target_idx].unsqueeze(0), dim=-1
        )
        all_norm = F.normalize(target_emb_pool, dim=-1)
        sims = torch.mm(candidate_emb, all_norm.t()).squeeze(0)  # (N,)

        # Top-(K+1) so we can drop the candidate itself
        k = min(self.config.analogy_top_k + 1, sims.size(0))
        top_values, top_indices = torch.topk(sims, k=k)

        # Reverse-map indices to NodeIDs
        idx_to_id: Dict[int, str] = {}
        for nid_str, idx in node_id_to_idx.get(target_type_str, {}).items():
            idx_to_id[idx] = nid_str

        analogies: List[AnalogyMatch] = []
        for sim_val, idx in zip(top_values.tolist(), top_indices.tolist()):
            if idx == target_idx:
                continue  # skip the candidate itself
            # Normalize from [-1, 1] to [0, 1] for clinician-friendly display
            norm_sim = (sim_val + 1.0) / 2.0
            if norm_sim < self.config.analogy_min_similarity:
                continue

            similar_id_str = idx_to_id.get(idx)
            if similar_id_str is None:
                continue

            similar_node = self.kg._nodes.get(similar_id_str)
            if similar_node is None:
                continue

            # Run path search on this similar known node
            similar_paths = self._find_direct_paths(source_ids, similar_node.id)
            if not similar_paths:
                continue  # not actually a useful analogy if no path

            shared = self._compute_shared_features(
                candidate.disease_id, similar_node.id
            )

            analogies.append(AnalogyMatch(
                similar_gene_id=similar_node.id,
                similar_gene_name=similar_node.name or similar_id_str,
                embedding_similarity=norm_sim,
                known_paths=similar_paths[: self.config.analogy_max_paths_per_gene],
                shared_features=shared,
            ))

            if len(analogies) >= self.config.analogy_top_k:
                break

        return analogies

    def _build_mode_b(
        self,
        candidate: DiagnosisCandidate,
        analogies: List[AnalogyMatch],
    ) -> EvidencePackage:
        """Construct a Mode B evidence package from analogy matches."""
        top_sim = max(a.embedding_similarity for a in analogies)
        top_name = max(analogies, key=lambda a: a.embedding_similarity).similar_gene_name

        summary = (
            f"No direct knowledge-graph path. Inference is based on "
            f"structural similarity to {len(analogies)} known node"
            f"{'s' if len(analogies) != 1 else ''} (closest: {top_name}, "
            f"similarity = {top_sim:.2f}). Treat as an indirect signal."
        )

        return EvidencePackage(
            mode=EvidenceMode.ANALOGY_BASED,
            confidence_label=ConfidenceLabel.ANALOGY_BASED,
            summary=summary,
            analogies=analogies,
        )

    # ------------------------------------------------------------------
    # Shared features (cheap heuristic — overlap of KG neighbors)
    # ------------------------------------------------------------------
    def _compute_shared_features(
        self,
        node_a: NodeID,
        node_b: NodeID,
    ) -> List[str]:
        """
        Compute a small set of human-readable shared features between
        two nodes by looking at their KG neighborhoods.

        Returns up to 5 short bullet strings, e.g.:
            ["3 shared phenotypes", "2 shared pathways"]

        Designed to be cheap (one neighbor query per node).
        """
        try:
            neighbors_a = {
                str(n) for n, _ in self.kg.get_neighbors(node_a, direction="both")
            }
            neighbors_b = {
                str(n) for n, _ in self.kg.get_neighbors(node_b, direction="both")
            }
        except Exception as e:
            logger.debug(f"Shared feature computation failed: {e}")
            return []

        shared = neighbors_a & neighbors_b
        if not shared:
            return []

        # Group shared neighbors by node type
        by_type: Dict[str, int] = {}
        for nid_str in shared:
            node = self.kg._nodes.get(nid_str)
            if node is not None:
                key = node.node_type.value
                by_type[key] = by_type.get(key, 0) + 1

        features: List[str] = []
        for ntype, count in sorted(by_type.items(), key=lambda x: -x[1]):
            label = ntype.replace("_", " ")
            features.append(f"{count} shared {label}{'s' if count != 1 else ''}")
            if len(features) >= 5:
                break
        return features


def create_evidence_panel(
    kg: "KnowledgeGraph",
    path_reasoner: Optional[PathReasoner] = None,
    config: Optional[EvidencePanelConfig] = None,
) -> EvidencePanel:
    """Factory function for EvidencePanel."""
    return EvidencePanel(kg=kg, path_reasoner=path_reasoner, config=config)
