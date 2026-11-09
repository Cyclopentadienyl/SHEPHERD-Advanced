"""
# ==============================================================================
# Module: src/reasoning/evidence_panel.py
# ==============================================================================
# Purpose: Generate clinician-facing evidence for diagnosis candidates.
#
# Evidence is computed AFTER scoring and never influences the ranking. The
# scoring is done in pipeline.py, by the served task model's own score. This
# module's job is to answer the question "WHY is this candidate ranked
# here?" in a form a doctor can read and trust.
#
#   Mode A — Direct Path Evidence
#     When the knowledge graph contains explicit short paths (≤3 hops by
#     default) from patient phenotypes to the candidate, surface those
#     paths verbatim as the evidence. Strong, traditional, easy to trust.
#
#   INSUFFICIENT
#     When no direct path exists, the package says so: no evidence is
#     assembled from anything else.
#
# An evidence label is attached to every evidence package so clinicians can
# see at a glance how strong the evidence for a candidate is. It grades the
# evidence, never a confidence in the diagnosis (`evidence_label`, Spec 1 §2.1).
#
# Dependencies:
#   - Internal: src.core.types, src.kg, src.reasoning.path_reasoning
# ==============================================================================
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from src.core.types import (
    Candidate,
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
    INSUFFICIENT = "insufficient"


class ConfidenceLabel(str, Enum):
    """
    Plain-language label for clinicians, derived from evidence quality: the
    candidate's `evidence_label` (the class keeps its older name).

    Triage hierarchy:
        STRONG_PATH    → direct path of length ≤2 hops
        WEAK_PATH      → direct path of length 3-4 hops
        INSUFFICIENT   → no usable evidence found
    """
    STRONG_PATH = "Strong path support"
    WEAK_PATH = "Weak path support"
    INSUFFICIENT = "Insufficient evidence"


@dataclass
class EvidencePackage:
    """
    Full evidence bundle attached to a single candidate.

    direct_paths is populated in Mode A and empty when the evidence is
    insufficient; the mode field encodes which. The evidence label
    summarises the evidence's strength at a glance for clinicians.
    """
    mode: EvidenceMode
    evidence_label: ConfidenceLabel
    summary: str  # one-line plain-language description for the UI

    # Mode A payload
    direct_paths: List[ReasoningPath] = field(default_factory=list)
    min_path_length: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "evidence_label": self.evidence_label.value,
            "summary": self.summary,
            "min_path_length": self.min_path_length,
            "direct_paths": [
                [str(n) for n in p.nodes] for p in self.direct_paths[:5]
            ],
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

    # Cross-cutting
    max_direct_paths_displayed: int = 5


class EvidencePanel:
    """
    Generates clinician-facing evidence packages for diagnosis candidates.

    DOES NOT score. Scoring is the pipeline's job (the served task model).
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
        candidate: Candidate,
        patient_input: PatientPhenotypes,
        source_ids: List[NodeID],
        existing_paths: Optional[List[ReasoningPath]] = None,
    ) -> EvidencePackage:
        """
        Compute the evidence package for a single candidate.

        Decision flow:
          1. If existing_paths (or freshly computed paths) are reasonable → Mode A
          2. Else → INSUFFICIENT

        Args:
            candidate: The ranked candidate (already scored)
            patient_input: Patient phenotype bundle
            source_ids: Phenotype NodeIDs
            existing_paths: Paths already computed by the pipeline (if any)

        Returns:
            EvidencePackage with mode + confidence label + payload
        """
        # Try Mode A first
        direct_paths = existing_paths
        if direct_paths is None:
            logger.info(f"[EvidencePanel] No existing_paths for {candidate.disease_name}, running fallback search")
            direct_paths = self._find_direct_paths(source_ids, candidate.disease_id)
        else:
            logger.info(
                f"[EvidencePanel] Using {len(direct_paths)} existing_paths for {candidate.disease_name}, "
                f"scores: {[f'{p.score:.4f} (len={p.length})' for p in direct_paths[:5]]}"
            )

        if direct_paths:
            return self._build_mode_a(direct_paths)

        # No path
        return EvidencePackage(
            mode=EvidenceMode.INSUFFICIENT,
            evidence_label=ConfidenceLabel.INSUFFICIENT,
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

        # Score the paths (find_paths returns unscored paths with score=0.0)
        if all_paths:
            all_paths = self.path_reasoner.score_paths(all_paths, self.kg)

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
            evidence_label=label,
            summary=summary,
            direct_paths=sorted_paths[: self.config.max_direct_paths_displayed],
            min_path_length=min_len,
        )


def create_evidence_panel(
    kg: "KnowledgeGraph",
    path_reasoner: Optional[PathReasoner] = None,
    config: Optional[EvidencePanelConfig] = None,
) -> EvidencePanel:
    """Factory function for EvidencePanel."""
    return EvidencePanel(kg=kg, path_reasoner=path_reasoner, config=config)
