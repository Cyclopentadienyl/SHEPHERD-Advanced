"""
Scoring primitives — the single authority for how a candidate is scored.
========================================================================
Every surface that produces or reproduces a disease score composes these
functions. Nothing re-implements the arithmetic.

Module: src/inference/scoring.py

Why this exists. The score a clinician sees and the score an offline evaluation
reports were computed by two separate implementations of the same formulas
(`DiagnosisPipeline._calculate_*` and `scripts/evaluate_model.py`). Two
implementations of one formula drift, and when they drift the evaluation stops
describing the thing being evaluated. The fix is not "be careful" — it is to
have one place the arithmetic lives.

Shape. Every primitive is **batched over candidates**: it takes a candidate
matrix or a candidate index vector and returns one value per candidate. The
pipeline currently calls them with a single candidate, which is a batch of one;
scoring the full disease universe is the same call with a longer vector. Batch
shape is therefore not an optimisation to retrofit later, it is the interface.

Separation of concerns, deliberately:

  - **Lookup** (`sp_mean_distances`) returns the *measured quantity* — mean hop
    distance — together with an availability mask.
  - **Transform** (`sp_scores_from_distances`) turns that into the score the
    system has historically used.

They are separate because the distance is what the graph actually says, while
`1/(1+d)` is one presentation of it, and because a caller may legitimately want
the distance: it is the quantity a clinician can reason about ("within three
steps"), whereas the score compresses the far end so heavily that most of its
range is spent on the first two steps.

Availability is a mask, not a sentinel. The legacy scalar wrapper collapses
unavailable to `0.0` to preserve existing behaviour, but that collapse is lossy
in a way that matters — `0.0` is *below* the value a genuine "no path found"
produces — so the information is kept here even though today's caller discards
it.

Dependencies: torch. Import this module lazily from anywhere that must remain
importable without torch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "SPLookup",
    "pool_patient_embeddings",
    "cosine_scores",
    "sp_mean_distances",
    "sp_scores_from_distances",
    "mix_embedding_and_sp_scores",
    "normalise_cosine_to_unit_interval",
]


# =============================================================================
# Shortest-path lookup table
# =============================================================================
@dataclass(frozen=True)
class SPLookup:
    """The CSR-style shortest-path table, as `_load_shortest_paths` builds it.

    ``offsets`` maps a phenotype index to its ``(start, end)`` slice of the three
    parallel tensors. A pair absent from a phenotype's slice means "no path found
    within ``max_hops``".

    Exactly one row exists per (phenotype, target, target_type): the offline BFS
    records a node the first time it is reached, which is its minimum distance
    (`scripts/compute_shortest_paths.py:79-89`). Lookup may therefore take the
    first match without ambiguity.
    """

    target: Tensor
    target_type: Tensor
    distance: Tensor
    offsets: Dict[int, Tuple[int, int]]
    max_hops: int

    @property
    def unreachable_distance(self) -> float:
        """What a phenotype contributes when it cannot reach the target.

        One more than the search bound, so an unreachable phenotype penalises the
        mean without erasing the contribution of the ones that did connect.
        """
        return float(self.max_hops + 1)


# =============================================================================
# Patient representation
# =============================================================================
def pool_patient_embeddings(
    phenotype_embeddings: Tensor,
    phenotype_indices: Sequence[int],
) -> Tensor:
    """Unweighted mean of the given phenotype embeddings. Returns ``(H,)``.

    This is the patient representation the deployed checkpoint's training
    objective optimised (`src/training/trainer.py:744-751`). It is *not* the
    reference paper's, which uses a transformer encoder and attention-weighted
    aggregation; see `docs/DISEASE_SCORER_POLICY.md`.

    Indices are clamped into range, matching the behaviour this replaces.
    """
    if len(phenotype_indices) == 0:
        raise ValueError("pool_patient_embeddings requires at least one phenotype index")

    idx = torch.as_tensor(list(phenotype_indices), dtype=torch.long)
    idx = idx.clamp(min=0, max=phenotype_embeddings.size(0) - 1)
    return phenotype_embeddings[idx].mean(dim=0)


# =============================================================================
# Embedding similarity
# =============================================================================
def cosine_scores(patient_vector: Tensor, candidate_matrix: Tensor) -> Tensor:
    """Cosine similarity of one patient against every candidate. Returns ``(C,)``.

    ``patient_vector`` is ``(H,)``; ``candidate_matrix`` is ``(C, H)``. Values lie
    in ``[-1, 1]``.
    """
    patient = F.normalize(patient_vector.unsqueeze(0), dim=-1)
    candidates = F.normalize(candidate_matrix, dim=-1)
    return torch.mm(patient, candidates.t()).squeeze(0)


def normalise_cosine_to_unit_interval(cosine: Tensor) -> Tensor:
    """Map ``[-1, 1]`` to ``[0, 1]`` via ``(x + 1) / 2``.

    Order-preserving, so it changes no ranking. It exists because the score is
    mixed with a shortest-path term that occupies a different range, and because
    a negative confidence reads badly on a clinical surface.
    """
    return (cosine + 1.0) / 2.0


# =============================================================================
# Shortest-path distance
# =============================================================================
def sp_mean_distances(
    lookup: SPLookup,
    phenotype_indices: Sequence[int],
    target_indices: Sequence[int],
    target_type_idx: int,
) -> Tuple[Tensor, Tensor]:
    """Mean hop distance from the patient's phenotypes to each candidate.

    Returns ``(mean_distance, available)``, both ``(C,)``. ``available`` is False
    for a candidate whose distance could not be computed at all — distinct from a
    candidate that is simply far away, which is available with a large distance.

    A phenotype with no path to a candidate contributes
    ``lookup.unreachable_distance`` rather than being dropped, so a candidate all
    of whose phenotypes are unreachable is still *computed*: it has a real value,
    the largest one.
    """
    n_candidates = len(target_indices)
    distances = torch.zeros(n_candidates, dtype=torch.float32)
    available = torch.zeros(n_candidates, dtype=torch.bool)

    if not phenotype_indices or n_candidates == 0:
        return distances, available

    unreachable = lookup.unreachable_distance

    for position, target_idx in enumerate(target_indices):
        total = 0.0
        for ph_idx in phenotype_indices:
            offsets = lookup.offsets.get(ph_idx)
            if offsets is None:
                total += unreachable
                continue
            start, end = offsets
            target_slice = lookup.target[start:end]
            type_slice = lookup.target_type[start:end]
            distance_slice = lookup.distance[start:end]
            match = (target_slice == target_idx) & (type_slice == target_type_idx)
            hits = distance_slice[match]
            total += float(hits[0]) if len(hits) > 0 else unreachable
        distances[position] = total / len(phenotype_indices)
        available[position] = True

    return distances, available


def sp_scores_from_distances(mean_distances: Tensor) -> Tensor:
    """Convert mean hop distance to the similarity score, ``1 / (1 + d)``.

    With the default ``max_hops = 5`` the computed range is ``[1/7, 1/2]``: never
    0 and never 1. The transform is strongly compressive at the far end — the gap
    between one and two hops is about seven times the gap between five hops and
    unreachable — so nearly all of its discriminating power sits at short
    distances. Prefer the distance itself where a caller needs a quantity to
    reason about.
    """
    return 1.0 / (1.0 + mean_distances)


# =============================================================================
# Mixture
# =============================================================================
def mix_embedding_and_sp_scores(
    embedding_scores: Tensor,
    sp_scores: Tensor,
    eta: float,
) -> Tensor:
    """``eta * embedding + (1 - eta) * sp``, elementwise over candidates.

    **This mixture is under review and is not the target design.** The reference
    paper applies it to candidate *gene* scoring, over a clinician-supplied short
    list; disease ranking in that paper uses embedding similarity alone. The
    approved target removes it from disease ranking entirely — see
    `docs/DISEASE_SCORER_POLICY.md`. It is kept because the current system uses
    it, and because the offline comparison that justifies removing it has to be
    able to compute it.

    Note also that `eta` is not the effective weight: the two terms occupy
    different ranges, so the nominal split overstates the shortest-path term's
    influence.
    """
    return eta * embedding_scores + (1.0 - eta) * sp_scores
