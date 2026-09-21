"""
Shared scoring primitives for served inference and offline evaluation.
======================================================================
**Migration status: in progress.** `DiagnosisPipeline` composes these functions;
`scripts/evaluate_model.py` does not yet and is migrated later in B-0. Until it
is, two implementations of these formulas still exist, which is the condition
this module is being built to end — not one it has already ended.

Scope, and a boundary that is mechanically enforced. This module covers the
**served inference** and **offline evaluation** paths. It deliberately does not
cover training: `.import-linter.ini` places `src.inference` above `src.training`,
so a training module importing this one would invert the layering. Training keeps
its own implementation, and the correspondence between the two is a property to
be held by equivalence tests rather than by shared code.

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

  - **Lookup** (`sp_index.sp_mean_distances`) returns the *measured quantity* —
    mean hop distance — together with an availability mask. It lives in
    `src/inference/sp_index.py` with the index it reads, because the lookup and
    the structure it searches are one design and were briefly two.
  - **Transform** (`sp_scores_from_distances`) turns that into the score the
    system has historically used.

They are separate because the distance is what the graph actually says, while
`1/(1+d)` is one presentation of it, and because a caller may legitimately want
the distance: it is the quantity a clinician can reason about ("within three
steps"), whereas the score compresses the far end so heavily that most of its
range is spent on the first two steps.

Availability is a mask, not a sentinel — but a narrow one at this stage. It
currently separates only "there was nothing to measure from" (no phenotypes, or
no candidates) from "a distance was produced". The other unavailable states —
no table loaded, target unmapped, no phenotype mapped — are still detected by the
caller before this module is reached, and the legacy wrapper collapses them to
`0.0`. That collapse is lossy in a way that matters, since `0.0` is *below* the
value a genuine "no path found" produces. **The full typed status
(`COMPUTED` / `COMPUTED_PARTIAL` / `NO_TABLE` / `TARGET_UNMAPPED` /
`NO_PHENOTYPE_MAPPED`) belongs to the B-1 analysis record; no caller should read
this Boolean as though it already carried those semantics.**

Dependencies: torch. Import this module lazily from anywhere that must remain
importable without torch.
"""
from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "pool_patient_embeddings",
    "masked_mean_pool",
    "cosine_scores",
    "cosine_score_matrix",
    "sp_scores_from_distances",
    "mix_embedding_and_sp_scores",
    "normalise_cosine_to_unit_interval",
]


# =============================================================================
# Shortest-path lookup table
# =============================================================================
#: The hop bound the producer validates before it writes anything
#: (`scripts/compute_shortest_paths.py` refuses outside this range), and the same
#: range `scripts/audit_sp_reachability.py` holds a sidecar to. Declared here so
#: the serving path cannot come to accept an artifact the evidence path refuses —
#: the split-authority failure is the one this constant exists to prevent.
PRODUCER_HOP_RANGE = (1, 127)

#: What the loader assumes when no sidecar records the ceiling. It is the
#: producer's own CLI default and the value every operator-facing build path in
#: this repository uses, so it is the right number for a real artifact whose
#: sidecar went missing — but it is an *assumption*, and `validate_hop_bound`
#: plus the floor check are what keep it from being a silent one.
ASSUMED_HOP_BOUND = 5


def validate_hop_bound(value: Any, source: str) -> int:
    """The hop bound's domain, in the one place both boundaries can read it.

    **`bool` is excluded by name.** `isinstance(True, int)` is True and `true` is
    valid JSON, so a sidecar carrying `"max_hops": true` would otherwise become a
    ceiling of 1 and an unreachable sentinel of 2.0 — below real recorded
    distances, which reorders candidates rather than merely mis-scoring them.

    **The range is the producer's, not one invented here.** A loader accepting
    what the producer refuses to write and the audit refuses to read is the split
    authority this is for.

    Args:
        value: the candidate bound.
        source: where it came from, for the message — a path, or a description.

    Raises:
        ValueError: naming the source and what was wrong.
    """
    low, high = PRODUCER_HOP_RANGE
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{source} carries max_hops={value!r}, which is not an integer. The "
            "hop bound sets the unreachable sentinel every shortest-path score "
            "is measured against."
        )
    if not low <= value <= high:
        raise ValueError(
            f"{source} declares max_hops={value}, outside the [{low}, {high}] "
            "the producer validates before writing. This bound did not come "
            "from it."
        )
    return value


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

    idx = torch.as_tensor(
        phenotype_indices, dtype=torch.long, device=phenotype_embeddings.device
    )
    idx = idx.clamp(min=0, max=phenotype_embeddings.size(0) - 1)
    return phenotype_embeddings[idx].mean(dim=0)


def masked_mean_pool(embeddings: Tensor, mask: Tensor) -> Tensor:
    """Mean over the valid entries of a padded batch. ``(B, N, H)`` -> ``(B, H)``.

    The offline evaluation path's counterpart to `pool_patient_embeddings`. The two
    are **deliberately not one implementation**: the served path holds an unpadded
    ``(N, H)`` tensor and an index list, and manufacturing a mask and a batch
    dimension for it would cost more clarity than sharing buys. An all-true-mask
    equivalence test binds them instead.

    **This mirrors `Trainer._compute_model_outputs` operation for operation**
    (`src/training/trainer.py:744-751`), because Mode A's purpose is to reproduce
    what that code measures:

      - the mask is cast with ``.float()``, so the output dtype is whatever
        promotion and autocast produce from that and ``embeddings`` — **not** an
        assumed "preserve the input dtype" rule, and no cast is added to force one;
      - the denominator is ``clamp(min=1)``, so an **all-false row yields a zero
        vector** rather than a division by zero. That is behaviour to preserve, not
        to improve: changing it would move Mode A off the control it exists to be.

    **The mask must be boolean.** A validity mask and a same-shaped floating-point
    tensor are otherwise indistinguishable at this boundary, and ``.float()``
    would silently turn the latter into a weighting vector. The guard costs no
    parity: the production dataloader constructs ``phenotype_mask`` as
    ``torch.bool`` (`src/kg/data_loader.py:704-707`) and indexes with it
    (`:912`), which already requires bool.

    Nothing is moved between devices. A mismatch raises from torch rather than
    being silently repaired.
    """
    if mask.dtype != torch.bool:
        raise ValueError(
            f"mask must be a boolean validity mask, not {mask.dtype}. A float mask "
            "would silently become a weighting vector"
        )
    if embeddings.dim() != 3:
        raise ValueError(f"embeddings must be (B, N, H); got {tuple(embeddings.shape)}")
    if mask.dim() != 2:
        raise ValueError(f"mask must be (B, N); got {tuple(mask.shape)}")
    if mask.shape != embeddings.shape[:2]:
        raise ValueError(
            f"mask {tuple(mask.shape)} does not match embeddings "
            f"{tuple(embeddings.shape[:2])}"
        )

    weights = mask.unsqueeze(-1).float()
    summed = (embeddings * weights).sum(dim=1)
    counts = weights.sum(dim=1).clamp(min=1)
    return summed / counts


# =============================================================================
# Embedding similarity
# =============================================================================
def cosine_score_matrix(patient_matrix: Tensor, candidate_matrix: Tensor) -> Tensor:
    """Cosine similarity of every patient against every candidate.

    ``patient_matrix`` is ``(B, H)``, ``candidate_matrix`` is ``(D, H)``; returns
    ``(B, D)`` with values in ``[-1, 1]``.

    Built from ``F.normalize`` and a matrix multiply. There is no custom kernel and
    no new dependency: the batched form is the same two library calls the scalar
    form already made.
    """
    if patient_matrix.dim() != 2:
        raise ValueError(f"patient_matrix must be (B, H); got {tuple(patient_matrix.shape)}")
    if candidate_matrix.dim() != 2:
        raise ValueError(
            f"candidate_matrix must be (D, H); got {tuple(candidate_matrix.shape)}"
        )

    patients = F.normalize(patient_matrix, dim=-1)
    candidates = F.normalize(candidate_matrix, dim=-1)
    return torch.mm(patients, candidates.t())


def cosine_scores(patient_vector: Tensor, candidate_matrix: Tensor) -> Tensor:
    """Cosine similarity of one patient against every candidate. Returns ``(C,)``.

    ``patient_vector`` is ``(H,)``; ``candidate_matrix`` is ``(C, H)``. Values lie
    in ``[-1, 1]``.

    A batch of one, delegating to `cosine_score_matrix`. The served signature is
    unchanged and is pinned by a ``B = 1`` equivalence test.
    """
    return cosine_score_matrix(patient_vector.unsqueeze(0), candidate_matrix).squeeze(0)


def normalise_cosine_to_unit_interval(cosine: Tensor) -> Tensor:
    """Map ``[-1, 1]`` to ``[0, 1]`` via ``(x + 1) / 2``.

    Order-preserving, so it changes no ranking. It exists because the score is
    mixed with a shortest-path term that occupies a different range, and because
    a negative confidence reads badly on a clinical surface.
    """
    return (cosine + 1.0) / 2.0


def sp_scores_from_distances(mean_distances: Tensor) -> Tensor:
    """Convert mean hop distance to the similarity score, ``1 / (1 + d)``.

    With the default ``max_hops = 5`` the computed range is ``[1/7, 1/2]``: never
    0 and never 1. The transform is strongly compressive at the far end — the gap
    between one and two hops is about seven times the gap between five hops and
    unreachable — so nearly all of its discriminating power sits at short
    distances. Prefer the distance itself where a caller needs a quantity to
    reason about.

    The output dtype follows the input's. Precision is therefore decided where the
    distance is measured, not here; `sp_index.sp_mean_distances` produces float64
    for the reason given there.
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
