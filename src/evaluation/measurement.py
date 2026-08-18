"""
Ranking for offline measurement — two streams, deliberately.
===========================================================
Module: src/evaluation/measurement.py

Mode A has to satisfy two requirements that one ranking cannot:

  - **reproduce the frozen evaluator exactly**, so its aggregate number can be
    compared against the historical one and the harness itself calibrated;
  - **rank deterministically**, so numbers from modes A, B, C and D can be
    compared with each other.

These conflict whenever scores tie. The frozen evaluator's order at a tie is
whatever `Tensor.sort` produced on that machine; a comparison across modes needs
an order that does not depend on which machine, which batch composition or which
input ordering produced it. So there are two streams, named for what each is for,
and neither is used for the other's purpose.

What this module does **not** do: load data, run a model, or decide what to do
about a ground truth that is absent from the candidate set. It ranks, and it maps
identifiers. The Mode A driver that wires a dataloader and a model to these
functions follows separately.

Dependencies: torch.
"""
from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor

__all__ = [
    "to_global_disease_ids",
    "canonical_ranking",
    "legacy_ranking",
    "ranks_of_truth",
]


def to_global_disease_ids(original_disease_indices: Tensor, local_ids: Tensor) -> Tensor:
    """Translate subgraph-local disease indices into global knowledge-graph ids.

    ``original_disease_indices`` is the dataloader's ``original_indices["disease"]``
    — it is *already* the local-to-global direction: built as
    ``torch.tensor(sorted(nodes))`` and indexed by local position
    (`src/kg/data_loader.py:336-342`), so entry *i* holds the global index of local
    node *i*. The sibling ``node_mapping`` is the **opposite** direction, a
    global-to-local dict; inverting it here would be work with an extra chance of
    getting the direction backwards.

    This exists as a named function rather than a bare index because a silent
    direction error produces plausible ids and therefore plausible, wrong metrics.

    Local indices are only comparable within the batch that produced them, so
    everything persisted or aggregated must be global.
    """
    if original_disease_indices.dim() != 1:
        raise ValueError(
            f"original_disease_indices must be 1-D; got {tuple(original_disease_indices.shape)}"
        )
    if local_ids.numel() and int(local_ids.max()) >= original_disease_indices.numel():
        raise ValueError(
            f"local id {int(local_ids.max())} is outside the subgraph's "
            f"{original_disease_indices.numel()} disease nodes"
        )
    if local_ids.numel() and int(local_ids.min()) < 0:
        raise ValueError("local ids must be non-negative")

    return original_disease_indices.to(local_ids.device)[local_ids.long()]


def canonical_ranking(scores: Tensor, global_disease_ids: Tensor) -> Tensor:
    """Rank by score descending, breaking ties by ascending global disease id.

    ``scores`` is ``(B, D)``, ``global_disease_ids`` is ``(D,)``; returns
    ``(B, D)`` of global ids, best first.

    **The authoritative stream.** Its order depends on the scores and the global
    ids and on nothing else — not on the order candidates arrived in, not on batch
    composition, not on the machine. That is what makes a number from Mode A
    comparable with one from Mode C.

    The tie rule is obtained from PyTorch's own stable sort rather than a custom
    comparator: order the candidates by global id first, then sort by score with
    ``stable=True``, and stability preserves ascending-id order inside every
    equal-score group.
    """
    if scores.dim() != 2:
        raise ValueError(f"scores must be (B, D); got {tuple(scores.shape)}")
    if global_disease_ids.dim() != 1:
        raise ValueError(
            f"global_disease_ids must be (D,); got {tuple(global_disease_ids.shape)}"
        )
    if global_disease_ids.numel() != scores.size(1):
        raise ValueError(
            f"{global_disease_ids.numel()} global ids for {scores.size(1)} score columns"
        )
    if global_disease_ids.unique().numel() != global_disease_ids.numel():
        raise ValueError("global_disease_ids contains duplicates; the ranking would be ambiguous")

    id_order = torch.argsort(global_disease_ids)
    ordered_ids = global_disease_ids[id_order]
    ordered_scores = scores[:, id_order.to(scores.device)]

    rank_order = torch.argsort(ordered_scores, dim=-1, descending=True, stable=True)
    return ordered_ids.to(rank_order.device)[rank_order]


def legacy_ranking(scores: Tensor, global_disease_ids: Tensor) -> Tensor:
    """Reproduce the frozen evaluator's order. ``(B, D)`` of global ids.

    `scripts/evaluate_model.py:295` sorts with ``scores.sort(dim=-1,
    descending=True)`` over the subgraph-local columns, so this calls the same
    thing on the same columns and translates the result. Tie behaviour is
    therefore whatever that call does — **which is the point**: this stream exists
    to match the historical number, not to be well-defined.

    **Used only for the truncated legacy parity metric.** Anything compared across
    modes uses `canonical_ranking`.
    """
    if scores.dim() != 2:
        raise ValueError(f"scores must be (B, D); got {tuple(scores.shape)}")
    if global_disease_ids.numel() != scores.size(1):
        raise ValueError(
            f"{global_disease_ids.numel()} global ids for {scores.size(1)} score columns"
        )

    _, local_order = scores.sort(dim=-1, descending=True)
    return global_disease_ids.to(local_order.device)[local_order]


def ranks_of_truth(ranked_global_ids: Tensor, truth_global_ids: Tensor) -> List[Optional[int]]:
    """1-based rank of each row's ground truth, or ``None`` where it is absent.

    ``ranked_global_ids`` is ``(B, D)``; ``truth_global_ids`` is ``(B,)``.

    **``None`` rather than a sentinel rank.** A ground truth outside the candidate
    set has no rank, and encoding that as a large integer would let it flow into a
    mean as though it were a measurement. `RankingMetrics.compute_from_ranks`
    refuses to invent a value for an empty cohort for the same reason; this is the
    same refusal one level up. The caller counts and reports absences separately.

    A Python list, not a tensor, because the value is genuinely optional and
    because the consumer accumulates across batches into a list anyway.
    """
    if ranked_global_ids.dim() != 2:
        raise ValueError(f"ranked_global_ids must be (B, D); got {tuple(ranked_global_ids.shape)}")
    if truth_global_ids.dim() != 1:
        raise ValueError(f"truth_global_ids must be (B,); got {tuple(truth_global_ids.shape)}")
    if truth_global_ids.numel() != ranked_global_ids.size(0):
        raise ValueError(
            f"{truth_global_ids.numel()} truths for {ranked_global_ids.size(0)} ranked rows"
        )

    truth = truth_global_ids.to(ranked_global_ids.device).unsqueeze(-1)
    matches = ranked_global_ids == truth
    found = matches.any(dim=-1)
    # argmax over a boolean row gives the first True, which is the best rank a
    # duplicated id could hold. Duplicates are rejected upstream, so this is the
    # only position.
    positions = matches.to(torch.uint8).argmax(dim=-1)

    return [
        int(position) + 1 if bool(is_found) else None
        for position, is_found in zip(positions.tolist(), found.tolist())
    ]
