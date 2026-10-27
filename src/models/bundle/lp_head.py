"""
The link-prediction head: DistMult with tanh, trained by a hinge loss.
=====================================================================
`docs/working/scorer-retraining/README.md` A2, U2, U5, F1.

`score(h, r, t) = tanh(Σ h ⊙ w_r ⊙ t)` with one diagonal vector `w_r` per
directed relation of the graph view; the loss is
`mean(max(0, margin − (s⁺ − s⁻)))` over (positive, negative) pairs
(`up:decoders.py`, `up:node_embedder_model.py:201-211,416-425`). The decision
used for LP accuracy is the raw score > 0, i.e. `sigmoid(raw) > 0.5`
(`up:node_embedder_model.py:246-249`).

Module: src/models/bundle/lp_head.py
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.models.bundle.spec import LPHeadSpec


class DistMultHead(nn.Module):
    def __init__(self, spec: LPHeadSpec, num_relations: int, width: int):
        super().__init__()
        self.spec = spec
        self.relation_weights = nn.Parameter(torch.empty(num_relations, width))
        # `up:node_embedder_model.py:114`.
        nn.init.xavier_uniform_(self.relation_weights, gain=nn.init.calculate_gain("leaky_relu"))

    def raw_score(self, head: Tensor, relation: Tensor, tail: Tensor) -> Tensor:
        return (head * self.relation_weights[relation] * tail).sum(dim=-1)

    def score(self, head: Tensor, relation: Tensor, tail: Tensor) -> Tensor:
        return torch.tanh(self.raw_score(head, relation, tail))

    def hinge(self, positive: Tensor, negative: Tensor) -> Tensor:
        """Mean hinge over aligned (positive, negative) score pairs. Called only
        with at least one pair: a step without a valid pair makes no loss
        (README §3.1, no empty mean)."""
        if positive.numel() == 0:
            raise ValueError("no (positive, negative) pair: a step without a valid LP pair "
                             "makes no loss and no update")
        return (self.spec.margin - (positive - negative)).clamp(min=0).mean()


def corrupt_tails(relations: Tensor, tails: Tensor, generator: torch.Generator
                  ) -> Tuple[Tensor, Tensor]:
    """One negative tail per positive, by the source's rule without its defect
    (README U5, F1): drawn with replacement among the tails of the same
    relation's positives in this batch, never the positive's own tail.

    Returns the indices (into the positives) of the drawn negatives' tails and a
    mask of the positives that have one. A positive whose relation has no other
    tail in the batch gets none and is counted by the caller.
    """
    n = relations.numel()
    chosen = torch.full((n,), -1, dtype=torch.long)
    if n == 0:
        return chosen, torch.zeros(0, dtype=torch.bool)
    order = torch.argsort(relations, stable=True)
    rel_sorted = relations[order]
    uniq, counts = torch.unique_consecutive(rel_sorted, return_counts=True)
    start = 0
    for count in counts.tolist():
        group = order[start:start + count]
        start += count
        g_tails = tails[group]
        for j, pos in enumerate(group.tolist()):
            allowed = group[g_tails != g_tails[j]]
            if allowed.numel() == 0:
                continue
            pick = torch.randint(allowed.numel(), (1,), generator=generator).item()
            chosen[pos] = allowed[pick]
    return chosen, chosen >= 0


__all__ = ["DistMultHead", "corrupt_tails"]
