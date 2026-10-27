"""
The one task-loss module: normalised squared distance, NCA, multi-similarity.
============================================================================
`docs/working/scorer-retraining/README.md` A7–A9, C8, U10, F7, F11 and §3.2.

No second copy of these losses exists for another entry; pytorch-metric-learning
is not a dependency. The LP hinge is the LP head's (`src.models.bundle.lp_head`).

**Distance (U10).** `‖â − b̂‖²` with `x̂ = x / max(‖x‖₂, ε)`, ε from the spec
(1e-12), computed as `‖â‖² + ‖b̂‖² − 2 â·b̂` clamped at 0. A zero vector stays
zero, so its distance to a unit vector is 1, not the 2 the cosine identity
would give.

**NCA (A7, A9).** Per query, `−log Σ_{pos} softmax(−scale · d)` over its
candidates, in log space, averaged over queries. Candidates a query may not
see (its own row in patients-like-me) are excluded before the softmax. A query
without a positive among its candidates is refused (F7): the caller binds the
truths into the candidate set, and nothing is dropped silently.

**Multi-similarity (C8).** The source's criterion
(`up:utils/loss_utils.py:92-151`): per anchor, pair mining with `margin`, then
`1/α·log(1 + Σ exp(−α(s⁺ − base))) + 1/β·log(1 + Σ exp(β(s⁻ − base)))`,
averaged over anchors. As in the source, mining applies only when it keeps at
least one positive and one negative; otherwise the anchor's unmined pairs are
used. An anchor without a positive or without a negative adds no term and is
counted; a batch with no anchor left makes no loss (`value is None`), never an
invented zero.

Module: src/models/tasks/losses.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class LossResult:
    """A loss and what it was computed over."""

    #: The scalar loss, or None when no unit contributed.
    value: Optional[Tensor]
    #: Units (queries, anchors) that contributed.
    units: int
    #: Units that did not, by reason.
    skipped: Dict[str, int] = field(default_factory=dict)


def normalised_sq_distance(query: Tensor, reference: Tensor, eps: float) -> Tensor:
    """`[B, D] × [M, D] → [B, M]` squared distances of L2-normalised vectors."""
    q = F.normalize(query, p=2, dim=-1, eps=eps)
    r = F.normalize(reference, p=2, dim=-1, eps=eps)
    d = (q * q).sum(-1, keepdim=True) + (r * r).sum(-1).unsqueeze(0) - 2.0 * q @ r.T
    return d.clamp(min=0.0)


def nca_loss(query: Tensor, reference: Tensor, positive: Tensor, *, scale: float, eps: float,
             exclude: Optional[Tensor] = None) -> LossResult:
    """NCA over `[B, M]` candidates; `positive` and `exclude` are `[B, M]` bool."""
    if query.size(0) == 0:
        return LossResult(None, 0, {})
    logits = -scale * normalised_sq_distance(query, reference, eps)
    visible = torch.ones_like(positive) if exclude is None else ~exclude
    usable = positive & visible
    lacking = ~usable.any(dim=1)
    if bool(lacking.any()):
        raise ValueError(f"{int(lacking.sum())} queries have no positive among their candidates; "
                         "the truths are bound into the candidate set before the loss (F7)")
    neg_inf = torch.finfo(logits.dtype).min
    log_all = torch.logsumexp(logits.masked_fill(~visible, neg_inf), dim=1)
    log_pos = torch.logsumexp(logits.masked_fill(~usable, neg_inf), dim=1)
    return LossResult((log_all - log_pos).mean(), int(query.size(0)), {})


def multi_similarity_loss(similarity: Tensor, positive: Tensor, valid: Tensor, *, alpha: float,
                          beta: float, base: float, margin: float) -> LossResult:
    """The source's multi-similarity criterion over `[B, M]` similarities;
    `positive` marks the truths, `valid` the real (unpadded) candidates."""
    terms = []
    skipped = {"no_positive": 0, "no_negative": 0}
    for i in range(similarity.size(0)):
        pos = similarity[i][positive[i] & valid[i]]
        neg = similarity[i][~positive[i] & valid[i]]
        if pos.numel() == 0:
            skipped["no_positive"] += 1
            continue
        if neg.numel() == 0:
            skipped["no_negative"] += 1
            continue
        keep_neg = (neg + margin) > pos.min()
        keep_pos = (pos - margin) < neg.max()
        if bool(keep_neg.any()) and bool(keep_pos.any()):
            pos, neg = pos[keep_pos], neg[keep_neg]
        zero = pos.new_zeros(1)
        pos_term = torch.logsumexp(torch.cat([zero, -alpha * (pos - base)]), 0) / alpha
        neg_term = torch.logsumexp(torch.cat([zero, beta * (neg - base)]), 0) / beta
        terms.append(pos_term + neg_term)
    value = torch.stack(terms).mean() if terms else None
    return LossResult(value, len(terms), {k: v for k, v in skipped.items() if v})


__all__ = ["LossResult", "multi_similarity_loss", "nca_loss", "normalised_sq_distance"]
