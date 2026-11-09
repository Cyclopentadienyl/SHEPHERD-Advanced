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

**The one LP core (README §3.2, F3, F4).** Fixed pairs (`fixed_lp_pairs`: every
edge once, against one negative drawn once by U5's rule) and their scoring
(`score_lp_pairs`: the head's raw scores over full-view node vectors, and the
whole-set metrics) live here, beside the negative rule, so base validation and
any LP measurement share one pair rule and one scoring.

Module: src/models/bundle/lp_head.py
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from src.models.bundle.metrics import lp_metrics
from src.models.bundle.spec import LPHeadSpec
from src.models.tasks.scoring import full_view_vectors


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


#: The fixed-pairs rule seeds its torch generator with `seed` plus this offset
#: (training rules 1 pin the base's validation pairs, so it never changes).
_PAIR_SEED_OFFSET = 7919
#: A pairs table's columns, in the order `lp_pairs_digest` reads each row.
LP_PAIR_COLUMNS = ("head", "relation", "tail", "negative_tail")


@dataclass
class LPPairs:
    """Fixed LP pairs: the kept edges `head -relation-> tail`, each with its
    one corrupted tail, in the rule's order (CPU, int64); the rule's counts;
    and the digest of the pairs' bytes (`lp_pairs_digest`)."""

    head: Tensor
    relation: Tensor
    tail: Tensor
    negative_tail: Tensor
    #: `edges` given, `without_negative` dropped for having no negative,
    #: `pairs` kept (`edges - without_negative`).
    counts: Dict[str, int]
    digest: str


def lp_pairs_digest(head: Tensor, relation: Tensor, tail: Tensor, negative_tail: Tensor) -> str:
    """SHA-256 over the exact pair bytes: a JSON header naming the layout, then
    the pairs as one row-major `[pairs, 4]` table of little-endian int64, a row
    per pair, columns `LP_PAIR_COLUMNS`. Any changed id, a reordered pair or a
    pair added or dropped changes it."""
    h = hashlib.sha256()
    h.update(json.dumps({"columns": list(LP_PAIR_COLUMNS), "dtype": "<i8", "order": "rows"},
                        sort_keys=True).encode())
    table = torch.stack([head, relation, tail, negative_tail], dim=1).to(torch.int64).cpu()
    h.update(np.ascontiguousarray(table.numpy(), dtype="<i8").tobytes())
    return h.hexdigest()


def fixed_lp_pairs(edges: Tensor, relations: Tensor, *, seed: int, batch_size: int) -> LPPairs:
    """Every edge once, as an LP pair against one negative tail drawn once
    (README F3; U5 and F1 through `corrupt_tails`).

    The rule, on CPU tensors: a torch generator seeded by `seed + 7919`
    permutes the edges (`edges` `[2, E]`, source then target, with
    `relations`); the permuted edges are cut into consecutive batches of
    `batch_size`; batch by batch, `corrupt_tails` draws each edge's negative
    from the same generator, among the tails of its relation's edges in that
    batch. An edge whose relation has no other tail in its batch has no
    negative: it is dropped and counted (`without_negative`), never given an
    invented one. The pairs keep the permuted order.

    Base validation takes its pairs here (the split's val edges, the split's
    seed, `BASE_FIXED`'s batch size), and training rules 1 pin those pairs, so
    the generator's consumption (one permutation, then each batch's draws in
    order) never changes.
    """
    if int(batch_size) <= 0:
        raise ValueError(f"batch_size must be positive, not {batch_size}")
    batch_size = int(batch_size)
    generator = torch.Generator().manual_seed(int(seed) + _PAIR_SEED_OFFSET)
    order = torch.randperm(edges.size(1), generator=generator)
    heads, tails, rels = edges[0][order], edges[1][order], relations[order]
    keep_h, keep_r, keep_t, neg = [], [], [], []
    without = 0
    for start in range(0, heads.numel(), batch_size):
        h = heads[start:start + batch_size]
        t = tails[start:start + batch_size]
        r = rels[start:start + batch_size]
        chosen, ok = corrupt_tails(r, t, generator)
        without += int((~ok).sum())
        keep_h.append(h[ok])
        keep_r.append(r[ok])
        keep_t.append(t[ok])
        neg.append(t[chosen[ok]])
    cat = (lambda xs: torch.cat(xs) if xs else torch.empty(0, dtype=torch.long))
    head, relation, tail, negative_tail = cat(keep_h), cat(keep_r), cat(keep_t), cat(neg)
    counts = {"edges": int(heads.numel()), "without_negative": without,
              "pairs": int(head.numel())}
    return LPPairs(head, relation, tail, negative_tail, counts,
                   lp_pairs_digest(head, relation, tail, negative_tail))


@dataclass
class LPScores:
    """Each pair's positive and negative raw (pre-tanh) score, in pair order, on
    the CPU, and the whole-set metrics over them."""

    positive: Tensor
    negative: Tensor
    #: `lp_metrics`' (accuracy at raw > 0, ROC-AUC, AP, the positives' and
    #: negatives' counts) and `lp_loss`, the head's hinge over the tanh scores.
    metrics: Dict[str, Any]


@torch.no_grad()
def score_lp_pairs(bundle: Any, pairs: LPPairs, edges: Tensor, relations: Tensor,
                   device: torch.device, *, chunk_size: int = 65536,
                   node_type: Optional[Tensor] = None) -> LPScores:
    """Score fixed pairs with `bundle`'s LP head, in evaluation mode.

    Node vectors are the full-view rule's (§2.8), message passing over `edges`
    with their `relations` and nothing else: the pairs are supervision, never
    message passing (G4). They come from `full_view_vectors`, the one
    full-view entry that validation, measurement and serving share, so this
    scoring has no forward of its own; `chunk_size` and `node_type` (every
    view node's type id, in the view's node order) are that entry's.

    Every positive is scored before every negative, each side in pair order.
    """
    z = full_view_vectors(bundle, edges, relations, device, chunk_size, node_type=node_type)
    head = z[pairs.head.to(device)]
    relation = pairs.relation.to(device)
    positive = bundle.lp_head.raw_score(head, relation, z[pairs.tail.to(device)])
    negative = bundle.lp_head.raw_score(head, relation, z[pairs.negative_tail.to(device)])
    metrics = lp_metrics(positive.cpu().tolist(), negative.cpu().tolist())
    metrics["lp_loss"] = float(bundle.lp_head.hinge(torch.tanh(positive), torch.tanh(negative)))
    return LPScores(positive.cpu(), negative.cpu(), metrics)


__all__ = ["DistMultHead", "LPPairs", "LPScores", "LP_PAIR_COLUMNS", "corrupt_tails",
           "fixed_lp_pairs", "lp_pairs_digest", "score_lp_pairs"]
