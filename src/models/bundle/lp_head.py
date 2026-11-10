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
any LP measurement share one pair rule and one scoring. So do the two byte
encodings of their results: a pairs table (`lp_pairs_table`, a KG edge set's
``pairs.bin``) and a scores table (`lp_scores_table`: each pair's actual raw
scores keyed to its row of the pairs table, with the row's state; a KG
link-prediction run's kept scores, research comparison F4).

Module: src/models/bundle/lp_head.py
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

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
#: (training rules 1 and 2 pin the base's validation pairs, so it never changes).
_PAIR_SEED_OFFSET = 7919
#: The name of `fixed_lp_pairs`' rule, as a record states which rule made a
#: set of pairs (with its `seed` and `batch_size`): another rule, seed or batch
#: size is other pairs.
LP_PAIR_RULE = "fixed-lp-pairs/v1"
#: A pairs table's columns, in the order `lp_pairs_digest` reads each row.
LP_PAIR_COLUMNS = ("head", "relation", "tail", "negative_tail")
#: How a pairs table is laid out as bytes (`lp_pairs_table`): one row per
#: pair, the columns `LP_PAIR_COLUMNS`, each a little-endian int64.
LP_PAIRS_LAYOUT: Dict[str, Any] = {"columns": list(LP_PAIR_COLUMNS), "dtype": "<i8",
                                   "order": "rows"}
_ROW_BYTES = 8 * len(LP_PAIR_COLUMNS)


@dataclass
class LPPairs:
    """Fixed LP pairs: the kept edges `head -relation-> tail`, each with its
    one corrupted tail, in the rule's order (CPU, int64); the rule's counts;
    the digest of the pairs' bytes (`lp_pairs_digest`); and how many pairs
    each batch kept, in batch order (`batch_pairs`)."""

    head: Tensor
    relation: Tensor
    tail: Tensor
    negative_tail: Tensor
    #: `edges` given, `without_negative` dropped for having no negative,
    #: `pairs` kept (`edges - without_negative`).
    counts: Dict[str, int]
    digest: str
    #: The pairs each batch kept, one count per batch of the permuted edges,
    #: in order: the batch boundaries in the pairs' own rows (batch `k`'s
    #: pairs are the `batch_pairs[k]` rows after the earlier batches'), and,
    #: against each batch's edge count, the edges it left out. They sum to
    #: ``counts["pairs"]``. Kept apart from `counts`, which base validation
    #: records as it always has.
    batch_pairs: Tuple[int, ...]


def lp_pairs_table(head: Tensor, relation: Tensor, tail: Tensor, negative_tail: Tensor) -> bytes:
    """The pairs as bytes (`LP_PAIRS_LAYOUT`): one row-major `[pairs, 4]` table
    of little-endian int64, a row per pair in the pairs' order, columns
    `LP_PAIR_COLUMNS`. What `lp_pairs_digest` hashes after its header, and
    what a kept pairs file holds, so the file and the digest are of one
    encoding."""
    table = torch.stack([head, relation, tail, negative_tail], dim=1).to(torch.int64).cpu()
    return np.ascontiguousarray(table.numpy(), dtype="<i8").tobytes()


def lp_pairs_from_table(data: bytes) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """The pairs `lp_pairs_table` encoded as `data`: head, relation, tail and
    negative tail, each a CPU int64 tensor in row order. Parses the bytes
    given and reads no file.

    Raises:
        ValueError: `data` is not a whole number of rows.
    """
    if len(data) % _ROW_BYTES:
        raise ValueError(f"{len(data)} bytes are not a whole number of {_ROW_BYTES}-byte pair "
                         f"rows ({', '.join(LP_PAIR_COLUMNS)}, little-endian int64)")
    table = np.frombuffer(data, dtype="<i8").reshape(-1, len(LP_PAIR_COLUMNS))
    columns = torch.from_numpy(table.astype(np.int64)).t().contiguous()
    return columns[0], columns[1], columns[2], columns[3]


def lp_pairs_digest(head: Tensor, relation: Tensor, tail: Tensor, negative_tail: Tensor) -> str:
    """SHA-256 over the exact pair bytes: a JSON header naming the layout
    (`LP_PAIRS_LAYOUT`), then the pairs' table (`lp_pairs_table`). Any changed
    id, a reordered pair or a pair added or dropped changes it."""
    h = hashlib.sha256()
    h.update(json.dumps(LP_PAIRS_LAYOUT, sort_keys=True).encode())
    h.update(lp_pairs_table(head, relation, tail, negative_tail))
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
    invented one. The pairs keep the permuted order, and each batch's count
    of kept pairs is recorded (`batch_pairs`), so the batch boundaries can be
    found in the pairs themselves.

    Base validation takes its pairs here (the split's val edges, the split's
    seed, `BASE_FIXED`'s batch size), and training rules 1 and 2 pin those pairs, so
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
    batch_pairs = []
    for start in range(0, heads.numel(), batch_size):
        h = heads[start:start + batch_size]
        t = tails[start:start + batch_size]
        r = rels[start:start + batch_size]
        chosen, ok = corrupt_tails(r, t, generator)
        without += int((~ok).sum())
        batch_pairs.append(int(ok.sum()))
        keep_h.append(h[ok])
        keep_r.append(r[ok])
        keep_t.append(t[ok])
        neg.append(t[chosen[ok]])
    cat = (lambda xs: torch.cat(xs) if xs else torch.empty(0, dtype=torch.long))
    head, relation, tail, negative_tail = cat(keep_h), cat(keep_r), cat(keep_t), cat(neg)
    counts = {"edges": int(heads.numel()), "without_negative": without,
              "pairs": int(head.numel())}
    return LPPairs(head, relation, tail, negative_tail, counts,
                   lp_pairs_digest(head, relation, tail, negative_tail), tuple(batch_pairs))


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
                   node_type: Optional[Tensor] = None,
                   pair_chunk: Optional[int] = None) -> LPScores:
    """Score fixed pairs with `bundle`'s LP head, in evaluation mode.

    Node vectors are the full-view rule's (§2.8), message passing over `edges`
    with their `relations` and nothing else: the pairs are supervision, never
    message passing (G4). They come from `full_view_vectors`, the one
    full-view entry that validation, measurement and serving share, so this
    scoring has no forward of its own; `chunk_size` and `node_type` (every
    view node's type id, in the view's node order) are that entry's.

    `pair_chunk` is how many pairs are scored at once (an execution
    parameter, so the pairs' vectors need not all be gathered at once): None
    scores them all in one pass, as base validation does; a measurement
    records the chunk it used. Every positive is scored before every
    negative, each side in pair order, and the metrics are the whole set's.
    """
    if pair_chunk is not None and int(pair_chunk) <= 0:
        raise ValueError(f"pair_chunk must be positive, not {pair_chunk}")
    z = full_view_vectors(bundle, edges, relations, device, chunk_size, node_type=node_type)
    n = int(pairs.head.numel())
    step = max(n, 1) if pair_chunk is None else int(pair_chunk)
    positives, negatives = [], []
    for start in range(0, max(n, 1), step):
        rows = slice(start, start + step)
        head = z[pairs.head[rows].to(device)]
        relation = pairs.relation[rows].to(device)
        positives.append(bundle.lp_head.raw_score(head, relation, z[pairs.tail[rows].to(device)]))
        negatives.append(bundle.lp_head.raw_score(head, relation,
                                                  z[pairs.negative_tail[rows].to(device)]))
    positive = positives[0] if len(positives) == 1 else torch.cat(positives)
    negative = negatives[0] if len(negatives) == 1 else torch.cat(negatives)
    metrics = lp_metrics(positive.cpu().tolist(), negative.cpu().tolist())
    metrics["lp_loss"] = float(bundle.lp_head.hinge(torch.tanh(positive), torch.tanh(negative)))
    return LPScores(positive.cpu(), negative.cpu(), metrics)


#: A scores table's columns (`lp_scores_table`), in each row's order: the row
#: of the pairs table the line scores (its index, from 0), the positive's and
#: the negative's raw (pre-tanh) score as computed, and the row's state.
LP_SCORE_COLUMNS = ("pair_row", "positive_raw", "negative_raw", "state")
#: A row's state, stored as its index here: both raw scores are finite, or at
#: least one is not (NaN or infinite), so no decision or rank is defined for it
#: and no whole-set metric can be computed over a set holding it.
LP_SCORE_STATES = ("scored", "not_finite")
#: The raw-score dtypes a scores table stores exactly, by the dtype the scores
#: were computed in: float64 as itself, every narrower float widened to float32
#: (exact for float32, float16 and bfloat16).
_SCORE_WIDTHS = {torch.float64: "<f8", torch.float32: "<f4", torch.float16: "<f4",
                 torch.bfloat16: "<f4"}


def lp_scores_layout(dtype: torch.dtype) -> Dict[str, Any]:
    """How a scores table of raw scores computed in `dtype` is laid out as
    bytes: one packed row per pair, `LP_SCORE_COLUMNS` as little-endian int64
    (the pairs-table row), the raw scores at `_SCORE_WIDTHS[dtype]` (exactly
    the values computed) and the state as one unsigned byte, an index into
    `LP_SCORE_STATES` (listed with the layout).

    Raises:
        ValueError: `dtype` is not a float dtype a table stores exactly.
    """
    if dtype not in _SCORE_WIDTHS:
        raise ValueError(f"raw scores of dtype {dtype} are not stored by a scores table")
    width = _SCORE_WIDTHS[dtype]
    return {"columns": list(LP_SCORE_COLUMNS), "dtypes": ["<i8", width, width, "|u1"],
            "order": "rows", "states": list(LP_SCORE_STATES)}


def _scores_dtype(layout: Dict[str, Any]) -> np.dtype:
    """The packed row dtype of a scores table's `layout`, which must name
    `LP_SCORE_COLUMNS` and `LP_SCORE_STATES` in order, rows first."""
    if not (isinstance(layout, dict) and layout.get("columns") == list(LP_SCORE_COLUMNS)
            and layout.get("order") == "rows" and layout.get("states") == list(LP_SCORE_STATES)
            and isinstance(layout.get("dtypes"), list)
            and len(layout["dtypes"]) == len(LP_SCORE_COLUMNS)
            and layout["dtypes"][0] == "<i8" and layout["dtypes"][3] == "|u1"
            and layout["dtypes"][1] == layout["dtypes"][2]
            and layout["dtypes"][1] in set(_SCORE_WIDTHS.values())):
        raise ValueError(f"{layout!r} is not a scores table layout this build reads")
    return np.dtype(list(zip(LP_SCORE_COLUMNS, layout["dtypes"], strict=True)))


def lp_score_states(positive: Tensor, negative: Tensor) -> Tensor:
    """Each pair's state (`LP_SCORE_STATES`) as a uint8 index: 0 when both its
    raw scores are finite, 1 when either is not."""
    finite = torch.isfinite(positive) & torch.isfinite(negative)
    return (~finite).to(torch.uint8).cpu()


def lp_scores_table(start: int, positive: Tensor, negative: Tensor, state: Tensor,
                    layout: Dict[str, Any]) -> bytes:
    """Rows ``start ..`` of a scores table as bytes, by `layout`
    (`lp_scores_layout`): one row per pair, its pairs-table row index, its raw
    scores exactly as computed and its state. Rows written chunk by chunk are
    one table, so a writer never holds the whole table as bytes."""
    rows = int(positive.numel())
    if not (int(negative.numel()) == int(state.numel()) == rows):
        raise ValueError("a scores table row needs a positive score, a negative score and a "
                         "state for every pair")
    dtype = _scores_dtype(layout)
    wide = torch.float64 if layout["dtypes"][1] == "<f8" else torch.float32
    table = np.empty(rows, dtype=dtype)
    table["pair_row"] = np.arange(int(start), int(start) + rows, dtype=np.int64)
    table["positive_raw"] = positive.detach().cpu().to(wide).numpy()
    table["negative_raw"] = negative.detach().cpu().to(wide).numpy()
    table["state"] = state.detach().cpu().to(torch.uint8).numpy()
    return table.tobytes()


def lp_scores_chunks(positive: Tensor, negative: Tensor, state: Tensor,
                     layout: Dict[str, Any], *, rows: int = 65536) -> Iterator[bytes]:
    """The whole scores table as consecutive chunks of at most `rows` rows each
    (`lp_scores_table`), in pair order: what a writer streams into a file."""
    if int(rows) <= 0:
        raise ValueError(f"rows must be positive, not {rows}")
    total = int(positive.numel())
    for start in range(0, total, int(rows)):
        stop = start + int(rows)
        yield lp_scores_table(start, positive[start:stop], negative[start:stop],
                              state[start:stop], layout)


def lp_scores_from_table(data: bytes, layout: Dict[str, Any]) -> np.ndarray:
    """The scores table `data`, laid out by `layout`, as a NumPy structured
    array of `LP_SCORE_COLUMNS` (a copy, in row order). Parses the bytes
    given and reads no file.

    Raises:
        ValueError: `layout` is not one this build reads, or `data` is not a
            whole number of rows.
    """
    dtype = _scores_dtype(layout)
    if len(data) % dtype.itemsize:
        raise ValueError(f"{len(data)} bytes are not a whole number of {dtype.itemsize}-byte "
                         "score rows")
    return np.frombuffer(data, dtype=dtype).copy()


__all__ = ["DistMultHead", "LPPairs", "LPScores", "LP_PAIRS_LAYOUT", "LP_PAIR_COLUMNS",
           "LP_PAIR_RULE", "LP_SCORE_COLUMNS", "LP_SCORE_STATES", "corrupt_tails",
           "fixed_lp_pairs", "lp_pairs_digest", "lp_pairs_from_table", "lp_pairs_table",
           "lp_score_states", "lp_scores_chunks", "lp_scores_from_table", "lp_scores_layout",
           "lp_scores_table", "score_lp_pairs"]
