"""The one LP core: fixed pairs and their scoring (README §3.2, F3, F4, U5).

`src/models/bundle/lp_head.py` holds the pair rule (`fixed_lp_pairs`: every
edge once, in a seeded order and fixed batches, against one negative drawn
once; an edge with no negative dropped and counted), the digest of the pairs'
bytes (`lp_pairs_digest`) and the scoring (`score_lp_pairs`: raw scores over
full-view vectors, whole-set metrics and the hinge). Base validation takes
both (test_base_lp_task.py; P0-3 pins it byte for byte). The oracles here are
hand arithmetic or short computations written apart from the code under test.

Module: tests/unit/test_lp_core.py
"""
from __future__ import annotations

import hashlib
import json
import math
import struct

import pytest

torch = pytest.importorskip("torch")

from src.models.bundle.lp_head import (  # noqa: E402
    LP_PAIR_COLUMNS,
    DistMultHead,
    LPPairs,
    fixed_lp_pairs,
    lp_pairs_digest,
    score_lp_pairs,
)
from src.models.bundle.spec import LPHeadSpec  # noqa: E402


def _ei(pairs):
    return torch.tensor(pairs, dtype=torch.long).T.reshape(2, -1)


def _rows(pairs):
    return list(zip(pairs.head.tolist(), pairs.relation.tolist(), pairs.tail.tolist(),
                    pairs.negative_tail.tolist(), strict=True))


def _graph(n_edges=60, n_nodes=25, n_relations=3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    edges = torch.randint(n_nodes, (2, n_edges), generator=generator)
    relations = torch.randint(n_relations, (n_edges,), generator=generator)
    return edges, relations


# --------------------------------------------------------------------------- fixed_lp_pairs


def test_the_same_edges_seed_and_batches_give_the_same_pairs_and_digest():
    edges, relations = _graph()
    a = fixed_lp_pairs(edges, relations, seed=3, batch_size=16)
    b = fixed_lp_pairs(edges.clone(), relations.clone(), seed=3, batch_size=16)
    assert _rows(a) == _rows(b) and a.counts == b.counts and a.digest == b.digest
    for column in (a.head, a.relation, a.tail, a.negative_tail):
        assert column.dtype == torch.long
    other = fixed_lp_pairs(edges, relations, seed=4, batch_size=16)
    assert _rows(other) != _rows(a) and other.digest != a.digest


def test_the_pairs_follow_the_seeded_permutation_each_edge_once():
    """The order is torch's permutation under `seed + 7919` (the rule training
    rules 1 pin); an edge with a negative is kept once, in that order, with a
    negative that is another tail of its relation in its batch."""
    edges, relations = _graph()
    batch = 16
    pairs = fixed_lp_pairs(edges, relations, seed=5, batch_size=batch)
    order = torch.randperm(edges.size(1), generator=torch.Generator().manual_seed(5 + 7919))
    permuted = [(int(edges[0, i]), int(relations[i]), int(edges[1, i])) for i in order]
    kept = [(h, r, t) for h, r, t, _ in _rows(pairs)]
    position = {edge: i for i, edge in enumerate(permuted)}
    assert len(set(permuted)) == len(permuted), "the toy graph repeats an edge"
    assert [position[e] for e in kept] == sorted(position[e] for e in kept)
    for (_, r, t, negative), i in zip(_rows(pairs), (position[e] for e in kept), strict=True):
        start = i // batch * batch
        same_batch = permuted[start:start + batch]
        assert negative != t
        assert negative in {tail for _, rel, tail in same_batch if rel == r}


def test_the_counts_name_every_edge_and_every_one_dropped_without_a_negative():
    """Relation 0's three edges have three tails, so each has a negative;
    relation 1's one edge has no other tail, and relation 2's two edges share
    their tail: those three have none, are dropped and counted."""
    edges = _ei([(0, 1), (2, 3), (4, 5), (1, 0), (3, 6), (5, 6)])
    relations = torch.tensor([0, 0, 0, 1, 2, 2])
    pairs = fixed_lp_pairs(edges, relations, seed=0, batch_size=6)
    assert pairs.counts == {"edges": 6, "without_negative": 3, "pairs": 3}
    assert sorted((h, r, t) for h, r, t, _ in _rows(pairs)) == [(0, 0, 1), (2, 0, 3), (4, 0, 5)]
    for _, _, tail, negative in _rows(pairs):
        assert negative in {1, 3, 5} - {tail}


def test_a_negative_is_drawn_within_a_batch_never_across_its_boundary():
    """Batches of one hold one edge each: no edge has another tail of its
    relation beside it, so every edge is dropped and counted, and the pairs
    are empty."""
    edges = _ei([(0, 1), (2, 3), (4, 5)])
    pairs = fixed_lp_pairs(edges, torch.zeros(3, dtype=torch.long), seed=0, batch_size=1)
    assert pairs.counts == {"edges": 3, "without_negative": 3, "pairs": 0}
    assert _rows(pairs) == []
    assert pairs.head.dtype == torch.long
    with pytest.raises(ValueError, match="batch_size must be positive"):
        fixed_lp_pairs(edges, torch.zeros(3, dtype=torch.long), seed=0, batch_size=0)


def _digest_by_hand(rows):
    """The documented layout, written apart: the JSON header, then one row of
    four little-endian int64 per pair."""
    h = hashlib.sha256()
    h.update(json.dumps({"columns": ["head", "relation", "tail", "negative_tail"],
                         "dtype": "<i8", "order": "rows"}, sort_keys=True).encode())
    for row in rows:
        h.update(struct.pack("<4q", *row))
    return h.hexdigest()


def test_the_digest_is_over_the_exact_pair_bytes():
    edges, relations = _graph()
    pairs = fixed_lp_pairs(edges, relations, seed=1, batch_size=16)
    rows = _rows(pairs)
    assert LP_PAIR_COLUMNS == ("head", "relation", "tail", "negative_tail")
    assert pairs.digest == _digest_by_hand(rows)
    assert fixed_lp_pairs(_ei([(0, 1)]), torch.zeros(1, dtype=torch.long), seed=0,
                          batch_size=1).digest == _digest_by_hand([])


def test_the_digest_changes_when_a_pair_changes():
    edges, relations = _graph()
    pairs = fixed_lp_pairs(edges, relations, seed=1, batch_size=16)
    columns = [pairs.head, pairs.relation, pairs.tail, pairs.negative_tail]
    assert lp_pairs_digest(*columns) == pairs.digest
    for c in range(4):
        changed = [column.clone() for column in columns]
        changed[c][0] += 1
        assert lp_pairs_digest(*changed) != pairs.digest, LP_PAIR_COLUMNS[c]
    swapped = [torch.cat([column[1:2], column[0:1], column[2:]]) for column in columns]
    assert _rows(pairs)[0] != _rows(pairs)[1]
    assert lp_pairs_digest(*swapped) != pairs.digest
    assert lp_pairs_digest(*(column[1:] for column in columns)) != pairs.digest


# --------------------------------------------------------------------------- score_lp_pairs


class _Bundle:
    """Fixed node vectors and a real DistMult head; records how its full-view
    entry was called."""

    def __init__(self, z, head):
        self.z, self.lp_head, self.calls, self.modes = z, head, [], []

    def eval(self):
        self.modes.append("eval")
        return self

    def node_vectors_full(self, edge_index, *, edge_type=None, node_type=None, chunk_size=65536):
        self.calls.append((edge_index.tolist(), edge_type.tolist(),
                           None if node_type is None else node_type.tolist(), chunk_size))
        return self.z


def test_scores_and_metrics_by_hand():
    """Four node vectors, two relations, two pairs. By hand:

    - positives: z0·w0·z2 = 1, z2·w1·z3 = 2·2 + 0.5·(−1) = 3.5;
    - negatives: z0·w0·z1 = 0, z2·w1·z0 = 2;
    - accuracy (raw > 0): 1, 3.5 and 2 are not ≤ 0 → 3 of 4 right = 0.75;
    - ROC-AUC: of the four (positive, negative) pairs, (1, 2) is the one the
      negative wins → 0.75;
    - AP: descending 3.5 P, 2 N, 1 P, 0 N → ½·1 + ½·⅔ = ⅚;
    - lp_loss: mean of max(0, 1 − (tanh 1 − tanh 0)) and
      max(0, 1 − (tanh 3.5 − tanh 2)).

    Message passing is exactly the edges given, in evaluation mode, with the
    chunk size and the view's node types given; the pairs never reach it."""
    head = DistMultHead(LPHeadSpec(), num_relations=2, width=2)
    with torch.no_grad():
        head.relation_weights.copy_(torch.tensor([[1.0, 1.0], [2.0, 0.5]]))
    bundle = _Bundle(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]]), head)
    columns = (torch.tensor([0, 2]), torch.tensor([0, 1]), torch.tensor([2, 3]),
               torch.tensor([1, 0]))
    pairs = LPPairs(*columns, counts={"edges": 2, "without_negative": 0, "pairs": 2},
                    digest=lp_pairs_digest(*columns))
    message_passing = _ei([(1, 0), (3, 2)])

    scored = score_lp_pairs(bundle, pairs, message_passing, torch.tensor([1, 0]),
                            torch.device("cpu"), chunk_size=7,
                            node_type=torch.tensor([0, 1, 1, 0]))

    assert bundle.modes == ["eval"]
    assert bundle.calls == [([[1, 3], [0, 2]], [1, 0], [0, 1, 1, 0], 7)]
    assert scored.positive.tolist() == [1.0, 3.5]
    assert scored.negative.tolist() == [0.0, 2.0]
    loss = (max(0.0, 1.0 - (math.tanh(1.0) - math.tanh(0.0)))
            + max(0.0, 1.0 - (math.tanh(3.5) - math.tanh(2.0)))) / 2
    assert scored.metrics == {
        "lp_accuracy": 0.75, "lp_roc_auc": 0.75,
        "lp_average_precision": pytest.approx(5 / 6),
        "lp_positives": 2, "lp_negatives": 2, "lp_loss": pytest.approx(loss)}


def test_no_pair_is_nothing_to_measure():
    head = DistMultHead(LPHeadSpec(), num_relations=1, width=2)
    bundle = _Bundle(torch.zeros(2, 2), head)
    pairs = fixed_lp_pairs(_ei([(0, 1)]), torch.zeros(1, dtype=torch.long), seed=0,
                           batch_size=1)
    with pytest.raises(ValueError, match="no LP pair was evaluated"):
        score_lp_pairs(bundle, pairs, _ei([(0, 1)]), torch.zeros(1, dtype=torch.long),
                       torch.device("cpu"))
