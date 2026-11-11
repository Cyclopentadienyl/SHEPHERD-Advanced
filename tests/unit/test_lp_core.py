"""The one LP core: fixed pairs and their scoring (README §3.2, F3, F4, U5).

`src/models/bundle/lp_head.py` holds the pair rule (`fixed_lp_pairs`: every
edge once, in a seeded order and fixed batches, against one negative drawn
once; an edge with no negative dropped and counted), the digest of the pairs'
bytes (`lp_pairs_digest`), the one statement of what pairs must hold before
they are scored (`lp_pairs_problem`: every id in range for any rule, and
fixed-lp-pairs/v1's own invariants for pairs declaring it, never refusing the
rule's own output; the reviewer's RC-P2-2) and the scoring (`score_lp_pairs`:
an id out of range refused before any pair is scored, then raw scores over
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
    LP_PAIR_RULE,
    DistMultHead,
    LPPairs,
    fixed_lp_pairs,
    lp_pairs_digest,
    lp_pairs_problem,
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
    assert pairs.batch_pairs == (3,)
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
    assert pairs.batch_pairs == (0, 0, 0)
    assert _rows(pairs) == []
    assert pairs.head.dtype == torch.long
    with pytest.raises(ValueError, match="batch_size must be positive"):
        fixed_lp_pairs(edges, torch.zeros(3, dtype=torch.long), seed=0, batch_size=0)


def test_each_batch_s_kept_pairs_are_counted_in_batch_order():
    """`batch_pairs` gives the batch boundaries in the pairs' own rows (plan
    review §5 item 7): batch k's pairs are the next `batch_pairs[k]` rows, each
    an edge of that batch, and the batch's edges less that count are the ones
    it left out; the counts sum to the pairs, the left-out to `without_negative`.
    Eight relations over 60 edges, in batches of 16, leave some edges out."""
    edges, relations = _graph(n_relations=8)
    batch = 16
    pairs = fixed_lp_pairs(edges, relations, seed=5, batch_size=batch)
    order = torch.randperm(edges.size(1), generator=torch.Generator().manual_seed(5 + 7919))
    permuted = [(int(edges[0, i]), int(relations[i]), int(edges[1, i])) for i in order]
    assert len(pairs.batch_pairs) == -(-edges.size(1) // batch)
    assert sum(pairs.batch_pairs) == pairs.counts["pairs"]
    rows = [(h, r, t) for h, r, t, _ in _rows(pairs)]
    start, left_out = 0, 0
    for k, kept in enumerate(pairs.batch_pairs):
        in_batch = permuted[k * batch:(k + 1) * batch]
        assert set(rows[start:start + kept]) <= set(in_batch), k
        left_out += len(in_batch) - kept
        start += kept
    assert left_out == pairs.counts["without_negative"] > 0


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


# --------------------------------------------------------------------------- lp_pairs_problem
#
# The one statement of what pairs hold before they are scored (the reviewer's
# RC-P2-2): every id in range for any rule, and fixed-lp-pairs/v1's own
# invariants for pairs that declare it. The oracles are hand-built pairs.


def _pairs(rows, batch_pairs):
    """`LPPairs` of hand rows ``(head, relation, tail, negative_tail)`` in
    batches of `batch_pairs` (counts and digest as the reader builds them)."""
    columns = tuple(torch.tensor([row[c] for row in rows], dtype=torch.long) for c in range(4))
    return LPPairs(*columns, counts={"edges": len(rows), "without_negative": 0,
                                     "pairs": len(rows)},
                   digest=lp_pairs_digest(*columns), batch_pairs=tuple(batch_pairs))


#: Two batches of a graph of 6 nodes and 2 relations. Batch 0: relation 0's
#: tails 1 and 2, relation 1's tails 3 and 4; batch 1: relation 0's tails 5 and
#: 1. Every negative is the other tail of its relation in its batch.
_VALID_ROWS = [(0, 0, 1, 2), (3, 0, 2, 1), (0, 1, 3, 4), (2, 1, 4, 3),
               (4, 0, 5, 1), (2, 0, 1, 5)]
_VALID_BATCHES = (4, 2)


def test_valid_hand_pairs_have_no_problem_by_their_rule_or_by_ranges_alone():
    pairs = _pairs(_VALID_ROWS, _VALID_BATCHES)
    assert lp_pairs_problem(pairs, nodes=6, relations=2, rule=LP_PAIR_RULE) is None
    assert lp_pairs_problem(pairs, nodes=6, relations=2, rule=None) is None
    empty = _pairs([], (0, 0))
    assert lp_pairs_problem(empty, nodes=6, relations=2, rule=LP_PAIR_RULE) is None


@pytest.mark.parametrize("row, column, says", [
    ((0, 0, 1, -1), 3, "negative_tail id outside [0, 6) (row 0: -1)"),
    ((0, 0, 1, 6), 3, "negative_tail id outside [0, 6) (row 0: 6)"),
    ((-1, 0, 1, 2), 0, "head id outside [0, 6) (row 0: -1)"),
    ((0, 0, 9, 2), 2, "tail id outside [0, 6) (row 0: 9)"),
    ((0, -1, 1, 2), 1, "relation id outside [0, 2) (row 0: -1)"),
    ((0, 2, 1, 2), 1, "relation id outside [0, 2) (row 0: 2)"),
])
def test_an_id_out_of_range_is_a_problem_for_any_rule(row, column, says):
    """A negative id (which torch would read as counted from the end) or one
    at or past the count is named with its column, row and value, whatever
    rule the pairs declare, or none."""
    pairs = _pairs([row] + _VALID_ROWS[1:], _VALID_BATCHES)
    for rule in (LP_PAIR_RULE, "another-rule/v1", None):
        problem = lp_pairs_problem(pairs, nodes=6, relations=2, rule=rule)
        assert problem == f"1 of the 6 pairs hold a {says}", (rule, problem)


def test_a_column_that_is_not_one_integer_id_per_pair_is_a_problem():
    pairs = _pairs(_VALID_ROWS, _VALID_BATCHES)
    short = LPPairs(pairs.head, pairs.relation, pairs.tail, pairs.negative_tail[1:],
                    pairs.counts, pairs.digest, pairs.batch_pairs)
    assert "the negative_tail column has shape (5,), not one id for each of the 6 pairs" == \
        lp_pairs_problem(short, nodes=6, relations=2, rule=None)
    real = LPPairs(pairs.head, pairs.relation, pairs.tail.double(), pairs.negative_tail,
                   pairs.counts, pairs.digest, pairs.batch_pairs)
    assert "the tail column holds torch.float64 values, not integer ids" == \
        lp_pairs_problem(real, nodes=6, relations=2, rule=None)


def test_a_negative_equal_to_its_own_tail_breaks_the_rule_only():
    """fixed-lp-pairs/v1 never draws a pair's own tail (`corrupt_tails`: a
    tail different by value); the pairs are still in range, so a pair set
    declaring another rule, whose semantics are its own, is not refused."""
    rows = [(0, 0, 1, 1)] + _VALID_ROWS[1:]
    pairs = _pairs(rows, _VALID_BATCHES)
    assert lp_pairs_problem(pairs, nodes=6, relations=2, rule=LP_PAIR_RULE) == (
        "1 of the 6 pairs have a negative_tail equal to their own tail (row 0: 1), which "
        "fixed-lp-pairs/v1 never draws")
    assert lp_pairs_problem(pairs, nodes=6, relations=2, rule="another-rule/v1") is None
    assert lp_pairs_problem(pairs, nodes=6, relations=2, rule=None) is None


@pytest.mark.parametrize("negative, what", [
    (3, "a tail of relation 1 in the same batch"),
    (5, "a tail of relation 0 in the other batch"),
    (0, "a node that is no kept tail"),
])
def test_a_negative_from_another_relation_or_batch_breaks_the_rule(negative, what):
    """The rule draws a negative among the tails of its own relation's kept
    pairs in its own batch: a tail of another relation, of another batch, or
    of no kept pair is not one it draws."""
    rows = [(0, 0, 1, negative)] + _VALID_ROWS[1:]
    problem = lp_pairs_problem(_pairs(rows, _VALID_BATCHES), nodes=6, relations=2,
                               rule=LP_PAIR_RULE)
    assert problem == (
        f"1 of the 6 pairs have a negative_tail that is not the tail of a kept pair of the same "
        f"relation in the same batch (row 0: {negative}, relation 0, batch 0), the only tails "
        f"fixed-lp-pairs/v1 draws from"), what


def test_a_negative_is_found_in_its_own_batch_by_its_batch_boundaries():
    """The batch boundaries decide: cut 4 and 2, row 5's negative (5) is row
    4's tail, in its batch; cut 5 and 1, row 4 is in batch 0 and row 5 alone
    in batch 1, whose one relation-0 tail is its own (1), so its negative is
    in no kept pair of its batch."""
    assert lp_pairs_problem(_pairs(_VALID_ROWS, (4, 2)), nodes=6, relations=2,
                            rule=LP_PAIR_RULE) is None
    problem = lp_pairs_problem(_pairs(_VALID_ROWS, (5, 1)), nodes=6, relations=2,
                               rule=LP_PAIR_RULE)
    assert problem.startswith("1 of the 6 pairs have a negative_tail that is not the tail of "
                              "a kept pair of the same relation in the same batch (row 5: 5, "
                              "relation 0, batch 1)"), problem


@pytest.mark.parametrize("batch_pairs, says", [
    ((4, 1), "batch_pairs sum to 5, not the 6 pairs"),
    ((4, 2, 1), "batch_pairs sum to 7, not the 6 pairs"),
    ((7, -1), "batch_pairs[1] is -1, not a count of the pairs a batch kept (a whole number of "
              "at least 0)"),
    ((4, True, 1), "batch_pairs[1] is True, not a count of the pairs a batch kept (a whole "
                   "number of at least 0)"),
])
def test_batch_counts_that_are_not_the_rows_batch_by_batch_break_the_rule(batch_pairs, says):
    pairs = _pairs(_VALID_ROWS, batch_pairs)
    assert lp_pairs_problem(pairs, nodes=6, relations=2, rule=LP_PAIR_RULE) == says
    assert lp_pairs_problem(pairs, nodes=6, relations=2, rule=None) is None


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("batch_size", [1, 2, 3, 7, 16, 64, 1000])
def test_fixed_lp_pairs_own_pairs_never_break_their_rule(seed, batch_size):
    """No false refusal: the rule's own output passes its own statement, over
    seeds and batch sizes, on a graph whose tails repeat within a batch (6
    nodes over 200 edges), where relation 2 has one tail only (every one of
    its edges dropped) and relation 3 two tails, one of them on most edges."""
    edges, relations = _graph(n_edges=200, n_nodes=6, n_relations=4, seed=seed)
    edges[1, relations == 2] = 4
    three = (relations == 3).nonzero().flatten()
    edges[1, three] = torch.where(three % 5 == 0, 1, 2)
    pairs = fixed_lp_pairs(edges, relations, seed=seed, batch_size=batch_size)
    assert lp_pairs_problem(pairs, nodes=6, relations=4, rule=LP_PAIR_RULE) is None
    assert 2 not in pairs.relation.tolist()
    if batch_size >= 16:
        first = [(int(r), int(t)) for r, t in zip(pairs.relation[:pairs.batch_pairs[0]],
                                                   pairs.tail[:pairs.batch_pairs[0]],
                                                   strict=True)]
        assert len(set(first)) < len(first), "no relation's tail repeats in the first batch"


def test_ids_near_the_int64_limit_are_checked_without_overflow():
    """No id is composed into a key: ids of a view as large as int64 allows
    are judged exactly (a key ``(batch * relations + relation) * nodes + id``
    would overflow here and could alias another triple)."""
    big = 2**62
    rows = [(big, 1, big + 1, big + 2), (big + 3, 1, big + 2, big + 1)]
    assert lp_pairs_problem(_pairs(rows, (2,)), nodes=2**63 - 1, relations=2**40,
                            rule=LP_PAIR_RULE) is None
    rows[1] = (big + 3, 1, big + 2, big + 5)
    problem = lp_pairs_problem(_pairs(rows, (2,)), nodes=2**63 - 1, relations=2**40,
                               rule=LP_PAIR_RULE)
    assert problem is not None and f"(row 1: {big + 5}, relation 1, batch 0)" in problem


def test_a_negative_whose_composed_key_would_alias_a_kept_tail_is_refused():
    """The case a composed key gets wrong (the review of RC-P2-2's fix): with
    ``nodes = 2**63 - 1`` the key of (batch 0, relation 2, id 7) is
    ``2 * nodes + 7``, which wraps in int64 to 5, the key of (batch 0,
    relation 0, id 5), a kept tail; a composed-key membership would accept
    row 2's negative. Relation 2's only kept tail in the batch is 8."""
    rows = [(0, 0, 5, 6), (1, 0, 6, 5), (2, 2, 8, 7)]
    nodes = 2**63 - 1
    assert (2 * nodes + 7) - 2**64 == 5, "the triple no longer aliases under an int64 key"
    problem = lp_pairs_problem(_pairs(rows, (3,)), nodes=nodes, relations=2**40,
                               rule=LP_PAIR_RULE)
    assert problem is not None and "(row 2: 7, relation 2, batch 0)" in problem
    assert lp_pairs_problem(_pairs(rows[:2], (2,)), nodes=nodes, relations=2**40,
                            rule=LP_PAIR_RULE) is None


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
                    digest=lp_pairs_digest(*columns), batch_pairs=(2,))
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


@pytest.mark.parametrize("column, value, says", [
    (3, -1, "negative_tail id outside [0, 4) (row 1: -1)"),
    (3, 4, "negative_tail id outside [0, 4) (row 1: 4)"),
    (0, -2, "head id outside [0, 4) (row 1: -2)"),
    (1, 2, "relation id outside [0, 2) (row 1: 2)"),
])
def test_an_id_out_of_range_is_refused_before_any_pair_is_scored(column, value, says):
    """The hand oracle's pairs with one id out of range: torch would read -1
    as the last node's vector and score it, and an id past the end would fail
    somewhere inside the scoring. The scoring refuses first, naming the id,
    before any pair reaches the head; the range is the vectors' rows and the
    head's relations."""
    head = DistMultHead(LPHeadSpec(), num_relations=2, width=2)
    scored = []
    raw_score = head.raw_score
    head.raw_score = lambda *a: scored.append(a) or raw_score(*a)
    bundle = _Bundle(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]]), head)
    columns = [torch.tensor([0, 2]), torch.tensor([0, 1]), torch.tensor([2, 3]),
               torch.tensor([1, 0])]
    columns[column] = torch.tensor([columns[column][0].item(), value])
    pairs = LPPairs(*columns, counts={"edges": 2, "without_negative": 0, "pairs": 2},
                    digest=lp_pairs_digest(*columns), batch_pairs=(2,))
    with pytest.raises(ValueError) as info:
        score_lp_pairs(bundle, pairs, _ei([(1, 0), (3, 2)]), torch.tensor([1, 0]),
                       torch.device("cpu"))
    assert str(info.value) == f"the LP pairs cannot be scored: 1 of the 2 pairs hold a {says}"
    assert scored == []


def test_no_pair_is_nothing_to_measure():
    head = DistMultHead(LPHeadSpec(), num_relations=1, width=2)
    bundle = _Bundle(torch.zeros(2, 2), head)
    pairs = fixed_lp_pairs(_ei([(0, 1)]), torch.zeros(1, dtype=torch.long), seed=0,
                           batch_size=1)
    with pytest.raises(ValueError, match="no LP pair was evaluated"):
        score_lp_pairs(bundle, pairs, _ei([(0, 1)]), torch.zeros(1, dtype=torch.long),
                       torch.device("cpu"))
