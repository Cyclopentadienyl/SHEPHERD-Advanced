"""The gene task's score parts against hand arithmetic (README §6, F10, C9;
regression 10). Expected values are computed by hand or by a few independent
lines, never by the code under test.

Module: tests/unit/test_gene_task_math.py
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from src.kg.graph_view import exact_distances  # noqa: E402
from src.models.tasks.gene import augment, mean_distances, neighbour_table, splsim  # noqa: E402


def test_the_worked_example_of_section_6():
    """Two phenotypes, three candidates: distances g1 (1, 3), g2 (2, 4), g3 (5, 8)
    give d̄ 2, 3, 6.5 and S +0.6575, +0.3151, −0.8836 with bounds 1 and 6.84;
    mixed with EMBSIM (η 0.5; g1 −0.8, g3 +0.9) g3 ranks above g1."""
    distances = torch.tensor([[1, 2, 5], [3, 4, 8]], dtype=torch.int32)
    mean, defined = mean_distances(distances, torch.tensor([[0, 1]]),
                                   torch.tensor([[True, True]]), torch.tensor([[0, 1, 2]]))
    assert mean.tolist() == [[2.0, 3.0, 6.5]] and defined.all()
    s = splsim(mean, 1.0, 6.84)[0].tolist()
    assert s == pytest.approx([0.6575, 0.3151, -0.8836], abs=5e-5)
    g1, g3 = 0.5 * -0.8 + 0.5 * s[0], 0.5 * 0.9 + 0.5 * s[2]
    assert g3 > g1


def test_a_value_outside_the_nominal_range_is_not_clipped():
    assert splsim(torch.tensor([10.0]), 1.0, 6.84).item() == pytest.approx(1 - 18 / 5.84)


def test_an_unreachable_phenotype_makes_the_pair_undefined_not_capped():
    distances = torch.tensor([[1, 2], [-1, 3]], dtype=torch.int32)
    mean, defined = mean_distances(distances, torch.tensor([[0, 1]]),
                                   torch.tensor([[True, True]]), torch.tensor([[0, 1]]))
    assert defined.tolist() == [[False, True]]
    assert mean[0, 1].item() == 2.5


def test_a_padded_phenotype_is_not_averaged():
    distances = torch.tensor([[2], [-1]], dtype=torch.int32)
    mean, defined = mean_distances(distances, torch.tensor([[0, 1]]),
                                   torch.tensor([[True, False]]), torch.tensor([[0]]))
    assert mean.item() == 2.0 and defined.item()


def _graph(n, edges):
    src = torch.tensor([a for a, b in edges] + [b for a, b in edges])
    dst = torch.tensor([b for a, b in edges] + [a for a, b in edges])
    return SimpleNamespace(num_nodes=n, src=src, dst=dst)


def test_exact_distances_are_uncapped_hops_and_unreachable_is_minus_one():
    # A path 0-1-2-3-4-5-6-7 (seven hops end to end) and an isolated node 8.
    view = _graph(9, [(i, i + 1) for i in range(7)])
    d = exact_distances(view, torch.tensor([0, 3]), torch.tensor([7, 3, 8]), chunk_size=1)
    assert d.tolist() == [[7, 3, -1], [4, 0, -1]]


def _view_with_genes():
    """Genes 0-3, phenotypes 4-6. g0 has p4, p5; g1 p4, p5, p6; g2 p6; g3 none
    (joined to g2 by another relation so it stays connected)."""
    node_types = ["gene", "phenotype"]
    node_type = torch.tensor([0, 0, 0, 0, 1, 1, 1])
    relations = ["gene|gene_has_phenotype|phenotype", "phenotype|rev_gene_has_phenotype|gene",
                 "gene|interacts|gene"]
    facts = [(0, 4, 0), (0, 5, 0), (1, 4, 0), (1, 5, 0), (1, 6, 0), (2, 6, 0)]
    src, dst, rel = [], [], []
    for a, b, r in facts:
        src += [a, b]
        dst += [b, a]
        rel += [r, 1]
    src += [2, 3]
    dst += [3, 2]
    rel += [2, 2]

    def type_nodes(t):
        return torch.nonzero(node_type == node_types.index(t)).flatten()

    return SimpleNamespace(num_nodes=7, node_type=node_type, relations=relations,
                           src=torch.tensor(src), dst=torch.tensor(dst), rel=torch.tensor(rel),
                           type_nodes=type_nodes)


def test_neighbours_are_the_genes_sharing_most_phenotypes():
    table = neighbour_table(_view_with_genes(), k=3)
    assert table.neighbours[0].tolist() == [1, -1, -1]
    assert table.weights[0].tolist() == [1.0, 0.0, 0.0]
    # g1 shares two with g0 and one with g2: weights 2/3 and 1/3.
    assert table.neighbours[1].tolist() == [0, 2, -1]
    assert table.weights[1].tolist() == pytest.approx([2 / 3, 1 / 3, 0.0])
    assert table.neighbours[2].tolist() == [1, -1, -1]
    # g3 shares no phenotype: no neighbour, recorded.
    assert table.neighbours[3].tolist() == [-1, -1, -1]
    assert table.record["genes_without_neighbour"] == 1
    assert table.record["relations"] == ["gene|gene_has_phenotype|phenotype"]
    # Degrees are the view's out-edges: g1 has three phenotype edges.
    assert table.degree[:4].tolist() == [2.0, 3.0, 2.0, 1.0]


def test_ties_keep_the_smaller_view_id():
    # Without g0's p5, g1 shares one phenotype with g0 (p4) and one with g2 (p6):
    # with K = 1 the tie goes to g0.
    view = _view_with_genes()
    keep = ~((view.src == 0) & (view.dst == 5) | (view.src == 5) & (view.dst == 0))
    view.src, view.dst, view.rel = view.src[keep], view.dst[keep], view.rel[keep]
    table = neighbour_table(view, k=1)
    assert table.neighbours[1].tolist() == [0]


def test_eq_10_on_two_genes_by_hand():
    """ω 0.7, degree 1: θ = 0.7·e^(−0.7) + 0.2; the gene [1, 0] with one
    neighbour [0, 1] becomes [1 − θ, θ]."""
    theta = 0.7 * math.exp(-0.7) + 0.2
    out, has = augment(torch.tensor([[1.0, 0.0]]), torch.tensor([[[0.0, 1.0]]]),
                       torch.tensor([[1.0]]), torch.tensor([1.0]), 0.7)
    assert has.tolist() == [True]
    assert out[0].tolist() == pytest.approx([1 - theta, theta])


def test_a_gene_without_a_neighbour_keeps_its_own_vector():
    own = torch.tensor([[0.3, -0.4]])
    out, has = augment(own, torch.zeros(1, 2, 2), torch.zeros(1, 2), torch.tensor([0.0]), 0.7)
    assert has.tolist() == [False]
    assert torch.equal(out, own)
