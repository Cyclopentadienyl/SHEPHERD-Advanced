"""The reference graph view and its fact split (README G1, G5, F2; regression 9).

`src/kg/graph_view.py` turns a workspace graph read into one homogeneous typed
graph: excluded relations leave with their reverse twins, native self-loops
and same-relation duplicates are dropped, the largest component is kept and
the excluded nodes are counted. Only then the 80/10/10 split cuts facts, so an
edge and its reverse twin share a part. Expected values are counted by hand on
a toy graph.

Module: tests/unit/test_graph_view.py
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.kg.graph_view import (  # noqa: E402
    GraphViewError,
    build_graph_view,
    split_facts,
)


def _ei(pairs):
    return torch.tensor(pairs, dtype=torch.long).T.reshape(2, -1)


def _toy():
    """gene 0..2, phenotype 0..2, disease 0..1.

    Forward edges with their `rev_` twins:
      gene0-pheno0 (twice: a duplicate), gene1-pheno0, gene0-gene0 (native
      self-loop), disease0-pheno0 (assoc) and disease0-pheno0 (a second
      relation, kept distinct), gene2-pheno2 (a second component of 2 nodes).
    pheno1 and disease1 are isolated.
    """
    gp = [(0, 0), (0, 0), (1, 0), (2, 2)]
    gg = [(0, 0)]
    dp = [(0, 0)]
    dp2 = [(0, 0)]
    return {
        "num_nodes_dict": {"gene": 3, "phenotype": 3, "disease": 2},
        "edge_index_dict": {
            ("gene", "has", "phenotype"): _ei(gp),
            ("phenotype", "rev_has", "gene"): _ei([(b, a) for a, b in gp]),
            ("gene", "interacts", "gene"): _ei(gg),
            ("gene", "rev_interacts", "gene"): _ei(gg),
            ("disease", "assoc", "phenotype"): _ei(dp),
            ("phenotype", "rev_assoc", "disease"): _ei([(b, a) for a, b in dp]),
            ("disease", "onset", "phenotype"): _ei(dp2),
            ("phenotype", "rev_onset", "disease"): _ei([(b, a) for a, b in dp2]),
        },
    }


def test_global_ids_follow_sorted_types():
    view = build_graph_view(_toy())
    assert view.node_types == ["disease", "gene", "phenotype"]
    assert view.type_offsets == {"disease": 0, "gene": 2, "phenotype": 5}
    assert view.raw_num_nodes == 8


def test_self_loops_duplicates_and_component_counted():
    view = build_graph_view(_toy())
    # 4 + 4 has/rev_has, 1 + 1 interacts, 2 assoc, 2 onset.
    assert view.counts["edges_exported"] == 14
    # gene0-gene0 forward and reverse.
    assert view.counts["native_self_loops_dropped"] == 2
    # gene0-pheno0 and its twin, once each.
    assert view.counts["duplicates_dropped"] == 2
    # {d0, g0, g1, p0}, {g2, p2}, {d1}, {p1}.
    assert view.counts["components"] == 4
    assert view.counts["nodes_excluded_by_type"] == {"disease": 1, "gene": 1, "phenotype": 2}
    assert view.counts["edges_outside_component"] == 2
    assert view.num_nodes == 4
    # has ×2, rev_has ×2, assoc, rev_assoc, onset, rev_onset.
    assert view.num_edges == 8
    assert view.num_facts == 4


def test_different_relations_between_the_same_nodes_stay_distinct():
    view = build_graph_view(_toy())
    d0 = int(view.view_ids("disease", torch.tensor([0])))
    p0 = int(view.view_ids("phenotype", torch.tensor([0])))
    rels = {view.relations[int(r)] for s, d, r in zip(view.src, view.dst, view.rel)
            if int(s) == d0 and int(d) == p0}
    assert rels == {"disease|assoc|phenotype", "disease|onset|phenotype"}


def test_mapping_round_trips_and_excluded_nodes_keep_their_ids():
    view = build_graph_view(_toy())
    assert torch.equal(view.raw_to_view[view.view_to_raw], torch.arange(view.num_nodes))
    assert int(view.view_ids("gene", torch.tensor([2]))) == -1
    genes = view.type_nodes("gene")
    assert view.local_index(genes).tolist() == [0, 1]


def test_reverse_twins_share_a_fact():
    view = build_graph_view(_toy())
    for f in range(view.num_facts):
        members = (view.fact == f).nonzero().flatten()
        assert members.numel() == 2
        a, b = members.tolist()
        assert int(view.src[a]) == int(view.dst[b]) and int(view.dst[a]) == int(view.src[b])
        assert int(view.reverse_relation[view.rel[a]]) == int(view.rel[b])


def test_excluding_a_relation_removes_its_twin():
    view = build_graph_view(_toy(), rule={
        "rule": "reference/v1", "drop_native_self_loops": True, "deduplicate": True,
        "largest_component": True, "exclude_relations": ["disease|onset|phenotype"]})
    assert "disease|onset|phenotype" not in view.relations
    assert "phenotype|rev_onset|disease" not in view.relations
    with pytest.raises(GraphViewError, match="does not have"):
        build_graph_view(_toy(), rule={"rule": "reference/v1",
                                       "exclude_relations": ["gene|nope|gene"]})


def test_unknown_rule_refused():
    with pytest.raises(GraphViewError, match="unknown graph view rule"):
        build_graph_view(_toy(), rule={"rule": "legacy"})


def test_same_graph_same_digest_and_split():
    a, b = build_graph_view(_toy()), build_graph_view(_toy())
    assert a.digest == b.digest
    sa, sb = split_facts(a, seed=7), split_facts(b, seed=7)
    assert sa.digest == sb.digest
    assert torch.equal(sa.part, sb.part)
    # A twin pair never straddles parts.
    for f in range(a.num_facts):
        assert (sa.part[a.fact == f]).unique().numel() == 1


def test_split_digest_changes_with_the_view():
    a = build_graph_view(_toy())
    b = build_graph_view(_toy(), rule={"rule": "reference/v1", "drop_native_self_loops": True,
                                       "deduplicate": True, "largest_component": False,
                                       "exclude_relations": []})
    assert a.digest != b.digest
    assert split_facts(a, 7).digest != split_facts(b, 7).digest


def test_split_proportions_over_facts():
    n = 50
    data = {"num_nodes_dict": {"gene": n + 1},
            "edge_index_dict": {
                ("gene", "link", "gene"): _ei([(0, i) for i in range(1, n + 1)]),
                ("gene", "rev_link", "gene"): _ei([(i, 0) for i in range(1, n + 1)])}}
    view = build_graph_view(data)
    split = split_facts(view, seed=1)
    assert view.num_facts == n
    assert split.record()["edges"] == {"train": 80, "val": 10, "test": 10}
