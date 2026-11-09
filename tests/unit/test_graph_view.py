"""The reference graph view and its fact split (README G1, G5, F2; regression 9).

`src/kg/graph_view.py` turns a workspace graph read into one homogeneous typed
graph: excluded relations leave with their reverse twins, native self-loops
and same-relation duplicates are dropped, the largest component is kept and
the excluded nodes are counted. Only then the 80/10/10 split cuts facts, so an
edge and its reverse twin share a part. Expected values are counted by hand on
a toy graph.

`view_ids` refuses a local id outside its type's raw range, and the disease
binder lets that refusal through (NEW-kg-1, K8-6b).

Module: tests/unit/test_graph_view.py
"""
from __future__ import annotations

from types import SimpleNamespace

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


# --------------------------------------------------------------------------- the raw id range
# On the toy graph the offsets are disease 0, gene 2, phenotype 5 and `raw_to_view`
# is [0, -1, 1, 2, -1, 3, -1, -1]. Unbounded, disease 2 read a gene node, gene 3 a
# phenotype node, disease -3 a phenotype node, gene -2 a disease node, disease -1
# the last raw node ("outside the view"), and phenotype 3 raised a bare IndexError.


@pytest.mark.parametrize("node_type, local, named, count", [
    ("disease", [2], 2, 2), ("gene", [3], 3, 3), ("gene", [0, 1, 3], 3, 3),  # into the next type
    ("phenotype", [3], 3, 3),                                                # past the last node
    ("disease", [-1], -1, 2), ("disease", [-3], -3, 2), ("gene", [-2], -2, 3),
])
def test_an_id_outside_its_types_raw_range_is_refused(node_type, local, named, count):
    view = build_graph_view(_toy())
    with pytest.raises(GraphViewError, match=rf"^{node_type} id {named} is outside the graph's "
                                             rf"{count} raw {node_type} nodes"):
        view.view_ids(node_type, torch.tensor(local))


def test_the_legal_boundary_and_an_empty_list_are_not_refused():
    view = build_graph_view(_toy())
    # The last id of each type; the toy view excludes d1 and g2, so both are -1.
    assert view.view_ids("disease", torch.tensor([1])).tolist() == [-1]
    assert view.view_ids("gene", torch.tensor([2])).tolist() == [-1]
    empty = view.view_ids("phenotype", torch.tensor([], dtype=torch.long))
    assert empty.numel() == 0 and empty.dtype == torch.long


def test_the_bound_is_the_raw_count_never_the_views():
    """gene 0..2 with gene 1 isolated, one phenotype, one disease: the view keeps
    two gene nodes, and gene 2 is still a legal id with its own view id."""
    gp, dp = [(0, 0), (2, 0)], [(0, 0)]
    view = build_graph_view({
        "num_nodes_dict": {"gene": 3, "phenotype": 1, "disease": 1},
        "edge_index_dict": {
            ("gene", "has", "phenotype"): _ei(gp),
            ("phenotype", "rev_has", "gene"): _ei([(b, a) for a, b in gp]),
            ("disease", "assoc", "phenotype"): _ei(dp),
            ("phenotype", "rev_assoc", "disease"): _ei([(b, a) for a, b in dp]),
        }})
    assert view.type_offsets == {"disease": 0, "gene": 1, "phenotype": 4}
    assert view.type_nodes("gene").numel() == 2
    assert view.view_ids("gene", torch.tensor([2])).tolist() == [2]
    assert view.view_ids("gene", torch.tensor([1])).tolist() == [-1]


@pytest.mark.parametrize("disease_id", [2, -3])
def test_the_disease_binder_lets_the_refusal_through(disease_id):
    """No binder catches it: the run stops with the reason, naming no case,
    instead of binding the case to another type's node."""
    from src.models.tasks.scoring import bind_disease_cases

    view = build_graph_view(_toy())
    samples = [SimpleNamespace(patient_id="MRN-1", phenotype_ids=[0], disease_id=0),
               SimpleNamespace(patient_id="MRN-2", phenotype_ids=[0], disease_id=disease_id)]
    with pytest.raises(GraphViewError, match=rf"disease id {disease_id} is outside the graph's "
                                             r"2 raw disease nodes") as refused:
        bind_disease_cases(view, samples)
    assert "MRN" not in str(refused.value)


def test_a_legal_disease_the_view_excludes_is_left_out_with_its_reason():
    from src.models.tasks.scoring import bind_disease_cases

    binding = bind_disease_cases(build_graph_view(_toy()), [
        SimpleNamespace(patient_id="MRN-1", phenotype_ids=[0], disease_id=1)])
    assert binding.cases == [] and binding.excluded == [("MRN-1", "truth_outside_view")]


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
