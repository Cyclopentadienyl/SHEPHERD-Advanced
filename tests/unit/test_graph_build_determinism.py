"""A rebuild must reproduce the workspace, not merely resemble it.

The deployment probe built the same annotation files twice on one machine, in
one process, and got two graphs: identical in content — 54,912 nodes, 615,031
edges, the same 1,419 distinct edge weights — and different in order, starting
at node 32. Canonically sorted, the two files were byte-identical.

Order is not cosmetic here. `KnowledgeGraphBuilder.add_ontology` inserts nodes
in the order the ontology yields them and `KnowledgeGraph.add_node` assigns each
the next integer index, so that order decides **which integer names which
disease**. Those integers are what the graph tensors are indexed by, what the
generated cohorts store, what `allocate_diseases` partitions, and what
`universe_digest` is computed over. Two builds of the same graph therefore
produced:

  * a `kg.json` digest that identified a build rather than a graph, so no
    artifact citing one could be reproduced and no two sites could confirm they
    held the same graph;
  * an allocation that withheld a *different set of diseases* under the same
    seed, because the seed shuffles indices and the indices had moved;
  * a `universe_digest` incomparable across builds.

`pronto` parses large ontologies across a thread pool, which is why it did not
reproduce on the eleven-term fixture and why it varied twice in one process
rather than only across processes. The fix does not depend on that being the
mechanism: the order is sorted where it enters, so it holds whatever the parser
does next.

Module: tests/unit/test_graph_build_determinism.py
"""
from __future__ import annotations

from pathlib import Path

import pytest

pronto = pytest.importorskip("pronto")

from src.ontology.hierarchy import Ontology  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "mini_hpo.obo"


def _load() -> Ontology:
    return Ontology(pronto.Ontology(str(FIXTURE)))


class TestTheOrderIsPartOfTheContract:
    """Asserted as a property, not by racing the parser.

    A test that loaded the ontology repeatedly and hoped to catch the thread
    pool would pass on a small fixture whatever the code did — which is exactly
    what happened when this was first investigated. Sorted output is the
    property that makes the order reproducible, so that is what is checked.
    """

    def test_terms_come_back_sorted(self):
        terms = _load().get_all_terms()

        assert terms, "the fixture yielded no terms"
        assert terms == sorted(terms)

    def test_hierarchy_edges_come_back_sorted(self):
        edges = _load().to_edges()

        assert edges, "the fixture yielded no hierarchy"
        assert edges == sorted(edges)

    def test_obsolete_terms_do_not_disturb_the_order(self):
        """The filter and the sort have to compose: a build that included
        obsolete terms would otherwise index every later term differently."""
        without = _load().get_all_terms(include_obsolete=False)
        with_obsolete = _load().get_all_terms(include_obsolete=True)

        assert without == sorted(without)
        assert with_obsolete == sorted(with_obsolete)
        assert set(without) <= set(with_obsolete)

    def test_repeated_loads_agree(self):
        """Weak on this fixture and kept anyway: it is the statement of intent
        that a reader checks the sort against."""
        first = _load()
        for _ in range(3):
            other = _load()
            assert other.get_all_terms() == first.get_all_terms()
            assert other.to_edges() == first.to_edges()


def test_the_same_terms_produce_the_same_integer_indices():
    """The consequence the sort exists for.

    Insertion order is index assignment. This builds the same nodes twice in
    two orders and asserts the mapping is the same both times — which is only
    true because the builder is handed a sorted list.
    """
    from src.core.types import DataSource, NodeType
    from src.kg.graph import KnowledgeGraph, Node, NodeID

    def build(term_ids):
        kg = KnowledgeGraph()
        for term_id in sorted(term_ids):
            kg.add_node(Node(
                id=NodeID(source=DataSource.MONDO, local_id=term_id),
                node_type=NodeType.DISEASE, name=term_id,
                attributes={"mondo_id": term_id, "name": term_id},
            ))
        return kg.get_node_id_mapping()["disease"]

    ids = ["MONDO:0000671", "MONDO:0000118", "MONDO:0019288", "MONDO:0005638"]
    assert build(ids) == build(list(reversed(ids)))
    # And the sort is what does it: the same nodes in the order the parser
    # happened to yield them gave different integers, which is the defect.
    assert build(ids)["mondo:MONDO:0000118"] == 0
