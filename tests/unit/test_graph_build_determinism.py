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


class TestTheFeatureDrawIsSeeded:
    """Sorting made the graph reproducible; the features were still random.

    `export_graph_data` drew node features from the global RNG, so a rebuild
    reproduced the graph and not the workspace. Two sites building from the same
    HPO release could confirm they held the same graph and still had to copy
    tensors to compare anything trained on it.

    The seed is a parameter, not a constant in the call: sharing an
    initialisation across rebuilds is the point, and a caller who wants a
    different one says so. It is deliberately unrelated to the allocation seed —
    those are independent streams, and coupling them would make changing a
    budget move the features.
    """

    @staticmethod
    def _kg():
        from src.core.types import DataSource, NodeType
        from src.kg.graph import KnowledgeGraph, Node, NodeID

        kg = KnowledgeGraph()
        for i in range(6):
            kg.add_node(Node(
                id=NodeID(source=DataSource.HPO, local_id=f"HP:{i:07d}"),
                node_type=NodeType.PHENOTYPE, name=f"p{i}",
                attributes={"hpo_id": f"HP:{i:07d}", "name": f"p{i}"},
            ))
        return kg

    def test_the_same_seed_draws_the_same_features(self):
        torch = pytest.importorskip("torch")

        first = self._kg().export_graph_data(feature_dim=8)
        second = self._kg().export_graph_data(feature_dim=8)

        assert torch.equal(first["x_dict"]["phenotype"], second["x_dict"]["phenotype"])

    def test_another_seed_draws_other_features(self):
        """Otherwise the test above would pass against a constant."""
        torch = pytest.importorskip("torch")

        first = self._kg().export_graph_data(feature_dim=8)
        other = self._kg().export_graph_data(feature_dim=8, feature_seed=7)

        assert not torch.equal(first["x_dict"]["phenotype"], other["x_dict"]["phenotype"])

    def test_the_global_random_stream_is_left_alone(self):
        """Seeding `torch.manual_seed` would make a graph export change what
        every later draw in the process produces — a side effect on a caller who
        asked for a file."""
        torch = pytest.importorskip("torch")

        torch.manual_seed(1)
        expected = torch.randn(3)
        torch.manual_seed(1)
        self._kg().export_graph_data(feature_dim=8)
        observed = torch.randn(3)

        assert torch.equal(expected, observed)

    @pytest.mark.parametrize(
        "seed,message",
        [
            (True, "must be an integer"),
            (1.5, "must be an integer"),
            ("42", "must be an integer"),
            (2 ** 64, "range torch's generator accepts"),
            (-(2 ** 63) - 1, "range torch's generator accepts"),
        ],
        ids=["bool", "float", "str", "above", "below"],
    )
    def test_a_seed_torch_would_refuse_is_refused_first(self, seed, message):
        """The bound is torch's, measured rather than invented: outside it
        `manual_seed` raises `Overflow when unpacking long long` — from inside
        the export, with `kg.json` already written."""
        from src.kg.graph import validate_feature_seed

        with pytest.raises(ValueError, match=message):
            validate_feature_seed(seed)

    @pytest.mark.parametrize("seed", [0, -1, 2 ** 64 - 1, -(2 ** 63)])
    def test_the_edges_of_that_range_are_accepted(self, seed):
        torch = pytest.importorskip("torch")

        from src.kg.graph import validate_feature_seed

        validate_feature_seed(seed)
        torch.Generator().manual_seed(seed)

    def test_a_refused_seed_writes_nothing(self, tmp_path):
        """The same ordering rule as every other writer input: the export runs
        after `kg.json` is saved, so a seed rejected at the point of use leaves
        a workspace half written."""
        from src.kg.workspace import SampleBudget, WorkspaceRefusal, write_workspace

        workspace = tmp_path / "ws"
        with pytest.raises(WorkspaceRefusal, match="feature_seed"):
            write_workspace(
                self._kg(), workspace, feature_dim=8, feature_seed=2 ** 64,
                samples=SampleBudget(num_train=2, num_val=1),
            )

        assert not workspace.exists()
