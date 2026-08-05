"""
SubgraphSampler._build_subgraph — vectorized path equals the legacy loop.
=========================================================================
``src/kg/data_loader.py`` states in the source that the vectorized induced-subgraph
construction is bit-identical to the legacy Python loop, and ``fast_subgraph_build``
defaults to True, so the vectorized path is what training actually runs. Until now
that guarantee was checked only by ``scripts/spikes/validate_fast_subgraph.py``, a
manual script needing a real workspace — nothing in ``make check`` covered it.

These tests need no data: a synthetic heterogeneous graph exercises the same code.

One trap this file is written around: ``_build_subgraph`` catches any exception from
the vectorized builder and falls back to the legacy loop with a warning. A test that
only compared the two configurations would therefore pass trivially if the fast path
were broken — it would be comparing legacy against legacy. Every case below asserts
that the fallback did not fire.
"""
import logging

import pytest

torch = pytest.importorskip("torch")

from src.kg.data_loader import DataLoaderConfig, SubgraphSampler  # noqa: E402

FALLBACK_MARKER = "fast_subgraph_build failed"

NUM_NODES = {"phenotype": 40, "gene": 30, "disease": 20}

EDGE_SPECS = [
    ("phenotype", "associated_with", "gene", 120),
    ("gene", "causes", "disease", 60),
    ("disease", "has_phenotype", "phenotype", 90),
    # Same source and destination type: the orig->local map is shared, which the
    # two implementations reach by different routes.
    ("gene", "interacts", "gene", 45),
]


def _build_sampler(seed: int = 0) -> SubgraphSampler:
    g = torch.Generator().manual_seed(seed)
    edge_index_dict = {}
    for src_type, rel, dst_type, n_edges in EDGE_SPECS:
        src = torch.randint(NUM_NODES[src_type], (n_edges,), generator=g)
        dst = torch.randint(NUM_NODES[dst_type], (n_edges,), generator=g)
        edge_index_dict[(src_type, rel, dst_type)] = torch.stack([src, dst])
    return SubgraphSampler(edge_index_dict, dict(NUM_NODES), DataLoaderConfig())


def _node_sets(case: str, seed: int = 1):
    """Node selections that hit the branches the two implementations treat differently."""
    g = torch.Generator().manual_seed(seed)

    def pick(node_type: str, fraction: float):
        n = NUM_NODES[node_type]
        k = int(n * fraction)
        return set(torch.randperm(n, generator=g)[:k].tolist())

    if case == "dense":
        return {nt: pick(nt, 0.7) for nt in NUM_NODES}
    if case == "sparse":
        return {nt: pick(nt, 0.15) for nt in NUM_NODES}
    if case == "full":
        return {nt: set(range(n)) for nt, n in NUM_NODES.items()}
    if case == "one_type_empty":
        # An empty node type makes the legacy loop short-circuit to an empty (2, 0)
        # tensor; the vectorized path has to produce the same shape and dtype.
        sets = {nt: pick(nt, 0.5) for nt in NUM_NODES}
        sets["phenotype"] = set()
        return sets
    if case == "all_empty":
        return {nt: set() for nt in NUM_NODES}
    if case == "single_node_each":
        return {nt: {0} for nt in NUM_NODES}
    raise AssertionError(f"unknown case {case!r}")


def _run(sampler: SubgraphSampler, sampled, fast: bool, caplog):
    """Run one configuration and fail loudly if the vectorized path silently fell back."""
    sampler.config.fast_subgraph_build = fast
    with caplog.at_level(logging.WARNING, logger="src.kg.data_loader"):
        result = sampler._build_subgraph(sampled)
    if fast:
        assert FALLBACK_MARKER not in caplog.text, (
            "the vectorized builder raised and fell back to the legacy loop, so this "
            f"comparison would be legacy-vs-legacy: {caplog.text}"
        )
    return result


CASES = ["dense", "sparse", "full", "one_type_empty", "all_empty", "single_node_each"]


@pytest.mark.parametrize("case", CASES)
def test_vectorized_matches_legacy(case, caplog):
    """Same nodes, same edges, same local index mappings — element for element."""
    sampler = _build_sampler()
    sampled = _node_sets(case)

    nodes_f, edges_f, map_f = _run(sampler, sampled, True, caplog)
    nodes_l, edges_l, map_l = _run(sampler, sampled, False, caplog)

    assert set(nodes_f) == set(nodes_l)
    for nt in nodes_f:
        assert torch.equal(nodes_f[nt], nodes_l[nt]), f"nodes differ for {nt}"

    assert set(edges_f) == set(edges_l), "edge-type keys differ"
    for et in edges_f:
        # torch.equal compares shape as well as values, which is the point for the
        # empty cases: (2, 0) and (0,) both "contain no edges" but are not the same
        # thing to a downstream consumer.
        assert edges_f[et].shape == edges_l[et].shape, f"edge shape differs for {et}"
        assert edges_f[et].dtype == edges_l[et].dtype, f"edge dtype differs for {et}"
        assert torch.equal(edges_f[et], edges_l[et]), f"edges differ for {et}"

    assert set(map_f) == set(map_l)
    for nt in map_f:
        assert torch.equal(map_f[nt], map_l[nt]), f"mapping tensor differs for {nt}"


@pytest.mark.parametrize("seed", [0, 7, 99])
def test_vectorized_matches_legacy_across_graphs(seed, caplog):
    """The equivalence is a property of the algorithms, not of one graph."""
    sampler = _build_sampler(seed=seed)
    sampled = _node_sets("dense", seed=seed + 1)

    _, edges_f, _ = _run(sampler, sampled, True, caplog)
    _, edges_l, _ = _run(sampler, sampled, False, caplog)

    for et in set(edges_f) | set(edges_l):
        assert torch.equal(edges_f[et], edges_l[et]), f"edges differ for {et} at seed {seed}"


def test_local_indices_follow_sorted_node_order(caplog):
    """The property the vectorized builder relies on, asserted directly.

    Both implementations must number a subgraph's nodes by sorted original index.
    If that ever stopped holding, the two paths could still agree with each other
    while both disagreeing with what the model was trained against.
    """
    sampler = _build_sampler()
    sampled = _node_sets("sparse")

    nodes, edges, _ = _run(sampler, sampled, True, caplog)

    for nt, tensor in nodes.items():
        expected = sorted(sampled[nt])
        assert tensor.tolist() == expected, f"{nt} nodes are not in sorted original order"

    # Every local index emitted must be addressable in the corresponding node tensor.
    for (src_type, _, dst_type), edge_index in edges.items():
        if edge_index.numel() == 0:
            continue
        assert int(edge_index[0].max()) < nodes[src_type].numel()
        assert int(edge_index[1].max()) < nodes[dst_type].numel()
        assert int(edge_index.min()) >= 0
