"""
`KnowledgeGraph.read_json`: the graph and its identity from one read.
=====================================================================
Contract M2.1, S3. `read_json` parses `kg.json` from the bytes `read_once`
returned and returns the graph with that read's `ReadIdentity`, so a caller that
records which graph it used names the bytes it parsed. `load_json` delegates to
it for callers that record nothing.

Module: tests/unit/test_graph_read_json.py
"""
from __future__ import annotations

import hashlib

import pytest

from src.kg.graph import GraphRead, KnowledgeGraph
from src.utils.fingerprint import ReadIdentity
from tests.fixtures.opens import count_opens
from tests.fixtures.release import check_release
from tests.fixtures.replacement import hook_read_once, replace_after_read


def _graph(extra_nodes: int = 0) -> KnowledgeGraph:
    from scripts.setup_demo import build_demo_kg
    from src.core.types import DataSource, Node, NodeID, NodeType

    kg = build_demo_kg()
    for i in range(extra_nodes):
        kg.add_node(Node(id=NodeID(source=DataSource.MONDO, local_id=f"MONDO:extra{i}"),
                         node_type=NodeType.DISEASE, name=f"extra {i}"))
    return kg


def _signature(kg: KnowledgeGraph):
    return kg.total_nodes, kg.total_edges, sorted(map(str, kg._nodes))


@pytest.fixture
def kg_path(tmp_path):
    path = tmp_path / "kg.json"
    _graph().save_json(str(path))
    return path


def test_the_identity_is_the_digest_of_the_bytes_parsed(kg_path):
    read = KnowledgeGraph.read_json(kg_path)

    assert isinstance(read, GraphRead)
    assert read.identity == ReadIdentity(
        path=kg_path, sha256=hashlib.sha256(kg_path.read_bytes()).hexdigest()
    )
    assert _signature(read.kg) == _signature(_graph())


def test_the_file_is_opened_once(kg_path):
    with count_opens() as opens:
        KnowledgeGraph.read_json(kg_path)

    assert opens.opens(kg_path) == 1


def test_load_json_is_the_same_graph_from_the_same_one_read(monkeypatch, kg_path):
    """It delegates: the one open is `read_once`'s, so the two cannot drift apart."""
    seen = []
    hook_read_once(monkeypatch, on_return=seen.append)

    with count_opens() as opens:
        kg = KnowledgeGraph.load_json(str(kg_path))

    assert opens.opens(kg_path) == 1
    assert seen == [kg_path]
    assert _signature(kg) == _signature(KnowledgeGraph.read_json(kg_path).kg)


def test_a_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        KnowledgeGraph.read_json(tmp_path / "kg.json")


@pytest.mark.parametrize("how", ["rename", "rewrite"])
def test_a_replacement_after_the_read_changes_neither_graph_nor_identity(
    monkeypatch, tmp_path, kg_path, how
):
    other = tmp_path / "other.json"
    _graph(extra_nodes=3).save_json(str(other))
    original = kg_path.read_bytes()
    record = replace_after_read(monkeypatch, after=kg_path, target=kg_path,
                                data=other.read_bytes(), how=how)

    read = KnowledgeGraph.read_json(kg_path)

    assert record.fired == 1
    assert kg_path.read_bytes() == other.read_bytes()
    assert read.identity.sha256 == hashlib.sha256(original).hexdigest()
    assert _signature(read.kg) == _signature(_graph())


def test_the_buffer_is_released(monkeypatch, kg_path):
    read, reads = check_release(monkeypatch, lambda: KnowledgeGraph.read_json(kg_path))

    assert reads == (kg_path,)
    assert read.kg.total_nodes == _graph().total_nodes
