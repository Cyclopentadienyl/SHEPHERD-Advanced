"""
What the server serves, as the dashboard reads it (M5-2, UI-A).
===============================================================
`src/webui/components/served_status.py`. What these pin:

- **the serving workspace is found through the server's list**, by resolved
  directory, with **one listing of the root for every name** (many workspaces
  cost one listing, not one per name, on every Datasets and model-list read);
- **`served_key` sees every part of what is served**: the checkpoint, the
  graph, the shortest-path table, the scoring settings and the data directory,
  so a reload by another session with the same checkpoint is still a change.

Module: tests/unit/test_webui_served_status.py
"""
from __future__ import annotations

import copy

import pytest

from src.webui.components import served_status as ss


def test_the_serving_workspace_costs_one_listing_of_the_root(tmp_path, monkeypatch):
    from src.api.services import server_lists

    root = tmp_path / "root"
    names = [f"ws{i:03d}" for i in range(60)]
    for name in names:
        (root / name).mkdir(parents=True)
    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(root))
    listings = []
    real = server_lists._workspace_entries
    monkeypatch.setattr(server_lists, "_workspace_entries",
                        lambda r: listings.append(r) or real(r))
    workspaces = [{"name": n} for n in names]
    status = {"current_data_dir": str(root / "ws059")}
    assert ss.serving_workspace(status, workspaces) == "ws059"
    assert len(listings) == 1
    # a caller that already listed the root passes its answer: no listing
    directories = server_lists.resolve_workspaces()
    listings.clear()
    assert ss.serving_workspace(status, workspaces, directories=directories) == "ws059"
    assert listings == []
    assert ss.serving_workspace({"current_data_dir": str(tmp_path)}, workspaces) is None
    assert ss.serving_workspace({}, workspaces) is None


def test_no_root_means_no_serving_workspace(tmp_path, monkeypatch):
    monkeypatch.delenv("SHEPHERD_WORKSPACES_ROOT", raising=False)
    assert ss.serving_workspace({"current_data_dir": str(tmp_path)}, [{"name": "x"}]) is None


BASE = {
    "current_data_dir": "/srv/ws/demo",
    "served_identity": {
        "scoring_mode": "gnn_only",
        "checkpoint": {"state": "verified", "sha256": "a" * 64},
        "graph": {"state": "verified", "kg_sha256": "b" * 64, "split_manifest_sha256": "c" * 64},
        "shortest_paths": {"state": "absent", "build_id": None, "max_hops": None},
        "dataset_record": {"state": "verified", "origin": "synthetic", "sha256": "d" * 64},
        "settings": {"eta": 1.0, "device": "cpu"},
    },
}


@pytest.mark.parametrize("path, value", [
    (("current_data_dir",), "/srv/ws/demo_copy"),
    (("served_identity", "checkpoint", "sha256"), "e" * 64),
    (("served_identity", "graph", "kg_sha256"), "f" * 64),
    (("served_identity", "shortest_paths", "build_id"), "build-2"),
    (("served_identity", "scoring_mode"), "gnn_plus_shortest_path"),
    (("served_identity", "settings", "eta"), 0.7),
    (("served_identity", "dataset_record", "origin"), "files"),
])
def test_served_key_sees_every_part_of_what_is_served(path, value):
    changed = copy.deepcopy(BASE)
    node = changed
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    assert ss.served_key(changed) != ss.served_key(BASE)
    assert ss.served_key(copy.deepcopy(BASE)) == ss.served_key(BASE)
    assert ss.served_key(None) is None
