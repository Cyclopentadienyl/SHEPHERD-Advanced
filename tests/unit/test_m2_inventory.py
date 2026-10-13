"""
The inventory before a deployment: what contract M2 refuses, and why (M2.6).
===========================================================================
`audit_checkpoint_family.py --inventory` over a directory of workspaces. Shown:

- a sound workspace's graph, training inputs, bound table and recorded
  checkpoint are accepted, for serving and measurement and for a resume;
- a checkpoint from before the contract, one from a same-shaped other
  workspace, and one that records only the graph roles (so serves but cannot be
  resumed) are each refused for what M2 refuses them for;
- an unbound table and a workspace without a manifest are refused, and that
  workspace's checkpoints are not judged;
- it is read-only: every file under the inventoried directory has the same bytes
  afterwards and nothing is added or removed; the report carries no absolute path.

Module: tests/unit/test_m2_inventory.py
"""
from __future__ import annotations

import hashlib
import json
import shutil

import pytest

torch = pytest.importorskip("torch")

from tests.fixtures.generated_workspace import write_generated_workspace  # noqa: E402

IDS = {"train_ids": [0, 1, 2], "val_ids": [3]}


def _snapshot(root):
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


def _record(workspace):
    from scripts.train_model import read_training_inputs, training_input_digests

    return training_input_digests(read_training_inputs(workspace))


def _checkpoint(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = {"state_dict": {"w": torch.zeros(2)}, "epoch": 1}
    if record is not None:
        contents["training_input_digests"] = record
    torch.save(contents, path)


def _bound_table(workspace, kg_digest):
    from src.inference.sp_artifact import SP_SCHEMA_VERSION

    build_id = "0123456789abcdef" * 2
    torch.save({"distance": torch.ones(1, dtype=torch.int8), "build_id": build_id},
               workspace / "shortest_paths.pt")
    (workspace / "shortest_paths.meta.json").write_text(json.dumps({
        "max_hops": 5, "schema_version": SP_SCHEMA_VERSION, "build_id": build_id,
        "kg_digest": kg_digest,
    }))


@pytest.fixture
def inventory(tmp_path):
    root = tmp_path / "workspaces"
    a = write_generated_workspace(root / "a", training_fields=True, **IDS)[0]
    record = _record(a)
    _checkpoint(a / "checkpoints" / "hgt" / "good.pt", record)
    _checkpoint(a / "checkpoints" / "hgt" / "old.pt", None)
    graph_only = {role: record[role] for role in ("kg", "node_features", "edge_indices",
                                                  "num_nodes")}
    _checkpoint(a / "checkpoints" / "hgt" / "graph_only.pt", graph_only)
    _bound_table(a, record["kg"])

    b = root / "b"
    shutil.copytree(a, b)
    (b / "split_manifest.json").unlink()
    (b / "kg.json").write_text(json.dumps({"workspace": "b"}))
    write_generated_workspace(b, training_fields=True, **IDS)
    (b / "shortest_paths.meta.json").write_text(json.dumps({"max_hops": 5}))

    c = root / "c"
    shutil.copytree(a, c)
    (c / "split_manifest.json").unlink()

    before = _snapshot(root)
    out = tmp_path / "out" / "inventory.json"
    import scripts.audit_checkpoint_family as audit

    assert audit.main(["--inventory", str(root), "--output", str(out)]) == 0
    assert _snapshot(root) == before, "the inventory changed the inventoried directory"
    return root, json.loads(out.read_text()), out.read_text()


def _verdicts(entry, name):
    return {use: v["verdict"] for use, v in entry["checkpoints"][name].items()}


def test_a_sound_workspace_and_its_recorded_checkpoint_are_accepted(inventory):
    _, report, _ = inventory
    a = report["workspaces"]["a"]
    assert a["graph"]["verdict"] == a["training"]["verdict"] == "accepted"
    assert a["shortest_paths"]["verdict"] == "accepted"
    assert a["shortest_paths"]["tensor_checked"] is False
    assert _verdicts(a, "a/checkpoints/hgt/good.pt") == {
        "serve_and_measure": "accepted", "resume": "accepted"}


def test_what_m2_refuses_is_listed_with_its_reason(inventory):
    _, report, _ = inventory
    a, b = report["workspaces"]["a"], report["workspaces"]["b"]

    old = a["checkpoints"]["a/checkpoints/hgt/old.pt"]
    assert {v["verdict"] for v in old.values()} == {"refused"}
    assert "records no training_input_digests" in old["serve_and_measure"]["reason"]

    graph_only = a["checkpoints"]["a/checkpoints/hgt/graph_only.pt"]
    assert graph_only["serve_and_measure"]["verdict"] == "accepted"
    assert graph_only["resume"]["verdict"] == "refused"
    assert "train_samples is not recorded" in graph_only["resume"]["reason"]

    crossed = b["checkpoints"]["b/checkpoints/hgt/good.pt"]
    assert crossed["serve_and_measure"]["verdict"] == "refused"
    assert "kg differs" in crossed["serve_and_measure"]["reason"]
    assert b["shortest_paths"]["verdict"] == "refused"
    assert "unrecorded" in b["shortest_paths"]["reason"]


def test_a_refused_workspace_leaves_its_checkpoints_unjudged(inventory):
    _, report, _ = inventory
    c = report["workspaces"]["c"]
    assert c["graph"]["verdict"] == c["training"]["verdict"] == "refused"
    assert c["shortest_paths"]["verdict"] == "not judged"
    assert {v["verdict"] for v in c["checkpoints"]["c/checkpoints/hgt/good.pt"].values()} \
        == {"not judged"}


def test_the_report_names_nothing_outside_the_inventoried_directory(inventory, tmp_path):
    root, report, text = inventory
    assert str(tmp_path) not in text and str(root.resolve()) not in text
    assert report["counts"]["checkpoint resume:accepted"] == 1
