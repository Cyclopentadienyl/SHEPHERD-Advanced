"""
The inventory before a deployment: what contract M2 refuses, and why (M2.6).
===========================================================================
`audit_checkpoint_family.py --inventory` over a directory of workspaces. The
checkpoints are the session's trained disease child (`trained_disease_child`)
over a copy of its own workspace, and copies of it with one part removed.
Shown:

- a sound workspace's graph and training inputs are accepted, and the child
  passes M2's record checks for serving and measurement and for a resume, and
  carries the model record this build reads, with the digest of the one read;
- a shortest-path table and its sidecar left in a workspace by an earlier build
  are not reported and not listed as a checkpoint: the report has no
  `shortest_paths` entry and no `shortest_paths` count (the inventory's
  shortest-path check was removed in K8-5);
- a copy from before the contract, the child over a same-shaped other workspace,
  and a copy that records only the graph roles (so serves but cannot be resumed)
  are each refused for what M2 refuses them for;
- a copy without a model spec is refused for both uses, by the model record's
  own reason (`read_model_record`), as serving, the task measurements and
  training refuse it;
- a workspace without a manifest is refused, and its checkpoints are not
  judged;
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


def _child_copy(child, path, change):
    """The disease child `child`, read with the safe loader, `change` applied to
    its payload, written to `path`."""
    from src.utils.checkpoint_io import read_checkpoint

    contents = read_checkpoint(child, map_location="cpu", weights_only=True).checkpoint
    change(contents)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(contents, path)


def _graph_roles_only(contents):
    recorded = contents["training_input_digests"]
    contents["training_input_digests"] = {
        role: recorded[role] for role in ("kg", "node_features", "edge_indices", "num_nodes")}


def _bound_table(workspace, kg_digest, max_hops=5):
    """A one-row table and its sidecar bound to `kg_digest`, as the retired
    producer wrote them: a pair an earlier build left in a workspace. The
    sidecar's values are literals; the module that defined them was removed in
    K8-5."""
    build_id = "0123456789abcdef" * 2
    torch.save({"phenotype_idx": torch.zeros(1, dtype=torch.int64),
                "target_idx": torch.zeros(1, dtype=torch.int64),
                "target_type": torch.ones(1, dtype=torch.int64),
                "distance": torch.ones(1, dtype=torch.int8), "build_id": build_id},
               workspace / "shortest_paths.pt")
    (workspace / "shortest_paths.meta.json").write_text(json.dumps({
        "max_hops": max_hops, "schema_version": 1, "build_id": build_id,
        "kg_digest": kg_digest,
    }))


@pytest.fixture
def inventory(tmp_path, trained_disease_child):
    root = tmp_path / "workspaces"
    # The child's own workspace, which holds no run directory (its runs are
    # beside it), so each copy below holds the four checkpoints placed here.
    a = root / "a"
    shutil.copytree(trained_disease_child["data_dir"], a)
    child, placed = trained_disease_child["disease"], a / "checkpoints" / "m2"
    placed.mkdir(parents=True)
    shutil.copy2(child, placed / "good.pt")
    _child_copy(child, placed / "old.pt", lambda c: c.pop("training_input_digests"))
    _child_copy(child, placed / "graph_only.pt", _graph_roles_only)
    _child_copy(child, placed / "no_model_spec.pt", lambda c: c.pop("model_spec"))
    record = _record(a)
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

    d = root / "d"
    shutil.copytree(a, d)
    _bound_table(d, record["kg"], max_hops=0)

    before = _snapshot(root)
    out = tmp_path / "out" / "inventory.json"
    import scripts.audit_checkpoint_family as audit

    assert audit.main(["--inventory", str(root), "--output", str(out)]) == 0
    assert _snapshot(root) == before, "the inventory changed the inventoried directory"
    return root, json.loads(out.read_text()), out.read_text()


def _verdicts(entry, name):
    return {use: v["verdict"] for use, v in entry["checkpoints"][name]["m2_contract"].items()}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_sound_workspace_and_its_recorded_checkpoint_are_accepted(inventory):
    root, report, _ = inventory
    a = report["workspaces"]["a"]
    assert a["graph"]["verdict"] == a["training"]["verdict"] == "accepted"
    assert _verdicts(a, "a/checkpoints/m2/good.pt") == {
        "serve_and_measure": "accepted", "resume": "accepted"}
    assert a["checkpoints"]["a/checkpoints/m2/good.pt"]["sha256"] == _sha(
        root / "a/checkpoints/m2/good.pt")
    # The pair an earlier build left (the fixture puts one in every workspace)
    # is not reported and not listed as a checkpoint; the fixture's snapshot
    # shows it unchanged.
    assert all("shortest_paths" not in entry for entry in report["workspaces"].values())
    assert not [key for key in report["counts"] if key.startswith("shortest_paths")]
    assert (root / "a/shortest_paths.pt").is_file()
    assert "a/shortest_paths.pt" not in a["checkpoints"]


def test_what_m2_refuses_is_listed_with_its_reason(inventory):
    from src.models.bundle.record import HISTORY_ONLY

    _, report, _ = inventory
    a, b = report["workspaces"]["a"], report["workspaces"]["b"]

    old = a["checkpoints"]["a/checkpoints/m2/old.pt"]["m2_contract"]
    assert {v["verdict"] for v in old.values()} == {"refused"}
    assert "records no training_input_digests" in old["serve_and_measure"]["reason"]

    graph_only = a["checkpoints"]["a/checkpoints/m2/graph_only.pt"]["m2_contract"]
    assert graph_only["serve_and_measure"]["verdict"] == "accepted"
    assert graph_only["resume"]["verdict"] == "refused"
    assert "train_samples is not recorded" in graph_only["resume"]["reason"]

    # A file without a model spec records this workspace's graph and inputs,
    # so only its missing model spec refuses it, for both uses, in the model
    # record's own words.
    specless = a["checkpoints"]["a/checkpoints/m2/no_model_spec.pt"]["m2_contract"]
    for use, verdict in specless.items():
        assert verdict["verdict"] == "refused", use
        assert verdict["reason"] == f"ModelRecordError: {HISTORY_ONLY}", use

    crossed = b["checkpoints"]["b/checkpoints/m2/good.pt"]["m2_contract"]
    assert crossed["serve_and_measure"]["verdict"] == "refused"
    assert "kg differs" in crossed["serve_and_measure"]["reason"]


def test_a_refused_workspace_leaves_its_checkpoints_unjudged(inventory):
    _, report, _ = inventory
    c = report["workspaces"]["c"]
    assert c["graph"]["verdict"] == c["training"]["verdict"] == "refused"
    assert set(_verdicts(c, "c/checkpoints/m2/good.pt").values()) == {"not judged"}


def test_the_report_states_its_scope_and_names_nothing_outside(inventory, tmp_path):
    root, report, text = inventory
    assert str(tmp_path) not in text and str(root.resolve()) not in text
    assert "source-contract" in report["checkpoint_verdicts_cover"]
    assert "model spec" in report["checkpoint_verdicts_cover"]
    counts = report["counts"]
    assert counts["checkpoint m2_contract resume:accepted"] == 2      # a and d
