"""The server's model list reads each file as a load of it does (the reviewer's K8-4r recheck).

- P2-A: the list judged a file's name against the workspace, then read it
  unbounded, so a name swapped at its open for a link out of the workspace
  was read and shown. The list's read is now bound to the file it opens
  (`read_checkpoint(within=)`): that file is not read, its row says why, and
  the other files still list.
- P2-B: a record value that is not plain data (a tensor the safe loader
  accepts) made the one reader's comparison raise a PyTorch error, and the
  whole list answered HTTP 500. The reader now compares every value it
  compares as plain data and refuses such a value naming its field, so the
  list refuses that file alone.
- P3-C: the list handed the reader the record without the training record it
  compares the record's claims with, so a file whose KG claims disagree was
  listed as a model while a load refused it. The list now carries every key
  the reader reads (`MODEL_RECORD_READ_KEYS`), from the same one read.

Module: tests/unit/test_listed_checkpoint_reads.py
"""
from __future__ import annotations

import os
import shutil
from collections import OrderedDict
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.config import server_settings as ss  # noqa: E402
from src.utils.fingerprint import file_sha256  # noqa: E402

#: Two runs of the listed workspace, the second the newer.
RUN_A = "20261020-000000_disease_aaaaaa"
RUN_B = "20261021-000000_disease_bbbbbb"


@pytest.fixture
def workspace(trained_disease_child, tmp_path, monkeypatch):
    """A copy of the trained child's workspace, listed by the name ``ws``
    under this test's workspaces root, with no file's read kept yet."""
    import src.utils.checkpoint_paths as checkpoint_paths

    root = tmp_path / "root"
    ws = Path(shutil.copytree(trained_disease_child["data_dir"], root / "ws"))
    monkeypatch.setattr(checkpoint_paths, "_READS", OrderedDict())
    monkeypatch.setenv(ss.ENV_WORKSPACES_ROOT, str(root))
    return ws


def _filed(ws, run):
    path = ws / "runs" / "gatv2" / run / "checkpoints" / "last.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _copy(ws, run, checkpoint):
    path = _filed(ws, run)
    shutil.copyfile(checkpoint, path)
    return path


def _save(ws, run, payload):
    path = _filed(ws, run)
    torch.save(payload, path)
    return path


def _payload(checkpoint):
    return torch.load(checkpoint, map_location="cpu", weights_only=True)


def _listed():
    from src.api.services.server_lists import list_checkpoints

    listing = list_checkpoints("ws")
    assert listing.available, listing.problem
    return {item.run: item for item in listing.items}


# --------------------------------------------------------------------------- P2-A
class _Counted:
    """An open file whose reads are counted."""

    def __init__(self, handle, counted):
        self._handle, self._counted = handle, counted

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._handle.close()

    def __getattr__(self, name):
        return getattr(self._handle, name)

    def read(self, *args, **kwargs):
        data = self._handle.read(*args, **kwargs)
        self._counted.append(len(data))
        return data


def test_a_listed_name_swapped_at_its_open_for_a_file_outside_the_workspace_is_not_read(
        workspace, trained_disease_child, tmp_path, monkeypatch):
    """The reviewer's P2-A (recheck.py, ``listing_race``): the list's boundary
    check passes the run's ``last.pt``; at the list's open of it, the name is
    replaced by a link to another valid disease child outside the workspace.
    Before the fix the list read all of the outside file and showed its
    SHA-256 as a model. Now the open is judged by its descriptor: not a byte
    of the outside file is read, its digest is nowhere in the list, the row
    says why, and the other run's file is listed as a model."""
    child = trained_disease_child["disease"]
    swapped = _copy(workspace, RUN_A, child)
    kept = _copy(workspace, RUN_B, child)
    outside = tmp_path / "outside" / "last.pt"
    outside.parent.mkdir()
    payload = _payload(child)
    payload["state_dict"] = {name: value + 1.0 if value.is_floating_point() else value
                             for name, value in payload["state_dict"].items()}
    torch.save(payload, outside)
    outside_sha = file_sha256(outside)
    assert outside_sha != file_sha256(swapped)

    real_open, opened, read_sizes = Path.open, [], []

    def open_swapped(path, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if path == swapped and mode == "rb" and not opened:
            opened.append(path)
            pending = swapped.with_suffix(".pending")
            pending.symlink_to(outside)
            os.replace(pending, swapped)
            return _Counted(real_open(path, *args, **kwargs), read_sizes)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_swapped)
    rows = _listed()
    monkeypatch.setattr(Path, "open", real_open)

    assert opened == [swapped], "the swap fell at the list's open of the file"
    assert sum(read_sizes) == 0, "the outside file was read"
    assert all(row.sha256 != outside_sha for row in rows.values())
    assert outside_sha not in repr(rows)
    row = rows[RUN_A]
    assert row.problem == f"{RUN_A}/last.pt was opened outside the workspace, so it is not read"
    assert row.sha256 is None and row.size_bytes is None and row.record is None
    other = rows[RUN_B]
    assert other.problem is None and other.record == "model"
    assert other.sha256 == file_sha256(kept)


# --------------------------------------------------------------------------- P2-B
@pytest.mark.parametrize("field, value, named", [
    ("data_roles.graph_view.rule.largest_component", torch.tensor([True, False]),
     "the record's data_roles.graph_view.rule.largest_component is a Tensor"),
    ("record_version", torch.tensor([2, 2]), "the record's record_version is a Tensor"),
    ("model_spec.spec_version", torch.tensor([1, 1]), "model spec version tensor([1, 1])"),
    ("graph_view", torch.tensor([1]), "the graph view record is Tensor, not a mapping"),
], ids=["view-rule-field", "record-version", "spec-version", "graph-view"])
def test_one_file_holding_a_value_that_is_not_plain_data_is_refused_alone_and_the_list_answers(
        workspace, trained_disease_child, monkeypatch, field, value, named):
    """The reviewer's P2-B (recheck.py, ``bad_one_among_good``; http_check.py):
    the disease child with one compared field holding a tensor of two values,
    which the safe loader accepts, beside a healthy child in the same
    workspace. Before the fix the reader's comparison raised "Boolean value of
    Tensor with more than one value is ambiguous" and
    ``GET /api/v1/workspaces/ws/checkpoints`` answered 500, the healthy model
    with it; a top-level ``graph_view`` (which holds a compared digest) that is
    a tensor raised from the view's own read the same way. Now it answers 200:
    that file is refused, the reason naming the field, and the healthy one is
    listed as a model."""
    from fastapi.testclient import TestClient

    from src.api.main import app
    from tests.fixtures.operator_access import OPERATOR_BASE_URL, operator_mode

    child = trained_disease_child["disease"]
    good = _copy(workspace, RUN_A, child)
    payload = _payload(child)
    *parents, leaf = field.split(".")
    part = payload
    for key in parents:
        part = part[key]
    part[leaf] = value
    bad = _save(workspace, RUN_B, payload)
    operator_mode(monkeypatch)

    response = TestClient(app, base_url=OPERATOR_BASE_URL, raise_server_exceptions=False).get(
        "/api/v1/workspaces/ws/checkpoints")

    assert response.status_code == 200, response.text[:300]
    rows = {row["run"]: row for row in response.json()["checkpoints"]}
    assert rows[RUN_A]["record"] == "model" and rows[RUN_A]["record_problem"] is None
    assert rows[RUN_A]["sha256"] == file_sha256(good)
    assert rows[RUN_B]["record"] == "refused" and rows[RUN_B]["sha256"] == file_sha256(bad)
    assert named in rows[RUN_B]["record_problem"], rows[RUN_B]["record_problem"]


# --------------------------------------------------------------------------- P3-C
@pytest.fixture
def api(monkeypatch):
    """App state as a fresh process has it, and no deployment in the environment."""
    import src.api.main as api_main

    for name in ("pipeline", "kg", "_current_data_dir", "_current_checkpoint_path"):
        monkeypatch.setattr(api_main.app_state, name, None, raising=False)
    monkeypatch.setattr(api_main.app_state, "is_ready", False, raising=False)
    for name in ("SHEPHERD_KG_PATH", "SHEPHERD_DATA_DIR", "SHEPHERD_CHECKPOINT_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SHEPHERD_DEVICE", "cpu")
    return api_main


def test_a_file_whose_kg_claims_disagree_is_refused_alike_by_the_list_and_the_load(
        workspace, trained_disease_child, api):
    """The reviewer's P3-C (recheck.py, ``contradictory_case``): the disease
    child with `data_roles.kg` changed and its training record's KG digest
    kept. Before the fix the list showed it as a model (``record_problem``
    null) while the reload refused it. Now the list and the load refuse the
    same file for the same contradiction, in the reader's one sentence."""
    from src.api.routes.pipeline import PipelineReloadRequest, run_reload

    payload = _payload(trained_disease_child["disease"])
    recorded = payload["training_input_digests"]["kg"]
    payload["data_roles"]["kg"] = "0" * 64
    path = _save(workspace, RUN_A, payload)
    phrase = (f"data_roles.kg {'0' * 12}... differs from "
              f"training_input_digests.kg {recorded[:12]}...")

    row = _listed()[RUN_A]
    result = run_reload(PipelineReloadRequest(
        data_dir=str(workspace), checkpoint_path=str(path), device="cpu",
        expected_checkpoint_sha256=row.sha256))

    assert row.sha256 == file_sha256(path) and row.problem is None
    assert row.record == "refused" and phrase in row.record_problem, row.record_problem
    assert result.success is False and phrase in result.message, result.message
    assert api.app_state.pipeline is None


def test_two_children_trained_on_different_cases_both_list_and_measure(
        workspace, trained_disease_child, tmp_path, monkeypatch):
    """The other side of P3-C: the check compares a file with itself only. A
    second disease child of the same base, trained on another workspace's
    cases over the same graph, names its inputs one way, as the first does,
    so the list shows both as models beside each other, and each is measured
    on its own workspace's validation cases."""
    import measure_scorer
    import train_model

    from scripts.setup_demo import demo_task_training
    from tests.fixtures.registered_cohort import write_demo_workspace

    first = trained_disease_child["disease"]
    other_ws = write_demo_workspace(tmp_path / "other", sample_seed=12345)
    run = tmp_path / "other-runs" / RUN_B
    fields = dict(data_dir=str(other_ws), run_dir=str(run), learning_stage="finetune",
                  task="disease", finetune_from=str(trained_disease_child["base"]),
                  num_epochs=1, save_top_k=1, device="cpu", **demo_task_training()["disease"])
    train_model.train(train_model.TrainConfig(**fields), set(fields) - {"data_dir", "run_dir"})
    second = run / "checkpoints" / "last.pt"
    one, two = _payload(first), _payload(second)
    assert one["data_roles"]["kg"] == two["data_roles"]["kg"]
    assert (one["training_input_digests"]["train_samples"]
            != two["training_input_digests"]["train_samples"])

    listed_first, listed_second = _copy(workspace, RUN_A, first), _copy(workspace, RUN_B, second)
    rows = _listed()

    for run_name, path in ((RUN_A, listed_first), (RUN_B, listed_second)):
        assert rows[run_name].record == "model", rows[run_name].record_problem
        assert rows[run_name].sha256 == file_sha256(path)
    for path, data_dir in ((listed_first, workspace), (listed_second, other_ws)):
        out = tmp_path / f"{path.parent.parent.name}.json"
        assert measure_scorer.main(["--mode", "disease", "--checkpoint", str(path),
                                    "--data-dir", str(data_dir), "--split", "val",
                                    "--output", str(out), "--device", "cpu"]) == 0
        assert out.exists()
