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

The K8-4r2 recheck:

- P2: a record the safe loader reads but the reader could not convert or walk
  still made the list answer HTTP 500: a graph view of infinitely many nodes
  (`int()` raised `OverflowError`), a rule field holding a list that holds
  itself or one nested a thousand deep (`RecursionError`). Each is now refused
  on its own row, its field named, and the healthy file still lists.
- P3: the list kept what a read found under the identity its name gave before
  and after the read, so a name switched to another file for the open and back
  kept that file's digest, or a refusal of it, under the first file. A read is
  now kept only when the file its open descriptor read is the file the name
  gives, and a refusal never.

The K8-4r3 verifier (left open 1 and 2): a lone surrogate in a string a list
shows (a family, a parent, a kind, an end, a dataset fingerprint) or the
reader's reason copied (a compared claim) made the server's list, the family
view and the training service's list answer HTTP 500, and the training list
did so for an epoch of 1.5 or "x", a kind that is not a string or a tensor
among its logs. Each list now answers 200 with that file refused, or listed
unread, naming the field.

The K8-4r4 verifier (defect 1): a file torch reads with its legacy loader is
refused with a message naming the protocol version the file holds, and the
listing copied that message into every list's reason, so a lone surrogate
there made the three lists answer HTTP 500 again. The reason is now written
with each character that is not printable escaped.

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
    _refused_alone_over_http(workspace, trained_disease_child, monkeypatch, field, value, named)


def _refused_alone_over_http(workspace, trained_disease_child, monkeypatch, field, value, named):
    """``GET /api/v1/workspaces/ws/checkpoints`` over the disease child (run A)
    and, beside it, the child with `field` holding `value` (run B), saved and
    read as a real file: 200, run A a model, run B refused with `named` in
    its reason."""
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


def _holding_itself():
    value = []
    value.append(value)
    return value


def _nested(depth):
    value = []
    for _ in range(depth):
        value = [value]
    return value


@pytest.mark.parametrize("field, make, named", [
    ("graph_view.num_nodes", lambda: float("inf"),
     "graph_view.num_nodes must be an integer above zero, not inf"),
    ("data_roles.graph_view.rule.largest_component", _holding_itself,
     "the record's data_roles.graph_view.rule.largest_component[0] is a list that contains "
     "itself"),
    ("data_roles.graph_view.rule.largest_component", lambda: _nested(1000),
     "the record's data_roles.graph_view.rule.largest_component" + "[0]" * 15
     + " nests deeper than 16 levels"),
], ids=["infinite-node-count", "rule-field-holding-itself", "rule-field-nested-1000-deep"])
def test_one_file_the_reader_cannot_convert_or_walk_is_refused_alone_and_the_list_answers(
        workspace, trained_disease_child, monkeypatch, field, make, named):
    """The reviewer's K8-4r2 recheck, P2 (edges.py, ``infinite_node_count``,
    ``cyclic_rule``): a disease child whose graph view counts infinitely many
    nodes, or whose view rule field holds a list that holds itself (or one
    nested a thousand deep), each a real `torch.save` file the safe loader
    reads, beside a healthy child. Before the fix the view's read raised
    `OverflowError` (`int(inf)`), the comparison recursed until
    `RecursionError` (and the list's copy of a deep record did), and the list
    answered HTTP 500. Now it answers 200: that file is refused, its reason
    naming the field and where the walk stopped, and the healthy one is
    listed as a model."""
    _refused_alone_over_http(workspace, trained_disease_child, monkeypatch, field, make(), named)


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


def _retarget(link, target):
    pending = link.with_suffix(".pending")
    pending.symlink_to(target)
    os.replace(pending, link)


@pytest.mark.parametrize("between", ["inside", "outside"])
def test_a_name_switched_to_another_file_for_the_open_and_back_keeps_nothing_under_the_first(
        workspace, trained_disease_child, tmp_path, monkeypatch, api, between):
    """The reviewer's K8-4r2 recheck, P3 (edges.py, ``inside-swap`` and
    ``outside-swap``): run A's ``last.pt`` is a link to a settled valid model
    A in the workspace; at the list's open it is switched to another file B
    (a valid model with another digest inside the workspace, or one outside
    it) and back to A once the file is open. Both looks at the name, before
    and after the read, gave A's identity, so the list kept what it found of
    B (B's digest, or the refusal of the file opened outside) under A, and the
    next list, the name settled on A, showed it without opening anything.
    Now a read is kept only when its open descriptor's file is the name's, and
    a refusal never: the next list opens the name, shows A's digest, reads it
    as a model, and A loads by that digest."""
    import src.utils.checkpoint_paths as checkpoint_paths

    monkeypatch.setattr(checkpoint_paths, "SETTLED_NS", 0)  # stands for settled files
    child = trained_disease_child["disease"]
    held = workspace / "payloads"
    held.mkdir()
    first = held / "A.pt"
    shutil.copyfile(child, first)
    other = (held if between == "inside" else tmp_path / "outside") / "B.pt"
    other.parent.mkdir(exist_ok=True)
    payload = _payload(child)
    payload["state_dict"] = {name: value + 1.0 if value.is_floating_point() else value
                             for name, value in payload["state_dict"].items()}
    torch.save(payload, other)
    sha_a, sha_b = file_sha256(first), file_sha256(other)
    assert sha_a != sha_b
    alias = _filed(workspace, RUN_A)
    alias.symlink_to(first)

    real_open, opens = Path.open, []

    def open_switched(path, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if path == alias and mode == "rb":
            opens.append(path)
            if len(opens) == 1:
                _retarget(alias, other)
                try:
                    return real_open(path, *args, **kwargs)
                finally:
                    _retarget(alias, first)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_switched)
    switched = _listed()[RUN_A]
    settled = _listed()[RUN_A]
    monkeypatch.setattr(Path, "open", real_open)

    if between == "inside":
        assert switched.sha256 == sha_b, "the switch fell at the list's open"
    else:
        assert switched.problem == (
            f"{RUN_A}/last.pt was opened outside the workspace, so it is not read")
    assert alias.resolve() == first.resolve() and file_sha256(alias) == sha_a
    assert len(opens) == 2, "the second list showed a kept read instead of opening the name"
    assert settled.sha256 == sha_a and settled.problem is None
    assert settled.record == "model", settled.record_problem

    from src.api.routes.pipeline import PipelineReloadRequest, run_reload

    loaded = run_reload(PipelineReloadRequest(
        data_dir=str(workspace), checkpoint_path=str(alias), device="cpu",
        expected_checkpoint_sha256=settled.sha256))
    assert loaded.success, loaded.message


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


# --------------------------------------------------------------------------- K8-4r3 verifier
#: The lists that show a workspace's checkpoints over HTTP: the server's list,
#: its family view and the training service's own list.
ENDPOINTS = {"checkpoints": "/api/v1/workspaces/ws/checkpoints",
             "family": "/api/v1/workspaces/ws/family",
             "training": "/api/v1/training/checkpoints"}


def _rows_over_http(workspace, monkeypatch, endpoint):
    """`endpoint`'s answer for the listed workspace, which must be 200, as
    its rows by run (a family row as its checkpoint with its group); the
    training service lists the workspace of its last request."""
    from fastapi.testclient import TestClient

    from src.api.main import app
    from src.api.routes import training as training_routes
    from tests.fixtures.operator_access import OPERATOR_BASE_URL, operator_mode

    operator_mode(monkeypatch)
    monkeypatch.setattr(training_routes.training_manager, "_config",
                        {"data_dir": str(workspace)})
    response = TestClient(app, base_url=OPERATOR_BASE_URL, raise_server_exceptions=False).get(
        ENDPOINTS[endpoint])
    assert response.status_code == 200, response.text[:300]
    body = response.json()
    if endpoint == "checkpoints":
        return {row["run"]: row for row in body["checkpoints"]}
    if endpoint == "family":
        return {row["item"]["run"]: {**row["item"], "group": row["group"],
                                     "parent_problem": row["parent_problem"]}
                for row in body["rows"]}
    return {row["run"]: row for row in body}


def _one_bad_beside_one_good(workspace, trained_disease_child, field, value):
    """The disease child as run A, and beside it the child with `field`
    holding `value` as run B, saved as a real file; returns both paths."""
    child = trained_disease_child["disease"]
    good = _copy(workspace, RUN_A, child)
    payload = _payload(child)
    *parents, leaf = field.split(".")
    part = payload
    for key in parents:
        part = part[key]
    part[leaf] = value
    return good, _save(workspace, RUN_B, payload)


def _listed_alone(rows, endpoint, good, bad, named, by):
    """Run A is listed as a model, and run B, with its digest, either refused
    by the record reader (`by` "record") or listed with its metadata unread
    (`by` "listing"), its reason naming the field (`named`)."""
    assert rows[RUN_A]["record"] == "model" and rows[RUN_A]["sha256"] == file_sha256(good)
    row = rows[RUN_B]
    assert row["sha256"] == file_sha256(bad)
    if by == "record":
        assert row["record"] == "refused" and named in row["record_problem"], row
    else:
        problem = row["metadata_problem" if endpoint == "training" else "problem"]
        assert row["record"] is None and named in problem, row
    if endpoint == "family":
        assert rows[RUN_A]["group"] == "family" and row["group"] == "refused"


@pytest.mark.parametrize("endpoint", list(ENDPOINTS))
@pytest.mark.parametrize("field, value, named, by", [
    ("data_roles.kg", "\udc80" + "0" * 63,
     "the record's data_roles.kg is not text: character 0 is a lone surrogate", "record"),
    ("family_id", "\udc80run",
     "the record's family_id is not text: character 0 is a lone surrogate", "record"),
    ("parent.run", "run\ud800",
     "the record's parent.run is not text: character 3 is a lone surrogate", "record"),
    ("parent.sha256", "\udc80" + "0" * 63,
     "the record's parent.sha256 is not text: character 0 is a lone surrogate", "record"),
    ("checkpoint_kind", "\udc80",
     "its checkpoint_kind is not text: character 0 is a lone surrogate", "listing"),
    ("progress.end", "finished\udc80",
     "its progress.end is not text: character 8 is a lone surrogate", "listing"),
    ("training_input_digests.split_manifest", "\ud800" + "0" * 63,
     "its training_input_digests.split_manifest is not text: character 0 is a lone surrogate",
     "listing"),
], ids=["compared-claim", "family-id", "parent-run", "parent-sha256", "checkpoint-kind",
        "progress-end", "split-manifest"])
def test_one_file_holding_a_string_that_is_not_text_is_refused_alone_by_every_list(
        workspace, trained_disease_child, monkeypatch, endpoint, field, value, named, by):
    """The K8-4r3 verifier, left open 1: one disease child with a lone
    surrogate (a string the safe loader reads and no answer can encode) in a
    field a list shows or the reader's reason copied, beside a healthy child.
    Before the fix the server's list and the family view answered HTTP 500
    for every one of them, and the training service's list for all but
    ``progress.end``, which it listed as a model. Now each list answers 200:
    that file is refused by the record reader, or listed with its metadata
    unread, the reason naming the field without echoing it, and the healthy
    one is listed as a model."""
    good, bad = _one_bad_beside_one_good(workspace, trained_disease_child, field, value)

    rows = _rows_over_http(workspace, monkeypatch, endpoint)

    _listed_alone(rows, endpoint, good, bad, named, by)


@pytest.mark.parametrize("field, value, named", [
    ("epoch", 1.5, "its epoch is a float, not a whole number"),
    ("epoch", "x", "its epoch is a str, not a whole number"),
    ("checkpoint_kind", 5, "its checkpoint_kind is a int, not text"),
    ("logs.val_pooled_mrr", torch.tensor([0.1, 0.2]),
     "its logs entry 'val_pooled_mrr' is a Tensor, not a number"),
], ids=["epoch-1.5", "epoch-x", "checkpoint-kind-5", "logs-tensor"])
def test_one_file_whose_epoch_or_kind_is_not_of_its_kind_is_listed_unread_by_the_training_list(
        workspace, trained_disease_child, monkeypatch, field, value, named):
    """The K8-4r3 verifier, left open 2: the training service's own list put
    a file's epoch and kind in its answer as read, so one file whose epoch is
    1.5 or "x", or whose kind is not a string, failed the answer's types and
    ``GET /api/v1/training/checkpoints`` answered HTTP 500; so did a tensor
    among its logs, which it shows whole. Now the listing reads each value as
    its field's kind or lists that file unread, saying which field, and the
    healthy file is listed as a model."""
    good, bad = _one_bad_beside_one_good(workspace, trained_disease_child, field, value)

    rows = _rows_over_http(workspace, monkeypatch, "training")

    _listed_alone(rows, "training", good, bad, named, "listing")
    assert rows[RUN_B]["epoch"] is None and rows[RUN_B]["checkpoint_kind"] is None


# --------------------------------------------------------------------------- K8-4r4 verifier
#: The first value of a file torch reads with its legacy (non-zip) loader.
_LEGACY_MAGIC = 0x1950A86A20F9469CFC6C


@pytest.mark.parametrize("endpoint", list(ENDPOINTS))
def test_a_refused_read_whose_message_holds_a_string_that_is_not_text_is_listed_alone(
        workspace, trained_disease_child, monkeypatch, endpoint):
    """The K8-4r4 verifier, defect 1 (``legacy-proto-surrogate``): torch reads
    a file that is not a zip archive with its legacy loader even under
    ``weights_only=True``, and refuses one whose protocol version is not its
    own with ``RuntimeError: Invalid protocol version: <the file's value>``.
    The listing copied that message into the file's reason, so a file holding
    a lone surrogate there, in the run the healthy child names as its parent,
    made the server's list, the family view (the file's row and the child's
    ``parent_problem``) and the training service's list answer HTTP 500, at
    the first request and the next. Now each answers 200 at both: that file
    is listed unread with the message escaped, and the child as a model."""
    import pickle

    child = trained_disease_child["disease"]
    good = _copy(workspace, RUN_A, child)
    parent_run = _payload(child)["parent"]["run"]
    _filed(workspace, parent_run).write_bytes(
        pickle.dumps(_LEGACY_MAGIC, protocol=2) + pickle.dumps("\udc80proto", protocol=2))
    message = ("the file cannot be read as a checkpoint; Ref: RuntimeError: Invalid protocol "
               "version: \\udc80proto")

    for request in ("first", "next"):
        rows = _rows_over_http(workspace, monkeypatch, endpoint)

        assert rows[RUN_A]["record"] == "model", request
        assert rows[RUN_A]["sha256"] == file_sha256(good), request
        row = rows[parent_run]
        problem = row["metadata_problem" if endpoint == "training" else "problem"]
        assert row["sha256"] is None and problem == message, (request, row)
        if endpoint == "family":
            assert row["group"] == "refused", request
            assert f"could not be read ({message})" in rows[RUN_A]["parent_problem"], request
