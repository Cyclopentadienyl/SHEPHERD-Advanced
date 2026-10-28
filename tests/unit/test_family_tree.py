"""The family view: each child under the exact parent file it records (README
§1, "Parents and the family tree"; provenance §9.3, "Parents in another
workspace").

`family_rows` places a file under the listed file whose SHA-256 its record
names, never under that run's other files; groups families by the family id of
their topmost file, the newest family first; says where a parent not listed
here is (another listed workspace, or unavailable with why); lists history and
refused files after the families; and, filtered by task or stage, keeps a
child's ancestors as context rows. `list_family` looks a missing parent up in
the other listed workspaces' run of the recorded name only, and the API offers
it behind D8 as it offers the list.

Module: tests/unit/test_family_tree.py
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from src.api.services import server_lists as sl
from src.config import server_settings as ss


def _item(run, filename="last.pt", *, sha, stage="base", task=None, parent=None,
          family="fam-a", record="model", problem=None):
    return sl.CheckpointItem(
        workspace="ws", architecture="gatv2", run=run, filename=filename, sha256=sha,
        size_bytes=1, checkpoint_kind="training_state", epoch=1, validation=None,
        dataset_fingerprint=None, trained_on_this_dataset=None, problem=problem,
        record=record, learning_stage=stage if record == "model" else None,
        task=task, operation=("new" if parent is None else "finetune") if record == "model"
        else None, family_id=family if record == "model" else None,
        parent_sha256=parent[0] if parent else None, parent_run=parent[1] if parent else None,
        encoder="gatv2" if record == "model" else None)


def _shape(rows):
    return [(r.group, r.item.run, r.item.filename, r.depth, r.parent, r.parent_state,
             r.context) for r in rows]


#: Listing order (latest run first, newest file first): a base run with two
#: files, its disease and gene children recording its top-k file (not its
#: last.pt), a branch of the disease child, a second family, history and a
#: refused file.
ITEMS = (
    _item("branch", sha="e" * 64, stage="finetune", task="disease",
          parent=("c" * 64, "disease")),
    _item("other-base", sha="f" * 64, family="fam-b"),
    _item("gene", sha="d" * 64, stage="finetune", task="gene", parent=("b" * 64, "base")),
    _item("disease", sha="c" * 64, stage="finetune", task="disease",
          parent=("b" * 64, "base")),
    _item("base", "last.pt", sha="a" * 64),
    _item("base", "model-3.pt", sha="b" * 64),
    _item("old-gnn", sha="9" * 64, record="history"),
    _item("torn", sha=None, record=None, problem="not a checkpoint record"),
)


def test_children_sit_under_the_exact_file_they_record():
    rows = sl.family_rows(ITEMS)
    assert _shape(rows) == [
        ("family", "base", "last.pt", 0, None, None, False),
        ("family", "base", "model-3.pt", 0, None, None, False),
        ("family", "gene", "last.pt", 1, "gatv2/base/model-3.pt", "listed", False),
        ("family", "disease", "last.pt", 1, "gatv2/base/model-3.pt", "listed", False),
        ("family", "branch", "last.pt", 2, "gatv2/disease/last.pt", "listed", False),
        ("family", "other-base", "last.pt", 0, None, None, False),
        ("history", "old-gnn", "last.pt", 0, None, None, False),
        ("refused", "torn", "last.pt", 0, None, None, False),
    ]
    assert {r.family_id for r in rows if r.group == "family"} == {"fam-a", "fam-b"}
    assert [r.family_id for r in rows[:5]] == ["fam-a"] * 5, "the newest family first"


def test_a_task_filter_keeps_the_ancestors_as_context():
    disease = sl.family_rows(ITEMS, task="disease")
    assert _shape(disease) == [
        ("family", "base", "model-3.pt", 0, None, None, True),
        ("family", "disease", "last.pt", 1, "gatv2/base/model-3.pt", "listed", False),
        ("family", "branch", "last.pt", 2, "gatv2/disease/last.pt", "listed", False),
    ]
    bases = sl.family_rows(ITEMS, learning_stage="base")
    assert [(r.item.run, r.item.filename, r.context) for r in bases] == [
        ("base", "last.pt", False), ("base", "model-3.pt", False), ("other-base", "last.pt", False)]
    assert sl.family_rows(ITEMS, task="patients_like_me") == ()


def test_a_parent_not_listed_here_says_where_it_is():
    """Its run listed but those bytes gone; found in another workspace; that
    workspace's run rewritten; nowhere listed."""
    gone = _item("child-1", sha="1" * 64, stage="finetune", task="gene",
                 parent=("0" * 64, "base"))
    away = _item("child-2", sha="2" * 64, stage="finetune", task="disease",
                 parent=("7" * 64, "far-base"))
    asked = []

    def lookup(run, sha256):
        asked.append((run, sha256))
        return {"7" * 64: ("found", "ws-b")}.get(sha256, ("absent", None))

    rows = {r.item.run: r for r in sl.family_rows(
        (gone, away, _item("base", sha="a" * 64)), lookup=lookup)}
    assert (rows["child-1"].parent_state, rows["child-1"].depth) == ("unavailable", 0)
    assert rows["child-1"].parent_problem == (
        "the run base no longer holds the parent's bytes (000000000000): rewritten or removed")
    assert (rows["child-2"].parent_state, rows["child-2"].parent_workspace) == (
        "elsewhere", "ws-b")
    assert asked == [("far-base", "7" * 64)], "a parent whose run is listed is not looked up"
    changed = sl.family_rows((away,), lookup=lambda run, sha256: ("changed", "ws-b"))[0]
    assert changed.parent_state == "unavailable" and changed.parent_workspace == "ws-b"
    assert "the run far-base in the workspace 'ws-b' no longer holds" in changed.parent_problem
    absent = sl.family_rows((away,))[0]
    assert absent.parent_problem == (
        "the parent run far-base (777777777777) is not in a listed workspace")


def test_a_parent_whose_run_holds_an_unread_file_is_not_called_rewritten():
    """The run listed here holds a file its listing could not read: whether it
    is the parent cannot be said, so the reason says that, not "rewritten or
    removed". A parent recorded without its run is never looked up."""
    child = _item("child", sha="1" * 64, stage="finetune", task="gene",
                  parent=("0" * 64, "base"))
    unread = _item("base", sha=None, record=None, problem="the read was refused")
    row = next(r for r in sl.family_rows((child, unread)) if r.item.run == "child")
    assert row.parent_state == "unavailable"
    assert row.parent_problem == (
        "the run base holds files that could not be read (the read was refused), so "
        "whether one is the parent (000000000000) cannot be said")
    elsewhere = sl.family_rows(
        (replace(child, parent_run="far"),), lookup=lambda run, sha256: ("unread", "ws-b"))[0]
    assert (elsewhere.parent_state, elsewhere.parent_workspace) == ("unavailable", "ws-b")
    assert elsewhere.parent_problem.startswith(
        "the run far in the workspace 'ws-b' holds files that could not be read")
    asked = []
    no_run = sl.family_rows((replace(child, parent_run=None),),
                            lookup=lambda run, sha256: asked.append(run) or ("absent", None))[0]
    assert asked == [] and no_run.parent_state == "unavailable"
    assert no_run.parent_problem == "the parent (000000000000) is recorded without its run"


def test_a_parent_copied_within_the_workspace_is_the_recorded_runs():
    """Identical bytes in two runs: the child sits under the run it records."""
    copy = replace(ITEMS[5], run="copy-of-base")
    rows = sl.family_rows((ITEMS[3], copy, ITEMS[5]))
    child = next(r for r in rows if r.item.run == "disease")
    assert child.parent == "gatv2/base/model-3.pt"


def test_a_record_naming_its_descendant_never_loops():
    a = _item("a", sha="a" * 64, stage="finetune", task="disease", parent=("b" * 64, "b"))
    b = _item("b", sha="b" * 64, stage="finetune", task="disease", parent=("a" * 64, "a"))
    rows = sl.family_rows((a, b))
    assert sorted(r.item.run for r in rows) == ["a", "b"]


# ---------------------------------------------------------------------------
# Across workspaces, with real checkpoints
# ---------------------------------------------------------------------------
BASE_RUN = "20261001-000000_base_aaaaaa"


@pytest.fixture
def bundles():
    """The bundle checkpoint writers of `test_task_first_selection` (torch)."""
    pytest.importorskip("torch")
    from tests.unit import test_task_first_selection as written

    assert written.BASE_RUN == BASE_RUN
    return written


def test_a_parent_in_another_listed_workspace_is_found_by_its_run_and_bytes(tmp_path, bundles):
    from src.utils.fingerprint import file_sha256

    _put, _record_payload, GNN_PAYLOAD = bundles._put, bundles._record_payload, bundles.GNN_PAYLOAD
    root = tmp_path / "root"
    base = _put(root / "a", "gatv2", BASE_RUN, _record_payload(), 100)
    parent = {"sha256": file_sha256(base), "run": BASE_RUN}
    _put(root / "b", "gatv2", "20261002-000000_disease_bbbbbb",
         _record_payload("disease", parent=parent), 200)
    _put(root / "b", "gat", "20261001-000000_gnn_aaaaaa", GNN_PAYLOAD, 50)
    environ = {ss.ENV_WORKSPACES_ROOT: str(root)}

    listing = sl.list_family("b", environ=environ)
    assert listing.available, listing.problem
    first, history = listing.items
    assert (first.item.filename, first.parent_state, first.parent_workspace) == (
        "last.pt", "elsewhere", "a")
    assert (history.group, history.item.architecture) == ("history", "gat")

    _put(root / "a", "gatv2", BASE_RUN, _record_payload(logs={"val_lp_accuracy": 0.9}), 300)
    rewritten = sl.list_family("b", environ=environ).items[0]
    assert (rewritten.parent_state, rewritten.parent_workspace) == ("unavailable", "a")
    assert "rewritten or removed" in rewritten.parent_problem
    torn = root / "a" / "runs" / "gatv2" / BASE_RUN / "checkpoints" / "last.pt"
    torn.write_bytes(b"not a checkpoint")
    unread = sl.list_family("b", environ=environ).items[0]
    assert (unread.parent_state, unread.parent_workspace) == ("unavailable", "a")
    assert "holds files that could not be read" in unread.parent_problem
    torn.unlink()
    gone = sl.list_family("b", environ=environ).items[0]
    assert gone.parent_problem.endswith("is not in a listed workspace")


def test_the_filters_are_checked_and_the_flat_list_takes_them(tmp_path, bundles):
    _put, _record_payload = bundles._put, bundles._record_payload
    root = tmp_path / "root"
    _put(root / "a", "gatv2", BASE_RUN, _record_payload(), 100)
    _put(root / "a", "gatv2", "20261002-000000_gene_bbbbbb", _record_payload("gene"), 200)
    environ = {ss.ENV_WORKSPACES_ROOT: str(root)}
    flat = sl.list_checkpoints("a", task="gene", environ=environ)
    assert [i.run for i in flat.items] == ["20261002-000000_gene_bbbbbb"]
    bases = sl.list_checkpoints("a", learning_stage="base", environ=environ)
    assert [i.run for i in bases.items] == [BASE_RUN]
    for listing in (sl.list_checkpoints("a", task="genes", environ=environ),
                    sl.list_family("a", task="genes", environ=environ)):
        assert listing.items == () and "'genes' is not a task" in listing.problem
    refused = sl.list_family("a", learning_stage="pretrain", environ=environ)
    assert "'pretrain' is not a learning stage" in refused.problem


def test_the_api_offers_the_family_behind_d8(tmp_path, monkeypatch, bundles):
    from fastapi.testclient import TestClient

    from src.api.main import app
    from tests.fixtures.operator_access import OPERATOR_BASE_URL, operator_mode

    root = tmp_path / "root"
    bundles._put(root / "a", "gatv2", BASE_RUN, bundles._record_payload(), 100)
    monkeypatch.setenv(ss.ENV_WORKSPACES_ROOT, str(root))
    refused = TestClient(app, base_url="http://192.0.2.10").get("/api/v1/workspaces/a/family")
    assert refused.status_code == 403
    operator_mode(monkeypatch)
    body = TestClient(app, base_url=OPERATOR_BASE_URL).get(
        "/api/v1/workspaces/a/family", params={"learning_stage": "base"}).json()
    assert body["problem"] is None
    (row,) = body["rows"]
    assert row["group"] == "family" and row["item"]["learning_stage"] == "base"
    assert row["item"]["run"] == BASE_RUN and str(root) not in str(body)
    flat = TestClient(app, base_url=OPERATOR_BASE_URL).get(
        "/api/v1/workspaces/a/checkpoints", params={"task": "disease"}).json()
    assert flat == {"checkpoints": [], "problem": None}


def test_one_run_is_listed_alone(tmp_path, bundles):
    from src.utils.checkpoint_paths import list_run_checkpoints, read_run_checkpoints

    a = bundles._put(tmp_path, "gatv2", BASE_RUN, bundles._record_payload(), 100)
    bundles._put(tmp_path, "gat", "20261001-000000_gnn_aaaaaa", bundles.GNN_PAYLOAD, 200)
    assert [p for _, p in list_run_checkpoints(tmp_path, run=BASE_RUN)] == [a]
    (entry,) = read_run_checkpoints(tmp_path, within=tmp_path, run=BASE_RUN)
    assert entry.path == a and entry.model_record["learning_stage"] == "base"
    assert list_run_checkpoints(tmp_path, run="missing") == []


def test_the_family_is_drawn_from_a_listing_given_without_listing_again(tmp_path, bundles,
                                                                        monkeypatch):
    """`list_family(listing=)`: a page that already listed the workspace draws
    its tree from that one listing (README §1); the filters still apply, and the
    record matcher the server filters by is the one a page shows by."""
    from src.utils.fingerprint import file_sha256

    root = tmp_path / "root"
    base = bundles._put(root / "a", "gatv2", BASE_RUN, bundles._record_payload(), 100)
    bundles._put(root / "a", "gatv2", "20261002-000000_gene_bbbbbb",
                 bundles._record_payload("gene", parent={"sha256": file_sha256(base),
                                                         "run": BASE_RUN}), 200)
    environ = {ss.ENV_WORKSPACES_ROOT: str(root)}
    listing = sl.list_checkpoints("a", environ=environ)
    monkeypatch.setattr(sl, "list_checkpoints",
                        lambda *a, **k: pytest.fail("listed again"))
    rows = sl.list_family("a", task="gene", listing=listing, environ=environ).items
    assert [(r.item.run, r.context) for r in rows] == [(BASE_RUN, True),
                                                       ("20261002-000000_gene_bbbbbb", False)]
    from dataclasses import asdict

    gene = rows[1].item
    assert sl.record_matches(gene, "gene") and sl.record_matches(asdict(gene), "gene")
    assert not sl.record_matches(asdict(gene), learning_stage="base")
    assert sl.record_matches(asdict(rows[0].item)) and not hasattr(sl, "_record_matches")
