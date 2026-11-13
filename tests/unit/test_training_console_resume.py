"""
Continuing a run from the Training page (contract M3a; M5-2, UI-C; K6).

Covers the Continue drawer's read path aligning with the run layout (contract
M3a-2) and the server's lists (M5-2): the parent list is the run bar's
workspace's model records only (K6 decision 1: a file without a model spec,
or one whose record cannot be read, is never continued from the dashboard),
through the shared listing; a listed parent is resolved by the server from
its list again (never a client path) and its bytes compared with the digest
listed, by the service's preview (P22-S1); the listed and the typed
checkpoint go through one check; Continue Training sends only operational
fields, the dataset fingerprint and the parent's model fingerprint the
session showed. The parents are model bundle bases trained for real on CPU.
"""
from pathlib import Path

import pytest

# training_console imports gradio (and the api/service stack) at module load.
pytest.importorskip("gradio")

from src.api.services.training_manager import training_manager  # noqa: E402
from src.webui.components import renderers as r  # noqa: E402
from src.webui.components import training_console as tc  # noqa: E402
from src.webui.components.workspaces import OUTSIDE_PATH  # noqa: E402
from src.webui.shell import standalone_context  # noqa: E402
from tests.fixtures.operator_access import gradio_request, operator_mode  # noqa: E402


#: A request D8 allows once the deployment is in mode M1 (`_operator_mode`):
#: the page's controls are operator controls since M5-1.
OP = gradio_request()


@pytest.fixture(autouse=True)
def _operator_mode(monkeypatch):
    operator_mode(monkeypatch)


def _run_file(ws, arch, run, name):
    path = ws / "runs" / arch / run / "checkpoints" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")
    return path


def _manifest(ws):
    """A workspace the server lists by name (a readable manifest is not needed
    to list its runs)."""
    ws.mkdir(parents=True, exist_ok=True)
    return ws


# --------------------------------------------------------------- the parent list
def test_the_parent_list_is_the_run_bars_workspaces_model_records_only(tmp_path, monkeypatch):
    """The run bar's workspace's runs, never another's; and of its files only
    the model records (a base and a task model): a file without a model spec
    and a file the read refuses are listed on Research › Models, never
    offered to continue (K6 decision 1)."""
    from tests.unit.test_task_first_selection import GNN_PAYLOAD, _put, _record_payload

    root = tmp_path / "root"
    ws, other = _manifest(root / "ws"), _manifest(root / "other")
    base = _put(ws, "gatv2", "20261001-000000_a_111111", _record_payload(), 100)
    child = _put(ws, "gatv2", "20261002-000000_b_222222", _record_payload("gene"), 200,
                 name="model-01-0.4000.pt")
    _put(ws, "hgt", "20261003-000000_hist_333333", GNN_PAYLOAD, 300)
    _run_file(ws, "gatv2", "20261004-000000_torn_444444", "model-1-0.9.pt")
    _put(other, "gatv2", "20261005-000000_c_555555", _record_payload(), 400)
    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(root))

    parents, runs, update, note = tc._refresh_checkpoints(OP, {"name": "ws"}, {}, None)
    child_key = "gatv2/20261002-000000_b_222222/model-01-0.4000.pt"
    assert set(parents) == {"gatv2/20261001-000000_a_111111/last.pt", child_key}
    run_labels = {value: label for label, value in runs["choices"]}
    assert run_labels == {"gatv2/20261001-000000_a_111111": "a · GATv2 2026-10-01 00:00",
                          "gatv2/20261002-000000_b_222222": "b · GATv2 2026-10-02 00:00"}
    # The chosen run's files only; the chosen run is the first listed.
    labels = dict((value, label) for label, value in update["choices"])
    assert set(labels) == {key for key in parents if key.startswith(runs["value"] + "/")}
    from src.webui.components.model_list import on_run_chosen

    child_files = on_run_chosen("gatv2/20261002-000000_b_222222", parents, None)
    assert [key for _label, key in child_files["choices"]] == [child_key]
    assert ">Kind</dt><dd class=\"shep-v\">Gene<" in tc.parent_details_html(parents[child_key])
    assert "2 runs, 2 checkpoints" in note
    # No client path anywhere: the values are keys the server resolves again.
    assert all(str(root) not in value for value in (*labels, *run_labels))
    assert {base.name, child.name} == {key.rsplit("/", 1)[1] for key in parents}

    # A workspace whose files hold no model record offers none, and says why.
    only_history = _manifest(root / "old")
    _put(only_history, "hgt", "20261006-000000_hist_666666", GNN_PAYLOAD, 500)
    parents, runs, update, note = tc._refresh_checkpoints(OP, {"name": "old"}, {}, None)
    assert parents == {} and update["choices"] == [] and runs["choices"] == []
    assert "holds a model record" in note and "never continued here" in note


def test_the_parent_list_does_not_list_the_old_layout(tmp_path, monkeypatch):
    root = tmp_path / "root"
    ws = _manifest(root / "ws")
    (ws / "checkpoints" / "hgt").mkdir(parents=True)
    (ws / "checkpoints" / "hgt" / "last.pt").write_text("x")
    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(root))

    parents, runs, update, _note = tc._refresh_checkpoints(OP, {"name": "ws"}, {}, None)
    assert parents == {} and update["choices"] == [] and update["value"] is None
    assert runs["choices"] == [] and runs["value"] is None


def test_the_manager_lists_every_architecture_with_its_run(tmp_path):
    ws = tmp_path / "ws"
    import os

    newer = _run_file(ws, "hgt", "20261001-000000_a_111111", "last.pt")
    older = _run_file(ws, "gatv2", "20261002-000000_b_222222", "last.pt")
    os.utime(newer, (2_000, 2_000))
    os.utime(older, (1_000, 1_000))

    listed = training_manager.get_checkpoints(str(ws))
    assert [(c["architecture"], c["run"]) for c in listed] == [
        ("hgt", "20261001-000000_a_111111"), ("gatv2", "20261002-000000_b_222222")]


# --------------------------------------------------------------- docstring
def test_cli_resume_docstring_names_the_run_layout():
    text = Path("scripts/train_model.py").read_text(encoding="utf-8")
    assert "--resume checkpoints/last.pt" not in text  # stale flat example gone
    assert "/checkpoints/hgt/last.pt" not in text      # the M2 layout is gone too
    # A run is filed under its model spec's encoder kind (gatv2, the reference).
    assert "runs/gatv2/<run>/checkpoints/last.pt" in text


# --------------------------------------------------------------- continue block (M3a)
PAUSED_RUN = "20261001-000000_paused_aaaaaa"
DONE_RUN = "20261002-000000_done_bbbbbb"
HISTORY_RUN = "20261003-000000_history_cccccc"


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A paused and a finished model bundle base (two epochs, the first
    paused after one), trained for real on the CPU in the run layout of a
    workspace a workspaces root lists (``data``), and a file without a model
    spec beside them."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import train_model
    from tests.fixtures.base_workspace import write_base_workspace
    from tests.unit.test_task_first_selection import GNN_PAYLOAD, _put
    from tests.unit.test_train_base_stage import _config

    root = tmp_path_factory.mktemp("console")
    ws = write_base_workspace(root)
    runs = ws / "runs" / "gatv2"
    train_model.train(*_config(ws, runs / PAUSED_RUN, pause_after_epoch=1))
    train_model.train(*_config(ws, runs / DONE_RUN))
    history = _put(ws, "hgt", HISTORY_RUN, GNN_PAYLOAD, 1_000)
    return {"root": root, "ws": ws,
            "paused": runs / PAUSED_RUN / "checkpoints" / "last.pt",
            "finished": runs / DONE_RUN / "checkpoints" / "last.pt", "history": history}


@pytest.fixture
def checkpoints(trained, monkeypatch):
    """The trained workspace listed by the server, with the session's state
    as the page holds it: the run bar's workspace and the parent list."""
    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(trained["root"]))
    ws = tc.workspace_state("data")
    parents, _problem = tc._list_parents(ws)
    return {**trained, "state": ws, "parents": parents,
            "paused_key": f"gatv2/{PAUSED_RUN}/last.pt",
            "done_key": f"gatv2/{DONE_RUN}/last.pt"}


def _values(**edited):
    """The Branch fields' values as the drawer holds them before a check
    (`_branch_field_value`), each edit in `edited` in its field's form."""
    return [edited.get(name, tc.form_value(tc._branch_field_value(name)))
            for name in tc.BRANCH_FIELDS]


def _check_all(c, source, listed, manual, device="cpu", extend=None, mode=None, *,
               operation=tc.CONTINUE, rate=None, touched=(), parents=None, **edited):
    """Check checkpoint, every output: the check's text, the Extra epochs and
    Extra-epoch rate updates, the mode, the comparison, the Branch box and
    each Branch field (`BRANCH_FIELDS`)."""
    return tc._on_check_continue(OP, operation, source, listed, manual, device, extend, rate,
                                 c["state"], parents or c["parents"], mode or {},
                                 sorted({*touched, *edited}), *_values(**edited))


def _check(c, source, listed, manual, device="cpu", extend=None, mode=None, **kwargs):
    """Check checkpoint's text, Extra epochs update, mode and comparison."""
    text, extend, _rate, mode, compare, *_branch = _check_all(
        c, source, listed, manual, device, extend, mode, **kwargs)
    return text, extend, mode, compare


def _resume_all(c, source, listed, manual, extend=None, mode=None, device="cpu", *,
                operation=tc.CONTINUE, rate=None, touched=(), state=None, **edited):
    """The drawer's start, every output: its notice, the check's text, the
    Extra epochs and Extra-epoch rate updates, the mode, the Branch box and
    each Branch field."""
    return tc._on_resume(OP, operation, state or c["state"], device, source, listed, manual,
                         extend, rate, c["parents"], mode or {}, sorted({*touched, *edited}),
                         *_values(**edited))


def _resume(c, source, listed, manual, extend=None, mode=None, device="cpu", **kwargs):
    """The drawer's start's notice, the check's text, the Extra epochs update
    and the mode."""
    said, text, extend, _rate, mode, *_branch = _resume_all(c, source, listed, manual, extend,
                                                           mode, device, **kwargs)
    return said, text, extend, mode


def _starts(started):
    def start(config, **kwargs):
        started.append({**config, "__kwargs__": kwargs})
        return {"success": True, "pid": 1, "run_dir": "/r"}
    return start


def test_the_listed_and_the_typed_checkpoint_go_through_one_check(checkpoints):
    from_list, _extend, mode, compare = _check(checkpoints, tc.CONTINUE_FROM_LIST,
                                               checkpoints["paused_key"], "")
    typed, *_rest = _check(checkpoints, tc.CONTINUE_FROM_PATH, "ignored",
                           str(checkpoints["paused"]))

    assert "from the run list" in from_list and "(manual path)" in typed
    for text in (from_list, typed):
        assert "Resume the original training (1/2 done, 1 left)" in text
        assert '<dt class="shep-k" title="learning_rate">Learning rate</dt>' in text
        # The settings cannot change when continuing: the note points to
        # Branch (ops-wiring round B1c).
        assert "to train with other settings, choose Branch above" in text
    # The listed parent was resolved by the server from its list: the file
    # read is the one listed (its run and name; its fingerprint is in the
    # chosen checkpoint's details, not repeated here).
    assert f"<b>Checked</b> <code>gatv2/{PAUSED_RUN}/last.pt</code> from the run list" in from_list
    assert "model fingerprint" not in from_list and "model fingerprint" in typed
    # The parent recorded this workspace's dataset: the same, so Continue
    # Training is on.
    assert mode["parent"]["fingerprint"] == checkpoints["state"]["row"]["dataset_fingerprint"]
    assert "Trained on the dataset selected in the run bar" in compare
    assert tc.continue_refusal(mode["parent"], checkpoints["state"]) is None


def test_the_checks_settings_are_said_in_words_and_the_spec_by_its_differences(checkpoints):
    """BACKLOG 72, ops-wiring round B0c: the check printed each setting taken
    from the checkpoint as its Python repr under its field name, so one None
    stood for early stopping off, a setting a base does not take and an unset
    seed, and its model spec row was the whole spec with its sources (about
    3,000 px). From a real base's preview: each setting by its words and its
    value in the one value-words rule (no repr quote, no bare None), each
    None by its meaning, and the model spec by its differences from the
    reference, the full spec one click away behind one "Details ›", never a
    disclosure inside another."""
    import html as html_lib
    import re

    text, *_rest = _check(checkpoints, tc.CONTINUE_FROM_LIST, checkpoints["paused_key"], "")
    words = html_lib.unescape(re.sub(r"<[^>]+>", " ", text))
    for repr_mark in ("'", "None", "True", "False", "{", "[", "repr"):
        assert repr_mark not in words, repr_mark
    rows = dict(re.findall(r'<dt class="shep-k"(?: title="[^"]*")?>([^<]*)</dt>'
                           r'<dd class="shep-v">(.*?)</dd>', text))
    assert rows["Early stopping patience"] == "off"            # None: no early stopping
    assert rows["Task loss weight"] == r.NOT_TAKEN_BY_A_BASE   # None: a base takes none
    assert rows["Task"] == r.NOT_TAKEN_BY_A_BASE
    assert rows["Neighbours per hop"] == "4, 4" and rows["Mixed precision"] in ("on", "off")
    assert rows["Graph backbone"] == "Trained"
    # The model spec: its own row, by its differences (a shrunk test spec),
    # its every field one click away; the settings in the other disclosure.
    assert rows["Model spec"].startswith("Differs from the reference GATv2 spec: ")
    spec = text[text.index('title="model_spec"'):text.index('<details class="shep-inherit">')]
    assert spec.count("<summary>Details ›</summary>") == 1
    assert ('<dt class="shep-k" title="encoder.layers">Encoder › Layers</dt>' in spec
            and "(reference 3)" in spec)
    assert "sources" not in spec and "not taken by a base" in spec
    assert text.count("<details") == 2                          # Details ›, Settings taken
    inherit = text[text.index('<details class="shep-inherit">'):]
    assert "Model spec" not in inherit and "<details" not in inherit[len("<details"):]
    assert text.index("Model spec") < text.index('<details class="shep-inherit">')


def test_a_refused_checks_reason_names_no_server_path(checkpoints, monkeypatch, tmp_path):
    """Ops-wiring round B0c (D7): a refused check showed the service's reason
    as it is, and the reason names the file the service looked for ("/srv/…
    /last.pt does not exist.", a listed file removed since it was listed) or
    a path a read error names, so a server path was shown on the page. The
    reason is shown as the shared redaction gives it: a path inside the
    workspaces root relative to it, any other as "a path outside the
    workspaces root"."""
    inside = checkpoints["paused"]
    outside = tmp_path / "elsewhere" / "x.pt"
    reasons = iter([f"{inside} does not exist.",
                    f"last.pt cannot be read as a checkpoint (OSError: {outside})."])
    monkeypatch.setattr(training_manager, "preview_continuation",
                        lambda *a, **k: {"ok": False, "parent": str(inside),
                                         "reason": next(reasons)})
    relative = inside.relative_to(checkpoints["root"]).as_posix()
    for shown in (f"{relative} does not exist.",
                  "(OSError: a path outside the workspaces root)."):
        text, *_rest = _check(checkpoints, tc.CONTINUE_FROM_PATH, "ignored",
                              str(checkpoints["paused"]))
        assert "Cannot continue." in text and shown in text, text
        assert str(tmp_path) not in text and str(checkpoints["root"]) not in text


@pytest.mark.parametrize("workspace", ["demo (copy)", "ws🧬", "a\xa0b", "工作 区"])
def test_a_path_inside_the_root_is_shown_whole_whatever_its_names_hold(workspace, tmp_path,
                                                                       monkeypatch):
    """The review of B0c: a path inside the workspaces root went through the
    free-text redaction, whose pattern started a new absolute path at any
    slash after a character that is no word's, so "demo (copy)/runs/…" read
    "demo (copy)a path outside the workspaces root", an emoji name alike, and
    its joining of whitespace turned a no-break space into a plain one. A
    path is one value (`workspaces.shown_path`): inside the root, its path
    there whole, on the check's path line, the run view's Details rows and a
    refused check's reason; outside, never shown."""
    root = tmp_path / "root"
    run = root / workspace / "runs" / "gatv2" / "20261113-124624_a\xa0b_8c0e70"
    checkpoint = run / "checkpoints" / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"x")
    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(root))
    inside = checkpoint.relative_to(root).as_posix()
    refused = tc._check_html({"ok": False, "parent": str(checkpoint),
                              "reason": f"{checkpoint} does not exist."}, tc.CONTINUE_FROM_PATH)
    assert f"<code>{r.e(inside)}</code>" in refused, refused
    assert f"<p>{r.e(inside)} does not exist.</p>" in refused, refused
    ref = tc.RunRef(id=f"{workspace}/gatv2/{run.name}", label=f"gatv2/{run.name}", path=run,
                    workspace=workspace, job=True, latest=False)
    monkeypatch.setattr(tc, "_newest_checkpoint", lambda _run: checkpoint)
    for head in (tc.run_head_html(ref, {}, operator=True, snapshot=None, live=True),
                 tc.run_head_html(ref, {}, operator=True, snapshot={"checkpoint": str(checkpoint)})):
        assert f"<code>{r.e(inside)}</code>" in head, head
    for page in (refused, head):
        assert str(root) not in page and OUTSIDE_PATH not in page
    # A path outside the root is never shown, in the line or the reason.
    outside = tmp_path / "elsewhere" / f"{workspace}/last.pt"
    elsewhere = tc._check_html({"ok": False, "parent": str(outside),
                                "reason": f"{outside} does not exist."}, tc.CONTINUE_FROM_PATH)
    assert str(tmp_path) not in elsewhere and elsewhere.count(OUTSIDE_PATH) == 2, elsewhere


def test_a_listed_parent_whose_bytes_changed_since_it_was_listed_is_refused(checkpoints):
    """The list carries each file's digest; the check sends it to the
    service's preview, which compares it on its read (a `last.pt` rewritten
    since the listing is not what was shown; root-cause round, P22-S1). The
    action is the list's: the same option sends the same digest until the list
    is read again."""
    parents = {key: dict(item) for key, item in checkpoints["parents"].items()}
    parents[checkpoints["paused_key"]]["sha256"] = "0" * 64
    text, _e, mode, _c = _check(checkpoints, tc.CONTINUE_FROM_LIST, checkpoints["paused_key"],
                                "", parents=parents)
    assert "The checkpoint changed since it was chosen." in text
    assert "Refresh the list, then check it again." in text
    assert mode["parent"]["refused"] == tc.REFUSED_CHANGED
    # Beside the start, its cause and what to do (the B1c review).
    assert tc.continue_refusal(mode["parent"], checkpoints["state"]) == (
        "the checkpoint changed since it was chosen. Refresh the list, then check it again.")


def test_a_parent_the_list_no_longer_has_is_not_opened(checkpoints, monkeypatch):
    opened = []
    monkeypatch.setattr(training_manager, "preview_continuation",
                        lambda *a, **k: opened.append(a) or {"ok": False})
    text, *_rest = _check(checkpoints, tc.CONTINUE_FROM_LIST, "gat/../../etc/last.pt", "")
    assert "Nothing to check" in text and opened == []


@pytest.mark.parametrize("where", ["copy_outside", "server_file", "other_workspace"])
def test_a_typed_path_only_locates_a_checkpoint_the_server_lists(checkpoints, monkeypatch,
                                                                  tmp_path, where):
    """D8 (2026-10-19): a typed parent "only locates an item the server lists
    and is checked like the rest". A file the run bar's workspace does not
    list, even the same bytes copied elsewhere, is refused unopened, for
    Check checkpoint and Continue Training alike."""
    import shutil

    if where == "copy_outside":
        typed = tmp_path / "elsewhere" / "last.pt"
        typed.parent.mkdir()
        shutil.copy2(checkpoints["paused"], typed)
    elif where == "server_file":
        typed = Path(__file__).resolve()
    else:
        other = checkpoints["root"] / "other" / "runs" / "gatv2" / PAUSED_RUN / "checkpoints"
        other.mkdir(parents=True)
        typed = other / "last.pt"
        shutil.copy2(checkpoints["paused"], typed)
    opened, started = [], []
    monkeypatch.setattr(training_manager, "preview_continuation",
                        lambda *a, **k: opened.append(a) or {"ok": False})
    monkeypatch.setattr(training_manager, "start_training", _starts(started))
    text, *_rest = _check(checkpoints, tc.CONTINUE_FROM_PATH, None, str(typed))
    assert "Nothing to check" in text and "not a checkpoint the server lists" in text
    assert str(typed) not in text
    said, *_rest = _resume(checkpoints, tc.CONTINUE_FROM_PATH, None, str(typed))
    assert "Not started: the typed path is not a checkpoint the server lists" in said
    assert opened == [] and started == []


def test_a_typed_path_of_a_listed_checkpoint_is_found_again_from_the_list(checkpoints,
                                                                          monkeypatch):
    """The typed path is only a locator: what is previewed and resumed is the
    file the server's list resolves for it. Its Check is what shows its model
    fingerprint (the run list is hidden while a typed path is the source), and
    the session records that one for Continue Training (P22-S1)."""
    seen = []
    real = training_manager.preview_continuation
    monkeypatch.setattr(training_manager, "preview_continuation",
                        lambda parent, **k: seen.append((parent, k)) or real(parent, **k))
    relative = Path(checkpoints["paused"]).resolve()
    text, _e, mode, _c = _check(checkpoints, tc.CONTINUE_FROM_PATH, None, f"  {relative}  ")
    assert "Resume the original training" in text
    assert seen and Path(seen[0][0]).resolve() == relative
    assert seen[0][1]["expected_sha256"] is None
    assert mode["parent"]["sha256"] == checkpoints["parents"][checkpoints["paused_key"]]["sha256"]


def test_a_finished_checkpoint_offers_extend_and_says_why_it_cannot_resume(checkpoints):
    path = str(checkpoints["finished"])
    text, extend, _mode, _compare = _check(checkpoints, tc.CONTINUE_FROM_PATH, None, path)
    # The plan's refusal names the field by the drawer's own words (B1c).
    assert "Cannot continue" in text and "extend it at a fixed rate with Extra epochs" in text
    assert "extend_epochs" not in text
    assert extend["visible"] is True

    text, *_rest = _check(checkpoints, tc.CONTINUE_FROM_PATH, None, path, extend=3)
    assert "Extend training by 3 epochs at a fixed rate" in text


def test_continue_sends_only_operational_fields_and_never_bypasses_the_check(
    checkpoints, monkeypatch
):
    started = []
    monkeypatch.setattr(training_manager, "start_training", _starts(started))
    refused = _resume(checkpoints, tc.CONTINUE_FROM_PATH, None, str(checkpoints["finished"]))
    assert started == [], "a refused checkpoint was started"
    # Said once (B1c): the check says why, the footer's reason beside the
    # start that it is disabled; no notice repeats it.
    assert refused[0] == tc.notice_html("") and "Cannot continue" in refused[1]
    assert tc.continue_refusal(refused[3]["parent"], checkpoints["state"]) == (
        "the check says this checkpoint cannot be continued.")

    _resume(checkpoints, tc.CONTINUE_FROM_LIST, checkpoints["paused_key"], "")
    assert started and set(started[0]) - {"__kwargs__"} == {
        "data_dir", "run_name", "device", "resume_from", "compile"}
    assert started[0]["resume_from"] == str(checkpoints["paused"])
    assert started[0]["data_dir"] == str(checkpoints["ws"].resolve())
    # The start carries the dataset fingerprint the run bar showed and the
    # parent's model fingerprint the run list showed (P22-S1).
    assert started[0]["__kwargs__"] == {
        "expected_dataset_fingerprint": checkpoints["state"]["row"]["dataset_fingerprint"],
        "expected_parent_sha256": checkpoints["parents"][checkpoints["paused_key"]]["sha256"]}


def test_continue_training_is_refused_when_the_parents_dataset_is_not_the_workspaces(
    checkpoints, monkeypatch
):
    """The parent recorded another dataset (the workspace was rebuilt since):
    the run bar's Continue Training is disabled with the reason, and a direct
    call starts nothing."""
    started = []
    monkeypatch.setattr(training_manager, "start_training", _starts(started))
    other = dict(checkpoints["state"])
    other["row"] = {**other["row"], "dataset_fingerprint": "e" * 64}
    out = _resume(checkpoints, tc.CONTINUE_FROM_LIST, checkpoints["paused_key"], "",
                  state=other)
    assert started == []
    # The footer's reason says it beside the start (B1c), and the check: no
    # notice repeats it.
    assert out[0] == tc.notice_html("")
    assert "cannot continue on the current dataset" in out[1]
    assert tc.continue_refusal(out[3]["parent"], other) == (
        "the parent’s dataset is not this workspace’s current one.")
    compare = tc.compare_html(out[3]["parent"], other)
    assert "The parent’s dataset is not this workspace’s current one" in compare


def test_stop_asks_without_naming_an_epoch_the_job_has_moved_past(checkpoints, monkeypatch):
    """The job trains on while the question is open (the GB10 acceptance:
    asked at epoch 183, 195 saved), so the question names no epoch and reads
    no checkpoint: the same words whether one is saved or not, and whatever
    is saved by the answer. Where it can be continued is said once it has
    stopped (test_a_stopped_run_shows_stopped_and_where_it_can_be_continued)."""
    monkeypatch.setattr(training_manager, "get_status",
                        lambda: {"status": "running", "job_id": "j1"})
    monkeypatch.setattr(training_manager, "latest_snapshot",
                        lambda *a, **k: pytest.fail("the question read a checkpoint"))
    asked = []
    for run_dir in (checkpoints["paused"].parent.parent, checkpoints["ws"] / "no-run"):
        monkeypatch.setattr(training_manager, "run_dir", run_dir)
        text, mode = tc._on_stop_request(OP, {})
        assert mode["confirm"] == "j1"                      # asked for the job running now
        asked.append(text)
    assert asked[0] == asked[1] == f'<div class="shep-rb-confirm-q">{tc.STOP_QUESTION}</div>'
    assert "epoch" not in tc.STOP_QUESTION.lower()
    assert "saved are kept" in tc.STOP_QUESTION and "not yet saved is lost" in tc.STOP_QUESTION

    # Asked before any checkpoint, answered after one was saved (and later
    # epochs ran): the notice names the checkpoint the stop left, not one
    # read when it was asked.
    monkeypatch.setattr(training_manager, "stop_training", lambda: {
        "success": True, "outcome": {"checkpoint": str(checkpoints["paused"]), "readable": True,
                                     "completed_epochs": 195, "target_epochs": 5000,
                                     "end": "paused"}})
    notice, mode = tc._on_stop(OP, {"confirm": "j1"})
    assert "Stopped. Paused at epoch 195 of 5000" in notice and mode["confirm"] is None

    # Keep training closes the question and stops nothing.
    monkeypatch.setattr(training_manager, "stop_training",
                        lambda: pytest.fail("Keep training stopped the job"))
    assert tc.close_stop_question({"confirm": "j1"}) == {"confirm": None}

    # Nothing running: nothing is asked.
    monkeypatch.setattr(training_manager, "get_status", lambda: {"status": "completed",
                                                                  "job_id": "j1"})
    text, mode = tc._on_stop_request(OP, {"confirm": "j0"})
    assert mode["confirm"] is None and "no training job is running" in text


def test_a_stopped_run_shows_stopped_and_where_it_can_be_continued(checkpoints):
    snapshot = training_manager.__class__()
    snapshot.run_dir = checkpoints["paused"].parent.parent
    outcome = snapshot.latest_snapshot()
    assert outcome["resumable"] is True and outcome["end"] == "paused"
    assert outcome["dataset_fingerprint"] == checkpoints["state"]["row"]["dataset_fingerprint"]

    kind, words = tc.pill_state({"status": "stopped", "current_epoch": 1, "total_epochs": 2})
    assert kind == "stopped" and words.startswith("Stopped · epoch 1 of 2")
    assert tc.PILL_KIND[kind] == "grey"
    sentence = tc._snapshot_sentence(outcome)
    assert sentence.startswith("Paused at epoch 1 of 2")
    assert f"data/runs/gatv2/{PAUSED_RUN}/checkpoints/last.pt" in sentence
    assert "No checkpoint was saved" in tc._snapshot_sentence(None)


def test_the_service_refuses_an_explicit_conflict_before_launching(checkpoints, monkeypatch):
    import subprocess as sp
    from unittest.mock import patch

    manager = training_manager.__class__()
    with patch.object(sp, "Popen") as popen:
        result = manager.start_training({"data_dir": str(checkpoints["ws"]),
                                         "resume_from": str(checkpoints["paused"]),
                                         "device": "cpu", "learning_rate": 0.5})
    assert result["success"] is False and "learning_rate" in result["error"]
    popen.assert_not_called()


def test_the_continue_inputs_pass_gradios_own_preprocessing_when_empty():
    """Found by driving the page: an empty, hidden extra-epochs Number is sent as
    0, and a widget minimum of 1 rejected the whole Check event before the
    handler ran. The handler tests call it directly and could not see that."""
    import gradio as gr

    with gr.Blocks() as demo:
        tc.create_training_tab(standalone_context(demo))
    widgets = {getattr(block, "elem_id", None): block for block in demo.blocks.values()}
    for elem_id, empty in (("extend_epochs", 0), ("extend_epochs", None),
                           ("manual_checkpoint", ""), ("checkpoint_dropdown", None)):
        widgets[elem_id].preprocess(empty)


# --------------------------------------------------------------- stale Extra epochs (review of 094b90d, P2-2)
def _apply(state, update):
    """What Gradio does with a returned update: set the fields it names."""
    for key in ("value", "visible"):
        if key in update:
            state[key] = update[key]
    return state


@pytest.mark.parametrize("source", [tc.CONTINUE_FROM_LIST, tc.CONTINUE_FROM_PATH])
def test_extra_epochs_left_from_a_finished_checkpoint_never_block_a_resume(
    checkpoints, monkeypatch, source
):
    """Finished checkpoint, Extra epochs 3; then the unfinished one; Check;
    Continue. The selection change clears the field and the old check, and the
    resume starts without extend_epochs. Both sources, with the component state
    the returned updates leave."""
    started = []
    monkeypatch.setattr(training_manager, "start_training", _starts(started))

    def chosen(which):
        if source == tc.CONTINUE_FROM_LIST:
            return checkpoints[f"{'done' if which == 'finished' else 'paused'}_key"], ""
        return None, str(checkpoints[which])

    extra = {"value": None, "visible": False}
    listed, manual = chosen("finished")
    text, update, _m, _c = _check(checkpoints, source, listed, manual, extend=extra["value"])
    _apply(extra, update)
    assert extra["visible"] is True and "Enter <b>Extra epochs</b> above" in text
    extra["value"] = 3
    text, update, _m, _c = _check(checkpoints, source, listed, manual, extend=extra["value"])
    _apply(extra, update)
    assert "Extend training by 3 epochs" in text and extra == {"value": 3, "visible": True}

    # The operator picks the unfinished checkpoint: the change event fires.
    listed, manual = chosen("paused")
    update, _rate, preview, *_rest = tc._on_continue_selection_changed()
    _apply(extra, update)
    assert extra == {"value": None, "visible": False} and preview == tc._CONTINUE_PROMPT

    text, update, _m, _c = _check(checkpoints, source, listed, manual, extend=extra["value"])
    _apply(extra, update)
    assert "Resume the original training" in text and extra["visible"] is False
    _resume(checkpoints, source, listed, manual, extend=extra["value"])
    assert len(started) == 1 and "extend_epochs" not in started[0]
    assert started[0]["resume_from"] == str(checkpoints["paused"])


def test_a_stale_value_that_reaches_the_handlers_is_not_sent_for_an_unfinished_checkpoint(
    checkpoints, monkeypatch
):
    """Without the change event (a value left however it was), Check and
    Continue still treat Extra epochs as the finished checkpoint's field."""
    started = []
    monkeypatch.setattr(training_manager, "start_training", _starts(started))
    paused = str(checkpoints["paused"])
    text, update, _m, _c = _check(checkpoints, tc.CONTINUE_FROM_PATH, None, paused, extend=3)
    assert "Resume the original training" in text
    assert update["visible"] is False and update["value"] is None
    _resume(checkpoints, tc.CONTINUE_FROM_PATH, None, paused, extend=3)
    assert len(started) == 1 and "extend_epochs" not in started[0]

    # A finished checkpoint still extends by the epochs entered.
    _resume(checkpoints, tc.CONTINUE_FROM_PATH, None, str(checkpoints["finished"]), extend=3)
    assert started[1]["extend_epochs"] == 3


def test_the_service_still_refuses_extend_epochs_for_an_unfinished_parent(checkpoints):
    """The console's rule does not reach other callers: an explicit
    extend_epochs on an unfinished parent is refused."""
    preview = training_manager.preview_continuation(str(checkpoints["paused"]), device="cpu",
                                                    extend_epochs=3)
    assert preview["ok"] is False and "applies only to a finished stage" in preview["reason"]
    # The preview names the dataset the parent recorded (the drawer compares it).
    assert preview["dataset_fingerprint"] == checkpoints["state"]["row"]["dataset_fingerprint"]


def test_every_selection_change_clears_the_check_and_extra_epochs():
    """Each component that is part of the check is wired to clear it: the
    source, the run list and the typed path (each only when it is the source),
    the operation (ops-wiring round B1c), and the drawer's own device (the
    check only). What is cleared is everything entered for the old one:
    Extra epochs and its rate, the start's last notice, the Branch fields'
    record of what was edited and their box."""
    import gradio as gr

    with gr.Blocks() as demo:
        tc.create_training_tab(standalone_context(demo))
    by_id = {block._id: getattr(block, "elem_id", None) for block in demo.blocks.values()}
    wiring = {}
    for fn in demo.fns.values():
        if fn.fn in (tc._on_continue_selection_changed, tc._on_listed_changed,
                     tc._on_manual_changed, tc._on_continue_device_changed, tc._on_operation):
            for block, event in fn.targets:
                wiring[(by_id[block], event)] = (
                    fn.fn.__name__, [getattr(o, "elem_id", None) for o in fn.outputs])
    cleared = ["extend_epochs", "extend_lr", "continue_preview", "shep-train-drawer-notice",
               "shep-train-branch-touched", "shep-train-branch"]
    assert wiring == {
        ("continue_source", "change"): ("_on_continue_selection_changed", cleared),
        ("checkpoint_dropdown", "change"): ("_on_listed_changed", cleared),
        ("manual_checkpoint", "change"): ("_on_manual_changed", cleared),
        ("continue_operation", "change"): ("_on_operation", [None, *cleared]),
        ("continue_device", "change"): ("_on_continue_device_changed", ["continue_preview"]),
    }
    assert tc._on_continue_selection_changed() == (
        gr_update(visible=False, value=None), gr_update(visible=False, value=None),
        tc._CONTINUE_PROMPT, "", [], gr_update(visible=False))


def test_refresh_is_followed_by_the_chosen_options_file():
    """The B1c review: Refresh kept the chosen option's key, so Gradio ran no
    selection event, and the record, the details and the check stayed the
    old file's. Refresh's next step is `_on_refreshed`, writing what a
    selection writes (the record, the comparison, the details) and the
    check's outputs (`check_cleared`); its cases are in
    test_training_parent_expected.py."""
    import gradio as gr

    with gr.Blocks() as demo:
        tc.create_training_tab(standalone_context(demo))
    fns = {fn._id: fn for fn in demo.fns.values()}
    [refresh] = [fn for fn in fns.values() if fn.fn is tc._refresh_checkpoints]
    [follow] = [fn for fn in fns.values() if fn.fn is tc._on_refreshed]
    assert follow.trigger_after == refresh._id
    assert [getattr(o, "elem_id", None) for o in follow.outputs] == [
        None, "shep-train-compare", "shep-train-parent-details", "extend_epochs", "extend_lr",
        "continue_preview", "shep-train-drawer-notice", "shep-train-branch-touched",
        "shep-train-branch"]
    assert [getattr(i, "elem_id", None) for i in follow.inputs] == [
        "continue_source", "checkpoint_dropdown", "manual_checkpoint", None, None, None]


def test_a_refreshed_run_list_does_not_clear_a_typed_paths_check():
    """A Refresh repopulates the run list, whose change event fires whatever
    the source; it clears only when the run list is the source (found by the
    verification round after the P2-2 fix)."""
    untouched = tc._on_listed_changed(tc.CONTINUE_FROM_PATH)
    assert all(update == gr_update() for update in untouched)
    assert tc._on_listed_changed(tc.CONTINUE_FROM_LIST) == tc._cleared_check()

    untouched = tc._on_manual_changed(tc.CONTINUE_FROM_LIST)
    assert all(update == gr_update() for update in untouched)
    assert tc._on_manual_changed(tc.CONTINUE_FROM_PATH) == tc._cleared_check()

    assert tc._on_continue_device_changed() == tc._CONTINUE_PROMPT


def gr_update(**kwargs):
    import gradio as gr

    return gr.update(**kwargs)


def test_continue_without_check_on_a_finished_checkpoint_shows_extra_epochs(
    checkpoints, monkeypatch
):
    started = []
    monkeypatch.setattr(training_manager, "start_training", _starts(started))
    result = _resume(checkpoints, tc.CONTINUE_FROM_PATH, None, str(checkpoints["finished"]))
    assert started == [] and result[3]["parent"]["refused"] == tc.REFUSED_PARENT
    assert "Enter <b>Extra epochs</b> above" in result[1]           # the field is above the check
    assert result[2]["visible"] is True


# --------------------------------------------------------------- K6: history files and Fine-tune children
def test_a_file_without_a_model_spec_is_never_continued_from_the_dashboard(checkpoints,
                                                                           monkeypatch):
    """K6 decision 1: the run list offers model records only, and a typed path
    that locates a file the server lists but that holds no model spec is
    refused unopened, for Check checkpoint and Continue Training alike."""
    opened, started = [], []
    monkeypatch.setattr(training_manager, "preview_continuation",
                        lambda *a, **k: opened.append(a) or {"ok": False})
    monkeypatch.setattr(training_manager, "start_training", _starts(started))
    assert not [key for key in checkpoints["parents"] if HISTORY_RUN in key]
    typed = str(checkpoints["history"])
    text, *_rest = _check(checkpoints, tc.CONTINUE_FROM_PATH, None, typed)
    assert "Nothing to check" in text and "not a model record the run list offers" in text
    said, *_rest = _resume(checkpoints, tc.CONTINUE_FROM_PATH, None, typed)
    assert "Not started: the typed path is not a model record" in said
    assert opened == [] and started == []


def test_stop_names_the_fine_tune_children_it_will_cancel(monkeypatch):
    """A stop reaches a Fine-tune start: the question says which queued
    children will not start (a stop cancels them), and the notice which
    were cancelled. Without a queue it is Stop's own question."""
    status = {"status": "running", "job_id": "j1",
              "finetune": {"children": [{"task": "disease", "status": "running"},
                                        {"task": "gene", "status": "queued"},
                                        {"task": "patients_like_me", "status": "queued"}]}}
    monkeypatch.setattr(training_manager, "get_status", lambda: status)
    text, mode = tc._on_stop_request(OP, {})
    assert mode["confirm"] == "j1" and tc.STOP_QUESTION in text
    assert "The queued Fine-tune children (Gene, Patients-like-me) will not start." in text
    assert tc.stop_question({"status": "running"}) == tc.STOP_QUESTION
    monkeypatch.setattr(training_manager, "stop_training", lambda: {
        "success": True, "status": "stopped", "outcome": None,
        "cancelled_children": ["gene", "patients_like_me"]})
    notice, _mode = tc._on_stop(OP, {"confirm": "j1"})
    assert "The queued Fine-tune children (Gene, Patients-like-me) were cancelled." in notice


def test_a_stop_after_the_child_ended_on_its_own_says_nothing_was_stopped(monkeypatch):
    """The running child ended on its own just before the stop arrived: the
    runner stopped nothing and only cancelled the queued children
    (`TrainingManager._stop`, its ``message``). The notice says so; it never
    reads "Stopped." while the child shows Completed."""
    status = {"status": "running", "job_id": "j1",
              "finetune": {"children": [{"task": "disease", "status": "running",
                                         "job_id": "j1"},
                                        {"task": "gene", "status": "queued"}]}}
    monkeypatch.setattr(training_manager, "get_status", lambda: status)
    monkeypatch.setattr(training_manager, "stop_training", lambda: {
        "success": True, "status": "completed", "cancelled_children": ["gene"],
        "outcome": {"checkpoint": "/x/model.pt", "readable": True, "completed_epochs": 100,
                    "target_epochs": 100, "end": "finished"},
        "message": "Training (job j1) had already ended on its own (completed, exit code 0) "
                   "before this stop arrived; nothing was stopped. The queued children (gene) "
                   "were cancelled."})
    notice, mode = tc._on_stop(OP, {"confirm": "j1"})
    assert "Stopped." not in notice and mode["confirm"] is None
    assert ("Nothing was stopped: the job had already ended on its own (completed). "
            "The queued Fine-tune children (Gene) were cancelled.") in notice


# --------------------------------------------------------------- Branch and the Extra-epoch rate (B1c)
def _fields(outputs):
    """The Branch box's update and each Branch field's, by name, from a
    Check's or a start's outputs (they end with the box, then the fields)."""
    box, *fields = outputs[-1 - len(tc.BRANCH_FIELDS):]
    return box, dict(zip(tc.BRANCH_FIELDS, fields, strict=True))


def _plain(text):
    import html as html_lib
    import re

    return html_lib.unescape(re.sub(r"<[^>]+>", " ", text))


@pytest.fixture
def zero_rate(checkpoints, tmp_path, monkeypatch):
    """The trained workspace copied, with a finished base whose optimizer
    ended at a rate of 0 in every parameter group beside the others (a
    schedule that decayed to 0): its own run, listed by the server."""
    import shutil

    import torch

    root = tmp_path / "root"
    shutil.copytree(checkpoints["root"], root)
    ws = root / "data"
    run = ws / "runs" / "gatv2" / "20261004-000000_zero_dddddd" / "checkpoints"
    run.mkdir(parents=True)
    payload = torch.load(ws / "runs" / "gatv2" / DONE_RUN / "checkpoints" / "last.pt",
                         map_location="cpu", weights_only=True)
    for group in payload["optimizer_state_dict"]["param_groups"]:
        group["lr"] = 0.0
    torch.save(payload, run / "last.pt")
    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(root))
    state = tc.workspace_state("data")
    parents, _problem = tc._list_parents(state)
    return {**checkpoints, "root": root, "ws": ws, "state": state, "parents": parents,
            "zero": run / "last.pt", "zero_key": "gatv2/20261004-000000_zero_dddddd/last.pt",
            "done_key": f"gatv2/{DONE_RUN}/last.pt",
            "groups": len(payload["optimizer_state_dict"]["param_groups"])}


def test_the_extra_epoch_rate_shows_only_for_a_parent_whose_rate_ended_at_zero(zero_rate,
                                                                               monkeypatch):
    """Ops-wiring round B1c: the drawer sent no extend_lr, so a finished parent
    whose schedule decayed to 0 could not be extended from the page. The
    preview names the groups whose rate ended at 0 (`zero_rate_groups`); the
    Extra-epoch rate field shows for such a parent only, its value is the
    plan's extend_lr, and the start sends the rate the check judged. A parent
    with no rate at 0 says its rates in the plan's own sentence, and a value
    left in the hidden field is no request."""
    started = []
    monkeypatch.setattr(training_manager, "start_training", _starts(started))
    c = zero_rate
    # A finished parent with its rates: no rate field; the plan says them.
    text, extend, rate, *_rest = _check_all(c, tc.CONTINUE_FROM_LIST, c["done_key"], "",
                                            extend=2)
    assert "Extend training by 2 epochs at a fixed rate of" in _plain(text)
    assert extend["visible"] is True and rate == {"__type__": "update", "visible": False,
                                                 "value": None}
    # The zero-rate parent: refused without the rate, in the drawer's words,
    # with the rate field shown.
    text, extend, rate, *_rest = _check_all(c, tc.CONTINUE_FROM_LIST, c["zero_key"], "",
                                            extend=2)
    assert "Cannot continue." in text and "Give Extra-epoch rate for those groups" in text
    assert "extend_lr" not in text and rate["visible"] is True
    text, extend, rate, *_rest = _check_all(c, tc.CONTINUE_FROM_LIST, c["zero_key"], "",
                                            extend=2, rate=1e-4)
    assert "Extend training by 2 epochs at a fixed rate of 0.0001" in _plain(text)
    assert rate["visible"] is True
    said, *_rest = _resume_all(c, tc.CONTINUE_FROM_LIST, c["zero_key"], "", extend=2, rate=1e-4)
    assert said == tc.notice_html("")
    assert started[-1]["extend_epochs"] == 2 and started[-1]["extend_lr"] == 1e-4
    assert started[-1]["resume_from"] == str(c["zero"])
    # A rate left in the hidden field is not sent for a parent with no rate
    # at 0, nor for an unfinished one.
    _resume_all(c, tc.CONTINUE_FROM_LIST, c["done_key"], "", extend=2, rate=1e-4)
    assert started[-1]["extend_epochs"] == 2 and "extend_lr" not in started[-1]
    _resume_all(c, tc.CONTINUE_FROM_LIST, c["paused_key"], "", rate=1e-4)
    assert "extend_lr" not in started[-1] and "extend_epochs" not in started[-1]
    # The service still refuses a rate sent for a parent with none at 0.
    preview = training_manager.preview_continuation(str(c["finished"]), device="cpu",
                                                    extend_epochs=2,
                                                    requested={"extend_lr": 1e-4})
    assert preview["ok"] is False and "none did" in preview["reason"]


def test_a_same_settings_branch_checks_and_starts_from_the_page(checkpoints, monkeypatch):
    """Ops-wiring round B1c, decisions B1, B2: Branch from the drawer with no
    setting edited. The check is the job runner's branch preview: no setting
    changed, in the page's words; the fields are the settings the parent's
    stage and task may change (the preview's `changeable`), each at the
    parent's value; the start is the runner's one start with `branch_from`,
    nothing else of the fields, and the digest shown."""
    started = []
    monkeypatch.setattr(training_manager, "start_training", _starts(started))
    c = checkpoints
    outputs = _check_all(c, tc.CONTINUE_FROM_LIST, c["paused_key"], "", operation=tc.BRANCH)
    text, extend, rate, mode, _compare = outputs[:5]
    words = _plain(text)
    assert "Checkpoint read; it can be branched on this workspace." in words
    assert ("Plan read from the checkpoint: no setting changed: a new run from these weights "
            "with a new optimizer.") in words
    for repr_mark in ("'", "None", "True", "False", "{", "->"):
        assert repr_mark not in words, repr_mark
    assert extend["visible"] is False and rate["visible"] is False
    preview = training_manager.preview_branch(str(c["paused"]), device="cpu")
    assert mode["operation"] == tc.BRANCH
    assert sorted(mode["parent"]["changeable"]) == sorted(preview["changeable"])
    box, fields = _fields(outputs)
    assert box["visible"] is True
    for name in tc.BRANCH_FIELDS:
        shown = fields[name]["visible"]
        if name not in preview["changeable"]:
            assert shown is False, name                     # a task's own, on a base
        elif name in preview["inherited"] and shown:
            assert fields[name]["value"] == tc.form_value(preview["inherited"][name]), name
    assert fields["num_neighbors"]["value"] == "4, 4" and fields["num_epochs"]["value"] == 2
    assert fields["candidate_sample_size"]["visible"] is False
    # Kept settings, not offered: in the check's own disclosure, none of the fields.
    keeps = text[text.index("Settings this branch keeps"):]
    assert "Split seed" in keeps and "Learning rate" not in keeps
    said, text, *_rest = _resume_all(c, tc.CONTINUE_FROM_LIST, c["paused_key"], "",
                                     operation=tc.BRANCH)
    assert said == tc.notice_html("")
    assert set(started[-1]) - {"__kwargs__"} == {"data_dir", "run_name", "device", "compile",
                                                 "branch_from"}
    assert started[-1]["branch_from"] == str(c["paused"])
    assert started[-1]["__kwargs__"] == {
        "expected_dataset_fingerprint": c["state"]["row"]["dataset_fingerprint"],
        "expected_parent_sha256": c["parents"][c["paused_key"]]["sha256"]}


def test_a_checks_branch_values_are_what_each_field_takes(checkpoints):
    """Found in the browser (B1c): a base records a task model's settings as
    unset, and the check wrote their "off" into those fields, number boxes
    hidden for a base; Gradio's Number refused the text and the whole check
    failed. A check writes only the fields it shows, each value in its
    field's form, so every value goes through its component as Gradio sends
    it, for a base and for a task model's parent."""
    import gradio as gr

    with gr.Blocks() as demo:
        tc.create_training_tab(standalone_context(demo))
    by_id = {getattr(block, "elem_id", None): block for block in demo.blocks.values()}
    c = checkpoints
    outputs = _check_all(c, tc.CONTINUE_FROM_LIST, c["paused_key"], "", operation=tc.BRANCH)
    _box, fields = _fields(outputs)
    preview = training_manager.preview_branch(str(c["paused"]), device="cpu")
    written = {name for name, update in fields.items() if "value" in update}
    assert written and written <= set(preview["changeable"])
    assert {"task_loss_weight", "candidate_sample_size"}.isdisjoint(written)
    for name in written:
        by_id[f"branch_{name}"].postprocess(fields[name]["value"])      # raises if not taken
    # Each field's own form of each kind of value: a list and an optional
    # number as text, a number as a number, None in a number box as empty.
    assert tc._branch_form_value("num_neighbors", [4, -1]) == "4, -1"
    assert tc._branch_form_value("early_stopping_patience", None) == "off"
    assert tc._branch_form_value("early_stopping_patience", 5) == 5      # a text box shows "5"
    assert tc._branch_form_value("task_loss_weight", None) is None
    assert tc._branch_form_value("learning_rate", 0.001) == 0.001
    for name in tc.BRANCH_FIELDS:
        by_id[f"branch_{name}"].postprocess(tc._branch_form_value(name, None))


def test_a_branch_sends_only_the_fields_edited_and_says_its_changes_in_words(checkpoints,
                                                                            monkeypatch):
    """Only a field the operator edited is sent (decision S2), whatever its
    value; a value other than the parent's is the branch's change, said by
    the setting's words with its parent's value and its own in the one value
    words (`renderers.change_value`), never repr; one edited at the parent's
    own value is sent and is no change. An edited field keeps the operator's
    value after the check."""
    started = []
    monkeypatch.setattr(training_manager, "start_training", _starts(started))
    c = checkpoints
    outputs = _check_all(c, tc.CONTINUE_FROM_LIST, c["paused_key"], "", operation=tc.BRANCH,
                         learning_rate=0.0005, scheduler_type="cosine")
    text = outputs[0]
    assert "Plan read from the checkpoint: 2 settings changed: a new run from these weights " \
        "with a new optimizer." in _plain(text)
    assert ('<dt class="shep-k" title="learning_rate">Learning rate</dt><dd class="shep-v">'
            '<code>0.0001</code> → <code>0.0005</code></dd>') in text
    assert ('<dt class="shep-k" title="scheduler_type">Scheduler</dt><dd class="shep-v">'
            '<code>none</code> → <code>cosine</code></dd>') in text
    assert "'" not in _plain(text) and "->" not in _plain(text)
    _box, fields = _fields(outputs)
    assert "value" not in fields["learning_rate"] and "value" not in fields["scheduler_type"]
    # Warmup is read under cosine: shown now; untouched, at the parent's value.
    assert fields["warmup_steps"]["visible"] is True and fields["warmup_steps"]["value"] == 0
    _resume_all(c, tc.CONTINUE_FROM_LIST, c["paused_key"], "", operation=tc.BRANCH,
                learning_rate=0.0005, scheduler_type="cosine")
    sent = {k: v for k, v in started[-1].items()
            if k not in ("data_dir", "run_name", "device", "compile", "branch_from", "__kwargs__")}
    assert sent == {"learning_rate": 0.0005, "scheduler_type": "cosine"}
    # Stated at the parent's value: sent, and no setting changed.
    seed = training_manager.preview_branch(str(c["paused"]), device="cpu")["inherited"]["seed"]
    outputs = _check_all(c, tc.CONTINUE_FROM_LIST, c["paused_key"], "", operation=tc.BRANCH,
                         seed=seed)
    assert "no setting changed" in _plain(outputs[0])
    _resume_all(c, tc.CONTINUE_FROM_LIST, c["paused_key"], "", operation=tc.BRANCH, seed=seed)
    assert started[-1]["seed"] == seed


def test_an_invalid_or_off_list_branch_edit_is_refused_with_the_plans_words(checkpoints,
                                                                           monkeypatch):
    """A value its field spec does not take is refused by that check, the one
    the plan makes (`value_problem`), named by the field's label; a setting
    the parent's stage does not read, and a rule across settings the edit
    breaks, are refused by the job runner's preview in the plan's own words
    (`run_plan.stated_problem`, `cross_field_problems`), each setting it
    leads with named by its label, as the base form names it
    (`labelled_problem`; the B1c review: the rule read "min_lr_ratio must
    be …" here and "Final learning-rate ratio must be …" on the base form).
    Nothing starts, and the fields keep the edit to fix. The reason beside
    Start branch says the check refused a setting edited, and what to do,
    never that the checkpoint cannot be branched (the B1c review)."""
    from src.config.training_fields import cross_field_problems, value_problem
    from src.utils.run_plan import new_run_values, stated_problem

    started = []
    monkeypatch.setattr(training_manager, "start_training", _starts(started))
    c = checkpoints
    bad = _check_all(c, tc.CONTINUE_FROM_LIST, c["paused_key"], "", operation=tc.BRANCH,
                     learning_rate=-1.0)
    words = tc._named("learning_rate", value_problem("learning_rate", -1.0), "Learning rate")
    assert "Cannot branch." in bad[0] and r.e(words) in bad[0], bad[0]
    assert bad[3]["parent"]["refused"] == tc.REFUSED_SETTINGS
    assert all(update == gr_update() for update in bad[-1 - len(tc.BRANCH_FIELDS):])
    off = _check_all(c, tc.CONTINUE_FROM_LIST, c["paused_key"], "", operation=tc.BRANCH,
                     candidate_sample_size=5)
    unread = stated_problem("base", (None,), {"candidate_sample_size": 5})
    assert unread.startswith("candidate_sample_size is not a setting")
    assert "Cannot branch." in off[0]
    assert r.e("Candidate sample size" + unread[len("candidate_sample_size"):]) in off[0], off[0]
    rule = _check_all(c, tc.CONTINUE_FROM_LIST, c["paused_key"], "", operation=tc.BRANCH,
                      scheduler_type="onecycle", min_lr_ratio=0.0)
    settings = {"scheduler_type": "onecycle", "min_lr_ratio": 0.0}
    [problem] = cross_field_problems(new_run_values("base", None, settings), settings)
    base_words = tc.base_start_settings(
        {**dict(zip(tc.base_form_keys(), tc.base_form_defaults(), strict=True)), **settings},
        list(settings))[1]
    assert base_words == [tc.labelled_problem(problem)]              # one rule, one reading
    assert base_words[0].startswith("Final learning-rate ratio must be more than 0 when")
    assert "Cannot branch." in rule[0] and r.e(base_words[0]) in rule[0], rule[0]
    assert "min_lr_ratio must" not in rule[0]
    for edit in ({"learning_rate": -1.0}, {"candidate_sample_size": 5},
                 {"scheduler_type": "onecycle", "min_lr_ratio": 0.0}):
        said, text, _extend, _rate, mode, *_rest = _resume_all(
            c, tc.CONTINUE_FROM_LIST, c["paused_key"], "", operation=tc.BRANCH, **edit)
        assert said == tc.notice_html("") and "Cannot branch." in text
        assert mode["parent"]["refused"] == tc.REFUSED_SETTINGS
        assert tc.continue_refusal(mode["parent"], c["state"], operation=tc.BRANCH) == (
            "the check refuses a setting you edited; change it, then check again.")
        view = tc.run_bar_view({"status": "idle"}, c["state"],
                               {"drawer": True, "operation": tc.BRANCH, "parent": mode["parent"]},
                               access=_allowed(), active=None, gpu=None)
        assert ("Start branch disabled: the check refuses a setting you edited; change it, then "
                "check again.") in view["drawer_reason"]
        assert "cannot be branched" not in view["drawer_reason"]
    assert started == []
    # An edit after a refused check of the settings: the check is stale and
    # no longer disables Start branch, which checks what it sends first.
    text, mode = tc._on_branch_edited({"parent": {**bad[3]["parent"]}})
    assert "A setting changed since the check" in text and not mode["parent"]["refused"]
    assert tc.continue_refusal(mode["parent"], c["state"], operation=tc.BRANCH) is None


def _allowed():
    """D8's answer to an operator's request (`operator_access`)."""
    return tc.operator_access(OP)


def test_an_edit_never_dismisses_a_refusal_of_the_parent_itself(checkpoints):
    """The B1c review: a check that refused the parent itself (here, a file
    other than the one shown) left the Branch box with the earlier read's
    values, and an edit in it replaced the refusal with "A setting changed
    since the check" and enabled Start branch. A refusal no edit can fix
    hides the box, and an edit leaves the refusal where it is said; a
    refusal of the settings keeps the box, the edit to fix (above)."""
    c = checkpoints
    ok = _check_all(c, tc.CONTINUE_FROM_LIST, c["paused_key"], "", operation=tc.BRANCH)
    box, _fields_shown = _fields(ok)
    assert box["visible"] is True
    parents = {key: dict(item) for key, item in c["parents"].items()}
    parents[c["paused_key"]]["sha256"] = "0" * 64             # another file than the one shown
    refused = _check_all(c, tc.CONTINUE_FROM_LIST, c["paused_key"], "", mode=ok[3],
                         operation=tc.BRANCH, parents=parents)
    assert "The checkpoint changed since it was chosen." in refused[0]
    mode = refused[3]
    assert mode["parent"]["refused"] == tc.REFUSED_CHANGED
    box, fields = _fields(refused)
    assert box == gr_update(visible=False)
    assert all(update == gr_update() for update in fields.values())
    text, after = tc._on_branch_edited(mode)
    assert text == gr_update() and after["parent"]["refused"] == tc.REFUSED_CHANGED
    assert tc.continue_refusal(after["parent"], c["state"], operation=tc.BRANCH) == (
        "the checkpoint changed since it was chosen. Refresh the list, then check it again.")
    # A parent the plan refuses, or one not found: the box hidden alike.
    for preview in ({"ok": False, "reason": "weights only"}, {}):
        assert tc._branch_updates(preview, tc.BRANCH, [], _values())[0] == gr_update(
            visible=False)
    nothing = _check_all(c, tc.CONTINUE_FROM_LIST, "gatv2/gone/last.pt", "", mode=ok[3],
                         operation=tc.BRANCH)
    assert "Nothing to check" in nothing[0] and _fields(nothing)[0] == gr_update(visible=False)
    # Continue leaves the Branch box as it is.
    assert tc._branch_updates({"ok": False}, tc.CONTINUE, [], _values())[0] == gr_update()


def test_the_branch_fields_are_the_settings_a_branch_may_change_through_the_one_field():
    """One field per setting any stage's branch may change
    (`recipe.branch_settings`), the one field of its setting
    (`setting_field`: label, kind, format), hidden until a check names it;
    the freeze, the split seed and the model spec, which a branch keeps, are
    none of them (decision B3: early stopping's patience is one)."""
    import gradio as gr

    from src.config.recipe import FINETUNE_TASKS, branch_settings

    every = set(branch_settings("base")).union(*(branch_settings("finetune", task)
                                                  for task in FINETUNE_TASKS))
    assert set(tc.BRANCH_FIELDS) == every
    assert {"frozen_parts", "split_seed", "model_spec"}.isdisjoint(tc.BRANCH_FIELDS)
    assert "early_stopping_patience" in tc.BRANCH_FIELDS
    with gr.Blocks() as demo:
        tc.create_training_tab(standalone_context(demo))
    by_id = {getattr(block, "elem_id", None): block for block in demo.blocks.values()}
    for name in tc.BRANCH_FIELDS:
        field = by_id[f"branch_{name}"]
        assert field.label == r.SETTING_LABELS[name] and field.visible is False, name
        assert f"branch_{name}" in tc.FIELD_HELP
    assert isinstance(by_id["branch_early_stopping_patience"], gr.Textbox)  # "off" or a number
    assert by_id["shep-train-branch"].visible is False
    # Each by its verb, so the choice is one line beside its label (the B1c
    # review: two long options wrapped to three rows at 1850 x 960); what
    # each does is the help's.
    assert by_id["continue_operation"].choices == [("Continue", "continue"),
                                                   ("Branch", "branch")]
    title, text = tc.FIELD_HELP["continue_operation"]
    assert "• Continue: resume a paused run, or extend a finished one" in text
    assert "• Branch: a new run from its weights, with a new optimizer" in text
    # Steps, numbered in the order they are taken (the skill's §4).
    assert by_id["continue_run"].label == tc.STEP_RUN == "1. Run"
    assert by_id["checkpoint_dropdown"].label == tc.STEP_CHECKPOINT == "2. Checkpoint"
    assert "3. Check" in by_id["shep-train-step-check"].value
    assert "4. Start" in by_id["shep-train-step-start"].value
