"""
⚙ Settings › Data locations (UI-06): chosen and saved in the dashboard.
=======================================================================
The owner's decision of 2026-11-16 withdrew "deployment settings come only
from environment variables". Before it, Settings had no way to set a data
location: a clinician who started the dashboard by double click saw no
dataset and every job refused, with words telling them to set ``SHEPHERD_*``
variables before the start. These pin the panel that replaced that:

- a first start opens with nothing set, says so once, and offers a suggested
  folder per required row, created only on its click (0700 for the registry);
- Save writes the saved file through the one writer, applies it at once, and
  the server's own lists (Datasets) read it with no restart;
- a row the environment sets keeps its box and Save (they change the saved
  value only) and never calls the saved value in effect;
- a pending change says why, with Apply; an unreadable file has one notice
  and Replace, Save off and Create hidden;
- a read-only request sees no path, every control locked with D8's reason;
- every value shown is escaped; Stop service asks before it stops.

Every test reads and writes a settings file and folders of its own
(`tests/conftest.py` points ``SHEPHERD_SETTINGS_FILE`` at an empty directory;
``HOME`` here is the test's own).

Module: tests/unit/test_webui_data_locations.py
"""
from __future__ import annotations

import json
import os
import stat

import pytest

gr = pytest.importorskip("gradio")

from src.config import server_settings  # noqa: E402
from src.config.deployment_settings import ENV_SETTINGS_FILE  # noqa: E402
from tests.fixtures.operator_access import gradio_request, operator_request  # noqa: E402

KEYS = ("workspaces_root", "registry", "intake_dir", "training_evidence_dirs")
ENV = {"workspaces_root": "SHEPHERD_WORKSPACES_ROOT", "registry": "SHEPHERD_REGISTRY",
       "intake_dir": "SHEPHERD_INTAKE_DIR",
       "training_evidence_dirs": "SHEPHERD_TRAINING_EVIDENCE_DIRS"}


@pytest.fixture
def dl(monkeypatch, tmp_path):
    """The panel with nothing set: no data location in the environment, a
    home of the test's own (where Create suggests its folders), no GPU job."""
    import src.webui.components.data_locations as module
    from src.api.services import data_locations as service

    for variable in ENV.values():
        monkeypatch.delenv(variable, raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    class Idle:
        @staticmethod
        def run_while_idle(fn):
            return None, fn()

    monkeypatch.setattr(service, "training_manager", Idle())
    return module


def _parts(module, outputs):
    """The panel's outputs by row and part (`ROW_PARTS`, `PANEL_PARTS`)."""
    rows, index = {}, 0
    for key in KEYS:
        rows[key] = {}
        for part in module.ROW_PARTS:
            rows[key][part] = outputs[index]
            index += 1
    panel = {name: outputs[index + i] for i, name in enumerate(module.PANEL_PARTS)}
    assert index + len(module.PANEL_PARTS) == len(outputs)
    return rows, panel


def _saved_file():
    return json.loads(open(os.environ[ENV_SETTINGS_FILE], encoding="utf-8").read())


def test_the_settings_page_has_the_data_locations_panel_beside_runtime_settings():
    """The page Settings is: Data locations on the left, Runtime settings on
    the right, in one row; each row's box, Check and Save; one Apply and one
    Replace for the panel; the panel read when Settings is shown."""
    from src.webui.app import create_gradio_app

    app = create_gradio_app()
    ids = {getattr(b, "elem_id", None) for b in app.blocks.values()}
    for key in KEYS:
        assert {f"dl_box_{key}", f"dl_check_{key}", f"dl_save_{key}",
                f"dl_state_{key}"} <= ids, key
    assert {"dl_create_workspaces_root", "dl_create_registry", "dl_create_intake_dir",
            "dl_apply", "dl_repair", "dl_notice", "runtime_stop"} <= ids
    cols = next(b for b in app.blocks.values() if getattr(b, "elem_id", None) == "shep-settings-cols")
    shown = [child.elem_id for child in cols.children if not isinstance(child, gr.State)]
    assert shown == ["shep-dl", "shep-settings"]
    tab = next(b for b in app.blocks.values()
               if isinstance(b, gr.Tab) and b.elem_id == "shep-tab-settings")
    opens = [fn for fn in app.fns.values() if (tab._id, "select") in fn.targets
             and fn.name == "_on_locations_open"]
    assert len(opens) == 1 and opens[0].queue is False


def test_a_first_start_says_nothing_is_set_and_offers_the_suggested_folders(dl, monkeypatch):
    rows, panel = _parts(dl, dl._on_locations_open(operator_request(monkeypatch)))
    assert dl.NOTHING_SET in panel["notice"]["value"]
    assert panel["apply"]["visible"] is False and panel["repair"]["visible"] is False
    for key in ("workspaces_root", "registry", "intake_dir"):
        row = rows[key]
        assert "Not set" in row["state"]["value"], key
        assert row["create"]["visible"] is True and row["create"]["interactive"] is True
        assert row["create"]["value"] == dl.CREATE_LABEL
        assert f"~/SHEPHERD-data/{dl.SUGGESTED[key]}" in row["suggest"]["value"]
        assert row["box"]["interactive"] is True and row["save"]["interactive"] is True
        assert ">Set to<" in row["key"]["value"]
    evidence = rows["training_evidence_dirs"]
    assert dl.EVIDENCE_NONE in evidence["state"]["value"]
    assert evidence["create"]["visible"] is False
    # The variable is named only under Advanced, never in the row's facts.
    assert "SHEPHERD_REGISTRY" not in rows["registry"]["state"]["value"]
    assert "Advanced: source" in rows["registry"]["adv"]["value"]
    assert "SHEPHERD_REGISTRY" in rows["registry"]["adv"]["value"]
    # Nothing was created or saved by opening the page.
    assert not (os.path.exists(os.environ[ENV_SETTINGS_FILE])
                or os.path.exists(os.path.join(os.environ["HOME"], "SHEPHERD-data")))


def test_create_makes_the_suggested_folder_on_its_click_saves_and_applies_it(dl, monkeypatch):
    request = operator_request(monkeypatch)
    rows, panel = _parts(dl, dl._on_create("registry", request))
    folder = os.path.join(os.environ["HOME"], "SHEPHERD-data", "registry")
    assert stat.S_IMODE(os.stat(folder).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(os.path.dirname(folder)).st_mode) == 0o700
    assert "Created ~/SHEPHERD-data/registry (only this account can open it)" in (
        rows["registry"]["result"]["value"])
    assert "Saved and in effect" in rows["registry"]["result"]["value"]
    assert _saved_file()["applied"]["registry"] == folder
    setting = server_settings.cohort_registry()
    assert setting.source == "saved" and str(setting.value) == os.path.realpath(folder)
    assert "Saved in Settings" in rows["registry"]["state"]["value"]
    assert rows["registry"]["create"]["visible"] is False
    # The workspaces folder is not patient-level data: the account's umask.
    _parts(dl, dl._on_create("workspaces_root", request))
    assert os.path.isdir(os.path.join(os.environ["HOME"], "SHEPHERD-data", "workspaces"))


def test_a_saved_workspaces_folder_is_listed_by_datasets_with_no_restart(dl, monkeypatch,
                                                                       tmp_path):
    """The Done-when of FL-UI-06: chosen in Settings, saved, and the server's
    own list (the one Datasets reads) has it, from the saved file alone."""
    from src.webui.components.workspaces import read_workspaces

    root = tmp_path / "data" / "workspaces"
    (root / "demo").mkdir(parents=True)
    request = operator_request(monkeypatch)
    rows, _panel = _parts(dl, dl._on_check("workspaces_root", f"  {root}  ", request))
    said = rows["workspaces_root"]["result"]["value"]
    assert "Usable" in said and str(root) in said and "1 workspace" in said
    assert not os.path.exists(os.environ[ENV_SETTINGS_FILE])  # Check saves nothing
    rows, panel = _parts(dl, dl._on_save("workspaces_root", str(root), request))
    assert "Saved and in effect" in rows["workspaces_root"]["result"]["value"]
    assert rows["workspaces_root"]["box"]["value"] == ""
    assert ">Change to<" in rows["workspaces_root"]["key"]["value"]
    assert [row["name"] for row in read_workspaces()["rows"]] == ["demo"]
    assert dl.NOTHING_SET not in panel["notice"]["value"]


def test_a_refused_value_is_not_saved_and_says_why(dl, monkeypatch, tmp_path):
    request = operator_request(monkeypatch)
    rows, _ = _parts(dl, dl._on_save("registry", "relative/dir", request))
    assert "Not saved" in rows["registry"]["result"]["value"]
    assert "not an absolute path" in rows["registry"]["result"]["value"]
    assert not os.path.exists(os.environ[ENV_SETTINGS_FILE])
    rows, _ = _parts(dl, dl._on_check("intake_dir", str(tmp_path / "missing"), request))
    assert "Not usable" in rows["intake_dir"]["result"]["value"]
    rows, _ = _parts(dl, dl._on_save("workspaces_root", "   ", request))
    assert "Nothing saved" in rows["workspaces_root"]["result"]["value"]


def test_an_environment_override_keeps_the_box_and_never_claims_the_saved_value(dl, monkeypatch,
                                                                              tmp_path):
    """Syncthing's form: the start-up setting wins and the page says so; the
    box and Save change the saved value only, which applies once the setting
    is removed. An environment-configured deployment works unchanged."""
    started, chosen = tmp_path / "started", tmp_path / "chosen"
    started.mkdir()
    chosen.mkdir()
    monkeypatch.setenv("SHEPHERD_REGISTRY", str(started))
    request = operator_request(monkeypatch)
    rows, _ = _parts(dl, dl._on_save("registry", str(chosen), request))
    assert "start-up setting stays in effect" in rows["registry"]["result"]["value"]
    state = rows["registry"]["state"]["value"]
    effect, saved = state.split(">Saved<")
    assert str(started) in effect and "Set when the service started" in effect
    assert str(chosen) in saved and dl.OVERRIDDEN in saved
    assert rows["registry"]["box"]["interactive"] is True
    assert rows["registry"]["save"]["interactive"] is True
    assert server_settings.cohort_registry().value == started.resolve()
    monkeypatch.delenv("SHEPHERD_REGISTRY")
    assert server_settings.cohort_registry().value == chosen.resolve()


def test_a_refused_value_is_never_quoted_to_a_read_only_request(dl, monkeypatch, tmp_path):
    """T03 review (P3): a relative or ``~`` value set at start was quoted in
    its row's Problem to a read-only request, since the redaction finds only
    an absolute path. Hidden, the row says it is refused in fixed words."""
    monkeypatch.setenv("SHEPHERD_REGISTRY", "clinic/MRN 0012345/registry")
    monkeypatch.setenv("SHEPHERD_INTAKE_DIR", "~/intake/Jane Doe")
    outputs = dl._on_locations_open(gradio_request(server=("192.0.2.10", 8264)))
    said = json.dumps(outputs, ensure_ascii=False)
    for given in ("clinic", "MRN", "Jane", "Doe"):
        assert given not in said, given
    rows, _ = _parts(dl, outputs)
    assert ("Results registry set when the service started is refused: an operator sees why "
            "in Settings › Data locations.") in rows["registry"]["state"]["value"]
    assert "Not usable" in rows["intake_dir"]["state"]["value"]
    operator = _parts(dl, dl._on_locations_open(operator_request(monkeypatch)))[0]
    assert "is not an absolute path" in operator["registry"]["state"]["value"]


def test_a_change_waits_while_the_old_gpu_lock_is_held_and_apply_applies_it(dl, monkeypatch,
                                                                           tmp_path):
    from src.utils.gpu_lock import LOCK_FILE_NAME, GpuLock

    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    request = operator_request(monkeypatch)
    _parts(dl, dl._on_save("workspaces_root", str(old), request))
    with GpuLock(old / LOCK_FILE_NAME, job_type="training", command="by hand").acquire():
        rows, panel = _parts(dl, dl._on_save("workspaces_root", str(new), request))
        assert "Saved, not applied yet" in rows["workspaces_root"]["result"]["value"]
        assert "Saved, not in effect yet" in rows["workspaces_root"]["state"]["value"]
        assert "Saved, not in effect yet: Workspaces folder" in panel["notice"]["value"]
        assert "The GPU lock is held by" in panel["notice"]["value"]
        assert panel["apply"]["visible"] is True and panel["apply"]["interactive"] is True
        assert server_settings.workspaces_root().value == old.resolve()
        rows, panel = _parts(dl, dl._on_apply_saved(request))
        assert "not applied yet" in panel["status"]["value"]
    rows, panel = _parts(dl, dl._on_apply_saved(request))
    assert "Applied" in panel["status"]["value"] and panel["apply"]["visible"] is False
    assert server_settings.workspaces_root().value == new.resolve()


def test_an_unreadable_file_has_one_notice_and_replace_keeps_a_copy(dl, monkeypatch, tmp_path):
    path = os.environ[ENV_SETTINGS_FILE]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("{not json")
    request = operator_request(monkeypatch)
    rows, panel = _parts(dl, dl._on_locations_open(request))
    assert dl.UNREADABLE in panel["notice"]["value"] and path in panel["notice"]["value"]
    assert panel["repair"]["visible"] is True and panel["repair"]["value"] == dl.REPAIR_LABEL
    for key in KEYS:
        assert rows[key]["save"]["interactive"] is False, key
        assert rows[key]["create"]["visible"] is False, key
        # Said once, in the notice; the rows say only that it is unknown.
        assert "cannot be read (" not in rows[key]["state"]["value"], key
    assert dl.UNKNOWN in rows["workspaces_root"]["state"]["value"]
    rows, panel = _parts(dl, dl._on_replace_unreadable(request))
    assert "renamed to" in panel["status"]["value"]
    kept = [name for name in os.listdir(os.path.dirname(path)) if ".unreadable-" in name]
    assert len(kept) == 1
    assert open(os.path.join(os.path.dirname(path), kept[0])).read() == "{not json"
    assert rows["registry"]["save"]["interactive"] is True
    assert dl.NOTHING_SET in panel["notice"]["value"]


def test_a_read_only_request_sees_no_path_and_every_control_locked(dl, monkeypatch, tmp_path):
    """Read-only: what is set and from where stay; no path, no problem text
    naming one; every box and button off with the lock; D8's reason once."""
    root, registry = tmp_path / "ws-secret", tmp_path / "reg-secret"
    root.mkdir()
    registry.mkdir()
    request = operator_request(monkeypatch)
    _parts(dl, dl._on_save("workspaces_root", str(root), request))
    monkeypatch.setenv("SHEPHERD_REGISTRY", str(registry))
    refused = gradio_request(server=("192.0.2.10", 8264))
    outputs = dl._on_locations_open(refused)
    said = json.dumps(outputs, ensure_ascii=False)
    assert str(tmp_path) not in said and "secret" not in said
    rows, panel = _parts(dl, outputs)
    assert "Operator controls only: request not local." in panel["notice"]["value"]
    assert dl.PATHS_OPERATOR_ONLY in panel["notice"]["value"]
    assert dl.PATH_HIDDEN in rows["workspaces_root"]["state"]["value"]
    assert "Saved in Settings" in rows["workspaces_root"]["state"]["value"]
    assert "Set when the service started" in rows["registry"]["state"]["value"]
    for key in KEYS:
        row = rows[key]
        assert row["box"]["interactive"] is False and row["box"]["value"] == ""
        for part, label in (("check", dl.CHECK_LABEL), ("save", dl.SAVE_LABEL)):
            assert (row[part]["value"], row[part]["interactive"]) == (f"🔒 {label}", False)
    assert rows["intake_dir"]["create"]["value"] == f"🔒 {dl.CREATE_LABEL}"
    # Each act checks its own request: refused, nothing read or written.
    before = _saved_file()
    for act in (lambda: dl._on_save("workspaces_root", str(registry), refused),
                lambda: dl._on_check("workspaces_root", str(registry), refused),
                lambda: dl._on_create("intake_dir", refused),
                lambda: dl._on_apply_saved(refused),
                lambda: dl._on_replace_unreadable(refused)):
        assert "Refused: " in json.dumps(act(), ensure_ascii=False)
    assert _saved_file() == before
    assert not os.path.exists(os.path.join(os.environ["HOME"], "SHEPHERD-data"))


def test_every_value_the_panel_shows_is_escaped(dl, monkeypatch, tmp_path):
    hostile = "<img src=x onerror=alert(1)>"
    folder = tmp_path / hostile.replace("/", "")
    folder.mkdir()
    request = operator_request(monkeypatch)
    for outputs in (dl._on_check("workspaces_root", str(folder), request),
                    dl._on_save("workspaces_root", str(folder), request),
                    dl._on_check("registry", f"rel/{hostile}", request)):
        said = json.dumps(outputs, ensure_ascii=False)
        assert "<img" not in said and "&lt;img src=x onerror=alert(1)&gt;" in said


def test_the_evidence_folders_are_saved_as_a_list_one_per_line(dl, monkeypatch, tmp_path):
    first, second = tmp_path / "e1", tmp_path / "e2"
    first.mkdir()
    second.mkdir()
    request = operator_request(monkeypatch)
    rows, _ = _parts(dl, dl._on_save("training_evidence_dirs", f"{first}\n\n {second} \n",
                                     request))
    assert _saved_file()["applied"]["training_evidence_dirs"] == [str(first), str(second)]
    assert rows["training_evidence_dirs"]["box"]["value"] == f"{first}\n{second}"
    assert server_settings.training_evidence_dirs().value == (first.resolve(), second.resolve())
    _parts(dl, dl._on_save("training_evidence_dirs", "", request))
    assert server_settings.training_evidence_dirs().value == ()


def test_stop_service_asks_once_then_stops_through_the_orderly_stop(monkeypatch):
    import src.webui.components.runtime_settings as rs

    stops = []
    monkeypatch.setattr(rs, "stop_backend", lambda: stops.append(1) or {"success": True})
    monkeypatch.setattr(rs, "is_training_active", lambda: False)
    request = operator_request(monkeypatch)
    row, status = rs._on_stop_service_request(request)
    assert row["visible"] is True and status == "" and stops == []
    assert rs._on_stop_service_cancel()[0]["visible"] is False and stops == []
    row, status = rs._on_stop_service(request)
    assert stops == [1] and row["visible"] is False and rs.STOP_DONE in status
    monkeypatch.setattr(rs, "is_training_active", lambda: True)
    row, status = rs._on_stop_service_request(request)
    assert row["visible"] is False and rs.RESTART_BUSY in status
    refused = gradio_request(server=("192.0.2.10", 8264))
    for act in (rs._on_stop_service_request, rs._on_stop_service):
        row, status = act(refused)
        assert row["visible"] is False and "Refused: " in status
    assert stops == [1]
