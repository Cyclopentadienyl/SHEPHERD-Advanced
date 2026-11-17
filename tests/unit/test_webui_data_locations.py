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
from src.config.deployment_settings import ENV_SETTINGS_FILE, SUGGESTED  # noqa: E402
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
        busy = None

        def run_while_idle(self, fn):
            return (self.busy, None) if self.busy else (None, fn())

        def busy_reason(self):
            return self.busy

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
        assert f"~/SHEPHERD-data/{SUGGESTED[key]}" in row["suggest"]["value"]
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


def test_the_not_set_words_name_the_folder_settings_suggests(dl, monkeypatch):
    """T08 (first run): the pages that need a workspaces folder said "for
    example this repository's data/workspaces" while Settings offered Create
    suggested folder ~/SHEPHERD-data/workspaces: two folders for one choice.
    The words Datasets shows with no folder set (the server's own reason,
    `read_workspaces`) and setup_demo.py's refusal (which quotes them) now
    name the folder the panel offers, drawn from one statement in the config
    layer, and no other."""
    import re

    from src.webui.components.workspaces import read_workspaces

    rows, _panel = _parts(dl, dl._on_locations_open(operator_request(monkeypatch)))
    offered = re.sub(r"<[^>]+>", "", rows["workspaces_root"]["suggest"]["value"])
    assert offered == "~/SHEPHERD-data/workspaces"
    said = read_workspaces()["problem"]
    assert "No workspaces folder is set" in said
    # The same folder, in words the page's path redaction leaves whole.
    assert "SHEPHERD-data/workspaces in the service account's home folder" in said
    assert offered[len("~/"):] in said and "path outside" not in said
    assert "data/workspaces" not in said.replace("SHEPHERD-data/workspaces", "")
    assert "repository" not in said
    # The Create button makes that very folder, under this account's home.
    _parts(dl, dl._on_create("workspaces_root", operator_request(monkeypatch)))
    assert (os.path.join(os.environ["HOME"], "SHEPHERD-data", "workspaces")
            == str(server_settings.workspaces_root().value))


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
    is removed. An environment-configured deployment works unchanged.

    Not applied while the setting is in place (T03 review, P2): applied, a
    command run by hand with no variable would read the saved folder while
    the service reads its own, and for the workspaces folder take another
    GPU lock. So the panel offers no Apply for it, and a start without the
    setting applies it."""
    from src.api.services import data_locations as service

    started, chosen = tmp_path / "started", tmp_path / "chosen"
    started.mkdir()
    chosen.mkdir()
    monkeypatch.setenv("SHEPHERD_REGISTRY", str(started))
    request = operator_request(monkeypatch)
    rows, panel = _parts(dl, dl._on_save("registry", str(chosen), request))
    assert "start-up setting stays in effect" in rows["registry"]["result"]["value"]
    state = rows["registry"]["state"]["value"]
    effect, saved = state.split(">Saved<")
    assert str(started) in effect and "Set when the service started" in effect
    assert str(chosen) in saved and dl.OVERRIDDEN in saved
    assert rows["registry"]["box"]["interactive"] is True
    assert rows["registry"]["save"]["interactive"] is True
    assert panel["apply"]["visible"] is False
    assert "Saved, not in effect yet" not in panel["notice"]["value"]
    assert server_settings.cohort_registry().value == started.resolve()
    without = {ENV_SETTINGS_FILE: os.environ[ENV_SETTINGS_FILE]}
    assert server_settings.cohort_registry(without).unset     # a hand command: unchanged
    monkeypatch.delenv("SHEPHERD_REGISTRY")
    assert service.apply_saved().applied                      # a start without it
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
        # The reason is said once, in the notice; the save's line points to it.
        assert dl.WAITS_ABOVE in rows["workspaces_root"]["result"]["value"]
        assert "Saved, not in effect yet" in rows["workspaces_root"]["state"]["value"]
        assert "Saved, not in effect yet: Workspaces folder" in panel["notice"]["value"]
        assert (f"The GPU lock {old.resolve() / LOCK_FILE_NAME} is held by a training job (by "
                "hand") in panel["notice"]["value"]
        assert panel["apply"]["visible"] is True and panel["apply"]["interactive"] is True
        assert server_settings.workspaces_root().value == old.resolve()
        rows, panel = _parts(dl, dl._on_apply_saved(request))
        assert dl.WAITS_ABOVE in panel["status"]["value"]
        assert "is held by a training job" in panel["notice"]["value"]
    rows, panel = _parts(dl, dl._on_apply_saved(request))
    assert "Applied" in panel["status"]["value"] and panel["apply"]["visible"] is False
    assert server_settings.workspaces_root().value == new.resolve()


def test_the_pending_notice_says_the_reason_apply_would_give(dl, monkeypatch, tmp_path):
    """T03 review leftover: the notice read the GPU lock only, so while this
    service's job runner held a job whose process had not taken the lock yet
    (or was stopping after it gave it back), or the backend was stopping, it
    said "press Apply" beside a Save and an Apply refused for that job. It now
    says the reason from the same check Apply makes (`data_locations.view`:
    the runner's slot, then the old values' GPU locks), in Apply's words."""
    from src.api.services import data_locations as service

    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    request = operator_request(monkeypatch)
    _parts(dl, dl._on_save("workspaces_root", str(old), request))
    busy = "training is running in this service's job runner (job training-1)"
    monkeypatch.setattr(service.training_manager, "busy", busy)
    rows, panel = _parts(dl, dl._on_save("workspaces_root", str(new), request))
    said = ("Training is running in this service&#x27;s job runner (job training-1), so the "
            "change applies when it ends: press Apply then, or it applies at the next start.")
    assert said in panel["notice"]["value"], panel["notice"]["value"]
    assert "Press Apply, or it applies" not in panel["notice"]["value"]
    assert dl.WAITS_ABOVE in rows["workspaces_root"]["result"]["value"]
    rows, panel = _parts(dl, dl._on_apply_saved(request))
    assert said in panel["notice"]["value"] and dl.WAITS_ABOVE in panel["status"]["value"]
    # The route says the same reason (read-only: no path in it to hide).
    from src.utils.deployment_view import as_dict

    assert as_dict(service.view(), paths_shown=False)["pending_reason"] == said.replace(
        "&#x27;", "'")
    monkeypatch.setattr(service.training_manager, "busy", None)
    rows, panel = _parts(dl, dl._on_apply_saved(request))
    assert "Applied" in panel["status"]["value"]
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


def _cells_text(html: str) -> dict:
    """A summary list's values as text, by key, read from the HTML alone (no
    style sheet): the text nodes as they come, a ``<br>`` a line break, as
    find-in-page, a copy and a screen reader read them; spaces collapsed."""
    from html.parser import HTMLParser

    class Cells(HTMLParser):
        def __init__(self):
            super().__init__()
            self.cells, self.key, self.cell, self.text = {}, None, None, []

        def handle_starttag(self, tag, attrs):
            if tag in ("dt", "dd"):
                self.cell, self.text = tag, []
            elif tag == "br":
                self.text.append("\n")

        def handle_endtag(self, tag):
            if tag == "dt":
                self.key = "".join(self.text).strip()
            elif tag == "dd":
                lines = "".join(self.text).split("\n")
                self.cells[self.key] = "\n".join(" ".join(line.split()) for line in lines)
            if tag in ("dt", "dd"):
                self.cell = None

        def handle_data(self, data):
            if self.cell:
                self.text.append(data)

    parser = Cells()
    parser.feed(html)
    return parser.cells


def test_the_value_in_effect_and_its_source_read_as_two_facts(dl, monkeypatch, tmp_path):
    """T08 flow (T08-flow-10): the In effect cell joined the path's span and
    its source's with nothing between them, the gap on screen being the
    path's CSS margin, so its text read "…/existing-wsrootSaved in
    Settings" and "…/ws-overrideSet when the service started" to a copy,
    find-in-page and a screen reader. Read from the HTML alone, the value
    in effect (a path, a list of them, a refused value, a hidden one) and its
    source are two facts; the Saved row's value and its mark are too."""
    root, registry, chosen = tmp_path / "ws-root", tmp_path / "ws-registry", tmp_path / "chosen"
    for folder in (root, registry, chosen):
        folder.mkdir()
    evidence = [tmp_path / "ev-a", tmp_path / "ev-b"]
    for folder in evidence:
        folder.mkdir()
    request = operator_request(monkeypatch)
    _parts(dl, dl._on_save("workspaces_root", str(root), request))
    _parts(dl, dl._on_save("registry", str(chosen), request))
    monkeypatch.setenv("SHEPHERD_REGISTRY", str(registry))
    monkeypatch.setenv("SHEPHERD_INTAKE_DIR", str(tmp_path / "gone"))
    monkeypatch.setenv("SHEPHERD_TRAINING_EVIDENCE_DIRS", os.pathsep.join(map(str, evidence)))
    rows, _ = _parts(dl, dl._on_locations_open(request))
    cells = {key: _cells_text(rows[key]["state"]["value"]) for key in KEYS}
    saved, started = "Saved in Settings", "Set when the service started"
    assert cells["workspaces_root"]["In effect"] == f"{root.resolve()} {saved}"
    assert cells["registry"]["In effect"] == f"{registry.resolve()} {started}"
    assert cells["registry"]["Saved"] == f"{chosen}\n{dl.OVERRIDDEN}"
    assert cells["intake_dir"]["In effect"].endswith(f"Not usable {tmp_path / 'gone'} {started}")
    assert cells["training_evidence_dirs"]["In effect"] == (
        f"{evidence[0].resolve()}\n{evidence[1].resolve()} {started}")
    hidden = _parts(dl, dl._on_locations_open(gradio_request(server=("192.0.2.10", 8264))))[0]
    assert _cells_text(hidden["workspaces_root"]["state"]["value"])["In effect"] == (
        f"{dl.PATH_HIDDEN} {saved}")


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
    """Stop service asks in the panel's one question row (`_ask`, shared with
    Restart Backend since R-12): its own question and its own confirming
    button, never Restart's."""
    import src.webui.components.runtime_settings as rs

    stops = []
    monkeypatch.setattr(rs, "stop_backend", lambda: stops.append(1) or {"success": True})
    monkeypatch.setattr(rs, "is_training_active", lambda: False)
    request = operator_request(monkeypatch)
    row, question, restart_now, stop_now, status = rs._on_stop_service_request(request)
    assert row["visible"] is True and status == "" and stops == []
    assert "Stop the service?" in question["value"]
    assert (restart_now["visible"], stop_now["visible"]) == (False, True)
    assert rs._on_confirm_cancel()[0]["visible"] is False and stops == []
    row, *_parts, status = rs._on_stop_service(request)
    assert stops == [1] and row["visible"] is False and rs.STOP_DONE in status
    monkeypatch.setattr(rs, "is_training_active", lambda: True)
    row, *_parts, status = rs._on_stop_service_request(request)
    assert row["visible"] is False and rs.RESTART_BUSY in status
    refused = gradio_request(server=("192.0.2.10", 8264))
    for act in (rs._on_stop_service_request, rs._on_stop_service):
        row, *_parts, status = act(refused)
        assert row["visible"] is False and "Refused: " in status
    assert stops == [1]


#: Settings at a phone's width and at the owner's window: each data
#: location's box, how wide it is and whether its placeholder fits in it.
_BOXES_JS = r"""
const { chromium } = require('playwright');
const QUEUE_JS = __QUEUE_JS__;
(async () => {
  const browser = await chromium.launch();
  const out = {};
  for (const [w, h] of [[390, 844], [960, 1080], [1850, 960]]) {
    const page = await browser.newPage({ viewport: { width: w, height: h } });
    await page.addInitScript(QUEUE_JS);
    await page.goto(process.argv[2]);
    await page.click('#shep-tab-settings-button');
    await page.waitForSelector('#dl_box_workspaces_root textarea, #dl_box_workspaces_root input',
                               { state: 'visible', timeout: 60000 });
    await page.waitForFunction(() => window.__shepQueue.idle(), null, { timeout: 60000 });
    out[`${w}x${h}`] = await page.evaluate(() => {
      const boxes = {};
      for (const key of ['workspaces_root', 'registry', 'intake_dir', 'training_evidence_dirs']) {
        const f = document.querySelector(`#dl_box_${key} textarea, #dl_box_${key} input`);
        const r = f.getBoundingClientRect();
        const c = document.createElement('canvas').getContext('2d');
        c.font = getComputedStyle(f).font;
        const s = getComputedStyle(f);
        const room = f.clientWidth - parseFloat(s.paddingLeft) - parseFloat(s.paddingRight);
        boxes[key] = { width: Math.round(r.width),
                       hint_fits: c.measureText(f.placeholder.split('\n')[0]).width <= room + 0.5 };
      }
      return { boxes, sideways: document.documentElement.scrollWidth > document.documentElement.clientWidth };
    });
    await page.close();
  }
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_every_path_box_keeps_room_for_a_path_at_every_window(dl, tmp_path, monkeypatch):
    """Verification of fcc3dea (V-RA-1): at 390x844 each "Set to" box was
    40 px wide in every state, its placeholder drawn as "Fc", since the set
    row never wrapped and the box took what its key and buttons left. A box
    now keeps room for a path (16em, or its whole row), its buttons wrapping
    under it where the row cannot hold them; at 960x1080 and 1850x960 the
    row stays one line (an operator's Settings, every answer in). Fails on
    fcc3dea."""
    from src.webui.app import create_gradio_app, mount_options
    from tests.fixtures.browser import QUEUE_JS, node_with_playwright, run_node
    from tests.fixtures.operator_access import operator_mode
    from tests.fixtures.served_page import served

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    operator_mode(monkeypatch)
    monkeypatch.setenv(ENV_SETTINGS_FILE, str(tmp_path / "settings.json"))
    with served(create_gradio_app(), **mount_options()) as base:
        seen = run_node(_BOXES_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS)), tmp_path,
                        base[:-len("gradio_api")])
    for size, at in seen.items():
        assert at["sideways"] is False, (size, at)
        for key, box in at["boxes"].items():
            assert box["hint_fits"] and box["width"] >= 200, (size, key, box)
