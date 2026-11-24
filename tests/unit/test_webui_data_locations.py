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
import re
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


def test_a_row_reads_top_to_bottom_and_check_and_save_move_together(dl):
    """The UI sweep of 2026-11-22: each row's name and what needs it stood in
    a column beside its box and buttons (the registry's 120 characters beside
    its box at 1850 x 960: a message never shares a row with fields or
    buttons); at 390 x 844 Save wrapped alone and Check stood centred in a
    row three lines tall; a window media query (899 px) chose the narrow
    form. Now the name and its words are above the row's facts, Check and
    Save are one group that wraps whole (no window media query), and the
    panel's heading is
    the go link's focus target (``#shep-dl`` and its ``h2``, a heading that
    takes focus, no control).
    The sweep's L7 (SW-LK1): the name took a line of its own above the facts
    at every width, the right half of it empty, and Create a line of its own
    under the buttons: the four rows no longer fit the owner's window (1850 x
    960, 223 px under the panel's foot at first use). Each row's head now
    holds its name and, beside it, what is in effect and its "Advanced:
    source ›" (words and a disclosure, no field or button); Create and its
    suggestion are one more whole group of the Set-to bar, after Check and
    Save. The sweep's L7 check (C3): the rows share their columns, Create after Save where the
    bar holds a box of `SET_BOX` and its buttons, else under the box
    (`test_in_a_browser_the_rows_share_their_columns_and_a_path_reads`)."""
    from src.webui.app import create_gradio_app

    css = " ".join(dl.PAGE_CSS.split())
    assert "@media" not in css
    assert (f"{dl._PAGE} .shep-dl-row > :is(.shep-dl-head, .shep-dl-main) "
            "{ flex: 1 1 100% !important;") in css
    # The key above its box in a panel under 27em served windows under the display floor only
    # (the panel is about 700 px wide at the floor): retired. The one container query is the
    # Set-to bar's own width, where every row's box is one width (the sweep's L7, C3).
    assert not hasattr(dl, "DL_NARROW")
    assert re.findall(r"@container \(([^)]*)\)", css) == [f"min-width: {dl.SET_WIDE}"]
    assert (f'.shep-dl-field > .block {{ flex: 0 1 {dl.SET_BOX} '
            "!important; }") in css
    app = create_gradio_app()
    blocks = {getattr(b, "elem_id", None): b for b in app.blocks.values()}
    for key in KEYS:
        check, save = blocks[f"dl_check_{key}"], blocks[f"dl_save_{key}"]
        group = check.parent
        assert group is save.parent and {"shep-dl-acts", "shep-grp"} <= set(group.elem_classes)
        # A read-only row's short phrase is a whole group of the bar after Check and Save's,
        # beside them where the line holds it (the sweep's L7 check, D6), never inside their
        # group, which it would take off the box's line (K4).
        bar = group.parent
        lock = blocks[f"dl_lock_{key}"]
        assert lock.parent is bar and bar.children.index(lock) == bar.children.index(group) + 1
        assert blocks[f"dl_box_{key}"].parent is bar      # the box beside the group
        create = blocks[f"dl_create_{key}"].parent
        assert create.parent is bar and {"shep-dl-create", "shep-grp"} <= set(create.elem_classes)
        assert blocks[f"dl_suggest_{key}"].parent is create
        facts = blocks[f"dl_state_{key}"].parent
        assert blocks[f"dl_adv_{key}"].parent is facts and "shep-dl-facts" in facts.elem_classes
        head = facts.parent
        assert "shep-dl-head" in head.elem_classes and head.parent is blocks[f"dl_row_{key}"]
        assert ["shep-dl-left" in (c.elem_classes or []) for c in head.children] == [True, False]
    head = blocks["shep-dl-hd"].value
    assert '<h2 tabindex="-1">Data locations</h2>' in head
    assert blocks["shep-dl-hd"].parent is blocks["shep-dl"]


def test_a_first_start_says_nothing_is_set_and_offers_the_suggested_folders(dl, monkeypatch):
    rows, panel = _parts(dl, dl._on_locations_open(operator_request(monkeypatch)))
    assert dl.NOTHING_SET in panel["notice"]["value"]
    assert panel["apply"]["visible"] is False and panel["repair"]["visible"] is False
    for key in ("workspaces_root", "registry", "intake_dir"):
        row = rows[key]
        assert "Not set" in row["state"]["value"], key
        assert row["create"]["visible"] is True and row["create"]["interactive"] is True
        assert row["create"]["value"] == dl.CREATE_LABEL
        assert f"~/SHEPHERD-data/{SUGGESTED[key]}" in re.sub(r"<[^>]+>", "", row["suggest"]["value"])
        assert row["box"]["interactive"] is True and row["save"]["interactive"] is True
        assert ">Set to<" in row["key"]["value"]
        assert row["lock"]["value"] == ""          # an operator: no lock, no reason
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
    state = re.sub(r"<(?!/?dt)[^>]+>", "", rows["registry"]["state"]["value"])
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
    naming one; every box and button off with the lock; D8's reason once,
    and each row's short phrase (`ROW_LOCKED`) beside its locked buttons."""
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
        assert row["lock"]["value"] == dl.reason_line("Operator controls only"), key
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
    # After a path its source starts a line of its own (the sweep's L7, D13: it ran on after the
    # path's last line at 1080 x 1920); after words it follows on their line.
    assert cells["workspaces_root"]["In effect"] == f"{root.resolve()}\n{saved}"
    assert cells["registry"]["In effect"] == f"{registry.resolve()}\n{started}"
    assert cells["registry"]["Saved"] == f"{chosen}\n{dl.OVERRIDDEN}"
    assert cells["intake_dir"]["In effect"].endswith(f"Not usable {tmp_path / 'gone'}\n{started}")
    assert cells["training_evidence_dirs"]["In effect"] == (
        f"{evidence[0].resolve()}\n{evidence[1].resolve()}\n{started}")
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
#: location's box, how wide it is and whether its placeholder fits in it,
#: and each of its row's buttons shown (Check, Save, Create): whether it lies
#: inside its row's content box.
_BOXES_JS = r"""
const { chromium } = require('playwright');
const QUEUE_JS = __QUEUE_JS__;
(async () => {
  const browser = await chromium.launch();
  const out = {};
  for (const [w, h] of [[1280, 720], [1440, 1600], [1850, 960]]) {
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
        const row = document.getElementById(`dl_row_${key}`);
        const rs = getComputedStyle(row), rr = row.getBoundingClientRect();
        const left = rr.left + parseFloat(rs.paddingLeft), right = rr.right - parseFloat(rs.paddingRight);
        boxes[key].buttons = [];
        for (const part of ['check', 'save', 'create']) {
          const b = document.getElementById(`dl_${part}_${key}`);
          const br = b ? b.getBoundingClientRect() : null;
          if (!br || br.width === 0 || b.offsetParent === null) { continue; }
          boxes[key].buttons.push({ part, left: br.left, right: br.right, row_left: left, row_right: right,
                                    inside: br.left >= left - 0.5 && br.right <= right + 0.5 });
        }
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
    fcc3dea. The review of 5188b4e (2026-11-19): at 390x844 Create, indented
    by the key column's 8em as if its box sat beside the key, ended at x 381
    past its row's 357; every button shown now lies inside its row (fails on
    5188b4e)."""
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
            assert box["buttons"] and all(b["inside"] for b in box["buttons"]), (size, key, box)


#: Settings for an operator: drafts typed in the evidence folders box and in
#: the registry box, then another row's Check and Save, a refused Save, a
#: return to Settings, and the evidence row's own Save: what each box holds
#: after each answer (each step waits for its own answer's words, then for
#: every request to be answered).
_DRAFTS_JS = r"""
const { chromium } = require('playwright');
const QUEUE_JS = __QUEUE_JS__;
(async () => {
  const [url, ws, draft] = process.argv.slice(2);
  const browser = await chromium.launch();
  const page = await (await browser.newContext({ viewport: { width: 1850, height: 960 } })).newPage();
  await page.addInitScript(QUEUE_JS);
  await page.goto(url, { waitUntil: 'load' });
  await page.waitForSelector('#shep-chips .shep-chip', { timeout: 60000 });
  const box = (key) => `#dl_box_${key} textarea, #dl_box_${key} input`;
  const idle = () => page.waitForFunction(() => window.__shepQueue.idle(), null, { timeout: 30000 });
  const said = (key, words) => page.waitForFunction(([key, words]) => {
    const line = document.querySelector(`#dl_result_${key}`);
    return !!line && line.innerText.includes(words); }, [key, words], { timeout: 30000 });
  const boxes = async () => { const out = {};
    for (const key of ['workspaces_root', 'registry', 'training_evidence_dirs']) {
      out[key] = await page.inputValue(box(key)); }
    return out; };
  const seen = {};
  await page.click('#shep-tab-settings-button');
  await page.waitForSelector('#dl_box_training_evidence_dirs textarea:enabled', { timeout: 30000 });
  await idle();
  seen.opened = await boxes();
  await page.fill(box('training_evidence_dirs'), draft);
  await page.fill(box('registry'), 'relative/registry-draft');
  await page.fill(box('workspaces_root'), ws);
  await page.click('#dl_check_workspaces_root');
  await said('workspaces_root', 'Usable');
  await idle();
  seen.otherCheck = await boxes();
  await page.click('#dl_save_workspaces_root');
  await said('workspaces_root', 'Saved');
  await idle();
  seen.otherSave = await boxes();
  await page.click('#dl_save_registry');
  await said('registry', 'Not saved');
  await idle();
  seen.refusedSave = await boxes();
  // Settings shown again (its open reads the panel again, every result line emptied).
  await page.click('#shep-tab-research-button');
  await idle();
  await page.click('#shep-tab-settings-button');
  await page.waitForFunction(() => document.querySelector('#dl_result_registry').innerText.trim() === '',
                             null, { timeout: 30000 });
  await idle();
  seen.shownAgain = await boxes();
  await page.fill(box('training_evidence_dirs'), `  ${draft}  \n\n`);
  await page.click('#dl_save_training_evidence_dirs');
  await said('training_evidence_dirs', 'Saved');
  await idle();
  seen.ownSave = await boxes();
  console.log(JSON.stringify(seen));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_a_draft_survives_every_other_rows_check_and_save(dl, tmp_path, monkeypatch):
    """OPS-C-R5 (the external review of 6a2d52e): every answer of the panel
    set the evidence folders box back to the saved list, so a draft typed
    there was lost when another row's Check answered. In a real browser, for
    an operator: a box's value is set when the panel first opens and when its
    own row's Save succeeds; another row's Check and Save, a refused Save and
    Settings shown again leave every draft as typed (the registry's too);
    the evidence row's own Save shows the list as saved. Fails on 6a2d52e
    (the evidence box reads the saved folder after the workspaces Check)."""
    from src.webui.app import create_gradio_app, mount_options
    from tests.fixtures.browser import QUEUE_JS, node_with_playwright, run_node
    from tests.fixtures.served_page import served

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    ws, old, draft = tmp_path / "ws", tmp_path / "old_evidence", tmp_path / "draft_evidence"
    for folder in (ws, old, draft):
        folder.mkdir()
    monkeypatch.setenv(ENV_SETTINGS_FILE, str(tmp_path / "settings.json"))
    _parts(dl, dl._on_save("training_evidence_dirs", str(old), operator_request(monkeypatch)))
    with served(create_gradio_app(), **mount_options()) as base:
        seen = run_node(_DRAFTS_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS)), tmp_path,
                        base[:-len("gradio_api")], str(ws), str(draft))
    assert seen["opened"] == {"workspaces_root": "", "registry": "",
                              "training_evidence_dirs": str(old)}, seen
    drafts = {"registry": "relative/registry-draft", "training_evidence_dirs": str(draft)}
    assert seen["otherCheck"] == {"workspaces_root": str(ws), **drafts}, seen
    for step in ("otherSave", "refusedSave", "shownAgain"):
        # The workspaces box was emptied by its own Save, and only by it.
        assert seen[step] == {"workspaces_root": "", **drafts}, (step, seen)
    assert seen["ownSave"] == {"workspaces_root": "", **drafts}, seen
    assert _saved_file()["applied"]["training_evidence_dirs"] == [str(draft)]


#: Settings at the owner's window and a phone's, the panel's page open
#: answered: for each row's visible locked button (Check, Save, Create
#: suggested folder), brought into view as a reader scrolls to it, the line in
#: the panel that says why it is locked (the lock mark, then words, in no
#: button) nearest it: the one in its own row when there is one, else any.
#: How far its drawn words are from the button, down the page (``dy``, the
#: gap between their boxes) and across (``dx``), and whether both are wholly
#: on screen; and how often the panel's long reason is said on the page.
_ROW_LOCKS_JS = r"""
const { chromium } = require('playwright');
const QUEUE_JS = __QUEUE_JS__;
(async () => {
  const browser = await chromium.launch();
  const out = {};
  for (const [w, h, wider] of [[1850, 960, false], [1280, 720, false], [1280, 720, true]]) {
    const page = await (await browser.newContext({ viewport: { width: w, height: h } })).newPage();
    await page.addInitScript(QUEUE_JS);
    await page.goto(process.argv[2], { waitUntil: 'load' });
    await page.waitForSelector('#shep-chips .shep-chip', { timeout: 60000 });
    await page.click('#shep-tab-settings-button');
    // The page's open answered: the rows' buttons drawn for this request.
    await page.waitForFunction(() => {
      const b = document.getElementById('dl_check_workspaces_root');
      return !!b && b.offsetParent !== null; }, null, { timeout: 30000 });
    await page.waitForFunction(() => window.__shepQueue.idle(), null, { timeout: 30000 });
    if (wider) {
      // Another host's wider text (the recheck of d4209c4 read Create 30 px
      // from its row's phrase at 390 x 844, below the display floor; walked at the floor since):
      // the panel's words 0.04em apart.
      await page.addStyleTag({ content: '#shep-dl { letter-spacing: 0.04em !important; }' });
      await page.waitForFunction(() => window.__shepQueue.idle(), null, { timeout: 30000 });
    }
    out[w + 'x' + h + (wider ? '-wider' : '')] = await page.evaluate(async () => {
      const frame = () => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
      const T = (e) => (e.innerText || '').replace(/\s+/g, ' ').trim();
      const box = (e) => e.getBoundingClientRect();
      const clip = (e) => { let x = { left: 0, top: 0, right: innerWidth, bottom: innerHeight };
        for (let a = e.parentElement; a; a = a.parentElement) { const s = getComputedStyle(a);
          if (/(auto|scroll|hidden|clip)/.test(s.overflowY + s.overflowX)) { const p = box(a);
            x = { left: Math.max(x.left, p.left), top: Math.max(x.top, p.top),
                  right: Math.min(x.right, p.right), bottom: Math.min(x.bottom, p.bottom) }; } }
        return x; };
      const whole = (e) => { const q = box(e), x = clip(e);
        return q.left >= x.left - 1 && q.top >= x.top - 1 && q.right <= x.right + 1 && q.bottom <= x.bottom + 1; };
      // A line's drawn words, not its block's width.
      const words = (e) => { const r = document.createRange(); r.selectNodeContents(e);
        return r.getBoundingClientRect(); };
      // Where a button's phrase lies from it: in its column (their spans across overlap),
      // and on the line next to it (no more than the button's own height between them).
      const gaps = (a, b) => { const p = box(a), q = words(b);
        const dy = Math.max(0, p.top - q.bottom, q.top - p.bottom);
        const bar = a.closest('.shep-bar'), s = getComputedStyle(bar);
        const start = box(bar).left + bar.clientLeft + parseFloat(s.paddingLeft);
        return { column: q.left < p.right && q.right > p.left, nextLine: dy <= p.height,
                 line: q.top < p.bottom && q.bottom > p.top, after: q.left >= p.right - 1,
                 bar: bar.contains(b), edge: Math.abs(q.left - start) <= 1 || Math.abs(box(b).left - start) <= 1,
                 dy: Math.round(dy) }; };
      // Whether a row's box and its Check and Save share a line, and whether the bar holds the
      // box at its least width with both buttons beside it.
      const besideBox = (key) => { const bx = document.getElementById(`dl_box_${key}`);
        const c = document.getElementById(`dl_check_${key}`), v = document.getElementById(`dl_save_${key}`);
        const bar = c.closest('.shep-bar'), s = getComputedStyle(bar), gap = parseFloat(s.columnGap) || 0;
        const room = bar.clientWidth - parseFloat(s.paddingLeft) - parseFloat(s.paddingRight);
        const least = parseFloat(getComputedStyle(bx).minWidth) || 0;
        const acts = c.closest('.shep-grp'), ag = parseFloat(getComputedStyle(acts).columnGap) || 0;
        const need = least + gap + box(c).width + ag + box(v).width;
        const p = box(bx), q = box(c);
        return { fits: need <= room + 0.5, line: q.top < p.bottom && q.bottom > p.top }; };
      const rowOf = (e) => e.closest('.shep-dl-row');
      const panel = document.getElementById('shep-dl');
      // The lines that say why: the lock mark, then words, in no button.
      const reasons = [...panel.querySelectorAll('*')].filter((e) =>
        T(e) === '🔒' && !e.closest('button') && e.offsetParent !== null).map((e) => e.parentElement);
      const rows = {};
      for (const key of ['workspaces_root', 'registry', 'intake_dir', 'training_evidence_dirs']) {
        rows[key] = {};
        for (const part of ['check', 'save', 'create']) {
          const b = document.getElementById(`dl_${part}_${key}`);
          if (!b || b.offsetParent === null) { continue; }
          // Brought to the middle of the view (the panel scrolls since its rows read top to bottom).
          b.scrollIntoView({ block: 'center' });
          await frame();
          const own = reasons.filter((e) => rowOf(e) === rowOf(b));
          const near = (own.length ? own : reasons).map((e) => [gaps(b, e), e])
            .sort((x, y) => x[0].dy - y[0].dy)[0];
          rows[key][part] = { text: T(b), disabled: b.disabled, ...(near ? near[0] : {}),
                              sameRow: own.length > 0, reason: near ? T(near[1]) : null,
                              whole: whole(b), reasonWhole: near ? whole(near[1]) : false,
                              ...(part === 'check' ? { box: besideBox(key) } : {}) };
        }
      }
      const said = (words) => (document.body.innerText.split(words).length - 1);
      return { rows, locks: (T(panel).match(/🔒/g) || []).length, reasons: reasons.map(T),
               longSaid: said('Folder paths are shown to operator controls only.'),
               sideways: document.documentElement.scrollWidth > document.documentElement.clientWidth };
    });
    await page.close();
  }
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


@pytest.mark.parametrize("operator", [False, True], ids=["read-only", "operator"])
def test_each_rows_locked_buttons_say_why_beside_them(dl, tmp_path, monkeypatch, operator):
    """Owner's decision 3 of 2026-11-19 (the recheck of d7ebfc3, section 7.3):
    for a read-only viewer the panel said why once, above its rows, so a
    row's locked Check, Save and Create were 37-375 px below that line at
    1850 x 960 and up to 1,312 px at 390 x 844 (the reviewer's figures; with
    Create offered here, 38-556 and 190-1,371). In a real browser, nothing
    set (Create offered), a request D8 refuses: each row's visible locked
    button has the short phrase (`ROW_LOCKED`, the lock and "Operator
    controls only") in its own row, on the line next to it (no more than
    the button's own height above or below it), both wholly on screen when the
    button is brought to the middle of the view (the UI sweep of 2026-11-22
    laid each row out top to bottom, so the panel scrolls at 1850 x 960 and
    a button brought only as far as the view's edge left its phrase under
    it out of view); the panel's long reason is said once on the
    page; beside Check and Save, a group of their bar on their line after
    Save where the line holds it, else the line under them on the bar's edge
    (the sweep's L7 check, D6: as a line of its own under them it made
    each read-only row a line taller than an operator's, and the panel's
    last line fell under its body's edge at 1850 x 960; the UI sweep's
    shared reason grid had once put it under the label column, 500 px from
    them); Check and Save on the box's line wherever the bar holds the box
    at its least width and both beside it (K4: bound into their group, the
    phrase took them off it from 1550 x 830 down). An operator sees
    no lock and no phrase. Fails on d7ebfc3. The
    recheck of d4209c4 read Create 30 px from its row's phrase at 390 x 844
    on its host: Create was centred against a suggestion that wrapped to
    more lines. A pass with the panel's words 0.04em apart (another host's
    metrics) reads the same 30 px on d4209c4; Create now shares its
    suggestion's first baseline."""
    from src.webui.app import create_gradio_app, mount_options
    from tests.fixtures.browser import QUEUE_JS, node_with_playwright, run_node
    from tests.fixtures.operator_access import DEPLOYMENT_MODE, operator_mode
    from tests.fixtures.served_page import served

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    if operator:
        operator_mode(monkeypatch)
    else:
        monkeypatch.delenv(DEPLOYMENT_MODE, raising=False)
    monkeypatch.setenv(ENV_SETTINGS_FILE, str(tmp_path / "settings.json"))
    with served(create_gradio_app(), **mount_options()) as base:
        seen = run_node(_ROW_LOCKS_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS)), tmp_path,
                        base[:-len("gradio_api")])
    phrase = "Operator controls only"
    for size, at in seen.items():
        assert at["sideways"] is False, (size, at)
        if operator:
            assert at["locks"] == 0 and at["reasons"] == [] and at["longSaid"] == 0, (size, at)
            continue
        assert at["longSaid"] == 1, (size, at)
        assert at["rows"]["workspaces_root"].get("create"), (size, at)       # Create offered
        for key, buttons in at["rows"].items():
            assert {"check", "save"} <= set(buttons), (size, key, buttons)
            for part, button in buttons.items():
                assert button["text"].startswith("🔒 ") and button["disabled"], (size, key, part)
                assert button["reason"] == f"🔒 {phrase}", (size, key, part, button)
                assert button["sameRow"] and button["nextLine"], (size, key, part, button)
                if part in ("check", "save"):
                    # Beside Check and Save (the owner's word, 2026-11-19): a group of their
                    # bar, on their line after them where it holds it (the sweep's L7 check, D6:
                    # a line of its own under them made each read-only row a line taller, and
                    # "Where these are kept ›" fell under the body's edge at 1850 x 960), else
                    # the line under them, on the bar's start edge.
                    assert button["bar"], (size, key, part, button)
                    assert (button["line"] and (part == "check" or button["after"])) or (
                        button["edge"] and button["nextLine"]), (size, key, part, button)
                if part == "check":
                    # Check and Save stand on the box's line wherever the bar holds the box at its
                    # least width and both beside it, as an operator's do: the phrase never takes
                    # them off it (K4: bound into their group, it did from 1550 x 830 down).
                    beside = button["box"]
                    assert beside["line"] or not beside["fits"], (size, key, part, button)
                assert button["whole"] and button["reasonWhole"], (size, key, part, button)


#: Settings with nothing set (Create offered on each required row), the
#: panel's page open answered, at the owner's window, a laptop's, a phone's
#: and the narrowest, the panel's words as drawn and 0.04em apart (another
#: host's wider metrics): for each row's Create brought into view, its
#: suggested folder as drawn. Its words (each character's line), the width the
#: suggestion is given, the width of each piece of its path up to and with a
#: separator, drawn on one line (``~/``, ``SHEPHERD-data/``, ``workspaces``),
#: the room beside Create, and where the suggestion lies (beside Create or
#: under it); Create's gap to its row's phrase (read-only); whether Create,
#: the suggestion and its drawn lines lie inside the row's content box.
_SUGGEST_JS = r"""
const { chromium } = require('playwright');
const QUEUE_JS = __QUEUE_JS__;
(async () => {
  const browser = await chromium.launch();
  const out = {};
  for (const [w, h] of [[1850, 960], [1366, 768], [1280, 720]]) for (const spacing of [0, 0.04]) {
    const page = await (await browser.newContext({ viewport: { width: w, height: h } })).newPage();
    await page.addInitScript(QUEUE_JS);
    await page.goto(process.argv[2], { waitUntil: 'load' });
    await page.waitForSelector('#shep-chips .shep-chip', { timeout: 60000 });
    await page.click('#shep-tab-settings-button');
    await page.waitForFunction(() => {
      const b = document.getElementById('dl_create_workspaces_root');
      return !!b && b.offsetParent !== null; }, null, { timeout: 30000 });
    await page.waitForFunction(() => window.__shepQueue.idle(), null, { timeout: 30000 });
    if (spacing) {
      await page.addStyleTag({ content: `#shep-dl { letter-spacing: ${spacing}em !important; }` });
      await page.waitForFunction(() => window.__shepQueue.idle(), null, { timeout: 30000 });
    }
    out[`${w}x${h}@${spacing}em`] = await page.evaluate(async () => {
      const frame = () => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
      const box = (e) => e.getBoundingClientRect();
      const words = (e) => { const r = document.createRange(); r.selectNodeContents(e);
        return r.getBoundingClientRect(); };
      // A text as one line in the suggestion's own type (laid out apart, out of flow).
      const oneLine = (code, text) => { const m = document.createElement('span');
        m.style.cssText = 'position: absolute; visibility: hidden; white-space: nowrap; padding: 0';
        m.textContent = text; code.appendChild(m); const width = box(m).width; m.remove(); return width; };
      const rows = {};
      for (const key of ['workspaces_root', 'registry', 'intake_dir']) {
        const b = document.getElementById(`dl_create_${key}`);
        if (!b || b.offsetParent === null) { continue; }
        b.scrollIntoView({ block: 'nearest' });
        await frame();
        const row = document.getElementById(`dl_row_${key}`), crow = b.closest('.shep-dl-create');
        const rs = getComputedStyle(row), rr = box(row);
        const left = rr.left + parseFloat(rs.paddingLeft), right = rr.right - parseFloat(rs.paddingRight);
        const inside = (q) => q.left >= left - 0.5 && q.right <= right + 0.5;
        const sug = document.getElementById(`dl_suggest_${key}`), code = sug.querySelector('code');
        const text = code.textContent;
        // Each character's line, and the lines the words are drawn on.
        const chars = [], drawn = [];
        const walk = document.createTreeWalker(code, NodeFilter.SHOW_TEXT);
        for (let n = walk.nextNode(); n; n = walk.nextNode()) {
          for (let i = 0; i < n.length; i++) { const g = document.createRange();
            g.setStart(n, i); g.setEnd(n, i + 1);
            const q = [...g.getClientRects()].find((x) => x.width > 0);
            chars.push({ c: n.data[i], top: q ? Math.round(q.top) : null });
            if (q) { drawn.push(q); } } }
        const inWord = /[\p{L}\p{N}]/u;
        const breaks = [];
        for (let i = 1; i < chars.length; i++) {
          if (inWord.test(chars[i - 1].c) && inWord.test(chars[i].c) && chars[i - 1].top !== chars[i].top) {
            breaks.push(text.slice(0, i) + '|' + text.slice(i)); } }
        const pieces = (text.match(/[^/]*\/|[^/]+$/g) || []).map((p) => ({ p, width: oneLine(code, p) }));
        const bb = box(b), sb = box(sug), cs = getComputedStyle(crow);
        const others = [...crow.children].filter((e) => e !== b && e.offsetParent !== null && box(e).height > 0);
        const phrase = document.querySelector(`#dl_lock_${key} .shep-reason`);
        const pq = phrase && phrase.offsetParent !== null ? words(phrase) : null;
        rows[key] = {
          text, breaks, pieces, lines: new Set(chars.map((x) => x.top).filter((t) => t !== null)).size,
          longest: Math.max(...pieces.map((x) => x.width)),
          width: sug.clientWidth,
          room: right - bb.right - parseFloat(cs.columnGap),
          under: sb.top >= bb.bottom, beside: sb.left >= bb.right && sb.top < bb.bottom,
          firstLine: others.every((e) => box(e).bottom > bb.top),
          phraseDy: pq ? Math.round(Math.max(0, bb.top - pq.bottom, pq.top - bb.bottom)) : null,
          cut: sug.scrollWidth > sug.clientWidth || drawn.length !== text.length,
          inside: { create: inside(bb), suggestion: inside(sb), lines: drawn.every(inside),
                    phrase: pq ? inside(pq) : null } };
      }
      return { rows, sideways: document.documentElement.scrollWidth > document.documentElement.clientWidth };
    });
    await page.close();
  }
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


@pytest.mark.parametrize("operator", [False, True], ids=["read-only", "operator"])
def test_the_suggested_folder_is_whole_beside_or_under_create(dl, tmp_path, monkeypatch, operator):
    """OPS-RECHECK-DL1 (the lanes' review of c2a64ac): the suggested folder
    beside Create got only what the row's 8em indent and the 220 px button
    left (83 px at 1366 x 768, 96 at 390 x 844, 26 at 320 x 640), so
    "~/SHEPHERD-data/workspaces" broke inside its words over 4 lines (9 to
    14 at 320 x 640): the create row never let it go under Create. In a real
    browser, nothing set, at 1850 x 960, 1366 x 768 and the display floor's 1280 x 720 (390 x 844
    and 320 x 640, where it was found, are below the floor, drawn as at it),
    the panel's words as drawn and 0.04em apart, for a read-only viewer and
    an operator: each Create offered has its suggestion whole (every
    character drawn, nothing cut) with no line break inside a word, given at
    least the width of its path's longest piece (up to and with a separator);
    under Create wherever the room beside Create is narrower than that;
    Create on its row's first line, its words at most 24 px from its row's
    phrase (read-only; `test_each_rows_locked_buttons_say_why_beside_them`'s
    limit); Create, the suggestion and its lines inside the row; no sideways
    page scroll. Fails on 490587c (at 1366 x 768 the suggestion is 83 px
    wide, broken as "~/SHEPHER|D-data/work|spaces")."""
    from src.webui.app import create_gradio_app, mount_options
    from tests.fixtures.browser import QUEUE_JS, node_with_playwright, run_node
    from tests.fixtures.operator_access import DEPLOYMENT_MODE, operator_mode
    from tests.fixtures.served_page import served

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    if operator:
        operator_mode(monkeypatch)
    else:
        monkeypatch.delenv(DEPLOYMENT_MODE, raising=False)
    monkeypatch.setenv(ENV_SETTINGS_FILE, str(tmp_path / "settings.json"))
    with served(create_gradio_app(), **mount_options()) as base:
        seen = run_node(_SUGGEST_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS)), tmp_path,
                        base[:-len("gradio_api")])
    assert len(seen) == 6, seen
    problems = []
    for size, at in seen.items():
        if at["sideways"]:
            problems.append(f"{size}: the page scrolls sideways")
        if set(at["rows"]) != {"workspaces_root", "registry", "intake_dir"}:
            problems.append(f"{size}: Create offered on {sorted(at['rows'])}")
        for key, row in at["rows"].items():
            where = f"{size} {key}"
            if row["text"] != f"~/SHEPHERD-data/{SUGGESTED[key]}" or row["cut"]:
                problems.append(f"{where}: suggestion not whole: {row['text']!r}")
            if row["breaks"]:
                problems.append(f"{where}: broken inside a word: {row['breaks']}")
            if row["width"] < row["longest"]:
                problems.append(f"{where}: {row['width']} px for a piece of {row['longest']:.1f} px "
                                f"({row['lines']} lines)")
            if row["room"] < row["longest"] and not row["under"]:
                problems.append(f"{where}: beside Create in {row['room']:.1f} px, not under it")
            if not (row["under"] or row["beside"]):
                problems.append(f"{where}: neither beside nor under Create")
            if not row["firstLine"]:
                problems.append(f"{where}: Create is not on its row's first line")
            if operator and row["phraseDy"] is not None:
                problems.append(f"{where}: an operator sees a lock phrase")
            if not operator and (row["phraseDy"] is None or row["phraseDy"] > 24):
                problems.append(f"{where}: Create {row['phraseDy']} px from its row's phrase")
            outside = [part for part, ok in row["inside"].items() if ok is False]
            if outside:
                problems.append(f"{where}: past the row's edge: {outside}")
    assert not problems, "\n".join(problems)


def test_in_a_browser_no_row_of_settings_puts_words_beside_its_fields(dl, tmp_path, monkeypatch):
    """The first-use walk's Settings findings (the UI sweep of 2026-11-22),
    as the dashboard draws them for an operator with nothing set, at the
    walk's windows (the owner's, his Firefox width, the laptops, a portrait
    column and a phone): no text block of 100 characters or more beside a
    field or button (the registry's words beside its box at 1850 x 960), no
    control centred in a row much taller than itself (Check in the 115 px
    set row at 390 x 844), no group split over lines, no control cut,
    overlapped or squeezed; in every row the name's words end above the
    row's box, Check and Save share one line, and a line with nothing to
    say (no result, no lock) is not drawn (its rule looked for an empty
    container that always holds the block's own element, so each empty
    line was drawn 0 px tall and still took a gap). Fails on 8e4c7b6."""
    from src.webui.app import create_gradio_app, mount_options
    from tests.fixtures.browser import layout_problems, node_with_playwright, walk_ui
    from tests.fixtures.operator_access import operator_mode
    from tests.fixtures.served_page import served

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    operator_mode(monkeypatch)
    monkeypatch.setenv(ENV_SETTINGS_FILE, str(tmp_path / "settings.json"))
    rows_js = """() => [...document.querySelectorAll('.shep-dl-row')].map((row) => {
      const box = (s) => { const e = row.querySelector(s); return e && e.getClientRects().length
        ? e.getBoundingClientRect() : null; };
      const name = box('.shep-dl-name p'), field = box('input, textarea');
      const check = box('[id^="dl_check_"]'), save = box('[id^="dl_save_"]');
      // A line with nothing to say takes no room (its block drawn 0 px tall still took a gap).
      const empties = [...row.querySelectorAll('.shep-dl-main > *')].filter((e) =>
        e.getClientRects().length && e.getBoundingClientRect().height < 0.5).map((e) => e.id || e.className);
      return { id: row.id, above: !!name && !!field && name.bottom <= field.top + 0.5,
               oneLine: !!check && !!save && Math.abs(check.top - save.top) < 1, empties };
    })"""
    sizes = ([1850, 960], [1550, 830], [1366, 768], [1280, 720], [1440, 1600])
    stops = [{"size": size, "page": "settings", "focus": False} for size in sizes]
    with served(create_gradio_app(), **mount_options()) as base:
        seen = walk_ui(base[:-len("gradio_api")], tmp_path, stops, extra_js=rows_js)
    rules = ("beside", "narrow-box", "overlap", "orphan", "centred", "split-group", "clipped")
    assert layout_problems(seen, rules=rules) == []
    for stop in seen:
        assert len(stop["extra"]) == len(KEYS), stop
        for row in stop["extra"]:
            assert row["above"] and row["oneLine"] and row["empties"] == [], (stop["size"], row)


#: Each Data locations row as drawn: its Set-to box, Check, Save and Create (when offered), the
#: path in effect (its lines and left edge), the name column and the facts; and the panel's body
#: and "Where these are kept ›".
_COLUMNS_JS = """() => {
  const rect = (e) => e && e.getClientRects().length ? e.getBoundingClientRect() : null;
  const lines = (e) => { const r = document.createRange(); r.selectNodeContents(e);
    return new Set([...r.getClientRects()].filter((q) => q.width > 1).map((q) => Math.round(q.bottom))).size; };
  const at = (q) => q && { left: q.left, right: q.right, top: q.top, bottom: q.bottom };
  const rows = [...document.querySelectorAll('.shep-dl-row')].map((row) => {
    const path = row.querySelector('.shep-dl-facts .shep-dl-path');
    return { id: row.id, box: at(rect(row.querySelector('[id^="dl_box_"] :is(input, textarea)'))),
             check: at(rect(row.querySelector('[id^="dl_check_"]'))),
             save: at(rect(row.querySelector('[id^="dl_save_"]'))),
             create: at(rect(row.querySelector('[id^="dl_create_"]'))),
             name: at(rect(row.querySelector('.shep-dl-left'))),
             facts: at(rect(row.querySelector('.shep-dl-facts'))),
             path: path && { lines: lines(path), left: rect(path).left } };
  });
  return { rows, body: at(rect(document.querySelector('#shep-dl-bd'))),
           where: at(rect(document.querySelector('#dl_where summary'))) };
}"""


def _columns_problems(stop) -> list:
    """**The rows share their columns** (the sweep's L7, C3): every row's box starts and ends
    where the other rows' do, and its Check and Save start where theirs do; each Create starts
    at its box's left edge (a line of its own under it) or where every other Create beside Save
    starts. Facts that hold a path in effect are never narrower than the name beside them
    (LOOK2-2: 288 px beside a name of 504 at 1550 x 830, the path 8 lines a row): beside it
    or under it, where the path starts on the box's edge."""
    problems, where = [], stop["size"]
    rows = stop["extra"]["rows"]
    for part in ("box", "check", "save"):
        for edge in (("left", "right") if part == "box" else ("left",)):
            xs = {row["id"]: round(row[part][edge]) for row in rows}
            if max(xs.values()) - min(xs.values()) > 1:
                problems.append(f"{where}: the rows' {part} {edge} edges differ: {xs}")
    beside = {row["id"]: round(row["create"]["left"]) for row in rows
              if row["create"] and row["create"]["top"] < row["box"]["bottom"]}
    if beside and max(beside.values()) - min(beside.values()) > 1:
        problems.append(f"{where}: Create beside Save starts at {beside}")
    for row in rows:
        if row["create"] and row["id"] not in beside and abs(row["create"]["left"] - row["box"]["left"]) > 1:
            problems.append(f"{where} {row['id']}: Create under its box starts off its edge")
        path, facts, name = row["path"], row["facts"], row["name"]
        side = bool(path) and facts["left"] >= name["right"] and facts["top"] < name["bottom"]
        if side and facts["right"] - facts["left"] < name["right"] - name["left"] - 0.5:
            problems.append(f"{where} {row['id']}: the path's facts are narrower than the name beside "
                            f"them ({path['lines']} lines)")
        if path and not side and abs(path["left"] - row["box"]["left"]) > 1:
            problems.append(f"{where} {row['id']}: the path under the name starts off the box's edge")
    return problems


@pytest.mark.parametrize("operator", [True, False], ids=["operator", "read-only"])
def test_in_a_browser_the_rows_share_their_columns_and_a_path_reads(dl, tmp_path, monkeypatch,
                                                                    operator):
    """The sweep's L7 check (C3, a regression of a4dc77d): each row's Set-to box took what
    Create and its suggestion left on its line, so at 1850 x 960 the four boxes ended at 425,
    441, 457 and 845 px (nothing set) and the Import row's at 458 against 846 (configured), and
    Check and Save stood at a different x in every row (d3040ca: one column). LOOK2-2: with a
    path in effect, the facts beside the name got what the name's 56ch left: 159 px at 1550 x
    830, the path 8 lines a row and two rows in view. For an operator, the results registry
    set at start (a long path) and the other folders not set (Create offered), at the owner's
    windows, a laptop and the display floor: the rows share their columns
    (`_columns_problems`), and at the owner's window all four rows and "Where these are kept
    ›" stay in the panel's body (SW-LK1). The same for a read-only viewer, whose paths are
    hidden and whose rows say why they are locked (the sweep's L7 check, D6: each row's lock
    line under its buttons made the body 28 px taller than its room at 1850 x 960)."""
    from src.webui.app import create_gradio_app, mount_options
    from tests.fixtures.browser import node_with_playwright, walk_ui
    from tests.fixtures.operator_access import DEPLOYMENT_MODE, operator_mode
    from tests.fixtures.served_page import served

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    if operator:
        operator_mode(monkeypatch)
    else:
        monkeypatch.delenv(DEPLOYMENT_MODE, raising=False)
    monkeypatch.setenv(ENV_SETTINGS_FILE, str(tmp_path / "settings.json"))
    registry = tmp_path / "a-results-registry-folder-kept-on-the-server" / "for-the-cohort-versions" / "registry"
    registry.mkdir(parents=True, mode=0o700)
    monkeypatch.setenv(ENV["registry"], str(registry))
    sizes = ([1850, 960], [1550, 830], [1366, 768], [1280, 720])
    stops = [{"size": size, "page": "settings", "focus": False} for size in sizes]
    with served(create_gradio_app(), **mount_options()) as base:
        seen = walk_ui(base[:-len("gradio_api")], tmp_path, stops, extra_js=_COLUMNS_JS)
    problems = []
    for stop in seen:
        rows = stop["extra"]["rows"]
        assert len(rows) == len(KEYS) and all(row["box"] and row["check"] for row in rows), stop
        assert [row["id"] for row in rows if row["path"]] == (["dl_row_registry"] if operator
                                                              else []), stop
        assert [row["id"] for row in rows if row["create"]] == ["dl_row_workspaces_root", "dl_row_intake_dir"], stop
        problems += _columns_problems(stop)
        if stop["size"] == "1850x960":
            body, last = stop["extra"]["body"], stop["extra"]["where"]
            if last["bottom"] > body["bottom"] + 0.5:
                problems.append(f"1850x960: Where these are kept › ends at {last['bottom']}, the body at "
                                f"{body['bottom']}")
    assert not problems, "\n".join(problems)


def test_the_settings_help_texts_meet_the_text_budget():
    """The UI sweep of 2026-11-22 (the text brief's §11): Settings' two help
    texts (Runtime settings' allocator and torch.compile) ran over the text
    layout's budget (items of 136 and 147 characters, a semicolon); every
    fact is kept as a term row of two bubble lines at most."""
    from src.webui.components import help as h
    from src.webui.components import runtime_settings as rs

    over = {key: h.budget_problems(text, h.TEXT_BUDGET) for key, (_title, text) in rs.FIELD_HELP.items()}
    assert {key: found for key, found in over.items() if found} == {}
    allocator, compile_ = rs.FIELD_HELP["allocator_preset"][1], rs.FIELD_HELP["torch_compile"][1]
    for fact in ("Restart Backend", "launch environment", "CPU run", "diagnosis"):
        assert fact.lower() in allocator.lower(), fact
    for fact in ("graph-break", "eager", "GATv2, the one architecture built today", "Restart"):
        assert fact in compile_, fact
