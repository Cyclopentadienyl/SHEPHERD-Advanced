"""
Research › Prepare › Fine-tune (K6 decisions 2, 9, 11).
======================================================
`src/webui/components/finetune_page.py`. What these pin:

- **the parents are listed bases only**: never a task model, a file without a
  model spec or a refused record; listing is an operator control;
- **the options follow the skill** (unique, at most 40 characters, one
  separator), the tasks' carrying their readiness, why not under them;
- **Start** sends the base's listed digest, the dataset fingerprint shown, the
  tasks in `TASKS` order and only the settings that differ from the recipe's,
  through the server's lists (never a client path); a refusal is the runner's
  sentence, inline and escaped;
- **the job card**: each child's status as colour (the ``--shep-status-*``
  tokens), shape and word; everything escaped; products for an operator only,
  once the child ended, the best save chosen by the Trainer validation figure
  the list records (never a file name), ties shared; its own words once the
  Fine-tune status is cleared;
- **Start and Stop** are decided as the Training run bar's are (D8, a job
  running or stopping, the GPU slot);
- **the page contract**: its own Dataset select (starting from Training's,
  never writing it), the slow timer without the queue doing nothing on
  another page, ``ft_*`` ids, help titles equal to labels.

Module: tests/unit/test_webui_finetune_page.py
"""
from __future__ import annotations

import re
import shutil

import pytest

gr = pytest.importorskip("gradio")

from src.webui import shell  # noqa: E402
from src.webui.components import finetune_page as fp  # noqa: E402
from src.webui.components import training_console as tc  # noqa: E402
from tests.fixtures.operator_access import gradio_request, operator_mode  # noqa: E402

TAG = "<img src=x onerror=alert(1)>"
BASE_RUN = "20261001-000000_base_aaaaaa"
OLD_BASE_RUN = "20260901-000000_old_bbbbbb"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    from tests.fixtures.registered_cohort import write_demo_workspace

    return write_demo_workspace(tmp_path_factory.mktemp("finetune") / "demo")


@pytest.fixture
def root(built, tmp_path, monkeypatch):
    """A workspaces root listing ``demo``, whose runs hold two bases, a task
    model, a file without a model spec and a refused file."""
    from tests.unit.test_task_first_selection import GNN_PAYLOAD, _put, _record_payload

    root = tmp_path / "root"
    shutil.copytree(built, root / "demo")
    ws = root / "demo"
    _put(ws, "gatv2", BASE_RUN, _record_payload(logs={"val_lp_accuracy": 0.81}), 300)
    _put(ws, "gatv2", BASE_RUN, _record_payload(logs={"val_lp_accuracy": 0.83}), 290,
         name="model-03-0.8300.pt")
    _put(ws, "gatv2", OLD_BASE_RUN, _record_payload(), 100)
    _put(ws, "gatv2", "20261002-000000_disease_cccccc", _record_payload("disease"), 400)
    _put(ws, "gat", "20261003-000000_hist_dddddd", GNN_PAYLOAD, 500)
    torn = ws / "runs" / "gatv2" / "20261004-000000_torn_eeeeee" / "checkpoints" / "last.pt"
    torn.parent.mkdir(parents=True)
    torn.write_text("not a checkpoint")
    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(root))
    return root


@pytest.fixture
def operator(monkeypatch):
    operator_mode(monkeypatch)
    return gradio_request()


READ_ONLY = gradio_request()  # without mode M1, D8 refuses every request


def _page():
    with gr.Blocks() as demo:
        ctx = shell.standalone_context(demo)
        fp.create_finetune_page(ctx)
    return demo, ctx, {getattr(b, "elem_id", None): b for b in demo.blocks.values()}


def _option_rules(labels):
    assert len(set(labels)) == len(labels), labels
    for label in labels:
        assert len(label) <= tc.OPTION_CHARS and label.count(" · ") <= 1, label


# --------------------------------------------------------------------------- the parents
def test_only_listed_bases_are_offered_as_parents(root, operator):
    parents, runs, checkpoints, note = fp._on_finetune_bases(operator, {"name": "demo"}, None)
    assert set(parents) == {f"gatv2/{BASE_RUN}/last.pt", f"gatv2/{BASE_RUN}/model-03-0.8300.pt",
                            f"gatv2/{OLD_BASE_RUN}/last.pt"}
    assert all(item["learning_stage"] == "base" for item in parents.values())
    assert runs["value"] == f"gatv2/{BASE_RUN}" and checkpoints["value"] == (
        f"gatv2/{BASE_RUN}/last.pt")
    assert "3 base checkpoints in 2 runs" in note
    _option_rules([label for label, _key in runs["choices"]] and
                  [label for label, _key in checkpoints["choices"]])
    # A choice kept by key stays chosen.
    kept = fp._on_finetune_bases(operator, {"name": "demo"}, f"gatv2/{OLD_BASE_RUN}/last.pt")
    assert kept[1]["value"] == f"gatv2/{OLD_BASE_RUN}" and kept[2]["value"] == (
        f"gatv2/{OLD_BASE_RUN}/last.pt")


def test_listing_the_bases_is_an_operator_control(root, monkeypatch):
    from src.api.services import server_lists

    monkeypatch.setattr(server_lists, "list_checkpoints",
                        lambda *a, **k: pytest.fail("a refused request listed checkpoints"))
    parents, runs, checkpoints, note = fp._on_finetune_bases(READ_ONLY, {"name": "demo"}, None)
    assert parents == {} and runs == checkpoints == gr.update(choices=[], value=None)
    assert "Refused" in note


def test_an_empty_list_says_why(root, operator):
    assert "Select a dataset first." in fp._on_finetune_bases(operator, {}, None)[3]
    assert "train one on Research › Training" in fp.bases_note_html({}, None)
    assert "&lt;img" in fp.bases_note_html({}, TAG) and "<img" not in fp.bases_note_html({}, TAG)


def test_the_chosen_bases_summary_says_what_start_sends_escaped(root, operator):
    parents = fp._on_finetune_bases(operator, {"name": "demo"}, None)[0]
    item = parents[f"gatv2/{BASE_RUN}/model-03-0.8300.pt"]
    html = fp._on_base_chosen(f"gatv2/{BASE_RUN}/model-03-0.8300.pt", parents)
    for words in ("Run", "GATv2 · 2026-10-01 00:00", "model-03-0.8300.pt", "Configuration",
                  "GATv2 · recipe changed in", "Holds", "Validation", "LP accuracy (Trainer)",
                  "0.8300", "Trained on", "not recorded", "Model fingerprint",
                  item["sha256"][:12], f'data-copy="{item["sha256"]}"'):
        assert words in html, words
    assert fp._on_base_chosen(None, parents) == ""
    odd = fp.base_summary_html({**item, "filename": TAG, "trained_on_this_dataset": False})
    assert "<img" not in odd and "&lt;img" in odd and "another dataset" in odd
    same = fp.base_summary_html({**item, "trained_on_this_dataset": True})
    assert "the dataset selected" in same


# --------------------------------------------------------------------------- the tasks
@pytest.mark.parametrize("readiness", [
    {},
    {"disease": None, "gene": "no functional fixture", "patients_like_me": "no functional fixture"},
    {"disease": "not a listed workspace", "gene": "fixture from another build",
     "patients_like_me": "fixture from another build"},
    {"disease": None, "gene": None, "patients_like_me": None},
])
def test_the_task_options_carry_readiness_and_follow_the_skill(readiness):
    choices = fp.task_choices(readiness)
    assert [value for _label, value in choices] == ["disease", "gene", "patients_like_me"]
    _option_rules([label for label, _value in choices])
    for (label, task) in choices:
        assert label.endswith(" · ready") == (readiness.get(task, "x") is None)


def test_why_a_task_is_not_ready_is_said_under_the_group():
    note = fp.tasks_note_html({"disease": None, "gene": "no functional fixture",
                               "patients_like_me": "no functional fixture"})
    assert "Not ready: Gene and Patients-like-me: no functional fixture." in note
    assert fp.tasks_note_html({"disease": None, "gene": None, "patients_like_me": None}) == ""
    mixed = fp.tasks_note_html({"disease": TAG, "gene": "x", "patients_like_me": "x"})
    assert "<img" not in mixed and "Disease: &lt;img" in mixed and "Gene and Patients-like-me: x" \
        in mixed


def test_the_page_opens_on_its_own_dataset_starting_from_trainings(root):
    state, dataset, identity, tasks, note = fp._open({}, "demo", ["gene", "nope"])
    assert state["name"] == "demo" and dataset["value"] == "demo"
    assert state["row"]["dataset_fingerprint"][:12] in identity
    assert tasks["value"] == ["gene"]
    assert [value for _label, value in tasks["choices"]] == ["disease", "gene", "patients_like_me"]
    # Its own choice wins over Training's afterwards.
    assert fp._open({"name": "demo"}, "elsewhere", [])[0]["name"] == "demo"


def test_the_page_never_writes_the_training_workspace():
    demo, ctx, _by_id = _page()
    for fn in demo.fns.values():
        assert ctx.training_workspace not in fn.outputs, fn.name


# --------------------------------------------------------------------------- Start
@pytest.fixture
def started(monkeypatch):
    calls = []

    def start(inputs, **kwargs):
        calls.append((inputs, kwargs))
        return {"success": True}

    monkeypatch.setattr(fp.training_manager, "start_finetune", start)
    return calls


def test_start_sends_the_listed_digest_the_fingerprint_shown_and_what_differs(
        root, operator, started):
    from src.api.services.server_lists import resolve_checkpoint, resolve_workspace

    ws = tc.workspace_state("demo")
    parents = fp._on_finetune_bases(operator, ws, None)[0]
    key = f"gatv2/{BASE_RUN}/last.pt"
    said = fp._on_finetune_start(operator, ws, parents, key, ["patients_like_me", "disease"],
                                 100, 7.0, "cpu", "  ft  ")
    assert said == ""
    inputs, kwargs = started[-1]
    assert inputs.tasks == ("disease", "patients_like_me")           # TASKS order
    assert inputs.settings == {"seed": 7} and type(inputs.settings["seed"]) is int
    assert inputs.expected_parent_sha256 == parents[key]["sha256"]
    assert inputs.finetune_from == str(resolve_checkpoint("demo", "gatv2", BASE_RUN, "last.pt"))
    assert inputs.data_dir == str(resolve_workspace("demo"))
    assert (inputs.device, inputs.run_name) == ("cpu", "ft")
    assert kwargs == {"expected_dataset_fingerprint": ws["row"]["dataset_fingerprint"]}
    # The recipe's own values are not sent.
    fp._on_finetune_start(operator, ws, parents, key, ["gene"], 100, 33, "auto", "")
    assert started[-1][0].settings == {}


@pytest.mark.parametrize("change, words", [
    ({"tasks": []}, "choose at least one task"),
    ({"key": None}, "choose a base checkpoint"),
    ({"epochs": 0}, "Epochs 0 is outside"),
    ({"seed": None}, "Seed must be set"),
    ({"ws": {"name": None, "row": None}}, "select a dataset"),
])
def test_start_refuses_what_it_cannot_send_and_starts_nothing(root, operator, started, change,
                                                              words):
    ws = tc.workspace_state("demo")
    parents = fp._on_finetune_bases(operator, ws, None)[0]
    args = {"ws": ws, "parents": parents, "key": f"gatv2/{BASE_RUN}/last.pt",
            "tasks": ["disease"], "epochs": 100, "seed": 33}
    args.update(change)
    said = fp._on_finetune_start(operator, args["ws"], args["parents"], args["key"],
                                 args["tasks"], args["epochs"], args["seed"], "cpu", "")
    assert started == [] and words in said and "Not started" in said


def test_a_parent_listed_for_another_workspace_is_not_started(root, operator, started):
    ws = tc.workspace_state("demo")
    parents = fp._on_finetune_bases(operator, ws, None)[0]
    key = f"gatv2/{BASE_RUN}/last.pt"
    parents = {key: {**parents[key], "workspace": "other"}}
    said = fp._on_finetune_start(operator, ws, parents, key, ["disease"], 100, 33, "cpu", "")
    assert started == [] and "the list shows other’s bases, not demo’s" in said


def test_a_refused_start_shows_the_runners_sentence_escaped(root, operator, monkeypatch):
    monkeypatch.setattr(fp.training_manager, "start_finetune",
                        lambda inputs, **k: {"success": False,
                                             "error": f"The parent {TAG} was trained on another "
                                                      "knowledge graph. Nothing was started."})
    ws = tc.workspace_state("demo")
    parents = fp._on_finetune_bases(operator, ws, None)[0]
    said = fp._on_finetune_start(operator, ws, parents, f"gatv2/{BASE_RUN}/last.pt",
                                 ["disease"], 100, 33, "cpu", "")
    assert "<img" not in said and "&lt;img" in said and "another knowledge graph" in said


def test_start_is_refused_for_a_read_only_request(root, started):
    said = fp._on_finetune_start(READ_ONLY, tc.workspace_state("demo"), {}, None, ["disease"],
                                 100, 33, "cpu", "")
    assert "Refused" in said and started == []


# --------------------------------------------------------------------------- the job card
def _status(*children, status="running", parent=None):
    return {"status": status, "job_id": "j2",
            "finetune": {"parent": parent or f"/w/demo/runs/gatv2/{BASE_RUN}/checkpoints/last.pt",
                         "parent_sha256": "b" * 64, "tasks": [c["task"] for c in children],
                         "cancelled": False, "children": list(children)}}


def _child(task, status, run=None, error=None):
    return {"task": task, "status": status, "job_id": None,
            "run_dir": f"/w/demo/runs/gatv2/{run}" if run else None, "error": error}


@pytest.mark.parametrize("status, shape, token, word", [
    ("queued", "○", "neutral", "Queued"),
    ("running", None, "info", "Running"),
    ("stopping", None, "info", "Stopping"),
    ("completed", "●", "ok", "Completed"),
    ("failed", "✕", "error", "Failed"),
    ("refused", "✕", "error", "Refused"),
    ("stopped", "○", "neutral", "Stopped"),
    ("cancelled", "○", "neutral", "Cancelled"),
])
def test_each_status_is_colour_shape_and_word(status, shape, token, word):
    html = fp.status_mark_html(status)
    assert f"shep-ft-st-{token}" in html and f"<span>{word}</span>" in html
    assert ("shep-ft-spin" in html) if shape is None else (f">{shape}</span>" in html)
    css = fp.PAGE_CSS
    assert f".shep-ft-st-{token} {{ color: var(--shep-status-{token}); }}" in css
    # Never green for a child that did not complete.
    assert (token == "ok") == (status == "completed")
    assert tc.PILL_KIND[status] in fp._MARKS


def test_the_card_draws_one_row_per_child_escaped_without_a_path(root):
    status = _status(_child("disease", "completed", "20261028-1_ft-disease_a"),
                     _child("gene", "failed", "20261028-2_ft-gene_b", error=f"{TAG}\nTraceback"),
                     _child("patients_like_me", "queued"))
    html = fp.job_card_html(status, operator=False)
    assert "<img" not in html and "&lt;img src=x onerror=alert(1)&gt;" in html
    assert "/w/demo" not in html and "Traceback" not in html        # the first line only
    rows = re.findall(r"<tr>(.*?)</tr>", html)
    assert len(rows) == 4                                          # the header and three children
    assert "Disease" in rows[1] and "Gene" in rows[2] and "Patients-like-me" in rows[3]
    assert "gatv2/20261028-1_ft-disease_a" in rows[1] and "not started yet" in rows[3]
    assert "after it ends" in rows[3]
    assert "GATv2 · 2026-10-01 00:00" in html and "bbbbbbbbbbbb" in html
    assert "not shown: reading the checkpoints is an operator control" in rows[1]
    assert "Validation (Trainer)" in rows[0] and "Best save" in rows[0]
    cancelled = fp.job_card_html(_status(_child("gene", "cancelled")), operator=True)
    assert "never started" in cancelled and "no run" in cancelled


def test_the_best_save_is_the_highest_trainer_figure_never_a_file_name(root, monkeypatch):
    """The products of an ended child, for an operator: the saved file with
    the highest Trainer validation figure its record holds, ties shared, read
    from the list's metadata (the file names here say otherwise)."""
    from src.api.services import server_lists

    def item(filename, value, epochs):
        return server_lists.CheckpointItem(
            workspace="demo", architecture="gatv2", run="20261028-1_ft-disease_a",
            filename=filename, sha256="c" * 64, size_bytes=1, checkpoint_kind="training_state",
            epoch=epochs - 1, validation={"metric": "val_pooled_mrr", "value": value},
            dataset_fingerprint=None, trained_on_this_dataset=None, problem=None,
            completed_epochs=epochs, record="model", learning_stage="finetune", task="disease")

    seen = []
    listing = server_lists.Listing((item("last.pt", 0.40, 5), item("model-04-0.9000.pt", 0.40, 4),
                                    item("model-02-0.9500.pt", 0.25, 2)))
    monkeypatch.setattr(server_lists, "list_checkpoints",
                        lambda *a, **k: seen.append((a, k)) or listing)
    monkeypatch.setattr(tc, "workspace_of_run", lambda run_dir: "demo")
    child = _child("disease", "completed", "20261028-1_ft-disease_a")
    products = fp.products_of(_status(child)["finetune"], {})
    assert seen == [(("demo", "gatv2"), {"task": "disease", "run": "20261028-1_ft-disease_a"})]
    (found,) = products.values()
    assert [entry["words"] for entry in found["best"]] == ["Latest save · 5 epochs completed",
                                                           "Top-k save · 4 epochs completed"]
    html = fp.job_card_html(_status(child), operator=True, products=products)
    assert "Latest save · 5 epochs completed" in html and "Top-k save · 4 epochs" in html
    assert "0.4000" in html and "Validation pooled MRR (Trainer)" in html
    assert "test" not in re.sub(r"<[^>]+>", "", html).lower().replace("latest", "")
    # Read once per child ending: the session's record is taken next time.
    assert fp.products_of(_status(child)["finetune"], products) == products and len(seen) == 1
    # A running child's products are not read.
    fp.products_of(_status(_child("disease", "running", "r"))["finetune"], {})
    assert len(seen) == 1


def test_the_card_says_why_there_is_no_fine_tune_job():
    idle = fp.job_card_html({"status": "idle", "finetune": None}, operator=True)
    assert "No Fine-tune job since the server started" in idle
    cleared = fp.job_card_html({"status": "completed", "finetune": None}, operator=True)
    assert "the latest training job is a base or continued run" in cleared


# --------------------------------------------------------------------------- Start and Stop
class _Access:
    def __init__(self, allowed, condition="enabled"):
        self.allowed, self.condition = allowed, condition


def test_start_and_stop_follow_the_training_run_bars_states():
    child = _child("disease", "running", "r")
    idle = fp.actions_view({"status": "idle"}, access=_Access(True), gpu=None, confirming=False)
    assert (idle["start"], idle["stop"]) == (True, False) and "Stop disabled" in idle["reason"]
    refused = fp.actions_view({"status": "idle"}, access=_Access(False, "deployment_mode"),
                              gpu=None, confirming=False)
    assert (refused["start"], refused["stop"]) == (False, False)
    assert "Operator controls only" in refused["reason"] and "shep-lock" in refused["reason"]
    running = fp.actions_view(_status(child), access=_Access(True), gpu=None, confirming=False)
    assert (running["start"], running["stop"]) == (False, True)
    asking = fp.actions_view(_status(child), access=_Access(True), gpu=None, confirming=True)
    assert asking["stop"] is False
    other = fp.actions_view({"status": "running"}, access=_Access(True), gpu=None,
                            confirming=False)
    assert (other["start"], other["stop"]) == (False, False)
    assert "a training job is running" in other["reason"]
    stopping = fp.actions_view({"status": "stopping"}, access=_Access(True), gpu=None,
                               confirming=False)
    assert "Stopping" in stopping["reason"] and not stopping["start"]
    held = fp.actions_view({"status": "idle"}, access=_Access(True),
                           gpu={"text": f"GPU job: test run {TAG}"}, confirming=False)
    assert held["start"] is False and "<img" not in held["reason"]


def test_stop_asks_with_the_training_pages_stop_logic(monkeypatch):
    """One stop logic: the card's Stop and its confirmation are thin handlers
    of their own (D8 on each request; the deployment probe finds the run bar's
    Stop registered once), over the Training page's question and stop; Keep is
    the Training page's own."""
    import inspect

    demo, _ctx, by_id = _page()
    wired = {}
    for fn in demo.fns.values():
        for block, event in fn.targets:
            elem = next((b.elem_id for b in demo.blocks.values() if b._id == block), None)
            if elem in ("ft_stop", "ft_stop_confirm", "ft_stop_keep") and event == "click":
                wired[elem] = fn.fn
    assert wired == {"ft_stop": fp._on_finetune_stop_request, "ft_stop_confirm": fp._on_finetune_stop,
                     "ft_stop_keep": tc.close_stop_question}
    assert "tc.ask_stop(operator_access(request)" in inspect.getsource(fp._on_finetune_stop_request)
    assert "tc.confirm_stop(operator_access(request)" in inspect.getsource(fp._on_finetune_stop)
    status = _status(_child("disease", "running", "r"), _child("gene", "queued"))
    monkeypatch.setattr(tc.training_manager, "get_status", lambda: status)
    operator_mode(monkeypatch)
    text, mode = fp._on_finetune_stop_request(gradio_request(), {})
    assert mode["confirm"] == "j2" and "The queued Fine-tune children (Gene) will not start." in text
    monkeypatch.setattr(tc.training_manager, "stop_training", lambda: {
        "success": True, "outcome": None, "cancelled_children": ["gene"]})
    notice, mode = fp._on_finetune_stop(gradio_request(), mode)
    assert "(Gene) were cancelled" in notice and mode["confirm"] is None


# --------------------------------------------------------------------------- the page contract
def test_the_page_polls_on_the_slow_timer_without_the_queue_only_when_shown(monkeypatch):
    demo, ctx, by_id = _page()
    ticks = [fn for fn in demo.fns.values() if any(t[1] == "tick" for t in fn.targets)]
    assert [fn.fn for fn in ticks] == [fp._poll]
    (tick,) = ticks
    assert tick.targets == [(ctx.timer_slow._id, "tick")] and tick.queue is False
    assert tick.inputs[0] is ctx.location and tick.show_progress == "hidden"
    monkeypatch.setattr(fp.training_manager, "get_status",
                        lambda: pytest.fail("a tick on another page read the status"))
    assert fp._poll("training", {}, {}, gradio_request()) == (gr.update(),) * 6
    assert not [b for b in demo.blocks.values() if isinstance(b, gr.Timer)
                and b not in (ctx.timer_fast, ctx.timer_slow)]


def test_the_page_follows_the_contract():
    from src.webui.app import PAGE_CSS, PAGES, mount_options

    demo, _ctx, by_id = _page()
    assert fp.PAGE == shell.FINETUNE and PAGES[shell.FINETUNE] is fp.create_finetune_page
    assert fp._PAGE == "#" + shell.page_elem_id(shell.FINETUNE)
    assert fp.PAGE_CSS in PAGE_CSS and fp.PAGE_CSS in mount_options()["head"]
    for value in re.findall(r"font-size:\s*([0-9.]+)px", fp.PAGE_CSS):
        assert float(value) >= 14, value
    assert not re.search(r"#[0-9a-fA-F]{3,6}\b(?!-)", fp.PAGE_CSS.replace("#fff", "")), (
        "a status colour outside the tokens")
    ids = [b.elem_id for b in demo.blocks.values() if getattr(b, "elem_id", None)]
    own = [i for i in ids if not i.startswith("shep-")]
    assert own and all(i.startswith("ft_") for i in own), own
    for elem_id, (title, text) in fp.FIELD_HELP.items():
        assert by_id[elem_id].label == title, elem_id
        assert len(text.split("\n", 1)[0]) <= 180, elem_id
    assert set(fp.FIELD_HELP) >= {"ft_dataset", "ft_run", "ft_checkpoint", "ft_tasks",
                                  "ft_num_epochs", "ft_seed", "ft_device", "ft_run_name"}
    # Epochs and Seed only, at the recipe's finetune values.
    from src.config.recipe import FINETUNE

    assert (by_id["ft_num_epochs"].value, by_id["ft_seed"].value) == (
        FINETUNE["num_epochs"], FINETUNE["seed"])
    assert by_id["ft_num_epochs"].precision == 0 and by_id["ft_seed"].precision == 0
    for gone in ("ft_learning_rate", "ft_batch_size", "ft_task_loss_weight"):
        assert gone not in by_id
