"""
Research › Prepare › Fine-tune (K6 decisions 2, 9, 11).
======================================================
`src/webui/components/finetune_page.py`. What these pin:

- **the parents are listed bases only**: never a task model, a file without a
  model spec or a refused record; listing is an operator control;
- **the checkpoint and task options follow the skill §8** (unique, at most
  40 characters, one separator), the tasks' carrying their readiness, why not
  under them; the Run options are the identity label the Training page's
  Continue drawer shares (`renderers.run_words`: the name or start time
  first, then the architecture and time, one separator), unique;
- **Start** sends the base's listed digest, the dataset fingerprint shown, the
  tasks in `TASKS` order and the settings the operator touched, whatever their
  value (decision S2), through the server's lists (never a client path); a
  refusal is the runner's sentence, beside Start with the shared mark of a
  refusal, escaped;
- **the settings** (ops-wiring round, B2): what every chosen task's run takes
  (`run_plan.new_run_settings`), the base's split seed and model never a field
  but facts of its summary; a task's own settings only with one task, from its
  reference; the hop list for any task set, its note the start's own check
  against the base's encoder;
- **the job card**: each child's status as colour (the ``--shep-status-*``
  tokens), shape and word; everything escaped; products for an operator only,
  once the child ended, the best save by the selection its files record (the
  recorded monitor in its recorded direction, named under it; never a file
  name), ties shared; its own words once the Fine-tune status is cleared;
- **Start and Stop** are decided as the Training run bar's are (D8, a job
  running or stopping, the GPU slot);
- **the page contract**: its own Dataset select (starting from Training's,
  never writing it), the slow timer without the queue doing nothing on
  another page, ``ft_*`` ids, help titles equal to labels.

Module: tests/unit/test_webui_finetune_page.py
"""
from __future__ import annotations

import json
import re
import shutil

import pytest

gr = pytest.importorskip("gradio")

from src.webui import shell  # noqa: E402
from src.webui.components import finetune_page as fp  # noqa: E402
from src.webui.components import model_list as ml  # noqa: E402
from src.webui.components import renderers as r  # noqa: E402
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
    _put(ws, "hgt", "20261003-000000_hist_dddddd", GNN_PAYLOAD, 500)
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
        assert len(label) <= r.OPTION_CHARS and label.count(" · ") <= 1, label


# --------------------------------------------------------------------------- the parents
def test_only_listed_bases_are_offered_as_parents(root, operator):
    parents, runs, checkpoints, note, refresh = fp._on_finetune_bases(operator, {"name": "demo"},
                                                                      None)
    assert refresh == gr.update(value="Refresh", interactive=True)      # no lock while allowed
    assert set(parents) == {f"gatv2/{BASE_RUN}/last.pt", f"gatv2/{BASE_RUN}/model-03-0.8300.pt",
                            f"gatv2/{OLD_BASE_RUN}/last.pt"}
    assert all(item["learning_stage"] == "base" for item in parents.values())
    assert runs["value"] == f"gatv2/{BASE_RUN}" and checkpoints["value"] == (
        f"gatv2/{BASE_RUN}/last.pt")
    assert "3 base checkpoints in 2 runs" in note
    # Each select checked on its own: the Run options are the identity label
    # (`renderers.run_words`, shared with the Continue drawer), unique; the
    # checkpoint options follow §8; each value is the listing's key.
    run_labels = [label for label, _key in runs["choices"]]
    assert run_labels == ["base · GATv2 2026-10-01 00:00", "old · GATv2 2026-09-01 00:00"]
    _option_rules(run_labels)
    assert [key for _label, key in runs["choices"]] == [f"gatv2/{BASE_RUN}",
                                                       f"gatv2/{OLD_BASE_RUN}"]
    _option_rules([label for label, _key in checkpoints["choices"]])
    assert [key for _label, key in checkpoints["choices"]] == [
        f"gatv2/{BASE_RUN}/last.pt", f"gatv2/{BASE_RUN}/model-03-0.8300.pt"]
    # A choice kept by key stays chosen.
    kept = fp._on_finetune_bases(operator, {"name": "demo"}, f"gatv2/{OLD_BASE_RUN}/last.pt")
    assert kept[1]["value"] == f"gatv2/{OLD_BASE_RUN}" and kept[2]["value"] == (
        f"gatv2/{OLD_BASE_RUN}/last.pt")


def test_listing_the_bases_is_an_operator_control(root, monkeypatch):
    from src.api.services import server_lists

    monkeypatch.setattr(server_lists, "list_checkpoints",
                        lambda *a, **k: pytest.fail("a refused request listed checkpoints"))
    parents, runs, checkpoints, note, refresh = fp._on_finetune_bases(READ_ONLY, {"name": "demo"},
                                                                      None)
    assert parents == {} and runs == checkpoints == gr.update(choices=[], value=None)
    # The one gated-button rule (the closing browser check: Refresh stayed
    # enabled, with no lock, for a read-only request): disabled, the lock, and
    # "Operator controls only: …" beside it, as Start's reason line says it.
    from src.webui.components.chips import LOCK, operator_only
    from src.webui.operator_access import operator_access

    assert refresh == gr.update(value=f"{LOCK} Refresh", interactive=False)
    assert r.e(operator_only(operator_access(READ_ONLY).condition)) in note
    assert "listing the bases opens them" in note


def test_an_empty_list_says_why(root, operator):
    assert "Select a dataset first." in fp._on_finetune_bases(operator, {}, None)[3]
    assert "train one on Research › Training" in fp.bases_note_html({}, None)
    assert "&lt;img" in fp.bases_note_html({}, TAG) and "<img" not in fp.bases_note_html({}, TAG)


def test_the_chosen_bases_summary_says_what_start_sends_escaped(root, operator):
    parents = fp._on_finetune_bases(operator, {"name": "demo"}, None)[0]
    item = parents[f"gatv2/{BASE_RUN}/model-03-0.8300.pt"]
    html = fp._on_base_chosen(f"gatv2/{BASE_RUN}/model-03-0.8300.pt", parents)
    for words in ("Run", f"<code>gatv2/{BASE_RUN}</code>", "model-03-0.8300.pt", "Configuration",
                  "GATv2 · recipe changed in", '<summary>Details ›</summary>',
                  "Holds", "Validation", "LP accuracy (Trainer)",
                  "0.8300", "Selection", "Trained on", "not recorded", "Split seed",
                  "Model fingerprint", item["sha256"][:12], f'data-copy="{item["sha256"]}"'):
        assert words in html, words
    # The run is named by its directory, a fact the Run select above does not
    # give; never its option's words again (the WebUI skill §1.2).
    assert r.run_words("gatv2", BASE_RUN) not in html
    assert fp._on_base_chosen(None, parents) == ""
    odd = fp.base_summary_html({**item, "filename": TAG, "trained_on_this_dataset": False})
    assert "<img" not in odd and "&lt;img" in odd and "another dataset" in odd
    same = fp.base_summary_html({**item, "trained_on_this_dataset": True})
    assert "the dataset selected" in same
    # Each state by the shared status mark (colour, shape and word; B2's
    # leftover: no ✓ of the page's own).
    assert r.status_mark("ok", "the dataset selected") in same
    assert r.status_mark("warn", "another dataset than the one selected (the start refuses a "
                                 "parent of another knowledge graph)") in odd
    assert r.status_mark("neutral", "not recorded") in html


def test_the_bases_split_seed_and_model_are_facts_of_its_summary_never_fields(root, operator):
    """B2 (decision S4): a finetune trains on its base's split and keeps its
    base's graph side, so the split seed and the model spec are never fields
    (`BASE_FACTS`); the summary says the split seed its record names
    (`listed_record`'s `split_seed`, from ``data_roles["split"]``) and its
    configuration. A record naming no split seed says so, never a number."""
    parents = fp._on_finetune_bases(operator, {"name": "demo"}, None)[0]
    item = parents[f"gatv2/{BASE_RUN}/last.pt"]
    # The fixture's record names its split by a digest alone: no seed.
    assert item["split_seed"] is None
    assert "<dt class=\"shep-k\">Split seed</dt><dd class=\"shep-v\">not recorded</dd>" in (
        fp.base_summary_html(item))
    seeded = fp.base_summary_html({**item, "split_seed": 7})
    assert "7 (every child trains on this split)" in seeded
    assert seeded.count("Split seed") == 1 and "Configuration" in seeded
    # Its model: the one configuration line, and beside it the whole spec
    # under its Details by the one renderer of a recorded spec, read from the
    # list's spec (`record.listed_spec`, without the recipe's notes), each
    # field that differs with the reference's value, the encoder's layers the
    # hop list must fit among them (review of B2).
    words, details = r.model_spec_html(item["spec"])
    assert "Unknown" not in words and details
    assert (f"<dt class=\"shep-k\">Configuration</dt><dd class=\"shep-v\">"
            f"{r.e(r.configuration_text(item))}{details}</dd>") in seeded
    assert ('<dt class="shep-k" title="encoder.layers">Encoder › Layers</dt>'
            '<dd class="shep-v">1 (reference 3)</dd>') in details
    for tasks in ([], ["disease"], ["gene"], ["disease", "gene", "patients_like_me"]):
        assert not set(fp.offered_settings(tasks)) & {"split_seed", "model_spec"}
    assert not set(fp.FINETUNE_FIELDS) & fp.BASE_FACTS


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


def _values(**changed):
    """The page's field values in `FINETUNE_FIELDS` order, each as its field
    holds it when the page is built (`built_values`, `form_value`), those in
    `changed` replaced."""
    built = fp.built_values()
    return [changed.get(name, tc.form_value(built[name])) for name in fp.FINETUNE_FIELDS]


def _start_with(operator, ws, parents, key, tasks, touched, device="cpu", run_name="",
                **changed):
    return fp._on_finetune_start(operator, ws, parents, key, tasks, device, run_name, touched,
                                 *_values(**changed))


def test_start_sends_the_listed_digest_the_fingerprint_shown_and_what_was_touched(
        root, operator, started):
    from src.api.services.server_lists import resolve_checkpoint, resolve_workspace

    ws = tc.workspace_state("demo")
    parents = fp._on_finetune_bases(operator, ws, None)[0]
    key = f"gatv2/{BASE_RUN}/last.pt"
    said, stop_notice = _start_with(operator, ws, parents, key, ["patients_like_me", "disease"],
                                    ["seed"], run_name="  ft  ", seed=7.0)
    assert (said, stop_notice) == ("", "")
    inputs, kwargs = started[-1]
    assert inputs.tasks == ("disease", "patients_like_me")           # TASKS order
    assert inputs.settings == {"seed": 7} and type(inputs.settings["seed"]) is int
    assert inputs.expected_parent_sha256 == parents[key]["sha256"]
    assert inputs.finetune_from == str(resolve_checkpoint("demo", "gatv2", BASE_RUN, "last.pt"))
    assert inputs.data_dir == str(resolve_workspace("demo"))
    assert (inputs.device, inputs.run_name) == ("cpu", "ft")
    assert kwargs == {"expected_dataset_fingerprint": ws["row"]["dataset_fingerprint"]}
    # Untouched, nothing is sent, whatever the fields hold (decision S2) ...
    _start_with(operator, ws, parents, key, ["gene"], [], device="auto", num_epochs=5, seed=7,
                batch_size=8, learning_rate=0.5)
    assert started[-1][0].settings == {}
    # ... and touched, the recipe's own values are sent: equal to the
    # reference is never read as "not set".
    _start_with(operator, ws, parents, key, ["gene"], ["num_epochs", "seed", "batch_size"],
                device="auto")
    assert started[-1][0].settings == {"num_epochs": 100, "seed": 33, "batch_size": 64}


def test_a_frozen_graph_backbone_reaches_the_start_and_the_note_says_what_trains(
        root, operator, started):
    """Ops-wiring round, decision F1: the freeze is a field of the page's
    common settings; Frozen reaches the runner's start as the start's
    settings (`FinetuneInputs.settings`, which the runner alone checks), for
    every chosen task; untouched, nothing is sent. While Frozen is
    chosen a visible note says what then trains (skill §7: not only a
    bubble)."""
    ws = tc.workspace_state("demo")
    parents = fp._on_finetune_bases(operator, ws, None)[0]
    key = f"gatv2/{BASE_RUN}/last.pt"
    assert _start_with(operator, ws, parents, key, ["disease", "gene"], ["frozen_parts"],
                       frozen_parts="graph_side") == ("", "")
    assert started[-1][0].settings == {"frozen_parts": "graph_side"}
    assert started[-1][0].tasks == ("disease", "gene")
    _start_with(operator, ws, parents, key, ["disease"], ["num_epochs"], num_epochs=5,
                frozen_parts="graph_side")
    assert started[-1][0].settings == {"num_epochs": 5}
    assert fp.freeze_note_html("none") == ""
    # Which values leave parts untrained is the recipe's rule, read from it
    # (`recipe.FROZEN_PARTS`; review of A2: no value of the page's own).
    from src.config.recipe import FROZEN_PARTS

    assert [value for value in FROZEN_PARTS if fp.freeze_note_html(value)] == [
        value for value, parts in FROZEN_PARTS.items() if parts] == ["graph_side"]
    assert fp.freeze_note_html(None) == fp.freeze_note_html("all") == ""
    note = fp.freeze_note_html("graph_side")
    assert note == (f'<p class="shep-ft-why">{fp.FROZEN_NOTE}</p>')
    assert "only the patient encoder and the task head train" in note
    assert "the LP term trains nothing" in note


@pytest.mark.parametrize("change, words", [
    ({"tasks": []}, "choose at least one task"),
    ({"key": None}, "choose a base checkpoint"),
    ({"num_epochs": 0}, "Epochs must be between 1 and 10000, not 0."),
    ({"seed": None}, "Seed must be set"),
    ({"frozen_parts": "all"}, "Graph backbone must be one of [&#x27;none&#x27;, "
                              "&#x27;graph_side&#x27;], not &#x27;all&#x27;."),
    ({"num_neighbors": "15, x"}, "Neighbours per hop must be whole numbers separated by "
                                 "commas"),
    ({"ws": {"name": None, "row": None}}, "select a dataset"),
])
def test_start_refuses_what_it_cannot_send_and_starts_nothing(root, operator, started, change,
                                                              words):
    ws = tc.workspace_state("demo")
    parents = fp._on_finetune_bases(operator, ws, None)[0]
    args = {"ws": ws, "parents": parents, "key": f"gatv2/{BASE_RUN}/last.pt",
            "tasks": ["disease"]}
    fields = {name: value for name, value in change.items() if name in fp.FINETUNE_FIELDS}
    args.update({name: value for name, value in change.items() if name not in fields})
    said, _stop = _start_with(operator, args["ws"], args["parents"], args["key"], args["tasks"],
                              list(fp.FINETUNE_FIELDS), **fields)
    assert started == [] and words in said and "Not started" in said
    # Beside Start, with the shared mark of a refusal (the skill's §6, §9).
    assert said.startswith('<span class="shep-notice-t"><span class="shep-st shep-st-error">')


def test_a_parent_listed_for_another_workspace_is_not_started(root, operator, started):
    ws = tc.workspace_state("demo")
    parents = fp._on_finetune_bases(operator, ws, None)[0]
    key = f"gatv2/{BASE_RUN}/last.pt"
    parents = {key: {**parents[key], "workspace": "other"}}
    said, _stop = _start_with(operator, ws, parents, key, ["disease"], [])
    assert started == [] and "the list shows other’s bases, not demo’s" in said


def test_a_refused_start_shows_the_runners_sentence_escaped(root, operator, monkeypatch):
    monkeypatch.setattr(fp.training_manager, "start_finetune",
                        lambda inputs, **k: {"success": False,
                                             "error": f"The parent {TAG} was trained on another "
                                                      "knowledge graph. Nothing was started."})
    ws = tc.workspace_state("demo")
    parents = fp._on_finetune_bases(operator, ws, None)[0]
    key = f"gatv2/{BASE_RUN}/last.pt"
    said, _stop = _start_with(operator, ws, parents, key, ["disease"], [])
    assert "<img" not in said and "&lt;img" in said and "another knowledge graph" in said
    # The runner's sentence already says nothing was started: not said twice
    # (K6 closing review: "Not started: … Nothing was started.").
    assert "Not started" not in said and said.count("Nothing was started.") == 1
    assert 'class="shep-st shep-st-error"' in said
    # A refusal that does not say it keeps the page's lead.
    monkeypatch.setattr(fp.training_manager, "start_finetune",
                        lambda inputs, **k: {"success": False, "error": "The slot is busy."})
    said, _stop = _start_with(operator, ws, parents, key, ["disease"], [])
    assert said == tc.notice_html("Not started: The slot is busy.", mark="error")


def test_the_runners_refusal_of_the_settings_is_said_beside_start_in_its_words(
        root, operator, monkeypatch):
    """B2 (decisions S4, F3): the start's invalid settings (422 on the API)
    and its checks against the base are said beside Start in the runner's
    words, a sentence that names a setting first named by the field's label
    (`training_console.labelled_problem`, as the Training page's Branch says
    them); its other words, a hop list's among them, as the runner wrote
    them."""
    from types import SimpleNamespace

    from src.api.services.training_manager import _finetune_base_problem
    from src.models.bundle.record import listed_spec

    ws = tc.workspace_state("demo")
    parents = fp._on_finetune_bases(operator, ws, None)[0]
    key = f"gatv2/{BASE_RUN}/last.pt"
    # The runner's own check against the base (the fixture's base has one
    # encoder layer): the reference hop list cannot run on it.
    base = SimpleNamespace(model_spec=listed_spec(parents[key]["spec"])[0], data_roles={})
    why = _finetune_base_problem({}, ("disease",), base)
    assert why and "does not fit its base's encoder: num_neighbors needs 1 numbers" in why
    # The setting leads the clause after the colon: named by its label, as
    # the hop note beside the field names it (review of B2: one fact, one
    # name on the screen), on the Branch page alike (`labelled_problem`).
    labelled = why.replace(": num_neighbors needs", ": Neighbours per hop needs")
    assert tc.labelled_problem(why) == labelled
    for refusal, words in (
            ({"error": f"{why} Nothing was started."},
             r.e(f"{labelled} Nothing was started.")),
            ({"error": "min_lr_ratio must be more than 0 when scheduler_type is onecycle. "
                       "Nothing was started."},
             "Final learning-rate ratio must be more than 0 when scheduler_type is onecycle.")):
        monkeypatch.setattr(fp.training_manager, "start_finetune",
                            lambda inputs, refusal=refusal, **k: {
                                "success": False, "invalid_settings": True, **refusal})
        said, _stop = _start_with(operator, ws, parents, key, ["disease"], [])
        assert words in said, said
        assert said.count("Nothing was started.") == 1 and "shep-st-error" in said
        assert "num_neighbors" not in said and "min_lr_ratio" not in said


def test_start_is_refused_for_a_read_only_request(root, started):
    said, _stop = fp._on_finetune_start(READ_ONLY, tc.workspace_state("demo"), {}, None,
                                        ["disease"], "cpu", "", [], *_values())
    assert "Refused" in said and started == [] and "shep-st-error" in said


# --------------------------------------------------------------------------- the settings (B2)
def _subsets():
    from itertools import combinations

    from src.models.bundle.spec import TASKS

    return [list(c) for n in (1, 2, 3) for c in combinations(TASKS, n)]


def test_the_fields_are_what_every_chosen_tasks_run_takes():
    """B2 (decision S1): the page offers what every chosen task's run takes,
    the intersection of `run_plan.new_run_settings` over the tasks (the job
    runner's own rule of a Fine-tune start), but the base's facts; with one
    task, everything its run takes; with several, a task's own setting only
    where any task set takes it (the hop list). The page builds a field for
    each, once (`FINETUNE_FIELDS`): common first, the rest one click away."""
    from src.config.recipe import FINETUNE_TASKS
    from src.utils.run_plan import new_run_settings

    assert fp.COMMON_SHOWN == ("num_epochs", "seed", "frozen_parts")
    assert len(set(fp.FINETUNE_FIELDS)) == len(fp.FINETUNE_FIELDS)
    assert fp.TASK_OWN == frozenset().union(*FINETUNE_TASKS.values())
    assert fp.ANY_TASKS == ("num_neighbors",) and fp.TASK_FIELDS[0] == "num_neighbors"
    for tasks in _subsets():
        every = frozenset.intersection(*(new_run_settings("finetune", t) for t in tasks))
        offered = set(fp.offered_settings(tasks))
        if len(tasks) == 1:
            assert offered == every - fp.BASE_FACTS, tasks
        else:
            assert offered == every - fp.BASE_FACTS - (fp.TASK_OWN - {"num_neighbors"}), tasks
            assert offered & fp.TASK_OWN == {"num_neighbors"}, tasks
        assert offered <= set(fp.FINETUNE_FIELDS)
        assert set(fp.COMMON_SHOWN + fp.MORE_FIELDS) <= offered, tasks
    assert fp.offered_settings([]) == ()
    assert set(fp.MORE_FIELDS) & fp.TASK_OWN == set()


def test_task_settings_show_with_one_task_from_its_reference_and_not_with_two(
        root, operator, started):
    """B2: a task's own settings (learning rate, sample sizes, loss weights,
    candidate size) show only when exactly one task is chosen, starting from
    that task's reference; with two, a visible line says each child takes its
    own task's reference, and a value left in a hidden field is never sent."""
    from src.utils.run_plan import new_run_values

    item = fp._on_finetune_bases(operator, {"name": "demo"}, None)[0][
        f"gatv2/{BASE_RUN}/last.pt"]
    one = fp.task_settings_view(["patients_like_me"], item)
    shown = {name for name, update in zip(fp.TASK_FIELDS, one[2:-3], strict=True)
             if update["visible"]}
    assert shown == {"num_neighbors", "learning_rate", "sparse_sample_size", "task_loss_weight",
                     "lp_loss_weight", "candidate_upsample_size"}
    assert one[0]["visible"] is True and one[0]["label"] == "Patients-like-me settings"
    assert one[1]["visible"] is True                                 # the hop row
    assert one[-2] == ""                                             # no several-task line
    reference = new_run_values("finetune", "patients_like_me", {})
    assert fp.task_starts(["patients_like_me"]) == {
        name: tc.form_value(reference[name]) for name in shown}
    assert fp.task_starts(["patients_like_me"])["learning_rate"] == 5e-05
    two = fp.task_settings_view(["disease", "gene"], item)
    assert {name for name, update in zip(fp.TASK_FIELDS, two[2:-3], strict=True)
            if update["visible"]} == {"num_neighbors"}
    # With several, no section of a task's own: the hop row alone, in view.
    assert two[0]["visible"] is False and two[1]["visible"] is True
    line = two[-2]
    assert "Each child takes its own task’s reference for these: Learning rate" in line
    assert "Candidate sample size" in line and "Candidate upsample size" in line
    assert [update["visible"] for update in fp.task_settings_view([], item)[:2]] == [False] * 2
    # Start: a task setting the operator touched is sent with one task ...
    ws = tc.workspace_state("demo")
    parents = {f"gatv2/{BASE_RUN}/last.pt": item}
    key = f"gatv2/{BASE_RUN}/last.pt"
    _start_with(operator, ws, parents, key, ["disease"], ["learning_rate", "num_neighbors"],
                learning_rate=0.002, num_neighbors="4")
    assert started[-1][0].settings == {"learning_rate": 0.002, "num_neighbors": [4]}
    # ... never with two, whatever its field still holds; the hop list is.
    _start_with(operator, ws, parents, key, ["disease", "gene"],
                ["learning_rate", "candidate_sample_size", "num_neighbors"],
                learning_rate=0.002, num_neighbors="4")
    assert started[-1][0].settings == {"num_neighbors": [4]}


def _fields(tasks=(), **changed):
    """The page's field values in `FINETUNE_FIELDS` order where they start
    for `tasks` (`start_values`), those in `changed` replaced."""
    return [changed.get(name, value)
            for name, value in zip(fp.FINETUNE_FIELDS, fp.start_values(tasks), strict=True)]


def test_a_change_of_tasks_starts_each_task_setting_again():
    """The operator's own change of the tasks (`_on_tasks_input`): each of a
    task's own settings starts again at the new tasks' reference, having left
    the form's record (in the browser, `record_drops`; the handler reads the
    record without them all the same), so a value set for one task is never
    sent for another; a hop list the operator set stays, theirs for every
    child. With several tasks whose references differ, the hop list starts
    empty: untouched, each child takes its own (the note beside it says
    which). The handler never writes the record (review of B2): it returns
    the task fields and the line on what the operator set, from the record
    without a task's own settings."""
    *updates, line = fp._on_tasks_input(["gene"], ["seed", "learning_rate", "batch_size"],
                                        *_fields(["disease"], seed=3, learning_rate=0.5))
    by_name = dict(zip(fp.TASK_FIELDS, updates, strict=True))
    assert by_name["learning_rate"]["value"] == 1e-05
    assert by_name["candidate_upsample_size"]["value"] == 2
    assert by_name["num_neighbors"]["value"] == "-1, 10, 5"
    assert "value" not in by_name["candidate_sample_size"]          # not Gene's: kept, hidden
    assert line == fp.set_by_html(["gene"], dict(zip(fp.FINETUNE_FIELDS, _fields(
        ["gene"], seed=3), strict=True)), ["batch_size", "seed"])
    assert "Seed 3 (reference 33)" in line and "Learning rate" not in line
    *updates, line = fp._on_tasks_input(["disease", "gene"],
                                        ["num_neighbors", "sparse_sample_size"],
                                        *_fields(["disease"], num_neighbors="4"))
    assert all("value" not in update for update in updates)          # the hop list is theirs
    assert "Neighbours per hop 4 (each task’s own reference)" in line
    *updates, line = fp._on_tasks_input(["disease", "gene"], [], *_fields())
    assert dict(zip(fp.TASK_FIELDS, updates, strict=True))["num_neighbors"]["value"] == ""
    assert "Set by you: nothing" in line
    *updates, _line = fp._on_tasks_input(["disease", "patients_like_me"], [], *_fields())
    assert dict(zip(fp.TASK_FIELDS, updates, strict=True))["num_neighbors"]["value"] == (
        "15, 10, 5")
    note = fp.hops_note_html(["disease", "gene"], None)
    assert "Untouched, each child takes its task’s reference: Disease 15, 10, 5; Gene all " \
           "neighbours, 10, 5." in note
    assert fp.hops_note_html(["disease", "patients_like_me"], None) == ""
    # The record leaves the browser's keys as they are but those dropped.
    assert fp.TASK_OWN_FIELDS == tuple(n for n in fp.TASK_FIELDS if n not in fp.ANY_TASKS)
    assert tc.touched_without(["num_neighbors", "learning_rate", "seed", 3],
                              fp.TASK_OWN_FIELDS) == ["num_neighbors", "seed"]


def test_the_drop_script_leaves_the_record_in_the_browser():
    """`DROP_JS` (the tasks' input, run in the browser before any answer of
    the server): the record's keys but those dropped, in order, returned as
    the listener's one output; a record that is not a list starts empty. A
    key touched while the server answers is never undone (review of B2)."""
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    script = tc.DROP_JS.replace("{keys}", json.dumps(sorted(fp.TASK_OWN_FIELDS)))
    program = (f"const f = {script};"
               "console.log(JSON.stringify([f(['learning_rate', 'num_neighbors', 'seed']),"
               " f([]), f(null), f('x')]));")
    result = subprocess.run([node, "-e", program], capture_output=True, text=True, timeout=30,
                            check=True)
    assert json.loads(result.stdout) == [[["num_neighbors", "seed"]], [[]], [[]], [[]]]


def test_the_line_says_what_start_sends_and_reset_brings_back_each_tasks_reference(
        root, operator, started):
    """Decision S2 on the Fine-tune page (review of B2): the base form's one
    line on what the operator set (`training_console.set_by_line_html`),
    said by each field's input, names what Start sends, with what the
    children take without it; a value typed back to its reference is still
    the operator's. Reset to reference (`training_console.reset_outputs`)
    empties the record and puts every field where it starts for the tasks:
    with two tasks whose hop references differ, a hop list typed and then
    cleared is refused at Start, and after Reset Start sends no hop list, so
    each child takes its own task's reference again."""
    ws = tc.workspace_state("demo")
    parents = fp._on_finetune_bases(operator, ws, None)[0]
    key = f"gatv2/{BASE_RUN}/last.pt"
    tasks = ["disease", "gene"]
    listener = fp._input_listener("batch_size")
    line = listener(tasks, ["num_epochs"], *_fields(tasks, num_epochs=5, batch_size=64))
    assert line == fp.set_by_html(tasks, dict(zip(fp.FINETUNE_FIELDS, _fields(
        tasks, num_epochs=5, batch_size=64), strict=True)), ["batch_size", "num_epochs"])
    assert ("<p>Set by you: Epochs 5 (reference 100), Batch size 64 (reference 64)</p>"
            in line)
    # Typed, then cleared: still touched, so empty is sent and refused.
    cleared = _fields(tasks, num_neighbors="")
    hops_line = fp._input_listener("num_neighbors")(tasks, [], *cleared)
    assert "Not valid: Neighbours per hop needs at least 1 item(s)." in hops_line
    said, _stop = fp._on_finetune_start(operator, ws, parents, key, tasks, "cpu", "",
                                        ["num_neighbors"], *cleared)
    assert "Neighbours per hop needs at least 1 item(s)" in said and started == []
    # Reset to reference: the record empty, every field where it starts.
    record, reset_line, *updates = fp._on_finetune_reset(tasks)
    assert record == [] and [u["value"] for u in updates] == fp.start_values(tasks)
    assert dict(zip(fp.FINETUNE_FIELDS, (u["value"] for u in updates)))["num_neighbors"] == ""
    assert reset_line == tc.set_by_line_html(
        [], [], "each child takes its task’s reference values and this build’s defaults.")
    assert "Untouched, each child takes its task’s reference" in fp.hops_note_html(
        tasks, None, record, "")
    said, _stop = fp._on_finetune_start(operator, ws, parents, key, tasks, "cpu", "", record,
                                        *(u["value"] for u in updates))
    assert said == "" and "num_neighbors" not in started[-1][0].settings
    assert started[-1][0].settings == {}
    # One task: Reset puts its own settings at its reference.
    _record, _line, *updates = fp._on_finetune_reset(["gene"])
    values = dict(zip(fp.FINETUNE_FIELDS, (u["value"] for u in updates)))
    assert (values["learning_rate"], values["num_neighbors"]) == (1e-05, "-1, 10, 5")
    # The base form's line and Reset are the same functions (one mechanism).
    assert tc._on_base_reset()[0] == [] and tc._on_base_reset()[1] == tc.set_by_you_html(
        dict(zip(tc.base_form_keys(), tc.base_form_defaults(), strict=True)), [])


def test_the_hop_list_against_the_bases_encoder_is_the_starts_own_check(root, operator):
    """B2 (2b; decision S4): the hop list must fit its base's encoder (the
    start's check of each child, `spec.hops_problem`). On a base whose
    encoder layers differ from a task's reference hop list (the fixture's
    one-layer base), that sentence is beside the field, by its label, for
    one task or several, in the row under the common settings (no accordion
    to open), and beside Start a pointer to the field, in view at the
    deployment window (review of B2); a list typed is checked the same way,
    and one that fits says nothing. A base whose encoder the reference fits
    says nothing either."""
    from dataclasses import replace

    from src.models.bundle.spec import hops_problem, reference_spec

    item = fp._on_finetune_bases(operator, {"name": "demo"}, None)[0][
        f"gatv2/{BASE_RUN}/last.pt"]
    encoder = item["spec"]["encoder"]
    assert encoder["layers"] == 1
    sentence = hops_problem([15, 10, 5], replace(reference_spec("base").encoder, layers=1))
    assert sentence.startswith("num_neighbors needs 1 numbers")
    labelled = sentence.replace("num_neighbors", "Neighbours per hop", 1)
    for tasks in (["disease"], ["disease", "gene", "patients_like_me"]):
        view = fp.task_settings_view(tasks, item)
        # The hop row under the common settings, the field and its sentence
        # in view without opening anything (review of B2: "up front").
        assert view[1]["visible"] is True and view[2]["visible"] is True, tasks
        assert "open" not in view[0]
        note = view[-3]
        assert r.status_mark("warn", f"{fp.HOPS_REFERENCE_LEAD}: {labelled}") in note, note
        # Beside Start: where the field is, the sentence said once (beside it).
        assert view[-1] == ('<span class="shep-ft-hops-warn">' + r.status_mark(
            "warn", "Neighbours per hop does not fit this base: see the field under Epochs.")
            + "</span>")
        assert labelled not in view[-1]
    # A list typed that fits: no note, no pointer.
    fits = fp.task_settings_view(["disease"], item, ["num_neighbors"], "15")
    assert fits[-3] == "" and fits[-1] == ""
    key = f"gatv2/{BASE_RUN}/last.pt"
    line, note, pointer = fp._on_hops_input(["disease"], key, {key: item}, [],
                                            *_fields(["disease"], num_neighbors="15"))
    assert (note, pointer) == ("", "")
    assert "Set by you: Neighbours per hop 15 (reference 15, 10, 5)" in line
    line, typed, pointer = fp._on_hops_input(["disease"], key, {key: item}, [],
                                             *_fields(["disease"], num_neighbors="15, 10"))
    assert "Neighbours per hop needs 1 numbers" in typed and fp.HOPS_REFERENCE_LEAD not in typed
    assert "see the field under Epochs" in pointer and "Neighbours per hop 15, 10" in line
    # The reference base's encoder takes the reference hop lists.
    reference = {**item, "spec": {k: v for k, v in reference_spec("base").to_dict().items()
                                  if k != "sources"}}
    view = fp.task_settings_view(["disease", "gene"], reference)
    assert "shep-st-warn" not in view[-3] and view[-1] == ""
    # The page asks the start's own fill of each child's hop list
    # (`run_plan.new_run_hops`, as `_finetune_base_problem` does): stated, the
    # list typed for every child; else each task's reference.
    from src.utils.run_plan import new_run_hops

    assert [new_run_hops("finetune", t, {}) for t in ("disease", "gene")] == [
        [15, 10, 5], [-1, 10, 5]]
    assert new_run_hops("finetune", "gene", {"num_neighbors": [4]}) == [4]
    # A value Start would refuse is said once, by the line, never the note.
    assert fp.hops_problems(["disease"], item, ["num_neighbors"], "") == []
    # No base chosen, or a spec not read: nothing said (the start says it).
    assert fp.hops_problems(["disease"], None) == []
    assert fp.hops_problems(["disease"], {"spec": {"encoder": "?"}}) == []


def test_a_setting_a_rule_reads_only_under_its_guard_is_shown_while_read():
    """The common settings' guards are answered by the base form's one rule
    (`training_console.guard_updates`): Warmup steps with the cosine scheduler
    only, the final learning-rate ratio with a scheduler, the mixed precision
    dtype with mixed precision on."""
    dependent, guards = tc.rule_fields(fp.FINETUNE_FIELDS)
    assert set(dependent) == {"warmup_steps", "min_lr_ratio", "amp_dtype"}
    assert set(guards) == {"scheduler_type", "use_amp"} and set(dependent + guards) <= set(
        fp.MORE_FIELDS)
    values = {"scheduler_type": "none", "use_amp": False}
    shown = dict(zip(dependent, (u["visible"] for u in fp._on_guards(
        *(values[name] for name in guards))), strict=True))
    assert shown == {"warmup_steps": False, "min_lr_ratio": False, "amp_dtype": False}
    values = {"scheduler_type": "cosine", "use_amp": True}
    shown = dict(zip(dependent, (u["visible"] for u in fp._on_guards(
        *(values[name] for name in guards))), strict=True))
    assert shown == {"warmup_steps": True, "min_lr_ratio": True, "amp_dtype": True}


# --------------------------------------------------------------------------- the job card
def _status(*children, status="running", parent=None, settings=None):
    return {"status": status, "job_id": "j2",
            "finetune": {"parent": parent or f"/w/demo/runs/gatv2/{BASE_RUN}/checkpoints/last.pt",
                         "parent_sha256": "b" * 64, "settings": dict(settings or {}),
                         "tasks": [c["task"] for c in children],
                         "cancelled": False, "children": list(children)}}


def _child(task, status, run=None, error=None):
    """A child as `_Finetune.describe` gives it: a running one is the job
    running (`_status`'s ``j2``)."""
    return {"task": task, "status": status,
            "job_id": "j2" if status in ("running", "stopping") else None,
            "run_dir": f"/w/demo/runs/gatv2/{run}" if run else None, "error": error}


@pytest.mark.parametrize("status, shape, token, word", [
    ("queued", "○", "neutral", "Queued"),
    ("running", None, "info", "Running"),
    ("stopping", "■", "info", "Stopping"),
    ("completed", "●", "ok", "Completed"),
    ("failed", "✕", "error", "Failed"),
    ("refused", "✕", "error", "Refused"),
    ("stopped", "○", "neutral", "Stopped"),
    ("cancelled", "○", "neutral", "Cancelled"),
])
def test_each_status_is_colour_shape_and_word(status, shape, token, word):
    """Each child's status through the shared status mark
    (`renderers.status_mark`, the WebUI skill §6): its colour token, its shape
    and its word, the word always shown."""
    from src.webui import style

    html = tc.job_status_html(status)
    assert f"shep-st shep-st-{token}" in html and f'<span class="shep-st-w">{word}</span>' in html
    assert ("shep-spinner" in html) if shape is None else (f">{shape}</span>" in html)
    # The spinner is Running's alone (the WebUI skill §6).
    assert ("shep-spinner" in html) == (status == "running")
    assert f".shep-st-{token} {{ color: var(--shep-status-{token}); }}" in style.SHELL_CSS
    # Never green for a child that did not complete.
    assert (token == "ok") == (status == "completed")
    assert tc.PILL_KIND[status] in tc.STATUS_MARK_OF_KIND
    assert html in fp.child_row_html(_child("disease", status), operator=True, products={})


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
    assert "base · GATv2 2026-10-01 00:00" in html and "bbbbbbbbbbbb" in html
    assert "not shown: reading the checkpoints is an operator control" in rows[1]
    assert "Validation (Trainer)" in rows[0] and "Best save" in rows[0]
    cancelled = fp.job_card_html(_status(_child("gene", "cancelled")), operator=True)
    assert "never started" in cancelled and "no run" in cancelled


def test_the_card_says_the_starts_graph_backbone_in_the_fields_words_escaped():
    """Ops-wiring round, decision F1: the card names the freeze the start
    sent to every child (the runner's own record, `_Finetune.describe`), in
    the field's words, so the frozen control's children read as such while
    they run and after; a start that sent none is the recipe's, said so."""
    child = _child("disease", "running", "20261028-1_ft-disease_a")
    frozen = fp.job_card_html(_status(child, settings={"frozen_parts": "graph_side"}),
                              operator=False)
    line = ('<p class="shep-ft-from"><span class="shep-ft-k">Graph backbone</span> '
            'Frozen</p>')
    assert line in frozen and frozen.count("Graph backbone") == 1
    reference = fp.job_card_html(_status(child, settings={"num_epochs": 5}), operator=False)
    assert ('<span class="shep-ft-k">Graph backbone</span> Trained (reference)</p>'
            in reference)
    stated = fp.job_card_html(_status(child, settings={"frozen_parts": "none"}), operator=False)
    assert '<span class="shep-ft-k">Graph backbone</span> Trained</p>' in stated
    # A status without the settings (nothing sent) reads as the reference.
    bare = _status(child)
    del bare["finetune"]["settings"]
    assert "Trained (reference)" in fp.job_card_html(bare, operator=False)
    # Whatever the status holds is escaped.
    odd = fp.job_card_html(_status(child, settings={"frozen_parts": TAG}), operator=False)
    assert "<img" not in odd and "&lt;img src=x onerror=alert(1)&gt;" in odd


def test_the_cards_run_is_its_name_part_with_the_whole_id_in_its_title():
    """K6 closing smoke: the Run cell cut the id at 22ch ("gatv2/20261028-14001…"),
    hiding the run's name, which only a hover showed. The cell shows the name
    part (between the start time and the 6-hex id), the whole id in its title;
    a name is at most 40 characters (`slugify_run_name`), the cell's cap."""
    run = "20261028-140013_smoke-ft-patients-like-me_0e8106"
    row = fp.child_row_html(_child("patients_like_me", "completed", run), operator=False,
                            products={})
    assert (f'<code class="shep-ident" title="gatv2/{run}">smoke-ft-patients-like-me</code>'
            in row)
    assert "max-width: 22ch" not in fp.PAGE_CSS
    assert f"{fp._PAGE} .shep-ft-table code.shep-ident {{ max-width: 40ch; }}" in fp.PAGE_CSS
    # A directory not named by its start time is its whole id.
    assert ">gatv2/odd-run</code>" in fp.child_row_html(
        _child("disease", "completed", "odd-run"), operator=False, products={})


def test_the_best_save_is_the_runs_recorded_selection_never_a_file_name(root, monkeypatch):
    """The products of an ended child, for an operator: its run's best saves
    by the selection its files record (a task model's validation pooled MRR,
    higher is better), ties shared, read from the list's metadata (the file
    names here say otherwise), named by that metric with its direction
    under them; a min-mode monitor's lowest; an older run that records no
    selection says so, never a best."""
    from src.api.services import server_lists

    def item(filename, value, epochs, *, mode="max", selection=True, top=True):
        return server_lists.CheckpointItem(
            workspace="demo", architecture="gatv2", run="20261028-1_ft-disease_a",
            filename=filename, sha256="c" * 64, size_bytes=1, checkpoint_kind="training_state",
            epoch=epochs - 1, validation={"metric": "val_pooled_mrr", "value": value},
            dataset_fingerprint=None, trained_on_this_dataset=None, problem=None,
            completed_epochs=epochs, record="model", learning_stage="finetune", task="disease",
            selection={"monitor": "val_pooled_mrr", "mode": mode} if selection else None,
            in_best_k=top if selection else None)

    def listed(**kwargs):
        return server_lists.Listing((item("last.pt", 0.40, 5, top=False, **kwargs),
                                     item("model-04-0.9000.pt", 0.40, 4, **kwargs),
                                     item("model-02-0.9500.pt", 0.25, 2, **kwargs)))

    seen = []
    listing = [listed()]
    monkeypatch.setattr(server_lists, "list_checkpoints",
                        lambda *a, **k: seen.append((a, k)) or listing[0])
    monkeypatch.setattr(tc, "workspace_of_run", lambda run_dir: "demo")
    child = _child("disease", "completed", "20261028-1_ft-disease_a")
    products = fp.products_of(_status(child)["finetune"], {})
    assert seen == [(("demo", "gatv2"), {"task": "disease", "run": "20261028-1_ft-disease_a"})]
    (found,) = products.values()
    assert [entry["words"] for entry in found["best"]] == ["Latest save · 5 epochs completed",
                                                           "Best pooled MRR · 4 epochs completed"]
    html = fp.job_card_html(_status(child), operator=True, products=products)
    assert "Latest save · 5 epochs completed" in html and "Best pooled MRR · 4 epochs" in html
    assert ("Selection by validation pooled MRR (Trainer), higher is better" in html)
    assert "0.4000" in html and "Validation pooled MRR (Trainer)" in html
    assert "LP" not in re.sub(r"<[^>]+>", "", html), "a task model's figure is never LP's"
    assert "test" not in re.sub(r"<[^>]+>", "", html).lower().replace("latest", "")
    # Read once per child ending: the session's record is taken next time.
    assert fp.products_of(_status(child)["finetune"], products) == products and len(seen) == 1
    # A running child's products are not read.
    fp.products_of(_status(_child("disease", "running", "r"))["finetune"], {})
    assert len(seen) == 1
    # A min-mode monitor: the lowest is best, whatever the names say.
    listing[0] = listed(mode="min")
    (low,) = fp.products_of(_status(child)["finetune"], {}).values()
    assert [entry["words"] for entry in low["best"]] == ["Best pooled MRR · 2 epochs completed"]
    assert low["best"][0]["selection"].endswith("lower is better")
    # An older run whose files record no selection: no best, said so.
    listing[0] = listed(selection=False)
    older = fp.products_of(_status(child)["finetune"], {})
    (none,) = older.values()
    assert none["best"] == [] and none["recorded"] is False
    html = fp.job_card_html(_status(child), operator=True, products=older)
    assert "selection not recorded in its saved files" in html


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
    # No reason beside an enabled Start (K6 closing smoke: the line beside it
    # spoke of Stop, which is in the job panel; the card says no job runs).
    assert (idle["start"], idle["stop"]) == (True, False) and idle["reason"] == ""
    assert idle["stop_reason"] == ""
    refused = fp.actions_view({"status": "idle"}, access=_Access(False, "deployment_mode"),
                              gpu=None, confirming=False)
    assert (refused["start"], refused["stop"]) == (False, False)
    assert "Operator controls only" in refused["reason"] and "shep-lock" in refused["reason"]
    # Stop is in the job panel, apart from Start: refused, it has the reason
    # beside it too (the lock is never without it, skill §3).
    assert refused["stop_reason"] == refused["reason"]
    running = fp.actions_view(_status(child), access=_Access(True), gpu=None, confirming=False)
    assert (running["start"], running["stop"]) == (False, True) and running["stop_reason"] == ""
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


def test_start_and_stop_are_labelled_by_the_one_gated_rule(monkeypatch):
    """`_card_view` labels Start and Stop by `chips.gated` from this
    request's D8 answer: no lock while allowed; refused, disabled with the
    lock and D8's short reason beside each (Start's under the form, Stop's
    in the job panel)."""
    status = _status(_child("disease", "running", "r"))
    monkeypatch.setattr(fp.training_manager, "get_status", lambda: status)
    operator_mode(monkeypatch)
    card, start, stop, reason, stop_reason, confirm, _cache = fp._card_view({}, {},
                                                                            gradio_request())
    assert start["value"] == fp.START_LABEL and "🔒" not in start["value"]
    assert stop["value"] == fp.STOP_LABEL and stop["interactive"] is True
    assert "Operator controls only" not in reason + stop_reason
    monkeypatch.delenv("SHEPHERD_DEPLOYMENT_MODE")
    card, start, stop, reason, stop_reason, confirm, _cache = fp._card_view({}, {},
                                                                            gradio_request())
    assert (start["value"], start["interactive"]) == (f"🔒 {fp.START_LABEL}", False)
    assert (stop["value"], stop["interactive"]) == (f"🔒 {fp.STOP_LABEL}", False)
    for line in (reason, stop_reason):
        assert "Operator controls only: deployment mode not set to M1" in line, line


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


def test_the_cards_stop_never_reaches_another_pages_job(monkeypatch):
    """The card's Stop reaches a running Fine-tune child only. A base run
    started on Research › Training (it clears the Fine-tune status) is not
    asked about, and a confirmation, left from an earlier question or sent
    directly, never stops it."""
    operator_mode(monkeypatch)
    base = {"status": "running", "job_id": "base-job-9", "finetune": None}
    monkeypatch.setattr(tc.training_manager, "get_status", lambda: base)
    monkeypatch.setattr(tc.training_manager, "stop_training",
                        lambda: pytest.fail("the card's Stop stopped another page's job"))
    text, mode = fp._on_finetune_stop_request(gradio_request(), {})
    assert mode["confirm"] is None and "Nothing to stop: no Fine-tune child is running." in text
    assert not tc.confirming(base, mode)
    for asked in ({"confirm": "j2"}, {"confirm": "base-job-9"}, {}):
        notice, mode = fp._on_finetune_stop(gradio_request(), asked)
        assert "Not stopped" in notice and mode["confirm"] is None, asked
    assert fp.running_child(base) is None
    assert fp.running_child(_status(_child("disease", "running", "r"))) == "j2"
    # A child the status still lists while another job runs is not reached.
    stale = {**_status(_child("disease", "running", "r")), "job_id": "base-job-9"}
    assert fp.running_child(stale) is None
    assert fp.actions_view(stale, access=_Access(True), gpu=None, confirming=False)["stop"] is False


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
    assert fp._poll("training", {}, {}, gradio_request()) == (gr.update(),) * 7
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
                                  "ft_num_epochs", "ft_seed", "ft_frozen_parts", "ft_device",
                                  "ft_run_name"}
    # Every field the page offers, each the one field of its setting at the
    # value the run takes without it (`built_values`: the recipe's finetune
    # values, else this build's defaults); the freeze's options the field
    # spec's values, shown by their words.
    from src.config.recipe import FINETUNE
    from src.config.training_fields import by_name

    assert fp.FINETUNE_FIELDS == fp.COMMON_SHOWN + fp.MORE_FIELDS + fp.TASK_FIELDS
    assert {f"ft_{name}" for name in fp.FINETUNE_FIELDS} <= set(by_id)
    assert {f"ft_{name}" for name in fp.FINETUNE_FIELDS} <= set(fp.FIELD_HELP)
    assert (by_id["ft_num_epochs"].value, by_id["ft_seed"].value) == (
        FINETUNE["num_epochs"], FINETUNE["seed"])
    assert by_id["ft_num_epochs"].precision == 0 and by_id["ft_seed"].precision == 0
    assert by_id["ft_early_stopping_patience"].value == "off"
    assert by_id["ft_num_neighbors"].value == "15, 10, 5"
    frozen = by_id["ft_frozen_parts"]
    assert isinstance(frozen, gr.Radio) and frozen.value == FINETUNE["frozen_parts"] == "none"
    assert frozen.choices == [(r.setting_words("frozen_parts", value), value)
                              for value in by_name("frozen_parts").choices]
    assert [words for words, _ in frozen.choices] == ["Trained", "Frozen"]
    # The common settings in view without opening anything, the hop list in
    # the row under them beside its note (offered for any task set); the rest
    # under "More settings"; a task's own settings in their own section,
    # hidden until one task is chosen; nothing of the base's.
    def parents_of(block):
        out, seen = [], block
        while getattr(seen, "parent", None) is not None:
            seen = seen.parent
            out.append(seen)
        return out

    def accordion(name):
        found = [b for b in parents_of(by_id[f"ft_{name}"]) if isinstance(b, gr.Accordion)]
        return found[0].elem_id if found else None

    assert [accordion(name) for name in fp.COMMON_SHOWN] == [None] * 3
    assert {accordion(name) for name in fp.MORE_FIELDS} == {"ft_more"}
    assert {accordion(name) for name in fp.TASK_OWN_FIELDS} == {"ft_task_more"}
    assert [accordion(name) for name in fp.ANY_TASKS] == [None]
    assert by_id["ft_more"].label == "More settings" and by_id["ft_more"].open is False
    assert by_id["ft_task_more"].visible is False
    def row(elem_id):
        return next(b for b in parents_of(by_id[elem_id]) if isinstance(b, gr.Row))

    assert row("ft_hops_note") is row("ft_num_neighbors") is by_id["ft_hops_row"]
    assert by_id["ft_hops_row"].visible is False                     # until a task is chosen
    # Its row directly under the common settings' (and the freeze note).
    form = list(row("ft_num_epochs").parent.children)
    assert form.index(by_id["ft_hops_row"]) == form.index(row("ft_num_epochs")) + 2
    assert form[form.index(row("ft_num_epochs")) + 1] is by_id["ft_freeze_note"]
    # The base form's one line on what the operator set, beside Reset.
    assert row("ft_setby") is row("ft_reset")
    assert by_id["ft_setby"].value == fp.set_by_html(
        (), dict(zip(fp.FINETUNE_FIELDS, fp.start_values(()), strict=True)), [])
    assert by_id["ft_reset"].value == "Reset to reference"
    for gone in ("ft_split_seed", "ft_model_spec"):
        assert gone not in by_id
    # The dataset with its identity beside it, and Run → Checkpoint side by
    # side with the list's Refresh, so the common settings come into view at
    # the deployment window (each row wraps on a narrow form).
    assert row("ft_identity") is row("ft_dataset")
    assert row("ft_run") is row("ft_checkpoint") is row("ft_refresh")
    # Start, its reason and its refusal on one row; Stop's notice beside Stop.
    assert row("ft_notice") is row("ft_start") is row("ft_reason")
    assert row("ft_stop_notice") is row("ft_stop")
    # The note starts empty (Trained) and is hidden while empty; the field's
    # change redraws it; Start takes the fields in `FINETUNE_FIELDS` order.
    assert by_id["ft_freeze_note"].value == ""
    assert f"{fp._PAGE} #ft_freeze_note:not(:has(p))" in fp.PAGE_CSS
    fns = list(demo.fns.values())
    (note,) = [fn for fn in fns if fn.fn is fp.freeze_note_html]
    assert note.targets == [(frozen._id, "change")]
    assert note.inputs == [frozen] and note.outputs == [by_id["ft_freeze_note"]]
    (start,) = [fn for fn in fns if fn.fn is fp._on_finetune_start]
    assert start.inputs[-len(fp.FINETUNE_FIELDS):] == [
        by_id[f"ft_{name}"] for name in fp.FINETUNE_FIELDS]
    assert start.outputs == [by_id["ft_notice"], by_id["ft_stop_notice"]]
    # Explicit is what the operator touched (decision S2): each field's input
    # (never its change) adds it to the form's record in the browser (the
    # shared `record_touches`, wired before any other input listener), which
    # Start's own request carries.
    touched = start.inputs[-len(fp.FINETUNE_FIELDS) - 1]
    assert touched is by_id["ft_touched"] and isinstance(touched, gr.JSON)
    assert touched.visible is False and touched.value == []
    fields = [by_id[f"ft_{name}"] for name in fp.FINETUNE_FIELDS]
    setby = by_id["ft_setby"]
    parents_state = start.inputs[1]
    for name in fp.FINETUNE_FIELDS:
        field = by_id[f"ft_{name}"]
        listeners = [fn for fn in fns if (field._id, "input") in fn.targets]
        # The record first, then one listener: two that each send the last
        # value ("always_last") retrigger each other in the browser.
        first, line = listeners
        assert first.fn is None and first.inputs == [touched] == first.outputs
        assert first.js == tc.TOUCH_JS.replace("{key}", json.dumps(name))
        assert first.queue is False
        assert line.queue is False and line.trigger_mode == "always_last"
        if name in fp.ANY_TASKS:
            # The hop list's: the line, its note and the note's pointer.
            assert line.fn is fp._on_hops_input
            assert line.inputs == [by_id["ft_tasks"], by_id["ft_checkpoint"], parents_state,
                                   touched, *fields]
            assert line.outputs == [setby, by_id["ft_hops_note"], by_id["ft_hops_warn"]]
            continue
        # Then the line, as the base form's fields say theirs.
        assert line.outputs == [setby] and line.inputs == [by_id["ft_tasks"], touched, *fields]
        assert line.fn(["gene"], ["x"], *fp.start_values(["gene"])) == fp.set_by_html(
            ["gene"], dict(zip(fp.FINETUNE_FIELDS, fp.start_values(["gene"]), strict=True)),
            sorted({"x", name}))
    # No record on the server: nothing a field's or the tasks' input answers
    # writes it, and no State either.
    tasks_id = by_id["ft_tasks"]._id
    inputs = [fn for fn in fns if any(t[1] == "input" and t[0] in {f._id for f in fields}
                                      | {tasks_id} for t in fn.targets)]
    assert not [fn for fn in inputs if fn.fn is not None and (
        touched in fn.outputs or any(isinstance(b, gr.State) for b in fn.outputs))]
    # The tasks: the operator's change first drops a task's own settings
    # from the record, in the browser (`record_drops`), then starts them
    # again and says the line; any change redraws the section.
    drop, restart = [fn for fn in fns if (tasks_id, "input") in fn.targets]
    assert drop.fn is None and drop.inputs == [touched] == drop.outputs
    assert drop.js == tc.DROP_JS.replace("{keys}", json.dumps(sorted(fp.TASK_OWN_FIELDS)))
    assert drop.queue is False and drop.trigger_mode == "multiple"
    assert restart.fn is fp._on_tasks_input
    assert restart.inputs == [by_id["ft_tasks"], touched, *fields]
    assert restart.outputs == [*(by_id[f"ft_{name}"] for name in fp.TASK_FIELDS), setby]
    views = [fn for fn in fns if fn.fn is fp._on_task_view]
    assert any((tasks_id, "change") in fn.targets for fn in views)
    assert all(fn.outputs[:2] == [by_id["ft_task_more"], by_id["ft_hops_row"]]
               for fn in views)
    # The hop note's pointer beside Start, said with the note (review of B2).
    assert row("ft_hops_warn") is row("ft_start")
    assert all(fn.outputs[-1] is by_id["ft_hops_warn"] for fn in views)
    (hop_input,) = [fn for fn in fns if fn.fn is fp._on_hops_input]
    assert hop_input.outputs[1:] == [by_id["ft_hops_note"], by_id["ft_hops_warn"]]
    # Reset to reference: the record, the line and every field, then the
    # hop note again for the tasks.
    (reset,) = [fn for fn in fns if fn.fn is fp._on_finetune_reset]
    assert reset.targets == [(by_id["ft_reset"]._id, "click")]
    assert reset.inputs == [by_id["ft_tasks"]] and reset.queue is False
    assert reset.outputs == [touched, setby, *fields]
    assert any(fn.fn is fp._on_task_view and (by_id["ft_reset"]._id, "click") not in fn.targets
               and fn.trigger_after == reset._id for fn in fns)
    (guards,) = [fn for fn in fns if fn.fn is fp._on_guards]
    assert {block.elem_id for block in guards.outputs} == {
        "ft_warmup_steps", "ft_min_lr_ratio", "ft_amp_dtype"}
