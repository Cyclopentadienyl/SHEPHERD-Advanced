"""
The Training page's base form and its Start (ops-wiring round, decisions S1, S2, S3, S5).
=========================================================================================
`src/webui/components/training_console.py`. The form offers what a new base
takes (`run_plan.new_run_settings("base")`) and the reference base spec's
fields; what these pin:

- **explicit is what the operator touched** (decision S2): Start sends
  ``learning_stage="base"``, the run keys (the workspace the server resolves
  from its list, the device, the run name stripped, compile from the runtime
  settings) and each setting whose field the operator touched, whatever its
  value (the reference's own value typed is sent); an untouched one is not;
  Reset to reference clears the record; the line under the form says what is
  set, with the value the run takes without it;
- **the form never offers or sends a setting the run would not read**
  (decision S3): warmup only with cosine, the minimum ratio only with a
  scheduler, the dtype only with mixed precision on;
- **the model spec** (decision S5): the architecture from the encoder
  declarations; the spec's fields from its sections' dataclasses; the spec
  sent whole only when a field of it was touched, refused inline by its own
  validation, and the hop list checked against the encoder it builds;
- **a value its field spec refuses starts nothing**, and the notice names it
  by its label, inline (no toast); every kind is parsed as its setting takes
  it; the help leads are the field spec's descriptions and the formats are
  visible text;
- D8, the dataset and the runner's own refusal, as before.

Module: tests/unit/test_training_console_config.py
"""
from __future__ import annotations

import dataclasses

import pytest

# training_console imports gradio (and the api/service stack) at module load.
gr = pytest.importorskip("gradio")

from src.config import training_fields as tf  # noqa: E402
from src.config.recipe import BASE, BASE_SOURCES  # noqa: E402
from src.models.bundle import spec as spec_module  # noqa: E402
from src.utils.run_plan import new_run_settings, new_run_values  # noqa: E402
from src.webui import shell  # noqa: E402
from src.webui.components import renderers as r  # noqa: E402
from src.webui.components import training_console as tc  # noqa: E402
from tests.fixtures.operator_access import gradio_request, operator_mode  # noqa: E402

#: A request D8 allows once the deployment is in mode M1 (`_operator_mode`):
#: Start is an operator control since M5-1.
OP = gradio_request()

#: The run bar's workspace as a session holds it (`workspace_state`): the
#: Start handler resolves the name through the server's list (here replaced)
#: and sends the full fingerprint shown.
FP = "a" * 64
WS = {"name": "demo", "row": {"name": "demo", "dataset_fingerprint": FP, "trainable": True}}
#: The run keys every start sends.
RUN_KEYS = {"data_dir", "learning_stage", "device", "run_name", "compile", "__kwargs__"}


@pytest.fixture(autouse=True)
def _operator_mode(monkeypatch):
    operator_mode(monkeypatch)


@pytest.fixture
def listed(monkeypatch, tmp_path):
    """The server's list resolves "demo" to a directory (`_workspace_path`)."""
    monkeypatch.setattr(tc, "_workspace_path", lambda name: tmp_path / name)
    monkeypatch.setattr(tc, "load_runtime_settings", lambda *a, **k: {})
    return tmp_path / "demo"


@pytest.fixture
def started(monkeypatch):
    """A recorder in place of the job runner's start."""
    calls = []

    def start(config, **kwargs):
        calls.append({**config, "__kwargs__": kwargs})
        return {"success": True, "pid": 1}

    monkeypatch.setattr(tc.training_manager, "start_training", start)
    return calls


def _values(**changed):
    """The base form's values by key (`base_form_keys`): the starting ones, but
    those named (a spec field by its path, with "." written as "__")."""
    values = dict(zip(tc.base_form_keys(), tc.base_form_defaults(), strict=True))
    values.update({key.replace("__", "."): value for key, value in changed.items()})
    return values


def _form(**changed):
    """The base form's values in `base_form_keys` order (`_values`)."""
    values = _values(**changed)
    return tuple(values[key] for key in tc.base_form_keys())


def _start(touched=(), device="cpu", run_name="", **changed):
    """Start as the browser sends it: the touched record and every field."""
    return tc._on_start(OP, WS, run_name, device, list(touched), *_form(**changed))


def _sent(call):
    return {key: value for key, value in call.items() if key not in RUN_KEYS}


# --------------------------------------------------------------------------- what the form offers
def test_the_form_offers_what_a_new_base_takes():
    """Decision S1: the form's settings and its model spec are what a new base
    takes (`run_plan.new_run_settings("base")`), never a page's own list: the
    presentation order holds every one of them once, the common ones first."""
    assert set(tc.BASE_FIELDS) | {"model_spec"} == new_run_settings("base")
    assert len(set(tc.BASE_FIELDS)) == len(tc.BASE_FIELDS)
    assert tc.BASE_FIELDS[:tc.BASE_SHOWN] == ("num_epochs", "learning_rate", "batch_size",
                                              "seed", "split_seed")
    # Each starts at the value the run takes without it: the one fill of a new run.
    references, sources = tc.base_references()
    assert references == new_run_values("base", None, {})
    assert {name for name in tc.BASE_FIELDS if name in sources} == set(BASE)
    # Each labelled once, short enough for a column with its ⓘ (`SETTING_LABELS`).
    for name in tc.BASE_FIELDS:
        assert len(r.SETTING_LABELS[name]) <= 25, name
    for field in tc.spec_form_fields():
        assert len(r.field_words(field.path.split(".", 1)[1])) <= 24, field.path


def test_the_spec_fields_are_the_sections_declared_fields():
    """Decision S5: the Model accordion's fields are generated from the input,
    encoder and LP head dataclasses (`spec.declared_fields`): numbers and
    switches to edit, an optional number (the relation edge dimension) off or
    a number, texts as facts; the encoder's kind is the architecture."""
    reference = spec_module.reference_spec("base")
    expected = [f"{section}.{name}" for section in tc.SPEC_SECTIONS
                for name, _type in spec_module.declared_fields(type(getattr(reference, section)))
                if f"{section}.{name}" != tc.ARCHITECTURE]
    fields = {field.path: field for field in tc.spec_form_fields()}
    assert list(fields) == expected
    facts = {path for path, field in fields.items() if not field.editable}
    assert facts == {"input.kind", "input.init", "encoder.norm", "lp_head.kind",
                     "lp_head.relation_init"}
    assert all(isinstance(field.reference, str) for path, field in fields.items()
               if path in facts)
    assert fields["encoder.relation_edge_dim"].kind == "int"
    assert fields["encoder.relation_edge_dim"].reference is None
    assert (fields["encoder.layers"].kind, fields["encoder.dropout"].kind,
            fields["encoder.concat"].kind) == ("int", "float", "bool")
    for path, field in fields.items():
        section, name = path.split(".")
        assert field.reference == getattr(getattr(reference, section), name)
    # The facts' summary list names each text and the architecture.
    facts_html = tc.spec_facts_html()
    assert "Architecture" in facts_html and "GATv2" in facts_html
    assert "Encoder › Normalisation" in facts_html and "graph_layer_norm+batch_norm" in facts_html


def test_the_architecture_options_are_the_encoder_declarations(monkeypatch):
    """Decision S5: the options are the encoder kinds this build implements
    (`ENCODER_DECLARATIONS`), never `ENCODER_KINDS`' names; with one it is a
    fact, with two or more a choice the start sends as the spec's kind."""
    from src.config.model_types import ENCODER_KINDS

    assert [kind for _words, kind in tc.architecture_choices()] == list(
        spec_module.ENCODER_DECLARATIONS)
    assert set(ENCODER_KINDS) - set(spec_module.ENCODER_DECLARATIONS)  # names never offered
    assert tc.ARCHITECTURE not in tc.base_form_keys()
    two = {**spec_module.ENCODER_DECLARATIONS, "sage": spec_module.GATv2Declaration()}
    monkeypatch.setattr(tc, "ENCODER_DECLARATIONS", two)
    assert [kind for _words, kind in tc.architecture_choices()] == ["gatv2", "sage"]
    assert tc.base_form_keys()[len(tc.BASE_FIELDS)] == tc.ARCHITECTURE
    with gr.Blocks() as demo:
        tc.create_training_tab(shell.standalone_context(demo))
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    choice = by_id["spec-encoder-kind"]
    assert isinstance(choice, gr.Radio) and choice.choices == tc.architecture_choices()
    assert "Architecture" not in tc.spec_facts_html()
    # One name on the form: the radio's label is the line's (and a refusal's).
    assert choice.label == tc.spec_key_words(tc.ARCHITECTURE) == "Architecture"
    line = tc.set_by_you_html(_values(encoder__kind="gatv2"), [tc.ARCHITECTURE])
    assert "Set by you: Architecture GATv2 (reference GATv2)" in line
    assert "Kind" not in line


# --------------------------------------------------------------------------- what Start sends
def test_start_sends_what_the_operator_touched_whatever_its_value(listed, started):
    """Decision S2: an untouched field is not sent (the run fills it and
    records it so); a touched one is, each as its setting takes it, and the
    reference's own value typed is sent too: equal to the reference is never
    read as "not set"."""
    assert _start(run_name="  Pilot Run  ") == ""
    assert started[-1] == {"data_dir": str(listed), "learning_stage": "base", "device": "cpu",
                           "run_name": "Pilot Run", "compile": False,
                           "__kwargs__": {"expected_dataset_fingerprint": FP}}
    # Changed but not touched (a page's write): not sent.
    _start(num_epochs=5)
    assert _sent(started[-1]) == {}
    # Touched, each as its setting takes it (a whole float of an integer box as
    # an int, the hop text as a list, an int of a float field as a float).
    _start(["num_epochs", "num_neighbors", "learning_rate", "use_amp",
            "early_stopping_patience", "split_seed"], device="auto",
           num_epochs=5.0, num_neighbors="20, 10, -1", learning_rate=1, use_amp=True,
           early_stopping_patience="7", split_seed=3)
    sent = _sent(started[-1])
    assert sent == {"num_epochs": 5, "num_neighbors": [20, 10, -1], "learning_rate": 1.0,
                    "use_amp": True, "early_stopping_patience": 7, "split_seed": 3}
    assert type(sent["num_epochs"]) is int and type(sent["learning_rate"]) is float
    # The reference's own values, typed: sent, and recorded as the operator's.
    _start(["num_epochs", "learning_rate", "num_neighbors", "max_grad_norm", "split_seed"],
           num_epochs=200, learning_rate=0.0001, num_neighbors="15,10,5", max_grad_norm=1,
           split_seed=0)
    assert _sent(started[-1]) == {"num_epochs": 200, "learning_rate": 0.0001,
                                  "num_neighbors": [15, 10, 5], "max_grad_norm": 1.0,
                                  "split_seed": 0}


def test_start_takes_compile_from_the_runtime_settings(listed, started, monkeypatch):
    monkeypatch.setattr(tc, "load_runtime_settings", lambda *a, **k: {"torch_compile": True})
    _start()
    assert started[-1]["compile"] is True


def test_each_fields_input_says_the_line_and_reset_clears_the_record():
    """Decision S2: a field's input (never a page's write) adds it to the
    form's record, in the browser (`TOUCH_JS`), and says the line from the
    record with it; Reset to reference clears the record, writes every
    field's starting value and says the line again."""
    listener = tc._base_input_listener("num_epochs")
    line = listener(["seed"], *_form(num_epochs=5))
    assert line == tc.set_by_you_html(_values(num_epochs=5), ["num_epochs", "seed"])
    assert "Epochs 5 (reference 200)" in line and "Seed 1 (reference 1)" in line
    # The record already holding the key (the browser added it first): the same line.
    assert listener(["num_epochs", "seed"], *_form(num_epochs=5)) == line
    assert tc.touched_with(None, "encoder.layers") == ["encoder.layers"]
    assert tc.touched_with(["a"], "a") == ["a"]
    reset = tc._on_base_reset()
    assert reset[0] == [] and reset[1] == tc.set_by_you_html(_values(), [])
    assert "Set by you: nothing" in reset[1]
    assert [update["value"] for update in reset[2:]] == list(tc.base_form_defaults())
    assert len(reset[2:]) == len(tc.base_form_keys())


def test_a_setting_leading_a_clause_is_named_by_its_label():
    """Review of B2: a check's sentence that names a setting first, at the
    text's start, after another sentence or after the colon that leads a
    clause (the runner's check of a Fine-tune child against its base), is
    said by the field's label (`labelled_problem`), on the Fine-tune and
    Branch pages alike; a setting named inside a clause stays as written."""
    why = ("The Disease child's hop list [15, 10, 5] (its task's reference) does not fit its "
           "base's encoder: num_neighbors needs 2 numbers, one hop per encoder layer, not 3.")
    assert tc.labelled_problem(why) == why.replace(": num_neighbors needs",
                                                   ": Neighbours per hop needs")
    assert tc.labelled_problem("min_lr_ratio must be more than 0. num_epochs must be set.") == (
        "Final learning-rate ratio must be more than 0. Epochs must be set.")
    inside = "warmup_steps is read only when scheduler_type is cosine."
    assert tc.labelled_problem(inside) == "Warmup steps is read only when scheduler_type is cosine."


def test_the_record_is_read_as_data_from_the_request(listed, started):
    """The record is the browser's (`touched_record`), carried by Start's own
    request: a list's strings are its keys; anything else is no key, so a
    request the page did not send cannot make a substring or a mapping
    "touched"."""
    assert tc.touched_keys(["num_epochs", "seed", 3, None]) == {"num_epochs", "seed"}
    for odd in (None, "num_epochs, seed", {"num_epochs": True}, 5):
        assert tc.touched_keys(odd) == set()
        assert tc._on_start(OP, WS, "", "cpu", odd, *_form(num_epochs=5, seed=3)) == ""
        assert _sent(started[-1]) == {}
    assert tc.touched_with("num_epochs", "seed") == ["seed"]


def test_the_touch_script_adds_the_key_once_in_the_browser():
    """`TOUCH_JS` (a field's input, run in the browser before Start's click is
    sent): the record's keys and the key, each once, in order, returned as
    the listener's one output; a record that is not a list starts empty."""
    import json
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    script = tc.TOUCH_JS.replace("{key}", json.dumps("num_epochs"))
    program = (f"const f = {script};"
               "console.log(JSON.stringify([f(['seed']), f(['num_epochs', 'a']), f(null),"
               " f('x')]));")
    result = subprocess.run([node, "-e", program], capture_output=True, text=True, timeout=30,
                            check=True)
    assert json.loads(result.stdout) == [[["num_epochs", "seed"]], [["a", "num_epochs"]],
                                         [["num_epochs"]], [["num_epochs"]]]


def test_the_line_says_what_the_operator_set_with_the_value_the_run_takes_without_it():
    """'Set by you: Epochs 5 (reference 200)': each setting and spec field
    Start sends, with the reference's value, or this build's default where
    the recipe sets none; a value equal to the reference is still listed."""
    assert tc.set_by_you_html(_values(), []) == (
        '<div class="shep-setby"><p>Set by you: nothing; the run takes the reference '
        "values and this build&#x27;s defaults.</p></div>")
    line = tc.set_by_you_html(
        _values(num_epochs=5, split_seed=0, num_neighbors="20, 10, 5"),
        ["num_epochs", "num_neighbors", "split_seed"])
    assert ("<p>Set by you: Epochs 5 (reference 200), Split seed 0 (default 0), Neighbours per "
            "hop 20, 10, 5 (reference 15, 10, 5)</p>" in line)
    assert "Details" not in line and "Not valid" not in line
    # The relation edge dimension's None is "off", the field's own word, and
    # its reference's too (never "not set").
    off = tc.set_by_you_html(_values(encoder__relation_edge_dim="off"),
                             ["encoder.relation_edge_dim"])
    assert "Encoder › Relation edge dimension off (reference off)" in off
    assert "not set" not in off
    assert "Relation edge dimension 8 (reference off)" in tc.set_by_you_html(
        _values(encoder__relation_edge_dim="8"), ["encoder.relation_edge_dim"])


def test_the_line_names_the_first_few_and_every_one_is_in_its_details():
    """The skill's §4: past two lines a text gets structure. The line names
    the first `SET_BY_SHOWN` in the form's order and counts the rest
    (`named_words`); every one is a row of its Details (`details_list`),
    whose key is the label and whose value says the reference."""
    touched = ["num_epochs", "use_amp", "early_stopping_patience", "num_neighbors",
               "split_seed", "encoder.hidden", "weight_decay"]
    line = tc.set_by_you_html(
        _values(num_epochs=5, use_amp=True, early_stopping_patience="10",
                num_neighbors="20, 10, 5", split_seed=0, encoder__hidden=64,
                weight_decay=0.01), touched)
    assert tc.SET_BY_SHOWN == 3
    assert ("<p>Set by you: Epochs 5 (reference 200), Split seed 0 (default 0), Neighbours per "
            "hop 20, 10, 5 (reference 15, 10, 5) and 4 more</p>" in line)
    assert '<details class="shep-more"><summary>Details ›</summary>' in line
    details = line[line.index("<details"):line.index("</details>")]
    rows = [("Epochs", "5 (reference 200)"), ("Split seed", "0 (default 0)"),
            ("Neighbours per hop", "20, 10, 5 (reference 15, 10, 5)"),
            ("Weight decay", "0.01 (reference 0.0)"),
            ("Mixed precision", "on (reference off)"),
            ("Early stopping patience", "10 (reference off)"),
            ("Encoder › Hidden", "64 (reference 256)")]
    assert details == r.details_list(rows, classes="shep-setby-kv").removesuffix("</details>")
    assert "Not valid" not in line


@pytest.mark.parametrize("name, value, words", [
    ("num_epochs", None, "Epochs must be set"),
    ("num_epochs", 0, "Epochs must be between 1 and 10000, not 0."),
    ("learning_rate", 0.0, "Learning rate must be more than 0 and at most 1, not 0.0."),
    ("batch_size", 4096, "Batch size must be between 1 and 2048, not 4096."),
    ("seed", 2 ** 32, "Seed must be between 0 and 4294967295, not 4294967296."),
    ("split_seed", -1, "Split seed must be 0 or more, not -1."),
    ("num_neighbors", "15, x, 5", "Neighbours per hop must be whole numbers"),
    ("num_neighbors", "", "Neighbours per hop needs at least 1"),
    ("num_neighbors", "15, 0, 5", "Neighbours per hop items must be 1 or more (or -1, every "
                                  "neighbour), not [15, 0, 5]."),
    # One hop per layer of the encoder the form's base builds (the reference
    # spec's three), refused here rather than by the training process.
    ("num_neighbors", "15, 10", "Neighbours per hop needs 3 numbers, one hop per encoder "
                                "layer, not 2"),
    ("num_neighbors", "15, 10, 5, 5", "Neighbours per hop needs 3 numbers"),
    ("weight_decay", -0.1, "Weight decay must be 0 or more, not -0.1."),
    ("max_grad_norm", 0.0, "Gradient clip must be more than 0, not 0.0."),
    ("gradient_accumulation_steps", 0, "Gradient accumulation must be 1 or more, not 0."),
    ("eval_every_n_epochs", 0, "Epochs per validation must be 1 or more, not 0."),
    ("save_top_k", 0, "Top-k saves kept must be 1 or more, not 0."),
    # What the browser sends for patience is text (`value_field`).
    ("early_stopping_patience", "0", "Early stopping patience must be 1 or more, not 0."),
    ("scheduler_type", "cyclic", "Scheduler must be one of"),
])
def test_a_value_its_field_spec_refuses_starts_nothing_and_is_named(listed, started, name,
                                                                    value, words):
    notice = _start([name], **{name: value})
    # A refusal, by the shared mark of one (B5: no longer plain amber text).
    assert started == [] and words in notice and notice.startswith(
        '<span class="shep-notice-t"><span class="shep-st shep-st-error">')
    assert '<span class="shep-st-w">Not started: ' in notice
    # The same words in the form's own line, before any Start.
    assert words in tc.set_by_you_html(_values(**{name: value}), [name])
    # Untouched, the same value is not the operator's: nothing is refused.
    assert "Not valid" not in tc.set_by_you_html(_values(**{name: value}), [])


def test_a_rule_across_settings_refuses_in_the_line_and_starts_nothing(listed, started):
    """The form's check is the start's: the rules across settings, on the
    run's settings as the start fills them (`cross_field_problems` on
    `new_run_values`, as `run_plan.new_run_problem` asks them), for what the
    form sends. Onecycle needs a minimum ratio above 0: named by its label in
    the line, before any Start, and Start starts nothing; the same ratio with
    cosine, or untouched under onecycle (the build's 0.01), is taken."""
    from src.api.services.training_manager import _new_bundle_run

    touched = ["scheduler_type", "min_lr_ratio"]
    values = _values(scheduler_type="onecycle", min_lr_ratio=0.0)
    words = "Final learning-rate ratio must be more than 0 when scheduler_type is 'onecycle', not 0.0"
    settings, problems = tc.base_start_settings(values, touched)
    assert [problem for problem in problems if problem.startswith(words)] == problems != []
    # The runner refuses what the form would have sent, by the same rule.
    assert "min_lr_ratio must be more than 0" in _new_bundle_run(
        {"learning_stage": "base", **settings})[1]
    assert f"Not valid: {r.e(words)}" in tc.set_by_you_html(values, touched)
    notice = _start(touched, scheduler_type="onecycle", min_lr_ratio=0.0)
    assert started == [] and r.e(words) in notice
    assert _start(touched, scheduler_type="cosine", min_lr_ratio=0.0) == ""
    assert _sent(started[-1]) == {"scheduler_type": "cosine", "min_lr_ratio": 0.0}
    assert _start(["scheduler_type"], scheduler_type="onecycle", min_lr_ratio=0.0) == ""
    assert _sent(started[-1]) == {"scheduler_type": "onecycle"}


def test_an_optional_setting_reads_off_and_what_the_browser_sends_is_the_recipes_none(
        listed, started):
    """K6 closing smoke: patience's reference is None (off). A number box showed
    it as 0 once opened, any later edit sent 0, and Start was refused. The field
    is a text box reading "off" (a state, not a placeholder), its format said
    under it; touched, "off" or empty is sent as None, the operator's explicit
    no early stopping (decision S2); a whole number is the int; anything else is
    named."""
    field = tc.setting_field("early_stopping_patience", BASE["early_stopping_patience"],
                             elem_id="p")
    assert BASE["early_stopping_patience"] is None
    assert isinstance(field, gr.Textbox) and field.value == "off"
    assert field.info == tc.FIELD_FORMATS["early_stopping_patience"]
    assert _values()["early_stopping_patience"] == "off"
    for typed in ("off", "", " Off "):
        settings, problems = tc.base_start_settings(
            _values(early_stopping_patience=typed), ["early_stopping_patience"])
        assert (settings, problems) == ({"early_stopping_patience": None}, [])
        assert _start(["early_stopping_patience"], early_stopping_patience=typed) == ""
        assert _sent(started[-1]) == {"early_stopping_patience": None}
    assert tc.base_start_settings(_values(early_stopping_patience=" 7 "),
                                  ["early_stopping_patience"]) == (
        {"early_stopping_patience": 7}, [])
    for typed in ("seven", "7.5"):
        assert tc.base_start_settings(_values(early_stopping_patience=typed),
                                      ["early_stopping_patience"])[1] == [
            'Early stopping patience must be a whole number or "off".']
    # The Fine-tune page's number fields (Epochs and Seed) have set references:
    # number boxes, as before.
    from src.config.recipe import FINETUNE

    for name in ("num_epochs", "seed"):
        assert isinstance(tc.setting_field(name, FINETUNE[name], elem_id=name), gr.Number)


def test_every_problem_is_listed_at_once_and_escaped(listed, started):
    notice = _start(["num_epochs", "num_neighbors"], num_epochs=None,
                    num_neighbors="<img src=x onerror=alert(1)>")
    assert started == [] and "Epochs must be set" in notice
    assert "Neighbours per hop must be whole numbers" in notice and "<img" not in notice
    line = tc.set_by_you_html(_values(scheduler_type="<b>x"), ["scheduler_type"])
    assert "<b>x" not in line and "&lt;b&gt;x" in line
    # A number's text where None is not taken: no "off" in its words.
    assert tc.base_start_settings(_values(encoder__dropout="x"), ["encoder.dropout"])[1] == [
        "Encoder › Dropout must be a number."]


# --------------------------------------------------------------------------- what the run would not read
def test_a_setting_the_run_would_not_read_is_neither_offered_nor_sent(listed, started):
    """Decision S3: warmup is read only with cosine, the minimum ratio only
    with a scheduler, the dtype only with mixed precision on. Hidden while its
    guard does not hold (the rule's own guard, `not_read_by_rules`), a touched
    one is not sent; once the guard holds it is shown and sent."""
    references, _ = tc.base_references()
    assert references["scheduler_type"] == "none" and references["use_amp"] is False
    dependent, guards = tc.rule_fields()
    assert dependent == ("warmup_steps", "min_lr_ratio", "amp_dtype")
    assert guards == ("scheduler_type", "use_amp")
    assert tf.not_read_by_rules(references) == set(dependent)
    visible = [update["visible"] for update in tc._on_base_guards("none", False)]
    assert visible == [False, False, False]
    assert [update["visible"] for update in tc._on_base_guards("cosine", True)] == [
        True, True, True]
    assert [update["visible"] for update in tc._on_base_guards("linear", False)] == [
        False, True, False]
    # Touched under scheduler none: not sent, not listed, never refused.
    touched = ["warmup_steps", "min_lr_ratio", "amp_dtype"]
    assert _start(touched, warmup_steps=500, min_lr_ratio=0.2, amp_dtype="bfloat16") == ""
    assert _sent(started[-1]) == {}
    assert "Set by you: nothing" in tc.set_by_you_html(_values(warmup_steps=500), touched)
    # Their guards set: shown and sent.
    _start([*touched, "scheduler_type", "use_amp"], scheduler_type="cosine", use_amp=True,
           warmup_steps=500, min_lr_ratio=0.2, amp_dtype="bfloat16")
    assert _sent(started[-1]) == {"scheduler_type": "cosine", "use_amp": True,
                                  "warmup_steps": 500, "min_lr_ratio": 0.2,
                                  "amp_dtype": "bfloat16"}
    # The guard's own value decides, touched or not (onecycle reads no warmup).
    assert tc.base_start_settings(_values(scheduler_type="onecycle", warmup_steps=5),
                                  ["warmup_steps"]) == ({}, [])


def test_the_runner_takes_what_the_form_sends(tmp_path, monkeypatch):
    """The form's start passes the runner's own check of a new base run
    (`_new_bundle_run`): its settings and its spec, the hops against it."""
    from src.api.services.training_manager import _new_bundle_run

    sent = []
    monkeypatch.setattr(tc, "_workspace_path", lambda name: tmp_path / name)
    monkeypatch.setattr(tc.training_manager, "start_training",
                        lambda config, **k: sent.append(config) or {"success": True})
    touched = ["num_epochs", "batch_size", "use_amp", "scheduler_type", "warmup_steps",
               "num_neighbors", "split_seed", "save_top_k", "encoder.layers",
               "encoder.relation_edge_dim"]
    assert _start(touched, num_epochs=3, batch_size=64, use_amp=True, scheduler_type="cosine",
                  warmup_steps=10, num_neighbors="20, -1", split_seed=4, save_top_k=1,
                  encoder__layers=2, encoder__relation_edge_dim="8") == ""
    assert _new_bundle_run(sent[-1]) == ("gatv2", None)
    assert sent[-1]["model_spec"]["encoder"]["layers"] == 2


# --------------------------------------------------------------------------- the model spec
def test_the_spec_is_sent_whole_only_when_a_field_of_it_was_touched(listed, started):
    """Decision S5: an untouched spec is not sent (the run builds the reference
    spec and records it so); a touched field sends the reference base spec
    with it replaced, whole, even at the reference's own value."""
    _start(encoder__layers=2)
    assert "model_spec" not in started[-1]
    _start(["encoder.hidden", "lp_head.margin", "input.type_embedding"],
           encoder__hidden=64, lp_head__margin=2, input__type_embedding=True)
    spec = spec_module.ModelSpec.from_dict(started[-1]["model_spec"])
    reference = spec_module.reference_spec("base")
    assert spec == dataclasses.replace(
        reference, encoder=dataclasses.replace(reference.encoder, hidden=64),
        lp_head=dataclasses.replace(reference.lp_head, margin=2.0),
        input=dataclasses.replace(reference.input, type_embedding=True))
    assert type(spec.lp_head.margin) is float
    _start(["encoder.layers"], encoder__layers=3)
    assert spec_module.ModelSpec.from_dict(started[-1]["model_spec"]) == reference


@pytest.mark.parametrize("changed, words", [
    ({"encoder__layers": 0}, "Encoder › Layers must be a positive integer (got 0)"),
    ({"encoder__dropout": 1.0}, "Encoder › Dropout must be a number in [0, 1)"),
    ({"encoder__concat": False}, "Encoder › Concatenate must be true"),
    ({"encoder__relation_edge_dim": "x"},
     'Encoder › Relation edge dimension must be a whole number or "off".'),
    ({"encoder__relation_edge_dim": "0"},
     "Encoder › Relation edge dimension must be a positive integer (got 0)"),
    ({"lp_head__margin": 0}, "LP head › Margin must be positive"),
    ({"input__dim": 2.5}, "Input › Dimension must be an integer, not 2.5"),
])
def test_a_spec_its_validation_refuses_starts_nothing_and_is_named(listed, started, changed,
                                                                   words):
    touched = [key.replace("__", ".") for key in changed]
    notice = _start(touched, **changed)
    assert started == [] and r.e(words) in notice
    assert r.e(words) in tc.set_by_you_html(_values(**changed), touched)


def test_the_hops_are_checked_against_the_encoder_the_run_builds(listed, started):
    """The hop list the run samples (the one sent, else the reference's) is
    checked against the edited encoder (`hops_problem`, the start's check)."""
    notice = _start(["encoder.layers"], encoder__layers=2)
    assert started == [] and ("Neighbours per hop needs 2 numbers, one hop per encoder layer, "
                              "not 3.") in notice
    assert _start(["encoder.layers", "num_neighbors"], encoder__layers=2,
                  num_neighbors="10, 5") == ""
    assert started[-1]["num_neighbors"] == [10, 5]
    assert started[-1]["model_spec"]["encoder"]["layers"] == 2
    # A refused hop list is named once, by its own check.
    notice = _start(["encoder.layers", "num_neighbors"], encoder__layers=2,
                    num_neighbors="10, x")
    assert notice.count("Neighbours per hop") == 1


# --------------------------------------------------------------------------- help and formats
def test_each_settings_help_leads_with_its_field_specs_description():
    """The help lead is the field spec's description (one text per setting),
    then the value the run takes without it: the reference's with its source,
    or this build's default where the recipe sets none."""
    references, sources = tc.base_references()
    for name in tc.BASE_FIELDS:
        title, text = tc.FIELD_HELP[name]
        lead = text.split("\n", 1)[0]
        assert title == r.SETTING_LABELS[name] and lead == tf.by_name(name).description
        assert len(lead) <= 90, (name, lead)
        if name in BASE:
            assert f"Reference: {r.setting_words(name, BASE[name])}." in text
            assert BASE_SOURCES[name] in text
        else:
            assert name not in sources
            assert (f"Default: {r.setting_words(name, references[name])}, this build's; the "
                    "reference recipe sets none.") in text
    assert not hasattr(tc, "_SETTING_LEADS")


def test_the_patience_lead_names_the_forms_word_for_none():
    """The lead is the field spec's description, shared with the API: it
    names the form's word for None ("off", the field's own) beside unset,
    never "none", which the field refuses."""
    lead = tc.FIELD_HELP["early_stopping_patience"][1].split("\n", 1)[0]
    assert f"{r.OFF.lower()} (unset) is no early stopping" in lead and "none" not in lead
    assert tc.base_start_settings(_values(early_stopping_patience=r.OFF.lower()),
                                  ["early_stopping_patience"]) == (
        {"early_stopping_patience": None}, [])


def test_a_fields_format_unit_or_off_is_visible_under_it():
    """The skill's §7: formats, units and "off" are visible text (``info=``),
    not only a bubble; a field with nothing to say has none."""
    references, _ = tc.base_references()
    for name in tc.BASE_FIELDS:
        field = tc.setting_field(name, references[name], elem_id=name)
        assert field.info == tc.FIELD_FORMATS.get(name), name
    assert set(tc.FIELD_FORMATS) <= set(tc.BASE_FIELDS) | {
        field.path for field in tc.spec_form_fields() if field.editable}
    edge = tc.value_field("int", "Relation edge dimension", None, elem_id="e",
                          info=tc.FIELD_FORMATS["encoder.relation_edge_dim"])
    assert isinstance(edge, gr.Textbox) and edge.value == "off" and '"off"' in edge.info
    # A bound is read from its root: the field spec's, and the rule across
    # settings that adds one (onecycle's ratio above 0). The warmup words are
    # held to a real Trainer (test_settings_projection).
    assert tc.FIELD_FORMATS["min_lr_ratio"] == (
        f"A fraction of the peak rate, {tf.bound_words(tf.by_name('min_lr_ratio').valid)}; "
        "more than 0 with onecycle.")
    assert tf.bound_words(tf.by_name("min_lr_ratio").valid) == "between 0 and 1"


# --------------------------------------------------------------------------- the freeze's field
def test_the_freeze_field_shows_plain_words_and_sends_the_field_specs_values():
    """Ops-wiring round, decision F1: the freeze is one field of the one
    helper, "Graph backbone", its options the field spec's values shown by
    their words (Gradio's ``(label, value)``; skill §5, §8), so no page holds a
    list of its own; touched, either value is sent (decision S2), and a value
    outside the spec is named by the label."""
    from src.config.recipe import FINETUNE, FINETUNE_SOURCES

    choices = tf.by_name("frozen_parts").choices
    assert set(r.SETTING_CHOICE_WORDS["frozen_parts"]) == set(choices)
    field = tc.setting_field("frozen_parts", FINETUNE["frozen_parts"], elem_id="f")
    assert isinstance(field, gr.Radio) and field.label == "Graph backbone"
    assert field.choices == [("Trained", "none"), ("Frozen", "graph_side")]
    assert [value for _words, value in field.choices] == list(choices)
    assert field.value == "none" == FINETUNE["frozen_parts"]
    words = [words for words, _value in field.choices]
    assert len(set(words)) == len(words)
    for label in (*words, r.SETTING_LABELS["frozen_parts"]):
        assert len(label) <= r.OPTION_CHARS and " · " not in label, label
    # The scheduler's values have no words: shown as they are.
    assert tc.setting_field("scheduler_type", "none", elem_id="s").choices == [
        (value, value) for value in tf.by_name("scheduler_type").choices]
    title, text = tc.setting_help("frozen_parts", FINETUNE, FINETUNE_SOURCES)
    lead = text.split("\n", 1)[0]
    assert title == "Graph backbone" and len(lead) < 90, lead
    assert "Reference: Trained." in text and FINETUNE_SOURCES["frozen_parts"] in text
    # The freeze is a study's two arms, not a tuned value: its help says what
    # each is, never "a provisional default ... not tuned" (review of A2).
    assert text.endswith("\n\nNot a tuned value: Trained is the source's full fine-tune, and "
                         "Frozen is the frozen-backbone control.")
    assert "provisional" not in text
    assert tc.setting_help("num_epochs", FINETUNE, FINETUNE_SOURCES)[1].endswith(
        "\n\nA provisional default from the source, not tuned on this data.")
    reference = {"frozen_parts": FINETUNE["frozen_parts"]}
    assert tc.sent_settings({"frozen_parts": "graph_side"}, ["frozen_parts"], reference) == (
        {"frozen_parts": "graph_side"}, [])
    assert tc.sent_settings({"frozen_parts": "none"}, ["frozen_parts"], reference) == (
        {"frozen_parts": "none"}, [])
    assert tc.sent_settings({"frozen_parts": "graph_side"}, [], reference) == ({}, [])
    assert tc.sent_settings({"frozen_parts": "all"}, ["frozen_parts"], reference) == (
        {}, ["Graph backbone must be one of ['none', 'graph_side'], not 'all'."])


# --------------------------------------------------------------------------- D8, the dataset, the runner
def test_start_is_refused_for_a_read_only_request_before_anything_is_read(monkeypatch,
                                                                          started):
    monkeypatch.delenv("SHEPHERD_DEPLOYMENT_MODE", raising=False)
    monkeypatch.setattr(tc, "_workspace_path", lambda name: pytest.fail("the list was read"))
    notice = tc._on_start(gradio_request(), WS, "", "cpu", [], *_form())
    assert "Refused" in notice and started == []


def test_on_start_refuses_without_a_trainable_listed_dataset(monkeypatch, listed, started):
    """No dataset selected, one not trainable, or one the server no longer
    lists: nothing is started, and the notice says why."""
    from src.api.services.server_lists import ServerListRefused

    assert "select a dataset" in tc._on_start(OP, {"name": None, "row": None}, "", "cpu", [],
                                              *_form())
    untrainable = {"name": "x", "row": {"name": "x", "dataset_fingerprint": FP, "trainable": False,
                                        "not_trainable": "files missing", "manifest": "read"}}
    assert "x is not trainable: files missing" in tc._on_start(OP, untrainable, "", "cpu", [],
                                                               *_form())

    def unlisted(name):
        raise ServerListRefused("gone")

    monkeypatch.setattr(tc, "_workspace_path", unlisted)
    assert "no longer listed" in _start()
    assert started == []


def test_on_start_says_the_runners_refusal(monkeypatch, listed):
    monkeypatch.setattr(tc.training_manager, "start_training",
                        lambda config, **kwargs: {"success": False,
                                                  "error": "The dataset changed since it was "
                                                           "shown; select it again."})
    assert _start() == tc.notice_html(
        "Not started: The dataset changed since it was shown; select it again.", mark="error")
