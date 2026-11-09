"""
The Training page's base form and its Start (K6 decisions 5 and 12).
===================================================================
`src/webui/components/training_console.py`. The ShepherdGNN form and its
collector are gone; the form is exactly the reference recipe's base settings
(`src.config.recipe.BASE`). What these pin:

- **Start sends a base run**: ``learning_stage="base"``, the run keys (the
  workspace the server resolves from its list, the device, the run name
  stripped, compile from the runtime settings) and only the settings whose
  value differs from the recipe's, with the dataset fingerprint shown;
- **a value its field spec refuses starts nothing**, and the notice names it by
  its label, inline (no toast); every kind is parsed as its setting takes it;
- **the line on the reference settings** says which differ, with the recipe's
  value beside each, and which value Start would refuse;
- D8, the dataset and the runner's own refusal, as before.

Module: tests/unit/test_training_console_config.py
"""
from __future__ import annotations

import pytest

# training_console imports gradio (and the api/service stack) at module load.
pytest.importorskip("gradio")

from src.config.recipe import BASE  # noqa: E402
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


def _form(**changed):
    """The base form's values in `BASE_FIELDS` order: the recipe's, but those
    named."""
    values = dict(zip(tc.BASE_FIELDS, tc.base_form_defaults()))
    values.update(changed)
    return tuple(values[name] for name in tc.BASE_FIELDS)


# --------------------------------------------------------------------------- what Start sends
def test_start_sends_a_base_run_with_only_the_settings_that_differ(listed, started):
    assert tc._on_start(OP, WS, "  Pilot Run  ", "cpu", *_form()) == ""
    assert started[-1] == {"data_dir": str(listed), "learning_stage": "base", "device": "cpu",
                           "run_name": "Pilot Run", "compile": False,
                           "__kwargs__": {"expected_dataset_fingerprint": FP}}
    # Changed values only, each as its setting takes it (a whole float of an
    # integer box as an int, the hop text as a list, an int of a float field
    # as a float, an empty patience as none).
    tc._on_start(OP, WS, "", "auto", *_form(num_epochs=5.0, num_neighbors="20, 10, -1",
                                            learning_rate=1, use_amp=True,
                                            early_stopping_patience=7))
    sent = {k: v for k, v in started[-1].items() if k not in (
        "data_dir", "learning_stage", "device", "run_name", "compile", "__kwargs__")}
    assert sent == {"num_epochs": 5, "num_neighbors": [20, 10, -1], "learning_rate": 1.0,
                    "use_amp": True, "early_stopping_patience": 7}
    assert type(sent["num_epochs"]) is int and type(sent["learning_rate"]) is float
    # The recipe's own values, typed again, are not sent: the run records them
    # as the recipe's, never as the operator's.
    tc._on_start(OP, WS, "", "auto", *_form(num_epochs=200, learning_rate=0.0001,
                                            num_neighbors="15,10,5", max_grad_norm=1))
    assert set(started[-1]) == {"data_dir", "learning_stage", "device", "run_name", "compile",
                                "__kwargs__"}


def test_start_takes_compile_from_the_runtime_settings(listed, started, monkeypatch):
    monkeypatch.setattr(tc, "load_runtime_settings", lambda *a, **k: {"torch_compile": True})
    tc._on_start(OP, WS, "", "cpu", *_form())
    assert started[-1]["compile"] is True


@pytest.mark.parametrize("name, value, words", [
    ("num_epochs", None, "Epochs must be set"),
    ("num_epochs", 0, "Epochs must be between 1 and 10000, not 0."),
    ("learning_rate", 0.0, "Learning rate must be more than 0 and at most 1, not 0.0."),
    ("batch_size", 4096, "Batch size must be between 1 and 2048, not 4096."),
    ("seed", 2 ** 32, "Seed must be between 0 and 4294967295, not 4294967296."),
    ("num_neighbors", "15, x, 5", "Neighbours per hop must be whole numbers"),
    ("num_neighbors", "", "Neighbours per hop needs at least 1"),
    ("num_neighbors", "15, 0, 5", "Neighbours per hop items must be 1 or more (or -1, every "
                                  "neighbour), not [15, 0, 5]."),
    # One hop per layer of the encoder the form's base builds (the reference
    # recipe's three), refused here rather than by the training process.
    ("num_neighbors", "15, 10", "Neighbours per hop needs 3 numbers, one hop per encoder "
                                "layer, not 2"),
    ("num_neighbors", "15, 10, 5, 5", "Neighbours per hop needs 3 numbers"),
    ("warmup_steps", -1, "Warmup steps must be 0 or more, not -1."),
    ("weight_decay", -0.1, "Weight decay must be 0 or more, not -0.1."),
    ("max_grad_norm", 0.0, "Gradient clip must be more than 0, not 0.0."),
    # What the browser sends for patience is text (`setting_field`).
    ("early_stopping_patience", "0", "Early stopping patience must be 1 or more, not 0."),
    ("scheduler_type", "cyclic", "Scheduler must be one of"),
])
def test_a_value_its_field_spec_refuses_starts_nothing_and_is_named(listed, started, name,
                                                                    value, words):
    notice = tc._on_start(OP, WS, "", "cpu", *_form(**{name: value}))
    assert started == [] and words in notice and notice.startswith(
        '<span class="shep-notice-t">Not started: ')
    # The same words in the form's own line, before any Start.
    assert words in tc.differs_html(dict(zip(tc.BASE_FIELDS, _form(**{name: value}))), BASE)


def test_an_optional_setting_reads_off_and_what_the_browser_sends_is_the_recipes_none(
        listed, started):
    """K6 closing smoke: patience's reference is None (off). A number box showed
    it as 0 once "More reference settings" was opened, any later edit sent 0,
    and Start was refused ("is outside"), so the reference recipe could not be
    started after looking at its settings. The field is a text box reading
    "off" (a state, not a placeholder); what the browser sends for it, "off" or
    empty once cleared, is the recipe's None and is not sent; a whole number is
    the int; anything else is named."""
    import gradio as gr

    field = tc.setting_field("early_stopping_patience", BASE["early_stopping_patience"],
                             elem_id="p")
    assert BASE["early_stopping_patience"] is None
    assert isinstance(field, gr.Textbox) and field.value == "off"
    assert dict(zip(tc.BASE_FIELDS, tc.base_form_defaults(), strict=True))["early_stopping_patience"] == "off"
    for sent in ("off", "", " Off "):
        values = _form(num_epochs=2, early_stopping_patience=sent)
        assert tc.recipe_differences(dict(zip(tc.BASE_FIELDS, values, strict=True)), BASE) == (
            {"num_epochs": 2}, [])
        assert "Not valid" not in tc._on_base_changed(*values)
        assert tc._on_start(OP, WS, "", "cpu", *values) == ""
        assert "early_stopping_patience" not in started[-1] and started[-1]["num_epochs"] == 2
    seven = dict(zip(tc.BASE_FIELDS, _form(early_stopping_patience=" 7 "), strict=True))
    assert tc.recipe_differences(seven, BASE) == ({"early_stopping_patience": 7}, [])
    assert "Early stopping patience 7 (off)" in tc.differs_html(seven, BASE)
    for typed in ("seven", "7.5"):
        _settings, problems = tc.recipe_differences(
            dict(zip(tc.BASE_FIELDS, _form(early_stopping_patience=typed), strict=True)), BASE)
        assert problems == ['Early stopping patience must be a whole number or "off".']
    # The Fine-tune page's fields (Epochs and Seed) have set references: number
    # boxes, as before.
    from src.config.recipe import FINETUNE

    for name in ("num_epochs", "seed"):
        assert isinstance(tc.setting_field(name, FINETUNE[name], elem_id=name), gr.Number)


def test_every_problem_is_listed_at_once_and_escaped(listed, started):
    notice = tc._on_start(OP, WS, "", "cpu", *_form(num_epochs=None,
                                                    num_neighbors="<img src=x onerror=alert(1)>"))
    assert started == [] and "Epochs must be set" in notice
    assert "Neighbours per hop must be whole numbers" in notice and "<img" not in notice


# --------------------------------------------------------------------------- the line on the reference
def test_the_form_says_which_settings_differ_from_the_reference_settings():
    """Decision 5: "Differs from the reference settings: Epochs 5 (200)"; the
    model spec keeps its own word ("reference recipe")."""
    values = dict(zip(tc.BASE_FIELDS, _form()))
    assert tc.differs_html(values, BASE) == (
        '<div class="shep-differs"><p>Uses the reference settings.</p></div>')
    changed = tc.differs_html({**values, "num_epochs": 5, "use_amp": True,
                               "early_stopping_patience": 10, "num_neighbors": "20, 10, 5"},
                              BASE)
    assert ("Differs from the reference settings: Epochs 5 (200), Neighbours per hop "
            "20, 10, 5 (15, 10, 5), Mixed precision on (off), Early stopping patience 10 (off)"
            in changed)
    assert "recipe" not in changed and tc._on_base_changed(*_form(num_epochs=5)) == (
        tc.differs_html({**values, "num_epochs": 5}, BASE))


def test_each_settings_help_says_what_it_is_its_reference_and_its_source():
    from src.config.recipe import BASE_SOURCES

    for name in tc.BASE_FIELDS:
        title, text = tc.FIELD_HELP[name]
        lead = text.split("\n", 1)[0]
        assert title == tc.SETTING_LABELS[name] and len(lead) <= 90, (name, lead)
        assert f"Reference: {tc.setting_shown(BASE[name])}." in text
        assert BASE_SOURCES[name] in text


# --------------------------------------------------------------------------- D8, the dataset, the runner
def test_start_is_refused_for_a_read_only_request_before_anything_is_read(monkeypatch,
                                                                          started):
    monkeypatch.delenv("SHEPHERD_DEPLOYMENT_MODE", raising=False)
    monkeypatch.setattr(tc, "_workspace_path", lambda name: pytest.fail("the list was read"))
    notice = tc._on_start(gradio_request(), WS, "", "cpu", *_form())
    assert "Refused" in notice and started == []


def test_on_start_refuses_without_a_trainable_listed_dataset(monkeypatch, listed, started):
    """No dataset selected, one not trainable, or one the server no longer
    lists: nothing is started, and the notice says why."""
    from src.api.services.server_lists import ServerListRefused

    assert "select a dataset" in tc._on_start(OP, {"name": None, "row": None}, "", "cpu",
                                              *_form())
    untrainable = {"name": "x", "row": {"name": "x", "dataset_fingerprint": FP, "trainable": False,
                                        "not_trainable": "files missing", "manifest": "read"}}
    assert "x is not trainable: files missing" in tc._on_start(OP, untrainable, "", "cpu",
                                                               *_form())

    def unlisted(name):
        raise ServerListRefused("gone")

    monkeypatch.setattr(tc, "_workspace_path", unlisted)
    assert "no longer listed" in tc._on_start(OP, WS, "", "cpu", *_form())
    assert started == []


def test_on_start_says_the_runners_refusal(monkeypatch, listed):
    monkeypatch.setattr(tc.training_manager, "start_training",
                        lambda config, **kwargs: {"success": False,
                                                  "error": "The dataset changed since it was "
                                                           "shown; select it again."})
    assert tc._on_start(OP, WS, "", "cpu", *_form()) == tc.notice_html(
        "Not started: The dataset changed since it was shown; select it again.")


def test_the_runner_takes_what_the_form_sends(tmp_path, monkeypatch):
    """The form's start passes the runner's own check of a new base run
    (`_new_bundle_run`): the run keys and the recipe's base settings only."""
    from src.api.services.training_manager import _new_bundle_run

    sent = []
    monkeypatch.setattr(tc, "_workspace_path", lambda name: tmp_path / name)
    monkeypatch.setattr(tc.training_manager, "start_training",
                        lambda config, **k: sent.append(config) or {"success": True})
    tc._on_start(OP, WS, "r", "cpu", *_form(num_epochs=3, batch_size=64, use_amp=True,
                                            scheduler_type="cosine", warmup_steps=10,
                                            num_neighbors="20, 10, -1"))
    assert _new_bundle_run(sent[-1]) == ("gatv2", None)
