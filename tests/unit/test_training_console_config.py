"""
Unit tests for Training Console numeric-input validation (_collect_config).

A cleared gr.Number arrives as None; bare int()/float() would raise a raw
TypeError. _collect_config now routes numeric fields through _num and raises a
user-facing ConfigValidationError listing every offending field, which the
Start/Resume handlers surface as a gr.Warning toast instead of launching.
"""
from __future__ import annotations

import pytest

# training_console imports gradio (and the api/service stack) at module load.
pytest.importorskip("gradio")

from src.webui.components import training_console as tc  # noqa: E402
from tests.fixtures.operator_access import gradio_request, operator_mode  # noqa: E402

#: A request D8 allows once the deployment is in mode M1 (`_operator_mode`):
#: Start is an operator control since M5-1.
OP = gradio_request()


@pytest.fixture(autouse=True)
def _operator_mode(monkeypatch):
    operator_mode(monkeypatch)


# _collect_config is positional; keep a valid baseline in signature order.
_DEFAULTS = {
    "data_dir": "data/workspaces/default",
    "run_name": "",
    "num_epochs": 100,
    "learning_rate": 1e-4,
    "batch_size": "32",
    "conv_type": "gat",
    "device": "auto",
    "seed": 42,
    "hidden_dim": "256",
    "num_layers": 4,
    "dropout": 0.1,
    "weight_decay": 0.01,
    "scheduler_type": "cosine",
    "warmup_steps": 500,
    "min_lr_ratio": 0.01,
    "early_stopping_patience": 10,
    "diagnosis_weight": 1.0,
    "link_prediction_weight": 0.5,
    "contrastive_weight": 0.3,
    "ortholog_weight": 0.2,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 1.0,
    "num_heads": "8",
    "use_ortholog_gate": True,
    "amp_mode": "float16",
    "temperature": 0.07,
    "label_smoothing": 0.1,
    "margin": 1.0,
    "num_neighbors_str": "15, 10, 5",
    "max_subgraph_nodes": 5000,
}
_ORDER = list(_DEFAULTS.keys())


def _args(**overrides):
    vals = dict(_DEFAULTS)
    vals.update(overrides)
    return tuple(vals[name] for name in _ORDER)


def _ufield(update, key):
    """Read a field from a gr.update(...) result, tolerant across Gradio versions."""
    if isinstance(update, dict):
        return update.get(key)
    return getattr(update, key, None)


def test_valid_config_returns_dict():
    cfg = tc._collect_config(*_args())
    assert cfg["num_epochs"] == 100
    assert cfg["learning_rate"] == pytest.approx(1e-4)
    assert cfg["batch_size"] == 32
    assert cfg["min_lr_ratio"] == pytest.approx(0.01)


def test_empty_numeric_field_raises_and_names_it():
    with pytest.raises(tc.ConfigValidationError) as ei:
        tc._collect_config(*_args(num_epochs=None))
    assert "Epochs" in str(ei.value)


def test_multiple_empty_fields_all_listed():
    with pytest.raises(tc.ConfigValidationError) as ei:
        tc._collect_config(*_args(seed=None, warmup_steps=None))
    msg = str(ei.value)
    assert "Seed" in msg and "Warmup Steps" in msg


@pytest.mark.parametrize("bad_lr", [0.0, -1e-4])
def test_nonpositive_learning_rate_raises(bad_lr):
    with pytest.raises(tc.ConfigValidationError) as ei:
        tc._collect_config(*_args(learning_rate=bad_lr))
    assert "Learning Rate" in str(ei.value)


def test_min_lr_ratio_below_minimum_rejected():
    # WS5 moved the widget's minimum=1e-4 into _collect_config so the message
    # names the field. (WS1's backend rule allows 0 for cosine/linear; relaxing
    # the UI to match is a separate opt-in, not part of this behavior-preserving
    # change.)
    with pytest.raises(tc.ConfigValidationError) as ei:
        tc._collect_config(*_args(min_lr_ratio=0.0))
    msg = str(ei.value)
    assert "Min LR Ratio" in msg and "0.0001" in msg


def test_below_minimum_names_field():
    with pytest.raises(tc.ConfigValidationError) as ei:
        tc._collect_config(*_args(max_subgraph_nodes=50))
    assert "Max Subgraph Nodes" in str(ei.value)


def test_above_maximum_names_field():
    with pytest.raises(tc.ConfigValidationError) as ei:
        tc._collect_config(*_args(num_epochs=10001))
    assert "Epochs" in str(ei.value)


def test_boundary_values_accepted():
    # seed=0 (lo=0), num_epochs=1 (lo=1), min_lr_ratio=1e-4 (lo) are all valid.
    cfg = tc._collect_config(*_args(seed=0, num_epochs=1, min_lr_ratio=1e-4))
    assert cfg["seed"] == 0 and cfg["num_epochs"] == 1


def test_multi_error_aggregation_lists_all():
    with pytest.raises(tc.ConfigValidationError) as ei:
        tc._collect_config(
            *_args(num_epochs=0, learning_rate=None, max_subgraph_nodes=50)
        )
    msg = str(ei.value)
    assert "Epochs" in msg and "Learning Rate" in msg and "Max Subgraph Nodes" in msg


# --------------------------------------------------------------- WS6: random seed
def test_random_seed_in_numpy_range():
    for _ in range(50):
        s = tc._random_seed()
        assert isinstance(s, int) and 0 <= s <= 2**32 - 1


def test_seed_above_numpy_max_rejected():
    with pytest.raises(tc.ConfigValidationError) as ei:
        tc._collect_config(*_args(seed=2**32))
    assert "Seed" in str(ei.value)


def test_seed_boundaries_accepted():
    assert tc._collect_config(*_args(seed=0))["seed"] == 0
    assert tc._collect_config(*_args(seed=2**32 - 1))["seed"] == 2**32 - 1


def test_seed_field_lock_toggles_interactive():
    assert _ufield(tc._seed_field_lock(True), "interactive") is False
    assert _ufield(tc._seed_field_lock(False), "interactive") is True


#: The run bar's workspace as a session holds it (`_workspace_state`): the
#: Start handler resolves the name through the server's list (here replaced)
#: and sends the full fingerprint shown.
FP = "a" * 64
WS = {"name": "demo", "row": {"name": "demo", "dataset_fingerprint": FP, "trainable": True}}


@pytest.fixture
def listed(monkeypatch, tmp_path):
    """The server's list resolves "demo" to a directory (`_workspace_path`)."""
    monkeypatch.setattr(tc, "_workspace_path", lambda name: tmp_path / name)
    return tmp_path / "demo"


def _form(**overrides):
    """The form's values as Start receives them: `_collect_config`'s order
    after the workspace's directory."""
    return _args(**overrides)[1:]


def _starts(captured, kwargs_seen=None):
    def start(config, **kwargs):
        captured.update(config)
        if kwargs_seen is not None:
            kwargs_seen.update(kwargs)
        return {"success": True, "pid": 1}
    return start


def test_on_start_randomize_overrides_and_echoes_seed(monkeypatch, listed):
    captured, kwargs = {}, {}
    monkeypatch.setattr(tc, "_random_seed", lambda: 777)
    monkeypatch.setattr(tc.training_manager, "start_training", _starts(captured, kwargs))
    # randomize=True overrides a typed seed (999) with the generated one, and the
    # seed field (last output) echoes the seed actually used.
    out = tc._on_start(OP, WS, *_form(seed=999), True)
    assert captured.get("seed") == 777
    assert _ufield(out[-1], "value") == 777
    assert len(out) == 2 and out[0] == ""
    # The workspace is the one the server lists, and the start carries the
    # full dataset fingerprint this session showed.
    assert captured["data_dir"] == str(listed)
    assert kwargs == {"expected_dataset_fingerprint": FP}


def test_on_start_randomize_empty_seed_still_launches(monkeypatch, listed):
    captured = {}
    monkeypatch.setattr(tc, "_random_seed", lambda: 4242)
    monkeypatch.setattr(tc.training_manager, "start_training", _starts(captured))
    # An empty/locked seed field must still launch when randomize is on.
    tc._on_start(OP, WS, *_form(seed=None), True)
    assert captured.get("seed") == 4242


def test_on_start_no_randomize_empty_seed_aborts(monkeypatch, listed):
    calls, warned = [], []
    monkeypatch.setattr(tc.gr, "Warning", lambda message, **k: warned.append(message))
    monkeypatch.setattr(tc.training_manager, "start_training",
                        lambda config, **kwargs: calls.append(config))
    out = tc._on_start(OP, WS, *_form(seed=None), False)
    assert calls == []  # validation aborted -> never launched
    assert "invalid parameters" in out[0] and "Seed" in warned[0]
    assert len(out) == 2


def test_on_start_refuses_without_a_trainable_listed_dataset(monkeypatch, listed):
    """No dataset selected, one not trainable, or one the server no longer
    lists: nothing is started, and the notice says why."""
    from src.api.services.server_lists import ServerListRefused

    calls = []
    monkeypatch.setattr(tc.training_manager, "start_training",
                        lambda config, **kwargs: calls.append(config))
    out = tc._on_start(OP, {"name": None, "row": None}, *_form(), False)
    assert "select a dataset" in out[0]
    untrainable = {"name": "x", "row": {"name": "x", "dataset_fingerprint": FP, "trainable": False,
                                        "not_trainable": "files missing", "manifest": "read"}}
    out = tc._on_start(OP, untrainable, *_form(), False)
    assert "x is not trainable: files missing" in out[0]

    def unlisted(name):
        raise ServerListRefused("gone")

    monkeypatch.setattr(tc, "_workspace_path", unlisted)
    out = tc._on_start(OP, WS, *_form(), False)
    assert "no longer listed" in out[0]
    assert calls == []


def test_on_start_says_the_runners_refusal(monkeypatch, listed):
    monkeypatch.setattr(tc.training_manager, "start_training",
                        lambda config, **kwargs: {"success": False,
                                                  "error": "The dataset changed since it was "
                                                           "shown; select it again."})
    out = tc._on_start(OP, WS, *_form(), False)
    assert out[0] == ("Not started: The dataset changed since it was shown; select it "
                      "again.")


def test_nonnumeric_value_reports_friendly_error():
    # Direct/programmatic call contract: garbage -> friendly error, not raw ValueError.
    with pytest.raises(tc.ConfigValidationError) as ei:
        tc._collect_config(*_args(num_epochs="abc"))
    assert "Epochs" in str(ei.value)


def test_the_seed_index_matches_the_collect_signature():
    """`_on_start` swaps the seed in by position; the index must follow the
    signature when path fields change (contract M3a-2 replaced two with one)."""
    import inspect

    names = list(inspect.signature(tc._collect_config).parameters)
    assert names == _ORDER
    assert names[tc.SEED_PARAM_INDEX] == "seed"
