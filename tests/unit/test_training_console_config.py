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


# _collect_config is positional; keep a valid baseline in signature order.
_DEFAULTS = {
    "data_dir": "data/workspaces/default",
    "output_dir": "outputs",
    "checkpoint_dir": "",
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


def test_min_lr_ratio_zero_is_allowed():
    # 0 is valid for cosine/linear (decay-to-zero); must NOT be rejected here.
    cfg = tc._collect_config(*_args(scheduler_type="cosine", min_lr_ratio=0.0))
    assert cfg["min_lr_ratio"] == 0.0


def test_nonnumeric_value_reports_friendly_error():
    # Direct/programmatic call contract: garbage -> friendly error, not raw ValueError.
    with pytest.raises(tc.ConfigValidationError) as ei:
        tc._collect_config(*_args(num_epochs="abc"))
    assert "Epochs" in str(ei.value)
