"""
Tests for the training field-spec (PR-4a).

Two groups:
  * self-consistency — torch-free, always run: the spec is well-formed.
  * parity — pin CURRENT behavior of the real surfaces (API model, TrainConfig). Guarded by
    importorskip so they skip cleanly where fastapi/pydantic/torch are absent, and pin reality
    where those are installed. PR-4a asserts current behavior only; enforcement is PR-4c.
"""
import dataclasses

import pytest

from src.config import training_fields as tf

NOOP = {"temperature", "label_smoothing", "margin"}


# --------------------------------------------------------------------------- self-consistency
def test_no_duplicate_names():
    n = tf.names()
    assert len(n) == len(set(n)), "duplicate field name in spec"


def test_kinds_and_scopes_valid():
    for f in tf.FIELDS:
        assert f.kind in tf.KINDS
        assert f.scope in tf.SCOPES


def test_ui_within_valid():
    """WebUI conservative range/choices must sit inside the hard valid range (ui ⊆ valid)."""
    for f in tf.FIELDS:
        vlo, vhi = (f.valid if f.valid else (None, None))
        if f.ui:
            ulo, uhi = f.ui[0], (f.ui[1] if len(f.ui) > 1 else None)
            if vlo is not None and ulo is not None:
                assert ulo >= vlo, f"{f.name}: ui lo {ulo} < valid lo {vlo}"
            if vhi is not None and uhi is not None:
                assert uhi <= vhi, f"{f.name}: ui hi {uhi} > valid hi {vhi}"
        if f.ui_choices and f.valid:
            for c in f.ui_choices:
                if isinstance(c, (int, float)):
                    if vlo is not None:
                        assert c >= vlo, f"{f.name}: ui choice {c} < valid lo {vlo}"
                    if vhi is not None:
                        assert c <= vhi, f"{f.name}: ui choice {c} > valid hi {vhi}"


def test_noop_fields_are_exactly_the_effective_false_set():
    assert {f.name for f in tf.FIELDS if not f.effective} == NOOP


def test_noop_fields_are_not_projected():
    for f in tf.FIELDS:
        if not f.effective:
            assert f.projects_to is None, f"{f.name}: no-op field must not declare a projection"


def test_effective_runtime_fields_declare_a_projection():
    """Every effective model/dataloader/trainer/loss field must project into a downstream config."""
    for f in tf.FIELDS:
        if f.effective and f.scope in {"model", "dataloader", "trainer", "loss"}:
            assert f.projects_to, f"{f.name}: effective {f.scope} field must declare projects_to"


def test_closed_enums_have_choices_and_device_has_a_pattern():
    for name in ("conv_type", "scheduler_type", "amp_dtype"):
        f = tf.by_name(name)
        assert f.choices, f"{name}: closed enum must declare choices"
        assert f.valid_pattern is None
    dev = tf.by_name("device")
    assert dev.valid_pattern is not None, "device must use a validator pattern, not a closed enum"
    assert dev.choices is None


def test_accessors():
    assert tf.by_name("seed").default == 42
    assert len(tf.names()) == len(tf.FIELDS)
    assert "temperature" in {f.name for f in tf.in_scope("loss")}
    assert NOOP.isdisjoint({f.name for f in tf.effective_fields()})
    with pytest.raises(KeyError):
        tf.by_name("does_not_exist")


# --------------------------------------------------------------------------- API parity (pin)
def _api_default(model_field):
    d = model_field.get_default()
    return list(d) if isinstance(d, (list, tuple)) else d


def _spec_default(f):
    return list(f.default) if isinstance(f.default, tuple) and f.kind == "list[int]" else f.default


def test_api_model_defaults_match_spec():
    pytest.importorskip("fastapi")
    pytest.importorskip("pydantic")
    from src.api.routes.training import TrainingStartRequest

    mf = TrainingStartRequest.model_fields
    for f in tf.FIELDS:
        if f.scope == "runtime_setting":
            assert f.name not in mf, f"{f.name} is a runtime setting; should not be an API field"
            continue
        assert f.name in mf, f"{f.name} missing from TrainingStartRequest"
        assert _api_default(mf[f.name]) == _spec_default(f), (
            f"{f.name} default drift: API {_api_default(mf[f.name])} != spec {_spec_default(f)}"
        )


def test_spec_covers_every_api_field():
    pytest.importorskip("fastapi")
    pytest.importorskip("pydantic")
    from src.api.routes.training import TrainingStartRequest

    api_names = set(TrainingStartRequest.model_fields)
    assert api_names <= set(tf.names()), (
        f"API fields not described by the spec: {api_names - set(tf.names())}"
    )


# ------------------------------------------------------------------- TrainConfig parity (pin)
def _load_trainconfig():
    """Load scripts/train_model.py::TrainConfig by path (scripts/ is not an importable package)."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts" / "train_model.py"
    spec = importlib.util.spec_from_file_location("train_model_for_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TrainConfig


def test_noop_fields_absent_from_trainconfig():
    """Pin the accepted-but-not-effective reality: TrainConfig has no temperature/label_smoothing/margin."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    TrainConfig = _load_trainconfig()
    tc_names = {f.name for f in dataclasses.fields(TrainConfig)}
    for name in NOOP:
        assert name not in tc_names, (
            f"{name} unexpectedly present in TrainConfig — the no-op pin is stale (did PR-4b land?)"
        )


def test_effective_loss_weights_present_in_trainconfig():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    TrainConfig = _load_trainconfig()
    tc_names = {f.name for f in dataclasses.fields(TrainConfig)}
    for name in ("diagnosis_weight", "link_prediction_weight", "contrastive_weight", "ortholog_weight"):
        assert name in tc_names, f"{name} missing from TrainConfig"
