"""
Tests for the training field-spec (PR-4a).

Groups:
  * self-consistency — torch-free, always run: the spec is well-formed.
  * API parity — pin CURRENT Pydantic behavior mechanically (defaults + ge/gt/le/lt + coverage).
  * WebUI collect parity — drive _collect_config and pin exposure, defaults, and _num ranges.
  * projection parity — every projects_to target attribute exists on the real config class.
Parity groups are importorskip-guarded so they skip where fastapi/pydantic/gradio/torch are
absent and pin reality where those are installed.
"""
import dataclasses
import inspect

import pytest

from src.config import training_fields as tf

NOOP = {"temperature", "label_smoothing", "margin"}
# _collect_config params that are WebUI composites, not 1:1 spec fields:
COMPOSITE_PARAMS = {"amp_mode", "num_neighbors_str"}
STR_PARAMS = {"batch_size", "hidden_dim", "num_heads"}  # passed to _collect_config as strings


# --------------------------------------------------------------------------- self-consistency
def test_no_duplicate_names():
    n = tf.names()
    assert len(n) == len(set(n))


def test_kinds_scopes_and_current_api_types():
    for f in tf.FIELDS:
        assert f.kind in tf.KINDS
        assert f.scope in tf.SCOPES
        assert f.current_api is None or isinstance(f.current_api, dict)


def test_ui_within_valid():
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


def test_projection_rule_is_exception_based():
    """Every field that must_project() declares a target; the only exceptions are runtime
    settings and effective=False no-ops (which must NOT project)."""
    for f in tf.FIELDS:
        if tf.must_project(f):
            assert f.projects_to, f"{f.name}: effective non-runtime field must declare projects_to"
    # paths are NOT an exception — confirm they now project:
    for name in ("data_dir", "output_dir", "checkpoint_dir", "log_dir", "resume_from"):
        assert tf.by_name(name).projects_to == f"TrainConfig.{name}"


def test_closed_enums_have_choices_and_device_has_a_pattern():
    for name in ("conv_type", "scheduler_type", "amp_dtype"):
        f = tf.by_name(name)
        assert f.choices and f.valid_pattern is None
    dev = tf.by_name("device")
    assert dev.valid_pattern is not None and dev.choices is None


def test_accessors():
    assert tf.by_name("seed").default == 42
    assert len(tf.names()) == len(tf.FIELDS)
    assert "temperature" in {f.name for f in tf.in_scope("loss")}
    assert NOOP.isdisjoint({f.name for f in tf.effective_fields()})
    with pytest.raises(KeyError):
        tf.by_name("does_not_exist")


# --------------------------------------------------------------------------- API parity (pin)
def _pyd_constraints(field_info) -> dict:
    """Extract ge/gt/le/lt from a Pydantic v2 FieldInfo's metadata (annotated_types objects)."""
    out = {}
    for m in field_info.metadata:
        for k in ("ge", "gt", "le", "lt"):
            v = getattr(m, k, None)
            if v is not None:
                out[k] = v
    return out


def _api_default(model_field):
    d = model_field.get_default()
    return list(d) if isinstance(d, (list, tuple)) else d


def _spec_default(f):
    return list(f.default) if isinstance(f.default, tuple) and f.kind == "list[int]" else f.default


def _api_model():
    pytest.importorskip("fastapi")
    pytest.importorskip("pydantic")
    from src.api.routes.training import TrainingStartRequest
    return TrainingStartRequest


def test_api_defaults_match_spec():
    mf = _api_model().model_fields
    for f in tf.FIELDS:
        if f.current_api is None:  # not an API field (compile)
            assert f.name not in mf, f"{f.name} should not be an API field"
            continue
        assert f.name in mf, f"{f.name} missing from TrainingStartRequest"
        assert _api_default(mf[f.name]) == _spec_default(f), (
            f"{f.name} default drift: API {_api_default(mf[f.name])} != spec {_spec_default(f)}"
        )


def test_api_constraints_match_spec():
    """Mechanical: the spec's current_api dict must equal the live Pydantic ge/gt/le/lt."""
    mf = _api_model().model_fields
    for f in tf.FIELDS:
        if f.current_api is None:
            continue
        actual = _pyd_constraints(mf[f.name])
        assert actual == f.current_api, (
            f"{f.name} API constraint drift: model {actual} != spec current_api {f.current_api}"
        )


def test_spec_covers_every_api_field():
    api_names = set(_api_model().model_fields)
    assert api_names <= set(tf.names()), f"API fields not in spec: {api_names - set(tf.names())}"


# --------------------------------------------------------------------- WebUI collect parity (pin)
def _training_console():
    pytest.importorskip("gradio")
    pytest.importorskip("pandas")
    from src.webui.components import training_console as tc
    return tc


def _default_collect_kwargs(tc):
    """Build _collect_config kwargs from the spec (composites hard-coded to their widget defaults)."""
    params = list(inspect.signature(tc._collect_config).parameters)
    kw = {}
    for p in params:
        if p == "amp_mode":
            kw[p] = "float16"
        elif p == "num_neighbors_str":
            kw[p] = "15, 10, 5"
        else:
            assert p in tf.names(), f"_collect_config param {p} is not a spec field"
            d = tf.by_name(p).default
            kw[p] = str(d) if p in STR_PARAMS else d
    return kw


def test_webui_collect_exposure_and_defaults(monkeypatch):
    tc = _training_console()
    # compile is sourced from the runtime-settings file; pin it deterministically for the test:
    monkeypatch.setattr(tc, "load_runtime_settings", lambda *a, **k: {})
    cfg = tc._collect_config(**_default_collect_kwargs(tc))

    # exposure: the emitted keys are exactly the webui_exposed fields
    assert set(cfg) == {f.name for f in tf.FIELDS if f.webui_exposed}

    # defaults: every emitted value matches the spec default
    for key, val in cfg.items():
        expected = _spec_default(tf.by_name(key))
        got = list(val) if isinstance(val, (list, tuple)) else val
        assert got == expected, f"{key}: collect default {got} != spec default {expected}"


def test_webui_collect_enforces_num_ranges(monkeypatch):
    tc = _training_console()
    monkeypatch.setattr(tc, "load_runtime_settings", lambda *a, **k: {})
    base = _default_collect_kwargs(tc)
    # each of these is guarded by _num in _collect_config; out-of-range must raise:
    for param, bad in (("num_epochs", 0), ("min_lr_ratio", 2.0), ("max_grad_norm", 0.0), ("seed", -1)):
        with pytest.raises(tc.ConfigValidationError):
            tc._collect_config(**{**base, param: bad})


# ------------------------------------------------------------------- projection parity (pin)
def _load_trainconfig():
    """Load scripts/train_model.py::TrainConfig by path (scripts/ is not an importable package)."""
    pytest.importorskip("torch")            # train_model.py imports torch / torch_geometric at module load
    pytest.importorskip("torch_geometric")
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts" / "train_model.py"
    spec = importlib.util.spec_from_file_location("train_model_for_test", path)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: train_model.py uses `from __future__ import annotations`, so @dataclass
    # resolves TrainConfig's string annotations via sys.modules[cls.__module__] during exec.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.TrainConfig


def _field_names(cls) -> set:
    if dataclasses.is_dataclass(cls):
        return {f.name for f in dataclasses.fields(cls)}
    return set(vars(cls()))


def _projection_classes():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from src.models.gnn.shepherd_gnn import ShepherdGNNConfig
    from src.kg.data_loader import DataLoaderConfig
    from src.training.trainer import TrainerConfig
    from src.training.loss_functions import LossConfig
    return {
        "ShepherdGNNConfig": ShepherdGNNConfig,
        "DataLoaderConfig": DataLoaderConfig,
        "TrainerConfig": TrainerConfig,
        "LossConfig": LossConfig,
        "TrainConfig": _load_trainconfig(),
    }


def test_projection_targets_exist_in_real_source():
    classes = _projection_classes()
    for f in tf.FIELDS:
        if not f.projects_to:
            continue
        cls_name, attr = f.projects_to.split(".", 1)
        assert cls_name in classes, f"{f.name}: unknown projection class {cls_name}"
        assert attr in _field_names(classes[cls_name]), (
            f"{f.name}: projects_to {f.projects_to} but {cls_name} has no attribute {attr}"
        )


def test_trainconfig_defaults_match_spec_and_noop_absent():
    TrainConfig = _load_trainconfig()
    ALLOW_NON_SPEC = {"config_file"}  # TrainConfig-internal, not a user-facing training field
    fields = {f.name: f for f in dataclasses.fields(TrainConfig)}

    for name, fld in fields.items():
        if name in ALLOW_NON_SPEC:
            continue
        assert name in tf.names(), f"TrainConfig.{name} is not described by the spec"
        tc_default = fld.default if fld.default is not dataclasses.MISSING else (
            fld.default_factory() if fld.default_factory is not dataclasses.MISSING else dataclasses.MISSING
        )
        spec_default = _spec_default(tf.by_name(name))
        got = list(tc_default) if isinstance(tc_default, (list, tuple)) else tc_default
        assert got == spec_default, f"{name}: TrainConfig default {got} != spec default {spec_default}"

    for n in NOOP:
        assert n not in fields, f"{n} unexpectedly present in TrainConfig — no-op pin is stale (PR-4b landed?)"
