"""
Tests for the training field-spec (PR-4a).

Groups:
  * self-consistency — torch-free, always run: the spec (incl. target policy and cross-field
    rules) is well-formed and self-validating.
  * API parity — pin CURRENT Pydantic behavior mechanically (defaults + ge/gt/le/lt + coverage).
  * projection parity — every projects_to target attribute exists on the real config class.
Parity groups are importorskip-guarded so they skip where fastapi/pydantic/gradio/torch are
absent and pin reality where those are installed.
The WebUI collect parity group went with the retired ShepherdGNN form (K6): the base form
reads `recipe.BASE` and checks values with `value_problem` (tests/unit/test_training_page.py).
"""
import ast
import dataclasses
from pathlib import Path

import pytest

from src.config import training_fields as tf

WIRED_IN_PR4B = ("temperature", "label_smoothing", "margin")  # were no-ops until PR-4b
DEVICE_ACCEPTS = ("auto", "cpu", "mps", "cuda", "cuda:0", "cuda:1", "cuda:7")
# Trailing-newline cases matter: Python's ``$`` matches just before a final newline, so a
# ``re.match`` guard would let "cuda\n" through and it would only fail deep in the subprocess.
DEVICE_REJECTS = (
    "gpu0", "cuda:", "cuda:x", "CUDA", "", "cuda:1:2",
    "cuda\n", "cuda:1\n", "cpu\n", " mps", "cuda ",
)


# --------------------------------------------------------------------------- self-consistency
def test_no_duplicate_names():
    n = tf.names()
    assert len(n) == len(set(n))


def test_kinds_and_scopes_valid():
    for f in tf.FIELDS:
        assert f.kind in tf.KINDS
        assert f.scope in tf.SCOPES


def test_constraint_dicts_are_well_formed():
    """Bound dicts use only ge/gt/le/lt and never mix inclusive with exclusive on a side."""
    for f in tf.FIELDS:
        for label in ("valid", "current_api", "item_valid"):
            d = getattr(f, label)
            if d is None:
                continue
            assert set(d) <= set(tf.BOUND_KEYS), f"{f.name}.{label}: bad keys {set(d)}"
            assert not ("ge" in d and "gt" in d), f"{f.name}.{label}: both ge and gt"
            assert not ("le" in d and "lt" in d), f"{f.name}.{label}: both le and lt"
        lo, _, hi, _ = tf.bounds(f.valid)
        if lo is not None and hi is not None:
            assert lo <= hi, f"{f.name}: valid lo {lo} > hi {hi}"


def test_no_field_is_marked_ineffective():
    """PR-4b wired the last three no-op knobs; nothing should be accepted-but-ineffective now."""
    assert {f.name for f in tf.FIELDS if not f.effective} == set()


def test_ineffective_fields_would_not_be_projected():
    """Invariant kept for the future: an effective=False field must not declare a projection."""
    for f in tf.FIELDS:
        if not f.effective:
            assert f.projects_to is None, f"{f.name}: no-op field must not declare a projection"


def test_pr4b_loss_knobs_are_effective_and_projected():
    for name in WIRED_IN_PR4B:
        f = tf.by_name(name)
        assert f.effective, f"{name} must be effective after PR-4b"
        assert f.projects_to == f"LossConfig.{name}", f"{name} must project into LossConfig"


# ------------------------------------------------- wiring parity (source-level, torch-free)
def _train_model_source() -> ast.Module:
    path = Path(__file__).resolve().parents[2] / "scripts" / "train_model.py"
    return ast.parse(path.read_text(encoding="utf-8"))


def _call_kwargs(tree: ast.Module, func_name: str) -> dict:
    """Return {kwarg: source-expression} for the first call to func_name in the module."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == func_name:
            return {kw.arg: ast.unparse(kw.value) for kw in node.keywords if kw.arg}
    return {}


def test_trainconfig_declares_every_projected_field():
    """Source-level: a field only has runtime effect if TrainConfig has the attribute
    (load_config copies YAML keys behind a hasattr gate)."""
    tree = _train_model_source()
    declared = {
        t.target.id
        for cls in ast.walk(tree)
        if isinstance(cls, ast.ClassDef) and cls.name == "TrainConfig"
        for t in cls.body
        if isinstance(t, ast.AnnAssign) and isinstance(t.target, ast.Name)
    }
    assert declared, "could not locate TrainConfig field declarations"
    for f in tf.FIELDS:
        if f.projects_to:
            assert f.name in declared, (
                f"{f.name} projects to {f.projects_to} but TrainConfig does not declare it — "
                f"load_config's hasattr gate would silently drop it"
            )


def test_projected_loss_fields_reach_the_lossconfig_call():
    """Declaring the field on TrainConfig is not enough: it must also be passed into
    LossConfig(...), otherwise the value is carried and then ignored (the PR-4b bug)."""
    kwargs = _call_kwargs(_train_model_source(), "LossConfig")
    assert kwargs, "could not locate the LossConfig(...) construction in train_model.py"
    for f in tf.FIELDS:
        if f.projects_to and f.projects_to.startswith("LossConfig."):
            attr = f.projects_to.split(".", 1)[1]
            assert attr in kwargs, f"{f.name}: LossConfig(...) never receives {attr}"
            assert kwargs[attr] == f"config.{f.name}", (
                f"{f.name}: LossConfig({attr}=...) is wired to {kwargs[attr]!r}, expected config.{f.name}"
            )


def test_projection_rule_is_exception_based():
    for f in tf.FIELDS:
        if tf.must_project(f):
            assert f.projects_to, f"{f.name}: effective non-runtime field must declare projects_to"
    for name in ("data_dir", "run_name", "run_dir", "resume_from"):
        assert tf.by_name(name).projects_to == f"TrainConfig.{name}"


def test_closed_enums_have_choices_and_device_has_a_pattern():
    for name in ("conv_type", "scheduler_type", "amp_dtype"):
        f = tf.by_name(name)
        assert f.choices and f.valid_pattern is None
    dev = tf.by_name("device")
    assert dev.valid_pattern is not None and dev.choices is None


def test_num_neighbors_item_policy_is_structured():
    """The list policy must be machine-readable, not prose (PR-4c consumes it)."""
    f = tf.by_name("num_neighbors")
    assert f.kind == "list[int]"
    assert f.item_valid == {"ge": 1}, "each element must be declared int >= 1"
    assert f.min_length == 1, "non-empty must be declared structurally"


def test_cross_field_rules_are_structured_and_reference_real_fields():
    assert tf.CROSS_FIELD_RULES, "cross-field rules must be represented structurally"
    names = set(tf.names())
    for rule in tf.CROSS_FIELD_RULES:
        assert rule.subject in names, f"{rule.name}: unknown subject {rule.subject}"
        if rule.operand:
            assert rule.operand in names, f"{rule.name}: unknown operand {rule.operand}"
        for guard_field, allowed in (rule.when or {}).items():
            assert guard_field in names, f"{rule.name}: unknown guard field {guard_field}"
            spec_choices = tf.by_name(guard_field).choices
            if spec_choices:
                assert set(allowed) <= set(spec_choices), (
                    f"{rule.name}: guard values {allowed} not in {guard_field} choices {spec_choices}"
                )
        assert rule.enforced_in, f"{rule.name}: must record where it is enforced"


def test_divisibility_rule_is_conditional_on_conv_type():
    """Regression guard for the approved refinement: sage must NOT be subject to divisibility."""
    rule = next(r for r in tf.CROSS_FIELD_RULES if r.name == "hidden_dim_divisible_by_num_heads")
    assert rule.kind == "divisible_by"
    assert set(rule.when["conv_type"]) == {"hgt", "gat"}, "sage takes no heads — must be excluded"


def test_accessors_and_bounds_helper():
    assert tf.by_name("seed").default == 42
    assert len(tf.names()) == len(tf.FIELDS)
    assert len(tf.effective_fields()) == len(tf.FIELDS)
    assert tf.bounds({"gt": 0, "le": 1.0}) == (0, True, 1.0, False)
    assert tf.bounds(None) == (None, False, None, False)
    with pytest.raises(KeyError):
        tf.by_name("does_not_exist")


# --------------------------------------------------------------------------- API parity (pin)
def _pyd_constraints(field_info) -> dict:
    out = {}
    for m in field_info.metadata:
        for k in tf.BOUND_KEYS:
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
        if f.current_api is None:
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


# ------------------------------------------------- PR-4c: API enforcement (behavioural pins)
def _validation_error():
    pytest.importorskip("pydantic")
    from pydantic import ValidationError
    return ValidationError


def test_api_enforces_declared_choices():
    """Every field declaring current_api_choices must reject a value outside the enum."""
    Model, VE = _api_model(), _validation_error()
    enforced = [f for f in tf.FIELDS if f.current_api_choices]
    assert {f.name for f in enforced} == {"conv_type", "scheduler_type", "amp_dtype"}
    for f in enforced:
        for good in f.current_api_choices:
            Model(**{f.name: good})                       # every declared choice is accepted
        with pytest.raises(VE):
            Model(**{f.name: "definitely-not-a-valid-choice"})


def test_api_enum_sets_exactly_equal_the_spec():
    """guard-not-derive: the live Literal must be exactly current_api_choices — not merely a
    superset that happens to accept the declared values."""
    from typing import get_args

    mf = _api_model().model_fields
    for f in tf.FIELDS:
        if not f.current_api_choices:
            continue
        live = get_args(mf[f.name].annotation)
        assert live, f"{f.name}: API annotation is not a Literal enum"
        assert set(live) == set(f.current_api_choices), (
            f"{f.name} enum drift: API {sorted(live)} != spec {sorted(f.current_api_choices)}"
        )


def test_api_enforces_device_grammar_and_keeps_multi_gpu():
    """device is a PyTorch grammar, not a 3-value enum: cuda:N must stay available."""
    Model, VE = _api_model(), _validation_error()
    spec = tf.by_name("device")
    assert spec.current_api_pattern == spec.valid_pattern
    for good in DEVICE_ACCEPTS:
        assert Model(device=good).device == good
    for bad in DEVICE_REJECTS:
        with pytest.raises(VE):
            Model(device=bad)


def test_api_head_divisibility_is_conditional_on_conv_type():
    """hgt/gat require hidden_dim % num_heads == 0; sage must remain exempt."""
    Model, VE = _api_model(), _validation_error()
    for conv in ("hgt", "gat"):
        with pytest.raises(VE):
            Model(conv_type=conv, hidden_dim=100, num_heads=8)
        Model(conv_type=conv, hidden_dim=256, num_heads=8)      # divisible -> fine
    # SAGEConv takes no heads, so a non-divisible pair is legitimate:
    assert Model(conv_type="sage", hidden_dim=100, num_heads=8).hidden_dim == 100


def test_api_enforces_num_neighbors_policy():
    Model, VE = _api_model(), _validation_error()
    spec = tf.by_name("num_neighbors")
    assert spec.current_api_item == spec.item_valid
    assert spec.current_api_min_length == spec.min_length
    assert Model(num_neighbors=[15, 10, 5]).num_neighbors == [15, 10, 5]
    with pytest.raises(VE):
        Model(num_neighbors=[])          # min_length
    with pytest.raises(VE):
        Model(num_neighbors=[15, 0])     # element >= 1
    with pytest.raises(VE):
        Model(num_neighbors=[-1])


def test_api_onecycle_rule_still_enforced():
    """The pre-existing cross-field rule must survive the PR-4c changes."""
    Model, VE = _api_model(), _validation_error()
    with pytest.raises(VE):
        Model(scheduler_type="onecycle", min_lr_ratio=0.0)
    Model(scheduler_type="cosine", min_lr_ratio=0.0)             # decay-to-zero stays legal
    Model(scheduler_type="onecycle", min_lr_ratio=0.01)


def test_cross_field_rules_report_where_they_are_enforced():
    """After PR-4c no rule may still be marked as merely planned."""
    for rule in tf.CROSS_FIELD_RULES:
        assert rule.enforced_in.startswith("current"), (
            f"{rule.name}: still marked '{rule.enforced_in}' — PR-4c should have enforced it"
        )


def test_cli_device_accepts_pytorch_grammar():
    """The CLI was restricted to {auto,cuda,cpu}; it must now follow the same grammar as the API."""
    tree = _train_model_source()
    src = ast.unparse(tree)
    assert 'choices=[\'auto\', \'cuda\', \'cpu\']' not in src, "CLI --device still hard-limited to 3 values"
    assert "_DEVICE_RE" in src, "CLI should validate --device against the device grammar"


def test_cli_device_arg_matches_the_api_grammar():
    """The CLI validator is a second input surface into TrainConfig — it must accept and reject
    exactly what the API does (including the trailing-newline cases)."""
    import argparse

    mod = _load_train_model()
    for good in DEVICE_ACCEPTS:
        assert mod._device_arg(good) == good
    for bad in DEVICE_REJECTS:
        with pytest.raises(argparse.ArgumentTypeError):
            mod._device_arg(bad)


# ------------------------------------------------------------------- projection parity (pin)
def _load_train_model():
    """Load scripts/train_model.py by path (scripts/ is not an importable package)."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[2] / "scripts" / "train_model.py"
    spec = importlib.util.spec_from_file_location("train_model_for_test", path)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: train_model.py uses `from __future__ import annotations`, so @dataclass
    # resolves TrainConfig's string annotations via sys.modules[cls.__module__] during exec.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_trainconfig():
    return _load_train_model().TrainConfig


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


def test_trainconfig_defaults_match_spec():
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

    # PR-4b: the formerly-dropped loss knobs must now exist here, with LossConfig's defaults.
    for n in WIRED_IN_PR4B:
        assert n in fields, f"{n} missing from TrainConfig — PR-4b wiring regressed"


def test_a_value_is_checked_by_its_field_spec_without_coercion():
    """`value_problem` (the Fine-tune start's value check): kinds strictly (a
    bool is not a number, a string never coerced), bounds, choices and list
    items, the gene task's "every neighbour" hop (-1) included."""
    from src.config.training_fields import value_problem

    assert value_problem("num_epochs", 3) is None
    assert value_problem("learning_rate", 1) is None, "an int is a float's value"
    assert value_problem("num_neighbors", [-1, 3]) is None
    assert value_problem("early_stopping_patience", None, allow_none=True) is None
    for name, value, fragment in (
            ("num_epochs", True, "must be an integer, not True"),
            ("num_epochs", "3", "must be an integer, not '3'"),
            ("use_amp", 1, "must be true or false, not 1"),
            ("num_epochs", 0, "num_epochs must be between 1 and 10000, not 0."),
            ("num_neighbors", [3, -2], "num_neighbors items must be 1 or more (or -1, every "
                                       "neighbour), not [3, -2]."),
            ("num_neighbors", [], "needs at least"),
            ("num_epochs", None, "must be set, not null")):
        problem = value_problem(name, value)
        assert problem is not None and fragment in problem, (name, value, problem)


def test_a_value_outside_its_bounds_is_refused_with_the_bounds_in_words():
    """K6 closing smoke: the base form and its Start said "Early stopping
    patience 0 is outside {'ge': 1}", a raw dict. The bound is said in words,
    from the same bounds the check reads (`bounds`), for every kind of bound."""
    from src.config.training_fields import FIELDS, bound_words, bounds, value_problem

    assert value_problem("early_stopping_patience", 0) == (
        "early_stopping_patience must be 1 or more, not 0.")
    assert value_problem("dropout", 1.5) == "dropout must be between 0 and 0.9, not 1.5."
    assert value_problem("learning_rate", 0.0) == (
        "learning_rate must be more than 0 and at most 1, not 0.0.")
    assert value_problem("seed", -1) == "seed must be between 0 and 4294967295, not -1."
    assert [bound_words(c) for c in ({"ge": 1}, {"gt": 0}, {"le": 5}, {"lt": 5},
                                     {"ge": 0.0, "le": 1.0}, {"gt": 0, "le": 1.0},
                                     {"ge": 1, "lt": 5}, {"gt": 0, "lt": 1})] == [
        "1 or more", "more than 0", "5 or less", "less than 5", "between 0 and 1",
        "more than 0 and at most 1", "at least 1 and less than 5", "more than 0 and less than 1"]
    for spec in FIELDS:
        for constraint in (spec.valid, spec.item_valid):
            if constraint:
                words = bound_words(constraint)
                assert "{" not in words and "'" not in words, (spec.name, words)
                lo, _lo_open, hi, _hi_open = bounds(constraint)
                assert all(str(v if not float(v).is_integer() else int(v)) in words
                           for v in (lo, hi) if v is not None), (spec.name, words)
