"""
Unit tests for the shared runtime-allocator presets module.

These are intentionally dependency-light (no gradio/torch) so they exercise the
exact logic shared by the WebUI Runtime Settings tab and the launcher.
"""
import importlib.util
from pathlib import Path

import pytest

from src.config.runtime_presets import (
    ALLOC_SOURCE_ENV,
    ALLOCATOR_PRESETS,
    DEFAULT_ALLOCATOR,
    allocator_env,
    load_runtime_settings,
    resolve_allocator,
    save_runtime_settings,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- load/save
def test_save_load_roundtrip(tmp_path):
    p = tmp_path / "rt.json"
    save_runtime_settings({"allocator_preset": "expandable"}, p)
    assert load_runtime_settings(p) == {"allocator_preset": "expandable"}


def test_load_missing_file_returns_empty(tmp_path):
    assert load_runtime_settings(tmp_path / "nope.json") == {}


def test_load_malformed_json_falls_back(tmp_path):
    p = tmp_path / "rt.json"
    p.write_text("{ this is not valid json", encoding="utf-8")
    assert load_runtime_settings(p) == {}  # must not raise


def test_load_non_object_json_falls_back(tmp_path):
    # Valid JSON but not an object (list / string / number) -> {} so downstream
    # .get(...) never crashes.
    for payload in ("[]", '"hello"', "42"):
        p = tmp_path / "rt.json"
        p.write_text(payload, encoding="utf-8")
        assert load_runtime_settings(p) == {}, payload


# --------------------------------------------------------------------------- resolve
def test_resolve_known_preset():
    assert resolve_allocator("expandable") == ("expandable", ALLOCATOR_PRESETS["expandable"])


def test_resolve_unknown_preset_falls_back_to_default():
    resolved, conf = resolve_allocator("bogus-preset")
    assert resolved == DEFAULT_ALLOCATOR
    assert conf == ALLOCATOR_PRESETS[DEFAULT_ALLOCATOR]


def test_resolve_none_falls_back_to_default():
    resolved, _ = resolve_allocator(None)
    assert resolved == DEFAULT_ALLOCATOR


def test_native_presets_state_backend_explicitly():
    for key in ("expandable", "native_roundup", "native"):
        assert ALLOCATOR_PRESETS[key].startswith("backend:native"), key
    assert ALLOCATOR_PRESETS["cuda_async"] == "backend:cudaMallocAsync"


# --------------------------------------------------------------------------- start environment
def test_start_env_applies_a_saved_non_default_preset():
    env = allocator_env({}, {"allocator_preset": "native"})
    assert env["PYTORCH_ALLOC_CONF"] == ALLOCATOR_PRESETS["native"]
    assert env[ALLOC_SOURCE_ENV] == "preset"


def test_start_env_without_settings_uses_the_default():
    env = allocator_env({}, {})
    assert env["PYTORCH_ALLOC_CONF"] == ALLOCATOR_PRESETS[DEFAULT_ALLOCATOR]
    assert env[ALLOC_SOURCE_ENV] == "preset"


def test_start_env_keeps_an_explicit_override_unmarked():
    for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        env = allocator_env({name: "backend:native"}, {"allocator_preset": "expandable"})
        assert env == {name: "backend:native"}, name


def test_start_env_keeps_what_the_launcher_marked_as_an_override():
    marked = {ALLOC_SOURCE_ENV: "env", "PYTORCH_ALLOC_CONF": "backend:native"}
    assert allocator_env(marked, {"allocator_preset": "expandable"}) == marked


def test_start_env_re_resolves_a_preset_derived_value():
    stale = {ALLOC_SOURCE_ENV: "preset", "PYTORCH_ALLOC_CONF": ALLOCATOR_PRESETS["native"]}
    env = allocator_env(stale, {"allocator_preset": "expandable"})
    assert env["PYTORCH_ALLOC_CONF"] == ALLOCATOR_PRESETS["expandable"]


def test_a_marker_without_a_value_overrides_nothing():
    # "env" left behind with no allocator variable is not an override; without
    # this the process would start on the native allocator.
    env = allocator_env({ALLOC_SOURCE_ENV: "env"}, {"allocator_preset": "expandable"})
    assert env["PYTORCH_ALLOC_CONF"] == ALLOCATOR_PRESETS["expandable"]
    assert env[ALLOC_SOURCE_ENV] == "preset"


def test_start_env_returns_a_copy():
    source = {}
    assert allocator_env(source, {}) is not source
    assert source == {}


# --------------------------------------------------------------------------- single source of truth
def _launcher():
    spec = importlib.util.spec_from_file_location(
        "shep_launch", REPO_ROOT / "scripts" / "launch" / "shep_launch.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_launcher_uses_shared_presets():
    """The launcher must use the shared rule (no divergent duplicate)."""
    mod = _launcher()
    # Imported the canonical functions => guaranteed in sync with the UI.
    assert mod.allocator_env is allocator_env
    assert mod.load_runtime_settings is load_runtime_settings


@pytest.mark.parametrize(
    "settings, expected",
    [
        ({"allocator_preset": "native"}, ALLOCATOR_PRESETS["native"]),
        ({"allocator_preset": "expandable"}, ALLOCATOR_PRESETS["expandable"]),
        ({}, ALLOCATOR_PRESETS[DEFAULT_ALLOCATOR]),
        ({"allocator_preset": "bogus"}, ALLOCATOR_PRESETS[DEFAULT_ALLOCATOR]),
    ],
)
def test_the_launcher_and_a_restart_resolve_the_same_preset(settings, expected):
    # The service unit starts through the launcher, so this is also a systemd
    # start, a crash restart and a reboot; allocator_env is Restart Backend's rule.
    env = {"PYTHONUNBUFFERED": "1"}
    messages = _launcher().apply_allocator(env, settings)
    assert env["PYTORCH_ALLOC_CONF"] == expected
    assert env[ALLOC_SOURCE_ENV] == "preset"
    assert allocator_env(env, settings)["PYTORCH_ALLOC_CONF"] == expected
    assert messages[-1].startswith(f"Runtime: PYTORCH_ALLOC_CONF={expected}")
    if settings.get("allocator_preset") == "bogus":
        assert messages[0].startswith("WARNING: unknown allocator preset 'bogus'")


def test_the_launcher_keeps_an_explicit_override_through_a_restart():
    env = {"PYTORCH_ALLOC_CONF": "backend:native"}
    settings = {"allocator_preset": "expandable"}
    _launcher().apply_allocator(env, settings)
    assert env == {"PYTORCH_ALLOC_CONF": "backend:native", ALLOC_SOURCE_ENV: "env"}
    assert allocator_env(env, settings) == env


def test_the_launcher_re_resolves_an_inherited_preset_value():
    # A shell that inherited a previous launch's environment is not an
    # explicit override: the marker says the value came from a preset.
    env = {"PYTORCH_ALLOC_CONF": ALLOCATOR_PRESETS["native"], ALLOC_SOURCE_ENV: "preset"}
    _launcher().apply_allocator(env, {"allocator_preset": "expandable"})
    assert env["PYTORCH_ALLOC_CONF"] == ALLOCATOR_PRESETS["expandable"]
