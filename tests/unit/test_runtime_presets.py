"""
Unit tests for the shared runtime-allocator presets module.

These are intentionally dependency-light (no gradio/torch) so they exercise the
exact logic shared by the WebUI Runtime Settings tab and the launcher.
"""
import importlib.util
from pathlib import Path

from src.runtime_presets import (
    ALLOCATOR_PRESETS,
    DEFAULT_ALLOCATOR,
    effective_allocator,
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


# --------------------------------------------------------------------------- env precedence
def test_env_override_takes_precedence():
    assert effective_allocator(
        {"PYTORCH_ALLOC_CONF": "backend:cudaMallocAsync"},
        {"allocator_preset": "expandable"},
    ) == (None, None)
    assert effective_allocator(
        {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}, {}
    ) == (None, None)


def test_no_env_uses_persisted_setting():
    preset, conf = effective_allocator({}, {"allocator_preset": "expandable"})
    assert preset == "expandable"
    assert conf == ALLOCATOR_PRESETS["expandable"]


def test_no_env_no_setting_uses_default():
    preset, conf = effective_allocator({}, {})
    assert preset == DEFAULT_ALLOCATOR
    assert conf == ALLOCATOR_PRESETS[DEFAULT_ALLOCATOR]


def test_no_env_unknown_setting_falls_back():
    preset, _ = effective_allocator({}, {"allocator_preset": "bogus"})
    assert preset == DEFAULT_ALLOCATOR


# --------------------------------------------------------------------------- single source of truth
def test_launcher_uses_shared_presets():
    """The launcher must use the shared resolver (no divergent duplicate map)."""
    spec = importlib.util.spec_from_file_location(
        "shep_launch", REPO_ROOT / "scripts" / "launch" / "shep_launch.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Imported the canonical functions => guaranteed in sync with the UI.
    assert mod.effective_allocator is effective_allocator
    assert mod.load_runtime_settings is load_runtime_settings
