"""
`scripts/train_model.py` puts the CUDA allocator in place before torch loads.

torch fixes its allocator backend when its CUDA library loads, so the only
moment that matters is the first import of torch. Each case runs the script's
module body as the program (``__main__``) in a fresh process, with an import hook that records the
environment at that moment and stops there, so torch is never actually loaded
and the saved settings are injected rather than read from the repository.
"""
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

from src.config.runtime_presets import ALLOC_SOURCE_ENV, ALLOCATOR_PRESETS, DEFAULT_ALLOCATOR

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "train_model.py"
TRAINING_MANAGER = REPO_ROOT / "src" / "api" / "services" / "training_manager.py"

_PROBE = r"""
import importlib.abc, json, os, runpy, sys
sys.path.insert(0, {repo!r})
import src.config.runtime_presets as presets
presets.load_runtime_settings = lambda path=None: {settings!r}


class _TorchReached(Exception):
    pass


class _Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "torch":
            print(json.dumps({{
                "PYTORCH_ALLOC_CONF": os.environ.get("PYTORCH_ALLOC_CONF"),
                "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                "marker": os.environ.get({marker!r}),
            }}))
            raise _TorchReached
        return None


sys.meta_path.insert(0, _Hook())
try:
    runpy.run_path({script!r}, run_name={run_name!r})
except _TorchReached:
    pass
else:
    print(json.dumps({{"error": "torch was never imported"}}))
"""


def _at_first_torch_import(settings, run_name="__main__", **env):
    clean = {k: v for k, v in os.environ.items()
             if k not in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", ALLOC_SOURCE_ENV)}
    clean.update(env)
    probe = _PROBE.format(repo=str(REPO_ROOT), settings=settings, marker=ALLOC_SOURCE_ENV,
                          script=str(SCRIPT), run_name=run_name)
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                            encoding="utf-8", env=clean, cwd=REPO_ROOT, timeout=120)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_a_shell_run_gets_the_saved_preset_before_torch():
    seen = _at_first_torch_import({"allocator_preset": "native"})
    assert seen["PYTORCH_ALLOC_CONF"] == ALLOCATOR_PRESETS["native"]
    assert seen["marker"] == "preset"


def test_a_shell_run_with_nothing_saved_gets_the_default():
    seen = _at_first_torch_import({})
    assert seen["PYTORCH_ALLOC_CONF"] == ALLOCATOR_PRESETS[DEFAULT_ALLOCATOR]


def test_an_explicit_override_is_kept():
    seen = _at_first_torch_import({"allocator_preset": "expandable"},
                                  PYTORCH_ALLOC_CONF="backend:native")
    assert seen["PYTORCH_ALLOC_CONF"] == "backend:native"
    assert seen["marker"] is None


def test_a_legacy_override_is_kept_and_nothing_is_added():
    seen = _at_first_torch_import({"allocator_preset": "expandable"},
                                  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    assert seen["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert seen["PYTORCH_ALLOC_CONF"] is None


def test_a_webui_run_inherits_the_backends_allocator():
    # The backend was started under cudaMallocAsync; the saved preset changed
    # since, which Runtime Settings says takes effect on a backend restart. The
    # training run it spawns runs under the backend's allocator until then.
    inherited = {"PYTORCH_ALLOC_CONF": ALLOCATOR_PRESETS["cuda_async"], ALLOC_SOURCE_ENV: "preset"}
    seen = _at_first_torch_import({"allocator_preset": "expandable"}, **inherited)
    assert seen["PYTORCH_ALLOC_CONF"] == ALLOCATOR_PRESETS["cuda_async"]


def test_a_marker_without_a_value_gets_the_saved_preset():
    seen = _at_first_torch_import({"allocator_preset": "expandable"}, **{ALLOC_SOURCE_ENV: "env"})
    assert seen["PYTORCH_ALLOC_CONF"] == ALLOCATOR_PRESETS["expandable"]


def test_importing_it_as_a_module_changes_nothing():
    # Tests import scripts.train_model; that must not rewrite their process's
    # environment, where torch is already loaded anyway.
    seen = _at_first_torch_import({"allocator_preset": "native"}, run_name="scripts.train_model")
    assert seen["PYTORCH_ALLOC_CONF"] is None
    assert seen["marker"] is None


def test_the_webui_starts_training_with_the_backends_environment():
    # The inheritance above holds only if the training subprocess is given the
    # backend's environment, i.e. Popen is called with no env= of its own.
    tree = ast.parse(TRAINING_MANAGER.read_text(encoding="utf-8"))
    popens = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "Popen"
    ]
    assert popens, "training_manager no longer starts training with subprocess.Popen"
    for call in popens:
        assert "env" not in {kw.arg for kw in call.keywords}
