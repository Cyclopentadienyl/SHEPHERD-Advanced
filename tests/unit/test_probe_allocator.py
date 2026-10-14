"""The deployment probe starts CUDA under the project's allocator rule.

M2's acceptance at `82d7573` ran the trainer as a module from the probe, which
skipped `train_model.py`'s start-up rule, so it used PyTorch's default backend
whatever the saved preset said. The probe now applies the same shared rule
(`allocator_env(..., inherit=True)`) when it is the program, before anything
imports torch, and records the variables, their source and the backend torch
reports. Every case runs in a fresh process: a process that has already imported
torch cannot show the order.

Module: tests/unit/test_probe_allocator.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ALLOCATOR_VARIABLES = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "SHEPHERD_ALLOC_SOURCE")

#: Applies one entry point's rule with the saved settings at SETTINGS, then
#: prints what it chose and whether torch was already imported.
APPLY = r"""
import importlib.util, json, os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import src.config.runtime_presets as presets
presets.RUNTIME_SETTINGS_FILE = Path(sys.argv[2])
torch_first = "torch" in sys.modules
if sys.argv[3] == "probe":
    spec = importlib.util.spec_from_file_location("probe_under_test",
                                                  Path(sys.argv[1]) / "scripts" / "probe_deployment.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    source = module._apply_allocator()["source"]
else:
    import scripts.train_model as module
    source = module._apply_allocator()
print(json.dumps({"torch_first": torch_first, "source": source,
                  **{k: os.environ.get(k) for k in
                     ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "SHEPHERD_ALLOC_SOURCE")}}))
"""


def _clean_env(**allocator):
    env = {k: v for k, v in os.environ.items() if k not in ALLOCATOR_VARIABLES}
    env.update(allocator)
    return env


def _apply(entry, settings, tmp_path, **allocator):
    settings_file = tmp_path / f"settings-{entry}.json"
    settings_file.write_text(json.dumps(settings))
    done = subprocess.run(
        [sys.executable, "-c", APPLY, str(REPO), str(settings_file), entry],
        env=_clean_env(**allocator), capture_output=True, text=True, timeout=300,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


CASES = {
    "no allocator set: the saved preset": (
        {"allocator_preset": "expandable"}, {},
        {"PYTORCH_ALLOC_CONF": "backend:native,expandable_segments:True",
         "PYTORCH_CUDA_ALLOC_CONF": None, "SHEPHERD_ALLOC_SOURCE": "preset",
         "source": "saved preset"}),
    "an explicit native override is kept": (
        {"allocator_preset": "cuda_async"}, {"PYTORCH_ALLOC_CONF": "backend:native"},
        {"PYTORCH_ALLOC_CONF": "backend:native", "PYTORCH_CUDA_ALLOC_CONF": None,
         "SHEPHERD_ALLOC_SOURCE": None, "source": "environment (explicit)"}),
    "an override under the legacy name is kept": (
        {"allocator_preset": "cuda_async"}, {"PYTORCH_CUDA_ALLOC_CONF": "backend:native"},
        {"PYTORCH_ALLOC_CONF": None, "PYTORCH_CUDA_ALLOC_CONF": "backend:native",
         "SHEPHERD_ALLOC_SOURCE": None, "source": "environment (explicit)"}),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_probe_chooses_as_the_training_entry_point_does(tmp_path, case):
    settings, allocator, expected = CASES[case]

    probe = _apply("probe", settings, tmp_path, **allocator)
    trainer = _apply("train_model", settings, tmp_path, **allocator)

    assert probe["torch_first"] is False, "torch was imported before the rule ran"
    assert {k: probe[k] for k in expected} == expected
    assert {k: probe[k] for k in expected} == {k: trainer[k] for k in expected}


def test_run_as_a_program_the_allocator_is_set_before_torch_is_imported(tmp_path):
    """The probe's own start-up, in a fresh process: on a host without CUDA the
    acceptance stops after A2, and the report still records the allocator the
    process was given before its first torch import."""
    import torch

    if torch.cuda.is_available():
        pytest.skip("this host has CUDA; the acceptance would run in full")
    from src.config.runtime_presets import allocator_env, load_runtime_settings

    work = tmp_path / "work"
    done = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "probe_deployment.py"),
         "--work-dir", str(work), "--report", str(work / "report.json"),
         "--m2-acceptance", "--device", "cuda"],
        env=_clean_env(), capture_output=True, text=True, timeout=600, cwd=REPO,
    )
    assert done.returncode == 1, done.stderr[-2000:]

    allocator = json.loads((work / "report.json").read_text())["allocator"]
    expected = allocator_env({}, load_runtime_settings(), inherit=True)
    assert allocator["PYTORCH_ALLOC_CONF"] == expected["PYTORCH_ALLOC_CONF"]
    assert allocator["SHEPHERD_ALLOC_SOURCE"] == "preset"
    assert allocator["source"] == "saved preset"
    assert allocator["torch_imported_before"] is False
    assert allocator["backend"] is None, "no CUDA device was resolved, so none is claimed"
