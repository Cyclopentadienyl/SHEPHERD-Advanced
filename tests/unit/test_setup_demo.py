"""The demo setup's printed serving command, under contract M3c's device rule.

Serving refuses `auto` without CUDA, as training does, so the command
`scripts/setup_demo.py` prints after `--train-model` names the device the demo
was trained on. These tests hold the command to that, and run it the way the
service would: a demo trained on the CPU here must be served by the very
environment the command sets, on this host without CUDA. Without a model, the
path-reasoning command names no device, because no device is resolved.

Module: tests/unit/test_setup_demo.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import setup_demo  # noqa: E402


def _printed(monkeypatch, capsys, tmp_path, *args):
    output = tmp_path / "demo"
    monkeypatch.setattr(sys, "argv", ["setup_demo.py", "--output-dir", str(output), *args])
    setup_demo.main()
    return capsys.readouterr().out, output


def _environment(printed: str) -> dict:
    """The `NAME=value \\` lines of the printed command."""
    return dict(re.findall(r"^\s+(SHEPHERD_[A-Z_]+)=(\S+) \\$", printed, flags=re.M))


@pytest.mark.parametrize("trained_on, named", [
    ("auto", "cuda"),         # training's auto ran on CUDA, or it refused
    ("cuda", "cuda"),
    ("cuda:1", "cuda:1"),
    ("cpu", "cpu"),
    (" cpu ", "cpu"),
    ("", "cuda"),
])
def test_the_command_names_the_device_the_demo_trained_on(trained_on, named):
    assert setup_demo.serving_device(trained_on) == named


@pytest.mark.parametrize("device, named", [("cpu", "cpu"), ("auto", "cuda")])
def test_the_printed_command_carries_shepherd_device(monkeypatch, capsys, tmp_path, device,
                                                     named):
    checkpoint = tmp_path / "demo" / "runs" / "gat" / "demo" / "checkpoints" / "best.pt"
    seen = {}

    def _trained(workspace, num_epochs, device):
        seen["device"] = device
        return checkpoint

    monkeypatch.setattr(setup_demo, "train_demo_model", _trained)
    printed, output = _printed(monkeypatch, capsys, tmp_path, "--train-model",
                               "--device", device, "--epochs", "1")
    assert seen["device"] == device
    assert _environment(printed) == {
        "SHEPHERD_KG_PATH": str(output / "kg.json"),
        "SHEPHERD_CHECKPOINT_PATH": str(checkpoint),
        "SHEPHERD_DATA_DIR": str(output),
        "SHEPHERD_DEVICE": named,
    }


def test_the_path_reasoning_command_names_no_device(monkeypatch, capsys, tmp_path):
    """No checkpoint, no device resolved (contract M3c): the command stays as it
    was, and the training hint says what a host without CUDA must add."""
    printed, output = _printed(monkeypatch, capsys, tmp_path)
    assert _environment(printed) == {"SHEPHERD_KG_PATH": str(output / "kg.json")}
    assert "path-reasoning only" in printed and "add --device cpu" in printed


def test_a_demo_trained_on_the_cpu_is_served_by_the_printed_environment(
        monkeypatch, capsys, tmp_path):
    """The command a CPU-only host is given serves here, with no CUDA: the
    service reads the environment it sets and identifies what it serves."""
    pytest.importorskip("torch")
    import torch

    if torch.cuda.is_available():
        pytest.skip("shows the CPU-only host's case; this host has CUDA")
    printed, _ = _printed(monkeypatch, capsys, tmp_path, "--train-model", "--device", "cpu",
                          "--epochs", "1")
    environment = _environment(printed)
    assert environment["SHEPHERD_DEVICE"] == "cpu"
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    import src.api.main as api

    bundle = api.build_pipeline()
    identity = bundle.config["served_identity"]
    assert bundle.config["gnn_ready"]
    assert identity["checkpoint"]["state"] == "verified"
    assert (identity["settings"]["device"], identity["settings"]["device_requested"]) == (
        "cpu", "cpu")

    # The same command without the device is what serving refuses here.
    from src.inference.pipeline import ServingDeviceUnavailable

    monkeypatch.delenv("SHEPHERD_DEVICE")
    with pytest.raises(ServingDeviceUnavailable):
        api.build_pipeline()
