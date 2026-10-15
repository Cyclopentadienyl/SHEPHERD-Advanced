"""Training's device rule: `auto` means CUDA, and without CUDA it refuses.

Contract M3a (`docs/working/PLAN_PROVENANCE_CONTRACT.md` §5.2). Training used to
turn `auto` into the CPU silently, so a run asked to train on the deployment's
hardware trained somewhere else and only the checkpoint's resolved device said
so. Measurement already refused (`scripts/measure_scorer.py`, `_resolve_device`);
training now does the same, before any input is read or any output exists. An
explicit `cpu` stays available for development.

Module: tests/unit/test_training_device.py
"""
from __future__ import annotations

import pytest
import torch


@pytest.fixture
def no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.mark.parametrize("requested", ["auto", "cuda", "cuda:1"])
def test_a_cuda_request_refuses_without_cuda(no_cuda, requested):
    from scripts.train_model import TrainingDeviceUnavailable, resolve_training_device

    with pytest.raises(TrainingDeviceUnavailable, match="requires CUDA"):
        resolve_training_device(requested)


def test_explicit_cpu_is_permitted(no_cuda):
    from scripts.train_model import resolve_training_device

    assert resolve_training_device("cpu") == "cpu"


def test_auto_means_cuda_where_cuda_exists(monkeypatch):
    from scripts.train_model import resolve_training_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_training_device("auto") == "cuda"
    assert resolve_training_device("cuda:0") == "cuda:0"


def test_train_refuses_before_reading_or_writing_anything(no_cuda, tmp_path, monkeypatch):
    """The refusal comes first: no workspace read, no directory created."""
    from scripts import train_model

    def _must_not_run(*args, **kwargs):
        raise AssertionError("a workspace input was read before the device refusal")

    monkeypatch.setattr(train_model, "read_training_inputs", _must_not_run)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    config = train_model.TrainConfig(
        data_dir=str(workspace),
        run_dir=str(tmp_path / "run"),
        device="auto",
    )

    with pytest.raises(train_model.TrainingDeviceUnavailable):
        train_model.train(config)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["ws"]


def test_the_demo_passes_its_device_through(monkeypatch, tmp_path):
    """`setup_demo.py --train-model` takes a device; `auto` is its default."""
    from scripts import setup_demo
    import scripts.train_model as train_model

    seen = {}

    def _train(config):
        seen["device"] = config.device
        return {}

    monkeypatch.setattr(train_model, "train", _train)
    with pytest.raises(SystemExit):
        setup_demo.train_demo_model(tmp_path, num_epochs=1, device="cpu")
    assert seen["device"] == "cpu"
