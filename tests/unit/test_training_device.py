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

from pathlib import Path

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
    """`setup_demo.py --train-model` takes a device; `auto` is its default.
    Both of its runs, the base and its disease child (K7, TM3), are given it,
    and the child starts from the base's `last.pt`."""
    from scripts import setup_demo
    import scripts.train_model as train_model

    seen = []

    def _train(config, explicit):
        seen.append(config)
        last = Path(config.run_dir) / "checkpoints" / "last.pt"
        last.parent.mkdir(parents=True)
        last.write_bytes(b"")
        return {}

    monkeypatch.setattr(train_model, "train", _train)
    child = setup_demo.train_demo_model(tmp_path, num_epochs=1, device="cpu")
    assert [(c.learning_stage, c.task, c.device) for c in seen] == [
        ("base", None, "cpu"), ("finetune", "disease", "cpu")]
    assert seen[1].finetune_from == str(Path(seen[0].run_dir) / "checkpoints" / "last.pt")
    assert child == Path(seen[1].run_dir) / "checkpoints" / "last.pt"
    assert not any(c.conv_type != train_model.TrainConfig().conv_type for c in seen), (
        "no ShepherdGNN setting is given")


def test_the_demo_refuses_when_a_run_writes_no_checkpoint(monkeypatch, tmp_path):
    """A run that wrote no `last.pt` ends the demo with the reason and the
    two-stage training to run instead, before its child is trained."""
    from scripts import setup_demo
    import scripts.train_model as train_model

    seen = []
    monkeypatch.setattr(train_model, "train", lambda config, explicit: seen.append(config) or {})
    with pytest.raises(SystemExit, match="Demo training wrote no checkpoint") as refused:
        setup_demo.train_demo_model(tmp_path, num_epochs=1, device="cpu")
    assert f"--data-dir {tmp_path} --stage base" in str(refused.value)
    assert [c.learning_stage for c in seen] == ["base"]
