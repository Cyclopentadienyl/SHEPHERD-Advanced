"""Training's device rule: `auto` means CUDA, and without CUDA it refuses.

Contract M3a (`docs/working/PLAN_PROVENANCE_CONTRACT.md` §5.2). Training used to
turn `auto` into the CPU silently, so a run asked to train on the deployment's
hardware trained somewhere else and only the checkpoint's resolved device said
so. Measurement already refused (`scripts/measure_scorer.py`, `_resolve_device`);
training now does the same, before any input is read or any output exists. An
explicit `cpu` stays available for development. The rule is
`src.utils.training_device`'s, which the training process and the job runner's
starts ask alike (ops-wiring round, decision S4).

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
    from src.utils.training_device import TrainingDeviceUnavailable, resolve_training_device

    with pytest.raises(TrainingDeviceUnavailable, match="requires CUDA"):
        resolve_training_device(requested)


def test_explicit_cpu_is_permitted(no_cuda):
    from src.utils.training_device import resolve_training_device

    assert resolve_training_device("cpu") == "cpu"


def test_auto_means_cuda_where_cuda_exists(monkeypatch):
    from src.utils.training_device import resolve_training_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_training_device("auto") == "cuda"
    assert resolve_training_device("cuda:0") == "cuda:0"


def test_train_refuses_before_reading_or_writing_anything(no_cuda, tmp_path, monkeypatch):
    """The refusal comes first: no workspace read, no directory created. A base
    run, given exactly the fields it sets as the operator's."""
    from scripts import train_model

    def _must_not_run(*args, **kwargs):
        raise AssertionError("a workspace input was read before the device refusal")

    monkeypatch.setattr(train_model, "read_training_inputs", _must_not_run)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    fields = {"data_dir": str(workspace), "run_dir": str(tmp_path / "run"),
              "learning_stage": "base", "device": "auto"}

    with pytest.raises(train_model.TrainingDeviceUnavailable, match="requires CUDA"):
        train_model.train(train_model.TrainConfig(**fields), set(fields))

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


def test_the_job_runner_refuses_what_the_process_refuses_before_any_start(
        no_cuda, trained_disease_child, monkeypatch, tmp_path):
    """Checkpoint A's verification (decision S4): device auto on a host
    without CUDA was refused by the CLI before anything and accepted by the
    job runner, whose child then failed. Every start now resolves its device
    by the process's resolver before anything is read or spawned, a new base,
    a Fine-tune start and a branch alike, in its sentence (409 over HTTP:
    the host, not the request), and nothing starts."""
    from unittest.mock import patch

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from scripts import train_model
    from src.api.routes.training import router
    from src.api.services.training_manager import FinetuneInputs, TrainingManager
    from src.utils.training_device import TrainingDeviceUnavailable, resolve_training_device
    from tests.fixtures.operator_access import OPERATOR_BASE_URL, operator_mode

    with pytest.raises(TrainingDeviceUnavailable) as raised:
        resolve_training_device("auto")
    said = str(raised.value)
    started = f"{said} Nothing was started."
    data_dir = str(trained_disease_child["data_dir"])
    base, child = str(trained_disease_child["base"]), str(trained_disease_child["disease"])

    for fields in ({"data_dir": data_dir, "learning_stage": "base", "device": "auto"},
                   {"data_dir": data_dir, "learning_stage": "finetune", "task": "disease",
                    "device": "auto", "finetune_from": base},
                   {"data_dir": data_dir, "learning_stage": "finetune", "task": "disease",
                    "device": "auto", "branch_from": child}):
        with pytest.raises(TrainingDeviceUnavailable) as cli:
            train_model.train(train_model.TrainConfig(**fields), set(fields))
        assert str(cli.value) == said

    manager = TrainingManager()
    monkeypatch.setattr("src.api.routes.training.training_manager", manager)
    operator_mode(monkeypatch)
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    http = TestClient(app, base_url=OPERATOR_BASE_URL)
    with patch("subprocess.Popen") as popen:
        answers = [
            manager.start_training({"data_dir": data_dir, "learning_stage": "base"}),
            manager.start_finetune(FinetuneInputs(data_dir=data_dir, finetune_from=base,
                                                  tasks=("disease",), device="auto",
                                                  run_name="", settings={})),
            manager.start_training({"data_dir": data_dir, "branch_from": child}),
        ]
        routes = [
            http.post("/api/v1/training/start", json={"data_dir": data_dir,
                                                      "learning_stage": "base"}),
            http.post("/api/v1/training/finetune", json={"data_dir": data_dir,
                                                         "finetune_from": base,
                                                         "tasks": ["disease"]}),
            http.post("/api/v1/training/branch", json={"data_dir": data_dir,
                                                       "branch_from": child}),
        ]
    for answer in answers:
        assert answer["success"] is False and answer["error"] == started, answer
        assert "invalid_settings" not in answer
    for resp in routes:
        assert resp.status_code == 409 and resp.json()["detail"] == started, resp.text
    popen.assert_not_called()
    assert manager.active_job() is None and manager.run_dir is None
