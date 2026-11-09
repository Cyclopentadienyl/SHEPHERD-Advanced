"""The demo setup's model and printed steps.

`--train-model` trains a base and one disease child of it through the
production trainer (K7, TM3: it used to train a ShepherdGNN of its own), and
the command it prints serves that child. Serving refuses `auto` without CUDA,
as training does (contract M3c), so the command names the device the demo was
trained on. These tests hold the command to that, and run it the way the
service would: a demo trained on the CPU here must be served by the very
environment the command sets, on this host without CUDA, and answer a disease
query. Without a model no serving command is printed: since K7 nothing is
served without one, and the training it teaches is the two stages (a base,
then its disease model), never a stage-less run or a shortest-path step (D8).

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
    checkpoint = tmp_path / "demo" / "runs" / "gatv2" / "demo-disease" / "checkpoints" / "last.pt"
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


def test_without_a_model_no_serving_command_is_printed(monkeypatch, capsys, tmp_path):
    """K7 (PATH1): a graph is never served without a model, so a workspace
    with no model gets no command to start the API, only the reason and how
    to train one; the hint says what a host without CUDA must add. The
    training it teaches is a base, then its disease model fine-tuned from it
    (K7, D8): no stage-less run and no shortest-path step."""
    printed, output = _printed(monkeypatch, capsys, tmp_path)
    assert _environment(printed) == {}
    assert "uvicorn" not in printed and "path-reasoning" not in printed
    assert "this workspace has nothing to serve yet" in printed
    assert "refuses them (503) until one\nis loaded" in printed
    assert "add --device cpu" in printed
    trains = [line.strip() for line in printed.splitlines() if "train_model.py" in line]
    assert trains == [
        f"python scripts/train_model.py --data-dir {output} --stage base",
        f"python scripts/train_model.py --data-dir {output} --stage finetune \\",
    ]
    assert "--task disease --finetune-from <base run>/checkpoints/last.pt" in printed
    assert "compute_shortest_paths" not in printed and "--epochs" not in printed


def test_a_demo_trained_on_the_cpu_is_a_disease_child_the_printed_environment_serves(
        monkeypatch, capsys, tmp_path):
    """K7 (TM3): `--train-model --device cpu` trains a base and one disease
    child of it through the production trainer, and the command printed
    names the child: a finetuned disease model with a model spec, whose
    parent is the base this run trained, and no ShepherdGNN run. The service
    builds it from the environment the command sets (`build_pipeline`, the
    serving builder), on this host without CUDA, and answers a disease query
    with it."""
    pytest.importorskip("torch")
    import torch

    if torch.cuda.is_available():
        pytest.skip("shows the CPU-only host's case; this host has CUDA")
    printed, output = _printed(monkeypatch, capsys, tmp_path, "--train-model", "--device", "cpu",
                               "--epochs", "1")
    environment = _environment(printed)
    assert environment["SHEPHERD_DEVICE"] == "cpu"

    from src.models.bundle.record import has_model_spec
    from src.utils.checkpoint_io import read_checkpoint

    child = Path(environment["SHEPHERD_CHECKPOINT_PATH"])
    (base,) = (output / "runs" / "gatv2").glob("*_demo-base_*/checkpoints/last.pt")
    assert child.parent.parent.parent == base.parent.parent.parent
    assert child.parent.parent.name.split("_")[1] == "demo-disease" and child.name == "last.pt"
    records = {name: read_checkpoint(path, map_location="cpu", weights_only=True)
               for name, path in (("base", base), ("child", child))}
    child_record = records["child"].checkpoint
    assert has_model_spec(child_record) and has_model_spec(records["base"].checkpoint)
    assert (child_record["learning_stage"], child_record["task"]) == ("finetune", "disease")
    assert records["base"].checkpoint["learning_stage"] == "base"
    assert child_record["parent"]["sha256"] == records["base"].identity.sha256
    assert sorted(p.name for p in (output / "runs").iterdir()) == ["gatv2"], "no ShepherdGNN run"

    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    import src.api.main as api
    from src.core.types import DISEASE_TASK, NodeType, PatientPhenotypes

    bundle = api.build_pipeline()
    identity = bundle.config["served_identity"]
    assert bundle.config["gnn_ready"]
    assert identity["checkpoint"]["state"] == "verified"
    assert identity["checkpoint"]["sha256"] == records["child"].identity.sha256
    assert (identity["settings"]["device"], identity["settings"]["device_requested"]) == (
        "cpu", "cpu")
    result = bundle.pipeline.run(
        patient_input=PatientPhenotypes(patient_id="demo", phenotypes=["HP:0001250",
                                                                        "HP:0001263"]),
        top_k=3, include_explanations=False)
    diseases = {node.id for node in bundle.kg.get_nodes_by_type(NodeType.DISEASE)}
    assert result.result_kind == DISEASE_TASK and len(result.candidates) == 3
    assert {candidate.disease_id for candidate in result.candidates} <= diseases

    # The same command without the device is what serving refuses here.
    from src.inference.pipeline import ServingDeviceUnavailable

    monkeypatch.delenv("SHEPHERD_DEVICE")
    with pytest.raises(ServingDeviceUnavailable):
        api.build_pipeline()


# --------------------------------------------------------------------------- the GPU lock
def test_the_printed_commands_do_not_reload(monkeypatch, capsys, tmp_path):
    """uvicorn's reloader restarts the server on every saved file, and the
    server's shutdown stops the GPU job it runs (the owner's decision of
    2026-10-22), so the printed command has no --reload (without a model none is
    printed). The training hint says the lock must be configured, since
    training refuses without it."""
    checkpoint = tmp_path / "demo" / "runs" / "gatv2" / "demo-disease" / "checkpoints" / "last.pt"
    monkeypatch.setattr(setup_demo, "train_demo_model", lambda *a: checkpoint)
    trained, _ = _printed(monkeypatch, capsys, tmp_path, "--train-model")
    plain, output = _printed(monkeypatch, capsys, tmp_path / "plain")
    assert "python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000\n" in trained
    assert "uvicorn" not in plain
    for printed in (trained, plain):
        assert "--reload" not in printed
    assert f"export SHEPHERD_WORKSPACES_ROOT={output.resolve().parent}" in plain
    assert "§4.11" in plain


def test_train_model_takes_the_gpu_lock_for_the_whole_training(monkeypatch, capsys, tmp_path):
    """Review of 8242b47, P2-4: the demo trains through the real trainer on the
    GPU, so it is a managed GPU job: the lock is held while it trains, and
    released after."""
    from src.utils.gpu_lock import LOCK_FILE_NAME, WORKSPACES_ROOT_ENV, probe

    lock = tmp_path / LOCK_FILE_NAME
    monkeypatch.setenv(WORKSPACES_ROOT_ENV, str(tmp_path))
    seen = {}

    def _trained(workspace, num_epochs, device):
        seen["held_by"] = probe(lock)
        return tmp_path / "demo" / "best.pt"

    monkeypatch.setattr(setup_demo, "train_demo_model", _trained)
    _printed(monkeypatch, capsys, tmp_path, "--train-model")
    assert "a training job (setup_demo.py --train-model" in seen["held_by"]
    assert probe(lock) is None


@pytest.mark.parametrize("case", ["not configured", "busy"])
def test_train_model_is_refused_before_anything_is_written(monkeypatch, tmp_path, caplog, case):
    """Without a configured lock, or with the lock held by another job, the
    demo is refused with exit 1 before the workspace is written or anything
    is trained."""
    import logging

    from src.utils.gpu_lock import LOCK_FILE_NAME, WORKSPACES_ROOT_ENV, GpuLock

    output = tmp_path / "demo"
    monkeypatch.setattr(sys, "argv", ["setup_demo.py", "--output-dir", str(output),
                                      "--train-model"])
    monkeypatch.setattr(setup_demo, "train_demo_model",
                        lambda *a: pytest.fail("trained without the lock"))
    monkeypatch.setattr(setup_demo, "write_workspace",
                        lambda *a, **k: pytest.fail("wrote without the lock"))
    with caplog.at_level(logging.ERROR):
        if case == "not configured":
            monkeypatch.delenv(WORKSPACES_ROOT_ENV, raising=False)
            assert setup_demo.main() == 1
            assert "setup_demo: refused: SHEPHERD_WORKSPACES_ROOT is not set, so the GPU " \
                "lock is not configured" in caplog.text
        else:
            lock = tmp_path / LOCK_FILE_NAME
            monkeypatch.setenv(WORKSPACES_ROOT_ENV, str(tmp_path))
            with GpuLock(lock, job_type="test", command="measure_scorer.py").acquire():
                assert setup_demo.main() == 1
            assert "is held by a test job (measure_scorer.py" in caplog.text
    assert "Nothing was written." in caplog.text and not output.exists()
