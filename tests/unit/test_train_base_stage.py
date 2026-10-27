"""`scripts/train_model.py --stage base` end to end on a small workspace.

The base stage reads the graph alone, forms the view and its split before any
output, trains the bundle through the Trainer with early stopping off, writes
bundle checkpoints with the model record and the split artifact, pauses and
resumes strictly (regression 8), and refuses before writing: ShepherdGNN-only
settings, a parent written before the model spec, a different split.

Module: tests/unit/test_train_base_stage.py
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from src.models.bundle import reference_spec  # noqa: E402
from src.models.bundle.record import MODEL_RECORD_VERSION, load_bundle  # noqa: E402
from src.models.bundle.spec import InputSpec  # noqa: E402
from tests.fixtures.base_workspace import write_base_workspace  # noqa: E402


def _small_spec():
    spec = reference_spec("base")
    return dataclasses.replace(
        spec, input=InputSpec(dim=8),
        encoder=dataclasses.replace(spec.encoder, layers=2, hidden=4, output=4, heads=2)
    ).validate().to_dict()


def _config(data_dir, run_dir, **kw):
    import train_model

    fields = dict(data_dir=str(data_dir), run_dir=str(run_dir), learning_stage="base",
                  model_spec=_small_spec(), num_epochs=2, batch_size=16, num_neighbors=[4, 4],
                  device="cpu", save_top_k=1)
    fields.update(kw)
    explicit = set(fields) - {"data_dir", "run_dir"}
    return train_model.TrainConfig(**fields), explicit


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    return write_base_workspace(tmp_path_factory.mktemp("base"))


def test_base_run_writes_bundle_checkpoints(workspace, tmp_path):
    import train_model

    config, explicit = _config(workspace, tmp_path / "run")
    metrics = train_model.train(config, explicit)
    assert "val_lp_accuracy" in metrics
    run = tmp_path / "run"
    for name in ("config.yaml", "graph_view.json", "graph_split.pt", "final_metrics.json"):
        assert (run / name).is_file(), name
    last = torch.load(run / "checkpoints" / "last.pt", map_location="cpu", weights_only=True)
    assert last["record_version"] == MODEL_RECORD_VERSION
    bundle, record = load_bundle(last, learning_stage="base")
    split = torch.load(run / "graph_split.pt", weights_only=True)
    assert record.data_roles["split"]["digest"] == split["record"]["digest"]
    assert record.operation == "new" and record.family_id == "run"
    digests = last["training_input_digests"]
    assert "train_samples" not in digests and "kg" in digests
    import json

    view_record = json.loads((run / "graph_view.json").read_text())
    assert view_record["source"]["kg"] == digests["kg"]
    assert view_record["digest"] == record.graph_view.digest
    recipe = last["config"]["effective"]["loaders"]["train"]["recipe"]
    assert "learning_rate" in recipe["reference_filled"]
    assert last["config"]["requested"]["learning_rate"] == 1e-4
    assert last["config"]["early_stopping"] is False
    assert not last["config"]["use_amp"]


def test_pause_and_resume_strictly(workspace, tmp_path):
    import train_model

    first, explicit = _config(workspace, tmp_path / "a", num_epochs=3, pause_after_epoch=1)
    train_model.train(first, explicit)
    parent = tmp_path / "a" / "checkpoints" / "last.pt"
    payload = torch.load(parent, map_location="cpu", weights_only=True)
    assert payload["progress"]["end"] == "paused"

    resumed = train_model.TrainConfig(data_dir=str(workspace), run_dir=str(tmp_path / "b"),
                                      learning_stage="base", resume_from=str(parent),
                                      device="cpu")
    train_model.train(resumed, {"learning_stage", "resume_from", "device"})
    child = torch.load(tmp_path / "b" / "checkpoints" / "last.pt", map_location="cpu",
                       weights_only=True)
    _, record = load_bundle(child, learning_stage="base")
    assert record.operation == "resume" and record.family_id == "a"
    assert record.parent["run"] == "a"
    assert child["progress"]["completed_epochs"] == 3
    assert child["model_spec"] == payload["model_spec"]


def test_refusals_before_anything_is_written(workspace, tmp_path):
    import train_model

    config, explicit = _config(workspace, tmp_path / "x", hidden_dim=64)
    with pytest.raises(ValueError, match="ShepherdGNN path"):
        train_model.train(config, explicit | {"hidden_dim"})
    config, explicit = _config(workspace, tmp_path / "y", num_neighbors=[4, 4, 4])
    with pytest.raises(ValueError, match="one hop per layer"):
        train_model.train(config, explicit)
    assert not (tmp_path / "x").exists() and not (tmp_path / "y").exists()


def test_parent_without_a_model_spec_is_refused(workspace, tmp_path):
    import train_model
    from src.utils.run_plan import RunPlanRefused

    config, explicit = _config(workspace, tmp_path / "p")
    train_model.train(config, explicit)
    parent = tmp_path / "p" / "checkpoints" / "last.pt"
    payload = torch.load(parent, map_location="cpu", weights_only=True)
    for key in ("model_spec",):
        payload.pop(key)
    payload["record_version"] = 1
    old = tmp_path / "old.pt"
    torch.save(payload, old)
    resumed = train_model.TrainConfig(data_dir=str(workspace), run_dir=str(tmp_path / "q"),
                                      learning_stage="base", resume_from=str(old), device="cpu")
    with pytest.raises(RunPlanRefused, match="written before the model spec"):
        train_model.train(resumed, {"learning_stage", "resume_from", "device"})
    assert not (tmp_path / "q").exists()


def test_bundle_runs_are_listed_but_not_auto_served(tmp_path):
    from src.utils.checkpoint_paths import list_runs, select_auto_run_checkpoint

    for arch, run in (("gat", "20261001-000000_old_aaaaaa"), ("gatv2", "20261027-000000_base_bbbbbb")):
        ckpt = tmp_path / "runs" / arch / run / "checkpoints"
        ckpt.mkdir(parents=True)
        (ckpt / "last.pt").write_bytes(b"x")
    names = {r.name for r in list_runs(tmp_path)}
    assert names == {"20261001-000000_old_aaaaaa", "20261027-000000_base_bbbbbb"}
    path, arch, _ = select_auto_run_checkpoint(tmp_path)
    assert arch == "gat" and "old" in str(path)
