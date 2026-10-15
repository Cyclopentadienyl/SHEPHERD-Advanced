"""One checkpoint payload for both writers, and what it records (contract M3a-1).

`src/training/checkpoint_payload.py` builds what `ModelCheckpoint` and
`Trainer.save_checkpoint` write; `src/utils/checkpoint_io.write_checkpoint`
writes it whole and returns the SHA-256 of the bytes written. The record is
versioned, carries the settings requested and in effect, the progress counted
three ways, the active stage, how the run stood, and the callbacks' state; its
metadata is plain data, so both writers' files load with `weights_only=True`.

Module: tests/unit/test_checkpoint_payload.py
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.training.callbacks import ModelCheckpoint, ModelCheckpointConfig  # noqa: E402
from src.training.checkpoint_payload import (  # noqa: E402
    CHECKPOINT_RECORD_VERSION,
    TENSOR_KEYS,
    build_checkpoint_payload,
)
from src.utils.checkpoint_io import write_checkpoint  # noqa: E402
from src.utils.run_environment import assert_primitive  # noqa: E402
from tests.fixtures.trainable_workspace import write_trainable_workspace  # noqa: E402


def _tiny_trainer(tmp_path, **config):
    from src.training.trainer import Trainer, TrainerConfig

    defaults = dict(device="cpu", scheduler_type="cosine", warmup_steps=0, use_amp=False,
                    num_epochs=3, checkpoint_dir=str(tmp_path / "ckpt"),
                    log_dir=str(tmp_path / "logs"))
    defaults.update(config)
    return Trainer(model=torch.nn.Linear(2, 1), train_dataloader=[{}] * 4,
                   config=TrainerConfig(**defaults))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# The two writers
# ---------------------------------------------------------------------------
def test_both_writers_write_the_same_keys_and_load_safely(tmp_path):
    trainer = _tiny_trainer(tmp_path)
    trainer.training_input_digests = {"kg": "a" * 64}

    by_trainer = tmp_path / "trainer.pt"
    by_callback = tmp_path / "callback.pt"
    trainer.save_checkpoint(by_trainer)
    ModelCheckpoint(ModelCheckpointConfig(dirpath=str(tmp_path)))._save_checkpoint(
        trainer, by_callback, epoch=0, logs={"val_mrr": 0.5})

    a = torch.load(by_trainer, map_location="cpu", weights_only=True)
    b = torch.load(by_callback, map_location="cpu", weights_only=True)
    assert set(a) == set(b)
    assert a["record_version"] == b["record_version"] == CHECKPOINT_RECORD_VERSION
    for key in ("state_dict", "optimizer_state_dict", "scheduler_state_dict"):
        assert key in a
    assert isinstance(a["state_dict"]["weight"], torch.Tensor)
    for key, value in a.items():
        if key not in TENSOR_KEYS:
            assert_primitive(value, key)


def test_the_writer_returns_the_digest_of_the_bytes_written(tmp_path):
    trainer = _tiny_trainer(tmp_path)
    path = tmp_path / "x.pt"

    digest = trainer.save_checkpoint(path)

    assert digest == _sha(path)
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_file()) == ["x.pt"]


def test_a_failed_write_leaves_the_old_file_and_no_staging_file(tmp_path, monkeypatch):
    path = tmp_path / "last.pt"
    first = write_checkpoint({"a": 1}, path)

    def _refuse(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", _refuse)
    with pytest.raises(OSError, match="disk full"):
        write_checkpoint({"a": 2}, path)

    assert _sha(path) == first
    assert [p.name for p in tmp_path.iterdir()] == ["last.pt"]


def test_metadata_that_is_not_plain_data_is_never_written(tmp_path):
    trainer = _tiny_trainer(tmp_path)
    trainer.run_record = {"requested": {"data_dir": Path("/somewhere")}}

    with pytest.raises(TypeError, match="requested"):
        trainer.save_checkpoint(tmp_path / "x.pt")
    assert not (tmp_path / "x.pt").exists()


# ---------------------------------------------------------------------------
# What the record says
# ---------------------------------------------------------------------------
def test_a_library_trainer_records_what_it_knows_and_none_for_the_rest(tmp_path):
    trainer = _tiny_trainer(tmp_path)

    payload = build_checkpoint_payload(trainer, epoch=0, logs={})

    assert payload["producer"] is None and payload["environment"] is None
    effective = payload["config"]["effective"]
    assert effective["device"] == {"requested": None, "effective": "cpu"}
    assert effective["amp"] == {"requested": False, "dtype_requested": "float16",
                                "enabled": False, "dtype": "float32", "grad_scaler": False}
    schedule = effective["schedule"]
    assert schedule["type"] == "cosine" and schedule["total_steps"] == 12
    assert schedule["steps_per_epoch"] == 4
    assert schedule["warmup_steps_effective"] == 1  # the curve uses at least one


def test_amp_requested_on_cpu_is_recorded_as_not_enabled(tmp_path):
    trainer = _tiny_trainer(tmp_path, use_amp=True, amp_dtype="bfloat16")

    amp = build_checkpoint_payload(trainer, epoch=0, logs={})["config"]["effective"]["amp"]

    assert amp["requested"] is True and amp["dtype_requested"] == "bfloat16"
    assert amp["enabled"] is False and amp["dtype"] == "float32"


def test_the_run_record_reaches_the_checkpoint(tmp_path):
    trainer = _tiny_trainer(tmp_path)
    trainer.run_record = {
        "producer": {"state": "unrecorded", "revision": None, "dirty": None,
                     "modified_tracked": None, "untracked": None},
        "environment": {"python": "3.12.0"},
        "requested": {"device": "auto", "batch_size": 8},
        "device_requested": "auto",
        "compile": {"requested": True, "installed": False},
        "loaders": {"train": {"num_neighbors_effective": [15, 10]}, "val": None},
    }

    payload = build_checkpoint_payload(trainer, epoch=0, logs={})

    assert payload["producer"]["state"] == "unrecorded"
    assert payload["environment"] == {"python": "3.12.0"}
    assert payload["config"]["requested"] == {"device": "auto", "batch_size": 8}
    assert payload["config"]["effective"]["device"]["requested"] == "auto"
    assert payload["config"]["effective"]["compile"] == {"requested": True, "installed": False}
    assert payload["config"]["effective"]["loaders"]["train"]["num_neighbors_effective"] == [15, 10]


def test_metrics_given_as_tensors_are_recorded_as_numbers(tmp_path):
    trainer = _tiny_trainer(tmp_path)

    logs = build_checkpoint_payload(
        trainer, epoch=0, logs={"val_mrr": torch.tensor(0.25), "train_loss": 1.5})["logs"]

    assert logs == {"val_mrr": 0.25, "train_loss": 1.5}
    assert type(logs["val_mrr"]) is float


def test_progress_and_the_end_state(tmp_path):
    trainer = _tiny_trainer(tmp_path, num_epochs=3)
    trainer.state.global_step, trainer.state.optimizer_updates = 7, 6
    trainer.state.scheduler_steps = 6

    trainer.state.completed_epochs = 2
    in_progress = build_checkpoint_payload(trainer, epoch=1, logs={})["progress"]
    assert in_progress["end"] is None and in_progress["next_epoch"] == 2
    assert (in_progress["optimizer_attempts"], in_progress["optimizer_updates"],
            in_progress["scheduler_steps"]) == (7, 6, 6)
    assert in_progress["stage"]["kind"] == "original"
    assert in_progress["stage"]["target_epochs"] == 3

    trainer.state.completed_epochs = 3
    assert build_checkpoint_payload(trainer, epoch=2, logs={})["progress"]["end"] == "finished"

    early = next(c for c in trainer.callbacks.callbacks if type(c).__name__ == "EarlyStopping")
    trainer.state.completed_epochs = 2
    early.should_stop = True
    stopped = build_checkpoint_payload(trainer, epoch=1, logs={})["progress"]
    assert stopped["end"] == "stopped_early"
    assert "no improvement" in stopped["stop_reason"]


def test_a_weights_only_checkpoint_keeps_the_provenance_and_nothing_to_resume(tmp_path):
    trainer = _tiny_trainer(tmp_path)
    trainer.training_input_digests = {"kg": "b" * 64}

    payload = build_checkpoint_payload(trainer, epoch=0, logs={}, weights_only=True)

    assert set(payload) == {"state_dict", "training_input_digests", "producer", "environment"}


# ---------------------------------------------------------------------------
# A real run, end to end
# ---------------------------------------------------------------------------
def test_a_run_records_its_progress_best_k_and_where_the_best_weights_are(tmp_path):
    import scripts.train_model as train_model

    workspace = write_trainable_workspace(tmp_path / "ws")
    run = tmp_path / "run"
    config = train_model.TrainConfig(
        data_dir=str(workspace), output_dir=str(run / "outputs"), log_dir=str(run / "logs"),
        checkpoint_dir=str(run / "checkpoints"), device="cpu", hidden_dim=16,
        num_layers=1, num_heads=2, num_workers=0, use_amp=False, num_epochs=2,
        batch_size=2, save_top_k=2, early_stopping_patience=5,
    )

    train_model.train(config)

    last = torch.load(run / "checkpoints" / "last.pt", map_location="cpu", weights_only=True)
    progress = last["progress"]
    assert progress["completed_epochs"] == 2 and progress["end"] == "finished"
    assert progress["optimizer_attempts"] == progress["optimizer_updates"] > 0
    assert progress["scheduler_steps"] == progress["optimizer_updates"]
    assert last["producer"]["state"] in {"recorded", "unrecorded"}
    assert last["environment"]["torch"] == str(torch.__version__)
    assert last["config"]["requested"]["device"] == "cpu"
    loaders = last["config"]["effective"]["loaders"]
    assert loaders["train"]["num_neighbors_effective"] == [15, 10]
    assert loaders["val"]["num_negative_samples"] == 5

    best_k = last["callbacks"]["model_checkpoint"]["best_k"]
    assert best_k, "the run saved no best-k file"
    for entry in best_k:
        assert entry["sha256"] == _sha(run / "checkpoints" / entry["file"])
    source = last["callbacks"]["best_weights_source"]
    early = last["callbacks"]["early_stopping"]
    assert source is not None
    assert any(e["file"] == source["file"] and e["epoch"] == early["best_epoch"]
               for e in best_k)
