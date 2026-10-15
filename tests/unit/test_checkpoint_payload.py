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

    assert set(payload) == {"record_version", "checkpoint_kind", "weights_from", "state_dict",
                            "training_input_digests", "producer", "environment"}
    assert payload["checkpoint_kind"] == "weights"
    assert payload["weights_from"] == {"kind": "latest", "epoch": 0}


# ---------------------------------------------------------------------------
# A real run, end to end
# ---------------------------------------------------------------------------
def test_a_run_records_its_progress_best_k_and_where_the_best_weights_are(tmp_path):
    import scripts.train_model as train_model

    workspace = write_trainable_workspace(tmp_path / "ws")
    run = tmp_path / "run"
    config = train_model.TrainConfig(
        data_dir=str(workspace), run_dir=str(run), device="cpu", hidden_dim=16,
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
    assert not {"data_dir", "output_dir", "resume_from"} & set(last["config"]["requested"])
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


# ---------------------------------------------------------------------------
# The best weights (review P2-1) and one step's state (review P2-2)
# ---------------------------------------------------------------------------
def _callback(trainer, name):
    return next(c for c in trainer.callbacks.callbacks if type(c).__name__ == name)


def _end_epoch(trainer, epoch, value):
    """What `Trainer.train` does at an epoch's end, with the weights set to a
    value of the epoch's own so every file's weights can be told apart."""
    with torch.no_grad():
        trainer.model.weight.fill_(float(epoch))
    trainer.state.epoch = epoch
    trainer.state.completed_epochs = epoch + 1
    trainer.callbacks.on_epoch_end(trainer, epoch, {"val_mrr": value})


def test_early_stoppings_best_file_survives_a_top_k_it_no_longer_ranks_in(tmp_path):
    """The reviewer's case: improvements below `min_delta` rank later epochs
    above early stopping's best, and pruning used to delete epoch 0's file, the
    one holding the weights early stopping restores and a resume needs."""
    trainer = _tiny_trainer(tmp_path, num_epochs=4, save_top_k=3)
    trainer.callbacks.on_train_begin(trainer)
    for epoch, value in enumerate([0.5, 0.50002, 0.50004, 0.50006]):
        _end_epoch(trainer, epoch, value)

    early = _callback(trainer, "EarlyStopping")
    assert early.best_epoch == 0, "the case needs early stopping to keep epoch 0"
    checkpoint_dir = tmp_path / "ckpt"
    kept = sorted(p.name for p in checkpoint_dir.glob("model-*.pt"))
    assert kept[0].startswith("model-00-") and len(kept) == 4

    last = torch.load(checkpoint_dir / "last.pt", map_location="cpu", weights_only=True)
    source = last["callbacks"]["best_weights_source"]
    assert source["kind"] == "file" and source["file"] == kept[0]
    best = checkpoint_dir / source["file"]
    assert source["sha256"] == _sha(best)
    best_weights = torch.load(best, map_location="cpu", weights_only=True)["state_dict"]
    assert torch.all(best_weights["weight"] == 0.0), "the file holds epoch 0's weights"


def test_the_first_best_file_says_its_own_weights_are_the_best(tmp_path):
    trainer = _tiny_trainer(tmp_path, num_epochs=2)
    trainer.callbacks.on_train_begin(trainer)

    _end_epoch(trainer, 0, 0.5)

    first = next((tmp_path / "ckpt").glob("model-00-*.pt"))
    record = torch.load(first, map_location="cpu", weights_only=True)
    assert record["callbacks"]["best_weights_source"] == {"kind": "self"}
    assert record["callbacks"]["early_stopping"]["best_epoch"] == 0


def test_a_model_holding_restored_best_weights_is_written_as_weights_only(tmp_path):
    """After `train`, early stopping has put epoch 0's weights back while the
    optimizer and progress stay at the last step. Written together they would
    be a state the run never passed through; they are written as weights, and
    say which ones."""
    from src.training.trainer import Trainer, TrainerConfig

    model = torch.nn.Linear(1, 1, bias=False)
    trainer = Trainer(
        model=model, train_dataloader=[],
        optimizer=torch.optim.SGD(model.parameters(), lr=1.0, momentum=0.9),
        config=TrainerConfig(device="cpu", scheduler_type="none", use_amp=False,
                             num_epochs=2, checkpoint_dir=str(tmp_path / "ckpt"),
                             log_dir=str(tmp_path / "logs")),
    )
    trainer.callbacks.on_train_begin(trainer)
    for epoch, value in enumerate([0.9, 0.1]):
        # One SGD step per epoch, so the momentum buffer moves with the weights.
        model.weight.grad = torch.ones_like(model.weight)
        trainer.optimizer.step()
        trainer.optimizer.zero_grad()
        trainer.state.epoch, trainer.state.completed_epochs = epoch, epoch + 1
        trainer.callbacks.on_epoch_end(trainer, epoch, {"val_mrr": value})
    latest_weight = model.weight.detach().clone()
    latest_momentum = trainer.optimizer.state_dict()["state"][0]["momentum_buffer"].clone()

    trainer.callbacks.on_train_end(trainer)
    assert not torch.equal(model.weight, latest_weight), "early stopping restored epoch 0"
    trainer.save_checkpoint(tmp_path / "after_train.pt")

    after = torch.load(tmp_path / "after_train.pt", map_location="cpu", weights_only=True)
    assert after["checkpoint_kind"] == "weights"
    assert after["weights_from"] == {"kind": "best", "epoch": 0}
    assert "optimizer_state_dict" not in after and "progress" not in after

    # `last.pt`, written at the last epoch's end, is one step's state.
    last = torch.load(tmp_path / "ckpt" / "last.pt", map_location="cpu", weights_only=True)
    assert last["checkpoint_kind"] == "training_state"
    assert torch.equal(last["state_dict"]["weight"], latest_weight)
    assert torch.equal(last["optimizer_state_dict"]["state"][0]["momentum_buffer"],
                       latest_momentum)
    assert last["progress"]["completed_epochs"] == 2
