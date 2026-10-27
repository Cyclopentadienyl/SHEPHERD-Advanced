"""The model record in a bundle checkpoint (contract §9.1–§9.3; regressions 6, 7).

The training payload writes the record beside M3a's and marks it with record
version 2; the one reader refuses a checkpoint without a model spec (history,
never run), another version, a record that disagrees with its spec, the wrong
stage or task (checked first), another graph view, and weights that do not
match the spec. Both writers carry the record, weights-only files included,
and everything loads with `weights_only=True`.

Module: tests/unit/test_model_record.py
"""
from __future__ import annotations

import dataclasses

import pytest

torch = pytest.importorskip("torch")

from src.models.bundle import build_bundle, finetune_spec_from_parent, reference_spec  # noqa: E402
from src.models.bundle.record import (  # noqa: E402
    MODEL_RECORD_VERSION,
    ModelRecord,
    ModelRecordError,
    load_bundle,
    read_model_record,
)
from src.models.bundle.spec import InputSpec  # noqa: E402
from src.training.callbacks import ModelCheckpoint, ModelCheckpointConfig  # noqa: E402
from src.utils.run_environment import assert_primitive  # noqa: E402

VIEW = {"digest": "e" * 64, "num_nodes": 4, "node_types": ["gene", "phenotype"],
        "relations": ["gene|has|phenotype", "phenotype|rev_has|gene"]}


def _base_spec():
    spec = reference_spec("base")
    return dataclasses.replace(
        spec, input=InputSpec(dim=4),
        encoder=dataclasses.replace(spec.encoder, layers=1, hidden=2, output=4, heads=2)
    ).validate()


def _record(spec=None, **kw):
    spec = spec or _base_spec()
    bundle = build_bundle(spec, VIEW)
    fields = dict(model_spec=spec, graph_view=bundle.view, operation="new", family_id="run-a",
                  data_roles={"kg": "k" * 64, "split": "s" * 64},
                  graph_exposure={"base": {"message_passing": "train_edges"}},
                  loss_identity={"terms": ["lp"], "lp_loss_weight": 1.0})
    fields.update(kw)
    return bundle, ModelRecord(**fields)


def _payload(bundle, record):
    return {"record_version": MODEL_RECORD_VERSION, "state_dict": bundle.state_dict(),
            **record.to_payload()}


def test_round_trip_and_strict_build():
    bundle, record = _record()
    again, read = load_bundle(_payload(bundle, record), learning_stage="base")
    assert read.family_id == "run-a" and read.task is None
    for (n1, t1), (n2, t2) in zip(again.state_dict().items(), bundle.state_dict().items()):
        assert n1 == n2 and torch.equal(t1, t2)


def test_checkpoint_without_spec_is_history():
    with pytest.raises(ModelRecordError, match="written before the model spec"):
        read_model_record({"record_version": 1, "state_dict": {}})


def test_other_version_and_inconsistent_record_refused():
    bundle, record = _record()
    payload = _payload(bundle, record)
    with pytest.raises(ModelRecordError, match="record version"):
        read_model_record({**payload, "record_version": 1})
    with pytest.raises(ModelRecordError, match="differs from its model spec"):
        read_model_record({**payload, "task": "disease"})
    with pytest.raises(ModelRecordError, match="lacks"):
        read_model_record({k: v for k, v in payload.items() if k != "graph_exposure"})


def test_stage_and_task_checked_first_then_view():
    base = _base_spec()
    spec = finetune_spec_from_parent(base, "disease")
    bundle, record = _record(spec, operation="finetune",
                             parent={"sha256": "p" * 64, "run": "run-a"})
    payload = _payload(bundle, record)
    with pytest.raises(ModelRecordError, match="not a base one"):
        load_bundle(payload, learning_stage="base")
    with pytest.raises(ModelRecordError, match="not 'gene'"):
        load_bundle(payload, task="gene")
    with pytest.raises(ModelRecordError, match="graph view"):
        load_bundle(payload, task="disease", view_digest="f" * 64)
    loaded, read = load_bundle(payload, task="disease", view_digest="e" * 64)
    assert read.parent["run"] == "run-a" and loaded.task_head is not None


def test_weights_not_matching_the_spec_refused():
    bundle, record = _record()
    payload = _payload(bundle, record)
    state = dict(payload["state_dict"])
    state.pop("lp_head.relation_weights")
    with pytest.raises(ModelRecordError, match="do not match"):
        load_bundle({**payload, "state_dict": state})


def test_lineage_rules():
    _, resumed = _record(operation="resume")
    with pytest.raises(ModelRecordError, match="parent"):
        resumed.validate()
    _, task_root = _record(finetune_spec_from_parent(_base_spec(), "gene"))
    with pytest.raises(ModelRecordError, match="new run is a base"):
        task_root.validate()


def _trainer(tmp_path, model):
    from src.training.trainer import Trainer, TrainerConfig

    config = TrainerConfig(device="cpu", scheduler_type="cosine", warmup_steps=0, use_amp=False,
                           num_epochs=2, checkpoint_dir=str(tmp_path / "ckpt"),
                           log_dir=str(tmp_path / "logs"))
    return Trainer(model=model, train_dataloader=[{}] * 2, config=config)


def test_both_writers_carry_the_record(tmp_path):
    bundle, record = _record()
    trainer = _trainer(tmp_path, bundle)
    trainer.model_record = record
    by_trainer, by_callback = tmp_path / "t.pt", tmp_path / "c.pt"
    trainer.save_checkpoint(by_trainer)
    ModelCheckpoint(ModelCheckpointConfig(dirpath=str(tmp_path)))._save_checkpoint(
        trainer, by_callback, epoch=0, logs={"val_mrr": 0.1})
    for path in (by_trainer, by_callback):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        assert payload["record_version"] == MODEL_RECORD_VERSION
        assert payload["operation"] == "new" and payload["learning_stage"] == "base"
        load_bundle(payload, learning_stage="base")
        for key, value in payload.items():
            if not key.endswith("state_dict"):
                assert_primitive(value, key)


def test_trainer_without_a_record_keeps_the_m3a_version(tmp_path):
    from src.utils.checkpoint_io import CHECKPOINT_RECORD_VERSION

    trainer = _trainer(tmp_path, torch.nn.Linear(2, 1))
    path = tmp_path / "old.pt"
    trainer.save_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["record_version"] == CHECKPOINT_RECORD_VERSION
    assert "model_spec" not in payload


def test_record_reads_without_torch():
    import subprocess
    import sys

    code = ("import sys; from src.models.bundle.record import read_model_record; "
            "from src.models.bundle import ModelSpec, ViewShape; "
            "assert 'torch' not in sys.modules, 'torch imported'")
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60)
