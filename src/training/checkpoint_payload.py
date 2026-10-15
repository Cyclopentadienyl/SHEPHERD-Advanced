"""
What a training checkpoint holds, built in one place.
=====================================================
Contract M3a-1 (`docs/working/PLAN_PROVENANCE_CONTRACT.md`). The two writers,
`ModelCheckpoint` (every epoch's `last.pt` and the best-k files) and
`Trainer.save_checkpoint`, each listed their own keys, so a checkpoint's content
depended on which wrote it: the callback's had no scaler, no global step and no
trainer state, and a key added to one was missing from the other. Both now call
`build_checkpoint_payload`, and `src.utils.checkpoint_io.write_checkpoint`
writes the result.

**The record is versioned** (`record_version`) and says enough of the run's
history to resume it: the settings in effect beside the ones requested, the
progress counted three ways (optimizer attempts, successful updates and
scheduler steps, which differ when AMP skips an update), the active stage, how
the run stood when this was written, and the callbacks' state. What restores
from it is M3a-3's; this module only writes it.

**Plain data around the tensors.** The model's, optimizer's, scheduler's and
scaler's state dicts stay as torch made them; every other value is a string, a
number, a boolean, `None`, or a dict or list of those
(`src.utils.run_environment.assert_primitive`), so the whole payload loads with
`torch.load(..., weights_only=True)`.

Module: src/training/checkpoint_payload.py
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Mapping

from src.utils.checkpoint_io import CHECKPOINT_RECORD_VERSION
from src.utils.run_environment import assert_primitive

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.training.trainer import Trainer

#: Keys whose values are torch state, written as torch made them.
TENSOR_KEYS = frozenset({
    "state_dict", "optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict",
})

#: The run-level facts `scripts/train_model.py` attaches to the trainer as
#: `run_record` before training: what was requested, and what the entry point
#: alone knows took effect. A trainer built without one (a library caller, a
#: test) writes `None` for each, which reads as "not known", never as "none".
RUN_RECORD_KEYS = ("producer", "environment", "requested", "device_requested",
                   "compile", "loaders")


def plain_number(value: Any) -> Any:
    """A metric as a plain float or int: a 0-dim tensor or NumPy scalar is
    converted; a non-finite float stays a float (JSON is not this file's
    format); anything else is returned for `assert_primitive` to judge."""
    if isinstance(value, bool):
        return value
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, (int, float)):
        try:
            value = item()
        except Exception:  # noqa: BLE001 — not a scalar: left for the check
            return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return value


def _plain_logs(logs: Mapping[str, Any]) -> Dict[str, Any]:
    return {str(k): plain_number(v) for k, v in logs.items()}


def _end_state(trainer: "Trainer", completed: int, target: int) -> Dict[str, Any]:
    """How the run stood when the checkpoint was written: stopped early,
    finished its stage, paused by `pause_after_epoch` (contract M3a-3), or
    still running (`None`, which a stop between epochs leaves too)."""
    early = _find_callback(trainer, "EarlyStopping")
    if early is not None and early.should_stop:
        reason = (
            f"early stopping: no improvement in {early.config.patience} epochs "
            f"on {early.config.monitor}"
        )
        return {"end": "stopped_early", "stop_reason": reason}
    if completed >= target:
        return {"end": "finished", "stop_reason": None}
    if getattr(trainer, "pausing", False):
        return {"end": "paused", "stop_reason": f"paused after epoch {completed} (requested)"}
    return {"end": None, "stop_reason": None}


def _find_callback(trainer: "Trainer", name: str) -> Any:
    callbacks = getattr(getattr(trainer, "callbacks", None), "callbacks", None) or []
    for callback in callbacks:
        if type(callback).__name__ == name:
            return callback
    return None


def _callback_state(trainer: "Trainer", epoch: int) -> Dict[str, Any]:
    """Early stopping's and best-k's state, and where the best weights are.

    Early stopping keeps its best weights in memory only, so a checkpoint says
    where they can be had again (`best_weights_source`):

    - **`{"kind": "self"}`** when this checkpoint is written at early stopping's
      best epoch: its own weights are the best weights. This is how the first
      best-k file says it, without carrying its own digest.
    - **`{"kind": "file", "file", "sha256"}`**: the best-k file of that epoch,
      by its path relative to this run's checkpoint directory (its name, for
      this run's own; `../../<run>/checkpoints/<name>` for a parent's, contract
      M3a-3) and the SHA-256 it was written with. `ModelCheckpoint` never prunes that file while early
      stopping counts its epoch as best (contract M3a-1 review, P2-1).
    - **`None`** before any best exists, or when early stopping does not restore
      weights.
    """
    early = _find_callback(trainer, "EarlyStopping")
    checkpoint = _find_callback(trainer, "ModelCheckpoint")

    best_k = None
    if checkpoint is not None:
        best_k = [
            {"file": _relative_to(entry["filepath"], checkpoint.dirpath),
             "value": plain_number(entry["value"]),
             "epoch": int(entry["epoch"]), "sha256": entry.get("sha256")}
            for entry in checkpoint.best_k_models
        ]

    early_state = None
    best_source = None
    if early is not None:
        early_state = {
            "monitor": early.config.monitor,
            "mode": early.config.mode,
            "patience": int(early.config.patience),
            "min_delta": float(early.config.min_delta),
            "restore_best_weights": bool(early.config.restore_best_weights),
            "best_value": plain_number(early.best_value),
            "best_epoch": int(early.best_epoch),
            "counter": int(early.counter),
            "stopped_epoch": int(early.stopped_epoch),
            "should_stop": bool(early.should_stop),
        }
        if early.best_value is not None and early.config.restore_best_weights:
            if int(early.best_epoch) == int(epoch):
                best_source = {"kind": "self"}
            else:
                match = [e for e in (best_k or []) if e["epoch"] == early.best_epoch]
                if match:
                    best_source = {"kind": "file", "file": match[0]["file"],
                                   "sha256": match[0]["sha256"]}
                else:
                    # Recorded as such: `src.utils.run_plan` refuses to continue
                    # from a checkpoint whose best weights cannot be located.
                    logger.warning(
                        "Early stopping's best weights (epoch %s) are in no listed "
                        "best-k file; this checkpoint cannot be continued.", early.best_epoch)

    model_checkpoint_state = None
    if checkpoint is not None:
        model_checkpoint_state = {
            "monitor": checkpoint.config.monitor,
            "mode": checkpoint.config.mode,
            "save_top_k": int(checkpoint.config.save_top_k),
            "best_k": best_k,
        }
    return {
        "early_stopping": early_state,
        "model_checkpoint": model_checkpoint_state,
        "best_weights_source": best_source,
    }


def _relative_to(path: Any, directory: Any) -> str:
    """A file named relative to a checkpoint directory, so a run's records
    travel with the layout of runs side by side and not with this machine's
    absolute paths. Across drives (Windows) there is no relative path, and the
    absolute one is kept."""
    import os

    try:
        return os.path.relpath(str(path), str(directory)).replace(os.sep, "/")
    except ValueError:
        return str(path)


def _progress(trainer: "Trainer") -> Dict[str, Any]:
    state = trainer.state
    stage = dict(trainer.stage_record())
    completed = int(state.completed_epochs)
    target = int(stage["start_epoch"]) + int(stage["target_epochs"])
    return {
        "completed_epochs": completed,
        "next_epoch": completed,
        "optimizer_attempts": int(state.global_step),
        "optimizer_updates": int(state.optimizer_updates),
        "scheduler_steps": int(state.scheduler_steps),
        "stage": stage,
        **_end_state(trainer, completed, target),
    }


def _training_state(trainer: "Trainer") -> Dict[str, Any]:
    state = trainer.state.__dict__
    return {
        "epoch": int(state["epoch"]),
        "global_step": int(state["global_step"]),
        "best_metric": plain_number(state["best_metric"]),
        "best_epoch": int(state["best_epoch"]),
        "train_loss_history": [plain_number(v) for v in state["train_loss_history"]],
        "val_metric_history": [_plain_logs(m) for m in state["val_metric_history"]],
        "completed_epochs": int(state["completed_epochs"]),
        "optimizer_updates": int(state["optimizer_updates"]),
        "scheduler_steps": int(state["scheduler_steps"]),
    }


def build_checkpoint_payload(
    trainer: "Trainer",
    *,
    epoch: int,
    logs: Mapping[str, Any],
    weights_only: bool = False,
) -> Dict[str, Any]:
    """The checkpoint `trainer` writes after epoch index `epoch`.

    `weights_only` keeps the model's weights and the provenance keys, without
    the optimizer, scheduler, scaler or the resume state: a weights-only file
    is still an artifact someone will have to identify, and cannot be resumed.

    Raises:
        TypeError: a metadata value is not plain data; a bug here, never
            written to disk.
    """
    run_record: Mapping[str, Any] = getattr(trainer, "run_record", None) or {}

    # **A training state is one step's model, optimizer and progress together**
    # (contract M3a-1 review, P2-2). After `train` returns, early stopping has
    # put its best weights back into the model while the optimizer, scheduler
    # and progress stay at the last step; written together they would describe
    # a state the run never passed through. Such a model is written as weights
    # only, saying which weights they are.
    restored = getattr(trainer, "weights_restored_from", None)
    if restored is not None and not weights_only:
        logger.warning(
            "The model holds early stopping's best weights (epoch %s) while the "
            "optimizer is at the last step; writing weights only, not a "
            "resumable training state.", restored.get("epoch"))
        weights_only = True

    if weights_only:
        payload: Dict[str, Any] = {
            "record_version": CHECKPOINT_RECORD_VERSION,
            "checkpoint_kind": "weights",
            "weights_from": (dict(restored) if restored is not None
                             else {"kind": "latest", "epoch": int(epoch)}),
            "state_dict": trainer.model.state_dict(),
        }
    else:
        config = dict(trainer._serialize_config())
        config["requested"] = run_record.get("requested")
        config["effective"] = {
            "device": {"requested": run_record.get("device_requested"),
                       "effective": str(trainer.device)},
            "amp": trainer.amp_record(),
            "compile": run_record.get("compile"),
            "loaders": run_record.get("loaders"),
            # The active stage's schedule: the original curve, or an
            # extension's fixed rates (contract M3a-4).
            "schedule": trainer.stage_record().get("schedule"),
        }
        payload = {
            "record_version": CHECKPOINT_RECORD_VERSION,
            "checkpoint_kind": "training_state",
            "epoch": int(epoch),
            "global_step": int(trainer.state.global_step),
            "state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "logs": _plain_logs(logs),
            "config": config,
            "state": _training_state(trainer),
            "progress": _progress(trainer),
            "callbacks": _callback_state(trainer, epoch),
        }
        if trainer.scheduler is not None:
            payload["scheduler_state_dict"] = trainer.scheduler.state_dict()
        if trainer.scaler is not None:
            payload["scaler_state_dict"] = trainer.scaler.state_dict()

    # Provenance, in every checkpoint, weights-only included: which inputs
    # produced it and under which code and environment. Set by the training
    # script; absent for a library caller.
    fingerprint = getattr(trainer, "data_fingerprint", None)
    if fingerprint is not None:
        payload["data_fingerprint"] = fingerprint
    digests = getattr(trainer, "training_input_digests", None)
    if digests is not None:
        payload["training_input_digests"] = dict(digests)
    payload["producer"] = run_record.get("producer")
    payload["environment"] = run_record.get("environment")

    for key, value in payload.items():
        if key not in TENSOR_KEYS:
            assert_primitive(value, key)
    return payload


__all__ = [
    "CHECKPOINT_RECORD_VERSION",
    "RUN_RECORD_KEYS",
    "TENSOR_KEYS",
    "build_checkpoint_payload",
    "plain_number",
]
