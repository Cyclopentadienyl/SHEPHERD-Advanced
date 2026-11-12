"""
The device a training run is given, resolved once
=================================================
`resolve_training_device` is **the one rule** for the device a model bundle
run trains on (contract M3a, §5.2): the training process resolves its run's
device by it (`scripts/train_model.py`, `train`) before any input is read,
and the job runner resolves the device of every start it makes by it before
anything is spawned (`TrainingManager.start_training`, `start_finetune`: a
new base, a continuation, a branch and the Fine-tune children), so one
request is refused in one sentence wherever it is sent, and a start the
process would refuse starts nothing (ops-wiring round, decision S4).

It sits in `src.utils` so the job runner can ask it without importing the
training stack, and imports torch only when it is asked.

Module: src/utils/training_device.py
"""
from __future__ import annotations


class TrainingDeviceUnavailable(RuntimeError):
    """The device a run asked for cannot be had on this host."""


def resolve_training_device(requested: str) -> str:
    """The device training runs on, or a refusal (contract M3a, §5.2).

    **`auto` means CUDA, and without CUDA it refuses**, as measurement's does
    (`scripts/measure_scorer.py`, `_resolve_device`). It used to become CPU
    silently, so a run asked to train on the deployment's hardware trained
    somewhere else and said nothing; the checkpoint recorded only where it ran.
    An explicit `cpu` stays available for development and is recorded as asked.

    A request for CUDA on a host without it refuses here too, before any input
    is read or any output exists, instead of failing later when the model is
    moved. Other device strings are handed to torch as given.
    """
    import torch

    wants_cuda = requested == "auto" or requested == "cuda" or requested.startswith("cuda:")
    if wants_cuda and not torch.cuda.is_available():
        raise TrainingDeviceUnavailable(
            f"--device {requested} requires CUDA, which is not available here. "
            "Hardware acceleration is a hard requirement for deployment and CUDA "
            "is currently the only supported backend, so training does not "
            "substitute the CPU for it. Pass --device cpu explicitly for "
            "development; the run records the device it was given."
        )
    return "cuda" if requested == "auto" else requested


__all__ = ["TrainingDeviceUnavailable", "resolve_training_device"]
