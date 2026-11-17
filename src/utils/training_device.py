"""
The device a run is given: one rule for training, measurement and serving
=========================================================================
**`auto` means CUDA, and without CUDA it is refused**: which requests need
CUDA (`wants_cuda`: ``auto``, ``cuda``, ``cuda:N``) and whether this host has
it (`cuda_visible`) are decided here once (contract M3a/M3c, §5.2; UI ledger
V-18). Three places ask it, each refusing in its own words with its own
guards: training (`resolve_training_device`, below), measurement
(`scripts/measure_scorer.py`, `_resolve_device`: its ``--device`` sentence)
and serving (`src.inference.pipeline.resolve_serving_device`: plain words
with a trailing Ref, after its own PyTorch check; blank means ``auto``).

`resolve_training_device` is the rule for the device a model bundle
run trains on (contract M3a, §5.2): the training process resolves its run's
device by it (`scripts/train_model.py`, `train`) before any input is read,
and the job runner resolves the device of every start it makes by it before
anything is spawned (`TrainingManager.start_training`, `start_finetune`: a
new base, a continuation, a branch and the Fine-tune children; and a Test
run's start, `start_test`), so one request is refused in one sentence
wherever it is sent, and a start the process would refuse starts nothing
(ops-wiring round, decision S4).

**One refusal, said in the words of where it is asked** (UI ledger R-32,
NEW-training-finetune-1): the command line's sentence names its flag
(``--device cpu``); a page and an HTTP request name their field, Device
(`unavailable_words`: "Device auto needs CUDA, which is not available here;
choose cpu for development."), which the job runner's refusal says
(`TrainingDeviceUnavailable.words`). A page says it before any start too, from
`cuda_visible` (the WebUI's shared Device note).

It sits in `src.utils` so the job runner and serving can ask it without
importing the training stack, and imports torch only when it is asked.

Module: src/utils/training_device.py
"""
from __future__ import annotations


def wants_cuda(requested: str) -> bool:
    """Whether the device `requested` needs CUDA: ``auto`` (CUDA, never the
    CPU in its place), ``cuda`` and ``cuda:N``."""
    return requested == "auto" or requested == "cuda" or requested.startswith("cuda:")


def cuda_visible() -> bool:
    """Whether this host has a CUDA device torch can use: without one, every
    device that `wants_cuda` is refused (`resolve_training_device`)."""
    import torch

    return bool(torch.cuda.is_available())


def unavailable_words(requested: str) -> str:
    """The refusal of `requested` on a host without CUDA in a request's words,
    its field named as a page and an HTTP request name it (Device), never the
    command line's flag."""
    return (f"Device {requested} needs CUDA, which is not available here; choose cpu for "
            "development.")


class TrainingDeviceUnavailable(RuntimeError):
    """The device a run asked for cannot be had on this host. Its message is
    the command line's sentence (the training process's refusal); `words` is
    the same refusal in a request's words (`unavailable_words`), which the job
    runner says to a page or an HTTP request."""

    def __init__(self, requested: str) -> None:
        self.requested = requested
        super().__init__(
            f"--device {requested} requires CUDA, which is not available here. "
            "Hardware acceleration is a hard requirement for deployment and CUDA "
            "is currently the only supported backend, so training does not "
            "substitute the CPU for it. Pass --device cpu explicitly for "
            "development; the run records the device it was given."
        )

    @property
    def words(self) -> str:
        """This refusal in a request's words (`unavailable_words`)."""
        return unavailable_words(self.requested)


def resolve_training_device(requested: str) -> str:
    """The device training runs on, or a refusal (contract M3a, §5.2).

    **`auto` means CUDA, and without CUDA it refuses**, by the rule that
    measurement (`scripts/measure_scorer.py`, `_resolve_device`) and serving
    (`resolve_serving_device`) ask too (`wants_cuda`, `cuda_visible`). It used
    to become CPU silently, so a run asked to train on the deployment's hardware trained
    somewhere else and said nothing; the checkpoint recorded only where it ran.
    An explicit `cpu` stays available for development and is recorded as asked.

    A request for CUDA on a host without it refuses here too, before any input
    is read or any output exists, instead of failing later when the model is
    moved. Other device strings are handed to torch as given.
    """
    if wants_cuda(requested) and not cuda_visible():
        raise TrainingDeviceUnavailable(requested)
    return "cuda" if requested == "auto" else requested


__all__ = ["TrainingDeviceUnavailable", "cuda_visible", "resolve_training_device",
           "unavailable_words", "wants_cuda"]
