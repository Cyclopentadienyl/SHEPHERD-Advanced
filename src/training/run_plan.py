"""
How a run that starts from a parent checkpoint continues, decided once.
=======================================================================
Contract M3a-3 and M3a-4 (`docs/working/PLAN_PROVENANCE_CONTRACT.md`,
decision 8). A run given a parent (`resume_from`) either **resumes** the
parent's active stage, which is unfinished, or **extends** a finished stage at a
fixed rate. Which of the two, with which settings and from which epoch, is
decided here, before the Trainer and before any output exists, from the
parent's own record, never from its file name or the run it sits in. The CLI,
the training service, the console's preview and the probe all go through
`plan_continuation`; the entry point decides again on the bytes it actually
restores, so a preview is never the check.

**What the parent's state allows:**

- an unfinished stage (`progress.end` is `None` or `"paused"`) is resumed:
  only its remaining epochs run;
- a finished stage (`"finished"`) is extended only when `extend_epochs` is
  given; resuming it would have nothing to run;
- a run that stopped early (`"stopped_early"`) is neither: resuming would reset
  the patience that stopped it, and extending would treat a stop as a pause;
- a weights-only checkpoint, or one without the record contract M3a writes,
  holds no state to continue from.

Each refusal says why. Nothing falls back to training from scratch.

**Learning settings are inherited and locked** (decision 8). Every
`TrainConfig` field outside `OPERATIONAL_FIELDS` takes the parent's recorded
value (`config.requested`). A field the operator set explicitly to another
value is refused, never ignored; one left at its default is inherited and shown
in the plan. Changing settings from a parent is a branch, which M3 does not
build (BACKLOG item 23).

**Provenance and mode rules stay apart.** Whether the parent was trained on
this workspace's data and graph is M2.3's check (`train_model.check_resume_
parent`), applied by the entry point; this module decides only what the parent's
state and settings allow.

The console calls it to preview a selection; it reads no file itself except to
see that a named best-weights file exists.

Module: src/training/run_plan.py
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Collection, Dict, List, Mapping, Optional

from src.training.checkpoint_payload import CHECKPOINT_RECORD_VERSION

#: Fields that say where a run reads and writes, how it is run on this host, or
#: how this run is continued: never inherited, free to differ from the parent.
#: Every other `TrainConfig` field is a learning setting.
OPERATIONAL_FIELDS = frozenset({
    "data_dir", "config_file", "run_name", "run_dir", "resume_from",
    "device", "num_workers", "compile",
    "pause_after_epoch", "extend_epochs", "extend_lr",
})


class RunPlanRefused(ValueError):
    """The parent cannot be continued as requested; the message says why and
    what to do instead."""


@dataclass(frozen=True)
class ContinuationPlan:
    """What a run started from a parent does.

    `settings` is the whole configuration the run uses: the parent's learning
    settings and the request's operational fields. `inherited` lists the
    learning settings taken from the parent that the request did not set
    explicitly, for the read-only summary shown before start.
    """

    mode: str                       # "resume" | "extend"
    action: str                     # one sentence, for people
    settings: Dict[str, Any]
    inherited: Dict[str, Any]
    stage: Dict[str, Any]           # the active stage this run continues or opens
    start_epoch: int                # the next epoch index to run
    end_epoch: int                  # the stage's end, exclusive
    #: Early stopping's best weights: `None`, `{"kind": "self"}` (the parent's
    #: own weights) or `{"kind": "file", "path": Path, "sha256": str}` (a file
    #: the entry point reads and checks by digest).
    best_weights: Optional[Dict[str, Any]]
    #: The parent's best-k entries, each with its file resolved to a path.
    #: They belong to the run that wrote them: ranked here, never deleted.
    best_k: List[Dict[str, Any]] = field(default_factory=list)
    #: The parent's callback state as recorded.
    callbacks: Optional[Dict[str, Any]] = None


def _same(a: Any, b: Any) -> bool:
    """Two setting values equal as settings: numbers by value, sequences
    element-wise, anything else exactly."""
    numbers = (int, float)
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b or a == b and type(a) is type(b)
    if isinstance(a, numbers) and isinstance(b, numbers):
        return float(a) == float(b)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _progress_text(completed: int, start: int, target: int) -> str:
    done = completed - start
    return f"{done}/{target} done, {target - done} left"


def resolve_recorded_file(recorded: str, parent_checkpoint_dir: Path) -> Path:
    """A file a checkpoint names relative to its own run's checkpoint
    directory (a best-k file, the best weights), as a path here."""
    return Path(os.path.normpath(parent_checkpoint_dir / recorded))


def plan_continuation(
    parent: Mapping[str, Any],
    parent_path: Path,
    request: Mapping[str, Any],
    explicit: Collection[str],
    *,
    device: str,
) -> ContinuationPlan:
    """Decide how a run over `request` continues from `parent`.

    Args:
        parent: the parent checkpoint, as read.
        parent_path: where it was read from; files it names (best-k, the best
            weights) are resolved beside it.
        request: the requested configuration, every `TrainConfig` field.
        explicit: the fields the operator set (configuration file keys, CLI
            flags, request fields); a learning setting among them that differs
            from the parent's is refused.
        device: the device the run will use, resolved.

    Raises:
        RunPlanRefused: with the reason and the remedy.
    """
    where = f"The parent {parent_path.name}"
    if not isinstance(parent, Mapping) or parent.get("record_version") != CHECKPOINT_RECORD_VERSION:
        raise RunPlanRefused(
            f"{where} holds no training state this version can continue: it was written "
            "before contract M3a's checkpoint record, so its schedule, progress and "
            "callback state are unknown. Start a new run; to keep its weights as a "
            "reference, serve or measure it by path.")
    if parent.get("checkpoint_kind") != "training_state":
        weights_from = parent.get("weights_from") or {}
        raise RunPlanRefused(
            f"{where} holds weights only ({weights_from.get('kind', 'unknown')} weights of "
            f"epoch {weights_from.get('epoch', '?')}), without the optimizer, schedule or "
            "progress a run continues from. Choose the run's last.pt or another "
            "training-state checkpoint.")

    progress = parent.get("progress") or {}
    stage = dict(progress.get("stage") or {})
    completed = int(progress.get("completed_epochs", 0))
    start = int(stage.get("start_epoch", 0))
    target = int(stage.get("target_epochs", 0))
    end = progress.get("end")
    kind = stage.get("kind", "original")
    stage_name = "extended training" if kind == "extension" else "original training"
    extend_epochs = request.get("extend_epochs")

    if end == "stopped_early":
        raise RunPlanRefused(
            f"{where} stopped early at epoch {completed} ({progress.get('stop_reason')}). "
            "Resuming would reset the patience that stopped it, and extending would treat "
            "a stop as a pause, so neither is offered. Serve or measure it, or start a new "
            "run.")
    if end == "finished":
        if not extend_epochs:
            raise RunPlanRefused(
                f"{where} finished its {stage_name} ({target}/{target} epochs), so there is "
                "nothing left to resume. To train further, extend it at a fixed rate with "
                "extend_epochs.")
        mode = "extend"
    elif end in (None, "paused"):
        if extend_epochs:
            raise RunPlanRefused(
                f"{where} has not finished its {stage_name} "
                f"({_progress_text(completed, start, target)}); extend_epochs applies only to "
                "a finished stage. Resume it without extend_epochs.")
        mode = "resume"
    else:
        raise RunPlanRefused(f"{where} records an unknown end state {end!r}.")

    settings, inherited = _inherit_settings(parent, request, explicit, where)
    _check_amp_takes_effect_alike(parent, settings, device, where)

    if mode == "resume":
        if stage.get("schedule", {}).get("type") == "external":
            raise RunPlanRefused(
                f"{where} was trained with a scheduler built outside the Trainer; its "
                "schedule cannot be rebuilt from the record.")
        run_stage = stage
        start_epoch, end_epoch = completed, start + target
        action = (f"Resume the {stage_name} ({_progress_text(completed, start, target)})")
    else:
        extend_epochs = int(extend_epochs)
        if extend_epochs < 1:
            raise RunPlanRefused(f"extend_epochs must be at least 1 (got {extend_epochs}).")
        # The fixed rates are the optimizer's after the restore; the entry
        # point fills them in (`Trainer.continue_from`), they are not guessed.
        run_stage = {"kind": "extension", "start_epoch": completed,
                     "target_epochs": extend_epochs, "schedule": None}
        start_epoch, end_epoch = completed, completed + extend_epochs
        action = (f"Extend training by {extend_epochs} epochs at a fixed rate "
                  f"(epochs {completed + 1}-{completed + extend_epochs})")

    pause = request.get("pause_after_epoch")
    if pause is not None and int(pause) <= start_epoch:
        raise RunPlanRefused(
            f"pause_after_epoch {pause} is not after this run's start (epoch {start_epoch} "
            "is already done).")

    best_weights, best_k = _locate_best_weights(parent, parent_path, where)
    return ContinuationPlan(
        mode=mode, action=action, settings=settings, inherited=inherited,
        stage=run_stage, start_epoch=start_epoch, end_epoch=end_epoch,
        best_weights=best_weights, best_k=best_k, callbacks=parent.get("callbacks"),
    )


def _inherit_settings(
    parent: Mapping[str, Any], request: Mapping[str, Any], explicit: Collection[str],
    where: str,
) -> "tuple[Dict[str, Any], Dict[str, Any]]":
    recorded = (parent.get("config") or {}).get("requested")
    if not isinstance(recorded, Mapping):
        raise RunPlanRefused(
            f"{where} does not record the settings it was trained with, so they cannot "
            "be inherited.")
    unknown = sorted(set(recorded) - set(request))
    if unknown:
        raise RunPlanRefused(
            f"{where} records settings this version does not have ({', '.join(unknown)}); "
            "continuing would silently drop them.")

    settings = dict(request)
    inherited: Dict[str, Any] = {}
    conflicts: List[str] = []
    missing: List[str] = []
    for name in request:
        if name in OPERATIONAL_FIELDS:
            continue
        if name not in recorded:
            missing.append(name)
            continue
        value = recorded[name]
        if name in explicit and not _same(request[name], value):
            conflicts.append(f"{name}: the parent's {value!r}, requested {request[name]!r}")
            continue
        settings[name] = value
        if name not in explicit:
            inherited[name] = value
    if missing:
        raise RunPlanRefused(
            f"{where} does not record {', '.join(sorted(missing))}, so the run cannot "
            "inherit them.")
    if conflicts:
        raise RunPlanRefused(
            f"{where} was trained with other settings than this request sets explicitly: "
            + "; ".join(conflicts) + ". A continued run keeps every learning setting of "
            "its parent; remove these to inherit them. Starting from a parent with "
            "changed settings (a branch) is not available yet.")
    return settings, inherited


def _check_amp_takes_effect_alike(
    parent: Mapping[str, Any], settings: Mapping[str, Any], device: str, where: str,
) -> None:
    """Mixed precision changes how a run learns, and takes effect only on CUDA:
    a device change that turns it on or off is a settings change."""
    amp = ((parent.get("config") or {}).get("effective") or {}).get("amp") or {}
    recorded = amp.get("enabled")
    if recorded is None:
        return
    here = bool(settings.get("use_amp")) and str(device).startswith("cuda")
    if bool(recorded) != here:
        raise RunPlanRefused(
            f"{where} trained with mixed precision {'on' if recorded else 'off'} "
            f"({amp.get('dtype')}); on {device} it would be {'on' if here else 'off'}. "
            "Continue it on a device where mixed precision takes effect as it did.")


def _locate_best_weights(
    parent: Mapping[str, Any], parent_path: Path, where: str,
) -> "tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]":
    callbacks = parent.get("callbacks") or {}
    checkpoint_dir = parent_path.parent
    best_k = []
    for entry in ((callbacks.get("model_checkpoint") or {}).get("best_k") or []):
        best_k.append({**entry, "path": resolve_recorded_file(entry["file"], checkpoint_dir)})

    source = callbacks.get("best_weights_source")
    if source is None:
        return None, best_k
    if source.get("kind") == "self":
        return {"kind": "self"}, best_k
    path = resolve_recorded_file(source["file"], checkpoint_dir)
    if not path.is_file():
        raise RunPlanRefused(
            f"{where} names {source['file']} as early stopping's best weights, and it is "
            f"not at {path}. Without it the run could not restore its best weights at the "
            "end; put the file back (its SHA-256 is "
            f"{source.get('sha256')}) or choose another checkpoint.")
    return {"kind": "file", "path": path, "sha256": source.get("sha256")}, best_k


__all__ = [
    "ContinuationPlan",
    "OPERATIONAL_FIELDS",
    "RunPlanRefused",
    "plan_continuation",
    "resolve_recorded_file",
]
