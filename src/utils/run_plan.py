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

It sits in `src.utils`, beside the run layout (`checkpoint_paths`), and imports
nothing from the training stack, so the training service and the console can
preview a selection without it (`.import-linter.ini`: the WebUI must not import
`src.training`). It reads no file itself except to see that a named
best-weights file exists.

Module: src/utils/run_plan.py
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Collection, Dict, List, Mapping, Optional

from src.utils.checkpoint_io import CHECKPOINT_RECORD_VERSION

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
    #: The parent's best-k entries, each with its file resolved to a path, and
    #: the parent itself when it is the file of early stopping's best epoch
    #: (`_locate_best_weights`). They belong to the run that wrote them: ranked
    #: here, never deleted.
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
    parent_sha256: Optional[str] = None,
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
        parent_sha256: the digest of the read `parent` came from. The parent
            is named by it when it is itself the file of early stopping's best
            epoch, so the child's checkpoints can name it in turn.

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
    if mode == "resume" and request.get("extend_lr") is not None:
        raise RunPlanRefused("extend_lr applies only when extending a finished stage.")

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
        schedule = _fixed_rates(parent, request.get("extend_lr"), where)
        run_stage = {"kind": "extension", "start_epoch": completed,
                     "target_epochs": extend_epochs, "schedule": schedule}
        start_epoch, end_epoch = completed, completed + extend_epochs
        action = (f"Extend training by {extend_epochs} epochs at {_rates_text(schedule['lrs'])} "
                  f"(epochs {completed + 1}-{completed + extend_epochs}, new target "
                  f"{completed + extend_epochs})")

    pause = request.get("pause_after_epoch")
    if pause is not None and int(pause) <= start_epoch:
        raise RunPlanRefused(
            f"pause_after_epoch {pause} is not after this run's start (epoch {start_epoch} "
            "is already done).")

    best_weights, best_k = _locate_best_weights(parent, parent_path, where, parent_sha256)
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


def _rates_text(rates: List[float]) -> str:
    """The extension's rate in words: one rate when every parameter group
    holds the same ("a fixed rate of 1e-05"), each group's when they differ
    (M5-3: two groups at one rate read "a fixed rate of 1e-05, 1e-05")."""
    shown = [f"{rate:.3g}" for rate in rates]
    if len(set(shown)) == 1:
        return f"a fixed rate of {shown[0]}"
    return f"fixed rates of {', '.join(shown)} (one per parameter group)"


def _fixed_rates(
    parent: Mapping[str, Any], extend_lr: Optional[float], where: str,
) -> Dict[str, Any]:
    """An extension's schedule (contract M3a-4): each parameter group's rate as
    the parent's optimizer holds it, held fixed. No warmup, no recomputed curve,
    no cosine past its end, no OneCycle stepped past its total.

    **A zero rate is never replaced by a guess.** A schedule that decayed to 0
    would extend without training; `extend_lr` replaces a zero rate, and only a
    zero rate, and the groups it applied to are recorded.
    """
    groups = (parent.get("optimizer_state_dict") or {}).get("param_groups") or []
    if not groups:
        raise RunPlanRefused(f"{where} records no optimizer rates to extend at.")
    rates = [float(group["lr"]) for group in groups]
    zero = [i for i, rate in enumerate(rates) if rate == 0.0]
    if extend_lr is not None:
        if not zero:
            raise RunPlanRefused(
                f"extend_lr replaces a rate that ended at 0, and none did (the rates are "
                f"{rates}). Extend without it to continue at those rates.")
        if not float(extend_lr) > 0:
            raise RunPlanRefused(f"extend_lr must be above 0 (got {extend_lr}).")
        rates = [float(extend_lr) if i in zero else rate for i, rate in enumerate(rates)]
    elif zero:
        raise RunPlanRefused(
            f"{where} ended with a rate of 0 in parameter group(s) {zero}; extending at 0 "
            "would not train them. Give extend_lr for those groups; no rate is guessed.")
    return {"type": "constant", "lrs": rates,
            "zero_rate_groups_set": zero if extend_lr is not None else [],
            "extend_lr": float(extend_lr) if extend_lr is not None else None}


def _locate_best_weights(
    parent: Mapping[str, Any], parent_path: Path, where: str,
    parent_sha256: Optional[str] = None,
) -> "tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]":
    """Where early stopping's best weights are, and the parent's best-k
    entries, each resolved to a path.

    **A parent that is itself the best epoch's file is carried as a best-k
    entry** (contract M3a review of `094b90d`, P2-1). `ModelCheckpoint` writes a
    best-k file before listing it, so that file records `best_weights_source`
    `{"kind": "self"}` and a best-k list without itself. Continued from it, the
    child restores the parent's own weights; unless the parent is also listed,
    the child's checkpoints cannot name where those weights are, and a
    grandchild would lose them. The entry names the parent by this read's
    digest, so the next generation checks it like any other best-k file.

    **Best weights the record says must be restored, and cannot be located,
    are refused**, never treated as absent.
    """
    callbacks = parent.get("callbacks") or {}
    checkpoint_dir = parent_path.parent
    model_checkpoint = callbacks.get("model_checkpoint") or {}
    best_k = []
    for entry in (model_checkpoint.get("best_k") or []):
        best_k.append({**entry, "path": resolve_recorded_file(entry["file"], checkpoint_dir)})

    early = callbacks.get("early_stopping") or {}
    restores = bool(early.get("restore_best_weights")) and early.get("best_value") is not None
    source = callbacks.get("best_weights_source")
    if source is None:
        if restores:
            raise RunPlanRefused(
                f"{where} records early stopping's best {early.get('monitor')} "
                f"({early.get('best_value')}, epoch {early.get('best_epoch')}) to be restored "
                "at the end of training, and names no file holding those weights. Continuing "
                "would lose them; choose another checkpoint.")
        return None, best_k
    if source.get("kind") == "self":
        best_epoch = int(early.get("best_epoch", 0))
        if model_checkpoint and not any(int(e["epoch"]) == best_epoch for e in best_k):
            if not parent_sha256:
                raise RunPlanRefused(
                    f"{where} holds early stopping's best weights itself and cannot be named "
                    "for the run's checkpoints without the digest of its read.")
            monitor = model_checkpoint.get("monitor")
            value = (parent.get("logs") or {}).get(monitor)
            if value is None and monitor == early.get("monitor"):
                value = early.get("best_value")
            best_k.append({"file": parent_path.name, "value": value, "epoch": best_epoch,
                           "sha256": parent_sha256, "path": parent_path})
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
