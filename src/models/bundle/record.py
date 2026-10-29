"""
The model record a bundle checkpoint carries, and the one way to read it.
========================================================================
`docs/working/PLAN_PROVENANCE_CONTRACT.md` §9.1–§9.3. The training payload
(`src.training.checkpoint_payload`) writes these fields beside M3a's record;
serving, measurement and the operations read them through `read_model_record`
and build through `load_bundle`, so no consumer reads a spec its own way.

**Record version 2** marks a checkpoint the bundle path wrote. A checkpoint
without a `model_spec` was written before it: it is listed as history and
never run, and nothing is guessed from its weight names (§9.3, hazard 1).

**Task first** (hazard 3): `load_bundle` compares the learning stage and task a
caller needs with the record before it builds anything.

Plain data only, so the payload still loads with `weights_only=True`.
Reading and checking a record needs no torch; only `load_bundle` builds.

Module: src/models/bundle/record.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from src.config.model_types import MODEL_RECORD_KEYS
from src.models.bundle.spec import LEARNING_STAGES, TASKS, ModelSpec, SpecError, ViewShape

#: The checkpoint `record_version` the bundle path writes (M3a's was 1).
MODEL_RECORD_VERSION = 2
OPERATIONS = ("new", "resume", "extend", "branch", "finetune")

#: Top-level payload keys this record owns (§9.1), defined once beside the
#: encoder kinds so the run listing can carry them without importing this.
RECORD_KEYS = MODEL_RECORD_KEYS


class ModelRecordError(ValueError):
    """A checkpoint the bundle path cannot run, with the reason."""


HISTORY_ONLY = ("written before the model spec: listed as history and never run by this "
                "build; start a new run")


@dataclass
class ModelRecord:
    model_spec: ModelSpec
    graph_view: ViewShape
    operation: str
    family_id: str
    parent: Optional[Dict[str, Any]] = None
    weight_transfer: Optional[Dict[str, Any]] = None
    data_roles: Dict[str, Any] = field(default_factory=dict)
    graph_exposure: Dict[str, Any] = field(default_factory=dict)
    loss_identity: Dict[str, Any] = field(default_factory=dict)
    scoring_identity: Optional[Dict[str, Any]] = None
    selection: Optional[Dict[str, Any]] = None

    @property
    def learning_stage(self) -> str:
        return self.model_spec.learning_stage

    @property
    def task(self) -> Optional[str]:
        return self.model_spec.task

    def validate(self) -> "ModelRecord":
        self.model_spec.validate()
        if self.operation not in OPERATIONS:
            raise ModelRecordError(f"unknown operation {self.operation!r}")
        if not isinstance(self.family_id, str) or not self.family_id:
            raise ModelRecordError("a model record names its family (the root run)")
        if (self.operation == "new") != (self.parent is None):
            raise ModelRecordError("a parent is recorded for every operation but 'new'")
        if self.parent is not None and not (
                isinstance(self.parent, dict) and isinstance(self.parent.get("sha256"), str)
                and self.parent["sha256"] and isinstance(self.parent.get("run"), str)
                and self.parent["run"]):
            raise ModelRecordError("a parent is named by the SHA-256 of its checkpoint file "
                                   "and its run")
        if self.operation == "finetune" and self.learning_stage != "finetune":
            raise ModelRecordError("a finetune operation produces a finetune spec")
        if self.operation == "new" and self.learning_stage != "base":
            raise ModelRecordError("a new run is a base; a task model comes from finetune")
        changes = (self.parent or {}).get("changes")
        if self.operation == "branch":
            # BACKLOG 23: a branch is its parent with the recorded changes,
            # each from the parent's value to the new one, its weights loaded.
            if not (isinstance(changes, dict) and changes and all(
                    isinstance(c, dict) and set(c) == {"parent", "new"}
                    for c in changes.values())):
                raise ModelRecordError("a branch records each setting it changed as "
                                       "parent.changes {name: {parent, new}}")
            if self.weight_transfer is None:
                raise ModelRecordError("a branch records how its parent's weights were loaded")
        elif changes is not None:
            raise ModelRecordError(f"only a branch records changed settings, not a "
                                   f"{self.operation} run")
        return self

    def to_payload(self) -> Dict[str, Any]:
        self.validate()
        return {
            "model_spec": self.model_spec.to_dict(),
            "graph_view": self.graph_view.record(),
            "learning_stage": self.learning_stage,
            "task": self.task,
            "operation": self.operation,
            "parent": dict(self.parent) if self.parent is not None else None,
            "family_id": self.family_id,
            "weight_transfer": (dict(self.weight_transfer)
                                if self.weight_transfer is not None else None),
            "data_roles": dict(self.data_roles),
            "graph_exposure": dict(self.graph_exposure),
            "loss_identity": dict(self.loss_identity),
            "scoring_identity": (dict(self.scoring_identity)
                                 if self.scoring_identity is not None else None),
            "selection": dict(self.selection) if self.selection is not None else None,
        }


def has_model_spec(payload: Mapping[str, Any]) -> bool:
    return isinstance(payload, Mapping) and payload.get("model_spec") is not None


def read_model_record(payload: Mapping[str, Any]) -> ModelRecord:
    """The model record of a loaded checkpoint, or the reason it has none."""
    if not has_model_spec(payload):
        raise ModelRecordError(HISTORY_ONLY)
    if payload.get("record_version") != MODEL_RECORD_VERSION:
        raise ModelRecordError(f"checkpoint record version {payload.get('record_version')!r} "
                               f"is not {MODEL_RECORD_VERSION}; this build reads only its own")
    missing = [k for k in RECORD_KEYS if k not in payload]
    if missing:
        raise ModelRecordError(f"the model record lacks {missing}")
    try:
        spec = ModelSpec.from_dict(payload["model_spec"])
        view = ViewShape.from_record(payload["graph_view"])
    except (SpecError, TypeError, ValueError, KeyError, AttributeError) as exc:
        raise ModelRecordError(f"the model spec cannot be read: {exc}") from exc
    if payload["learning_stage"] != spec.learning_stage or payload["task"] != spec.task:
        raise ModelRecordError("the record's learning stage or task differs from its model spec")
    # Each part as plain data of its kind, or the record is refused as a whole,
    # never half read (a list shows it as refused, with this reason).
    for key in ("data_roles", "graph_exposure", "loss_identity"):
        if not isinstance(payload[key], dict):
            raise ModelRecordError(f"the record's {key} is not a mapping")
    for key in ("parent", "weight_transfer", "scoring_identity", "selection"):
        if payload[key] is not None and not isinstance(payload[key], dict):
            raise ModelRecordError(f"the record's {key} is neither a mapping nor null")
    record = ModelRecord(
        model_spec=spec, graph_view=view, operation=payload["operation"],
        family_id=payload["family_id"], parent=payload["parent"],
        weight_transfer=payload["weight_transfer"], data_roles=dict(payload["data_roles"]),
        graph_exposure=dict(payload["graph_exposure"]),
        loss_identity=dict(payload["loss_identity"]),
        scoring_identity=payload["scoring_identity"], selection=payload["selection"])
    return record.validate()


def check_model_record(record: ModelRecord, *, learning_stage: Optional[str] = None,
                       task: Optional[str] = None, view_digest: Optional[str] = None) -> None:
    """Refuse a record that is not what the caller needs: the stage and task
    first, then the graph view."""
    if learning_stage is not None:
        if learning_stage not in LEARNING_STAGES:
            raise ValueError(f"unknown learning stage {learning_stage!r}")
        if record.learning_stage != learning_stage:
            raise ModelRecordError(f"a {record.learning_stage} checkpoint, not a "
                                   f"{learning_stage} one")
    if task is not None:
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}")
        if record.task != task:
            raise ModelRecordError(f"a checkpoint for task {record.task!r}, not {task!r}")
    if view_digest is not None and record.graph_view.digest != view_digest:
        raise ModelRecordError(f"built on graph view {record.graph_view.digest[:12]}, not "
                               f"{view_digest[:12]}")


def finetune_parent_problem(record: ModelRecord, payload: Mapping[str, Any]) -> Optional[str]:
    """Why a read checkpoint cannot be a finetune's parent (contract §9.2), as
    the rest of a sentence naming it, or None: a base with weights; children
    never chain. Its graph is compared with the workspace's where the
    workspace is read (`train_model.read_finetune_parent`)."""
    if record.learning_stage != "base":
        return (f"is a {record.learning_stage} checkpoint; a finetune's parent is a base, "
                "and children never chain.")
    if not isinstance(payload, Mapping) or payload.get("state_dict") is None:
        return "holds no model weights."
    return None


#: What a list says of a checkpoint without a model spec (README §4: listed as
#: history, never run). The WebUI never loads, measures or continues one; only
#: the ShepherdGNN path outside it (the command line, and the HTTP API's model
#: reload) still runs such a file, until K8 retires that path (README §7).
LISTED_WITHOUT_SPEC = ("no model spec (a ShepherdGNN checkpoint): listed as history and never "
                       "run from the WebUI; only the ShepherdGNN path outside it (command line, "
                       "API reload) still runs it, until that path is retired (K8)")

#: What a model list shows of a checkpoint's record (`listed_record`).
LISTED_RECORD_FIELDS = ("record", "record_problem", "learning_stage", "task", "operation",
                        "family_id", "parent_sha256", "parent_run", "encoder", "reference",
                        "differences")


def listed_record(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """What a model list shows of a checkpoint's record, read by
    `read_model_record` alone, so the lists apply no rule of their own.

    `record` is "model" (a record this build reads), "history" (no model spec:
    `LISTED_WITHOUT_SPEC`) or "refused" (a record this build cannot
    read, its reason in `record_problem`). A model adds its learning stage,
    task, operation, family (the root run), parent (the SHA-256 of the exact
    parent file and the parent run), encoder kind, and how its spec stands to
    the reference recipe (README §5, axis 2): `differences`, the sorted names
    (``section.field``) of the spec's fields that differ from it
    (`ModelSpec.differences_from_reference`), and `reference`, true exactly
    when there are none."""
    summary: Dict[str, Any] = dict.fromkeys(LISTED_RECORD_FIELDS)
    if payload is None or not has_model_spec(payload):
        summary.update(record="history", record_problem=LISTED_WITHOUT_SPEC)
        return summary
    try:
        record = read_model_record(payload)
    except ModelRecordError as exc:
        summary.update(record="refused", record_problem=str(exc))
        return summary
    parent = record.parent or {}
    differences = tuple(sorted(record.model_spec.differences_from_reference()))
    summary.update(record="model", learning_stage=record.learning_stage, task=record.task,
                   operation=record.operation, family_id=record.family_id,
                   parent_sha256=parent.get("sha256"), parent_run=parent.get("run"),
                   encoder=record.model_spec.encoder.kind, reference=not differences,
                   differences=differences)
    return summary


def load_bundle(payload: Mapping[str, Any], *, learning_stage: Optional[str] = None,
                task: Optional[str] = None, view_digest: Optional[str] = None) -> tuple:
    """Read, check and build a checkpoint's bundle, its weights loaded strictly.
    Returns `(ModelBundle, ModelRecord)`."""
    from src.models.bundle.builder import build_bundle

    record = read_model_record(payload)
    check_model_record(record, learning_stage=learning_stage, task=task, view_digest=view_digest)
    state = payload.get("state_dict")
    if state is None:
        raise ModelRecordError("the checkpoint holds no model weights")
    bundle = build_bundle(record.model_spec, record.graph_view)
    try:
        bundle.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ModelRecordError(f"the weights do not match the model spec: {exc}") from exc
    return bundle, record


__all__ = [
    "HISTORY_ONLY",
    "LISTED_RECORD_FIELDS",
    "LISTED_WITHOUT_SPEC",
    "MODEL_RECORD_VERSION",
    "ModelRecord",
    "ModelRecordError",
    "OPERATIONS",
    "RECORD_KEYS",
    "check_model_record",
    "finetune_parent_problem",
    "has_model_spec",
    "listed_record",
    "load_bundle",
    "read_model_record",
]
