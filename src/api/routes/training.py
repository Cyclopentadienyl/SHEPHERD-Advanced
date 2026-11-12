"""
SHEPHERD-Advanced Training API Routes
======================================
REST endpoints for training control and monitoring.

Module: src/api/routes/training.py
Absolute Path: /home/user/SHEPHERD-Advanced/src/api/routes/training.py

Purpose:
    Provide training management API endpoints:
    - POST /training/start: Start a base (KG link prediction) training run,
      or continue a run of either stage from its checkpoint (a strict start:
      every setting a new base run reads, a thin projection of the training
      process's settings)
    - POST /training/finetune: Start one to three finetune children of a base,
      in order (README §1; contract §9.3 hazard 6)
    - POST /training/branch: Branch a checkpoint: a new run of its stage and
      task from its weights, with a new optimizer and schedule, changing the
      settings given (none: a same-settings branch; contract §9.2,
      ops-wiring round decisions B1, B2)
    - POST /training/stop: Stop the current training run (and cancel the
      queued finetune children)
    - GET  /training/status: Get current training status
    - GET  /training/metrics: Get full metrics history
    - GET  /training/checkpoints: List model checkpoints

    Start, Fine-tune, branch, stop and the checkpoint list (which opens
    checkpoint files, SPEC_4 C3) are operator controls: each takes D8's one dependency
    (`src/api/operator_access.py`) and is refused before it runs unless D8
    allows the request. Status and metrics stay open.

Dependencies:
    - fastapi: Router, request/response models
    - pydantic: Request validation
    - src.api.services.training_manager: TrainingManager singleton

Version: 1.0.0
"""
from __future__ import annotations

import logging
import re
from typing import Annotated, Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import (
    BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator,
)

from src.api.operator_access import OPERATOR_ONLY
from src.api.services.training_manager import FinetuneInputs, training_manager
from src.utils.run_plan import new_run_settings

logger = logging.getLogger(__name__)

router = APIRouter()

# PyTorch device grammar. Deliberately NOT a closed {auto,cuda,cpu} enum: torch accepts an
# ordinal (``cuda:1``) and ``mps``, and a multi-GPU host must be able to target a specific card.
# The WebUI keeps its conservative three-choice Radio; see src/config/training_fields.py
# (field "device", valid_pattern) which this must stay in step with.
_DEVICE_RE = re.compile(r"^(auto|cpu|mps|cuda(:\d+)?)$")


def _checked_device(value: str) -> str:
    """**The one device check of the start requests** (a base or a
    continuation, a Fine-tune start, a branch): `value`, or a ValueError (422).
    fullmatch, not match: Python's ``$`` also matches just before a trailing
    newline, so ``match()`` would accept "cuda\\n" — not a valid torch device
    string, and it would then travel into the YAML and the training
    subprocess before failing."""
    if not _DEVICE_RE.fullmatch(value):
        raise ValueError(
            f"device must be 'auto', 'cpu', 'mps', 'cuda' or 'cuda:N' (got {value!r}).")
    return value


#: The start request's operational fields for which an explicit null means
#: "not given": the parent and its extension.
_NULL_IS_NOT_GIVEN = frozenset({"resume_from", "extend_epochs", "extend_lr"})

#: The settings a new base run reads (`run_plan.new_run_settings("base")`, the
#: CLI's rule): the start request's settings, beside its run and continuation
#: fields.
_BASE_SETTINGS = tuple(sorted(new_run_settings("base")))
#: The start request's fields whose values the field spec checks first
#: (`value_problem`): every setting, and the operational ones that are not
#: text (text is never coerced; `device` has its grammar below).
_CHECKED_BY_THE_SPEC = (*_BASE_SETTINGS, "compile", "extend_epochs", "extend_lr")


# =============================================================================
# Request/Response Models
# =============================================================================
def _setting(name: str) -> Any:
    """A setting of a start request: unset by default (the training process
    fills it, by the stage's recipe or this build's default, and records its
    source), its bounds and words the field spec's (`training_fields`: the
    field's `valid` and description, for the schema; each value is checked
    first by `value_problem`, `TrainingStartRequest._check_by_the_spec`), so
    the request is a thin projection and the runner's one check and the
    request agree."""
    from src.config.training_fields import by_name

    spec = by_name(name)
    return Field(default=None, description=spec.description, **(spec.valid or {}))


class TrainingStartRequest(BaseModel):
    """A training start (ops-wiring round, decision S6): a new base (KG link
    prediction) run, or a run of either stage continued from its checkpoint
    (`resume_from`: resumed, or extended with `extend_epochs`), through the
    job runner's one start (`TrainingManager.start_training`), as the WebUI's
    base form and Continue start one. **A thin projection** of what the
    training process takes: every setting a new base run reads
    (`run_plan.new_run_settings("base")`, the model spec and split seed among
    them), each unset by default, its bounds the field spec's; and the run's
    operational fields.

    **Strict** (BACKLOG 54), as the Fine-tune start is: a field this request
    does not define, a ShepherdGNN setting among them, is refused (422), never
    dropped (a setting a new base does not read, in the sentence of the one
    check of a new run's settings, `run_plan.new_run_problem`); and each value
    is checked first by the field spec's one check (`value_problem`),
    strictly (a string is never a number, 1 is not true) and in its words, as
    the job runner and the training process check it. `learning_stage` is
    required and ``"base"`` for a new run (a finetune is POST
    /training/finetune's start); with `resume_from` it may be left out, and
    when given is compared with the parent's, so a finetune child is resumed
    or extended here too. **Only the fields sent reach the service** (the
    route dumps the fields set, an explicit null included, so it is judged:
    refused where the field takes no null, or as another value than the
    parent's; a null parent or extension is not given), and the runner checks
    them again (`_new_bundle_run`, the continuation preview) before anything
    starts."""

    model_config = ConfigDict(extra="forbid")

    # The run. Every run writes into a directory of its own, allocated under
    # the workspace (contract M3a-2); the name is optional and for people.
    data_dir: str = Field(..., description="Workspace directory")
    learning_stage: Optional[Literal["base", "finetune"]] = Field(
        default=None, description="'base' for a new run (a finetune is started with POST "
                                  "/training/finetune); with resume_from optional, and "
                                  "compared with the parent's stage when given")
    run_name: str = Field(default="", max_length=200,
                          description="Optional run name, part of the allocated directory's name")
    device: str = Field(default="auto", description="Device: auto, cpu, mps, cuda, or cuda:N")
    compile: bool = Field(default=False, description="torch.compile (experimental; falls back "
                                                     "to eager on failure)")
    resume_from: Optional[str] = Field(
        default=None,
        description="A checkpoint to continue from: an unfinished run is resumed with its "
                    "settings, a finished one extended with extend_epochs")
    extend_epochs: Optional[int] = Field(
        default=None, ge=1, description="Extend a finished parent by this many epochs at a "
                                        "fixed rate")
    extend_lr: Optional[float] = _setting("extend_lr")

    # The settings a new base run reads (`run_plan.new_run_settings("base")`).
    # Unset is the recipe's value or this build's default; the bounds are the
    # field spec's (training_fields).
    num_epochs: Optional[int] = _setting("num_epochs")
    learning_rate: Optional[float] = _setting("learning_rate")
    weight_decay: Optional[float] = _setting("weight_decay")
    batch_size: Optional[int] = _setting("batch_size")
    num_neighbors: Optional[List[int]] = _setting("num_neighbors")
    max_grad_norm: Optional[float] = _setting("max_grad_norm")
    seed: Optional[int] = _setting("seed")
    scheduler_type: Optional[Literal["cosine", "onecycle", "linear", "none"]] = _setting(
        "scheduler_type")
    warmup_steps: Optional[int] = _setting("warmup_steps")
    min_lr_ratio: Optional[float] = _setting("min_lr_ratio")
    use_amp: Optional[bool] = _setting("use_amp")
    amp_dtype: Optional[Literal["float16", "bfloat16"]] = _setting("amp_dtype")
    gradient_accumulation_steps: Optional[int] = _setting("gradient_accumulation_steps")
    early_stopping_patience: Optional[int] = _setting("early_stopping_patience")
    eval_every_n_epochs: Optional[int] = _setting("eval_every_n_epochs")
    save_top_k: Optional[int] = _setting("save_top_k")
    split_seed: Optional[int] = _setting("split_seed")
    model_spec: Optional[Dict[str, Any]] = _setting("model_spec")

    @model_validator(mode="before")
    @classmethod
    def _refuse_retired_locations(cls, data: Any) -> Any:
        # output_dir / checkpoint_dir / log_dir were retired by the run layout:
        # a value names that reason here; any key the request does not define,
        # these ones empty included, is refused by extra="forbid".
        from src.utils.checkpoint_paths import refuse_retired_locations

        refuse_retired_locations(data, source="This request")
        return data

    @model_validator(mode="before")
    @classmethod
    def _refuse_what_a_base_does_not_read(cls, data: Any) -> Any:
        # A setting a new base does not read (a finetune's, or the freeze) is
        # refused by the one check of a new run's settings, in the sentence
        # the job runner and the training process refuse it in
        # (`run_plan.new_run_problem`); any other field this request does not
        # define is refused by extra="forbid". A continuation's settings are
        # its parent's: the request does not take a finetune's.
        from src.utils.run_plan import RUN_SETTINGS, new_run_problem

        if isinstance(data, dict) and not data.get("resume_from"):
            stated = {name: value for name, value in data.items()
                      if name in RUN_SETTINGS and name not in cls.model_fields}
            problem = new_run_problem("base", (None,), stated) if stated else None
            if problem:
                raise ValueError(problem)
        return data

    @field_validator("device")
    @classmethod
    def _validate_device(cls, v: str) -> str:
        return _checked_device(v)

    @field_validator(*_CHECKED_BY_THE_SPEC, mode="before")
    @classmethod
    def _check_by_the_spec(cls, value: Any, info: ValidationInfo) -> Any:
        # Each value as sent, by the field spec's one check (`value_problem`),
        # before any coercion: its kind strictly, its bounds, choices and
        # list items (a hop list of at least one hop, each 1 or more or -1,
        # every neighbour, as the CLI and the WebUI take it), in the words the
        # job runner and the training process refuse it in. A null setting is
        # passed on, for the runner to judge by the run's reference
        # (`run_plan.new_run_problem`) or a parent's value; a null parent or
        # extension is not given.
        from src.config.training_fields import value_problem

        name = info.field_name
        problem = value_problem(name, value, allow_none=(name in _BASE_SETTINGS
                                                         or name in _NULL_IS_NOT_GIVEN))
        if problem:
            raise ValueError(problem)
        return value

    @model_validator(mode="after")
    def _validate_the_run(self) -> "TrainingStartRequest":
        # Extending continues a parent; without one the training process
        # refuses (scripts/train_model.py), so the request is refused here,
        # before a process is started for nothing. A new run names its stage,
        # a base (decision S6).
        if not self.resume_from:
            for name in ("extend_epochs", "extend_lr"):
                if getattr(self, name) is not None:
                    raise ValueError(f"{name} continues a parent; give resume_from.")
            if self.learning_stage != "base":
                raise ValueError(
                    "A new run names its learning stage, learning_stage 'base' (a finetune "
                    "is started with POST /training/finetune); a continuation (resume_from) "
                    "may leave it out.")
        return self


class FinetuneStartRequest(BaseModel):
    """A Fine-tune start: one to three tasks from one base checkpoint, each an
    independent child run, in order (README §1, §8). `settings` apply to every
    child and must be read by every chosen task; the task's recipe fills the
    rest. Strict: any other field is refused (422)."""

    model_config = ConfigDict(extra="forbid")

    data_dir: str = Field(..., description="Workspace directory")
    finetune_from: str = Field(..., description="The base checkpoint the children start from")
    tasks: List[Literal["disease", "gene", "patients_like_me"]] = Field(
        ..., min_length=1, max_length=3, description="The tasks, one child each, in order")
    device: str = Field(default="auto", description="auto, cpu, cuda or cuda:N")
    run_name: str = Field(default="", max_length=200, description="Optional run name prefix")
    settings: Dict[str, Any] = Field(default_factory=dict,
                                     description="Training settings for every child")

    @field_validator("device")
    @classmethod
    def _validate_device(cls, value: str) -> str:
        return _checked_device(value)


class BranchStartRequest(BaseModel):
    """A branch start (contract §9.2, BACKLOG 23; ops-wiring round, decisions
    B1, B2): a new run of a checkpoint's stage and task from its weights,
    with a new optimizer and schedule, through the job runner's one start
    (`TrainingManager.start_training` with `branch_from`), which the branch's
    one preview decides (`preview_branch`, `run_plan.plan_branch`).

    **A thin projection**: the run (`data_dir`, `branch_from`, `device`,
    `run_name`) and `settings`, the settings to change, each on the closed
    list of the parent's stage and task (the preview's `changeable`); `{}` is
    a same-settings branch, every setting its parent's. The parent's stage,
    task and architecture are its record's. Strict: a field this request
    does not define is refused (422), and so is a name in `settings` that is
    not a setting a run states (a run field, a continuation's or a ShepherdGNN
    setting). What the runner judges of `settings` against the parent (a
    setting its stage does not read or a branch keeps, a value its field spec
    does not take, a rule across settings, a hop list its encoder does not
    take) is refused with 422; a parent it cannot branch, one changed since
    it was shown, a dataset changed or a busy slot with 409."""

    model_config = ConfigDict(extra="forbid")

    data_dir: str = Field(..., description="Workspace directory")
    branch_from: str = Field(..., description="The checkpoint to branch: a training-state "
                                              "checkpoint of a model bundle run")
    device: str = Field(default="auto", description="Device: auto, cpu, mps, cuda, or cuda:N")
    run_name: str = Field(default="", max_length=200,
                          description="Optional run name, part of the allocated directory's name")
    settings: Dict[str, Any] = Field(
        default_factory=dict,
        description="The settings to change, each on the branch list of the parent's stage and "
                    "task; {} branches with every setting of the parent")

    @field_validator("device")
    @classmethod
    def _validate_device(cls, value: str) -> str:
        return _checked_device(value)

    @field_validator("settings")
    @classmethod
    def _only_settings(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        # A name that is not a setting a run states (`run_plan.RUN_SETTINGS`)
        # would reach the start as a run field: refused here, never passed on.
        from src.utils.run_plan import RUN_SETTINGS

        others = sorted(name for name in value if name not in RUN_SETTINGS)
        if others:
            one = len(others) == 1
            raise ValueError(
                f"settings holds {', '.join(map(repr, others))}, which "
                f"{'is not a training setting' if one else 'are not training settings'}; a "
                "branch's run is named by data_dir, branch_from, device and run_name.")
        return value


class TrainingStatusResponse(BaseModel):
    """Training status response."""

    status: str
    pid: Optional[int] = None
    start_time: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    latest_metrics: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    run_dir: Optional[str] = None
    # How the run stands once its process is not running, from its last
    # snapshot's record (contract M3a); None when it saved none.
    outcome: Optional[Dict[str, Any]] = None
    # The job runner's own record of the latest training job (M5).
    job_id: Optional[str] = None
    end_time: Optional[str] = None
    exit_code: Optional[int] = None
    # The one GPU slot: the job holding it, of either type, and the GPU lock.
    gpu: Optional[Dict[str, Any]] = None
    # The latest Fine-tune start's children, each with its own status (README
    # §1); None until one is started, and after another training start.
    finetune: Optional[Dict[str, Any]] = None


class CheckpointInfo(BaseModel):
    """Checkpoint file information."""

    filename: str
    filepath: str
    size_mb: float
    modified: str
    epoch: Optional[int] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    run: Optional[str] = None
    run_dir: Optional[str] = None
    architecture: Optional[str] = None
    # The entry's identity in the listing, apart from the names shown above
    # (BACKLOG 66: a name that is not text is shown with backslash escapes and
    # identified by its bytes): architecture/run/file for names that are text;
    # and its run's identity, architecture/run, by which a list groups entries.
    key: Optional[str] = None
    run_key: Optional[str] = None
    checkpoint_kind: Optional[str] = None
    # From the same one read as the metadata (M5-1, BACKLOG 19.3): the SHA-256
    # of the bytes read, the Trainer's validation figure, the dataset
    # fingerprint the checkpoint records, and why a file could not be read.
    sha256: Optional[str] = None
    validation: Optional[Dict[str, Any]] = None
    dataset_fingerprint: Optional[str] = None
    metadata_problem: Optional[str] = None
    # The model record's summary (`src.models.bundle.record.listed_record`):
    # model or refused (with why); a model's stage, task, operation, family and
    # parent, encoder kind, the spec fields that differ from the reference
    # recipe, whether there are none, and its capability description.
    record: Optional[str] = None
    record_problem: Optional[str] = None
    learning_stage: Optional[str] = None
    task: Optional[str] = None
    operation: Optional[str] = None
    family_id: Optional[str] = None
    parent_sha256: Optional[str] = None
    parent_run: Optional[str] = None
    encoder: Optional[str] = None
    reference: Optional[bool] = None
    differences: Optional[List[str]] = None
    capabilities: Optional[Dict[str, Any]] = None


# =============================================================================
# Endpoints
# =============================================================================
@router.post("/training/start", response_model=Dict[str, Any], dependencies=OPERATOR_ONLY)
async def start_training(
    request: TrainingStartRequest,
    expected_dataset_fingerprint: Annotated[Optional[str], Query(
        pattern=r"^[0-9a-f]{64}$",
        description="Refuse the start (409) unless the workspace's dataset fingerprint, the "
                    "SHA-256 of its split_manifest.json, is still this one")] = None,
    expected_parent_sha256: Annotated[Optional[str], Query(
        pattern=r"^[0-9a-f]{64}$",
        description="With resume_from: refuse the start (409) unless the parent checkpoint "
                    "still has this SHA-256 (the model fingerprint shown)")] = None,
) -> Dict[str, Any]:
    """
    Start a base training run, or continue a run of either stage.

    Launches scripts/train_model.py as a subprocess, through the job runner's
    one start (`TrainingManager.start_training`), with **only the fields the
    caller set** (contract M3a-3; decision S2), an explicit null included: a
    new base's ``learning_stage="base"``, the run keys and its settings. The
    reference recipe, then this build's default, fills every setting not set;
    a run continued from a parent inherits every learning setting the caller
    did not set, and refuses one it set to another value, a stated stage
    other than the parent's among them (409). The runner refuses, before
    anything is started, written or claimed (the GPU slot, a run directory),
    a setting a base does not read, a value its field spec does not take, or
    a setting the run would not read by the rules across settings (422).

    ``expected_dataset_fingerprint`` (an optional query parameter, M5-2: a
    precondition on the request, not a training setting, so it is not in the
    body the training field spec pins) makes the start refuse (409) unless the
    workspace's split manifest is still the one the caller showed. The
    dashboard's Start always sends it, through this same service call.

    ``expected_parent_sha256`` (optional, the same kind of precondition, for a
    run continued from ``resume_from``) makes the start refuse (409) unless the
    parent is still the checkpoint the caller showed. Without it the training
    process is still bound to the parent the service decided on (root-cause
    round, P22-S1): a file replaced after that is refused before it is loaded.
    """
    # The fields the caller set, an explicit null included (decision S2): a
    # null setting is the caller's and is judged by the runner; for the parent
    # and its extension, null is not given.
    config = {name: value for name, value in request.model_dump(exclude_unset=True).items()
              if not (value is None and name in _NULL_IS_NOT_GIVEN)}
    result = training_manager.start_training(
        config, expected_dataset_fingerprint=expected_dataset_fingerprint,
        expected_parent_sha256=expected_parent_sha256)

    if not result.get("success"):
        raise HTTPException(
            # 422 under its RFC 9110 name: Starlette 1.x warns on the old
            # HTTP_422_UNPROCESSABLE_ENTITY. Same status code.
            status_code=(status.HTTP_422_UNPROCESSABLE_CONTENT if result.get("invalid_settings")
                         else status.HTTP_409_CONFLICT),
            detail=result.get("error", "Failed to start training"),
        )

    return result


@router.post("/training/finetune", response_model=Dict[str, Any], dependencies=OPERATOR_ONLY)
def start_finetune(
    request: FinetuneStartRequest,
    expected_dataset_fingerprint: Annotated[Optional[str], Query(
        pattern=r"^[0-9a-f]{64}$",
        description="Refuse the start (409) unless the workspace's dataset fingerprint is "
                    "still this one")] = None,
    expected_parent_sha256: Annotated[Optional[str], Query(
        pattern=r"^[0-9a-f]{64}$",
        description="Refuse the start (409) unless the base checkpoint still has this "
                    "SHA-256 (the model fingerprint shown)")] = None,
) -> Dict[str, Any]:
    """Start one to three finetune children of a base, in order, through the
    one job runner (`TrainingManager.start_finetune`): each its own run,
    status and record; a failed child does not stop its siblings, and
    /training/stop ends the running child and cancels the queued ones.
    **A plain `def`**: the start reads the whole base checkpoint, so FastAPI
    runs it in its threadpool, never on the event loop the API and the
    dashboard share."""
    result = training_manager.start_finetune(
        FinetuneInputs(data_dir=request.data_dir, finetune_from=request.finetune_from,
                       tasks=tuple(request.tasks), device=request.device,
                       run_name=request.run_name, settings=dict(request.settings),
                       expected_parent_sha256=expected_parent_sha256),
        expected_dataset_fingerprint=expected_dataset_fingerprint)
    if not result.get("success"):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail=result.get("error", "Failed to start the finetune"))
    return result


@router.post("/training/branch", response_model=Dict[str, Any], dependencies=OPERATOR_ONLY)
def start_branch(
    request: BranchStartRequest,
    expected_dataset_fingerprint: Annotated[Optional[str], Query(
        pattern=r"^[0-9a-f]{64}$",
        description="Refuse the start (409) unless the workspace's dataset fingerprint is "
                    "still this one")] = None,
    expected_parent_sha256: Annotated[Optional[str], Query(
        pattern=r"^[0-9a-f]{64}$",
        description="Refuse the start (409) unless the parent checkpoint still has this "
                    "SHA-256 (the model fingerprint shown)")] = None,
) -> Dict[str, Any]:
    """Branch a checkpoint through the job runner's one start
    (`TrainingManager.start_training` with `branch_from`; ops-wiring round,
    decision B2): its preview of the parent decides the branch as the
    training process does, and the process is bound to that read's digest
    (`--expect-parent-sha256`), whether or not ``expected_parent_sha256`` is
    sent; with it, another parent is refused (409) before anything starts.
    ``settings`` `{}` is a same-settings branch (decision B1). Invalid
    settings answer 422, any other refusal 409. **A plain `def`**: the start
    reads the whole parent checkpoint, so FastAPI runs it in its threadpool,
    never on the event loop the API and the dashboard share."""
    config = {"data_dir": request.data_dir, "branch_from": request.branch_from,
              "device": request.device, "run_name": request.run_name, **request.settings}
    result = training_manager.start_training(
        config, expected_dataset_fingerprint=expected_dataset_fingerprint,
        expected_parent_sha256=expected_parent_sha256)
    if not result.get("success"):
        raise HTTPException(
            status_code=(status.HTTP_422_UNPROCESSABLE_CONTENT if result.get("invalid_settings")
                         else status.HTTP_409_CONFLICT),
            detail=result.get("error", "Failed to start the branch"))
    return result


@router.post("/training/stop", response_model=Dict[str, Any], dependencies=OPERATOR_ONLY)
def stop_training() -> Dict[str, Any]:
    """
    Stop the currently running training; a test run is never stopped by it.

    Sends SIGINT to the job's main process for graceful shutdown, falls back to
    SIGTERM, then SIGKILL, to its whole process group; answers once no process
    of the job is left.
    **A plain `def`**: the stop waits for the process to end (tens of seconds
    at worst), so FastAPI runs it in its threadpool instead of on the event
    loop, which the rest of the API and the mounted dashboard share.
    """
    result = training_manager.stop_training()

    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=result.get("error", "Failed to stop training"),
        )

    return result


@router.get("/training/status", response_model=TrainingStatusResponse)
async def get_training_status() -> Dict[str, Any]:
    """Get current training status with latest metrics."""
    return training_manager.get_status()


@router.get("/training/metrics")
async def get_training_metrics() -> Dict[str, Any]:
    """
    Get full training metrics history for chart rendering.

    Returns train and validation metric lists from the MetricsLogger JSON logs.
    """
    return training_manager.get_metrics_history()


@router.get("/training/checkpoints", response_model=List[CheckpointInfo],
            dependencies=OPERATOR_ONLY)
async def get_checkpoints() -> List[Dict[str, Any]]:
    """List the checkpoints of the last requested workspace's runs, with
    each file's own metadata (contract M3a-2). It opens every file (SPEC_4
    C3), so D8 gates it like the controls."""
    return training_manager.get_checkpoints()
