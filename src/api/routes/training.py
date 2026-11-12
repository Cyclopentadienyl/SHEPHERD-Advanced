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
    - POST /training/stop: Stop the current training run (and cancel the
      queued finetune children)
    - GET  /training/status: Get current training status
    - GET  /training/metrics: Get full metrics history
    - GET  /training/checkpoints: List model checkpoints

    Start, Fine-tune, stop and the checkpoint list (which opens checkpoint
    files, SPEC_4 C3) are operator controls: each takes D8's one dependency
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
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.api.operator_access import OPERATOR_ONLY
from src.api.services.training_manager import FinetuneInputs, training_manager

logger = logging.getLogger(__name__)

router = APIRouter()

# PyTorch device grammar. Deliberately NOT a closed {auto,cuda,cpu} enum: torch accepts an
# ordinal (``cuda:1``) and ``mps``, and a multi-GPU host must be able to target a specific card.
# The WebUI keeps its conservative three-choice Radio; see src/config/training_fields.py
# (field "device", valid_pattern) which this must stay in step with.
_DEVICE_RE = re.compile(r"^(auto|cpu|mps|cuda(:\d+)?)$")


#: The start request's operational fields for which an explicit null means
#: "not given": the parent and its extension.
_NULL_IS_NOT_GIVEN = frozenset({"resume_from", "extend_epochs", "extend_lr"})


# =============================================================================
# Request/Response Models
# =============================================================================
def _setting(name: str) -> Any:
    """A setting of a start request: unset by default (the training process
    fills it, by the stage's recipe or this build's default, and records its
    source), its bounds and words the field spec's (`training_fields`: the
    field's `valid` and description), so the request is a thin projection and
    the runner's one check (`value_problem`) and the request agree."""
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
    dropped. `learning_stage` is required and ``"base"`` for a new run (a
    finetune is POST /training/finetune's start); with `resume_from` it may be
    left out, and when given is compared with the parent's, so a finetune
    child is resumed or extended here too. **Only the fields sent reach the
    service** (the route dumps the fields set, an explicit null included, so
    it is judged: refused where the field takes no null, or as another value
    than the parent's), and the runner checks them again (`_new_bundle_run`,
    the continuation preview) before anything starts."""

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

    @field_validator("device")
    @classmethod
    def _validate_device(cls, v: str) -> str:
        # fullmatch, not match: Python's ``$`` also matches just before a trailing newline, so
        # ``match()`` would accept "cuda\n" — not a valid torch device string, and it would then
        # travel into the YAML and the training subprocess before failing.
        if not _DEVICE_RE.fullmatch(v):
            raise ValueError(
                f"device must be 'auto', 'cpu', 'mps', 'cuda' or 'cuda:N' (got {v!r})."
            )
        return v

    @field_validator("num_neighbors")
    @classmethod
    def _validate_hops(cls, value: Optional[List[int]]) -> Optional[List[int]]:
        # The hop list by the field spec's one check (`value_problem`): at
        # least one hop, each 1 or more or -1 (every neighbour), as the CLI
        # and the WebUI take it.
        from src.config.training_fields import value_problem

        problem = None if value is None else value_problem("num_neighbors", value)
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
        if not _DEVICE_RE.fullmatch(value):
            raise ValueError(f"device {value!r} is not auto, cpu, mps, cuda or cuda:N")
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
