"""
SHEPHERD-Advanced Training API Routes
======================================
REST endpoints for training control and monitoring.

Module: src/api/routes/training.py
Absolute Path: /home/user/SHEPHERD-Advanced/src/api/routes/training.py

Purpose:
    Provide training management API endpoints:
    - POST /training/start: Start a training run
    - POST /training/stop: Stop the current training run
    - GET  /training/status: Get current training status
    - GET  /training/metrics: Get full metrics history
    - GET  /training/checkpoints: List model checkpoints

    Start, stop and the checkpoint list (which opens checkpoint files, SPEC_4
    C3) are operator controls: each takes D8's one dependency
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
from typing import Annotated, Any, Dict, List, Literal, Mapping, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator, model_validator

from src.api.operator_access import OPERATOR_ONLY
from src.api.services.training_manager import training_manager

logger = logging.getLogger(__name__)

router = APIRouter()

# PyTorch device grammar. Deliberately NOT a closed {auto,cuda,cpu} enum: torch accepts an
# ordinal (``cuda:1``) and ``mps``, and a multi-GPU host must be able to target a specific card.
# The WebUI keeps its conservative three-choice Radio; see src/config/training_fields.py
# (field "device", valid_pattern) which this must stay in step with.
_DEVICE_RE = re.compile(r"^(auto|cpu|mps|cuda(:\d+)?)$")


# =============================================================================
# Rules that span fields
# =============================================================================
def cross_field_errors(settings: Mapping[str, Any]) -> List[str]:
    """The training request's rules that span fields, on complete settings.

    One definition for both places they apply (contract M3a review of
    `094b90d`, P2-3): a new run's request, its unset fields at their defaults
    (`TrainingStartRequest`'s validators), and a continued run's settings as
    the shared run plan resolves them from its parent
    (`TrainingManager.start_training`), never the defaults it will not use.

    - Only the attention convolutions split `hidden_dim` across heads: HGTConv
      divides out_channels by heads, and GATConv is built as
      `(hidden_dim // num_heads) * heads` with concat=True, so a non-divisible
      pair silently truncates and breaks the residual add /
      LayerNorm(hidden_dim). SAGEConv takes no heads, so it is exempt.
    - `min_lr_ratio == 0` is valid for cosine/linear (decay to zero); only
      onecycle divides by it (`final_div_factor = 1 / min_lr_ratio`).
    """
    errors: List[str] = []
    conv_type = settings.get("conv_type")
    hidden_dim, num_heads = settings.get("hidden_dim"), settings.get("num_heads")
    if (conv_type in ("hgt", "gat") and isinstance(hidden_dim, int)
            and isinstance(num_heads, int) and num_heads > 0 and hidden_dim % num_heads != 0):
        errors.append(
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads}) "
            f"when conv_type is '{conv_type}'.")
    min_lr_ratio = settings.get("min_lr_ratio")
    if (settings.get("scheduler_type") == "onecycle" and isinstance(min_lr_ratio, (int, float))
            and min_lr_ratio <= 0):
        errors.append("min_lr_ratio must be > 0 when scheduler_type is 'onecycle'.")
    return errors


# =============================================================================
# Request/Response Models
# =============================================================================
class TrainingStartRequest(BaseModel):
    """Training start request with configuration parameters."""

    # Paths. Every run writes into a directory of its own, allocated under the
    # workspace (contract M3a-2); the name is optional and for people.
    data_dir: str = Field(default="data/workspaces/default", description="Workspace directory")
    run_name: str = Field(default="", max_length=200,
                          description="Optional run name, part of the allocated directory's name")

    # Tier 1 — Basic
    num_epochs: int = Field(default=100, ge=1, le=10000, description="Number of epochs")
    learning_rate: float = Field(default=1e-4, gt=0, le=1.0, description="Learning rate")
    batch_size: int = Field(default=32, ge=1, le=2048, description="Batch size")
    conv_type: Literal["gat", "hgt", "sage"] = Field(default="gat", description="GNN convolution type")
    device: str = Field(default="auto", description="Device: auto, cpu, mps, cuda, or cuda:N")
    resume_from: Optional[str] = Field(
        default=None,
        description="Parent checkpoint to continue from: an unfinished stage is resumed with "
                    "the parent's settings, a finished one extended with extend_epochs")
    extend_epochs: Optional[int] = Field(
        default=None, ge=1, description="Extend a finished parent by this many epochs at a "
                                        "fixed rate")
    seed: int = Field(default=42, ge=0, le=2**32 - 1, description="Random seed")

    # Tier 2 — Advanced
    hidden_dim: int = Field(default=256, ge=32, description="Hidden dimension size")
    num_layers: int = Field(default=4, ge=1, le=16, description="Number of GNN layers")
    dropout: float = Field(default=0.1, ge=0.0, le=0.9, description="Dropout rate")
    weight_decay: float = Field(default=0.01, ge=0.0, description="Weight decay")
    scheduler_type: Literal["cosine", "onecycle", "linear", "none"] = Field(
        default="cosine", description="LR scheduler type"
    )
    warmup_steps: int = Field(default=500, ge=0, description="Warmup steps")
    early_stopping_patience: int = Field(default=10, ge=1, description="Early stopping patience")
    diagnosis_weight: float = Field(default=1.0, ge=0.0, description="Diagnosis loss weight")
    link_prediction_weight: float = Field(default=0.5, ge=0.0, description="Link prediction weight")
    contrastive_weight: float = Field(default=0.3, ge=0.0, description="Contrastive loss weight")
    ortholog_weight: float = Field(default=0.2, ge=0.0, description="Ortholog loss weight")

    # Tier 3 — Expert
    gradient_accumulation_steps: int = Field(default=1, ge=1, description="Gradient accumulation")
    max_grad_norm: float = Field(default=1.0, gt=0.0, description="Max gradient norm")
    num_heads: int = Field(default=8, ge=1, description="Number of attention heads")
    use_ortholog_gate: bool = Field(default=True, description="Use ortholog gate")
    use_amp: bool = Field(default=True, description="Use automatic mixed precision")
    amp_dtype: Literal["float16", "bfloat16"] = Field(
        default="float16", description="AMP dtype: float16 or bfloat16"
    )
    temperature: float = Field(default=0.07, gt=0.0, description="Contrastive temperature")
    label_smoothing: float = Field(default=0.1, ge=0.0, le=1.0, description="Label smoothing")
    margin: float = Field(default=1.0, gt=0.0, description="Margin for ranking loss")
    num_neighbors: List[Annotated[int, Field(ge=1)]] = Field(
        default=[15, 10, 5], min_length=1,
        description="Neighbor fan-out per sampling hop (non-empty; each entry >= 1)",
    )
    max_subgraph_nodes: int = Field(default=5000, ge=100, description="Max subgraph nodes")
    min_lr_ratio: float = Field(
        default=0.01, ge=0.0, le=1.0,
        description="Final LR as a fraction of peak (a ratio above 1.0 would raise the final LR "
                    "above the peak, which no scheduler here supports)",
    )
    num_workers: int = Field(default=4, ge=0, description="Data loader workers")
    num_negative_samples: int = Field(default=5, ge=1, description="Negative samples")
    eval_every_n_epochs: int = Field(default=1, ge=1, description="Evaluate every N epochs")
    save_top_k: int = Field(default=3, ge=1, description="Save top K checkpoints")

    @model_validator(mode="before")
    @classmethod
    def _refuse_retired_locations(cls, data: Any) -> Any:
        # output_dir / checkpoint_dir / log_dir were retired by the run layout.
        # Ignoring them, as an unknown key would be, would start a run that
        # writes somewhere other than where its caller looks.
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

    @model_validator(mode="after")
    def _validate_head_divisibility(self) -> "TrainingStartRequest":
        # A continued run (resume_from) uses its parent's settings, not these
        # defaults: its cross-field rules are checked on the settings the run
        # plan resolves (start_training below). Each field's own type and range
        # are checked here for every request.
        if self.resume_from:
            return self
        errors = [e for e in cross_field_errors(self.model_dump()) if "num_heads" in e]
        if errors:
            raise ValueError(errors[0])
        return self

    @model_validator(mode="after")
    def _validate_extend_needs_a_parent(self) -> "TrainingStartRequest":
        # Extending continues a parent; without one the training process
        # refuses (scripts/train_model.py), so the request is refused here,
        # before a process is started for nothing.
        if self.extend_epochs is not None and not self.resume_from:
            raise ValueError("extend_epochs continues a parent; give resume_from.")
        return self

    @model_validator(mode="after")
    def _validate_onecycle_min_lr(self) -> "TrainingStartRequest":
        if self.resume_from:
            return self
        errors = [e for e in cross_field_errors(self.model_dump()) if "onecycle" in e]
        if errors:
            raise ValueError(errors[0])
        return self


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
    checkpoint_kind: Optional[str] = None
    # From the same one read as the metadata (M5-1, BACKLOG 19.3): the SHA-256
    # of the bytes read, the Trainer's validation figure, the dataset
    # fingerprint the checkpoint records, and why a file could not be read.
    sha256: Optional[str] = None
    validation: Optional[Dict[str, Any]] = None
    dataset_fingerprint: Optional[str] = None
    metadata_problem: Optional[str] = None


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
) -> Dict[str, Any]:
    """
    Start a new training run.

    Launches scripts/train_model.py as a subprocess with the provided config.

    **Only the fields the caller set** are passed on (contract M3a-3): the
    defaults are the training script's own, and a run continued from a parent
    inherits every learning setting the caller did not set, and refuses one it
    set to another value.

    ``expected_dataset_fingerprint`` (an optional query parameter, M5-2: a
    precondition on the request, not a training setting, so it is not in the
    body the training field spec pins) makes the start refuse (409) unless the
    workspace's split manifest is still the one the caller showed. The
    dashboard's Start always sends it, through this same service call.
    """
    config = request.model_dump(exclude_unset=True, exclude_none=True)
    expected = expected_dataset_fingerprint

    # A continued run's cross-field rules apply to the settings it will use,
    # resolved from its parent by the shared run plan, inside the service.
    result = training_manager.start_training(config, effective_rules=cross_field_errors,
                                             expected_dataset_fingerprint=expected)
    # Each field's own type and range apply to every request, a value repeating
    # the parent's included; omitting a field inherits it.

    if not result.get("success"):
        raise HTTPException(
            # 422 under its RFC 9110 name: Starlette 1.x warns on the old
            # HTTP_422_UNPROCESSABLE_ENTITY. Same status code.
            status_code=(status.HTTP_422_UNPROCESSABLE_CONTENT if result.get("invalid_settings")
                         else status.HTTP_409_CONFLICT),
            detail=result.get("error", "Failed to start training"),
        )

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
