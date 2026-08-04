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

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator, model_validator

from src.api.services.training_manager import training_manager

logger = logging.getLogger(__name__)

router = APIRouter()

# PyTorch device grammar. Deliberately NOT a closed {auto,cuda,cpu} enum: torch accepts an
# ordinal (``cuda:1``) and ``mps``, and a multi-GPU host must be able to target a specific card.
# The WebUI keeps its conservative three-choice Radio; see src/config/training_fields.py
# (field "device", valid_pattern) which this must stay in step with.
_DEVICE_RE = re.compile(r"^(auto|cpu|mps|cuda(:\d+)?)$")


# =============================================================================
# Request/Response Models
# =============================================================================
class TrainingStartRequest(BaseModel):
    """Training start request with configuration parameters."""

    # Paths
    data_dir: str = Field(default="data/workspaces/default", description="Workspace directory")
    output_dir: str = Field(default="outputs", description="Output directory")
    checkpoint_dir: str = Field(default="", description="Checkpoint directory (blank = auto-derive from workspace)")
    log_dir: str = Field(default="logs", description="Log directory")

    # Tier 1 — Basic
    num_epochs: int = Field(default=100, ge=1, le=10000, description="Number of epochs")
    learning_rate: float = Field(default=1e-4, gt=0, le=1.0, description="Learning rate")
    batch_size: int = Field(default=32, ge=1, le=2048, description="Batch size")
    conv_type: Literal["gat", "hgt", "sage"] = Field(default="gat", description="GNN convolution type")
    device: str = Field(default="auto", description="Device: auto, cpu, mps, cuda, or cuda:N")
    resume_from: Optional[str] = Field(default=None, description="Checkpoint path to resume from")
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

    @field_validator("device")
    @classmethod
    def _validate_device(cls, v: str) -> str:
        if not _DEVICE_RE.match(v):
            raise ValueError(
                f"device must be 'auto', 'cpu', 'mps', 'cuda' or 'cuda:N' (got {v!r})."
            )
        return v

    @model_validator(mode="after")
    def _validate_head_divisibility(self) -> "TrainingStartRequest":
        # Only the attention convolutions split hidden_dim across heads: HGTConv divides
        # out_channels by heads, and GATConv is built as (hidden_dim // num_heads) * heads with
        # concat=True, so a non-divisible pair silently truncates and breaks the residual add /
        # LayerNorm(hidden_dim). SAGEConv takes no heads, so it is deliberately exempt.
        if self.conv_type in ("hgt", "gat") and self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads "
                f"({self.num_heads}) when conv_type is '{self.conv_type}'."
            )
        return self

    @model_validator(mode="after")
    def _validate_onecycle_min_lr(self) -> "TrainingStartRequest":
        # min_lr_ratio == 0 is valid for cosine/linear (decay-to-zero), so the
        # field keeps ge=0.0. Only onecycle divides by it (final_div_factor =
        # 1 / min_lr_ratio) and therefore needs a strictly positive value.
        if self.scheduler_type == "onecycle" and self.min_lr_ratio <= 0:
            raise ValueError(
                "min_lr_ratio must be > 0 when scheduler_type is 'onecycle'."
            )
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


class CheckpointInfo(BaseModel):
    """Checkpoint file information."""

    filename: str
    filepath: str
    size_mb: float
    modified: str
    epoch: Optional[int] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Endpoints
# =============================================================================
@router.post("/training/start", response_model=Dict[str, Any])
async def start_training(request: TrainingStartRequest) -> Dict[str, Any]:
    """
    Start a new training run.

    Launches scripts/train_model.py as a subprocess with the provided config.
    """
    config = request.model_dump(exclude_none=True)

    result = training_manager.start_training(config)

    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=result.get("error", "Failed to start training"),
        )

    return result


@router.post("/training/stop", response_model=Dict[str, Any])
async def stop_training() -> Dict[str, Any]:
    """
    Stop the currently running training.

    Sends SIGINT for graceful shutdown, falls back to SIGTERM if needed.
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


@router.get("/training/checkpoints", response_model=List[CheckpointInfo])
async def get_checkpoints() -> List[Dict[str, Any]]:
    """List available model checkpoints with metadata."""
    return training_manager.get_checkpoints()
