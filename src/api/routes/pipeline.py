"""
SHEPHERD-Advanced Pipeline Management API Routes
==================================================
Endpoints for pipeline status, reload, and configuration.

Module: src/api/routes/pipeline.py

Endpoints:
    GET  /pipeline/status  — Current pipeline state (gnn_ready, sp_ready, etc.)
    POST /pipeline/reload  — Reload pipeline with new data_dir / checkpoint_path
    GET  /pipeline/config   — Get saved UI config (.shepherd_ui_config.json)
    POST /pipeline/config  — Save UI config

Version: 1.0.0
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()

CONFIG_FILE = Path(".shepherd_ui_config.json")
DEFAULT_DATA_DIR = "data/processed"
DEFAULT_CHECKPOINT_DIR = "models/production"


# =============================================================================
# Request/Response Models
# =============================================================================
class PipelineStatusResponse(BaseModel):
    """Pipeline status information."""
    initialized: bool
    gnn_ready: bool = False
    sp_ready: bool = False
    scoring_mode: str = "not_initialized"
    eta_configured: float = 0.7
    eta_effective: float = 0.0
    sp_max_hops: Optional[int] = None
    kg_nodes: int = 0
    kg_edges: int = 0
    has_model: bool = False
    vector_index_ready: bool = False
    fingerprint_warnings: List[str] = Field(default_factory=list)
    current_data_dir: Optional[str] = None
    current_checkpoint_path: Optional[str] = None


class PipelineReloadRequest(BaseModel):
    """Request to reload the pipeline with new paths."""
    data_dir: str = Field(
        ...,
        description="Path to data directory containing kg.json, node_features.pt, etc.",
    )
    checkpoint_path: Optional[str] = Field(
        None,
        description="Path to model checkpoint .pt file. If null, scans data_dir for checkpoint files.",
    )
    device: Optional[str] = Field(
        None,
        description="Device for inference (null = auto-detect).",
    )


class PipelineReloadResponse(BaseModel):
    """Response after pipeline reload attempt."""
    success: bool
    message: str
    status: PipelineStatusResponse
    files_found: Dict[str, bool] = Field(default_factory=dict)


class UIConfigResponse(BaseModel):
    """Saved UI configuration."""
    data_dir: str = DEFAULT_DATA_DIR
    checkpoint_path: Optional[str] = None


# =============================================================================
# File completeness check
# =============================================================================
REQUIRED_DATA_FILES = [
    "kg.json",
    "node_features.pt",
    "edge_indices.pt",
    "num_nodes.json",
]
OPTIONAL_DATA_FILES = [
    "shortest_paths.pt",
    "shortest_paths.meta.json",
]


def _check_files(data_dir: str, checkpoint_path: Optional[str] = None) -> Dict[str, bool]:
    """Check which required/optional files exist in data_dir."""
    d = Path(data_dir)
    result = {}
    for f in REQUIRED_DATA_FILES + OPTIONAL_DATA_FILES:
        result[f] = (d / f).exists()

    if checkpoint_path:
        result["checkpoint"] = Path(checkpoint_path).exists()
    else:
        pts = list(d.glob("*.pt"))
        ckpts = [p for p in pts if "checkpoint" in p.name or "model" in p.name]
        result["checkpoint"] = len(ckpts) > 0
        if ckpts:
            result["checkpoint_files"] = [p.name for p in ckpts]

    return result


# =============================================================================
# Endpoints
# =============================================================================
@router.get("/pipeline/status", response_model=PipelineStatusResponse)
async def get_pipeline_status() -> PipelineStatusResponse:
    """Get current pipeline status."""
    from src.api.main import app_state

    if app_state.pipeline is None:
        return PipelineStatusResponse(
            initialized=False,
            current_data_dir=app_state._current_data_dir if hasattr(app_state, "_current_data_dir") else None,
            current_checkpoint_path=app_state._current_checkpoint_path if hasattr(app_state, "_current_checkpoint_path") else None,
        )

    config = app_state.pipeline.get_pipeline_config()
    return PipelineStatusResponse(
        initialized=True,
        gnn_ready=config.get("gnn_ready", False),
        sp_ready=config.get("sp_ready", False),
        scoring_mode=config.get("scoring_mode", "unknown"),
        eta_configured=config.get("eta_configured", 0.7),
        eta_effective=config.get("eta_effective", 0.0),
        sp_max_hops=config.get("sp_max_hops"),
        kg_nodes=config.get("kg_nodes", 0),
        kg_edges=config.get("kg_edges", 0),
        has_model=config.get("has_model", False),
        vector_index_ready=config.get("vector_index_ready", False),
        fingerprint_warnings=config.get("fingerprint_warnings", []),
        current_data_dir=getattr(app_state, "_current_data_dir", None),
        current_checkpoint_path=getattr(app_state, "_current_checkpoint_path", None),
    )


@router.post("/pipeline/reload", response_model=PipelineReloadResponse)
async def reload_pipeline(request: PipelineReloadRequest) -> PipelineReloadResponse:
    """
    Reload the diagnosis pipeline with new data directory and/or checkpoint.

    This releases the current pipeline (if any) and initializes a new one.
    """
    from src.api.main import app_state, initialize_pipeline

    data_dir = request.data_dir
    checkpoint_path = request.checkpoint_path
    device = request.device

    # Check files first
    files = _check_files(data_dir, checkpoint_path)

    missing_required = [f for f in REQUIRED_DATA_FILES if not files.get(f, False)]
    if missing_required:
        return PipelineReloadResponse(
            success=False,
            message=f"Missing required files in {data_dir}: {missing_required}",
            status=PipelineStatusResponse(initialized=False),
            files_found=files,
        )

    # Resolve checkpoint path
    if not checkpoint_path:
        d = Path(data_dir)
        ckpts = [p for p in d.glob("*.pt")
                 if "checkpoint" in p.name or "model" in p.name]
        if ckpts:
            checkpoint_path = str(ckpts[0])
            logger.info(f"Auto-detected checkpoint: {checkpoint_path}")

    if not checkpoint_path or not Path(checkpoint_path).exists():
        return PipelineReloadResponse(
            success=False,
            message=f"No valid checkpoint found. Provide checkpoint_path or place a *checkpoint*.pt / *model*.pt file in {data_dir}.",
            status=PipelineStatusResponse(initialized=False),
            files_found=files,
        )

    # Release existing pipeline
    if app_state.pipeline is not None:
        logger.info("Releasing existing pipeline for reload...")
        app_state.pipeline = None
        app_state.kg = None

    # Derive kg_path from data_dir
    kg_path = str(Path(data_dir) / "kg.json")

    try:
        # Force re-initialization (bypass the "already initialized" guard)
        initialize_pipeline(
            kg_path=kg_path,
            checkpoint_path=checkpoint_path,
            data_dir=data_dir,
            device=device,
        )
    except Exception as e:
        logger.error(f"Pipeline reload failed: {e}")
        return PipelineReloadResponse(
            success=False,
            message=f"Pipeline initialization failed: {e}",
            status=PipelineStatusResponse(initialized=False),
            files_found=files,
        )

    if app_state.pipeline is None:
        return PipelineReloadResponse(
            success=False,
            message="Pipeline initialization returned but pipeline is still None. Check server logs.",
            status=PipelineStatusResponse(initialized=False),
            files_found=files,
        )

    # Store current paths for status reporting
    app_state._current_data_dir = data_dir
    app_state._current_checkpoint_path = checkpoint_path

    config = app_state.pipeline.get_pipeline_config()
    fp_warns = config.get("fingerprint_warnings", [])

    msg = "Pipeline reloaded successfully."
    if fp_warns:
        msg += f" WARNING: {len(fp_warns)} fingerprint mismatch(es) detected."

    status_resp = PipelineStatusResponse(
        initialized=True,
        gnn_ready=config.get("gnn_ready", False),
        sp_ready=config.get("sp_ready", False),
        scoring_mode=config.get("scoring_mode", "unknown"),
        eta_configured=config.get("eta_configured", 0.7),
        eta_effective=config.get("eta_effective", 0.0),
        sp_max_hops=config.get("sp_max_hops"),
        kg_nodes=config.get("kg_nodes", 0),
        kg_edges=config.get("kg_edges", 0),
        has_model=config.get("has_model", False),
        vector_index_ready=config.get("vector_index_ready", False),
        fingerprint_warnings=fp_warns,
        current_data_dir=data_dir,
        current_checkpoint_path=checkpoint_path,
    )

    return PipelineReloadResponse(
        success=True,
        message=msg,
        status=status_resp,
        files_found=files,
    )


@router.get("/pipeline/config", response_model=UIConfigResponse)
async def get_ui_config() -> UIConfigResponse:
    """Get saved UI configuration (paths, defaults)."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                data = json.load(f)
            return UIConfigResponse(**data)
        except Exception as e:
            logger.warning(f"Failed to read UI config: {e}")

    return UIConfigResponse()


@router.post("/pipeline/config", response_model=UIConfigResponse)
async def save_ui_config(config: UIConfigResponse) -> UIConfigResponse:
    """Save UI configuration to .shepherd_ui_config.json."""
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config.model_dump(), f, indent=2)
        logger.info(f"UI config saved to {CONFIG_FILE}")
        return config
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save config: {e}",
        )
