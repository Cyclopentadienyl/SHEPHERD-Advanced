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
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

from src.config.model_types import SUPPORTED_CONV_TYPES

logger = logging.getLogger(__name__)

router = APIRouter()

CONFIG_FILE = Path(".shepherd_ui_config.json")
DEFAULT_WORKSPACE_DIR = "data/workspaces/default"


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
    fingerprint_warnings: List[str] = Field(default_factory=list)
    checkpoint_meta: Dict[str, Any] = Field(default_factory=dict)
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
        description="Explicit path to a .pt checkpoint. If null, auto-selects from "
        "the architecture subdirs under {data_dir}/checkpoints/.",
    )
    conv_type: Optional[str] = Field(
        None,
        description="GNN architecture to serve: 'auto' (latest-trained), or "
        "'hgt'/'gat'/'sage' to load from that architecture's subdir. Ignored when "
        "checkpoint_path is given.",
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
    files_found: Dict[str, Any] = Field(default_factory=dict)
    checkpoint_path: Optional[str] = None
    architecture: Optional[str] = None
    selection_reason: Optional[str] = None


class UIConfigResponse(BaseModel):
    """Saved UI configuration."""
    data_dir: str = DEFAULT_WORKSPACE_DIR
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
        # Distinguish auto-selectable checkpoints (in architecture subdirs,
        # {data_dir}/checkpoints/{conv_type}/) from LEGACY FLAT ones sitting
        # directly under checkpoints/ (or the root). Auto only serves the former,
        # so reporting them separately avoids "files say checkpoint=True while
        # reload says none found".
        ckpt_dir = d / "checkpoints"
        arch_pts: List[Path] = []
        flat_pts: List[Path] = []
        if ckpt_dir.is_dir():
            for p in ckpt_dir.rglob("*.pt"):
                rel = p.relative_to(ckpt_dir)
                if len(rel.parts) == 1:
                    flat_pts.append(p)  # legacy flat, not auto-selected
                elif rel.parts[0] in SUPPORTED_CONV_TYPES:
                    arch_pts.append(p)  # architecture-scoped, auto-selectable
        else:
            flat_pts = [p for p in d.glob("*.pt")
                        if "checkpoint" in p.name or "model" in p.name]
        result["checkpoint"] = len(arch_pts) > 0  # auto-selectable
        result["checkpoint_legacy_flat"] = len(flat_pts) > 0
        all_pts = arch_pts + flat_pts
        if all_pts:
            result["checkpoint_files"] = [str(p.relative_to(d)) for p in all_pts]

    return result


# =============================================================================
# Endpoints
# =============================================================================
def _status_of(
    config: Optional[Dict[str, Any]],
    data_dir: Optional[str],
    checkpoint_path: Optional[str],
) -> PipelineStatusResponse:
    """Render one pipeline configuration as a status response."""
    if config is None:
        return PipelineStatusResponse(
            initialized=False,
            current_data_dir=data_dir,
            current_checkpoint_path=checkpoint_path,
        )
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
        fingerprint_warnings=config.get("fingerprint_warnings", []),
        checkpoint_meta=config.get("checkpoint_meta", {}),
        current_data_dir=data_dir,
        current_checkpoint_path=checkpoint_path,
    )


def _still_serving() -> str:
    """The tail every rejected-candidate message carries.

    Empty when nothing was loaded to begin with: a service that has never built
    a pipeline is not "still serving" one, and a reassurance that is sometimes
    false is worth less than no reassurance.
    """
    from src.api.main import app_state

    if app_state.pipeline is None:
        return ""
    return " The previously loaded pipeline is still being served."


def _live_status() -> PipelineStatusResponse:
    """The pipeline this service is serving at this moment.

    **Every failed reload reports this, not `initialized=False`.** A rejected
    candidate never becomes the served pipeline, so answering "not initialized"
    would tell an operator the service is down when it is still diagnosing
    patients out of the workspace it had before — and, on the path that used to
    tear the old pipeline down first, that answer was true only because the
    teardown had already happened.
    """
    from src.api.main import app_state

    pipeline = app_state.pipeline
    return _status_of(
        None if pipeline is None else pipeline.get_pipeline_config(),
        app_state._current_data_dir,
        app_state._current_checkpoint_path,
    )


@router.get("/pipeline/status", response_model=PipelineStatusResponse)
async def get_pipeline_status() -> PipelineStatusResponse:
    """Get current pipeline status."""
    return _live_status()


@router.post("/pipeline/reload", response_model=PipelineReloadResponse)
async def reload_pipeline(request: PipelineReloadRequest) -> PipelineReloadResponse:
    """
    Reload the diagnosis pipeline with new data directory and/or checkpoint.

    Builds the replacement first and swaps it in only once it is complete and
    verified; a rejected candidate leaves the running pipeline serving.
    """
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
            status=_live_status(),
            files_found=files,
        )

    # Resolve which checkpoint to serve, from the architecture-scoped layout
    # ({data_dir}/checkpoints/{conv_type}/). Priority:
    #   1. explicit checkpoint_path (also the escape hatch for legacy flat files)
    #   2. a specified conv_type -> that architecture's best/last/newest
    #   3. auto -> latest-trained architecture, serving its best checkpoint
    # Legacy flat checkpoints are NOT auto-scanned here (see checkpoint_paths).
    from src.utils.checkpoint_paths import (
        normalize_conv_type,
        ranking_score_detail,
        select_auto_checkpoint,
        select_checkpoint_in_dir,
    )

    # score_fn reads the ranking metric from a checkpoint's OWN metadata (never
    # the filename, whose number's meaning is config-dependent). Higher is better.
    # Any read failure / missing metric returns None so one bad checkpoint can't
    # block the rest; if none score, selection falls back to last.pt -> newest.
    def _checkpoint_score(path: Path) -> Optional[float]:
        try:
            import torch
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001 — skip unreadable, don't fail reload
            logger.debug("checkpoint score read failed for %s: %s", path, exc)
            return None
        detail = ranking_score_detail(ckpt.get("logs") if isinstance(ckpt, dict) else None)
        return detail[1] if detail else None

    # architecture = the real GNN arch when known (None for an explicit path,
    # where it's only known after load); selection_reason = how it was chosen.
    architecture: Optional[str] = None
    selection_reason: str = ""
    requested_conv = (request.conv_type or "auto").strip().lower()

    if checkpoint_path:
        selection_reason = "explicit checkpoint_path"
    else:
        base = Path(data_dir) / "checkpoints"
        if requested_conv and requested_conv != "auto":
            try:
                architecture = normalize_conv_type(requested_conv)
            except ValueError as exc:
                return PipelineReloadResponse(
                    success=False,
                    message=str(exc),
                    status=_live_status(),
                    files_found=files,
                    selection_reason="invalid conv_type",
                )
            selected = select_checkpoint_in_dir(base / architecture, score_fn=_checkpoint_score)
            selection_reason = f"architecture '{architecture}'"
        else:
            selected, architecture, selection_reason = select_auto_checkpoint(
                base, score_fn=_checkpoint_score
            )
        if selected is not None:
            checkpoint_path = str(selected)
            # Record which metric/score chose it, so "why this one?" is answerable.
            try:
                import torch
                _logs = torch.load(selected, map_location="cpu", weights_only=False).get("logs")
                _detail = ranking_score_detail(_logs)
            except Exception:  # noqa: BLE001
                _detail = None
            if _detail:
                selection_reason += f"; best {_detail[0]}={_detail[1]:.4f}"
            logger.info("Auto-selected checkpoint (%s): %s", selection_reason, checkpoint_path)

    if not checkpoint_path or not Path(checkpoint_path).exists():
        # If legacy flat checkpoints exist but weren't auto-selected, say so —
        # this is the migration-only design, so guide the user rather than
        # contradict files_found.
        if files.get("checkpoint_legacy_flat"):
            hint = (
                "Legacy flat checkpoints exist under checkpoints/ but are not "
                "auto-selected; migrate them (scripts/migrate_checkpoints.py) or "
                "pass an explicit checkpoint_path."
            )
        else:
            hint = "Train a model or pass an explicit checkpoint_path."
        return PipelineReloadResponse(
            success=False,
            message=f"No checkpoint found ({selection_reason}). {hint}",
            status=_live_status(),
            files_found=files,
            architecture=architecture,
            selection_reason=selection_reason,
        )

    # Derive kg_path from data_dir
    kg_path = str(Path(data_dir) / "kg.json")

    # **Build the candidate before touching what is being served.** Everything
    # that can reject this reload happens inside `build_pipeline`: a crossed
    # graph source, a replaced graph artifact, a manifest at an unsupported
    # schema, an unreadable checkpoint, a configuration that fails to read back.
    # Releasing the running pipeline first — as this endpoint used to — meant
    # every one of those rejections landed on a service that had already
    # discarded a healthy pipeline, and turned a refused workspace into a
    # clinical outage. A swap of references is the whole mechanism; there is no
    # rollback path to get wrong because nothing is undone.
    #
    # The cost is a window where both pipelines are resident. That is the right
    # trade: an allocation failure in that window fails the *reload* and leaves
    # the served pipeline exactly as it was, whereas freeing first buys headroom
    # by making every later failure unrecoverable.
    from src.api.main import build_pipeline, publish_pipeline

    try:
        candidate = build_pipeline(
            kg_path=kg_path,
            checkpoint_path=checkpoint_path,
            data_dir=data_dir,
            device=device,
        )
    except Exception as e:
        logger.error(f"Pipeline reload failed: {e}")
        return PipelineReloadResponse(
            success=False,
            message=f"Pipeline initialization failed: {e}.{_still_serving()}",
            status=_live_status(),
            files_found=files,
        )

    if candidate is None:
        return PipelineReloadResponse(
            success=False,
            message=f"No pipeline could be built from {data_dir}. "
            f"Check server logs.{_still_serving()}",
            status=_live_status(),
            files_found=files,
        )

    # **The whole response is built, validated and encoded before anything is
    # published.** `build_pipeline` obtains the configuration dictionary but
    # makes no claim about its *shape*, and Pydantic construction does not close
    # that either: `checkpoint_meta` is `Dict[str, Any]`, so a value that no
    # encoder can serialise is accepted by the model and rejected only at the
    # HTTP boundary -- after this function has returned, and after publication.
    # Encoding here is what moves that failure back in front of the swap.
    # Publication is the last thing this endpoint does that changes anything.
    try:
        config = candidate.config
        fp_warns = config.get("fingerprint_warnings", [])

        msg = "Pipeline reloaded successfully."
        if selection_reason:
            msg += f" ({selection_reason})"
        if fp_warns:
            msg += f" WARNING: {len(fp_warns)} fingerprint mismatch(es) detected."

        response = PipelineReloadResponse(
            success=True,
            message=msg,
            status=_status_of(config, data_dir, checkpoint_path),
            files_found=files,
            checkpoint_path=checkpoint_path,
            architecture=architecture,
            selection_reason=selection_reason,
        )
        # The encoder FastAPI itself runs on the way out, run here for its
        # refusal rather than its output: the response object is what gets
        # returned, and the second encode costs one small object.
        jsonable_encoder(response)
    except Exception as e:
        # Reported as a refused reload rather than a 500, because that is what it
        # is: a candidate this service cannot describe is one it should not serve.
        logger.error(f"Pipeline reload rejected while rendering its response: {e}")
        return PipelineReloadResponse(
            success=False,
            message=f"Pipeline built but its configuration cannot be reported: {e}."
            f"{_still_serving()}",
            status=_live_status(),
            files_found=files,
        )

    publish_pipeline(candidate)
    return response


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
    """Save UI configuration to .shepherd_ui_config.json.

    **Written beside the target and renamed onto it**, so an interrupted write
    leaves the previous selection intact instead of a truncated file. This is
    the same rule as the reload above, applied to the persisted form of the same
    fact: which workspace this deployment serves. The consequence is milder —
    `get_ui_config` falls back to the default workspace on an unreadable file,
    and the operator sees that default in the path field rather than being
    served from it silently — but the failure it removes is a real one, and
    losing an operator's workspace selection has no upside.
    """
    try:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            "w", dir=str(CONFIG_FILE.parent), prefix=CONFIG_FILE.name,
            suffix=".tmp", delete=False,
        )
        try:
            with handle:
                json.dump(config.model_dump(), handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(handle.name, CONFIG_FILE)
        except BaseException:
            Path(handle.name).unlink(missing_ok=True)
            raise
        logger.info(f"UI config saved to {CONFIG_FILE}")
        return config
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save config: {e}",
        )
