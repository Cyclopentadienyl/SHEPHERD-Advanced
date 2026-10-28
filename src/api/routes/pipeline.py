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

**The two POSTs are operator controls** (D8, M5-1): each route takes the one
D8 dependency (`src/api/operator_access.py`), and does its work through one
function: `run_reload`, which the dashboard's shared Load
(`src/webui/components/model_list.py`, "Change model…" and Research › Models)
also calls in process after its own check, each entrance judging its own
request (the Load binds it to the model fingerprint it showed,
`expected_checkpoint_sha256`, which an API client can pass too); and
`write_ui_config`, which only the route calls since M5-2 removed the Diagnosis
tab's Save Config. The GETs stay open.

Version: 1.0.0
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException, status
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

from src.api.operator_access import OPERATOR_ONLY
from src.api.routes.diagnose import ServedIdentity
from src.utils.file_modes import create_staging_file

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
    # None where the served model has no η: a disease or patients-like-me
    # model scores by its own head alone (the served identity's eta).
    eta_effective: Optional[float] = 0.0
    sp_max_hops: Optional[int] = None
    #: "sidecar": where the hop bound in effect was read. The only source since
    #: the configured bound was retired (contract M2.5); kept so a status reader
    #: sees where the ceiling every shortest-path score uses came from.
    sp_hop_bound_source: Optional[str] = None
    #: "verified" — the table names the graph being served and the two agree.
    #: A table published before the pairing protocol ("unrecorded") is refused
    #: at load since contract M2.5, so it is never served. **Never absent while
    #: shortest paths are on.**
    sp_kg_binding: Optional[str] = None
    #: The served workspace's dataset record (contract M3b): `binding` (always
    #: "verified" here; anything else refuses the build) and `origin` ("files"
    #: or "synthetic") apart, its digest, the sources it names and what it says
    #: it does not establish. None when no workspace was verified.
    dataset_record: Optional[Dict[str, Any]] = None
    #: What the pipeline serves, part by part (contract M3c): the value every
    #: diagnosis it answers carries. None when no pipeline is loaded.
    served_identity: Optional[ServedIdentity] = None
    kg_nodes: int = 0
    kg_edges: int = 0
    has_model: bool = False
    fingerprint_warnings: List[str] = Field(default_factory=list)
    checkpoint_meta: Dict[str, Any] = Field(default_factory=dict)
    current_data_dir: Optional[str] = None
    current_checkpoint_path: Optional[str] = None


#: The task each model bundle scoring mode serves (the served identity's
#: `scoring_mode`, `src.core.types.result_kind_for`).
_TASK_OF_MODE = {"disease_task": "disease", "gene_task": "gene",
                 "patients_like_me": "patients_like_me"}


class PipelineReloadRequest(BaseModel):
    """Request to reload the pipeline with new paths."""
    data_dir: str = Field(
        ...,
        description="Path to data directory containing kg.json, node_features.pt, etc.",
    )
    checkpoint_path: Optional[str] = Field(
        None,
        description="Explicit path to a .pt checkpoint. If null, auto-selects from "
        "the latest run under {data_dir}/runs/.",
    )
    conv_type: Optional[str] = Field(
        None,
        description="GNN architecture to serve: 'auto' (latest-trained), or "
        "'hgt'/'gat'/'sage' to load from that architecture's subdir. Ignored when "
        "checkpoint_path is given.",
    )
    task: Optional[Literal["disease", "gene", "patients_like_me"]] = Field(
        None,
        description="The task to serve: automatic selection then takes the latest run "
        "holding a model finetuned for it, judged by each file's model record, never by "
        "its directory (a base, another task's model and a checkpoint without a model "
        "spec are never chosen); with checkpoint_path, the loaded model must be that "
        "task's or nothing is published. Null: the ShepherdGNN selection, which never "
        "chooses a model bundle.",
    )
    device: Optional[str] = Field(
        None,
        description="The model's device: null or 'auto' for CUDA, refused without "
        "it (contract M3c); 'cuda' or 'cuda:N'; 'cpu' for development. Null falls "
        "back to SHEPHERD_DEVICE first. The served identity records what was asked "
        "for and what took effect.",
    )
    expected_checkpoint_sha256: Optional[str] = Field(
        None,
        pattern=r"^[0-9a-f]{64}$",
        description="The model fingerprint the caller was shown (the checkpoint's "
        "SHA-256, 64 lowercase hex). When given, the reload serves only those "
        "bytes: the build's own checkpoint read is compared with it before the "
        "bytes are parsed, and other bytes, or no file, are refused as not the "
        "model that was shown, with the running pipeline still serving. Null "
        "keeps the reload unbound (automatic selection serves what it reads).",
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
# **The manifest belongs here.** Reload builds a file-backed pipeline, which
# refuses a workspace whose artifacts nothing records — so a directory without
# one was always going to be rejected, just later and with a longer story. Listed
# as required, the operator's file report names the missing piece directly. The
# dataset record likewise (contract M3b): the build refuses a workspace without
# it, and whether the one present is the one bound is the build's check.
REQUIRED_DATA_FILES = [
    "kg.json",
    "node_features.pt",
    "edge_indices.pt",
    "num_nodes.json",
    "split_manifest.json",
    "kg.provenance.json",
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
        # Distinguish checkpoints in the run layout ({data_dir}/runs/..., the
        # only place automatic selection looks, contract M3a-2) from ones in an
        # older layout ({data_dir}/checkpoints/, or the workspace root). Both are
        # reported, so "files say checkpoint=True while reload says none found"
        # cannot happen; a file outside the runs is served by its explicit path.
        from src.utils.checkpoint_paths import list_run_checkpoints

        run_pts = [p for _, p in list_run_checkpoints(d)]
        ckpt_dir = d / "checkpoints"
        outside = sorted(ckpt_dir.rglob("*.pt")) if ckpt_dir.is_dir() else []
        outside += [p for p in sorted(d.glob("*.pt"))
                    if "checkpoint" in p.name or "model" in p.name]
        result["checkpoint"] = len(run_pts) > 0  # auto-selectable
        result["checkpoint_outside_runs"] = len(outside) > 0
        all_pts = run_pts + outside
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
        eta_effective=config.get("eta_effective"),
        sp_max_hops=config.get("sp_max_hops"),
        sp_hop_bound_source=config.get("sp_hop_bound_source"),
        sp_kg_binding=config.get("sp_kg_binding"),
        dataset_record=config.get("dataset_record"),
        served_identity=config.get("served_identity"),
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


def live_status() -> PipelineStatusResponse:
    """The status `GET /pipeline/status` answers, for a caller in this process:
    the dashboard reads what is served through this one function (M5-2), never
    a second reader. Reads application state only."""
    return _live_status()


def _refused(message: str, files: Dict[str, Any], **extra: Any) -> PipelineReloadResponse:
    """A refused reload, built so it can actually reach the client.

    **The live pipeline's configuration is arbitrary too.** `_live_status` reads
    it from whatever is currently loaded, and a value no encoder can serialise
    there would turn a refusal into a 500 after this endpoint had returned —
    reporting a crash for a request that correctly declined to do anything. So
    the refusal crosses the encoder here, and falls back to a status built from
    nothing but primitives if it cannot.

    The fallback still answers the question that matters: `initialized` says
    whether a pipeline is serving, which is a boolean read off application
    state and cannot fail to encode.
    """
    from src.api.main import app_state

    try:
        response = PipelineReloadResponse(
            success=False, message=message, status=_live_status(),
            files_found=files, **extra,
        )
        jsonable_encoder(response)
        return response
    except Exception as exc:  # noqa: BLE001 — a refusal must still be deliverable
        logger.error("The live pipeline's status could not be reported: %s", exc)
        return PipelineReloadResponse(
            success=False,
            message=f"{message} (the running pipeline's status could not be "
            f"rendered: {exc})",
            status=PipelineStatusResponse(initialized=app_state.pipeline is not None),
            files_found=files,
            **extra,
        )


@router.get("/pipeline/status", response_model=PipelineStatusResponse)
async def get_pipeline_status() -> PipelineStatusResponse:
    """Get current pipeline status."""
    return _live_status()


@router.post("/pipeline/reload", response_model=PipelineReloadResponse,
             dependencies=OPERATOR_ONLY)
async def reload_pipeline(request: PipelineReloadRequest) -> PipelineReloadResponse:
    """
    Reload the diagnosis pipeline with new data directory and/or checkpoint.

    An operator control: refused by D8's dependency before anything here runs.
    `run_reload` does the work. **Still ``async`` with nothing awaited**, so the
    reload runs on the event loop and waits behind a running diagnosis, as it
    always has (BACKLOG 28 records that this serialisation is the blocked loop).
    """
    return run_reload(request)


def run_reload(request: PipelineReloadRequest) -> PipelineReloadResponse:
    """
    The one reload: the route above and the dashboard's shared Load
    (`model_list`'s confirmation) both call this, each after D8's check on its
    own request. Synchronous; a caller on
    the event loop calls it without awaiting anything, so a reload never runs
    beside a diagnosis.

    Builds the replacement first and swaps it in only once it is complete and
    verified; a rejected candidate leaves the running pipeline serving. **A
    caller that was shown a model binds the reload to it**
    (`expected_checkpoint_sha256`): the build's one checkpoint read is compared
    with it before torch parses the bytes (`read_checkpoint(expected_sha256=)`,
    inside the build), so the bytes served are the bytes confirmed or nothing
    changes, and other bytes -- a checkpoint or not, or no file -- are refused
    as not the model that was shown.
    """
    data_dir = request.data_dir

    # **Naming a workspace is the request, whatever becomes of it.** The refusals
    # below — missing files, no checkpoint, a build that fails — all return
    # before `build_pipeline` records anything, and a caller who reached any of
    # them has still asked this service for a real pipeline. Without this,
    # `/diagnose` would afterwards read them as a demo user and answer with
    # invented candidates. Recorded here rather than after a successful build for
    # exactly that reason.
    from src.api.main import app_state as _app_state

    _app_state.real_pipeline_requested = True
    checkpoint_path = request.checkpoint_path
    device = request.device

    # Check files first
    files = _check_files(data_dir, checkpoint_path)

    missing_required = [f for f in REQUIRED_DATA_FILES if not files.get(f, False)]
    if missing_required:
        return _refused(
            f"Missing required files in {data_dir}: {missing_required}", files
        )

    # **The device before the scan** (contract M3c, §5.2). Every reload serves a
    # checkpoint, so a device this host cannot provide refuses it whatever is
    # chosen; settled here, it refuses before automatic selection reads every
    # candidate in the run layout. The same resolution `build_pipeline` applies
    # (the request, then SHEPHERD_DEVICE, then `auto`), and the request is still
    # passed on as given, so the served identity records what was asked for.
    from src.config.serving_device import requested_serving_device
    from src.inference.pipeline import PipelineBuildError, resolve_serving_device

    try:
        resolve_serving_device(requested_serving_device(device))
    except PipelineBuildError as exc:
        return _refused(f"{str(exc).rstrip('.')}.{_still_serving()}", files)

    # Resolve which checkpoint to serve, from the run layout
    # ({data_dir}/runs/{conv_type}/{run}/checkpoints/, contract M3a-2). Priority:
    #   1. explicit checkpoint_path (any location; accepted by its bytes)
    #   2. a specified conv_type -> that architecture's latest run, its
    #      best/last/newest
    #   3. auto -> the latest-trained run of any architecture, its best
    # Runs are not ranked against each other by metric. Checkpoints in an older
    # layout are NOT scanned here (see checkpoint_paths).
    from src.models.bundle.record import has_model_spec, listed_record
    from src.utils.checkpoint_io import read_checkpoint
    from src.utils.checkpoint_paths import (
        RUN_LAYOUT_KEYS,
        normalize_conv_type,
        ranking_score_detail,
        select_auto_run_checkpoint,
        select_run_checkpoint,
    )

    # score_fn reads the ranking metric from a checkpoint's OWN metadata (never
    # the filename, whose number's meaning is config-dependent). Higher is better.
    # Any read failure / missing metric returns None so one bad checkpoint can't
    # block the rest; if none score, selection falls back to last.pt -> newest.
    #
    # **Each read's digest is kept** (contract M3c, BACKLOG 19.1): the build
    # reads the chosen file again, and only that read is served (M2.1 decision
    # 4), so a file replaced in between is stated beside the reason below.
    # Which candidates the scan scored is kept too: selection ranks by score
    # whenever any candidate has one, so the chosen file was ranked exactly when
    # it is among them.
    scored_digests: Dict[str, str] = {}
    scored_paths: set = set()
    #: Each candidate's one read in this scan: its digest, its ranking metric
    #: and its model record's summary; None for a file the read refused.
    inspected: Dict[str, Optional[Dict[str, Any]]] = {}

    def _inspect(path: Path) -> Optional[Dict[str, Any]]:
        if str(path) not in inspected:
            try:
                # Tensors and plain data only (contract decision 10): this scan
                # reads files in a directory a request names.
                scored = read_checkpoint(path, map_location="cpu", weights_only=True)
            except Exception as exc:  # noqa: BLE001 — skip unreadable, don't fail reload
                logger.debug("checkpoint score read failed for %s: %s", path, exc)
                inspected[str(path)] = None
            else:
                ckpt = scored.checkpoint
                meta = ckpt if isinstance(ckpt, dict) else {}
                inspected[str(path)] = {
                    "sha256": scored.identity.sha256,
                    "detail": ranking_score_detail(meta.get("logs")),
                    "record": listed_record(meta) if has_model_spec(meta) else None,
                }
        return inspected[str(path)]

    def _checkpoint_score(path: Path) -> Optional[float]:
        seen = _inspect(path)
        if seen is None:
            return None
        scored_digests[str(path)] = seen["sha256"]
        if seen["detail"]:
            scored_paths.add(str(path))
        return seen["detail"][1] if seen["detail"] else None

    def _accept(path: Path) -> bool:
        """The record decides (provenance §9.3, hazard 3). With a task: a
        model bundle finetuned for it. Without one (the ShepherdGNN reload): a
        file without a model spec. A file the scan could not read is judged by
        nothing, so it is never a candidate on either path."""
        seen = _inspect(path)
        if seen is None:
            return False
        record = seen["record"]
        if request.task is None:
            return record is None
        return (record is not None and record["record"] == "model"
                and record["learning_stage"] == "finetune"
                and record["task"] == request.task)

    # architecture = the real GNN arch when known (None for an explicit path,
    # where it's only known after load); selection_reason = how it was chosen.
    architecture: Optional[str] = None
    selection_reason: str = ""
    requested_conv = (request.conv_type or "auto").strip().lower()
    #: The file automatic selection chose; None for an explicit path.
    selected: Optional[Path] = None

    if checkpoint_path:
        selection_reason = "explicit checkpoint_path"
    else:
        if request.task is not None:
            if requested_conv and requested_conv != "auto":
                return _refused(
                    "conv_type names a ShepherdGNN architecture; a task is served by "
                    "the model finetuned for it, chosen by its record. Give one of them.",
                    files, selection_reason="task with conv_type")
            # Every run-layout key: a model bundle is filed under its encoder
            # kind, and the record, never the directory, says what it is.
            selected, architecture, selection_reason = select_auto_run_checkpoint(
                data_dir, score_fn=_checkpoint_score, accept=_accept, keys=RUN_LAYOUT_KEYS)
            selection_reason = f"task {request.task}: {selection_reason}"
        elif requested_conv and requested_conv != "auto":
            try:
                architecture = normalize_conv_type(requested_conv)
            except ValueError as exc:
                return _refused(
                    str(exc), files, selection_reason="invalid conv_type"
                )
            selected, selection_reason = select_run_checkpoint(
                data_dir, architecture, score_fn=_checkpoint_score, accept=_accept
            )
        else:
            selected, architecture, selection_reason = select_auto_run_checkpoint(
                data_dir, score_fn=_checkpoint_score, accept=_accept
            )
        # What automatic selection could choose for this request, so the file
        # report and the reason never disagree.
        files["checkpoint"] = selected is not None
        if selected is not None:
            checkpoint_path = str(selected)

    # **With an expectation, the build's read decides whether the file is
    # there**: a file removed since it was shown is not the model shown
    # (`CheckpointChanged`, below), not "no checkpoint found, train a model".
    expected = request.expected_checkpoint_sha256
    if not checkpoint_path or (expected is None and not Path(checkpoint_path).exists()):
        # If checkpoints exist outside the run layout, say so: they are not
        # selected automatically (decision 9: no runtime fallback), so guide the
        # user rather than contradict files_found.
        if request.checkpoint_path:
            hint = f"{request.checkpoint_path} does not exist."
        elif request.task is not None:
            hint = (f"No run holds a model finetuned for the {request.task} task; a base, "
                    "another task's model and a checkpoint without a model spec are not "
                    "chosen for it. Finetune one, or pass an explicit checkpoint_path.")
        elif files.get("checkpoint_outside_runs"):
            hint = (
                "Checkpoints exist outside the run layout (runs/) and are not "
                "selected automatically; move them once with "
                "scripts/migrate_checkpoints.py, or pass an explicit checkpoint_path."
            )
        else:
            hint = ("Train a model or pass an explicit checkpoint_path; a model bundle is "
                    "selected by its task (task).")
        return _refused(
            f"No checkpoint found ({selection_reason}). {hint}", files,
            architecture=architecture, selection_reason=selection_reason,
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
    from src.utils.checkpoint_io import CheckpointChanged

    # **The model the caller was shown is a precondition of this reload**,
    # compared on the build's one checkpoint read before torch parses the bytes.
    # A file rewritten after the caller hashed it (a training run renames a new
    # `last.pt` into place every epoch, and the build reads the whole graph
    # before the checkpoint), torn, of another graph, or removed, is refused as
    # what it is -- not the model that was shown -- before any model is built
    # from it; there is no "reloaded, but with other bytes" to report afterwards.
    # Without an expectation the reload is unbound, as before: automatic
    # selection serves what the build read (19.1, below).
    try:
        candidate = build_pipeline(
            kg_path=kg_path,
            checkpoint_path=checkpoint_path,
            data_dir=data_dir,
            device=device,
            expected_checkpoint_sha256=expected,
        )
    except CheckpointChanged as exc:
        held = (f"it holds {exc.found[:12]} now, not {exc.expected[:12]}"
                if exc.found is not None else f"it is gone; {exc.expected[:12]} was shown")
        return _refused(
            f"{checkpoint_path} is not the model that was shown: {held}. Nothing was "
            f"published; read it again to see what it holds.{_still_serving()}", files
        )
    except Exception as e:
        logger.error(f"Pipeline reload failed: {e}")
        return _refused(
            f"Pipeline initialization failed: {str(e).rstrip('.')}."
            f"{_still_serving()}", files
        )

    if candidate is None:
        return _refused(
            f"No pipeline could be built from {data_dir}. "
            f"Check server logs.{_still_serving()}", files
        )

    # The digest of the build's own single checkpoint read: the model it serves
    # (19.1's note below compares the scan's read with it).
    served_sha = ((candidate.config.get("served_identity") or {}).get("checkpoint")
                  or {}).get("sha256")

    # **A task asked for is the loaded model's, or nothing is published**: an
    # explicit checkpoint_path is judged by the build's own read of its record.
    if request.task is not None:
        served_mode = (candidate.config.get("served_identity") or {}).get("scoring_mode")
        served_task = _TASK_OF_MODE.get(served_mode)
        if served_task != request.task:
            served = (f"the {served_task} task" if served_task is not None
                      else f"scoring mode {served_mode!r}")
            return _refused(
                f"{checkpoint_path} serves {served}, not the {request.task} task asked "
                f"for. Nothing was published.{_still_serving()}", files)

    if selected is not None:
        # **The metric reported is the served read's** (contract M3c, BACKLOG
        # 19.2), from the checkpoint metadata the build took from the bytes it
        # loaded, never from another read of the path: chose A, reported B,
        # served C is gone. The build's read is what is served (M2.1 decision
        # 4); a file replaced after the scan chose it is still served as read,
        # and the reason says so (19.1), so a choice made on other bytes is
        # never presented as this model's.
        #
        # **"best" only for the bytes that were ranked.** When no candidate
        # carries a metric, selection falls back to `last.pt`, else the newest
        # file, and the reason says that instead. When the file ranked best was
        # replaced before the build read it, the loaded bytes were never ranked
        # against the others. Either way the loaded file's metric, if its
        # metadata has one, is shown as the loaded file's, never as a winner.
        detail = ranking_score_detail(candidate.config.get("checkpoint_meta"))
        scored_sha = scored_digests.get(str(selected))
        replaced = scored_sha is not None and scored_sha != served_sha
        if str(selected) not in scored_paths:
            selection_reason += (
                "; not ranked by metric (no candidate carries one): last.pt, "
                "else the newest file"
            )
        if detail:
            ranked = str(selected) in scored_paths and not replaced
            selection_reason += (
                f"; {'best' if ranked else 'loaded file'} {detail[0]}={detail[1]:.4f}"
            )
        if replaced:
            selection_reason += (
                f"; the file was replaced after it was scored (scored sha256 "
                f"{scored_sha}, loaded sha256 {served_sha}): the choice was made "
                "on the earlier bytes"
                + (", and the metric shown is the loaded file's" if detail else "")
            )
            logger.warning("Auto-selected checkpoint replaced after scoring: %s", checkpoint_path)
        logger.info("Auto-selected checkpoint (%s): %s", selection_reason, checkpoint_path)

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
        return _refused(
            f"Pipeline built but its configuration cannot be reported: "
            f"{str(e).rstrip('.')}.{_still_serving()}", files
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


@router.post("/pipeline/config", response_model=UIConfigResponse,
             dependencies=OPERATOR_ONLY)
async def save_ui_config(config: UIConfigResponse) -> UIConfigResponse:
    """Save UI configuration to .shepherd_ui_config.json (`write_ui_config`).

    An operator control: refused by D8's dependency before anything is written.
    """
    try:
        return write_ui_config(config)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save config: {e}",
        )


def write_ui_config(config: UIConfigResponse, target: Optional[Path] = None) -> UIConfigResponse:
    """The one writer of .shepherd_ui_config.json, called by the route above
    after D8's check (the dashboard's Save Config was removed in M5-2). `target`
    is the file, `CONFIG_FILE` unless a test names another one to write.

    **Written beside the target and renamed onto it**, so an interrupted write
    leaves the previous selection intact instead of a truncated file. This is
    the same rule as the reload above, applied to the persisted form of the same
    fact: which workspace this deployment serves. The consequence is milder —
    `get_ui_config` falls back to the default workspace on an unreadable file,
    and the operator sees that default in the path field rather than being
    served from it silently — but the failure it removes is a real one, and
    losing an operator's workspace selection has no upside.

    Raises whatever the write raised (``OSError``, typically), after removing
    its staged file: the file named is then as it was.
    """
    path = CONFIG_FILE if target is None else Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, staged = create_staging_file(path, binary=False)
    try:
        with handle:
            json.dump(config.model_dump(), handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staged, path)
    except BaseException:
        Path(staged).unlink(missing_ok=True)
        raise
    logger.info(f"UI config saved to {path}")
    return config
