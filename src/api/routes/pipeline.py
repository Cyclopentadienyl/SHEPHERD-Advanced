"""
SHEPHERD-Advanced Pipeline Management API Routes
==================================================
Endpoints for the served pipeline's status, its load and its stop.

Module: src/api/routes/pipeline.py

Endpoints:
    GET  /pipeline/status  — What is served and the lifecycle around it (gnn_ready,
                             scoring_mode, served identity, serving, in flight, …)
    POST /pipeline/reload  — Load a model, or change the one served (data_dir,
                             checkpoint_path or task)
    POST /pipeline/stop    — Stop inference: accepted requests finish, the model
                             and its GPU lock are released

**The two POSTs are operator controls** (D8, M5-1): each route takes the one
D8 dependency (`src/api/operator_access.py`), and does its work through one
function the dashboard calls in process too, each entrance judging its own
request: `run_reload`, which the shared Load
(`src/webui/components/model_list.py`, "Change model…" and Research › Models)
calls after its own check (the Load binds it to the model fingerprint it
showed, `expected_checkpoint_sha256`, which an API client can pass too); and
`run_stop`, the one stop. Both run in the service's one lifecycle
(`src.api.main.AppState`): one load or stop at a time, refused rather than
queued, CUDA serving under the deployment's GPU lock. **Neither runs its
work on the event loop** (BACKLOG 28): a reload's build runs in a worker
thread, and the stop closes admission on the loop as it arrives and drains in
a thread of its own, so the status and the health check answer meanwhile.
The GET stays open.
**No configuration route**: `/pipeline/config` and its writer saved a
selection to `.shepherd_ui_config.json` that nothing read back (no start-up,
no page), so they are retired (the ops round's I10); a file an operator has is
left as it is, and nothing persists "the last model".

Version: 1.0.0
"""
from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, ConfigDict, Field

from src.api.operator_access import OPERATOR_ONLY
from src.api.routes.diagnose import ServedIdentity
from src.core.types import SCORING_MODE_TASKS, TASK_SCORING_MODES
from src.utils.fingerprint import ReadOutside

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================
class PipelineStatusResponse(BaseModel):
    """Pipeline status information."""
    initialized: bool
    gnn_ready: bool = False
    scoring_mode: str = "not_initialized"
    # The served task's η, the served identity's eta: the gene task's own
    # (EMBSIM against SPLSIM, from its spec); None for a disease or
    # patients-like-me model, which scores by its own head alone.
    eta_effective: Optional[float] = 0.0
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
    checkpoint_meta: Dict[str, Any] = Field(default_factory=dict)
    #: The served checkpoint record's weight transfer, as the model lists show
    #: it (`record.listed_record`, read once when the model was loaded): each
    #: part trainable or frozen, plain data; a page names a task model whose
    #: graph backbone kept its base's weights by it. None when nothing is
    #: served or the record names none.
    weight_transfer: Optional[Dict[str, Any]] = None
    current_data_dir: Optional[str] = None
    current_checkpoint_path: Optional[str] = None
    #: The lifecycle (`AppState.snapshot`): "not_loaded", "serving", or
    #: "stopping" while a stop lets the accepted requests finish.
    serving: Literal["not_loaded", "serving", "stopping"] = "not_loaded"
    #: A load (start-up's or a reload's) is running now.
    loading: bool = False
    #: Accepted diagnoses still running.
    in_flight: int = 0
    #: This service holds the deployment's GPU lock (CUDA inference). No lock
    #: path: this route is open.
    holds_gpu_lock: bool = False
    #: Why no diagnosis is admitted now (stopping, stopped, a load or the
    #: start-up refused, nothing loaded), None while one is: the reason
    #: `/diagnose` and `/ready` refuse with. Names no path.
    not_serving_reason: Optional[str] = None
    #: The same, as its parts (`AppState.not_serving_words`), None while a
    #: diagnosis is admitted: `state` ("stopping", "loading", "not_loaded",
    #: "stopped", "load_refused" or "startup_refused"), `lead` (the reason's
    #: first words: who stopped it, why a load was refused, the GPU lock's
    #: holder), `at` (the record's ISO time) and `model_sha12`, the last two
    #: None where the record names none. Names no path.
    not_serving: Optional[Dict[str, Optional[str]]] = None


class PipelineReloadRequest(BaseModel):
    """Request to reload the pipeline with new paths.

    Five fields: `data_dir`, `checkpoint_path`, `task`, `device` and
    `expected_checkpoint_sha256`. **Any other field is refused** (HTTP 422,
    `extra_forbidden`, naming the field) by request validation, before the
    reload reads or selects anything: a field this service does not declare is
    one it cannot honour, so it is never dropped and the rest served. The
    architecture is read from the checkpoint's own model spec, never from the
    request.
    """
    model_config = ConfigDict(extra="forbid")

    data_dir: str = Field(
        ...,
        description="Path to data directory containing kg.json, node_features.pt, etc.",
    )
    checkpoint_path: Optional[str] = Field(
        None,
        description="Explicit path to a .pt checkpoint; read by its bytes wherever it "
        "is. Without it a task must be named.",
    )
    task: Optional[Literal["disease", "gene", "patients_like_me"]] = Field(
        None,
        description="The task to serve: automatic selection then takes the latest run "
        "holding a model finetuned for it, judged by each file's model record, never by "
        "its directory (a base, another task's model and a checkpoint without a model "
        "spec are never chosen); with checkpoint_path, the loaded model must be that "
        "task's or nothing is published. A reload names a task or a checkpoint_path "
        "(or both).",
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
    architecture: Optional[str] = Field(
        None, description="The loaded model spec's encoder kind, for a task or a "
        "checkpoint_path alike; null when nothing was built. A run directory's name "
        "only locates the file.")
    selection_reason: Optional[str] = None


class PipelineStopRequest(BaseModel):
    """A stop of inference (`POST /pipeline/stop`). **The body is required**:
    an empty or ill-shaped one, or one with another field, is refused in
    validation (HTTP 422) and nothing is stopped. `{}` stops whatever is
    served."""
    model_config = ConfigDict(extra="forbid")

    expected_checkpoint_sha256: Optional[str] = Field(
        None,
        pattern=r"^[0-9a-f]{64}$",
        description="The model fingerprint the caller was shown (64 lowercase hex): "
        "when given, only that model is stopped; another one keeps serving and the "
        "stop is refused. Null stops whatever is served.",
    )


class PipelineStopResponse(BaseModel):
    """What a stop did. `released` only once the accepted requests have ended
    and the model, its graph and the GPU lock are given back."""
    success: bool
    released: bool
    message: str
    status: PipelineStatusResponse


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


def _check_files(data_dir: str, checkpoint_path: Optional[str] = None) -> Dict[str, bool]:
    """Check which required files exist in data_dir."""
    d = Path(data_dir)
    result = {}
    for f in REQUIRED_DATA_FILES:
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
#: The status fields the lifecycle fills (`AppState.snapshot`).
_LIFECYCLE_FIELDS = ("serving", "loading", "in_flight", "holds_gpu_lock",
                     "not_serving_reason", "not_serving")


def _lifecycle_of(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    return {name: snapshot[name] for name in _LIFECYCLE_FIELDS}


def _status_of(
    config: Optional[Dict[str, Any]],
    data_dir: Optional[str],
    checkpoint_path: Optional[str],
    lifecycle: Optional[Dict[str, Any]] = None,
) -> PipelineStatusResponse:
    """Render one pipeline configuration as a status response, with the
    lifecycle's fields when given (`_lifecycle_of`)."""
    lifecycle = lifecycle or {}
    if config is None:
        return PipelineStatusResponse(
            initialized=False,
            current_data_dir=data_dir,
            current_checkpoint_path=checkpoint_path,
            **lifecycle,
        )
    return PipelineStatusResponse(**lifecycle,
        initialized=True,
        gnn_ready=config.get("gnn_ready", False),
        scoring_mode=config.get("scoring_mode", "unknown"),
        eta_effective=config.get("eta_effective"),
        dataset_record=config.get("dataset_record"),
        served_identity=config.get("served_identity"),
        kg_nodes=config.get("kg_nodes", 0),
        kg_edges=config.get("kg_edges", 0),
        has_model=config.get("has_model", False),
        checkpoint_meta=config.get("checkpoint_meta", {}),
        weight_transfer=config.get("weight_transfer"),
        current_data_dir=data_dir,
        current_checkpoint_path=checkpoint_path,
    )


def _still_serving() -> str:
    """The tail every rejected-candidate message carries.

    Empty unless a model is admitting requests (`AppState.accepting`): a
    service that has never built a pipeline is not "still serving" one, and
    neither is one whose stop is letting the accepted requests finish before
    it releases the model. A reassurance that is sometimes false is worth less
    than no reassurance.
    """
    from src.api.main import app_state

    if not app_state.accepting:
        return ""
    return " The previously loaded pipeline is still being served."


def _not_inside(exc: ReadOutside) -> str:
    """A bounded read's refusal (`ReadOutside`) in the reload's words: the
    file it opened lies outside the workspace, or it cannot be placed."""
    if exc.opened is not None:
        return "was opened outside the workspace"
    return f"cannot be placed inside the workspace ({exc.why})"


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

    # **One snapshot**: the pipeline, the paths it was built from and the
    # lifecycle are read in one step under admission, so they are one
    # publication's.
    snapshot = app_state.snapshot()
    pipeline = snapshot["pipeline"]
    return _status_of(
        None if pipeline is None else pipeline.get_pipeline_config(),
        snapshot["data_dir"],
        snapshot["checkpoint_path"],
        _lifecycle_of(snapshot),
    )


def _with_lifecycle(status: PipelineStatusResponse) -> PipelineStatusResponse:
    """`status` with the lifecycle's fields as they are now: what a reload
    answers once its load has left the lifecycle. Primitives only, so this
    cannot fail where the response's own encoding did not."""
    from src.api.main import app_state

    return status.model_copy(update=_lifecycle_of(app_state.snapshot()))


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
def reload_pipeline(request: PipelineReloadRequest) -> PipelineReloadResponse:
    """
    Reload the diagnosis pipeline with new data directory and/or checkpoint.

    An operator control: refused by D8's dependency before anything here runs.
    `run_reload` does the work. **A plain ``def``** (BACKLOG 28): the build
    runs in FastAPI's threadpool, never on the event loop, so the status, the
    health check and a stop are answered while it runs, and a diagnosis
    accepted on the served model meanwhile finishes on it (its lease; the
    publish swaps under admission). Another load or a stop running meanwhile
    refuses it at once. A CUDA load that finds the GPU lock held by another
    process waits for the lock's grace (`gpu_lock.GRACE_SECONDS`, a second),
    in its thread, before it is refused.
    """
    return run_reload(request)


@router.post("/pipeline/stop", response_model=PipelineStopResponse,
             dependencies=OPERATOR_ONLY)
async def stop_pipeline(request: PipelineStopRequest) -> PipelineStopResponse:
    """
    Stop inference (I1, I4): new diagnoses refused at once, the accepted ones
    finished, then the model, its graph and the GPU lock released.

    An operator control: refused by D8's dependency before anything here runs,
    and its body is required (`PipelineStopRequest`), so an empty or
    ill-shaped one is refused in validation and nothing is stopped. **The
    admission cut-off is set as the stop arrives** (checkpoint A's
    verification): this handler is ``async`` and closes admission first, on
    the event loop, before it first yields (`begin_stop`, which waits for
    nothing; D8's dependency takes no worker thread either), so no diagnosis
    arriving after it is admitted, whatever the worker threads are doing.
    Then it awaits the drain in a thread of its own (`finish_stop`), which
    runs to its end whatever becomes of the request, so a stop begun always
    releases the model and the lifecycle.
    """
    begun = begin_stop(request)
    if isinstance(begun, PipelineStopResponse):
        return begun
    return await _to_its_end_in_a_thread(lambda: finish_stop(begun))


def run_stop(request: PipelineStopRequest) -> PipelineStopResponse:
    """The one stop for a caller in a worker thread (the dashboard's Stop
    inference, in process, after D8's check on its own request): `begin_stop`,
    then `finish_stop`, as the route makes them. Blocks while the accepted
    requests finish: call it from a thread."""
    begun = begin_stop(request)
    if isinstance(begun, PipelineStopResponse):
        return begun
    return finish_stop(begun)


def begin_stop(request: PipelineStopRequest) -> Any:
    """The stop's admission cut-off (`AppState.begin_stop`): waits for
    nothing, so the route calls it on the event loop. The answer of a stop
    that does nothing (refused at once while a load or another stop runs;
    nothing loaded is "nothing to stop"; bound to a shown model, another one
    keeps serving), or the begun stop that `finish_stop` ends."""
    from src.api.main import LifecycleBusy, StopOutcome, app_state

    try:
        begun = app_state.begin_stop(
            by="operator", expected_checkpoint_sha256=request.expected_checkpoint_sha256)
    except LifecycleBusy as busy:
        return PipelineStopResponse(success=False, released=False, message=str(busy),
                                    status=_live_status())
    if isinstance(begun, StopOutcome):
        return PipelineStopResponse(success=begun.success, released=begun.released,
                                    message=begun.message, status=_live_status())
    return begun


def finish_stop(begun: Any) -> PipelineStopResponse:
    """The rest of a begun stop (`AppState.finish_stop`): the accepted
    requests waited for, then the model, its graph and the GPU lock
    released; its answer, with the status after it. Blocks: call it from a
    worker thread."""
    from src.api.main import app_state

    outcome = app_state.finish_stop(begun)
    return PipelineStopResponse(success=outcome.success, released=outcome.released,
                                message=outcome.message, status=_live_status())


async def _to_its_end_in_a_thread(work: Callable[[], Any]) -> Any:
    """`work` run in a thread of its own, started now, and its answer awaited:
    **the thread runs to its end whatever becomes of the awaiting request**
    (a client gone, a shutdown), which a worker of the threadpool, waiting
    for a free token at a cancellation point, does not promise. A stop that
    closed admission therefore always drains and gives the lifecycle back.
    Not a daemon: the interpreter waits for it before it exits."""
    loop = asyncio.get_running_loop()
    answer = loop.create_future()

    def settle(result: Any, error: Optional[BaseException]) -> None:
        if answer.done():
            return
        if error is not None:
            answer.set_exception(error)
        else:
            answer.set_result(result)

    def run() -> None:
        try:
            result, error = work(), None
        except BaseException as exc:  # noqa: BLE001 - handed to the awaiting request
            result, error = None, exc
        try:
            loop.call_soon_threadsafe(settle, result, error)
        except RuntimeError:  # the event loop has closed: no request awaits it
            pass

    threading.Thread(target=run, name="inference-stop", daemon=False).start()
    return await asyncio.shield(answer)


def _out_of_memory(exc: BaseException) -> bool:
    """Whether a build failed for memory: the host's, or the CUDA device's,
    raised as it is or wrapped by the build (a checkpoint read that runs out
    of memory is refused as unreadable, `from` its cause), so the whole chain
    of causes is read."""
    kinds: List[type] = [MemoryError]
    try:
        import torch

        kinds.append(torch.cuda.OutOfMemoryError)
    except (ImportError, AttributeError):
        pass
    seen = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, tuple(kinds)):
            return True
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return False


def run_reload(request: PipelineReloadRequest) -> PipelineReloadResponse:
    """
    The one reload: the route above and the dashboard's shared Load
    (`model_list`'s confirmation) both call this, each after D8's check on its
    own request. **Synchronous and blocking: every caller runs it in a worker
    thread** (the route is a plain ``def``, the Load's confirmation a plain
    ``def`` handler), never on the event loop (BACKLOG 28). A diagnosis
    accepted on the served model while it builds finishes on that model (its
    lease), and the publish swaps the served fields under admission.

    **It runs in the service's lifecycle** (`AppState.loading`): refused at
    once while another load or a stop runs, never queued. Its resolved device
    decides the GPU lock, held right after the device check and before any
    candidate is read; a holding it took is given back unless the model it
    published uses it. A reload refused while nothing is served is recorded as
    why nothing is (`LOAD_REFUSED`).

    Builds the replacement first and swaps it in only once it is complete and
    verified; a rejected candidate leaves the running pipeline serving, and
    its holding. **A caller that was shown a model binds the reload to it**
    (`expected_checkpoint_sha256`): the build's one checkpoint read is compared
    with it before torch parses the bytes (`read_checkpoint(expected_sha256=)`,
    inside the build), so the bytes served are the bytes confirmed or nothing
    changes, and other bytes -- a checkpoint or not, or no file -- are refused
    as not the model that was shown.
    """
    from src.api.main import LOAD_REFUSED, RELOAD_REFUSED, LifecycleBusy, app_state

    try:
        with app_state.loading() as load:
            response = _reload(request, load)
            if not response.success:
                # The service log is the operator surface that may name paths:
                # the open record of a refusal points here (`RELOAD_REFUSED`),
                # so every refused reload says why here, once.
                logger.warning("Reload refused: %s", response.message)
                shown = request.expected_checkpoint_sha256
                app_state.record_refusal(LOAD_REFUSED, load.lock_refusal or RELOAD_REFUSED,
                                         shown[:12] if shown else None)
    except LifecycleBusy as busy:
        return _refused(f"{busy}{_still_serving()}", {})
    return response.model_copy(update={"status": _with_lifecycle(response.status)})


def _reload(request: PipelineReloadRequest, load: Any) -> PipelineReloadResponse:
    """`run_reload`'s work inside the lifecycle; `load` is its `ServingLoad`."""
    data_dir = request.data_dir
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
    from src.api.main import ServingRefused
    from src.config.serving_device import requested_serving_device
    from src.inference.pipeline import PipelineBuildError, resolve_serving_device

    try:
        resolved = resolve_serving_device(requested_serving_device(device))
    except PipelineBuildError as exc:
        return _refused(f"{str(exc).rstrip('.')}.{_still_serving()}", files)

    # **The GPU lock, by that device, before any candidate is read** (I2, I3):
    # CUDA holds the deployment's one lock, the one already held when a CUDA
    # model is served (CUDA to CUDA), and reads the job runner's slot after
    # taking it; a busy or unconfigured lock, or a job in the slot, refuses
    # here. The CPU takes none.
    try:
        load.hold(resolved)
    except ServingRefused as refused:
        return _refused(f"{refused.reason}{_still_serving()}", files)

    # **Neither a task nor a path names nothing to serve** (SG2): refused here,
    # after the file and device checks (the file report stays) and before the
    # scan, so no candidate is read.
    if not checkpoint_path and request.task is None:
        return _refused(
            f"Name a task ({', '.join(TASK_SCORING_MODES)}) or a checkpoint_path: "
            "automatic selection serves the latest model finetuned for the task named, "
            f"and nothing is chosen without one.{_still_serving()}",
            files, selection_reason="neither task nor checkpoint_path")

    # Resolve which checkpoint to serve, from the run layout
    # ({data_dir}/runs/{layout key}/{run}/checkpoints/, contract M3a-2). Priority:
    #   1. explicit checkpoint_path (any location; accepted by its bytes)
    #   2. a task -> the latest run holding a model finetuned for it, judged by
    #      each file's model record, its best/last/newest
    # Runs are not ranked against each other by metric. Checkpoints in an older
    # layout are NOT scanned here (see checkpoint_paths), and neither is a file
    # that resolves outside the workspace: 2 keeps the listing's boundary and
    # never reads one. **2's reads bind that boundary to the file each one
    # opens** (`within=workspace`, judged on the open descriptor), the scan's
    # and the build's alike: a name swapped for a link out of the workspace
    # after the boundary check is refused before a byte of it is read, never
    # read, chosen or published. 1 is an operator's explicit choice (D8), read
    # by its bytes wherever it is.
    from src.models.bundle.record import has_model_spec, listed_record
    from src.utils.checkpoint_io import read_checkpoint
    from src.utils.checkpoint_paths import (
        ranking_score_detail,
        select_auto_run_checkpoint,
        with_not_read,
    )

    #: The workspace automatic selection reads inside, resolved once: the
    #: boundary `select_auto_run_checkpoint` checks names against.
    workspace = Path(data_dir).resolve()

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
    #: The candidates the scan's bounded read refused (opened outside the
    #: workspace, or not placeable), in the listing's words.
    opened_outside: List[str] = []

    def _inspect(path: Path) -> Optional[Dict[str, Any]]:
        if str(path) not in inspected:
            try:
                # Tensors and plain data only (contract decision 10): this scan
                # reads files in a directory a request names, and only a file
                # its open finds inside the workspace.
                scored = read_checkpoint(path, map_location="cpu", weights_only=True,
                                         within=workspace)
            except ReadOutside as exc:
                inspected[str(path)] = None
                opened_outside.append(
                    f"{path.parent.parent.name}/{path.name} {_not_inside(exc)}, so it is not read")
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
        """The record decides (provenance §9.3, hazard 3): a model bundle
        finetuned for the task asked for. A file the scan could not read is
        judged by nothing, so it is never a candidate."""
        seen = _inspect(path)
        if seen is None:
            return False
        record = seen["record"]
        return (record is not None and record["record"] == "model"
                and record["learning_stage"] == "finetune"
                and record["task"] == request.task)

    # selection_reason = how the file was chosen. The response's architecture
    # is the built model spec's encoder kind (below), for a path or a task
    # alike: a run's layout key only locates the file.
    selection_reason: str = ""
    #: The file automatic selection chose; None for an explicit path.
    selected: Optional[Path] = None
    #: The files automatic selection did not read for the workspace boundary,
    #: in the listing's words (all of them; the reason names the first few).
    not_read: List[str] = []

    if checkpoint_path:
        selection_reason = "explicit checkpoint_path"
    else:
        # Every run-layout key: a model bundle is filed under its encoder kind,
        # and the record, never the directory, says what it is.
        selected, _layout_key, selection_reason = select_auto_run_checkpoint(
            data_dir, score_fn=_checkpoint_score, accept=_accept, not_read=not_read)
        # The files the scan's bounded reads refused are not read for the same
        # boundary: named in the reason, in the listing's words, and counted in
        # the hint below, whether another file is chosen or none.
        not_read.extend(opened_outside)
        selection_reason = with_not_read(selection_reason, opened_outside)
        selection_reason = f"task {request.task}: {selection_reason}"
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
        # Say why nothing is served: the explicit file is missing, the runs
        # resolve outside the workspace, or no run holds the task's model.
        # Checkpoints outside the run layout stay in files_found and are
        # reached by an explicit path (decision 9: no runtime fallback).
        if request.checkpoint_path:
            hint = f"{request.checkpoint_path} does not exist."
        elif selected is None and not_read:
            # Nothing chosen, and some of the runs not read: files behind a
            # link out of the workspace, or a directory under runs/ this
            # account cannot list (each named in the reason). Whether a new
            # run can be trained here depends on where the link is
            # (`run_dir_outside` refuses only a run that would land outside),
            # so what was not read is stated and the task's own hint follows.
            hint = (f"Automatic selection did not read {len(not_read)} item(s) of the "
                    "workspace's runs (named above: a checkpoint file that resolves outside "
                    "the workspace, or a directory under runs/ that cannot be listed); keep "
                    "the runs inside the workspace (runs/ and its directories not links out "
                    "of it) and listable, or pass an explicit checkpoint_path."
                    f" No run it read holds a model finetuned for the {request.task} task.")
        else:
            hint = (f"No run holds a model finetuned for the {request.task} task; a base, "
                    "another task's model and a checkpoint without a model spec are not "
                    "chosen for it. Finetune one, or pass an explicit checkpoint_path.")
        return _refused(
            f"No checkpoint found ({selection_reason}). {hint}", files,
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
    #
    # **An automatically selected file is read only inside the workspace**
    # (`checkpoint_within`): the build's one checkpoint read refuses a file its
    # open finds outside, by the open descriptor and before any byte, so a name
    # swapped for a link out of the workspace after the selection is never read
    # or published (`ReadOutside`). Before the expected digest, which that same
    # read compares next. An explicit checkpoint_path passes no bound.
    try:
        candidate = build_pipeline(
            kg_path=kg_path,
            checkpoint_path=checkpoint_path,
            data_dir=data_dir,
            device=device,
            expected_checkpoint_sha256=expected,
            checkpoint_within=workspace if selected is not None else None,
        )
    except CheckpointChanged as exc:
        held = (f"it holds {exc.found[:12]} now, not {exc.expected[:12]}"
                if exc.found is not None else f"it is gone; {exc.expected[:12]} was shown")
        return _refused(
            f"{checkpoint_path} is not the model that was shown: {held}. Nothing was "
            f"published; read it again to see what it holds.{_still_serving()}", files
        )
    except ReadOutside as exc:
        return _refused(
            f"{checkpoint_path} {_not_inside(exc)}: automatic selection reads only files "
            "inside the workspace, so nothing was read from it or published."
            f"{_still_serving()}", files
        )
    except Exception as e:
        if _out_of_memory(e):
            # **Never by unloading first** (I5): the candidate is refused, and
            # the model in use keeps serving and its holding.
            keeps = (" The model in use keeps serving; stop inference first to free its "
                     "memory, then load again." if _still_serving() else "")
            return _refused(
                f"Pipeline initialization failed: out of memory while the candidate was "
                f"built ({str(e).rstrip('.')}).{keeps} Nothing was unloaded to make room.",
                files)
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
        served_task = SCORING_MODE_TASKS.get(served_mode)
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

        msg = "Pipeline reloaded successfully."
        if selection_reason:
            msg += f" ({selection_reason})"

        response = PipelineReloadResponse(
            success=True,
            message=msg,
            status=_status_of(config, data_dir, checkpoint_path),
            files_found=files,
            checkpoint_path=checkpoint_path,
            # What runs is what the built model's own spec names, never the
            # directory the file is filed under; read here, before publication.
            architecture=candidate.pipeline.model.spec.encoder.kind,
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
