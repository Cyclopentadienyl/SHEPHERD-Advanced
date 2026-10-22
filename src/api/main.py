"""
SHEPHERD-Advanced API Service
=============================
FastAPI application for rare disease diagnosis.

Module: src/api/main.py
Absolute Path: /home/user/SHEPHERD-Advanced/src/api/main.py

Purpose:
    Main FastAPI application entry point providing:
    - REST API endpoints for diagnosis, HPO search, disease info
    - Health check and readiness probes
    - CORS and security middleware
    - OpenAPI documentation

Endpoints:
    POST /api/v1/diagnose     - Run diagnosis on patient phenotypes
    GET  /api/v1/hpo/search   - Search HPO terms
    GET  /api/v1/disease/{id} - Get disease information
    GET  /api/v1/evaluations/loaded-model
                              - The loaded checkpoint's recorded test results
                                (read-only; contract M4)
    GET  /health              - Health check
    GET  /ready               - Readiness probe

Dependencies:
    - fastapi: Web framework
    - uvicorn: ASGI server
    - src.inference.pipeline: DiagnosisPipeline
    - src.ontology: HPO ontology access
    - src.kg: Knowledge graph access

Input:
    - HTTP requests with JSON payloads
    - Patient phenotypes as HPO term lists

Output:
    - JSON responses with diagnosis results
    - OpenAPI schema at /docs

Called by:
    - Frontend WebUI
    - External API clients
    - Health monitoring systems

Version: 1.0.0
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.routing import Mount

from src.api.middleware.frame_guard import RefuseFraming
from src.api.middleware.gradio_routes import RefuseUnusedGradioRoutes
from src.api.middleware.server_address import RecordServerAddress
from src.config.serving_device import requested_serving_device

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Application State
# =============================================================================
class AppState:
    """Application state container"""

    def __init__(self):
        self.pipeline = None
        self.kg = None
        self.ontology = None
        #: Startup's half of what `/ready` answers; the other half is
        #: `requested_pipeline_missing`. Set when startup completes, cleared
        #: when startup's build fails and at shutdown, and set again by
        #: `publish_pipeline` -- so a reload that succeeds after a failed
        #: startup restores it, and a refused reload leaves the flag as it was.
        self.is_ready = False
        self.start_time = None
        #: The service's own version. What a diagnosis was produced by is not
        #: kept here: each result carries its served identity (contract M3c),
        #: and shared state a reload swaps cannot say which pipeline answered.
        self.version = "1.0.0"
        # Which workspace and checkpoint the loaded pipeline was built from.
        # Declared here rather than attached by whoever loaded it, so "what is
        # being served?" has an answer from the first moment it can be asked —
        # including after an environment-configured startup, which used to
        # publish a pipeline and leave these reading as unknown.
        self._current_data_dir = None
        #: Whether any real workspace has ever been named for this service —
        #: through `SHEPHERD_KG_PATH` at startup, or through a reload that
        #: pointed at one. It records a **request**, not an outcome, so it stays
        #: True after a build that failed. `/diagnose` needs exactly that: the
        #: difference between "nobody configured a pipeline" and "a pipeline was
        #: configured and cannot answer" is not visible from `pipeline is None`.
        self.real_pipeline_requested = False
        self._current_checkpoint_path = None

    @property
    def requested_pipeline_missing(self) -> bool:
        """A real pipeline was asked for and none is being served.

        **One definition, two surfaces.** `/diagnose` refuses with 503 in this
        state, and `/ready` reports it from the same definition, so the probe
        cannot call ready a service whose every diagnosis is refused -- a
        configured graph file that is missing at startup, or a refused reload
        on a service that had no pipeline, leave `is_ready` set and this true.
        """
        return self.pipeline is None and self.real_pipeline_requested


app_state = AppState()


# =============================================================================
# Lifespan Management
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager

    Handles startup and shutdown events.

    **The bind guard's third enforcement point** (SPEC_4 §3.1), first: the
    host the launcher or `main()` passed in (`SHEPHERD_BIND_HOST`) goes
    through the same rule they applied (`server_settings.startup_bind`), and
    a refused one ends startup by raising, so uvicorn exits without serving.
    None passed in (a uvicorn started by hand) is not checked, as §3.1 says.

    **Stopping the backend stops its jobs** (the owner's decision of
    2026-10-22): each GPU job leads its own process group (review of 8242b47,
    P2-3), so a Ctrl-C, a Ctrl-\\ or a hangup of the terminal the backend
    runs in no longer reaches it, and neither does a SIGTERM sent to the
    backend alone (`JOB_DETACHED_SIGNALS`).
    So the backend stops the job its runner holds (`end_active_job`: SIGINT,
    then SIGTERM, then SIGKILL to its group) **from the signal on**, and ends
    the dashboard's heartbeat streams at the same signal (`_ShutdownSignals`):
    uvicorn waits for every open connection to close before it runs the
    lifespan's end, and an open dashboard tab keeps one open, so a stop begun
    there waited for every tab to be closed. A hangup (SIGHUP, a closed
    terminal) and a Ctrl-\\ (SIGQUIT), which uvicorn does not handle, are
    routed to its SIGTERM handler while the application runs, so it shuts
    down the same way (the backend still ends by SIGQUIT after it). **The
    stop closes the job runner's GPU slot first** (`TrainingManager.close`,
    root-cause review P23-1), so no job starts while uvicorn waits for its
    connections: a start still under way at the signal is refused, by every
    route. The lifespan's end waits for that stop, then closes the slot and
    stops its job itself when no signal began the shutdown (an in-process
    client, a server off the main thread). A second Ctrl-C (uvicorn's forced
    quit, which stops waiting for connections and skips the lifespan's
    shutdown event) leaves no job either: the stop began at the first, and
    the backend waits for it before it exits. Not covered: a SIGKILL or the out-of-memory killer; the
    job then keeps running and keeps the GPU lock until it is stopped by its
    pid. Under systemd the unit's ``KillMode=mixed`` sends its stop's SIGTERM
    to the server alone, which stops the job this same way, and systemd's
    SIGKILL then finds nothing of it (root-cause review, P23-3).
    """
    _check_startup_bind()

    # Startup
    logger.info("Starting SHEPHERD-Advanced API...")
    app_state.start_time = datetime.now()

    # Suppress harmless ProactorEventLoop pipe cleanup errors on Windows.
    # These ConnectionResetError exceptions come from Gradio WebSocket/SSE
    # connections closing and are non-critical noise.
    if sys.platform == "win32":
        loop = asyncio.get_running_loop()
        _default_handler = loop.get_exception_handler()

        def _quiet_connection_reset(loop, context):
            exception = context.get("exception")
            if isinstance(exception, ConnectionResetError):
                logger.debug(
                    "Suppressed ConnectionResetError: %s",
                    context.get("message", ""),
                )
                return
            if _default_handler:
                _default_handler(loop, context)
            else:
                loop.default_exception_handler(context)

        loop.set_exception_handler(_quiet_connection_reset)

    try:
        # Attempt eager initialization if a pipeline is configured. A checkpoint
        # alone counts: `build_pipeline` refuses it, and startup is where an
        # operator should see that, not the first diagnosis.
        if os.environ.get("SHEPHERD_KG_PATH") or os.environ.get(
            "SHEPHERD_CHECKPOINT_PATH"
        ):
            initialize_pipeline()
        else:
            logger.info(
                "No SHEPHERD_KG_PATH set. Pipeline will initialize "
                "lazily on first request if configured later."
            )

        app_state.is_ready = True
        logger.info("API service ready")
    except Exception as e:
        # Not ready until a pipeline is published; a reload that succeeds does
        # that (`publish_pipeline`), so this is not a state only a restart clears.
        logger.error(f"Startup failed: {e}")
        app_state.is_ready = False

    _open_the_gpu_slot()
    shutdown_signals = _ShutdownSignals(app)
    shutdown_signals.install()
    try:
        yield
    finally:
        # Shutdown
        logger.info("Shutting down SHEPHERD-Advanced API...")
        app_state.is_ready = False
        try:
            await asyncio.to_thread(shutdown_signals.end_jobs)
        finally:
            shutdown_signals.restore()


class _ShutdownSignals:
    """**The job's stop starts at the signal that stops the backend**
    (verification of the owner's decision, 2026-10-22). uvicorn's shutdown
    waits for every open connection to close ("Waiting for connections to
    close") before it runs the lifespan's end, and each open dashboard tab
    keeps one open: Gradio's session heartbeat, which the dashboard asks for
    because it keeps session state, and which mounted ends only when the tab
    is closed. A stop begun at the lifespan's end left the job training, and
    holding the GPU lock, until every tab was closed; a SIGTERM or a closed
    terminal has no second signal to force the shutdown on.

    So, while the application runs, **every signal that no longer reaches a
    job because it leads its own group** (`JOB_DETACHED_SIGNALS`, defined
    once beside the spawn; root-cause review P23-2) ends in the server's
    orderly shutdown: each one whose handler is the server's own, the one
    SIGTERM has, is wrapped (uvicorn's SIGINT and SIGTERM, and SIGBREAK on
    Windows), and each other one that still has its default action (SIGHUP,
    a closed terminal; SIGQUIT, the terminal's Ctrl-\\, which used to leave
    the job training with the lock) gets the SIGTERM handler, wrapped the
    same way. **The server's handlers are told by the handler, never by a
    list of signal names** (verification of P23-2, T2-V4), so whatever set
    of signals the server handles itself, each is wrapped. SIGQUIT keeps its meaning: uvicorn
    raises the signal it captured again once it has shut down, so the backend
    still ends by SIGQUIT, after the job's stop. Each runs the server's
    handler as before (a second Ctrl-C is still uvicorn's forced quit), then
    hands the event loop `begin`: the first time, the job's stop starts in a
    thread of its own (it closes the GPU slot, then stops the job that held
    it: `_stop_the_active_job`), and every time, the dashboard's
    heartbeat streams end (`_end_dashboard_streams`), so an open tab holds up
    neither the job nor the backend. The thread is not a daemon, so the
    interpreter waits for it before it exits, after a forced quit too.
    `end_jobs`, at the lifespan's end, waits for it, then does the same
    itself, which finds the slot already closed and empty after a signal.

    Installed only in the main thread, where a server sets its handlers (an
    in-process test client runs the lifespan elsewhere), and only when
    SIGTERM has a handler to route to (a server's); a handler that is not the
    server's is not wrapped (SIGINT's default handler, which no server set,
    or one of someone else's), and a routed signal is taken only where it
    still has its default action (SIGHUP ignored under ``nohup`` is left as
    it is). `restore` puts back each
    handler replaced here that is still the one installed here (after a
    forced quit uvicorn has already put back its own).
    """

    def __init__(self, application: FastAPI) -> None:
        self._application = application
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        #: signal -> (the handler it had, the one installed here)
        self._installed: Dict[int, Tuple[Any, Callable[[int, Any], None]]] = {}
        self._stopping: Optional[threading.Thread] = None

    def install(self) -> None:
        """Wrap the server's handlers (each listed signal whose handler is
        SIGTERM's), and route the other signals that no longer reach a job
        (`JOB_DETACHED_SIGNALS`) to its SIGTERM handler; in the event loop,
        at startup."""
        from src.api.services.training_manager import JOB_DETACHED_SIGNALS

        if threading.current_thread() is not threading.main_thread():
            return
        termination = signal.getsignal(signal.SIGTERM)
        if not callable(termination):
            return
        self._loop = asyncio.get_running_loop()
        for name in JOB_DETACHED_SIGNALS:
            number = getattr(signal, name, None)
            if number is None:
                continue
            handler = signal.getsignal(number)
            if handler is signal.SIG_DFL:
                self._wrap(number, termination, replaced=signal.SIG_DFL)
            elif handler == termination:  # the server's own: a bound method, compared equal
                self._wrap(number, handler, replaced=handler)

    def _wrap(self, number: int, server_handler: Callable[[int, Any], Any],
              replaced: Any) -> None:
        def handler(signum: int, frame: Any) -> None:
            try:
                server_handler(signum, frame)
            finally:
                self._hand_to_the_loop()

        signal.signal(number, handler)
        self._installed[number] = (replaced, handler)

    def _hand_to_the_loop(self) -> None:
        """In the signal handler, only a callback handed to the event loop
        (`call_soon_threadsafe` also wakes it): a thread started or a lock
        taken here could wait on one the interrupted main thread holds."""
        try:
            self._loop.call_soon_threadsafe(self.begin)
        except RuntimeError:  # the event loop has closed: nothing is served
            pass

    def begin(self) -> None:
        """In the event loop: the job's stop started (once), the dashboard's
        heartbeat streams ended."""
        if self._stopping is None:
            self._stopping = threading.Thread(target=_stop_the_active_job, daemon=False,
                                              name="backend-stops-its-job")
            self._stopping.start()
        _end_dashboard_streams(self._application)

    def end_jobs(self) -> None:
        """At the lifespan's end: the stop a signal began waited for, then the
        slot closed and its job stopped here, which only a shutdown that no
        signal began relies on (after a signal the slot is closed already, so
        no job started meanwhile). Blocks; call it from a thread."""
        if self._stopping is not None:
            self._stopping.join()
        _stop_the_active_job()

    def restore(self) -> None:
        """Put back each handler replaced here that is still ours; at the
        lifespan's end, in the event loop."""
        for number, (replaced, installed) in self._installed.items():
            if signal.getsignal(number) is installed:
                signal.signal(number, replaced)
        self._installed.clear()


def _end_dashboard_streams(application: FastAPI) -> None:
    """End the session heartbeat streams of the dashboard mounted at
    `DASHBOARD_PATH`, as Gradio's own server does when it stops
    (`Blocks._end_streaming_responses` sets the served app's `stop_event`;
    mounted, nothing sets it). Each open tab keeps one, which otherwise ends
    only when the tab is closed. In the event loop; with no dashboard
    mounted, there is nothing to end."""
    for route in application.routes:
        if isinstance(route, Mount) and route.path == DASHBOARD_PATH:
            stop_event = getattr(route.app, "stop_event", None)
            if stop_event is not None:
                stop_event.set()


def _stop_the_active_job() -> None:
    """At shutdown: the job runner's GPU slot closed, so no job starts from
    here on, whatever request is still being handled (root-cause review,
    P23-1), and the job that held it stopped, or one an operator is already
    stopping waited for (`TrainingManager.close`, `end_active_job`), bounded
    by the stop's escalation; said in the log either way. The slot is closed
    and its job read in one step, so a start either came first and is the
    job stopped here, or comes after and is refused."""
    from src.api.services.training_manager import training_manager

    active = training_manager.close()
    if active is None:
        return
    logger.warning("The backend is stopping, so it stops the %s job %s it runs (pid %s): "
                   "SIGINT, then SIGTERM, then SIGKILL to its process group.",
                   active["job_type"], active["job_id"], active["pid"])
    ended = training_manager.end_active_job()
    if ended is None:
        return
    if ended.get("success"):
        logger.info("The %s job %s ended as %s.", ended["job_type"], ended["job_id"],
                    ended.get("status"))
    else:
        logger.warning("The %s job %s: %s", ended["job_type"], ended["job_id"],
                       ended.get("error") or f"still {ended.get('status')}")


def _open_the_gpu_slot() -> None:
    """At startup: the job runner's GPU slot open. A shutdown closes it
    (`_stop_the_active_job`), and the runner is a process-wide singleton, so
    an in-process client's next lifespan would otherwise find it closed."""
    from src.api.services.training_manager import training_manager

    training_manager.open()


def _check_startup_bind() -> None:
    """Raise `RuntimeError` when the host passed in to startup is a bind
    `network_bind` refuses; log the warning of an accepted network bind."""
    from src.config.server_settings import ENV_BIND_HOST, startup_bind

    decision = startup_bind()
    if decision is None:
        return
    if not decision.allowed:
        message = (f"Startup refused: {ENV_BIND_HOST}, the address the launcher or "
                   f"src.api.main:main() passed in, is {decision.host!r}; {decision.message}")
        logger.error(message)
        raise RuntimeError(message)
    if decision.message:
        logger.warning(decision.message)


# =============================================================================
# FastAPI Application
# =============================================================================
app = FastAPI(
    title="SHEPHERD-Advanced API",
    description="Rare Disease Diagnosis Engine with Knowledge Graph Reasoning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# **By default no cross-origin browser access is granted; administrative
# operations are still checked by D8 in the backend** (M5-1). An empty allow
# list sends no Access-Control-Allow-Origin to any other origin and answers
# every preflight 400, on /api and on the mounted dashboard alike: since Gradio
# 6.25 a mounted app adds no CORS headers of its own when the parent's
# middleware is exactly this class (`middleware.cls is CORSMiddleware`), so
# **it must stay `fastapi.middleware.cors.CORSMiddleware` itself, never a
# subclass**, or Gradio's own middleware reflects any origin on a LAN Host.
# CORS only decides what a browser lets a page read: a cross-origin request
# that needs no preflight still reaches the server, which is why D8, not CORS,
# guards the operator controls.
app.add_middleware(CORSMiddleware, allow_origins=[])

# The in-process WebUI calls this API at the address the server accepted its
# requests on, recorded from each request (`src/utils/server_address.py`).
app.add_middleware(RecordServerAddress)

#: Where the Gradio dashboard is mounted (below).
DASHBOARD_PATH = "/ui"

# The routes Gradio registers on the dashboard with no switch to turn them off
# and the dashboard does not use (`file=<URL>`, fetched server-side,
# `process_recording`, and `upload` with its progress stream, which another
# site's page can post to) are refused here, before the body is read. Added
# after RecordServerAddress and before the request logging below, so it runs
# inside the logging and each refusal is logged.
app.add_middleware(RefuseUnusedGradioRoutes, mount_path=DASHBOARD_PATH)

# No other page may frame the dashboard or the API (clickjacking): D8 sees the
# request's arrival, not the frame it was made from, so this is the control
# (`src/api/middleware/frame_guard.py`). Added after the guard above, so a
# refusal carries the headers too.
app.add_middleware(RefuseFraming)


# =============================================================================
# Request/Response Logging Middleware
# =============================================================================
# Paths that generate high-frequency polling noise — suppress from logs
_QUIET_PREFIXES = (
    "/ui/gradio_api/queue/",
    "/ui/gradio_api/heartbeat/",
    # Events run without the queue (M5-2): the header chips' read on the
    # shell's timer, every 5 s per open page, is a plain POST here.
    "/ui/gradio_api/run/",
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing (except Gradio internal polling)"""
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time

    if not request.url.path.startswith(_QUIET_PREFIXES):
        logger.info(
            f"{request.method} {request.url.path} "
            f"- {response.status_code} "
            f"- {process_time:.3f}s"
        )

    response.headers["X-Process-Time"] = str(process_time)
    return response


# =============================================================================
# Exception Handlers
# =============================================================================
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url.path),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.exception(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "path": str(request.url.path),
        },
    )


# =============================================================================
# Health & Readiness Endpoints
# =============================================================================
@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """
    Basic health check

    Returns service status and uptime.
    """
    uptime = None
    if app_state.start_time:
        uptime = (datetime.now() - app_state.start_time).total_seconds()

    return {
        "status": "healthy",
        "version": app_state.version,
        "uptime_seconds": uptime,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/ready", tags=["Health"])
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness probe

    Returns whether the service is ready to accept requests: startup succeeded
    or a pipeline has since been published (`is_ready`), and the service is not
    in the state `/diagnose` refuses (`AppState.requested_pipeline_missing`).
    That state is checked first, so a failed startup -- the commonest way into
    it -- is told how to recover, as `/diagnose` tells it.
    """
    if app_state.requested_pipeline_missing:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "A pipeline is configured and none is being served; load one "
                "(POST /api/v1/pipeline/reload) or restart the service."
            ),
        )
    if not app_state.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready",
        )

    return {
        "ready": True,
        "pipeline_loaded": app_state.pipeline is not None,
        "kg_loaded": app_state.kg is not None,
        "ontology_loaded": app_state.ontology is not None,
    }


# =============================================================================
# Import and Register Routes
# =============================================================================
from src.api.routes import diagnose, search, disease, training, system, pipeline, evaluations, lists

app.include_router(diagnose.router, prefix="/api/v1", tags=["Diagnosis"])
app.include_router(pipeline.router, prefix="/api/v1", tags=["Pipeline"])
# A router of its own, outside `pipeline.router`: a deployment that unmounts or
# protects that router (SPEC_4 B2, the C3 class) keeps this read-only view.
app.include_router(evaluations.router, prefix="/api/v1", tags=["Evaluations"])
app.include_router(search.router, prefix="/api/v1", tags=["Search"])
app.include_router(disease.router, prefix="/api/v1", tags=["Disease"])
app.include_router(training.router, prefix="/api/v1", tags=["Training"])
app.include_router(system.router, prefix="/api/v1", tags=["System"])
# The server's lists (M5-1): workspaces and cohort versions, open; a workspace's
# checkpoints, behind D8's dependency (SPEC_4 C3; routes/lists.py).
app.include_router(lists.router, prefix="/api/v1", tags=["Lists"])


# =============================================================================
# Mount Static Files (fonts, etc.) — must be BEFORE Gradio mount
# =============================================================================
from starlette.staticfiles import StaticFiles

_static_dir = Path(__file__).parent.parent / "webui" / "static"
if _static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
    logger.info(f"Static files mounted at /static from {_static_dir}")

# =============================================================================
# Mount Gradio Dashboard
# =============================================================================
# `mount_dashboard` passes the theme, CSS and switches to the mount, where
# Gradio 6 reads them (src/webui/app.py).
try:
    from src.webui.app import mount_dashboard

    mount_dashboard(app, DASHBOARD_PATH)
    logger.info(f"Gradio dashboard mounted at {DASHBOARD_PATH}")
except ModuleNotFoundError as e:
    if (e.name or "").split(".")[0] != "gradio":
        logger.error(f"The dashboard could not be imported, so it is not mounted: {e}")
    else:
        logger.warning(f"Gradio not available, dashboard disabled: {e}")
except ImportError as e:
    # Not a missing Gradio: a defect (a circular import, say), said as one.
    logger.error(f"The dashboard could not be imported, so it is not mounted: {e}")
except Exception as e:
    logger.warning(f"Failed to mount Gradio dashboard: {e}")


# =============================================================================
# PWA Manifest (silences browser 404 for /manifest.json)
# =============================================================================
@app.get("/manifest.json", include_in_schema=False)
async def manifest() -> JSONResponse:
    """Minimal PWA manifest for the Gradio dashboard."""
    return JSONResponse({
        "name": "SHEPHERD-Advanced Dashboard",
        "short_name": "SHEPHERD",
        "start_url": "/ui",
        "display": "standalone",
        "background_color": "#ffffff",
        "theme_color": "#2563eb",
    })


# =============================================================================
# Chrome DevTools config (silences 404 when F12 is open)
# =============================================================================
@app.get("/.well-known/appspecific/com.chrome.devtools.json", include_in_schema=False)
async def chrome_devtools_config() -> JSONResponse:
    """Empty config to silence Chrome DevTools 404."""
    return JSONResponse({})


# =============================================================================
# Root Endpoint
# =============================================================================
@app.get("/", tags=["Root"])
async def root() -> Dict[str, Any]:
    """API root - returns basic information"""
    return {
        "name": "SHEPHERD-Advanced API",
        "version": app_state.version,
        "description": "Rare Disease Diagnosis Engine",
        "docs": "/docs",
        "health": "/health",
    }


# =============================================================================
# Utility Functions
# =============================================================================
def get_app_state() -> AppState:
    """Get application state (for dependency injection)"""
    return app_state


class PipelineBundle(NamedTuple):
    """A constructed, fully checked pipeline that nothing is serving yet.

    Construction is the part that can fail: the graph is loaded, the workspace
    is checked against the composition it claims, a checkpoint is opened and a
    model is rebuilt from it. Naming the finished product lets that whole
    sequence happen in local variables, so a caller replacing a live pipeline
    can decide *after* it succeeded.
    """

    kg: Any
    pipeline: Any
    config: Dict[str, Any]
    data_dir: Optional[str]
    checkpoint_path: Optional[str]


def build_pipeline(
    kg_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    device: Optional[str] = None,
    expected_checkpoint_sha256: Optional[str] = None,
) -> Optional[PipelineBundle]:
    """
    Build a diagnosis pipeline without publishing it.

    Loads the knowledge graph, creates the DiagnosisPipeline with optional GNN,
    and reads back its configuration — everything that can fail. **Application
    state is not touched.** Returns the finished bundle, or ``None`` when there
    is nothing to build because no KG path is configured or the configured one
    does not exist; raises when a configured pipeline fails to build.

    Configuration is read from environment variables if not passed directly:
        SHEPHERD_KG_PATH: Path to KG JSON file. It may name a file outside
            ``SHEPHERD_DATA_DIR`` holding the bytes the workspace manifest binds;
            the graph checked is the one parsed from this file, and
            ``data_dir/kg.json`` is then not read
        SHEPHERD_CHECKPOINT_PATH: Path to model checkpoint
        SHEPHERD_DATA_DIR: Path to processed graph data directory
        SHEPHERD_DEVICE: The model's device: unset or ``auto`` for CUDA,
            refused without it; ``cuda``, ``cuda:N``, or ``cpu`` for development
            (contract M3c, `resolve_serving_device`)

    Args:
        kg_path: Path to KG JSON file (overrides env var)
        checkpoint_path: Path to trained model checkpoint (overrides env var)
        data_dir: Path to processed data directory (overrides env var)
        device: Inference device (overrides env var)
        expected_checkpoint_sha256: The checkpoint's SHA-256 a caller was shown
            (a bound reload); the build's one checkpoint read is compared with
            it before the bytes are parsed, and other bytes, or no file, raise
            `CheckpointChanged` (`DiagnosisPipeline`). None at startup.

    Returns:
        The built pipeline and the paths it was built from, or None if the
        service is not configured to serve one.
    """
    logger.info("Building diagnosis pipeline...")

    # Resolve paths from args or environment. **Blank is unset**: an
    # exported-but-empty variable is how a shell says nothing, and an empty
    # checkpoint path read as a request for a model would refuse a deployment
    # that asked for none.
    kg_path = kg_path or os.environ.get("SHEPHERD_KG_PATH") or None
    checkpoint_path = (
        checkpoint_path or os.environ.get("SHEPHERD_CHECKPOINT_PATH") or None
    )
    data_dir = data_dir or os.environ.get("SHEPHERD_DATA_DIR") or None
    device = requested_serving_device(device)

    if not kg_path:
        if checkpoint_path:
            # **A model was asked for, so this is not "nothing configured".**
            # Returning None here sent the service to its demo answer, invented
            # candidates over HTTP 200, for a deployment that named a checkpoint.
            app_state.real_pipeline_requested = True
            from src.inference.pipeline import PipelineBuildError

            raise PipelineBuildError(
                f"A checkpoint is configured ({checkpoint_path}) and no "
                "knowledge graph is; set SHEPHERD_KG_PATH. The model cannot be "
                "served without the graph it scores over."
            )
        logger.warning(
            "No KG path configured. Set SHEPHERD_KG_PATH or pass kg_path. "
            "Pipeline will not be available."
        )
        return None

    # **Recorded here because this is where both entry points meet.** Startup
    # resolves `SHEPHERD_KG_PATH` and a reload passes a workspace it derived from
    # the request; keying the distinction on the environment variable alone
    # covered only the first, so a caller who named a real workspace through the
    # reload API and had it refused still received mock candidates. This is the
    # request, not its result — it stays True when the build below fails, which
    # is the whole point — and it touches nothing the published bundle owns, so
    # the build-then-publish separation is unchanged.
    app_state.real_pipeline_requested = True

    try:
        from src.kg.graph import KnowledgeGraph
        from src.inference.pipeline import (
            create_diagnosis_pipeline,
            resolve_serving_device,
        )

        # **The device is settled before the graph is read** (contract M3c):
        # `auto` without CUDA refuses before a multi-gigabyte kg.json is parsed
        # for a model that cannot be served here. The pipeline applies the same
        # rule itself; the request is passed on as given, so the served identity
        # records what was asked for beside what took effect.
        if checkpoint_path:
            resolve_serving_device(device)

        # Step 1: Load knowledge graph
        kg_file = Path(kg_path)
        if not kg_file.exists():
            logger.error(f"KG file not found: {kg_file}")
            return None

        logger.info(f"Loading knowledge graph from {kg_file}...")
        # **One read, kept whole** (contract M2.1, S6). The graph and the
        # identity of the bytes it was parsed from travel together into the
        # pipeline, which checks that identity against the workspace manifest
        # when it builds a GNN over `data_dir`; a pipeline with no checkpoint
        # checks nothing (BACKLOG 19.20). Nothing hashes the file again by path.
        graph = KnowledgeGraph.read_json(kg_file)
        kg = graph.kg
        logger.info(f"KG loaded: {kg.total_nodes} nodes, {kg.total_edges} edges")

        # Step 2: Create pipeline (with optional GNN)
        #
        # **The graph read is passed whole, and the composition is checked
        # there.** These two settings are resolved independently from the
        # environment, so a deployment can point them at two different
        # workspaces — each internally consistent, and together producing
        # embeddings from one graph's tensors read through the other graph's node
        # identifiers. The pipeline refuses that composition, and because this
        # function publishes nothing, a refused one cannot leave a graph visible
        # as though it had been accepted.
        # The hop bound comes from the shortest-path table's sidecar alone: the
        # configured bound for a table without one (`SHEPHERD_SP_HOP_BOUND`) was
        # retired with contract M2.5, which refuses such a table before any bound
        # is read.
        pipeline = create_diagnosis_pipeline(
            graph=graph,
            checkpoint_path=checkpoint_path,
            data_dir=data_dir,
            device=device,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
        )
        # Read the configuration back **before returning**, not after publishing.
        # It is the last thing that can fail, and a caller that published first
        # would have to undo the publication to report the failure honestly.
        config = pipeline.get_pipeline_config()
        return PipelineBundle(
            kg=kg,
            pipeline=pipeline,
            config=config,
            data_dir=data_dir,
            checkpoint_path=checkpoint_path,
        )

    except Exception as e:
        logger.error(f"Failed to build pipeline: {e}")
        raise


def publish_pipeline(bundle: PipelineBundle) -> None:
    """Make a built pipeline the one this service serves.

    **The only writer of the served-pipeline fields, and it writes all of them.**
    They answer one question together — what is being served, and out of which
    workspace — so a caller that set some of them would leave the service
    describing a pipeline that is not the one loaded. Everything that can fail
    happened in `build_pipeline`; these assignments cannot.

    **And it is the one path that restores readiness after startup.** Startup
    sets `is_ready` once; a startup whose build failed clears it, as shutdown
    does. The way back is a reload that succeeds, which reaches the service only
    through here, so `is_ready` is set with the fields it vouches for. Restoring
    it anywhere else would let the flag claim a pipeline before a candidate is
    built. A refused reload never gets here and leaves the flag as it was; what
    `/ready` answers also depends on `AppState.requested_pipeline_missing`.
    """
    # **Formatted before the first assignment.** These are values out of an
    # arbitrary configuration dictionary, and formatting one calls its `__str__`;
    # a type whose `__str__` raises would leave the pipeline published and this
    # function raising, which is the failure the caller's ordering exists to
    # prevent. Built here, the same failure happens before anything changed.
    announcement = (
        f"Pipeline published: scoring_mode={bundle.config.get('scoring_mode')}, "
        f"gnn_ready={bundle.config.get('gnn_ready')}, "
        f"sp_ready={bundle.config.get('sp_ready')}"
    )

    app_state.kg = bundle.kg
    app_state.pipeline = bundle.pipeline
    app_state._current_data_dir = bundle.data_dir
    app_state._current_checkpoint_path = bundle.checkpoint_path
    app_state.is_ready = True
    logger.info(announcement)


def initialize_pipeline(
    kg_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    device: Optional[str] = None,
):
    """
    Initialize the diagnosis pipeline if one is not already loaded.

    Called at startup, or lazily on first diagnosis request. Reloading a running
    service does not go through here: it must build its candidate first and keep
    serving the current pipeline until that succeeds, so it calls
    `build_pipeline` and `publish_pipeline` itself.

    Args:
        kg_path: Path to KG JSON file (overrides env var)
        checkpoint_path: Path to trained model checkpoint (overrides env var)
        data_dir: Path to processed data directory (overrides env var)
        device: Inference device (overrides env var)
    """
    if app_state.pipeline is not None:
        return

    bundle = build_pipeline(
        kg_path=kg_path,
        checkpoint_path=checkpoint_path,
        data_dir=data_dir,
        device=device,
    )
    if bundle is None:
        return
    publish_pipeline(bundle)


def initialize_ontology():
    """Initialize HPO ontology"""
    if app_state.ontology is not None:
        return

    logger.info("Initializing ontology...")
    try:
        from src.ontology.loader import OntologyLoader

        # TODO: Load actual HPO ontology
        # loader = OntologyLoader()
        # app_state.ontology = loader.load_hpo()
        logger.info("Ontology initialization placeholder - awaiting data")
    except Exception as e:
        logger.error(f"Failed to initialize ontology: {e}")
        raise


# =============================================================================
# CLI Entry Point
# =============================================================================
def main(host: Optional[str] = None, port: int = 8000):
    """Run the API server (``python -m src.api.main``), on ``127.0.0.1`` unless
    `host` names another address.

    **The bind guard** (SPEC_4 §3.1): a bind other than loopback is refused
    (`network_bind`) unless ``SHEPHERD_ACCEPT_UNAUTHENTICATED_NETWORK_BIND=1``,
    which accepts the risk and is no control. The host goes to application
    startup as ``SHEPHERD_BIND_HOST``, which checks it again (`lifespan`).
    Proxy headers are not trusted: the client address uvicorn reports is the
    socket's, though D8 never reads it anyway.

    **No auto-reload**: uvicorn's reloader restarts the server by SIGTERM on
    every saved source file, and shutdown stops the GPU job the server runs
    (`lifespan`), so a code save would stop a training.
    """
    import uvicorn

    from src.config.server_settings import DEFAULT_BIND_HOST, ENV_BIND_HOST, network_bind

    decision = network_bind(DEFAULT_BIND_HOST if host is None else host)
    if not decision.allowed:
        raise SystemExit(f"src.api.main: {decision.message}")
    if decision.message:
        logger.warning(decision.message)
    os.environ[ENV_BIND_HOST] = decision.host
    uvicorn.run(
        "src.api.main:app",
        host=decision.host,
        port=port,
        log_level="info",
        access_log=False,
        proxy_headers=False,
    )


if __name__ == "__main__":
    main()
