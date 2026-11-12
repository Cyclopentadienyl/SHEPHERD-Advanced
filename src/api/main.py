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
    GET  /api/v1/pipeline/status, POST /api/v1/pipeline/reload,
    POST /api/v1/pipeline/stop
                              - What is served; load a model; stop inference
                                (src/api/routes/pipeline.py, the lifecycle in
                                `AppState`)
    GET  /api/v1/hpo/search   - Search HPO terms
    GET  /api/v1/disease/{id} - Get disease information
    GET  /api/v1/evaluations/loaded-model
                              - The loaded checkpoint's recorded results
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
import gc
import logging
import os
import signal
import sys
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Set, Tuple

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
#: What a status says of serving (`AppState.snapshot`): no model loaded, a
#: model served, or a stop letting the accepted requests finish.
NOT_LOADED, SERVING, STOPPING = "not_loaded", "serving", "stopping"
#: The lifecycle's other transition: a load (start-up's or a reload's).
LOADING = "loading"
#: Why nothing is served, as the one record of it says (`NotServingRecord`).
STOPPED, LOAD_REFUSED, STARTUP_REFUSED = "stopped", "load_refused", "startup_refused"

#: How every not-serving reason tells an operator to load a model: the
#: Diagnosis page's entry as the page shows it with no model loaded ("Load a
#: model…") and the reload route.
LOAD_HINT = ("An operator loads one with “Load a model…” on the Diagnosis page or POST "
             "/api/v1/pipeline/reload")
#: What a refused build says on the open surfaces: its own message can name
#: server paths, so it goes to the log and the operator's reload response only.
BUILD_REFUSED = ("it could not be initialized (the service logs name the condition that "
                 "stopped it)")
#: A refused reload's words on the open surfaces, for the same reason.
RELOAD_REFUSED = "the operator's reload response and the service log say why"
#: Who stopped inference, in a stopped record's words.
STOPPED_BY = {"operator": "by an operator", "restart": "for Restart Backend",
              "shutdown": "as the service shut down"}


class LifecycleBusy(RuntimeError):
    """A load or a stop is already running in this service: a second is
    refused at once, never queued (`AppState.loading`, `stop_serving`)."""


class ServingRefused(RuntimeError):
    """A CUDA load refused before any candidate is read: the GPU lock is busy,
    not configured or unusable, or the job runner's slot is taken. `reason`
    names no path, so it is also what the open not-serving reason says."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class NotServing(RuntimeError):
    """No request is accepted now: nothing is loaded, or a stop is letting the
    accepted ones finish. `reason` is `AppState.not_serving_reason`'s."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class NotServingRecord:
    """**Why nothing is served: one record** (it replaces the
    `real_pipeline_requested` flag). `kind` is `STOPPED`, `LOAD_REFUSED` or
    `STARTUP_REFUSED`; `reason` names no path (a stop's: who stopped it, as
    `STOPPED_BY` says); `model_sha12` is the model it was about, when known.
    Written by the lifecycle only while nothing is served, and cleared by the
    next publication."""

    kind: str
    at: str
    reason: str
    model_sha12: Optional[str] = None


class StopOutcome(NamedTuple):
    """What a stop did (`AppState.stop_serving`): `released` only once the
    model, its graph and the GPU lock are given back."""

    success: bool
    released: bool
    message: str


def _on_cuda(device: Any) -> bool:
    """Whether a device name is a CUDA device, the one rule that decides the
    GPU lock: the resolved device before a build, the device in effect at
    publication."""
    return isinstance(device, str) and device.startswith("cuda")


def _served_part(config: Any, *keys: str) -> Any:
    """A value of a built pipeline's served identity, None when the
    configuration does not hold it (a stand-in's)."""
    value = config.get("served_identity") if isinstance(config, dict) else None
    for key in keys:
        value = value.get(key) if isinstance(value, dict) else None
    return value


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class AppState:
    """Application state container, and **the inference lifecycle** (I1-I10
    of the ops round, `docs/working/ops-wiring-and-ui-ux/README.md` §4): one
    state for what is served, how it is admitted and what it holds.

    - **One load or stop at a time** (`_lifecycle`, taken non-blocking): a
      second is refused (`LifecycleBusy`), never queued, so the event loop,
      where a reload and a diagnosis run, never waits for one.
    - **Admission** (`_admission`): a request takes a lease on the pipeline
      served when it is accepted (`lease`) and gives it back when it ends; a
      stop refuses new ones at once and waits for the accepted ones, with no
      timeout and nothing killed.
    - **The holding**: CUDA serving holds the deployment's one GPU lock
      (`gpu_lock`, a `GpuLock` taken through `take_gpu_lock`), decided by the
      resolved device, from before a candidate is read until nothing uses it:
      the model served on CUDA, a replaced CUDA model whose accepted requests
      still run, or a load holding it for its CUDA candidate. CPU serving
      takes no lock and checks no job slot.
    - **The transition** (`transition`: `LOADING`, `STOPPING` or None) and
      **one not-serving record** (`not_serving`).

    **Lock order, never the other way round:** the lifecycle, then the GPU
    lock's flock, then the job runner's lock (`TrainingManager.
    serving_refusal`), then admission. Admission is held only for a few reads
    and assignments and the lock's release, and nothing waits under it but a
    stop's `wait`, which gives it up; the lifecycle is never taken under
    admission, and the flock is taken under neither the runner's lock nor
    admission. It is given back under admission, in the step that takes it
    out of `gpu_lock` (a release waits for no one), so what the status says
    of the lock is what the flock is.
    """

    def __init__(self):
        self.pipeline = None
        self.kg = None
        self.ontology = None
        #: Startup's half of what `/ready` answers; the other half is
        #: `not_serving_reason`. Set when startup completes, cleared
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
        self._current_checkpoint_path = None
        #: One load or stop at a time, taken non-blocking.
        self._lifecycle = threading.Lock()
        #: Admission: the served fields are swapped and the leases counted
        #: under it.
        self._admission = threading.Condition(threading.Lock())
        #: Accepted requests still running, by the pipeline each was accepted
        #: on (`id`; a leased pipeline is alive, so its id is its own).
        self._leases: Dict[int, int] = {}
        #: Replaced CUDA pipelines whose accepted requests still run: each
        #: keeps the GPU lock until its last lease ends.
        self._retired_on_gpu: Set[int] = set()
        #: A load holds the GPU lock for its CUDA candidate.
        self._load_holds_gpu = False
        #: The deployment's GPU lock as this service holds it, or None.
        self.gpu_lock = None
        #: The served model's device in effect and checkpoint digest, as its
        #: served identity names them (None when nothing is served).
        self.serving_device: Optional[str] = None
        self.served_sha256: Optional[str] = None
        #: `LOADING` or `STOPPING` while one runs, else None.
        self.transition: Optional[str] = None
        #: Why nothing is served (`NotServingRecord`), None when nothing has
        #: been stopped or refused since the last publication.
        self.not_serving: Optional[NotServingRecord] = None

    # ----------------------------------------------------------- what is said
    @property
    def accepting(self) -> bool:
        """Whether a request is admitted now: a pipeline is served and no stop
        is letting the accepted requests finish."""
        return self.pipeline is not None and self.transition != STOPPING

    @property
    def not_serving_reason(self) -> Optional[str]:
        """Why no diagnosis can be answered, or None while one is admitted.

        **One definition, every surface** (K7, DEMO1): `/diagnose` refuses
        with 503 and this reason, `/ready` answers 503 from it and the status
        reports it, so the service is never called ready while every
        diagnosis is refused. It covers a stop letting the accepted requests
        finish, a stopped model, a load or a start-up refused (naming the GPU
        lock's holder when that refused it) and nothing loaded. Every answer
        comes from a model: no illustrative candidates are invented.
        `/health` stays 200, so the launcher starts the service and the
        dashboard's Load entry can load a model. Every reason names that entry
        as the Diagnosis page shows it in this state ("Load a model…"), and
        that loading is an operator control. **No reason names a path**: the
        routes that carry it are open.
        """
        with self._admission:
            return self._reason_locked()

    def _reason_locked(self) -> Optional[str]:
        if self.transition == STOPPING:
            return ("Inference is stopping: the diagnoses already accepted finish, and new ones "
                    f"are refused; then no model is loaded. {LOAD_HINT} once the stop has "
                    "ended.")
        if self.pipeline is not None:
            return None
        if self.transition == LOADING:
            return ("No model is loaded yet: a load is in progress, and no diagnosis can be "
                    f"produced until it publishes one. {LOAD_HINT} if it is refused.")
        record = self.not_serving
        if record is None:
            return ("No model is loaded, so no diagnosis can be produced. An operator loads one "
                    "with “Load a model…” on the Diagnosis page or POST /api/v1/pipeline/reload, "
                    "or starts the service with SHEPHERD_KG_PATH, SHEPHERD_DATA_DIR and "
                    "SHEPHERD_CHECKPOINT_PATH naming a workspace and a model trained on it.")
        model = f"model {record.model_sha12}" if record.model_sha12 else "the model"
        if record.kind == STOPPED:
            return (f"Inference was stopped {record.reason} at {record.at}: {model} is no "
                    f"longer loaded, so no diagnosis can be produced. {LOAD_HINT}.")
        if record.kind == STARTUP_REFUSED:
            return ("A pipeline is configured for this deployment's start-up but none is being "
                    f"served: {record.reason.rstrip('.')}. {LOAD_HINT}, or restarts the "
                    "service to try again.")
        of = f" of model {record.model_sha12}" if record.model_sha12 else ""
        return (f"No model is loaded: the last load{of} was refused at {record.at}: "
                f"{record.reason.rstrip('.')}. {LOAD_HINT}.")

    def snapshot(self) -> Dict[str, Any]:
        """What is served and the lifecycle around it, read in one step under
        admission, so the pipeline and the paths it was built from are one
        publication's: `pipeline`, `data_dir`, `checkpoint_path`, `serving`
        (`NOT_LOADED`, `SERVING` or `STOPPING`), `loading`, `in_flight` (the
        accepted requests still running), `holds_gpu_lock` and
        `not_serving_reason`. No lock path: the status route is open."""
        with self._admission:
            return {
                "pipeline": self.pipeline,
                "data_dir": self._current_data_dir,
                "checkpoint_path": self._current_checkpoint_path,
                "serving": (STOPPING if self.transition == STOPPING
                            else SERVING if self.pipeline is not None else NOT_LOADED),
                "loading": self.transition == LOADING,
                "in_flight": sum(self._leases.values()),
                "holds_gpu_lock": self.gpu_lock is not None and self.gpu_lock.held,
                "not_serving_reason": self._reason_locked(),
            }

    def lifecycle_busy(self) -> bool:
        """Whether a load or a stop is running now (Restart Backend refuses
        then)."""
        return self._lifecycle.locked()

    def _busy_words(self) -> str:
        with self._admission:
            doing = {LOADING: "A model is being loaded",
                     STOPPING: "Inference is stopping"}.get(self.transition,
                                                            "A load or a stop is running")
        return (f"{doing} in this service; one load or stop runs at a time, so this one was "
                "refused, never queued. Nothing was changed.")

    # -------------------------------------------------------------- admission
    @contextmanager
    def lease(self) -> Iterator[Any]:
        """The served pipeline for one request, admitted now or refused with
        `NotServing` and the reason. Held until the request ends; a stop waits
        for it, and a replaced CUDA pipeline keeps the GPU lock until its last
        lease ends.

        **No reference to the pipeline outlives its lease**: a stop frees the
        device memory once the last lease has ended, so the lease drops its
        own before it wakes the stop, and a caller drops the one it bound
        inside the `with` (`/diagnose`). A request that ends while a stop
        runs leaves the GPU lock to that stop, which frees the memory and
        gives the lock back before it reports released."""
        with self._admission:
            pipeline = self.pipeline
            if not self.accepting:
                raise NotServing(self._reason_locked())
            key = id(pipeline)
            self._leases[key] = self._leases.get(key, 0) + 1
        try:
            yield pipeline
        finally:
            del pipeline
            with self._admission:
                left = self._leases.pop(key) - 1
                if left:
                    self._leases[key] = left
                else:
                    self._retired_on_gpu.discard(key)
                stopping = self.transition == STOPPING
                self._admission.notify_all()
            if not stopping:
                self._release_gpu_if_unused()

    # ------------------------------------------------------------------- load
    @contextmanager
    def loading(self) -> Iterator["ServingLoad"]:
        """One load (start-up's or a reload's) in the lifecycle, or
        `LifecycleBusy`. At its end, a GPU lock it took and no model uses is
        given back (a refused CUDA candidate, a CPU load replacing a CUDA
        model whose requests have ended). The lifecycle is released last,
        whatever happened before it, so no failure leaves every later load
        and stop refused as busy."""
        if not self._lifecycle.acquire(blocking=False):
            raise LifecycleBusy(self._busy_words())
        try:
            with self._admission:
                self.transition = LOADING
            yield ServingLoad(self)
        finally:
            try:
                with self._admission:
                    self._load_holds_gpu = False
                    self.transition = None
                self._release_gpu_if_unused()
            finally:
                self._lifecycle.release()

    def _hold_gpu_for(self, device: Optional[str]) -> None:
        """Inside a load, right after its device is resolved and before any
        candidate is read: **the GPU lock held for a CUDA device**, the one
        this service already holds (a CUDA model served, or a replaced one
        still running: a second taker in this process would refuse itself),
        else taken now (`take_gpu_lock`); then the job runner's slot read
        under its lock (`serving_refusal`), which closes the window between
        the runner spawning a job and that job's process taking the lock.
        Nothing for a CPU device. Raises `ServingRefused`, the lock given
        back unless a served model uses it."""
        if not _on_cuda(device):
            return
        from src.api.services.training_manager import training_manager
        from src.config.server_settings import ENV_WORKSPACES_ROOT
        from src.utils.gpu_lock import (
            SERVING_COMMAND,
            SERVING_JOB_TYPE,
            GpuLockBusy,
            GpuLockMisconfigured,
            GpuLockNotConfigured,
            take_gpu_lock,
        )

        with self._admission:
            held = self.gpu_lock is not None and self.gpu_lock.held
            if held:
                self._load_holds_gpu = True
        if not held:
            try:
                lock = take_gpu_lock(SERVING_JOB_TYPE, SERVING_COMMAND, log=logger)
            except GpuLockNotConfigured as exc:
                raise ServingRefused(f"CUDA inference is refused: {exc}") from exc
            except GpuLockMisconfigured as exc:
                logger.error("CUDA inference is refused: %s", exc)
                raise ServingRefused(
                    f"CUDA inference is refused: the GPU lock cannot be used ({ENV_WORKSPACES_ROOT} "
                    "is set but cannot be used; the service log says why), and nothing runs "
                    "on the GPU without excluding the others.") from exc
            except GpuLockBusy as exc:
                raise ServingRefused(
                    f"CUDA inference is refused: the GPU is in use by {exc.holder}. Load "
                    "again once it has ended.") from exc
            except OSError as exc:
                logger.error("CUDA inference is refused: %s", exc)
                raise ServingRefused("CUDA inference is refused: the GPU lock cannot be taken "
                                     "(the service log says why).") from exc
            with self._admission:
                self.gpu_lock = lock
                self._load_holds_gpu = True
        refusal = training_manager.serving_refusal()
        if refusal is not None:
            with self._admission:
                self._load_holds_gpu = False
            self._release_gpu_if_unused()
            raise ServingRefused(refusal)

    def record_refusal(self, kind: str, reason: str, model_sha12: Optional[str] = None) -> None:
        """A load or start-up refused while nothing is served, recorded as why
        nothing is (`LOAD_REFUSED`, `STARTUP_REFUSED`); a refusal while a
        model serves changes nothing."""
        with self._admission:
            if self.pipeline is None:
                self.not_serving = NotServingRecord(kind, _now(), reason, model_sha12)

    def _publish(self, bundle: "PipelineBundle", device: Any, sha256: Any) -> None:
        """`publish_pipeline`'s swap, under admission. **A model on CUDA is
        published only while this service holds the GPU lock**, refused
        before anything changes otherwise. The replaced pipeline: a CUDA one
        whose accepted requests still run keeps the lock until they end;
        CUDA to CUDA keeps the one `GpuLock`; CUDA to CPU gives it back once
        nothing uses it, which is now when no request of the old one runs."""
        with self._admission:
            if _on_cuda(device) and not (self.gpu_lock is not None and self.gpu_lock.held):
                raise RuntimeError(
                    "A model on CUDA is served only while this service holds the GPU lock; "
                    "load it through the lifecycle (start-up or a reload). Nothing was "
                    "published.")
            old, old_device = self.pipeline, self.serving_device
            if (old is not None and old is not bundle.pipeline and _on_cuda(old_device)
                    and self._leases.get(id(old))):
                self._retired_on_gpu.add(id(old))
            self.kg = bundle.kg
            self.pipeline = bundle.pipeline
            self._current_data_dir = bundle.data_dir
            self._current_checkpoint_path = bundle.checkpoint_path
            self.serving_device = device if isinstance(device, str) else None
            self.served_sha256 = sha256 if isinstance(sha256, str) else None
            self.is_ready = True
            self.not_serving = None
        del old
        self._release_gpu_if_unused()

    # ------------------------------------------------------------------- stop
    def stop_serving(self, *, by: str = "operator",
                     expected_checkpoint_sha256: Optional[str] = None) -> StopOutcome:
        """**The one stop** (I4): new requests refused at once, with the
        reason; the accepted ones waited for, with no timeout and nothing
        killed; then the pipeline, its graph, the current paths and the
        diagnosis reference-sample cache cleared, `gc`, and while the GPU lock
        is held a synchronize and `empty_cache` (no claim that memory is
        returned: the measured reading is the resource route's); then the GPU
        lock and its description given back, by this stop alone (a request
        ending meanwhile leaves them to it); only then is "released"
        reported. Nothing
        loaded is "nothing to stop". With `expected_checkpoint_sha256`, a
        stop of another model is refused and it keeps serving. Refused with
        `LifecycleBusy` while a load or another stop runs. `by` is who stops
        it (`STOPPED_BY`). Blocks while it waits: call it from a thread."""
        if not self._lifecycle.acquire(blocking=False):
            raise LifecycleBusy(self._busy_words())
        try:
            with self._admission:
                if self.pipeline is None:
                    return StopOutcome(False, False, "Nothing to stop: no model is loaded.")
                sha = self.served_sha256
                shown = expected_checkpoint_sha256
                if shown is not None and sha != shown:
                    serving = (f"the model in use is {sha[:12]}" if sha else
                               "the model in use records no checkpoint fingerprint")
                    return StopOutcome(False, False, (
                        f"Not stopped: {serving}, not {shown[:12]} that was shown; it keeps "
                        "serving. Read the status again to see what is served."))
                self.transition = STOPPING
                accepted = sum(self._leases.values())
                while self._leases:
                    self._admission.wait()
                device = self.serving_device
                held = self.gpu_lock is not None
                self.pipeline = None
                self.kg = None
                self._current_data_dir = None
                self._current_checkpoint_path = None
                self.serving_device = None
                self.served_sha256 = None
            from src.api.routes.diagnose import clear_reference_samples

            clear_reference_samples()
            # The CUDA cache whenever the GPU lock is held: a CUDA model, or a
            # replaced CUDA model whose last request the drain just waited for
            # (the service switched to the CPU while it ran).
            _free_device_memory("cuda" if held else device)
            self._release_gpu_if_unused(free=False)
            with self._admission:
                gave_back = held and self.gpu_lock is None
                self.not_serving = NotServingRecord(STOPPED, _now(), STOPPED_BY.get(by, by),
                                                    sha[:12] if sha else None)
                self.transition = None
            model = f"model {sha[:12]}" if sha else "the model"
            return StopOutcome(True, True, (
                f"Inference stopped: {accepted} accepted request(s) finished first, then "
                f"{model} and its graph were released"
                f"{' and the GPU lock given back' if gave_back else ''}. New diagnoses are "
                "refused until a model is loaded."))
        finally:
            with self._admission:
                if self.transition == STOPPING:
                    self.transition = None
            self._lifecycle.release()

    # ---------------------------------------------------------------- the GPU
    def _gpu_in_use_locked(self) -> bool:
        return (self._load_holds_gpu or bool(self._retired_on_gpu)
                or (self.pipeline is not None and _on_cuda(self.serving_device)))

    def _release_gpu_if_unused(self, *, free: bool = True) -> None:
        """Give the GPU lock back once nothing uses it: no CUDA model served,
        no replaced CUDA model's request running, no load holding it.

        **The memory first, then the lock in one step.** A `gc` and the CUDA
        cache emptied come first, outside admission (`free`; a stop has done
        both already); then, under admission and only if nothing has started
        using it meanwhile, the flock is released and the lock taken out of
        `gpu_lock` together (a release waits for no one), so the status never
        says no lock while this service still holds it. A release that
        reports an error is logged, never raised: the `GpuLock` has dropped
        its descriptor, so it cannot be given back twice, and every caller (a
        load's end, a lease's, a stop) finishes its own step."""
        with self._admission:
            if self.gpu_lock is None or self._gpu_in_use_locked():
                return
        if free:
            _free_device_memory("cuda")
        with self._admission:
            lock = self.gpu_lock
            if lock is None or self._gpu_in_use_locked():
                return
            self.gpu_lock = None
            try:
                lock.release()
            except OSError as exc:
                logger.error("Giving the GPU lock back reported an error (%s); its descriptor "
                             "is closed, and the lock file's flock is what a job and the GPU "
                             "chip read.", exc)
                return
        logger.info("This service's CUDA inference gave the GPU lock back.")


class ServingLoad:
    """One load in the lifecycle (`AppState.loading`): `hold` is where its
    resolved device decides the GPU lock, and `lock_refusal` keeps why that
    refused it, the words the open not-serving reason then says."""

    def __init__(self, state: AppState) -> None:
        self._state = state
        self.lock_refusal: Optional[str] = None

    def hold(self, device: Optional[str]) -> None:
        try:
            self._state._hold_gpu_for(device)
        except ServingRefused as refused:
            self.lock_refusal = refused.reason
            raise


def _free_device_memory(device: Optional[str]) -> None:
    """`gc`, and on CUDA a synchronize and `empty_cache`: what a stop and a
    lock given back do. `empty_cache` returns the allocator's unused blocks
    and claims nothing about the host's memory (the GB10's is shared); a
    failure is logged, never raised."""
    gc.collect()
    if not _on_cuda(device):
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001 - freeing is best effort
        logger.warning("The CUDA cache could not be emptied: %s", exc)


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

    **Serving** (I6, I7): the job runner's slot opens first, then the model
    the environment configures for start-up, if any, loads through the one
    lifecycle (`initialize_pipeline`); with none configured nothing is loaded
    and no other model is chosen. At the end, after the job's stop, serving
    stops through the one stop, so the model and the GPU lock are given back.

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

    # **The job slot opens before the configured load** (I6): a CUDA load
    # reads the runner's slot after it holds the GPU lock
    # (`TrainingManager.serving_refusal`), and an in-process client's earlier
    # lifespan left it closed.
    _open_the_gpu_slot()
    try:
        # Attempt eager initialization if a pipeline is configured. A graph
        # alone or a checkpoint alone counts: `build_pipeline` refuses either,
        # and startup is where an operator should see that, not the first
        # diagnosis. **Nothing else is loaded**: with no model configured the
        # service starts with none, and no other model is picked (I6).
        if os.environ.get("SHEPHERD_KG_PATH") or os.environ.get(
            "SHEPHERD_CHECKPOINT_PATH"
        ):
            initialize_pipeline()
        else:
            logger.info(
                "No model is configured (SHEPHERD_KG_PATH, SHEPHERD_DATA_DIR, "
                "SHEPHERD_CHECKPOINT_PATH): diagnoses are refused until one is loaded."
            )

        app_state.is_ready = True
        logger.info("API service ready")
    except Exception as e:
        # Not ready until a pipeline is published; a reload that succeeds does
        # that (`publish_pipeline`), so this is not a state only a restart clears.
        # Why is the not-serving record's (`initialize_pipeline`).
        logger.error(f"Startup failed: {e}")
        app_state.is_ready = False

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
            # **Then serving stops** (I7): uvicorn has closed the connections,
            # so the accepted requests have ended, and the model, its graph and
            # the GPU lock are given back. A process killed instead has its
            # lock released by the kernel; no state file pretends a holding.
            await asyncio.to_thread(_stop_serving_at_shutdown)
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
    still ends by SIGQUIT, after the job's stop. Each first records the stop
    with the job runner (`record_stop_signal`, a plain store), so Restart
    Backend's exec is refused from the signal on (round 3 of the review
    fixes, P3-3), then runs the server's handler as before (a second Ctrl-C
    is still uvicorn's forced quit), then
    hands the event loop `begin`: the first time, the job's stop starts in a
    thread of its own (it closes the GPU slot, then stops the job that held
    it: `_stop_the_active_job`), and every time, the dashboard's
    heartbeat streams end (`_end_dashboard_streams`), so an open tab holds up
    neither the job nor the backend. The thread is not a daemon, so the
    interpreter waits for it before it exits, after a forced quit too.
    `end_jobs`, at the lifespan's end, waits for it and does not repeat it;
    it closes the slot and stops the job itself only when no signal began
    the shutdown.

    Installed only in the main thread, where a server sets its handlers (an
    in-process test client runs the lifespan elsewhere), and only when
    SIGTERM has a handler to route to (a server's); **neither fallback is
    silent** (round 3 of the review fixes, F4): in the main thread a SIGTERM
    with no server's handler is a warning, since the job's stop and the
    closed slot then wait for the lifespan's end, and off it a debug line. A
    handler that is not the server's is not wrapped (SIGINT's default
    handler, which no server set, or one of someone else's), and a routed
    signal is taken only where it still has its default action (SIGHUP
    ignored under ``nohup`` is left as it is). `restore` puts back each
    handler replaced here that is still the one installed here (after a
    forced quit uvicorn has already put back its own).
    """

    def __init__(self, application: FastAPI) -> None:
        self._application = application
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        #: signal -> (the handler it had, the one installed here)
        self._installed: Dict[int, Tuple[Any, Callable[[int, Any], None]]] = {}
        self._stopping: Optional[threading.Thread] = None
        #: The job runner's record of a shutdown signal, called first in each
        #: wrapped handler (`TrainingManager.record_stop_signal`).
        self._record_stop_signal: Callable[[], None] = lambda: None

    def install(self) -> None:
        """Wrap the server's handlers (each listed signal whose handler is
        SIGTERM's), and route the other signals that no longer reach a job
        (`JOB_DETACHED_SIGNALS`) to its SIGTERM handler; in the event loop,
        at startup."""
        from src.api.services.training_manager import JOB_DETACHED_SIGNALS, training_manager

        if threading.current_thread() is not threading.main_thread():
            logger.debug("The application runs off the main thread (an in-process client), "
                         "so the shutdown signals are not taken here: the lifespan's end "
                         "stops the job.")
            return
        termination = signal.getsignal(signal.SIGTERM)
        if not callable(termination):
            logger.warning(
                "The backend cannot attach to the server's shutdown signals (SIGTERM has no "
                "handler of the server's: %r), so a stop of the backend begins the job's stop, "
                "and closes the GPU slot, only at the lifespan's end, after every open "
                "connection has closed (an open dashboard tab holds it).", termination)
            return
        self._loop = asyncio.get_running_loop()
        self._record_stop_signal = training_manager.record_stop_signal
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
            # Recorded before anything else, a plain store (round 3 of the
            # review fixes, P3-3): from here on Restart Backend's exec is
            # refused, before this stop has closed the GPU slot.
            self._record_stop_signal()
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
        """At the lifespan's end, **one stop per shutdown** (round 3 of the
        review fixes, P3-4): the stop a signal began is waited for, never
        repeated; only when no signal began the shutdown are the slot closed
        and its job stopped here. After a signal's `close` no job can start,
        so the only job that can still hold the slot is the one that stop
        ended or gave up on (one that SIGKILL cannot end) and said so; a
        second stop would only wait `END_ACTIVE_JOB_SECONDS` more for it.
        Blocks; call it from a thread."""
        if self._stopping is not None:
            self._stopping.join()
            return
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
    mounted, there is nothing to end; a dashboard with no `stop_event` (a
    Gradio that renamed it) is a warning, never a silent no-op."""
    for route in application.routes:
        if isinstance(route, Mount) and route.path == DASHBOARD_PATH:
            stop_event = getattr(route.app, "stop_event", None)
            if stop_event is None:
                logger.warning("The dashboard mounted at %s has no stop_event to end its "
                               "heartbeat streams with, so each open dashboard tab holds the "
                               "backend's shutdown until it is closed.", DASHBOARD_PATH)
                return
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


def _stop_serving_at_shutdown() -> None:
    """At the lifespan's end: the model served stopped through the one stop
    (`AppState.stop_serving`), said in the log. A load or stop still running
    is said and left: the process ends, and the kernel releases its lock."""
    try:
        outcome = app_state.stop_serving(by="shutdown")
    except LifecycleBusy as busy:
        logger.warning("Serving was not stopped at shutdown: %s", busy)
        return
    if outcome.released:
        logger.info(outcome.message)


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

    Returns whether the service is ready to accept requests: a pipeline is
    served and admitted (`AppState.not_serving_reason`, the state `/diagnose`
    refuses) and startup succeeded or a pipeline has since been published
    (`is_ready`). The pipeline is checked first, so a service with none is
    told why and how to recover, as `/diagnose` tells it.
    """
    reason = app_state.not_serving_reason
    if reason is not None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=reason)
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
    checkpoint_within: Optional[Path] = None,
) -> Optional[PipelineBundle]:
    """
    Build a diagnosis pipeline without publishing it.

    Loads the knowledge graph, creates the DiagnosisPipeline with its model, and
    reads back its configuration — everything that can fail. **Application
    state is not touched, and no lock is taken**: the lifecycle that calls it
    (`initialize_pipeline`, `run_reload`) holds the GPU lock for a CUDA device
    and records a refusal. Returns the finished bundle, or ``None`` when there
    is nothing to build because no KG path is configured or the configured one
    does not exist; raises when a configured pipeline fails to build, a graph
    named with no checkpoint included (a graph is never served without a
    model).

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
        checkpoint_within: The resolved workspace an automatically selected
            checkpoint must lie inside (the reload's selection). The build's one
            checkpoint read refuses a file its open finds outside it, before any
            byte is read, and raises `ReadOutside` (`read_once(within=)`). None
            for an explicit path, at startup and everywhere else: no bound.

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
            # Returning None here would leave a deployment that named a
            # checkpoint described as one that configured nothing.
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

    try:
        from src.inference.pipeline import (
            build_served_pipeline,
            require_model_source,
            resolve_serving_device,
        )

        # **A model, or no pipeline** (K7, PATH1): a graph named with no
        # checkpoint is refused, never served by path reasoning. **The device is
        # settled before the graph is read** (contract M3c). Both before the
        # file check below; the shared build repeats both rules for its other
        # callers.
        require_model_source(checkpoint_path, data_dir)
        resolve_serving_device(device)

        kg_file = Path(kg_path)
        if not kg_file.exists():
            logger.error(f"KG file not found: {kg_file}")
            return None

        # **The build itself is the shared one** (`build_served_pipeline`): this
        # function resolves the paths and records the request.
        pipeline = build_served_pipeline(
            kg_file,
            checkpoint_path=checkpoint_path,
            data_dir=data_dir,
            device=device,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            checkpoint_within=checkpoint_within,
        )
        kg = pipeline.kg
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

    **The only writer of the served-pipeline fields, and it writes all of them**,
    swapped in one step under admission (`AppState._publish`): what is served,
    out of which workspace and checkpoint, on which device, with which digest.
    They answer one question together, so a caller that set some of them would
    leave the service describing a pipeline that is not the one loaded.
    Everything that can fail happened in `build_pipeline`; the one refusal here
    comes before anything changes: **a model on CUDA is published only while
    this service holds the GPU lock**, which its load took
    (`ServingLoad.hold`). The replaced pipeline's holding follows the switch
    rules (CUDA to CUDA keeps the one `GpuLock`; CUDA to CPU gives it back once
    the replaced model's accepted requests have ended).

    **And it is the one path that restores readiness after startup.** Startup
    sets `is_ready` once; a startup whose build failed clears it, as shutdown
    does. The way back is a reload that succeeds, which reaches the service only
    through here, so `is_ready` is set with the fields it vouches for. Restoring
    it anywhere else would let the flag claim a pipeline before a candidate is
    built. A refused reload never gets here and leaves the flag as it was; what
    `/ready` answers also depends on `AppState.not_serving_reason`. The
    not-serving record is cleared with the swap.
    """
    # **Formatted before the first assignment.** These are values out of an
    # arbitrary configuration dictionary, and formatting one calls its `__str__`;
    # a type whose `__str__` raises would leave the pipeline published and this
    # function raising, which is the failure the caller's ordering exists to
    # prevent. Built here, the same failure happens before anything changed.
    announcement = (
        f"Pipeline published: scoring_mode={bundle.config.get('scoring_mode')}, "
        f"gnn_ready={bundle.config.get('gnn_ready')}"
    )
    app_state._publish(bundle, _served_part(bundle.config, "settings", "device"),
                       _served_part(bundle.config, "checkpoint", "sha256"))
    logger.info(announcement)


def initialize_pipeline(
    kg_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> None:
    """
    **Start-up's configured load** (I6): the lifespan's, when the environment
    names a model; nothing else calls it, and a diagnosis never loads one (I9).

    It goes through the one lifecycle, as a reload does (`AppState.loading`):
    the device resolved by the shared resolver, the GPU lock held for a CUDA
    device before anything is read (busy, not configured or the runner's slot
    taken refuses it), the shared build, then publication. A refusal is
    recorded as why nothing is served (`STARTUP_REFUSED`, naming the lock's
    holder when that refused it, else pointing at the log, since a build's own
    words can name server paths) and raised, so the lifespan reports the
    service not ready; no other model is chosen. A graph file that is not
    there (`build_pipeline` returns None) is recorded the same way.

    Args:
        kg_path: Path to KG JSON file (overrides env var)
        checkpoint_path: Path to trained model checkpoint (overrides env var)
        data_dir: Path to processed data directory (overrides env var)
        device: Inference device (overrides env var)
    """
    from src.inference.pipeline import resolve_serving_device

    with app_state.loading() as load:
        try:
            load.hold(resolve_serving_device(requested_serving_device(device)))
            bundle = build_pipeline(
                kg_path=kg_path,
                checkpoint_path=checkpoint_path,
                data_dir=data_dir,
                device=device,
            )
            if bundle is None:
                app_state.record_refusal(STARTUP_REFUSED, BUILD_REFUSED)
                return
            publish_pipeline(bundle)
        except Exception:
            app_state.record_refusal(STARTUP_REFUSED, load.lock_refusal or BUILD_REFUSED)
            raise


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
