"""
SHEPHERD-Advanced Training Manager Service — the job runner
===========================================================
Runs the GPU jobs as subprocesses, one at a time, and reads what they leave;
also reads training metrics and reports system resources.

Module: src/api/services/training_manager.py
Absolute Path: /home/user/SHEPHERD-Advanced/src/api/services/training_manager.py

Purpose:
    Provide an API-layer service that:
    - Runs each job type's own script as a subprocess (M5): **training**,
      scripts/train_model.py with its config in a temp YAML, and **test runs**,
      scripts/measure_scorer.py on one registered cohort version
    - Holds **one GPU slot**: a job that is running or stopping keeps it,
      whatever its type, and the deployment's GPU lock
      (src/utils/gpu_lock.py) is checked before a start, so a job a CLI
      started elsewhere is refused early, naming its holder
    - Tracks each job's own process, status, exit code and output tail; the
      latest job of each type keeps its state after it ends
    - Stops a job by type: SIGINT (CTRL_BREAK_EVENT on Windows), then SIGTERM,
      then SIGKILL; a training stop never reaches a test run, nor the reverse
    - Reads training metrics from MetricsLogger's JSON log files, and, once a
      test run has ended, which report it left in the report store
    - Lists checkpoint files with metadata
    - Reports GPU/RAM resource stats via nvidia-smi subprocess

Architecture Note:
    This module lives in src.api.services, NOT in src.training.
    The webui layer imports this service (webui -> api is allowed).
    Training and measurement are invoked via subprocess to respect
    import-linter rules and avoid blocking the API event loop. The runner is
    `training_manager` reshaped in place (PLAN_TEST_RESULTS §4.0, §6): its
    training entry points (`start_training`, `stop_training`, `get_status`)
    keep their names and shapes, so the routes and the Training Console call
    it as before.

    **Each job's child takes the GPU lock itself** (the CLIs do, at the start
    of their `main`), so the lock follows the real worker and is released by
    the kernel however that process ends; the runner only checks it before
    starting, to refuse early.

    **Nothing here gates access.** Every caller that starts or stops a job
    applies D8 first (WP2 wires the routes and handlers).

Dependencies:
    - subprocess: Launch job processes
    - signal: Send stop signals
    - json: Read metrics logs
    - yaml: Write config files
    - pathlib: File path handling
    - psutil (optional): RAM monitoring

Called by:
    - src/api/routes/training.py (API endpoints)
    - src/api/services/backend_control.py (the restart lock: any job active)
    - src/webui/components/training_console.py (direct call via AppState)

Version: 2.0.0
"""
from __future__ import annotations

import json
import logging
import os
import re
import secrets
import signal
import subprocess
import sys
import tempfile
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import yaml

from src.utils.gpu_lock import lock_state

logger = logging.getLogger(__name__)

# Project root (two levels up from src/api/services/)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

#: The statuses that hold the one GPU slot, for every job type: a job being
#: stopped still runs until its process has ended.
BUSY_STATUSES = ("running", "stopping")

#: Lines of a job's own output (stdout and stderr, merged) kept in memory, per
#: job; the latest job of each type keeps its tail after it ends.
LOG_TAIL_LINES = 1000
_MAX_LINE_CHARS = 4000

#: A stop's escalation: after SIGINT (CTRL_BREAK_EVENT on Windows), after
#: SIGTERM, after SIGKILL.
STOP_GRACE_SECONDS = 30.0
TERMINATE_GRACE_SECONDS = 10.0
KILL_GRACE_SECONDS = 10.0
#: How long the monitor waits for a job's output after its process has ended: a
#: process the job forked may keep the pipe open.
OUTPUT_GRACE_SECONDS = 5.0
#: How long a stop waits, once the process has ended, for its monitor to record
#: how (and to read what a test run left) before answering.
RESULT_WAIT_SECONDS = 60.0

_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass
class TrainingStatus:
    """Snapshot of training state."""

    # idle, running, stopping, stopped (by the operator), completed (the
    # process exited cleanly; how the stage stood is the run's `outcome`),
    # failed.
    status: str = "idle"
    pid: Optional[int] = None
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    start_time: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    latest_metrics: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None


# =============================================================================
# Job types
# =============================================================================
@dataclass(frozen=True)
class JobType:
    """What differs between the runner's job types.

    Statuses, for both: idle (none started yet), running, stopping, then
    completed (exit 0), failed, or `stopped_status` after an operator's stop.
    """

    name: str
    #: How a message names a job of this type, as a sentence's subject.
    noun: str
    #: The refusal of a stop when no job of this type is running.
    none_running: str
    #: Its script, relative to the project root: each job type names its own.
    script: str
    #: The status an operator's stop ends in: training's "stopped" (contract
    #: M3a; the run can be continued from its last snapshot) and a test run's
    #: "cancelled", distinct from failed and completed.
    stopped_status: str


JOB_TYPES: Dict[str, JobType] = {
    "training": JobType("training", "Training", "No training is running",
                        "scripts/train_model.py", "stopped"),
    "test": JobType("test", "A test run", "No test run is running",
                    "scripts/measure_scorer.py", "cancelled"),
}

#: What measure_scorer's exit statuses say, as its module docstring defines
#: them. Descriptive: whether a report was kept and registered is read from
#: the disk, never from these.
_TEST_EXIT_MEANING = {
    0: "measured (with --record: registered, or already held)",
    1: "refused before anything was written, or measured but the report was not written, "
       "or the run failed after writing it; its output says which",
    2: "the command line was rejected",
    3: "measured and kept, but not registered",
}


#: Reads, for one report file, whether it was kept and whether it is
#: registered, from the disk: `reader(report_path, checkpoint_path)` returns
#: `{"kept": bool | None, "registered": bool | None, "reason": str}`, None
#: meaning "cannot be checked". **WP4 integration point:** the one public
#: helper in src/evaluation/registration.py is passed to the runner here.
ReportStateReader = Callable[[Path, Path], Mapping[str, Any]]


def _no_report_state_reader(report: Path, checkpoint: Path) -> Dict[str, Any]:
    return {"kept": None, "registered": None,
            "reason": "this build has no report-state reader, so whether the report was "
                      "kept and registered is not checked here"}


@dataclass(frozen=True)
class TestRunInputs:
    """A test run's inputs, every one taken from the server's lists and
    settings, never from a request (D8: server-side lists only).

    The caller resolves a request's names through those lists (WP1): the
    registry from its setting, the version by label with its `cohort.json`
    digest, the workspace by name under the workspace root, the checkpoint by
    name and digest from that workspace's listing, the evidence directories
    from their setting. `record` and `deterministic` are the operator's two
    choices. Nothing else is chosen: the batch size is the CLI's default and
    the device the deployment's (D5).
    """

    __test__ = False  # a dataclass, not a pytest test class, whatever its name says

    registry: Path
    version_label: str
    version_sha256: str
    workspace: Path
    checkpoint: Path
    checkpoint_sha256: str
    evidence_dirs: Tuple[Path, ...] = ()
    record: bool = False
    deterministic: bool = False

    def reports_dir(self) -> Path:
        """The version's report store directory, after the inputs' shape is
        checked. The run checks the files themselves when it reads them.

        Raises:
            ValueError: a digest that is not a full SHA-256, a path that is not
                absolute, or a registry or label the registry's rules refuse.
        """
        from src.evaluation import cohort_registry

        for name, digest in (("checkpoint", self.checkpoint_sha256),
                             ("cohort version", self.version_sha256)):
            if not (isinstance(digest, str) and _SHA256.fullmatch(digest)):
                raise ValueError(f"The {name} digest {digest!r} is not a full SHA-256 "
                                 "hexdigest.")
        paths = [("workspace", self.workspace), ("checkpoint", self.checkpoint),
                 *(("evidence directory", d) for d in self.evidence_dirs)]
        for name, path in paths:
            if not Path(path).is_absolute():
                raise ValueError(f"The {name} {path} is not an absolute path, as the "
                                 "server's lists give it.")
        root = cohort_registry.validate_registry(self.registry)
        label = cohort_registry.validate_label(self.version_label)
        return root / cohort_registry.VERSIONS_DIR / label / cohort_registry.REPORTS_DIR

    def summary(self) -> Dict[str, Any]:
        """The inputs as a status shows them."""
        return {
            "registry": str(self.registry),
            "version_label": self.version_label,
            "version_sha256": self.version_sha256,
            "workspace": str(self.workspace),
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha256,
            "evidence_dirs": [str(d) for d in self.evidence_dirs],
            "record": bool(self.record),
            "deterministic": bool(self.deterministic),
        }


def deployment_device() -> Optional[str]:
    """The device a test run is given: the deployment's (D5), `SHEPHERD_DEVICE`
    as the served pipeline reads it (`requested_serving_device`). None leaves
    the CLI's own `auto`, which requires CUDA.

    Raises:
        ValueError: a value `measure_scorer.py --device` does not take
            (`cuda:1`, say), so the deployment's device cannot be passed on.
    """
    from src.api.main import requested_serving_device

    value = requested_serving_device(None)
    if value is None or value == "auto":
        return None
    if value in ("cuda", "cpu"):
        return value
    raise ValueError(
        f"SHEPHERD_DEVICE is {value!r}, and a test run takes auto, cuda or cpu "
        "(scripts/measure_scorer.py --device), so it cannot be given the deployment's device.")


def build_test_argv(inputs: TestRunInputs, device: Optional[str]) -> List[str]:
    """measure_scorer's command line for a test run, from `inputs` only.

    The expected digests make the run refuse before measuring when the
    checkpoint or the version it reads is not the one chosen. `--batch-size`
    is not passed: the CLI's default stays its one definition.
    """
    argv = [
        sys.executable, str(PROJECT_ROOT / JOB_TYPES["test"].script),
        "--checkpoint", str(inputs.checkpoint),
        "--data-dir", str(inputs.workspace),
        "--cohort", inputs.version_label,
        "--registry", str(inputs.registry),
        "--expect-checkpoint-sha256", inputs.checkpoint_sha256,
        "--expect-version-sha256", inputs.version_sha256,
    ]
    for directory in inputs.evidence_dirs:
        argv += ["--training-evidence", str(directory)]
    if device is not None:
        argv += ["--device", device]
    if inputs.record:
        argv.append("--record")
    if inputs.deterministic:
        argv.append("--deterministic")
    return argv


@dataclass(eq=False)
class _Job:
    """One job: its own process, which only its own monitor and stop touch."""

    kind: JobType
    job_id: str
    started: datetime
    process: Optional[subprocess.Popen] = None
    status: str = "running"
    stop_requested: bool = False
    #: The last signal a stop had to send: SIGINT, SIGTERM or SIGKILL.
    stop_signal: Optional[str] = None
    exit_code: Optional[int] = None
    ended: Optional[datetime] = None
    error_message: Optional[str] = None
    #: What the job was given, as its status shows it.
    inputs: Dict[str, Any] = field(default_factory=dict)
    #: What it left (a test run's report, read from the disk once it ended).
    outcome: Optional[Dict[str, Any]] = None
    log: Deque[str] = field(default_factory=lambda: deque(maxlen=LOG_TAIL_LINES))
    log_lock: threading.Lock = field(default_factory=threading.Lock)
    #: Files the job owns and removes when it ends (training's temp YAML).
    cleanup: List[Path] = field(default_factory=list)
    #: The report store listed before launch (test runs).
    reports_dir: Optional[Path] = None
    reports_before: Set[str] = field(default_factory=set)
    done: threading.Event = field(default_factory=threading.Event)

    def tail(self, n: Optional[int] = None) -> List[str]:
        with self.log_lock:
            lines = list(self.log)
        return lines if n is None else lines[-n:]


def _report_names(reports: Path) -> Set[str]:
    """The run directories in a version's report store; none before the first
    report (`create_report_dir` makes `reports/`)."""
    try:
        return set(os.listdir(reports))
    except FileNotFoundError:
        return set()


def _signal_name(returncode: int) -> str:
    try:
        return signal.Signals(-returncode).name
    except ValueError:
        return f"signal {-returncode}"


class TrainingManager:
    """
    The one job runner: training and test runs, one GPU job at a time.

    Thread-safe singleton designed to be shared across API routes
    and Gradio event handlers in the same FastAPI process.
    """

    def __init__(self, report_state: Optional[ReportStateReader] = None):
        self._lock = threading.Lock()
        #: The job holding the one GPU slot, until its monitor has finished.
        self._active: Optional[_Job] = None
        #: The latest job of each type, kept after it ends.
        self._last: Dict[str, _Job] = {}
        self._report_state: ReportStateReader = report_state or _no_report_state_reader
        #: The last training request (as started, with its run directory).
        self._config: Optional[Dict[str, Any]] = None

        #: The directory of the run started last (contract M3a-2). Named here
        #: and claimed by the training process once its inputs pass, so a
        #: refused run leaves nothing behind; its logs are read from inside it.
        self.run_dir: Optional[Path] = None
        self._snapshot_cache: Optional[tuple] = None

    @property
    def log_dir(self) -> Optional[Path]:
        """The logs of the run started last, or `None` before any."""
        from src.utils.checkpoint_paths import run_log_dir

        return run_log_dir(self.run_dir) if self.run_dir is not None else None

    @staticmethod
    def workspace_path(data_dir: str) -> Path:
        """A workspace path as the training process resolves it: relative to
        the project root, where the process runs."""
        path = Path(data_dir or "data/workspaces/default")
        return path if path.is_absolute() else PROJECT_ROOT / path

    # =========================================================================
    # The GPU slot
    # =========================================================================

    def _status_locked(self, job_type: str) -> str:
        job = self._last.get(job_type)
        return job.status if job is not None else "idle"

    def _slot_refusal_locked(self, wanted: JobType) -> Optional[str]:
        """Why `wanted` cannot start now, or None. Called under the lock.

        **Busy means running or stopping, for any type**, so a start while a
        stop is still waiting for its process is refused, and the process that
        stop is ending can never be replaced under it. Then the deployment's
        GPU lock: a job started elsewhere (a CLI over SSH) is refused here,
        naming its holder, and a lock setting that cannot be used refuses
        every start rather than running unlocked.
        """
        active = self._active
        if active is not None and active.status in BUSY_STATUSES:
            doing = "is already running" if active.status == "running" else "is still stopping"
            return (f"{active.kind.noun} {doing} (job {active.job_id}); one GPU job runs at "
                    f"a time, so {wanted.noun.lower()} cannot start until it has ended. "
                    "Nothing was started.")
        lock = lock_state()
        if lock["problem"]:
            return f"The GPU lock cannot be used: {lock['problem']} Nothing was started."
        if lock["held_by"]:
            return (f"The GPU is in use: the GPU lock {lock['path']} is held by "
                    f"{lock['held_by']}. One managed GPU job runs at a time. "
                    "Nothing was started.")
        return None

    def _spawn_locked(self, kind: JobType, argv: List[str], *,
                      inputs: Dict[str, Any], cleanup: Sequence[Path] = (),
                      reports_dir: Optional[Path] = None,
                      reports_before: Optional[Set[str]] = None) -> Tuple[Optional[_Job], str]:
        """Launch `argv` as a job of `kind` and start its monitor. Called under
        the lock, after `_slot_refusal_locked` found the slot free.

        The child inherits the backend's environment (no `env=`): the CUDA
        allocator (tests/unit/test_train_allocator.py) and the GPU lock's
        settings, so it takes the lock the runner checked.
        """
        job_id = f"{kind.name}-{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"
        logger.info("Starting %s job %s: %s", kind.name, job_id, " ".join(argv))
        try:
            process = subprocess.Popen(
                argv,
                cwd=str(PROJECT_ROOT),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                # Output that is not valid in the locale's encoding must not stop
                # the reader: a reader that stops lets the pipe fill and blocks
                # the job.
                errors="replace",
                # On Windows use CREATE_NEW_PROCESS_GROUP for graceful stop
                creationflags=(
                    subprocess.CREATE_NEW_PROCESS_GROUP
                    if sys.platform == "win32"
                    else 0
                ),
            )
        except (OSError, ValueError) as e:
            error = f"The {kind.name} process could not be started: {e}"
            self._last[kind.name] = _Job(kind, job_id, datetime.now(), status="failed",
                                         error_message=error, inputs=inputs,
                                         ended=datetime.now())
            for path in cleanup:
                path.unlink(missing_ok=True)
            logger.error(error)
            return None, error

        job = _Job(kind, job_id, datetime.now(), process=process, inputs=inputs,
                   cleanup=list(cleanup), reports_dir=reports_dir,
                   reports_before=set(reports_before or ()))
        self._last[kind.name] = job
        self._active = job
        threading.Thread(target=self._monitor, args=(job,), daemon=True,
                         name=f"{kind.name}-monitor").start()
        return job, ""

    def active_job(self) -> Optional[Dict[str, Any]]:
        """The job holding the GPU slot (running or stopping), or None."""
        with self._lock:
            job = self._active
            if job is None:
                return None
            return {"job_type": job.kind.name, "job_id": job.job_id, "status": job.status,
                    "pid": job.process.pid if job.process else None,
                    "start_time": job.started.isoformat()}

    def gpu_state(self) -> Dict[str, Any]:
        """The GPU slot as both job types' statuses show it: the job holding it,
        if any, and the deployment's GPU lock at this instant."""
        active = self.active_job()
        return {"busy": active is not None, "job": active, "lock": lock_state()}

    # =========================================================================
    # Training Control
    # =========================================================================

    def start_training(
        self,
        config: Dict[str, Any],
        effective_rules: Optional[Callable[[Dict[str, Any]], List[str]]] = None,
    ) -> Dict[str, Any]:
        """
        Start a training run as a subprocess.

        Args:
            config: Training configuration dict (mirrors TrainConfig fields).
            effective_rules: for a run continued from a parent, rules checked on
                the settings the run plan resolves (the API's cross-field
                rules); a violation is refused with `invalid_settings`.

        Returns:
            Status dict with success/error info. Refused while any GPU job is
            running or stopping, or while the GPU lock is held elsewhere.
        """
        # A run continued from a parent is decided by the parent's record
        # (contract M3a-3): refused here as the training process would refuse
        # it, and placed under the parent's architecture, which it inherits.
        # The process decides again on the bytes it restores. Read before the
        # lock, so the status stays readable meanwhile.
        architecture = config.get("conv_type")
        if config.get("resume_from"):
            preview = self.preview_continuation(
                config["resume_from"], device=config.get("device", "auto"),
                extend_epochs=config.get("extend_epochs"),
                requested={k: v for k, v in config.items() if k != "resume_from"})
            if not preview["ok"]:
                return {"success": False, "error": preview["reason"],
                        "status": self.get_job_status_word("training")}
            errors = effective_rules(preview["learning_settings"]) if effective_rules else []
            if errors:
                return {"success": False, "invalid_settings": True,
                        "status": self.get_job_status_word("training"),
                        "error": "The settings this run would use, its parent's with this "
                                 "request's: " + " ".join(errors)}
            architecture = preview.get("architecture") or architecture

        kind = JOB_TYPES["training"]
        with self._lock:
            refusal = self._slot_refusal_locked(kind)
            if refusal:
                return {"success": False, "error": refusal,
                        "status": self._status_locked("training")}

            # **The run's own directory** (contract M3a-2): named here, so the
            # status and the console know where its logs and checkpoints will be,
            # and handed to the training process, which claims it after its
            # inputs pass. An explicit one must be free; nothing is created here.
            from src.utils.checkpoint_paths import (
                check_run_dir_free, propose_run_dir, refuse_retired_locations,
            )

            try:
                refuse_retired_locations(config, source="This training request")
                if config.get("run_dir"):
                    run_dir = check_run_dir_free(config["run_dir"])
                else:
                    run_dir = propose_run_dir(
                        self.workspace_path(config.get("data_dir", "")),
                        architecture, config.get("run_name"))
            except (ValueError, OSError) as e:
                return {"success": False, "error": str(e),
                        "status": self._status_locked("training")}

            config = {**config, "run_dir": str(run_dir)}
            self._config = config
            self.run_dir = run_dir

            # Write config to temp YAML
            temp_fd, temp_path = tempfile.mkstemp(suffix=".yaml", prefix="train_config_")
            config_path = Path(temp_path)
            try:
                with os.fdopen(temp_fd, "w") as f:
                    yaml.dump(config, f, default_flow_style=False)
            except Exception:
                config_path.unlink(missing_ok=True)
                raise

            cmd = [sys.executable, str(PROJECT_ROOT / kind.script), "--config", str(config_path)]
            job, error = self._spawn_locked(kind, cmd, inputs={"config_path": str(config_path)},
                                            cleanup=[config_path])
            if job is None:
                return {"success": False, "error": error, "status": "failed"}

            logger.info(f"Training started (PID: {job.process.pid}), run directory {run_dir}")
            return {
                "success": True,
                "job_id": job.job_id,
                "pid": job.process.pid,
                "status": "running",
                "config_path": str(config_path),
                "run_dir": str(run_dir),
            }

    def stop_training(self) -> Dict[str, Any]:
        """
        Stop the running training subprocess gracefully; a test run is never
        stopped by it.

        Returns:
            Status dict with result info.
        """
        result = self.stop("training")
        if not result.get("success"):
            return result
        # **Stopped, not completed** (contract M3a): the stage did not finish.
        # What can be continued is the run's last saved snapshot, if any;
        # progress after it is lost.
        return {"success": True, "status": result["status"], "outcome": self.latest_snapshot()}

    # =========================================================================
    # Test runs
    # =========================================================================

    def start_test(self, inputs: TestRunInputs) -> Dict[str, Any]:
        """Start a test run: scripts/measure_scorer.py on one registered cohort
        version (M5), the real CLI, so it stays the single producer.

        `inputs` come from the server's lists and settings; the caller gates
        the call with D8 first (WP2). The run checks the model and the cohort
        itself when it starts, against the digests chosen, never trusting a
        preview. The version's report store is listed before launch, so the
        report this run creates can be told apart when it ends.
        """
        kind = JOB_TYPES["test"]
        try:
            reports_dir = inputs.reports_dir()
            device = deployment_device()
        except ValueError as exc:
            return {"success": False, "error": f"{exc} Nothing was started.",
                    "status": self.get_job_status_word("test")}

        with self._lock:
            refusal = self._slot_refusal_locked(kind)
            if refusal:
                return {"success": False, "error": refusal, "status": self._status_locked("test")}
            try:
                before = _report_names(reports_dir)
            except OSError as exc:
                return {"success": False, "status": self._status_locked("test"),
                        "error": f"The report store {reports_dir} cannot be listed ({exc}), so "
                                 "this run's report could not be told from others when it "
                                 "ends. Nothing was started."}
            job, error = self._spawn_locked(
                kind, build_test_argv(inputs, device),
                inputs={**inputs.summary(), "device": device or "auto"},
                reports_dir=reports_dir, reports_before=before)
            if job is None:
                return {"success": False, "error": error, "status": "failed"}
            return {"success": True, "job_id": job.job_id, "pid": job.process.pid,
                    "status": "running"}

    def cancel_test(self) -> Dict[str, Any]:
        """Cancel the running test run; training is never stopped by it.

        Returns, once the process has ended, the run's status (`cancelled`, or
        `completed` when the run had already finished cleanly) and what it left
        (`outcome`, read from the disk).
        """
        result = self.stop("test")
        if not result.get("success"):
            return result
        with self._lock:
            job = self._last.get("test")
            outcome = job.outcome if job is not None else None
        return {"success": True, "status": result["status"], "outcome": outcome}

    # =========================================================================
    # Stop, by job type
    # =========================================================================

    def stop(self, job_type: str) -> Dict[str, Any]:
        """Stop the running job of `job_type`: SIGINT (CTRL_BREAK_EVENT on
        Windows), up to `STOP_GRACE_SECONDS`; then SIGTERM, up to
        `TERMINATE_GRACE_SECONDS`; then SIGKILL. Blocks until the process has
        ended and its monitor has recorded how (routes call it from a thread).

        **Refused across types**: a training stop never stops a test run, nor
        the reverse.
        """
        kind = JOB_TYPES[job_type]
        with self._lock:
            job = self._active
            if job is None or job.kind.name != job_type:
                other = (f": the GPU job running is {job.kind.noun.lower()}, which this stop "
                         "does not reach" if job is not None else "")
                return {"success": False, "error": kind.none_running + other,
                        "status": self._status_locked(job_type)}
            if job.status != "running":
                return {"success": False, "error": f"{kind.noun} is already stopping.",
                        "status": job.status}
            job.status = "stopping"
            job.stop_requested = True
            process = job.process

        logger.info("Stopping %s job %s (PID: %s)...", job_type, job.job_id, process.pid)
        try:
            self._end_process(job, process)
        except OSError as e:
            logger.error(f"Error stopping {job_type}: {e}")
            with self._lock:
                job.error_message = f"Stop error: {e}"
            # The process may still run: it keeps the slot, as "stopping", until
            # its monitor sees it end.
            return {"success": False, "error": str(e), "status": job.status}

        if process.poll() is None:
            return {"success": False, "status": "stopping",
                    "error": f"{kind.noun} did not end after SIGKILL; it keeps the GPU slot "
                             "until it does."}
        # Ended: its monitor records how, and reads what a test run left.
        job.done.wait(timeout=RESULT_WAIT_SECONDS)
        with self._lock:
            status = job.status
        logger.info("%s job %s ended: %s", job_type, job.job_id, status)
        return {"success": True, "status": status}

    @staticmethod
    def _end_process(job: _Job, process: subprocess.Popen) -> None:
        if sys.platform == "win32":
            # On Windows, send CTRL_BREAK_EVENT to the process group
            process.send_signal(signal.CTRL_BREAK_EVENT)
            job.stop_signal = "CTRL_BREAK_EVENT"
        else:
            process.send_signal(signal.SIGINT)
            job.stop_signal = "SIGINT"
        try:
            process.wait(timeout=STOP_GRACE_SECONDS)
            return
        except subprocess.TimeoutExpired:
            logger.warning("%s job %s did not stop gracefully, terminating...",
                           job.kind.name, job.job_id)
        process.terminate()
        job.stop_signal = "SIGTERM"
        try:
            process.wait(timeout=TERMINATE_GRACE_SECONDS)
            return
        except subprocess.TimeoutExpired:
            pass
        process.kill()
        job.stop_signal = "SIGKILL"
        try:
            process.wait(timeout=KILL_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            logger.error("%s job %s did not end after SIGKILL", job.kind.name, job.job_id)

    # =========================================================================
    # The monitor
    # =========================================================================

    @staticmethod
    def _drain(job: _Job) -> None:
        """Read the job's output until it closes, keeping its tail (prevents the
        pipe buffer from filling up)."""
        stream = job.process.stdout
        try:
            for line in iter(stream.readline, ""):
                text = line.rstrip("\r\n")[:_MAX_LINE_CHARS]
                logger.debug("[%s] %s", job.kind.name, text)
                with job.log_lock:
                    job.log.append(text)
        except (OSError, ValueError):
            pass

    def _monitor(self, job: _Job) -> None:
        """Wait for the job's own process (never another job's), read what it
        left, and record how it ended; the GPU slot is free only then."""
        returncode: Optional[int] = None
        outcome: Optional[Dict[str, Any]] = None
        try:
            reader = threading.Thread(target=self._drain, args=(job,), daemon=True,
                                      name=f"{job.kind.name}-output")
            reader.start()
            returncode = job.process.wait()
            reader.join(timeout=OUTPUT_GRACE_SECONDS)
            if job.kind.name == "test":
                outcome = self._test_outcome(job, returncode)
        except Exception as exc:  # noqa: BLE001 - the slot must be freed whatever happened
            logger.exception("Monitoring %s job %s failed", job.kind.name, job.job_id)
            outcome = {"report_dir": None, "report": None, "kept": None, "registered": None,
                       "summary": f"What this run left cannot be checked ({exc})."} \
                if job.kind.name == "test" else None
        finally:
            with self._lock:
                job.exit_code = returncode
                job.ended = datetime.now()
                job.outcome = outcome
                job.status = self._final_status(job, returncode)
                if job.status == "failed" and job.error_message is None:
                    job.error_message = self._failure_text(job, returncode)
                if self._active is job:
                    self._active = None
            for path in job.cleanup:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            job.done.set()
            logger.info(f"{job.kind.name.capitalize()} process ended (code={returncode}, "
                        f"status={job.status})")

    @staticmethod
    def _final_status(job: _Job, returncode: Optional[int]) -> str:
        if job.stop_requested:
            # An operator's stop: training is "stopped" whatever the exit code
            # (contract M3a; its snapshot says how the stage stood). A test run
            # that had already finished cleanly when the cancel arrived is
            # completed: its report says so, and calling it cancelled would not
            # be true.
            if not (job.kind.name == "test" and returncode == 0):
                return job.kind.stopped_status
        return "completed" if returncode == 0 else "failed"

    @staticmethod
    def _failure_text(job: _Job, returncode: Optional[int]) -> str:
        # Filter out Intel MKL/Fortran noise from error output
        filtered = [
            line for line in job.tail(20)
            if "forrtl:" not in line
            and "KERNELBASE" not in line
            and "KERNEL32" not in line
            and "ntdll.dll" not in line
            and "Unknown               Unknown  Unknown" not in line
        ]
        tail = "\n".join(filtered) if filtered else ""
        how = (f"Process ended by {_signal_name(returncode)}"
               if isinstance(returncode, int) and returncode < 0
               else f"Process exited with code {returncode}")
        return how + (f"\n\n--- Last output ---\n{tail}" if tail else "")

    def _test_outcome(self, job: _Job, returncode: int) -> Dict[str, Any]:
        """What a test run left, read from the disk once its process ended,
        whatever its exit status: the publication contract decides, never the
        exit code alone.

        **Which report is this run's**: the run directory that appeared in the
        version's report store during the run with this checkpoint's digest in
        its name (the run id, `create_report_dir`). None means nothing was
        kept; two or more (another writer in the same version, against D7's
        single-writer rule) means it cannot be told, and nothing is claimed.
        Whether that report was kept and registered comes from the report-state
        reader (WP4's helper), never from the exit status.
        """
        reports = job.reports_dir
        sha12 = job.inputs["checkpoint_sha256"][:12]
        base: Dict[str, Any] = {"report_dir": None, "report": None, "kept": None,
                                "registered": None}
        try:
            after = _report_names(reports)
        except OSError as exc:
            return {**base, "summary": f"The report store {reports} cannot be listed ({exc}), "
                                       "so whether this run kept a report cannot be checked."}
        new = sorted(name for name in after - job.reports_before if f"-{sha12}-" in name)
        if not new:
            return {**base, "kept": False, "registered": False,
                    "summary": f"No report directory was created under {reports}: nothing "
                               "was kept, so nothing was registered."}
        if len(new) > 1:
            return {**base, "summary": f"{len(new)} report directories for this checkpoint "
                                       f"appeared during the run ({', '.join(new)}); which one "
                                       "is this run's cannot be told, so neither is claimed."}
        run_dir = reports / new[0]
        report = run_dir / "report.json"
        state = dict(self._report_state(report, Path(job.inputs["checkpoint"])))
        kept, registered = state.get("kept"), state.get("registered")
        reason = state.get("reason")
        if kept is True:
            said = (f"A report was kept at {run_dir}"
                    + ("; it is registered." if registered is True
                       else "; it is not registered." if registered is False
                       else "; whether it is registered cannot be checked."))
        elif kept is False:
            said = (f"No report was kept: {run_dir} holds no report.json (an interrupted "
                    "write, which nothing registers).")
        else:
            said = f"Whether a report was kept in {run_dir} cannot be checked."
        if reason:
            said += f" ({reason})"
        if job.stop_signal in ("SIGTERM", "SIGKILL"):
            said += (f" The run was ended by {job.stop_signal}, so staged files (*.tmp) may be "
                     f"left in {run_dir}.")
        return {"report_dir": str(run_dir), "report": str(report), "kept": kept,
                "registered": registered, "summary": said}

    # =========================================================================
    # Status & Metrics
    # =========================================================================

    def get_job_status_word(self, job_type: str) -> str:
        """The status of the latest job of `job_type` ("idle" before any)."""
        with self._lock:
            return self._status_locked(job_type)

    @staticmethod
    def _job_fields(job: Optional[_Job]) -> Dict[str, Any]:
        if job is None:
            return {"status": "idle", "job_id": None, "pid": None, "start_time": None,
                    "end_time": None, "elapsed_seconds": None, "exit_code": None,
                    "error_message": None}
        elapsed = None
        if job.status in BUSY_STATUSES:
            elapsed = (datetime.now() - job.started).total_seconds()
        return {
            "status": job.status,
            "job_id": job.job_id,
            "pid": job.process.pid if job.process else None,
            "start_time": job.started.isoformat(),
            "end_time": job.ended.isoformat() if job.ended else None,
            "elapsed_seconds": elapsed,
            "exit_code": job.exit_code,
            "error_message": job.error_message,
        }

    def get_test_status(self) -> Dict[str, Any]:
        """The latest test run: its status, what it was given, its exit status
        and what that status means, and what it left (`outcome`: the report,
        kept, registered, read from the disk once it ended)."""
        with self._lock:
            job = self._last.get("test")
            result = self._job_fields(job)
            result["inputs"] = dict(job.inputs) if job is not None else None
            result["outcome"] = dict(job.outcome) if job is not None and job.outcome else None
            code = job.exit_code if job is not None else None
        result["exit_meaning"] = (
            None if code is None
            else f"ended by {_signal_name(code)}" if code < 0
            else _TEST_EXIT_MEANING.get(code, "an exit status measure_scorer does not define"))
        result["gpu"] = self.gpu_state()
        return result

    def get_log(self, job_type: str) -> Dict[str, Any]:
        """The latest `job_type` job's own output tail (up to `LOG_TAIL_LINES`
        lines, stdout and stderr merged)."""
        JOB_TYPES[job_type]
        with self._lock:
            job = self._last.get(job_type)
        if job is None:
            return {"job_type": job_type, "job_id": None, "lines": []}
        return {"job_type": job_type, "job_id": job.job_id, "lines": job.tail()}

    def _read_progress_file(self) -> Optional[Dict[str, Any]]:
        """Read the lightweight progress.json file."""
        if self.log_dir is None:
            return None
        progress_file = self.log_dir / "progress.json"
        if not progress_file.exists():
            return None
        try:
            with open(progress_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def get_status(self) -> Dict[str, Any]:
        """Get current training status snapshot, with the GPU slot (`gpu`):
        a test run holding it shows there, never as training."""
        with self._lock:
            result = self._job_fields(self._last.get("training"))
            result["config"] = self._config
            result["run_dir"] = str(self.run_dir) if self.run_dir is not None else None
        result["gpu"] = self.gpu_state()

        # How the run stands, from its last snapshot's own record, once the
        # process is not running: finished, paused, stopped early, or where it
        # can be continued from (contract M3a).
        if result["status"] in ("stopped", "completed", "failed"):
            result["outcome"] = self.latest_snapshot()

        # Read progress file for real-time batch-level progress
        progress = self._read_progress_file()
        if progress:
            result["phase"] = progress.get("phase")
            result["batch"] = progress.get("batch")
            result["total_batches"] = progress.get("total_batches")

        # Augment with latest metrics from log files
        metrics = self._read_latest_metrics()
        if metrics:
            # Convert 0-indexed epoch to 1-indexed for display
            raw_epoch = metrics.get("epoch")
            result["current_epoch"] = (raw_epoch + 1) if raw_epoch is not None else None
            # The run's own count (its active stage's end), which an inherited
            # or extended run does not carry in the request.
            result["total_epochs"] = (progress or {}).get("total_epochs") or (
                self._config.get("num_epochs") if self._config else None
            )
            result["latest_metrics"] = metrics
        elif progress:
            # No epoch-level metrics yet; use progress file for epoch info
            raw_epoch = progress.get("epoch")
            result["current_epoch"] = (raw_epoch + 1) if raw_epoch is not None else None
            result["total_epochs"] = progress.get("total_epochs") or (
                self._config.get("num_epochs") if self._config else None
            )

        return result

    def get_metrics_history(self) -> Dict[str, Any]:
        """
        Read full metrics history from MetricsLogger JSON logs.

        Returns:
            Dict with 'train' and 'validation' metric lists.
        """
        log_file = self._find_latest_log_file()
        if log_file is None:
            return {"train": [], "validation": []}

        try:
            with open(log_file) as f:
                data = json.load(f)
            return {
                "train": data.get("train", []),
                "validation": data.get("validation", []),
            }
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read metrics log: {e}")
            return {"train": [], "validation": []}

    def _read_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """Read the most recent metrics entry from log files."""
        log_file = self._find_latest_log_file()
        if log_file is None:
            return None

        try:
            with open(log_file) as f:
                data = json.load(f)
            train_history = data.get("train", [])
            if train_history:
                return train_history[-1]
        except (json.JSONDecodeError, OSError):
            pass
        return None

    def _find_latest_log_file(self) -> Optional[Path]:
        """Find the most recently modified training log file."""
        if self.log_dir is None or not self.log_dir.exists():
            return None

        log_files = sorted(
            self.log_dir.glob("train_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return log_files[0] if log_files else None

    def latest_snapshot(self) -> Optional[Dict[str, Any]]:
        """The run's latest saved training state (`last.pt`) as its own record
        describes it, or `None` when the run saved none.

        Read whole with the safe loader, so only when asked (the stop
        confirmation, a status after the process ended) and cached by the
        file's size and modification time.
        """
        from src.utils.checkpoint_paths import run_checkpoint_dir

        if self.run_dir is None:
            return None
        last = run_checkpoint_dir(self.run_dir) / "last.pt"
        try:
            stat = last.stat()
        except OSError:
            return None
        key = (str(last), stat.st_mtime_ns, stat.st_size)
        if self._snapshot_cache and self._snapshot_cache[0] == key:
            return self._snapshot_cache[1]
        try:
            from src.utils.checkpoint_io import read_checkpoint

            record = read_checkpoint(last, map_location="cpu", weights_only=True).checkpoint
        except Exception as exc:  # noqa: BLE001 - shown, not raised
            snapshot = {"checkpoint": str(last), "readable": False,
                        "reason": f"{type(exc).__name__}: {exc}"}
        else:
            progress = (record.get("progress") or {}) if isinstance(record, dict) else {}
            stage = progress.get("stage") or {}
            end = progress.get("end")
            snapshot = {
                "checkpoint": str(last),
                "readable": True,
                "completed_epochs": progress.get("completed_epochs"),
                "target_epochs": (int(stage.get("start_epoch", 0))
                                  + int(stage.get("target_epochs", 0))) if stage else None,
                "end": end,
                "stop_reason": progress.get("stop_reason"),
                "resumable": (isinstance(record, dict)
                              and record.get("checkpoint_kind") == "training_state"
                              and end in (None, "paused")),
            }
        self._snapshot_cache = (key, snapshot)
        return snapshot

    def preview_continuation(
        self,
        parent_path: str,
        *,
        device: str = "auto",
        extend_epochs: Optional[int] = None,
        requested: Optional[Dict[str, Any]] = None,
        extend_if_finished: bool = False,
    ) -> Dict[str, Any]:
        """What starting from `parent_path` would do, for the console to show
        before start (contract M3a-3, M3a-4).

        `requested` is a training request's own fields, set explicitly: a
        learning setting among them that differs from the parent's is refused
        here, as the training process would refuse it.

        `extend_if_finished` is the console's: its Extra epochs field belongs to
        a finished parent, so for any other parent the value is not sent
        (`training_console._preview_for`). Every other caller leaves it off, and
        an `extend_epochs` given for an unfinished parent is refused.

        **The same decision the training process makes**
        (`src.utils.run_plan.plan_continuation`), on the parent's own
        record read with the safe loader. A preview is not the check: the
        training process reads the parent again, checks it against the
        workspace's data and graph (M2.3) and decides again on those bytes.
        """
        from src.utils.run_plan import OPERATIONAL_FIELDS, RunPlanRefused, plan_continuation
        from src.utils.checkpoint_io import read_checkpoint

        path = Path(parent_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            return {"ok": False, "reason": f"{path} does not exist."}
        try:
            read = read_checkpoint(path, map_location="cpu", weights_only=True)
        except Exception as exc:  # noqa: BLE001 - a refusal to show
            return {"ok": False,
                    "reason": f"{path.name} cannot be read as a checkpoint "
                              f"({type(exc).__name__}: {exc})."}
        record = read.checkpoint if isinstance(read.checkpoint, dict) else {}
        progress = record.get("progress") or {}
        stage = progress.get("stage") or {}
        if extend_if_finished and progress.get("end") != "finished":
            extend_epochs = None
        facts = {
            "parent": str(path),
            "sha256": read.identity.sha256,
            "end": progress.get("end"),
            "completed_epochs": progress.get("completed_epochs"),
            "target_epochs": (int(stage.get("start_epoch", 0))
                              + int(stage.get("target_epochs", 0))) if stage else None,
            "architecture": ((record.get("config") or {}).get("model_config") or {}).get(
                "conv_type"),
        }
        # The parent's own settings, with the request's fields over them; only
        # the request's fields count as set explicitly, so every other learning
        # setting shows as inherited.
        explicit = dict(requested or {})
        recorded = (record.get("config") or {}).get("requested")
        request = {**(recorded if isinstance(recorded, dict) else {}), **explicit,
                   "resume_from": str(path), "run_dir": "", "extend_epochs": extend_epochs,
                   "extend_lr": explicit.get("extend_lr"), "pause_after_epoch": None,
                   "device": device}
        effective_device = "cuda" if device == "auto" else device
        try:
            plan = plan_continuation(record, path, request, set(explicit),
                                     device=effective_device,
                                     parent_sha256=read.identity.sha256)
        except RunPlanRefused as exc:
            return {**facts, "ok": False, "reason": str(exc)}
        return {**facts, "ok": True, "mode": plan.mode, "action": plan.action,
                "inherited": plan.inherited, "start_epoch": plan.start_epoch,
                "end_epoch": plan.end_epoch, "stage": plan.stage,
                # The learning settings the run would use (the parent's, with
                # the request's explicit values), for checks that need them
                # complete: the API's cross-field rules. Operational fields are
                # left out: the process takes them from the request or its own
                # defaults, which this preview does not know.
                "learning_settings": {k: v for k, v in plan.settings.items()
                                      if k not in OPERATIONAL_FIELDS}}

    # =========================================================================
    # Checkpoint Management
    # =========================================================================

    def get_checkpoints(
        self, data_dir: Optional[str] = None, conv_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List a workspace's run checkpoints with metadata (contract M3a-2).

        Every run's, or one architecture's, through the shared listing, each
        file read once (`checkpoint_paths.read_run_checkpoints`, M5-1): runs
        most recently trained first, newest file first within a run. The
        workspace defaults to the last training request's. What each file is
        comes from its own record; a listed file is not thereby verified, and a
        file outside the run layout is reached by its path.

        Each entry carries the SHA-256 of the bytes read (`sha256`), the
        Trainer's validation figure (`validation`: metric and value, or None),
        the dataset fingerprint it records (`dataset_fingerprint`) and, for a
        file the read refused, why (`metadata_problem`), beside the keys the
        console and the route already read.

        Returns:
            List of checkpoint info dicts.
        """
        from src.utils.checkpoint_paths import read_run_checkpoints, run_architecture

        if data_dir is None:
            if not self._config:
                return []
            data_dir = self._config.get("data_dir", "")
        workspace = self.workspace_path(data_dir)

        checkpoints = []
        for entry in read_run_checkpoints(workspace, conv_type):
            try:
                stat = entry.path.stat()
            except OSError:
                continue  # removed since it was listed: there is nothing to show
            info: Dict[str, Any] = {
                "filename": entry.path.name,
                "filepath": str(entry.path),
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "run": entry.run_dir.name,
                "run_dir": str(entry.run_dir),
                "architecture": run_architecture(entry.run_dir),
                "sha256": entry.sha256,
                "validation": ({"metric": entry.validation[0], "value": entry.validation[1]}
                               if entry.validation else None),
                "dataset_fingerprint": entry.dataset_fingerprint,
                "metadata_problem": entry.problem,
            }
            # The file's own record, from the same read as the digest; a file
            # the read refused has none.
            if entry.logs is not None:
                info["epoch"] = entry.epoch
                info["metrics"] = entry.logs
                info["checkpoint_kind"] = entry.checkpoint_kind

            checkpoints.append(info)

        return checkpoints

    # =========================================================================
    # System Resource Monitoring
    # =========================================================================

    @staticmethod
    def get_system_resources() -> Dict[str, Any]:
        """
        Get GPU and RAM resource utilization.

        Uses nvidia-smi subprocess for GPU stats (avoids importing torch in webui).
        Uses /proc/meminfo or psutil for RAM.

        Returns:
            Dict with gpu and ram fields.
        """
        gpu_info = TrainingManager._get_gpu_info()
        ram_info = TrainingManager._get_ram_info()
        resources: Dict[str, Any] = {
            "gpu": gpu_info,
            "ram": ram_info,
            "unified_memory": TrainingManager._detect_unified_memory(gpu_info, ram_info),
            "timestamp": datetime.now().isoformat(),
        }
        return resources

    # Truly integrated SoCs where the GPU and CPU share ONE physical pool and
    # nvidia-smi (and NVML) report GPU memory as [N/A] / NOT_SUPPORTED, so there
    # is no separate VRAM figure to show. Matched case-insensitively vs GPU name.
    # NOTE: Grace Hopper (GH200) and Grace-Blackwell superchips (GB200/GB300) are
    # deliberately EXCLUDED — they have discrete HBM that nvidia-smi reports
    # correctly, so their VRAM bar is meaningful and must stay visible.
    _UNIFIED_GPU_TAGS = ("GB10", "JETSON", "ORIN", "XAVIER", "THOR", "TEGRA")

    @staticmethod
    def _detect_unified_memory(gpu_info: Dict[str, Any], ram_info: Dict[str, Any]) -> bool:
        """Detect a unified-memory architecture (e.g. DGX Spark GB10, Grace Hopper).

        Two independent signals, either is sufficient:
          1. GPU name matches a known integrated/unified SoC.
          2. GPU-visible memory is about as large as total system RAM — physically
             only possible when they are the same shared pool (a discrete GPU's
             VRAM is far smaller than system RAM).
        torch is intentionally not imported here (webui avoids the torch dep).
        """
        devices = gpu_info.get("devices", [])
        if not gpu_info.get("available") or not devices:
            return False

        # Signal 1: known integrated SoC by name.
        for dev in devices:
            name = str(dev.get("name", "")).upper()
            if any(tag in name for tag in TrainingManager._UNIFIED_GPU_TAGS):
                return True

        # Signal 2: GPU memory ~= system RAM (shared pool). Conservative ratio so
        # high-VRAM discrete cards on small-RAM hosts don't false-positive.
        ram_total_mb = float(ram_info.get("total_gb", 0) or 0) * 1024.0
        gpu_total_mb = max((float(d.get("memory_total_mb", 0) or 0) for d in devices), default=0.0)
        if ram_total_mb > 0 and gpu_total_mb > 0:
            if gpu_total_mb / ram_total_mb >= 0.85:
                return True

        return False

    @staticmethod
    def _get_gpu_info() -> Dict[str, Any]:
        """Query GPU info via nvidia-smi."""
        gpu_info: Dict[str, Any] = {
            "available": False,
            "devices": [],
        }

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                gpu_info["available"] = True
                for line in result.stdout.strip().split("\n"):
                    if not line.strip():
                        continue
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 6:
                        # nvidia-smi prints "[N/A]" / "[Not Supported]" for fields a
                        # driver can't read — integrated GPUs like the GB10 return
                        # [N/A] for memory. Parse numerics defensively (-> None) so a
                        # placeholder doesn't crash the whole resource poll; the
                        # frontend renders "N/A" for missing values.
                        try:
                            index = int(parts[0])
                        except ValueError:
                            continue
                        gpu_info["devices"].append(
                            {
                                "index": index,
                                "name": parts[1],
                                "utilization_percent": TrainingManager._safe_float(parts[2]),
                                "memory_used_mb": TrainingManager._safe_float(parts[3]),
                                "memory_total_mb": TrainingManager._safe_float(parts[4]),
                                "temperature_c": TrainingManager._safe_float(parts[5]),
                            }
                        )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass

        return gpu_info

    @staticmethod
    def _safe_float(value: str) -> Optional[float]:
        """Parse an nvidia-smi numeric field, returning None for non-numeric
        placeholders like '[N/A]' or '[Not Supported]' (seen on integrated GPUs)."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _get_ram_info() -> Dict[str, Any]:
        """Get RAM usage info."""
        ram_info: Dict[str, Any] = {}

        try:
            import psutil

            mem = psutil.virtual_memory()
            ram_info = {
                "total_gb": round(mem.total / (1024**3), 2),
                "used_gb": round(mem.used / (1024**3), 2),
                "available_gb": round(mem.available / (1024**3), 2),
                "percent": mem.percent,
            }
        except ImportError:
            # Fallback 1: read /proc/meminfo on Linux
            try:
                meminfo: Dict[str, float] = {}
                with open("/proc/meminfo") as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) >= 2:
                            key = parts[0].rstrip(":")
                            value_kb = float(parts[1])
                            meminfo[key] = value_kb

                total = meminfo.get("MemTotal", 0) / (1024 * 1024)
                available = meminfo.get("MemAvailable", 0) / (1024 * 1024)
                ram_info = {
                    "total_gb": round(total, 2),
                    "used_gb": round(total - available, 2),
                    "available_gb": round(available, 2),
                    "percent": round((total - available) / total * 100, 1) if total > 0 else 0,
                }
            except OSError:
                # Fallback 2: Windows via ctypes kernel32
                try:
                    import ctypes

                    class MEMORYSTATUSEX(ctypes.Structure):
                        _fields_ = [
                            ("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                        ]

                    mem_status = MEMORYSTATUSEX()
                    mem_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem_status))  # type: ignore[attr-defined]

                    total_gb = mem_status.ullTotalPhys / (1024**3)
                    avail_gb = mem_status.ullAvailPhys / (1024**3)
                    used_gb = total_gb - avail_gb
                    ram_info = {
                        "total_gb": round(total_gb, 2),
                        "used_gb": round(used_gb, 2),
                        "available_gb": round(avail_gb, 2),
                        "percent": round(used_gb / total_gb * 100, 1) if total_gb > 0 else 0,
                    }
                except (OSError, AttributeError, ValueError):
                    ram_info = {"error": "Unable to read memory info"}

        return ram_info


# Module-level singleton instance
training_manager = TrainingManager()

__all__ = [
    "TrainingManager",
    "TrainingStatus",
    "training_manager",
]
