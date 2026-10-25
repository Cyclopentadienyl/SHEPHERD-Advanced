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
    - Stops a job by type: SIGINT (CTRL_BREAK_EVENT on Windows) to its main
      process, then SIGTERM, then SIGKILL to its whole process group; a
      training stop never reaches a test run, nor the reverse
    - **Each job leads its own process group** (review of 8242b47, P2-3), so
      the DataLoader workers it forks, which share the GPU lock's open file,
      end with it: after its main process ends the monitor ends what is left
      of the group, and only then records how the job ended and frees the
      slot. A backend being stopped closes the slot, so nothing starts from
      then on, and stops the job it runs (`close`, `end_active_job`, called by
      `src.api.main` once per shutdown: at the shutdown signal, or at the
      lifespan's end when no signal began the shutdown)
    - Reads training metrics from MetricsLogger's JSON log files, and, once a
      test run has ended, what it published (its `--outcome` file) and whether
      that report is kept and registered (the registration's one helper)
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

    **A job's processes, on POSIX** (P2-3): the child calls `setpgid(0, 0)`
    before it runs the script (`process_group=0`), so it and every process
    it forks form a group of their own, never the backend's. A stop sends
    SIGINT to the main process only, the graceful path (training stops where
    it stands, measure_scorer says cancelled, a DataLoader ends its own
    workers as the process exits), and SIGTERM and SIGKILL to the whole
    group, as systemd's ``KillMode=mixed`` does. Once the main process has
    been waited for, the monitor ends whatever of the group is left
    (SIGTERM, then SIGKILL), reaping any of its zombies this process adopted;
    on Linux a member counts as left only while it is not a zombie
    (`/proc/<pid>/stat`), since a zombie holds no descriptor, so not the lock.
    A group is recorded only for a process this module started, with a pid
    above 1, that leads it; anything else is signalled through its own
    methods, so a group signal can never reach every process (pid 1) or the
    backend's own group.

    **On Windows** (the backend is untested there): the job gets its own
    console process group (CREATE_NEW_PROCESS_GROUP), which CTRL_BREAK_EVENT
    is sent to; a backend with no console cannot send it (`OSError`), and the
    stop goes on to TerminateProcess, which ends the main process only.
    DataLoader workers are spawned there, not forked, and do not inherit the
    lock's handle; the release of a dead process's file lock by the system
    can be delayed. A Job Object, the true equivalent of a process group, is
    not added while the Windows backend is untested.

    **Nothing here gates access.** Every caller that starts, stops or lists
    applies D8 first, on its own request (`src/utils/operator_access.py`):
    the training routes through their FastAPI dependency, the Training
    Console's handlers through the Gradio helper. The Test tab's start and
    cancel (M5-2) do the same before calling `start_test` and `cancel_test`.

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
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Set, Tuple

import yaml

from src.config.serving_device import requested_serving_device
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
#: The longest `end_active_job` takes, so the longest a stopping backend
#: spends on its job, which it stops once per shutdown (`src.api.main`'s
#: `end_jobs`; P3-4): a stop's whole escalation, then the wait for its monitor
#: (or, for a job an operator is already stopping, the same bound). The
#: systemd unit's TimeoutStopSec covers it (scripts/service/systemd/
#: shepherd.service; root-cause review, P23-3), so the service manager's
#: SIGKILL never pre-empts this stop.
END_ACTIVE_JOB_SECONDS = (STOP_GRACE_SECONDS + TERMINATE_GRACE_SECONDS + KILL_GRACE_SECONDS
                          + RESULT_WAIT_SECONDS)

_SHA256 = re.compile(r"[0-9a-f]{64}")

#: The class every job's process is started with, as this module imported it.
#: Only a process started through it leads a group of its own (P2-3); a
#: stand-in that is not one (a test's) has no group and is signalled through
#: its own methods.
_POPEN = subprocess.Popen
#: **The signals that no longer reach a job because it leads its own process
#: group** (`_spawn_locked`; root-cause review, P23-2), by name, defined once
#: here where the decoupling is made: a terminal's Ctrl-C (SIGINT), Ctrl-\
#: (SIGQUIT) and hangup (SIGHUP), a SIGTERM sent to the hosting process
#: alone, and on Windows Ctrl-Break (SIGBREAK). Each process that hosts the
#: runner routes every one it does not handle otherwise to its own orderly
#: shutdown, which stops the job: the backend (`src.api.main`'s
#: `_ShutdownSignals`) and the deployment probe
#: (`_stop_on_termination_signals`). A name the platform lacks is skipped.
JOB_DETACHED_SIGNALS: Tuple[str, ...] = ("SIGINT", "SIGQUIT", "SIGHUP", "SIGTERM", "SIGBREAK")
#: How long the monitor lets a job's group settle after its main process was
#: waited for (workers a SIGTERM to the group has just reached, or that a
#: DataLoader is ending) before it says that processes remain and ends them.
_GROUP_SETTLE_SECONDS = 1.0
_GROUP_POLL_SECONDS = 0.05


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
    completed (exit 0), failed, or `stopped_status` after an operator's stop
    (for a test run, only when the stop is what ended it). A stop that arrives
    after the process ended on its own is refused and changes nothing.
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
#: the disk, never from these (`TrainingManager._test_outcome`).
_TEST_EXIT_MEANING = {
    0: "measured (with --record: registered, or already held)",
    1: "refused before anything was written, or measured but the report was not written, "
       "or the run failed after writing it; its output says which",
    2: "the command line was rejected",
    3: "measured, but not registered (the registration was refused, or the report was not "
       "as published); whether a report was kept: see the outcome",
}


#: Reads, for one report file, whether it was kept and whether it is
#: registered, from the disk: `reader(report_path, checkpoint=...,
#: expected_sha256=...)` returns a `registration.ReportState`. The runner's
#: is the registration's one public helper (`registration.report_state`); a
#: test may pass another.
ReportStateReader = Callable[..., Any]

#: The run's outcome file in its job directory (`measure_scorer.py --outcome`).
OUTCOME_FILE = "outcome.json"

#: A `ReportState` answer as the status's ``kept`` and ``registered`` give it:
#: True, False, or None for "cannot be checked".
_STATE_ANSWER = {"kept": True, "registered": True, "not_kept": False,
                 "not_registered": False, "cannot_be_checked": None}


def _registration_report_state(report: Path, **kwargs: Any) -> Any:
    """`src.evaluation.registration.report_state`, imported when a test run
    ends rather than when the runner is."""
    from src.evaluation.registration import report_state

    return report_state(report, **kwargs)


def _remove(path: Path) -> None:
    """Remove a file a job owned, or a directory it made (its job directory)."""
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    except OSError:
        pass


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
    value = requested_serving_device(None)
    if value is None or value == "auto":
        return None
    if value in ("cuda", "cpu"):
        return value
    raise ValueError(
        f"SHEPHERD_DEVICE is {value!r}, and a test run takes auto, cuda or cpu "
        "(scripts/measure_scorer.py --device), so it cannot be given the deployment's device.")


def build_test_argv(inputs: TestRunInputs, device: Optional[str], outcome: Path) -> List[str]:
    """measure_scorer's command line for a test run, from `inputs` only, and
    `outcome`, the new file in the job's own directory where the run says what
    it published (`--outcome`).

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
        "--outcome", str(outcome),
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
    #: Files the job owns and removes when it ends (training's temp YAML, a
    #: test run's job directory).
    cleanup: List[Path] = field(default_factory=list)
    #: A training job's own run directory, where its snapshot is read after a
    #: stop (never the run directory of a job started since).
    run_dir: Optional[Path] = None
    #: The report store listed before launch (test runs).
    reports_dir: Optional[Path] = None
    reports_before: Set[str] = field(default_factory=set)
    #: Where a test run writes its outcome (`--outcome`), in its job directory.
    outcome_path: Optional[Path] = None
    #: The process group the job leads (its main process's pid), which a stop's
    #: SIGTERM and SIGKILL and the monitor's cleanup reach; None on Windows and
    #: for a process that does not lead one (`_job_group`).
    group: Optional[int] = None
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


def _ledger_dir(job: _Job) -> Path:
    """Where a test run's registration writes: the checkpoint's ledger is in
    the directory of the checkpoint's resolved path (M4 decision 15, the rule
    src/evaluation/registration.py files by)."""
    checkpoint = Path(job.inputs["checkpoint"])
    try:
        return checkpoint.resolve().parent
    except (OSError, RuntimeError):
        return checkpoint.parent


def _recorded_dataset(record: Any) -> Optional[str]:
    """The dataset fingerprint a checkpoint record names
    (``training_input_digests["split_manifest"]``, as the shared listing reads
    it, `checkpoint_paths.read_run_checkpoints`), or None."""
    digests = record.get("training_input_digests") if isinstance(record, dict) else None
    value = digests.get("split_manifest") if isinstance(digests, dict) else None
    return value if isinstance(value, str) and _SHA256.fullmatch(value) else None


def _signal_name(returncode: int) -> str:
    try:
        return signal.Signals(-returncode).name
    except ValueError:
        return f"signal {-returncode}"


# ---------------------------------------------------------------------------
# A job's process group (P2-3)
# ---------------------------------------------------------------------------
def _job_group(process: Any) -> Optional[int]:
    """The process group `process` leads, or None.

    Its pid, for a process this module started on POSIX (`process_group=0`:
    the child set its group before it ran anything, so every process it forks
    is in it), once `getpgid` confirms that it leads the group. None on
    Windows, for a stand-in that is not a `Popen`, and for a pid of 0 or 1:
    `killpg(1, sig)` is `kill(-1, sig)`, every process this account may
    signal, and `killpg(0, sig)` the backend's own group, so neither is ever
    a job's."""
    if sys.platform == "win32" or not isinstance(process, _POPEN):
        return None
    pid = process.pid
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1:
        return None
    try:
        leads = os.getpgid(pid) == pid
    except OSError:
        leads = False
    if not leads:
        logger.warning("Job process %s does not lead a process group of its own; a stop "
                       "reaches it alone.", pid)
        return None
    return pid


def _signal_group(group: int, sig: int) -> bool:
    """`sig` to every process of the job's group; False when none is left.

    The group's id is the job's own pid, which the system does not give to
    another process while any process of the group remains (the main process
    included, until it is waited for), so this reaches no other job and never
    the backend."""
    try:
        os.killpg(group, sig)
    except ProcessLookupError:
        return False
    return True


def _live_member(group: int) -> bool:
    """Linux: whether a process of `group` is neither a zombie nor dead, read
    from `/proc/<pid>/stat` (its state, and its process group, the fifth
    field). A zombie holds no descriptor, so not the GPU lock, and one adopted
    by a reaper that never reaps (a container whose PID 1 does not wait, say)
    would otherwise keep the slot forever. True when /proc cannot be listed:
    the group signal said a process is there."""
    try:
        names = os.listdir("/proc")
    except OSError:
        return True
    for name in names:
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat", "rb") as stat:
                fields = stat.read().rsplit(b")", 1)[1].split()
            if int(fields[2]) == group and fields[0] not in (b"Z", b"X"):
                return True
        except (OSError, IndexError, ValueError):
            continue
    return False


def _reap_group(group: int) -> None:
    """Reap the zombies of `group` that are this process's children, without
    waiting for a member that has not ended."""
    while True:
        try:
            pid, _ = os.waitpid(-group, os.WNOHANG)
        except ChildProcessError:  # no child of this process is in the group
            return
        if pid == 0:
            return


def _group_left(group: int) -> bool:
    """Whether a process of the job's group remains. **Only once its main
    process has been waited for** (by `Popen.wait`): this reaps the group's
    zombies that are this process's children, and the main process is one
    until then, whose exit status `Popen` must read itself.

    The zombies reaped are the group's processes this process adopted: one
    orphaned by the job's end is adopted by PID 1 or the nearest subreaper,
    and when that is this backend (PID 1 of a container without an init, say)
    nothing else would ever reap it. Only this group's are waited for, never
    another job's or the backend's own children. Then the group is asked
    (`killpg(group, 0)`), and on Linux only a member that is not a zombie
    counts (`_live_member`); elsewhere every member answering does. When none
    is left on Linux, the group's zombies are reaped once more before the
    answer: a member that ended after the first reap and before it was read
    (an adopted worker still exiting at the reap) is a zombie of this process
    the answer counts as gone, and no later call would come to reap it."""
    _reap_group(group)
    if not _signal_group(group, 0):
        return False
    if not sys.platform.startswith("linux"):
        return True
    if _live_member(group):
        return True
    _reap_group(group)
    return False


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
        #: Who closed the slot, None while it is open: `CLOSED_BY_STOP` once
        #: the backend begins to stop (`close`), `CLOSED_BY_RESTART` while
        #: Restart Backend replaces the process (`close_if_idle`). No job starts
        #: while it is closed, whatever the request (P23-1).
        self._closed_by: Optional[str] = None
        #: A shutdown signal has arrived (`record_stop_signal`), recorded by
        #: the signal handler before the stop it starts has closed the slot;
        #: cleared at startup (`open`).
        self._stop_signalled = False
        #: The latest job of each type, kept after it ends.
        self._last: Dict[str, _Job] = {}
        self._report_state: ReportStateReader = report_state or _registration_report_state
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

    #: Every start's refusal once the slot is closed.
    STOPPING = "The backend is stopping; nothing was started."
    #: Who closed the slot (`_closed_by`): the backend's stop, or a restart.
    CLOSED_BY_STOP = "stop"
    CLOSED_BY_RESTART = "restart"

    def close(self) -> Optional[Dict[str, Any]]:
        """**Close the GPU slot: the backend is stopping** (root-cause
        review, P23-1). From here on every start is refused, by whichever
        route it comes (`_slot_refusal_locked`), and the job holding the slot,
        if any, is returned as `active_job` describes it, **read under the same
        lock acquisition**: a start either took the slot before this, and is
        the job returned, which the caller stops, or comes after it and is
        refused. So no job can start while the backend waits for its
        connections to close, and none outlives it. `src.api.main` closes it
        once per shutdown (round 3 of the review fixes, P3-4): at the
        shutdown signal, or at the lifespan's end when no signal began the
        shutdown. **The stop owns the slot from then on**, also when a restart
        closed it first, so a failed exec does not open it again (`open`)."""
        with self._lock:
            self._closed_by = self.CLOSED_BY_STOP
            return self._describe_active_locked()

    def close_if_idle(self) -> bool:
        """Close the slot for a restart only when it is idle and open, in one
        step under the lock; False, and the slot left as it is, when a job
        holds it or it is closed already (the backend is stopping, or another
        restart runs: verification of P23-1, T2-V2). Restart Backend closes
        it this way before it replaces the process image, so a job started
        during its delay is either seen here or refused, never orphaned by the
        exec; a stop begun since is seen by the exec's own final check
        (`exec_if_closed_for_restart`)."""
        with self._lock:
            if self._active is not None or self._closed_by is not None:
                return False
            self._closed_by = self.CLOSED_BY_RESTART
            return True

    def record_stop_signal(self) -> None:
        """**A shutdown signal has arrived**, recorded by the signal handler
        before anything else it does (round 3 of the review fixes, P3-3), so
        Restart Backend's exec is refused from that moment, also before the
        stop the signal starts has closed the slot. A plain store, never the
        lock: a signal handler runs in the main thread between two of that
        thread's bytecodes, and the thread may be holding the lock."""
        self._stop_signalled = True

    def exec_if_closed_for_restart(self, exec_image: Callable[[], None]) -> bool:
        """**Restart Backend's final check and its exec, one step under the
        lock** (round 3 of the review fixes, P3-3): `exec_image` runs, with
        the lock held, only while the restart still owns the closed slot
        (`close_if_idle`) and no shutdown signal has been recorded
        (`record_stop_signal`); False, and nothing run, otherwise. A `close`
        asked for meanwhile either came first, and the exec is refused, or
        waits on the lock until the process image is replaced. What no check
        can close: a signal whose handler had not recorded it when this check
        ran is lost with the old image. A failed exec raises out of here with
        the lock released (it is not reentrant), for the caller to open its
        own close again (`open(closed_by=CLOSED_BY_RESTART)`)."""
        with self._lock:
            if self._closed_by != self.CLOSED_BY_RESTART or self._stop_signalled:
                return False
            exec_image()
        return True

    def open(self, closed_by: Optional[str] = None) -> None:
        """Open the slot again. At the application's startup, whoever closed
        it (the runner is a process-wide singleton, which an in-process
        client's next lifespan reuses), and a shutdown signal recorded is
        forgotten. With `closed_by`, only while that closer still owns it:
        Restart Backend's failed exec passes `CLOSED_BY_RESTART`, so it
        reopens its own close and never a stop's that came meanwhile, and the
        process serves on."""
        with self._lock:
            if closed_by is None or self._closed_by == closed_by:
                self._closed_by = None
            if closed_by is None:
                self._stop_signalled = False

    def _slot_refusal_locked(self, wanted: JobType) -> Optional[str]:
        """Why `wanted` cannot start now, or None. Called under the lock.

        **A closed slot refuses first** (`close`): the backend is stopping, so
        nothing starts, by any route. **Busy means running or stopping, for
        any type**, so a start while a stop is still waiting for its process
        is refused, and the process that stop is ending can never be replaced
        under it. Then the deployment's GPU lock: a job started elsewhere (a
        CLI over SSH) is refused here, naming its holder, and a lock setting
        that cannot be used refuses every start rather than running unlocked.
        """
        if self._closed_by is not None:
            return self.STOPPING
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
                      run_dir: Optional[Path] = None,
                      reports_dir: Optional[Path] = None,
                      reports_before: Optional[Set[str]] = None,
                      outcome_path: Optional[Path] = None) -> Tuple[Optional[_Job], str]:
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
                # **Each job leads its own process group** (P2-3): on POSIX the
                # child calls setpgid(0, 0) before it runs the script, so every
                # process it forks (DataLoader workers, which share the GPU
                # lock's open file) is in the group a stop and the monitor end,
                # never the backend's; so `JOB_DETACHED_SIGNALS` no longer
                # reach it, and its host routes them. On Windows
                # CREATE_NEW_PROCESS_GROUP, which CTRL_BREAK_EVENT is sent to.
                **({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                   if sys.platform == "win32" else {"process_group": 0}),
            )
        except (OSError, ValueError) as e:
            error = f"The {kind.name} process could not be started: {e}"
            self._last[kind.name] = _Job(kind, job_id, datetime.now(), status="failed",
                                         error_message=error, inputs=inputs,
                                         ended=datetime.now())
            for path in cleanup:
                _remove(path)
            logger.error(error)
            return None, error

        job = _Job(kind, job_id, datetime.now(), process=process, inputs=inputs,
                   cleanup=list(cleanup), run_dir=run_dir, reports_dir=reports_dir,
                   reports_before=set(reports_before or ()), outcome_path=outcome_path,
                   group=_job_group(process))
        self._last[kind.name] = job
        self._active = job
        threading.Thread(target=self._monitor, args=(job,), daemon=True,
                         name=f"{kind.name}-monitor").start()
        return job, ""

    def active_job(self) -> Optional[Dict[str, Any]]:
        """The job holding the GPU slot (running or stopping), or None."""
        with self._lock:
            return self._describe_active_locked()

    def _describe_active_locked(self) -> Optional[Dict[str, Any]]:
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
        *,
        expected_dataset_fingerprint: Optional[str] = None,
        expected_parent_sha256: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Start a training run as a subprocess.

        Args:
            config: Training configuration dict (mirrors TrainConfig fields).
            effective_rules: for a run continued from a parent, rules checked on
                the settings the run plan resolves (the API's cross-field
                rules); a violation is refused with `invalid_settings`.
            expected_dataset_fingerprint: the full dataset fingerprint (the
                SHA-256 of the workspace's ``split_manifest.json``) the caller
                showed when the run was asked for (M5-2: the Training run bar
                sends the one it displayed; the API takes it as an optional
                field). The manifest is read again first, and a different one is
                refused (`dataset_refusal`) before anything is written or
                started. **The training process checks it again on its own one
                manifest read** (`--expect-dataset-sha256`, review of 8242b47,
                P2-2), so a manifest replaced after this check is refused
                there, before the graph and samples are read and before
                anything is written, never trained on.
            expected_parent_sha256: for a run continued from a parent, the
                SHA-256 of the parent the caller showed (the Continue drawer's
                model fingerprint; the API takes it as an optional query
                parameter). Compared by this start's own preview of the parent
                (`preview_continuation(expected_sha256=)`, the one place a
                caller's digest is compared); a different file is refused
                before anything is written or started (``parent_changed:
                True``, for a caller to add what to do). **Whether or not it is
                given, the training process is bound to the parent this start
                decided on** (`--expect-resume-sha256` with the digest of that
                preview's read, root-cause round P22-S1): the run's placement
                under the parent's architecture, the cross-field verdict and the
                extend mode come from that read, and a file replaced after it is
                refused by the process before it is loaded.

        Returns:
            Status dict with success/error info. Refused while any GPU job is
            running or stopping, or while the GPU lock is held elsewhere.
        """
        # The dataset the caller showed, checked first: nothing is read from a
        # parent, named, written or started for a dataset that changed since.
        if expected_dataset_fingerprint is not None:
            refusal = self.dataset_refusal(config.get("data_dir", ""),
                                           expected_dataset_fingerprint)
            if refusal:
                return {"success": False, "error": refusal, "dataset_changed": True,
                        "status": self.get_job_status_word("training")}

        # A run continued from a parent is decided by the parent's record
        # (contract M3a-3): refused here as the training process would refuse
        # it, and placed under the parent's architecture, which it inherits.
        # The process decides again on the same bytes: it is given the digest
        # of this read and refuses any other (P22-S1). Read before the lock, so
        # the status stays readable meanwhile.
        architecture = config.get("conv_type")
        parent_sha256 = None
        if expected_parent_sha256 is not None and not config.get("resume_from"):
            return {"success": False, "status": self.get_job_status_word("training"),
                    "error": "A parent's model fingerprint was sent, but this request "
                             "continues no checkpoint (resume_from). Nothing was started."}
        if config.get("resume_from"):
            preview = self.preview_continuation(
                config["resume_from"], device=config.get("device", "auto"),
                extend_epochs=config.get("extend_epochs"),
                requested={k: v for k, v in config.items() if k != "resume_from"},
                expected_sha256=expected_parent_sha256)
            if not preview["ok"]:
                return {"success": False, "error": preview["reason"],
                        "parent_changed": preview.get("changed", False),
                        "status": self.get_job_status_word("training")}
            parent_sha256 = preview["sha256"]
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
            # The process's own checks on its one read of the parent (P22-S1)
            # and of the manifest (P2-2): arguments, not configuration keys,
            # since they are preconditions of this job and not settings the run
            # records.
            if parent_sha256 is not None:
                cmd += ["--expect-resume-sha256", parent_sha256]
            if expected_dataset_fingerprint is not None:
                cmd += ["--expect-dataset-sha256", expected_dataset_fingerprint]
            job, error = self._spawn_locked(kind, cmd, inputs={"config_path": str(config_path)},
                                            cleanup=[config_path], run_dir=run_dir)
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

    def dataset_refusal(self, data_dir: str, expected: Any) -> Optional[str]:
        """Why a run on `data_dir` must not start for a caller that showed the
        dataset fingerprint `expected`, or None: the workspace's
        ``split_manifest.json`` is read again through its one reader, which
        compares `expected` before decoding (`read_split_manifest(
        expected_sha256=)`, P22-S2) and raises `DatasetChanged`, the refusal
        the training process says too. A value that is not a full SHA-256
        hexdigest is refused, and so is a manifest that cannot be read now:
        nothing then says the run would use what was shown."""
        if not (isinstance(expected, str) and _SHA256.fullmatch(expected)):
            return ("The dataset fingerprint sent with the start is not a full SHA-256 "
                    "hexdigest, so the dataset cannot be checked. Nothing was started.")
        from src.kg.artifacts import MANIFEST_FILENAME, DatasetChanged, read_split_manifest

        try:
            read_split_manifest(self.workspace_path(data_dir), expected_sha256=expected)
        except DatasetChanged as exc:
            return f"{exc} Nothing was started."
        except (ValueError, OSError) as exc:  # the reader's refusals name the path
            return (f"The dataset's {MANIFEST_FILENAME} cannot be read now "
                    f"({type(exc).__name__}). Nothing was started.")
        return None

    def stop_training(self) -> Dict[str, Any]:
        """
        Stop the running training subprocess gracefully; a test run is never
        stopped by it.

        Returns:
            Status dict with result info.
        """
        result, job = self._stop("training")
        if not result.get("success"):
            return result
        # **Stopped, not completed** (contract M3a): the stage did not finish.
        # What can be continued is the run's last saved snapshot, if any;
        # progress after it is lost. Read from the stopped job's own run
        # directory, whatever was started since the stop ended.
        return {"success": True, "status": result["status"],
                "outcome": self.latest_snapshot(job.run_dir)}

    # =========================================================================
    # Test runs
    # =========================================================================

    def start_test(self, inputs: TestRunInputs) -> Dict[str, Any]:
        """Start a test run: scripts/measure_scorer.py on one registered cohort
        version (M5), the real CLI, so it stays the single producer.

        `inputs` come from the server's lists and settings; the caller gates
        the call with D8 first. The run checks the model and the cohort
        itself when it starts, against the digests chosen, never trusting a
        preview. It is given a job directory of its own (removed once the run
        has ended and been read), where it writes its outcome
        (`--outcome`). The version's report store is listed before launch
        too, so a report directory this run created without writing its
        outcome can still be told apart when it ends.
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
            try:
                job_dir = Path(tempfile.mkdtemp(prefix="shepherd-test-job-"))
            except OSError as exc:
                return {"success": False, "status": self._status_locked("test"),
                        "error": f"A directory for this run's outcome could not be made "
                                 f"({exc}). Nothing was started."}
            outcome = job_dir / OUTCOME_FILE
            job, error = self._spawn_locked(
                kind, build_test_argv(inputs, device, outcome),
                inputs={**inputs.summary(), "device": device or "auto"},
                cleanup=[job_dir], reports_dir=reports_dir, reports_before=before,
                outcome_path=outcome)
            if job is None:
                return {"success": False, "error": error, "status": "failed"}
            return {"success": True, "job_id": job.job_id, "pid": job.process.pid,
                    "status": "running"}

    def cancel_test(self) -> Dict[str, Any]:
        """Cancel the running test run; training is never stopped by it.

        Returns, once the process has ended, the run's status (`cancelled`, or
        how the run ended on its own when the cancel arrived as it was ending)
        and what it left (`outcome`, read from the disk), both from the run this
        cancel reached, whatever was started since. A run that had already
        ended before the cancel is refused, with how it ended and its outcome.
        """
        result, job = self._stop("test")
        if job is None:
            return result
        with self._lock:
            outcome = dict(job.outcome) if job.outcome else None
        return {**result, "outcome": outcome}

    # =========================================================================
    # Stop, by job type
    # =========================================================================

    def stop(self, job_type: str) -> Dict[str, Any]:
        """Stop the running job of `job_type`: SIGINT (CTRL_BREAK_EVENT on
        Windows) to its main process, up to `STOP_GRACE_SECONDS`; then SIGTERM
        to its process group, up to `TERMINATE_GRACE_SECONDS`; then SIGKILL to
        the group. Blocks until the main process has ended and its monitor has
        ended the rest of the group and recorded how the job ended (routes call
        it from a thread); "stopped" means no process of the job is left.

        **Refused across types**: a training stop never stops a test run, nor
        the reverse. **Refused when the process has already ended on its own**
        (its monitor is still reading what it left): the stop is not what ended
        it, so the job keeps the status its own ending gives, and the refusal
        says which, once its monitor has recorded it.
        """
        return self._stop(job_type)[0]

    def _stop(self, job_type: str) -> Tuple[Dict[str, Any], Optional[_Job]]:
        """`stop`, with the job it reached (None when it reached none), so a
        caller reads that job's result rather than whichever job is latest."""
        kind = JOB_TYPES[job_type]
        with self._lock:
            job = self._active
            if job is None or job.kind.name != job_type:
                other = (f": the GPU job running is {job.kind.noun.lower()}, which this stop "
                         "does not reach" if job is not None else "")
                return {"success": False, "error": kind.none_running + other,
                        "status": self._status_locked(job_type)}, None
            if job.status != "running":
                return {"success": False, "error": f"{kind.noun} is already stopping.",
                        "status": job.status}, None
            process = job.process
            ended_on_its_own = process.poll() is not None
            if not ended_on_its_own:
                job.status = "stopping"
                job.stop_requested = True

        if ended_on_its_own:
            return self._ended_before_stop(job, process), job

        logger.info("Stopping %s job %s (PID: %s)...", job_type, job.job_id, process.pid)
        try:
            self._end_process(job, process)
        except OSError as e:
            logger.error(f"Error stopping {job_type}: {e}")
            with self._lock:
                job.error_message = f"Stop error: {e}"
            # The process may still run: it keeps the slot, as "stopping", until
            # its monitor sees it end.
            return {"success": False, "error": str(e), "status": job.status}, job

        if process.poll() is None:
            return {"success": False, "status": "stopping",
                    "error": f"{kind.noun} did not end after SIGKILL; it keeps the GPU slot "
                             "until it does."}, job
        # Its main process ended: the monitor ends what is left of its process
        # group, records how the job ended and reads what a test run left.
        # Stopped only once that is done; until then the job keeps the slot.
        if not job.done.wait(timeout=RESULT_WAIT_SECONDS):
            return {"success": False, "status": "stopping",
                    "error": f"{kind.noun}'s main process has ended, but the rest of its "
                             "process group is still being ended, or what it left read; it "
                             "keeps the GPU slot until that is done."}, job
        with self._lock:
            status = job.status
        logger.info("%s job %s ended: %s", job_type, job.job_id, status)
        return {"success": True, "status": status}, job

    def _ended_before_stop(self, job: _Job, process: subprocess.Popen) -> Dict[str, Any]:
        """The refusal of a stop that reached a job whose process had already
        ended on its own, once its monitor has recorded how (up to
        `RESULT_WAIT_SECONDS`)."""
        recorded = job.done.wait(timeout=RESULT_WAIT_SECONDS)
        with self._lock:
            status = job.status
        said = (f"{job.kind.noun} (job {job.job_id}) had already ended on its own "
                f"({status + ', ' if recorded else ''}exit code {process.returncode}) before "
                "this stop arrived; nothing was stopped.")
        if not recorded:
            said += (" Its monitor is still ending what is left of its process group, or "
                     "reading what it left; the job keeps the GPU slot until then.")
        logger.info("Stop of %s job %s refused: %s", job.kind.name, job.job_id, said)
        return {"success": False, "error": said, "status": status}

    @staticmethod
    def _end_process(job: _Job, process: subprocess.Popen) -> None:
        """The stop's escalation. SIGINT (CTRL_BREAK_EVENT on Windows) to the
        job's main process only: its graceful stop. Then SIGTERM and SIGKILL to
        its whole process group (`_terminate`), so a forked worker that shares
        the GPU lock's open file ends with it. Returns once the main process
        has ended or SIGKILL was sent; the monitor ends what is left of the
        group (`_end_group`)."""
        if TrainingManager._interrupt(job, process):
            try:
                process.wait(timeout=STOP_GRACE_SECONDS)
                return
            except subprocess.TimeoutExpired:
                logger.warning("%s job %s did not stop gracefully, terminating...",
                               job.kind.name, job.job_id)
        TrainingManager._terminate(job, process, kill=False)
        job.stop_signal = "SIGTERM"
        try:
            process.wait(timeout=TERMINATE_GRACE_SECONDS)
            return
        except subprocess.TimeoutExpired:
            pass
        TrainingManager._terminate(job, process, kill=True)
        job.stop_signal = "SIGKILL"
        try:
            process.wait(timeout=KILL_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            logger.error("%s job %s did not end after SIGKILL", job.kind.name, job.job_id)

    @staticmethod
    def _interrupt(job: _Job, process: subprocess.Popen) -> bool:
        """The graceful signal to the job's main process: SIGINT, or on Windows
        CTRL_BREAK_EVENT to its console process group. False when it could not
        be sent: a backend with no console cannot send CTRL_BREAK_EVENT
        (`os.kill` raises OSError there), and the stop goes on to terminate."""
        if sys.platform != "win32":
            process.send_signal(signal.SIGINT)
            job.stop_signal = "SIGINT"
            return True
        try:  # pragma: no cover - the backend is untested on Windows
            process.send_signal(signal.CTRL_BREAK_EVENT)
        except OSError as exc:  # pragma: no cover - Windows
            logger.warning("%s job %s: CTRL_BREAK_EVENT could not be sent (%s); "
                           "terminating it", job.kind.name, job.job_id, exc)
            return False
        job.stop_signal = "CTRL_BREAK_EVENT"  # pragma: no cover - Windows
        return True  # pragma: no cover - Windows

    @staticmethod
    def _terminate(job: _Job, process: subprocess.Popen, *, kill: bool) -> None:
        """SIGTERM (SIGKILL with `kill`) to the job's process group; without a
        group (Windows, a stand-in) `terminate()` / `kill()` of its main
        process, which on Windows is TerminateProcess and ends that process
        only."""
        if job.group is None:
            if kill:
                process.kill()
            else:
                process.terminate()
            return
        _signal_group(job.group, signal.SIGKILL if kill else signal.SIGTERM)

    @staticmethod
    def _end_group(job: _Job) -> None:
        """Once the job's main process has been waited for: end what is left of
        its process group, SIGTERM then SIGKILL, and return only when nothing
        of it is left, so the slot and the job's final status mean that the
        whole job has ended, forked workers too (P2-3). A job that ended by
        itself normally leaves none (a DataLoader ends its workers as its
        process exits); one that was killed, or crashed, may. A group that
        SIGKILL cannot end keeps the slot, as a main process would (the
        monitor's `wait` has no limit either)."""
        group = job.group
        if group is None:
            return
        deadline = time.monotonic() + _GROUP_SETTLE_SECONDS
        while _group_left(group):
            if time.monotonic() >= deadline:
                break
            time.sleep(_GROUP_POLL_SECONDS)
        else:
            return
        logger.warning("%s job %s: its main process has ended and processes of its group "
                       "remain (forked workers, say); ending them", job.kind.name, job.job_id)
        for sig, grace in ((signal.SIGTERM, TERMINATE_GRACE_SECONDS),
                           (signal.SIGKILL, KILL_GRACE_SECONDS)):
            if not _signal_group(group, sig):
                return
            deadline = time.monotonic() + grace
            while time.monotonic() < deadline:
                if not _group_left(group):
                    return
                time.sleep(_GROUP_POLL_SECONDS)
        logger.error("%s job %s: processes of its group did not end after SIGKILL; it keeps "
                     "the GPU slot until they do", job.kind.name, job.job_id)
        while _group_left(group):
            time.sleep(0.5)

    def end_active_job(self) -> Optional[Dict[str, Any]]:
        """**Stopping the backend stops its jobs** (the owner's decision of
        2026-10-22): the job holding the GPU slot, whatever its type, stopped
        through `stop` (SIGINT, then SIGTERM, then SIGKILL to its group), or,
        when an operator's stop is already ending it, waited for, bounded by
        the same escalation. `src.api.main` calls it once per shutdown, after
        closing the slot (`close`): when a shutdown signal arrives, or at the
        lifespan's end when no signal began it, so a backend stopped by
        Ctrl-C, a closed terminal, SIGTERM or the service manager ends its
        job, open connections or not. Returns what happened, or None when no
        job held the slot. Blocks; call it from a thread."""
        with self._lock:
            job = self._active
            if job is None:
                return None
            job_type, status = job.kind.name, job.status
        said: Dict[str, Any] = {"job_type": job_type, "job_id": job.job_id}
        if status == "running":
            return {**said, **self.stop(job_type)}
        ended = job.done.wait(timeout=END_ACTIVE_JOB_SECONDS)
        with self._lock:
            status = job.status
        return {**said, "success": ended, "status": status}

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
        """Wait for the job's own process (never another job's), end what is
        left of its process group, read what it left, and record how it ended;
        the GPU slot is free only then."""
        returncode: Optional[int] = None
        outcome: Optional[Dict[str, Any]] = None
        try:
            reader = threading.Thread(target=self._drain, args=(job,), daemon=True,
                                      name=f"{job.kind.name}-output")
            reader.start()
            returncode = job.process.wait()
            # The rest of its process group, before its output is read to the
            # end: a forked worker holds the pipe open as long as it lives.
            self._end_group(job)
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
                _remove(path)
            job.done.set()
            logger.info(f"{job.kind.name.capitalize()} process ended (code={returncode}, "
                        f"status={job.status})")

    @staticmethod
    def _final_status(job: _Job, returncode: Optional[int]) -> str:
        if job.stop_requested:
            # An operator's stop: training is "stopped" whatever the exit code
            # (contract M3a; its snapshot says how the stage stood). A test run
            # is "cancelled" only when the stop can have ended it; one that
            # ended on its own as the cancel arrived keeps that ending
            # (completed, or failed with its failure text): calling it
            # cancelled would not be true, and would hide a failure.
            if job.kind.name != "test" or TrainingManager._ended_by_stop(returncode):
                return job.kind.stopped_status
        return "completed" if returncode == 0 else "failed"

    @staticmethod
    def _ended_by_stop(returncode: Optional[int]) -> bool:
        """Whether a test run's exit status can be a stop's doing.

        On POSIX, only a signal's: measure_scorer turns the stop's SIGINT into
        one "cancelled" line and then ends itself by SIGINT (the signal's
        default action, `_exit_as_program`), as SIGTERM and SIGKILL end it by
        theirs, and a status of 0 or more is the run's own ending. Windows
        has no signal statuses (CTRL_BREAK_EVENT and TerminateProcess end it
        with a positive code), so there any status but 0 is taken as the
        stop's. None (the process was not waited for) is taken as the stop's.
        """
        if returncode is None:
            return True
        if sys.platform == "win32":  # pragma: no cover - the backend is untested on Windows
            return returncode != 0
        return returncode < 0

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

        **The run says what it published** in its outcome file (`--outcome`,
        a new file in its job directory, written as soon as its report is
        published and again at the end): the report and the digest of the
        bytes it published. **Whether that report is kept, and whether it is
        registered,** is then read from the report store and the ledger by the
        registration's one helper (`registration.report_state`, given that
        digest), never from the outcome file or the exit status alone.

        **No outcome file means no report was published**, with one check
        that keeps it true: a run that stopped or failed between publishing
        its report and writing the file leaves a report directory and no
        outcome. So the version's report store is listed too; a run directory
        for this checkpoint that appeared during the run is read as above
        (two or more cannot be told apart, and neither is claimed).
        """
        base: Dict[str, Any] = {"report_dir": None, "report": None, "kept": None,
                                "registered": None, "run_outcome": None}
        path = job.outcome_path
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return self._outcome_without_file(job, base)
        except OSError as exc:
            return {**base, "summary": f"The run's outcome file {path} cannot be read ({exc}), "
                                       "so what it left cannot be checked."}
        try:
            run_outcome = json.loads(raw.decode("utf-8"))
            report, published = run_outcome["report"], run_outcome["report_sha256"]
            if not (isinstance(report, str) and report
                    and isinstance(published, str) and _SHA256.fullmatch(published)):
                raise ValueError("it names no report and published digest")
        except (ValueError, KeyError, TypeError, RecursionError) as exc:
            return {**base, "summary": f"The run's outcome file {path} is not one this runner "
                                       f"reads ({type(exc).__name__}: {exc}), so what it left "
                                       "cannot be checked."}
        return self._report_outcome(job, Path(report), published, run_outcome, base)

    def _outcome_without_file(self, job: _Job, base: Dict[str, Any]) -> Dict[str, Any]:
        """A run that wrote no outcome file: no report was published, unless a
        report directory for its checkpoint appeared in the store meanwhile."""
        reports = job.reports_dir
        sha12 = job.inputs["checkpoint_sha256"][:12]
        try:
            after = _report_names(reports)
        except OSError as exc:
            return {**base, "summary": f"The run wrote no outcome file, and the report store "
                                       f"{reports} cannot be listed ({exc}), so whether it "
                                       "published a report cannot be checked."}
        new = sorted(name for name in after - job.reports_before if f"-{sha12}-" in name)
        if not new:
            return {**base, "kept": False, "registered": False,
                    "summary": f"No report was published: the run wrote no outcome file and "
                               f"created no report directory under {reports}, so it kept and "
                               "registered nothing."}
        if len(new) > 1:
            return {**base, "summary": f"The run wrote no outcome file, and {len(new)} report "
                                       f"directories for this checkpoint appeared during it "
                                       f"({', '.join(new)}); which one is this run's cannot be "
                                       "told, so neither is claimed."}
        from src.evaluation.cohort_registry import REPORT_FILE

        return self._report_outcome(job, reports / new[0] / REPORT_FILE, None, None, base)

    def _report_outcome(self, job: _Job, report: Path, published: Optional[str],
                        run_outcome: Optional[Dict[str, Any]],
                        base: Dict[str, Any]) -> Dict[str, Any]:
        """Kept and registered for the report this run published (`published`,
        from its outcome file) or left without an outcome (`published` None),
        from the registration's helper, and the summary that says so. Both are
        read once, as the run ends, and the summary says them as that moment's
        reading: a report registered later reads so in the Test page's Result,
        not here (the M5 close-out, 2026-10-25)."""
        state = self._report_state(report, checkpoint=Path(job.inputs["checkpoint"]),
                                   expected_sha256=published)
        kept = _STATE_ANSWER.get(state.kept)
        registered = _STATE_ANSWER.get(state.registered)
        run_dir = report.parent
        said = ("" if run_outcome is not None else
                f"The run wrote no outcome file, but a report directory for this checkpoint "
                f"appeared during it: {run_dir}. ")
        if kept is True:
            said += f"When the run ended, a report was kept at {run_dir}; " + (
                f"it was registered (at that time: {state.why_registered})." if registered is True
                else f"it was not registered (at that time: {state.why_registered})."
                if registered is False
                else f"whether it was registered could not be checked (at that time: "
                     f"{state.why_registered}).")
        elif kept is False:
            said += (f"The run published a report at {report}, but when it ended the report was "
                     f"not kept: {state.why_kept}." if run_outcome is not None
                     else f"When the run ended, no report was kept in {run_dir} "
                          f"({state.why_kept}).")
        else:
            said += (f"When the run ended, whether the report at {report} was kept could not be "
                     f"checked ({state.why_kept}).")
        registration = (run_outcome or {}).get("registration")
        why = registration.get("why") if isinstance(registration, dict) else None
        refused_itself = (isinstance(registration, dict) and registration.get("requested") is True
                          and registration.get("registered") is False)
        if registered is True and refused_itself and isinstance(why, str) and why:
            # Repeated runs of one measurement on one machine can publish
            # byte-identical reports, so the record found may be an earlier
            # run's registration of the same bytes while this run's own was
            # refused (M5-3 review EVAL-2): both are said, never the record alone.
            said += (f" This run's own registration was refused: {why}"
                     + ("" if why.endswith(".") else ".")
                     + " The record found names the same report bytes, which an earlier run "
                       "of this measurement registered; it says nothing of this run's refusal.")
        elif registered is not True and isinstance(why, str) and why:
            said += f" The run said of its registration: {why}"
            said += "" if why.endswith(".") else "."
        if run_outcome is not None and run_outcome.get("complete") is False:
            said += (" Its outcome file is the one written when the report was published: the "
                     "run did not get as far as writing it again at its end.")
        if job.stop_signal in ("SIGTERM", "SIGKILL"):
            said += (f" The run was ended by {job.stop_signal}, so staged files (*.tmp) may be "
                     f"left in {run_dir}")
            if job.inputs.get("record"):
                said += (f", and, as it was registering the report, in {_ledger_dir(job)} "
                         "beside the checkpoint's ledger")
            said += "."
        return {**base, "report_dir": str(run_dir), "report": str(report), "kept": kept,
                "registered": registered, "run_outcome": run_outcome, "summary": said}

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

    def get_metrics_history(self, run_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Read full metrics history from MetricsLogger JSON logs.

        Args:
            run_dir: the run whose log is read (M5-2: the Training page's run
                view, a run the server lists); the run started last when None.

        Returns:
            Dict with 'train' and 'validation' metric lists.
        """
        log_file = self.metrics_log_file(run_dir)
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
        return self.metrics_log_file()

    def metrics_log_file(self, run_dir: Optional[Path] = None) -> Optional[Path]:
        """The most recently modified training log file of `run_dir` (the run
        started last when None), or None. Lists the run's ``logs/`` only."""
        from src.utils.checkpoint_paths import run_log_dir

        log_dir = run_log_dir(run_dir) if run_dir is not None else self.log_dir
        if log_dir is None or not log_dir.exists():
            return None

        log_files = sorted(
            log_dir.glob("train_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return log_files[0] if log_files else None

    def latest_snapshot(self, run_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
        """The run's latest saved training state (`last.pt`) as its own record
        describes it, or `None` when the run saved none. The run is `run_dir`
        when given (a stopped job's own), else the run started last.

        Read whole with the safe loader, so only when asked (the stop
        confirmation, a status after the process ended) and cached by the
        file's size and modification time.
        """
        from src.utils.checkpoint_paths import run_checkpoint_dir

        run_dir = run_dir if run_dir is not None else self.run_dir
        if run_dir is None:
            return None
        last = run_checkpoint_dir(run_dir) / "last.pt"
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
                # The dataset the run trained on, as the checkpoint records it
                # (M5-2: the Training page's run view, for operators).
                "dataset_fingerprint": _recorded_dataset(record),
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
        expected_sha256: Optional[str] = None,
    ) -> Dict[str, Any]:
        """What starting from `parent_path` would do, for the console to show
        before start (contract M3a-3, M3a-4).

        `expected_sha256` is the parent's SHA-256 a caller showed (the run
        list's or an earlier Check's model fingerprint): **the one place a
        caller's digest of the parent is compared** (root-cause round,
        P22-S1), by the read itself before the file is loaded
        (`read_checkpoint(expected_sha256=)`); other bytes, or none, are
        refused as changed (``changed: True``, so a caller can add what to do
        for where it showed the fingerprint: the console's "refresh the list"
        or "check it again").

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
        workspace's data and graph (M2.3) and decides again on those bytes;
        `start_training` binds it to this preview's read.
        """
        from src.utils.run_plan import OPERATIONAL_FIELDS, RunPlanRefused, plan_continuation
        from src.utils.checkpoint_io import CheckpointChanged, read_checkpoint

        if expected_sha256 is not None and not (
                isinstance(expected_sha256, str) and _SHA256.fullmatch(expected_sha256)):
            return {"ok": False,
                    "reason": "The parent's model fingerprint sent is not a full SHA-256 "
                              "hexdigest, so the checkpoint cannot be checked."}
        path = Path(parent_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if expected_sha256 is None and not path.is_file():
            return {"ok": False, "parent": str(path), "reason": f"{path} does not exist."}
        try:
            read = read_checkpoint(path, map_location="cpu", weights_only=True,
                                   expected_sha256=expected_sha256)
        except CheckpointChanged as exc:
            now = (f"It holds {exc.found[:12]}... now" if exc.found is not None
                   else "It is gone")
            return {"ok": False, "changed": True, "parent": str(path),
                    "reason": f"The checkpoint changed since it was chosen. {now} "
                              f"(chosen: {exc.expected[:12]}...)."}
        except Exception as exc:  # noqa: BLE001 - a refusal to show
            return {"ok": False, "parent": str(path),
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
            # The dataset the parent trained on, as it records it (M5-2: the
            # Continue drawer compares it with the workspace's current one).
            "dataset_fingerprint": _recorded_dataset(record),
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
