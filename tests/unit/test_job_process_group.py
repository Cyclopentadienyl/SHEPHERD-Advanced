"""
A job's processes end with it (review of 8242b47, P2-3)
=======================================================
The job runner starts each job as the leader of its own process group; a stop
sends SIGINT to the main process only (its graceful path), then SIGTERM and
SIGKILL to the whole group; once the main process has been waited for, the
monitor ends whatever of the group is left, and only then says how the job
ended and frees the GPU slot. A forked DataLoader worker shares the GPU lock's
open file, so a worker left alive kept the lock while the runner said
"stopped" and refused every next job (the reviewer's reproduction:
test_cancel_tree.py).

Everything here goes through the runner's real entry points (`_spawn_locked`,
`stop_training`, `start_training`, the monitor) with real children: a script
whose real torch DataLoader worker (fork) is blocked in `__getitem__`, and the
real `scripts/train_model.py`, whose forked workers a start-up hook holds in
their first sample (`HOLD`). A member of a group counts as left only while it
is not a zombie (`/proc/<pid>/stat`, read by the tests' one reader in
tests/fixtures/job_processes.py): a zombie holds no descriptor. Every child a
test starts is stopped by its pid in `finally`.

Module: tests/unit/test_job_process_group.py
"""
from __future__ import annotations

import ctypes
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("torch")

import src.api.services.training_manager as tm  # noqa: E402
from src.utils.gpu_lock import LOCK_FILE_NAME, WORKSPACES_ROOT_ENV, probe  # noqa: E402
from tests.fixtures.job_processes import (  # noqa: E402
    REPO,
    FakeProcess,
    group_members,
    live,
    process_state,
)

LINUX = pytest.mark.skipif(not sys.platform.startswith("linux"),
                           reason="/proc and fork DataLoader workers: Linux")

#: A job whose real DataLoader worker (fork) is blocked in its sample, so only
#: a signal ends it: the reviewer's reproduction, with the main process's
#: behaviour chosen by argv[2]. "graceful": SIGINT ends it (KeyboardInterrupt).
#: "ignore-int": it ignores SIGINT. "ignore-int-term": it ignores SIGINT and
#: SIGTERM, whatever its workers' deaths raise in it. "crash": it ends by
#: itself (`os._exit`) and leaves its worker behind.
CHILD = r'''import os, signal, sys, threading, time
from pathlib import Path
sys.path.insert(0, os.getcwd())     # the runner starts a job in the project root
from torch.utils.data import DataLoader, Dataset
from src.utils.gpu_lock import take_gpu_lock

mode = sys.argv[2]


class SlowSample(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        Path(sys.argv[1]).write_text(str(os.getpid()))
        time.sleep(600)
        return index


if mode in ("ignore-int", "ignore-int-term"):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
if mode == "ignore-int-term":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
with take_gpu_lock("training", "blocked-loader"):
    batches = iter(DataLoader(SlowSample(), num_workers=1, multiprocessing_context="fork"))
    if mode == "crash":
        threading.Timer(1.5, lambda: os._exit(7)).start()
    if mode == "ignore-int-term":
        threading.Thread(target=lambda: next(batches), daemon=True).start()
        while True:
            try:
                time.sleep(1)
            except BaseException:
                pass
    try:
        next(batches)
    except KeyboardInterrupt:
        sys.exit(0)
'''


@pytest.fixture
def blocked(tmp_path, monkeypatch):
    """The lock (under the test's directory as its workspaces root), the
    child script and short escalation steps."""
    lock = tmp_path / LOCK_FILE_NAME
    monkeypatch.setenv(WORKSPACES_ROOT_ENV, str(tmp_path))
    for name, value in (("STOP_GRACE_SECONDS", 15.0), ("TERMINATE_GRACE_SECONDS", 2.0),
                        ("KILL_GRACE_SECONDS", 2.0), ("OUTPUT_GRACE_SECONDS", 0.5)):
        monkeypatch.setattr(tm, name, value)
    child = tmp_path / "blocked_loader.py"
    child.write_text(CHILD)
    return tmp_path, lock, child


def _start(manager: tm.TrainingManager, child: Path, pidfile: Path, mode: str):
    with manager._lock:
        job, error = manager._spawn_locked(tm.JOB_TYPES["training"],
                                           [sys.executable, str(child), str(pidfile), mode],
                                           inputs={})
    assert job is not None, error
    deadline = time.monotonic() + 60
    while not pidfile.exists() and time.monotonic() < deadline:
        assert job.process.poll() is None, job.tail(30)
        time.sleep(0.05)
    assert pidfile.exists(), job.tail(30)
    time.sleep(0.3)
    return job, int(pidfile.read_text())


def _cleanup(job, worker) -> None:
    """Stop, by pid, whatever a failed test left of its own children."""
    for pid in ([worker] if worker else []) + ([job.process.pid] if job and job.process else []):
        if live(pid):
            os.kill(pid, signal.SIGKILL)
    if job and job.process and job.process.poll() is None:
        job.process.wait(timeout=10)
    if job:
        job.done.wait(30)


# --------------------------------------------------------------------------- the tests' reader
@LINUX
def test_a_process_reaped_while_its_state_is_read_is_gone(monkeypatch):
    """The tests' one reader of a process's state (`process_stat`) counts a
    process reaped between the opening of its /proc entry and the read, which
    then fails with ESRCH, as gone, as it counts one whose entry is absent. A
    copy of the reader that knew only the absent entry failed the probe's
    parent-death test with ProcessLookupError (review of 2887506)."""
    import tests.fixtures.job_processes as job_processes

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])

    def opened_then_reaped(path, *args, **kwargs):
        handle = open(path, *args, **kwargs)
        child.kill()
        child.wait()
        return handle

    monkeypatch.setattr(job_processes, "open", opened_then_reaped, raising=False)
    try:
        assert process_state(child.pid) == "gone"
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    monkeypatch.undo()
    assert process_state(child.pid) == "gone" and not live(child.pid)


# --------------------------------------------------------------------------- a stop
@LINUX
@pytest.mark.parametrize("mode, grace, last", [
    ("graceful", 15.0, "SIGINT"),
    ("ignore-int", 0.5, "SIGTERM"),          # the reviewer's case: SIGINT ignored
    ("ignore-int-term", 0.5, "SIGKILL"),
])
def test_a_stop_ends_the_whole_job_before_it_says_stopped(blocked, monkeypatch, mode, grace,
                                                          last):
    """Graceful, SIGTERM escalation, SIGKILL escalation: the job's own group
    (never the test's), the worker in it; "stopped" only once no process of
    it is left, the lock free and the slot free. It used to answer "stopped"
    with the worker alive and holding the lock (exit -15)."""
    tmp, lock, child = blocked
    monkeypatch.setattr(tm, "STOP_GRACE_SECONDS", grace)
    manager = tm.TrainingManager()
    job = worker = None
    try:
        job, worker = _start(manager, child, tmp / "worker.pid", mode)
        assert job.group == job.process.pid != os.getpgrp()
        assert os.getpgid(worker) == job.group, "the forked worker is in the job's group"
        assert f"pid {job.process.pid}" in (probe(lock) or "")
        answer = manager.stop_training()
        assert answer["success"] is True and answer["status"] == "stopped", answer
        assert job.stop_signal == last
        assert not live(worker), "a forked worker outlived a stop that said stopped"
        assert group_members(job.group) == []
        assert probe(lock) is None, "the GPU lock is still held after the stop"
        assert manager.active_job() is None
    finally:
        _cleanup(job, worker)


@LINUX
def test_a_job_that_ends_by_itself_leaves_no_worker_and_keeps_the_slot_until_then(
        blocked, monkeypatch):
    """A main process that dies by itself (`os._exit`) with its worker still in
    a sample: the monitor ends the worker before it records "failed" and frees
    the slot; a start meanwhile is refused for the slot."""
    tmp, lock, child = blocked
    manager = tm.TrainingManager()
    seen = {}
    real_end_group = tm.TrainingManager._end_group

    def observed(job_):
        seen["active"] = manager.active_job()
        seen["start"] = manager.start_training({"data_dir": str(tmp)})
        real_end_group(job_)

    monkeypatch.setattr(tm.TrainingManager, "_end_group", staticmethod(observed))
    job = worker = None
    try:
        job, worker = _start(manager, child, tmp / "worker.pid", "crash")
        assert job.done.wait(30)
        assert job.status == "failed" and job.exit_code == 7
        assert seen["active"] and seen["active"]["status"] == "running"
        assert seen["start"]["success"] is False
        assert "Training is already running" in seen["start"]["error"]
        assert not live(worker) and probe(lock) is None and manager.active_job() is None
    finally:
        _cleanup(job, worker)


PR_SET_CHILD_SUBREAPER = 36


@LINUX
def test_a_worker_the_backend_adopts_is_reaped_not_waited_for_forever(blocked, monkeypatch):
    """The backend as the reaper of the job's orphans (PID 1 of a container
    without an init; simulated with PR_SET_CHILD_SUBREAPER on this process):
    the killed worker becomes this process's zombie, which answers a group
    signal forever. The monitor reaps it, and the stop ends, bounded."""
    tmp, lock, child = blocked
    monkeypatch.setattr(tm, "STOP_GRACE_SECONDS", 0.5)
    libc = ctypes.CDLL(None, use_errno=True)
    assert libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) == 0
    manager = tm.TrainingManager()
    job = worker = None
    try:
        job, worker = _start(manager, child, tmp / "worker.pid", "ignore-int")
        answer = {}
        stopper = threading.Thread(target=lambda: answer.update(manager.stop_training()),
                                   daemon=True)
        stopper.start()
        stopper.join(60)
        assert not stopper.is_alive(), "the stop never returned: a zombie kept the group"
        assert answer["success"] is True and answer["status"] == "stopped", answer
        assert process_state(worker) == "gone", "the adopted worker was not reaped"
        assert probe(lock) is None
    finally:
        libc.prctl(PR_SET_CHILD_SUBREAPER, 0, 0, 0, 0)
        _cleanup(job, worker)


@LINUX
def test_an_adopted_worker_that_ends_while_its_group_is_checked_is_reaped(blocked,
                                                                          monkeypatch):
    """The backend as the reaper of the job's orphans, and the adopted worker
    ending after the monitor has reaped the group's zombies and before it
    reads which members are left: the read finds only that zombie, so nothing
    of the group is left, and the zombie is reaped before the job ends. It
    used to be left unreaped for good (the verification of the review of
    2887506: the test above failed now and then with state 'Z'). The main
    process ends by itself (`crash`) and leaves its worker alive; the test
    kills the worker at that point of the monitor's check (`_live_member`,
    after the reap and the group signal)."""
    tmp, lock, child = blocked
    libc = ctypes.CDLL(None, use_errno=True)
    assert libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) == 0
    manager = tm.TrainingManager()
    real_live_member = tm._live_member
    ended_in_check = []
    job = worker = None

    def worker_ends_as_checked(group):
        if job is not None and group == job.group and not ended_in_check and live(worker):
            os.kill(worker, signal.SIGKILL)
            deadline = time.monotonic() + 30
            while process_state(worker) != "Z" and time.monotonic() < deadline:
                time.sleep(0.01)
            ended_in_check.append(process_state(worker))
        return real_live_member(group)

    monkeypatch.setattr(tm, "_live_member", worker_ends_as_checked)
    try:
        job, worker = _start(manager, child, tmp / "worker.pid", "crash")
        assert job.done.wait(60), "the job never ended"
        assert job.status == "failed" and job.exit_code == 7
        assert ended_in_check == ["Z"], "the worker did not end during the group check"
        assert process_state(worker) == "gone", "the adopted worker was not reaped"
        assert probe(lock) is None and manager.active_job() is None
    finally:
        libc.prctl(PR_SET_CHILD_SUBREAPER, 0, 0, 0, 0)
        if worker is not None:
            try:
                os.waitpid(worker, os.WNOHANG)  # this process's zombie, if it was left
            except ChildProcessError:
                pass
        _cleanup(job, worker)


#: The critic's experiment (S/cxr-jobs/critic/e1): a reaper that adopts the
#: job's orphans and never reaps them (docker's `sleep infinity` as PID 1, say)
#: runs the backend; the backend stops a job whose worker is blocked in a
#: sample. The worker ends as that reaper's zombie, which a group signal still
#: answers. `OUTER` waits only for its own child, then reaps what it adopted.
OUTER = r'''import ctypes, json, os, subprocess, sys
libc = ctypes.CDLL(None, use_errno=True)
assert libc.prctl(36, 1, 0, 0, 0) == 0
backend = subprocess.Popen([sys.executable] + sys.argv[1:])
os.waitpid(backend.pid, 0)
adopted = []
while True:
    try:
        pid, _ = os.waitpid(-1, os.WNOHANG)
    except ChildProcessError:
        break
    if pid == 0:
        break
    adopted.append(pid)
print(json.dumps({"adopted": adopted}), flush=True)
'''
INNER = r'''import json, os, sys, threading, time
from pathlib import Path
repo, tmp = sys.argv[1], Path(sys.argv[2])
sys.path.insert(0, repo)
os.environ["SHEPHERD_WORKSPACES_ROOT"] = str(tmp)
from src.api.services import training_manager as tm
from src.utils.gpu_lock import probe
from tests.fixtures.job_processes import process_state
tm.STOP_GRACE_SECONDS, tm.TERMINATE_GRACE_SECONDS, tm.KILL_GRACE_SECONDS = 0.5, 2.0, 2.0
tm.OUTPUT_GRACE_SECONDS, tm.RESULT_WAIT_SECONDS = 0.5, 30.0
manager = tm.TrainingManager()
with manager._lock:
    job, error = manager._spawn_locked(tm.JOB_TYPES["training"], [sys.executable,
        sys.argv[3], str(tmp / "worker.pid"), "ignore-int"], inputs={})
deadline = time.monotonic() + 60
while not (tmp / "worker.pid").exists() and time.monotonic() < deadline:
    time.sleep(0.05)
time.sleep(0.3)
worker = int((tmp / "worker.pid").read_text())
started = time.monotonic()
answer = {}
stopper = threading.Thread(target=lambda: answer.update(manager.stop_training()), daemon=True)
stopper.start()
stopper.join(60)
with manager._lock:
    refusal = manager._slot_refusal_locked(tm.JOB_TYPES["training"])
print(json.dumps({"worker": worker, "returned": not stopper.is_alive(),
                  "seconds": time.monotonic() - started, "answer": answer,
                  "worker_state": process_state(worker),
                  "lock": probe(tmp / ".shepherd-gpu.lock"),
                  "active": manager.active_job(), "slot_refusal": refusal},
                 default=str), flush=True)
os._exit(0)
'''


@LINUX
def test_a_zombie_left_with_a_reaper_that_never_reaps_does_not_keep_the_slot(tmp_path):
    """The critic's break (review design, P2-3): counting a group as alive
    while `killpg(group, 0)` succeeds kept the slot forever when the orphaned
    worker became the zombie of a reaper that never reaps: "stopping" after
    60 s, every start refused. On Linux only a member that is not a zombie
    counts, so the stop ends as stopped, the lock and the slot free."""
    (tmp_path / "outer.py").write_text(OUTER)
    (tmp_path / "inner.py").write_text(INNER)
    (tmp_path / "child.py").write_text(CHILD)
    done = subprocess.run(
        [sys.executable, str(tmp_path / "outer.py"), str(tmp_path / "inner.py"), str(REPO),
         str(tmp_path), str(tmp_path / "child.py")],
        cwd=str(REPO), capture_output=True, text=True, timeout=240)
    lines = [json.loads(line) for line in done.stdout.splitlines() if line.startswith("{")]
    assert len(lines) == 2, done.stdout + done.stderr[-3000:]
    seen, outer = lines
    assert seen["returned"] and seen["seconds"] < 30, seen
    assert seen["answer"].get("success") is True and seen["answer"]["status"] == "stopped", seen
    assert seen["worker_state"] in ("Z", "gone"), seen
    assert seen["lock"] is None and seen["active"] is None
    assert seen["slot_refusal"] is None, "the slot or the lock is still taken"
    if seen["worker_state"] == "Z":
        assert seen["worker"] in outer["adopted"], "the zombie was the reaper's"


# --------------------------------------------------------------------------- never a group signal by mistake
@pytest.mark.parametrize("pid", [0, 1, 2, os.getpid()],
                         ids=["0", "1", "2", "own"])  # one id in every xdist worker
def test_a_stand_in_process_is_never_signalled_as_a_group(monkeypatch, tmp_path, pid):
    """`killpg(1)` is `kill(-1)`, every process this account may signal, and
    `killpg(0)` the backend's own group. A stand-in (a test's FakeProcess) has
    no group, whatever its pid, and is ended through its own methods."""
    monkeypatch.setattr(tm, "STOP_GRACE_SECONDS", 0.1)
    monkeypatch.setattr(tm, "TERMINATE_GRACE_SECONDS", 0.1)
    monkeypatch.setattr(tm.os, "killpg", lambda *a: pytest.fail(f"killpg{a} was called"))
    monkeypatch.setattr(tm.os, "waitpid", lambda *a: pytest.fail(f"waitpid{a} was called"))
    fake = FakeProcess(pid=pid, ignore_sigint=True, ignore_terminate=True)
    manager = tm.TrainingManager()
    with patch("subprocess.Popen", return_value=fake):
        assert manager.start_training({"data_dir": str(tmp_path)})["success"]
    assert manager._last["training"].group is None
    answer = manager.stop_training()
    assert fake.signals == ["SIGINT", "SIGTERM", "SIGKILL"] and answer["status"] == "stopped"


def test_only_a_started_process_that_leads_its_group_has_one(monkeypatch):
    """Pid 0 or 1, even for a `Popen`, gets no group; neither does a process
    whose group is not its own pid."""
    class Started(tm._POPEN):
        def __init__(self, pid):  # no process is started
            self.pid = pid

    assert tm._job_group(Started(0)) is None and tm._job_group(Started(1)) is None
    monkeypatch.setattr(tm.os, "getpgid", lambda pid: pid)
    assert tm._job_group(Started(4242)) == 4242
    monkeypatch.setattr(tm.os, "getpgid", lambda pid: os.getpgrp())
    assert tm._job_group(Started(4242)) is None
    assert tm._job_group(FakeProcess(pid=4242)) is None


@LINUX
def test_a_real_job_leads_its_own_group_and_a_test_run_is_never_reached(blocked):
    """A real job is started with `process_group=0`: it leads a group that is
    not the backend's; a training stop never reaches a test run (the other
    job type) and leaves it running."""
    tmp, lock, child = blocked
    manager = tm.TrainingManager()
    with manager._lock:
        job, error = manager._spawn_locked(tm.JOB_TYPES["test"],
                                           [sys.executable, "-c", "import time; time.sleep(60)"],
                                           inputs={"checkpoint_sha256": "c" * 64})
    assert job is not None, error
    try:
        assert job.group == job.process.pid and os.getpgid(job.process.pid) == job.group
        assert job.group != os.getpgrp()
        refused = manager.stop_training()
        assert refused["success"] is False and "does not reach" in refused["error"]
        assert job.process.poll() is None, "the training stop reached the test run"
    finally:
        if job.process.poll() is None:
            job.process.kill()
            job.process.wait(timeout=10)
        job.done.wait(30)


# --------------------------------------------------------------------------- the real trainer
#: Loaded at start-up by the real trainer's job (`sitecustomize`, from the
#: PYTHONPATH the job inherits from the runner): each DataLoader worker the
#: trainer forks writes its pid into $SHEPHERD_TEST_HOLD_DIR and holds in its
#: first sample, so it cannot end by itself before the stop. Left alone, a tiny
#: cohort's epoch ends its workers within moments, between two scans of /proc
#: (review of 2887506). With $SHEPHERD_TEST_IGNORE_SIGINT the trainer ignores
#: SIGINT, so the stop must escalate however soon the trainer would have
#: ended. Nothing else changes: the trainer, its DataLoader and the workers it
#: forks are the real ones.
HOLD = r'''import os, signal, sys, time
from pathlib import Path

held = os.environ["SHEPHERD_TEST_HOLD_DIR"]


def hold_in_first_sample():
    loader = sys.modules.get("src.kg.data_loader")
    if loader is None:          # a fork before the trainer's DataLoader exists
        return

    def held_sample(self, index):
        Path(held, str(os.getpid())).touch()
        time.sleep(600)

    loader.DiagnosisDataset.__getitem__ = held_sample


os.register_at_fork(after_in_child=hold_in_first_sample)
if os.environ.get("SHEPHERD_TEST_IGNORE_SIGINT"):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
'''


@LINUX
@pytest.mark.parametrize("sigint, stop_grace, last", [
    ("handled", 60.0, "SIGINT"),
    ("ignored", 0.5, "SIGTERM"),
], ids=["graceful", "escalated"])
def test_the_real_training_cli_with_forked_workers_leaves_nothing_after_a_stop(
        tmp_path, monkeypatch, sigint, stop_grace, last):
    """scripts/train_model.py through `start_training`, num_workers=2 (fork),
    both workers held alive in a sample (`HOLD`) when the stop comes: a
    graceful stop (SIGINT) and, the trainer ignoring SIGINT, one escalated to
    SIGTERM to the whole group both end every process of the job and free the
    lock before the stop answers. The workers' pids are their own record, not
    a scan of /proc, and held they cannot end by themselves, so how fast the
    machine is decides neither whether the workers are there nor which signal
    ends the job."""
    from tests.fixtures.trainable_workspace import write_trainable_workspace

    data = write_trainable_workspace(tmp_path / "case")
    lock = tmp_path / LOCK_FILE_NAME
    hook, held = tmp_path / "hook", tmp_path / "held"
    hook.mkdir()
    held.mkdir()
    (hook / "sitecustomize.py").write_text(HOLD)
    monkeypatch.setenv("PYTHONPATH", str(hook), prepend=os.pathsep)
    monkeypatch.setenv("SHEPHERD_TEST_HOLD_DIR", str(held))
    if sigint == "ignored":
        monkeypatch.setenv("SHEPHERD_TEST_IGNORE_SIGINT", "1")
    monkeypatch.setenv(WORKSPACES_ROOT_ENV, str(tmp_path))
    monkeypatch.setattr(tm, "STOP_GRACE_SECONDS", stop_grace)
    manager = tm.TrainingManager()
    started = manager.start_training(dict(
        data_dir=str(data), run_dir=str(tmp_path / "run"), device="cpu", num_epochs=100000,
        early_stopping_patience=100000, hidden_dim=16, num_heads=2, num_layers=1,
        conv_type="gat", batch_size=2, num_workers=2, use_amp=False))
    assert started["success"], started
    job = manager._last["training"]
    try:
        deadline = time.monotonic() + 180
        while len(list(held.iterdir())) < 2 and time.monotonic() < deadline:
            assert job.process.poll() is None, job.tail(30)
            time.sleep(0.2)
        workers = sorted(int(entry.name) for entry in held.iterdir())
        assert len(workers) == 2, job.tail(30)
        assert set(workers) <= set(group_members(job.group)), "a held worker is not in the group"
        assert f"pid {job.process.pid}" in (probe(lock) or "")
        answer = manager.stop_training()
        assert answer["success"] is True and answer["status"] == "stopped", answer
        assert job.stop_signal == last
        assert not any(live(pid) for pid in workers), "a forked worker outlived the stop"
        assert group_members(job.group) == [] and probe(lock) is None
        assert manager.active_job() is None
    finally:
        for pid in group_members(job.group or -1):
            os.kill(pid, signal.SIGKILL)
        job.done.wait(30)
