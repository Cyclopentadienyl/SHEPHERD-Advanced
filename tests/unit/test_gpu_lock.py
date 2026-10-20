"""
The GPU lock (M5): one per deployment, held for a job's whole working life.
==========================================================================
Where it is, who holds it, that a busy lock is refused naming its holder, and
that it is released however the holder ends: a normal exit, SIGTERM, SIGKILL.
A program the holder starts does not keep it (close-on-exec). A process forked
without exec shares it: a holder that ends normally releases it for that child
too, and one that exits without releasing leaves it with the child until the
child exits, the message then never naming the dead holder as the holder.
The CLIs take it at the start of `main`, before anything is read, and refuse
with "Nothing was written."

Module: tests/unit/test_gpu_lock.py
"""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from src.utils import gpu_lock
from src.utils.gpu_lock import (
    LOCK_FILE_ENV,
    LOCK_FILE_NAME,
    WORKSPACES_ROOT_ENV,
    GpuLock,
    GpuLockBusy,
    GpuLockMisconfigured,
    holder_path,
    lock_location,
    lock_state,
    probe,
    take_gpu_lock,
)
from tests.fixtures.job_processes import REPO, start_lock_holder

POSIX = pytest.mark.skipif(sys.platform == "win32", reason="signals and fork are POSIX")


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A workspace root named by the setting; no explicit lock file."""
    root = tmp_path / "workspaces"
    root.mkdir()
    monkeypatch.delenv(LOCK_FILE_ENV, raising=False)
    monkeypatch.setenv(WORKSPACES_ROOT_ENV, str(root))
    return root


def _wait_free(path: Path, seconds: float = 10.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if probe(path) is None:
            return True
        time.sleep(0.05)
    return False


# --------------------------------------------------------------------------- location
def test_the_lock_is_under_the_workspace_root_whatever_the_workspace(root):
    assert lock_location() == (root / LOCK_FILE_NAME, f"under {WORKSPACES_ROOT_ENV}")


def test_an_explicit_lock_file_wins_over_the_root(root, tmp_path, monkeypatch):
    monkeypatch.setenv(LOCK_FILE_ENV, str(tmp_path / "gpu.lock"))
    assert lock_location().path == tmp_path / "gpu.lock"


def test_neither_setting_means_not_configured_with_the_reason(monkeypatch):
    monkeypatch.delenv(LOCK_FILE_ENV, raising=False)
    monkeypatch.delenv(WORKSPACES_ROOT_ENV, raising=False)
    location = lock_location()
    assert location.path is None
    assert LOCK_FILE_ENV in location.source and WORKSPACES_ROOT_ENV in location.source


@pytest.mark.parametrize("setting, value, says", [
    (LOCK_FILE_ENV, "relative/gpu.lock", "not an absolute path"),
    (LOCK_FILE_ENV, "/no/such/dir/gpu.lock", "does not exist"),
    (WORKSPACES_ROOT_ENV, "relative", "not an absolute path"),
    (WORKSPACES_ROOT_ENV, "/no/such/root", "does not exist"),
])
def test_a_setting_that_cannot_name_a_lock_is_refused_never_defaulted(
        monkeypatch, setting, value, says):
    monkeypatch.delenv(LOCK_FILE_ENV, raising=False)
    monkeypatch.delenv(WORKSPACES_ROOT_ENV, raising=False)
    monkeypatch.setenv(setting, value)
    with pytest.raises(GpuLockMisconfigured, match=says):
        lock_location()
    assert lock_state()["problem"] and not lock_state()["configured"]


def _unsearchable(monkeypatch, below: Path) -> None:
    """`Path.is_dir` as Python 3.12 answers for a path under a directory this
    account may not search: it raises (EACCES) instead of answering. Simulated,
    since the tests may run as root, which the real check does not stop."""
    import errno

    real = Path.is_dir

    def is_dir(self, *args, **kwargs):
        if str(self).startswith(str(below)):
            raise PermissionError(errno.EACCES, "Permission denied", str(self))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "is_dir", is_dir)


@pytest.mark.parametrize("setting", [LOCK_FILE_ENV, WORKSPACES_ROOT_ENV])
def test_a_location_that_cannot_be_checked_is_refused_with_its_reason(
        monkeypatch, tmp_path, setting):
    """Status, start and the CLIs refuse with the reason; nothing raises
    `PermissionError` out of a status poll."""
    monkeypatch.delenv(LOCK_FILE_ENV, raising=False)
    monkeypatch.delenv(WORKSPACES_ROOT_ENV, raising=False)
    locked = tmp_path / "locked"
    monkeypatch.setenv(setting, str(locked / "sub" / ("gpu.lock" if setting == LOCK_FILE_ENV
                                                       else "")))
    _unsearchable(monkeypatch, locked)
    with pytest.raises(GpuLockMisconfigured, match=r"cannot be checked \(.*Permission denied"):
        lock_location()
    state = lock_state()
    assert state["configured"] is False and "cannot be checked" in state["problem"]
    with pytest.raises(GpuLockMisconfigured, match="cannot be checked"):
        take_gpu_lock("test", "measure_scorer.py")


def test_a_lock_state_never_raises_an_os_error_from_the_location(monkeypatch):
    def unreadable(environ=None):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(gpu_lock, "lock_location", unreadable)
    assert "cannot be checked" in lock_state()["problem"]


def test_a_directory_is_not_a_lock_file(monkeypatch, tmp_path):
    monkeypatch.setenv(LOCK_FILE_ENV, str(tmp_path))
    with pytest.raises(GpuLockMisconfigured, match="is a directory"):
        lock_location()


# --------------------------------------------------------------------------- in one process
def test_a_held_lock_refuses_a_second_taker_naming_the_holder(root):
    path = root / LOCK_FILE_NAME
    with GpuLock(path, job_type="training", command="train_model.py").acquire() as held:
        assert held.held
        with pytest.raises(GpuLockBusy) as refused:
            GpuLock(path, job_type="test", command="measure_scorer.py").acquire()
        message = str(refused.value)
        assert "is held by a training job (train_model.py" in message
        assert f"pid {os.getpid()}" in message
        assert lock_state()["held_by"].startswith("a training job (train_model.py")
    assert probe(path) is None
    GpuLock(path, job_type="test", command="measure_scorer.py").acquire().release()


def test_the_holder_describes_itself_while_it_holds_and_removes_that_on_release(root):
    path = root / LOCK_FILE_NAME
    lock = GpuLock(path, job_type="test", command="measure_scorer.py").acquire()
    record = json.loads(holder_path(path).read_text())
    assert record["job_type"] == "test" and record["command"] == "measure_scorer.py"
    assert record["pid"] == os.getpid() and record["started_at"].endswith("+00:00")
    assert "Descriptive only" in record["note"]
    lock.release()
    assert not holder_path(path).exists()
    lock.release()  # twice is harmless


def test_the_descriptor_is_close_on_exec(root):
    lock = GpuLock(root / LOCK_FILE_NAME, job_type="training", command="x").acquire()
    try:
        assert os.get_inheritable(lock._fd) is False
    finally:
        lock.release()


def test_an_undescribed_holder_gets_a_grace_then_a_refusal_saying_so(root):
    path = root / LOCK_FILE_NAME
    with GpuLock(path, job_type="training", command="x").acquire():
        holder_path(path).unlink()
        started = time.monotonic()
        with pytest.raises(GpuLockBusy, match="has not described itself"):
            GpuLock(path, job_type="test", command="y").acquire(grace=0.3)
        assert time.monotonic() - started >= 0.3


def test_a_busy_lock_is_refused_after_the_grace_never_waited_for(root):
    path = root / LOCK_FILE_NAME
    with GpuLock(path, job_type="training", command="x").acquire():
        started = time.monotonic()
        with pytest.raises(GpuLockBusy, match="is held by a training job"):
            GpuLock(path, job_type="test", command="y").acquire(grace=0.3)
        assert 0.3 <= time.monotonic() - started < 2.0


def test_a_holder_that_lets_go_within_the_grace_is_stepped_over(root):
    """The runner's instant check, or a job's workers just exiting, does not
    refuse a taker."""
    import threading

    path = root / LOCK_FILE_NAME
    brief = GpuLock(path, job_type="training", command="x").acquire()
    threading.Timer(0.2, brief.release).start()
    GpuLock(path, job_type="test", command="y").acquire(grace=2.0).release()


@POSIX
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_a_forked_child_never_releases_its_parents_lock(root):
    """A forked child shares the open file: an unlock there would free the
    holder's lock. `release` in the child only closes its own descriptor."""
    path = root / LOCK_FILE_NAME
    lock = GpuLock(path, job_type="training", command="x").acquire()
    try:
        pid = os.fork()
        if pid == 0:
            lock.release()
            os._exit(0)
        os.waitpid(pid, 0)
        assert probe(path) is not None, "still held by the parent"
        assert holder_path(path).exists(), "and still described"
    finally:
        lock.release()
    assert probe(path) is None


def test_the_lock_is_the_flock_never_the_description(root):
    """A description left behind (a killed holder's) neither blocks nor names
    anyone while the lock is free; the next holder replaces it."""
    path = root / LOCK_FILE_NAME
    holder_path(path).write_text(json.dumps({"job_type": "training", "pid": 1}))
    assert probe(path) is None
    with GpuLock(path, job_type="test", command="measure_scorer.py").acquire():
        assert json.loads(holder_path(path).read_text())["job_type"] == "test"


def _dead_pid() -> int:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait(10)
    return process.pid


def _dead_description(path: Path) -> None:
    holder_path(path).write_text(json.dumps({
        "job_type": "training", "command": "hold.py", "pid": _dead_pid(),
        "host": gpu_lock.socket.gethostname(), "started_at": "2026-10-20T09:00:00+00:00"}))


@POSIX
def test_a_new_holder_removes_a_killed_holders_description_even_if_it_cannot_write_its_own(
        root, monkeypatch):
    """A refusal never names the killed holder, or claims an inheritance, while
    a new holder that could not describe itself holds the lock."""
    path = root / LOCK_FILE_NAME
    _dead_description(path)

    def unwritable(*args, **kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(gpu_lock, "publish_bytes", unwritable)
    with GpuLock(path, job_type="test", command="measure_scorer.py").acquire():
        assert not holder_path(path).exists()
        with pytest.raises(GpuLockBusy) as refused:
            GpuLock(path, job_type="test", command="y").acquire(grace=0.1)
    message = str(refused.value)
    assert "a process that has not described itself" in message
    assert "could not write its description" in message
    assert "hold.py" not in message and "training job" not in message


@POSIX
def test_a_description_whose_process_has_exited_never_names_the_holder(root):
    """A killed holder's description the new holder could not remove (an
    account that may not write the directory): the words say the recorded
    holder exited and the lock is held by another process, of either kind."""
    path = root / LOCK_FILE_NAME
    with GpuLock(path, job_type="test", command="measure_scorer.py").acquire():
        _dead_description(path)
        held_by = probe(path)
    assert held_by.startswith("a process other than the one its description names. "
                              "The description names a training job (hold.py, pid ")
    assert held_by.endswith(", which has exited; the lock is held by a process that "
                            "inherited its descriptor, or by a new holder that has not "
                            "described itself (yet, or because it could not)")


def test_probe_creates_nothing(root):
    assert probe(root / LOCK_FILE_NAME) is None
    assert list(root.iterdir()) == []
    assert lock_state() == {"configured": True, "path": str(root / LOCK_FILE_NAME),
                            "source": f"under {WORKSPACES_ROOT_ENV}", "held_by": None,
                            "problem": None}


def test_not_configured_runs_unlocked_and_says_so(monkeypatch, caplog):
    monkeypatch.delenv(LOCK_FILE_ENV, raising=False)
    monkeypatch.delenv(WORKSPACES_ROOT_ENV, raising=False)
    with caplog.at_level(logging.WARNING):
        with take_gpu_lock("training", "train_model.py"):
            pass
    assert "not configured" in caplog.text
    assert "does not exclude other managed GPU jobs" in caplog.text


def test_windows_never_probes_a_pid(monkeypatch):
    """On Windows `os.kill(pid, 0)` terminates the process: the liveness hint
    is not given there."""
    monkeypatch.setattr(gpu_lock.sys, "platform", "win32")
    monkeypatch.setattr(gpu_lock.os, "kill", lambda *a: pytest.fail("os.kill called"))
    assert gpu_lock._holder_alive({"pid": os.getpid(), "host": gpu_lock.socket.gethostname()}) \
        is None


# --------------------------------------------------------------------------- across processes
@POSIX
def test_released_when_the_holder_exits_normally(root, tmp_path):
    holder = start_lock_holder("exit", {}, tmp_path)
    assert holder.wait(10) == 0
    assert probe(root / LOCK_FILE_NAME) is None
    assert not holder_path(root / LOCK_FILE_NAME).exists(), "a normal exit removes its description"


@POSIX
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGKILL])
def test_released_when_the_holder_is_killed(root, tmp_path, sig):
    path = root / LOCK_FILE_NAME
    holder = start_lock_holder("sleep", {}, tmp_path)
    try:
        assert f"pid {holder.pid}" in probe(path)
        holder.send_signal(sig)
        assert holder.wait(10) == -sig
        assert _wait_free(path), "the kernel releases the flock with the process"
        # The description is left behind by a killed holder; it decides nothing.
        GpuLock(path, job_type="test", command="y").acquire().release()
    finally:
        holder.kill()


@POSIX
def test_a_program_the_holder_starts_does_not_keep_the_lock(root, tmp_path):
    """close_fds=False: only close-on-exec keeps the descriptor from the child."""
    path = root / LOCK_FILE_NAME
    holder = start_lock_holder("spawn", {}, tmp_path)
    child = int(holder.stdout.readline().split()[1])
    try:
        assert holder.wait(10) == 0
        os.kill(child, 0)  # the grandchild still runs ...
        assert _wait_free(path), "... and does not hold the lock"
    finally:
        os.kill(child, signal.SIGKILL)


@POSIX
def test_a_child_forked_without_exec_keeps_the_lock_when_the_holder_does_not_release(
        root, tmp_path):
    """The stated limit: a forked child shares the open file, and so the lock,
    after a holder that exited without releasing (`os._exit`; SIGKILL alike);
    the message says the recorded holder exited, never that it holds."""
    path = root / LOCK_FILE_NAME
    holder = start_lock_holder("fork", {}, tmp_path)
    child = int(holder.stdout.readline().split()[1])
    try:
        assert holder.wait(10) == 0
        time.sleep(0.2)
        held_by = probe(path)
        assert held_by is not None, "the forked child inherited the lock"
        assert held_by.startswith("a process other than the one its description names")
        assert f"pid {holder.pid}" in held_by and "inherited its descriptor" in held_by
        with pytest.raises(GpuLockBusy, match="inherited its descriptor"):
            GpuLock(path, job_type="test", command="y").acquire(grace=0.2)
    finally:
        os.kill(child, signal.SIGKILL)
    assert _wait_free(path), "released once the last descriptor closes"


@POSIX
def test_a_holder_that_ends_normally_releases_the_lock_for_its_forked_child(root, tmp_path):
    """`release` unlocks the open file the child shares, so a child still
    running after a normal end does not keep the lock."""
    path = root / LOCK_FILE_NAME
    holder = start_lock_holder("fork-release", {}, tmp_path)
    child = int(holder.stdout.readline().split()[1])
    try:
        assert holder.wait(10) == 0
        os.kill(child, 0)  # the forked child still runs ...
        assert probe(path) is None, "... and does not hold the lock"
        assert not holder_path(path).exists()
    finally:
        os.kill(child, signal.SIGKILL)


@POSIX
def test_two_workspaces_share_one_lock(root, tmp_path):
    """Whatever workspace a job uses, the lock is the root's."""
    holder = start_lock_holder("sleep", {}, tmp_path)
    try:
        for workspace in ("ws_a", "ws_b"):
            (root / workspace).mkdir()
            with pytest.raises(GpuLockBusy):
                take_gpu_lock("test", f"measure_scorer.py --data-dir {root / workspace}")
    finally:
        holder.kill()
        holder.wait(10)


# --------------------------------------------------------------------------- the suite
def test_a_test_never_sees_the_deployments_lock_settings():
    """What tests/conftest.py guarantees; the next test sets them in the
    environment as an operator's shell that loaded the service's would."""
    assert LOCK_FILE_ENV not in os.environ and WORKSPACES_ROOT_ENV not in os.environ


def test_the_lock_settings_of_the_shell_never_reach_the_suite(tmp_path):
    env = {**os.environ, LOCK_FILE_ENV: str(tmp_path / "gpu.lock"),
           WORKSPACES_ROOT_ENV: str(tmp_path)}
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=",
         f"{__file__}::test_a_test_never_sees_the_deployments_lock_settings"],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr
    assert list(tmp_path.iterdir()) == [], "no lock file in the shell's root"


# --------------------------------------------------------------------------- the CLIs
def _measure_argv(root: Path, workspace: str = "ws_b"):
    return ["--checkpoint", str(root / workspace / "best.pt"), "--data-dir",
            str(root / workspace), "--cohort", "c1", "--registry", str(root / "registry")]


def test_measure_scorer_refuses_a_busy_lock_before_anything_is_read(root, capsys):
    """Before the device is resolved (no CUDA here, so `auto` would refuse with
    another message) and before any input is read (none of them exist)."""
    import scripts.measure_scorer as ms

    with GpuLock(root / LOCK_FILE_NAME, job_type="training", command="train_model.py").acquire():
        status = ms._run_as_program(_measure_argv(root))
    err = capsys.readouterr().err
    assert status == 1
    assert err.startswith("measure_scorer: refused: the GPU lock ")
    assert "is held by a training job (train_model.py" in err
    assert err.rstrip().endswith("Nothing was written.")
    assert [p.name for p in root.iterdir()] == [LOCK_FILE_NAME], "nothing else was written"


def test_measure_scorer_holds_the_lock_for_its_run_and_releases_it(root, monkeypatch):
    import scripts.measure_scorer as ms

    seen = {}

    def run(args):
        seen["held_by"] = probe(root / LOCK_FILE_NAME)
        seen["record"] = json.loads(holder_path(root / LOCK_FILE_NAME).read_text())
        raise RuntimeError("the run failed")

    monkeypatch.setattr(ms, "_measure", run)
    with pytest.raises(RuntimeError):
        ms.main(_measure_argv(root))
    assert seen["held_by"] and seen["record"]["job_type"] == "test"
    assert seen["record"]["command"] == "measure_scorer.py"
    assert probe(root / LOCK_FILE_NAME) is None, "released when main ends, by an exception too"


def test_measure_scorer_refuses_an_unusable_setting(root, monkeypatch, capsys):
    import scripts.measure_scorer as ms

    monkeypatch.setenv(LOCK_FILE_ENV, "gpu.lock")
    assert ms._run_as_program(_measure_argv(root)) == 1
    assert "is not an absolute path" in capsys.readouterr().err


def test_train_model_refuses_a_busy_lock_before_anything_is_read(root, monkeypatch, caplog):
    import scripts.train_model as tm

    monkeypatch.setattr(sys, "argv", ["train_model.py", "--data-dir", str(root / "ws_a")])
    monkeypatch.setattr(tm, "_train_main", lambda args: pytest.fail("ran under a busy lock"))
    with GpuLock(root / LOCK_FILE_NAME, job_type="test", command="measure_scorer.py").acquire():
        with caplog.at_level(logging.ERROR):
            assert tm.main() == 1
    assert "train_model: refused: the GPU lock" in caplog.text
    assert "is held by a test job (measure_scorer.py" in caplog.text
    assert "Nothing was written." in caplog.text


def test_a_lock_file_that_cannot_be_opened_refuses_both_clis(tmp_path, monkeypatch, caplog,
                                                              capsys):
    import scripts.measure_scorer as ms
    import scripts.train_model as tm

    dangling = tmp_path / "gpu.lock"
    dangling.symlink_to(tmp_path / "missing" / "gpu.lock")
    monkeypatch.setenv(LOCK_FILE_ENV, str(dangling))
    monkeypatch.setattr(sys, "argv", ["train_model.py", "--data-dir", str(tmp_path / "ws")])
    monkeypatch.setattr(tm, "_train_main", lambda args: pytest.fail("ran without the lock"))
    with caplog.at_level(logging.ERROR):
        assert tm.main() == 1
    assert f"the GPU lock {dangling} cannot be opened" in caplog.text
    assert ms._run_as_program(_measure_argv(tmp_path)) == 1
    err = capsys.readouterr().err
    assert f"refused: the GPU lock {dangling} cannot be opened" in err
    assert "Nothing was written." in err


def test_train_model_holds_the_lock_for_its_run_and_releases_it(root, monkeypatch):
    import scripts.train_model as tm

    seen = {}

    def run(args):
        seen["record"] = json.loads(holder_path(root / LOCK_FILE_NAME).read_text())
        seen["held_by"] = probe(root / LOCK_FILE_NAME)
        return 0

    monkeypatch.setattr(sys, "argv", ["train_model.py", "--data-dir", str(root / "ws_a")])
    monkeypatch.setattr(tm, "_train_main", run)
    assert tm.main() == 0
    assert seen["held_by"] and seen["record"]["job_type"] == "training"
    assert seen["record"]["command"] == "train_model.py"
    assert probe(root / LOCK_FILE_NAME) is None


@POSIX
def test_train_model_as_a_program_refuses_naming_the_holder(root, tmp_path):
    """The real program, as the runner and an operator start it: exit 1, the
    holder named on stderr, and no run directory."""
    workspace = root / "ws_a"
    workspace.mkdir()
    holder = start_lock_holder("sleep", {}, tmp_path)
    try:
        done = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "train_model.py"), "--data-dir",
             str(workspace)], capture_output=True, text=True, timeout=300, cwd=REPO,
            env={**os.environ, WORKSPACES_ROOT_ENV: str(root)})
    finally:
        holder.kill()
        holder.wait(10)
    assert done.returncode == 1
    assert f"is held by a training job (lock_holder.py, pid {holder.pid}" in done.stderr
    assert "Nothing was written." in done.stderr
    assert list(workspace.iterdir()) == []
