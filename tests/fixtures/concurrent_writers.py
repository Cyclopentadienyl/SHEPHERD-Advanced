"""
Real separate processes for the lock tests (K8-6r.1)
====================================================
The ledger's lock and the report store's locks exclude writers across
processes, so their acceptance runs writers in separate interpreters: each
target below runs in a process of its own, started with the `spawn` method
(`sys.executable`, the repository on its path), and talks to the test only
through multiprocessing events, barriers and a queue.

- `register` registers one report into the ledger beside a checkpoint
  directory, the way ``record_evaluation.py --checkpoint-dir`` does, and can be
  held at a barrier right after `write_ledger`'s guard reads the ledger's digest
  (the window between the check and the replace);
- `hold_lock` holds one lock file until it is told to let go;
- `append_test_run` and `append_registration` append one usage-history entry;
- `failing_append` appends a ``test_run`` entry whose write fails (ENOSPC), and
  whose write and undo each wait for other writers first, so that without a
  lock its undo would cut their lines;
- `measure_generated` (K8-6r.4) runs `scripts/measure_scorer.py`'s `main` on a
  generated split, held at a file-marker barrier once it has passed `main`'s
  check of its `--output`, and again right before it publishes, so two of them
  publish to one `--output` at once.

**Every wait is bounded** (`WAIT`, `BARRIER_WAIT`): a hang fails the test that
waits, never the suite. `processes` stops, by its pid, any process still alive
when the test ends.

Module: tests/fixtures/concurrent_writers.py
"""
from __future__ import annotations

import errno
import multiprocessing
import os
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

#: The longest any process or the test waits for another (a hang fails).
WAIT = 60.0
#: How long a writer held at a barrier waits for the others before it goes on
#: alone: with the lock, the others cannot arrive while it holds it.
BARRIER_WAIT = 1.0
#: When a registration says it was made.
WHEN = "2026-11-07T12:00:00Z"

CONTEXT = multiprocessing.get_context("spawn")


@contextmanager
def processes(*specs: tuple[Any, tuple]) -> Iterator[list[Any]]:
    """Start each ``(target, args)`` in a process of its own; on leaving, wait
    a little for each and stop, by its pid, any still alive."""
    started: list[Any] = []
    try:
        for target, args in specs:
            process = CONTEXT.Process(target=target, args=args, daemon=True)
            process.start()
            started.append(process)
        yield started
    finally:
        for process in started:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=10)


def answers_from(queue: Any, n: int) -> list[dict]:
    """The first ``n`` answers put on ``queue``, each within `WAIT`."""
    return [queue.get(timeout=WAIT) for _ in range(n)]


def _wait(barrier: Any) -> None:
    """At a barrier for at most `BARRIER_WAIT`; a broken or timed-out barrier
    lets the process go on."""
    try:
        barrier.wait(BARRIER_WAIT)
    except threading.BrokenBarrierError:
        pass


def hold_lock(path: str, held: Any, release: Any) -> None:
    """Hold the lock file at ``path`` (the shared primitive) until ``release``
    is set, saying ``held`` once it does."""
    from src.utils.file_lock import exclusive_lock

    with exclusive_lock(Path(path), grace=WAIT):
        held.set()
        release.wait(WAIT)


def register(name: str, registry: str, report: str, checkpoint_dir: str, start: Any,
             barrier: Any, answers: Any, after: Any = None, done: Any = None) -> None:
    """Register ``report`` into the ledger beside the checkpoint in
    ``checkpoint_dir``, with ``registry`` as the deployment's registry, once
    every process at ``start`` is ready (and ``after`` is set, when given).
    With ``barrier``, held there right after `write_ledger`'s guard has read
    the ledger's digest. Puts what it answered on ``answers``; sets ``done``
    when it has returned."""
    os.environ["SHEPHERD_REGISTRY"] = registry
    import src.evaluation.sidecar as sidecar
    from src.evaluation.registration import RegistrationRefused, register_report

    if barrier is not None:
        real = sidecar.ledger_digest

        def ledger_digest(path):
            digest = real(path)
            _wait(barrier)
            return digest

        sidecar.ledger_digest = ledger_digest
    if start is not None:
        _wait_long(start)
    if after is not None:
        after.wait(WAIT)
    try:
        result = register_report(Path(report), checkpoint_dir=Path(checkpoint_dir),
                                 registered_at=WHEN)
        answers.put({"name": name, "outcome": result.outcome,
                     "sequence": result.record["registration"]["sequence"],
                     "source": result.record["source_artifact_digest"],
                     "run_id": result.record.get("run_id"),
                     "ledger": str(result.ledger_path),
                     "problem": result.usage_history_problem})
    except RegistrationRefused as exc:
        answers.put({"name": name, "outcome": "refused", "kind": type(exc).__name__,
                     "message": str(exc)})
    finally:
        if done is not None:
            done.set()


def _wait_long(barrier: Any) -> None:
    """Every process ready before any goes on (imports done), within `WAIT`."""
    try:
        barrier.wait(WAIT)
    except threading.BrokenBarrierError:
        pass


def _version(registry: str, label: str) -> Any:
    from src.evaluation.cohort_registry import resolve_version

    return resolve_version(Path(registry), label)


def append_test_run(name: str, registry: str, label: str, run_id: str, start: Any,
                    answers: Any) -> None:
    """Append one ``test_run`` entry for ``run_id`` once every process at
    ``start`` is ready."""
    from src.evaluation.report_store import record_test_run

    version = _version(registry, label)
    if start is not None:
        _wait_long(start)
    try:
        record_test_run(version, run_id=run_id, report_sha256="e" * 64,
                        checkpoint_sha256="e" * 64, metrics={"pooled_mrr": 0.5}, n_ranked=8,
                        deterministic=True, device="cpu", record_requested=False,
                        mode="disease")
        answers.put({"name": name, "outcome": "appended"})
    except OSError as exc:
        answers.put({"name": name, "outcome": "failed", "message": str(exc)})


def append_registration(name: str, registry: str, label: str, run_id: str, after: Any,
                        done: Any, answers: Any) -> None:
    """Append one ``registered`` entry for ``run_id`` once ``after`` is set;
    sets ``done`` when the append has returned."""
    from src.evaluation.report_store import record_registration

    version = _version(registry, label)
    after.wait(WAIT)
    try:
        record_registration(version, run_id=run_id, report_sha256="b" * 64,
                            checkpoint_sha256="b" * 64, outcome="registered", sequence=1)
        answers.put({"name": name, "outcome": "appended"})
    except OSError as exc:
        answers.put({"name": name, "outcome": "failed", "message": str(exc)})
    finally:
        done.set()


def failing_append(name: str, registry: str, label: str, sized: Any,
                   others_done: Sequence[Any], answers: Any) -> None:
    """Writer A: append a ``test_run`` entry to the version's ``usage.jsonl``
    (which must exist) whose write fails. Its first write to that file says
    ``sized`` (A has read the file's size), waits for each of ``others_done``
    (at most `BARRIER_WAIT` each), writes 10 bytes and fails with ENOSPC; its
    undo waits for them again before it cuts the file back. Without a lock the
    others append in between and the cut removes their lines; with it they
    wait until A has undone its own part and let go."""
    from src.evaluation.report_store import USAGE_FILE, record_test_run

    version = _version(registry, label)
    usage = version.directory / "reports" / USAGE_FILE
    inode = os.stat(usage).st_ino
    real_write, real_truncate = os.write, os.ftruncate

    def ours(fd: int) -> bool:
        return os.fstat(fd).st_ino == inode

    def write(fd, data):
        if not ours(fd):
            return real_write(fd, data)
        sized.set()
        for event in others_done:
            event.wait(BARRIER_WAIT)
        real_write(fd, bytes(data[:10]))
        raise OSError(errno.ENOSPC, "No space left on device")

    def ftruncate(fd, length):
        if ours(fd):
            for event in others_done:
                event.wait(BARRIER_WAIT)
        return real_truncate(fd, length)

    os.write, os.ftruncate = write, ftruncate
    try:
        record_test_run(version, run_id="A-failed", report_sha256="a" * 64,
                        checkpoint_sha256="a" * 64, metrics={"pooled_mrr": 0.5}, n_ranked=8,
                        deterministic=True, device="cpu", record_requested=False,
                        mode="disease")
        answers.put({"name": name, "outcome": "appended"})
    except OSError as exc:
        answers.put({"name": name, "outcome": "failed", "message": str(exc)})
    finally:
        os.write, os.ftruncate = real_write, real_truncate


def _marker_barrier(markers: Path, step: str, name: str, names: Sequence[str]) -> bool:
    """A barrier across processes made of files: leave ``<step>-<name>`` in
    ``markers`` and wait, within `WAIT`, until every one of ``names`` has left
    its own. False when the wait ran out."""
    (markers / f"{step}-{name}").touch()
    deadline = time.monotonic() + WAIT
    while time.monotonic() < deadline:
        if all((markers / f"{step}-{other}").exists() for other in names):
            return True
        time.sleep(0.005)
    return False


def measure_generated(name: str, argv: Sequence[str], lock_root: str, markers: str,
                      names: Sequence[str], no_hard_links: bool, answers: Any) -> None:
    """Run `scripts/measure_scorer.py`'s `main` with ``argv`` (a generated
    split and its `--output`), under a GPU lock of its own (``lock_root``: two
    deployments, or two roots, writing to one shared place), and put what it
    did on ``answers``.

    Held at two file-marker barriers in ``markers``: right after `main`'s check
    of the output has passed (``checked``), and right before publication
    (``publishing``), so every process in ``names`` has passed the check and
    publishes at the same moment. With ``no_hard_links``, `os.link` answers as
    a filesystem without hard links does (EPERM). Every publication is
    recorded with the digest of the bytes it tried (``tried``) and, when it
    placed them, of the bytes placed (``placed``)."""
    import contextlib
    import hashlib
    import io

    os.environ["SHEPHERD_WORKSPACES_ROOT"] = lock_root
    import scripts.measure_scorer as cli
    import src.utils.file_modes as file_modes

    if no_hard_links:
        def no_link(src, dst, *args, **kwargs):
            raise OSError(errno.EPERM, "Operation not permitted", str(src), None, str(dst))

        os.link = no_link
    where = Path(markers)
    passed = {"checked": False, "publishing": False}
    tried: dict[str, str] = {}
    placed: dict[str, str] = {}

    real_check = getattr(cli, "_check_generated_targets", None)
    if real_check is not None:
        def check(output):
            real_check(output)
            passed["checked"] = _marker_barrier(where, "checked", name, names)

        cli._check_generated_targets = check
    else:  # a revision without the check: every run passes it
        passed["checked"] = True
    real_publish_generated = cli._publish_generated

    def publish_generated(result, output):
        passed["publishing"] = _marker_barrier(where, "publishing", name, names)
        return real_publish_generated(result, output)

    cli._publish_generated = publish_generated
    real_publish = file_modes.publish_bytes

    def publish(target, data, **kwargs):
        tried[Path(target).name] = hashlib.sha256(data).hexdigest()
        digest = real_publish(target, data, **kwargs)
        placed[Path(target).name] = digest
        return digest

    file_modes.publish_bytes = publish
    answer: dict[str, Any] = {"name": name, "status": None, "error": None, "cause": None,
                              "message": None}
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            answer["status"] = cli.main(list(argv))
    except Exception as exc:  # noqa: BLE001 - reported to the test, which judges it
        answer.update(error=type(exc).__name__, message=str(exc),
                      cause=type(exc.__cause__).__name__ if exc.__cause__ else None)
    answer.update(passed, tried=tried, placed=placed)
    answers.put(answer)


__all__ = [
    "BARRIER_WAIT",
    "CONTEXT",
    "WAIT",
    "WHEN",
    "answers_from",
    "append_registration",
    "append_test_run",
    "failing_append",
    "hold_lock",
    "measure_generated",
    "processes",
    "register",
]
