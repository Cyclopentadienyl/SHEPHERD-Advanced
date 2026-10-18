"""
The launcher opens the dashboard when the server answers, however long that takes.

It gave up after 60 s, and a start that loads a large SP table eagerly took
about 84 s, so the browser never opened (BACKLOG 17, F7). uvicorn binds only
after the lifespan, where that load runs, so the first `/health` answer is the
end of startup and there is no deadline to guess.

Nothing here waits or opens a socket. The launcher's own `time`, `urlopen` and
`webbrowser` names are replaced by a simulated clock and recorders, and the
poll runs in the test's thread, not in a real daemon thread. A poll that would
not end on its own is ended by the simulated clock raising.
"""
import importlib.util
import re
import types
from pathlib import Path
from urllib.error import URLError

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "scripts" / "launch" / "shep_launch.py"
UI = "http://127.0.0.1:8000/ui"
HEALTH = "http://127.0.0.1:8000/health"
PROGRESS = re.compile(r"Server still starting \((\d+) s\)")


def _launcher():
    spec = importlib.util.spec_from_file_location("shep_launch", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _SimulationEndedError(Exception):
    """Raised by the simulated clock: the run being simulated has ended."""


class _Server:
    """Simulated time, and a server whose `/health` answers after `refusals` attempts.

    `refusals=None` never answers. Each failed attempt costs `attempt_cost`
    simulated seconds -- 0 for a refused connection, 3 for the per-request
    timeout -- and raises `failure`. `stop_at` ends the run, in elapsed seconds.
    """

    def __init__(self, refusals=None, attempt_cost=0.0, failure=None, stop_at=None):
        self.now = self.start = 5000.0  # a monotonic clock's origin is arbitrary
        self.refusals = refusals
        self.attempt_cost = attempt_cost
        self.failure = failure or URLError(ConnectionRefusedError(111, "Connection refused"))
        self.stop_at = stop_at
        self.attempts = []  # (elapsed, url, timeout)
        self.sleeps = []
        self.opened = []

    @property
    def elapsed(self):
        return self.now - self.start

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds
        if self.stop_at is not None and self.elapsed >= self.stop_at:
            raise _SimulationEndedError

    def urlopen(self, url, timeout):
        self.attempts.append((self.elapsed, url, timeout))
        if self.refusals is None or len(self.attempts) <= self.refusals:
            self.now += self.attempt_cost
            raise self.failure
        return types.SimpleNamespace(status=200)

    def install(self, monkeypatch, launcher):
        clock = types.SimpleNamespace(monotonic=self.monotonic, sleep=self.sleep)
        monkeypatch.setattr(launcher, "time", clock)
        monkeypatch.setattr(launcher, "urlopen", self.urlopen)
        monkeypatch.setattr(launcher, "webbrowser", types.SimpleNamespace(open=self.opened.append))


def _poll(monkeypatch, server):
    launcher = _launcher()
    server.install(monkeypatch, launcher)
    return launcher._open_browser_when_ready(UI, HEALTH)


def test_it_opens_the_browser_once_when_health_first_answers(monkeypatch, capsys):
    server = _Server(refusals=5)
    _poll(monkeypatch, server)
    assert server.opened == [UI]
    # One attempt a second, each with the 3 s per-request timeout, and none
    # after the answer: the browser opens once and the thread is done.
    assert server.attempts == [(float(t), HEALTH, 3) for t in range(6)]
    assert server.sleeps == [1.0] * 5
    assert f"Server ready — opening browser: {UI}" in capsys.readouterr().out


def test_it_does_not_give_up_before_the_server_answers(monkeypatch, capsys):
    # 200 attempts that each time out after 3 s: the answer comes at 800 s,
    # far past the 60 s after which the launcher used to stop waiting.
    server = _Server(refusals=200, attempt_cost=3.0, failure=TimeoutError("timed out"))
    _poll(monkeypatch, server)
    assert server.opened == [UI]
    assert len(server.attempts) == 201
    assert server.elapsed == 800.0
    out = capsys.readouterr().out
    assert "WARNING" not in out
    assert "skipping" not in out


@pytest.mark.parametrize(
    "attempt_cost, failure",
    [(0.0, None), (3.0, TimeoutError("timed out"))],
    ids=["refused", "timing-out"],
)
def test_it_logs_progress_about_every_30_seconds(monkeypatch, capsys, attempt_cost, failure):
    server = _Server(refusals=150, attempt_cost=attempt_cost, failure=failure)
    _poll(monkeypatch, server)
    assert server.opened == [UI]
    marks = [int(s) for s in PROGRESS.findall(capsys.readouterr().out)]
    # Measured on the clock, not counted in attempts: a timing-out attempt takes
    # 4 s with its sleep, so a mark can land up to one attempt after 30 s.
    longest_gap = 30 + 1 + attempt_cost
    gaps = [b - a for a, b in zip([0, *marks[:-1]], marks, strict=True)]
    assert marks and all(30 <= gap <= longest_gap for gap in gaps), marks
    assert server.elapsed // longest_gap <= len(marks) <= server.elapsed // 30


def test_it_does_not_open_while_health_never_answers(monkeypatch, capsys):
    # Three simulated hours of refusals. The poll is still going when the
    # simulation ends it: it neither opened a browser nor returned on its own.
    hours = 3 * 3600
    server = _Server(refusals=None, stop_at=hours)
    with pytest.raises(_SimulationEndedError):
        _poll(monkeypatch, server)
    assert server.opened == []
    assert len(server.attempts) == hours
    assert server.attempts[-1][0] == hours - 1
    out = capsys.readouterr().out
    assert "opening browser" not in out
    assert "WARNING" not in out
    assert len(PROGRESS.findall(out)) == hours // 30 - 1
