"""
One local start per configuration (UI-06, plan §4.5): the state directory, the
instance lock, its info file, and the service's side of it.
==============================================================================
`src/utils/launch_instance.py` is shared by the launcher, which takes the lock
and writes the info file (tests/unit/test_service_unit.py runs its `main`),
and the service, which answers the instance id on ``/health`` and removes its
info file at shutdown. Each test names its own state directory; nothing here
reads or writes the account's.

Module: tests/unit/test_launch_instance.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.utils import launch_instance as li

REPO_ROOT = Path(__file__).resolve().parents[2]
POSIX = pytest.mark.skipif(os.name != "posix", reason="POSIX locks and exec")


@pytest.mark.parametrize("environ, platform, expected", [
    ({"SHEPHERD_STATE_DIR": "/srv/state", "HOME": "/home/u"}, "linux", "/srv/state"),
    ({"XDG_STATE_HOME": "/x/state", "HOME": "/home/u"}, "linux", "/x/state/shepherd-advanced"),
    ({"HOME": "/home/u"}, "linux", "/home/u/.local/state/shepherd-advanced"),
    ({"XDG_STATE_HOME": "relative", "HOME": "/home/u"}, "linux",
     "/home/u/.local/state/shepherd-advanced"),
    ({"HOME": "/Users/u"}, "darwin", "/Users/u/Library/Application Support/shepherd-advanced"),
    ({"LOCALAPPDATA": "C:\\Users\\u\\AppData\\Local", "HOME": "/home/u"}, "win32", None),
    ({}, "linux", None),
    ({"SHEPHERD_STATE_DIR": "relative"}, "linux", None),
], ids=["explicit", "xdg", "home", "relative-xdg-ignored", "macos", "windows", "nothing",
        "relative-explicit-ignored"])
def test_the_state_directory_is_read_from_the_mapping_given(environ, platform, expected):
    found = li.state_directory(environ, platform=platform)
    if platform == "win32" and os.name != "nt":
        # A Windows path is not absolute on POSIX; the rule is checked where it runs.
        assert found is None
        return
    assert found == (None if expected is None else Path(expected))


def test_a_configuration_is_the_repository_and_its_settings_file():
    a = li.instance_key("/repo", "/home/u/.config/shepherd-advanced/deployment.json")
    assert a == li.instance_key("/repo", "/home/u/.config/shepherd-advanced/deployment.json")
    assert a != li.instance_key("/repo", "/slot-b/deployment.json")
    assert a != li.instance_key("/other-repo", "/home/u/.config/shepherd-advanced/deployment.json")
    assert li.instance_key("/repo", None) != a and len(a) == 16


def test_the_lock_is_one_holder_at_a_time_and_freed_with_its_descriptor(tmp_path):
    files = li.instance_files(tmp_path / "state", "k")
    fd = li.claim(files.lock)
    assert fd is not None and not os.get_inheritable(fd)
    assert li.claim(files.lock) is None
    os.close(fd)
    again = li.claim(files.lock)
    assert again is not None
    os.close(again)
    assert (files.lock.parent.stat().st_mode & 0o777) == 0o700


@POSIX
def test_the_lock_goes_through_the_exec_into_the_server(tmp_path):
    """The launcher takes the lock and execs into uvicorn: the descriptor is
    made inheritable first, so the process it becomes holds the lock until it
    ends; the kernel frees it then, with nothing left to clean."""
    files = li.instance_files(tmp_path / "state", "k")
    program = (
        "import os, sys\n"
        "from src.utils import launch_instance as li\n"
        "fd = li.claim(li.Path(sys.argv[1]))\n"
        "li.hand_to_exec(fd)\n"
        "os.execv(sys.executable, [sys.executable, '-c', "
        "'import sys, time; print(\"served\", flush=True); sys.stdin.read()'])\n")
    child = subprocess.Popen([sys.executable, "-c", program, str(files.lock)], cwd=REPO_ROOT,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "served"
        assert li.claim(files.lock) is None, "the exec'd process does not hold the lock"
    finally:
        child.stdin.close()
        child.wait(timeout=30)
        child.stdout.close()
    fd = li.claim(files.lock)
    assert fd is not None, "the lock outlived its holder"
    os.close(fd)


def test_the_info_file_is_written_whole_and_read_only_when_it_is_one(tmp_path):
    path = tmp_path / "state" / "instances" / "k.json"
    info = li.write_info(path, instance_id="abc", pid=12, port=8001, url="http://127.0.0.1:8001",
                         repo_root=REPO_ROOT, settings_file=None, log=None)
    assert li.read_info(path) == info and info["format"] == li.INFO_FORMAT
    assert (path.stat().st_mode & 0o777) == 0o600
    for text in ("not json", "[]", json.dumps({**info, "format": "other/1"}),
                 json.dumps({k: v for k, v in info.items() if k != "url"})):
        path.write_text(text, encoding="utf-8")
        assert li.read_info(path) is None
    assert li.read_info(tmp_path / "absent.json") is None


def test_the_server_removes_only_its_own_info_file(tmp_path):
    path = tmp_path / "k.json"
    li.write_info(path, instance_id="mine", pid=1, port=8000, url="http://127.0.0.1:8000",
                  repo_root=REPO_ROOT, settings_file=None, log=None)
    env = {li.ENV_INSTANCE_ID: "theirs", li.ENV_INSTANCE_FILE: str(path)}
    assert li.remove_own_info(env) is False and path.exists()
    assert li.remove_own_info({li.ENV_INSTANCE_FILE: str(path)}) is False and path.exists()
    assert li.remove_own_info({**env, li.ENV_INSTANCE_ID: "mine"}) is True
    assert not path.exists()
    assert li.remove_own_info({**env, li.ENV_INSTANCE_ID: "mine"}) is False


def test_a_large_launch_log_is_moved_aside_once(tmp_path):
    log = tmp_path / "launch.log"
    log.write_bytes(b"x" * 10)
    assert li.rotate_log(log, limit=20) is False
    log.write_bytes(b"y" * 30)
    assert li.rotate_log(log, limit=20) is True
    assert not log.exists() and (tmp_path / "launch.log.1").read_bytes() == b"y" * 30
    assert li.rotate_log(tmp_path / "absent.log") is False


# --------------------------------------------------------------------------- the service
def test_health_answers_the_launcher_s_instance_id_and_shutdown_removes_its_file(
        monkeypatch, tmp_path):
    """The service's side: /health names the id the launcher gave it (a second
    start tells it from another program on the port by that), and at
    shutdown its info file is removed, so a later start finds none."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from src.api import main

    path = tmp_path / "state" / "instances" / "k.json"
    li.write_info(path, instance_id="mine", pid=os.getpid(), port=8000,
                  url="http://127.0.0.1:8000", repo_root=REPO_ROOT, settings_file=None, log=None)
    monkeypatch.setenv(li.ENV_INSTANCE_ID, "mine")
    monkeypatch.setenv(li.ENV_INSTANCE_FILE, str(path))
    with TestClient(main.app) as client:
        assert client.get("/health").json()["instance_id"] == "mine"
        assert path.exists()
    assert not path.exists()

    monkeypatch.delenv(li.ENV_INSTANCE_ID)
    monkeypatch.delenv(li.ENV_INSTANCE_FILE)
    with TestClient(main.app) as client:
        assert client.get("/health").json()["instance_id"] is None


def test_shutdown_keeps_an_info_file_of_another_start(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from src.api import main

    path = tmp_path / "k.json"
    li.write_info(path, instance_id="later-start", pid=1, port=8000,
                  url="http://127.0.0.1:8000", repo_root=REPO_ROOT, settings_file=None, log=None)
    monkeypatch.setenv(li.ENV_INSTANCE_ID, "mine")
    monkeypatch.setenv(li.ENV_INSTANCE_FILE, str(path))
    with TestClient(main.app):
        pass
    assert li.read_info(path)["instance_id"] == "later-start"
