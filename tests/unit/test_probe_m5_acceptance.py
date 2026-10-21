"""The deployment probe's M5 acceptance: its parts that are not phases.

The phases themselves (W1-W11 in phase M, RW1-RW3 in phase R) run end to end in
tests/unit/test_probe_m2_phase.py. This shows, without starting a service or a
job, that what they rest on holds and that their checks fire:

- every M5 probe is required, so a missing one leaves the acceptance incomplete;
- A5 records the installed packages against the lock beside the checkout and
  states a set that differs (the homelab's: huggingface-hub 1.33.0, click
  8.5.0, hf-xet 1.7.0, aiofiles and ffmpy left over), never hides it: a
  package installed twice, the deploy scripts' packages told from leftovers,
  whether the lock is the committed one, all said again at the end of a run;
  without a lock it is skipped, so an acceptance is incomplete;
- a temporary directory in which a DataLoader worker's socket cannot be made
  is said, as the Python that runs the jobs answers it, and RW3 stops early;
- the M5 settings are the probe's own while it runs and are put back, so an
  operator's shell holding the deployment's settings changes nothing; the
  deployment's own lock is read and stated;
- the service is started as the launcher starts it, with the probe's settings,
  and ends with the probe; SIGTERM ends the probe through its cleanups;
- D8's, CORS's and the route guard's checks each fail on the answer they exist
  to catch; every gated API route and the gated dashboard events are found;
  the URL-fetch count fails when the guard is gone;
- a training stop is required not to reach a test run.

Module: tests/unit/test_probe_m5_acceptance.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _probe():
    spec = importlib.util.spec_from_file_location(
        "probe_deployment", REPO / "scripts" / "probe_deployment.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


@pytest.fixture(scope="module")
def probe():
    return _probe()


# --------------------------------------------------------------------------- required
def test_every_m5_probe_is_required_and_a_missing_one_leaves_its_part_incomplete(probe):
    assert {"A5", *probe.M5_DEMO_PROBES} <= set(probe.M2_DEMO_REQUIRED)
    assert set(probe.M5_REAL_PROBES) <= set(probe.M2_REAL_REQUIRED)
    assert probe.M5_DEMO_PROBES == tuple(f"W{i}" for i in range(1, 12))
    assert len(set(probe.M2_DEMO_REQUIRED)) == len(probe.M2_DEMO_REQUIRED)
    assert len(set(probe.M2_REAL_REQUIRED)) == len(probe.M2_REAL_REQUIRED)
    report = probe.Report()
    for probe_id in (*probe.M2_DEMO_REQUIRED, *probe.M2_REAL_REQUIRED):
        report.probes.append(probe.Probe(probe_id, "m2", "x", status="passed"))
    assert probe.m2_acceptance(report, True)["complete"]
    report.probes = [p for p in report.probes if p.probe_id not in ("W4", "RW3")]
    acceptance = probe.m2_acceptance(report, True)
    assert acceptance["demo"]["not_passed"] == {"W4": "not run"}
    assert acceptance["real_data"]["not_passed"] == {"RW3": "not run"}
    assert "M5" in acceptance["scope"] and "no long training" in acceptance["scope"]


# --------------------------------------------------------------------------- A5
def _lock(path: Path, packages) -> Path:
    lines = ["version = 1", 'requires-python = ">=3.12"', ""]
    for name, version in packages:
        lines += ["[[package]]", f'name = "{name}"', f'version = "{version}"', ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _installed(monkeypatch, distributions):
    """Fake installed distributions: (name, version) or (name, version,
    requires)."""
    import importlib.metadata

    fakes = [SimpleNamespace(metadata={"Name": entry[0]}, version=entry[1],
                             requires=list(entry[2]) if len(entry) > 2 else None)
             for entry in distributions]
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: iter(fakes))


def test_a_set_that_differs_from_the_lock_is_stated_never_hidden(probe, tmp_path, monkeypatch):
    """The homelab's check set against the committed lock (M5-0): three
    versions moved, two leftovers of Gradio 5.29.1, the deploy scripts' PyG
    with what it requires, an extra not installed, and a stale second copy of
    gradio. A package the lock pins at two versions (one per platform) matches
    either; versions compare as PEP 440 does."""
    lock = _lock(tmp_path / "uv.lock", [
        ("huggingface-hub", "1.16.1"), ("click", "8.3.3"), ("hf-xet", "1.5.0"),
        ("gradio", "6.30.0"), ("numpy", "2.3.4"), ("numpy", "2.2.6"),
        ("torch", "2.10.0+cu130"), ("tqdm", "4.67.1"), ("spacy", "3.8.14")])
    _installed(monkeypatch, [
        ("huggingface_hub", "1.33.0"), ("Click", "8.5.0"), ("hf_xet", "1.7.0"),
        ("gradio", "6.30.0"), ("numpy", "2.2.6", ["numpy-helper"]), ("numpy_helper", "1.0"),
        ("torch", "2.10.0+cu130"),
        ("tqdm", "4.67.1.0"), ("aiofiles", "24.1.0"), ("ffmpy", "1.0.0", ["old-helper"]),
        ("torch_geometric", "2.8.0.post1",
         ["aiohttp", "numpy", "xxhash; python_version >= '3'", "pytest; extra == 'test'"]),
        ("aiohttp", "3.13.0", ["Multidict>=4.5"]), ("multidict", "6.7.0"),
        ("xxhash", "3.6.0"), ("pytest", "9.0.0"), ("old_helper", "0.1"),
        ("gradio", "5.29.1")])            # a stale .dist-info beside the current one

    recorded = probe.installed_against_lock(lock)

    assert recorded["lock_sha256"] == hashlib.sha256(lock.read_bytes()).hexdigest()
    assert recorded["lock_is_the_committed_one"] is None          # not in a checkout
    assert recorded["versions_match_lock"] is False
    assert recorded["version_differs"] == [
        {"name": "click", "installed": "8.5.0", "locked": ["8.3.3"]},
        {"name": "gradio", "installed": "5.29.1", "locked": ["6.30.0"]},
        {"name": "hf-xet", "installed": "1.7.0", "locked": ["1.5.0"]},
        {"name": "huggingface-hub", "installed": "1.33.0", "locked": ["1.16.1"]},
    ]
    assert recorded["installed_more_than_once"] == [
        {"name": "gradio", "versions": ["5.29.1", "6.30.0"]}]
    deploy, required, leftover = (probe.OUTSIDE_LOCK_DEPLOY, probe.OUTSIDE_LOCK_REQUIRED,
                                  probe.OUTSIDE_LOCK_LEFTOVER)
    assert recorded["installed_not_in_lock"] == [
        {"name": "aiofiles", "version": "24.1.0", "why": leftover},
        {"name": "aiohttp", "version": "3.13.0", "why": deploy},
        {"name": "ffmpy", "version": "1.0.0", "why": leftover},
        {"name": "multidict", "version": "6.7.0", "why": deploy},
        # A lock package's requirement: the deploy scripts' closure does not
        # pass through what the sync installed.
        {"name": "numpy-helper", "version": "1.0", "why": required},
        {"name": "old-helper", "version": "0.1", "why": required},
        {"name": "pytest", "version": "9.0.0", "why": leftover},     # only an extra asks
        {"name": "torch-geometric", "version": "2.8.0.post1", "why": deploy},
        {"name": "xxhash", "version": "3.6.0", "why": deploy}]
    assert recorded["locked_not_installed"] == ["spacy"]
    assert (recorded["lock_packages"], recorded["installed_distributions"]) == (8, 16)
    assert str(tmp_path) not in json.dumps(recorded)

    said = probe.lock_sentence(recorded)
    assert "4 version(s) differ (click 8.5.0, lock 8.3.3, gradio 5.29.1, lock 6.30.0" in said
    assert "installed more than once: gradio (5.29.1, 6.30.0)" in said
    assert "9 installed package(s) are not in the lock: 4 from the deploy scripts" in said
    assert "required by nothing installed: aiofiles 24.1.0, ffmpy 1.0.0, pytest 9.0.0" in said
    assert "required by another installed package: numpy-helper 1.0, old-helper 0.1" in said
    assert "whether it is the committed one could not be read" in said


def test_a_second_installed_copy_is_stated_whatever_the_directory_order(
        probe, tmp_path, monkeypatch):
    """Two real .dist-info directories of one package in one site directory:
    importlib.metadata finds both in directory order; A5 compares both and
    says the package is installed twice (it kept the first one found, and
    said nothing of the other, before M5-3's review)."""
    from importlib import metadata

    site = tmp_path / "site-packages"
    for version in ("6.30.0", "5.29.1"):
        info = site / f"gradio-{version}.dist-info"
        info.mkdir(parents=True)
        (info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: gradio\nVersion: {version}\n", encoding="utf-8")
    lock = _lock(tmp_path / "uv.lock", [("gradio", "6.30.0")])
    found = metadata.distributions
    monkeypatch.setattr(metadata, "distributions", lambda: found(path=[str(site)]))

    recorded = probe.installed_against_lock(lock)

    assert recorded["installed_more_than_once"] == [
        {"name": "gradio", "versions": ["5.29.1", "6.30.0"]}]
    assert recorded["version_differs"] == [
        {"name": "gradio", "installed": "5.29.1", "locked": ["6.30.0"]}]
    assert recorded["installed_distributions"] == 1


def _git(cwd, *args):
    import subprocess

    subprocess.run(["git", "-c", "user.name=probe", "-c", "user.email=probe@example.invalid",
                    "-c", "commit.gpgsign=false", *args], cwd=cwd, check=True,
                   capture_output=True)


def test_whether_the_lock_read_is_the_committed_one_is_said(probe, tmp_path, monkeypatch):
    """A lock made again on the machine and not committed is not taken for the
    committed one, a line-ending conversion git would undo is no difference,
    and outside a checkout it cannot be said."""
    _installed(monkeypatch, [("gradio", "6.30.0")])
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    lock = _lock(checkout / "uv.lock", [("gradio", "6.30.0")])
    (checkout / ".gitattributes").write_text("* text=auto\n", encoding="utf-8")
    _git(checkout, "init", "-q")
    assert probe.installed_against_lock(lock)["lock_is_the_committed_one"] is None  # no commit
    (checkout / "README").write_text("x\n", encoding="utf-8")
    _git(checkout, "add", "README", ".gitattributes")
    _git(checkout, "commit", "-q", "-m", "no lock yet")
    assert probe.installed_against_lock(lock)["lock_is_the_committed_one"] is False
    _git(checkout, "add", "uv.lock")
    _git(checkout, "commit", "-q", "-m", "lock")
    recorded = probe.installed_against_lock(lock)
    assert recorded["lock_is_the_committed_one"] is True
    assert "(the one committed at the tested commit)" in probe.lock_sentence(recorded)
    lock.write_bytes(lock.read_bytes().replace(b"\n", b"\r\n"))
    assert probe.installed_against_lock(lock)["lock_is_the_committed_one"] is True
    _lock(lock, [("gradio", "6.30.0"), ("click", "8.5.0")])       # locked again, not committed
    recorded = probe.installed_against_lock(lock)
    assert recorded["lock_is_the_committed_one"] is False
    assert "which is NOT the one committed at the tested commit" in probe.lock_sentence(recorded)


def test_the_deploy_scripts_install_the_roots_a5_names(probe):
    """`DEPLOY_INSTALLED_ROOTS` are what deploy.sh installs after the sync, so
    A5's "from the deploy scripts" follows them."""
    script = (REPO / "deploy.sh").read_text(encoding="utf-8")
    spelled = {probe._normalised(word)
               for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9._-]*[A-Za-z0-9]", script)}
    assert set(probe.DEPLOY_INSTALLED_ROOTS) <= spelled


def test_a_matching_set_says_so(probe, tmp_path, monkeypatch):
    lock = _lock(tmp_path / "uv.lock", [("gradio", "6.30.0"), ("torch-geometric", "2.8.0-post1")])
    _installed(monkeypatch, [("gradio", "6.30.0"), ("torch_geometric", "2.8.0.post1")])
    recorded = probe.installed_against_lock(lock)
    assert recorded["versions_match_lock"] is True
    assert recorded["version_differs"] == recorded["installed_not_in_lock"] == []


def test_without_a_lock_the_record_is_skipped_and_the_acceptance_incomplete(probe, tmp_path):
    with pytest.raises(probe.SkipProbe, match="no uv.lock"):
        probe.installed_against_lock(tmp_path / "uv.lock")


def _differing():
    return {"lock_sha256": "0" * 64, "lock_is_the_committed_one": False, "lock_packages": 2,
            "installed_distributions": 4, "versions_match_lock": False,
            "version_differs": [{"name": "click", "installed": "8.5.0", "locked": ["8.3.3"]}],
            "installed_more_than_once": [],
            "installed_not_in_lock": [
                {"name": "aiofiles", "version": "24.1.0", "why": "required_by_nothing_installed"},
                {"name": "torch-geometric", "version": "2.8.0", "why": "deploy_scripts"}],
            "locked_not_installed": []}


def test_a5_states_a_differing_set_on_the_console_and_in_the_environment(
        probe, monkeypatch, capsys):
    monkeypatch.setattr(probe, "installed_against_lock", lambda path: _differing())
    report = probe.Report()

    environment = probe.phase_environment(report, "cpu", describe_only=True)

    a5 = {p.probe_id: p for p in report.probes}["A5"]
    assert a5.status == "passed" and "1 installed version(s) differ" in a5.detail
    said = capsys.readouterr().out
    assert "click 8.5.0, lock 8.3.3" in said and "aiofiles 24.1.0" in said
    assert "NOT the one committed at the tested commit" in said
    assert "1 from the deploy scripts" in said and "torch-geometric" not in said
    assert environment["installed_against_lock"]["version_differs"] == _differing()[
        "version_differs"]
    assert "socket_fits" in environment["installed_against_lock"]["temporary_directory"]


def test_a5_is_said_again_beside_the_acceptance_verdict(probe, monkeypatch):
    """At the end of the run and in the report's `m2_acceptance`, not only on
    a line an hour up the console."""
    report = probe.Report()
    assert probe.environment_line(report) is None
    assert probe.m2_acceptance(report, False)["environment"] == {"A5": "not run"}
    report.probes.append(probe.Probe("A5", "environment", "x", status="passed",
                                     facts=_differing()))

    verdict = probe.m2_acceptance(report, False)["environment"]
    line = probe.environment_line(report)

    assert verdict == {
        "A5": "passed", "lock_sha256": "0" * 64, "lock_is_the_committed_one": False,
        "versions_match_lock": False, "version_differs": ["click"],
        "installed_more_than_once": [], "not_in_lock": 2, "not_in_lock_from_deploy_scripts": 1,
        "not_in_lock_required_by_nothing_installed": ["aiofiles"]}
    assert line.startswith("A5, installed packages against this checkout's uv.lock, which is "
                           "NOT the one committed")
    assert "click 8.5.0, lock 8.3.3" in line and "aiofiles 24.1.0" in line
    report.probes[-1] = probe.Probe("A5", "environment", "x", status="skipped",
                                    detail="there is no uv.lock beside this checkout")
    assert "skipped (there is no uv.lock" in probe.environment_line(report)
    assert probe.m2_acceptance(report, False)["environment"] == {"A5": "skipped"}


def test_a5_runs_after_the_device_probes_in_a_full_run(probe, monkeypatch):
    monkeypatch.setattr(probe, "installed_against_lock",
                        lambda path: {"version_differs": [], "installed_not_in_lock": []})
    report = probe.Report()
    probe.phase_environment(report, "cpu")
    assert [p.probe_id for p in report.probes] == ["A1", "A2", "A3", "A4", "A5"]


def test_the_committed_lock_is_read_as_it_is(probe):
    """The repository's own uv.lock and this interpreter's packages: whatever
    they are, each difference is a name with its versions, and gradio, which
    M5-0 pinned, is among the lock's packages."""
    recorded = probe.installed_against_lock(REPO / "uv.lock")
    assert recorded["lock_sha256"] == hashlib.sha256((REPO / "uv.lock").read_bytes()).hexdigest()
    assert recorded["lock_packages"] > 100 and "gradio" not in recorded["locked_not_installed"]
    assert recorded["lock_is_the_committed_one"] in (True, False, None)
    assert all(set(entry) == {"name", "installed", "locked"}
               for entry in recorded["version_differs"])
    assert all(set(entry) == {"name", "version", "why"} and entry["why"] in (
        probe.OUTSIDE_LOCK_DEPLOY, probe.OUTSIDE_LOCK_REQUIRED, probe.OUTSIDE_LOCK_LEFTOVER)
        for entry in recorded["installed_not_in_lock"])


def _long_directory(base: Path, length: int) -> Path:
    """A real directory whose path is `length` bytes long, under `base`."""
    pad = length - len(str(base)) - 1
    if pad < 1:
        pytest.skip("the test's temporary directory is already too long")
    path = base / ("d" * pad)
    path.mkdir()
    assert len(str(path)) == length
    return path


@pytest.mark.skipif(sys.platform != "linux", reason="Linux's AF_UNIX path limit")
def test_a_temporary_directory_a_worker_socket_cannot_be_made_in_is_said(
        probe, tmp_path, monkeypatch):
    """Asked of Python, not computed: a listener opened and closed under the
    temporary directory, as multiprocessing opens one for DataLoader workers."""
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    short = probe._socket_room()
    assert short["socket_fits"] is True and short["path_bytes"] == len(str(tmp_path))
    if sys.version_info[:2] != (3, 12):
        pytest.skip("the failing case is Python 3.12's")
    monkeypatch.setattr(tempfile, "tempdir", str(_long_directory(tmp_path, 100)))
    long = probe._socket_room()
    assert long == {"path_bytes": 100, "socket_fits": False, "refusal": "AF_UNIX path too long",
                    "checked": long["checked"]}
    assert str(tmp_path) not in json.dumps(long)


@pytest.mark.skipif(sys.platform != "linux" or not Path("/usr/bin/python3.13").exists(),
                    reason="needs Linux and a Python 3.13 beside this one")
def test_python_3_13_makes_its_sockets_under_tmp_whatever_the_temporary_directory(
        probe, tmp_path, monkeypatch):
    """`requires-python` is >=3.12: on 3.13 a long TMPDIR does not fail
    training with workers (its sockets fall back to /tmp), so A5 and RW3 must
    not say it does. A computed path length said it did."""
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(_long_directory(tmp_path, 100)))
    assert probe._socket_room("/usr/bin/python3.13")["socket_fits"] is True


def test_a_socket_check_that_cannot_run_says_so(probe, tmp_path):
    room = probe._socket_room(str(tmp_path / "no-such-python"))
    assert room["socket_fits"] is None and room["checked"] == "the check could not run"


# --------------------------------------------------------------------------- settings
def test_the_m5_settings_cover_every_deployment_setting(probe):
    from src.config import server_settings
    from src.utils.gpu_lock import LOCK_FILE_ENV, WORKSPACES_ROOT_ENV

    names = {setting.variable for setting in server_settings.all_settings()}
    names |= {LOCK_FILE_ENV, WORKSPACES_ROOT_ENV, server_settings.ENV_BIND_HOST,
              server_settings.ENV_ACCEPT_UNAUTHENTICATED_NETWORK_BIND, "SHEPHERD_DEVICE",
              "SHEPHERD_KG_PATH", "SHEPHERD_CHECKPOINT_PATH", "SHEPHERD_DATA_DIR"}
    assert names <= set(probe._M5_SETTINGS)
    assert "SHEPHERD_ALLOC_SOURCE" not in probe._M5_SETTINGS     # the allocator's, left as it is


def test_the_m5_settings_are_the_probe_s_own_while_it_runs_and_put_back(probe, monkeypatch):
    import os

    monkeypatch.setenv("SHEPHERD_REGISTRY", "/deployment/registry")
    monkeypatch.setenv("SHEPHERD_GPU_LOCK_FILE", "/deployment/gpu.lock")
    monkeypatch.setenv("SHEPHERD_ALLOC_SOURCE", "preset")
    monkeypatch.delenv("SHEPHERD_WORKSPACES_ROOT", raising=False)

    with pytest.raises(RuntimeError):
        with probe._deployment_settings({"SHEPHERD_WORKSPACES_ROOT": "/probe/root"}):
            assert os.environ["SHEPHERD_WORKSPACES_ROOT"] == "/probe/root"
            assert "SHEPHERD_REGISTRY" not in os.environ
            assert "SHEPHERD_GPU_LOCK_FILE" not in os.environ
            assert os.environ["SHEPHERD_ALLOC_SOURCE"] == "preset"
            raise RuntimeError("a probe that raised")

    assert os.environ["SHEPHERD_REGISTRY"] == "/deployment/registry"
    assert os.environ["SHEPHERD_GPU_LOCK_FILE"] == "/deployment/gpu.lock"
    assert "SHEPHERD_WORKSPACES_ROOT" not in os.environ
    with pytest.raises(AssertionError, match="not an M5 setting"):
        with probe._deployment_settings({"SHEPHERD_ALLOC_SOURCE": "x"}):
            pass


def test_the_service_is_started_as_the_launcher_starts_it(probe, tmp_path, monkeypatch):
    """uvicorn on 127.0.0.1 with the launcher's own app and forced arguments,
    startup's bind check given its host, the probe's settings and none of the
    operator's, its Gradio and temporary files under its directory."""
    import subprocess

    from scripts.launch.shep_launch import UVICORN_APP, UVICORN_FORCED_ARGS

    monkeypatch.setenv("SHEPHERD_KG_PATH", "/deployment/kg.json")
    monkeypatch.setenv("SHEPHERD_DEPLOYMENT_MODE", "M1")
    monkeypatch.setenv("SHEPHERD_ALLOC_SOURCE", "preset")
    started = {}

    class _Popen:
        def __init__(self, argv, **kwargs):
            started.update(argv=argv, **kwargs)
            self.pid = 1

    monkeypatch.setattr(subprocess, "Popen", _Popen)
    server = probe._ProbeServer("read-only", tmp_path / "service",
                                {"SHEPHERD_WORKSPACES_ROOT": "/probe/root"})
    server.start()
    server._log.close()

    argv, env = started["argv"], started["env"]
    assert argv[1:4] == ["-m", "uvicorn", UVICORN_APP]
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--port") + 1] == str(server.port) and server.port > 0
    assert argv[-len(UVICORN_FORCED_ARGS):] == UVICORN_FORCED_ARGS
    assert env["SHEPHERD_BIND_HOST"] == "127.0.0.1"
    assert env["SHEPHERD_WORKSPACES_ROOT"] == "/probe/root"
    assert "SHEPHERD_KG_PATH" not in env and "SHEPHERD_DEPLOYMENT_MODE" not in env
    assert env["SHEPHERD_ALLOC_SOURCE"] == "preset"
    assert env["GRADIO_TEMP_DIR"] == str(tmp_path / "service" / "gradio")
    assert env["TMPDIR"] == str(tmp_path / "service" / "tmp")
    assert started["cwd"] == probe.project_root
    # It ends with the probe, however the probe ends (Linux).
    assert callable(started["preexec_fn"]) == sys.platform.startswith("linux")


_CHILD = "import time; time.sleep(120)"


def _gone(pid: int) -> bool:
    """Whether process `pid` has ended (a zombie nobody reaped has ended)."""
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii") as handle:
            return handle.read().rsplit(")", 1)[1].split()[0] == "Z"
    except FileNotFoundError:
        return True


def _run_parent(script: str):
    """A Python process running `script` with the probe module as `probe`;
    returns it and the child pid it prints first."""
    import subprocess

    source = str(REPO / "scripts" / "probe_deployment.py")
    code = (
        "import importlib.util, os, subprocess, sys, time\n"
        f"spec = importlib.util.spec_from_file_location('probe', {source!r})\n"
        "probe = sys.modules['probe'] = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(probe)\n"
        + script)
    parent = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    child = int(parent.stdout.readline())
    return parent, child


def _stop(pid: int) -> None:
    import os
    import signal

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_sigterm_ends_the_probe_through_its_cleanups(probe):
    """`kill <pid>` (SIGTERM) runs the probe's `finally` blocks, which stop
    the services and jobs it started, as Ctrl-C does; without the handler the
    process ends at once and a child it started keeps running."""
    import signal

    parent, child = _run_parent(
        "probe._stop_on_termination_signals()\n"
        f"started = subprocess.Popen([sys.executable, '-c', {_CHILD!r}],\n"
        "                           stdout=subprocess.DEVNULL)\n"
        "try:\n"
        "    print(started.pid, flush=True)\n"         # inside: the signal comes after it
        "    time.sleep(120)\n"
        "finally:\n"
        "    started.kill(); started.wait()\n"
        "    print('cleaned up', flush=True)\n")
    try:
        parent.send_signal(signal.SIGTERM)
        said, _ = parent.communicate(timeout=60)
        assert parent.returncode == 128 + signal.SIGTERM and "cleaned up" in said
        assert _gone(child)
    finally:
        _stop(child)
        _stop(parent.pid)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_a_signal_set_to_be_ignored_is_left_ignored(probe):
    """Under nohup SIGHUP is ignored; the probe keeps it so."""
    import signal

    saved = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGHUP)}
    try:
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        probe._stop_on_termination_signals()
        assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
        assert signal.getsignal(signal.SIGTERM) is probe._exit_on_signal
    finally:
        for number, handler in saved.items():
            signal.signal(number, handler)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG is Linux's")
def test_a_service_started_by_the_probe_ends_when_the_probe_is_killed(probe):
    """SIGKILL runs no cleanup; the kernel's parent-death signal still ends
    the service (`_ProbeServer.start` passes `_with_parent_death_signal`)."""
    import time

    parent, child = _run_parent(
        f"started = subprocess.Popen([sys.executable, '-c', {_CHILD!r}],\n"
        "                           stdout=subprocess.DEVNULL,\n"
        "                           preexec_fn=probe._with_parent_death_signal())\n"
        "print(started.pid, flush=True)\n"
        "time.sleep(120)\n")
    try:
        _stop(parent.pid)
        parent.wait(timeout=30)
        deadline = time.monotonic() + 20
        while not _gone(child) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert _gone(child), "the child outlived the killed probe"
    finally:
        _stop(child)


class _Runner:
    """A job runner's status word and stops, for the probe's job helpers."""

    def __init__(self, words, stop_answer=None):
        self.words, self.stopped, self.stop_answer = dict(words), [], stop_answer

    def get_job_status_word(self, job_type):
        return self.words.get(job_type, "idle")

    def stop(self, job_type):
        self.stopped.append(job_type)
        self.words[job_type] = "cancelled"
        return {"success": True}

    def stop_training(self):
        self.stopped.append("training")
        if self.stop_answer is None:                  # a stop that reaches whatever runs
            self.words["test"] = "cancelled"
            return {"success": True, "status": "stopped"}
        return self.stop_answer


def test_a_job_still_running_when_a_probe_ends_is_stopped(probe):
    runner = _Runner({"test": "running"})
    with pytest.raises(SystemExit):
        with probe._job_ended_on_exit(runner, "test"):
            raise SystemExit(143)                     # SIGTERM through `_exit_on_signal`
    assert runner.stopped == ["test"]
    runner = _Runner({"test": "completed"})
    with probe._job_ended_on_exit(runner, "test"):
        pass
    assert runner.stopped == []


def test_a_training_stop_is_required_not_to_reach_a_test_run(probe):
    refused = {"success": False, "error": "No training is running: the GPU job running is a "
                                          "test run, which this stop does not reach"}
    assert "refused" in probe._training_stop_does_not_reach_a_test_run(
        _Runner({"test": "running"}, stop_answer=refused))
    with pytest.raises(AssertionError, match="training stop was not refused"):
        probe._training_stop_does_not_reach_a_test_run(_Runner({"test": "running"}))
    with pytest.raises(AssertionError, match="reached the test run"):
        probe._training_stop_does_not_reach_a_test_run(
            _Runner({"test": "cancelled"}, stop_answer=refused))


@pytest.mark.skipif(sys.platform == "win32", reason="the holder is a POSIX process here")
def test_the_deployment_s_own_lock_is_read_and_stated(probe, tmp_path, monkeypatch, capsys):
    """M5's probes run on their own lock; whether the lock the operator's
    settings name is held meanwhile is stated, the holder named on the
    console only."""
    import subprocess

    from src.utils.gpu_lock import LOCK_FILE_ENV, WORKSPACES_ROOT_ENV

    monkeypatch.delenv(WORKSPACES_ROOT_ENV, raising=False)
    monkeypatch.delenv(LOCK_FILE_ENV, raising=False)
    assert probe._deployment_gpu_lock() == {"configured": False, "held": False,
                                            "unreadable": False}
    lock = tmp_path / "gpu.lock"
    monkeypatch.setenv(LOCK_FILE_ENV, str(lock))
    holder = subprocess.Popen([sys.executable, "-c", probe._LOCK_HOLDER], cwd=REPO,
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().startswith("held ")
        assert probe._deployment_gpu_lock() == {"configured": True, "held": True,
                                                "unreadable": False}
        assert f"pid {holder.pid}" in capsys.readouterr().err
    finally:
        holder.kill()
        holder.wait(timeout=30)
        holder.stdin.close()
        holder.stdout.close()
    assert probe._deployment_gpu_lock()["held"] is False


def test_rw3_stops_before_training_when_no_worker_socket_can_be_made(
        probe, tmp_path, monkeypatch):
    """Phase R's RW3 with workers in its settings and a temporary directory in
    which their sockets cannot be made: failed early, saying to use a short
    TMPDIR, no training job started."""
    import src.api.services.training_manager as training_manager

    started = []

    class _NoJobs:
        def __init__(self, *args, **kwargs):
            started.append(1)
            raise AssertionError("a job runner was used")

    monkeypatch.setattr(training_manager, "TrainingManager", _NoJobs)
    monkeypatch.setattr(probe, "_socket_room", lambda python=None: {
        "path_bytes": 120, "socket_fits": False, "refusal": "AF_UNIX path too long",
        "checked": "an AF_UNIX listener opened and closed under it"})
    monkeypatch.setattr(probe, "_deployment_gpu_lock", lambda: {
        "configured": False, "held": False, "unreadable": False})
    real = tmp_path / "workspaces" / "real"
    real.mkdir(parents=True)
    report = probe.Report()

    probe._m5_real(report, tmp_path / "work", real, tmp_path / "workspaces", "cpu",
                   {"read": {}, "settings": {"num_workers": 2}, "m4": {}})

    rw3 = {p.probe_id: p for p in report.probes}["RW3"]
    assert rw3.status == "failed" and "short TMPDIR" in rw3.detail and "120 bytes" in rw3.detail
    assert started == []


def test_a_service_that_exits_before_it_answers_is_a_failure_in_the_probe_s_words(
        probe, tmp_path):
    server = probe._ProbeServer("M1", tmp_path, {})
    (tmp_path / "server.log").write_text("Traceback ... /secret/path\n")
    server.process = SimpleNamespace(poll=lambda: 3, returncode=3)
    server.port = 1
    with pytest.raises(AssertionError, match="exited") as failed:
        server.wait_ready(timeout=5)
    assert "/secret/path" not in str(failed.value)


def test_a_port_taken_meanwhile_is_tried_once_more(probe, tmp_path, monkeypatch):
    server = probe._ProbeServer("M1", tmp_path, {})
    (tmp_path / "server.log").write_text("ERROR: [Errno 98] Address already in use\n")
    server.process = SimpleNamespace(poll=lambda: 1, returncode=1)
    restarts = []
    monkeypatch.setattr(server, "start", lambda: restarts.append(1))
    with pytest.raises(AssertionError, match="exited"):
        server.wait_ready(timeout=5)
    assert restarts == [1]


# --------------------------------------------------------------------------- checks fire
LISTED = "a" * 64
REFUSAL = '{"error": "Refused: words. Nothing was done."}'
EVENTS = [("training_console._on_stop", "_on_stop", [{}], "No training is running"),
          ("test_page._on_start", "_on_start_1", [{}] * 7, "Nothing was started."),
          ("test_page._on_register", "_on_register", [{}] * 5, "Nothing was registered.")]
STOP = "POST /api/v1/training/stop"
LIST = "GET /api/v1/workspaces/{name}/checkpoints"
RELOAD = "POST /api/v1/pipeline/reload"


def _response(status, body):
    return (status, {}, body.encode("utf-8"))


def _routes(probe, answer):
    return {f"{method} {path}": answer(method, path, expected)
            for method, path, expected in probe.OPERATOR_ROUTES}


def _refused(probe, condition="forwarding_header", routes=None, events=None):
    seen = {
        "access": {"enabled": False, "condition": condition, "reason": "words"},
        "routes": _routes(probe, lambda method, path, expected: _response(403, REFUSAL)),
        "events": {label: '["Refused: words. Nothing was done.", null]'
                   for label, _, _, _ in EVENTS},
    }
    seen["routes"].update(routes or {})
    seen["events"].update(events or {})
    return seen


def _allowed_answer(method, path, expected):
    if path.endswith("/{name}/checkpoints"):
        return _response(200, json.dumps({"checkpoints": [{"sha256": LISTED}]}))
    if path.endswith("/training/stop"):
        return _response(409, '{"error": "No training is running"}')
    return _response(expected, '{"detail": "words"}')


def _allowed(probe, access=None, routes=None, events=None):
    seen = {
        "access": access or {"enabled": True, "condition": "enabled", "reason": "words"},
        "routes": _routes(probe, _allowed_answer),
        "events": {"training_console._on_stop": '["Not stopped: No training is running", null]',
                   "test_page._on_start": '["Choose a cohort version. Nothing was started."]',
                   "test_page._on_register": '["Select a kept report. Nothing was registered."]'},
    }
    seen["routes"].update(routes or {})
    seen["events"].update(events or {})
    return seen


def test_d8_s_refusal_check_passes_a_refusal(probe):
    facts = probe._check_refused(_refused(probe), "forwarding_header", ("203.0.113.9",), LISTED)
    assert facts["condition"] == "forwarding_header"
    assert facts["routes"] == dict.fromkeys(_routes(probe, lambda *a: None), 403)
    assert facts["gradio_events"] == dict.fromkeys([e[0] for e in EVENTS], "refused")


@pytest.mark.parametrize("change, message", [
    ({"condition": "cross_site"}, "not forwarding_header"),
    ({"routes": {LIST: _response(200, json.dumps({"checkpoints": [{"sha256": LISTED}]}))}},
     "GET /api/v1/workspaces/{name}/checkpoints answered 200"),
    ({"routes": {STOP: _response(409, '{"error": "No training is running"}')}},
     "POST /api/v1/training/stop answered 409"),
    # A reload that lost D8's dependency answers the wrong-shaped body in
    # validation: refused for the wrong reason, which is not D8's refusal.
    ({"routes": {RELOAD: _response(422, '{"detail": "Input should be a valid dict"}')}},
     "POST /api/v1/pipeline/reload answered 422"),
    ({"routes": {"POST /api/v1/later/route": _response(200, "{}")}},
     "POST /api/v1/later/route answered 200"),
    ({"events": {"test_page._on_start": '["Choose a cohort version. Nothing was started."]'}},
     "test_page._on_start was not refused"),
    ({"routes": {LIST: _response(403, f'{{"error": "Refused: {LISTED}. Nothing was done."}}')}},
     "showed a checkpoint"),
    ({"routes": {STOP: _response(403, '{"error": "Refused: 203.0.113.9. Nothing was done."}')}},
     "repeated a header's value"),
])
def test_d8_s_refusal_check_fails_on_what_it_exists_to_catch(probe, change, message):
    with pytest.raises(AssertionError, match=re.escape(message)):
        probe._check_refused(_refused(probe, **change), "forwarding_header", ("203.0.113.9",),
                             LISTED)


def test_d8_s_refusal_check_requires_every_operator_route_and_an_event(probe):
    seen = _refused(probe)
    del seen["routes"][RELOAD]
    with pytest.raises(AssertionError, match="not asked"):
        probe._check_refused(seen, "forwarding_header", (), LISTED)
    seen = _refused(probe)
    seen["events"] = {}
    with pytest.raises(AssertionError, match="not asked"):
        probe._check_refused(seen, "forwarding_header", (), LISTED)


def test_d8_s_allowance_check_passes_controls_that_ran(probe):
    facts = probe._check_allowed(_allowed(probe), LISTED, "loopback", EVENTS)
    assert facts["gradio_events"] == dict.fromkeys([e[0] for e in EVENTS], "ran")
    assert facts["routes"][RELOAD] == 422 and facts["routes"][STOP] == 409


@pytest.mark.parametrize("change, message", [
    ({"access": {"enabled": False, "condition": "deployment_mode"}}, "D8 refused"),
    ({"routes": {LIST: _response(403, "Refused: words")}}, "answered 403"),
    ({"routes": {LIST: _response(200, json.dumps({"checkpoints": [{"sha256": "b" * 64}]}))}},
     "does not hold the listed checkpoint"),
    ({"routes": {STOP: _response(403, "Refused: words")}}, "D8 did not let it through"),
    ({"routes": {RELOAD: _response(403, '{"error": "Refused: words"}')}},
     "POST /api/v1/pipeline/reload answered 403"),
    ({"routes": {RELOAD: _response(200, '{"success": true}')}},     # it ran: nothing may
     "POST /api/v1/pipeline/reload answered 200"),
    ({"events": {"test_page._on_register": '["Refused: words. Nothing was done."]'}},
     "test_page._on_register did not run its handler"),
])
def test_d8_s_allowance_check_fails_on_a_refused_control(probe, change, message):
    with pytest.raises(AssertionError, match=re.escape(message)):
        probe._check_allowed(_allowed(probe, **change), LISTED, "loopback", EVENTS)


def test_every_gated_api_route_is_d8_s_and_found(probe):
    """The routes D8's dependency gates are the six its contract names
    (src/api/operator_access.py); the probe finds them from the application,
    so one added later is asked too, and asks the six whatever is found."""
    import src.api.main as api

    assert probe._gated_api_routes(api.app) == sorted(
        (method, path) for method, path, _ in probe.OPERATOR_ROUTES)


def test_the_gated_dashboard_events_are_found_by_their_handlers(probe):
    import src.api.main as api

    found = probe._gated_events(api.app)
    assert [(label, data) for label, _, data, _ in found] == [
        (f"{module}.{name}", list(data)) for module, name, data, _ in probe.GATED_EVENTS]
    names = [api_name for _, api_name, _, _ in found]
    assert names[0] == "_on_stop" and len(set(names)) == 3


def test_an_event_registered_otherwise_than_the_probe_sends_it_fails(probe, monkeypatch):
    monkeypatch.setattr(probe, "GATED_EVENTS", (("test_page", "_on_start", ({},), "x"),))
    import src.api.main as api

    with pytest.raises(AssertionError, match="other inputs than the probe sends"):
        probe._gated_events(api.app)


def test_the_url_fetch_count_is_where_a_fetch_happens(probe, monkeypatch):
    """In process: through the application the route guard answers every
    file= URL form and nothing reaches Gradio's URL fetch; with the guard's
    refusal taken away (the defect W4 exists to catch), the forms reach it.
    The control past the guard reaches it either way."""
    import src.api.main as api
    from src.api.middleware.gradio_routes import RefuseUnusedGradioRoutes

    forms = ("file=http://198.51.100.7/probe.png", "file=%68ttp://198.51.100.7/probe.png",
             "file=//198.51.100.7/probe.png")
    held = probe._url_fetches(api.app, forms)
    assert held == {"through_the_guard": 0, "past_the_guard_control": 1, "not_the_guard": []}

    monkeypatch.setattr(RefuseUnusedGradioRoutes, "refusal", lambda self, scope: None)
    gone = probe._url_fetches(api.app, forms)
    assert gone["through_the_guard"] >= 2 and gone["not_the_guard"]
    assert gone["past_the_guard_control"] == 1


def test_a_cors_grant_is_seen_whatever_its_case(probe):
    assert probe._cors_grants({"vary": "Origin"}) == []
    assert probe._cors_grants({"access-control-allow-origin": "https://other.example",
                               "access-control-allow-credentials": "true"}) == [
        "access-control-allow-credentials", "access-control-allow-origin"]


GUARD = "Refused: the dashboard takes no file uploads; nothing was written."


def _raw(status_line, headers, body=b""):
    return (status_line + b"\r\n" + b"".join(h + b"\r\n" for h in headers) + b"\r\n" + body)


def test_the_route_guard_check_passes_its_refusal(probe):
    probe._check_guard_refusal(
        _raw(b"HTTP/1.1 403 Forbidden", [b"content-type: text/plain", b"connection: close"],
             GUARD.encode()), GUARD, "upload")
    probe._check_guard_refusal(
        _raw(b"HTTP/1.1 403 Forbidden", [b"connection: close"]), None, "a HEAD")


@pytest.mark.parametrize("raw, message", [
    (_raw(b"HTTP/1.1 200 OK", [b"connection: close"], GUARD.encode()), "answered 200"),
    (_raw(b"HTTP/1.1 403 Forbidden", [b"connection: keep-alive"], GUARD.encode()),
     "connection closed"),
    (_raw(b"HTTP/1.1 403 Forbidden", [b"connection: close"], b"File not allowed"),
     "not by the route guard"),
    (b"HTTP/1.1 100 Continue\r\n\r\n" + _raw(b"HTTP/1.1 403 Forbidden",
                                            [b"connection: close"], GUARD.encode()),
     "answered 100"),
    (b"", "answered None"),
])
def test_the_route_guard_check_fails_on_what_it_exists_to_catch(probe, raw, message):
    with pytest.raises(AssertionError, match=message):
        probe._check_guard_refusal(raw, GUARD, "upload")


def test_a_holder_is_named_by_its_pid_exactly(probe):
    said = "a training job (train_model.py, pid 4242 on host, since 2026-10-21T09:00:00+00:00"
    assert probe._names_pid(said, 4242)
    assert not probe._names_pid(said, 424)
    assert not probe._names_pid(said.replace(" on host", ""), 42)


def test_a_list_item_s_identity_fields_leave_its_reasons_out(probe):
    item = {"name": "ws", "problem": "/srv/x is unreadable", "build": {
        "state": "not_recorded", "reason": "/srv/x/workspace_build.json is absent"},
        "not_trainable": "/srv/x: no manifest", "items": [{"problem": "/srv/y"}]}
    assert probe._identity_fields(item) == {"name": "ws", "build": {"state": "not_recorded"},
                                            "items": [{}]}
