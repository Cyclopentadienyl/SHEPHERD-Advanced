"""The deployment probe's M5 acceptance: its parts that are not phases.

The phases themselves (W1-W11 in phase M, RW1-RW3 in phase R) run end to end in
tests/unit/test_probe_m2_phase.py. This shows, without starting a service or a
job, that what they rest on holds and that their checks fire:

- every M5 probe is required, so a missing one leaves the acceptance incomplete;
- A5 records the installed packages against the committed lock and states a
  set that differs (the homelab's: huggingface-hub 1.33.0, click 8.5.0, hf-xet
  1.7.0, aiofiles and ffmpy left over), never hides it, and is skipped, so an
  acceptance is incomplete, without a lock to compare with;
- a temporary directory too long for a DataLoader worker's socket is said;
- the M5 settings are the probe's own while it runs and are put back, so an
  operator's shell holding the deployment's settings changes nothing;
- the service is started as the launcher starts it, with the probe's settings;
- D8's, CORS's and the route guard's checks each fail on the answer they exist
  to catch.

Module: tests/unit/test_probe_m5_acceptance.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
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
    import importlib.metadata

    fakes = [SimpleNamespace(metadata={"Name": name}, version=version)
             for name, version in distributions]
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: iter(fakes))


def test_a_set_that_differs_from_the_lock_is_stated_never_hidden(probe, tmp_path, monkeypatch):
    """The homelab's check set against the committed lock (M5-0): three
    versions moved, two leftovers of Gradio 5.29.1, the out-of-lock PyG
    install, an extra not installed. A package the lock pins at two versions
    (one per platform) matches either; versions compare as PEP 440 does."""
    lock = _lock(tmp_path / "uv.lock", [
        ("huggingface-hub", "1.16.1"), ("click", "8.3.3"), ("hf-xet", "1.5.0"),
        ("gradio", "6.30.0"), ("numpy", "2.3.4"), ("numpy", "2.2.6"),
        ("torch", "2.10.0+cu130"), ("tqdm", "4.67.1"), ("spacy", "3.8.14")])
    _installed(monkeypatch, [
        ("huggingface_hub", "1.33.0"), ("Click", "8.5.0"), ("hf_xet", "1.7.0"),
        ("gradio", "6.30.0"), ("numpy", "2.2.6"), ("torch", "2.10.0+cu130"),
        ("tqdm", "4.67.1.0"), ("aiofiles", "24.1.0"), ("ffmpy", "1.0.0"),
        ("torch_geometric", "2.8.0.post1"),
        ("gradio", "5.29.1")])            # a second copy later on the path: not the one imported

    recorded = probe.installed_against_lock(lock)

    assert recorded["lock_sha256"] == hashlib.sha256(lock.read_bytes()).hexdigest()
    assert recorded["versions_match_lock"] is False
    assert recorded["version_differs"] == [
        {"name": "click", "installed": "8.5.0", "locked": ["8.3.3"]},
        {"name": "hf-xet", "installed": "1.7.0", "locked": ["1.5.0"]},
        {"name": "huggingface-hub", "installed": "1.33.0", "locked": ["1.16.1"]},
    ]
    assert recorded["installed_not_in_lock"] == [
        {"name": "aiofiles", "version": "24.1.0"}, {"name": "ffmpy", "version": "1.0.0"},
        {"name": "torch-geometric", "version": "2.8.0.post1"}]
    assert recorded["locked_not_installed"] == ["spacy"]
    assert (recorded["lock_packages"], recorded["installed_distributions"]) == (8, 10)
    assert str(tmp_path) not in json.dumps(recorded)


def test_a_matching_set_says_so(probe, tmp_path, monkeypatch):
    lock = _lock(tmp_path / "uv.lock", [("gradio", "6.30.0"), ("torch-geometric", "2.8.0-post1")])
    _installed(monkeypatch, [("gradio", "6.30.0"), ("torch_geometric", "2.8.0.post1")])
    recorded = probe.installed_against_lock(lock)
    assert recorded["versions_match_lock"] is True
    assert recorded["version_differs"] == recorded["installed_not_in_lock"] == []


def test_without_a_lock_the_record_is_skipped_and_the_acceptance_incomplete(probe, tmp_path):
    with pytest.raises(probe.SkipProbe, match="no uv.lock"):
        probe.installed_against_lock(tmp_path / "uv.lock")


def test_a5_states_a_differing_set_on_the_console_and_in_the_environment(
        probe, monkeypatch, capsys):
    differing = {"lock_sha256": "0" * 64, "lock_packages": 2, "installed_distributions": 2,
                 "versions_match_lock": False,
                 "version_differs": [{"name": "click", "installed": "8.5.0",
                                      "locked": ["8.3.3"]}],
                 "installed_not_in_lock": [{"name": "aiofiles", "version": "24.1.0"}],
                 "locked_not_installed": []}
    monkeypatch.setattr(probe, "installed_against_lock", lambda path: dict(differing))
    report = probe.Report()

    environment = probe.phase_environment(report, "cpu", describe_only=True)

    a5 = {p.probe_id: p for p in report.probes}["A5"]
    assert a5.status == "passed" and "1 installed version(s) differ" in a5.detail
    said = capsys.readouterr().out
    assert "click 8.5.0, lock 8.3.3" in said and "aiofiles 24.1.0" in said
    assert environment["installed_against_lock"]["version_differs"] == differing[
        "version_differs"]
    assert "socket_fits" in environment["installed_against_lock"]["temporary_directory"]


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
    assert all(set(entry) == {"name", "installed", "locked"}
               for entry in recorded["version_differs"])
    assert all(set(entry) == {"name", "version"} for entry in recorded["installed_not_in_lock"])


@pytest.mark.skipif(sys.platform != "linux", reason="Linux's AF_UNIX path limit")
def test_a_temporary_directory_too_long_for_a_worker_socket_is_said(probe, monkeypatch):
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", "/tmp")
    assert probe._socket_room() == {"path_bytes": 4, "socket_path_limit": 107,
                                    "socket_fits": True}
    monkeypatch.setattr(tempfile, "tempdir", "/" + "s" * 75)       # 76 bytes: 108 with the socket
    assert probe._socket_room()["socket_fits"] is False
    monkeypatch.setattr(tempfile, "tempdir", "/" + "s" * 74)       # 75: exactly 107
    assert probe._socket_room()["socket_fits"] is True


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


def _response(status, body):
    return (status, {}, body.encode("utf-8"))


def _refused(condition="forwarding_header", **change):
    seen = {
        "access": {"enabled": False, "condition": condition, "reason": "words"},
        "listed": _response(403, '{"error": "Refused: words. Nothing was done."}'),
        "stop": _response(403, '{"error": "Refused: words. Nothing was done."}'),
        "event": '["Refused: words. Nothing was done.", null]',
    }
    seen.update(change)
    return seen


def _allowed(**change):
    seen = {
        "access": {"enabled": True, "condition": "enabled", "reason": "words"},
        "listed": _response(200, json.dumps({"checkpoints": [{"sha256": LISTED}]})),
        "stop": _response(409, '{"error": "No training is running"}'),
        "event": '["Not stopped: No training is running", null]',
    }
    seen.update(change)
    return seen


def test_d8_s_refusal_check_passes_a_refusal(probe):
    facts = probe._check_refused(_refused(), "forwarding_header", ("203.0.113.9",), LISTED)
    assert facts == {"condition": "forwarding_header", "checkpoint_list": 403,
                     "training_stop": 403, "gradio_event": "refused"}


@pytest.mark.parametrize("change, message", [
    ({"access": {"enabled": False, "condition": "cross_site"}}, "not forwarding_header"),
    ({"access": {"enabled": True, "condition": "enabled"}}, "not forwarding_header"),
    ({"listed": _response(200, json.dumps({"checkpoints": [{"sha256": LISTED}]}))},
     "the checkpoint list answered 200"),
    ({"stop": _response(409, '{"error": "No training is running"}')},
     "the training stop answered 409"),
    ({"event": '["Not stopped: No training is running", null]'}, "event was not refused"),
    ({"listed": _response(403, f'{{"error": "Refused: {LISTED}. Nothing was done."}}')},
     "showed a checkpoint"),
    ({"stop": _response(403, '{"error": "Refused: from 203.0.113.9. Nothing was done."}')},
     "repeated a header's value"),
])
def test_d8_s_refusal_check_fails_on_what_it_exists_to_catch(probe, change, message):
    with pytest.raises(AssertionError, match=message):
        probe._check_refused(_refused(**change), "forwarding_header", ("203.0.113.9",), LISTED)


def test_d8_s_allowance_check_passes_controls_that_ran(probe):
    assert probe._check_allowed(_allowed(), LISTED, "loopback")["gradio_event"] == "ran"


@pytest.mark.parametrize("change, message", [
    ({"access": {"enabled": False, "condition": "deployment_mode"}}, "D8 refused"),
    ({"listed": _response(403, "Refused")}, "answered 403"),
    ({"listed": _response(200, json.dumps({"checkpoints": [{"sha256": "b" * 64}]}))},
     "does not hold the listed checkpoint"),
    ({"stop": _response(403, "Refused")}, "D8 did not let it run"),
    ({"event": '["Refused: words. Nothing was done.", null]'}, "did not run its handler"),
])
def test_d8_s_allowance_check_fails_on_a_refused_control(probe, change, message):
    with pytest.raises(AssertionError, match=message):
        probe._check_allowed(_allowed(**change), LISTED, "loopback")


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
