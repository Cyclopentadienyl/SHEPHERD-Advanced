"""
The resume parent a training job was started for is the one it continues from (P22-S1)
=======================================================================================
Root-cause round of the review of 8242b47. Continue Training confirmed the
parent (the Continue drawer showed its model fingerprint, and the console
compared the bytes it read with the run list's digest), but only its path
went to the job runner. The runner previewed the parent again and decided
the run's placement under the parent's architecture, the API's cross-field
verdict and the extend mode from that read; the training process then read
the path a third time and continued from whatever bytes were there. Swapping
another paused run's ``last.pt`` in at the start was continued from and
recorded, under the first parent's architecture directory.

Now the parent's SHA-256 is bound to every read the run rests on, by the one
reader (`read_checkpoint(expected_sha256=)`, compared before the file is
loaded; `CheckpointChanged` carries the facts, and each caller words its own
refusal from them, with the action for where the fingerprint was shown):

- the console records the digest shown (the run list's; a typed path's
  Check) and sends it; it compares nothing itself;
- `preview_continuation(expected_sha256=)` is the one place a caller's digest
  is compared (`start_training(expected_parent_sha256=)`, the API's optional
  query parameter);
- `start_training` always passes ``--expect-resume-sha256`` with the digest
  of its own preview read, so the process continues from exactly the parent
  its decisions came from, whether or not the caller sent a digest.

Through the real runner, the real training process, the real route and the
console's handlers. The runner's and the process's tests continue ShepherdGNN
runs, which the API and the CLI still continue until K7; the console's
continue model bundle bases, since the dashboard offers model records only
(K6 decision 1).

Module: tests/unit/test_training_parent_expected.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("torch")
pytest.importorskip("gradio")

import src.api.services.training_manager as tm  # noqa: E402
from tests.fixtures.operator_access import (  # noqa: E402
    OPERATOR_BASE_URL,
    gradio_request,
    operator_mode,
)

REPO = Path(__file__).resolve().parents[2]
#: The refusal's lead, the job runner's preview's and the training process's.
CHANGED = "The checkpoint changed since it was chosen"
#: The training process's refusal: the lead, its action, then the reader's facts.
PROCESS_CHANGED = f"train_model: refused: {CHANGED}; choose it again."
#: The console's action, for where the fingerprint was shown: the run list's
#: digest is sent again until the list is read again; a typed path's is its
#: last Check's (verification of the root-cause round, P3-3).
REFRESH = "Refresh the list, then choose it again."
CHECK_AGAIN = "Click Check checkpoint to read the file there now."
TINY = {"device": "cpu", "hidden_dim": 16, "num_heads": 2, "num_layers": 1, "num_workers": 0,
        "use_amp": False, "batch_size": 4, "num_epochs": 2, "warmup_steps": 1,
        "pause_after_epoch": 1}
#: Three paused runs on one workspace: two of the same architecture, one not.
RUNS = {"a": ("gat", 1), "same": ("gat", 2), "other": ("sage", 3)}


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A workspace a workspaces root lists (``data``), with three paused runs
    trained for real on the CPU in its run layout."""
    import scripts.train_model as train_model
    from tests.fixtures.trainable_workspace import write_trainable_workspace

    root = tmp_path_factory.mktemp("parents")
    ws = write_trainable_workspace(root)
    for name, (arch, seed) in RUNS.items():
        run = ws / "runs" / arch / f"20261001-00000{seed}_{name}_aaaaa{seed}"
        train_model.train(train_model.TrainConfig(data_dir=str(ws), run_dir=str(run),
                                                  conv_type=arch, seed=seed, **TINY))
    return root


@pytest.fixture
def case(trained, tmp_path, monkeypatch):
    """A copy of the trained workspace per test (the tests swap its files),
    listed by the server."""
    root = tmp_path / "root"
    shutil.copytree(trained, root)
    ws = root / "data"
    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(root))
    parents = {name: next((ws / "runs" / arch).glob(f"*_{name}_*")) / "checkpoints" / "last.pt"
               for name, (arch, _seed) in RUNS.items()}
    return {"root": root, "ws": ws, "parents": parents,
            "dataset": _sha(ws / "split_manifest.json")}


def _start(manager, case, monkeypatch, swap=None, **kwargs):
    """A continued run through the real runner, `swap` applied when the
    runner starts the process (after its own preview passed)."""
    real = subprocess.Popen
    seen = {}

    def spawn(argv, **popen_kwargs):
        seen["argv"] = list(argv)
        if swap is not None:
            swap()
        return real(argv, **popen_kwargs)

    monkeypatch.setattr(tm.subprocess, "Popen", spawn)
    return manager.start_training(
        {"data_dir": str(case["ws"]), "device": "cpu", "num_workers": 0,
         "resume_from": str(case["parents"]["a"])},
        expected_dataset_fingerprint=case["dataset"], **kwargs), seen


def _wait(manager):
    job = manager._last["training"]
    try:
        assert job.done.wait(240), job.tail(30)
    finally:
        if job.process.poll() is None:
            job.process.kill()
            job.process.wait(timeout=10)
    return job, "\n".join(job.tail())


def _swap(case, name):
    return lambda: shutil.copyfile(case["parents"][name], case["parents"]["a"])


# --------------------------------------------------------------------------- the runner and the process
def test_positive_control_the_parent_shown_is_continued_and_recorded(case, monkeypatch):
    import torch

    shown = _sha(case["parents"]["a"])
    manager = tm.TrainingManager()
    started, seen = _start(manager, case, monkeypatch, expected_parent_sha256=shown)
    assert started["success"], started
    assert seen["argv"][4:6] == ["--expect-resume-sha256", shown]
    job, log = _wait(manager)
    assert job.status == "completed", log[-3000:]
    run = Path(started["run_dir"])
    assert run.parent.name == "gat"
    record = torch.load(run / "checkpoints" / "last.pt", weights_only=True, map_location="cpu")
    assert record["training_input_digests"]["resume_checkpoint"] == shown
    # A precondition, not a setting: never in what the run records.
    assert "expect" not in json.dumps(record["config"], default=str)
    assert "expect" not in (run / "config.yaml").read_text()


@pytest.mark.parametrize("swapped", ["same", "other"])
def test_a_parent_replaced_after_the_runners_preview_is_refused_by_the_process(
        case, monkeypatch, swapped):
    """The finder's and the skeptic's cases: another paused run's last.pt
    (``same``: the same architecture; ``other``: SAGE where the runner placed
    the run under GAT) put at the parent's path when the process starts. It
    used to complete and record the other parent; it is refused before it is
    loaded, exit 1, no traceback, and no run directory under either
    architecture."""
    shown = _sha(case["parents"]["a"])
    before = set((case["ws"] / "runs").rglob("*"))
    manager = tm.TrainingManager()
    started, _seen = _start(manager, case, monkeypatch, _swap(case, swapped),
                            expected_parent_sha256=shown)
    assert started["success"], started
    job, log = _wait(manager)
    assert job.status == "failed" and job.exit_code == 1, log[-3000:]
    assert PROCESS_CHANGED in log, log[-3000:]
    other = _sha(case["parents"][swapped])
    assert f"holds {other[:12]}..., not the {shown[:12]}... expected. Nothing was written." in log
    assert "Traceback" not in log and "parent accepted:" not in log
    assert not Path(started["run_dir"]).exists(), "a refused run left its directory"
    assert set((case["ws"] / "runs").rglob("*")) == before


def test_without_a_callers_digest_the_process_is_bound_to_the_runners_preview(case,
                                                                              monkeypatch):
    """(a) of the skeptic: a caller that sends no parent digest (an API client)
    still gets a process bound to the parent the runner decided on."""
    shown = _sha(case["parents"]["a"])
    manager = tm.TrainingManager()
    started, seen = _start(manager, case, monkeypatch, _swap(case, "other"))
    assert started["success"], started
    assert seen["argv"][4:6] == ["--expect-resume-sha256", shown]
    job, log = _wait(manager)
    assert job.exit_code == 1 and PROCESS_CHANGED in log, log[-3000:]
    assert not Path(started["run_dir"]).exists()


def test_the_runners_preview_refuses_a_parent_that_is_not_the_one_shown(case, monkeypatch):
    shown = _sha(case["parents"]["a"])
    _swap(case, "same")()
    with patch.object(tm.subprocess, "Popen") as popen:
        manager = tm.TrainingManager()
        result = manager.start_training(
            {"data_dir": str(case["ws"]), "resume_from": str(case["parents"]["a"])},
            expected_parent_sha256=shown)
    assert result["success"] is False and result["error"].startswith(f"{CHANGED}. It holds ")
    assert f"(chosen: {shown[:12]}...)" in result["error"] and result["parent_changed"] is True
    popen.assert_not_called()
    assert manager.run_dir is None


def test_a_parent_digest_without_a_parent_is_refused(case):
    with patch.object(tm.subprocess, "Popen") as popen:
        result = tm.TrainingManager().start_training({"data_dir": str(case["ws"])},
                                                     expected_parent_sha256="a" * 64)
    assert result["success"] is False and "continues no checkpoint" in result["error"]
    popen.assert_not_called()


# --------------------------------------------------------------------------- the preview
def test_the_preview_is_the_one_comparison_of_a_callers_digest(case):
    manager = tm.TrainingManager()
    parent = case["parents"]["a"]
    shown = _sha(parent)
    assert manager.preview_continuation(str(parent), device="cpu",
                                        expected_sha256=shown)["ok"] is True
    malformed = manager.preview_continuation(str(parent), device="cpu", expected_sha256="A" * 64)
    assert malformed["ok"] is False and "not a full SHA-256" in malformed["reason"]
    parent.unlink()
    gone = manager.preview_continuation(str(parent), device="cpu", expected_sha256=shown)
    assert gone == {"ok": False, "changed": True, "parent": str(parent),
                    "reason": f"{CHANGED}. It is gone (chosen: {shown[:12]}...)."}


@pytest.mark.parametrize("how", ["torn", "other-checkpoint", "gone"])
def test_the_reader_compares_before_it_loads(case, monkeypatch, how):
    import torch

    import src.utils.checkpoint_io as checkpoint_io

    parent = case["parents"]["a"]
    shown = _sha(parent)
    if how == "torn":
        data = parent.read_bytes()
        parent.write_bytes(data[: len(data) // 2])
    elif how == "other-checkpoint":
        _swap(case, "same")()
    else:
        parent.unlink()
    found = None if how == "gone" else _sha(parent)
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("loaded"))
    with pytest.raises(checkpoint_io.CheckpointChanged) as refused:
        checkpoint_io.read_checkpoint(parent, map_location="cpu", weights_only=True,
                                      expected_sha256=shown)
    assert isinstance(refused.value, ValueError)
    changed = refused.value
    assert (changed.path, changed.expected, changed.found) == (parent, shown, found)
    # The facts only: what to do is each caller's, by where the digest came from.
    assert str(changed).startswith(str(parent)) and "choose" not in str(changed)
    assert ("is gone" in str(changed)) == (how == "gone")


# --------------------------------------------------------------------------- the CLI
def _cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(REPO / "scripts" / "train_model.py"), *args],
                          cwd=REPO, capture_output=True, text=True, timeout=300)


def test_the_cli_refuses_another_parent_before_anything_is_written(case, tmp_path):
    shown = _sha(case["parents"]["a"])
    _swap(case, "other")()
    done = _cli("--data-dir", str(case["ws"]), "--device", "cpu", "--resume",
                str(case["parents"]["a"]), "--run-dir", str(tmp_path / "run"),
                "--expect-resume-sha256", shown)
    assert done.returncode == 1, done.stderr[-2000:]
    assert PROCESS_CHANGED in done.stderr and "Traceback" not in done.stderr
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("bad", ["abc", "A" * 64, ""])
def test_the_cli_rejects_a_malformed_parent_digest_at_parsing(case, bad):
    done = _cli("--data-dir", str(case["ws"]), "--expect-resume-sha256", bad)
    assert done.returncode == 2 and "is not a SHA-256" in done.stderr


def test_a_parent_digest_for_a_run_that_continues_none_is_refused(case, tmp_path):
    import scripts.train_model as train_model

    config = train_model.TrainConfig(data_dir=str(case["ws"]), device="cpu",
                                     run_dir=str(tmp_path / "run"))
    with pytest.raises(ValueError, match="continues none"):
        train_model.train(config, set(), expected_resume_sha256="a" * 64)
    assert not (tmp_path / "run").exists()


# --------------------------------------------------------------------------- the route
def _client(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.api.routes import training as training_routes

    operator_mode(monkeypatch)
    app = FastAPI()
    app.include_router(training_routes.router, prefix="/api/v1")
    return TestClient(app, base_url=OPERATOR_BASE_URL), training_routes


def test_an_api_start_is_bound_to_the_parent_with_or_without_the_parameter(case, monkeypatch):
    """The route's optional ``expected_parent_sha256``: a different parent is
    refused (409) by the runner's preview; without it, a parent replaced after
    the preview is refused by the process."""
    client, routes = _client(monkeypatch)
    fresh = tm.TrainingManager()
    monkeypatch.setattr(routes, "training_manager", fresh)
    shown = _sha(case["parents"]["a"])
    body = {"data_dir": str(case["ws"]), "device": "cpu", "num_workers": 0,
            "resume_from": str(case["parents"]["a"])}

    with patch.object(tm.subprocess, "Popen") as popen:
        refused = client.post(f"/api/v1/training/start?expected_parent_sha256={'0' * 64}",
                              json=body)
    assert refused.status_code == 409 and CHANGED in refused.text
    popen.assert_not_called()
    assert client.post("/api/v1/training/start?expected_parent_sha256=abc",
                       json=body).status_code == 422

    real = subprocess.Popen

    def spawn(argv, **kwargs):
        _swap(case, "same")()
        return real(argv, **kwargs)

    monkeypatch.setattr(tm.subprocess, "Popen", spawn)
    started = client.post("/api/v1/training/start", json=body)
    assert started.status_code == 200, started.text
    job, log = _wait(fresh)
    assert job.exit_code == 1 and PROCESS_CHANGED in log, log[-3000:]
    assert f"not the {shown[:12]}... expected" in log
    assert not Path(started.json()["run_dir"]).exists()


def test_the_route_passes_the_parent_digest_to_the_service(monkeypatch):
    client, routes = _client(monkeypatch)
    seen = {}

    def start(config, **kwargs):
        seen.update(kwargs)
        return {"success": True}

    monkeypatch.setattr(routes.training_manager, "start_training", start)
    assert client.post("/api/v1/training/start", json={}).status_code == 200
    assert seen["expected_parent_sha256"] is None
    client.post(f"/api/v1/training/start?expected_parent_sha256={'b' * 64}", json={})
    assert seen["expected_parent_sha256"] == "b" * 64


# --------------------------------------------------------------------------- the console
OP = gradio_request()


@pytest.fixture(scope="module")
def bundle_trained(tmp_path_factory):
    """A workspace a workspaces root lists (``data``), with three paused model
    bundle bases trained for real on the CPU (the Continue drawer offers
    model records only)."""
    sys.path.insert(0, str(REPO / "scripts"))
    import train_model
    from tests.fixtures.base_workspace import write_base_workspace
    from tests.unit.test_train_base_stage import _config

    root = tmp_path_factory.mktemp("bundle_parents")
    ws = write_base_workspace(root)
    for name, (_arch, seed) in RUNS.items():
        run = ws / "runs" / "gatv2" / f"20261001-00000{seed}_{name}_aaaaa{seed}"
        train_model.train(*_config(ws, run, seed=seed, pause_after_epoch=1))
    return root


@pytest.fixture
def bundle_case(bundle_trained, tmp_path, monkeypatch):
    """A copy of the bundle workspace per test (the tests swap its files),
    listed by the server."""
    root = tmp_path / "root"
    shutil.copytree(bundle_trained, root)
    ws = root / "data"
    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(root))
    parents = {name: next((ws / "runs" / "gatv2").glob(f"*_{name}_*")) / "checkpoints" / "last.pt"
               for name in RUNS}
    return {"root": root, "ws": ws, "parents": parents,
            "dataset": _sha(ws / "split_manifest.json")}


@pytest.fixture
def console(bundle_case, monkeypatch):
    """The Training page's session state for the copied workspace, and a
    recorder in place of the start."""
    from src.webui.components import training_console as tc

    operator_mode(monkeypatch)
    ws = tc.workspace_state("data")
    parents, _problem = tc._list_parents(ws)
    started = []

    def start(config, **kwargs):
        started.append({**config, "__kwargs__": kwargs})
        return {"success": True, "pid": 1, "run_dir": "/r"}

    monkeypatch.setattr(tc.training_manager, "start_training", start)
    key = next(k for k in parents if "_a_" in k and k.endswith("/last.pt"))
    return {**bundle_case, "tc": tc, "state": ws, "listed": parents, "key": key,
            "started": started}


def _check(c, source, typed="", mode=None):
    return c["tc"]._on_check_continue(source, c["key"], typed, "cpu", None, c["state"],
                                      c["listed"], mode or {}, OP)


def _continue(c, source, typed="", mode=None):
    return c["tc"]._on_resume(c["state"], "cpu", source, c["key"], typed, None, c["listed"],
                              mode or {}, OP)


@pytest.mark.parametrize("source", ["list", "typed"])
def test_continue_sends_the_parent_shown_and_the_preview_refuses_another(console, source):
    """The swap between Check and the click: the run list's digest (its option
    names it) or a typed path's Check is what was shown; the service's preview
    compares it on its read, so another file at the path starts nothing. The
    console compares nothing itself."""
    tc = console["tc"]
    src = tc.CONTINUE_FROM_LIST if source == "list" else tc.CONTINUE_FROM_PATH
    typed = str(console["parents"]["a"]) if source == "typed" else ""
    shown = _sha(console["parents"]["a"])

    text, _extend, mode, _compare = _check(console, src, typed)
    assert "Checkpoint read; it can continue" in text and mode["parent"]["sha256"] == shown

    # The positive control: nothing changed, the start carries the digest shown.
    said, *_rest = _continue(console, src, typed, mode)
    assert said == tc.notice_html("")
    assert console["started"][-1]["__kwargs__"] == {
        "expected_dataset_fingerprint": console["dataset"], "expected_parent_sha256": shown}

    _swap(console, "same")()
    said, text, _extend, mode = _continue(console, src, typed, mode)
    assert len(console["started"]) == 1, "another parent was started"
    assert "this checkpoint cannot be continued" in said
    assert CHANGED in text and f"(chosen: {shown[:12]}...)" in text
    assert (REFRESH if source == "list" else CHECK_AGAIN) in text
    assert mode["parent"]["refused"] is True and mode["parent"]["sha256"] == shown


def test_a_typed_paths_check_shows_the_file_again(console):
    """A Check is what shows a typed path's parent: after the file changed, a
    new Check reads and shows it (no stale refusal), and Continue then sends
    the new digest."""
    tc = console["tc"]
    typed = str(console["parents"]["a"])
    _text, _e, mode, _c = _check(console, tc.CONTINUE_FROM_PATH, typed)
    _swap(console, "same")()
    text, _e, mode, _c = _check(console, tc.CONTINUE_FROM_PATH, typed, mode)
    now = _sha(console["parents"]["a"])
    assert "Checkpoint read; it can continue" in text and mode["parent"]["sha256"] == now
    _continue(console, tc.CONTINUE_FROM_PATH, typed, mode)
    assert console["started"][-1]["__kwargs__"]["expected_parent_sha256"] == now


def test_a_listed_parent_whose_bytes_changed_since_it_was_listed_is_refused(console):
    """The run list carries each file's digest (its option shows it); a Check
    after the file changed is refused by the service's preview."""
    tc = console["tc"]
    shown = _sha(console["parents"]["a"])
    _swap(console, "other")()
    text, _e, mode, _c = _check(console, tc.CONTINUE_FROM_LIST)
    assert CHANGED in text and f"(chosen: {shown[:12]}...)" in text and REFRESH in text
    assert console["key"] in text, "the refused check names the file by its list key"
    assert mode["parent"]["refused"] is True


def test_a_listed_parent_refused_as_changed_is_shown_again_by_refresh(console):
    """The action said is the one that helps (verification P3-3): choosing
    the same option from the same list sends the list's digest again and is
    refused again; Refresh reads the list again, and the file there now is
    checked and shown."""
    tc = console["tc"]
    _swap(console, "same")()
    text, _e, mode, _c = _check(console, tc.CONTINUE_FROM_LIST)
    assert CHANGED in text and REFRESH in text
    mode, _compare, _details = tc._on_parent_chosen(tc.CONTINUE_FROM_LIST, console["key"], "",
                                                    console["listed"], console["state"],
                                                    {**mode, "drawer": True})
    again, _e, mode, _c = _check(console, tc.CONTINUE_FROM_LIST, mode=mode)
    assert CHANGED in again and REFRESH in again
    refreshed, _problem = tc._list_parents(console["state"])
    now = _sha(console["parents"]["a"])
    assert refreshed[console["key"]]["sha256"] == now
    text, _e, mode, _c = tc._on_check_continue(tc.CONTINUE_FROM_LIST, console["key"], "", "cpu",
                                               None, console["state"], refreshed, {}, OP)
    assert "Checkpoint read; it can continue" in text and mode["parent"]["sha256"] == now


def test_a_typed_parent_changed_after_the_consoles_check_is_refused_by_the_start(bundle_case,
                                                                              monkeypatch):
    """Continue Training on a typed path without a Check first: the console's
    own check reads and records the file, and the start's preview compares
    that digest. A file replaced between the two is refused by the start,
    with the typed path's action, and nothing is spawned."""
    from src.webui.components import training_console as tc

    operator_mode(monkeypatch)
    ws = tc.workspace_state("data")
    parents, _problem = tc._list_parents(ws)
    key = next(k for k in parents if "_a_" in k and k.endswith("/last.pt"))
    manager = tm.TrainingManager()
    monkeypatch.setattr(tc, "training_manager", manager)
    real = manager.preview_continuation
    sent = []

    def preview(parent, **kwargs):
        sent.append(kwargs.get("expected_sha256"))
        result = real(parent, **kwargs)
        if len(sent) == 1:
            _swap(bundle_case, "same")()
        return result

    monkeypatch.setattr(manager, "preview_continuation", preview)
    shown = _sha(bundle_case["parents"]["a"])
    with patch.object(tm.subprocess, "Popen") as popen:
        said, *_rest = tc._on_resume(ws, "cpu", tc.CONTINUE_FROM_PATH, key,
                                     str(bundle_case["parents"]["a"]), None, parents, {}, OP)
    assert sent == [None, shown]
    assert CHANGED in said and CHECK_AGAIN in said, said
    popen.assert_not_called()
