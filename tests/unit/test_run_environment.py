"""The code revision and environment a training run records (contract M3a-1).

`src/utils/run_environment.py`. The revision is never guessed and an unknown
dirty state is never "clean" (§8 question 1); the dirty state counts changed
tracked files anywhere and untracked files outside a run's own products; no
file name, path or hostname is recorded; every value is plain data, so a
checkpoint carrying it loads with `weights_only=True`.

Module: tests/unit/test_run_environment.py
"""
from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys

import pytest
import torch

from src.utils.run_environment import (
    assert_primitive,
    code_revision,
    collect_environment,
)

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def _git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.invalid")
    _git(root, "config", "user.name", "t")
    (root / "src").mkdir()
    (root / "src" / "a.py").write_text("x = 1\n")
    (root / "data").mkdir()
    (root / "data" / "tracked.yaml").write_text("k: 1\n")
    (root / ".gitignore").write_text("*.ignored\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    return root


@needs_git
def test_a_clean_checkout_records_its_commit_and_clean(repo):
    record = code_revision(repo)

    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    assert record == {"state": "recorded", "revision": head, "dirty": False,
                      "modified_tracked": 0, "untracked": 0}


@needs_git
def test_a_changed_tracked_file_counts_wherever_it_is(repo):
    (repo / "src" / "a.py").write_text("x = 2\n")
    (repo / "data" / "tracked.yaml").write_text("k: 2\n")

    record = code_revision(repo)

    assert record["dirty"] is True
    assert record["modified_tracked"] == 2


@needs_git
def test_untracked_code_counts_and_a_runs_products_do_not(repo):
    (repo / "src" / "new.py").write_text("y = 1\n")
    ws = repo / "data" / "workspaces" / "w" / "runs" / "hgt" / "r1"
    ws.mkdir(parents=True)
    (ws / "config.yaml").write_text("a: 1\n")
    (repo / "outputs").mkdir()
    (repo / "outputs" / "final_metrics.yaml").write_text("{}\n")
    (repo / "src" / "scratch.ignored").write_text("ignored by .gitignore\n")

    record = code_revision(repo)

    assert record["untracked"] == 1
    assert record["modified_tracked"] == 0
    assert record["dirty"] is True


@needs_git
def test_a_staged_rename_is_one_change(repo):
    _git(repo, "mv", "src/a.py", "src/b.py")

    assert code_revision(repo)["modified_tracked"] == 1


def test_no_repository_is_unrecorded_never_clean(tmp_path):
    record = code_revision(tmp_path)

    assert record == {"state": "unrecorded", "revision": None, "dirty": None,
                      "modified_tracked": None, "untracked": None}


def test_no_git_binary_is_unrecorded(tmp_path, monkeypatch):
    def _missing(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", _missing)

    assert code_revision(tmp_path)["state"] == "unrecorded"


@needs_git
def test_a_status_that_fails_leaves_dirty_unknown(repo, monkeypatch):
    real = subprocess.run

    def _status_fails(cmd, *args, **kwargs):
        if "status" in cmd:
            return subprocess.CompletedProcess(cmd, 128, b"", b"")
        return real(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", _status_fails)
    record = code_revision(repo)

    assert record["state"] == "recorded" and record["revision"]
    assert record["dirty"] is None


def test_the_environment_on_this_host_is_plain_data_and_names_no_path():
    env = collect_environment("cpu")

    assert_primitive(env, "environment")
    assert env["gpu"] is None
    assert env["torch"] == torch.__version__
    assert set(env["tf32"]) == {"matmul_allow_tf32", "cudnn_allow_tf32",
                                "matmul_fp32_precision", "cudnn_fp32_precision"}
    text = json.dumps(env)
    assert "/home/" not in text and "/tmp/" not in text and "site-packages" not in text


def test_the_environment_loads_with_the_safe_loader():
    env = collect_environment("cpu")
    buffer = io.BytesIO()
    torch.save({"environment": env, "producer": code_revision(".")}, buffer)
    buffer.seek(0)

    assert torch.load(buffer, weights_only=True)["environment"] == env


def test_the_gpu_facts_name_the_device_the_run_uses(monkeypatch):
    import src.utils.run_environment as module

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: f"GPU{i}")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i: (12, 1))
    monkeypatch.setattr(module, "_driver_version", lambda i: "580.00")

    assert collect_environment("cuda:1")["gpu"] == {
        "name": "GPU1", "capability": [12, 1], "driver": "580.00"}
    assert collect_environment("cuda")["gpu"]["name"] == "GPU0"


@pytest.mark.parametrize("bad", [
    {"p": __import__("pathlib").Path("/x")},
    {"t": torch.zeros(1)},
    {1: "non-string key"},
    {"nested": [{"d": torch.device("cpu")}]},
    {"tuple": (1, 2)},
    # A str subclass is pickled as its class, which the safe loader refuses.
    {"version": torch.__version__},
])
def test_assert_primitive_refuses_what_is_not_plain_data(bad):
    with pytest.raises(TypeError):
        assert_primitive(bad)


@needs_git
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file names are bytes")
def test_a_file_name_that_does_not_decode_is_counted_not_fatal(repo):
    """`git status -z` reports raw path bytes. One that does not decode in the
    locale's encoding used to raise `UnicodeDecodeError` out of the probe and
    stop the training run whose revision was only being recorded."""
    import os

    os.close(os.open(os.path.join(os.fsencode(repo / "src"), b"caf\xe9.py"),
                     os.O_CREAT | os.O_WRONLY, 0o644))

    record = code_revision(repo)

    assert record["state"] == "recorded"
    assert record["untracked"] == 1 and record["dirty"] is True
    assert "caf" not in json.dumps(record)


def test_output_that_is_not_a_commit_id_is_unrecorded(tmp_path, monkeypatch):
    def _odd(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, b"\xff\xfe not a commit\n", b"")

    monkeypatch.setattr(subprocess, "run", _odd)

    assert code_revision(tmp_path)["state"] == "unrecorded"
