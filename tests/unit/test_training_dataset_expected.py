"""
The dataset a training job was started for is the one it trains on (review of 8242b47, P2-2)
=============================================================================================
The job runner checked the dataset fingerprint the Training run bar showed
against the workspace's manifest in the backend only; the training process
then read the manifest again and trained on whatever it found, so a workspace
republished between the two was trained on and recorded although the start
was confirmed for another (the reviewer's reproduction:
test_review_boundaries.py::test_training_uses_dataset_different_from_displayed_digest).

The fingerprint now goes to the formal training entry as a command-line
argument (`--expect-dataset-sha256`: neither a configuration key, which the
loader would drop silently, nor a TrainConfig field, which would become a
recorded setting) and is compared inside `read_training_inputs` right after
its one manifest read, before the graph and the samples are read: a mismatch
is `DatasetChanged`, exit 1, no traceback, no run directory. The runner's
early check stays (the API's 409). Through the real runner and the real CLI.

**The comparison is the manifest reader's** (root-cause round, P22-S2):
`read_split_manifest(expected_sha256=)` compares the digest of its one read
before the bytes are decoded and raises the one `DatasetChanged`, also for a
manifest that is gone. It used to be compared by the caller after the parse,
so torn bytes, another schema or a missing manifest reached the child as a
"Training failed" traceback advising a rebuild, while the runner's check on
the same bytes said "the dataset changed". Both now call the reader and say
the same sentence. **Nothing about the workspace is checked before that
read** (verification of the root-cause round, P3-1): `main` stat-ed the
graph and sample files and `train` the directory first, so a workspace gone,
or replaced by an empty one, after the runner's check got "Training data not
found ... Build training data first" from the process.

Module: tests/unit/test_training_dataset_expected.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("torch")

import src.api.services.training_manager as tm  # noqa: E402
from tests.fixtures.trainable_workspace import write_trainable_workspace  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SMALL = dict(device="cpu", num_epochs=1, hidden_dim=16, num_heads=2, num_layers=1,
             conv_type="gat", batch_size=4, num_workers=0, use_amp=False)
CHANGED = "The dataset changed since it was shown; select it again."


@pytest.fixture
def workspace(tmp_path):
    return write_trainable_workspace(tmp_path / "case")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(data: Path, run: Path, shown: str, monkeypatch, swap=None):
    """A training job through the real runner, `swap` applied when the
    runner starts the process (after its own check passed)."""
    real = subprocess.Popen
    seen = {}

    def spawn(argv, **kwargs):
        seen["argv"] = list(argv)
        if swap is not None:
            swap()
        return real(argv, **kwargs)

    monkeypatch.setattr(tm.subprocess, "Popen", spawn)
    manager = tm.TrainingManager()
    started = manager.start_training(dict(data_dir=str(data), run_dir=str(run), **SMALL),
                                     expected_dataset_fingerprint=shown)
    assert started["success"], started
    job = manager._last["training"]
    try:
        assert job.done.wait(180), job.tail(30)
    finally:
        if job.process.poll() is None:
            job.process.kill()
            job.process.wait(timeout=10)
    return job, seen["argv"]


def test_positive_control_the_dataset_shown_is_trained_on_and_recorded(workspace, tmp_path,
                                                                        monkeypatch):
    import torch

    shown = _digest(workspace / "split_manifest.json")
    job, argv = _run(workspace, tmp_path / "run", shown, monkeypatch)
    assert argv[-2:] == ["--expect-dataset-sha256", shown]
    assert job.status == "completed", job.tail(30)
    checkpoint = torch.load(tmp_path / "run" / "checkpoints" / "last.pt", weights_only=True,
                            map_location="cpu")
    assert checkpoint["training_input_digests"]["split_manifest"] == shown
    # A precondition, not a setting: never in what the run records.
    assert "expect" not in json.dumps(checkpoint["config"], default=str)
    assert "expect" not in (tmp_path / "run" / "config.yaml").read_text()
    # The full fingerprint the run read, always in its log.
    assert f"Dataset fingerprint (split_manifest.json sha256): {shown}" in "\n".join(job.tail())


@pytest.mark.parametrize("how", ["overwrite", "atomic-rename"])
def test_the_reviewers_counterexample_a_manifest_replaced_after_the_check_is_refused(
        workspace, tmp_path, monkeypatch, how):
    """The reviewer's case: valid manifest bytes replaced when the runner
    starts the process (a newline appended, the workspace still consistent).
    It used to complete one epoch and record the new digest."""
    path = workspace / "split_manifest.json"
    shown = _digest(path)
    after = path.read_bytes() + b"\n"

    def swap():
        if how == "overwrite":
            path.write_bytes(after)
        else:
            staged = path.with_suffix(".new")
            staged.write_bytes(after)
            os.replace(staged, path)

    job, _ = _run(workspace, tmp_path / "run", shown, monkeypatch, swap)
    log = "\n".join(job.tail())
    assert job.status == "failed" and job.exit_code == 1, log[-3000:]
    assert f"train_model: refused: {CHANGED}" in log
    assert f"(shown: {shown[:12]}...). Nothing was written." in log
    assert "Traceback" not in log
    assert not (tmp_path / "run").exists(), "a refused run left its directory"


def test_a_republish_in_progress_is_said_as_a_changed_dataset(workspace, tmp_path, monkeypatch):
    """The critic's case: the new manifest no longer matches the graph files
    yet (a publish in progress). Compared right after the manifest read, it is
    "the dataset changed", not "the manifest contradicts itself" after the
    graph and samples were read."""
    path = workspace / "split_manifest.json"
    shown = _digest(path)
    text = path.read_text()
    digest = re.findall(r'"([0-9a-f]{64})"', text)[0]
    changed = text.replace(digest, ("0" if digest[0] != "0" else "1") + digest[1:], 1)

    def swap():
        staged = path.with_suffix(".new")
        staged.write_text(changed)
        os.replace(staged, path)

    job, _ = _run(workspace, tmp_path / "run", shown, monkeypatch, swap)
    log = "\n".join(job.tail())
    assert job.status == "failed" and CHANGED in log, log[-3000:]
    assert "contradicts itself" not in log and "Traceback" not in log
    assert not (tmp_path / "run").exists()


def _torn(path: Path) -> None:
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])


def _other_schema(path: Path) -> None:
    manifest = json.loads(path.read_text())
    manifest["schema_version"] = 99
    path.write_text(json.dumps(manifest))


def _workspace_gone(path: Path) -> None:
    path.parent.rename(path.parent.with_name(f"{path.parent.name}-moved"))


def _workspace_emptied(path: Path) -> None:
    _workspace_gone(path)
    path.parent.mkdir()


#: Changes the manifest's own parse would refuse (torn mid-overwrite, written
#: by another revision) or not find, the whole workspace gone or replaced by
#: an empty directory included, each said as a changed dataset.
UNREADABLE = {"torn": _torn, "other-schema": _other_schema, "gone": Path.unlink,
              "workspace-gone": _workspace_gone, "workspace-emptied": _workspace_emptied}
#: The cases that leave no manifest at all.
GONE = {"gone", "workspace-gone", "workspace-emptied"}


@pytest.mark.parametrize("how", sorted(UNREADABLE))
def test_a_manifest_that_no_longer_parses_is_said_as_changed_by_both_checks(
        workspace, tmp_path, monkeypatch, how):
    """P22-S2: the job runner's check passed, then the manifest became bytes
    the reader cannot parse, or went away, before the process read it. The
    process refuses it as changed (exit 1, no traceback, no run directory),
    in the words the runner's own check uses on the same workspace."""
    path = workspace / "split_manifest.json"
    shown = _digest(path)

    job, _ = _run(workspace, tmp_path / "run", shown, monkeypatch,
                  lambda: UNREADABLE[how](path))
    log = "\n".join(job.tail())
    assert job.status == "failed" and job.exit_code == 1, log[-3000:]
    assert f"train_model: refused: {CHANGED}" in log, log[-3000:]
    assert "Traceback" not in log and "Training failed" not in log and "Rebuild" not in log
    assert "not found" not in log and "Build training data" not in log
    assert not (tmp_path / "run").exists(), "a refused run left its directory"
    runner = tm.TrainingManager().dataset_refusal(str(workspace), shown)
    assert runner.endswith(" Nothing was started.")
    assert f"train_model: refused: {runner[:-len(' Nothing was started.')]}" in log


@pytest.mark.parametrize("how", sorted(UNREADABLE))
def test_the_reader_compares_before_it_decodes(workspace, monkeypatch, how):
    import src.kg.artifacts as artifacts

    path = workspace / "split_manifest.json"
    shown = _digest(path)
    UNREADABLE[how](path)
    monkeypatch.setattr(artifacts.json, "loads", lambda *a, **k: pytest.fail("decoded"))
    with pytest.raises(artifacts.DatasetChanged, match=re.escape(CHANGED)) as refused:
        artifacts.read_split_manifest(workspace, expected_sha256=shown)
    assert isinstance(refused.value, ValueError)
    assert f"(shown: {shown[:12]}...)" in str(refused.value)
    assert ("is gone" in str(refused.value)) == (how in GONE)


def test_the_reader_reads_the_manifest_shown_as_before(workspace):
    from src.kg.artifacts import read_split_manifest

    shown = _digest(workspace / "split_manifest.json")
    read = read_split_manifest(workspace, expected_sha256=shown)
    assert read.identity.sha256 == shown and read.manifest["schema_version"] == 3


def test_the_comparison_comes_before_the_graph_and_the_samples_are_read(workspace,
                                                                        monkeypatch):
    import scripts.train_model as train_model
    import src.kg.storage.file_storage as storage

    monkeypatch.setattr(storage, "read_graph_artifacts",
                        lambda *a, **k: pytest.fail("the graph was read"))
    monkeypatch.setattr(storage, "read_samples", lambda *a, **k: pytest.fail("samples read"))
    with pytest.raises(train_model.DatasetChanged, match=re.escape(CHANGED)):
        train_model.read_training_inputs(workspace, expected_manifest_sha256="0" * 64)


def test_the_comparison_comes_before_the_parent_and_any_output(workspace, tmp_path):
    """A wrong expectation with a parent that does not exist is refused for
    the dataset: the parent is never reached, the run directory never made."""
    import scripts.train_model as train_model

    config = train_model.TrainConfig(data_dir=str(workspace), device="cpu",
                                     resume_from=str(tmp_path / "missing.pt"),
                                     run_dir=str(tmp_path / "run"))
    with pytest.raises(train_model.DatasetChanged):
        train_model.train(config, set(), expected_dataset_sha256="0" * 64)
    assert not (tmp_path / "run").exists()


def _cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(REPO / "scripts" / "train_model.py"), *args],
                          cwd=REPO, capture_output=True, text=True, timeout=300)


def test_the_cli_refuses_a_wrong_expectation_before_anything_is_written(workspace, tmp_path):
    done = _cli("--data-dir", str(workspace), "--device", "cpu", "--epochs", "1",
                "--hidden-dim", "16", "--num-layers", "1", "--batch-size", "4",
                "--run-dir", str(tmp_path / "run"), "--expect-dataset-sha256", "0" * 64)
    assert done.returncode == 1, done.stderr[-2000:]
    assert f"train_model: refused: {CHANGED}" in done.stderr and "Traceback" not in done.stderr
    assert not (tmp_path / "run").exists()


def test_the_cli_refuses_a_torn_manifest_as_changed(workspace, tmp_path):
    """The formal entry run by hand with the fingerprint it was given, after
    the manifest was torn (P22-S2): refused as changed, not a traceback."""
    path = workspace / "split_manifest.json"
    shown = _digest(path)
    _torn(path)
    done = _cli("--data-dir", str(workspace), "--device", "cpu", "--epochs", "1",
                "--run-dir", str(tmp_path / "run"), "--expect-dataset-sha256", shown)
    assert done.returncode == 1, done.stderr[-2000:]
    assert f"train_model: refused: {CHANGED}" in done.stderr
    assert "Traceback" not in done.stderr and "not UTF-8 JSON" not in done.stderr
    assert not (tmp_path / "run").exists()


def test_without_an_expectation_a_missing_workspace_is_the_manifest_readers_refusal(tmp_path):
    """The CLI by hand naming no workspace that exists: nothing stats the
    directory first (P3-1), so the manifest reader refuses it, with the
    command that builds one, and nothing is written."""
    done = _cli("--data-dir", str(tmp_path / "nowhere"), "--device", "cpu",
                "--run-dir", str(tmp_path / "run"))
    assert done.returncode == 1, done.stderr[-2000:]
    assert f"{tmp_path / 'nowhere'} has no split_manifest.json" in done.stderr
    assert "scripts/build_knowledge_graph.py --generate-samples" in done.stderr
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("bad", ["abc", "A" * 64, "g" * 64, ""])
def test_the_cli_rejects_a_malformed_expectation_at_parsing(workspace, bad):
    done = _cli("--data-dir", str(workspace), "--expect-dataset-sha256", bad)
    assert done.returncode == 2 and "is not a SHA-256" in done.stderr


def test_without_an_expectation_the_process_is_given_none(workspace):
    """A caller that shows no dataset (an API client that sends no
    fingerprint) starts the run as before; the CLI then trains on the
    manifest it reads and records that one."""
    from tests.fixtures.job_processes import FakeProcess

    with patch("subprocess.Popen", return_value=FakeProcess(pid=8)) as popen:
        result = tm.TrainingManager().start_training({"data_dir": str(workspace)})
    assert result["success"] and "--expect-dataset-sha256" not in popen.call_args[0][0]
