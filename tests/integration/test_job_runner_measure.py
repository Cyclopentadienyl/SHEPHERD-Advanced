"""
A test run through the job runner, with the real measure_scorer (M5).
=====================================================================
The runner builds the command line from its inputs, the real CLI takes the GPU
lock and measures the fixture cohort on CPU and writes its outcome file
(`--outcome`, in the job's own directory), and once it has ended the runner
reads from that file which report it published and, through the registration's
one helper, whether that report is kept and registered. Expected digests that
are not the files' make the run refuse before measuring: no outcome file, no
report, nothing kept.

Runs once `measure_scorer` takes the expected digests (`--expect-checkpoint-
sha256`, `--expect-version-sha256`, WP4 of M5-1); skipped before that.

> The checkpoint is untrained and the graph synthetic: nothing here is a result
> about the scorer, only about the runner's flow.

Module: tests/integration/test_job_runner_measure.py
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")

from src.api.services.training_manager import TestRunInputs, TrainingManager  # noqa: E402
from src.evaluation.cohort_registry import is_run_id, resolve_version  # noqa: E402
from src.evaluation.report_store import read_usage_history  # noqa: E402
from src.utils.gpu_lock import (  # noqa: E402
    LOCK_FILE_NAME,
    WORKSPACES_ROOT_ENV,
    probe,
)
from tests.fixtures.registered_cohort import LABEL, build_registered_cohort  # noqa: E402


def _cli_takes_expected_digests() -> bool:
    import contextlib
    import io

    import scripts.measure_scorer as ms

    try:
        with contextlib.redirect_stderr(io.StringIO()):
            ms.parse_args(["--checkpoint", "c", "--data-dir", "d", "--cohort", "x",
                           "--registry", "/r", "--expect-checkpoint-sha256", "a" * 64,
                           "--expect-version-sha256", "b" * 64])
    except SystemExit:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _cli_takes_expected_digests(),
    reason="measure_scorer does not take --expect-checkpoint-sha256/--expect-version-sha256 yet")


@pytest.fixture(scope="module")
def cohort(tmp_path_factory):
    return build_registered_cohort(tmp_path_factory.mktemp("runner") / "c")


def _run(cohort, monkeypatch, **change):
    monkeypatch.setenv("SHEPHERD_DEVICE", "cpu")
    # The service's registry setting, which the run inherits: it measures only
    # in the deployment's registry (BACKLOG 37).
    monkeypatch.setenv("SHEPHERD_REGISTRY", str(cohort.registry))
    monkeypatch.setenv(WORKSPACES_ROOT_ENV, str(cohort.data_dir.parent))
    fields = {"registry": cohort.registry, "version_label": LABEL,
              "version_sha256": cohort.version.identity.sha256, "workspace": cohort.data_dir,
              "checkpoint": cohort.checkpoint, "checkpoint_sha256": cohort.checkpoint_sha256}
    asked = []

    def reader(report: Path, **kwargs):
        # The registration's own helper, recording what it was asked.
        from src.evaluation.registration import report_state

        asked.append((report, kwargs["checkpoint"], kwargs["expected_sha256"]))
        return report_state(report, **kwargs)

    manager = TrainingManager(report_state=reader)
    started = manager.start_test(TestRunInputs(**{**fields, **change}))
    assert started["success"], started
    assert manager._last["test"].done.wait(600), "the run did not end"
    return manager, asked


def test_a_real_test_run_keeps_its_report_and_the_runner_finds_it(cohort, monkeypatch):
    reports = cohort.registry / "versions" / LABEL / "reports"
    before = {p.name for p in reports.iterdir()} if reports.is_dir() else set()
    manager, asked = _run(cohort, monkeypatch)
    status = manager.get_test_status()
    log = "\n".join(manager.get_log("test")["lines"])
    assert status["status"] == "completed" and status["exit_code"] == 0, log
    assert "GPU lock" in log and "held" in log, "the child took the lock itself"

    created = sorted(n for n in {p.name for p in reports.iterdir()} - before if is_run_id(n))
    assert len(created) == 1
    run_dir = reports / created[0]
    report = run_dir / "report.json"
    entries, problems = read_usage_history(resolve_version(cohort.registry, LABEL))
    assert not problems
    assert [e["kind"] for e in entries if e.get("run_id") == created[0]] == ["test_run"], \
        "a run started by the runner counts in the usage history like one from the CLI"
    assert status["outcome"]["report_dir"] == str(run_dir)
    published = status["outcome"]["run_outcome"]["report_sha256"]
    assert published == hashlib.sha256(report.read_bytes()).hexdigest()
    assert asked == [(report, cohort.checkpoint, published)]
    assert status["outcome"]["kept"] is True
    # No --record: the run said so, and the ledger beside the checkpoint holds
    # no record of it.
    assert status["outcome"]["registered"] is False
    assert "--record was not given" in status["outcome"]["summary"]
    assert probe(cohort.data_dir.parent / LOCK_FILE_NAME) is None, "released at the end"


def test_a_digest_that_is_not_the_files_refuses_before_measuring(cohort, monkeypatch):
    manager, asked = _run(cohort, monkeypatch, checkpoint_sha256="0" * 64)
    status = manager.get_test_status()
    assert status["status"] == "failed", manager.get_log("test")["lines"]
    assert status["outcome"]["kept"] is False and asked == []
    assert status["outcome"]["summary"].startswith(
        "No report was published: the run wrote no outcome file")
