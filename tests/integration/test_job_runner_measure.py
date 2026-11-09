"""
A test run through the job runner, with the real measure_scorer (M5, K6).
=========================================================================
The runner judges the protocol asked on the listed model record and version
(`src.evaluation.protocols`) and builds the command line from its inputs
(`--mode <protocol>`), the real CLI takes the GPU lock and measures the
fixture cohort on CPU with a disease child trained for one epoch
(`tests/fixtures/trained_children.train_children`, filed under the
workspace's `runs/gatv2`) and writes its outcome file (`--outcome`, in the
job's own directory), and once it has ended the runner reads from that file
which report it published and, through the registration's one helper, whether
that report is kept and registered. Expected digests that are not the files'
make the run refuse before measuring: no outcome file, no report, nothing
kept. A listed record the protocol does not take is refused by the runner
before any job starts; listed fields the files do not bear out are refused by
the run on the bytes it reads, with the definition's sentence.

Runs once `measure_scorer` takes the expected digests (`--expect-checkpoint-
sha256`, `--expect-version-sha256`, WP4 of M5-1); skipped before that.

> The model is trained for one epoch on CPU and the graph is synthetic: nothing
> here is a result about the model, only about the runner's flow.

Module: tests/integration/test_job_runner_measure.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")

from src.api.services.training_manager import TestRunInputs, TrainingManager  # noqa: E402
from src.evaluation import protocols  # noqa: E402
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
            ms.parse_args(["--mode", "disease", "--checkpoint", "c", "--data-dir", "d",
                           "--cohort", "x", "--registry", "/r",
                           "--expect-checkpoint-sha256", "a" * 64,
                           "--expect-version-sha256", "b" * 64])
    except SystemExit:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _cli_takes_expected_digests(),
    reason="measure_scorer does not take --expect-checkpoint-sha256/--expect-version-sha256 yet")


@pytest.fixture(scope="module")
def cohort(tmp_path_factory):
    """The registered fixture cohort, its workspace holding a base and a
    disease child trained for one epoch (`child`, its digest `child_sha256`)."""
    from types import SimpleNamespace

    from src.utils.fingerprint import file_sha256
    from tests.fixtures.trained_children import train_children

    built = build_registered_cohort(tmp_path_factory.mktemp("runner") / "c")
    trained = train_children(built.data_dir, ("disease",))
    return SimpleNamespace(registry=built.registry, version=built.version,
                           data_dir=built.data_dir, base=trained["base"],
                           child=trained["disease"],
                           child_sha256=file_sha256(trained["disease"]))


def _fields(cohort):
    """The inputs as the server's lists give them for the disease child: the
    protocol asked, the listed record's stage and task, the version's counts."""
    return {"registry": cohort.registry, "version_label": LABEL,
            "version_sha256": cohort.version.identity.sha256, "workspace": cohort.data_dir,
            "checkpoint": cohort.child, "checkpoint_sha256": cohort.child_sha256,
            "protocol": "disease", "learning_stage": "finetune", "task": "disease",
            "version_counts": cohort.version.manifest["counts"]}


def _run(cohort, monkeypatch, **change):
    monkeypatch.setenv("SHEPHERD_DEVICE", "cpu")
    # The service's registry setting, which the run inherits: it measures only
    # in the deployment's registry (BACKLOG 37).
    monkeypatch.setenv("SHEPHERD_REGISTRY", str(cohort.registry))
    monkeypatch.setenv(WORKSPACES_ROOT_ENV, str(cohort.data_dir.parent))
    fields = _fields(cohort)
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
    assert (status["inputs"]["protocol"], status["inputs"]["learning_stage"],
            status["inputs"]["task"]) == ("disease", "finetune", "disease")

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
    assert json.loads(report.read_text())["manifest"]["mode"] == "disease", \
        "the report's mode is the protocol asked"
    assert asked == [(report, cohort.child, published)]
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


def _stored(cohort):
    reports = cohort.registry / "versions" / LABEL / "reports"
    return {p.name for p in reports.iterdir()} if reports.is_dir() else set()


@pytest.mark.parametrize("change, says", [
    # The disease child asked as gene: its listed record has the disease head.
    ({"protocol": "gene"},
     "the gene measurement takes a finetune model with the gene head; this checkpoint is a "
     "finetune model with the disease head. It can be measured by disease."),
    # The base asked as disease: its listed record is a base's.
    ({"base": True}, protocols.BASE_MESSAGE),
])
def test_a_listed_record_the_protocol_does_not_take_is_refused_before_a_job_starts(
        cohort, monkeypatch, change, says):
    """The runner judges the protocol asked on the listed record, whoever calls
    it: the definition's reason and "Nothing was started.", and no job."""
    from src.utils.fingerprint import file_sha256

    fields, change = _fields(cohort), dict(change)
    if change.pop("base", False):
        fields.update(checkpoint=cohort.base, checkpoint_sha256=file_sha256(cohort.base),
                      learning_stage="base", task=None)
    fields.update(change)
    before = _stored(cohort)
    spawned = []
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: spawned.append(a) or pytest.fail(
        "a job was spawned"))
    manager = TrainingManager()
    started = manager.start_test(TestRunInputs(**fields))
    assert started["success"] is False and started["error"] == f"{says} Nothing was started."
    assert spawned == [] and manager._last.get("test") is None, "no job was started"
    assert _stored(cohort) == before


@pytest.mark.parametrize("listed, says", [
    # Listed as a gene child on a version that lists genes; the files are the
    # disease child and a version that lists none. The run reads the version
    # first and refuses before it opens the checkpoint.
    ({"protocol": "gene", "task": "gene",
      "version_counts": {"genes": {"cases_with_causal_genes": 1,
                                   "cases_with_candidate_genes": 1}}},
     "this cohort version lists no causal genes, which the gene measurement needs. Import a "
     "source that lists causal_genes. Nothing was written."),
    # Listed as a finetuned disease child; the file is the base.
    ({"base_file": True}, f"{protocols.BASE_MESSAGE} Nothing was written."),
])
def test_listed_fields_the_files_do_not_bear_out_are_refused_by_the_run(
        cohort, monkeypatch, listed, says):
    """The listing passes the runner's judgment; the run judges again on the
    bytes it reads (`--expect-checkpoint-sha256` and `--expect-version-sha256`
    bind them to the choice) and refuses with the definition's sentence before
    anything is written."""
    before = _stored(cohort)
    change = dict(listed)
    if change.pop("base_file", False):
        from src.utils.fingerprint import file_sha256

        change.update(checkpoint=cohort.base, checkpoint_sha256=file_sha256(cohort.base))
    manager, asked = _run(cohort, monkeypatch, **change)
    status = manager.get_test_status()
    log = "\n".join(manager.get_log("test")["lines"])
    assert status["status"] == "failed", log
    assert says in log, log
    assert status["outcome"]["kept"] is False and asked == []
    assert _stored(cohort) == before, "the refused run left a report"
