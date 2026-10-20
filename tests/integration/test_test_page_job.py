"""
The Test page's start, outcome and registration with a real test run (M5-2, UI-D).
=================================================================================
`src/webui/components/test_page.py` over the real job runner and the real
`scripts/measure_scorer.py` (CPU, the fixture cohort, an untrained checkpoint):
the page's Start hands the runner the inputs from the server's lists and the
digests shown; once the run has ended, the job card says what the runner read
from disk (kept, not registered without "Register the result"), the version's
report store lists the new report, and the page's Register files it in the
ledger beside the listed checkpoint, after which the page reads it as
registered.

> The checkpoint is untrained and the graph synthetic: nothing here is a result
> about the scorer, only about the page's flow.

Module: tests/integration/test_test_page_job.py
"""
from __future__ import annotations

import json
import shutil

import pytest

pytest.importorskip("torch")
pytest.importorskip("gradio")

from tests.fixtures.operator_access import operator_request  # noqa: E402

RUN = "20261017-091500_run_3fa9c1"
LABEL = "demo-test-1"


@pytest.fixture
def site(tmp_path, monkeypatch):
    from src.webui.components import served_status
    from tests.fixtures.registered_cohort import build_registered_cohort

    cohort = build_registered_cohort(tmp_path / "c")
    workspaces = tmp_path / "ws"
    workspaces.mkdir()
    demo = workspaces / "demo"
    shutil.move(str(cohort.data_dir), demo)
    checkpoints = demo / "runs" / "gat" / RUN / "checkpoints"
    checkpoints.mkdir(parents=True)
    shutil.copy2(cohort.checkpoint, checkpoints / "best.pt")
    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(workspaces))
    monkeypatch.setenv("SHEPHERD_REGISTRY", str(cohort.registry))
    monkeypatch.setenv("SHEPHERD_DEVICE", "cpu")
    monkeypatch.delenv("SHEPHERD_GPU_LOCK_FILE", raising=False)
    monkeypatch.setattr(served_status, "read_status", lambda: (None, "not read"))
    return cohort, checkpoints / "best.pt"


def test_a_run_started_from_the_page_is_kept_shown_and_registered(site, monkeypatch):
    from src.api.services.training_manager import TrainingManager
    from src.webui.components import test_page as tp

    cohort, checkpoint = site
    request = operator_request(monkeypatch)
    page = tp.read_page(request)
    models = tp.read_models("demo", request)
    key = f"gat/{RUN}/best.pt"
    manager = TrainingManager()
    started = tp.start_test_run(page, models, "demo", LABEL, key, False, False, request,
                                manager=manager)
    assert started["success"], started
    assert manager._last["test"].done.wait(600), "the run did not end"

    job = tp.read_job(manager=manager)
    assert job["status"] == "completed", job.get("log")
    assert job["inputs"]["checkpoint_sha256"] == cohort.checkpoint_sha256
    assert job["outcome"]["kept"] is True and job["outcome"]["registered"] is False
    card = tp.job_html(job, tp.path_roots(), ("none", "GPU jobs: none"))
    assert 'data-status="completed"' in card and "Report kept" in card
    assert "Not registered" in card and str(cohort.registry) not in card

    reports = tp.read_reports(LABEL)
    (run,) = reports["runs"]
    assert run["run_id"] == job["outcome"]["report_dir"].rsplit("/", 1)[-1]
    allowed, _why = tp.register_state(page, reports, run["run_id"])
    assert allowed is True
    done, said = tp.register_selected(page, reports, models, "demo", run["run_id"], request)
    assert done is True and said.startswith("Registered as record 1."), said
    ledger = json.loads((checkpoint.parent / "evaluations.json").read_text())
    assert ledger["records"][0]["source_artifact_digest"] == run["state"]["report_sha256"]

    shown = tp.refine_registration(page, tp.read_reports(LABEL), run["run_id"], models)
    assert shown["runs"][0]["state"]["registered"] == "registered"
    assert tp.register_state(page, shown, run["run_id"]) == (False, "Already registered.")
