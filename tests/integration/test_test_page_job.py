"""
The Test page's start, outcome and registration with a real test run (M5-2, UI-D; K6).
=====================================================================================
`src/webui/components/test_page.py` over the real job runner and the real
`scripts/measure_scorer.py` (CPU, the fixture cohort, a base and a disease
child trained for one epoch by `tests/fixtures/trained_children.train_children`
into the workspace's `runs/gatv2`, beside a ShepherdGNN checkpoint kept as
history): the page's Start hands the runner the inputs from the server's lists,
the digests shown and the task the listing's model record names; once the run
has ended, the job card says what the runner read from disk (kept, not
registered without "Register the result") and which task it measured, the
version's report store lists the new report under the task's own metrics, and
the page's Register files it in the ledger beside the listed child, under
`runs/gatv2`, after which the page reads it as registered. The base and the
history file are listed, and Start says why it does not measure them.

> The model is trained for one epoch on CPU and the graph is synthetic: nothing
> here is a result about the model, only about the page's flow.

Module: tests/integration/test_test_page_job.py
"""
from __future__ import annotations

import json
import shutil

import pytest

pytest.importorskip("torch")
pytest.importorskip("gradio")

from tests.fixtures.operator_access import operator_request  # noqa: E402
from tests.fixtures.trained_children import BASE_RUN, CHILD_RUNS  # noqa: E402

#: The ShepherdGNN run kept beside the bundles: listed as history, never run.
HISTORY_RUN = "20261017-091500_run_3fa9c1"
LABEL = "demo-test-1"


@pytest.fixture
def site(tmp_path, monkeypatch):
    from src.webui.components import served_status
    from tests.fixtures.registered_cohort import build_registered_cohort
    from tests.fixtures.trained_children import train_children

    cohort = build_registered_cohort(tmp_path / "c")
    workspaces = tmp_path / "ws"
    workspaces.mkdir()
    demo = workspaces / "demo"
    shutil.move(str(cohort.data_dir), demo)
    history = demo / "runs" / "gat" / HISTORY_RUN / "checkpoints"
    history.mkdir(parents=True)
    shutil.copy2(cohort.checkpoint, history / "best.pt")
    trained = train_children(demo, ("disease",))
    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(workspaces))
    monkeypatch.setenv("SHEPHERD_REGISTRY", str(cohort.registry))
    monkeypatch.setenv("SHEPHERD_DEVICE", "cpu")
    monkeypatch.setattr(served_status, "read_status", lambda: (None, "not read"))
    return cohort, demo, trained["disease"]


def _key(models, architecture, run, filename="last.pt"):
    from src.webui.components import test_page as tp

    (item,) = [i for i in models["items"]
               if (i["architecture"], i["run"], i["filename"]) == (architecture, run, filename)]
    return tp.item_key(item)


def test_a_run_started_from_the_page_is_kept_shown_and_registered(site, monkeypatch):
    from src.api.services.training_manager import TrainingManager
    from src.utils.fingerprint import file_sha256
    from src.webui.components import test_page as tp

    cohort, demo, child = site
    request = operator_request(monkeypatch)
    page = tp.read_page(request)
    models = tp.read_models("demo", request)
    key = _key(models, "gatv2", CHILD_RUNS["disease"])
    binding = tp.workspace_binding(demo, tp.find(page["workspaces"], "name", "demo"))
    assert tp.start_state(page, models, "demo", LABEL, key, binding, ("none", "")) == (True, None)
    manager = TrainingManager()
    started = tp.start_test_run(page, models, "demo", LABEL, key, False, False, request,
                                manager=manager)
    assert started["success"], started
    assert manager._last["test"].done.wait(600), "the run did not end"

    job = tp.read_job(manager=manager)
    assert job["status"] == "completed", job.get("log")
    assert job["inputs"]["checkpoint_sha256"] == file_sha256(child)
    assert job["inputs"]["task"] == "disease"
    assert job["outcome"]["kept"] is True and job["outcome"]["registered"] is False
    card = tp.job_html(job, tp.path_roots(), ("none", "GPU jobs: none"))
    assert 'data-status="completed"' in card and "Report kept" in card
    assert "Disease task" in card
    assert "Not registered" in card and str(cohort.registry) not in card

    reports = tp.read_reports(LABEL)
    (run,) = reports["runs"]
    assert run["run_id"] == job["outcome"]["report_dir"].rsplit("/", 1)[-1]
    assert run["record"]["scoring"]["mode"] == "disease"
    result = tp.result_html(reports, run["run_id"], tp.path_roots())
    assert "Disease pooled MRR" in result and "Mode C" not in result
    assert "disease task: report kept" in tp.reports_html(reports, None)
    allowed, _why = tp.register_state(page, reports, run["run_id"])
    assert allowed is True
    done, said = tp.register_selected(page, reports, models, "demo", run["run_id"], request)
    assert done is True and said.startswith("Registered as record 1."), said
    # The ledger is the child's own, under runs/gatv2 (K5a's layout key).
    ledger_path = demo / "runs" / "gatv2" / CHILD_RUNS["disease"] / "checkpoints" / \
        "evaluations.json"
    assert child.parent / "evaluations.json" == ledger_path
    ledger = json.loads(ledger_path.read_text())
    assert ledger["records"][0]["source_artifact_digest"] == run["state"]["report_sha256"]
    assert ledger["records"][0]["scoring"]["mode"] == "disease"

    shown = tp.refine_registration(page, tp.read_reports(LABEL), run["run_id"], models)
    assert shown["runs"][0]["state"]["registered"] == "registered"
    assert tp.register_state(page, shown, run["run_id"]) == (False, "Already registered.")

    # The M5 close-out on the GB10 (2026-10-25): registered from Result after
    # a run that did not register, the card still read "Not registered" and
    # said in the present tense that no ledger existed, beside Result's
    # "Registered". The card is the run's end, said as such, and points to
    # Result for the registration now; the two no longer disagree.
    card = tp.job_html(tp.read_job(manager=manager), tp.path_roots(), ("none", ""))
    result = tp.result_html(shown, run["run_id"], tp.path_roots())
    assert "registered as record 1" in result
    assert card.index("When the run ended:") < card.index("Not registered")
    assert "The report's registration now is shown in Result." in card
    assert "When the run ended, a report was kept at" in card
    assert "it was not registered (at that time: " in card
    assert "it is not registered" not in card and "there is no" not in card.split(
        "at that time: ", 1)[0]


def test_a_base_and_a_history_file_are_listed_and_never_started(site, monkeypatch):
    """K6 decision 13 on the records the production path wrote: the base and
    the ShepherdGNN checkpoint are listed, led by what they are, and Start
    says why it does not measure them; the start itself refuses them before
    the runner is asked."""
    from src.webui.components import test_page as tp

    _cohort, demo, _child = site
    request = operator_request(monkeypatch)
    page = tp.read_page(request)
    models = tp.read_models("demo", request)
    binding = tp.workspace_binding(demo, tp.find(page["workspaces"], "name", "demo"))
    labels = dict((key, label) for label, key in tp.model_choices(models))
    base = _key(models, "gatv2", BASE_RUN)
    history = _key(models, "gat", HISTORY_RUN, "best.pt")
    child = _key(models, "gatv2", CHILD_RUNS["disease"])
    assert labels[base].startswith("Base · ") and labels[child].startswith("Disease · ")
    assert labels[history].startswith("History · never run · ")

    class Unasked:
        def start_test(self, inputs):
            raise AssertionError("the runner was asked to start a refused checkpoint")

    for key, says in ((base, "this checkpoint is a base: it has no task to measure"),
                      (history, "this checkpoint has no model spec (a ShepherdGNN checkpoint)")):
        allowed, why = tp.start_state(page, models, "demo", LABEL, key, binding, ("none", ""))
        assert allowed is False and why.startswith(f"Disabled: {says}"), why
        result = tp.start_test_run(page, models, "demo", LABEL, key, False, False, request,
                                   manager=Unasked())
        assert result["success"] is False and result["error"].endswith("Nothing was started.")
        assert result["error"].startswith(says[:1].upper() + says[1:])
    # The fixture version lists no genes; the disease task needs none.
    # The child's option names its task; the Measurement row refers to it (rule 1.2).
    assert "Measurement</span><span>same task as the chosen model" in tp.fixed_html(
        page, tp.find_model(models, child))
