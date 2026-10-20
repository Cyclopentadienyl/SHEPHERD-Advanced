"""
The model status shows the loaded model's recorded results (M4; PLAN_TEST_RESULTS D8).
======================================================================================
`_on_load_status` and `_on_reload_pipeline` append a "Recorded test results
(Mode C)" block to the status Markdown, from one GET of
`/api/v1/evaluations/loaded-model`. What is pinned here:

- each state the API answers with renders its sentence, and an answer about
  another checkpoint than the status names is said to be one, never shown;
- a record shows its role and version (the label and the version's manifest
  digest, so two imports of one label read as two versions), Mode C metrics
  over the cases ranked, coverage with every denominator, relation states,
  R10's four labels (an unverifiable label as words, never a number, with the
  reason its scope's evidence state gives), environment class and GPU, dataset
  origin (synthetic called out), registration and corroboration;
- the Mode C MRR is labelled apart from the trainer's "Val MRR", the §6 line is
  shown once for the block, and nothing says "nothing was checked";
- a records read that fails never breaks the status, and with no pipeline loaded
  nothing is looked up; the status stays one Markdown output of the load and
  reload events.

The answers are built by the route's own conversion from records the sidecar
builds, so these tests render the shape the API sends. The route itself is in
`test_loaded_model_evaluations_api.py`.

Module: tests/unit/test_diagnosis_panel_records.py
"""
from __future__ import annotations

import re

import pytest

pytest.importorskip("gradio")
pytest.importorskip("fastapi")

from src.evaluation.caveats import (  # noqa: E402
    MODE_C_RESULT_LABEL,
    PRE_B1_NOTE,
    R10_STATE_TEXT,
    R10_UNVERIFIABLE_REASON,
    RECORD_ABSENCE,
)
from src.evaluation.sidecar import append_record, build_record, empty_ledger  # noqa: E402
from src.webui.components import diagnosis_panel as dp  # noqa: E402
from tests.unit.test_diagnosis_panel import CHECKPOINT_SHA, _loaded_status  # noqa: E402
from tests.unit.test_evaluation_sidecar import (  # noqa: E402
    cohort_version,
    environment_block,
    r10_block,
    registered_report,
)

WHEN = "2026-10-16T12:00:00Z"
OTHER_SHA = "b" * 64
REPORT_SHA = "e" * 64
HEADING = f"**{dp.RECORDED_RESULTS_HEADING}**"
FORBIDDEN = (
    dp.IDENTITY_NOTHING_CHECKED,
    dp.IDENTITY_NOTHING_CHECKED_LOADED,
    dp.IDENTITY_CHECKS_NOT_REPORTED,
    "nothing in it was checked",
    "model_version",
)


# ---------------------------------------------------------------------------
# Answers, built as the route builds them
# ---------------------------------------------------------------------------
def _record(checkpoint=CHECKPOINT_SHA, report_sha=REPORT_SHA, corroborating=0, sequence=1,
            **manifest_overrides):
    """A ledger record as registration writes it, from a report built by
    `MeasurementManifest`."""
    report = registered_report(
        artifact_digests={"checkpoint": checkpoint, "samples": "5" * 64,
                          "node_features": "2" * 64, "edge_indices": "3" * 64,
                          "num_nodes": "4" * 64, "kg": "1" * 64},
        **manifest_overrides,
    )
    ledger = empty_ledger()
    if sequence > 1:
        ledger["records"] = [{"checkpoint_digest": "0" * 64, "cohort_role": "x",
                              "cohort_digest": "0" * 64,
                              "measurement_semantics_digest": "0" * 64,
                              "registration": {"sequence": sequence - 1}}]
    _, ledger = append_record(ledger, build_record(report, report_sha), registered_at=WHEN)
    record = ledger["records"][-1]
    record["corroborating_reports"] = ["f" * 64] * corroborating
    return record


def _r10(training, validation):
    """An R10 block (spec §3.2) whose labels' states follow their counts."""
    block = r10_block()

    def scope(evidence, disease, phenotype):
        return {
            "recorded_sha256": "7" * 64,
            "evidence": {"state": evidence, "path": None, "problem": None},
            "disease_overlap": {"state": disease[0], "k": disease[1], "n": 50,
                                "cases": disease[2]},
            "phenotype_set_overlap": {"state": phenotype[0], "k": phenotype[1], "n": 80},
        }

    block["training"] = scope(*training)
    block["validation"] = scope(*validation)
    return block


COUNTED_R10 = _r10(
    ("found", ("overlap", 3, 5), ("none", 0, None)),
    ("found", ("none", 0, 0), ("overlap", 80, None)),
)
UNCOUNTED_R10 = _r10(
    ("not_found", ("unverifiable", None, None), ("unverifiable", None, None)),
    ("not_recorded", ("no_validation_inputs_recorded", None, None),
     ("no_validation_inputs_recorded", None, None)),
)


def _answer(state, records=(), not_shown=0, sha=CHECKPOINT_SHA, problem=None):
    """The route's answer for `state`, validated by its own response model."""
    from src.api.routes.evaluations import LoadedModelEvaluations, _ledger_view, _record_view

    if state in ("no_model", "not_identified"):
        sha, ledger = None, None
    elif state == "no_ledger":
        ledger = _ledger_view()
    elif state == "ledger_not_read":
        ledger = _ledger_view(problem=problem or "evaluations.json is ledger schema 2")
    else:
        ledger = _ledger_view(sha256="d" * 64)
    return LoadedModelEvaluations(
        state=state, checkpoint_sha256=sha, ledger=ledger,
        records=[_record_view(record) for record in records],
        records_not_shown=not_shown,
    ).model_dump(mode="json")


def _block(answer, served=CHECKPOINT_SHA):
    return dp._format_recorded_results(answer, served)


def _line(block, start):
    """The block's line that starts with `start` once its list marker is off."""
    lines = (line.lstrip("- ") for line in block.splitlines())
    return next(line for line in lines if line.startswith(start))


# ---------------------------------------------------------------------------
# 1. Each state
# ---------------------------------------------------------------------------
NO_RECORD_STATES = {
    "no_model": (None, "no trained model is loaded, so there are no recorded results to "
                       "look up."),
    "not_identified": (None, "the loaded model was supplied in memory, not read from a "
                             "checkpoint file, so it has no digest to look recorded "
                             "results up by."),
    "no_ledger": (CHECKPOINT_SHA, "No ledger (`evaluations.json`) is beside the loaded "
                                  "checkpoint, so no recorded result can be shown; that "
                                  "does not mean these weights were never evaluated."),
    "ledger_not_read": (CHECKPOINT_SHA, "The ledger (`evaluations.json`) beside the loaded "
                                        "checkpoint was not read, so none of its records "
                                        "are shown: evaluations.json is ledger schema 2"),
    "no_records": (CHECKPOINT_SHA, RECORD_ABSENCE),
}


@pytest.mark.parametrize("state", sorted(NO_RECORD_STATES))
def test_a_state_without_records_says_what_it_means(state):
    served, sentence = NO_RECORD_STATES[state]

    block = _block(_answer(state), served)

    assert block.startswith(HEADING)
    assert sentence in block
    assert MODE_C_RESULT_LABEL not in block and "Mode C MRR" not in block
    assert dp.IDENTITY_CLAIM_BOUNDARY not in block
    if served is not None:
        assert f"for checkpoint `sha256:{CHECKPOINT_SHA[:12]}…`" in block


def test_a_ledger_not_read_shows_its_problem_and_no_record():
    problem = ("evaluations.json is ledger schema 2; this code reads schema 3. Archive "
               "it, for example as evaluations.v2.json, and register again")

    block = _block(_answer("ledger_not_read", problem=problem))

    assert problem in block
    assert RECORD_ABSENCE not in block, "not read is not an empty ledger"


def test_a_record_shows_every_part():
    record = _record(r10=COUNTED_R10, corroborating=2)

    block = _block(_answer("records", [record]))

    assert block.startswith(f"{HEADING} for checkpoint `sha256:{CHECKPOINT_SHA[:12]}…`")
    assert f"{MODE_C_RESULT_LABEL}. {PRE_B1_NOTE}" in block
    head = _line(block, "**mygene2 · mygene2-2026-10**")
    assert head == (
        f"**mygene2 · mygene2-2026-10** (version `sha256:{'a' * 12}…`) — "
        "Mode C MRR (untruncated) 0.5000 · Mode C mean rank 3.0000 · "
        "Mode C Hits@1 0.2500 · Mode C Hits@10 0.7500, over 80 ranked cases")
    assert _line(block, "Coverage:") == (
        "Coverage: 80 of 100 source cases ranked (80 eligible) · 20 excluded: "
        "ground truth absent 12 · phenotype unmappable 8")
    assert _line(block, "Relations:") == (
        "Relations: R1 graph files match the workspace manifest ✓ · R2 generated split's "
        "samples match the workspace manifest: not applicable · R3 model trained on this "
        "graph ✓ · R5 dataset record bound to this graph ✓ · cohort version mapped "
        "against this graph ✓")
    assert _line(block, "R10, training") == (
        "R10, training inputs (evidence found): Training disease overlap: overlap — 3 of "
        "50 distinct test diseases; 5 test cases with one of them · Training "
        f"phenotype-set overlap: {R10_STATE_TEXT['none']} — 0 of 80 test cases")
    assert _line(block, "R10, validation") == (
        "R10, validation inputs (evidence found): Validation disease overlap: "
        f"{R10_STATE_TEXT['none']} — 0 of 50 distinct test diseases; 0 test cases with "
        "one of them · Validation phenotype-set overlap: overlap — 80 of 80 test cases")
    assert _line(block, "Environment:") == (
        "Environment: CUDA, deterministic algorithms on · GPU NVIDIA GeForce RTX 4090, "
        "driver 560.35")
    assert _line(block, "Dataset:") == (
        "Dataset: **synthetic** — a generated stand-in, not a clinical knowledge graph")
    assert _line(block, "Registered") == (
        f"Registered #1 at {WHEN} · 2 corroborating reports · from report "
        f"`sha256:{REPORT_SHA[:12]}…`")
    assert "phenotype-set overlap ignores order and repetition" in block


def test_a_cpu_run_from_identified_files_says_so():
    record = _record(device="cpu", cuda_executed=False, deterministic_algorithms=False,
                     dataset_record={"binding": "verified", "origin": "files",
                                     "sha256": "d" * 64, "sources": [],
                                     "incomplete_by_design": []},
                     environment={**environment_block(), "facts": {"gpu": None}})

    block = _block(_answer("records", [record]))

    assert _line(block, "Environment:") == (
        "Environment: CPU (development; a different measurement from a CUDA run)")
    assert _line(block, "Dataset:") == "Dataset: files — built from identified source files"
    assert _line(block, "Registered").startswith(f"Registered #1 at {WHEN} · no corroborating report")


def test_a_gpu_fact_that_was_not_read_is_said_to_be_not_read():
    environment = environment_block()
    environment["facts"]["gpu"] = {"name": None, "capability": None, "driver": None}

    block = _block(_answer("records", [_record(environment=environment)]))

    assert "GPU name not read, driver not read" in block


# ---------------------------------------------------------------------------
# 2. Only for the checkpoint the status names
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("answer, served", [
    pytest.param(lambda: _answer("records", [_record(checkpoint=OTHER_SHA)], sha=OTHER_SHA),
                 CHECKPOINT_SHA, id="records of another checkpoint"),
    pytest.param(lambda: _answer("no_records", sha=OTHER_SHA), CHECKPOINT_SHA,
                 id="no records for another checkpoint"),
    pytest.param(lambda: _answer("records", [_record()]), None,
                 id="records while the status names no checkpoint"),
    pytest.param(lambda: _answer("no_model"), CHECKPOINT_SHA,
                 id="no model while the status names one"),
])
def test_an_answer_about_another_checkpoint_is_never_shown(answer, served):
    """A reload landed between the status read and the records read. Catches a
    block that shows the records it got under a status naming other weights."""
    block = _block(answer(), served)

    assert block == f"⚠️ {HEADING}: {dp.RECORDS_MODEL_CHANGED}"


def test_the_served_checkpoint_is_taken_only_from_a_verified_identity():
    status = _loaded_status()
    assert dp._served_checkpoint_sha(status) == CHECKPOINT_SHA

    status["served_identity"]["checkpoint"] = {"state": "supplied_in_memory", "sha256": None}
    assert dp._served_checkpoint_sha(status) is None
    # A digest beside any other state is not a served checkpoint.
    status["served_identity"]["checkpoint"] = {"state": "absent", "sha256": CHECKPOINT_SHA}
    assert dp._served_checkpoint_sha(status) is None
    assert dp._served_checkpoint_sha({"initialized": True}) is None


# ---------------------------------------------------------------------------
# 3. What the block must and must not say
# ---------------------------------------------------------------------------
def test_unverifiable_is_a_word_never_a_number():
    """Contract §5.4: "unverifiable" and "no validation inputs recorded" carry no
    count. Catches a 0 that reads as "no overlap" and a denominator that reads
    as a count."""
    block = _block(_answer("records", [_record(r10=UNCOUNTED_R10)]))

    training, validation = _line(block, "R10, training"), _line(block, "R10, validation")
    assert training == (
        "R10, training inputs (evidence not found): Training disease overlap: "
        f"{R10_UNVERIFIABLE_REASON['not_found']} · Training phenotype-set overlap: "
        f"{R10_UNVERIFIABLE_REASON['not_found']}")
    assert validation == (
        "R10, validation inputs (evidence not recorded by the checkpoint): Validation "
        "disease overlap: no validation inputs recorded · Validation phenotype-set "
        "overlap: no validation inputs recorded")
    assert not re.search(r"\d", training.removeprefix("R10") + validation.removeprefix("R10"))


def _computed_r10(tmp_path, situation):
    """An R10 block from `compute_r10` itself, in one of the situations whose
    *unverifiable* has its own reason (contract §5.4)."""
    import hashlib

    from src.evaluation.r10 import compute_r10
    from src.kg.data_loader import DiagnosisSample

    test = [DiagnosisSample(patient_id="t0", phenotype_ids=[10], disease_id=1)]
    if situation == "graph roles only":
        recorded = {"kg": "1" * 64, "node_features": "2" * 64, "edge_indices": "3" * 64,
                    "num_nodes": "4" * 64}
        return compute_r10(recorded, test, [])
    if situation == "not found":
        return compute_r10({"train_samples": "a" * 64, "val_samples": "b" * 64}, test, [])
    assert situation == "unreadable"
    path = tmp_path / "train_samples.json"
    path.write_bytes(b"not json")
    digest = hashlib.sha256(b"not json").hexdigest()
    return compute_r10({"train_samples": digest, "val_samples": digest}, test, [path])


@pytest.mark.parametrize("situation, evidence", [
    ("graph roles only", "not_recorded"), ("not found", "not_found"),
    ("unreadable", "unreadable"),
])
def test_an_unverifiable_label_gives_its_scopes_reason(tmp_path, situation, evidence):
    """Contract §5.4 gives three reasons, and they call for different actions.
    A checkpoint that records only the graph roles names no samples file, so
    --training-evidence can never help; the line must not tell the operator the
    recorded inputs "were not found". Catches one wording for every
    *unverifiable*, which contradicted the scope's own evidence words."""
    block = _block(_answer("records", [_record(r10=_computed_r10(tmp_path, situation))]))

    for scope, name in (("training", "Training"), ("validation", "Validation")):
        line = _line(block, f"R10, {scope}")
        reason = R10_UNVERIFIABLE_REASON[evidence]
        assert line == (
            f"R10, {scope} inputs (evidence {dp._R10_EVIDENCE_WORDS[evidence]}): "
            f"{name} disease overlap: {reason} · {name} phenotype-set overlap: {reason}")
        assert R10_STATE_TEXT["unverifiable"] not in line
        if evidence == "not_recorded":
            assert "not found" not in line and "could not be read" not in line
            assert "--training-evidence cannot help" in line


def test_the_three_unverifiable_reasons_are_the_evidence_states_without_evidence():
    """One reason for each evidence state that leaves a label unverifiable, and
    none of them a number."""
    from src.evaluation.r10 import EVIDENCE_STATES

    assert set(R10_UNVERIFIABLE_REASON) == set(EVIDENCE_STATES) - {"found"}
    for reason in R10_UNVERIFIABLE_REASON.values():
        assert reason.startswith("unverifiable: ") and "no count" in reason
        assert not re.search(r"\d", reason)


def test_two_imports_of_one_label_read_as_two_versions():
    """A label is unique only within one registry. Two records on two imports of
    it have the same role, label, samples and source and different manifest
    digests; the cohort line shows the digest, so they do not read as one
    cohort measured twice."""
    records = [_record(), _record(cohort_version=cohort_version(manifest_sha256="9" * 64),
                                  sequence=2)]

    block = _block(_answer("records", records))

    heads = [line.lstrip("- ") for line in block.splitlines()
             if line.startswith("- **mygene2 · mygene2-2026-10**")]
    assert [head.split(" — ")[0] for head in heads] == [
        f"**mygene2 · mygene2-2026-10** (version `sha256:{'a' * 12}…`)",
        f"**mygene2 · mygene2-2026-10** (version `sha256:{'9' * 12}…`)",
    ]


def test_the_mode_c_mrr_is_labelled_apart_from_the_trainers_val_mrr(monkeypatch):
    status = _loaded_status(checkpoint_meta={"epoch": 3, "params": 10, "device": "cpu",
                                             "val_mrr": 0.42})
    monkeypatch.setattr(dp, "_get_pipeline_status", lambda: status)
    monkeypatch.setattr(dp, "_get_loaded_model_evaluations",
                        lambda: _answer("records", [_record()]))

    rendered, _, _ = dp._on_load_status()

    status_part, block = rendered.split(HEADING)
    assert "Val MRR: 0.4200" in status_part
    assert "Val MRR" not in block
    assert "Mode C MRR (untruncated) 0.5000" in block
    metrics = _line(block, "**mygene2").split(" — ", 1)[1].split(", over")[0]
    assert all(part.startswith("Mode C ") for part in metrics.split(" · "))


def test_the_claim_boundary_is_shown_once_for_the_block():
    """Not once per record: the same claim for every record."""
    records = [_record(), _record(cohort_version=cohort_version(
        label="mygene2-2026-11", manifest_sha256="9" * 64), sequence=2)]

    block = _block(_answer("records", records))

    assert block.count("**mygene2 · ") == 2
    assert block.count(dp.IDENTITY_CLAIM_BOUNDARY) == 1
    assert block.rstrip().endswith(f"<sub>{dp.IDENTITY_CLAIM_BOUNDARY}</sub>")


ALL_ANSWERS = {
    **{state: (lambda state=state: _answer(state), served)
       for state, (served, _) in NO_RECORD_STATES.items()},
    "records": (lambda: _answer("records", [_record(r10=UNCOUNTED_R10)], not_shown=1),
                CHECKPOINT_SHA),
    "only not shown": (lambda: _answer("records", not_shown=1), CHECKPOINT_SHA),
    "unavailable": (lambda: {"state": "unavailable", "error": "API error 500"}, CHECKPOINT_SHA),
    "changed": (lambda: _answer("no_records", sha=OTHER_SHA), CHECKPOINT_SHA),
}


@pytest.mark.parametrize("name", sorted(ALL_ANSWERS))
def test_the_block_never_says_nothing_was_checked(name):
    """The status's qualifier strings are the identity's, decided by its ✓ marks;
    the block must not add them, and the panel reads no model version."""
    make, served = ALL_ANSWERS[name]

    block = _block(make(), served)

    for text in FORBIDDEN:
        assert text not in block, text


def test_records_not_shown_are_counted_beside_those_shown():
    block = _block(_answer("records", [_record()], not_shown=2))

    assert ("2 records for these weights are not shown: they do not have the shape "
            "registration writes (the server's log says why).") in block
    assert block.count("**mygene2 · ") == 1


def test_records_none_of_which_can_be_shown_are_not_called_absent():
    """The ledger holds records for these weights; saying it holds none would be
    false, and with no number shown there is nothing to label."""
    block = _block(_answer("records", not_shown=1))

    assert ("1 record for these weights is not shown: it does not have the shape "
            "registration writes") in block
    assert RECORD_ABSENCE not in block
    assert MODE_C_RESULT_LABEL not in block and dp.IDENTITY_CLAIM_BOUNDARY not in block


def test_an_unavailable_answer_gives_its_reason():
    block = _block({"state": "unavailable", "error": "API error 500"})

    assert block == f"⚠️ {HEADING} could not be read: API error 500"


@pytest.mark.parametrize("reply", [{}, {"state": "records"}, {"state": "maybe",
                                                              "checkpoint_sha256": None}, []])
def test_a_reply_of_another_shape_is_said_to_be_one(reply):
    block = _block(reply)

    assert block == f"⚠️ {HEADING}: the API's reply was not in a shape this page reads."


# ---------------------------------------------------------------------------
# 4. The read, and the status it sits under
# ---------------------------------------------------------------------------
class _Response:
    def __init__(self, body=None, status=200, json_error=None):
        self._body, self.status_code, self._json_error = body, status, json_error

    def raise_for_status(self):
        if self.status_code >= 400:
            raise dp.requests.HTTPError(str(self.status_code), response=self)

    def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._body


def _serve(monkeypatch, status, evaluations):
    """Answer the tab's GETs: the status, then the records (a reply, or an
    exception to raise). Returns the list of calls."""
    calls = []

    def self_request(method, path, **kwargs):
        calls.append((method, path))
        if path == "/api/v1/pipeline/status":
            return _Response(status)
        if isinstance(evaluations, BaseException):
            raise evaluations
        return evaluations

    monkeypatch.setattr(dp, "_self_request", self_request)
    return calls


def test_the_records_are_one_get_after_the_status(monkeypatch):
    calls = _serve(monkeypatch, _loaded_status(), _Response(_answer("no_records")))

    rendered, _, _ = dp._on_load_status()

    assert isinstance(rendered, str)
    assert calls == [("GET", "/api/v1/pipeline/status"),
                     ("GET", "/api/v1/evaluations/loaded-model")]
    assert rendered.startswith(dp._format_pipeline_status(_loaded_status()))
    assert rendered.index("**Serving:**") < rendered.index(HEADING)
    assert rendered.endswith(RECORD_ABSENCE)


@pytest.mark.parametrize("status", [
    {"initialized": False},
    {"initialized": False, "error": "API not reachable"},
])
def test_nothing_is_looked_up_without_a_loaded_pipeline(monkeypatch, status):
    calls = _serve(monkeypatch, status, AssertionError("no records read is expected"))

    rendered, _, _ = dp._on_load_status()

    assert calls == [("GET", "/api/v1/pipeline/status")]
    assert rendered == dp._format_pipeline_status(status)


@pytest.mark.parametrize("failure, reason", [
    (dp.requests.ConnectionError("refused"), "API not reachable"),
    (dp.requests.ReadTimeout("slow"), "API did not answer within 5 s"),
    (_Response(status=500), "API error 500"),
    (_Response(json_error=dp.requests.exceptions.JSONDecodeError("x", "<html>", 0)),
     "the API's reply was not JSON"),
    (_Response(["not", "an", "object"]), "the API's reply was not an object"),
    (RuntimeError("boom"), "unexpected error: boom"),
])
def test_a_records_read_that_fails_never_breaks_the_status(monkeypatch, failure, reason):
    """The status is rendered whole, and the block says why the records are
    missing. Catches a failure that would replace the status with an error."""
    _serve(monkeypatch, _loaded_status(), failure)

    rendered, _, _ = dp._on_load_status()

    assert rendered == (f"{dp._format_pipeline_status(_loaded_status())}\n\n"
                        f"⚠️ {HEADING} could not be read: {dp._md_text(reason)}")


def test_an_unknown_server_address_is_an_unavailable_answer(monkeypatch):
    from src.utils import server_address

    monkeypatch.setattr(server_address, "_address", None)

    answer = dp._get_loaded_model_evaluations()

    assert answer["state"] == "unavailable"
    assert "no TCP request has been recorded" in answer["error"]


def _reload(monkeypatch, *fields):
    """The Reload handler as Gradio runs it (``async``), for a request D8
    allows (mode M1, loopback)."""
    import asyncio

    from tests.fixtures.operator_access import operator_request

    return asyncio.run(dp._on_reload_pipeline(*fields, operator_request(monkeypatch)))


def test_a_reload_shows_the_new_models_records(monkeypatch):
    """The reload renders the candidate's status from its response; the records
    read after it must be the new model's, or say that they are not."""
    new_status = _loaded_status()
    new_status["served_identity"]["checkpoint"]["sha256"] = OTHER_SHA
    monkeypatch.setattr(dp, "_reload_pipeline",
                        lambda *a, **k: {"success": True, "status": new_status})
    monkeypatch.setattr(dp, "_get_loaded_model_evaluations",
                        lambda: _answer("records", [_record(checkpoint=OTHER_SHA)], sha=OTHER_SHA))

    status_md, *_ = _reload(monkeypatch, "data", "", "auto")

    assert f"{HEADING} for checkpoint `sha256:{OTHER_SHA[:12]}…`" in status_md
    assert "**mygene2 · mygene2-2026-10**" in status_md

    monkeypatch.setattr(dp, "_get_loaded_model_evaluations",
                        lambda: _answer("records", [_record()]))
    status_md, *_ = _reload(monkeypatch, "data", "", "auto")

    assert dp.RECORDS_MODEL_CHANGED in status_md
    assert "**mygene2 · " not in status_md


def test_a_refused_reload_reads_no_records(monkeypatch):
    monkeypatch.setattr(dp, "_reload_pipeline",
                        lambda *a, **k: {"success": False, "message": "refused"})

    def no_read():
        raise AssertionError("a refused reload shows its message, not records")

    monkeypatch.setattr(dp, "_get_loaded_model_evaluations", no_read)

    status_md, message, *_ = _reload(monkeypatch, "data", "", "auto")

    assert status_md == "❌ refused" and message == "refused"
