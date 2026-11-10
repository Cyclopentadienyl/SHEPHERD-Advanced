"""The deployment probe's research comparison probes: their parts that are not the phase.

Research comparison Stage 10 (the owner's O9). The phase itself (L1-L8 after
W11 in phase M) runs end to end in tests/unit/test_probe_m2_phase.py. This
shows, on CPU, that the checks those probes rest on fire on what they exist to
catch:

- **A6** requires A1's components: it passes for whatever versions, digests
  and routes are installed (another build, a route recorded or not) and never
  compares one with anything; it fails when A1 did not describe the runtime,
  records no component, or records one outside the shape every run's
  environment records;
- **an LP run's outputs verify each other** (`_lp_scores_verified`, L2, L3,
  L5): a real run of a base on a KG edge set passes; a scores file whose bytes
  are not the ones its report names, a raw score that is not the model's score
  of its pair (its file rebound to the report, so only the recomputation can
  tell), rows out of the pairs file's order, and a summary its scores file
  does not give each fail, naming what disagrees;
- **an LP run's report** (`_lp_run_facts`) must read the exposure the model's
  record gives, so a base's figure claimed after full-view exposure fails;
- **a record is found by its run** (`_registered_record`): two runs whose
  reports are equal bytes are two records, each found as its own.

Module: tests/unit/test_probe_research_comparison.py
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

LP = "kg_link_prediction"
LABEL = "kg-test"


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


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- A6
def _environment(probe, monkeypatch, components):
    """Phase A on the CPU with A1's components `components` (the collector
    replaced) and A5's record a fixed one; A6's probe."""
    import src.utils.run_environment as run_environment

    monkeypatch.setattr(run_environment, "collect_components", lambda: components)
    monkeypatch.setattr(probe, "installed_against_lock",
                        lambda path: {"version_differs": [], "installed_not_in_lock": []})
    report = probe.Report()
    probe.phase_environment(report, "cpu")
    return {p.probe_id: p for p in report.probes}["A6"]


def _component(**observed):
    from src.utils.run_environment import PYG_NATIVE_KIND, component_facts

    return component_facts(PYG_NATIVE_KIND, installed=True, available=True, **observed)


def test_a6_states_each_identity_and_route_and_compares_no_version(probe, monkeypatch):
    """Another build of each, a route recorded for one and not the other: A6
    passes and says so, with the route only where the source is recorded."""
    from src.utils.run_environment import native_identity

    components = {
        "pyg_lib": _component(
            version="0.9.9", native={"state": "recorded", "count": 1,
                                     "aggregate_sha256": "a" * 64},
            source={"state": "recorded", "route": "local build",
                    "requested": {"kind": "registry"}, "resolved": {"version": "0.9.9"},
                    "wheel_sha256": "b" * 64, "compiled_against": {}, "toolchain": None,
                    "flags": None}),
        "torch_scatter": _component(
            version="1.0.0", native={"state": "recorded", "count": 2,
                                     "aggregate_sha256": "c" * 64},
            source={"state": "not recorded", "reason": "the source receipt does not "
                    "describe what is installed: installed version 1.0.0, the receipt's 0.9"}),
    }
    a6 = _environment(probe, monkeypatch, components)
    assert a6.status == "passed", a6.detail
    assert a6.facts["native_identity"] == native_identity(components)
    assert a6.facts["routes"] == {"pyg_lib": "local build", "torch_scatter": "not recorded"}
    assert a6.facts["components"] == 2
    assert "pyg_lib 0.9.9 aaaaaaaaaaaa (local build)" in a6.facts["line"]


def test_a6_passes_on_this_process_s_own_components(probe, monkeypatch):
    from src.utils.run_environment import collect_components, native_identity

    monkeypatch.setattr(probe, "installed_against_lock",
                        lambda path: {"version_differs": [], "installed_not_in_lock": []})
    report = probe.Report()
    probe.phase_environment(report, "cpu")
    a6 = {p.probe_id: p for p in report.probes}["A6"]
    assert a6.status == "passed", a6.detail
    assert a6.facts["native_identity"] == native_identity(collect_components())


@pytest.mark.parametrize("components, says", [
    ({}, "A1 records no named component"),
    ({"pyg_lib": {"kind": "PyG native extension", "requested": "unknown",
                  "installed": True, "available": True, "selected": "unknown",
                  "in_effect": "unknown", "fallback_reason": None, "version": "0.8.0",
                  "native": {"state": "recorded", "count": 1, "aggregate_sha256": "a" * 64},
                  # A route without a recorded source: outside the one shape.
                  "source": {"state": "not recorded", "route": "local build"}}},
     "the component pyg_lib's facts are not in the shape"),
    # Its identity readable, but its states left out: held to the shape, they
    # would read as not observed, so A6 would state facts A1 did not record.
    ({"pyg_lib": {"kind": "PyG native extension", "version": "0.8.0", "fallback_reason": None,
                  "native": {"state": "recorded", "count": 1, "aggregate_sha256": "a" * 64},
                  "source": {"state": "not recorded", "reason": "no receipt"}}},
     "the component pyg_lib's facts change when held to the one shape"),
])
def test_a6_fails_on_components_it_cannot_state(probe, monkeypatch, components, says):
    a6 = _environment(probe, monkeypatch, components)
    assert a6.status == "failed" and says in a6.detail, (a6.status, a6.detail)


def test_a6_fails_when_a1_did_not_describe_the_runtime(probe, monkeypatch):
    import src.utils.version_checker as version_checker

    def broken(force=False):
        raise RuntimeError("no runtime")

    monkeypatch.setattr(version_checker, "probe_runtime", broken)
    monkeypatch.setattr(probe, "installed_against_lock",
                        lambda path: {"version_differs": [], "installed_not_in_lock": []})
    report = probe.Report()
    probe.phase_environment(report, "cpu")
    statuses = {p.probe_id: (p.status, p.detail) for p in report.probes}
    assert statuses["A1"][0] == "error"
    assert statuses["A6"] == ("failed", "A1 did not describe the runtime")


# --------------------------------------------------------------------------- LP runs
@pytest.fixture(scope="module")
def measured(trained_disease_child, tmp_path_factory, probe):
    """The session's demo workspace and base copied, a KG edge set of the
    base's test part imported, and the base measured on it twice with
    --record (two runs whose reports are equal bytes on the CPU)."""
    from src.evaluation.cohort_registry import (
        commit_import,
        prepare_kg_edges_import,
        resolve_version,
    )

    root = tmp_path_factory.mktemp("rc_probe").resolve()
    workspace = Path(shutil.copytree(trained_disease_child["data_dir"], root / "workspace",
                                     ignore=shutil.ignore_patterns("runs")))
    base = workspace / "runs" / "gatv2" / "listed-base" / "checkpoints" / "last.pt"
    base.parent.mkdir(parents=True)
    shutil.copyfile(trained_disease_child["base"], base)
    registry = root / "registry"
    registry.mkdir()
    plan = prepare_kg_edges_import(registry, workspace, base, LABEL, "test")
    commit_import(plan, registry, plan.preview_id)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("SHEPHERD_REGISTRY", str(registry))
        for _ in range(2):
            assert probe._measure_cohort(base, workspace, registry, LABEL, "cpu", "--record",
                                         protocol=LP) == 0
    runs = sorted((registry / "versions" / LABEL / "reports").glob("*/report.json"))
    assert len(runs) == 2
    return {"workspace": workspace, "base": base, "registry": registry,
            "version": resolve_version(registry, LABEL), "runs": [p.parent for p in runs]}


@pytest.fixture
def run_copy(measured, tmp_path):
    """The first run's directory copied, with its report parsed, for a test
    that changes them."""
    run = Path(shutil.copytree(measured["runs"][0], tmp_path / measured["runs"][0].name))
    return run, json.loads((run / "report.json").read_text())


def _verified(probe, measured, run, report):
    return probe._lp_scores_verified(run, report, measured["version"], measured["base"],
                                     measured["workspace"], "cpu")


def test_a_real_lp_run_s_scores_pairs_and_summary_verify_each_other(probe, measured,
                                                                     run_copy):
    run, report = run_copy
    verified = _verified(probe, measured, run, report)
    assert verified["rows"] == measured["version"].manifest["pairs"]["rows"] > 0
    assert verified["largest_raw_score_difference"] == 0.0
    assert verified["metrics_recomputed_from_the_scores"] == sorted(
        report["authoritative_metrics"])


def _scores_table(run, report):
    from src.models.bundle.lp_head import lp_scores_from_table

    return lp_scores_from_table((run / report["scores"]["file"]).read_bytes(),
                                report["scores"]["layout"])


def _rebound(run, report, table):
    """`table` written as the run's scores file and rebound in its report (its
    digest and size), so only what the bytes hold can disagree."""
    data = table.tobytes()
    (run / report["scores"]["file"]).write_bytes(data)
    rebound = copy.deepcopy(report)
    rebound["scores"].update(sha256=_sha(data), bytes=len(data))
    return rebound


def test_a_scores_file_that_is_not_the_report_s_bytes_fails(probe, measured, run_copy):
    run, report = run_copy
    path = run / report["scores"]["file"]
    data = bytearray(path.read_bytes())
    data[-2] ^= 1
    path.write_bytes(bytes(data))
    with pytest.raises(AssertionError, match="the scores file is not the bytes the report names"):
        _verified(probe, measured, run, report)


def test_a_raw_score_that_is_not_the_model_s_score_of_its_pair_fails(probe, measured,
                                                                     run_copy):
    """The scores file rebound to its report, its summary still true to it
    for the ranking metrics: only the recomputation through the LP core tells
    that a raw score is not the model's."""
    run, report = run_copy
    table = _scores_table(run, report)
    table["positive_raw"][0] += 0.5
    rebound = _rebound(run, report, table)
    with pytest.raises(AssertionError, match="are not the model's scores of the pairs"):
        _verified(probe, measured, run, rebound)


def test_rows_out_of_the_pairs_file_s_order_fail(probe, measured, run_copy):
    run, report = run_copy
    table = _scores_table(run, report)
    table["pair_row"] = table["pair_row"][::-1].copy()
    with pytest.raises(AssertionError, match="rows are not the pairs file's, in its order"):
        _verified(probe, measured, run, _rebound(run, report, table))


def test_a_summary_its_scores_file_does_not_give_fails(probe, measured, run_copy):
    run, report = run_copy
    changed = copy.deepcopy(report)
    changed["authoritative_metrics"]["lp_accuracy"] += 0.25
    with pytest.raises(AssertionError, match="the report's lp_accuracy is not the one its "
                                             "scores file gives"):
        _verified(probe, measured, run, changed)


def test_a_report_must_read_the_exposure_its_model_s_record_gives(probe, measured):
    from src.evaluation.caveats import AFTER_FULL_VIEW_EXPOSURE, HELD_OUT

    run = measured["runs"][0]
    sha = _sha(measured["base"].read_bytes())
    facts = probe._lp_run_facts(run, measured["version"], measured["base"], sha,
                                measured["workspace"], "cpu", stage="base",
                                exposure_state=HELD_OUT)
    assert facts["exposure"]["state"] == HELD_OUT and facts["run_id"] == run.name
    with pytest.raises(AssertionError, match="the exposure reads 'held_out', not "
                                             "'after_full_view_exposure'"):
        probe._lp_run_facts(run, measured["version"], measured["base"], sha,
                            measured["workspace"], "cpu", stage="base",
                            exposure_state=AFTER_FULL_VIEW_EXPOSURE)


def test_two_runs_with_equal_report_bytes_are_two_records_each_found_by_its_run(probe,
                                                                               measured):
    first, second = measured["runs"]
    report_sha = _sha((first / "report.json").read_bytes())
    assert _sha((second / "report.json").read_bytes()) == report_sha
    sha = _sha(measured["base"].read_bytes())
    found = [probe._registered_record(measured["base"], report_sha, run.name,
                                      measured["version"], sha) for run in (first, second)]
    # Each run its own record, registered one after the other (in the order
    # the runs ran, which their directories' names need not sort into).
    assert [entry["run_id"] for entry in found] == [first.name, second.name]
    assert sorted(entry["sequence"] for entry in found) == [1, 2]
    assert {entry["view"] for entry in found} == {"KgEdgesRecordView"}
    with pytest.raises(AssertionError, match="the ledger holds 0 records of the kept run"):
        probe._registered_record(measured["base"], report_sha, "20260101T000000Z-x-y",
                                 measured["version"], sha)
