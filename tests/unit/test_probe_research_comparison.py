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
  record gives, so a base's figure claimed after full-view exposure fails, and
  must have run on the probe's device (`_ran_on`), so a run recorded off CUDA
  never passes for a CUDA one;
- **the LP core's one summary rule** (`lp_set_metrics`) is the one a run's
  report and the probe's recomputation both give;
- **a record is found by its run** (`_registered_record`): two runs whose
  reports are equal bytes are two records, each found as its own;
- **an exported row holds what its run kept** (`_export_row_verified`, L8):
  real rows of a disease child's supplied and KG edge set runs pass against
  their reports on disk, native identity included; a finetune's weight
  transfer without its parts, a KG edge set's row without its exposure and a
  supplied cohort's without its R10 each fail;
- **the frozen-backbone control's fp32 check** (`_frozen_arm_verified`,
  L7.disease_frozen; ops-wiring round, decision F2): a real paused and
  resumed frozen child of a base passes and says it is fp32 only; a
  graph-side buffer changed (named), downstream parts equal to their start,
  a resume whose downstream parts are the paused file's, an extra parameter
  in the optimizer groups, an optimizer group holding a graph-side parameter
  in place of a downstream one (the count kept), a held parameter without a
  saved state, a full arm's record and mixed precision in effect each fail,
  and so does a CPU run for a CUDA device.

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
    facts, report = probe._lp_run_facts(run, measured["version"], measured["base"], sha,
                                        measured["workspace"], "cpu", stage="base",
                                        exposure_state=HELD_OUT)
    assert facts["exposure"]["state"] == HELD_OUT and facts["run_id"] == run.name
    assert report == json.loads((run / "report.json").read_text())
    assert facts["cuda_executed"] is False
    with pytest.raises(AssertionError, match="the exposure reads 'held_out', not "
                                             "'after_full_view_exposure'"):
        probe._lp_run_facts(run, measured["version"], measured["base"], sha,
                            measured["workspace"], "cpu", stage="base",
                            exposure_state=AFTER_FULL_VIEW_EXPOSURE)


def test_a_run_recorded_off_cuda_never_passes_for_the_probe_s_cuda_device(probe, measured):
    """`cuda_executed` is checked where every kept run's report is read
    (`_ran_on`, in `_lp_run_facts` and the flow's measurement helper), before
    anything else is recomputed: a CPU run read as one of a CUDA flow fails,
    and so does a report claiming CUDA on the CPU."""
    from src.evaluation.caveats import HELD_OUT

    run = measured["runs"][0]
    sha = _sha(measured["base"].read_bytes())
    with pytest.raises(AssertionError, match="the report records cuda_executed False for a "
                                             "run on cuda"):
        probe._lp_run_facts(run, measured["version"], measured["base"], sha,
                            measured["workspace"], "cuda", stage="base",
                            exposure_state=HELD_OUT)
    manifest = json.loads((run / "report.json").read_text())["manifest"]
    probe._ran_on(manifest, "cpu")
    with pytest.raises(AssertionError, match="records cuda_executed True for a run on cpu"):
        probe._ran_on({**manifest, "cuda_executed": True}, "cpu")


def test_a_run_s_summary_is_the_lp_core_s_one_rule_over_its_kept_scores(probe, measured,
                                                                          run_copy):
    """The report's summary is `lp_set_metrics` over the scores the run kept,
    the rule `score_lp_pairs` gives and the probe recomputes by."""
    from src.evaluation.task_measurement import lp_metric_statuses
    from src.models.bundle.lp_head import lp_set_metrics
    from src.models.bundle.record import load_bundle
    from src.utils.checkpoint_io import read_checkpoint

    run, report = run_copy
    table = _scores_table(run, report)
    view = probe._child_view(measured["base"], measured["workspace"])
    payload = read_checkpoint(measured["base"], map_location="cpu",
                              weights_only=True).checkpoint
    bundle, _record = load_bundle(payload, view_digest=view.digest)
    summary = lp_set_metrics(bundle.lp_head, torch.from_numpy(table["positive_raw"].copy()),
                             torch.from_numpy(table["negative_raw"].copy()))
    values, _statuses = lp_metric_statuses(summary, pairs=len(table), not_finite=0)
    assert report["authoritative_metrics"] == values


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


# --------------------------------------------------------------------------- L8 rows
@pytest.fixture(scope="module")
def exported(trained_disease_child, tmp_path_factory, probe):
    """A deployment laid out as L8's (a workspaces root holding one workspace
    that lists the session's base and disease child, and a registry): a KG
    edge set of the base's test part and a supplied cohort imported, the child
    measured on each with --record, and the results export over it. Returns
    the export's rows by kind, and each run's entry as the flow keeps it
    (`_rc_demo`'s `_keep`), built from what is on disk."""
    import scripts.export_results as export_results
    from src.evaluation.cohort_registry import (
        commit_import,
        prepare_kg_edges_import,
        resolve_version,
    )

    root = tmp_path_factory.mktemp("rc_export").resolve()
    workspace = Path(shutil.copytree(trained_disease_child["data_dir"],
                                     root / "workspaces" / "workspace",
                                     ignore=shutil.ignore_patterns("runs")))
    listed = {}
    for run, source in (("listed-base", "base"), ("listed-child", "disease")):
        target = workspace / "runs" / "gatv2" / run / "checkpoints" / "last.pt"
        target.parent.mkdir(parents=True)
        shutil.copyfile(trained_disease_child[source], target)
        listed[source] = target
    registry = root / "registry"
    registry.mkdir()
    plan = prepare_kg_edges_import(registry, workspace, listed["base"], LABEL, "test")
    commit_import(plan, registry, plan.preview_id)
    cases = probe._write_cohort_source(root / "cases.json", probe._m4_demo_cases(workspace))
    supplied, _ = probe._import_version(registry, workspace, cases, "supplied-1")
    child, child_sha = listed["disease"], _sha(listed["disease"].read_bytes())
    entries = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("SHEPHERD_REGISTRY", str(registry))
        patch.setenv("SHEPHERD_WORKSPACES_ROOT", str(root / "workspaces"))
        for label, protocol in ((LABEL, LP), ("supplied-1", "disease")):
            assert probe._measure_cohort(child, workspace, registry, label, "cpu", "--record",
                                         protocol=protocol) == 0
            version = resolve_version(registry, label)
            [path] = (registry / "versions" / label / "reports").glob("*/report.json")
            report = json.loads(path.read_text())
            entries[path.parent.name] = {
                "probe": label, "run_id": path.parent.name,
                "report_sha256": _sha(path.read_bytes()), "checkpoint_sha256": child_sha,
                "version_sha256": version.identity.sha256, "kind": version.kind,
                "role": version.manifest["role"],
                "native_identity": probe._recorded_native_identity(report),
                "components": probe._recorded_components(report)}
        output = root / "export.json"
        assert export_results.main(["--workspaces-root", f"workspaces={root / 'workspaces'}",
                                    "--registry", f"registry={registry}", "--format", "json",
                                    "--output", str(output)]) == 0
    rows = json.loads(output.read_text())["rows"]
    assert {row["run_id"] for row in rows} == set(entries)
    by_kind = {}
    for row in rows:
        by_kind.setdefault(row["data_kind"], []).append(row)
    assert set(by_kind) == {"kg_edges", "supplied"}
    assert supplied.kind == "supplied"
    return {"rows": rows, "by_kind": by_kind, "entries": entries}


def test_every_real_exported_row_holds_what_its_run_kept(probe, exported, monkeypatch):
    """Both runs' rows pass, each held to the native identity its own report
    recorded, as the record views show it: never this process's, so another
    process's components (the job runner's, L3) change nothing."""
    import src.utils.run_environment as run_environment

    monkeypatch.setattr(run_environment, "collect_components", lambda: {})
    for row in exported["rows"]:
        entry = exported["entries"][row["run_id"]]
        assert entry["native_identity"], "the run's report records no component"
        assert set(entry["components"]) == set(entry["native_identity"])
        probe._export_row_verified(row, entry)
        assert row["learning_stage"] == "finetune"


def _case(exported, kind):
    row = copy.deepcopy(exported["by_kind"][kind][0])
    return row, exported["entries"][row["run_id"]]


def test_a_finetune_row_whose_weight_transfer_lists_no_parts_fails(probe, exported):
    row, entry = _case(exported, "supplied")
    assert row["weight_transfer"]["parts"]
    del row["weight_transfer"]["parts"]
    with pytest.raises(AssertionError, match="finetune is exported without its weight "
                                             "transfer"):
        probe._export_row_verified(row, entry)


def test_a_kg_edge_set_row_without_its_exposure_fails(probe, exported):
    row, entry = _case(exported, "kg_edges")
    assert row["exposure_state"] == "after_full_view_exposure"
    row["exposure_state"] = None
    with pytest.raises(AssertionError, match="KG edge set run is exported without its "
                                             "exposure"):
        probe._export_row_verified(row, entry)


def test_a_supplied_row_without_its_r10_fails(probe, exported):
    row, entry = _case(exported, "supplied")
    assert row["r10"] is not None
    row["r10"] = None
    with pytest.raises(AssertionError, match="run is exported without its R10"):
        probe._export_row_verified(row, entry)


def test_a_row_whose_native_identity_is_not_its_report_s_fails(probe, exported):
    row, entry = _case(exported, "kg_edges")
    name = sorted(row["native_identity"])[0]
    row["native_identity"][name]["version"] = "0.0.0-another"
    with pytest.raises(AssertionError, match="another native identity than its report "
                                             "records"):
        probe._export_row_verified(row, entry)


def test_a_row_whose_component_facts_are_not_its_report_s_fails(probe, exported):
    """Review fix RC-P2-1: a row whose component facts differ from what its
    run recorded (here only what was in effect: the native identity is the
    same) fails L8."""
    row, entry = _case(exported, "supplied")
    name = sorted(row["components"])[0]
    assert row["components"][name]["in_effect"] != "yes"
    row["components"][name]["in_effect"] = "yes"
    assert row["native_identity"] == entry["native_identity"]
    with pytest.raises(AssertionError, match="other component facts than its report records"):
        probe._export_row_verified(row, entry)


# --------------------------------------------------------------------------- L7.disease_frozen
@pytest.fixture(scope="module")
def frozen_arms(trained_disease_child, tmp_path_factory, probe):
    """The session's base listed in a copy of its workspace, and from it, by
    the probe's own trainer call (`_train_run`, `train_model.train`), the
    frozen-backbone disease child at the demo's sizes paused after one of two
    epochs, its resume in a run of its own, and a full arm (L7.disease's
    shape: no freeze, one epoch), all on the CPU."""
    from scripts.setup_demo import demo_task_training

    root = tmp_path_factory.mktemp("rc_frozen").resolve()
    workspace = Path(shutil.copytree(trained_disease_child["data_dir"], root / "workspace",
                                     ignore=shutil.ignore_patterns("runs")))
    runs = workspace / "runs" / "gatv2"
    base = runs / "listed-base" / "checkpoints" / "last.pt"
    base.parent.mkdir(parents=True)
    shutil.copyfile(trained_disease_child["base"], base)
    sizes = demo_task_training()["disease"]
    paused = probe._last(probe._train_run(
        workspace, runs / probe.RC_FROZEN_RUNS["paused"], "cpu", stage="finetune",
        task="disease", epochs=probe.RC_FROZEN_EPOCHS,
        settings={**sizes, **probe.RC_FROZEN_SETTINGS}, finetune_from=base,
        pause_after=probe.RC_FROZEN_PAUSE_AFTER))
    resumed = probe._last(probe._train_run(
        workspace, runs / probe.RC_FROZEN_RUNS["resumed"], "cpu", stage="finetune",
        task="disease", resume_from=paused))
    full = probe._last(probe._train_run(workspace, runs / "full", "cpu", stage="finetune",
                                        task="disease", epochs=1, settings=sizes,
                                        finetune_from=base))
    return {"base": base, "paused": paused, "resumed": resumed, "full": full}


def _edited(path: Path, target: Path, change) -> Path:
    """A copy of the checkpoint `path` at `target`, its payload changed by
    `change` (a function of the payload)."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    change(payload)
    torch.save(payload, target)
    return target


def _frozen_verified(probe, arms, **files):
    """`_frozen_arm_verified` on the CPU over `arms`, a file of it replaced
    where `files` names one."""
    chosen = {**arms, **files}
    return probe._frozen_arm_verified(chosen["base"], chosen["paused"], chosen["resumed"], "cpu",
                                      full=chosen["full"])


def test_the_frozen_arm_passes_on_real_runs_and_says_it_is_an_fp32_check(probe, frozen_arms):
    """Ops-wiring round, decision F2: a real paused and resumed frozen child
    of a base passes, its graph side (parameters and BatchNorm buffers) the
    base's, its downstream parts trained, its record kept across the resume,
    its downstream start the full arm's; without a full arm the start is "not
    compared"; the facts say the check is fp32 only."""
    facts = _frozen_verified(probe, frozen_arms)
    assert facts["precision"].startswith("fp32 only") and "not verified" in facts["precision"]
    assert facts["same_downstream_start_as_the_full_arm"] is True
    assert facts["graph_side"]["batch_norm_buffers"] >= 1
    assert facts["paused"]["completed_epochs"] == probe.RC_FROZEN_PAUSE_AFTER
    assert (facts["resumed"]["operation"], facts["resumed"]["completed_epochs"]) == (
        "resume", probe.RC_FROZEN_EPOCHS)
    unpaired = probe._frozen_arm_verified(frozen_arms["base"], frozen_arms["paused"],
                                          frozen_arms["resumed"], "cpu", full=None)
    assert unpaired["same_downstream_start_as_the_full_arm"] == "not compared"
    # A run recorded off CUDA never passes for the probe's CUDA device.
    with pytest.raises(AssertionError, match="environment names no GPU for a run on cuda"):
        probe._frozen_arm_verified(frozen_arms["base"], frozen_arms["paused"],
                                   frozen_arms["resumed"], "cuda", full=None)


def test_a_graph_side_buffer_changed_fails_naming_it(probe, frozen_arms, tmp_path):
    def change(payload):
        key = next(k for k in sorted(payload["state_dict"])
                   if k.split(".", 1)[0] == "encoder" and k.endswith("running_mean"))
        payload["state_dict"][key] = payload["state_dict"][key] + 1.0

    resumed = _edited(frozen_arms["resumed"], tmp_path / "resumed.pt", change)
    with pytest.raises(AssertionError, match=r"the resumed file's graph side changed: "
                                             r"encoder\..*running_mean is not the base's"):
        _frozen_verified(probe, frozen_arms, resumed=resumed)


def test_downstream_parts_equal_to_their_start_fail(probe, frozen_arms, tmp_path):
    from src.models.bundle.builder import DOWNSTREAM_PARTS
    from src.models.bundle.record import load_bundle

    def change(payload):
        bundle, _record = load_bundle(payload, learning_stage="finetune", task="disease")
        payload["weight_transfer"]["downstream_initial_sha256"] = bundle.parts_sha256(
            DOWNSTREAM_PARTS)

    paused = _edited(frozen_arms["paused"], tmp_path / "paused.pt", change)
    with pytest.raises(AssertionError, match="the paused file's downstream parts are still "
                                             "those training started from"):
        _frozen_verified(probe, frozen_arms, paused=paused)


def test_a_resume_whose_downstream_parts_are_the_paused_files_fails(probe, frozen_arms,
                                                                    tmp_path):
    """Review of A2: each run's downstream parts trained, the resumed half's
    too: a resume that learned nothing (its downstream parts the paused
    file's, still unlike their start) fails."""
    paused = torch.load(frozen_arms["paused"], map_location="cpu", weights_only=True)

    def change(payload):
        for key, value in paused["state_dict"].items():
            if key.split(".", 1)[0] in ("patient_encoder", "task_head"):
                payload["state_dict"][key] = value.clone()

    resumed = _edited(frozen_arms["resumed"], tmp_path / "resumed.pt", change)
    with pytest.raises(AssertionError, match="the resumed file's downstream parts are the paused "
                                             "file's: the resumed run trained nothing"):
        _frozen_verified(probe, frozen_arms, resumed=resumed)


def _moment(like):
    return {"step": torch.tensor(1.0), "exp_avg": torch.zeros_like(like),
            "exp_avg_sq": torch.zeros_like(like)}


def test_an_extra_parameter_in_the_optimizer_groups_fails(probe, frozen_arms, tmp_path):
    def change(payload):
        optimizer = payload["optimizer_state_dict"]
        held = [i for group in optimizer["param_groups"] for i in group["params"]]
        extra = max(held) + 1
        weight = next(v for k, v in payload["state_dict"].items()
                      if k.startswith("encoder.") and k.endswith("weight") and v.dim() == 2)
        optimizer["param_groups"][0]["params"].append(extra)
        optimizer["state"][extra] = _moment(weight)

    resumed = _edited(frozen_arms["resumed"], tmp_path / "resumed.pt", change)
    with pytest.raises(AssertionError, match="the resumed file's optimizer groups hold .* "
                                             "parameters, not the .* of the downstream parts"):
        _frozen_verified(probe, frozen_arms, resumed=resumed)


def test_an_optimizer_group_holding_a_graph_side_parameter_fails(probe, frozen_arms, tmp_path):
    """Review of A2: the count kept, a graph-side parameter's moment in place
    of a downstream one's fails by the shapes counted, though its shape is a
    downstream parameter's too (a membership test of the shapes passed)."""
    from src.models.bundle.builder import DOWNSTREAM_PARTS, GRAPH_SIDE_PARTS

    def change(payload):
        state = payload["state_dict"]
        downstream = {tuple(v.shape) for k, v in state.items()
                      if k.split(".", 1)[0] in DOWNSTREAM_PARTS}
        # A graph-side parameter with a downstream parameter's shape (a bias,
        # the LP head's relation weights): only counting the shapes tells it.
        weight = next(v for k, v in sorted(state.items())
                      if k.split(".", 1)[0] in GRAPH_SIDE_PARTS and v.dim()
                      and not k.endswith(("running_mean", "running_var"))
                      and tuple(v.shape) in downstream)
        optimizer = payload["optimizer_state_dict"]
        held = [i for group in optimizer["param_groups"] for i in group["params"]]
        other = next(i for i in held
                     if tuple(optimizer["state"][i]["exp_avg"].shape) != tuple(weight.shape))
        optimizer["state"][other] = _moment(weight)

    resumed = _edited(frozen_arms["resumed"], tmp_path / "resumed.pt", change)
    with pytest.raises(AssertionError, match=r"the resumed file's optimizer holds a parameter of "
                                             r"shape \([\d, ]+\) beyond the downstream parts' "
                                             r"parameters of that shape"):
        _frozen_verified(probe, frozen_arms, resumed=resumed)


def test_a_held_parameter_without_a_saved_state_fails(probe, frozen_arms, tmp_path):
    def change(payload):
        optimizer = payload["optimizer_state_dict"]
        held = [i for group in optimizer["param_groups"] for i in group["params"]]
        del optimizer["state"][held[0]]

    paused = _edited(frozen_arms["paused"], tmp_path / "paused.pt", change)
    with pytest.raises(AssertionError, match="1 of the parameters the paused file's optimizer "
                                             "holds have no saved state"):
        _frozen_verified(probe, frozen_arms, paused=paused)


def test_a_full_arms_record_fails_the_parts_check(probe, frozen_arms):
    with pytest.raises(AssertionError, match="the paused file's record does not mark the graph "
                                             "side frozen"):
        _frozen_verified(probe, frozen_arms, paused=frozen_arms["full"])


def test_mixed_precision_in_effect_fails_as_not_the_fp32_check(probe, frozen_arms, tmp_path):
    def change(payload):
        payload["config"]["effective"]["amp"].update(enabled=True, dtype="float16")

    paused = _edited(frozen_arms["paused"], tmp_path / "paused.pt", change)
    with pytest.raises(AssertionError, match="the paused file ran with mixed precision "
                                             "'float16' in effect: not the fp32 check"):
        _frozen_verified(probe, frozen_arms, paused=paused)
