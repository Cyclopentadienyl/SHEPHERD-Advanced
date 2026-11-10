"""
The one read-only results service and its CSV/JSON export.
==========================================================
Research comparison §5 Stage 8 (owner's O6; plan review F2, F3, E3, E5;
acceptance matrix "Registration and query"). One world of kept reports and
ledgers, written by the production writers (`model_records.register` and
`keep`, `append_record`, `write_ledger`), is read by the service
(`src/api/services/results.py`) through two workspaces roots and two
registries named explicitly, as the command line names them:

- a report kept and registered is one row; kept only, its own row ("not
  registered in a listed ledger"); registered only (its store not listed), its
  own row ("report not available");
- a record merged before each run had its own record is its own row, run
  null, never joined to a report, and its reports say they may overlap it;
- a kept report of a mode this build does not describe keeps its mode and
  metrics as recorded;
- one item that cannot be read (a damaged report, a hand-edited ledger record,
  a checkpoint file, a refused root) is a row with its reason and hides none;
- two runs of one condition group with different numbers are two rows of one
  group, with the metric they differ on named;
- raw values, semantics, units, settings and environment are kept in CSV and
  JSON, by one serializer, with no server path, case id or report path in it;
- several sources are read through the server's own readers with their own
  settings, and this process's settings are never changed;
- a report's ``checkpoint_path`` and ``data_dir`` are never opened;
- the Diagnosis CSV, whose cell rules the export now shares, is byte-identical.

Module: tests/unit/test_results_export.py
"""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path

import pytest

pytest.importorskip("fastapi")  # the one reader's views are the API's models

from tests.fixtures.model_records import (  # noqa: E402
    disagree,
    keep,
    register,
    report_for,
    version_block,
)

LABEL = "mygene2-2026-10"
OTHER_LABEL = "elsewhere-1"
BASE_RUN = "20261001-000000_base_aaaaaa"
CHILD_RUN = "20261002-000000_disease_bbbbbb"


def _metrics(mrr):
    from tests.unit.test_evaluation_sidecar import DISEASE_METRICS

    return {**DISEASE_METRICS, "pooled_mrr": mrr}


def _training_state(payload, learning_rate=0.001):
    """A training state's payload: the record, and the settings it was trained
    with as the training payload records them (a location field among them,
    which no export may show)."""
    return {**payload, "checkpoint_kind": "training_state", "epoch": 3, "training_rules": 1,
            "config": {"requested": {"learning_rate": learning_rate, "seed": 7,
                                     "batch_size": 16, "run_name": "base",
                                     "data_dir": "/secret/workspace/data"},
                       "effective": {"device": {"requested": "auto", "effective": "cpu"},
                                     "amp": {"enabled": False}}}}


def _put(workspace, run, name, payload):
    from src.utils.checkpoint_io import write_checkpoint

    path = workspace / "runs" / "gatv2" / run / "checkpoints" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path, write_checkpoint(payload, path)


def _kg_record(checkpoint, run_id):
    """A KG edge set's ledger record for `checkpoint`, one metric not
    computable, registered from a store this export does not list."""
    from src.evaluation.sidecar import build_record
    from tests.unit.test_ledger_by_data_kind import KG_METRICS, exposure_block, kg_edges_report

    from src.utils.run_environment import component_facts

    exposure = exposure_block()
    exposure["sources"]["model_record"]["checkpoint_sha256"] = checkpoint
    report = kg_edges_report(metrics={**KG_METRICS, "lp_roc_auc": None}, exposure=exposure,
                             artifact_digests={"checkpoint": checkpoint, "pairs": "c3" * 32,
                                               "kg": "1" * 64, "node_features": "2" * 64,
                                               "edge_indices": "3" * 64, "num_nodes": "4" * 64})
    # The measuring run's native build identity (Stage 2), as the collector
    # records a component.
    report["manifest"]["environment"]["facts"]["components"] = {
        "torch_scatter": component_facts("pyg native extension", installed=True,
                                         version="2.1.2", native={
                                             "state": "recorded", "count": 1,
                                             "aggregate_sha256": "a7" * 32})}
    data = json.dumps(report).encode()
    return build_record(report, hashlib.sha256(data).hexdigest()), run_id


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Two workspaces roots, each with a workspace named ``lab``; a listed
    registry (``main``) and one this export does not list (``unlisted``)."""
    from src.evaluation.sidecar import (
        LEDGER_FILENAME,
        append_record,
        build_record,
        empty_ledger,
        write_ledger,
    )
    from tests.fixtures.report_store import store_bytes, version_of
    from tests.unit import test_task_first_selection as bundles

    for name in ("SHEPHERD_WORKSPACES_ROOT", "SHEPHERD_REGISTRY"):
        monkeypatch.delenv(name, raising=False)
    sentinel = tmp_path / "sentinel"
    (sentinel / "data").mkdir(parents=True)
    (sentinel / "model.pt").write_bytes(b"never opened")
    paths = {"checkpoint_path": str(sentinel / "model.pt"), "data_dir": str(sentinel / "data")}

    root_a, root_b = tmp_path / "roots-a", tmp_path / "roots-b"
    lab_a, lab_b = root_a / "lab", root_b / "lab"
    base_path, base = _put(lab_a, BASE_RUN, "last.pt",
                           _training_state(bundles._record_payload(logs={"val_lp_accuracy": .8})))
    child_path, child = _put(lab_a, CHILD_RUN, "best.pt", {
        **bundles._record_payload("disease", parent={"sha256": base, "run": BASE_RUN}),
        "checkpoint_kind": "weights", "training_rules": 1})
    (lab_a / "runs" / "gatv2" / CHILD_RUN / "checkpoints" / "broken.pt").write_bytes(b"no")
    other_path, other = _put(lab_b, "20261004-000000_base_dddddd", "last.pt",
                             _training_state(bundles._record_payload(family_id="other-family"),
                                             learning_rate=0.01))

    registry, unlisted = tmp_path / "registry-main", tmp_path / "registry-unlisted"
    block = version_block(LABEL, "research_selection")
    # Kept and registered; a second run of its group with another number,
    # kept only; registered from a store this export does not list.
    first = report_for(child, block, metrics=_metrics(0.1 + 0.2), **paths)
    registered_path, _ = register(registry, child_path, first, registered_at="2026-10-17T10:00:00Z")
    second_path = disagree(registry, first, _metrics(0.25))
    elsewhere = report_for(child, version_block(OTHER_LABEL, "research_selection"),
                           metrics=_metrics(0.5))
    elsewhere_path, _ = register(unlisted, child_path, elsewhere,
                                 registered_at="2026-10-18T10:00:00Z")
    # A mode this build does not describe, kept as recorded.
    odd = report_for(child, block, metrics=None, mode="Z9")
    odd["authoritative_metrics"] = {"hits@1": 0.25, "odd_count": 3}
    odd_path, _ = keep(registry, odd)
    # Two kept reports of `other` merged into one record before each run had its
    # own record (a schema-3 ledger), then a KG edge set's run appended.
    merged_paths, records = [], []
    for mrr in (0.4, 0.41):
        path, record = keep(registry, report_for(other, block, metrics=_metrics(mrr)))
        merged_paths.append(path)
        records.append(record)
    merged = copy.deepcopy(records[0])
    merged["corroborating_reports"] = [records[1]["source_artifact_digest"]]
    merged["registration"] = {"sequence": 1, "registered_at": "2026-10-16T12:30:00Z"}
    ledger = {**empty_ledger(), "schema_version": 3, "records": [merged]}
    kg_record, kg_run = _kg_record(other, f"20261020T000000Z-{other[:12]}-0000000a")
    outcome, ledger = append_record(ledger, kg_record, run_id=kg_run,
                                    registered_at="2026-10-20T00:00:00Z")
    assert outcome == "registered"
    write_ledger(other_path.parent / LEDGER_FILENAME, ledger, None)
    # One damaged report, kept beside the others.
    damaged = store_bytes(registry, b"{not json", version=version_of(registered_path),
                          checkpoint_sha256=child)
    return {"tmp": tmp_path, "root_a": root_a, "root_b": root_b, "registry": registry,
            "unlisted": unlisted, "base": base, "child": child, "other": other,
            "child_path": child_path, "registered": registered_path, "second": second_path,
            "elsewhere": elsewhere_path, "odd": odd_path, "merged": merged_paths,
            "kg_run": kg_run, "damaged": damaged, "sentinel": sentinel,
            "build_record": build_record}


def _sources(world, **more):
    from src.api.services import results

    return results.explicit_sources(
        workspaces_roots=[("a", world["root_a"]), ("b", world["root_b"]), *more.get("roots", [])],
        registries=[("main", world["registry"])])


def _query(world, **more):
    from src.api.services import results

    return results.query(_sources(world, **more))


def _by_run(rows):
    return {row["run_id"]: row for row in rows if row["result"] != "unreadable"
            and row["run_id"] is not None}


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_report_and_ledger_present_absent_are_each_a_row_joined_only_by_run_and_bytes(world):
    """Kept and registered: one row, both places. Kept only: its own row, not
    registered. Registered from a store not listed: its own row, "report not
    available", with the fields its record has."""
    rows = _by_run(_query(world))
    registered = rows[world["registered"].parent.name]
    assert (registered["report"], registered["registered"]) == ("kept", "registered")
    assert registered["report_sha256"] == _digest(world["registered"])
    assert registered["stores"] == [{"source": "main", "label": LABEL}]
    assert [(ledger["source"], ledger["workspace"], ledger["sequence"])
            for ledger in registered["ledgers"]] == [("a", "lab", 1)]
    kept = rows[world["second"].parent.name]
    assert (kept["report"], kept["registered"], kept["ledgers"]) == (
        "kept", "not registered in a listed ledger", [])
    ledger_only = rows[world["elsewhere"].parent.name]
    assert (ledger_only["report"], ledger_only["registered"]) == ("report not available",
                                                                 "registered")
    assert ledger_only["stores"] == [] and ledger_only["data"]["label"] == OTHER_LABEL
    assert ledger_only["unknown"]["references"] == "its data version is not in a listed registry"
    assert ledger_only["metrics"][0]["value"] is not None


def test_a_historical_merged_record_is_its_own_row_never_a_run(world):
    """A record merged before each run had its own record: run null, never
    joined, never counted as a run; the reports it holds say so."""
    found = _query(world)
    (historical,) = [row for row in found if row["result"] == "historical_record"]
    assert historical["run_id"] is None and historical["n_corroborating"] == 1
    assert (historical["report"], historical["registered"]) == (
        "report not available", "registered, merged before each run had its own record")
    assert historical["stores"] == []
    rows = _by_run(found)
    for path in world["merged"]:
        row = rows[path.parent.name]
        assert row["result"] == "run" and row["ledgers"] == []
        assert row["registered"].startswith("not registered as its own run; a record merged")
    assert sum(row["result"] == "run" for row in found
               if row["report_sha256"] in {_digest(p) for p in world["merged"]}) == 2


def test_an_unknown_mode_keeps_its_mode_and_metrics_as_recorded(world):
    """Each value as the record holds it: an integer stays an integer (the
    one reader's view holds every metric as a float), in JSON and in CSV."""
    from src.api.services import results

    found = _query(world)
    row = _by_run(found)[world["odd"].parent.name]
    assert row["protocol"] == "Z9"
    assert [(m["name"], m["value"]) for m in row["metrics"]] == [("hits@1", 0.25),
                                                                 ("odd_count", 3)]
    assert type(row["metrics"][1]["value"]) is int
    assert row["unknown"]["metric.unit"].startswith("its record holds no unit per metric")
    table = list(csv.DictReader(io.StringIO(results.to_csv(found))))
    (count,) = [r for r in table if r["run_id"] == world["odd"].parent.name
                and r["metric_name"] == "odd_count"]
    assert count["value"] == "3"
    (exported,) = [r for r in json.loads(results.to_json(found))["rows"]
                   if r["run_id"] == world["odd"].parent.name and r["metric_name"] == "odd_count"]
    assert type(exported["value"]) is int


def test_one_unreadable_item_is_a_row_with_its_reason_and_hides_none(world):
    """A damaged report, a hand-edited ledger record, a checkpoint file that
    is not one and a refused root: each a row with its reason, the rest read."""
    from src.evaluation.sidecar import LEDGER_FILENAME, read_ledger, write_ledger

    ledger_path = world["child_path"].parent / LEDGER_FILENAME
    ledger, digest = read_ledger(ledger_path)
    ledger["records"][1]["metrics"]["pooled_mrr"] = "high"
    write_ledger(ledger_path, ledger, digest)
    found = _query(world, roots=[("refused", "relative/root")])
    unreadable = {(row["item"], row["where"].get("run_id") or row["where"].get("checkpoint")
                   or row["where"].get("source")): row["reason"]
                  for row in found if row["result"] == "unreadable"}
    assert "it does not parse as UTF-8 JSON" in unreadable[("kept report",
                                                            world["damaged"].parent.name)]
    assert unreadable[("ledger record", world["elsewhere"].parent.name)] == (
        "metrics.pooled_mrr: Input should be a valid number")
    assert ("checkpoint", f"gatv2/{CHILD_RUN}/broken.pt") in unreadable
    assert "is not an absolute path" in unreadable[("workspaces root", "refused")]
    rows = _by_run(found)
    assert {world["registered"].parent.name, world["second"].parent.name,
            world["odd"].parent.name, world["kg_run"]} <= set(rows)


def _ledger_of(path):
    from src.evaluation.sidecar import LEDGER_FILENAME, read_ledger

    ledger_path = Path(path).parent / LEDGER_FILENAME
    return ledger_path, *read_ledger(ledger_path)


def test_a_ledgers_records_are_rows_when_their_weights_file_is_gone(world):
    """An ordinary cleanup removes a checkpoint file: every readable record
    the ledger beside it holds is still a row, its model unknown with why, and
    the kept report it registered still reads registered."""
    world["child_path"].unlink()
    found = _query(world)
    rows = _by_run(found)
    registered = rows[world["registered"].parent.name]
    assert (registered["report"], registered["registered"]) == ("kept", "registered")
    assert registered["ledgers"][0]["ledger"] == f"gatv2/{CHILD_RUN}"
    ledger_only = rows[world["elsewhere"].parent.name]
    assert (ledger_only["report"], ledger_only["registered"]) == ("report not available",
                                                                 "registered")
    for row in (registered, ledger_only):
        assert row["model"]["checkpoint_sha256"] == world["child"]
        assert row["model"]["locations"] == []
        assert row["unknown"]["model.family_id"] == "its checkpoint is not in a listed workspace"
    assert not [row for row in found if row["result"] == "unreadable"
                and row["item"] in ("ledger", "ledger record")]


def test_a_ledgers_records_are_rows_when_their_checkpoint_is_listed_with_a_problem(world):
    """A checkpoint listed with a problem (its metadata is not of its kind)
    whose digest is known: its own unreadable row, and every record the
    ledger beside it holds for it a row, its location named and its model's
    record unknown with why."""
    from src.utils.checkpoint_io import write_checkpoint

    run = world["root_a"] / "lab" / "runs" / "gatv2" / "20261005-000000_odd_eeeeee"
    path = run / "checkpoints" / "best.pt"
    path.parent.mkdir(parents=True)
    digest = write_checkpoint({"checkpoint_kind": "weights", "epoch": "3"}, path)
    report = report_for(digest, version_block(LABEL, "research_selection"),
                        metrics=_metrics(0.2))
    kept, _ = register(world["registry"], path, report, registered_at="2026-10-19T10:00:00Z")
    found = _query(world)
    (problem,) = [row for row in found if row["result"] == "unreadable"
                  and row["where"].get("checkpoint") == f"gatv2/{run.name}/best.pt"]
    assert problem["item"] == "checkpoint" and problem["where"]["checkpoint_sha256"] == digest
    row = _by_run(found)[kept.parent.name]
    assert (row["report"], row["registered"]) == ("kept", "registered")
    assert row["model"]["locations"] == [{"source": "a", "workspace": "lab",
                                          "checkpoint": f"gatv2/{run.name}/best.pt"}]
    assert row["unknown"]["model.family_id"].startswith("its checkpoint is listed only with a "
                                                        "problem")


def test_a_ledger_record_that_names_no_weights_is_a_row_with_its_reason(world):
    """A hand edit that removes a record's checkpoint digest: the record is a
    row with its reason and its place in the ledger; the others still read."""
    from src.evaluation.sidecar import write_ledger

    ledger_path, ledger, digest = _ledger_of(world["child_path"])
    orphan = copy.deepcopy(ledger["records"][0])
    del orphan["checkpoint_digest"]
    ledger["records"].append(orphan)
    write_ledger(ledger_path, ledger, digest)
    found = _query(world)
    (row,) = [row for row in found if row["result"] == "unreadable"
              and row["where"].get("ledger_index") == len(ledger["records"]) - 1]
    assert row["item"] == "ledger record"
    assert row["reason"] == "checkpoint_digest: missing, so it names no weights"
    assert row["where"]["ledger"] == f"gatv2/{CHILD_RUN}"
    assert world["registered"].parent.name in _by_run(found)


def test_an_unreadable_record_of_a_kept_run_is_not_read_as_no_record(world):
    """A listed ledger's record of a kept run that a hand edit made
    unreadable: the kept report's row says so, never "not registered"."""
    from src.api.services import results
    from src.evaluation.sidecar import write_ledger

    ledger_path, ledger, digest = _ledger_of(world["child_path"])
    run = world["registered"].parent.name
    (index,) = [i for i, r in enumerate(ledger["records"]) if r.get("run_id") == run]
    ledger["records"][index]["metrics"]["pooled_mrr"] = "high"
    write_ledger(ledger_path, ledger, digest)
    found = _query(world)
    assert _by_run(found)[run]["registered"] == results.RECORD_UNREADABLE
    assert [row["item"] for row in found if row["result"] == "unreadable"
            and row["run_id"] == run] == ["ledger record"]
    assert _by_run(found)[world["second"].parent.name]["registered"] == results.NOT_REGISTERED


def test_a_workspace_entry_outside_the_root_is_a_row_with_its_reason(world):
    """An entry of the root that resolves outside it (a link out) is not read,
    and is a row naming it, its reason with no server path."""
    outside = world["tmp"] / "outside-ws"
    outside.mkdir()
    os.symlink(outside, world["root_a"] / "linked")
    found = _query(world)
    (row,) = [row for row in found if row["result"] == "unreadable"
              and row["where"].get("workspace") == "linked"]
    assert row["item"] == "workspace" and row["where"]["source"] == "a"
    assert row["reason"] == ("the workspace 'linked' resolves outside the workspaces root <a>, "
                             "so it is not in the server's one workspace namespace")
    assert world["registered"].parent.name in _by_run(found)


def test_copies_of_one_run_are_one_row_with_each_place(world):
    """A registry listed twice and a workspace copied under a third root: the
    run is one row, with each store and each ledger that holds it."""
    import shutil

    from src.api.services import results

    copy_registry = shutil.copytree(world["registry"], world["tmp"] / "registry-copy")
    root_c = world["tmp"] / "roots-c"
    shutil.copytree(world["root_a"] / "lab", root_c / "lab", symlinks=True)
    found = results.query(results.explicit_sources(
        workspaces_roots=[("a", world["root_a"]), ("c", root_c)],
        registries=[("main", world["registry"]), ("copy", copy_registry)]))
    run = world["registered"].parent.name
    rows = [row for row in found if row["run_id"] == run]
    assert len(rows) == 1 and rows[0]["conflict"] is None
    assert rows[0]["stores"] == [{"source": "main", "label": LABEL},
                                 {"source": "copy", "label": LABEL}]
    assert [ledger["source"] for ledger in rows[0]["ledgers"]] == ["a", "c"]


def test_a_run_whose_report_bytes_differ_between_two_places_is_kept_twice_none_chosen(world):
    import shutil

    from src.api.services import results

    copy_registry = shutil.copytree(world["registry"], world["tmp"] / "registry-copy")
    run = world["second"].parent.name
    changed = copy_registry / world["second"].relative_to(world["registry"])
    report = json.loads(changed.read_text(encoding="utf-8"))
    report["authoritative_metrics"]["pooled_mrr"] = 0.9
    changed.write_text(json.dumps(report), encoding="utf-8")
    found = results.query(results.explicit_sources(
        registries=[("main", world["registry"]), ("copy", copy_registry)]))
    rows = [row for row in found if row["run_id"] == run]
    assert sorted(row["stores"][0]["source"] for row in rows) == ["copy", "main"]
    assert len({row["report_sha256"] for row in rows}) == 2
    for row in rows:
        assert row["conflict"].startswith(f"run {run} holds other report bytes elsewhere")
        assert row["conflict"].endswith("each is kept as found, none chosen")


def test_a_store_that_cannot_be_read_leaves_its_groups_unchecked_saying_why(world, monkeypatch):
    """The group's state comes from the one turning of a scan into a state
    (`record_states.group_state`): a store whose scan fails leaves each group
    of its version cannot_be_checked with the store's reason."""
    from src.evaluation import report_store

    def fails(_version):
        raise RuntimeError("no")

    monkeypatch.setattr(report_store, "scan_report_store", fails)
    found = _query(world)
    row = _by_run(found)[world["registered"].parent.name]
    assert row["report"] == "report not available"
    assert row["group"]["state"] == "cannot_be_checked"
    assert row["group"]["why"] == "its report store could not be read (RuntimeError)"


def test_two_equal_condition_runs_with_different_numbers_are_one_group_differences_named(world):
    rows = _by_run(_query(world))
    a, b = rows[world["registered"].parent.name], rows[world["second"].parent.name]
    assert a["group"]["condition_group"] == b["group"]["condition_group"]
    assert a["group"]["state"] == b["group"]["state"] == "contested"
    assert a["group"]["differing"] == b["group"]["differing"] == ["pooled_mrr"]
    assert {m["value"] for row in (a, b) for m in row["metrics"]
            if m["name"] == "pooled_mrr"} == {0.1 + 0.2, 0.25}


def test_the_model_its_lineage_and_its_recorded_settings(world):
    """Identity from the server's lists; settings as the checkpoint records
    them, the location fields left out; a weights-only file's settings are
    unknown with why, never filled from the recipe."""
    found = _query(world)
    child = _by_run(found)[world["registered"].parent.name]["model"]
    assert child["checkpoint_sha256"] == world["child"]
    assert child["locations"] == [{"source": "a", "workspace": "lab",
                                   "checkpoint": f"gatv2/{CHILD_RUN}/best.pt"}]
    assert (child["learning_stage"], child["task"], child["operation"]) == (
        "finetune", "disease", "finetune")
    assert (child["parent_sha256"], child["parent_run"]) == (world["base"], BASE_RUN)
    assert child["learning_settings"] is None and child["training_rules"] == 1
    row = _by_run(found)[world["registered"].parent.name]
    assert row["unknown"]["model.learning_settings"] == (
        "a weights-only file records no training settings")
    # The structure, every value of its spec as its record holds it: two
    # variants that differ on one field differ here by value.
    from src.models.bundle.spec import finetune_spec_from_parent
    from tests.unit import test_task_first_selection as bundles

    spec = finetune_spec_from_parent(bundles._base_spec(), "disease").to_dict()
    del spec["sources"]
    assert child["spec"] == json.loads(json.dumps(spec))
    for name in child["differences"]:
        section, part = name.split(".", 1)
        assert part in child["spec"][section]
    kg = _by_run(found)[world["kg_run"]]
    assert kg["model"]["family_id"] == "other-family"
    assert kg["model"]["learning_settings"] == {"learning_rate": 0.01, "seed": 7,
                                                "batch_size": 16}
    assert kg["model"]["effective_settings"]["device"] == {"requested": "auto",
                                                          "effective": "cpu"}


def test_a_kg_edge_sets_units_statuses_and_exposure(world):
    row = _by_run(_query(world))[world["kg_run"]]
    assert row["environment"]["native_identity"]["torch_scatter"]["native"] == {
        "state": "recorded", "count": 1, "aggregate_sha256": "a7" * 32}
    assert "environment.native_identity" not in row["unknown"]
    metrics = {m["name"]: m for m in row["metrics"]}
    assert metrics["lp_roc_auc"]["value"] == metrics["lp_roc_auc"]["status"] == "not_computable"
    assert metrics["lp_accuracy"]["status"] == "computed"
    assert metrics["lp_accuracy"]["unit"] and metrics["lp_accuracy"]["denominator"]
    assert row["exposure"]["state"] == "held_out"
    assert row["exposure"]["reading"].startswith("held out relative to the recorded")
    assert row["exposure"]["r10"] is None
    assert (row["data"]["kind"], row["data"]["role"]) == ("kg_edges", "test")
    assert row["report"] == "report not available"
    # The run's own counts, each with its unit: how many pairs were scored and
    # how many of them had a score that was not finite.
    assert set(row["run_counts"]["counts"]) == set(row["run_counts"]["units"])
    assert {"pairs_scored", "pairs_not_finite"} <= set(row["run_counts"]["counts"])


def test_csv_and_json_are_one_long_form_with_raw_values_and_no_path(world):
    from src.api.services import results

    found = _query(world)
    rows = results.long_rows(found)
    table = list(csv.DictReader(io.StringIO(results.to_csv(found))))
    exported = json.loads(results.to_json(found))
    assert exported["columns"] == list(results.COLUMN_NAMES) == list(table[0])
    assert len(table) == len(exported["rows"]) == len(rows)
    for got, want in zip(exported["rows"], rows, strict=True):
        assert got == json.loads(json.dumps(want))
    run = world["registered"].parent.name
    (mrr,) = [r for r in table if r["run_id"] == run and r["metric_name"] == "pooled_mrr"]
    assert mrr["value"] == "0.30000000000000004" and mrr["protocol"] == "disease"
    assert mrr["semantics"] and mrr["role"] == "research_selection"
    assert mrr["checkpoint_sha256"] == f"sha256:{world['child']}"
    assert mrr["learning_settings"] == "unknown" and mrr["unit"] == "unknown"
    assert mrr["device"] == "cuda" and mrr["deterministic_algorithms"] == "true"
    assert mrr["environment_class"] == "CUDA, deterministic algorithms on"
    assert mrr["group_state"] == "contested"
    # A supplied cohort's relation to the model's training: its R10 states
    # and counts, never "not applicable".
    assert mrr["exposure_state"] == results.EXPOSURE_AS_R10
    r10 = json.loads(mrr["r10"])
    assert set(r10) == {"training", "validation"}
    assert set(r10["training"]) == {"evidence", "disease_overlap", "phenotype_set_overlap"}
    row = _by_run(found)[run]
    assert r10 == row["exposure"]["r10"]
    assert r10["training"]["disease_overlap"]["state"] in (
        "overlap", "none", "unverifiable", "no_validation_inputs_recorded")
    assert mrr["run_counts"] == "unknown" and "run_counts" in row["unknown"]
    (lp,) = [r for r in exported["rows"]
             if r["run_id"] == world["kg_run"] and r["metric_name"] == "lp_roc_auc"]
    assert lp["value"] == "not_computable" and lp["unit"]
    text = results.to_csv(found) + results.to_json(found)
    for secret in (str(world["tmp"]), "/secret/workspace/data", str(world["sentinel"])):
        assert secret not in text
    assert "data_dir" not in text and "checkpoint_path" not in text


def test_a_text_cell_a_spreadsheet_would_run_is_defused(world):
    from src.api.services import results

    odd = report_for(world["child"], version_block(LABEL, "research_selection"),
                     score_semantics="=HYPERLINK(\"x\")")
    keep(world["registry"], odd)
    table = list(csv.DictReader(io.StringIO(results.to_csv(_query(world)))))
    assert any(r["semantics"] == "'=HYPERLINK(\"x\")" for r in table)


def test_several_sources_are_read_through_the_servers_own_readers_and_settings(world,
                                                                            monkeypatch):
    """Each root and registry is read by the setting's own rule with its own
    context, never the process's: settings in this process that name a decoy
    registry and a decoy workspaces root are not read (none of their rows
    appear), and nothing sets or removes a process setting while the query
    runs."""
    import shutil

    from src.config import server_settings

    decoy_registry = shutil.copytree(world["unlisted"], world["tmp"] / "decoy-registry")
    decoy_root = world["tmp"] / "decoy-root"
    shutil.copytree(world["root_b"] / "lab", decoy_root / "decoy-lab")
    monkeypatch.setenv("SHEPHERD_REGISTRY", str(decoy_registry))
    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(decoy_root))
    seen = []
    for name in ("workspaces_root", "cohort_registry"):
        real = getattr(server_settings, name)

        def spy(environ=None, _real=real, _name=name):
            seen.append((_name, environ))
            return _real(environ)

        monkeypatch.setattr(server_settings, name, spy)

    def refused(*_args, **_kwargs):
        raise AssertionError("the query changed a process setting")

    from src.api.services import results

    sources = _sources(world)
    before = dict(os.environ)
    with monkeypatch.context() as during:
        for name in ("__setitem__", "__delitem__"):
            during.setattr(type(os.environ), name, refused)
        for name in ("putenv", "unsetenv"):
            during.setattr(os, name, refused)
        found = results.query(sources)
    assert dict(os.environ) == before
    assert {name for name, _environ in seen} == {"workspaces_root", "cohort_registry"}
    own = [dict(source.environ) for source in sources]
    for _name, environ in seen:
        assert environ is not os.environ and dict(environ) in own
    assert {"SHEPHERD_WORKSPACES_ROOT": str(world["root_a"])} in own
    rows = [row for row in found if row["result"] != "unreadable"]
    assert {loc["source"] for row in rows for loc in row["model"]["locations"]} == {"a", "b"}
    assert {loc["workspace"] for row in rows for loc in row["model"]["locations"]} == {"lab"}
    assert {loc["source"] for row in found for loc in row.get("ledgers") or []} == {"a", "b"}
    assert {store["source"] for row in rows for store in row["stores"]} == {"main"}
    # The decoy registry keeps the report of the run registered "elsewhere":
    # had it been read, that row would read "kept".
    assert _by_run(found)[world["elsewhere"].parent.name]["report"] == "report not available"


def test_every_kind_of_report_kept_today_is_a_row_mode_c_and_d_included(tmp_path):
    """The P0-1 corpus's registry (each task protocol as measured, a second
    run, a kept Mode C and Mode D report), read through the service: every
    kept report is one row with its own mode and metrics, none unreadable."""
    import shutil

    from src.api.services import results
    from tests.fixtures.goldens import CORPUS, golden

    registry = shutil.copytree(CORPUS / "registry", tmp_path / "registry")
    found = results.query(results.explicit_sources(registries=[("corpus", registry)]))
    kept = golden("p0_1_records.json")["reports"]
    assert sorted((row["stores"][0]["label"], row["run_id"]) for row in found) == sorted(
        (report["label"], report["run_id"]) for report in kept)
    by_run = {row["run_id"]: row for row in found}
    for report in kept:
        row = by_run[report["run_id"]]
        assert row["protocol"] == report["record"]["scoring"]["mode"]
        assert {m["name"]: m["value"] for m in row["metrics"]} == report["record"]["metrics"]
        assert row["report"] == "kept" and row["model"]["locations"] == []
        assert row["unknown"]["model.family_id"] == "its checkpoint is not in a listed workspace"
    assert {row["protocol"] for row in found} >= {"C", "D"}


@pytest.mark.parametrize("kind", ["supplied", "kg_edges", "generated"])
def test_a_kept_reports_record_is_read_by_the_one_reader_as_its_kinds_result(kind):
    """`result_view` reads `build_record`'s record (no registration, no run)
    by the same discrimination and parts as a ledger record."""
    from src.api.services.record_views import RESULT_VIEWS, result_view
    from src.evaluation.sidecar import build_record
    from tests.unit.test_evaluation_sidecar import registered_report
    from tests.unit.test_ledger_by_data_kind import generated_report, kg_edges_report

    report = {"supplied": registered_report, "kg_edges": kg_edges_report,
              "generated": generated_report}[kind]()
    view = result_view(build_record(report, "f" * 64))
    assert type(view) is RESULT_VIEWS[kind] and view.cohort.kind == kind
    assert not {"registration", "run_id", "n_corroborating"} & set(type(view).model_fields)


def test_a_generated_splits_relation_to_training_is_its_allocations_not_compared():
    """A generated split's run: its relation to the model's training is its
    allocation's (in its data identity), never compared, never "not
    applicable"; it records no R10."""
    from src.api.services import results
    from src.api.services.record_views import result_view
    from src.evaluation.sidecar import build_record
    from tests.unit.test_ledger_by_data_kind import generated_report

    view = result_view(build_record(generated_report(), "f" * 64))
    exposure = results._exposure(view)
    assert exposure["state"] == results.EXPOSURE_BY_ALLOCATION
    assert exposure["r10"] is None and exposure["reading"] is None


def test_the_record_views_keep_the_field_order_the_api_has_always_had():
    """Each kind's record view is its result view and its registration's
    fields, in the order of the API's views before the result views were split
    out (e3e50ca): the JSON a client reads is unchanged."""
    from src.api.services import record_views as rv

    common = ["cohort", "counts", "mode", "score_semantics", "metrics", "n_ranked",
              "relations", "r10", "environment_class", "deterministic_algorithms", "gpu",
              "native_identity", "dataset_origin", "scoring_semantics_digest",
              "measurement_semantics_digest", "registration", "run_id", "historical_overlap",
              "n_corroborating", "source_artifact_digest"]
    assert list(rv.RecordView.model_fields) == common
    assert list(rv.KgEdgesRecordView.model_fields) == [
        *common, "count_units", "measured", "exposure", "exposure_problem"]
    assert list(rv.GeneratedRecordView.model_fields) == [*common, "count_units"]


def test_the_service_and_its_reader_import_no_route():
    """The one reader of records and the results service build on the
    service layer's answer base (`response_models`): importing them loads no
    HTTP route."""
    import subprocess
    import sys

    code = ("import sys; import src.api.services.results, src.api.services.record_views; "
            "print(sorted(m for m in sys.modules if m.startswith('src.api.routes')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=Path(__file__).resolve().parents[2], check=True)
    assert out.stdout.strip() == "[]"


def test_source_names_are_distinct():
    from src.api.services import results

    with pytest.raises(ValueError, match="given twice: lab"):
        results.explicit_sources(workspaces_roots=[("lab", "/x"), ("lab", "/y")])


def test_no_path_a_report_records_is_opened(world):
    from src.api.services import results
    from tests.fixtures.opens import count_opens

    with count_opens() as counter:
        results.to_csv(_query(world))
    assert counter.opens(world["sentinel"] / "model.pt") == 0
    assert counter.listed(world["sentinel"] / "data") == 0
    assert counter.opens(world["damaged"]) == 1


def test_filters_keep_matching_rows_and_every_unreadable_item(world):
    from src.api.services import results

    found = _query(world)
    kept = results.filtered(found, {"protocol": ["kg_link_prediction"]})
    assert {row["protocol"] for row in kept if row["result"] != "unreadable"} == {
        "kg_link_prediction"}
    assert [row for row in kept if row["result"] == "unreadable"] == [
        row for row in found if row["result"] == "unreadable"]
    assert results.filtered(found, {"family": ["other-family"], "role": ["test"]})
    with pytest.raises(ValueError, match="unknown filter"):
        results.filtered(found, {"colour": ["red"]})


def test_the_workspace_filter_keeps_what_was_found_there_by_its_model_or_its_ledger(world):
    """Stage 8b (Models › Results scopes by it): a row is in a workspace where
    a listed checkpoint of its model is, or a ledger holding its record; a
    ledger-only record whose model is listed nowhere is still in its ledger's
    workspace; the values are what a page offers (`filter_values`)."""
    from src.api.services import results

    found = _query(world)
    rows = _by_run(found)
    registered = rows[world["registered"].parent.name]
    assert results.filter_values(registered, "workspace") == ["lab"]
    assert results.filter_values(registered, "protocol") == ["disease"]
    # Two roots, each with a workspace named "lab": one name, as the filter
    # compares names (a page reads one source; the command line's help says
    # the name is matched in every named root).
    kept = results.filtered(found, {"workspace": ["lab"]})
    assert {row["run_id"] for row in kept if row["result"] == "run"} == {
        row["run_id"] for row in found if row["result"] == "run"
        and (row["model"]["locations"] or row["ledgers"])}
    assert results.filtered(found, {"workspace": ["elsewhere"]}) == [
        row for row in found if row["result"] == "unreadable"]
    # A ledger-only record: in its ledger's workspace, its model's location or not.
    ledger_only = dict(rows[world["elsewhere"].parent.name])
    ledger_only["model"] = {**ledger_only["model"], "locations": []}
    assert results.filter_values(ledger_only, "workspace") == ["lab"]
    assert results.filtered([ledger_only], {"workspace": ["lab"]}) == [ledger_only]


def test_a_field_without_a_value_gives_no_filter_value_and_matches_none(world):
    """Stage 8b review: as `FILTERS` says, a row whose field is unknown or
    absent (null) has no value there, so no filter value matches it: never
    the text "None", which a page would offer as a value and a recorded
    "None" would share."""
    from src.api.services import results

    row = next(row for row in _query(world) if row["result"] == "run")
    unknown = {**row, "environment": {**row["environment"], "class": None},
               "measurement": {key: value for key, value in row["measurement"].items()
                               if key != "device"}}
    assert results.filter_values(unknown, "environment_class") == []
    assert results.filter_values(unknown, "device") == []
    assert results.filtered([unknown], {"environment_class": ["None"]}) == []
    assert results.filtered([unknown], {"device": ["None"]}) == []
    recorded = {**row, "environment": {**row["environment"], "class": "None"}}
    assert results.filter_values(recorded, "environment_class") == ["None"]


def test_the_command_line_exports_and_never_overwrites(world, tmp_path, capsys):
    from scripts import export_results

    out = tmp_path / "out.csv"
    argv = ["--workspaces-root", f"a={world['root_a']}", "--workspaces-root",
            f"b={world['root_b']}", "--registry", str(world["registry"]), "--format", "csv",
            "--output", str(out)]
    assert export_results.main(argv) == 0
    table = list(csv.DictReader(io.StringIO(out.read_text(encoding="utf-8"))))
    assert {row["run_id"] for row in table} >= {world["kg_run"],
                                                world["registered"].parent.name}
    assert "could not be read" in capsys.readouterr().err
    before = out.read_bytes()
    assert export_results.main(argv) == 2
    assert out.read_bytes() == before
    assert export_results.main([*argv[:-2], "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["format"] == "shepherd-results-export/1"


def test_a_source_given_without_a_name_is_named_by_its_kind_never_its_directory(world, capsys):
    """A directory's own name may say more than an export should: a source
    given without NAME= is named by a fixed word and its place."""
    from scripts import export_results

    assert export_results.main(["--workspaces-root", str(world["root_a"]), "--workspaces-root",
                                str(world["root_b"]), "--registry", str(world["registry"]),
                                "--format", "json"]) == 0
    text = capsys.readouterr().out
    rows = json.loads(text)["rows"]
    assert {loc["source"] for row in rows for loc in row["model_locations"] or []} == {
        "workspaces-root-1", "workspaces-root-2"}
    assert {store["source"] for row in rows for store in row["stores"] or []} == {"registry-1"}
    for name in ("roots-a", "roots-b", "registry-main"):
        assert name not in text


def test_the_command_lines_help_says_it_runs_as_the_operator(capsys):
    from scripts import export_results

    with pytest.raises(SystemExit):
        export_results.main(["--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "operator on this machine" in text
    # The workspace filter compares a name: matched in every named root.
    assert "the name is matched in every named workspaces root" in text


# ---------------------------------------------------------------------------
# The Diagnosis CSV, whose cell rules the export shares (`src.utils.csv_cells`)
# ---------------------------------------------------------------------------
def _diagnosis_cases():
    """Results whose CSV was pinned on the tree before the cell rules moved
    (`tests/fixtures/diagnosis_csv_pin.json`): each task kind, both identity
    kinds, a response without one, and values a rule changes (formula starts,
    digests and not, null, empty, booleans, an unrounded float)."""
    from tests.unit import test_diagnosis_panel as t

    out = {"disease_synthetic": t._sample_result(),
           "disease_files": t._result_with(t._served_identity(origin="files")),
           "gene_task": t._gene_result(), "patients_like_me": t._patients_result(),
           "parts_left_out": t._result_with(t._parts_left_out())}
    none = t._sample_result()
    del none["served_identity"]
    out["not_reported"] = none
    odd = t._sample_result()
    identity = odd["served_identity"]
    identity["checkpoint"]["sha256"] = None
    identity["graph"]["kg_sha256"] = "=not-a-digest"
    identity["graph"]["split_manifest_sha256"] = ""
    identity["dataset_record"]["origin"] = "+files"
    identity["settings"].update(device="-cuda", device_requested="@auto", eta=None,
                                max_path_length=True, path_length_penalty=0.1 + 0.2,
                                max_paths_per_candidate=None)
    del identity["settings"]["phenotype_normalisation_version"]
    odd["candidates"][0].update(disease_name="\tTabbed", evidence_label="\rReturn",
                                supporting_genes=["=A", "B"], score=None)
    odd["candidates"][1].update(disease_id="-x:y", matching_phenotypes=["+HP"], score=True)
    out["odd_values"] = odd
    gene, patients = t._escaped_results()
    patients["candidates"][0]["case_id"] = "=HYPERLINK(\"http://x\")"
    gene["candidates"][1]["gene_name"] = "-PTEN"
    patients["reference_cohort"]["version_sha256"] = "@notadigest"
    out["escaped_gene"], out["escaped_patients"] = gene, patients
    unknown = t._sample_result()
    unknown["result_kind"] = "other/v9"
    out["unknown_kind"] = unknown
    return out


def test_the_diagnosis_csv_is_byte_identical_to_before_the_cell_rules_moved():
    pytest.importorskip("gradio")
    from src.webui.components import diagnosis_panel as dp

    pinned = json.loads((Path(__file__).resolve().parents[1] / "fixtures"
                         / "diagnosis_csv_pin.json").read_text(encoding="utf-8"))
    cases = _diagnosis_cases()
    assert sorted(cases) == sorted(pinned)
    for name, result in cases.items():
        assert dp._build_results_csv(result) == pinned[name], name


def test_the_diagnosis_csv_and_the_export_share_one_set_of_cell_rules():
    pytest.importorskip("gradio")
    from src.utils import csv_cells
    from src.webui.components import diagnosis_panel as dp

    assert dp._csv_text is csv_cells.text_cell
    assert dp._is_sha256 is csv_cells.is_sha256
    assert not hasattr(dp, "_CSV_FORMULA_START")
