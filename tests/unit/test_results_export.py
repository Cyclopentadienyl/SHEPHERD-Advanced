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


# ---------------------------------------------------------------------------
# Every recorded component fact (review fix RC-P2-1; §6 E1)
# ---------------------------------------------------------------------------
#: The export's columns before the component facts joined them, in order:
#: each keeps its name and place among the others.
COLUMNS_BEFORE_COMPONENTS = (
    'result', 'protocol', 'metric_name', 'value', 'value_status', 'unit', 'denominator', 'role',
    'semantics', 'data_kind', 'data_label', 'data_version_sha256', 'dataset_origin',
    'data_identity', 'data_counts', 'data_count_units', 'checkpoint_sha256', 'model_locations',
    'learning_stage', 'task', 'operation', 'family_id', 'parent_sha256', 'parent_run', 'encoder',
    'spec_reference', 'spec_differences', 'spec', 'capabilities', 'training_rules',
    'learning_settings', 'effective_settings', 'weight_transfer', 'loss_identity', 'run_id',
    'report_sha256', 'report', 'registered', 'stores', 'ledgers', 'n_corroborating',
    'historical_overlap', 'conflict', 'condition_group', 'group_state', 'group_differing',
    'group_why', 'candidate_construction', 'model_construction', 'tie_policy',
    'metric_schema_version', 'batch_size', 'n_ranked', 'run_counts', 'scoring_semantics_sha256',
    'measurement_semantics_sha256', 'environment_class', 'device', 'dtype', 'amp_enabled',
    'amp_dtype', 'torch_compile_wrapped', 'deterministic_algorithms', 'cudnn_deterministic',
    'cudnn_benchmark', 'torch_version', 'cuda_version', 'software_revision', 'gpu',
    'native_identity', 'exposure_state', 'exposure_reading', 'exposure_definition',
    'exposure_measured_edges', 'exposure_in_recorded_graph_use', 'exposure_problem', 'r10',
    'references', 'unreadable_item', 'unreadable_where', 'reason', 'unknown')
#: A fallback reason as a run records it.
FELL_BACK = "dtype unsupported; used default"


def attention(in_effect, **source):
    """A component no part of the stack has (a test double, E6), as the
    collector records it: one version and one native digest, requested,
    installed, available and selected; in effect, or fallen back with its
    reason; its source's own build facts (a commit and a compiler) beside its
    route, as a generic reader's are."""
    from src.utils.run_environment import component_facts

    return component_facts(
        "attention kernel", requested=True, installed=True, available=True, selected=True,
        in_effect=in_effect, version="2.0", fallback_reason=None if in_effect else FELL_BACK,
        native={"state": "recorded", "count": 1, "aggregate_sha256": "e" * 64},
        source={"state": "recorded", "route": "local build", "commit": "a" * 40,
                "compiler": "audit-compiler 1.0", **source})


def registered_with(world, components, registry="registry", mrr=0.5):
    """A run of the child on the listed version whose environment records
    `components`, kept in `registry` and registered beside the child, through
    the production writers; its run id."""
    report = report_for(world["child"], version_block(LABEL, "research_selection"),
                        metrics=_metrics(mrr))
    report["manifest"]["environment"]["facts"]["components"] = components
    path, _ = register(world[registry], world["child_path"], report)
    return path.parent.name


def test_two_runs_of_one_build_are_told_apart_by_what_was_in_effect_and_why(world):
    """RC-P2-1: two kept and registered runs of one component, version and
    native digest, one in effect and one fallen back with its reason, read
    through the query, the JSON and the CSV: each carries its five states,
    its reason and its source's build facts as recorded, so the two are told
    apart; the native build summary (`native_identity`) is unchanged, and
    equal for both."""
    from src.api.services import results
    from src.utils.run_environment import native_identity

    used = registered_with(world, {"test_attention": attention(True)})
    fell = registered_with(world, {"test_attention": attention(False)})
    found = _query(world)
    rows = _by_run(found)
    for run, in_effect in ((used, True), (fell, False)):
        environment = rows[run]["environment"]
        assert environment["components"] == {"test_attention": attention(in_effect)}
        # The summary is what it was: one identity for both runs.
        assert environment["native_identity"] == native_identity(
            {"test_attention": attention(in_effect)}) == {"test_attention": {
                "kind": "attention kernel", "version": "2.0",
                "native": {"state": "recorded", "count": 1, "aggregate_sha256": "e" * 64},
                "source": {"state": "recorded", "route": "local build"}}}
        assert "environment.components" not in rows[run]["unknown"]
    assert rows[used]["environment"] != rows[fell]["environment"]
    shown = rows[fell]["environment"]["components"]["test_attention"]
    assert (shown["requested"], shown["installed"], shown["available"], shown["selected"],
            shown["in_effect"], shown["fallback_reason"]) == (True, True, True, True, False,
                                                             FELL_BACK)
    assert (shown["source"]["commit"], shown["source"]["compiler"]) == ("a" * 40,
                                                                      "audit-compiler 1.0")

    exported = json.loads(results.to_json(found))["rows"]
    table = list(csv.DictReader(io.StringIO(results.to_csv(found))))
    for run, in_effect in ((used, True), (fell, False)):
        (row,) = [r for r in exported if r["run_id"] == run and r["metric_name"] == "pooled_mrr"]
        assert row["components"] == {"test_attention": attention(in_effect)}
        (cell,) = [r for r in table if r["run_id"] == run and r["metric_name"] == "pooled_mrr"]
        assert json.loads(cell["components"]) == row["components"]
        assert json.loads(cell["native_identity"]) == row["native_identity"]
    text = results.to_json(found) + results.to_csv(found)
    for fact in (FELL_BACK, "audit-compiler 1.0", "a" * 40, '"in_effect"'):
        assert fact in text


def test_a_ledger_only_row_carries_its_components_facts(world):
    """A run registered from a store the export does not list: its row reads
    "report not available" and carries every component fact its ledger
    record holds."""
    run = registered_with(world, {"test_attention": attention(False)}, registry="unlisted")
    row = _by_run(_query(world))[run]
    assert row["report"] == "report not available"
    assert row["environment"]["components"] == {"test_attention": attention(False)}


def test_an_older_record_reads_with_its_components_unknown_and_nothing_refused(world):
    """A record whose environment names no components (written before they
    were recorded) reads, its components null and named in ``unknown``, the
    CSV cell "unknown" as the native identity's is; a component written
    before ``requested`` was recorded reads too, that state left out (read as
    unknown), never filled in. Nothing is refused or rewritten."""
    from src.api.services import results

    older = attention(False)
    del older["requested"]
    run = registered_with(world, {"test_attention": older})
    ledger_before = (world["child_path"].parent / "evaluations.json").read_bytes()
    found = _query(world)
    assert (world["child_path"].parent / "evaluations.json").read_bytes() == ledger_before
    rows = _by_run(found)
    shown = rows[run]["environment"]["components"]["test_attention"]
    assert "requested" not in shown and shown == older
    assert not [row for row in found if row["result"] == "unreadable"
                and row.get("run_id") == run]

    before = rows[world["registered"].parent.name]
    assert before["environment"]["components"] is None
    assert before["unknown"]["environment.components"] == (
        "its record names no environment components")
    (cell,) = [r for r in csv.DictReader(io.StringIO(results.to_csv(found)))
               if r["run_id"] == world["registered"].parent.name
               and r["metric_name"] == "pooled_mrr"]
    assert cell["components"] == cell["native_identity"] == results.UNKNOWN


def test_a_path_in_a_components_reason_or_source_fact_is_never_exported(world):
    """D7: a server path in a recorded fallback reason, a source's build
    fact or a key of one is replaced by the one redaction before any surface
    gets it; the facts' other words are kept."""
    from src.api.services import results

    facts = attention(False, compiler="/opt/toolchains/lab-7/bin/cc 13.3",
                      flags={"/srv/lab-7/build": "-O2"})
    facts["fallback_reason"] = "no kernel at /srv/lab-7/kernels/attn.so; used default"
    run = registered_with(world, {"test_attention": facts})
    found = _query(world)
    shown = _by_run(found)[run]["environment"]["components"]["test_attention"]
    assert shown["fallback_reason"] == "no kernel at a server path; used default"
    assert shown["source"]["compiler"] == "a server path 13.3"
    assert shown["source"]["flags"] == {"a server path": "-O2"}
    text = results.to_json(found) + results.to_csv(found)
    assert "/srv/lab-7" not in text and "/opt/toolchains" not in text


def test_the_exports_columns_keep_their_names_and_order_with_components_added():
    from src.api.services import results

    assert tuple(name for name in results.COLUMN_NAMES if name != "components") == (
        COLUMNS_BEFORE_COMPONENTS)
    names = list(results.COLUMN_NAMES)
    assert names.index("components") == names.index("native_identity") + 1


# ---------------------------------------------------------------------------
# Review fix RC-R1: one plain-data rule for every value read from a file and shown
# ---------------------------------------------------------------------------
#: A lone surrogate: JSON's escapes let a file hold one (``\\ud800``), and no
#: answer can encode it.
LONE = chr(0xd800)
#: Valid text of several scripts, which every answer carries exactly.
SCRIPTS = "編譯器 ✓ café 🧬"


def _command_line(world, tmp_path, fmt):
    """The command line's export of the world's sources, as `_sources` names
    them, to a new file: its exit status and the file."""
    from scripts import export_results

    out = tmp_path / f"export-{fmt}.{fmt}"
    code = export_results.main(["--workspaces-root", f"a={world['root_a']}",
                                "--workspaces-root", f"b={world['root_b']}",
                                "--registry", f"main={world['registry']}", "--format", fmt,
                                "--output", str(out)])
    return code, out


def _reviewers_case(report):
    """The reviewer's case (RC-R1): a component's source text, as a record read
    from a file holds it."""
    facts = attention(False)
    facts["source"]["compiler"] = "cc " + LONE
    report["manifest"]["environment"]["facts"]["components"] = {"test_attention": facts}


def _nested_key(report):
    facts = attention(False)
    facts["source"]["flags"] = {"MAX" + LONE: "8"}
    report["manifest"]["environment"]["facts"]["components"] = {"test_attention": facts}


def _gpu_name(report):
    report["manifest"]["environment"]["facts"]["gpu"] = {"name": "GB10 " + LONE,
                                                         "driver": "580"}


@pytest.mark.parametrize("edit, refused", [
    (_reviewers_case, "environment.facts.components.test_attention.source.compiler is not "
                      "text: character 3 is a lone surrogate"),
    (_nested_key, "environment.facts.components.test_attention.source.flags has a key that is "
                  "not text: character 3 is a lone surrogate"),
    (_gpu_name, "environment.facts.gpu.name is not text: character 5 is a lone surrogate"),
], ids=["component-source-text", "component-nested-key", "gpu-name"])
def test_text_no_answer_can_carry_is_an_unreadable_item_and_every_other_result_exports(
        world, tmp_path, edit, refused):
    """RC-R1: a lone surrogate in a value a view shows made the reader list the
    run as a result, and then the whole CSV and JSON export, the command line
    (leaving empty files) and the record view's JSON answer failed. Now its
    kept report and its ledger record are each an item that could not be read,
    with a reason naming the field and why, never the value; every other
    result is read and exported; both formats encode, and the command line
    exits 0 with complete files."""
    from src.api.services import results

    before = set(_by_run(_query(world)))
    report = report_for(world["child"], version_block(LABEL, "research_selection"),
                        metrics=_metrics(0.5))
    edit(report)
    path, _ = register(world["registry"], world["child_path"], report)
    run = path.parent.name

    found = _query(world)
    assert set(_by_run(found)) == before
    unread = [row for row in found if row["result"] == "unreadable" and row.get("run_id") == run]
    assert sorted(row["item"] for row in unread) == ["kept report", "ledger record"]
    for row in unread:
        assert refused in row["reason"], row["reason"]
    csv_text, json_text = results.to_csv(found), results.to_json(found)
    for text in (csv_text, json_text):
        text.encode("utf-8")
        assert LONE not in text and "\\ud800" not in text
    assert {row["run_id"] for row in json.loads(json_text)["rows"]} >= before | {run}
    for fmt, text in (("csv", csv_text), ("json", json_text)):
        code, out = _command_line(world, tmp_path, fmt)
        assert code == 0
        assert out.read_bytes() == text.encode("utf-8")


def test_a_measurement_setting_no_answer_can_carry_is_unknown_with_why_and_the_result_kept(
        world):
    """RC-R1: a software revision holding a lone surrogate is a measurement
    setting the record holds as no answer can carry it: that setting reads
    unknown, saying why without the value, and the result is kept and
    exported (each reader keeps its own granularity: a setting is read field
    by field, `recorded_settings`)."""
    from src.api.services import results

    report = report_for(world["child"], version_block(LABEL, "research_selection"),
                        metrics=_metrics(0.5))
    report["manifest"]["software_revision"] = "rev " + LONE
    path, _ = register(world["registry"], world["child_path"], report)
    found = _query(world)
    row = _by_run(found)[path.parent.name]
    assert "software_revision" not in row["measurement"]
    assert row["unknown"]["measurement.regime.software_revision"] == (
        "its record holds it as a value no answer can carry: it is not text: character 4 is a "
        "lone surrogate")
    (cell,) = [r for r in csv.DictReader(io.StringIO(results.to_csv(found)))
               if r["run_id"] == path.parent.name and r["metric_name"] == "pooled_mrr"]
    assert cell["software_revision"] == results.UNKNOWN
    results.to_json(found).encode("utf-8")


@pytest.mark.parametrize("setting, words", [
    ({"optimizer_note": "adam " + LONE},
     "its config.requested.optimizer_note is not text: character 5 is a lone surrogate, so it "
     "is not shown"),
    ({"note" + LONE: 1},
     "its config.requested has a key that is not text: character 4 is a lone surrogate, so it "
     "is not shown"),
], ids=["setting-text", "setting-key"])
def test_a_training_setting_no_answer_can_carry_is_not_shown_says_why_and_the_list_answers(
        world, setting, words):
    """RC-R1: the checkpoint listing's training settings were never checked,
    so one setting holding a lone surrogate made the listing's JSON answer,
    the query's rows and the export fail. Now that part is not shown and says
    why, by its field and never its value; the file's other parts, its record
    and its results are listed and exported. (A number that is not finite is
    a value a setting may record, and is shown:
    `test_a_training_setting_that_is_not_finite_is_shown_as_its_text_everywhere`.)"""
    from starlette.responses import JSONResponse

    from src.api.services import results, server_lists
    from tests.unit import test_task_first_selection as bundles

    payload = _training_state(bundles._record_payload(family_id="odd-settings"))
    payload["config"]["requested"].update(setting)
    checkpoint, digest = _put(world["root_a"] / "lab", "20261005-000000_base_eeeeee",
                              "last.pt", payload)
    listing = server_lists.list_checkpoints(
        "lab", environ={"SHEPHERD_WORKSPACES_ROOT": str(world["root_a"])})
    (item,) = [item for item in listing.items if item.sha256 == digest]
    assert item.problem is None and item.record == "model" and item.family_id == "odd-settings"
    assert item.training_settings["learning"] is None
    assert item.training_settings["unshown"] == {"learning": words}
    assert item.training_settings["unrecorded"] == []
    assert item.training_settings["effective"]["device"]["effective"] == "cpu"
    JSONResponse(listing.as_dict("checkpoints"))  # the list answers

    report = report_for(digest, version_block(LABEL, "research_selection"),
                        metrics=_metrics(0.5))
    path, _ = register(world["registry"], checkpoint, report)
    found = _query(world)
    model = _by_run(found)[path.parent.name]["model"]
    assert model["learning_settings"] is None and model["family_id"] == "odd-settings"
    assert model["effective_settings"]["device"]["effective"] == "cpu"
    assert _by_run(found)[path.parent.name]["unknown"]["model.learning_settings"] == (
        f"its checkpoint records it, but {words}")
    results.to_csv(found).encode("utf-8")
    results.to_json(found).encode("utf-8")


@pytest.mark.parametrize("number, text", [
    (float("inf"), "inf"), (float("-inf"), "-inf"), (float("nan"), "nan"),
], ids=["inf", "minus-inf", "nan"])
def test_a_training_setting_that_is_not_finite_is_shown_as_its_text_everywhere(
        world, tmp_path, number, text):
    """RC-R1 review: a number that is not finite is a value a training setting
    may record (``max_grad_norm: inf``, which `training_fields.value_problem`
    accepts), and the JSON export always showed it, as its text. The first
    RC-R1 fix hid the checkpoint's whole learning part for it (in the list,
    the rows, Compare and both exports). Now the one listing reader gives it
    its text, so the list's JSON answer, the query's row, the CSV and the JSON
    all carry it, beside every other setting of its part; a nested one (the
    schedule's rate) too. The JSON export shows what it showed before
    (``"inf"``); the CSV cell, compact JSON, now holds the same text where it
    held ``Infinity``, which is not JSON."""
    from starlette.responses import JSONResponse

    from src.api.services import results, server_lists
    from src.config.training_fields import value_problem
    from tests.unit import test_task_first_selection as bundles

    assert value_problem("max_grad_norm", float("inf")) is None  # a value a run is given
    payload = _training_state(bundles._record_payload(family_id="not-finite"))
    payload["config"]["requested"]["max_grad_norm"] = number
    payload["config"]["effective"]["schedule"] = {"type": "cosine", "base_lr": number}
    checkpoint, digest = _put(world["root_a"] / "lab", "20261005-000000_base_eeeeee",
                              "last.pt", payload)
    listing = server_lists.list_checkpoints(
        "lab", environ={"SHEPHERD_WORKSPACES_ROOT": str(world["root_a"])})
    (item,) = [item for item in listing.items if item.sha256 == digest]
    learning = {"learning_rate": 0.001, "seed": 7, "batch_size": 16, "max_grad_norm": text}
    assert item.training_settings["learning"] == learning
    assert item.training_settings["effective"]["schedule"] == {"type": "cosine",
                                                               "base_lr": text}
    assert item.training_settings["unshown"] == {} and item.training_settings["unrecorded"] == []
    JSONResponse(listing.as_dict("checkpoints"))  # the list answers

    report = report_for(digest, version_block(LABEL, "research_selection"),
                        metrics=_metrics(0.5))
    path, _ = register(world["registry"], checkpoint, report)
    found = _query(world)
    row = _by_run(found)[path.parent.name]
    assert row["model"]["learning_settings"] == learning
    assert row["model"]["effective_settings"]["schedule"]["base_lr"] == text
    assert "model.learning_settings" not in row["unknown"]
    csv_text, json_text = results.to_csv(found), results.to_json(found)
    (exported,) = [r for r in json.loads(json_text)["rows"]
                   if r["run_id"] == path.parent.name and r["metric_name"] == "pooled_mrr"]
    assert exported["learning_settings"] == learning
    (cell,) = [r for r in csv.DictReader(io.StringIO(csv_text))
               if r["run_id"] == path.parent.name and r["metric_name"] == "pooled_mrr"]
    assert json.loads(cell["learning_settings"]) == learning
    for fmt, exported_text in (("csv", csv_text), ("json", json_text)):
        code, out = _command_line(world, tmp_path, fmt)
        assert code == 0 and out.read_bytes() == exported_text.encode("utf-8")


def test_valid_unicode_is_kept_exactly_through_the_query_the_csv_and_the_json(world, tmp_path):
    """RC-R1 keeps valid text: Chinese, symbols, accents and emoji in a
    component's source and fallback reason, the GPU's name, the software
    revision and a training setting are kept byte for byte through the query,
    the CSV and the JSON (and the command line's files); no ASCII-only rule."""
    from src.api.services import results
    from tests.unit import test_task_first_selection as bundles

    texts = {"compiler": SCRIPTS, "fallback": "回退：dtype — é", "gpu": "GB10 測試 🚀",
             "revision": "修訂-ü", "setting": "学习率 ✓"}
    facts = attention(False)
    facts["source"]["compiler"] = texts["compiler"]
    facts["fallback_reason"] = texts["fallback"]
    payload = _training_state(bundles._record_payload(family_id="unicode-settings"))
    payload["config"]["requested"]["optimizer_note"] = texts["setting"]
    checkpoint, digest = _put(world["root_a"] / "lab", "20261006-000000_base_ffffff",
                              "last.pt", payload)
    report = report_for(digest, version_block(LABEL, "research_selection"),
                        metrics=_metrics(0.5))
    report["manifest"]["environment"]["facts"]["components"] = {"test_attention": facts}
    report["manifest"]["environment"]["facts"]["gpu"] = {"name": texts["gpu"], "driver": "580"}
    report["manifest"]["software_revision"] = texts["revision"]
    path, _ = register(world["registry"], checkpoint, report)

    found = _query(world)
    row = _by_run(found)[path.parent.name]
    shown = row["environment"]["components"]["test_attention"]
    assert (shown["source"]["compiler"], shown["fallback_reason"]) == (texts["compiler"],
                                                                     texts["fallback"])
    assert row["environment"]["gpu"]["name"] == texts["gpu"]
    assert row["measurement"]["software_revision"] == texts["revision"]
    assert row["model"]["learning_settings"]["optimizer_note"] == texts["setting"]
    csv_text, json_text = results.to_csv(found), results.to_json(found)
    (exported,) = [r for r in json.loads(json_text)["rows"]
                   if r["run_id"] == path.parent.name and r["metric_name"] == "pooled_mrr"]
    assert exported["components"]["test_attention"]["source"]["compiler"] == texts["compiler"]
    assert exported["gpu"]["name"] == texts["gpu"]
    for fmt, text in (("csv", csv_text), ("json", json_text)):
        data = text.encode("utf-8")
        for value in texts.values():
            assert value.encode("utf-8") in data, (fmt, value)
        code, out = _command_line(world, tmp_path, fmt)
        assert code == 0 and out.read_bytes() == data


def test_a_path_no_view_shows_never_hides_its_result_whatever_it_holds(world):
    """The rule reads what a reader shows, not what stays in the ledger: an
    R10 evidence path (a server path no view or export shows) holding a lone
    surrogate, as a workspace whose directory name is not valid UTF-8 gives
    one, leaves the result read and exported, and its path out of every
    output."""
    from src.api.services import results
    from tests.unit.test_evaluation_sidecar import r10_block

    r10 = r10_block("none")
    r10["training"]["evidence"]["path"] = "/srv/lab-" + chr(0xdcff) + "/train_samples.json"
    report = report_for(world["child"], version_block(LABEL, "research_selection"),
                        metrics=_metrics(0.5), r10=r10)
    path, _ = register(world["registry"], world["child_path"], report)
    found = _query(world)
    row = _by_run(found)[path.parent.name]
    assert row["exposure"]["r10"]["training"]["evidence"] == {"state": "found"}
    for text in (results.to_csv(found), results.to_json(found)):
        text.encode("utf-8")
        assert "/srv/lab-" not in text


def _redacted_compiler(facts):
    """A compiler path the one D7 redaction replaces, under a directory whose
    name is not valid UTF-8 (``os.fsdecode`` gives such a byte as a lone
    surrogate)."""
    facts["source"]["compiler"] = "/opt/lab-" + chr(0xdcff) + "/bin/cc"


def _out_of_shape(facts):
    """A fact outside the component's shape (a newer writer's), which the one
    projection leaves out."""
    facts["future_note"] = "x" + LONE


@pytest.mark.parametrize("edit, compiler", [
    (_redacted_compiler, "a server path"), (_out_of_shape, "audit-compiler 1.0"),
], ids=["redacted-path", "out-of-shape-fact"])
def test_a_component_text_no_view_shows_never_hides_its_result_whatever_it_holds(
        world, tmp_path, edit, compiler):
    """RC-R1 review: the record reader held each component as written to the
    rule, so a lone surrogate in text no view shows (a path the D7 redaction
    replaces, a field outside the shape) made its kept report and its ledger
    record unreadable, where they were read and exported before. The rule now
    reads what the one projection shows (`recorded_components`): the run is
    one result, both formats and the command line encode, and the surrogate
    is in no output."""
    from src.api.services import results

    facts = attention(False)
    edit(facts)
    run = registered_with(world, {"test_attention": facts})
    found = _query(world)
    assert [row["result"] for row in found if row.get("run_id") == run] == ["run"]
    shown = _by_run(found)[run]["environment"]["components"]["test_attention"]
    assert shown["source"]["compiler"] == compiler and "future_note" not in shown
    for fmt, render in (("csv", results.to_csv), ("json", results.to_json)):
        data = render(found).encode("utf-8")
        assert b"\\udcff" not in data and b"\\ud800" not in data and b"/opt/lab-" not in data
        code, out = _command_line(world, tmp_path, fmt)
        assert code == 0 and out.read_bytes() == data


def _ledger_records_hold(world, edit):
    """A hand edit of the child's ledger, the sanctioned correction path: each
    record naming components has `edit` applied to its ``test_attention``."""
    ledger_path = world["child_path"].parent / "evaluations.json"
    ledger = json.loads(ledger_path.read_text())
    for record in ledger["records"]:
        components = record.get("environment", {}).get("facts", {}).get("components")
        if components:
            edit(components["test_attention"])
    ledger_path.write_text(json.dumps(ledger))


def test_a_refusal_names_a_fact_by_what_the_view_shows_never_a_server_path(world):
    """RC-R1 review: a refused component fact was named from the record as
    written, so a key the view shows as "a server path" (an absolute path) was
    written out in the reason, and the CSV and JSON ``reason`` carried it. Now
    the refusal names the projection's own keys: for a lone surrogate (kept
    report and ledger record) and a number that is not finite (a hand edit of
    the ledger) under such a key, the reason says "a server path", and no
    output holds the path."""
    from src.api.services import results

    secret = "/home/secret-user/build"
    facts = attention(False)
    facts["source"]["flags"] = {secret: "x" + LONE}
    surrogate = registered_with(world, {"test_attention": facts})
    found = _query(world)
    unread = [row for row in found
              if row["result"] == "unreadable" and row.get("run_id") == surrogate]
    assert sorted(row["item"] for row in unread) == ["kept report", "ledger record"]
    for row in unread:
        assert ("environment.facts.components.test_attention.source.flags.a server path is "
                "not text: character 1 is a lone surrogate") in row["reason"], row["reason"]

    not_finite = registered_with(world, {"test_attention": attention(False)})
    _ledger_records_hold(world, lambda entry: entry["source"].update(
        flags={secret: float("nan")}))
    found = _query(world)
    (row,) = [row for row in found if row["result"] == "unreadable"
              and row.get("run_id") == not_finite]
    assert row["item"] == "ledger record"
    assert row["reason"] == ("environment.facts.components.test_attention.source.flags."
                             "a server path is not a finite number")
    for text in (results.to_csv(found), results.to_json(found)):
        text.encode("utf-8")
        assert "/home/secret-user" not in text and "secret-user" not in text


def test_a_component_nested_deeper_than_a_report_may_is_unreadable_not_a_crash(world):
    """The one projection walks a component as written, so the reader first
    holds the block to a report's own depth (`MAX_REPORT_DEPTH`, counted, not
    recursed): a hand-edited ledger record whose component fact nests deeper
    than the interpreter's stack is one unreadable record, never a
    `RecursionError` of the whole query; every other result is read."""
    from src.api.services import results

    before = set(_by_run(_query(world)))
    run = registered_with(world, {"test_attention": attention(False)})
    deep: list = []
    node = deep
    for _ in range(2000):
        node.append([])
        node = node[0]
    _ledger_records_hold(world, lambda entry: entry["source"].update(flags=deep))
    found = _query(world)
    assert set(_by_run(found)) >= before
    (row,) = [row for row in found if row["result"] == "unreadable" and row.get("run_id") == run]
    assert row["reason"] == ("environment.facts.components nests deeper than 64 levels, which "
                             "no measurement writes")
    results.to_csv(found).encode("utf-8")
    results.to_json(found).encode("utf-8")


def _nested_facts(levels, leaf):
    """`leaf` under `levels` nested objects, each named by its level."""
    for level in range(levels):
        leaf = {f"level_{level}": leaf}
    return leaf


def _record_views_of(world, run):
    """The run's ledger record as the one reader's view (`read_records`) and the
    reader's unreadable items naming the run."""
    from src.api.services.record_views import read_records
    from src.evaluation.sidecar import read_ledger

    ledger, _ = read_ledger(world["child_path"].parent / "evaluations.json")
    viewed, unreadable = read_records(ledger, world["child"])
    return ([view for _, view in viewed if view.run_id == run],
            [item for item in unreadable if item.run_id == run])


def test_a_record_holding_many_and_deep_component_facts_is_read_queried_and_exported(
        world, tmp_path):
    """RC-R1 advice (1), the reviewer's case: a component recording 10,001
    legitimate build facts, and one nested deeper than 16 levels, which the
    shared view returned whole before. The rule's walk carried the
    checkpoint reader's bounds (more than 10,000 values; 16 levels, 17 with
    the component's name) over to the facts the collector records and to
    the whole record, so the collector refused the component and the reader
    the record. A value parsed from JSON is now held to no count and no depth
    of the rule's: the collector records it, its kept report and its ledger
    record are one result, and the query, both formats, the command line and
    the record views' JSON answer carry every fact as recorded."""
    from starlette.responses import JSONResponse

    from src.api.services import results

    build = {f"fact_{index:05d}": f"flag {index}" for index in range(10_001)}
    facts = attention(False, build=build, nested=_nested_facts(20, "the deepest fact"))
    run = registered_with(world, {"test_attention": facts})
    found = _query(world)
    assert [row["result"] for row in found if row.get("run_id") == run] == ["run"]
    row = _by_run(found)[run]
    assert (row["report"], row["registered"]) == (results.REPORT_KEPT, results.REGISTERED)
    assert row["environment"]["components"] == {"test_attention": facts}

    exported = json.loads(results.to_json(found))["rows"]
    (cell,) = [r for r in exported if r["run_id"] == run and r["metric_name"] == "pooled_mrr"]
    assert cell["components"] == {"test_attention": facts}
    limit = csv.field_size_limit(1 << 24)  # this reader's own limit: the cell is ~230 KB
    try:
        (line,) = [r for r in csv.DictReader(io.StringIO(results.to_csv(found)))
                   if r["run_id"] == run and r["metric_name"] == "pooled_mrr"]
    finally:
        csv.field_size_limit(limit)
    assert json.loads(line["components"]) == {"test_attention": facts}
    for fmt, render in (("csv", results.to_csv), ("json", results.to_json)):
        code, out = _command_line(world, tmp_path, fmt)
        assert code == 0 and out.read_bytes() == render(found).encode("utf-8")

    (view,), unreadable = _record_views_of(world, run)
    assert unreadable == []
    answer = json.loads(JSONResponse(view.model_dump(mode="json")).body)
    assert answer["components"] == {"test_attention": facts}


def _nested_list(levels, leaf="kept"):
    for _ in range(levels):
        leaf = [leaf]
    return leaf


def test_a_collected_component_is_held_to_the_depth_its_report_can_be_registered_with(world):
    """RC-R1 advice (1): the one depth the collector holds a component to is
    evaluated, not carried: as deep as its run's report can hold it and still
    be registered (`run_environment.COMPONENT_DEPTH`, pinned here against the
    registration's own bound, `sidecar.MAX_REPORT_DEPTH`). The deepest the
    collector records registers and reads as a result; one level deeper it
    refuses, naming where and why, as a report holding it would be refused."""
    from src.evaluation.sidecar import MAX_REPORT_DEPTH, ReportNestsTooDeeply, build_record
    from src.utils.run_environment import COMPONENT_DEPTH, component_facts

    def collected(levels):
        return component_facts("attention kernel", installed=True, source={
            "state": "recorded", "route": "local build", "nested": _nested_list(levels)})

    deepest = COMPONENT_DEPTH - 2  # below the facts' own object and their source
    assert MAX_REPORT_DEPTH == 64 and COMPONENT_DEPTH == MAX_REPORT_DEPTH - 6
    run = registered_with(world, {"test_attention": collected(deepest)})
    assert [row["result"] for row in _query(world) if row.get("run_id") == run] == ["run"]
    with pytest.raises(ValueError) as refused:
        collected(deepest + 1)
    assert str(refused.value) == (f"component.source.nested{'[0]' * deepest} nests deeper than "
                                  f"{COMPONENT_DEPTH} levels")
    held = collected(deepest)
    held["source"]["nested"] = [held["source"]["nested"]]  # one level more, by hand
    report = report_for(world["child"], version_block(LABEL, "research_selection"),
                        metrics=_metrics(0.5))
    report["manifest"]["environment"]["facts"]["components"] = {"test_attention": held}
    with pytest.raises(ReportNestsTooDeeply):
        build_record(report, "0" * 64)


def test_an_annotation_no_view_shows_leaves_its_record_read_exactly_as_before(world, tmp_path):
    """RC-R1 advice (2): the rule reads what is used and published, never
    unused extra content. A record whose only text no answer can carry lies
    in an annotation no view shows (an extra fact of its environment, a value
    and a key holding a lone surrogate), in its kept report and its ledger
    record alike, reads exactly as it reads without it: one result, the same
    views, nothing of the annotation in any output, both formats, the command
    line and the record views' JSON answer encoded."""
    from starlette.responses import JSONResponse

    from src.api.services import results
    from src.api.services.record_views import read_records, result_view
    from src.evaluation.sidecar import build_record, read_ledger

    report = report_for(world["child"], version_block(LABEL, "research_selection"),
                        metrics=_metrics(0.5))
    report["manifest"]["environment"]["facts"]["operator_note"] = {
        "text": "annotated " + LONE, "seen" + LONE: True}
    path, _ = register(world["registry"], world["child_path"], report)
    run = path.parent.name
    found = _query(world)
    assert [row["result"] for row in found if row.get("run_id") == run] == ["run"]
    row = _by_run(found)[run]
    assert (row["report"], row["registered"]) == (results.REPORT_KEPT, results.REGISTERED)

    def without_note(record):
        plain = copy.deepcopy(record)
        del plain["environment"]["facts"]["operator_note"]
        return plain

    data = path.read_bytes()
    kept = build_record(json.loads(data), hashlib.sha256(data).hexdigest())
    assert "operator_note" in kept["environment"]["facts"]
    assert result_view(kept) == result_view(without_note(kept))
    ledger, _ = read_ledger(world["child_path"].parent / "evaluations.json")
    plain_ledger = copy.deepcopy(ledger)
    plain_ledger["records"] = [without_note(record) if record.get("run_id") == run else record
                               for record in ledger["records"]]
    annotated, unread = read_records(ledger, world["child"])
    plain, unread_plain = read_records(plain_ledger, world["child"])
    assert [view for _, view in annotated] == [view for _, view in plain]
    assert unread == unread_plain
    (view,), unreadable = _record_views_of(world, run)
    assert unreadable == []
    JSONResponse(view.model_dump(mode="json"))

    for fmt, render in (("csv", results.to_csv), ("json", results.to_json)):
        encoded = render(found).encode("utf-8")
        assert b"operator_note" not in encoded and b"annotated" not in encoded
        code, out = _command_line(world, tmp_path, fmt)
        assert code == 0 and out.read_bytes() == encoded


class _ReadingFails(dict):
    """A part whose reading fails as a programming error or the machine would,
    never as the rule's refusal."""

    def __init__(self, error):
        super().__init__()
        self.error = error

    def __iter__(self):
        raise self.error


@pytest.mark.parametrize("error", [RuntimeError("a programming error"), MemoryError()],
                         ids=["programming-error", "memory-error"])
def test_the_rows_guard_makes_only_the_rules_refusal_an_unreadable_item(world, monkeypatch,
                                                                         error):
    """RC-R1 advice (3): the results service's last-line guard (`_carried`)
    reads a row by the one rule and makes only the rule's own refusal
    (`NotPlainData`) the existing unreadable item; an unexpected exception
    met while reading a row (a programming error, `MemoryError`) propagates
    and is never reported as a damaged record."""
    from src.api.services import results

    real = results._exposure
    monkeypatch.setattr(results, "_exposure",
                        lambda view: {**real(view), "reading": _ReadingFails(error)})
    with pytest.raises(type(error)) as raised:
        _query(world)
    assert raised.value is error


def test_what_the_service_itself_reads_is_shown_only_as_an_answer_can_carry_it(
        world, monkeypatch):
    """RC-R1, the service's own readers: a version's content references (read
    from its manifest) and a group's description (named by the store's other
    records) holding a lone surrogate are not shown, each saying why; the
    results are kept and exported."""
    from src.api.services import record_states, results
    from src.evaluation import cohort_registry

    real_check, real_group = cohort_registry.check_references, record_states.group_state

    def check(version, **where):
        states = list(real_check(version, **where))
        reference = states[0].reference._replace(sha256="a" * 63 + LONE)
        return (states[0]._replace(reference=reference), *states[1:])

    def group(store, key, *, operator):
        return {**real_group(store, key, operator=operator), "state": "contested",
                "differing": ["pooled_mrr" + LONE]}

    monkeypatch.setattr(cohort_registry, "check_references", check)
    monkeypatch.setattr(record_states, "group_state", group)
    found = _query(world)
    row = _by_run(found)[world["registered"].parent.name]
    assert row["references"] is None
    assert row["unknown"]["references"] == (
        "its data version's manifest gives what no answer can carry (references[0].sha256 is "
        "not text: character 63 is a lone surrogate), so its content references are not shown")
    assert (row["group"]["state"], row["group"]["differing"]) == ("cannot_be_checked", [])
    assert row["group"]["why"] == (
        "the listed stores of its version name what no answer can carry (differing[0] is not "
        "text: character 10 is a lone surrogate)")
    for text in (results.to_csv(found), results.to_json(found)):
        text.encode("utf-8")


def test_an_items_name_that_is_not_text_is_left_out_saying_so(world):
    """RC-R1: a directory the registry lists whose name is not valid UTF-8 (a
    lone surrogate once read) is an item that could not be read, as before,
    and its name, which no answer can carry, is left out of it, saying so;
    every result is still read and exported."""
    from src.api.services import results

    versions = os.fsencode(str(world["registry"] / "versions"))
    os.mkdir(versions + b"/odd-\xff")
    found = _query(world)
    (item,) = [row for row in found if row["result"] == "unreadable"
               and row["item"] == "data version" and "label" not in row["where"]]
    assert item["reason"].endswith(
        "(its label is not text: character 4 is a lone surrogate, so it is not shown)")
    assert world["registered"].parent.name in _by_run(found)
    for text in (results.to_csv(found), results.to_json(found)):
        text.encode("utf-8")


def test_a_row_an_answer_cannot_carry_is_an_unreadable_item_never_cleaned(world, monkeypatch):
    """RC-R1, the service's own contract: a row it would list that still holds
    what no answer can carry (here a part made by hand) is an item that could
    not be read, by where its result was found and why, naming the field and
    never the value; every other row is listed, and nothing is altered."""
    from src.api.services import results

    run = world["registered"].parent.name
    real = results._exposure

    def exposure(view):
        shown = real(view)
        if view.source_artifact_digest == _digest(world["registered"]):
            shown = {**shown, "reading": "held " + LONE}
        return shown

    monkeypatch.setattr(results, "_exposure", exposure)
    found = _query(world)
    assert run not in _by_run(found)
    (item,) = [row for row in found if row["result"] == "unreadable" and row["item"] == "result"]
    assert item["run_id"] == run and item["report_sha256"] == _digest(world["registered"])
    assert item["where"] == {"source": "main", "label": LABEL, "run_id": run,
                             "report_sha256": _digest(world["registered"])}
    assert item["reason"] == ("its row holds what no answer can carry (exposure.reading is not "
                              "text: character 5 is a lone surrogate), so it is not shown as a "
                              "result")
    for text in (results.to_csv(found), results.to_json(found)):
        text.encode("utf-8")


def test_a_failing_export_leaves_no_file(world, tmp_path, monkeypatch):
    """RC-R1: the command line opened its file before the export was encoded,
    so a failure left an empty file that the never-overwrite rule then
    refused. The whole export is encoded first and the file published whole:
    a failure leaves none, and the next export writes it."""
    from src.api.services import results

    monkeypatch.setattr(results, "to_csv", lambda rows: "a" + LONE)
    out = tmp_path / "export-csv.csv"
    with pytest.raises(UnicodeEncodeError):
        _command_line(world, tmp_path, "csv")
    assert not out.exists() and list(tmp_path.glob("export-csv.csv*")) == []
    monkeypatch.undo()
    code, out = _command_line(world, tmp_path, "csv")
    assert code == 0 and out.read_text(encoding="utf-8").startswith("result,")


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
              # Review fix RC-P2-1: every recorded component fact, beside the
              # summary (a named addition).
              "native_identity", "components", "dataset_origin", "scoring_semantics_digest",
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
