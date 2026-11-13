"""
Research › Evaluate › Models (M5-2, UI-D): the research preset of the shared list.
==================================================================================
`src/webui/components/models_page.py`. What these pin, on checkpoints listed by
the server, ledgers written by the ledger's own functions and reports kept in a
registry's report store (`tests/fixtures/model_records.py`):

- **By test score retired (K8-6d)**: the Order switch offers Newest first and
  By family only; no limit of the retired ordering (its role list, its Mode C
  rule, one record per checkpoint) is left: the retired setting changes
  nothing, and no record is dropped or set apart for its family, training
  dataset or cohort version (a guard); a kept ``model_load`` usage entry is
  shown with its recorded fields;
- **D5 (policy rev 16)**: a record whose group's runs differ is described
  ("runs under these conditions differ", with what differs; neutral, and amber
  with ▲ and "— investigate" only for a group that records deterministic
  algorithms on), a
  comparison that is incomplete is never read as agreeing, and every record is
  shown with its run ("not recorded" for one merged before each run had its own
  record);
- **task and family**: the Task select's options (the list's one vocabulary,
  short, unique); every row's kind in the shared list's Model cell, under
  every order; By family's groups (parent first, indented Lineage cells, a
  context row in grey with ○, an unavailable parent in amber with ▲, escaped;
  the not-readable and not-read groups under All, each under its own word),
  the Order switch on for every operator's list;
- **the detail**: what the model is (kind, operation, parent and where it is,
  family, encoder, relation to the paper) without Model Facts' Task and
  Configuration rows; Trainer validation in its own card, the test records in
  theirs, never together; a registered task record under its own metric
  names; *unverifiable* never a number; the usage history segment (this
  checkpoint's entries, newest first, read through the page's cohort version
  list: a guard); ledger states said, never a path;
- **read-only (D8)**: no list, D8's three conditions named, and "Recorded
  results of the model in use" read by the API's read-only route function from
  the ledger beside the served checkpoint, by its digest, opening no
  checkpoint; no server path, even when a report store cannot be read; the
  status unreadable said as such;
- **the wiring**: the page refreshes when it is shown, reads again a new
  listing and once after every Load (a Load that changed nothing listed
  included), selects the same checkpoint again after a Load, redraws on its
  controls and its selection, all through one listener; the dataset identity
  under the workspace select; escaping.

Module: tests/unit/test_webui_models_page.py
"""
from __future__ import annotations

import html as html_lib
import json
import os
import re
import shutil
import time

import pytest

gr = pytest.importorskip("gradio")

from src.evaluation.caveats import mode_label  # noqa: E402
from src.utils.fingerprint import file_sha256  # noqa: E402
from src.webui.components import model_list as ml  # noqa: E402
from src.webui.components import models_page as mp  # noqa: E402
from src.webui.components import renderers as r  # noqa: E402
from src.webui.components.renderers import e  # noqa: E402
from tests.fixtures.model_records import (  # noqa: E402
    ACCEPTANCE_ROLE,
    RESEARCH_ROLE,
    disagree,
    register,
    report_for,
    version_block,
)
from tests.fixtures.operator_access import gradio_request, operator_request  # noqa: E402

MYGENE = "mygene2-2026-09"
INST = "inst-accept-2026-08"
EXPLORE = "explore-2026-07"

#: (architecture, run, file, epoch, val_pooled_mrr), newest first as the listing orders them.
CHECKPOINTS = [
    ("hgt", "20261018-091200_run_8c2e7a", "model-12.pt", 12, 0.318),
    ("hgt", "20261018-091200_run_8c2e7a", "model-08.pt", 8, 0.297),
    ("hgt", "20261017-091500_run_3fa9c1", "model-05.pt", 5, 0.241),
    ("hgt", "20261017-091500_run_3fa9c1", "model-03.pt", 3, 0.196),
    ("hgt", "20261016-221000_run_b71d04", "model-40.pt", 40, 0.352),
    ("sage", "20261015-083000_run_e4c2a9", "model-30.pt", 30, 0.205),
    ("sage", "20261015-083000_run_e4c2a9", "broken.pt", None, None),
    ("hgt", "20261012-081500_run_51f0d3", "model-20.pt", 20, 0.224),
]


def _metrics(mrr):
    """A disease task report's metrics (`DISEASE_METRICS`) with this pooled MRR."""
    from tests.unit.test_evaluation_sidecar import DISEASE_METRICS

    return {**DISEASE_METRICS, "pooled_mrr": mrr}


@pytest.fixture(scope="module")
def demo_workspace(tmp_path_factory):
    from tests.fixtures.registered_cohort import write_demo_workspace

    return write_demo_workspace(tmp_path_factory.mktemp("models-build") / "demo")


@pytest.fixture
def world(demo_workspace, tmp_path, monkeypatch):
    """A workspaces root with one workspace and its runs, a registry with three
    versions (research_selection, institutional_acceptance, exploration), and
    disease task records beside the checkpoints (`report_for`, a disease
    measurement's report; K8-6d moved them from Mode C reports):

    - model-05: inst #1 (contested in the store), mygene #2 (non-deterministic,
      0.2307), mygene #3 (deterministic, 0.2310);
    - model-08: mygene, deterministic, 0.2250; model-20: mygene on CPU, 0.2195;
      model-40: mygene 0.3402, contested; model-03: explore only; model-12
      (served) and model-30: nothing; broken.pt: not readable.
    """
    from src.utils.checkpoint_io import write_checkpoint

    root = tmp_path / "root"
    workspace = shutil.copytree(demo_workspace, root / "demo")
    fingerprint = file_sha256(workspace / "split_manifest.json")
    now = time.time()
    paths, shas = {}, {}
    for index, (arch, run, name, epoch, val) in enumerate(CHECKPOINTS):
        directory = workspace / "runs" / arch / run / "checkpoints"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        if epoch is None:
            path.write_bytes(b"not a checkpoint")
        else:
            shas[name] = write_checkpoint(
                {"checkpoint_kind": "weights", "epoch": epoch, "logs": {"val_pooled_mrr": val},
                 "training_input_digests": {"split_manifest": fingerprint}}, path)
        stamp = now - index * 3600
        os.utime(path, (stamp, stamp))
        paths[name] = path
    registry = tmp_path / "registry"
    mygene, inst, explore = (version_block(MYGENE, RESEARCH_ROLE),
                             version_block(INST, ACCEPTANCE_ROLE),
                             version_block(EXPLORE, "exploration"))
    inst_report = report_for(shas["model-05.pt"], inst, metrics=_metrics(0.2051),
                             deterministic=False)
    register(registry, paths["model-05.pt"], inst_report, registered_at="2026-10-17T13:02:00Z")
    disagree(registry, inst_report, _metrics(0.2077))
    register(registry, paths["model-05.pt"],
             report_for(shas["model-05.pt"], mygene, metrics=_metrics(0.2307),
                        deterministic=False),
             registered_at="2026-10-17T14:12:00Z")
    det_path, det = register(registry, paths["model-05.pt"],
                             report_for(shas["model-05.pt"], mygene, metrics=_metrics(0.2310)),
                             registered_at="2026-10-17T15:31:00Z")
    register(registry, paths["model-08.pt"],
             report_for(shas["model-08.pt"], mygene, metrics=_metrics(0.2250)))
    register(registry, paths["model-20.pt"],
             report_for(shas["model-20.pt"], mygene, metrics=_metrics(0.2195), device="cpu"))
    hgt = report_for(shas["model-40.pt"], mygene, metrics=_metrics(0.3402))
    register(registry, paths["model-40.pt"], hgt)
    disagree(registry, hgt, _metrics(0.3398))
    register(registry, paths["model-03.pt"],
             report_for(shas["model-03.pt"], explore, metrics=_metrics(0.18)))
    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(root))
    monkeypatch.setenv("SHEPHERD_REGISTRY", str(registry))
    served = {"value": _status(workspace, shas["model-12.pt"])}
    monkeypatch.setattr(ml, "read_status", lambda: (served["value"], None))
    monkeypatch.setattr(mp, "read_status", lambda: (served["value"], None))
    return {"root": root, "workspace": workspace, "registry": registry, "paths": paths,
            "shas": shas, "served": served, "det_report": det_path, "det": det,
            "mygene_block": mygene}


def _status(workspace, sha):
    kg = json.loads((workspace / "split_manifest.json").read_text())["artifacts"]["kg"]
    return {
        "initialized": True, "current_data_dir": str(workspace),
        "checkpoint_meta": {"epoch": 12},
        "served_identity": {
            "scoring_mode": "disease_task",
            "checkpoint": {"state": "verified" if sha else "absent", "sha256": sha},
            "graph": {"state": "verified", "kg_sha256": kg,
                      "split_manifest_sha256": file_sha256(workspace / "split_manifest.json")},
            "shortest_paths": {"state": "absent"},
            "dataset_record": {"state": "verified", "origin": "synthetic", "sha256": "e" * 64},
            "settings": {"device": "cpu", "device_requested": "cpu", "eta": None,
                         "max_path_length": 4, "path_length_penalty": 0.5,
                         "max_paths_per_candidate": 10,
                         "use_direct_path_optimization": True,
                         "phenotype_normalisation_version": 1},
        },
    }


def _key(world, name):
    for arch, run, file, *_rest in CHECKPOINTS:
        if file == name:
            return f"{arch}/{run}/{file}"
    raise KeyError(name)


#: Model bundles beside the files written before the model spec (refused
#: records, listed with why: every record the world fixture registers is
#: theirs): a base and its disease and gene
#: children, filed under ``gatv2``, older than every file above so the orders
#: over those stay.
BASE_RUN = "20261001-000000_base_aaaaaa"
BUNDLES = {"base": (BASE_RUN, "base.pt"), "disease": ("20261002-000000_disease_bbbbbb", "last.pt"),
           "gene": ("20261003-000000_gene_cccccc", "last.pt")}


def bundle_key(kind):
    run, name = BUNDLES[kind]
    return f"gatv2/{run}/{name}"


def _add_bundles(world, *, orphan_parent_run=None):
    """Write the base and its children (`test_task_first_selection`'s record
    writer); with `orphan_parent_run`, one more disease child whose recorded
    parent is in that run and not available. Returns the files' digests."""
    from src.utils.checkpoint_io import write_checkpoint
    from tests.unit import test_task_first_selection as bundles

    assert bundles.BASE_RUN == BASE_RUN
    fingerprint = file_sha256(world["workspace"] / "split_manifest.json")
    meta = {"checkpoint_kind": "weights", "training_input_digests": {"split_manifest": fingerprint}}
    now, shas = time.time(), {}

    def put(run, name, record, epoch, age):
        path = world["workspace"] / "runs" / "gatv2" / run / "checkpoints" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        sha = write_checkpoint({**record, **meta, "epoch": epoch}, path)
        os.utime(path, (now - age * 3600,) * 2)
        world["paths"][f"{run}/{name}"] = path
        return sha

    shas["base"] = put(*BUNDLES["base"], bundles._record_payload(
        logs={"val_lp_accuracy": 0.81}), 4, 22)
    parent = {"sha256": shas["base"], "run": BASE_RUN}
    for age, task in ((20, "disease"), (21, "gene")):
        shas[task] = put(*BUNDLES[task], bundles._record_payload(
            task, logs={"val_pooled_mrr": 0.42}, parent=parent), 2, age)
    if orphan_parent_run is not None:
        orphan = "20261004-000000_disease_dddddd"
        shas["orphan"] = put(orphan, "last.pt", {**bundles._record_payload(
            "disease", logs={"val_pooled_mrr": 0.4},
            parent={"sha256": "9" * 64, "run": orphan_parent_run}), "family_id": orphan}, 3, 23)
    return shas


def _read(world, monkeypatch):
    listing = ml.list_models("demo", operator_request(monkeypatch), family=True)
    return listing, mp.read_page(listing)


def _text(markup):
    """What a reader sees of a cell: its text, entities as characters."""
    return html_lib.unescape(re.sub(r"<[^>]+>", "", markup)).replace("\xa0", " ")


def _words(markup):
    """What a reader sees of a block, cell by cell: each tag a space, every
    run of white space one space."""
    return " ".join(html_lib.unescape(re.sub(r"<[^>]+>", " ", markup)).split())


#: The controls as the page builds them by default: every task, newest first.
NEWEST_CTL = {"task": "all", "order": "newest"}


# --------------------------------------------------------------------------- D5
def test_a_contested_state_that_cannot_be_checked_is_never_shown_uncontested(world, monkeypatch):
    """The detail checks of two retired By test score tests, kept (K8-6d): a
    report store that cannot be read makes every key's state "cannot be
    checked", and a key the page holds no state for is "it was not read";
    neither is ever shown as uncontested (green)."""
    import src.evaluation.report_store as report_store

    def broken(version, **kwargs):
        raise OSError("store unreadable")

    monkeypatch.setattr(report_store, "scan_report_store", broken)
    listing, data = _read(world, monkeypatch)
    assert {s["state"] for s in data["contest"].values()} == {"cannot_be_checked"}
    records = mp.detail_parts(listing, data, _key(world, "model-05.pt"), NEWEST_CTL)["records"]
    assert "comparison incomplete" in records
    assert "registered #3" in records and "shep-tag-ok" not in records
    # A record key with no state in the page's data (read after the store was
    # asked, say): every record says its state was not read.
    data["contest"] = {}
    records = mp.detail_parts(listing, data, _key(world, "model-05.pt"), NEWEST_CTL)["records"]
    assert records.count("comparison incomplete: it was not read") == 3
    assert "shep-tag-ok" not in records and "runs under these conditions differ" not in records


def test_a_record_whose_version_the_registry_does_not_list_cannot_be_checked(world, monkeypatch):
    from src.api.services.record_states import CohortVersions, contest_states, key_text
    from src.evaluation.sidecar import KEY_FIELDS

    resolver = CohortVersions([], None)
    key = key_text(tuple(world["det"][field] for field in KEY_FIELDS))
    state = contest_states([world["det"]], resolver)[key]
    assert state["state"] == "cannot_be_checked"
    assert "not one the server's registry lists" in state["problems"][0]
    # For a read-only request the reason's own words (which may name a path) stay out.
    assert contest_states([world["det"]], resolver, operator=False)[key]["problems"] == []


@pytest.mark.parametrize("bytes_now", ["torn", "another schema", "another version", "gone"])
def test_a_version_whose_bytes_changed_since_it_was_listed_is_said_changed(
        world, monkeypatch, bytes_now):
    """Root-cause round 3 (INT-R2): a version the page listed is read again for
    its usage history bound to the digest it was listed with
    (`CohortVersions.by_digest` -> `resolve_version(expected_sha256=)`,
    compared before the bytes are decoded). Other bytes put there after the
    listing -- torn, another schema, another valid version, or none -- are
    "changed since it was listed", never a version that "cannot be read", and
    none of its entries is shown."""
    import src.evaluation.cohort_registry as cohort_registry

    path = world["registry"] / "versions" / MYGENE / "cohort.json"
    listing = ml.list_models("demo", operator_request(monkeypatch))
    listed = mp.read_versions

    def listed_then_changed(environ=None):
        versions = listed(environ)
        if bytes_now == "gone":
            path.unlink()
        elif bytes_now == "torn":
            path.write_bytes(path.read_bytes()[:40])
        else:
            manifest = json.loads(path.read_bytes())
            if bytes_now == "another schema":
                manifest["schema_version"] = 99
            else:
                manifest["imported"]["at"] = "2026-10-22T00:00:00Z"
            path.write_bytes(cohort_registry.encode_json(manifest))
        return versions

    monkeypatch.setattr(mp, "read_versions", listed_then_changed)
    data = mp.read_page(listing)

    assert f"{MYGENE}: {MYGENE} changed since it was listed" in data["usage"]["problems"]
    assert MYGENE not in data["usage"]["read_from"]


def test_a_listed_version_that_cannot_be_read_is_named_in_the_usage_history(world, monkeypatch):
    """A version the registry lists with a problem has no usage history to
    read. The usage history names it and says why, as the retired By test
    score options once did; it is never skipped silently."""
    path = world["registry"] / "versions" / EXPLORE / "cohort.json"
    path.write_bytes(path.read_bytes()[:40])
    listing = ml.list_models("demo", operator_request(monkeypatch))
    data = mp.read_page(listing)

    problem = next(v["problem"] for v in data["versions"] if v["label"] == EXPLORE)
    assert problem
    said = f"{EXPLORE}: not read: {r.short_reason(problem, 220)}"
    assert data["usage"]["problems"] == [said]
    assert sorted(data["usage"]["read_from"]) == sorted([MYGENE, INST])
    usage = mp.detail_parts(listing, data, _key(world, "model-05.pt"),
                            {"task": mp.ALL_TASKS, "order": mp.NEWEST})["usage"]
    assert f'<p class="shep-mp-p shep-amber-t">{e(said)}</p>' in usage


@pytest.mark.parametrize("value, shown", [
    (float("nan"), "not a finite number"), (None, "not reported"),
    ("0.3402", "not a finite number"), ({"k": 1}, "not a finite number"),
    (10**400, "not a finite number"), (-10**400, "not a finite number"),
], ids=["nan", "null", "text", "object", "huge_int", "huge_negative_int"])
def test_a_disagreeing_value_that_is_not_a_finite_number_leaves_the_record_contested(
    world, monkeypatch, value, shown
):
    """The critic's break (P2-6): a report under model-40's contested key whose
    MRR is NaN, null, text, an object or an integer too large for a float (the
    verification's break: `math.isfinite` cannot take it, and an
    `OverflowError` took the page with it). The operator's view and the read-only
    view (the API's answer) both say "runs under these conditions differ"; the value is never
    formatted as a number: the operator's view says "not reported" for null and
    "not a finite number" otherwise, and the read-only view, which gets null
    for every value that is not a finite number, says "not a finite number"."""
    from src.api.services.record_states import CohortVersions, contest_states, key_text
    from src.evaluation.sidecar import read_ledger, record_key
    from tests.fixtures.report_store import store_bytes, version_of

    checkpoint = world["paths"]["model-40.pt"]
    ledger, _ = read_ledger(checkpoint.parent / "evaluations.json")
    (record,) = [r for r in ledger["records"] if r["checkpoint_digest"] == world["shas"]["model-40.pt"]]
    kept = next(world["registry"].glob(f"versions/*/reports/*-{world['shas']['model-40.pt'][:12]}-*/report.json"))
    report = json.loads(kept.read_bytes())
    report["authoritative_metrics"]["pooled_mrr"] = value
    store_bytes(world["registry"], json.dumps(report, sort_keys=True, allow_nan=True).encode(),
                version=version_of(kept), checkpoint_sha256=world["shas"]["model-40.pt"])

    operator = contest_states([record], CohortVersions.listed())[key_text(record_key(record))]
    assert operator["state"] == "contested"
    # model-40's runs record deterministic algorithms on: flagged for investigation.
    assert record["regime"]["deterministic_algorithms"] is True
    operator_html = mp.contest_html(operator, mode=record["scoring"]["mode"],
                                    deterministic=record["regime"]["deterministic_algorithms"])
    assert ("runs under these conditions differ — investigate" in operator_html
            and f"{shown} (run" in operator_html)
    assert "nan" not in operator_html.lower().replace("not a finite number", "")

    _served_read_only(world, monkeypatch, name="model-40.pt")
    data = mp.read_readonly()
    (view,) = data["answer"]["records"]
    assert view["contest"]["state"] == "contested"
    readonly_html = mp.contest_html(data["contest"][key_text(record_key(record))],
                                    mode=mp.record_mode(view),
                                    deterministic=view["deterministic_algorithms"])
    assert "runs under these conditions differ — investigate" in readonly_html
    assert "not a finite number (run" in readonly_html
    assert "runs differ #" in mp.readonly_parts(data)["records"]


def test_the_page_reads_no_ledger_and_scans_no_store_the_listing_already_read(world,
                                                                              monkeypatch):
    """M5-3 review ARCH-3: the shared listing read every ledger and scanned
    every store for its rows, and Models read them all again. The page now
    shows what the listing read, so the rows and the detail cannot disagree."""
    import src.evaluation.report_store as report_store
    import src.evaluation.sidecar as sidecar

    listing = ml.list_models("demo", operator_request(monkeypatch))
    assert listing["ledgers"] and listing["contest"]
    monkeypatch.setattr(sidecar, "read_ledger",
                        lambda *a, **k: pytest.fail("Models read a ledger again"))
    monkeypatch.setattr(report_store, "scan_report_store",
                        lambda *a, **k: pytest.fail("Models scanned a report store again"))
    data = mp.read_page(listing)
    assert data["ledgers"] == listing["ledgers"] and data["contest"] == listing["contest"]


# --------------------------------------------------------------------------- the shared Load
def test_newest_first_gives_the_load_no_ordering(world, monkeypatch):
    """Newest first: no column of the page's own; what each row is
    (`renderers.record_kind_text`) is said by the shared list in every row's
    Model cell, as under every other order; the Task filter keeps its rows (an
    unknown filter is All)."""
    _add_bundles(world)
    listing, data = _read(world, monkeypatch)
    ctl = mp.controls(data, mp.NEWEST, None)
    view = mp.shape(listing, data, ctl)
    assert view == {"task": "all", "note": mp.LOAD_NOTE}
    html = ml.list_html(listing, None, ml.PRESETS["research"], view=view)
    for key, kind in ((_key(world, "model-05.pt"), "Record not readable"),
                      (_key(world, "broken.pt"), "Record not read"),
                      (bundle_key("base"), "Base"), (bundle_key("gene"), "Gene")):
        row = html[html.index(f'data-key="{key}"'):]
        assert row.split("</tr>")[0].count(f'<div class="shep-sub shep-row-kind">{kind}</div>') \
            == 1, key
    ctl = mp.controls(data, mp.NEWEST, "disease")
    assert mp.shape(listing, data, ctl) == {"task": "disease", "note": mp.LOAD_NOTE}
    assert mp.controls(data, mp.NEWEST, "gatv2")["task"] == "all"



def test_a_record_of_another_or_no_mode_is_shown_as_recorded(world, monkeypatch):
    """A record or usage entry whose mode this version does not describe, or
    whose mode cannot be read, is named for what it is, by the one display for
    every mode (R2, K8-6e.2): its own label; a "Recorded metrics" column with
    every metric as recorded in place of the fixed columns and their "not
    described" cells; no claim of what its cases are; never another mode's
    words. An entry written before `mode` existed reads "Measurement mode not
    recorded", never Mode C, with its metrics as recorded one click away."""
    _listing, data = _read(world, monkeypatch)
    record = next(r for entry in data["ledgers"].values() for r in entry.get("records") or [])
    view = mp.record_view(record)
    assert view["mode"] == "disease"
    for mode, label in (("E", "Mode E (a measurement mode this version does not describe)"),
                        (None, "Measurement mode not recorded")):
        raw = mp.records_block_html([{**view, "mode": mode}], {})
        text = _text(raw)
        assert label in text
        assert "Mode C" not in text and "Disease" not in text
        assert re.findall(r"<th[^>]*>([^<]*)</th>", raw)[1:3] == ["Recorded metrics", "Coverage"]
        assert '<th scope="col" colspan="2">Recorded metrics</th>' in raw
        assert ">not described<" not in raw and "what they are is not described" in text
        for name, value in view["metrics"].items():          # every one, unrounded
            assert f"{name} {json.dumps(value)}" in _words(raw), name
        assert "The Trainer validation figure is not among these records' metrics." in text
    entry = {"kind": "test_run", "at": "2026-10-26T12:00:00Z", "version": "v", "run_id": "r",
             "metrics": {"untruncated_mrr": 0.5}, "n_ranked": 3, "device": "cpu",
             "deterministic": True, "record_requested": False}
    row = mp.usage_entry_html(entry)
    assert "Measurement mode not recorded, 3 cases" in _text(row) and "Mode C" not in _text(row)
    assert ('<details class="shep-mp-fold"><summary>Recorded metrics (1)</summary>' in row
            and "<details class=\"shep-mp-fold\" open" not in row)
    assert "untruncated_mrr 0.5" in _words(row)
    for bad in (None, "", 7):
        shown = _text(mp.usage_entry_html({**entry, "mode": bad}))
        assert "Measurement mode not recorded" in shown and "Mode C" not in shown


def _kept_c_and_d_records(*more):
    """A kept Mode C record and a kept Mode D record of one checkpoint and
    version, as a ledger holds them (Batch 3 registered both; neither mode runs
    now), then `more` reports, registered in that order."""
    from src.evaluation.sidecar import append_record, build_record, empty_ledger
    from tests.unit.test_evaluation_sidecar import mode_c_report, mode_d_report

    ledger = empty_ledger()
    reports = ((mode_c_report(), "c" * 64), (mode_d_report(), "d" * 64), *more)
    for n, (report, digest) in enumerate(reports, start=1):
        record = build_record(report, digest)
        run_id = f"20261027T120000Z-{record['checkpoint_digest'][:12]}-0000000{n}"
        ledger = append_record(ledger, record, run_id=run_id,
                               registered_at="2026-10-27T12:00:00Z")[1]
    return ledger["records"]


def _table(raw, mode_label_text):
    """The records table named for `mode_label_text` (its aria-label)."""
    start = raw.index(f'aria-label="{e("Recorded results, " + mode_label_text)}"')
    return raw[start:raw.index("</table>", start)]


def test_kept_c_and_d_records_are_shown_as_recorded_by_the_one_display():
    """Kept Mode C and Mode D records (no rule refuses or hides them, policy rev
    15) are shown by the one display for every mode, as recorded (R2): each
    mode's own generic label above its table, the table named for it; one
    "Recorded metrics" column listing every metric by its recorded name and its
    value unrounded; each record's score semantics exactly as recorded, two D
    records with different semantics each with its own; the tables in the order
    first shown, never C first; no C/D label, note, case-set words, derived
    served scoring or mean-rank rounding. A usage entry of either mode reads
    its label, its count and its metrics as recorded."""
    records = _kept_c_and_d_records()
    views = [mp.record_view(record) for record in records]
    assert [view["mode"] for view in views] == ["C", "D"]
    raw = mp.records_block_html(views, {})
    block, words = _text(raw), _words(raw)
    c_label, d_label = mode_label("C"), mode_label("D")
    assert c_label == "Mode C (a measurement mode this version does not describe)"
    assert d_label == "Mode D (a measurement mode this version does not describe)"
    assert f"{c_label}." in block and f"{d_label}." in block
    c_table, d_table = _table(raw, c_label), _table(raw, d_label)
    for table in (c_table, d_table):
        assert '<th scope="col" colspan="2">Recorded metrics</th>' in table
        assert ">not described<" not in table and 'class="shep-mp-fold"' not in table
    c_main = c_table[c_table.index('class="shep-mp-main"'):c_table.index('class="shep-mp-more"')]
    for line in ("untruncated_mrr 0.5", "hits@1 0.25", "hits@10 0.75", "mean_rank 3.0"):
        assert line in _words(c_main), line
    d_main = d_table[d_table.index('class="shep-mp-main"'):d_table.index('class="shep-mp-more"')]
    for line in ("post_import_mrr 0.35", "post_import_hits@1 0.2", "coverage 0.7",
                 "reached_mrr 0.5", "reached_mean_rank 2.43", "reached_hits@10 0.9"):
        assert line in _words(d_main), line
    assert "Score semantics: raw cosine, no eta mixture and no shortest-path term" in words
    assert "Score semantics: served:gnn_only:eta=1.0:sp=none" in words
    for gone in ("approved scorer policy", "Served ranking", "import-eligible)", "reached only",
                 "η =", "served scoring", "MRR (untruncated)", "Mode C metrics", "Mode D metrics",
                 "October 2026", "Diagnosis page", "2.43 ", "reached by"):
        assert gone not in block, gone
    # first shown first: the D record first, then C; never C first by its name
    raw_dc = mp.records_block_html(views[::-1], {})
    assert raw_dc.index(e(d_label)) < raw_dc.index(e(c_label))
    assert raw.index(e(c_label)) < raw.index(e(d_label))
    # Two D records with different semantics: one table, each its own semantics.
    other = {**views[1], "score_semantics": "served:gnn_plus_shortest_path:eta=0.7:sp=abc"}
    two = mp.records_block_html([views[1], other], {})
    assert two.count('<table class="shep-tbl shep-mp-rec"') == 1
    assert "Score semantics: served:gnn_only:eta=1.0:sp=none" in _words(two)
    assert "Score semantics: served:gnn_plus_shortest_path:eta=0.7:sp=abc" in _words(two)
    assert "η" not in _text(two)

    for entry_mode, metrics, n in (("D", {"post_import_mrr": 0.35, "coverage": 0.7}, 56),
                                   ("C", {"untruncated_mrr": 0.5, "mean_rank": 3.0}, 80)):
        entry = {"kind": "test_run", "at": "2026-10-27T12:00:00Z", "version": "v",
                 "run_id": "r", "mode": entry_mode, "metrics": metrics, "n_ranked": n,
                 "device": "cuda", "deterministic": True, "record_requested": True}
        row = mp.usage_entry_html(entry)
        assert f"{mode_label(entry_mode)}, {n} cases" in _text(row)
        assert f"Recorded metrics ({len(metrics)})" in row
        for name, value in metrics.items():
            assert f"{name} {value}" in _words(row)
        assert "reached" not in _text(row) and "import-eligible" not in _text(row)


def test_a_records_native_build_is_one_line_of_its_detail():
    """research-comparison Stage 2b: each record's detail has one "Native
    build" line, in the run environment's one wording (the Test page shows
    its words a component per line): "not recorded" for the kept records, whose environment names no
    components, and each component's identity and route for one that does."""
    from src.utils.run_environment import PYG_NATIVE_KIND, component_facts, native_identity_text
    from tests.unit.test_evaluation_sidecar import environment_block, registered_report

    environment = environment_block()
    environment["facts"]["components"] = {"pyg_lib": component_facts(
        PYG_NATIVE_KIND, installed=True, available=True, version="0.8.0",
        native={"state": "recorded", "count": 1, "aggregate_sha256": "a" * 64},
        source={"state": "recorded", "route": "local build", "wheel_sha256": "b" * 64})}
    records = _kept_c_and_d_records((registered_report(environment=environment), "e" * 64))
    views = [mp.record_view(record) for record in records]

    raw = mp.records_block_html(views, {})

    words = _words(raw)
    assert raw.count("Native build:") == 3
    assert words.count("Native build: not recorded") == 2
    assert native_identity_text(views[2]["native_identity"]) == (
        "local build — pyg_lib 0.8.0 aaaaaaaaaaaa")
    assert "Native build: local build — pyg_lib 0.8.0 aaaaaaaaaaaa" in words
    assert "b" * 12 not in raw


def test_a_mode_never_seen_before_is_shown_as_recorded_with_no_list_of_modes():
    """No list of modes decides what is shown (the reviewer, 2026-11-07: the
    three tasks are not a read whitelist): a record whose mode no build has
    ever named, beside a disease record and kept C and D ones, is read by the
    shared readers and shown like any mode without words of its own, its
    label naming its recorded mode, every metric and its score semantics as
    recorded; nothing is dropped, hidden or put last for its mode's name."""
    from tests.unit.test_evaluation_sidecar import registered_report

    new = registered_report({"zz_new_metric@7": 0.123456789, "pooled_mrr": 0.5},
                            mode="zz_protocol_never_seen", score_semantics="zz: a new score")
    records = _kept_c_and_d_records((new, "e" * 64), (registered_report(), "f" * 64))
    views = [mp.record_view(record) for record in records]
    assert all(view is not None for view in views)
    assert [view["mode"] for view in views] == ["C", "D", "zz_protocol_never_seen", "disease"]
    views = views[::-1]                     # newest first, as the page shows them
    raw = mp.records_block_html(views, {})
    label = mode_label("zz_protocol_never_seen")
    assert label == "Mode zz_protocol_never_seen (a measurement mode this version does not describe)"
    order = [raw.index(e(text)) for text in (
        mode_label("disease"), label, mode_label("D"), mode_label("C"))]
    assert order == sorted(order), order
    table = _table(raw, label)
    assert "zz_new_metric@7 0.123456789" in _words(table) and "pooled_mrr 0.5" in _words(table)
    assert "Score semantics: zz: a new score" in _words(table)
    assert raw.count('<table class="shep-tbl shep-mp-rec"') == 4


def test_a_task_record_keeps_its_columns_and_shows_every_metric_one_click_away():
    """A record of a measurement this build describes keeps its two columns and
    its readable words (R2); every metric as recorded is one click away in a
    closed native disclosure, "Recorded metrics (N)", with the raw names and
    unrounded values; its score semantics as recorded on its detail line."""
    from tests.unit.test_evaluation_sidecar import DISEASE_METRICS, registered_report

    records = _kept_c_and_d_records((registered_report(), "f" * 64))
    view = mp.record_view(records[2])
    raw = mp.records_block_html([view], {})
    heads = re.findall(r'<th scope="col" class="shep-num">([^<]*)</th>', raw)
    assert heads == ["Disease pooled MRR", "Disease pooled top-10"]
    assert "Recorded metrics</th>" not in raw
    fold = re.search(r'<details class="shep-mp-fold"><summary>Recorded metrics \((\d+)\)'
                     r'</summary>(.*?)</details>', raw)
    assert fold and int(fold.group(1)) == len(DISEASE_METRICS)
    assert "macro_precision@3 0.16666666666666666" in _words(fold.group(2))
    assert "pooled_top10 0.75" in _words(fold.group(2))
    assert "Disease conservative pooled MRR 0.4500" in _text(raw)
    assert ("Score semantics: disease head: -||z_d - z_P||^2 on L2-normalised projected vectors"
            in _words(raw))


def test_the_read_only_view_shows_kept_c_and_d_records_and_a_task_record_the_same_way(
        tmp_path, monkeypatch):
    """The read-only view reads the API's route (no checkpoint opened) and
    draws its kept C and D records and its task record with the shared
    `records_block_html`, as the operator's detail draws the ledger: the same
    labels, columns, recorded metrics and score semantics."""
    import src.api.main as api_main
    from src.api.services.record_states import key_text
    from src.evaluation.sidecar import LEDGER_FILENAME, append_record, build_record, empty_ledger
    from src.utils.fingerprint import ReadIdentity
    from tests.unit.test_evaluation_sidecar import (
        mode_c_report,
        mode_d_report,
        registered_report,
    )

    weights = tmp_path / "runs" / "checkpoints" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"stand-in weights")
    sha = file_sha256(weights)
    ledger = empty_ledger()
    for n, report in enumerate((mode_c_report(), mode_d_report(), registered_report()), 1):
        report["manifest"]["artifact_digests"]["checkpoint"] = sha
        record = build_record(report, f"{n}" * 64)
        ledger = append_record(ledger, record, run_id=f"20261027T120000Z-{sha[:12]}-0000000{n}",
                               registered_at="2026-10-27T12:00:00Z")[1]
    (weights.parent / LEDGER_FILENAME).write_text(json.dumps(ledger))
    monkeypatch.setattr(api_main.app_state, "pipeline",
                        _ServedPipeline(ReadIdentity(weights.resolve(), sha)))
    answer = mp._route_records()
    assert [r["mode"] for r in answer["records"]] == ["C", "D", "disease"]
    contest = {}
    for record in answer["records"]:
        contest[key_text(mp.route_record_key(record, sha))] = {**record["contest"],
                                                              "problems": []}
    readonly = mp.readonly_parts({"answer": answer, "served_sha": sha, "status_at": "now",
                                  "contest": contest})["records"]
    views = sorted((mp.record_view(r) for r in ledger["records"]),
                   key=lambda v: -v["registration"]["sequence"])
    operator = mp.records_block_html(views, contest)
    assert readonly.replace(f'<p class="shep-note">{e(mp.NO_LOAD_HERE)}</p>', "") == operator
    for text in (mode_label("C"), mode_label("D"), mode_label("disease"),
                 "Score semantics: served:gnn_only:eta=1.0:sp=none", "untruncated_mrr 0.5",
                 "post_import_mrr 0.35", "Recorded metrics (", "Disease pooled MRR"):
        assert text in _words(readonly), text


def test_a_hostile_metric_name_and_score_semantics_are_literal_text_on_models():
    """A metric name and a score semantics holding markup are shown as literal
    text (escaped) in the records table, its disclosure and a usage row."""
    from tests.unit.test_evaluation_sidecar import registered_report

    hostile = "<img src=x onerror=alert(1)>"
    metrics = {hostile: 0.5, "pooled_mrr": 0.25}
    for mode in ("zz_new", "disease"):
        report = registered_report(metrics, mode=mode, score_semantics=hostile)
        view = mp.record_view(_kept_c_and_d_records((report, "f" * 64))[2])
        raw = mp.records_block_html([view], {})
        assert hostile not in raw and e(hostile) in raw, mode
        assert f"{hostile} 0.5" in _words(raw) and f"Score semantics: {hostile}" in _words(raw)
    row = mp.usage_entry_html({"kind": "test_run", "at": "2026-10-27T12:00:00Z", "version": "v",
                               "run_id": "r", "mode": hostile, "metrics": metrics,
                               "n_ranked": 3, "device": "cpu", "deterministic": True})
    assert hostile not in row and f"{hostile} 0.5" in _words(row)


def test_a_usage_row_lists_its_metrics_as_recorded_or_says_there_are_none():
    """Each `test_run` usage row: a described mode's first column figure "over
    N ranked cases", any other mode's label and count ("case count not
    recorded" when it has none), then how it ran on a line
    of its own (one " · " per line, skill §5), then its metrics as recorded
    behind a closed disclosure; "no metric recorded" when it records none, and
    a metrics value that is not a list of names "not readable as recorded"."""
    base = {"kind": "test_run", "at": "2026-10-27T12:00:00Z", "version": "v", "run_id": "r",
            "n_ranked": 8, "device": "cuda", "deterministic": True, "record_requested": True}
    disease = mp.usage_entry_html({**base, "mode": "disease",
                                   "metrics": {"pooled_mrr": 0.45, "pooled_top10": 0.75}})
    assert "Disease pooled MRR 0.4500 over 8 ranked cases" in _text(disease)
    assert "CUDA · deterministic; registration asked" in _text(disease)
    assert "Recorded metrics (2)" in disease and "pooled_top10 0.75" in _words(disease)
    for cell in re.findall(r"<div[^>]*>([^<]*)</div>", disease):
        assert cell.count(" · ") <= 1, cell
    empty = mp.usage_entry_html({**base, "mode": "disease", "metrics": {}})
    assert "no metric recorded" in _text(empty) and "Recorded metrics" not in empty
    absent = mp.usage_entry_html({**base, "mode": "disease"})
    assert "no metric recorded" in _text(absent)
    damaged = mp.usage_entry_html({**base, "mode": "disease", "metrics": [0.5]})
    assert "metrics not readable as recorded" in _text(damaged)
    assert "no metric recorded" not in _text(damaged)
    # An entry with no case count (a hand edit: the writer always records it)
    # says so in words, never Python's None.
    for n_ranked in ({}, {"n_ranked": None}, {"n_ranked": "8"}, {"n_ranked": True}):
        entry = {**base, **n_ranked}
        if not n_ranked:
            del entry["n_ranked"]
        kept = _text(mp.usage_entry_html({**entry, "mode": "D", "metrics": {"x": 1}}))
        assert ("Mode D (a measurement mode this version does not describe), case count not "
                "recorded") in kept and "None" not in kept
        described = _text(mp.usage_entry_html({**entry, "mode": "disease",
                                               "metrics": {"pooled_mrr": 0.45}}))
        assert "Disease pooled MRR 0.4500, case count not recorded" in described
        assert "None" not in described


# --------------------------------------------------------------------------- the detail
def test_the_detail_keeps_validation_and_mode_c_apart(world, monkeypatch):
    """The Trainer's validation figure in its own card, the recorded test
    results in theirs, never together (the world's records are disease task
    records since K8-6d); the records newest registration first, a contested
    one with what its runs say, and no role note (By test score retired)."""
    listing, data = _read(world, monkeypatch)
    detail = mp.detail_parts(listing, data, _key(world, "model-05.pt"), NEWEST_CTL)
    # The title names the row, focusable: narrow, choosing a row waits for this
    # title before it scrolls to the detail and moves the focus there.
    assert f'<h2 tabindex="-1" data-key="{e(_key(world, "model-05.pt"))}">' in detail["header"]
    assert "h2[data-key]" in mp.SCROLL_JS and "shep-mp-back" in detail["header"]
    assert "not in use" in detail["header"]
    assert "in use for diagnosis: model" in detail["header"]
    assert world["shas"]["model-12.pt"][:12] in detail["header"]
    top, records = detail["top"], detail["records"]
    assert world["shas"]["model-05.pt"] in top                  # the full fingerprint, Copy
    assert "Validation pooled MRR (Trainer) 0.2410" in top and "Disease pooled MRR" not in top
    assert ("Validation pooled MRR" not in records and "(Trainer)" not in records
            and "0.2410" not in records)
    assert e(mode_label("disease")) in records and "Disease pooled MRR" in records
    # newest registration first: no record is picked or put ahead of the others
    assert records.index("registered #3") < records.index("registered #2")
    assert records.index("registered #2") < records.index("runs differ #1")
    assert "D4 rule" not in records and "Δ " not in records
    # the acceptance record's group has runs that differ, with what they say:
    # its runs record deterministic algorithms off, so in neutral words
    assert ("runs differ #1" in records
            and "runs under these conditions differ: " in records
            and "investigate" not in records)
    assert "0.2051" in records and "0.2077" in records
    # no role says anything about ordering
    assert "ordering" not in records
    # The heading names no mode: each record's table carries its own (PLAN_B05 §6).
    assert detail["seg"][0] == ("Recorded results · 3 (1 where runs differ)", mp.RECORDS)
    # a role code is never broken inside its name in its narrow column: not
    # "research_selectio" / "n" (1440x900), nor "research_" / "selection" (M5-3);
    # it stays whole, in a titled ellipsis where it does not fit
    assert f"role {r.ident_html('research_selection')}" in records
    assert f"role {r.ident_html('institutional_acceptance')}" in records
    assert "<wbr>" not in records


def test_a_record_that_cannot_be_read_is_named_apart_from_the_results(world, monkeypatch):
    """K8-6e.1 (R3): model-05's ledger holds three records; one loses
    ``n_ranked`` by a hand edit. The segment counts the results and names the
    rest with one " · " ("Recorded results · 2 (1 not readable; 1 where
    runs differ)"); the detail lists it by index, registration, report and
    reason under the amber sentence, never as a result and never pointing to
    the server's log; the row reads "2 · 1 not readable; runs differ for 1".
    HEAD: "Recorded test results · 3 (1 where runs differ)", "the server's log
    says why", and the row counted 3."""
    ledger_path = world["paths"]["model-05.pt"].parent / "evaluations.json"
    data = json.loads(ledger_path.read_text())
    (index, bad), = [(i, rec) for i, rec in enumerate(data["records"])
                     if rec["checkpoint_digest"] == world["shas"]["model-05.pt"]
                     and rec["registration"]["sequence"] == 2]
    del bad["n_ranked"]
    ledger_path.write_text(json.dumps(data, indent=2))
    listing, page = _read(world, monkeypatch)
    detail = mp.detail_parts(listing, page, _key(world, "model-05.pt"), NEWEST_CTL)

    assert detail["seg"][0] == (
        "Recorded results · 2 (1 not readable; 1 where runs differ)", mp.RECORDS)
    assert detail["seg"][0][0].count(" · ") == 1
    said = _words(detail["records"])
    assert ("▲ 1 record for these weights cannot be read as a result. The ledger keeps it as "
            "written.") in said
    assert (f"Index Registration Report Reason {index} #2 "
            f"{bad['source_artifact_digest'][:12]} n_ranked: missing") in said
    assert "registered #2" not in detail["records"]
    assert "server's log" not in said
    assert r.recorded_tests_text(ml._recorded(listing, _key(world, "model-05.pt"))) == (
        "Recorded results: 2 · 1 not readable; runs differ for 1")


def test_the_pages_amber_sentences_are_amber_and_the_note_after_the_list_stands_apart():
    """K8-6e.1 review (skill §6, §2): the page's ``.shep-mp-p`` rule, scoped
    by the page's id, outranks the plain ``.shep-amber-t`` class, so a
    sentence marked amber (the ▲ sentence over the records that cannot be
    read, "the ledger was not read") was drawn in body grey (sampled
    rgb(67,74,88) at 1850 x 960). A rule of the same scope gives it the amber
    token. In the read-only view the closing note sat 6 px under the
    unreadable records' table and read as its caption; it now stands 16 px
    apart, and the records table's own R10 note is not moved."""
    css = mp.PAGE_CSS.replace("{{", "{").replace("}}", "}")
    base = f"{mp._PAGE} .shep-mp-p {{ margin: 0;".replace("{{", "{")
    amber = f"{mp._PAGE} .shep-mp-p.shep-amber-t {{ color: var(--shep-amber-800); }}".replace(
        "{{", "{").replace("}}", "}")
    assert base in css and amber in css and css.index(base) < css.index(amber)
    gap = f"{mp._PAGE} .shep-mp-unreadwrap + .shep-note {{ margin-top: 16px; }}".replace(
        "{{", "{").replace("}}", "}")
    assert gap in css and ".shep-mp-recwrap + .shep-note" not in css
    html = mp.unreadable_html([{"index": 0, "sequence": None, "source_artifact_digest": None,
                                "reason": "n_ranked: missing"}])
    assert '<p class="shep-mp-p shep-mp-unread-p"><span class="shep-st shep-st-warn"><span class="shep-st-shape" aria-hidden="true">▲</span><span class="shep-st-w">1 record' in html
    assert '<div class="shep-tbl-wrap shep-mp-recwrap shep-mp-unreadwrap">' in html


def test_unverifiable_is_a_word_and_never_a_count(world, monkeypatch):
    listing, data = _read(world, monkeypatch)
    records = mp.detail_parts(listing, data, _key(world, "model-08.pt"), NEWEST_CTL)["records"]
    assert "Training disease overlap: <b>unverifiable</b>" in records
    assert "there is no count" in records
    line = records[records.index("<b>R10, training</b>"):]
    line = line[:line.index('<span class="shep-mp-ln">')]
    assert " of 50 " not in line and " of 80 " not in line
    text = r.r10_label_text("disease_overlap", "D", {"state": "unverifiable", "k": None,
                                                     "n": 50, "cases": None}, "not_found")
    assert "0" not in text.replace("D:", "")


@pytest.mark.parametrize("version, state, evidence_words, reason", [
    (1, "not_found", "not found",
     "no file looked at holds the bytes this checkpoint records, so there is no count"),
    (2, "search_incomplete", "search incomplete",
     "a place that may hold the bytes this checkpoint records could not be read, so there "
     "is no count. Make it readable to the account that measures, and measure again; "
     "the report names the place"),
], ids=["version-1-not-found", "search-incomplete"])
def test_a_records_r10_reads_in_its_states_words_never_a_count(version, state,
                                                                evidence_words, reason):
    """D11 (guard; Decision 7) and the K8-6r.3 words. A ledger record whose R10
    block is definition version 1 with a `not_found` scope reads with its
    version-1 meaning; a version-2 search left incomplete says a place could
    not be read and what to do. Both *unverifiable*, never a number, a dash or
    "none", and no path."""
    from tests.unit.test_evaluation_sidecar import r10_block

    block = r10_block()
    block["definition_version"] = version
    block["training"]["evidence"] = {
        "state": state, "path": None,
        "problem": None if state == "not_found" else "... /srv/secret/x_samples.json ..."}

    html = mp.r10_html(block)

    # The one R10 rendering (`renderers.r10_rows`), its scope's label leading.
    line = html[:html.index('<span class="shep-mp-ln">', 1)]
    assert line.startswith('<span class="shep-mp-ln"><b>R10, training</b> <span '
                           f'class="shep-muted">inputs (evidence {evidence_words}):</span> ')
    for name in ("Training disease overlap", "Training phenotype-set overlap"):
        assert f"{name}: <b>unverifiable</b>: {reason}" in line
    text = re.sub(r"<[^>]+>", "", line).split("):", 1)[1]  # the labels, after "R10, ..."
    assert not re.search(r"\d", text) and "—" not in text and " none" not in text
    assert "/srv/secret" not in html


def test_the_usage_history_segment_lists_this_checkpoints_entries_newest_first(world, monkeypatch):
    """A test run and kept model loads (written before By test score retired,
    K8-6d): only the entries naming a listed checkpoint are kept, and the
    detail lists this checkpoint's, newest first."""
    from datetime import datetime, timezone

    from src.evaluation.report_store import record_test_run
    from tests.fixtures.report_store import kept_model_load, version_of

    version = version_of(world["det_report"])
    sha, other = world["shas"]["model-05.pt"], world["shas"]["model-08.pt"]
    record_test_run(version, run_id="20261017T152000Z-aaaaaaaaaaaa-11111111",
                    report_sha256="f" * 64, checkpoint_sha256=sha, metrics=_metrics(0.2310),
                    n_ranked=80, deterministic=True, device="cuda", record_requested=True,
                    mode="disease", at=datetime(2026, 10, 17, 15, 20, tzinfo=timezone.utc))
    kept_model_load(version, checkpoint_sha256=sha, metric="untruncated_mrr",
                    scoring_semantics_digest="d" * 64, loaded=False, why="refused",
                    at="2026-10-18T08:00:00Z")
    kept_model_load(version, checkpoint_sha256=other, metric="untruncated_mrr",
                    scoring_semantics_digest="d" * 64, loaded=True)
    kept_model_load(version, checkpoint_sha256="a" * 64, metric="untruncated_mrr",
                    scoring_semantics_digest="d" * 64, loaded=True)   # listed nowhere here
    listing, data = _read(world, monkeypatch)
    # Only the entries naming a listed checkpoint are kept for the page.
    assert set(data["usage"]["entries"]) == {sha, other}
    detail = mp.detail_parts(listing, data, _key(world, "model-05.pt"), NEWEST_CTL)
    usage = detail["usage"]
    assert detail["seg"][1] == ("Usage history · 2", mp.USAGE)
    assert usage.index("2026-10-18 08:00 UTC") < usage.index("2026-10-17 15:20 UTC")
    assert "not loaded: refused" in usage and "Test run" in usage
    assert "registration asked" in usage and MYGENE in usage
    assert other[:12] not in usage


def test_the_usage_history_lists_every_listed_versions_entries_for_a_listed_checkpoint(
        world, monkeypatch):
    """Guard (K8-6d): the usage history is read through the page's own cohort
    version list (`read_versions`, then `usage_entries`), never through the
    retired ordering. For a listed checkpoint it lists the `test_run` and
    `registered` entries of every listed version and a kept `model_load` entry
    (written before its writer retired), and the record detail still shows the
    checkpoint's records."""
    from src.evaluation.cohort_registry import resolve_version
    from src.evaluation.report_store import record_registration, record_test_run
    from tests.fixtures.report_store import kept_model_load

    sha = world["shas"]["model-05.pt"]
    labels = (MYGENE, INST, EXPLORE)
    for n, label in enumerate(labels):
        version = resolve_version(world["registry"], label)
        run_id = f"2026101{n}T120000Z-{sha[:12]}-1111111{n}"
        record_test_run(version, run_id=run_id, report_sha256=str(n) * 64,
                        checkpoint_sha256=sha, metrics={"pooled_mrr": 0.41, "pooled_top10": 0.7},
                        n_ranked=80, deterministic=True, device="cuda", record_requested=True,
                        mode="disease")
        record_registration(version, run_id=run_id, report_sha256=str(n) * 64,
                            checkpoint_sha256=sha, outcome="registered", sequence=n + 1)
    kept_model_load(resolve_version(world["registry"], MYGENE), checkpoint_sha256=sha,
                    metric="untruncated_mrr", scoring_semantics_digest="a" * 64,
                    loaded=False, why="refused")
    listing = ml.list_models("demo", operator_request(monkeypatch), family=True)
    data = mp.read_page(listing)
    assert {v["label"] for v in data["versions"]} == set(labels)
    assert sorted(data["usage"]["read_from"]) == sorted(labels)
    assert data["usage"]["problems"] == []
    assert sorted((e["version"], e["kind"]) for e in data["usage"]["entries"][sha]) == sorted(
        [*((label, kind) for label in labels for kind in ("test_run", "registered")),
         (MYGENE, "model_load")])
    detail = mp.detail_parts(listing, data, _key(world, "model-05.pt"),
                             {"task": mp.ALL_TASKS, "order": mp.NEWEST})
    assert detail["seg"] == [("Recorded results · 3 (1 where runs differ)", mp.RECORDS),
                             ("Usage history · 7", mp.USAGE), ("Results", mp.RESULTS)]
    usage = detail["usage"]
    assert usage.count(">Test run<") == 3 and usage.count(">Registered<") == 3
    assert usage.count(">Model load from an ordered list<") == 1
    assert "not loaded: refused" in usage and "scoring semantics aaaaaaaaaaaa" in usage
    for label in labels:
        assert f"<td>{label}</td>" in usage
    records = detail["records"]
    assert all(f"{word} #{n}" in records
               for word, n in (("runs differ", 1), ("registered", 2), ("registered", 3)))
    assert world["shas"]["model-05.pt"] in detail["top"]


def test_no_record_is_dropped_or_set_apart_for_its_family_dataset_or_cohort(world, monkeypatch):
    """Guard (K8-6d): the page finds a checkpoint's records and usage entries by
    its digest and by nothing else. Two disease models of two families, trained
    on two datasets, each with records on two cohort versions: under Newest
    first and By family each is listed, and every record and entry is in its
    own checkpoint's detail; none is left out or marked as not compared."""
    from src.evaluation.cohort_registry import resolve_version
    from src.evaluation.report_store import record_test_run
    from src.utils.checkpoint_io import write_checkpoint
    from tests.unit import test_task_first_selection as bundles

    shas = _add_bundles(world)
    # A second family, its base and disease child trained on another dataset.
    other_base, other_child = "20261005-000000_base_eeeeee", "20261006-000000_disease_ffffff"
    meta = {"checkpoint_kind": "weights", "epoch": 2,
            "training_input_digests": {"split_manifest": "f" * 64}}
    runs = world["workspace"] / "runs" / "gatv2"
    (runs / other_base / "checkpoints").mkdir(parents=True)
    (runs / other_child / "checkpoints").mkdir(parents=True)
    base_sha = write_checkpoint({**bundles._record_payload(logs={"val_lp_accuracy": 0.8},
                                                           family_id=other_base), **meta},
                                runs / other_base / "checkpoints" / "base.pt")
    child_path = runs / other_child / "checkpoints" / "last.pt"
    child_sha = write_checkpoint({**bundles._record_payload(
        "disease", logs={"val_pooled_mrr": 0.4}, parent={"sha256": base_sha, "run": other_base},
        family_id=other_base), **meta}, child_path)
    for path, age in ((runs / other_base / "checkpoints" / "base.pt", 25), (child_path, 24)):
        os.utime(path, (time.time() - age * 3600,) * 2)
    models = {bundle_key("disease"): (shas["disease"],
                                      world["paths"][f"{BUNDLES['disease'][0]}/last.pt"]),
              f"gatv2/{other_child}/last.pt": (child_sha, child_path)}
    blocks = {MYGENE: world["mygene_block"], INST: version_block(INST, ACCEPTANCE_ROLE)}
    for n, (sha, path) in enumerate(models.values()):
        for label, block in blocks.items():
            register(world["registry"], path, report_for(sha, block))
            record_test_run(resolve_version(world["registry"], label),
                            run_id=f"2026101{n}T120000Z-{sha[:12]}-22222222",
                            report_sha256="9" * 64, checkpoint_sha256=sha,
                            metrics={"pooled_mrr": 0.4}, n_ranked=80, deterministic=True,
                            device="cuda", record_requested=False, mode="disease")
    listing = ml.list_models("demo", operator_request(monkeypatch), family=True)
    data = mp.read_page(listing)
    items = {key: ml.find_item(listing, key) for key in models}
    # Two families, two training datasets.
    assert len({mp.family_row(listing, key)["family_id"] for key in models}) == 2
    assert len({item["dataset_fingerprint"] for item in items.values()}) == 2
    for order in (mp.NEWEST, mp.BY_FAMILY):
        ctl = {"task": mp.ALL_TASKS, "order": order}
        shown = ml.shown_items(listing, mp.shown_view(listing, ctl))
        for key, (_sha, _path) in models.items():
            assert items[key] in shown, (order, key)
            detail = mp.detail_parts(listing, data, key, ctl)
            assert detail["seg"] == [("Recorded results · 2", mp.RECORDS),
                                     ("Usage history · 2", mp.USAGE),
                                     ("Results", mp.RESULTS)], (order, key)
            for label in blocks:
                assert detail["records"].count(f'<div class="shep-w600">{label}</div>') == 1
                assert detail["usage"].count(f"<td>{label}</td>") == 1
            said = _text(detail["records"] + detail["usage"]).lower()
            for word in ("not compared", "not comparable", "not ordered", "not tested",
                         "not shown", "different scoring"):
                assert word not in said, (order, key, word)


def test_a_kept_model_load_entry_is_shown_with_its_recorded_fields():
    """A `model_load` entry written before By test score retired (K8-6d) is
    shown with the fields it records: its metric's name as recorded and its
    scoring semantics, never a mode the entry does not record."""
    entry = {"kind": "model_load", "at": "2026-10-18T08:00:00Z", "version": MYGENE,
             "checkpoint_sha256": "c" * 64, "metric": "untruncated_mrr",
             "scoring_semantics_digest": "a" * 64, "loaded": True, "why": None}
    shown = _text(mp.usage_entry_html(entry))
    assert "ordered by untruncated_mrr · scoring semantics aaaaaaaaaaaa" in shown
    assert "Model load from an ordered list" in shown and "Mode C" not in shown
    unsaid = _text(mp.usage_entry_html({**entry, "metric": None, "loaded": False}))
    assert "ordered by a metric not recorded" in unsaid
    assert "not loaded: no reason recorded" in unsaid


def test_the_retired_ordering_roles_setting_changes_nothing_on_models(world, monkeypatch):
    """A deployment that still sets SHEPHERD_ORDERING_ROLES is simply not read
    (K8-6d): the page's versions, its view under each order and a record's
    detail are the same with and without it, and say nothing of ordering."""
    _add_bundles(world)

    def page():
        listing, data = _read(world, monkeypatch)
        shown = [data["versions"]]
        for order in (mp.NEWEST, mp.BY_FAMILY):
            ctl = mp.controls(data, order, None)
            view = mp.shape(listing, data, ctl)
            detail = mp.detail_parts(listing, data, _key(world, "model-05.pt"), ctl)
            shown.append((ctl, view, mp.options_html(data, view.get("refused")),
                          {k: detail[k] for k in ("top", "records", "usage", "seg")}))
        return shown

    without = page()
    monkeypatch.setenv("SHEPHERD_ORDERING_ROLES", f"{RESEARCH_ROLE},{ACCEPTANCE_ROLE}")
    assert page() == without
    assert all(set(v) == {"label", "sha256", "role", "problem"} for v in without[0])
    said = json.dumps(without)
    assert "ordering" not in said.lower() and "ORDERING_ROLES" not in said


def test_with_no_cohort_version_read_the_usage_history_says_why():
    """The registry's reason for listing no version (once said under the
    retired order options) is said where the versions are read: the usage
    history."""
    empty = {"entries": {}, "problems": [], "read_from": []}
    html, count = mp.usage_html({"sha256": "c" * 64}, {
        "usage": empty, "problem": "SHEPHERD_REGISTRY is not set <b>"})
    assert count == 0
    assert "no cohort version was read (SHEPHERD_REGISTRY is not set &lt;b&gt;)." in html
    assert "no cohort version was read." in mp.usage_html({"sha256": "c" * 64},
                                                          {"usage": empty})[0]


def test_a_long_usage_history_shows_the_latest_entries_and_says_how_many(world, monkeypatch):
    """Long use: the usage history only grows; the detail draws the latest
    `USAGE_SHOWN` entries of the checkpoint and says how many there are."""
    sha = world["shas"]["model-05.pt"]
    entries = [{"kind": "model_load", "checkpoint_sha256": sha, "version": MYGENE,
                "at": f"2026-10-18T{n // 60:02d}:{n % 60:02d}:00Z", "metric": "untruncated_mrr",
                "scoring_semantics_digest": "d" * 64, "loaded": True}
               for n in range(mp.USAGE_SHOWN, -1, -1)]          # newest first, as read
    usage = {"entries": {sha: entries}, "problems": [], "read_from": [MYGENE]}
    html, count = mp.usage_html({"sha256": sha}, {"usage": usage})
    assert count == mp.USAGE_SHOWN + 1 == 201
    assert html.count('<tr><td class="shep-nowrap">') == mp.USAGE_SHOWN
    assert f"The latest {mp.USAGE_SHOWN} of 201 entries are shown." in html
    assert "2026-10-18 03:20 UTC" in html and "2026-10-18 00:00 UTC" not in html  # the oldest
    html, count = mp.usage_html({"sha256": sha}, {"usage": {**usage, "entries": {
        sha: entries[:mp.USAGE_SHOWN]}}})
    assert count == mp.USAGE_SHOWN and "The latest" not in html


def test_the_ledgers_states_are_said_and_name_no_directory(world, monkeypatch):
    ledger = world["paths"]["model-12.pt"].parent / "evaluations.json"
    ledger.write_text("{not json")
    listing, data = _read(world, monkeypatch)
    ctl = NEWEST_CTL
    key = _key(world, "model-12.pt")
    assert data["ledgers"][key]["state"] == "not_read"
    detail = mp.detail_parts(listing, data, key, ctl)
    assert "was not read, so none of its records are shown" in detail["records"]
    assert str(world["root"]) not in detail["records"] + detail["top"]
    none = mp.detail_parts(listing, data, _key(world, "model-30.pt"), ctl)
    assert "No ledger" in none["records"] and "never evaluated" in none["records"]


def test_markup_in_a_run_name_is_escaped(world, monkeypatch):
    from src.utils.checkpoint_io import write_checkpoint

    # One run named by its start time, one not (its whole name is then the title).
    for name in ("20261019-090000_run_<i>x", "<i>y_run"):
        run = world["workspace"] / "runs" / "gatv2" / name / "checkpoints"
        run.mkdir(parents=True)
        write_checkpoint({"checkpoint_kind": "weights", "epoch": 2, "logs": {}}, run / "best.pt")
    listing, data = _read(world, monkeypatch)
    ctl = mp.controls(data, mp.NEWEST, None)
    for mark in ("x", "y"):
        key = next(k for k in (ml.item_key(i) for i in listing["items"]) if f"<i>{mark}" in k)
        detail = mp.detail_parts(listing, data, key, ctl)
        html = detail["header"] + detail["top"]
        assert f"<i>{mark}" not in html and f"&lt;i&gt;{mark}" in html
    assert "&lt;i&gt;y_run" in detail["header"]               # the title names it, escaped


# --------------------------------------------------------------------------- task and family
EVIL = "<img src=x onerror=alert(1)>"


def test_the_task_and_order_options_are_short_unique_and_plain():
    """Skill §8: options 40 characters or fewer, unique, the distinguishing
    token first, at most one separator; the task words are the list's own; the
    order is Newest first or By family only (By test score retired, K8-6d)."""
    for choices in (mp.TASK_CHOICES, mp.ORDER_CHOICES):
        labels = [label for label, _value in choices]
        values = [value for _label, value in choices]
        assert len(set(labels)) == len(labels) and len(set(values)) == len(values)
        assert all(len(label) <= 40 and label.count(" · ") <= 1 for label in labels)
    assert mp.TASK_CHOICES == [("All", "all"), ("Base", "base"), ("Disease", "disease"),
                               ("Gene", "gene"), ("Patients-like-me", "patients_like_me")]
    assert {value for _label, value in mp.TASK_CHOICES} <= set(ml.TASK_FILTERS)
    assert mp.ORDER_CHOICES == [("Newest first", mp.NEWEST), ("By family", mp.BY_FAMILY)]
    title, text = mp.PAGE_HELP["models-ordering"]
    assert title == "Order" and text.split("\n")[0].startswith("How the list is ordered")
    # Skill §7's budget: a lead of about 90 characters, about 60-80 words in all.
    words = [w for w in text.split() if w != "•"]
    assert len(text.split("\n")[0]) <= 90 and len(words) <= 80, len(words)
    assert "test score" not in text and "models-regime" not in mp.PAGE_HELP


def test_by_family_draws_each_family_parent_first_with_its_lineage(world, monkeypatch):
    """By family: one group per family,
    the base first and its children under it, indented, each with its kind,
    operation and exact parent; a child whose parent is not available says so
    in amber with ▲ and its words, escaped; under All the records not readable
    (the files without a model spec among them) and the records not read
    follow as two groups; a task's view keeps its base as a context row."""
    shas = _add_bundles(world, orphan_parent_run=EVIL)
    listing, data = _read(world, monkeypatch)
    ctl = mp.controls(data, mp.BY_FAMILY, None)
    assert ctl["order"] == mp.BY_FAMILY
    view = mp.shape(listing, data, ctl)
    assert view["column"] == mp.LINEAGE_COLUMN
    titles = [g["title"] for g in view["groups"]]
    # A file whose record was never read is under its own word, "Record not
    # read" (`record_kind_text`), never under "Record not readable"; the files
    # without a model spec, refused by the record reader, under the word each
    # of their rows leads with (K6 closing review: no page word of its own).
    refused_word = r.record_kind_text({"record": "refused"})
    assert titles == ["Family from run 2026-10-01 00:00 · 3 files",
                      "Family from run 2026-10-04 00:00 · 1 file",
                      refused_word, "Record not read"]
    assert not hasattr(mp, "GROUP_HISTORY")
    assert {r.record_kind_text(ml.find_item(listing, key))
            for key in view["groups"][2]["keys"]} == {refused_word}
    assert view["groups"][0]["keys"] == [bundle_key("base"), bundle_key("disease"),
                                         bundle_key("gene")]
    assert view["groups"][3]["keys"] == [_key(world, "broken.pt")]
    assert len(view["groups"][2]["keys"]) == 7                       # every ShepherdGNN file
    cells = view["cells"]
    child = cells[bundle_key("disease")]
    # The kind is the row's own (the shared list's Model cell), never said
    # again in the Lineage cell; a file with no lineage has no value there.
    assert child.startswith('<div class="shep-mp-lin shep-mp-d1"><div class="shep-sub">Finetune')
    assert f"Finetune of <code class=\"shep-fpv\">{shas['base'][:12]}</code>" in child
    assert 'class="shep-mp-lin shep-mp-d0"' in cells[bundle_key("base")]
    assert "New run" in cells[bundle_key("base")]
    orphan = cells["gatv2/20261004-000000_disease_dddddd/last.pt"]
    assert ('<span class="shep-st shep-st-warn"><span class="shep-st-shape" aria-hidden="true">▲</span><span class="shep-st-w">parent unavailable: '
            in orphan)
    assert EVIL not in orphan and html_lib.escape(EVIL) in orphan
    assert cells[_key(world, "model-05.pt")] == (
        '<div class="shep-mp-lin shep-mp-d0"><span class="shep-muted">—</span></div>')
    # The shared list draws them: groups in order, the Lineage column.
    html = ml.list_html(listing, None, ml.PRESETS["research"], view=view)
    assert ">Lineage</th>" in html and EVIL not in html
    assert html.index("Family from run 2026-10-01") < html.index(f">{refused_word}<")
    # A record this build refused and a file never read: two groups, each
    # titled with its own word (skill §6: missing states are never merged).
    broken = ml.find_item(listing, _key(world, "broken.pt"))
    # Another file: its own identity (keyed by its names), never the copy's.
    refused = {**broken, "filename": "refused.pt", "key": None, "record": "refused",
               "record_problem": "an unknown field"}
    row = {**mp.family_row(listing, _key(world, "broken.pt")), "key": ml.item_key(refused)}
    both = {**listing, "items": [*listing["items"], refused],
            "family": {**listing["family"], "all": [*listing["family"]["all"], row]}}
    groups = {g["title"]: g["keys"] for g in mp.family_view(both, mp.ALL_TASKS)["groups"]}
    assert groups["Record not read"] == [_key(world, "broken.pt")]
    assert groups["Record not readable"][-1] == ml.item_key(refused)
    assert _key(world, "broken.pt") not in groups["Record not readable"]
    # A task's view: its base as a context row, grey, ○ and its words.
    ctl = mp.controls(data, mp.BY_FAMILY, "gene")
    view = mp.shape(listing, data, ctl)
    assert [g["keys"] for g in view["groups"]] == [[bundle_key("base"), bundle_key("gene")]]
    assert ('<div class="shep-sub">' + mp.r.status_mark("neutral", "Context: parent of a shown "
                                                        "file") + '</div>'
            ) in view["cells"][bundle_key("base")]
    assert mp.shown_view(listing, ctl) == {"keys": [bundle_key("base"), bundle_key("gene")]}
    # No filter sets the files without a model spec apart (an unknown filter
    # is All); the marks are the shared status marks, in their status tokens.
    assert mp.controls(data, mp.BY_FAMILY, "history")["task"] == mp.ALL_TASKS
    assert "shep-mp-warn" not in mp.PAGE_CSS and "shep-mp-ctx" not in mp.PAGE_CSS
    # Without a family view (a listing read without one): newest first, said.
    bare = {k: v for k, v in listing.items() if k != "family"}
    view = mp.shape(bare, data, mp.controls(data, mp.BY_FAMILY, None))
    assert view["refused"] == mp.FAMILY_NOT_READ and "groups" not in view
    assert "Not ordered, so the list stays newest first: the family view was not read" in (
        mp.options_html(data, view["refused"]))


def test_the_order_switch_reaches_by_family_with_no_version_offering_ordering(
        world, monkeypatch):
    """The Order switch is the one control that reaches By family, and it is
    built off: the page turns it on for every operator's list (no cohort
    version or setting is asked), and leaves it off for a read-only request."""
    demo, _ctx = _page()
    opened = _fn(demo, "_open")
    order = next(b for b in demo.blocks.values() if getattr(b, "elem_id", None) == "shep-mp-order")
    assert order.interactive is False

    def open_as(request):
        out = opened.fn("demo", False, None, None, None, mp.BY_FAMILY, mp.ALL_TASKS,
                        mp.RECORDS, "idle", request)
        by = dict(zip(map(id, opened.outputs), out, strict=True))
        return next(v for v in out if isinstance(v, dict) and "listing_key" in v), by[id(order)]

    data, switch = open_as(operator_request(monkeypatch))
    assert data["readonly"] is False
    assert (switch["value"], switch["interactive"]) == (mp.BY_FAMILY, True)
    _served_read_only(world, monkeypatch)
    data, switch = open_as(gradio_request())
    assert data["readonly"] is True
    assert (switch["value"], switch["interactive"]) == (mp.NEWEST, False)


def test_the_detail_says_what_the_model_is_and_its_relation_to_the_paper(world, monkeypatch):
    """The detail's identity (README §5): kind, operation, the exact parent
    and where it is, family, encoder and axis 2 (the fields that differ, or
    "matches the reference recipe" with what that recipe is); Model Facts'
    Task and Configuration rows are not repeated; a file without a model
    spec says its kind only; Load is refused for a base with its reason."""
    shas = _add_bundles(world)
    listing, data = _read(world, monkeypatch)
    ctl = mp.controls(data, mp.NEWEST, None)
    top = mp.detail_parts(listing, data, bundle_key("disease"), ctl)["top"]
    ident = top[top.index('<div class="shep-kv shep-mp-ident">'):]
    ident = ident[:ident.index("</div></div>") + 12]
    keys = re.findall(r'<span class="shep-k">([^<]*)</span>', ident)
    assert keys == ["Kind", "Operation", "Parent", "Family", "Encoder", "Relation to the paper"]
    text = _text(ident)
    assert "Disease" in text and "Finetune" in text and "GATv2" in text
    assert f"{shas['base'][:12]} of run 2026-10-01 00:00 · listed here" in text
    assert "Family" in text and "run 2026-10-01 00:00" in text
    assert "differs in: " in text and "encoder.hidden" in text   # a shrunk test spec
    assert "Configuration" not in top and ">Task<" not in top
    specless = mp.detail_parts(listing, data, _key(world, "model-05.pt"), ctl)["top"]
    ident = specless[specless.index('<div class="shep-kv shep-mp-ident">'):]
    ident = ident[:ident.index("</span></div>") + 13]
    assert re.findall(r'<span class="shep-k">([^<]*)</span>', ident) == ["Kind"]
    assert "Record not readable" in ident
    # Axis 2 for a reference spec, and for a parent elsewhere or unavailable.
    item = {"record": "model", "learning_stage": "finetune", "task": "gene",
            "operation": "finetune", "family_id": BASE_RUN, "encoder": "gatv2",
            "differences": (), "parent_sha256": "a" * 64, "parent_run": EVIL}
    html = mp.identity_html(item, {"parent_state": "elsewhere", "parent_workspace": EVIL})
    assert "matches the reference recipe" in html and mp.REFERENCE_ANCHOR in _text(html)
    assert EVIL not in html and "in workspace <b>&lt;img" in html
    html = mp.identity_html(item, {"parent_state": "unavailable", "parent_problem": EVIL})
    assert '▲</span><span class="shep-st-w">parent unavailable: &lt;img' in html
    assert EVIL not in html
    assert "none: a new run" in mp.identity_html({**item, "parent_sha256": None}, None)
    # The frozen-backbone control is a task model with a frozen graph
    # backbone (ops-wiring round, decision F1), from its record's transfer.
    frozen = {**item, "task": "disease", "weight_transfer": {
        "trainable": ["patient_encoder", "task_head"],
        "parts": {"input": "frozen", "encoder": "frozen", "lp_head": "frozen",
                  "patient_encoder": "trainable", "task_head": "trainable"}}}
    assert ('<span class="shep-k">Kind</span><span class="shep-v">Disease · frozen graph '
            'backbone</span>') in mp.identity_html(frozen, None)
    assert '<span class="shep-v">Gene</span>' in mp.identity_html(item, None)
    # A branch says what it changed from its parent, or that it changed
    # nothing (decision B4): every value in the details, the names in By
    # family's Lineage; text from a record is escaped.
    branch = {**item, "operation": "branch", "parent_changes": {}}
    html = mp.identity_html(branch, None)
    keys = re.findall(r'<span class="shep-k">([^<]*)</span>', html)
    assert keys == ["Kind", "Operation", "Parent", "Settings changed", "Family", "Encoder",
                    "Relation to the paper"]
    assert mp.SAME_SETTINGS in _text(html)
    row = {"depth": 1, "parent_state": "listed"}
    assert mp.SAME_SETTINGS in _text(mp.lineage_cell_html(branch, row))
    changed = {**branch, "parent_changes": {"seed": {"parent": 1, "new": 5},
                                            "scheduler_type": {"parent": "cosine", "new": "none"},
                                            "early_stopping_patience": {"parent": 10,
                                                                        "new": None},
                                            EVIL: {"parent": EVIL, "new": 2}}}
    html = mp.identity_html(changed, None)
    # Each setting by its words, each value in the one value-words rule,
    # never Python's repr: no quotes, no bare None (B0c).
    assert "Seed 1 → 5" in _text(html) and EVIL not in html
    assert "Scheduler cosine → none" in _text(html)
    assert "Early stopping patience 10 → off" in _text(html)
    assert "'" not in _text(html) and "None" not in _text(html)
    lineage = mp.lineage_cell_html(changed, row)
    assert "Changed: " in _text(lineage) and "Seed" in _text(lineage) and EVIL not in lineage
    assert "Settings changed" not in mp.identity_html(item, None)
    assert "Changed" not in mp.lineage_cell_html(item, row)
    # Load is off for the base, with its reason (the shared list's).
    assert ml.load_refusal(ml.find_item(listing, bundle_key("base"))) == ml.BASE_REFUSAL


def test_a_task_records_coverage_reads_the_same_on_models_and_on_test(world, monkeypatch):
    """K6 closing review (P2): for one registered task record of a cohort with
    import exclusions, Models said "8 of 13 source cases ranked by the model"
    and Test "8 of 10 import-eligible cases ranked by the model · 13 source
    cases". A task measurement ranks the version's import-eligible samples
    (`counts.ranked_eligible`), so both pages say n of those, the source cases
    and the import exclusions on the line under it."""
    from src.webui.components import test_page as tp

    _listing, data = _read(world, monkeypatch)
    raw = next(r for entry in data["ledgers"].values() for r in entry.get("records") or [])
    counts = {**raw["cohort"]["counts"], "source_cases": 13, "ranked_eligible": 10,
              "n_excluded": 3, "excluded": {"ground_truth_absent": 3}}
    record = {**raw, "n_ranked": 8, "scoring": {**raw["scoring"], "mode": "disease"},
              "cohort": {**raw["cohort"], "counts": counts},
              "metrics": {**raw["metrics"], "coverage": 0.8}}
    view = mp.record_view(record)
    assert view is not None and view["mode"] == "disease"
    said = re.compile(r"(\d+) of (\d+) (?:[a-z-]+ )*cases ranked by the model")
    models = said.findall(_text(mp.coverage_html(view)))
    test = said.findall(_text(tp._record_html(record, {}, {})))
    assert models == test == [("8", "10")], (models, test)
    assert "13 source cases; 3 excluded at import: ground truth absent 3" in _text(
        mp.coverage_html(view))


def test_a_registered_task_record_shows_its_own_named_metrics(world, monkeypatch):
    """A disease, gene or patients-like-me record (K6 Test registers them) is
    shown in its own table under its task's metric names
    (`caveats.metric_text`): pooled MRR beside pooled top-10 or macro
    recall@10, the other figures after them in E1/E5/E2's order, its coverage
    in its own words; never "not described" under bare headers."""
    _listing, data = _read(world, monkeypatch)
    record = next(r for entry in data["ledgers"].values() for r in entry.get("records") or [])
    base = mp.record_view(record)
    metrics = {"pooled_mrr": 0.4123, "pooled_top10": 0.75, "macro_recall@10": 0.6,
               "macro_mrr": 0.4, "conservative_pooled_mrr": 0.38, "best_positive_mrr": 0.5,
               "coverage": 0.9, "pooled_top1": 0.25, "macro_precision@10": 0.06}
    for task, second, value in (("disease", "Disease pooled top-10", "0.7500"),
                                ("gene", "Gene macro recall@10", "0.6000"),
                                ("patients_like_me", "Patients-like-me macro recall@10",
                                 "0.6000")):
        view = {**base, "mode": task, "metrics": metrics}
        html = mp.records_block_html([view], {})
        heads = re.findall(r'<th scope="col" class="shep-num">([^<]*)</th>', html)
        name = {"disease": "Disease", "gene": "Gene", "patients_like_me": "Patients-like-me"}[task]
        assert heads == [f"{name} pooled MRR", second], heads
        assert "0.4123" in html and value in html and "not described" not in html
        text = _text(html)
        others = text[text.index(f"{name} coverage"):]
        order = [others.index(f"{name} {words}") for words in (
            "coverage", "conservative pooled MRR", "pooled top-1 ", "macro precision@10")]
        assert order == sorted(order), order
        assert "import-eligible cases ranked by the model" in text
    entry = {"kind": "test_run", "at": "2026-10-26T12:00:00Z", "version": "v", "run_id": "r",
             "metrics": {"pooled_mrr": 0.5}, "n_ranked": 3, "device": "cpu",
             "deterministic": True, "record_requested": False, "mode": "gene"}
    assert "Gene pooled MRR 0.5000 over 3 ranked cases" in _text(mp.usage_entry_html(entry))


# --------------------------------------------------------------------------- read-only
class _ServedPipeline:
    def __init__(self, read):
        self._read = read

    def served_checkpoint_read(self):
        return self._read


def test_read_only_shows_the_model_in_use_from_the_ledger_and_opens_no_checkpoint(
        world, monkeypatch):
    import src.api.main as api_main
    from src.utils.fingerprint import ReadIdentity
    from tests.fixtures.opens import count_opens

    checkpoint = world["paths"]["model-05.pt"]
    sha = world["shas"]["model-05.pt"]
    world["served"]["value"] = _status(world["workspace"], sha)
    monkeypatch.setattr(api_main.app_state, "pipeline",
                        _ServedPipeline(ReadIdentity(checkpoint.resolve(), sha)))
    monkeypatch.delenv("SHEPHERD_DEPLOYMENT_MODE", raising=False)
    listing = ml.list_models("demo", gradio_request())
    assert listing["refused"] is True
    with count_opens() as opens:
        data = mp.read_readonly()
    assert opens.opens(checkpoint) == 0
    assert opens.opens(checkpoint.parent / "evaluations.json") == 1
    parts = mp.detail_parts(listing, data, None, mp.controls(data, None, None))
    assert mp.READONLY_TITLE in parts["header"] and "opens no checkpoint" in parts["header"]
    assert sha in parts["top"] and parts["seg"] is None
    records = parts["records"]
    assert "registered #3" in records and "runs differ #1" in records
    # each record's determinism, from the environment class the route answers
    assert 'registered #3</span> <span class="shep-sub">deterministic on</span>' in records
    assert 'registered #2</span> <span class="shep-sub">deterministic off</span>' in records
    assert ("runs under these conditions differ: " in records and "investigate" not in records
            and mp.NO_LOAD_HERE in records)
    assert str(world["root"]) not in records and str(world["registry"]) not in records
    header = mp.left_header_html(listing, data, "deployment_mode")
    assert "Model list unavailable: read-only, deployment mode not set to M1." in header
    assert e(mp.D8_CONDITIONS) in header and "shep-mp-ro" in header
    assert "Nothing was done" not in header and "Refused" not in header      # said once (UI-6)
    assert "read-only, request not local." in mp.left_header_html(listing, data, "not_loopback")
    ctl = mp.controls(data, mp.BY_FAMILY, None)
    assert ctl["order"] == mp.NEWEST and mp.shape(listing, data, ctl) == {"note": ""}


def test_a_route_records_regime_is_read_from_its_environment_class():
    from src.evaluation.sidecar import ENVIRONMENT_CLASSES

    sha = "a" * 64
    det, nondet, cpu = (mp.view_from_route({"environment_class": c}, sha)
                        for c in ENVIRONMENT_CLASSES)
    assert (det["deterministic"], det["device"]) == (True, "cuda")
    assert (nondet["deterministic"], nondet["device"]) == (False, "cuda")
    assert (cpu["deterministic"], cpu["device"]) == (None, "cpu")
    other = mp.view_from_route({"environment_class": "something else"}, sha)
    assert (other["deterministic"], other["device"]) == (None, None)


def test_a_route_records_own_determinism_setting_is_read_for_a_cpu_run():
    """M5-3: read-only Models said "determinism not read" for CPU records that
    the operator's view showed as "deterministic on". The route carries the
    record's own setting (read from the ledger, no checkpoint opened), so
    both read the same fact."""
    from src.evaluation.sidecar import ENVIRONMENT_CLASSES

    sha = "a" * 64
    for flag in (True, False):
        cpu = mp.view_from_route({"environment_class": ENVIRONMENT_CLASSES[2],
                                  "deterministic_algorithms": flag}, sha)
        assert (cpu["deterministic"], cpu["device"]) == (flag, "cpu")
    assert mp.regime_text("cpu", True) != mp.regime_text("cpu", None)
    unsaid = mp.view_from_route({"environment_class": ENVIRONMENT_CLASSES[2],
                                 "deterministic_algorithms": None}, sha)
    assert unsaid["deterministic"] is None


def test_the_api_record_view_carries_the_records_own_determinism():
    from src.api.services.record_views import _recorded_determinism

    assert _recorded_determinism({"regime": {"deterministic_algorithms": True}}) is True
    assert _recorded_determinism({"regime": {"deterministic_algorithms": False}}) is False
    assert _recorded_determinism({"regime": {"deterministic_algorithms": "yes"}}) is None
    assert _recorded_determinism({"regime": None}) is None and _recorded_determinism({}) is None


def test_a_record_is_shown_only_with_what_an_answer_can_carry(world, monkeypatch):
    """Review fix RC-R1: the page read its record's device straight from the
    record, beside the view, and a GPU name holding a lone surrogate reached
    the view as text no answer can encode. The device now comes through the
    one reader of the record's settings (`recorded_settings`), which does not
    read such a value; a value the view shows is refused by the one reader,
    so the record is not shown and the listing says why; valid Unicode is
    shown exactly."""
    import json

    _listing, data = _read(world, monkeypatch)
    raw = next(r for entry in data["ledgers"].values() for r in entry.get("records") or [])
    odd_device = {**raw, "regime": {**raw["regime"], "device": "cuda:0" + chr(0xd800)}}
    view = mp.record_view(odd_device)
    assert view is not None and view["device"] is None
    json.dumps(view, ensure_ascii=False).encode("utf-8")
    facts = {**raw["environment"]["facts"], "gpu": {"name": "GB10 " + chr(0xd800),
                                                    "driver": "580"}}
    assert mp.record_view({**raw, "environment": {**raw["environment"], "facts": facts}}) is None
    facts = {**raw["environment"]["facts"], "gpu": {"name": "GB10 測試 🚀", "driver": "580"}}
    shown = mp.record_view({**raw, "environment": {**raw["environment"], "facts": facts}})
    assert shown["gpu"]["name"] == "GB10 測試 🚀"


def _served_read_only(world, monkeypatch, name="model-05.pt"):
    import src.api.main as api_main
    from src.utils.fingerprint import ReadIdentity

    checkpoint = world["paths"][name]
    world["served"]["value"] = _status(world["workspace"], world["shas"][name])
    monkeypatch.setattr(api_main.app_state, "pipeline", _ServedPipeline(
        ReadIdentity(checkpoint.resolve(), world["shas"][name])))
    monkeypatch.delenv("SHEPHERD_DEPLOYMENT_MODE", raising=False)


def test_read_only_names_no_server_path_when_a_report_store_cannot_be_read(world, monkeypatch):
    """A report of the record's condition group that cannot be compared makes
    the group's comparison incomplete, and the store's own words name the file
    by its absolute path: the operator's view keeps them, a read-only request
    never gets them (D8). A report that cannot be read at all is its own run's
    problem only (K8-6r.2, C1): it leaves the group's description as it was."""
    from src.evaluation.report_store import scan_report_store
    from src.evaluation.sidecar import record_key
    from tests.fixtures.report_store import store_bytes, version_of

    _served_read_only(world, monkeypatch)
    version = version_of(world["det_report"])
    key = record_key(dict(world["det"]))
    broken = store_bytes(world["registry"], b"{not json", version=version,
                         checkpoint_sha256=world["shas"]["model-05.pt"])
    scan = scan_report_store(version)
    assert scan.status(key).state == "uncontested"
    assert [problem for run, problem in scan.unplaced if run == broken.parent.name]
    report = json.loads(world["det_report"].read_bytes())
    del report["authoritative_metrics"]
    bad = store_bytes(world["registry"], json.dumps(report, sort_keys=True).encode(),
                      version=version, checkpoint_sha256=world["shas"]["model-05.pt"])
    status = scan_report_store(version).status(key)
    assert status.state == "cannot_be_checked" and str(bad) in " ".join(status.problems)
    records = mp.readonly_parts(mp.read_readonly())["records"]
    assert 'registered #3</span>' in records and "comparison incomplete" in records
    assert str(world["registry"].parent) not in records and "report.json" not in records
    assert "could not be read" not in records and "cannot be compared" not in records


def test_read_only_says_when_what_is_served_cannot_be_read(world, monkeypatch):
    _served_read_only(world, monkeypatch)
    monkeypatch.setattr(mp, "read_status", lambda: (None, "the status could not be read"))
    data = mp.read_readonly()
    assert data["served_sha"] is None and data["answer"]["state"] == "records"
    parts = mp.readonly_parts(data)
    assert ("What is served cannot be read now (the status could not be read), so no recorded "
            "result is shown.") in parts["top"]
    assert mp.RECORDS_MODEL_CHANGED not in parts["records"]
    assert "registered #" not in parts["records"] and mp.NO_LOAD_HERE in parts["records"]


def test_read_only_never_shows_another_models_records(world, monkeypatch):
    import src.api.main as api_main
    from src.utils.fingerprint import ReadIdentity

    checkpoint = world["paths"]["model-05.pt"]
    monkeypatch.setattr(api_main.app_state, "pipeline", _ServedPipeline(
        ReadIdentity(checkpoint.resolve(), world["shas"]["model-05.pt"])))
    world["served"]["value"] = _status(world["workspace"], world["shas"]["model-08.pt"])
    parts = mp.readonly_parts(mp.read_readonly())
    assert mp.RECORDS_MODEL_CHANGED in parts["records"] and "registered #" not in parts["records"]


# --------------------------------------------------------------------------- the wiring
def _page():
    from src.webui.shell import standalone_context

    with gr.Blocks() as demo:
        ctx = standalone_context(demo)
        mp.create_models_page(ctx)
    return demo, ctx


def _fn(demo, name):
    found = [fn for fn in demo.fns.values() if fn.name == name]
    assert len(found) == 1, (name, len(found))
    return found[0]


def test_the_page_is_read_when_shown_and_shapes_the_shared_list(world, monkeypatch):
    demo, ctx = _page()
    opened = _fn(demo, "_open")
    assert (ctx.page_tokens["models"]._id, "change") in opened.targets
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    lst = by_id["shep-models-research-rows"]
    assert lst in opened.outputs and by_id["shep-mp-records"] in opened.outputs
    states = [o for o in opened.outputs if isinstance(o, gr.State)]
    assert len(states) >= 4                    # listing, selected, page data, view, stale
    inputs = [getattr(i, "elem_id", None) for i in opened.inputs]
    assert {"shep-mp-order", "shep-mp-task"} <= set(inputs)
    assert not {"shep-mp-version", "shep-mp-metric", "shep-mp-scoring"} & set(inputs)
    stale = _sync_states(demo)["stale"]
    assert stale in opened.outputs
    out = opened.fn("demo", False, None, None, None, mp.BY_FAMILY, mp.ALL_TASKS, mp.RECORDS,
                    "idle", operator_request(monkeypatch))
    assert len(out) == len(opened.outputs)
    by_output = dict(zip(map(id, opened.outputs), out, strict=True))
    listing = out[0]
    data = next(v for v in out if isinstance(v, dict) and "listing_key" in v)
    assert data["listing_key"] == mp.listing_key(listing)
    views = [v for v in out if isinstance(v, dict) and "groups" in v]
    assert views and views[0]["column"] == mp.LINEAGE_COLUMN
    assert by_output[id(by_id["shep-mp-order"])]["interactive"] is True
    assert by_output[id(stale)] is False                   # read now: nothing is stale
    assert by_output[id(by_id["shep-mp-wsline"])] == mp.workspace_line_html(listing)
    # Shown while a Load runs: read now, and again once the Load has finished.
    out = opened.fn("demo", False, None, None, None, mp.NEWEST, None, mp.RECORDS, "loading",
                    operator_request(monkeypatch))
    assert dict(zip(map(id, opened.outputs), out, strict=True))[id(stale)] == gr.update()


def test_the_workspace_line_shows_the_dataset_identity_under_the_select(world, monkeypatch):
    """A long workspace name is cut in the select; the dataset fingerprint and
    "synthetic" are said again under it, by the shared renderer."""
    from src.webui.components.renderers import dataset_identity_html

    listing, _data = _read(world, monkeypatch)
    row = next(w for w in listing["workspaces"] if w["name"] == "demo")
    html = mp.workspace_line_html(listing)
    assert html == (f'<div class="shep-mp-wsline">'
                    f'{dataset_identity_html(row, name=False)}</div>')
    assert row["dataset_fingerprint"][:12] in html and "synthetic" in html
    assert "shep-ds-name" not in html                          # the select names it
    assert mp.workspace_line_html({**listing, "workspace": "gone"}) == ""
    assert mp.workspace_line_html({**listing, "refused": True}) == ""


def _sync_states(demo):
    """The one listener's States, by their place among its inputs."""
    sync = _fn(demo, "_sync")
    listing, data, stale, last, phase = sync.inputs[:5]
    return {"fn": sync, "listing": listing, "data": data, "stale": stale, "last": last,
            "phase": phase, "selected": sync.inputs[-2]}


def _sync(states, monkeypatch, listing, data, *, trigger, stale=False, last=None, phase="idle",
          order=mp.NEWEST, task=mp.ALL_TASKS, selected=None, seg=mp.RECORDS):
    sync = states["fn"]
    out = sync.fn(listing, data, stale, last, phase, order, task, selected, seg,
                  operator_request(monkeypatch), gr.EventData(trigger, None))
    assert len(out) == len(sync.outputs)
    by = dict(zip(map(id, sync.outputs), out, strict=True))
    return {name: by[id(states[name])] for name in ("stale", "last", "selected", "data")} | {
        "page": out[3:], "by": by}


def _session_copy(block):
    """A control as an event names it once the page has updated it: Gradio
    keeps a new instance in the session (`Blocks.postprocess_data`), built from
    the same constructor arguments, with another ``_id``."""
    copy = block.__class__(**{**block.constructor_args, "render": False})
    assert copy._id != block._id and copy.elem_id == block.elem_id
    return copy


def test_a_listing_already_read_is_not_read_again(world, monkeypatch):
    demo, _ctx = _page()
    states = _sync_states(demo)
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    detail = [by_id[i] for i in ("shep-mp-detail-hd", "shep-mp-detail-top", "shep-mp-seg",
                                 "shep-mp-records", "shep-mp-usage", "shep-rs")]
    listing, data = _read(world, monkeypatch)
    monkeypatch.setattr(mp, "read_page", lambda listing: pytest.fail("read again"))
    key = _key(world, "model-05.pt")
    # A selection: the detail only, nothing read; the selection is remembered.
    out = _sync(states, monkeypatch, listing, data, trigger=states["selected"], order=mp.NEWEST,
                selected=key)
    assert out["data"] == gr.update() and out["selected"] == gr.update()
    assert out["stale"] == gr.update() and out["last"] == {"workspace": "demo", "key": key}
    others = [o for o in states["fn"].outputs[3:] if o not in detail]
    assert all(out["by"][id(o)] == gr.update() for o in others)
    assert world["shas"]["model-05.pt"] in out["by"][id(by_id["shep-mp-detail-top"])]
    # A control (as the session holds it): the view drawn again from the data
    # held, nothing read.
    out = _sync(states, monkeypatch, listing, data,
                trigger=_session_copy(by_id["shep-mp-order"]), order=mp.BY_FAMILY)
    view = out["page"][1]
    assert out["data"] == gr.update() and view["column"] == mp.LINEAGE_COLUMN
    # A listing not read yet: read once.
    changed = {**listing, "served_sha": world["shas"]["model-05.pt"]}
    calls = []
    monkeypatch.setattr(mp, "read_page", lambda listing: calls.append(1) or data)
    _sync(states, monkeypatch, changed, data, trigger=states["listing"], order=mp.NEWEST)
    assert calls == [1]


def test_after_a_load_the_page_reads_again_and_selects_the_same_checkpoint(world, monkeypatch):
    """A Load that leaves the listing as it was (a refused reload here; the
    model in use reloaded is another): the page reads its own data again once
    the Load has finished, so the usage history shows what was written
    meanwhile (a test run from the command line, here), and selects the same
    checkpoint again (the shared refresh cleared the selection), so Close
    returns to its detail."""
    from src.api.routes.pipeline import PipelineReloadResponse, PipelineStatusResponse
    from src.evaluation.report_store import record_test_run
    from tests.fixtures.report_store import version_of

    demo, _ctx = _page()
    states = _sync_states(demo)
    shas = _add_bundles(world)
    listing, data = _read(world, monkeypatch)
    key, sha = bundle_key("disease"), shas["disease"]
    assert sha not in data["usage"]["entries"]
    plan, _html = ml.request_load(listing, key, operator_request(monkeypatch))
    served = world["shas"]["model-12.pt"]

    def refused(request):
        # Written while the Load ran: a test run of the same model, say.
        record_test_run(version_of(world["det_report"]),
                        run_id="20261019T100000Z-bbbbbbbbbbbb-33333333", report_sha256="8" * 64,
                        checkpoint_sha256=sha, metrics=_metrics(0.4), n_ranked=80,
                        deterministic=True, device="cuda", record_requested=False,
                        mode="disease")
        return PipelineReloadResponse(success=False, message="refused for this test",
                                      status=PipelineStatusResponse(
                                          **_status(world["workspace"], served)))

    step = ml.finish_load(plan, operator_request(monkeypatch), reload=refused)
    assert "Not reloaded" in step.confirm_html
    # the shared refresh, as the research preset reads it (with its family view)
    after = ml.list_models("demo", operator_request(monkeypatch), family=True)
    assert mp.listing_key(after) == data["listing_key"]                # nothing listed changed
    reads = []
    real = mp.read_page
    monkeypatch.setattr(mp, "read_page", lambda listing: reads.append(1) or real(listing))
    last = {"workspace": "demo", "key": key}
    # Still loading: nothing is read yet, and the page stays stale.
    out = _sync(states, monkeypatch, after, data, trigger=states["selected"], stale=True,
                last=last, phase="loading")
    assert reads == [] and out["stale"] == gr.update() and out["selected"] == gr.update()
    # The Load has finished (the refresh cleared the selection): read once, the
    # same checkpoint selected again, its usage history with the new entry.
    out = _sync(states, monkeypatch, after, data, trigger=states["selected"], stale=True,
                last=last, seg=mp.USAGE)
    assert reads == [1] and out["stale"] is False
    assert out["selected"] == key and out["last"] == last
    entries = out["data"]["usage"]["entries"][sha]
    assert [(e["kind"], e["run_id"]) for e in entries] == [
        ("test_run", "20261019T100000Z-bbbbbbbbbbbb-33333333")]
    page = dict(zip(map(id, states["fn"].outputs[3:]), out["page"], strict=True))
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    seg = page[id(by_id["shep-mp-seg"])]
    assert seg["choices"][1] == ("Usage history · 1", mp.USAGE) and seg["value"] == mp.USAGE
    usage = page[id(by_id["shep-mp-usage"])]
    assert "Test run" in usage["value"] and usage["visible"] is True
    view = out["page"][1]
    assert "redraw" in view and view["note"] == mp.LOAD_NOTE
    # The selection it set comes back as a change: the detail only, no read.
    again = _sync(states, monkeypatch, after, out["data"], trigger=states["selected"],
                  last=last, selected=key)
    assert reads == [1] and again["data"] == gr.update() and again["selected"] == gr.update()
    # A checkpoint the view no longer shows is not selected again.
    out = _sync(states, monkeypatch, after, data, trigger=states["selected"], stale=True,
                last=last, task="gene")
    assert reads == [1, 1] and out["selected"] == gr.update() and "redraw" not in out["page"][1]
    # By family shows it (among its family's rows): selected again.
    out = _sync(states, monkeypatch, after, data, trigger=states["selected"], stale=True,
                last=last, order=mp.BY_FAMILY)
    assert out["selected"] == key and out["page"][1]["column"] == mp.LINEAGE_COLUMN


def test_controls_selection_and_segments_are_wired(world, monkeypatch):
    demo, _ctx = _page()
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    states = _sync_states(demo)
    sync = states["fn"]
    # One listener answers a new listing, a selection and the controls, so its
    # answers for one session never overlap; the last trigger always runs.
    assert {(demo.blocks[t[0]], t[1]) for t in sync.targets} == {
        (states["listing"], "change"), (states["selected"], "change"),
        *((by_id[c], "input") for c in ("shep-mp-order", "shep-mp-task"))}
    assert not {"shep-mp-version", "shep-mp-metric", "shep-mp-scoring",
                "shep-mp-independence"} & set(by_id)
    assert sync.trigger_mode == "always_last"
    assert sync.outputs[:3] == [states["stale"], states["last"], states["selected"]]
    assert sync.outputs[-6:] == [by_id["shep-mp-detail-hd"], by_id["shep-mp-detail-top"],
                                 by_id["shep-mp-seg"], by_id["shep-mp-records"],
                                 by_id["shep-mp-usage"], by_id["shep-rs"]]
    assert not [fn for fn in demo.fns.values() if fn.name in ("_recompute", "_reorder", "_detail")]
    # "Reload with …" marks the page's data stale, at once, beside the shared Load.
    started = _fn(demo, "_load_started")
    assert started.targets == [(by_id["shep-models-research-reload"]._id, "click")]
    assert started.queue is False and started.outputs == [states["stale"]]
    assert started.fn() is True
    segment = _fn(demo, "_segment")
    assert segment.queue is False
    records, usage, results = segment.fn(mp.USAGE)
    assert records["visible"] is False and usage["visible"] is True
    assert results["visible"] is False
    # The third segment, Models › Results (research comparison Stage 8b): its
    # column only; its content is its own gated handlers'.
    records, usage, results = segment.fn(mp.RESULTS)
    assert (records["visible"], usage["visible"], results["visible"]) == (False, False, True)


def test_the_page_registers_its_help_and_every_i_it_draws(world, monkeypatch):
    from src.webui.components.help import HelpRegistry, registered_keys_in
    from src.webui.shell import standalone_context

    registry = HelpRegistry()
    with gr.Blocks() as demo:
        mp.create_models_page(standalone_context(demo, registry))
    _add_bundles(world)
    listing, data = _read(world, monkeypatch)
    ctl = mp.controls(data, mp.BY_FAMILY, None)
    view = mp.shape(listing, data, ctl)
    detail = mp.detail_parts(listing, data, _key(world, "model-05.pt"), ctl)
    rendered = ("".join(detail[k] for k in ("top", "records", "usage"))
                + mp.options_html(data, view.get("refused"))
                + "".join(view["cells"].values()) + mp.workspace_line_html(listing))
    assert 'data-help="models-contested"' in rendered         # the contested record's ⓘ
    from src.webui.components.renderers import RENDERER_HELP

    registry.topics(RENDERER_HELP)
    for key in registered_keys_in(rendered):
        assert key in registry, key
    assert set(mp.PAGE_HELP) <= {e.key for e in registry.entries()}


def test_the_app_registers_the_page_and_its_style_sheet():
    from src.webui import shell
    from src.webui.app import PAGE_CSS, PAGES, dashboard_head

    assert PAGES[shell.MODELS] is mp.create_models_page
    assert mp.PAGE_CSS in PAGE_CSS and mp.PAGE_CSS in dashboard_head()
    assert "#shep-page-models" in mp.PAGE_CSS


def test_before_the_first_read_the_detail_never_says_nothing_is_in_use():
    """M5-3 review UI-4: the page as built (and until its first listing came,
    seconds with many checkpoints) said "nothing is in use for diagnosis"
    while a model was served. Not read is said as not read."""
    built = mp.detail_parts({}, {}, None, {})["header"]
    assert "in use for diagnosis: not read yet" in built
    assert "nothing is in use" not in built
    assert "not read yet" in mp.detail_parts(ml.empty_listing(), {}, None, {})["header"]
    read = {**ml.empty_listing(), "status_read": True, "status_problem": None}
    assert "nothing is in use for diagnosis" in mp.detail_parts(read, {}, None, {})["header"]
    served = {**read, "served_sha": "a" * 64}
    assert "in use for diagnosis: model" in mp.detail_parts(served, {}, None, {})["header"]


def test_the_list_header_is_never_a_scroll_region_of_its_own():
    """M5-3 review UI-1: at 1536x760 the header (workspace, Order, the
    independence note, one two-line radio option per cohort version, every
    version without ordering) was capped and scrolled on its own above the
    table's scroll region, cutting the amber note mid-sentence. The header and
    the rows are now one column (``shep-model-main``) that scrolls as one
    region when they do not both fit. Since K8-6d the header holds no version
    select or ordering note at all."""
    demo, _ctx = _page()
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    assert "shep-mp-version" not in by_id and "shep-mp-independence" not in by_id
    config = demo.get_config_file()

    def find(node, wanted):
        if node.get("id") == wanted:
            return node
        for child in node.get("children", []):
            hit = find(child, wanted)
            if hit:
                return hit
        return None

    classes = {c["id"]: (c.get("props") or {}).get("elem_classes") or []
               for c in config["components"]}
    listing = find(config["layout"], by_id["shep-models-research"]._id)
    (main,) = [c for c in listing["children"] if "shep-model-main" in classes[c["id"]]]
    inside = [classes[c["id"]] for c in main["children"]]
    assert [c for c in inside if "shep-model-head" in c or "shep-model-body" in c] == [
        ["shep-model-head"], ["shep-model-body"]]
    css = mp.PAGE_CSS
    assert "fit-content(" not in css
    head_rules = [rule for rule in css.split("}") if ".shep-model-head {" in rule]
    assert head_rules and not [rule for rule in head_rules if "overflow" in rule]


def test_the_list_panels_title_stays_while_the_panel_scrolls():
    """M5-3: at 1440 x 900 the Models title scrolled away with the controls and
    the rows. The header row draws no box at a wide size (its items are the
    panel's grid items), so the title sticks to the panel's top, and the
    table's header sticks under it."""
    wide = mp.PAGE_CSS[mp.PAGE_CSS.index("@media (min-width: 1400px) and (min-height: 760px) {"):]
    assert "> .shep-model-main > .shep-model-head {{ display: contents !important; }}".replace(
        "{{", "{").replace("}}", "}") in wide
    assert "#shep-mp-hd { position: sticky; top: 0; z-index: 45;" in wide
    assert "#shep-models-research-rows .shep-tbl th { top: 48px; }" in wide


def test_the_table_fits_its_frame_beside_a_classic_scrollbar():
    """M5-3: at 1440 x 900 with a classic (non-overlay) 15 px scrollbar the
    table (602 px: its columns' longest words and 10 px a side) ran 13 px past
    its 589 px frame; with 7 px a side it is 572 px (measured in a browser with
    the scrollbar forced classic)."""
    wide = mp.PAGE_CSS[mp.PAGE_CSS.index("@media (min-width: 1400px) and (min-height: 760px) {"):]
    wide = wide[:wide.index("@media", 10)].replace(f"{mp._PAGE} ", "")
    assert ".shep-mtbl th, .shep-mtbl td { padding-left: 7px; padding-right: 7px; }" in wide


# --------------------------------------------------------------------------- K8-6r.2
def test_every_record_is_shown_with_its_run_and_its_groups_description(world, monkeypatch):
    """G7, G9 (K8-6r.2): every record the ledger holds for the checkpoint is
    shown, each with the run it was registered from, the group whose runs
    differ described in plain words ("runs differ", "runs under these
    conditions differ", neutral for these runs without deterministic
    algorithms), nothing hidden or ordered out by it.
    HEAD: no run was shown, and the words were "contested" / "runs disagree"."""
    from src.evaluation.sidecar import read_ledger

    listing, data = _read(world, monkeypatch)
    records = mp.detail_parts(listing, data, _key(world, "model-05.pt"), NEWEST_CTL)["records"]
    ledger, _ = read_ledger(world["paths"]["model-05.pt"].parent / "evaluations.json")
    mine = [r for r in ledger["records"]
            if r["checkpoint_digest"] == world["shas"]["model-05.pt"]]
    assert len(mine) == 3
    for record in mine:
        assert f"from run {record['run_id']}, report" in records
        assert f"#{record['registration']['sequence']}" in records
    assert "runs differ #1" in records
    assert "runs under these conditions differ: " in records and "investigate" not in records
    shown = re.sub(r"<[^>]+>", " ", records)      # the words a person reads
    assert "contested" not in shown and "disagree" not in shown


def test_runs_that_differ_are_flagged_only_in_a_deterministic_group(world, monkeypatch):
    """Policy rev 16 (a difference is described, never a problem): model-40's
    group records deterministic algorithms on, so its runs were expected to
    agree and the record is flagged: an amber "runs differ #1" chip with ▲, and
    "runs under these conditions differ — investigate" saying they were
    expected to agree. model-05's acceptance group records them off: the same
    difference reads "runs differ #1" in a grey chip and "runs under these
    conditions differ: …" in neutral words, with no ▲, amber or "investigate".
    A determinism that is not known is never assumed: it reads as neutral."""
    listing, data = _read(world, monkeypatch)
    flagged = mp.detail_parts(listing, data, _key(world, "model-40.pt"), NEWEST_CTL)["records"]
    assert mp.r.warn_chip("runs differ #1") in flagged
    assert ('<div class="shep-mp-delta"><span class="shep-st shep-st-warn"><span class="shep-st-shape" aria-hidden="true">▲</span><span class="shep-st-w">runs '
            "under these conditions differ — investigate: ") in flagged
    assert "Deterministic algorithms were on, so they were expected to agree." in flagged

    plain = mp.detail_parts(listing, data, _key(world, "model-05.pt"), NEWEST_CTL)["records"]
    assert '<span class="shep-tag shep-tag-grey">runs differ #1</span>' in plain
    assert '<div class="shep-mp-delta shep-muted">runs under these conditions differ: ' in plain
    assert "investigate" not in plain and "▲" not in plain and "expected to agree" not in plain
    assert "shep-tag-amber\">runs differ" not in plain and "shep-amber-t\">runs" not in plain

    # Without the runs' values (an API answer that could not show them), the
    # fields alone, the verb said once; with no field either, the words alone.
    state = {"state": "contested", "differing": ["pooled_mrr"], "runs": []}
    for deterministic in (False, None):
        text = mp.contest_html(state, mode="disease", deterministic=deterministic)
        assert text == ('<div class="shep-mp-delta shep-muted">runs under these conditions '
                        "differ on: pooled_mrr</div>")
        assert mp.contest_html({"state": "contested", "differing": [], "runs": []},
                               mode="disease", deterministic=deterministic) == (
            '<div class="shep-mp-delta shep-muted">runs under these conditions differ</div>')
    assert mp.contest_html(state, mode="disease", deterministic=True) == (
        '<div class="shep-mp-delta"><span class="shep-st shep-st-warn"><span class="shep-st-shape" aria-hidden="true">▲</span><span class="shep-st-w">runs under '
        "these conditions differ — investigate: pooled_mrr. Deterministic algorithms were on, "
        "so they were expected to agree.</span></span></div>")


@pytest.mark.parametrize("mode, values, shown", [
    ("C", (0.47333333333333333, 0.47334), ("0.47333333333333333", "0.47334")),
    ("D", (2.123456789, 2.1234567), ("2.123456789", "2.1234567")),
    ("disease", (0.47333333333333333, 0.47334), ("0.47333333333333333", "0.47334")),
], ids=["kept_c", "kept_d", "described"])
def test_runs_that_differ_past_the_fourth_decimal_never_read_as_equal(mode, values, shown):
    """The K8-6e.2 review: a contested group's runs are shown with each value as
    recorded, unrounded, whatever the record's mode (the Batch 3 ledger holds
    two Mode C records of one checkpoint), so two runs that differ past the
    fourth decimal never read as equal ("0.4733 vs 0.4733")."""
    state = {"state": "contested", "differing": ["untruncated_mrr"],
             "runs": [{"run": "r1", "values": {"untruncated_mrr": values[0]}},
                      {"run": "r2", "values": {"untruncated_mrr": values[1]}}]}
    for deterministic in (True, False):
        text = _text(mp.contest_html(state, mode=mode, deterministic=deterministic))
        assert f"untruncated_mrr {shown[0]} (run r1) vs {shown[1]} (run r2)" in text, text


def test_a_merged_record_and_a_named_run_show_their_possible_overlap(world):
    """G7, A7 (d) (K8-6r.2): a record merged before each run had its own record
    shows its run as "not recorded" and its merged report counted; it and the
    named run whose report it holds each show "possible overlap with record N
    (not recorded)"; a record whose run identity cannot be read is listed apart
    with its reason (K8-6e.1), never shown as merged and never counted as a
    result. HEAD: no run, no overlap, "1 corroborating report"."""
    import copy as _copy

    from src.api.services.record_views import ledger_entry

    named = _copy.deepcopy(dict(world["det"]))
    named["registration"] = {"sequence": 2, "registered_at": "2026-11-07T10:00:00Z"}
    merged = _copy.deepcopy(named)
    merged.pop("run_id")
    merged["registration"] = {"sequence": 1, "registered_at": "2026-10-01T10:00:00Z"}
    merged["source_artifact_digest"] = "d" * 64
    merged["corroborating_reports"] = [named["source_artifact_digest"]]
    lost = _copy.deepcopy(named)
    lost.pop("run_id")
    lost["registration"] = {"sequence": 3, "registered_at": "2026-11-07T11:00:00Z"}
    ledger = {"schema_version": 4, "run_identity_from_sequence": 2,
              "records": [merged, named, lost]}
    entry = ledger_entry(ledger, named["checkpoint_digest"])

    html, count, _contested, unreadable = mp._records_for_item(entry, {"contest": {}})

    assert (count, unreadable) == (2, 1)
    assert "from run not recorded, report dddddddddddd" in html
    assert f"from run {named['run_id']}, report" in html
    assert "1 report merged into this record before each run had its own record" in html
    assert "possible overlap with record 2 (not recorded)" in html
    assert "possible overlap with record 1 (not recorded)" in html
    assert "1 record for these weights cannot be read as a result" in _words(html)
    assert (f"2 #3 {lost['source_artifact_digest'][:12]} its run identity is missing"
            in _words(html))
    assert "corroborating" not in html


def test_records_whose_run_cannot_be_told_are_listed_with_the_boundary_reason(world):
    """A8 (review of K8-6r.2), K8-6e.1: with the file's run-identity boundary
    lost, each record that names no run is listed on the page by its index,
    registration and report with the reason the boundary gives (never shown as
    merged, never counted as a result), and the page names no path and no
    server log; the record that names its run is shown. HEAD: counted, with
    "(the server's log says why)"."""
    import copy as _copy

    from src.api.services.record_views import ledger_entry

    named = _copy.deepcopy(dict(world["det"]))
    named["registration"] = {"sequence": 2, "registered_at": "2026-11-07T10:00:00Z"}
    unnamed = _copy.deepcopy(named)
    unnamed.pop("run_id")
    unnamed["registration"] = {"sequence": 1, "registered_at": "2026-10-01T10:00:00Z"}
    ledger = {"schema_version": 4, "records": [unnamed, named]}   # no boundary
    entry = ledger_entry(ledger, named["checkpoint_digest"], "/srv/x/evaluations.json")

    html, count, _contested, unreadable = mp._records_for_item(entry, {"contest": {}})

    assert (count, unreadable) == (1, 1)
    assert f"from run {named['run_id']}, report" in html
    assert "from run not recorded" not in html
    said = _words(html)
    assert "▲ 1 record for these weights cannot be read as a result." in said
    assert ("0 #1 " + named["source_artifact_digest"][:12] + " whether it names its run cannot "
            "be told: the file's run-identity boundary (run_identity_from_sequence) cannot be "
            "read (it is missing while the file holds 2 records)") in said
    assert "/srv/x" not in html and "server's log" not in said


def test_an_old_corroborated_usage_entry_is_shown_as_recorded():
    """G6 (K8-6r.2), a guard: a usage-history entry written before K8-6r.2 with
    the retired outcome ``corroborated`` is shown as it was recorded, never
    reworded or dropped; a new one names its run and its own sequence."""
    old = mp.usage_entry_html({"kind": "registered", "at": "2026-10-20T00:00:00Z",
                               "version": "v", "run_id": "r-old", "outcome": "corroborated",
                               "sequence": 1})
    new = mp.usage_entry_html({"kind": "registered", "at": "2026-11-07T00:00:00Z",
                               "version": "v", "run_id": "r-new", "outcome": "registered",
                               "sequence": 2})
    assert "#1 · corroborated" in old and "report of run r-old" in old
    assert "#2 · registered" in new and "report of run r-new" in new
