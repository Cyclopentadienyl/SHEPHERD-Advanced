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
  under the workspace select; escaping;
- **in a browser** (UI-05, UI-11): a control changed while the listener
  answers something else is applied (the switch, the count line, the groups
  and Lineage agree), a fast second change wins, "Applying …" is said while a
  change is pending, and one pick of the Task select is one request.

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

from src.evaluation.caveats import metric_text, mode_label  # noqa: E402
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


def _add_child(world, run, parent_sha, parent_run):
    """One more disease fine-tune, a file of `run` whose recorded parent is
    `parent_sha` of `parent_run` (in the base's family: a lineage one run
    deeper than the base's children)."""
    from src.utils.checkpoint_io import write_checkpoint
    from tests.unit import test_task_first_selection as bundles

    fingerprint = file_sha256(world["workspace"] / "split_manifest.json")
    path = world["workspace"] / "runs" / "gatv2" / run / "checkpoints" / "last.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    sha = write_checkpoint({**bundles._record_payload(
        "disease", logs={"val_pooled_mrr": 0.43},
        parent={"sha256": parent_sha, "run": parent_run}),
        "checkpoint_kind": "weights", "training_input_digests": {"split_manifest": fingerprint},
        "epoch": 3}, path)
    os.utime(path, (time.time() - 19 * 3600,) * 2)
    world["paths"][f"{run}/last.pt"] = path
    return sha


def _run_row(html, run):
    """The tree's summary row of the run `run` (``architecture/run``)."""
    at = html.index(f'data-fold="r:{run}"')
    start = html.rfind("<tr", 0, at)
    return html[start:html.index("</tr>", at)]


def _file_row(html, key):
    """The tree's row of the file `key`."""
    at = html.index(f'data-key="{key}"')
    start = html.rfind("<tr", 0, at)
    return html[start:html.index("</tr>", at)]


def _cells(row):
    """A table row's cells as the words they show (tags dropped)."""
    return [re.sub(r"<[^>]+>", "", cell) for cell in re.findall(r"<td[^>]*>(.*?)</td>", row)]


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


def _sections(markup):
    """Each ledger record on Models, in order (`record_block_html`'s section)."""
    return re.findall(r'<section class="shep-mp-rec".*?</section>', markup, flags=re.S)


def _rows(section):
    """A record's rows as the one record rendering draws them
    (`renderers.record_rows`, R-82): each top-level key of its summary list and
    its value's words, in order (a value's own lists and tables inside it)."""
    from html.parser import HTMLParser

    class Rows(HTMLParser):
        def __init__(self):
            super().__init__()
            self.depth, self.inside, self.rows, self.cell = 0, False, [], None

        def handle_starttag(self, tag, attrs):
            classes = dict(attrs).get("class") or ""
            if tag == "dl":
                self.depth += 1
                if "shep-rec-rows" in classes:
                    self.inside, self.depth = True, 1
            elif self.inside and self.depth == 1 and tag in ("dt", "dd"):
                self.cell = [tag, []]
            if self.cell is not None and tag not in ("dt", "dd"):
                self.cell[1].append(" ")

        def handle_endtag(self, tag):
            if tag == "dl" and self.inside:
                self.depth -= 1
                if self.depth == 0:
                    self.inside = False
            elif self.inside and self.depth == 1 and tag in ("dt", "dd") and self.cell:
                text = " ".join("".join(self.cell[1]).split())
                if tag == "dt":
                    self.rows.append([text, None])
                else:
                    self.rows[-1][1] = text
                self.cell = None
            elif self.cell is not None:
                self.cell[1].append(" ")

        def handle_data(self, data):
            if self.cell is not None:
                self.cell[1].append(data)

    parser = Rows()
    parser.feed(section)
    return [tuple(row) for row in parser.rows]


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
                                    mode=r.record_mode(view),
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
    assert view == {"task": "all", "note": mp.LOAD_NOTE, "shaped_for": "newest|all"}
    html = ml.list_html(listing, None, ml.PRESETS["research"], view=view)
    from src.webui.style import CELL_WHOLE

    # The rows say what they were drawn for (the "Applying …" line compares it).
    assert html.endswith('<span class="shep-view-for" hidden data-for="newest|all"></span>')
    # Each run's summary row says what its files are (D04: a run's files share
    # their record), its files folded under it.
    # A run whose files read differently says what its readable file is; the
    # one never read says so in its own row.
    for key, kind in ((_key(world, "model-05.pt"), "Record not readable"),
                      (_key(world, "broken.pt"), "Record not readable"),
                      (bundle_key("base"), "Base"), (bundle_key("gene"), "Gene")):
        run = key.rsplit("/", 1)[0]
        assert f'<td class="{CELL_WHOLE}">{kind}</td>' in _run_row(html, run), key
        assert ' hidden><td><div class="shep-tree-name shep-tree-leaf' in _file_row(html, key)
    assert "Metadata not readable: the file cannot be read as a checkpoint. It cannot be loaded." in (
        _file_row(html, _key(world, "broken.pt")))
    ctl = mp.controls(data, mp.NEWEST, "disease")
    assert mp.shape(listing, data, ctl) == {"task": "disease", "note": mp.LOAD_NOTE,
                                            "shaped_for": "newest|disease"}
    assert mp.controls(data, mp.NEWEST, "gatv2")["task"] == "all"



def test_a_record_of_another_or_no_mode_is_shown_as_recorded(world, monkeypatch):
    """A record or usage entry whose mode this version does not describe, or
    whose mode cannot be read, is named for what it is, by the one display for
    every mode (R2, K8-6e.2): its own label; a "Recorded metrics" column with
    every metric as recorded in place of the fixed columns and their "not
    described" cells; no claim of what its cases are; never another mode's
    words. An entry written before `mode` existed reads "Measurement mode not
    recorded", never Mode C, with its metrics as recorded one click away."""
    from tests.fixtures.r10_table import without_r10

    _listing, data = _read(world, monkeypatch)
    record = next(r for entry in data["ledgers"].values() for r in entry.get("records") or [])
    view = mp.record_view(record)
    assert view["mode"] == "disease"
    for mode, label in (("E", "Mode E (a measurement mode this version does not describe)"),
                        (None, "Measurement mode not recorded")):
        raw = mp.records_block_html([{**view, "mode": mode}], {})
        text = _text(raw)
        assert label in text
        assert "Mode C" not in text and "Disease" not in _text(without_r10(raw))
        # The one record rendering (R-82): its coverage, then every metric as
        # recorded as one row, no tiles and no disclosure of them again.
        (section,) = _sections(raw)
        assert [key for key, _value in _rows(section)][:2] == ["Coverage", "Recorded metrics"]
        assert "shep-rec-tile" not in raw and "shep-rec-fold" not in raw
        assert ">not described<" not in raw and "What they are is not described" in text
        for name, value in view["metrics"].items():          # every one, unrounded
            assert f"{name} {json.dumps(value)}" in _words(raw), name
        assert "The Trainer validation figure is not among these records' metrics." in text
    entry = {"kind": "test_run", "at": "2026-10-26T12:00:00Z", "version": "v", "run_id": "r",
             "metrics": {"untruncated_mrr": 0.5}, "n_ranked": 3, "device": "cpu",
             "deterministic": True, "record_requested": False}
    row = mp.usage_entry_html(entry)
    assert "Measurement mode not recorded, 3 cases" in _text(row) and "Mode C" not in _text(row)
    assert ('<details class="shep-rec-fold"><summary>Recorded metrics (1)</summary>' in row
            and "<details class=\"shep-rec-fold\" open" not in row)
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
    """The records of the mode `mode_label_text` names: from its label's line
    (said once above them) to the next mode's, or the block's closing note."""
    start = raw.index(f'<p class="shep-mp-p"><b>{e(mode_label_text)}.</b>')
    ends = [at for at in (raw.find('<p class="shep-mp-p"><b>', start + 1),
                          raw.find('<p class="shep-note">', start + 1)) if at > 0]
    return raw[start:min(ends)]


def _value(markup, key):
    """The value's words of the first record row named `key` in `markup`."""
    return dict(_rows(_sections(markup)[0]))[key]



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
        assert '<dt class="shep-k">Recorded metrics</dt>' in table
        assert ">not described<" not in table and 'class="shep-rec-fold"' not in table
    c_main = _value(c_table, "Recorded metrics")
    for line in ("untruncated_mrr 0.5", "hits@1 0.25", "hits@10 0.75", "mean_rank 3.0"):
        assert line in c_main, line
    d_main = _value(d_table, "Recorded metrics")
    for line in ("post_import_mrr 0.35", "post_import_hits@1 0.2", "coverage 0.7",
                 "reached_mrr 0.5", "reached_mean_rank 2.43", "reached_hits@10 0.9"):
        assert line in d_main, line
    assert "Score semantics raw cosine, no eta mixture and no shortest-path term" in words
    assert "Score semantics served:gnn_only:eta=1.0:sp=none" in words
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
    assert two.count(f"<b>{e(d_label)}.</b>") == 1 and len(_sections(two)) == 2
    assert "Score semantics served:gnn_only:eta=1.0:sp=none" in _words(two)
    assert "Score semantics served:gnn_plus_shortest_path:eta=0.7:sp=abc" in _words(two)
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

    from src.utils.run_environment import native_identity_lines

    # One "Native build" row per record (the one record rendering: a line per
    # component under its lead, as the Test page shows it).
    sections = _sections(raw)
    assert raw.count('<dt class="shep-k">Native build</dt>') == len(sections) == 3
    native = [dict(_rows(section))["Native build"] for section in sections]
    assert native.count("not recorded") == 2
    assert native_identity_text(views[2]["native_identity"]) == (
        "local build — pyg_lib 0.8.0 aaaaaaaaaaaa")
    assert " ".join(native_identity_lines(views[2]["native_identity"])) in native
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
    assert "Score semantics zz: a new score" in _words(table)
    assert raw.count('<p class="shep-mp-p"><b>') == 4 and len(_sections(raw)) == 4


def test_a_task_record_keeps_its_columns_and_shows_every_metric_one_click_away():
    """A record of a measurement this build describes keeps its two columns and
    its readable words (R2); every metric as recorded is one click away in a
    closed native disclosure, "Recorded metrics (N)", with the raw names and
    unrounded values; its score semantics as recorded on its detail line."""
    from tests.unit.test_evaluation_sidecar import DISEASE_METRICS, registered_report

    records = _kept_c_and_d_records((registered_report(), "f" * 64))
    view = mp.record_view(records[2])
    raw = mp.records_block_html([view], {})
    # Its answer first: the task's tiles (the one record rendering, R-82).
    # R-71: each tile's name less the task's, which the group's label says once.
    tiles = re.findall(r'shep-rec-tile-l">([^<]*)<', raw)
    assert tiles == ["Pooled MRR", "Pooled top-1", "Pooled top-10", "Coverage"]
    assert all(metric_text("disease", name).lower() == f"disease {tile.lower()}"
               for name, tile in zip(r.RECORD_TILES["disease"], tiles, strict=True))
    assert '<dt class="shep-k">Recorded metrics</dt>' not in raw
    # B52-E: in the record's one "Details ›", with its engineering rows.
    fold = re.search(r'<dt class="shep-k">Recorded metrics \((\d+)\)</dt>'
                     r'<dd class="shep-v">(.*?)</dd>', raw)
    assert fold and int(fold.group(1)) == len(DISEASE_METRICS)
    assert raw.count('<details class="shep-more"><summary>Details ›</summary>') == 1
    assert "macro_precision@3 0.16666666666666666" in _words(fold.group(2))
    assert "pooled_top10 0.75" in _words(fold.group(2))
    assert ("Conservative pooled MRR", "0.4500 an unranked case whose truth can be assessed "
            "counts 0") in _rows(raw)
    assert ("Score semantics disease head: -||z_d - z_P||^2 on L2-normalised projected vectors"
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
                 "Score semantics served:gnn_only:eta=1.0:sp=none", "untruncated_mrr 0.5",
                 "post_import_mrr 0.35", "Recorded metrics (", "Pooled MRR"):
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
        assert f"{hostile} 0.5" in _words(raw) and f"Score semantics {hostile}" in _words(raw)
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
    assert "CUDA · deterministic, registration asked" in _text(disease)
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
    """The Trainer's validation figure in its own row of the detail's summary
    list (never a box of its own inside the panel: the skill's §2; the
    ops-wiring round's B3), the recorded test results in theirs, never together (the world's records are disease task
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
    assert ('<dt class="shep-k">Validation<button' in top
            and "pooled MRR (Trainer) 0.2410 — " in top and "Disease pooled MRR" not in top)
    assert "Validation Validation" not in _text(top) and "shep-card" not in top
    assert ("Validation pooled MRR" not in records and "(Trainer)" not in records
            and "0.2410" not in records)
    assert e(mode_label("disease")) in records and "Disease pooled MRR" in records
    # Each relation a row of its own under its record, its meaning first and
    # its code after it (`renderers.relations_html`; never a chain of short
    # names with the full ones in a title: BACKLOG 72, the ops-wiring round's
    # B3; R-72), in the list whose key column keeps the name on one line.
    assert '<dl class="shep-kv shep-kv-rel">' in records
    assert ('<dt class="shep-k">Model trained on this graph (R3)</dt><dd class="shep-v">'
            + r.status_mark("ok", "Verified")) in records
    assert "R3 model" not in records
    assert "Relations:" not in records and "version↔graph" not in records
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


def test_test_and_models_draw_one_record_by_one_rendering_with_no_chain(world, monkeypatch):
    """R-82, NEW-models-results-1, NEW-test-2: the same report record was drawn
    by two code paths, Test's summary list and Models' table, which put about
    24 metrics in one ' · ' chain and 'Environment: … · Dataset: … ·
    Registered …' in another. Both pages now draw it by the one record
    rendering (`renderers.record_html`): the same rows in the same order with
    the same values, but for what only each page knows (the registration's
    words; the dataset, said by its relation where it is the selected
    workspace's), metrics and environment facts each a row, no value with
    more than one ' · ' (the skill's §5)."""
    from src.webui.components import test_page as tp

    _listing, data = _read(world, monkeypatch)
    record = next(rec for entry in data["ledgers"].values() for rec in entry.get("records") or [])
    models = _rows(mp.record_block_html(mp.record_view(record), None))
    test = _rows(tp._record_html(record, {}, {}))
    own = ("Dataset", "Registration", "Run id")
    assert [row for row in models if row[0] not in own] == [
        row for row in test if row[0] not in own]
    keys = [key for key, _value in models]
    for key in ("Coverage", "Conservative pooled MRR", "Pooled top-k", "Score semantics",
                "Relations", r.R10_TITLE, "Environment", "Device", "Deterministic algorithms",
                "Native build", "Registration"):
        assert key in keys, key
    assert keys.index("Coverage") < keys.index("Environment") < keys.index("Registration")
    for markup in (mp.record_block_html(mp.record_view(record), None),
                   tp._record_html(record, {}, {})):
        for key, value in _rows(markup):
            assert value.count(" · ") <= 1, (key, value)
        assert "Environment:" not in _text(markup) and "Dataset:" not in _text(markup)


def test_the_records_dataset_is_said_once_by_its_relation(world, monkeypatch):
    """R-66+B51f, NEW-models-results-3: each record said its dataset's origin
    three times ('Dataset of every record below: synthetic …', a 'synthetic'
    chip in its cell, 'Dataset: synthetic' in its environment line): with
    two records, 'synthetic' six times on one screen. Records that measured
    the selected workspace's dataset record are said so once, by relation
    (the workspace line says its origin); records that share another origin
    say it once above them; only records whose origins differ say their own."""
    _listing, data = _read(world, monkeypatch)
    records = [rec for entry in data["ledgers"].values() for rec in entry.get("records") or []]
    views = [mp.record_view(rec) for rec in records[:2]]
    assert len(views) == 2 and all(view["dataset_origin"] == "synthetic" for view in views)
    ours = views[0]["dataset_sha256"]
    assert r.is_digest(ours)
    same = [{**view, "dataset_sha256": ours} for view in views]
    block = mp.records_block_html(same, {}, workspace_dataset=ours)
    assert mp.RECORDS_WORKSPACE_DATASET in _text(block) and "synthetic" not in _text(block)
    assert all("Dataset" not in dict(_rows(section)) for section in _sections(block))
    other = mp.records_block_html(same, {}, workspace_dataset="f" * 64)
    assert _text(other).count("synthetic") == 1
    assert "Dataset of every record below: synthetic" in _text(other)
    mixed = mp.records_block_html([same[0], {**same[1], "dataset_origin": "files"}], {})
    assert [dict(_rows(section))["Dataset"] for section in _sections(mixed)] == [
        "synthetic — a generated stand-in, not a clinical knowledge graph",
        "files — built from identified source files"]


def test_the_detail_header_says_what_is_in_use_only_when_it_is_another(world, monkeypatch):
    """R-67: with no model served, the detail said one fact twice, a grey "not
    in use" chip and "nothing is in use for diagnosis" on the right. It says
    the chip alone; the right-hand words appear only where another model is
    in use (or what is in use could not be read)."""
    listing, data = _read(world, monkeypatch)
    key = _key(world, "model-05.pt")
    nothing = {**listing, "served_sha": None, "status_read": True, "status_problem": None}
    header = mp.detail_parts(nothing, data, key, NEWEST_CTL)["header"]
    assert "not in use" in header and "nothing is in use" not in header
    assert "shep-panel-right" not in header
    other = {**listing, "served_sha": world["shas"]["model-12.pt"], "status_read": True}
    header = mp.detail_parts(other, data, key, NEWEST_CTL)["header"]
    assert "in use for diagnosis: model" in header
    unread = {**nothing, "status_problem": "the status could not be read"}
    assert "in use for diagnosis: not read" in mp.detail_parts(unread, data, key,
                                                               NEWEST_CTL)["header"]


def test_an_operators_list_header_says_no_operator_controls(world, monkeypatch):
    """R-18+R-40+R-75+B68+B72: for an operator on loopback, whose header chip
    reads 'Operator controls: on (this machine)', the list header still read
    'listed by the server (operator controls only)'. An operator's header is
    the list's title alone; "Operator controls only" is said only where D8
    refuses (the read-only form)."""
    listing, data = _read(world, monkeypatch)
    assert not listing.get("refused")
    header = mp.left_header_html(listing, data)
    assert "operator controls" not in _text(header).lower() and "🔒" not in header
    assert header == f'<div class="shep-panel-hd"><h2>{e(mp.list_title(listing))}</h2></div>'


def test_a_records_investigation_notes_are_shown_on_models(world, monkeypatch):
    """B40: a contested record's investigation notes were shown on the Test
    page only. Models shows its group's notes under the group's state, by the
    one rendering (`renderers.investigation_notes_html`), the operator's view
    and the read-only one alike; notes that cannot all be read say so, never
    "no note is kept"; every value escaped."""
    from src.api.services.record_states import key_text

    _listing, data = _read(world, monkeypatch)
    record = next(rec for entry in data["ledgers"].values() for rec in entry.get("records") or [])
    view = mp.record_view(record)
    state = {"state": "contested", "differing": ["pooled_mrr"], "runs": [], "problems": [],
             "notes": [{"who": "<b>Dr A</b>", "at": "2026-11-16T10:00:00Z",
                        "what": "seed drift <img src=x onerror=alert(1)>"}],
             "notes_problems": []}
    block = mp.records_block_html([view], {key_text(view["key"]): state})
    said = _text(block)
    assert "Investigation notes (1)" in said and "<b>Dr A</b>" in said
    assert "2026-11-16 10:00:00 UTC" in said and "seed drift <img" in said
    assert "<img" not in block and "<b>Dr A" not in block
    unread = {**state, "notes": [], "notes_problems": [""]}
    said = _text(mp.records_block_html([view], {key_text(view["key"]): unread}))
    assert "The investigation notes cannot all be read" in said
    assert "No investigation note is kept" not in said
    # An operator's reason names no server path (the one redaction).
    told = {**state, "notes": [], "notes_problems": [
        "/srv/secret/reports/notes.jsonl line 2 is not a JSON object"]}
    said = _text(mp.records_block_html([view], {key_text(view["key"]): told}))
    assert "line 2 is not a JSON object" in said and "/srv/secret" not in said
    none = {**state, "notes": []}
    assert "No investigation note is kept for this group." in _text(
        mp.records_block_html([view], {key_text(view["key"]): none}))


def test_a_record_that_cannot_be_read_is_named_apart_from_the_results(world, monkeypatch):
    """K8-6e.1 (R3): model-05's ledger holds three records; one loses
    ``n_ranked`` by a hand edit. The segment counts the results and names the
    rest with one " · " ("Recorded results · 2 (1 not readable, 1 where
    runs differ)"); the detail lists it by index, registration, report and
    reason under the amber sentence, never as a result and never pointing to
    the server's log; the row reads "2 · 1 not readable, runs differ for 1".
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
        "Recorded results · 2 (1 not readable, 1 where runs differ)", mp.RECORDS)
    assert detail["seg"][0][0].count(" · ") == 1
    said = _words(detail["records"])
    assert ("▲ 1 record for these weights cannot be read as a result. It is kept as "
            "written.") in said
    assert (f"Index Registration Report Reason {index} #2 "
            f"{bad['source_artifact_digest'][:12]} n_ranked: missing") in said
    assert "registered #2" not in detail["records"]
    assert "server's log" not in said
    assert r.recorded_tests_text(ml._recorded(listing, _key(world, "model-05.pt"))) == (
        "Recorded results: 2 · 1 not readable, runs differ for 1")


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
    assert '<p class="shep-mp-p shep-mp-unread-p"><span class="shep-st shep-st-warn"><span class="shep-st-shape" aria-hidden="true">▲</span><span class="shep-st-sp">\u00a0</span><span class="shep-st-w">1 record' in html
    assert '<div class="shep-tbl-wrap shep-mp-recwrap shep-mp-unreadwrap">' in html


def test_unverifiable_is_a_word_and_never_a_count(world, monkeypatch):
    from tests.fixtures.r10_table import cell_text, r10_cells

    listing, data = _read(world, monkeypatch)
    records = mp.detail_parts(listing, data, _key(world, "model-08.pt"), NEWEST_CTL)["records"]
    training = r10_cells(records)["Training"]
    for column in ("Disease overlap", "Phenotype-set overlap"):
        cell = training[column]
        assert cell.startswith(r.status_mark("neutral", "Unverifiable"))
        assert "there is no count" in cell
        assert " of " not in cell_text(cell) and not re.search(r"\d", cell_text(cell))
    state, words, count = r.r10_state_parts(
        "disease_overlap", {"state": "unverifiable", "k": None, "n": 50, "cases": None},
        "not_found")
    assert (state, count) == ("unverifiable", None) and "0" not in words


def test_a_records_r10_is_a_table_with_a_mark_on_each_overlap_and_no_stutter():
    """R-63: Models' record shows R10 as its table (training / validation by
    disease / phenotype set), each overlap cell amber with ▲ and its count,
    denominator, unit and cases, never "overlap: overlap" nor a run-on " · "
    sentence; "none" in its own quiet words with its count, never a green
    pass and never "no overlap"; the evidence's path never shown. At
    ba6e85e it was one dark ' · ' sentence per scope with no mark."""
    from tests.fixtures.r10_table import cell_text, r10_cells
    from tests.unit.test_evaluation_sidecar import r10_block

    block = r10_block("overlap")
    html = r.r10_table_html(block)
    cells = r10_cells(html)
    assert list(cells) == ["Training", "Validation"]
    assert set(cells["Training"]) == {"evidence", "Disease overlap", "Phenotype-set overlap"}
    disease = cells["Training"]["Disease overlap"]
    assert disease.startswith(r.status_mark("warn", "Overlap"))
    assert cell_text(disease) == ("▲ Overlap 3 of 50 distinct test diseases, 5 test cases "
                                  "with one of them")
    none = cells["Training"]["Phenotype-set overlap"]
    assert "shep-st" not in none and cell_text(none) == (
        "None in the inputs this checkpoint records (not evidence about data outside them) "
        "0 of 80 test cases")
    text = cell_text(html)
    assert "overlap: overlap" not in text.lower() and " · " not in text
    assert "no overlap" not in text.lower() and "/abs/" not in html
    assert '<table class="shep-r10" aria-label="Train/test overlap (R10)">' in html


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

    html = r.r10_table_html(block)

    # The one R10 rendering (`renderers.r10_table_html`) under its title: a
    # row per scope with its evidence's state, a cell per label.
    from tests.fixtures.r10_table import cell_text, r10_cells

    assert html.startswith(f'<table class="shep-r10" aria-label="{r.R10_TITLE}">')
    training = r10_cells(html)["Training"]
    assert training["evidence"] == f"evidence {evidence_words}"
    for column in ("Disease overlap", "Phenotype-set overlap"):
        cell = training[column]
        assert cell == (r.status_mark("neutral", "Unverifiable")
                        + f'<div class="shep-r10-n">{e(reason[:1].upper() + reason[1:])}</div>')
        text = cell_text(cell)
        assert not re.search(r"\d", text) and "—" not in text and "none" not in text.lower()
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
        tree = ml.tree_view(listing, mp.shape(listing, data, ctl), ml.PRESETS["research"])
        shown = [ml.find_item(listing, k) for k in tree.matched]
        for key, (_sha, _path) in models.items():
            assert items[key] in shown, (order, key)
            detail = mp.detail_parts(listing, data, key, ctl)
            assert detail["seg"] == [("Recorded results · 2", mp.RECORDS),
                                     ("Usage history · 2", mp.USAGE),
                                     ("Results", mp.RESULTS)], (order, key)
            for label in blocks:
                assert detail["records"].count(f'<div class="shep-mp-rec-hd"><b>{label}</b>') == 1
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
    assert ("The recorded results beside this checkpoint cannot be read, so none is shown: "
            in _text(detail["records"]))
    assert str(world["root"]) not in detail["records"] + detail["top"]
    none = mp.detail_parts(listing, data, _key(world, "model-30.pt"), ctl)
    # R-68: a titled state, what it means and one next step (its action,
    # `test_the_empty_records_next_step_opens_test_with_this_model_chosen`);
    # no file named, never "ledger", and "· 0" nowhere.
    said = _text(none["records"])
    assert said.startswith("No recorded result yet") and "never evaluated" in said
    assert said.endswith("Next: start a test of it with Register when the run ends checked. "
                         "Its result then appears here. Test this model ›")
    assert "ledger" not in said.lower() and "evaluations.json" not in said
    assert [label for label, _value in none["seg"]] == ["Recorded results", "Usage history",
                                                       "Results"]


def test_the_empty_records_next_step_opens_test_with_this_model_chosen(world, monkeypatch):
    """R-68: the empty state's next step was words ("Next: on the Test page,
    choose this model and …"), a dead end for the reader. It is one action,
    "Test this model ›", through the cross-link the pages share
    (`PageContext.go` / ``nav_outputs``, as Fine-tune's link to Training):
    a queued click that hands the selected checkpoint to the Test page
    (`ctx.test_model`) and opens it; the Test page's opening chooses that
    workspace and checkpoint and clears the hand-off. A read-only request
    keeps its refusal words and gets no action."""
    from src.webui.components import test_page as tp
    from src.webui.shell import standalone_context

    operator = mp.records_empty_html("none")
    assert operator.count("data-act=") == 1
    assert ('<button type="button" class="shep-link" data-act="test">Test this model ›'
            '</button>') in operator
    readonly = mp.records_empty_html("no_ledger", readonly=True)
    assert "data-act" not in readonly and "<button" not in readonly
    assert _text(readonly).endswith(mp.RECORDS_NEXT_READONLY)

    with gr.Blocks() as demo:
        ctx = standalone_context(demo)
        mp.create_models_page(ctx)
        tp.create_test_page(ctx)
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    to_test = _fn(demo, "_to_test")
    assert to_test.targets == [(by_id["shep-mp-records"]._id, "click")]
    assert to_test.outputs == [ctx.test_model, *ctx.nav_outputs]
    assert to_test.queue is not False                 # its State write precedes the opening
    listing, _data = _read(world, monkeypatch)
    key = _key(world, "model-30.pt")
    handed, *nav = to_test.fn(listing, key, gr.EventData(None, {"act": "test", "key": ""}))
    assert handed == {"workspace": "demo", "key": key}
    assert [getattr(u, "selected", u) for u in nav] == [
        getattr(u, "selected", u) for u in ctx.go("test")]
    for act, selected in (("dismiss", key), ("test", None), ("test", "hgt/no-such/run.pt")):
        kept, *nav = to_test.fn(listing, selected, gr.EventData(None, {"act": act}))
        assert kept == gr.update() and nav == list(ctx.no_nav()), (act, selected)

    # The Test page's opening takes the hand-off, chooses it and clears it.
    opened = [fn for fn in demo.fns.values() if fn.name == "_refresh"
              and (ctx.page_tokens["test"]._id, "change") in fn.targets]
    assert len(opened) == 1
    opened = opened[0]
    assert ctx.test_model in opened.inputs and ctx.test_model in opened.outputs
    assert by_id["shep-test-model"] in opened.outputs
    request = operator_request(monkeypatch)
    out = dict(zip(map(id, opened.outputs),
                   opened.fn(None, None, None, None, {}, None, handed, request)))
    assert out[id(ctx.test_model)] is None
    assert out[id(by_id["shep-test-model"])]["value"] == key
    assert out[id(by_id["shep-test-workspace"])]["value"] == "demo"
    # With nothing handed, the opening keeps the reader's own choice.
    kept = _key(world, "model-12.pt")
    out = dict(zip(map(id, opened.outputs),
                   opened.fn("demo", None, kept, None, {}, None, None, request)))
    assert out[id(by_id["shep-test-model"])]["value"] == kept


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


def test_by_family_draws_each_run_under_the_exact_parent_file_it_records(world, monkeypatch):
    """By family (D04, RC-D; the family graph since the stage-lineage round,
    G1-G5): the view the page shapes from the listing's family view, one
    group per family with its runs in lineage order and depth, **each run's
    edge from the exact parent file its record names** (by SHA-256 on the
    server, never a name), a run whose parent is not listed here a boundary
    in the family view's own words (escaped); under All the records not
    readable (the files without a model spec among them) and the records not
    read follow, each group under its own word; a task's view keeps its base
    as a context run, said in words. The shared list draws it as the graph
    (nested lists, one node per run) and those files as the table, under "Not
    in a family"; the family table's group header rows are gone."""
    _add_bundles(world, orphan_parent_run=EVIL)
    listing, data = _read(world, monkeypatch)
    ctl = mp.controls(data, mp.BY_FAMILY, None)
    assert ctl["order"] == mp.BY_FAMILY
    view = mp.shape(listing, data, ctl)
    assert "column" not in view and "cells" not in view and not hasattr(mp, "FAMILY_TITLE")
    # A file whose record was never read is under its own word, "Record
    # unknown" (`record_kind_text`), never under "Record not readable"; the files
    # without a model spec, refused by the record reader, under the word each
    # of their rows leads with (K6 closing review: no page word of its own).
    refused_word = r.record_kind_text({"record": "refused"})
    assert [(g["title"], g["family"]) for g in view["groups"]] == [
        ("", True), ("", True), (refused_word, False), ("Record unknown", False)]
    assert not hasattr(mp, "GROUP_HISTORY") and not hasattr(mp, "_run_note")
    assert {r.record_kind_text(ml.find_item(listing, key))
            for key in view["groups"][2]["keys"]} == {refused_word}
    assert view["groups"][0]["keys"] == [bundle_key("base"), bundle_key("disease"),
                                         bundle_key("gene")]
    assert [run["run"].split("/")[1] for run in view["groups"][0]["runs"]] == [
        "20261001-000000_base_aaaaaa", "20261002-000000_disease_bbbbbb",
        "20261003-000000_gene_cccccc"]
    # A run is its key only: how deep it is, the edges say (nothing kept that no one reads).
    assert all(set(run) == {"run"} for g in view["groups"] for run in g["runs"])
    assert view["parents"][bundle_key("gene")] == bundle_key("base")
    # Each child run's edge from the exact parent file, keyed by that file.
    assert view["edges"] == [{"from": bundle_key("base"), "to": "gatv2/" + BUNDLES["disease"][0]},
                             {"from": bundle_key("base"), "to": "gatv2/" + BUNDLES["gene"][0]}]
    assert view["groups"][3]["keys"] == [_key(world, "broken.pt")]
    assert len(view["groups"][2]["keys"]) == 7                       # every ShepherdGNN file
    # A parent not listed: a boundary in the family view's words, never a parent guessed.
    (orphan,) = view["groups"][1]["runs"]
    bound = view["bounds"][orphan["run"]]
    row = mp.family_row(listing, _key_of(listing, "20261004-000000_disease_dddddd"))
    assert (bound["state"], bound["problem"], bound["run"], bound["sha256"]) == (
        "unavailable", row["parent_problem"], EVIL, "9" * 64)
    assert row["parent_state"] == "unavailable" and EVIL in row["parent_problem"]
    # The shared list draws the graph: one node per run, a child in its parent's list.
    html = ml.list_html(listing, None, ml.PRESETS["research"], view=view)
    assert EVIL not in html and html_lib.escape(EVIL) in html and "<img" not in html
    assert ">Lineage</th>" not in html and "shep-grp" not in html and "Family from base" not in (
        re.sub(r'aria-label="[^"]*"', "", html))
    base_li = html[html.index('<li class="shep-fam-li" data-run="gatv2/' + BASE_RUN + '"'):]
    kids = base_li[base_li.index('<ul class="shep-fam-kids"'):]
    for child in ("disease", "gene"):
        assert f'data-run="gatv2/{BUNDLES[child][0]}"' in kids.split("</ul>")[0], child
    assert html.count('<li class="shep-fam-li" data-run=') == 4
    assert _words(_node(html, "gatv2/" + BUNDLES["disease"][0])).startswith(
        "disease · GATv2 2026-10-02 00:00 Disease · Fine-tune")
    assert "▲ Parent unavailable" in _words(html[html.index("shep-fam-bound"):][:600])
    # The files the graph cannot place: under "Not in a family", as the table draws them.
    rest = html[html.index(ml.NOT_IN_A_FAMILY):]
    assert f'data-key="{_key(world, "broken.pt")}"' in rest
    refused = _run_row(rest, _key(world, "model-05.pt").rsplit("/", 1)[0])
    # A file with no lineage says its state's own word (Unknown), never a dash.
    assert _cells(refused)[1:3] == [refused_word, "Unknown"] and "—" not in refused
    # A record this build refused and a file never read: two groups, each
    # under its own word (skill §6: missing states are never merged).
    broken = ml.find_item(listing, _key(world, "broken.pt"))
    # Another file: its own identity (keyed by its names), never the copy's.
    refused = {**broken, "filename": "refused.pt", "key": None, "record": "refused",
               "record_problem": "an unknown field"}
    row = {**mp.family_row(listing, _key(world, "broken.pt")), "key": ml.item_key(refused)}
    both = {**listing, "items": [*listing["items"], refused],
            "family": {**listing["family"], "all": [*listing["family"]["all"], row]}}
    groups = {g["title"]: g["keys"] for g in mp.family_view(both, mp.ALL_TASKS)["groups"]
              if not g["family"]}
    assert groups["Record unknown"] == [_key(world, "broken.pt")]
    assert groups["Record not readable"][-1] == ml.item_key(refused)
    assert _key(world, "broken.pt") not in groups["Record not readable"]
    # A task's view: its base as a context run, quieter, ○ and its words.
    ctl = mp.controls(data, mp.BY_FAMILY, "gene")
    view = mp.shape(listing, data, ctl)
    assert [g["keys"] for g in view["groups"]] == [[bundle_key("base"), bundle_key("gene")]]
    assert view["context"] == [bundle_key("base")]
    html = ml.list_html(listing, None, ml.PRESETS["research"], view=view)
    base = _node(html, "gatv2/" + BASE_RUN)
    assert "shep-ctx" in base and mp.r.status_mark("neutral", "Parent of a shown run") in base
    assert "shep-ctx" not in _node(html, "gatv2/" + BUNDLES["gene"][0])
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


def _key_of(listing, run):
    """The listed key of the first file of the run directory `run`."""
    return next(ml.item_key(i) for i in listing["items"] if i.get("run") == run)


def _node(html, run):
    """A family graph node (its box, its body and its arrow) of the run `run`."""
    at = html.index(f'<li class="shep-fam-li" data-run="{run}"')
    end = html.find('<ul class="shep-fam-kids"', at)
    close = html.index("</li>", at)
    return html[at:end if 0 < end < close else close]


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
        out = opened.fn("demo", False, None, None, None, "", {}, mp.BY_FAMILY, mp.ALL_TASKS,
                        mp.RECORDS, "idle", request)
        by = dict(zip(map(id, opened.outputs), out, strict=True))
        return next(v for v in out if isinstance(v, dict) and "listing_key" in v), by[id(order)]

    data, switch = open_as(operator_request(monkeypatch))
    assert data["readonly"] is False
    # By family applies as it reads: its value is not written (an answer that
    # ends after the operator changed it again must not set it back, UI-05).
    assert switch["interactive"] is True and "value" not in switch
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
    ident = top[top.index('<dl class="shep-kv shep-mp-ident">'):]
    ident = ident[:ident.index("</dl>") + 5]
    keys = re.findall(r'<dt class="shep-k[^"]*"[^>]*>([^<]*)', ident)
    # One summary list: the fingerprint, the file (the title names its run,
    # which two files of one run share: review of B3), the dataset by its
    # relation, what the model is, and the Trainer's figure as a row (the
    # ops-wiring B3).
    assert keys == ["Fingerprint", "File", "Dataset", "Kind", "Operation", "Parent", "Family",
                    "Encoder", "Relation to the paper", "Validation"]
    text = _text(ident)
    assert "Disease" in text and "GATv2" in text
    # The operation in the shared tree's words, as the run's own row in the
    # list beside it says it: one spelling on the screen (it read "Finetune"
    # beside the row's "Fine-tune, full", and never said full or frozen).
    disease_run = BUNDLES["disease"][0]
    for item in listing["items"]:
        if item.get("run") == disease_run:
            item["weight_transfer"] = {"trainable": "all"}
    said = mp.detail_parts(listing, data, bundle_key("disease"), ctl)["top"]
    operation = _text(re.search(r">Operation</dt>(.*?)</dd>", said, re.S).group(1)).strip()
    row = _cells(_run_row(ml.list_html(listing, None, ml.PRESETS["research"]),
                          f"gatv2/{disease_run}"))
    assert operation == row[2] == "Fine-tune, full", (operation, row)
    # The parent's run in the one run words, its name first (the review of
    # B0b: "run 2026-10-01 00:00" dropped it), one " · "; the family, the
    # same run, by its relation to the Parent row (rule 1.2: the browser
    # check of B3 found the run said in both rows).
    # G03 (the stage-lineage round): the exact parent file by the words of its
    # Run → Checkpoint pair, its 12 hex with the whole fingerprint one Copy away.
    pair = ml.pair_words({ml.item_key(i): i for i in listing["items"]})[bundle_key("base")]
    assert pair[1] in text and f"of run {pair[0]}" in text, (pair, text)
    assert f"{shas['base'][:12]} Copy · listed here" in text
    assert f'data-copy="{shas["base"]}"' in said
    assert f"Family{mp.PARENT_FAMILY}Encoder" in text
    assert text.count(pair[0]) == 1
    # A shrunk test spec: each field that differs in words, its record path
    # its title (skill §5; the review of B0c named them by their paths).
    assert "differs in: " in text and "Encoder › Hidden" in text
    assert '<span title="encoder.hidden">Encoder › Hidden</span>' in ident
    assert "Configuration" not in top and ">Task<" not in top
    specless = mp.detail_parts(listing, data, _key(world, "model-05.pt"), ctl)["top"]
    ident = specless[specless.index('<dl class="shep-kv shep-mp-ident">'):]
    ident = ident[:ident.index("</dl>") + 5]
    assert re.findall(r'<dt class="shep-k[^"]*"[^>]*>([^<]*)', ident) == [
        "Fingerprint", "File", "Dataset", "Kind", "Validation"]
    assert ">model-05.pt<" in ident
    assert "Record not readable" in ident
    # Axis 2 for a reference spec, and for a parent elsewhere or unavailable.
    item = {"record": "model", "learning_stage": "finetune", "task": "gene",
            "operation": "finetune", "family_id": BASE_RUN, "encoder": "gatv2",
            "differences": (), "parent_sha256": "a" * 64, "parent_run": EVIL}
    html = mp.identity_html(item, {"parent_state": "elsewhere", "parent_workspace": EVIL})
    assert "matches the reference recipe" in html and mp.REFERENCE_ANCHOR in _text(html)
    assert EVIL not in html and "in workspace <b>&lt;img" in html
    html = mp.identity_html(item, {"parent_state": "unavailable", "parent_problem": EVIL})
    assert '▲</span><span class="shep-st-sp">\u00a0</span><span class="shep-st-w">parent unavailable: &lt;img' in html
    assert EVIL not in html and "Parent unavailable" not in html     # short: whole in its row
    # A reason the Parent row shortens is whole one "Details ›" away (the second
    # round's review: the first round's run panel was its only whole copy).
    long_reason = "the parent file " + EVIL + " was not found in any listed workspace; " * 4
    html = mp.identity_html(item, {"parent_state": "unavailable", "parent_problem": long_reason})
    more = html[html.index('<details class="shep-more"'):]
    whole = " ".join(long_reason.split())
    assert "…" in html[:html.index(more)] and e(whole) not in html[:html.index(more)]
    assert "Parent unavailable" in more and e(whole) in more and EVIL not in html
    assert "none: a new run" in mp.identity_html({**item, "parent_sha256": None}, None)
    # The parent's digest is the one digest value: its 12 hex, the whole value
    # its title and one Copy away (G03; the review of B0c: a bare code without it).
    assert (f'<div>{r.fingerprint_value("a" * 64, what="parent fingerprint")}</div>'
            in mp.identity_html(item, None)) and f'data-copy="{"a" * 64}"' in r.fingerprint_value(
                "a" * 64, what="parent fingerprint")
    # A run's name past an option's 40 characters is named whole in the
    # details (the review of B0c: the Parent and Family rows took the
    # option's middle cut, "heads-abla…ne-control", with no way to read it
    # whole), as it was before B0c for a directory not named by its time; the
    # By family titles keep the option's short form, the Family row says it.
    named = "20261024-173909_heads-ablation-relation-aware-frozen-backbone-control_c1d9cf"
    legacy = "legacy-experiment-relation-aware-heads-ablation-v2-final"
    for run, words in ((named, "heads-ablation-relation-aware-frozen-backbone-control · "
                               "2026-10-24 17:39"), (legacy, legacy)):
        long = mp.identity_html({**item, "family_id": run, "parent_run": run},
                                {"parent_state": "listed"})
        assert f"<div>run {words}</div>" in long and "· </span>listed here</div></dd>" in long, long
        assert f'Family</dt><dd class="shep-v">{r.e(mp.PARENT_FAMILY)}</dd>' in long, long
        assert "…" not in long
        # A family root other than the parent's run is named whole.
        long = mp.identity_html({**item, "family_id": run, "parent_run": BASE_RUN},
                                {"parent_state": "listed"})
        assert f'Family</dt><dd class="shep-v">run {words}</dd>' in long, long
        assert "…" not in long
        assert len(r.run_words(None, run)) <= r.OPTION_CHARS and "…" in r.run_words(None, run)
    # The frozen-backbone control is a task model with a frozen graph
    # backbone (ops-wiring round, decision F1), from its record's transfer.
    frozen = {**item, "task": "disease", "weight_transfer": {
        "trainable": ["patient_encoder", "task_head"],
        "parts": {"input": "frozen", "encoder": "frozen", "lp_head": "frozen",
                  "patient_encoder": "trainable", "task_head": "trainable"}}}
    assert ('<dt class="shep-k">Kind</dt><dd class="shep-v">Disease · frozen graph '
            'backbone</dd>') in mp.identity_html(frozen, None)
    assert '<dd class="shep-v">Gene</dd>' in mp.identity_html(item, None)
    # A branch says what it changed from its parent, or that it changed
    # nothing (decision B4): every value in the details, the names in By
    # family's Lineage; text from a record is escaped.
    branch = {**item, "operation": "branch", "parent_changes": {}}
    html = mp.identity_html(branch, None)
    keys = re.findall(r'<dt class="shep-k[^"]*"[^>]*>([^<]*)', html)
    assert keys == ["Fingerprint", "File", "Dataset", "Kind", "Operation", "Parent",
                    "Settings changed", "Family", "Encoder", "Relation to the paper",
                    "Validation",
                    # the run directory, the whole fingerprint and the parent's one
                    # Details › away (G3: the exact parent's digest read whole)
                    "Run directory", "Whole fingerprint", "Parent fingerprint"]
    assert mp.SAME_SETTINGS in _text(html)
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
    settings = _text(html[html.index(">Settings changed<"):])
    settings = settings[:settings.index("Family")]
    assert "'" not in settings and "None" not in _text(html)
    assert "Settings changed" not in mp.identity_html(item, None)
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
    models = dict(_rows(mp.records_block_html([view], {})))["Coverage"]
    test = dict(_rows(tp._record_html(record, {}, {})))["Coverage"]
    assert models == test == ("8 of 10 import-eligible cases ranked by the model 13 source "
                              "cases, 3 excluded at import: ground truth absent 3")


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
        tiles = re.findall(r'shep-rec-tile-l">([^<]*)<', html)
        # R-71: the task is named once, by the group's label, not by each tile.
        assert tiles[0] == "Pooled MRR" and tiles[-1] == "Coverage", tiles
        assert second.split(" ", 1)[1].capitalize() in tiles or task == "gene", tiles
        assert "0.4123" in html and value in html and "not described" not in html
        # Its other figures as rows, in E1/E5/E2's order, under the task's name
        # said once (by its label and tiles).
        keys = [key for key, _value in _rows(html)]
        order = [keys.index(words) for words in (
            "Coverage", "Conservative pooled MRR", "Macro precision@k", "Macro MRR")]
        assert order == sorted(order), keys
        assert "import-eligible cases ranked by the model" in _text(html)
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
    assert mp.READONLY_TITLE in parts["header"] and "no checkpoint is opened" in parts["header"]
    assert sha in parts["top"] and parts["seg"] is None
    records = parts["records"]
    assert "registered #3" in records and "runs differ #1" in records
    # each record's determinism, from the environment class the route answers,
    # a row of its own (the one record rendering, R-82)
    determinism = [dict(_rows(section))["Deterministic algorithms"]
                   for section in _sections(records)]
    assert determinism[:2] == ["on", "off"]
    assert ("runs under these conditions differ: " in records and "investigate" not in records
            and mp.NO_LOAD_HERE in records)
    assert str(world["root"]) not in records and str(world["registry"]) not in records
    header = mp.left_header_html(listing, data, "deployment_mode")
    assert e(mp.D8_CONDITIONS) in header and "shep-mp-ro" in header
    assert "Nothing was done" not in header and "Refused" not in header      # said once (UI-6)
    for condition in ("deployment_mode", "not_loopback", "forwarding_header"):
        _says_the_refusal_as_every_page_does(mp.left_header_html(listing, data, condition),
                                             condition)
    ctl = mp.controls(data, mp.BY_FAMILY, None)
    assert ctl["order"] == mp.NEWEST and mp.shape(listing, data, ctl) == {"note": ""}


def _says_the_refusal_as_every_page_does(header, condition):
    """The read-only list's reason: after the lock, one lead in plain words
    (why there is no list, who may list it), then D8's conditions and the
    codes under "Details ›" (the message shape; the WebUI skill's §5). This
    request's own condition ("Operator controls only: forwarded by a proxy.")
    is not said here: the gated Load says it, and the header chip (the
    independent check of the UI sweep, C3: it was said three times on one
    screen). At the T08 bare start it read "Listing opens checkpoint files,
    which SPEC_4 classes as checkpoint inspection (C3), …"
    (`scratchpad/t08pre/shots/s7-xff-07-models.png`); at the sweep, a 4-line
    paragraph and a Ref line of four codes in the flow."""
    from src.webui.components.chips import LOCK, operator_only

    words = _visible_words(header)
    lead = f"{LOCK} {mp.LIST_NOT_SHOWN}"
    assert words.startswith("Checkpoints ") and lead in words, words
    assert operator_only(condition) not in words, words
    plain, more = header.split('<details class="shep-more">', 1)
    assert not re.search(r"\bD8\b|SPEC_4|\bC3\b|\bM1\b|SHEPHERD_", _visible_words(plain))
    assert e(mp.D8_CONDITIONS) in more and ">Ref<" in more
    ref = _visible_words(more).split("Ref", 1)[1]
    assert ref.strip().startswith("D8") and "SPEC_4 C3" in ref and ";" not in ref


def test_the_list_panel_is_titled_by_its_content_and_count(world, monkeypatch):
    """R-30 (Models' half): the first panel was titled "Models" under the
    selected "Models" tab; it is titled by what it holds, the saved files the
    workspace's listing holds and the runs they are in ("Checkpoints (8 in 5
    runs)"), whatever the list shows of them (the count line under it says
    that: a filter, a search); "Checkpoints" alone before a list is read and
    for a read-only request, whose list is not read."""
    listing, data = _read(world, monkeypatch)
    files = [i for i in listing["items"] if r.entry_place(i) is None]
    runs = {ml.item_run_key(i) for i in files}
    assert (len(files), len(runs)) == (8, 5)
    header = mp.left_header_html(listing, data)
    assert "<h2>Checkpoints (8 in 5 runs)</h2>" in header and ">Models<" not in header
    assert "<h2>Checkpoints</h2>" in mp.left_header_html({}, {})
    one = {**listing, "items": [files[0]]}
    assert "<h2>Checkpoints (1 in 1 run)</h2>" in mp.left_header_html(one, data)
    refused = {"refused": True, "items": [], "condition": "not_loopback"}
    assert "<h2>Checkpoints</h2>" in mp.left_header_html(refused, {}, "not_loopback")


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
    # The list's rows are drawn by its one draw, which the opening asks for
    # (its draw token among the opening's outputs), with a progress cue there.
    (draw,) = [fn for fn in demo.fns.values() if lst in fn.outputs]
    assert draw.name == "_draw" and by_id["shep-mp-records"] in opened.outputs
    token = next(o for o in opened.outputs if (o._id, "change") in draw.targets)
    assert isinstance(token, gr.State) and lst in opened.show_progress_on
    states = [o for o in opened.outputs if isinstance(o, gr.State)]
    assert len(states) >= 4                    # listing, selected, page data, view, stale
    inputs = [getattr(i, "elem_id", None) for i in opened.inputs]
    assert {"shep-mp-order", "shep-mp-task"} <= set(inputs)
    assert not {"shep-mp-version", "shep-mp-metric", "shep-mp-scoring"} & set(inputs)
    stale = _sync_states(demo)["stale"]
    assert stale in opened.outputs
    out = opened.fn("demo", False, None, None, None, "", {}, mp.BY_FAMILY, mp.ALL_TASKS,
                    mp.RECORDS, "idle", operator_request(monkeypatch))
    assert len(out) == len(opened.outputs)
    by_output = dict(zip(map(id, opened.outputs), out, strict=True))
    listing = out[0]
    data = next(v for v in out if isinstance(v, dict) and "listing_key" in v)
    assert data["listing_key"] == mp.listing_key(listing)
    views = [v for v in out if isinstance(v, dict) and "groups" in v]
    assert views and views[0]["groups"] and views[0]["shaped_for"] == "family|all"
    assert by_output[id(by_id["shep-mp-order"])]["interactive"] is True
    assert by_output[id(stale)] is False                   # read now: nothing is stale
    assert by_output[id(by_id["shep-mp-wsline"])] == mp.ml.workspace_line_html(listing)
    # Shown while a Load runs: read now, and again once the Load has finished.
    out = opened.fn("demo", False, None, None, None, "", {}, mp.NEWEST, None, mp.RECORDS,
                    "loading", operator_request(monkeypatch))
    assert dict(zip(map(id, opened.outputs), out, strict=True))[id(stale)] == gr.update()


def test_the_workspace_line_shows_the_dataset_identity_under_the_select(world, monkeypatch):
    """A long workspace name is cut in the select; the dataset fingerprint and
    "synthetic" are said again under it, by the shared renderer."""
    from src.webui.components.renderers import dataset_identity_html

    listing, _data = _read(world, monkeypatch)
    row = next(w for w in listing["workspaces"] if w["name"] == "demo")
    # The shared list's one line (B4: Diagnosis' drawer says it too).
    html = mp.ml.workspace_line_html(listing)
    assert html == (f'<div class="shep-ws-line">'
                    f'{dataset_identity_html(row, name=False)}</div>')
    assert row["dataset_fingerprint"][:12] in html and "synthetic" in html
    assert "shep-ds-name" not in html                          # the select names it
    assert mp.ml.workspace_line_html({**listing, "workspace": "gone"}) == ""
    assert mp.ml.workspace_line_html({**listing, "refused": True}) == ""
    assert not hasattr(mp, "workspace_line_html")


def _sync_states(demo):
    """The one listener's States, by their place among its inputs (the list's
    clock last, `_sync`'s last argument)."""
    sync = _fn(demo, "_sync")
    listing, data, stale, last, phase = sync.inputs[:5]
    return {"fn": sync, "listing": listing, "data": data, "stale": stale, "last": last,
            "phase": phase, "selected": sync.inputs[-5], "view": sync.inputs[-3],
            "search": sync.inputs[-2], "clock": sync.inputs[-1]}


#: `_sync`'s `current` when a test does not name it: the list already holds the
#: view of the controls the call is given.
HELD = object()


def _sync(states, monkeypatch, listing, data, *, current=HELD, stale=False, last=None,
          phase="idle", order=mp.NEWEST, task=mp.ALL_TASKS, selected=None, seg=mp.RECORDS,
          search="", clock=None):
    sync = states["fn"]
    if current is HELD:
        current = mp.shape(listing, data, mp.controls(data, order, task))
    out = sync.fn(listing, data, stale, last, phase, order, task, selected, seg, current,
                  search, operator_request(monkeypatch), clock)
    assert len(out) == len(sync.outputs)
    by = dict(zip(map(id, sync.outputs), out, strict=True))
    return {name: by[id(states[name])] for name in ("stale", "last", "selected", "data")} | {
        "page": out[3:], "by": by}


def test_a_listing_already_read_is_not_read_again(world, monkeypatch):
    demo, _ctx = _page()
    states = _sync_states(demo)
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    detail = [by_id[i] for i in ("shep-mp-detail-hd", "shep-mp-detail-top", "shep-mp-seg",
                                 "shep-mp-records", "shep-mp-usage", "shep-rs")]
    listing, data = _read(world, monkeypatch)
    monkeypatch.setattr(mp, "read_page", lambda listing: pytest.fail("read again"))
    key = _key(world, "model-05.pt")
    # A selection, the list holding the controls' view: the detail only (and
    # the line under the options, as it reads), nothing read; the selection is
    # remembered.
    out = _sync(states, monkeypatch, listing, data, order=mp.NEWEST, selected=key)
    assert out["data"] == gr.update() and out["selected"] == gr.update()
    assert out["stale"] == gr.update() and out["last"] == {"workspace": "demo", "key": key}
    others = [o for o in states["fn"].outputs[3:] if o not in detail]
    assert out["by"][id(by_id["shep-mp-options"])] == ""
    assert all(out["by"][id(o)] == gr.update() for o in others
               if o is not by_id["shep-mp-options"])
    assert world["shas"]["model-05.pt"] in out["by"][id(by_id["shep-mp-detail-top"])]
    # Order reads By family while the list holds Newest first's view: the view
    # shaped again from the data held, nothing read. Whichever trigger asked:
    # the call carries none (Gradio 6.30 runs a deferred call again with the
    # first trigger's id, UI-05).
    newest = mp.shape(listing, data, mp.controls(data, mp.NEWEST, mp.ALL_TASKS))
    out = _sync(states, monkeypatch, listing, data, current=newest, order=mp.BY_FAMILY,
                selected=key)
    view = out["page"][1]
    assert out["data"] == gr.update() and "groups" in view
    assert view["shaped_for"] == "family|all"
    # The switch's value is not written back: it reads By family already.
    assert "value" not in out["by"][id(by_id["shep-mp-order"])]
    assert out["by"][id(by_id["shep-mp-task"])] == gr.update()
    # A listing not read yet: read once.
    changed = {**listing, "served_sha": world["shas"]["model-05.pt"]}
    calls = []
    monkeypatch.setattr(mp, "read_page", lambda listing: calls.append(1) or data)
    _sync(states, monkeypatch, changed, data, order=mp.NEWEST)
    assert calls == [1]


def test_an_answer_writes_a_control_back_only_where_it_cannot_apply(world, monkeypatch):
    """UI-05's fast switch: an answer that ends after the operator changed a
    control again must not set it back (its next call reads the value set
    back and the operator's change is lost). A value is written only where it
    cannot apply as it reads: an unknown task (All), By family for a
    read-only request (Newest first)."""
    demo, _ctx = _page()
    states = _sync_states(demo)
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    listing, data = _read(world, monkeypatch)
    out = _sync(states, monkeypatch, listing, data, current=None, order=mp.BY_FAMILY,
                task="disease")
    order, task = out["by"][id(by_id["shep-mp-order"])], out["by"][id(by_id["shep-mp-task"])]
    assert "value" not in order and order["interactive"] is True and task == gr.update()
    assert out["page"][1]["shaped_for"] == "family|disease"
    out = _sync(states, monkeypatch, listing, data, current=None, task="gatv2")
    assert out["by"][id(by_id["shep-mp-task"])]["value"] == mp.ALL_TASKS
    _served_read_only(world, monkeypatch)
    refused = ml.list_models("demo", gradio_request(), family=True)
    sync = states["fn"]
    answer = sync.fn(refused, None, False, None, "idle", mp.BY_FAMILY, mp.ALL_TASKS, None,
                     mp.RECORDS, None, "", gradio_request())
    order = dict(zip(map(id, sync.outputs), answer, strict=True))[id(by_id["shep-mp-order"])]
    assert (order["value"], order["interactive"]) == (mp.NEWEST, False)


def test_an_answer_that_fails_says_what_was_not_applied_and_changes_nothing(world, monkeypatch):
    """D05: a failure says why, under the options, for the controls it was
    asked with (the "Applying …" line compares ``data-failed-for``); the list,
    the selection and the detail keep what they show. Its words name the
    error's kind, never its message (a path can be in it). Gradio drops a
    deferred call after a failed one, so the failure never escapes."""
    demo, _ctx = _page()
    states = _sync_states(demo)
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    listing, data = _read(world, monkeypatch)

    def broken(*_args, **_kwargs):
        raise OSError("/srv/secret/path cannot be read")

    monkeypatch.setattr(mp, "shape", broken)
    out = _sync(states, monkeypatch, listing, data, current=None, order=mp.BY_FAMILY)
    line = out["by"][id(by_id["shep-mp-options"])]
    assert 'data-failed-for="family|all"' in line and "shep-st-error" in line
    assert _words(line) == ("✕ Not applied: By family. The page could not shape the list "
                            "(OSError), so it still shows what it showed before.")
    assert "/srv" not in line
    assert all(value == gr.update() for o, value in zip(states["fn"].outputs, out["by"].values(),
                                                         strict=True)
               if o is not by_id["shep-mp-options"])
    assert mp.controls_words({"order": mp.NEWEST, "task": "disease"}) == \
        "Newest first, Task: Disease"


def test_the_rows_say_the_view_they_were_drawn_for_and_another_workspace_drops_it(
        world, monkeypatch):
    """The shared list's rows carry the view's ``shaped_for`` (escaped), an
    empty list's too; another workspace's listing is drawn with the view's
    filter only and not as shaped for anything, so the page's "Applying …"
    line shows until the page shapes the view for it. By family's view names
    the workspace its keys were read from, so the list's one draw tells
    (`model_list.list_drawing`)."""
    listing, data = _read(world, monkeypatch)
    family = mp.shape(listing, data, mp.controls(data, mp.BY_FAMILY, mp.ALL_TASKS))
    assert family["shaped_for"] == "family|all" and "keys" in family
    assert family["workspace"] == listing["workspace"] == "demo"
    elsewhere = {**listing, "workspace": "elsewhere"}
    other = ml.view_for_listing(family, listing, elsewhere)
    assert "shaped_for" not in other and "keys" not in other and other["note"] == mp.LOAD_NOTE
    # The view's own workspace, whatever the listing before: this one's keys apply.
    assert ml.view_for_listing(family, elsewhere, listing) is family
    assert ml.view_for_listing(family, listing, listing) is family
    drawn = ml.list_drawing(elsewhere, None, ml.PRESETS["research"], view=family)
    assert "shep-view-for" not in drawn.rows and "shep-fam" not in drawn.rows
    newest = mp.shape(listing, data, mp.controls(data, mp.NEWEST, "gene"))
    assert ml.view_for_listing(newest, {"workspace": "elsewhere"}, listing) is newest
    assert ml.view_marker({"shaped_for": '"><img src=x onerror=alert(1)>'}) == (
        '<span class="shep-view-for" hidden data-for="&quot;&gt;&lt;img src=x '
        'onerror=alert(1)&gt;"></span>')
    assert ml.view_marker(None) == "" and ml.view_marker({"task": "all"}) == ""
    empty = ml.list_html({**listing, "items": []}, None, ml.PRESETS["research"], view=newest)
    assert empty.endswith(ml.view_marker(newest))
    refused = ml.list_html({**listing, "refused": True}, None, ml.PRESETS["research"],
                           view=newest)
    assert "shep-view-for" not in refused


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
    out = _sync(states, monkeypatch, after, data, stale=True,
                last=last, phase="loading")
    assert reads == [] and out["stale"] == gr.update() and out["selected"] == gr.update()
    # The Load has finished (the refresh cleared the selection): read once, the
    # same checkpoint selected again, its usage history with the new entry.
    out = _sync(states, monkeypatch, after, data, stale=True,
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
    again = _sync(states, monkeypatch, after, out["data"], current=out["page"][1],
                  last=last, selected=key)
    assert reads == [1] and again["data"] == gr.update() and again["selected"] == gr.update()
    # A checkpoint the view does not show is selected again too (D04: a filter
    # never changes what is selected); the list says it is outside the filter.
    out = _sync(states, monkeypatch, after, data, stale=True,
                last=last, task="gene")
    assert reads == [1, 1] and out["selected"] == key and "redraw" in out["page"][1]
    out = _sync(states, monkeypatch, after, data, stale=True,
                last=last, order=mp.BY_FAMILY)
    assert out["selected"] == key and "groups" in out["page"][1]


def test_after_a_load_a_pick_made_meanwhile_is_never_set_back(world, monkeypatch):
    """The review of dcaed68 (SL-R3): the page selects the last checkpoint
    again after a Load through the list's clock (`model_list.ListClock`), only
    while nothing is selected there and the listing is the one committed now:
    a pick made while the Load's re-read was answered stays, and a listing of
    a generation the list no longer shows selects nothing again."""
    demo, _ctx = _page()
    states = _sync_states(demo)
    assert states["clock"] is next(o for o in _fn(demo, "_select").inputs
                                   if isinstance(getattr(o, "value", None), ml.ListClock))
    listing, data = _read(world, monkeypatch)
    last_key, newer = _key(world, "model-05.pt"), _key(world, "model-12.pt")
    last = {"workspace": "demo", "key": last_key}
    clock = ml.ListClock.seeded(listing)
    clock.selected = newer                       # picked meanwhile; its change not yet in
    out = _sync(states, monkeypatch, listing, data, stale=True, last=last, clock=clock)
    assert out["selected"] == gr.update() and clock.selected == newer
    clock = ml.ListClock.seeded({**listing, ml.GENERATION: 2})
    out = _sync(states, monkeypatch, {**listing, ml.GENERATION: 1}, data, stale=True,
                last=last, clock=clock)
    assert out["selected"] == gr.update() and clock.selected is None
    # Nothing picked, the listing the one committed: selected again, as before.
    clock = ml.ListClock.seeded(listing)
    out = _sync(states, monkeypatch, listing, data, stale=True, last=last, clock=clock)
    assert out["selected"] == last_key and clock.selected == last_key


def test_controls_selection_and_segments_are_wired(world, monkeypatch):
    demo, _ctx = _page()
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    states = _sync_states(demo)
    sync = states["fn"]
    # One listener answers a new listing, a selection, the controls and the
    # list's search (By family's "Saved files" offers what the list draws), so
    # its answers for one session never overlap; the last trigger always runs.
    # A pick of the Task select is one request (`shell.picked`: its blur), the
    # Order radio's its input; the view the list holds is read with them. The
    # search is followed on the list's search token, never on the box's change
    # (the list's search is its one listener: two "always_last" listeners of
    # one event set each other off without end in Gradio 6.30).
    box = by_id["shep-models-research-search"]
    searched = next(fn for fn in demo.fns.values() if (box._id, "change") in fn.targets)
    assert {(demo.blocks[t[0]], t[1]) for t in sync.targets} == {
        (states["listing"], "change"), (states["selected"], "change"),
        (searched.outputs[1], "change"),
        (by_id["shep-mp-order"], "input"), (by_id["shep-mp-task"], "blur")}
    assert [fn.name for fn in demo.fns.values() if (box._id, "change") in fn.targets] == [
        "_searched"]
    assert states["view"] in sync.inputs
    assert states["search"] is by_id["shep-models-research-search"]
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


#: Research › Models in a browser, the page's queue watched (`QUEUE_JS`): each
#: case holds an answer of the page's one listener (`_sync`, its join held by
#: ``page.route`` until the case lets it go), changes a control meanwhile, and
#: reads the page once the rows say they were drawn for what the case asked
#: (`model_list.view_marker`), a drawing the page shows only after the answers
#: it waits for. Then a Task pick by pointer and one by keys, nothing else
#: running: the listener's requests each sent (all sent before its first
#: answer is drawn).
_MODELS_JS = r"""
const { chromium } = require('playwright');
const QUEUE_JS = __QUEUE_JS__;
const TASKS = { All: 'all', Base: 'base', Disease: 'disease', Gene: 'gene' };
(async () => {
  const base = process.argv[2];
  const browser = await chromium.launch();
  const out = {};
  // The rows drawn for `token` ("<order>|<task>"): true once they are, false
  // when they are not within `ms` (the case then reads what the page shows).
  const drawn = (page, token, ms = 20000) => page.waitForFunction((t) => {
    const m = document.querySelector('#shep-models-research-rows .shep-view-for');
    return !!m && m.getAttribute('data-for') === t;
  }, token, { timeout: ms }).then(() => true, () => false);
  const open = async () => {
    const ctx = await browser.newContext({ viewport: { width: 1850, height: 960 }, deviceScaleFactor: 1 });
    const page = await ctx.newPage();
    await page.addInitScript(QUEUE_JS);
    await page.goto(base, { waitUntil: 'domcontentloaded' });
    await page.click('#shep-tab-research-button', { timeout: 60000 });
    await page.click('#shep-page-models-button');
    await page.waitForSelector('#shep-mp-order input[value="family"]:not([disabled])', { timeout: 60000 });
    await page.waitForSelector('#shep-models-research-rows table');
    // A page that draws no marker (before UI-05) is read once its queue is idle.
    await drawn(page, 'newest|all', (await page.$('#shep-mp-pending')) ? 30000 : 2000);
    await page.waitForFunction(() => window.__shepQueue.inflight === 0);
    const cfg = await page.evaluate(async (b) => (await fetch(b + 'config')).json(), base);
    const sync = cfg.dependencies.find((d) => d.api_name === '_sync');
    return { ctx, page, ids: { sync: sync.id, selected: sync.targets[1][0] } };
  };
  const read = (page) => page.evaluate(() => {
    const radio = document.querySelector('#shep-mp-order input[type=radio]:checked');
    const pending = document.querySelector('#shep-mp-pending .shep-mp-pending.shep-on .shep-st-w');
    return {
      order: radio ? radio.value : null,
      task: document.querySelector('#shep-mp-task input').value,
      count: document.querySelector('#shep-models-research-count').innerText.trim(),
      lineage: !!document.querySelector('#shep-models-research-rows .shep-fam ul.shep-fam-tree'),
      groups: [...document.querySelectorAll('#shep-models-research-rows ul.shep-fam-tree')].map((t) => t.getAttribute('aria-label')),
      pending: pending ? pending.innerText.trim() : '',
    };
  });
  // Holds the joins `preds` name, the first of each in turn, until each one's
  // `release()`; `held` once it is held.
  const hold = async (page, ...preds) => {
    const gates = preds.map(() => { const g = {}; g.held = new Promise((r) => { g.mark = r; }); g.open = new Promise((r) => { g.release = r; }); return g; });
    let next = 0;
    await page.route('**/queue/join*', async (route) => {
      let body = {};
      try { body = route.request().postDataJSON() || {}; } catch (e) { body = {}; }
      if (next < preds.length && preds[next](body)) {
        const g = gates[next++];
        g.mark();
        await g.open;
      }
      await route.continue();
    });
    return gates;
  };
  const pendingSays = (page, words) => page.waitForFunction((w) => {
    const el = document.querySelector('#shep-mp-pending .shep-mp-pending.shep-on .shep-st-w');
    return !!el && el.innerText.trim() === w;
  }, words, { timeout: 10000 }).then(() => words, () => null);
  const pickTask = async (page, label) => {
    await page.click('#shep-mp-task input');
    await page.click(`ul[role=listbox] li:has-text("${label}")`);
  };
  const joins = (page, ids) => page.evaluate((id) => window.__shepQueue.count(id), ids.sync);

  // A: a row chosen, its answer held; By family meanwhile.
  {
    const { ctx, page, ids } = await open();
    const [h] = await hold(page, (b) => b.fn_index === ids.sync && b.trigger_id === ids.selected);
    await page.click('#shep-models-research-rows tbody tr[data-key]:not([hidden]) >> nth=1');
    await h.held;
    await page.click('#shep-mp-order label:has-text("By family")');
    const during = await pendingSays(page, 'Applying By family…');
    h.release();
    const settled = await drawn(page, 'family|all');
    out.selection_then_family = { during, settled, after: await read(page) };
    await ctx.close();
  }
  // B: By family, its answer held; Newest first at once. The second answer
  // is held too, until the first one's rows are drawn.
  {
    const { ctx, page, ids } = await open();
    const [first, second] = await hold(page, (b) => b.fn_index === ids.sync && b.data[5] === 'family',
                                        (b) => b.fn_index === ids.sync);
    await page.click('#shep-mp-order label:has-text("By family")');
    await first.held;
    await page.click('#shep-mp-order label:has-text("Newest first")');
    first.release();
    await second.held;
    await drawn(page, 'family|all', 30000);
    const between = await read(page);
    second.release();
    const settled = await drawn(page, 'newest|all');
    out.family_then_newest = { between, settled, after: await read(page) };
    await ctx.close();
  }
  // C: a row chosen, its answer held; the Task select's Disease meanwhile.
  {
    const { ctx, page, ids } = await open();
    const [h] = await hold(page, (b) => b.fn_index === ids.sync && b.trigger_id === ids.selected);
    await page.click('#shep-models-research-rows tbody tr[data-key]:not([hidden]) >> nth=1');
    await h.held;
    await pickTask(page, 'Disease');
    const during = await pendingSays(page, 'Applying Task: Disease…');
    h.release();
    const settled = await drawn(page, 'newest|disease');
    out.selection_then_task = { during, settled, after: await read(page) };
    await ctx.close();
  }
  // D: one pick of the Task select by pointer, then one by keys.
  {
    const { ctx, page, ids } = await open();
    let before = await joins(page, ids);
    await pickTask(page, 'Base');
    const pointerSettled = await drawn(page, 'newest|base');
    const pointer = (await joins(page, ids)) - before;
    before = await joins(page, ids);
    await page.click('#shep-mp-task input');
    await page.keyboard.press('ArrowDown');
    await page.keyboard.press('Enter');
    const keysSettled = await page.waitForFunction((tasks) => {
      const v = document.querySelector('#shep-mp-task input').value;
      const m = document.querySelector('#shep-models-research-rows .shep-view-for');
      return v !== 'Base' && v in tasks && !!m && m.getAttribute('data-for') === 'newest|' + tasks[v];
    }, TASKS, { timeout: 20000 }).then(() => true, () => false);
    const keys = (await joins(page, ids)) - before;
    out.task_picks = { pointer, keys, settled: pointerSettled && keysSettled,
                       after_keys: (await read(page)).task };
    await ctx.close();
  }
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_a_control_changed_while_the_page_answers_is_applied_and_one_pick_is_one_request(
        tmp_path, world, monkeypatch):
    """UI-05 and UI-11 on the page as the dashboard mounts it. On ba6e85e, By
    family chosen while the listener answered a row's selection was dropped
    (the switch read By family, the count line "newest first", no family
    title, no Lineage: Gradio ran the deferred call again with the first
    trigger's id, and the page judged by who fired); a fast By family then
    Newest first ended By family (the late answer set the switch back); one
    Task pick sent two requests. Now every case ends with the switch, the
    count line, the groups and Lineage in agreement with the operator's last
    change, "Applying …" said while it was pending, and one request per
    pick, by pointer and by keys."""
    from src.webui.app import create_gradio_app, mount_options
    from tests.fixtures.browser import QUEUE_JS, node_with_playwright, run_node
    from tests.fixtures.served_page import served

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    operator_request(monkeypatch)
    _add_bundles(world)
    with served(create_gradio_app(), **mount_options()) as base:
        seen = run_node(_MODELS_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS)), tmp_path,
                        base[:-len("gradio_api")])

    family = seen["selection_then_family"]
    after = family["after"]
    assert after["order"] == mp.BY_FAMILY and after["count"].startswith("1 family, "), family
    assert after["lineage"] is True and after["pending"] == ""
    assert any(g.startswith("Family from base") for g in after["groups"]), after  # the graph
    assert family["during"] == "Applying By family…" and family["settled"], family
    newest = seen["family_then_newest"]
    # While the second answer waits: the switch reads Newest first (the first
    # answer did not set it back), the rows By family's, and the line says so.
    between, after = newest["between"], newest["after"]
    assert after["order"] == mp.NEWEST and "famil" not in after["count"], newest
    assert between["order"] == mp.NEWEST and between["lineage"] is True, between
    assert between["pending"] == "Applying Newest first…", between
    assert after["lineage"] is False and after["groups"] == [] and after["pending"] == ""
    assert newest["settled"], newest
    task = seen["selection_then_task"]
    assert task["after"]["task"] == "Disease" and "Disease" in task["after"]["count"], task
    assert task["after"]["pending"] == "" and task["settled"]
    assert task["during"] == "Applying Task: Disease…", task
    picks = seen["task_picks"]
    assert (picks["pointer"], picks["keys"]) == (1, 1), picks
    assert picks["settled"], picks
    assert picks["after_keys"] not in ("Base", ""), picks


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
                + ml.list_html(listing, None, ml.PRESETS["research"], view=view)
                + mp.ml.workspace_line_html(listing))
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


def test_the_list_panel_has_one_scroll_region_at_every_wide_height():
    """The skill's §2 (one scroll region per area; nothing sticky over a table
    row). M5-3 made the title sticky over the panel's own scroll region below
    1000 px tall (at 1440 x 900 it had scrolled away with the controls), and
    B1b gave the panel a scroll padding as tall as the title: the round's
    closing check still found the title over 3 to 6 rows at 1850 x 960 and
    1536 x 760. Now every wide height has the one structure the tall layout
    had: the header rows in flow (the header row draws no box, its items the
    panel's grid items), the rows' region the panel's last grid row and the
    one thing that scrolls (its table's header sticking to its own top),
    taking what the header leaves; no title sticks, so nothing measures one.
    The rows' floor of 160 px is gone (T04 review): below about 660 px of
    window it made the panel scroll around the rows' own region, two regions
    in one panel at 1366 x 620 and 1280 x 600 (the shared walk measures it)."""
    from tests.unit.test_webui_shell import side

    wide = side(mp.PAGE_CSS)
    # The header and the tools bar are boxes of their own in flow (a row of fields, a bar of
    # groups; the UI sweep of 2026-11-22: flattened into the panel's grid, a note in the
    # tools widened a track and squeezed the search box), the rows' region the last row.
    assert "display: contents" not in mp.PAGE_CSS
    assert "grid-template-rows: max-content max-content minmax(0, 1fr);" in wide
    assert "#shep-models-research-rows { overflow: auto !important; min-height: 0;" in wide
    assert "> .shep-model-main { height: 100%; overflow-y: auto; }" in wide
    assert "#shep-mp-hd { position: sticky" not in mp.PAGE_CSS
    assert "max-height: 999px" not in mp.PAGE_CSS and "min-height: 1000px" not in mp.PAGE_CSS
    assert "--shep-mp-hd-h" not in mp.PAGE_CSS and not hasattr(mp, "HEADER_ROOM_JS")
    from src.webui.shell import standalone_context

    with gr.Blocks() as demo:
        mp.create_models_page(standalone_context(demo))
    (title,) = [b for b in demo.blocks.values() if getattr(b, "elem_id", None) == "shep-mp-hd"]
    assert title.js_on_load == r.KEEP_OPEN_JS


def test_a_row_focused_under_the_tables_sticky_header_scrolls_clear_of_it():
    """B1b: the table's header is sticky at the top of the region its rows
    scroll in (then under the title, 760-999 px high; since the closing
    check, the rows' own region at every wide height); a scroll padding alone
    let Shift+Tab focus a row in the header's band without scrolling it
    (1536 x 760: rows focused at 165 px under a header from 165 to 225 px).
    The shared list script keeps the header's height on the component, 0
    where it does not stick, and the shared table's rows take it as their
    scroll margin, in place before a row takes focus (one set on focus came
    too late for the browser's scroll: the rows still landed at 165 px), so a
    focused row lands under both."""
    from src.webui.components import model_list as ml
    from src.webui.shell import standalone_context
    from src.webui.style import SHELL_CSS

    js = ml.MODEL_LIST_JS
    room = js[js.index("var headRoom = function () {"):]
    assert "element.querySelector('table > thead th')" in room
    assert "getComputedStyle(cell).position === 'sticky'" in room
    assert ("element.style.setProperty('--shep-tbl-head-h', stuck ? "
            "cell.closest('thead').offsetHeight + 'px' : '0px');") in room
    assert "new ResizeObserver(headRoom).observe(element);" in room and "focusin" not in js
    assert (".shep-tbl tbody tr[data-key] { cursor: pointer; "
            "scroll-margin-top: var(--shep-tbl-head-h, 0px); }") in SHELL_CSS
    with gr.Blocks() as demo:
        mp.create_models_page(standalone_context(demo))
    (rows,) = [b for b in demo.blocks.values() if getattr(b, "elem_id", None) == "shep-models-research-rows"]
    # The family graph's own script after it (the research preset only).
    assert rows.js_on_load == r.KEEP_OPEN_JS + js + ml.FAMILY_JS


def test_the_table_fits_its_frame_beside_a_classic_scrollbar():
    """M5-3: at 1440 x 900 with a classic (non-overlay) 15 px scrollbar the
    table (602 px: its columns' longest words and 10 px a side) ran 13 px past
    its 589 px frame; with 7 px a side it is 572 px (measured in a browser with
    the scrollbar forced classic). The rule is the shared tree's, in the shared
    style sheet, so the Diagnosis load drawer's tree is as narrow (it kept
    Gradio's 10 px there and ran 601 px in its 524 px list at 1280 x 900);
    two classes outrank Gradio's own ``.gradio-container-… th``."""
    from src.webui.style import SHELL_CSS
    from tests.unit.test_webui_shell import side

    rule = ".shep-tbl.shep-mtbl th, .shep-tbl.shep-mtbl td { padding-left: 7px; padding-right: 7px; }"
    assert rule in side(SHELL_CSS)
    assert ".shep-mtbl th" not in mp.PAGE_CSS


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
        # The Registration row: its time, then its run and its report each on
        # a line of their own (never a chain), its sequence its heading's chip.
        assert f'<div class="shep-sub">from run {record["run_id"]}</div>' in records
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
    assert ('<div class="shep-mp-delta"><span class="shep-st shep-st-warn"><span class="shep-st-shape" aria-hidden="true">▲</span><span class="shep-st-sp">\u00a0</span><span class="shep-st-w">runs '
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
        '<div class="shep-mp-delta"><span class="shep-st shep-st-warn"><span class="shep-st-shape" aria-hidden="true">▲</span><span class="shep-st-sp">\u00a0</span><span class="shep-st-w">runs under '
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
    assert ('<div class="shep-sub">from run not recorded</div><div class="shep-sub">report '
            '<code class="shep-fpv"') in html and 'title="' + "d" * 64 + '">dddddddddddd' in html
    assert f'<div class="shep-sub">from run {named["run_id"]}</div>' in html
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
    assert f'<div class="shep-sub">from run {named["run_id"]}</div>' in html
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


# --------------------------------------------------------------------------- one fact once (B3)
def _visible_words(markup):
    """A screen's visible words: a closed disclosure's body is not shown, so
    it is left out (its summary stays); every tag a space."""
    shown = re.sub(r'<details class="shep-more"(?![^>]* open)[^>]*>.*?</details>', " ", markup,
                   flags=re.S)
    return " ".join(_text(re.sub(r"<[^>]+>", " ", shown)).split())


def test_one_screen_says_each_fingerprint_and_the_dataset_once(world, monkeypatch):
    """The skill's §1.2 and §12 (the ops-wiring round's B3): counted in the
    visible text of one screen, the workspace line, the table and the
    selected row's detail. The table's Fingerprint column shows each model's
    12 hex digits only (never the dataset's, nor a second digest); the
    workspace's dataset fingerprint is said once, under its select; a row and
    its detail trained on that dataset say nothing more of it ("the selected
    workspace's current dataset"). The selected model's 12 hex digits are
    said twice, with a reason: the table compares every row by them, and the
    detail is the one place to copy them from (`renderers.fingerprint_value`
    with its Copy); its whole value is one "Details ›" away. Its run is its
    row's and its title's words (a list and the detail it opens), never its
    directory, which is in the detail's provenance."""
    from src.webui.style import COL_TWIN

    listing, data = _read(world, monkeypatch)
    key = _key(world, "model-05.pt")
    item = ml.find_item(listing, key)
    view = mp.shape(listing, data, NEWEST_CTL)
    table = ml.list_html(listing, key, ml.PRESETS["research"], view=view)
    detail = mp.detail_parts(listing, data, key, NEWEST_CTL)
    # The line of a file's dropped columns (`style.COL_TWIN`) is drawn only while its columns
    # are not (the narrow tree), so one screen shows the column or the line, never both.
    shown = re.sub(rf'<div class="{COL_TWIN}[^"]*">.*?</div>', "", table)
    screen = _visible_words(mp.ml.workspace_line_html(listing) + shown + detail["header"]
                            + detail["top"])
    sha = item["sha256"]
    dataset = next(w["dataset_fingerprint"] for w in listing["workspaces"]
                   if w["name"] == "demo")
    assert screen.count(sha[:12]) == 2, screen          # its row and its detail's Copy
    assert sha not in screen                             # the whole value one click away
    assert sha in detail["top"] and "Whole fingerprint" in detail["top"]
    digits = r.distinguishing_digits(w.get("dataset_fingerprint") for w in listing["workspaces"])
    assert screen.count(dataset[:digits]) == 1, screen   # the workspace line, once
    assert r.CURRENT_DATASET in screen and "trained on demo" not in screen
    assert item["run"] not in screen                     # the directory: provenance only
    words = r.run_words(item["architecture"], item["run"])
    row = re.search(rf'<tr [^>]*data-key="{re.escape(key)}".*?</tr>', table, re.S).group(0)
    run = _run_row(table, ml.item_run_key(item))
    # its run's row (its files under it say what each is, not the run again)
    # and the detail's title
    assert words not in _visible_words(row)
    assert _visible_words(run + row + detail["header"] + detail["top"]).count(words) == 2
    # The table's Fingerprint column: a model's 12 hex only, the header one word.
    assert ">Fingerprint</th>" in table and "Fingerprints:" not in table
    cell = re.findall(r"<td(?: class=\"[^\"]*\")?>(.*?)</td>", row, re.S)[-1]
    assert cell == r.fingerprint_value(sha, what="model fingerprint", copy=False,
                                       digits=r.distinguishing_digits(
                                           i.get("sha256") for i in listing["items"]))


def test_a_refresh_keeps_the_selection_and_its_detail(world, monkeypatch):
    """D04: the page read again when shown (a refresh) keeps a selection still
    listed, and draws its detail, never the empty one beside a list that
    still marks it (seen on the multi-family demo: the row stayed selected and
    the detail said "Details")."""
    demo, _ctx = _page()
    opened = _fn(demo, "_open")
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    request = operator_request(monkeypatch)
    out = opened.fn("demo", False, None, None, None, "", {}, mp.BY_FAMILY, mp.ALL_TASKS,
                    mp.RECORDS, "idle", request)
    listing, key = out[0], _key(world, "model-05.pt")
    out = opened.fn("demo", False, None, listing, key, "", {}, mp.BY_FAMILY, mp.ALL_TASKS,
                    mp.RECORDS, "idle", request)
    by = dict(zip(map(id, opened.outputs), out, strict=True))
    assert out[1] == key
    assert f'data-key="{key}"' in by[id(by_id["shep-mp-detail-hd"])]
    assert world["shas"]["model-05.pt"] in by[id(by_id["shep-mp-detail-top"])]


_TREE_PAGE_JS = r"""
const { chromium } = require('playwright');
const QUEUE_JS = __QUEUE_JS__, TREE_JS = __TREE_JS__;
const ROWS = '#shep-models-research-rows';
(async () => {
  const base = process.argv[2];
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: 1850, height: 960 }, deviceScaleFactor: 1 });
  const page = await ctx.newPage();
  await page.addInitScript(QUEUE_JS);
  await page.addInitScript(TREE_JS);
  const idle = () => page.waitForFunction(() => window.__shepQueue.idle(), null, { timeout: 30000 });
  const drawn = (t) => page.waitForFunction((x) => {
    const m = document.querySelector('#shep-models-research-rows .shep-view-for');
    return !!m && m.getAttribute('data-for') === x;
  }, t, { timeout: 30000 });
  const tree = () => page.evaluate((r) => window.__shepTree(r), ROWS);
  const count = () => page.$eval('#shep-models-research-count', (e) => e.innerText.trim());
  // A run's toggle, by its run directory (its node is "r:<layout>/<run>").
  const fold = async (run) => {
    const b = await page.evaluateHandle((p) => [...document.querySelectorAll('#shep-models-research-rows button.shep-fold')]
      .find((x) => x.getAttribute('data-fold') === 'r:gatv2/' + p), run);
    if (!b.asElement()) {
      throw new Error('no toggle for ' + run + ': ' + await page.evaluate(() => [...document.querySelectorAll('#shep-models-research-rows button.shep-fold')].map((x) => x.getAttribute('data-fold') + (x.offsetParent ? '' : ' (hidden)')).join(', ')));
    }
    await b.asElement().click();
  };
  const pick = async (part) => {
    await page.click(`${ROWS} tr[data-key*="${part}"]:not([hidden])`);
    await page.waitForFunction((p) => !!document.querySelector(`#shep-models-research-rows tr[aria-current="true"][data-key*="${p}"]`), part);
    await idle();
  };
  const task = async (label) => {
    await page.click('#shep-mp-task input');
    await page.click(`ul[role=listbox] li:text-is("${label}")`);
    await drawn('newest|' + label.toLowerCase());
    await idle();
  };
  const out = {};
  await page.goto(base, { waitUntil: 'domcontentloaded' });
  await page.click('#shep-tab-research-button', { timeout: 60000 });
  await page.click('#shep-page-models-button');
  await drawn('newest|all');
  await idle();
  out.folded = await tree();
  out.count = await count();
  // A fold toggle unfolds at once in the page, before any answer.
  await fold('20261001-000000_base_aaaaaa');
  out.unfolded_at_once = await tree();
  await idle();
  await fold('20261002-000000_disease_bbbbbb');
  await pick('disease_bbbbbb');
  out.selected = await tree();
  const frames = () => page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
  // A smaller side-by-side window (a UI_SIZES size), the page still side by side.
  await page.setViewportSize({ width: 1280, height: 900 });
  await frames();
  await page.waitForFunction(() => document.documentElement.getAttribute('data-shep-layout') === 'side');
  out.mid = await tree();
  // The narrow layout: the shell stacks the page two frames after a resize.
  await page.setViewportSize({ width: 1280, height: 1280 });
  await page.waitForFunction(() => document.documentElement.getAttribute('data-shep-layout') !== 'side');
  await frames();
  out.narrow = await tree();
  await page.setViewportSize({ width: 1850, height: 960 });
  await page.waitForFunction(() => document.documentElement.getAttribute('data-shep-layout') === 'side');
  await frames();
  // Search by name: the selection outside it is said, never changed.
  await page.click('#shep-models-research-search input, #shep-models-research-search textarea');
  await page.keyboard.type('GENE');
  await page.waitForFunction(() => document.querySelector('#shep-models-research-count').innerText.includes('GENE'));
  await idle();
  out.searched = { tree: await tree(), count: await count() };
  await page.click('#shep-models-research-search-clear');
  await page.waitForFunction(() => !document.querySelector('#shep-models-research-count').innerText.includes('“'));
  await idle();
  out.cleared = { tree: await tree(), count: await count() };
  await task('Gene');
  out.gene = { tree: await tree(), count: await count(),
               detail: await page.$eval('#shep-mp-detail-hd h2', (e) => e.getAttribute('data-key')) };
  await task('All');
  // Read again when shown: the selection, its detail and the folds stay.
  await page.click('#shep-page-test-button');
  await idle();
  await page.click('#shep-page-models-button');
  await page.waitForFunction(() => !!document.querySelector('#shep-mp-detail-hd h2[data-key]'), null, { timeout: 30000 });
  await idle();
  out.refreshed = { tree: await tree(),
                    detail: await page.$eval('#shep-mp-detail-hd h2', (e) => e.getAttribute('data-key')) };
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_the_tree_folds_searches_filters_and_keeps_the_selection_in_a_browser(
        tmp_path, world, monkeypatch):
    """D04 on the page as the dashboard mounts it, by real events, on the
    table Newest first draws (By family was this table until the stage-lineage
    round; its graph is `test_the_family_graph_in_a_browser`'s): each run
    folded but the model in use's; a run's toggle unfolds its files at once in
    the page; a file is chosen; a search by name ignoring case keeps what it
    found and says the selection is outside it, never dropping it; Clear
    brings every row back; the Task filter the same; shown again (a refresh),
    the selection, its detail and the folds stay."""
    from src.webui.app import create_gradio_app, mount_options
    from tests.fixtures.browser import QUEUE_JS, TREE_JS, node_with_playwright, run_node
    from tests.fixtures.served_page import served

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    operator_request(monkeypatch)
    shas = _add_bundles(world)
    _add_child(world, "20261004-000000_demo-disease-extend_eeeeee", shas["disease"],
               BUNDLES["disease"][0])
    script = (_TREE_PAGE_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS))
              .replace("__TREE_JS__", json.dumps(TREE_JS)))
    with served(create_gradio_app(), **mount_options()) as base:
        seen = run_node(script, tmp_path, base[:-len("gradio_api")])

    def files(tree, part):
        return [f for f in tree["files"] if part in f["key"]]

    def run(tree, part):
        return next(r for r in tree["runs"] if part in r["node"])

    folded = seen["folded"]
    assert len(folded["groups"]) == 1 and "title" not in folded["groups"][0]
    assert not any(f["shown"] for f in files(folded, "gatv2/"))          # each run folded
    assert run(folded, "base_aaaaaa")["shown"] and not run(folded, "base_aaaaaa")["open"]
    assert "famil" not in seen["count"] and " · " not in seen["count"]
    assert all(f["shown"] for f in files(seen["unfolded_at_once"], "base_aaaaaa"))
    chosen = files(seen["selected"], "disease_bbbbbb")
    assert any(f["selected"] for f in chosen) and all(f["shown"] for f in chosen)
    # The table fits its frame, side by side and stacked, and no word is cut
    # over two lines (it was 48 px wider than its frame at 1850 x 960, and
    # "Disease" broke as "Dise/ase" at 1280 x 1280; at 1280 x 900, with a lineage
    # two runs deep under a long run name, 125 px wider, its Fingerprint cut).
    for size in ("selected", "mid", "narrow"):
        assert seen[size]["fit"]["over"] <= 0 and seen[size]["fit"]["split"] == [], (size, seen[size]["fit"])
    searched = seen["searched"]
    assert searched["count"] == "1 run matches “GENE”"
    assert [f["key"] for f in searched["tree"]["files"]] == [bundle_key("gene")]
    assert "Selected model not shown by this filter (search “GENE”)" in searched["tree"]["outside"]
    cleared = seen["cleared"]
    assert cleared["tree"]["outside"] is None and "“" not in cleared["count"]
    assert any(f["selected"] for f in files(cleared["tree"], "disease_bbbbbb"))
    gene = seen["gene"]
    assert gene["count"].startswith("Gene: 1 of ") and "Task: Gene" in gene["tree"]["outside"]
    assert gene["detail"] == bundle_key("disease")                     # its details kept
    refreshed = seen["refreshed"]
    assert refreshed["detail"] == bundle_key("disease")
    assert any(f["selected"] and f["shown"] for f in files(refreshed["tree"], "disease_bbbbbb"))
    assert run(refreshed["tree"], "base_aaaaaa")["open"]               # the operator's fold


#: Models served whole: a file chosen, then the row of an entry the listing
#: could not list hovered and clicked, and what the page shows after.
_ENTRY_CLICK_JS = r"""
const { chromium } = require('playwright');
const QUEUE_JS = __QUEUE_JS__;
const ROWS = '#shep-models-research-rows';
(async () => {
  const browser = await chromium.launch();
  const page = await (await browser.newContext({ viewport: { width: 1850, height: 960 },
                                                 deviceScaleFactor: 1 })).newPage();
  await page.addInitScript(QUEUE_JS);
  const idle = () => page.waitForFunction(() => window.__shepQueue.idle(), null, { timeout: 30000 });
  const frames = () => page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
  await page.goto(process.argv[2], { waitUntil: 'domcontentloaded' });
  await page.click('#shep-tab-research-button', { timeout: 60000 });
  await page.click('#shep-page-models-button');
  await page.waitForFunction(() => {
    const m = document.querySelector('#shep-models-research-rows .shep-view-for');
    return !!m && m.getAttribute('data-for') === 'newest|all';
  }, null, { timeout: 30000 });
  await idle();
  await page.click(`${ROWS} tr[data-key*="model-12.pt"]`);
  await page.waitForFunction(() => {
    const h = document.querySelector('#shep-mp-detail-hd h2[data-key]');
    return !!h && h.getAttribute('data-key').endsWith('model-12.pt');
  }, null, { timeout: 30000 });
  await idle();
  const entry = page.locator(`${ROWS} tr.shep-row-dis`, { hasText: 'cannot be listed' });
  const look = () => entry.evaluate((tr) => ({ bg: getComputedStyle(tr.querySelector('td')).backgroundColor,
                                               cursor: getComputedStyle(tr.querySelector('td')).cursor }));
  await page.mouse.move(5, 5);
  await frames();
  const rest = await look();
  await entry.hover();
  await frames();
  const hover = await look();
  // As a mouse clicks, whatever the row says of itself.
  await entry.click({ force: true });
  await frames();
  await idle();
  const after = await page.evaluate((R) => {
    const rows = document.querySelector(R);
    const tr = [...rows.querySelectorAll('tr.shep-row-dis')].find((t) => t.innerText.includes('cannot be listed'));
    const current = [...rows.querySelectorAll('tr[aria-current="true"]')].map((t) => t.getAttribute('data-key'));
    return { focused: document.activeElement === tr, tabIndex: tr.tabIndex,
             keyed: tr.hasAttribute('data-key'), current,
             entryCurrent: tr.getAttribute('aria-current'),
             detail: document.querySelector('#shep-mp-detail-hd h2[data-key]').getAttribute('data-key'),
             detailText: document.querySelector('#shep-mp-detail-hd h2').innerText };
  }, ROWS);
  console.log(JSON.stringify({ rest, hover, after }));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_an_entry_it_could_not_list_is_no_choice_in_a_browser(tmp_path, world, monkeypatch):
    """V-21 (the verification of fcc3dea, part RB2): on Models, with a run
    folder whose name is not valid text, its row looked like a choice (the
    hover's background, a tab stop) but a click on it chose nothing, so the
    details stayed on the model chosen before while that row had the
    reader's focus. An entry the listing could not list holds no model to
    show or load: its row says what it is and why, once, and is no choice
    (the shared tree's rule, Models and the Load drawer alike; the drawer
    lists task models only). In a real browser (skipped without Node and
    Playwright), Models served whole at 1850 x 960: a file chosen, the
    entry's row hovered and clicked: no hover cue, no pointer, no focus, no
    key; the chosen file stays the one row marked chosen and its details
    stay drawn. Fails on 78444a6 (the row took the focus and the hover)."""
    from src.webui.app import create_gradio_app, mount_options
    from tests.fixtures.browser import QUEUE_JS, node_with_playwright, run_node
    from tests.fixtures.served_page import served

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    operator_request(monkeypatch)
    (world["workspace"] / "runs" / "hgt" / os.fsdecode(b"bad\xff")).mkdir()
    with served(create_gradio_app(), **mount_options()) as base:
        seen = run_node(_ENTRY_CLICK_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS)), tmp_path,
                        base[:-len("gradio_api")])
    after = seen["after"]
    assert seen["hover"] == seen["rest"] and seen["rest"]["cursor"] == "default", seen
    assert not after["focused"] and after["tabIndex"] < 0 and not after["keyed"], after
    assert after["entryCurrent"] is None, after
    assert after["current"] == [after["detail"]] and after["detail"].endswith("model-12.pt"), after


#: Models served whole: a file with no recorded result chosen, its "Test this
#: model ›" clicked; the Test page as it opens (the page shown, the Run and
#: Checkpoint selects as drawn). Screens of both, at 1850 x 960 and the display floor's 1280 x 720.
_TEST_THIS_JS = r"""
const { chromium } = require('playwright');
const QUEUE_JS = __QUEUE_JS__;
const ROWS = '#shep-models-research-rows';
(async () => {
  const [base, shots] = process.argv.slice(2);
  const browser = await chromium.launch();
  const out = {};
  for (const [w, h] of [[1850, 960], [1280, 720]]) {
    const page = await (await browser.newContext({ viewport: { width: w, height: h },
                                                   deviceScaleFactor: 1 })).newPage();
    await page.addInitScript(QUEUE_JS);
    const idle = () => page.waitForFunction(() => window.__shepQueue.idle(), null, { timeout: 30000 });
    await page.goto(base, { waitUntil: 'domcontentloaded' });
    await page.click('#shep-tab-research-button', { timeout: 60000 });
    await page.click('#shep-page-models-button');
    await page.waitForFunction(() => {
      const m = document.querySelector('#shep-models-research-rows .shep-view-for');
      return !!m && m.getAttribute('data-for') === 'newest|all';
    }, null, { timeout: 30000 });
    await idle();
    // Its run unfolded (each run is drawn folded but the newest).
    const row = page.locator(`${ROWS} tr[data-key*="model-30.pt"]`);
    if (!(await row.isVisible())) {
      const fold = page.locator(`${ROWS} button.shep-fold[data-fold="r:sage/20261015-083000_run_e4c2a9"]`);
      await fold.scrollIntoViewIfNeeded();
      await fold.click();
      await row.waitFor({ state: 'visible', timeout: 30000 });
    }
    await row.scrollIntoViewIfNeeded();
    await row.click();
    await page.waitForFunction(() => document.querySelector('#shep-mp-records').innerText
      .includes('No recorded result yet'), null, { timeout: 30000 });
    await idle();
    await page.locator('#shep-mp-records').scrollIntoViewIfNeeded();
    const said = await page.$eval('#shep-mp-records', (e) => e.innerText.trim());
    await page.screenshot({ path: `${shots}/models_empty_${w}x${h}.png` });
    const action = page.locator('#shep-mp-records button[data-act="test"]');
    await action.waitFor({ state: 'visible', timeout: 10000 });
    await action.click();
    await page.waitForFunction(() => {
      const t = document.querySelector('#shep-page-test-button');
      const m = document.querySelector('#shep-test-model input');
      return !!t && t.getAttribute('aria-selected') === 'true' && !!m && m.value.trim() !== '';
    }, null, { timeout: 30000 });
    await idle();
    const seen = await page.evaluate(() => ({
      run: document.querySelector('#shep-test-model-run input').value,
      checkpoint: document.querySelector('#shep-test-model input').value,
      workspace: document.querySelector('#shep-test-workspace input').value,
      testShown: document.querySelector('#shep-page-test-button').getAttribute('aria-selected'),
    }));
    await page.screenshot({ path: `${shots}/test_opened_${w}x${h}.png` });
    out[`${w}x${h}`] = { said, ...seen };
    await page.close();
  }
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_test_this_model_opens_the_test_page_with_it_chosen_in_a_browser(
        tmp_path, world, monkeypatch):
    """R-68 in a real browser (skipped without Node and Playwright), Models
    served whole at 1850 x 960 and the display floor's 1280 x 720: model-30.pt, with no recorded
    result, chosen; its empty state's "Test this model ›" clicked; the Test
    page is the page shown, with that workspace, run and checkpoint chosen
    (the options the Test page draws for it). Fails on 6a2d52e: there was no
    action to click."""
    from src.webui.app import create_gradio_app, mount_options
    from src.webui.components import test_page as tp
    from src.webui.shell import standalone_context
    from tests.fixtures.browser import QUEUE_JS, node_with_playwright, run_node
    from tests.fixtures.served_page import served

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    request = operator_request(monkeypatch)
    key = _key(world, "model-30.pt")
    # The labels the Test page's selects draw for that checkpoint.
    with gr.Blocks() as alone:
        ctx = standalone_context(alone)
        tp.create_test_page(ctx)
    opened = next(fn for fn in alone.fns.values() if fn.name == "_refresh"
                  and (ctx.page_tokens["test"]._id, "change") in fn.targets)
    by_id = {getattr(b, "elem_id", None): b for b in alone.blocks.values()}
    # The page opened on that workspace and checkpoint chosen by hand.
    chosen = ["demo", None, key, None, {}, None]
    out = dict(zip(map(id, opened.outputs), opened.fn(
        *chosen, *[None] * (len(opened.inputs) - len(chosen)), request)))

    def label(elem_id):
        update = out[id(by_id[elem_id])]
        return next(shown for shown, value in update["choices"] if value == update["value"])

    labels = tuple(label(i) for i in ("shep-test-workspace", "shep-test-model-run",
                                      "shep-test-model"))
    shots = tmp_path / "shots"
    shots.mkdir()
    with served(create_gradio_app(), **mount_options()) as base:
        seen = run_node(_TEST_THIS_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS)), tmp_path,
                        base[:-len("gradio_api")], str(shots))
    for size, at in seen.items():
        assert at["said"].startswith("No recorded result yet"), (size, at)
        assert at["said"].endswith("Test this model ›"), (size, at)
        assert at["testShown"] == "true", (size, at)
        assert (at["workspace"], at["run"], at["checkpoint"]) == labels, (size, at)


#: The page's Order switch, Task select and "Applying …" line as Gradio draws
#: them, and rows the shared list drew before the page shaped its view (no
#: marker); By family clicked, then the page's own rows drawn.
_PENDING_EARLY_JS = r"""
const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1850, height: 960 } });
  await page.goto('file://' + process.argv[2]);
  const said = () => page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(
    () => r(document.querySelector('#shep-mp-pending .shep-st-w').textContent)))));
  const before = await said();
  await page.click('#family');
  const clicked = await said();
  await page.evaluate(() => {
    document.getElementById('shep-models-research-rows').innerHTML =
      '<table><tr data-key="f"><td>family rows</td></tr></table>'
      + '<span class="shep-view-for" hidden data-for="family|all"></span>';
  });
  const shaped = await said();
  console.log(JSON.stringify({ before, clicked, shaped }));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_an_order_switched_before_the_first_answer_is_announced(tmp_path):
    """Verification of fcc3dea (V-RB2-5, FL-UI-05): By family clicked while
    Models' opening answer was on its way read By family for 0.6-1 s over the
    newest-first rows with no "Applying …": the line waited for rows with a
    view marker, and the rows the shared list draws before the page first
    shapes its view carry none. Those rows are the list's own order (newest
    first, every task), so the switch is announced at once, and the line
    goes once the page's rows answer it. Fails on fcc3dea."""
    from tests.fixtures.browser import node_with_playwright, run_node

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    page = tmp_path / "pending.html"
    page.write_text(f"""<!doctype html><html><head><meta charset="utf-8"></head><body>
<div id="shep-mp-order"><label><input type="radio" name="o" value="newest" checked>Newest first</label>
<label><input id="family" type="radio" name="o" value="family">By family</label></div>
<div id="shep-mp-task"><input value="All"></div>
<div id="shep-mp-pending">{mp.PENDING_HTML}</div>
<div id="shep-models-research-rows"><table><tr data-key="a"><td>newest rows</td></tr></table></div>
<script>{mp.PENDING_JS}</script></body></html>""", encoding="utf-8")
    seen = run_node(_PENDING_EARLY_JS, tmp_path, str(page))
    assert seen == {"before": "", "clicked": "Applying By family…", "shaped": ""}, seen


#: Models for a read-only viewer: Load this model and its reason's boxes.
_LOAD_REASON_JS = r"""
const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch();
  const out = {};
  for (const [w, h] of [[1280, 720], [1440, 1600], [1850, 960]]) {
    const page = await browser.newPage({ viewport: { width: w, height: h } });
    await page.goto(process.argv[2]);
    await page.click('#shep-tab-research-button');
    await page.click('#shep-page-models-button');
    await page.waitForFunction(() => {
      const r = document.querySelector('#shep-models-research-reason');
      return r && /Operator controls only/.test(r.innerText);
    }, null, { timeout: 60000 });
    await page.locator('#shep-models-research-load').scrollIntoViewIfNeeded();
    out[`${w}x${h}`] = await page.evaluate(() => {
      const box = (s) => { const r = document.querySelector(s).getBoundingClientRect();
        return { left: r.left, top: r.top, right: r.right, bottom: r.bottom, width: r.width, height: r.height }; };
      const words = document.querySelector('#shep-models-research-reason .shep-reason, #shep-models-research-reason');
      const range = document.createRange(); range.selectNodeContents(words);
      const lines = new Set([...range.getClientRects()].map((r) => Math.round(r.top))).size;
      // The foot's last line against its bottom border, and the panel's own padding there.
      const foot = document.querySelector('#shep-models-research-load').closest('.shep-model-foot');
      const end = foot.getBoundingClientRect().bottom - parseFloat(getComputedStyle(foot).borderBottomWidth);
      const pad = parseFloat(getComputedStyle(document.getElementById('shep-mp-detail-bd')).paddingBottom);
      return { load: box('#shep-models-research-load'), reason: box('#shep-models-research-reason'), lines,
               under: end - range.getBoundingClientRect().bottom, pad };
    });
    await page.close();
  }
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_the_load_reason_has_room_beside_or_under_its_button_at_every_window(
        tmp_path, world, monkeypatch):
    """Verification of fcc3dea (V-RB2-4): for a read-only viewer at 390x844
    the reason beside Load this model was an 84 px column eight lines tall,
    its left edge inside the button's box: the foot is one grid row (load,
    reason, note) at every width. On a phone's column each takes a row of
    its own. Fails on fcc3dea. The sweep's L7: beside the button a reason is
    one line or none (the layout rules' "beside"); the read-only reason
    took two lines beside it, so it stands under the button wherever it
    cannot be one line beside it. The sweep's L7 (LOOK3): the reason, the
    foot's last line, stood 3-6 px from the panel's bottom border, where
    every other panel's last line keeps the panel's padding: it keeps the
    detail body's padding under it."""
    from src.webui.app import create_gradio_app, mount_options
    from tests.fixtures.browser import node_with_playwright, run_node
    from tests.fixtures.served_page import served

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    monkeypatch.setenv("SHEPHERD_DEPLOYMENT_MODE", "read-only")
    with served(create_gradio_app(), **mount_options()) as base:
        seen = run_node(_LOAD_REASON_JS, tmp_path, base[:-len("gradio_api")])
    for size, at in seen.items():
        load, reason = at["load"], at["reason"]
        apart = reason["left"] >= load["right"] or reason["top"] >= load["bottom"]
        assert apart and reason["width"] >= 200 and at["lines"] <= 3, (size, at)
    phone = seen["1280x720"]
    assert phone["reason"]["top"] >= phone["load"]["bottom"], phone
    for size, at in seen.items():
        beside = at["reason"]["top"] < at["load"]["bottom"]
        assert not beside or at["lines"] == 1, (size, at)
        assert at["under"] >= at["pad"] - 0.5, (size, at)


# --------------------------------------------------------------------------- the UI sweep of 2026-11-22
def _layout_children(demo, elem_id):
    """The config's layout node of the component `elem_id` and its children's
    (elem_id, elem_classes) in order."""
    config = demo.get_config_file()
    props = {c["id"]: (c.get("props") or {}) for c in config["components"]}
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}

    def find(node, wanted):
        if node.get("id") == wanted:
            return node
        for child in node.get("children", []):
            hit = find(child, wanted)
            if hit:
                return hit
        return None

    def named(node):
        p = props[node["id"]]
        return p.get("elem_id"), p.get("elem_classes") or []

    node = find(config["layout"], by_id[elem_id]._id)
    return node, [named(c) for c in node.get("children", [])], named, find, config


def test_order_is_a_group_of_the_lists_tools_bar_after_the_search():
    """GB10-UI-03 and B-03 (P2): the "Not ordered…" note sat in the head's grid
    area beside Search and widened its track until the search box was 22 px,
    Clear over it. Order is now a group (``shep-grp``) of the list's tools bar
    (``shep-bar``, `model_list.create_model_list`'s `tools`), right after the
    search group (``shep-grp shep-grp-grow``); the note, the "Applying …" line
    and the count are the bar's notes, rows of their own under the groups; the
    head holds the title, Workspace, Task and the workspace's line only. Every
    elem_id the page wires is still there."""
    demo, _ctx = _page()
    listing, kids, named, find, config = _layout_children(demo, "shep-models-research")
    main = next(c for c in listing["children"] if "shep-model-main" in named(c)[1])
    head, tools = [c for c in main["children"]
                   if {"shep-model-head", "shep-model-tools"} & set(named(c)[1])]
    assert "shep-model-head" in named(head)[1] and "shep-model-tools" in named(tools)[1]
    assert "shep-bar" in named(tools)[1] and "shep-edge" in named(main)[1]
    in_tools = [named(c) for c in tools["children"]]
    ids = [i for i, _cls in in_tools]
    assert ids == ["shep-models-research-searchrow", "shep-mp-controls", "shep-mp-options",
                   "shep-mp-pending", "shep-models-research-count"], in_tools
    classes = dict(in_tools)
    assert {"shep-grp", "shep-grp-grow"} <= set(classes["shep-models-research-searchrow"])
    assert "shep-grp" in classes["shep-mp-controls"]
    for note in ("shep-mp-options", "shep-mp-pending", "shep-models-research-count"):
        assert "shep-bar-note" in classes[note], (note, classes[note])
    order_group = find(tools, next(c["id"] for c in tools["children"]
                                   if named(c)[0] == "shep-mp-controls"))
    assert [named(c)[0] for c in order_group["children"]] == [
        "shep-mp-order-lbl", "shep-mp-order", "shep-mp-order-i"]

    def ids_under(node):
        out = [named(node)[0]]
        for child in node.get("children", []):
            out += ids_under(child)
        return out

    assert "shep-mp-controls" not in ids_under(head) and "shep-mp-task" in ids_under(head)
    # No track sized for one window: Task is a select at the select floor, the header a
    # row of fields, nothing flattened into the panel's grid.
    css = mp.PAGE_CSS
    assert "170px" not in css and "display: contents" not in css
    assert "flex: 0 1 var(--shep-select-floor)" in css


def test_the_order_note_and_the_details_empty_state_follow_what_is_listed():
    """GB10-UI-03 and the skill's §9: "Not ordered…" says why By family could
    not order a list, so it is said only where there is a list (first use and
    an empty workspace list nothing: the list's own empty state says why), and
    the Details panel's empty state says, per cause, what selecting shows or
    that nothing is listed to select (it said "Select a model…" with no list)."""
    refused = mp.FAMILY_NOT_READ
    data = {"readonly": False}
    nothing = {**ml.empty_listing(), "status_read": True}
    listed = {**nothing, "workspace": "demo", "items": [{"run": "r", "filename": "last.pt"}]}
    assert mp.options_html(data, refused, nothing) == ""
    assert "Not ordered, so the list stays newest first" in mp.options_html(data, refused, listed)
    assert "Not ordered" in mp.options_html(data, refused)           # no listing given: as before
    assert mp.options_html(data, None, listed) == ""
    empty = mp.detail_parts(nothing, {}, None, {})["top"]
    assert mp.DETAIL_NOTHING in empty and 'class="shep-empty"' in empty
    assert mp.DETAIL_SELECT in mp.detail_parts(listed, {}, None, {})["top"]
    # Before the list is read, nothing is said to be missing.
    assert mp.DETAIL_SELECT in mp.detail_parts({}, {}, None, {})["top"]


def test_every_models_help_text_is_within_the_text_budget():
    """The text brief's §11 (TEXT_BUDGET): each ⓘ of Models (its own and the
    Results segment's) a lead that answers the question in 90 characters or
    fewer, then 2-4 items of one form, each 76 characters or fewer, 80 words
    at most, no semicolon, Ref last (four texts were over it)."""
    from src.webui.components import help as h
    from src.webui.components import results_segment as rs

    texts = {**mp.PAGE_HELP, **rs.SEGMENT_HELP}
    assert set(texts) == {"models-ordering", "models-contested", "models-usage", "models-results"}
    for key, (_title, text) in texts.items():
        assert h.budget_problems(text, h.TEXT_BUDGET) == (), (key, h.budget_problems(text, h.TEXT_BUDGET))
    # Every fact kept: the ones each text carried before it was laid out again.
    assert "deterministic algorithms" in texts["models-contested"][1]
    assert "Test page notes" in texts["models-contested"][1]
    assert "Ref: reports/usage.jsonl" in texts["models-usage"][1]
    assert "uncapped" in texts["models-results"][1] and "first five" in texts["models-results"][1]


def test_the_load_note_is_a_row_under_load_and_results_actions_follow_their_fields():
    """The layout rules' "beside" (a message never shares a row with a button):
    what Load does (the view's note, two sentences) stood beside Load this
    model in the foot's one row; it is a row of its own under Load and its
    reason, at the panel's edge. Since the sweep's L7 the reason is a row of
    its own under Load at every width too (it took two lines beside Load at
    1550, 1366 and 1280, and a reason beside its button is one line or
    none), so no width needs a query of its own. In Models › Results (the L0 check's
    item 5) Export and Read again were stretched to the row's far end, under
    the choices' line: its rows are the shared rows of fields (a button as
    wide as its words, on the boxes' bottom line) and Export's two choices as
    wide as their options, so the button follows them."""
    css = mp.PAGE_CSS
    assert 'grid-template-areas: "detail" "load" "reason" "note";' in css
    assert "@media" not in css and "@container" not in css
    assert ".shep-rs-row > .form:has(> fieldset.block) { flex: 0 1 auto !important;" in css
    from src.webui.components import results_segment as rs

    with gr.Blocks():
        parts = rs.create_results_segment()
    rows = [b for b in parts["column"].children if "shep-rs-row" in (b.elem_classes or [])]
    assert len(rows) == 4 and all("shep-field-row" in row.elem_classes for row in rows)


def test_an_empty_results_segment_says_what_will_appear_and_keeps_its_command_in_ref():
    """The text brief's §8 and §9: the Results segment with nothing to show
    said its gap across workspaces with the command inside the sentence and a
    semicolon; it is an empty state in the flow, what is missing, what will
    appear, the gap, and the command in a Ref line only."""
    from src.webui.components import results_segment as rs

    html = rs.tables_html({**rs.EMPTY_STATE, "read_id": 1, "read_at": "now"},
                          {"scope": rs.WORKSPACE, "workspace": "demo"},
                          {"shown": [], "in_scope": 0, "unplaced_in_scope": 0})
    assert 'class="shep-empty shep-rs-empty"' in html and ";" not in rs.CROSS_WORKSPACE_GAP
    sentences = html.split('<p class="shep-rs-ref">')[0]
    assert rs.EXPORT_SCRIPT not in sentences and "export_results" not in sentences
    assert '<p class="shep-rs-ref">Ref: <code class="shep-tok">' in html


#: Models at first use (no workspaces folder), By family chosen, at each window: the search
#: box, Clear, the Order group and the notes, as drawn.
_FIRST_USE_TOOLS_JS = r"""
const { chromium } = require('playwright');
const QUEUE_JS = __QUEUE_JS__;
(async () => {
  const browser = await chromium.launch();
  const page = await (await browser.newContext({ viewport: { width: 1536, height: 864 }, deviceScaleFactor: 1 })).newPage();
  await page.addInitScript(QUEUE_JS);
  const idle = () => page.waitForFunction(() => window.__shepQueue.idle(), null, { timeout: 30000 });
  await page.goto(process.argv[2], { waitUntil: 'domcontentloaded' });
  await page.click('#shep-tab-research-button', { timeout: 60000 });
  await page.click('#shep-page-models-button');
  await page.waitForSelector('#shep-mp-order input[value="family"]:not([disabled])', { timeout: 60000 });
  await idle();
  await page.click('#shep-mp-order label:has(input[value="family"])');
  await page.waitForFunction(() => {
    const m = document.querySelector('#shep-models-research-rows .shep-view-for');
    return !!m && m.getAttribute('data-for').startsWith('family');
  }, null, { timeout: 30000 });
  await idle();
  const out = {};
  for (const [w, h] of [[1536, 864], [1366, 768], [1280, 720]]) {
    await page.setViewportSize({ width: w, height: h });
    await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
    out[`${w}x${h}`] = await page.evaluate(() => {
      const q = (s) => document.querySelector(s);
      const box = (e) => { const r = e.getBoundingClientRect(); return { left: r.left, top: r.top, right: r.right, bottom: r.bottom, width: r.width }; };
      const input = q('#shep-models-research-search input, #shep-models-research-search textarea');
      const shown = (e) => !!e && e.getClientRects().length > 0 && (e.innerText || '').trim() !== '';
      return { input: box(input), floor: parseFloat(getComputedStyle(input).minWidth) || 0,
               clear: box(q('#shep-models-research-search-clear')),
               search: box(q('#shep-models-research-searchrow')), order: box(q('#shep-mp-controls')),
               panel: box(q('#shep-page-models .shep-model-main')),
               note: shown(q('#shep-mp-options')),
               notices: document.querySelectorAll('#shep-page-models [data-notice="workspaces_root"]').length,
               detail: (q('#shep-mp-detail-top') || {}).innerText || '' };
    });
  }
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_at_first_use_the_search_keeps_its_floor_and_order_moves_whole_in_a_browser(
        tmp_path, monkeypatch):
    """GB10-UI-03 (P2), B-03, as the owner saw it (Node and Playwright, skipped
    without): no workspaces folder, By family chosen. The search box was a
    22 px sliver at 1536 x 864, 1366 x 768 and 1280 x 720, Clear over it (and,
    once the box kept its floor, Clear over Order). At every window now the box
    is at least its floor, Clear beside it on its line, Order a group of its
    own that never overlaps the search group (beside it or under it, whole),
    both inside the panel; no "Not ordered" note (there is no list to order);
    the workspaces folder's notice once, as the list's empty state; Details
    says there is nothing to select. Fails before the sweep's L4 lane."""
    from src.webui.app import create_gradio_app, mount_options
    from tests.fixtures.browser import QUEUE_JS, node_with_playwright, run_node
    from tests.fixtures.served_page import served

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    monkeypatch.delenv("SHEPHERD_WORKSPACES_ROOT", raising=False)
    monkeypatch.delenv("SHEPHERD_REGISTRY", raising=False)
    operator_request(monkeypatch)
    with served(create_gradio_app(), **mount_options()) as base:
        seen = run_node(_FIRST_USE_TOOLS_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS)), tmp_path,
                        base[:-len("gradio_api")])
    for size, at in seen.items():
        box, clear, search, order, panel = at["input"], at["clear"], at["search"], at["order"], at["panel"]
        assert at["floor"] > 0 and box["width"] >= at["floor"] - 0.5, (size, at)
        assert clear["left"] >= box["right"] - 0.5, (size, at)                     # beside, after it
        assert clear["top"] < box["bottom"] and clear["bottom"] > box["top"], (size, at)
        apart = (order["left"] >= search["right"] - 0.5 or order["top"] >= search["bottom"] - 0.5)
        assert apart, (size, at)                                                   # never over it
        for part in (search, order):
            assert panel["left"] - 0.5 <= part["left"] and part["right"] <= panel["right"] + 0.5, (size, at)
        assert not at["note"] and at["notices"] == 1, (size, at)
        assert mp.DETAIL_NOTHING in at["detail"], (size, at)


#: Newest first at a laptop's window and a wide column: the tree's frame, its optional
#: columns and whether it scrolls sideways.
_OPTIONAL_COLUMNS_JS = r"""
const { chromium } = require('playwright');
const QUEUE_JS = __QUEUE_JS__;
(async () => {
  const browser = await chromium.launch();
  const page = await (await browser.newContext({ viewport: { width: 1366, height: 768 }, deviceScaleFactor: 1 })).newPage();
  await page.addInitScript(QUEUE_JS);
  const idle = () => page.waitForFunction(() => window.__shepQueue.idle(), null, { timeout: 30000 });
  await page.goto(process.argv[2], { waitUntil: 'domcontentloaded' });
  await page.click('#shep-tab-research-button', { timeout: 60000 });
  await page.click('#shep-page-models-button');
  await page.waitForSelector('#shep-models-research-rows table.shep-mtree', { timeout: 60000 });
  await idle();
  const out = {};
  for (const [w, h] of [[1850, 960], [1550, 830], [1366, 768], [1440, 1600]]) {
    await page.setViewportSize({ width: w, height: h });
    await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
    out[`${w}x${h}`] = await page.evaluate(() => {
      const table = document.querySelector('#shep-models-research-rows table.shep-mtree'), wrap = table.parentElement;
      const rows = document.querySelector('#shep-models-research-rows');
      const opt = [...table.querySelectorAll('thead th')].filter((th) => th.classList.contains('shep-col-opt'));
      const list = document.querySelector('#shep-page-models .shep-model-main'), detail = document.querySelector('#shep-mp-detail');
      return { layout: document.documentElement.getAttribute('data-shep-layout'),
               list: list.getBoundingClientRect().width, detail: detail.getBoundingClientRect().width,
               sideways: Math.max(wrap.scrollWidth - wrap.clientWidth, rows.scrollWidth - rows.clientWidth),
               optional: opt.map((th) => th.textContent), drawn: opt.filter((th) => th.getClientRects().length).length,
               heads: [...table.querySelectorAll('thead th')].filter((th) => th.getClientRects().length).map((th) => th.textContent),
               // The lines of each drawn count's words in its place ("recorded epoch 4").
               epochs: [...table.querySelectorAll('tbody td.shep-num .shep-muted')].filter((e) => e.getClientRects().length)
                 .map((e) => { const r = document.createRange(); r.selectNodeContents(e);
                   return new Set([...r.getClientRects()].filter((q) => q.width > 0).map((q) => Math.round(q.top))).size; }) };
    });
  }
  // A row chosen: its details are what the page shows (the sweep's L7 check, D1).
  await page.setViewportSize({ width: 1850, height: 960 });
  await page.click('#shep-models-research-rows tr[data-key]:not([hidden])');
  await page.waitForSelector('#shep-mp-detail-hd h2[data-key]');
  await idle();
  for (const [w, h] of [[1850, 960], [1550, 830], [1366, 768], [1280, 720]]) {
    await page.setViewportSize({ width: w, height: h });
    await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
    out[`chosen ${w}x${h}`] = await page.evaluate(() => {
      const table = document.querySelector('#shep-models-research-rows table.shep-mtree'), wrap = table.parentElement;
      const rows = document.querySelector('#shep-models-research-rows');
      const opt = [...table.querySelectorAll('thead th')].filter((th) => th.classList.contains('shep-col-opt'));
      const body = document.querySelector('#shep-mp-detail-bd'), b = body.getBoundingClientRect();
      const seg = [...body.querySelectorAll('.shep-seg')].filter((e) => e.getClientRects().length).pop();
      const list = document.querySelector('#shep-page-models .shep-model-main'), detail = document.querySelector('#shep-mp-detail');
      const foot = document.querySelector('#shep-page-models .shep-model-foot');
      return { list: list.getBoundingClientRect().width, detail: detail.getBoundingClientRect().width,
               foot: foot.getBoundingClientRect().width,
               footMin: parseFloat(getComputedStyle(foot).getPropertyValue('--shep-panel-min')),
               layout: document.documentElement.getAttribute('data-shep-layout'),
               fits: body.scrollHeight <= body.clientHeight + 1,
               seg: seg ? seg.getBoundingClientRect().bottom <= b.top + body.clientHeight + 1 : null,
               sideways: Math.max(wrap.scrollWidth - wrap.clientWidth, rows.scrollWidth - rows.clientWidth),
               drawn: opt.filter((th) => th.getClientRects().length).length };
    });
  }
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_the_tree_drops_its_optional_columns_before_it_scrolls_sideways_in_a_browser(
        tmp_path, world, monkeypatch):
    """B-13 and the sweep's M6 (Node and Playwright, skipped without): with a
    run whose name is long (a family of fine-tunes), Newest first's table was
    wider than its panel at 1366 x 768 and scrolled sideways inside it, its
    Epochs, Files and Fingerprint off to the right. Those columns are optional
    (the selected model's details say each): the table drops them while it does
    not fit, so nothing scrolls sideways and Model, Task and Operation stay;
    where the whole table fits (a tall window's single column), every column is
    drawn. Fails before the sweep's L4 lane.
    The sweep's L7 (SW-LK4): the list took 46 parts of 100 beside an empty
    details panel wider than it, so at the owner's window (1850 x 960) the
    tree dropped its Epochs, Files and Fingerprint (file rows a name and empty
    cells, no fingerprint for "Name or fingerprint"). Side by side, the list,
    the page's main region, is now at least as wide as the details, and at the
    owner's windows the whole tree is drawn, nothing sideways.
    The sweep's L7 check (D1): once a row is chosen its details are what the
    page shows, yet they kept the empty panel's third, so at the owner's
    windows their recorded results fell under the region's edge. Chosen, the
    details are wider than empty, whole at the owner's largest window with the
    list's whole tree beside them, and their records segment in view at the
    smaller. The sweep's L7 check (D13): at 1850 x 960 every older run's
    "recorded epoch" words stood on two lines, doubling its row, while the
    Model column had room; they read as one line where the whole tree still
    fits with them so (`style.CELL_ROOMY`), and wrap where it would not."""
    from src.webui.app import create_gradio_app, mount_options
    from tests.fixtures.browser import QUEUE_JS, node_with_playwright, run_node
    from tests.fixtures.served_page import served

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    operator_request(monkeypatch)
    shas = _add_bundles(world)
    _add_child(world, "20261005-000000_disease-fine-tune-of-the-long-named-family_ffffff",
               shas["disease"], BUNDLES["disease"][0])
    # A name longer still (K2): its narrow form is wider than the list's fixed minimum.
    _add_child(world, "20261006-000000_disease-fine-tune-of-the-long-named-family-with-a-longer-tail_eeeeee",
               shas["disease"], BUNDLES["disease"][0])
    with served(create_gradio_app(), **mount_options()) as base:
        seen = run_node(_OPTIONAL_COLUMNS_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS)), tmp_path,
                        base[:-len("gradio_api")])
    laptop, column = seen["1366x768"], seen["1440x1600"]
    for size in ("1850x960", "1550x830"):
        owner = seen[size]
        assert owner["layout"] == "side" and owner["drawn"] == 3 and owner["sideways"] <= 0, (size, owner)
    # The sweep's L7 check (D13): at the owner's largest window a count's words in its place
    # read as one line ("recorded" / "epoch 4" doubled the row while Model had room); where
    # they would cost the table its columns (1550 x 830) they wrap and every column stays.
    assert seen["1850x960"]["epochs"] and set(seen["1850x960"]["epochs"]) == {1}, seen["1850x960"]
    for size in ("1850x960", "1550x830", "1366x768"):
        assert seen[size]["list"] >= seen[size]["detail"], (size, seen[size])
    assert laptop["optional"] == ["Epochs", "Files", "Fingerprint"] == column["optional"], seen
    assert laptop["layout"] == "side" and laptop["sideways"] <= 0, laptop
    assert laptop["drawn"] == 0 and laptop["heads"] == ["Model", "Task", "Operation"], laptop
    assert column["layout"] == "stack" and column["drawn"] == 3 and column["sideways"] <= 0, column
    wide, mid = seen["chosen 1850x960"], seen["chosen 1550x830"]
    assert wide["fits"] and wide["drawn"] == 3 and wide["sideways"] <= 0, wide
    for size, chosen in (("1850x960", wide), ("1550x830", mid)):
        assert chosen["detail"] > seen[size]["detail"] and chosen["sideways"] <= 0, (size, chosen, seen[size])
    assert mid["seg"] in (True, None), mid
    # The sweep's L7 check (K2): chosen, the details took all but the list's fixed minimum, which
    # was under its table's narrow form beside a long run name, and the rows scrolled sideways at
    # 1366 x 768 and 1280 x 720. Side by side, the details give the list the width its narrow form
    # needs unless that would take them under their own minimum.
    for size in ("1366x768", "1280x720"):
        chosen = seen[f"chosen {size}"]
        assert chosen["layout"] == "side" and chosen["footMin"] > 0, (size, chosen)
        assert chosen["sideways"] <= 0 or chosen["foot"] <= chosen["footMin"] + 0.5, (size, chosen)


# --------------------------------------------------------------------------- the family graph (G/I)
@pytest.fixture(scope="module")
def family_root(demo_workspace, tmp_path_factory):
    """The synthetic family world (`tests/fixtures/family_world.py`), written
    once: several families, a chain seven runs deep, alike and long names, two
    children from two files of one parent run, a parent in another workspace,
    one not available (its run named with markup) and a refused record."""
    from tests.fixtures.family_world import write_family_world

    root = tmp_path_factory.mktemp("family-world") / "root"
    root.mkdir()
    built = write_family_world(root, demo_workspace)
    return root, built["shas"]


@pytest.fixture
def family(family_root, tmp_path, monkeypatch):
    """The family world served: an operator's request, the model in use the
    alpha disease fine-tune."""
    root, shas = family_root
    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(root))
    monkeypatch.setenv("SHEPHERD_REGISTRY", str(tmp_path / "registry"))
    served = {"value": _status(root / "demo", shas[("alpha-disease", "last.pt")])}
    monkeypatch.setattr(ml, "read_status", lambda: (served["value"], None))
    monkeypatch.setattr(mp, "read_status", lambda: (served["value"], None))
    operator_request(monkeypatch)
    return {"root": root, "shas": shas, "served": served}


def _family_view(task=mp.ALL_TASKS):
    listing = ml.list_models("demo", gradio_request(), family=True)
    data = mp.read_page(listing)
    return listing, mp.shape(listing, data, mp.controls(data, mp.BY_FAMILY, task))


def _depth(html, run):
    """How many lists of child runs a run's node sits in (its depth as drawn)."""
    at = html.index(f'<li class="shep-fam-li" data-run="{run}"')
    before = html[:at]
    return before.count('<ul class="shep-fam-kids"') - before.count("</ul>") + before.count(
        '<ul class="shep-fam-tree"') - 1


def test_the_family_graph_is_the_family_views_runs_and_edges_and_reads_nothing_again(
        family, monkeypatch):
    """G01, G05, S03 (server side): the graph's runs, edges and boundaries are
    the family view's (`server_lists.list_family`, read once with the listing):
    every edge from the exact parent file a run's first file records, a run
    whose parent is in another listed workspace or not available a boundary
    with the reader's own state and words (never a parent guessed); shaping
    and drawing it lists nothing and reads no file again; the refused record
    is kept apart, and no model is refused or hidden for the graph."""
    from src.api.services import server_lists
    from tests.fixtures.family_world import EVIL, RUNS, file_key, run_key

    listing, view = _family_view()
    reads = []
    for name in ("list_checkpoints", "list_family", "read_run_checkpoints"):
        if hasattr(server_lists, name):
            real = getattr(server_lists, name)
            monkeypatch.setattr(server_lists, name,
                                lambda *a, _n=name, _r=real, **k: reads.append(_n) or _r(*a, **k))
    data = mp.read_page(listing)
    view = mp.shape(listing, data, mp.controls(data, mp.BY_FAMILY, mp.ALL_TASKS))
    html = ml.list_html(listing, None, ml.PRESETS["research"], view=view, fold={})
    assert reads == []                                   # nothing listed or read again
    rows = {row["key"]: row for row in listing["family"]["all"]}
    edges = {edge["to"]: edge["from"] for edge in view["edges"]}
    for label, *_rest in RUNS:
        row = rows[file_key(label)]
        if row["parent_state"] == "listed":
            assert edges[run_key(label)] == row["parent"], label
        else:
            assert run_key(label) not in edges, label
    assert edges[run_key("alpha-ext-5")] == file_key("alpha-ext-4")
    assert {k.split("/")[1]: (b["state"], b["workspace"]) for k, b in view["bounds"].items()} == {
        "20261114-000000_gamma-disease_c00002": ("elsewhere", "other"),
        "20261115-000000_delta-disease_d00002": ("unavailable", None)}
    delta = view["bounds"][run_key("delta-disease")]
    assert delta["problem"] == rows[file_key("delta-disease")]["parent_problem"] and (
        delta["run"] == EVIL)
    # One node per listed run of a family; the refused record apart, under its word.
    assert html.count('<li class="shep-fam-li" data-run=') == len(RUNS)
    assert html.count('class="shep-fam-node shep-fam-bound"') == 2
    rest = html[html.index(ml.NOT_IN_A_FAMILY):]
    assert "20261116-000000_old-gnn_e00001" in rest and "Record not readable" in rest
    assert EVIL not in html and "<img" not in html and html_lib.escape(EVIL) in html
    # The graph's own state is never kept by the shared disclosure keeper.
    assert 'class="shep-fam" data-shep-folds-own' in html
    assert "d.closest('[data-shep-folds-own]')" in r.KEEP_OPEN_JS
    # Every model the listing reads Load for stays loadable: the graph refuses nothing.
    assert {k for k, i in ((ml.item_key(i), i) for i in listing["items"]) if ml.can_load(i)} == {
        file_key(label) for label, _run, _files, _p, task, *_ in RUNS if task}


def test_two_children_of_one_parent_run_name_the_exact_file_each_came_from(family):
    """G02, G03, G06: two runs from two files of one parent run are two edges
    from two files; each node says which file it came from (its words, as the
    Checkpoint select names it), and the selected file's details, the one
    place for them, name that file and the operation, its 12 hex digits with
    the whole value one Copy away and the whole value read one "Details ›"
    away (not only copied); shortened words never change a key or a
    digest."""
    from tests.fixtures.family_world import file_key, run_key

    shas = family["shas"]
    listing, view = _family_view()
    edges = {edge["to"]: edge["from"] for edge in view["edges"]}
    assert edges[run_key("alpha-disease")] == file_key("alpha-base", "model-03.pt")
    assert edges[run_key("alpha-gene")] == file_key("alpha-base", "last.pt")
    html = ml.list_html(listing, file_key("alpha-gene"), ml.PRESETS["research"], view=view)
    disease, gene = _node(html, run_key("alpha-disease")), _node(html, run_key("alpha-gene"))
    assert "from model-03.pt" in _words(disease) and "from Latest save" in _words(gene)
    words = ml.pair_words({ml.item_key(i): i for i in listing["items"]})
    data = mp.read_page(listing)
    by_family = {"order": mp.BY_FAMILY, "task": mp.ALL_TASKS}
    for label, parent, other in (("alpha-disease", "model-03.pt", "last.pt"),
                                 ("alpha-gene", "last.pt", "model-03.pt")):
        top = mp.detail_parts(listing, data, file_key(label), by_family)["top"]
        rows = dict(re.findall(r'<dt class="shep-k[^"]*">(.*?)</dt><dd class="shep-v[^"]*">(.*?)</dd>',
                               top, re.S))
        sha = shas[("alpha-base", parent)]
        assert words[file_key("alpha-base", parent)][1] in _text(rows["Parent"]), label
        assert _text(rows["Operation"]) == "Fine-tune, full"
        # The whole parent digest drawn (one "Details ›" away), beside the model's own.
        more = top[top.index('<details class="shep-more"'):]
        assert f'<code class="shep-full" title="{sha}">{sha}</code>' in rows["Parent fingerprint"]
        assert more.index("Whole fingerprint") < more.index("Parent fingerprint")
        assert shas[("alpha-base", other)] not in top, label
    # The detail of the selected file: its exact parent, the whole digest one Copy away.
    detail = mp.detail_parts(listing, data, file_key("alpha-gene"), {})["top"]
    parent = re.search(r'<dt class="shep-k">Parent</dt><dd class="shep-v">(.*?)</dd>', detail,
                       re.S).group(1)
    sha = shas[("alpha-base", "last.pt")]
    assert words[file_key("alpha-base", "last.pt")][1] in _text(parent)
    assert f'>{sha[:12]}</code>' in parent and f'data-copy="{sha}"' in parent
    assert 'aria-label="Copy the full parent fingerprint"' in parent
    assert "listed here" in parent and shas[("alpha-base", "model-03.pt")][:12] not in parent
    # A boundary's run: its parent where the family view says it is.
    gamma = mp.detail_parts(listing, data, file_key("gamma-disease"), {})["top"]
    assert "in workspace <b>other</b>" in gamma


def test_a_deep_chain_is_drawn_nested_and_alike_names_are_told_apart(family):
    """G04, G06: a chain is drawn as deep as it is (the table indented four
    levels at most); each family shows its root and its direct children, a
    deeper subtree folded unless it holds the selection or the model in use
    (its ancestors unfolded); two runs named alike are told apart by their
    start time, and a long name is drawn whole."""
    from tests.fixtures.family_world import file_key, run_key

    listing, view = _family_view()
    html = ml.list_html(listing, file_key("alpha-ext-5"), ml.PRESETS["research"], view=view)
    assert _depth(html, run_key("alpha-ext-5")) == 6 and _depth(html, run_key("alpha-base")) == 0
    shape = ml.graph_shape(listing, view, ml.tree_view(listing, view, ml.PRESETS["research"]),
                           {}, file_key("alpha-ext-5"))
    assert {run_key(r) for r in ("alpha-base", "alpha-disease", "alpha-ext-1", "alpha-ext-2",
                                 "alpha-ext-3", "alpha-ext-4")} <= shape.open_runs
    plain = ml.graph_shape(listing, view, ml.tree_view(listing, view, ml.PRESETS["research"]),
                           {}, None)
    assert run_key("alpha-base") in plain.open_runs and run_key("beta-base") in plain.open_runs
    # The model in use (alpha-disease) is the root's child: its own subtree stays folded.
    assert run_key("alpha-disease") not in plain.open_runs
    assert run_key("alpha-ext-1") not in plain.open_runs
    beta = [_words(_node(html, run_key(r))) for r in ("beta-disease-1", "beta-disease-2")]
    assert beta[0] != beta[1] and "08:00" in beta[0] and "09:00" in beta[1]
    assert "beta-disease-with-a-rather-long-descriptive-run-name" in _words(
        _node(html, run_key("beta-long")))


def test_a_search_reveals_ancestors_an_arrow_selects_nothing_and_a_body_selects_its_run(family):
    """G7, I06, I8, I9 (server side): a search unfolds the ancestors of what
    it finds at drawing, the operator's folds unchanged, and clearing it
    brings them back; an arrow's press (``{tree: {node: open}}``, the subtree
    entries only) is kept by the fold state and changes nothing else. **A
    node's body selects its run** (``{run: key}``): the server looks the run up
    in what the list draws and chooses its checkpoint as the Run → Checkpoint
    choice does (`model_list.chosen_in_run`: the selection while it is one of
    the run's, else the run's latest save), then answers exactly as a row's
    pick of that file; a run it does not draw, or another act, changes
    nothing; nothing loads. One run reads selected, no run panel is drawn,
    and a fold state holding a run focused on an earlier page (``f``) is
    ignored."""
    from tests.fixtures.family_world import file_key, run_key

    listing, view = _family_view()
    research = ml.PRESETS["research"]
    manual = {ml.subtree_node(run_key("alpha-base")): False}
    tree = ml.tree_view(listing, view, research, search="ext-5", fold=manual)
    shape = ml.graph_shape(listing, view, tree, manual, None)
    assert run_key("alpha-base") in shape.open_runs and run_key("alpha-ext-4") in shape.open_runs
    cleared = ml.graph_shape(listing, view, ml.tree_view(listing, view, research, fold=manual),
                             manual, None)
    assert run_key("alpha-base") not in cleared.open_runs
    demo, _ctx = _page()
    select = _fn(demo, "_select")
    # The folds are the session's one dict, changed in place (`ListClock`), never an output.
    assert not [o for o in select.outputs if getattr(o, "value", None) == {}]
    handle_selected, handle_drawn = select.outputs[0], select.outputs[-1]
    picked = file_key("alpha-gene")

    def answer(raw, current=picked, view_=view):
        fold = {}
        out = select.fn(listing, current, False, view_, "", fold, gr.EventData(None, raw),
                        gradio_request())
        by = dict(zip(map(id, select.outputs), out, strict=True))
        # The draw token: whether a draw is asked for (its value is a time).
        if by[id(handle_drawn)] != gr.update():
            by[id(handle_drawn)] = "draw"
        return {**by, "fold": fold}

    for raw, kept in (
            ({"tree": {ml.subtree_node(run_key("alpha-disease")): True,
                       ml.subtree_node(run_key("alpha-ext-1")): False, "r:x": True, "f": "y"}},
             {ml.subtree_node(run_key("alpha-disease")): True,
              ml.subtree_node(run_key("alpha-ext-1")): False}),
            ({"tree": {ml.subtree_node(run_key("alpha-disease")): False}, "searched": True},
             {ml.search_node(ml.subtree_node(run_key("alpha-disease"))): False})):
        by = answer(raw)
        assert by["fold"] == kept, raw
        assert all(v == gr.update() for k, v in by.items() if k != "fold"), raw
    # A body's pick: the run's latest save (the selection is another run's) ...
    base_last, base_03 = file_key("alpha-base"), file_key("alpha-base", "model-03.pt")
    assert ml.item_key(next(i for i in listing["items"] if ml.item_run_key(i) == run_key(
        "alpha-base"))) == base_last                      # the run's first in the list's order
    by_run, by_row = answer({"run": run_key("alpha-base")}), answer({"key": base_last})
    assert by_run[id(handle_selected)] == base_last and by_run == by_row
    # ... the selection kept while it is one of the run's files ...
    by_run = answer({"run": run_key("alpha-base")}, current=base_03)
    assert by_run[id(handle_selected)] == base_03 and by_run == answer({"key": base_03},
                                                                        current=base_03)
    # ... and nothing for a run the list does not draw (never a key taken on trust),
    # a run the Task filter hides, a page's act that is not a pick, a stale focus.
    gene_view = _family_view("gene")[1]
    for raw, view_ in (({"run": "gatv2/no-such-run"}, view),
                       ({"run": run_key("alpha-disease")}, gene_view),
                       ({"act": "test", "key": base_last}, view),
                       ({"act": ml.PICK_ACT, "key": "__older__"}, view),
                       ({"focus": run_key("beta-base")}, view)):
        by = answer(raw, view_=view_)
        assert by.pop("fold") == {} and all(v == gr.update() for v in by.values()), raw
    html = ml.list_html(listing, picked, research, view=view, fold={})
    # One run looks selected and is ``aria-current``: the one holding the selected model;
    # no run panel, no body controlling one, no focused run's outline.
    forest = html[html.index('class="shep-fam-forest"'):]
    assert forest.count('aria-current="true"') == 1
    assert 'aria-current="true"' in _node(html, run_key("alpha-gene"))
    assert "shep-fam-run" not in html and "data-focus" not in html and "Back to the graph" not in html
    assert not re.search(r'class="shep-fam-body"[^>]*aria-controls', html)
    assert not re.search(r'class="shep-fam-node[^"]*\bshep-on\b', html)
    # A fold state an earlier page left holding a focused run: ignored, the same drawing.
    assert ml.list_html(listing, picked, research, view=view,
                        fold={"f": run_key("beta-base")}) == html


def test_a_pick_in_the_graph_folds_nothing_it_shows(family):
    """I5 (the server's half): a pick made in the graph keeps what it shows
    unfolded. A subtree unfolded only because it led to the selection before
    (the default: G6) would fold once another run is selected, moving every
    node under it; the pick stores it unfolded (`model_list.kept_unfolded`, a
    search's own entry while one is shown), so the drawing that answers shows
    the same nodes. Newest first and a pick that folds nothing store none."""
    from tests.fixtures.family_world import REFUSED_RUN, file_key, run_key

    listing, view = _family_view()
    research = ml.PRESETS["research"]
    deep = file_key("alpha-ext-3")
    chain = [ml.subtree_node(run_key(r)) for r in ("alpha-disease", "alpha-ext-1", "alpha-ext-2")]
    shown = ml.list_html(listing, deep, research, view=view, fold={})
    assert all(f'data-tree="{node}" aria-expanded="true"' in shown for node in chain)
    demo, _ctx = _page()
    select = _fn(demo, "_select")
    # The session's one fold dict, changed in place by the pick (`ListClock`).
    fold = {}
    out = select.fn(listing, deep, False, view, "", fold,
                    gr.EventData(None, {"run": run_key("beta-base")}), gradio_request())
    by = dict(zip(map(id, select.outputs), out, strict=True))
    assert by[id(select.outputs[0])] == file_key("beta-base")
    assert fold == dict.fromkeys(chain, True)
    # The drawing that answers it (the list's one draw, from the state it left).
    rows = ml.list_drawing(listing, file_key("beta-base"), research, view=view,
                           fold=fold).rows
    assert all(f'data-tree="{node}" aria-expanded="true"' in rows for node in chain)
    visible = re.findall(r'<li class="shep-fam-li" data-run="([^"]+)"', rows)
    assert visible == re.findall(r'<li class="shep-fam-li" data-run="([^"]+)"', shown)
    assert not re.search(r'<ul class="shep-fam-kids" id="[^"]+" hidden>(?:(?!</ul>).)*'
                         + re.escape(run_key("alpha-ext-3")), rows, re.S)
    # The table under the graph too: the refused record's run, unfolded while it holds
    # the selection; under a search, as the search's own entry (cleared with it, G7).
    refused = next(i for i in listing["items"] if i.get("run") == REFUSED_RUN)
    refused_run = ml.run_node(ml.item_run_key(refused))
    assert ml.kept_unfolded(listing, view, research, {}, ml.item_key(refused),
                            file_key("beta-base")) == {refused_run: True}
    assert ml.kept_unfolded(listing, view, research, {}, ml.item_key(refused),
                            file_key("beta-base"), search="00001") == {
        ml.search_node(refused_run): True}
    # Newest first draws no graph; a pick in the same run folds nothing.
    newest = mp.shape(listing, mp.read_page(listing), mp.controls(mp.read_page(listing),
                                                                     mp.NEWEST, mp.ALL_TASKS))
    assert ml.kept_unfolded(listing, newest, research, {}, deep, file_key("beta-base")) == {}
    fold = {}
    out = select.fn(listing, file_key("alpha-base"), False, view, "", fold,
                    gr.EventData(None, {"act": ml.PICK_ACT,
                                        "key": file_key("alpha-base", "model-03.pt")}),
                    gradio_request())
    assert out[0] == file_key("alpha-base", "model-03.pt") and fold == {}


def test_the_details_name_a_runs_saved_files_and_pick_them_through_the_list(family):
    """G2, G3, I8: By family, the details are the one place for a run's saved
    files: a run with several gets a "Saved files" row, each file by its
    Checkpoint words in the list's order, the one shown said in words and
    ``aria-current`` (not by colour alone), each other one a button whose
    pick (``{act: "pick", key}``, `renderers.TRIGGER_JS`) goes through the
    list's one selection, answered exactly as a row's pick; everything
    escaped. A run with one file, and Newest first (whose rows name each
    file), get no row."""
    from tests.fixtures.family_world import EVIL, file_key, run_key

    listing, view = _family_view()
    data = mp.read_page(listing)
    by_family = {"order": mp.BY_FAMILY, "task": mp.ALL_TASKS}
    last, other = file_key("alpha-base"), file_key("alpha-base", "model-03.pt")
    listed = {ml.item_key(i): i for i in listing["items"]}
    words = ml.checkpoint_choices(listed, [last, other])

    def saved(top):
        found = re.search(r'<dt class="shep-k">Saved files</dt><dd class="shep-v">(.*?)</dd>',
                          top, re.S)
        return found.group(1) if found else None

    row = saved(mp.detail_parts(listing, data, last, by_family)["top"])
    items = re.findall(r"<li[^>]*>.*?</li>", row, re.S)
    assert [_text(i).replace(f" ({mp.SHOWN_FILE})", "") for i in items] == [w for w, _k in words]
    assert 'aria-current="true"' in items[0] and "<button" not in items[0]
    assert f"({mp.SHOWN_FILE})" in _text(items[0]) and row.count("aria-current") == 1
    assert (f'<button type="button" class="shep-link" data-act="{ml.PICK_ACT}" '
            f'data-key="{other}">') in items[1]
    picked_row = saved(mp.detail_parts(listing, data, other, by_family)["top"])
    assert re.search(r'<li aria-current="true"[^>]*data-key="' + re.escape(other), picked_row)
    # One file, or Newest first: no row.
    assert saved(mp.detail_parts(listing, data, file_key("alpha-gene"), by_family)["top"]) is None
    assert saved(mp.detail_parts(listing, data, last, {"order": mp.NEWEST})["top"]) is None
    # A file named with markup: text, never HTML, its key escaped too (the family view's
    # rows name it by the same key, so the list draws it).
    evil_key = run_key("alpha-base") + "/" + EVIL + ".pt"

    def renamed(value):
        if isinstance(value, dict):
            return {k: renamed(v) for k, v in value.items()}
        if isinstance(value, list):
            return [renamed(v) for v in value]
        return evil_key if value == other else value

    hostile = {**listing, "family": renamed(listing["family"]), "items": [
        {**i, "filename": EVIL + ".pt", "key": evil_key}
        if ml.item_key(i) == other else i for i in listing["items"]]}
    row = saved(mp.detail_parts(hostile, data, last, by_family)["top"])
    assert "<img" not in row and html_lib.escape(EVIL, quote=True) in row
    # The pick: the details' region is a trigger of the list's one selection.
    demo, _ctx = _page()
    select = _fn(demo, "_select")
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    top_html = by_id["shep-mp-detail-top"]
    assert {(by_id["shep-models-research-rows"]._id, "click"),
            (top_html._id, "click")} <= set(select.targets)
    assert r.TRIGGER_JS in top_html.js_on_load and mp.PICKED_JS in top_html.js_on_load

    def answer(raw):
        out = select.fn(listing, last, False, view, "", {}, gr.EventData(None, raw),
                        gradio_request())
        by = dict(zip(map(id, select.outputs), out, strict=True))
        # The draw token asked for (its value is a time).
        assert by.pop(id(select.outputs[-1])) != gr.update()
        return by

    pick = answer({"act": ml.PICK_ACT, "key": other})
    assert pick[id(select.outputs[0])] == other and pick == answer({"key": other})


def test_the_saved_files_offer_only_what_the_list_draws_and_say_which_is_in_use(family):
    """G3, G7, I5 (the second round's reviews): "Saved files" offers the
    run's files the list draws (`model_list.drawn_files`, the files a node's
    body selects among), so each pick there is one the list's one selection
    accepts and keeps drawn, and changes nothing the list shows but the file
    selected: under a Task filter that keeps only a shown run's parent file,
    no other file (a pick of it was refused, the button doing nothing); under
    a search, what it shows (a file it hides, picked, put the "not shown by
    this filter" line above the graph and moved it); none for a selection the
    list does not draw. Each file keeps its words among all its run's files.
    Another file of the run in use for diagnosis says so on its line (the
    shown one's is the header's chip, said once)."""
    from tests.fixtures.family_world import file_key, run_key

    research = ml.PRESETS["research"]
    base = run_key("alpha-base")
    last, other = file_key("alpha-base"), file_key("alpha-base", "model-03.pt")
    demo, _ctx = _page()
    select = _fn(demo, "_select")

    def offered(listing, data, key, task=mp.ALL_TASKS, **search):
        top = mp.detail_parts(listing, data, key, {"order": mp.BY_FAMILY, "task": task},
                              **search)["top"]
        found = re.search(r'<dt class="shep-k">Saved files</dt><dd class="shep-v">(.*?)</dd>',
                          top, re.S)
        return None if found is None else [
            {"key": html_lib.unescape(re.search(r'data-key="([^"]+)"', li).group(1)),
             "words": _text(re.sub(r'<span class="shep-(?:muted|tag)[^"]*">.*?</span>', "", li)),
             "button": "<button" in li, "in_use": "shep-tag-accent" in li}
            for li in re.findall(r"<li[^>]*>.*?</li>", found.group(1), re.S)]

    def drawn_by_tree(listing, view, search=""):
        # The run's files as the list's tree draws them (the oracle, read apart).
        return next((run.files for group in ml.tree_view(listing, view, research,
                                                          search=search).groups
                     for run in group.runs if run.key == base), ())

    listing = _family_view()[0]
    data = mp.read_page(listing)
    listed = {ml.item_key(i): i for i in listing["items"]}
    words = {k: w for w, k in ml.checkpoint_choices(listed, [last, other])}
    seen = {}
    for _label, task in mp.TASK_CHOICES:
        view = mp.shape(listing, data, mp.controls(data, mp.BY_FAMILY, task))
        drawn = drawn_by_tree(listing, view)
        seen[task] = drawn
        for key in drawn:
            row = offered(listing, data, key, task)
            if len(drawn) < 2:
                assert row is None, (task, key)
                continue
            assert [f["key"] for f in row] == list(drawn), (task, key)
            assert [f["words"].strip() for f in row] == [words[k] for k in drawn], (task, key)
            for f in row:
                assert f["button"] == (f["key"] != key), (task, key)
                if f["button"]:
                    out = select.fn(listing, key, False, view, "", {},
                                    gr.EventData(None, {"act": ml.PICK_ACT, "key": f["key"]}),
                                    gradio_request())
                    assert out[0] == f["key"], (task, key, f["key"])     # accepted
    # A Task filter that keeps the base run for one shown run's parent file draws that one.
    assert seen[mp.ALL_TASKS] == (last, other) and seen["gene"] == (last,)
    assert offered(listing, data, last, "gene") is None
    # A search: what it shows; a selection it hides gets no row; cleared, both again.
    plain = mp.shape(listing, data, mp.controls(data, mp.BY_FAMILY, mp.ALL_TASKS))
    assert drawn_by_tree(listing, plain, "model-03") == (other,)
    assert offered(listing, data, other, search="model-03") is None
    assert offered(listing, data, last, search="model-03") is None
    assert [f["key"] for f in offered(listing, data, last, search="")] == [last, other]
    assert all(not f["in_use"] for f in offered(listing, data, last))
    # The one lookup, shared with a node's body (`run_choice`).
    for search in ("", "model-03", "ext"):
        assert ml.drawn_files(listing, plain, research, base, search=search) == drawn_by_tree(
            listing, plain, search)
    # Another file of the run in use: said on its line; the shown one's is the header's.
    family["served"]["value"] = _status(family["root"] / "demo",
                                        family["shas"][("alpha-base", "model-03.pt")])
    listing = _family_view()[0]
    data = mp.read_page(listing)
    row = offered(listing, data, last)
    assert [(f["key"], f["in_use"]) for f in row] == [(last, False), (other, True)]
    in_use = re.search(r"<li>.*?</li>", re.search(
        r'<dt class="shep-k">Saved files</dt><dd class="shep-v">(.*?)</dd>',
        mp.detail_parts(listing, data, last, {"order": mp.BY_FAMILY})["top"], re.S).group(1),
        re.S).group(0)
    assert r.chip(research.in_use_label, "accent") in in_use
    assert not any(f["in_use"] for f in offered(listing, data, other))
    assert "In use (Diagnosis)" in mp.detail_parts(listing, data, other,
                                                   {"order": mp.BY_FAMILY})["header"]


def test_a_selection_in_a_folded_branch_keeps_its_details_and_is_said(family):
    """I10: a selected model whose run is in a folded branch keeps its
    selection and details; the graph says where it is ("in a folded branch
    under" its folded ancestor, with "Show it"), never selecting another."""
    from tests.fixtures.family_world import file_key, run_key

    listing, view = _family_view()
    picked = file_key("alpha-ext-3")
    fold = {ml.subtree_node(run_key("alpha-disease")): False}
    html = ml.list_html(listing, picked, ml.PRESETS["research"], view=view, fold=fold)
    note = html[html.index('class="shep-list-outside shep-fam-hid"'):]
    note = note[:note.index("</div>")]
    assert " hidden" not in note.split(">")[0] and "alpha-disease" in _text(note)
    assert "Show it" in note and ml.FOLDED_SELECTION in _text(note)
    # Said on the folded node too, under its text and its arrow (where the fold is: the
    # note above may be out of view); every other node's words hidden.
    holder = _node(html, run_key("alpha-disease"))
    assert "shep-fam-holds" in holder
    held = re.findall(r'<span class="shep-fam-held"( hidden)?>([^<]*)</span>', html)
    assert [h for h in held if not h[0]] == [("", ml.HOLDS_SELECTION)] and len(held) > 1
    assert holder.index('class="shep-fam-held"') > holder.index('class="shep-fam-act"')
    assert 'data-fam-locate="' + run_key("alpha-ext-3") + '"' in html
    open_html = ml.list_html(listing, picked, ml.PRESETS["research"], view=view, fold={})
    hid = open_html[open_html.index('class="shep-list-outside shep-fam-hid"'):].split(">")[0]
    assert " hidden" in hid and "shep-fam-holds" not in open_html
    assert all(h[0] for h in re.findall(r'<span class="shep-fam-held"( hidden)?>', open_html))
    data = mp.read_page(listing)
    assert f'data-key="{picked}"' in mp.detail_parts(listing, data, picked, {})["header"]


def test_what_the_graph_is_sized_from_ignores_folds_picks_and_the_model_in_use(family):
    """I10, I5 (the server's half): the graph says what its size is drawn from
    (``data-shape``: its runs, their places and words), the same whatever is
    folded, selected or in use, so the page decides its orientation again only
    for a new graph (here a search's), a new width or new fonts, never for a
    pick or a fold; the drawing's own token (``data-draw``) still changes with
    each of them, but not for a focused run an earlier page left in the fold
    state (``f``, ignored)."""
    from tests.fixtures.family_world import file_key, run_key

    listing, view = _family_view()
    research = ml.PRESETS["research"]

    def tokens(selected=None, fold=None, search=""):
        html = ml.list_html(listing, selected, research, view=view, fold=fold or {},
                            search=search)
        return (re.search(r'data-shape="([0-9a-f]+)"', html).group(1),
                re.search(r'data-draw="([0-9a-f]+)"', html).group(1))

    plain = tokens()
    states = [tokens(file_key("alpha-gene")),
              tokens(fold={ml.subtree_node(run_key("alpha-disease")): True})]
    assert {shape for shape, _draw in states} == {plain[0]}
    assert len({draw for _shape, draw in states} | {plain[1]}) == 3
    assert tokens(fold={"f": run_key("beta-base")}) == plain
    family["served"]["value"] = _status(family["root"] / "demo",
                                        family["shas"][("beta-base", "last.pt")])
    listing, view = _family_view()
    assert tokens()[0] == plain[0]                      # the model in use moved: same graph
    assert tokens(search="ext-5")[0] != plain[0]        # a search draws another graph


def test_a_node_keeps_its_start_time_whole_and_its_arrow_names_its_child_runs(family):
    """P3 of the interaction review: a node's start time is one unbreakable
    "date time" (a narrow node never leaves "00:00" alone on a line); the
    arrow's name says what it shows or hides, its child runs counted
    (``aria-expanded`` saying which)."""
    from tests.fixtures.family_world import run_key

    listing, view = _family_view()
    html = ml.list_html(listing, None, ml.PRESETS["research"], view=view, fold={})
    base = _node(html, run_key("alpha-base"))
    assert re.search(r'<span class="shep-nowrap">2026-11-01\xa000:00</span>', base), base
    label = re.search(r'class="shep-fam-tog"[^>]*aria-label="([^"]+)"', base).group(1)
    assert label == "Child runs of alpha-base (4)"


def test_a_cycle_in_a_hand_made_view_draws_each_run_once(family):
    """G04, S03: the graph follows no cycle a hand-made view could give (a run
    under its own descendant, a run under its own file): each run is drawn
    once, every family still drawn, nothing lost."""
    from tests.fixtures.family_world import RUNS, file_key, run_key

    listing, view = _family_view()
    looped = {**view, "edges": [{"from": file_key("alpha-ext-5"), "to": run_key("alpha-base")},
                                {"from": file_key("beta-base"), "to": run_key("beta-base")},
                                *view["edges"]]}
    html = ml.list_html(listing, None, ml.PRESETS["research"], view=looped, fold={})
    drawn = re.findall(r'<li class="shep-fam-li" data-run="([^"]+)"', html)
    assert sorted(drawn) == sorted(run_key(label) for label, *_rest in RUNS)


#: The family graph served whole (By family on the synthetic family world), by
#: real pointer moves, presses, keys and taps; what each step requested
#: (the queue's joins) and what the page drew, read where the browser draws it.
_FAMILY_PAGE_JS = r"""
const { chromium } = require('playwright');
const QUEUE_JS = __QUEUE_JS__, RUN = __RUNS__;
const ROWS = '#shep-models-research-rows';
(async () => {
  const base = process.argv[2];
  const browser = await chromium.launch();
  const open = async (options, newest) => {
    const ctx = await browser.newContext({ viewport: { width: 1850, height: 960 }, deviceScaleFactor: 1, ...options });
    const page = await ctx.newPage();
    await page.addInitScript(QUEUE_JS);
    const joins = [];
    page.on('request', (r) => { if (r.url().includes('/queue/join')) { joins.push((r.postDataJSON() || {}).event_data); } });
    const idle = () => page.waitForFunction(() => window.__shepQueue.idle(), null, { timeout: 30000 });
    await page.goto(base, { waitUntil: 'domcontentloaded' });
    await page.click('#shep-tab-research-button', { timeout: 60000 });
    await page.click('#shep-page-models-button');
    await page.waitForFunction(() => { const m = document.querySelector('#shep-models-research-rows .shep-view-for'); return !!m && m.getAttribute('data-for') === 'newest|all'; }, null, { timeout: 30000 });
    await idle();
    if (newest) { await newest(page, idle); }
    await page.click('#shep-mp-order label:has-text("By family")');
    await page.waitForFunction(() => { const m = document.querySelector('#shep-models-research-rows .shep-view-for'); return !!m && m.getAttribute('data-for') === 'family|all'; }, null, { timeout: 30000 });
    await idle();
    return { ctx, page, joins, idle };
  };
  const node = (run) => `${ROWS} li[data-run="${run}"] > .shep-fam-node`;
  const look = (page, run) => page.evaluate((sel) => {
    const n = document.querySelector(sel);
    if (!n) { return null; }
    const box = (e) => { if (!e) { return null; } const r = e.getBoundingClientRect(); return { x: r.x, y: r.y, w: r.width, h: r.height, right: r.right, bottom: r.bottom }; };
    const body = n.querySelector('.shep-fam-body'), act = n.querySelector('.shep-fam-act'), tog = n.querySelector('.shep-fam-tog');
    const range = document.createRange();
    range.selectNodeContents(body);
    const lines = new Set([...range.getClientRects()].filter((r) => r.width > 0).map((r) => Math.round(r.top))).size;
    const kids = tog ? document.getElementById(tog.getAttribute('aria-controls')) : null;
    // The fill a reader sees at a point: the first one drawn up from what is there.
    const fill = (e) => { for (; e; e = e.parentElement) { const c = getComputedStyle(e).backgroundColor; if (c !== 'rgba(0, 0, 0, 0)' && c !== 'transparent') { return c; } } return null; };
    const at = (x, y) => document.elementFromPoint(x, y);
    const a = act ? act.getBoundingClientRect() : null;
    // Inside each edge of the arrow area (clear of its rounded corners): the arrow there too.
    const inner = a && a.height > 0 ? [[a.left + 2, a.top + a.height / 2], [a.right - 2, a.top + a.height / 2], [a.left + a.width / 2, a.top + 2], [a.left + a.width / 2, a.bottom - 2]] : [];
    const centre = (r) => [r.left + r.width / 2, r.top + r.height / 2];
    // Down the arrow area's middle, every point: what is drawn there (a line over it would not be the node's).
    const column = [];
    if (a && a.height > 0) { for (let y = Math.ceil(a.top) + 1; y < a.bottom - 1; y += 1) { column.push(at(a.left + a.width / 2, y)); } }
    return { node: box(n), body: box(body), act: box(act), tog: box(tog), lines,
             fillBody: fill(at(...centre(body.getBoundingClientRect()))), fillAct: a && a.height > 0 ? fill(at(...centre(a))) : null,
             edgesOnTog: inner.length ? inner.every(([x, y]) => { const e = at(x, y); return !!e && tog.contains(e); }) : null,
             columnInNode: column.length ? column.every((e) => !!e && n.contains(e)) : null,
             actOpacity: act ? getComputedStyle(act).opacity : null,
             actTransition: act ? getComputedStyle(act).transitionDuration : null,
             expanded: tog ? tog.getAttribute('aria-expanded') : null,
             kidsShown: kids ? !kids.hidden && kids.getBoundingClientRect().height > 0 : null,
             shown: n.getBoundingClientRect().height > 0, focused: n.contains(document.activeElement),
             active: document.activeElement ? (document.activeElement.className || '').toString() : '' };
  }, node(run));
  const state = (page) => page.evaluate((rows) => {
    const el = document.querySelector(rows);
    const holder = document.querySelector('[data-fam-orient]');
    const nodes = [...el.querySelectorAll('.shep-fam-node')].filter((n) => n.getBoundingClientRect().height > 0);
    const rects = nodes.map((n) => n.getBoundingClientRect());
    let overlaps = 0;
    for (let i = 0; i < rects.length; i++) { for (let j = i + 1; j < rects.length; j++) {
      const a = rects[i], b = rects[j];
      if (a.left < b.right - 0.5 && b.left < a.right - 0.5 && a.top < b.bottom - 0.5 && b.top < a.bottom - 0.5) { overlaps += 1; }
    } }
    const idle = nodes.filter((n) => n.querySelector('.shep-fam-body')).map((n) => { const b = n.querySelector('.shep-fam-body').getBoundingClientRect(), r = n.getBoundingClientRect();
      return { parent: n.classList.contains('shep-fam-parent'), dh: r.height - b.height, dw: r.width - b.width }; });
    const se = document.scrollingElement;
    const hid = el.querySelector('.shep-fam-hid');
    return {
      orient: holder ? holder.getAttribute('data-fam-orient') : null,
      selected: (document.querySelector('#shep-mp-detail-hd h2[data-key]') || {getAttribute: () => null}).getAttribute('data-key'),
      folds: Object.fromEntries([...el.querySelectorAll('button.shep-fam-tog')].map((b) => [b.getAttribute('data-tree'), b.getAttribute('aria-expanded')])),
      visible: nodes.map((n) => n.parentElement.getAttribute('data-run')).filter(Boolean),
      overlaps, idle, pageOver: se.scrollWidth - se.clientWidth,
      hidNote: hid && !hid.hidden ? hid.innerText.replace(/\s+/g, ' ').trim() : null,
      holds: [...el.querySelectorAll('.shep-fam-holds')].map((n) => n.parentElement.getAttribute('data-run')),
      panel: el.querySelectorAll('.shep-fam-run, .shep-fam-back').length,
      currents: [...el.querySelectorAll('[aria-current="true"]')].map((b) => b.getAttribute('data-run') || b.getAttribute('data-key')),
      saved: [...document.querySelectorAll('#shep-mp-detail-top .shep-mp-files > li')].map((l) => [l.getAttribute('aria-current') === 'true', (l.getAttribute('data-key') || (l.querySelector('[data-key]') || {getAttribute: () => null}).getAttribute('data-key'))]),
      room: (el.querySelector('.shep-fam-forest') || {style: {}}).style.marginTop || '',
      held: [...el.querySelectorAll('.shep-fam-held')].filter((h) => !h.hidden && h.getBoundingClientRect().height > 0).map((h) => [h.closest('li').getAttribute('data-run'), h.textContent]),
      scroll: { top: el.scrollTop, left: el.scrollLeft },
      count: document.querySelector('#shep-models-research-count').innerText.trim(),
    };
  }, ROWS);
  const out = {};
  // ---- A mouse, 1850 x 960: idle, then the pointer from a body onto its arrow.
  let { ctx, page, joins, idle } = await open({});
  out.idle = await state(page);
  await page.$eval(node(RUN.base), (n) => n.scrollIntoView({ block: 'center' }));
  const before = await look(page, RUN.base);
  const b = before.body, t = before.tog;
  const start = { x: b.x + b.w / 2, y: b.y + b.h / 2 };
  await page.mouse.move(start.x, start.y);
  await page.waitForFunction((sel) => getComputedStyle(document.querySelector(sel + ' > .shep-fam-act')).opacity === '1', node(RUN.base));
  const shown = await look(page, RUN.base);
  const end = { x: shown.tog.x + shown.tog.w / 2, y: shown.tog.y + shown.tog.h / 2 };
  const path = [];
  joins.length = 0;
  for (let i = 0; i <= 16; i++) {
    const x = start.x + (end.x - start.x) * i / 16, y = start.y + (end.y - start.y) * i / 16;
    await page.mouse.move(x, y);
    path.push(await page.evaluate(([x, y, sel]) => {
      const at = document.elementFromPoint(x, y), n = document.querySelector(sel);
      return { same: !!at && n.contains(at), opacity: getComputedStyle(n.querySelector('.shep-fam-act')).opacity };
    }, [x, y, node(RUN.base)]));
  }
  out.hover = { before, shown, after: await look(page, RUN.base), path, joins: joins.length,
                others: await page.evaluate(([sel]) => {
                  const act = document.querySelector(sel + ' > .shep-fam-act').getBoundingClientRect();
                  return [...document.querySelectorAll('.shep-fam-node')].filter((n) => n !== document.querySelector(sel) && n.getBoundingClientRect().height > 0)
                    .filter((n) => { const r = n.getBoundingClientRect(); return act.left < r.right && r.left < act.right && act.top < r.bottom && r.top < act.bottom; }).length;
                }, [node(RUN.base)]) };
  // One press is one toggle and one request, nothing selected or loaded.
  const selected0 = (await state(page)).selected;
  joins.length = 0;
  await page.mouse.down(); await page.mouse.up();
  await idle();
  out.press = { joins: joins.slice(), look: await look(page, RUN.base), selected: (await state(page)).selected, selected0 };
  await page.mouse.down(); await page.mouse.up();         // and back
  await idle();
  // Out of the node, focus elsewhere: the arrow area goes.
  await page.mouse.move(5, 5);
  await page.evaluate(() => document.activeElement && document.activeElement.blur());
  await page.waitForFunction((sel) => getComputedStyle(document.querySelector(sel + ' > .shep-fam-act')).opacity === '0', node(RUN.base));
  out.left = await look(page, RUN.base);
  // ---- The keyboard: Tab onto the body, then the arrow; Enter and Space each one toggle.
  const opaque = (sel) => page.waitForFunction((x) => getComputedStyle(document.querySelector(x + ' > .shep-fam-act')).opacity === '1', sel);
  await page.focus(node(RUN.base) + ' .shep-fam-body');
  await page.keyboard.press('Shift+Tab');
  await page.keyboard.press('Tab');                     // the keyboard's own focus, onto the body
  await opaque(node(RUN.base));
  out.keys = { body: await look(page, RUN.base) };
  joins.length = 0;
  await page.keyboard.press('Tab');
  await opaque(node(RUN.base));
  out.keys.arrow = await look(page, RUN.base);
  await page.keyboard.press('Enter'); await idle();
  out.keys.enter = await look(page, RUN.base);
  await page.keyboard.press(' '); await idle();
  out.keys.space = await look(page, RUN.base);
  out.keys.joins = joins.slice();
  // Enter on a body selects its run, and Space on another's; the focus stays on the body.
  const activeRun = () => page.evaluate(() => { const a = document.activeElement; return (a.className || '') + '|' + ((a.closest('li') || {getAttribute: () => ''}).getAttribute('data-run')); });
  const selectedIs = (run) => page.waitForFunction((r) => { const h = document.querySelector('#shep-mp-detail-hd h2[data-key]'); return !!h && h.getAttribute('data-key').startsWith(r + '/'); }, run);
  joins.length = 0;
  await page.focus(node(RUN.base) + ' .shep-fam-body');
  await page.keyboard.press('Enter');
  await selectedIs(RUN.base); await idle();
  out.keys.enterSelect = { joins: joins.filter(Boolean), state: await state(page), active: await activeRun() };
  joins.length = 0;
  await page.focus(node(RUN.top) + ' .shep-fam-body');
  await page.keyboard.press(' ');
  await selectedIs(RUN.top); await idle();
  out.keys.spaceSelect = { joins: joins.filter(Boolean), state: await state(page), active: await activeRun() };
  // ---- A deep subtree: unfolded twice down, folded at the top, unfolded again (I9). In a window
  // whose list region is as wide as the owner's was when the list took 46 parts of 100 (it takes
  // two thirds since the sweep's L7, SW-LK4), so the unfolded chain runs past it.
  await page.setViewportSize({ width: 1292, height: 960 });
  await page.waitForFunction(() => document.querySelector('[data-fam-orient]').getAttribute('data-fam-orient') === 'h');
  await idle();
  const press = async (run) => { await page.hover(node(run) + ' .shep-fam-body'); await page.click(node(run) + ' .shep-fam-tog'); await idle(); };
  // Unfolded past the region's width left to right: the graph keeps its orientation and the
  // arrow stays under the pointer (I5, I10); the region, never the page, pans to what it shows.
  await page.$eval(node(RUN.disease), (n) => n.scrollIntoView({ block: 'center', inline: 'nearest' }));
  await page.hover(node(RUN.disease) + ' .shep-fam-body');
  const dTog = (await look(page, RUN.disease)).tog, dAt = [dTog.x + dTog.w / 2, dTog.y + dTog.h / 2];
  const onTog = () => page.evaluate(([x, y, sel]) => { const e = document.elementFromPoint(x, y); return !!e && document.querySelector(sel + ' .shep-fam-tog').contains(e); }, [...dAt, node(RUN.disease)]);
  const dBefore = await state(page);
  await page.mouse.click(...dAt); await idle();
  const dAfter = await state(page);
  out.deepPress = { before: dBefore.orient, after: dAfter.orient, onTog: await onTog(), expanded: (await look(page, RUN.disease)).expanded,
                    pageOver: dAfter.pageOver, wider: await page.evaluate((rows) => { const e = document.querySelector(rows); return e.scrollWidth > e.clientWidth; }, ROWS) };
  await press(RUN.ext1); await press(RUN.ext2);
  out.deep = (await state(page)).visible;
  joins.length = 0;
  await press(RUN.disease);
  out.folded = { visible: (await state(page)).visible, joins: joins.slice() };
  await press(RUN.disease);
  out.unfolded = await state(page);
  // ---- A body clicked by the pointer: its run selected, nothing moved under the pointer.
  // `place`: 'center' (the node centred in the region), or 'cut' (its lower half below the
  // region's bottom edge, the middle of the part in view clicked: a drawing located again
  // would scroll the node into view).
  const pickBody = async (run, place) => {
    let cut = null;
    if (place === 'center') { await page.$eval(node(run), (n) => n.scrollIntoView({ block: 'center', inline: 'center' })); }
    if (place === 'cut') {
      cut = await page.$eval(ROWS, (el, sel) => {
        const b = document.querySelector(sel + ' .shep-fam-body');
        b.scrollIntoView({ block: 'nearest', inline: 'center' });
        const v = el.getBoundingClientRect(), r0 = b.getBoundingClientRect();
        el.scrollTop += (r0.top + r0.height / 2) - v.bottom;
        const r = b.getBoundingClientRect();
        return { below: r.bottom - v.bottom, above: v.bottom - r.top, edge: v.bottom };
      }, node(run));
    }
    const box = (await look(page, run)).body;
    const at = [box.x + box.w / 2, (box.y + (cut ? Math.min(box.bottom, cut.edge) : box.bottom)) / 2];
    const before = await state(page);
    joins.length = 0;
    await page.mouse.click(...at);
    await selectedIs(run); await idle();
    return { joins: joins.filter(Boolean), before, cut, state: await state(page), moved: (await look(page, run)).body.y - box.y,
             same: await page.evaluate(([x, y, sel]) => { const e = document.elementFromPoint(x, y); return !!e && document.querySelector(sel + ' .shep-fam-body').contains(e); }, [...at, node(run)]) };
  };
  // A run with two saved files, half below the region's bottom edge: its latest save, the
  // region not scrolled; the other one picked in the details' "Saved files" (the graph not
  // moved, the focus on the line that names it, its ring as wide as its words).
  out.multi = await pickBody(RUN.base, 'cut');
  const other = out.multi.state.saved.find(([cur]) => !cur);
  const sBox = (await look(page, RUN.base)).body;
  joins.length = 0;
  await page.click(`#shep-mp-detail-top .shep-mp-files button[data-key="${other ? other[1] : ''}"]`);
  await page.waitForFunction((k) => { const h = document.querySelector('#shep-mp-detail-hd h2[data-key]'); return !!h && h.getAttribute('data-key') === k; }, other ? other[1] : '');
  await idle();
  out.savedPick = { joins: joins.filter(Boolean), before: out.multi.state, state: await state(page),
                    moved: (await look(page, RUN.base)).body.y - sBox.y,
                    active: await page.evaluate(() => { const a = document.activeElement; return [a.tagName, a.getAttribute('aria-current'), a.getAttribute('data-key')]; }),
                    fit: await page.evaluate(() => { const a = document.activeElement, rg = document.createRange(); rg.selectNodeContents(a); return a.getBoundingClientRect().width - rg.getBoundingClientRect().width; }) };
  // Searched for the file shown: "Saved files" offers what the list draws (here that one
  // file: no row); cleared, both again.
  await page.click('#shep-models-research-search input, #shep-models-research-search textarea');
  await page.keyboard.type('model-03');
  await page.waitForFunction(() => document.querySelector('#shep-models-research-count').innerText.includes('model-03'));
  await idle();
  out.savedSearched = await state(page);
  await page.click('#shep-models-research-search-clear');
  await page.waitForFunction(() => !document.querySelector('#shep-models-research-count').innerText.includes('“'));
  await idle();
  out.savedCleared = await state(page);
  out.pick = await pickBody(RUN.ext1, 'center');
  // Folded over the selection: said on the folded node and above the graph; the focus to the arrow.
  await page.focus(node(RUN.ext1) + ' .shep-fam-body');
  await page.evaluate((sel) => document.querySelector(sel + ' .shep-fam-tog').click(), node(RUN.disease));
  await idle();
  out.hidden = { state: await state(page), active: await page.evaluate(() => document.activeElement.className + '|' + (document.activeElement.closest('li') || {getAttribute: () => ''}).getAttribute('data-run')),
                 // The folded node's words in the colour of its own lines, never a link's.
                 colors: await page.evaluate((sel) => { const n = document.querySelector(sel); return [n.querySelector(':scope > .shep-fam-held'), n.querySelector('.shep-fam-ln')].map((x) => getComputedStyle(x).color); }, node(RUN.disease)) };
  joins.length = 0;
  await page.click(`${ROWS} button.shep-fam-show`);
  await idle();
  out.shownAgain = { state: await state(page), joins: joins.slice(), active: await page.evaluate(() => (document.activeElement.closest('li') || {getAttribute: () => ''}).getAttribute('data-run')) };
  // Folded over it again, a body below picked: the lines the old selection had above it
  // (the note, the folded node's words) leave their room, so the node stays under the pointer.
  await press(RUN.disease);
  await page.mouse.move(2, 2);
  out.room = await pickBody(RUN.frozen, 'center');
  // The pointer and then the focus gone from the rows (as a pick in "Saved files" takes it to
  // the details): the room given back as the region's scroll, nothing in view moved.
  const gBox = (await look(page, RUN.frozen)).body, gBefore = await state(page);
  await page.mouse.move(2, 2);
  await page.evaluate(() => document.querySelector('#shep-mp-detail-hd h2[data-key]').focus({ preventScroll: true }));
  await page.waitForFunction((rows) => { const f = document.querySelector(rows + ' .shep-fam-forest'); return !!f && !f.style.marginTop; }, ROWS);
  out.roomGone = { before: gBefore, state: await state(page), moved: (await look(page, RUN.frozen)).body.y - gBox.y };
  // ---- A search reveals a deeper match; clearing it brings the folds back (G7).
  out.beforeSearch = await state(page);
  await page.click('#shep-models-research-search input, #shep-models-research-search textarea');
  await page.keyboard.type('ext-3');
  await page.waitForFunction(() => document.querySelector('#shep-models-research-count').innerText.includes('ext-3'));
  await idle();
  out.searched = await state(page);
  await page.click('#shep-models-research-search-clear');
  await page.waitForFunction(() => !document.querySelector('#shep-models-research-count').innerText.includes('“'));
  await idle();
  out.cleared = await state(page);
  // ---- A pick decides no orientation again (I10): top to bottom for a narrower region with
  // the chain unfolded, then the chain folded (it would fit left to right) and a body picked.
  await press(RUN.disease);                                        // the chain unfolded again
  // A narrower window, still side by side: the list's region about as wide as at 1848 px when the
  // list took 46 parts of 100 (it takes two thirds since the sweep's L7, SW-LK4).
  await page.setViewportSize({ width: 1290, height: 960 });
  await page.waitForFunction(() => document.querySelector('[data-fam-orient]').getAttribute('data-fam-orient') === 'v');
  await press(RUN.disease);                                        // fold the chain again
  const vFolded = (await state(page)).orient;
  await page.click(node(RUN.base) + ' .shep-fam-body');
  await selectedIs(RUN.base); await idle();
  out.repick = { vFolded, state: await state(page) };
  // ---- Both orientations and the switch between them (the selection and the folds kept). The
  // chain unfolded again (a fold decides no orientation): at the display floor's width, the
  // narrowest region at or above it, the unfolded graph runs top to bottom.
  await press(RUN.disease);
  await page.setViewportSize({ width: 1849, height: 960 });
  await page.waitForFunction(() => document.querySelector('[data-fam-orient]').getAttribute('data-fam-orient') === 'h');
  const wide = await state(page);
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.waitForFunction(() => document.querySelector('[data-fam-orient]').getAttribute('data-fam-orient') === 'v');
  const narrow = await state(page);
  await page.setViewportSize({ width: 1850, height: 960 });
  await page.waitForFunction(() => document.querySelector('[data-fam-orient]').getAttribute('data-fam-orient') === 'h');
  out.switch = { wide, narrow, back: await state(page) };
  // In the narrow layout, the arrow area under the body: the node grows down, its top and text kept.
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.waitForFunction(() => document.querySelector('[data-fam-orient]').getAttribute('data-fam-orient') === 'v');
  await page.mouse.move(2, 2);
  await page.$eval(node(RUN.base), (n) => n.scrollIntoView({ block: 'center' }));
  const vBefore = await look(page, RUN.base);
  const nextBefore = await look(page, RUN.branch);
  await page.hover(node(RUN.base) + ' .shep-fam-body');
  await page.waitForFunction((sel) => document.querySelector(sel + ' > .shep-fam-act').getBoundingClientRect().height > 0, node(RUN.base));
  out.vertical = { before: vBefore, after: await look(page, RUN.base), next: nextBefore, nextAfter: await look(page, RUN.branch) };
  await page.mouse.move(2, 2);
  // ---- Larger type: the page at 150 % (its CSS pixels a third fewer; a 2560 x 1440 window, so
  // the area stays above the display floor). The body reached by
  // the keyboard (a focus set by the page after a pointer is no keyboard focus): from it Tab
  // onto its arrow, Shift+Tab back. Its ring read where it is drawn (2 px inside the body's
  // edge, `outline-offset: -3px`): what is there on each side is the body.
  await page.setViewportSize({ width: 1707, height: 960 });
  await page.waitForTimeout(0);
  await page.focus(node(RUN.base) + ' .shep-fam-body');
  await page.keyboard.press('Tab');
  await page.keyboard.press('Shift+Tab');
  out.large = { state: await state(page), ring: await page.evaluate(() => {
    const a = document.activeElement, cs = getComputedStyle(a), r = a.getBoundingClientRect();
    const sides = [[r.left + 2, r.top + r.height / 2], [r.right - 2, r.top + r.height / 2],
                   [r.left + r.width / 2, r.top + 2], [r.left + r.width / 2, r.bottom - 2]];
    return { body: a.classList.contains('shep-fam-body'), keyboard: a.matches(':focus-visible'),
             style: cs.outlineStyle, width: cs.outlineWidth,
             clear: sides.map(([x, y]) => { const e = document.elementFromPoint(x, y); return !!e && a.contains(e); }) };
  }) };
  // ---- Newest first draws the table again.
  await page.setViewportSize({ width: 1850, height: 960 });
  await page.click('#shep-mp-order label:has-text("Newest first")');
  await page.waitForFunction(() => { const m = document.querySelector('#shep-models-research-rows .shep-view-for'); return !!m && m.getAttribute('data-for') === 'newest|all'; }, null, { timeout: 30000 });
  out.newest = await page.evaluate((rows) => ({ table: !!document.querySelector(rows + ' tr.shep-tree-run'), graph: !!document.querySelector(rows + ' .shep-fam') }), ROWS);
  await ctx.close();
  // ---- A selection made elsewhere (Newest first: alpha-ext-3) unfolds its chain in By family;
  // with the region at its end, a pick in the chain keeps it unfolded, nothing moved (I5).
  ({ ctx, page, joins, idle } = await open({}, async (p, ready) => {
    await p.click(`${ROWS} tr.shep-tree-run[data-fold="r:${RUN.ext3}"] button.shep-fold`);
    await p.click(`${ROWS} tr[data-key^="${RUN.ext3}/"]`);
    await p.waitForFunction((r) => { const h = document.querySelector('#shep-mp-detail-hd h2[data-key]'); return !!h && h.getAttribute('data-key').startsWith(r + '/'); }, RUN.ext3);
    await ready();
  }));
  await page.$eval(ROWS, (e) => { e.scrollTop = e.scrollHeight; });
  await page.waitForTimeout(0);
  out.chain = { shown: await state(page) };
  const cBody = (await look(page, RUN.ext2)).body, cAt = [cBody.x + cBody.w / 2, cBody.y + cBody.h / 2];
  joins.length = 0;
  await page.mouse.click(...cAt);
  await page.waitForFunction((r) => { const h = document.querySelector('#shep-mp-detail-hd h2[data-key]'); return !!h && h.getAttribute('data-key').startsWith(r + '/'); }, RUN.ext2);
  await idle();
  out.chain.picked = await state(page);
  out.chain.moved = (await look(page, RUN.ext2)).body.y - cBody.y;
  out.chain.same = await page.evaluate(([x, y, sel]) => { const e = document.elementFromPoint(x, y); return !!e && document.querySelector(sel + ' .shep-fam-body').contains(e); }, [...cAt, node(RUN.ext2)]);
  out.chain.joins = joins.filter(Boolean);
  await ctx.close();
  // ---- A touch screen: the arrows there with no hover; a tap toggles once.
  ({ ctx, page, joins, idle } = await open({ hasTouch: true, isMobile: true, viewport: { width: 1280, height: 900 } }));
  const touched = await look(page, RUN.base);
  joins.length = 0;
  await page.tap(node(RUN.base) + ' .shep-fam-tog');
  await idle();
  out.touch = { before: touched, after: await look(page, RUN.base), joins: joins.slice() };
  await ctx.close();
  // ---- Reduced motion: the arrow area shows with no transition.
  ({ ctx, page, joins, idle } = await open({ reducedMotion: 'reduce' }));
  out.reduced = await look(page, RUN.base);
  await ctx.close();
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def _acts_said_with(seen, generation):
    """`seen` with each act the page sent (its ``joins``) checked to carry the
    generation of the listing its rows were drawn from (``gen``, the rows'
    `model_list.generation_marker`; Models' "Saved files" say it too), then
    without it, so an act reads as what it asks."""
    if isinstance(seen, list):
        return [_acts_said_with(v, generation) for v in seen]
    if not isinstance(seen, dict):
        return seen
    out = {}
    for key, value in seen.items():
        if key == "joins" and isinstance(value, list):
            for act in value:
                assert not isinstance(act, dict) or act.get("gen") == generation, act
            value = [{k: v for k, v in act.items() if k != "gen"} if isinstance(act, dict)
                     else act for act in value]
        out[key] = _acts_said_with(value, generation)
    return out


def test_the_family_graph_in_a_browser(tmp_path, family):
    """G6-G8, I1-I10 on the page as the dashboard mounts it, by real events
    (relative geometry only, no pixel figure: the stage-lineage package §6):

    - **idle** (I01): every node is its body's box and borders, a node with
      child runs as tight as a leaf; no node overlaps another;
    - **the pointer from a node's body onto its arrow** (I02-I05): the arrow
      area shows on the body and stays along the whole path, which never
      leaves the node; the body's box and lines are unchanged; left to right
      the area lies to the body's right and covers no other node, no line
      drawn over it, the body's fill its fill, the whole area the arrow;
      nothing is requested by a hover (S03);
    - **one press, one toggle, one request** (I06, I09): ``{tree: ...}`` and
      nothing selected; the area goes when the pointer and the focus leave;
    - **the keyboard** (I05): Tab reaches the body and then the arrow, the
      area shown while the focus is in it; Enter and Space each toggle once;
      Enter or Space on a body selects its run (``{run}``, one request), the
      focus kept on the body;
    - **folding a subtree** (I07, I9): hides every run under it and clears
      their entries; unfolding shows its direct children only; a subtree
      unfolded past the region's width keeps the orientation and the arrow
      under the pointer, the region (never the page) panning to it;
    - **a body selects its run** (I8, G2, G3, I5): a pointer's click sends
      ``{run}`` once and selects the run's latest save, the details following
      with its "Saved files" where it holds several; the other file picked
      there (``{act: "pick", key}``) is the selection, the focus on the line
      that now names it (its ring as wide as its words); neither moves the
      graph (the node under the pointer, the region's scroll and the
      orientation kept, the node picked half below the region's edge too),
      and one run reads selected, no run panel; a search hides the run's
      other file, and "Saved files" no longer offers it; folded away, the
      selection and its details stay, the graph says where it is, on the
      folded node too (in its lines' colour: no link), and the focus moves to
      the arrow; "Show it" unfolds the way; folded again, a body picked below
      keeps its place though the old selection's lines go, and once the
      pointer and the focus leave, their room is given back as the region's
      scroll, nothing in view moved;
    - **a selection made elsewhere** (Newest first): its chain unfolded in
      By family; a pick in it, the region at its end, keeps the chain
      unfolded and the node under the pointer;
    - **a search** (G7): reveals a deeper match, clearing brings the folds back;
    - **orientation** (I10): decided again by a new width, never by a fold or
      a pick; left to right wide, top to bottom narrow, and back, the
      selection and folds kept; top to bottom the arrow area is under the
      body, the whole band the arrow, the node's top and text kept, the next
      node moved down;
    - **larger type** (I11): no sideways page scroll; a body reached by the
      keyboard (Tab onto its arrow, Shift+Tab back: a focus the page sets
      after a pointer is no keyboard focus, so ``:focus-visible`` is not
      set) has its focus ring drawn and covered by nothing;
    - **touch** (I05): arrows shown with no hover, a tap one toggle;
    - **reduced motion** (I09): no transition; Newest first draws the table."""
    from src.webui.app import create_gradio_app, mount_options
    from tests.fixtures.browser import QUEUE_JS, node_with_playwright, run_node
    from tests.fixtures.family_world import run_key
    from tests.fixtures.served_page import served

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    runs = {"top": run_key("beta-base"), "base": run_key("alpha-base"),
            "branch": run_key("alpha-branch"),
            "disease": run_key("alpha-disease"), "ext1": run_key("alpha-ext-1"),
            "ext2": run_key("alpha-ext-2"), "ext3": run_key("alpha-ext-3"),
            "frozen": run_key("alpha-frozen")}
    script = (_FAMILY_PAGE_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS))
              .replace("__RUNS__", json.dumps(runs)))
    with served(create_gradio_app(), **mount_options()) as base:
        seen = run_node(script, tmp_path, base[:-len("gradio_api")], timeout=600)
    # Each page opened is a session of its own, its one workspace read once.
    seen = _acts_said_with(seen, "1")
    t = ml.subtree_node
    idle = seen["idle"]
    assert idle["orient"] == "h" and idle["overlaps"] == 0 and idle["pageOver"] <= 0
    assert any(n["parent"] for n in idle["idle"]) and any(not n["parent"] for n in idle["idle"])
    # A node with child runs is its body's box as a leaf is: no band, no column kept (I01).
    leaf = next(n for n in idle["idle"] if not n["parent"])
    assert all(abs(n["dh"] - leaf["dh"]) < 0.5 and abs(n["dw"] - leaf["dw"]) < 0.5
               for n in idle["idle"]), idle["idle"]
    hover = seen["hover"]
    assert hover["before"]["actOpacity"] == "0" and hover["shown"]["actOpacity"] == "1"
    assert all(p["same"] and p["opacity"] == "1" for p in hover["path"]), hover["path"]
    for side in ("x", "y", "w", "h"):
        assert abs(hover["after"]["body"][side] - hover["before"]["body"][side]) < 0.5, side
    assert hover["after"]["lines"] == hover["before"]["lines"]
    assert hover["shown"]["act"]["x"] >= hover["shown"]["body"]["right"] - 1.5
    assert hover["others"] == 0 and hover["joins"] == 0
    # One block (I2): the body's fill is the arrow area's; the whole area is the arrow
    # (P3-5); no line drawn over it (I04: the line ends at the visible frame).
    shown = hover["shown"]
    assert shown["fillBody"] == shown["fillAct"] and shown["fillBody"] is not None, shown
    assert shown["edgesOnTog"] and shown["columnInNode"], shown
    press = seen["press"]
    assert press["joins"] == [{"tree": {t(runs["base"]): False}, "searched": False}], press
    assert press["look"]["expanded"] == "false" and not press["look"]["kidsShown"]
    # The arrow stays under the pointer after the press (I5): nothing it hid moves it.
    for side in ("x", "y"):
        assert abs(press["look"]["tog"][side] - hover["shown"]["tog"][side]) < 0.5, side
    assert press["selected"] == press["selected0"]
    assert seen["left"]["actOpacity"] == "0"
    keys = seen["keys"]
    assert keys["body"]["actOpacity"] == "1" and "shep-fam-body" in keys["body"]["active"]
    assert "shep-fam-tog" in keys["arrow"]["active"] and keys["arrow"]["actOpacity"] == "1"
    assert (keys["enter"]["expanded"], keys["space"]["expanded"]) == ("false", "true")
    assert keys["joins"] == [{"tree": {t(runs["base"]): False}, "searched": False},
                             {"tree": {t(runs["base"]): True}, "searched": False}], keys["joins"]
    # Enter, then Space on another body: each selects its run, the focus kept on the body.
    for step, run in (("enterSelect", runs["base"]), ("spaceSelect", runs["top"])):
        got = keys[step]
        assert got["joins"] == [{"run": run}], got["joins"]
        assert got["state"]["selected"].startswith(run + "/") and got["active"] == "shep-fam-body|" + run
        assert got["state"]["currents"] == [run], got["state"]["currents"]
    deep = seen["deepPress"]
    assert (deep["before"], deep["after"], deep["expanded"]) == ("h", "h", "true"), deep
    assert deep["onTog"] and deep["pageOver"] <= 0 and deep["wider"], deep
    assert {runs["ext1"], runs["ext2"], runs["ext3"]} <= set(seen["deep"])
    folded = seen["folded"]
    assert not {runs["ext1"], runs["ext2"], runs["ext3"]} & set(folded["visible"])
    assert folded["joins"] == [{"tree": {t(runs["ext1"]): False, t(runs["ext2"]): False,
                                         t(runs["disease"]): False}, "searched": False}], folded
    unfolded = seen["unfolded"]
    assert runs["ext1"] in unfolded["visible"] and runs["ext2"] not in unfolded["visible"]
    assert unfolded["folds"][t(runs["ext1"])] == "false"

    def kept_still(step):
        # The node under the pointer, the region's scroll and the orientation as they were.
        assert step["same"] and abs(step["moved"]) < 0.5, step
        assert step["state"]["scroll"] == step["before"]["scroll"], step
        assert step["state"]["orient"] == step["before"]["orient"], step

    # A body by the pointer: the run's latest save (its first in the list's order), one
    # request, the run the one marked current in the graph, no run panel drawn.
    multi = seen["multi"]
    base_last = runs["base"] + "/last.pt"
    assert multi["joins"] == [{"run": runs["base"]}] and multi["state"]["selected"] == base_last
    # Picked half below the region's bottom edge: the answer is located already, the region
    # not scrolled to show the rest of the node.
    assert multi["cut"]["below"] > 0.5 and multi["cut"]["above"] > 0.5, multi["cut"]
    kept_still(multi)
    assert multi["state"]["panel"] == 0 and multi["state"]["currents"] == [runs["base"]]
    assert multi["state"]["saved"] == [[True, base_last], [False, runs["base"] + "/model-03.pt"]]
    saved = seen["savedPick"]
    other = runs["base"] + "/model-03.pt"
    assert saved["joins"] == [{"act": ml.PICK_ACT, "key": other}] and saved["state"]["selected"] == other
    assert saved["state"]["scroll"] == saved["before"]["scroll"] and abs(saved["moved"]) < 0.5
    assert saved["state"]["orient"] == saved["before"]["orient"]
    assert saved["state"]["saved"] == [[False, base_last], [True, other]]
    assert saved["active"] == ["LI", "true", other], saved["active"]
    assert abs(saved["fit"]) < 1, saved["fit"]               # the focus ring hugs its words
    assert saved["state"]["currents"] == [runs["base"]]
    # Searched: "Saved files" offers only what the list draws (the run shows that one
    # file: no row), the selection kept; cleared, both again.
    assert seen["savedSearched"]["selected"] == other and seen["savedSearched"]["saved"] == []
    assert seen["savedCleared"]["saved"] == [[False, base_last], [True, other]]
    pick = seen["pick"]
    assert pick["joins"] == [{"run": runs["ext1"]}] and pick["state"]["selected"].startswith(
        runs["ext1"] + "/")
    kept_still(pick)
    assert pick["state"]["saved"] == []                          # one saved file: no row
    hidden = seen["hidden"]
    assert hidden["state"]["selected"] == pick["state"]["selected"]
    assert runs["ext1"] not in hidden["state"]["visible"] and hidden["state"]["holds"] == [runs["disease"]]
    assert "folded branch under alpha-disease" in hidden["state"]["hidNote"]
    assert hidden["state"]["held"] == [[runs["disease"], ml.HOLDS_SELECTION]], hidden["state"]
    assert hidden["active"] == "shep-fam-tog|" + runs["disease"], hidden["active"]
    # The folded node's words act on nothing: its own lines' colour, never a link's.
    assert hidden["colors"][0] == hidden["colors"][1], hidden["colors"]
    again = seen["shownAgain"]
    assert runs["ext1"] in again["state"]["visible"] and again["state"]["hidNote"] is None
    assert again["joins"] == [{"tree": {t(runs["disease"]): True}, "searched": False}]
    assert again["active"] == runs["ext1"] and again["state"]["selected"] == pick["state"]["selected"]
    # Folded again, a body below picked: the note and the folded node's words go with the
    # old selection, and their room stays above the graph; nothing moved under the pointer.
    room = seen["room"]
    assert room["before"]["hidNote"] and room["before"]["held"], room["before"]
    assert room["state"]["hidNote"] is None and room["state"]["held"] == []
    assert room["state"]["selected"].startswith(runs["frozen"] + "/") and room["state"]["room"]
    kept_still(room)
    # Pointer and focus gone from the rows: the room given back as the region's scroll, the
    # node where it was (within the scroll position's own rounding).
    gone = seen["roomGone"]
    given = float(gone["before"]["room"].removesuffix("px"))
    assert gone["before"]["scroll"]["top"] >= given and gone["state"]["room"] == "", gone
    assert abs(gone["moved"]) <= 1, gone
    assert abs(gone["before"]["scroll"]["top"] - gone["state"]["scroll"]["top"] - given) <= 1, gone
    assert gone["state"]["orient"] == gone["before"]["orient"]
    searched, cleared = seen["searched"], seen["cleared"]
    assert runs["ext3"] in searched["visible"] and searched["count"].startswith("1 run matches")
    assert runs["ext3"] not in cleared["visible"]
    assert cleared["folds"] == seen["beforeSearch"]["folds"]
    assert cleared["selected"] == room["state"]["selected"]
    repick = seen["repick"]
    assert repick["vFolded"] == "v" and repick["state"]["orient"] == "v", repick
    assert repick["state"]["selected"].startswith(runs["base"] + "/")
    # A selection made elsewhere unfolds its chain; a pick in the chain keeps it unfolded.
    chain = seen["chain"]
    assert chain["shown"]["selected"].startswith(runs["ext3"] + "/")
    assert runs["ext3"] in chain["shown"]["visible"]
    assert chain["joins"] == [{"run": runs["ext2"]}] and runs["ext3"] in chain["picked"]["visible"]
    assert chain["same"] and abs(chain["moved"]) < 0.5, chain
    assert chain["picked"]["scroll"] == chain["shown"]["scroll"], chain
    assert chain["picked"]["orient"] == chain["shown"]["orient"]
    switch = seen["switch"]
    assert (switch["wide"]["orient"], switch["narrow"]["orient"], switch["back"]["orient"]) == (
        "h", "v", "h")
    for size in ("narrow", "back"):
        assert switch[size]["folds"] == switch["wide"]["folds"], size
        assert switch[size]["selected"] == repick["state"]["selected"]
        assert switch[size]["overlaps"] == 0
        assert switch[size]["pageOver"] <= 0, size
    vertical = seen["vertical"]
    assert vertical["before"]["act"]["h"] == 0 and vertical["after"]["act"]["h"] > 0
    assert abs(vertical["after"]["node"]["y"] - vertical["before"]["node"]["y"]) < 0.5
    assert vertical["after"]["lines"] == vertical["before"]["lines"]
    assert vertical["after"]["act"]["y"] >= vertical["after"]["body"]["bottom"] - 0.5
    assert vertical["after"]["edgesOnTog"], vertical["after"]
    centre = vertical["after"]["tog"]["x"] + vertical["after"]["tog"]["w"] / 2
    body = vertical["after"]["body"]
    assert abs(centre - (body["x"] + body["w"] / 2)) <= 1
    moved = vertical["nextAfter"]["node"]["y"] - vertical["next"]["node"]["y"]
    assert abs(moved - vertical["after"]["act"]["h"]) < 1, (moved, vertical)
    large = seen["large"]
    assert large["state"]["pageOver"] <= 0 and large["state"]["overlaps"] == 0
    # The keyboard's focus on the body: its ring drawn, nothing over it on any side.
    ring = large["ring"]
    assert ring["body"] and ring["keyboard"] and ring["style"] != "none", ring
    assert float(ring["width"].removesuffix("px")) >= 2 and all(ring["clear"]), ring
    assert seen["newest"] == {"table": True, "graph": False}
    touch = seen["touch"]
    assert touch["before"]["actOpacity"] == "1" and touch["before"]["act"]["h"] > 0
    assert touch["joins"] == [{"tree": {t(runs["base"]): False}, "searched": False}]
    assert touch["after"]["expanded"] == "false"
    assert seen["reduced"]["actTransition"] in ("0s", None)


#: A pick whose answer comes late (G07; the review of 00258f4, SL-R1): the
#: server's handling of a body's pick is held where it chooses the run's file
#: (a file handshake: ``hold`` while it is held, ``entered`` once it is), a
#: newer search (``search``) or a newer order (``order``) is answered
#: meanwhile, then the pick is let go; what the page shows once every answer
#: has come is read.
_LATE_PICK_JS = r"""
const { chromium } = require('playwright');
const fs = require('fs');
const QUEUE_JS = __QUEUE_JS__, RUN = __RUNS__;
const ROWS = '#shep-models-research-rows', COUNT = '#shep-models-research-count';
const [base, hold, entered, act] = process.argv.slice(2);
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1850, height: 960 }, deviceScaleFactor: 1 });
  await page.addInitScript(QUEUE_JS);
  const idle = () => page.waitForFunction(() => window.__shepQueue.idle(), null, { timeout: 30000 });
  const viewFor = (v) => page.waitForFunction((v) => { const m = document.querySelector('#shep-models-research-rows .shep-view-for'); return !!m && m.getAttribute('data-for') === v; }, v, { timeout: 30000 });
  const selectedIs = (run) => page.waitForFunction((r) => { const h = document.querySelector('#shep-mp-detail-hd h2[data-key]'); return !!h && h.getAttribute('data-key').startsWith(r + '/'); }, run, { timeout: 30000 });
  const frames = () => page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
  // The pick held on the server: asked with `hold` there, until the server says it is in.
  const pickHeld = async (run) => {
    fs.writeFileSync(hold, 'hold');
    await page.click(`${ROWS} li[data-run="${run}"] > .shep-fam-node .shep-fam-body`);
    const start = Date.now();
    while (!fs.existsSync(entered)) {
      if (Date.now() - start > 30000) { throw new Error('the pick never reached the server'); }
      await new Promise((r) => setTimeout(r, 10));
    }
  };
  // Let go, then everything answered: the selection in the details, the queue idle, drawn.
  const release = async (run) => {
    fs.unlinkSync(entered); fs.unlinkSync(hold);
    await selectedIs(run); await idle(); await frames(); await idle();
  };
  const shown = () => page.evaluate(([rows, count]) => {
    const list = document.querySelector(rows), marker = list.querySelector('.shep-view-for');
    const box = document.querySelector('#shep-models-research-search input, #shep-models-research-search textarea');
    const current = list.querySelector('[aria-current="true"]');
    const outside = list.querySelector('.shep-list-outside');
    const head = document.querySelector('#shep-mp-detail-hd h2[data-key]');
    return {
      query: box.value,
      runs: [...list.querySelectorAll('.shep-fam-body')].filter((e) => e.getClientRects().length).map((e) => e.dataset.run),
      graph: !!list.querySelector('.shep-fam'), table: !!list.querySelector('tr.shep-tree-run'),
      viewFor: marker ? marker.getAttribute('data-for') : null,
      // The count's words as drawn, never a hidden status tracker's timer (a draw in flight).
      count: (document.querySelector(count + ' .html-container') || document.querySelector(count)).innerText.trim(),
      current: current ? (current.getAttribute('data-key') || current.getAttribute('data-run') || (current.closest('li[data-run]') || { dataset: {} }).dataset.run || '') : null,
      outside: outside ? outside.innerText.trim() : '',
      selected: head ? head.getAttribute('data-key') : null,
    };
  }, [ROWS, COUNT]);
  const out = {};
  await page.goto(base, { waitUntil: 'domcontentloaded' });
  await page.click('#shep-tab-research-button', { timeout: 60000 });
  await page.click('#shep-page-models-button');
  await viewFor('newest|all'); await idle();
  await page.click('#shep-mp-order label:has-text("By family")');
  await viewFor('family|all'); await idle();
  await pickHeld(RUN.picked);
  if (act === 'search') {
    // A search typed and answered while the pick is held.
    const box = page.locator('#shep-models-research-search input, #shep-models-research-search textarea');
    await box.pressSequentially('beta');
    await page.waitForFunction((c) => document.querySelector(c).innerText.includes('“beta”'), COUNT, { timeout: 30000 });
  } else {
    // The Order changed and drawn while the pick is held.
    await page.click('#shep-mp-order label:has-text("Newest first")');
    await viewFor('newest|all');
  }
  out.during = await shown();
  await release(RUN.picked);
  out.after = await shown();
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


@pytest.mark.parametrize("act", ["search", "order"])
def test_an_answer_that_comes_late_never_draws_an_older_state_in_a_browser(
        tmp_path, family, monkeypatch, act):
    """G07 (an update never applies an old state; the review of 00258f4,
    SL-R1): By family, a body's pick is held on the server (a file
    handshake, no timed wait) while a search is typed and answered; once the
    pick's answer has come and every request is answered, the search box,
    the runs drawn and the count agree (the search's), and the pick is the
    selection, said above the graph as not shown by the search. The same
    root, a pick held while the Order changes: the list then draws the new
    order (Newest first's table, its count) with the pick selected in it."""
    from src.webui.app import create_gradio_app, mount_options
    from tests.fixtures.browser import QUEUE_JS, node_with_playwright, run_node
    from tests.fixtures.family_world import run_key
    from tests.fixtures.served_page import served

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    hold, entered = tmp_path / "hold", tmp_path / "entered"
    original = ml.run_choice

    def held(*args, **kwargs):
        # The pick's handler has read its inputs; it waits here while held.
        if hold.exists():
            entered.write_text("in")
            deadline = time.monotonic() + 60
            while hold.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("the held pick was never let go")
                time.sleep(0.01)
        return original(*args, **kwargs)

    monkeypatch.setattr(ml, "run_choice", held)
    picked = run_key("alpha-base")
    script = (_LATE_PICK_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS))
              .replace("__RUNS__", json.dumps({"picked": picked})))
    with served(create_gradio_app(), **mount_options()) as base:
        try:
            seen = run_node(script, tmp_path, base[:-len("gradio_api")], str(hold), str(entered),
                            act)
        finally:
            hold.unlink(missing_ok=True)
    during, after = seen["during"], seen["after"]
    # Settled: the pick made before the newer act is the selection.
    assert after["selected"].startswith(picked + "/"), after
    if act == "search":
        assert during["query"] == "beta" and "“beta”" in during["count"], during
        assert during["runs"] and all("beta" in run for run in during["runs"]), during
        # The box, the runs drawn and the count are the search's, and the
        # selection is said as not shown by it.
        assert after["query"] == "beta", after
        assert after["runs"] == during["runs"] and after["count"] == during["count"], (
            during, after)
        assert "not shown by this filter" in after["outside"], after
        assert "search “beta”" in after["outside"], after
        return
    assert during["viewFor"] == "newest|all" and during["table"] and not during["graph"], during
    # Newest first's table and count, the pick selected in it.
    assert after["viewFor"] == "newest|all", after
    assert after["table"] and not after["graph"] and after["count"] == during["count"], (
        during, after)
    assert (after["current"] or "").startswith(picked + "/"), after


#: A search answered late (G07; the review of 5171661, P2-1): "b" typed and
#: drawn, then "e" (its search, asked with "be", held on the server: a file
#: handshake, ``hold`` while it is held, ``entered`` once it is), Clear
#: clicked (the box emptied in the page; its own search waits behind the held
#: one, the search's handler running one at a time), then the search let go;
#: what the page shows once every answer has come is read.
_LATE_SEARCH_JS = r"""
const { chromium } = require('playwright');
const fs = require('fs');
const QUEUE_JS = __QUEUE_JS__;
const ROWS = '#shep-models-research-rows', COUNT = '#shep-models-research-count';
const BOX = '#shep-models-research-search input, #shep-models-research-search textarea';
const [base, hold, entered] = process.argv.slice(2);
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1850, height: 960 }, deviceScaleFactor: 1 });
  await page.addInitScript(QUEUE_JS);
  const idle = () => page.waitForFunction(() => window.__shepQueue.idle(), null, { timeout: 30000 });
  const viewFor = (v) => page.waitForFunction((v) => { const m = document.querySelector('#shep-models-research-rows .shep-view-for'); return !!m && m.getAttribute('data-for') === v; }, v, { timeout: 30000 });
  const frames = () => page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
  const shown = () => page.evaluate(([rows, count, box]) => {
    const list = document.querySelector(rows);
    return {
      query: document.querySelector(box).value,
      runs: [...list.querySelectorAll('.shep-fam-body')].filter((e) => e.getClientRects().length).map((e) => e.dataset.run),
      count: (document.querySelector(count + ' .html-container') || document.querySelector(count)).innerText.trim(),
    };
  }, [ROWS, COUNT, BOX]);
  const out = {};
  await page.goto(base, { waitUntil: 'domcontentloaded' });
  await page.click('#shep-tab-research-button', { timeout: 60000 });
  await page.click('#shep-page-models-button');
  await viewFor('newest|all'); await idle();
  await page.click('#shep-mp-order label:has-text("By family")');
  await viewFor('family|all'); await idle(); await frames(); await idle();
  out.before = await shown();
  // "b" typed and drawn; then "e", its search held on the server.
  await page.locator(BOX).pressSequentially('b');
  await page.waitForFunction((c) => document.querySelector(c).innerText.includes('“b”'), COUNT, { timeout: 30000 });
  await idle();
  fs.writeFileSync(hold, 'hold');
  await page.keyboard.press('e');
  const start = Date.now();
  while (!fs.existsSync(entered)) {
    if (Date.now() - start > 30000) { throw new Error('the search never reached the server'); }
    await new Promise((r) => setTimeout(r, 10));
  }
  // Clear while the search is held: the box emptied in the page at once.
  await page.click('#shep-models-research-search-clear');
  await page.waitForFunction((box) => document.querySelector(box).value === '', BOX, { timeout: 30000 });
  await frames();
  out.during = await shown();
  // Let go, then everything answered and drawn.
  fs.unlinkSync(entered); fs.unlinkSync(hold);
  await idle(); await frames(); await idle(); await frames(); await idle();
  out.after = await shown();
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_a_search_answered_after_its_clear_draws_no_search_in_a_browser(tmp_path, family):
    """G07 (an update never applies an old state; the review of 5171661,
    P2-1): By family, "b" typed and drawn, then "e", its search (asked with
    "be") held on the server (a file handshake, no timed wait) while Clear is
    clicked (the box emptied in the page, its own search answered once the
    held one is); "be" is never drawn, and once the search is let go and
    every request is answered, the search box is empty and the runs drawn
    and the count are the whole list's, as before the search: a search
    answered late draws the box as it is, never the text it was typed with."""
    import functools

    from src.webui.app import create_gradio_app, mount_options
    from tests.fixtures.browser import QUEUE_JS, node_with_playwright, run_node
    from tests.fixtures.served_page import served

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    hold, entered = tmp_path / "hold", tmp_path / "entered"
    demo = create_gradio_app()
    box = next(b for b in demo.blocks.values()
               if getattr(b, "elem_id", None) == "shep-models-research-search")
    (searched,) = [fn for fn in demo.fns.values()
                   if fn.name == "_searched" and (box._id, "change") in fn.targets]
    original = searched.fn

    @functools.wraps(original)
    def held(*args, **kwargs):
        # The search's handler is processing; it waits here while held.
        if hold.exists():
            entered.write_text("in")
            deadline = time.monotonic() + 60
            while hold.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("the held search was never let go")
                time.sleep(0.01)
        return original(*args, **kwargs)

    searched.fn = held
    script = _LATE_SEARCH_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS))
    with served(demo, **mount_options()) as base:
        try:
            seen = run_node(script, tmp_path, base[:-len("gradio_api")], str(hold), str(entered))
        finally:
            hold.unlink(missing_ok=True)
    before, during, after = seen["before"], seen["during"], seen["after"]
    assert before["query"] == "" and len(before["runs"]) > 4, before
    assert during["query"] == "" and "“be”" not in during["count"], (before, during)
    # Settled: the empty box, and the whole list drawn under it.
    assert after == before, (before, after)


#: Every way the search box's text changes (the adversarial check of
#: 37912a9, F2): By family, a body picked, then "d" and "e" typed, Backspace,
#: select all and Delete, "al" filled (one input event, no key, as a paste),
#: "beta" composed by an input method (Chromium's own, over CDP: no key
#: event), and the list's "Clear search" where "beta" hides the selection;
#: what each typed text draws is read as the reference. After each act the
#: page waits for the count to say the box's search (at most 10 s: an act
#: whose search never runs is read as it is then), then for every request
#: to be answered.
_SEARCH_EDITS_JS = r"""
const { chromium } = require('playwright');
const QUEUE_JS = __QUEUE_JS__, RUN = __RUNS__;
const ROWS = '#shep-models-research-rows', COUNT = '#shep-models-research-count';
const BOX = '#shep-models-research-search input, #shep-models-research-search textarea';
const [base] = process.argv.slice(2);
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1850, height: 960 }, deviceScaleFactor: 1 });
  await page.addInitScript(QUEUE_JS);
  const cdp = await page.context().newCDPSession(page);
  const idle = () => page.waitForFunction(() => window.__shepQueue.idle(), null, { timeout: 30000 });
  const viewFor = (v) => page.waitForFunction((v) => { const m = document.querySelector('#shep-models-research-rows .shep-view-for'); return !!m && m.getAttribute('data-for') === v; }, v, { timeout: 30000 });
  const frames = () => page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
  const settled = async () => { await frames(); await idle(); await frames(); await idle(); };
  const shown = () => page.evaluate(([rows, count, box]) => {
    const list = document.querySelector(rows);
    const current = list.querySelector('[aria-current="true"]');
    const outside = list.querySelector('.shep-list-outside');
    const head = document.querySelector('#shep-mp-detail-hd h2[data-key]');
    return {
      query: document.querySelector(box).value,
      runs: [...list.querySelectorAll('.shep-fam-body')].filter((e) => e.getClientRects().length).map((e) => e.dataset.run),
      count: (document.querySelector(count + ' .html-container') || document.querySelector(count)).innerText.trim(),
      current: current ? (current.getAttribute('data-key') || current.getAttribute('data-run') || (current.closest('li[data-run]') || { dataset: {} }).dataset.run || '') : null,
      outside: outside ? outside.innerText.trim() : '',
      selected: head ? head.getAttribute('data-key') : null,
    };
  }, [ROWS, COUNT, BOX]);
  // The count saying the search `q` (none for ''), then every request answered.
  const drawnFor = async (q) => {
    try {
      await page.waitForFunction(([c, q]) => { const t = document.querySelector(c).innerText;
        return q ? t.includes('\u201c' + q + '\u201d') : !t.includes('\u201c'); }, [COUNT, q], { timeout: 10000 });
    } catch (e) { /* never drawn: read as it is */ }
    await settled();
    return shown();
  };
  const box = async () => { await page.locator(BOX).focus(); };
  const emptied = async () => { await box(); await page.keyboard.press('Control+A'); await page.keyboard.press('Delete'); };
  const out = {};
  await page.goto(base, { waitUntil: 'domcontentloaded' });
  await page.click('#shep-tab-research-button', { timeout: 60000 });
  await page.click('#shep-page-models-button');
  await viewFor('newest|all'); await idle();
  await page.click('#shep-mp-order label:has-text("By family")');
  await viewFor('family|all'); await settled();
  // A body picked: the selection every act keeps.
  await page.click(`${ROWS} li[data-run="${RUN.picked}"] > .shep-fam-node .shep-fam-body`);
  await page.waitForFunction((r) => { const h = document.querySelector('#shep-mp-detail-hd h2[data-key]'); return !!h && h.getAttribute('data-key').startsWith(r + '/'); }, RUN.picked, { timeout: 30000 });
  out.full = await drawnFor('');
  await box(); await page.keyboard.type('d'); out.d = await drawnFor('d');
  await box(); await page.keyboard.type('e'); out.de = await drawnFor('de');
  await box(); await page.keyboard.press('Backspace'); out.backspace = await drawnFor('d');
  await emptied(); out.deleted = await drawnFor('');
  await page.locator(BOX).fill('al'); out.filled = await drawnFor('al');
  await emptied(); await drawnFor('');
  await box(); await page.keyboard.type('al'); out.typedAl = await drawnFor('al');
  await emptied(); await drawnFor('');
  await box();
  for (const t of ['b', 'be', 'bet', 'beta']) { await cdp.send('Input.imeSetComposition', { text: t, selectionStart: t.length, selectionEnd: t.length }); }
  await cdp.send('Input.insertText', { text: 'beta' });
  out.composed = await drawnFor('beta');
  await emptied(); await drawnFor('');
  await box(); await page.keyboard.type('beta'); out.typedBeta = await drawnFor('beta');
  await page.click(`${ROWS} .shep-list-outside button[data-key="__clear_search__"]`);
  out.clearSearch = await drawnFor('');
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_every_change_of_the_search_box_draws_its_search_in_a_browser(tmp_path, family):
    """The adversarial check of 37912a9 (F2: Gradio 6.30's Textbox sends
    ``input`` from its keypress handler alone, so a deleted or pasted text
    was never searched and the list kept drawing the old search under the
    box): By family, a body picked, then "de" typed; Backspace draws "d" as
    typing "d" did; select all and Delete draws the whole list again; "al"
    filled as a paste (no key) and "beta" composed by an input method (no
    key) draw as the same text typed; the list's "Clear search" (where
    "beta" hides the selection) draws the whole list. The selection is kept
    throughout, and the box, the runs drawn and the count agree each time."""
    from src.webui.app import create_gradio_app, mount_options
    from tests.fixtures.browser import QUEUE_JS, node_with_playwright, run_node
    from tests.fixtures.family_world import run_key
    from tests.fixtures.served_page import served

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    picked = run_key("alpha-base")
    script = (_SEARCH_EDITS_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS))
              .replace("__RUNS__", json.dumps({"picked": picked})))
    with served(create_gradio_app(), **mount_options()) as base:
        seen = run_node(script, tmp_path, base[:-len("gradio_api")])
    full = seen["full"]
    assert full["query"] == "" and len(full["runs"]) > 4, full
    assert full["selected"].startswith(picked + "/") and "“" not in full["count"], full
    assert all(seen[step]["selected"] == full["selected"] for step in seen), seen
    d, de = seen["d"], seen["de"]
    assert (d["query"], de["query"]) == ("d", "de"), (d, de)
    assert "“d”" in d["count"] and "“de”" in de["count"], (d, de)
    # Backspace: "d" drawn, as typed.
    assert seen["backspace"] == d, (d, seen["backspace"])
    # Select all and Delete: the whole list, as before the search.
    assert seen["deleted"] == full, (full, seen["deleted"])
    # Filled (a paste's one input event, no key): "al" drawn, as typed.
    assert seen["typedAl"]["query"] == "al" and "“al”" in seen["typedAl"]["count"]
    assert seen["filled"] == seen["typedAl"], (seen["typedAl"], seen["filled"])
    # Composed by an input method: "beta" drawn, as typed; the selection said
    # as not shown by it, with "Clear search".
    beta = seen["typedBeta"]
    assert "“beta”" in beta["count"] and "search “beta”" in beta["outside"], beta
    assert seen["composed"] == beta, (beta, seen["composed"])
    # "Clear search": the whole list again.
    assert seen["clearSearch"] == full, (full, seen["clearSearch"])


#: **An answer made for a state the list no longer shows** (G07; the review of
#: dcaed68, SL-R3), By family, by real events; one handler held on the server
#: by a file handshake (``hold`` while it is held, ``entered`` once it is), no
#: timed wait:
#:
#: - ``refresh`` (SL-R3b): alpha-base picked; Models left and opened again,
#:   its re-read held; beta-base picked and its details drawn meanwhile; the
#:   re-read let go;
#: - ``workspace`` (SL-R3a): a body's pick in ``demo`` held; ``other`` chosen
#:   and its rows drawn; the pick let go;
#: - ``newer`` (the same, then a body of ``other`` picked before the held pick
#:   is let go: its request sent, queued behind it);
#: - ``reopen-workspace`` (SL-R3c, the recheck of bd5e678): alpha-base picked;
#:   Models left and opened again, its opening held; ``other`` chosen and its
#:   rows drawn with nothing selected meanwhile; the opening let go.
#:
#: What the page shows once every answer has come is read.
_HELD_ANSWER_JS = r"""
const { chromium } = require('playwright');
const fs = require('fs');
const QUEUE_JS = __QUEUE_JS__, RUN = __RUNS__;
const ROWS = '#shep-models-research-rows';
const [base, hold, entered, mode] = process.argv.slice(2);
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1850, height: 960 }, deviceScaleFactor: 1 });
  await page.addInitScript(QUEUE_JS);
  const idle = () => page.waitForFunction(() => window.__shepQueue.idle(), null, { timeout: 30000 });
  const frames = () => page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
  const viewFor = (v) => page.waitForFunction((v) => { const m = document.querySelector('#shep-models-research-rows .shep-view-for'); return !!m && m.getAttribute('data-for') === v; }, v, { timeout: 30000 });
  const selectedIs = (run) => page.waitForFunction((r) => { const h = document.querySelector('#shep-mp-detail-hd h2[data-key]'); return !!h && h.getAttribute('data-key').startsWith(r + '/'); }, run, { timeout: 30000 });
  const body = (run) => `${ROWS} li[data-run="${run}"] > .shep-fam-node .shep-fam-body`;
  const inside = async () => {
    const start = Date.now();
    while (!fs.existsSync(entered)) {
      if (Date.now() - start > 30000) { throw new Error('the held answer never reached the server'); }
      await new Promise((r) => setTimeout(r, 10));
    }
  };
  const shown = () => page.evaluate((rows) => {
    const head = document.querySelector('#shep-mp-detail-hd h2[data-key]');
    const current = [...document.querySelectorAll(rows + ' [aria-current="true"]')].map((e) => e.getAttribute('data-key') || e.getAttribute('data-run') || (e.closest('li[data-run]') || { dataset: {} }).dataset.run || '');
    const load = document.querySelector('#shep-models-research-load');
    const gen = document.querySelector(rows + ' .shep-list-gen');
    return {
      workspace: document.querySelector('#shep-models-research-workspace input').value,
      selected: head ? head.getAttribute('data-key') : null,
      detail: document.querySelector('#shep-mp-detail').innerText,
      current,
      loadEnabled: !!load && !load.disabled,
      gen: gen ? gen.getAttribute('data-gen') : null,
    };
  }, ROWS);
  const out = {};
  await page.goto(base, { waitUntil: 'domcontentloaded' });
  await page.click('#shep-tab-research-button', { timeout: 60000 });
  await page.click('#shep-page-models-button');
  await viewFor('newest|all'); await idle();
  await page.click('#shep-mp-order label:has-text("By family")');
  await viewFor('family|all'); await idle(); await frames();
  if (mode === 'refresh' || mode === 'reopen-workspace') {
    await page.click(body(RUN.first));
    await selectedIs(RUN.first); await idle(); await frames();
    out.before = await shown();
    await page.click('#shep-page-training-button'); await idle();
    fs.writeFileSync(hold, 'hold');
    await page.click('#shep-page-models-button');
    await inside();
    if (mode === 'refresh') {
      // The newer pick, made and drawn while the re-read is held.
      await page.click(body(RUN.newer));
      await selectedIs(RUN.newer); await frames();
    } else {
      // Another workspace chosen, its rows drawn with nothing selected,
      // while the opening is held.
      await page.click('#shep-models-research-workspace input');
      await page.getByRole('option', { name: /^other(?:\s|$)/ }).click();
      await page.waitForFunction((sel) => !!document.querySelector(sel), body(RUN.other), { timeout: 30000 });
      await page.waitForFunction(() => !document.querySelector('#shep-mp-detail-hd h2[data-key]'), null, { timeout: 30000 });
      await frames();
    }
  } else {
    fs.writeFileSync(hold, 'hold');
    await page.click(body(RUN.first));
    await inside();
    await page.click('#shep-models-research-workspace input');
    await page.getByRole('option', { name: /^other(?:\s|$)/ }).click();
    await page.waitForFunction((sel) => !!document.querySelector(sel), body(RUN.other), { timeout: 30000 });
    await frames();
    if (mode === 'newer') {
      // Picked in other's rows before the held pick is let go: sent, queued behind it.
      const joins = await page.evaluate(() => window.__shepQueue.joins.length);
      await page.click(body(RUN.other));
      await page.waitForFunction((n) => window.__shepQueue.joins.length > n, joins, { timeout: 30000 });
    }
  }
  out.during = await shown();
  // From here on, every selection the details or the list show is noted.
  await page.evaluate((rows) => {
    const seen = window.__shepSeen = { keys: [], current: [] };
    const note = () => {
      const head = document.querySelector('#shep-mp-detail-hd h2[data-key]');
      if (head && !seen.keys.includes(head.getAttribute('data-key'))) { seen.keys.push(head.getAttribute('data-key')); }
      document.querySelectorAll(rows + ' [aria-current="true"]').forEach((e) => {
        const run = e.getAttribute('data-key') || e.getAttribute('data-run') || (e.closest('li[data-run]') || { dataset: {} }).dataset.run || '';
        if (!seen.current.includes(run)) { seen.current.push(run); }
      });
    };
    ['#shep-mp-detail-hd', rows].forEach((sel) => new MutationObserver(note).observe(document.querySelector(sel), { childList: true, subtree: true, attributes: true, characterData: true }));
  }, ROWS);
  fs.unlinkSync(entered); fs.unlinkSync(hold);
  if (mode === 'newer') { await selectedIs(RUN.other); }
  await idle(); await frames(); await idle(); await frames();
  out.after = await shown();
  out.seen = await page.evaluate(() => window.__shepSeen);
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def _held_world(family, tmp_path, monkeypatch, *, same_key=False):
    """The family world copied for one test (the module's is shared), served:
    with `same_key`, ``other`` holds a file at gamma-disease's relative key
    too, another checkpoint (epoch 42, another fingerprint) the shared reader
    reads and would load. Returns that key and both fingerprints."""
    from src.utils.checkpoint_io import read_checkpoint, write_checkpoint
    from tests.fixtures.family_world import file_key, run_key

    root = shutil.copytree(family["root"], tmp_path / "world", symlinks=True)
    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(root))
    family["served"]["value"] = _status(root / "demo", family["shas"][("alpha-disease",
                                                                      "last.pt")])
    key = file_key("gamma-disease")
    shas = {"demo": family["shas"][("gamma-disease", "last.pt")], "other": None}
    if same_key:
        source = root / "demo" / "runs" / run_key("gamma-disease") / "checkpoints" / "last.pt"
        target = root / "other" / "runs" / run_key("gamma-disease") / "checkpoints" / "last.pt"
        payload = read_checkpoint(source, map_location="cpu", weights_only=False).checkpoint
        payload["epoch"] = 42
        shas["other"] = write_checkpoint(payload, target)
        assert shas["other"] != shas["demo"]
        other = ml.list_models("other", gradio_request(), family=True)
        assert ml.can_load(ml.find_item(other, key))           # read, and loadable, today
    return key, shas


def _held_run(tmp_path, monkeypatch, mode, runs, held_fn, module=ml):
    """Serve the dashboard with `held_fn` of `module` (the list's
    ``run_choice`` or ``list_models``; this page's ``read_page``) held while
    ``hold`` exists (a refresh's read only once armed), run `_HELD_ANSWER_JS`
    in `mode`, and return what the page showed, with every selection the
    list's draw was asked to draw, by workspace (``drawn``: the session's
    state as each draw read it)."""
    from src.webui.app import create_gradio_app, mount_options
    from tests.fixtures.browser import QUEUE_JS, run_node
    from tests.fixtures.served_page import served

    hold, entered = tmp_path / "hold", tmp_path / "entered"
    original = getattr(module, held_fn)

    def held(*args, **kwargs):
        # The handler has read its inputs; it waits here while held.
        if hold.exists() and not entered.exists():
            entered.write_text("in")
            deadline = time.monotonic() + 60
            while hold.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("the held answer was never let go")
                time.sleep(0.01)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, held_fn, held)
    drawn, drawing = [], ml.list_drawing

    def noted(listing, key, *args, **kwargs):
        drawn.append((listing.get("workspace"), key))
        return drawing(listing, key, *args, **kwargs)

    monkeypatch.setattr(ml, "list_drawing", noted)
    script = (_HELD_ANSWER_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS))
              .replace("__RUNS__", json.dumps(runs)))
    with served(create_gradio_app(), **mount_options()) as base:
        try:
            seen = run_node(script, tmp_path, base[:-len("gradio_api")], str(hold),
                            str(entered), mode)
        finally:
            hold.unlink(missing_ok=True)
    return {**seen, "drawn": drawn}


def test_a_re_read_answered_after_a_newer_pick_never_sets_it_back_in_a_browser(
        tmp_path, family, monkeypatch):
    """SL-R3b (the review of dcaed68): alpha-base picked; Models left and
    opened again, its re-read held on the server while beta-base is picked
    and its details drawn (the page lets the operator pick meanwhile); once
    the re-read is let go and every answer has come, beta-base is still the
    selection, in the details and in the list alike, never the selection the
    re-read began with."""
    from tests.fixtures.browser import node_with_playwright
    from tests.fixtures.family_world import run_key

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    first, newer = run_key("alpha-base"), run_key("beta-base")
    seen = _held_run(tmp_path, monkeypatch, "refresh", {"first": first, "newer": newer},
                     "list_models")
    before, during, after = seen["before"], seen["during"], seen["after"]
    assert before["selected"].startswith(first + "/"), before
    assert during["selected"].startswith(newer + "/") and during["current"] == [newer], during
    assert after["selected"] == during["selected"], (during, after)
    assert after["current"] == [newer] and after["detail"].startswith("beta-base"), after
    assert after["gen"] == "1" and after["workspace"] == "demo", after


@pytest.mark.parametrize("same_key", [False, True], ids=["no-file-there", "same-key-other-file"])
def test_a_pick_answered_after_another_workspace_was_chosen_selects_nothing_in_a_browser(
        tmp_path, family, monkeypatch, same_key):
    """SL-R3a (the review of dcaed68): By family in ``demo``, gamma-disease's
    body picked, its answer held on the server; ``other`` chosen and its rows
    drawn (nothing selected, Load off); the pick let go. Once every answer has
    come nothing is selected in ``other``, Load stays off and no details are
    drawn, whether ``other`` lists no file at that relative key or another
    checkpoint at the same key (epoch 42, another fingerprint, one the
    shared reader reads and would load): the same relative key is not the
    file that was picked."""
    from tests.fixtures.browser import node_with_playwright
    from tests.fixtures.family_world import OTHER_BASE_RUN, run_key

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    key, shas = _held_world(family, tmp_path, monkeypatch, same_key=same_key)
    seen = _held_run(tmp_path, monkeypatch, "workspace",
                     {"first": run_key("gamma-disease"), "other": "gatv2/" + OTHER_BASE_RUN},
                     "run_choice")
    during, after = seen["during"], seen["after"]
    assert during["workspace"] == "other" and during["selected"] is None, during
    assert not during["loadEnabled"], during
    # The session's selection never held the picked key, in either workspace.
    assert [ws for ws, k in seen["drawn"] if k == key] == [], seen["drawn"]
    assert after["workspace"] == "other" and after["selected"] is None, after
    assert not after["loadEnabled"] and after["current"] == [], after
    assert "Select a model" in after["detail"] and "epoch 42" not in after["detail"], after
    if same_key:
        assert shas["other"][:12] not in after["detail"], after
    # The rows say the generation they were drawn from: other's, the second.
    assert during["gen"] == after["gen"] == "2", (during, after)


def test_a_pick_made_in_the_new_workspace_stays_after_an_older_pick_is_let_go_in_a_browser(
        tmp_path, family, monkeypatch):
    """SL-R3a, the newer act (the review of dcaed68, acceptance 3): a body's
    pick in ``demo`` held; ``other`` chosen and drawn; a body of ``other``
    picked before the held pick is let go (its request sent, queued behind
    it). Once every answer has come, the pick made in ``other`` is the
    selection and the details', and the held one never selected anything on
    the way (neither the details nor the list showed it once it was let go),
    even with a file at the held pick's relative key in ``other``. Gradio
    runs one list's picks one at a time, so here the newer pick is answered
    after the held one; the held answer landing after the newer pick is the
    unit test's (`test_webui_model_list.py::
    test_a_newer_pick_stays_when_an_older_pick_answers_after_it`)."""
    from tests.fixtures.browser import node_with_playwright
    from tests.fixtures.family_world import OTHER_BASE_RUN, run_key

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    _held_world(family, tmp_path, monkeypatch, same_key=True)
    other = "gatv2/" + OTHER_BASE_RUN
    seen = _held_run(tmp_path, monkeypatch, "newer",
                     {"first": run_key("gamma-disease"), "other": other}, "run_choice")
    during, after = seen["during"], seen["after"]
    assert during["workspace"] == "other" and during["selected"] is None, during
    held = run_key("gamma-disease")
    assert [k for _ws, k in seen["drawn"] if k and k.startswith(held + "/")] == [], seen["drawn"]
    assert not [k for k in seen["seen"]["keys"] if k.startswith(held + "/")], seen["seen"]
    assert held not in seen["seen"]["current"], seen["seen"]
    assert after["workspace"] == "other" and after["selected"].startswith(other + "/"), after
    assert after["current"] == [other] and after["detail"].startswith("gamma-base"), after
    assert after["loadEnabled"] is False                        # a base: never loaded


def test_a_pick_made_while_the_opening_reads_the_pages_data_stays_in_a_browser(
        tmp_path, family, monkeypatch):
    """SL-R3c (the recheck of bd5e678), the opening's own read: alpha-base
    picked; Models left and opened again, the page's data (`read_page`: the
    cohort versions and usage histories) held on the server while beta-base
    is picked and drawn in the list and its details; once the read is let go
    and every answer has come, beta-base is still the selection, never set
    back to the selection the opening's commit kept (it read its data after
    the commit, then wrote what it had committed)."""
    from tests.fixtures.browser import node_with_playwright
    from tests.fixtures.family_world import run_key

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    first, newer = run_key("alpha-base"), run_key("beta-base")
    seen = _held_run(tmp_path, monkeypatch, "refresh", {"first": first, "newer": newer},
                     "read_page", module=mp)
    before, during, after = seen["before"], seen["during"], seen["after"]
    assert before["selected"].startswith(first + "/"), before
    assert during["selected"].startswith(newer + "/") and during["current"] == [newer], during
    assert after["selected"] == during["selected"], (during, after)
    assert after["current"] == [newer] and after["detail"].startswith("beta-base"), after
    # Once let go, neither the details nor the list showed alpha-base again.
    assert not [k for k in seen["seen"]["keys"] if k.startswith(first + "/")], seen["seen"]
    assert first not in seen["seen"]["current"], seen["seen"]
    assert after["gen"] == "1" and after["workspace"] == "demo", after


def test_another_workspace_chosen_while_the_opening_reads_the_pages_data_stays_in_a_browser(
        tmp_path, family, monkeypatch):
    """SL-R3c (the recheck of bd5e678), the opening's own read: alpha-base
    picked in ``demo``; Models left and opened again, the page's data held on
    the server while ``other`` is chosen and its rows drawn with nothing
    selected; once the read is let go and every answer has come, the
    workspace, the list and the details are still ``other``'s with nothing
    selected and Load off: the opening, answered for a workspace the list no
    longer shows, changed nothing (never ``demo`` and alpha-base again)."""
    from tests.fixtures.browser import node_with_playwright
    from tests.fixtures.family_world import OTHER_BASE_RUN, run_key

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    first, other = run_key("alpha-base"), "gatv2/" + OTHER_BASE_RUN
    seen = _held_run(tmp_path, monkeypatch, "reopen-workspace",
                     {"first": first, "other": other}, "read_page", module=mp)
    before, during, after = seen["before"], seen["during"], seen["after"]
    assert before["selected"].startswith(first + "/") and before["workspace"] == "demo", before
    assert during["workspace"] == "other" and during["selected"] is None, during
    assert during["gen"] == "2", during
    assert after["workspace"] == "other" and after["selected"] is None, after
    assert after["current"] == [] and not after["loadEnabled"], after
    assert "Select a model" in after["detail"] and "alpha-base" not in after["detail"], after
    assert after["gen"] == "2", after                          # other's rows, never demo's
    assert seen["seen"] == {"keys": [], "current": []}, seen["seen"]


def test_the_opening_reads_everything_before_it_commits_and_writes_only_what_it_committed(
        world, monkeypatch):
    """SL-R3c (the recheck of bd5e678): Models' opening reads the list and
    its own data (`read_page`) first, then commits through the list's clock,
    then builds its outputs from what it read and what the commit kept, with
    no read after the commit. A pick committed while the page's data is read
    is the selection it writes, with its details; another workspace's listing
    committed meanwhile makes it change nothing at all (no listing,
    workspace, selection, page data, view or details). The page's data is
    keyed by the listing committed, so its change reads nothing again."""
    demo, _ctx = _page()
    opened = _fn(demo, "_open")
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    request = operator_request(monkeypatch)
    clock = ml.ListClock()
    rest = ("", {}, mp.BY_FAMILY, mp.ALL_TASKS, mp.RECORDS, "idle", request, clock)
    out = opened.fn("demo", False, None, None, None, *rest)
    listing = out[0]
    first, newer = _key(world, "model-05.pt"), _key(world, "model-08.pt")
    assert clock.commit_pick(listing, None, first, {}, {})
    real_read, real_commit, order = mp.read_page, ml.ListClock.commit_listing, []
    committed = []

    def commit(self, *args, **kwargs):
        order.append("commit")
        done = real_commit(self, *args, **kwargs)
        committed.append(True)
        return done

    def never(name):
        def read(*args, **kwargs):
            assert not committed, f"{name} read after the commit"
            return {"read_page": real_read, "list_models": real_list,
                    "read_status": real_status, "read_versions": real_versions}[name](
                        *args, **kwargs)
        return read

    real_list, real_status, real_versions = ml.list_models, mp.read_status, mp.read_versions
    monkeypatch.setattr(ml.ListClock, "commit_listing", commit)
    monkeypatch.setattr(ml, "list_models", never("list_models"))
    monkeypatch.setattr(mp, "read_status", never("read_status"))
    monkeypatch.setattr(mp, "read_versions", never("read_versions"))

    def meanwhile(read):
        order.append("read_page")
        assert clock.commit_pick(listing, None, newer, {}, {})    # a newer pick, committed
        return never("read_page")(read)

    monkeypatch.setattr(mp, "read_page", meanwhile)
    out = opened.fn("demo", False, None, listing, first, *rest)
    by = dict(zip(map(id, opened.outputs), out, strict=True))
    assert order == ["read_page", "commit"]
    assert out[1] == newer and clock.selected == newer
    assert f'data-key="{newer}"' in by[id(by_id["shep-mp-detail-hd"])]
    assert world["shas"]["model-08.pt"] in by[id(by_id["shep-mp-detail-top"])]
    data = next(v for v in out if isinstance(v, dict) and "listing_key" in v)
    assert data["listing_key"] == mp.listing_key(out[0])
    assert out[0][ml.GENERATION] == 1

    # Another workspace's listing committed while the page's data is read.
    committed.clear()

    def switched(read):
        committed.clear()
        real_commit(clock, clock.began(), {**listing, "workspace": "other"}, view=None,
                    preset=ml.PRESETS["research"], search="", fold={})
        return real_read(read)

    monkeypatch.setattr(mp, "read_page", switched)
    out = opened.fn("demo", False, None, listing, newer, *rest)
    assert len(out) == len(opened.outputs) and all(v == gr.update() for v in out), out
    assert clock.workspace == "other" and clock.generation == 2 and clock.selected is None
