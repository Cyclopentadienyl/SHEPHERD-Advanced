"""
Research › Evaluate › Models (M5-2, UI-D): the research preset of the shared list.
==================================================================================
`src/webui/components/models_page.py`. What these pin, on checkpoints listed by
the server, ledgers written by the ledger's own functions and reports kept in a
registry's report store (`tests/fixtures/model_records.py`):

- **ordering per role (§4.9 question 1)**: offered only for a cohort version
  whose role is in ``SHEPHERD_ORDERING_ROLES``; ``institutional_acceptance``
  listed but off with its reason, even when listed; the independence note
  wherever ordering is offered, and nowhere else; newest first otherwise;
- **D4 through WP4's one function**: one record per checkpoint (the first
  registered deterministic one), comparable records only (one version, one
  scoring semantics, a second semantics offered as a choice), mean rank lower
  first, the regime on each ordered row (wrapping only after its dot) and
  "regime differs" with its ⓘ where it does; the groups "Ordered by …",
  "Contested on this version · not ordered", "Contested status cannot be
  checked · not ordered", "Not compared on this version · not ordered", "Not
  tested on this cohort version · newest first"; a role the setting no longer
  lists refused by WP4's function itself;
- **contested records left out (D5)**, and a state that cannot be checked
  never read as uncontested;
- **the shared Load writes the usage history entry** for a Load from an ordered
  list (the page's ``ordering``), and none for newest first; a history file
  (every ordered row) is refused, a task model loaded from the list instead;
- **task and family**: the Task select's options (the list's one vocabulary,
  short, unique); the task filter orders only what is shown (History keeps
  the Mode C order, Disease has none); newest first's Kind column under All;
  By family's groups (parent first, indented Lineage cells, a context row in
  grey with ○, an unavailable parent in amber with ▲, escaped; the history and
  not-readable groups under All), offered with no ordering version;
- **the detail**: what the model is (kind, operation, parent and where it is,
  family, encoder, relation to the paper) without Model Facts' Task and
  Configuration rows; Trainer validation in its own card, Mode C in the
  records, never together; a registered task record under its own metric
  names; *unverifiable* never a number; D4's pick and the others' difference;
  the usage history segment (this checkpoint's entries, newest first); ledger
  states said, never a path;
- **read-only (D8)**: no list, D8's three conditions named, and "Recorded test
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

from src.evaluation.caveats import MODE_C_RESULT_LABEL, PRE_B1_NOTE  # noqa: E402
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

#: (architecture, run, file, epoch, val_mrr), newest first as the listing orders them.
CHECKPOINTS = [
    ("gat", "20261018-091200_run_8c2e7a", "model-12.pt", 12, 0.318),
    ("gat", "20261018-091200_run_8c2e7a", "model-08.pt", 8, 0.297),
    ("gat", "20261017-091500_run_3fa9c1", "model-05.pt", 5, 0.241),
    ("gat", "20261017-091500_run_3fa9c1", "model-03.pt", 3, 0.196),
    ("hgt", "20261016-221000_run_b71d04", "model-40.pt", 40, 0.352),
    ("sage", "20261015-083000_run_e4c2a9", "model-30.pt", 30, 0.205),
    ("sage", "20261015-083000_run_e4c2a9", "broken.pt", None, None),
    ("gat", "20261012-081500_run_51f0d3", "model-20.pt", 20, 0.224),
]


def _metrics(mrr, *, h10=0.4464, rank=58.7):
    return {"untruncated_mrr": mrr, "hits@10": h10, "hits@1": 0.1339, "hits@5": 0.3304,
            "mean_rank": rank}


@pytest.fixture(scope="module")
def demo_workspace(tmp_path_factory):
    from tests.fixtures.registered_cohort import write_demo_workspace

    return write_demo_workspace(tmp_path_factory.mktemp("models-build") / "demo")


@pytest.fixture
def world(demo_workspace, tmp_path, monkeypatch):
    """A workspaces root with one workspace and its runs, a registry with three
    versions (research_selection, institutional_acceptance, exploration), and
    records beside the checkpoints:

    - model-05: inst #1 (contested in the store), mygene #2 (non-deterministic,
      0.2307), mygene #3 (deterministic, 0.2310): D4 picks #3;
    - model-08: mygene, deterministic, 0.2250; model-20: mygene on CPU, 0.2195
      (regime differs); model-40: mygene 0.3402, contested; model-03: explore
      only; model-12 (served) and model-30: nothing; broken.pt: not readable.
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
                {"checkpoint_kind": "weights", "epoch": epoch, "logs": {"val_mrr": val},
                 "training_input_digests": {"split_manifest": fingerprint}}, path)
        stamp = now - index * 3600
        os.utime(path, (stamp, stamp))
        paths[name] = path
    registry = tmp_path / "registry"
    mygene, inst, explore = (version_block(MYGENE, RESEARCH_ROLE),
                             version_block(INST, ACCEPTANCE_ROLE),
                             version_block(EXPLORE, "exploration"))
    inst_report = report_for(shas["model-05.pt"], inst, metrics=_metrics(0.2051, rank=41.2),
                             deterministic=False)
    register(registry, paths["model-05.pt"], inst_report, registered_at="2026-10-17T13:02:00Z")
    disagree(registry, inst_report, _metrics(0.2077, rank=41.0))
    register(registry, paths["model-05.pt"],
             report_for(shas["model-05.pt"], mygene, metrics=_metrics(0.2307), deterministic=False),
             registered_at="2026-10-17T14:12:00Z")
    det_path, det = register(registry, paths["model-05.pt"],
                             report_for(shas["model-05.pt"], mygene, metrics=_metrics(0.2310)),
                             registered_at="2026-10-17T15:31:00Z")
    register(registry, paths["model-08.pt"],
             report_for(shas["model-08.pt"], mygene, metrics=_metrics(0.2250, rank=60.1)))
    register(registry, paths["model-20.pt"],
             report_for(shas["model-20.pt"], mygene, metrics=_metrics(0.2195, rank=55.0),
                        device="cpu"))
    hgt = report_for(shas["model-40.pt"], mygene, metrics=_metrics(0.3402))
    register(registry, paths["model-40.pt"], hgt)
    disagree(registry, hgt, _metrics(0.3398))
    register(registry, paths["model-03.pt"],
             report_for(shas["model-03.pt"], explore, metrics=_metrics(0.18)))
    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(root))
    monkeypatch.setenv("SHEPHERD_REGISTRY", str(registry))
    monkeypatch.setenv("SHEPHERD_ORDERING_ROLES", f"{RESEARCH_ROLE},{ACCEPTANCE_ROLE}")
    served = {"value": _status(workspace, shas["model-12.pt"])}
    monkeypatch.setattr(ml, "read_status", lambda: (served["value"], None))
    monkeypatch.setattr(mp, "read_status", lambda: (served["value"], None))
    return {"root": root, "workspace": workspace, "registry": registry, "paths": paths,
            "shas": shas, "served": served, "det_report": det_path, "det": det,
            "mygene_block": mygene}


def _status(workspace, sha):
    kg = json.loads((workspace / "split_manifest.json").read_text())["artifacts"]["kg"]
    return {
        "initialized": True, "current_data_dir": str(workspace), "eta_configured": 0.7,
        "checkpoint_meta": {"epoch": 12},
        "served_identity": {
            "scoring_mode": "gnn_only",
            "checkpoint": {"state": "verified" if sha else "absent", "sha256": sha},
            "graph": {"state": "verified", "kg_sha256": kg,
                      "split_manifest_sha256": file_sha256(workspace / "split_manifest.json")},
            "shortest_paths": {"state": "absent", "build_id": None, "max_hops": None},
            "dataset_record": {"state": "verified", "origin": "synthetic", "sha256": "e" * 64},
            "settings": {"device": "cpu", "device_requested": "cpu", "eta": 1.0,
                         "max_path_length": 4, "path_length_penalty": 0.5,
                         "aggregation_method": "max", "max_paths_per_candidate": 10,
                         "use_direct_path_optimization": True,
                         "phenotype_normalisation_version": 1},
        },
    }


def _key(world, name):
    for arch, run, file, *_rest in CHECKPOINTS:
        if file == name:
            return f"{arch}/{run}/{file}"
    raise KeyError(name)


#: Model bundles beside the ShepherdGNN files (which are history: every Mode C
#: record is theirs): a base and its disease and gene children, filed under
#: ``gatv2``, older than every file above so the orders over those stay.
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


def _ordered(world, monkeypatch, metric="untruncated_mrr", task=None, scoring=None):
    listing, data = _read(world, monkeypatch)
    ctl = mp.controls(listing, data, mp.BY_SCORE, MYGENE, metric, scoring, task)
    view, ordering = mp.shape(listing, data, ctl)
    return listing, data, ctl, view, ordering


# --------------------------------------------------------------------------- ordering per role
def test_ordering_is_offered_only_for_a_listed_role_and_never_for_acceptance(world, monkeypatch):
    listing, data = _read(world, monkeypatch)
    offers = {v["label"]: v["offers"] for v in data["versions"]}
    assert offers == {MYGENE: True, INST: False, EXPLORE: False}   # listed, acceptance still off
    ctl = mp.controls(listing, data, mp.BY_SCORE, None, None, None, None)
    assert (ctl["order"], ctl["version"]) == (mp.BY_SCORE, MYGENE)
    assert mp.version_choices(data) == [(f"{MYGENE} (role {RESEARCH_ROLE})", MYGENE)]
    html = mp.options_html(data, ctl)
    assert f"<b>{INST}</b> (role {r.ident_html(ACCEPTANCE_ROLE)}) — {e(mp.ROLE_NEVER)}" in html
    assert f"<b>{EXPLORE}</b> (role {r.ident_html("exploration")}) — {e(mp.ROLE_NOT_LISTED)}" in html
    assert e(mp.ROLE_NEVER_LISTED) in html                   # listed, and ignored
    assert mp.INDEPENDENCE_NOTE in mp.independence_html(data)  # wherever ordering is offered
    # A version the role does not offer cannot be asked for: it falls back.
    assert mp.controls(listing, data, mp.BY_SCORE, INST, None, None, None)["version"] == MYGENE


def test_acceptance_is_said_ignored_only_when_the_setting_lists_it(world, monkeypatch):
    """§4.9: "(listed …, and ignored)" is true only when the deployment lists
    institutional_acceptance; otherwise the reason alone."""
    monkeypatch.setenv("SHEPHERD_ORDERING_ROLES", RESEARCH_ROLE)
    listing, data = _read(world, monkeypatch)
    assert not data["roles_note"]
    html = mp.options_html(data, mp.controls(listing, data, mp.BY_SCORE, MYGENE, None, None,
                                             None))
    assert f"<b>{INST}</b> (role {r.ident_html(ACCEPTANCE_ROLE)}) — {e(mp.ROLE_NEVER)}<" in html
    assert e(mp.ROLE_NEVER_LISTED) not in html and "ignored" not in html


def test_a_role_the_setting_no_longer_lists_is_refused_by_the_ordering_itself(world, monkeypatch):
    """The version list said the role offers ordering, the setting read with
    it does not list the role (it changed in between, say): WP4's function
    re-checks the role against the setting, never against the version's own
    role, and refuses; the list stays newest first and says why."""
    listing, data = _read(world, monkeypatch)
    data["roles"] = [ACCEPTANCE_ROLE]
    ctl = mp.controls(listing, data, mp.BY_SCORE, MYGENE, None, None, None)
    assert (ctl["order"], ctl["version"]) == (mp.BY_SCORE, MYGENE)
    view, ordering = mp.shape(listing, data, ctl)
    assert ordering is None and "groups" not in view and view["note"] == mp.LOAD_NOTE
    assert f"not offered for role {RESEARCH_ROLE!r}" in view["refused"]


def test_with_no_role_listed_the_list_is_newest_first_and_says_why(world, monkeypatch):
    monkeypatch.delenv("SHEPHERD_ORDERING_ROLES")
    listing, data = _read(world, monkeypatch)
    ctl = mp.controls(listing, data, mp.BY_SCORE, MYGENE, None, None, None)
    assert ctl["order"] == mp.NEWEST and ctl["version"] is None
    view, ordering = mp.shape(listing, data, ctl)
    assert ordering is None and "groups" not in view and view["note"] == mp.LOAD_NOTE
    html = mp.options_html(data, ctl)
    assert "Ordering by test score is not offered" in html
    assert mp.independence_html(data) == ""


def test_an_ordering_the_records_refuse_stays_newest_first_and_says_why(world, monkeypatch):
    """The version's role as listed is not the one its records name (a stale
    list, say): WP4's function refuses, and the page says so."""
    listing, data = _read(world, monkeypatch)
    for version in data["versions"]:
        if version["label"] == MYGENE:
            version["role"] = "other_role"
    data["roles"] = [*data["roles"], "other_role"]
    ctl = mp.controls(listing, data, mp.BY_SCORE, MYGENE, None, None, None)
    view, ordering = mp.shape(listing, data, ctl)
    assert ordering is None and "groups" not in view
    assert "name role" in view["refused"]
    assert "Not ordered, so the list stays newest first" in mp.options_html(
        data, ctl, view["refused"])


# --------------------------------------------------------------------------- D4
def test_the_order_is_d4s_with_its_groups_and_the_regime_on_each_ordered_row(world, monkeypatch):
    listing, data, ctl, view, ordering = _ordered(world, monkeypatch)
    titles = [g["title"] for g in view["groups"]]
    assert titles == [f"Ordered by Mode C MRR (untruncated) on {MYGENE} · highest first",
                      mp.GROUP_CONTESTED, mp.GROUP_NOT_COMPARED, mp.GROUP_NOT_TESTED]
    keys = [g["keys"] for g in view["groups"]]
    assert keys[0] == [_key(world, n) for n in ("model-05.pt", "model-08.pt", "model-20.pt")]
    assert keys[1] == [_key(world, "model-40.pt")]               # contested: left out
    assert keys[2] == [_key(world, "broken.pt")]
    assert keys[3] == [_key(world, n) for n in ("model-12.pt", "model-03.pt", "model-30.pt")]
    cells = view["cells"]
    first = cells[_key(world, "model-05.pt")]
    # "record #3", as Test and the records say it, never a bare "#3" that reads
    # as a rank beside an ordering (M5-3 review UI-7).
    assert "0.2310" in first and "record #3 · D4 rule" in first
    assert "CUDA · deterministic" in _text(first)
    assert "regime differs" not in first and "models-regime" not in first
    cpu = cells[_key(world, "model-20.pt")]
    assert "CPU · deterministic" in _text(cpu) and "regime differs" in cpu
    # The mark's explanation is a ⓘ beside it, reachable by keyboard and touch.
    assert re.search(r"regime differs</span>\s*<button[^>]*data-help=\"models-regime\"", cpu), cpu
    # The regime wraps after its dot (a narrow column never scrolls sideways),
    # but never inside a word or between the device and its dot.
    regime = re.search(r'<div class="shep-sub shep-mp-regime">(.*?)</div>', first).group(1)
    assert regime == ('<span class="shep-nowrap">CUDA&nbsp;·</span> '
                      '<span class="shep-nowrap">deterministic</span>')
    assert "non-deterministic" in mp.regime_html("cuda:0", False)
    assert '<span class="shep-nowrap">non-deterministic</span>' in mp.regime_html("cuda", False)
    assert "contested — not ordered" in cells[_key(world, "model-40.pt")]
    assert view["column"] == f"Mode C MRR, {MYGENE} · regime"
    # A checkpoint listed after the page read its records is never said untested.
    assert view["rest_title"] == mp.GROUP_UNREAD
    later = {**listing, "items": [*listing["items"], {**listing["items"][0], "filename": "new.pt"}]}
    assert mp.GROUP_UNREAD in ml.list_html(later, None, ml.PRESETS["research"], view=view)
    assert ordering == {"version": MYGENE,
                        "version_sha256": next(v["sha256"] for v in data["versions"]
                                               if v["label"] == MYGENE),
                        "metric": "untruncated_mrr",
                        "scoring_semantics_digest": world["det"]["scoring_semantics_digest"]}
    assert view["note"] == mp.LOAD_NOTE_ORDERED.format(metric="Mode C MRR (untruncated)",
                                                       version=MYGENE)
    # The shared list draws it: groups in order, the column, the note.
    html = ml.list_html(listing, None, ml.PRESETS["research"], view=view)
    assert html.index("Ordered by") < html.index("run_3fa9c1") < html.index(mp.GROUP_CONTESTED)


def test_mean_rank_orders_lowest_first(world, monkeypatch):
    _listing, _data, _ctl, view, _ordering = _ordered(world, monkeypatch, metric="mean_rank")
    assert view["groups"][0]["title"].endswith("· lowest first")
    assert view["groups"][0]["keys"] == [_key(world, n)
                                         for n in ("model-20.pt", "model-05.pt", "model-08.pt")]


def test_the_task_filter_orders_only_what_is_shown(world, monkeypatch):
    """By test score keeps its policy (Mode C records only, every one a history
    file's): History keeps the Mode C order; Disease has no ordered group, its
    rows not tested on the version; an unknown filter is All."""
    _add_bundles(world)
    listing, _data, ctl, view, _ordering = _ordered(world, monkeypatch, task="history")
    assert ctl["task"] == "history" and view["task"] == "history"
    assert view["groups"][0]["keys"] == [_key(world, n)
                                         for n in ("model-05.pt", "model-08.pt", "model-20.pt")]
    _listing, _data, ctl, view, ordering = _ordered(world, monkeypatch, task="disease")
    assert [g["title"] for g in view["groups"]] == [mp.GROUP_NOT_TESTED]
    assert view["groups"][0]["keys"] == [bundle_key("disease")]
    assert ordering is not None                   # a Load from it is still a Load from the order
    assert mp.controls(listing, _data, mp.BY_SCORE, MYGENE, None, None, "gat")["task"] == "all"


def test_a_second_scoring_semantics_is_a_choice_never_interleaved(world, monkeypatch):
    other = report_for(world["shas"]["model-12.pt"], world["mygene_block"],
                       metrics=_metrics(0.9), score_semantics="another scoring")
    register(world["registry"], world["paths"]["model-12.pt"], other)
    listing, data, ctl, view, _ordering = _ordered(world, monkeypatch)
    assert [n for _d, n in ctl["scoring_groups"]] == [4, 1]     # most checkpoints first
    assert ctl["scoring"] == world["det"]["scoring_semantics_digest"]
    assert _key(world, "model-12.pt") in view["groups"][2]["keys"]
    assert "tested under different scoring semantics" in view["cells"][_key(world, "model-12.pt")]
    picked = mp.controls(listing, data, mp.BY_SCORE, MYGENE, None, ctl["scoring_groups"][1][0],
                         None)
    _view, ordering = mp.shape(listing, data, picked)
    assert _view["groups"][0]["keys"] == [_key(world, "model-12.pt")]
    assert ordering["scoring_semantics_digest"] == ctl["scoring_groups"][1][0]
    assert len(mp.scoring_choices(ctl["scoring_groups"])) == 2


# --------------------------------------------------------------------------- D5
def test_a_contested_state_that_cannot_be_checked_is_never_uncontested(world, monkeypatch):
    import src.evaluation.report_store as report_store

    def broken(version, **kwargs):
        raise OSError("store unreadable")

    monkeypatch.setattr(report_store, "scan_report_store", broken)
    listing, data, ctl, view, _ordering = _ordered(world, monkeypatch)
    assert {s["state"] for s in data["contest"].values()} == {"cannot_be_checked"}
    # Their own group: never ordered, and never under a heading that says the
    # reports disagree when that is not known.
    titles = [g["title"] for g in view["groups"]]
    assert titles[:1] == [mp.GROUP_UNCHECKED] and mp.GROUP_CONTESTED not in titles
    assert view["groups"][0]["keys"] == [_key(world, n) for n in (
        "model-08.pt", "model-05.pt", "model-40.pt", "model-20.pt")]
    assert "contested status cannot be checked — not ordered" in view["cells"][
        _key(world, "model-05.pt")]
    html = ml.list_html(listing, None, ml.PRESETS["research"], view=view)
    assert mp.GROUP_UNCHECKED in html and mp.GROUP_CONTESTED not in html
    detail = mp.detail_parts(listing, data, _key(world, "model-05.pt"), ctl)
    assert "contested status cannot be checked" in detail["records"]
    assert "registered #3" in detail["records"] and "shep-tag-ok" not in detail["records"]


def test_a_key_the_page_holds_no_contested_state_for_is_not_ordered(world, monkeypatch):
    """A record key with no state in the page's data (read after the store was
    asked, say) is taken as "cannot be checked", never as uncontested."""
    listing, data = _read(world, monkeypatch)
    data["contest"] = {}
    ctl = mp.controls(listing, data, mp.BY_SCORE, MYGENE, None, None, None)
    view, _ordering = mp.shape(listing, data, ctl)
    assert [g["title"] for g in view["groups"]][:1] == [mp.GROUP_UNCHECKED]
    assert "contested status cannot be checked — not ordered" in view["cells"][
        _key(world, "model-05.pt")]
    # D5 in the detail too: every record says its state was not read, none is
    # shown as uncontested (green).
    records = mp.detail_parts(listing, data, _key(world, "model-05.pt"), ctl)["records"]
    assert records.count("contested status cannot be checked: it was not read") == 3
    assert "shep-tag-ok" not in records and "runs disagree" not in records


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
    view (the API's answer) both say "runs disagree"; the value is never
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
    report["authoritative_metrics"]["untruncated_mrr"] = value
    store_bytes(world["registry"], json.dumps(report, sort_keys=True, allow_nan=True).encode(),
                version=version_of(kept), checkpoint_sha256=world["shas"]["model-40.pt"])

    operator = contest_states([record], CohortVersions.listed())[key_text(record_key(record))]
    assert operator["state"] == "contested"
    operator_html = mp.contest_html(operator)
    assert "runs disagree — investigate" in operator_html and f"{shown} (run" in operator_html
    assert "nan" not in operator_html.lower().replace("not a finite number", "")

    _served_read_only(world, monkeypatch, name="model-40.pt")
    data = mp.read_readonly()
    (view,) = data["answer"]["records"]
    assert view["contest"]["state"] == "contested"
    readonly_html = mp.contest_html(data["contest"][key_text(record_key(record))])
    assert "runs disagree — investigate" in readonly_html
    assert "not a finite number (run" in readonly_html
    assert "contested #" in mp.readonly_parts(data)["records"]


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
def test_a_load_from_the_ordered_list_writes_the_usage_history_entry(world, monkeypatch):
    """The ordered rows are history files (every Mode C record is one), which
    are never run from the UI (K6 decision 1): a Load of one is refused before
    anything is read. A task model loaded from the ordered list (not tested on
    the version) writes the usage history entry."""
    from src.api.routes.pipeline import PipelineReloadResponse, PipelineStatusResponse
    from src.evaluation.report_store import read_usage_history
    from tests.fixtures.report_store import version_of

    shas = _add_bundles(world)
    listing, _data, _ctl, view, ordering = _ordered(world, monkeypatch)
    plan, html = ml.request_load(listing, _key(world, "model-05.pt"), ordering,
                                 operator_request(monkeypatch))
    assert plan is None and ml.HISTORY_REFUSAL in html
    key = bundle_key("disease")
    assert key in view["groups"][-1]["keys"]                   # not tested on this version
    plan, _html = ml.request_load(listing, key, ordering, operator_request(monkeypatch))
    sha = shas["disease"]

    def reload(request):
        assert request.task == "disease"
        return PipelineReloadResponse(success=True, message="ok", status=PipelineStatusResponse(
            **_status(world["workspace"], sha)))

    step = ml.finish_load(plan, operator_request(monkeypatch), reload=reload)
    assert "usage history records this load" in step.confirm_html
    entries, problems = read_usage_history(version_of(world["det_report"]))
    assert problems == []
    assert [(e["kind"], e["checkpoint_sha256"], e["metric"], e["scoring_semantics_digest"],
             e["loaded"]) for e in entries] == [
        ("model_load", sha, "untruncated_mrr", world["det"]["scoring_semantics_digest"], True)]


def test_newest_first_gives_the_load_no_ordering(world, monkeypatch):
    """Newest first: no ordering; under All a Kind column says what each row
    is (`renderers.record_kind_text`), under one filter none."""
    _add_bundles(world)
    listing, data = _read(world, monkeypatch)
    ctl = mp.controls(listing, data, mp.NEWEST, MYGENE, None, None, None)
    view, ordering = mp.shape(listing, data, ctl)
    assert ordering is None and view["task"] == "all" and view["note"] == mp.LOAD_NOTE
    assert view["column"] == mp.KIND_COLUMN and "groups" not in view
    cells = view["cells"]
    assert cells[_key(world, "model-05.pt")] == "History · never run"
    assert cells[_key(world, "broken.pt")] == "Record not read"
    assert (cells[bundle_key("base")], cells[bundle_key("gene")]) == ("Base", "Gene")
    ctl = mp.controls(listing, data, mp.NEWEST, MYGENE, None, None, "disease")
    assert mp.shape(listing, data, ctl) == ({"task": "disease", "note": mp.LOAD_NOTE}, None)



def test_a_record_of_another_or_no_mode_is_never_shown_as_mode_c(world, monkeypatch):
    """A record or usage entry whose mode this version does not describe, or
    whose mode cannot be read, is named for what it is: never under Mode C's
    label, note, metric names or coverage words (the internal review of Batch 3,
    ui_api-4). An entry written before `mode` existed still reads as Mode C."""
    listing, data, ctl, _view, _ordering = _ordered(world, monkeypatch)
    record = next(r for entry in data["ledgers"].values() for r in entry.get("records") or [])
    view = mp.record_view(record)
    assert view["mode"] == "C"
    for mode, label in (("E", "Mode E (a measurement mode this version does not describe)"),
                        (None, "Measurement mode not recorded")):
        text = _text(mp.records_block_html([{**view, "mode": mode}], {}))
        assert label in text
        assert MODE_C_RESULT_LABEL not in text and PRE_B1_NOTE not in text
        assert "Mode C" not in text and "source cases ranked" not in text
        assert "not described" in text and "what they are is not described" in text
        assert "The Trainer validation figure is not among these records' metrics." in text
    entry = {"kind": "test_run", "at": "2026-10-26T12:00:00Z", "version": "v", "run_id": "r",
             "metrics": {"untruncated_mrr": 0.5}, "n_ranked": 3, "device": "cpu",
             "deterministic": True, "record_requested": False}
    assert "Mode C MRR (untruncated) 0.5000 over 3 ranked cases" in _text(
        mp.usage_entry_html(entry))
    for bad in (None, "", 7):
        shown = _text(mp.usage_entry_html({**entry, "mode": bad}))
        assert "Measurement mode not recorded" in shown and "Mode C" not in shown

# --------------------------------------------------------------------------- the detail
def test_the_detail_keeps_validation_and_mode_c_apart(world, monkeypatch):
    listing, data, ctl, _view, _ordering = _ordered(world, monkeypatch)
    detail = mp.detail_parts(listing, data, _key(world, "model-05.pt"), ctl)
    # The title names the row, focusable: narrow, choosing a row waits for this
    # title before it scrolls to the detail and moves the focus there.
    assert f'<h2 tabindex="-1" data-key="{e(_key(world, "model-05.pt"))}">' in detail["header"]
    assert "h2[data-key]" in mp.SCROLL_JS and "shep-mp-back" in detail["header"]
    assert "not in use" in detail["header"]
    assert "in use for diagnosis: model" in detail["header"]
    assert world["shas"]["model-12.pt"][:12] in detail["header"]
    top, records = detail["top"], detail["records"]
    assert world["shas"]["model-05.pt"] in top                  # the full fingerprint, Copy
    assert "Validation MRR (Trainer, top-20) 0.2410" in top and "Mode C" not in top
    assert "Validation MRR" not in records and "top-20" not in records and "0.2410" not in records
    assert MODE_C_RESULT_LABEL in records and "Mode C MRR (untruncated)" in records
    # D4's pick first, the other record of its group with its difference
    assert records.index("registered #3") < records.index("registered #2")
    assert "D4 rule picks this record" in records
    assert "Δ Mode C MRR (untruncated) -0.0003 vs #3 · deterministic off" in records
    # the acceptance record is contested, with what the runs say
    assert "contested #1" in records and "runs disagree — investigate" in records
    assert "0.2051" in records and "0.2077" in records
    # the acceptance record's role says ordering is off for it, and only there
    note = "ordering off: awaits the formal evaluation rules"
    assert records.count(note) == 1
    assert records.index(INST) < records.index(note) < records.index("contested #1")
    # The heading names no mode: each record's table carries its own (PLAN_B05 §6).
    assert detail["seg"][0] == ("Recorded test results · 3 · 1 contested", mp.RECORDS)
    # a role code is never broken inside its name in its narrow column: not
    # "research_selectio" / "n" (1440x900), nor "research_" / "selection" (M5-3);
    # it stays whole, in a titled ellipsis where it does not fit
    assert f"role {r.ident_html('research_selection')}" in records
    assert f"role {r.ident_html('institutional_acceptance')}" in records
    assert "<wbr>" not in records


def test_d4s_pick_comes_first_in_the_detail_before_a_newer_record(world, monkeypatch):
    sha = world["shas"]["model-08.pt"]
    _path, newer = register(world["registry"], world["paths"]["model-08.pt"],
                            report_for(sha, world["mygene_block"], deterministic=False,
                                       metrics=_metrics(0.2262, rank=59.0)),
                            registered_at="2026-10-19T10:00:00Z")
    listing, data, ctl, view, _ordering = _ordered(world, monkeypatch)
    key = _key(world, "model-08.pt")
    first = data["ledgers"][key]["records"][0]["registration"]["sequence"]
    later = newer["registration"]["sequence"]
    assert later > first and f"record #{first} · D4 rule" in view["cells"][key]
    records = mp.detail_parts(listing, data, key, ctl)["records"]
    assert records.index(f"registered #{first}") < records.index(f"registered #{later}")
    assert records.index("D4 rule picks this record") < records.index(f"registered #{later}")


def test_unverifiable_is_a_word_and_never_a_count(world, monkeypatch):
    listing, data, ctl, _view, _ordering = _ordered(world, monkeypatch)
    records = mp.detail_parts(listing, data, _key(world, "model-08.pt"), ctl)["records"]
    assert "Training disease overlap: unverifiable" in records
    assert "there is no count" in records
    line = records[records.index("R10, training inputs"):]
    line = line[:line.index("</span>")]
    assert " of 50 " not in line and " of 80 " not in line
    text = mp.r10_label_text("disease_overlap", "D", {"state": "unverifiable", "k": None,
                                                      "n": 50, "cases": None}, "not_found")
    assert "0" not in text.replace("D:", "")


def test_the_usage_history_segment_lists_this_checkpoints_entries_newest_first(world, monkeypatch):
    from datetime import datetime, timezone

    from src.evaluation.report_store import record_model_load, record_test_run
    from tests.fixtures.report_store import version_of

    version = version_of(world["det_report"])
    sha, other = world["shas"]["model-05.pt"], world["shas"]["model-08.pt"]
    record_test_run(version, run_id="20261017T152000Z-aaaaaaaaaaaa-11111111",
                    report_sha256="f" * 64, checkpoint_sha256=sha, metrics=_metrics(0.2310),
                    n_ranked=80, deterministic=True, device="cuda", record_requested=True,
                    at=datetime(2026, 10, 17, 15, 20, tzinfo=timezone.utc))
    record_model_load(version, checkpoint_sha256=sha, metric="untruncated_mrr",
                      scoring_semantics_digest="d" * 64, loaded=False, why="refused",
                      at=datetime(2026, 10, 18, 8, 0, tzinfo=timezone.utc))
    record_model_load(version, checkpoint_sha256=other, metric="untruncated_mrr",
                      scoring_semantics_digest="d" * 64, loaded=True)
    record_model_load(version, checkpoint_sha256="a" * 64, metric="untruncated_mrr",
                      scoring_semantics_digest="d" * 64, loaded=True)   # listed nowhere here
    listing, data, ctl, _view, _ordering = _ordered(world, monkeypatch)
    # Only the entries naming a listed checkpoint are kept for the page.
    assert set(data["usage"]["entries"]) == {sha, other}
    detail = mp.detail_parts(listing, data, _key(world, "model-05.pt"), ctl)
    usage = detail["usage"]
    assert detail["seg"][1] == ("Usage history · 2", mp.USAGE)
    assert usage.index("2026-10-18 08:00 UTC") < usage.index("2026-10-17 15:20 UTC")
    assert "not loaded: refused" in usage and "Test run" in usage
    assert "registration asked" in usage and MYGENE in usage
    assert other[:12] not in usage


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
    listing, data, ctl, view, _ordering = _ordered(world, monkeypatch)
    key = _key(world, "model-12.pt")
    assert data["ledgers"][key]["state"] == "not_read"
    assert key in view["groups"][2]["keys"]
    assert "ledger not readable — not ordered" in view["cells"][key]
    detail = mp.detail_parts(listing, data, key, ctl)
    assert "was not read, so none of its records are shown" in detail["records"]
    assert str(world["root"]) not in detail["records"] + detail["top"]
    none = mp.detail_parts(listing, data, _key(world, "model-30.pt"), ctl)
    assert "No ledger" in none["records"] and "never evaluated" in none["records"]


def test_markup_in_a_run_name_is_escaped(world, monkeypatch):
    from src.utils.checkpoint_io import write_checkpoint

    # One run named by its start time, one not (its whole name is then the title).
    for name in ("20261019-090000_run_<i>x", "<i>y_run"):
        run = world["workspace"] / "runs" / "gat" / name / "checkpoints"
        run.mkdir(parents=True)
        write_checkpoint({"checkpoint_kind": "weights", "epoch": 2, "logs": {}}, run / "best.pt")
    listing, data = _read(world, monkeypatch)
    ctl = mp.controls(listing, data, mp.NEWEST, None, None, None, None)
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
    token first, at most one separator; the task words are the list's own."""
    for choices in (mp.TASK_CHOICES, mp.ORDER_CHOICES):
        labels = [label for label, _value in choices]
        values = [value for _label, value in choices]
        assert len(set(labels)) == len(labels) and len(set(values)) == len(values)
        assert all(len(label) <= 40 and label.count(" · ") <= 1 for label in labels)
    assert mp.TASK_CHOICES == [("All", "all"), ("Base", "base"), ("Disease", "disease"),
                               ("Gene", "gene"), ("Patients-like-me", "patients_like_me"),
                               ("History", "history")]
    assert {value for _label, value in mp.TASK_CHOICES} <= set(ml.TASK_FILTERS)
    assert [label for label, _value in mp.ORDER_CHOICES] == ["Newest first", "By family",
                                                             "By test score"]
    title, text = mp.PAGE_HELP["models-ordering"]
    assert title == "Order" and text.split("\n")[0].startswith("How the list is ordered")


def test_by_family_draws_each_family_parent_first_with_its_lineage(world, monkeypatch):
    """By family (offered with no ordering version): one group per family,
    the base first and its children under it, indented, each with its kind,
    operation and exact parent; a child whose parent is not available says so
    in amber with ▲ and its words, escaped; under All the history files and the
    records not readable follow as two groups; a task's view keeps its base as
    a context row."""
    monkeypatch.delenv("SHEPHERD_ORDERING_ROLES")
    shas = _add_bundles(world, orphan_parent_run=EVIL)
    listing, data = _read(world, monkeypatch)
    assert not mp.offered_versions(data)
    ctl = mp.controls(listing, data, mp.BY_FAMILY, None, None, None, None)
    assert ctl["order"] == mp.BY_FAMILY
    view, ordering = mp.shape(listing, data, ctl)
    assert ordering is None and view["column"] == mp.LINEAGE_COLUMN
    titles = [g["title"] for g in view["groups"]]
    assert titles == ["Family from run 2026-10-01 00:00 · 3 files",
                      "Family from run 2026-10-04 00:00 · 1 file",
                      mp.GROUP_HISTORY, mp.GROUP_REFUSED]
    assert view["groups"][0]["keys"] == [bundle_key("base"), bundle_key("disease"),
                                         bundle_key("gene")]
    assert view["groups"][3]["keys"] == [_key(world, "broken.pt")]
    assert len(view["groups"][2]["keys"]) == 7                       # every ShepherdGNN file
    cells = view["cells"]
    child = cells[bundle_key("disease")]
    assert 'class="shep-mp-lin shep-mp-d1"' in child and "Disease" in child
    assert f"Finetune of <code class=\"shep-fpv\">{shas['base'][:12]}</code>" in child
    assert 'class="shep-mp-lin shep-mp-d0"' in cells[bundle_key("base")]
    assert "New run" in cells[bundle_key("base")]
    orphan = cells["gatv2/20261004-000000_disease_dddddd/last.pt"]
    assert ('<span class="shep-mp-warn"><span aria-hidden="true">▲</span> parent unavailable: '
            in orphan)
    assert EVIL not in orphan and html_lib.escape(EVIL) in orphan
    assert "History · never run" in cells[_key(world, "model-05.pt")]
    # The shared list draws them: groups in order, the Lineage column.
    html = ml.list_html(listing, None, ml.PRESETS["research"], view=view)
    assert ">Lineage</th>" in html and EVIL not in html
    assert html.index("Family from run 2026-10-01") < html.index(mp.GROUP_HISTORY)
    # A task's view: its base as a context row, grey, ○ and its words.
    ctl = mp.controls(listing, data, mp.BY_FAMILY, None, None, None, "gene")
    view, _ordering = mp.shape(listing, data, ctl)
    assert [g["keys"] for g in view["groups"]] == [[bundle_key("base"), bundle_key("gene")]]
    assert ('<div class="shep-sub shep-mp-ctx"><span aria-hidden="true">○</span> '
            'Context: parent of a shown file</div>') in view["cells"][bundle_key("base")]
    assert mp.shown_view(listing, ctl) == {"keys": [bundle_key("base"), bundle_key("gene")]}
    # History alone; the CSS gives the marks their status tokens.
    ctl = mp.controls(listing, data, mp.BY_FAMILY, None, None, None, "history")
    view, _ordering = mp.shape(listing, data, ctl)
    assert [g["title"] for g in view["groups"]] == [mp.GROUP_HISTORY]
    assert ".shep-mp-warn {{ color: var(--shep-status-warn); }}".replace(
        "{{", "{").replace("}}", "}") in mp.PAGE_CSS.replace(f"{mp._PAGE} ", "")
    assert "var(--shep-status-neutral)" in mp.PAGE_CSS
    # Without a family view (a listing read without one): newest first, said.
    bare = {k: v for k, v in listing.items() if k != "family"}
    view, _ordering = mp.shape(bare, data, mp.controls(bare, data, mp.BY_FAMILY, None, None,
                                                        None, None))
    assert view["refused"] == mp.FAMILY_NOT_READ and "groups" not in view


def test_the_detail_says_what_the_model_is_and_its_relation_to_the_paper(world, monkeypatch):
    """The detail's identity (README §5): kind, operation, the exact parent
    and where it is, family, encoder and axis 2 (the fields that differ, or
    "matches the reference recipe" with what that recipe is); Model Facts'
    Task and Configuration rows are not repeated; a history file says its
    kind only; Load is refused for a base with its reason."""
    shas = _add_bundles(world)
    listing, data = _read(world, monkeypatch)
    ctl = mp.controls(listing, data, mp.NEWEST, None, None, None, None)
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
    history = mp.detail_parts(listing, data, _key(world, "model-05.pt"), ctl)["top"]
    ident = history[history.index('<div class="shep-kv shep-mp-ident">'):]
    ident = ident[:ident.index("</span></div>") + 13]
    assert re.findall(r'<span class="shep-k">([^<]*)</span>', ident) == ["Kind"]
    assert "History · never run" in ident
    # Axis 2 for a reference spec, and for a parent elsewhere or unavailable.
    item = {"record": "model", "learning_stage": "finetune", "task": "gene",
            "operation": "finetune", "family_id": BASE_RUN, "encoder": "gatv2",
            "differences": (), "parent_sha256": "a" * 64, "parent_run": EVIL}
    html = mp.identity_html(item, {"parent_state": "elsewhere", "parent_workspace": EVIL})
    assert "matches the reference recipe" in html and mp.REFERENCE_ANCHOR in _text(html)
    assert EVIL not in html and "in workspace <b>&lt;img" in html
    html = mp.identity_html(item, {"parent_state": "unavailable", "parent_problem": EVIL})
    assert "▲</span> parent unavailable: &lt;img" in html and EVIL not in html
    assert "none: a new run" in mp.identity_html({**item, "parent_sha256": None}, None)
    # Load is off for the base, with its reason (the shared list's).
    assert ml.load_refusal(ml.find_item(listing, bundle_key("base"))) == ml.BASE_REFUSAL


def test_a_registered_task_record_shows_its_own_named_metrics(world, monkeypatch):
    """A disease, gene or patients-like-me record (K6 Test registers them) is
    shown in its own table under its task's metric names
    (`caveats.metric_text`): pooled MRR beside pooled top-10 or macro
    recall@10, the other figures after them in E1/E5/E2's order, its coverage
    in its own words; never "not described" under bare headers."""
    listing, data, ctl, _view, _ordering = _ordered(world, monkeypatch)
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
        assert "source cases ranked by the model" in text
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
    parts = mp.detail_parts(listing, data, None, mp.controls(listing, data, None, None, None,
                                                             None, None))
    assert mp.READONLY_TITLE in parts["header"] and "opens no checkpoint" in parts["header"]
    assert sha in parts["top"] and parts["seg"] is None
    records = parts["records"]
    assert "registered #3" in records and "contested #1" in records
    # each record's determinism, from the environment class the route answers
    assert 'registered #3</span> <span class="shep-sub">deterministic on</span>' in records
    assert 'registered #2</span> <span class="shep-sub">deterministic off</span>' in records
    assert "runs disagree — investigate" in records and mp.NO_LOAD_HERE in records
    assert str(world["root"]) not in records and str(world["registry"]) not in records
    header = mp.left_header_html(listing, data, "deployment_mode")
    assert "Model list unavailable: read-only, deployment mode not set to M1." in header
    assert e(mp.D8_CONDITIONS) in header and "shep-mp-ro" in header
    assert "Nothing was done" not in header and "Refused" not in header      # said once (UI-6)
    assert "read-only, request not local." in mp.left_header_html(listing, data, "not_loopback")
    view, ordering = mp.shape(listing, data, mp.controls(listing, data, mp.BY_SCORE, MYGENE,
                                                         None, None, None))
    assert ordering is None and view == {"note": ""}


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
    from src.api.routes.evaluations import _recorded_determinism

    assert _recorded_determinism({"regime": {"deterministic_algorithms": True}}) is True
    assert _recorded_determinism({"regime": {"deterministic_algorithms": False}}) is False
    assert _recorded_determinism({"regime": {"deterministic_algorithms": "yes"}}) is None
    assert _recorded_determinism({"regime": None}) is None and _recorded_determinism({}) is None


def _served_read_only(world, monkeypatch, name="model-05.pt"):
    import src.api.main as api_main
    from src.utils.fingerprint import ReadIdentity

    checkpoint = world["paths"][name]
    world["served"]["value"] = _status(world["workspace"], world["shas"][name])
    monkeypatch.setattr(api_main.app_state, "pipeline", _ServedPipeline(
        ReadIdentity(checkpoint.resolve(), world["shas"][name])))
    monkeypatch.delenv("SHEPHERD_DEPLOYMENT_MODE", raising=False)


def test_read_only_names_no_server_path_when_a_report_store_cannot_be_read(world, monkeypatch):
    """A report in the version's store that cannot be read makes its keys'
    state "cannot be checked", and the store's own words name the file by its
    absolute path: the operator's view keeps them, a read-only request never
    gets them (D8)."""
    from src.evaluation.report_store import scan_report_store
    from src.evaluation.sidecar import record_key
    from tests.fixtures.report_store import store_bytes, version_of

    _served_read_only(world, monkeypatch)
    version = version_of(world["det_report"])
    bad = store_bytes(world["registry"], b"{not json", version=version,
                      checkpoint_sha256=world["shas"]["model-05.pt"])
    status = scan_report_store(version).status(record_key(dict(world["det"])))
    assert status.state == "cannot_be_checked" and str(bad) in " ".join(status.problems)
    records = mp.readonly_parts(mp.read_readonly())["records"]
    assert 'registered #3</span>' in records and "contested status cannot be checked" in records
    assert str(world["registry"].parent) not in records and "report.json" not in records
    assert "could not be read" not in records


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
    assert len(states) >= 4                    # listing, selected, page data, view, ordering
    inputs = [getattr(i, "elem_id", None) for i in opened.inputs]
    assert {"shep-mp-order", "shep-mp-version", "shep-mp-metric", "shep-mp-scoring",
            "shep-mp-task"} <= set(inputs)
    stale = _sync_states(demo)["stale"]
    assert stale in opened.outputs
    out = opened.fn("demo", False, None, None, None, mp.BY_SCORE, MYGENE, "untruncated_mrr", None,
                    mp.ALL_TASKS, mp.RECORDS, "idle", operator_request(monkeypatch))
    assert len(out) == len(opened.outputs)
    by_output = dict(zip(map(id, opened.outputs), out, strict=True))
    listing = out[0]
    data = next(v for v in out if isinstance(v, dict) and "listing_key" in v)
    assert data["listing_key"] == mp.listing_key(listing)
    views = [v for v in out if isinstance(v, dict) and "groups" in v]
    assert views and views[0]["groups"][0]["title"].startswith("Ordered by")
    orderings = [v for v in out if isinstance(v, dict) and v.get("version") == MYGENE]
    assert orderings and orderings[0]["metric"] == "untruncated_mrr"
    assert by_output[id(by_id["shep-mp-order"])]["interactive"] is True
    assert by_output[id(stale)] is False                   # read now: nothing is stale
    assert by_output[id(by_id["shep-mp-wsline"])] == mp.workspace_line_html(listing)
    # Shown while a Load runs: read now, and again once the Load has finished.
    out = opened.fn("demo", False, None, None, None, mp.NEWEST, None, None, None, None, mp.RECORDS,
                    "loading", operator_request(monkeypatch))
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
          order=mp.BY_SCORE, version=MYGENE, task=mp.ALL_TASKS, selected=None,
          seg=mp.RECORDS):
    sync = states["fn"]
    out = sync.fn(listing, data, stale, last, phase, order, version, "untruncated_mrr", None,
                  task, selected, seg, operator_request(monkeypatch), gr.EventData(trigger, None))
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
                                 "shep-mp-records", "shep-mp-usage")]
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
                trigger=_session_copy(by_id["shep-mp-order"]))
    view = out["page"][1]
    assert out["data"] == gr.update() and view["groups"][0]["title"].startswith("Ordered by")
    # A listing not read yet: read once.
    changed = {**listing, "served_sha": world["shas"]["model-05.pt"]}
    calls = []
    monkeypatch.setattr(mp, "read_page", lambda listing: calls.append(1) or data)
    _sync(states, monkeypatch, changed, data, trigger=states["listing"], order=mp.NEWEST)
    assert calls == [1]


def test_after_a_load_the_page_reads_again_and_selects_the_same_checkpoint(world, monkeypatch):
    """A Load from the ordered list that leaves the listing as it was (a
    refused reload here; the model in use reloaded is another): the page reads
    its own data again once the Load has finished, so the usage history shows
    the entry that Load wrote, and selects the same checkpoint again (the
    shared refresh cleared the selection), so Close returns to its detail."""
    from src.api.routes.pipeline import PipelineReloadResponse, PipelineStatusResponse

    demo, _ctx = _page()
    states = _sync_states(demo)
    shas = _add_bundles(world)
    listing, data, _ctl, _view, ordering = _ordered(world, monkeypatch)
    key, sha = bundle_key("disease"), shas["disease"]
    assert sha not in data["usage"]["entries"]
    plan, _html = ml.request_load(listing, key, ordering, operator_request(monkeypatch))
    served = world["shas"]["model-12.pt"]

    def refused(request):
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
    assert [(e["kind"], e["loaded"], e["why"]) for e in entries] == [
        ("model_load", False, "refused for this test")]
    page = dict(zip(map(id, states["fn"].outputs[3:]), out["page"], strict=True))
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    seg = page[id(by_id["shep-mp-seg"])]
    assert seg["choices"][1] == ("Usage history · 1", mp.USAGE) and seg["value"] == mp.USAGE
    usage = page[id(by_id["shep-mp-usage"])]
    assert "not loaded: refused for this test" in usage["value"] and usage["visible"] is True
    view = out["page"][1]
    assert "redraw" in view and view["groups"][0]["title"].startswith("Ordered by")
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
        *((by_id[c], "input") for c in ("shep-mp-order", "shep-mp-version", "shep-mp-metric",
                                        "shep-mp-scoring", "shep-mp-task"))}
    assert sync.trigger_mode == "always_last"
    assert sync.outputs[:3] == [states["stale"], states["last"], states["selected"]]
    assert sync.outputs[-5:] == [by_id["shep-mp-detail-hd"], by_id["shep-mp-detail-top"],
                                 by_id["shep-mp-seg"], by_id["shep-mp-records"],
                                 by_id["shep-mp-usage"]]
    assert not [fn for fn in demo.fns.values() if fn.name in ("_recompute", "_reorder", "_detail")]
    # "Reload with …" marks the page's data stale, at once, beside the shared Load.
    started = _fn(demo, "_load_started")
    assert started.targets == [(by_id["shep-models-research-reload"]._id, "click")]
    assert started.queue is False and started.outputs == [states["stale"]]
    assert started.fn() is True
    segment = _fn(demo, "_segment")
    assert segment.queue is False
    records, usage = segment.fn(mp.USAGE)
    assert records["visible"] is False and usage["visible"] is True


def test_the_page_registers_its_help_and_every_i_it_draws(world, monkeypatch):
    from src.webui.components.help import HelpRegistry, registered_keys_in
    from src.webui.shell import standalone_context

    registry = HelpRegistry()
    with gr.Blocks() as demo:
        mp.create_models_page(standalone_context(demo, registry))
    listing, data, ctl, view, _ordering = _ordered(world, monkeypatch)
    detail = mp.detail_parts(listing, data, _key(world, "model-05.pt"), ctl)
    rendered = ("".join(detail[k] for k in ("top", "records", "usage")) + mp.options_html(data, ctl)
                + "".join(view["cells"].values()) + mp.workspace_line_html(listing))
    assert 'data-help="models-regime"' in rendered            # the "regime differs" ⓘ
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
    region when they do not both fit; the version is a one-line select, what
    offers ordering is said once, and the versions without it are behind a
    closed disclosure."""
    demo, _ctx = _page()
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    assert isinstance(by_id["shep-mp-version"], gr.Dropdown)
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


def test_the_version_options_are_one_line_and_the_role_clause_is_said_once(world, monkeypatch):
    listing, data = _read(world, monkeypatch)
    ctl = mp.controls(listing, data, mp.BY_SCORE, None, None, None, None)
    assert all(mp.ROLE_ON not in label for label, _value in mp.version_choices(data))
    html = mp.options_html(data, ctl)
    assert html.count(e(mp.ROLE_ON)) == 1
    assert '<details class="shep-mp-offd"><summary>2 cohort versions without ordering' in html
    newest = mp.controls(listing, data, mp.NEWEST, None, None, None, None)
    assert e(mp.ROLE_ON) not in mp.options_html(data, newest)



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
