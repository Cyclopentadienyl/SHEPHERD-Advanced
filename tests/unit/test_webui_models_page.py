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
  first, the regime on each ordered row and "regime differs" where it does;
  the groups "Ordered by …", "Contested on this version · not ordered", "Not
  compared on this version · not ordered", "Not tested on this cohort version ·
  newest first";
- **contested records left out (D5)**, and a state that cannot be checked
  never read as uncontested;
- **the shared Load writes the usage history entry** for a Load from an ordered
  list (the page's ``ordering``), and none for newest first;
- **the detail**: Trainer validation in its own card, Mode C in the records,
  never together; *unverifiable* never a number; D4's pick and the others'
  difference; the usage history segment (this checkpoint's entries, newest
  first); ledger states said, never a path;
- **read-only (D8)**: no list, D8's three conditions named, and "Recorded test
  results of the model in use" read by the API's read-only route function from
  the ledger beside the served checkpoint, by its digest, opening no
  checkpoint; no server path;
- **the wiring**: the page refreshes when it is shown, reads again only a new
  listing, redraws on its controls and its selection; escaping.

Module: tests/unit/test_webui_models_page.py
"""
from __future__ import annotations

import json
import os
import shutil
import time

import pytest

gr = pytest.importorskip("gradio")

from src.utils.fingerprint import file_sha256  # noqa: E402
from src.webui.components import model_list as ml  # noqa: E402
from src.webui.components import models_page as mp  # noqa: E402
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


def _read(world, monkeypatch):
    listing = ml.list_models("demo", operator_request(monkeypatch))
    return listing, mp.read_page(listing)


def _ordered(world, monkeypatch, metric="untruncated_mrr", arch=None, scoring=None):
    listing, data = _read(world, monkeypatch)
    ctl = mp.controls(listing, data, mp.BY_SCORE, MYGENE, metric, scoring, arch)
    view, ordering = mp.shape(listing, data, ctl)
    return listing, data, ctl, view, ordering


# --------------------------------------------------------------------------- ordering per role
def test_ordering_is_offered_only_for_a_listed_role_and_never_for_acceptance(world, monkeypatch):
    listing, data = _read(world, monkeypatch)
    offers = {v["label"]: v["offers"] for v in data["versions"]}
    assert offers == {MYGENE: True, INST: False, EXPLORE: False}   # listed, acceptance still off
    ctl = mp.controls(listing, data, mp.BY_SCORE, None, None, None, None)
    assert (ctl["order"], ctl["version"]) == (mp.BY_SCORE, MYGENE)
    assert mp.version_choices(data) == [(f"{MYGENE} (role {RESEARCH_ROLE}) — {mp.ROLE_ON}", MYGENE)]
    html = mp.options_html(data, ctl)
    assert f"<b>{INST}</b> (role <code>{ACCEPTANCE_ROLE}</code>) — {e(mp.ROLE_NEVER)}" in html
    assert f"<b>{EXPLORE}</b> (role <code>exploration</code>) — {e(mp.ROLE_NOT_LISTED)}" in html
    assert e(mp.ROLE_NEVER_LISTED) in html                   # listed, and ignored
    assert mp.INDEPENDENCE_NOTE in mp.independence_html(data)  # wherever ordering is offered
    # A version the role does not offer cannot be asked for: it falls back.
    assert mp.controls(listing, data, mp.BY_SCORE, INST, None, None, None)["version"] == MYGENE


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
    assert "0.2310" in first and "#3 · D4 rule" in first and "CUDA · deterministic" in first
    assert "regime differs" not in first
    cpu = cells[_key(world, "model-20.pt")]
    assert "CPU · deterministic" in cpu and "regime differs" in cpu
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


def test_the_architecture_filter_orders_only_what_is_shown(world, monkeypatch):
    _listing, _data, ctl, view, _ordering = _ordered(world, monkeypatch, arch="hgt")
    assert ctl["arch"] == "hgt" and view["architecture"] == "hgt"
    assert [g["title"] for g in view["groups"]] == [mp.GROUP_CONTESTED]


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
    assert [g["title"] for g in view["groups"]][:1] == [mp.GROUP_CONTESTED]
    assert "contested status cannot be checked — not ordered" in view["cells"][
        _key(world, "model-05.pt")]
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
    assert [g["title"] for g in view["groups"]][:1] == [mp.GROUP_CONTESTED]
    assert "contested status cannot be checked — not ordered" in view["cells"][
        _key(world, "model-05.pt")]


def test_a_record_whose_version_the_registry_does_not_list_cannot_be_checked(world, monkeypatch):
    from src.evaluation.sidecar import KEY_FIELDS

    resolver = mp._Versions([], None)
    key = mp._key_text(tuple(world["det"][field] for field in KEY_FIELDS))
    state = mp.contest_states([world["det"]], resolver)[key]
    assert state["state"] == "cannot_be_checked"
    assert "not one the server's registry lists" in state["problems"][0]
    # For a read-only request the reason's own words (which may name a path) stay out.
    assert mp.contest_states([world["det"]], resolver, operator=False)[key]["problems"] == []


# --------------------------------------------------------------------------- the shared Load
def test_a_load_from_the_ordered_list_writes_the_usage_history_entry(world, monkeypatch):
    from src.api.routes.pipeline import PipelineReloadResponse, PipelineStatusResponse
    from src.evaluation.report_store import read_usage_history
    from tests.fixtures.report_store import version_of

    listing, _data, _ctl, _view, ordering = _ordered(world, monkeypatch)
    key = _key(world, "model-05.pt")
    plan, _html = ml.request_load(listing, key, ordering, operator_request(monkeypatch))
    sha = world["shas"]["model-05.pt"]

    def reload(request):
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
    listing, data = _read(world, monkeypatch)
    ctl = mp.controls(listing, data, mp.NEWEST, MYGENE, None, None, None)
    view, ordering = mp.shape(listing, data, ctl)
    assert ordering is None and view == {"architecture": None, "note": mp.LOAD_NOTE}


# --------------------------------------------------------------------------- the detail
def test_the_detail_keeps_validation_and_mode_c_apart(world, monkeypatch):
    listing, data, ctl, _view, _ordering = _ordered(world, monkeypatch)
    detail = mp.detail_parts(listing, data, _key(world, "model-05.pt"), ctl)
    assert "not in use" in detail["header"]
    assert "in use for diagnosis: model" in detail["header"]
    assert world["shas"]["model-12.pt"][:12] in detail["header"]
    top, records = detail["top"], detail["records"]
    assert world["shas"]["model-05.pt"] in top                  # the full fingerprint, Copy
    assert "Validation MRR (Trainer, top-20) 0.2410" in top and "Mode C" not in top
    assert "Validation MRR" not in records and "top-20" not in records and "0.2410" not in records
    assert mp.MODE_C_RESULT_LABEL in records and "Mode C MRR (untruncated)" in records
    # D4's pick first, the other record of its group with its difference
    assert records.index("registered #3") < records.index("registered #2")
    assert "D4 rule picks this record" in records
    assert "Δ Mode C MRR (untruncated) -0.0003 vs #3 · deterministic off" in records
    # the acceptance record is contested, with what the runs say
    assert "contested #1" in records and "runs disagree — investigate" in records
    assert "0.2051" in records and "0.2077" in records
    assert detail["seg"][0] == ("Recorded test results (Mode C) · 3 · 1 contested", mp.RECORDS)


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
    assert later > first and f"#{first} · D4 rule" in view["cells"][key]
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
    assert "runs disagree — investigate" in records and mp.NO_LOAD_HERE in records
    assert str(world["root"]) not in records and str(world["registry"]) not in records
    header = mp.left_header_html(listing, data, "deployment_mode")
    assert "Model list unavailable: read-only, deployment mode not set to M1." in header
    assert e(mp.D8_CONDITIONS) in header and "shep-mp-ro" in header
    assert "read-only, request not local." in mp.left_header_html(listing, data, "not_loopback")
    view, ordering = mp.shape(listing, data, mp.controls(listing, data, mp.BY_SCORE, MYGENE,
                                                         None, None, None))
    assert ordering is None and view == {"note": ""}


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
            "shep-mp-arch"} <= set(inputs)
    out = opened.fn("demo", False, None, mp.BY_SCORE, MYGENE, "untruncated_mrr", None,
                    mp.ALL_ARCHITECTURES, mp.RECORDS, operator_request(monkeypatch))
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


def test_a_listing_already_read_is_not_read_again(world, monkeypatch):
    demo, _ctx = _page()
    recompute = _fn(demo, "_recompute")
    listing, data = _read(world, monkeypatch)
    monkeypatch.setattr(mp, "read_page", lambda listing: pytest.fail("read again"))
    out = recompute.fn(listing, data, mp.NEWEST, None, None, None, None, None, mp.RECORDS,
                       operator_request(monkeypatch))
    assert all(v == gr.update() for v in out)
    changed = {**listing, "served_sha": world["shas"]["model-05.pt"]}
    calls = []
    monkeypatch.setattr(mp, "read_page", lambda listing: calls.append(1) or data)
    recompute.fn(changed, data, mp.NEWEST, None, None, None, None, None, mp.RECORDS,
                 operator_request(monkeypatch))
    assert calls == [1]


def test_controls_selection_and_segments_are_wired(world, monkeypatch):
    demo, _ctx = _page()
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    reorder = [fn for fn in demo.fns.values() if fn.name == "_reorder"]
    triggers = {getattr(demo.blocks[t[0]], "elem_id", None) for fn in reorder for t in fn.targets}
    assert triggers == {"shep-mp-order", "shep-mp-version", "shep-mp-metric", "shep-mp-scoring",
                        "shep-mp-arch"}
    assert all(t[1] == "input" for fn in reorder for t in fn.targets)
    detail = _fn(demo, "_detail")
    assert detail.outputs == [by_id["shep-mp-detail-hd"], by_id["shep-mp-detail-top"],
                              by_id["shep-mp-seg"], by_id["shep-mp-records"],
                              by_id["shep-mp-usage"]]
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
    listing, data, ctl, _view, _ordering = _ordered(world, monkeypatch)
    detail = mp.detail_parts(listing, data, _key(world, "model-05.pt"), ctl)
    rendered = "".join(detail[k] for k in ("top", "records", "usage")) + mp.options_html(data, ctl)
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
