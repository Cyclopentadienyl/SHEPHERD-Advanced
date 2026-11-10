"""
Research › Models › Results: the results service's rows listed, compared and exported.
======================================================================================
Research comparison §5 Stage 8b (owner's O6, O8; `src/webui/components/results_segment.py`).
The world of `tests/unit/test_results_export.py` (kept reports, ledgers, a
record merged before each run had its own record, a mode this build does not
describe, a KG edge set's record, a damaged report) is read as the server's
own source and shown by scope and filters:

- one table per protocol, newest first (the service's order), each row with
  its state (kept, registered, ledger-only "report not available") and its
  exposure where recorded; every unreadable item listed with its reason;
  every result found in no listed workspace listed in this workspace and
  counted in either scope, never hidden;
- every value escaped, in the tables, the status line and Compare;
- Compare: at most five side by side, differences only by default, values as
  recorded with no computed difference, each metric row one protocol's, each
  group once, each missing value in its own words;
- Export: every chosen or shown row through the service's one serializer,
  uncapped, its bytes the serializer's;
- D8: every handler asks the one check first; a refused request reads and
  lists nothing and says why.

Module: tests/unit/test_models_results_segment.py
"""
from __future__ import annotations

import copy
import html as html_lib
import re
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from tests.unit.test_results_export import (  # noqa: E402,F401  (the fixture)
    BASE_RUN,
    LABEL,
    OTHER_LABEL,
    world,
)

HOSTILE = "<img src=x onerror=alert(1)>"


def _text(markup: str) -> str:
    return re.sub(r"\s+", " ", html_lib.unescape(re.sub(r"<[^>]+>", " ", markup))).strip()


def _environ(world, root="root_a"):
    return {"SHEPHERD_WORKSPACES_ROOT": str(world[root]),
            "SHEPHERD_REGISTRY": str(world["registry"])}


def _state(world, monkeypatch, root="root_a"):
    from src.webui.components import results_segment as rs

    for name, value in _environ(world, root).items():
        monkeypatch.setenv(name, value)
    return rs.read_results()


def _listing(workspace="lab", family=BASE_RUN):
    """The shared list's listing as the segment reads it: the workspace and the
    selected item (its family)."""
    item = {"key": "gatv2/x/last.pt", "workspace": workspace, "family_id": family,
            "sha256": "0" * 64, "filename": "last.pt", "run": "x", "architecture": "gatv2"}
    return {"workspace": workspace, "items": [item]}, "gatv2/x/last.pt"


def _shown(state, scope="workspace", **filters):
    from src.webui.components import results_segment as rs

    listing, selected = _listing()
    named = rs.scope_of(listing, selected, scope)
    shown = rs.shown_results(state["rows"], named, filters)
    return {**state, "shown": shown["shown"]}, named, shown


def _operator(monkeypatch):
    from src.webui.components import results_segment as rs

    class Allowed:
        allowed, condition, refusal = True, "enabled", None

    monkeypatch.setattr(rs, "operator_access", lambda request: Allowed())


def _refused(monkeypatch):
    from src.webui.components import results_segment as rs

    class Refused:
        allowed, condition = False, "not_loopback"
        refusal = "Refused: this request did not arrive on loopback. Nothing was done."

    monkeypatch.setattr(rs, "operator_access", lambda request: Refused())


# ---------------------------------------------------------------------------
# The tables
# ---------------------------------------------------------------------------
def test_the_tables_are_one_per_protocol_newest_first_with_each_rows_state(world, monkeypatch):
    """This workspace's results: one table per protocol, the rows in the
    service's own order (newest first by run time; nothing ranked), each with
    its state: kept and registered, kept only, ledger-only "report not
    available"; the unreadable items after the tables with their reasons."""
    from src.api.services import results as service
    from src.webui.components import results_segment as rs

    state, scope, shown = _shown(_state(world, monkeypatch))
    markup = rs.tables_html(state, scope, shown)
    text = _text(markup)
    rows = [state["rows"][i] for i in state["shown"]]
    results = [row for row in rows if row["result"] != service.UNREADABLE]
    # In the service's order, grouped by protocol in the order first shown.
    protocols = list(dict.fromkeys(row["protocol"] for row in results))
    tables = re.findall(r'aria-label="Results, ([^"]*)"', markup)
    assert len(tables) == len(protocols) == len(set(protocols))
    for protocol in protocols:
        labels = [rs.row_label(row) for row in results if row["protocol"] == protocol]
        table = markup.split(f'aria-label="Results, {html_lib.escape(rs.mode_label(protocol))}"'
                             )[1].split("</table>")[0]
        assert re.findall(r'<td class="shep-nowrap">([^<]*)</td>', table) == [
            html_lib.escape(label) for label in labels]
    runs = [row["run_id"] for row in results if row["result"] == service.RUN]
    assert runs == sorted(runs, reverse=True)
    by_run = {row["run_id"]: row for row in results}
    registered = by_run[world["registered"].parent.name]
    second = by_run[world["second"].parent.name]
    elsewhere = by_run[world["elsewhere"].parent.name]
    assert (registered["report"], registered["registered"]) == ("kept", "registered")
    assert (second["report"], elsewhere["report"]) == ("kept", "report not available")
    assert "Report not available (ledger record only)" in text
    assert "Not registered" in text and "Registered" in text
    # A mode this build does not describe: its own table, every metric as recorded.
    assert "Mode Z9 (a measurement mode this version does not describe)" in text
    assert "odd_count 3" in text
    # The unreadable items: listed, with their reasons, never counted.
    unreadable = [row for row in rows if row["result"] == service.UNREADABLE]
    assert unreadable and f"{len(unreadable)} item" in text
    assert all(html_lib.escape(row["reason"]) in markup for row in unreadable)
    assert str(world["tmp"]) not in markup
    # Nothing ordered by a metric: the disease table's rows are in run order
    # whatever their pooled MRR.
    assert "Results of workspace lab:" in _text(rs.status_html(state, scope, shown))


def test_the_family_scope_and_the_filters_narrow_and_never_hide_an_unreadable_item(
        world, monkeypatch):
    from src.api.services import results as service
    from src.webui.components import results_segment as rs

    state = _state(world, monkeypatch)
    _all, _scope, everything = _shown(state)
    family, scope, shown = _shown(state, scope="family")
    assert scope["scope"] == rs.FAMILY and scope["family"] == BASE_RUN
    rows = [state["rows"][i] for i in shown["shown"]]
    assert all(row["model"]["family_id"] == BASE_RUN for row in rows
               if row["result"] != service.UNREADABLE)
    # A protocol filter: only its rows, and every unreadable item still.
    narrowed, _scope, only = _shown(state, protocol="disease")
    kept = [state["rows"][i] for i in only["shown"]]
    assert {row["protocol"] for row in kept if row["result"] != service.UNREADABLE} == {
        "disease"}
    assert sum(row["result"] == service.UNREADABLE for row in kept) == sum(
        state["rows"][i]["result"] == service.UNREADABLE for i in everything["shown"])
    # The filters offer what the scope's rows hold; a value no longer offered reads All.
    assert ("Disease", "disease") in everything["choices"]["protocol"]
    # A data kind's option in the caveats' one wording, as a label.
    assert ("Supplied cohort version", "supplied") in everything["choices"]["data_kind"]
    assert ("KG edge set", "kg_edges") not in everything["choices"]["data_kind"]
    _s, _n, gone = _shown(state, protocol="kg_link_prediction")
    assert gone["chosen"]["protocol"] == rs.ALL
    # A family scope without a family: this workspace, saying why.
    listing, selected = _listing(family=None)
    fallback = rs.scope_of(listing, selected, rs.FAMILY)
    assert fallback["scope"] == rs.WORKSPACE and "names no family" in fallback["note"]


def test_an_empty_scope_says_what_appears_and_names_the_cross_workspace_gap(world, monkeypatch):
    """Nothing in scope: what will be listed, and that comparing across
    workspaces is not built here yet (a gap, not a rule), the command line's
    export joining them; filtered to nothing: how to see the scope's rows."""
    from src.webui.components import results_segment as rs

    state = _state(world, monkeypatch)
    listing, selected = _listing(workspace="nowhere")
    scope = rs.scope_of(listing, selected, rs.WORKSPACE)
    shown = rs.shown_results(state["rows"], scope, {})
    text = _text(rs.tables_html({**state, "shown": shown["shown"]}, scope, shown))
    assert "No result of workspace nowhere yet" in text
    assert rs.CROSS_WORKSPACE_GAP in text and "not built in this page yet" in text
    # Only results found in no listed workspace are listed (the status line
    # says what they are): the workspace itself holds none yet.
    assert shown["in_scope"] == shown["unplaced_in_scope"] == 2
    assert "scripts/export_results.py" in rs.SEGMENT_HELP["models-results"][1]
    filtered, named, narrowed = _shown(state, role="no-such-role")
    assert narrowed["chosen"]["role"] == rs.ALL  # not offered: All


def test_a_result_found_in_no_listed_workspace_is_listed_and_counted_never_hidden(
        world, monkeypatch):
    """Stage 8b review (O8; invariants 2 and 8): a kept report no listed ledger
    registered, whose model's weights are not listed now (moved, or a
    ``last.pt`` saved over), has no workspace to be scoped by. This workspace
    lists it, its model Unknown with why, and the status line counts it; the
    family scope, which cannot hold it (no family is known for it), says where
    it is listed. The world's two such reports are of the checkpoint the
    registry's store measured and no listed workspace holds."""
    from src.webui.components import results_segment as rs

    state = _state(world, monkeypatch)
    homeless = [index for index, row in enumerate(state["rows"]) if rs.unplaced(row)]
    assert len(homeless) == 2
    assert all(state["rows"][i]["report"] == "kept" and not state["rows"][i]["ledgers"]
               for i in homeless)
    # This workspace: listed, Unknown with why, and counted apart.
    named, scope, shown = _shown(state)
    assert set(homeless) <= set(shown["shown"])
    assert (shown["unplaced"], shown["unplaced_in_scope"]) == (2, 2)
    tables = _text(rs.tables_html(named, scope, shown))
    assert "Unknown: its checkpoint is not in a listed workspace" in tables
    status = _text(rs.status_html(named, scope, shown))
    assert f"{shown['in_scope']} shown of {shown['in_scope']}" in status
    assert "Listed here too: 2 results found in no listed workspace" in status
    # The family scope: not listed (no family is known), and said where they are.
    named, scope, shown = _shown(state, scope="family")
    assert not set(homeless) & set(shown["shown"])
    assert (shown["unplaced"], shown["unplaced_in_scope"]) == (2, 0)
    assert ("2 results found in no listed workspace (the weights they measured are not "
            "listed now, and no listed ledger holds them) have no known family: This "
            "workspace lists them.") in _text(rs.status_html(named, scope, shown))
    # Compare and Export reach them like any shown result.
    named, _scope, _shown_rows = _shown(state)
    keys = [value for _label, value in rs.compare_choices(named)]
    assert {f"{named['read_id']}:{i}" for i in homeless} <= set(keys)


def test_the_family_is_said_once_in_the_family_scope(world, monkeypatch):
    """Skill rule 1.2: the family scope's status line names the family every
    row shares, so no row repeats it; the workspace scope, whose rows' families
    differ, names each row's."""
    from src.webui.components import results_segment as rs

    state = _state(world, monkeypatch)
    for scope_value, rows_say in ((rs.FAMILY, False), (rs.WORKSPACE, True)):
        named, scope, shown = _shown(state, scope=scope_value)
        text = _text(rs.tables_html(named, scope, shown))
        assert ("family from" in text) is rows_say, scope_value


def test_a_figure_the_group_differs_on_is_shown_unrounded(world, monkeypatch):
    """Stage 8b review: beside "runs under these conditions differ", the
    figures they differ on are shown as recorded, unrounded (Models' contest,
    `contested_value_text`), so two runs that differ past the fourth decimal
    never read as equal; a figure they agree on keeps four decimals."""
    from src.webui.components import results_segment as rs

    state, _scope, _shown_rows = _shown(_state(world, monkeypatch), protocol="disease")
    row = copy.deepcopy(next(row for row in _results(state) if row["protocol"] == "disease"))
    twin = copy.deepcopy(row)
    for one, value in ((row, 0.30000001), (twin, 0.3)):
        one["group"] = {"state": "contested", "differing": ["pooled_mrr"]}
        one["measurement"]["deterministic_algorithms"] = True
        for metric in one["metrics"]:
            if metric["name"] == "pooled_mrr":
                metric["value"] = value
            if metric["name"] == "pooled_top10":
                metric["value"] = 0.75
    markup = rs.protocol_table_html("disease", [row, twin])
    cells = re.findall(r'<td class="shep-num">([^<]*)</td>', markup)
    assert cells == ["0.30000001", "0.7500", "0.3", "0.7500"]
    assert "differ: pooled_mrr" in _text(markup)


def test_every_value_in_the_tables_is_escaped(world, monkeypatch):
    from src.webui.components import results_segment as rs

    state, scope, shown = _shown(_state(world, monkeypatch))
    rows = copy.deepcopy(state["rows"])
    for row in rows:
        if row["result"] == "unreadable":
            row["reason"] = HOSTILE
            continue
        row["data"]["label"] = HOSTILE
        row["environment"]["class"] = HOSTILE
        for metric in row["metrics"]:
            metric["name"] = HOSTILE
    markup = rs.tables_html({**state, "rows": rows}, scope, shown)
    assert "<img" not in markup and html_lib.escape(HOSTILE) in markup


def test_every_value_in_compare_and_the_status_line_is_escaped(world, monkeypatch):
    """Task item 5 (Stage 8b review): Compare renders spec keys and values,
    metric names, data labels, run ids, reasons and the protocol from rows a
    hand-edited report controls, and the status line the workspace, the family
    and the note; each is escaped, never markup."""
    from src.webui.components import results_segment as rs

    state, _scope, _shown_rows = _shown(_state(world, monkeypatch))
    rows = copy.deepcopy(_results(state))[:2]
    for row in rows:
        row["model"]["spec"] = {HOSTILE: HOSTILE}
        row["model"]["learning_settings"] = {HOSTILE: HOSTILE}
        row["data"]["label"] = HOSTILE
        row["run_id"] = HOSTILE
        row["protocol"] = HOSTILE
        row["unknown"] = {"model.operation": HOSTILE, "measurement.regime.device": HOSTILE}
        row["model"]["operation"] = None
        for metric in row["metrics"]:
            metric["name"] = HOSTILE
            metric["value"] = HOSTILE
    for differences in (True, False):
        markup = rs.compare_html(rows, differences)
        assert "<img" not in markup and html_lib.escape(HOSTILE) in markup
    scope = {"scope": rs.FAMILY, "workspace": HOSTILE, "family": HOSTILE, "note": HOSTILE}
    status = rs.status_html({**state, "read_at": HOSTILE}, scope,
                            {"shown": [], "in_scope": 0, "unplaced": 1, "unplaced_in_scope": 0})
    assert "<img" not in status and html_lib.escape(HOSTILE) in status


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------
def _results(state):
    return [state["rows"][i] for i in state["shown"] if state["rows"][i]["result"] != "unreadable"]


def test_compare_shows_values_side_by_side_differences_only_and_no_computed_difference(
        world, monkeypatch):
    from src.webui.components import results_segment as rs

    state, _scope, _shown_rows = _shown(_state(world, monkeypatch), protocol="disease")
    disease = _results(state)
    first, second = (row for row in disease if row["report"] == "kept"
                     and row["run_id"] in (world["registered"].parent.name,
                                           world["second"].parent.name))
    markup = rs.compare_html([first, second], True)
    text = _text(markup)
    # The two runs of one condition group differ in pooled MRR: both values as
    # recorded, side by side, and no difference between them anywhere.
    mrr = re.search(r"pooled_mrr \(Disease\)</th>(.*?)</tr>", markup).group(1)
    values = re.findall(r'<td class="shep-num">([^<]*)</td>', mrr)
    assert sorted(values) == sorted([rs.recorded_value_text(0.1 + 0.2), "0.25"])
    assert "0.05" not in text and "Δ" not in text and "difference:" not in text.lower()
    # Differences only: rows recorded alike are hidden (the model, the data).
    assert "Differences only:" in text and "rows recorded alike in every column are hidden" in text
    assert "Learning stage" not in text and "Data label" not in text
    # All rows: they are shown.
    every = _text(rs.compare_html([first, second], False))
    assert "Learning stage" in every and "Data label" in every


def test_compare_never_puts_two_protocols_metrics_on_one_row_and_caps_at_five(
        world, monkeypatch):
    from src.webui.components import results_segment as rs

    state, _scope, _shown_rows = _shown(_state(world, monkeypatch))
    results = _results(state)
    disease = next(row for row in results if row["protocol"] == "disease")
    odd = next(row for row in results if row["protocol"] == "Z9")
    markup = rs.compare_html([disease, odd], True)
    hits = re.search(r"<th scope=\"row\">hits@1 \(Mode Z9\)</th>(.*?)</tr>", markup).group(1)
    assert 'another protocol' in hits and "0.25" in hits
    mrr = re.search(r"<th scope=\"row\">pooled_mrr \(Disease\)</th>(.*?)</tr>", markup).group(1)
    assert "another protocol" in mrr
    # Each group once, in its place: the second protocol's metrics are not
    # split from the first's by References (Stage 8b review).
    for every in (rs.compare_html([disease, odd], False), markup):
        heads = re.findall(r'<th scope="colgroup"[^>]*>([^<]*)</th>', every)
        assert len(heads) == len(set(heads))
        assert heads == [group for group in rs.COMPARE_GROUPS if group in heads]
    # The protocol is said once, in each column's head, in words: no
    # "Protocol" row and no measurement "Mode" row (rule 1.2).
    every = rs.compare_html([disease, odd], False)
    assert '<th scope="row">Protocol</th>' not in every
    assert '<th scope="row">Mode</th>' not in every
    assert "Disease" in re.search(r"<thead>(.*?)</thead>", every).group(1)
    # Six chosen: five columns, saying so; a query and an export are never capped.
    six = (results * 6)[:6]
    capped = rs.compare_html(six, False, asked=6)
    assert len(re.findall(r'<th scope="col">', capped)) == 1 + rs.COMPARE_CAP
    assert "6 were chosen; Compare shows 5 at a time" in _text(capped)
    # A field without a source: Unknown with why, never a number.
    ledger_only = next(row for row in results if row["report"] == "report not available")
    assert "Unknown" in _text(rs.compare_html([ledger_only], False))


def test_compare_says_each_missing_value_in_its_own_words(world, monkeypatch):
    """Stage 8b review (skill §6): a result of the same protocol that does not
    record a metric reads "not recorded" there, never "another protocol"; a
    field a result's records do not hold at all (a spec key another spec has)
    reads "not recorded", never "none", which is a value recorded as null; a
    setting its record does not hold as a value reads Unknown with why."""
    from src.webui.components import results_segment as rs

    state, _scope, _shown_rows = _shown(_state(world, monkeypatch), protocol="disease")
    listed = [row for row in _results(state) if "model.spec" not in row["unknown"]]
    first, second = (copy.deepcopy(row) for row in listed[:2])
    assert first["protocol"] == second["protocol"] == "disease"
    first["metrics"].append({"name": "only_here", "value": 0.8, "status": "recorded",
                             "unit": None, "denominator": None})
    first["model"]["spec"] = {"chunk_size": 64, "hidden": 128, "dropout": None}
    second["model"]["spec"] = {"hidden": 128, "dropout": 0.1}
    second["measurement"].pop("device", None)
    second["unknown"] = {**second["unknown"],
                         "measurement.regime.device": "its record does not hold it as a value"}
    first["measurement"]["device"] = "cuda:0"
    markup = rs.compare_html([first, second], True)

    def cells(label):
        found = re.search(rf'<th scope="row">{re.escape(label)}</th>(.*?)</tr>', markup)
        return [_text(cell) for cell in re.findall(r"<td[^>]*>.*?</td>", found.group(1))]

    assert cells("only_here (Disease)") == ["0.8", "○ not recorded"]
    assert cells("Spec · chunk_size") == ["64", "○ not recorded"]
    assert cells("Spec · dropout") == ["none", "0.1"]
    assert cells("Device") == ["cuda:0", "○ Unknown: its record does not hold it as a value"]
    assert "another protocol" not in markup


def test_compare_choices_are_unique_and_name_this_read_only(world, monkeypatch):
    from src.webui.components import results_segment as rs

    state, _scope, _shown_rows = _shown(_state(world, monkeypatch))
    choices = rs.compare_choices(state)
    labels = [label for label, _value in choices]
    assert len(set(labels)) == len(labels) and all(len(label) <= 40 for label in labels)
    keys = [value for _label, value in choices]
    assert [row["run_id"] for row in rs.chosen_rows(state, keys[:2])] == [
        state["rows"][int(key.rpartition(":")[2])]["run_id"] for key in keys[:2]]
    # A key of an earlier read names nothing now.
    stale = f"0:{keys[0].rpartition(':')[2]}"
    assert rs.chosen_rows(state, [stale]) == []


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def test_export_writes_the_shown_or_chosen_rows_through_the_one_serializer(world, monkeypatch,
                                                                           tmp_path):
    from src.api.services import results as service
    from src.webui import exports
    from src.webui.components import results_segment as rs

    monkeypatch.setattr(exports, "_EXPORT_DIR", tmp_path / "downloads")
    (tmp_path / "downloads").mkdir()
    _operator(monkeypatch)
    state, _scope, _shown_rows = _shown(_state(world, monkeypatch))
    listing, selected = _listing()
    file_update, note = rs._on_results_export(None, state, listing, selected, rs.WORKSPACE, [],
                                              rs.SHOWN, "csv")
    path = Path(file_update["value"])
    shown = [state["rows"][i] for i in state["shown"]]
    assert path.parent == tmp_path / "downloads" and path.name.startswith("results_workspace-lab")
    # The file's bytes are the serializer's text, untranslated (as the
    # command line writes its file).
    assert path.read_bytes() == service.to_csv(shown).encode("utf-8")
    assert "unreadable item" in _text(note)
    # Uncapped: more rows than Compare's five, every one exported, shown or
    # chosen (the choice itself is not capped: Compare shows the first five).
    many = {**state, "rows": _results(state) * 3}
    many["shown"] = list(range(len(many["rows"])))
    assert len(many["rows"]) > rs.COMPARE_CAP
    file_update, _note = rs._on_results_export(None, many, listing, selected, rs.WORKSPACE, [],
                                               rs.SHOWN, "json")
    assert Path(file_update["value"]).read_bytes() == service.to_json(
        many["rows"]).encode("utf-8")
    keys = [f"{many['read_id']}:{index}" for index in many["shown"]][:rs.COMPARE_CAP + 2]
    file_update, _note = rs._on_results_export(None, many, listing, selected, rs.WORKSPACE,
                                               keys, rs.SELECTED, "json")
    assert Path(file_update["value"]).read_bytes() == service.to_json(
        many["rows"][:rs.COMPARE_CAP + 2]).encode("utf-8")
    # Two exports of one name in one second: two files, neither replaced.
    first = rs.export_file(shown, "csv", "same")
    second = rs.export_file(shown, "csv", "same")
    assert first != second and Path(first).read_bytes() == Path(second).read_bytes()
    # Nothing chosen: nothing written, the next step said.
    file_update, note = rs._on_results_export(None, state, listing, selected, rs.WORKSPACE, [],
                                              rs.SELECTED, "csv")
    assert file_update["value"] is None and "Choose results" in _text(note)


# ---------------------------------------------------------------------------
# D8
# ---------------------------------------------------------------------------
def test_every_handler_asks_d8_first_and_a_refused_request_reads_nothing(world, monkeypatch):
    from src.api.services import results as service
    from src.webui.components import results_segment as rs

    state, _scope, _shown_rows = _shown(_state(world, monkeypatch))
    _refused(monkeypatch)
    monkeypatch.setattr(service, "query", lambda *a, **k: pytest.fail("read while refused"))
    listing, selected = _listing()
    filters = ("all",) * 4
    for out in (rs._on_results_open(None, rs.RESULTS, rs.EMPTY_STATE, listing, selected,
                                    rs.WORKSPACE, *filters, [], True),
                rs._on_results_read(None, listing, selected, rs.WORKSPACE, *filters, True),
                rs._on_results_shown(None, rs.RESULTS, state, listing, selected, rs.WORKSPACE,
                                     *filters, [], True)):
        new_state = out[0]
        assert new_state["rows"] == [] and new_state["refused"].startswith("Refused:")
        assert "Refused:" in out[6] and "<table" not in out[6]
        assert out[7]["choices"] == []
    assert rs._on_results_compare(None, state, ["x"], True).count("Refused:") == 1
    file_update, note = rs._on_results_export(None, state, listing, selected, rs.WORKSPACE, [],
                                              rs.SHOWN, "csv")
    assert file_update["value"] is None and "Refused:" in note


def test_an_operator_reads_once_on_opening_and_again_only_when_asked(world, monkeypatch):
    from src.api.services import results as service
    from src.webui.components import results_segment as rs

    for name, value in _environ(world).items():
        monkeypatch.setenv(name, value)
    _operator(monkeypatch)
    calls = []
    real = service.query
    monkeypatch.setattr(service, "query", lambda sources: calls.append(1) or real(sources))
    listing, selected = _listing()
    filters = ("all",) * 4
    out = rs._on_results_open(None, rs.RESULTS, rs.EMPTY_STATE, listing, selected, rs.FAMILY,
                              *filters, [], True)
    state = out[0]
    assert calls == [1] and state["rows"] and "Results of the family from" in _text(out[5])
    # Opened again, a filter or the scope: shown from the rows read, not read again.
    rs._on_results_open(None, rs.RESULTS, state, listing, selected, rs.FAMILY, *filters, [],
                        True)
    again = rs._on_results_shown(None, rs.RESULTS, state, listing, selected, rs.WORKSPACE,
                                 *filters, [], True)
    assert calls == [1] and "Results of workspace lab:" in _text(again[5])
    # Another segment: nothing written.
    assert all(update == {"__type__": "update"} for update in rs._on_results_shown(
        None, "usage", state, listing, selected, rs.WORKSPACE, *filters, [], True))
    # ↻ reads again and clears Compare's choices (they named the earlier read).
    read = rs._on_results_read(None, listing, selected, rs.WORKSPACE, *filters, True)
    assert calls == [1, 1] and read[0]["read_id"] != state["read_id"]
    assert read[7]["value"] == []


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------
def test_the_segment_is_models_third_and_registers_every_i_it_draws(world, monkeypatch):
    """The Results segment is built in Models' detail (no new page), its ⓘ
    registered by the page; the Test page renders a kept report through the
    one reader, with no kind rendering of its own (`_kind_lines` retired)."""
    import gradio as gr

    from src.webui.components import models_page as mp
    from src.webui.components import results_segment as rs
    from src.webui.components import test_page as tp
    from src.webui.components.help import HelpRegistry, registered_keys_in
    from src.webui.shell import standalone_context

    registry = HelpRegistry()
    with gr.Blocks() as demo:
        mp.create_models_page(standalone_context(demo, registry))
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    assert {"shep-rs", "shep-rs-scope", "shep-rs-chosen", "shep-rs-export"} <= set(by_id)
    # The choice is not capped (Export › Selected takes every one chosen);
    # Compare shows the first five and says so.
    assert by_id["shep-rs-chosen"].max_choices is None
    assert by_id["shep-rs-differences"].value is True
    state, scope, shown = _shown(_state(world, monkeypatch))
    for key in registered_keys_in(rs.status_html(state, scope, shown)):
        assert key in registry, key
    assert not hasattr(tp, "_kind_lines")
