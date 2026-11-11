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
    names = rs.row_names(state["rows"])
    for protocol in protocols:
        labels = [names[i] for i in state["shown"] if state["rows"][i] in results
                  and state["rows"][i]["protocol"] == protocol]
        table = markup.split(f'aria-label="Results, {html_lib.escape(rs.mode_label(protocol))}"'
                             )[1].split("</table>")[0]
        assert [_text(cell) for cell in re.findall(
            r'<td class="shep-rs-name">(.*?)</td>', table)] == labels
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
        row["run_id"] = HOSTILE
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
        # Nested entries, named items and measurement keys are row words and titles.
        row["model"]["effective_settings"] = {HOSTILE: {HOSTILE: HOSTILE}}
        row["references"] = [{"name": HOSTILE, "sha256": HOSTILE}]
        # A component's name and its facts, as a hand-edited record holds them.
        row["environment"]["components"] = {HOSTILE: {"kind": HOSTILE, "source": {
            "state": "recorded", "route": HOSTILE, HOSTILE: HOSTILE}}}
        row["measurement"][HOSTILE] = HOSTILE
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
    # A key or a name is a row's title too: a quote never ends the attribute
    # (review of the fix: HOSTILE holds no quote).
    quote = 'k" onmouseover="x'
    for index, row in enumerate(rows):
        row["model"]["effective_settings"] = {quote: {quote: f"{quote}{index}"}}
        row["measurement"][quote] = f"{quote}{index}"
        row["references"] = [{"name": quote, "sha256": f"{quote}{index}"}]
        row["environment"]["components"] = {quote: {"kind": f"{quote}{index}"}}
        row["unknown"]["model.operation"] = f"{quote}{index}"
    for differences in (True, False):
        markup = rs.compare_html(rows, differences)
        assert 'onmouseover="' not in markup and "&quot; onmouseover=&quot;x" in markup
        assert f'title="model.effective_settings.{html_lib.escape(quote)}.' in markup
        assert f'title="measurement.{html_lib.escape(quote)}"' in markup
        assert f'title="environment.components.{html_lib.escape(quote)}.kind"' in markup
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
    mrr = re.search(r'title="metrics.pooled_mrr">Disease pooled MRR</th>(.*?)</tr>',
                    markup).group(1)
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
    # Each metric in the protocol tables' words, its protocol's
    # (`caveats.metric_text`), its recorded name the row's title.
    hits = re.search(r'<th scope="row" title="metrics.hits@1">Mode Z9 hits@1</th>(.*?)</tr>',
                     markup).group(1)
    assert 'another protocol' in hits and "0.25" in hits
    mrr = re.search(r'<th scope="row" title="metrics.pooled_mrr">Disease pooled MRR</th>'
                    r'(.*?)</tr>', markup).group(1)
    assert "another protocol" in mrr
    # Each group once, in its place: the second protocol's metrics are not
    # split from the first's by References (Stage 8b review).
    for every in (rs.compare_html([disease, odd], False), markup):
        heads = [_text(head) for head in
                 re.findall(r'<th scope="colgroup"[^>]*>(.*?)</th>', every)]
        assert heads
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
    assert _cells(markup, "Disease only_here") == ["0.8", "○ not recorded"]
    assert _cells(markup, "Spec · Chunk size") == ["64", "○ not recorded"]
    assert _cells(markup, "Spec · Dropout") == ["none", "0.1"]
    assert _cells(markup, "Device") == [
        "cuda:0", "○ Unknown: its record does not hold it as a value"]
    assert "another protocol" not in markup


def test_compare_shows_what_was_in_effect_why_and_each_source_fact_as_its_own_row(
        world, monkeypatch):
    """Review fix RC-P2-1: two runs of one build (one component, version and
    native digest) that differ only in whether it was in effect, why it fell
    back and two source facts (a commit and a build flag). With Differences
    only, the environment shows those rows alone, each in the segment's words
    under the component's name as recorded, its record key the row's title; a
    build flag's environment variable and a tool keep their names as
    recorded, the VCS its abbreviation (review of the fix); the Native build
    line is alike for both, so hidden, and shown with every row. A result
    whose record names no components reads Unknown there, with why, once
    under Differences only (review of the fix) and on every row with all
    rows shown."""
    from src.webui.components import results_segment as rs
    from tests.unit.test_results_export import FELL_BACK, attention, registered_with

    requested = {"kind": "vcs", "vcs": "git", "url": "https://example.org/attn", "ref": "v2"}
    used = registered_with(world, {"test_attention": attention(
        True, requested=requested, flags={"MAX_JOBS": "8", "FORCE_CUDA": "1"})})
    fell = registered_with(world, {"test_attention": attention(
        False, commit="b" * 40, requested=requested, flags={"MAX_JOBS": "4", "FORCE_CUDA": "1"})})
    state, _scope, _shown_rows = _shown(_state(world, monkeypatch))
    rows = {row["run_id"]: row for row in _results(state)}
    first, second = rows[used], rows[fell]

    markup = rs.compare_html([first, second], True)
    environment = [label for label in _labels(markup)
                   if label.startswith(("Component", "Native build", "GPU", "Environment"))]
    assert environment == ["Component · test_attention › In effect",
                           "Component · test_attention › Fallback reason",
                           "Component · test_attention › Source › Commit",
                           "Component · test_attention › Source › Flags › MAX_JOBS"]
    assert _cells(markup, "Component · test_attention › In effect") == ["true", "false"]
    assert _cells(markup, "Component · test_attention › Fallback reason") == ["none", FELL_BACK]
    assert _cells(markup, "Component · test_attention › Source › Commit") == ["a" * 40, "b" * 40]
    assert _cells(markup, "Component · test_attention › Source › Flags › MAX_JOBS") == ["8", "4"]
    assert 'title="environment.components.test_attention.in_effect"' in markup
    assert 'title="environment.components.test_attention.source.commit"' in markup
    assert 'title="environment.components.test_attention.source.flags.MAX_JOBS"' in markup

    every = rs.compare_html([first, second], False)
    native = _cells(every, "Native build")
    assert native[0] == native[1] and "test_attention 2.0" in native[0]
    for label, value in (("Kind", "attention kernel"), ("Version", "2.0"),
                         ("Requested", "true"), ("Source › Compiler", "audit-compiler 1.0"),
                         ("Source › Route", "local build"), ("Source › Requested › VCS", "git"),
                         ("Source › Flags › FORCE_CUDA", "1")):
        assert _cells(every, f"Component · test_attention › {label}") == [value, value]
    assert rs.field_words("TORCH_CUDA_ARCH_LIST") == "TORCH_CUDA_ARCH_LIST"

    before = next(row for row in _results(state)
                  if row["run_id"] == world["registered"].parent.name)
    older = rs.compare_html([before, first], True)
    assert [label for label in _labels(older) if label.startswith("Component")] == [
        "Component · test_attention › Kind"]
    assert _cells(older, "Component · test_attention › Kind") == [
        "○ Unknown: its record names no environment components", "attention kernel"]
    assert _cells(rs.compare_html([before, first], False),
                  "Component · test_attention › In effect") == ["○ Unknown", "true"]


def _local_build(name, version, digest, **changes):
    """A PyG extension as a local build's source receipt records it
    (`scripts/validate_pyg_ext.py`: the requested and resolved source, the
    wheel, the build's basis, toolchain and flags)."""
    from src.utils.run_environment import PYG_NATIVE_KIND, component_facts

    facts = component_facts(
        PYG_NATIVE_KIND, installed=True, available=True, version=version,
        native={"state": "recorded", "count": 2, "aggregate_sha256": digest * 64},
        source={"state": "recorded", "route": "local build",
                "requested": {"kind": "registry", "name": name, "specifier": f"=={version}"},
                "resolved": {"version": version}, "wheel_sha256": "b" * 64,
                "compiled_against": {"torch": "2.10.0+cu130", "cuda": "13.0",
                                     "basis": "the build's target environment"},
                "toolchain": {"nvcc": "13.0.48", "gcc": "13.3.0", "g++": "13.3.0"},
                "flags": {"TORCH_CUDA_ARCH_LIST": "12.1", "FORCE_CUDA": "1", "MAX_JOBS": "8"}})
    facts.update(changes)
    return facts


def test_differences_only_says_an_older_records_unknown_once_not_per_component_fact(
        world, monkeypatch):
    """Review of RC-P2-1 (skill §1.2: each fact once): a result whose record
    names no components (every record written before them, the kept corpus
    among them) beside one recording four local builds. Under Differences
    only, the components show one row, the first, Unknown with why; the
    other 27 facts per component read "Unknown" only because of that why,
    so they are not differences and are hidden, the note saying so. With
    every row shown, each is there, "Unknown" with the why its title. A fact
    the newer columns record differently is still a difference, the older
    one's Unknown beside it. The rows of other fields are as before."""
    from src.webui.components import results_segment as rs
    from tests.unit.test_results_export import registered_with

    builds = (("pyg_lib", "0.8.0", "a"), ("torch_scatter", "2.1.2", "c"),
              ("torch_sparse", "0.6.18", "d"), ("torch_cluster", "1.6.3", "e"))
    four = registered_with(world, {name: _local_build(name, version, digest)
                                   for name, version, digest in builds})
    other = registered_with(world, {name: _local_build(name, version, digest, **(
        {"in_effect": False, "fallback_reason": "no kernel for this device"}
        if name == "torch_sparse" else {})) for name, version, digest in builds})
    state, _scope, _shown_rows = _shown(_state(world, monkeypatch))
    rows = {row["run_id"]: row for row in _results(state)}
    before = rows[world["registered"].parent.name]
    assert before["environment"]["components"] is None
    assert "environment.components" in before["unknown"]

    markup = rs.compare_html([before, rows[four]], True)
    components = [label for label in _labels(markup) if label.startswith("Component")]
    assert components == ["Component · pyg_lib › Kind"]
    assert _cells(markup, components[0]) == [
        "○ Unknown: its record names no environment components", "PyG native extension"]
    every = rs.compare_html([before, rows[four]], False)
    entries = [label for label in _labels(every) if label.startswith("Component")]
    assert len(entries) > 4 * 20
    assert _cells(every, "Component · torch_cluster › Source › Flags › MAX_JOBS") == [
        "○ Unknown", "8"]
    # Only the components' later rows are hidden for this reason, and said apart.
    hidden = len(entries) - 1
    assert (f"and {hidden} that differ only by an Unknown explained above."
            in _text(markup))
    without = copy.deepcopy(rows[four])
    without["environment"]["components"] = None
    without["unknown"]["environment.components"] = before["unknown"]["environment.components"]
    assert ([label for label in _labels(markup) if not label.startswith("Component")]
            == _labels(rs.compare_html([before, without], True)))

    # A real difference among the columns that record the facts is still shown.
    three = rs.compare_html([before, rows[four], rows[other]], True)
    assert [label for label in _labels(three) if label.startswith("Component")] == [
        "Component · pyg_lib › Kind", "Component · torch_sparse › In effect",
        "Component · torch_sparse › Fallback reason"]
    assert _cells(three, "Component · torch_sparse › In effect") == [
        "○ Unknown", "unknown", "false"]


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
# The research comparison browser check (WebUI skill §12): Models › Results
# and Compare as the screens read
# ---------------------------------------------------------------------------
def _named(markup):
    """Compare's rows as ``(whole name, cells' markup)``: a field's own row is
    named by its words; a nested entry's row shows its own words under the
    sub-heading that says its path, its whole name ("Weight transfer · Parts
    › Encoder") its accessible name."""
    for attrs, words, cells in re.findall(r'<th scope="row"([^>]*)>(.*?)</th>(.*?)</tr>',
                                          markup):
        name = re.search(r'aria-label="([^"]*)"', attrs)
        yield html_lib.unescape(name.group(1) if name else words), cells


def _cells(markup, label):
    """A Compare row's cells as text, by the row's whole name (None: no such row)."""
    found = next((cells for name, cells in _named(markup) if name == label), None)
    return None if found is None else [
        _text(cell) for cell in re.findall(r"<td[^>]*>.*?</td>", found)]


def _labels(markup):
    return [name for name, _cells in _named(markup)]


def _twins(world, monkeypatch):
    """A result, two more runs of its model in the same minute whose reports
    are byte-equal to its own (research comparison §7: both kept), one of them
    in the same second, and a result of another minute, as one read's state."""
    state, _scope, _shown_rows = _shown(_state(world, monkeypatch))
    first = copy.deepcopy(next(row for row in _results(state) if row["protocol"] == "disease"))
    model12 = first["model"]["checkpoint_sha256"][:12]
    first["run_id"] = f"20261110T110547Z-{model12}-9bfd7c23"
    second, third = copy.deepcopy(first), copy.deepcopy(first)
    second["run_id"] = f"20261110T110543Z-{model12}-34ea9f49"
    third["run_id"] = f"20261110T110547Z-{model12}-0a1b2c3d"
    other = copy.deepcopy(first)
    other["run_id"] = f"20261110T110716Z-{model12}-45ad9e27"
    other["report_sha256"] = "5ad3b2bac629" + "0" * 52
    rows = [other, first, second, third]
    return {"read_id": "7", "read_at": "now", "rows": rows, "refused": None,
            "shown": list(range(len(rows)))}


def test_two_runs_with_byte_equal_reports_are_told_apart_alike_everywhere(world, monkeypatch):
    """Browser check models-2 (skill §5, §8; plan §7: different runs are both
    kept, whether their bytes are equal): the runs' names differ, by the time
    to the second and then the run id's own last part, never by a counter;
    the table row, the Compare option and the Compare head say the same name;
    a result no other reads alike keeps its time to the minute."""
    from src.webui.components import results_segment as rs

    state = _twins(world, monkeypatch)
    names = rs.row_names(state["rows"])
    report = state["rows"][1]["report_sha256"][:12]
    assert names == ["2026-11-10 11:07 UTC · 5ad3b2bac629",
                     "2026-11-10 11:05:47 UTC · run …9bfd7c23",
                     f"2026-11-10 11:05:43 UTC · {report}",
                     "2026-11-10 11:05:47 UTC · run …0a1b2c3d"]
    assert len(set(names)) == len(names)
    assert all(len(name) <= 40 and name.count(" · ") == 1 and not re.search(r"\(\d+\)$", name)
               for name in names)
    scope = {"scope": rs.WORKSPACE, "workspace": "lab", "family": None, "note": None}
    shown = {"shown": state["shown"], "in_scope": 4, "unplaced": 0, "unplaced_in_scope": 0}
    table = rs.tables_html(state, scope, shown)
    assert [_text(cell) for cell in re.findall(r'<td class="shep-rs-name">(.*?)</td>',
                                               table)] == names
    choices = rs.compare_choices(state)
    assert [label for label, _value in choices] == names
    compare = rs.compare_state_html(state, [value for _label, value in choices], False, asked=4)
    heads = [_text(head) for head in re.findall(r'<th scope="col">(.*?)</th>', compare)][1:]
    assert heads == [f"{name} Disease" for name in names]
    # Compare's handler heads its columns by the read's names too.
    _operator(monkeypatch)
    handled = rs._on_results_compare(None, state, [value for _label, value in choices[1:3]],
                                     False)
    assert [_text(head) for head in re.findall(r'<th scope="col">(.*?)</th>', handled)][1:] == [
        f"{name} Disease" for name in names[1:3]]
    # A counter only where nothing recorded tells two apart (one run listed twice).
    assert rs.row_names([state["rows"][1], state["rows"][1]])[1].endswith(" (2)")


def test_a_result_name_wraps_only_at_its_separator_and_table_words_stay_whole():
    """Browser check models-1: a protocol table's columns are sized with every
    word, fingerprint and raw value whole (``break-word``, not ``anywhere``,
    which made every character a break point), and a result's name wraps only
    after its date or its " · ", its date, its clock and its fingerprint each
    kept on one line (a name to the second as one unbreakable part set a
    column too wide for a protocol table and a five-column Compare: review of
    the fix); escaped."""
    from src.webui.components import models_page as mp
    from src.webui.components import results_segment as rs

    rule = re.search(r"\.shep-rs-tbl td \{\{?([^}]*)\}", mp.PAGE_CSS).group(1)
    assert "overflow-wrap: break-word" in rule and "anywhere" not in rule
    html = rs._name_html("2026-11-10 11:05:47 UTC · 3b2c7aea27be")
    assert html == ('<span class="shep-nowrap">2026-11-10</span> '
                    '<span class="shep-nowrap">11:05:47 UTC ·</span> '
                    '<span class="shep-nowrap">3b2c7aea27be</span>')
    assert rs._name_html("2026-11-10 11:05:47 UTC · run …9bfd7c23").endswith(
        '<span class="shep-nowrap">run …9bfd7c23</span>')
    assert rs._name_html("Merged record · 3b2c7aea27be") == (
        '<span class="shep-nowrap">Merged</span> <span class="shep-nowrap">record ·</span> '
        '<span class="shep-nowrap">3b2c7aea27be</span>')
    hostile = rs._name_html(f"run {HOSTILE} · {HOSTILE}")
    assert "<img" not in hostile and html_lib.escape(HOSTILE) in hostile


#: The longest part of a result's name that never breaks: "11:05:47 UTC ·".
#: At the dashboard's 15 px a part this long is about 110 px, so a name sets
#: no column wider than that (the browser check's panels are 892 and 937 px:
#: a five-column Compare's heads fit, as the protocol tables' Result column).
_NAME_PART_MAX = len("11:05:47 UTC ·")


def _unbreakable(markup):
    return [html_lib.unescape(part)
            for part in re.findall(r'<span class="shep-nowrap">(.*?)</span>', markup)]


def test_a_name_sets_no_column_wider_than_its_longest_part(world, monkeypatch):
    """Review of the fix (models-1, models-2): every name a read gives, to the
    minute, to the second or with its run id's own part, is shown as parts
    that each stay whole, none longer than a clock to the second with its
    " ·", and the table's name cell and Compare's column head are exactly
    those parts (the head's protocol words, under it, wrap as words), so
    neither widens its table past the panel. Measured in the browser by the
    author's re-look; here, in the rendered HTML."""
    from src.webui.components import results_segment as rs

    twins = _twins(world, monkeypatch)
    state, scope, shown = _shown(_state(world, monkeypatch))
    for read in (state, twins):
        names = [name for name in rs.row_names(read["rows"]) if name is not None]
        for name in names:
            parts = _unbreakable(rs._name_html(name))
            assert " ".join(parts) == name
            assert max(len(part) for part in parts) <= _NAME_PART_MAX, name
        table = rs.tables_html(read, scope, {**shown, "shown": read["shown"]})
        cells = re.findall(r'<td class="shep-rs-name">(.*?)</td>', table)
        assert sorted(cells) == sorted(rs._name_html(name) for name in names)
        keys = [value for _label, value in rs.compare_choices(read)][:rs.COMPARE_CAP]
        compare = rs.compare_state_html(read, keys, False, asked=len(keys))
        heads = re.findall(r'<th scope="col">(<span class="shep-nowrap">.*?)<div '
                           r'class="shep-sub">', compare)
        assert len(heads) == min(len(names), rs.COMPARE_CAP) and all(max(map(len, _unbreakable(head))) <= _NAME_PART_MAX
                             and " ".join(_unbreakable(head)) == _text(head)
                             for head in heads)


def test_the_shared_run_time_reads_a_run_id_as_the_test_page_does():
    """Review of the fix (AGENTS.md: no second implementation): the one
    `renderers.run_time` names a report store run id's time, to the minute and
    to the second, and names nothing else; the Test page names its report rows
    by this same function, not a copy of its own."""
    from src.webui.components import renderers as r
    from src.webui.components import test_page as tp

    assert tp.run_time is r.run_time
    for run_id in ["20261110T110547Z-a6eb26c11d90", "20261110T110547Z-", "usage.jsonl", "", None,
                   HOSTILE, "20261110T110547Z-A6EB26C11D90-9BFD7C23"]:
        for seconds in (False, True):
            assert r.run_time(run_id, seconds=seconds) is None, (run_id, seconds)
    assert r.run_time("20261110T110547Z-a6eb26c11d90-9bfd7c23") == "2026-11-10 11:05 UTC"
    assert r.run_time("20261110T110547Z-a6eb26c11d90-9bfd7c23",
                      seconds=True) == "2026-11-10 11:05:47 UTC"
    assert r.run_time("20261017T152412Z-15f098277b54-0a1b2c3d") == "2026-10-17 15:24 UTC"


def _read_in_full(state):
    """The disease results whose model and data version were read (no
    ``unknown`` for its spec or its references): the world's runs of one
    second are listed by their run ids' own random part, so a test that takes
    the first results takes these, never by their place."""
    return [row for row in _results(state)
            if not {"model.spec", "references"} & set(row["unknown"])]


def _finetune_base_supplied_kg(world, monkeypatch):
    """A finetune (its weight transfer in full), a base (its weight transfer
    recorded as null: it loads nothing), both of a supplied cohort, and a KG
    edge set's link-prediction result (its exposure block, no R10)."""
    state, _scope, _shown_rows = _shown(_state(world, monkeypatch), protocol="disease")
    row = _read_in_full(state)[0]
    finetune, base, kg = (copy.deepcopy(row) for _ in range(3))
    finetune["model"]["weight_transfer"] = {
        "loaded_from_parent": ["encoder", "input", "lp_head"],
        "initialised_new": ["patient_encoder", "task_head"], "trainable": "all",
        "optimizer": "new", "parts": {"encoder": "trainable", "task_head": "trainable"},
        "frozen_forward": "none", "downstream_initial_sha256": "acc5f61b6704" + "1" * 52}
    base["model"].update(learning_stage="base", task=None, weight_transfer=None)
    base["report_sha256"] = "b" * 64
    kg.update(protocol="kg_link_prediction", report_sha256="c" * 64)
    # A KG edge set's own identity, counts, content references and run counts,
    # as its kind records them (`results._data`, `version_references`).
    kg["data"] = {**kg["data"], "kind": "kg_edges", "role": "test",
                  "identity": {"split": {"rule": "facts-80/10/10/v1", "seed": 0},
                               "pairs": {"sha256": "2" * 64, "rows": 6}},
                  "counts": {"pairs": 6, "positive_edges": 8}}
    kg["references"] = [*(item for item in kg["references"] if item["name"] == "graph.kg"),
                        {"name": "pairs", "sha256": "3" * 64, "kept": "in the version",
                         "state": "verified", "reason": None}]
    kg["run_counts"] = {"counts": {"pairs": 6, "pairs_scored": 6},
                        "units": {"pairs": "pairs stored", "pairs_scored": "pairs scored"}}
    kg["unknown"] = {name: why for name, why in kg["unknown"].items() if name != "run_counts"}
    kg["exposure"] = {"state": "measured", "reading": "6 of 6 measured edges in the recorded "
                      "graph use", "definition_version": "kg-lp-exposure/v1",
                      "measured_edges": 6, "in_recorded_graph_use": 0, "problem": None,
                      "r10": None}
    return finetune, base, row, kg


def test_compare_keeps_not_applicable_not_recorded_and_none_apart(world, monkeypatch):
    """Browser check models-3 (skill §6: missing states keep their own words):
    a base's weight transfer recorded as null reads "none" on every row of
    the finetune's weight transfer, the finetune never "not recorded" there,
    and there is no bare row of it beside its entries; a KG edge set's
    exposure counts read "not applicable" for a supplied cohort's result
    (never "none", which would read as zero edges), and R10 "not applicable"
    for the KG edge set's; R10 says each label's state beside its name, the
    name not repeated (models-6)."""
    from src.webui.components import results_segment as rs

    finetune, base, supplied, kg = _finetune_base_supplied_kg(world, monkeypatch)
    for differences in (True, False):
        markup = rs.compare_html([finetune, base], differences)
        transfer = [label for label in _labels(markup) if label.startswith("Weight transfer")]
        assert "Weight transfer" not in transfer and len(transfer) >= 6
        for label in transfer:
            finetune_cell, base_cell = _cells(markup, label)
            assert base_cell == "none", label
            assert "not recorded" not in finetune_cell, label
    assert _cells(markup, "Weight transfer · Parts › Encoder") == ["trainable", "none"]
    markup = rs.compare_html([kg, supplied], False)
    for label in ("Exposure definition", "Measured edges", "In recorded graph use",
                  "Exposure problem"):
        kg_cell, supplied_cell = _cells(markup, label)
        assert supplied_cell == "○ not applicable" and kg_cell != "○ not applicable", label
    assert _cells(markup, "Measured edges") == ["6", "○ not applicable"]
    r10 = [label for label in _labels(markup) if label.startswith("R10 · ")]
    assert len(r10) == 4
    for label in r10:
        kg_cell, supplied_cell = _cells(markup, label)
        assert kg_cell == "○ not applicable", label
        assert not supplied_cell.startswith(label[len("R10 · "):]), label
    # A KG edge set's result whose record holds no exposure block: its
    # counts are not recorded, and the problem says why.
    bare = copy.deepcopy(kg)
    bare["exposure"] = {**kg["exposure"], "state": "not recorded", "reading": None,
                        "definition_version": None, "measured_edges": None,
                        "in_recorded_graph_use": None, "problem": "no exposure block"}
    markup = rs.compare_html([bare, kg], True)
    assert _cells(markup, "Measured edges") == ["○ not recorded", "6"]
    assert _cells(markup, "Exposure problem") == ["no exposure block", "none"]
    # The data's identity, counts and content references are each kind's own
    # (review of the fix: a KG edge set's split read "not recorded" for a
    # supplied cohort's result, beside its exposure's "not applicable"): an
    # entry only another kind's results hold is "not applicable"; one a result
    # of the column's own kind holds is "not recorded" for one that lacks it.
    markup = rs.compare_html([kg, supplied], False)
    assert _cells(markup, "Data identity · Split › Rule") == [
        "facts-80/10/10/v1", "○ not applicable"]
    assert _cells(markup, "Data identity · Samples SHA-256")[0] == "○ not applicable"
    assert _cells(markup, "Data counts · Pairs") == ["6", "○ not applicable"]
    assert _cells(markup, "Data counts · Source cases")[0] == "○ not applicable"
    assert _cells(markup, "Content references · pairs › SHA-256") == [
        "3" * 12, "○ not applicable"]
    assert _cells(markup, "Content references · samples › SHA-256")[0] == "○ not applicable"
    assert _cells(markup, "Content references · graph.kg › SHA-256")[0] == _cells(
        markup, "Content references · graph.kg › SHA-256")[1]
    lacking = copy.deepcopy(supplied)
    lacking["data"]["identity"].pop("source_sha256")
    markup = rs.compare_html([kg, supplied, lacking], False)
    assert _cells(markup, "Data identity · Source SHA-256") == [
        "○ not applicable", supplied["data"]["identity"]["source_sha256"][:12], "○ not recorded"]
    assert "not recorded" not in "".join(_cells(markup, "Data identity · Split › Rule"))


def test_compare_goes_down_into_nested_settings_and_says_the_spec_once(world, monkeypatch):
    """Browser check models-4 (skill §4: two things that differ, differences
    only; rule 1.2): a difference inside a nested setting is the row of that
    one entry, not the whole object in every column; the learning settings'
    copy of the spec, stage and task is not shown again (the Model group's
    rows say them); a list of named items (the content references) is one
    row per item and field; a field whose shape differs between columns
    stays one row, each value as recorded."""
    from src.webui.components import results_segment as rs

    state, _scope, _shown_rows = _shown(_state(world, monkeypatch), protocol="disease")
    frozen, full = (copy.deepcopy(row) for row in _read_in_full(state)[:2])
    loaders = {"train": {"batch_size": 8, "kind": "FinetuneSampler",
                         "recipe": {"learning_stage": "finetune", "operator_set": ["seed"],
                                    "reference_filled": {}}}}
    for row in (frozen, full):
        # The spec and capabilities as a record lists them: the stage, the
        # encoder's kind and the task are the model's own rows' facts.
        row["model"].update(learning_stage="finetune", task="disease", encoder="gatv2")
        row["model"]["spec"] = {"learning_stage": "finetune",
                                "encoder": {"kind": "gatv2", "heads": 2},
                                "task_head": {"task": "disease", "softmax_scale": 1.0}}
        row["model"]["capabilities"] = {"learning_stage": "finetune", "encoder": "gatv2",
                                        "sampled_hops": 2}
        row["model"]["learning_settings"] = {"model_spec": row["model"]["spec"],
                                             "learning_stage": "finetune", "task": "disease",
                                             "learning_rate": 1e-05, "save_top_k": 1}
        row["model"]["effective_settings"] = {"loaders": copy.deepcopy(loaders)}
        row["references"] = [{"name": "graph.kg", "sha256": "7" * 64, "kept": "workspace"},
                             {"name": "split.json", "sha256": "8" * 64, "kept": "workspace"}]
    frozen["model"]["effective_settings"]["loaders"]["train"]["recipe"].update(
        operator_set=["frozen_parts", "seed"], reference_filled={"frozen_parts": "none"})
    frozen["references"][1]["sha256"] = "9" * 64
    frozen["model"]["learning_settings"]["save_top_k"] = 3
    markup = rs.compare_html([frozen, full], True)
    training = [label for label in _labels(markup) if label.startswith(("Learning setting",
                                                                        "Effective setting"))]
    assert training == [
        "Learning setting · Top-k saves kept",
        "Effective setting · Loaders › Train › Recipe › Operator set",
        "Effective setting · Loaders › Train › Recipe › Reference filled › Frozen parts"]
    assert _cells(markup, training[-1]) == ["none", "○ not recorded"]
    assert _cells(markup, training[1]) == ['["frozen_parts", "seed"]', '["seed"]']
    assert [label for label in _labels(markup) if label.startswith("Content references")] == [
        "Content references · split.json › SHA-256"]
    every = _labels(rs.compare_html([frozen, full], False))
    assert "Learning setting · Learning rate" in every
    assert not any(label.startswith(("Learning setting · Model spec",
                                     "Learning setting · Learning stage",
                                     "Learning setting · Task")) for label in every)
    # The stage, the encoder's kind and the task: each one row, the Model
    # group's own (review of the fix: the spec's, the capabilities' and the
    # loaders' recipe's copies were rows of their own); the rest of the spec
    # and the capabilities as recorded.
    copies = {"Spec · Learning stage", "Capabilities · Learning stage",
              "Effective setting · Loaders › Train › Recipe › Learning stage",
              "Spec · Encoder › Kind", "Capabilities · Encoder", "Spec · Task head › Task"}
    assert not copies & set(every), copies & set(every)
    for words, value in (("Learning stage", "finetune"), ("Encoder", "gatv2"),
                         ("Task", "disease")):
        assert every.count(words) == 1, words
        assert _cells(rs.compare_html([frozen, full], False), words) == [value, value]
    assert {"Spec · Encoder › Heads", "Spec · Task head › Softmax scale",
            "Capabilities · Sampled hops", "Effective setting · Loaders › Train › Kind"
            } <= set(every)
    # One field, two shapes (a mapping and a plain value): one row, as recorded.
    mixed = copy.deepcopy(full)
    mixed["model"]["effective_settings"]["loaders"] = "none recorded"
    markup = rs.compare_html([frozen, mixed], True)
    assert _cells(markup, "Effective setting · Loaders")[1] == "none recorded"
    assert "operator_set" in _cells(markup, "Effective setting · Loaders")[0]


def test_compare_rows_are_named_in_the_segments_words(world, monkeypatch):
    """Browser check models-5 (skill §5: sentence case, full words): a
    metric row in the protocol tables' words (`caveats.metric_text`), a
    learning setting by the Training page's label, every other key in its
    words, its record key the row's title; no row of a disease result is
    named by a raw key."""
    from src.evaluation.caveats import metric_text
    from src.webui.components import results_segment as rs

    finetune, base, _supplied, kg = _finetune_base_supplied_kg(world, monkeypatch)
    finetune["model"]["learning_settings"] = {"learning_rate": 1e-05, "save_top_k": 1,
                                              "frozen_parts": "none"}
    finetune["data"]["identity"] = {"samples_sha256": "5" * 64}
    markup = rs.compare_html([finetune, base], False)
    labels = _labels(markup)
    for label in ("Learning setting · Learning rate", "Learning setting · Top-k saves kept",
                  "Learning setting · Frozen parts",
                  "Weight transfer · Downstream initial SHA-256",
                  "Data identity · Samples SHA-256", "Disease pooled MRR",
                  metric_text("disease", "macro_precision@3")):
        assert label in labels, label
    # Every row in words but the content references' items, named as recorded.
    assert not [label for label in labels if "_" in label
                and not label.startswith("Content references")]
    assert ('title="model.learning_settings.save_top_k" aria-label="Learning setting · Top-k '
            'saves kept">Top-k saves kept</th>') in markup
    assert 'title="metrics.pooled_mrr">Disease pooled MRR</th>' in markup
    assert rs.field_words("cudnn_benchmark") == "cuDNN benchmark"
    assert rs.field_words("cuda_version") == "CUDA version"
    # Short forms in full words, a count by the Test page's word (review of
    # the fix: "Base lr", "N excluded", "Val edges" and "eps" were left).
    finetune["model"]["effective_settings"] = {
        "schedule": {"base_lr": 0.001}, "loaders": {"val": {"val_edges": 6,
                                                            "val_without_negative": 0}}}
    markup = rs.compare_html([kg, finetune], False)
    labels = _labels(markup)
    for label in ("Effective setting · Schedule › Base learning rate",
                  "Effective setting · Loaders › Validation › Validation edges",
                  "Effective setting · Loaders › Validation › Validation edges without a "
                  "negative", "Data counts · Excluded"):
        assert label in labels, label
    assert 'title="data.counts.n_excluded" aria-label="Data counts · Excluded">Excluded</th>' \
        in markup
    assert (rs.field_words("distance_normalisation_eps"), rs.field_words("augmentation_k"),
            rs.field_words("sp_lower"), rs.field_words("dim")) == (
        "Distance normalisation epsilon", "Augmentation neighbours (K)",
        "Shortest-path lower", "Dimension")
    # The spec's and the effective settings' other short forms, as the real
    # rows hold them ("Spec · Input › Init", "Effective setting · AMP › Grad
    # scaler", "Spec · Encoder › Norm" and "Concat" were left).
    assert [rs.field_words(key) for key in ("init", "relation_init", "grad_scaler", "norm",
                                            "concat")] == [
        "Initialisation", "Relation initialisation", "Gradient scaler", "Normalisation",
        "Concatenate"]
    # What n_ranked counts is its mode's (a KG link-prediction run's pairs):
    # one row in words of neither, beside a KG edge set's column.
    assert _cells(markup, "Number ranked") == [
        str(kg["measurement"]["n_ranked"]), str(finetune["measurement"]["n_ranked"])]
    assert 'title="measurement.n_ranked">Number ranked</th>' in markup
    assert not [label for label in labels if "ases ranked" in label]
    # The cohort kind is the data's kind: the Data group's row says it, never
    # a "cohort" row for a KG edge set.
    assert "measurement.cohort_kind" not in markup and "Cohort kind" not in labels
    assert _cells(markup, "Data kind") is not None


def test_compare_says_a_columns_unknown_reason_once_per_field(world, monkeypatch):
    """Review of the fix (rule 1.2): a column whose field has no source (a
    result whose checkpoint is no longer listed, an ordinary case: results
    are kept whatever became of the weights they name; a disease result's run
    counts) says why once, on the field's first row; its later entry rows
    read "Unknown", the why their title, so differences only does not repeat
    one reason on every row of the other column's nested value. Nor does it
    list those rows (review of RC-P2-1): their only difference is the
    Unknown its first row explains, so differences only shows each such
    field's first row alone, and every row shows them all."""
    from src.api.services import results as service
    from src.webui.components import results_segment as rs

    finetune, _base, supplied, kg = _finetune_base_supplied_kg(world, monkeypatch)
    gone = copy.deepcopy(finetune)
    why = "its checkpoint is not in a listed workspace"
    for name in (*service._MODEL_FIELDS, *service._SETTINGS_FIELDS):
        gone["model"][name] = None
        gone["unknown"][f"model.{name}"] = why
    for differences in (True, False):
        markup = rs.compare_html([finetune, gone], differences)
        transfer = [label for label in _labels(markup) if label.startswith("Weight transfer")]
        assert (len(transfer) == 1) if differences else (len(transfer) >= 6)
        assert [_cells(markup, label)[1] for label in transfer] == (
            [f"○ Unknown: {why}"] + ["○ Unknown"] * (len(transfer) - 1))
        assert (f'title="{html_lib.escape(why)}">○ Unknown</td>' in markup) is not differences
        # Once per field: of each model field's rows, the column says why on
        # its first only.
        said: dict = {}
        for ref, cells in re.findall(r'<th scope="row" title="([^"]*)"[^>]*>.*?</th>(.*?)</tr>',
                                     markup):
            cell = re.findall(r"<td[^>]*>.*?</td>", cells)[1]
            if f'title="{html_lib.escape(why)}"' in cell:
                said.setdefault(".".join(ref.split(".")[:2]), []).append(_text(cell))
        assert {"model.spec", "model.weight_transfer", "model.capabilities"} <= set(said)
        for field, texts in said.items():
            assert texts == [f"○ Unknown: {why}"] + ["○ Unknown"] * (len(texts) - 1), field
        assert _text(markup).count(f"Unknown: {why}") == len(said)
    for differences, shown in ((True, 1), (False, 4)):
        markup = rs.compare_html([kg, supplied], differences)
        counts = [label for label in _labels(markup) if label.startswith("Run counts · ")]
        assert len(counts) == shown
        assert [_cells(markup, label)[1].split(":")[0] for label in counts] == (
            ["○ Unknown"] * shown)
        assert _text(markup).count("its record holds no run counts") == 1


def test_compare_says_each_columns_report_once(world, monkeypatch):
    """Browser check models-6 (rule 1.2): each column's head says its
    report's 12 hex digits, so no Report row repeats them below; where two
    reports share their first 12, the Report row shows every digit, which no
    head can."""
    from src.webui.components import results_segment as rs

    state, _scope, _shown_rows = _shown(_state(world, monkeypatch), protocol="disease")
    first, second = (copy.deepcopy(row) for row in _results(state)[:2])
    for differences in (True, False):
        markup = rs.compare_html([first, second], differences)
        text = _text(markup)
        for row in (first, second):
            assert text.count(row["report_sha256"][:12]) == 1
        assert _cells(markup, "Report") is None
    first["report_sha256"] = "ab" * 6 + "1" * 52
    second["report_sha256"] = "ab" * 6 + "2" * 52
    markup = rs.compare_html([first, second], True)
    assert _cells(markup, "Report") == [first["report_sha256"], second["report_sha256"]]


def _css_rule(selector):
    """The declarations of Models' CSS rule for `selector` (its exact text)."""
    from src.webui.components import models_page as mp

    found = re.search(re.escape(selector) + r" \{\{?([^}]*)\}", mp.PAGE_CSS)
    return found and found.group(1)


def test_compare_keeps_numbers_and_run_ids_whole_and_scrolls_inside_its_wrapper(
        world, monkeypatch):
    """Browser re-look relook-1 (plan §4 invariant 5, raw values; skill §5):
    a raw number never breaks inside (its cell does not wrap; split mid-digit,
    "0.2916666666666" over "6663" read as two numbers in a right-aligned
    column), and a run id breaks only after one of its own hyphens (an
    explicit ``<wbr>``: no break is allowed between a hyphen and a digit);
    prose, JSON and reasons still break anywhere, so long values never set the
    columns' widths. Five columns of raw values can be wider than the panel:
    the table scrolls sideways inside its own wrapper, never the page, its
    Field column pinned and its group and sub-heading words kept in view.
    The geometry is measured in headless Chromium with the app's CSS by the
    author's scripted check; here, the markup and the rules it relies on."""
    from src.webui.components import results_segment as rs

    finetune, base, _supplied, _kg = _finetune_base_supplied_kg(world, monkeypatch)
    finetune["run_id"] = "20261110T110716Z-31ab26b3b0fe-45ad9e27"
    base["run_id"] = "20261110T110549Z-eea723a97e5d-9364843b"
    finetune["model"].update(parent_run="20261110-110512_probe-base_d4e3e7",
                             family_id="20261110-110512_probe-base_d4e3e7")
    base["model"].update(parent_run=None, family_id="20261017-152412_other_0a1b2c")
    finetune["metrics"][0]["value"] = 0.29166666666666663
    base["metrics"][0]["value"] = 0.125
    markup = rs.compare_html([finetune, base], True)
    # A number: its own cell, marked as one, unrounded; the cell never wraps.
    assert '<td class="shep-num">0.29166666666666663</td>' in markup
    number = _css_rule("#shep-page-models .shep-rs-cmp td.shep-num")
    assert "white-space: nowrap" in number and "overflow-wrap: normal" in number
    # A run id: each part to and with its hyphen kept whole, a break allowed
    # only between them; the parent's and the family's training runs alike.
    run = re.search(r'title="run_id">Run</th>(.*?)</tr>', markup).group(1)
    assert run.startswith(
        '<td><span class="shep-nowrap">20261110T110716Z-</span><wbr>'
        '<span class="shep-nowrap">31ab26b3b0fe-</span><wbr>'
        '<span class="shep-nowrap">45ad9e27</span></td>')

    def whole(cells):  # a cell's text, its parts joined as the line reads them
        return [html_lib.unescape(re.sub(r"<[^>]+>", "", cell))
                for cell in re.findall(r"<td[^>]*>.*?</td>", cells)]

    assert whole(run) == [finetune["run_id"], base["run_id"]]
    for ref, value in (("model.parent_run", "20261110-110512_probe-base_d4e3e7"),
                       ("model.family_id", "20261110-110512_probe-base_d4e3e7")):
        cells = re.search(rf'title="{re.escape(ref)}">[^<]*</th>(.*?)</tr>', markup).group(1)
        assert cells.startswith('<td><span class="shep-nowrap">20261110-</span><wbr>'
                                '<span class="shep-nowrap">110512_probe-</span><wbr>'
                                '<span class="shep-nowrap">base_d4e3e7</span></td>'), ref
        assert whole(cells)[0] == value
    # A run id's missing value keeps its own word ("none", recorded as null).
    assert _cells(markup, "Parent run")[1] == "none"
    assert "<img" not in rs._run_id_html(f"{HOSTILE}-{HOSTILE}")
    assert rs._run_id_html("a-b-") == ('<span class="shep-nowrap">a-</span><wbr>'
                                       '<span class="shep-nowrap">b-</span>')
    # Prose, JSON and reasons still break anywhere (relook-1: break-word on
    # every cell made a five-column Compare 2,699 px wide).
    assert "overflow-wrap: anywhere" in _css_rule(
        "#shep-page-models .shep-rs-cmp td, #shep-page-models .shep-rs-unread td")
    # Wider than the panel: the wrapper scrolls sideways, the Field column
    # (its head and every row's name) pinned at the left over an opaque
    # background, and the group and sub-heading words kept in view.
    assert "overflow-x: auto" in _css_rule("#shep-page-models .shep-rs-wrap")
    pinned = _css_rule('#shep-page-models .shep-rs-cmp tbody th[scope="row"], '
                       '#shep-page-models .shep-rs-cmp thead th:first-child')
    assert "position: sticky" in pinned and "left: 0" in pinned
    assert "background: var(--shep-panel)" in _css_rule(
        '#shep-page-models .shep-rs-cmp th[scope="row"]')
    assert "position: sticky" in _css_rule("#shep-page-models .shep-rs-cmp .shep-rs-pin")
    assert '<tr class="shep-rs-grp"><th scope="colgroup" colspan="3"><span class="shep-rs-pin">' \
           'Result</span></th></tr>' in markup


def _layers(declarations):
    """A rule's ``background`` layers, top first (split at its top-level commas)."""
    value = re.search(r"background: ([^;]*);", declarations).group(1)
    layers, depth, start = [], 0, 0
    for at, char in enumerate(value):
        depth += {"(": 1, ")": -1}.get(char, 0)
        if char == "," and depth == 0:
            layers.append(value[start:at])
            start = at + 1
    return [" ".join(layer.split()) for layer in layers + [value[start:]]]


def test_a_results_table_wider_than_its_panel_shades_its_right_edge():
    """Review of the re-look fix: five columns of raw values (1,057 px) scroll
    inside a 937 or 892 px wrapper, and nothing on screen said a column lay
    beyond the edge but the wrapper's scrollbar, at the bottom of a table
    about 19,000 px tall; the cut column read as partial tokens. The
    wrapper's right edge is shaded while it is not scrolled to its end: a
    shade fixed at the edge (``scroll``) under a cover in the panel's colour
    that moves with the table (``local``), at its end covering the shade
    whole, as at a table that fits. Compare's value cells and sub-heading
    rows have no background of their own, so the shade shows through them.
    Measured by the author's scripted check in headless Chromium (the edge's
    pixels: shaded at the start of a five-column Compare, the panel's white
    at its end and on a table that fits); here, the rule it relies on."""
    cover, shade, base = _layers(_css_rule("#shep-page-models .shep-rs-wrap"))
    found = re.fullmatch(r"linear-gradient\(to left, var\(--shep-panel\) (\d+)px, transparent\) "
                         r"right / (\d+)px 100% no-repeat local", cover)
    assert found, cover
    opaque, wide = (int(n) for n in found.groups())
    edge = re.fullmatch(r"linear-gradient\(to left, rgba\(20, 24, 40, (\.\d+)\), transparent\) "
                        r"right / (\d+)px 100% no-repeat scroll", shade)
    assert edge, shade
    assert opaque >= int(edge.group(2)) and wide > opaque
    assert base == "var(--shep-panel)"  # the cover's colour, the panel's
    assert "background" not in _css_rule("#shep-page-models .shep-rs-cmp tr.shep-rs-sub td")
    # The lightest words in a value cell (a state word, ``shep-muted``) keep
    # 4.5:1 over the shade's darkest pixel (skill §10).
    from src.webui import style
    from tests.unit.test_webui_button_contrast import contrast

    tokens = dict(re.findall(r"(--shep-[a-z0-9-]+): (#[0-9a-f]{6})", style.SHELL_CSS))
    assert tokens["--shep-panel"] == "#ffffff"
    alpha = float(edge.group(1))
    darkest = "#" + "".join(f"{round(255 * (1 - alpha) + c * alpha):02x}" for c in (20, 24, 40))
    assert 0 < alpha and contrast(tokens["--shep-muted"], darkest) >= 4.5


def test_compare_names_a_nested_entry_by_its_own_words_under_its_path(world, monkeypatch):
    """Browser re-look relook-2 (skill §5: front-load): a nested entry's row
    is named by its own words, first; the path above it is said once, as a
    sub-heading row under its group, shown where one of its entries is shown
    (differences only hides a path whose entries are all alike) and not again
    while the rows shown next share it. The row's whole name is its
    accessible name and its record key its title; `compare_fields` keeps the
    whole name. (Up to 63 characters of shared path came first and wrapped
    a label to 4-5 lines in the Field column, its meaning on the last.)
    The sub-heading is a plain cell, not a header (review of the re-look
    fix): every row is in the table's one row group, where a row-group
    header heads every later cell of the table (HTML's table model), so a
    path headed the entries of later paths and groups too, and each entry
    said its path twice; the entry's accessible name already says it."""
    from src.webui.components import results_segment as rs

    state, _scope, _shown_rows = _shown(_state(world, monkeypatch), protocol="disease")
    first, second = (copy.deepcopy(row) for row in _read_in_full(state)[:2])
    for row, rate in ((first, "1e-5 (README U3)"), (second, "5e-5 (README U3)")):
        row["model"]["effective_settings"] = {"loaders": {"train": {
            "batch_size": 8, "kind": "FinetuneSampler",
            "recipe": {"reference_filled": {"learning_rate": rate,
                                            "scheduler": f"none ({rate})",
                                            "gradient_clip": "1.0"}}}}}
    path = "Effective setting · Loaders › Train › Recipe › Reference filled"
    markup = rs.compare_html([first, second], True)
    sub = re.findall(r'<tr class="shep-rs-sub"><td colspan="3">'
                     r'<span class="shep-rs-pin">([^<]*)</span></td></tr>', markup)
    # Said once, above its two differing entries; the alike gradient clip and
    # the train loader's alike entries are hidden, and so is their path.
    assert [html_lib.unescape(words) for words in sub].count(path) == 1
    assert "Effective setting · Loaders › Train" not in [html_lib.unescape(w) for w in sub]
    ref = "model.effective_settings.loaders.train.recipe.reference_filled"
    opening = re.escape(f'<span class="shep-rs-pin">{html_lib.escape(path)}</span></td></tr>')
    block = re.search(opening + r'(.*?)(?:<tr class="shep-rs-(?:sub|grp)"|</tbody>)',
                      markup).group(1)
    assert re.findall(r'<tr class="shep-rs-entry"><th scope="row" title="([^"]*)" '
                      r'aria-label="([^"]*)">([^<]*)</th>', block) == [
        (f"{ref}.learning_rate", html_lib.escape(f"{path} › Learning rate"), "Learning rate"),
        (f"{ref}.scheduler", html_lib.escape(f"{path} › Scheduler"), "Scheduler")]
    assert _cells(markup, f"{path} › Learning rate") == ["1e-5 (README U3)", "5e-5 (README U3)"]
    # Every row's words, every view: an entry's row says its own words (no
    # path in them), under the sub-heading of its path; a field's own row
    # says its words and has no sub-heading of its own.
    for differences in (True, False):
        markup = rs.compare_html([first, second], differences)
        # The sub-heading heads nothing: no row-group header (in the table's
        # one row group it would head every later cell), no sub-heading as a
        # header cell, and no header cell without a scope (HTML's auto scope
        # makes one alone in its row a column header over the rows below).
        body = re.search(r"<tbody>(.*)</tbody>", markup).group(1)
        assert 'scope="rowgroup"' not in markup and "<tbody" not in body
        assert '<tr class="shep-rs-sub"><th' not in markup
        assert set(re.findall(r'<th(?=[ >])( scope="[a-z]+")?', body)) == {
            ' scope="row"', ' scope="colgroup"'}
        assert len(re.findall(r'<tr class="shep-rs-sub">', markup)) == len(
            re.findall(r'<tr class="shep-rs-sub"><td colspan="3"><span class="shep-rs-pin">'
                       r'[^<]*</span></td></tr>', markup)) > 0
        heading = None
        for kind, words, attrs, name in re.findall(
                r'<tr class="shep-rs-(sub|grp)"><t[hd][^>]*><span class="shep-rs-pin">([^<]*)'
                r'</span></t[hd]></tr>|<tr(?: class="shep-rs-entry")?><th scope="row"([^>]*)>'
                r'([^<]*)</th>', markup):
            if kind:
                heading = html_lib.unescape(words) if kind == "sub" else None
                continue
            whole = re.search(r'aria-label="([^"]*)"', attrs)
            name = html_lib.unescape(name)
            if whole is None:
                assert " › " not in name, name
                continue
            assert heading is not None and " › " not in name and " · " not in name, name
            assert html_lib.unescape(whole.group(1)) in (f"{heading} › {name}",
                                                          f"{heading} · {name}"), name
    # The whole name stays the row's label for every other reader.
    fields = {label: value for _group, label, value, _why in rs.compare_fields(first)}
    assert fields[f"{path} › Learning rate"] == "1e-5 (README U3)"
    assert fields["Effective setting · Loaders › Train › Batch size"] == 8


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
