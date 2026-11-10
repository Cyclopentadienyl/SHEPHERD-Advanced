"""
Research › Evaluate › Models › Results: every result in scope, compared and exported (research comparison Stage 8b).
===================================================================================================================
The minimum functional interface to the one read-only results service
(`src/api/services/results.py`; research comparison §5 Stage 8, the owner's O6
and O8): a third segment of the selected model's detail on Research › Models,
beside "Recorded results" and "Usage history". No new page.

**What it lists.** The service's rows (`results.query` over the server's own
source, `deployment_sources`), read when the segment is opened and again on
"↻ Read again", kept in the session's state, and shown by scope and filters
without reading again:

- **Scope**: this family (the selected model's family, in this workspace) or
  this workspace (every result found in it: a listed checkpoint of its model
  or a ledger holding its record), the service's own ``family`` and
  ``workspace`` filters. A result this server keeps that is found in no
  listed workspace (a kept report no listed ledger registered, whose model's
  weights are not listed now: moved, or a ``last.pt`` saved over) has no
  workspace to be scoped by, so this workspace lists it too, its model
  "Unknown" with why, and the status line counts it apart in either scope:
  never hidden. Comparing across workspaces is not built in this page this
  round: a capability gap, said in the empty and help text, not a rule; the
  command line's export (`scripts/export_results.py`) joins workspaces
  through the same service.
- **Filters**: protocol, data kind, role and environment class, each offering
  the values the scope's rows hold (the service's `filter_values`; a field
  without a value offers none and matches no value).
- **One table per protocol**, the tables in the order their protocols are
  first shown, each newest first by run time: the service's own listing order
  (runs by their run directory's time, then records merged before each run had
  its own record by their registration time), **chosen so that nothing is
  ranked**. Nothing is ordered by a metric; a default order for task records
  and the figure each task is ordered by stay the owner's open item
  (`docs/working/BACKLOG.md` 63). Each row: its result (run time and report),
  its model (its family too in the workspace scope), its data (kind and
  role, never "test" for another role), its metrics as Models shows a record
  of its protocol (a described task's two figures, unrounded where its
  condition group's runs differ on them, as Models' contest shows them; any
  other protocol's every metric as recorded), its state (report
  kept or "report not available" for a ledger-only record; registered, not
  registered, merged, or a record naming it that cannot be read; bytes that
  differ elsewhere; how its condition group compares) and its exposure where
  its record holds one (a KG edge set's reading, a supplied cohort's R10 one
  click away; a generated split's is its allocation's, not compared), with
  its environment. Every item that could not be read is listed with its
  reason after the tables, whatever the scope and filters (the service's
  filters never hide one).

**Compare.** The first five of the results chosen from those shown
(`COMPARE_CAP`, a cap of this view only: the choice, a query and an export
are never capped), in one table with the results as columns (each headed by
its protocol's words) and **differences only** on by default: the rows of
model-side and run conditions (the operation, the parent and family, every
value of the spec and its relation to the reference recipe, the recorded and
effective training settings, the data's identity and role, the measurement's
settings, the environment, the exposure) and each metric, each group once,
in that order. **Values side by side, as recorded; no difference is
computed.** Each metric row is one protocol's (its name with the protocol's
words): a result of another protocol reads "another protocol" there, so two
protocols' metrics are never one row as one scale. Each missing value keeps
its own words: a field without a source reads "Unknown" with why (the row's
``unknown``), never a number; a value recorded as null reads "none"; a field
the result's records do not hold at all reads "not recorded".

**Export.** Every result chosen or every row shown (the scope's and filters'
rows, the unreadable items among them), uncapped, through the service's one
serializer (`results.to_csv`, `results.to_json`), as a new file to download
from the dashboard's one download directory (`src/webui/exports.py`,
`write_new`: its bytes are the serializer's text).

**D8.** Listing results opens checkpoint files (the service lists every
workspace's checkpoints, SPEC_4 C3), so every handler here, reading,
showing, comparing and exporting, takes its event's ``gr.Request`` and asks
D8's one check first (`operator_access`); a refused request reads and lists
nothing and says why, as the shared model list does (`model_list.list_models`).
No new HTTP route: the handlers call the service in process.

**D7.** The rows hold measurement aggregates only (no case id, rank or server
path; their reasons are redacted by the service); every value is escaped here.

Module: src/webui/components/results_segment.py
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import gradio as gr

from src.api.services import results as service
from src.evaluation.caveats import (
    DATA_KIND_WORDS,
    MEASUREMENT_NAMES,
    R10_LABEL_TEXT,
    TASK_NAMES,
    label_words,
    metric_text,
    mode_label,
    mode_note,
    recorded_value_text,
    version_words,
)
from src.utils.run_environment import native_identity_text
from src.webui.components import model_list as ml
from src.webui.components import models_page as mp
from src.webui.components import renderers as r
from src.webui.components.help import help_button
from src.webui.exports import file_slug, write_new
from src.webui.operator_access import operator_access
from src.webui.shell import plain_html

#: The segment's value in the detail's segment control (`models_page`'s).
RESULTS = mp.RESULTS
#: The scopes: the selected model's family in this workspace, or this workspace.
FAMILY, WORKSPACE = "family", "workspace"
SCOPE_CHOICES = [("This family", FAMILY), ("This workspace", WORKSPACE)]
#: "Any value" in a filter.
ALL = "all"
#: The filters offered here, each a filter of the service (`results.FILTERS`),
#: with its label.
FILTERS: Tuple[Tuple[str, str], ...] = (("protocol", "Protocol"), ("data_kind", "Data kind"),
                                        ("role", "Role"), ("environment_class", "Environment"))
#: How many results Compare shows side by side: this view's cap only.
COMPARE_CAP = 5
#: What Export writes.
SHOWN, SELECTED = "shown", "selected"
EXPORT_ROWS = [("All shown", SHOWN), ("Selected", SELECTED)]
EXPORT_FORMATS = [("CSV", "csv"), ("JSON", "json")]

#: Said where the scope holds nothing, and in the help: a gap, not a rule.
CROSS_WORKSPACE_GAP = (
    "Comparing results across workspaces is not built in this page yet; on the machine, "
    "scripts/export_results.py joins workspaces through the same results service.")

#: The help texts this segment's ⓘ name (registered by Models' page builder).
SEGMENT_HELP = {
    "models-results": (
        "Results",
        "Every kept report and readable ledger record of this family or this workspace, read "
        "by the one results service.\n"
        "• One table per protocol, newest first by run time; nothing is ordered by a metric.\n"
        "• Compare shows the first five chosen side by side, values as recorded, with no "
        "computed difference.\n"
        "• Export writes every chosen or every shown row as CSV or JSON, uncapped.\n"
        + CROSS_WORKSPACE_GAP,
    ),
}

#: The session's state before anything is read.
EMPTY_STATE: Dict[str, Any] = {"read_id": None, "read_at": None, "rows": [], "refused": None,
                               "shown": []}


# ---------------------------------------------------------------------------
# Words
# ---------------------------------------------------------------------------
def protocol_words(protocol: Any) -> str:
    """A protocol as an option names it: its measurement's words ("Disease",
    "KG link prediction"), any other as "Mode <name>", never another's."""
    if not isinstance(protocol, str) or not protocol.strip():
        return "Mode not recorded"
    return MEASUREMENT_NAMES.get(protocol) or f"Mode {protocol}"


def _option_words(name: str, value: str) -> str:
    """A filter value as its option reads: the words of a protocol or of a
    data kind (`caveats.DATA_KIND_WORDS` as a label), else as recorded. A
    field without a value offers no option (`results.filter_values`)."""
    if name == "protocol":
        return protocol_words(value)
    if name == "data_kind" and value in DATA_KIND_WORDS:
        return label_words(DATA_KIND_WORDS[value])
    return value


def _model_words(model: Mapping[str, Any]) -> str:
    """What the result's model is, as the shared list's first word says it:
    "Base", a task's name, or that its record was not read here."""
    if model.get("learning_stage") == "base":
        return "Base"
    task = model.get("task")
    if task in TASK_NAMES:
        return TASK_NAMES[task]
    return "Model record not read"


def _utc(moment: Optional[datetime] = None) -> str:
    return (moment or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M UTC")


def row_label(row: Mapping[str, Any]) -> str:
    """A result's short name in the tables and Compare's options: its run's
    time (a record merged before each run had its own record has none) and
    its report's 12 hex digits."""
    report = r.short_hex(row.get("report_sha256"))
    if row.get("result") == service.HISTORICAL:
        return f"Merged record · {report}"
    when = mp.run_time(row.get("run_id"))
    return f"{when or 'run ' + str(row.get('run_id'))} · {report}"


# ---------------------------------------------------------------------------
# Scope and filters (pure)
# ---------------------------------------------------------------------------
def scope_of(listing: Mapping[str, Any], selected: Any, scope: Any) -> Dict[str, Any]:
    """The scope asked, as the shared listing allows it: this workspace (the
    listing's), or the selected model's family in it. A selected file whose
    record names no family has no family scope: this workspace, saying why."""
    workspace = listing.get("workspace") if isinstance(listing, Mapping) else None
    item = ml.find_item(listing, selected) if isinstance(listing, Mapping) else None
    family = item.get("family_id") if isinstance(item, Mapping) else None
    out = {"scope": WORKSPACE, "workspace": workspace, "family": None, "note": None}
    if scope == FAMILY:
        if isinstance(family, str) and family:
            out.update(scope=FAMILY, family=family)
        else:
            out["note"] = ("The selected file's record names no family, so this workspace's "
                           "results are shown.")
    return out


def _criteria(scope: Mapping[str, Any]) -> Dict[str, List[str]]:
    criteria = {"workspace": [str(scope.get("workspace"))]}
    if scope.get("scope") == FAMILY:
        criteria["family"] = [str(scope["family"])]
    return criteria


def scope_words(scope: Mapping[str, Any]) -> str:
    if scope.get("scope") == FAMILY:
        return (f"the family from {mp.run_words(scope.get('family'))} in workspace "
                f"{scope.get('workspace')}")
    return f"workspace {scope.get('workspace')}"


def filter_choices(rows: Iterable[Mapping[str, Any]], name: str) -> List[Tuple[str, str]]:
    """A filter's options: "All", then each value the rows hold, in the order
    first found (no value is put first by its name), unreadable items aside."""
    values: Dict[str, None] = {}
    for row in rows:
        if row.get("result") != service.UNREADABLE:
            values.update(dict.fromkeys(service.filter_values(row, name)))
    return [("All", ALL), *((_option_words(name, value), value) for value in values)]


def unplaced(row: Mapping[str, Any]) -> bool:
    """Whether a result was found in no listed workspace: no listed
    checkpoint is its model's and no listed ledger holds its record (a kept
    report no listed ledger registered, whose weights are not listed now), so
    the service's ``workspace`` filter has no value for it."""
    return (row.get("result") != service.UNREADABLE
            and not service.filter_values(row, "workspace"))


def shown_results(rows: Sequence[Mapping[str, Any]], scope: Mapping[str, Any],
                  filters: Mapping[str, Any]) -> Dict[str, Any]:
    """The rows of `scope` (the service's ``workspace`` and ``family``
    filters, and in this workspace also every result found in no listed
    workspace, `unplaced`, which no workspace could hold otherwise), each
    filter's options from them, and the rows the chosen filter values keep (a
    value no longer offered reads as All): by index into `rows`, in the
    service's order. Unreadable items are always kept. ``unplaced`` counts the
    results found in no listed workspace, ``unplaced_in_scope`` those of them
    in scope, so neither scope hides one without saying so."""
    indexed = list(enumerate(rows))
    criteria = _criteria(scope)
    in_scope_ids = {id(row) for row in service.filtered([row for _i, row in indexed], criteria)}
    # A result found in no listed workspace has no workspace to match: kept
    # by its family alone (none is known today: its model is not listed).
    homeless = [row for _i, row in indexed if unplaced(row)]
    homeless_in_scope = service.filtered(
        homeless, {name: values for name, values in criteria.items() if name != "workspace"})
    in_scope_ids |= {id(row) for row in homeless_in_scope}
    in_scope = [(i, row) for i, row in indexed if id(row) in in_scope_ids]
    choices = {name: filter_choices((row for _i, row in in_scope), name) for name, _l in FILTERS}
    chosen = {}
    for name, _label in FILTERS:
        value = filters.get(name)
        offered = {v for _w, v in choices[name]}
        chosen[name] = value if value in offered and value != ALL else ALL
    criteria = {name: [value] for name, value in chosen.items() if value != ALL}
    kept_ids = {id(row) for row in service.filtered([row for _i, row in in_scope], criteria)}
    shown = [i for i, row in in_scope if id(row) in kept_ids]
    return {"choices": choices, "chosen": chosen, "shown": shown,
            "in_scope": sum(1 for _i, row in in_scope if row.get("result") != service.UNREADABLE),
            "unplaced": len(homeless), "unplaced_in_scope": len(homeless_in_scope)}


def _key(state: Mapping[str, Any], index: int) -> str:
    return f"{state.get('read_id')}:{index}"


def compare_choices(state: Mapping[str, Any]) -> List[Tuple[str, str]]:
    """Compare's options: each shown result (never an unreadable item) by its
    label, made unique, its value the read and the row's index, so a choice
    from an earlier read names nothing now."""
    rows = state.get("rows") or []
    out: List[Tuple[str, str]] = []
    seen: Dict[str, int] = {}
    for index in state.get("shown") or []:
        row = rows[index]
        if row.get("result") == service.UNREADABLE:
            continue
        label = row_label(row)
        seen[label] = seen.get(label, 0) + 1
        out.append((label if seen[label] == 1 else f"{label} ({seen[label]})",
                    _key(state, index)))
    return out


def chosen_rows(state: Mapping[str, Any], chosen: Any) -> List[Mapping[str, Any]]:
    """The rows `chosen` names (keys of this read's shown results), in the
    order chosen; a key of another read or not shown names none."""
    rows = state.get("rows") or []
    shown = set(state.get("shown") or [])
    picked: List[int] = []
    for key in chosen or []:
        read, _sep, index = str(key).rpartition(":")
        if read == str(state.get("read_id")) and index.isdigit() and int(index) in shown \
                and int(index) not in picked \
                and rows[int(index)].get("result") != service.UNREADABLE:
            picked.append(int(index))
    return [rows[index] for index in picked]


# ---------------------------------------------------------------------------
# The tables (pure HTML)
# ---------------------------------------------------------------------------
def _digest_html(value: Any) -> str:
    if not r.is_digest(value):
        return '<span class="shep-muted">Unknown</span>'
    return f'<code class="shep-fpv" title="{r.e(value)}">{r.e(r.short_hex(value))}</code>'


def _model_cell(row: Mapping[str, Any], *, family: bool = True) -> str:
    """The result's model: its fingerprint and what it is, and its family
    where `family` (not in the family scope, whose status line names the one
    family every row shares: one fact once)."""
    model = row.get("model") or {}
    unknown = row.get("unknown") or {}
    words = _model_words(model)
    if words == "Model record not read":
        why = unknown.get("model.learning_stage") or "its record was not read"
        words_html = (f'<span class="shep-muted" title="{r.e(why)}">○ Unknown: '
                      f'{r.e(r.short_reason(why, 80))}</span>')
    else:
        words_html = r.e(words)
    family_id = model.get("family_id")
    sub = (f'<div class="shep-sub">family from {r.e(mp.run_words(family_id))}</div>'
           if family and isinstance(family_id, str) else "")
    return f'<div>{_digest_html(model.get("checkpoint_sha256"))} {words_html}</div>{sub}'


def _data_cell(row: Mapping[str, Any]) -> str:
    data = row.get("data") or {}
    kind, role = data.get("kind"), data.get("role")
    which = (f"supplied cohort, role {role}" if kind == "supplied"
             else version_words(kind, role))
    synthetic = (" " + r.chip("synthetic", "amber")
                 if data.get("dataset_origin") == "synthetic" else "")
    return (f'<div class="shep-w600">{r.e(data.get("label"))}</div>'
            f'<div class="shep-sub">{r.e(which)}{synthetic}</div>')


def _metric_value(metric: Mapping[str, Any]) -> str:
    """A metric as recorded: its value unrounded, or its status (a value that
    could not be computed is a status, never a number)."""
    if metric.get("status") not in (None, "recorded", "computed"):
        return str(metric.get("status"))
    return recorded_value_text(metric.get("value"))


def _metrics_cells(protocol: Any, row: Mapping[str, Any]) -> str:
    """A described task's two figures, four decimals, as Models' columns show
    them; a figure its condition group's runs differ on as recorded,
    unrounded (`models_page.contested_value_text`, Models' contest), so two
    runs that differ never read as equal beside "runs ... differ". Any other
    protocol's every metric as recorded."""
    metrics = row.get("metrics") or []
    columns = mp.RECORD_COLUMNS.get(protocol)
    if columns:
        by_name = {m.get("name"): m.get("value") for m in metrics}
        group = row.get("group") or {}
        differing = (set(group.get("differing") or []) if group.get("state") == "contested"
                     else set())
        cells = []
        for name in columns:
            value = by_name.get(name)
            text = (mp.contested_value_text(value) if name in differing
                    else mp.fmt_metric(name, value))
            cells.append(f'<td class="shep-num">{r.e(text)}</td>')
        return "".join(cells)
    if not metrics:
        return '<td><span class="shep-muted">no metric recorded</span></td>'
    lines = "".join(f'<div><code>{r.e(m.get("name"))}</code> {r.e(_metric_value(m))}</div>'
                    for m in metrics)
    return f"<td>{lines}</td>"


#: A row's registration as its State cell says it (the service's states).
_REGISTERED_WORDS = {
    service.REGISTERED: ("Registered", ""),
    service.NOT_REGISTERED: ("○ Not registered", "shep-muted"),
    service.MERGED: ("Merged record: registered before each run had its own record", ""),
    service.POSSIBLY_MERGED: ("○ Not registered as its own run; possible overlap with a merged "
                              "record", "shep-muted"),
}


def _state_cell(row: Mapping[str, Any]) -> str:
    lines = []
    if row.get("report") == service.REPORT_KEPT:
        lines.append("<div>Report kept</div>")
    else:
        lines.append('<div class="shep-muted">○ Report not available (ledger record only)</div>')
    registered = row.get("registered")
    words, css = _REGISTERED_WORDS.get(registered, (None, None))
    if words is None:
        lines.append(f'<div>{mp.warn_mark_html(str(registered))}</div>')
    else:
        lines.append(f'<div class="{css}">{r.e(words)}</div>')
    if row.get("conflict"):
        lines.append(f'<div>{mp.warn_mark_html("report bytes differ elsewhere")}</div>')
    group = row.get("group") or {}
    deterministic = (row.get("measurement") or {}).get("deterministic_algorithms")
    if group.get("state") == "contested":
        differing = ", ".join(str(d) for d in group.get("differing") or [])
        text = f"runs under these conditions differ: {differing}" if differing else \
            "runs under these conditions differ"
        lines.append(f'<div>{mp.warn_mark_html(text + " — investigate")}</div>'
                     if deterministic is True else
                     f'<div class="shep-muted">{r.e(text)}</div>')
    elif group.get("state") == "cannot_be_checked":
        lines.append(f'<div class="shep-muted" title="{r.e(group.get("why"))}">○ comparison '
                     'incomplete</div>')
    return "".join(lines)


def _exposure_cell(row: Mapping[str, Any]) -> str:
    exposure = row.get("exposure") or {}
    kind = (row.get("data") or {}).get("kind")
    if kind == "supplied":
        return (f'<details class="shep-mp-fold"><summary>R10 recorded</summary>'
                f'{mp.r10_html(exposure.get("r10"))}</details>')
    if kind == "kg_edges":
        if exposure.get("reading"):
            return r.e(exposure["reading"])
        return (f'<span class="shep-muted">○ Exposure not recorded: '
                f'{r.e(exposure.get("problem") or "no reason given")}</span>')
    return '<span class="shep-muted">○ Not compared (the split\'s allocation)</span>'


def _environment_cell(row: Mapping[str, Any]) -> str:
    environment = row.get("environment") or {}
    gpu = environment.get("gpu") if isinstance(environment.get("gpu"), Mapping) else None
    sub = (f'<div class="shep-sub">GPU {r.e(gpu.get("name") or "name not read")}</div>'
           if gpu else "")
    return f'<div>{r.e(environment.get("class") or "not reported")}</div>{sub}'


def _metric_heads(protocol: Any) -> str:
    columns = mp.RECORD_COLUMNS.get(protocol)
    if columns:
        return "".join(f'<th scope="col" class="shep-num">{r.e(metric_text(protocol, name))}'
                       '</th>' for name in columns)
    return '<th scope="col">Recorded metrics</th>'


def protocol_table_html(protocol: Any, rows: Sequence[Mapping[str, Any]],
                        scope: Optional[Mapping[str, Any]] = None) -> str:
    """One protocol's results as one table, in the order given (newest first,
    the service's), each with its label, model (its family unless `scope` is
    a family's), data, metrics, state, exposure and environment; every value
    escaped."""
    family = (scope or {}).get("scope") != FAMILY
    body = "".join(
        f'<tr><td class="shep-nowrap">{r.e(row_label(row))}</td>'
        f'<td>{_model_cell(row, family=family)}</td>'
        f'<td>{_data_cell(row)}</td>{_metrics_cells(protocol, row)}'
        f'<td>{_state_cell(row)}</td><td>{_exposure_cell(row)}</td>'
        f'<td>{_environment_cell(row)}</td></tr>' for row in rows)
    label = mode_label(protocol)
    note = mode_note(protocol)
    count = len(rows)
    return (f'<p class="shep-mp-p"><b>{r.e(label)} · {count} result{"s" if count != 1 else ""}'
            f'.</b>{" " + r.e(note) if note else ""}</p>'
            '<div class="shep-tbl-wrap shep-rs-wrap"><table class="shep-tbl shep-rs-tbl" '
            f'aria-label="{r.e("Results, " + label)}"><thead><tr>'
            '<th scope="col">Result</th><th scope="col">Model</th><th scope="col">Data</th>'
            f'{_metric_heads(protocol)}<th scope="col">State</th><th scope="col">Exposure</th>'
            f'<th scope="col">Environment</th></tr></thead><tbody>{body}</tbody></table></div>')


def _where_text(where: Any) -> str:
    if not isinstance(where, Mapping):
        return ""
    return "; ".join(f"{key.replace('_', ' ')} {value}" for key, value in where.items()
                     if value is not None and key != "source")


def unreadable_table_html(rows: Sequence[Mapping[str, Any]]) -> str:
    """The items that could not be read, each with what it is, where the
    server's lists hold it and its reason (the service's, redacted of server
    paths): amber with ▲, never counted as results, never hidden."""
    if not rows:
        return ""
    many = len(rows) != 1
    body = "".join(f'<tr><td>{r.e(row.get("item"))}</td><td>{r.e(_where_text(row.get("where")))}'
                   f'</td><td>{r.e(row.get("reason"))}</td></tr>' for row in rows)
    return (f'<p class="shep-mp-p shep-amber-t">{r.WARN_MARK} {len(rows)} item'
            f'{"s" if many else ""} could not be read, so {"they are" if many else "it is"} '
            'listed with the reason and never counted as a result (no scope or filter hides '
            'one).</p><div class="shep-tbl-wrap shep-rs-wrap"><table class="shep-tbl '
            'shep-rs-unread" aria-label="Items that could not be read"><thead><tr>'
            '<th scope="col">Item</th><th scope="col">Where</th><th scope="col">Reason</th>'
            f'</tr></thead><tbody>{body}</tbody></table></div>')


def tables_html(state: Mapping[str, Any], scope: Mapping[str, Any],
                shown: Mapping[str, Any]) -> str:
    """The segment's body: one table per protocol (in the order first shown,
    each newest first), then the unreadable items; or, with nothing to show,
    what will appear and the next step, in its own words per cause."""
    if state.get("refused"):
        return (f'<div class="shep-mp-ro" role="status"><div class="shep-reason">'
                f'{r.e(state["refused"])}</div></div>')
    if state.get("read_id") is None:
        return ('<p class="shep-mp-p shep-muted">Results are read when this segment is '
                'opened.</p>')
    rows = state.get("rows") or []
    listed = [rows[i] for i in shown.get("shown") or []]
    results = [row for row in listed if row.get("result") != service.UNREADABLE]
    unreadable = [row for row in listed if row.get("result") == service.UNREADABLE]
    parts = []
    empty = (f'<p class="shep-mp-p">No result of {r.e(scope_words(scope))} yet: a Test run '
             'that keeps a report, or a registered record, is listed here. '
             f'{r.e(CROSS_WORKSPACE_GAP)}</p>')
    if not results:
        if shown.get("in_scope"):
            parts.append(f'<p class="shep-mp-p">No result of {r.e(scope_words(scope))} matches '
                         f'these filters; choose All to see its {shown["in_scope"]} '
                         'results.</p>')
        else:
            parts.append(empty)
    elif shown.get("in_scope", 0) == shown.get("unplaced_in_scope", 0):
        # Only results found in no listed workspace (the status line says
        # so): the scope itself holds none yet.
        parts.append(empty)
    groups: Dict[Any, List[Mapping[str, Any]]] = {}
    for row in results:
        groups.setdefault(row.get("protocol"), []).append(row)
    parts.extend(protocol_table_html(protocol, group, scope)
                 for protocol, group in groups.items())
    parts.append(unreadable_table_html(unreadable))
    return "".join(parts)


def status_html(state: Mapping[str, Any], scope: Mapping[str, Any],
                shown: Mapping[str, Any]) -> str:
    """One line over the tables: the scope, when the rows were read and how
    many are in scope and shown; under it, a note when the family scope fell
    back, and the results found in no listed workspace (`unplaced_text`)."""
    if state.get("refused") or state.get("read_id") is None:
        return ""
    rows = state.get("rows") or []
    listed = [rows[i] for i in shown.get("shown") or []]
    n_shown = sum(1 for row in listed if row.get("result") != service.UNREADABLE)
    notes = [scope.get("note"), unplaced_text(scope, shown)]
    note = "".join(f'<div class="shep-sub">{r.e(text)}</div>' for text in notes if text)
    return (f'<p class="shep-mp-p">Results of {r.e(scope_words(scope))}: {n_shown} shown of '
            f'{shown.get("in_scope", 0)} <span class="shep-muted">(read {r.e(state.get("read_at"))}'
            f')</span>{help_button("models-results", "Results")}</p>{note}')


def unplaced_text(scope: Mapping[str, Any], shown: Mapping[str, Any]) -> Optional[str]:
    """What the results found in no listed workspace (`unplaced`) are, and
    where they are listed: among this workspace's, or, in the family scope,
    under This workspace (no family is known for them). None when there are
    none."""
    count = shown.get("unplaced") or 0
    if not count:
        return None
    one = count == 1
    what = (f"{count} result{'' if one else 's'} found in no listed workspace (the weights "
            f"{'it' if one else 'they'} measured are not listed now, and no listed ledger holds "
            f"{'it' if one else 'them'})")
    if shown.get("unplaced_in_scope"):
        return f"Listed here too: {what}."
    return (f"{what[:1].upper()}{what[1:]} {'has' if one else 'have'} no known family: "
            f"This workspace lists {'it' if one else 'them'}.")


# ---------------------------------------------------------------------------
# Compare (pure HTML)
# ---------------------------------------------------------------------------
#: A value Compare shows for a metric row of another protocol's result.
OTHER_PROTOCOL = object()
#: A value Compare shows where a result's records do not hold the row's field
#: at all ("not recorded"), apart from a value recorded as null ("none") and a
#: field without a source ("Unknown" with why).
ABSENT = object()
#: Compare's groups of rows, in the order shown, each once.
COMPARE_GROUPS = ("Result", "Model", "Training settings", "Data", "Measurement", "Environment",
                  "Exposure", "Metrics", "References")
#: Measurement settings Compare does not show as rows of their own: the mode
#: is the result's protocol, which heads its column, and the score semantics
#: has its own row (one fact once).
_SHOWN_ELSEWHERE = ("mode", "score_semantics")
#: How a result's condition group compares, in words (a description: it
#: decides nothing).
_GROUP_WORDS = {"uncontested": "agree", "contested": "differ",
                "cannot_be_checked": "comparison incomplete"}


def _flat(prefix: str, value: Any) -> List[Tuple[str, Any]]:
    """A mapping's entries as rows of their own (one level, in its order); any
    other value as one row."""
    if isinstance(value, Mapping) and value:
        return [(f"{prefix} · {key}", item) for key, item in value.items()]
    return [(prefix, value)]


def compare_fields(row: Mapping[str, Any]) -> List[Tuple[str, str, Any, Optional[str]]]:
    """A result's comparison rows, ``(group, label, value, unknown reason)``,
    in the order Compare shows them (`COMPARE_GROUPS`): the result, the model
    (its operation, parent, family, encoder, relation to the reference
    recipe, every value of its spec, its capabilities), the training settings
    (rules, each recorded learning setting and each effective one, then each
    part of its weight transfer and of its loss identity as its record holds
    them: how its stage's weights began and its loss's terms), the data
    (identity and role), the measurement (each setting its record holds as a
    value, and each it does not, Unknown with why), the environment, the
    exposure and the metrics (each named with its protocol). The protocol is
    not a row: it heads the result's column. Values are the row's own."""
    unknown = row.get("unknown") or {}
    model = row.get("model") or {}
    data = row.get("data") or {}
    measurement = row.get("measurement") or {}
    environment = row.get("environment") or {}
    exposure = row.get("exposure") or {}

    def why(*names: str) -> Optional[str]:
        return next((unknown[name] for name in names if name in unknown), None)

    out: List[Tuple[str, str, Any, Optional[str]]] = [
        ("Result", "Run", row.get("run_id"), None),
        ("Result", "Report", row.get("report_sha256"), None),
        ("Result", "Report state", row.get("report"), why("report")),
        ("Result", "Registration", row.get("registered"), None),
        ("Result", "Report bytes elsewhere", row.get("conflict"), None),
        ("Result", "Runs under these conditions",
         _GROUP_WORDS.get((row.get("group") or {}).get("state")), None),
        ("Model", "Model", model.get("checkpoint_sha256"), None),
    ]
    for name, label in (("learning_stage", "Learning stage"), ("task", "Task"),
                        ("operation", "Operation"), ("parent_sha256", "Parent"),
                        ("parent_run", "Parent run"), ("family_id", "Family"),
                        ("encoder", "Encoder"), ("reference", "Reference recipe"),
                        ("differences", "Differs from the reference in"),
                        ("capabilities", "Capabilities")):
        out.append(("Model", label, model.get(name), why(f"model.{name}")))
    for label, value in _flat("Spec", model.get("spec")):
        out.append(("Model", label, value, why("model.spec")))
    out.append(("Training settings", "Training rules", model.get("training_rules"),
                why("model.training_rules")))
    for prefix, name in (("Learning setting", "learning_settings"),
                         ("Effective setting", "effective_settings"),
                         ("Weight transfer", "weight_transfer"), ("Loss", "loss_identity")):
        for label, value in _flat(prefix, model.get(name)):
            out.append(("Training settings", label, value, why(f"model.{name}")))
    out += [("Data", "Data kind", data.get("kind"), None),
            ("Data", "Data label", data.get("label"), None),
            ("Data", "Role", data.get("role"), None),
            ("Data", "Data version", data.get("version_sha256"), None),
            ("Data", "Dataset origin", data.get("dataset_origin"), None)]
    for label, value in _flat("Data identity", data.get("identity")):
        out.append(("Data", label, value, None))
    out.append(("Data", "Data counts", data.get("counts"), None))
    out.append(("Measurement", "Score semantics", row.get("semantics"), None))
    for name, value in measurement.items():
        if name not in _SHOWN_ELSEWHERE:
            out.append(("Measurement", name.replace("_", " ").capitalize(), value, None))
    # A setting its record does not hold as a value (the service's
    # ``unknown``, "measurement.<block>.<field>"): Unknown, with why.
    for path, reason in unknown.items():
        block, _dot, name = path.partition(".")[2].partition(".")
        if path.startswith("measurement.") and name and name not in _SHOWN_ELSEWHERE \
                and block in ("scoring", "regime"):
            out.append(("Measurement", name.replace("_", " ").capitalize(), None, reason))
    out.append(("Measurement", "Run counts", row.get("run_counts"), why("run_counts")))
    out += [("Environment", "Environment class", environment.get("class"), None),
            ("Environment", "GPU", environment.get("gpu"), None),
            ("Environment", "Native build",
             (native_identity_text(environment.get("native_identity"))
              if environment.get("native_identity") is not None else None),
             why("environment.native_identity"))]
    out += [("Exposure", "Exposure", exposure.get("reading") or exposure.get("state"), None),
            ("Exposure", "Exposure definition", exposure.get("definition_version"), None),
            ("Exposure", "Measured edges", exposure.get("measured_edges"), None),
            ("Exposure", "In recorded graph use", exposure.get("in_recorded_graph_use"), None),
            ("Exposure", "Exposure problem", exposure.get("problem"), None)]
    r10 = exposure.get("r10")
    if isinstance(r10, Mapping):
        for scope, labels in R10_LABEL_TEXT.items():
            part = r10.get(scope) if isinstance(r10.get(scope), Mapping) else {}
            evidence = (part.get("evidence") or {}).get("state")
            for label, words in labels.items():
                out.append(("Exposure", f"R10 · {words['name']}",
                            r.r10_label_text(label, words["name"], part.get(label), evidence),
                            None))
    words = protocol_words(row.get("protocol"))
    for metric in row.get("metrics") or []:
        name = metric.get("name")
        # The value as recorded (a number unrounded), or its status when it
        # could not be computed; its unit and denominator as rows of their
        # own, where its record holds them.
        computed = metric.get("status") in (None, "recorded", "computed")
        out.append(("Metrics", f"{name} ({words})",
                    metric.get("value") if computed else metric.get("status"), None))
        for part in ("unit", "denominator"):
            if metric.get(part) is not None:
                out.append(("Metrics", f"{name} {part} ({words})", metric[part], None))
    out.append(("References", "Content references", row.get("references"), why("references")))
    return out


def _compare_value(value: Any) -> str:
    """A value's comparable text: its JSON, keys sorted (a number as
    recorded), so two values are equal exactly when they are recorded alike."""
    if value is OTHER_PROTOCOL:
        return "\x00other"
    if value is ABSENT:
        return "\x00absent"
    return json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)


def _cell_html(value: Any, why: Optional[str], *, full: bool) -> str:
    """One Compare cell: "another protocol" on a metric row of another
    protocol's result; "not recorded" for a field the result's records do not
    hold at all; "Unknown" with why for a field without a source; "none" for
    a value recorded as null; a digest by its 12 hex digits (all of them when
    its column's 12 do not tell it apart), the full value as its title; an
    object as compact JSON; a number unrounded; text as recorded."""
    if value is OTHER_PROTOCOL:
        return '<td class="shep-muted">another protocol</td>'
    if value is ABSENT:
        return '<td class="shep-muted">○ not recorded</td>'
    if value is None:
        if why:
            return (f'<td class="shep-muted" title="{r.e(why)}">○ Unknown: '
                    f'{r.e(r.short_reason(why, 120))}</td>')
        return '<td class="shep-muted">none</td>'
    if r.is_digest(value):
        shown = value if full else r.short_hex(value)
        return f'<td><code class="shep-fpv" title="{r.e(value)}">{r.e(shown)}</code></td>'
    if isinstance(value, bool) or isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str,
                          separators=(", ", ": "))
        return f"<td><code>{r.e(text)}</code></td>"
    if isinstance(value, (int, float)):
        return f'<td class="shep-num">{r.e(recorded_value_text(value))}</td>'
    return f"<td>{r.e(value)}</td>"


def compare_html(rows: Sequence[Mapping[str, Any]], differences_only: bool, *,
                 asked: int = 0) -> str:
    """The chosen results side by side (at most `COMPARE_CAP`, a cap of this
    view only), one column each, one row per field (`compare_fields`), the
    rows grouped (`COMPARE_GROUPS`, each group once; within one, in the order
    first found); with `differences_only`, only the rows whose values are not
    recorded alike in every column. Values as recorded: nothing is computed
    from two of them. A metric row is one protocol's; a result of another
    protocol reads "another protocol" there, and one of the same protocol
    that does not record that metric reads "not recorded", as does any field
    a result's records do not hold. `asked` is how many were chosen (more
    than the cap: the first five are shown, saying so)."""
    if not rows:
        return ('<p class="shep-mp-p shep-muted">Choose results above to compare them side by '
                f'side ({COMPARE_CAP} at a time).</p>')
    columns = list(rows[:COMPARE_CAP])
    fields = [compare_fields(row) for row in columns]
    order: List[Tuple[str, str]] = []
    values: Dict[Tuple[str, str], List[Any]] = {}
    reasons: Dict[Tuple[str, str], List[Optional[str]]] = {}
    for index, found in enumerate(fields):
        protocol = columns[index].get("protocol")
        for group, label, value, why in found:
            key = (group, label)
            if key not in values:
                order.append(key)
                # A metric row is its protocol's (its label names it): a
                # result of another protocol reads "another protocol", one of
                # its protocol without the metric "not recorded".
                values[key] = [OTHER_PROTOCOL if group == "Metrics"
                               and column.get("protocol") != protocol else ABSENT
                               for column in columns]
                reasons[key] = [None] * len(columns)
            values[key][index] = value
            reasons[key][index] = why
    # Each group once, in its place; within one, the order first found.
    rank = {group: place for place, group in enumerate(COMPARE_GROUPS)}
    order.sort(key=lambda key: rank.get(key[0], len(rank)))
    many = len(columns) > 1
    body, hidden, group_shown = [], 0, None
    for key in order:
        texts = {_compare_value(v) + "\x01" + str(w) for v, w in zip(values[key], reasons[key])}
        if differences_only and many and len(texts) == 1:
            hidden += 1
            continue
        group, label = key
        if group != group_shown:
            body.append(f'<tr class="shep-rs-grp"><th scope="colgroup" '
                        f'colspan="{len(columns) + 1}">{r.e(group)}</th></tr>')
            group_shown = group
        shorts = [r.short_hex(v) for v in values[key] if r.is_digest(v)]
        full = len(set(shorts)) < len({v for v in values[key] if r.is_digest(v)})
        cells = "".join(_cell_html(v, w, full=full) for v, w in zip(values[key], reasons[key]))
        body.append(f'<tr><th scope="row">{r.e(label)}</th>{cells}</tr>')
    heads = "".join(f'<th scope="col">{r.e(row_label(row))}<div class="shep-sub">'
                    f'{r.e(protocol_words(row.get("protocol")))}</div></th>' for row in columns)
    notes = []
    if asked > COMPARE_CAP:
        notes.append(f"{asked} were chosen; Compare shows {COMPARE_CAP} at a time, the first "
                     f"{COMPARE_CAP} chosen (Export takes them all).")
    if differences_only and many:
        notes.append(f"Differences only: {hidden} rows recorded alike in every column are "
                     "hidden." if hidden else "Differences only: every row differs.")
    elif differences_only:
        notes.append("One result: every row is shown; choose another to see differences.")
    notes.append("Values as recorded, side by side; no difference is computed, and each "
                 "metric row is one protocol's.")
    note_html = "".join(f'<p class="shep-mp-p shep-sub">{r.e(n)}</p>' for n in notes)
    if not body:
        body.append(f'<tr><td colspan="{len(columns) + 1}" class="shep-muted">Every row is '
                    'recorded alike in every column.</td></tr>')
    return (f'{note_html}<div class="shep-tbl-wrap shep-rs-wrap"><table class="shep-tbl '
            f'shep-rs-cmp" aria-label="Compare results"><thead><tr><th scope="col">Field</th>'
            f'{heads}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')


# ---------------------------------------------------------------------------
# Reading and export
# ---------------------------------------------------------------------------
def read_results(environ: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """Every result of the server's own source (`results.query` over
    `deployment_sources`), as the session's state: the rows, when they were
    read and the read's identity (Compare's choices name it)."""
    rows = service.query(service.deployment_sources(environ))
    return {"read_id": str(time.time_ns()), "read_at": _utc(), "rows": rows, "refused": None,
            "shown": []}


def export_file(rows: Sequence[Mapping[str, Any]], fmt: str, name: str) -> str:
    """`rows` through the service's one serializer, CSV or JSON, written as a
    new file of the dashboard's download directory named by `name` (no server
    path) and the time (`exports.write_new`: never replacing a file, its
    bytes the serializer's text); the file's path."""
    text = service.to_json(rows) if fmt == "json" else service.to_csv(rows)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return str(write_new(f"results_{file_slug(name, 'results')}_{stamp}", fmt, text))


# ---------------------------------------------------------------------------
# The handlers (each D8 first)
# ---------------------------------------------------------------------------
def _refused_state(refusal: str) -> Dict[str, Any]:
    return {**EMPTY_STATE, "refused": refusal}


def _filters(*values: Any) -> Dict[str, Any]:
    return {name: value for (name, _label), value in zip(FILTERS, values)}


def _outputs(state: Mapping[str, Any], listing: Any, selected: Any, scope_value: Any,
             filters: Mapping[str, Any], chosen: Any, differences: Any) -> Tuple[Any, ...]:
    """What every listing handler writes: the state (with the rows shown),
    each filter, the scope, the status line, the tables, Compare's choices
    and Compare."""
    if state.get("refused") or state.get("read_id") is None:
        scope = scope_of(listing if isinstance(listing, Mapping) else {}, selected, scope_value)
        empty = gr.update(choices=[("All", ALL)], value=ALL)
        return (dict(state), *(empty for _f in FILTERS), status_html(state, scope, {}),
                tables_html(state, scope, {}), gr.update(choices=[], value=[]),
                compare_html([], True))
    scope = scope_of(listing, selected, scope_value)
    shown = shown_results(state.get("rows") or [], scope, filters)
    new_state = {**state, "shown": shown["shown"]}
    choices = compare_choices(new_state)
    offered = {value for _label, value in choices}
    kept = [key for key in (chosen or []) if key in offered]
    rows = chosen_rows(new_state, kept)
    return (new_state,
            *(gr.update(choices=shown["choices"][name], value=shown["chosen"][name])
              for name, _label in FILTERS),
            status_html(new_state, scope, shown), tables_html(new_state, scope, shown),
            gr.update(choices=choices, value=kept),
            compare_html(rows, bool(differences), asked=len(kept)))


def _on_results_open(request: gr.Request, seg, state, listing, selected, scope, protocol,
                     data_kind, role, environment, chosen, differences):
    """The segment chosen: D8 first (the read opens checkpoints); refused,
    nothing is read or listed and the reason is shown. The results are read
    once when the segment is first opened (↻ reads them again), then shown."""
    access = operator_access(request)
    if not access.allowed:
        return _outputs(_refused_state(access.refusal), listing, selected, scope,
                        {}, [], differences)
    if seg != RESULTS:
        return tuple(gr.update() for _ in range(len(FILTERS) + 5))
    if not isinstance(state, Mapping) or state.get("read_id") is None or state.get("refused"):
        state = read_results()
    return _outputs(state, listing, selected, scope,
                    _filters(protocol, data_kind, role, environment), chosen, differences)


def _on_results_read(request: gr.Request, listing, selected, scope, protocol, data_kind, role,
                     environment, differences):
    """↻ Read again: D8 first, then every result read again; Compare's
    choices from an earlier read are cleared (they named that read's rows)."""
    access = operator_access(request)
    if not access.allowed:
        return _outputs(_refused_state(access.refusal), listing, selected, scope,
                        {}, [], differences)
    return _outputs(read_results(), listing, selected, scope,
                    _filters(protocol, data_kind, role, environment), [], differences)


def _on_results_shown(request: gr.Request, seg, state, listing, selected, scope, protocol,
                      data_kind, role, environment, chosen, differences):
    """The scope, a filter, the selection or the listing changed: D8 first,
    then the rows already read shown again (nothing is read)."""
    access = operator_access(request)
    if not access.allowed:
        return _outputs(_refused_state(access.refusal), listing, selected, scope,
                        {}, [], differences)
    if seg != RESULTS or not isinstance(state, Mapping):
        return tuple(gr.update() for _ in range(len(FILTERS) + 5))
    return _outputs(state, listing, selected, scope,
                    _filters(protocol, data_kind, role, environment), chosen, differences)


def _on_results_compare(request: gr.Request, state, chosen, differences):
    """Compare: D8 first; the chosen results side by side (`compare_html`)."""
    access = operator_access(request)
    if not access.allowed:
        return (f'<div class="shep-reason">{r.e(access.refusal)}</div>')
    state = state if isinstance(state, Mapping) else EMPTY_STATE
    rows = chosen_rows(state, chosen)
    return compare_html(rows, bool(differences), asked=len(rows))


def _on_results_export(request: gr.Request, state, listing, selected, scope, chosen, which, fmt):
    """Export: D8 first; then the chosen results, or every row shown (the
    unreadable items among them), uncapped, as CSV or JSON through the
    service's one serializer; the file to download and what it holds."""
    access = operator_access(request)
    if not access.allowed:
        return (gr.update(value=None, visible=False),
                f'<div class="shep-reason">{r.e(access.refusal)}</div>')
    state = state if isinstance(state, Mapping) else EMPTY_STATE
    if state.get("read_id") is None:
        return (gr.update(value=None, visible=False),
                '<p class="shep-mp-p shep-muted">Nothing has been read to export yet.</p>')
    fmt = fmt if fmt in ("csv", "json") else "csv"
    all_rows = state.get("rows") or []
    if which == SELECTED:
        rows = chosen_rows(state, chosen)
        if not rows:
            return (gr.update(value=None, visible=False),
                    '<p class="shep-mp-p shep-muted">Choose results above to export them, or '
                    'export all shown.</p>')
    else:
        rows = [all_rows[i] for i in state.get("shown") or []]
    named = scope_of(listing if isinstance(listing, Mapping) else {}, selected, scope)
    name = (f"family-{named['family']}" if named.get("scope") == FAMILY
            else f"workspace-{named.get('workspace')}") + ("-selected" if which == SELECTED
                                                           else "")
    path = export_file(rows, fmt, name)
    results = sum(1 for row in rows if row.get("result") != service.UNREADABLE)
    unreadable = len(rows) - results
    said = (f"Exported {results} result{'s' if results != 1 else ''}"
            + (f" and {unreadable} unreadable item{'s' if unreadable != 1 else ''}"
               if unreadable else "")
            + f" as {fmt.upper()}, one row per result and metric (read {state.get('read_at')}).")
    return (gr.update(value=path, visible=True), f'<p class="shep-mp-p">{r.e(said)}</p>')


# ---------------------------------------------------------------------------
# The components (built inside Models' detail)
# ---------------------------------------------------------------------------
def create_results_segment() -> Dict[str, Any]:
    """The segment's components, in the detail's body (hidden until the
    segment is chosen): scope, ↻, filters, the status line and tables,
    Compare and Export. Returns them by name for `wire_results_segment`."""
    parts: Dict[str, Any] = {"state": gr.State(dict(EMPTY_STATE))}
    with gr.Column(visible=False, elem_id="shep-rs", elem_classes=["shep-rs"]) as column:
        parts["column"] = column
        with gr.Row(elem_classes=["shep-rs-row"]):
            parts["scope"] = gr.Radio(choices=SCOPE_CHOICES, value=FAMILY, label="Scope",
                                      show_label=False, elem_id="shep-rs-scope",
                                      elem_classes=["shep-seg"], container=False)
            parts["again"] = gr.Button("↻ Read again", size="sm", elem_id="shep-rs-again")
        with gr.Row(elem_classes=["shep-rs-row"]):
            for name, label in FILTERS:
                parts[name] = gr.Dropdown(choices=[("All", ALL)], value=ALL, label=label,
                                          interactive=True, elem_id=f"shep-rs-{name}")
        parts["status"] = plain_html("", elem_id="shep-rs-status")
        parts["tables"] = plain_html(tables_html(EMPTY_STATE, {}, {}), elem_id="shep-rs-tables")
        with gr.Row(elem_classes=["shep-rs-row"]):
            # No cap on the choice: Compare shows the first five and says
            # so, and Export › Selected writes every one chosen (O6).
            parts["chosen"] = gr.Dropdown(choices=[], value=[], multiselect=True,
                                          label=f"Chosen results (Compare shows {COMPARE_CAP})",
                                          interactive=True, elem_id="shep-rs-chosen")
            parts["differences"] = gr.Checkbox(value=True, label="Differences only",
                                               elem_id="shep-rs-differences")
        parts["compare"] = plain_html(compare_html([], True), elem_id="shep-rs-compare")
        with gr.Row(elem_classes=["shep-rs-row"]):
            parts["which"] = gr.Radio(choices=EXPORT_ROWS, value=SHOWN, label="Export",
                                      elem_id="shep-rs-which")
            parts["format"] = gr.Radio(choices=EXPORT_FORMATS, value="csv", label="Format",
                                       elem_id="shep-rs-format")
            parts["export"] = gr.Button("Export", size="sm", elem_id="shep-rs-export")
        parts["file"] = gr.File(label="Export file", interactive=False, visible=False,
                                elem_id="shep-rs-file")
        parts["export_note"] = plain_html("", elem_id="shep-rs-export-note")
    return parts


def wire_results_segment(parts: Mapping[str, Any], *, seg: Any, listing: Any,
                         selected: Any) -> None:
    """The segment's events, each through a gated handler (D8 first)."""
    filters = [parts[name] for name, _label in FILTERS]
    outputs = [parts["state"], *filters, parts["status"], parts["tables"], parts["chosen"],
               parts["compare"]]
    shown_inputs = [seg, parts["state"], listing, selected, parts["scope"], *filters,
                    parts["chosen"], parts["differences"]]
    seg.input(_on_results_open, inputs=shown_inputs, outputs=outputs, show_progress="minimal")
    parts["again"].click(_on_results_read,
                         inputs=[listing, selected, parts["scope"], *filters,
                                 parts["differences"]],
                         outputs=outputs, show_progress="minimal")
    gr.on([parts["scope"].input, *(f.input for f in filters), selected.change, listing.change],
          _on_results_shown, inputs=shown_inputs, outputs=outputs, show_progress="hidden",
          trigger_mode="always_last")
    gr.on([parts["chosen"].input, parts["differences"].input], _on_results_compare,
          inputs=[parts["state"], parts["chosen"], parts["differences"]],
          outputs=[parts["compare"]], show_progress="hidden", trigger_mode="always_last")
    parts["export"].click(_on_results_export,
                          inputs=[parts["state"], listing, selected, parts["scope"],
                                  parts["chosen"], parts["which"], parts["format"]],
                          outputs=[parts["file"], parts["export_note"]],
                          show_progress="minimal")


__all__ = [
    "ALL",
    "COMPARE_CAP",
    "COMPARE_GROUPS",
    "CROSS_WORKSPACE_GAP",
    "EMPTY_STATE",
    "FAMILY",
    "RESULTS",
    "SEGMENT_HELP",
    "WORKSPACE",
    "chosen_rows",
    "compare_choices",
    "compare_fields",
    "compare_html",
    "create_results_segment",
    "export_file",
    "filter_choices",
    "protocol_table_html",
    "read_results",
    "row_label",
    "scope_of",
    "shown_results",
    "status_html",
    "tables_html",
    "unplaced",
    "unplaced_text",
    "unreadable_table_html",
    "wire_results_segment",
]
