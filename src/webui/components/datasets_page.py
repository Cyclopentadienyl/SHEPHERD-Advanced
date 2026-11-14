"""
Research › Prepare › Datasets: the server's workspaces, read-only (M5-2, UI-A).
================================================================================
`docs/working/PLAN_TEST_RESULTS.md` §6, "M5's boundaries": a read-only Datasets
page over the server's workspace list. The same list Training, Test and Models
choose from (`server_lists.list_workspaces`), drawn with the shared dataset
renderer (`renderers.dataset_identity_html` and its pieces).

- **Columns:** name, built, dataset fingerprint (12 hex in bold monospace, the
  full value on Copy), origin, record, sources, train / val samples, in use. Below
  1800 px the origin and the record fold into the name cell (CSS only).
- **Markers:** "serving reads this workspace" (accent) from the status's
  ``current_data_dir``; "same dataset as served" for another listed workspace
  with the same **full** fingerprint; "in your Training run bar" (neutral,
  outlined) from this browser session's run-bar selection
  (`PageContext.training_workspace`); "not trainable: <why>" rows stay listed.
- **Details** of the selected row: the full fingerprint with Copy, what the
  manifest binds with short digests, the seeds and split counts, the sources and
  the build record, the path **relative to the server's workspace root**, its
  **registered data versions** (`versions_html`), and how it compares with the
  other listed workspaces ("Same graph build as …, another cut: …"; "Same
  dataset as …"). The manifests and the registry are read once per refresh
  (`read_datasets`), not on each selection.
- **Data versions**: the registry's versions mapped against the workspace's
  graph, read by the registry's one listing (`server_lists.list_cohort_versions`)
  and compared by the one comparison Test and Diagnosis use
  (`workspaces.workspace_binding`, `binding_state`): each by its label, its kind
  and role, and the measurements its data admits (`protocols.inputs_refusal`,
  the data side of Test's measurement rule; the model chosen on Test decides
  the rest), the others' reasons one click away. Versions are imported where
  they always were (Test's "Import a new version…", `scripts/import_cohort.py`);
  this page adds no importer.
- **No absolute path, for any request**: the list is read through the
  dashboard's one workspace reader (`src/webui/components/workspaces.py`): a
  path under the root is cut to the part inside it, any other absolute path (a
  link's target outside the root) becomes "a path outside the workspaces
  root", and a refused root setting is said without its path.
- **"Use for training ›"** selects the workspace in this session's Training run
  bar and opens Training; it changes nothing on the server (Start is the gated
  act there). It is queued in the group of every writer of that workspace
  (`PageContext.training_workspace_writes`), after any render of Training's run
  bar asked before it.
- The one line on building datasets: they are built on the server with the CLI.

Nothing here is gated: it reads small JSON files (the manifest, the dataset
record, the build record, the registry's version manifests), never a
checkpoint, so a request D8 refuses sees the same page.

Module: src/webui/components/datasets_page.py
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional

import gradio as gr

from src.evaluation import protocols
from src.evaluation.caveats import DATA_KIND_WORDS, MEASUREMENT_NAMES, label_words
from src.webui.components import renderers as r
from src.webui.components.help import help_button
from src.webui.components.served_status import read_status, serving_workspace
from src.webui.components.workspaces import (
    OUTSIDE_PATH,
    ROOT_REFUSED,
    binding_state,
    manifest_facts,
    read_workspaces,
    redact,
    version_row,
    workspace_binding,
)
from src.webui.shell import page_elem_id, plain_html

logger = logging.getLogger(__name__)

#: The one line on building datasets (no building in the WebUI, BACKLOG 29):
#: the command that writes a trainable workspace (graph, samples, manifest and
#: dataset record), and the demo's.
BUILD_LINE = (
    "Datasets are built on the server with "
    "<code>scripts/build_knowledge_graph.py --generate-samples</code> (the demo with "
    "<code>scripts/setup_demo.py</code>)."
)
#: The markers' meaning, in plain words; the names they are read from only in
#: the trailing "Ref:" (the WebUI skill's §5).
FOOTNOTE = (
    "“In your Training run bar” is this browser tab’s selection; nothing on the server "
    "changes. “Serving reads this workspace” is the workspace the served model reads; "
    "another with the same full fingerprint reads “same dataset as served”. A build time "
    "appears once a build records one, outside the fingerprint. Ref: the pipeline status’s "
    "<code>current_data_dir</code>; <code>workspace_build.json</code>."
)
USE_NOTE = ("Selects it in your Training run bar (this browser tab) and opens Training. Nothing "
            "changes on the server; Start is the gated act.")


def use_note_html(data: Mapping[str, Any], name: Optional[str]) -> str:
    """The line beside "Use for training ›": what it does, or, when it is
    disabled, why, as visible text (M5-3: a not-trainable row disabled it with
    no reason beside it)."""
    row = next((x for x in data.get("rows") or [] if x["name"] == name), None)
    if row is None:
        return '<span class="shep-note">Select a dataset to use it for training.</span>'
    if not row.get("trainable"):
        why = row.get("not_trainable") or "not trainable"
        return (f'<span class="shep-note" title="{r.e(why)}">Not available: {r.e(row["name"])} is '
                f'not trainable — {r.e(r.not_trainable_short(row, 120))}.</span>')
    return f'<span class="shep-note">{USE_NOTE}</span>'

def read_datasets() -> Dict[str, Any]:
    """The page's data for one session, read once per refresh: the server's
    workspaces as the dashboard shows them (`workspaces.read_workspaces`: every
    reason without an absolute path, each manifest's state), each manifest's
    facts (what it binds, seeds, split counts), the serving workspace's name,
    and why none is listed. Each name resolves through one listing of the root."""
    listed = read_workspaces()
    root, directories = listed["root"], listed["directories"]
    rows = []
    for row in listed["rows"]:
        facts, problem = (manifest_facts(directories[row["name"]], root)
                          if row["manifest"] == "read" and row["name"] in directories
                          else (None, row["not_trainable"]))
        row["facts"], row["facts_problem"] = facts, problem
        rows.append(row)
    status, _problem = read_status()
    serving = (serving_workspace(status, rows, directories=directories)
               if status is not None else None)
    versions, registry_problem = _read_versions(root)
    for row in rows:
        row["versions"] = _bound_versions(row, versions, directories)
    return {"rows": rows, "serving": serving, "problem": listed["problem"],
            "status_read": status is not None, "versions": versions,
            "registry_problem": registry_problem}


def _read_versions(root: Any) -> tuple:
    """The registry's versions as plain data (`workspaces.version_row`, their
    manifests only, no report store), read by the registry's one listing
    (`server_lists.list_cohort_versions`, Test's), and why none is listed,
    without a path."""
    from src.api.services.server_lists import list_cohort_versions

    listing = list_cohort_versions()
    problem = redact(listing.problem, root) if listing.problem else None
    return [version_row(item) for item in listing.items], problem


def _bound_versions(row: Mapping[str, Any], versions: List[Mapping[str, Any]],
                    directories: Mapping[str, Any]) -> Dict[str, Any]:
    """Which registered versions were mapped against `row`'s graph, by the one
    comparison Test and Diagnosis use (`workspace_binding`, `binding_state`):
    the labels bound to it, how many were mapped against another graph or
    cannot be read, and why none could be checked (`problem`, its words
    without a path). A listed row whose directory was not reached (a name
    that resolves outside the root, or cannot be resolved) is not checked for
    the row's own reason, never Test's "no workspace is chosen": on this page
    the row is the chosen one."""
    if not versions:
        return {"bound": [], "another": 0, "unknown": 0, "problem": None}
    directory = directories.get(row["name"])
    if directory is None:
        why = str(row.get("not_trainable") or "the workspace was not reached").rstrip(". ")
        return {"bound": [], "another": 0, "unknown": 0, "problem": why}
    binding = workspace_binding(directory, row)
    if binding.get("problem"):
        return {"bound": [], "another": 0, "unknown": 0, "problem": binding["problem"]}
    states: Dict[str, List[str]] = {"bound": [], "another": [], "unknown": []}
    for version in versions:
        states[binding_state(version, binding)[0]].append(str(version.get("label")))
    return {"bound": states["bound"], "another": len(states["another"]),
            "unknown": len(states["unknown"]), "problem": None}


def _marks(row: Mapping[str, Any], data: Mapping[str, Any], run_bar: Optional[str]) -> str:
    marks = []
    serving_row = next((x for x in data.get("rows") or [] if x["name"] == data.get("serving")),
                       None)
    if row["name"] == data.get("serving"):
        marks.append('<span class="shep-mark shep-mark-serve">serving reads this workspace</span>')
    elif (serving_row is not None and r.is_digest(row.get("dataset_fingerprint"))
          and row.get("dataset_fingerprint") == serving_row.get("dataset_fingerprint")):
        marks.append('<span class="shep-mark shep-mark-serve-same">same dataset as served</span>')
    if run_bar and row["name"] == run_bar:
        marks.append('<span class="shep-mark shep-mark-outline">in your Training run bar</span>')
    if not row.get("trainable"):
        why = row.get("not_trainable") or "not trainable"
        marks.append(f'<span class="shep-mark shep-mark-grey" title="{r.e(why)}">'
                     f'not trainable: {r.e(r.not_trainable_short(row, 60))}</span>')
    return "".join(marks) or f'<span class="shep-muted">{r.NO_VALUE}</span>'


def _record_cell(row: Mapping[str, Any]) -> str:
    """The dataset record's state (`renderers.record_words`): verified, or its
    own missing or contradicted state, never one "not verified" for all."""
    kind, words = r.record_words(row)
    return r.status_mark(kind, words.removeprefix("record ").capitalize())


def _origin_cell(row: Mapping[str, Any]) -> str:
    """The record's origin, or why none is shown: "Not recorded" when the
    manifest binds no record, "Unknown" when the record was not read or names
    no origin this build knows (never "—")."""
    origin = r.origin_of(row)
    if origin == "synthetic":
        return r.chip("synthetic", "amber")
    if origin:
        return r.e(origin)
    record = row.get("dataset_record") or {}
    unrecorded = isinstance(record, Mapping) and record.get("binding") == "unrecorded"
    return r.status_mark("neutral", r.NOT_RECORDED if unrecorded else r.UNKNOWN)


def _fingerprint_missing(row: Mapping[str, Any]) -> str:
    """Why a workspace has no dataset fingerprint: its split manifest is not
    there ("Not recorded"), or it is there and was not read ("Unknown", its
    reason in the title): never "—"."""
    if row.get("manifest") == "absent":
        return r.status_mark("neutral", r.NOT_RECORDED)
    why = row.get("not_trainable") or "the split manifest was not read"
    return f'<span title="{r.e(why)}">{r.status_mark("neutral", r.UNKNOWN)}</span>'


#: The selected row's mark: a plain table's rows take `aria-current`, never
#: `aria-selected` (the WebUI skill's §8).
CURRENT = ' aria-current="true"'


def table_html(data: Mapping[str, Any], run_bar: Optional[str], selected: Optional[str]) -> str:
    """The workspaces table; each row selects its details."""
    rows = data.get("rows") or []
    if not rows:
        why = data.get("problem") or "No workspace is listed."
        return f'<div class="shep-reason">{r.e(why)}</div>'
    digits = r.distinguishing_digits(x.get("dataset_fingerprint") for x in rows)
    body = []
    for row in rows:
        name = row["name"]
        fingerprint = row.get("dataset_fingerprint")
        built = r.built_text(row.get("build"))
        when = built.replace("built ", "", 1) if built else ""
        built_cell = (f'<span class="shep-nowrap">{r.e(when[:-4])}</span> UTC'
                      if when.endswith(" UTC") else r.e(when) if when
                      else '<span class="shep-muted">not recorded</span>')
        sources = r.sources_text(row.get("declared_releases"))
        fp_cell = (r.fingerprint_value(fingerprint, what=f"dataset fingerprint of {name}",
                                       digits=digits)
                   if r.is_digest(fingerprint) else _fingerprint_missing(row))
        on = name == selected
        body.append(
            # A plain table's selected row is `aria-current` (the skill's §8).
            f'<tr class="{"shep-on" if on else ""}" data-key="{r.e(name)}" tabindex="0"'
            f'{CURRENT if on else ""}>'
            f'<td class="shep-ds-name"><div class="shep-w600">{r.e(name)}</div>'
            f'<div class="shep-ds-narrow shep-sub">{_origin_cell(row)} · {_record_cell(row)}</div></td>'
            f'<td>{built_cell}</td>'
            f'<td class="shep-nowrap">{fp_cell}</td>'
            f'<td class="shep-ds-wide">{_origin_cell(row)}</td>'
            f'<td class="shep-ds-wide">{_record_cell(row)}</td>'
            f'<td class="shep-ds-src">{r.e(sources) if sources else "<span class=shep-muted>—</span>"}</td>'
            f'<td class="shep-num">{r.count_text(row.get("train_samples"))}'
            f'<div class="shep-sub">{r.count_text(row.get("val_samples"))}</div></td>'
            f'<td class="shep-ds-use">{_marks(row, data, run_bar)}</td></tr>')
    return (
        '<div class="shep-tbl-wrap"><table class="shep-tbl shep-dstbl" aria-label="Workspaces">'
        '<thead><tr><th scope="col">Name</th><th scope="col">Built</th>'
        '<th scope="col">Dataset fingerprint</th><th scope="col" class="shep-ds-wide">Origin</th>'
        '<th scope="col" class="shep-ds-wide">Record</th><th scope="col">Sources</th>'
        '<th scope="col" class="shep-num">Train / val samples</th><th scope="col">In use</th>'
        f'</tr></thead><tbody>{"".join(body)}</tbody></table></div>')


def header_html(data: Mapping[str, Any]) -> str:
    """The panel's title with its count, in plain words (never "read-only",
    which names the role D8 refuses), and its one-line description."""
    count = len(data.get("rows") or [])
    return (f'<div class="shep-panel-hd"><h2>Datasets</h2><span class="shep-panel-right">'
            f'{count} workspace{"" if count == 1 else "s"} on the server</span></div>'
            '<p class="shep-intro">The same list Training, Test and Models choose from. Dataset '
            'fingerprint: the SHA-256 of each workspace’s split manifest '
            '(<code>split_manifest.json</code>)'
            f'{help_button("dataset-fingerprint", "Dataset fingerprint")}</p>')


#: The bound files two workspaces share when they come from one graph build.
_GRAPH_BUILD = ("kg", "node_features", "edge_indices", "num_nodes", "record")


def _names(names: List[str], limit: int = 3) -> str:
    shown = ", ".join(f"<b>{r.e(n)}</b>" for n in names[:limit])
    return shown + (f" and {len(names) - limit} more" if len(names) > limit else "")


def comparison_html(row: Mapping[str, Any], rows: List[Mapping[str, Any]]) -> str:
    """How the selected workspace compares with the other listed ones, on the
    full digests its manifest binds: "Same dataset as …" (the same full
    fingerprint), "Same graph build as … (kg, tensors and record identical),
    another cut: … differ"."""
    fingerprint, facts = row.get("dataset_fingerprint"), row.get("facts")
    if not r.is_digest(fingerprint) or not facts:
        return ""
    bound = dict(facts["bound"])
    same_dataset, cuts = [], {}
    for other in rows:
        if other is row or not other.get("facts"):
            continue
        if other.get("dataset_fingerprint") == fingerprint:
            same_dataset.append(other["name"])
            continue
        theirs = dict(other["facts"]["bound"])
        if not all(r.is_digest(bound.get(k)) and bound.get(k) == theirs.get(k)
                   for k in _GRAPH_BUILD):
            continue
        differ = []
        if other["facts"].get("allocation_seed") != facts.get("allocation_seed"):
            differ.append("the allocation seed")
        if (other.get("train_samples"), other.get("val_samples")) != (
                row.get("train_samples"), row.get("val_samples")):
            differ.append("the sample counts")
        cuts.setdefault(tuple(differ), []).append(other["name"])
    parts = []
    if same_dataset:
        parts.append(f"Same dataset as {_names(same_dataset)}: the same full fingerprint, so "
                     "the same recorded build.")
    for differ, names in cuts.items():
        verb = "differs" if len(differ) == 1 else "differ"
        what = (" and ".join(differ) + f" {verb}, so the samples and the dataset fingerprints "
                "differ") if differ else "the samples differ, so the dataset fingerprints differ"
        parts.append(f"Same graph build as {_names(names)} (kg, tensors and record identical), "
                     f"another cut: {what}.")
    return "".join(f'<p class="shep-compare">{p}</p>' for p in parts)


def details_html(data: Mapping[str, Any], name: Optional[str]) -> str:
    """The selected workspace's details: its name, record and origin, the
    shared details body (`renderers.dataset_details_html`, as Training's
    popover shows it) and how it compares with the other listed workspaces."""
    row = next((x for x in data.get("rows") or [] if x["name"] == name), None)
    if row is None:
        return ('<div class="shep-panel-hd"><h2>Details</h2></div><div class="shep-panel-bd">'
                '<p class="shep-muted">Select a dataset to see what its manifest binds.</p></div>')
    head = (f'<div class="shep-panel-hd"><h2>{r.e(row["name"])}</h2>{r.record_chip(row)}'
            f'<span class="shep-muted">{r.e(r.origin_of(row) or "")}</span></div>')
    body = r.dataset_details_html(row, row.get("facts"), problem=row.get("facts_problem"))
    return (head + '<div class="shep-panel-bd">' + body + versions_html(data, row)
            + comparison_html(row, list(data.get("rows") or [])) + "</div>")


#: Where a data version is imported: the entries that always made one (no
#: importer on this page).
IMPORT_LINE = ("Import one on the Test page (“Import a new version…”) or on the server with "
               "<code>scripts/import_cohort.py</code>.")


def _kind_text(kind: Any) -> str:
    """A version's data kind as a label names it ("Supplied cohort version",
    "KG edge set", "Generated split"; `caveats.DATA_KIND_WORDS`), or Unknown
    for a kind this build does not read, beside its recorded name."""
    if kind in DATA_KIND_WORDS:
        return r.e(label_words(DATA_KIND_WORDS[kind]))
    return f'{r.status_mark("neutral", r.UNKNOWN)} {r.ident_html(kind)}'


def admitted_measurements(version: Mapping[str, Any]) -> tuple:
    """The measurements a version's data admits and, for each other one that
    measures its kind, why not: the data side of Test's one rule
    (`protocols.inputs_refusal`, which Start's reason and the run apply too),
    in definition order, each by its plain words (`caveats.MEASUREMENT_NAMES`).
    The model chosen on Test decides the rest (`protocols.available`)."""
    kind, counts = version.get("kind"), version.get("counts")
    admitted, refused = [], []
    for name in protocols.names():
        reason = protocols.inputs_refusal(name, kind, counts)
        if reason is None:
            admitted.append(MEASUREMENT_NAMES.get(name, name))
        elif kind in protocols.PROTOCOLS[name].data_kinds:
            refused.append((MEASUREMENT_NAMES.get(name, name), reason))
    return admitted, refused


def _measurements_cell(version: Mapping[str, Any]) -> str:
    admitted, refused = admitted_measurements(version)
    cell = (r.e(", ".join(admitted)) if admitted
            else r.status_mark("neutral", "None admitted"))
    if refused:
        names = ", ".join(name for name, _why in refused)
        reasons = "".join(f"<p><b>{r.e(name)}</b>: {r.e(why)}</p>" for name, why in refused)
        cell += (f'<details class="shep-more"><summary>Not admitted: {r.e(names)} ›</summary>'
                 f"{reasons}</details>")
    return cell


def versions_html(data: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    """The workspace's registered data versions (the module docstring): a
    table of the versions mapped against its graph, each by its label, kind,
    role and the measurements its data admits; under it how many others the
    registry holds; or, in its own words, why none is shown (no registry, the
    workspace not checked, none mapped against it yet), with where a version
    is imported."""
    versions = {str(v.get("label")): v for v in data.get("versions") or []}
    mine = row.get("versions") or {}
    bound = [versions[label] for label in mine.get("bound") or [] if label in versions]
    title = (f'<div class="shep-ds-vt">Data versions <span class="shep-muted">'
             f'({len(bound)})</span></div>')
    problem = data.get("registry_problem")
    if problem:
        return (f'<div class="shep-ds-versions">{title}<div class="shep-reason">No registry is '
                f"listed: {r.e(problem)}</div></div>")
    if mine.get("problem"):
        return (f'<div class="shep-ds-versions">{title}<div class="shep-reason">Not checked '
                f"against the registry: {r.e(mine['problem'])}.</div></div>")
    others = []
    if mine.get("another"):
        n = mine["another"]
        others.append(f"{n} version{'' if n == 1 else 's'} mapped against another graph")
    if mine.get("unknown"):
        n = mine["unknown"]
        others.append(f"{n} that cannot be read")
    tail = (f'<p class="shep-sub">Not listed here: {r.e("; ".join(others))}.</p>'
            if others else "")
    if not bound:
        return (f'<div class="shep-ds-versions">{title}<div class="shep-reason"><span>No data '
                f"version is mapped against this workspace’s graph yet. {IMPORT_LINE}</span>"
                f"</div>{tail}</div>")
    body = "".join(
        f'<tr><td class="shep-w600">{r.e(version.get("label"))}</td>'
        f'<td>{_kind_text(version.get("kind"))}</td>'
        f'<td>{r.ident_html(version.get("role"))}</td>'
        f"<td>{_measurements_cell(version)}</td></tr>" for version in bound)
    table = ('<div class="shep-tbl-wrap"><table class="shep-tbl shep-dsvtbl" '
             'aria-label="Data versions"><thead><tr><th scope="col">Version</th>'
             '<th scope="col">Kind</th><th scope="col">Role</th>'
             f'<th scope="col">Measurements</th></tr></thead><tbody>{body}</tbody></table></div>')
    lead = ('<p class="shep-sub">In the server’s registry, mapped against this workspace’s '
            "graph: what its data admits; the model chosen on Test must take it too.</p>")
    return f'<div class="shep-ds-versions">{title}{lead}{table}{tail}</div>'


#: Row clicks (and Enter on a focused row) become the table's click event.
_ROWS_JS = r"""
element.addEventListener('click', function (e) {
  if (e.target.closest && e.target.closest('button')) { return; }
  var row = e.target.closest ? e.target.closest('tr[data-key]') : null;
  if (row) { trigger('click', {key: row.getAttribute('data-key')}); }
});
element.addEventListener('keydown', function (e) {
  var row = e.target.closest ? e.target.closest('tr[data-key]') : null;
  if (row && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); trigger('click', {key: row.getAttribute('data-key')}); }
});
"""


def create_datasets_page(ctx: Any) -> None:
    """Build Research › Prepare › Datasets (the page contract: `src/webui/shell.py`)."""
    data_state = gr.State({"rows": [], "serving": None, "problem": None})
    selected = gr.State(None)
    with gr.Row(elem_classes=["shep-page", "shep-datasets"], equal_height=False):
        with gr.Column(elem_classes=["shep-panel", "shep-ds-main"], scale=17):
            header = plain_html(header_html({"rows": []}), elem_id="shep-ds-header")
            table = plain_html('<div class="shep-muted">Reading the server’s workspace list…</div>',
                               elem_id="shep-ds-table", js_on_load=_ROWS_JS)
            # One paragraph: the build line's words and code chips are one flex
            # item of the reason row (M5-3: each was its own column at 1536 px).
            plain_html(f'<div class="shep-reason"><span>{BUILD_LINE}</span></div>'
                       f'<p class="shep-note">{FOOTNOTE}</p>', elem_id="shep-ds-notes")
        with gr.Column(elem_classes=["shep-panel", "shep-ds-details"], scale=10):
            details = plain_html(details_html({}, None), elem_id="shep-ds-details")
            with gr.Row(elem_classes=["shep-panel-ft"]):
                use = gr.Button("Use for training ›", variant="primary", interactive=False,
                                elem_id="shep-ds-use", scale=0)
                use_note = plain_html(use_note_html({}, None), elem_id="shep-ds-use-note")

    def _refresh(current, run_bar):
        data = read_datasets()
        names = [x["name"] for x in data["rows"]]
        chosen = current if current in names else (names[0] if names else None)
        return (data, chosen, header_html(data), table_html(data, run_bar, chosen),
                details_html(data, chosen), _use_update(data, chosen),
                use_note_html(data, chosen))

    refresh = {"fn": _refresh, "inputs": [selected, ctx.training_workspace],
               "outputs": [data_state, selected, header, table, details, use, use_note],
               "show_progress": "hidden"}
    ctx.on_page_open("datasets", refresh)

    def _select(data, run_bar, evt: gr.EventData):
        raw = getattr(evt, "_data", None)
        key = raw.get("key") if isinstance(raw, Mapping) else None
        names = [x["name"] for x in data.get("rows") or []]
        if key not in names:
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        return (key, table_html(data, run_bar, key), details_html(data, key),
                _use_update(data, key), use_note_html(data, key))

    table.click(_select, inputs=[data_state, ctx.training_workspace],
                outputs=[selected, table, details, use, use_note], show_progress="hidden")

    def _use(data, name):
        row = next((x for x in data.get("rows") or [] if x["name"] == name), None)
        if row is None or not row.get("trainable"):
            return (gr.update(), *ctx.no_nav())
        return (name, *ctx.go("training"))

    use.click(_use, inputs=[data_state, selected],
              outputs=[ctx.training_workspace, *ctx.nav_outputs], show_progress="hidden",
              **ctx.training_workspace_writes)

    def _run_bar_changed(data, name, run_bar):
        return table_html(data, run_bar, name)

    ctx.training_workspace.change(_run_bar_changed,
                                  inputs=[data_state, selected, ctx.training_workspace],
                                  outputs=[table], show_progress="hidden")


def _use_update(data: Mapping[str, Any], name: Optional[str]) -> Any:
    row = next((x for x in data.get("rows") or [] if x["name"] == name), None)
    return gr.update(interactive=bool(row and row.get("trainable")))


_PAGE = "#" + page_elem_id("datasets")

#: This page's style sheet (the page contract: listed in `src/webui/app.py`'s
#: ``PAGE_CSS``). A long name or source list wraps in its cell at every size,
#: so the "In use" column stays in view.
PAGE_CSS = f"""
{_PAGE} #shep-ds-notes {{ padding: 0 16px 12px; }}
{_PAGE} #shep-ds-table .shep-tbl-wrap {{ margin: 10px 16px; }}
{_PAGE} #shep-ds-notes .shep-note {{ margin: 6px 0 0; }}
{_PAGE} .shep-dstbl td.shep-ds-name, {_PAGE} .shep-dstbl td.shep-ds-src {{ overflow-wrap: anywhere; }}
{_PAGE} .shep-dstbl td.shep-ds-name {{ min-width: 11.5em; }}
{_PAGE} .shep-dstbl td.shep-ds-src {{ min-width: 7em; }}
{_PAGE} .shep-dstbl td.shep-ds-use {{ min-width: 11em; }}
{_PAGE} .shep-compare {{ margin: 0; font-size: 15px; color: var(--shep-text-2); overflow-wrap: anywhere; }}
{_PAGE} .shep-ds-versions {{ display: flex; flex-direction: column; gap: 6px; }}
{_PAGE} .shep-ds-vt {{ font-size: 15px; font-weight: 600; }}
{_PAGE} .shep-ds-versions .shep-sub {{ margin: 0; }}
{_PAGE} .shep-ds-versions .shep-tbl-wrap {{ margin: 0; }}
{_PAGE} .shep-dsvtbl td {{ overflow-wrap: anywhere; }}
{_PAGE} .shep-dsvtbl .shep-more p {{ margin: 4px 0 0; font-size: 14px; color: var(--shep-text-2); }}
{_PAGE} #shep-ds-details .shep-v, {_PAGE} #shep-ds-details .shep-panel-hd h2 {{ overflow-wrap: anywhere; min-width: 0; }}
{_PAGE} .shep-ds-narrow {{ display: none; white-space: nowrap; }}
@media (max-width: 1799px) {{
  {_PAGE} .shep-dstbl .shep-ds-wide {{ display: none; }}
  {_PAGE} .shep-dstbl .shep-ds-narrow {{ display: block; }}
}}
@media (min-width: 1400px) and (min-height: 760px) {{
  {_PAGE} #shep-ds-table {{ flex: 0 1 auto !important; min-height: 120px; overflow: auto !important; }}
  {_PAGE} #shep-ds-table .shep-tbl-wrap {{ overflow: visible; }}
  {_PAGE} #shep-ds-details {{ flex: 1 1 auto !important; min-height: 0; overflow: auto !important; }}
}}
"""


__all__ = [
    "BUILD_LINE",
    "IMPORT_LINE",
    "OUTSIDE_PATH",
    "PAGE_CSS",
    "ROOT_REFUSED",
    "admitted_measurements",
    "comparison_html",
    "create_datasets_page",
    "details_html",
    "header_html",
    "read_datasets",
    "redact",
    "table_html",
    "use_note_html",
    "versions_html",
]
