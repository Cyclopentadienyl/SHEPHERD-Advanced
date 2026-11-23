"""
Research › Prepare › Datasets: the server's workspaces, read-only (M5-2, UI-A).
================================================================================
`docs/working/PLAN_TEST_RESULTS.md` §6, "M5's boundaries": a read-only Datasets
page over the server's workspace list. The same list Training, Test and Models
choose from (`server_lists.list_workspaces`), drawn with the shared dataset
renderer (`renderers.dataset_identity_html` and its pieces).

- **The panel's title** names its content and count, "Workspaces (5)" (never
  the tab's name again), over one line on what the list is.
- **Columns**, one line each at the owner's window: name, built (UTC), dataset
  fingerprint (12 hex in bold monospace, the full value on Copy; its ⓘ says
  what it is), sources (`renderers.sources_html`: "none (synthetic)" for a
  synthetic build, never "—"), train / val samples, in use (its ⓘ says what
  each marker means).
- **Markers:** "serving reads this workspace" (accent, as Test marks it) from
  the status's ``current_data_dir``; "same dataset as served" for another
  listed workspace with the same **full** fingerprint (a relation, plain
  text); "● in your Training run bar" from this browser session's run-bar
  selection (`PageContext.training_workspace`); "not trainable: <why>" (the
  run bar's grey tag, `renderers.not_trainable_chip`) rows stay listed.
- **Details** of the selected row, titled by that relation ("Selected
  workspace", never its name again), with its origin and record in the run
  bar's form (`renderers.dataset_marks_html`): the shared body
  (`renderers.dataset_details_html`: the fingerprint with Copy, the disease
  split and seeds, one fact per row, the manifest's files behind "Manifest
  contents (N files) ›"), its **registered data versions** (`versions_html`),
  and how it compares with the other listed workspaces ("Same graph build as
  …, another cut: …"; "Same dataset as …"). The manifests and the registry
  are read once per refresh (`read_datasets`), not on each selection.
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
  bar and opens Training; it changes nothing on the server (`USE_NOTE`, the one
  place the page says so). It is queued in the group of every writer of that workspace
  (`PageContext.training_workspace_writes`), after any render of Training's run
  bar asked before it.
- **Building datasets**: they are built on the server with the CLI
  (`BUILD_COMMAND`, the demo's `DEMO_COMMAND`). With no workspaces folder,
  the table's place shows the folder's notice (`workspaces.setting_notice`,
  drawn once by `renderers.notice_block`) as numbered steps: choose the
  folder (the notice's link to Settings › Data locations), then build a
  dataset in it (`BUILD_STEP`), the commands a row each under the steps
  (`commands_html`: the token and its Copy, never inside a sentence); with
  a folder that lists no workspace, its own empty words, the same step and
  rows. Under a listed table, one quiet line (`BUILD_LINE`) and the rows;
  nothing else on the page repeats it.
- **Laid out by what it holds** (the UI sweep of 2026-11-22): the two panels
  grow from their own bases (`--shep-panel-basis`, `--shep-panel-grow`), the
  table's panel no narrower than the table with its optional columns
  dropped (Built and Sources, `shep-col-opt`; the shell's fit drops them
  while the whole table does not fit, and the selected row's details then
  say them), and an empty list asks for no table's width.

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
from src.webui.components.help import copy_button, help_button
from src.webui.components.served_status import read_status, serving_workspace
from src.webui.components.workspaces import (
    OUTSIDE_PATH,
    ROOT_REFUSED,
    binding_state,
    manifest_facts,
    read_workspaces,
    redact,
    setting_notice,
    version_row,
    workspace_binding,
)
from src.webui.shell import page_elem_id, plain_html
from src.webui.style import COL_OPT, FIT, FIT_WIDE, NARROW_ATTR, SIDE

logger = logging.getLogger(__name__)

#: How a dataset is built (no building in the WebUI, BACKLOG 29): the command
#: that writes a trainable workspace (graph, samples, manifest and dataset
#: record), and the demo's. Each is a key-value row of its own, the command a
#: literal token with its Copy (`commands_html`), never inside a sentence (the
#: text brief's §8: a token of over 30 characters, or a second one, is a row).
BUILD_COMMAND = "scripts/build_knowledge_graph.py --generate-samples"
DEMO_COMMAND = "scripts/setup_demo.py"
#: The commands' rows: each key names what its command does.
COMMANDS = (("Build", BUILD_COMMAND, "the build command"),
            ("Demo", DEMO_COMMAND, "the demo's command"))
#: The step after choosing the workspaces folder (the first-use notice's step
#: 2), and the next step of a folder that lists no workspace yet: the act in
#: words, its commands the rows under it.
BUILD_STEP = "Build a dataset in it on the server, with a command below."
#: The one quiet line under a listed table: how another dataset is added.
BUILD_LINE = "Add a dataset on the server, with a command below."
#: A workspaces folder that is set and lists no workspace (its own words; the
#: folder's notice is for a folder not set or not usable).
NO_WORKSPACE_YET = "No workspace in this folder yet."
#: What "Use for training ›" does, beside it: the one place the page says what
#: "in your Training run bar" means (the In use ⓘ names the button, B51-h).
USE_NOTE = ("Selects it in your Training run bar, for this browser tab only, and opens "
            "Training.")
#: The In use column's markers, in plain words (its header's ⓘ; the WebUI
#: skill's §7: the markers stay visible text, the bubble explains them).
PAGE_HELP = {
    "datasets-in-use": (
        "In use",
        "What uses each workspace now, or a dash when nothing does.\n"
        "- Serving reads this workspace: the served model reads its files.\n"
        "- Same dataset as served: another workspace, same full fingerprint.\n"
        "- In your Training run bar: chosen with Use for training ›.\n"
        "- Not trainable: listed, but training refuses it, for the reason given.",
    ),
}


#: The control the line beside "Use for training ›" names while it is
#: disabled (`renderers.disabled_words`): with no workspace listed, its
#: relation to the workspaces folder's notice (`renderers.see_above`; the
#: notice is the table's, in view of the button: one line beside it).
USE_FOR_TRAINING = "Use for training"


def use_note_html(data: Mapping[str, Any], name: Optional[str]) -> str:
    """The line beside "Use for training ›": what it does, or, when it is
    disabled, why, as visible text (M5-3: a not-trainable row disabled it with
    no reason beside it). With no workspace listed for want of a usable
    workspaces folder, its relation to the folder's notice (said once, in the
    table's place)."""
    rows = data.get("rows") or []
    row = next((x for x in rows if x["name"] == name), None)
    if not rows and data.get("notice"):
        return r.see_above(USE_FOR_TRAINING, data["notice"])
    if not rows:
        return r.marked_line("neutral", r.disabled_words(USE_FOR_TRAINING,
                                                         "no workspace is listed yet."))
    if row is None:
        return r.marked_line("neutral", r.disabled_words(USE_FOR_TRAINING,
                                                         "select a workspace in the list first."))
    if not row.get("trainable"):
        why = row.get("not_trainable") or "not trainable"
        words = r.disabled_words(USE_FOR_TRAINING, f"{row['name']} is not trainable: "
                                                   f"{r.not_trainable_short(row, 120)}.")
        return (f'<p class="shep-reason" title="{r.e(why)}">{r.DISABLED_MARK}'
                f'<span>{r.e(words)}</span></p>')
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
    versions, registry_problem, registry_notice = _read_versions(root)
    for row in rows:
        row["versions"] = _bound_versions(row, versions, directories)
    return {"rows": rows, "serving": serving, "problem": listed["problem"],
            "notice": listed["notice"], "status_read": status is not None,
            "versions": versions, "registry_problem": registry_problem,
            "registry_notice": registry_notice}


def _read_versions(root: Any) -> tuple:
    """The registry's versions as plain data (`workspaces.version_row`, their
    manifests only, no report store), read by the registry's one listing
    (`server_lists.list_cohort_versions`, Test's), why none is listed,
    without a path, and the registry's notice (`workspaces.setting_notice`:
    not set, or set and not usable; chosen by the setting, never by words)."""
    from src.api.services.server_lists import list_cohort_versions
    from src.config.server_settings import cohort_registry

    listing = list_cohort_versions(contents=True)
    problem = redact(listing.problem, root) if listing.problem else None
    return ([version_row(item) for item in listing.items], problem,
            setting_notice(cohort_registry()) if problem else None)


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
    none = {"bound": [], "another": 0, "changed": 0, "unknown": 0}
    if not versions:
        return {**none, "problem": None}
    directory = directories.get(row["name"])
    if directory is None:
        why = str(row.get("not_trainable") or "the workspace was not reached").rstrip(". ")
        return {**none, "problem": why}
    binding = workspace_binding(directory, row)
    if binding.get("problem"):
        return {**none, "problem": binding["problem"]}
    states: Dict[str, List[str]] = {"bound": [], "another": [], "changed": [], "unknown": []}
    for version in versions:
        states[binding_state(version, binding)[0]].append(str(version.get("label")))
    return {"bound": states["bound"], "another": len(states["another"]),
            "changed": len(states["changed"]), "unknown": len(states["unknown"]),
            "problem": None}


def _marks(row: Mapping[str, Any], data: Mapping[str, Any], run_bar: Optional[str]) -> str:
    marks = []
    serving_row = next((x for x in data.get("rows") or [] if x["name"] == data.get("serving")),
                       None)
    if row["name"] == data.get("serving"):
        marks.append('<span class="shep-mark shep-mark-serve">serving reads this workspace</span>')
    elif (serving_row is not None and r.is_digest(row.get("dataset_fingerprint"))
          and row.get("dataset_fingerprint") == serving_row.get("dataset_fingerprint")):
        marks.append('<span class="shep-sub">same dataset as served</span>')
    if run_bar and row["name"] == run_bar:
        marks.append(r.status_mark("ok", "in your Training run bar"))
    if not row.get("trainable"):
        marks.append(r.not_trainable_chip(row))
    return " ".join(marks) or f'<span class="shep-muted">{r.NO_VALUE}</span>'


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


def empty_html(data: Mapping[str, Any]) -> str:
    """**The table's place with no workspace listed**, on the panel's content
    edge (`shep-ds-empty`), in its own words per cause (the WebUI skill's
    §9): no workspaces folder, the folder's notice (`renderers.notice_block`,
    said once on the page) with its link as step 1 and the build as step 2
    (`BUILD_STEP`); a folder set and not usable, its notice alone (the
    folder comes first); a folder that lists none, `NO_WORKSPACE_YET` and the
    build; a list that could not be read, its reason."""
    notice = data.get("notice")
    if notice:
        build = notice.get("kind", "neutral") == "neutral"
        inner = (r.notice_block(notice, steps=(BUILD_STEP,) if build else ())
                 + (commands_html() if build else ""))
    elif data.get("problem"):
        inner = f'<p class="shep-empty">{r.status_mark("error", data["problem"])}</p>'
    else:
        inner = (f'<div class="shep-empty"><p>{r.e(NO_WORKSPACE_YET)}</p>'
                 f"<p>{r.e(BUILD_STEP)}</p></div>{commands_html()}")
    return f'<div class="shep-ds-empty">{inner}</div>'


def commands_html() -> str:
    """**The build commands**, a key-value row each (`COMMANDS`): its key,
    the command a literal token broken only before its own separators, and
    its Copy (the full command on the clipboard)."""
    return r.summary_list([(key, f"{r.token_html(command)} {copy_button(command, what)}")
                           for key, command, what in COMMANDS], classes="shep-ds-cmds")


#: The columns a narrow table drops (the shell's fit, `style.FIT`), their
#: facts then said by the selected row's details (`optional_html`).
_OPT = f' class="{COL_OPT}"'


def table_html(data: Mapping[str, Any], run_bar: Optional[str], selected: Optional[str]) -> str:
    """The workspaces table; each row selects its details. A fit box
    (`style.FIT`): while the whole table does not fit its panel, its optional
    columns (Built, Sources) are not drawn (`style.NARROW_ATTR`)."""
    rows = data.get("rows") or []
    if not rows:
        return empty_html(data)
    digits = r.distinguishing_digits(x.get("dataset_fingerprint") for x in rows)
    body = []
    for row in rows:
        name = row["name"]
        fingerprint = row.get("dataset_fingerprint")
        built = r.built_date(row.get("build"))
        fp_cell = (r.fingerprint_value(fingerprint, what=f"dataset fingerprint of {name}",
                                       digits=digits)
                   if r.is_digest(fingerprint) else _fingerprint_missing(row))
        on = name == selected
        body.append(
            # A plain table's selected row is `aria-current` (the skill's §8).
            f'<tr class="{"shep-on" if on else ""}" data-key="{r.e(name)}" tabindex="0"'
            f'{CURRENT if on else ""}>'
            f'<td class="shep-ds-name shep-w600">{r.name_pieces_html(name)}</td>'
            f'<td class="shep-nowrap shep-ds-date {COL_OPT}">'
            f'{r.e(built) if built else r.NOT_RECORDED.lower()}</td>'
            f'<td class="shep-nowrap">{fp_cell}</td>'
            f'<td class="shep-ds-src {COL_OPT}">{r.sources_html(row)}</td>'
            f'<td class="shep-num shep-nowrap">'
            f'{r.e(r.pair_text(row.get("train_samples"), row.get("val_samples")))}</td>'
            f'<td class="shep-ds-use">{_marks(row, data, run_bar)}</td></tr>')
    return (
        f'<div class="shep-tbl-wrap {FIT}"><table class="shep-tbl shep-dstbl {FIT_WIDE}" '
        'aria-label="Workspaces">'
        f'<thead><tr><th scope="col">Name</th><th scope="col"{_OPT}>Built (UTC)</th>'
        '<th scope="col">Dataset fingerprint'
        f'{help_button("dataset-fingerprint", "Dataset fingerprint")}</th>'
        f'<th scope="col"{_OPT}>Sources</th><th scope="col" class="shep-num">Train / val</th>'
        f'<th scope="col">In use{help_button("datasets-in-use", "In use")}</th>'
        f'</tr></thead><tbody>{"".join(body)}</tbody></table></div>')


def build_line_html() -> str:
    """The quiet line under a listed table (`BUILD_LINE`), its commands the
    rows under it (`commands_html`)."""
    return f'<p class="shep-ds-build">{r.e(BUILD_LINE)}</p>{commands_html()}'


#: The first panel's one-line description, under its title.
INTRO = "On the server: the same list Training, Test and Models choose from."


def header_html(data: Mapping[str, Any]) -> str:
    """The panel's title by its content and count, "Workspaces (5)" (R-30:
    never the selected tab's name again; never "read-only", which names the
    role D8 refuses), and its one-line description, which defines nothing (the
    fingerprint's ⓘ is on its column, R-34)."""
    count = len(data.get("rows") or [])
    return (f'<div class="shep-panel-hd"><h2>Workspaces <span class="shep-muted">({count})'
            f'</span></h2></div><p class="shep-intro">{INTRO}</p>')


#: The bound files two workspaces share when they come from one graph build
#: (by their names in `workspaces.BOUND_ROLES`).
_GRAPH_BUILD = ("Graph", "Node features", "Edge index", "Node counts", "Dataset record")


def _names(names: List[str], limit: int = 3) -> str:
    shown = ", ".join(f"<b>{r.e(n)}</b>" for n in names[:limit])
    return shown + (f" and {len(names) - limit} more" if len(names) > limit else "")


def comparison_html(row: Mapping[str, Any], rows: List[Mapping[str, Any]]) -> str:
    """How the selected workspace compares with the other listed ones, on the
    full digests its manifest binds: "Same dataset as …" (the same full
    fingerprint), "Same graph build as … (graph, tensors and record
    identical), another cut: … differ"."""
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
        parts.append(f"Same graph build as {_names(names)} (graph, tensors and record "
                     f"identical), another cut: {what}.")
    return "".join(f'<p class="shep-compare">{p}</p>' for p in parts)


#: The details panel's title: the row it shows by its relation (B51-h: the
#: selected row beside it names it; the title never says the name again).
DETAILS_TITLE = "Selected workspace"
#: The details with no row chosen: a row to select, or, with none listed, what
#: will appear (the list's own place says why none is listed and what to do).
DETAILS_SELECT = "Select a dataset to see what its manifest binds."
DETAILS_NONE = "A workspace’s details show here once the list holds one."


def details_html(data: Mapping[str, Any], name: Optional[str]) -> str:
    """The selected workspace's details: titled by its relation
    (`DETAILS_TITLE`), with its origin and record in the run bar's form
    (`renderers.dataset_marks_html`; the table shows neither), the shared
    details body (`renderers.dataset_details_html`, as Training's popover
    shows it, less what the selected row shows) and how it compares with the
    other listed workspaces."""
    rows = data.get("rows") or []
    row = next((x for x in rows if x["name"] == name), None)
    if row is None:
        said = (DETAILS_SELECT if rows else DETAILS_NONE)
        return (f'<div class="shep-panel-hd"><h2>{DETAILS_TITLE}</h2></div>'
                f'<div class="shep-panel-bd"><p class="shep-empty">{said}</p></div>')
    head = (f'<div class="shep-panel-hd"><h2>{DETAILS_TITLE}</h2>'
            f'{r.dataset_marks_html(row)}</div>')
    body = r.dataset_details_html(row, row.get("facts"), problem=row.get("facts_problem"))
    # Named by the workspace (`renderers.keep_as`): its manifest's and its
    # versions' disclosures are never drawn open for another one selected.
    return (head + f'<div class="shep-panel-bd"{r.keep_as(row["name"])}>' + optional_html(row)
            + body + versions_html(data, row)
            + comparison_html(row, list(rows)) + "</div>")


def optional_html(row: Mapping[str, Any]) -> str:
    """The selected row's facts of the columns a narrow table drops (Built,
    Sources: `table_html`'s optional columns), drawn only while the table
    drops them (`PAGE_CSS`): each fact is said once on the screen, in the
    row while it shows the column, else here, first (what the row said
    first)."""
    built = r.built_date(row.get("build"))
    return ('<div class="shep-ds-optkv">' + r.summary_list([
        ("Built (UTC)", r.e(built) if built else r.NOT_RECORDED.lower()),
        ("Sources", r.sources_html(row))]) + "</div>")


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
        # Each name whole and the chevron joined to the last one: the label wraps only
        # between names, never at a name's own hyphen, never "›" alone on a line.
        cell += (f'<details class="shep-more"><summary>Not admitted: {r.whole_words(names)}'
                 f"\u00a0›</summary>{reasons}</details>")
    return cell


def versions_html(data: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    """The workspace's registered data versions (the module docstring): a
    table of the versions mapped against its graph, each by its label, kind,
    role and the measurements its data admits; under it how many others the
    registry holds; or, in its own words, why none is shown (no registry, the
    workspace not checked, none mapped against it yet), with where a version
    is imported. No usable registry is the registry's notice
    (`renderers.notice_block`, said once on the screen, here), its link to
    Settings › Data locations only when the page shows no other notice (the
    workspaces folder's, whose link goes there too)."""
    versions = {str(v.get("label")): v for v in data.get("versions") or []}
    mine = row.get("versions") or {}
    bound = [versions[label] for label in mine.get("bound") or [] if label in versions]
    title = (f'<div class="shep-ds-vt">Data versions <span class="shep-muted">'
             f'({len(bound)})</span></div>')
    problem = data.get("registry_problem")
    if problem and data.get("registry_notice"):
        said = r.notice_block(data["registry_notice"], go=not data.get("notice"))
        return f'<div class="shep-ds-versions">{title}{said}</div>'
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
    if mine.get("changed"):
        n = mine["changed"]
        others.append(f"{n} whose files changed after the import")
    if mine.get("unknown"):
        n = mine["unknown"]
        others.append(f"{n} that cannot be read or checked now")
    tail = (f'<p class="shep-sub">Not listed here: {r.e(", ".join(others))}.</p>'
            if others else "")
    if not bound:
        return (f'<div class="shep-ds-versions">{title}<div class="shep-reason"><span>No data '
                f"version is mapped against this workspace’s graph yet. {IMPORT_LINE}</span>"
                f"</div>{tail}</div>")
    body = "".join(
        f'<tr><td class="shep-w600">{r.whole_words(version.get("label"))}</td>'
        f'<td>{_kind_text(version.get("kind"))}</td>'
        f'<td>{r.ident_html(version.get("role"))}</td>'
        f"<td>{_measurements_cell(version)}</td></tr>" for version in bound)
    table = ('<div class="shep-tbl-wrap"><table class="shep-tbl shep-dsvtbl" '
             'aria-label="Data versions"><thead><tr><th scope="col">Version</th>'
             '<th scope="col">Kind</th><th scope="col">Role</th>'
             f'<th scope="col">Measurements</th></tr></thead><tbody>{body}</tbody></table></div>')
    lead = ('<p class="shep-sub">In the server’s registry, mapped against this workspace’s '
            "graph: what its data admits. The model chosen on Test must take it too.</p>")
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
    ctx.help.topics(PAGE_HELP)
    data_state = gr.State({"rows": [], "serving": None, "problem": None})
    selected = gr.State(None)
    with gr.Row(elem_classes=["shep-page", "shep-datasets"], equal_height=False):
        # Sized by what each holds (`PAGE_CSS`, the shared `shep-sized`), not by
        # shares fitted to one window.
        with gr.Column(elem_classes=["shep-panel", "shep-sized", "shep-ds-main"]):
            header = plain_html(header_html({"rows": []}), elem_id="shep-ds-header")
            table = plain_html('<p class="shep-ds-empty shep-empty">Reading the server’s workspace '
                               'list…</p>', elem_id="shep-ds-table", js_on_load=_ROWS_JS)
            # The one quiet line under a listed table (drawn only with one,
            # `PAGE_CSS`): an empty list says how in its own place.
            plain_html(build_line_html(), elem_id="shep-ds-notes")
        with gr.Column(elem_classes=["shep-panel", "shep-sized", "shep-ds-details"]):
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
/* One content edge (the panel's inset): the empty list's words and the quiet build line start
   where the title and the intro do. */
{_PAGE} .shep-ds-empty {{ padding: var(--shep-space-2) var(--shep-inset) var(--shep-space-3); }}
{_PAGE} .shep-ds-empty .shep-empty > p + p {{ margin-top: var(--shep-space-para); }}
{_PAGE} #shep-ds-notes {{ padding: 0 var(--shep-inset) var(--shep-space-3); }}
{_PAGE} .shep-ds-build {{ margin: 0; font-size: 14px; line-height: var(--shep-lh-prose); color: var(--shep-text-2); }}
/* The build commands, a row each under the step or the line that names them. */
{_PAGE} .shep-ds-cmds {{ margin-top: var(--shep-space-item); }}
/* The build line belongs to a listed table: an empty list says how in its own place. */
{_PAGE} .shep-ds-main:not(:has(table.shep-dstbl)) #shep-ds-notes {{ display: none !important; }}
{_PAGE} #shep-ds-table .shep-tbl-wrap {{ margin: 10px var(--shep-inset); }}
/* A name breaks only before its own separators (`renderers.name_pieces_html`): its column is as
   narrow as its longest piece, never an em floor (11.5em held 173 px for "demo" and wrapped In use
   and the headers at 1850 x 960; the sweep's L7, SW-LK3). A source list wraps anywhere. */
{_PAGE} .shep-dstbl td.shep-ds-src {{ overflow-wrap: anywhere; min-width: 7em; }}
{_PAGE} .shep-dstbl td.shep-ds-use {{ min-width: 11em; }}
/* The build date in tabular figures: every date is one width, so the table's narrowest width
   (and the panel minimum below) does not move with the minute a workspace was built. */
{_PAGE} .shep-dstbl td.shep-ds-date {{ font-variant-numeric: tabular-nums; }}
/* One line a row at the owner's window, at most 40 px (R-36): the 28 px Copy
   keeps its size and target, its 2 px above and below the text's line laid
   into the cell's padding. */
{_PAGE} .shep-dstbl td {{ padding-top: 7px; padding-bottom: 7px; }}
{_PAGE} .shep-dstbl button.shep-copy {{ margin: -2px 0; }}
{_PAGE} .shep-dstbl .shep-tag {{ height: auto; min-height: 24px; white-space: normal; border-radius: 12px; }}
{_PAGE} .shep-dstbl th .shep-i {{ margin-left: 4px; }}
{_PAGE} .shep-compare {{ margin: 0; font-size: 15px; color: var(--shep-text-2); overflow-wrap: anywhere; }}
{_PAGE} .shep-ds-versions {{ display: flex; flex-direction: column; gap: 6px; }}
{_PAGE} .shep-ds-vt {{ font-size: 15px; font-weight: 600; }}
{_PAGE} .shep-ds-versions .shep-sub {{ margin: 0; }}
{_PAGE} .shep-ds-versions .shep-tbl-wrap {{ margin: 0; }}
{_PAGE} .shep-dsvtbl td {{ overflow-wrap: break-word; }}
{_PAGE} .shep-dsvtbl .shep-more p {{ margin: 4px 0 0; font-size: 14px; color: var(--shep-text-2); }}
{_PAGE} #shep-ds-details .shep-v, {_PAGE} #shep-ds-details .shep-panel-hd h2 {{ overflow-wrap: anywhere; min-width: 0; }}
/* The selected row's Built and Sources, said by its details only while the table drops those
   columns (the shell's fit marks the table's box, `{NARROW_ATTR}`): each fact once on the screen. */
{_PAGE}:not(:has(#shep-ds-table [{NARROW_ATTR}])) .shep-ds-optkv {{ display: none !important; }}
/* The details' summary lists share one key column, so their values start on one edge. */
{_PAGE} #shep-ds-details .shep-panel-bd > .shep-kv,
{_PAGE} #shep-ds-details .shep-ds-optkv > .shep-kv {{ grid-template-columns: 10em minmax(0, 1fr); }}
/* What "Use for training ›" does: beside it while its one line fits there, else a line of its own
   under it (never a column of words beside a button): its one line's width, never shrunk, at most
   the footer's (it shrank to a column of three lines beside the button at 1850 x 960). */
{_PAGE} #shep-ds-use-note {{ flex: 0 0 max-content !important; max-width: 100%; min-width: 0 !important; }}
/* **The two panels, sized by what they hold** (M4, the shared `shep-sized`): each grows from its
   basis, the list three times as fast; the details keep their digest grid's four columns (31rem); the
   list keeps the table with its optional columns dropped (Name, the fingerprint, Train / val and
   In use at their narrowest, in the dashboard's type: 548 px with the demo's two rows, 589 px
   with a workspace whose counts are not recorded, plus the wrap's margins and edges, when Name
   held an 11.5em floor; 435 px since a name is as narrow as its longest piece, the sweep's L7;
   40rem leaves room for longer names, wider counts and more digits; wider still, the table
   scrolls in its own box), and with
   no table only the measure of its words (32rem). The shell's layout rule reads these minimums
   (`src/webui/shell.py`): side by side from about 1190 px of window with a table, 1060 without. */
{_PAGE} .shep-panel.shep-ds-main {{ --shep-panel-basis: 40rem; --shep-panel-grow: 3; --shep-panel-min: 32rem; }}
{_PAGE} .shep-panel.shep-ds-main:has(table.shep-dstbl) {{ --shep-panel-min: 40rem; }}
{_PAGE} .shep-panel.shep-ds-details {{ --shep-panel-basis: 31rem; --shep-panel-grow: 1; --shep-panel-min: 31rem; }}
{SIDE} {{
  {_PAGE} #shep-ds-table {{ flex: 0 1 auto !important; overflow: auto !important; }}
  {_PAGE} #shep-ds-table .shep-tbl-wrap {{ overflow: visible; }}
  {_PAGE} #shep-ds-details {{ flex: 1 1 auto !important; min-height: 0; overflow: auto !important; }}
}}
"""


__all__ = [
    "BUILD_COMMAND",
    "BUILD_LINE",
    "BUILD_STEP",
    "COMMANDS",
    "DEMO_COMMAND",
    "DETAILS_NONE",
    "DETAILS_SELECT",
    "DETAILS_TITLE",
    "IMPORT_LINE",
    "INTRO",
    "NO_WORKSPACE_YET",
    "OUTSIDE_PATH",
    "PAGE_CSS",
    "PAGE_HELP",
    "ROOT_REFUSED",
    "USE_NOTE",
    "USE_FOR_TRAINING",
    "admitted_measurements",
    "build_line_html",
    "commands_html",
    "comparison_html",
    "create_datasets_page",
    "details_html",
    "empty_html",
    "header_html",
    "optional_html",
    "read_datasets",
    "redact",
    "table_html",
    "use_note_html",
    "versions_html",
]
