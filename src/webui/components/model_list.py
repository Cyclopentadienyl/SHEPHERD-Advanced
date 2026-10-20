"""
The shared model list and the one Load (M5-2, UI-A).
====================================================
One component, two presets: Diagnosis' "Change model…" drawer uses the
``clinical`` preset (UI-B), Research › Models the ``research`` preset (UI-D).
Both list the same checkpoints the same way and load through the same handler,
so the two views cannot drift apart (PLAN §6, "M5's boundaries": one shared
model list, one gated Load, no second implementation).

**The list** is WP1's checkpoint list for a workspace the server lists
(`server_lists.list_checkpoints`). Listing opens every checkpoint of the
workspace (SPEC_4 C3), so it is an operator control: D8's helper is asked on the
triggering request first, and a read-only request gets the reason and no list.
Rows are the shared renderer's (`renderers.model_row_html`): "In use" is pinned
on top, matched by the **full** model fingerprint the status serves, and marks
the checkpoint only; an unreadable entry is greyed "metadata not readable ·
cannot be loaded". Each row's recorded tests are a state read from the ledger
beside the checkpoint (`read_ledger`, `records_for`) and, for contested, from
the cohort version's report store (`scan_report_store`): never a number of a
result.

**The Load**, in three steps, each a handler of its own:

1. "🔒 Load this model" (`request_load`): D8 on that request; the item is found
   in the server's list again (`resolve_checkpoint`, never a client path); one
   inline confirmation, "Reload the pipeline with <12 hex> for every session?",
   from → to: model, model fingerprint, trained on, serving workspace and graph
   digest, SP table, scoring mode with eta, synthetic flag, the changed rows
   marked;
2. "🔒 Reload with <new>" (`finish_load`): D8 again, on that request; the
   status is read again, and a model in use that changed since the
   confirmation was shown is said, never reloaded over; then WP2's one reload
   (`run_reload`, in process, on the event loop as the API route runs it) with
   the workspace and the checkpoint the server resolves from the item; the
   served model fingerprint is compared with the one shown; a usage history
   entry (WP4 `record_model_load`) only when the list was ordered by a cohort
   version's score (`ordering`, which Models sets); then the chips and the
   "served model changed" trigger (`PageContext.served_model_changed`);
3. "Keep <old>" closes the confirmation; nothing is done.

**The page waits 180 s.** The reload holds the event loop (it never runs beside
a diagnosis, BACKLOG 28), so nothing answers meanwhile: the page itself counts
(the head script, `MODEL_LIST_JS`) and after 180 s says "The page stopped
waiting after 180 s; the server may still finish the reload." with "Read again"
(a status read). When the reload's answer arrives later, it says so ("The
reload finished after the page stopped waiting.").

**Per browser session**: the listing, the selection and the confirmation are
`gr.State`; nothing here is module state.

Module: src/webui/components/model_list.py
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import gradio as gr

from src.webui.components import renderers as r
from src.webui.components.served_status import (
    read_status,
    served_checkpoint_sha,
    serving_workspace,
)
from src.webui.operator_access import operator_access
from src.webui.shell import plain_html

logger = logging.getLogger(__name__)

#: How long the page waits for a reload before it says it stopped waiting.
PAGE_WAIT_SECONDS = 180

#: What a confirmation says it replaces (the mockup's words).
REPLACES_NOTE = (
    "Replaces what every session is served; results already produced keep their own "
    "identity. Diagnoses wait while the pipeline reloads."
)
#: The note "In use" carries.
IN_USE_NOTE = "matched by the full model fingerprint (the checkpoint only, not the whole pipeline)"


@dataclass(frozen=True)
class Preset:
    """What differs between the two views of the one list."""

    key: str
    #: Runs shown before "Show older runs" (None: every run).
    latest_runs: Optional[int]
    #: The words of the "In use" mark.
    in_use_label: str


PRESETS: Dict[str, Preset] = {
    "clinical": Preset("clinical", 4, "In use"),
    "research": Preset("research", None, "In use (Diagnosis)"),
}


# ---------------------------------------------------------------------------
# The listing (server side)
# ---------------------------------------------------------------------------
def item_key(item: Mapping[str, Any]) -> str:
    """A listed checkpoint's key in this workspace: architecture/run/file."""
    return f"{item.get('architecture')}/{item.get('run')}/{item.get('filename')}"


def recorded_tests_for(workspace_dir: Path, items: Sequence[Mapping[str, Any]], *,
                       environ: Optional[Mapping[str, str]] = None) -> Dict[str, r.RecordedTests]:
    """Each listed checkpoint's recorded tests, as states: its ledger beside the
    checkpoint (read once per directory) and, for contested, the cohort version's
    report store (scanned once per version). Never raises: a problem is a state."""
    from src.evaluation.sidecar import LEDGER_FILENAME, read_ledger, record_key, records_for
    from src.utils.checkpoint_paths import RUN_CHECKPOINTS_DIRNAME, RUNS_DIRNAME

    ledgers: Dict[Path, Tuple[Optional[dict], Optional[str], bool]] = {}
    scans: Dict[str, Any] = {}
    versions: Optional[Dict[str, Any]] = None
    workspace_dir = Path(workspace_dir).resolve()

    def version_scan(digest: Any) -> Any:
        nonlocal versions
        if versions is None:
            versions = {}
            try:
                from src.config import server_settings
                from src.evaluation.cohort_registry import list_versions

                setting = server_settings.cohort_registry(environ)
                if setting.available:
                    for listed in list_versions(setting.value):
                        if listed.version is not None:
                            versions[listed.version.identity.sha256] = listed.version
            except Exception:  # noqa: BLE001 - contested then reads "not checked"
                logger.exception("The cohort registry could not be listed")
        if digest not in scans:
            version = versions.get(digest)
            if version is None:
                scans[digest] = None
            else:
                try:
                    from src.evaluation.report_store import scan_report_store

                    scans[digest] = scan_report_store(version)
                except Exception:  # noqa: BLE001
                    logger.exception("A report store could not be scanned")
                    scans[digest] = None
        return scans[digest]

    out: Dict[str, r.RecordedTests] = {}
    for item in items:
        sha = item.get("sha256")
        if not r.is_digest(sha) or item.get("problem"):
            continue
        file = (workspace_dir / RUNS_DIRNAME / str(item.get("architecture")) / str(item.get("run"))
                / RUN_CHECKPOINTS_DIRNAME / str(item.get("filename")))
        try:
            directory = file.resolve().parent
        except (OSError, RuntimeError):
            out[item_key(item)] = r.RecordedTests("not_read", problem="cannot be resolved")
            continue
        if directory != workspace_dir and workspace_dir not in directory.parents:
            out[item_key(item)] = r.RecordedTests("not_read", problem="outside the workspace")
            continue
        if directory not in ledgers:
            try:
                ledger, digest = read_ledger(directory / LEDGER_FILENAME)
                ledgers[directory] = (ledger, None, digest is not None)
            except (OSError, ValueError) as exc:
                ledgers[directory] = (None, type(exc).__name__, True)
        ledger, problem, present = ledgers[directory]
        if problem is not None:
            out[item_key(item)] = r.RecordedTests("not_read", problem=problem)
            continue
        if not present:
            out[item_key(item)] = r.RecordedTests("no_ledger")
            continue
        records = records_for(ledger, sha)
        if not records:
            out[item_key(item)] = r.RecordedTests("none")
            continue
        contested: Optional[int] = 0
        for record in records:
            scan = version_scan(record.get("cohort_digest"))
            try:
                state = scan.state(record_key(record)) if scan is not None else None
            except Exception:  # noqa: BLE001
                state = None
            if state == "contested":
                contested = None if contested is None else contested + 1
            elif state != "uncontested":
                contested = None
        out[item_key(item)] = r.RecordedTests("records", len(records), contested)
    return out


def _workspace_choices(workspaces: Sequence[Mapping[str, Any]]) -> List[Tuple[str, str]]:
    """The workspace select's options: "name · dataset fingerprint <12> · synthetic"."""
    digits = r.distinguishing_digits(w.get("dataset_fingerprint") for w in workspaces)
    choices = []
    for w in workspaces:
        label = f"{w['name']} · dataset fingerprint {r.short_hex(w.get('dataset_fingerprint'), digits)}"
        if r.origin_of(w) == "synthetic":
            label += " · synthetic"
        if not w.get("dataset_fingerprint"):
            label = f"{w['name']} · no readable split manifest"
        choices.append((label, w["name"]))
    return choices


def empty_listing(problem: Optional[str] = None) -> Dict[str, Any]:
    return {"workspace": None, "items": [], "workspaces": [], "recorded": {},
            "served_sha": None, "serving": None, "problem": problem}


def list_models(workspace: Any, request: Any, *,
                environ: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """The listing for one session: D8 on the triggering request first (listing
    opens checkpoints), then the server's lists and the status, as plain data
    for a `gr.State`. A refused request lists nothing and says why."""
    from src.api.services.server_lists import (
        list_checkpoints,
        list_workspaces,
        resolve_workspace,
    )

    access = operator_access(request)
    if not access.allowed:
        return {**empty_listing(access.refusal), "refused": True}
    workspaces = list_workspaces(environ=environ)
    rows = [asdict(w) for w in workspaces.items]
    status, status_problem = read_status()
    serving = serving_workspace(status, workspaces.items, environ=environ)
    names = [w["name"] for w in rows]
    chosen = workspace if workspace in names else (serving if serving in names else
                                                   (names[0] if names else None))
    listing = {**empty_listing(workspaces.problem), "workspaces": rows, "workspace": chosen,
               "served_sha": served_checkpoint_sha(status), "serving": serving,
               "status_problem": status_problem}
    if chosen is None:
        return listing
    checkpoints = list_checkpoints(chosen, environ=environ)
    items = [asdict(i) for i in checkpoints.items]
    listing.update(items=items, problem=checkpoints.problem)
    try:
        directory = resolve_workspace(chosen, environ=environ)
        recorded = recorded_tests_for(directory, items, environ=environ)
    except Exception:  # noqa: BLE001 - the recorded tests then read "not read"
        logger.exception("Recorded tests could not be read")
        recorded = {}
    listing["recorded"] = {k: v._asdict() for k, v in recorded.items()}
    return listing


def _recorded(listing: Mapping[str, Any], key: str) -> Optional[r.RecordedTests]:
    raw = (listing.get("recorded") or {}).get(key)
    return r.RecordedTests(**raw) if isinstance(raw, Mapping) else None


def _runs_in_order(items: Sequence[Mapping[str, Any]]) -> List[Tuple[str, List[Mapping[str, Any]]]]:
    """Items grouped by run, in the listing's order (latest run first)."""
    runs: Dict[str, List[Mapping[str, Any]]] = {}
    for item in items:
        runs.setdefault(f"{item.get('architecture')}/{item.get('run')}", []).append(item)
    return list(runs.items())


def list_html(listing: Mapping[str, Any], selected: Optional[str], preset: Preset, *,
              show_older: bool = False) -> str:
    """The list: "In use now" pinned, then runs newest first (the clinical
    preset shows the latest runs and "Show older runs")."""
    if listing.get("refused"):
        return f'<div class="shep-reason shep-reason-grey">{r.e(listing.get("problem"))}</div>'
    items = list(listing.get("items") or [])
    if not items:
        why = listing.get("problem") or "No checkpoint is listed in this workspace."
        return f'<div class="shep-reason">{r.e(why)}</div>'
    workspaces = listing.get("workspaces") or []
    digits = r.distinguishing_digits(i.get("sha256") for i in items)
    served = listing.get("served_sha")
    parts = []
    in_use = [i for i in items if served and i.get("sha256") == served]
    if in_use:
        parts.append(f'<div class="shep-list-hd"><b>In use now</b> · {r.e(IN_USE_NOTE)}</div>')
        for item in in_use[:1]:
            key = item_key(item)
            parts.append(r.model_row_html(item, workspaces, recorded=_recorded(listing, key),
                                          in_use=True, selected=key == selected, digits=digits,
                                          key=key, in_use_label=preset.in_use_label))
    runs = _runs_in_order(items)
    shown = runs if (preset.latest_runs is None or show_older) else runs[:preset.latest_runs]
    for run, run_items in shown:
        started = r.run_started(run_items[0].get("run"))
        parts.append(f'<div class="shep-list-hd"><b>Run {r.e(run)}</b>'
                     + (f" · started {r.e(started)}" if started else "") + "</div>")
        for item in run_items:
            key = item_key(item)
            if served and item.get("sha256") == served:
                parts.append(f'<div class="shep-list-note">epoch {r.e(item.get("epoch"))} · model '
                             f'fingerprint {r.e(r.short_hex(served, digits))} — in use, pinned above</div>')
                continue
            parts.append(r.model_row_html(item, workspaces, recorded=_recorded(listing, key),
                                          selected=key == selected, digits=digits, key=key,
                                          in_use_label=preset.in_use_label))
    hidden = len(runs) - len(shown)
    if hidden > 0:
        parts.append(f'<button type="button" class="shep-row shep-row-more" data-key="__older__">'
                     f'Show older runs ({hidden})</button>')
    return '<div class="shep-list" role="list">' + "".join(parts) + "</div>"


def count_line(listing: Mapping[str, Any], preset: Preset, show_older: bool) -> str:
    items = listing.get("items") or []
    runs = _runs_in_order(items)
    if listing.get("refused") or not items:
        return ""
    if preset.latest_runs is not None and not show_older and len(runs) > preset.latest_runs:
        shown = sum(len(v) for _, v in runs[:preset.latest_runs])
        return (f"newest first · latest {preset.latest_runs} runs · {shown} checkpoints "
                f"(of {len(items)} in {len(runs)} runs)")
    return f"newest first · {len(items)} checkpoints in {len(runs)} runs"


def find_item(listing: Mapping[str, Any], key: Any) -> Optional[Mapping[str, Any]]:
    if not isinstance(key, str):
        return None
    for item in listing.get("items") or []:
        if item_key(item) == key:
            return item
    return None


def loadable(item: Optional[Mapping[str, Any]]) -> bool:
    return bool(item) and item.get("problem") is None and r.is_digest(item.get("sha256"))


def facts_html(listing: Mapping[str, Any], key: Any) -> str:
    item = find_item(listing, key)
    if item is None:
        return '<div class="shep-muted">Select a model to see its facts.</div>'
    workspaces = listing.get("workspaces") or []
    serving = next((w for w in workspaces if w.get("name") == listing.get("serving")), None)
    facts = r.facts_from_checkpoint(item, workspaces, recorded=_recorded(listing, key),
                                    serving_workspace=serving)
    title = (f'<div class="shep-facts-title"><b>{r.e(facts.name_line)}</b> '
             f'{r.chip("selected", "neutral")}</div>')
    return title + r.model_facts_html(facts)


# ---------------------------------------------------------------------------
# The Load (server side)
# ---------------------------------------------------------------------------
def _graph_digest(workspace_dir: Path) -> Optional[str]:
    from src.kg.artifacts import read_split_manifest

    try:
        artifacts = read_split_manifest(workspace_dir).manifest.get("artifacts")
    except (ValueError, OSError):
        return None
    kg = artifacts.get("kg") if isinstance(artifacts, Mapping) else None
    return kg if r.is_digest(kg) else None


def _sp_in_workspace(workspace_dir: Path) -> Tuple[bool, Optional[str]]:
    """Whether the workspace holds a shortest-path table, and its build id."""
    from src.inference.sp_artifact import read_sidecar

    table = workspace_dir / "shortest_paths.pt"
    try:
        if not table.is_file():
            return False, None
        sidecar = read_sidecar(table)
    except Exception:  # noqa: BLE001 - present but unreadable: the reload decides
        return True, None
    build = sidecar.meta.get("build_id") if sidecar is not None else None
    return True, build if isinstance(build, str) else None


def _from_side(status: Optional[Mapping[str, Any]], listing: Mapping[str, Any]) -> Dict[str, Any]:
    """What is served now, as the confirmation's "From" column needs it."""
    identity = (status or {}).get("served_identity") or {}
    sha = served_checkpoint_sha(status)
    item = next((i for i in listing.get("items") or [] if sha and i.get("sha256") == sha), None)
    workspaces = listing.get("workspaces") or []
    meta = (status or {}).get("checkpoint_meta") or {}
    if item is not None:
        model = r.model_name_line(item)
    elif sha:
        model = f"epoch {meta.get('epoch', '—')} (not in this workspace's list)"
    else:
        model = "no trained model" if (status or {}).get("initialized") else "no model loaded"
    sp = identity.get("shortest_paths") or {}
    settings = identity.get("settings") or {}
    record = identity.get("dataset_record") or {}
    return {
        "model": model,
        "sha256": sha,
        "trained_on": item.get("dataset_fingerprint") if item else None,
        "trained_on_known": item is not None,
        "workspace": listing.get("serving"),
        "graph": (identity.get("graph") or {}).get("kg_sha256"),
        "sp": sp.get("state") == "verified",
        "sp_build": sp.get("build_id"),
        "mode": identity.get("scoring_mode"),
        "eta": settings.get("eta"),
        "synthetic": record.get("origin") == "synthetic",
        "workspaces": workspaces,
    }


def _to_side(item: Mapping[str, Any], workspace: str, workspace_dir: Path,
             status: Optional[Mapping[str, Any]], workspaces: Sequence[Any]) -> Dict[str, Any]:
    sp, build = _sp_in_workspace(workspace_dir)
    origin = next((r.origin_of(w) for w in workspaces if r._get(w, "name") == workspace), None)
    eta = (status or {}).get("eta_configured")
    return {
        "model": r.model_name_line(item),
        "sha256": item.get("sha256"),
        "trained_on": item.get("dataset_fingerprint"),
        "trained_on_known": True,
        "workspace": workspace,
        "graph": _graph_digest(workspace_dir),
        "sp": sp,
        "sp_build": build,
        "mode": "gnn_plus_shortest_path" if sp else "gnn_only",
        "eta": eta if sp else 1.0,
        "synthetic": origin == "synthetic",
    }


def _cell(side: Mapping[str, Any], row: str, workspaces: Sequence[Any], *, to: bool,
          other: Optional[Mapping[str, Any]] = None) -> str:
    if row == "model":
        return r.e(side["model"])
    if row == "sha":
        return f'<code class="shep-fpv">{r.e(r.short_hex(side["sha256"]))}</code>'
    if row == "trained":
        if not side.get("trained_on_known"):
            return '<span class="shep-muted">not in this workspace\'s list</span>'
        html_ = r.trained_on_html(side["trained_on"], workspaces)
        if to and other is not None and other.get("trained_on") and \
                other.get("trained_on") == side.get("trained_on"):
            html_ += ' <span class="shep-muted">same dataset</span>'
        return html_
    if row == "workspace":
        if to and other is not None and other.get("workspace") == side["workspace"] and \
                other.get("graph") == side["graph"]:
            return "same"
        name = f"<b>{r.e(side['workspace'])}</b>" if side["workspace"] else \
            '<span class="shep-muted">not a listed workspace</span>'
        return f'{name} · graph <code>{r.e(r.short_hex(side["graph"]))}</code>'
    if row == "sp":
        if not side["sp"]:
            return "absent"
        build = f": build <code>{r.e(side['sp_build'])}</code>" if side.get("sp_build") else ""
        return f"present{build}" + (' <span class="shep-muted">(as found in the workspace)</span>'
                                    if to else "")
    if row == "mode":
        eta = side.get("eta")
        eta_text = f" · eta {eta:g}" if isinstance(eta, (int, float)) else ""
        note = (' <span class="shep-muted">(server setting)</span>'
                if to and side["mode"] == "gnn_plus_shortest_path" else "")
        return f'<code>{r.e(side["mode"] or "—")}</code>{r.e(eta_text)}{note}'
    if row == "data":
        return r.chip("synthetic", "amber") if side["synthetic"] else "files"
    return ""


_ROWS = (("model", "Model"), ("sha", "Model fingerprint"), ("trained", "Trained on"),
         ("workspace", "Serving workspace"), ("sp", "SP table"), ("mode", "Scoring mode"),
         ("data", "Data"))
_COMPARED = {"model": ("sha256",), "sha": ("sha256",), "trained": ("trained_on",),
             "workspace": ("workspace", "graph"), "sp": ("sp", "sp_build"),
             "mode": ("mode", "eta"), "data": ("synthetic",)}


def confirmation_html(plan: Mapping[str, Any], *, notice: Optional[str] = None) -> str:
    """The one confirmation: "Reload the pipeline with <12 hex> for every
    session?", from → to with the changed rows marked, and what it replaces."""
    frm, to = plan["from"], plan["to"]
    workspaces = plan.get("workspaces") or []
    rows = []
    for key, label in _ROWS:
        changed = any(frm.get(f) != to.get(f) for f in _COMPARED[key])
        rows.append(
            f'<tr class="{"shep-changed" if changed else ""}"><th scope="row">{r.e(label)}'
            f'{"<span class=shep-vh> (changes)</span>" if changed else ""}</th>'
            f'<td>{_cell(frm, key, workspaces, to=False)}</td>'
            f'<td>{_cell(to, key, workspaces, to=True, other=frm)}</td></tr>')
    note = (f'<div class="shep-notice shep-notice-amber" role="alert">{r.e(notice)}</div>'
            if notice else "")
    new, old = r.short_hex(to["sha256"]), r.short_hex(frm.get("sha256"))
    return (
        f'<div class="shep-confirm" data-new="{r.e(new)}">{note}'
        f'<h3 class="shep-confirm-q">Reload the pipeline with <code>{r.e(new)}</code> for every '
        f'session?</h3>'
        '<table class="shep-ftbl"><thead><tr><th></th><th>From — in use now</th><th>To</th>'
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table>'
        f'<p class="shep-confirm-note">{r.e(REPLACES_NOTE)}</p>'
        f'<div class="shep-load-wait" hidden data-wait="{PAGE_WAIT_SECONDS}">'
        f'<span class="shep-spinner" aria-hidden="true"></span> <b>Loading <code>{r.e(new)}</code> '
        'for every session… <span class="shep-elapsed">00:00:00</span></b> Diagnoses wait until '
        'the reload finishes. Closing this does not stop it.</div>'
        '<div class="shep-load-timeout" hidden><b>The page stopped waiting after '
        f'{PAGE_WAIT_SECONDS} s; the server may still finish the reload.</b> '
        f'<button type="button" class="shep-btn shep-read-again" data-key="__read_again__">'
        'Read again</button></div>'
        f'<span class="shep-vh" data-old="{r.e(old)}"></span></div>'
    )


def request_load(listing: Mapping[str, Any], key: Any, ordering: Any, request: Any, *,
                 environ: Optional[Mapping[str, str]] = None
                 ) -> Tuple[Optional[Dict[str, Any]], str]:
    """Step 1: D8 on this request, the item found in the server's list again,
    and the confirmation. Returns ``(plan, html)``; a refusal returns no plan
    and says why."""
    from src.api.services.server_lists import (
        ServerListRefused,
        resolve_checkpoint,
        resolve_workspace,
    )

    access = operator_access(request)
    if not access.allowed:
        return None, f'<div class="shep-reason shep-reason-grey">{r.e(access.refusal)}</div>'
    item = find_item(listing, key)
    if not loadable(item):
        return None, ('<div class="shep-reason">Select a model that can be loaded first. '
                      'Nothing was done.</div>')
    workspace = listing.get("workspace")
    try:
        directory = resolve_workspace(workspace, environ=environ)
        resolve_checkpoint(workspace, item["architecture"], item["run"], item["filename"],
                           environ=environ)
    except ServerListRefused as exc:
        return None, f'<div class="shep-reason">Refused: {r.e(exc)} Nothing was done.</div>'
    status, problem = read_status()
    if status is None:
        return None, (f'<div class="shep-reason">Cannot load now: {r.e(problem)}. Nothing was '
                      'done.</div>')
    workspaces = listing.get("workspaces") or []
    frm = _from_side(status, listing)
    frm.pop("workspaces", None)
    plan = {"workspace": workspace, "architecture": item["architecture"], "run": item["run"],
            "filename": item["filename"], "from": frm,
            "to": _to_side(item, workspace, directory, status, workspaces),
            "workspaces": workspaces,
            "ordering": dict(ordering) if isinstance(ordering, Mapping) else None}
    return plan, confirmation_html(plan)


def _record_usage(plan: Mapping[str, Any], loaded: bool, why: Optional[str],
                  environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """The usage history entry for a Load from an ordered list (D4), or why it
    could not be written; None when the list was not ordered or it was written."""
    ordering = plan.get("ordering")
    if not ordering:
        return None
    try:
        from src.config import server_settings
        from src.evaluation.cohort_registry import resolve_version
        from src.evaluation.report_store import record_model_load

        setting = server_settings.cohort_registry(environ)
        if not setting.available:
            return f"the usage history was not written: {setting.problem}"
        version = resolve_version(setting.value, ordering["version"])
        expected = ordering.get("version_sha256")
        if expected and version.identity.sha256 != expected:
            return ("the usage history was not written: the cohort version changed since the "
                    "list was ordered")
        record_model_load(version, checkpoint_sha256=plan["to"]["sha256"],
                          metric=ordering["metric"],
                          scoring_semantics_digest=ordering["scoring_semantics_digest"],
                          loaded=loaded, why=why)
    except Exception as exc:  # noqa: BLE001 - the load stands; the page says what failed
        logger.exception("The usage history entry could not be written")
        return f"the usage history was not written ({type(exc).__name__})"
    return None


def _outcome(kind: str, headline: str, text: str = "") -> str:
    waited = ('<span class="shep-if-waited">The reload finished after the page stopped waiting. '
              '</span>')
    return (f'<div class="shep-outcome shep-outcome-{kind}" role="status">{waited}<b>'
            f'{r.e(headline)}</b> {r.e(text)}</div>')


class LoadStep(NamedTuple):
    """What step 2 of a Load leaves on the page.

    `plan` stays set (with the new from-values) when the confirmation is shown
    again; `confirm_html` is that confirmation or the outcome; `reason_html` a
    refusal beside the buttons; `served` the model fingerprint served after a
    reload; `changed` whether the served model changed (the "served model
    changed" trigger)."""

    plan: Optional[Dict[str, Any]]
    confirm_html: str
    reason_html: str
    served: Optional[str]
    changed: bool


def finish_load(plan: Optional[Mapping[str, Any]], request: Any, *,
                reload: Any = None, environ: Optional[Mapping[str, str]] = None) -> LoadStep:
    """Step 2, synchronous (the caller runs it on the event loop, as the API
    route runs its reload): D8 on this request, the status read again, WP2's
    reload with what the server resolves from the item, the served fingerprint
    checked against the one shown, the usage history."""
    from src.api.services.server_lists import (
        ServerListRefused,
        resolve_checkpoint,
        resolve_workspace,
    )

    access = operator_access(request)
    if not access.allowed:
        return LoadStep(dict(plan) if plan else None,
                        confirmation_html(plan) if plan else "",
                        f'<div class="shep-reason shep-reason-grey">{r.e(access.refusal)}</div>',
                        None, False)
    if not plan:
        return LoadStep(None, "", '<div class="shep-reason">Nothing to load. Nothing was done.</div>',
                        None, False)
    status, problem = read_status()
    if status is None:
        return LoadStep(dict(plan), confirmation_html(plan),
                        f'<div class="shep-reason">Not reloaded: {r.e(problem)}. Nothing was '
                        'done.</div>', None, False)
    now_sha = served_checkpoint_sha(status)
    if now_sha != plan["from"].get("sha256"):
        # What is served changed since this was shown (another session loaded a
        # model): say so with the new from-values; never reload over it unasked.
        frm = _from_side(status, {"items": [], "workspaces": plan.get("workspaces"),
                                  "serving": None})
        frm.pop("workspaces", None)
        if now_sha is not None:
            frm["model"] = f"model {r.short_hex(now_sha)} (loaded since this was shown)"
        new_plan = {**plan, "from": frm}
        notice = (f"The model in use changed since this was shown: it is now "
                  f"{r.short_hex(now_sha) if now_sha else 'none'}. Nothing was reloaded; check "
                  "the rows again, then reload or keep it.")
        return LoadStep(new_plan, confirmation_html(new_plan, notice=notice), "", now_sha, False)
    try:
        directory = resolve_workspace(plan["workspace"], environ=environ)
        checkpoint = resolve_checkpoint(plan["workspace"], plan["architecture"], plan["run"],
                                        plan["filename"], environ=environ)
    except ServerListRefused as exc:
        why = str(exc)
        note = _record_usage(plan, False, why, environ)
        return LoadStep(None, _outcome("grey", "Not reloaded.", f"Refused: {why} Nothing was done."
                                       + (f" Also, {note}." if note else "")), "", now_sha, False)

    from fastapi.encoders import jsonable_encoder

    from src.api.routes import pipeline as pipeline_routes

    run = reload if reload is not None else pipeline_routes.run_reload
    try:
        answer = jsonable_encoder(run(pipeline_routes.PipelineReloadRequest(
            data_dir=str(directory), checkpoint_path=str(checkpoint))))
    except Exception as exc:  # noqa: BLE001 - a reload that raises is a refused reload
        logger.exception("Pipeline reload failed")
        answer = {"success": False, "message": f"Reload failed ({type(exc).__name__})."}
    served = served_checkpoint_sha(answer.get("status"))
    expected = plan["to"]["sha256"]
    if not answer.get("success"):
        why = str(answer.get("message") or "the reload was refused")
        note = _record_usage(plan, False, why, environ)
        return LoadStep(None, _outcome("grey", "Not reloaded; the model in use is unchanged.",
                                       why + (f" Also, {note}." if note else "")),
                        "", now_sha, False)
    changed = served != now_sha
    if served != expected:
        why = (f"the file's bytes changed since the list was read: the server now serves "
               f"{r.short_hex(served)}, not {r.short_hex(expected)}")
        note = _record_usage(plan, False, why, environ)
        return LoadStep(None, _outcome("amber", "Reloaded, but not with the model shown.",
                                       why + "." + (f" Also, {note}." if note else "")),
                        "", served, changed)
    note = _record_usage(plan, True, None, environ)
    text = f"The model in use for every session is now {r.short_hex(served)}."
    if plan.get("ordering") and note is None:
        text += " The usage history records this load as made from the ordered list."
    elif note:
        text += f" But {note}."
    return LoadStep(None, _outcome("ok", "Reloaded.", text), "", served, changed)


def read_again() -> str:
    """"Read again" after the page stopped waiting: what is served now."""
    status, problem = read_status()
    at = datetime.now().strftime("%H:%M:%S")
    if status is None:
        return _outcome("grey", f"Status read again at {at}:", f"{problem}.")
    sha = served_checkpoint_sha(status)
    return _outcome("grey", f"Status read again at {at}:",
                    f"the model in use is {r.short_hex(sha) if sha else 'none'}.")


# ---------------------------------------------------------------------------
# The component
# ---------------------------------------------------------------------------
#: The page side: row clicks and "Read again" become the component's ``click``
#: event with ``{"key": …}``; Reload starts the page's 180 s count.
MODEL_LIST_JS = r"""
element.addEventListener('click', function (e) {
  var row = e.target.closest ? e.target.closest('[data-key]') : null;
  if (!row || row.getAttribute('aria-disabled') === 'true') { return; }
  trigger('click', {key: row.getAttribute('data-key')});
});
"""

#: Runs in the browser when "Reload with …" is clicked, before the handler.
START_WAIT_JS = r"""
(...args) => {
  var root = document.querySelector('.shep-model-confirm');
  if (root) {
    root.classList.remove('shep-waited');
    var wait = root.querySelector('.shep-load-wait'), out = root.querySelector('.shep-load-timeout');
    if (wait) {
      wait.hidden = false;
      var started = Date.now(), limit = (parseInt(wait.getAttribute('data-wait'), 10) || 180) * 1000;
      var timer = setInterval(function () {
        if (!wait.isConnected) { clearInterval(timer); return; }
        var s = Math.floor((Date.now() - started) / 1000);
        var hh = String(Math.floor(s / 3600)).padStart(2, '0'), mm = String(Math.floor(s / 60) % 60).padStart(2, '0'), ss = String(s % 60).padStart(2, '0');
        var el = wait.querySelector('.shep-elapsed'); if (el) { el.textContent = hh + ':' + mm + ':' + ss; }
        if (Date.now() - started >= limit) { wait.hidden = true; if (out) { out.hidden = false; } root.classList.add('shep-waited'); clearInterval(timer); }
      }, 1000);
    }
  }
  return args;
}
"""


@dataclass
class ModelList:
    """The components of one model list, for the page that places it.

    - ``workspace``: the workspace select (the server's list);
    - ``listing``, ``selected``, ``plan``: this session's listing, selection and
      open confirmation (`gr.State`);
    - ``ordering``: `gr.State`, None, or ``{"version", "version_sha256",
      "metric", "scoring_semantics_digest"}`` when the page orders the list by a
      cohort version's score (Models): a Load then writes a usage history entry;
    - ``phase``: `gr.State`, ``"idle"``, ``"confirming"`` or ``"loading"``
      (Diagnosis disables Run Diagnosis while it is not idle);
    - ``refresh``: keyword arguments for an event that (re)reads the list:
      ``tab.select(**handle.refresh)``;
    - the rendered parts: ``count``, ``rows``, ``facts``, ``reason``,
      ``load_button``, ``confirm``, ``reload_button``, ``keep_button``.
    """

    preset: Preset
    workspace: Any
    listing: Any
    selected: Any
    plan: Any
    ordering: Any
    phase: Any
    show_older: Any
    count: Any
    rows: Any
    facts: Any
    reason: Any
    load_button: Any
    confirm: Any
    reload_button: Any
    keep_button: Any
    refresh: Dict[str, Any]


def create_model_list(ctx: Any, preset: str = "clinical", *, elem_id: str = "shep-models") -> ModelList:
    """Build the shared list in the current layout context and wire its events.

    `ctx` is the page context (`src.webui.shell.PageContext`): its chips are
    refreshed and its ``served_model_changed`` trigger set after a Load."""
    chosen = PRESETS[preset]
    listing = gr.State(empty_listing())
    selected = gr.State(None)
    plan = gr.State(None)
    ordering = gr.State(None)
    phase = gr.State("idle")
    show_older = gr.State(False)
    with gr.Column(elem_id=elem_id, elem_classes=["shep-model-list", f"shep-preset-{preset}"]):
        with gr.Row(elem_classes=["shep-model-head"]):
            workspace = gr.Dropdown(label="Workspace", choices=[], value=None,
                                    elem_id=f"{elem_id}-workspace", interactive=True, scale=3)
            count = plain_html("", elem_id=f"{elem_id}-count")
        with gr.Row(elem_classes=["shep-model-body"]):
            rows = plain_html('<div class="shep-muted">The list is read when it is opened.</div>',
                              elem_id=f"{elem_id}-rows", js_on_load=MODEL_LIST_JS,
                              elem_classes=["shep-model-rows"])
            facts = plain_html(facts_html(empty_listing(), None), elem_id=f"{elem_id}-facts",
                               elem_classes=["shep-model-facts"])
        with gr.Row(elem_classes=["shep-model-foot"]):
            load_button = gr.Button("🔒 Load this model", variant="primary", interactive=False,
                                    elem_id=f"{elem_id}-load", scale=0)
            reason = plain_html("", elem_id=f"{elem_id}-reason")
        with gr.Column(visible=False, elem_classes=["shep-model-confirm-box"]) as confirm:
            confirm_html = plain_html("", elem_id=f"{elem_id}-confirm", js_on_load=MODEL_LIST_JS,
                                      elem_classes=["shep-model-confirm"])
            with gr.Row():
                reload_button = gr.Button("🔒 Reload", variant="primary", scale=0,
                                          elem_id=f"{elem_id}-reload")
                keep_button = gr.Button("Keep", scale=0, elem_id=f"{elem_id}-keep")

    def _refresh(name, older, request: gr.Request):
        data = list_models(name, request)
        choices = _workspace_choices(data.get("workspaces") or [])
        reason_html = (f'<div class="shep-reason shep-reason-grey">{r.e(data["problem"])}</div>'
                       if data.get("refused") else "")
        return (data, None, gr.update(choices=choices, value=data.get("workspace")),
                count_line(data, chosen, older), list_html(data, None, chosen, show_older=older),
                facts_html(data, None),
                gr.update(interactive=False), reason_html)

    refresh_outputs = [listing, selected, workspace, count, rows, facts, load_button, reason]
    refresh = {"fn": _refresh, "inputs": [workspace, show_older], "outputs": refresh_outputs,
               "show_progress": "minimal"}
    workspace.input(**refresh)

    def _select(data, older, evt: gr.EventData):
        raw = getattr(evt, "_data", None)
        key = raw.get("key") if isinstance(raw, Mapping) else None
        if key == "__older__":
            older = True
            return (gr.update(), older, list_html(data, None, chosen, show_older=True),
                    count_line(data, chosen, True), gr.update(), gr.update())
        item = find_item(data, key)
        if item is None:
            return (gr.update(),) * 6
        return (key, older, list_html(data, key, chosen, show_older=older),
                count_line(data, chosen, older), facts_html(data, key),
                gr.update(interactive=loadable(item)))

    rows.click(_select, inputs=[listing, show_older],
               outputs=[selected, show_older, rows, count, facts, load_button],
               show_progress="hidden")

    def _request(data, key, order, request: gr.Request):
        new_plan, html_ = request_load(data, key, order, request)
        if new_plan is None:
            return None, gr.update(visible=False), "", html_, gr.update(), gr.update(), "idle"
        new, old = r.short_hex(new_plan["to"]["sha256"]), r.short_hex(new_plan["from"].get("sha256"))
        return (new_plan, gr.update(visible=True), html_, "",
                gr.update(value=f"🔒 Reload with {new}"),
                gr.update(value=f"Keep {old}" if new_plan["from"].get("sha256") else "Cancel"),
                "confirming")

    load_button.click(_request, inputs=[listing, selected, ordering],
                      outputs=[plan, confirm, confirm_html, reason, reload_button, keep_button,
                               phase]).then(**ctx.chip_refresh)

    def _loading(current):
        return "loading" if current else "idle"

    async def _confirm(current, request: gr.Request):
        # **D8, the status and the reload run here, on the event loop, with
        # nothing awaited before the reload**: it waits behind a running
        # diagnosis and never runs beside one, as the API route's does.
        step = finish_load(current, request)
        token = (f"{step.served}@{datetime.now().isoformat(timespec='seconds')}"
                 if step.changed else gr.update())
        still_open = step.plan is not None
        old = r.short_hex(step.plan["from"].get("sha256")) if still_open else ""
        return (step.plan, gr.update(visible=True), step.confirm_html, step.reason_html, token,
                "confirming" if still_open else "idle",
                gr.update(visible=still_open),
                gr.update(value=(f"Keep {old}" if still_open and old != "—" else
                                 "Cancel" if still_open else "Close")))

    reload_button.click(_loading, inputs=[plan], outputs=[phase], queue=False,
                        js=START_WAIT_JS, show_progress="hidden").then(
        _confirm, inputs=[plan],
        outputs=[plan, confirm, confirm_html, reason, ctx.served_model_changed, phase,
                 reload_button, keep_button],
        show_progress="hidden").then(**ctx.chip_refresh).then(**refresh)

    def _keep():
        return (None, gr.update(visible=False), "", "", "idle", gr.update(visible=True))

    keep_button.click(_keep, inputs=None,
                      outputs=[plan, confirm, confirm_html, reason, phase, reload_button],
                      show_progress="hidden")

    def _read_again(evt: gr.EventData):
        raw = getattr(evt, "_data", None)
        if not (isinstance(raw, Mapping) and raw.get("key") == "__read_again__"):
            return gr.update()
        return read_again()

    confirm_html.click(_read_again, inputs=None, outputs=[reason], show_progress="hidden")

    return ModelList(chosen, workspace, listing, selected, plan, ordering, phase, show_older,
                     count, rows, facts, reason, load_button, confirm, reload_button,
                     keep_button, refresh)


__all__ = [
    "IN_USE_NOTE",
    "LoadStep",
    "MODEL_LIST_JS",
    "ModelList",
    "PAGE_WAIT_SECONDS",
    "PRESETS",
    "Preset",
    "REPLACES_NOTE",
    "confirmation_html",
    "count_line",
    "create_model_list",
    "facts_html",
    "find_item",
    "finish_load",
    "item_key",
    "list_html",
    "list_models",
    "loadable",
    "read_again",
    "recorded_tests_for",
    "request_load",
]
