"""
The server's workspace list as the dashboard shows it (M5-2, UI-A).
===================================================================
One reader for every page that offers or shows a workspace: Datasets, the
shared model list's workspace select (Diagnosis' drawer, Models), Training's
run bar and Test's select. It reads WP1's list
(`server_lists.list_workspaces`) and resolves every listed name with **one
listing of the root** (`server_lists.resolve_workspaces`), and adds what the
dashboard shows besides:

- ``manifest``: whether the workspace's ``split_manifest.json`` was read
  (``read``), is absent (``absent``: "not trainable: no split manifest"), is
  there but not readable (``unreadable``: "no readable split manifest"), or was
  not reached (``not_checked``: a name that resolves outside the root, whose own
  reason is shown);
- **no absolute path in any reason** (`redact`): a path under the root is cut
  to the part inside it, the root itself reads "the workspaces root", any other
  absolute path (a link's target outside the root) reads "a path outside the
  workspaces root", and a refused root setting is said without its path
  (`ROOT_REFUSED`). Datasets is the same page for a read-only request, and the
  wording is one for every page.
- the server directories any shown text may name (`path_roots`) and the
  one redaction of a text that may name any of them (`redact_server`): the
  Test page's job output and Models' notes problems read the same pair.

**The notices** (the WebUI's first-use messages, said once per screen): a
setting's state in parts (`server_settings.Notice`), chosen by the reader's
``unset`` flag, never by its words: `setting_notice` (the workspaces folder or
the results registry: unset, a normal setup step; set but refused, its own
short words and where to see why), `lock_notice` (the GPU lock: unset, which
follows from the workspaces folder; another problem, said to an operator) and
`read_workspaces`' ``notice``, from the read it already makes. Each is plain
data for `renderers.notice_block`, `notices_block` and `see_above`.

It also holds the one way a page says whether a registered cohort version was
mapped against a workspace's graph (`workspace_binding`, `binding_state`,
`version_row`): the Test page's version select and the Diagnosis panel's
reference cohorts read the same comparison, the registry's. **Bound** says
two facts together: the version's record names this graph, and the files it
keeps were read and are the bytes that record binds (`contents`, the
registry's `kept_contents`); a version whose files changed after its import
is never bound, and one whose files cannot be checked now is said so.

Module: src/webui/components/workspaces.py
"""
from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.config.server_settings import (
    DATA_LOCATIONS,
    ENV_REGISTRY,
    ENV_WORKSPACES_ROOT,
    GPU_LOCK_NOTICE,
    REGISTRY_NOTICE,
    WORKSPACES_ROOT_NOTICE,
    Notice,
)
from src.utils.path_redaction import SERVER_PATH, redact_paths
from src.webui.components import renderers as r

#: What replaces an absolute path that is not under the workspaces root.
OUTSIDE_PATH = "a path outside the workspaces root"
#: What a refused root setting reads as (its path is not shown).
ROOT_REFUSED = ("The workspaces folder set for this service is refused, so the server lists "
                "no workspace until it is corrected; Settings › Data locations says why. Ref: "
                "SHEPHERD_WORKSPACES_ROOT")

#: The whitespace `redact` joins: plain spaces, tabs and line breaks (a line
#: or record separator among them), never a space a name may hold.
_JOINED = re.compile("[\t\n\x0b\x0c\r\x1c-\x1f \x85\u2028\u2029]+")
#: WP1's words for a link whose target is outside the root.
_RESOLVES_OUTSIDE = re.compile(r"resolves to .+?, outside the workspaces root")


def redact(text: Any, root: Optional[Path]) -> str:
    """`text` on one line with no absolute path left: a path under the
    workspaces root is cut to the part inside it, the root itself is "the
    workspaces root", and any other absolute path becomes "a path outside the
    workspaces root" (`redact_paths`). Its lines and runs of plain
    whitespace are joined into one space (`_JOINED`); any other space is a
    name's own and stays as it is (a no-break space, an ideographic space)."""
    flat = _JOINED.sub(" ", str(text or "")).strip(" ")
    roots: List[Tuple[str, str]] = []
    if root is not None:
        prefix = str(root)
        flat = flat.replace(f"the workspaces root {prefix}", "the workspaces root")
        flat = _RESOLVES_OUTSIDE.sub("resolves outside the workspaces root", flat)
        roots += [(prefix + "/", ""), (prefix + "\\", ""), (prefix, "the workspaces root")]
    return redact_paths(flat, roots, other=OUTSIDE_PATH)


def shown_path(path: Any, root: Optional[Path]) -> str:
    """**One server path as a page shows it** (D7), a path and not a text
    that may name one (that is `redact`'s): inside the workspaces root, its
    path there as it is, every name whole ("demo (copy)/runs/…", an emoji, a
    no-break space); the root itself "the workspaces root"; any other
    absolute path `OUTSIDE_PATH`, never the path itself. It is resolved
    first, so a link into the root reads as the file it names there and a
    link out of it as outside. A relative path names no server place of its
    own and is shown as it is."""
    text = str(path if path is not None else "")
    if not text or not Path(text).is_absolute():
        return text
    if root is not None:
        try:
            resolved, base = Path(text).resolve(), Path(root).resolve()
            return ("the workspaces root" if resolved == base
                    else resolved.relative_to(base).as_posix())
        except (OSError, RuntimeError, ValueError):
            pass
    return OUTSIDE_PATH


def path_roots(environ: Optional[Mapping[str, str]] = None) -> List[Tuple[str, str]]:
    """The server directories a shown text may name, with the name each is
    shown under, longest first: the registry, the workspaces root, the intake
    directory, each evidence directory and the repository."""
    from src.config import server_settings

    roots: List[Tuple[str, str]] = []
    for reader, token in ((server_settings.cohort_registry, "<registry>"),
                          (server_settings.workspaces_root, "<workspaces>"),
                          (server_settings.intake_directory, "<intake>")):
        try:
            setting = reader(environ)
        except Exception:  # noqa: BLE001 - a root not read is a path shown as "a server path"
            continue
        if setting.available:
            roots.append((str(setting.value), token))
    try:
        evidence = server_settings.training_evidence_dirs(environ)
        for index, directory in enumerate(evidence.value or (), start=1):
            roots.append((str(directory), f"<evidence {index}>"))
    except Exception:  # noqa: BLE001
        pass
    roots.append((str(server_settings.REPO_ROOT), "<repository>"))
    return sorted(roots, key=lambda pair: -len(pair[0]))


def redact_server(text: Any, roots: Sequence[Tuple[str, str]]) -> str:
    """`text` with no absolute server path left (any server directory, not
    only the workspaces root, which is `redact`'s): one under a named server
    directory is shown under its name (``<registry>/versions/…``), any other
    as `SERVER_PATH` (the dashboard's one redaction, `redact_paths`): the
    Test page's job output and a record's notes problems alike."""
    return redact_paths(text, roots, other=SERVER_PATH)


def _root_of(setting: Any) -> Tuple[Optional[Path], Optional[str]]:
    """A workspaces root setting as `workspaces_root` says it."""
    if setting.available:
        return setting.value, None
    if setting.problem and " is refused:" in setting.problem:
        return None, ROOT_REFUSED
    return None, redact(setting.problem, None)


def workspaces_root(environ: Optional[Mapping[str, str]] = None
                    ) -> Tuple[Optional[Path], Optional[str]]:
    """The workspaces root, or why there is none: unset (its own words, which
    name no path) or refused (`ROOT_REFUSED`: the setting's path is not shown)."""
    from src.config import server_settings

    return _root_of(server_settings.workspaces_root(environ))


# ---------------------------------------------------------------------------
# The notices (module docstring)
# ---------------------------------------------------------------------------
#: The notice of each setting a page says a notice for, by its variable.
NOTICES = {ENV_WORKSPACES_ROOT: WORKSPACES_ROOT_NOTICE, ENV_REGISTRY: REGISTRY_NOTICE}
#: **A setting that is set and cannot be used** (refused by its check, or its
#: saved file not readable): its state in short words, by its variable. Why is
#: said in Settings › Data locations, to an operator, with its path; a page
#: says where to see it (`SEE_WHY`), never the setting's own words.
NOT_USABLE = {ENV_WORKSPACES_ROOT: "The workspaces folder cannot be used.",
              ENV_REGISTRY: "The results registry cannot be used."}
#: The GPU lock set and not usable (a problem that is not "unset").
LOCK_NOT_USABLE = "The GPU lock cannot be used."
#: Where an operator sees why a setting that is set cannot be used.
SEE_WHY = f"See why in {DATA_LOCATIONS}"


def notice_view(notice: Notice, kind: str = "neutral") -> Dict[str, Any]:
    """`notice` as the plain data a page keeps and `renderers.notice_block`
    draws: its fields, and its `kind` (``neutral``: a missing setting, a
    normal setup step, never red; ``error``: a value set and refused)."""
    return {**notice._asdict(), "details": tuple(notice.details), "ref": tuple(notice.ref),
            "kind": kind}


def _short(summary: str) -> str:
    return summary[:1].lower() + summary[1:].rstrip(".")


def _not_usable(base: Notice, summary: str, details: Tuple[str, ...] = ()) -> Dict[str, Any]:
    """A notice for a setting that is set and cannot be used: its own short
    words, the same key, place and reference, and where to see why."""
    return {**notice_view(base, "error"), "summary": summary, "short": _short(summary),
            "action": SEE_WHY, "details": details}


def setting_notice(setting: Any) -> Optional[Dict[str, Any]]:
    """**The notice of a setting as read now** (`server_settings.Setting`):
    None while it is usable; unset, its notice (`NOTICES`, kind ``neutral``);
    set and not usable, its short words (`NOT_USABLE`, kind ``error``) and
    where to see why, never the setting's value or path. Chosen by
    ``unset`` and ``available``, never by the words."""
    if setting is None or setting.available:
        return None
    base = NOTICES.get(setting.variable)
    if base is None:
        return None
    if setting.unset:
        return notice_view(base)
    return _not_usable(base, NOT_USABLE[setting.variable])


def lock_notice(lock: Optional[Mapping[str, Any]], *, operator: bool = False
                ) -> Optional[Dict[str, Any]]:
    """**The GPU lock's notice** (`gpu_lock.lock_state`'s answer): None while
    it can be used; unset (no workspaces folder), `GPU_LOCK_NOTICE`, which
    follows from the workspaces folder's (its ``cause``); another problem,
    `LOCK_NOT_USABLE`, with the problem itself as a Details row for an
    operator only (it may name a server path), as the header chip says it."""
    if not lock or not lock.get("problem"):
        return None
    if lock.get("unset"):
        return notice_view(GPU_LOCK_NOTICE)
    details = (f"Why: {' '.join(str(lock['problem']).split())}",) if operator else ()
    return _not_usable(GPU_LOCK_NOTICE, LOCK_NOT_USABLE, details)


#: What the manifest binds: each file's role in the manifest and its plain
#: name, as the details show it ("Manifest contents", the WebUI skill's §5:
#: no file or field name as a label).
BOUND_ROLES = (("kg", "Graph"), ("node_features", "Node features"), ("edge_indices", "Edge index"),
               ("num_nodes", "Node counts"), ("train_samples", "Train samples"),
               ("val_samples", "Val samples"), ("provenance", "Dataset record"))


def manifest_facts(directory: Path, root: Optional[Path]
                   ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """What a listed workspace's manifest binds and how it was split, read once
    (`read_split_manifest`) from its resolved directory; or why it was not read
    (without an absolute path). Datasets' details and Training's run bar both
    show it (`renderers.dataset_details_html`)."""
    from src.kg.artifacts import read_split_manifest

    try:
        manifest = read_split_manifest(directory).manifest
    except (ValueError, OSError) as exc:
        return None, redact(exc, root)

    def at(*keys: str) -> Any:
        value: Any = manifest
        for key in keys:
            value = value.get(key) if isinstance(value, Mapping) else None
        return value

    artifacts = at("artifacts") if isinstance(at("artifacts"), Mapping) else {}
    return {
        "bound": [(label, artifacts.get(role)) for role, label in BOUND_ROLES],
        "allocation_seed": at("allocation", "seed"),
        "feature_seed": at("graph_export", "feature_seed"),
        "train_diseases": at("realised", "train_diseases"),
        "val_diseases": at("realised", "val_diseases"),
    }, None


def manifest_state(row: Mapping[str, Any], directories: Mapping[str, Path]) -> str:
    """``read``, ``absent``, ``unreadable`` or ``not_checked`` (module docstring)."""
    if r.is_digest(row.get("dataset_fingerprint")):
        return "read"
    directory = directories.get(row.get("name"))
    if directory is None:
        return "not_checked"
    try:
        return "unreadable" if (directory / "split_manifest.json").is_file() else "absent"
    except OSError:
        return "unreadable"


def read_workspaces(environ: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """The listed workspaces as plain data: ``rows`` (each item's fields, its
    ``manifest`` state, every reason without an absolute path), ``directories``
    (each listed name's resolved directory, from one listing of the root),
    ``root``, ``problem`` (why none is listed, without a path) and ``notice``
    (the workspaces folder's notice, `setting_notice`, from the same read of
    the setting; None while the folder is usable)."""
    from src.api.services.server_lists import (
        ServerListRefused,
        list_workspaces,
        resolve_workspaces,
    )

    from src.config import server_settings

    listing = list_workspaces(environ=environ)
    setting = server_settings.workspaces_root(environ)
    root, root_problem = _root_of(setting)
    try:
        directories = resolve_workspaces(environ=environ)
    except ServerListRefused:
        directories = {}
    rows = []
    for item in listing.items:
        row = asdict(item)
        row["not_trainable"] = redact(row["not_trainable"], root) if row["not_trainable"] else None
        record = dict(row.get("dataset_record") or {})
        if record.get("problem"):
            record["problem"] = redact(record["problem"], root)
        row["dataset_record"] = record
        build = dict(row.get("build") or {})
        if build.get("reason"):
            build["reason"] = redact(build["reason"], root)
        row["build"] = build
        row["manifest"] = manifest_state(row, directories)
        rows.append(row)
    problem = None
    if listing.problem:
        problem = root_problem if root is None and root_problem else redact(listing.problem, root)
    return {"rows": rows, "directories": directories, "root": root, "problem": problem,
            "notice": setting_notice(setting)}


# ---------------------------------------------------------------------------
# A cohort version against a workspace (the Test page; the Diagnosis panel's
# reference cohorts): one comparison, the registry's
# ---------------------------------------------------------------------------
def version_row(item: Any) -> Dict[str, Any]:
    """A listed cohort version as plain data for a page's state: its fields
    without the report store's runs, which a page reads on their own."""
    row = asdict(item)
    row.pop("reports", None)
    row.pop("reports_problem", None)
    return row


def workspace_binding(directory: Optional[Path], row: Optional[Mapping[str, Any]]
                      ) -> Dict[str, Any]:
    """What a cohort version is compared with to say whether it was mapped
    against this workspace's graph: the four graph digests its manifest binds
    (one read of ``split_manifest.json``) and its verified dataset record's
    digest. A display hint only: the run compares the bytes it reads."""
    from src.kg.artifacts import GRAPH_ARTIFACTS, read_split_manifest

    if directory is None or row is None:
        return {"graph": None, "record_sha256": None, "problem": "no workspace is chosen"}
    try:
        artifacts = read_split_manifest(directory).manifest.get("artifacts")
    except (ValueError, OSError):
        return {"graph": None, "record_sha256": None,
                "problem": "its split_manifest.json cannot be read"}
    artifacts = artifacts if isinstance(artifacts, Mapping) else {}
    record = row.get("dataset_record") or {}
    sha = record.get("sha256") if record.get("binding") == "verified" else None
    return {"graph": {role: artifacts.get(role) for role in GRAPH_ARTIFACTS},
            "record_sha256": sha,
            "problem": None if sha else "its dataset record is not verified"}


#: The states of a version's kept files (`cohort_registry.kept_contents`)
#: that a run on it would refuse: other bytes, or a file that is not there.
CHANGED_CONTENTS = ("differs", "missing")


def contents_reason(version: Mapping[str, Any]) -> Optional[str]:
    """Why a version's kept files are not known to be its record's bytes: the
    first kept file's reason that is not verified (`contents`, naming the
    file, never a path), or None when every file was verified or none was
    checked."""
    contents = version.get("contents")
    files = contents.get("files") if isinstance(contents, Mapping) else None
    return next((str(f.get("reason")) for f in files or ()
                 if isinstance(f, Mapping) and f.get("state") != "verified"), None)


def binding_state(version: Mapping[str, Any], binding: Optional[Mapping[str, Any]]
                  ) -> Tuple[str, str]:
    """``("bound" | "another" | "changed" | "unknown", words)``: whether
    `version` was mapped against the workspace `binding` describes, by the
    registry's one comparison (`verify_cohort_binding`, R3's), and whether
    the files it keeps are the bytes its record binds (its ``contents``, the
    registry's `kept_contents`):

    - ``changed``: a kept file is other bytes or not there, whatever the
      graph: a run would refuse it, so it is never bound (`contents_reason`
      names the file);
    - ``another``: its record names another graph;
    - ``bound``: its record names this graph and its files were checked;
    - ``unknown``: it cannot be read, no workspace is chosen, or its record
      names this graph and its files cannot be checked now (or were not).
    """
    from src.evaluation.cohort_registry import CohortBindingMismatch, verify_cohort_binding

    if version.get("problem"):
        return "unknown", "the version cannot be read"
    contents = version.get("contents") if isinstance(version.get("contents"), Mapping) else {}
    if contents.get("state") in CHANGED_CONTENTS:
        return "changed", "not bound: its files changed after the import"
    if not binding or binding.get("problem") or not binding.get("graph"):
        why = (binding or {}).get("problem") or "no workspace is chosen"
        return "unknown", f"not checked against this workspace: {why}"
    try:
        verify_cohort_binding({"graph": version.get("graph"),
                               "dataset_record": version.get("dataset_record"),
                               "label": version.get("label")},
                              binding["graph"], binding["record_sha256"])
    except CohortBindingMismatch as exc:
        return "another", f"mapped against another graph ({', '.join(exc.roles)} differ)"
    if contents.get("state") == "verified":
        return "bound", "bound to this graph"
    if contents.get("state") == "unreadable":
        return "unknown", ("mapped against this graph; its files cannot be checked now: "
                           f"{contents_reason(version)}")
    return "unknown", "mapped against this graph; its files are not checked"


__all__ = [
    "BOUND_ROLES",
    "CHANGED_CONTENTS",
    "LOCK_NOT_USABLE",
    "NOTICES",
    "NOT_USABLE",
    "OUTSIDE_PATH",
    "ROOT_REFUSED",
    "SEE_WHY",
    "binding_state",
    "contents_reason",
    "lock_notice",
    "manifest_facts",
    "manifest_state",
    "notice_view",
    "read_workspaces",
    "redact",
    "setting_notice",
    "shown_path",
    "version_row",
    "workspace_binding",
    "workspaces_root",
]
