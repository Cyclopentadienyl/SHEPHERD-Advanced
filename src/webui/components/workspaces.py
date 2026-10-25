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

Module: src/webui/components/workspaces.py
"""
from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.webui.components import renderers as r

#: What replaces an absolute path that is not under the workspaces root.
OUTSIDE_PATH = "a path outside the workspaces root"
#: What a refused root setting reads as (its path is not shown).
ROOT_REFUSED = ("The workspaces root setting (SHEPHERD_WORKSPACES_ROOT) is refused, so the "
                "server lists no workspace until it is corrected on the server.")

#: **The dashboard's one pattern for an absolute path in a text** (M5-3 review
#: ARCH-7: Datasets and Test had a copy each): POSIX (``/…``) or a drive letter
#: (``C:\\…``), not right after a word, a dot, a colon, a slash or a NUL (a root
#: already replaced, `redact_paths`). Where it ends (SECURITY-4: a path ended at
#: its first space, so the rest of ``/mnt/clinic share/MRN 0012345/cases.json``
#: was shown):
#:
#: - after a quote (``'…'``, as Python's messages quote a path), at the
#:   closing quote, spaces included;
#: - otherwise at a space, a quote, a bracket, a comma or a semicolon, but it
#:   runs on through the words after a space up to the last one that holds a
#:   path separator (``share/MRN 0012345/cases.json``), and then takes one more
#:   word that ends in a file extension (``MRN 0012345.json``). A word that
#:   starts another absolute path is never taken; it is replaced on its own.
#:
#: It may take a few words of prose too (``… cannot be read: train/val``):
#: shown less, never a path. A space-holding last part with no separator or
#: extension after it (``/data/Jane Doe``) still ends at its space.
_P = r"""[^\s'"(),;\x00]"""
_NOT_ABSOLUTE = r"""(?![/\\]|[A-Za-z]:[\\/])"""
_ABSOLUTE = re.compile(rf"""
    (?<![\w.:/\\\x00])
    (?:
        (?<=['"]) (?:/[^\s'"/\x00]|[A-Za-z]:[\\/]) [^'"\n\x00]*
      | (?:/[^\s'"(),;/\x00]|[A-Za-z]:[\\/]) {_P}*
        (?: (?:[ \t]+{_NOT_ABSOLUTE}{_P}+)* [ \t]+{_NOT_ABSOLUTE}{_P}*[/\\]{_P}* )?
        (?: [ \t]+{_NOT_ABSOLUTE}[^\s'"(),;\x00/\\]*\.[A-Za-z][A-Za-z0-9]{{0,7}}(?!{_P}) )?
    )""", re.VERBOSE)
#: WP1's words for a link whose target is outside the root.
_RESOLVES_OUTSIDE = re.compile(r"resolves to .+?, outside the workspaces root")


def redact_paths(text: Any, roots: Sequence[Tuple[str, str]], *, other: str) -> str:
    """`text` with no absolute server path left: one under a named server
    directory (`roots`: ``(path, name)``, longest first) is shown under its
    name, any other replaced by `other` (`_ABSOLUTE`). The one redaction every
    page uses, for one root (`redact`) or several (the Test page's)."""
    flat = str(text if text is not None else "")
    marks: Dict[str, str] = {}
    for index, (prefix, token) in enumerate(roots):
        if not prefix:
            continue
        mark = f"\x00{index}\x00"
        flat = re.sub(re.escape(prefix) + r"(?=$|[/\\\s'\"(),;:])", mark, flat)
        marks[mark] = token
    flat = _ABSOLUTE.sub(other, flat)
    for mark, token in marks.items():
        flat = flat.replace(mark, token)
    return flat


def redact(text: Any, root: Optional[Path]) -> str:
    """`text` on one line with no absolute path left: a path under the
    workspaces root is cut to the part inside it, the root itself is "the
    workspaces root", and any other absolute path becomes "a path outside the
    workspaces root" (`redact_paths`)."""
    flat = " ".join(str(text or "").split())
    roots: List[Tuple[str, str]] = []
    if root is not None:
        prefix = str(root)
        flat = flat.replace(f"the workspaces root {prefix}", "the workspaces root")
        flat = _RESOLVES_OUTSIDE.sub("resolves outside the workspaces root", flat)
        flat = re.sub(re.escape(prefix) + r"[/\\]", "", flat)
        roots.append((prefix, "the workspaces root"))
    return redact_paths(flat, roots, other=OUTSIDE_PATH)


def workspaces_root(environ: Optional[Mapping[str, str]] = None
                    ) -> Tuple[Optional[Path], Optional[str]]:
    """The workspaces root, or why there is none: unset (its own words, which
    name no path) or refused (`ROOT_REFUSED`: the setting's path is not shown)."""
    from src.config import server_settings

    setting = server_settings.workspaces_root(environ)
    if setting.available:
        return setting.value, None
    if setting.problem and " is refused:" in setting.problem:
        return None, ROOT_REFUSED
    return None, redact(setting.problem, None)


#: What the manifest binds, by its role names, as the details show them.
BOUND_ROLES = (("kg", "kg"), ("node_features", "node_features"), ("edge_indices", "edge_indices"),
               ("num_nodes", "num_nodes"), ("train_samples", "train_samples"),
               ("val_samples", "val_samples"), ("provenance", "record"))


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
    ``root`` and ``problem`` (why none is listed, without a path)."""
    from src.api.services.server_lists import (
        ServerListRefused,
        list_workspaces,
        resolve_workspaces,
    )

    listing = list_workspaces(environ=environ)
    root, root_problem = workspaces_root(environ)
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
    return {"rows": rows, "directories": directories, "root": root, "problem": problem}


__all__ = [
    "BOUND_ROLES",
    "OUTSIDE_PATH",
    "ROOT_REFUSED",
    "manifest_facts",
    "manifest_state",
    "read_workspaces",
    "redact",
    "redact_paths",
    "workspaces_root",
]
