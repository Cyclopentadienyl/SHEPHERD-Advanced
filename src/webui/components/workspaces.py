"""
The server's workspace list as the dashboard shows it (M5-2, UI-A).
===================================================================
One reader for every page that offers or shows a workspace: Datasets, the
shared model list's workspace select (Diagnosis' drawer, Models), and next
Training's run bar and Test's select. It reads WP1's list
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
from typing import Any, Dict, Mapping, Optional, Tuple

from src.webui.components import renderers as r

#: What replaces an absolute path that is not under the workspaces root.
OUTSIDE_PATH = "a path outside the workspaces root"
#: What a refused root setting reads as (its path is not shown).
ROOT_REFUSED = ("The workspaces root setting (SHEPHERD_WORKSPACES_ROOT) is refused, so the "
                "server lists no workspace until it is corrected on the server.")

#: An absolute path in a reason: POSIX (``/…``) or a drive letter (``C:\\…``),
#: up to a space, a quote, a bracket, a comma or a semicolon.
_ABSOLUTE = re.compile(r"""(?<![\w.:/\\])(?:/[^\s'"(),;/]|[A-Za-z]:[\\/])[^\s'"(),;]*""")
#: WP1's words for a link whose target is outside the root.
_RESOLVES_OUTSIDE = re.compile(r"resolves to .+?, outside the workspaces root")


def redact(text: Any, root: Optional[Path]) -> str:
    """`text` on one line with no absolute path left: a path under the
    workspaces root is cut to the part inside it, the root itself is "the
    workspaces root", and any other absolute path becomes "a path outside the
    workspaces root"."""
    flat = " ".join(str(text or "").split())
    if root is not None:
        prefix = str(root)
        flat = flat.replace(f"the workspaces root {prefix}", "the workspaces root")
        flat = _RESOLVES_OUTSIDE.sub("resolves outside the workspaces root", flat)
        flat = re.sub(re.escape(prefix) + r"[/\\]", "", flat)
        flat = re.sub(re.escape(prefix) + r"(?![\w.-])", "the workspaces root", flat)
    return _ABSOLUTE.sub(OUTSIDE_PATH, flat)


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
    "OUTSIDE_PATH",
    "ROOT_REFUSED",
    "manifest_state",
    "read_workspaces",
    "redact",
    "workspaces_root",
]
