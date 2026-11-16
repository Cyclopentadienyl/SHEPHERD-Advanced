"""
Absolute server paths out of a text: the one redaction every surface uses.
=========================================================================
D7 (`docs/working/PLAN_TEST_RESULTS.md`): no page, answer or export shows a
server path. A problem's words may name one (an `OSError`'s file name, a
refusal naming the directory it checked), so a surface that shows such words
passes them through `redact_paths` first: a path under a named directory is
shown under that directory's name, any other absolute path replaced. Moved
here from the dashboard (`src/webui/components/workspaces.py`, which every
page used) in research comparison Stage 8, when the results service
(`src/api/services/results.py`) became its second client.

Standard library only, no I/O.

Module: src/utils/path_redaction.py
"""
from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

#: The words every surface uses for a server path it does not name.
SERVER_PATH = "a server path"

#: **The one pattern for an absolute path in a text** (M5-3 review
#: ARCH-7: Datasets and Test had a copy each): POSIX (``/…``) or a drive letter
#: (``C:\\…``), not right after a word, a dot, a colon (``http://``), a slash or a
#: NUL. Where it ends (SECURITY-4: a path ended at its first space, so the rest
#: of ``/mnt/clinic share/MRN 0012345/cases.json`` was shown):
#:
#: - after a quote (``'…'``, as Python's messages quote a path), at the
#:   closing quote, spaces included;
#: - otherwise at a space, a quote, a bracket, a comma or a semicolon (a
#:   bracketed part of a name is taken, spaces inside it included:
#:   ``MRN(0012345)/x``, ``Jane (MRN 1)/x.json``), but it runs on through the
#:   words after a space up to the last one that holds a path separator
#:   (``share/MRN 0012345/cases.json``), and then takes one more word that ends
#:   in a file extension (``MRN 0012345.json``). A word that starts another
#:   absolute path is never taken; it is replaced on its own.
#: - **never at a sentence's punctuation** (``.``, ``:``, ``!`` or ``?``): a
#:   path's last character is none of them, so ``Cannot read /data/x.json:
#:   denied`` keeps its colon after the replacement, as a path under a named
#:   directory does.
#:
#: It may take a few words of prose too (``… cannot be read: train/val``):
#: shown less, never a path. A space-holding last part with no separator or
#: extension after it (``/data/Jane Doe``, ``/data/Jane (MRN 1)``) still ends
#: at its space; a bracket that only follows a path after a space ("…/x.json
#: (permission denied)") is prose and stays.
#:
#: **A path under a named directory ends by the same rules** (`_ROOTED`): the
#: directory is a mark (`redact_paths`), and the rest of that path is read with
#: it, as one match the scan keeps, so no name inside it starts another path
#: (``<workspaces>/demo (copy)/runs/…``, ``ws🧬/runs/…``).
#:
#: **A named directory's text inside a longer path is that path's own**
#: (`_MARK` in `_G`, `_SEP` and `_QUOTED`): a path that runs into a mark goes
#: on through it as through a separator, so ``/mnt/old/srv/ws/Jane (MRN 1)/x``
#: is one path, replaced whole, not "a server path" then ``<workspaces>/Jane
#: (MRN 1)/x``, and ``/srv/clinic/api/v1/Jane Doe/x`` is not kept from
#: ``/api/v1/`` on (the service's routes, `src.api.operator_access`). A mark
#: no path runs into starts its own path, as before (``see /srv/ws/demo``,
#: ``'/srv/ws/x y'``, ``(see above)/srv/ws/x``).
_P = r"""[^\s'"(),;\x00]"""
#: A named directory's mark (`redact_paths`).
_MARK = r"""\x00\d+\x00"""
#: A path's character, or a bracketed part of a name (no bracket inside it;
#: spaces and tabs may be: ``Jane (MRN 1)/x.json``, B72-N5), or a mark.
_G = rf"""(?:{_P}|\((?:{_P}|[ \t])*\)|{_MARK})"""
#: A separator inside a path: a slash, a backslash, or a mark (a named
#: directory's own path starts at a separator).
_SEP = rf"""(?:[/\\]|{_MARK})"""
#: The rest of a quoted path, up to its closing quote.
_QUOTED = rf"""(?:[^'"\n\x00]|{_MARK})*"""
#: A word that starts another absolute path, or a named directory's.
_NOT_ABSOLUTE = r"""(?![/\\\x00]|[A-Za-z]:[\\/])"""
#: What a word ending in a file extension may be followed by: nothing of a
#: path, after a sentence's punctuation if any (``MRN 0012345.json: denied``).
_ENDS = rf"""(?=[.:!?]*(?!{_P}))"""
#: The rest of a path after its first character, or after a named directory.
_REST = rf"""
    {_G}*
    (?: (?:[ \t]+{_NOT_ABSOLUTE}{_G}+)* [ \t]+{_NOT_ABSOLUTE}{_G}*{_SEP}{_G}* )?
    (?: [ \t]+{_NOT_ABSOLUTE}(?:[^\s'"(),;\x00/\\]|\((?:[^'"(),;\x00\n/\\])*\))*
        \.[A-Za-z][A-Za-z0-9]{{0,7}}{_ENDS} )?"""
_ABSOLUTE = rf"""
    (?<![\w.:/\\\x00])
    (?:
        (?<=['"]) (?:/[^\s'"/\x00]|[A-Za-z]:[\\/]) {_QUOTED}
      | (?:/[^\s'"(),;/\x00]|[A-Za-z]:[\\/]) {_REST} (?<![.:!?])
    )"""
#: A named directory's mark (`redact_paths`) and the rest of its path.
_ROOTED = rf"""
    (?<=['"]) {_MARK} {_QUOTED}
  | {_MARK} {_REST}"""
_SCAN = re.compile(rf"(?P<rooted>{_ROOTED})|{_ABSOLUTE}", re.VERBOSE)
#: The named directory's mark a `_ROOTED` match starts with.
_LEAD = re.compile(_MARK)


def redact_paths(text: Any, roots: Sequence[Tuple[str, Optional[str]]], *, other: str) -> str:
    """`text` with no absolute server path left: one under a named server
    directory (`roots`: ``(path, name)``, longest first) is shown under its
    name, any other replaced by `other` (`_SCAN`). A path that ends in a
    separator names the directory's inside: what follows it is shown under
    `name` as it is (an empty name: the path inside it, the dashboard's
    ``<workspace>/runs/…``). The rest of a path under a named directory is
    read with it (`_ROOTED`), so no name in it starts another path. The one
    redaction every surface uses, for one root (the dashboard's
    `workspaces.redact`) or several (the Test page's, the results service's).

    **A directory named None is known, not shown** (an open route's answer to
    a request D8 does not allow, `src.api.operator_access.answer_for`): its
    text is found exactly, spaces included (``/srv/Jane (MRN 1) ws, so…``,
    where the pattern alone ends a path at its last part's space), and it and
    the rest of any path under it are replaced whole by `other`."""
    flat = str(text if text is not None else "")
    marks: Dict[str, str] = {}
    hidden = set()
    for index, (prefix, token) in enumerate(roots):
        if not prefix:
            continue
        mark = f"\x00{index}\x00"
        bound = "" if prefix.endswith(("/", "\\")) else r"(?=$|[/\\\s'\"(),;:])"
        flat = re.sub(re.escape(prefix) + bound, mark, flat)
        if token is None:
            hidden.add(mark)
        else:
            marks[mark] = token

    def replaced(match: "re.Match[str]") -> str:
        if match.group("rooted") is None:
            return other
        lead = _LEAD.search(match.group(0))
        return other if lead is not None and lead.group(0) in hidden else match.group(0)

    flat = _SCAN.sub(replaced, flat)
    for mark in hidden:
        flat = flat.replace(mark, other)
    for mark, token in marks.items():
        flat = flat.replace(mark, token)
    return flat


def redact_value(value: Any, roots: Sequence[Tuple[str, Optional[str]]], *,
                 other: str) -> Any:
    """`value`, an answer's JSON-ready data, with every text in it through
    `redact_paths`, an object's keys included; every other value as it is (a
    tuple is answered as a list). The one form for a structure: a recorded
    component's facts (`run_environment.recorded_components`) and what an open
    read route answers a request D8 does not allow
    (`src.api.operator_access.answer_for`)."""
    if isinstance(value, str):
        return redact_paths(value, roots, other=other)
    if isinstance(value, Mapping):
        return {redact_value(key, roots, other=other): redact_value(item, roots, other=other)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_value(item, roots, other=other) for item in value]
    return value


__all__ = ["SERVER_PATH", "redact_paths", "redact_value"]
