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
from typing import Any, Dict, Sequence, Tuple

#: **The one pattern for an absolute path in a text** (M5-3 review
#: ARCH-7: Datasets and Test had a copy each): POSIX (``/…``) or a drive letter
#: (``C:\\…``), not right after a word, a dot, a colon (``http://``), a slash or a
#: NUL. Where it ends (SECURITY-4: a path ended at its first space, so the rest
#: of ``/mnt/clinic share/MRN 0012345/cases.json`` was shown):
#:
#: - after a quote (``'…'``, as Python's messages quote a path), at the
#:   closing quote, spaces included;
#: - otherwise at a space, a quote, a bracket, a comma or a semicolon (a
#:   bracketed part with no space inside a name is taken: ``MRN(0012345)/x``),
#:   but it runs on through the words after a space up to the last one that
#:   holds a path separator
#:   (``share/MRN 0012345/cases.json``), and then takes one more word that ends
#:   in a file extension (``MRN 0012345.json``). A word that starts another
#:   absolute path is never taken; it is replaced on its own.
#:
#: It may take a few words of prose too (``… cannot be read: train/val``):
#: shown less, never a path. A space-holding last part with no separator or
#: extension after it (``/data/Jane Doe``) still ends at its space, and so does
#: a bracketed part with a space inside it (``/data/Jane (MRN 1)/x.json``: its
#: bracket on is shown).
#:
#: **A path under a named directory ends by the same rules** (`_ROOTED`): the
#: directory is a mark (`redact_paths`), and the rest of that path is read with
#: it, as one match the scan keeps, so no name inside it starts another path
#: (``<workspaces>/demo (copy)/runs/…``, ``ws🧬/runs/…``).
_P = r"""[^\s'"(),;\x00]"""
#: A path's character, or a bracketed part with no space inside it.
_G = rf"""(?:{_P}|\({_P}*\))"""
_NOT_ABSOLUTE = r"""(?![/\\]|[A-Za-z]:[\\/])"""
#: The rest of a path after its first character, or after a named directory.
_REST = rf"""
    {_G}*
    (?: (?:[ \t]+{_NOT_ABSOLUTE}{_G}+)* [ \t]+{_NOT_ABSOLUTE}{_G}*[/\\]{_G}* )?
    (?: [ \t]+{_NOT_ABSOLUTE}[^\s'"(),;\x00/\\]*\.[A-Za-z][A-Za-z0-9]{{0,7}}(?!{_P}) )?"""
_ABSOLUTE = rf"""
    (?<![\w.:/\\\x00])
    (?:
        (?<=['"]) (?:/[^\s'"/\x00]|[A-Za-z]:[\\/]) [^'"\n\x00]*
      | (?:/[^\s'"(),;/\x00]|[A-Za-z]:[\\/]) {_REST}
    )"""
#: A named directory's mark (`redact_paths`) and the rest of its path.
_ROOTED = rf"""
    (?<=['"]) \x00\d+\x00 [^'"\n\x00]*
  | \x00\d+\x00 {_REST}"""
_SCAN = re.compile(rf"(?P<rooted>{_ROOTED})|{_ABSOLUTE}", re.VERBOSE)


def redact_paths(text: Any, roots: Sequence[Tuple[str, str]], *, other: str) -> str:
    """`text` with no absolute server path left: one under a named server
    directory (`roots`: ``(path, name)``, longest first) is shown under its
    name, any other replaced by `other` (`_SCAN`). A path that ends in a
    separator names the directory's inside: what follows it is shown under
    `name` as it is (an empty name: the path inside it, the dashboard's
    ``<workspace>/runs/…``). The rest of a path under a named directory is
    read with it (`_ROOTED`), so no name in it starts another path. The one
    redaction every surface uses, for one root (the dashboard's
    `workspaces.redact`) or several (the Test page's, the results service's)."""
    flat = str(text if text is not None else "")
    marks: Dict[str, str] = {}
    for index, (prefix, token) in enumerate(roots):
        if not prefix:
            continue
        mark = f"\x00{index}\x00"
        bound = "" if prefix.endswith(("/", "\\")) else r"(?=$|[/\\\s'\"(),;:])"
        flat = re.sub(re.escape(prefix) + bound, mark, flat)
        marks[mark] = token
    flat = _SCAN.sub(lambda m: m.group(0) if m.group("rooted") is not None else other, flat)
    for mark, token in marks.items():
        flat = flat.replace(mark, token)
    return flat


__all__ = ["redact_paths"]
