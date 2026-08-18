"""Verbatim section extraction from the specs.

One implementation, shared by everything that quotes a normative section —
`affected_sections.py` here, and the submission generator that lives outside this
repository. It is a module rather than a copy in each caller because the last
time the rule lived inside one script it was wrong there and nowhere else could
notice.

**The defect this exists to prevent.** A block used to end at the next heading of
*any* level (`^#{2,3} \\d`), so extracting `## 11` stopped at `### 11.1` and
dropped that section's own subsections. The submitted freeze artefact contained
§11's preamble and invariant table only; the reviewer read exactly what was sent
and correctly reported the test-suite shape and reopen conditions as absent. They
were in the spec. Comparing the two rules over all five specs afterwards found
**13 sections** where they disagree, three of which would have been emitted as two
lines.

The rule now: a block ends at the first heading whose level is **less than or
equal to** the starting heading's. Never at a deeper one. No Markdown parser —
headings here are ATX only, at the start of a line, which a regex settles.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path

HEADING = re.compile(r'^(#{1,6}) +(.*)$')


def _find_start(lines: Sequence[str], section: str) -> tuple[int, int]:
    """Return ``(index, level)`` of the heading that opens ``section``.

    Matches `## 7.3 Atomic publication` and `#### 5.1.1 The minimum invariants`
    alike — level 1 to 6, so a section is never unfindable because of its depth.
    The trailing `[.\\s]` keeps `11` from matching `11.1`.
    """
    pattern = re.compile(rf'^(#{{1,6}}) +{re.escape(section)}[.\s]')
    for index, line in enumerate(lines):
        match = pattern.match(line)
        if match:
            return index, len(match.group(1))
    raise KeyError(section)


def extract(path: Path, section: str) -> str:
    """The section and **all of its subsections**, verbatim."""
    lines = path.read_text().splitlines()
    start, level = _find_start(lines, section)

    end = len(lines)
    for index in range(start + 1, len(lines)):
        match = HEADING.match(lines[index])
        if match and len(match.group(1)) <= level:
            end = index
            break
    return "\n".join(lines[start:end]).rstrip()


def heading_tree(block: str) -> list[str]:
    """Every heading inside an extracted block, in order, indented by depth.

    Printed beside each block in the closure bundle so a truncation is visible to
    a reader rather than only to whoever re-runs the generator.
    """
    found = [
        (len(match.group(1)), match.group(2))
        for match in (HEADING.match(line) for line in block.splitlines())
        if match
    ]
    if not found:
        return []
    # Indent relative to the block's own root, not to the document's. A block
    # that starts at `###` should read as a root, not as something four spaces in.
    root = min(depth for depth, _ in found)
    return [f"{'  ' * (depth - root)}{'#' * depth} {title}" for depth, title in found]


def file_sha256(path: Path) -> str:
    """Raw content digest of a normative file. Bytes, nothing else."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def between(path: Path, start: str, end: str) -> str:
    """The lines from the one beginning with ``start`` up to (not including) the
    one beginning with ``end``. Verbatim.

    Quoting a whole section when three paragraphs changed is how a submission
    grows without anyone deciding that it should. This quotes the passage that
    actually changed, and still cannot drift from the spec, because it is read
    from the spec at generation time.

    Both markers must exist, in order, or generation fails: a silently empty or
    inverted passage would read as "this section is empty".
    """
    lines = path.read_text().splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith(start)]
    if not starts:
        raise KeyError(f"start marker not found in {path.name}: {start!r}")
    ends = [i for i in range(starts[0] + 1, len(lines)) if lines[i].startswith(end)]
    if not ends:
        raise KeyError(f"end marker not found after start in {path.name}: {end!r}")
    return "\n".join(lines[starts[0]:ends[0]]).rstrip()
