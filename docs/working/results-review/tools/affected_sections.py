#!/usr/bin/env python3
"""Which spec sections the *old* extractor truncated — derived, not typed.

The submission generator used to end an extracted block at the next heading of
any level, so a parent section was cut off at its first subsection. Blocks sent
to the reviewer before that was fixed may therefore have been incomplete, and the
earlier submissions were overwritten in place, so which ones cannot be
reconstructed.

Rather than assert a list, this runs both rules over every numbered section of
all five specs and prints where they disagree. That is the audit surface.

    python tools/affected_sections.py              # the table
    python tools/affected_sections.py --show 2:7   # Spec 2 §7, complete

Both rules live here: the old one exists only for this comparison and produces no
output anywhere else.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spec_extract import HEADING, extract, file_sha256, heading_tree  # noqa: E402

SPECS = Path(__file__).resolve().parents[1]
FILES = {
    "0": "SPEC_0_INDEX.md",
    "1": "SPEC_1_RESULTS_REVIEW.md",
    "2": "SPEC_2_SNAPSHOT_REPOSITORY.md",
    "3": "SPEC_3_EVIDENCE_AND_AUDIT.md",
    "4": "SPEC_4_DEPLOYMENT_SECURITY.md",
}


def old_rule(lines: list[str], section: str):
    """The rule as it was when the earlier blocks were sent. Comparison only."""
    start_pat = re.compile(rf'^#{{2,3}} {re.escape(section)}[.\s]')
    any_head = re.compile(r'^#{2,3} \d')
    start = next((i for i, line in enumerate(lines) if start_pat.match(line)), None)
    if start is None:
        return None
    end = next((i for i in range(start + 1, len(lines)) if any_head.match(lines[i])), len(lines))
    return end - start


def new_rule(lines: list[str], section: str):
    pat = re.compile(rf'^(#{{1,6}}) +{re.escape(section)}[.\s]')
    hit = next(((i, len(m.group(1))) for i, line in enumerate(lines) if (m := pat.match(line))), None)
    if hit is None:
        raise SystemExit(f"section {section} not found")
    start, level = hit
    end = next(
        (i for i in range(start + 1, len(lines))
         if (m := HEADING.match(lines[i])) and len(m.group(1)) <= level),
        len(lines),
    )
    return end - start


def disagreements():
    for key, name in FILES.items():
        lines = (SPECS / name).read_text().splitlines()
        sections = [m.group(1) for line in lines
                    if (m := re.match(r'^#{2,6} +(\d+(?:\.\d+)*)[.\s]', line))]
        for section in sections:
            old, new = old_rule(lines, section), new_rule(lines, section)
            if old != new:
                yield key, name, section, old, new


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--show", metavar="SPEC:SECTION",
                        help="print one section complete, e.g. 2:7")
    args = parser.parse_args()

    if args.show:
        key, section = args.show.split(":", 1)
        if key not in FILES:
            raise SystemExit(f"spec must be one of {sorted(FILES)}")
        block = extract(SPECS / FILES[key], section)
        print(f"# {FILES[key]} §{section}  (sha256 {file_sha256(SPECS / FILES[key])[:16]}…)\n")
        print("\n".join(heading_tree(block)))
        print()
        print(block)
        return 0

    rows = list(disagreements())
    print(f"{'Spec':<6}{'Section':<10}{'old rule':>10}{'complete':>10}")
    for key, _, section, old, new in rows:
        old_text = "not found" if old is None else f"{old} lines"
        print(f"{key:<6}§{section:<9}{old_text:>10}{new:>7} lines")
    print(f"\n{len(rows)} sections where the two rules disagree.")
    print("Nested sections are contained in their parent's block.")
    for name in FILES.values():
        print(f"{file_sha256(SPECS / name)}  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
