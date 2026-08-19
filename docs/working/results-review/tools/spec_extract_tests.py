"""Regression tests for nested heading extraction.

Plain asserts and a temporary file — no pytest, no Markdown parser. Run directly
(`python spec_extract_tests.py`), or via `run()` from any generator that quotes
these specs: a generator that silently truncates is worse than one that refuses
to run.

Not collected by the project's suite — `testpaths = ["tests"]` and
`python_files = ["test_*.py"]` both exclude it, deliberately. This tests a
document tool, not the product.

Every case below is a shape the previous rule got wrong, or a shape the new rule
could plausibly get wrong in the other direction.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from spec_extract import between, extract, heading_tree

DOC = """\
# Spec 2 — Snapshot repository

Preamble that belongs to no section.

## 7. Rotation

Section seven preamble.

### 7.1 Depth

Depth text.

#### A named subheading with no number

Deeper text.

### 7.2 Retention

Retention text.

## 11. Frozen invariants

Eleven preamble.

### 11.1 Tests

Test text.

### 11.2 Reopen

Reopen text.

## 12. Next
"""


def _doc(text: str = DOC) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".md", delete=False)
    handle.write(text)
    handle.close()
    return Path(handle.name)


def _check(label: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{label} FAILED {detail}")


def run() -> None:
    path = _doc()

    # The exact regression: a parent section keeps its own subsections.
    block = extract(path, "11")
    _check("subsections kept", "### 11.1 Tests" in block and "### 11.2 Reopen" in block, block)
    _check("stops at the next sibling", "## 12. Next" not in block)
    _check("subsection bodies kept", "Reopen text." in block)

    # A deeper heading must not terminate a block, whether numbered or not.
    seven = extract(path, "7")
    _check("deeper numbered heading kept", "### 7.2 Retention" in seven)
    _check("deeper unnumbered heading kept", "A named subheading with no number" in seven)
    _check("stops before the next level-2", "## 11." not in seven)

    # A subsection extracted on its own stops at its own sibling, and keeps its
    # deeper children.
    seven_one = extract(path, "7.1")
    _check("subsection keeps its children", "A named subheading" in seven_one)
    _check("subsection stops at its sibling", "7.2 Retention" not in seven_one)

    # A level-1 heading terminates a level-2 block. The old rule matched only
    # levels 2-3 and would have run past it.
    twelve = extract(path, "12")
    tail = _doc(DOC + "\nTrailing.\n\n# Another document\n\nUnrelated.\n")
    _check("level-1 terminates", "Unrelated." not in extract(tail, "12"))
    _check("last section reaches EOF", "Next" in twelve)

    # Prefix collision: "11" must not match "11.1", and "1" must not match "11".
    _check("no prefix collision on 11.1", extract(path, "11").startswith("## 11."))
    try:
        extract(path, "1")
    except KeyError:
        pass
    else:
        raise AssertionError("section '1' matched something; there is no section 1")

    # A missing section raises rather than returning an empty block, because an
    # empty block in a submission reads as "this section is empty".
    try:
        extract(path, "99")
    except KeyError:
        pass
    else:
        raise AssertionError("a missing section did not raise")

    # The heading tree reports what was captured, at the right depths.
    tree = heading_tree(extract(path, "7"))
    _check("tree lists every heading", len(tree) == 4, str(tree))
    _check("tree indents by depth", tree[0].startswith("## ") and tree[1].startswith("  ### "))

    # --- passage extraction ----------------------------------------------
    _check("between() takes the passage and stops at the end marker",
           between(path, "### 7.1", "### 7.2").splitlines()[0] == "### 7.1 Depth")
    _check("between() keeps everything in the span",
           "Deeper text." in between(path, "### 7.1", "### 7.2"))
    for start, end, why in [
        ("### 9.9 missing", "### 7.2", "missing start"),
        ("### 7.1", "### 9.9 missing", "missing end"),
        ("### 7.2", "### 7.1", "end before start"),
    ]:
        try:
            between(path, start, end)
        except KeyError:
            continue
        raise AssertionError(f"between() accepted {why}")

    path.unlink()
    tail.unlink()


if __name__ == "__main__":
    run()
    print("spec_extract: all regression checks passed")
