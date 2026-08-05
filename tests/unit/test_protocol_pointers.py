"""
`src/core/protocols.py` implementation pointers must match the file tree.
=========================================================================
Every Protocol in that file records where it is (or will be) implemented::

    實現模組: src/ontology/loader.py (IMPLEMENTED)

Those labels are structured factual claims about the repository, and before they
existed the same line was written whether the target was implemented, an empty
reserved home, or a file that had never been created — 13 of 32 pointers read as
statements of fact and were not. Labelling them fixed that once; this file keeps
it fixed, so the next engineer to create `src/llm/medical_llm.py` is told to
update its label instead of leaving a stale `PLANNED` behind.

Scope, deliberately narrow: this is a **repository-state** test. It checks that a
path exists (or does not) and whether it contains implementation code. It does
**not** check that an `IMPLEMENTED` module satisfies its Protocol's methods or
signatures — doing so would mean importing implementation modules, pulling in
torch / PyG / FastAPI and turning a millisecond test into one that skips on any
host without the full stack. It also could not cover `PLANNED` or `RESERVED`
pointers at all, and those are the drift class this exists to prevent.

The label wording in `protocols.py` is kept aligned with exactly this scope.
"""
import ast
import re
import subprocess
from pathlib import Path
from typing import List, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOLS = REPO_ROOT / "src" / "core" / "protocols.py"

LABELS = ("IMPLEMENTED", "RESERVED", "PLANNED")

# `\S+?` stops before the space preceding "(", so the bare form below still matches
# a pointer whose label was dropped — which is the case we want to fail loudly.
LABELLED_RE = re.compile(r"實現模組:\s*(?P<target>\S+?)\s*\((?P<label>[A-Za-z_]+)\)")
BARE_RE = re.compile(r"實現模組:\s*(?P<target>\S+)")


def _has_code(path: Path) -> bool:
    """Whether a module has any statement beyond its docstring.

    AST rather than file size: this repository has reserved homes in both physical
    states — `src/reasoning/constraint_checker.py` is over a kilobyte of docstring,
    `src/nlp/hpo_matcher.py` is zero bytes — and both are RESERVED. A size rule
    would be wrong on the first and right on the second by accident.
    """
    try:
        body = ast.parse(path.read_text(encoding="utf-8")).body
    except (SyntaxError, UnicodeDecodeError):
        # Unparseable is not "empty". Treat as code so a broken file cannot pass
        # itself off as a reserved home.
        return True
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    return bool(body)


def _repo_python_files(directory: Path) -> List[Path]:
    """Python files under `directory`, bounded to what the repository tracks.

    A plain filesystem walk would let an untracked scratch file in a reserved
    package fail the suite, which is noise rather than drift. `git ls-files` is the
    right boundary because the claim being checked is about the repository, not
    about one working directory. The walk is kept as a fallback for a source tree
    that is not a git checkout, with generated and hidden paths excluded.
    """
    try:
        rel = directory.relative_to(REPO_ROOT)
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files", "-z", "--", str(rel)],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        tracked = [REPO_ROOT / p for p in completed.stdout.split("\0") if p.endswith(".py")]
        return [p for p in tracked if p.is_file()]
    except (OSError, ValueError, subprocess.SubprocessError):
        return [
            p
            for p in directory.rglob("*.py")
            if "__pycache__" not in p.parts and not any(x.startswith(".") for x in p.parts)
        ]


def _collect_pointers() -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str]]]:
    """Return (labelled pointers, protocols with a missing or malformed pointer)."""
    tree = ast.parse(PROTOCOLS.read_text(encoding="utf-8"))
    labelled: List[Tuple[str, str, str]] = []
    problems: List[Tuple[str, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        doc = ast.get_docstring(node) or ""
        match = LABELLED_RE.search(doc)
        if match is None:
            bare = BARE_RE.search(doc)
            problems.append(
                (
                    node.name,
                    f"pointer `實現模組: {bare.group('target')}` has no status label"
                    if bare
                    else "no `實現模組:` pointer at all",
                )
            )
            continue
        label = match.group("label")
        if label not in LABELS:
            problems.append((node.name, f"unknown status label {label!r}"))
            continue
        labelled.append((node.name, match.group("target"), label))

    return labelled, problems


POINTERS, PROBLEMS = _collect_pointers()


def test_every_protocol_carries_a_known_status_label():
    """A new Protocol without a labelled pointer is drift, and fails here.

    The expected count is not hard-coded — the file is traversed — so adding a
    Protocol is fine, and adding one without a pointer is not.
    """
    assert not PROBLEMS, "\n".join(f"  {name}: {why}" for name, why in PROBLEMS)
    assert POINTERS, "no labelled pointers found — has the pointer format changed?"


@pytest.mark.parametrize(
    "protocol,target,label",
    POINTERS,
    ids=[f"{name}->{target}" for name, target, _ in POINTERS],
)
def test_label_matches_the_file_tree(protocol, target, label):
    """IMPLEMENTED / RESERVED / PLANNED must describe what is actually there."""
    path = REPO_ROOT / target.rstrip("/")
    is_directory_pointer = target.endswith("/")

    if is_directory_pointer:
        exists = path.is_dir()
        members = _repo_python_files(path) if exists else []
        has_code = any(_has_code(p) for p in members)
        detail = f"directory exists={exists}, python files={len(members)}, any with code={has_code}"
    else:
        exists = path.is_file()
        has_code = exists and _has_code(path)
        detail = f"file exists={exists}, has code={has_code}"

    context = f"{protocol} -> {target} labelled {label} but {detail}"

    if label == "IMPLEMENTED":
        assert exists, f"{context}. Label PLANNED if it has not been created yet."
        assert has_code, f"{context}. Label RESERVED if it is an empty reserved home."
    elif label == "RESERVED":
        assert exists, f"{context}. A reserved home must exist; label PLANNED otherwise."
        assert not has_code, f"{context}. It now contains code — relabel IMPLEMENTED."
    else:  # PLANNED
        assert not exists, (
            f"{context}. It now exists — relabel RESERVED if it is an empty reserved "
            f"home, or IMPLEMENTED if it contains code."
        )


def test_scope_is_recorded_in_the_protocols_legend():
    """The legend must not claim more than this test verifies.

    Writing a label whose wording exceeds what is enforced would reproduce, in
    miniature, the defect this whole convention exists to remove.
    """
    header = ast.get_docstring(ast.parse(PROTOCOLS.read_text(encoding="utf-8"))) or ""
    for label in LABELS:
        assert label in header, f"{label} is not explained in the module docstring legend"
    assert "tests/unit/test_protocol_pointers.py" in header, (
        "the legend should name the test that enforces the labels"
    )
    assert "方法簽章" in header, (
        "the legend should state that Protocol signature conformance is NOT checked here"
    )
