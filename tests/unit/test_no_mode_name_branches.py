"""
No code branches on the mode values C or D (R2, K8-6e.2).
=========================================================
Policy rev 15 and the owner's natural retirement: the presentation and parsing
that existed only for kept Mode C and Mode D records retired; a kept C or D
record is read by the shared readers and shown by the one generic display as
recorded, never relabelled, hidden, ordered or refused by its mode's name. No
list of modes decides what a reader reads or a page shows (the reviewer,
2026-11-07: the three tasks are not a permanent read whitelist).

This is plan document B revision 6 §3.5 (iv)'s sweep row as a test: every
Python file under ``src/`` and ``scripts/`` is tokenized, and

- no string literal is exactly "C" or "D" (a mode value to branch on, key a
  table by or compare with), double- or single-quoted, str or bytes, except
  the two that are not a mode (`NOT_A_MODE`);
- no name of the retired C/D code is used (``MODE_C_*``, ``MODE_D_*``,
  ``PRE_B1*``, ``served_scoring``, ``mode_c*``/``mode_d*``);
- the words "Mode C"/"Mode D", in a string, an f-string's text or a
  comment, appear only on the dated lines listed with their reason (`DATED`,
  each pinned to its line's text): docstrings and comments stating what
  retired when, with no code under them.

Module: tests/unit/test_no_mode_name_branches.py
"""
from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: The string literals "C"/"D" that are not a measurement mode: (file, the
#: line's text that holds it) -> why.
NOT_A_MODE = {
    ("src/kg/graph_view.py", 'shortest_path(adj, method="D"'):
        "SciPy's shortest-path method (Dijkstra), not a mode",
    ("src/utils/run_environment.py", 'b"C" in code'):
        "a git status code (copied), not a mode",
}

#: The lines whose docstrings or comments name Mode C or Mode D, each a dated
#: statement of what retired when, with no code under it: (file, the line's
#: text that holds the words) -> why it stays. Pinned to the line, as
#: `NOT_A_MODE` is, so a new "Mode C"/"Mode D" text in the same file is caught.
DATED = {
    ("src/evaluation/__init__.py", "It does not correct.** Mode C, which measured a"):
        "Mode C retired in K8-6a/b",
    ("src/evaluation/__init__.py", "in K8-6b); a kept Mode C report is"):
        "a kept report is read as recorded",
    ("src/evaluation/measurement.py", "Mode C's executor, which ranked a ShepherdGNN rebuild"):
        "Mode C's executor retired in K8-6b",
    ("src/evaluation/measurement.py", "Mode C report keeps this format."):
        "kept reports keep the format",
    ("src/evaluation/measurement.py", "kept Mode C report names `build_shepherd_model`"):
        "the field a kept Mode C report names",
    ("src/evaluation/task_measurement.py", "metric schema 1 a kept Mode C report records."):
        "metric schema 1 is a kept Mode C report's",
    ("src/inference/scoring.py", "and Mode C's offline measurement in `src/evaluation`"):
        "what the scorer served until K8-4",
    ("src/webui/components/models_page.py", "its role list, its Mode C rule and"):
        "By test score's Mode C rule retired in K8-6d",
    ("scripts/measure_scorer.py", "**Mode D** (Batch 3's measurement of the served ranking)"):
        "Mode D removed in K7",
    ("scripts/measure_scorer.py", "A kept Mode D report registers by the general rules"):
        "its kept reports register by the general rules (K8-6c)",
}

#: The retired C/D code's names, and only those: a name that merely starts with
#: MODE_C or MODE_D (``MODE_CHOICES``, ``MODE_DEFAULT``) is not one.
_RETIRED_NAME = re.compile(r"PRE_B1\w*|served_scoring|\w*(?i:mode_[cd])(?:_\w*)?")
_WORDS = re.compile(r"\bMode [CD]\b")
#: The tokens whose text a person may read: strings, comments and, from Python
#: 3.12, the literal text inside an f-string (`FSTRING_MIDDLE`).
_TEXT_TOKENS = {tokenize.STRING, tokenize.COMMENT} | (
    {tokenize.FSTRING_MIDDLE} if hasattr(tokenize, "FSTRING_MIDDLE") else set())


def _python_files():
    for base in ("src", "scripts"):
        for path in sorted((ROOT / base).rglob("*.py")):
            yield path.relative_to(ROOT).as_posix(), path.read_text(encoding="utf-8")


def _hits(sources):
    """Each source's "C"/"D" string literals, retired names and "Mode C"/"Mode
    D" words, each as (file, line, the line's text or the name)."""
    literals, names, words = [], [], []
    for rel, source in sources:
        lines = source.splitlines()
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.STRING:
                try:
                    value = ast.literal_eval(tok.string)
                except (ValueError, SyntaxError):  # an f-string before Python 3.12
                    value = None
                if value in ("C", "D", b"C", b"D"):
                    literals.append((rel, tok.start[0], tok.line.strip()))
            elif tok.type == tokenize.NAME and _RETIRED_NAME.fullmatch(tok.string):
                names.append((rel, tok.start[0], tok.string))
            if tok.type in _TEXT_TOKENS:
                for match in _WORDS.finditer(tok.string):
                    row = tok.start[0] + tok.string.count("\n", 0, match.start())
                    words.append((rel, row, lines[row - 1].strip()))
    return literals, names, words


def _listed(hit, table):
    rel, _line, text = hit
    return any(rel == file and held in text for file, held in table)


def test_no_code_branches_on_the_mode_values_c_or_d():
    literals, names, words = _hits(_python_files())
    assert [hit for hit in literals if not _listed(hit, NOT_A_MODE)] == [], literals
    assert len(literals) == len(NOT_A_MODE), literals
    assert names == [], names
    assert [hit for hit in words if not _listed(hit, DATED)] == []


def test_every_listed_exception_is_still_there():
    """The lists name only what is there, so an exception that went is taken
    off them rather than left to excuse a new branch."""
    literals, _names, words = _hits(_python_files())
    for file, held in NOT_A_MODE:
        assert any(_listed(hit, {(file, held): ""}) for hit in literals), (file, held)
    for file, held in DATED:
        assert any(_listed(hit, {(file, held): ""}) for hit in words), (file, held)
    assert len(words) == len(DATED), words


def test_the_sweep_sees_what_it_is_for():
    """The sweep itself: an f-string naming Mode D (the form `metric_text` used),
    a comparison with "C", a comment and a retired name are each caught, and a
    name that only starts with MODE_C or MODE_D is not."""
    source = ('label = f"Mode D {name} ({case_set})"\n'
              'if mode == "C":\n'
              '    pass\n'
              'MODE_CHOICES = MODE_DEFAULT = served_scoring_mode = 1\n'
              'x = MODE_D_NOTE, MODE_C, served_scoring, run_mode_c, PRE_B1_NOTE\n'
              '# Mode C note\n'
              'y = """a docstring\nthat names Mode C here"""\n')
    literals, names, words = _hits([("probe.py", source)])
    assert [line for _rel, line, _text in words] == [1, 6, 8]
    assert [line for _rel, line, _text in literals] == [2]
    assert [name for _rel, _line, name in names] == [
        "MODE_D_NOTE", "MODE_C", "served_scoring", "run_mode_c", "PRE_B1_NOTE"]
    # A word in a listed file but on a line the list does not pin is caught.
    file, held = next(iter(DATED))
    assert not _listed((file, 1, "a new line naming Mode C"), DATED)
