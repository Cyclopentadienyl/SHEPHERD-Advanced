"""
The `--split` caveat has to reach the person running the command.
=================================================================
Backlog item 2. Both measurement entry points warn that `val` is not held-out
data, and the warning now names **two** independent reasons rather than one: the
split selects the checkpoint, and the generator never partitions by disease.

**Why this file exists at all.** The caveat lives in argparse `help=` text, which
nothing rendered until now — and argparse interpolates that text with
``help % params``, so the literal ``100%`` added for item 2 raised
``ValueError: unsupported format character ':'`` and broke ``--help`` outright on
both entry points. A caveat that makes ``--help`` crash is worse than no caveat.
The rendering assertion below is the point of this file; the content assertions
are what stop the caveat being quietly deleted or narrowed back to one reason.

Deliberately not asserted: exact wording. These check that each claim is present
and attributed, not that a sentence is preserved verbatim, so the text can be
improved without editing a test.

Module: tests/unit/test_split_caveat.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ENTRY_POINTS = ("measure_scorer", "calibrate_mode_a")


def _split_help(script: str) -> str:
    """The rendered `--split` help, through argparse's own formatter."""
    module = importlib.import_module(f"scripts.{script}")
    parser = None
    # `parse_args` builds the parser; borrow it by parsing nothing and catching
    # the exit, rather than duplicating the parser definition here.
    import argparse

    original = argparse.ArgumentParser.parse_args

    def capture(self, *args, **kwargs):
        nonlocal parser
        parser = self
        raise SystemExit(0)

    argparse.ArgumentParser.parse_args = capture
    try:
        with pytest.raises(SystemExit):
            module.parse_args([])
    finally:
        argparse.ArgumentParser.parse_args = original

    assert parser is not None, f"{script} did not build an ArgumentParser"
    action = next(a for a in parser._actions if "--split" in a.option_strings)
    # `format_help()` is what `--help` runs, interpolation included.
    return parser.format_help(), action.help


@pytest.mark.parametrize("script", ENTRY_POINTS)
def test_help_renders_at_all(script):
    """`help % params` runs over this text. A bare `%` in it makes `--help` raise
    `ValueError` before printing anything — which is how item 2 first landed."""
    rendered, _ = _split_help(script)
    assert "--split" in rendered


@pytest.mark.parametrize("script", ENTRY_POINTS)
def test_the_caveat_names_both_contamination_kinds(script):
    """One reason was already there. M4 established the second, and a caveat that
    names only checkpoint selection understates what the split cannot support."""
    _, help_text = _split_help(script)

    assert "early_stopping_monitor=val_mrr" in help_text, "checkpoint-selection contamination"
    assert "sample_generator" in help_text, "the generator's role"
    assert "never partitions by disease" in help_text, "why the overlap is structural"


@pytest.mark.parametrize("script", ENTRY_POINTS)
def test_the_measured_figure_is_stated_and_attributed(script):
    """BACKLOG §5.2's rule for this figure: it may reach user-facing help only
    behind the evidence, and it arrives citing where the evidence is."""
    _, help_text = _split_help(script)

    assert "7,970" in help_text
    assert "EVIDENCE_M4.json" in help_text


@pytest.mark.parametrize("script", ENTRY_POINTS)
def test_the_help_string_survives_argparse_interpolation(script):
    """Pinned separately from the rendering test, and against the mechanism rather
    than a proxy for it.

    argparse formats help as ``self._get_help_string(action) % params``. A literal
    ``%`` that is not doubled raises there, before anything is printed — which is
    how item 2's ``100%`` first broke ``--help`` on both entry points. Applying the
    same interpolation to the stored string catches that at the source, so a future
    edit adding a second percent fails here whether or not the rendering test
    happens to exercise the same path.
    """
    _, help_text = _split_help(script)

    # No named references are used, so an empty mapping is the whole contract:
    # this raises ValueError on a bare %, and returns the text on %%.
    help_text % {}
