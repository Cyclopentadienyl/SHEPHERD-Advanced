"""
The `--split` caveat has to reach the person running the command.
=================================================================
Backlog item 2. The measurement entry point warns that `val` is not held-out
data. The split still selects the checkpoint. The overlap reason is gone —
generation consumes a disease allocation and `src/evaluation/cohort.py` refuses a
workspace without a manifest, so a pre-allocation cohort cannot reach a
measurement at all, and the help states disjointness as the contract it now is,
with the channel a disease-level cut does not close.

**Since M4 a supplied cohort is not a split** (decision 8): `--split` names the
generator's `train` and `val` only, and an institutional or external cohort is a
registered version named by `--cohort`. The third limit -- that a supplied
cohort's overlap with training is an open measurement -- moved with it to the
`--cohort` help, where it now says the measurement takes that overlap (R10).
`--cohort-kind` is gone: the argument that names the cohort decides its kind.

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

ENTRY_POINTS = ("measure_scorer",)


def _option_help(script: str, option: str):
    """The rendered help and one option's help, through argparse's own formatter."""
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
    action = next(a for a in parser._actions if option in a.option_strings)
    # `format_help()` is what `--help` runs, interpolation included.
    return parser.format_help(), action.help


def _split_help(script: str):
    return _option_help(script, "--split")


def _cohort_help(script: str):
    return _option_help(script, "--cohort")


@pytest.mark.parametrize("script", ENTRY_POINTS)
def test_help_renders_at_all(script):
    """`help % params` runs over this text. A bare `%` in it makes `--help` raise
    `ValueError` before printing anything — which is how item 2 first landed."""
    rendered, _ = _split_help(script)
    assert "--split" in rendered


@pytest.mark.parametrize("script", ENTRY_POINTS)
def test_the_split_caveat_names_two_distinct_limits(script):
    """They are not the same limit and must not blur together.

    (1) `val` cannot be an independent evaluation, because it selects the
    checkpoint. (2) `val` *is* disease-disjoint, so it carries unseen-disease
    evidence -- but not for two diseases with identical phenotype content on
    opposite sides. And a supplied cohort is pointed at the registered path,
    since it is no split of the workspace.
    """
    _, help_text = _split_help(script)

    assert "early_stopping_monitor=val_mrr" in help_text, "checkpoint selection"
    assert "disease-disjoint from `train`, by construction" in help_text, "the contract"
    assert "identical" in help_text and "phenotype content" in help_text, (
        "the channel the disease-level cut does not close"
    )
    assert "train or val" in help_text, "the generator's splits only"
    assert "--cohort" in help_text and "import_cohort.py" in help_text, (
        "where a supplied cohort goes instead"
    )
    assert "not registered" in help_text, "a generated split's report is not registrable"


@pytest.mark.parametrize("script", ENTRY_POINTS)
def test_the_cohort_help_says_the_overlap_is_measured_and_never_zero_without_evidence(script):
    """The third limit, where a supplied cohort is now named: its overlap with
    training is a measurement, which every registered run takes as R10 from the
    bytes the checkpoint names, and which is unverifiable rather than zero when
    those bytes are not found."""
    rendered, help_text = _cohort_help(script)

    assert "--cohort" in rendered
    assert "measurement, not a contract" in help_text, "supplied cohorts"
    assert "R10" in help_text
    assert "--training-evidence" in help_text
    assert "unverifiable, never zero" in help_text
    assert "audit_split_overlap.py" in help_text, "the audit of a version"
    assert "report store" in help_text and "--output does not" in help_text


@pytest.mark.parametrize("script", ENTRY_POINTS)
def test_the_caveat_no_longer_hedges_about_which_regime_it_describes(script):
    """A negative assertion, because this claim has now inverted twice.

    It once said the generator "does not enforce" disjointness, then that overlap
    was "a property of the workspace" because two regimes coexisted. Only one
    regime exists: a pre-allocation workspace is refused before a measurement can
    run. Hedging language would now describe a case the code makes unreachable,
    which is the backward compatibility this pipeline no longer carries.
    """
    _, help_text = _split_help(script)

    for stale in (
        "property of the workspace", "may overlap completely",
        "workspaces generated before", "audited workspace", "7,970",
    ):
        assert stale not in help_text, (
            f"{stale!r} describes a regime that can no longer reach a measurement"
        )


@pytest.mark.parametrize("script", ENTRY_POINTS)
def test_the_cohort_kind_is_decided_by_the_argument_not_stated(script):
    """`--cohort-kind` stated what the argument now decides: `--split` is always
    generated and `--cohort` always a registered (supplied) version, so the kind
    cannot be stated wrongly. The flag is refused, and its help text is gone."""
    import src.evaluation.caveats as caveats

    module = importlib.import_module(f"scripts.{script}")
    with pytest.raises(SystemExit):
        module.parse_args(["--checkpoint", "c.pt", "--data-dir", "d", "--split", "val",
                           "--output", "o.json", "--cohort-kind", "generated"])
    assert not hasattr(caveats, "COHORT_KIND_HELP")


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
    for help_of in (_split_help, _cohort_help):
        _, help_text = help_of(script)

        # No named references are used, so an empty mapping is the whole contract:
        # this raises ValueError on a bare %, and returns the text on %%.
        help_text % {}
