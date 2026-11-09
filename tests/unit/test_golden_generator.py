"""
The golden generator never rewrites a strict golden.
====================================================
`tests/fixtures/goldens/generate.py`'s `main` writes the goldens it is asked
for from the code under test. A strict golden (`goldens.STRICT`: the P0-1
records and keys, base validation, the supplied importer's `cohort.json`
bytes) never changes this round (`docs/working/research-comparison/README.md`
§4 invariant 1, §5 Stages 1 and 4), so when what the code computes differs
from one, or one is missing, `main` refuses: it names the file and its rule,
writes nothing at all and exits 1. A run after a product change therefore
cannot replace a strict golden and make its test pass. Other goldens are
written when they differ and left untouched when they do not.

The writers are stand-ins and the golden directory a temporary one, so these
tests run `main`'s own decision, not the goldens' computation.

Module: tests/unit/test_golden_generator.py
"""
from __future__ import annotations

import pytest

from tests.fixtures.goldens import GOLDENS, STRICT, generate

#: A strict golden and one that is not.
STRICT_NAME = "p0_1_records.json"
OTHER_NAME = "p0_1_views.json"


@pytest.fixture()
def goldens(tmp_path, monkeypatch):
    """A golden directory holding a committed strict and other golden, and
    `main` writing into it from one stand-in writer whose outputs a test
    sets."""
    (tmp_path / STRICT_NAME).write_bytes(b"strict, committed\n")
    (tmp_path / OTHER_NAME).write_bytes(b"other, committed\n")
    outputs: dict[str, bytes] = {}
    monkeypatch.setattr(generate, "GOLDENS", tmp_path)
    monkeypatch.setattr(generate, "GOLDEN_WRITERS", {"p0_1": lambda scratch: dict(outputs)})
    return tmp_path, outputs


def test_the_declared_strict_goldens_are_committed_and_include_the_p0_1_records():
    """`STRICT` names committed goldens, the P0-1 records among them (README §4
    invariant 1), and says each one's rule."""
    assert STRICT_NAME in STRICT and OTHER_NAME not in STRICT
    for filename, rule in STRICT.items():
        assert (GOLDENS / filename).is_file(), filename
        assert rule.startswith("README §"), filename


def test_a_strict_golden_that_would_change_is_refused_and_nothing_is_written(goldens, capsys):
    """The default run (every golden): a strict golden whose computed bytes
    differ is refused, named with its rule, and no golden of the run is
    written, not even one that is not strict."""
    directory, outputs = goldens
    outputs.update({STRICT_NAME: b"strict, changed\n", OTHER_NAME: b"other, changed\n"})
    assert generate.main([]) == 1
    assert (directory / STRICT_NAME).read_bytes() == b"strict, committed\n"
    assert (directory / OTHER_NAME).read_bytes() == b"other, committed\n"
    err = capsys.readouterr().err
    assert f"refused: {STRICT_NAME} would change" in err
    assert STRICT[STRICT_NAME] in err
    assert "nothing was written" in err


def test_a_missing_strict_golden_is_refused_not_regenerated(goldens, capsys):
    """A strict golden that is not there is not written from today's code
    either: it is restored from the repository, not regenerated."""
    directory, outputs = goldens
    (directory / STRICT_NAME).unlink()
    outputs.update({STRICT_NAME: b"strict, changed\n"})
    assert generate.main(["p0_1"]) == 1
    assert not (directory / STRICT_NAME).exists()
    assert f"refused: {STRICT_NAME} is missing" in capsys.readouterr().err


def test_an_unchanged_strict_golden_lets_another_golden_be_written(goldens, capsys):
    """A named change to a golden that is not strict (a view addition) is
    written while the strict golden computed with it is unchanged, which is
    left as it is."""
    directory, outputs = goldens
    outputs.update({STRICT_NAME: b"strict, committed\n", OTHER_NAME: b"other, changed\n"})
    assert generate.main(["p0_1"]) == 0
    assert (directory / STRICT_NAME).read_bytes() == b"strict, committed\n"
    assert (directory / OTHER_NAME).read_bytes() == b"other, changed\n"
    out = capsys.readouterr().out
    assert f"unchanged {STRICT_NAME}" in out and f"wrote {OTHER_NAME}" in out
