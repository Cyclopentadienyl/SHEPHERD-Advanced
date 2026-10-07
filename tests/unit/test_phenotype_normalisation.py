"""
The one rule for a case's phenotype list, tested on its own.
============================================================
Every entry point applies `normalise_phenotypes`; these tests pin what it keeps.
How the pipeline, the API and the WebUI use it is tested through those entry
points (test_inference.py, test_diagnose_phenotype_input.py,
test_diagnosis_panel.py), not by restating the rule beside them.
"""
import pytest

from src.kg.phenotype_normalisation import (
    PHENOTYPE_NORMALISATION_VERSION,
    normalise_phenotypes,
)


def test_a_repeat_is_kept_once():
    result = normalise_phenotypes(["A", "A", "B"])

    assert result.kept == ("A", "B")
    assert result.repeats_removed == 1


def test_a_list_without_repeats_is_unchanged():
    result = normalise_phenotypes(["B", "A", "C"])

    assert result.kept == ("B", "A", "C")
    assert result.kept_positions == (0, 1, 2)
    assert result.repeats_removed == 0


def test_order_is_first_occurrence_wherever_the_repeats_fall():
    """Not sorted, and not whatever a set iterates in."""
    assert normalise_phenotypes(["C", "A", "C", "B", "A"]).kept == ("C", "A", "B")
    assert normalise_phenotypes(["A", "C", "A", "C", "B"]).kept == ("A", "C", "B")


def test_one_hundred_repeats_and_one_more_keep_both():
    """"100 × A, then B": nothing past the repeats is lost, and 99 are removed."""
    result = normalise_phenotypes(["A"] * 100 + ["B"])

    assert result.kept == ("A", "B")
    assert result.kept_positions == (0, 100)
    assert result.repeats_removed == 99


def test_positions_are_the_callers_original_ones():
    """A caller that already dropped an unknown id at position 0 passes the
    surviving positions, so a per-position field can follow them."""
    result = normalise_phenotypes(["A", "A", "B"], positions=[1, 2, 3])

    assert result.kept == ("A", "B")
    assert result.kept_positions == (1, 3)


def test_positions_must_match_the_identities():
    with pytest.raises(ValueError, match="2 positions for 3 phenotypes"):
        normalise_phenotypes(["A", "A", "B"], positions=[0, 1])


def test_the_key_is_whatever_identity_the_caller_mapped_to():
    """Graph indices in a sample file, node ids at serving: the rule compares the
    identities it is given, so two source ids mapped to one node are one."""
    assert normalise_phenotypes([7, 3, 7]).kept == (7, 3)


def test_an_empty_list_stays_empty():
    result = normalise_phenotypes([])

    assert result.kept == ()
    assert result.kept_positions == ()
    assert result.repeats_removed == 0


def test_the_rule_has_a_version():
    assert PHENOTYPE_NORMALISATION_VERSION == 1
