"""
# ==============================================================================
# Module: tests/unit/test_diagnose_reserved_fields.py
# ==============================================================================
# Purpose: `candidate_genes` is a reserved interface. These tests hold it to
#          what "reserved" must mean on a clinical API: accepted, validated,
#          inert, and **visibly** inert to the caller who sent it.
#
# The field was requested by the deploying institution so the capability can be
# enabled later without operating on the pipeline. Nothing reads it today. The
# defect this pins is not that it does nothing — it is that a caller could not
# tell. See docs/working/task-scope/README.md Q1.
#
# Tests:
#   - the warning appears when the field is supplied, on the mock path too
#   - no warning when it is absent
#   - results are invariant to the field's contents
#   - blank and whitespace-only entries are rejected
#   - over-limit lists are rejected, and a realistic 244-gene list is not
# ==============================================================================
"""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pydantic")

from pydantic import ValidationError  # noqa: E402

from src.api.routes.diagnose import (  # noqa: E402
    CANDIDATE_GENES_IGNORED_WARNING,
    CANDIDATE_GENES_MAX,
    DiagnoseRequest,
)

PHENOTYPES = ["HP:0001250", "HP:0002311"]


def _request(**kwargs) -> DiagnoseRequest:
    return DiagnoseRequest(phenotypes=list(PHENOTYPES), **kwargs)


# ==============================================================================
# Validation
# ==============================================================================
def test_blank_entries_are_rejected():
    for bad in ([""], ["BRCA1", "  "], ["\t"]):
        with pytest.raises(ValidationError, match="must not be blank"):
            _request(candidate_genes=bad)


def test_entries_are_stripped():
    request = _request(candidate_genes=["  BRCA1  ", "SCN1A"])
    assert request.candidate_genes == ["BRCA1", "SCN1A"]


def test_over_limit_lists_are_rejected():
    with pytest.raises(ValidationError):
        _request(candidate_genes=[f"GENE{i}" for i in range(CANDIDATE_GENES_MAX + 1)])


def test_a_realistic_variant_filtered_list_is_accepted():
    """The bound is request safety, not the future scorer's selection limit.

    The reference paper's variant-filtered candidate lists average 244.3 genes
    with SD 244.0, so a bound near `phenotypes`' 100 would reject ordinary real
    input. This is the case that fixes the number.
    """
    request = _request(candidate_genes=[f"GENE{i}" for i in range(244)])
    assert len(request.candidate_genes) == 244


def test_the_bound_is_not_the_phenotype_bound():
    assert CANDIDATE_GENES_MAX == 1000


# ==============================================================================
# Visible inertness
# ==============================================================================
def _run(client, payload):
    response = client.post("/api/v1/diagnose/", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from src.api.main import app

    return TestClient(app)


def test_supplying_the_field_warns_the_caller(client):
    body = _run(client, {"phenotypes": PHENOTYPES, "candidate_genes": ["BRCA1"]})
    assert CANDIDATE_GENES_IGNORED_WARNING in body["warnings"]


def test_omitting_the_field_produces_no_such_warning(client):
    body = _run(client, {"phenotypes": PHENOTYPES})
    assert CANDIDATE_GENES_IGNORED_WARNING not in body["warnings"]


def test_an_explicitly_supplied_empty_list_still_warns(client):
    """`[]` was supplied, so the contract says it warns.

    An earlier version asserted the opposite, encoding a "non-empty" contract
    through truthiness that neither the settled requirement nor the field's
    description states. Omitted and null do not warn; `[]` and non-empty do.
    """
    body = _run(client, {"phenotypes": PHENOTYPES, "candidate_genes": []})
    assert CANDIDATE_GENES_IGNORED_WARNING in body["warnings"]


def test_an_explicit_null_does_not_warn(client):
    body = _run(client, {"phenotypes": PHENOTYPES, "candidate_genes": None})
    assert CANDIDATE_GENES_IGNORED_WARNING not in body["warnings"]


def test_results_are_invariant_to_the_field(client):
    """The property the warning is honest about.

    Two requests differing only in `candidate_genes` must produce identical
    candidates. Whoever implements gene prioritisation has to delete this test
    deliberately, which is the point: until then the reservation cannot rot into
    a silent no-op unnoticed.

    **What this covers, and what it does not.** With no checkpoint loaded the
    route takes its mock fallback, so the comparison is over that path. The
    warning is appended before the pipeline branch precisely because the field
    is equally inert on both, but invariance *through the real pipeline* rests
    on the separate fact that no scoring path reads `candidate_genes` at all:
    it reaches `PatientPhenotypes` and stops. That fact is recorded in
    docs/working/task-scope/README.md F2; it is not shown here.
    """
    without = _run(client, {"phenotypes": PHENOTYPES})
    with_genes = _run(
        client, {"phenotypes": PHENOTYPES, "candidate_genes": ["BRCA1", "SCN1A"]}
    )

    assert without["candidates"], "fixture returned no candidates to compare"
    assert with_genes["candidates"] == without["candidates"]


def test_the_field_description_says_it_is_ignored():
    """A description reading like a working feature is the original defect."""
    description = DiagnoseRequest.model_fields["candidate_genes"].description
    assert "RESERVED" in description
    assert "ignored" in description
