"""Genes in the cohort import format and the one gene identifier lookup
(README §8: reference cohorts carry causal genes; PLAN_TEST_RESULTS §8.3–8.4).

Module: tests/unit/test_cohort_gene_fields.py
"""
from __future__ import annotations

import json

import pytest

from src.evaluation.cohort_registry import MalformedSource, map_cases, parse_source


def _source(*cases):
    return json.dumps({"format": "shepherd-cases/1", "cases": list(cases)}).encode()


def _case(case_id="c1", **extra):
    return {"case_id": case_id, "phenotypes": ["HP:0001250"], "disease": "MONDO:0011073",
            **extra}


def test_gene_fields_are_optional_and_parsed():
    plain, genes = parse_source(_source(
        _case("a"), _case("b", causal_genes=["SCN1A"], candidate_genes=["SCN1A", "PTEN"])))
    assert plain.causal_genes is None and plain.candidate_genes is None
    assert genes.causal_genes == ("SCN1A",) and genes.candidate_genes == ("SCN1A", "PTEN")


@pytest.mark.parametrize("bad", [[], "SCN1A", [""], [3], ["A", "A"]])
def test_a_malformed_gene_list_is_refused_with_the_other_problems(bad):
    with pytest.raises(MalformedSource) as caught:
        parse_source(_source(_case("a", causal_genes=bad), _case("b", genes=["X"])))
    text = str(caught.value)
    assert "causal_genes" in text and "keys this format does not define: ['genes']" in text


def test_mapping_by_node_id_or_unique_symbol_and_counting_the_rest():
    node_mapping = {"phenotype": {"hpo:HP:0001250": 0}, "disease": {"mondo:MONDO:0011073": 4}}
    lookup = {"disgenet:SCN1A": 0, "SCN1A": 0, "disgenet:PTEN": 1, "PTEN": 1}
    cases = parse_source(_source(
        _case("a", causal_genes=["scn1a"], candidate_genes=["disgenet:PTEN", "SCN1A", "NOPE"]),
        _case("b")))
    mapped = map_cases(cases, node_mapping, "keep", lookup)
    first, second = mapped.samples
    assert first["causal_gene_ids"] == [0] and first["gene_ids"] == [1, 0]
    assert "causal_gene_ids" not in second and "gene_ids" not in second
    assert mapped.counts["genes"] == {"cases_with_causal_genes": 1,
                                      "cases_with_candidate_genes": 1,
                                      "identifiers_listed": 4, "identifiers_unmappable": 1,
                                      "identifiers_repeated": 0,
                                      "cases_with_unmapped_causal_genes": 0}
    assert "causal_genes_unmapped" not in first


def test_a_causal_gene_the_graph_lacks_is_kept_by_identifier_and_aliases_count_once():
    """E5/F9: a case's truth is the causal genes it declared; one that names
    no single gene of the graph is kept as its normalised identifier, so the
    case is not read as having no causal gene and two cases declaring it
    share it. Two identifiers of one graph gene are one gene, a repeat."""
    node_mapping = {"phenotype": {"hpo:HP:0001250": 0}, "disease": {"mondo:MONDO:0011073": 4}}
    lookup = {"disgenet:SCN1A": 0, "SCN1A": 0}
    cases = parse_source(_source(
        _case("a", causal_genes=["SCN1A", "disgenet:SCN1A", "NOPE"]),
        _case("b", causal_genes=[" nope", "Other"])))
    first, second = map_cases(cases, node_mapping, "keep", lookup).samples
    assert first["causal_gene_ids"] == [0] and first["causal_genes_unmapped"] == ["NOPE"]
    assert second["causal_gene_ids"] == [] and second["causal_genes_unmapped"] == ["NOPE",
                                                                                  "OTHER"]


@pytest.mark.parametrize("bad", [[], ["nope"], ["NOPE", "NOPE"], [3], "NOPE", [""]])
def test_the_reader_takes_the_importers_identifiers_and_refuses_anything_else(tmp_path, bad):
    from src.kg.storage.file_storage import read_samples_file

    path = tmp_path / "samples.json"
    row = {"patient_id": "a", "phenotype_ids": [0], "disease_id": 4, "causal_gene_ids": [0]}
    path.write_text(json.dumps([{**row, "causal_genes_unmapped": ["NOPE", "OTHER"]}]))
    assert read_samples_file(path, gene_fields=True).samples[0].causal_genes_unmapped == [
        "NOPE", "OTHER"]
    path.write_text(json.dumps([{**row, "causal_genes_unmapped": bad}]))
    with pytest.raises(ValueError, match="causal_genes_unmapped"):
        read_samples_file(path, gene_fields=True)


def test_a_source_without_genes_has_no_gene_counts():
    node_mapping = {"phenotype": {"hpo:HP:0001250": 0}, "disease": {"mondo:MONDO:0011073": 4}}
    mapped = map_cases(parse_source(_source(_case("a"))), node_mapping, "keep")
    assert "genes" not in mapped.counts


def test_gene_lookup_maps_ids_and_unique_symbols_only():
    from src.core.types import DataSource, Node, NodeID, NodeType
    from src.kg.graph import KnowledgeGraph

    kg = KnowledgeGraph()
    for source, local, name in [(DataSource.DISGENET, "1", "ABC"), (DataSource.CLINVAR, "2", "ABC"),
                                (DataSource.DISGENET, "3", "XYZ")]:
        kg.add_node(Node(id=NodeID(source=source, local_id=local), node_type=NodeType.GENE,
                         name=name, attributes={"symbol": name}))
    lookup = kg.gene_lookup()
    assert "ABC" not in lookup
    assert lookup["XYZ"] == lookup["disgenet:3"]
    assert {"disgenet:1", "clinvar:2", "disgenet:3"} <= set(lookup)


_ROW = {"phenotype_ids": [0], "disease_id": 0, "gene_ids": [0]}


@pytest.mark.parametrize("row", [
    {"patient_id": "partly-mapped", "causal_gene_ids": [0], "causal_genes_listed": 2},
    {"patient_id": "unmapped", "causal_gene_ids": [], "causal_genes_listed": 1},
    {"patient_id": "both", "causal_gene_ids": [], "causal_genes_unmapped": ["GENE_X"],
     "causal_genes_listed": 1},
], ids=["partly-mapped", "unmapped", "beside-its-replacement"])
def test_a_gene_read_refuses_the_retired_count_even_with_the_right_digest(tmp_path, row):
    """Recheck of d7221f4: the retired `causal_genes_listed` (a count, which
    cannot say which genes two patients share) is refused by the reader of
    the gene fields, alone or beside `causal_genes_unmapped`, never read past;
    the version is re-imported from its source."""
    import hashlib

    from src.kg.storage.file_storage import read_samples_file

    path = tmp_path / "samples.json"
    path.write_text(json.dumps([{**_ROW, **row}]))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="retired field 'causal_genes_listed'.*Re-import"):
        read_samples_file(path, gene_fields=True, expected_sha256=digest)


def test_the_current_gene_fields_and_rows_without_them_read_as_before(tmp_path):
    """The controls: the importer's current form, and a row with no gene
    field, read unchanged; a read that takes no gene field (the disease task,
    measurement's defaults) reads no gene truth, so it ignores every gene
    field, retired or not."""
    from src.kg.storage.file_storage import read_samples_file

    path = tmp_path / "samples.json"
    rows = [{**_ROW, "patient_id": "new", "causal_gene_ids": [],
             "causal_genes_unmapped": ["GENE_X"]},
            {"patient_id": "plain", "phenotype_ids": [0], "disease_id": 0}]
    path.write_text(json.dumps(rows))
    new, plain = read_samples_file(path, gene_fields=True).samples
    assert new.causal_genes_unmapped == ["GENE_X"] and new.causal_gene_ids == []
    assert plain.causal_gene_ids is None and plain.causal_genes_unmapped is None
    path.write_text(json.dumps([{**_ROW, "patient_id": "old", "causal_gene_ids": [0],
                                 "causal_genes_listed": 2}]))
    (old,) = read_samples_file(path).samples
    assert old.patient_id == "old" and old.causal_gene_ids is None
