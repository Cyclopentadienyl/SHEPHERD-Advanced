"""The provenance record, through the entry point an operator actually runs.

The unit tests hand `write_workspace` a list of source entries. That proves the
writer and the reader, and nothing about whether the build script collects the
right four files, reads the raw `data-version` rather than the fallback, or
keeps the parser's counters. This drives `build_knowledge_graph` with real OBO
fixtures so the loader, the parser, the builder, the writer and the reader are
all the production ones.

The ontologies are tiny but genuine: pronto parses them, `Ontology.source_path`
and `Ontology.declared_version` come from the files, and the OMIM→MONDO mapping
is built from their xrefs exactly as it is for a real MONDO.

Module: tests/integration/test_build_provenance.py
"""
from __future__ import annotations

import hashlib

import pytest

pytest.importorskip("torch")
pytest.importorskip("pronto")

MONDO_OBO = """format-version: 1.2
data-version: releases/2026-06-11

[Term]
id: MONDO:0000001
name: disease one
xref: OMIM:100100

[Term]
id: MONDO:0000002
name: disease two
xref: OMIM:100200
"""

# Deliberately declares no `data-version`, so the "absent stays null" contract
# is exercised by a real parse rather than by a hand-built entry.
HPO_OBO = """format-version: 1.2

[Term]
id: HP:0000001
name: phenotype one

[Term]
id: HP:0000002
name: phenotype two
is_a: HP:0000001
"""

HPOA = "#description: test\nOMIM:100100\tdisease one\t\tHP:0000001\t\t\t\t\t\t\t\t\n"
G2P = (
    "ncbi_gene_id\tgene_symbol\thpo_id\thpo_name\tfrequency\tdisease_id\n"
    "1\tGENE1\tHP:0000001\tphenotype one\t\tOMIM:100100\n"
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def built(tmp_path):
    """One real build, and the files it consumed."""
    from scripts.build_knowledge_graph import build_knowledge_graph

    cache, external, workspace = (tmp_path / n for n in ("cache", "external", "ws"))
    cache.mkdir()
    external.mkdir()
    files = {
        "mondo": cache / "mondo.obo",
        "hpo": cache / "hpo.obo",
        "phenotype_hpoa": external / "phenotype.hpoa",
        "genes_to_phenotype": external / "genes_to_phenotype.txt",
    }
    for key, body in (
        ("mondo", MONDO_OBO), ("hpo", HPO_OBO),
        ("phenotype_hpoa", HPOA), ("genes_to_phenotype", G2P),
    ):
        files[key].write_text(body)

    build_knowledge_graph(
        external_dir=external, workspace=workspace,
        ontology_cache_dir=cache, generate_samples=False,
    )
    return workspace, files


def test_the_build_records_the_four_files_it_read(built):
    """Digests computed here from the fixture bytes, not by the code that wrote
    them — a writer hashing the wrong file would otherwise agree with a test
    hashing it the same wrong way."""
    from src.kg.provenance import SOURCE_ROLES, read_provenance

    workspace, files = built
    record = read_provenance(workspace)

    assert record["origin"] == "files"
    assert record["missing_roles"] == []
    by_role = {entry["role"]: entry for entry in record["sources"]}
    assert set(by_role) == set(SOURCE_ROLES)
    for role, path in files.items():
        assert by_role[role]["digest"] == _sha256(path), f"{role} digest is not the file's"
        assert by_role[role]["filename"] == path.name


def test_the_declared_version_comes_from_the_file_and_absence_stays_absent(built):
    """MONDO declares a release and HPO does not. `Ontology.version` would have
    returned the OBO format version for the second, recording `1.2` as though a
    release had been declared."""
    from src.kg.provenance import read_provenance

    workspace, _ = built
    by_role = {e["role"]: e for e in read_provenance(workspace)["sources"]}

    assert by_role["mondo"]["declared_version"] == "releases/2026-06-11"
    assert by_role["hpo"]["declared_version"] is None


def test_the_parser_counters_survive_to_the_record(built):
    """They were a log line and reached no artifact. Named for the rows of one
    file at one parsing stage, which is what they count."""
    from src.kg.provenance import read_provenance

    workspace, _ = built
    counters = read_provenance(workspace)["counters"]

    assert counters["rows_parsed"] == 1
    assert counters["rows_skipped_unresolved_disease_id"] == 0
    assert "rows_skipped_negated" in counters


def test_the_record_names_the_graph_this_build_wrote(built):
    from src.kg.provenance import verify_provenance

    workspace, _ = built

    record = verify_provenance(workspace, _sha256(workspace / "kg.json"))
    assert record is not None


def test_a_graph_only_build_still_produced_one(built):
    """`generate_samples=False`, so there is no split manifest — the path a
    manifest-carried record would have missed entirely."""
    from src.kg.artifacts import MANIFEST_FILENAME
    from src.kg.provenance import PROVENANCE_FILENAME

    workspace, _ = built

    assert not (workspace / MANIFEST_FILENAME).exists()
    assert (workspace / PROVENANCE_FILENAME).is_file()
