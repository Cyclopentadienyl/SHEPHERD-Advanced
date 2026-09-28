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
def built(tmp_path, monkeypatch):
    """One real build, and the files it consumed."""
    from scripts.build_knowledge_graph import build_knowledge_graph
    import src.ontology.settings as settings_module

    # **Not the operator's configuration.** The build reads
    # `configs/deployment.yaml` for its ontology roots and sources; a
    # deployment that configured `paths.ontology_roots` would add candidates
    # beside this fixture's cache and make it refuse as ambiguous, and one
    # with a reachable source could fetch. Pinned to "no roots, nothing to
    # fetch".
    monkeypatch.setattr(
        settings_module, "load_ontology_settings",
        lambda config_path=None: settings_module.OntologySettings(
            sources={"mondo": (), "hpo": (), "go": (), "mp": ()}
        ),
    )

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


# ---------------------------------------------------------------------------
# Phase 2: the same record when the operator chose the files
# ---------------------------------------------------------------------------

RIVAL_MONDO = MONDO_OBO.replace("releases/2026-06-11", "releases/2026-09-01")
CHOSEN_HPO = HPO_OBO.replace(
    "format-version: 1.2\n", "format-version: 1.2\ndata-version: hp/releases/2026-03-09\n"
)


def test_named_files_are_what_the_record_and_the_rebuild_command_name(
    tmp_path, monkeypatch, capsys
):
    """**Acceptance 2 and 30 through the entry point, with rivals on disk.**

    The configured root and the cache each hold a MONDO and an HPO that are
    not the named ones — so without the two paths this build refuses as
    ambiguous, and with them it has to take exactly the named bytes. A
    source is configured and every socket entry point detonates, so a build
    that fetched instead of opening the named file cannot pass by accident.

    The printed rebuild command is checked too: without the two paths it
    rebuilt from whatever the roots held next time, which with these rivals
    is a refusal, and after a new release lands is a different ontology."""
    import socket

    import src.ontology.download as download_module
    import src.ontology.settings as settings_module
    from scripts.build_knowledge_graph import build_knowledge_graph
    from src.kg.provenance import read_provenance

    root, cache, chosen, external, workspace = (
        tmp_path / n for n in ("root", "cache", "chosen", "external", "ws")
    )
    for directory in (root, cache, chosen, external):
        directory.mkdir()
    (root / "mondo.obo").write_text(RIVAL_MONDO)
    (root / "hp.obo").write_text(HPO_OBO)
    (cache / "mondo.obo").write_text(MONDO_OBO)
    (cache / "hpo.obo").write_text(HPO_OBO + "\n[Term]\nid: HP:0000003\nname: three\n")
    named = {"mondo": chosen / "mondo-picked.obo", "hpo": chosen / "hpo-picked.obo"}
    named["mondo"].write_text(MONDO_OBO.replace("disease two", "disease 2"))
    named["hpo"].write_text(CHOSEN_HPO)
    (external / "phenotype.hpoa").write_text(HPOA)
    (external / "genes_to_phenotype.txt").write_text(G2P)

    monkeypatch.setattr(
        settings_module, "load_ontology_settings",
        lambda config_path=None: settings_module.OntologySettings(
            roots=(root,),
            sources={name: ("https://purl.example.org/x.obo",)
                     for name in ("mondo", "hpo", "go", "mp")},
        ),
    )
    fetched = []
    monkeypatch.setattr(
        download_module, "download_ontology",
        lambda url, *a, **k: fetched.append(url) or (_ for _ in ()).throw(
            AssertionError(f"fetched {url}")
        ),
    )

    def explode(*_args, **_kwargs):
        raise AssertionError("a build given explicit paths touched the network")

    for name in ("socket", "create_connection", "getaddrinfo", "gethostbyname"):
        monkeypatch.setattr(socket, name, explode, raising=False)

    build_knowledge_graph(
        external_dir=external, workspace=workspace, ontology_cache_dir=cache,
        mondo_path=named["mondo"], hpo_path=named["hpo"], generate_samples=False,
    )

    assert fetched == []
    by_role = {e["role"]: e for e in read_provenance(workspace)["sources"]}
    for role, path in named.items():
        assert by_role[role]["digest"] == _sha256(path), f"{role} is not the named file"
        assert by_role[role]["filename"] == path.name
    assert by_role["mondo"]["declared_version"] == "releases/2026-06-11"
    assert by_role["hpo"]["declared_version"] == "hp/releases/2026-03-09"

    printed = capsys.readouterr().out
    assert f"--mondo-path {named['mondo']}" in printed
    assert f"--hpo-path {named['hpo']}" in printed


def test_each_slot_is_hashed_when_it_loads_not_after_both(tmp_path, monkeypatch):
    """A file replaced after its slot loaded must not have the replacement's
    digest recorded against that slot — the graph was built from what was
    parsed. The overwrite that used to do this (a misfiled `hpo.obo` replaced
    by the HPO download) is now refused before it happens, so the replacement
    is simulated: the HPO load rewrites the MONDO file."""
    import src.ontology.settings as settings_module
    from scripts.build_knowledge_graph import build_knowledge_graph
    from src.kg.provenance import read_provenance
    from src.ontology.loader import OntologyLoader

    monkeypatch.setattr(
        settings_module, "load_ontology_settings",
        lambda config_path=None: settings_module.OntologySettings(
            sources={"mondo": (), "hpo": (), "go": (), "mp": ()}
        ),
    )
    chosen, external = tmp_path / "chosen", tmp_path / "external"
    chosen.mkdir()
    external.mkdir()
    mondo, hpo = chosen / "mondo.obo", chosen / "hpo.obo"
    mondo.write_text(MONDO_OBO)
    hpo.write_text(HPO_OBO)
    (external / "phenotype.hpoa").write_text(HPOA)
    (external / "genes_to_phenotype.txt").write_text(G2P)
    parsed = _sha256(mondo)

    genuine = OntologyLoader.load

    def load(self, path, expect=None):
        if expect == "hpo":
            mondo.write_text(MONDO_OBO + "\n[Term]\nid: MONDO:0000003\nname: late\n")
        return genuine(self, path, expect=expect)

    monkeypatch.setattr(OntologyLoader, "load", load)

    build_knowledge_graph(
        external_dir=external, workspace=tmp_path / "ws", ontology_cache_dir=tmp_path / "cache",
        mondo_path=mondo, hpo_path=hpo, generate_samples=False,
    )

    by_role = {e["role"]: e for e in read_provenance(tmp_path / "ws")["sources"]}
    assert _sha256(mondo) != parsed, "the simulated replacement did not happen"
    assert by_role["mondo"]["digest"] == parsed
