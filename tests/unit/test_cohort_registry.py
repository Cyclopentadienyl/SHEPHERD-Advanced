"""
The cohort registry: a source mapped to one graph, published as an immutable version.
====================================================================================
Contract M4 (§0.1-§0.7, §2 of the design). What is pinned here:

- **where** a registry may be (absolute, existing, never inside the repository);
- **what a source is** (`shepherd-cases/1`, strict, every problem listed, OMIM
  and ORPHA refused as a whole);
- **how cases map** (serving's lookup, then N1's `normalise_phenotypes`; the two
  exclusion categories with their precedence; both partial-phenotype policies);
- **what a commit writes** (source verbatim and never overwritten, samples, the
  manifest last; nothing at all on a refusal) and that it publishes only the
  previewed version;
- **how a version is read back** (one read of `cohort.json`, every refusal) and
  **compared with a run's graph** (§0.7);
- the **registered-cohort fixture**: its checkpoint loads safely, matches the
  workspace's graph, carries a pipeline-shaped record, and Mode C ranks the
  version's samples against it.

Module: tests/unit/test_cohort_registry.py
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.evaluation import cohort_registry
from src.evaluation.cohort_registry import (
    EXCLUSION_CATEGORIES,
    REPO_ROOT,
    REQUIRED_VERSION_FIELDS,
    CohortBindingMismatch,
    CohortVersionRead,
    CohortVersionRefused,
    MalformedSource,
    RegistryRefused,
    UnsupportedDiseaseNamespace,
    build_version_manifest,
    commit_import,
    create_report_dir,
    encode_json,
    map_cases,
    parse_source,
    prepare_import,
    preview_id,
    read_workspace_graph,
    resolve_version,
    validate_registry,
    verify_cohort_binding,
)
from src.utils.fingerprint import ReadIdentity
from tests.fixtures.registered_cohort import (
    GRAPH_SOURCE_COUNTS,
    GRAPH_SOURCE_UNMAPPABLE_TERMS,
    IMPORTED_AT,
    LABEL,
    ROLE,
    UNKNOWN_HPO,
    UNKNOWN_MONDO,
    build_registered_cohort,
    import_version,
    val_source_cases,
    write_fixture_checkpoint,
    write_source,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tree(root: Path) -> dict:
    """Every file under `root` with its bytes, and every directory: what a
    "nothing was written" assertion compares."""
    return {
        str(path.relative_to(root)): (path.read_bytes() if path.is_file() else None)
        for path in sorted(root.rglob("*"))
    }


def _source_bytes(cases, **top) -> bytes:
    return json.dumps({"format": "shepherd-cases/1", "cases": cases, **top}).encode()


@pytest.fixture(scope="module")
def registered(tmp_path_factory):
    """One workspace, checkpoint, source and committed version, built once.
    Tests read its workspace and source and never write into them."""
    return build_registered_cohort(tmp_path_factory.mktemp("registered"))


@pytest.fixture
def registry(tmp_path):
    path = tmp_path / "registry"
    path.mkdir()
    return path


# ---------------------------------------------------------------------------
# Where a registry may be (§0.1)
# ---------------------------------------------------------------------------
def test_a_registry_is_resolved_and_returned(tmp_path):
    assert validate_registry(tmp_path) == tmp_path.resolve()
    assert validate_registry(str(tmp_path)) == tmp_path.resolve()


def test_a_relative_registry_is_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "reg").mkdir()
    with pytest.raises(RegistryRefused, match="not an absolute path"):
        validate_registry("reg")


@pytest.mark.parametrize("inside", ["", "data", "data/no-such-registry", "src/evaluation"])
def test_a_registry_inside_the_repository_is_refused(inside):
    # Refused before the existence check, so a path that does not exist inside
    # the tree is refused for the reason that matters.
    with pytest.raises(RegistryRefused, match="inside the repository"):
        validate_registry(REPO_ROOT / inside)


def test_a_symlink_into_the_repository_is_refused_after_resolving(tmp_path):
    link = tmp_path / "looks-outside"
    link.symlink_to(REPO_ROOT / "docs")
    with pytest.raises(RegistryRefused, match="inside the repository"):
        validate_registry(link)


def test_a_missing_registry_is_refused_and_not_created(tmp_path):
    with pytest.raises(RegistryRefused, match="does not exist"):
        validate_registry(tmp_path / "missing")
    assert not (tmp_path / "missing").exists()


def test_a_file_is_not_a_registry(tmp_path):
    (tmp_path / "file").write_text("x")
    with pytest.raises(RegistryRefused, match="not a directory"):
        validate_registry(tmp_path / "file")


def test_the_repository_root_is_computed_from_the_module():
    assert REPO_ROOT == Path(cohort_registry.__file__).resolve().parents[2]
    assert (REPO_ROOT / "src" / "evaluation" / "cohort_registry.py").is_file()


# ---------------------------------------------------------------------------
# Labels, roles and the policy (§0.2)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("label", ["../x", "/abs", "Upper", "", "a/b", "x" * 65, "ok\n"])
def test_a_label_follows_the_split_name_rule(label):
    with pytest.raises(CohortVersionRefused, match="label is refused"):
        cohort_registry.validate_label(label)


@pytest.mark.parametrize("role", ["train", "val"])
def test_a_role_may_not_be_a_generator_split(role):
    with pytest.raises(CohortVersionRefused, match="generator's own split names"):
        cohort_registry.validate_role(role)


def test_a_role_follows_the_split_name_rule():
    with pytest.raises(CohortVersionRefused, match="role is refused"):
        cohort_registry.validate_role("My Role")
    assert cohort_registry.validate_role("mygene2") == "mygene2"


def test_an_unknown_policy_is_refused():
    with pytest.raises(CohortVersionRefused, match="keep, exclude"):
        cohort_registry.validate_partial_policy("drop")


# ---------------------------------------------------------------------------
# The source (§0.4)
# ---------------------------------------------------------------------------
GOOD_CASE = {"case_id": "c1", "phenotypes": ["HP:0000001"], "disease": "MONDO:0000001"}


def test_a_well_formed_source_parses():
    cases = parse_source(_source_bytes([
        GOOD_CASE,
        {"case_id": "c2", "phenotypes": ["HP:0000002", "HP:0000002"], "disease": "MONDO:0000002"},
    ]))
    assert [case.case_id for case in cases] == ["c1", "c2"]
    assert cases[1].phenotypes == ("HP:0000002", "HP:0000002")
    assert cases[1].disease == "MONDO:0000002"


@pytest.mark.parametrize("data, problem", [
    (b"\xff\xfe", "not UTF-8 JSON"),
    (b"{not json", "not UTF-8 JSON"),
    (b"[]", "top level is not a JSON object"),
    (json.dumps({"format": "other/1", "cases": []}).encode(), "format is 'other/1'"),
    (json.dumps({"cases": [GOOD_CASE]}).encode(), "format is None"),
    (json.dumps({"format": "shepherd-cases/1"}).encode(), "cases is not a list"),
    (json.dumps({"format": "shepherd-cases/1", "cases": {}}).encode(), "cases is not a list"),
    (_source_bytes([]), "cases is empty"),
    (_source_bytes([GOOD_CASE], notes="x"), "keys this format does not define: ['notes']"),
    (b'{"format": "shepherd-cases/1", "format": "shepherd-cases/1", "cases": []}',
     "states the key 'format' twice"),
])
def test_a_malformed_top_level_is_refused(data, problem):
    with pytest.raises(MalformedSource) as caught:
        parse_source(data, source="cases.json")
    assert problem in str(caught.value)
    assert "cases.json" in str(caught.value)


@pytest.mark.parametrize("case, problem", [
    ("not a case", "cases[0] is not a JSON object"),
    ({**GOOD_CASE, "case_id": ""}, "case_id must be a non-empty string"),
    ({**GOOD_CASE, "case_id": 7}, "case_id must be a non-empty string"),
    ({"case_id": "c1", "phenotypes": ["HP:0000001"]}, "missing ['disease']"),
    ({**GOOD_CASE, "sex": "F"}, "keys this format does not define: ['sex']"),
    ({**GOOD_CASE, "phenotypes": []}, "phenotypes must be a non-empty list"),
    ({**GOOD_CASE, "phenotypes": "HP:0000001"}, "phenotypes must be a non-empty list"),
    ({**GOOD_CASE, "phenotypes": ["HP:1"]}, "phenotypes[0] 'HP:1' is not an HPO id"),
    ({**GOOD_CASE, "phenotypes": ["HP:0000001", 5]}, "phenotypes[1] 5 is not an HPO id"),
    ({**GOOD_CASE, "phenotypes": ["HP:١٢٣٤٥٦٧"]},
     "is not an HPO id"),
    ({**GOOD_CASE, "disease": "MONDO:12"}, "disease 'MONDO:12' is not a MONDO id"),
    ({**GOOD_CASE, "disease": None}, "disease None is not a MONDO id"),
])
def test_a_malformed_case_is_refused_by_index_and_id(case, problem):
    with pytest.raises(MalformedSource) as caught:
        parse_source(_source_bytes([case]))
    assert problem in str(caught.value)
    assert any(problem in listed for listed in caught.value.problems)
    assert str(caught.value).count("cases[0]") >= 1


def test_every_problem_is_listed_not_only_the_first():
    with pytest.raises(MalformedSource) as caught:
        parse_source(_source_bytes([
            GOOD_CASE,
            {"case_id": "c2", "phenotypes": ["bad"], "disease": "MONDO:0000001"},
            {**GOOD_CASE},  # repeats c1
            {"case_id": "c4", "phenotypes": ["HP:0000001"], "disease": "nope"},
        ]))
    problems = caught.value.problems
    assert len(problems) == 3
    assert "cases[1] (case_id 'c2'): phenotypes[0] 'bad'" in problems[0]
    assert "cases[2] (case_id 'c1'): case_id repeats cases[0]" in problems[1]
    assert "cases[3] (case_id 'c4'): disease 'nope'" in problems[2]


def test_a_case_stating_a_key_twice_is_refused():
    data = (b'{"format": "shepherd-cases/1", "cases": [{"case_id": "c1", "phenotypes": '
            b'["HP:0000001"], "disease": "MONDO:0000001", "disease": "MONDO:0000002"}]}')
    with pytest.raises(MalformedSource, match="states the key 'disease' twice"):
        parse_source(data)


def test_ground_truth_outside_mondo_refuses_the_import_naming_namespace_and_count():
    with pytest.raises(UnsupportedDiseaseNamespace) as caught:
        parse_source(_source_bytes([
            GOOD_CASE,
            {"case_id": "c2", "phenotypes": ["HP:0000001"], "disease": "OMIM:100100"},
            {"case_id": "c3", "phenotypes": ["HP:0000001"], "disease": "OMIM:100200"},
            {"case_id": "c4", "phenotypes": ["HP:0000001"], "disease": "ORPHA:558"},
            {"case_id": "c5", "phenotypes": ["HP:0000001"], "disease": "Orphanet:99"},
        ]), source="cases.json")
    assert caught.value.namespaces == {"OMIM": 2, "ORPHA": 1, "Orphanet": 1}
    message = str(caught.value)
    assert "4 case(s)" in message and "OMIM: 2" in message and "ORPHA: 1" in message
    assert not isinstance(caught.value, MalformedSource)


def test_a_malformed_source_also_names_its_unsupported_namespaces():
    with pytest.raises(MalformedSource) as caught:
        parse_source(_source_bytes([
            {"case_id": "c1", "phenotypes": ["x"], "disease": "MONDO:0000001"},
            {"case_id": "c2", "phenotypes": ["HP:0000001"], "disease": "OMIM:100100"},
        ]))
    assert len(caught.value.problems) == 1
    assert "outside MONDO (OMIM: 1)" in str(caught.value)


# ---------------------------------------------------------------------------
# Mapping (§0.5)
# ---------------------------------------------------------------------------
MAPPING = {
    "phenotype": {"hpo:HP:0000001": 0, "hpo:HP:0000002": 1, "hpo:HP:0000003": 2},
    "disease": {"mondo:MONDO:0000001": 0, "mondo:MONDO:0000002": 1},
}


def _case(case_id, phenotypes, disease="MONDO:0000001"):
    return cohort_registry.SourceCase(case_id, tuple(phenotypes), disease)


def test_cases_map_to_per_type_indices_in_source_order():
    mapped = map_cases([
        _case("a", ["HP:0000003", "HP:0000001"], "MONDO:0000002"),
        _case("b", ["HP:0000002"]),
    ], MAPPING, "keep")
    assert mapped.samples == [
        {"patient_id": "a", "phenotype_ids": [2, 0], "disease_id": 1},
        {"patient_id": "b", "phenotype_ids": [1], "disease_id": 0},
    ]
    assert mapped.exclusions == []
    assert mapped.counts["ranked_eligible"] == 2 and mapped.counts["n_excluded"] == 0


def test_repeats_are_removed_by_node_through_the_one_rule(monkeypatch):
    calls = []
    real = cohort_registry.normalise_phenotypes

    def spy(identities, positions=None):
        calls.append((list(identities), list(positions)))
        return real(identities, positions=positions)

    monkeypatch.setattr(cohort_registry, "normalise_phenotypes", spy)
    mapped = map_cases([
        _case("a", ["HP:0000002", "HP:9999999", "HP:0000001", "HP:0000002"]),
    ], MAPPING, "keep")
    # Mapped first, then normalised with the source positions: the unknown term
    # at position 1 is gone before the rule sees the list.
    assert calls == [([1, 0, 1], [0, 2, 3])]
    assert mapped.samples[0]["phenotype_ids"] == [1, 0]
    assert mapped.counts["repeats_removed"] == 1
    assert mapped.counts["cases_with_repeats"] == 1


def test_a_partial_case_is_kept_or_excluded_by_policy():
    cases = [_case("p", ["HP:0000001", "HP:9999999"])]
    kept = map_cases(cases, MAPPING, "keep")
    assert kept.samples == [{"patient_id": "p", "phenotype_ids": [0], "disease_id": 0}]
    assert kept.counts["cases_with_lost_terms"] == 1

    excluded = map_cases(cases, MAPPING, "exclude")
    assert excluded.samples == []
    assert excluded.exclusions == [{
        "case_id": "p", "category": "phenotype_unmappable",
        "reasons": ["phenotype_unmappable"], "terms_total": 2, "terms_unmappable": 1,
    }]
    assert excluded.counts["cases_with_lost_terms"] == 0


@pytest.mark.parametrize("policy", ["keep", "exclude"])
def test_a_case_with_no_mappable_term_is_always_excluded(policy):
    mapped = map_cases([_case("z", ["HP:9999999", "HP:9999998"])], MAPPING, policy)
    assert mapped.samples == []
    assert mapped.exclusions[0]["category"] == "phenotype_unmappable"
    assert mapped.exclusions[0]["terms_unmappable"] == 2


def test_ground_truth_absent_takes_precedence_and_every_reason_is_listed():
    mapped = map_cases([
        _case("g", ["HP:0000001"], "MONDO:9999999"),
        _case("both", ["HP:9999999"], "MONDO:9999999"),
    ], MAPPING, "keep")
    assert [(e["case_id"], e["category"], e["reasons"]) for e in mapped.exclusions] == [
        ("g", "ground_truth_absent", ["ground_truth_absent"]),
        ("both", "ground_truth_absent", ["ground_truth_absent", "phenotype_unmappable"]),
    ]
    assert mapped.counts["excluded"] == {"ground_truth_absent": 2, "phenotype_unmappable": 0}


def test_unmappable_terms_count_cases_and_the_counts_add_up():
    mapped = map_cases([
        _case("a", ["HP:9999999", "HP:9999999", "HP:0000001"]),
        _case("b", ["HP:9999999"], "MONDO:9999999"),
        _case("c", ["HP:9999998"]),
        _case("d", ["HP:0000002"]),
    ], MAPPING, "keep")
    # Per term, the number of cases listing it; excluded cases included.
    assert mapped.unmappable_terms == {"HP:9999998": 1, "HP:9999999": 2}
    counts = mapped.counts
    assert counts["terms_unmappable"] == 2
    assert mapped.exclusions[0]["terms_unmappable"] == 1  # b
    assert counts["source_cases"] == counts["ranked_eligible"] + counts["n_excluded"] == 4
    assert counts["n_excluded"] == sum(counts["excluded"].values())
    assert tuple(counts["excluded"]) == EXCLUSION_CATEGORIES
    assert counts["ranked_eligible"] == len(mapped.samples) == 2
    # Per case, listed entries: `a` listed the unknown term twice.
    kept_a = map_cases([_case("a", ["HP:9999999", "HP:9999999"])], MAPPING, "keep")
    assert kept_a.exclusions[0]["terms_unmappable"] == 2


def test_a_graph_without_disease_nodes_excludes_every_case():
    mapped = map_cases([_case("a", ["HP:0000001"])], {"phenotype": MAPPING["phenotype"]}, "keep")
    assert mapped.samples == [] and mapped.exclusions[0]["category"] == "ground_truth_absent"


def test_mapping_refuses_an_unknown_policy():
    with pytest.raises(CohortVersionRefused):
        map_cases([], MAPPING, "maybe")


# ---------------------------------------------------------------------------
# The manifest and the preview id (§0.6, §2.3)
# ---------------------------------------------------------------------------
def _record(sources=(), origin="files"):
    return SimpleNamespace(
        identity=ReadIdentity(Path("/w/kg.provenance.json"), "d" * 64),
        origin=origin,
        record={"sources": list(sources)},
    )


BOUND = {"kg": "a" * 64, "node_features": "b" * 64, "edge_indices": "c" * 64,
         "num_nodes": "e" * 64}


def _manifest(record=None, policy="keep"):
    mapped = map_cases([_case("a", ["HP:0000001"]), _case("b", ["HP:9999999"])], MAPPING, policy)
    workspace = cohort_registry.WorkspaceGraph(BOUND, record or _record(), MAPPING)
    return build_version_manifest(
        label="v1", role="mygene2", source_sha256="f" * 64, source_filename="cases.json",
        workspace=workspace, partial_policy=policy, mapped=mapped, samples_sha256="1" * 64,
    )


def test_the_manifest_records_what_was_read_and_computed():
    record = _record([
        {"role": "hpo", "filename": "hp.obo", "digest": "2" * 64,
         "declared_version": "hp/releases/2024-01-01"},
        {"role": "phenotype_hpoa", "filename": "phenotype.hpoa", "digest": "3" * 64,
         "declared_version": None},
    ])
    manifest = _manifest(record)
    assert manifest["schema_version"] == 1
    assert manifest["what_this_is"] == cohort_registry.WHAT_THIS_IS
    assert manifest["source"] == {"sha256": "f" * 64, "filename": "cases.json",
                                  "format": "shepherd-cases/1", "n_cases": 2}
    assert manifest["graph"] == BOUND
    assert manifest["dataset_record"] == {
        "sha256": "d" * 64, "origin": "files",
        "hpo": {"digest": "2" * 64, "declared_version": "hp/releases/2024-01-01"},
        "mondo": None,
    }
    assert manifest["mapping"] == {"phenotype_normalisation_version": 1,
                                   "phenotype_namespace": "HP", "disease_namespace": "MONDO",
                                   "partial_phenotypes": "keep"}
    assert manifest["samples"] == {"sha256": "1" * 64, "n": 1}
    assert manifest["unmappable_terms"] == {"HP:9999999": 1}
    assert manifest["exclusions"][0]["case_id"] == "b"
    assert "imported" not in manifest
    assert set(manifest) | {"imported"} == set(REQUIRED_VERSION_FIELDS)


def test_the_preview_id_covers_everything_but_the_import_block():
    manifest = _manifest()
    pid = preview_id(manifest)
    assert re.fullmatch(r"[0-9a-f]{16}", pid)
    assert preview_id({**manifest, "imported": {"at": "now"}}) == pid
    assert preview_id(dict(reversed(list(manifest.items())))) == pid
    assert preview_id(_manifest(policy="exclude")) != pid
    assert preview_id({**manifest, "label": "v2"}) != pid
    assert preview_id({**manifest, "graph": {**BOUND, "kg": "0" * 64}}) != pid


# ---------------------------------------------------------------------------
# Preview: reads once, writes nothing
# ---------------------------------------------------------------------------
def test_the_preview_writes_nothing(registered, registry):
    workspace_before = _tree(registered.data_dir)
    plan = prepare_import(registry, registered.data_dir, registered.source, "p1", ROLE, "keep")
    assert plan.manifest["counts"] == GRAPH_SOURCE_COUNTS["keep"]
    assert plan.manifest["unmappable_terms"] == GRAPH_SOURCE_UNMAPPABLE_TERMS
    assert list(registry.iterdir()) == []
    assert _tree(registered.data_dir) == workspace_before


def test_the_exclude_policy_gives_its_own_counts(registered, registry):
    plan = prepare_import(registry, registered.data_dir, registered.source, "p1", ROLE,
                          "exclude")
    assert plan.manifest["counts"] == GRAPH_SOURCE_COUNTS["exclude"]
    assert plan.manifest["mapping"]["partial_phenotypes"] == "exclude"


def test_each_input_is_opened_once(registered, registry):
    from tests.fixtures.opens import count_opens

    with count_opens() as opens:
        prepare_import(registry, registered.data_dir, registered.source, "p1", ROLE, "keep")
    for name in ("split_manifest.json", "kg.json", "node_features.pt", "edge_indices.pt",
                 "num_nodes.json", "kg.provenance.json"):
        assert opens.opens(registered.data_dir / name) == 1, name
    assert opens.opens(registered.source) == 1
    for name in ("train_samples.json", "val_samples.json"):
        assert opens.opens(registered.data_dir / name) == 0, name


@pytest.mark.parametrize("how", ["rename", "rewrite"])
def test_a_source_replaced_after_its_read_is_kept_as_it_was_parsed(
        how, registered, registry, tmp_path, monkeypatch):
    from tests.fixtures.replacement import replace_after_read

    source = tmp_path / "cases.json"
    original = registered.source.read_bytes()
    source.write_bytes(original)
    replacement = replace_after_read(monkeypatch, after=source, target=source,
                                     data=_source_bytes([GOOD_CASE]), how=how)
    plan = prepare_import(registry, registered.data_dir, source, "p1", ROLE, "keep")
    assert replacement.fired == 1
    assert plan.source_sha256 == _sha(original) == plan.manifest["source"]["sha256"]
    assert plan.source_bytes == original
    assert plan.manifest["counts"]["source_cases"] == 13

    version = commit_import(plan, registry, plan.preview_id, now=IMPORTED_AT)
    kept = registry / "sources" / f"{_sha(original)}.json"
    assert kept.read_bytes() == original
    assert version.manifest["source"]["sha256"] == _sha(original)


def test_the_workspace_graph_read_keeps_no_raw_buffer(registered, monkeypatch):
    from src.kg.storage import file_storage
    from tests.fixtures.release import check_release

    graph, reads = check_release(monkeypatch, lambda: read_workspace_graph(registered.data_dir),
                                 bytesio_modules=[file_storage])
    assert sorted(path.name for path in reads) == [
        "edge_indices.pt", "kg.json", "kg.provenance.json", "node_features.pt",
        "num_nodes.json", "split_manifest.json"]
    assert set(graph.bound) == {"kg", "node_features", "edge_indices", "num_nodes"}


def test_the_workspace_graph_is_read_through_the_binding_chain(registered, tmp_path):
    copy = tmp_path / "ws"
    shutil.copytree(registered.data_dir, copy)
    graph = read_workspace_graph(copy)
    assert set(graph.bound) == {"kg", "node_features", "edge_indices", "num_nodes"}
    assert graph.dataset_record.origin == "synthetic"
    assert "hpo:HP:0001250" in graph.node_mapping["phenotype"]

    kg = json.loads((copy / "kg.json").read_text())
    kg["nodes"] = kg["nodes"][:-1]
    (copy / "kg.json").write_text(json.dumps(kg))
    with pytest.raises(ValueError, match="is not the graph"):
        read_workspace_graph(copy)


# ---------------------------------------------------------------------------
# Commit (§0.3, §0.6)
# ---------------------------------------------------------------------------
def test_a_commit_publishes_the_source_the_samples_and_the_manifest(registered, registry,
                                                                     tmp_path):
    from src.kg.storage.file_storage import read_samples

    plan = prepare_import(registry, registered.data_dir, registered.source, "v1", ROLE, "keep")
    version = commit_import(plan, registry, plan.preview_id, now=IMPORTED_AT)

    source = registered.source.read_bytes()
    directory = registry / "versions" / "v1"
    assert sorted(_tree(registry)) == [
        "sources", f"sources/{_sha(source)}.json",
        "versions", "versions/v1", "versions/v1/cohort.json", "versions/v1/samples.json",
    ]
    assert (registry / "sources" / f"{_sha(source)}.json").read_bytes() == source

    manifest_bytes = (directory / "cohort.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    assert manifest_bytes == encode_json(manifest)  # ASCII, sorted, indent 2
    assert version.manifest == manifest
    assert version.identity.sha256 == _sha(manifest_bytes)
    assert version.directory == directory.resolve()
    assert version.samples_path == directory.resolve() / "samples.json"
    assert preview_id(manifest) == plan.preview_id
    assert manifest["imported"]["at"] == "2026-10-16T12:00:00Z"
    assert manifest["imported"]["code_revision"]["state"] in ("recorded", "unrecorded")

    samples_bytes = (directory / "samples.json").read_bytes()
    assert manifest["samples"]["sha256"] == _sha(samples_bytes)
    # The format `read_samples` parses, one row per ranked case.
    (tmp_path / "x_samples.json").write_bytes(samples_bytes)
    samples = read_samples(tmp_path, "x").samples
    assert len(samples) == manifest["samples"]["n"] == manifest["counts"]["ranked_eligible"]
    assert samples[0].patient_id == "case-01"


def test_the_manifest_is_published_last(registered, registry, monkeypatch):
    from src.utils import file_modes

    order = []
    real = file_modes.publish_bytes
    monkeypatch.setattr(file_modes, "publish_bytes",
                        lambda target, data: order.append(Path(target).name) or real(target, data))
    plan = prepare_import(registry, registered.data_dir, registered.source, "v1", ROLE, "keep")
    commit_import(plan, registry, plan.preview_id, now=IMPORTED_AT)
    assert order == [f"{plan.source_sha256}.json", "samples.json", "cohort.json"]


def test_an_interrupted_commit_leaves_a_directory_every_reader_refuses(registered, registry,
                                                                       monkeypatch):
    from src.utils import file_modes

    real = file_modes.publish_bytes

    def fail_on_manifest(target, data):
        if Path(target).name == "cohort.json":
            raise OSError("disk full")
        return real(target, data)

    monkeypatch.setattr(file_modes, "publish_bytes", fail_on_manifest)
    plan = prepare_import(registry, registered.data_dir, registered.source, "v1", ROLE, "keep")
    with pytest.raises(OSError, match="disk full"):
        commit_import(plan, registry, plan.preview_id, now=IMPORTED_AT)
    monkeypatch.undo()

    directory = registry / "versions" / "v1"
    assert (directory / "samples.json").is_file() and not (directory / "cohort.json").exists()
    with pytest.raises(CohortVersionRefused, match="interrupted import"):
        resolve_version(registry, "v1")
    with pytest.raises(CohortVersionRefused, match="interrupted import") as caught:
        prepare_import(registry, registered.data_dir, registered.source, "v1", ROLE, "keep")
    assert str(directory) in str(caught.value)
    assert directory.exists(), "nothing deletes an interrupted import"


def test_a_commit_publishes_only_the_previewed_version(registered, registry):
    plan = prepare_import(registry, registered.data_dir, registered.source, "v1", ROLE, "keep")
    other = prepare_import(registry, registered.data_dir, registered.source, "v1", ROLE,
                           "exclude")
    with pytest.raises(CohortVersionRefused, match="inputs changed since the preview") as caught:
        commit_import(other, registry, plan.preview_id)
    assert "Nothing was written" in str(caught.value)
    assert list(registry.iterdir()) == []


@pytest.mark.parametrize("given", ["", "ABCDEF0123456789", "0123", None])
def test_a_commit_refuses_something_that_is_not_a_preview_id(given, registered, registry):
    plan = prepare_import(registry, registered.data_dir, registered.source, "v1", ROLE, "keep")
    with pytest.raises(CohortVersionRefused, match="is not a preview id"):
        commit_import(plan, registry, given)
    assert list(registry.iterdir()) == []


def test_a_commit_recomputes_the_id_rather_than_trusting_the_plan(registered, registry):
    plan = prepare_import(registry, registered.data_dir, registered.source, "v1", ROLE, "keep")
    forged = plan._replace(manifest={**plan.manifest, "role": "other"})
    with pytest.raises(CohortVersionRefused, match="inputs changed"):
        commit_import(forged, registry, plan.preview_id)
    assert list(registry.iterdir()) == []


def test_a_committed_label_is_never_replaced(registered, registry, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    early = prepare_import(empty, registered.data_dir, registered.source, "v1", ROLE, "keep")
    import_version(registry, registered.data_dir, registered.source, label="v1")
    before = _tree(registry)
    # The preview refuses a taken label, and so does a commit of a plan made
    # before the label was taken.
    with pytest.raises(CohortVersionRefused, match="already exists"):
        prepare_import(registry, registered.data_dir, registered.source, "v1", ROLE, "keep")
    with pytest.raises(CohortVersionRefused, match="immutable") as caught:
        commit_import(early, registry, early.preview_id)
    assert "Nothing was written" in str(caught.value)
    assert _tree(registry) == before


def test_the_label_claim_is_atomic(registered, registry, monkeypatch):
    # Another import takes the label between the pre-check and the claim.
    real = cohort_registry._keep_source

    def keep_then_race(sources, plan):
        real(sources, plan)
        (registry / "versions" / plan.label).mkdir(parents=True)

    monkeypatch.setattr(cohort_registry, "_keep_source", keep_then_race)
    plan = prepare_import(registry, registered.data_dir, registered.source, "v1", ROLE, "keep")
    with pytest.raises(CohortVersionRefused, match="created by another import"):
        commit_import(plan, registry, plan.preview_id)
    assert not (registry / "versions" / "v1" / "samples.json").exists()


def test_a_kept_source_that_does_not_hash_to_its_name_refuses(registered, registry):
    plan = prepare_import(registry, registered.data_dir, registered.source, "v1", ROLE, "keep")
    sources = registry / "sources"
    sources.mkdir()
    damaged = sources / f"{plan.source_sha256}.json"
    damaged.write_bytes(b"not the source")
    with pytest.raises(RegistryRefused, match="registry damaged") as caught:
        commit_import(plan, registry, plan.preview_id)
    assert str(damaged) in str(caught.value) and "nothing was written" in str(caught.value)
    assert damaged.read_bytes() == b"not the source"
    assert not (registry / "versions").exists()


def test_a_source_imported_before_is_kept_once(registered, registry):
    first = import_version(registry, registered.data_dir, registered.source, label="v1")
    second = import_version(registry, registered.data_dir, registered.source, label="v2",
                            partial_phenotypes="exclude")
    assert [p.name for p in (registry / "sources").iterdir()] == [
        f"{first.manifest['source']['sha256']}.json"]
    assert second.manifest["source"] == first.manifest["source"]
    assert second.identity.sha256 != first.identity.sha256


def test_a_version_that_would_rank_nobody_is_refused(registered, registry, tmp_path):
    source = write_source(tmp_path / "none.json", [
        {"case_id": "x", "phenotypes": [UNKNOWN_HPO], "disease": UNKNOWN_MONDO},
    ])
    plan = prepare_import(registry, registered.data_dir, source, "v1", ROLE, "keep")
    assert plan.manifest["samples"]["n"] == 0
    with pytest.raises(CohortVersionRefused, match="would measure nobody") as caught:
        commit_import(plan, registry, plan.preview_id)
    assert "Nothing was written" in str(caught.value)
    assert list(registry.iterdir()) == []


@pytest.mark.parametrize("forge", [
    lambda plan: plan._replace(samples_bytes=b"[]"),
    lambda plan: plan._replace(source_bytes=b"{}"),
    lambda plan: plan._replace(source_sha256="0" * 64),
    lambda plan: plan._replace(label="other"),
], ids=["samples", "source-bytes", "source-name", "label"])
def test_a_plan_whose_bytes_are_not_the_ones_bound_is_not_committed(forge, registered,
                                                                    registry):
    # The preview id covers the manifest; the bytes and the label beside it
    # must be the ones it binds, or a file would be kept under another's name.
    plan = prepare_import(registry, registered.data_dir, registered.source, "v1", ROLE, "keep")
    with pytest.raises(CohortVersionRefused, match="does not bind") as caught:
        commit_import(forge(plan), registry, plan.preview_id)
    assert "Nothing was written" in str(caught.value)
    assert list(registry.iterdir()) == []


def test_a_commit_needs_a_zoned_time_before_it_writes(registered, registry):
    plan = prepare_import(registry, registered.data_dir, registered.source, "v1", ROLE, "keep")
    with pytest.raises(ValueError, match="no time zone"):
        commit_import(plan, registry, plan.preview_id, now=datetime(2026, 10, 16, 12, 0, 0))
    assert list(registry.iterdir()) == []


def test_a_commit_refuses_a_registry_inside_the_repository(registered, registry):
    plan = prepare_import(registry, registered.data_dir, registered.source, "v1", ROLE, "keep")
    with pytest.raises(RegistryRefused, match="inside the repository") as caught:
        commit_import(plan, REPO_ROOT / "data" / "no-such-registry", plan.preview_id)
    assert "Nothing was written" in str(caught.value)


def test_a_preview_refuses_a_registry_inside_the_repository(registered):
    with pytest.raises(RegistryRefused, match="inside the repository"):
        prepare_import(REPO_ROOT / "data" / "no-such-registry", registered.data_dir,
                       registered.source, "v1", ROLE, "keep")


# ---------------------------------------------------------------------------
# Reading a version back
# ---------------------------------------------------------------------------
@pytest.fixture
def committed(registered, registry):
    version = import_version(registry, registered.data_dir, registered.source, label="v1")
    return registry, version


def _rewrite(version: CohortVersionRead, manifest) -> None:
    (version.directory / "cohort.json").write_bytes(
        manifest if isinstance(manifest, bytes) else encode_json(manifest))


def test_a_version_resolves_from_one_read_and_leaves_its_samples_unread(committed):
    from tests.fixtures.opens import count_opens

    registry, version = committed
    with count_opens() as opens:
        again = resolve_version(registry, "v1")
    assert opens.opens(version.directory / "cohort.json") == 1
    assert opens.opens(version.samples_path) == 0
    assert again.manifest == version.manifest
    assert again.identity.sha256 == _sha((version.directory / "cohort.json").read_bytes())


def test_resolving_keeps_no_raw_buffer(committed, monkeypatch):
    from tests.fixtures.release import check_release

    registry, version = committed
    resolved, reads = check_release(monkeypatch, lambda: resolve_version(registry, "v1"))
    assert reads == (version.directory / "cohort.json",)
    assert resolved.identity == version.identity


def test_resolving_refuses_an_unknown_or_invalid_label(committed):
    registry, _ = committed
    with pytest.raises(CohortVersionRefused, match="holds no cohort version 'v9'"):
        resolve_version(registry, "v9")
    with pytest.raises(CohortVersionRefused, match="label is refused"):
        resolve_version(registry, "../v1")


def test_resolving_refuses_a_registry_inside_the_repository():
    with pytest.raises(RegistryRefused):
        resolve_version(REPO_ROOT, "v1")


def _without(manifest, key):
    return {k: v for k, v in manifest.items() if k != key}


@pytest.mark.parametrize("edit, problem", [
    (lambda m: b"\xff not json", "not UTF-8 JSON"),
    (lambda m: [m], "not a JSON object"),
    (lambda m: {**m, "schema_version": 2}, "schema 2"),
    (lambda m: {**m, "schema_version": True}, "schema True"),
    (lambda m: {**m, "label": "v2"}, "names the label 'v2'"),
    (lambda m: _without(m, "samples"), "missing samples"),
    (lambda m: _without(m, "imported"), "missing imported"),
    (lambda m: {**m, "graph": _without(m["graph"], "num_nodes")}, "graph must hold"),
    (lambda m: {**m, "dataset_record": {}}, "dataset_record must hold a sha256"),
    (lambda m: {**m, "samples": {"sha256": "x", "n": 3}}, "samples must hold"),
    (lambda m: {**m, "samples": {**m["samples"], "n": True}}, "samples must hold"),
    (lambda m: {**m, "role": "train"}, "generator's own split names"),
    (lambda m: {**m, "exclusions": {}}, "exclusions must be a list"),
])
def test_resolving_refuses_a_manifest_this_revision_does_not_read(committed, edit, problem):
    registry, version = committed
    _rewrite(version, edit(version.manifest))
    with pytest.raises(CohortVersionRefused) as caught:
        resolve_version(registry, "v1")
    assert problem in str(caught.value)
    assert "cohort.json" in str(caught.value)


# ---------------------------------------------------------------------------
# Cohort and graph binding (§0.7)
# ---------------------------------------------------------------------------
def _version_manifest():
    return {"label": "v1", "graph": dict(BOUND), "dataset_record": {"sha256": "d" * 64}}


def test_a_version_bound_to_the_runs_graph_passes():
    verify_cohort_binding(_version_manifest(), BOUND, "d" * 64)


@pytest.mark.parametrize("role", ["kg", "node_features", "edge_indices", "num_nodes"])
def test_each_differing_graph_role_is_named(role):
    with pytest.raises(CohortBindingMismatch) as caught:
        verify_cohort_binding(_version_manifest(), {**BOUND, role: "0" * 64}, "d" * 64)
    assert caught.value.roles == (role,)
    assert role in str(caught.value) and "'v1'" in str(caught.value)


def test_a_differing_dataset_record_and_several_roles_are_all_named():
    with pytest.raises(CohortBindingMismatch) as caught:
        verify_cohort_binding(_version_manifest(), {**BOUND, "kg": "0" * 64,
                                                    "num_nodes": "0" * 64}, "0" * 64)
    assert caught.value.roles == ("kg", "num_nodes", "dataset_record")


def test_a_role_the_version_does_not_record_is_a_difference():
    manifest = _version_manifest()
    del manifest["graph"]["edge_indices"]
    with pytest.raises(CohortBindingMismatch) as caught:
        verify_cohort_binding(manifest, BOUND, "d" * 64)
    assert caught.value.roles == ("edge_indices",)


def test_the_split_manifest_is_not_bound():
    verify_cohort_binding(_version_manifest(), {**BOUND, "split_manifest": "9" * 64}, "d" * 64)


def test_a_version_from_another_graph_export_is_refused(registered, tmp_path):
    from tests.fixtures.registered_cohort import write_demo_workspace

    other = read_workspace_graph(write_demo_workspace(tmp_path / "other", feature_seed=7))
    with pytest.raises(CohortBindingMismatch) as caught:
        verify_cohort_binding(registered.version.manifest, other.bound,
                              other.dataset_record.identity.sha256)
    assert caught.value.roles == ("node_features",)
    same = read_workspace_graph(registered.data_dir)
    verify_cohort_binding(registered.version.manifest, same.bound,
                          same.dataset_record.identity.sha256)


# ---------------------------------------------------------------------------
# The report directory
# ---------------------------------------------------------------------------
def _fake_version(tmp_path):
    directory = tmp_path / "versions" / "v1"
    directory.mkdir(parents=True)
    return CohortVersionRead({}, ReadIdentity(directory / "cohort.json", "0" * 64), directory,
                             directory / "samples.json")


def test_a_report_directory_is_named_by_time_weights_and_chance(tmp_path):
    version = _fake_version(tmp_path)
    taipei = timezone(timedelta(hours=8))
    run_id, run_dir = create_report_dir(version, "ab" * 32,
                                        datetime(2026, 10, 16, 20, 0, 5, tzinfo=taipei))
    assert re.fullmatch(r"20261016T120005Z-abababababab-[0-9a-f]{8}", run_id)
    assert run_dir == version.directory / "reports" / run_id and run_dir.is_dir()
    assert list(run_dir.iterdir()) == []


def test_two_runs_never_share_a_directory(tmp_path, monkeypatch):
    version = _fake_version(tmp_path)
    monkeypatch.setattr(cohort_registry.secrets, "token_hex", lambda n: "0" * (2 * n))
    create_report_dir(version, "ab" * 32, IMPORTED_AT)
    with pytest.raises(FileExistsError):
        create_report_dir(version, "ab" * 32, IMPORTED_AT)


def test_a_report_directory_needs_a_zoned_time_and_a_digest(tmp_path):
    version = _fake_version(tmp_path)
    with pytest.raises(ValueError, match="no time zone"):
        create_report_dir(version, "ab" * 32, datetime(2026, 10, 16, 12, 0, 0))
    with pytest.raises(ValueError, match="not a SHA-256"):
        create_report_dir(version, "abc", IMPORTED_AT)
    assert not (version.directory / "reports").exists()


# ---------------------------------------------------------------------------
# The registered-cohort fixture
# ---------------------------------------------------------------------------
def test_the_fixture_commits_through_the_production_functions(registered):
    version = registered.version
    assert version.manifest["label"] == LABEL and version.manifest["role"] == ROLE
    assert version.manifest["counts"] == GRAPH_SOURCE_COUNTS["keep"]
    assert resolve_version(registered.registry, LABEL).identity == version.identity
    assert not str(registered.registry.resolve()).startswith(str(REPO_ROOT))


def test_the_fixture_checkpoint_loads_safely_and_matches_the_workspace_graph(registered):
    from src.kg.artifacts import GRAPH_ARTIFACTS
    from src.utils.checkpoint_io import read_checkpoint
    from src.utils.fingerprint import verify_training_record

    read = read_checkpoint(registered.checkpoint, map_location="cpu", weights_only=True)
    assert read.identity.sha256 == registered.checkpoint_sha256
    graph = read_workspace_graph(registered.data_dir)
    verify_training_record(read.checkpoint, graph.bound, tuple(GRAPH_ARTIFACTS),
                           recorded_by="the fixture checkpoint", compared_with="its workspace")
    record = read.checkpoint["training_input_digests"]
    assert set(record) == {"train_samples", "val_samples", "node_features", "edge_indices",
                           "num_nodes", "split_manifest", "kg"}
    for role, name in (("train_samples", "train_samples.json"),
                       ("val_samples", "val_samples.json"),
                       ("split_manifest", "split_manifest.json")):
        assert record[role] == _sha((registered.data_dir / name).read_bytes()), role


@pytest.mark.parametrize("option, roles", [
    ("no_validation", {"train_samples", "node_features", "edge_indices", "num_nodes",
                       "split_manifest", "kg"}),
    ("graph_only", {"node_features", "edge_indices", "num_nodes", "kg"}),
])
def test_the_fixture_record_options(option, roles, registered, tmp_path):
    from src.utils.checkpoint_io import read_checkpoint

    path = tmp_path / "ck" / f"{option}.pt"
    digest = write_fixture_checkpoint(registered.data_dir, path, training_record=option)
    read = read_checkpoint(path, map_location="cpu", weights_only=True)
    assert read.identity.sha256 == digest
    assert set(read.checkpoint["training_input_digests"]) == roles


def test_a_source_from_the_val_cases_overlaps_validation_in_every_case(registered, registry,
                                                                       tmp_path):
    from src.kg.storage.file_storage import read_samples

    source = write_source(tmp_path / "val.json", val_source_cases(registered.data_dir))
    version = import_version(registry, registered.data_dir, source, label="val-1")
    val = read_samples(registered.data_dir, "val").samples
    mapped = json.loads(version.samples_path.read_text())
    assert version.manifest["counts"]["n_excluded"] == 0
    assert [(s["disease_id"], frozenset(s["phenotype_ids"])) for s in mapped] == \
        [(s.disease_id, frozenset(s.phenotype_ids)) for s in val]


def test_mode_c_ranks_the_versions_samples_against_the_fixture_checkpoint(registered):
    """Mode C's own function over the fixture: the model `build_shepherd_model`
    rebuilds from the checkpoint read, the workspace's tensors, and the
    version's samples. The manifest is a stand-in carrying the three fields
    `run_mode_c` reads; building the real one is the measurement entry point's
    job, not this fixture's."""
    import torch

    from src.evaluation.measurement import encode_full_graph, run_mode_c
    from src.kg.data_loader import DiagnosisSample
    from src.kg.storage.file_storage import read_graph_artifacts
    from src.models.gnn.shepherd_gnn import build_shepherd_model
    from src.utils.checkpoint_io import read_checkpoint

    device = torch.device("cpu")
    graph_data = read_graph_artifacts(registered.data_dir).graph_data
    checkpoint = read_checkpoint(registered.checkpoint, map_location=device,
                                 weights_only=True).checkpoint
    model = build_shepherd_model(checkpoint, graph_data, device)
    rows = json.loads(registered.version.samples_path.read_text())
    samples = [DiagnosisSample(patient_id=r["patient_id"], phenotype_ids=r["phenotype_ids"],
                               disease_id=r["disease_id"]) for r in rows]
    stand_in = SimpleNamespace(amp_enabled=False, amp_dtype=None, n_samples=len(samples))
    result = run_mode_c(full_graph_embeddings=encode_full_graph(model, graph_data, device),
                        samples=samples, manifest=stand_in, device=device, batch_size=4)
    assert result.n_ranked == len(samples) == GRAPH_SOURCE_COUNTS["keep"]["ranked_eligible"]
    n_diseases = graph_data["x_dict"]["disease"].size(0)
    assert all(1 <= row["rank"] <= n_diseases for row in result.to_ranks())
    assert [row["sample_id"] for row in result.to_ranks()] == [r["patient_id"] for r in rows]


def test_the_fixture_refuses_a_root_inside_the_repository():
    with pytest.raises(ValueError, match="inside the repository"):
        build_registered_cohort(REPO_ROOT / "data" / "no-such-fixture")
    assert not (REPO_ROOT / "data" / "no-such-fixture").exists()
