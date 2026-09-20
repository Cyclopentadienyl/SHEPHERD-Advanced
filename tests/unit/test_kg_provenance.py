"""What a graph was built from, and whether the record belongs to it.

`kg.json` has carried a digest since the manifest bound it, and nothing said
which MONDO release, which HPO release or which annotation files produced it.
Two sites whose ontologies differ by a deployment date see different
`kg_digest` values and cannot say what differed — detection without diagnosis.

**The case these tests exist for is the one where every file is well-formed.**
Two graph-only workspaces, a copy, and B's record ends up beside A's graph.
Nothing is corrupt, nothing is missing, and a reader that trusted the directory
would report B's inputs as A's.

Expected digests here are computed from the fixture bytes with `hashlib`, never
by the helper under test — otherwise a writer that hashed the wrong thing would
agree with a test that hashed it the same wrong way.

Module: tests/unit/test_kg_provenance.py
"""
from __future__ import annotations

import hashlib
import json

import pytest

torch = pytest.importorskip("torch")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(workspace, *, kg=None, samples=None, sources=None, counters=None):
    """A real workspace through the real writer."""
    from scripts.setup_demo import build_demo_kg
    from src.kg.workspace import write_workspace

    return write_workspace(
        kg if kg is not None else build_demo_kg(),
        workspace,
        feature_dim=8,
        samples=samples,
        sources=sources,
        source_counters=counters,
    )


def _budget():
    from src.kg.workspace import SampleBudget

    return SampleBudget(num_train=20, num_val=5, val_disease_fraction=0.2)


class TestTheRecordIsWrittenOnEveryPath:

    @pytest.mark.parametrize("with_samples", [False, True], ids=["graph-only", "with-samples"])
    def test_it_exists_and_names_the_graph_beside_it(self, tmp_path, with_samples):
        """Both modes, because the graph-only build is the one that produces no
        manifest — and so the one a manifest-only record would have missed."""
        from src.kg.provenance import PROVENANCE_FILENAME, read_provenance

        workspace = tmp_path / "ws"
        _write(workspace, samples=_budget() if with_samples else None)

        assert (workspace / PROVENANCE_FILENAME).is_file()
        record = read_provenance(workspace)
        assert record["kg_digest"] == _sha256(workspace / "kg.json"), (
            "the record does not name the graph it was written beside"
        )

    def test_a_synthetic_build_says_so_rather_than_inventing_sources(self, tmp_path):
        """A demo graph has no ontology and no annotation file. Recording four
        digests for it would attribute the graph to inputs nothing opened."""
        from src.kg.provenance import SOURCE_ROLES, read_provenance

        workspace = tmp_path / "ws"
        _write(workspace)

        record = read_provenance(workspace)
        assert record["origin"] == "synthetic"
        assert record["sources"] == []
        assert record["missing_roles"] == list(SOURCE_ROLES)

    def test_the_manifest_binds_the_record(self, tmp_path):
        written = _write(tmp_path / "ws", samples=_budget())

        assert written.manifest["artifacts"]["provenance"] == written.provenance_digest
        assert written.provenance_digest == _sha256(
            tmp_path / "ws" / "kg.provenance.json"
        )


class TestSuppliedSourcesAreRecordedAsGiven:

    @staticmethod
    def _sources(tmp_path):
        from src.kg.provenance import source_entry

        files = {}
        for role, name, body in (
            ("mondo", "mondo.obo", "format-version: 1.2\ndata-version: 2026-06-11\n"),
            ("hpo", "hpo.obo", "format-version: 1.2\n"),
            ("phenotype_hpoa", "phenotype.hpoa", "#desc\nOMIM:1\tx\t\tHP:1\n"),
            ("genes_to_phenotype", "genes_to_phenotype.txt", "g\tHP:1\n"),
        ):
            path = tmp_path / name
            path.write_text(body)
            files[role] = path
        return files, [
            source_entry("mondo", files["mondo"], _sha256(files["mondo"]), "2026-06-11"),
            source_entry("hpo", files["hpo"], _sha256(files["hpo"]), None),
            source_entry("phenotype_hpoa", files["phenotype_hpoa"],
                         _sha256(files["phenotype_hpoa"])),
            source_entry("genes_to_phenotype", files["genes_to_phenotype"],
                         _sha256(files["genes_to_phenotype"])),
        ]

    def test_all_four_roles_land_with_their_digests(self, tmp_path):
        from src.kg.provenance import SOURCE_ROLES, read_provenance

        files, sources = self._sources(tmp_path)
        workspace = tmp_path / "ws"
        _write(workspace, sources=sources)

        record = read_provenance(workspace)
        assert record["origin"] == "files"
        assert record["missing_roles"] == []
        by_role = {entry["role"]: entry for entry in record["sources"]}
        assert set(by_role) == set(SOURCE_ROLES)
        for role, path in files.items():
            assert by_role[role]["digest"] == _sha256(path)

    def test_a_declared_version_is_kept_and_an_absent_one_stays_null(self, tmp_path):
        """`Ontology.version` falls back to the OBO format version, so a record
        built from it would report `1.2` as though it were a release. Absence is
        recorded as absence."""
        from src.kg.provenance import read_provenance

        _, sources = self._sources(tmp_path)
        workspace = tmp_path / "ws"
        _write(workspace, sources=sources)

        by_role = {e["role"]: e for e in read_provenance(workspace)["sources"]}
        assert by_role["mondo"]["declared_version"] == "2026-06-11"
        assert by_role["hpo"]["declared_version"] is None

    def test_changing_only_an_annotation_file_moves_the_record(self, tmp_path):
        """The divergence two ontologies alone could not explain: same MONDO,
        same HPO, same directory, different graph."""
        from src.kg.provenance import read_provenance, source_entry

        files, sources = self._sources(tmp_path)
        first = tmp_path / "first"
        _write(first, sources=sources)
        before = read_provenance(first)

        files["phenotype_hpoa"].write_text("#desc\nOMIM:1\tx\t\tHP:1\nOMIM:2\ty\t\tHP:2\n")
        changed = [e for e in sources if e["role"] != "phenotype_hpoa"] + [
            source_entry("phenotype_hpoa", files["phenotype_hpoa"],
                         _sha256(files["phenotype_hpoa"]))
        ]
        second = tmp_path / "second"
        _write(second, sources=changed)
        after = read_provenance(second)

        def digest_of(record, role):
            return {e["role"]: e["digest"] for e in record["sources"]}[role]

        assert digest_of(before, "phenotype_hpoa") != digest_of(after, "phenotype_hpoa")
        assert digest_of(before, "mondo") == digest_of(after, "mondo"), (
            "the ontologies did not change and their digests should not have"
        )

    def test_the_counters_name_what_they_count(self, tmp_path):
        from src.kg.provenance import read_provenance

        _, sources = self._sources(tmp_path)
        workspace = tmp_path / "ws"
        _write(workspace, sources=sources,
               counters={"rows_skipped_unresolved_disease_id": 7})

        counters = read_provenance(workspace)["counters"]
        assert counters["rows_skipped_unresolved_disease_id"] == 7
        assert not any("disease" == k for k in counters), (
            "a count of annotation rows must not be named as a count of diseases"
        )

    def test_the_limits_travel_with_the_record(self, tmp_path):
        """A source list is not a rebuild recipe, and the file says so rather
        than leaving a reader to assume."""
        from src.kg.provenance import read_provenance

        _, sources = self._sources(tmp_path)
        workspace = tmp_path / "ws"
        _write(workspace, sources=sources)

        limits = read_provenance(workspace)["incomplete_by_design"]
        assert any("import" in line for line in limits)
        assert any("version" in line for line in limits)


class TestTheRecordRefusesToContradictItself:
    """Direct-API guards. No writer can reach these today — `write_workspace`
    derives the origin from whether sources were supplied — but
    `build_provenance` is importable, and a record that says a graph had no
    inputs while listing four of them is worse than no record.
    """

    def test_a_synthetic_origin_cannot_carry_sources(self):
        from src.kg.provenance import ProvenanceError, build_provenance, source_entry

        with pytest.raises(ProvenanceError, match="never had"):
            build_provenance(
                kg_digest="a" * 64,
                sources=[source_entry("mondo", "mondo.obo", "b" * 64)],
                origin="synthetic",
            )

    def test_a_record_without_its_graph_is_refused(self):
        """The binding is not optional. A record assembled without a graph
        digest could never be checked against anything."""
        from src.kg.provenance import ProvenanceError, build_provenance

        with pytest.raises(ProvenanceError, match="which graph it describes"):
            build_provenance(kg_digest="too short")

    @pytest.mark.parametrize("role,digest,fragment", [
        ("mitochondria", "c" * 64, "not one of the roles"),
        ("mondo", "not-a-digest", "not a SHA-256"),
    ], ids=["unknown-role", "malformed-digest"])
    def test_an_entry_that_identifies_nothing_is_refused(self, role, digest, fragment):
        from src.kg.provenance import ProvenanceError, source_entry

        with pytest.raises(ProvenanceError, match=fragment):
            source_entry(role, "some.obo", digest)


class TestARecordThatDoesNotBelongIsCaught:
    """The scenario that made the downward binding necessary."""

    def test_two_graph_only_workspaces_with_swapped_records(self, tmp_path):
        from src.kg.provenance import PROVENANCE_FILENAME, ProvenanceError, verify_provenance
        from src.core.types import DataSource, NodeType
        from src.kg.graph import Node, NodeID
        from scripts.setup_demo import build_demo_kg

        a = tmp_path / "a"
        _write(a)

        other = build_demo_kg()
        other.add_node(Node(
            id=NodeID(source=DataSource.HPO, local_id="HP:9999999"),
            node_type=NodeType.PHENOTYPE, name="extra",
            attributes={"hpo_id": "HP:9999999", "name": "extra"},
        ))
        b = tmp_path / "b"
        _write(b, kg=other)

        assert _sha256(a / "kg.json") != _sha256(b / "kg.json"), (
            "the fixture built the same graph twice; the swap would prove nothing"
        )

        (a / PROVENANCE_FILENAME).write_bytes((b / PROVENANCE_FILENAME).read_bytes())

        with pytest.raises(ProvenanceError, match="describes another build"):
            verify_provenance(a, _sha256(a / "kg.json"))

    def test_the_matching_pair_reads_back(self, tmp_path):
        """Without this, the refusal above would hold for a verifier that
        rejects everything."""
        from src.kg.provenance import verify_provenance

        workspace = tmp_path / "ws"
        _write(workspace)

        record = verify_provenance(workspace, _sha256(workspace / "kg.json"))
        assert record is not None and record["origin"] == "synthetic"

    def test_the_status_reader_catches_a_record_from_another_build(self, tmp_path):
        """The call site, not the rule.

        The swap test above calls `verify_provenance` directly, which proves the
        rule and nothing about who applies it. This goes through
        `workspace_provenance_status`, which is what a caller reaches for, and
        builds the case that needs the graph binding: the record *and* the
        manifest entry both come from another build, so the declared-digest
        check agrees and only the graph binding can tell.
        """
        from src.core.types import DataSource, NodeType
        from src.kg.artifacts import MANIFEST_FILENAME, workspace_provenance_status
        from src.kg.graph import Node, NodeID
        from src.kg.provenance import PROVENANCE_FILENAME
        from scripts.setup_demo import build_demo_kg

        a = tmp_path / "a"
        _write(a, samples=_budget())

        other = build_demo_kg()
        other.add_node(Node(
            id=NodeID(source=DataSource.HPO, local_id="HP:9999999"),
            node_type=NodeType.PHENOTYPE, name="extra",
            attributes={"hpo_id": "HP:9999999", "name": "extra"},
        ))
        b = tmp_path / "b"
        written_b = _write(b, kg=other, samples=_budget())
        assert _sha256(a / "kg.json") != _sha256(b / "kg.json")

        (a / PROVENANCE_FILENAME).write_bytes((b / PROVENANCE_FILENAME).read_bytes())
        manifest = json.loads((a / MANIFEST_FILENAME).read_text())
        manifest["artifacts"]["provenance"] = written_b.provenance_digest
        (a / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2))

        status = workspace_provenance_status(a)

        assert status.state == "mismatched"
        assert "another build" in status.detail
        assert status.record is None, "a record that does not belong must not be returned"

    def test_a_declared_record_that_was_replaced_is_reported(self, tmp_path):
        from src.kg.artifacts import workspace_provenance_status
        from src.kg.provenance import PROVENANCE_FILENAME

        workspace = tmp_path / "ws"
        _write(workspace, samples=_budget())
        assert workspace_provenance_status(workspace).state == "recorded"  # control

        record = json.loads((workspace / PROVENANCE_FILENAME).read_text())
        record["counters"]["rows_skipped_unresolved_disease_id"] = 999
        (workspace / PROVENANCE_FILENAME).write_text(json.dumps(record, indent=2))

        status = workspace_provenance_status(workspace)
        assert status.state == "mismatched"
        assert "replaced after the build" in status.detail

    def test_a_declared_record_that_is_gone_is_missing_not_absent(self, tmp_path):
        """**Finding 2.** The file's absence alone cannot say which state this
        is; the manifest's declaration is what separates a broken workspace from
        an old one, and calling both `absent` would name opposite situations
        alike."""
        from src.kg.artifacts import workspace_provenance_status
        from src.kg.provenance import PROVENANCE_FILENAME

        workspace = tmp_path / "ws"
        _write(workspace, samples=_budget())
        (workspace / PROVENANCE_FILENAME).unlink()

        status = workspace_provenance_status(workspace)

        assert status.state == "missing"
        assert status.state != "absent"

    def test_an_unbound_record_is_not_reported_as_bound(self, tmp_path):
        """Graph-only: the record matches the graph, and nothing declared it, so
        it could have been placed there. Distinguished rather than accepted."""
        from src.kg.artifacts import workspace_provenance_status

        workspace = tmp_path / "ws"
        _write(workspace)

        status = workspace_provenance_status(workspace)
        assert status.state == "undeclared_present"
        assert status.record is not None


class TestProvenanceDoesNotGateService:
    """**Finding 1.** The reason this is its own class.

    `verify_graph_artifacts` is not only an audit: `verify_graph_source` calls
    it, and `DiagnosisPipeline._init_gnn_inference` calls that before loading a
    tensor. An earlier revision raised from it when a declared record was
    missing, so a lost note about where a graph came from stopped a model from
    initialising — and through the API a cold start then answered `/diagnose`
    with 503 while the tensors were sound.

    Reporting provenance is Phase 1's scope. Refusing on it is a policy nobody
    approved.
    """

    @staticmethod
    def _break_it(workspace, how):
        from src.kg.provenance import PROVENANCE_FILENAME

        path = workspace / PROVENANCE_FILENAME
        if how == "deleted":
            path.unlink()
        elif how == "replaced":
            record = json.loads(path.read_text())
            record["kg_digest"] = "f" * 64
            path.write_text(json.dumps(record))
        else:
            path.write_text("{not json")

    @pytest.mark.parametrize("how", ["deleted", "replaced", "corrupt"])
    def test_the_artifact_verifier_still_accepts_a_sound_graph(self, tmp_path, how):
        from src.kg.artifacts import verify_graph_artifacts

        workspace = tmp_path / "ws"
        _write(workspace, samples=_budget())
        self._break_it(workspace, how)

        observed = verify_graph_artifacts(workspace)

        assert observed["kg"] == _sha256(workspace / "kg.json")

    @pytest.mark.parametrize("how", ["deleted", "replaced", "corrupt"])
    def test_the_graph_source_check_the_pipeline_runs_still_passes(self, tmp_path, how):
        """The exact function `_init_gnn_inference` calls before a tensor is
        read. This is the assertion that would have caught the defect."""
        from src.kg.artifacts import verify_graph_source

        workspace = tmp_path / "ws"
        _write(workspace, samples=_budget())
        self._break_it(workspace, how)

        verify_graph_source(workspace / "kg.json", workspace)

    def test_a_real_tensor_mismatch_is_still_refused(self, tmp_path):
        """The other half: loosening provenance must not loosen the bindings
        that say the tensors are not this graph's."""
        from src.kg.artifacts import verify_graph_artifacts

        workspace = tmp_path / "ws"
        _write(workspace, samples=_budget())
        (workspace / "node_features.pt").write_bytes(b"another graph's tensor")

        with pytest.raises(ValueError, match="is not the node_features artifact"):
            verify_graph_artifacts(workspace)


class TestAWorkspaceFromBeforeThisStillWorks:
    """Provenance must not become a file every existing deployment has to
    rebuild for."""

    def test_no_record_reads_as_unknown_rather_than_as_a_failure(self, tmp_path):
        from src.kg.provenance import PROVENANCE_FILENAME, read_provenance, verify_provenance

        workspace = tmp_path / "ws"
        _write(workspace)
        (workspace / PROVENANCE_FILENAME).unlink()

        assert read_provenance(workspace) is None
        assert verify_provenance(workspace, _sha256(workspace / "kg.json")) is None

    def test_an_undeclared_record_does_not_stop_artifact_verification(self, tmp_path):
        """A schema-3 workspace built before provenance existed declares none in
        its manifest, and every other binding still holds."""
        from src.kg.artifacts import MANIFEST_FILENAME, verify_graph_artifacts
        from src.kg.provenance import PROVENANCE_FILENAME
        from src.utils.fingerprint import file_sha256

        workspace = tmp_path / "ws"
        _write(workspace, samples=_budget())

        manifest = json.loads((workspace / MANIFEST_FILENAME).read_text())
        del manifest["artifacts"]["provenance"]
        (workspace / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2))
        (workspace / PROVENANCE_FILENAME).unlink()

        observed = verify_graph_artifacts(workspace)

        assert "provenance" not in observed
        assert observed["kg"] == file_sha256(workspace / "kg.json")

    def test_a_corrupt_record_is_reported_rather_than_read_as_absent(self, tmp_path):
        """The state an implementation is most likely to collapse into
        "unknown": the file is there and says nothing usable."""
        from src.kg.provenance import PROVENANCE_FILENAME, ProvenanceError, read_provenance

        workspace = tmp_path / "ws"
        _write(workspace)
        (workspace / PROVENANCE_FILENAME).write_text("{not json")

        with pytest.raises(ProvenanceError, match="not readable JSON"):
            read_provenance(workspace)


class TestTheNewWriterInputsRefuseBeforeTheFirstByte:
    """**Finding 3.** `sources` and `source_counters` arrived after the graph.

    The writer's standing contract is that everything knowable from the
    arguments refuses before anything is created. These two parameters were
    added outside it: the record was assembled and serialised *after* `kg.json`
    and three tensors had been written, so a counter no encoder takes left a
    half-built workspace — and over an existing one, new tensors beside the
    previous manifest.
    """

    @staticmethod
    def _bad_counters():
        return {"rows_skipped_unresolved_disease_id": object()}

    def test_a_new_workspace_is_never_created(self, tmp_path):
        from src.kg.workspace import WorkspaceRefusal

        workspace = tmp_path / "ws"

        with pytest.raises(WorkspaceRefusal, match="cannot be serialised"):
            _write(workspace, sources=[], counters=self._bad_counters())

        assert not workspace.exists(), (
            "the refusal left the directory it was about to fill"
        )

    def test_an_existing_workspace_is_left_byte_for_byte(self, tmp_path):
        """The loss this prevents: a rebuild that overwrites the graph and then
        discovers its own record will not encode."""
        import os

        from src.kg.workspace import WorkspaceRefusal

        workspace = tmp_path / "ws"
        _write(workspace, samples=_budget())
        names = sorted(p.name for p in workspace.iterdir())
        for name in names:
            os.utime(workspace / name, (1_600_000_000, 1_600_000_000))
        before = {
            name: ((workspace / name).read_bytes(), (workspace / name).stat().st_mtime_ns)
            for name in names
        }

        # A different feature width, so a write that happened would be visible.
        with pytest.raises(WorkspaceRefusal, match="cannot be serialised"):
            from src.kg.workspace import write_workspace
            from scripts.setup_demo import build_demo_kg

            write_workspace(
                build_demo_kg(), workspace, feature_dim=16,
                sources=[], source_counters=self._bad_counters(),
            )

        for name, (payload, mtime) in before.items():
            assert (workspace / name).read_bytes() == payload, f"{name} was rewritten"
            assert (workspace / name).stat().st_mtime_ns == mtime, f"{name} was touched"

    def test_a_malformed_source_entry_is_refused_as_a_workspace_refusal(self, tmp_path):
        """One vocabulary. The build script translates nothing, so a bad entry
        has to arrive as the refusal every other bad input arrives as."""
        from src.kg.workspace import WorkspaceRefusal

        workspace = tmp_path / "ws"

        with pytest.raises(WorkspaceRefusal, match="unusable"):
            _write(workspace, sources=[{"role": "mondo"}])

        assert not workspace.exists()

    def test_the_sound_call_still_writes(self, tmp_path):
        """Without this, every refusal above holds for a writer that refuses
        every provenance argument."""
        from src.kg.provenance import read_provenance

        workspace = tmp_path / "ws"
        _write(workspace, sources=[], counters={"rows_parsed": 3})

        assert read_provenance(workspace)["counters"] == {"rows_parsed": 3}


class TestAFileThatParsesIsNotARecord:
    """Parsing as JSON says nothing about being provenance.

    A file carrying only a schema version and a graph digest states no origin
    and lists no sources. A reader that accepted it would report the build's
    inputs as an empty set rather than as unrecorded — the same conflation
    between "nothing there" and "nothing known" that the missing/absent split
    exists to prevent.
    """

    @pytest.mark.parametrize("drop", ["origin", "sources", "kg_digest"])
    def test_a_record_missing_a_required_field_is_refused(self, tmp_path, drop):
        from src.kg.provenance import PROVENANCE_FILENAME, ProvenanceError, read_provenance

        workspace = tmp_path / "ws"
        _write(workspace)
        record = json.loads((workspace / PROVENANCE_FILENAME).read_text())
        del record[drop]
        (workspace / PROVENANCE_FILENAME).write_text(json.dumps(record))

        with pytest.raises(ProvenanceError, match="is missing"):
            read_provenance(workspace)

    def test_such_a_file_reports_as_unreadable_rather_than_recorded(self, tmp_path):
        """Through the status reader, so the state a caller sees is the honest
        one rather than a record with fields missing."""
        from src.kg.artifacts import workspace_provenance_status
        from src.kg.provenance import PROVENANCE_FILENAME

        workspace = tmp_path / "ws"
        _write(workspace, samples=_budget())
        record = json.loads((workspace / PROVENANCE_FILENAME).read_text())
        del record["sources"]
        (workspace / PROVENANCE_FILENAME).write_text(json.dumps(record))

        status = workspace_provenance_status(workspace)

        # The manifest still declares the old digest, so the replacement is seen
        # first; either way it is not `recorded`, which is the contract.
        assert status.state in ("unreadable", "mismatched")
        assert not status.is_known
