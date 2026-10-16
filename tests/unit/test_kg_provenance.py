"""What a graph was built from, and whether the record belongs to it.

`kg.json` has carried a digest since the manifest bound it, and nothing said
which MONDO release, which HPO release or which annotation files produced it.
Two sites whose ontologies differ by a deployment date see different
`kg_digest` values and cannot say what differed — detection without diagnosis.

**The case these tests exist for is the one where every file is well-formed.**
Two graph-only workspaces, a copy, and B's record ends up beside A's graph.
Nothing is corrupt, nothing is missing, and a reader that trusted the directory
would report B's inputs as A's.

**Required since contract M3b.** The record was reported and never enforced;
now training, serving and measurement refuse a workspace whose record is not the
one its manifest binds or does not name its graph (`verify_dataset_record`).
The status reader's states are here as the refusals the formal pipeline makes,
each naming its binding state; the consumers' own refusals are in
`test_dataset_record_required.py`.

Expected digests here are computed from the fixture bytes with `hashlib`, never
by the helper under test — otherwise a writer that hashed the wrong thing would
agree with a test that hashed it the same wrong way.

Module: tests/unit/test_kg_provenance.py
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mode_000_denies_this_process(kind: str) -> bool:
    """Whether `chmod(x, 0)` actually stops **this** process reading.

    **Measured, not inferred.** The precondition these tests need is not "am I
    root" and not "is this POSIX" — it is whether a mode of 000 denies this
    process, and that is one `chmod` and one read away from being known. Asking
    the proxy instead gets three environments wrong: `os.geteuid` does not exist
    on Windows and raises at import, before pytest can skip anything; Windows
    `chmod` sets a read-only bit and never denies a read, so a platform test
    that merely excluded root would run a case that cannot hold there; and a
    filesystem mounted without mode enforcement fails the same way for a
    perfectly ordinary non-root user.

    Returns False on anything unexpected, which skips rather than fails: a
    probe that cannot establish the precondition has not established it.
    """
    try:
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            if kind == "file":
                target = root / "probe.txt"
                target.write_text("x")
                readable = target
            else:
                target = root / "inner"
                target.mkdir()
                readable = target / "probe.txt"
                readable.write_text("x")
            os.chmod(target, 0o000)
            try:
                readable.read_text()
                return False
            except OSError:
                return True
            finally:
                os.chmod(target, 0o700)
    except Exception:
        return False


#: Evaluated once, at import, and only through `os.chmod` and `open` — both of
#: which exist on every platform this runs on.
FILE_MODES_ARE_ENFORCED = _mode_000_denies_this_process("file")

_NO_FILE_MODES = (
    "chmod(000) does not deny this process a read here (root, or a filesystem "
    "that ignores modes), so the refusal cannot be provoked by file mode"
)


def read_provenance(workspace):
    """The record a write produced, held to the reader's rules
    (`validate_record`). For the writer's tests: a graph-only write has no
    manifest to bind it, so the required check is exercised below, where one
    does."""
    from src.kg.provenance import PROVENANCE_FILENAME, validate_record

    return validate_record(json.loads((Path(workspace) / PROVENANCE_FILENAME).read_text()))


def _four_sources():
    """A complete, well-formed source list: every role a real build consumes."""
    from src.kg.provenance import SOURCE_ROLES, source_entry

    return [source_entry(role, f"{role}.input", c * 64)
            for role, c in zip(SOURCE_ROLES, "abcd")]


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
        from src.kg.provenance import PROVENANCE_FILENAME

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
        from src.kg.provenance import SOURCE_ROLES

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
        from src.kg.provenance import SOURCE_ROLES

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

        _, sources = self._sources(tmp_path)
        workspace = tmp_path / "ws"
        _write(workspace, sources=sources)

        by_role = {e["role"]: e for e in read_provenance(workspace)["sources"]}
        assert by_role["mondo"]["declared_version"] == "2026-06-11"
        assert by_role["hpo"]["declared_version"] is None

    def test_changing_only_an_annotation_file_moves_the_record(self, tmp_path):
        """The divergence two ontologies alone could not explain: same MONDO,
        same HPO, same directory, different graph."""
        from src.kg.provenance import source_entry

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


class TestTheRecordRefusesToContradictItselfWhenRead:
    """`validate_record`, the rules the writer and the reader share (contract
    M3b). A record that contradicts itself is read as neither of its claims."""

    @staticmethod
    def _record(**changes):
        from src.kg.provenance import build_provenance

        return {**build_provenance("a" * 64, origin="synthetic"), **changes}

    @pytest.mark.parametrize("changes,fragment", [
        ({"origin": "imagined"}, "not one of"),
        ({"origin": ["files"]}, "not one of"),
        ({"kg_digest": "A" * 64}, "not a SHA-256"),
        ({"sources": {"mondo": "x"}}, "not a list"),
        ({"missing_roles": []}, "contradicts itself"),
        ({"incomplete_by_design": "nothing"}, "not a list of statements"),
        ({"schema_version": 2}, "provenance schema 2"),
        ({"schema_version": True}, "provenance schema True"),
        ({"schema_version": 1.0}, "provenance schema 1.0"),
    ], ids=["origin", "origin-list", "kg-digest", "sources-shape", "missing-roles", "limits",
            "schema", "schema-bool", "schema-float"])
    def test_a_malformed_record_is_refused(self, changes, fragment):
        from src.kg.provenance import ProvenanceError, validate_record

        with pytest.raises(ProvenanceError, match=fragment):
            validate_record(self._record(**changes))

    def test_a_record_without_its_limits_is_refused_not_shown_as_complete(self):
        """Every writer has written `incomplete_by_design`; without it the record
        would be shown as establishing everything."""
        from src.kg.provenance import ProvenanceError, validate_record

        record = self._record()
        del record["incomplete_by_design"]
        with pytest.raises(ProvenanceError, match="is missing"):
            validate_record(record)

    @staticmethod
    def _files_record(**entry_changes):
        """A files record with all four roles, its first source changed."""
        from src.kg.provenance import SOURCE_ROLES, build_provenance, source_entry

        record = build_provenance("a" * 64, sources=[
            source_entry(role, f"{role}.f", c * 64) for role, c in zip(SOURCE_ROLES, "bcde")])
        record["sources"][0] = {**record["sources"][0], **entry_changes}
        return record

    @pytest.mark.parametrize("changes,fragment", [
        ({"digest": "B" * 64}, "not a SHA-256"),
        ({"digest": " " * 64}, "not a SHA-256"),
        ({"filename": 5}, "not a path"),
        ({"filename": ["mondo.obo"]}, "not a path"),
        ({"filename": "/home/operator/mondo.obo"}, "not in the form a build writes"),
        ({"note": "extra"}, "not in the form a build writes"),
    ], ids=["upper-case-digest", "blank-digest", "int-filename", "list-filename",
            "directory-in-filename", "extra-key"])
    def test_a_source_not_in_the_form_a_build_writes_is_refused(self, changes, fragment):
        """Each source is exactly what `source_entry` builds: a digest no file can
        have identifies nothing, and a path or an extra key would be published
        wherever the record is shown."""
        from src.kg.provenance import ProvenanceError, validate_record

        with pytest.raises(ProvenanceError, match=fragment):
            validate_record(self._files_record(**changes))

    def test_the_writer_stores_sources_as_source_entry_builds_them(self):
        """A caller's absolute path and extra keys do not reach the record."""
        from src.kg.provenance import SOURCE_ROLES, build_provenance

        entries = [{"role": role, "filename": f"/home/operator/{role}.f", "digest": c * 64,
                    "operator": "someone"} for role, c in zip(SOURCE_ROLES, "bcde")]
        record = build_provenance("a" * 64, sources=entries)
        assert record["sources"][0] == {"role": "mondo", "filename": "mondo.f",
                                        "digest": "b" * 64, "declared_version": None}

    def test_a_synthetic_record_listing_sources_is_refused(self):
        from src.kg.provenance import ProvenanceError, source_entry, validate_record

        entry = source_entry("mondo", "mondo.obo", "b" * 64)
        with pytest.raises(ProvenanceError, match="synthetic and lists source files"):
            validate_record(self._record(sources=[entry], missing_roles=[]))

    def test_a_role_listed_twice_is_refused(self):
        from src.kg.provenance import ProvenanceError, build_provenance, source_entry

        entries = [source_entry(role, f"{role}.f", c * 64) for role, c in
                   (("mondo", "a"), ("mondo", "b"), ("hpo", "c"),
                    ("phenotype_hpoa", "d"), ("genes_to_phenotype", "e"))]
        with pytest.raises(ProvenanceError, match="more than once"):
            build_provenance("f" * 64, sources=entries)

    def test_a_files_record_leaving_an_input_unidentified_is_not_built(self):
        """The writer's half of the missing-role rule: such a record is refused
        by every consumer, so it is never produced."""
        from src.kg.provenance import ProvenanceError, build_provenance, source_entry

        entries = [source_entry(role, f"{role}.f", "a" * 64)
                   for role in ("hpo", "phenotype_hpoa", "genes_to_phenotype")]
        with pytest.raises(ProvenanceError, match=r"leaves \['mondo'\] unidentified"):
            build_provenance("f" * 64, sources=entries)

    def test_a_synthetic_record_is_not_missing_anything(self):
        """Every role is listed as missing by construction: its origin, not an
        omission."""
        from src.kg.provenance import SOURCE_ROLES, build_provenance, unidentified_roles

        record = build_provenance("f" * 64, origin="synthetic")
        assert record["missing_roles"] == list(SOURCE_ROLES)
        assert unidentified_roles(record) == []


def _bound_workspace(tmp_path, name="ws", kg=None):
    """A workspace with a manifest, through the real writer: its record is
    bound and names its graph."""
    workspace = tmp_path / name
    _write(workspace, kg=kg, samples=_budget())
    return workspace


def _check(workspace):
    """The required check as a consumer makes it: one manifest reading, the
    graph check, then the record against both (contract M3b)."""
    from src.kg.artifacts import read_split_manifest, verify_dataset_record
    from tests.fixtures.workspace_reads import verify_workspace_reads

    observed, _ = verify_workspace_reads(workspace)
    return verify_dataset_record(read_split_manifest(workspace), observed["kg"])


def _refusal(workspace):
    from src.kg.artifacts import DatasetRecordRefused

    with pytest.raises(DatasetRecordRefused) as excinfo:
        _check(workspace)
    return excinfo.value


def _rebind(workspace, record_bytes):
    """Write `record_bytes` as the record and bind them in the manifest, so
    only the record's content can refuse it."""
    from src.kg.artifacts import MANIFEST_FILENAME
    from src.kg.provenance import PROVENANCE_FILENAME

    (workspace / PROVENANCE_FILENAME).write_bytes(record_bytes)
    manifest = json.loads((workspace / MANIFEST_FILENAME).read_text())
    manifest["artifacts"]["provenance"] = hashlib.sha256(record_bytes).hexdigest()
    (workspace / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2))


def _other_kg():
    from src.core.types import DataSource, NodeType
    from src.kg.graph import Node, NodeID
    from scripts.setup_demo import build_demo_kg

    other = build_demo_kg()
    other.add_node(Node(
        id=NodeID(source=DataSource.HPO, local_id="HP:9999999"),
        node_type=NodeType.PHENOTYPE, name="extra",
        attributes={"hpo_id": "HP:9999999", "name": "extra"},
    ))
    return other


class TestTheRecordMustBeTheOneBound:
    """`verify_dataset_record` (contract M3b): anything but *verified* refuses,
    and the refusal names the binding state. These replace the status reader
    that reported and never refused; each of its states is here as the refusal
    the formal pipeline now makes."""

    def test_the_bound_record_of_this_graph_is_verified_and_synthetic(self, tmp_path):
        """The control: without it every refusal below holds for a check that
        refuses everything. Shown with its origin, never as plain verified."""
        from src.kg.provenance import PROVENANCE_FILENAME

        workspace = _bound_workspace(tmp_path)
        record = _check(workspace)

        assert (record.binding, record.origin) == ("verified", "synthetic")
        assert record.identity.sha256 == _sha256(workspace / PROVENANCE_FILENAME)
        assert record.record["kg_digest"] == _sha256(workspace / "kg.json")
        summary = record.summary()
        assert (summary["binding"], summary["origin"]) == ("verified", "synthetic")
        assert summary["incomplete_by_design"], "what the record does not establish is shown"

    def test_another_workspaces_record_in_its_place_is_a_mismatch(self, tmp_path):
        """The swap every file survives well-formed: B's record beside A's graph.
        The manifest's binding catches the bytes."""
        from src.kg.provenance import PROVENANCE_FILENAME

        a, b = _bound_workspace(tmp_path, "a"), _bound_workspace(tmp_path, "b", _other_kg())
        assert _sha256(a / "kg.json") != _sha256(b / "kg.json")
        (a / PROVENANCE_FILENAME).write_bytes((b / PROVENANCE_FILENAME).read_bytes())

        refused = _refusal(a)
        assert refused.binding == "mismatch"
        assert "is not the dataset record" in str(refused)

    def test_another_graphs_record_rebound_in_the_manifest_is_a_mismatch(self, tmp_path):
        """The record *and* the manifest's binding from another build: the bytes
        agree with the binding, and only the graph the record names can tell."""
        from src.kg.provenance import PROVENANCE_FILENAME

        a, b = _bound_workspace(tmp_path, "a"), _bound_workspace(tmp_path, "b", _other_kg())
        _rebind(a, (b / PROVENANCE_FILENAME).read_bytes())

        refused = _refusal(a)
        assert refused.binding == "mismatch"
        assert "describes the graph" in str(refused)
        assert refused.origin == "synthetic"

    def test_a_record_edited_after_the_build_is_a_mismatch(self, tmp_path):
        from src.kg.provenance import PROVENANCE_FILENAME

        workspace = _bound_workspace(tmp_path)
        record = json.loads((workspace / PROVENANCE_FILENAME).read_text())
        record["counters"]["rows_skipped_unresolved_disease_id"] = 999
        (workspace / PROVENANCE_FILENAME).write_text(json.dumps(record, indent=2))

        refused = _refusal(workspace)
        assert refused.binding == "mismatch"
        assert "replaced after the build" in str(refused)

    def test_a_bound_record_that_is_gone_is_unverifiable(self, tmp_path):
        """Lost, not absent: the manifest binds one, so this is a broken
        workspace and not an old one."""
        from src.kg.provenance import PROVENANCE_FILENAME

        workspace = _bound_workspace(tmp_path)
        (workspace / PROVENANCE_FILENAME).unlink()

        refused = _refusal(workspace)
        assert refused.binding == "unverifiable"
        assert "is not here" in str(refused)

    def test_a_manifest_that_binds_no_record_is_unrecorded(self, tmp_path):
        """A workspace from before the record was bound is not supported
        (decision 2): unrecorded refuses, with the file present or not."""
        from src.kg.artifacts import MANIFEST_FILENAME

        workspace = _bound_workspace(tmp_path)
        manifest = json.loads((workspace / MANIFEST_FILENAME).read_text())
        del manifest["artifacts"]["provenance"]
        (workspace / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2))

        refused = _refusal(workspace)
        assert refused.binding == "unrecorded"
        assert "binds no kg.provenance.json" in str(refused)

    @pytest.mark.parametrize("payload,fragment", [
        (b"{not json", "not one this revision reads"),
        (b"\xff\xfe not utf-8", "not one this revision reads"),
        (b"[1, 2]", "not a JSON object"),
    ], ids=["not-json", "not-utf8", "not-an-object"])
    def test_a_bound_file_that_is_not_a_record_is_unverifiable(self, tmp_path, payload, fragment):
        """Bound, so the bytes are the ones the manifest names, and still not a
        record: the binding cannot be established."""
        workspace = _bound_workspace(tmp_path)
        _rebind(workspace, payload)

        refused = _refusal(workspace)
        assert refused.binding == "unverifiable"
        assert fragment in str(refused)

    def test_a_bound_record_with_a_shape_no_rule_names_is_unverifiable(self, tmp_path):
        """A source filename that is not a path, and nesting past the parser's
        limit: refused with the binding state, never escaping as another error."""
        workspace = _bound_workspace(tmp_path)
        for payload in (
            json.dumps({**json.loads((workspace / "kg.provenance.json").read_text()),
                        "origin": "files", "missing_roles": [],
                        "sources": [{"role": role, "filename": 5, "digest": c * 64,
                                     "declared_version": None}
                                    for role, c in zip(
                                        ("mondo", "hpo", "phenotype_hpoa",
                                         "genes_to_phenotype"), "abcd")]}).encode(),
            b"[" * 200_000 + b"]" * 200_000,
        ):
            _rebind(workspace, payload)
            refused = _refusal(workspace)
            assert refused.binding == "unverifiable"
            assert "is not one this revision reads" in str(refused)

    def test_a_refusal_names_a_binding_state_or_is_not_made(self):
        from src.kg.artifacts import DatasetRecordRefused

        with pytest.raises(ValueError, match="not a binding state"):
            DatasetRecordRefused("message", "refused")

    @pytest.mark.parametrize("drop", ["origin", "sources", "kg_digest", "incomplete_by_design"])
    def test_a_bound_record_missing_a_required_field_is_unverifiable(self, tmp_path, drop):
        """Parsing is not being a record: without these it states no origin and
        lists no sources."""
        from src.kg.provenance import PROVENANCE_FILENAME

        workspace = _bound_workspace(tmp_path)
        record = json.loads((workspace / PROVENANCE_FILENAME).read_text())
        del record[drop]
        _rebind(workspace, json.dumps(record).encode())

        refused = _refusal(workspace)
        assert refused.binding == "unverifiable"
        assert "is missing" in str(refused)

    def test_a_bound_files_record_leaving_an_input_unidentified_is_refused(self, tmp_path):
        """Bound and naming this graph, so its binding state is verified; its
        origin is files and an input is unidentified, so it is refused for what
        it says. The check is not relaxed."""
        from src.kg.provenance import PROVENANCE_FILENAME, encode_provenance

        workspace = _bound_workspace(tmp_path)
        record = json.loads((workspace / PROVENANCE_FILENAME).read_text())
        record.update(origin="files", missing_roles=["mondo"], sources=[
            {"role": role, "filename": f"{role}.f", "digest": "a" * 64,
             "declared_version": None}
            for role in ("hpo", "phenotype_hpoa", "genes_to_phenotype")])
        _rebind(workspace, encode_provenance(record).encode())

        refused = _refusal(workspace)
        assert (refused.binding, refused.origin) == ("verified", "files")
        assert "unidentified" in str(refused)

    def test_a_graph_digest_from_another_reading_is_a_caller_error(self, tmp_path):
        """Not a state of the record: the caller paired this manifest reading
        with another graph."""
        from src.kg.artifacts import (
            DatasetRecordRefused,
            read_split_manifest,
            verify_dataset_record,
        )

        workspace = _bound_workspace(tmp_path)
        with pytest.raises(ValueError, match="is not the one") as excinfo:
            verify_dataset_record(read_split_manifest(workspace), "f" * 64)
        assert not isinstance(excinfo.value, DatasetRecordRefused)

    def test_a_read_the_disk_refuses_is_unverifiable(self, tmp_path, monkeypatch):
        """The read raising, on any host: here, as the operating system raises
        it for a file this process may not open. The mode-based test below is
        the same refusal provoked for real, where modes are enforced."""
        import src.utils.fingerprint as fingerprint
        from src.kg.provenance import PROVENANCE_FILENAME

        workspace = _bound_workspace(tmp_path)
        record_path = workspace / PROVENANCE_FILENAME
        real = fingerprint.read_once

        def refusing(path):
            if Path(path) == record_path:
                raise PermissionError(13, "Permission denied", str(path))
            return real(path)

        monkeypatch.setattr(fingerprint, "read_once", refusing)
        refused = _refusal(workspace)
        assert refused.binding == "unverifiable"
        assert "could not be read (PermissionError" in str(refused)

    @pytest.mark.skipif(not FILE_MODES_ARE_ENFORCED, reason=_NO_FILE_MODES)
    def test_a_record_the_disk_refuses_is_unverifiable(self, tmp_path):
        """With a real file mode: the read fails, and the refusal says the
        binding cannot be established rather than that the record is absent."""
        from src.kg.provenance import PROVENANCE_FILENAME

        workspace = _bound_workspace(tmp_path)
        record = workspace / PROVENANCE_FILENAME
        os.chmod(record, 0o000)
        try:
            refused = _refusal(workspace)
        finally:
            os.chmod(record, 0o644)
        assert refused.binding == "unverifiable"
        assert "could not be read" in str(refused)


class TestTheRecordIsReadOnce:
    """Through M2.1, like every other input (contract M3b)."""

    def test_the_file_is_opened_once(self, tmp_path):
        from tests.fixtures.opens import count_opens

        workspace = _bound_workspace(tmp_path)
        with count_opens() as opens:
            _check(workspace)
        assert opens.opens(workspace / "kg.provenance.json") == 1

    def test_the_buffer_is_released(self, monkeypatch, tmp_path):
        from src.kg.artifacts import read_split_manifest, verify_dataset_record
        from tests.fixtures.release import check_release
        from tests.fixtures.workspace_reads import verify_workspace_reads

        workspace = _bound_workspace(tmp_path)
        observed, _ = verify_workspace_reads(workspace)
        manifest = read_split_manifest(workspace)

        _, reads = check_release(
            monkeypatch, lambda: verify_dataset_record(manifest, observed["kg"]))
        assert reads == (workspace / "kg.provenance.json",)

    @pytest.mark.parametrize("how", ["rename", "rewrite"])
    def test_a_replacement_after_the_read_changes_nothing_returned(
        self, monkeypatch, tmp_path, how
    ):
        """The identity returned is of the bytes parsed, whatever the path holds
        a moment later."""
        from src.kg.artifacts import read_split_manifest, verify_dataset_record
        from tests.fixtures.replacement import replace_after_read
        from tests.fixtures.workspace_reads import verify_workspace_reads

        workspace = _bound_workspace(tmp_path)
        target = workspace / "kg.provenance.json"
        original = target.read_bytes()
        observed, _ = verify_workspace_reads(workspace)
        manifest = read_split_manifest(workspace)

        record = replace_after_read(monkeypatch, after=target, target=target,
                                    data=b'{"replaced": true}', how=how)
        result = verify_dataset_record(manifest, observed["kg"])

        assert record.fired == 1
        assert result.identity.sha256 == hashlib.sha256(original).hexdigest()
        assert result.identity.sha256 != _sha256(target)
        assert result.record == json.loads(original)


class TestTheGraphCheckIsItsOwnQuestion:
    """The graph bindings and the record are separate checks with separate
    states; the consumers make both. Neither stands in for the other."""

    def test_the_graph_check_alone_does_not_read_the_record(self, tmp_path):
        from src.kg.provenance import PROVENANCE_FILENAME
        from tests.fixtures.workspace_reads import verify_workspace_reads

        workspace = _bound_workspace(tmp_path)
        (workspace / PROVENANCE_FILENAME).unlink()

        observed, _ = verify_workspace_reads(workspace)
        assert observed["kg"] == _sha256(workspace / "kg.json")

    def test_a_real_tensor_mismatch_is_still_refused(self, tmp_path):
        """A sound record does not loosen the bindings that say the tensors are
        not this graph's. A valid tensor file with other values, not junk."""
        import torch

        from tests.fixtures.workspace_reads import verify_workspace_reads

        workspace = _bound_workspace(tmp_path)
        features = torch.load(workspace / "node_features.pt", weights_only=True)
        torch.save({key: value + 1 for key, value in features.items()},
                   workspace / "node_features.pt")

        with pytest.raises(ValueError, match="is not the node_features artifact"):
            verify_workspace_reads(workspace)


#: Provenance arguments no build can write, each with the fragment of the
#: refusal that says which check caught it. Parametrised rather than written out
#: once, because the defect this class exists for was a *second* encoder that
#: agreed with the writer on some of these and not others: a single unencodable
#: value would have been caught either way, and the pair only diverges on inputs
#: `sort_keys` rejects and the default accepts.
UNWRITABLE_PROVENANCE = {
    "counter-value-no-encoder-takes": (
        {"sources": "four", "counters": {"rows_parsed": object()}},
        "cannot be serialised",
    ),
    # `json.dumps(record)` encodes this; `json.dumps(record, sort_keys=True)`
    # raises `TypeError: '<' not supported between instances of 'str' and 'int'`.
    # This is the input that passed the gate and then failed the writer.
    "counter-keys-of-mixed-type": (
        {"sources": "four", "counters": {"by_identifier": {1: 2, "OMIM": 3}}},
        "cannot be serialised",
    ),
    "source-entry-that-identifies-nothing": (
        {"sources": [{"role": "mondo"}], "counters": None},
        "unusable",
    ),
    # Contract M3b: a record from files that leaves a required input
    # unidentified is refused by every consumer, so it is not written either.
    "a-required-input-unidentified": (
        {"sources": [], "counters": None},
        "unidentified",
    ),
}


def _arguments(kind):
    """The arguments of an `UNWRITABLE_PROVENANCE` case, with "four" standing
    for a complete source list, so only the named defect is wrong."""
    arguments, fragment = UNWRITABLE_PROVENANCE[kind]
    if arguments["sources"] == "four":
        arguments = {**arguments, "sources": _four_sources()}
    return arguments, fragment


class TestTheNewWriterInputsRefuseBeforeTheFirstByte:
    """**Finding 3, and the encoder split that reopened it.**

    The writer's standing contract is that everything knowable from the
    arguments refuses before anything is created. `sources` and
    `source_counters` were added outside it: the record was assembled and
    serialised *after* `kg.json` and three tensors had been written, so a
    counter no encoder takes left a half-built workspace.

    The gate that closed that then called `json.dumps(record)` while the writer
    called `json.dumps(record, indent=2, sort_keys=True)` — two encoders, not
    one question — and a counter with mixed-type keys passed the gate and raised
    in the writer, with the graph already overwritten. So the matrix below runs
    every refusal against a directory that does not exist yet *and* against a
    workspace that already holds a graph, in both states a build can find one.
    """

    @pytest.mark.parametrize("kind", sorted(UNWRITABLE_PROVENANCE))
    def test_a_new_workspace_is_never_created(self, tmp_path, kind):
        from src.kg.workspace import WorkspaceRefusal

        arguments, fragment = _arguments(kind)
        workspace = tmp_path / "ws"

        with pytest.raises(WorkspaceRefusal, match=fragment):
            _write(workspace, **arguments)

        assert not workspace.exists(), (
            "the refusal left the directory it was about to fill"
        )

    @pytest.mark.parametrize("state", ["graph-only", "with-samples"])
    @pytest.mark.parametrize("kind", sorted(UNWRITABLE_PROVENANCE))
    def test_an_existing_workspace_is_left_byte_for_byte(self, tmp_path, kind, state):
        """The loss this prevents: a rebuild that overwrites the graph and then
        discovers its own record will not encode. Both states are covered
        because they end differently — a graph-only build returns before the
        manifest, so the half-written shape it leaves is not the other's."""
        from scripts.setup_demo import build_demo_kg
        from src.kg.workspace import WorkspaceRefusal, write_workspace

        arguments, fragment = _arguments(kind)
        budget = _budget() if state == "with-samples" else None
        workspace = tmp_path / "ws"
        _write(workspace, samples=budget)
        names = sorted(p.name for p in workspace.iterdir())
        for name in names:
            os.utime(workspace / name, (1_600_000_000, 1_600_000_000))
        before = {
            name: ((workspace / name).read_bytes(), (workspace / name).stat().st_mtime_ns)
            for name in names
        }

        # A different feature width, so a write that happened would be visible.
        with pytest.raises(WorkspaceRefusal, match=fragment):
            write_workspace(
                build_demo_kg(), workspace, feature_dim=16, samples=budget,
                sources=arguments["sources"], source_counters=arguments["counters"],
            )

        assert sorted(p.name for p in workspace.iterdir()) == names, (
            "the refusal added or removed a file"
        )
        for name, (payload, mtime) in before.items():
            assert (workspace / name).read_bytes() == payload, f"{name} was rewritten"
            assert (workspace / name).stat().st_mtime_ns == mtime, f"{name} was touched"

    def test_the_gate_and_the_writer_call_one_encoder(self, tmp_path, monkeypatch):
        """The call site, not the rule.

        `counter-keys-of-mixed-type` above proves the two agree on the input
        that split them. This proves *why*: both reach `encode_provenance`, so
        a later edit to the writer's serialisation cannot leave the gate asking
        a question the writer no longer asks. A gate that reverted to
        `json.dumps` would encode once here, not twice.
        """
        import src.kg.provenance as provenance

        encoded = []
        real = provenance.encode_provenance

        def spy(record):
            encoded.append(record)
            return real(record)

        monkeypatch.setattr(provenance, "encode_provenance", spy)
        _write(tmp_path / "ws", sources=_four_sources(), counters={"rows_parsed": 3})

        assert len(encoded) == 2, (
            "expected the pre-write proof and the real write, both through the "
            f"shared encoder; got {len(encoded)}"
        )
        assert encoded[0]["kg_digest"] == "0" * 64, "the gate encoded the real record"
        assert encoded[1]["kg_digest"] != "0" * 64, "the writer wrote the placeholder"

    @pytest.mark.parametrize("state", ["new", "graph-only", "with-samples"])
    def test_the_sound_call_still_writes(self, tmp_path, state):
        """Without this, every refusal above holds for a writer that refuses
        every provenance argument — in every state one can be handed."""

        budget = _budget() if state == "with-samples" else None
        workspace = tmp_path / "ws"
        if state != "new":
            _write(workspace, samples=budget)

        _write(workspace, samples=budget, sources=_four_sources(),
               counters={"rows_parsed": 3})

        assert read_provenance(workspace)["counters"] == {"rows_parsed": 3}


#: Imports this module with the POSIX-only id functions removed, which is what
#: a Windows `os` amounts to for the purpose at hand. Defined once so the
#: simulation and the test of the simulation run the same bytes.
#:
#: **The removal is conditional, and that is the whole point.** `del os.geteuid`
#: raises `AttributeError` on a host that never had it — so an unconditional
#: delete turns this regression test into a failure on exactly the platform it
#: was written to protect, before the import it is checking ever runs.
#:
#: The whole family goes, not just `geteuid`: the defect is *a POSIX-only `os`
#: attribute read at import time*, and pinning the test to one symbol would let
#: the same mistake back in under `getuid`.
_IMPORT_WITH_NO_POSIX_IDS = """
import importlib
import os
import sys

sys.path.insert(0, sys.argv[1])
for name in ("geteuid", "getuid", "getegid", "getgid"):
    if hasattr(os, name):
        delattr(os, name)
    assert not hasattr(os, name), name

importlib.import_module("tests.unit.test_kg_provenance")
print("collected")
"""


class TestThisModuleCollectsOnAPlatformWithoutPosixIds:
    """**A skip condition runs at import, before pytest can skip anything.**

    The first version of the permission guards called `os.geteuid()` in a
    `skipif` decorator. That attribute exists only on Unix; on Windows the call
    happens while the class is being defined, so the whole unit suite fails to
    collect with `AttributeError` — a test that meant to skip taking the run
    down with it. The predicates are measured now and touch only `os.chmod` and
    `open`, which exist everywhere.

    Checked by importing this module in a subprocess whose `os` has had those
    attributes removed, which is what the platform difference amounts to.
    Asserting on the source text instead would pass for any spelling that still
    called them.

    **Neither test here is a Windows run**, and nothing in this file can be:
    both machines this project is developed on are Linux aarch64. What they
    establish is that the module imports without those attributes, and that the
    check itself does not assume the host has them.
    """

    @staticmethod
    def _run(program, *, host_has_posix_ids=True):
        import subprocess
        import sys

        root = Path(__file__).resolve().parents[2]
        if host_has_posix_ids:
            argv = [sys.executable, "-c", program, str(root)]
        else:
            # A host that never had them, so the program meets the same `os` a
            # Windows interpreter would hand it.
            argv = [
                sys.executable, "-c",
                "import os, sys\n"
                "for name in ('geteuid', 'getuid', 'getegid', 'getgid'):\n"
                "    if hasattr(os, name):\n"
                "        delattr(os, name)\n"
                "exec(compile(sys.argv[2], '<program>', 'exec'))\n",
                str(root), program,
            ]
        return subprocess.run(argv, capture_output=True, text=True, cwd=str(root))

    def test_importing_it_without_the_posix_ids_succeeds(self):
        result = self._run(_IMPORT_WITH_NO_POSIX_IDS)

        assert result.returncode == 0, (
            "this module cannot be imported where the POSIX id functions are "
            f"absent, which is every Windows checkout:\n{result.stderr[-2000:]}"
        )
        assert "collected" in result.stdout

    def test_the_check_itself_does_not_assume_a_posix_host(self):
        """**The regression test for the regression test.**

        Its first version opened with an unconditional `del os.geteuid`, which
        raises on a host that does not have it — so on Windows this file failed
        at the one test written to prove it would not. The failure is invisible
        from Linux, where the delete always succeeds, so the host is simulated
        rather than assumed: the program runs against an `os` with the family
        already gone, which is the state a Windows interpreter starts in.

        Still not a Windows run. It is the part of the Windows failure that can
        be reproduced here, and it is the part that was wrong.
        """
        result = self._run(_IMPORT_WITH_NO_POSIX_IDS, host_has_posix_ids=False)

        assert result.returncode == 0, (
            "the check assumes the host has the attributes it removes, so it "
            "fails on the platform it exists to protect:\n"
            f"{result.stderr[-2000:]}"
        )
        assert "collected" in result.stdout
