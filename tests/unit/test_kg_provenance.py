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
DIRECTORY_MODES_ARE_ENFORCED = _mode_000_denies_this_process("directory")

_NO_FILE_MODES = (
    "chmod(000) does not deny this process a read here (root, or a filesystem "
    "that ignores modes), so the guard cannot be provoked by file mode"
)
_NO_DIRECTORY_MODES = (
    "chmod(000) on a directory does not stop this process entering it here, so "
    "the guard cannot be provoked by directory mode"
)


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


#: Provenance arguments no build can write, each with the fragment of the
#: refusal that says which check caught it. Parametrised rather than written out
#: once, because the defect this class exists for was a *second* encoder that
#: agreed with the writer on some of these and not others: a single unencodable
#: value would have been caught either way, and the pair only diverges on inputs
#: `sort_keys` rejects and the default accepts.
UNWRITABLE_PROVENANCE = {
    "counter-value-no-encoder-takes": (
        {"sources": [], "counters": {"rows_parsed": object()}},
        "cannot be serialised",
    ),
    # `json.dumps(record)` encodes this; `json.dumps(record, sort_keys=True)`
    # raises `TypeError: '<' not supported between instances of 'str' and 'int'`.
    # This is the input that passed the gate and then failed the writer.
    "counter-keys-of-mixed-type": (
        {"sources": [], "counters": {"by_identifier": {1: 2, "OMIM": 3}}},
        "cannot be serialised",
    ),
    "source-entry-that-identifies-nothing": (
        {"sources": [{"role": "mondo"}], "counters": None},
        "unusable",
    ),
}


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

        arguments, fragment = UNWRITABLE_PROVENANCE[kind]
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

        arguments, fragment = UNWRITABLE_PROVENANCE[kind]
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
        _write(tmp_path / "ws", sources=[], counters={"rows_parsed": 3})

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
        from src.kg.provenance import read_provenance

        budget = _budget() if state == "with-samples" else None
        workspace = tmp_path / "ws"
        if state != "new":
            _write(workspace, samples=budget)

        _write(workspace, samples=budget, sources=[], counters={"rows_parsed": 3})

        assert read_provenance(workspace)["counters"] == {"rows_parsed": 3}


class TestTheStatusEntryReportsWhenTheDiskRefuses:
    """**`workspace_provenance_status` says it never raises; the disk was outside
    that.**

    Both files it opens — the manifest that says whether a record was declared,
    and the `kg.json` a record is checked against — are opened after an
    `is_file()` check that says nothing about whether *this process* may read
    them. A mode that excludes the caller, a volume that errors, a manifest
    written half-way: each produced an exception out of a function documented to
    report, and every caller was written not to wrap it. A note beside the graph
    would then stop a pipeline it was explicitly kept out of the way of.

    **And the existence probes, not only the reads.** `Path.is_file` calls
    `stat`, and `pathlib` re-raises EACCES rather than returning False — only
    ENOENT, ENOTDIR, EBADF and ELOOP become False. So a workspace directory this
    process cannot traverse fails at *is there a graph here*, before anything is
    opened. Reproduced as a real non-root process against the previous commit:
    `PermissionError` naming `kg.json`, straight out of the entry point.

    The permission cases are the honest ones and they need an environment where
    a mode of 000 actually denies a read; `FILE_MODES_ARE_ENFORCED` and
    `DIRECTORY_MODES_ARE_ENFORCED` measure that rather than guessing it from the
    platform or the user id. Where it does not hold they skip and the
    substitution tests carry the guard. The decode and manifest cases are real
    everywhere.
    """

    @staticmethod
    def _refuse_reads_of(monkeypatch, filename):
        """Make `file_sha256` fail for one file, the way a mode of 000 would."""
        from pathlib import Path as _Path

        import src.utils.fingerprint as fingerprint

        real = fingerprint.file_sha256

        def guard(path):
            if _Path(path).name == filename:
                raise PermissionError(13, "Permission denied", str(path))
            return real(path)

        monkeypatch.setattr(fingerprint, "file_sha256", guard)

    @pytest.mark.parametrize(
        "target", ["kg.json", "kg.provenance.json"],
        ids=["graph-unreadable", "declared-record-unreadable"],
    )
    def test_a_file_that_cannot_be_hashed_is_a_state(self, tmp_path, monkeypatch, target):
        from src.kg.artifacts import workspace_provenance_status

        workspace = tmp_path / "ws"
        _write(workspace, samples=_budget())  # a manifest, so the record is declared
        assert workspace_provenance_status(workspace).state == "recorded"  # control

        self._refuse_reads_of(monkeypatch, target)
        status = workspace_provenance_status(workspace)

        assert status.state == "unreadable"
        assert "PermissionError" in status.detail, (
            "the cause was dropped; a reader cannot tell a permission problem "
            f"from a corrupt file: {status.detail}"
        )
        # The boundary around the whole body would also report `unreadable`
        # with the type in it, so this asserts the *specific* guard ran: only
        # that one names the file and what its loss leaves unestablished.
        assert "could not be read" in status.detail, status.detail
        assert "could not be inspected" not in status.detail, (
            "the named guard was lost and the catch-all boundary answered "
            f"instead, which cannot say which file failed: {status.detail}"
        )
        assert not status.is_known

    @pytest.mark.skipif(not FILE_MODES_ARE_ENFORCED, reason=_NO_FILE_MODES)
    @pytest.mark.parametrize(
        "target", ["kg.json", "kg.provenance.json"],
        ids=["graph-unreadable", "declared-record-unreadable"],
    )
    def test_the_same_thing_with_a_real_file_mode(self, tmp_path, target):
        from src.kg.artifacts import workspace_provenance_status

        workspace = tmp_path / "ws"
        _write(workspace, samples=_budget())
        os.chmod(workspace / target, 0o000)
        try:
            status = workspace_provenance_status(workspace)
        finally:
            os.chmod(workspace / target, 0o644)

        assert status.state == "unreadable"
        assert not status.is_known

    @pytest.mark.parametrize(
        "scope", ["kg.json", "whole-workspace"],
        ids=["graph-probe-fails", "nothing-can-be-stat-ed"],
    )
    def test_an_existence_probe_that_fails_is_a_state(self, tmp_path, monkeypatch, scope):
        """**`is_file()` is a read.** It calls `stat`, and EACCES comes back out
        rather than as `False`, so the question *is there a graph here* fails
        before anything is opened. Substituted so this runs everywhere; the
        directory-mode test below is the same condition for real."""
        from src.kg.artifacts import workspace_provenance_status

        workspace = tmp_path / "ws"
        _write(workspace, samples=_budget())
        assert workspace_provenance_status(workspace).state == "recorded"  # control

        real_stat = Path.stat

        def guard(self, *args, **kwargs):
            blocked = str(self) == str(workspace / "kg.json") if scope == "kg.json" \
                else str(self).startswith(str(workspace))
            if blocked:
                raise PermissionError(13, "Permission denied", str(self))
            return real_stat(self, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", guard)
        status = workspace_provenance_status(workspace)

        assert status.state == "unreadable", (
            f"a failed stat was reported as {status.state!r}; not being able to "
            "find out is not the same as the answer being no"
        )
        assert "PermissionError" in status.detail
        assert not status.is_known

    def test_the_low_level_entry_reports_a_failed_probe_too(self, tmp_path, monkeypatch):
        """`provenance_status` carries the same promise and is reachable on its
        own, so it needs the boundary rather than inheriting one."""
        from src.kg.provenance import PROVENANCE_FILENAME, provenance_status

        workspace = tmp_path / "ws"
        _write(workspace)
        real_stat = Path.stat

        def guard(self, *args, **kwargs):
            if self.name == PROVENANCE_FILENAME:
                raise PermissionError(13, "Permission denied", str(self))
            return real_stat(self, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", guard)
        status = provenance_status(workspace, _sha256(workspace / "kg.json"))

        assert status.state == "unreadable"
        assert "PermissionError" in status.detail

    @pytest.mark.skipif(not DIRECTORY_MODES_ARE_ENFORCED, reason=_NO_DIRECTORY_MODES)
    def test_a_workspace_that_cannot_be_entered_with_a_real_directory_mode(self, tmp_path):
        """The condition as the filesystem produces it: the files are intact and
        the directory cannot be traversed, so every probe inside it fails."""
        from src.kg.artifacts import workspace_provenance_status

        workspace = tmp_path / "ws"
        _write(workspace, samples=_budget())
        os.chmod(workspace, 0o000)
        try:
            status = workspace_provenance_status(workspace)
        finally:
            os.chmod(workspace, 0o755)

        assert status.state == "unreadable"
        assert not status.is_known

    def test_a_record_that_is_not_utf8_is_unreadable_not_absent(self, tmp_path):
        """Real on any user, and a different failure from malformed JSON: the
        bytes never became text, so a message calling it unparseable JSON sends
        a reader to edit something they could not decode."""
        from src.kg.artifacts import workspace_provenance_status
        from src.kg.provenance import PROVENANCE_FILENAME

        workspace = tmp_path / "ws"
        _write(workspace)  # graph-only: nothing declares the record
        (workspace / PROVENANCE_FILENAME).write_bytes(b"\xff\xfe\x00not utf-8")

        status = workspace_provenance_status(workspace)

        assert status.state == "unreadable"
        assert status.state != "absent"
        assert "could not be read" in status.detail

    @pytest.mark.parametrize(
        "payload", ["{ not json", "[]"], ids=["unparseable", "not-an-object"],
    )
    @pytest.mark.parametrize(
        "record", ["present", "deleted"], ids=["record-present", "record-gone"],
    )
    def test_a_manifest_that_cannot_be_read_declared_nothing_readable(
        self, tmp_path, payload, record
    ):
        """**`declared = None` is a claim, not a fallback.**

        It is the value that means *nothing ever declared a record*, and feeding
        it from a failed manifest read made a workspace whose record is gone
        report `absent` — whose detail says the workspace predates provenance,
        about one that may have declared a record and lost it. What is true is
        narrower and is what the status now says.
        """
        from src.kg.artifacts import MANIFEST_FILENAME, workspace_provenance_status
        from src.kg.provenance import PROVENANCE_FILENAME

        workspace = tmp_path / "ws"
        _write(workspace, samples=_budget())
        if record == "deleted":
            (workspace / PROVENANCE_FILENAME).unlink()
        (workspace / MANIFEST_FILENAME).write_text(payload)

        status = workspace_provenance_status(workspace)

        assert status.state == "unreadable"
        assert "predates provenance" not in status.detail
        assert MANIFEST_FILENAME in status.detail

    @pytest.mark.parametrize(
        "state", ["recorded", "missing", "absent", "mismatched"],
    )
    def test_a_readable_workspace_still_tells_the_four_states_apart(self, tmp_path, state):
        """**The control the boundary needs most.**

        An `except OSError` around a whole function is the cheapest way to
        satisfy every test above and report `unreadable` for everything. This
        says it does not: on a directory that reads normally, the four states a
        caller acts on are still distinguished.
        """
        from src.kg.artifacts import workspace_provenance_status
        from src.kg.provenance import PROVENANCE_FILENAME

        workspace = tmp_path / "ws"
        _write(workspace, samples=_budget())
        if state == "missing":
            (workspace / PROVENANCE_FILENAME).unlink()
        elif state == "absent":
            # No graph to describe: the earliest state, before any record.
            workspace = tmp_path / "empty"
            workspace.mkdir()
        elif state == "mismatched":
            record = json.loads((workspace / PROVENANCE_FILENAME).read_text())
            record["counters"]["rows_parsed"] = 999
            (workspace / PROVENANCE_FILENAME).write_text(json.dumps(record))

        assert workspace_provenance_status(workspace).state == state

    def test_the_low_level_reader_still_raises_its_own_error_type(self, tmp_path, monkeypatch):
        """`read_provenance` is not a status function — it raises by contract,
        and `ProvenanceError` is the type its callers catch. A `PermissionError`
        from its existence probe would go past every one of them, including the
        `except ProvenanceError` inside `provenance_status`."""
        from src.kg.provenance import (
            PROVENANCE_FILENAME, ProvenanceError, read_provenance,
        )

        workspace = tmp_path / "ws"
        _write(workspace)
        real_stat = Path.stat

        def guard(self, *args, **kwargs):
            if self.name == PROVENANCE_FILENAME:
                raise PermissionError(13, "Permission denied", str(self))
            return real_stat(self, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", guard)

        with pytest.raises(ProvenanceError, match="could not be looked up"):
            read_provenance(workspace)

    def test_a_sound_workspace_still_hands_back_its_sources(self, tmp_path):
        """The guards above hold for a status reader that reports `unreadable`
        for everything. This is the control that says they do not."""
        from src.kg.artifacts import workspace_provenance_status
        from src.kg.provenance import source_entry

        workspace = tmp_path / "ws"
        _write(
            workspace, samples=_budget(),
            sources=[source_entry("mondo", "mondo.obo", "a" * 64, "2025-01-01")],
            counters={"rows_parsed": 3},
        )

        status = workspace_provenance_status(workspace)

        assert status.state == "recorded" and status.is_known
        assert [e["role"] for e in status.record["sources"]] == ["mondo"]
        assert status.record["sources"][0]["declared_version"] == "2025-01-01"
        assert status.record["counters"] == {"rows_parsed": 3}


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


class TestThisModuleCollectsOnAPlatformWithoutPosixIds:
    """**A skip condition runs at import, before pytest can skip anything.**

    The first version of the permission guards called `os.geteuid()` in a
    `skipif` decorator. That attribute exists only on Unix; on Windows the call
    happens while the class is being defined, so the whole unit suite fails to
    collect with `AttributeError` — a test that meant to skip taking the run
    down with it. The predicates are measured now and touch only `os.chmod` and
    `open`, which exist everywhere.

    Checked by importing this module in a subprocess whose `os` has no
    `geteuid`, which is what the platform difference amounts to. Asserting on
    the source text instead would pass for any spelling that still called it.
    """

    def test_importing_it_without_os_geteuid_succeeds(self):
        import subprocess
        import sys

        root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [sys.executable, "-c",
             "import os, sys\n"
             "sys.path.insert(0, sys.argv[1])\n"
             "del os.geteuid\n"
             "assert not hasattr(os, 'geteuid')\n"
             "import importlib\n"
             "importlib.import_module('tests.unit.test_kg_provenance')\n"
             "print('collected')",
             str(root)],
            capture_output=True, text=True, cwd=str(root),
        )

        assert result.returncode == 0, (
            "this module cannot be imported where os.geteuid is absent, which "
            f"is every Windows checkout:\n{result.stderr[-2000:]}"
        )
        assert "collected" in result.stdout
