"""
The verifiers compare reader results with one manifest reading, and read nothing.
=================================================================================
Contract M2.1, S5. A run reads its manifest once (`read_split_manifest`) and each
input once (the S4 readers); the verifiers then compare those identities with
that one manifest reading. They open no file, so a file replaced between two
reads shows up as a mismatch, and the refusal names the file and the manifest.

The `kg.json` identification check (M2.1 decision 1) is the one exception that
touches the disk: it hashes `kg.json`, for a run that does not parse it, and
compares it with the same manifest reading, which it never reads itself.

The path forms (`verify_graph_artifacts`, `verify_graph_source`,
`verify_generated_cohorts`) remain for callers not yet moved, until S9.

Module: tests/unit/test_verifiers_from_reads.py
"""
from __future__ import annotations

import json
import os

import pytest

from src.kg.artifacts import (
    check_unparsed_kg_json,
    read_split_manifest,
    verify_graph_reads,
    verify_graph_source_read,
)
from src.utils.fingerprint import ReadIdentity
from tests.fixtures.generated_workspace import write_generated_workspace
from tests.fixtures.opens import count_opens
from tests.fixtures.replacement import replace_after_read

torch = pytest.importorskip("torch")

from src.evaluation.cohort import verify_cohort_reads, verify_generated_cohorts  # noqa: E402
from src.kg.graph import KnowledgeGraph  # noqa: E402
from src.kg.storage.file_storage import read_graph_artifacts, read_samples  # noqa: E402

GRAPH_ROLES = ("kg", "node_features", "edge_indices", "num_nodes")


@pytest.fixture
def workspace(tmp_path):
    root, _ = write_generated_workspace(tmp_path / "ws", train_ids=[0, 1, 2], val_ids=[3])
    return root


@pytest.fixture
def other(tmp_path):
    root, _ = write_generated_workspace(tmp_path / "other", train_ids=[5, 6], val_ids=[7])
    return root


def _opens_inside(opens, root) -> dict:
    """Opens of files in `root`; imports and logging elsewhere do not count."""
    prefix = os.path.realpath(root) + os.sep
    return {path: n for path, n in opens.counts.items() if path.startswith(prefix)}


def _samples(root, splits=("train", "val")):
    return {split: read_samples(root, split) for split in splits}


# ---------------------------------------------------------------------------
# verify_graph_reads
# ---------------------------------------------------------------------------
class TestVerifyGraphReads:
    def test_a_sound_run_returns_the_manifest_bound_map_and_opens_nothing(self, workspace):
        manifest = read_split_manifest(workspace)
        graph = read_graph_artifacts(workspace)

        with count_opens() as opens:
            bound = verify_graph_reads(manifest, graph.reads)

        assert _opens_inside(opens, workspace) == {}
        assert bound == {role: manifest.manifest["artifacts"][role] for role in GRAPH_ROLES}

    def test_a_file_from_another_workspace_is_refused_naming_it_and_the_manifest(
        self, workspace, other
    ):
        manifest = read_split_manifest(workspace)
        reads = dict(read_graph_artifacts(workspace).reads)
        reads["edge_indices"] = read_graph_artifacts(other).reads["edge_indices"]

        with pytest.raises(ValueError, match="is not the edge_indices artifact") as refused:
            verify_graph_reads(manifest, reads)
        assert str(other / "edge_indices.pt") in str(refused.value)
        assert str(workspace / "split_manifest.json") in str(refused.value)

    @pytest.mark.parametrize("role", ["node_features", "edge_indices", "num_nodes"])
    def test_a_tensor_role_the_run_did_not_read_is_refused(self, workspace, role):
        """An absent file has no identity; it is refused, not skipped."""
        manifest = read_split_manifest(workspace)
        reads = dict(read_graph_artifacts(workspace).reads)
        del reads[role]

        with pytest.raises(ValueError, match="was not read") as refused:
            verify_graph_reads(manifest, reads)
        assert role in str(refused.value)
        assert str(workspace / "split_manifest.json") in str(refused.value)

    def test_a_kg_identity_in_the_reads_is_compared_too(self, workspace, other):
        manifest = read_split_manifest(workspace)
        reads = dict(read_graph_artifacts(workspace).reads)
        reads["kg"] = KnowledgeGraph.read_json(other / "kg.json").identity

        with pytest.raises(ValueError, match="is not the kg artifact"):
            verify_graph_reads(manifest, reads)

    def test_a_manifest_missing_a_role_is_refused(self, workspace):
        manifest = read_split_manifest(workspace)
        del manifest.manifest["artifacts"]["num_nodes"]

        with pytest.raises(ValueError, match="records no digest for num_nodes"):
            verify_graph_reads(manifest, read_graph_artifacts(workspace).reads)

    def test_an_identity_for_a_role_the_manifest_does_not_bind_is_refused(self, workspace):
        manifest = read_split_manifest(workspace)
        reads = dict(read_graph_artifacts(workspace).reads)
        reads["shortest_paths"] = ReadIdentity(workspace / "shortest_paths.pt", "0" * 64)

        with pytest.raises(ValueError, match="shortest_paths"):
            verify_graph_reads(manifest, reads)


# ---------------------------------------------------------------------------
# A replacement between reads, and a republish (acceptance items 1 and 2)
# ---------------------------------------------------------------------------
class TestAReplacementBetweenReads:
    @pytest.mark.parametrize("how", ["rename", "rewrite"])
    def test_a_file_replaced_after_the_manifest_read_is_refused_by_name(
        self, monkeypatch, workspace, other, how
    ):
        target = workspace / "node_features.pt"
        record = replace_after_read(monkeypatch, after=workspace / "split_manifest.json",
                                    target=target,
                                    data=(other / "node_features.pt").read_bytes(), how=how)

        manifest = read_split_manifest(workspace)
        graph = read_graph_artifacts(workspace)

        assert record.fired == 1
        with pytest.raises(ValueError, match="is not the node_features artifact") as refused:
            verify_graph_reads(manifest, graph.reads)
        assert str(target) in str(refused.value)

    @pytest.mark.parametrize("how", ["rename", "rewrite"])
    def test_a_samples_file_replaced_after_the_manifest_read_is_refused_by_name(
        self, monkeypatch, workspace, other, how
    ):
        target = workspace / "val_samples.json"
        record = replace_after_read(monkeypatch, after=workspace / "split_manifest.json",
                                    target=target,
                                    data=(other / "val_samples.json").read_bytes(), how=how)

        manifest = read_split_manifest(workspace)
        samples = _samples(workspace)

        assert record.fired == 1
        with pytest.raises(ValueError, match="is not the file") as refused:
            verify_cohort_reads(manifest, samples)
        assert str(target) in str(refused.value)

    def test_a_republish_after_both_reads_leaves_the_run_on_what_it_read(
        self, monkeypatch, workspace, other
    ):
        target = workspace / "node_features.pt"
        original = read_graph_artifacts(workspace).reads["node_features"]
        record = replace_after_read(monkeypatch, after=target, target=target,
                                    data=(other / "node_features.pt").read_bytes(),
                                    how="rewrite", republish="node_features")

        manifest = read_split_manifest(workspace)
        graph = read_graph_artifacts(workspace)
        bound = verify_graph_reads(manifest, graph.reads)

        assert record.fired == 1
        assert graph.reads["node_features"] == original
        assert bound["node_features"] == original.sha256

    @pytest.mark.parametrize("first", ["manifest", "file"])
    def test_a_republish_between_the_two_reads_is_refused(
        self, monkeypatch, workspace, other, first
    ):
        target = workspace / "node_features.pt"
        manifest_path = workspace / "split_manifest.json"
        record = replace_after_read(monkeypatch,
                                    after=manifest_path if first == "manifest" else target,
                                    target=target,
                                    data=(other / "node_features.pt").read_bytes(),
                                    how="rename", republish="node_features")

        if first == "manifest":
            manifest = read_split_manifest(workspace)
            graph = read_graph_artifacts(workspace)
        else:
            graph = read_graph_artifacts(workspace)
            manifest = read_split_manifest(workspace)

        assert record.fired == 1
        with pytest.raises(ValueError, match="is not the node_features artifact"):
            verify_graph_reads(manifest, graph.reads)

    def test_a_republish_before_either_read_is_read_as_published(
        self, workspace, other
    ):
        from tests.fixtures.replacement import replace_file

        data = (other / "node_features.pt").read_bytes()
        replace_file(workspace / "node_features.pt", data, "rename")
        manifest_path = workspace / "split_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        import hashlib

        manifest["artifacts"]["node_features"] = hashlib.sha256(data).hexdigest()
        replace_file(manifest_path, json.dumps(manifest).encode("utf-8"), "rename")

        bound = verify_graph_reads(read_split_manifest(workspace),
                                   read_graph_artifacts(workspace).reads)

        assert bound["node_features"] == hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# verify_graph_source_read
# ---------------------------------------------------------------------------
class TestVerifyGraphSourceRead:
    def test_the_bound_graph_passes_and_opens_nothing(self, workspace):
        manifest = read_split_manifest(workspace)
        graph = KnowledgeGraph.read_json(workspace / "kg.json")
        reads = read_graph_artifacts(workspace).reads

        with count_opens() as opens:
            bound = verify_graph_source_read(manifest, graph.identity, reads)

        assert _opens_inside(opens, workspace) == {}
        assert bound["kg"] == graph.identity.sha256

    def test_another_workspaces_graph_is_refused_naming_it_and_the_manifest(
        self, workspace, other
    ):
        manifest = read_split_manifest(workspace)
        graph = KnowledgeGraph.read_json(other / "kg.json")

        with pytest.raises(ValueError, match="is not the graph") as refused:
            verify_graph_source_read(manifest, graph.identity,
                                     read_graph_artifacts(workspace).reads)
        assert str(other / "kg.json") in str(refused.value)
        assert str(workspace / "split_manifest.json") in str(refused.value)

    def test_the_bound_bytes_elsewhere_are_the_bound_graph(self, tmp_path, workspace):
        """Identity is the bytes, not the path (`verify_graph_source`'s rule, kept)."""
        copy = tmp_path / "mounted" / "kg.json"
        copy.parent.mkdir()
        copy.write_bytes((workspace / "kg.json").read_bytes())

        verify_graph_source_read(read_split_manifest(workspace),
                                 KnowledgeGraph.read_json(copy).identity,
                                 read_graph_artifacts(workspace).reads)


# ---------------------------------------------------------------------------
# check_unparsed_kg_json (decision 1)
# ---------------------------------------------------------------------------
class TestTheKgJsonIdentificationCheck:
    def test_a_matching_file_passes_records_nothing_and_never_reads_the_manifest(
        self, workspace
    ):
        manifest = read_split_manifest(workspace)

        with count_opens() as opens:
            result = check_unparsed_kg_json(workspace, manifest)

        assert result is None
        assert _opens_inside(opens, workspace) == {
            os.path.realpath(workspace / "kg.json"): 1
        }

    def test_another_graph_is_refused_naming_it_and_the_manifest(self, workspace, other):
        manifest = read_split_manifest(workspace)
        (workspace / "kg.json").write_bytes((other / "kg.json").read_bytes())

        with pytest.raises(ValueError, match="is not the kg artifact") as refused:
            check_unparsed_kg_json(workspace, manifest)
        assert str(workspace / "split_manifest.json") in str(refused.value)

    def test_an_absent_file_is_refused(self, workspace):
        manifest = read_split_manifest(workspace)
        (workspace / "kg.json").unlink()

        with pytest.raises(ValueError, match="kg.json is absent"):
            check_unparsed_kg_json(workspace, manifest)


# ---------------------------------------------------------------------------
# verify_cohort_reads
# ---------------------------------------------------------------------------
class TestVerifyCohortReads:
    def test_a_sound_workspace_gives_what_the_path_form_gives_and_opens_nothing(
        self, workspace
    ):
        manifest = read_split_manifest(workspace)
        samples = _samples(workspace)

        with count_opens() as opens:
            result = verify_cohort_reads(manifest, samples)

        assert _opens_inside(opens, workspace) == {}
        assert result == verify_generated_cohorts(workspace)
        assert result.verified == ("train", "val")
        assert result.disjointness_claim_checked and result.disjointness_measured

    def test_the_scope_is_the_splits_passed(self, workspace):
        result = verify_cohort_reads(read_split_manifest(workspace),
                                     _samples(workspace, ("train",)))

        assert result.verified == ("train",)
        assert set(result.disease_sets) == {"train"}
        assert not result.disjointness_claim_checked and not result.disjointness_measured

    def test_the_scope_is_reported_in_the_generators_order(self, workspace):
        samples = _samples(workspace)
        reversed_samples = {"val": samples["val"], "train": samples["train"]}

        assert verify_cohort_reads(read_split_manifest(workspace),
                                   reversed_samples).verified == ("train", "val")

    def test_training_fields_do_not_change_the_verdict(self, tmp_path):
        root, _ = write_generated_workspace(tmp_path / "ws", train_ids=[0, 1], val_ids=[2],
                                            training_fields=True)
        samples = {split: read_samples(root, split, training_fields=True)
                   for split in ("train", "val")}

        assert verify_cohort_reads(read_split_manifest(root), samples).verified == (
            "train", "val")

    @pytest.mark.parametrize("scope", [(), ("test",)])
    def test_a_scope_outside_the_generated_splits_is_refused(self, workspace, scope):
        manifest = read_split_manifest(workspace)
        samples = {split: read_samples(workspace, "train") for split in scope}

        with pytest.raises(ValueError, match="non-empty subset"):
            verify_cohort_reads(manifest, samples)

    def test_a_samples_file_from_another_workspace_is_refused_naming_both(
        self, workspace, other
    ):
        manifest = read_split_manifest(workspace)
        samples = _samples(workspace)
        samples["val"] = read_samples(other, "val")

        with pytest.raises(ValueError, match="is not the file") as refused:
            verify_cohort_reads(manifest, samples)
        assert str(other / "val_samples.json") in str(refused.value)
        assert str(workspace / "split_manifest.json") in str(refused.value)

    def test_a_disease_set_the_manifest_does_not_record_is_refused(self, workspace):
        manifest = read_split_manifest(workspace)
        manifest.manifest["realised"]["train_digest"] = "0" * 64

        with pytest.raises(ValueError, match="records as realised") as refused:
            verify_cohort_reads(manifest, _samples(workspace))
        assert str(workspace / "train_samples.json") in str(refused.value)

    def test_realised_contradicting_allocated_is_refused(self, workspace):
        manifest = read_split_manifest(workspace)
        realised = manifest.manifest["realised"]["val_digest"]
        manifest.manifest["allocation"]["allocated"]["val_digest"] = "1" * 64
        assert realised != "1" * 64

        with pytest.raises(ValueError, match="contradicts itself"):
            verify_cohort_reads(manifest, _samples(workspace))

    def test_a_manifest_that_does_not_claim_disjointness_is_refused(self, workspace):
        manifest = read_split_manifest(workspace)
        manifest.manifest["disjoint"] = False

        with pytest.raises(ValueError, match="claims disjoint=False"):
            verify_cohort_reads(manifest, _samples(workspace))

    def test_overlapping_cohorts_under_a_consistent_forged_manifest_are_refused(
        self, workspace
    ):
        """Every other check passes; only the measured comparison is left."""
        import hashlib

        from src.kg.disease_allocation import disease_set_digest

        rows = json.loads((workspace / "val_samples.json").read_text())
        rows[0]["disease_id"] = 0
        (workspace / "val_samples.json").write_text(json.dumps(rows))
        manifest = read_split_manifest(workspace)
        digest = disease_set_digest(sorted({row["disease_id"] for row in rows}))
        manifest.manifest["realised"]["val_digest"] = digest
        manifest.manifest["allocation"]["allocated"]["val_digest"] = digest
        manifest.manifest["artifacts"]["val_samples"] = hashlib.sha256(
            (workspace / "val_samples.json").read_bytes()).hexdigest()

        with pytest.raises(ValueError, match="does not hold disease-disjoint cohorts") as refused:
            verify_cohort_reads(manifest, _samples(workspace))
        assert str(workspace / "split_manifest.json") in str(refused.value)
